"""Box packs that stay in device memory between segments.

A ``DevParent`` holds the survivors of one segment on the device in a compact,
lossless encoding: lower and upper bounds as float32 and one unsigned "meta" word
per box that packs the origin (leaf of the start set) and the action-history ring.
A ``DevPiece`` is a row range of a parent. Splitting a pack into pieces creates
views only; the parent's memory is freed when its last piece is gone.

``assemble`` decodes and concatenates pieces into the zero-padded, cap-sized
inputs of the segment kernel, on the device.
"""
from __future__ import annotations

import math
import threading

import jax
import jax.numpy as jnp
import numpy as np

_LOCK = threading.Lock()
_ASM_CACHE: dict = {}
_EMIT_CACHE: dict = {}
_CACHE_MAX = 384


def _bits(nvals: int) -> int:
    return max(1, int(math.ceil(math.log2(max(int(nvals), 2)))))


def meta_layout(n_origin: int, ncol: int, n_actions: int):
    """(origin bits, action bits, total bits, word dtype or None if > 64 bits)."""
    bo = _bits(n_origin)
    ba = _bits(n_actions)
    tot = bo + ncol * ba
    for limit, dt in ((8, np.uint8), (16, np.uint16), (32, np.uint32), (64, np.uint64)):
        if tot <= limit:
            return bo, ba, tot, dt
    return bo, ba, tot, None


def meta_nbytes(K: int, lookback: int, n_actions: int) -> int:
    """Bytes of the meta word (5 if it would need more than 64 bits)."""
    _, _, _, dt = meta_layout(K, max(lookback, 1), n_actions)
    return 5 if dt is None else np.dtype(dt).itemsize


class DevParent:
    """Encoded boxes ``arrs = {lo, hi, meta}`` of ``n`` rows on one device."""
    __slots__ = ("arrs", "mhdr", "n", "nbytes", "pid")
    _next = [0]

    def __init__(self, arrs, mhdr, n):
        self.arrs = arrs
        self.mhdr = mhdr
        self.n = int(n)
        self.nbytes = int(sum(int(np.prod(a.shape)) * a.dtype.itemsize for a in arrs.values()))
        with _LOCK:
            DevParent._next[0] += 1
            self.pid = DevParent._next[0]


class DevPiece:
    """Rows [a, b) of a DevParent."""
    __slots__ = ("parent", "a", "b")

    def __init__(self, parent, a, b):
        self.parent = parent
        self.a = int(a)
        self.b = int(b)

    @property
    def n(self):
        return self.b - self.a

    def __getitem__(self, k):
        if k == "n":
            return self.n
        raise KeyError(k)

    def nbytes(self):
        """Share of the parent's bytes, proportional to the rows."""
        p = self.parent
        return int(p.nbytes * self.n / max(p.n, 1))

    def slice(self, a, b):
        b = min(int(b), self.n)
        return DevPiece(self.parent, self.a + int(a), self.a + b)


def is_dev(p) -> bool:
    return isinstance(p, DevPiece)


def parent_device(par):
    """The device that holds a parent's buffers."""
    return next(iter(next(iter(par.arrs.values())).devices()))


def _unpack_meta(m, bo, ba, ncol):
    dt = m.dtype
    o = (m & jnp.asarray((1 << bo) - 1, dt)).astype(jnp.int32)
    cols = [((m >> jnp.asarray(bo + j * ba, dt)) & jnp.asarray((1 << ba) - 1, dt)).astype(jnp.uint8)
            for j in range(ncol)]
    return o, jnp.stack(cols, axis=1)


def _pack_meta(o, ah, bo, ba, ncol, dt):
    m = o.astype(dt)
    for j in range(ncol):
        m = m | (ah[:, j].astype(dt) << jnp.asarray(bo + j * ba, dt))
    return m


def _emit_fn(n, bo, ba, ncol, mdt):
    key = (n, "compact", bo, ba, ncol, str(mdt), True, None)
    f = _EMIT_CACHE.get(key)
    if f is not None:
        return f

    def impl(s_lo, s_hi, s_orig, s_acth):
        lo, hi, o = s_lo[:n], s_hi[:n], s_orig[:n]
        ah = s_acth[:n]
        omax = jnp.max(o) if n else jnp.int32(0)
        amax = jnp.max(ah).astype(jnp.int32) if n else jnp.int32(0)
        omin = jnp.min(o) if n else jnp.int32(0)
        m = _pack_meta(o, ah, bo, ba, ncol, mdt)
        return {"lo": lo, "meta": m, "omax": omax, "omin": omin, "amax": amax, "hi": hi}

    with _LOCK:
        if len(_EMIT_CACHE) > _CACHE_MAX:
            _EMIT_CACHE.clear()
        f = _EMIT_CACHE[key] = jax.jit(impl)
    return f


def emit_compact(s_lo, s_hi, s_orig, s_acth, n, *, K, n_actions, ncol):
    """First ``n`` rows of the kernel's output buffers -> DevPiece (lossless).

    Refuses to encode if an origin or action does not fit its bit field.
    """
    n = int(n)
    bo, ba, _tot, mdt = meta_layout(K, ncol, n_actions)
    if mdt is None:
        raise ValueError("meta word does not fit 64 bits")
    mdt = np.dtype(mdt)
    out = _emit_fn(n, bo, ba, ncol, mdt)(s_lo, s_hi, s_orig, s_acth)
    if n:
        omax, omin, amax = int(out["omax"]), int(out["omin"]), int(out["amax"])
        if omin < 0 or omax >= (1 << bo) or amax >= (1 << ba):
            raise RuntimeError(
                f"meta overflow (origin {omin}..{omax} vs {bo} bits, action max {amax} "
                f"vs {ba} bits)")
    mhdr = {"bo": bo, "ba": ba, "ncol": ncol, "mdt": mdt.str}
    arrs = {"lo": out["lo"], "hi": out["hi"], "meta": out["meta"]}
    return DevPiece(DevParent(arrs, mhdr, n), 0, n)


def _asm_fn(specs, cap, d, ncol, fdtype, acth_dtype):
    key = (specs, cap, d, ncol, str(fdtype), str(acth_dtype))
    f = _ASM_CACHE.get(key)
    if f is not None:
        return f

    def impl(arr_lists):
        los, his, ors, acs = [], [], [], []
        M = 0
        for (enc, a, b, bo, ba, nc, wexp), arrs in zip(specs, arr_lists):
            if b <= a:
                continue
            M += b - a
            lo, hi, m = (y[a:b] for y in arrs)
            o, ah = _unpack_meta(m, bo, ba, nc)
            los.append(lo.astype(fdtype))
            his.append(hi.astype(fdtype))
            ors.append(o.astype(jnp.int32))
            acs.append(ah.astype(acth_dtype))
        pad = cap - M
        los.append(jnp.zeros((pad, d), fdtype))
        his.append(jnp.zeros((pad, d), fdtype))
        ors.append(jnp.zeros((pad,), jnp.int32))
        acs.append(jnp.zeros((pad, ncol), acth_dtype))
        return (jnp.concatenate(los), jnp.concatenate(his),
                jnp.concatenate(ors), jnp.concatenate(acs))

    with _LOCK:
        if len(_ASM_CACHE) > _CACHE_MAX:
            _ASM_CACHE.clear()
        f = _ASM_CACHE[key] = jax.jit(impl)
    return f


def assemble(pieces, *, cap, d, ncol, fdtype, acth_dtype):
    """DevPieces -> cap-row kernel inputs (lo, hi, origin, act_hist) on the
    device; rows [M:cap) are zero."""
    specs, arr_lists = [], []
    for p in pieces:
        par = p.parent
        mh = par.mhdr
        specs.append(("compact", p.a, p.b, int(mh["bo"]), int(mh["ba"]), int(mh["ncol"]), None))
        arr_lists.append((par.arrs["lo"], par.arrs["hi"], par.arrs["meta"]))
    return _asm_fn(tuple(specs), int(cap), int(d), int(ncol), fdtype, acth_dtype)(tuple(arr_lists))
