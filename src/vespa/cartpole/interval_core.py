"""Segment kernel of the CartPole verifier: exact box reachability on one device.

The reachable set of a decision tree policy is a union of boxes. One step maps
every box through the interval extension of the dynamics under the action of the
leaf that contains it, cuts the image along the leaf partition of the tree, and
reduces the resulting boxes without losing a single state:

  1. drop exact duplicates (bit-identical bounds in the same leaf), first in
     hash order, then in (leaf, Morton code) order;
  2. drop every box that is contained in another box of the same leaf
     (banded sweep over the Morton order plus a pass against the largest boxes).

Both reductions remove only boxes whose states are covered by a kept box, so the
union is unchanged and the verification stays exact up to the outward rounding
of the interval step.

``build_segment_kernel`` compiles this step loop once for a fixed buffer size
(the kernel batch cap). A call runs a pack of boxes forward step by step until
the horizon, a violation of the safety box, an empty set, or until the next step
would no longer fit the cap. The box pool (``segment_pool``) splits packs that
hit the cap and dispatches them again. At the first violating step the kernel
captures every violating box with its action history: these are the inputs of
the counterexample regions.
"""

import jax

# Interval bounds need float64 constants: enable x64 before any JAX computation.
jax.config.update("jax_enable_x64", True)

import threading  # noqa: E402
import time  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vespa.cartpole import device_pool  # noqa: E402
from vespa.cartpole.env_specs import EPS_ULP, N_ACTIONS, STATE_DIM, interval_step  # noqa: E402

# Storage precision: boxes are float32, the interval step adds a pad of
# EPS_ULP[axis] float32 unit roundoffs per step (outward rounding).
FDTYPE = jnp.float32
_U_FP32 = 2.0 ** -24
_EPS = np.asarray(EPS_ULP, dtype=np.float64) * _U_FP32
ACTIONS_DTYPE = jnp.uint8      # per-leaf action index
ACTH_DTYPE = jnp.uint8         # action-history ring entries

# Kernel constants of the paper runs.
SOURCE_CHUNK = 262144          # source rows per (source x leaf) intersection chunk
SP_BAND = 2048                 # subset prune: band of Morton neighbours checked
SP_LINEAR = 64                 # ... of which offsets 1..64 linearly, then powers of two
SP_HEAD = 256                  # ... plus the 256 largest boxes of each leaf
SP_UNROLL = 16                 # scan unroll of the subset prune passes
MORTON_BITS = 16               # Morton bits per axis
VELOCITY_AXIS_RANGE = 3.0      # Morton range of the unbounded velocity axes
REDUCE_TIERS = 4               # reductions run on the smallest of 4 prefix sizes
UNSAFE_CAPTURE_CAP = 1_048_576  # violating boxes captured per segment

# Serialises staging and slicing on the device (worker thread vs. pool thread).
DISPATCH_LOCK = threading.Lock()
# Serialises tracing and compilation of kernels.
_TRACE_LOCK = threading.Lock()

# Per-step reduction counters reported in ``reduce_stats``.
REDUCE_STATS_COLUMNS = (
    "n_pre",            # boxes entering the reduction (after the leaf cut)
    "qd_rm",            # removed by the hash-ordered duplicate drop
    "qd_nonbit",        # ... not bit-identical to their kept box (always 0)
    "qd_acth_diff",     # ... with a different action history than the kept box
    "qd_hash_collide",  # hash-adjacent rows with different keys
    "mzo_rm",           # removed by the Morton-ordered duplicate drop
    "mzo_nonbit",
    "mzo_acth_diff",
    "sp_rm",            # removed by the subset prune
    "n_post",           # boxes after the reduction
)

# Device memory model of one kernel: fixed + per_box * cap bytes (linear upper
# envelope measured on an H200, fp32, history ring of 2 columns), plus 3 bytes per
# box for every further ring column.
_KERNEL_MEM_FIXED = 0.618 * 2 ** 30
_KERNEL_MEM_PER_BOX = 165.07
_KERNEL_MEM_PER_RING_COL = 3.0


def kernel_memory_bytes(cap, lookback):
    """Predicted peak device memory (bytes) of one kernel at ``cap`` boxes."""
    per_box = _KERNEL_MEM_PER_BOX + _KERNEL_MEM_PER_RING_COL * max(0, max(int(lookback), 1) - 2)
    return float(_KERNEL_MEM_FIXED) + float(per_box) * int(cap)


# --------------------------------------------------------------------------- #
# Safety check and propagation                                                #
# --------------------------------------------------------------------------- #

def _make_hits_unsafe(safe_lo, safe_hi):
    """Per-box test 'box leaves the safety box' for the finite bounds of the box."""
    def hits_unsafe(lo, hi):
        out = None
        for ax in range(len(safe_lo)):
            if np.isfinite(safe_hi[ax]):
                t = hi[..., ax] > float(safe_hi[ax])
                out = t if out is None else (out | t)
            if np.isfinite(safe_lo[ax]):
                t = lo[..., ax] < float(safe_lo[ax])
                out = t if out is None else (out | t)
        return out
    return hits_unsafe


def _propagate_all(boxes_lo, boxes_hi, box_leaf, n_active, actions):
    """One interval step of every box under its leaf's action.

    Rows ``>= n_active`` become the empty box (lo = +inf, hi = -inf), so they never
    intersect a leaf and never count as unsafe.
    """
    cap = boxes_lo.shape[0]
    actions_per = actions[box_leaf]
    prop_lo, prop_hi = jax.vmap(interval_step, in_axes=(0, 0, 0, None))(
        boxes_lo, boxes_hi, actions_per, _EPS)
    valid_src = jnp.arange(cap, dtype=jnp.int64) < n_active
    prop_lo = jnp.where(valid_src[:, None], prop_lo, jnp.inf)
    prop_hi = jnp.where(valid_src[:, None], prop_hi, -jnp.inf)
    return prop_lo, prop_hi, valid_src


# --------------------------------------------------------------------------- #
# Hash and Morton keys                                                        #
# --------------------------------------------------------------------------- #

_HASH_MIX_LEAF = jnp.uint64(0x9E3779B97F4A7C15)
_HASH_MIX_LO = (jnp.uint64(0xBF58476D1CE4E5B9), jnp.uint64(0x94D049BB133111EB),
                jnp.uint64(0xD6E8FEB86659FD93), jnp.uint64(0xCA2C7DA0E2D62B33))
_HASH_MIX_HI = (jnp.uint64(0xC2B2AE3D27D4EB4F), jnp.uint64(0x165667B19E3779F9),
                jnp.uint64(0x85EBCA77C2B2AE63), jnp.uint64(0xC3A5C85C97CB3127))


def _hash64(klo, khi, leaf):
    """64-bit hash of (leaf, lo keys, hi keys) with a Murmur3 finaliser."""
    h = leaf.astype(jnp.uint64) * _HASH_MIX_LEAF
    for i in range(int(klo.shape[1])):
        h = h ^ (klo[:, i].astype(jnp.uint64) * _HASH_MIX_LO[i])
        h = h ^ (khi[:, i].astype(jnp.uint64) * _HASH_MIX_HI[i])
    h = h ^ (h >> jnp.uint64(33))
    h = h * jnp.uint64(0xFF51AFD7ED558CCD)
    h = h ^ (h >> jnp.uint64(33))
    return h


def _bit_keys(lo, hi):
    """int64 images of the float32 bit patterns: equal keys iff bit-identical bounds."""
    def key(x):
        return jax.lax.bitcast_convert_type(x, jnp.int32).astype(jnp.int64)
    return key(lo), key(hi)


def _morton_cascade(d, bits):
    """(shift, mask) rounds that spread ``bits`` bits with d-1 zeros in between."""
    rounds = []
    if bits > 1:
        k = 1
        while k * 2 < bits:
            k *= 2
        while k >= 1:
            shift = k * (d - 1)
            period = k * d
            grp = (1 << k) - 1
            m = 0
            j = 0
            while j * period < 64:
                m |= grp << (j * period)
                j += 1
            m &= (1 << 64) - 1
            rounds.append((shift, m))
            k //= 2
    return rounds


def _make_morton(d, bits):
    """Bit interleave of d coordinates of ``bits`` bits into a 64-bit Z-order code.

    The constants are numpy scalars so that a closure built during a trace holds
    no tracers.
    """
    rounds = [(np.uint64(s), np.uint64(m)) for s, m in _morton_cascade(d, bits)]
    init_mask = np.uint64((1 << bits) - 1)
    shifts = [np.uint64(i) for i in range(d)]

    def expand(v):
        v = v & init_mask
        for s, m in rounds:
            v = (v | (v << s)) & m
        return v

    def morton(cols):
        out = expand(cols[0])
        for i in range(1, d):
            out = out | (expand(cols[i]) << shifts[i])
        return out

    return morton


_MORTON = _make_morton(STATE_DIM, 64 // STATE_DIM)


def _box_morton_codes(lo, hi, axis_lo, axis_hi, morton_bits):
    """Morton code (int64) of each box centre on a 2^morton_bits grid per axis."""
    centers = ((lo + hi) * 0.5).astype(jnp.float64)
    span = (axis_hi - axis_lo).astype(jnp.float64)
    span = jnp.where(span <= 0.0, jnp.float64(1.0), span)
    bins = jnp.uint64(1) << jnp.uint64(morton_bits)
    norm = (centers - axis_lo.astype(jnp.float64)) / span
    norm = jnp.clip(norm, 0.0, 1.0 - 1e-12)
    bin_idx = (norm * bins.astype(jnp.float64)).astype(jnp.uint64)
    bin_idx = jnp.minimum(bin_idx, bins - jnp.uint64(1))
    return _MORTON([bin_idx[:, i] for i in range(STATE_DIM)]).astype(jnp.int64)


def _morton_max_bits(d, morton_bits):
    """Bits reserved for the Morton term in the packed (leaf, Morton) sort key."""
    return min(int(morton_bits) * int(d), 56)


def _float_bits_u64(col):
    """Bit pattern of a float32 column widened to uint64."""
    return jax.lax.bitcast_convert_type(col, jnp.uint32).astype(jnp.uint64)


def _float_sort_key_u64(col):
    """uint64 image of a float32 column whose integer order is the float order."""
    u = _float_bits_u64(col)
    top = jnp.uint64(1) << jnp.uint64(31)
    neg = (u & top) != jnp.uint64(0)
    return jnp.where(neg, (~u) & jnp.uint64(0xFFFFFFFF), u | top)


# --------------------------------------------------------------------------- #
# Leaf cut                                                                    #
# --------------------------------------------------------------------------- #

def _intersect_and_compact(prop_lo, prop_hi, valid_src, box_origin, leaf_lo, leaf_hi,
                           leaf_template, K, cap, chunk, act_hist, n_src):
    """Cut every propagated box along the leaf partition and compact the pieces.

    Output row (source i, leaf j) is ``prop_i ∩ leaf_j`` for every non-empty
    intersection, written to the front of the buffers; it inherits the source's
    origin and action history. Only the live prefix ``[0, n_src)`` is scanned.
    Returns (lo, hi, leaf, origin, count, overflow, act_hist); ``overflow`` is set
    if more than ``cap`` pieces were produced (then pieces were dropped).
    """
    fdtype = prop_lo.dtype
    d = int(prop_lo.shape[1])
    Sc = chunk
    num_src_chunks = (cap + Sc - 1) // Sc
    n_col = int(act_hist.shape[1])

    next_lo0 = jnp.zeros((cap, d), dtype=fdtype)
    next_hi0 = jnp.zeros((cap, d), dtype=fdtype)
    # uint8/uint16 leaf ids and origins keep their dtype, wider ones become int64/int32.
    narrow = (jnp.dtype(jnp.uint8), jnp.dtype(jnp.uint16))
    leaf_dt = leaf_template.dtype if leaf_template.dtype in narrow else jnp.int64
    orig_dt = box_origin.dtype if box_origin.dtype in narrow else jnp.int32
    next_leaf0 = jnp.zeros((cap,), dtype=leaf_dt)
    next_origin0 = jnp.zeros((cap,), dtype=orig_dt)
    next_acth0 = jnp.zeros((cap, n_col), dtype=act_hist.dtype)

    def chunk_body(carry, src_chunk_idx):
        next_lo, next_hi, next_leaf, next_origin, next_acth, running_count = carry
        start = src_chunk_idx * Sc
        # dynamic_slice clamps a start whose slice would run past cap; rows of such
        # a last chunk that the previous chunk already consumed are masked out.
        start_c = jnp.minimum(start, jnp.int64(cap - Sc))
        src_lo = jax.lax.dynamic_slice_in_dim(prop_lo, start_c, Sc, axis=0)
        src_hi = jax.lax.dynamic_slice_in_dim(prop_hi, start_c, Sc, axis=0)
        src_v = jax.lax.dynamic_slice_in_dim(valid_src, start_c, Sc, axis=0)
        src_orig = jax.lax.dynamic_slice_in_dim(box_origin, start_c, Sc, axis=0)
        src_idx = start_c + jnp.arange(Sc, dtype=jnp.int64)
        src_v = src_v & (src_idx >= start)

        inter_lo = jnp.maximum(src_lo[:, None, :], leaf_lo[None, :, :])
        inter_hi = jnp.minimum(src_hi[:, None, :], leaf_hi[None, :, :])
        non_empty = jnp.all(inter_lo <= inter_hi, axis=-1) & src_v[:, None]

        flat_lo = inter_lo.reshape(-1, d)
        flat_hi = inter_hi.reshape(-1, d)
        flat_keep = non_empty.reshape(-1)
        keep_int = flat_keep.astype(jnp.int64)
        flat_origin = jnp.broadcast_to(src_orig[:, None], (Sc, K)).reshape(-1)

        local_pos = jnp.cumsum(keep_int) - keep_int
        write_pos = running_count + local_pos
        in_range = flat_keep & (write_pos < jnp.int64(cap))
        safe_write = jnp.where(in_range, write_pos, jnp.int64(cap))

        next_lo = next_lo.at[safe_write].set(flat_lo, mode="drop")
        next_hi = next_hi.at[safe_write].set(flat_hi, mode="drop")
        next_leaf = next_leaf.at[safe_write].set(leaf_template, mode="drop")
        next_origin = next_origin.at[safe_write].set(flat_origin, mode="drop")
        src_acth = jax.lax.dynamic_slice_in_dim(act_hist, start_c, Sc, axis=0)
        flat_acth = jnp.broadcast_to(src_acth[:, None, :], (Sc, K, n_col)).reshape(-1, n_col)
        next_acth = next_acth.at[safe_write].set(flat_acth, mode="drop")
        running_count = running_count + jnp.sum(keep_int, dtype=jnp.int64)
        return (next_lo, next_hi, next_leaf, next_origin, next_acth, running_count), None

    init = (next_lo0, next_hi0, next_leaf0, next_origin0, next_acth0, jnp.int64(0))
    n_chunks = jnp.minimum((jnp.asarray(n_src, jnp.int64) + Sc - 1) // Sc,
                           jnp.int64(num_src_chunks))
    (next_lo, next_hi, next_leaf, next_origin, next_acth, running_count) = jax.lax.fori_loop(
        jnp.int64(0), n_chunks, lambda i, c: chunk_body(c, i)[0], init)
    overflow = running_count > jnp.int64(cap)
    next_count = jnp.minimum(running_count, jnp.int64(cap))
    return next_lo, next_hi, next_leaf, next_origin, next_count, overflow, next_acth


def _count_pieces(prop_lo, prop_hi, valid_src, leaf_lo, leaf_hi, cap, chunk, n_src):
    """Number of pieces ``_intersect_and_compact`` would produce, without writing them."""
    Sc = chunk
    num_src_chunks = (cap + Sc - 1) // Sc

    def chunk_body(acc, src_chunk_idx):
        start = src_chunk_idx * Sc
        start_c = jnp.minimum(start, jnp.int64(cap - Sc))
        src_lo = jax.lax.dynamic_slice_in_dim(prop_lo, start_c, Sc, axis=0)
        src_hi = jax.lax.dynamic_slice_in_dim(prop_hi, start_c, Sc, axis=0)
        src_v = jax.lax.dynamic_slice_in_dim(valid_src, start_c, Sc, axis=0)
        src_idx = start_c + jnp.arange(Sc, dtype=jnp.int64)
        src_v = src_v & (src_idx >= start)
        inter_lo = jnp.maximum(src_lo[:, None, :], leaf_lo[None, :, :])
        inter_hi = jnp.minimum(src_hi[:, None, :], leaf_hi[None, :, :])
        non_empty = jnp.all(inter_lo <= inter_hi, axis=-1) & src_v[:, None]
        return acc + jnp.sum(non_empty.astype(jnp.int64)), None

    n_chunks = jnp.minimum((jnp.asarray(n_src, jnp.int64) + Sc - 1) // Sc,
                           jnp.int64(num_src_chunks))
    return jax.lax.fori_loop(jnp.int64(0), n_chunks,
                             lambda i, acc: chunk_body(acc, i)[0], jnp.int64(0))


def leaf_arrays(leaf_regions, safe_lo, safe_hi):
    """(K, actions, leaf_lo, leaf_hi): leaf boxes clipped to the safety box."""
    K = len(leaf_regions)
    np_dtype = np.dtype(FDTYPE)
    raw_lo = np.array([[r["bounds"][i][0] for i in range(STATE_DIM)] for r in leaf_regions],
                      dtype=np_dtype)
    raw_hi = np.array([[r["bounds"][i][1] for i in range(STATE_DIM)] for r in leaf_regions],
                      dtype=np_dtype)
    actions = jnp.array([r["action"] for r in leaf_regions], dtype=ACTIONS_DTYPE)
    leaf_lo = jnp.array(np.clip(raw_lo, np.asarray(safe_lo, dtype=np_dtype), None), dtype=FDTYPE)
    leaf_hi = jnp.array(np.clip(raw_hi, None, np.asarray(safe_hi, dtype=np_dtype)), dtype=FDTYPE)
    return K, actions, leaf_lo, leaf_hi


def make_init_boxes(leaf_regions, spec, lookback):
    """The start set cut along the leaf partition: one box per leaf that meets it.

    Returns a host pack {lo, hi (M, 4) float32, origin (M,) int32 = leaf id,
    act_hist (M, lookback) uint8 zeros}. The origin of a box is the leaf of the
    start set it descends from.
    """
    _K, _actions, leaf_lo, leaf_hi = leaf_arrays(leaf_regions, spec.safe_lo, spec.safe_hi)
    init_lo = jnp.array(spec.init_lo, dtype=FDTYPE)
    init_hi = jnp.array(spec.init_hi, dtype=FDTYPE)
    state_lo = jnp.maximum(leaf_lo, init_lo)
    state_hi = jnp.minimum(leaf_hi, init_hi)
    active = np.asarray(jnp.all(state_lo <= state_hi, axis=-1))
    idx = np.flatnonzero(active).astype(np.int32)
    return {"lo": np.asarray(state_lo)[idx].astype(np.float32),
            "hi": np.asarray(state_hi)[idx].astype(np.float32),
            "origin": idx.astype(np.int32),
            "act_hist": np.zeros((idx.size, max(int(lookback), 1)), dtype=np.uint8)}


# --------------------------------------------------------------------------- #
# Lossless reduction on a prefix of the buffers                               #
# --------------------------------------------------------------------------- #

def _reduce_tier_sizes(cap, n_tiers):
    """Prefix sizes [cap/2^(n-1), ..., cap/2, cap], rounded up to multiples of 1024."""
    tiers = set()
    for i in range(max(1, int(n_tiers))):
        t = -(-int(cap) // (1 << i))
        t = min(int(cap), ((t + 1023) // 1024) * 1024)
        tiers.add(int(t))
    tiers.add(int(cap))
    return sorted(tiers)


def _tier_index(n_live, tiers):
    """Index of the smallest tier that holds ``n_live`` rows (traced)."""
    idx = jnp.int32(0)
    for t in tiers[:-1]:
        idx = idx + (jnp.asarray(n_live, jnp.int64) > jnp.int64(t)).astype(jnp.int32)
    return idx


def _pad_rows(x, cap):
    """Zero-pad the leading axis of ``x`` to ``cap`` rows."""
    T = int(x.shape[0])
    if T == cap:
        return x
    cfg = [(0, cap - T, 0)] + [(0, 0, 0)] * (x.ndim - 1)
    return jax.lax.pad(x, jnp.zeros((), dtype=x.dtype), cfg)


def _small_uint_dtype(max_value):
    """Narrowest unsigned dtype that holds 0..max_value."""
    m = int(max_value)
    if m <= 0xFF:
        return jnp.uint8
    if m <= 0xFFFF:
        return jnp.uint16
    return jnp.uint32


def _iota_idx(n):
    """Row index vector (int32 for any cap below 2^31)."""
    return jax.lax.iota(jnp.int32 if int(n) < 2 ** 31 else jnp.int64, int(n))


def _take0(x, idx):
    """Row gather ``x[idx]`` with in-range unsigned indices."""
    return x.at[idx].get(mode="promise_in_bounds")


def _sort_perm_u64(key_u64):
    """Stable argsort of a uint64 key as a uint32 permutation."""
    n = int(key_u64.shape[0])
    _, perm = jax.lax.sort((key_u64, jax.lax.iota(jnp.uint32, n)), num_keys=1, is_stable=True)
    return perm


def _sort_perm_i64(key_i64):
    """Stable argsort of an int64 key through its order-preserving uint64 image."""
    u = jax.lax.bitcast_convert_type(key_i64, jnp.uint64) ^ jnp.uint64(1 << 63)
    return _sort_perm_u64(u)


def _keep_index(keep_mask):
    """(indices of the True rows in ascending order at the front, their count)."""
    n = int(keep_mask.shape[0])
    km = keep_mask.astype(jnp.uint32)
    pos = jnp.cumsum(km, dtype=jnp.uint32) - km
    dest = jnp.where(keep_mask, pos, jnp.uint32(n))
    keep_idx = jnp.zeros((n,), jnp.uint32).at[dest].set(jax.lax.iota(jnp.uint32, n), mode="drop")
    return keep_idx, jnp.sum(keep_mask, dtype=jnp.int64)


def _gather_fill(x, src, valid_out):
    """``x[src]`` on the valid output rows, zero elsewhere."""
    g = _take0(x, src)
    z = jnp.zeros((), dtype=x.dtype)
    if x.ndim == 2:
        return jnp.where(valid_out[:, None], g, z)
    return jnp.where(valid_out, g, z)


def _all_axes(pairs):
    """AND of per-axis boolean columns."""
    out = None
    for p in pairs:
        out = p if out is None else (out & p)
    return out


def _compact(keep_mask, cap, perm, arrays):
    """Keep the rows of ``keep_mask`` (read through ``perm``) at the front."""
    keep_idx, n_keep = _keep_index(keep_mask)
    valid_out = _iota_idx(cap) < n_keep
    src = keep_idx if perm is None else _take0(perm, keep_idx)
    return [_gather_fill(x, src, valid_out) for x in arrays], n_keep


def _drop_stats(lo, hi, act_hist, perm, same, valid_s):
    """Counters of one duplicate drop: (rows not bit-identical to their kept row,
    rows with a different action history, max protrusion per axis)."""
    d = int(lo.shape[1])
    n = int(perm.shape[0])
    idx = _iota_idx(n)
    dropped = valid_s & same
    surv_pos = jax.lax.cummax(jnp.where(same, jnp.zeros_like(idx), idx), axis=0)
    surv = _take0(perm, surv_pos.astype(jnp.uint32))
    lo_r, hi_r = _take0(lo, perm), _take0(hi, perm)
    lo_v, hi_v = _take0(lo, surv), _take0(hi, surv)
    diff = None
    for a in range(d):
        da = ((_float_bits_u64(lo_r[:, a]) != _float_bits_u64(lo_v[:, a]))
              | (_float_bits_u64(hi_r[:, a]) != _float_bits_u64(hi_v[:, a])))
        diff = da if diff is None else (diff | da)
    n_nonbit = jnp.sum(dropped & diff, dtype=jnp.int64)
    out = jnp.maximum(lo_v.astype(jnp.float64) - lo_r.astype(jnp.float64),
                      hi_r.astype(jnp.float64) - hi_v.astype(jnp.float64))
    prot = jnp.max(jnp.where(dropped[:, None], out, 0.0), axis=0)
    prot = jnp.maximum(prot, 0.0)
    a_r, a_v = _take0(act_hist, perm), _take0(act_hist, surv)
    adiff = None
    for j in range(int(act_hist.shape[1])):
        dj = a_r[:, j] != a_v[:, j]
        adiff = dj if adiff is None else (adiff | dj)
    n_acth = jnp.sum(dropped & adiff, dtype=jnp.int64)
    return n_nonbit, n_acth, prot


def _adjacent_duplicates(lo, hi, leaf, perm):
    """``same[i]``: sorted row i has the same leaf and bit-identical bounds as row i-1."""
    d = int(lo.shape[1])
    prev = jnp.roll(perm, 1)
    kl_c, kh_c = _bit_keys(_take0(lo, perm), _take0(hi, perm))
    kl_p, kh_p = _bit_keys(_take0(lo, prev), _take0(hi, prev))
    same = (_all_axes([kl_c[:, i] == kl_p[:, i] for i in range(d)])
            & _all_axes([kh_c[:, i] == kh_p[:, i] for i in range(d)])
            & (_take0(leaf, perm) == _take0(leaf, prev)))
    return same.at[0].set(False)


def _hash_dedup(lo, hi, leaf, origin, n_active, *, cap, act_hist):
    """Drop exact duplicates in hash order. Returns the compacted rows and the
    counters (n_nonbit, n_acth_diff, n_hash_collide, prot)."""
    valid = _iota_idx(cap) < n_active
    klo, khi = _bit_keys(lo, hi)
    h = _hash64(klo, khi, leaf)
    sort_key = jnp.where(valid, h.astype(jnp.int64), jnp.iinfo(jnp.int64).max)
    perm = _sort_perm_i64(sort_key)
    same = _adjacent_duplicates(lo, hi, leaf, perm)
    valid_s = perm < n_active
    keep_mask = valid_s & (~same)
    (out_lo, out_hi, out_leaf, out_origin, out_acth), out_n = _compact(
        keep_mask, cap, perm, [lo, hi, leaf, origin, act_hist])
    n_nonbit, n_acth, prot = _drop_stats(lo, hi, act_hist, perm, same, valid_s)
    hs = _take0(h, perm)
    hp = jnp.roll(hs, 1)
    vprev = jnp.roll(valid_s, 1).at[0].set(False)
    n_coll = jnp.sum(valid_s & vprev & (hs == hp) & (~same), dtype=jnp.int64)
    return (out_lo, out_hi, out_leaf, out_origin, out_n, out_acth,
            (n_nonbit, n_acth, n_coll, prot))


def _morton_dedup(lo, hi, leaf, origin, n_active, *, cap, axis_lo, axis_hi, act_hist):
    """Drop exact duplicates in (leaf, Morton) order; leaves the rows sorted by
    that key, which the subset prune relies on."""
    d = int(lo.shape[1])
    valid = _iota_idx(cap) < n_active
    morton = _box_morton_codes(lo, hi, axis_lo, axis_hi, MORTON_BITS)
    morton_max = jnp.int64(1) << jnp.int64(_morton_max_bits(d, MORTON_BITS))
    sort_key = jnp.where(valid, leaf.astype(jnp.int64) * morton_max + morton,
                         jnp.iinfo(jnp.int64).max)
    perm = _sort_perm_i64(sort_key)
    same = _adjacent_duplicates(lo, hi, leaf, perm)
    valid_s = perm < n_active
    keep_mask = valid_s & (~same)
    (out_lo, out_hi, out_leaf, out_origin, out_acth), out_n = _compact(
        keep_mask, cap, perm, [lo, hi, leaf, origin, act_hist])
    n_nonbit, n_acth, prot = _drop_stats(lo, hi, act_hist, perm, same, valid_s)
    return (out_lo, out_hi, out_leaf, out_origin, out_n, out_acth,
            (n_nonbit, n_acth, jnp.int64(0), prot))


def _subset_prune(lo, hi, leaf, origin, n_active, cap, K, *, act_hist):
    """Drop every box contained in another box of the same leaf.

    The rows are sorted by (leaf, Morton code of the centre). Each box is compared
    with its neighbours at offsets +-1..64 and +-128, +-256, ..., +-2048 in that
    order, and with the 256 largest boxes of its leaf. A box is dropped only if it
    lies in a distinct kept box (exact duplicates are already gone), so the union
    of the boxes is unchanged.
    """
    d = int(lo.shape[1])
    idx = _iota_idx(cap)
    valid_s = idx < n_active
    lo_s, hi_s, leaf_s = lo, hi, leaf

    offsets = list(range(1, min(SP_BAND, SP_LINEAR) + 1))
    o = 128
    while o <= SP_BAND:
        offsets.append(o)
        o *= 2
    offsets = offsets + [-o for o in offsets]
    offsets_arr = jnp.asarray(offsets, dtype=idx.dtype)

    def body(is_dom, o):
        j = idx - o
        in_bounds = (j >= 0) & (j < cap)
        sj = jnp.clip(j, 0, cap - 1).astype(jnp.uint32)
        lj, hj = _take0(lo_s, sj), _take0(hi_s, sj)
        vj = sj < n_active
        same_leaf = (_take0(leaf_s, sj) == leaf_s) & valid_s & vj
        cont = _all_axes([lj[:, a] <= lo_s[:, a] for a in range(d)]
                         + [hi_s[:, a] <= hj[:, a] for a in range(d)])
        return is_dom | (in_bounds & same_leaf & cont), None

    is_dominated, _ = jax.lax.scan(body, jnp.zeros((cap,), dtype=jnp.bool_), offsets_arr,
                                   unroll=SP_UNROLL)

    V = SP_HEAD
    vol = jnp.prod((hi_s - lo_s).astype(jnp.float32), axis=1)
    vol_key = _float_sort_key_u64(vol) & jnp.uint64(0xFFFFFFFF)
    vkey = (leaf_s.astype(jnp.uint64) << jnp.uint64(32)) | (jnp.uint64(0xFFFFFFFF) - vol_key)
    vkey = jnp.where(valid_s, vkey, jnp.uint64(0xFFFFFFFFFFFFFFFF))
    vperm = _sort_perm_u64(vkey)
    cnt = jnp.bincount(jnp.where(valid_s, leaf_s, K), length=K + 1)[:K]
    starts = cnt.cumsum() - cnt
    vpos = starts[:, None] + jnp.arange(V, dtype=jnp.int64)[None, :]
    head_ok = jnp.arange(V, dtype=jnp.int64)[None, :] < cnt[:, None]
    hidx = _take0(vperm, jnp.clip(vpos, 0, cap - 1).astype(jnp.uint32))
    head_lo = lo_s[hidx]
    head_hi = hi_s[hidx]

    def head_body(dom, v):
        cl = head_lo[leaf_s, v]
        ch = head_hi[leaf_s, v]
        hv = head_ok[leaf_s, v]
        cont = _all_axes([cl[:, a] <= lo_s[:, a] for a in range(d)]
                         + [hi_s[:, a] <= ch[:, a] for a in range(d)])
        cont = cont & hv & valid_s
        same = _all_axes([cl[:, a] == lo_s[:, a] for a in range(d)]
                         + [ch[:, a] == hi_s[:, a] for a in range(d)])
        return dom | (cont & ~same), None

    is_dominated, _ = jax.lax.scan(head_body, is_dominated, jnp.arange(V, dtype=jnp.int64),
                                   unroll=SP_UNROLL)
    keep_mask = valid_s & (~is_dominated)
    (n_lo, n_hi, n_leaf, n_orig, n_acth), n_next = _compact(
        keep_mask, cap, None, [lo, hi, leaf, origin, act_hist])
    return n_lo, n_hi, n_leaf, n_orig, n_next, n_acth


# --------------------------------------------------------------------------- #
# Segment kernel                                                              #
# --------------------------------------------------------------------------- #

def build_segment_kernel(leaf_regions, spec, *, cap, horizon, lookback, device=None):
    """Compile the segment kernel for one policy and return ``seg_fn``.

    ``leaf_regions``: leaf boxes of the policy (``policy.leaf_regions``);
    ``spec``: safety box (``env_specs.CartPoleSpec``); ``cap``: kernel batch cap
    (boxes per buffer); ``horizon``: number of steps to verify; ``lookback``:
    columns of the action-history ring kept per box; ``device``: the device that
    runs the kernel.

    ``seg_fn(pack, start_step) -> dict`` runs the boxes of ``pack`` (all at
    absolute step ``start_step``) forward until the horizon, a violation, an empty
    set, or until the next step would produce more than ``cap`` pieces
    (``hit_cap``). ``pack`` is a host pack (dict of arrays, the start set) or
    ``{"parts": [device_pool.DevPiece, ...], "n": M}``. The survivors are returned
    as a ``DevPiece`` on the device.
    """
    d = STATE_DIM
    safe_lo, safe_hi = spec.safe_lo, spec.safe_hi
    hits_unsafe = _make_hits_unsafe(safe_lo, safe_hi)
    acth_id = jnp.array(0, dtype=ACTH_DTYPE)

    cap = int(cap)
    if not 0 < cap < 2 ** 31:
        raise ValueError(f"kernel batch cap must lie in (0, 2^31); got {cap}")
    horizon_steps = int(horizon)
    n_col = max(int(lookback), 1)
    K, actions, leaf_lo, leaf_hi = leaf_arrays(leaf_regions, safe_lo, safe_hi)
    if int(np.unique(np.asarray(actions)).size) > 255:
        raise ValueError("the action history stores action indices as uint8")
    chunk = min(SOURCE_CHUNK, cap) or cap

    # Leaf ids and origins use the narrowest unsigned dtype for K (the value K
    # itself is the sentinel of the head pass).
    leaf_dtype = _small_uint_dtype(K)
    orig_dtype = _small_uint_dtype(K)
    leaf_template = jnp.broadcast_to(jnp.arange(K, dtype=leaf_dtype)[None, :],
                                     (chunk, K)).reshape(-1)
    ax_lo, ax_hi = spec.axis_ranges(VELOCITY_AXIS_RANGE)
    axis_lo = jnp.array(ax_lo, dtype=FDTYPE)
    axis_hi = jnp.array(ax_hi, dtype=FDTYPE)
    horizon_j = jnp.int64(horizon_steps)
    U = int(min(UNSAFE_CAPTURE_CAP, cap))
    U_j = jnp.int64(U)
    tiers = _reduce_tier_sizes(cap, REDUCE_TIERS)
    rs_ncol = len(REDUCE_STATS_COLUMNS)
    print(f"[kernel] cap={cap:,} K={K} horizon={horizon_steps} lookback={n_col} "
          f"tiers={tiers} capture_cap={U}", flush=True)

    def _reduce_at(T, nl, nh, nleaf, norig, ncount, nacth):
        """Reduction on the prefix [:T]; returns the rows, the per-step counters
        ``rs`` and the max protrusion per axis ``rp``."""
        if T != cap:
            nl, nh, nleaf, norig, nacth = nl[:T], nh[:T], nleaf[:T], norig[:T], nacth[:T]
        n0 = jnp.asarray(ncount, jnp.int64)
        nl, nh, nleaf, norig, ncount, nacth, q_st = _hash_dedup(
            nl, nh, nleaf, norig, ncount, cap=T, act_hist=nacth)
        n1 = jnp.asarray(ncount, jnp.int64)
        nl, nh, nleaf, norig, ncount, nacth, m_st = _morton_dedup(
            nl, nh, nleaf, norig, ncount, cap=T, axis_lo=axis_lo, axis_hi=axis_hi,
            act_hist=nacth)
        n2 = jnp.asarray(ncount, jnp.int64)
        nl, nh, nleaf, norig, ncount, nacth = _subset_prune(
            nl, nh, nleaf, norig, ncount, T, K, act_hist=nacth)
        n3 = jnp.asarray(ncount, jnp.int64)
        rs = jnp.stack([n0, n0 - n1, q_st[0], q_st[1], q_st[2],
                        n1 - n2, m_st[0], m_st[1], n2 - n3, n3]).astype(jnp.int64)
        rp = jnp.maximum(q_st[3], m_st[3]).astype(jnp.float64)
        if T != cap:
            nl, nh = _pad_rows(nl, cap), _pad_rows(nh, cap)
            nleaf, norig = _pad_rows(nleaf, cap), _pad_rows(norig, cap)
            nacth = _pad_rows(nacth, cap)
        return nl, nh, nleaf, norig, ncount, nacth, rs, rp

    def _reduce(nl, nh, nleaf, norig, ncount, nacth):
        """Run the reduction on the smallest prefix tier that holds the live rows."""
        if len(tiers) == 1:
            return _reduce_at(cap, nl, nh, nleaf, norig, ncount, nacth)
        branches = [(lambda args, T=T: _reduce_at(T, *args)) for T in tiers]
        return jax.lax.switch(_tier_index(ncount, tiers), branches,
                              (nl, nh, nleaf, norig, ncount, nacth))

    def _assign_leaf(boxes_lo, boxes_hi, n_in):
        """Leaf of every box: the leaf that contains its centre. Every pooled box
        was produced as (image ∩ leaf), so this recovers its leaf exactly."""
        center = (boxes_lo + boxes_hi) * jnp.array(0.5, dtype=FDTYPE)
        Sc = chunk
        num_src_chunks = (cap + Sc - 1) // Sc

        def chunk_body(box_leaf, ci):
            start = jnp.minimum(ci * Sc, jnp.int64(cap - Sc))
            c = jax.lax.dynamic_slice_in_dim(center, start, Sc, axis=0)
            contains = jnp.all((c[:, None, :] >= leaf_lo[None, :, :])
                               & (c[:, None, :] <= leaf_hi[None, :, :]), axis=-1)
            leaf_chunk = jnp.argmax(contains, axis=-1).astype(leaf_dtype)
            return jax.lax.dynamic_update_slice_in_dim(box_leaf, leaf_chunk, start, axis=0), None

        n_chunks = jnp.minimum((jnp.asarray(n_in, jnp.int64) + Sc - 1) // Sc,
                               jnp.int64(num_src_chunks))
        return jax.lax.fori_loop(jnp.int64(0), n_chunks, lambda i, bl: chunk_body(bl, i)[0],
                                 jnp.zeros((cap,), dtype=leaf_dtype))

    def _fits(lo, hi, leaf, n_act):
        """Would the next step of these boxes produce at most ``cap`` pieces?
        The bound n * K is tried first, the exact count only if it exceeds cap."""
        p_lo, p_hi, p_valid = _propagate_all(lo, hi, leaf, n_act, actions)
        ub = n_act * jnp.int64(K)
        cnt = jax.lax.cond(
            ub > jnp.int64(cap),
            lambda: _count_pieces(p_lo, p_hi, p_valid, leaf_lo, leaf_hi, cap, chunk, n_act),
            lambda: ub)
        return cnt <= jnp.int64(cap)

    def _core(boxes_lo, boxes_hi, box_origin, act_hist_in, n_in, start_step):
        s_lo, s_hi, s_acth = boxes_lo, boxes_hi, act_hist_in
        # An origin outside [0, K) would not survive the narrow cast: report it as
        # a loss of soundness (it cannot occur for packs of this kernel).
        _vin = jnp.arange(cap, dtype=jnp.int64) < n_in
        seed_overflow = jnp.any(_vin & ((box_origin < 0) | (box_origin >= jnp.int32(K))))
        s_orig = box_origin.astype(orig_dtype)
        s_leaf = _assign_leaf(s_lo, s_hi, n_in)
        n0 = n_in
        seed_hit_cap = ~_fits(s_lo, s_hi, s_leaf, n0)

        traj0 = jnp.zeros((horizon_steps + 1,), dtype=jnp.int64).at[0].set(n0)
        unsafe_origins0 = jnp.zeros((K,), dtype=jnp.bool_)
        cu_lo0 = jnp.full((K, d), jnp.nan, dtype=FDTYPE)
        cu_hi0 = jnp.full((K, d), jnp.nan, dtype=FDTYPE)
        cu_orig0 = jnp.zeros((K,), dtype=jnp.bool_)
        cu_acth0 = jnp.full((K, n_col), acth_id, dtype=ACTH_DTYPE)
        cub_lo0 = jnp.zeros((U, d), dtype=FDTYPE)
        cub_hi0 = jnp.zeros((U, d), dtype=FDTYPE)
        cub_orig0 = jnp.zeros((U,), dtype=jnp.int32)
        cub_acth0 = jnp.full((U, n_col), acth_id, dtype=ACTH_DTYPE)
        rs_traj0 = jnp.zeros((horizon_steps + 1, rs_ncol), dtype=jnp.int64)
        rp_traj0 = jnp.zeros((horizon_steps + 1, d), dtype=jnp.float64)

        def cond_fn(carry):
            (local_step, n_active, _bl, _bh, _bleaf, _borig, unsafe_found,
             _fus, _pk, _sl, _ah, _cl, _ch, _co, _ca, _uo, _traj, hit_cap, *_pb) = carry
            abs_t = start_step + local_step
            return ((n_active > jnp.int64(0)) & (abs_t < horizon_j) & (~hit_cap)
                    & (~unsafe_found))

        def body_fn(carry):
            (local_step, n_active, b_lo, b_hi, b_leaf, b_orig, unsafe_found,
             first_unsafe_step, peak_boxes, soundness_lost, act_hist,
             cu_lo, cu_hi, cu_orig, cu_acth, unsafe_origins, traj_n, hit_cap,
             cub_lo, cub_hi, cub_orig, cub_acth, cu_n, cu_n_total,
             rs_traj, rp_traj) = carry
            abs_t = start_step + local_step

            # Record this step's action in column (abs_t % lookback) of every box.
            col = (abs_t % jnp.int64(n_col)).astype(jnp.int64)
            cur_action = actions[b_leaf].astype(ACTH_DTYPE)
            act_hist = jax.lax.dynamic_update_slice_in_dim(
                act_hist, cur_action[:, None], col, axis=1)

            # Interval step and safety check.
            prop_lo, prop_hi, valid_src = _propagate_all(b_lo, b_hi, b_leaf, n_active, actions)
            per_box_unsafe = hits_unsafe(prop_lo, prop_hi) & valid_src
            n_prop = n_active
            any_unsafe = jnp.any(per_box_unsafe)
            ever = jnp.zeros((K,), dtype=jnp.int32).at[b_orig].max(
                per_box_unsafe.astype(jnp.int32))
            unsafe_origins = unsafe_origins | (ever > 0)

            # Violating boxes at the first violating step: per-origin hull and
            # history maximum, and every violating box individually (strided
            # subsample above the capture cap).
            is_first_unsafe = any_unsafe & (~unsafe_found)
            unsafe_lo_in = jnp.where(per_box_unsafe[:, None], prop_lo, jnp.inf)
            unsafe_hi_in = jnp.where(per_box_unsafe[:, None], prop_hi, -jnp.inf)
            cand_lo = jnp.full((K, d), jnp.inf, dtype=prop_lo.dtype).at[b_orig].min(unsafe_lo_in)
            cand_hi = jnp.full((K, d), -jnp.inf, dtype=prop_hi.dtype).at[b_orig].max(unsafe_hi_in)
            origin_unsafe_now = jnp.zeros((K,), dtype=jnp.bool_).at[b_orig].max(
                per_box_unsafe.astype(jnp.bool_))
            acth_rows = act_hist
            acth_unsafe_in = jnp.where(per_box_unsafe[:, None], acth_rows, acth_id)
            cand_acth = jnp.full((K, n_col), acth_id, dtype=ACTH_DTYPE).at[b_orig].max(
                acth_unsafe_in)
            cu_lo = jnp.where(is_first_unsafe, cand_lo, cu_lo)
            cu_hi = jnp.where(is_first_unsafe, cand_hi, cu_hi)
            cu_orig = jnp.where(is_first_unsafe, origin_unsafe_now, cu_orig)
            cu_acth = jnp.where(is_first_unsafe, cand_acth, cu_acth)

            def _capture(_):
                n_u = jnp.sum(per_box_unsafe.astype(jnp.int64))
                stride = jnp.maximum((n_u + U_j - 1) // U_j, jnp.int64(1))
                rank = jnp.cumsum(per_box_unsafe.astype(jnp.int64)) - 1
                keep_u = per_box_unsafe & ((rank % stride) == 0)
                pos = jnp.cumsum(keep_u.astype(jnp.int64)) - 1
                pos = jnp.where(keep_u & (pos < U_j), pos, U_j)
                return (cub_lo0.at[pos].set(prop_lo, mode="drop"),
                        cub_hi0.at[pos].set(prop_hi, mode="drop"),
                        cub_orig0.at[pos].set(b_orig.astype(jnp.int32), mode="drop"),
                        cub_acth0.at[pos].set(acth_rows, mode="drop"),
                        jnp.minimum(n_u, U_j), n_u)

            def _keep(_):
                return cub_lo, cub_hi, cub_orig, cub_acth, cu_n, cu_n_total

            (cub_lo, cub_hi, cub_orig, cub_acth, cu_n, cu_n_total) = jax.lax.cond(
                is_first_unsafe, _capture, _keep, None)

            first_unsafe_step = jnp.where(any_unsafe & (~unsafe_found), abs_t + 1,
                                          first_unsafe_step)
            unsafe_found = unsafe_found | any_unsafe

            # Pin the capture before the leaf cut so that XLA does not keep the
            # step's inputs alive across the reduction (identity, saves memory).
            (prop_lo, prop_hi, valid_src, b_orig, act_hist, n_prop,
             cu_lo, cu_hi, cu_orig, cu_acth, unsafe_origins,
             cub_lo, cub_hi, cub_orig, cub_acth, cu_n, cu_n_total,
             first_unsafe_step, unsafe_found, soundness_lost) = jax.lax.optimization_barrier(
                (prop_lo, prop_hi, valid_src, b_orig, act_hist, n_prop,
                 cu_lo, cu_hi, cu_orig, cu_acth, unsafe_origins,
                 cub_lo, cub_hi, cub_orig, cub_acth, cu_n, cu_n_total,
                 first_unsafe_step, unsafe_found, soundness_lost))

            # The previous prediction guarantees that this step fits the cap.
            (n_lo, n_hi, n_leaf, n_orig, n_cnt, overflow, n_acth) = _intersect_and_compact(
                prop_lo, prop_hi, valid_src, b_orig, leaf_lo, leaf_hi, leaf_template,
                K, cap, chunk, act_hist, n_prop)
            soundness_lost = soundness_lost | overflow

            n_lo, n_hi, n_leaf, n_orig, n_cnt, n_acth, rs, rp = _reduce(
                n_lo, n_hi, n_leaf, n_orig, n_cnt, n_acth)
            rs_traj = rs_traj.at[local_step + 1].set(rs)
            rp_traj = rp_traj.at[local_step + 1].set(rp)
            peak_boxes = jnp.maximum(peak_boxes, n_cnt)
            traj_n = traj_n.at[local_step + 1].set(n_cnt)

            # Stop before a step that would not fit: this box set is returned.
            hit_cap = ~_fits(n_lo, n_hi, n_leaf, n_cnt)

            return (local_step + jnp.int64(1), n_cnt, n_lo, n_hi, n_leaf, n_orig,
                    unsafe_found, first_unsafe_step, peak_boxes, soundness_lost,
                    n_acth, cu_lo, cu_hi, cu_orig, cu_acth, unsafe_origins,
                    traj_n, hit_cap, cub_lo, cub_hi, cub_orig, cub_acth, cu_n, cu_n_total,
                    rs_traj, rp_traj)

        init_carry = (
            jnp.int64(0), n0, s_lo, s_hi, s_leaf, s_orig,
            jnp.bool_(False), jnp.int64(-1), n0, seed_overflow,
            s_acth, cu_lo0, cu_hi0, cu_orig0, cu_acth0, unsafe_origins0,
            traj0, seed_hit_cap,
            cub_lo0, cub_hi0, cub_orig0, cub_acth0, jnp.int64(0), jnp.int64(0),
            rs_traj0, rp_traj0,
        )
        (local_step, n_active, b_lo, b_hi, b_leaf, b_orig, unsafe_found,
         first_unsafe_step, peak_boxes, soundness_lost, act_hist,
         cu_lo, cu_hi, cu_orig, cu_acth, unsafe_origins, traj_n, hit_cap,
         cub_lo, cub_hi, cub_orig, cub_acth, cu_n, cu_n_total,
         rs_traj, rp_traj) = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        return {
            "n_active": n_active, "local_step": local_step,
            "boxes_lo": b_lo, "boxes_hi": b_hi, "box_leaf": b_leaf,
            "box_origin": b_orig.astype(jnp.int32), "act_hist": act_hist,
            "unsafe_found": unsafe_found, "first_unsafe_step": first_unsafe_step,
            "peak_boxes": peak_boxes, "soundness_lost": soundness_lost,
            "hit_cap": hit_cap, "unsafe_origins": unsafe_origins,
            "cu_lo": cu_lo, "cu_hi": cu_hi, "cu_orig": cu_orig, "cu_acth": cu_acth,
            "cub_lo": cub_lo, "cub_hi": cub_hi, "cub_orig": cub_orig, "cub_acth": cub_acth,
            "cu_n": cu_n, "cu_n_total": cu_n_total, "traj_n_stage2": traj_n,
            "seed_overflow": seed_overflow, "rs_traj": rs_traj, "rp_traj": rp_traj,
        }

    # Ahead-of-time compilation with every constant of the kernel hoisted into an
    # explicit argument that is committed to ``device`` once: the compiled object
    # then has a plain call path with no implicit inputs.
    pin = dict(device=device) if device is not None else {}
    in_avals = (
        jax.ShapeDtypeStruct((cap, d), FDTYPE),              # boxes_lo
        jax.ShapeDtypeStruct((cap, d), FDTYPE),              # boxes_hi
        jax.ShapeDtypeStruct((cap,), jnp.int32),             # box_origin
        jax.ShapeDtypeStruct((cap, n_col), ACTH_DTYPE),      # act_hist
        jax.ShapeDtypeStruct((), jnp.int64),                 # n_in
        jax.ShapeDtypeStruct((), jnp.int64),                 # start_step
    )
    put = (lambda a: jax.device_put(a, device)) if device is not None else jax.device_put
    with _TRACE_LOCK:
        core_closed = jax.make_jaxpr(_core)(*in_avals)
        core_out_tree = jax.tree_util.tree_structure(jax.eval_shape(_core, *in_avals))
        core_jaxpr = core_closed.jaxpr

        def _core_conv(consts, *args):
            flat = jax.core.eval_jaxpr(core_jaxpr, consts, *args)
            return jax.tree_util.tree_unflatten(core_out_tree, flat)

        core_consts = [put(c) for c in core_closed.consts]
        const_avals = [jax.ShapeDtypeStruct(c.shape, c.dtype) for c in core_consts]
        core_closed = None
        # The four cap-sized inputs are dead once the loop has its initial carry:
        # donating them lets XLA reuse their memory for the outputs.
        core_compiled = (jax.jit(_core_conv, donate_argnums=(1, 2, 3, 4), **pin)
                         .lower(const_avals, *in_avals).compile())

    # Zero padding from each smaller tier to cap on the device (start-set staging).
    pad_compiled = {}
    for T in tiers:
        if T == cap:
            continue
        pad_avals = (jax.ShapeDtypeStruct((T, d), FDTYPE), jax.ShapeDtypeStruct((T, d), FDTYPE),
                     jax.ShapeDtypeStruct((T,), jnp.int32),
                     jax.ShapeDtypeStruct((T, n_col), ACTH_DTYPE))
        with _TRACE_LOCK:
            pad_compiled[T] = (
                jax.jit(lambda lo, hi, o, a: (_pad_rows(lo, cap), _pad_rows(hi, cap),
                                              _pad_rows(o, cap), _pad_rows(a, cap)), **pin)
                .lower(*pad_avals).compile())

    # Host staging buffers for host packs (allocated on first use, sliced to a tier).
    stage = {}

    def _stage_bufs(T_in):
        if not stage:
            stage["lo"] = np.zeros((cap, d), dtype=np.float32)
            stage["hi"] = np.zeros((cap, d), dtype=np.float32)
            stage["orig"] = np.zeros((cap,), dtype=np.int32)
            stage["acth"] = np.zeros((cap, n_col), dtype=np.uint8)
            stage["dirty"] = 0          # rows [0:dirty) may be non-zero
        return stage["lo"][:T_in], stage["hi"][:T_in], stage["orig"][:T_in], stage["acth"][:T_in]

    def _stage_host(parts, M):
        """Copy host packs into the staging buffers, move them to the device and
        pad them to cap there."""
        T_in = next(t for t in tiers if t >= M)
        lo_pad, hi_pad, orig_pad, acth_pad = _stage_bufs(T_in)
        off = 0
        for q in parts:
            m = int(q["lo"].shape[0])
            if m == 0:
                continue
            lo_pad[off:off + m] = np.asarray(q["lo"])
            hi_pad[off:off + m] = np.asarray(q["hi"])
            orig_pad[off:off + m] = np.asarray(q["origin"])
            acth_pad[off:off + m] = np.asarray(q["act_hist"])
            off += m
        # Rows [M:T_in) must be zero: clear what a previous larger pack wrote.
        dirty = stage["dirty"]
        if dirty > M:
            hi_clear = min(dirty, T_in)
            lo_pad[M:hi_clear] = 0
            hi_pad[M:hi_clear] = 0
            orig_pad[M:hi_clear] = 0
            acth_pad[M:hi_clear] = 0
        stage["dirty"] = M
        with DISPATCH_LOCK:
            arrays = [put(lo_pad), put(hi_pad), put(orig_pad), put(acth_pad)]
        if T_in != cap:
            arrays = pad_compiled[T_in](*arrays)
        return arrays

    def seg_fn(boxes_in, start_step):
        is_holder = isinstance(boxes_in, dict) and "parts" in boxes_in
        parts = boxes_in["parts"] if is_holder else [boxes_in]
        on_device = all(device_pool.is_dev(q) for q in parts)
        M = int(sum(int(q["n"]) if device_pool.is_dev(q) else int(q["lo"].shape[0])
                    for q in parts))
        if M > cap:
            raise ValueError(f"pack has {M} boxes > cap {cap}")

        t0 = time.time()
        if on_device:
            with DISPATCH_LOCK:
                lo_dev, hi_dev, orig_dev, acth_dev = device_pool.assemble(
                    parts, cap=cap, d=d, ncol=n_col, fdtype=FDTYPE, acth_dtype=ACTH_DTYPE)
            # Drop the references to the source pieces so their memory is freed
            # before the kernel runs.
            if is_holder:
                boxes_in["parts"] = []
            parts = None
        else:
            lo_dev, hi_dev, orig_dev, acth_dev = _stage_host(parts, M)
        t_stage = time.time() - t0
        t0 = time.time()
        out = core_compiled(core_consts, lo_dev, hi_dev, orig_dev, acth_dev,
                            jnp.int64(M), jnp.int64(int(start_step)))
        n_out = int(out["n_active"])
        t_core = time.time() - t0
        lo_dev = hi_dev = orig_dev = acth_dev = None
        steps_run = int(out["local_step"])
        any_unsafe = bool(out["unsafe_found"])
        fus = int(out["first_unsafe_step"])
        boxes_out = device_pool.emit_compact(
            out["boxes_lo"], out["boxes_hi"], out["box_origin"], out["act_hist"], n_out,
            K=K, n_actions=int(np.asarray(actions).max()) + 1, ncol=n_col)
        result = {
            "boxes_out": boxes_out,
            "n_out": n_out,
            "start_step": int(start_step),
            "end_step": int(start_step) + steps_run,
            "steps_run": steps_run,
            "any_unsafe": any_unsafe,
            "first_unsafe_step": fus if (any_unsafe and fus >= 0) else None,
            "soundness_lost": bool(out["soundness_lost"]) or bool(out["seed_overflow"]),
            "hit_cap": bool(out["hit_cap"]),
            "reduce_stats": np.asarray(out["rs_traj"])[:steps_run + 1],
            "reduce_prot": np.asarray(out["rp_traj"])[:steps_run + 1],
            "peak_boxes": int(out["peak_boxes"]),
            "traj": np.asarray(out["traj_n_stage2"])[:steps_run + 1].astype(np.int64),
            "t_stage_sec": round(t_stage, 3),
            "t_core_sec": round(t_core, 3),
            "n_unsafe_boxes": int(out["cu_n_total"]) if any_unsafe else 0,
        }
        if any_unsafe:
            result["capture"] = _violating_boxes(out, fus)
        return result

    seg_fn.cap = cap
    seg_fn.K = K
    return seg_fn


def _violating_boxes(out, first_unsafe_step):
    """Captured violating boxes of a segment: bounds, origin, action history and
    crossing step of each box, and the true number of violating boxes."""
    n_c = int(out["cu_n"])
    return {
        "first_unsafe_step": int(first_unsafe_step),
        "lo": np.asarray(out["cub_lo"])[:n_c],
        "hi": np.asarray(out["cub_hi"])[:n_c],
        "origin": np.asarray(out["cub_orig"])[:n_c].astype(np.int32),
        "act_hist": np.asarray(out["cub_acth"])[:n_c],
        "step": np.full((n_c,), int(first_unsafe_step), dtype=np.int32),
        "n_total": int(out["cu_n_total"]),
    }
