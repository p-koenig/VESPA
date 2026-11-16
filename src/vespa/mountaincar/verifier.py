"""Zonotope reachability verification of a decision-tree policy (goal property).

Property: every trajectory that starts in S0 = {x in [-0.6, -0.4], v = 0} reaches the goal x >= 0.45
within HORIZON = 200 steps. S0 is split into 2^14 start state regions. The verifier carries one zonotope
per (reachable piece, tree region): each step propagates every zonotope under its region's action,
removes the zonotopes whose hull lies entirely in the goal, and routes the rest to every region its hull
overlaps (restricted to that region with ``dynamics.contract_to_box``). A zonotope that is still in the
frontier after step 200 is a violation. The verifier records, for every start state region, whether a
violating zonotope descends from it, and returns the ratio of safe start state regions (weighted by width
in x). An empty frontier before step 200 proves the property.

The population lives in a static buffer of ``cap`` slots. An overflow before the last step makes the
result undecided (status SOUNDNESS_LOST), never safe. Every violating zonotope is captured with its own
history of the last 200 actions, which the counterexample extraction walks back in time.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from vespa.mountaincar import dynamics
from vespa.mountaincar.config import HORIZON, INIT_SPLITS, UNSAFE_CAPTURE_CAP

_IDX = jnp.int64
_F = jnp.float32           # storage dtype of the zonotopes and of the action history
D = dynamics.STATE_DIM
OL = dynamics.ORDER_LIMIT
LOOKBACK = HORIZON         # length of the per-object action history

# ---------------------------------------------------------------------------
# GPU compiler. The CUDA-12 ptxas 12.9 bundled with jax 0.9.2 miscompiles 64-bit address offsets
# >= 2 GiB on sm_120 (Blackwell, e.g. RTX PRO 6000), which buffers above 2^21 objects reach. The solver
# is therefore compiled with the toolchain in VESPA_PTXAS_DIR (bin/ptxas >= 13, bin/nvlink and
# nvvm/libdevice/libdevice.10.bc). Unset: the bundled toolchain, and caps above 2^21 raise on sm_120.
# CPU and other GPU architectures need no setting.
# ---------------------------------------------------------------------------
PTXAS_MIN_MAJOR = 13
SAFE_CAP_DEFAULT_PTXAS = 2 ** 31 // 1024
PTXAS_USED = {"dir": None, "version": None}
_PTXAS_CACHE: dict = {}


def _ptxas_version(directory: str):
    """(major, minor, patch) of <directory>/bin/ptxas if the directory is a usable CUDA toolchain."""
    if directory in _PTXAS_CACHE:
        return _PTXAS_CACHE[directory]
    version = None
    ptxas = os.path.join(directory, "bin", "ptxas")
    nvlink = os.path.join(directory, "bin", "nvlink")
    libdevice = os.path.join(directory, "nvvm", "libdevice", "libdevice.10.bc")
    if os.access(ptxas, os.X_OK) and os.access(nvlink, os.X_OK) and os.path.isfile(libdevice):
        try:
            out = subprocess.run([ptxas, "--version"], capture_output=True, text=True, timeout=30).stdout
            m = re.search(r"V(\d+)\.(\d+)\.(\d+)", out)
            version = tuple(int(x) for x in m.groups()) if m else None
        except Exception:
            version = None
    _PTXAS_CACHE[directory] = version
    return version


def _is_sm120() -> bool:
    """True on compute capability 12.x. An unknown capability counts as 12.x."""
    try:
        cc = str(getattr(jax.devices()[0], "compute_capability", "") or "")
        return (cc.split(".")[0] == "12") if cc else True
    except Exception:
        return True


def _solver_jit(fn, cap: int):
    """``jax.jit(fn)``. On GPU the program is compiled with ptxas >= 13 when VESPA_PTXAS_DIR is set."""
    if jax.default_backend() != "gpu":
        return jax.jit(fn)
    choice = os.environ.get("VESPA_PTXAS_DIR", "")
    if choice:
        v = _ptxas_version(choice)
        if v is None or v[0] < PTXAS_MIN_MAJOR:
            raise ValueError(f"VESPA_PTXAS_DIR={choice!r} has no bin/ptxas >= {PTXAS_MIN_MAJOR} "
                             f"with bin/nvlink and libdevice")
        version = ".".join(map(str, v))
        if PTXAS_USED.get("dir") != choice:
            print(f"[verify] solver compiled with ptxas {version} ({choice})", flush=True)
        PTXAS_USED.update(dir=choice, version=version)
        return jax.jit(fn, compiler_options={"xla_gpu_cuda_data_dir": choice})
    if int(cap) > SAFE_CAP_DEFAULT_PTXAS and _is_sm120():
        raise ValueError(f"cap {int(cap)} > {SAFE_CAP_DEFAULT_PTXAS} needs ptxas >= {PTXAS_MIN_MAJOR} on this GPU "
                         f"(ptxas 12.9 miscompiles address offsets >= 2 GiB on sm_120); set VESPA_PTXAS_DIR")
    PTXAS_USED.update(dir=None, version="default")
    return jax.jit(fn)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

_SOLVERS: dict = {}


def _build_solver(cap: int, K: int):
    """Compile (once per buffer size and number of regions) the 200-step reachability loop.

    Inputs: initial population (c0 [cap, d], G0 [cap, d, OL], region [cap], origin [cap], valid [cap]),
    region actions [K] and routing boxes [K, d]. Outputs: steps run, violation flag, first violating
    step, peak population, overflow flag, overflow step, population per step, the captured violating
    objects and the per-origin violation bits.
    """
    eps = jnp.asarray(dynamics.step_eps(), dtype=_F)
    hold = jnp.asarray(dynamics.MIN_ACTION, dtype=_F)     # history entry of an empty slot
    goal = jnp.asarray(dynamics.GOAL_POSITION, dtype=_F)
    gax = dynamics.GOAL_AXIS

    step_v = jax.vmap(lambda c, G, a: dynamics.zonotope_step(c, G, a, eps), in_axes=(0, 0, 0))
    hull_v = jax.vmap(dynamics.zono_hull, in_axes=(0, 0))
    contract_v = jax.vmap(partial(dynamics.contract_to_box, order_limit=OL), in_axes=(0, 0, 0, 0))

    def solve(c0, G0, region0, origin0, valid0, actions, route_lo, route_hi):
        n_per_step = jnp.zeros((HORIZON + 1,), dtype=_IDX).at[0].set(jnp.sum(valid0.astype(_IDX)))

        def cond(carry):
            t, valid, overflow = carry[0], carry[5], carry[10]
            return (t < HORIZON) & jnp.any(valid) & (~overflow)

        def body(carry):
            (t, c, G, region, origin, valid, history, found, first, peak, overflow, cap_step,
             n_per_step, capture, bad_origin) = carry

            act = actions[region]
            col = (t % jnp.int64(LOOKBACK)).astype(jnp.int64)
            new_col = jnp.where(valid, act.astype(_F), hold)
            history = jax.lax.dynamic_update_slice_in_dim(history, new_col[:, None], col, axis=1)

            pc, pG = step_v(c, G, act)
            pc = jnp.where(valid[:, None], pc, 0.0)
            pG = jnp.where(valid[:, None, None], pG, 0.0)
            plo, phi = hull_v(pc, pG)
            plo = jnp.where(valid[:, None], plo, jnp.inf)
            phi = jnp.where(valid[:, None], phi, -jnp.inf)

            # Goal: a zonotope whose hull lies at x >= 0.45 has reached the goal and leaves the frontier.
            reached = plo[:, gax] >= goal
            valid = valid & (~reached)
            plo = jnp.where(valid[:, None], plo, jnp.inf)
            phi = jnp.where(valid[:, None], phi, -jnp.inf)
            phi = phi.at[:, gax].set(jnp.minimum(phi[:, gax], goal))

            # Violation: still in the frontier after the last step.
            hit = valid & (t + 1 == HORIZON)
            any_hit = jnp.any(hit)
            first = jnp.where(any_hit & (~found), t + 1, first)
            found = found | any_hit
            bad_origin = bad_origin.at[jnp.where(hit, origin, jnp.int32(cap))].set(True, mode="drop")

            # Capture every violating zonotope with its action history, then drop it.
            (cap_lo, cap_hi, cap_origin, cap_history, n_cap, n_total, cap_c, cap_G) = capture
            n_hit = jnp.sum(hit.astype(jnp.int64))
            pos = n_cap + jnp.cumsum(hit.astype(jnp.int64)) - 1
            dest = jnp.where(hit & (pos < UNSAFE_CAPTURE_CAP), pos, jnp.int64(UNSAFE_CAPTURE_CAP)).astype(jnp.int32)
            cap_c = cap_c.at[dest].set(pc, mode="drop")
            cap_G = cap_G.at[dest].set(pG, mode="drop")
            cap_lo = cap_lo.at[dest].set(plo, mode="drop")
            cap_hi = cap_hi.at[dest].set(phi, mode="drop")
            cap_origin = cap_origin.at[dest].set(origin, mode="drop")
            cap_history = cap_history.at[dest].set(history, mode="drop")
            n_cap = jnp.minimum(n_cap + n_hit, jnp.int64(UNSAFE_CAPTURE_CAP))
            n_total = n_total + n_hit
            capture = (cap_lo, cap_hi, cap_origin, cap_history, n_cap, n_total, cap_c, cap_G)
            valid = valid & (~hit)

            # Route every zonotope to each region its hull overlaps.
            slo, shi = hull_v(pc, pG)
            slo = jnp.where(valid[:, None], slo, jnp.inf)
            shi = jnp.where(valid[:, None], shi, -jnp.inf)
            overlap = jnp.all((slo[:, None, :] <= route_hi[None, :, :])
                              & (shi[:, None, :] >= route_lo[None, :, :]), axis=-1) & valid[:, None]

            # A zonotope whose overlapped regions all carry the same action stays whole (one destination,
            # no contraction). This is exact: the zonotope contains the union of its per-region parts.
            action_f = actions.astype(jnp.float64)[None, :]
            a_min = jnp.min(jnp.where(overlap, action_f, jnp.inf), axis=1)
            a_max = jnp.max(jnp.where(overlap, action_f, -jnp.inf), axis=1)
            whole_src = valid & (a_max == a_min)
            first_region = jnp.argmax(overlap, axis=1)
            onehot = jnp.arange(K, dtype=jnp.int32)[None, :] == first_region[:, None]
            overlap = jnp.where(whole_src[:, None], onehot & overlap, overlap)

            # Compact the (zonotope, region) pairs into the buffer.
            flat = overlap.reshape(-1)
            pos = jnp.cumsum(flat.astype(jnp.int32)) - 1
            total = jnp.sum(flat.astype(_IDX))
            dest = jnp.where(flat, pos, jnp.int32(cap))
            src_i = jnp.broadcast_to(jnp.arange(cap, dtype=jnp.int32)[:, None], (cap, K)).reshape(-1)
            src_l = jnp.broadcast_to(jnp.arange(K, dtype=jnp.int32)[None, :], (cap, K)).reshape(-1)
            g_i = jnp.zeros((cap,), jnp.int32).at[dest].set(src_i, mode="drop")
            g_l = jnp.zeros((cap,), jnp.int32).at[dest].set(src_l, mode="drop")
            overflow_now = total > cap
            n_new = jnp.minimum(total, jnp.int64(cap))
            nvalid = jnp.arange(cap, dtype=_IDX) < n_new

            nc_raw = pc[g_i]
            nG_raw = pG[g_i]
            nc, nG, ok = contract_v(nc_raw, nG_raw, route_lo[g_l], route_hi[g_l])
            whole = whole_src[g_i]
            nc = jnp.where(whole[:, None], nc_raw, nc)
            nG = jnp.where(whole[:, None, None], nG_raw, nG)
            ok = ok | whole
            nvalid = nvalid & ok
            nc = jnp.where(nvalid[:, None], nc, 0.0)
            nG = jnp.where(nvalid[:, None, None], nG, 0.0)

            n_valid = jnp.sum(nvalid.astype(_IDX))
            peak = jnp.maximum(peak, n_valid)
            n_per_step = jax.lax.dynamic_update_index_in_dim(n_per_step, n_valid, t + 1, axis=0)
            cap_step = jnp.where(overflow_now & (cap_step < 0), t + 1, cap_step)
            return (t + 1, nc, nG, g_l, origin[g_i], nvalid, history[g_i], found, first, peak,
                    overflow | overflow_now, cap_step, n_per_step, capture, bad_origin)

        n = UNSAFE_CAPTURE_CAP
        capture0 = (
            jnp.full((n, D), jnp.nan, dtype=_F),
            jnp.full((n, D), jnp.nan, dtype=_F),
            jnp.zeros((n,), dtype=jnp.int32),
            jnp.full((n, LOOKBACK), hold, dtype=_F),
            jnp.int64(0), jnp.int64(0),
            jnp.full((n, D), jnp.nan, dtype=_F),
            jnp.zeros((n, D, OL), dtype=_F),
        )
        init = (jnp.int64(0), c0, G0, region0, origin0, valid0,
                jnp.full((cap, LOOKBACK), hold, dtype=_F),
                jnp.bool_(False), jnp.int64(-1), jnp.sum(valid0.astype(_IDX)), jnp.bool_(False),
                jnp.int64(-1), n_per_step, capture0, jnp.zeros((cap,), dtype=jnp.bool_))
        out = jax.lax.while_loop(cond, body, init)
        (t, _c, _G, _r, _o, _v, _h, found, first, peak, overflow, cap_step, n_per_step, capture, bad_origin) = out
        return t, found, first, peak, overflow, cap_step, n_per_step, capture, bad_origin

    return _solver_jit(solve, cap)


def _solver(cap: int, K: int):
    key = (int(cap), int(K))
    if key not in _SOLVERS:
        _SOLVERS[key] = _build_solver(int(cap), int(K))
    return _SOLVERS[key]


# ---------------------------------------------------------------------------
# Initial population
# ---------------------------------------------------------------------------

def _split_start_set(c, G, region, n_splits):
    """Bisect every initial zonotope ``n_splits`` times along its widest generator (exact union).

    Halving a piece quarters its per-step linearisation remainder, so 2^n pieces buy a 4^n smaller
    accumulated remainder over the horizon.
    """
    for _ in range(int(n_splits)):
        n = c.shape[0]
        rows = np.arange(n)
        j = np.argmax(np.abs(G).sum(axis=1), axis=1)
        gsel = G[rows, :, j]
        G_half = G.copy()
        G_half[rows, :, j] = 0.5 * gsel
        c = np.concatenate([c - 0.5 * gsel, c + 0.5 * gsel], axis=0)
        G = np.concatenate([G_half, G_half], axis=0)
        region = np.concatenate([region, region], axis=0)
    return c, G, region


@dataclass
class Verification:
    """Result of one verification.

    ``safe``: the property is proven. ``status``: "OK" or "SOUNDNESS_LOST" (buffer overflow before the
    last step, undecided). ``violated``: a violating zonotope was found (the policy is not proven).
    ``certified_fraction``: ratio of safe start state regions, None when undecided.
    ``capture``: the violating zonotopes (centre, generators, hull, origin piece, action history),
    ``piece_lo``/``piece_hi`` the hulls of the start state regions and ``piece_violating`` whether a
    violating zonotope descends from each of them.
    """
    safe: bool
    status: str
    violated: bool
    certified_fraction: float | None
    n_init_pieces: int
    n_violating_pieces: int
    n_violating_objects: int
    peak: int
    cap: int
    cap_step: int | None
    steps_run: int
    n_per_step: list
    capture: dict
    piece_lo: np.ndarray
    piece_hi: np.ndarray
    piece_violating: np.ndarray


def verify_regions(regions: list[dict], cap: int) -> Verification:
    """Verify the goal property for a policy given by its leaf regions, with a buffer of ``cap`` objects."""
    K = len(regions)
    raw_lo = np.array([[r["bounds"][i][0] for i in range(D)] for r in regions], dtype=np.float32)
    raw_hi = np.array([[r["bounds"][i][1] for i in range(D)] for r in regions], dtype=np.float32)
    actions = jnp.array([r["action"] for r in regions], dtype=_F)
    leaf_lo = jnp.array(raw_lo, dtype=_F)
    leaf_hi = jnp.array(raw_hi, dtype=_F)

    # Routing boxes: the region intersected with the state box and with x <= goal.
    route_lo = jnp.maximum(leaf_lo, jnp.asarray(dynamics.STATE_LO, dtype=_F))
    route_hi = jnp.minimum(leaf_hi, jnp.asarray(dynamics.STATE_HI, dtype=_F))
    gax = dynamics.GOAL_AXIS
    route_hi = route_hi.at[:, gax].set(jnp.minimum(route_hi[:, gax], jnp.asarray(dynamics.GOAL_POSITION, dtype=_F)))

    # One initial zonotope per region that meets S0, split into 2^INIT_SPLITS start state regions.
    state_lo = jnp.maximum(leaf_lo, jnp.array(dynamics.INIT_LO, dtype=_F))
    state_hi = jnp.minimum(leaf_hi, jnp.array(dynamics.INIT_HI, dtype=_F))
    active = np.asarray(jnp.all(state_lo <= state_hi, axis=-1)).astype(bool)
    idx = np.flatnonzero(active)
    lo_np = np.asarray(state_lo)[idx]
    hi_np = np.asarray(state_hi)[idx]
    c_init = np.ascontiguousarray(0.5 * (lo_np + hi_np), dtype=np.float32)
    G_init = np.zeros((idx.size, D, OL), dtype=np.float32)
    for j in range(D):
        G_init[:, j, j] = 0.5 * (hi_np[:, j] - lo_np[:, j])
    c_init, G_init, region_init = _split_start_set(c_init, G_init, idx.astype(np.int32), INIT_SPLITS)
    n_init = int(c_init.shape[0])

    c0 = np.zeros((cap, D), dtype=np.float32)
    G0 = np.zeros((cap, D, OL), dtype=np.float32)
    c0[:n_init] = c_init
    G0[:n_init] = G_init
    region0 = np.zeros((cap,), dtype=np.int32)
    region0[:n_init] = region_init
    origin0 = np.zeros((cap,), dtype=np.int32)
    origin0[:n_init] = np.arange(n_init, dtype=np.int32)
    valid0 = np.arange(cap) < n_init

    out = _solver(cap, K)(jnp.asarray(c0), jnp.asarray(G0), jnp.asarray(region0), jnp.asarray(origin0),
                          jnp.asarray(valid0), actions, route_lo, route_hi)
    t, found, first, peak, overflow, cap_step, n_per_step, capture, bad_origin = out

    found = bool(found)
    first = int(first)
    cap_step = int(cap_step)
    # An overflow at or after the step of the first violation cannot change the verdict.
    late_overflow = bool(overflow) and found and cap_step >= first
    soundness_lost = bool(overflow) and not late_overflow
    safe = (not found) and (not soundness_lost)
    violated = (not safe) and first >= 0

    absG = np.abs(np.asarray(G_init, dtype=np.float64)).sum(axis=2)
    piece_lo = np.asarray(c_init, dtype=np.float64) - absG
    piece_hi = np.asarray(c_init, dtype=np.float64) + absG
    bad = np.asarray(bad_origin).astype(bool)[:n_init]
    width_x = np.maximum(piece_hi[:, 0] - piece_lo[:, 0], 0.0)
    certified = None if soundness_lost else float(1.0 - width_x[bad].sum() / width_x.sum())

    captured = {}
    n_total = 0
    if violated:
        cap_lo, cap_hi, cap_origin, cap_history, n_cap, n_total, cap_c, cap_G = capture
        n_cap = int(n_cap)
        captured = {"c": np.asarray(cap_c)[:n_cap], "G": np.asarray(cap_G)[:n_cap],
                    "lo": np.asarray(cap_lo)[:n_cap], "hi": np.asarray(cap_hi)[:n_cap],
                    "origin": np.asarray(cap_origin)[:n_cap], "history": np.asarray(cap_history)[:n_cap]}
    return Verification(
        safe=safe, status="SOUNDNESS_LOST" if soundness_lost else "OK", violated=violated,
        certified_fraction=certified, n_init_pieces=n_init, n_violating_pieces=int(bad.sum()),
        n_violating_objects=int(n_total) if violated else 0, peak=int(peak), cap=int(cap),
        cap_step=None if cap_step < 0 else cap_step, steps_run=int(t),
        n_per_step=[int(v) for v in np.asarray(n_per_step).tolist()], capture=captured,
        piece_lo=piece_lo, piece_hi=piece_hi, piece_violating=bad)
