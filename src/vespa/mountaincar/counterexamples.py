"""Counterexample regions: restart boxes at step 1 obtained from the violating zonotopes.

A violating zonotope is still outside the goal at step 200. Walking it back 200 steps with its own recorded
actions (``walk_back``) gives a set of start states whose trajectories end in it. Every such set is
intersected with its own start state region (the true ancestor is a start state of that region), and the
interval hull, rounded outward to float32, becomes one counterexample region. VESPA fine-tuning starts
episodes uniformly in these regions.
"""
from __future__ import annotations

import numpy as np

from vespa.mountaincar import dynamics
from vespa.mountaincar.config import HORIZON, MAX_COUNTEREXAMPLE_OBJECTS
from vespa.mountaincar.verifier import Verification
from vespa.mountaincar.walk_back import walk_back

# Every violating zonotope is captured at step HORIZON and walked back HORIZON steps, so every
# counterexample region lives at the first step of an episode.
RESTART_STEP = 1


class CounterexampleExtractionFailed(Exception):
    """Raised when no violating zonotope produced a counterexample region."""


def _contract_rows_to_box(c, G, lo, hi, chunk=65536):
    """Restrict each row's zonotope to the box [lo, hi] with the verifier's ``contract_to_box``.

    Only rows whose hull leaves the box are touched. Returns ``(c, G, ok)`` with ``ok`` False for rows
    whose intersection is empty.
    """
    import jax
    import jax.numpy as jnp
    from functools import partial
    rad = np.abs(G).sum(axis=2)
    need = np.any((c - rad < lo[None, :]) | (c + rad > hi[None, :]), axis=1)
    ok = np.ones((c.shape[0],), dtype=bool)
    if not need.any():
        return c, G, ok
    order_limit = int(G.shape[2])
    fn = jax.jit(jax.vmap(partial(dynamics.contract_to_box, order_limit=order_limit), in_axes=(0, 0, None, None)))
    idx = np.flatnonzero(need)
    c = c.copy()
    G = G.copy()
    for s in range(0, idx.size, chunk):
        ii = idx[s:s + chunk]
        cc, GG, oo = fn(jnp.asarray(c[ii]), jnp.asarray(G[ii]), jnp.asarray(lo), jnp.asarray(hi))
        c[ii] = np.asarray(cc)
        G[ii] = np.asarray(GG)
        ok[ii] = np.asarray(oo)
    return c, G, ok


def _outward_f32(lo, hi):
    """Round float64 bounds outward to float32."""
    lo32 = lo.astype(np.float32)
    hi32 = hi.astype(np.float32)
    lo32 = np.where(lo32.astype(np.float64) > lo, np.nextafter(lo32, np.float32(-np.inf)), lo32)
    hi32 = np.where(hi32.astype(np.float64) < hi, np.nextafter(hi32, np.float32(np.inf)), hi32)
    return lo32, hi32


def counterexample_regions(result: Verification, seed: int, log=print) -> dict:
    """Counterexample regions of a refuted policy.

    When more than MAX_COUNTEREXAMPLE_OBJECTS violating zonotopes were captured, a uniform random subset
    of that size (drawn with ``seed``) is walked back. A row whose walk fails or whose result misses its
    start state region falls back to that whole start state region, which contains the true ancestor.

    Returns ``{"lo", "hi"}`` (float32 arrays of shape [n, 2]) plus ``n_rows`` (captured zonotopes used),
    ``n_dropped`` and ``n_piece_fallback``. Raises ``CounterexampleExtractionFailed`` if no region remains.
    """
    cap = result.capture
    lo_in = np.asarray(cap["lo"], dtype=np.float64)
    hi_in = np.asarray(cap["hi"], dtype=np.float64)
    origin = np.asarray(cap["origin"]).astype(np.int64)
    history = np.asarray(cap["history"])
    c_in = np.asarray(cap["c"], dtype=np.float64)
    G_in = np.asarray(cap["G"], dtype=np.float64)
    B = int(lo_in.shape[0])

    if B > MAX_COUNTEREXAMPLE_OBJECTS:
        rng = np.random.default_rng(seed)
        k = int(min(np.maximum(1, np.floor(B * (MAX_COUNTEREXAMPLE_OBJECTS / B))), B))
        keep = np.sort(rng.choice(np.arange(B), size=k, replace=False))
        log(f"{B:,} violating zonotopes > {MAX_COUNTEREXAMPLE_OBJECTS:,}: random subset of {keep.shape[0]:,} "
            f"(seed {seed})")
        lo_in, hi_in, origin, history = lo_in[keep], hi_in[keep], origin[keep], history[keep]
        c_in, G_in = c_in[keep], G_in[keep]
        B = int(lo_in.shape[0])

    use = (np.all(np.isfinite(lo_in), axis=1) & np.all(np.isfinite(hi_in), axis=1)
           & np.all(lo_in <= hi_in, axis=1)
           & np.all(np.isfinite(c_in), axis=1) & np.all(np.isfinite(G_in.reshape(B, -1)), axis=1))

    # Start the walk from the part of each zonotope inside the state box and at x < goal.
    state_lo = dynamics.STATE_LO.astype(np.float64)
    state_hi = dynamics.STATE_HI.astype(np.float64)
    box_lo = state_lo.copy()
    box_hi = state_hi.copy()
    gax = dynamics.GOAL_AXIS
    box_hi[gax] = min(box_hi[gax], float(np.nextafter(float(dynamics.GOAL_POSITION), -np.inf)))
    rows = np.flatnonzero(use)
    c0, G0, ok_start = _contract_rows_to_box(c_in[rows], G_in[rows], box_lo, box_hi)
    n_start_empty = int((~ok_start).sum())
    rows, c0, G0 = rows[ok_start], c0[ok_start], G0[ok_start]
    if rows.size == 0:
        raise CounterexampleExtractionFailed(f"all {B} violating zonotopes are empty inside the state box")

    # history[:, j] is the action of the transition from step j to step j + 1, so the newest comes last.
    actions = np.asarray(history[rows])[:, ::-1].astype(np.float64)
    walked = walk_back(c0, G0, actions)
    ok = np.asarray(walked["ok"], dtype=bool)
    lo = np.maximum(np.asarray(walked["lo"], dtype=np.float64), state_lo)
    hi = np.minimum(np.asarray(walked["hi"], dtype=np.float64), state_hi)
    ok &= np.all(np.isfinite(lo), axis=1) & np.all(np.isfinite(hi), axis=1)
    ok &= np.all(lo <= hi, axis=1)

    # The ancestor at step 0 is a start state of the row's own start state region.
    init_lo = dynamics.INIT_LO.astype(np.float64)
    init_hi = dynamics.INIT_HI.astype(np.float64)
    piece = origin[rows]
    lo = np.maximum(np.maximum(lo, init_lo[None, :]), result.piece_lo[piece])
    hi = np.minimum(np.minimum(hi, init_hi[None, :]), result.piece_hi[piece])
    empty = ok & np.any(lo > hi, axis=1)
    ok &= ~empty

    # Fallback: the whole start state region (a superset of the exact restart set).
    fallback = ~ok
    if fallback.any():
        piece_lo = np.maximum(result.piece_lo[piece], init_lo[None, :])
        piece_hi = np.minimum(result.piece_hi[piece], init_hi[None, :])
        fallback &= np.all(piece_lo <= piece_hi, axis=1)
        lo = np.where(fallback[:, None], piece_lo, lo)
        hi = np.where(fallback[:, None], piece_hi, hi)
        ok |= fallback
    n_fallback = int(fallback.sum())

    if not ok.any():
        raise CounterexampleExtractionFailed(f"all {B} violating zonotopes were dropped")
    lo32, hi32 = _outward_f32(lo[ok], hi[ok])
    # Identical regions carry no new states.
    _, unique = np.unique(np.concatenate([lo32, hi32], 1), axis=0, return_index=True)
    unique = np.sort(unique)
    n_duplicates = int(lo32.shape[0] - unique.shape[0])
    lo32, hi32 = lo32[unique], hi32[unique]
    n_dropped = int(B - ok.sum())
    log(f"{B:,} violating zonotopes -> {lo32.shape[0]:,} counterexample regions ({n_dropped} dropped, "
        f"{n_start_empty} empty in the state box, {int(empty.sum())} outside their start region, "
        f"{n_duplicates} duplicates removed, {n_fallback} start region fallbacks)")
    return {"lo": lo32, "hi": hi32, "n_rows": B, "n_dropped": n_dropped, "n_piece_fallback": n_fallback}
