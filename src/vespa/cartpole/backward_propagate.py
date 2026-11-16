"""Counterexample regions: predecessor boxes of the violating boxes.

The verifier captures every violating box of the first violating step together
with the actions its box took on the last steps (``segment_pool``). For each
violating box and each safety bound it crosses, the part beyond the bound (a
slab) is walked back ``n_lookback`` steps through the inverse dynamics, using the
box's recorded actions. The resulting predecessor boxes, each with the absolute
step it lives at, are the counterexample regions that fine-tuning starts
episodes from.

The inverse step linearises at the box (``env_specs.inverse_step``). It is not
sound and need not be: fine-tuning only samples start states from the regions.
"""

from __future__ import annotations

import numpy as np

from vespa.cartpole.env_specs import backward_walk_valid, inverse_step


class BackwardPropagationFailed(Exception):
    """No violating box produced a predecessor box."""


def _unsafe_slabs(box_lo, box_hi, halfspaces):
    """One (lo, hi) per crossed unsafe half-space: the part of the box beyond it."""
    out = []
    for ax, sign, thr in halfspaces:
        if sign > 0:
            if box_hi[ax] > thr:
                lo, hi = box_lo.copy(), box_hi.copy()
                lo[ax] = max(lo[ax], thr)
                if lo[ax] <= hi[ax]:
                    out.append((lo, hi))
        elif box_lo[ax] < thr:
            lo, hi = box_lo.copy(), box_hi.copy()
            hi[ax] = min(hi[ax], thr)
            if lo[ax] <= hi[ax]:
                out.append((lo, hi))
    return out


def _actions_newest_first(act_hist_row, write_idx, n_steps):
    """The last ``n_steps`` actions of a box, newest first.

    The kernel writes the action of step t into column t % N, so ``write_idx``
    (the violating step % N) is the column of the oldest action.
    """
    row = np.asarray(act_hist_row)
    N = row.shape[0]
    oldest_first = [int(row[(write_idx + j) % N]) for j in range(N)]
    return list(reversed(oldest_first))[:min(n_steps, N)]


def backward_propagate(capture: dict, spec, n_lookback: int, max_boxes: int, seed: int,
                       log=print) -> dict:
    """Predecessor boxes of the violating boxes in ``capture``.

    ``capture``: the arrays of ``violating_boxes.npz`` (box_lo, box_hi,
    box_origin, box_action_history, box_step). ``max_boxes``: above this many
    rows, a random subsample stratified by violating step is walked back.

    Returns {regions_lo, regions_hi (float32), regions_step (int32,
    1-indexed absolute step of each region), unsafe_step, n_failed_rows,
    n_unsafe_rows}; duplicate regions are removed.
    """
    halfspaces = spec.unsafe_halfspaces()
    lo_in = np.asarray(capture["box_lo"], dtype=np.float64)
    hi_in = np.asarray(capture["box_hi"], dtype=np.float64)
    orig_in = np.asarray(capture["box_origin"]).astype(np.int64)
    acth_in = np.asarray(capture["box_action_history"])
    step_in = np.asarray(capture["box_step"]).astype(np.int64)
    B = int(lo_in.shape[0])
    if B == 0:
        raise BackwardPropagationFailed("no violating boxes captured")
    N_avail = int(acth_in.shape[1])

    if max_boxes and B > max_boxes:
        rng = np.random.default_rng(seed)
        steps, inv, cnts = np.unique(step_in, return_inverse=True, return_counts=True)
        quota = np.maximum(1, np.floor(cnts * (max_boxes / B))).astype(np.int64)
        keep = []
        for si in range(steps.shape[0]):
            rows = np.flatnonzero(inv == si)
            keep.append(rng.choice(rows, size=int(min(quota[si], rows.shape[0])), replace=False))
        keep = np.sort(np.concatenate(keep))
        log(f"[backprop] per-box rows {B:,} > max_boxes {max_boxes:,}: "
            f"stratified subsample -> {keep.shape[0]:,} (seed {seed})")
        lo_in, hi_in, orig_in = lo_in[keep], hi_in[keep], orig_in[keep]
        acth_in, step_in = acth_in[keep], step_in[keep]
        B = int(lo_in.shape[0])

    out_lo, out_hi, out_step = [], [], []
    n_failed = 0
    n_slabs = 0
    for i in range(B):
        box_lo, box_hi = lo_in[i], hi_in[i]
        if (not np.all(np.isfinite(box_lo)) or not np.all(np.isfinite(box_hi))
                or np.any(box_lo > box_hi)):
            n_failed += 1
            continue
        fus = int(step_in[i])
        N = min(int(n_lookback), N_avail, fus)
        if N < 1:
            n_failed += 1
            continue
        slabs = _unsafe_slabs(box_lo, box_hi, halfspaces)
        if not slabs:
            n_failed += 1
            continue
        actions = _actions_newest_first(acth_in[i], fus % N_avail, N)
        ok_any = False
        for cur_lo, cur_hi in slabs:
            n_slabs += 1
            ok = True
            for step_back in range(N):
                if not backward_walk_valid(cur_lo, cur_hi):
                    ok = False
                    break
                cur_lo, cur_hi = inverse_step(cur_lo, cur_hi, actions[step_back])
                if (np.any(~np.isfinite(cur_lo)) or np.any(~np.isfinite(cur_hi))
                        or np.any(cur_lo > cur_hi)):
                    ok = False
                    break
            if not ok:
                continue
            ok_any = True
            out_lo.append(cur_lo)
            out_hi.append(cur_hi)
            out_step.append(max(fus - N, 0) + 1)
        if not ok_any:
            n_failed += 1

    if not out_lo:
        raise BackwardPropagationFailed(
            f"all {B} violating boxes were dropped (lookback {n_lookback})")
    lo_arr = np.stack(out_lo).astype(np.float32)
    hi_arr = np.stack(out_hi).astype(np.float32)
    step_arr = np.asarray(out_step, dtype=np.int32)
    rows = np.concatenate([lo_arr, hi_arr, step_arr[:, None].astype(np.float32)], 1)
    uniq = np.sort(np.unique(rows, axis=0, return_index=True)[1])
    n_dup = int(lo_arr.shape[0] - uniq.shape[0])
    lo_arr, hi_arr, step_arr = lo_arr[uniq], hi_arr[uniq], step_arr[uniq]
    log(f"[backprop] per-box: {B:,} unsafe rows -> {n_slabs:,} slabs -> "
        f"{lo_arr.shape[0]:,} predecessor boxes ({n_failed} rows dropped, "
        f"{n_dup} duplicates removed); steps {int(step_arr.min())}..{int(step_arr.max())}")
    return {
        "regions_lo": lo_arr,
        "regions_hi": hi_arr,
        "regions_step": step_arr,
        "unsafe_step": int(step_in.min()),
        "n_failed_rows": int(n_failed),
        "n_unsafe_rows": int(B),
    }
