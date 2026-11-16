"""Sound zonotope inverse of the MountainCarContinuous step (numpy, float64).

The counterexample extraction walks each violating zonotope at step 200 back in time with its recorded
actions. The result after k steps contains every state at step 200 - k whose trajectory under these
actions ends in the violating zonotope.

Forward step (gymnax): v1 = v + da - g cos(3x), v2 = clip(v1, -vmax, vmax), x' = clip(x + v2, xmin, xmax),
v' = v2, with da = clip(a, -1, 1) * power. Every predecessor of a point (x', v') falls into one branch:

  R  regular (no clip active): x = x' - v', v = v' - da + g cos(3x). Two shears. The x-shear is linear
     and exact on the generators. cos(3x) is enclosed by an affine form plus one remainder (mean-value
     form with a monotone-piece bound on f' = -3 sin(3x), Taylor form with a local bound on |f''|, or
     the plain interval, whichever remainder is smallest). This keeps the x-v correlation.
  P  v1 >= vmax (v' = +vmax): x = x' - vmax and vmax - da + g cos(3x) <= v <= vmax. Enclosed by a box.
  N  v1 <= -vmax (v' = -vmax): x = x' + vmax and -vmax <= v <= -vmax - da + g cos(3x). Box.
  W  left wall (x' = xmin, v' < 0 kept): x in [xmin, xmin - v'] and v = v' - da + g cos(3x). Box.
  G  goal side: states with x >= goal have reached the goal and are never on a trajectory that is still
     outside the goal at step 200. The part of a set at x >= goal is cut from the hull, and a zonotope
     that lies entirely at x >= goal is dropped.

The boxes of P, N and W are joined to the zonotope Z of branch R through Z + K with
K = box hull({0} u (B - c_Z)), which contains conv(Z u B) and keeps Z's correlation.

Rounding: every step adds an outward pad to the axis-aligned generators, hulls are padded outward and all
trig bounds are padded, so no true state is lost to float64 rounding. The fixed-width Girard step encloses
the two least important generators and the two new axis-aligned ones by two axis-aligned generators.
"""
from __future__ import annotations

import math

import numpy as np

from vespa.mountaincar import dynamics

POWER = dynamics.POWER
GRAVITY = dynamics.GRAVITY
MAX_SPEED = dynamics.MAX_SPEED
MIN_POS = dynamics.MIN_POSITION
MAX_POS = dynamics.MAX_POSITION
GOAL = dynamics.GOAL_POSITION

_EPS = float(np.finfo(np.float64).eps)
_TWO_PI = 2.0 * math.pi
_TRIG_ABS_ERR = 4.0 * _EPS     # outward pad of np.cos / np.sin values

# Reason a zonotope was dropped during the walk.
REASON_OK, REASON_NONFINITE, REASON_EMPTY, REASON_GOAL = 0, 1, 2, 3


def cos_bounds(a_lo, a_hi):
    """Outward-padded bounds of cos over [a_lo, a_hi] (monotone pieces)."""
    a_lo = np.asarray(a_lo, np.float64)
    a_hi = np.asarray(a_hi, np.float64)
    width = a_hi - a_lo
    full = width >= _TWO_PI
    a = a_lo - np.floor(a_lo / _TWO_PI) * _TWO_PI
    b = a + width
    has_max = (a <= 0.0) | (b >= _TWO_PI)
    has_min = ((a <= math.pi) & (b >= math.pi)) | (b >= 3.0 * math.pi)
    c_a = np.cos(a_lo)
    c_b = np.cos(a_hi)
    lo = np.where(full | has_min, -1.0, np.minimum(c_a, c_b) - _TRIG_ABS_ERR)
    hi = np.where(full | has_max, 1.0, np.maximum(c_a, c_b) + _TRIG_ABS_ERR)
    # A missed extremum (argument rounding right at k pi) costs O(delta^2), far below the pad.
    return np.maximum(lo, -1.0), np.minimum(hi, 1.0)


def sin_bounds(a_lo, a_hi):
    """sin(x) = cos(x - pi/2)."""
    return cos_bounds(np.asarray(a_lo) - 0.5 * math.pi, np.asarray(a_hi) - 0.5 * math.pi)


def _pad_out(lo, hi, mag, k):
    p = k * _EPS * mag + 1e-300
    return lo - p, hi + p


def _hull(c, g):
    """Outward-padded interval hull of one zonotope coordinate: c (B,), g (B, m)."""
    r = np.sum(np.abs(g), axis=-1)
    m = g.shape[-1]
    lo, hi = _pad_out(c - r, c + r, np.abs(c) + r, m + 4.0)
    return lo, hi, r


def _inverse_rows(cx, cv, gx, gv, da, hull):
    """One inverse step of B zonotopes given as rows.

    ``hull`` is the padded hull of the step-(t+1) sets. Returns ``(cx, cv, gx, gv, col_x, col_v)``: the
    predecessor zonotope is (cx, cv) + [gx; gv] e + diag(col_x, col_v) e'.
    """
    B, m = gx.shape
    xlo1, xhi1, rx1, vlo1, vhi1, rv1 = hull

    # Regular branch, x-shear (linear, exact up to rounding).
    ncx = cx - cv
    ngx = gx - gv
    pad_x = (4.0 * _EPS) * (np.abs(cx) + np.abs(cv) + rx1 + rv1) + 1e-300
    r = np.sum(np.abs(ngx), axis=1) * (1.0 + (m + 2) * _EPS) + pad_x
    xl, xh = ncx - r, ncx + r
    xl, xh = _pad_out(xl, xh, np.abs(ncx) + r, 4.0)

    # Affine enclosure of cos(3x) over x in [xl, xh].
    a3l, a3h = 3.0 * xl, 3.0 * xh
    a3l, a3h = _pad_out(a3l, a3h, np.abs(a3l) + np.abs(a3h), 2.0)
    cl, ch = cos_bounds(a3l, a3h)
    sl, sh = sin_bounds(a3l, a3h)
    fc = np.cos(3.0 * ncx)
    fpc = -3.0 * np.sin(3.0 * ncx)
    fc_err = (3.0 * np.abs(ncx) + 2.0) * 2.0 * _EPS
    fp_lo = -3.0 * sh - 4.0 * _EPS
    fp_hi = -3.0 * sl + 4.0 * _EPS
    fpp_max = 9.0 * np.maximum(np.abs(cl), np.abs(ch)) * (1.0 + 4.0 * _EPS)
    # The mean-value form uses the computed slope fpc. The Taylor form also pays the slope error times r.
    rem_mv = np.maximum(np.abs(fp_lo - fpc), np.abs(fp_hi - fpc)) * r
    rem_tl = 0.5 * fpp_max * r * r + (3.0 * np.abs(ncx) + 2.0) * 6.0 * _EPS * r
    rem_tan = np.minimum(rem_mv, rem_tl)
    s_mr = 0.5 * (fp_lo + fp_hi)
    rem_mr = np.maximum(fp_hi - s_mr, s_mr - fp_lo) * r
    use_tan = rem_tan <= rem_mr
    slope = np.where(use_tan, fpc, s_mr)
    rem = np.where(use_tan, rem_tan, rem_mr) + fc_err
    ival_c = 0.5 * (cl + ch)
    ival_r = np.maximum(ch - ival_c, ival_c - cl)
    use_lin = rem <= ival_r
    f0 = np.where(use_lin, fc, ival_c)
    slope = np.where(use_lin, slope, 0.0)
    rem = np.where(use_lin, rem, ival_r) * (1.0 + 4.0 * _EPS)
    # Now cos(3x) is in f0 + slope (x - ncx) + [-rem, rem] for every x in [xl, xh].

    # Regular branch, v-shear: v = v' - da + g cos(3x).
    gs = GRAVITY * slope
    ncv = cv - da + GRAVITY * f0
    ngv = gv + gs[:, None] * ngx
    sum_ngv = np.sum(np.abs(ngv), axis=1)
    rem_v = (GRAVITY * rem
             + GRAVITY * np.abs(slope) * pad_x
             + (8.0 + m) * _EPS * (np.abs(cv) + np.abs(da) + GRAVITY * np.abs(f0)
                                   + np.abs(ncv) + rv1 + sum_ngv
                                   + GRAVITY * np.abs(slope) * r)
             + 1e-300)

    # Clip branches: boxes, joined through Z + K.
    big = 1e300
    bl_x = np.full(B, big)
    bh_x = np.full(B, -big)
    bl_v = np.full(B, big)
    bh_v = np.full(B, -big)

    def add_box(mask, lx, hx, lv, hv):
        nonlocal bl_x, bh_x, bl_v, bh_v
        lx = np.maximum(lx, MIN_POS)
        hx = np.minimum(hx, min(MAX_POS, GOAL))
        lv = np.maximum(lv, -MAX_SPEED)
        hv = np.minimum(hv, MAX_SPEED)
        mask = mask & (lx <= hx) & (lv <= hv)
        bl_x = np.where(mask, np.minimum(bl_x, lx), bl_x)
        bh_x = np.where(mask, np.maximum(bh_x, hx), bh_x)
        bl_v = np.where(mask, np.minimum(bl_v, lv), bl_v)
        bh_v = np.where(mask, np.maximum(bh_v, hv), bh_v)

    # P: v' = +vmax is possible.
    m_p = vhi1 >= MAX_SPEED
    if np.any(m_p):
        lx, hx = _pad_out(xlo1 - MAX_SPEED, xhi1 - MAX_SPEED, np.abs(xlo1) + np.abs(xhi1) + 1, 4.0)
        cbl, _ = cos_bounds(3.0 * lx - 8 * _EPS, 3.0 * hx + 8 * _EPS)
        lv = MAX_SPEED - da + GRAVITY * cbl - 8.0 * _EPS
        add_box(m_p, lx, hx, lv, np.full(B, MAX_SPEED))
    # N: v' = -vmax is possible.
    m_n = vlo1 <= -MAX_SPEED
    if np.any(m_n):
        lx, hx = _pad_out(xlo1 + MAX_SPEED, xhi1 + MAX_SPEED, np.abs(xlo1) + np.abs(xhi1) + 1, 4.0)
        _, cbh = cos_bounds(3.0 * lx - 8 * _EPS, 3.0 * hx + 8 * _EPS)
        hv = -MAX_SPEED - da + GRAVITY * cbh + 8.0 * _EPS
        add_box(m_n, lx, hx, np.full(B, -MAX_SPEED), hv)
    # W: x' = xmin with v' < 0 (gymnax keeps v'). Predecessors need x + v' <= xmin, so x lies in
    # [xmin, xmin - vlo1] and v = v' - da + g cos(3x) for v' in [vlo1, min(vhi1, 0)]. When v' can be the
    # saturated -vmax, every v in [-vmax, ...] may map onto it.
    m_w = (xlo1 <= MIN_POS) & (vlo1 < 0.0)
    if np.any(m_w):
        lx = np.full(B, MIN_POS)
        hx = MIN_POS - np.minimum(vlo1, 0.0) + 8.0 * _EPS
        cbl, cbh = cos_bounds(np.full(B, 3.0 * MIN_POS - 1e-14), 3.0 * hx + 1e-14)
        vh2 = np.minimum(vhi1, 0.0)
        lv = vlo1 - da + GRAVITY * cbl - 8.0 * _EPS
        lv = np.where(vlo1 <= -MAX_SPEED, -MAX_SPEED, lv)
        hv = vh2 - da + GRAVITY * cbh + 8.0 * _EPS
        add_box(m_w, lx, hx, lv, hv)

    any_box = bl_x <= bh_x
    # K = hull({0} u (box - c)): new centre c + mid(K), axis generators rad(K).
    kxl = np.where(any_box, np.minimum(0.0, bl_x - ncx), 0.0)
    kxh = np.where(any_box, np.maximum(0.0, bh_x - ncx), 0.0)
    kvl = np.where(any_box, np.minimum(0.0, bl_v - ncv), 0.0)
    kvh = np.where(any_box, np.maximum(0.0, bh_v - ncv), 0.0)
    kmx, krx = 0.5 * (kxl + kxh), 0.5 * (kxh - kxl)
    kmv, krv = 0.5 * (kvl + kvh), 0.5 * (kvh - kvl)
    jpad_x = np.where(any_box, 4.0 * _EPS * (np.abs(ncx) + np.abs(kxl) + np.abs(kxh) + 1.0), 0.0)
    jpad_v = np.where(any_box, 4.0 * _EPS * (np.abs(ncv) + np.abs(kvl) + np.abs(kvh) + 0.1), 0.0)
    ncx = ncx + kmx
    ncv = ncv + kmv
    col_x = krx + pad_x + jpad_x
    col_v = krv + rem_v + jpad_v
    return ncx, ncv, ngx, ngv, col_x, col_v


def _insert_reduce(gx, gv, col_x, col_v):
    """Fixed-width Girard step (in place).

    The two existing generators with the smallest min(|g_x|, |g_v|) and the two new axis-aligned
    generators (col_x, 0), (0, col_v) are enclosed by two axis-aligned generators stored in the freed
    slots. This equals Girard reduction from m + 2 to m columns.
    """
    B, m = gx.shape
    ar = np.arange(B)
    score = np.minimum(np.abs(gx), np.abs(gv))
    i1 = np.argmin(score, axis=1)
    score[ar, i1] = np.inf
    i2 = np.argmin(score, axis=1)
    bx = np.abs(gx[ar, i1]) + np.abs(gx[ar, i2]) + col_x
    bv = np.abs(gv[ar, i1]) + np.abs(gv[ar, i2]) + col_v
    bx = bx * (1.0 + 4.0 * _EPS) + 1e-300
    bv = bv * (1.0 + 4.0 * _EPS) + 1e-300
    gx[ar, i1] = bx
    gv[ar, i1] = 0.0
    gx[ar, i2] = 0.0
    gv[ar, i2] = bv
    return gx, gv


def _hull_and_finite(cx, cv, gx, gv):
    hull = _hull(cx, gx) + _hull(cv, gv)
    finite = np.isfinite(cx) & np.isfinite(cv) & np.all(np.isfinite(gx), 1) & np.all(np.isfinite(gv), 1)
    return hull, finite


def walk_back(c, G, actions_newest_first):
    """Walk B zonotopes back N steps with their recorded actions.

    c: (B, 2), G: (B, 2, m), actions_newest_first: (B, N) where column k is the action applied from
    step T - k - 1 to step T - k. Returns a dict with ``lo``/``hi`` (B, 2), the hull of the walked-back
    zonotope intersected with the state box and with x <= goal, ``ok`` (B,) and ``reason`` (B,). A
    zonotope is dropped (ok False, NaN bounds) when its set becomes non-finite, empty or lies entirely
    at x >= goal.
    """
    c = np.asarray(c, np.float64)
    G = np.asarray(G, np.float64)
    A = np.asarray(actions_newest_first)
    B, N = A.shape
    s_lo = dynamics.STATE_LO.astype(np.float64)
    s_hi = dynamics.STATE_HI.astype(np.float64)
    sx_hi = min(float(s_hi[0]), GOAL)

    cx = c[:, 0].copy()
    cv = c[:, 1].copy()
    gx = np.ascontiguousarray(np.array(G[:, 0]))
    gv = np.ascontiguousarray(np.array(G[:, 1]))
    reason = np.zeros(B, np.int8)
    idx = np.arange(B)                    # input row of each live row
    live = np.ones(B, bool)

    def check(hull, finite):
        xlo, xhi, _, vlo, vhi, _ = hull
        lo = np.stack([np.maximum(xlo, s_lo[0]), np.maximum(vlo, s_lo[1])], -1)
        hi = np.stack([np.minimum(xhi, sx_hi), np.minimum(vhi, s_hi[1])], -1)
        goal_only = xlo >= GOAL
        empty = np.any(lo > hi, axis=-1)
        new = np.where(~finite, REASON_NONFINITE,
                       np.where(goal_only, REASON_GOAL, np.where(empty, REASON_EMPTY, REASON_OK))).astype(np.int8)
        return lo, hi, new

    def record(new):
        nonlocal live
        died = live & (new != 0)
        if np.any(died):
            reason[idx[died]] = new[died]
            live = live & ~died

    hull, finite = _hull_and_finite(cx, cv, gx, gv)
    lo, hi, new = check(hull, finite)
    record(new)
    da_all = np.clip(np.asarray(A, dtype=np.float64), -1.0, 1.0) * POWER
    for k in range(N):
        n_dead = int((~live).sum())
        if n_dead > 0.05 * len(live):     # compact: drop dead rows
            cx, cv, gx, gv = cx[live], cv[live], gx[live], gv[live]
            hull = tuple(h[live] for h in hull)
            idx = idx[live]
            live = np.ones(len(idx), bool)
        elif n_dead:                      # keep dead rows finite and inert
            d = ~live
            cx[d] = -0.5
            cv[d] = 0.0
            gx[d] = 0.0
            gv[d] = 0.0
            hull = tuple(np.where(d, v, h) for h, v in zip(hull, (-0.5, -0.5, 0.0, 0.0, 0.0, 0.0)))
        if len(idx) == 0:
            continue
        cx, cv, gx, gv, col_x, col_v = _inverse_rows(cx, cv, gx, gv, da_all[idx, k], hull)
        gx, gv = _insert_reduce(gx, gv, col_x, col_v)
        hull, finite = _hull_and_finite(cx, cv, gx, gv)
        lo, hi, new = check(hull, finite)
        record(new)

    out_lo = np.full((B, 2), np.nan)
    out_hi = np.full((B, 2), np.nan)
    rows = idx[live]
    out_lo[rows] = lo[live]
    out_hi[rows] = hi[live]
    return {"lo": out_lo, "hi": out_hi, "ok": reason == 0, "reason": reason}
