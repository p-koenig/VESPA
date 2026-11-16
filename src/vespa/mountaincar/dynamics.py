"""Zonotope extension of the MountainCarContinuous step (stock gymnax dynamics).

A zonotope is the set Z = {c + G xi : xi in [-1, 1]^m} with centre c of shape [d] and generator matrix
G of shape [d, m]. The verifier carries one zonotope per reachable piece of the state set. The affine part
of a step is exact on zonotopes, and every nonlinear operation (cos(3x), the two clips) adds one outward
remainder generator, so each step over-approximates the true successor set.

Forward step of gymnax MountainCarContinuous-v0 with action a:

    v' = clip(v + clip(a, -1, 1) * 0.0015 - 0.0025 * cos(3x), -0.07, 0.07)
    x' = clip(x + v', -1.2, 0.6)

gymnax keeps v' at the left wall (no inelastic reset). States with x >= 0.45 have reached the goal.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

# Physical constants of gymnax MountainCarContinuous-v0.
POWER = 0.0015
GRAVITY = 0.0025
MAX_SPEED = 0.07
MIN_POSITION = -1.2
MAX_POSITION = 0.6
GOAL_POSITION = 0.45
MIN_ACTION = -1.0
MAX_ACTION = 1.0

STATE_DIM = 2
STATE_LO = np.array([MIN_POSITION, -MAX_SPEED])
STATE_HI = np.array([MAX_POSITION, MAX_SPEED])
INIT_LO = np.array([-0.6, 0.0])
INIT_HI = np.array([-0.4, 0.0])
GOAL_AXIS = 0

# Generator budget of a zonotope: 4d exact directions plus room for the step's remainders (4 * 2 + 16).
ORDER_LIMIT = 24
# Per-step float-rounding pad on axis i: EPS_SAFETY * EPS_ULP[i] * u (u = unit roundoff of float32).
# x: one addition plus the inherited velocity chain and outward rounding, about 4u.
# v (|v| <= 0.07): the chain g*cos, v + push, subtraction and outward rounding, about 0.5u.
EPS_ULP = (4.0, 0.5)
EPS_SAFETY = 4.0
UNIT_ROUNDOFF_FP32 = 2.0 ** -24

# Residual columns the step appends: cos(3x) (1), velocity clip (3), position clip (3) and one spare
# column that stays zero. The width is part of the computation (it enters the order reduction).
_RESIDUALS = 8
# Symbol-domain contraction: number of sweeps over the constrained axes.
_N_SWEEPS = 2
# Slack, in units of the unit roundoff, before the guaranteed-containment fallback of a contraction fires.
_CONTRACT_TOL_ULP = 64.0
_TWO_PI = 2.0 * math.pi


def step_eps() -> np.ndarray:
    """Per-axis outward rounding pad of one float32 step (float64 array of shape [2])."""
    return EPS_SAFETY * np.asarray(EPS_ULP, dtype=np.float64) * float(UNIT_ROUNDOFF_FP32)


# ---------------------------------------------------------------------------
# Interval bounds of cos and sin
# ---------------------------------------------------------------------------

def cos_bounds(a_lo, a_hi):
    """Bounds of cos over an arbitrary interval [a_lo, a_hi] (branchless, jit friendly).

    cos reaches +1 at the multiples of 2 pi and -1 at pi + 2 k pi. The lower endpoint is shifted into
    [0, 2 pi). If the interval still reaches an extremum it is a bound, otherwise the endpoints are.
    """
    width = a_hi - a_lo
    full = width >= _TWO_PI
    a = a_lo - jnp.floor(a_lo / _TWO_PI) * _TWO_PI
    b = a + width
    has_max = (a <= 0.0) | (b >= _TWO_PI)
    has_min = ((a <= math.pi) & (b >= math.pi)) | (b >= 3.0 * math.pi)
    c_a = jnp.cos(a_lo)
    c_b = jnp.cos(a_hi)
    lo = jnp.where(full | has_min, -1.0, jnp.minimum(c_a, c_b))
    hi = jnp.where(full | has_max, 1.0, jnp.maximum(c_a, c_b))
    return lo, hi


def sin_bounds(a_lo, a_hi):
    """Bounds of sin over an arbitrary interval, using sin(x) = cos(x - pi/2)."""
    return cos_bounds(a_lo - 0.5 * math.pi, a_hi - 0.5 * math.pi)


# ---------------------------------------------------------------------------
# Zonotope primitives. Every primitive is an outward over-approximation.
# ---------------------------------------------------------------------------

def zono_hull(c, G):
    """Interval hull of a zonotope: c -/+ sum |G| over the generators."""
    rad = jnp.sum(jnp.abs(G), axis=-1)
    return c - rad, c + rad


def zono_reduce_order(c, G, order_limit, pad=None):
    """Girard order reduction to a generator matrix of shape [d, order_limit].

    The order_limit - d generators with the largest L1 norm stay exact. The others are enclosed by d
    axis-aligned generators, whose radii also receive the per-axis rounding pad ``pad``.
    """
    d = int(G.shape[0])
    n_keep = int(order_limit) - d
    gnorm = jnp.sum(jnp.abs(G), axis=0)
    _, keep_idx = jax.lax.top_k(gnorm, n_keep)
    G_keep = G[:, keep_idx]
    box_rad = jnp.maximum(jnp.sum(jnp.abs(G), axis=1) - jnp.sum(jnp.abs(G_keep), axis=1), 0.0)
    if pad is not None:
        box_rad = box_rad + jnp.asarray(pad, dtype=G.dtype)
    G_box = jnp.zeros((d, d), dtype=G.dtype).at[jnp.arange(d), jnp.arange(d)].set(box_rad)
    return c, jnp.concatenate([G_keep, G_box], axis=1)


def ztrig_tight(c_th, g_th, f_c, fp_c, fpp_abs_max, fp_lo, fp_hi, ival_lo, ival_hi, col):
    """Tightest of three sound affine enclosures of a function f of one zonotope coordinate.

    ``fp_lo``/``fp_hi`` bound f' and ``ival_lo``/``ival_hi`` bound f over the coordinate's range.
    Tangent form: slope f'(c), remainder min(0.5 max|f''| r^2, max|f' - f'(c)| r). Min-range form:
    slope mid(f'), remainder 0.5 (f'_hi - f'_lo) r. Interval form: no linear term, taken only when the
    remainder alone exceeds the interval radius. The remainder goes into residual column ``col``.
    """
    r = jnp.sum(jnp.abs(g_th))
    rem_tan = jnp.minimum(
        0.5 * fpp_abs_max * r * r,
        jnp.maximum(jnp.abs(fp_lo - fp_c), jnp.abs(fp_hi - fp_c)) * r)
    rem_mr = 0.5 * (fp_hi - fp_lo) * r
    use_tan = rem_tan <= rem_mr
    slope = jnp.where(use_tan, fp_c, 0.5 * (fp_lo + fp_hi))
    rem = jnp.where(use_tan, rem_tan, rem_mr)
    ival_c = 0.5 * (ival_lo + ival_hi)
    ival_rad = 0.5 * (ival_hi - ival_lo)
    use_lin = rem <= ival_rad
    c_out = jnp.where(use_lin, f_c, ival_c)
    g_out = jnp.where(use_lin, slope * g_th, 0.0)
    return c_out, g_out.at[col].set(jnp.where(use_lin, rem, ival_rad))


def zrelu(cq, gq, col):
    """max(q, 0) as a sound zonotope (parallel ReLU relaxation).

    lo >= 0 gives the identity, hi <= 0 the constant 0. A straddling q gets slope hi / (hi - lo) with
    offset and remainder -slope * lo / 2, which keeps the (scaled) correlation of q.
    """
    r = jnp.sum(jnp.abs(gq))
    lo = cq - r
    hi = cq + r
    span = jnp.maximum(hi - lo, 1e-300)
    straddle = (lo < 0.0) & (hi > 0.0)
    lam = jnp.where(lo >= 0.0, 1.0, jnp.where(hi <= 0.0, 0.0, jnp.clip(hi / span, 0.0, 1.0)))
    mu = jnp.where(straddle, -0.5 * lam * lo, 0.0)
    return lam * cq + mu, (lam * gq).at[col].set(mu)


def zsat(cq, gq, lo, hi, col_lo, col_hi):
    """clip(q, lo, hi) as max(q, lo) = lo + relu(q - lo) followed by min(q, hi) = hi - relu(hi - q).

    Bounds may be infinite. An infinite side is skipped and leaves its residual column at zero.
    """
    fin = jnp.isfinite(lo)
    b = jnp.where(fin, lo, 0.0)
    c1, g1 = zrelu(cq - b, gq, col_lo)
    cq = jnp.where(fin, c1 + b, cq)
    gq = jnp.where(fin, g1, gq)
    fin = jnp.isfinite(hi)
    b = jnp.where(fin, hi, 0.0)
    c2, g2 = zrelu(b - cq, -gq, col_hi)
    cq = jnp.where(fin, b - c2, cq)
    gq = jnp.where(fin, -g2, gq)
    return cq, gq


def _contract_symbol_domains(a, b, c, G, axes, lo, hi):
    """Tighten the symbol domains xi_j in [a_j, b_j] against lo_i <= c_i + sum_j G_ij xi_j <= hi_i.

    Solving the constraint of axis i for one symbol, with the others at their current domains, removes
    only symbol values that cannot satisfy it, so every point of Z intersected with the box survives.
    """
    for _ in range(_N_SWEEPS):
        for i in axes:
            g = G[i]
            t_a = g * a
            t_b = g * b
            t_lo = jnp.minimum(t_a, t_b)
            t_hi = jnp.maximum(t_a, t_b)
            rest_lo = jnp.sum(t_lo) - t_lo
            rest_hi = jnp.sum(t_hi) - t_hi
            p_lo = lo[i] - c[i] - rest_hi
            p_hi = hi[i] - c[i] - rest_lo
            usable = jnp.abs(g) > 0.0
            gs = jnp.where(usable, g, 1.0)
            q_lo = p_lo / gs
            q_hi = p_hi / gs
            pos = g > 0.0
            xi_lo = jnp.where(usable, jnp.where(pos, q_lo, q_hi), -jnp.inf)
            xi_hi = jnp.where(usable, jnp.where(pos, q_hi, q_lo), jnp.inf)
            a = jnp.maximum(a, xi_lo)
            b = jnp.minimum(b, xi_hi)
    return a, b


def _contract_core(c, G, lo, hi, axes):
    """Symbol-domain contraction followed by the exact affine map back to xi in [-1, 1]^m.

    Returns ``(c, G, ok)``, where ``ok`` says whether the contracted domain is non-empty.
    """
    m = int(G.shape[1])
    a = jnp.full((m,), -1.0, dtype=G.dtype)
    b = jnp.full((m,), 1.0, dtype=G.dtype)
    a, b = _contract_symbol_domains(a, b, c, G, axes, lo, hi)
    a = jnp.clip(a, -1.0, 1.0)
    b = jnp.clip(b, -1.0, 1.0)
    ok = jnp.all(a <= b)
    mid = jnp.where(ok, 0.5 * (a + b), 0.0)
    half = jnp.where(ok, 0.5 * (b - a), 1.0)
    return c + G @ mid, G * half[None, :], ok


def _contract_tol(dtype) -> float:
    return _CONTRACT_TOL_ULP * float(np.finfo(np.dtype(dtype)).eps)


def _fallback_extent(c, G, lo, hi, axes, tol_u):
    """Per-axis (need, clipped centre, clipped radius, empty) of the guaranteed-containment fallback.

    The fallback zeroes an axis' generator row and replaces it by one box generator over the clipped
    extent. It fires only where the hull overshoots the box by more than the rounding tolerance.
    """
    d = int(G.shape[0])
    rad = jnp.sum(jnp.abs(G), axis=-1)
    h_lo = c - rad
    h_hi = c + rad
    tol = tol_u * (jnp.abs(c) + rad + 1.0)
    need = jnp.zeros((d,), dtype=jnp.bool_)
    for i in axes:
        need = need.at[i].set((h_lo[i] < lo[i] - tol[i]) | (h_hi[i] > hi[i] + tol[i]))
    c_lo = jnp.maximum(h_lo, lo)
    c_hi = jnp.minimum(h_hi, hi)
    empty = jnp.any(need & (c_lo > c_hi))
    return need, 0.5 * (c_lo + c_hi), 0.5 * jnp.maximum(c_hi - c_lo, 0.0), empty


def _restrict(c, G, lo, hi, axes, sat_cols):
    """Contract, saturate, contract again.

    Returns ``(c, G, ok)``. ``ok`` is the emptiness verdict of the plain intersection, taken before
    the saturation (a saturated set is never empty).
    """
    c, G, ok = _contract_core(c, G, lo, hi, axes)
    for k, i in enumerate(axes):
        ci, gi = zsat(c[i], G[i], lo[i], hi[i], sat_cols[k][0], sat_cols[k][1])
        c = c.at[i].set(ci)
        G = G.at[i].set(gi)
    c, G, _ = _contract_core(c, G, lo, hi, axes)
    return c, G, ok


def contract_to_box(c, G, lo, hi, order_limit):
    """Restrict a zonotope to the box [lo, hi] (set intersection, used for leaf routing).

    Three tiers, tightest first: symbol-domain contraction, saturation with a second contraction, and
    the guaranteed-containment axis box. Returns ``(c, G, ok)`` with G reduced to ``order_limit``
    columns and ``ok`` False when the intersection is empty.
    """
    d = int(G.shape[0])
    axes = tuple(range(d))
    tol_u = _contract_tol(G.dtype)
    m = int(G.shape[1])
    G = jnp.concatenate([G, jnp.zeros((d, 2 * d), dtype=G.dtype)], axis=1)
    sat_cols = [(m + 2 * k, m + 2 * k + 1) for k in range(d)]
    c, G, ok = _restrict(c, G, lo, hi, axes, sat_cols)
    need, mid, rad, empty = _fallback_extent(c, G, lo, hi, axes, tol_u)
    G_box = jnp.zeros((d, d), dtype=G.dtype).at[jnp.arange(d), jnp.arange(d)].set(jnp.where(need, rad, 0.0))
    G = jnp.concatenate([G * (~need).astype(G.dtype)[:, None], G_box], axis=1)
    c = jnp.where(need, mid, c)
    c, G = zono_reduce_order(c, G, order_limit)
    return c, G, ok & (~empty)


def _clip_value(c, G, lo, hi, axes, cols, tol_u):
    """Sound enclosure of the value clip q_i -> clip(q_i, lo_i, hi_i) on ``axes``.

    The saturation acts on all of Z, so states outside the box keep their other coordinate (the x of a
    state whose velocity saturates, the v of a state that hits the left wall). The image lies in the box
    on the clipped axes and is then contracted against it, with the axis-box fallback last.
    ``cols[3k:3k+3]`` are three zero columns (saturation low, saturation high, fallback) for ``axes[k]``.
    """
    for k, i in enumerate(axes):
        ci, gi = zsat(c[i], G[i], lo[i], hi[i], cols[3 * k], cols[3 * k + 1])
        c = c.at[i].set(ci)
        G = G.at[i].set(gi)
    c, G, _ = _contract_core(c, G, lo, hi, axes)
    need, mid, rad, _ = _fallback_extent(c, G, lo, hi, axes, tol_u)
    G = G * (~need).astype(G.dtype)[:, None]
    for k, i in enumerate(axes):
        G = G.at[i, cols[3 * k + 2]].set(jnp.where(need[i], rad[i], 0.0))
    c = jnp.where(need, mid, c)
    return c, G


# ---------------------------------------------------------------------------
# One step
# ---------------------------------------------------------------------------

def zonotope_step(c, G, action, eps):
    """Zonotope enclosure of one MountainCarContinuous step under a constant action.

    c: [2], G: [2, ORDER_LIMIT], action: scalar force, eps: per-axis rounding pad.
    cos(3x) is enclosed by ``ztrig_tight`` with the local bound of f' = -3 sin(3x). Both clips go
    through ``_clip_value``, which keeps the x-v correlation. Girard reduction brings the result back to
    ORDER_LIMIT generators and adds the rounding pad. Returns the successor ``(c', G')``.
    """
    OL = ORDER_LIMIT
    Gp = jnp.concatenate([G, jnp.zeros((2, _RESIDUALS), dtype=G.dtype)], axis=1)
    inf = jnp.asarray(jnp.inf, dtype=G.dtype)
    tol_u = _contract_tol(G.dtype)

    c_x, g_x = c[0], Gp[0]
    c_v, g_v = c[1], Gp[1]

    r_x = jnp.sum(jnp.abs(g_x))
    a3_lo, a3_hi = 3.0 * (c_x - r_x), 3.0 * (c_x + r_x)
    ci_lo, ci_hi = cos_bounds(a3_lo, a3_hi)
    si_lo, si_hi = sin_bounds(a3_lo, a3_hi)
    c_cos, g_cos = ztrig_tight(
        c_x, g_x, jnp.cos(3.0 * c_x), -3.0 * jnp.sin(3.0 * c_x), 9.0,
        -3.0 * si_hi, -3.0 * si_lo, ci_lo, ci_hi, OL + 0)

    push = jnp.clip(jnp.asarray(action, G.dtype), MIN_ACTION, MAX_ACTION) * POWER
    c_v1 = c_v + push - GRAVITY * c_cos
    g_v1 = g_v - GRAVITY * g_cos
    c_s, G_s = _clip_value(
        jnp.stack([c_x, c_v1]), jnp.stack([g_x, g_v1]),
        jnp.stack([-inf, jnp.asarray(-MAX_SPEED, G.dtype)]),
        jnp.stack([inf, jnp.asarray(MAX_SPEED, G.dtype)]),
        (1,), (OL + 1, OL + 2, OL + 3), tol_u)
    c_x, g_x, c_v1, g_v1 = c_s[0], G_s[0], c_s[1], G_s[1]

    c_xr = c_x + c_v1
    g_xr = g_x + g_v1
    c_s, G_s = _clip_value(
        jnp.stack([c_xr, c_v1]), jnp.stack([g_xr, g_v1]),
        jnp.stack([jnp.asarray(MIN_POSITION, G.dtype), -inf]),
        jnp.stack([jnp.asarray(MAX_POSITION, G.dtype), inf]),
        (0,), (OL + 4, OL + 5, OL + 6), tol_u)

    c_out = jnp.stack([c_s[0], c_s[1]])
    G_out = jnp.stack([G_s[0], G_s[1]])
    return zono_reduce_order(c_out, G_out, OL, pad=eps)
