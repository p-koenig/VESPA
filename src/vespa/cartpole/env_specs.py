"""CartPole dynamics for the verifier, the counterexample regions and fine-tuning.

The verified system is gymnasium's CartPole with the Euler integrator (tau = 0.02 s,
force 10 N). A state is (x, x_dot, theta, theta_dot). The policy acts on the state
itself (the observation is the state).

This module provides
  * ``interval_step``: a sound interval extension of one CartPole step, used by the
    verifier. For every state in [lo, hi] the successor lies in the returned box;
    an outward pad of a few float ULPs per axis covers the rounding of the step;
  * ``inverse_step``: an approximate inverse of one step on boxes (numpy), used to
    walk violating boxes back to counterexample regions. It only shapes the start
    states of fine-tuning, so it need not be sound;
  * the safety box |x| <= x_threshold, |theta| <= theta_threshold and the start
    set [-0.05, 0.05]^4, bundled in a ``CartPoleSpec``.
"""

from __future__ import annotations

# The interval arithmetic needs float64 constants: enable x64 before any JAX use.
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import math  # noqa: E402
from dataclasses import dataclass  # noqa: E402

ENV_ID = "VerifyCartPole-v0"
STATE_DIM = 4
N_ACTIONS = 2

# Physical constants of gymnasium's CartPoleEnv.
_G = 9.8          # gravitational acceleration (m/s^2)
_MP = 0.1         # mass of the pole (kg)
_MT = 1.1         # total mass of cart and pole (kg)
_L = 0.5          # half pole length (m)
_PL = 0.05        # pole mass times half length
_FM = 10.0        # force magnitude (N)
_TAU = 0.02       # time step (s)
# Shifted-sine identity: G*MT*sin(theta) - force*cos(theta) = R * sin(theta - phi_a).
_R = math.sqrt((_G * _MT) ** 2 + _FM ** 2)
_PHI = math.atan2(_FM, _G * _MT)

# Outward rounding pad per axis in units of the storage dtype's unit roundoff u.
# It bounds the accumulated float rounding of one step on that axis in the region
# that matters for verification (|x| <= 2.5, |x_dot| <= 4, |theta| <= 0.26,
# |theta_dot| <= 4): x about 8u, x_dot and theta_dot about 16u, theta about 2u.
EPS_ULP = (8.0, 16.0, 2.0, 16.0)

# The backward walk is only meaningful while |theta| < pi/4 (monotone sine).
_THETA_PRECONDITION = math.pi / 4.0


# --------------------------------------------------------------------------- #
# Interval primitives (JAX)                                                   #
# --------------------------------------------------------------------------- #

def _imul(a_lo, a_hi, b_lo, b_hi):
    """Interval product [a] * [b]: min and max over the four corner products."""
    p = jnp.stack([a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi])
    return jnp.min(p, axis=0), jnp.max(p, axis=0)


def _idiv(a_lo, a_hi, c_lo, c_hi):
    """Interval quotient [a] / [c] for a strictly positive [c]."""
    return _imul(a_lo, a_hi, 1.0 / c_hi, 1.0 / c_lo)


def _isquare(lo, hi):
    """Interval square [lo, hi]^2."""
    sq_lo = lo * lo
    sq_hi = hi * hi
    has_zero = (lo <= 0.0) & (hi >= 0.0)
    return (
        jnp.where(has_zero, 0.0, jnp.minimum(sq_lo, sq_hi)),
        jnp.maximum(sq_lo, sq_hi),
    )


def _sin_cos_bounds(theta_lo, theta_hi):
    """Bounds of sin and cos over [theta_lo, theta_hi] within (-pi/2, pi/2)."""
    sin_lo = jnp.sin(theta_lo)
    sin_hi = jnp.sin(theta_hi)
    abs_max = jnp.maximum(jnp.abs(theta_lo), jnp.abs(theta_hi))
    abs_min = jnp.minimum(jnp.abs(theta_lo), jnp.abs(theta_hi))
    cos_lo = jnp.cos(abs_max)
    cos_hi = jnp.where((theta_lo <= 0.0) & (theta_hi >= 0.0), 1.0, jnp.cos(abs_min))
    return sin_lo, sin_hi, cos_lo, cos_hi


def _sin_cos_product_bounds(theta_lo, theta_hi):
    """Bounds of sin(theta) * cos(theta) = sin(2 theta) / 2 for |theta| < pi/4."""
    a = 0.5 * jnp.sin(2.0 * theta_lo)
    b = 0.5 * jnp.sin(2.0 * theta_hi)
    return jnp.minimum(a, b), jnp.maximum(a, b)


def _cos_sq_bounds(theta_lo, theta_hi):
    """Bounds of cos^2(theta) = (1 + cos(2 theta)) / 2."""
    abs_max = jnp.maximum(jnp.abs(theta_lo), jnp.abs(theta_hi))
    abs_min = jnp.minimum(jnp.abs(theta_lo), jnp.abs(theta_hi))
    c2_min = jnp.cos(2.0 * abs_max)
    c2_max = jnp.where((theta_lo <= 0.0) & (theta_hi >= 0.0), 1.0, jnp.cos(2.0 * abs_min))
    return 0.5 * (1.0 + c2_min), 0.5 * (1.0 + c2_max)


# --------------------------------------------------------------------------- #
# Numpy versions of the primitives (host-side backward walk)                  #
# --------------------------------------------------------------------------- #

def _np_imul(a_lo, a_hi, b_lo, b_hi):
    p = np.stack([a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi])
    return p.min(axis=0), p.max(axis=0)


def _np_idiv(a_lo, a_hi, c_lo, c_hi):
    return _np_imul(a_lo, a_hi, 1.0 / c_hi, 1.0 / c_lo)


def _np_isquare(lo, hi):
    sq_lo = lo * lo
    sq_hi = hi * hi
    has_zero = (lo <= 0.0) & (hi >= 0.0)
    return (
        np.where(has_zero, 0.0, np.minimum(sq_lo, sq_hi)),
        np.maximum(sq_lo, sq_hi),
    )


def _np_sin_cos_bounds(theta_lo, theta_hi):
    sin_lo = np.sin(theta_lo)
    sin_hi = np.sin(theta_hi)
    abs_max = np.maximum(np.abs(theta_lo), np.abs(theta_hi))
    abs_min = np.minimum(np.abs(theta_lo), np.abs(theta_hi))
    cos_lo = np.cos(abs_max)
    cos_hi = np.where((theta_lo <= 0.0) & (theta_hi >= 0.0), 1.0, np.cos(abs_min))
    return sin_lo, sin_hi, cos_lo, cos_hi


def _np_sin_cos_product_bounds(theta_lo, theta_hi):
    a = 0.5 * np.sin(2.0 * theta_lo)
    b = 0.5 * np.sin(2.0 * theta_hi)
    return np.minimum(a, b), np.maximum(a, b)


def _np_cos_sq_bounds(theta_lo, theta_hi):
    abs_max = np.maximum(np.abs(theta_lo), np.abs(theta_hi))
    abs_min = np.minimum(np.abs(theta_lo), np.abs(theta_hi))
    c2_min = np.cos(2.0 * abs_max)
    c2_max = np.where((theta_lo <= 0.0) & (theta_hi >= 0.0), 1.0, np.cos(2.0 * abs_min))
    return 0.5 * (1.0 + c2_min), 0.5 * (1.0 + c2_max)


# --------------------------------------------------------------------------- #
# Dynamics                                                                    #
# --------------------------------------------------------------------------- #

def interval_step(lo, hi, action, eps):
    """Interval extension of one CartPole Euler step.

    ``lo``, ``hi``: (4,) box bounds; ``action``: 0 (push left) or 1 (push right);
    ``eps``: (4,) outward pad added at the end. Uses the closed form
        D        = 4 MT / 3 - MP cos^2(theta)
        thetaacc = (R sin(theta - phi_a) - PL theta_dot^2 sin(2 theta) / 2) / (L D)
        xacc     = (4/3 force + 4/3 PL theta_dot^2 sin(theta) - MP G sin(2 theta) / 2) / D
    which equals gymnasium's chained formulas and has tighter interval bounds.
    The order of the operations fixes the floating-point result.
    """
    force = jnp.where(action == 1, _FM, -_FM)

    sin_lo, sin_hi, _cos_lo, _cos_hi = _sin_cos_bounds(lo[2], hi[2])
    cos2_lo, cos2_hi = _cos_sq_bounds(lo[2], hi[2])
    sc_lo, sc_hi = _sin_cos_product_bounds(lo[2], hi[2])
    td2_lo, td2_hi = _isquare(lo[3], hi[3])

    D_lo = 4.0 * _MT / 3.0 - _MP * cos2_hi
    D_hi = 4.0 * _MT / 3.0 - _MP * cos2_lo

    phi = jnp.where(action == 1, _PHI, -_PHI)
    Rs_lo, Rs_hi = _R * jnp.sin(lo[2] - phi), _R * jnp.sin(hi[2] - phi)
    td2sc_lo, td2sc_hi = _imul(td2_lo, td2_hi, sc_lo, sc_hi)
    N_th_lo = Rs_lo - _PL * td2sc_hi
    N_th_hi = Rs_hi - _PL * td2sc_lo

    LD_lo = _L * D_lo
    LD_hi = _L * D_hi
    thetaacc_lo, thetaacc_hi = _idiv(N_th_lo, N_th_hi, LD_lo, LD_hi)

    td2sin_lo, td2sin_hi = _imul(td2_lo, td2_hi, sin_lo, sin_hi)
    M_x_lo = (4.0 / 3.0) * force + (4.0 / 3.0) * _PL * td2sin_lo - _MP * _G * sc_hi
    M_x_hi = (4.0 / 3.0) * force + (4.0 / 3.0) * _PL * td2sin_hi - _MP * _G * sc_lo
    xacc_lo, xacc_hi = _idiv(M_x_lo, M_x_hi, D_lo, D_hi)

    x_lo = lo[0] + _TAU * lo[1]
    x_hi = hi[0] + _TAU * hi[1]
    xd_lo = lo[1] + _TAU * xacc_lo
    xd_hi = hi[1] + _TAU * xacc_hi
    th_lo = lo[2] + _TAU * lo[3]
    th_hi = hi[2] + _TAU * hi[3]
    thd_lo = lo[3] + _TAU * thetaacc_lo
    thd_hi = hi[3] + _TAU * thetaacc_hi

    next_lo = jnp.stack([x_lo, xd_lo, th_lo, thd_lo])
    next_hi = jnp.stack([x_hi, xd_hi, th_hi, thd_hi])
    eps = jnp.asarray(eps, dtype=next_lo.dtype)
    return next_lo - eps, next_hi + eps


def _interval_accelerations(lo, hi, action):
    """(xacc_lo, xacc_hi, thetaacc_lo, thetaacc_hi) over one box (numpy)."""
    force = _FM if action == 1 else -_FM

    sin_lo, sin_hi, _cl, _ch = _np_sin_cos_bounds(lo[2], hi[2])
    cos2_lo, cos2_hi = _np_cos_sq_bounds(lo[2], hi[2])
    sc_lo, sc_hi = _np_sin_cos_product_bounds(lo[2], hi[2])
    td2_lo, td2_hi = _np_isquare(lo[3], hi[3])

    D_lo = 4.0 * _MT / 3.0 - _MP * cos2_hi
    D_hi = 4.0 * _MT / 3.0 - _MP * cos2_lo

    phi = _PHI if action == 1 else -_PHI
    Rs_lo = _R * np.sin(lo[2] - phi)
    Rs_hi = _R * np.sin(hi[2] - phi)
    td2sc_lo, td2sc_hi = _np_imul(np.asarray(td2_lo), np.asarray(td2_hi),
                                  np.asarray(sc_lo), np.asarray(sc_hi))
    N_th_lo = Rs_lo - _PL * td2sc_hi
    N_th_hi = Rs_hi - _PL * td2sc_lo

    LD_lo = _L * D_lo
    LD_hi = _L * D_hi
    thetaacc_lo, thetaacc_hi = _np_idiv(np.asarray(N_th_lo), np.asarray(N_th_hi),
                                        np.asarray(LD_lo), np.asarray(LD_hi))

    td2sin_lo, td2sin_hi = _np_imul(np.asarray(td2_lo), np.asarray(td2_hi),
                                    np.asarray(sin_lo), np.asarray(sin_hi))
    M_x_lo = (4.0 / 3.0) * force + (4.0 / 3.0) * _PL * td2sin_lo - _MP * _G * sc_hi
    M_x_hi = (4.0 / 3.0) * force + (4.0 / 3.0) * _PL * td2sin_hi - _MP * _G * sc_lo
    xacc_lo, xacc_hi = _np_idiv(np.asarray(M_x_lo), np.asarray(M_x_hi),
                                np.asarray(D_lo), np.asarray(D_hi))
    return float(xacc_lo), float(xacc_hi), float(thetaacc_lo), float(thetaacc_hi)


def inverse_step(next_lo, next_hi, action):
    """Approximate predecessor box of ``[next_lo, next_hi]`` under ``action``.

    The accelerations are bounded at the successor box (a close approximation of
    their range at the predecessor for the small time step), then every Euler
    line ``next = prev + tau * derivative`` is solved for ``prev``.
    """
    next_lo = np.asarray(next_lo)
    next_hi = np.asarray(next_hi)
    xacc_lo, xacc_hi, thetaacc_lo, thetaacc_hi = _interval_accelerations(
        next_lo, next_hi, action)
    prev_x_dot_lo = next_lo[1] - _TAU * xacc_hi
    prev_x_dot_hi = next_hi[1] - _TAU * xacc_lo
    prev_theta_dot_lo = next_lo[3] - _TAU * thetaacc_hi
    prev_theta_dot_hi = next_hi[3] - _TAU * thetaacc_lo
    prev_x_lo = next_lo[0] - _TAU * prev_x_dot_hi
    prev_x_hi = next_hi[0] - _TAU * prev_x_dot_lo
    prev_theta_lo = next_lo[2] - _TAU * prev_theta_dot_hi
    prev_theta_hi = next_hi[2] - _TAU * prev_theta_dot_lo
    prev_lo = np.array([prev_x_lo, prev_x_dot_lo, prev_theta_lo, prev_theta_dot_lo],
                       dtype=next_lo.dtype)
    prev_hi = np.array([prev_x_hi, prev_x_dot_hi, prev_theta_hi, prev_theta_dot_hi],
                       dtype=next_hi.dtype)
    return prev_lo, prev_hi


def backward_walk_valid(lo, hi) -> bool:
    """True while the box stays in |theta| < pi/4, where ``inverse_step`` applies."""
    return bool(abs(float(lo[2])) < _THETA_PRECONDITION
                and abs(float(hi[2])) < _THETA_PRECONDITION)


def set_env_state(env, state):
    """Place a gym CartPole env in ``state`` (the gym state is the verified state)."""
    env.unwrapped.state = np.asarray(state, dtype=np.float64)


def obs_from_state(state):
    """Observation of a state (identity, as float32 like gym's observations)."""
    return np.asarray(state, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Safety box and start set                                                    #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CartPoleSpec:
    """Safety box and start set of one verification setting.

    ``safe_lo``/``safe_hi``: (4,) float64, +-inf on the unbounded velocity axes.
    ``init_lo``/``init_hi``: (4,) float64 start set.
    """
    x_threshold: float
    theta_threshold: float
    safe_lo: np.ndarray
    safe_hi: np.ndarray
    init_lo: np.ndarray
    init_hi: np.ndarray

    def unsafe_obs(self, obs) -> bool:
        """True if an observed state violates |x| <= x_thr and |theta| <= theta_thr."""
        return bool(abs(float(obs[0])) > self.x_threshold
                    or abs(float(obs[2])) > self.theta_threshold)

    def unsafe_halfspaces(self) -> list:
        """The unsafe set as half-spaces ``(axis, sign, threshold)``; sign +1 means
        ``state[axis] > threshold`` is unsafe, sign -1 means ``state[axis] < threshold``."""
        out = []
        for ax in range(STATE_DIM):
            if np.isfinite(self.safe_hi[ax]):
                out.append((ax, +1, float(self.safe_hi[ax])))
            if np.isfinite(self.safe_lo[ax]):
                out.append((ax, -1, float(self.safe_lo[ax])))
        return out

    def axis_ranges(self, velocity_range: float):
        """Per-axis ranges for the Morton codes of the box reduction: the safety
        box, with +-``velocity_range`` on the unbounded velocity axes."""
        lo = np.where(np.isfinite(self.safe_lo), self.safe_lo, -float(velocity_range))
        hi = np.where(np.isfinite(self.safe_hi), self.safe_hi, float(velocity_range))
        return lo, hi


def make_spec(x_threshold: float = 2.4, theta_threshold: float = 0.20943951,
              init_halfwidth: float = 0.05) -> CartPoleSpec:
    """Safety box |x| <= x_threshold, |theta| <= theta_threshold and start set
    [-init_halfwidth, init_halfwidth]^4."""
    x_threshold, theta_threshold = float(x_threshold), float(theta_threshold)
    return CartPoleSpec(
        x_threshold=x_threshold,
        theta_threshold=theta_threshold,
        safe_lo=np.array([-x_threshold, -np.inf, -theta_threshold, -np.inf]),
        safe_hi=np.array([x_threshold, np.inf, theta_threshold, np.inf]),
        init_lo=np.full(4, -float(init_halfwidth)),
        init_hi=np.full(4, float(init_halfwidth)),
    )
