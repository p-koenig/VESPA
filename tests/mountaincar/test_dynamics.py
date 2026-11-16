"""Dynamics tests: the gymnasium front-end and the zonotope step against stock gymnax.

* the gymnasium environment reproduces gymnax ``step_env`` bitwise, including the left wall;
* every gymnax successor of a point of a zonotope lies in the hull of ``dynamics.zonotope_step``.

Run on CPU from the repository root:  JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest tests/mountaincar
(or run this file directly).
"""
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from gymnax.environments.classic_control.continuous_mountain_car import ContinuousMountainCar, EnvState

from vespa.mountaincar import dynamics, environment
from vespa.mountaincar.config import ENV_ID

GX = ContinuousMountainCar()
GP = GX.default_params
_GX_STEP = jax.jit(jax.vmap(lambda x, v, t, a: GX.step_env(
    jax.random.PRNGKey(0), EnvState(position=x, velocity=v, time=t), a, GP)))


def gymnax_step(x, v, a):
    """Batched gymnax step in float32: returns (x', v', reward)."""
    _, st, r, _, _ = _GX_STEP(jnp.asarray(x, jnp.float32), jnp.asarray(v, jnp.float32),
                              jnp.zeros(len(x), jnp.int32), jnp.asarray(a, jnp.float32))
    return np.asarray(st.position), np.asarray(st.velocity), np.asarray(r)


def test_environment_equals_gymnax():
    environment.register()
    env = gym.make(ENV_ID)
    assert env.spec.max_episode_steps == 999
    rng = np.random.default_rng(1)
    n_mismatch = n_wall = 0
    for episode in range(30):
        obs, _ = env.reset(seed=episode)
        mode = episode % 3          # random, hard-left bursts, energy pumping
        for t in range(999):
            if mode == 0:
                a = rng.uniform(-2.0, 2.0, size=(1,)).astype(np.float32)
            elif mode == 1:
                a = np.array([-1.5 if (t // 60) % 2 == 0 else 1.2], np.float32)
            else:
                a = np.array([1.0 if obs[1] > 0 else -1.0], np.float32)
            x0, v0 = np.float32(obs[0]), np.float32(obs[1])
            obs, r, terminated, truncated, _ = env.step(a)
            px, pv, pr = gymnax_step([x0], [v0], a[None])
            goal = bool(px[0] >= 0.45 and pv[0] >= 0.0)
            same = obs[0] == px[0] and obs[1] == pv[0] and np.float32(r) == np.float32(pr[0]) and terminated == goal
            n_mismatch += 0 if same else 1
            n_wall += int(px[0] <= -1.2 and pv[0] < 0)
            if terminated or truncated:
                break
    assert n_mismatch == 0
    assert n_wall > 0


def _zonotope_cases(rng, n):
    """Random zonotopes in the regions where the clips act (wall, velocity limits) and in general position."""
    out = []
    for lo, hi, scale in (([-1.2, -0.07], [-1.05, 0.0], [0.01, 0.002]), ([-1.2, -0.07], [0.44, 0.07], [0.02, 0.003]),
                          ([-0.6, 0.060], [0.3, 0.07], [0.01, 0.003]), ([-0.9, -0.07], [-0.2, -0.060], [0.01, 0.003])):
        for _ in range(n):
            G = rng.standard_normal((2, dynamics.ORDER_LIMIT)) * np.asarray(scale)[:, None]
            out.append((rng.uniform(lo, hi), G, float(rng.uniform(-2.0, 2.0))))
    for kind in ("wall", "vhi", "vlo"):
        for _ in range(n):
            G = rng.standard_normal((2, dynamics.ORDER_LIMIT)) * np.array([[2e-4], [4e-5]])
            G[:, 0] = [rng.uniform(0.005, 0.03), rng.uniform(0.001, 0.006) * rng.choice([-1, 1])]
            if kind == "wall":
                c, a = np.array([rng.uniform(-1.19, -1.15), rng.uniform(-0.04, -0.005)]), float(rng.uniform(-2.0, 0.3))
            elif kind == "vhi":
                c, a = np.array([rng.uniform(-0.6, 0.2), rng.uniform(0.062, 0.069)]), float(rng.uniform(0.0, 2.0))
            else:
                c, a = np.array([rng.uniform(-0.9, -0.3), rng.uniform(-0.069, -0.062)]), float(rng.uniform(-2.0, 0.0))
            out.append((c, G, a))
    return out


def test_zonotope_step_contains_gymnax():
    rng = np.random.default_rng(3)
    eps = jnp.asarray(dynamics.step_eps(), jnp.float32)
    step = jax.jit(lambda c, G, a: dynamics.zonotope_step(c, G, a, eps))
    escapes = 0
    for c, G, a in _zonotope_cases(rng, 60):
        c2, G2 = step(jnp.asarray(c, jnp.float32), jnp.asarray(G, jnp.float32), jnp.float32(a))
        c2 = np.asarray(c2, np.float64)
        rad = np.abs(np.asarray(G2, np.float64)).sum(1)
        xi = np.concatenate([rng.uniform(-1, 1, (64, dynamics.ORDER_LIMIT)),
                             np.sign(rng.uniform(-1, 1, (64, dynamics.ORDER_LIMIT)))])
        points = np.asarray(c, np.float32)[None] + xi.astype(np.float32) @ np.asarray(G, np.float32).T
        qx, qv, _ = gymnax_step(points[:, 0], points[:, 1], np.full((len(points), 1), a, np.float32))
        q = np.stack([qx, qv], -1).astype(np.float64)
        q = q[q[:, 0] < dynamics.GOAL_POSITION]          # successors in the goal leave the frontier
        escapes += int((np.maximum(c2 - rad - q, q - c2 - rad).max(1) > 0).sum()) if q.size else 0
    assert escapes == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name, flush=True)
