"""Tests of the zonotope inverse step (``walk_back``).

Trajectories of the float64 gymnax step that end outside the goal are walked back with their actions. The
walked-back hull must contain the true start state, also for trajectories that hit the left wall.

Run on CPU from the repository root:  JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest tests/mountaincar
(or run this file directly).
"""
import numpy as np

from vespa.mountaincar import dynamics
from vespa.mountaincar.walk_back import walk_back

OL = dynamics.ORDER_LIMIT


def step64(x, v, a):
    """gymnax MountainCarContinuous step in float64 (no left-wall reset, goal-side reset of negative v)."""
    v = np.clip(v + np.clip(a, -1.0, 1.0) * dynamics.POWER - dynamics.GRAVITY * np.cos(3.0 * x),
                -dynamics.MAX_SPEED, dynamics.MAX_SPEED)
    x = np.clip(x + v, dynamics.MIN_POSITION, dynamics.MAX_POSITION)
    v = v * (1 - (x >= dynamics.GOAL_POSITION) * (v < 0))
    return x, v


def tiny_zonotopes(centres, radius=1e-7):
    G = np.zeros((len(centres), 2, OL))
    G[:, 0, 0] = radius
    G[:, 1, 1] = radius
    return np.asarray(centres, np.float64), G


def test_single_step_wall_branch():
    rng = np.random.default_rng(5)
    x0 = rng.uniform(-1.2, -1.13, 20000)
    v0 = rng.uniform(-0.07, 0.0, 20000)
    a = rng.uniform(-1.0, 1.0, 20000)
    x1, v1 = step64(x0, v0, a)
    clipped = (x1 <= -1.2) & (v1 < 0)
    assert clipped.sum() > 100
    c, G = tiny_zonotopes(np.stack([x1, v1], -1)[clipped])
    out = walk_back(c, G, a[clipped][:, None])
    start = np.stack([x0, v0], -1)[clipped]
    assert out["ok"].all()
    assert np.all((out["lo"] <= start) & (start <= out["hi"]))


def test_walk_back_contains_start():
    rng = np.random.default_rng(6)
    n, T = 400, 60
    x0 = rng.uniform(-0.6, -0.4, n)
    x, v = x0.copy(), np.zeros(n)
    actions, hit_wall = [], np.zeros(n, bool)
    for t in range(T):
        a = np.where((np.arange(n) % 2 == 0) | (t < 30), -1.0, 0.7)
        actions.append(a)
        x, v = step64(x, v, a)
        hit_wall |= (x <= -1.2) & (v < 0)
    c, G = tiny_zonotopes(np.stack([x, v], -1))
    out = walk_back(c, G, np.stack(actions[::-1], 1))
    ok = out["ok"]
    start = np.stack([x0, np.zeros(n)], -1)
    excess = np.maximum(out["lo"] - start, start - out["hi"]).max(1)
    assert ok.sum() > 0 and (ok & hit_wall).sum() > 0
    assert np.all(excess[ok] <= 1e-12)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name, flush=True)
