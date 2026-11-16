"""End-to-end tests of the verifier and the counterexample regions on two-region policies.

Policies (split at v = 1e-6): push along the velocity with force 1 (energy pumping) or 0.35, and push left everywhere.
* every start state region the verifier certifies reaches the goal in concrete gymnax rollouts, and the
  ratio of safe start state regions does not exceed the concrete goal fraction on a dense grid;
* every concrete start that misses the goal lies in a counterexample region.

Run on CPU from the repository root:  JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest tests/mountaincar
(or run this file directly).
"""
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments.classic_control.continuous_mountain_car import ContinuousMountainCar, EnvState

from vespa.mountaincar import verifier
from vespa.mountaincar.counterexamples import counterexample_regions

GX = ContinuousMountainCar()
GP = GX.default_params
_STEP = jax.jit(jax.vmap(lambda x, v, a: GX.step_env(
    jax.random.PRNGKey(0), EnvState(position=x, velocity=v, time=0), a, GP)[1]))
CAP = 65536
# Split of the two-region policies. A threshold above 0 puts the start states (v = 0) in one region only.
THRESHOLD = np.float32(1e-6)


def velocity_regions(a_neg, a_pos):
    """Two regions split at v = THRESHOLD: action a_neg below it, a_pos above it."""
    t = float(THRESHOLD)
    return [{"action": a_neg, "bounds": [[-np.inf, np.inf], [-np.inf, t]]},
            {"action": a_pos, "bounds": [[-np.inf, np.inf], [t, np.inf]]}]


def reaches_goal(x0, a_neg, a_pos, horizon=200):
    """Concrete gymnax rollouts from (x0, 0): True where the goal is reached within ``horizon`` steps."""
    x = jnp.asarray(x0, jnp.float32)
    v = jnp.zeros_like(x)
    alive = np.ones(len(x0), bool)
    for _ in range(horizon):
        a = jnp.where(v < THRESHOLD, a_neg, a_pos)[:, None].astype(jnp.float32)
        st = _STEP(x, v, a)
        x = jnp.asarray(np.where(alive, np.asarray(st.position), np.asarray(x)))
        v = jnp.asarray(np.where(alive, np.asarray(st.velocity), np.asarray(v)))
        alive &= ~(np.asarray(x) >= 0.45)
    return ~alive


def _check_policy(a_neg, a_pos, rng):
    result = verifier.verify_regions(velocity_regions(a_neg, a_pos), CAP)
    assert result.status == "OK"
    certified = np.flatnonzero(~result.piece_violating)
    if certified.size:
        pick = rng.choice(certified, size=min(2000, certified.size), replace=False)
        xs = rng.uniform(result.piece_lo[pick, 0], result.piece_hi[pick, 0])
        assert reaches_goal(xs, a_neg, a_pos).all()
    dense = reaches_goal(np.linspace(-0.6, -0.4, 4001), a_neg, a_pos).mean()
    assert result.certified_fraction <= dense + 1e-3
    return result


def test_energy_pumping_certified_regions_reach_goal():
    _check_policy(-1.0, 1.0, np.random.default_rng(0))


def test_weak_pumping_and_counterexample_regions():
    a_neg, a_pos = -0.35, 0.35
    result = _check_policy(a_neg, a_pos, np.random.default_rng(1))
    if not result.violated:
        return
    regions = counterexample_regions(result, seed=0, log=lambda m: None)
    grid = np.linspace(-0.6, -0.4, 2001)
    missed = grid[~reaches_goal(grid, a_neg, a_pos)]
    inside = ((regions["lo"][None, :, 0] <= missed[:, None]) & (missed[:, None] <= regions["hi"][None, :, 0])
              & (regions["lo"][None, :, 1] <= 0.0) & (0.0 <= regions["hi"][None, :, 1])).any(1)
    assert inside.all()


def test_push_left_certifies_nothing():
    result = verifier.verify_regions(velocity_regions(-1.0, -1.0), CAP)
    assert result.violated and not result.safe
    assert result.certified_fraction == 0.0
    assert result.n_violating_pieces == result.n_init_pieces


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name, flush=True)
