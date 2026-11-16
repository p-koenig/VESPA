"""CPU tests of the pretraining package: SYMPOL tree semantics, checkpoint format, pruning and the PPO schedule.

    python tests/pretraining/test_pretraining.py
"""
import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from vespa.pretraining import envs
from vespa.pretraining.checkpoint import load_actor, load_checkpoint, rollout_mean_return, save_checkpoint
from vespa.pretraining.ppo import accumulation_factor
from vespa.pretraining.pruning import prune_checkpoint
from vespa.sympol import SYMPOL_RL

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHIPPED_PRUNED = os.path.join(REPO, "policies", "cartpole", "evaluation", "s6p")


def random_tree(depth, seed=0, action_dim=2):
    actor = SYMPOL_RL(obs_dim=4, action_dim=action_dim, action_type="discrete", depth=depth, n_estimators=1)
    key = jax.random.PRNGKey(seed)
    params = actor.init(key)
    params["split_values"] = params["split_values"] * 20  # thresholds of the order of the states
    params["leaf_array"] = params["leaf_array"] * 100
    return actor, params, actor.init_indices(key)


def walk(params, indices, x):
    """Greedy action by explicit traversal: go left iff x[feature] < threshold."""
    sel = np.asarray(params["split_idx_array"])[0]
    thr = np.asarray(params["split_values"])[0]
    features = np.asarray(indices["features_by_estimator"])[0]
    leaves = np.asarray(params["leaf_array"])[0]
    node, n_internal = 0, sel.shape[0]
    while node < n_internal:
        k = int(np.argmax(sel[node]))
        node = 2 * node + 1 if x[features[k]] < thr[node, k] else 2 * node + 2
    return int(np.argmax(leaves[node - n_internal]))


def test_forward_pass_is_the_decision_tree():
    actor, params, indices = random_tree(depth=4)
    states = np.random.default_rng(1).normal(0, 2, (500, 4)).astype(np.float32)
    logits = np.asarray(actor.apply(params, jnp.asarray(states), indices=indices))
    assert [int(a) for a in logits.argmax(-1)] == [walk(params, indices, x) for x in states]


def test_path_indices_are_heap_ordered():
    _, _, indices = random_tree(depth=3)
    internal = np.asarray(indices["internal_node_index_list"])
    direction = np.asarray(indices["path_identifier_list"])
    for leaf in range(8):
        node = 0
        for d in range(3):
            assert internal[leaf, d] == node
            node = 2 * node + 1 + int(direction[leaf, d])
        assert node - 7 == leaf


def test_checkpoint_round_trip():
    tmp_dir = tempfile.mkdtemp()
    actor, params, indices = random_tree(depth=3)
    path = os.path.join(tmp_dir, "ckpt")
    config = {"obs_dim": 4, "action_dim": 2, "depth": 3, "n_estimators": 1}
    save_checkpoint(path, params, indices, config, "discrete", envs.CARTPOLE_ENV_ID)
    loaded = load_checkpoint(path)
    assert loaded["env_id"] == envs.CARTPOLE_ENV_ID and loaded["action_type"] == "discrete"
    assert {k: int(v) for k, v in loaded["actor_config"].items()} == config
    for name in params:
        assert np.array_equal(np.asarray(loaded["params"][name]), np.asarray(params[name]))


def test_pruning_keeps_every_action():
    tmp_dir = tempfile.mkdtemp()
    _, params, indices = random_tree(depth=5, seed=3)
    path = os.path.join(tmp_dir, "ckpt")
    save_checkpoint(path, params, indices, {"obs_dim": 4, "action_dim": 2, "depth": 5, "n_estimators": 1},
                    "discrete", envs.CARTPOLE_ENV_ID)
    pruned_path, leaves = prune_checkpoint(path)
    assert leaves[0] <= 32
    _, actor, p_unpruned, idx = load_actor(path)
    _, _, p_pruned, _ = load_actor(pruned_path)
    low, high = envs.OBSERVATION_BOUNDS[envs.CARTPOLE_ENV_ID]
    states = np.random.default_rng(2).uniform(np.maximum(low, -10), np.minimum(high, 10), (20000, 4))
    states = jnp.asarray(states.astype(np.float32))
    a = np.asarray(actor.apply(p_unpruned, states, indices=idx)).argmax(-1)
    b = np.asarray(actor.apply(p_pruned, states, indices=idx)).argmax(-1)
    assert np.array_equal(a, b)


def test_observation_bounds_match_cartpole():
    env = gym.make(envs.CARTPOLE_ENV_ID)
    low, high = envs.OBSERVATION_BOUNDS[envs.CARTPOLE_ENV_ID]
    assert np.array_equal(env.observation_space.low, low) and np.array_equal(env.observation_space.high, high)
    assert env.spec.max_episode_steps == 500


def test_accumulation_factor_doubles_every_eighth():
    total = 8_000
    factors = [accumulation_factor(step, total) for step in range(0, total, 1000)]
    assert factors == [1, 2, 4, 8, 16, 32, 64, 128]


def test_shipped_policy_balances():
    _, actor, params, indices = load_actor(SHIPPED_PRUNED)
    mean, _, n = rollout_mean_return(lambda p, o: actor.apply(p, o, indices=indices), params, 3, 0)
    assert n == 3 and mean == 500.0


if __name__ == "__main__":
    failed = []
    for name, test in [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]:
        try:
            test()
            print("ok  ", name, flush=True)
        except Exception as error:
            failed.append(name)
            print("FAIL", name, repr(error), flush=True)
    if failed:
        print(f"FAILED ({len(failed)}): " + ", ".join(failed))
        raise SystemExit(1)
    print("all tests passed")
