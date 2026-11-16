"""SYMPOL decision-tree policies: checkpoint I/O, leaf regions and greedy validation return.

A policy is a single SYMPOL decision tree of depth 7 over the observation (x, v) with one Gaussian
action per leaf. The deterministic policy the verifier checks applies the clipped leaf mean.
"""
from __future__ import annotations

import os

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import orbax.checkpoint.args as orbax_args

from vespa.mountaincar import dynamics, environment
from vespa.mountaincar.config import ENV_ID
from vespa.sympol import SYMPOL_RL


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> dict:
    """Restore an Orbax checkpoint of a SYMPOL policy onto the CPU device.

    Returns a dict with ``params``, ``indices``, ``actor_config`` (obs_dim, action_dim, depth,
    n_estimators), ``action_type`` and ``env_id``.
    """
    checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.StandardCheckpointHandler())
    cpu_sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    raw = checkpointer.restore(os.path.abspath(path),
                               args=orbax_args.StandardRestore(fallback_sharding=cpu_sharding))
    if str(raw.get("actor_type")) != "sympol":
        raise ValueError(f"{path}: not a SYMPOL checkpoint (actor_type={raw.get('actor_type')!r})")
    return {
        "params": raw["actor"]["params"],
        "indices": raw["actor"]["indices"],
        "actor_config": dict(raw["actor_config"]),
        "action_type": str(raw["action_type"]),
        "env_id": str(raw["env_id"]),
    }


def save_checkpoint(path: str, params, indices, actor_config: dict, action_type: str, env_id: str) -> None:
    """Write a SYMPOL checkpoint in the format of the input policies (readable by ``load_checkpoint``)."""
    payload = {
        "actor_type": "sympol",
        "env_id": env_id,
        "action_type": action_type,
        "actor_config": actor_config,
        "actor": {"params": params, "indices": indices},
        "sympol": {"params": params, "indices": indices},
    }
    orbax.checkpoint.PyTreeCheckpointer().save(os.path.abspath(path), payload, force=True)


def make_actor(actor_config: dict, action_type: str) -> SYMPOL_RL:
    """The SYMPOL module of a checkpoint (its ``apply`` returns [mean, log_std])."""
    return SYMPOL_RL(obs_dim=int(actor_config["obs_dim"]), action_dim=int(actor_config["action_dim"]),
                     action_type=action_type, depth=int(actor_config["depth"]),
                     n_estimators=int(actor_config["n_estimators"]))


# ---------------------------------------------------------------------------
# Leaf regions
# ---------------------------------------------------------------------------

def leaf_regions(ckpt: dict) -> list[dict]:
    """Axis-aligned regions of the tree, one per maximal subtree whose leaves share one mean action.

    Each region is ``{"action": float, "bounds": [[x_lo, x_hi], [v_lo, v_hi]]}`` with the action
    clip(leaf mean, -1, 1). Subtrees are merged on exact equality of the raw leaf means. Regions with
    contradictory bounds (lo > hi) are dropped. The order is the left-to-right order of the tree.
    """
    params, indices = ckpt["params"], ckpt["indices"]
    depth = int(ckpt["actor_config"]["depth"])
    obs_dim = int(ckpt["actor_config"]["obs_dim"])
    n_internal = 2 ** depth - 1

    split_idx = np.asarray(params["split_idx_array"][0])
    split_vals = np.asarray(params["split_values"][0])
    features = np.asarray(indices["features_by_estimator"][0])
    leaf_means = [float(v) for v in np.asarray(params["leaf_array"][0])[:, 0]]

    node_sel = np.argmax(split_idx, axis=-1)
    node_feature = features[node_sel]
    node_threshold = split_vals[np.arange(n_internal), node_sel]

    # uniform[n] = the mean shared by every leaf below node n, or None.
    uniform = [None] * (2 * 2 ** depth - 1)
    for leaf in range(2 ** depth):
        uniform[n_internal + leaf] = leaf_means[leaf]
    for node in range(n_internal - 1, -1, -1):
        left, right = 2 * node + 1, 2 * node + 2
        if uniform[left] is not None and uniform[left] == uniform[right]:
            uniform[node] = uniform[left]

    regions = []

    def recurse(node, level, bounds):
        if any(bounds[d][0] > bounds[d][1] for d in range(obs_dim)):
            return
        if uniform[node] is not None or level == depth:
            action = float(np.clip(uniform[node], dynamics.MIN_ACTION, dynamics.MAX_ACTION))
            regions.append({"action": action, "bounds": [list(b) for b in bounds]})
            return
        feature = int(node_feature[node])
        threshold = float(node_threshold[node])
        left_bounds = [list(b) for b in bounds]          # obs[feature] < threshold
        if threshold < left_bounds[feature][1]:
            left_bounds[feature][1] = threshold
        right_bounds = [list(b) for b in bounds]         # obs[feature] >= threshold
        if threshold > right_bounds[feature][0]:
            right_bounds[feature][0] = threshold
        recurse(2 * node + 1, level + 1, left_bounds)
        recurse(2 * node + 2, level + 1, right_bounds)

    recurse(0, 0, [[-np.inf, np.inf] for _ in range(obs_dim)])
    return regions


# ---------------------------------------------------------------------------
# Greedy validation return
# ---------------------------------------------------------------------------

def greedy_return(apply_actor, params, n_episodes: int, seed: int) -> tuple[float, float]:
    """Mean and standard deviation of the undiscounted return of the greedy policy.

    Episodes start from the environment's own start distribution and run for at most 999 steps. The
    greedy action is the clipped Gaussian mean, the action the verifier assigns to each leaf.
    ``apply_actor(params, obs_batch)`` returns the actor output [mean, log_std].
    """
    environment.register()
    env = gym.make(ENV_ID)
    cap = int(env.spec.max_episode_steps)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    rng = np.random.default_rng(seed)
    returns = np.empty(int(n_episodes), dtype=np.float64)
    try:
        for episode in range(int(n_episodes)):
            obs, _ = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
            episode_return = 0.0
            for _ in range(cap):
                out = apply_actor(params, np.asarray(obs, dtype=np.float32)[None, :])
                mean = np.asarray(out[0], dtype=np.float32)[0]
                action = np.clip(mean, action_low, action_high).astype(np.float32)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                if terminated or truncated:
                    break
            returns[episode] = episode_return
    finally:
        env.close()
    return float(returns.mean()), float(returns.std())


def checkpoint_greedy_return(path: str, n_episodes: int, seed: int) -> tuple[float, float]:
    """``greedy_return`` of the policy stored at ``path``."""
    ckpt = load_checkpoint(path)
    actor = make_actor(ckpt["actor_config"], ckpt["action_type"])
    indices = jax.tree.map(jnp.asarray, ckpt["indices"])
    apply_actor = jax.jit(lambda p, o: actor.apply(p, o, indices=indices))
    params = jax.tree.map(jnp.asarray, ckpt["params"])
    return greedy_return(apply_actor, params, n_episodes, seed)

