"""Checkpoint I/O for SYMPOL input policies and greedy validation rollouts.

A checkpoint is an Orbax PyTree directory with the entries ``actor_type`` ("sympol"), ``env_id``, ``action_type``
("discrete" for CartPole, "continuous" for MountainCarContinuous), ``actor_config`` (``obs_dim``, ``action_dim``,
``depth``, ``n_estimators``) and the actor parameters and path indices under ``actor``. The same parameters are also
stored under ``sympol``. Both entries belong to the format of the shipped policies.

Importing this module switches JAX to 64-bit mode. The post-processing of the paper's input policies (pruning checks,
validation rollouts, export) ran in this mode, so it is kept to reproduce those results exactly.
"""
import os

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import orbax.checkpoint.args as orbax_args

from vespa.pretraining.envs import CARTPOLE_ENV_ID
from vespa.sympol import SYMPOL_RL

jax.config.update("jax_enable_x64", True)


def restore_raw(path: str) -> dict:
    """Restore the complete Orbax tree of a checkpoint onto the CPU."""
    checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.StandardCheckpointHandler())
    cpu_sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    restore_args = orbax_args.StandardRestore(fallback_sharding=cpu_sharding)
    return checkpointer.restore(os.path.abspath(path), args=restore_args)


def load_checkpoint(path: str) -> dict:
    """Load a SYMPOL checkpoint.

    Returns a dict with ``actor_type``, ``params``, ``indices``, ``actor_config``, ``action_type`` and ``env_id``.
    """
    raw = restore_raw(path)
    if str(raw.get("actor_type")) != "sympol":
        raise ValueError(f"{path} is not a SYMPOL checkpoint")
    return {
        "actor_type": "sympol",
        "params": raw["actor"]["params"],
        "indices": raw["actor"]["indices"],
        "actor_config": dict(raw["actor_config"]),
        "action_type": str(raw["action_type"]),
        "env_id": str(raw["env_id"]),
    }


def save_checkpoint(path: str, params, indices, actor_config: dict, action_type: str, env_id: str) -> None:
    """Write a SYMPOL checkpoint in the format described in the module docstring (overwrites ``path``)."""
    actor = {"params": params, "indices": indices}
    payload = {
        "actor_type": "sympol",
        "env_id": env_id,
        "action_type": action_type,
        "actor_config": actor_config,
        "actor": actor,
        "sympol": actor,
    }
    orbax.checkpoint.PyTreeCheckpointer().save(os.path.abspath(path), payload, force=True)


def load_actor(path: str):
    """Load a checkpoint and build its actor.

    Returns ``(checkpoint, actor, params, indices)``, where ``params`` and ``indices`` are JAX arrays and
    ``actor.apply(params, observations, indices=indices)`` evaluates the policy.
    """
    checkpoint = load_checkpoint(path)
    config = checkpoint["actor_config"]
    actor = SYMPOL_RL(
        obs_dim=int(config["obs_dim"]),
        action_dim=int(config["action_dim"]),
        action_type=checkpoint["action_type"],
        depth=int(config["depth"]),
        n_estimators=int(config["n_estimators"]),
    )
    params = jax.tree.map(jnp.asarray, checkpoint["params"])
    indices = jax.tree.map(jnp.asarray, checkpoint["indices"])
    return checkpoint, actor, params, indices


def rollout_mean_return(apply_actor, params, n_episodes: int, seed: int):
    """Mean and standard deviation of the return of the greedy CartPole policy over ``n_episodes`` episodes.

    ``apply_actor(params, observations)`` returns the action logits. Episode ``i`` starts from the environment's
    default start distribution, reset with the ``i``-th draw of
    ``numpy.random.default_rng(seed).integers(0, 2**31 - 1)``. The action is the argmax of the logits.
    Returns ``(mean, std, n_episodes)``.
    """
    env = gym.make(CARTPOLE_ENV_ID)
    max_steps = int(env.spec.max_episode_steps)
    rng = np.random.default_rng(seed)
    returns = np.empty(int(n_episodes), dtype=np.float64)
    try:
        for episode in range(int(n_episodes)):
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            episode_return = 0.0
            for _ in range(max_steps):
                logits = np.asarray(apply_actor(params, np.asarray(obs, dtype=np.float32)[None, :]))[0]
                obs, reward, terminated, truncated, _ = env.step(int(np.argmax(logits)))
                episode_return += float(reward)
                if terminated or truncated:
                    break
            returns[episode] = episode_return
    finally:
        env.close()
    return float(returns.mean()), float(returns.std()), int(n_episodes)
