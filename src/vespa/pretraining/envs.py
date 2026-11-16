"""Environments of the input-policy pretraining.

CartPole: ``VerifyCartPole-v0`` is gymnasium's classic CartPole (CartPole-v1 dynamics, 500-step episodes) under the
environment id that the checkpoints record and the verifier reads. It is registered when this module is imported.

MountainCarContinuous: the input policies are trained by the upstream SYMPOL code on gymnax ``MountainCarContinuous-v0``
(see ``experiments/pretraining/mountaincar``). Their checkpoints record the environment id
``VerifyMountainCarContinuousGymnax-v0``. Pretraining only needs its observation bounds (for pruning).
"""
import math

import gymnasium as gym
import numpy as np

CARTPOLE_ENV_ID = "VerifyCartPole-v0"
MOUNTAINCAR_ENV_ID = "VerifyMountainCarContinuousGymnax-v0"

if CARTPOLE_ENV_ID not in gym.registry:
    gym.register(
        id=CARTPOLE_ENV_ID,
        entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
        max_episode_steps=500,
    )

_CARTPOLE_THETA_LIMIT = 12 * 2 * math.pi / 360

# Observation-space bounds (low, high) of each environment, as float32 like the gymnasium observation spaces.
OBSERVATION_BOUNDS = {
    CARTPOLE_ENV_ID: (
        np.array([-2.4 * 2, -np.inf, -_CARTPOLE_THETA_LIMIT * 2, -np.inf], dtype=np.float32),
        np.array([2.4 * 2, np.inf, _CARTPOLE_THETA_LIMIT * 2, np.inf], dtype=np.float32),
    ),
    MOUNTAINCAR_ENV_ID: (
        np.array([-1.2, -0.07], dtype=np.float32),
        np.array([0.6, 0.07], dtype=np.float32),
    ),
}


def make_cartpole():
    """Create one CartPole environment (used by the vectorised training environments)."""
    return gym.make(CARTPOLE_ENV_ID)
