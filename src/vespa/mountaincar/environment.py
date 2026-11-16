"""Gymnasium front-end for stock gymnax MountainCarContinuous-v0.

Fine-tuning runs in gymnasium environments. This class calls gymnax's own ``step_env`` and ``reset_env``
(jitted, on the CPU device), so pretraining, fine-tuning and verification share one MDP:

* force = clip(a, -1, 1) * 0.0015, v = clip(v + force - 0.0025 cos(3x), -0.07, 0.07),
  x = clip(x + v, -1.2, 0.6), and v = 0 if x >= 0.45 and v < 0. There is no inelastic left wall.
* reward = -0.1 a^2 (raw action) + 100 [x >= 0.45 and v >= 0].
* terminated = [x >= 0.45 and v >= 0]. Episodes are truncated after 999 steps by gymnasium's TimeLimit.
* reset: x0 ~ U[-0.6, -0.4] (float32), v0 = 0, with a gymnax key drawn from gymnasium's ``np_random``.
* ``state`` (get and set) is [x, v], which lets fine-tuning start an episode in a counterexample region.
"""
from __future__ import annotations

import gymnasium as gym
import jax
import numpy as np
from gymnasium import spaces

from vespa.mountaincar.config import ENV_ID

MAX_EPISODE_STEPS = 999

_FUNCTIONS = None


def _cpu():
    return jax.devices("cpu")[0]


def _gymnax_functions():
    """Jitted (step, reset) wrappers around gymnax, built once per process."""
    global _FUNCTIONS
    if _FUNCTIONS is None:
        from gymnax.environments.classic_control.continuous_mountain_car import ContinuousMountainCar, EnvState
        env = ContinuousMountainCar()
        params = env.default_params

        def step(pos, vel, t, action):
            state = EnvState(position=pos, velocity=vel, time=t)
            _, state2, reward, _done, _ = env.step_env(jax.random.PRNGKey(0), state, action, params)
            goal = (state2.position >= params.goal_position) & (state2.velocity >= params.goal_velocity)
            return state2.position, state2.velocity, state2.time, reward, goal

        def reset(key):
            _, state = env.reset_env(key, params)
            return state.position, state.velocity

        _FUNCTIONS = (jax.jit(step), jax.jit(reset))
    return _FUNCTIONS


class MountainCarContinuousGymnaxEnv(gym.Env):
    """Gymnasium environment whose dynamics, reward and termination are those of gymnax."""

    metadata = {"render_modes": []}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=np.array([-1.2, -0.07], dtype=np.float32), high=np.array([0.6, 0.07], dtype=np.float32),
            dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._pos = np.float32(-0.5)
        self._vel = np.float32(0.0)
        self._t = np.int32(0)

    @property
    def state(self):
        return np.array([self._pos, self._vel], dtype=np.float32)

    @state.setter
    def state(self, s):
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        self._pos = np.float32(s[0])
        self._vel = np.float32(s[1])

    def _obs(self):
        return np.array([self._pos, self._vel], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        _, reset = _gymnax_functions()
        key = jax.random.PRNGKey(int(self.np_random.integers(0, 2 ** 31 - 1)))
        with jax.default_device(_cpu()):
            pos, vel = reset(key)
        self._pos, self._vel = np.float32(pos), np.float32(vel)
        self._t = np.int32(0)
        return self._obs(), {}

    def step(self, action):
        step, _ = _gymnax_functions()
        a = np.asarray(action, dtype=np.float32).reshape(-1)[:1]
        with jax.default_device(_cpu()):
            pos, vel, t, reward, goal = step(self._pos, self._vel, self._t, a)
        self._pos, self._vel = np.float32(pos), np.float32(vel)
        self._t = np.int32(t)
        return self._obs(), float(reward), bool(goal), False, {}


def register() -> None:
    """Register the environment under ``config.ENV_ID`` (999-step episodes)."""
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="vespa.mountaincar.environment:MountainCarContinuousGymnaxEnv",
                     max_episode_steps=MAX_EPISODE_STEPS)
