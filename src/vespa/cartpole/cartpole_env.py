"""gymnasium's CartPole (Euler integrator), registered as VerifyCartPole-v0.

Identical to gymnasium.envs.classic_control.CartPoleEnv without rendering:
reward 1 per step, termination when |x| > 2.4 or |theta| > 12 degrees, start
state uniform in [-0.05, 0.05]^4. Fine-tuning and the validation episodes run
on it; the verifier uses the interval extension in ``env_specs``.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils


class CartPoleEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5                       # half the pole length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02                         # seconds per step
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        high = np.array([self.x_threshold * 2, np.inf, self.theta_threshold_radians * 2, np.inf],
                        dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.render_mode = render_mode
        self.state = None
        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * np.square(theta_dot) * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(x < -self.x_threshold or x > self.x_threshold
                          or theta < -self.theta_threshold_radians
                          or theta > self.theta_threshold_radians)
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0    # the pole just fell
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn("step() called after the episode terminated; call reset() first.")
            self.steps_beyond_terminated += 1
            reward = 0.0
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(options, -0.05, 0.05)
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None
        return np.array(self.state, dtype=np.float32), {}
