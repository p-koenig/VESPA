#!/usr/bin/env python3
"""Fine-tuning episodes: horizon budget, truncation and GAE.

  1. GAE: a truncation bootstraps gamma * V(final observation), a termination 0,
     and no advantage carries across an episode end (vs. a plain loop).
  2. Budget: an episode starting at step s0 runs at most horizon - s0 steps
     (at least 1) and then truncates, never terminates.

    JAX_PLATFORMS=cpu PYTHONPATH=src python tests/cartpole/test_horizon_cap.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import gymnasium as gym
import jax.numpy as jnp
from vespa.cartpole import finetune as F
from vespa.cartpole.policy import ENV_ID, ensure_env_registered

GAMMA, LAM = 0.99, 0.95


def gae_reference(rewards, values, terms, truncs, vfinals, next_value):
    T, n = rewards.shape
    adv = np.zeros((T, n))
    last = np.zeros(n)
    for t in reversed(range(T)):
        v_t1 = next_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - np.clip(terms[t] + truncs[t], 0, 1)
        delta = rewards[t] + GAMMA * (nonterm * v_t1 + truncs[t] * vfinals[t]) - values[t]
        last = delta + GAMMA * LAM * nonterm * last
        adv[t] = last
    return adv, adv + values


def gae(rewards, values, terms, truncs, vfinals, next_value):
    a, r = F.gae(*(jnp.asarray(x, jnp.float32) for x in
                   (next_value, terms, truncs, vfinals, values, rewards)), GAMMA, LAM)
    return np.asarray(a), np.asarray(r)


def test_gae():
    rng = np.random.default_rng(0)
    T, n = 9, 3
    rewards, values = rng.normal(size=(T, n)), rng.normal(size=(T, n))
    vfinals, next_value = rng.normal(size=(T, n)), rng.normal(size=n)
    terms, truncs = np.zeros((T, n)), np.zeros((T, n))
    terms[3, 0] = terms[7, 2] = 1.0
    truncs[5, 1] = truncs[2, 2] = 1.0
    vfinals = vfinals * truncs
    a, r = gae(rewards, values, terms, truncs, vfinals, next_value)
    ra, rr = gae_reference(rewards, values, terms, truncs, vfinals, next_value)
    assert np.allclose(a, ra, atol=1e-5) and np.allclose(r, rr, atol=1e-5)
    z = np.zeros((1, 1))
    # truncation at the last step: delta = r + gamma * vfinal - v
    a, _ = gae(np.ones((1, 1)), z, z, np.ones((1, 1)), np.full((1, 1), 2.0), np.full(1, 5.0))
    assert np.isclose(a[0, 0], 1.0 + GAMMA * 2.0), a
    # termination: successor value 0
    a, _ = gae(np.ones((1, 1)), z, np.ones((1, 1)), z, np.full((1, 1), 2.0), np.full(1, 5.0))
    assert np.isclose(a[0, 0], 1.0), a


def wrapper(start_step, mix_p, horizon=200, seed=0):
    ensure_env_registered()
    base = gym.make(ENV_ID, max_episode_steps=F._INNER_TIME_LIMIT)
    lo, hi = np.zeros((1, 4)), np.full((1, 4), 0.01)
    return F.MixedStartWrapper(base, lo, hi, [start_step], mix_p=mix_p,
                               rng=np.random.default_rng(seed), horizon=horizon)


def test_budget():
    w = wrapper(147, 0.0)
    assert w.budget_for(0) == 200 and w.budget_for(147) == 53
    assert w.budget_for(200) == 1 and w.budget_for(500) == 1
    w.reset(seed=1)                                   # mix_p 0: start distribution
    assert w.start_step == 0 and w.step_budget == 200
    w = wrapper(147, 1.0)                             # mix_p 1: counterexample region
    w.reset(seed=1)
    assert w.start_step == 147 and w.step_budget == 53


def test_truncates_not_terminates():
    w = wrapper(197, 1.0)                             # budget 3
    w.reset(seed=2)
    seen = [tuple(w.step(0)[2:4]) for _ in range(3)]
    assert seen[-1] == (False, True), seen
    assert not any(t or u for t, u in seen[:-1]), seen


if __name__ == "__main__":
    for t in (test_gae, test_budget, test_truncates_not_terminates):
        t()
        print(f"  PASS  {t.__name__}")
    print("PASS test_horizon_cap")
