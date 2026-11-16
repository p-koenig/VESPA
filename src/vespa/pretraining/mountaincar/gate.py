"""Pruning check and acceptance gate of one MountainCarContinuous input policy.

    python -m vespa.pretraining.mountaincar.gate <seed_dir>

``<seed_dir>`` holds the exported checkpoint ``base`` and its pruned version ``base-pruned``.

1. Pruning check: the pruned tree must give identical raw action means on a 1201 x 701 state grid and on 200,000
   uniform states, and no sampled state may reach a leaf whose value pruning changed (test: the changed leaves are
   set to 1e30 in a copy of the unpruned tree).
2. Gate: rollouts in stock gymnax MountainCarContinuous (``jax.vmap`` of ``step_env``). Greedy: the raw leaf mean is
   passed to the environment (which clips the force), from 1,001 evenly spaced x0 in [-0.6, -0.4] with v0 = 0.
   Sampled: mean + exp(log_std) * N(0, 1) without clipping (the SYMPOL training semantics), 1,000 episodes with
   x0 ~ U[-0.6, -0.4]. Success means reaching the goal (x >= 0.45, v >= 0) within 200 steps. The policy passes if at
   least 10% of the greedy and of the sampled episodes succeed. Success within 999 steps is reported as context.

Writes ``<seed_dir>/gate.json``.
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments.classic_control.continuous_mountain_car import ContinuousMountainCar, EnvState

from vespa.pretraining.checkpoint import load_actor

THRESHOLD = 0.10
ENV = ContinuousMountainCar()
ENV_PARAMS = ENV.default_params
_step = jax.vmap(lambda x, v, a: ENV.step_env(
    jax.random.PRNGKey(0), EnvState(position=x, velocity=v, time=0), a, ENV_PARAMS)[1])


def rollout(actor, params, idx, x0, key, sampled, T=999):
    """Roll out the policy from positions ``x0`` (v0 = 0) for ``T`` steps.

    Returns the step at which each episode reached the goal (-1 if never), its raw return and its return with
    clipped actions.
    """
    lstd = jnp.asarray(params["log_std"]).reshape(-1)[0]

    def body(carry, t):
        x, v, alive, tg, rr, rc, key = carry
        mean = actor.apply(params, jnp.stack([x, v], -1), indices=idx)[0][:, 0]
        key, sk = jax.random.split(key)
        a = mean + (jnp.exp(lstd) * jax.random.normal(sk, x.shape) if sampled else 0.0)
        st = _step(x, v, a[:, None].astype(jnp.float32))
        x2, v2 = st.position, st.velocity
        goal = (x2 >= ENV_PARAMS.goal_position) & (v2 >= ENV_PARAMS.goal_velocity)
        r_raw = -0.1 * a ** 2 + 100.0 * goal
        r_clip = -0.1 * jnp.clip(a, -1, 1) ** 2 + 100.0 * goal
        rr = rr + jnp.where(alive, r_raw, 0.0)
        rc = rc + jnp.where(alive, r_clip, 0.0)
        tg = jnp.where(alive & goal, t + 1, tg)
        x = jnp.where(alive, x2, x)
        v = jnp.where(alive, v2, v)
        alive = alive & (~goal)
        updated = (x, v, alive, tg, rr, rc, key)
        # In 64-bit mode t is int64 and the rewards may be promoted: keep every carry entry at its initial dtype.
        return tuple(jnp.asarray(n_).astype(o.dtype) for n_, o in zip(updated, carry)), None

    n = x0.shape[0]
    init = (jnp.asarray(x0, jnp.float32), jnp.zeros(n, jnp.float32), jnp.ones(n, bool),
            jnp.full(n, -1, jnp.int32), jnp.zeros(n), jnp.zeros(n), key)
    out, _ = jax.jit(lambda c: jax.lax.scan(body, c, jnp.arange(T)))(init)
    _, _, _, tg, rr, rc, _ = out
    return np.asarray(tg), np.asarray(rr), np.asarray(rc)


def gate_metrics(actor, params, idx):
    """Goal-reaching rates of greedy and sampled rollouts and summary statistics of the policy."""
    xg = np.linspace(-0.6, -0.4, 1001)
    tg, rr, rc = rollout(actor, params, idx, xg, jax.random.PRNGKey(0), False)
    rng = np.random.default_rng(2026)
    xs = rng.uniform(-0.6, -0.4, 1000)
    ts, sr, _ = rollout(actor, params, idx, xs, jax.random.PRNGKey(2026), True)
    g200 = (tg > 0) & (tg <= 200)
    s200 = (ts > 0) & (ts <= 200)
    gx = np.linspace(-1.2, 0.6, 241)
    gv = np.linspace(-0.07, 0.07, 141)
    X, V = np.meshgrid(gx, gv, indexing="ij")
    m = np.asarray(actor.apply(params, jnp.asarray(np.stack([X.ravel(), V.ravel()], -1), jnp.float32),
                               indices=idx)[0][:, 0])
    lstd = float(np.asarray(params["log_std"]).reshape(-1)[0])
    rec = {
        "greedy_goal200": float(g200.mean()), "greedy_goal999": float((tg > 0).mean()),
        "greedy_steps_mean_goal": float(tg[tg > 0].mean()) if (tg > 0).any() else None,
        "greedy_return_raw_mean": float(rr.mean()), "greedy_return_clip_mean": float(rc.mean()),
        "sampled_goal200": float(s200.mean()), "sampled_goal999": float((ts > 0).mean()),
        "sampled_return_raw_mean": float(sr.mean()),
        "log_std": lstd, "sigma": float(np.exp(lstd)),
        "grid_saturated_frac": float((np.abs(m) >= 1.0).mean()),
        "grid_n_distinct_clipped_actions": int(np.unique(np.round(np.clip(m, -1, 1), 6)).size),
        "leaf_mean_min": float(np.asarray(params["leaf_array"]).min()),
        "leaf_mean_max": float(np.asarray(params["leaf_array"]).max()),
        "threshold": THRESHOLD,
    }
    rec["pass_gate"] = bool(rec["greedy_goal200"] >= THRESHOLD and rec["sampled_goal200"] >= THRESHOLD)
    return rec


def prune_check(actor, p_par, idx, p_pr):
    """Compare the unpruned (``p_par``) and pruned (``p_pr``) parameters on a grid and on random states."""
    rng = np.random.default_rng(7)
    gx = np.linspace(-1.2, 0.6, 1201)
    gv = np.linspace(-0.07, 0.07, 701)
    X, V = np.meshgrid(gx, gv, indexing="ij")
    S = np.concatenate([np.stack([X.ravel(), V.ravel()], -1),
                        np.stack([rng.uniform(-1.2, 0.6, 200000), rng.uniform(-0.07, 0.07, 200000)], -1)]
                       ).astype(np.float32)
    f = jax.jit(lambda p, s: actor.apply(p, s, indices=idx)[0][:, 0])

    def means(p):
        return np.concatenate([np.asarray(f(p, jnp.asarray(S[i:i + 200000]))) for i in range(0, len(S), 200000)])

    a, b = means(p_par), means(p_pr)
    la, lb = np.asarray(p_par["leaf_array"]), np.asarray(p_pr["leaf_array"])
    changed = la != lb
    sentinel = dict(p_par)
    sentinel["leaf_array"] = jnp.asarray(np.where(changed, 1e30, la))
    c = means(sentinel)
    rec = {"n_states": int(len(S)), "n_changed_actions_raw": int(np.sum(a != b)),
           "n_changed_actions_clipped": int(np.sum(np.clip(a, -1, 1) != np.clip(b, -1, 1))),
           "n_changed_leaf_entries": int(changed.sum()),
           "n_states_reaching_changed_leaf": int(np.sum(np.abs(c) > 1e29)),
           "other_params_identical": bool(all(np.array_equal(np.asarray(p_par[k]), np.asarray(p_pr[k]))
                                              for k in p_par if k != "leaf_array"))}
    rec["ok"] = (rec["n_changed_actions_raw"] == 0 and rec["n_states_reaching_changed_leaf"] == 0
                 and rec["other_params_identical"])
    return rec


def main(seed_dir):
    t0 = time.time()
    up, pr = os.path.join(seed_dir, "base"), os.path.join(seed_dir, "base-pruned")
    ck_u, actor, p_u, idx_u = load_actor(up)
    _, _, p_p, idx_p = load_actor(pr)
    same_idx = all(np.array_equal(np.asarray(idx_u[k]), np.asarray(idx_p[k])) for k in idx_u)
    rec = {"ckpt_unpruned": os.path.abspath(up), "ckpt_pruned": os.path.abspath(pr),
           "env_id": ck_u["env_id"], "action_type": ck_u["action_type"],
           "prune_check": {**prune_check(actor, p_u, idx_u, p_p), "indices_identical": bool(same_idx)}}
    rec["prune_check"]["ok"] = bool(rec["prune_check"]["ok"] and same_idx)
    rec["gate_unpruned"] = gate_metrics(actor, p_u, idx_u)
    rec["gate_pruned"] = gate_metrics(actor, p_p, idx_p)
    rec["pass_gate"] = bool(rec["gate_unpruned"]["pass_gate"] and rec["prune_check"]["ok"])
    rec["sec"] = round(time.time() - t0, 1)
    with open(os.path.join(seed_dir, "gate.json"), "w") as f:
        json.dump(rec, f, indent=1)
    g = rec["gate_unpruned"]
    print(f"[gate] {up}: prune_ok={rec['prune_check']['ok']} greedy@200={g['greedy_goal200']:.3f} "
          f"sampled@200={g['sampled_goal200']:.3f} greedy@999={g['greedy_goal999']:.3f} "
          f"sigma={g['sigma']:.3g} -> pass={rec['pass_gate']} ({rec['sec']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
