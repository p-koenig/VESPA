"""Fine-tuning of a decision tree policy with PPO on counterexample regions.

Episodes start, with probability ``mix_p``, uniformly in one of the
counterexample regions (first a uniform start step, then a uniform region of
that step); otherwise from the env's start distribution. VESPA uses mix_p 0.5,
the uninformed baseline mix_p 0. An episode that starts at absolute step s0 runs
for at most horizon - s0 steps (truncated, i.e. bootstrapped, at the end) and
terminates with a penalty of -100 when it leaves the safety box.

PPO trains the soft SYMPOL tree (Adam for the estimator weights and split
values, AdamW for the split selection, leaves and log_std) and a fresh MLP critic.
After training, the mean return of the deterministic policy over 30 episodes
from the env's start distribution is reported before and after.
"""

from __future__ import annotations

import os

import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from flax.training.train_state import TrainState

# Fine-tuning runs with JAX float64 enabled, as in the paper runs (env_specs
# enables it on import).
from vespa.cartpole.env_specs import make_spec, obs_from_state, set_env_state
from vespa.cartpole.policy import ENV_ID, ensure_env_registered, load_checkpoint
from vespa.sympol import SYMPOL_RL

_INNER_TIME_LIMIT = 10 ** 9     # the start-dependent budget below truncates instead
_ADAM_KEYS = ("estimator_weights", "split_values")   # AdamW for the other parameters


class Critic(nn.Module):
    """Value function: MLP with ReLU hidden layers."""
    num_layers: int = 2
    neurons_per_layer: int = 256

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.relu(nn.Dense(self.neurons_per_layer)(x))
        return nn.Dense(1)(x)


class MixedStartWrapper(gym.Wrapper):
    """Start episodes in a counterexample region with probability ``mix_p``.

    ``regions_lo/hi`` (B, 4) with ``region_steps`` (B,) the absolute step of each
    region. A region start draws a step uniformly from the distinct steps, then
    a region of that step, then a state uniformly in it. Every episode is
    truncated after ``horizon - start_step`` steps (at least 1).
    """

    def __init__(self, env, regions_lo, regions_hi, region_steps, mix_p, rng, horizon):
        super().__init__(env)
        self.regions_lo = np.asarray(regions_lo, dtype=np.float64)
        self.regions_hi = np.asarray(regions_hi, dtype=np.float64)
        self.region_steps = np.asarray(region_steps, dtype=np.int64)
        self.step_groups = [np.flatnonzero(self.region_steps == s)
                            for s in np.unique(self.region_steps)]
        self.mix_p = float(mix_p)
        self.rng = rng
        self.horizon = int(horizon)
        self.start_step = 0
        self.step_budget = None
        self._elapsed = 0

    def budget_for(self, start_step):
        return max(self.horizon - int(start_step), 1)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.start_step = 0
        if self.regions_lo.shape[0] > 0 and self.rng.random() < self.mix_p:
            if len(self.step_groups) > 1:
                g = self.step_groups[int(self.rng.integers(0, len(self.step_groups)))]
                i = int(g[int(self.rng.integers(0, g.shape[0]))])
            else:
                i = int(self.rng.integers(0, self.regions_lo.shape[0]))
            state = np.asarray(self.rng.uniform(self.regions_lo[i], self.regions_hi[i]),
                               dtype=np.float64)
            set_env_state(self.env, state)
            obs = obs_from_state(state)
            self.start_step = int(self.region_steps[i])
        self.step_budget = self.budget_for(self.start_step)
        self._elapsed = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed += 1
        if self._elapsed >= self.step_budget and not terminated:
            truncated = True
        return obs, reward, terminated, truncated, info


class UnsafePenaltyWrapper(gym.Wrapper):
    """Terminate with ``penalty`` added to the reward when the state leaves the safety box."""

    def __init__(self, env, penalty, spec):
        super().__init__(env)
        self.penalty = float(penalty)
        self.spec_ = spec

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.spec_.unsafe_obs(obs):
            terminated = True
            reward = float(reward) + self.penalty
        return obs, reward, terminated, truncated, info


def gae(next_value, terms, truncs, vfinals, values, rewards, gamma, gae_lambda):
    """Advantages and returns over a rollout of shape (T, n_envs).

    The advantage is cut at every episode end. A terminated transition has
    successor value 0; a truncated one bootstraps through the value of its
    true final observation (``vfinals``).
    """
    bound = jnp.clip(terms + truncs, 0.0, 1.0)
    values_aug = jnp.concatenate([values, next_value[None, :]], axis=0)

    def body(adv, inp):
        bound_t, trunc_t, vfin_t, v_t1, v_t, r_t = inp
        nonterm = 1.0 - bound_t
        v_next = nonterm * v_t1 + trunc_t * vfin_t
        delta = r_t + gamma * v_next - v_t
        new_adv = delta + gamma * gae_lambda * nonterm * adv
        return new_adv, new_adv

    _, advantages = jax.lax.scan(
        body, jnp.zeros_like(rewards[0]),
        (bound, truncs, vfinals, values_aug[1:], values_aug[:-1], rewards), reverse=True)
    return advantages, advantages + values


def save_checkpoint(path, params, indices, actor_config, action_type, env_id):
    payload = {"actor_type": "sympol", "env_id": env_id, "action_type": action_type,
               "actor_config": actor_config, "actor": {"params": params, "indices": indices}}
    orbax.checkpoint.PyTreeCheckpointer().save(os.path.abspath(path), payload, force=True)


def mean_return(apply_actor, params, n_episodes, seed):
    """Mean and std of the undiscounted return of the deterministic (argmax)
    policy over ``n_episodes`` episodes of the plain env (500-step limit)."""
    env = gym.make(ENV_ID)
    cap = int(env.spec.max_episode_steps)
    rng = np.random.default_rng(seed)
    rets = np.empty(int(n_episodes), dtype=np.float64)
    for ep in range(int(n_episodes)):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_ret = 0.0
        for _ in range(cap):
            logits = np.asarray(apply_actor(params, np.asarray(obs, dtype=np.float32)[None, :]))[0]
            obs, reward, terminated, truncated, _ = env.step(int(np.argmax(logits)))
            ep_ret += float(reward)
            if terminated or truncated:
                break
        rets[ep] = ep_ret
    env.close()
    return float(rets.mean()), float(rets.std())


def finetune_policy(ckpt_path, regions, cfg, out_ckpt_path, seed):
    """Fine-tune the policy at ``ckpt_path`` and save it to ``out_ckpt_path``.

    ``regions``: {regions_lo, regions_hi, regions_step} from
    ``backward_propagate``. ``cfg``: the ``finetune`` settings of the run
    (see experiments/cartpole/config.toml) plus ``mix_p``, ``horizon``,
    ``x_threshold`` and ``theta_threshold``. Returns a dict of metrics.
    """
    ckpt = load_checkpoint(ckpt_path)
    actor_config = ckpt["actor_config"]
    action_type, env_id = ckpt["action_type"], ckpt["env_id"]
    spec = make_spec(cfg["x_threshold"], cfg["theta_threshold"])
    obs_dim = int(actor_config["obs_dim"])
    n_envs, T = int(cfg["n_envs"]), int(cfg["n_steps"])
    gamma, gae_lambda = float(cfg["gamma"]), float(cfg["gae_lambda"])
    clip_coef, ent_coef, vf_coef = float(cfg["clip_coef"]), float(cfg["ent_coef"]), float(cfg["vf_coef"])
    ensure_env_registered()

    envs, mixers = [], []
    for i in range(n_envs):
        mix = MixedStartWrapper(
            gym.make(env_id, max_episode_steps=_INNER_TIME_LIMIT),
            regions["regions_lo"], regions["regions_hi"], regions["regions_step"],
            mix_p=cfg["mix_p"], rng=np.random.default_rng(seed * 1000 + i),
            horizon=cfg["horizon"])
        mixers.append(mix)
        envs.append(UnsafePenaltyWrapper(mix, penalty=cfg["terminal_penalty"], spec=spec))
    steps = np.asarray(regions["regions_step"])
    print(f"[finetune] counterexample regions: {steps.shape[0]:,} across {len(mixers[0].step_groups)} "
          f"start step(s) {int(steps.min())}..{int(steps.max())}; mix_p={cfg['mix_p']} "
          f"terminal_penalty={cfg['terminal_penalty']} horizon={cfg['horizon']} n_envs={n_envs}",
          flush=True)

    actor = SYMPOL_RL(obs_dim=obs_dim, action_dim=int(actor_config["action_dim"]),
                      action_type=action_type, depth=int(actor_config["depth"]),
                      n_estimators=int(actor_config["n_estimators"]))
    critic = Critic(num_layers=int(cfg["critic_layers"]),
                    neurons_per_layer=int(cfg["critic_width"]))
    critic.apply = jax.jit(critic.apply)

    key = jax.random.PRNGKey(seed)
    key, critic_key = jax.random.split(key)
    reference_params = jax.tree.map(jnp.asarray, ckpt["params"])
    indices = jax.tree.map(jnp.asarray, ckpt["indices"])

    lr = float(cfg["learning_rate"])
    actor_tx = optax.chain(
        optax.clip_by_global_norm(float(cfg["max_grad_norm"])),
        optax.multi_transform(
            {k: (optax.adam(learning_rate=lr) if k in _ADAM_KEYS
                 else optax.adamw(learning_rate=lr)) for k in ckpt["params"]},
            lambda params: {k: k for k in params}),
    )
    actor_state = TrainState.create(apply_fn=None, tx=actor_tx,
                                    params=jax.tree.map(jnp.asarray, ckpt["params"]))
    critic_tx = optax.chain(optax.clip_by_global_norm(float(cfg["max_grad_norm"])),
                            optax.adamw(learning_rate=float(cfg["critic_lr"])))
    critic_state = TrainState.create(
        apply_fn=None, tx=critic_tx,
        params=critic.init(critic_key, jnp.zeros((1, obs_dim), dtype=jnp.float32)))

    def dist(actor_params, x):
        return distrax.Categorical(logits=actor.apply(actor_params, x, indices=indices))

    @jax.jit
    def sample_action_value(actor_params, critic_params, obs_b, sample_key):
        d = dist(actor_params, obs_b)
        value = jnp.reshape(critic.apply(critic_params, obs_b), (-1,))
        a = d.sample(seed=sample_key)
        return a, d.log_prob(a), value

    @jax.jit
    def value_only(critic_params, obs_b):
        return jnp.reshape(critic.apply(critic_params, obs_b), (-1,))

    @jax.jit
    def compute_gae(critic_params, next_obs_b, terms, truncs, vfinals, values, rewards):
        next_value = jnp.reshape(critic.apply(critic_params, next_obs_b), (-1,))
        return gae(next_value, terms, truncs, vfinals, values, rewards, gamma, gae_lambda)

    def loss_fn(actor_params, critic_params, x, a, logp_old, mb_adv, mb_ret, kl_coef):
        new_value = jnp.reshape(critic.apply(critic_params, x), (-1,))
        cur = dist(actor_params, x)
        newlogprob = cur.log_prob(a)
        entropy = cur.entropy()
        logratio = jnp.clip(newlogprob - logp_old, -20.0, 20.0)
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()
        pg_loss1 = -mb_adv * ratio
        pg_loss2 = -mb_adv * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        v_loss = 0.5 * ((new_value - mb_ret) ** 2).mean()
        ent = entropy.mean()
        # KL to the policy before fine-tuning. Its weight is 0 in the paper
        # runs; the term stays in the loss so the compiled update is the one
        # that produced the paper results.
        anchor_kl = cur.kl_divergence(dist(reference_params, x)).mean()
        total = pg_loss - ent_coef * ent + vf_coef * v_loss + kl_coef * anchor_kl
        return total, (pg_loss, v_loss, ent, jax.lax.stop_gradient(approx_kl),
                       jax.lax.stop_gradient(anchor_kl))

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

    @jax.jit
    def update_step(actor_state, critic_state, x, a, logp, adv, ret, kl_coef):
        (loss, aux), (a_grads, c_grads) = grad_fn(actor_state.params, critic_state.params,
                                                  x, a, logp, adv, ret, kl_coef)
        return (actor_state.apply_gradients(grads=a_grads),
                critic_state.apply_gradients(grads=c_grads), loss, aux)

    # --- training ---------------------------------------------------------
    obs = np.stack([np.asarray(e.reset(seed=seed + 7919 * i)[0], dtype=np.float32)
                    for i, e in enumerate(envs)])
    total_steps = int(cfg["total_steps"])
    n_epochs, minibatch = int(cfg["n_epochs"]), int(cfg["minibatch_size"])
    rng = np.random.default_rng(seed + 1)
    global_step = iteration = 0
    last_loss, last_aux = float("nan"), None
    train_returns: list[float] = []
    cur_ret = np.zeros(n_envs, dtype=np.float64)

    while global_step < total_steps:
        iteration += 1
        buf_obs = np.zeros((T, n_envs, obs_dim), dtype=np.float32)
        buf_act = np.zeros((T, n_envs), dtype=np.int32)
        buf_logp = np.zeros((T, n_envs), dtype=np.float32)
        buf_val = np.zeros((T, n_envs), dtype=np.float32)
        buf_term = np.zeros((T, n_envs), dtype=np.float32)    # per transition
        buf_trunc = np.zeros((T, n_envs), dtype=np.float32)
        buf_vfin = np.zeros((T, n_envs), dtype=np.float32)
        buf_rew = np.zeros((T, n_envs), dtype=np.float32)

        for t in range(T):
            key, sub = jax.random.split(key)
            a_jax, logp, value = sample_action_value(actor_state.params, critic_state.params,
                                                     obs, sub)
            a_np = np.asarray(a_jax)
            buf_obs[t], buf_act[t] = obs, a_np
            buf_logp[t], buf_val[t] = np.asarray(logp), np.asarray(value)
            next_obs = np.empty_like(obs)
            trunc_idx = []
            for i, e in enumerate(envs):
                nobs, reward, terminated, truncated, _ = e.step(int(a_np[i]))
                terminated = bool(terminated)
                truncated = bool(truncated) and not terminated
                buf_rew[t, i] = float(reward)
                buf_term[t, i] = float(terminated)
                buf_trunc[t, i] = float(truncated)
                cur_ret[i] += float(reward)
                next_obs[i] = np.asarray(nobs, dtype=np.float32)
                if truncated:
                    trunc_idx.append(i)
                if terminated or truncated:
                    train_returns.append(float(cur_ret[i]))
                    cur_ret[i] = 0.0
            if trunc_idx:
                # Value of the true final observation, before the reset replaces it.
                buf_vfin[t, trunc_idx] = np.asarray(
                    value_only(critic_state.params, jnp.asarray(next_obs[trunc_idx])))
            for i, e in enumerate(envs):
                if buf_term[t, i] or buf_trunc[t, i]:
                    next_obs[i] = np.asarray(e.reset()[0], dtype=np.float32)
            obs = next_obs
        global_step += T * n_envs

        adv, ret = compute_gae(critic_state.params, jnp.asarray(obs), jnp.asarray(buf_term),
                               jnp.asarray(buf_trunc), jnp.asarray(buf_vfin),
                               jnp.asarray(buf_val), jnp.asarray(buf_rew))
        flat_obs = jnp.asarray(buf_obs).reshape(-1, obs_dim)
        flat_act = jnp.asarray(buf_act).reshape(-1)
        flat_logp = jnp.asarray(buf_logp).reshape(-1)
        flat_adv = jnp.asarray(adv).reshape(-1)
        flat_ret = jnp.asarray(ret).reshape(-1)
        n_samples = flat_obs.shape[0]
        for _ in range(n_epochs):
            perm = rng.permutation(n_samples)
            n_mb = max(n_samples // minibatch, 1)
            mb_size = n_samples // n_mb
            for mi in range(n_mb):
                idx = jnp.asarray(perm[mi * mb_size:(mi + 1) * mb_size])
                actor_state, critic_state, last_loss, last_aux = update_step(
                    actor_state, critic_state, flat_obs[idx], flat_act[idx], flat_logp[idx],
                    flat_adv[idx], flat_ret[idx], jnp.float32(0.0))

        if iteration % 10 == 0 or global_step >= total_steps:
            pg, vl, ent, ppo_kl, akl = (float(x) for x in last_aux)
            print(f"[finetune] step={global_step}/{total_steps} loss={float(last_loss):.4f} "
                  f"pg={pg:.4f} v={vl:.4f} ent={ent:.4f} ppo_kl={ppo_kl:.4f} "
                  f"anchor_kl={akl:.4f}", flush=True)

    save_checkpoint(out_ckpt_path, actor_state.params, indices, dict(actor_config),
                    action_type, env_id)
    print(f"[finetune] saved to {os.path.abspath(out_ckpt_path)}", flush=True)

    # --- validation: same start states before and after -------------------
    apply_actor = jax.jit(lambda p, o: actor.apply(p, o, indices=indices))
    n_val = int(cfg["n_val_episodes"])
    after, after_std = mean_return(apply_actor, actor_state.params, n_val, seed + 12345)
    before, _ = mean_return(apply_actor, reference_params, n_val, seed + 12345)
    print(f"[finetune] val_return: before={before:.2f} after={after:.2f} "
          f"(Δ={after - before:+.2f}) std={after_std:.2f} over {n_val} eps (greedy=True)",
          flush=True)
    for e in envs:
        e.close()
    pg, vl, ent, ppo_kl, akl = (float(x) for x in last_aux)
    return {
        "mean_val_return": after, "std_val_return": after_std,
        "mean_val_return_before": before, "mean_val_return_delta": after - before,
        "n_val_episodes": n_val,
        "mean_train_return": float(np.mean(train_returns)) if train_returns else None,
        "n_train_episodes": len(train_returns),
        "final_pg_loss": pg, "final_value_loss": vl, "final_entropy": ent,
        "final_ppo_kl": ppo_kl, "final_anchor_kl": akl,
    }
