"""PPO fine-tuning of a SYMPOL policy with mixed starts.

With probability p an episode starts uniformly in a uniformly chosen counterexample region (at step 1 of
the property's horizon), otherwise from the environment's start distribution. VESPA uses p = 0.5 and
Uninformed p = 0. An episode is truncated (never terminated) when it leaves the property's horizon,
so the critic bootstraps through the cut. The critic is a fresh MLP. The actor uses SYMPOL's optimizer
structure with one learning rate per parameter group. The result is the candidate policy.
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
from flax.training.train_state import TrainState

from vespa.mountaincar import config as cfg
from vespa.mountaincar import environment, policy

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0
# The episode length is governed by MixedStartWrapper, so the registered time limit is pushed out of the way.
_INNER_TIME_LIMIT = 10 ** 9


class Critic(nn.Module):
    """Value network: MLP with ReLU hidden layers."""
    num_layers: int = cfg.CRITIC_NUM_LAYERS
    neurons_per_layer: int = cfg.CRITIC_NEURONS_PER_LAYER

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.neurons_per_layer)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x)


class MixedStartWrapper(gym.Wrapper):
    """Start episodes in a counterexample region with probability ``mix_p``.

    A counterexample region lives at step ``restart_step`` of the property's horizon, so an episode that
    starts there has HORIZON + HORIZON_MARGIN - restart_step steps left. A nominal episode starts at
    step 0. The wrapper truncates the episode when this budget is used up.
    """

    def __init__(self, env, regions_lo, regions_hi, restart_step: int, mix_p: float, rng):
        super().__init__(env)
        self.regions_lo = np.asarray(regions_lo, dtype=np.float64)
        self.regions_hi = np.asarray(regions_hi, dtype=np.float64)
        self.restart_step = int(restart_step)
        self.mix_p = float(mix_p)
        self.rng = rng
        self.step_budget = None
        self._elapsed = 0

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        start_step = 0
        if self.regions_lo.shape[0] > 0 and self.rng.random() < self.mix_p:
            i = int(self.rng.integers(0, self.regions_lo.shape[0]))
            state = np.asarray(self.rng.uniform(self.regions_lo[i], self.regions_hi[i]), dtype=np.float64)
            self.env.unwrapped.state = state
            obs = np.asarray(state, dtype=np.float32)
            start_step = self.restart_step
        self.step_budget = max(cfg.HORIZON + cfg.HORIZON_MARGIN - start_step, 1)
        self._elapsed = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed += 1
        if self._elapsed >= self.step_budget and not terminated:
            truncated = True
        return obs, reward, terminated, truncated, info


def _actor_optimizer(param_names):
    """Gradient clipping, then Adam or AdamW with the learning rate of each SYMPOL parameter group."""
    unknown = [k for k in param_names if k not in cfg.ACTOR_OPTIMIZER]
    if unknown:
        raise ValueError(f"no learning rate for actor parameter groups {unknown}")

    def group_optimizer(name):
        kind, lr = cfg.ACTOR_OPTIMIZER[name]
        return optax.adam(learning_rate=lr) if kind == "adam" else optax.adamw(learning_rate=lr)

    return optax.chain(
        optax.clip_by_global_norm(cfg.MAX_GRAD_NORM),
        optax.multi_transform({k: group_optimizer(k) for k in param_names}, lambda params: {k: k for k in params}),
    )


def finetune(ckpt_path: str, regions_lo, regions_hi, restart_step: int, mix_p: float, seed: int,
             out_path: str) -> tuple[str, dict]:
    """Fine-tune the input policy at ``ckpt_path`` and save the candidate policy at ``out_path``.

    ``regions_lo``/``regions_hi`` ([n, 2]) are the counterexample regions, ``seed`` drives the PPO run.
    Returns the absolute candidate path and a dict of metrics (greedy validation return of the candidate
    and of the input policy on the same 30 start states, training return, final entropy and PPO KL).
    """
    ckpt = policy.load_checkpoint(ckpt_path)
    actor_config, action_type, env_id = ckpt["actor_config"], ckpt["action_type"], ckpt["env_id"]
    obs_dim = int(actor_config["obs_dim"])
    action_dim = int(actor_config["action_dim"])

    # Clamp the action log-std before both the live parameters and the reference copy are taken.
    log_std_in = np.asarray(ckpt["params"]["log_std"])
    ckpt["params"]["log_std"] = jnp.minimum(jnp.asarray(log_std_in), float(cfg.LOG_STD_MAX_AT_LOAD))
    print(f"[finetune] log_std clamp at load: max {log_std_in.max():.3f} -> <= {cfg.LOG_STD_MAX_AT_LOAD:.1f}")

    environment.register()
    envs = [MixedStartWrapper(gym.make(env_id, max_episode_steps=_INNER_TIME_LIMIT), regions_lo, regions_hi,
                              restart_step, mix_p, np.random.default_rng(seed * 1000 + i))
            for i in range(cfg.N_ENVS)]
    print(f"[finetune] {np.asarray(regions_lo).shape[0]:,} counterexample regions at step {restart_step}, "
          f"mix_p={mix_p}, seed={seed}", flush=True)

    actor = policy.make_actor(actor_config, action_type)
    critic = Critic()
    critic.apply = jax.jit(critic.apply)

    key = jax.random.PRNGKey(seed)
    key, critic_key = jax.random.split(key)
    reference_params = jax.tree.map(jnp.asarray, ckpt["params"])
    indices = jax.tree.map(jnp.asarray, ckpt["indices"])

    actor_state = TrainState.create(apply_fn=None, params=jax.tree.map(jnp.asarray, ckpt["params"]),
                                    tx=_actor_optimizer(list(ckpt["params"].keys())))
    critic_tx = optax.chain(optax.clip_by_global_norm(cfg.MAX_GRAD_NORM), optax.adamw(learning_rate=cfg.CRITIC_LR))
    critic_params = critic.init(critic_key, jnp.zeros((1, obs_dim), jnp.float32))
    critic_state = TrainState.create(apply_fn=None, params=critic_params, tx=critic_tx)

    def dist(actor_params, x):
        """Gaussian policy MultivariateNormalDiag(mean, exp(log_std)) with a clipped log-std."""
        out = actor.apply(actor_params, x, indices=indices)
        return distrax.MultivariateNormalDiag(out[0], jnp.exp(jnp.clip(out[1], LOG_STD_MIN, LOG_STD_MAX)))

    @jax.jit
    def sample_action_value(actor_params, critic_params, obs_b, sample_key):
        """Raw Gaussian action (the environment clips the force), its log-probability and V(s)."""
        d = dist(actor_params, obs_b)
        value = jnp.reshape(critic.apply(critic_params, obs_b), (-1,))
        a = d.sample(seed=sample_key)
        return a, d.log_prob(a), value

    @jax.jit
    def value_only(critic_params, obs_b):
        """V(s) of the true final observation of a truncated episode."""
        return jnp.reshape(critic.apply(critic_params, obs_b), (-1,))

    @jax.jit
    def compute_gae(critic_params, next_obs_b, terms, truncs, vfinals, values, rewards):
        """GAE with terminations and truncations handled differently.

        The advantage carry is cut at every episode boundary. A terminated transition has successor value
        0, a truncated one bootstraps through ``vfinals[t]``, the value of its true final observation.
        ``terms``, ``truncs`` and ``vfinals`` are indexed by transition.
        """
        next_value = jnp.reshape(critic.apply(critic_params, next_obs_b), (-1,))
        bound = jnp.clip(terms + truncs, 0.0, 1.0)
        values_aug = jnp.concatenate([values, next_value[None, :]], axis=0)

        def body(adv, inp):
            bound_t, trunc_t, vfin_t, v_t1, v_t, r_t = inp
            nonterm = 1.0 - bound_t
            v_next = nonterm * v_t1 + trunc_t * vfin_t
            delta = r_t + cfg.GAMMA * v_next - v_t
            new_adv = delta + cfg.GAMMA * cfg.GAE_LAMBDA * nonterm * adv
            return new_adv, new_adv

        _, advantages = jax.lax.scan(
            body, jnp.zeros_like(rewards[0]),
            (bound, truncs, vfinals, values_aug[1:], values_aug[:-1], rewards), reverse=True)
        return advantages, advantages + values

    def loss_fn(actor_params, critic_params, x, a, logp_old, mb_adv, mb_ret, kl_coef):
        """Clipped PPO loss with entropy bonus, value loss and a KL term to the input policy.

        The paper configuration gives the KL term the weight ``kl_coef`` = 0 (passed at run time).
        """
        new_value = jnp.reshape(critic.apply(critic_params, x), (-1,))
        cur_dist = dist(actor_params, x)
        newlogprob = cur_dist.log_prob(a)
        entropy = cur_dist.entropy()

        logratio = jnp.clip(newlogprob - logp_old, -20.0, 20.0)
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        pg_loss1 = -mb_adv * ratio
        pg_loss2 = -mb_adv * jnp.clip(ratio, 1 - cfg.CLIP_COEF, 1 + cfg.CLIP_COEF)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        v_loss = 0.5 * ((new_value - mb_ret) ** 2).mean()
        ent_loss = entropy.mean()
        anchor_kl = cur_dist.kl_divergence(dist(reference_params, x)).mean()

        total = pg_loss - cfg.ENT_COEF * ent_loss + cfg.VF_COEF * v_loss + kl_coef * anchor_kl
        return total, (pg_loss, v_loss, ent_loss, jax.lax.stop_gradient(approx_kl), jax.lax.stop_gradient(anchor_kl))

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

    @jax.jit
    def update_step(actor_state, critic_state, x, a, logp, adv, ret, kl_coef):
        (loss, aux), (a_grads, c_grads) = grad_fn(actor_state.params, critic_state.params, x, a, logp, adv, ret,
                                                  kl_coef)
        return actor_state.apply_gradients(grads=a_grads), critic_state.apply_gradients(grads=c_grads), loss, aux

    n_envs = cfg.N_ENVS
    T = cfg.N_STEPS
    obs = np.stack([np.asarray(e.reset(seed=seed + 7919 * i)[0], dtype=np.float32) for i, e in enumerate(envs)])
    kl_coef = 0.0
    global_step = 0
    iteration = 0
    rng = np.random.default_rng(seed + 1)
    last_aux = None
    last_loss = float("nan")
    train_returns: list[float] = []
    episode_return = np.zeros(n_envs, dtype=np.float64)

    while global_step < cfg.TOTAL_STEPS:
        iteration += 1
        buf_obs = np.zeros((T, n_envs, obs_dim), dtype=np.float32)
        buf_act = np.zeros((T, n_envs, action_dim), dtype=np.float32)
        buf_logp = np.zeros((T, n_envs), dtype=np.float32)
        buf_val = np.zeros((T, n_envs), dtype=np.float32)
        buf_term = np.zeros((T, n_envs), dtype=np.float32)
        buf_trunc = np.zeros((T, n_envs), dtype=np.float32)
        buf_vfin = np.zeros((T, n_envs), dtype=np.float32)
        buf_rew = np.zeros((T, n_envs), dtype=np.float32)

        for t in range(T):
            key, sub = jax.random.split(key)
            a_jax, logp, value = sample_action_value(actor_state.params, critic_state.params, obs, sub)
            a_np = np.asarray(a_jax)
            buf_obs[t] = obs
            buf_act[t] = a_np
            buf_logp[t] = np.asarray(logp)
            buf_val[t] = np.asarray(value)

            next_obs = np.empty_like(obs)
            truncated_envs = []
            for i, e in enumerate(envs):
                nobs, reward, terminated, truncated, _ = e.step(np.asarray(a_np[i], dtype=np.float32))
                # A real termination wins over a simultaneous truncation (its successor value is 0).
                terminated = bool(terminated)
                truncated = bool(truncated) and not terminated
                buf_rew[t, i] = float(reward)
                buf_term[t, i] = float(terminated)
                buf_trunc[t, i] = float(truncated)
                episode_return[i] += float(reward)
                next_obs[i] = np.asarray(nobs, dtype=np.float32)
                if truncated:
                    truncated_envs.append(i)
                if terminated or truncated:
                    train_returns.append(float(episode_return[i]))
                    episode_return[i] = 0.0
            if truncated_envs:
                # V(s_final) before the reset below overwrites the final observation.
                vf = np.asarray(value_only(critic_state.params, jnp.asarray(next_obs[truncated_envs])))
                buf_vfin[t, truncated_envs] = vf
            for i, e in enumerate(envs):
                if buf_term[t, i] or buf_trunc[t, i]:
                    next_obs[i] = np.asarray(e.reset()[0], dtype=np.float32)
            obs = next_obs

        global_step += T * n_envs
        adv, ret = compute_gae(critic_state.params, jnp.asarray(obs), jnp.asarray(buf_term), jnp.asarray(buf_trunc),
                               jnp.asarray(buf_vfin), jnp.asarray(buf_val), jnp.asarray(buf_rew))

        flat_obs = jnp.asarray(buf_obs).reshape(-1, obs_dim)
        flat_act = jnp.asarray(buf_act).reshape(-1, action_dim)
        flat_logp = jnp.asarray(buf_logp).reshape(-1)
        flat_adv = jnp.asarray(adv).reshape(-1)
        flat_ret = jnp.asarray(ret).reshape(-1)
        n_samples = flat_obs.shape[0]
        for _ in range(cfg.N_UPDATE_EPOCHS):
            perm = rng.permutation(n_samples)
            n_mb = max(n_samples // cfg.MINIBATCH_SIZE, 1)
            mb_size = n_samples // n_mb
            for mi in range(n_mb):
                idx = jnp.asarray(perm[mi * mb_size:(mi + 1) * mb_size])
                actor_state, critic_state, last_loss, last_aux = update_step(
                    actor_state, critic_state, flat_obs[idx], flat_act[idx], flat_logp[idx], flat_adv[idx],
                    flat_ret[idx], jnp.float32(kl_coef))

        if iteration % 10 == 0 or global_step >= cfg.TOTAL_STEPS:
            pg, vl, ent, ppo_kl, _ = (float(x) for x in last_aux)
            print(f"[finetune] step={global_step}/{cfg.TOTAL_STEPS} loss={float(last_loss):.4f} pg={pg:.4f} "
                  f"v={vl:.4f} ent={ent:.4f} ppo_kl={ppo_kl:.4f}", flush=True)

    policy.save_checkpoint(out_path, params=actor_state.params, indices=indices, actor_config=dict(actor_config),
                           action_type=action_type, env_id=env_id)
    print(f"[finetune] saved to {os.path.abspath(out_path)}", flush=True)

    # Greedy validation return of the candidate and of the input policy on the same start states.
    apply_actor = jax.jit(lambda p, o: actor.apply(p, o, indices=indices))
    val_seed = seed + 12345
    mean_val, std_val = policy.greedy_return(apply_actor, actor_state.params, cfg.N_VAL_EPISODES, val_seed)
    mean_val_before, _ = policy.greedy_return(apply_actor, reference_params, cfg.N_VAL_EPISODES, val_seed)
    print(f"[finetune] val_return: before={mean_val_before:.2f} after={mean_val:.2f} std={std_val:.2f} "
          f"over {cfg.N_VAL_EPISODES} episodes", flush=True)

    _, _, entropy, ppo_kl, _ = (float(x) for x in last_aux)
    metrics = {
        "mean_val_return": mean_val,
        "std_val_return": std_val,
        "mean_val_return_before": mean_val_before,
        "mean_train_return": float(np.mean(train_returns)) if train_returns else None,
        "n_train_episodes": len(train_returns),
        "final_entropy": entropy,
        "final_ppo_kl": ppo_kl,
    }
    for e in envs:
        e.close()
    return os.path.abspath(out_path), metrics
