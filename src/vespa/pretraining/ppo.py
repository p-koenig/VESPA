"""PPO training of the CartPole SYMPOL input policies (one depth-7 tree, SYMPOL's CartPole hyperparameters).

    python -m vespa.pretraining.ppo --seed 6 --out checkpoints/cartpole/s6 [--total-steps 50000000]

The training loop follows SYMPOL (Marton et al.), which builds on CleanRL's JAX PPO
(https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py). Specifics:

* Actor: a SYMPOL tree (``vespa/sympol.py``). Critic: an MLP with two hidden layers of 256 ReLU units.
* Optimisers: Adam for the tree weights and thresholds, AdamW for feature-selection logits, leaves and ``log_std``,
  then SWAG averaging. Every evaluation feeds the mean evaluation return into a reduce-on-plateau schedule that
  scales all actor learning rates.
* Rollouts of 512 steps in 7 environments. The gradient-accumulation factor doubles in each eighth of training
  (1, 2, 4, ..., 128).
* Evaluation (5 greedy episodes) at the first iteration, after every ``--eval-freq`` environment steps and at the
  end. The best evaluation so far (ties go to the later policy) is saved to ``<out>/checkpoint``, together with
  ``ckpt_eval.json`` (its evaluation) and, at the end, ``training.json`` (steps, wall time, episodes,
  hyperparameters).
"""
import argparse
import json
import os
import time
from functools import partial

import distrax
import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from optax_swag import swag

from vespa.pretraining.envs import CARTPOLE_ENV_ID, make_cartpole
from vespa.sympol import SYMPOL_RL

# On a GPU, allocate memory on demand instead of reserving most of the device at start (setting of the paper runs).
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

# SYMPOL's CartPole hyperparameters (the paper's input policies use exactly these).
DEPTH = 7
N_ESTIMATORS = 1
N_ENVS = 7
N_STEPS = 512
MINIBATCH_SIZE = 64
N_UPDATE_EPOCHS = 7
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.1
ENT_COEF = 0.2
VF_COEF = 0.5
MAX_GRAD_NORM = 1000
LEARNING_RATE_CRITIC = 0.0013329992676131342
LEARNING_RATES_ACTOR = {
    "estimator_weights": 0.04760308580677977,
    "split_values": 0.000222274485191996,
    "split_idx_array": 0.025528008432059508,
    "leaf_array": 0.019530943718321373,
    "log_std": 0.0012313062437960766,
}
ADAMW_PARAMETERS = ("split_idx_array", "leaf_array", "log_std")
N_EVAL_EPISODES = 5
TEST_EVAL_SEED = 123456


class Critic(nn.Module):
    """Value function: MLP with two hidden layers of 256 ReLU units."""

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x)


class ActorTrainState(TrainState):
    """Actor state with the accumulated gradient and the fixed path indices of the tree."""

    grad_accum: jnp.ndarray
    indices: dict


@flax.struct.dataclass
class Storage:
    """One rollout, indexed (step, environment)."""

    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    """Running and last completed episode return and length per training environment (for the progress log)."""

    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


def actor_optimizer():
    """Gradient clipping, one Adam/AdamW per parameter group with an adjustable learning rate, then SWAG."""
    def optimizer(name):
        adam = optax.adamw if name in ADAMW_PARAMETERS else optax.adam
        return optax.inject_hyperparams(adam)(LEARNING_RATES_ACTOR[name])

    return optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.multi_transform({name: optimizer(name) for name in LEARNING_RATES_ACTOR},
                              lambda params: {name: name for name in params}),
        swag(10, 2),
    )


def accumulation_factor(global_step, total_steps):
    """Gradient-accumulation factor: 2 ** (k - 1) in the k-th eighth of training."""
    return int(2**(np.ceil((((global_step+1)*8)/(1+total_steps)))-1))


def evaluate(actor, actor_state, n_episodes, seed):
    """Returns of ``n_episodes`` greedy episodes, episode ``i`` reset with seed ``seed + i``."""
    env = gym.make(CARTPOLE_ENV_ID)
    scores = []
    for episode_index in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode_index)
        done, trunc = False, False
        running_reward = 0
        while not done and not trunc:
            action_logits = actor.apply(actor_state.params, np.array([obs]), indices=actor_state.indices)
            action = jnp.squeeze(jnp.argmax(action_logits, axis=1), axis=0)
            obs, reward, done, trunc, _ = env.step(jax.device_get(action))
            running_reward += reward
        scores.append(running_reward)
    env.close()
    return scores


def train(seed, total_steps, eval_freq, out):
    """Train one input policy and write ``<out>/checkpoint`` (see the module docstring)."""
    start_time = time.time()
    checkpoint_path = os.path.abspath(os.path.join(out, "checkpoint"))
    envs = gym.vector.AsyncVectorEnv([make_cartpole for _ in range(N_ENVS)], context="spawn")
    obs_dim = envs.single_observation_space.shape[-1]
    action_dim = envs.single_action_space.n
    sample_observation_batch = jnp.zeros((1, obs_dim), dtype=jnp.float32)

    env_seed = seed + 100
    key = jax.random.PRNGKey(env_seed)
    _, actor_key, critic_key = jax.random.split(jax.random.PRNGKey(env_seed), 3)

    critic = Critic()
    critic.apply = jax.jit(critic.apply)
    actor = SYMPOL_RL(obs_dim=obs_dim, action_dim=action_dim, action_type="discrete", depth=DEPTH,
                      n_estimators=N_ESTIMATORS)
    critic_state = TrainState.create(
        apply_fn=None,
        params=critic.init(critic_key, sample_observation_batch),
        tx=optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adamw(learning_rate=LEARNING_RATE_CRITIC)),
    )
    actor_state = ActorTrainState.create(
        apply_fn=None,
        params=actor.init(actor_key, sample_observation_batch),
        tx=actor_optimizer(),
        grad_accum=jax.tree.map(jnp.zeros_like, actor.init(actor_key, sample_observation_batch)),
        indices=actor.init_indices(actor_key),
    )
    lr_scheduler = optax.contrib.reduce_on_plateau(patience=3, factor=0.5)
    lr_scheduler_state = lr_scheduler.init(actor_state.params)
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(N_ENVS, dtype=jnp.float32),
        episode_lengths=jnp.zeros(N_ENVS, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(N_ENVS, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(N_ENVS, dtype=jnp.int32),
    )

    @jax.jit
    def get_action_and_value(actor_state, critic_state, next_obs, next_done, storage, step, key):
        """Sample an action, compute value and log-probability, and store them at ``step``."""
        action_logits = actor.apply(actor_state.params, next_obs, indices=actor_state.indices)
        action_distribution = distrax.Categorical(logits=action_logits)
        value = jnp.reshape(critic.apply(critic_state.params, next_obs), (-1,))
        key, subkey = jax.random.split(key)
        action = action_distribution.sample(seed=subkey)
        logprob = action_distribution.log_prob(action)
        storage = storage.replace(
            obs=storage.obs.at[step].set(next_obs),
            dones=storage.dones.at[step].set(next_done),
            actions=storage.actions.at[step].set(action),
            logprobs=storage.logprobs.at[step].set(logprob),
            values=storage.values.at[step].set(value),
        )
        return storage, action, key

    @jax.jit
    def get_action_and_value2(actor_state_params, critic_state_params, x, action):
        """Log-probability of ``action``, entropy and value under the given parameters."""
        logits = actor.apply(actor_state_params, x, indices=actor_state.indices)
        value = jnp.reshape(critic.apply(critic_state_params, x), (-1,))
        action_distribution = distrax.Categorical(logits=logits)
        return action_distribution.log_prob(action), action_distribution.entropy(), value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone
        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(compute_gae_once, gamma=GAMMA, gae_lambda=GAE_LAMBDA)

    @jax.jit
    def compute_gae(critic_state, next_obs, next_done, storage):
        """Generalised advantage estimates and returns of the rollout."""
        next_value = jnp.reshape(critic.apply(critic_state.params, next_obs), (-1,))
        advantages = jnp.zeros((N_ENVS,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        return storage.replace(advantages=advantages, returns=advantages + storage.values)

    @jax.jit
    def ppo_loss(actor_state_params, critic_state_params, x, a, logp, mb_advantages, mb_returns):
        """Clipped PPO objective with entropy bonus and value loss."""
        newlogprob, entropy, newvalue = get_action_and_value2(actor_state_params, critic_state_params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, argnums=(0, 1), has_aux=True)

    @jax.jit
    def update_ppo(actor_state, critic_state, storage, key, accumulate_gradients_every):
        """``N_UPDATE_EPOCHS`` epochs over shuffled minibatches of the rollout.

        The critic takes one AdamW step per minibatch. The actor takes one step with the minibatch gradient and,
        every ``accumulate_gradients_every`` actor steps, one more step with the mean of the gradients accumulated
        since the last such step.
        """
        def update_epoch(carry, unused_inp):
            actor_state, critic_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # Shuffling as in https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                num_minibatches = int(np.floor(x.shape[0] / MINIBATCH_SIZE))
                size = num_minibatches * MINIBATCH_SIZE
                x = jax.random.permutation(subkey, x)[:size]
                x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree.map(flatten, storage)
            shuffled_storage = jax.tree.map(convert_data, flatten_storage)

            def update_minibatch(carry, minibatch):
                actor_state, critic_state = carry
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), (actor_grads, critic_grads) = ppo_loss_grad_fn(
                    actor_state.params,
                    critic_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                )
                critic_state = critic_state.apply_gradients(grads=critic_grads)
                actor_grad_accum = jax.tree_util.tree_map(lambda x, y: x + y, actor_grads, actor_state.grad_accum)
                actor_state = actor_state.apply_gradients(grads=actor_grads)

                def update_fn():
                    grads = jax.tree_util.tree_map(lambda x: x / accumulate_gradients_every, actor_grad_accum)
                    return actor_state.apply_gradients(
                        grads=grads,
                        grad_accum=jax.tree_util.tree_map(jnp.zeros_like, grads),
                    )

                actor_state = jax.lax.cond(
                    actor_state.step % accumulate_gradients_every == 0,
                    lambda _: update_fn(),
                    lambda _: actor_state.replace(grad_accum=actor_grad_accum, step=actor_state.step + 1),
                    None,
                )
                return (actor_state, critic_state), (loss, pg_loss, v_loss, entropy_loss, approx_kl)

            (actor_state, critic_state), losses = jax.lax.scan(
                update_minibatch, (actor_state, critic_state), shuffled_storage
            )
            return (actor_state, critic_state, key), losses

        (actor_state, critic_state, key), losses = jax.lax.scan(
            update_epoch, (actor_state, critic_state, key), (), length=N_UPDATE_EPOCHS
        )
        return actor_state, critic_state, losses, key

    def rollout(actor_state, critic_state, episode_stats, next_obs, next_done, storage, key, global_step):
        """Collect ``N_STEPS`` steps in each training environment."""
        for step in range(0, N_STEPS):
            global_step += N_ENVS
            storage, action, key = get_action_and_value(
                actor_state, critic_state, next_obs, next_done, storage, step, key
            )
            next_obs, reward, next_done, trunc, _ = envs.step(jax.device_get(action))
            new_episode_return = episode_stats.episode_returns + reward
            new_episode_length = episode_stats.episode_lengths + 1
            episode_stats = episode_stats.replace(
                episode_returns=(new_episode_return) * (1 - next_done) * (1 - trunc),
                episode_lengths=(new_episode_length) * (1 - next_done) * (1 - trunc),
                returned_episode_returns=jnp.where(
                    next_done + trunc, new_episode_return, episode_stats.returned_episode_returns),
                returned_episode_lengths=jnp.where(
                    next_done + trunc, new_episode_length, episode_stats.returned_episode_lengths),
            )
            storage = storage.replace(rewards=storage.rewards.at[step].set(reward))
        return episode_stats, next_obs, next_done, storage, key, global_step

    batch_size = N_ENVS * N_STEPS
    global_step = 0
    next_obs, _ = envs.reset(seed=env_seed)
    next_done = np.zeros(N_ENVS).astype(bool)
    best_eval_score = float("-inf")
    last_eval = 0
    iteration = 1
    episode_terminations_total = 0
    train_start = time.time()

    while global_step < total_steps:
        accumulate_gradients_every = accumulation_factor(global_step, total_steps)
        current_eval = global_step // eval_freq
        storage = Storage(
            obs=jnp.zeros((N_STEPS, N_ENVS) + envs.single_observation_space.shape),
            actions=jnp.zeros((N_STEPS, N_ENVS) + envs.single_action_space.shape, dtype=jnp.int32),
            logprobs=jnp.zeros((N_STEPS, N_ENVS)),
            dones=jnp.zeros((N_STEPS, N_ENVS)),
            values=jnp.zeros((N_STEPS, N_ENVS)),
            advantages=jnp.zeros((N_STEPS, N_ENVS)),
            returns=jnp.zeros((N_STEPS, N_ENVS)),
            rewards=jnp.zeros((N_STEPS, N_ENVS)),
        )
        episode_stats, next_obs, next_done, storage, key, global_step = rollout(
            actor_state, critic_state, episode_stats, next_obs, next_done, storage, key, global_step
        )
        episode_terminations_total += int(np.sum(jax.device_get(storage.dones)))
        storage = compute_gae(critic_state, next_obs, next_done, storage)
        actor_state, critic_state, losses, key = update_ppo(
            actor_state, critic_state, storage, key, accumulate_gradients_every
        )
        check_finite(iteration, global_step, storage, losses, actor_state, critic_state)
        final_iteration = global_step + batch_size >= total_steps

        if iteration == 1 or current_eval > last_eval or final_iteration:
            last_eval = current_eval
            if iteration == 1:
                # The input policies of the paper were trained with JAX switching to 64-bit mode here, at the first
                # evaluation, so all later rollouts and updates use 64-bit defaults (for example the float64
                # rollout storage). Kept to reproduce them exactly.
                jax.config.update("jax_enable_x64", True)
            avg_score = np.mean(evaluate(actor, actor_state, N_EVAL_EPISODES, env_seed)).item()
            _, lr_scheduler_state = lr_scheduler.update(
                updates=actor_state.params, state=lr_scheduler_state, value=avg_score
            )
            inner_states = actor_state.opt_state[1][0]
            for name, learning_rate in LEARNING_RATES_ACTOR.items():
                inner_states[name][0].hyperparams["learning_rate"] = learning_rate * lr_scheduler_state.scale
            print(f"[eval] step={global_step}/{total_steps} | ret={avg_score:.2f}")
            if final_iteration:
                avg_score_test = np.mean(evaluate(actor, actor_state, N_EVAL_EPISODES, TEST_EVAL_SEED)).item()
                print(f"[test-eval] step={global_step}/{total_steps} | ret={avg_score_test:.2f}")
            if avg_score >= best_eval_score:
                best_eval_score = avg_score
                print(f"[ckpt] step={global_step} | saved best eval={best_eval_score:.2f}")
                save_policy(checkpoint_path, actor_state, obs_dim, action_dim)
                with open(os.path.join(checkpoint_path, "ckpt_eval.json"), "w") as f:
                    json.dump({"best_eval": float(best_eval_score), "global_step": int(global_step)}, f)
            loss, pg_loss, v_loss, entropy_loss, approx_kl = (float(np.mean(x[-1])) for x in losses)
            print(f"iter={iteration} | step={global_step}/{total_steps} | "
                  f"ret={float(np.mean(jax.device_get(episode_stats.returned_episode_returns))):.2f} | "
                  f"vloss={v_loss:.4f} | ploss={pg_loss:.4f} | ent={entropy_loss:.4f} | kl={approx_kl:.4f} | "
                  f"time={time.time() - start_time:.1f}s", flush=True)
        iteration += 1

    envs.close()
    summary = {
        "global_step": int(global_step),
        "wall_time_sec": float(time.time() - train_start),
        "episode_terminations_total": int(episode_terminations_total),
        "hyperparameters": hyperparameters(seed, total_steps, eval_freq),
    }
    with open(os.path.join(checkpoint_path, "training.json"), "w") as f:
        json.dump(summary, f, indent=1)


def save_policy(path, actor_state, obs_dim, action_dim):
    """Write the actor of ``actor_state`` as a checkpoint (format: ``checkpoint.py``)."""
    # Imported here and not at the top: importing the checkpoint module switches JAX to 64-bit mode, which training
    # does only at its first evaluation (before the first checkpoint is written).
    from vespa.pretraining.checkpoint import save_checkpoint

    actor_config = {"obs_dim": int(obs_dim), "action_dim": int(action_dim), "depth": DEPTH,
                    "n_estimators": N_ESTIMATORS}
    save_checkpoint(path, actor_state.params, actor_state.indices, actor_config, "discrete", CARTPOLE_ENV_ID)


def check_finite(iteration, global_step, storage, losses, actor_state, critic_state):
    """Stop training with an error when the rollout, the losses or the parameters contain NaN or inf."""
    trees = {"storage": storage, "losses": losses, "actor": actor_state.params, "critic": critic_state.params}
    bad = [f"{name}{jax.tree_util.keystr(path)}"
           for name, tree in trees.items()
           for path, leaf in jax.tree_util.tree_leaves_with_path(tree)
           if not np.isfinite(np.asarray(jax.device_get(leaf), dtype=np.float64)).all()]
    if bad:
        raise FloatingPointError(f"non-finite values at iteration {iteration}, step {global_step}: {', '.join(bad)}")


def hyperparameters(seed, total_steps, eval_freq):
    """The settings of a run, as recorded in ``training.json``."""
    return {
        "env_id": CARTPOLE_ENV_ID, "seed": seed, "total_steps": total_steps, "eval_freq": eval_freq,
        "depth": DEPTH, "n_estimators": N_ESTIMATORS, "n_envs": N_ENVS, "n_steps": N_STEPS,
        "minibatch_size": MINIBATCH_SIZE, "n_update_epochs": N_UPDATE_EPOCHS, "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA, "clip_coef": CLIP_COEF, "ent_coef": ENT_COEF, "vf_coef": VF_COEF,
        "max_grad_norm": MAX_GRAD_NORM, "learning_rate_critic": LEARNING_RATE_CRITIC,
        "learning_rates_actor": LEARNING_RATES_ACTOR, "n_eval_episodes": N_EVAL_EPISODES,
    }


def main():
    parser = argparse.ArgumentParser(description="Train one CartPole SYMPOL input policy with PPO.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True, help="output directory (the checkpoint goes to <out>/checkpoint)")
    parser.add_argument("--total-steps", type=int, default=50_000_000, help="environment steps (paper: 50,000,000)")
    parser.add_argument("--eval-freq", type=int, default=1_000_000,
                        help="environment steps between evaluations (paper: 1,000,000)")
    args = parser.parse_args()
    print(f"[ppo] env={CARTPOLE_ENV_ID} | steps={args.total_steps} | seed={args.seed}", flush=True)
    train(args.seed, args.total_steps, args.eval_freq, args.out)


if __name__ == "__main__":
    main()
