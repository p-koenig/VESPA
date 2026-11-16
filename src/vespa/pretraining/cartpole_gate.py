"""Pruning and acceptance check of one CartPole input policy.

    python -m vespa.pretraining.cartpole_gate <run_dir> <seed>

``<run_dir>/checkpoint`` is the best checkpoint written by ``vespa.pretraining.ppo``. The check

1. prunes it to ``<run_dir>/checkpoint-pruned`` (``vespa.pretraining.pruning``),
2. compares the greedy actions of the pruned and the unpruned tree on 500,000 uniform states of the safe box and
   200,000 uniform states of a box twice as large (the pruned tree must agree everywhere),
3. evaluates both trees on 30 greedy validation episodes (validation seed ``seed + 12345``).

Writes ``<run_dir>/gate.json`` and exits with status 5 if the pruned tree disagrees anywhere.
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from vespa.pretraining.checkpoint import load_actor, rollout_mean_return
from vespa.pretraining.pruning import prune_checkpoint

N_VALIDATION_EPISODES = 30
SAFE_BOX = np.array([2.4, 3.0, 0.20943951, 3.5])
WIDE_BOX = np.array([4.8, 10, 0.4189, 10])


def greedy_policy(path):
    """Load a checkpoint and return ``(apply, params, act)``.

    ``apply(params, observations)`` gives the logits, ``act(states)`` the greedy actions as a NumPy array.
    """
    _, actor, params, indices = load_actor(path)
    apply = jax.jit(lambda p, o: actor.apply(p, o, indices=indices))
    greedy = jax.jit(lambda o: jnp.argmax(actor.apply(params, o, indices=indices), axis=-1))
    return apply, params, (lambda states: np.asarray(greedy(jnp.asarray(states, jnp.float32))))


def main():
    run_dir, seed = os.path.abspath(sys.argv[1]), int(sys.argv[2])
    checkpoint = os.path.join(run_dir, "checkpoint")
    pruned, leaves = prune_checkpoint(checkpoint)

    rng = np.random.default_rng(0)
    safe_states = (rng.uniform(-1, 1, size=(500000, 4)) * SAFE_BOX).astype(np.float32)
    wide_states = (rng.uniform(-1, 1, size=(200000, 4)) * WIDE_BOX).astype(np.float32)
    apply_unpruned, params_unpruned, act_unpruned = greedy_policy(checkpoint)
    apply_pruned, params_pruned, act_pruned = greedy_policy(pruned)
    disagreement = {
        "disagree_safe_box": float((act_unpruned(safe_states) != act_pruned(safe_states)).mean()),
        "disagree_wide_box": float((act_unpruned(wide_states) != act_pruned(wide_states)).mean()),
    }
    print("[equiv]", disagreement, flush=True)

    validation = {}
    for name, apply, params in (("unpruned", apply_unpruned, params_unpruned), ("pruned", apply_pruned, params_pruned)):
        mean, std, n = rollout_mean_return(apply, params, N_VALIDATION_EPISODES, seed + 12345)
        validation[name] = dict(val_mean=float(mean), val_std=float(std), n=n, val_seed=seed + 12345)
        print("[val]", name, validation[name], flush=True)

    with open(os.path.join(checkpoint, "training.json")) as f:
        training = json.load(f)
    with open(os.path.join(checkpoint, "ckpt_eval.json")) as f:
        checkpoint_eval = json.load(f)
    record = dict(seed=seed, ckpt=checkpoint, ckpt_pruned=pruned, pruned_leaves=leaves[0], equivalence=disagreement,
                  val=validation, global_step=training["global_step"], wall_time_sec=training["wall_time_sec"],
                  episodes=training["episode_terminations_total"], hyperparameters=training["hyperparameters"],
                  ckpt_eval=checkpoint_eval)
    with open(os.path.join(run_dir, "gate.json"), "w") as f:
        json.dump(record, f, indent=1)
    ok = disagreement["disagree_safe_box"] == 0.0 and disagreement["disagree_wide_box"] == 0.0
    print("[gate]", "OK" if ok else "FAIL: the pruned tree disagrees with the unpruned tree", flush=True)
    sys.exit(0 if ok else 5)


if __name__ == "__main__":
    main()
