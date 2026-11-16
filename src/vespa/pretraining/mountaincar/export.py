"""Convert the best of the five upstream SYMPOL trials of one seed into a VESPA checkpoint.

    python -m vespa.pretraining.mountaincar.export <export_dir> <checkpoint>

``<export_dir>`` holds one directory per trial written by the export hook of the upstream SYMPOL code
(``experiments/pretraining/mountaincar/sympol_export_hook.patch``): ``actor.npz`` (parameters, path indices and the
actor's outputs on a fixed probe set) and ``meta.json``. The selected trial has the highest final test evaluation
(SYMPOL protocol: 5 greedy episodes, test seed 123456), ties broken by the last regular evaluation and then by the
lower trial number.

The checkpoint is evaluated again with ``vespa.sympol`` on the probe set. The export succeeds if
``log_std`` agrees to 1e-6 and the action means agree to 1e-6 on every probe, except for at most 10 probes that lie
on a split threshold. A probe counts as lying on a threshold if random perturbations of at most 1e-6 reproduce both
the upstream output and ours: the straight-through arithmetic of SYMPOL can route a state within about one ulp of a
threshold to either child, depending on the rounding of the einsum in a given JAX version.

Writes ``<checkpoint>`` (environment id ``VerifyMountainCarContinuousGymnax-v0``), ``<checkpoint>/export.json``
(selection and probe comparison) and ``<checkpoint>/ckpt_eval.json``.
"""
import glob
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from vespa.pretraining.checkpoint import load_actor, save_checkpoint
from vespa.pretraining.envs import MOUNTAINCAR_ENV_ID

TOLERANCE = 1e-6
MAX_THRESHOLD_TIES = 10


def selection_key(trial):
    """Sort key of a trial: test evaluation, then final evaluation, then the lower trial number."""
    test_eval, final_eval = trial.get("test_eval"), trial.get("final_eval")
    return (-np.inf if test_eval is None else test_eval, -np.inf if final_eval is None else final_eval,
            -int(trial["trial"]))


def classify_mismatches(apply, mismatches, probe_obs, probe_mean, our_mean):
    """Split the mismatching probes (the first 50) into threshold ties and real differences."""
    rng = np.random.default_rng(0)
    ties, non_ties = [], []
    for i in mismatches[:50]:
        obs = probe_obs[i].astype(np.float64)
        perturbed = (obs[None] + rng.uniform(-1e-6, 1e-6, (4000, obs.shape[0]))).astype(np.float32)
        perturbed_mean = np.asarray(apply(jnp.asarray(perturbed))[0])
        hits_upstream = int(np.sum(np.abs(perturbed_mean - probe_mean[i]).max(-1) <= TOLERANCE))
        hits_ours = int(np.sum(np.abs(perturbed_mean - our_mean[i]).max(-1) <= TOLERANCE))
        (ties if (hits_upstream > 0 and hits_ours > 0) else non_ties).append(
            {"probe": int(i), "obs": probe_obs[i].tolist(), "diff": float(np.abs(our_mean[i] - probe_mean[i]).max()),
             "hits_paper": hits_upstream, "hits_ours": hits_ours})
    return ties, non_ties


def main(src, out):
    trials = []
    for meta_path in sorted(glob.glob(os.path.join(src, "*", "meta.json"))):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["_dir"] = os.path.dirname(meta_path)
        trials.append(meta)
    if not trials:
        raise SystemExit(f"no trials under {src}")
    best = max(trials, key=selection_key)
    exported = np.load(os.path.join(best["_dir"], "actor.npz"))
    params = {k.split("/", 1)[1]: jnp.asarray(exported[k]) for k in exported.files if k.startswith("params/")}
    indices = {k.split("/", 1)[1]: jnp.asarray(exported[k]) for k in exported.files if k.startswith("indices/")}
    save_checkpoint(out, params, indices, dict(best["actor_config"]), best["action_type"], MOUNTAINCAR_ENV_ID)

    _, actor, saved_params, saved_indices = load_actor(out)
    apply = jax.jit(lambda o: actor.apply(saved_params, o, indices=saved_indices))
    probe_obs, probe_mean = exported["probe/obs"], exported["probe/mean"]
    result = apply(jnp.asarray(probe_obs))
    our_mean = np.asarray(result[0])
    mean_diff = np.abs(our_mean - probe_mean).max(-1)
    mismatches = np.flatnonzero(mean_diff > TOLERANCE)
    ties, non_ties = classify_mismatches(apply, mismatches, probe_obs, probe_mean, our_mean)
    log_std_diff = float(np.max(np.abs(np.asarray(result[1]) - exported["probe/log_std"])))
    clipped_equal = bool(np.array_equal(np.clip(our_mean, -1, 1), np.clip(probe_mean, -1, 1)))
    ok = (log_std_diff <= TOLERANCE and mismatches.size <= MAX_THRESHOLD_TIES and not non_ties
          and len(ties) == int(mismatches.size))
    trial_keys = ("trial", "run_name", "test_eval", "final_eval", "global_step", "env_seed", "seed_training",
                  "backend", "jax_version")
    record = {
        "src": src, "out": os.path.abspath(out), "env_id": MOUNTAINCAR_ENV_ID,
        "selected_trial": int(best["trial"]), "selected_run": best["run_name"],
        "selection_key": "test_eval, then final_eval, then lower trial",
        "trials": [{k: t.get(k) for k in trial_keys} for t in trials],
        "equivalence": {"max_abs_mean_diff": float(np.max(mean_diff)), "max_abs_log_std_diff": log_std_diff,
                        "clipped_actions_identical": clipped_equal, "n_probe": int(probe_obs.shape[0]),
                        "n_mismatch": int(mismatches.size), "n_ties": len(ties), "ties": ties,
                        "non_ties": non_ties, "ok": bool(ok)},
        "selected_meta": {k: v for k, v in best.items() if k != "_dir"},
    }
    with open(os.path.join(out, "export.json"), "w") as f:
        json.dump(record, f, indent=1)
    with open(os.path.join(out, "ckpt_eval.json"), "w") as f:
        json.dump({"best_eval": best.get("test_eval"), "final_eval": best.get("final_eval"),
                   "trial": int(best["trial"]), "global_step": best.get("global_step"),
                   "source": "upstream SYMPOL ppo_gymnax (best of 5 trials by test evaluation)"}, f)
    print(f"[export] {src} -> {out}: trial {best['trial']} test_eval={best.get('test_eval')} "
          f"final_eval={best.get('final_eval')} | max|dmean|={np.max(mean_diff):.2e} "
          f"max|dlogstd|={log_std_diff:.2e} clipped_identical={clipped_equal} mismatches={int(mismatches.size)} "
          f"(threshold ties {len(ties)}) -> {'OK' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
