"""Test of the acceptance rule of the loop with stub verifier, extraction and fine-tuning.

A scripted sequence of verification outcomes checks the decisions (base, accept, reject with reason),
the incumbent, the fine-tuning seed (seed + 1000 * rejects), the policy fine-tuned and the final status.

Run on CPU from the repository root:  JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest tests/mountaincar
(or run this file directly).
"""
import json
import os
import tempfile

import numpy as np

from vespa.mountaincar import loop, verifier


def outcome(cert):
    """Stub verification result: 'safe', 'undecided' or a ratio of safe start state regions."""
    safe = cert == "safe"
    violated = not safe and cert != "undecided"
    return verifier.Verification(
        safe=safe, status="SOUNDNESS_LOST" if cert == "undecided" else "OK", violated=violated,
        certified_fraction=1.0 if safe else (cert if violated else None), n_init_pieces=4, n_violating_pieces=1,
        n_violating_objects=3 if violated else 0, peak=10, cap=262144, cap_step=None, steps_run=200,
        n_per_step=[4], capture={}, piece_lo=np.zeros((4, 2)), piece_hi=np.zeros((4, 2)),
        piece_violating=np.zeros(4, bool))


def run_script(script, tmp):
    """Run the loop on a stub policy whose k-th verification returns script[k - 1]."""
    calls = {"verify": [], "finetune": [], "regions": []}

    def verify_policy(ckpt, alive):
        calls["verify"].append(ckpt)
        return outcome(script[len(calls["verify"]) - 1]), 0.0, 0

    def regions(result, seed, log):
        calls["regions"].append(len(calls["verify"]))
        n = len(calls["verify"])
        return {"lo": np.zeros((n, 2), np.float32), "hi": np.zeros((n, 2), np.float32), "n_rows": n,
                "n_dropped": 0, "n_piece_fallback": 0}

    def finetune(ckpt, lo, hi, restart_step, mix_p, seed, out):
        calls["finetune"].append((os.path.basename(ckpt), seed, lo.shape[0]))
        metrics = {key: 0.0 for key in loop.FT_FIELDS}
        metrics.update(mean_val_return=50.0, std_val_return=1.0)
        return f"cand{len(calls['finetune'])}", metrics

    saved = (loop.verify_policy, loop.counterexample_regions, loop.finetune, loop.policy.checkpoint_greedy_return)
    loop.verify_policy, loop.counterexample_regions, loop.finetune = verify_policy, regions, finetune
    loop.policy.checkpoint_greedy_return = lambda path, n, seed: (50.0, 1.0)
    try:
        summary = loop.run("input", "vespa", 7, os.path.join(tmp, "run"))
    finally:
        loop.verify_policy, loop.counterexample_regions, loop.finetune, loop.policy.checkpoint_greedy_return = saved
    assert json.load(open(os.path.join(tmp, "run", "summary.json")))["per_seed"][0]["status"] == \
        summary["per_seed"][0]["status"]
    return summary["per_seed"][0], calls


def test_accept_reject_rule():
    with tempfile.TemporaryDirectory() as tmp:
        s, calls = run_script([0.5, 0.7, 0.6, 0.7, "undecided", 0.8], tmp)
    decisions = [(r["inc_decision"], r["inc_reason"]) for r in s["iters"]]
    assert decisions == [("base", "base"), ("accept", "higher"), ("reject", "lower"), ("reject", "tie"),
                         ("reject", "undecided"), ("accept", "higher")]
    assert s["status"] == "STUCK_MAX_ITERS" and s["n_iters"] == 6
    assert s["n_rejects"] == 3 and s["n_accepts"] == 2
    assert s["incumbent_iter"] == 6 and s["certified_fraction_incumbent"] == 0.8 and s["best_iter"] == 6
    # Fine-tuning: always from the incumbent, seed + 1000 * rejects, counterexamples of the incumbent.
    assert [f[1] for f in calls["finetune"]] == [7, 7, 1007, 2007, 3007]
    assert [r["ft_from_iter"] for r in s["iters"][:5]] == [1, 2, 2, 2, 2]
    assert calls["regions"] == [1, 2]
    assert [f[2] for f in calls["finetune"]] == [1, 2, 2, 2, 2]
    assert s["restart_steps"] == [1, 1, 1, 1, 1, None]


def test_stops_at_first_proof():
    with tempfile.TemporaryDirectory() as tmp:
        s, calls = run_script([0.5, 0.4, "safe"], tmp)
    assert s["status"] == "SAFE" and s["n_iters"] == 3
    assert [r["inc_decision"] for r in s["iters"]] == ["base", "reject", "accept"]
    assert s["certified_fraction_incumbent"] == 1.0 and s["certified_fraction_best"] == 1.0


def test_undecided_input_policy_ends_the_run():
    with tempfile.TemporaryDirectory() as tmp:
        s, calls = run_script(["undecided"], tmp)
    assert s["status"] == "NO_COUNTEREXAMPLE_SOUNDNESS_LOST" and s["n_iters"] == 1 and not calls["finetune"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name, flush=True)
