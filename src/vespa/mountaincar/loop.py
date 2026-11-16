"""The VESPA loop for MountainCarContinuous: one input policy, one method, one fine-tuning seed.

1. Verify the input policy (the first incumbent).
2. Unless the incumbent is proven safe: extract counterexample regions from its verification, fine-tune
   the incumbent (VESPA starts half of the episodes in these regions, Uninformed none) and verify the
   candidate policy.
3. Accept the candidate if it is proven safe or its ratio of safe start state regions is strictly higher
   than the incumbent's. Otherwise reject it and fine-tune the incumbent again with
   seed + 1000 * (number of rejects).
4. Stop at the first proof or after MAX_VERIFICATIONS verifications (the input policy included).

Results: ``<out>/<run_name>/summary.json`` with one record per verification under ``per_seed[0].iters``,
and the candidate policies under ``seed<seed>/iter_<k>/candidate``.
"""
from __future__ import annotations

import json
import os
import time
import traceback

import numpy as np

from vespa.mountaincar import config as cfg
from vespa.mountaincar import policy, verifier
from vespa.mountaincar.counterexamples import RESTART_STEP, CounterexampleExtractionFailed, counterexample_regions
from vespa.mountaincar.finetune import finetune

# Greedy validation return at or below which a policy never reaches the goal: a goal episode pays at least
# 100 - 0.1 * 999 > 0, an episode without the goal at most 0.
DEAD_RETURN = 0.0
# Fine-tuning metrics stored with the iteration the fine-tuning started from.
FT_FIELDS = ("mean_val_return", "mean_val_return_before", "mean_train_return", "n_train_episodes", "final_entropy",
             "final_ppo_kl")


def verify_policy(ckpt_path: str, alive: bool) -> tuple[verifier.Verification, float, int]:
    """Verify a policy, with a larger buffer when the buffer overflows on a policy that reaches the goal.

    A policy that never reaches the goal (``alive`` False) is not escalated: its population grows
    geometrically and its verification has no use. Returns ``(result, seconds, n_escalations)``.
    """
    t0 = time.time()
    regions = policy.leaf_regions(policy.load_checkpoint(ckpt_path))
    result = verifier.verify_regions(regions, cfg.ZONOTOPE_CAP)
    n_escalations = 0
    if result.status == "SOUNDNESS_LOST" and alive:
        for cap in cfg.ESCALATION_CAPS:
            print(f"[loop]   buffer overflow at cap {result.cap:,} -> verify again with cap {cap:,}", flush=True)
            result = verifier.verify_regions(regions, cap)
            n_escalations += 1
            if result.status != "SOUNDNESS_LOST":
                break
    return result, time.time() - t0, n_escalations


def _decide(k: int, result: verifier.Verification, incumbent: dict | None) -> tuple[str, str]:
    """Acceptance rule: (decision, reason) for verification ``k`` given the incumbent."""
    if k == 1:
        return "base", ("safe" if result.safe else "undecided" if not result.violated else "base")
    if result.safe:
        return "accept", "safe"
    if not result.violated:
        return "reject", "undecided"
    if result.certified_fraction > incumbent["cert"]:
        return "accept", "higher"
    if result.certified_fraction == incumbent["cert"]:
        return "reject", "tie"
    return "reject", "lower"


def run_loop(policy_path: str, mix_p: float, seed: int, seed_dir: str) -> dict:
    """Run the loop and return the per-seed summary (see module docstring)."""
    os.makedirs(seed_dir, exist_ok=True)
    ckpt_cur = os.path.abspath(policy_path)
    val_cur, val_cur_std = policy.checkpoint_greedy_return(ckpt_cur, cfg.N_VAL_EPISODES, seed)
    status = "STUCK_MAX_ITERS"
    rows = []
    incumbent = None
    n_rejects = n_accepts = 0

    for k in range(1, cfg.MAX_VERIFICATIONS + 1):
        try:
            result, verify_sec, n_esc = verify_policy(ckpt_cur, alive=val_cur > DEAD_RETURN)
        except Exception as e:
            status = "VERIFY_FAILED"
            print(f"[loop] iter {k} VERIFY_FAILED: {e}", flush=True)
            traceback.print_exc()
            break

        cert = 1.0 if result.safe else (result.certified_fraction if result.violated else None)
        decision, reason = _decide(k, result, incumbent)
        if decision == "reject":
            n_rejects += 1
        else:
            n_accepts += int(k > 1)
            incumbent = {"ckpt": ckpt_cur, "iter": k, "cert": cert, "regions": None}
        row = {
            "iter": k, "ckpt": ckpt_cur,
            "verify_status": result.status, "strict_safe": int(result.safe),
            "undecided": int(not result.safe and not result.violated),
            "certified_fraction": cert, "n_violating_pieces": result.n_violating_pieces,
            "n_init_pieces": result.n_init_pieces,
            "n_violating_objects": 0 if result.safe else result.n_violating_objects,
            "val_return": val_cur, "val_return_std": val_cur_std, "verify_sec": verify_sec,
            "zono_peak": result.peak, "zono_cap_used": result.cap, "zono_escalations": n_esc,
            "zono_cap_step": result.cap_step, "zono_n_per_step": result.n_per_step,
            "ptxas": verifier.PTXAS_USED.get("version"),
            "inc_decision": decision, "inc_reason": reason,
            "incumbent_iter": incumbent["iter"] if incumbent else None,
            "incumbent_cert": incumbent["cert"] if incumbent else None,
            "n_rejects": n_rejects, "n_accepts": n_accepts,
            "ft_seed": None, "ft_from_iter": None, "restart_step": None, "n_restart_boxes": None,
        }
        rows.append(row)
        cert_text = "n/a" if cert is None else f"{cert:.6f}"
        print(f"[loop] iter {k} {decision} ({reason}): safe={result.safe} ratio of safe start state regions "
              f"{cert_text} val={val_cur:.1f} ({verify_sec:.1f}s verify, cap {result.cap:,})", flush=True)

        if result.safe:
            status = "SAFE"
            break
        if k == cfg.MAX_VERIFICATIONS:
            if not result.violated and k == 1:
                status = f"NO_COUNTEREXAMPLE_{result.status}"
            break
        if decision == "reject":
            regions = incumbent["regions"]
        elif not result.violated:
            # Undecided input policy: there are no counterexamples to train on.
            status = f"NO_COUNTEREXAMPLE_{result.status}"
            break
        else:
            try:
                regions = counterexample_regions(result, seed, log=lambda m: print(f"[loop]   {m}", flush=True))
            except CounterexampleExtractionFailed as e:
                status = "BP_FAILED"
                print(f"[loop] iter {k} BP_FAILED: {e}", flush=True)
                break
            incumbent["regions"] = regions
        row["restart_step"] = RESTART_STEP
        row["n_restart_boxes"] = int(regions["lo"].shape[0])

        ft_seed = int(seed) + cfg.REJECT_SEED_STRIDE * n_rejects
        row["ft_seed"], row["ft_from_iter"] = ft_seed, int(incumbent["iter"])
        t1 = time.time()
        try:
            ckpt_next, metrics = finetune(incumbent["ckpt"], regions["lo"], regions["hi"], RESTART_STEP, mix_p, ft_seed,
                                          os.path.join(seed_dir, f"iter_{k:03d}", "candidate"))
        except Exception as e:
            status = "FINETUNE_FAILED"
            print(f"[loop] iter {k} FINETUNE_FAILED: {e}", flush=True)
            traceback.print_exc()
            break
        row["ft"] = {key: metrics[key] for key in FT_FIELDS}
        row["ft"].update(sec=time.time() - t1, near_miss_n_boxes=int(regions["lo"].shape[0]),
                         n_piece_fallback=regions["n_piece_fallback"])
        ckpt_cur = ckpt_next
        val_cur, val_cur_std = metrics["mean_val_return"], metrics["std_val_return"]

    certs = [r["certified_fraction"] for r in rows]
    candidate_certs = [c for r, c in zip(rows, certs) if r["iter"] >= 2 and c is not None]
    return {
        "seed": int(seed),
        "status": status,
        "best_iter": incumbent["iter"] if incumbent else 0,
        "best_ckpt": incumbent["ckpt"] if incumbent else None,
        "n_iters": len(rows),
        "n_undecided": sum(r["undecided"] for r in rows),
        "n_escalations": sum(r["zono_escalations"] for r in rows),
        "max_cap_used": max([r["zono_cap_used"] for r in rows] or [0]),
        "certified_fraction_base": certs[0] if certs else None,
        "certified_fraction_best": max(candidate_certs) if candidate_certs else None,
        "certified_fraction_last": certs[-1] if certs else None,
        "restart_steps": [r["restart_step"] for r in rows],
        "n_rejects": int(n_rejects),
        "n_accepts": int(n_accepts),
        "incumbent_iter": incumbent["iter"] if incumbent else None,
        "certified_fraction_incumbent": incumbent["cert"] if incumbent else None,
        "iters": rows,
    }


def run(policy_path: str, method: str, seed: int, run_dir: str) -> dict:
    """Run one experiment unit and write ``run_dir/summary.json``. Returns the summary."""
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    try:
        per_seed = run_loop(policy_path, cfg.MIX_P[method], seed, os.path.join(run_dir, f"seed{seed}"))
    except Exception as e:
        print(f"[loop] seed {seed} CRASH: {e}", flush=True)
        traceback.print_exc()
        per_seed = {"seed": int(seed), "status": "CRASH", "best_iter": 0, "best_ckpt": None, "n_iters": 0,
                    "iters": []}
    per_seed["seed_sec"] = time.time() - t0
    summary = {
        "env_id": cfg.ENV_ID,
        "policy": os.path.abspath(policy_path),
        "method": method,
        "seed": int(seed),
        "mix_p": cfg.MIX_P[method],
        "per_seed": [per_seed],
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[loop] TERMINAL {per_seed['status']} after {per_seed['n_iters']} verification(s) "
          f"({per_seed['seed_sec']:.0f}s)", flush=True)
    return summary
