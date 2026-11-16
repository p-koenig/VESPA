"""Verify one MountainCarContinuous policy (no fine-tuning).

Goal property at 200 steps from S0 = {x0 in [-0.6, -0.4], v0 = 0} split into 2^14 start state regions,
with the buffer escalation of the loop. Prints the verdict and the ratio of safe start state regions.

    PYTHONPATH=src python -m vespa.mountaincar.verify --policy policies/mountaincar/evaluation/s0/base
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from vespa.mountaincar import config as cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="policy (Orbax checkpoint directory)")
    a = p.parse_args(argv)

    for key, value in cfg.XLA_ENV.items():
        os.environ.setdefault(key, value)
    from vespa.mountaincar import loop, policy

    val_return, _ = policy.checkpoint_greedy_return(a.policy, cfg.N_VAL_EPISODES, 0)
    result, sec, n_escalations = loop.verify_policy(a.policy, alive=val_return > loop.DEAD_RETURN)
    print(json.dumps({"policy": a.policy, "safe": result.safe, "verify_status": result.status,
                      "certified_fraction": result.certified_fraction,
                      "n_violating_pieces": result.n_violating_pieces, "n_init_pieces": result.n_init_pieces,
                      "cap": result.cap, "n_escalations": n_escalations, "val_return": val_return,
                      "verify_sec": round(sec, 1)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
