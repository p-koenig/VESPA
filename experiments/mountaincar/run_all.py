#!/usr/bin/env python
"""Run the 120 MountainCarContinuous experiment units of the paper.

Units: for k = 1..5, for each of the 12 input policies (pretrain seeds 0-5, unpruned and pruned), for each
method (VESPA, Uninformed): fine-tuning seed = pretrain seed + 100000 k, run name ``<policy>-<method>-k<k>``
(for example ``s0-pruned-vespa-k1``).

    PYTHONPATH=src python experiments/mountaincar/run_all.py --list       # the 120 units
    PYTHONPATH=src python experiments/mountaincar/run_all.py              # all units, one after the other
    PYTHONPATH=src python experiments/mountaincar/run_all.py --index 17   # one unit (e.g. a Slurm array task)

Units whose ``summary.json`` exists are skipped, so the driver can be restarted.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

POLICIES = [(f"s{n}-{variant}", n, f"evaluation/s{n}/{directory}")
            for n in range(6) for variant, directory in (("unpruned", "base"), ("pruned", "base-pruned"))]
METHODS = ("vespa", "uninformed")
SEED_OFFSET = 100_000


def units(policies_dir: str) -> list[dict]:
    """All units in the order of the paper runs."""
    out = []
    for k in range(1, 6):
        for name, pretrain_seed, path in POLICIES:
            for method in METHODS:
                out.append({"index": len(out), "run_name": f"{name}-{method}-k{k}", "method": method,
                            "policy": os.path.join(policies_dir, path), "seed": pretrain_seed + SEED_OFFSET * k})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policies-dir", default=os.path.join("policies", "mountaincar"))
    p.add_argument("--out", default=os.path.join("runs", "mountaincar"), help="output root of the run directories")
    p.add_argument("--index", type=int, default=None, help="run only the unit with this index (0..119)")
    p.add_argument("--list", action="store_true", help="list the units and exit")
    a = p.parse_args(argv)
    todo = units(a.policies_dir)
    if a.index is not None:
        todo = [todo[a.index]]
    if a.list:
        for u in todo:
            print(u["index"], u["run_name"], u["method"], u["seed"], u["policy"])
        return 0
    os.makedirs(a.out, exist_ok=True)
    failures = 0
    for u in todo:
        if os.path.exists(os.path.join(a.out, u["run_name"], "summary.json")):
            continue
        cmd = [sys.executable, "-m", "vespa.mountaincar.run", "--policy", u["policy"], "--method", u["method"],
               "--seed", str(u["seed"]), "--run-name", u["run_name"], "--out", a.out]
        with open(os.path.join(a.out, u["run_name"] + ".log"), "a") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"{u['run_name']}: rc={rc}", flush=True)
        failures += rc != 0
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
