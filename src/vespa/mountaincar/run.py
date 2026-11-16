"""Run one MountainCarContinuous experiment unit: input policy x method x fine-tuning seed.

    PYTHONPATH=src python -m vespa.mountaincar.run \\
        --policy policies/mountaincar/evaluation/s0/base-pruned --method vespa --seed 100000 \\
        --run-name s0-pruned-vespa-k1

The paper configuration is ``vespa.mountaincar.config``. Results go to ``<out>/<run-name>/summary.json``
and ``run_info.json``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

from vespa.mountaincar import config as cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="input policy (Orbax checkpoint directory)")
    p.add_argument("--method", required=True, choices=sorted(cfg.MIX_P))
    p.add_argument("--seed", type=int, required=True, help="fine-tuning seed (paper: pretrain seed + 100000 k)")
    p.add_argument("--run-name", required=True, help="run directory name, e.g. s0-pruned-vespa-k1")
    p.add_argument("--out", default=os.path.join("runs", "mountaincar"), help="output root")
    a = p.parse_args(argv)

    for key, value in cfg.XLA_ENV.items():
        os.environ.setdefault(key, value)
    from vespa.mountaincar import loop

    run_dir = os.path.join(os.path.abspath(a.out), a.run_name)
    start = datetime.datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    loop.run(a.policy, a.method, a.seed, run_dir)
    info = {"policy": a.policy, "method": a.method, "seed": a.seed, "start": start,
            "end": datetime.datetime.now().isoformat(timespec="seconds"), "sec": round(time.time() - t0, 1)}
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
