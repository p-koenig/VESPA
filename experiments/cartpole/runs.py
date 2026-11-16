#!/usr/bin/env python
"""All CartPole units of the VESPA paper.

Tables:
  tab_cp    96 loop runs: 24 input policies x {vespa, uninformed} x seeds {0, 1}
            (Table tab:cp, Fig. cp-progress)
  figure3   3 verifications of the SYMPOL trees in Fig. 3 (s8u, s27u, s40u)
  viper     21 verifications of the VIPER trees (Fig. 3, appendix)

  python experiments/cartpole/runs.py check                         # every unit's policy exists
  python experiments/cartpole/runs.py list --table tab_cp           # the units, numbered
  python experiments/cartpole/runs.py run  --table viper            # one after another
  python experiments/cartpole/runs.py run-index --table tab_cp --index 7   # one unit (array jobs)

`run` skips units whose run directory already holds a summary.json. Each unit
needs one GPU; units are independent and can run in parallel on different GPUs.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R  # noqa: E402


def units(cfg, table):
    pol = cfg["policies"]
    if table == "tab_cp":
        return [dict(unit="loop", policy=p, method=m, seed=s) for p in pol["evaluation"]
                for m in ("vespa", "uninformed") for s in cfg["loop"]["seeds"]]
    return [dict(unit="verify", policy=p, method=None, seed=0) for p in pol[table]]


def check(cfg) -> int:
    """Print how many policy directories of each table exist; 1 if any is missing."""
    missing = 0
    for t in ("tab_cp", "figure3", "viper"):
        names = sorted({u["policy"] for u in units(cfg, t)})
        bad = [n for n in names if not any(os.path.isdir(os.path.join(R.REPO, d, n))
                                           for d in cfg["policies"]["dirs"])]
        print(f"[check] {t}: {len(names) - len(bad)} of {len(names)} policies found"
              + (f", missing: {', '.join(bad)}" if bad else ""))
        missing += len(bad)
    return int(missing > 0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("check", "list", "run", "run-index"))
    p.add_argument("--table", choices=("tab_cp", "figure3", "viper"),
                   help="required except for check (all tables)")
    p.add_argument("--index", type=int, help="run-index: 0-based unit index in the table")
    p.add_argument("--config", default=R.DEFAULT_CONFIG)
    p.add_argument("--runs-dir", help="default: $VESPA_RUNS_DIR or runs/cartpole")
    p.add_argument("--box-memory-gib", help="GPU memory of the box pool in GiB, or auto")
    a = p.parse_args(argv)
    cfg = R.load_config(a.config)
    if a.action == "check":
        return check(cfg)
    if a.table is None:
        p.error("--table is required")
    us = units(cfg, a.table)
    if a.action == "run-index":
        us = [us[a.index]]
    runs_dir = a.runs_dir or os.environ.get("VESPA_RUNS_DIR") or "runs/cartpole"
    rc = 0
    for i, u in enumerate(us):
        name = R.run_name(u["unit"], u["policy"], u["method"], u["seed"])
        if a.action == "list":
            print(f"{i:3d} {name}")
            continue
        if os.path.isfile(os.path.join(R.REPO, runs_dir, name, "summary.json")):
            print(f"[runs] {name}: done, skipped", flush=True)
            continue
        status = R.run_unit(cfg, u["unit"], u["policy"], u["method"], u["seed"], runs_dir,
                            box_memory_gib=a.box_memory_gib)
        print(f"[runs] {name}: {status}", flush=True)
        rc = rc or int(status in R.FAILED)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
