#!/usr/bin/env python
"""Run one CartPole unit of the VESPA paper.

  loop    VESPA loop on one input policy with one method and fine-tuning seed
          (Table tab:cp, Fig. cp-progress)
  verify  one verification of a policy, without fine-tuning (Fig. 3)

  python experiments/cartpole/run.py loop   --policy s13u --method vespa --seed 0
  python experiments/cartpole/run.py verify --policy viper_L8_s4

Settings come from experiments/cartpole/config.toml. A run writes
<runs-dir>/<run-name>/ (default runs/cartpole/<method>-<policy>-f<seed> or
verify-<policy>); see vespa/cartpole/loop.py for its contents. Rerunning the
same command resumes a stopped run after its last complete iteration.

--box-memory-gib sets the GPU memory of the box pool (default: the config's
"auto", all GPU memory beside the kernel). A GPU with less memory than an H200
needs a smaller kernel_cap in a copy of config.toml (--config).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_CONFIG = os.path.join(HERE, "config.toml")


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def find_policy(cfg, name):
    """Repository-relative checkpoint directory of a policy name (or a path)."""
    if os.path.isdir(os.path.join(REPO, name)):
        return name
    for d in cfg["policies"]["dirs"]:
        if os.path.isdir(os.path.join(REPO, d, name)):
            return f"{d}/{name}"
    raise SystemExit(f"policy {name!r} not found in {cfg['policies']['dirs']}")


def run_name(unit, policy, method=None, seed=None):
    name = os.path.basename(policy.rstrip("/"))
    return f"{method}-{name}-f{seed}" if unit == "loop" else f"verify-{name}"


def settings(cfg, unit, method=None, seed=0, box_memory_gib=None):
    """The settings block of a run's config.json."""
    s = dict(cfg["verifier"])
    s.update(lookback=cfg["loop"]["lookback"], backprop_max_boxes=cfg["loop"]["backprop_max_boxes"],
             finetune=dict(cfg["finetune"]))
    if unit == "loop":
        if method not in cfg["methods"]:
            raise SystemExit(f"--method must be one of {sorted(cfg['methods'])}")
        s.update(max_iterations=cfg["loop"]["max_iterations"], mix_p=cfg["methods"][method])
    else:   # verification only: the input policy is verified once
        s.update(max_iterations=1, mix_p=0.0)
    if box_memory_gib is not None:
        s["box_memory_gib"] = box_memory_gib
    return s


def run_unit(cfg, unit, policy, method=None, seed=0, runs_dir=None, name=None,
             box_memory_gib=None):
    """Run one unit from the repository root; returns its terminal status."""
    sys.path.insert(0, os.path.join(REPO, "src"))
    from vespa.cartpole.loop import run_loop

    os.chdir(REPO)
    policy = find_policy(cfg, policy)
    seed = int(seed or 0)
    run_dir = os.path.join(runs_dir or os.environ.get("VESPA_RUNS_DIR") or "runs/cartpole",
                           name or run_name(unit, policy, method, seed))
    os.makedirs(run_dir, exist_ok=True)
    config = {"unit": unit, "policy": policy, "method": method, "seed": seed,
              "settings": settings(cfg, unit, method, seed, box_memory_gib)}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[run] {unit} {policy} method={method} seed={seed} -> {run_dir}", flush=True)
    return run_loop(run_dir, policy, config["settings"], seed)


FAILED = ("INCOMPLETE_COVERAGE", "FINETUNE_FAILED", "BACKWARD_PROPAGATION_FAILED")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("unit", choices=("loop", "verify"))
    p.add_argument("--policy", required=True, help="policy name or checkpoint directory")
    p.add_argument("--method", choices=("vespa", "uninformed"), help="loop only")
    p.add_argument("--seed", type=int, default=0, help="fine-tuning seed (loop only)")
    p.add_argument("--runs-dir", help="default: $VESPA_RUNS_DIR or runs/cartpole")
    p.add_argument("--run-name")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--box-memory-gib", help="GPU memory of the box pool in GiB, or auto")
    a = p.parse_args(argv)
    if a.unit == "loop" and a.method is None:
        p.error("loop needs --method")
    status = run_unit(load_config(a.config), a.unit, a.policy, a.method, a.seed, a.runs_dir,
                      a.run_name, a.box_memory_gib)
    return 1 if status in FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
