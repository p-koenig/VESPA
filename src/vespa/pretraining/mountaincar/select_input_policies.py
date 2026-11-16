"""Select the gate-passing MountainCarContinuous input policies of a group of seeds.

    python -m vespa.pretraining.mountaincar.select_input_policies <root> <group> <seed,seed,...> <n_needed> <out.json>

Reads ``<root>/checkpoints/<group>/s<seed>/{base,base-pruned,gate.json}``. Seeds are taken in the given order and
the first ``n_needed`` seeds that pass the gate and the pruning check are used, each as an unpruned and a pruned input
policy. Prints the gate numbers of every seed and writes the list of input policies to ``out.json``. Exits with
status 2 if fewer than ``n_needed`` seeds pass (then more seeds must be trained).
"""
import json
import os
import sys


def selected_test_eval(export):
    """Test evaluation of the trial that the export selected."""
    for trial in export.get("trials", []):
        if trial.get("trial") == export.get("selected_trial"):
            return trial.get("test_eval")
    return None


def main():
    root, group, seed_list, needed, out = sys.argv[1:6]
    seeds, needed = [int(s) for s in seed_list.split(",")], int(needed)
    policies, used, missing = [], [], []
    for seed in seeds:
        seed_dir = f"{root}/checkpoints/{group}/s{seed}"
        if not os.path.exists(f"{seed_dir}/gate.json"):
            missing.append(seed)
            print(f"s{seed}: no gate.json (pretraining or post-processing not done)")
            continue
        with open(f"{seed_dir}/gate.json") as f:
            gate = json.load(f)
        with open(f"{seed_dir}/base/export.json") as f:
            export = json.load(f)
        unpruned = gate["gate_unpruned"]
        print(f"s{seed}: pass={gate['pass_gate']} prune_ok={gate['prune_check']['ok']} "
              f"greedy@200={unpruned['greedy_goal200']:.3f} sampled@200={unpruned['sampled_goal200']:.3f} "
              f"sigma={unpruned['sigma']:.2f} trial={export.get('selected_trial')} "
              f"test_eval={selected_test_eval(export)}")
        if gate["pass_gate"] and gate["prune_check"]["ok"] and len(used) < needed:
            used.append(seed)
            for variant, suffix in (("unpruned", ""), ("pruned", "-pruned")):
                policies.append({"name": f"{group}-s{seed}-{variant}", "ckpt": f"{seed_dir}/base{suffix}",
                              "seed": seed, "variant": variant})
    print(f"{group}: using seeds {used} ({len(used)}/{needed}), missing {missing}")
    if len(used) < needed:
        print("not enough seeds pass the gate: train more seeds")
        sys.exit(2)
    with open(out, "w") as f:
        json.dump(policies, f, indent=1)
    print(f"wrote {len(policies)} input policies to {out}")


if __name__ == "__main__":
    main()
