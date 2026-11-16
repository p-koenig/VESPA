#!/usr/bin/env bash
# List (or with --run, execute one after the other) the commands for every MountainCarContinuous input policy.
#   experiments/pretraining/mountaincar/all.sh [--run]
# Evaluation seeds 0-5. Tuning seeds 1000-1004: seed 1000 fails the gate, so the first four passing seeds are used.
# select_input_policies writes the lists of input policies of both groups.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=runs/mountaincar
cmds=()
for s in 0 1 2 3 4 5; do
  cmds+=("$HERE/pretrain_seed.sh evaluation $s $ROOT && $HERE/postprocess_seed.sh evaluation $s $ROOT")
done
for s in 1000 1001 1002 1003 1004; do
  cmds+=("$HERE/pretrain_seed.sh tuning $s $ROOT && $HERE/postprocess_seed.sh tuning $s $ROOT")
done
SELECT="python -m vespa.pretraining.mountaincar.select_input_policies $ROOT"
cmds+=("$SELECT evaluation 0,1,2,3,4,5 6 $ROOT/evaluation_input_policies.json")
cmds+=("$SELECT tuning 1000,1001,1002,1003,1004 4 $ROOT/tuning_input_policies.json")
for c in "${cmds[@]}"; do
  if [ "${1:-}" = "--run" ]; then bash -c "$c"; else echo "$c"; fi
done
