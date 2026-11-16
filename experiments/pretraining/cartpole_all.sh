#!/usr/bin/env bash
# List (or with --run, execute one after the other) the commands for every CartPole input policy of the paper.
#   experiments/pretraining/cartpole_all.sh [--run]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
EVALUATION="6 10 12 13 14 17 20 21 27 28 42 43"   # 24 evaluation policies (pruned and unpruned)
TUNING="3 9 11 16"                                 # tuning policies
FIGURE3="8 40"                                     # additional policies of Fig. 3
for s in $EVALUATION $TUNING $FIGURE3; do
  if [ "${1:-}" = "--run" ]; then "$HERE/cartpole.sh" "$s"; else echo "$HERE/cartpole.sh $s"; fi
done
