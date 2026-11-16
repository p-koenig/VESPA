#!/usr/bin/env bash
# Train and check one CartPole input policy (paper command).
#   experiments/pretraining/cartpole.sh <seed> [out_root]        (default out_root: checkpoints/cartpole)
# Writes <out_root>/s<seed>/checkpoint (best evaluation in 50M steps, 29,700,608 steps for tuning seed 3),
# <out_root>/s<seed>/checkpoint-pruned and <out_root>/s<seed>/gate.json. Needs PYTHONPATH=<repo>/src.
# The paper policies were trained on one GPU each (about 3 h).
set -euo pipefail
SEED=$1
OUT=${2:-checkpoints/cartpole}/s$SEED
STEPS=50000000
[ "$SEED" = 3 ] && STEPS=29700608
[ -e "$OUT" ] && { echo "refusing to overwrite $OUT"; exit 4; }
python -u -m vespa.pretraining.ppo --seed "$SEED" --total-steps "$STEPS" --out "$OUT"
JAX_PLATFORMS=cpu python -u -m vespa.pretraining.cartpole_gate "$OUT" "$SEED"
