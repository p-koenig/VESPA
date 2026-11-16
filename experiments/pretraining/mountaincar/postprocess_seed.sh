#!/usr/bin/env bash
# Export the best of the 5 trials, prune it and run the acceptance gate, for one seed (CPU, VESPA environment).
#   experiments/pretraining/mountaincar/postprocess_seed.sh <group> <seed> [root]   (default root: runs/mountaincar)
# Writes <root>/checkpoints/<group>/s<seed>/{base,base-pruned,gate.json}. Needs PYTHONPATH=<repo>/src.
set -euo pipefail
G=$1; S=$2
ROOT=$(realpath -m "${3:-runs/mountaincar}")
SRC=$ROOT/pretrain/$G/s$S
SEED_DIR=$ROOT/checkpoints/$G/s$S
[ -f "$SRC/DONE" ] && [ -d "$SRC/export" ] || { echo "pretraining not done: $SRC"; exit 2; }
export JAX_PLATFORMS=cpu
rm -rf "$SEED_DIR/base" "$SEED_DIR/base-pruned"; mkdir -p "$SEED_DIR"
python -u -m vespa.pretraining.mountaincar.export "$SRC/export" "$SEED_DIR/base"
python -u -m vespa.pretraining.pruning "$SEED_DIR/base"
python -u -m vespa.pretraining.mountaincar.gate "$SEED_DIR"
