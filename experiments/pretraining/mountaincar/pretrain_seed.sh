#!/usr/bin/env bash
# MountainCarContinuous pretraining of one seed with the upstream SYMPOL code and its command
# (upstream commands_to_run.txt, MountainCarContinuous line, with --seed added and without W&B tracking):
# 5 random trials of 1,002,320 steps on the CPU. The export hook writes the final actor of every trial.
#   experiments/pretraining/mountaincar/pretrain_seed.sh <group> <seed> [root]      (default root: runs/mountaincar)
# <group> is evaluation or tuning. Writes <root>/pretrain/<group>/s<seed>/.
# About 9 min per seed with 16 CPU threads.
set -euo pipefail
G=$1; S=$2
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
DIR=$REPO/third_party/sympol
ENVP=$REPO/third_party/sympol-env
ROOT=$(realpath -m "${3:-runs/mountaincar}")
OUT=$ROOT/pretrain/$G/s$S
[ -f "$OUT/DONE" ] && { echo "already done: $OUT"; exit 0; }
rm -rf "$OUT"; mkdir -p "$OUT/export" "$OUT/wd"
cd "$OUT/wd"                      # the upstream code writes videos and tree images into the working directory
export JAX_PLATFORMS=cpu
export PATH="$ENVP/bin:$PATH"     # graphviz `dot` for the upstream tree plots
t0=$(date +%s)
SYMPOL_EXPORT_DIR="$OUT/export" "$ENVP/bin/python" -u "$DIR/ppo_gymnax.py" \
  --env_id MountainCarContinuous-v0 --n_estimators 1 --gpu_number 0 \
  --random_trials 5 --n_trials 60 --actor sympol --run_name SYMPOL_useBest \
  --exp_name results_useBest --use_best_config --action_type continuous --dynamic_buffer \
  --seed "$S" > "$OUT/train.log" 2>&1
n=$(ls "$OUT"/export/*/meta.json 2>/dev/null | wc -l)
echo "[pretrain] group=$G seed=$S trials exported: $n wall: $(( $(date +%s) - t0 ))s"
[ "$n" -eq 5 ] || { echo "expected 5 exported trials"; exit 2; }
touch "$OUT/DONE"
