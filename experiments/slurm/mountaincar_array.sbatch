#!/usr/bin/env bash
# Slurm array example: one MountainCarContinuous experiment unit per array task
# (120 units = 12 input policies x 5 fine-tuning seeds x 2 methods). Adjust partition, account and GPU type.
#SBATCH --job-name=vespa-mountaincar
#SBATCH --array=0-119
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${VESPA_REPO:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# On Blackwell GPUs (sm_120), point VESPA_PTXAS_DIR to a CUDA >= 13 toolchain (see experiments/mountaincar/README.md):
# export VESPA_PTXAS_DIR=/path/to/cuda-13.x
python experiments/mountaincar/run_all.py --index "$SLURM_ARRAY_TASK_ID"
