#!/bin/bash
# One CartPole unit per array task:
#   sbatch --array=0-95 experiments/slurm/cartpole_array.sbatch tab_cp
#   sbatch --array=0-2  experiments/slurm/cartpole_array.sbatch figure3
#   sbatch --array=0-20 experiments/slurm/cartpole_array.sbatch viper
# Adapt partition/account/GPU type to your cluster. The paper used one NVIDIA
# H200 per unit; see experiments/cartpole/README.md for smaller GPUs.
#SBATCH --job-name=vespa-cartpole
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
TABLE=${1:?table: tab_cp | figure3 | viper}
cd "${SLURM_SUBMIT_DIR:-.}"
python experiments/cartpole/runs.py run-index --table "$TABLE" --index "$SLURM_ARRAY_TASK_ID"
