#!/usr/bin/env bash
# Rebuild all tables and figures of the paper from the run directories of this repository.
#   analysis/run_all.sh OUT
# Inputs (defaults are the locations written by experiments/cartpole/run.py and experiments/mountaincar/run_all.py):
#   VESPA_CP_RUNS    CartPole run directory        (default: <repo>/runs/cartpole)
#   VESPA_MCC_RUNS   MountainCarContinuous runs    (default: <repo>/runs/mountaincar)
#   VESPA_PYTHON     python interpreter            (default: python)
# Set VESPA_SKIP_RETURNS=1 to reuse OUT/returns from an earlier call (the 1,000-episode evaluation takes ~6 min on CPU).
# Output: see analysis/README.md ("Expected output").
set -euo pipefail
OUT=${1:?usage: analysis/run_all.sh OUT}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/.." && pwd)
PY=${VESPA_PYTHON:-python}
CP_RUNS=${VESPA_CP_RUNS:-$REPO/runs/cartpole}
MCC_RUNS=${VESPA_MCC_RUNS:-$REPO/runs/mountaincar}
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
mkdir -p "$OUT"
"$PY" "$HERE/collect_cartpole.py" --runs-dir "$CP_RUNS" --out "$OUT/cp_per_verify.csv"
"$PY" "$HERE/collect_mountaincar.py" --runs-dir "$MCC_RUNS" --out-dir "$OUT/mountaincar"
"$PY" "$HERE/mountaincar_per_run.py" "$OUT/mountaincar/runs.csv" "$OUT/mountaincar/iters.csv" "$OUT/mountaincar"
"$PY" "$HERE/make_tables.py" --cp "$OUT/cp_per_verify.csv" --mcc-per-run "$OUT/mountaincar/mcc_per_run.csv" \
      --mcc-runs "$OUT/mountaincar/mcc_runs.csv" --out "$OUT/tables"
if [ "${VESPA_SKIP_RETURNS:-0}" != 1 ]; then
  P="$REPO/policies"
  "$PY" "$HERE/eval_returns.py" --out-dir "$OUT/returns" \
    --policy viper_L4_s6=$P/viper/viper_L4_s6 --policy viper_L32_s5=$P/viper/viper_L32_s5 \
    --policy viper_L32_s4=$P/viper/viper_L32_s4 --policy s8u=$P/cartpole/figure3/s8u \
    --policy s27u=$P/cartpole/evaluation/s27u --policy s40u=$P/cartpole/figure3/s40u
fi
"$PY" "$HERE/fig3_data.py" --returns-dir "$OUT/returns" --runs-dir "$CP_RUNS" --out "$OUT/fig3_data.csv"
"$PY" "$HERE/make_figures.py" --cp "$OUT/cp_per_verify.csv" --mcc-progress "$OUT/tables/mcc_progress.csv" \
      --fig3 "$OUT/fig3_data.csv" --out "$OUT/figures"
echo "tables: $OUT/tables  figures: $OUT/figures"
