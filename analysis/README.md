# Analysis: from run directories to the paper's tables and figures

All scripts read run outputs read-only and write CSV / LaTeX / PNG files into an output directory of your choice.
Run them from the repository root. `run_all.sh` chains everything:

```bash
analysis/run_all.sh OUT          # reads runs/cartpole and runs/mountaincar
```

`run_all.sh` expects the complete set of runs: the 96 CartPole fine-tuning runs, the verification-only runs of the
six Fig. 3 policies (`runs.py run --table viper` and `--table figure3`) and the 120 MountainCarContinuous runs. The
collectors and `make_tables.py` can also be called on a subset of runs.

Inputs are the run directories written by `experiments/cartpole/run.py` (`runs/cartpole/`) and
`experiments/mountaincar/run_all.py` (`runs/mountaincar/`). Override them with `VESPA_CP_RUNS` and `VESPA_MCC_RUNS`.
`VESPA_PYTHON` selects the interpreter. `VESPA_SKIP_RETURNS=1` reuses `OUT/returns/` from an earlier call; the
1,000-episode return evaluation of Fig. 3 takes about 6 min on a CPU and imports `vespa.pretraining` (`run_all.sh`
puts `src/` on the Python path).

| Step | Script | Input | Output |
|---|---|---|---|
| CartPole collector | `collect_cartpole.py` | `runs/cartpole/<vespa\|uninformed>-<policy>-f<seed>/` | `cp_per_verify.csv` (one row per verification) |
| MountainCarContinuous collector | `collect_mountaincar.py` | `runs/mountaincar/<policy>-<vespa\|uninformed>-k<k>/` | `runs.csv`, `iters.csv` |
| MountainCarContinuous per-run tables | `mountaincar_per_run.py` | `runs.csv`, `iters.csv` | `mcc_runs.csv`, `mcc_per_run.csv` |
| Tables tab:cp, tab:mcc, tab:pruning | `make_tables.py` | `cp_per_verify.csv`, `mcc_per_run.csv`, `mcc_runs.csv` | `tab_*.tex`, `tab_*.csv`, `mcc_progress.csv` |
| Fig. 3 returns (1,000 episodes) | `eval_returns.py` | policies in `policies/` | `ret1000_<policy>.json` |
| Fig. 3 data | `fig3_data.py` | `ret1000_*.json`, `runs/cartpole/verify-<policy>/` | `fig3_data.csv` |
| Figures | `make_figures.py` | `cp_per_verify.csv`, `mcc_progress.csv`, `fig3_data.csv` | three PNGs |
| Sampled witnesses (10^8 starts) | `witness_search.py` | VIPER trees / SYMPOL checkpoints | per-policy failure counts and failing starts (not part of `run_all.sh`) |

Methods are labelled `VESPA` and `Uninformed` (column `method`). CartPole results are verified safe steps
(column `depth`, a lower bound when `exact` is False). MountainCarContinuous results are the ratio of safe start
state regions (`safe_ratio`).

Conventions (see the docstring of `make_tables.py`):
* Each value describes the accepted policy after k fine-tuning iterations.
* Means and sample standard deviations (ddof = 1) over runs, rounded half up.
* CartPole verifications that ran out of GPU memory enter with their verified depth as a lower bound (`#>=`).
* Pruned and unpruned input policies are pooled in tab:cp and tab:mcc and split in tab:pruning.

## Expected output of `run_all.sh OUT`

```
OUT/cp_per_verify.csv                  CartPole, one row per verification (96 runs)
OUT/mountaincar/runs.csv               MountainCarContinuous, one row per run (120 runs)
OUT/mountaincar/iters.csv              MountainCarContinuous, one row per verification
OUT/mountaincar/mcc_runs.csv           per-run summary (safe, iterations until the proof, returns)
OUT/mountaincar/mcc_per_run.csv        per run and fine-tuning iteration k = 0..5
OUT/tables/tab_cp.tex, tab_cp.csv      Table tab:cp
OUT/tables/tab_mcc.tex, tab_mcc.csv    Table tab:mcc
OUT/tables/tab_pruning.tex, tab_pruning.csv   Table tab:pruning
OUT/tables/mcc_progress.csv            data of the MountainCarContinuous progress figure
OUT/returns/ret1000_<policy>.json      1,000-episode returns of the six Fig. 3 policies
OUT/fig3_data.csv                      data of Fig. 3
OUT/figures/fig3_viper_sympol_bars.png Fig. 3 (return over 1,000 episodes and verified safe steps)
OUT/figures/cp_progress.png            CartPole progress (verified safe steps per fine-tuning iteration)
OUT/figures/mcc_progress.png           MountainCarContinuous progress (ratio of safe start state regions)
OUT/figures/cp_progress_plotted_both.csv   plotted CartPole values
```

## Run-directory format

`collect_cartpole.py` reads, per fine-tuning run, `summary.json` (block `loop_result.accept_reject`, one entry per
verification), `console.log` (lines `val_return: before=... after=... std=... over 30 eps` give the validation
returns) and `iter_NNN/{iter_NNN.json, status_0-1_of_1.json, pool_0-1_of_1.log}`. Iteration 1 is the verification
of the input policy. The leaf count of each verified tree is read from `iter_NNN/pool_0-1_of_1.log`, else from the
worker lines of `console.log`. `fig3_data.py` reads `loop_result.accept_reject[0]` of the verification-only runs
`verify-<policy>`.

`collect_mountaincar.py` reads `summary.json` (`per_seed[0].iters`) and `run_info.json` of each run.

## Fonts

The paper figures use the macOS system font Avenir Next. `make_figures.py` falls back to the first installed font
of `Nunito Sans, Helvetica Neue, Helvetica, Arial, Liberation Sans, DejaVu Sans` and prints a note; the plotted data
and layout are unchanged.
