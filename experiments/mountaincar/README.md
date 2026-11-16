# MountainCarContinuous experiments

This directory reproduces the MountainCarContinuous results of the paper: 12 pretrained SYMPOL decision-tree
policies (pretrain seeds 0-5, each unpruned and pruned) x 5 fine-tuning seeds x 2 methods (VESPA, Uninformed)
= **120 runs**. All commands run from the repository root.

## One unit at a time

```bash
# verify one policy (no fine-tuning): verdict and ratio of safe start state regions
python -m vespa.mountaincar.verify --policy policies/mountaincar/evaluation/s0/base

# one run = input policy x method x fine-tuning seed
python -m vespa.mountaincar.run --policy policies/mountaincar/evaluation/s0/base-pruned \
    --method vespa --seed 100000 --run-name s0-pruned-vespa-k1
python -m vespa.mountaincar.run --policy policies/mountaincar/evaluation/s0/base-pruned \
    --method uninformed --seed 100000 --run-name s0-pruned-uninformed-k1
```

## All 120 runs and collection

```bash
python experiments/mountaincar/run_all.py --list          # the 120 units (index, name, method, seed, policy)
python experiments/mountaincar/run_all.py                 # all units (skips runs with summary.json)
python experiments/mountaincar/run_all.py --index 17      # one unit (e.g. one Slurm array task)
python analysis/collect_mountaincar.py --runs-dir runs/mountaincar --out-dir results/mountaincar
```

A Slurm array script is in `experiments/slurm/mountaincar_array.sbatch`.

Enumeration: for k = 1..5, for each policy, for each method: fine-tuning seed = pretrain seed N + 100000 k
(100000 ... 500005), run name `<policy>-<method>-k<k>`. The pruned and the unpruned policy of a pretrain seed
and both methods use the same seed. The input policies are `policies/mountaincar/evaluation/s<N>/{base,base-pruned}`
(`base-pruned` = the same tree with same-action leaves merged). `policies/mountaincar/tuning/` holds the 8
policies used only for hyperparameter selection.

## Configuration

All result-defining values are constants in `src/vespa/mountaincar/config.py`. The two methods differ only
in the mixing probability p.

| setting | value |
|---|---|
| property | reachability: every trajectory from S0 reaches x >= 0.45 within n = 200 steps |
| start set S0 | x0 in [-0.6, -0.4], v0 = 0, split into 2^14 = 16,384 start state regions |
| verifier | no-merge zonotope verifier, float32, order limit 24 |
| zonotope buffer | 262,144 objects; after an overflow on a policy that reaches the goal: 4,194,304, then 16,777,216 |
| counterexample regions | every violating zonotope walked back 200 steps with its own actions (regions at step 1), at most 250,000 zonotopes, start state region as fallback |
| iterations | 6 verifications = input policy + up to 5 fine-tuning iterations, stop at the first proof |
| acceptance | accept iff proven safe or the ratio of safe start state regions is strictly higher; otherwise fine-tune the incumbent again with seed + 1000 x rejects |
| fine-tuning | PPO, 10^5 environment steps per iteration, 5 environments x 128 steps, 8 epochs, minibatch 64, gamma 0.999, GAE 0.99, clip 0.1, entropy 0.1, value coefficient 0.5, gradient-norm clip 1000, raw Gaussian actions, horizon margin 1 |
| learning rates | split thresholds 1.67e-4 (Adam), split features 1.46e-4, leaves 4e-3, action log-std 1e-2 (AdamW), tree weights 1.23e-4 (Adam), critic 2.06e-3; action log-std clamped to <= -1 at load |
| mixing probability p | VESPA 0.5, Uninformed 0.0 |
| validation | 30 greedy episodes |

The runs set `XLA_PYTHON_CLIENT_ALLOCATOR=platform` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` (memory settings).

## GPU and ptxas

The paper runs used NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (sm_120). With JAX 0.9.2, the bundled
CUDA-12 ptxas 12.9.86 **miscompiles** a 64-bit address add with an immediate >= 2^31 on sm_120. Programs that
address an operand at an offset >= 2 GiB, i.e. zonotope buffers above 2^21 objects, then read or write the
wrong memory. The paper runs therefore compiled the verifier with **ptxas 13.2.86**. To do the same, set
`VESPA_PTXAS_DIR` to a CUDA >= 13 toolchain directory containing `bin/ptxas`, `bin/nvlink` and
`nvvm/libdevice/libdevice.10.bc`. If it is unset, the bundled toolchain is used, and on sm_120 a buffer
above 2^21 objects raises an error instead of producing wrong results. Other GPUs (e.g. H200, sm_90) and CPU
need no setting. Each verification records the ptxas version in `summary.json` (`iters[*].ptxas`).

## Output

`runs/mountaincar/<run>/summary.json` holds the run summary. `per_seed[0].iters` has one record per
verification, with fields such as `certified_fraction` (ratio of safe start state regions),
`n_violating_pieces`, `inc_decision`, `inc_reason` and `ft` (fine-tuning metrics). Candidate policies are under
`seed<seed>/iter_*/candidate`, and `run_info.json` records the wall time.

`analysis/collect_mountaincar.py` collects the run directories into `runs.csv` (one row per run) and
`iters.csv` (one row per verification); its docstring lists the columns.

## Runtime

On one RTX PRO 6000, a run took a median of 531 s in the paper runs (with several runs sharing one GPU) and a verification a median of 23 s.
Escalations to 16.7M zonotopes need up to about 64 GB of GPU memory. On CPU a verification at the default
buffer takes a few minutes.
