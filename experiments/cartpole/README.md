# CartPole experiments

This directory reproduces the CartPole results of the paper. The code is
the package `vespa.cartpole` (in `src/vespa/cartpole/`). The tree policy itself is `src/vespa/sympol.py`.

| module | content |
|---|---|
| `env_specs.py` | CartPole dynamics: sound interval step (verifier), approximate inverse step (counterexample regions), safety box and start set |
| `interval_core.py` | segment kernel: propagates boxes step by step on the GPU, splits them along the tree's leaves, removes duplicate and contained boxes, finds violations |
| `device_pool.py` | compact storage of the kernel's output boxes in GPU memory |
| `segment_pool.py` | breadth-first verification over the horizon with a box pool in GPU memory |
| `policy.py` | SYMPOL checkpoints and the leaf boxes of the hard tree policy |
| `backward_propagate.py` | counterexample regions: violating boxes walked back `lookback` steps |
| `finetune.py` | PPO fine-tuning of the tree on the counterexample regions |
| `verify_worker.py`, `finetune_worker.py` | the two steps of the loop, each run as its own process |
| `loop.py` | the VESPA loop (verify, accept/reject, fine-tune) |
| `cartpole_env.py` | the gymnasium environment |

| file here | content |
|---|---|
| `config.toml` | every setting of the paper's CartPole runs (verifier, loop, PPO, methods, policy lists) |
| `run.py` | one experiment unit |
| `runs.py` | lists or runs all units of a table |

All commands are run from the repository root.

## Requirements

* The Python environment of the repository (JAX 0.9.2 with CUDA 12 for GPU runs).
* One NVIDIA GPU per run. The paper used one NVIDIA H200 (140 GiB) per run (see [GPU memory](#gpu-memory-and-lower-bounds)).
* Input policies (SYMPOL Orbax checkpoint directories): the 24 evaluation policies in
  `policies/cartpole/evaluation/`, the Fig. 3 trees `s8u` and `s40u` in `policies/cartpole/figure3/`
  (`s27u` is an evaluation policy), 8 tuning policies in `policies/cartpole/tuning/` and the VIPER
  trees in `policies/viper/`. `--policy` takes a name, looked up in these directories, or a
  checkpoint path. `python experiments/cartpole/runs.py check` checks that every unit's policy exists.

## Experiment units

1. **Loop** (Table *tab:cp*, Fig. *cp-progress*): one input policy × method × fine-tuning seed.
   ```bash
   python experiments/cartpole/run.py loop --policy s13u --method vespa --seed 0
   python experiments/cartpole/run.py loop --policy s13u --method uninformed --seed 0
   ```
   `vespa` starts fine-tuning episodes in a counterexample region with probability 0.5,
   `uninformed` never (p = 0). A run verifies the input policy, then alternates fine-tuning
   and verification, with at most 12 verifications.
2. **Verification only** (Fig. 3 trees and the 21 VIPER trees): one verification over the horizon 200.
   ```bash
   python experiments/cartpole/run.py verify --policy viper_L8_s4
   ```
3. **All units of a table:**
   ```bash
   python experiments/cartpole/runs.py list --table tab_cp     # 96 units: 24 policies x 2 methods x seeds 0, 1
   python experiments/cartpole/runs.py list --table figure3    # s8u, s27u, s40u
   python experiments/cartpole/runs.py list --table viper      # 21 VIPER trees
   python experiments/cartpole/runs.py run  --table viper      # one after another
   python experiments/cartpole/runs.py run-index --table tab_cp --index 0   # one unit (array jobs)
   ```
   Units are independent; run them in parallel on different GPUs (`CUDA_VISIBLE_DEVICES`).
   `experiments/slurm/cartpole_array.sbatch` is a Slurm array example. `run` skips units whose
   run directory already holds a `summary.json`.

Options of `run.py`: `--runs-dir` (default `$VESPA_RUNS_DIR` or `runs/cartpole`), `--run-name`,
`--config` (a modified copy of `config.toml`) and `--box-memory-gib` (GPU memory of the box pool).

## Run directories

Run names are `<method>-<policy>-f<seed>` (e.g. `vespa-s13u-f0`) or `verify-<policy>`.
Rerunning a command resumes a stopped run after its last complete iteration.

```
runs/cartpole/<run-name>/
  config.json                 policy, method, seed and settings
  console.log                 console output of the loop and its workers
  summary.json                final status and per-verification outcomes
  iter_001/ ... iter_NNN/     one directory per verification
    iter_NNN.json             iteration record (outcome, depth, accept/reject, timings, boxes per step)
    status_0-1_of_1.json      verifier result: status, first violating step, segments, boxes per step
    pool_0-1_of_1.log         verifier log
    violating_boxes.npz       violating boxes with their action history (if UNSAFE)
    status_finetune.json      counterexample regions and validation return (if fine-tuned)
    candidate/                the fine-tuned tree (if fine-tuned)
```

`summary.json` holds `status`, `iterations`, `elapsed_sec`, `checkpoint`, `final_ckpt` and the block
`loop_result`: `accept_reject` (one entry per verification: outcome, verified depth `depth` or bounds
`depth_lb`/`depth_ub`, exactness, accepted), `final_exact_depth`, `best_verified_steps`,
`vram_limited_ge`, `incumbent` and `n_rejects`.

**Final statuses**

| status | meaning |
|---|---|
| `SAFE` | proven safe for 200 steps |
| `MAX_ITERS` | all 12 verifications done; the result is the incumbent's exact verified depth |
| `VRAM_LIMITED` | a candidate was verified deeper than the incumbent before GPU memory ran out; it is accepted, and its depth is a lower bound (≥ d) |
| `VRAM_LIMITED_AT_BASE` | the input policy's verification ran out of GPU memory; no counterexample, no fine-tuning |

`run.py` exits with 1 only if the verifier crashed or fine-tuning failed.

**Verified safe steps**

* An exact verdict at first violating step t gives t − 1 verified safe steps.
* A memory-limited verification gives d = (lowest step still in the box pool) − 1, reported as "≥ d".

## Settings (`config.toml`)

All values are those of the paper's runs.

* **Verifier:** horizon 200; unsafe set |x| > 2.4 or |θ| > 0.20943951 rad; start set [−0.05, 0.05]^4;
  boxes in fp32 with an outward pad per step; breadth-first propagation (the first violation found is
  the exact first violating step); lossless box reductions (bit-identical duplicates and boxes
  contained in another box of the same leaf are removed); kernel cap 178,000,000 boxes; box pool in
  GPU memory; 14 actions kept per box.
* **Loop:** at most 12 verifications; a candidate verified less deep than the incumbent is rejected
  and the incumbent is fine-tuned again with seed + 1000 · (number of rejections); counterexample
  regions 10 steps before the violation; fine-tuning seeds 0 and 1.
* **Fine-tuning:** 100,000 environment steps; learning rate 3e-4 for all tree parameters, critic
  1.333e-3; entropy coefficient 0.2, 7 epochs, clip 0.1, γ = 0.99, GAE λ = 0.95; 7 environments ×
  512 steps, minibatch 64; reward −100 at the first unsafe state (episode ends); episodes truncated
  at the remaining horizon; 30 deterministic validation episodes ("Return").

## GPU memory and lower bounds

The verifier keeps every box of the current reachable set in GPU memory (no host memory, no disk).

**Paper configuration (1 × NVIDIA H200, 143,771 MiB = 140.4 GiB):**

* JAX may use 92 % of the GPU (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.92`, set by the loop for its workers).
* The kernel processes at most 178,000,000 boxes at a time, about 34 GiB.
* `box_memory_gib = "auto"` gives the pool the GPU memory left after the kernel and a margin of
  max(2 GiB, 3 %); two kernel outputs are reserved on top. The paper runs logged a pool target of
  88.9 GiB, about 2.6·10^9 boxes at 37 B per box.

**When the pool is full before a verdict,** the verification stops (`VRAM_POOL_FULL`) and gives the
lower bound d = (lowest step in the pool) − 1. The loop then:

* ends at once if the input policy is memory-limited (`VRAM_LIMITED_AT_BASE`);
* accepts a candidate whose lower bound exceeds the incumbent's depth and ends (`VRAM_LIMITED`);
* otherwise rejects the candidate and fine-tunes the incumbent again.

A violation whose earlier boxes could not all be checked (pool full) also counts as a lower bound.

**Smaller GPUs.** The pool shrinks with the memory, so verifications stop at smaller depths, more
results are lower bounds, and the loop can take other decisions than in the paper. Exact verdicts
that fit are not affected. If 178,000,000 boxes per kernel call do not fit, lower `kernel_cap` in a
copy of `config.toml` and pass it with `--config`. A smaller cap splits each step into more kernel
calls; duplicates and contained boxes are removed within a call, so more boxes may be stored.
Results obtained this way are not the paper configuration.

**CPU runs** (tests only): `JAX_PLATFORMS=cpu` with a fixed pool size, e.g.
`--box-memory-gib 4` and a config copy with `kernel_cap = 8000000`.

## Blackwell GPUs (sm_120) and ptxas

JAX 0.9.2 compiles GPU programs with the `ptxas` in `$CUDA_ROOT/bin` if `CUDA_ROOT` is set, and
otherwise with ptxas 12.9.86 from the `nvidia-cuda-nvcc-cu12` pip package. On sm_120 GPUs
(Blackwell, e.g. RTX PRO 6000), ptxas 12.9 miscompiles a 64-bit address add with an immediate
≥ 2^31: an operand addressed at a constant offset of 2 GiB or more is read or written at the wrong
place, without an error. At the paper's kernel cap each per-box bound array is larger than 2 GiB
(178,000,000 × 4 × 4 B), so on sm_120 set `CUDA_ROOT` to a CUDA ≥ 13 toolchain directory with
`bin/ptxas`, `bin/nvlink` and `nvvm/libdevice/libdevice.10.bc`:

```bash
CUDA_ROOT=/path/to/cuda-13 python experiments/cartpole/run.py verify --policy s27u
```

The paper's CartPole runs used H200 GPUs (sm_90) with the bundled ptxas. H200, A100 and CPU runs
need no setting.
