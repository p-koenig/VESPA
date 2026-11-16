# VESPA: Verifiable Symbolic Policy Adaptation for Safe Reinforcement Learning

Paul König, Sascha Marton, Stefan Lüdtke, Christian Bartelt

This repository contains the implementation and the pretrained input policies of the paper
*VESPA: Verifiable Symbolic Policy Adaptation for Safe Reinforcement Learning*.

VESPA turns the result of a formal verification into a training signal. It verifies a decision tree policy with
reachability analysis, extracts counterexample regions from the violating states, fine-tunes the policy with
episodes that start in those regions, and verifies the fine-tuned policy again. The loop repeats until the policy
is proven safe or the iteration budget is used up.

| | Input policies | Uninformed (same loop, no counterexamples) | VESPA |
|---|---|---|---|
| CartPole: mean verified safe steps (horizon 200) | 63.8 | ≥ 85.2 | ≥ 92.7 |
| MountainCarContinuous: runs ending with a policy proven safe (of 60) | 0 | 46 | 53 |

## Repository layout

```
src/vespa/sympol.py        the decision tree policy (SYMPOL)
src/vespa/cartpole/        CartPole: box verifier, counterexample regions, fine-tuning, loop
src/vespa/mountaincar/     MountainCarContinuous: zonotope verifier, counterexample regions, fine-tuning, loop
src/vespa/pretraining/     training and pruning of the input policies
experiments/               one command per run and one driver per table, for each environment
analysis/                  from run directories to the tables and figures of the paper
policies/                  pretrained input policies (SYMPOL trees for both environments, VIPER trees)
baselines/viper/           how the VIPER trees were produced and converted
environment/               pinned software environment
tests/                     CPU tests
```

Each directory has a README with the details. The settings of the paper are fixed in
`experiments/cartpole/config.toml` and `src/vespa/mountaincar/config.py`.

## Installation

```bash
conda env create -f environment/environment.yml    # Python 3.11, JAX 0.9.2 (CUDA 12), pinned versions
conda activate vespa
pip install -e . --no-deps
```

All commands are run from the repository root. Every run needs one NVIDIA GPU. The tests and the analysis run
on CPU.

## Reproducing the paper

### 1. Run the experiments

| Paper result | Command | Runs | GPU used in the paper |
|---|---|---|---|
| CartPole results table and progress figure | `python experiments/cartpole/runs.py run --table tab_cp` | 96 | NVIDIA H200, one per run |
| Verified safe steps of the 21 VIPER trees | `python experiments/cartpole/runs.py run --table viper` | 21 | NVIDIA H200 |
| Verified safe steps of the SYMPOL trees in the return-vs-safety figure | `python experiments/cartpole/runs.py run --table figure3` | 3 | NVIDIA H200 |
| MountainCarContinuous results table and progress figure | `python experiments/mountaincar/run_all.py` | 120 | NVIDIA RTX PRO 6000 |

The runs are independent of each other. `runs.py run-index --table tab_cp --index I` and `run_all.py --index I`
run a single one, for example as one task of an array job (`experiments/slurm/`). Finished runs are skipped when
a driver is started again.

A single run is one input policy, one method and one fine-tuning seed:

```bash
python experiments/cartpole/run.py loop --policy s13u --method vespa --seed 0
python experiments/cartpole/run.py loop --policy s13u --method uninformed --seed 0
python experiments/cartpole/run.py verify --policy viper_L8_s4          # verification only

python -m vespa.mountaincar.run --policy policies/mountaincar/evaluation/s0/base-pruned \
    --method vespa --seed 100000 --run-name s0-pruned-vespa-k1
python -m vespa.mountaincar.verify --policy policies/mountaincar/evaluation/s0/base    # verification only
```

Runs write to `runs/cartpole/<run-name>/` and `runs/mountaincar/<run-name>/`.

In the paper, a CartPole run took between 11 minutes and 3.3 hours (median 33 minutes) and a
MountainCarContinuous run a median of 9 minutes.

### 2. Build the tables and figures

```bash
analysis/run_all.sh results
```

This reads `runs/cartpole` and `runs/mountaincar` and writes the three result tables (LaTeX and CSV) to
`results/tables/` and the three data figures to `results/figures/`. `analysis/README.md` lists every output file.

### 3. Optional: retrain the input policies

The input policies of the paper are in `policies/`, so this step is not needed to reproduce the results.
`experiments/pretraining/README.md` gives the commands that trained them.

## Hardware notes

* **CartPole results depend on the GPU memory.** The CartPole verifier keeps all boxes in GPU memory. When the
  memory is full, the result is a lower bound ("≥ d verified safe steps"). The paper used GPUs with 143 GB.
  `experiments/cartpole/README.md` explains what changes on a smaller GPU.
* **NVIDIA Blackwell GPUs (for example RTX PRO 6000) need a newer `ptxas`** than the one shipped with JAX 0.9.2.
  See `experiments/cartpole/README.md` and `experiments/mountaincar/README.md`.

## Tests

```bash
export JAX_PLATFORMS=cpu
for t in tests/*/test_*.py; do python $t || break; done
```

## Scope

* The refinement check described in the appendix of the paper is an additional analysis and not part of this
  implementation.
* The VIPER trees were trained with third-party code, which is not included. `baselines/viper/README.md` gives
  the source, the commands and our conversion script. The converted trees are in `policies/viper/`.

## Citation

```bibtex
@misc{koenig2026vespa,
  title  = {{VESPA}: Verifiable Symbolic Policy Adaptation for Safe Reinforcement Learning},
  author = {K{\"o}nig, Paul and Marton, Sascha and L{\"u}dtke, Stefan and Bartelt, Christian},
  year   = {2026}
}
```

## Authors

* Paul König and Christian Bartelt: CORE (Machine Learning and Cognitive Software), Institute for Software and
  Systems Engineering, Clausthal University of Technology
* Sascha Marton: NVIDIA
* Stefan Lüdtke: Marine Data Science, Institute for Visual and Analytic Computing, University of Rostock

## License and acknowledgements

The code is released under the MIT License (see `LICENSE`). The tree policies and their training build on
[SYMPOL](https://github.com/s-marton/sympol) (MIT License, © 2024 Sascha Marton). The environments follow
[gymnax](https://github.com/RobertTLange/gymnax).
