# Software environments

## Main environment (`requirements.txt`, `environment.yml`)

Used for everything except MountainCarContinuous pretraining: CartPole pretraining, pruning, verification,
fine-tuning (VESPA and Uninformed), the VIPER tree conversion, return evaluations and the analysis scripts.

```bash
conda env create -f environment/environment.yml      # Python 3.11 + the pins in requirements.txt
conda activate vespa
pip install -e . --no-deps
```

`requirements.txt` pins the result-relevant packages exactly (JAX 0.9.2 with the CUDA 12 plugin, flax 0.12.6,
optax 0.2.8, orbax-checkpoint 0.11.36, distrax 0.1.7, gymnax 0.0.9, gymnasium 1.2.3, numpy 2.4.3,
scipy 1.17.1). The paper environment ran Python 3.11.15. `pip_freeze_full.txt` is the complete `pip freeze`
of that environment; use it as a constraints file (`pip install -r requirements.txt -c pip_freeze_full.txt`)
to also pin every transitive dependency.

## SYMPOL-paper environment (`sympol_paper/`)

MountainCarContinuous input policies were trained with the original SYMPOL code (upstream commit `4f24add`)
and its own pinned requirements (Python 3.11.4, JAX 0.4.31, flax 0.8.4, optax 0.2.3, gymnax 0.0.8,
numpy 1.26.4), which differ from the main environment in every one of these packages.
`experiments/pretraining/mountaincar/setup_sympol.sh` creates this environment
(`sympol_paper/requirements.txt` with `sympol_paper/pip_freeze.txt` as constraints) and a patched checkout of
upstream SYMPOL. The paper runs used this environment on CPU only.

## Hardware used for the paper

- CartPole pretraining: one NVIDIA RTX PRO 6000 Blackwell (96 GB) per seed (tuning seed 3: one NVIDIA H200);
  about 3 h per 50M-step policy.
- MountainCarContinuous pretraining: CPU only (16 threads per seed, about 9 min per seed).
- Verification and fine-tuning: one GPU per run, see `experiments/cartpole/README.md` and
  `experiments/mountaincar/README.md`.

## CUDA toolchain note (ptxas)

JAX 0.9.2's CUDA 12 plugin ships `ptxas` 12.9. NVIDIA Blackwell GPUs (for example RTX PRO 6000) need `ptxas` 13 or
newer for the verifiers. `experiments/cartpole/README.md` and `experiments/mountaincar/README.md` describe the
setting. Other GPUs (for example H200, A100) and CPU runs use the bundled `ptxas`.
