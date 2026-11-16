# VIPER baseline (CartPole)

The paper compares against 21 VIPER decision trees (Bastani, Pu & Solar-Lezama, NeurIPS 2018) for CartPole:
leaf budgets 2, 4, 8, 16, 32, 64, 128 x seeds 4, 5, 6. The converted trees are shipped in `policies/viper/`
(with `policies/viper/manifest.json`). This directory documents how they were produced.

## Training (third-party code, not included)

The trees were trained with the VIPER implementation of the DTPO repository
(Vos & Verwer, "Optimizing Interpretable Decision Tree Policies for Reinforcement Learning", arXiv:2408.11632),
<https://github.com/tudelft-cda-lab/DTPO>. That code is not redistributed here because the copy we used carries no
license file. Its upstream commit is not recorded. Obtain the code from the upstream repository.

Each run trains its own DQN teacher and then distils it into a tree with VIPER (`run_viper.py`, environment
`CartPole-v1` from gymnax, 500-step episodes). The 21 runs used the runner's defaults except leaf budget and seed:

```bash
# inside a checkout of the DTPO repository, CPU only, one process per run (21 runs in parallel)
export PYTHONPATH=$(pwd) MPLBACKEND=Agg OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
       XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for L in 2 4 8 16 32 64 128; do for S in 4 5 6; do
  python run_viper.py --env-name CartPole-v1 --output-dir out_viper/viper_${L}_${S} \
                      --max-leaf-nodes $L --seed $S &
done; done; wait
```

Settings recorded in each run's `config.json`:

| | value |
|---|---|
| DQN teacher | trained per run (`pretrained_model_path: null`), 10,000,000 steps, lr 2.5e-4, buffer 10,000, gamma 0.99, tau 1.0, target update every 500 steps, batch 128, epsilon 1 -> 0.05 over 50% of training, learning starts 10,000, train frequency 10 |
| VIPER | `max_leaf_nodes` = leaf budget, `max_depth` none, 10 batch rollouts, 50 test rollouts, max 200,000 samples, 80 iterations, train fraction 0.8, importance reweighting on, 1,000 evaluation rollouts |

Environment of the training runs (CPU, Python 3.11): torch 2.1.2 (CPU wheel), jax 0.4.23, jaxlib 0.4.23,
flax 0.7.5, optax 0.1.8, gymnax 0.0.6, gymnasium 0.29.1, gym 0.26.2, scikit-learn 1.2.2,
stable-baselines3 2.2.1, numpy 1.26.3, scipy 1.12.0, orbax-checkpoint 0.5.0 (the remaining packages from the
repository's minimal requirements file).

The return reported by each VIPER run (`results.json`, 1,000 rollouts) is listed in `policies/README.md`.
Most teachers did not reach the maximal return (see the paper's appendix).

## Conversion to the verifier format (`tree2sympol.py`, ours)

```bash
export PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu
python baselines/viper/tree2sympol.py out_viper policies_viper_converted
```

`tree2sympol.py` embeds each sklearn tree into a complete heap-ordered SYMPOL tree of the tree's own depth
(filler splits below early leaves, threshold `nextafter32(round_down32(t), +inf)` so that SYMPOL's strict `<`
reproduces sklearn's `<=` on float32 inputs) and checks the result against the sklearn tree: 200,000 sampled
states through the SYMPOL forward pass (0 mismatches for all 21 trees) and 10,000,000 states against the leaf
boxes that the CartPole verifier extracts (mismatches only for states exactly on a box boundary: 1-2 states for
4 of the 21 trees). Running it on the original VIPER outputs reproduces the parameters of `policies/viper/`
bit for bit.
