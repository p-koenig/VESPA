# Pretraining the input policies

The paper's input policies are shipped in `policies/` (see `policies/README.md`). These scripts are the commands
that produced them. Run them from the repository root.

## CartPole (SYMPOL, VESPA environment, one GPU per seed)

```bash
experiments/pretraining/cartpole.sh 6          # one seed: train, prune, check (about 3 h on one GPU)
experiments/pretraining/cartpole_all.sh        # list the commands for all paper seeds (--run to execute them)
```

`cartpole.sh <seed> [out_root]` runs `python -m vespa.pretraining.ppo`: PPO with SYMPOL's CartPole hyperparameters
(one tree of depth 7), 50,000,000 steps (tuning seed 3: 29,700,608 steps), evaluation every 1,000,000 steps. The best
evaluated policy is saved to `<out_root>/s<seed>/checkpoint`. Then `python -m vespa.pretraining.cartpole_gate` prunes
it to `checkpoint-pruned`, checks that the pruned tree takes the same action as the unpruned one on 500,000 + 200,000
sampled states and validates both on 30 greedy episodes (`gate.json`). Evaluation seeds: 6, 10, 12, 13, 14, 17, 20,
21, 27, 28, 42, 43. Tuning seeds: 3, 9, 11, 16. Additional Fig. 3 policies: seeds 8 and 40.

## MountainCarContinuous (upstream SYMPOL code, separate environment, CPU)

```bash
experiments/pretraining/mountaincar/setup_sympol.sh                   # SYMPOL at 4f24add + export hook
experiments/pretraining/mountaincar/pretrain_seed.sh evaluation 0     # 5 trials, about 9 min on 16 CPU threads
experiments/pretraining/mountaincar/postprocess_seed.sh evaluation 0  # export best trial, prune, gate
experiments/pretraining/mountaincar/all.sh                            # list all commands (--run to execute them)
```

`pretrain_seed.sh` runs upstream SYMPOL's `ppo_gymnax.py` with the paper's command for `MountainCarContinuous-v0`
(`--random_trials 5 --use_best_config --action_type continuous --dynamic_buffer`, plus `--seed`, without W&B
tracking). `sympol_export_hook.patch` adds a hook that writes the final actor of each trial after training. It does
not change training. `postprocess_seed.sh` (VESPA environment) selects the trial with the best final test return,
converts it to a checkpoint (`vespa.pretraining.mountaincar.export`), prunes it (`vespa.pretraining.pruning`) and runs
the acceptance gate (`vespa.pretraining.mountaincar.gate`: pruning equivalence and at least 10% goal-reaching greedy
and sampled rollouts within 200 steps). Seeds: evaluation 0-5 (group `evaluation`). Tuning 1000-1004 (group
`tuning`), of which 1000 fails the gate, so the first four passing seeds 1001-1004 are used
(`vespa.pretraining.mountaincar.select_input_policies`).
