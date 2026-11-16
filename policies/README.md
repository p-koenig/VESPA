# Input policies

All policies are single axis-aligned decision trees stored as Orbax (OCDBT) checkpoints in the SYMPOL format
(depth-7 complete trees for SYMPOL, i.e. at most 127 splits and 128 leaves; VIPER trees are embedded in a complete
tree of their own depth). Load them with `vespa.pretraining.checkpoint.load_checkpoint(path)`, or with the loaders of
the verification / fine-tuning backends. Only the checkpoint files are shipped (`_CHECKPOINT_METADATA`, `_METADATA`,
`_sharding`, `_strings.json`, `manifest.ocdbt`, `d/`, `ocdbt.process_0/`, `array_metadatas/`); training logs and
evaluation side files were not copied. `SHA256SUMS` lists every file (`sha256sum -c SHA256SUMS` from this directory).

Naming: `s<seed>u` = the tree as trained (unpruned), `s<seed>p` = the same tree after pruning. Pruning
(`vespa.pretraining.pruning`) only collapses split nodes whose subtree is constant or unreachable, so a pruned tree takes the same
action as its unpruned tree in every state (checked on 500,000 states in the safe box and 200,000 states in a
box twice as large, plus 30 validation episodes); the array layout stays depth 7.

## CartPole (`cartpole/`)

Trained with SYMPOL PPO (`experiments/pretraining/cartpole.sh`) on `VerifyCartPole-v0` (CartPole-v1 dynamics,
500-step episodes), hyperparameters of the SYMPOL paper for CartPole, depth 7, one estimator, one random trial. The
checkpoint is the best periodic evaluation (ties keep the later one).

### `evaluation/` — the 24 evaluation input policies (12 seeds x pruned/unpruned)
| policy | training seed | variant | PPO environment steps | effective leaves (of 128) |
|---|---|---|---|---|
| `s6p` | 6 | pruned | 50,000,384 | 89 |
| `s6u` | 6 | unpruned | 50,000,384 | 89 |
| `s10p` | 10 | pruned | 50,000,384 | 80 |
| `s10u` | 10 | unpruned | 50,000,384 | 80 |
| `s12p` | 12 | pruned | 50,000,384 | 56 |
| `s12u` | 12 | unpruned | 50,000,384 | 56 |
| `s13p` | 13 | pruned | 50,000,384 | 83 |
| `s13u` | 13 | unpruned | 50,000,384 | 83 |
| `s14p` | 14 | pruned | 50,000,384 | 56 |
| `s14u` | 14 | unpruned | 50,000,384 | 56 |
| `s17p` | 17 | pruned | 50,000,384 | 30 |
| `s17u` | 17 | unpruned | 50,000,384 | 30 |
| `s20p` | 20 | pruned | 50,000,384 | 28 |
| `s20u` | 20 | unpruned | 50,000,384 | 28 |
| `s21p` | 21 | pruned | 50,000,384 | 56 |
| `s21u` | 21 | unpruned | 50,000,384 | 56 |
| `s27p` | 27 | pruned | 50,000,384 | 59 |
| `s27u` | 27 | unpruned | 50,000,384 | 59 |
| `s28p` | 28 | pruned | 50,000,384 | 52 |
| `s28u` | 28 | unpruned | 50,000,384 | 52 |
| `s42p` | 42 | pruned | 50,000,384 | 57 |
| `s42u` | 42 | unpruned | 50,000,384 | 57 |
| `s43p` | 43 | pruned | 50,000,384 | 45 |
| `s43u` | 43 | unpruned | 50,000,384 | 45 |

### `tuning/` — the 8 policies used for hyperparameter tuning (4 seeds x pruned/unpruned)
| policy | training seed | variant | PPO environment steps | effective leaves (of 128) |
|---|---|---|---|---|
| `s3p` | 3 | pruned | 29,700,608 | n/a |
| `s3u` | 3 | unpruned | 29,700,608 | n/a |
| `s9p` | 9 | pruned | 50,000,384 | 46 |
| `s9u` | 9 | unpruned | 50,000,384 | 46 |
| `s11p` | 11 | pruned | 50,000,384 | 53 |
| `s11u` | 11 | unpruned | 50,000,384 | 53 |
| `s16p` | 16 | pruned | 50,000,384 | 71 |
| `s16u` | 16 | unpruned | 50,000,384 | 71 |

Seed 3 was trained for 29,700,608 steps with the same command (`--total_steps 29700608`); its leaf count was not recorded at training time.

### `figure3/` — two further unpruned policies shown in Fig. 3 (bars s1 = `s8u`, s3 = `s40u`; s2 = `evaluation/s27u`)
| policy | training seed | variant | PPO environment steps | effective leaves (of 128) |
|---|---|---|---|---|
| `s8u` | 8 | unpruned | 50,000,384 | 76 |
| `s40u` | 40 | unpruned | 50,000,384 | 71 |

## MountainCarContinuous (`mountaincar/`)

Trained with the original SYMPOL code and command for `MountainCarContinuous-v0` (gymnax), 5 random trials of
1,002,320 steps each; the trial with the highest final test return (5 greedy episodes) is exported
(`experiments/pretraining/mountaincar/all.sh`). Environment id inside the checkpoint: `VerifyMountainCarContinuousGymnax-v0`.
`base` = exported tree, `base-pruned` = pruned copy.

### `evaluation/` — the 12 evaluation input policies (seeds 0-5)
| policy | training seed | variant | selected trial (of 5) | SYMPOL test return | steps per trial | effective leaves |
|---|---|---|---|---|---|---|
| `evaluation/s0/base` | 0 | unpruned | 5 | 97.72 | 1,002,320 | 128 (raw) |
| `evaluation/s0/base-pruned` | 0 | pruned | 5 | 97.72 | 1,002,320 | 74 |
| `evaluation/s1/base` | 1 | unpruned | 4 | 97.71 | 1,002,320 | 128 (raw) |
| `evaluation/s1/base-pruned` | 1 | pruned | 4 | 97.71 | 1,002,320 | 85 |
| `evaluation/s2/base` | 2 | unpruned | 4 | 95.09 | 1,002,320 | 128 (raw) |
| `evaluation/s2/base-pruned` | 2 | pruned | 4 | 95.09 | 1,002,320 | 81 |
| `evaluation/s3/base` | 3 | unpruned | 3 | 96.95 | 1,002,320 | 128 (raw) |
| `evaluation/s3/base-pruned` | 3 | pruned | 3 | 96.95 | 1,002,320 | 84 |
| `evaluation/s4/base` | 4 | unpruned | 2 | 96.35 | 1,002,320 | 128 (raw) |
| `evaluation/s4/base-pruned` | 4 | pruned | 2 | 96.35 | 1,002,320 | 94 |
| `evaluation/s5/base` | 5 | unpruned | 4 | 97.44 | 1,002,320 | 128 (raw) |
| `evaluation/s5/base-pruned` | 5 | pruned | 4 | 97.44 | 1,002,320 | 86 |

### `tuning/` — the 8 tuning policies (seeds 1001-1004; seed 1000 failed the base-policy gate and was not used)
| policy | training seed | variant | selected trial (of 5) | SYMPOL test return | steps per trial | effective leaves |
|---|---|---|---|---|---|---|
| `tuning/s1001/base` | 1001 | unpruned | 3 | 96.90 | 1,002,320 | 128 (raw) |
| `tuning/s1001/base-pruned` | 1001 | pruned | 3 | 96.90 | 1,002,320 | 93 |
| `tuning/s1002/base` | 1002 | unpruned | 1 | 95.03 | 1,002,320 | 128 (raw) |
| `tuning/s1002/base-pruned` | 1002 | pruned | 1 | 95.03 | 1,002,320 | 83 |
| `tuning/s1003/base` | 1003 | unpruned | 4 | 96.97 | 1,002,320 | 128 (raw) |
| `tuning/s1003/base-pruned` | 1003 | pruned | 4 | 96.97 | 1,002,320 | 82 |
| `tuning/s1004/base` | 1004 | unpruned | 4 | 94.91 | 1,002,320 | 128 (raw) |
| `tuning/s1004/base-pruned` | 1004 | pruned | 4 | 94.91 | 1,002,320 | 86 |

## VIPER (`viper/`)

21 VIPER trees for CartPole (leaf budgets 2-128, seeds 4-6), trained with the VIPER implementation of the DTPO
repository (see `baselines/viper/README.md`) and converted to the SYMPOL format with
`baselines/viper/tree2sympol.py`. `viper/manifest.json` holds the conversion checks for each tree.

| policy | leaf budget | seed | leaves | depth | return reported by the VIPER run |
|---|---|---|---|---|---|
| `viper_L2_s4` | 2 | 4 | 2 | 1 | 114.6 |
| `viper_L2_s5` | 2 | 5 | 2 | 1 | 174.7 |
| `viper_L2_s6` | 2 | 6 | 2 | 1 | 198.5 |
| `viper_L4_s4` | 4 | 4 | 3 | 2 | 254.8 |
| `viper_L4_s5` | 4 | 5 | 4 | 3 | 254.4 |
| `viper_L4_s6` | 4 | 6 | 4 | 3 | 490.3 |
| `viper_L8_s4` | 8 | 4 | 7 | 4 | 249.0 |
| `viper_L8_s5` | 8 | 5 | 7 | 5 | 84.5 |
| `viper_L8_s6` | 8 | 6 | 8 | 5 | 48.4 |
| `viper_L16_s4` | 16 | 4 | 15 | 5 | 311.2 |
| `viper_L16_s5` | 16 | 5 | 16 | 8 | 128.8 |
| `viper_L16_s6` | 16 | 6 | 16 | 7 | 185.2 |
| `viper_L32_s4` | 32 | 4 | 31 | 10 | 500.0 |
| `viper_L32_s5` | 32 | 5 | 31 | 9 | 500.0 |
| `viper_L32_s6` | 32 | 6 | 32 | 10 | 279.0 |
| `viper_L64_s4` | 64 | 4 | 64 | 10 | 51.5 |
| `viper_L64_s5` | 64 | 5 | 61 | 12 | 126.8 |
| `viper_L64_s6` | 64 | 6 | 64 | 11 | 293.7 |
| `viper_L128_s4` | 128 | 4 | 125 | 16 | 168.7 |
| `viper_L128_s5` | 128 | 5 | 128 | 14 | 213.2 |
| `viper_L128_s6` | 128 | 6 | 125 | 15 | 449.1 |
