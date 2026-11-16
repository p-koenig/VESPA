"""Paper configuration of the MountainCarContinuous experiments.

Every value that defines the paper results is listed here once. The two methods differ only in
the mixing probability p of the fine-tuning start distribution.
"""

ENV_ID = "VerifyMountainCarContinuousGymnax-v0"

# Reachability property: every trajectory from the start set reaches x >= 0.45 within 200 steps.
HORIZON = 200

# Verifier: the start set is bisected 14 times into 2^14 start state regions. The no-merge zonotope
# verifier runs in float32 with a static buffer of 262,144 objects. A candidate whose validation return
# shows that it reaches the goal is verified again with larger buffers when the buffer overflows.
INIT_SPLITS = 14
ZONOTOPE_CAP = 262_144
ESCALATION_CAPS = (4_194_304, 16_777_216)
UNSAFE_CAPTURE_CAP = 1_048_576

# Counterexample regions: at most this many captured violating objects are walked back in time.
MAX_COUNTEREXAMPLE_OBJECTS = 250_000

# Loop: the base verification plus up to five fine-tuning iterations. A rejected candidate is
# replaced by a new fine-tuning of the incumbent with seed + REJECT_SEED_STRIDE * (number of rejects).
MAX_VERIFICATIONS = 6
REJECT_SEED_STRIDE = 1000
N_VAL_EPISODES = 30

# Mixing probability p: share of fine-tuning episodes that start in a counterexample region.
MIX_P = {"vespa": 0.5, "uninformed": 0.0}

# PPO fine-tuning.
TOTAL_STEPS = 100_000
N_ENVS = 5
N_STEPS = 128
N_UPDATE_EPOCHS = 8
MINIBATCH_SIZE = 64
GAMMA = 0.999
GAE_LAMBDA = 0.99
CLIP_COEF = 0.1
ENT_COEF = 0.1
VF_COEF = 0.5
MAX_GRAD_NORM = 1000.0
HORIZON_MARGIN = 1
LOG_STD_MAX_AT_LOAD = -1.0
CRITIC_LR = 0.0020613382527496695
CRITIC_NUM_LAYERS = 2
CRITIC_NEURONS_PER_LAYER = 256

# Learning rate and optimizer per SYMPOL parameter group (Adam or AdamW, as in SYMPOL).
ACTOR_OPTIMIZER = {
    "estimator_weights": ("adam", 0.00012314339663197326),
    "split_values": ("adam", 0.0001666108178422791),
    "split_idx_array": ("adamw", 0.00014576617673214828),
    "leaf_array": ("adamw", 0.004),
    "log_std": ("adamw", 0.01),
}

# Memory settings of the paper runs (no influence on results).
XLA_ENV = {"XLA_PYTHON_CLIENT_ALLOCATOR": "platform", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
