"""Decision tree policies: checkpoint loading and leaf regions.

A SYMPOL checkpoint stores a soft decision tree ensemble. The deployed policy is
its hard version: at every internal node the feature with the largest selection
weight is compared with the node's threshold, and a leaf acts with the argmax of
its logits. ``leaf_regions`` turns that policy into axis-aligned boxes of the state
space, each with one action. Neighbouring leaves with the same action under a
common subtree are merged into one box.
"""

from __future__ import annotations

import itertools
import math
import os

import gymnasium as gym
import jax
import numpy as np
import orbax.checkpoint
import orbax.checkpoint.args as orbax_args

ENV_ID = "VerifyCartPole-v0"
_MAX_REGIONS = 10_000_000


def ensure_env_registered():
    """Register the CartPole env (500-step episodes) with gymnasium once."""
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="vespa.cartpole.cartpole_env:CartPoleEnv",
                     max_episode_steps=500)


def load_checkpoint(path: str) -> dict:
    """Restore a SYMPOL checkpoint onto the CPU.

    Returns {params, indices, actor_config (obs_dim, action_dim, depth,
    n_estimators, ...), action_type, env_id}.
    """
    cpu_sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.StandardCheckpointHandler())
    raw = checkpointer.restore(os.path.abspath(path),
                               args=orbax_args.StandardRestore(fallback_sharding=cpu_sharding))
    if str(raw.get("actor_type", "")) != "sympol":
        raise ValueError(f"{path}: expected a SYMPOL decision tree checkpoint")
    params = raw["actor"]["params"]
    indices = raw["actor"]["indices"]
    config = dict(raw.get("actor_config", {}))
    leaf_arr = params["leaf_array"]                       # (estimators, 2^depth, actions)
    config.setdefault("obs_dim", int(np.max(indices["features_by_estimator"])) + 1)
    config.setdefault("action_dim", leaf_arr.shape[2])
    config["depth"] = int(math.log2(leaf_arr.shape[1]))
    config["n_estimators"] = leaf_arr.shape[0]
    return {"params": params, "indices": indices, "actor_config": config,
            "action_type": str(raw.get("action_type", "discrete")),
            "env_id": str(raw.get("env_id", "CartPole-v1"))}


def _merged_subtree_boxes(params, indices, depth, obs_dim, e):
    """Boxes of estimator ``e``: one per maximal subtree whose leaves share an action.

    Each entry is (action, bounds, leftmost leaf id of the subtree). The left
    child of a node holds obs[feature] < threshold, the right child the rest.
    Boxes with an empty side (contradicting splits on a path) are dropped.
    """
    internal_node_num = 2 ** depth - 1
    split_idx_raw = np.asarray(params["split_idx_array"][e])
    split_vals = np.asarray(params["split_values"][e])
    features = np.asarray(indices["features_by_estimator"][e])
    leaf_dists = np.asarray(params["leaf_array"][e])

    node_sel_idx = np.argmax(split_idx_raw, axis=-1)
    node_obs_feat = features[node_sel_idx]
    threshold = split_vals[np.arange(internal_node_num), node_sel_idx]
    leaf_actions = [int(a) for a in np.argmax(leaf_dists, axis=-1)]

    # uniform_action[node] = the action of all leaves below the node, or None.
    uniform_action = [None] * (2 * (2 ** depth) - 1)
    for leaf in range(2 ** depth):
        uniform_action[internal_node_num + leaf] = leaf_actions[leaf]
    for nid in range(internal_node_num - 1, -1, -1):
        lc, rc = 2 * nid + 1, 2 * nid + 2
        if uniform_action[lc] is not None and uniform_action[lc] == uniform_action[rc]:
            uniform_action[nid] = uniform_action[lc]

    out = []

    def recurse(node_id, depth_at, bounds, leftmost_leaf):
        for d in range(obs_dim):
            if bounds[d][0] > bounds[d][1]:
                return
        if uniform_action[node_id] is not None or depth_at == depth:
            out.append((int(uniform_action[node_id]), bounds, leftmost_leaf))
            return
        feat = int(node_obs_feat[node_id])
        thr = float(threshold[node_id])
        left_bounds = [list(b) for b in bounds]
        if thr < left_bounds[feat][1]:
            left_bounds[feat][1] = thr
        right_bounds = [list(b) for b in bounds]
        if thr > right_bounds[feat][0]:
            right_bounds[feat][0] = thr
        half = 2 ** (depth - depth_at - 1)
        recurse(2 * node_id + 1, depth_at + 1, left_bounds, leftmost_leaf)
        recurse(2 * node_id + 2, depth_at + 1, right_bounds, leftmost_leaf + half)

    recurse(0, 0, [[-np.inf, np.inf] for _ in range(obs_dim)], 0)
    return out


def leaf_regions(checkpoint_path: str) -> list:
    """Leaf boxes of the hard policy in a checkpoint.

    Returns [{"leaf": representative leaf ids, "action": int,
    "bounds": [[lo, hi]] * 4}]. For an ensemble the boxes of all estimators are
    intersected and each non-empty intersection acts with the argmax of the
    ensemble's mixed logits.
    """
    info = load_checkpoint(checkpoint_path)
    params, indices = info["params"], info["indices"]
    cfg = info["actor_config"]
    if info["action_type"] != "discrete":
        raise ValueError(f"{checkpoint_path}: expected a discrete-action policy")
    depth, obs_dim, n_estimators = cfg["depth"], cfg["obs_dim"], cfg["n_estimators"]
    per_est = [_merged_subtree_boxes(params, indices, depth, obs_dim, e)
               for e in range(n_estimators)]
    if math.prod(max(len(b), 1) for b in per_est) > _MAX_REGIONS:
        raise ValueError(f"the leaf partition would exceed {_MAX_REGIONS} regions")

    est_weights = np.asarray(params["estimator_weights"])
    leaf_array = np.asarray(params["leaf_array"])
    regions = []
    for combo in itertools.product(*per_est):
        bounds = [[-np.inf, np.inf] for _ in range(obs_dim)]
        empty = False
        for _action, box, _rep in combo:
            for d in range(obs_dim):
                bounds[d][0] = max(bounds[d][0], box[d][0])
                bounds[d][1] = min(bounds[d][1], box[d][1])
                if bounds[d][0] > bounds[d][1]:
                    empty = True
                    break
            if empty:
                break
        if empty:
            continue
        rep = tuple(r for _a, _b, r in combo)
        if n_estimators == 1:
            action = int(combo[0][0])
        else:
            w = np.array([est_weights[e, rep[e]] for e in range(n_estimators)])
            w = np.exp(w - np.max(w))
            logits = (w / w.sum()) @ np.stack([leaf_array[e, rep[e]] for e in range(n_estimators)])
            action = int(np.argmax(logits))
        regions.append({"leaf": rep, "action": action,
                        "bounds": [[float(lo), float(hi)] for lo, hi in bounds]})
    return regions
