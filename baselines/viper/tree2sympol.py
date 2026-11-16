"""Convert VIPER decision trees into SYMPOL checkpoints that the VESPA verifier reads.

    python baselines/viper/tree2sympol.py <viper_out_root> <out_root>

``<viper_out_root>`` holds one directory per VIPER run (``viper_<L>_<seed>/``), each with the run directory
``CartPole-v1_viper_<timestamp>_<seed>/`` written by the VIPER runner (``tree.npz`` with the sklearn tree arrays
``feature``, ``threshold``, ``children_left``, ``children_right``, ``value``, plus ``config.json`` and
``results.json``). Every tree is embedded into a complete, heap-ordered SYMPOL tree of its own depth and saved as
``<out_root>/viper_L<leaf budget>_s<seed>/``. ``<out_root>/manifest.json`` records the checks.

Semantics: sklearn goes left iff x <= t (x float32), SYMPOL goes left iff x < t'. We set
t' = nextafter32(round_down32(t), +inf), which is exact for float32 inputs. Two checks per tree:
(a) the SYMPOL forward pass against the sklearn tree on 200,000 uniform states of the CartPole safe box,
(b) the leaf boxes extracted by the CartPole verifier against the sklearn tree on 10,000,000 states
    (states exactly on a box boundary are reported as ``on_region_boundary``).
Before converting, the heap index layout of the converter is checked against a shipped depth-7 SYMPOL policy.
"""
import argparse
import glob
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

from vespa.cartpole.policy import leaf_regions
from vespa.pretraining.checkpoint import load_checkpoint, save_checkpoint
from vespa.sympol import SYMPOL_RL

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_POLICY = os.path.join(HERE, "..", "..", "policies", "cartpole", "evaluation", "s6p")
SAFE_BOX_LOW = np.array([-2.4, -3, -0.2094, -3.5], np.float32)


def depth_of(cl, cr, n=0):
    """Depth of the sklearn tree below node ``n``."""
    return 0 if cl[n] == cr[n] else 1 + max(depth_of(cl, cr, cl[n]), depth_of(cl, cr, cr[n]))


def f32_thr(t):
    """Smallest float32 threshold t' with (x < t') == (x <= t) for every float32 x."""
    d = np.float32(t)
    if float(d) > float(t):
        d = np.nextafter(d, np.float32(-np.inf))
    return np.nextafter(d, np.float32(np.inf))


def build(T, D, nfeat=4, nact=2):
    """SYMPOL parameters and indices of a complete depth-``D`` tree that acts like the sklearn tree ``T``.

    Split logits are one-hot (10 for the split feature, -10 elsewhere) and leaf logits are 10 for the tree's action.
    An sklearn leaf above depth ``D`` becomes a filler split whose two subtrees carry the same leaf.
    """
    cl, cr, fe, th, val = T["children_left"], T["children_right"], T["feature"], T["threshold"], T["value"]
    NI, NL = 2**D - 1, 2**D
    sidx = np.full((NI, nfeat), -10.0, np.float32)
    sval = np.zeros((NI, nfeat), np.float32)
    leaf = np.full((NL, nact), -10.0, np.float32)

    def act_of(n):
        return int(np.argmax(np.asarray(val[n]).reshape(-1)[:nact]))

    def rec(h, n, d, frozen):
        if h >= NI:
            leaf[h - NI, act_of(n)] = 10.0
            return
        if frozen or cl[n] == cr[n]:
            sidx[h, 0] = 10.0
            sval[h, 0] = 0.0
            rec(2*h+1, n, d+1, True)
            rec(2*h+2, n, d+1, True)
            return
        f = int(fe[n])
        sidx[h, f] = 10.0
        sval[h, f] = f32_thr(th[n])
        rec(2*h+1, cl[n], d+1, False)
        rec(2*h+2, cr[n], d+1, False)

    rec(0, 0, 0, False)
    # Indices in heap order: the path of leaf l runs from the root to its parent, direction 0 = left.
    inl = np.zeros((NL, D), np.int32)
    pid = np.zeros((NL, D), np.float32)
    for leaf_index in range(NL):
        h = leaf_index + NI
        path = []
        while h > 0:
            p = (h - 1) // 2
            path.append((p, 0.0 if h == 2*p+1 else 1.0))
            h = p
        path = path[::-1]
        inl[leaf_index] = [p for p, _ in path]
        pid[leaf_index] = [b for _, b in path]
    params = {"estimator_weights": np.zeros((1, NL), np.float32), "leaf_array": leaf[None],
              "log_std": np.zeros((nact,), np.float32), "split_idx_array": sidx[None], "split_values": sval[None]}
    indices = {"features_by_estimator": np.arange(nfeat, dtype=np.int32)[None], "internal_node_index_list": inl,
               "path_identifier_list": pid}
    return params, indices


def sk_predict(T, X):
    """Actions of the sklearn tree ``T`` on the states ``X`` (left iff x <= t, compared in float64)."""
    cl, cr, fe, th, val = T["children_left"], T["children_right"], T["feature"], T["threshold"], T["value"]
    n = np.zeros(len(X), np.int64)
    for _ in range(64):
        internal = cl[n] != cr[n]
        if not internal.any():
            break
        go_left = X[np.arange(len(X)), np.where(internal, fe[n], 0)].astype(np.float64) <= th[n]
        n = np.where(internal, np.where(go_left, cl[n], cr[n]), n)
    v = np.asarray(val)[n].reshape(len(X), -1)
    return np.argmax(v[:, :2], 1)


def check_layout():
    """Assert that ``build`` reproduces the heap index layout of a SYMPOL policy of depth 7."""
    reference = load_checkpoint(REFERENCE_POLICY)
    single_leaf = {"children_left": np.array([-1]), "children_right": np.array([-1]), "feature": np.array([-2]),
                   "threshold": np.array([0.0]), "value": np.array([[[1.0, 0.0]]])}
    _, ind7 = build(single_leaf, 7)
    assert np.array_equal(np.asarray(reference["indices"]["internal_node_index_list"]),
                          ind7["internal_node_index_list"]), "node index layout differs"
    assert np.array_equal(np.asarray(reference["indices"]["path_identifier_list"]),
                          ind7["path_identifier_list"]), "path id layout differs"


def check_actions(T, path, depth, rng):
    """Check (a): mismatches of the SYMPOL forward pass against the sklearn tree on 200,000 states."""
    ck = load_checkpoint(path)
    actor = SYMPOL_RL(obs_dim=4, action_dim=2, action_type="discrete", depth=depth, n_estimators=1)
    prm = jax.tree.map(jnp.asarray, ck["params"])
    idx = jax.tree.map(jnp.asarray, ck["indices"])
    fwd = jax.jit(lambda o: jnp.argmax(actor.apply(prm, o, indices=idx), -1))
    mismatches, n_states = 0, 0
    for _ in range(20):  # batches, since the dense forward pass grows with 2^depth
        X = rng.uniform(SAFE_BOX_LOW, -SAFE_BOX_LOW, (10_000, 4)).astype(np.float32)
        mismatches += int((np.asarray(fwd(jnp.asarray(X))) != sk_predict(T, X)).sum())
        n_states += len(X)
    return {"n_states": n_states, "mismatches": mismatches}


def check_regions(T, path, rng):
    """Check (b): the verifier's leaf boxes against the sklearn tree on 10,000,000 states."""
    regs = leaf_regions(path)
    Bl = np.array([[b[0] for b in r["bounds"]] for r in regs])
    Bh = np.array([[b[1] for b in r["bounds"]] for r in regs])
    Ba = np.array([int(r["action"]) for r in regs])
    mismatches, n_states, ambiguous = 0, 0, 0
    for _ in range(100):
        X = rng.uniform(SAFE_BOX_LOW, -SAFE_BOX_LOW, (100_000, 4)).astype(np.float32)
        X64 = X.astype(np.float64)
        a_sk = sk_predict(T, X)
        a_reg = np.full(len(X), -1)
        for k in range(len(regs)):
            m = np.all((X64 >= Bl[k]) & (X64 <= Bh[k]), 1)
            ambiguous += int((m & (a_reg >= 0) & (a_reg != Ba[k])).sum())
            a_reg[m & (a_reg < 0)] = Ba[k]
        mismatches += int((a_reg != a_sk).sum())
        n_states += len(X)
    return {"n_states": n_states, "mismatches": mismatches, "on_region_boundary": ambiguous}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("src", help="VIPER output root (one viper_<L>_<seed>/ directory per run)")
    parser.add_argument("out", help="output root for the converted checkpoints and manifest.json")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    check_layout()
    rng = np.random.default_rng(0)
    manifest = {}
    for tree_file in sorted(glob.glob(os.path.join(args.src, "*", "*_viper_*", "tree.npz"))):
        run_dir = os.path.dirname(tree_file)
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
        name = f"viper_L{cfg['max_leaf_nodes']}_s{cfg['seed']}"
        assert int(os.path.basename(run_dir).rsplit("_", 1)[1]) == int(cfg["seed"]), f"seed mismatch in {run_dir}"
        T = dict(np.load(tree_file))
        D = max(1, depth_of(T["children_left"], T["children_right"]))
        params, indices = build(T, D)
        path = os.path.join(args.out, name)
        save_checkpoint(path, params, indices, {"action_dim": 2, "depth": D, "n_estimators": 1, "obs_dim": 4},
                        "discrete", "VerifyCartPole-v0")
        with open(os.path.join(run_dir, "results.json")) as f:
            viper_return = json.load(f)["mean_return"]
        record = {"max_leaf_nodes": cfg["max_leaf_nodes"], "seed": cfg["seed"], "depth": D,
                  "n_leaves": int((T["children_left"] == T["children_right"]).sum()),
                  "viper_reported_mean_return": viper_return,
                  "check_actions_vs_tree": check_actions(T, path, D, rng),
                  "check_regions_vs_tree": check_regions(T, path, rng)}
        manifest[name] = record
        print(name, record, flush=True)
        with open(os.path.join(args.out, "manifest.json"), "w") as f:
            json.dump(dict(sorted(manifest.items())), f, indent=1)


if __name__ == "__main__":
    main()
