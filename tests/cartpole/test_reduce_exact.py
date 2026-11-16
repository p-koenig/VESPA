#!/usr/bin/env python3
"""The box reductions of the segment kernel are lossless.

Runs the three reductions as the kernel calls them (hash dedup -> Morton dedup
-> subset prune) on boxes with exact duplicates, duplicates that differ only in
the action history, 1-2 ULP near-duplicates and nested boxes, and checks:
  * every input box lies inside an output box of the same leaf (fp32 compare);
  * the two dedups drop only rows bit-identical in (lo, hi, leaf) to a kept row;
  * the counters agree (no non-identical drop, no protrusion, no hash collision).

    JAX_PLATFORMS=cpu PYTHONPATH=src python tests/cartpole/test_reduce_exact.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax.numpy as jnp
from vespa.cartpole import interval_core as ic

RNG = np.random.default_rng(20261030)
D, K = 4, 5
AX_LO = jnp.asarray([-2.4, -3.0, -0.21, -3.0], jnp.float32)
AX_HI = jnp.asarray([2.4, 3.0, 0.21, 3.0], jnp.float32)


def make_boxes(n_base=1500):
    c = RNG.uniform([-1, -1, -0.15, -1], [1, 1, 0.15, 1], (n_base, D))
    w = RNG.uniform(1e-4, 5e-3, (n_base, D))
    lo = (c - w).astype(np.float32)
    hi = (c + w).astype(np.float32)
    leaf = RNG.integers(0, K, n_base)
    ah = RNG.integers(0, 2, (n_base, 2)).astype(np.uint8)
    rows = [(lo, hi, leaf, ah)]
    i = RNG.choice(n_base, 300, replace=False)                 # exact duplicates
    rows.append((lo[i].copy(), hi[i].copy(), leaf[i].copy(), ah[i].copy()))
    i = RNG.choice(n_base, 300, replace=False)                 # other action history
    rows.append((lo[i].copy(), hi[i].copy(), leaf[i].copy(), (1 - ah[i]).astype(np.uint8)))
    i = RNG.choice(n_base, 400, replace=False)                 # near duplicates (ULPs)
    l2, h2 = lo[i].copy(), hi[i].copy()
    for _ in range(2):
        ax = RNG.integers(0, D, len(i))
        l2[np.arange(len(i)), ax] = np.nextafter(l2[np.arange(len(i)), ax], np.float32(-10))
        ax = RNG.integers(0, D, len(i))
        h2[np.arange(len(i)), ax] = np.nextafter(h2[np.arange(len(i)), ax], np.float32(10))
    rows.append((l2, h2, leaf[i].copy(), ah[i].copy()))
    i = RNG.choice(n_base, 300, replace=False)                 # nested boxes
    f = RNG.uniform(0.1, 0.4, (len(i), D)).astype(np.float32)
    span = hi[i] - lo[i]
    rows.append(((lo[i] + f * span).astype(np.float32), (hi[i] - f * span).astype(np.float32),
                 leaf[i].copy(), ah[i].copy()))
    lo, hi, leaf, ah = (np.concatenate([r[j] for r in rows]) for j in range(4))
    p = RNG.permutation(len(lo))
    return lo[p], hi[p], leaf[p].astype(np.uint8), ah[p]


def run_stack(lo, hi, leaf, ah, cap_pad=256):
    n = len(lo)
    cap = n + cap_pad
    pad = lambda a: np.concatenate([a, np.zeros((cap - n,) + a.shape[1:], a.dtype)])
    L, H, LF = jnp.asarray(pad(lo)), jnp.asarray(pad(hi)), jnp.asarray(pad(leaf))
    O, A = jnp.asarray(pad(np.zeros(n, np.uint8))), jnp.asarray(pad(ah))
    l1, h1, f1, o1, n1, a1, q_st = ic._hash_dedup(L, H, LF, O, jnp.int64(n), cap=cap, act_hist=A)
    l2, h2, f2, o2, n2, a2, m_st = ic._morton_dedup(l1, h1, f1, o1, n1, cap=cap, axis_lo=AX_LO,
                                                    axis_hi=AX_HI, act_hist=a1)
    l3, h3, f3, o3, n3, a3 = ic._subset_prune(l2, h2, f2, o2, n2, cap, K, act_hist=a2)
    cut = lambda x, m: np.asarray(x)[:int(m)]
    return dict(mzo=(cut(l2, n2), cut(h2, n2), cut(f2, n2)),
                out=(cut(l3, n3), cut(h3, n3), cut(f3, n3)),
                n=(n, int(n1), int(n2), int(n3)),
                q_st=[np.asarray(x) for x in q_st], m_st=[np.asarray(x) for x in m_st])


def covered(lo, hi, leaf, olo, ohi, oleaf):
    """bool[n]: input row contained in some output row of the same leaf."""
    ok = np.zeros(len(lo), bool)
    for k in np.unique(leaf):
        I, J = np.flatnonzero(leaf == k), np.flatnonzero(oleaf == k)
        for s in range(0, len(I), 512):
            ii = I[s:s + 512]
            c = (np.all(olo[J][None] <= lo[ii][:, None], axis=2)
                 & np.all(hi[ii][:, None] <= ohi[J][None], axis=2))
            ok[ii] = c.any(axis=1)
    return ok


def identical_in(lo, hi, leaf, olo, ohi, oleaf):
    """bool[n]: input row has a bit-identical (lo, hi, leaf) row in the output."""
    key = lambda a, b, c: [x.tobytes() for x in np.concatenate(
        [a.view(np.uint32), b.view(np.uint32), c[:, None].astype(np.uint32)], 1)]
    s = set(key(olo, ohi, oleaf))
    return np.array([k in s for k in key(lo, hi, leaf)])


def main():
    lo, hi, leaf, ah = make_boxes()
    r = run_stack(lo, hi, leaf, ah)
    n_uncovered = int((~covered(lo, hi, leaf, *r["out"])).sum())
    n_not_identical = int((~identical_in(lo, hi, leaf, *r["mzo"])).sum())
    print(f"n in/hash/morton/prune = {r['n']} uncovered={n_uncovered} "
          f"dedup drops without identical survivor={n_not_identical}")
    assert n_uncovered == 0, n_uncovered
    assert n_not_identical == 0, n_not_identical
    assert int(r["q_st"][0]) == 0 and int(r["m_st"][0]) == 0      # only bit-identical drops
    assert float(np.max(r["q_st"][3])) == 0.0                     # no protrusion
    assert r["n"][0] - r["n"][1] >= 600, r["n"]                   # 300 + 300 duplicates dropped
    assert int(r["q_st"][1]) >= 300                               # ... 300 with another history
    assert int(r["q_st"][2]) == 0                                 # no hash collisions
    assert r["n"][3] < r["n"][2]                                  # nested boxes pruned
    print(f"PASS test_reduce_exact: lossless, {r['n'][0]} -> {r['n'][3]} boxes")


if __name__ == "__main__":
    main()
