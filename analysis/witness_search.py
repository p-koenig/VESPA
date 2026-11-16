"""Sampled witness search for CartPole policies (paper: 10^8 start states per policy).

    python analysis/witness_search.py POLICIES.json OUT_DIR [N_total=100000000] [name,name,...]

POLICIES.json: {name: {"kind": "sk_tree", "path": ".../tree.npz"}}   sklearn / VIPER tree arrays (left iff x <= t)
           or  {name: {"kind": "sympol", "path": "<SYMPOL checkpoint dir>"}}   (left iff x < t, float32)
Start states: the 16 corners and a 9^4 grid of the CartPole start box [-0.05, 0.05]^4, then scrambled Sobol
points (seed 0) and uniform draws (seed 1) in the same box, N_total in all. Every start is rolled out for 500 steps
of the Gym CartPole dynamics (float64) with the policy acting on float32 observations, and the first step with
|x| > 2.4 or |theta| > 12 deg is recorded. A failure is a concrete witness that the policy is unsafe from that
start state (sampling can never prove safety). Writes OUT_DIR/<name>.json
(n, fail_200, fail_500, first_fail, cart, pole, ...) and OUT_DIR/<name>_fails.npy (up to 200,000 failing starts
with failure step and type). A GPU makes 10^8 starts take about 20 s per tree.
"""
import sys, os, json, time, itertools
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scipy.stats import qmc
G, MC, MP, L, F, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
MT = MC + MP; PML = MP * L; XT = 2.4; TH = 12 * 2 * np.pi / 360


def step(s, a):
    x, xd, th, thd = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    f = jnp.where(a == 1, F, -F); c, sn = jnp.cos(th), jnp.sin(th)
    temp = (f + PML * thd**2 * sn) / MT
    thacc = (G * sn - c * temp) / (L * (4.0 / 3.0 - MP * c**2 / MT)); xacc = temp - PML * thacc * c / MT
    return jnp.stack([x + TAU * xd, xd + TAU * xacc, th + TAU * thd, thd + TAU * thacc], 1)


def heap_policy(feat_map, SI, SV, LA):
    col = np.argmax(SI, 1); fe = jnp.asarray(feat_map[col], jnp.int32); th = jnp.asarray(SV[np.arange(len(col)), col].astype(np.float32))
    act = jnp.asarray(np.argmax(LA, 1), jnp.int32); NI = len(col); D = int(np.log2(NI + 1))
    def pol(o):
        h = jnp.zeros(o.shape[0], jnp.int32)
        for _ in range(D):
            x = jnp.take_along_axis(o, fe[h][:, None], 1)[:, 0]
            h = jnp.where(x < th[h], 2 * h + 1, 2 * h + 2).astype(jnp.int32)
        return act[h - NI]
    return pol


def make_policy(spec):
    if spec["kind"] == "sympol":   # SYMPOL checkpoint (single estimator), left iff x < t (float32)
        from vespa.pretraining.checkpoint import load_checkpoint
        ck = load_checkpoint(spec["path"]); p, i = ck["params"], ck["indices"]
        return heap_policy(np.asarray(i["features_by_estimator"])[0], np.asarray(p["split_idx_array"])[0],
                           np.asarray(p["split_values"])[0], np.asarray(p["leaf_array"])[0])
    Z = np.load(spec["path"], allow_pickle=True)
    if spec["kind"] == "sk_tree":   # sklearn / VIPER arrays, left iff x <= t (x float32 promoted)
        cl = jnp.asarray(Z["children_left"], jnp.int32); cr = jnp.asarray(Z["children_right"], jnp.int32); fe = jnp.asarray(np.maximum(Z["feature"], 0), jnp.int32)
        th = jnp.asarray(Z["threshold"].astype(np.float64)); val = np.asarray(Z["value"]).reshape(len(Z["children_left"]), -1)[:, :2]
        act = jnp.asarray(np.argmax(val, 1), jnp.int32)
        _cl, _cr = np.asarray(Z["children_left"]), np.asarray(Z["children_right"])
        def _d(n): return 0 if _cl[n] == _cr[n] else 1 + max(_d(_cl[n]), _d(_cr[n]))
        depth = max(1, _d(0))
        def pol(o):
            n = jnp.zeros(o.shape[0], jnp.int32); o64 = o.astype(jnp.float64)
            def body(_, n):
                internal = cl[n] != cr[n]
                go = jnp.take_along_axis(o64, fe[n][:, None], 1)[:, 0] <= th[n]
                return jnp.where(internal, jnp.where(go, cl[n], cr[n]), n).astype(jnp.int32)
            n = jax.lax.fori_loop(0, depth, body, n)
            return act[n]
        return pol
    raise ValueError(f"unknown policy kind {spec['kind']!r}")


def run_chunk(pol, S0, H=500):
    @jax.jit
    def go(S):
        def body(t, c):
            S, alive, tf, typ = c
            a = pol(S.astype(jnp.float32)); S2 = step(S, a)
            bx = alive & (jnp.abs(S2[:, 0]) > XT); bt = alive & (jnp.abs(S2[:, 2]) > TH) & ~bx
            tf = jnp.where(bx | bt, (t + 1).astype(jnp.int32), tf); typ = jnp.where(bx, jnp.int8(1), jnp.where(bt, jnp.int8(2), typ))
            return (jnp.where(alive[:, None], S2, S), alive & ~(bx | bt), tf, typ)
        n = S.shape[0]
        return jax.lax.fori_loop(0, H, body, (S, jnp.ones(n, bool), jnp.full(n, H + 1, jnp.int32), jnp.zeros(n, jnp.int8)))[2:]
    return go(S0)


def starts(N_total, chunk, seed=0):
    yield "structured", np.concatenate([np.array(list(itertools.product([-0.05, 0.05], repeat=4))),
                                        np.array(list(itertools.product(np.linspace(-0.05, 0.05, 9), repeat=4)))])
    n_sob = min(2**25, 2**int(np.log2(max(2, N_total // 2)))); sob = qmc.Sobol(4, scramble=True, seed=seed)
    done = 0
    while done < n_sob:
        m = min(chunk, n_sob - done); yield "sobol", sob.random(m) * 0.1 - 0.05; done += m
    rng = np.random.default_rng(seed + 1); rest = N_total - n_sob
    while rest > 0:
        m = min(chunk, rest); yield "uniform", rng.uniform(-0.05, 0.05, (m, 4)); rest -= m


if __name__ == "__main__":
    pols = json.load(open(sys.argv[1])); out = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000_000
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    os.makedirs(out, exist_ok=True); print("devices", jax.devices(), flush=True)
    for name, spec in pols.items():
        if only and name not in only: continue
        if os.path.exists(f"{out}/{name}.json"): print("skip", name); continue
        pol = make_policy(spec); t0 = time.time()
        n = f200 = f500 = cart = pole = 0; first = 10**9; fails = []
        for src, S0 in starts(N, 10_000_000):
            tf, typ = run_chunk(pol, jnp.asarray(S0)); tf = np.asarray(tf); typ = np.asarray(typ)
            n += len(S0); f200 += int((tf <= 200).sum()); f500 += int((tf <= 500).sum())
            cart += int((typ == 1).sum()); pole += int((typ == 2).sum())
            if (tf <= 500).any():
                first = min(first, int(tf.min())); idx = np.where(tf <= 500)[0]
                if sum(len(f) for f in fails) < 200_000: fails.append(np.concatenate([S0[idx], tf[idx, None], typ[idx, None]], 1)[:200_000])
        k = f500
        ub = 1 - 0.05 ** (1 / n) if k == 0 else None
        res = dict(name=name, n=n, fail_200=f200, fail_500=f500, frac_500=f500 / n, first_fail=None if first == 10**9 else first,
                   cart=cart, pole=pole, ub95_if_zero=ub, sec=round(time.time() - t0, 1))
        json.dump(res, open(f"{out}/{name}.json", "w"), indent=1)
        if fails: np.save(f"{out}/{name}_fails.npy", np.concatenate(fails)[:200_000])
        print(json.dumps(res), flush=True)
