"""Build the paper's result tables (tab:cp, tab:mcc, tab:pruning) as LaTeX and CSV.

    python analysis/make_tables.py --cp cp_per_verify.csv --mcc-per-run mcc_per_run.csv --mcc-runs mcc_runs.csv --out OUT

Inputs come from analysis/collect_cartpole.py and analysis/mountaincar_per_run.py. Conventions:
  * Each value describes the accepted policy after k fine-tuning iterations. CartPole: the last accepted
    verification with ft <= k (ft = 0 is the input policy); a verification that ran out of memory carries its
    depth forward as a lower bound (column '#>=' counts these runs). MountainCarContinuous: row k of the
    per-run table (a run that is proven safe keeps its policy).
  * mean and sample standard deviation (ddof=1) over runs, rounded half up.
  * Pruned and unpruned input policies are pooled in tab:cp and tab:mcc and split in tab:pruning.
  * W/E/L (CartPole): two exact values are ordered by value; a lower bound counts as more steps only if it
    exceeds the exact value of the other run; two lower bounds cannot be ordered (E).
    W/E/L (MountainCarContinuous): higher / equal / lower ratio of safe start state regions.
Writes OUT/tab_cp.{tex,csv}, OUT/tab_mcc.{tex,csv}, OUT/tab_pruning.{tex,csv} and OUT/mcc_progress.csv (data of the
MountainCarContinuous progress figure).
"""
import argparse, os
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

VESPA, UNINF = "VESPA", "Uninformed"     # values of the column ``method``


def r(x, nd):
    """round half up to nd decimals, as a string"""
    q = Decimal(1) if nd == 0 else Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def pm(m, s, nd, sign=False):
    v = r(m, nd)
    if sign and not v.startswith("-"):
        v = "+" + v
    return f"{v}\\,{{\\scriptsize$\\pm${r(s, nd)}}}"


def cp_incumbents(cp, n_k=12):
    inc = []
    for run, g in cp.groupby("run"):
        g = g.sort_values("ft")
        acc = g[g.accepted]
        g0 = g[g.ft == 0].iloc[0]
        assert bool(g0.accepted) and bool(g0.exact)
        for k in range(n_k):
            x = acc[acc.ft <= k].iloc[-1]
            inc.append(dict(run=run, method=x.method, policy=x.policy, seed=x.seed, k=k, depth=x.depth,
                            exact=bool(x.exact), ret=x.ret, d0=g0.depth))
    inc = pd.DataFrame(inc)
    inc["delta"] = inc.depth - inc.d0
    inc["var"] = inc.policy.str[-1].map({"p": "pruned", "u": "unpruned"})
    inc["bseed"] = inc.policy.str[:-1]
    return inc


def cmp(ad, ae, bd, be):
    if ae and be:
        return int(np.sign(ad - bd))
    if not ae and not be:
        return 0
    if not ae:
        return 1 if ad > bd else 0
    return -1 if bd > ad else 0


def wel_cp(V, U):
    V = V.set_index(["policy", "seed"]).sort_index(); U = U.set_index(["policy", "seed"]).sort_index()
    assert (V.index == U.index).all()
    c = [cmp(V.depth[i], V.exact[i], U.depth[i], U.exact[i]) for i in V.index]
    return c.count(1), c.count(0), c.count(-1)


def wel_mcc(V, U):
    d = V.set_index("pair").sort_index().safe_ratio - U.set_index("pair").sort_index().safe_ratio
    return int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())


HEAD = r"""\begin{tabular}{l cc@{\hspace{7pt}} c c@{\hspace{5pt}} c c@{\hspace{7pt}} ccc}
  \toprule
   & \multicolumn{2}{c}{Return} & \multicolumn{4}{c}{%s} & \multicolumn{3}{c}{Pairs} \\
  \cmidrule(l{2pt}r{7pt}){2-3}\cmidrule(l{2pt}r{7pt}){4-7}\cmidrule(l{2pt}r{0pt}){8-10}
  It. & \method{} & Uninf. & \method{} & %s & Uninf. & %s & W & E & L \\
  \midrule
"""


def tab_cp(inc):
    rows, csv = [], []
    for k in sorted(inc.k.unique()):
        s = inc[inc.k == k]; V, U = s[s.method == VESPA], s[s.method == UNINF]
        w, e, l = wel_cp(V, U)
        col = (lambda X: X.depth) if k == 0 else (lambda X: X.delta)
        cells = [str(k), pm(V.ret.mean(), V.ret.std(ddof=1), 0), pm(U.ret.mean(), U.ret.std(ddof=1), 0),
                 pm(col(V).mean(), col(V).std(ddof=1), 1, sign=k > 0), str(int((~V.exact).sum())),
                 pm(col(U).mean(), col(U).std(ddof=1), 1, sign=k > 0), str(int((~U.exact).sum())), str(w), str(e), str(l)]
        rows.append("  " + " & ".join(cells) + r" \\" + (r" \midrule" if k == 0 else ""))
        csv.append(dict(k=k, ret_vespa_mean=V.ret.mean(), ret_vespa_std=V.ret.std(ddof=1), ret_uninf_mean=U.ret.mean(),
                        ret_uninf_std=U.ret.std(ddof=1), steps_vespa_mean=col(V).mean(), steps_vespa_std=col(V).std(ddof=1),
                        n_lb_vespa=int((~V.exact).sum()), steps_uninf_mean=col(U).mean(), steps_uninf_std=col(U).std(ddof=1),
                        n_lb_uninf=int((~U.exact).sum()), W=w, E=e, L=l, n_runs=len(V)))
    tex = HEAD % (r"$\Delta$ Verified safe steps", r"\#$\geq$", r"\#$\geq$") + "\n".join(rows) + "\n  \\bottomrule\n\\end{tabular}\n"
    return tex, pd.DataFrame(csv)


def tab_mcc(mp):
    rows, csv = [], []
    for k in sorted(mp.k.unique()):
        s = mp[mp.k == k]; V, U = s[s.method == VESPA], s[s.method == UNINF]
        w, e, l = wel_mcc(V, U)
        cells = [str(k), pm(V.ret.mean(), V.ret.std(ddof=1), 0), pm(U.ret.mean(), U.ret.std(ddof=1), 0),
                 pm(100 * V.safe_ratio.mean(), 100 * V.safe_ratio.std(ddof=1), 1), str(int(V.safe.sum())),
                 pm(100 * U.safe_ratio.mean(), 100 * U.safe_ratio.std(ddof=1), 1), str(int(U.safe.sum())), str(w), str(e), str(l)]
        rows.append("  " + " & ".join(cells) + r" \\" + (r" \midrule" if k == 0 else ""))
        csv.append(dict(k=k, ret_vespa_mean=V.ret.mean(), ret_vespa_std=V.ret.std(ddof=1), ret_uninf_mean=U.ret.mean(),
                        ret_uninf_std=U.ret.std(ddof=1), safe_pct_vespa_mean=100 * V.safe_ratio.mean(),
                        safe_pct_vespa_std=100 * V.safe_ratio.std(ddof=1), n_safe_vespa=int(V.safe.sum()),
                        safe_pct_uninf_mean=100 * U.safe_ratio.mean(), safe_pct_uninf_std=100 * U.safe_ratio.std(ddof=1),
                        n_safe_uninf=int(U.safe.sum()), W=w, E=e, L=l, n_runs=len(V)))
    tex = HEAD % (r"Safe start state regions (\%)", r"\#safe", r"\#safe") + "\n".join(rows) + "\n  \\bottomrule\n\\end{tabular}\n"
    return tex, pd.DataFrame(csv)


def mcc_progress(mp):
    D = mp.copy(); D["safe_pct"] = 100 * D.safe_ratio
    A = D.groupby(["method", "k"]).agg(mean=("safe_pct", "mean"), median=("safe_pct", "median"),
                                    n_safe=("safe", "sum"), n=("run", "size")).reset_index()
    W = A.pivot(index="k", columns="method", values=["mean", "median", "n_safe", "n"])
    W.columns = [f"{a}_{s}" for s, a in W.columns]
    W = W[[f"{a}_{s}" for s in ["mean", "median", "n_safe", "n"] for a in ["VESPA", "Uninformed"]]]
    return W.round(4).astype({c: int for c in W.columns if c.endswith(("_n_safe", "_n"))})


def tab_pruning(inc, mp, mh, k_cp=11, k_mcc=5):
    fin = inc[inc.k == k_cp]; ini = inc[(inc.k == 0) & (inc.method == VESPA)]
    mp = mp.copy(); mh = mh.copy()
    mp["var"] = mp.pair.str.extract(r"-(pruned|unpruned)-")[0]
    mh["var"] = mh.policy.str.split("-").str[1]
    mh["iters"] = mh.ft_to_proof.fillna(k_mcc)
    mf = mp[mp.k == k_mcc]
    order_cp = [("pruned", VESPA), ("pruned", UNINF), ("unpruned", VESPA), ("unpruned", UNINF)]
    order_mc = order_cp
    mc2 = lambda x: r"\multicolumn{2}{c}{" + x + "}"
    L, csv = [], []
    n_cp = len(fin[(fin["var"] == "pruned") & (fin.method == VESPA)]); n_mc = len(mf[(mf["var"] == "pruned") & (mf.method == VESPA)])
    L.append(r"  \multicolumn{5}{l}{\textit{CP, %d runs per cell, after %d iterations}} \\" % (n_cp, k_cp))
    inp = [ini[ini["var"] == v].depth for v in ("pruned", "unpruned")]
    L.append("  Input & " + " & ".join(mc2(pm(x.mean(), x.std(ddof=1), 1)) for x in inp) + r" \\")
    S = [fin[(fin["var"] == v) & (fin.method == a)] for v, a in order_cp]
    L.append(r"  $\Delta$ steps & " + " & ".join(pm(s.delta.mean(), s.delta.std(ddof=1), 1, sign=True) for s in S) + r" \\")
    L.append(r"  \#$\geq$ & " + " & ".join(str(int((~s.exact).sum())) for s in S) + r" \\")
    L.append("  Return & " + " & ".join(pm(s.ret.mean(), s.ret.std(ddof=1), 0) for s in S) + r" \\")
    wl = [wel_cp(fin[(fin["var"] == v) & (fin.method == VESPA)], fin[(fin["var"] == v) & (fin.method == UNINF)]) for v in ("pruned", "unpruned")]
    L.append("  W/E/L & " + " & ".join(mc2("/".join(map(str, x))) for x in wl) + r" \\")
    L.append(r"  \midrule")
    for (v, a), s in zip(order_cp, S):
        csv.append(dict(env="CP", variant=v, method=a, delta_mean=s.delta.mean(), delta_std=s.delta.std(ddof=1),
                        n_lb=int((~s.exact).sum()), ret_mean=s.ret.mean(), ret_std=s.ret.std(ddof=1)))
    L.append(r"  \multicolumn{5}{l}{\textit{MC-C, %d runs per cell, after %d iterations}} \\" % (n_mc, k_mcc))
    inm = [100 * mp[(mp.k == 0) & (mp.method == VESPA) & (mp["var"] == v)].safe_ratio for v in ("pruned", "unpruned")]
    L.append(r"  Input (\%) & " + " & ".join(mc2(pm(x.mean(), x.std(ddof=1), 1)) for x in inm) + r" \\")
    M = [mf[(mf["var"] == v) & (mf.method == a)] for v, a in order_mc]
    L.append(r"  Safe reg.\ (\%) & " + " & ".join(pm(100 * s.safe_ratio.mean(), 100 * s.safe_ratio.std(ddof=1), 1) for s in M) + r" \\")
    L.append(r"  \#safe & " + " & ".join(str(int(s.safe.sum())) for s in M) + r" \\")
    its = [int(mh[(mh.method == a) & (mh["var"] == v)].iters.sum()) for v, a in order_mc]
    L.append("  Iterations & " + " & ".join(map(str, its)) + r" \\")
    L.append("  Return & " + " & ".join(pm(s.ret.mean(), s.ret.std(ddof=1), 0) for s in M) + r" \\")
    wm = [wel_mcc(mf[(mf["var"] == v) & (mf.method == VESPA)], mf[(mf["var"] == v) & (mf.method == UNINF)]) for v in ("pruned", "unpruned")]
    L.append("  W/E/L & " + " & ".join(mc2("/".join(map(str, x))) for x in wm) + r" \\")
    for (v, a), s, it in zip(order_mc, M, its):
        csv.append(dict(env="MC-C", variant=v, method=a, safe_pct_mean=100 * s.safe_ratio.mean(), safe_pct_std=100 * s.safe_ratio.std(ddof=1),
                        n_safe=int(s.safe.sum()), iterations=it, ret_mean=s.ret.mean(), ret_std=s.ret.std(ddof=1)))
    head = (r"\begin{tabular}{l cc@{\hspace{7pt}} cc}" "\n  \\toprule\n"
            r"   & \multicolumn{2}{c}{Pruned} & \multicolumn{2}{c}{Unpruned} \\" "\n"
            r"  \cmidrule(l{2pt}r{7pt}){2-3}\cmidrule(l{2pt}r{0pt}){4-5}" "\n"
            r"   & \method{} & Uninf. & \method{} & Uninf. \\" "\n  \\midrule\n")
    return head + "\n".join(L) + "\n  \\bottomrule\n\\end{tabular}\n", pd.DataFrame(csv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cp", required=True); ap.add_argument("--mcc-per-run", required=True)
    ap.add_argument("--mcc-runs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cp = pd.read_csv(a.cp); mp = pd.read_csv(a.mcc_per_run); mh = pd.read_csv(a.mcc_runs)
    inc = cp_incumbents(cp)
    for name, (tex, df) in (("tab_cp", tab_cp(inc)), ("tab_mcc", tab_mcc(mp)), ("tab_pruning", tab_pruning(inc, mp, mh))):
        open(os.path.join(a.out, name + ".tex"), "w").write(tex)
        df.to_csv(os.path.join(a.out, name + ".csv"), index=False)
    mcc_progress(mp).to_csv(os.path.join(a.out, "mcc_progress.csv"))
    print("tables ->", a.out)


if __name__ == "__main__":
    main()
