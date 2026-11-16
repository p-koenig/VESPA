"""Collect the MountainCarContinuous fine-tuning runs (read-only) into two CSV files.

    python analysis/collect_mountaincar.py [--runs-dir RUNS] [--out-dir OUT]

RUNS is the run directory of experiments/mountaincar/run_all.py (default: $VESPA_MCC_RUNS, else
runs/mountaincar) with one directory per run, named <policy>-<method>-k<k> (policy s<seed>-<pruned|unpruned>,
method vespa | uninformed, k = 1..5; see `python experiments/mountaincar/run_all.py --list`). Each run directory
holds the loop's summary.json (per_seed[0].iters: one record per verification) and run_info.json (wall time).

Writes
  OUT/runs.csv   one row per run:
                 run, policy, method, k, seed, status, best_iter, n_iters (verifications incl. the input policy),
                 n_rejects, n_accepts, safe_ratio_input, safe_ratio_final (ratio of safe start state regions of the
                 input policy / of the final accepted policy), n_init_pieces, nviol_input, safe_ratios (per
                 verification, ';'-separated), decisions (inc_decision:inc_reason per verification), undecided,
                 max_cap, vals (greedy validation returns per verification), ft_seeds, restart_steps, ptxas,
                 seed_sec, task_sec, end
  OUT/iters.csv  one row per verification, with the field names of summary.json:
                 run, iter, certified_fraction (ratio of safe start state regions), n_violating_pieces,
                 n_init_pieces, verify_status, strict_safe, undecided, inc_decision, inc_reason, n_rejects,
                 incumbent_iter, incumbent_cert (ratio of the accepted policy), ft_seed, ft_from_iter, val_return,
                 zono_cap_used, zono_peak, ptxas, ft_sec, verify_sec, near_miss_n_boxes, n_piece_fallback
``method`` is VESPA or Uninformed.
"""
import argparse, csv, glob, json, os, re, sys

RUN = re.compile(r"^(s\d+-(?:un)?pruned)-(vespa|uninformed)-k(\d+)$")
METHOD = {"vespa": "VESPA", "uninformed": "Uninformed"}
IF = ["iter", "certified_fraction", "n_violating_pieces", "n_init_pieces", "verify_status", "strict_safe", "undecided",
      "inc_decision", "inc_reason", "n_rejects", "incumbent_iter", "incumbent_cert", "ft_seed", "ft_from_iter",
      "val_return", "zono_cap_used", "zono_peak", "ptxas", "ft_sec", "verify_sec"]


def collect(runs_dir):
    rows, irows = [], []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "summary.json"))):
        name = os.path.basename(os.path.dirname(p))
        m = RUN.match(name)
        if not m:
            continue
        policy, method, k = m.groups()
        s = json.load(open(p))["per_seed"][0]; its = s["iters"]
        end = sec = None
        ri = os.path.join(os.path.dirname(p), "run_info.json")
        if os.path.exists(ri):
            rj = json.load(open(ri)); end = rj.get("end"); sec = rj.get("sec")
        ratios = [i.get("certified_fraction") for i in its]
        fts = [i for i in its if i.get("iter", 0) >= 2]
        rs = s.get("restart_steps")
        rows.append(dict(run=name, policy=policy, method=METHOD[method], k=int(k), seed=s.get("seed"),
            status=s.get("status"), best_iter=s.get("best_iter"), n_iters=s.get("n_iters"), n_rejects=s.get("n_rejects"),
            n_accepts=s.get("n_accepts"), safe_ratio_input=ratios[0], safe_ratio_final=s.get("certified_fraction_incumbent"),
            n_init_pieces=its[0].get("n_init_pieces"), nviol_input=its[0].get("n_violating_pieces"),
            safe_ratios=";".join("" if c is None else f"{c:.5f}" for c in ratios),
            decisions=";".join(str(i.get("inc_decision")) + ":" + str(i.get("inc_reason")) for i in its),
            undecided=sum(bool(i.get("undecided")) for i in its), max_cap=max((i.get("zono_cap_used") or 0) for i in its),
            vals=";".join("" if i.get("val_return") is None else f"{i['val_return']:.2f}" for i in its),
            ft_seeds=";".join(str(i.get("ft_seed")) for i in fts),
            restart_steps=";".join(str(x) for x in (rs if isinstance(rs, list) else [rs])),
            ptxas=its[0].get("ptxas"), seed_sec=s.get("seed_sec"), task_sec=sec, end=end))
        for i in its:
            d = {"run": name}; d.update({f: i.get(f) for f in IF})
            ft = i.get("ft") or {}
            d.update(near_miss_n_boxes=ft.get("near_miss_n_boxes"), n_piece_fallback=ft.get("n_piece_fallback"))
            irows.append(d)
    return rows, irows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default=os.environ.get("VESPA_MCC_RUNS", os.path.join("runs", "mountaincar")))
    ap.add_argument("--out-dir", default=os.path.join("results", "mountaincar"))
    a = ap.parse_args(argv)
    rows, irows = collect(a.runs_dir)
    if not rows:
        print(f"no run directories <policy>-<method>-k<k>/summary.json under {a.runs_dir}", file=sys.stderr)
        return 1
    os.makedirs(a.out_dir, exist_ok=True)
    for fn, rr in (("runs.csv", rows), ("iters.csv", irows)):
        out = os.path.join(a.out_dir, fn)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
        print(len(rr), "rows ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
