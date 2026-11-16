"""Per-run MountainCarContinuous tables used by the paper tables and the MC-C progress figure.

    python analysis/mountaincar_per_run.py RUNS_CSV ITERS_CSV OUT_DIR

From the collector output (``runs.csv``, ``iters.csv`` of analysis/collect_mountaincar.py) it writes
  OUT_DIR/mcc_runs.csv     one row per run: safe, best_iter, ft_to_proof (fine-tuning iterations until the proof),
                           n_ft, safe_ratio_input, safe_ratio_final, ret_input, ret_best, ...
  OUT_DIR/mcc_per_run.csv  one row per run and iteration k = 0..5 (the accepted policy after k fine-tuning
                           iterations): run, method, pair, k, safe_ratio (ratio of safe start state regions),
                           ret (30-episode greedy return), safe (1 once the run is proven safe).
``pair`` is the run name without the method, i.e. the VESPA and the Uninformed run of one policy and seed.
"""
import sys, os
import numpy as np
import pandas as pd

METHOD = r'-(vespa|uninformed)-'
LABEL = {'vespa': 'VESPA', 'uninformed': 'Uninformed'}


def run_table(R):
    h = R.copy()
    h['safe'] = h.status.eq('SAFE')
    h['ft_to_proof'] = np.where(h.safe, h.best_iter - 1, np.nan)
    h['n_ft'] = h.n_iters - 1
    def val_at_best(r):
        v = str(r.vals).split(';'); i = int(r.best_iter) - 1
        try: return float(v[i])
        except (ValueError, IndexError): return np.nan
    h['ret_best'] = h.apply(val_at_best, axis=1)
    h['ret_input'] = h.vals.astype(str).str.split(';').str[0].astype(float)
    cols = ['run', 'policy', 'method', 'k', 'seed', 'status', 'safe', 'best_iter', 'ft_to_proof', 'n_ft', 'n_rejects',
            'safe_ratio_input', 'safe_ratio_final', 'ret_input', 'ret_best', 'undecided', 'max_cap', 'task_sec']
    return h[cols].sort_values(['policy', 'k', 'method'])


def per_iteration(it, n_k=6):
    it = it.copy()
    it['method'] = it.run.str.extract(METHOD)[0].map(LABEL)
    it['pair'] = it.run.str.replace(METHOD, '-', regex=True)
    ret = it.set_index(['run', 'iter']).val_return
    rows = []
    for run, g in it.groupby('run'):
        g = g.sort_values('iter')
        inc = None
        for k in range(0, n_k):
            sub = g[g.iter == k + 1]
            if len(sub):
                r_ = sub.iloc[0]
                incit = int(r_.incumbent_iter)
                inc = (incit, r_.incumbent_cert, ret[(run, incit)], int(r_.strict_safe == 1 and r_.incumbent_iter == r_.iter) or (inc[3] if inc else 0))
            rows.append(dict(run=run, method=g.method.iloc[0], pair=g.pair.iloc[0], k=k, safe_ratio=inc[1], ret=inc[2], safe=inc[3]))
    return pd.DataFrame(rows)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        raise SystemExit(__doc__)
    R = pd.read_csv(argv[0]); I = pd.read_csv(argv[1]); out = argv[2]
    os.makedirs(out, exist_ok=True)
    run_table(R).to_csv(os.path.join(out, 'mcc_runs.csv'), index=False)
    per_iteration(I).to_csv(os.path.join(out, 'mcc_per_run.csv'), index=False)
    print('wrote', os.path.join(out, 'mcc_runs.csv'), os.path.join(out, 'mcc_per_run.csv'))


if __name__ == '__main__':
    main()
