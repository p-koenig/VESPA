"""Assemble the data of Fig. 3 (return over 1,000 episodes and verified safe steps of single policies).

    python analysis/fig3_data.py --returns-dir RET [--runs-dir RUNS] [--out fig3_data.csv] \
        [--policy PID[=RUN][:VARIANT] ...] [--viper-manifest policies/viper/manifest.json]

RET        output directory of analysis/eval_returns.py (ret1000_<PID>.json)
RUNS       CartPole run directory (default: $VESPA_CP_RUNS, else runs/cartpole); the verified safe steps of a policy
           are those of its verification-only run RUN (default verify-<PID>; ``summary.json``:
           ``loop_result.accept_reject[0]``; a verification that ran out of memory gives a lower bound,
           ``verified_kind = '>='``).
Default policies (the paper's Fig. 3): VIPER d3 = viper_L4_s6, d9 = viper_L32_s5, d10 = viper_L32_s4 and the
unpruned SYMPOL policies s1 = s8u, s2 = s27u, s3 = s40u (chosen with numpy default_rng(20261105) among the
pretraining seeds with a policy of return > 490).
"""
import argparse, json, os

import numpy as np
import pandas as pd

DEFAULT = ['viper_L4_s6', 'viper_L32_s5', 'viper_L32_s4', 's8u:unpruned', 's27u:unpruned', 's40u:unpruned']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--returns-dir', required=True)
    ap.add_argument('--runs-dir', default=os.environ.get('VESPA_CP_RUNS', os.path.join('runs', 'cartpole')))
    ap.add_argument('--out', default='fig3_data.csv'); ap.add_argument('--episodes', type=int, default=1000)
    ap.add_argument('--policy', action='append', default=None)
    ap.add_argument('--viper-manifest', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'policies', 'viper', 'manifest.json'))
    a = ap.parse_args(argv)
    man = json.load(open(a.viper_manifest)) if os.path.exists(a.viper_manifest) else {}
    rows = []
    for spec in (a.policy or DEFAULT):
        spec, _, variant = spec.partition(':')
        pid, _, run = spec.partition('=')
        run = run or f'verify-{pid}'
        J = json.load(open(os.path.join(a.returns_dir, f'ret{a.episodes}_{pid}.json')))
        r = np.array(J['rets']); te = np.array(J['term']); sT = np.array(J['sT'])
        cause = []
        for k in np.where(te)[0]:
            x, _, th, _ = sT[k]; cause.append('cart' if abs(x) > 2.4 else ('pole' if abs(th) > 0.2095 else '?'))
        summary = os.path.join(a.runs_dir, run, 'summary.json')
        if not os.path.exists(summary):
            raise SystemExit(f'no verification-only run {run} under {a.runs_dir} (expected {summary}). The Fig. 3 '
                             'policies are verified by `python experiments/cartpole/runs.py run --table viper` and '
                             '`python experiments/cartpole/runs.py run --table figure3`.')
        h = json.load(open(summary))['loop_result']['accept_reject'][0]
        exact = bool(h['exact']); val = h['depth_lb'] if 'depth_lb' in h else h['depth']
        if not variant and pid in man:
            m = man[pid]; variant = f"{m['max_leaf_nodes']} leaves nominal ({m['n_leaves']} encoded, depth {m['depth']})"
        rows.append(dict(policy_id=pid, N=J['N'], mean=r.mean(), std=r.std(), min=r.min(), n_fail=int(te.sum()),
                         n_lt200=int((te & (r < 200)).sum()),
                         cause_min=(cause[int(np.argmin(r[te]))] if te.any() else ''),
                         causes=dict(pd.Series(cause).value_counts()) if cause else {},
                         mean30=r[:30].mean(), offN=J['official_meanN'], cap=J['cap'],
                         ci95=1.96 * r.std(ddof=1) / np.sqrt(len(r)),
                         verified_value=float(val), verified_kind='exact' if exact else '>=', variant=variant))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(len(rows), 'policies ->', a.out)


if __name__ == '__main__':
    main()
