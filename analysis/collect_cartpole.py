"""Collect one row per verification from the CartPole fine-tuning runs (read-only).

    python analysis/collect_cartpole.py [--runs-dir RUNS] [--out cp_per_verify.csv]

RUNS is the run directory of the CartPole loop (default: $VESPA_CP_RUNS, else runs/cartpole), with one directory
per run as written by experiments/cartpole/run.py:
  <RUNS>/<method>-<policy>-f<seed>/   method vespa | uninformed (e.g. vespa-s13u-f0)
      summary.json                    block loop_result.accept_reject: one entry per verification
      console.log                     the loop's console output ('val_return: before=... after=... std=...' lines)
      iter_NNN/{iter_NNN.json, status_0-1_of_1.json, pool_0-1_of_1.log}
Iteration 1 is the verification of the input policy; iteration i > 1 verifies the policy after i - 1
fine-tunings. Verify-only runs (verify-<policy>) are not part of this table (see fig3_data.py).

Output columns (one row per verification):
run, method, policy, seed, iter, ft, outcome, depth, exact, accepted, decision, kind, leaves, peak_pool,
peak_worker, n_seg, seg1_end, seg1_nout, exit_min_step, vram_full_gib, peak_vram_gib, verify_sec,
ret, ret_std, exact_upto, N20, N30, N40, N50, N60, t1e6, t1e7, t1e8, N_at_exact_end.
``method`` is VESPA or Uninformed; ``depth`` is the number of verified safe steps (a lower bound when ``exact``
is False); ``ret``/``ret_std`` are mean/std of the greedy validation episodes of the verified policy (iteration 1:
the input policy's return logged before the first fine-tuning); ``leaves`` is the leaf count of the verified tree.
"""
import argparse, glob, json, os, re

import numpy as np
import pandas as pd

VAL = re.compile(r'val_return: before=([-\d.]+) after=([-\d.]+) \(.*?\) std=([-\d.]+) over (\d+) eps')
LEAVES = re.compile(r'\[verify-worker\] .*?\bleaves=(\d+)')
POOL_LEAVES = re.compile(r'\bK=(\d+) leaves\b')
POOL_FULL = re.compile(r'VRAM_POOL_FULL: device \d+ ([\d.]+)GiB')
TIERS = re.compile(r'TIERS at exit: .*? min_step (\d+|None) ; peaks: vram [\d,]+/([\d.]+)GiB')
RUN = re.compile(r'^(vespa|uninformed)-(s\d+[pu])-f(\d+)$')
METHOD = {'vespa': 'VESPA', 'uninformed': 'Uninformed'}
COLS = ['run', 'method', 'policy', 'seed', 'iter', 'ft', 'outcome', 'depth', 'exact', 'accepted', 'decision', 'kind',
        'leaves', 'peak_pool', 'peak_worker', 'n_seg', 'seg1_end', 'seg1_nout', 'exit_min_step', 'vram_full_gib',
        'peak_vram_gib', 'verify_sec', 'ret', 'ret_std', 'exact_upto', 'N20', 'N30', 'N40', 'N50', 'N60',
        't1e6', 't1e7', 't1e8', 'N_at_exact_end']


def jl(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def log_entries(path):
    """('val', before, after, std, n) and ('leaves', n) entries of one console log, in log order."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, errors='replace') as f:
        for line in f:
            m = VAL.search(line)
            if m:
                out.append(('val',) + m.groups()); continue
            m = LEAVES.search(line)
            if m:
                out.append(('leaves', m.group(1)))
    return out


def pool_leaves(d):
    """Number of leaves of the verified tree, from the verification worker log of one iter_NNN directory."""
    pl = os.path.join(d, 'pool_0-1_of_1.log')
    if os.path.exists(pl):
        with open(pl, errors='replace') as f:
            for line in f:
                m = POOL_LEAVES.search(line)
                if m:
                    return int(m.group(1))
    return None


def verify_record(d):
    """Per-verification fields from one iter_NNN directory."""
    out = {}
    its = sorted(glob.glob(os.path.join(d, 'iter_*.json')))
    it = jl(its[-1]) if its else None
    st = jl(os.path.join(d, 'status_0-1_of_1.json'))
    if it:
        bx = it.get('box') or {}
        out.update(active_per_step=bx.get('active_total_per_step'), peak_active_total=bx.get('peak_active_total'),
                   verify_sec=it.get('verify_sec'), n_segments=it.get('n_segments'))
    if st:
        out['peak_pool_boxes'] = st.get('peak_pool_boxes')
        out['seg'] = [[r['order'], r['start_step'], r['end_step'], r['n_out'], r['peak_boxes'], int(bool(r['hit_cap']))]
                      for r in st.get('seg_records', [])]
    out['exit_min_step'] = out['vram_full_gib'] = out['peak_vram_gib'] = None
    pl = os.path.join(d, 'pool_0-1_of_1.log')
    if os.path.exists(pl):
        with open(pl, errors='replace') as f:
            for line in f:
                m = POOL_FULL.search(line)
                if m and out['vram_full_gib'] is None:
                    out['vram_full_gib'] = float(m.group(1))
                m = TIERS.search(line)
                if m:
                    out['exit_min_step'] = None if m.group(1) == 'None' else int(m.group(1))
                    out['peak_vram_gib'] = float(m.group(2))
    return out


def metrics(vd, depth):
    a = vd.get('active_per_step')
    seg = vd.get('seg') or []
    if not a:
        return {}
    a = np.array(a, dtype=float)
    e1 = seg[0][2] if len(seg) > 1 else (len(a) - 1)
    if len(seg) <= 1:
        nz = np.nonzero(a)[0]
        e1 = int(nz.max()) if len(nz) else 0
    lim = min(e1, depth if depth is not None else e1)
    out = {'exact_upto': lim}
    for t in (20, 30, 40, 50, 60):
        out[f'N{t}'] = a[t] if t <= lim else np.nan
    for thr, name in [(1e6, 't1e6'), (1e7, 't1e7'), (1e8, 't1e8')]:
        idx = np.nonzero(a[:lim + 1] >= thr)[0]
        out[name] = int(idx[0]) if len(idx) else np.nan
    out['N_at_exact_end'] = a[lim]
    return out


def collect_run(rd):
    """Rows of one fine-tuning run directory (None if the name or summary.json does not match)."""
    run = os.path.basename(rd.rstrip('/'))
    m = RUN.match(run)
    s = jl(os.path.join(rd, 'summary.json'))
    if not m or not s or 'loop_result' not in s:
        return None
    method, policy, seed = METHOD[m.group(1)], m.group(2), int(m.group(3))
    log = log_entries(os.path.join(rd, 'console.log'))
    vals = [e for e in log if e[0] == 'val']
    log_leaves = [int(e[1]) for e in log if e[0] == 'leaves']
    rows = []
    for x in s['loop_result']['accept_reject']:
        it = x['iter']
        d = os.path.join(rd, f'iter_{it:03d}')
        vd = verify_record(d)
        dep = x['depth_lb'] if 'depth_lb' in x else x.get('depth')
        leaves = pool_leaves(d)
        if leaves is None and it - 1 < len(log_leaves):
            leaves = log_leaves[it - 1]
        if it == 1:
            ret, ret_std = (float(vals[0][1]) if vals else None), None
        elif it - 2 < len(vals):
            ret, ret_std = float(vals[it - 2][2]), float(vals[it - 2][3])
        else:
            ret = ret_std = None
        row = dict(run=run, method=method, policy=policy, seed=seed, iter=it, ft=it - 1,
                   outcome=x['outcome'], depth=dep, exact=bool(x['exact']),
                   accepted=bool(x['accepted']), decision=x.get('decision'), kind=x.get('kind'), leaves=leaves,
                   peak_pool=vd.get('peak_pool_boxes'), peak_worker=vd.get('peak_active_total'),
                   n_seg=vd.get('n_segments'),
                   seg1_end=(vd['seg'][0][2] if vd.get('seg') else None),
                   seg1_nout=(vd['seg'][0][3] if vd.get('seg') else None),
                   exit_min_step=vd.get('exit_min_step'), vram_full_gib=vd.get('vram_full_gib'),
                   peak_vram_gib=vd.get('peak_vram_gib'), verify_sec=vd.get('verify_sec'), ret=ret, ret_std=ret_std)
        row.update(metrics(vd, dep))
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs-dir', default=os.environ.get('VESPA_CP_RUNS', os.path.join('runs', 'cartpole')))
    ap.add_argument('--out', default='cp_per_verify.csv')
    a = ap.parse_args(argv)
    rows = []
    for rd in sorted(glob.glob(os.path.join(a.runs_dir, '*'))):
        r = collect_run(rd) if os.path.isdir(rd) else None
        rows += r or []
    if not rows:
        raise SystemExit(f'no fine-tuning runs (<vespa|uninformed>-<policy>-f<seed>/summary.json) under {a.runs_dir}')
    df = pd.DataFrame(rows).reindex(columns=COLS)
    df.to_csv(a.out, index=False)
    print(f'{df.run.nunique()} runs, {len(df)} verifications -> {a.out}')


if __name__ == '__main__':
    main()
