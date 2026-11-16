"""Unit tests of the table conventions (synthetic data only; no result files).

Run: python -m pytest tests/analysis   or   python tests/analysis/test_tables.py   (from the repository root)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'analysis'))
import make_tables as T              # noqa: E402
import mountaincar_per_run as M      # noqa: E402


def test_round_half_up():
    assert T.r(0.125, 2) == '0.13'
    assert T.r(2.5, 0) == '3'
    assert T.r(-1.25, 1) == '-1.3'
    assert T.r(85.15, 1) == '85.2'


def test_pm_sign():
    assert T.pm(15.24, 17.4, 1, sign=True) == r'+15.2\,{\scriptsize$\pm$17.4}'
    assert T.pm(-0.04, 1.0, 1, sign=True) == r'-0.0\,{\scriptsize$\pm$1.0}'


def test_lower_bound_ordering():
    assert T.cmp(90, True, 80, True) == 1          # exact vs exact: by value
    assert T.cmp(80, True, 80, True) == 0
    assert T.cmp(95, False, 90, True) == 1         # lower bound beats a smaller exact value
    assert T.cmp(85, False, 90, True) == 0         # lower bound below an exact value: cannot be ordered
    assert T.cmp(90, True, 95, False) == -1
    assert T.cmp(100, False, 50, False) == 0       # two lower bounds: never ordered


def _cp_rows():
    rows = []
    # one input policy, one seed, two methods; VESPA: accepted 40 -> rejected -> lower bound 60
    for method, seq in (('VESPA', [(40, True, True), (30, True, False), (60, False, True)]),
                     ('Uninformed', [(40, True, True), (45, True, True), (44, True, False)])):
        for i, (d, ex, acc) in enumerate(seq):
            rows.append(dict(run=f'r-{method}', method=method, policy='s1p', seed=0, iter=i + 1, ft=i, depth=d, exact=ex,
                             accepted=acc, ret=100.0 + i))
    return pd.DataFrame(rows)


def test_cp_incumbent_carry_forward():
    inc = T.cp_incumbents(_cp_rows(), n_k=4)
    v = inc[inc.method == 'VESPA'].set_index('k')
    assert list(v.depth) == [40, 40, 60, 60]            # rejected candidate skipped, last accepted carried
    assert list(v.exact) == [True, True, False, False]
    assert list(v.delta) == [0, 0, 20, 20]
    u = inc[inc.method == 'Uninformed'].set_index('k')
    assert list(u.depth) == [40, 45, 45, 45]


def test_tab_cp_row_format():
    rows = _cp_rows()
    rows2 = rows.copy(); rows2['seed'] = 1; rows2['run'] = rows2.run + '-b'
    inc = T.cp_incumbents(pd.concat([rows, rows2]), n_k=3)
    tex, df = T.tab_cp(inc)
    assert df.loc[2, 'W'] == 2 and df.loc[2, 'E'] == 0 and df.loc[2, 'n_lb_vespa'] == 2
    assert r'0 & 100\,{\scriptsize$\pm$0}' in tex and r'\midrule' in tex


def test_mcc_per_iteration_keeps_proof():
    it = pd.DataFrame([
        dict(run='s0-pruned-vespa-k1', iter=1, incumbent_iter=1, incumbent_cert=0.5, strict_safe=0, val_return=80.0),
        dict(run='s0-pruned-vespa-k1', iter=2, incumbent_iter=2, incumbent_cert=1.0, strict_safe=1, val_return=90.0),
    ])
    pr = M.per_iteration(it, n_k=4).set_index('k')
    assert list(pr.safe_ratio) == [0.5, 1.0, 1.0, 1.0]
    assert list(pr.safe) == [0, 1, 1, 1]
    assert list(pr.ret) == [80.0, 90.0, 90.0, 90.0]
    assert pr.pair.iloc[0] == 's0-pruned-k1' and pr.method.iloc[0] == 'VESPA'


def test_wel_mcc():
    V = pd.DataFrame(dict(pair=['a', 'b', 'c'], safe_ratio=[1.0, 0.5, 0.2]))
    U = pd.DataFrame(dict(pair=['a', 'b', 'c'], safe_ratio=[0.9, 0.5, 0.3]))
    assert T.wel_mcc(V, U) == (1, 1, 1)


if __name__ == '__main__':
    # plain-python runner (no pytest needed): python tests/analysis/test_tables.py
    import inspect, pathlib, tempfile, traceback
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    for n, f in tests:
        kw = {'tmp_path': pathlib.Path(tempfile.mkdtemp())} if 'tmp_path' in inspect.signature(f).parameters else {}
        try:
            f(**kw); print('PASS', n)
        except Exception:  # noqa: BLE001
            fails += 1; print('FAIL', n); traceback.print_exc()
    print(f'{len(tests) - fails} passed, {fails} failed')
    sys.exit(1 if fails else 0)
