"""Parser tests of the two collectors on synthetic run directories in the release layout.

Run: python -m pytest tests/analysis   or   python tests/analysis/test_collectors.py   (from the repository root)
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'analysis'))
import collect_cartpole as C      # noqa: E402
import collect_mountaincar as MC  # noqa: E402

VAL1 = '[x] [finetune] val_return: before=500.00 after=201.73 (\u0394=-298.27) std=33.72 over 30 eps (greedy=True)\n'
VAL2 = '[x] [finetune] val_return: before=201.73 after=489.40 (\u0394=+287.67) std=36.96 over 30 eps (greedy=True)\n'


def worker(n):
    return f'[x] [verify-worker] tag=0-1_of_1 env=VerifyCartPole-v0 d=4 leaves={n} cap=178,000,000\n'


def test_log_entries(tmp_path):
    log = tmp_path / 'console.log'
    log.write_text(worker(25) + VAL1 + '[x] [pool] env=VerifyCartPole-v0 G=1 worker(s); K=24 leaves; cap=1\n' + worker(24),
                   encoding='utf-8')
    assert C.log_entries(str(log)) == [('leaves', '25'), ('val', '500.00', '201.73', '33.72', '30'), ('leaves', '24')]


def test_verify_record(tmp_path):
    d = tmp_path / 'iter_002'; d.mkdir()
    (d / 'iter_002.json').write_text(json.dumps({'box': {'active_total_per_step': [1, 2, 3], 'peak_active_total': 3},
                                                 'verify_sec': 10.0, 'n_segments': 1}))
    (d / 'status_0-1_of_1.json').write_text(json.dumps({'peak_pool_boxes': 4, 'seg_records': [
        {'order': 1, 'start_step': 0, 'end_step': 2, 'n_out': 3, 'peak_boxes': 3, 'hit_cap': False}]}))
    (d / 'pool_0-1_of_1.log').write_text(
        '[pool] env=VerifyCartPole-v0 G=1 worker(s); K=24 leaves; cap=1\n'
        '[pool] VRAM_POOL_FULL: device 0 89.4GiB > target\n'
        '[pool] TIERS at exit: vram 5 boxes 0.1GiB | min_step 87 ; peaks: vram 5/94.32GiB ram 0/0.00GiB\n')
    r = C.verify_record(str(d))
    assert r['peak_pool_boxes'] == 4 and r['seg'][0] == [1, 0, 2, 3, 3, 0]
    assert r['vram_full_gib'] == 89.4 and r['exit_min_step'] == 87 and r['peak_vram_gib'] == 94.32
    assert C.pool_leaves(str(d)) == 24
    m = C.metrics(r, 2)
    assert m['exact_upto'] == 2 and m['N_at_exact_end'] == 3


def _cp_run(root, name, with_pool_logs=True):
    rd = root / name; rd.mkdir()
    ar = [dict(iter=1, outcome='UNSAFE', depth=41, exact=True, accepted=True),
          dict(iter=2, outcome='UNSAFE', depth=41, exact=True, accepted=True),
          dict(iter=3, outcome='VRAM_CENSORED', depth_lb=86, exact=False, accepted=True, decision='accept', kind='vram')]
    (rd / 'summary.json').write_text(json.dumps({'status': 'VRAM_LIMITED', 'loop_result': {'accept_reject': ar}}))
    (rd / 'console.log').write_text(worker(25) + VAL1 + worker(24) + VAL2 + worker(21), encoding='utf-8')
    for i, k in ((1, 25), (2, 24), (3, 21)):
        d = rd / f'iter_{i:03d}'; d.mkdir()
        if with_pool_logs:
            (d / 'pool_0-1_of_1.log').write_text(f'[pool] env=VerifyCartPole-v0 G=1 worker(s); K={k} leaves; cap=1\n')
    return rd


def test_collect_cartpole_run(tmp_path):
    for pool in (True, False):           # leaf counts from the worker log of each iteration, else from console.log
        root = tmp_path / str(pool); root.mkdir()
        rows = C.collect_run(str(_cp_run(root, 'uninformed-s6p-f1', pool)))
        assert [r['method'] for r in rows] == ['Uninformed'] * 3 and rows[0]['policy'] == 's6p' and rows[0]['seed'] == 1
        assert [r['leaves'] for r in rows] == [25, 24, 21]
        assert [r['ret'] for r in rows] == [500.0, 201.73, 489.4] and rows[0]['ret_std'] is None
        assert [r['depth'] for r in rows] == [41, 41, 86] and [r['exact'] for r in rows] == [True, True, False]
        assert [r['ft'] for r in rows] == [0, 1, 2]
    assert C.collect_run(str(tmp_path / 'True')) is None                    # not a run name
    (tmp_path / 'verify-s8u').mkdir()
    assert C.collect_run(str(tmp_path / 'verify-s8u')) is None              # verification-only units are skipped


def test_collect_mountaincar(tmp_path):
    its = [dict(iter=1, certified_fraction=0.5, n_violating_pieces=3, n_init_pieces=8, incumbent_iter=1,
                incumbent_cert=0.5, strict_safe=0, val_return=80.0),
           dict(iter=2, certified_fraction=1.0, n_violating_pieces=0, n_init_pieces=8, incumbent_iter=2,
                incumbent_cert=1.0, strict_safe=1, val_return=90.0, ft_seed=100000, ft={'near_miss_n_boxes': 7})]
    for name in ('s0-pruned-vespa-k1', 's0-pruned-uninformed-k1', 'other-run'):
        d = tmp_path / name; d.mkdir()
        (d / 'summary.json').write_text(json.dumps({'per_seed': [dict(seed=100000, status='SAFE', best_iter=2, n_iters=2,
                                         n_rejects=0, n_accepts=1, certified_fraction_incumbent=1.0, iters=its)]}))
    (tmp_path / 's0-pruned-vespa-k1' / 'run_info.json').write_text(json.dumps({'sec': 12.5, 'end': 'x'}))
    rows, irows = MC.collect(str(tmp_path))
    assert [r['run'] for r in rows] == ['s0-pruned-uninformed-k1', 's0-pruned-vespa-k1']
    assert [r['method'] for r in rows] == ['Uninformed', 'VESPA'] and rows[1]['task_sec'] == 12.5
    assert rows[0]['policy'] == 's0-pruned' and rows[0]['k'] == 1 and rows[0]['safe_ratios'] == '0.50000;1.00000'
    assert rows[0]['vals'] == '80.00;90.00' and rows[0]['ft_seeds'] == '100000'
    assert len(irows) == 4 and irows[1]['near_miss_n_boxes'] == 7


if __name__ == '__main__':
    # plain-python runner (no pytest needed): python tests/analysis/test_collectors.py
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
