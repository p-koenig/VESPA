#!/usr/bin/env python3
"""Accept/reject rules of the loop with scripted verifier outcomes.

The workers are replaced by a script: each verification writes a status file
(and violating boxes), each fine-tuning a candidate directory. Checked:
  a) input policy censored (pool full)          -> VRAM_LIMITED_AT_BASE (>= d)
  b) exact 42, then censored d = 59 > 42        -> accept, VRAM_LIMITED
  c) exact 42, censored d = 30 <= 42            -> reject, the incumbent is
     fine-tuned again with seed 1000; exact 50 accepted, 44 rejected; MAX_ITERS
  d) non-exact UNSAFE                           -> lower bound only
  e) a stopped run resumes after its last complete iteration

    JAX_PLATFORMS=cpu PYTHONPATH=src python tests/cartpole/test_vram_censor_loop.py
"""
import json
import os
import tempfile

import numpy as np

from vespa.cartpole import loop as L


class ScriptedLoop(L.Loop):
    def __init__(self, run_dir, script, max_iterations):
        settings = {"horizon": 200, "max_iterations": max_iterations}
        super().__init__(run_dir, "ckpts/base", settings, seed=0)
        self.script, self.calls = list(script), []

    def run_worker(self, module, args):
        get = lambda k: args[args.index(k) + 1]
        os.makedirs(get("--out"), exist_ok=True)
        if module.endswith("verify_worker"):
            kind, kw = self.script.pop(0)
            st = {"UNSAFE": "UNSAFE", "FULL": "VRAM_POOL_FULL", "SAFE": "SAFE"}[kind]
            json.dump({"status": st, "first_unsafe_step": kw.get("fus"),
                       "first_unsafe_exact": kw.get("exact", kind == "UNSAFE"),
                       "pool_min_step": kw.get("min_step"), "boxes_per_step": [1, 2, 1]},
                      open(os.path.join(get("--out"), L.STATUS_FILE), "w"))
            if kind == "UNSAFE":
                np.savez(os.path.join(get("--out"), L.BOXES_FILE), first_unsafe_step=kw["fus"])
            self.calls.append(("verify", get("--checkpoint")))
        else:
            os.makedirs(get("--out-checkpoint"), exist_ok=True)
            json.dump({"verdict": "OK", "finetune": {}},
                      open(os.path.join(get("--out"), "status_finetune.json"), "w"))
            self.calls.append(("finetune", get("--checkpoint"), get("--seed"),
                               os.path.basename(os.path.dirname(get("--boxes")))))
        return 0


def run(script, max_iterations=6, run_dir=None):
    run_dir = run_dir or tempfile.mkdtemp(prefix="vespa_loop_")
    lp = ScriptedLoop(run_dir, script, max_iterations)
    status = lp.run()
    summ = json.load(open(os.path.join(run_dir, "summary.json")))["loop_result"]
    return status, summ, lp.calls, lp.history, run_dir


def main():
    D = L.censor_decision
    assert D(None, 57) == "AT_BASE" and D({"depth": 42}, 59) == "ACCEPT_STOP"
    assert D({"depth": 42}, 42) == "REJECT" and D({"depth": 42}, None) == "NO_DEPTH"

    st, sm, calls, *_ = run([("FULL", dict(min_step=58))])
    assert st == "VRAM_LIMITED_AT_BASE" and sm["vram_limited_ge"] == 57, (st, sm)
    assert [c[0] for c in calls] == ["verify"]
    assert sm["accept_reject"][0]["accepted"] is False and sm["n_rejects"] == 0
    print(f"a) {st} >= {sm['vram_limited_ge']}")

    st, sm, calls, *_ = run([("UNSAFE", dict(fus=43)), ("FULL", dict(min_step=60))])
    assert st == "VRAM_LIMITED" and sm["vram_limited_ge"] == 59
    assert sm["incumbent"]["depth"] == 59 and sm["incumbent"]["censored"] is True
    assert sm["final_exact_depth"] == 42 and sm["best_verified_steps"] == 59
    assert [c[0] for c in calls] == ["verify", "finetune", "verify"]
    print(f"b) {st} >= {sm['vram_limited_ge']}, final exact {sm['final_exact_depth']}")

    st, sm, calls, recs, _ = run([("UNSAFE", dict(fus=43)), ("FULL", dict(min_step=31)),
                                  ("UNSAFE", dict(fus=51)), ("UNSAFE", dict(fus=45))],
                                 max_iterations=4)
    fts = [c for c in calls if c[0] == "finetune"]
    assert st == "MAX_ITERS", st
    assert fts[0][1:] == ("ckpts/base", "0", "iter_001"), fts
    assert fts[1][1:] == ("ckpts/base", "1000", "iter_001"), fts    # reject -> incumbent
    assert recs[1]["censor_decision"] == "REJECT" and recs[1]["censored_depth"] == 30
    assert recs[2]["candidate_accepted"] is True and recs[2]["incumbent"]["depth"] == 50
    assert recs[3]["candidate_accepted"] is False                    # 44 < 50
    assert sm["final_exact_depth"] == 50 and sm["n_rejects"] == 2
    assert [s.get("depth", s.get("depth_lb")) for s in sm["accept_reject"]] == [42, 30, 50, 44]
    print(f"c) {st}; fine-tuned from/seed {[(c[1], c[2]) for c in fts]}")

    st, sm, calls, recs, _ = run([("UNSAFE", dict(fus=99, exact=False, min_step=80))])
    assert st == "VRAM_LIMITED_AT_BASE" and sm["vram_limited_ge"] == 79
    e = sm["accept_reject"][0]
    assert e["kind"] == "nonexact_unsafe" and e["depth_ub"] == 98 and e["depth_lb"] == 79
    print(f"d) non-exact UNSAFE -> {st} >= {sm['vram_limited_ge']} (ub {e['depth_ub']})")

    # e) iterations 1-2 done, then the process stopped: resume runs 3-4 only
    _, _, _, _, run_dir = run([("UNSAFE", dict(fus=43)), ("UNSAFE", dict(fus=50))],
                              max_iterations=2)
    os.remove(os.path.join(run_dir, "summary.json"))
    rec2 = os.path.join(run_dir, "iter_002", "iter_002.json")
    r = json.load(open(rec2))
    r.update(terminal=None, ft_verdict="OK")       # as if iteration 2 had fine-tuned
    json.dump(r, open(rec2, "w"))
    os.makedirs(os.path.join(run_dir, "iter_002", "candidate"), exist_ok=True)
    st, sm, calls, recs, _ = run([("UNSAFE", dict(fus=60)), ("UNSAFE", dict(fus=55))],
                                 max_iterations=4, run_dir=run_dir)
    assert calls[0] == ("verify", os.path.join(run_dir, "iter_002", "candidate")), calls
    assert [s["depth"] for s in sm["accept_reject"]] == [42, 49, 59, 54] and st == "MAX_ITERS"
    print(f"e) resume: {[s['depth'] for s in sm['accept_reject']]} {st}")
    print("PASS test_vram_censor_loop")


if __name__ == "__main__":
    main()
