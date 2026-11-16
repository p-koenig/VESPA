"""The VESPA loop for CartPole: verify, fine-tune on counterexamples, repeat.

Iteration i verifies the current policy (``verify_worker``). Its verified depth is
the number of steps without violation:
  * UNSAFE at first violating step u: depth u - 1 (exact);
  * the box pool filled the GPU (VRAM_POOL_FULL), or a violation was found but the
    pool filled before every earlier box was checked: depth >= m - 1, with m
    the lowest step left in the pool (a lower bound, "censored");
  * SAFE: the whole horizon.

Accept/reject (incumbent rule). The incumbent is the accepted policy with the
largest verified depth. A policy verified UNSAFE with depth below the
incumbent's is rejected; then the incumbent is fine-tuned again on the
counterexamples of its own verification with a new seed
(seed + 1000 * number of rejections). Otherwise the policy becomes the incumbent
and is fine-tuned on its own counterexamples (``finetune_worker``).

The loop ends with status
  SAFE                  a policy is verified safe for the horizon;
  MAX_ITERS             after the last iteration (it is not fine-tuned);
  VRAM_LIMITED          censored depth above the incumbent's: the censored policy
                        is the incumbent;
  VRAM_LIMITED_AT_BASE  censored at iteration 1;
  VRAM_EXHAUSTED        no depth (boxes lost or an empty pool);
  INCOMPLETE_COVERAGE   the verifier crashed;
  BACKWARD_PROPAGATION_FAILED / FINETUNE_FAILED.
A censored depth at or below the incumbent's is rejected like an UNSAFE one.

Run directory:
  config.json                 policy, method, seed and settings of the run
  console.log                 the loop's output including both workers
  summary.json                status and per-iteration outcomes (loop_result)
  iter_NNN/iter_NNN.json      record of iteration NNN
  iter_NNN/status_0-1_of_1.json, pool_0-1_of_1.log, violating_boxes.npz
                              (verify_worker), status_finetune.json, candidate/
A run whose process was stopped resumes after its last complete iteration.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

from vespa.cartpole.verify_worker import BOXES_FILE, STATUS_FILE

CENSOR_TERMINAL = {"AT_BASE": "VRAM_LIMITED_AT_BASE", "ACCEPT_STOP": "VRAM_LIMITED",
                   "NO_DEPTH": "VRAM_EXHAUSTED"}

# Environment of the worker processes (GPU memory: JAX may use 92 % of the GPU
# and allocates on demand, so nvidia-smi reports what is really in use).
WORKER_ENV = {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.92",
              "XLA_PYTHON_CLIENT_ALLOCATOR": "platform"}


def censor_decision(incumbent, depth_lb):
    """What to do with a censored verification (lower bound ``depth_lb``)."""
    if depth_lb is None:
        return "NO_DEPTH"
    if incumbent is None:
        return "AT_BASE"
    if int(depth_lb) > int(incumbent["depth"]):
        return "ACCEPT_STOP"
    return "REJECT"


class Loop:
    def __init__(self, run_dir, checkpoint, settings, seed):
        self.run_dir = os.path.abspath(run_dir)
        self.checkpoint = checkpoint
        self.settings = settings
        self.seed = int(seed)
        self.config_path = os.path.join(self.run_dir, "config.json")
        self.log_file = open(os.path.join(self.run_dir, "console.log"), "a", buffering=1)
        self.history: list[dict] = []
        self.incumbent = None
        self.n_rejects = 0

    # -- output --------------------------------------------------------------
    def log(self, msg, prefix="[loop] "):
        line = prefix + msg
        print(line, flush=True)
        self.log_file.write(line + "\n")

    def iter_dir(self, it):
        return os.path.join(self.run_dir, f"iter_{it:03d}")

    def run_worker(self, module, args):
        """Run a worker module, teeing its output to the console and console.log."""
        import vespa
        src = os.path.dirname(os.path.dirname(os.path.abspath(vespa.__file__)))
        env = dict(os.environ)
        env.update({k: env.get(k, v) for k, v in WORKER_ENV.items()})
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [sys.executable, "-u", "-m", module, *args, "--settings", self.config_path]
        with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as p:
            for line in p.stdout:
                sys.stdout.write(line)
                self.log_file.write(line)
            return p.wait()

    # -- one iteration -------------------------------------------------------
    def verify(self, it, ckpt):
        """Run the verifier. Returns (outcome, record) with outcome in UNSAFE,
        SAFE, VRAM_CENSORED, VRAM_EXHAUSTED, CRASH."""
        d = self.iter_dir(it)
        t0 = time.time()
        self.run_worker("vespa.cartpole.verify_worker", ["--checkpoint", ckpt, "--out", d])
        verify_sec = time.time() - t0
        try:
            with open(os.path.join(d, STATUS_FILE)) as f:
                st = json.load(f)
        except (OSError, ValueError):
            st = {"status": "CRASH"}
        per_step = st.get("boxes_per_step") or []
        rec = {"iter": it, "checkpoint": ckpt, "verify_status": st["status"],
               "verify_sec": round(verify_sec, 1), "n_segments": st.get("n_segments"),
               "first_unsafe_step": st.get("first_unsafe_step"),
               "first_unsafe_exact": bool(st.get("first_unsafe_exact", False)),
               "pool_min_step": st.get("pool_min_step"),
               "box": {"active_total_per_step": per_step,
                       "peak_active_total": int(max(per_step, default=0)),
                       "peak_active_total_step": int(np.argmax(per_step)) if per_step else 0}}
        status = st["status"]
        if status == "UNSAFE" and rec["first_unsafe_exact"]:
            outcome = "UNSAFE"
        elif status == "UNSAFE":
            outcome = "VRAM_CENSORED"
            rec["censor_info"] = {"kind": "nonexact_unsafe",
                                  "upper_bound_depth": int(st["first_unsafe_step"]) - 1}
        elif status == "VRAM_POOL_FULL":
            outcome = "VRAM_CENSORED"
            rec["censor_info"] = {"kind": "vram_pool_full", "upper_bound_depth": None}
        elif status == "SAFE":
            outcome = "SAFE"
        elif status in ("SOUNDNESS_LOST", "VRAM_EXHAUSTED"):
            outcome = "VRAM_EXHAUSTED"
        else:
            outcome = "CRASH"
        self.log(f"iter {it} verify outcome={outcome} (verifier status {status}, "
                 f"{verify_sec:.1f}s)")
        return outcome, rec

    def finetune(self, it, ckpt, boxes_iter, seed):
        """Fine-tune ``ckpt`` on the violating boxes of iteration ``boxes_iter``.
        Returns (verdict, candidate checkpoint or None, metrics)."""
        d = self.iter_dir(it)
        cand = os.path.join(d, "candidate")
        self.run_worker("vespa.cartpole.finetune_worker", [
            "--checkpoint", ckpt, "--boxes", os.path.join(self.iter_dir(boxes_iter), BOXES_FILE),
            "--out-checkpoint", cand, "--out", d, "--seed", str(seed)])
        try:
            with open(os.path.join(d, "status_finetune.json")) as f:
                st = json.load(f)
        except (OSError, ValueError):
            return "FAILED", None, {}
        if st.get("verdict") != "OK":
            return st.get("verdict", "FAILED"), None, {}
        return "OK", cand, st.get("finetune", {})

    def commit(self, rec):
        self.history.append(rec)
        with open(os.path.join(self.iter_dir(rec["iter"]), f"iter_{rec['iter']:03d}.json"), "w") as f:
            json.dump(rec, f, indent=2)

    # -- resume ----------------------------------------------------------------
    def resume_point(self):
        """(first iteration, its checkpoint, terminal status or None) from the
        iteration records of an earlier, stopped process."""
        recs = []
        for p in sorted(glob.glob(os.path.join(self.run_dir, "iter_*", "iter_*.json"))):
            with open(p) as f:
                rec = json.load(f)
            if rec.get("terminal") is None and not os.path.isdir(
                    os.path.join(self.iter_dir(rec["iter"]), "candidate")):
                break
            recs.append(rec)
            if rec.get("terminal") is not None:
                break
        if not recs:
            return 1, self.checkpoint, None
        self.history = recs
        for rec in recs:
            self.incumbent = rec.get("incumbent") or self.incumbent
            self.n_rejects += int(bool(rec.get("rejected")))
        last = recs[-1]
        if last.get("terminal") is not None:
            return last["iter"], self.checkpoint, last["terminal"]
        self.log(f"RESUME after iter {last['iter']} ({len(recs)} iteration record(s))")
        return last["iter"] + 1, os.path.join(self.iter_dir(last["iter"]), "candidate"), None

    # -- loop ------------------------------------------------------------------
    def run(self):
        max_iters = int(self.settings["max_iterations"])
        it, ckpt, status = self.resume_point()
        if status is not None:
            self.log(f"run already concluded with status {status}")
            return status
        t0 = time.time()
        status = "MAX_ITERS"
        for it in range(it, max_iters + 1):
            self.log(f"=== iter {it}/{max_iters} (ckpt={ckpt}) ===")
            os.makedirs(self.iter_dir(it), exist_ok=True)
            outcome, rec = self.verify(it, ckpt)
            rec["outcome"] = outcome
            terminal = None
            ft_from, boxes_iter, ft_seed = ckpt, it, self.seed
            accepted, reject = True, False

            if outcome == "UNSAFE":
                depth = int(rec["first_unsafe_step"]) - 1
                rec["verified_depth"] = depth
                inc = self.incumbent
                if inc is not None and depth < int(inc["depth"]):
                    accepted, reject = False, True
                else:
                    self.incumbent = {"ckpt": ckpt, "depth": depth, "iter": it}
                    self.log(f"iter {it} ACCEPT (verified {depth}); incumbent={ckpt}")
            elif outcome == "VRAM_CENSORED":
                pm = rec["pool_min_step"]
                depth_lb = None if pm is None else int(pm) - 1
                dec = censor_decision(self.incumbent, depth_lb)
                rec.update(censored_depth=depth_lb, censor_decision=dec)
                self.log(f"iter {it} CENSORED: safe through >= {depth_lb} "
                         f"({rec['censor_info']['kind']}, upper bound "
                         f"{rec['censor_info']['upper_bound_depth']}) -> {dec}")
                # only a censored depth above the incumbent's is an accepted policy
                accepted, reject = dec == "ACCEPT_STOP", dec == "REJECT"
                if not reject:
                    if dec == "ACCEPT_STOP":
                        self.incumbent = {"ckpt": ckpt, "depth": depth_lb, "iter": it,
                                          "censored": True}
                    terminal = CENSOR_TERMINAL[dec]
            elif outcome == "SAFE":
                terminal = "SAFE"
            elif outcome == "VRAM_EXHAUSTED":
                terminal = "VRAM_EXHAUSTED"
            else:
                terminal = "INCOMPLETE_COVERAGE"

            if reject:
                inc = self.incumbent
                self.n_rejects += 1
                ft_from, boxes_iter = inc["ckpt"], int(inc["iter"])
                ft_seed = self.seed + 1000 * self.n_rejects
                self.log(f"iter {it} REJECT candidate (verified "
                         f"{rec.get('verified_depth', rec.get('censored_depth'))} < incumbent "
                         f"{inc['depth']} @iter {inc['iter']}); re-fine-tune incumbent with "
                         f"seed {ft_seed}")
            rec.update(candidate_accepted=accepted, rejected=reject,
                       incumbent=dict(self.incumbent or {}))

            if terminal is None:
                if it >= max_iters:
                    rec["ft_verdict"] = "SKIPPED_LAST"
                    terminal = "MAX_ITERS"
                else:
                    t_ft = time.time()
                    verdict, cand, metrics = self.finetune(it, ft_from, boxes_iter, ft_seed)
                    rec.update(ft_verdict=verdict, ft_from=ft_from, ft_seed=ft_seed,
                               finetune_sec=round(time.time() - t_ft, 1), val=metrics)
                    if verdict == "OK":
                        ckpt = cand
                    elif verdict == "BACKWARD_PROPAGATION_FAILED":
                        terminal = verdict
                    else:
                        terminal = "FINETUNE_FAILED"
            rec["terminal"] = terminal
            self.commit(rec)
            if terminal is not None:
                status = terminal
                break

        elapsed = time.time() - t0
        self.log(f"TERMINAL status={status} iters={it} elapsed={elapsed:.1f}s final_ckpt={ckpt}")
        self.write_summary(status, it, elapsed, ckpt)
        return status

    def write_summary(self, status, iterations, elapsed, final_ckpt):
        """summary.json: ``loop_result.accept_reject`` has one entry per
        verification with its outcome and verified depth (exact) or depth
        bounds (censored)."""
        horizon = int(self.settings["horizon"])
        seq, depths = [], []
        for h in self.history:
            o = h["outcome"]
            if o == "SAFE":
                seq.append({"iter": h["iter"], "outcome": o, "depth": horizon, "exact": True,
                            "accepted": True})
                depths.append(horizon)
            elif o == "VRAM_CENSORED":
                ci = h["censor_info"]
                seq.append({"iter": h["iter"], "outcome": o,
                            "depth_lb": h["censored_depth"], "depth_ub": ci["upper_bound_depth"],
                            "kind": ci["kind"], "exact": False,
                            "decision": h["censor_decision"],
                            "accepted": h["candidate_accepted"]})
                if h["censored_depth"] is not None:
                    depths.append(int(h["censored_depth"]))
            elif o == "UNSAFE":
                seq.append({"iter": h["iter"], "outcome": o, "depth": h["verified_depth"],
                            "exact": True, "accepted": h["candidate_accepted"]})
                depths.append(int(h["verified_depth"]))
            else:
                seq.append({"iter": h["iter"], "outcome": o})
        censored = [h for h in self.history if h["outcome"] == "VRAM_CENSORED"
                    and h["censor_decision"] != "REJECT"]
        summary = {
            "status": status,
            "iterations": iterations,
            "elapsed_sec": round(elapsed, 1),
            "checkpoint": self.checkpoint,
            "final_ckpt": final_ckpt,
            "loop_result": {
                "status": status,
                "best_verified_steps": max(depths) if depths else None,
                "verified_depth_per_iter": depths,
                "incumbent": self.incumbent,
                "n_rejects": self.n_rejects,
                "censored_depth": censored[-1]["censored_depth"] if censored else None,
                "accept_reject": seq,
                "start": seq[0] if seq else None,
                "final_exact_depth": max([s["depth"] for s in seq if s.get("exact")
                                          and s.get("accepted")] or [None],
                                         key=lambda x: -1 if x is None else x),
                "vram_limited_ge": (censored[-1]["censored_depth"]
                                    if status.startswith("VRAM_LIMITED") else None),
            },
        }
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


def run_loop(run_dir, checkpoint, settings, seed) -> str:
    """Run (or resume) the loop in ``run_dir``; returns the terminal status."""
    os.makedirs(run_dir, exist_ok=True)
    return Loop(run_dir, checkpoint, settings, seed).run()
