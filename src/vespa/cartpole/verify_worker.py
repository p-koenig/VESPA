"""Verification of one policy checkpoint (run as a subprocess by the loop).

    python -m vespa.cartpole.verify_worker --checkpoint CKPT --out DIR --settings CONFIG

Verifies the decision tree policy in CKPT over the horizon (``segment_pool``) and
writes to DIR:
  * ``status_0-1_of_1.json``: the outcome (status, first violating step and
    whether it is exact, lowest pool step at exit, boxes per step, segments);
  * ``violating_boxes.npz``: the violating boxes of the first violating step
    (only if the policy is unsafe);
  * ``pool_0-1_of_1.log``: the log of the box pool.
CONFIG is the run's ``config.json``; its ``settings`` block holds the horizon,
the safety box, the kernel batch cap, the action-history length and the pool
memory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback

STATUS_FILE = "status_0-1_of_1.json"
POOL_LOG_FILE = "pool_0-1_of_1.log"
BOXES_FILE = "violating_boxes.npz"


def _write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def run(checkpoint: str, out: str, settings: dict) -> int:
    """Verify ``checkpoint`` with ``settings``; returns 0 unless the verifier crashed."""
    os.makedirs(out, exist_ok=True)
    status_path = os.path.join(out, STATUS_FILE)
    t0 = time.time()
    log_file = open(os.path.join(out, POOL_LOG_FILE), "a", buffering=1)

    def log(msg):
        log_file.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        print(msg, flush=True)

    try:
        from vespa.cartpole.env_specs import make_spec
        from vespa.cartpole.policy import leaf_regions
        from vespa.cartpole.segment_pool import save_violating_boxes, verify_segmented

        spec = make_spec(settings["x_threshold"], settings["theta_threshold"],
                         settings["start_halfwidth"])
        regions = leaf_regions(checkpoint)
        cap = int(settings["kernel_cap"])
        horizon = int(settings["horizon"])
        lookback = int(settings["action_history"])
        print(f"[verify-worker] leaves={len(regions)} horizon={horizon} cap={cap:,} "
              f"lookback={lookback}", flush=True)
        res = verify_segmented(regions, spec, horizon=horizon, cap=cap, lookback=lookback,
                               box_memory_gib=settings["box_memory_gib"], log=log)
        status = "SAFE" if res["status"] == "OK" else res["status"]
        if status == "UNSAFE" and res["capture"] is not None:
            save_violating_boxes(os.path.join(out, BOXES_FILE), res["capture"],
                                 spec.init_lo, spec.init_hi)
        payload = {k: v for k, v in res.items() if k != "capture"}
        payload.update(status=status, n_leaves=len(regions),
                       wall_clock_sec=round(time.time() - t0, 2))
        _write_json(status_path, payload)
        print(f"[verify-worker] status={status} first_unsafe={res['first_unsafe_step']} "
              f"exact={res['first_unsafe_exact']} crossings={res['n_crossings']} "
              f"unsafe_boxes={res['unsafe_boxes_total']:,}->{res['unsafe_boxes_final']:,} "
              f"peak_pool={res['peak_pool_boxes']:,} segments={res['n_segments']} "
              f"(post-crossing {res['post_crossing_segments']})", flush=True)
        return 1 if status == "CRASH" else 0
    except Exception as e:
        _write_json(status_path, {"status": "CRASH",
                                  "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                                  "wall_clock_sec": round(time.time() - t0, 2)})
        print(f"[verify-worker] CRASH: {e}", flush=True)
        return 1
    finally:
        log_file.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--settings", required=True, help="config.json of the run")
    a = p.parse_args(argv)
    with open(a.settings) as f:
        settings = json.load(f)["settings"]
    return run(a.checkpoint, a.out, settings)


if __name__ == "__main__":
    raise SystemExit(main())
