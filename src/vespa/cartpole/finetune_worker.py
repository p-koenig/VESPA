"""Fine-tuning step of the loop (run as a subprocess by the loop).

    python -m vespa.cartpole.finetune_worker --checkpoint CKPT --boxes NPZ
        --out-checkpoint DIR --out STATUS_DIR --seed S --settings CONFIG

Walks the violating boxes in NPZ back to counterexample regions
(``backward_propagate``), fine-tunes CKPT on them (``finetune``), saves the new
policy to DIR and writes ``status_finetune.json`` to STATUS_DIR with the verdict
(OK, BACKWARD_PROPAGATION_FAILED or FAILED) and the fine-tuning metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback

import numpy as np

STATUS_FILE = "status_finetune.json"


def run(checkpoint, boxes, out_checkpoint, out, seed, settings) -> int:
    os.makedirs(out, exist_ok=True)
    status_path = os.path.join(out, STATUS_FILE)

    def write(payload):
        with open(status_path, "w") as f:
            json.dump(payload, f, indent=2)

    from vespa.cartpole.backward_propagate import BackwardPropagationFailed, backward_propagate
    from vespa.cartpole.env_specs import make_spec

    spec = make_spec(settings["x_threshold"], settings["theta_threshold"])
    try:
        with np.load(boxes) as z:
            capture = {k: z[k] for k in z.files}
        regions = backward_propagate(capture, spec, n_lookback=settings["lookback"],
                                     max_boxes=settings["backprop_max_boxes"], seed=seed)
    except (OSError, BackwardPropagationFailed) as e:
        write({"verdict": "BACKWARD_PROPAGATION_FAILED", "error": str(e)})
        print(f"[finetune-worker] BACKWARD_PROPAGATION_FAILED: {e}", flush=True)
        return 1
    steps = regions["regions_step"]
    print(f"[finetune-worker] {regions['n_unsafe_rows']:,} violating boxes -> "
          f"{steps.shape[0]} counterexample regions, start steps "
          f"{int(steps.min())}..{int(steps.max())}", flush=True)

    from vespa.cartpole.finetune import finetune_policy
    cfg = dict(settings["finetune"], mix_p=settings["mix_p"], horizon=settings["horizon"],
               x_threshold=settings["x_threshold"], theta_threshold=settings["theta_threshold"])
    try:
        metrics = finetune_policy(checkpoint, regions, cfg, out_checkpoint, seed)
    except Exception as e:
        write({"verdict": "FAILED", "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"})
        print(f"[finetune-worker] FAILED: {e}", flush=True)
        return 1
    write({
        "verdict": "OK",
        "checkpoint": os.path.abspath(out_checkpoint),
        "seed": int(seed),
        "n_unsafe_rows": regions["n_unsafe_rows"],
        "n_regions": int(steps.shape[0]),
        "n_failed_rows": regions["n_failed_rows"],
        "unsafe_step": regions["unsafe_step"],
        "region_step_min": int(steps.min()),
        "region_step_max": int(steps.max()),
        "finetune": metrics,
    })
    print(f"[finetune-worker] OK -> {out_checkpoint}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="policy to fine-tune")
    p.add_argument("--boxes", required=True, help="violating_boxes.npz of a verification")
    p.add_argument("--out-checkpoint", required=True)
    p.add_argument("--out", required=True, help="directory of status_finetune.json")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--settings", required=True, help="config.json of the run")
    a = p.parse_args(argv)
    with open(a.settings) as f:
        settings = json.load(f)["settings"]
    return run(a.checkpoint, a.boxes, a.out_checkpoint, a.out, a.seed, settings)


if __name__ == "__main__":
    raise SystemExit(main())
