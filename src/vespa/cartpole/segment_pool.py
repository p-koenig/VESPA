"""Box pool of the CartPole verifier: breadth-first verification over the horizon.

The reachable boxes wait in a pool in GPU memory, keyed by their absolute step.
The pool always dispatches the packs of the lowest step first (breadth-first), so
the first violation it finds is the exact first violating step.

Verification of a policy:
  1. The start set, cut along the leaf partition, is the first pack (step 0).
  2. A worker thread runs a pack with the segment kernel (``interval_core``)
     until the horizon, a violation, an empty set, or until the next step would
     exceed the kernel batch cap. In the last case the survivors are split into
     ``PARTITION_FACTOR`` pieces that go back to the pool at their step.
  3. After the first violation at step u, every pack that starts below u still
     runs, so every violation at the earliest step is found and its violating
     boxes are captured for the counterexample regions. Packs at steps >= u are
     dropped.
  4. When the boxes in GPU memory exceed the pool budget, no further pack is
     dispatched. The verification then ends with status VRAM_POOL_FULL: every box
     below the lowest step left in the pool has been checked, which gives a lower
     bound on the verified safe steps.

Packs of the same step are concatenated up to half the cap per dispatch, so the
reduction of the kernel also removes duplicates between packs. Boxes are never
dropped except at the horizon (verified) and after a violation as described.
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time

import numpy as np

# Import the kernel first: it enables JAX float64.
from vespa.cartpole import interval_core
from vespa.cartpole.device_pool import is_dev, meta_nbytes, parent_device
from vespa.cartpole.env_specs import STATE_DIM

import jax  # noqa: E402

# Pool settings of the paper runs.
PARTITION_FACTOR = 4           # a pack that hits the cap is split into 4 packs
DISPATCH_COALESCE_FRAC = 0.5   # same-step packs are concatenated up to cap / 2
CAPTURE_MAX_BOXES = 4_000_000  # captured violating boxes kept (random subsample above)
IN_FLIGHT = 2                  # packs queued for the single kernel worker
OUTPUT_SLOTS = 2               # cap-sized kernel outputs reserved in the pool budget
HEARTBEAT_SEC = 15.0           # interval of the progress line and the memory check


# --------------------------------------------------------------------------- #
# Device memory                                                               #
# --------------------------------------------------------------------------- #

def _gpu_mem_info(devices):
    """{device.id: (total_bytes, used_bytes)} from nvidia-smi, {} without it.
    JAX's local device ids are mapped through CUDA_VISIBLE_DEVICES."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:  # nvidia-smi missing or failing: no information
        return {}
    rows = {}
    for ln in out.splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) == 4 and f[2].isdigit() and f[3].isdigit():
            rows[f[0]] = rows[f[1]] = (int(f[2]) * 2**20, int(f[3]) * 2**20)
    cvd = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    res = {}
    for d in devices:
        key = cvd[d.id] if d.id < len(cvd) else str(d.id)
        r = rows.get(key)
        if r is None:
            r = next((v for k, v in rows.items() if k.startswith(key)), None)
        if r is not None:
            res[d.id] = r
    return res


def _pool_budgets(devices, box_memory_gib, *, cap, lookback, log):
    """Pool budget per device in bytes, and {device.id: (total, margin)} for the
    devices whose memory size is known.

    ``box_memory_gib`` "auto": everything a GPU has beside the kernel, i.e.
    total - used now - predicted kernel peak - margin, with margin = max(2 GiB,
    3 % of the GPU). A number: that many GiB in total, split over the devices.
    """
    def margin(total):
        return max(2 * 2**30, 0.03 * total)

    info = _gpu_mem_info(devices)
    if str(box_memory_gib).strip().lower() != "auto":
        b = int(float(box_memory_gib) * 2**30 / max(len(devices), 1))
        return {d.id: b for d in devices}, {k: (t, margin(t)) for k, (t, _u) in info.items()}
    kernel_bytes = interval_core.kernel_memory_bytes(cap, lookback)
    out, sizes = {}, {}
    for d in devices:
        if d.id not in info:
            raise ValueError(f"box memory 'auto': no memory information for {d} "
                             f"(nvidia-smi); give the pool memory in GiB")
        total, used = info[d.id]
        out[d.id] = int(total - used - kernel_bytes - margin(total))
        sizes[d.id] = (total, margin(total))
        log(f"[pool] memory auto {d}: total {total / 2**30:.1f} - used {used / 2**30:.1f} "
            f"- kernel {kernel_bytes / 2**30:.1f} (cap {cap:,}, lookback {lookback}) "
            f"- margin {margin(total) / 2**30:.1f} = pool budget {out[d.id] / 2**30:.1f}GiB")
    return out, sizes


# --------------------------------------------------------------------------- #
# Packs and the kernel worker                                                 #
# --------------------------------------------------------------------------- #

class _PoolItem:
    """One pack in the pool: a host pack (the start set) or a DevPiece."""
    __slots__ = ("step", "n", "ram", "dev", "dev_nbytes", "no_coalesce", "dev_pid")

    def __init__(self, step, n, ram=None):
        self.step = int(step)
        self.n = int(n)
        self.ram = ram
        self.dev = None
        self.dev_nbytes = 0
        # Pieces of a split pack are dispatched alone (a merged pack could hit
        # the cap again at the same step).
        self.no_coalesce = False
        self.dev_pid = None


def _split_chunk(piece, n_parts):
    """Split a DevPiece into up to ``n_parts`` row ranges of equal size."""
    n = int(piece["n"])
    if n_parts <= 1 or n <= 1:
        return [piece]
    sz = (n + n_parts - 1) // n_parts
    return [piece.slice(a, min(a + sz, n)) for a in range(0, n, sz) if min(a + sz, n) > a]


class _Worker(threading.Thread):
    """Kernel worker: compiles the segment kernel once on its device, then runs
    the packs of ``pack_q`` and puts the results on ``result_q``."""

    def __init__(self, wid, device, kernel_kwargs, pack_q, result_q, stop_evt, log):
        super().__init__(daemon=True)
        self.wid = wid
        self.device = device
        self.kernel_kwargs = kernel_kwargs
        self.pack_q = pack_q
        self.result_q = result_q
        self.stop_evt = stop_evt
        self.log = log
        self.error = None

    def run(self):
        try:
            with jax.default_device(self.device):
                t0 = time.time()
                seg_fn = interval_core.build_segment_kernel(**self.kernel_kwargs,
                                                            device=self.device)
                self.log(f"[worker {self.wid}] kernel built on {self.device} "
                         f"(cap={seg_fn.cap:,} K={seg_fn.K}) in {time.time() - t0:.1f}s")
                while not self.stop_evt.is_set():
                    try:
                        item = self.pack_q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if item is None:
                        self.pack_q.task_done()
                        break
                    pack, start_step, pack_id = item
                    item = None
                    t0 = time.time()
                    if not (isinstance(pack, dict) and "parts" in pack):
                        pack = {"parts": [pack], "n": int(pack["n"])}
                    res = seg_fn(pack, start_step)
                    pack = None
                    res["pack_id"] = pack_id
                    res["worker"] = self.wid
                    res["seg_elapsed_sec"] = time.time() - t0
                    self.result_q.put(res)
                    res = None
                    self.pack_q.task_done()
        except Exception as e:  # reported to the pool as a crash
            import traceback
            self.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            self.log(f"[worker {self.wid}] CRASH: {e}")
            self.result_q.put({"_crash": self.error, "worker": self.wid})


def _devices():
    devs = [d for d in jax.local_devices() if d.platform == "gpu"]
    return devs or jax.local_devices()


# --------------------------------------------------------------------------- #
# Verification                                                                #
# --------------------------------------------------------------------------- #

def verify_segmented(leaf_regions, spec, *, horizon, cap, lookback, box_memory_gib, log=print):
    """Verify ``leaf_regions`` (the policy) from ``spec``'s start set over ``horizon``.

    ``cap``: kernel batch cap; ``lookback``: columns of the action history kept
    per box; ``box_memory_gib``: pool budget ("auto" or GiB).

    Returns a dict with ``status`` in {OK (safe for the whole horizon), UNSAFE,
    VRAM_POOL_FULL, SOUNDNESS_LOST, VRAM_EXHAUSTED, CRASH}, ``first_unsafe_step``,
    ``first_unsafe_exact``, ``capture`` (the violating boxes, on UNSAFE), the
    lowest pool step at exit, per-step box counts and segment records.
    """
    cap, horizon = int(cap), int(horizon)
    devices = _devices()[:1]
    K = len(leaf_regions)
    log(f"[pool] G=1 worker(s) {[str(d) for d in devices]}; K={K} leaves; cap={cap:,} "
        f"partition_factor={PARTITION_FACTOR} horizon={horizon} lookback={lookback}")

    pack_q: "queue.Queue" = queue.Queue()
    result_q: "queue.Queue" = queue.Queue()
    stop_evt = threading.Event()

    # Pool budget per device, counted on the physical device buffers. A cap-sized
    # output is reserved twice on top of the pool content (OUTPUT_SLOTS).
    ncol = max(int(lookback), 1)
    bytes_per_box = 8 * STATE_DIM + meta_nbytes(K, ncol, 256)
    dev_out_max = int(cap) * int(bytes_per_box)
    dev_budget_d, dev_size_d = _pool_budgets(devices, box_memory_gib, cap=cap,
                                             lookback=lookback, log=log)
    dev_target_d = {}
    for dev_id, budget in dev_budget_d.items():
        dev_target_d[dev_id] = max(0, budget - OUTPUT_SLOTS * dev_out_max)
        if dev_target_d[dev_id] < dev_out_max:
            log(f"[pool] pool target on device {dev_id} {dev_target_d[dev_id] / 2**30:.1f}GiB "
                f"< one cap output {dev_out_max / 2**30:.1f}GiB")
    dev_budget = sum(dev_budget_d.values())
    dev_target = sum(dev_target_d.values())
    log(f"[pool] budget {dev_budget / 2**30:.1f}GiB, output reserve {OUTPUT_SLOTS} x "
        f"{dev_out_max / 2**30:.1f}GiB -> pool target "
        + " ".join(f"{d}:{t / 2**30:.1f}GiB" for d, t in dev_target_d.items())
        + f" (~{dev_target / max(bytes_per_box, 1) / 1e9:.2f}G boxes at {bytes_per_box} B/box)")

    kernel_kwargs = dict(leaf_regions=leaf_regions, spec=spec, cap=cap, horizon=horizon,
                         lookback=lookback)
    worker = _Worker(0, devices[0], kernel_kwargs, pack_q, result_q, stop_evt, log)
    worker.start()

    # --- pool state ---------------------------------------------------------
    pool: dict[int, list] = {}          # step -> [_PoolItem]
    pool_counts: dict[int, int] = {}    # step -> boxes
    seed = interval_core.make_init_boxes(leaf_regions, spec, lookback)
    seed["n"] = int(seed["lo"].shape[0])
    pool[0] = [_PoolItem(0, seed["n"], ram=seed)]
    pool_counts[0] = seed["n"]
    log(f"[pool] seeded step 0 with {seed['n']} start box(es)")

    # Device accounting: proportional piece bytes (dev_bytes, statistics) and the
    # physical parent buffers per device (dev_phys, budget). A parent is counted
    # until its last piece has left the pool and its segment has finished.
    dev_bytes = 0
    boxes_dev = 0
    n_packs_dev = 0
    dev_par: dict = {}                  # parent pid -> [device id, bytes, pieces held]
    dev_phys: dict = {d: 0 for d in dev_target_d}
    dev_phys_peak: dict = {d: 0 for d in dev_target_d}
    dev_inflight_pids: dict = {}        # pack id -> [parent pid]
    dev_guard = {"cut": {d: 0 for d in dev_target_d}, "used": {}}
    vram_pool_full = False
    peak = {"dev_bytes": 0, "dev_boxes": 0, "total_boxes": 0}

    # --- results ------------------------------------------------------------
    any_unsafe = False
    unsafe_capture = None
    unsafe_captures: list = []          # captures of every violating segment
    unsafe_boxes_total = 0
    unsafe_boxes_captured = 0
    n_crossings = 0
    post_segments = 0                   # segments dispatched after the first violation
    soundness_lost = False
    vram_exhausted = False
    crash = None
    first_unsafe_step = None
    peak_pool_boxes = sum(pool_counts.values())
    peak_worker_boxes = 0
    packs_dispatched = 0
    segments_completed = 0
    reduce_acc = reduce_prot_acc = None
    seg_records: list[dict] = []
    step_peak = np.zeros(horizon + 1, dtype=np.int64)   # boxes per step (max over packs)
    step_peak[0] = pool_counts.get(0, 0)
    in_flight = 0
    coalesce_target = int(cap * DISPATCH_COALESCE_FRAC)
    coalesce_stats = {"dispatches": 0, "packs_merged": 0, "boxes": 0}
    log(f"[pool] dispatch coalescing: target={coalesce_target:,} boxes")
    wrk_t = {"stage": 0.0, "core": 0.0}

    def _pool_total():
        return sum(pool_counts.values())

    def _select_step():
        return min(pool.keys()) if pool else None

    def _make_pack():
        """Take the first pack of the lowest step: (piece, step, item)."""
        nonlocal dev_bytes, boxes_dev, n_packs_dev
        s = _select_step()
        if s is None:
            return None
        items = pool[s]
        item = items.pop(0)
        pool_counts[s] -= item.n
        if not items:
            del pool[s]
            del pool_counts[s]
        if item.dev is not None:
            piece = item.dev
            dev_bytes -= item.dev_nbytes
            boxes_dev -= item.n
            n_packs_dev -= 1
            item.dev_pid = piece.parent.pid     # counted until its segment finishes
            item.dev = None
            return piece, s, item
        piece = item.ram
        item.ram = None
        return piece, s, item

    def _par_ref(par):
        """A piece of ``par`` enters the pool."""
        e = dev_par.get(par.pid)
        if e is None:
            d = parent_device(par).id
            e = dev_par[par.pid] = [d, int(par.nbytes), 0]
            dev_phys[d] = dev_phys.get(d, 0) + e[1]
            if dev_phys[d] > dev_phys_peak.get(d, 0):
                dev_phys_peak[d] = dev_phys[d]
        e[2] += 1

    def _par_unref(pid):
        """A piece of parent ``pid`` has finished its segment."""
        e = dev_par.get(pid)
        if e is None:
            return
        e[2] -= 1
        if e[2] <= 0:
            dev_phys[e[0]] -= e[1]
            del dev_par[pid]

    def _add_to_pool(piece, step, no_coalesce=False):
        nonlocal dev_bytes, boxes_dev, n_packs_dev
        n = int(piece["n"])
        if n == 0:
            return
        nb = piece.nbytes()
        it = _PoolItem(step, n)
        it.dev = piece
        it.dev_nbytes = nb
        _par_ref(piece.parent)
        it.no_coalesce = bool(no_coalesce)
        pool.setdefault(step, []).append(it)
        pool_counts[step] = pool_counts.get(step, 0) + n
        dev_bytes += nb
        boxes_dev += n
        n_packs_dev += 1

    def _check_pool_budget():
        """Stop dispatching once the pool's device buffers exceed the target."""
        nonlocal vram_pool_full
        for d in dev_target_d:
            if dev_phys.get(d, 0) > dev_target_d[d] and not vram_pool_full:
                vram_pool_full = True
                log(f"[pool] VRAM_POOL_FULL: device {d} {dev_phys[d] / 2**30:.1f}GiB > target "
                    f"{dev_target_d[d] / 2**30:.1f}GiB -> no further dispatch, draining "
                    f"{in_flight} in-flight segment(s)")
                return

    def _memory_guard():
        """Measured check of the memory model: if nvidia-smi reports more than
        total - margin/2 in use, lower that device's pool target by the excess
        plus one cap output."""
        if not dev_size_d:
            return
        info = _gpu_mem_info(devices)
        for d, (total, margin) in dev_size_d.items():
            if d not in info:
                continue
            used = info[d][1]
            dev_guard["used"][d] = used
            ceil_ = total - margin / 2
            if used > ceil_ and d in dev_target_d:
                cut = int(used - ceil_) + dev_out_max
                dev_target_d[d] = max(0, dev_target_d[d] - cut)
                dev_guard["cut"][d] = dev_guard["cut"].get(d, 0) + cut
                log(f"[pool] MEMORY GUARD device {d}: nvidia-smi used {used / 2**30:.1f}GiB > "
                    f"{ceil_ / 2**30:.1f}GiB (pool {dev_phys.get(d, 0) / 2**30:.1f}GiB) -> target "
                    f"lowered by {cut / 2**30:.1f} to {dev_target_d[d] / 2**30:.1f}GiB")

    def _heartbeat():
        _memory_guard()
        log(f"[pool] hb t={time.time() - t_start:.0f}s seg={segments_completed} "
            f"step={_select_step()} packs={n_packs_dev} boxes={boxes_dev:,} "
            f"infl={in_flight} wrk(stage/core)={wrk_t['stage']:.0f}/{wrk_t['core']:.0f}s "
            f"vram={sum(dev_phys.values()) / 2**30:.1f}/{dev_target / 2**30:.0f}GiB"
            + (f" unsafe@{first_unsafe_step} x{n_crossings} cap={unsafe_boxes_captured:,} "
               f"wave={post_segments}" if any_unsafe else ""))
        for k in wrk_t:
            wrk_t[k] = 0.0

    def _wave_dispatchable():
        """After a violation: is there a pack below the earliest violating step?"""
        if not any_unsafe:
            return False
        s_min = _select_step()
        return s_min is not None and s_min < first_unsafe_step

    t_start = time.time()
    last_hb = 0.0
    try:
        while True:
            # 1. Dispatch the lowest-step packs (concatenating same-step packs).
            if ((not any_unsafe) or _wave_dispatchable()) and not vram_pool_full:
                while in_flight < IN_FLIGHT and _pool_total() > 0:
                    if any_unsafe and not _wave_dispatchable():
                        break
                    made = _make_pack()
                    if made is None:
                        break
                    piece, s_disp, it0 = made
                    src_items = [it0]
                    if piece["n"] < coalesce_target and not it0.no_coalesce:
                        parts = [piece]
                        tot = int(piece["n"])
                        while True:
                            items_s = pool.get(s_disp)
                            if not items_s or _select_step() != s_disp or items_s[0].no_coalesce:
                                break
                            if tot + int(items_s[0].n) > coalesce_target:
                                break
                            p2, _s2, it2 = _make_pack()
                            parts.append(p2)
                            src_items.append(it2)
                            tot += int(p2["n"])
                        if len(parts) > 1:
                            piece = {"parts": parts, "n": tot}
                            coalesce_stats["dispatches"] += 1
                            coalesce_stats["packs_merged"] += len(parts) - 1
                            coalesce_stats["boxes"] += tot
                        parts = None
                    pid = packs_dispatched
                    dpids = [it_.dev_pid for it_ in src_items if it_.dev_pid is not None]
                    if dpids:
                        dev_inflight_pids[pid] = dpids
                    for it_ in src_items:
                        it_.dev_nbytes = 0
                        it_.dev_pid = None
                    src_items = None
                    pack_q.put((piece, s_disp, pid))
                    piece = None
                    in_flight += 1
                    packs_dispatched += 1
                    if any_unsafe:
                        post_segments += 1

            # 2. Pool budget and periodic progress line.
            _check_pool_budget()
            now = time.time()
            if now - last_hb >= HEARTBEAT_SEC:
                _heartbeat()
                last_hb = now

            if vram_pool_full and in_flight == 0:
                break
            if in_flight == 0 and (_pool_total() == 0 or (any_unsafe and not _wave_dispatchable())):
                break

            # 3. Retire one finished segment.
            try:
                res = result_q.get(timeout=2.0)
            except queue.Empty:
                continue
            in_flight -= 1
            for p_ in dev_inflight_pids.pop(res.get("pack_id"), ()):
                _par_unref(p_)
            if "_crash" in res:
                crash = res["_crash"]
                break
            wrk_t["stage"] += float(res.get("t_stage_sec", 0.0))
            wrk_t["core"] += float(res.get("t_core_sec", 0.0))
            segments_completed += 1
            seg_records.append({
                "order": segments_completed,
                "worker": int(res.get("worker", -1)),
                "start_step": int(res["start_step"]),
                "end_step": int(res["end_step"]),
                "steps_run": int(res["steps_run"]),
                "seg_elapsed_sec": round(float(res.get("seg_elapsed_sec", 0.0)), 3),
                "hit_cap": bool(res.get("hit_cap", False)),
                "n_out": int(res["n_out"]),
                "peak_boxes": int(res["peak_boxes"]),
            })
            peak_worker_boxes = max(peak_worker_boxes, res["peak_boxes"])
            rs = np.asarray(res["reduce_stats"], dtype=np.int64)
            if len(rs) > 1:
                rp = np.asarray(res["reduce_prot"], dtype=np.float64)
                if reduce_acc is None:
                    reduce_acc = np.zeros((horizon + 1, rs.shape[1]), np.int64)
                    reduce_prot_acc = np.zeros((horizon + 1, rp.shape[1]), np.float64)
                s0 = int(res["start_step"])
                for i in range(1, rs.shape[0]):
                    s = s0 + i
                    if 0 <= s <= horizon:
                        reduce_acc[s] += rs[i]
                        reduce_prot_acc[s] = np.maximum(reduce_prot_acc[s], rp[i])
            if res["soundness_lost"]:
                soundness_lost = True
            s0 = res["start_step"]
            for i, c in enumerate(res["traj"]):
                if 0 <= s0 + i <= horizon:
                    step_peak[s0 + i] = max(step_peak[s0 + i], int(c))

            if res["any_unsafe"]:
                fus = res["first_unsafe_step"]
                n_crossings += 1
                cap_i = res.get("capture")
                n_tot = int(res.get("n_unsafe_boxes", 0))
                n_cap = int(cap_i["lo"].shape[0]) if cap_i is not None else 0
                unsafe_boxes_total += n_tot
                unsafe_boxes_captured += n_cap
                if (not any_unsafe) or fus < first_unsafe_step:
                    prev_fus = first_unsafe_step
                    first_unsafe_step = fus
                    unsafe_capture = cap_i
                    log(f"[pool] UNSAFE @step {fus} (worker {res['worker']}, pack started "
                        f"@{res['start_step']}) unsafe_boxes={n_tot:,} captured={n_cap:,}"
                        + (f" [earlier than {prev_fus}]" if any_unsafe else ""))
                else:
                    log(f"[pool] crossing @step {fus} (worker {res['worker']}, pack started "
                        f"@{res['start_step']}) unsafe_boxes={n_tot:,} captured={n_cap:,}")
                if cap_i is not None and n_cap > 0:
                    unsafe_captures.append(cap_i)
                any_unsafe = True
                continue

            end = res["end_step"]
            if any_unsafe and not end < first_unsafe_step:
                continue            # these boxes can only violate later
            if end >= horizon:
                pass                # verified for the whole horizon
            elif res["n_out"] == 0:
                # Every step maps a non-empty box to a non-empty box: an empty set
                # before the horizon means boxes were lost.
                soundness_lost = True
                if res["steps_run"] == 0:
                    vram_exhausted = True
                log(f"[pool] EMPTY reach set @step {end} (pack @{res['start_step']}, "
                    f"steps_run={res['steps_run']}) -> soundness lost")
            elif res["hit_cap"]:
                bo = res["boxes_out"]
                if bo["n"] > 0:
                    for part in _split_chunk(bo, PARTITION_FACTOR):
                        _add_to_pool(part, end, no_coalesce=True)
            else:
                bo = res["boxes_out"]
                if bo["n"] > 0:
                    _add_to_pool(bo, end)
            res = None

            total = _pool_total()
            peak_pool_boxes = max(peak_pool_boxes, total)
            for key, val in (("dev_bytes", dev_bytes), ("dev_boxes", boxes_dev),
                             ("total_boxes", total)):
                if val > peak[key]:
                    peak[key] = int(val)
    finally:
        stop_evt.set()
        pack_q.put(None)
        worker.join(timeout=10.0)

    if worker.error and crash is None:
        crash = worker.error
    elapsed = time.time() - t_start

    if any_unsafe and unsafe_captures:
        unsafe_capture = _merge_captures(unsafe_captures, unsafe_capture, CAPTURE_MAX_BOXES, 0, log)
    # A full pool can stop the post-violation wave: then the first violating
    # step is only an upper bound.
    post_crossing_cut = bool(any_unsafe and vram_pool_full)
    first_unsafe_exact = bool(any_unsafe and not post_crossing_cut)
    if any_unsafe:
        log(f"[pool] crossings={n_crossings} unsafe_boxes_total={unsafe_boxes_total:,} "
            f"captured={unsafe_boxes_captured:,} post_crossing_segments={post_segments} "
            f"first_unsafe_exact={first_unsafe_exact}")

    if crash is not None:
        status = "CRASH"
    elif any_unsafe:
        status = "UNSAFE"
    elif vram_pool_full:
        status = "VRAM_POOL_FULL"
    elif vram_exhausted:
        status = "VRAM_EXHAUSTED"
    elif soundness_lost:
        status = "SOUNDNESS_LOST"
    else:
        status = "OK"

    pool_min_step = min(pool) if pool else None
    log(f"[pool] TIERS at exit: vram {boxes_dev:,} boxes {dev_bytes / 2**30:.2f}GiB | "
        f"min_step {pool_min_step} ; peaks: vram {peak['dev_boxes']:,}/"
        f"{peak['dev_bytes'] / 2**30:.2f}GiB")
    log(f"[pool] DONE status={status} first_unsafe={first_unsafe_step} "
        f"segments={segments_completed} packs={packs_dispatched} "
        f"peak_pool_boxes={peak_pool_boxes:,} peak_worker_boxes={peak_worker_boxes:,} "
        f"elapsed={elapsed:.1f}s")
    reduce_stats = None
    if reduce_acc is not None:
        totals = [int(x) for x in reduce_acc.sum(axis=0)]
        log("[pool] REDUCE totals=" + " ".join(
            f"{c}={v:,}" for c, v in zip(interval_core.REDUCE_STATS_COLUMNS, totals))
            + f" max_prot={[float(x) for x in reduce_prot_acc.max(axis=0)]}")
        reduce_stats = {
            "columns": list(interval_core.REDUCE_STATS_COLUMNS),
            "per_step": {str(s): [int(x) for x in reduce_acc[s]]
                         for s in range(horizon + 1) if reduce_acc[s].any()},
            "totals": totals,
        }

    return {
        "status": status,
        "first_unsafe_step": first_unsafe_step,
        "first_unsafe_exact": first_unsafe_exact,
        "capture": unsafe_capture,
        "post_crossing_segments": int(post_segments),
        "post_crossing_cut": post_crossing_cut,
        "n_crossings": int(n_crossings),
        "unsafe_boxes_total": int(unsafe_boxes_total),
        "unsafe_boxes_captured": int(unsafe_boxes_captured),
        "unsafe_boxes_final": int(unsafe_capture["lo"].shape[0]) if unsafe_capture else 0,
        "crash": crash,
        "peak_pool_boxes": int(peak_pool_boxes),
        "peak_worker_boxes": int(peak_worker_boxes),
        "n_segments": segments_completed,
        "n_packs": packs_dispatched,
        "dispatch_coalesced": int(coalesce_stats["dispatches"]),
        "dispatch_coalesced_packs": int(coalesce_stats["packs_merged"]),
        "pool_min_step": pool_min_step,
        "vram_pool_full": bool(vram_pool_full),
        "pool_budget_bytes": int(dev_budget),
        "pool_target_bytes": int(dev_target),
        "pool_phys_peak_bytes": {str(d): int(v) for d, v in dev_phys_peak.items()},
        "pool_guard_cut_bytes": {str(d): int(v) for d, v in dev_guard["cut"].items()},
        "pool_peak": dict(peak),
        "boxes_per_step": step_peak.tolist(),
        "reduce_stats": reduce_stats,
        "seg_records": seg_records,
        "elapsed_sec": elapsed,
    }


def _merge_captures(captures, earliest, max_boxes, seed, log=print):
    """Union of the violating boxes of every violating segment. Above
    ``max_boxes`` rows a uniform random subsample is kept (and logged)."""
    base = dict(earliest if earliest is not None else captures[0])
    caps = [c for c in captures if c["lo"].shape[0] > 0]
    if not caps:
        return base
    lo = np.concatenate([c["lo"] for c in caps], 0)
    hi = np.concatenate([c["hi"] for c in caps], 0)
    orig = np.concatenate([c["origin"] for c in caps], 0)
    acth = np.concatenate([c["act_hist"] for c in caps], 0)
    step = np.concatenate([c["step"] for c in caps], 0)
    n_total = int(sum(int(c["n_total"]) for c in caps))
    n = int(lo.shape[0])
    if max_boxes and n > max_boxes:
        sel = np.sort(np.random.default_rng(seed).choice(n, size=int(max_boxes), replace=False))
        log(f"[pool] captured {n:,} rows > cap {max_boxes:,}: random subsample (seed {seed})")
        lo, hi, orig, acth, step = lo[sel], hi[sel], orig[sel], acth[sel], step[sel]
    steps, cnts = np.unique(step, return_counts=True)
    log(f"[pool] captured violating boxes: {int(lo.shape[0]):,} rows from {len(caps)} "
        f"segment(s), true count {n_total:,}; per step: "
        + ", ".join(f"{int(a)}:{int(b):,}" for a, b in zip(steps, cnts)))
    base.update({"lo": lo, "hi": hi, "origin": orig, "act_hist": acth, "step": step,
                 "n_total": n_total})
    return base


def save_violating_boxes(path, capture, init_lo, init_hi):
    """Write the captured violating boxes to ``path`` (.npz): per box its bounds,
    origin, action history and violating step, plus the start set."""
    np.savez(
        path,
        first_unsafe_step=np.asarray(int(capture["first_unsafe_step"])),
        box_lo=np.asarray(capture["lo"]),
        box_hi=np.asarray(capture["hi"]),
        box_origin=np.asarray(capture["origin"], dtype=np.int32),
        box_action_history=np.asarray(capture["act_hist"]),
        box_step=np.asarray(capture["step"], dtype=np.int32),
        n_violating_total=np.asarray(int(capture["n_total"])),
        init_lo=np.asarray(init_lo, dtype=np.float64),
        init_hi=np.asarray(init_hi, dtype=np.float64),
    )
    return path
