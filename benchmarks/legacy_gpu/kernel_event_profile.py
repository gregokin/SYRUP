"""FIXED-WET-INPUT ISOLATED SEDIMENT replay: per-launch CUDA-event profile of the CUDA legacy sediment step (task gpu_sediment, P1).

    python benchmarks/legacy_gpu/kernel_event_profile.py --case-kind chastre --case <case dir> --state-npz <wet_state.npz> \
        --steps 60 --repeats 2 --output <NEW report.json> [--record-strategy compact|all] [--expect-state-sha256 HEX] \
        [--hash-only-tile-verify] [--allow-maple-source-change] [--applied-rainfall <csv>] [--end-s S] [--gpu-memory-gib G]

NOT a full storm, NOT hydrology, NOT end-to-end physics acceptance. It replays the sediment step of an actual `CudaLegacyContext` (the unchanged
production kernels, record strategy, static arrays and launch order; the legacy fixed-composition, unlimited-supply, artificial-clip-source
benchmark replay, no MAPLE bed) on supplied wet state(s) (`depth_m`, `velocity_m_s`, `rain_m_s`, as in `record_strategy_microbench.load_state`; one
state for the actual fixtures, K > 1 states are cycled step by step), uploaded once. Hydrology, case intake and the water accounting are excluded. All physical classes stay evaluated.

How it measures. A temporary wrapper around `ctx._launch` records a CUDA event before and after each launch call on the current stream and then
calls the ORIGINAL method with the SAME arguments (grid, block, argument tuple and order are never touched). The wrapper is installed only for
instrumented passes and removed (the original method restored, also on failure). Each repeat runs one UNINSTRUMENTED and one INSTRUMENTED pass over
the same input sequence from `reset()`; the order alternates by repeat. After every pass (outside the measured loop) the ledger, tallies, flags and all
maps are read back and must be BITWISE equal across every pass, instrumented or not, otherwise the harness refuses (and writes a failure record).
Measured: whole-loop device (CUDA events) and wall time of each pass; per-launch event intervals aggregated by kernel name and group. Event intervals
bracket the host-side enqueue, so they include instrumentation overhead and any device idle time while the host was enqueuing (small kernels are
host-bound): they are EXPLORATORY and not an exact production-time attribution; use the instrumented/uninstrumented loop ratio to size the distortion.
Setup (case, network, physics, NVRTC/context build) and warm-up are reported separately from the steady passes. Two repeats on one device under
background load are exploratory only.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABEL = ("FIXED-WET-INPUT ISOLATED SEDIMENT replay of the CUDA legacy context (per-launch CUDA-event profile); not a full storm, not "
         "hydrology, not end-to-end physics acceptance; legacy fixed-composition unlimited-supply replay, no MAPLE bed")
GROUPS = {
    "sg_laws": "wet laws", "sg_post_infiltration": "wet laws",
    "sg_values": "walk values",
    "sg_gather": "ordered gather (+ ring/inactive)", "sg_ring_inactive": "ordered gather (+ ring/inactive)",
    "sg_cn_level": "Crank-Nicolson", "sg_cn_block": "Crank-Nicolson",
    "sg_reduce_partial": "accounting/reductions/tallies", "sg_reduce_final": "accounting/reductions/tallies",
    "sg_outlet": "accounting/reductions/tallies", "sg_tally": "accounting/reductions/tallies",
}
UNCLASSIFIED = "unclassified"
CAVEATS = (
    ("per-launch intervals bracket the host-side enqueue on the current stream: they include event overhead and device idle time while the host "
     "enqueues; they are not isolated kernel times"),
    "the instrumented loop is slower than the uninstrumented loop by the reported ratio; production time is the uninstrumented one",
    ("the supplied input state(s) are replayed for every step, cycled when the file holds K > 1 states (the actual fixtures have K = 1, i.e. one "
     "state replayed every step); the pools evolve; not a storm; two repeats on one device under background load: exploratory"),
    "diagnostic host reads (flags, ledger, tallies, maps) happen after each pass, outside the measured loops, and are reported separately",
)


class ProfileError(RuntimeError):
    """A refused or inconsistent profile (nothing is fabricated)."""


class OutputRefused(ProfileError):
    """The output path is refused (exists, protected tree): nothing, not even a failure record, is written there."""


# ---- pure helpers (CPU-testable) --------------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_controls(steps: Any, repeats: Any, gpu_memory_gib: Any, strategy: str) -> None:
    for name, value in (("steps", steps), ("repeats", repeats)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ProfileError(f"--{name} must be a positive integer, got {value!r}")
    if gpu_memory_gib is not None and (isinstance(gpu_memory_gib, bool) or not isinstance(gpu_memory_gib, (int, float))
                                       or not math.isfinite(gpu_memory_gib) or gpu_memory_gib <= 0.0):
        raise ProfileError("--gpu-memory-gib must be finite and > 0")
    if strategy not in ("compact", "all"):
        raise ProfileError("--record-strategy must be compact or all")


def validate_output(out: Path, forbidden: dict[str, Path]) -> Path:
    """Exclusive-create report path: absent (with its `.FAILED` record), parent an existing directory, and neither equal to, inside nor
    containing any forbidden tree (case, state file, project source, references), after resolving symlinks."""
    out = Path(out)
    failed = out.with_name(out.name + ".FAILED")
    for candidate in (out, failed):
        if os.path.lexists(candidate):
            raise OutputRefused(f"refusing to write: {candidate} exists")
    if not out.resolve().parent.is_dir():
        raise OutputRefused(f"the output directory {out.resolve().parent} does not exist")
    real = out.resolve()
    for label, root in forbidden.items():
        r = Path(root).resolve()
        if real == r or real.is_relative_to(r) or r.is_relative_to(real):
            raise OutputRefused(f"refusing to write {out}: it is, lies inside or contains the {label} tree {r}")
    return out


def group_of(name: str) -> str:
    return GROUPS.get(name, UNCLASSIFIED)


def aggregate(names: list[str], ms: list[float]) -> dict[str, Any]:
    """Per-kernel and per-group totals of one pass from the launch names and their event intervals (ms)."""
    if len(names) != len(ms):
        raise ProfileError("launch names and intervals differ in length")
    by_kernel: dict[str, dict[str, float]] = {}
    by_group: dict[str, dict[str, float]] = {}
    for name, t in zip(names, ms, strict=True):
        if not math.isfinite(t) or t < 0.0:
            raise ProfileError(f"a launch interval is not finite and >= 0 ({name}: {t!r})")
        for table, key in ((by_kernel, name), (by_group, group_of(name))):
            rec = table.setdefault(key, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
            rec["count"] += 1
            rec["total_ms"] += t
            rec["max_ms"] = max(rec["max_ms"], t)
    for table in (by_kernel, by_group):
        for rec in table.values():
            rec["mean_us"] = 1000.0 * rec["total_ms"] / rec["count"]
    return {"by_kernel": by_kernel, "by_group": by_group, "launches": len(names), "sum_intervals_ms": float(sum(ms)),
            "unclassified_kernels": sorted({n for n in names if group_of(n) == UNCLASSIFIED})}


def run_length(names: list[str]) -> list[list[Any]]:
    """The launch-name sequence as `[name, run_length]` pairs (the order of one step is thereby recorded, compactly)."""
    out: list[list[Any]] = []
    for n in names:
        if out and out[-1][0] == n:
            out[-1][1] += 1
        else:
            out.append([n, 1])
    return out


class LaunchRecorder:
    """Temporary `ctx._launch` wrapper with a preallocated, bounded set of event pairs. `ops.event()` creates an event with `.record()`;
    `ops.elapsed_ms(a, b)` reads an interval after synchronisation. The wrapper never alters a launch: it records, calls the original with
    the very same arguments and records again."""

    def __init__(self, ctx: Any, capacity: int, ops: Any):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ProfileError("the event capacity must be a positive integer")
        self.ctx, self.capacity, self.ops = ctx, capacity, ops
        self.start = [ops.event() for _ in range(capacity)]
        self.stop = [ops.event() for _ in range(capacity)]
        self.names: list[str] = []
        self._missing = object()
        self._saved: Any = self._missing
        self._installed = False

    def _wrapped(self, name: str, grid: int, block: int, args: tuple) -> None:
        i = len(self.names)
        if i >= self.capacity:
            raise ProfileError(f"more launches than the declared capacity {self.capacity} (the launch count diverged from the warm-up)")
        self.start[i].record()
        self._real(name, grid, block, args)
        self.stop[i].record()
        self.names.append(name)

    def install(self) -> None:
        if self._installed:
            raise ProfileError("the launch recorder is already installed")
        self._saved = self.ctx.__dict__.get("_launch", self._missing)
        self._real = type(self.ctx)._launch.__get__(self.ctx) if self._saved is self._missing else self._saved
        self.ctx._launch = self._wrapped
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        if self._saved is self._missing:
            self.ctx.__dict__.pop("_launch", None)
        else:
            self.ctx.__dict__["_launch"] = self._saved
        self._installed = False

    def clear(self) -> None:
        self.names = []

    def intervals_ms(self) -> list[float]:
        """Call after the stream was synchronised; refuses a partial pass (the count must equal the capacity exactly)."""
        if len(self.names) != self.capacity:
            raise ProfileError(f"{len(self.names)} launches were recorded but the capacity (the warm-up count) is {self.capacity}")
        return [float(self.ops.elapsed_ms(a, b)) for a, b in zip(self.start, self.stop, strict=True)]


@contextlib.contextmanager
def profiling(ctx: Any, capacity: int, ops: Any):
    """Install the recorder; ALWAYS restore the original `_launch`, also when the body raises."""
    recorder = LaunchRecorder(ctx, capacity, ops)
    recorder.install()
    try:
        yield recorder
    finally:
        recorder.restore()


def capture(ctx: Any, steps: int) -> dict[str, np.ndarray]:
    """Host copies of every result of a pass (after `check_flags`): ledger rows, tallies, flag words and all per-cell maps."""
    out = {"ledger": np.array(ctx.host_ledger[:steps]), "counts": np.array(ctx.host_counts[:steps]), "flags": np.array(ctx.host_flags[:steps])}
    for key, value in ctx.download_maps().items():
        out[f"map_{key}"] = np.array(value)
    return out


#: every field a pass must capture: the ledger rows, the walk/regime tallies, the flag words and the five per-cell maps of `download_maps`
REQUIRED_FIELDS = ("ledger", "counts", "flags", "map_cum_det", "map_cum_dep", "map_cum_clip", "map_mobile", "map_v_prev")


def check_complete(cap: dict[str, np.ndarray], label: str) -> None:
    """A capture must hold EVERY required field (a missing ledger, counters or map is a failure, not a smaller comparison), each non-empty,
    finite where floating, and without a set flag word."""
    if not cap:
        raise ProfileError(f"{label}: an empty capture")
    missing = [k for k in REQUIRED_FIELDS if k not in cap]
    if missing:
        raise ProfileError(f"{label}: the capture lacks the required field(s) {missing}")
    for key, a in cap.items():
        if a.size == 0:
            raise ProfileError(f"{label}: {key} is empty")
        if a.dtype.kind in "fc" and not np.isfinite(a).all():
            raise ProfileError(f"{label}: {key} holds non-finite values")
    if np.any(cap["flags"]):
        raise ProfileError(f"{label}: a device flag word is nonzero")


def require_unchanged(pairs: dict[str, tuple[Any, Any]]) -> None:
    """Refuse publication when any before/after provenance pair differs; the message names every changed item."""
    changed = [name for name, (before, after) in pairs.items() if before != after]
    if changed:
        raise ProfileError(f"changed during the run ({', '.join(changed)}); nothing is published")


INPUT_KEYS = ("depth_m", "velocity_m_s", "rain_m_s")
NONVACUITY_NOTE = ("profile diagnostic of THIS legacy replay (fixed composition, unlimited supply, artificial clip source): it shows the profiled "
                   "input and result were not vacuous; it is not a conservation, acceptance or full-storm claim and adds no bound")


def input_statistics(state: dict[str, np.ndarray]) -> dict[str, Any]:
    """Min, max and positive-cell counts of the host input state(s), shape `(K, ny, nx)` per key (no device read)."""
    out: dict[str, Any] = {}
    for key in INPUT_KEYS:
        if key not in state:
            raise ProfileError(f"the input state lacks {key}")
        a = np.asarray(state[key])
        if a.dtype.kind != "f" or a.ndim != 3 or a.size == 0 or not np.isfinite(a).all():
            raise ProfileError(f"the input {key} must be a non-empty finite float array of shape (K, ny, nx)")
        positive = a > 0.0
        out[key] = {"min": float(a.min()), "max": float(a.max()), "cells_per_state": int(a[0].size), "n_states": int(a.shape[0]),
                    "positive_cells_per_state": [int(p.sum()) for p in positive], "positive_cells_total": int(positive.sum())}
    return out


def nonvacuity(ledger: np.ndarray, columns: tuple[str, ...], state: dict[str, np.ndarray]) -> dict[str, Any]:
    """Per-class and total pickup / active deposition (sums of the per-step kg columns over the pass) and the final mobile inventory (the last
    row of the `new_mobile_kg` column), from the HOST ledger the pass already read (no additional device read), plus the input statistics. A dry or
    zero-flow input is supported but identified as such."""
    ledger = np.asarray(ledger)
    if ledger.ndim != 3 or ledger.shape[0] < 1 or ledger.shape[1] != len(columns) or not np.isfinite(ledger).all():
        raise ProfileError("the ledger is not a finite (steps, columns, classes) array matching the column names")
    index = {}
    for name in ("pickup_kg", "deposition_active_kg", "new_mobile_kg"):
        if name not in columns:
            raise ProfileError(f"the ledger columns lack {name}")
        index[name] = columns.index(name)
    pickup = ledger[:, index["pickup_kg"], :].sum(axis=0)
    deposition = ledger[:, index["deposition_active_kg"], :].sum(axis=0)
    mobile = ledger[-1, index["new_mobile_kg"], :]
    inputs = input_statistics(state)
    wet_inputs = inputs["depth_m"]["positive_cells_total"] > 0 and inputs["velocity_m_s"]["positive_cells_total"] > 0
    moved = bool(pickup.sum() > 0.0)
    if wet_inputs and moved:
        status = "wet: positive depth and velocity inputs and nonzero pickup"
    elif wet_inputs:
        status = "wet inputs but ZERO pickup (check composition and laws); the profile may not be representative"
    else:
        status = "DRY or zero-flow input state: no wet-law work; the profile is not representative of a wet storm"
    return {"status": status, "inputs_wet": bool(wet_inputs), "pickup_nonzero": moved,
            "pickup_kg_total_by_class": pickup.tolist(), "pickup_kg_total": float(pickup.sum()),
            "deposition_active_kg_total_by_class": deposition.tolist(), "deposition_active_kg_total": float(deposition.sum()),
            "final_mobile_kg_by_class": mobile.tolist(), "final_mobile_kg_total": float(mobile.sum()),
            "columns_used": {k: int(v) for k, v in index.items()}, "inputs": inputs,
            "source": "ctx.host_ledger of the last measured pass (rows already read by check_flags); no additional device read or download",
            "note": NONVACUITY_NOTE}


def bitwise_differences(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[str]:
    bad = [k for k in sorted(set(a) | set(b)) if k not in a or k not in b]
    for k in sorted(set(a) & set(b)):
        if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or a[k].tobytes() != b[k].tobytes():
            bad.append(k)
    return bad


def _pass(ctx: Any, dev: list, steps: int, ops: Any, recorder: LaunchRecorder | None) -> dict[str, Any]:
    """One pass from `reset()` over the supplied inputs. Only the step loop is timed; the diagnostic reads follow it."""
    ctx.reset()
    if recorder is not None:
        recorder.clear()
    k = len(dev)
    start, stop = ops.event(), ops.event()
    w0 = ops.clock()
    start.record()
    for row in range(steps):
        ctx.step(row, *dev[row % k])
    stop.record()
    ops.sync(stop)
    wall = ops.clock() - w0
    device_s = ops.elapsed_ms(start, stop) / 1000.0
    launches = int(ctx.stats["launches"])
    result: dict[str, Any] = {"loop_wall_s": wall, "loop_device_s": device_s, "launches": launches, "steps": int(ctx.stats["steps"])}
    if recorder is not None:
        result["event_intervals_ms"] = recorder.intervals_ms()
        result["launch_names"] = list(recorder.names)
    t0 = ops.clock()
    ctx.check_flags()  # diagnostic host read, outside the measured loop
    cap = capture(ctx, steps)
    result["diagnostic_read_wall_s"] = ops.clock() - t0
    result["diagnostic_transfers"] = {k: int(v) for k, v in ctx.stats.items() if k.startswith("d2h_")}
    result["capture"] = cap
    return result


def run_profile(ctx: Any, dev: list, steps: int, repeats: int, ops: Any) -> dict[str, Any]:
    """Warm up at the measured shape, then `repeats` paired passes (uninstrumented / instrumented, order alternating by repeat) on the same inputs
    from the same reset state. Bitwise equivalence of every capture is REQUIRED."""
    if not dev:
        raise ProfileError("no input states")
    t0 = ops.clock()
    warm_plain = _pass(ctx, dev, steps, ops, None)
    launches_per_pass = warm_plain["launches"]
    if launches_per_pass < 1 or launches_per_pass % steps:
        raise ProfileError(f"{launches_per_pass} launches in {steps} steps: not a constant per-step count")
    check_complete(warm_plain["capture"], "warm-up")
    with profiling(ctx, launches_per_pass, ops) as recorder:  # events are preallocated from the ACTUAL warm-up launch count
        warm_instr = _pass(ctx, dev, steps, ops, recorder)
    check_complete(warm_instr["capture"], "instrumented warm-up")
    bad = bitwise_differences(warm_plain["capture"], warm_instr["capture"])
    if bad:
        raise ProfileError(f"the instrumented warm-up differs from the plain warm-up in {bad}")
    reference = warm_plain["capture"]
    warmup_s = ops.clock() - t0
    per_step = launches_per_pass // steps
    first_step = run_length(warm_instr["launch_names"][:per_step])
    passes, pairs = [], []
    with profiling(ctx, launches_per_pass, ops) as recorder:
        recorder.restore()  # the recorder is installed only around instrumented passes
        for r in range(repeats):
            order = ("plain", "instrumented") if r % 2 == 0 else ("instrumented", "plain")
            done: dict[str, dict[str, Any]] = {}
            for mode in order:
                if mode == "instrumented":
                    recorder.install()
                    try:
                        res = _pass(ctx, dev, steps, ops, recorder)
                    finally:
                        recorder.restore()
                else:
                    res = _pass(ctx, dev, steps, ops, None)
                check_complete(res["capture"], f"repeat {r} {mode}")
                diff = bitwise_differences(reference, res["capture"])
                if diff:
                    raise ProfileError(f"repeat {r} {mode}: results differ from the warm-up reference in {diff} (instrumentation or state changed outputs)")
                if res["launches"] != launches_per_pass:
                    raise ProfileError(f"repeat {r} {mode}: {res['launches']} launches, expected {launches_per_pass}")
                res.pop("capture")
                done[mode] = res
            instr = done["instrumented"]
            agg = aggregate(instr.pop("launch_names"), instr.pop("event_intervals_ms"))
            instr["aggregate"] = agg
            for mode in order:
                passes.append({"repeat": r, "mode": mode, "order_in_repeat": order.index(mode), **{k: v for k, v in done[mode].items()}})
            pairs.append({"repeat": r, "order": list(order),
                          "plain_loop_device_s": done["plain"]["loop_device_s"], "instrumented_loop_device_s": instr["loop_device_s"],
                          "plain_loop_wall_s": done["plain"]["loop_wall_s"], "instrumented_loop_wall_s": instr["loop_wall_s"],
                          "instrumented_over_plain_device": instr["loop_device_s"] / done["plain"]["loop_device_s"],
                          "instrumented_over_plain_wall": instr["loop_wall_s"] / done["plain"]["loop_wall_s"],
                          "sum_launch_intervals_s": agg["sum_intervals_ms"] / 1000.0,
                          "sum_intervals_over_instrumented_device_loop": agg["sum_intervals_ms"] / 1000.0 / instr["loop_device_s"]})
    instr_aggs = [p["aggregate"] for p in passes if p["mode"] == "instrumented"]

    def med(table: str) -> dict[str, Any]:
        keys = sorted({k for a in instr_aggs for k in a[table]})
        return {k: {"median_total_ms": statistics.median(a[table][k]["total_ms"] for a in instr_aggs if k in a[table]),
                    "count": instr_aggs[0][table][k]["count"] if k in instr_aggs[0][table] else None} for k in keys}

    plain_dev = [p["loop_device_s"] for p in passes if p["mode"] == "plain"]
    instr_dev = [p["loop_device_s"] for p in passes if p["mode"] == "instrumented"]
    return {
        "launches_per_pass": launches_per_pass, "launches_per_step": per_step, "steps": steps, "repeats": repeats,
        "first_step_launch_sequence": first_step, "warmup_s_both_passes_and_checks": warmup_s,
        "warmup": {"plain_loop_device_s": warm_plain["loop_device_s"], "instrumented_loop_device_s": warm_instr["loop_device_s"]},
        "passes": passes, "pairs": pairs,
        "medians": {"plain_loop_device_s": statistics.median(plain_dev), "instrumented_loop_device_s": statistics.median(instr_dev),
                    "by_kernel": med("by_kernel"), "by_group": med("by_group")},
        "equivalence": {"bitwise_equal_across_all_passes_and_toggle": True, "fields": sorted(reference),
                        "note": "every pass capture (ledger, tallies, flags, all maps) equals the warm-up reference byte for byte"},
        "caveats": list(CAVEATS),
    }


# ---- GPU/case driver (needs CuPy and the project; not used by the CPU tests) --------------------------------------------------------
def _load_microbench() -> Any:
    path = Path(__file__).with_name("record_strategy_microbench.py")
    spec = importlib.util.spec_from_file_location("record_strategy_microbench_for_profile", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case-kind", required=True, choices=("plot1", "rfid", "chastre"))
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--state-npz", type=Path, required=True)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--record-strategy", choices=("compact", "all"), default="compact")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expect-state-sha256", default=None, help="refuse unless the state file has this SHA-256")
    p.add_argument("--end-s", type=float, default=None)
    p.add_argument("--applied-rainfall", type=Path, default=None)
    p.add_argument("--allow-maple-source-change", action="store_true")
    p.add_argument("--hash-only-tile-verify", action="store_true")
    p.add_argument("--gpu-memory-gib", type=float, default=None)
    return p


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    from maple_syrup import legacy_driver as D
    from maple_syrup import legacy_gpu_driver as G
    from maple_syrup import legacy_native as N
    from maple_syrup.column_experiment import _source_digests, _syrup_provenance
    from maple_syrup.legacy_case import legacy_case_for
    from maple_syrup.legacy_native_cuda import CudaLegacyContext
    from maple_syrup.legacy_physics_numba import prepare_legacy_physics
    from maple_syrup.routing_cuda import _cupy

    args.output_validated = False  # main() writes a failure record only for a path that passed validation
    validate_controls(args.steps, args.repeats, args.gpu_memory_gib, args.record_strategy)
    case_dir, state_path = args.case.resolve(), args.state_npz.resolve()
    if not state_path.is_file():
        raise ProfileError(f"{state_path} is not a file")
    forbidden = {f"project {n}": ROOT / n for n in ("src", "tests", "benchmarks", "cases", "docs")}
    forbidden.update({"case": case_dir, "state file": state_path, **D.protected_roots(case_dir)})
    out = validate_output(args.output, forbidden)  # before any case load or device work
    args.output_validated = True
    state_sha = sha256_file(state_path)
    if args.expect_state_sha256 is not None and state_sha != args.expect_state_sha256:
        raise ProfileError(f"the state file SHA-256 {state_sha} differs from the expected {args.expect_state_sha256}")
    self_sha = sha256_file(Path(__file__))
    micro_path = Path(__file__).with_name("record_strategy_microbench.py")
    micro_sha = sha256_file(micro_path)
    modules_before = G.module_digests()  # the explicit CPU-driver and GPU modules only; NOT a complete model-chain binding (see below)
    syrup_prov = _syrup_provenance()
    cp = _cupy()  # explicit CudaUnavailableError: no CPU fallback
    device = G.device_record(cp)
    micro = _load_microbench()
    t0 = time.perf_counter()
    case = legacy_case_for(args.case_kind, args.case, allow_maple_source_change=args.allow_maple_source_change,
                           hash_only_tile_verify=args.hash_only_tile_verify, end_s=args.end_s, applied_rainfall=args.applied_rainfall)
    case_s = time.perf_counter() - t0
    dep = case.verified.maple_dependency
    roots = dict(forbidden)
    roots.update({"verified MAPLE source": Path(dep.source_root), "verified MAPLE package": Path(dep.package_dir)})
    mah = (case.verified.report.get("mahleran") or {}).get("root")
    if mah:
        roots["verified MAHLERAN"] = Path(mah)
    for i, p in enumerate(case.pin_paths):
        roots[f"bound input {i}"] = Path(p)
    args.output_validated = False
    validate_output(out, roots)  # again, with the verified dependency and every bound input, before the first device allocation
    args.output_validated = True
    # package-wide source digests of the actual SYRUP package and the verified MAPLE package (the same guard as the GPU driver), taken BEFORE
    # any physics preparation or GPU context work and compared after the profile
    digests_before = _source_digests(Path(syrup_prov["package_dir"]), dep.package_dir)
    pins_before = {str(p): D._sha256_stream(Path(p)) for p in case.pin_paths}
    state = micro.load_state(state_path, tuple(case.shape))
    for a in state.values():
        a.setflags(write=False)
    t0 = time.perf_counter()
    network = N.native_network(case.graph)
    network_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    physics = prepare_legacy_physics(case.sediment, case.grid, case.vegetation, case.holdings_kg)
    physics_s = time.perf_counter() - t0
    budget = int(args.gpu_memory_gib * 2**30) if args.gpu_memory_gib is not None else None
    t0 = time.perf_counter()
    ctx = CudaLegacyContext(network, physics, case.graph, limits=N.walk_limits(network.dx_m), dt=1.0, n_steps=args.steps,
                            memory_budget_bytes=budget, record_strategy=args.record_strategy)
    cp.cuda.Device().synchronize()
    context_s = time.perf_counter() - t0
    n_states = state["depth_m"].shape[0]
    dev = [tuple(cp.asarray(state[key][j]) for key in ("depth_m", "velocity_m_s", "rain_m_s")) for j in range(n_states)]  # uploaded once
    ops = SimpleNamespace(event=lambda: cp.cuda.Event(), elapsed_ms=lambda a, b: cp.cuda.get_elapsed_time(a, b),
                          sync=lambda e: e.synchronize(), clock=time.perf_counter)
    profile = run_profile(ctx, dev, args.steps, args.repeats, ops)
    ctx._guard("profile end")  # the sealed context structure must be intact
    for j, triple in enumerate(dev):  # the uploaded inputs are unchanged by every pass
        for key, arr in zip(("depth_m", "velocity_m_s", "rain_m_s"), triple, strict=True):
            if cp.asnumpy(arr).tobytes() != state[key][j].tobytes():
                raise ProfileError(f"the device input {key}[{j}] changed during the profile")
    if "_launch" in ctx.__dict__:
        raise ProfileError("ctx._launch was not restored")
    modules_after = G.module_digests()
    digests_after = _source_digests(Path(syrup_prov["package_dir"]), dep.package_dir)
    pins_after = {str(p): D._sha256_stream(Path(p)) for p in case.pin_paths}
    from maple_syrup.legacy_native_numba import (
        LEDGER_COLUMNS,  # the actual column order of the ledger the context fills
    )

    evidence = nonvacuity(np.array(ctx.host_ledger[:args.steps]), tuple(LEDGER_COLUMNS), state)  # host data only, outside every loop
    require_unchanged({"explicit driver/GPU modules": (modules_before, modules_after),
                       "SYRUP and verified MAPLE package sources": (digests_before, digests_after),
                       "bound input artifacts": (pins_before, pins_after),
                       "state file": (state_sha, sha256_file(state_path)),
                       "this harness": (self_sha, sha256_file(Path(__file__))),
                       "record_strategy_microbench.py": (micro_sha, sha256_file(micro_path))})
    return {
        "label": LABEL, "utc_unix_s": time.time(),
        "inputs": {"state_npz": str(state_path), "state_sha256": state_sha, "n_states": int(n_states),
                   "case_kind": args.case_kind, "case": str(case_dir), "bound_input_sha256_before_and_after": pins_before,
                   "tile_check": "case verification only (hash-only allowed); no post-run tile hash for this isolated short profile"},
        "provenance": {"harness_sha256": self_sha, "record_strategy_microbench_sha256": micro_sha,
                       "explicit_module_sha256": modules_before,
                       "explicit_module_note": "only the CPU-driver chain and the GPU modules; the package-wide digests below bind the rest",
                       "maple_syrup": syrup_prov, "maple": case.verified.maple_provenance,
                       "source_digests_before": digests_before, "source_digests_after": digests_after,
                       "unchanged_during_run": True,
                       "scope": "initial verification only: no full voxel-bed scan and no post-run tile hash for this short isolated profile"},
        "device": device,
        "case_record": {"shape": list(case.shape), "n_classes": int(case.n_classes), "graph_input_sha256": case.graph.input_sha256,
                        "network": network.summary()},
        "controls": {"steps": args.steps, "repeats": args.repeats, "record_strategy": args.record_strategy,
                     "gpu_memory_gib": args.gpu_memory_gib, "physical_classes_evaluated": int(case.n_classes)},
        "context": {"record_strategy": ctx.record_info, "estimate_bytes": ctx.estimate, "memory": ctx.memory, "cn_mode": ctx.cn_mode,
                    "n_levels": ctx.n_levels, "walk_records": int(ctx.tables.n_records)},
        "setup_s": {"case_verify_and_adapt": case_s, "network": network_s, "physics": physics_s,
                    "context_build_including_compile_and_tables": context_s, "ctx_compile_and_tables": ctx.compile_s},
        "nonvacuity": evidence,
        "profile": profile,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = _execute(args)
        with args.output.open("x") as handle:
            handle.write(json.dumps(report, indent=2, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001 - intentional top-level handler: any failure (including device/event errors) is reported, status 1
        print(f"kernel event profile failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(args, "output_validated", False):  # never write anything to a path that was refused or not yet validated
            failed = args.output.with_name(args.output.name + ".FAILED")
            with contextlib.suppress(Exception), failed.open("x") as handle:  # exclusive create
                handle.write(json.dumps({"label": LABEL, "error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n")
        return 1
    print(json.dumps({"label": LABEL, "nonvacuity_status": report["nonvacuity"]["status"], "launches_per_step": report["profile"]["launches_per_step"],
                      "medians": {k: v for k, v in report["profile"]["medians"].items() if k.endswith("_s")},
                      "equivalence": report["profile"]["equivalence"]["bitwise_equal_across_all_passes_and_toggle"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
