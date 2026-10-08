"""Coupled resident-GPU native-walk MAHLERAN LEGACY sediment storm (task gpu_sediment, stage B1): Plot 1, RFID_2014, Chastre.

    python -m maple_syrup.legacy_gpu_driver --case-kind chastre --case outputs/chastre/case_v2 --output <NEW> \\
        --warmup-s 60 --hash-only-tile-verify --allow-maple-source-change --snapshot-times 600,1200,1800,2640,2700

The same case adapters, controls, output protection, publication (`<out>.partial` -> `<out>`, `<out>.FAILED` on failure), guards,
ledger/map files and summary contract as the CPU driver `maple_syrup.legacy_driver` (whose helpers are reused unchanged), but the
whole step is on the GPU: rainfall field -> the accepted CUDA hydrology (`hydrology_cuda`, bisection or Newton) -> MAHLERAN wet laws
-> native detachment-distance walk (static record tables, CPU-order gather) -> the accepted Crank-Nicolson pool -> ledger/maps, with
hydrology and sediment state resident on the device. No CPU fallback, no per-step full-grid transfer, no fast math.

Scientific status (never a conservation claim): fixed composition, UNLIMITED supply, explicit artificial clipping source, ring and
inactive deposition are diagnostics, terminal pits retain mobile mass, frozen terrain/routing, 1 s steps, no splash/ET/dry reset, the
window ends while flow may persist, no MAPLE bed is read or written. Unsupported modes (`--source-order legacy`,
`--legacy-depos-erase`, `--allow-python-kernels`) are rejected before any work.

Depth time levels: `previous` (default), `current`, and the native `post_infiltration` (the ORIGINAL infilt.for d(1), computed on the
device from the old state depth and the step's rain and intake). The wet-law velocity is the new step's velocity in all modes.

Host traffic: per step the hydrology packet (144 bytes) is read. In addition, every `--check-every-steps` steps (default 60) and at the
end small slices of sediment flags, ledger rows and tallies are read; depth/mobile snapshots at the requested times, the final water
maps and initial volume sums once, and the sediment maps once. Counters (`summary["gpu"]["transfers"]`) cover the measured run only;
the warm-up counters are kept separately. Nothing is published before the final check.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import resource
import sys
import time
import traceback
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup import legacy_driver as D
from maple_syrup import legacy_native as N
from maple_syrup.column_experiment import _source_digests, _syrup_provenance
from maple_syrup.legacy_case import LegacyCase, legacy_case_for
from maple_syrup.legacy_native_cuda import (
    RECORD_STRATEGIES,
    CudaLegacyContext,
    CudaLegacyError,
    kernel_provenance,
)
from maple_syrup.legacy_native_numba import LEDGER_COLUMNS, LEDGER_UNITS
from maple_syrup.legacy_physics_numba import prepare_legacy_physics
from maple_syrup.legacy_water_cuda import MODES as WATER_MODES
from maple_syrup.legacy_water_cuda import WaterAccounting
from maple_syrup.provenance import environment_record
from maple_syrup.routing_cuda import CudaUnavailableError, _cupy
from maple_syrup.sediment_physics import REGIME_CODES
from maple_syrup.storm import plan_boundaries

__all__ = ["build_parser", "main", "module_digests", "run", "validate_gpu_controls"]

_GPU_MODULES = ("maple_syrup.legacy_native_cuda", "maple_syrup.legacy_water_cuda", "maple_syrup.legacy_gpu_driver")
_LIMITATIONS = (
    "fixed composition (verified initial active layer), UNLIMITED supply, explicit artificial clipping source",
    "ring and inactive-cell deposition are walk diagnostics, not debited from the active pools; terminal pits retain mobile mass",
    "no splash, no ET, no dry reset, fixed 1 s step, frozen elevation and routing, window ends while flow may persist",
    "no MAPLE bed mutation, no conservation claim, no restart, no wind handoff; GPU resident (CUDA), no CPU fallback",
    ("source order 'index' only, no order-dependent erasure (rejected, not ignored); laws, velocities, mobile pools and ledgers of ALL "
     "classes are evaluated (only the walk record values are compacted to the composition-eligible classes, see gpu.record_strategy)"),
    "the whole-storm water budget uses the unchanged MAPLE-derived volume bound; it says nothing about sediment conservation",
)


def build_parser() -> argparse.ArgumentParser:
    p = D.build_parser()
    p.prog = "python -m maple_syrup.legacy_gpu_driver"
    p.description = __doc__.split("\n\n")[0]
    p.add_argument("--check-every-steps", type=int, default=60, help="read the device flags/ledger rows every N steps (and at the end)")
    p.add_argument("--cuda-mode", choices=("auto", "fused", "split"), default="auto", help="hydrology launch structure")
    p.add_argument("--cn-mode", choices=("auto", "level", "block"), default="auto", help="Crank-Nicolson launch structure")
    p.add_argument("--gpu-memory-gib", type=float, default=None, help="refuse if the estimated device memory exceeds this")
    p.add_argument("--water-accounting", choices=WATER_MODES, default="fused",
                   help="per-step water series and cumulative maps: 'fused' = one device launch per step, 'separate' = the B1/B2 reference "
                        "(3 scalar copies + 4 adds); identical arithmetic, device-only either way (default fused: provisional)")
    p.add_argument("--record-strategy", choices=RECORD_STRATEGIES, default="compact",
                   help="walk record values: 'compact' = only classes with a positive composition fraction in an active cell "
                        "(static eligibility, exactly the same science), 'all' = every class (the B1 reference)")
    return p


def validate_gpu_controls(args: argparse.Namespace) -> None:
    """Before anything is allocated: the CPU controls, then the GPU-specific ones and the unsupported modes."""
    if args.source_order != "index":
        raise D.DriverError("--source-order legacy is not supported by the CUDA replay (rejected before any work)")
    if args.legacy_depos_erase:
        raise D.DriverError("--legacy-depos-erase is not supported by the CUDA replay (rejected before any work)")
    if args.allow_python_kernels:
        raise D.DriverError("--allow-python-kernels does not apply to the CUDA replay (there is no CPU fallback)")
    D.validate_controls(args)
    if isinstance(args.check_every_steps, bool) or args.check_every_steps < 1:
        raise D.DriverError("--check-every-steps must be an integer >= 1")
    if args.gpu_memory_gib is not None and not (math.isfinite(args.gpu_memory_gib) and args.gpu_memory_gib > 0.0):
        raise D.DriverError("--gpu-memory-gib must be finite and > 0")


def module_digests() -> dict[str, str]:
    """Source-file digests of the CPU driver chain AND the two GPU modules (resolved from files, not `sys.modules`)."""
    out = dict(D.module_digests())
    for name in _GPU_MODULES:
        path = Path(__file__).resolve() if name == "maple_syrup.legacy_gpu_driver" else None
        if path is None:
            spec = importlib.util.find_spec(name)
            if spec is None or not spec.origin or not Path(spec.origin).is_file():
                raise D.DriverError(f"cannot locate the source file of {name} for provenance")
            path = Path(spec.origin)
        out[name] = D._sha256_stream(path)
    return out


def device_record(cp) -> dict[str, Any]:
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(int(dev.id))
    name = props.get("name", b"")
    uuid = props.get("uuid", None)
    return {"device_id": int(dev.id), "name": name.decode() if isinstance(name, bytes) else str(name),
            "uuid_hex": uuid.hex() if isinstance(uuid, (bytes, bytearray)) else None,
            "pci_bus_id": cp.cuda.runtime.deviceGetPCIBusId(int(dev.id)),
            "compute_capability": f"{props.get('major')}.{props.get('minor')}",
            "total_global_mem_bytes": int(props.get("totalGlobalMem", 0)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cupy": getattr(cp, "__version__", None)}


def gpu_hydrology_inputs(case: LegacyCase):
    """The device graph/column/field/initial state for the SAME case (the host graph's digest must agree)."""
    if case.kind == "plot1":
        from maple_syrup.sediment_experiment import prepare_verified_sediment_case

        prepared = prepare_verified_sediment_case(case.verified, backend="cupy", end_s=case.end_s)
        graph, column, field, storm0 = prepared["graph"], prepared["column"], prepared["field"], prepared["state0"].storm
    else:
        from maple_syrup.rfid_case import rfid_inputs
        from maple_syrup.storm import initial_state

        inputs = rfid_inputs(case.verified, "cupy", with_geometry=False)
        graph, column, field = inputs.graph, inputs.params, inputs.field
        storm0 = initial_state(graph, inputs.depth0, inputs.soil0)
    if graph.input_sha256 != case.graph.input_sha256:
        raise D.DriverError("the device routing graph differs from the host graph (digest mismatch)")
    return graph, column, field, storm0


def _loop(cp, case, args, ctx, hctx, hc, control, field, storm0, boundaries, snapshots, collect: bool, label: str) -> dict[str, Any]:
    ny, nx = case.shape
    nc = case.n_classes
    steps = boundaries.size
    # outlet discharge / export / budget residual series and the four cumulative water maps: device-only (`legacy_water_cuda`)
    acct = WaterAccounting(cp, (ny, nx), steps, args.water_accounting)
    rate_dev = cp.zeros((ny, nx), dtype=np.float64)
    hpre = cp.zeros((ny, nx), dtype=np.float64) if args.depth_time_level == "post_infiltration" else None
    storm = storm0
    prev_depth = storm0.depth_m
    snap = {"t": [], "depth": [], "mobile": []}
    pending = list(snapshots)
    events = [cp.cuda.Event() for _ in range(2 * steps + 1)] if collect else []
    if collect:
        events[0].record()
    last = None
    t = 0.0
    progress_s = 0.0
    next_progress = args.progress_every_s if args.progress_every_s > 0 else math.inf
    checked = 0
    t_loop0 = time.perf_counter()
    first_step_s = None
    for row, boundary in enumerate(boundaries.tolist()):
        dt = boundary - t
        if dt != 1.0:
            raise D.DriverError(f"step {row}: dt = {dt!r}; the legacy replay requires exactly 1 s steps")
        field.apply(case.schedule.rate_after_m_per_s(t), out=rate_dev)
        step, packet = hc.cuda_step_with_packet(hctx, rate_dev, storm, dt, control)  # the ONE counted packet read of the step
        route, col = step.route, step.column
        if collect:
            events[1 + 2 * row].record()
        mode = args.depth_time_level
        if mode == "previous":
            law_depth = prev_depth
        elif mode == "current":
            law_depth = route.depth_m
        else:
            ctx.post_infiltration_depth(prev_depth, col.rain_m, col.intake_m, hpre)
            law_depth = hpre
        ctx.step(row, law_depth, route.velocity_m_s, rate_dev)
        if collect:
            events[2 + 2 * row].record()
            acct.account(row, step, packet)  # separate: 7 device operations, fused: ONE launch; no host read or copy either way
            while pending and boundary >= pending[0]:
                snap["t"].append(boundary)
                snap["depth"].append(cp.asnumpy(route.depth_m))
                snap["mobile"].append(cp.asnumpy(ctx.M1).reshape(ny, nx, nc))
                ctx.stats["d2h_driver_snapshot_bytes"] = ctx.stats.get("d2h_driver_snapshot_bytes", 0) + int(
                    snap["depth"][-1].nbytes + snap["mobile"][-1].nbytes)
                ctx.stats["d2h_driver_snapshot_reads"] = ctx.stats.get("d2h_driver_snapshot_reads", 0) + 2
                pending.pop(0)
        if row == 0:
            cp.cuda.Stream.null.synchronize()
            first_step_s = time.perf_counter() - t_loop0  # includes any lazy first-launch cost
        prev_depth = route.depth_m
        storm = replace(step.state, t_s=boundary)
        last = step
        t = boundary
        if (row + 1) % args.check_every_steps == 0:
            checked = _check(ctx, checked, row + 1)
        if t >= next_progress:
            p0 = time.perf_counter()
            print(f"[{label}] t = {t:.0f} s / {boundaries[-1]:.0f} s, wall {p0 - t_loop0:.1f} s", file=sys.stderr, flush=True)
            next_progress += args.progress_every_s
            progress_s += time.perf_counter() - p0
    cp.cuda.Stream.null.synchronize()
    loop_s = time.perf_counter() - t_loop0 - progress_s
    _check(ctx, checked, steps)
    out = {"loop_s": loop_s, "progress_s": progress_s, "first_step_s": first_step_s, "water": acct.water, "cum": acct.cum, "water_accounting": acct.summary(), "last": last,
           "storm": storm, "snap": snap, "events": events}
    return out


def _check(ctx: CudaLegacyContext, checked: int, upto: int) -> int:
    ctx.check_flags(upto)  # raises CudaLegacyError (and poisons) on any device flag
    if upto > checked:
        D._check_identity(ctx.host_ledger[checked:upto])  # the CPU guard, per step and class, on the rows just read
    return upto


def _event_split(cp, events, steps: int) -> dict[str, float]:
    """CUDA-event segments of the measured loop (read after synchronisation). Honest labels: the segment from the END of the previous
    sediment step to the END of the hydrology step is NOT isolated hydrology: it also contains the PREVIOUS step's water accounting
    launches (enqueued after the sediment-end event; 7 operations with `--water-accounting separate`, one launch with `fused`), the
    sediment flag/ledger slice reads at the check cadence, snapshot downloads and progress output of the previous iteration, plus the
    packet read's host wait. Only the second segment (hydrology end -> sediment launches end, device timeline) is sediment work, and it
    includes the post-infiltration law-depth launch when selected (the water accounting is enqueued AFTER the sediment-end event, so it is
    in the first segment of the NEXT iteration; the last step's accounting falls in no segment)."""
    gap_ms = sed_ms = 0.0
    for row in range(steps):
        gap_ms += cp.cuda.get_elapsed_time(events[2 * row], events[1 + 2 * row])
        sed_ms += cp.cuda.get_elapsed_time(events[1 + 2 * row], events[2 + 2 * row])
    return {"previous_sediment_end_to_hydrology_end_s": gap_ms / 1000.0,
            "hydrology_end_to_sediment_end_s": sed_ms / 1000.0,
            "label_note": "the first segment contains hydrology + water accounting + host reads/snapshots/progress; it is not an "
                          "isolated hydrology time. Use separate hydrology-only runs for an isolated hydrology timing."}


def _output_pins(out_dir: Path, names: tuple[str, ...]) -> dict[str, Any]:
    """SHA-256 and byte count of the CLOSED data archives (streamed, once, after they are written), plus their zip member names. The
    summary cannot pin itself (it is written after this and hashing it would be self-referential); a reader verifies the archives
    against these pins and treats the summary as the receipt. No CRC re-read of the array data is made."""
    t0 = time.perf_counter()
    files: dict[str, Any] = {}
    for name in names:
        path = out_dir / name
        if not path.is_file():
            continue
        with zipfile.ZipFile(path) as z:  # directory only: cheap, no decompression
            members = sorted(i.filename for i in z.infolist())
        files[name] = {"sha256": D._sha256_stream(path), "bytes": int(path.stat().st_size), "members": members}
    return {"files": files, "algorithm": "sha256 of the closed file, streamed after writing", "summary_file": "not pinned (self-reference)",
            "seconds": time.perf_counter() - t0}


def _execute(args: argparse.Namespace, out_dir: Path, roots: dict[str, Path]) -> dict[str, Any]:
    t_wall0 = time.perf_counter()
    cp = _cupy()  # explicit CudaUnavailableError: there is no CPU fallback
    from maple_syrup import hydrology_cuda as hc

    device = device_record(cp)
    modules_before = module_digests()
    syrup_prov = _syrup_provenance()
    t0 = time.perf_counter()
    case = legacy_case_for(args.case_kind, args.case, allow_maple_source_change=args.allow_maple_source_change,
                           hash_only_tile_verify=args.hash_only_tile_verify, end_s=args.end_s, applied_rainfall=args.applied_rainfall)
    case_s = time.perf_counter() - t0
    dep = case.verified.maple_dependency
    roots.update({"verified MAPLE source": Path(dep.source_root), "verified MAPLE package": Path(dep.package_dir)})
    mah_root = (case.verified.report.get("mahleran") or {}).get("root")
    if mah_root:
        roots["verified MAHLERAN"] = Path(mah_root)
    for i, p in enumerate(case.pin_paths):
        roots[f"bound input {i}"] = Path(p)
    D.check_output(out_dir, roots, require_absent=False)
    pins_before = {str(p): D._sha256_stream(Path(p)) for p in case.pin_paths}
    digests_before = _source_digests(Path(syrup_prov["package_dir"]), dep.package_dir)
    end = float(args.end_s) if args.end_s is not None else case.end_s
    D._whole_seconds("end time", end, allow_zero=False)
    boundaries = plan_boundaries(case.schedule, 0.0, end, 1.0)
    if not np.array_equal(np.diff(np.concatenate(([0.0], boundaries))), np.ones(boundaries.size)):
        raise D.DriverError("the legacy replay needs one-second boundaries (dt = 1 s)")
    snapshots = D._parse_times(args.snapshot_times, end)
    ny, nx = case.shape
    nc = case.n_classes
    steps = int(boundaries.size)
    control = replace(D._control(args, case.default_bisection_iterations), implementation="cuda").validated()

    t0 = time.perf_counter()
    network = N.native_network(case.graph)
    network_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    physics_ctx = prepare_legacy_physics(case.sediment, case.grid, case.vegetation, case.holdings_kg)
    physics_prep_s = time.perf_counter() - t0
    gpu_budget = int(args.gpu_memory_gib * 2**30) if args.gpu_memory_gib is not None else None
    t0 = time.perf_counter()
    graph_gpu, column_gpu, field_gpu, storm0 = gpu_hydrology_inputs(case)
    hydro_inputs_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    hctx = hc.prepare_cuda_hydrology(graph_gpu, column_gpu, mode=args.cuda_mode)
    newton_load = hc.load_newton_kernels() if control.root_solver == "newton" else None
    hydro_prep_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    ctx = CudaLegacyContext(network, physics_ctx, case.graph, limits=N.walk_limits(network.dx_m), dt=1.0, n_steps=steps,
                            memory_budget_bytes=gpu_budget, cn_mode=args.cn_mode, record_strategy=args.record_strategy)
    context_s = time.perf_counter() - t0
    pool = cp.get_default_memory_pool()

    warm = None
    if args.warmup_s > 0.0:
        warm_end = min(float(args.warmup_s), end)
        warm_b = plan_boundaries(case.schedule, 0.0, warm_end, 1.0)
        w = _loop(cp, case, args, ctx, hctx, hc, control, field_gpu, storm0, warm_b, [], False, "warmup")
        warm = {"model_s": warm_end, "loop_s": w["loop_s"], "first_step_s": w["first_step_s"],
                "water_accounting_setup_s": w["water_accounting"]["setup_s"], "water_accounting_compile_s": w["water_accounting"]["compile_s"],
                "water_accounting_compile_cached": w["water_accounting"]["compile_cached"]}
        del w
        ctx.reset()
    r = _loop(cp, case, args, ctx, hctx, hc, control, field_gpu, storm0, boundaries, snapshots, True, "run")
    split = _event_split(cp, r["events"], steps)

    # ---- everything below is publication work (outside the timed loop) ----
    ledger = ctx.host_ledger[:steps].copy()
    residual, worst = D._check_identity(ledger)
    maps = ctx.download_maps()
    water = cp.asnumpy(r["water"])
    cw = {k: cp.asnumpy(v) for k, v in r["cum"].items()}
    last, storm = r["last"], r["storm"]
    final_depth = cp.asnumpy(last.route.depth_m)
    final_soil = cp.asnumpy(storm.soil_water_m)
    final_discharge = cp.asnumpy(storm.discharge_m2_s)
    final_velocity = cp.asnumpy(last.route.velocity_m_s)
    final_arrays = (water, *cw.values(), final_depth, final_soil, final_discharge, final_velocity)
    ctx.stats["d2h_driver_final_water_bytes"] = int(sum(a.nbytes for a in final_arrays))  # 4 cumulative maps, 4 final fields, series
    ctx.stats["d2h_driver_final_water_reads"] = len(final_arrays)
    ctx.stats["d2h_driver_initial_volume_bytes"] = int(2 * np.prod(case.shape) * 8)  # storm0 depth and soil sums (two reads)
    walk_counts, regime_counts = ctx.walk_tallies().copy(), ctx.regime_tallies().copy()
    area = case.cell_area_m2
    volumes = {
        "surface_initial": float(cp.asnumpy(storm0.depth_m).sum()) * area, "soil_initial": float(cp.asnumpy(storm0.soil_water_m).sum()) * area,
        "rain": float(cw["rain"].sum()) * area, "intake": float(cw["intake"].sum()) * area,
        "saturation_return": float(cw["saturation_return"].sum()) * area, "drainage": float(cw["drainage"].sum()) * area,
        "export": float(water[:, 1].sum()), "surface_final": float(final_depth.sum()) * area, "soil_final": float(final_soil.sum()) * area}
    rain_expected = case.schedule.depth_m(0.0, end) * float(case.rainfall_scale.sum()) * area
    water_budget = D.water_closure(volumes, ny, nx, steps, rain_expected)
    digests_after = _source_digests(Path(syrup_prov["package_dir"]), dep.package_dir)
    if digests_after != digests_before:
        raise D.DriverError("source changed during the run; nothing published")
    pins_after = {str(p): D._sha256_stream(Path(p)) for p in case.pin_paths}
    if pins_after != pins_before:
        raise D.DriverError(f"bound input artifacts changed during the run: {sorted(k for k in pins_before if pins_before[k] != pins_after.get(k))}")
    modules_after = module_digests()
    if modules_after != modules_before:
        raise D.DriverError("a driver module file changed during the run; nothing published")
    tiles = {"performed": False, "note": "persisted Chastre tiles were hashed by the case verification only; use --hash-tiles-after-run"}
    if args.case_kind == "chastre" and args.hash_tiles_after_run:
        after = case.verified.case.persisted_digest()
        if after != case.verified.case.bound_tiles_digest:
            raise D.DriverError("persisted Chastre tiles changed during the run; nothing published")
        tiles = {"performed": True, "digest": after, "equals_bound_digest": True}
    elif args.case_kind != "chastre":
        tiles = {"performed": False, "note": "no tile bed in this case kind"}

    cols = {name: i for i, name in enumerate(LEDGER_COLUMNS)}
    per_step_kg = [c for c in LEDGER_COLUMNS if LEDGER_UNITS[c] == "kg per step"]
    totals = {name: ledger[:, cols[name]].sum(axis=0).tolist() for name in per_step_kg}
    mobile = maps["mobile"]
    term = network.terminal
    terminal_diag = {"n_terminal_storage_cells": int(network.terminal_idx.size),
                     "final_mobile_in_terminal_storage_by_class_kg": mobile[term].sum(axis=0).tolist(),
                     "cumulative_deposition_in_terminal_storage_by_class_kg": maps["cum_dep"][term].sum(axis=0).tolist(),
                     "final_mobile_total_by_class_kg": mobile.sum(axis=0).tolist()}
    flux_series = ledger[:, cols["outlet_flux_kg_s"]].sum(axis=1)
    out_dir.mkdir(parents=True, exist_ok=False)
    np.savez(out_dir / "legacy_ledger.npz", t_s=boundaries, ledger=ledger, columns=np.array(LEDGER_COLUMNS),
             column_units=np.array([LEDGER_UNITS[c] for c in LEDGER_COLUMNS]), walk_counts=walk_counts,
             walk_count_names=np.array(N.COUNT_NAMES), regime_counts=regime_counts, regime_names=np.array(list(REGIME_CODES)),
             water_outlet_m3_s=water[:, 0], water_export_m3=water[:, 1], water_budget_residual_m3=water[:, 2],
             identity_residual_kg=residual, cumulative_detachment_kg=maps["cum_det"].reshape(ny, nx, nc),
             cumulative_deposition_kg=maps["cum_dep"].reshape(ny, nx, nc),
             cumulative_clipping_source_kg=maps["cum_clip"].reshape(ny, nx, nc), final_mobile_kg=mobile.reshape(ny, nx, nc),
             final_depth_m=final_depth, final_soil_water_m=final_soil, final_discharge_m2_s=final_discharge,
             final_velocity_m_s=final_velocity, cumulative_rain_m=cw["rain"], cumulative_intake_m=cw["intake"],
             cumulative_saturation_return_m=cw["saturation_return"], cumulative_drainage_m=cw["drainage"],
             active=network.active.reshape(ny, nx), terminal_storage=network.terminal.reshape(ny, nx),
             outlet=network.outlet.reshape(ny, nx))
    snap = r["snap"]
    if snap["t"]:
        np.savez(out_dir / "legacy_snapshots.npz", t_s=np.array(snap["t"]), depth_m=np.stack(snap["depth"]),
                 mobile_kg=np.stack(snap["mobile"]))
    output_pins = _output_pins(out_dir, ("legacy_ledger.npz", "legacy_snapshots.npz"))  # after the files are closed, outside the loop
    law_depth_record = {
        "selected": args.depth_time_level,
        "meaning": {"previous": "the previous step's routed depth (accepted legacy-replay default)",
                    "current": "the new routed depth (diagnostic)",
                    "post_infiltration": "NATIVE original infilt.for d(1): hpre = max(h_old - max(intake - rain, 0), 0), complete "
                                         "branch = 0, saturation return excluded (computed on the device)"}[args.depth_time_level],
        "native_original_option": args.depth_time_level == "post_infiltration", "velocity": "the new step's velocity (all modes)"}
    limitations = [*_LIMITATIONS, f"wet-law depth = '{args.depth_time_level}'"]
    pool_stats = {"pool_used_bytes": int(pool.used_bytes()), "pool_total_bytes": int(pool.total_bytes())}
    summary = {
        "status": "native-walk legacy replay on the GPU, benchmark only (see limitations)", "limitations": limitations,
        "backend": "cuda", "depth_time_level": args.depth_time_level, "law_depth": law_depth_record, "case_kind": args.case_kind,
        "steps": steps, "dt_s": 1.0, "end_s": end, "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "ledger_columns": list(LEDGER_COLUMNS), "ledger_units": dict(LEDGER_UNITS), "totals_by_class": totals,
        "totals_units": "kg (sum over steps of the per-step masses)", "totals": {k: float(np.sum(v)) for k, v in totals.items()},
        "peak_outlet_flux_kg_s": float(flux_series.max()) if flux_series.size else 0.0,
        "final_mobile_by_class_kg": mobile.sum(axis=0).tolist(), "terminal_storage": terminal_diag,
        "walk_tallies": dict(zip(N.COUNT_NAMES, walk_counts.sum(axis=0).tolist(), strict=True)),
        "wet_regime_cell_class_steps": dict(zip(REGIME_CODES, regime_counts.sum(axis=0).tolist(), strict=True)),
        "identity": {"rule": "new - old - (pickup - deposition_active) + cn_export - clip, per step and class",
                     "worst_relative_residual": worst, "guard_rtol": D.IDENTITY_RTOL,
                     "max_abs_residual_kg": float(np.abs(residual).max()) if residual.size else 0.0},
        "water": {"peak_outlet_m3_s": float(water[:, 0].max()), "export_m3": float(water[:, 1].sum()),
                  "per_step_budget_residual_m3": {"max_abs": float(np.abs(water[:, 2]).max()), "sum": float(water[:, 2].sum()),
                                                  "note": "reported from the routing step"},
                  "whole_storm_budget": water_budget},
        "onset": {"first_positive_pickup_s": D.first_positive(ledger[:, cols["pickup_kg"]], boundaries),
                  "first_positive_deposition_s": D.first_positive(ledger[:, cols["deposition_active_kg"]], boundaries),
                  "wet_law_cell_class_steps": int(sum(regime_counts.sum(axis=0)[c] for name, c in REGIME_CODES.items()
                                                      if name not in ("dry", "wet_no_law"))),
                  "note": "model times (end of the step) of the first step with a positive class-summed value; None = never"},
        "network": network.summary(),
        "walk_tables": {"records": ctx.tables.n_records, "max_path": ctx.tables.max_path, "cap": ctx.tables.cap,
                        "ring_records": int(ctx.tables.ring_rec.size), "inactive_targets": int(ctx.tables.inactive_targets.size),
                        "table_bytes": ctx.tables.nbytes(),
                        "gather": "static source/target CSR; thread per (target, record class) sums record values in CPU source order",
                        "record_classes": ctx.ne, "physical_classes": nc,
                        "record_values_bytes_allocated": int(ctx.values.nbytes)},
        "maps_total_kg": {"cumulative_detachment_kg": maps["cum_det"].sum(axis=0).tolist(),
                          "cumulative_deposition_kg": maps["cum_dep"].sum(axis=0).tolist(),
                          "cumulative_clipping_source_kg": maps["cum_clip"].sum(axis=0).tolist()},
        "artifact_pins": {"checked_unchanged_before_publish": True, "sha256": pins_before}, "chastre_tiles": tiles,
        "output_pins": output_pins,
        "qualification": {
            "case_kind": args.case_kind, "sediment_bound_rtol_atol": [2.0e-11, 1.0e-14], "water_bound_rtol_atol": [2.0e-12, 1.0e-14],
            "identity_guard_rtol": D.IDENTITY_RTOL,
            "statement": "MAHLERAN legacy replay (fixed composition, unlimited supply, explicit clipping source, no MAPLE bed): not a "
                         "conservation claim; the bounds are those a CPU/GPU comparison of THIS case is judged against, unchanged"},
        "gpu": {"device": device, "kernels": kernel_provenance(nc, ctx.ne), "record_strategy": ctx.record_info,
                "hydrology_kernels": hc.kernel_provenance(), "water_accounting": r["water_accounting"],
                "hydrology_context": hctx.summary(), "cn_mode": ctx.cn_mode, "n_levels": ctx.n_levels,
                "max_level_width": ctx.max_level_width,
                "memory": {**ctx.memory, "estimate_bytes": ctx.estimate, **pool_stats},
                "transfers": {**ctx.stats, "scope": "measured run only (counters restart at reset; static uploads are context-lifetime)",
                              "before_reset_warmup": getattr(ctx, "stats_before_reset", None),
                              "hydrology_packet_bytes_per_step": hc.PACKET_WORDS * 8,
                              "flag_check_every_steps": args.check_every_steps,
                              "hydrology_static_transfers": "the hydrology context's own static upload and the graph/column host "
                                                            "builds (hydrology_inputs_s, hydrology_prepare_s) are NOT included in "
                                                            "these counters",
                              "note": "per step: the hydrology packet (144 bytes) plus, every flag_check_every_steps, small sediment "
                                      "flag/ledger/tally slices; snapshots (depth, mobile) at the requested times, one set of final "
                                      "water maps and the volume sums once (d2h_driver_* counters); sediment maps once at the end"},
                "launches_per_step_sediment": (ctx.stats["launches"] // max(ctx.stats["steps"], 1)) if ctx.stats["steps"] else None,
                "launches_scope": "ctx.stats launches and launches_per_step_sediment count SEDIMENT launches only; the driver water "
                                  "accounting is in gpu.water_accounting"},
        "performance": {
            "loop_wall_s_excluding_progress": r["loop_s"], "event_split_s": split, "first_step_s": r["first_step_s"], "warmup": warm,
            "case_verify_and_adapt_s": case_s, "network_build_s": network_s, "physics_prepare_s": physics_prep_s,
            "hydrology_inputs_s": hydro_inputs_s, "hydrology_prepare_s": hydro_prep_s,
            "water_accounting_setup_s": r["water_accounting"]["setup_s"], "water_accounting_compile_s": r["water_accounting"]["compile_s"],
            "water_accounting_compile_cached": r["water_accounting"]["compile_cached"],
            "newton_kernel_load_s": None if newton_load is None else newton_load["seconds"],
            "sediment_context_build_s": context_s, "sediment_kernel_compile_and_tables_s": ctx.compile_s,
            "whole_process_wall_s": time.perf_counter() - t_wall0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "note": "cold = compile/context/preparation; steady state = the loop after --warmup-s; timers exclude progress output"},
        "hydrology": {"root_solver": control.root_solver, "bisection_iterations": control.bisection_iterations,
                      "newton_max_iterations": control.newton_max_iterations},
        "case": case.record,
        "provenance": {"maple_syrup": syrup_prov, "maple": case.verified.maple_provenance,
                       "source_digests_before": digests_before, "source_digests_after": digests_after,
                       "graph_input_sha256": case.graph.input_sha256, "environment": environment_record(),
                       "module_sha256": modules_after},
    }
    (out_dir / "legacy_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_gpu_controls(args)
    out = args.output.resolve()
    roots = D.protected_roots(args.case.resolve())
    D.check_output(out, roots, require_absent=True)
    partial = out.with_name(out.name + ".partial")
    failed = out.with_name(out.name + ".FAILED")
    try:
        summary = _execute(args, partial, roots)
    except BaseException as exc:
        try:
            D.check_output(out, roots, require_absent=False)
        except D.DriverError:
            raise exc from None
        partial.mkdir(parents=True, exist_ok=True)
        (partial / "FAILED.json").write_text(json.dumps(
            {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
             "args": {k: str(v) for k, v in vars(args).items()}}, indent=2) + "\n")
        os.replace(partial, failed)
        raise
    os.replace(partial, out)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except (D.DriverError, CudaLegacyError, CudaUnavailableError) as exc:
        print(f"legacy GPU driver failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report and set the exit status
        print(f"legacy GPU driver failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: summary[k] for k in ("totals", "totals_units", "terminal_storage", "walk_tallies", "identity", "performance")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
