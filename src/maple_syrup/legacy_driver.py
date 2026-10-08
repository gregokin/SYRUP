"""Reusable native-walk MAHLERAN LEGACY sediment benchmark driver: Plot 1 (small), RFID_2014 and Chastre (large), CPU/Numba.

    python -m maple_syrup.legacy_driver --case-kind chastre --case outputs/chastre/case_v2 --output <NEW> \\
        --end-s 120 --snapshot-times 60,120 --warmup-s 30

Run it as a module (`python -m maple_syrup.legacy_driver`); no console script is registered. The accepted Plot 1 default
(`maple_syrup.legacy_experiment`, `maple-syrup-legacy`) is NOT modified and does not need this module.

SCIENTIFIC STATUS. Frozen terrain and routing, fixed 1 s step, no splash, no evaporation, no dry reset, the window ends while
flow is still present. Fixed composition (the verified initial active layer), UNLIMITED supply, explicit artificial clipping
source, ring / inactive-cell deposition are diagnostics (not debited), terminal pits keep their mobile mass, no MAPLE bed is read
after case verification and none is written. It is NOT a conservative event, restart, or wind-handoff model, and its legacy
behaviour is not MAPLE authority. Chastre has zero outlets, so its export is identically zero; judge it by the per-cell maps
and per-class ledgers, not by export.

One step (same order and time levels as `legacy_experiment`): rainfall -> prepared Numba hydrology (bisection or Newton) -> wet
laws with the NEW velocity and, per `--depth-time-level`, the PREVIOUS step's depth (default, the accepted behaviour), the NEW
depth (`current`), or the ORIGINAL's post-infiltration `d(1)` (`post_infiltration`, native option, never the default) ->
detachment rate, virtual velocity (law value, else 0.9 recession memory) -> source walk (`legacy_native`) -> Crank-Nicolson pool
(the accepted kernel) -> compiled validation of the finished step -> reduction. `post_infiltration` reproduces `infilt.for`'s
`d(1)`, which is the existing hydrology `hpre = max(h_old - max(intake - rain, 0), 0)` with the complete-infiltration branch
(`intake >= h_old + rain`, tested first) forced to exactly 0; saturation return is excess and is NOT added to it
(`post_infiltration_depth`). Buffers are allocated once. Progress goes to stderr outside the model timers.

Outputs (a NEW directory, assembled as `<out>.partial` and renamed only on success; failure leaves `<out>.FAILED` with
FAILED.json): legacy_ledger.npz (per-step per-class ledger columns, walk tallies, per-step wet regime counts, water series,
cumulative per-cell detachment/deposition/clipping maps, final mobile/depth/soil water/discharge/velocity, masks, identity
residual), legacy_snapshots.npz (optional), legacy_summary.json (units, totals, guards, pins, provenance).

Output protection: the output and its `.partial` / `.FAILED` siblings are resolved (symlinks included) and must not equal, lie
inside or contain the case directory, the project src/tests/benchmarks/cases trees, the MAPLE dependency, the MAHLERAN
reference tree or any bound input directory; this is checked before anything is written, the FAILED marker included.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup import legacy_native as N
from maple_syrup import legacy_transport as L
from maple_syrup.column_experiment import _source_digests, _syrup_provenance
from maple_syrup.legacy_case import LegacyCase, legacy_case_for
from maple_syrup.legacy_native_numba import (
    LEDGER_COLUMNS,
    LEDGER_UNITS,
    StepEngine,
    WetLawRunner,
    get_kernels,
)
from maple_syrup.legacy_physics_numba import prepare_legacy_physics
from maple_syrup.provenance import environment_record
from maple_syrup.routing import RoutingStepRejected
from maple_syrup.sediment_physics import REGIME_CODES
from maple_syrup.storm import StormControl, StormError, plan_boundaries

__all__ = ["DEPTH_TIME_LEVELS", "LEDGER_COLUMNS", "build_parser", "main", "module_digests", "post_infiltration_depth", "run",
           "select_law_depth", "validate_controls"]

DEPTH_TIME_LEVELS = ("previous", "current", "post_infiltration")
_PROVENANCE_MODULES = ("maple_syrup.legacy_native", "maple_syrup.legacy_native_numba", "maple_syrup.legacy_case",
                       "maple_syrup.legacy_driver")

IDENTITY_RTOL = 1.0e-10  # predeclared per-step guard on the pool identity, relative to the sum of |terms|
MAX_SNAPSHOTS = 16
_LIMITATIONS = (
    "fixed composition (verified initial active layer), UNLIMITED supply, explicit artificial clipping source",
    "ring and inactive-cell deposition are walk diagnostics, not debited from the active pools; terminal pits retain mobile mass",
    "no splash, no ET, no dry reset, fixed 1 s step, frozen elevation and routing, window ends while flow may persist",
    "no MAPLE bed mutation, no conservation claim, no restart, no wind handoff, CPU only",
    "the whole-storm water budget uses the unchanged MAPLE-derived volume bound; it says nothing about sediment conservation",
)


def first_positive(series_by_class: np.ndarray, boundaries: np.ndarray):
    """Model time (step end) of the first step whose class-summed value is > 0, or None."""
    total = np.asarray(series_by_class).sum(axis=1)
    hits = np.flatnonzero(total > 0.0)
    return float(boundaries[hits[0]]) if hits.size else None


class DriverError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m maple_syrup.legacy_driver", description=__doc__.split("\n\n")[0])
    p.add_argument("--case-kind", choices=("plot1", "rfid", "chastre"), required=True)
    p.add_argument("--case", "--case-dir", dest="case", type=Path, required=True, help="verified imported case directory")
    p.add_argument("--output", "--output-dir", dest="output", type=Path, required=True, help="NEW output directory")
    p.add_argument("--end-s", type=float, default=None, help="default: the case storm length; whole seconds")
    p.add_argument("--snapshot-times", default="", help=f"comma-separated model seconds (at most {MAX_SNAPSHOTS}); depth and mobile maps")
    p.add_argument("--warmup-s", type=float, default=0.0, help="whole model seconds run first on throwaway state in-process")
    p.add_argument("--progress-every-s", type=float, default=60.0, help="stderr progress cadence in model seconds; 0 = off")
    p.add_argument("--root-solver", choices=("bisection", "newton"), default="bisection")
    p.add_argument("--bisection-iterations", type=int, default=None, help="default 40 (Plot 1) / 64 (RFID, Chastre)")
    p.add_argument("--newton-max-iterations", type=int, default=None)
    p.add_argument("--depth-time-level", choices=DEPTH_TIME_LEVELS, default="previous",
                   help="wet-law depth: previous (default, accepted), current (new depth), post_infiltration (the original's d(1))")
    p.add_argument("--source-order", choices=("index", "legacy"), default="index")
    p.add_argument("--legacy-depos-erase", action="store_true",
                   help="reproduce the order-dependent erasure of deposition at wet no-rain cells (needs --source-order legacy)")
    p.add_argument("--applied-rainfall", type=Path, default=None, help="Plot 1 only: applied-rainfall CSV (as legacy_experiment)")
    p.add_argument("--allow-maple-source-change", action="store_true")
    p.add_argument("--hash-only-tile-verify", action="store_true", help="Chastre: reload tile 0 only during verification")
    p.add_argument("--hash-tiles-after-run", action="store_true",
                   help="Chastre: stream-hash every persisted tile after the run (reads the whole ~33 GB bed state once)")
    p.add_argument("--max-memory-gib", type=float, default=32.0, help="refuse if the estimated engine buffers exceed this")
    p.add_argument("--allow-python-kernels", action="store_true", help="TESTS ONLY: pure-Python kernels (very slow)")
    return p


def _parse_times(text: str, end: float) -> list[float]:
    if not text.strip():
        return []
    try:
        times = sorted({float(v) for v in text.split(",") if v.strip()})
    except ValueError as exc:
        raise DriverError(f"--snapshot-times is not a list of numbers: {exc}") from None
    if len(times) > MAX_SNAPSHOTS or any(not math.isfinite(t) or t <= 0.0 or t > end for t in times):
        raise DriverError(f"--snapshot-times needs at most {MAX_SNAPSHOTS} finite values in (0, end]")
    return times


def _whole_seconds(name: str, value: float, *, allow_zero: bool) -> None:
    if not math.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero) or not float(value).is_integer():
        raise DriverError(f"{name} must be a finite whole number of seconds {'>= 0' if allow_zero else '> 0'}, got {value!r}")


def _control(args: argparse.Namespace, default_bisection: int) -> StormControl:
    kwargs: dict[str, Any] = {"max_dt_s": 1.0, "min_dt_s": 1.0, "max_retries": 1, "implementation": "numba",
                              "root_solver": args.root_solver,
                              "bisection_iterations": default_bisection if args.bisection_iterations is None
                              else args.bisection_iterations}  # an explicit 0 reaches the real validation
    if args.newton_max_iterations is not None:
        kwargs["newton_max_iterations"] = args.newton_max_iterations
    try:
        return StormControl(**kwargs).validated()
    except StormError as exc:
        raise DriverError(f"invalid hydrology control: {exc}") from None


def validate_controls(args: argparse.Namespace) -> None:
    """Every scalar control, before any case adaptation, allocation or write. Explicit zeros are validated, never replaced."""
    if not (math.isfinite(args.max_memory_gib) and args.max_memory_gib > 0.0):
        raise DriverError(f"--max-memory-gib must be finite and > 0, got {args.max_memory_gib!r}")
    if not (math.isfinite(args.progress_every_s) and args.progress_every_s >= 0.0):
        raise DriverError(f"--progress-every-s must be finite and >= 0, got {args.progress_every_s!r}")
    _whole_seconds("--warmup-s", args.warmup_s, allow_zero=True)
    if args.end_s is not None:
        _whole_seconds("--end-s", args.end_s, allow_zero=False)
    if args.legacy_depos_erase and args.source_order != "legacy":
        raise DriverError("--legacy-depos-erase needs --source-order legacy")
    _control(args, 40)
    _parse_times(args.snapshot_times, args.end_s if args.end_s is not None else math.inf)


# --- output protection -------------------------------------------------------------------------------------------------
def _real(path: Path | str) -> Path:
    return Path(os.path.realpath(path))


def _maple_roots() -> list[Path]:
    """The MAPLE dependency source and package directories (best effort; tests monkeypatch this)."""
    try:
        from maple_syrup.dependency import resolve_maple_dependency

        dep = resolve_maple_dependency()
        return [Path(dep.source_root), Path(dep.package_dir)]
    except Exception:  # noqa: BLE001 - the verified dependency is added again after the case is loaded
        return []


def protected_roots(case_dir: Path) -> dict[str, Path]:
    repo = Path(__file__).resolve().parents[2]
    roots = {f"project {name}": repo / name for name in ("src", "tests", "benchmarks", "cases")}
    roots["case"] = case_dir
    roots["MAHLERAN reference"] = Path(os.environ.get("MAPLE_SYRUP_MAHLERAN_ROOT", "/home/okin/MAHLERAN"))
    for i, root in enumerate(_maple_roots()):
        roots[f"MAPLE dependency {i}"] = root
    return roots


def check_output(out: Path, roots: dict[str, Path], *, require_absent: bool) -> None:
    """Refuse an output (or its `.partial` / `.FAILED` sibling) that equals, lies inside or contains a protected tree, after
    resolving symlinks, and (initially) one that already exists. Nothing is created."""
    for candidate in (out, out.with_name(out.name + ".partial"), out.with_name(out.name + ".FAILED")):
        if require_absent and os.path.lexists(candidate):
            raise DriverError(f"refusing to write: {candidate} exists")
        real = _real(candidate)
        for label, root in roots.items():
            r = _real(root)
            if real == r or real.is_relative_to(r) or r.is_relative_to(real):
                raise DriverError(f"refusing to write {candidate}: it is, lies inside or contains the {label} tree {r}")


def _peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _sha256_stream(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def module_digests() -> dict[str, str]:
    """SHA-256 of the actual source file of each provenance module. The driver itself is hashed from its own `__file__`: under
    `python -m` it is `__main__`, not `maple_syrup.legacy_driver`, so `sys.modules` cannot be used. Others come from their import
    spec. Raises (before any model work) if a file cannot be found."""
    out = {}
    for name in _PROVENANCE_MODULES:
        if name == "maple_syrup.legacy_driver":
            path = Path(__file__).resolve()
        else:
            spec = importlib.util.find_spec(name)
            if spec is None or not spec.origin or not Path(spec.origin).is_file():
                raise DriverError(f"cannot locate the source file of {name} for provenance")
            path = Path(spec.origin)
        out[name] = _sha256_stream(path)
    return out


def post_infiltration_depth(old_depth: np.ndarray, rain_m: np.ndarray, intake_m: np.ndarray, active: np.ndarray,
                            out: np.ndarray | None = None, scratch: np.ndarray | None = None) -> np.ndarray:
    """The ORIGINAL `infilt.for` surface depth `d(1)` seen by the sediment laws, from the hydrology's own quantities:

        hpre = max(h_old - max(intake - rain, 0), 0);   hpre = 0 where  intake >= h_old + rain  (complete branch, tested first)

    i.e. the existing `storm.coupled_step` / prepared-hydrology `hpre` (infilt.for 105-148: partial infiltration removes only the
    net intake from the OLD depth; the complete branch sets d(1) = 0 explicitly so a roundoff leftover cannot create a wet cell).
    Saturation return is excess and is NOT added; the new routed depth and the column depth are not used. `old_depth` is the
    state depth BEFORE the step; `rain_m` and `intake_m` are the step's `column.rain_m` / `column.intake_m`. Inputs are validated
    (host float64 `(ny, nx)`, finite, non-negative) and never modified; `out` / `scratch` are reusable buffers."""
    arrays = {"old_depth": old_depth, "rain_m": rain_m, "intake_m": intake_m}
    for name, a in arrays.items():
        if type(a) is not np.ndarray or a.dtype != np.float64 or a.ndim != 2:
            raise DriverError(f"{name} must be a host float64 (ny, nx) numpy array")
        if a.shape != old_depth.shape:
            raise DriverError(f"{name} shape {a.shape} differs from the depth shape {old_depth.shape}")
        if not np.all(np.isfinite(a)) or np.any(a < 0.0):
            raise DriverError(f"{name} must be finite and >= 0")
    active_in = active
    active = np.asarray(active, dtype=bool)
    if active.shape != old_depth.shape:
        raise DriverError("the active mask shape differs from the depth shape")
    # Validate EVERY buffer before anything is written: exact writable C-contiguous float64 ndarrays of the depth shape that share
    # no memory (views included, read-only inputs included) with any input or with each other.
    buffers = {}
    for name, b in (("out", out), ("scratch", scratch)):
        if b is None:
            continue
        if type(b) is not np.ndarray or b.dtype != np.float64 or b.shape != old_depth.shape:
            raise DriverError(f"{name} must be a float64 ndarray of shape {old_depth.shape}")
        if not b.flags.c_contiguous or not b.flags.writeable:
            raise DriverError(f"{name} must be a writable C-contiguous array")
        buffers[name] = b
    inputs = {"old_depth": old_depth, "rain_m": rain_m, "intake_m": intake_m, "active": active_in}
    for name, b in buffers.items():
        for in_name, a in inputs.items():
            if isinstance(a, np.ndarray) and np.shares_memory(b, a):
                raise DriverError(f"{name} shares memory with the input {in_name}")
    if len(buffers) == 2 and np.shares_memory(buffers["out"], buffers["scratch"]):
        raise DriverError("out and scratch share memory")
    out = np.empty_like(old_depth) if out is None else out
    scratch = np.empty_like(old_depth) if scratch is None else scratch
    np.add(old_depth, rain_m, out=scratch)
    complete = active & (intake_m >= scratch)  # infilt.for 106 is tested first
    np.subtract(intake_m, rain_m, out=scratch)
    np.maximum(scratch, 0.0, out=scratch)
    np.subtract(old_depth, scratch, out=out)
    np.maximum(out, 0.0, out=out)
    out[complete] = 0.0
    return out


def select_law_depth(mode: str, previous_depth, new_depth, *, old_state_depth=None, column=None, active=None, out=None,
                     scratch=None):
    """The depth array the wet laws receive. `previous` returns the previous step's depth and `current` the new routed depth
    (the accepted behaviours, returned as the very objects passed in); `post_infiltration` computes the original's `d(1)`."""
    if mode == "previous":
        return previous_depth
    if mode == "current":
        return new_depth
    if mode == "post_infiltration":
        return post_infiltration_depth(old_state_depth, column.rain_m, column.intake_m, active, out, scratch)
    raise DriverError(f"unknown depth time level {mode!r}")


def _loop(case: LegacyCase, args: argparse.Namespace, hydrology_step, network, physics_ctx, boundaries: np.ndarray,
          snapshots: list[float], collect: bool, label: str) -> dict[str, Any]:
    ny, nx = case.shape
    nc = case.n_classes
    runner = WetLawRunner(physics_ctx)
    engine = StepEngine(network, runner, dt=1.0, compiled=not args.allow_python_kernels, source_order=args.source_order,
                        erase_on=args.legacy_depos_erase)
    steps = boundaries.size
    out: dict[str, Any] = {"engine": engine, "steps": steps}
    if collect:
        out["ledger"] = np.zeros((steps, len(LEDGER_COLUMNS), nc))
        out["walk_counts"] = np.zeros((steps, N.N_COUNTS), dtype=np.int64)
        out["regime_counts"] = np.zeros((steps, len(REGIME_CODES)), dtype=np.int64)
        out["water_q"] = np.zeros(steps)
        out["water_export"] = np.zeros(steps)
        out["budget_residual"] = np.zeros(steps)
        # preallocated per-cell cumulative water depths (m): accumulated in place, no per-step allocation
        cum_water = {k: np.zeros((ny, nx)) for k in ("rain", "intake", "saturation_return", "drainage")}
        out["cum_water"] = cum_water
        out["snap_depth"], out["snap_mobile"], out["snap_t"] = [], [], []
    storm = case.initial_storm
    prev_depth = np.array(storm.depth_m, dtype=np.float64, copy=True)  # the state depth BEFORE each step (also the old depth)
    active2d = network.active.reshape(ny, nx)
    hpre_out = hpre_scratch = None
    if args.depth_time_level == "post_infiltration":
        hpre_out, hpre_scratch = np.empty((ny, nx)), np.empty((ny, nx))
    rate = np.empty((ny, nx))
    pending = list(snapshots)
    hydro_s = progress_s = 0.0
    first = {}
    next_progress = args.progress_every_s if args.progress_every_s > 0 else math.inf
    t = 0.0
    t_loop0 = time.perf_counter()
    route = None
    for row, boundary in enumerate(boundaries.tolist()):
        dt = boundary - t
        if dt != 1.0:
            raise DriverError(f"step {row}: dt = {dt!r}; the legacy replay requires exactly 1 s steps")
        case.field.apply(case.schedule.rate_after_m_per_s(t), out=rate)
        t0 = time.perf_counter()
        try:
            hydro = hydrology_step(rate, storm, dt)
        except RoutingStepRejected as exc:
            raise DriverError(f"water step rejected at t={t}: {exc}; the legacy program has no retry") from exc
        hydro_s += time.perf_counter() - t0
        route = hydro.route
        new_depth = np.asarray(route.depth_m)
        law_depth = select_law_depth(args.depth_time_level, prev_depth, new_depth, old_state_depth=prev_depth,
                                     column=hydro.column, active=active2d, out=hpre_out, scratch=hpre_scratch)
        res = engine.step(law_depth, route.velocity_m_s, rate)
        if row == 0:
            first = {"hydrology_s": hydro_s, "physics_s": engine.timers["physics_s"],
                     "glue_walk_cn_s": engine.timers["glue_walk_cn_s"], "total_s": time.perf_counter() - t_loop0}
        if collect:
            out["ledger"][row] = res.ledger_row
            out["walk_counts"][row] = res.walk_counts
            out["regime_counts"][row] = res.regime_counts
            out["water_q"][row] = float(route.outlet_discharge_m3_s)
            out["water_export"][row] = float(route.export_m3)
            out["budget_residual"][row] = float(route.budget_residual_m3)
            col = hydro.column
            np.add(cum_water["rain"], col.rain_m, out=cum_water["rain"])
            np.add(cum_water["intake"], col.intake_m, out=cum_water["intake"])
            np.add(cum_water["saturation_return"], col.saturation_return_m, out=cum_water["saturation_return"])
            np.add(cum_water["drainage"], col.drainage_m, out=cum_water["drainage"])
            while pending and boundary >= pending[0]:
                out["snap_t"].append(boundary)
                out["snap_depth"].append(new_depth.copy())
                out["snap_mobile"].append(engine.mobile.reshape(ny, nx, nc).copy())
                pending.pop(0)
        prev_depth = np.array(new_depth, dtype=np.float64, copy=True)
        t = boundary
        storm = replace(hydro.state, t_s=t)
        if t >= next_progress:
            p0 = time.perf_counter()
            print(f"[{label}] t = {t:.0f} s / {boundaries[-1]:.0f} s, wall {p0 - t_loop0:.1f} s, mobile "
                  f"{float(engine.mobile.sum()):.6g} kg", file=sys.stderr, flush=True)
            next_progress += args.progress_every_s
            progress_s += time.perf_counter() - p0
    loop_s = time.perf_counter() - t_loop0 - progress_s
    out.update(loop_s=loop_s, hydrology_s=hydro_s, progress_s=progress_s, first_step=first, final_depth=prev_depth)
    if collect:
        out.update(final_soil=np.asarray(storm.soil_water_m), final_discharge=np.asarray(storm.discharge_m2_s),
                   final_velocity=np.array(route.velocity_m_s, copy=True))
    return out


def water_closure(v: dict[str, float], ny: int, nx: int, accepted_steps: int, rain_expected_m3: float) -> dict[str, Any]:
    """Whole-storm water guard with EXACTLY the rule of `benchmarks/hydraulic_candidates/compare_plot1.water_budget`: the
    MAPLE-derived `conservation.volume_roundoff_bound_m3(4 n_cells n_steps + 7, largest operand)` for the water, surface and soil
    residuals, and the same rule with `max(expected, accumulated)` for the rainfall integral. No new or relaxed bound. Raises
    `DriverError` (nothing published) on a non-finite or exceeding residual. This says the storm's cumulative fluxes balance the
    stored/exported water; it makes no sediment or legacy-conservation claim."""
    from maple_syrup.conservation import volume_roundoff_bound_m3

    if not all(math.isfinite(x) for x in v.values()) or not math.isfinite(rain_expected_m3):
        raise DriverError("a water budget volume is not finite")
    residual = (v["surface_final"] + v["soil_final"] + v["drainage"] + v["export"]
                - v["surface_initial"] - v["soil_initial"] - v["rain"])
    surface = v["surface_final"] - (v["surface_initial"] + v["rain"] - v["intake"] + v["saturation_return"] - v["export"])
    soil = v["soil_final"] - (v["soil_initial"] + v["intake"] - v["drainage"] - v["saturation_return"])
    n_terms = 4 * ny * nx * max(accepted_steps, 1) + 7
    bound = volume_roundoff_bound_m3(n_terms, max(abs(x) for x in v.values()))
    rain_bound = volume_roundoff_bound_m3(n_terms, max(rain_expected_m3, v["rain"]))
    checks = {"water": (residual, bound), "surface": (surface, bound), "soil": (soil, bound),
              "rainfall_integral": (v["rain"] - rain_expected_m3, rain_bound)}
    for label, (value, tol) in checks.items():
        if not abs(value) <= tol:
            raise DriverError(f"{label} budget residual {value} m3 exceeds the MAPLE-derived bound {tol} m3; nothing published")
    return {"units": "m3", **{f"{k}_m3": x for k, x in v.items()}, "rain_expected_m3": rain_expected_m3,
            "water_residual_m3": residual, "surface_residual_m3": surface, "soil_residual_m3": soil, "bound_m3": bound,
            "rainfall_bound_m3": rain_bound, "n_terms": n_terms, "closed": True,
            "rule": "conservation.volume_roundoff_bound_m3(4 n_cells n_steps + 7, largest operand), as compare_plot1.water_budget"}


def _check_identity(ledger: np.ndarray) -> tuple[np.ndarray, float]:
    c = {name: ledger[:, i] for i, name in enumerate(LEDGER_COLUMNS)}
    residual = (c["new_mobile_kg"] - c["old_mobile_kg"] - (c["pickup_kg"] - c["deposition_active_kg"])
                + c["cn_export_kg"] - c["effective_clip_source_kg"])
    scale = (np.abs(c["pickup_kg"]) + np.abs(c["deposition_active_kg"]) + np.abs(c["effective_clip_source_kg"])
             + np.abs(c["old_mobile_kg"]) + np.abs(c["new_mobile_kg"]) + np.abs(c["cn_export_kg"]))
    ratio = np.abs(residual) / np.maximum(scale, 1e-300)
    worst = float(ratio.max()) if ratio.size else 0.0
    if not np.all(np.isfinite(residual)) or worst > IDENTITY_RTOL:
        raise DriverError(f"per-step pool identity violated: worst |residual| / sum|terms| = {worst:.3e} > {IDENTITY_RTOL}")
    return residual, worst


def _execute(args: argparse.Namespace, out_dir: Path, roots: dict[str, Path]) -> dict[str, Any]:
    from maple_syrup import hydrology_numba as hn

    t_wall0 = time.perf_counter()
    if not args.allow_python_kernels:
        get_kernels(True)  # explicit error if Numba is missing; no silent fallback
    modules_before = module_digests()  # provenance is resolved BEFORE any expensive work (a failure here wastes nothing)
    syrup_prov = _syrup_provenance()
    t0 = time.perf_counter()
    case = legacy_case_for(args.case_kind, args.case, allow_maple_source_change=args.allow_maple_source_change,
                           hash_only_tile_verify=args.hash_only_tile_verify, end_s=args.end_s,
                           applied_rainfall=args.applied_rainfall)
    case_s = time.perf_counter() - t0
    # the verified dependency and bound inputs are now known: protect them too, still before anything is written
    dep = case.verified.maple_dependency
    roots.update({"verified MAPLE source": Path(dep.source_root), "verified MAPLE package": Path(dep.package_dir)})
    mah_root = (case.verified.report.get("mahleran") or {}).get("root")
    if mah_root:
        roots["verified MAHLERAN"] = Path(mah_root)
    for i, p in enumerate(case.pin_paths):
        roots[f"bound input {i}"] = Path(p)
    check_output(out_dir, roots, require_absent=False)
    pins_before = {str(p): _sha256_stream(Path(p)) for p in case.pin_paths}
    digests_before = _source_digests(Path(syrup_prov["package_dir"]), case.verified.maple_dependency.package_dir)
    end = float(args.end_s) if args.end_s is not None else case.end_s
    _whole_seconds("end time", end, allow_zero=False)
    boundaries = plan_boundaries(case.schedule, 0.0, end, 1.0)
    if not np.array_equal(np.diff(np.concatenate(([0.0], boundaries))), np.ones(boundaries.size)):
        raise DriverError("the legacy replay needs one-second boundaries (dt = 1 s); use an integer --end-s and a 1 s rain schedule")
    snapshots = _parse_times(args.snapshot_times, end)
    ny, nx = case.shape
    nc = case.n_classes
    n = ny * nx
    if snapshots and len(snapshots) * n * (nc + 1) * 8 > 4 * 2**30:
        raise DriverError("the requested snapshots would exceed 4 GiB")
    control = _control(args, case.default_bisection_iterations)
    estimate_bytes = 20 * n * nc * 8 + 3 * n * nc + 9 * n * 8 + physics_ctx_bytes_estimate(n, nc)  # 9 = 5 + 4 water maps
    if estimate_bytes > args.max_memory_gib * 2**30:
        raise DriverError(f"estimated engine buffers {estimate_bytes / 2**30:.1f} GiB exceed --max-memory-gib")
    t0 = time.perf_counter()
    network = N.native_network(case.graph)
    network_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    physics_ctx = prepare_legacy_physics(case.sediment, case.grid, case.vegetation, case.holdings_kg)
    physics_prep_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    hydro_ctx = hn.prepare_hydrology(case.graph, case.column)
    hydro_prep_s = time.perf_counter() - t0

    def hydrology_step(rate_field, storm_state, step_dt):
        return hn.prepared_coupled_step(hydro_ctx, rate_field, storm_state, step_dt, control)

    warm = None
    if args.warmup_s > 0.0:
        warm_end = min(float(args.warmup_s), end)
        warm_b = plan_boundaries(case.schedule, 0.0, warm_end, 1.0)
        w = _loop(case, args, hydrology_step, network, physics_ctx, warm_b, [], False, "warmup")
        warm = {"model_s": warm_end, "loop_s": w["loop_s"], "first_step": w["first_step"]}
        del w
    r = _loop(case, args, hydrology_step, network, physics_ctx, boundaries, snapshots, True, "run")
    engine: StepEngine = r["engine"]
    ledger = r["ledger"]
    residual, worst = _check_identity(ledger)
    area = case.cell_area_m2
    cw = r["cum_water"]
    storm_volumes = {
        "surface_initial": float(np.asarray(case.initial_storm.depth_m).sum()) * area,
        "soil_initial": float(np.asarray(case.initial_storm.soil_water_m).sum()) * area,
        "rain": float(cw["rain"].sum()) * area, "intake": float(cw["intake"].sum()) * area,
        "saturation_return": float(cw["saturation_return"].sum()) * area, "drainage": float(cw["drainage"].sum()) * area,
        "export": float(r["water_export"].sum()),
        "surface_final": float(np.asarray(r["final_depth"]).sum()) * area,
        "soil_final": float(np.asarray(r["final_soil"]).sum()) * area}
    rain_expected = case.schedule.depth_m(0.0, end) * float(case.rainfall_scale.sum()) * area  # independent of the loop
    water = water_closure(storm_volumes, ny, nx, int(boundaries.size), rain_expected)
    digests_after = _source_digests(Path(syrup_prov["package_dir"]), case.verified.maple_dependency.package_dir)
    if digests_after != digests_before:
        raise DriverError(f"source changed during the run ({digests_before} -> {digests_after}); nothing published")
    pins_after = {str(p): _sha256_stream(Path(p)) for p in case.pin_paths}
    if pins_after != pins_before:
        changed = sorted(k for k in pins_before if pins_before[k] != pins_after.get(k))
        raise DriverError(f"bound input artifacts changed during the run: {changed}; nothing published")
    tiles = {"performed": False, "note": "persisted Chastre tiles were hashed by the case verification only; "
                                         "use --hash-tiles-after-run for a post-run re-hash"}
    if args.case_kind == "chastre" and args.hash_tiles_after_run:
        after = case.verified.case.persisted_digest()
        if after != case.verified.case.bound_tiles_digest:
            raise DriverError("persisted Chastre tiles changed during the run; nothing published")
        tiles = {"performed": True, "digest": after, "equals_bound_digest": True}
    elif args.case_kind != "chastre":
        tiles = {"performed": False, "note": "no tile bed in this case kind"}

    cols = {name: i for i, name in enumerate(LEDGER_COLUMNS)}
    per_step_kg = [c for c in LEDGER_COLUMNS if LEDGER_UNITS[c] == "kg per step"]
    totals = {name: ledger[:, cols[name]].sum(axis=0).tolist() for name in per_step_kg}
    final_mobile = engine.mobile
    term = network.terminal
    maps = {"cumulative_detachment_kg": engine.cum_det, "cumulative_deposition_kg": engine.cum_dep,
            "cumulative_clipping_source_kg": engine.cum_clip}
    terminal_diag = {
        "n_terminal_storage_cells": int(network.terminal_idx.size),
        "final_mobile_in_terminal_storage_by_class_kg": final_mobile[term].sum(axis=0).tolist(),
        "cumulative_deposition_in_terminal_storage_by_class_kg": engine.cum_dep[term].sum(axis=0).tolist(),
        "final_mobile_total_by_class_kg": final_mobile.sum(axis=0).tolist(),
    }
    wet_q, wet_x, budget = r["water_q"], r["water_export"], r["budget_residual"]
    flux_series = ledger[:, cols["outlet_flux_kg_s"]].sum(axis=1)
    out_dir.mkdir(parents=True, exist_ok=False)
    np.savez(out_dir / "legacy_ledger.npz", t_s=boundaries, ledger=ledger, columns=np.array(LEDGER_COLUMNS),
             column_units=np.array([LEDGER_UNITS[c] for c in LEDGER_COLUMNS]),
             walk_counts=r["walk_counts"], walk_count_names=np.array(N.COUNT_NAMES),
             regime_counts=r["regime_counts"], regime_names=np.array(list(REGIME_CODES)),
             water_outlet_m3_s=wet_q, water_export_m3=wet_x, water_budget_residual_m3=budget, identity_residual_kg=residual,
             cumulative_detachment_kg=engine.cum_det.reshape(ny, nx, nc),
             cumulative_deposition_kg=engine.cum_dep.reshape(ny, nx, nc),
             cumulative_clipping_source_kg=engine.cum_clip.reshape(ny, nx, nc),
             final_mobile_kg=final_mobile.reshape(ny, nx, nc), final_depth_m=np.asarray(r["final_depth"]),
             final_soil_water_m=r["final_soil"], final_discharge_m2_s=r["final_discharge"],
             final_velocity_m_s=r["final_velocity"],
             cumulative_rain_m=cw["rain"], cumulative_intake_m=cw["intake"],
             cumulative_saturation_return_m=cw["saturation_return"], cumulative_drainage_m=cw["drainage"],
             active=network.active.reshape(ny, nx), terminal_storage=network.terminal.reshape(ny, nx),
             outlet=network.outlet.reshape(ny, nx))
    if r["snap_t"]:
        np.savez(out_dir / "legacy_snapshots.npz", t_s=np.array(r["snap_t"]), depth_m=np.stack(r["snap_depth"]),
                 mobile_kg=np.stack(r["snap_mobile"]))
    module_hashes = module_digests()
    if module_hashes != modules_before:
        raise DriverError("a driver module file changed during the run; nothing published")
    law_depth_record = {
        "selected": args.depth_time_level,
        "meaning": {"previous": "the previous step's routed depth (accepted legacy-replay default)",
                    "current": "the new routed depth (diagnostic)",
                    "post_infiltration": "NATIVE original infilt.for d(1): hpre = max(h_old - max(intake - rain, 0), 0), complete "
                                         "branch (intake >= h_old + rain) = 0, saturation return excluded"}[args.depth_time_level],
        "native_original_option": args.depth_time_level == "post_infiltration",
        "velocity": "the new step's velocity (all modes)"}
    limitations = [*_LIMITATIONS, (
        "wet-law depth = the ORIGINAL post-infiltration d(1) (native option); other conventions are unchanged defaults"
        if args.depth_time_level == "post_infiltration" else
        f"wet-law depth = '{args.depth_time_level}' (accepted replay convention); the original reads the post-infiltration d(1): "
        "use --depth-time-level post_infiltration for the native option")]
    summary = {
        "status": "native-walk legacy replay, benchmark only (see limitations)", "limitations": limitations,
        "depth_time_level": args.depth_time_level, "law_depth": law_depth_record,
        "case_kind": args.case_kind, "steps": int(boundaries.size), "dt_s": 1.0, "end_s": end,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "ledger_columns": list(LEDGER_COLUMNS), "ledger_units": dict(LEDGER_UNITS),
        "totals_by_class": totals, "totals_units": "kg (sum over steps of the per-step masses)",
        "totals": {k: float(np.sum(v)) for k, v in totals.items()},
        "peak_outlet_flux_kg_s": float(flux_series.max()) if flux_series.size else 0.0,
        "final_mobile_by_class_kg": final_mobile.sum(axis=0).tolist(),
        "terminal_storage": terminal_diag, "walk_tallies": dict(zip(N.COUNT_NAMES, engine.total_counts.tolist(), strict=True)),
        "wet_regime_cell_class_steps": dict(zip(REGIME_CODES, engine.total_regime_counts.tolist(), strict=True)),
        "identity": {"rule": "new - old - (pickup - deposition_active) + cn_export - clip, per step and class",
                     "worst_relative_residual": worst, "guard_rtol": IDENTITY_RTOL,
                     "max_abs_residual_kg": float(np.abs(residual).max()) if residual.size else 0.0},
        "water": {"peak_outlet_m3_s": float(wet_q.max()), "export_m3": float(wet_x.sum()),
                  "per_step_budget_residual_m3": {"max_abs": float(np.abs(budget).max()), "sum": float(budget.sum()),
                                                  "note": "reported from the routing step"},
                  "whole_storm_budget": water},
        "onset": {"first_positive_pickup_s": first_positive(ledger[:, cols["pickup_kg"]], boundaries),
                  "first_positive_deposition_s": first_positive(ledger[:, cols["deposition_active_kg"]], boundaries),
                  "wet_law_cell_class_steps": int(sum(engine.total_regime_counts[c] for n_, c in REGIME_CODES.items()
                                                      if n_ not in ("dry", "wet_no_law"))),
                  "note": "model times (end of the step) of the first step with a positive class-summed value; None = never"},
        "network": network.summary(),
        "maps_total_kg": {k: np.asarray(v).sum(axis=0).tolist() for k, v in maps.items()},
        "artifact_pins": {"checked_unchanged_before_publish": True, "sha256": pins_before},
        "chastre_tiles": tiles,
        "performance": {
            "loop_wall_s_excluding_progress": r["loop_s"], "hydrology_s": r["hydrology_s"],
            "wet_laws_s": engine.timers["physics_s"], "glue_walk_cn_reduce_s": engine.timers["glue_walk_cn_s"],
            "progress_s": r["progress_s"], "first_step_including_jit": r["first_step"], "warmup": warm,
            "case_verify_and_adapt_s": case_s, "network_build_s": network_s, "physics_prepare_s": physics_prep_s,
            "hydrology_prepare_s": hydro_prep_s, "whole_process_wall_s": time.perf_counter() - t_wall0,
            "peak_rss_kib": _peak_rss_kib(), "engine_buffer_bytes": engine.nbytes(), "estimated_engine_bytes": estimate_bytes,
            "note": "first step includes JIT unless --warmup-s was used; timers exclude progress output"},
        "hydrology": {"root_solver": control.root_solver, "bisection_iterations": control.bisection_iterations,
                      "newton_max_iterations": control.newton_max_iterations, "context": hydro_ctx.summary(),
                      "kernels": hn.kernel_provenance()},
        "kernels": {"legacy_cn": "accepted legacy_transport._cn" + (" (Numba)" if L.KERNEL_IMPLEMENTATION == "numba" else ""),
                    "native_kernels_compiled": not args.allow_python_kernels},
        "case": case.record,
        "provenance": {"maple_syrup": syrup_prov, "maple": case.verified.maple_provenance,
                       "source_digests_before": digests_before, "source_digests_after": digests_after,
                       "graph_input_sha256": case.graph.input_sha256, "environment": environment_record(),
                       "module_sha256": module_hashes},
    }
    (out_dir / "legacy_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def physics_ctx_bytes_estimate(n: int, nc: int) -> int:
    """Static wet-law context: slope_power, fractions, cap_rate (n*nc) plus ~6 per-cell arrays."""
    return 3 * n * nc * 8 + 6 * n * 8


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_controls(args)  # no case adaptation, allocation or write happens before the controls are valid
    out = args.output.resolve()
    roots = protected_roots(args.case.resolve())
    check_output(out, roots, require_absent=True)
    partial = out.with_name(out.name + ".partial")
    failed = out.with_name(out.name + ".FAILED")
    try:
        summary = _execute(args, partial, roots)
    except BaseException as exc:
        try:
            check_output(out, roots, require_absent=False)  # the roots may have grown while the case was loaded
        except DriverError:
            raise exc from None  # unsafe location: report the failure but write nothing
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
    except DriverError as exc:
        print(f"legacy native driver failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report and set the exit status
        print(f"legacy native driver failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: summary[k] for k in ("totals", "totals_units", "terminal_storage", "walk_tallies", "identity",
                                              "performance")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
