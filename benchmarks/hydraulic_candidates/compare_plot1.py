"""Bounded comparison harness: legacy GPU / BEST prepared-Numba legacy / the two EXPERIMENTAL hydraulic candidates (NumPy
reference and CUDA) on the SAME Plot 1 geometry, column parameters, rainfall schedule/field and antecedent soil water.

    python benchmarks/hydraulic_candidates/compare_plot1.py --case-dir outputs/plot1 --output-dir <NEW> \\
        --contenders legacy_numba_prepared,legacy_cuda,explicit_numpy,explicit_cuda,local_inertial_numpy,local_inertial_cuda \\
        --dts 1,0.5,0.25 --end-s 5400 --report-every-s 60 --snapshot-times-s 600,1200,1620 --repeats 1 \\
        --allow-maple-source-change

Contenders: `legacy_numba_prepared` = the shared legacy scheduler (`storm.evolve`) stepping through the prepared batched
Numba hydrology (the BEST CPU comparator; `storm.coupled_step` is redirected only inside a fresh try/finally scope entered
for every segment of every run by THIS script, never edited in production); `legacy_cuda` = `storm.evolve` with
`implementation="cuda"` and a context prepared once; `explicit_*` / `local_inertial_*` =
`experimental_storm.evolve_experimental` with the NumPy or CUDA solver; `explicit_numba` / `local_inertial_numba` = the same driver
with the compiled CPU `experimental_numba.NumbaHydraulicSolver` (accepted by name; not in the default list; its kernels compile
in the untimed warm-up and its contexts are built once per run like every other contender's).

Protocol (every number below is a raw observation of one process; nothing is claimed equivalent and no speed claim is made):
  * every numeric control is validated BEFORE any run (nothing is prepared or compiled on bad input), the output path is
    refused when it exists or lies inside the bound case / MAPLE / MAHLERAN / recipe / SYRUP package trees (the same
    `_refuse_output` rule as the experimental CLI), and files are created exclusively: nothing is ever overwritten;
  * static preparation is timed separately; one UNTIMED warm-up run (the first `--warmup-s` seconds) absorbs compilation and
    first use; each timed run covers the whole storm including recession and `--repeats` samples are taken;
  * the timed region holds only the evolution and a device synchronization. EVERY sample (and every snapshot-free legacy
    single-event run) is validated AFTER its own timer and BEFORE it can be selected: diagnostics are downloaded, the event
    WATER BUDGET is computed with the same `volume_roundoff_bound_m3` rule as the CLI (surface_final + soil_final + drainage
    + export = surface_initial + soil_initial + rain, the surface and soil identities and the rainfall integral), the MAPLE
    bed digest and the SYRUP/MAPLE source digests are compared with their values from before that run; any violation aborts
    the script BEFORE any output is written. Each sample's budget, guard status and timing are saved; the reported run is the
    fastest VALIDATED sample;
  * DISCLOSURE: the legacy driver has no snapshot facility, so legacy snapshot maps come from running the shared scheduler in
    SEGMENTS ending at the requested times (use times that are multiples of dt: the step grid is then unchanged). That adds
    segment overhead (extra boundaries, validations, snapshot copies) which the candidates' single event does not have; for
    legacy contenders with snapshots a second, snapshot-free single-event run is therefore also timed and validated
    (`single_event`, disable with `--no-legacy-single-event`). Snapshot copies are device-to-device inside the timer for both;
  * comparisons: every contender against `legacy_numba_prepared` at the SAME dt, every contender against its own finest dt
    (self-refinement), and every run against the finest legacy run; metrics are reported, not judged.
Written (into the NEW directory): `comparison.json` (runs, samples, budgets, provenance, comparisons), `maps.npz` (synchronous
depth/velocity maps), `final_arrays.npz` (full final depth, soil water, face fluxes / discharge of every run).
Nothing here was run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CONTENDERS = ("legacy_numba_prepared", "legacy_cuda", "explicit_numpy", "explicit_cuda", "local_inertial_numpy",
              "local_inertial_cuda")
# The default contender list is the original six (unchanged: other scripts iterate it). The compiled CPU forms of the two
# candidates are accepted by name only: `--contenders ...,explicit_numba,local_inertial_numba` (fresh lifetime, guards and
# per-sample validation exactly as every other contender).
NUMBA_CONTENDERS = ("explicit_numba", "local_inertial_numba")
ALL_CONTENDERS = (*CONTENDERS, *NUMBA_CONTENDERS)
REFERENCE = "legacy_numba_prepared"
MIN_DT_S = 1.0 / 1024.0  # the retry floor of both drivers' default controls


@contextlib.contextmanager
def scoped_coupled_step(replacement):
    """BENCHMARK ONLY: point `storm.coupled_step` at `replacement` for the shared scheduler; restored in `finally`. A
    generator context manager can be entered ONCE, so callers create a FRESH one for every use (never store one)."""
    from maple_syrup import storm

    original = storm.coupled_step
    storm.coupled_step = replacement
    try:
        yield
    finally:
        storm.coupled_step = original


def to_host_array(x):
    from maple.core.backend import to_host

    return np.asarray(to_host(x)).copy()


def run_legacy_segments(graph, params, field, schedule, depth0, soil0, control, end_s, snapshots, cadence, *, xp,
                        step_override=None, cuda_context=None):
    """The shared legacy scheduler (`storm.evolve`), stepped in segments that end at the snapshot times and at `end_s`. A fresh
    `scoped_coupled_step` is entered for EVERY segment when `step_override` is given, so warm-up, repeated runs and snapshot
    segmentation can never reuse a spent context manager, and the original `storm.coupled_step` is restored after every
    segment, also on failure. Returns device objects only (nothing is downloaded)."""
    from maple.core.backend import synchronize

    from maple_syrup.storm import evolve, initial_state

    extra = {} if cuda_context is None else {"cuda_context": cuda_context}
    state = initial_state(graph, depth0, soil0)
    segments, maps = [], {}
    for stop in sorted({*snapshots, end_s}):
        scope = contextlib.nullcontext() if step_override is None else scoped_coupled_step(step_override)
        with scope:
            result = evolve(graph, params, field, schedule, state, stop, control, report_every_s=cadence, **extra)
        segments.append(result)
        state = result.state
        if stop in snapshots:
            maps[stop] = {"depth_m": xp.array(state.depth_m, copy=True),
                          "velocity_m_s": xp.array(result.last_velocity_m_s, copy=True)}
    synchronize(xp)
    return {"kind": "legacy", "segments": segments, "maps": maps, "state": state}


def parse_floats(text: str, name: str) -> list[float]:
    try:
        return [float(v) for v in text.split(",") if v.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of numbers, got {text!r}") from exc


def validate_controls(args) -> tuple[list[str], list[float], list[float]]:
    """Strict validation of every numeric control BEFORE any run; raises ValueError."""
    names = [n for n in args.contenders.split(",") if n]
    unknown = [n for n in names if n not in ALL_CONTENDERS]
    if not names or unknown or len(set(names)) != len(names):
        raise ValueError(f"contenders must be a non-empty, duplicate-free subset of {ALL_CONTENDERS}; unknown: {unknown}")
    dts = parse_floats(args.dts, "--dts")
    snapshots = parse_floats(args.snapshot_times_s, "--snapshot-times-s")
    for label, value in (("--end-s", args.end_s), ("--report-every-s", args.report_every_s)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    if not dts or any(not (math.isfinite(d) and d >= MIN_DT_S) for d in dts):
        raise ValueError(f"--dts must be a non-empty list of finite numbers >= {MIN_DT_S} (the drivers' retry floor), "
                         f"got {dts!r}")
    if len(set(dts)) != len(dts):
        raise ValueError("--dts must not repeat a value")
    if not (math.isfinite(args.warmup_s) and args.warmup_s >= 0.0):
        raise ValueError(f"--warmup-s must be finite and >= 0, got {args.warmup_s!r}")
    if isinstance(args.repeats, bool) or args.repeats < 1:
        raise ValueError(f"--repeats must be an int >= 1, got {args.repeats!r}")
    if any(not (math.isfinite(s) and 0.0 < s <= args.end_s) for s in snapshots):
        raise ValueError(f"--snapshot-times-s must lie in (0, end_s], got {snapshots!r}")
    if not (math.isfinite(args.cfl_max) and 0.0 < args.cfl_max <= 0.5):
        raise ValueError(f"--cfl-max must lie in (0, 0.5], got {args.cfl_max!r}")
    return names, dts, snapshots


def build_runner(name: str, dt: float, verified, args, *, inputs_factory=None, storm_overrides=None):
    """Returns `(run, record, inputs, provenance)`. `run(end_s, snapshot_times)` returns a dict of DEVICE objects and downloads
    nothing; `record` holds the preparation time; `provenance` the actual control/context/kernel metadata of this contender.

    `inputs_factory(verified, backend, with_geometry=...)` (default `plot1_inputs`) lets another case (benchmarks/rfid) supply the
    same input namespace; `storm_overrides` (default none) are extra `StormControl` fields for the legacy contenders (e.g.
    `{"bisection_iterations": 64}`). With both omitted the behaviour is exactly the original."""
    from maple.core.backend import synchronize

    from maple_syrup.experimental_experiment import plot1_inputs
    from maple_syrup.experimental_hydrology import (
        CpuHydraulicSolver,
        HydraulicControl,
    )
    from maple_syrup.experimental_storm import (
        ExperimentalControl,
        evolve_experimental,
    )
    from maple_syrup.storm import StormControl

    family = name.rsplit("_", 1)[0] if name.startswith(("explicit", "local_inertial")) else "legacy"
    backend = "cupy" if name.endswith("cuda") else "numpy"
    inputs = (inputs_factory or plot1_inputs)(verified, backend, with_geometry=family == "local_inertial")
    xp = inputs.xp
    cadence = args.report_every_s
    t0 = time.perf_counter()
    if family == "legacy":
        control = StormControl(max_dt_s=dt, implementation="cuda" if backend == "cupy" else "numba",
                               **(storm_overrides or {})).validated()
        step_override, cuda_context = None, None
        if backend == "cupy":
            from maple_syrup import hydrology_cuda as hc

            cuda_context = hc.prepare_cuda_hydrology(inputs.graph, inputs.params)
            newton_load = hc.load_newton_kernels() if control.root_solver == "newton" else None  # startup, not a sample
            provenance = {"control": dataclasses.asdict(control), "context": cuda_context.summary(),
                          "kernels": hc.kernel_provenance(),
                          "newton_kernel_load_s": None if newton_load is None else newton_load["seconds"]}
        else:
            from maple_syrup import hydrology_numba as hn

            prepared = hn.prepare_hydrology(inputs.graph, inputs.params)

            def step_override(graph, params, rate, state, dt_, control_):  # plain closure: no wrapper layers in the hot path
                return hn.prepared_coupled_step(prepared, rate, state, dt_, control_)

            provenance = {"control": dataclasses.asdict(control), "context": prepared.summary(),
                          "kernels": "recorded after the warm-up compiled the kernels"}
        record = {"preparation_s": time.perf_counter() - t0}

        def run(end_s, snapshots):
            return run_legacy_segments(inputs.graph, inputs.params, inputs.field, inputs.schedule, inputs.depth0,
                                       inputs.soil0, control, end_s, snapshots, cadence, xp=xp,
                                       step_override=step_override, cuda_context=cuda_context)

        return run, record, inputs, provenance
    hydraulic = HydraulicControl(cfl_max=args.cfl_max, limiter=args.limiter if family == "local_inertial" else "off")
    if backend == "cupy":
        from maple_syrup.experimental_cuda import CudaHydraulicSolver

        solver = CudaHydraulicSolver(family, inputs.graph, inputs.params, geometry=inputs.geometry, control=hydraulic)
    elif name.endswith("_numba"):  # compiled CPU form; its context, kernels and column context live as long as this run
        from maple_syrup.experimental_numba import NumbaHydraulicSolver

        solver = NumbaHydraulicSolver(family, inputs.graph, inputs.params, geometry=inputs.geometry, control=hydraulic)
    else:
        solver = CpuHydraulicSolver(family, inputs.graph, inputs.params, geometry=inputs.geometry, control=hydraulic)
    state0 = solver.initial_state(inputs.depth0, inputs.soil0)
    record = {"preparation_s": time.perf_counter() - t0}
    control = ExperimentalControl(max_dt_s=dt)
    provenance = {"control": dataclasses.asdict(control), "hydraulics": solver.describe(),
                  "geometry_sha256": None if inputs.geometry is None else inputs.geometry.input_sha256}

    def run(end_s, snapshots):
        result = evolve_experimental(solver, inputs.field, inputs.schedule, state0, end_s, control,
                                     report_every_s=cadence, snapshot_times_s=tuple(snapshots))
        synchronize(xp)
        return {"kind": "candidate", "result": result}

    return run, record, inputs, provenance


def collect(raw, inputs):
    """HOST numbers, budget terms, maps and final arrays (the only downloads; called AFTER the timer)."""
    area = inputs.area
    if raw["kind"] == "legacy":
        segs = raw["segments"]
        export = sum(float(to_host_array(s.cumulative_export_m3)) for s in segs)
        peaks = [(float(to_host_array(s.peak_outlet_discharge_m3_s)), float(to_host_array(s.time_of_peak_outlet_s)))
                 for s in segs]
        peak_q, peak_t = max(peaks, key=lambda p: p[0])
        rows = np.concatenate([to_host_array(s.hydrograph) for s in segs])
        cum = {key: sum(float(to_host_array(getattr(s, attr)).sum()) for s in segs) for key, attr in (
            ("rain", "cumulative_rain_m"), ("intake", "cumulative_intake_m"),
            ("saturation_return", "cumulative_saturation_return_m"), ("drainage", "cumulative_drainage_m"))}
        state = raw["state"]
        accepted = sum(s.n_accepted_steps for s in segs)
        rejected = sum(s.n_rejected_attempts for s in segs)
        maps = {t: {k: to_host_array(v) for k, v in m.items()} for t, m in raw["maps"].items()}
        finals = {"depth_m": to_host_array(state.depth_m), "soil_water_m": to_host_array(state.soil_water_m),
                  "discharge_m2_s": to_host_array(state.discharge_m2_s)}
        n_segments = len(segs)
    else:
        r = raw["result"]
        export = float(to_host_array(r.cumulative_export_m3))
        peak_q, peak_t = float(to_host_array(r.peak_outlet_discharge_m3_s)), float(to_host_array(r.time_of_peak_outlet_s))
        rows = to_host_array(r.hydrograph)
        cum = {"rain": float(to_host_array(r.cumulative_rain_m).sum()),
               "intake": float(to_host_array(r.cumulative_intake_m).sum()),
               "saturation_return": float(to_host_array(r.cumulative_saturation_return_m).sum()),
               "drainage": float(to_host_array(r.cumulative_drainage_m).sum())}
        state = r.state
        accepted, rejected = r.n_accepted_steps, r.n_rejected_attempts
        maps = {t: {"depth_m": to_host_array(s["depth_m"]), "velocity_m_s": to_host_array(s["velocity_m_s"])}
                for t, s in r.snapshots.items()}
        finals = {"depth_m": to_host_array(state.depth_m), "soil_water_m": to_host_array(state.soil_water_m)}
        if state.qx_m2_s is not None:
            finals["qx_m2_s"], finals["qy_m2_s"] = to_host_array(state.qx_m2_s), to_host_array(state.qy_m2_s)
        n_segments = 1
    surface0 = float(to_host_array(inputs.depth0).sum())
    soil0 = float(to_host_array(inputs.soil0).sum())
    volumes = {"surface_initial": surface0 * area, "soil_initial": soil0 * area, "rain": cum["rain"] * area,
               "intake": cum["intake"] * area, "saturation_return": cum["saturation_return"] * area,
               "drainage": cum["drainage"] * area, "export": export,
               "surface_final": float(finals["depth_m"].sum()) * area, "soil_final": float(finals["soil_water_m"].sum()) * area}
    host = {"accepted": accepted, "rejected": rejected, "export": export, "peak_q": peak_q, "peak_t": peak_t, "rows": rows,
            "volumes": volumes, "n_segments": n_segments}
    return host, maps, finals


def water_budget(volumes: dict, inputs, accepted: int, end_s: float) -> dict:
    """The CLI's event budget with the MAPLE-derived `volume_roundoff_bound_m3`; raises on any violation."""
    from maple_syrup.conservation import volume_roundoff_bound_m3

    v = volumes
    residual = (v["surface_final"] + v["soil_final"] + v["drainage"] + v["export"]
                - v["surface_initial"] - v["soil_initial"] - v["rain"])
    surface = v["surface_final"] - (v["surface_initial"] + v["rain"] - v["intake"] + v["saturation_return"] - v["export"])
    soil = v["soil_final"] - (v["soil_initial"] + v["intake"] - v["drainage"] - v["saturation_return"])
    n_terms = 4 * inputs.ny * inputs.nx * max(accepted, 1) + 7
    bound = volume_roundoff_bound_m3(n_terms, max(abs(x) for x in v.values()))
    rain_expected = inputs.schedule.depth_m(0.0, end_s) * float(inputs.host["rainfall_scale"].sum()) * inputs.area
    rain_bound = volume_roundoff_bound_m3(n_terms, max(rain_expected, v["rain"]))
    checks = {"water": (residual, bound), "surface": (surface, bound), "soil": (soil, bound),
              "rainfall_integral": (v["rain"] - rain_expected, rain_bound)}
    for label, (value, tol) in checks.items():
        if not abs(value) <= tol:
            raise RuntimeError(f"{label} budget residual {value} m3 exceeds the MAPLE-derived bound {tol} m3; "
                               "no output written")
    return {"units": "m3", **{f"{k}_m3": x for k, x in v.items()}, "rain_expected_m3": rain_expected,
            "water_residual_m3": residual, "surface_residual_m3": surface, "soil_residual_m3": soil, "bound_m3": bound,
            "rainfall_bound_m3": rain_bound, "closed": True,
            "rule": "conservation.volume_roundoff_bound_m3(4 n_cells n_steps + 7, largest operand), as the CLI"}


class RunGuard:
    """Per-run integrity check: the MAPLE bed digest and the SYRUP/MAPLE source digests recorded BEFORE a run must equal their
    values after every sample, and every sample's water budget must close. Everything here runs outside the timers."""

    def __init__(self, inputs, package_dirs, label: str, bed_digest=None):
        from maple_syrup.column_experiment import (
            _bed_digest,
            _source_digests,
        )

        # `bed_digest` (optional `callable(case) -> str`): a case without in-memory bed arrays supplies its own persisted-bed
        # digest; the default is the unchanged MAPLE array digest
        self._custom_bed = bed_digest is not None
        self._bed, self._sources = (_bed_digest if bed_digest is None else bed_digest), _source_digests
        self.inputs, self.dirs, self.label = inputs, package_dirs, label
        self.bed_before, self.sources_before = self._bed(inputs.case), _source_digests(*package_dirs)

    def validate(self, raw, end_s: float) -> dict:
        host, maps, finals = collect(raw, self.inputs)
        budget = water_budget(host["volumes"], self.inputs, host["accepted"], end_s)  # raises before anything is selected
        bed_after, sources_after = self._bed(self.inputs.case), self._sources(*self.dirs)
        if bed_after != self.bed_before:
            raise RuntimeError(f"{self.label}: the MAPLE bed changed during a water-only run; no output written")
        if sources_after != self.sources_before:
            raise RuntimeError(f"{self.label}: SYRUP/MAPLE source changed during the run; no output written")
        guard = {"budget_closed": True, "bed_unchanged": True, "sources_stable": True, "bed_digest": bed_after}
        if self._custom_bed:
            guard["bed_digest_kind"] = "custom callback (persisted artifacts), not an in-memory bed array digest"
        return {"host": host, "maps": maps, "finals": finals, "budget": budget, "guard": guard}


def compare(ref, other, ref_maps, maps):
    out = {"export_rel_diff": (other["export_m3"] - ref["export_m3"]) / ref["export_m3"] if ref["export_m3"] else None,
           "peak_rel_diff": (other["peak_outlet_m3_s"] - ref["peak_outlet_m3_s"]) / ref["peak_outlet_m3_s"]
           if ref["peak_outlet_m3_s"] else None,
           "time_of_peak_diff_s": other["time_of_peak_s"] - ref["time_of_peak_s"],
           "surface_final_diff_m3": other["budget"]["surface_final_m3"] - ref["budget"]["surface_final_m3"],
           "soil_final_diff_m3": other["budget"]["soil_final_m3"] - ref["budget"]["soil_final_m3"], "maps": {}}
    st, sq = np.asarray(ref["series_t"]), np.asarray(ref["series_q"])
    ot, oq = np.asarray(other["series_t"]), np.asarray(other["series_q"])
    common = np.intersect1d(st, ot)
    if common.size and np.linalg.norm(sq[np.isin(st, common)]) > 0:
        a, b = sq[np.isin(st, common)], oq[np.isin(ot, common)]
        out["hydrograph_rel_l2"] = float(np.linalg.norm(b - a) / np.linalg.norm(a))
    for t, m in maps.items():
        if t in ref_maps:
            out["maps"][str(t)] = {k: float(np.linalg.norm(m[k] - ref_maps[t][k]) / max(np.linalg.norm(ref_maps[t][k]), 1e-300))
                                   for k in ("depth_m", "velocity_m_s")}
    return out


def time_sample(run, end_s, snapshots, read_counters):
    """One timed run: evolution + synchronization ONLY (the counters are read around it; nothing else happens inside)."""
    before = read_counters()
    t0 = time.perf_counter()
    raw = run(end_s, snapshots)
    wall = time.perf_counter() - t0
    return wall, raw, dataclasses.asdict(read_counters().delta(before))


def sample_record(wall, transfers, checked):
    return {"wall_s": wall, "transfers": transfers, "budget": checked["budget"], "guard": checked["guard"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW directory (an existing path is refused)")
    parser.add_argument("--contenders", default=",".join(CONTENDERS))
    parser.add_argument("--dts", default="1,0.5,0.25")
    parser.add_argument("--end-s", type=float, default=5400.0)
    parser.add_argument("--report-every-s", type=float, default=60.0)
    parser.add_argument("--snapshot-times-s", default="600,1200,1620")
    parser.add_argument("--warmup-s", type=float, default=60.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cfl-max", type=float, default=0.5)
    parser.add_argument("--limiter", default="off", choices=("off", "donor"))
    parser.add_argument("--root-solver", choices=("bisection", "newton"), default="bisection",
                        help="root solver of the legacy contenders (default bisection, unchanged); 'newton' selects the "
                             "safeguarded Newton of the CPU Numba form and of legacy_cuda (CUDA Newton variant; see "
                             "benchmarks/newton_cpu/compare_cases.py / benchmarks/gpu_newton for the matched comparison)")
    parser.add_argument("--newton-max-iterations", type=int, default=50)
    parser.add_argument("--no-legacy-single-event", action="store_true",
                        help="skip the extra snapshot-free single-event timing of legacy contenders")
    parser.add_argument("--allow-maple-source-change", action="store_true")
    args = parser.parse_args(argv)
    try:
        names, dts, snapshots = validate_controls(args)
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"refusing to overwrite {output_dir}")
    from maple.core.backend import read_transfer_counters

    from maple_syrup.case_import import (
        Plot1ImportError,
        _refuse_output,
        verify_plot1_case,
    )
    from maple_syrup.column_experiment import (
        _source_digests,
        _syrup_provenance,
    )
    from maple_syrup.dependency import MapleDependencyError
    from maple_syrup.experimental_cuda import kernel_provenance
    from maple_syrup.provenance import environment_record

    try:
        verified = verify_plot1_case(args.case_dir, allow_maple_source_change=args.allow_maple_source_change)
        syrup = _syrup_provenance()
        dependency, recipe = verified.maple_dependency, verified.report["recipe"]
        _refuse_output(output_dir, {  # the experimental CLI's protected trees, plus the bound case and the SYRUP package
            "MAPLE source": dependency.source_root, "MAPLE package": dependency.package_dir,
            "recorded MAHLERAN": Path(recipe["mahleran_root"]).resolve(),
            "recipe": Path(recipe["recipe_path"]).resolve().parent,
            "bound case": Path(args.case_dir).resolve(), "SYRUP package": Path(syrup["package_dir"]).resolve(),
        })
    except (Plot1ImportError, MapleDependencyError) as exc:
        parser.error(str(exc))
    package_dirs = (Path(syrup["package_dir"]), Path(dependency.package_dir))
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    digests_start = _source_digests(*package_dirs)
    runs, best_maps, best_finals = {}, {}, {}
    for dt in dts:
        for name in names:
            snaps_for_dt = [s for s in snapshots if abs(s / dt - round(s / dt)) < 1e-9]
            skipped = [s for s in snapshots if s not in snaps_for_dt]
            overrides = (None if args.root_solver == "bisection"
                         else {"root_solver": args.root_solver, "newton_max_iterations": args.newton_max_iterations})
            run, prep, inputs, provenance = build_runner(name, dt, verified, args, storm_overrides=overrides)
            guard = RunGuard(inputs, package_dirs, f"{name} dt={dt}")  # digests BEFORE this contender's runs
            if args.warmup_s > 0.0:
                run(min(args.warmup_s, args.end_s), [])  # untimed warm-up: compilation, first use, pool growth
            if name == "legacy_numba_prepared":
                from maple_syrup import hydrology_numba as hn

                provenance["kernels"] = hn.kernel_provenance()  # after the warm-up compiled them; outside any timer
            samples = []
            for _ in range(args.repeats):
                wall, raw, transfers = time_sample(run, args.end_s, snaps_for_dt, read_transfer_counters)
                checked = guard.validate(raw, args.end_s)  # AFTER the timer, BEFORE the sample may be selected
                samples.append((wall, transfers, checked))
            single_event = None
            if samples[0][2]["host"]["n_segments"] > 1 and not args.no_legacy_single_event:
                wall1, raw1, transfers1 = time_sample(run, args.end_s, [], read_transfer_counters)  # the same storm, one event
                single_event = sample_record(wall1, transfers1, guard.validate(raw1, args.end_s))
            wall, transfers, checked = min(samples, key=lambda s: s[0])
            host = checked["host"]
            rows = host["rows"]
            runs[(name, dt)] = {
                "contender": name, "dt_s": dt, "wall_s": wall, "samples": [sample_record(*s) for s in samples],
                "single_event": single_event, "n_segments": host["n_segments"],
                "event_structure": (f"legacy scheduler in {host['n_segments']} segment(s) ending at the snapshot times; the "
                                    "timed wall includes that segment overhead and its device snapshot copies")
                if host["n_segments"] > 1 or name.startswith("legacy")
                else "one evolve_experimental event with snapshot boundaries",
                "snapshots_taken": snaps_for_dt, "snapshots_skipped_not_multiples_of_dt": skipped,
                "accepted_steps": host["accepted"], "rejected_attempts": host["rejected"], "export_m3": host["export"],
                "peak_outlet_m3_s": host["peak_q"], "time_of_peak_s": host["peak_t"], "budget": checked["budget"],
                "guard": checked["guard"],
                "runoff_coefficient": host["export"] / checked["budget"]["rain_m3"] if checked["budget"]["rain_m3"] > 0
                else None,
                "transfers": transfers, "preparation": prep, "provenance": provenance,
                "inputs_binding": {"graph_input_sha256": inputs.graph.input_sha256, "parameters": inputs.parameter_record,
                                   "rainfall_sha256": inputs.schedule.provenance.sha256,
                                   "geometry_sha256": None if inputs.geometry is None else inputs.geometry.input_sha256},
                "bed_digest": {"before": guard.bed_before, "after": checked["guard"]["bed_digest"], "unchanged": True},
                "source_digests": {"before": guard.sources_before, "stable": True},
                "peak_rss_kib_so_far": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "series_t": rows[:, 0].tolist(), "series_q": rows[:, 8].tolist()}
            best_maps[(name, dt)], best_finals[(name, dt)] = checked["maps"], checked["finals"]
    comparisons = {}
    finest = min(dts)
    for (name, dt), run_record in runs.items():
        if name != REFERENCE and (REFERENCE, dt) in runs:  # the same dt
            comparisons[f"{name}@dt{dt}_vs_{REFERENCE}@dt{dt}"] = compare(
                runs[(REFERENCE, dt)], run_record, best_maps[(REFERENCE, dt)], best_maps[(name, dt)])
        if dt != finest and (name, finest) in runs:  # self-refinement: the same contender at its finest dt
            comparisons[f"{name}@dt{dt}_vs_{name}@dt{finest}_self"] = compare(
                runs[(name, finest)], run_record, best_maps[(name, finest)], best_maps[(name, dt)])
        if (REFERENCE, finest) in runs and (name, dt) != (REFERENCE, finest):  # refinement reference: finest legacy
            comparisons[f"{name}@dt{dt}_vs_{REFERENCE}@dt{finest}"] = compare(
                runs[(REFERENCE, finest)], run_record, best_maps[(REFERENCE, finest)], best_maps[(name, dt)])
    digests_end = _source_digests(*package_dirs)
    if digests_end != digests_start:
        raise RuntimeError("SYRUP/MAPLE source changed during the comparison; no output written")
    payload = {
        "schema": "maple_syrup.hydraulic_candidates.comparison.v3", "python": platform.python_version(),
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "script_sha256": script_sha, "environment": environment_record(), "syrup_provenance": syrup,
        "maple_provenance": verified.maple_provenance, "case_binding": dict(verified.binding),
        "source_digests": {"start": digests_start, "end": digests_end, "stable": True},
        "protected_output_trees_checked": True, "cuda_kernel_provenance": kernel_provenance(),
        "runs": list(runs.values()), "comparisons": comparisons,
        "scope": ("raw observations of one process; every timed sample was validated (budget, bed digest, source digests) "
                  "after its own timer and before selection; diagnostics are downloaded outside the timers; legacy snapshot "
                  "maps need segmented legacy runs (see event_structure / single_event); the reference is the "
                  "corrected-coherent MAHLERAN-inspired method-5 hydrology; nothing is claimed equivalent and this script "
                  "makes no speed claim")}
    output_dir.mkdir(parents=True)  # no exist_ok: a directory created since the check is refused
    with (output_dir / "comparison.json").open("x") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    for file_name, arrays in (
        ("maps.npz", {f"{name}_dt{dt:g}_t{t:g}_{k}": v for (name, dt), maps in best_maps.items() for t, m in maps.items()
                      for k, v in m.items()}),
        ("final_arrays.npz", {f"{name}_dt{dt:g}_{k}": v for (name, dt), finals in best_finals.items()
                              for k, v in finals.items()}),
    ):
        with (output_dir / file_name).open("xb") as handle:
            np.savez(handle, **arrays)
    print(json.dumps({f"{n}@{d}": {"wall_s": r["wall_s"], "export_m3": r["export_m3"], "peak": r["peak_outlet_m3_s"],
                                   "budget_closed": r["budget"]["closed"]} for (n, d), r in runs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
