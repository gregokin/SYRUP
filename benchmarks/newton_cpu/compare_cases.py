"""EXPERIMENTAL Newton-versus-bisection root-solver comparison on the verified Plot 1 and RFID_2014 cases (task newton_cpu).

    python benchmarks/newton_cpu/compare_cases.py --case rfid  --case-dir outputs/rfid/case  --output-dir <NEW> \\
        [--contenders ...] [--rounds 3] [--order balanced] [--fortran-build-dir <NEW>] [--allow-maple-source-change]
    python benchmarks/newton_cpu/compare_cases.py --case plot1 --case-dir outputs/plot1       --output-dir <NEW> ...

Contenders (`--list-contenders`), all on the SAME current source, the SAME verified case, forcing, column parameters, frozen
elevation/routing, no splash, WATER ONLY, full storm (defaults: Plot 1 5400 s, RFID 2700 s, max dt 1 s, report 60 s):

  bisection_numpy   shared scheduler `storm.evolve`, `implementation="array"`, NumPy column + level sweep, bisection (default)
  bisection_numba   the same scheduler stepping the PREPARED compiled hydrology, bisection (the accepted CPU comparator)
  newton_numpy      as bisection_numpy with `root_solver="newton"` (vectorized per dependency level)
  newton_numba      as bisection_numba with `root_solver="newton"` (compiled per-cell safeguarded Newton)
  bisection_cuda    the same scheduler stepping the PREPARED CUDA hydrology (water only), bisection (accepted GPU form)   -- needs CuPy+device
  newton_cuda       the same with `root_solver="newton"` (device safeguarded Newton, routing_newton_cuda)               -- needs CuPy+device
  fortran_newton    ORIGINAL MAHLERAN routines, `iroute = 2` (native Newton-Crank-Nicolson)       -- RFID only
  fortran_bisection ORIGINAL MAHLERAN routines, `iroute = 5` (bisection-Crank-Nicolson)            -- RFID only

The Fortran contenders reuse `benchmarks/rfid/fortran_timing.py` and its unchanged `rfid_water_driver.f90` (the originals'
own qualification of every sample). That driver is an RFID adapter: it hardcodes `inf_model = 1` / `inf_type = 1` (fixed
Ksat) and `fortran_timing.common_arrays` writes a zero pavement array, which matches the RFID `fixed_ksat` columns. The actual
Plot 1 columns are `pavement_hawkins` (model 2 with a prescribed pavement map), so reusing it there would compare different
hydrology. `--case plot1` therefore REFUSES the Fortran contenders in this bounded task: an unsupported harness
configuration, not an intrinsic impossibility (a model-2/pavement-aware adapter around the same unchanged original
routines is feasible later and is NOT built here).
CUDA contenders (task gpu_newton; excluded from the default list, name them in `--contenders`) use the SAME verified case, the
same RunGuard (water budget with the MAPLE-derived bound, bed digest, source digests), the same first call / complete warm-up /
balanced rounds protocol and capture final fields and the full outlet hydrograph outside every timer. `--cuda-mode
auto|fused|split` forces the launch structure of the prepared context (recorded in the context summary). The Newton kernel
variant is compiled/loaded in the contender's PREPARATION (recorded as `newton_kernel_load_s`), the static context is
prepared once; neither is inside a sample. The device Newton production step reports no counters (`root_stats` None), so no
GPU pass statistics are recorded; the `newton_numba` statistics describe the same equation on the same case.
Reuse, not copies: the case loaders (`verify_plot1_case` / `verify_rfid_case`, `plot1_inputs` / `rfid_inputs`),
`RunGuard` (per-sample water budget with the MAPLE-derived bound, MAPLE bed digest, source digests), `time_sample`,
`sample_record` of benchmarks/hydraulic_candidates/compare_plot1.py, and the balanced order / summaries of
benchmarks/rfid/run_rfid_timing.py.

Protocol. Case verification and every static preparation (prepared contexts, Fortran build and input files) are timed apart
and never part of a sample. Every SYRUP contender then records (a) `first_call`: ONE first short run (end = max dt, fresh
state; wall time INCLUDING JIT compilation, first-step and driver setup -- NOT an isolated compiler cost), validated after its
timer; (b) `warmup`: ONE complete untimed storm, whose evolution wall time and guard (validation) time are recorded
separately (neither is a pure compile or warm-loop time); then `--rounds` rounds of complete storms in `--order` (balanced = forward/reverse/forward..., or forward / reverse only), each
from a FRESH state. Each sample is validated AFTER its own timer and BEFORE it counts. Final fields, the hydrograph and the
budgets of the last qualified sample are captured OUTSIDE every timer and compared to the reference contender (default
`bisection_numba`) and pairwise (NumPy vs Numba of the same solver, against the declared backend water bounds rtol 2e-12 /
atol 1e-14); deviations are REPORTED quantitatively, never judged and never absorbed into a tolerance. A separate untimed
diagnostic storm per Newton contender (unless `--no-newton-diagnostics`) records the Newton pass statistics. The slow NumPy
bisection can be run in a separate invocation (`--contenders`) and order. Failures are recorded as failures with their
reason; no time is reported for a failed or relabelled run, and a different dt is never substituted.

Inside a SYRUP timer: the scheduler's step loop with its required validation, accumulation, report rows and the final
synchronisation. Inside the Fortran timer: the driver's step loop. The workloads are NOT identical (the originals keep their
stale-inflow/bracket behaviour and are not claimed conservative; the SYRUP forms close a water budget): no ratio is an
algorithmic claim. Nothing here was run by its author (file-only tools); Codex records results.
"""

from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for sub in ("src", "benchmarks/hydraulic_candidates", "benchmarks/rfid"):
    sys.path.insert(0, str(ROOT / sub))

SYRUP_CONTENDERS = ("bisection_numpy", "bisection_numba", "newton_numpy", "newton_numba")
CUDA_CONTENDERS = ("bisection_cuda", "newton_cuda")  # explicit only: never in the default contender list
FORTRAN_CONTENDERS = ("fortran_newton", "fortran_bisection")
ALL_CONTENDERS = (*SYRUP_CONTENDERS, *CUDA_CONTENDERS, *FORTRAN_CONTENDERS)
CUDA_MODES = ("auto", "fused", "split")
IROUTE = {"fortran_newton": 2, "fortran_bisection": 5}
ORDERS = ("balanced", "forward", "reverse")
CASES = {"plot1": {"end_s": 5400.0, "bisection_iterations": 40}, "rfid": {"end_s": 2700.0, "bisection_iterations": 64}}
MIN_DT_S = 1.0 / 1024.0
MAX_STEPS = 1_000_000
BACKEND_RTOL, BACKEND_ATOL = 2.0e-12, 1.0e-14  # the declared water bounds of the CPU backends (tests/phase7h)
FIELDS = ("depth_m", "soil_water_m", "discharge_m2_s")


def validate_args(args) -> list[str]:
    names = [n for n in args.contenders.split(",") if n]
    if not names or len(set(names)) != len(names) or any(n not in ALL_CONTENDERS for n in names):
        raise ValueError(f"--contenders must be a non-empty duplicate-free subset of {ALL_CONTENDERS}")
    if args.case == "plot1" and any(n in FORTRAN_CONTENDERS for n in names):
        raise ValueError("the Fortran contenders are supported for --case rfid only: the original-routine driver is an RFID "
                         "adapter (fixed-Ksat model 1, zero pavement) and the actual Plot 1 columns are model 2 / pavement "
                         "(unsupported harness configuration in this task; see the module docstring)")
    if any(n in FORTRAN_CONTENDERS for n in names) and not (args.fortran_build_dir or args.fortran_exe):
        raise ValueError("Fortran contenders need --fortran-build-dir (NEW, built here) or --fortran-exe (a timing build record)")
    if args.fortran_build_dir and args.fortran_exe:
        raise ValueError("give --fortran-build-dir OR --fortran-exe, not both")
    for label, value in (("--end-s", args.end_s), ("--max-dt-s", args.max_dt_s), ("--report-every-s", args.report_every_s)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    if args.max_dt_s < MIN_DT_S:
        raise ValueError(f"--max-dt-s must be >= the drivers' retry floor {MIN_DT_S} s")
    ratio = args.end_s / args.max_dt_s
    if ratio != round(ratio) or float(np.float32(args.max_dt_s)) != args.max_dt_s:
        raise ValueError("--end-s must be an integer multiple of --max-dt-s and dt exactly representable in REAL32 "
                         "(the fixed-step Fortran driver needs both)")
    if ratio > MAX_STEPS:
        raise ValueError(f"--end-s / --max-dt-s = {ratio:g} steps exceeds the bound {MAX_STEPS}")
    cadence = args.report_every_s / args.max_dt_s
    if cadence != round(cadence) or cadence < 1:
        raise ValueError("--report-every-s must be an integer multiple of --max-dt-s")
    if isinstance(args.rounds, bool) or args.rounds < 1:
        raise ValueError("--rounds must be an int >= 1")
    for label, value, hi in (("--bisection-iterations", args.bisection_iterations, 200),
                             ("--newton-max-iterations", args.newton_max_iterations, 1000)):
        if value is not None and (isinstance(value, bool) or not 1 <= value <= hi):
            raise ValueError(f"{label} must be an int in [1, {hi}]")
    if getattr(args, "cuda_mode", "auto") not in CUDA_MODES:
        raise ValueError(f"--cuda-mode must be one of {CUDA_MODES}")
    if args.reference is None:
        args.reference = next((n for n in ("bisection_numba", "bisection_numpy", "newton_numba", "newton_numpy",
                                           "bisection_cuda", "newton_cuda") if n in names), names[0])
    if args.reference not in names:
        raise ValueError(f"--reference {args.reference!r} must be one of the selected contenders")
    return names


def round_orders(names: list[str], rounds: int, order: str) -> list[list[str]]:
    if order == "forward":
        return [list(names) for _ in range(rounds)]
    if order == "reverse":
        return [list(reversed(names)) for _ in range(rounds)]
    return [list(names) if k % 2 == 0 else list(reversed(names)) for k in range(rounds)]


class NewtonStats:
    """Untimed aggregation of `RouteStep.root_stats` over the accepted steps of one diagnostic storm."""

    def __init__(self):
        self.steps = 0
        self.max_passes = 0
        self.total_passes = 0
        self.iterated_cell_steps = 0
        self.safeguard_steps = 0
        self.fallback_cell_steps = 0

    def add(self, step):
        s = step.route.root_stats
        self.steps += 1
        self.max_passes = max(self.max_passes, s["max_newton_iterations"])
        self.total_passes += s["total_newton_iterations"]
        self.iterated_cell_steps += s["iterated_cells"]
        self.safeguard_steps += s["bisection_safeguard_steps"]
        self.fallback_cell_steps += s["fallback_cells"]

    def record(self):
        mean = self.total_passes / self.iterated_cell_steps if self.iterated_cell_steps else None
        return {"accepted_steps_seen": self.steps, "max_passes_of_any_cell": self.max_passes,
                "mean_passes_per_iterated_cell_step": mean, "iterated_cell_steps": self.iterated_cell_steps,
                "bisection_safeguard_steps": self.safeguard_steps, "fallback_cell_steps": self.fallback_cell_steps}


def build_syrup_runner(name, dt, verified, args, inputs_factory, collector=None):
    """`(run, record, inputs, provenance)` like `compare_plot1.build_runner`. `run(end_s, snapshots)` returns DEVICE (here
    host) objects from a FRESH `initial_state`. `collector` (diagnostic storms only) receives every accepted CoupledStep."""
    import compare_plot1 as cmp

    from maple_syrup import storm

    solver, form = name.split("_")
    inputs = inputs_factory(verified, "cupy" if form == "cuda" else "numpy", with_geometry=False)
    kwargs = {"bisection_iterations": args.bisection_iterations, "root_solver": solver}
    if solver == "newton":
        kwargs["newton_max_iterations"] = args.newton_max_iterations
    control = storm.StormControl(max_dt_s=dt, implementation={"numba": "numba", "cuda": "cuda"}.get(form, "array"),
                                 **kwargs).validated()
    t0 = time.perf_counter()
    cuda_context = None
    if form == "cuda":
        from maple_syrup import hydrology_cuda as hc

        cuda_context = hc.prepare_cuda_hydrology(inputs.graph, inputs.params, mode=getattr(args, "cuda_mode", "auto"))
        newton_load = hc.load_newton_kernels() if solver == "newton" else None
        step_override = None  # the scheduler steps the prepared device context directly; no per-step wrapper
        provenance = {"control": dataclasses.asdict(control), "context": cuda_context.summary(),
                      "kernels": {**hc.kernel_provenance(), "selected_root_solver": solver},
                      "newton_kernel_load_s": None if newton_load is None else newton_load["seconds"],
                      "newton_counters": "not collected: the device production step is uninstrumented (root_stats None)"}
    elif form == "numba":
        from maple_syrup import hydrology_numba as hn

        prepared = hn.prepare_hydrology(inputs.graph, inputs.params)

        def step_override(graph, params, rate, state, dt_, control_):
            step = hn.prepared_coupled_step(prepared, rate, state, dt_, control_)
            if collector is not None:
                collector.add(step)
            return step

        provenance = {"control": dataclasses.asdict(control), "context": prepared.summary(),
                      "kernels": "recorded after the warm-up compiled them"}
    else:
        original = storm.coupled_step

        def step_override(graph, params, rate, state, dt_, control_):
            step = original(graph, params, rate, state, dt_, control_)
            if collector is not None:
                collector.add(step)
            return step

        provenance = {"control": dataclasses.asdict(control), "context": "NumPy namespace; no prepared context"}
        if collector is None:
            step_override = None  # the timed NumPy form calls the unmodified storm.coupled_step
    record = {"preparation_s": time.perf_counter() - t0}

    def run(end_s, snapshots):
        return cmp.run_legacy_segments(inputs.graph, inputs.params, inputs.field, inputs.schedule, inputs.depth0, inputs.soil0,
                                       control, end_s, snapshots, args.report_every_s, xp=inputs.xp, step_override=step_override,
                                       cuda_context=cuda_context)

    return run, record, inputs, provenance


def timed_validated(run, guard, end_s):
    """One fresh-state run, its evolution wall time, then its validation (guard) time measured apart."""
    t0 = time.perf_counter()
    raw = run(end_s, [])
    evolution = time.perf_counter() - t0
    t1 = time.perf_counter()
    checked = guard.validate(raw, end_s)
    return {"evolution_wall_s": evolution, "guard_wall_s": time.perf_counter() - t1, "checked": checked}


def deviation(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    return {"max_abs": float(d.max()), "max_abs_over_field_max": float(d.max() / scale) if scale > 0.0 else None,
            "rel_l2": float(np.linalg.norm(a - b) / np.linalg.norm(b)) if np.linalg.norm(b) > 0.0 else None,
            "within_backend_bounds": bool(np.allclose(a, b, rtol=BACKEND_RTOL, atol=BACKEND_ATOL)),
            "bitwise_equal": bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))}


def fortran_grid(cells: np.ndarray, shape: tuple[int, int]) -> dict:
    """Final cells of the original driver ((i, j) 1-based legacy north-first; depth m, cum_inf m, q m2/s) on the SYRUP
    south-first interior grid (inactive cells are 0 in both)."""
    ny = shape[0]
    out = {k: np.zeros(shape) for k in FIELDS}
    r = ny + 1 - cells[:, 0].astype(np.int64)
    c = cells[:, 1].astype(np.int64) - 2
    for key, col in zip(FIELDS, (2, 3, 4), strict=True):
        out[key][r, c] = cells[:, col]
    return out


def compare_finals(name, last, ref_name):
    """Final-field/hydrograph/scalar deviation of one contender against another (both captured outside every timer)."""
    a, b = last[name], last[ref_name]
    out = {"reference": ref_name, "fields": {}}
    fa, fb = a["finals"], b["finals"]
    for key in FIELDS:
        if key in fa and key in fb:
            out["fields"][key] = deviation(np.asarray(fa[key], dtype=np.float64), np.asarray(fb[key], dtype=np.float64))
    for key in ("export_m3", "peak_outlet_m3_s", "time_of_peak_s"):
        if a.get(key) is not None and b.get(key) is not None:
            out[key] = {"value": a[key], "reference": b[key], "diff": a[key] - b[key],
                        "rel_diff": (a[key] - b[key]) / b[key] if b[key] else None}
    ra, rb = a.get("series_q"), b.get("series_q")
    if ra is not None and rb is not None and len(ra) == len(rb) and np.linalg.norm(rb) > 0.0:
        out["outlet_hydrograph_rel_l2"] = float(np.linalg.norm(np.asarray(ra) - np.asarray(rb)) / np.linalg.norm(rb))
        out["outlet_hydrograph_max_abs_m3_s"] = float(np.max(np.abs(np.asarray(ra) - np.asarray(rb))))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=tuple(CASES), required=False)
    parser.add_argument("--case-dir")
    parser.add_argument("--output-dir", type=Path, help="NEW directory (an existing path is refused)")
    parser.add_argument("--contenders", default=",".join(SYRUP_CONTENDERS))
    parser.add_argument("--list-contenders", action="store_true")
    parser.add_argument("--reference", help="default: the first of bisection_numba, bisection_numpy, newton_numba, "
                        "newton_numpy among the selected contenders")
    parser.add_argument("--end-s", type=float)
    parser.add_argument("--max-dt-s", type=float, default=1.0)
    parser.add_argument("--report-every-s", type=float, default=60.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--order", choices=ORDERS, default="balanced")
    parser.add_argument("--bisection-iterations", type=int, help="default per case: Plot 1 40, RFID 64 (pit storage needs 64)")
    parser.add_argument("--newton-max-iterations", type=int, default=50)
    parser.add_argument("--cuda-mode", choices=CUDA_MODES, default="auto",
                        help="launch structure of the prepared CUDA hydrology for the *_cuda contenders (default auto)")
    parser.add_argument("--no-newton-diagnostics", action="store_true")
    parser.add_argument("--fortran-build-dir", type=Path)
    parser.add_argument("--fortran-exe", type=Path)
    parser.add_argument("--allow-maple-source-change", action="store_true")
    args = parser.parse_args(argv)
    if args.list_contenders:
        print("\n".join(ALL_CONTENDERS))
        return 0
    if not (args.case and args.case_dir and args.output_dir):
        parser.error("--case, --case-dir and --output-dir are required")
    args.end_s = CASES[args.case]["end_s"] if args.end_s is None else args.end_s
    args.bisection_iterations = (CASES[args.case]["bisection_iterations"] if args.bisection_iterations is None
                                 else args.bisection_iterations)
    try:
        names = validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"refusing to overwrite {output_dir}")

    import compare_plot1 as cmp
    import fortran_timing as ft
    import run_rfid_timing as rrt
    from maple.core.backend import read_transfer_counters

    from maple_syrup import hydrology_numba as hn
    from maple_syrup import routing_newton, routing_newton_cuda
    from maple_syrup.case_import import (
        Plot1ImportError,
        _refuse_output,
        verify_plot1_case,
    )
    from maple_syrup.column_experiment import _source_digests, _syrup_provenance
    from maple_syrup.dependency import MapleDependencyError
    from maple_syrup.experimental_experiment import plot1_inputs
    from maple_syrup.provenance import environment_record
    from maple_syrup.rfid_case import rfid_inputs, verify_rfid_case

    if any(n in CUDA_CONTENDERS for n in names):
        from maple.core.backend import gpu_execution_available

        if not gpu_execution_available():
            parser.error("a *_cuda contender needs CuPy and a CUDA device; none is available (no CPU substitution)")
    verify = verify_rfid_case if args.case == "rfid" else verify_plot1_case
    inputs_factory = rfid_inputs if args.case == "rfid" else plot1_inputs
    t_verify = time.perf_counter()
    try:
        verified = verify(args.case_dir, allow_maple_source_change=args.allow_maple_source_change)
        syrup = _syrup_provenance()
        dependency = verified.maple_dependency
        mahleran = Path(verified.report["mahleran"]["root"] if args.case == "rfid"
                        else verified.report["recipe"]["mahleran_root"]).resolve()
        protected = {"MAPLE source": dependency.source_root, "MAPLE package": dependency.package_dir,
                     "recorded MAHLERAN": mahleran, "bound case": Path(args.case_dir).resolve(),
                     "SYRUP package": Path(syrup["package_dir"]).resolve()}
        if args.case == "plot1":
            protected["recipe"] = Path(verified.report["recipe"]["recipe_path"]).resolve().parent
        _refuse_output(output_dir, protected)
    except (Plot1ImportError, MapleDependencyError) as exc:
        parser.error(str(exc))
    verification_s = time.perf_counter() - t_verify
    package_dirs = (Path(syrup["package_dir"]), Path(dependency.package_dir))
    digests_start = _source_digests(*package_dirs)
    end_s, dt = args.end_s, args.max_dt_s
    n_steps = round(end_s / dt)
    output_dir.mkdir(parents=True)
    records: dict = {n: {"preparation": None, "samples": [], "warmup": None, "status": "pending"} for n in names}

    # ---------------- Fortran preparation (apart from every sample) ----------------
    fortran_inputs: dict = {}
    fortran_build = None
    base_inputs = None
    if any(n in FORTRAN_CONTENDERS for n in names):
        try:
            if args.fortran_build_dir:
                fortran_build = ft.build(args.fortran_build_dir, variant="timing",
                                         extra_protected={"bound case": Path(args.case_dir).resolve(),
                                                          "MAPLE source": dependency.source_root})
            else:
                fortran_build = ft.load_build(args.fortran_exe, variant="timing")
        except (RuntimeError, ValueError, OSError) as exc:
            parser.error(f"Fortran build unusable: {exc}")
        base_inputs = rfid_inputs(verified, "numpy", with_geometry=False)
        arrays = ft.common_arrays(base_inputs)
        rates = np.array([base_inputs.schedule.rate_after_m_per_s(k * dt) * 1000.0 for k in range(n_steps)])
        rain_expected = base_inputs.schedule.depth_m(0.0, end_s) * float(base_inputs.host["rainfall_scale"].sum()) * base_inputs.area
        fdir = output_dir / "fortran"
        fdir.mkdir()
        cadence_steps = round(args.report_every_s / dt)
        active_ij = ft.expected_active_ij(arrays)

        def expect(name):
            return {"n_steps": n_steps, "dt_s": dt, "report_every_steps": cadence_steps, "iroute": IROUTE[name],
                    "rain_expected_m3": rain_expected, "active_ij": active_ij}

        for n in names:
            if n in FORTRAN_CONTENDERS:
                path = fdir / f"input_{n}.dat"
                t0 = time.perf_counter()
                digest = ft.write_input(path, arrays, dt_s=dt, rates_mm_s=rates, iroute=IROUTE[n],
                                        report_every_steps=cadence_steps)
                records[n]["preparation"] = {"input_write_s": time.perf_counter() - t0, "input_sha256": digest}
                fortran_inputs[n] = path

    # ---------------- SYRUP preparation + one complete untimed warm-up storm ----------------
    prepared, guards = {}, {}
    for n in names:
        if n in FORTRAN_CONTENDERS:
            continue
        try:
            run, prep, inputs, provenance = build_syrup_runner(n, dt, verified, args, inputs_factory)
            guard = cmp.RunGuard(inputs, package_dirs, n)
            first = timed_validated(run, guard, dt)  # first short run: JIT + first step/driver setup, NOT a pure compile
            warm = timed_validated(run, guard, end_s)  # complete untimed warm-up, evolution and guard timed separately
            records[n].update(
                preparation=prep, provenance=provenance, status="ready", root_solver=n.split("_")[0],
                first_call={"end_s": dt, "evolution_wall_s": first["evolution_wall_s"], "guard_wall_s": first["guard_wall_s"],
                            "accepted": first["checked"]["host"]["accepted"],
                            "note": "first-call time INCLUDING JIT compilation, first step and driver setup; not isolated "
                                    "compiler cost; validated after its timer"},
                warmup={"end_s": end_s, "evolution_wall_s": warm["evolution_wall_s"], "guard_wall_s": warm["guard_wall_s"],
                        "accepted": warm["checked"]["host"]["accepted"], "rejected": warm["checked"]["host"]["rejected"],
                        "note": "complete untimed storm after the first call; evolution and validation timed apart; not a "
                                "pure compilation or steady-state loop time"})
            prepared[n], guards[n] = run, guard
        except Exception as exc:  # noqa: BLE001 - a failing contender is a recorded failure, never a time
            records[n].update(status="failed_preparation_or_warmup", reason=f"{type(exc).__name__}: {exc}")
    for n in names:
        if n in FORTRAN_CONTENDERS:
            warm = ft.run_once(Path(fortran_build["executable"]), fortran_inputs[n], fdir / f"warmup_{n}",
                               expected=expect(n), build_record=fortran_build)
            if warm["status"] != "complete":
                records[n].update(status="failed_warmup", reason=warm.get("reason", warm["status"]))
            else:
                records[n].update(status="ready", warmup={"loop_s": warm["loop_s"], "process_wall_s": warm["process_wall_s"]})

    # ---------------- timed rounds; validation and capture OUTSIDE every timer ----------------
    last: dict = {}
    live = [n for n in names if records[n]["status"] == "ready"]
    plan = round_orders(live, args.rounds, args.order)
    for round_index, order in enumerate(plan):
        for n in order:
            rec = records[n]
            if n in FORTRAN_CONTENDERS:
                out = ft.run_once(Path(fortran_build["executable"]), fortran_inputs[n], fdir / f"{n}_round{round_index}",
                                  expected=expect(n), build_record=fortran_build)
                if out["status"] != "complete":
                    rec["samples"].append({"round": round_index, "status": "failed", "reason": out.get("reason", out["status"])})
                    continue
                bud = ft.budget(out, base_inputs, rain_expected)
                rec["samples"].append({
                    "round": round_index, "status": "ok", "wall_s": out["loop_s"], "process_wall_s": out["process_wall_s"],
                    "max_rss_kib": out["max_rss_kib"], "steps": n_steps, "export_m3": bud["export_m3"],
                    "peak_outlet_m3_s": out["peak_outlet_m3_s"], "time_of_peak_s": out["time_of_peak_s"], "budget": bud,
                    "output_sha256": out["output_sha256"]})
                last[n] = {"finals": fortran_grid(out["final_cells"], base_inputs.graph.shape), "export_m3": bud["export_m3"],
                           "peak_outlet_m3_s": out["peak_outlet_m3_s"], "time_of_peak_s": out["time_of_peak_s"],
                           "series_q": out["history"][:, 6].tolist()}
            else:
                try:
                    wall, raw, transfers = cmp.time_sample(prepared[n], end_s, [], read_transfer_counters)
                    checked = guards[n].validate(raw, end_s)  # AFTER the timer, BEFORE the sample may count
                    s = cmp.sample_record(wall, transfers, checked)
                    host = checked["host"]
                    s.update(round=round_index, status="ok", accepted=host["accepted"], rejected=host["rejected"],
                             export_m3=host["export"], peak_outlet_m3_s=host["peak_q"], time_of_peak_s=host["peak_t"])
                    rec["samples"].append(s)
                    last[n] = {"finals": checked["finals"], "export_m3": host["export"], "peak_outlet_m3_s": host["peak_q"],
                               "time_of_peak_s": host["peak_t"], "series_q": host["rows"][:, 8].tolist(),
                               "rows": host["rows"]}
                except Exception as exc:  # noqa: BLE001
                    rec["samples"].append({"round": round_index, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})

    # ---------------- untimed Newton pass statistics ----------------
    if not args.no_newton_diagnostics:
        for n in [x for x in live if x.startswith("newton_") and not x.endswith("_cuda")]:
            try:
                stats = NewtonStats()
                run, _prep, inputs, _prov = build_syrup_runner(n, dt, verified, args, inputs_factory, collector=stats)
                guard = cmp.RunGuard(inputs, package_dirs, f"{n} diagnostics")
                guard.validate(run(end_s, []), end_s)
                records[n]["newton_statistics"] = stats.record()
            except Exception as exc:  # noqa: BLE001
                records[n]["newton_statistics"] = {"failed": f"{type(exc).__name__}: {exc}"}

    for n, rec in records.items():
        ok = [s for s in rec["samples"] if s.get("status") == "ok"]
        rec["n_ok"], rec["n_failed"] = len(ok), len(rec["samples"]) - len(ok)
        if ok:
            rec["timing"] = rrt.summarize([s["wall_s"] for s in ok])
            if n in FORTRAN_CONTENDERS:
                rec["process_timing"] = rrt.summarize([s["process_wall_s"] for s in ok])
                rec["max_rss_kib"] = max(s["max_rss_kib"] for s in ok)
        if rec["status"] == "ready":
            rec["status"] = "completed_with_failures" if rec["n_failed"] else "completed"
    for n in ("bisection_numba", "newton_numba"):  # EVERY selected compiled solver gets its own actual metadata
        if n in records and records[n].get("provenance") is not None:
            records[n]["provenance"]["kernels"] = {**hn.kernel_provenance(), "selected_root_solver": n.split("_")[0]}

    # ---------------- deviations (reported, not judged) ----------------
    comparisons: dict = {}
    ref = args.reference
    if ref in last:
        for n in last:
            if n != ref:
                comparisons[f"{n}_vs_{ref}"] = compare_finals(n, last, ref)
    for a, b in (("bisection_numpy", "bisection_numba"), ("newton_numpy", "newton_numba"), ("newton_numba", "bisection_numba"),
                 ("newton_numpy", "bisection_numpy"), ("bisection_cuda", "bisection_numba"), ("newton_cuda", "newton_numba"),
                 ("newton_cuda", "bisection_cuda"), ("fortran_newton", "newton_numba"),
                 ("fortran_bisection", "bisection_numba"), ("fortran_newton", "fortran_bisection")):
        if a in last and b in last and f"{a}_vs_{b}" not in comparisons:
            comparisons[f"{a}_vs_{b}"] = compare_finals(a, last, b)
    speed = {}
    for n, rec in records.items():
        if rec.get("timing"):
            speed[n] = rec["timing"]["median_s"]

    digests_end = _source_digests(*package_dirs)
    if digests_end != digests_start:
        raise RuntimeError("SYRUP/MAPLE source changed during the benchmark; no result written")
    memory = {"host_maxrss_kib_so_far": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "note": "process-wide maximum so far; run contenders in separate invocations for per-contender peaks"}
    payload = {
        "schema": "maple_syrup.newton_cpu.compare.v1", "case": args.case, "python": platform.python_version(),
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "routing_newton_sha256": hashlib.sha256(Path(routing_newton.__file__).read_bytes()).hexdigest(),
        "routing_newton_cuda_sha256": hashlib.sha256(Path(routing_newton_cuda.__file__).read_bytes()).hexdigest(),
        "environment": environment_record(), "syrup_provenance": syrup, "maple_provenance": verified.maple_provenance,
        "case_binding": dict(verified.binding), "case_checks": verified.checks, "case_verification_s": verification_s,
        "source_digests": {"start": digests_start, "end": digests_end, "stable": True},
        "fortran_build": fortran_build, "orders": plan, "records": records, "median_wall_s": speed,
        "comparisons": comparisons, "memory": memory,
        "scope": ("water only, fixed terrain, no splash, one verified case, full storm; SYRUP forms close a water budget per "
                  "sample, the original Fortran routines keep their own behaviour and are not claimed conservative; "
                  "deviations between root solvers are reported, not judged; no sediment/erosion/wind/disk-restart claim; GPU "
                  "contenders are the water-only prepared CUDA hydrology on one device"),
    }
    with (output_dir / "comparison.json").open("x") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    with (output_dir / "final_fields.npz").open("xb") as handle:
        np.savez(handle, **{f"{n}_{k}": np.asarray(v) for n, d in last.items() for k, v in d["finals"].items()})
    with (output_dir / "hydrographs.npz").open("xb") as handle:
        np.savez(handle, **{f"{n}_rows": d["rows"] for n, d in last.items() if "rows" in d},
                 **{f"{n}_outlet_m3_s": np.asarray(d["series_q"]) for n, d in last.items()})
    print(json.dumps({n: {"status": r["status"], "median_s": r.get("timing", {}).get("median_s")} for n, r in records.items()},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
