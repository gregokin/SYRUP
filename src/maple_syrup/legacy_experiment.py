"""Default run workflow: Plot1 frozen-terrain Python/Numba replay of the MAHLERAN LEGACY transport.

Promoted from benchmarks/phase7e/run_legacy_benchmark.py (numerics unchanged). Entry points: `maple-syrup-legacy`,
`python -m maple_syrup.legacy_experiment`, and `maple-syrup-benchmark` (default --transport-scheme legacy).

Reuses SYRUP's hydrology (rainfall column + Crank-Nicolson water routing, identical to the matched
benchmark) and the wet MAHLERAN laws (sediment_physics_step) on the imported MAPLE case, then routes
sediment with maple_syrup.legacy_transport: source-based flow_distrib deposition and the legacy
Crank-Nicolson pool with explicit clipping-source accounting. NO MAPLE bed exchange happens: the
legacy has unlimited supply and fixed composition, so the initial active-layer composition is held
fixed. Outputs per-step per-class ledgers in the same columns as the Fortran audit
(outputs/phase7b/mahleran_ledger_run_v3/Output/syrup_sediment_ledger.dat) for direct comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from maple_syrup import legacy_transport as L
from maple_syrup.benchmark_experiment import load_applied_rainfall
from maple_syrup.case_import import verify_plot1_case
from maple_syrup.column_experiment import _source_digests, _syrup_provenance
from maple_syrup.legacy_physics_numba import legacy_physics_step, prepare_legacy_physics
from maple_syrup.provenance import environment_record
from maple_syrup.routing import RoutingStepRejected
from maple_syrup.routing_numba import numba_available
from maple_syrup.sediment_experiment import prepare_verified_sediment_case
from maple_syrup.sediment_physics import (
    REGIME_CODES,
    recession_velocity,
    sediment_physics_step,
)
from maple_syrup.storm import StormControl, coupled_step, plan_boundaries

LEDGER_COLUMNS = ("pickup_kg", "deposition_active_kg", "deposition_outside_active_kg", "effective_clip_source_kg",
                  "old_mobile_kg", "new_mobile_kg", "cn_export_kg", "endpoint_export_kg", "outlet_flux_kg_s")


LEGACY_DEFAULT_CASE = "outputs/plot1"
LEGACY_DEFAULT_RAINFALL = "outputs/phase7/mahleran_reference_audit/applied_rainfall.csv"
LEGACY_DEFAULT_LEDGER = "outputs/phase7b/mahleran_ledger_run_v3/Output/syrup_sediment_ledger.dat"
LEGACY_DEFAULT_REFERENCE_RUN = "outputs/phase7/mahleran_deterministic_ksat_run"

# Options of the conservative (characteristic/upwind) benchmark CLI that the legacy replay cannot honour.
# They default to None here so that an explicitly supplied value is detected and refused, never ignored.
_UNSUPPORTED = {
    "--phase-bins": "multi-bin characteristic transport (deferred to Phase 7g; use --transport-scheme characteristic)",
    "--transport-implementation": "characteristic kernel selection",
    "--sediment-courant-max": "sediment Courant control (the legacy replay has none)",
    "--max-transport-substeps": "transport substeps (the legacy replay has none)",
    "--min-dt-s": "adaptive time stepping (the legacy replay is fixed dt = 1 s)",
    "--max-retries": "step retries (the legacy program has no retry)",
    "--max-steps": "step limit",
    "--max-report-rows": "hydrograph report rows",
    "--expected-ksat-mm-s": "conductivity check (the matched benchmark only)",
    "--mahleran-root": "MAHLERAN root override (the matched benchmark only)",
    "--expected-maple-root": "MAPLE root pin (the matched benchmark only)",
}

DESCRIPTION = (
    "MAPLE-SYRUP default run: Python/Numba replay of the MAHLERAN LEGACY sediment transport on frozen Plot1 terrain "
    "(dt = 1 s, no splash, fixed routing). SYRUP hydrology + wet MAHLERAN laws + source-based flow_distrib deposition "
    "and the legacy Crank-Nicolson pool with explicit clipping-source accounting. Scientific status: fixed composition, "
    "UNLIMITED supply, explicit clipping source, NO evolving MAPLE bed. It is NOT a conservative complete-event, "
    "restart or wind-handoff model and its legacy behaviour is not MAPLE authority. The conservative multi-bin "
    "characteristic model is kept and selected explicitly with --transport-scheme characteristic (or upwind); multi-bin "
    "convergence/performance work is deferred (Phase 7g).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m maple_syrup.legacy_experiment", description=DESCRIPTION)
    p.add_argument("--output", "--output-dir", dest="output", type=Path, required=True,
                   help="NEW output directory (legacy_ledger.npz, legacy_summary.json)")
    p.add_argument("--case", "--case-dir", dest="case", default=LEGACY_DEFAULT_CASE)
    p.add_argument("--applied-rainfall", default=LEGACY_DEFAULT_RAINFALL)
    p.add_argument("--depth-time-level", choices=("previous", "current"), default="previous",
                   help="legacy evaluates the laws with the PREVIOUS step's depth d(1) and the new velocity")
    p.add_argument("--end-s", type=float, default=None)
    p.add_argument("--implementation", choices=("numba", "array"), default="numba",
                   help="SYRUP water-routing implementation. numba (default) requires compiled water AND compiled "
                        "legacy sediment kernels, with no silent Python fallback. array is a declared slow "
                        "diagnostic of the water sweep only.")
    p.add_argument("--physics-implementation", choices=("numba", "array"), default=None,
                   help="wet physical-law evaluation. Default: the same as --implementation. numba = fused compiled "
                        "CPU kernel on a prepared frozen-composition context (explicit error if Numba is missing, no "
                        "silent fallback). array = the NumPy sediment_physics_step reference (declared slower; may be "
                        "combined with compiled hydrology and legacy transport for controlled comparisons).")
    p.add_argument("--hydrology-implementation", choices=("prepared", "reference"), default=None,
                   help="rainfall/infiltration/routing step. Default: prepared with --implementation numba, "
                        "reference with --implementation array. prepared = fused compiled CPU kernels on a "
                        "prepared fixed-terrain context (same equations, bisection and checks; requires "
                        "--implementation numba and host NumPy, explicit error if Numba is missing, no silent "
                        "fallback). reference = the NumPy/CuPy-reference storm.coupled_step (with the numba or "
                        "array sweep per --implementation); declared slower, kept for controlled comparisons.")
    p.add_argument("--fortran-ledger", type=Path, default=Path(LEGACY_DEFAULT_LEDGER))
    p.add_argument("--reference-run", "--reference-run-dir", dest="reference_run", type=Path,
                   default=Path(LEGACY_DEFAULT_REFERENCE_RUN))
    p.add_argument("--allow-maple-source-change", action="store_true")
    # accepted only at the single supported value, so the generic CLI spelling works
    p.add_argument("--transport-scheme", choices=("legacy",), default="legacy")
    p.add_argument("--max-dt-s", type=float, default=None, help="only 1 is supported")
    p.add_argument("--report-every-s", type=float, default=None, help="only 1 is supported")
    p.add_argument("--backend", default=None, help="only numpy is supported (no GPU)")
    for name, why in _UNSUPPORTED.items():
        p.add_argument(name, default=None, help=f"unsupported for legacy: {why}")
    return p


def check_supported(args: argparse.Namespace) -> list[str]:
    """Return refusal messages for explicitly supplied controls the legacy replay cannot honour."""
    problems = []
    for name, why in _UNSUPPORTED.items():
        if getattr(args, name[2:].replace("-", "_")) is not None:
            problems.append(f"{name} is not supported by the legacy transport ({why})")
    if args.max_dt_s is not None and args.max_dt_s != 1.0:
        problems.append("--max-dt-s must be 1 for the legacy transport (the Fortran dt)")
    if args.report_every_s is not None and args.report_every_s != 1.0:
        problems.append("--report-every-s must be 1 for the legacy transport")
    if args.backend is not None and args.backend != "numpy":
        problems.append(f"--backend {args.backend} is not supported by the legacy transport (CPU only, no GPU)")
    return problems


def resolve_physics_implementation(args: argparse.Namespace) -> str:
    """The wet-law implementation: explicit --physics-implementation, else the --implementation choice."""
    return args.physics_implementation or args.implementation


def resolve_hydrology_implementation(args: argparse.Namespace) -> str:
    """The hydrology step: explicit --hydrology-implementation, else prepared with compiled water, reference
    with the array water diagnostic."""
    explicit = getattr(args, "hydrology_implementation", None)
    return explicit or ("prepared" if args.implementation == "numba" else "reference")


def require_compiled(implementation: str, physics_implementation: str | None = None,
                     hydrology_implementation: str | None = None) -> list[str]:
    """Numba-default path: compiled water AND compiled sediment kernels, never a silent Python fallback.
    A compiled wet-law evaluation (default with --implementation numba) additionally needs Numba, and so does
    the prepared hydrology (default with --implementation numba), which also refuses array water."""
    physics = physics_implementation or implementation
    hydrology = hydrology_implementation or ("prepared" if implementation == "numba" else "reference")
    problems = []
    if implementation == "numba":
        if not numba_available():
            problems.append("Numba is not importable, so the compiled water sweep is unavailable")
        if L.KERNEL_IMPLEMENTATION != "numba":
            problems.append("the legacy sediment kernels are running as pure Python (Numba import failed)")
    if physics == "numba" and not numba_available():
        problems.append("Numba is not importable, so the compiled wet physical laws are unavailable")
    if hydrology == "prepared" and implementation == "numba" and not numba_available():
        problems.append("Numba is not importable, so the prepared compiled hydrology is unavailable")
    refusals = [p + "; refusing the numba legacy replay (use --implementation array only as a declared diagnostic)"
                for p in problems]
    if hydrology == "prepared" and implementation != "numba":
        refusals.append(f"--hydrology-implementation prepared requires --implementation numba (got "
                        f"{implementation}); the prepared hydrology is the compiled numba sweep and there is no "
                        "silent fallback (use --hydrology-implementation reference for the array diagnostic)")
    return refusals


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    problems = check_supported(args) + require_compiled(args.implementation, args.physics_implementation,
                                                        args.hydrology_implementation)
    if problems:
        for msg in problems:
            print(f"legacy replay refused: {msg}", file=sys.stderr)
        return 2
    try:
        run(args)
    except SystemExit as exc:  # the replay reports its own validation failures this way
        if exc.code in (None, 0):
            return 0
        print(f"legacy replay failed: {exc.code}", file=sys.stderr)
        return 1
    return 0


def run(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    if out.exists():
        raise SystemExit("new output directory required")
    if args.end_s is not None and not (math.isfinite(args.end_s) and args.end_s > 0.0):
        raise SystemExit("--end-s must be finite and positive")
    if not args.fortran_ledger.is_file() or not (args.reference_run / "execution.json").is_file():
        raise SystemExit("actual Fortran reference ledger and reference run execution.json are required")
    reference = {"fortran_ledger": str(args.fortran_ledger.resolve()),
                 "fortran_ledger_sha256": hashlib.sha256(args.fortran_ledger.read_bytes()).hexdigest(),
                 "reference_run": str(args.reference_run.resolve()),
                 "reference_execution_sha256": hashlib.sha256((args.reference_run / "execution.json").read_bytes()).hexdigest(),
                 "reference_outputs_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                              for f in sorted((args.reference_run / "Output").iterdir())}}
    t_wall0 = time.perf_counter()
    verified = verify_plot1_case(args.case, allow_maple_source_change=args.allow_maple_source_change)
    syrup_prov = _syrup_provenance()
    digests_before = _source_digests(Path(syrup_prov["package_dir"]), verified.maple_dependency.package_dir)
    prepared = prepare_verified_sediment_case(verified, end_s=args.end_s)
    schedule, applied_record = load_applied_rainfall(args.applied_rainfall)
    state0, column, field, vegetation = prepared["state0"], prepared["column"], prepared["field"], prepared["vegetation"]
    sediment, graph, grid = prepared["sediment"], prepared["graph"], state0.grid
    end = float(prepared["end"])
    if not (math.isfinite(end) and end > 0.0):
        raise SystemExit(f"invalid end time {end}")
    ny, nx, nc = graph.shape[0], graph.shape[1], sediment.n_classes
    n = ny * nx
    storm_control = StormControl(max_dt_s=1.0, min_dt_s=1.0, max_retries=1, implementation=args.implementation).validated()  # rejections abort below
    boundaries = plan_boundaries(schedule, 0.0, end, 1.0)
    if not np.allclose(np.diff(np.concatenate(([0.0], boundaries))), 1.0):
        raise SystemExit("legacy replay requires one-second steps (dt = 1 s, the Fortran dt)")
    net = L.legacy_network(graph)
    if np.any((np.asarray(graph.slope).reshape(-1) <= 0.0) & net.active):
        raise SystemExit("legacy diffuse branch with zero slope (sed_temp sink) is not reproduced; active zero-slope cell")
    holdings0 = np.asarray(state0.bed.active_layer.mass_kg, dtype=np.float64)  # fixed composition (legacy sed_propn)
    physics_impl = resolve_physics_implementation(args)
    physics_prep_s = 0.0
    physics_ctx = None
    if physics_impl == "numba":  # static preparation is timed separately from the loop and from JIT compilation
        t_prep0 = time.perf_counter()
        physics_ctx = prepare_legacy_physics(sediment, grid, vegetation, holdings0)
        physics_prep_s = time.perf_counter() - t_prep0
    hydrology_impl = resolve_hydrology_implementation(args)
    hydrology_prep_s = 0.0
    hydrology_ctx = None
    hydrology_kernels = None
    if hydrology_impl == "prepared":  # fixed-terrain static preparation, timed apart from the loop and from JIT
        from maple_syrup.hydrology_numba import (
            kernel_provenance,
            prepare_hydrology,
            prepared_coupled_step,
        )

        t_prep0 = time.perf_counter()
        hydrology_ctx = prepare_hydrology(graph, column)
        hydrology_prep_s = time.perf_counter() - t_prep0

        def hydrology_step(rate_field, storm_state, step_dt):
            return prepared_coupled_step(hydrology_ctx, rate_field, storm_state, step_dt, storm_control)
    else:
        def hydrology_step(rate_field, storm_state, step_dt):
            return coupled_step(graph, column, rate_field, storm_state, step_dt, storm_control)
    storm = state0.storm
    prev_depth = np.asarray(storm.depth_m, dtype=np.float64).copy()
    v_prev = np.zeros((ny, nx, nc))
    M = np.zeros((n, nc)); Q = np.zeros((n, nc)); Qin = np.zeros((n, nc))
    rate = np.empty((ny, nx))
    steps = boundaries.size
    ledger = np.zeros((steps, nc, len(LEDGER_COLUMNS)))
    water_q = np.zeros(steps); water_export = np.zeros(steps)
    cum_detach_cell = np.zeros((n, nc)); cum_depos_cell = np.zeros((n, nc)); cum_clip_cell = np.zeros((n, nc))
    regime_counts = {k: 0 for k in REGIME_CODES}
    t = 0.0
    cn_s = hydro_s = physics_s = 0.0
    first_step_s = {}
    t_loop0 = time.perf_counter()
    for row, boundary in enumerate(boundaries.tolist()):
        dt = boundary - t
        field.apply(schedule.rate_after_m_per_s(t), out=rate)
        t0 = time.perf_counter()
        try:
            hydro = hydrology_step(rate, storm, dt)
        except RoutingStepRejected as exc:
            raise SystemExit(f"water step rejected at t={t}: {exc}; the legacy program has no retry") from exc
        hydro_s += time.perf_counter() - t0
        route = hydro.route
        depth_for_laws = prev_depth if args.depth_time_level == "previous" else np.asarray(route.depth_m)
        t0 = time.perf_counter()
        if physics_ctx is not None:
            physics = legacy_physics_step(physics_ctx, depth_for_laws, route.velocity_m_s, rate, v_prev, dt)
        else:
            physics = sediment_physics_step(sediment, grid, depth_for_laws, route.velocity_m_s, rate, vegetation,
                                            holdings0, v_prev, dt)
        physics_s += time.perf_counter() - t0
        # legacy virtual velocity: the law's value where a law applies, the decayed memory elsewhere (never zeroed)
        v_used = np.where(physics.law_applies, physics.sediment_velocity_m_s, recession_velocity(v_prev, dt))
        v_used = np.where(np.asarray(graph.active)[..., None], v_used, 0.0)
        det_rate = physics.requested_pickup_kg / dt
        t0 = time.perf_counter()
        step = L.legacy_transport_step(net, det_rate, physics.deposition_rate_per_m, physics.law_applies,
                                       physics.regime, v_used, M, Q, Qin, dt)
        cn_s += time.perf_counter() - t0
        if row == 0:  # first step includes JIT compilation / warm-up of hydrology, physics and legacy kernels
            first_step_s = {"hydrology_s": hydro_s, "physics_s": physics_s, "legacy_walk_plus_cn_s": cn_s,
                            "total_s": time.perf_counter() - t_loop0}
        for k, v in physics.regime_counts.items():
            regime_counts[k] += int(v)
        ledger[row, :, 0] = step.detachment_rate_kg_s.sum(0) * dt
        ledger[row, :, 1] = step.deposition_rate_kg_s.sum(0) * dt
        ledger[row, :, 2] = step.ring_deposition_rate_kg_s * dt
        ledger[row, :, 3] = step.clipping_source_kg.sum(0)
        ledger[row, :, 4] = step.mobile_before_kg.sum(0)
        ledger[row, :, 5] = step.mobile_after_kg.sum(0)
        ledger[row, :, 6] = step.cn_export_kg
        ledger[row, :, 7] = step.endpoint_export_kg
        ledger[row, :, 8] = step.outlet_flux_kg_s
        water_q[row] = float(route.outlet_discharge_m3_s)
        water_export[row] = float(route.export_m3)
        cum_detach_cell += step.detachment_rate_kg_s * dt
        cum_depos_cell += step.deposition_rate_kg_s * dt
        cum_clip_cell += step.clipping_source_kg
        M, Q, Qin = step.mobile_after_kg, step.flux_after_kg_s, step.inflow_after_kg_s
        v_prev = v_used
        prev_depth = np.asarray(route.depth_m, dtype=np.float64).copy()
        t = boundary
        storm = replace(hydro.state, t_s=t)
    loop_s = time.perf_counter() - t_loop0
    if hydrology_ctx is not None:
        hydrology_kernels = kernel_provenance()
    residual = (ledger[:, :, 5] - ledger[:, :, 4] - (ledger[:, :, 0] - ledger[:, :, 1]) + ledger[:, :, 6] - ledger[:, :, 3])
    digests_after = _source_digests(Path(syrup_prov["package_dir"]), verified.maple_dependency.package_dir)
    if digests_after != digests_before:
        raise SystemExit(f"source changed during the replay ({digests_before} -> {digests_after}); no output written")
    if hashlib.sha256(args.fortran_ledger.read_bytes()).hexdigest() != reference["fortran_ledger_sha256"]:
        raise SystemExit("Fortran reference ledger changed during the replay; no output written")
    out.mkdir(parents=True)
    np.savez(out / "legacy_ledger.npz", t_s=boundaries, ledger=ledger, columns=np.array(LEDGER_COLUMNS),
             water_outlet_m3_s=water_q, water_export_m3=water_export,
             cumulative_detachment_kg=cum_detach_cell.reshape(ny, nx, nc),
             cumulative_deposition_kg=cum_depos_cell.reshape(ny, nx, nc),
             cumulative_clipping_source_kg=cum_clip_cell.reshape(ny, nx, nc),
             final_mobile_kg=M.reshape(ny, nx, nc), identity_residual_kg=residual)
    totals = {name: ledger[:, :, i].sum(axis=0).tolist() for i, name in enumerate(LEDGER_COLUMNS) if name not in ("old_mobile_kg", "new_mobile_kg")}
    summary = {
        "status": "legacy replay (default run workflow, frozen terrain, dt=1, no splash); NOT conservative by construction (clipping source, ring deposition, "
                  "unlimited supply, fixed composition, no evolving MAPLE bed); not a conservative complete-event, restart or wind-handoff model",
        "steps": int(steps), "dt_s": 1.0, "end_s": end, "depth_time_level": args.depth_time_level,
        "kernel_implementation": L.KERNEL_IMPLEMENTATION, "water_implementation": args.implementation,
        "hydrology_implementation": hydrology_impl,
        "physics_implementation": physics_impl,
        "totals_by_class_kg": totals,
        "totals_kg": {k: float(np.sum(v)) for k, v in totals.items()},
        "final_mobile_by_class_kg": M.sum(0).tolist(),
        "max_abs_identity_residual_kg": float(np.abs(residual).max()),
        "regime_cell_class_steps": regime_counts,
        "peak_outlet_flux_kg_s": float(ledger[:, :, 8].sum(1).max()),
        "time_of_peak_outlet_flux_s": float(boundaries[int(np.argmax(ledger[:, :, 8].sum(1)))]),
        "water": {"peak_outlet_m3_s": float(water_q.max()), "time_of_peak_s": float(boundaries[int(np.argmax(water_q))]),
                  "export_m3": float(water_export.sum())},
        "performance": {"loop_wall_s": loop_s, "hydrology_s": hydro_s, "physics_s": physics_s,
                        "legacy_walk_plus_cn_s": cn_s, "whole_process_wall_s": time.perf_counter() - t_wall0,
                        "first_step_including_warmup_s": first_step_s,
                        "steady_state_excluding_first_step_s": {"hydrology_s": hydro_s - first_step_s["hydrology_s"],
                                                                "physics_s": physics_s - first_step_s["physics_s"],
                                                                "legacy_walk_plus_cn_s": cn_s - first_step_s["legacy_walk_plus_cn_s"],
                                                                "steps": int(steps) - 1},
                        "water_implementation": args.implementation,
                        "physics_implementation": physics_impl,
                        "hydrology_implementation": hydrology_impl,
                        "hydrology_preparation_s": hydrology_prep_s,
                        "hydrology_context": None if hydrology_ctx is None else hydrology_ctx.summary(),
                        "hydrology_kernels": hydrology_kernels,
                        "physics_preparation_s": physics_prep_s,
                        "physics_context": None if physics_ctx is None else physics_ctx.summary(),
                        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        "note": "single run; first step includes JIT compilation; do not quote per-step means that include it"},
        "provenance": {"maple_syrup": syrup_prov, "maple": verified.maple_provenance,
                       "source_digests_before": digests_before, "source_digests_after": digests_after,
                       "source_stable": True,
                       "reference": reference,
                       "applied_rainfall": applied_record, "graph_input_sha256": graph.input_sha256,
                       "environment": environment_record(),
                       "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "sediment_parameters": prepared["sediment_record"]["parameters"],
    }
    (out / "legacy_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("totals_kg", "final_mobile_by_class_kg", "max_abs_identity_residual_kg",
                                              "peak_outlet_flux_kg_s", "time_of_peak_outlet_flux_s", "performance")}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
