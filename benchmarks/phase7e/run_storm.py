"""Frozen Plot1 characteristic benchmark with low-overhead phase timers (Phase 7e copy).

Adds the selective-exchange counters of the Phase 7e MAPLE candidate (when present) and validates the
voxel surface cache of the final bed against a fresh full recomputation.

Run only after the candidate source is frozen. The normal runner binds source,
case and reference provenance and refuses an existing output directory.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import maple_syrup.sediment_event as event
from maple_syrup.benchmark_experiment import run_plot1_matched_benchmark

parser = argparse.ArgumentParser()
parser.add_argument("--bins", type=int, default=32)
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--allow-maple-source-change", action="store_true",
                    help="explicitly record an isolated MAPLE candidate differing from case import")
parser.add_argument("--touched-inventory", action="store_true",
                    help="Phase 7e candidate opt-in at the RUNNER level: call apply_water_process_demand with "
                         "trust_surface_cache=True and bed_change_from_touched_storage=True (same adapter contract as "
                         "maple_syrup.sediment_bed.apply_bed_demand; production source unchanged)")
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("new output directory required")

if args.touched_inventory:
    from dataclasses import replace as _replace

    import maple_syrup.sediment_bed as _bed

    def apply_bed_demand_touched(state, context, demand):
        """`maple_syrup.sediment_bed.apply_bed_demand` with the Phase 7e opt-ins (runner integration only)."""
        from maple.water import apply_water_process_demand, validate_water_state

        g = context.geometry
        validate_water_state(state.water, g.ny, g.nx, len(context.grain_classes.classes))
        result = apply_water_process_demand(
            state.voxel_column, state.active_layer, state.water, state.ledger, demand,
            g, context.grain_classes, context.mass_resolution_kg,
            sediment_availability=state.sediment_availability,
            initial_available_fraction=context.initial_available_fraction,
            adapter_name=_bed.ADAPTER_NAME, detachment_integration='rate_times_dt',
            trust_surface_cache=True, bed_change_from_touched_storage=True)
        return _replace(state, voxel_column=result.new_voxel_column, active_layer=result.new_active_layer,
                        water=result.new_water, ledger=result.new_ledger,
                        sediment_availability=result.new_sediment_availability), result

    event.apply_bed_demand = apply_bed_demand_touched
    import maple_syrup.complete_event as _complete
    _complete.apply_bed_demand = apply_bed_demand_touched

timers = {}
originals = {}
for name in ("coupled_step", "sediment_physics_step", "characteristic_step", "apply_bed_demand", "with_water"):
    fn = getattr(event, name)
    originals[name] = fn

    def wrap(fn=fn, name=name):
        @functools.wraps(fn)
        def measured(*a, **kw):
            row = timers.setdefault(name, {"calls": 0, "wall_s": 0.0})
            label = name
            if name == "apply_bed_demand":
                label = "maple_pickup" if row["calls"] % 2 == 0 else "maple_deposit_export"
            start = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                elapsed = time.perf_counter() - start
                row["calls"] += 1
                row["wall_s"] += elapsed
                if label != name:
                    sub = timers.setdefault(label, {"calls": 0, "wall_s": 0.0})
                    sub["calls"] += 1
                    sub["wall_s"] += elapsed
        return measured

    setattr(event, name, wrap())

original_attempt = event.sediment_coupled_step
peak = {"outlet_discharge_m3_s": -1.0}


@functools.wraps(original_attempt)
def observe_attempt(*a, **kw):
    result = original_attempt(*a, **kw)
    route = result.storm.route
    q = float(route.outlet_discharge_m3_s)
    if q > peak["outlet_discharge_m3_s"]:
        peak.update(outlet_discharge_m3_s=q, t_s=a[0].storm.t_s + a[6],
                    depth_m=route.depth_m.copy(), velocity_m_s=route.velocity_m_s.copy())
    return result


event.sediment_coupled_step = observe_attempt
try:
    run = run_plot1_matched_benchmark(
        "outputs/plot1", args.output,
        applied_rainfall_csv="outputs/phase7/mahleran_reference_audit/applied_rainfall.csv",
        reference_run_dir="outputs/phase7/mahleran_deterministic_ksat_run",
        max_dt_s=args.dt, implementation="numba", transport_scheme="characteristic", phase_bins=args.bins,
        allow_maple_source_change=args.allow_maple_source_change,
    )
finally:
    for name, fn in originals.items():
        setattr(event, name, fn)
    event.sediment_coupled_step = original_attempt

assert run.result.n_rejected_attempts == 0, "peak observer requires no rejected attempts"
assert timers["apply_bed_demand"]["calls"] == 2 * run.result.n_accepted_steps
assert peak["t_s"] == float(run.result.time_of_peak_outlet_s)
assert peak["outlet_discharge_m3_s"] == float(run.result.peak_outlet_discharge_m3_s)
np.savez(args.output / "peak_outlet_snapshot.npz", **peak)
selective = None
from maple.surface.voxels import transfer as _transfer

if hasattr(_transfer, "selective_statistics"):
    selective = dict(_transfer.selective_statistics)
    final_column = run.result.state.bed.voxel_column
    selective["final_cache_validated"] = bool(_transfer.validate_voxel_surface_cache(final_column))
    ny, nx, nv, nc = final_column.mass_kg.shape
    selective["mass_entries_per_call"] = ny * nx * nv * nc
report = {"component_timers": timers, "performance": run.summary["performance"], "selective_exchange": selective,
          "touched_inventory_opt_in": bool(args.touched_inventory),
          "source": run.summary["provenance"], "closure": run.result.closure(),
          "phase_bins": args.bins, "max_dt_s": args.dt,
          "measurement_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "measurement_script_text": Path(__file__).read_text(),
          "note": "MAPLE subrows overlap apply_bed_demand. CPU run; timers and peak snapshot observe unchanged returns."}
(args.output / "component_times.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"output": str(args.output), "sediment": run.summary["sediment"]["totals"],
                  "component_timers": timers}, indent=2))
