"""CW-B104 old-vs-new evidence driver (one tree per process; values-only JSON + NPZ out).

Runs the SAME small fixtures on whichever SYRUP tree is first on PYTHONPATH (the frozen C1 old-cadence oracle, or the
B104 net-routing candidate) and records what each tree did: commits, routing refreshes (graph builds counted by
monkeypatch), reroutes, final physical arrays (NPZ), closure, and wall times of a FIRST run (includes lazy
compilation) and a SECOND identical run (steady state). Nothing here asserts; the comparator reports differences.

Fixtures (all tiny; no Chastre):
  valley45   4 x 5 valley, 45 s of 72 mm/h, interval commits every 10 s, dense bed (array sweep / CuPy array sweep)
  chain_dry  3-cell chain, complete_event to the dry handoff, interval commits every 2 s
  window12   B102 seeded resident-window valley 64 x 32, 12 s (numba on NumPy, cuda on CuPy), commits every 2 s

Usage: python -B b104_compare_driver.py --backend numpy|cupy --tmp DIR --out JSON
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
for sub in ("phase5", "conservative_water", "phase6"):
    sys.path.insert(0, str(TESTS / sub))


def H(value):
    from maple.core.backend import to_host

    return np.asarray(to_host(value))


def b102_build(xp, tmp):
    path = TESTS / "cw_b102" / "b102_experiment.py"
    spec = importlib.util.spec_from_file_location("_b104cmp_b102_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    s, ctx, control, schedule, _, _ = module.build(None, xp, tmp)
    return s, ctx, control, schedule


def counters():
    """Count every routing-graph build of the tree under test (C1 refreshes at every commit; B104 at entry + reroutes)."""
    import maple_syrup.sediment_bed as bed_module

    counts = {"graph_builds": 0}
    real = bed_module.build_routing_graph

    def counted(*a, **k):
        counts["graph_builds"] += 1
        return real(*a, **k)

    bed_module.build_routing_graph = counted
    return counts, lambda: setattr(bed_module, "build_routing_graph", real)


def record(result, state, arrays, prefix):
    from maple.core.types.topographic_commit import committed_elevation_m

    from maple_syrup.sediment_bed import bed_by_cell_class

    arrays[f"{prefix}_committed_elevation_m"] = H(committed_elevation_m(state.bed.committed_topography))
    arrays[f"{prefix}_bed_by_cell_class_kg"] = H(bed_by_cell_class(state.bed))
    arrays[f"{prefix}_mobile_kg"] = H(state.bed.water.mobile_mass_by_cell_class_kg)
    arrays[f"{prefix}_depth_m"] = H(state.storm.depth_m)
    arrays[f"{prefix}_discharge_m2_s"] = H(state.storm.discharge_m2_s)
    arrays[f"{prefix}_cumulative_pickup_kg"] = H(result.cumulative_pickup_kg)
    arrays[f"{prefix}_cumulative_deposition_kg"] = H(result.cumulative_deposition_kg)
    arrays[f"{prefix}_graph_slope"] = np.asarray(state.graph.slope)
    arrays[f"{prefix}_graph_aspect"] = np.asarray(state.graph.aspect)
    closure = result.closure()
    log = [{k: c.get(k) for k in ("t_s", "forced", "commit_reason", "rerouted", "graph_changed", "rerouted_cells",
                                    "net_displacement_m", "max_pending_elevation_change_m",
                                    "max_abs_committed_elevation_change_m")} for c in result.commit_log]
    return {
        "t_s": float(state.t_s), "n_accepted_steps": result.n_accepted_steps, "n_commits": result.n_commits,
        "n_forced_commits": result.n_forced_commits, "n_graph_changes": result.n_graph_changes,
        "rerouted_cells_total": result.rerouted_cells_total,
        "n_reroutes": getattr(result, "n_reroutes", None), "n_net_commits": getattr(result, "n_net_commits", None),
        "max_unrouted_net_displacement_m": getattr(result, "max_unrouted_net_displacement_m", None),
        "commit_log": log, "commit_wall_s": result.commit_wall_s,
        "export_actual_by_class_kg": H(result.by_class["export_actual"]).tolist(),
        "actual_pickup_by_class_kg": H(result.by_class["actual_pickup"]).tolist(),
        "deposition_actual_by_class_kg": H(result.by_class["deposition_actual"]).tolist(),
        "cumulative_export_m3": float(H(result.cumulative_export_m3)),
        "peak_outlet_discharge_m3_s": float(H(result.peak_outlet_discharge_m3_s)),
        "closure_closed": bool(closure["closed"]), "closure_residual_kg": closure["residual_kg"],
        "closure_tolerance_kg": closure["tolerance_kg"], "request_reconciled": bool(closure["request_reconciled"]),
    }


def run_valley(xp):
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from test_coupled_backends import setup_xp
    from test_sediment_event import make_bed, valley_elevation

    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_event import SedimentEventControl, evolve_sediment_event
    from maple_syrup.storm import StormControl

    bed, ctx = make_bed(valley_elevation(4, 5), commit_spec=TopographicCommitSpec(commit_interval_s=10.0))
    implementation = "array" if xp is np else "cuda"
    control = SedimentEventControl(storm=StormControl(implementation=implementation),
                                   transport_implementation="auto" if xp is np else "array")
    s = setup_xp(bed, ctx, xp, south_ring=valley_elevation(5, 5)[0] - 0.01, control=control)
    r = evolve_sediment_event(s["state0"], s["ctx"], s["column"], s["field"], constant_rainfall(0.0, 45.0, 72.0),
                              s["vegetation"], s["sediment"], 45.0, control, report_every_s=15.0)
    return r, r.state


def run_chain(xp):
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from test_coupled_backends import setup_xp
    from test_sediment_event import chain_elevation, make_bed

    from maple_syrup.complete_event import CompletionPolicy, complete_event
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_event import SedimentEventControl
    from maple_syrup.storm import StormControl

    bed, ctx = make_bed(chain_elevation(3), depth_m=.001, commit_spec=TopographicCommitSpec(commit_interval_s=2.0))
    control = SedimentEventControl(storm=StormControl(max_dt_s=1, max_steps=10000))
    s = setup_xp(bed, ctx, xp, ksat=1e-5, control=control)
    out = complete_event(s["state0"], s["ctx"], s["column"], s["field"], constant_rainfall(0, 10, 100),
                         s["vegetation"], s["sediment"], max_end_s=180, control=control,
                         policy=CompletionPolicy(hold_s=2), report_every_s=2.)
    return out.progress.result, out.progress.result.state


def run_window(xp, tmp):
    from maple_syrup.sediment_event import evolve_sediment_event

    s, ctx, control, schedule = b102_build(xp, tmp)
    r = evolve_sediment_event(s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"], s["sediment"],
                              12.0, control, report_every_s=6.0)
    return r, r.state


def sync(xp):
    if xp is not np:
        xp.cuda.Device().synchronize()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    p.add_argument("--tmp", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from maple.core.backend import read_transfer_counters, resolve_backend

    import maple_syrup

    xp = resolve_backend(a.backend).xp
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=False)
    out = {"tree": str(Path(maple_syrup.__file__).resolve().parents[2]), "backend": a.backend, "fixtures": {}}
    arrays = {}
    for name, fn in (("valley45", lambda k: run_valley(xp)), ("chain_dry", lambda k: run_chain(xp)),
                     ("window12", lambda k: run_window(xp, tmp / f"w{k}"))):
        timings = []
        for k in range(2):  # first run includes lazy compilation; the second is the steady-state repeat
            counts, restore = counters()
            before = read_transfer_counters()
            sync(xp)
            t0 = time.perf_counter()
            result, state = fn(k)
            sync(xp)
            wall = time.perf_counter() - t0
            delta = read_transfer_counters().delta(before)
            restore()
            timings.append({"wall_s": wall, "graph_builds": counts["graph_builds"],
                            "device_to_host": int(delta.device_to_host),
                            "device_to_host_bytes": int(delta.device_to_host_bytes),
                            "host_to_device_bytes": int(delta.host_to_device_bytes),
                            "scalar_reads": int(delta.scalar_reads)})
        entry = record(result, state, arrays, name)
        entry["runs"] = timings
        out["fixtures"][name] = entry
        del result, state
    np.savez(Path(a.out).with_suffix(".npz"), **arrays)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
