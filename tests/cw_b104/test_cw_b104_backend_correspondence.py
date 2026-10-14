"""CW-B104: CPU / GPU correspondence under net-elevation-only rerouting (GPU tests skip honestly without CuPy and a
device; a skip claims nothing).

The accepted CW-C1 comparator `test_coupled_backends.compare_events` (declared bounds: water rtol 2e-12 / atol 1e-14,
sediment rtol 1e-10 / atol 1e-14, positions atol 1e-9; integer controls and the commit schedule exact) is applied
unchanged, plus EXACT equality of the routing decisions (which commits rerouted, why, the counters). No tolerance is
added. Source-aware: the routing source is compared at the water bound (it is a committed elevation).

Risk recorded, not hidden: a net displacement within cross-backend round-off of the threshold could flip a decision
and legitimately split the two schedules; the tests below would then fail on the exact decision equality.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("maple")

from test_coupled_backends import (
    NUMBA_SKIP,
    SED_ATOL,
    SED_RTOL,
    WATER_ATOL,
    WATER_RTOL,
    H,
    assert_closed_any,
    close,
    commit_valley,
    compare_events,
    run,
)
from test_windowed_storm import window_inputs

from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_event import (
    CudaHydraulics,
    evolve_sediment_event,
)

ROUTING_KEYS = ("t_s", "commit_reason", "rerouted", "forced")


def gpu_xp():
    from maple.core.backend import BackendUnavailableError, resolve_backend

    try:
        return resolve_backend("cupy").xp
    except BackendUnavailableError as exc:
        pytest.skip(f"cupy unavailable: {exc}")


def labelled(inputs, threshold):
    """Apply an EXPLICITLY LABELLED fixture threshold (also MAPLE's pending trigger) to an input dict's context."""
    from dataclasses import replace

    from maple.core.parameters.topographic_commit import TopographicCommitSpec

    if threshold is None:
        return inputs
    ctx = replace(inputs["ctx"], topographic_commit_spec=TopographicCommitSpec(commit_interval_s=10.0,
                                                                               pending_elevation_trigger_m=threshold))
    return dict(inputs, ctx=ctx)


def routing_equal(ref, cand):
    for key in ROUTING_KEYS:
        assert [c[key] for c in ref.commit_log] == [c[key] for c in cand.commit_log], key
    for name in ("n_reroutes", "n_net_commits", "n_routing_gate_evaluations", "routing_source_origin",
                 "routing_source_t_s"):
        assert getattr(ref, name) == getattr(cand, name), name
    close(ref.state.routing_source_elevation_m, cand.state.routing_source_elevation_m, WATER_RTOL, WATER_ATOL,
          "routing_source")


@pytest.mark.parametrize("threshold", [None, 2e-5])
@pytest.mark.parametrize("root_solver", ["bisection", "newton"])
def test_cuda_event_matches_the_host_under_net_routing(root_solver, threshold):
    """60 s committing valley: host array sweep vs device CUDA sweep, default threshold (no reroute: graph, discharge
    and context never refreshed) and a LABELLED 2e-5 m fixture value (reroutes happen on both)."""
    xp = gpu_xp()
    schedule = constant_rainfall(0.0, 40.0, 72.0)
    ref_inputs, ref_control = commit_valley(np, "array", root_solver)
    ref_inputs = labelled(ref_inputs, threshold)
    ref = run(ref_inputs, schedule, 60.0, ref_control, cadence=20.0)
    assert_closed_any(ref, ref_inputs["state0"])
    dev_inputs, dev_control = commit_valley(xp, "cuda", root_solver)
    dev_inputs = labelled(dev_inputs, threshold)
    hyd = CudaHydraulics()
    dev = run(dev_inputs, schedule, 60.0, dev_control, cadence=20.0, hydraulics=hyd)
    assert_closed_any(dev, dev_inputs["state0"])
    compare_events(ref, dev)
    routing_equal(ref, dev)
    assert ref.n_commits >= 5
    if threshold is None:
        assert dev.n_reroutes == 0 and hyd.n_preparations == 1
        assert dev.state.graph is dev_inputs["state0"].graph
    else:
        assert 1 <= dev.n_reroutes < dev.n_commits and hyd.n_preparations == 1 + dev.n_reroutes
        assert [e["reason"] for e in hyd.log] == ["initial"] + ["commit"] * dev.n_reroutes
    assert hyd.n_commits_recorded == dev.n_commits


@NUMBA_SKIP
@pytest.mark.parametrize("threshold", [None, 2e-5])
def test_cuda_event_matches_the_numba_host_sweep_under_net_routing(threshold):
    xp = gpu_xp()
    schedule = constant_rainfall(0.0, 40.0, 72.0)
    nb_inputs, nb_control = commit_valley(np, "numba", "bisection", transport_implementation="array")
    nb = run(labelled(nb_inputs, threshold), schedule, 60.0, nb_control, cadence=20.0)
    dev_inputs, dev_control = commit_valley(xp, "cuda", "bisection")
    dev = run(labelled(dev_inputs, threshold), schedule, 60.0, dev_control, cadence=20.0)
    compare_events(nb, dev)
    routing_equal(nb, dev)


@pytest.mark.parametrize("threshold", [None, 5e-6])
def test_windowed_csr_numba_and_cuda_events_agree_under_net_routing(tmp_path, threshold):
    """Resident-window CSR storm, 12 s: host numba vs device CUDA. Routing decisions exact; committed elevation,
    storm, routing source at the water bound; mobile mass and per-class exchange at the sediment bound."""
    from dataclasses import replace

    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.types.topographic_commit import committed_elevation_m

    xp = gpu_xp()
    results = {}
    for name, ns in (("host", np), ("device", xp)):
        s, ctx, control, schedule = window_inputs(ns, tmp_path / name, "csr")
        if threshold is not None:
            ctx = replace(ctx, topographic_commit_spec=TopographicCommitSpec(commit_interval_s=2.0,
                                                                             pending_elevation_trigger_m=threshold))
        results[name] = evolve_sediment_event(s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"],
                                              s["sediment"], 12.0, control, report_every_s=6.0)
    a, b = results["host"], results["device"]
    assert a.closure()["closed"] and b.closure()["closed"]
    assert a.n_accepted_steps == b.n_accepted_steps and a.n_commits == b.n_commits
    routing_equal(a, b)
    if threshold is not None:
        assert a.n_reroutes >= 1
    close(committed_elevation_m(a.state.bed.committed_topography),
          committed_elevation_m(b.state.bed.committed_topography), WATER_RTOL, WATER_ATOL, "committed")
    for name in ("depth_m", "discharge_m2_s", "soil_water_m"):
        close(getattr(a.state.storm, name), getattr(b.state.storm, name), WATER_RTOL, WATER_ATOL, name)
    close(a.state.bed.water.mobile_mass_by_cell_class_kg, b.state.bed.water.mobile_mass_by_cell_class_kg, SED_RTOL,
          SED_ATOL, "mobile")
    for key in ("actual_pickup", "deposition_actual", "export_actual"):
        close(a.by_class[key], b.by_class[key], SED_RTOL, SED_ATOL, key)
    np.testing.assert_array_equal(H(a.state.graph.aspect), H(b.state.graph.aspect))
