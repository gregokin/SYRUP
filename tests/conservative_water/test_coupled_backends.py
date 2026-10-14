"""CW-A1: the conservative coupled event (`maple_syrup.sediment_event`) with
the explicit CUDA hydrology on small REAL MAPLE beds, and its backend
control.

CPU tests (always run): control pairing rules, namespace-binding refusals
before any kernel or MAPLE call, the `CudaHydraulics` lifecycle logic
against fake kernels (one preparation per graph binding, none per retry,
rebuild before a commit is published, convenience form per invocation,
preparation failure publishes nothing), the device memory preflight
against a fake CuPy module, and the runner's pairing refusals.

GPU tests (skip honestly without CuPy and a device; a skip claims
nothing): the same equations on CUDA (bisection and Newton) against the
host array / numba sweeps at the DECLARED bounds (water rtol 2e-12 /
atol 1e-14; sediment rtol 1e-10 / atol 1e-14, predeclared for the coupled
event; the isolated T11 kernel test uses atol 1e-15) with exact integer
controls; limited supply and class-selective refill, dry
settlement / fines export, interval commits with actual topography change
and routing refresh, failed-attempt atomicity on the device, cached
context count / rebuild, repeat bitwise, the generic array reference on
CuPy, and the Plot 1 runner's explicit `cupy` + `cuda` form against the
host run of the same 60 s window.

Nothing here executes MAHLERAN; nothing here was run by its author
(file-only tools): the root records the executed results.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("maple")

from maple.core.backend import (
    is_array,
    to_device,
    to_device_tree,
    to_host,
)
from test_sediment_event import (  # tests/phase5 via conftest
    NC,
    SCOL,
    chain_elevation,
    make_bed,
    valley_elevation,
    within,
)

from maple_syrup import routing_numba, sediment_event
from maple_syrup.infiltration import (
    column_parameters,
    initial_soil_water_m,
)
from maple_syrup.rainfall import constant_rainfall, rainfall_field
from maple_syrup.routing import RoutingGraphError, build_routing_graph
from maple_syrup.sediment_bed import terrain_reference
from maple_syrup.sediment_event import (
    CudaHydraulics,
    SedimentEventControl,
    SedimentEventError,
    evolve_sediment_event,
    initial_event_state,
    sediment_coupled_step,
)
from maple_syrup.sediment_experiment import (
    PREFLIGHT_CASE_FIELDS,
    PREFLIGHT_COEXISTING_STATES,
    PREFLIGHT_PHASE_STATE_SETS,
    SUMMARY_NAME,
    cuda_memory_preflight,
    main,
    run_plot1_sediment_event,
)
from maple_syrup.sediment_physics import (
    REGIME_CODES,
    plot1_sediment_parameters,
)
from maple_syrup.storm import StormControl, coupled_step

EPS = np.finfo(np.float64).eps
MRES = 1e-10
# Declared cross-backend bounds (never fitted): water as the accepted GPU hydrology bound; sediment rtol 1e-10 with
# atol 1e-14 is the bound PREDECLARED for the coupled event (the isolated T11 kernel test uses rtol 1e-10 / atol 1e-15).
WATER_RTOL, WATER_ATOL = 2e-12, 1e-14
SED_RTOL, SED_ATOL = 1e-10, 1e-14
POSITION_ATOL = 1e-9
ROOT_SOLVERS = ("bisection", "newton")
NUMBA_SKIP = pytest.mark.skipif(not routing_numba.numba_available(), reason="Numba not installed; host numba leg "
                                                                            "not exercised, no claim made")


# --- helpers: beds, graphs and columns in either namespace -----------------------------------------------------------
def H(value):
    """Host NumPy view/copy of a NumPy or CuPy value (counted on a device)."""
    return np.asarray(to_host(value))


def leaves(value):
    """Every array reachable through dataclass fields (NumPy or CuPy)."""
    if is_array(value):
        yield value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            yield from leaves(getattr(value, f.name))


def snapshot_any(state):
    extra = [state.storm.depth_m, state.storm.soil_water_m, state.storm.discharge_m2_s, state.sediment_velocity_m_s]
    if state.phase is not None:
        extra += [state.phase.fraction, state.phase.position_m]
    return [H(a).copy() for a in list(leaves(state.bed)) + extra]


def compare_any(state, saved):
    extra = [state.storm.depth_m, state.storm.soil_water_m, state.storm.discharge_m2_s, state.sediment_velocity_m_s]
    if state.phase is not None:
        extra += [state.phase.fraction, state.phase.position_m]
    current = list(leaves(state.bed)) + extra
    for a, b in zip(saved, current, strict=True):
        np.testing.assert_array_equal(a, H(b))


def device_bed(bed, ctx, xp):
    if xp is np:
        return bed, ctx
    return to_device_tree(bed, xp), replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, xp))


def graph_for_xp(bed, xp, *, ff=1.0, walls=1.0, south_ring=None):
    """`test_sediment_event.graph_for` with the runtime arrays placed in `xp`."""
    from maple.core.types.topographic_commit import committed_elevation_m

    zc = H(committed_elevation_m(bed.committed_topography))
    ny, nx = zc.shape
    z = np.full((ny + 2, nx + 2), float(zc.max()) + walls)
    z[1:-1, 1:-1] = zc
    z[0, 1:-1] = zc[0] - 0.02 if south_ring is None else south_ring
    exports = np.zeros(z.shape, dtype=bool)
    exports[0, 1:-1] = True
    graph = build_routing_graph(z, exports, np.full((ny, nx), float(ff)), 0.5, xp=xp)
    return graph, terrain_reference(z, exports, bed, graph)


def column_for_xp(graph, ksat, xp, *, theta0=0.25, theta_sat=0.4, thickness=0.3):
    def full(v):
        return xp.full(graph.shape, float(v), dtype=np.float64)

    params = column_parameters(model="fixed_ksat", ksat_m_per_s=full(ksat), suction_m=full(0.0),
                               drainage_parameter=full(0.0), theta_sat=full(theta_sat), soil_thickness_m=full(thickness))
    return params, initial_soil_water_m(params, full(theta0))


def setup_xp(bed, ctx, xp, *, ksat=1e-7, south_ring=None, control=None):
    bed, ctx = device_bed(bed, ctx, xp)
    graph, terrain = graph_for_xp(bed, xp, south_ring=south_ring)
    column, soil0 = column_for_xp(graph, ksat, xp)
    state0 = initial_event_state(graph, terrain, bed, ctx, soil0, control=control)
    return {"state0": state0, "column": column, "field": rainfall_field(*graph.shape, xp=xp),
            "vegetation": xp.zeros(graph.shape, dtype=np.float64), "sediment": plot1_sediment_parameters(xp=xp),
            "ctx": ctx, "xp": xp}


def control_for(implementation, root_solver="bisection", *, storm=None, **kwargs):
    return SedimentEventControl(storm=StormControl(implementation=implementation, root_solver=root_solver,
                                                   **(storm or {})), **kwargs)


def run(inputs, schedule, end, control, *, cadence=10.0, hydraulics=None, state=None):
    return evolve_sediment_event(state or inputs["state0"], inputs["ctx"], inputs["column"], inputs["field"],
                                 schedule, inputs["vegetation"], inputs["sediment"], end, control,
                                 report_every_s=cadence, hydraulics=hydraulics)


def commit_valley(xp, implementation, root_solver="bisection", **kwargs):
    """4x5 valley with interval commits every 10 s (deterministic commit times on every backend)."""
    from maple.core.parameters.topographic_commit import TopographicCommitSpec

    bed, ctx = make_bed(valley_elevation(4, 5), commit_spec=TopographicCommitSpec(commit_interval_s=10.0))
    control = control_for(implementation, root_solver, **kwargs)
    inputs = setup_xp(bed, ctx, xp, south_ring=valley_elevation(5, 5)[0] - 0.01, control=control)
    assert int(inputs["state0"].graph.outlet.sum()) == 1
    return inputs, control


# --- closure and cross-backend comparison -----------------------------------------------------------------------------
def assert_closed_any(result, state0, *, check_dry_settling=True):
    """`test_sediment_event.assert_closed` through host conversions (device results)."""
    closure = result.closure()
    assert closure["closed"] and closure["request_reconciled"]
    within(closure["residual_kg"], closure["tolerance_kg"])
    bc = {k: H(v) for k, v in result.by_class.items()}
    n = result.n_accepted_steps
    cells = result.state.graph.shape[0] * result.state.graph.shape[1]
    allowance = 2 * n * cells * MRES + 64 * EPS * n * np.maximum(bc["deposition_requested"] + bc["export_requested"], 1.0)
    within(bc["deposition_unmet"], allowance)
    within(bc["export_unmet"], allowance)
    within(bc["transport_budget_residual"], bc["transport_budget_tolerance"])
    assert np.all(bc["actual_pickup"] <= bc["requested_pickup"] * (1 + 4 * EPS))
    st = result.state
    np.testing.assert_array_equal(H(st.bed.water.depth_m), H(st.storm.depth_m))
    area = st.graph.dx_m ** 2
    h0, s0 = float(H(state0.storm.depth_m).sum()), float(H(state0.storm.soil_water_m).sum())
    lhs = float(H(st.storm.depth_m).sum() + H(st.storm.soil_water_m).sum() + H(result.cumulative_drainage_m).sum()) \
        * area + float(H(result.cumulative_export_m3))
    rhs = (h0 + s0 + float(H(result.cumulative_rain_m).sum())) * area
    assert abs(lhs - rhs) <= 1e-9 * max(rhs, 1.0)
    change = H(result.bed_change_by_cell_class_kg())
    exchange = H(result.cumulative_deposition_kg) - H(result.cumulative_pickup_kg)
    within(change - exchange, 4 * cells * MRES * (n + 1) + 64 * EPS * (n + 1) * np.maximum(np.abs(exchange), 1e-3))
    if check_dry_settling:
        dry = H(st.storm.depth_m) == 0.0
        assert not np.any(H(st.bed.water.mobile_mass_by_cell_class_kg)[dry])
    return closure, bc


EXACT_INTS = ("n_accepted_steps", "n_rejected_attempts", "n_commits", "n_forced_commits", "n_graph_changes",
              "rerouted_cells_total", "max_transport_substeps_used", "n_transport_rejections", "n_maple_water_calls",
              "n_inventory_terms", "phase_steered_cells_total")
EXACT_COUNTS = ("cell_steps_no_runon", "cell_steps_partial_runon", "cell_steps_complete_runon")
WATER_FIELDS = ("cumulative_rain_m", "cumulative_intake_m", "cumulative_saturation_return_m", "cumulative_drainage_m",
                "peak_depth_m", "peak_velocity_m_s", "last_velocity_m_s", "hydrograph", "cumulative_export_m3",
                "peak_outlet_discharge_m3_s", "time_of_peak_outlet_s", "max_courant_old", "max_courant_new")
SEDIMENT_FIELDS = ("cumulative_pickup_kg", "cumulative_deposition_kg", "cumulative_export_request_kg",
                   "sediment_hydrograph", "ledger_process_totals_reset_kg", "committed_net_bed_change_kg",
                   "initial_bed_by_cell_class_kg", "initial_bed_inventory_kg", "initial_mobile_kg", "peak_mobile_kg",
                   "time_of_peak_mobile_s", "peak_export_rate_kg_s", "time_of_peak_export_s", "max_sediment_courant",
                   "max_decay_exponent")


def close(a, b, rtol, atol, name):
    np.testing.assert_allclose(H(b), H(a), rtol=rtol, atol=atol, err_msg=name)


def compare_events(ref, cand, *, water=(WATER_RTOL, WATER_ATOL), sediment=(SED_RTOL, SED_ATOL),
                   position_atol=POSITION_ATOL):
    """Physical fields of two events at the declared bounds; integer controls exact. Round-off diagnostics
    (`max_transport_cell_residual_kg`, `max_phase_reconciliation_residual_kg`, the numerical residual
    accumulators) are NOT compared across backends: each side must satisfy its own unchanged bound (closure)."""
    from maple.core.types.topographic_commit import committed_elevation_m

    for name in EXACT_INTS:
        assert getattr(ref, name) == getattr(cand, name), name
    for name in EXACT_COUNTS:
        assert int(H(getattr(ref, name))) == int(H(getattr(cand, name))), name
    for name in REGIME_CODES:
        assert int(H(ref.regime_cell_steps[name])) == int(H(cand.regime_cell_steps[name])), name
    for key in ("t_s", "forced", "rerouted_cells", "outlets_changed", "n_outlets", "graph_changed", "commit_count"):
        assert [c[key] for c in ref.commit_log] == [c[key] for c in cand.commit_log], key
    np.testing.assert_array_equal(ref.boundaries, cand.boundaries)
    for name in WATER_FIELDS:
        close(getattr(ref, name), getattr(cand, name), *water, name)
    for name in ("depth_m", "soil_water_m", "discharge_m2_s"):
        close(getattr(ref.state.storm, name), getattr(cand.state.storm, name), *water, name)
    for name in SEDIMENT_FIELDS:
        close(getattr(ref, name), getattr(cand, name), *sediment, name)
    assert set(ref.by_class) == set(cand.by_class)
    for key in ref.by_class:
        if key in ("pickup_numerical_residual", "deposit_numerical_residual", "transport_budget_residual"):
            continue  # round-off diagnostics; bounded per backend by the closure checks
        close(ref.by_class[key], cand.by_class[key], *sediment, key)
    rb, cb = ref.state.bed, cand.state.bed
    close(rb.water.mobile_mass_by_cell_class_kg, cb.water.mobile_mass_by_cell_class_kg, *sediment, "mobile")
    close(rb.voxel_column.mass_kg, cb.voxel_column.mass_kg, *sediment, "voxel")
    close(rb.active_layer.mass_kg, cb.active_layer.mass_kg, *sediment, "active")
    close(rb.sediment_availability.available_mass_kg, cb.sediment_availability.available_mass_kg, *sediment,
          "available")
    close(rb.ledger.pending_bed_mass_change_kg, cb.ledger.pending_bed_mass_change_kg, *sediment, "pending")
    close(committed_elevation_m(rb.committed_topography), committed_elevation_m(cb.committed_topography), *water,
          "committed_elevation")
    assert rb.committed_topography.commit_count == cb.committed_topography.commit_count
    close(ref.state.sediment_velocity_m_s, cand.state.sediment_velocity_m_s, *sediment, "sediment_velocity")
    np.testing.assert_array_equal(H(ref.state.graph.aspect), H(cand.state.graph.aspect))
    np.testing.assert_array_equal(H(ref.state.graph.outlet), H(cand.state.graph.outlet))
    close(ref.state.graph.slope, cand.state.graph.slope, *water, "slope")
    close(ref.state.graph.conveyance, cand.state.graph.conveyance, *water, "conveyance")
    assert (ref.state.phase is None) == (cand.state.phase is None)
    if ref.state.phase is not None:
        wr = H(rb.water.mobile_mass_by_cell_class_kg)[..., None] * H(ref.state.phase.fraction)
        wc = H(cb.water.mobile_mass_by_cell_class_kg)[..., None] * H(cand.state.phase.fraction)
        np.testing.assert_allclose(wc, wr, rtol=sediment[0], atol=sediment[1], err_msg="phase mass")
        np.testing.assert_allclose(H(cand.state.phase.position_m), H(ref.state.phase.position_m), rtol=0,
                                   atol=position_atol, err_msg="phase position")


def compare_bitwise(ref, cand):
    compare_events(ref, cand, water=(0.0, 0.0), sediment=(0.0, 0.0), position_atol=0.0)


# =====================================================================================================================
# CPU: control, binding refusals, lifecycle logic, preflight, runner pairing
# =====================================================================================================================
def test_control_pairing_rules_and_default_cpu_behaviour():
    cuda = StormControl(implementation="cuda")
    with pytest.raises(SedimentEventError, match="numba"):
        SedimentEventControl(storm=cuda, transport_implementation="numba").validated()
    c = SedimentEventControl(storm=cuda).validated()
    assert c.cuda and c.resolved_transport_implementation() == "array"
    assert SedimentEventControl(storm=cuda, transport_implementation="array").validated().resolved_transport_implementation() == "array"
    rec = c.backend_record()
    assert rec["hydraulics"] == "cuda" and rec["required_namespace"] == "cupy" and rec["transport_implementation"] == "array"
    assert SedimentEventControl(storm=StormControl(implementation="cuda", root_solver="newton")).validated().backend_record()["root_solver"] == "newton"
    upwind = SedimentEventControl(storm=cuda, transport_scheme="upwind", sediment_courant_max=1.0).validated()
    assert upwind.backend_record()["transport_implementation"] is None
    # defaults and the CPU pairings are unchanged
    default = SedimentEventControl().validated()
    assert not default.cuda and default.storm.implementation == "array"
    assert default.resolved_transport_implementation() == "array"
    assert default.backend_record()["required_namespace"] == "numpy_or_cupy"
    numba_control = SedimentEventControl(storm=StormControl(implementation="numba")).validated()
    assert numba_control.resolved_transport_implementation() == "numba"
    assert numba_control.backend_record()["required_namespace"] == "numpy"
    assert SedimentEventControl(transport_implementation="numba").validated().backend_record()["required_namespace"] == "numpy"
    with pytest.raises(SedimentEventError, match="transport_implementation"):
        SedimentEventControl(transport_implementation="cuda").validated()
    with pytest.raises(SedimentEventError, match="mode"):
        CudaHydraulics(mode="bogus")
    hyd = CudaHydraulics()
    assert hyd.context is None and hyd.n_preparations == 0 and hyd.summary()["context"] is None
    with pytest.raises(SedimentEventError, match="reason"):
        hyd.prepare(None, None, t_s=0.0, reason="whenever")
    with pytest.raises(SedimentEventError, match="adopt"):
        hyd.adopt("not prepared")


def test_cuda_control_on_a_numpy_state_is_refused_before_any_kernel_or_maple_call(monkeypatch):
    bed, ctx = make_bed(valley_elevation(4, 5))
    inputs = setup_xp(bed, ctx, np, south_ring=valley_elevation(5, 5)[0] - 0.01)
    state0 = inputs["state0"]
    saved = snapshot_any(state0)
    schedule = constant_rainfall(0.0, 30.0, 72.0)

    def boom(*args, **kwargs):
        raise AssertionError("a kernel or MAPLE call ran after a refused binding")

    monkeypatch.setattr(sediment_event, "apply_bed_demand", boom)
    monkeypatch.setattr(sediment_event, "_prepare_cuda_context", boom)
    monkeypatch.setattr(sediment_event, "coupled_step", boom)
    cuda = control_for("cuda")
    with pytest.raises(SedimentEventError, match="CuPy"):
        run(inputs, schedule, 10.0, cuda)
    rate = inputs["field"].apply(72.0 / 3.6e6)
    with pytest.raises(SedimentEventError, match="CuPy"):
        sediment_coupled_step(state0, ctx, inputs["column"], rate, inputs["vegetation"], inputs["sediment"], 1.0, cuda)
    with pytest.raises(SedimentEventError, match="CuPy"):
        sediment_coupled_step(state0, ctx, inputs["column"], rate, inputs["vegetation"], inputs["sediment"], 1.0, cuda,
                              hydraulics=CudaHydraulics())
    # a lifecycle is only meaningful with the CUDA sweep
    with pytest.raises(SedimentEventError, match="hydraulics"):
        run(inputs, schedule, 10.0, SedimentEventControl(), hydraulics=CudaHydraulics())
    with pytest.raises(SedimentEventError, match="hydraulics"):
        sediment_coupled_step(state0, ctx, inputs["column"], rate, inputs["vegetation"], inputs["sediment"], 1.0,
                              SedimentEventControl(), hydraulics=CudaHydraulics())
    with pytest.raises(SedimentEventError, match="CudaHydraulics"):
        run(inputs, schedule, 10.0, cuda, hydraulics=object())
    compare_any(state0, saved)


class FakeContext:
    """Stands in for `CudaHydrologyContext` in the CPU lifecycle test: bound by identity, delegates to the array
    sweep so the coupled results must be BITWISE those of the array control."""

    def __init__(self, graph, column):
        self.graph, self.column = graph, column
        self.graph_input_sha256 = graph.input_sha256
        self.preparation_s, self.kernel_load_s = 0.25, 0.5
        self.host_to_device_bytes, self.device_to_host_bytes, self.static_bytes = 8, 16, 32
        self.mode = "fake"

    def is_bound_to(self, graph, column):
        return graph is self.graph and column is self.column

    def summary(self):
        return {"fake": True, "graph_input_sha256": self.graph_input_sha256}


def install_fake_cuda(monkeypatch, prepared, *, fail_at=None, attempts=None):
    """`attempts` (optional) records every preparation REQUEST `(graph, column)` before it succeeds or fails, so a
    test can prove that a preparation failure was actually executed (not merely configured)."""
    def fake_prepare(graph, column, *, mode):
        if attempts is not None:
            attempts.append((graph, column))
        if fail_at is not None and len(prepared) + 1 == fail_at:
            raise RuntimeError("device lost (test)")
        context = FakeContext(graph, column)
        prepared.append(context)
        return context

    def fake_step(context, rain, storm, dt, control):
        assert isinstance(context, FakeContext) and control.implementation == "cuda"
        return coupled_step(context.graph, context.column, rain, storm, dt, replace(control, implementation="array"))

    monkeypatch.setattr(sediment_event, "_prepare_cuda_context", fake_prepare)
    monkeypatch.setattr(sediment_event, "_cuda_prepared_step", fake_step)
    monkeypatch.setattr(sediment_event, "_check_backend_binding", lambda xp, control: None)  # NumPy stands in for CuPy


# CW-E47C1: since CW-B104 the routing (and therefore the CUDA hydraulic context bound to the routing graph) is
# refreshed ONLY by a genuine net-elevation reroute: the post-commit max per-cell |committed surface - held routing
# source| reaches MAPLE's actual `pending_elevation_trigger_m` (default 0.00075 m, `>=`). Ordinary accounting commits
# (time / turnover / pending triggers below that net displacement, and the final forced commit) keep the graph and
# the context. `n_graph_changes` is NOT a substitute (a reroute may keep every receiver direction while the slopes /
# conveyance change); the contract is `commit_log[i]["rerouted"]` and `n_reroutes`.
# The original 45 s / 72 mm/h valley stays as the below-threshold NEGATIVE control (max net ~5.2e-5 m); the same
# actual valley under a stronger physical storm (600 mm/h for 90 s, default threshold) GENUINELY crosses it.
REROUTE_RAIN_MM_H, REROUTE_END_S = 600.0, 90.0
THRESHOLD_M = 0.00075


def assert_below_threshold_accounting_only(result):
    assert result.n_reroutes == 0 and result.n_net_commits == 0 and result.n_commits >= 1
    assert all(not c["rerouted"] and c["net_displacement_m"] < c["reroute_threshold_m"] == THRESHOLD_M
               for c in result.commit_log)


def assert_genuine_reroutes(result):
    rerouted = [c for c in result.commit_log if c["rerouted"]]
    assert rerouted and result.n_reroutes == len(rerouted) == len(result.commit_log) - sum(
        not c["rerouted"] for c in result.commit_log)
    assert all(c["net_displacement_m"] >= c["reroute_threshold_m"] == THRESHOLD_M for c in rerouted)
    assert all(c["net_displacement_m"] < c["reroute_threshold_m"] for c in result.commit_log if not c["rerouted"])
    assert result.routing_source_origin == "net_reroute"
    return rerouted


def assert_lifecycle_summary(hyd, final_graph):
    s = hyd.summary()
    assert s["n_preparations"] == hyd.n_preparations and s["preparation_wall_s"] == pytest.approx(0.25 * hyd.n_preparations)
    assert s["kernel_load_s"] == pytest.approx(0.5 * hyd.n_preparations) and s["context"] == {"fake": True, "graph_input_sha256": final_graph.input_sha256}
    assert s["preparation_host_to_device_bytes"] == 8 * hyd.n_preparations and not s["log_truncated"]


def test_lifecycle_prepares_once_per_graph_binding_rebuilds_only_per_genuine_reroute_and_never_per_retry(monkeypatch):
    # (a) NEGATIVE control: below-threshold accounting commits (incl. the final forced one) never re-prepare
    inputs, control = commit_valley(np, "cuda")
    state0, column = inputs["state0"], inputs["column"]
    schedule = constant_rainfall(0.0, 45.0, 72.0)
    ref = run(inputs, schedule, 45.0, control_for("array"), cadence=15.0)
    assert ref.n_commits >= 5 and ref.n_forced_commits == 1 and ref.n_rejected_attempts == 0
    assert_below_threshold_accounting_only(ref)
    prepared = []
    install_fake_cuda(monkeypatch, prepared)
    hyd = CudaHydraulics()
    fake = run(inputs, schedule, 45.0, control, cadence=15.0, hydraulics=hyd)
    compare_bitwise(ref, fake)  # the lifecycle threads the same state through the sweep
    assert_closed_any(fake, state0)
    assert_below_threshold_accounting_only(fake)
    assert hyd.n_preparations == len(prepared) == 1  # the initial context only: no per-commit rebuild
    assert [e["reason"] for e in hyd.log] == ["initial"] and [e["replaced_context"] for e in hyd.log] == [False]
    assert hyd.n_binding_hits == fake.n_accepted_steps  # one cheap binding check per attempt, no preparation
    assert fake.state.graph is state0.graph and hyd.context is prepared[0] and hyd.is_bound_to(fake.state.graph, column)
    assert hyd.n_commits_recorded == fake.n_commits  # every accounting commit still records its transfers
    assert_lifecycle_summary(hyd, fake.state.graph)
    # (b) POSITIVE: the same valley under a stronger physical storm genuinely crosses the default threshold;
    # exactly one re-preparation per genuine reroute, at its commit time, none for the other commits or retries
    pos, pos_control = commit_valley(np, "cuda")
    storm = constant_rainfall(0.0, REROUTE_END_S, REROUTE_RAIN_MM_H)
    ref2 = run(pos, storm, REROUTE_END_S, control_for("array"), cadence=15.0)
    rerouted = assert_genuine_reroutes(ref2)
    prepared2 = []
    install_fake_cuda(monkeypatch, prepared2)
    hyd2 = CudaHydraulics()
    fake2 = run(pos, storm, REROUTE_END_S, pos_control, cadence=15.0, hydraulics=hyd2)
    compare_bitwise(ref2, fake2)
    assert_closed_any(fake2, pos["state0"])
    assert [c["t_s"] for c in assert_genuine_reroutes(fake2)] == [c["t_s"] for c in rerouted]
    n = fake2.n_reroutes
    assert fake2.n_commits > n  # accounting commits occurred too, without any preparation
    assert hyd2.n_preparations == len(prepared2) == 1 + n
    assert [e["reason"] for e in hyd2.log] == ["initial"] + ["commit"] * n
    assert [e["replaced_context"] for e in hyd2.log] == [False] + [True] * n
    assert [e["t_s"] for e in hyd2.log][1:] == [c["t_s"] for c in fake2.commit_log if c["rerouted"]]
    assert [e["graph_input_sha256"] for e in hyd2.log][1:] == [c["graph_input_sha256"] for c in fake2.commit_log
                                                               if c["rerouted"]]
    assert hyd2.n_binding_hits == fake2.n_accepted_steps
    assert fake2.state.graph is not pos["state0"].graph
    assert hyd2.context is prepared2[-1] and hyd2.is_bound_to(fake2.state.graph, pos["column"])
    assert hyd2.n_commits_recorded == fake2.n_commits
    assert_lifecycle_summary(hyd2, fake2.state.graph)
    # the lifecycle is bound to the FINAL graph: a continuation from the final state re-uses it (no preparation
    # unless that continuation itself reroutes genuinely)
    again = run(pos, constant_rainfall(REROUTE_END_S, REROUTE_END_S + 15.0, 0.0), REROUTE_END_S + 15.0, pos_control,
                cadence=15.0, hydraulics=hyd2, state=fake2.state)
    assert hyd2.n_preparations == 1 + n + again.n_reroutes
    # a lifecycle pre-bound to a FOREIGN graph is re-prepared (counted, logged), never reused stale
    install_fake_cuda(monkeypatch, prepared)
    other, _ = commit_valley(np, "cuda")
    foreign = CudaHydraulics()
    foreign.bind(other["state0"].graph, other["column"], t_s=0.0, reason="initial")
    r2 = run(inputs, schedule, 45.0, control, cadence=15.0, hydraulics=foreign)
    assert_below_threshold_accounting_only(r2)
    assert foreign.n_preparations == 2 and foreign.log[1]["reason"] == "initial" and foreign.log[1]["replaced_context"]
    compare_bitwise(ref, r2)
    # direct single-step convenience form: a context per invocation (documented); a held lifecycle: one
    rate = inputs["field"].apply(72.0 / 3.6e6)
    before = len(prepared)
    one = sediment_coupled_step(state0, inputs["ctx"], column, rate, inputs["vegetation"], inputs["sediment"], 1.0, control)
    two = sediment_coupled_step(state0, inputs["ctx"], column, rate, inputs["vegetation"], inputs["sediment"], 1.0, control)
    assert len(prepared) == before + 2
    np.testing.assert_array_equal(one.storm.state.depth_m, two.storm.state.depth_m)
    lifecycle = CudaHydraulics()
    held = sediment_coupled_step(state0, inputs["ctx"], column, rate, inputs["vegetation"], inputs["sediment"], 1.0,
                                 control, hydraulics=lifecycle)
    held2 = sediment_coupled_step(state0, inputs["ctx"], column, rate, inputs["vegetation"], inputs["sediment"], 1.0,
                                  control, hydraulics=lifecycle)
    assert len(prepared) == before + 3 and lifecycle.n_preparations == 1 and lifecycle.n_binding_hits == 1
    np.testing.assert_array_equal(one.storm.state.depth_m, held.storm.state.depth_m)
    np.testing.assert_array_equal(held.bed.water.mobile_mass_by_cell_class_kg, held2.bed.water.mobile_mass_by_cell_class_kg)
    reference = sediment_coupled_step(state0, inputs["ctx"], column, rate, inputs["vegetation"], inputs["sediment"],
                                      1.0, control_for("array"))
    np.testing.assert_array_equal(reference.bed.water.mobile_mass_by_cell_class_kg, held.bed.water.mobile_mass_by_cell_class_kg)
    np.testing.assert_array_equal(reference.storm.state.discharge_m2_s, held.storm.state.discharge_m2_s)


def test_lifecycle_retries_and_accounting_commits_reuse_the_context_without_re_preparation(monkeypatch):
    bed, ctx = make_bed(chain_elevation(4), depth_m=0.05)
    storm = {"max_dt_s": 8.0, "min_dt_s": 1.0 / 64.0, "max_retries": 10}
    control = control_for("cuda", storm=storm)
    inputs = setup_xp(bed, ctx, np, ksat=0.0, control=control)
    state0 = inputs["state0"]
    saved = snapshot_any(state0)
    schedule = constant_rainfall(0.0, 16.0, 0.0)
    ref = run(inputs, schedule, 16.0, control_for("array", storm=storm), cadence=16.0)
    assert ref.n_rejected_attempts > 0 and ref.rejections[0]["kind"] == "RoutingStepRejected"
    assert_below_threshold_accounting_only(ref)  # the (final forced) accounting commit stays below the threshold
    prepared = []
    install_fake_cuda(monkeypatch, prepared)
    hyd = CudaHydraulics()
    r = run(inputs, schedule, 16.0, control, cadence=16.0, hydraulics=hyd)
    compare_bitwise(ref, r)
    assert_below_threshold_accounting_only(r)
    assert r.n_rejected_attempts > 0 and r.n_commits >= 1 and hyd.n_preparations == 1 == 1 + r.n_reroutes
    assert hyd.n_binding_hits == r.n_accepted_steps + r.n_rejected_attempts  # every retry: the same context
    assert len(prepared) == hyd.n_preparations and hyd.n_commits_recorded == r.n_commits
    compare_any(state0, saved)  # rejected attempts published nothing into the caller's state


def test_lifecycle_preparation_failure_at_a_genuine_reroute_publishes_nothing(monkeypatch):
    inputs, control = commit_valley(np, "cuda")
    state0 = inputs["state0"]
    saved = snapshot_any(state0)
    storm = constant_rainfall(0.0, REROUTE_END_S, REROUTE_RAIN_MM_H)
    ref = run(inputs, storm, REROUTE_END_S, control_for("array"), cadence=15.0)
    assert_genuine_reroutes(ref)
    first = next(i for i, c in enumerate(ref.commit_log) if c["rerouted"])
    assert first > 0  # ordinary accounting commits precede the first genuine crossing
    prepared, attempts = [], []
    install_fake_cuda(monkeypatch, prepared, fail_at=2, attempts=attempts)  # the first reroute's re-preparation fails
    hyd = CudaHydraulics()
    with pytest.raises(RuntimeError, match="device lost"):
        run(inputs, storm, REROUTE_END_S, control, cadence=15.0, hydraulics=hyd)
    # the failure genuinely EXECUTED at the rerouting commit: a second preparation was requested for a REFRESHED graph
    assert len(attempts) == 2 and attempts[0][0] is state0.graph and attempts[1][0] is not state0.graph
    assert attempts[1][0].input_sha256 == ref.commit_log[first]["graph_input_sha256"]
    compare_any(state0, saved)
    assert hyd.n_preparations == 1 and hyd.is_bound_to(state0.graph, inputs["column"])  # the commit was not adopted
    # the transfers of every commit up to and including the refused rerouting one were recorded; nothing after it
    assert hyd.n_commits_recorded == first + 1 and len(prepared) == 1


def preflight_case(bed, ctx):
    """A stand-in for the compiled case: exactly the MAPLE state fields `bed_from_case` places on the device."""
    return SimpleNamespace(config=SimpleNamespace(geometry=ctx.geometry), voxel_column=bed.voxel_column,
                           active_layer=bed.active_layer, water=bed.water, sediment_ledger=bed.ledger,
                           committed_topography=bed.committed_topography, sediment_availability=bed.sediment_availability)


def unique_array_bytes(*objects):
    seen, total = set(), 0
    for obj in objects:
        for a in leaves(obj):
            if id(a) not in seen:
                seen.add(id(a))
                total += int(a.nbytes)
    return total


def fake_cupy(free_bytes):
    return SimpleNamespace(cuda=SimpleNamespace(Device=lambda i: SimpleNamespace(mem_info=(free_bytes, 2 * free_bytes))))


def test_memory_preflight_counts_actual_maple_arrays_and_coexisting_states(monkeypatch):
    from maple.core import backend

    bed, ctx = make_bed(valley_elevation(4, 5))
    case = preflight_case(bed, ctx)
    monkeypatch.setattr(backend, "cupy_module", lambda: fake_cupy(1 << 30))
    rec = cuda_memory_preflight(case, phase_bins=32, device_id=0)
    voxel = np.asarray(bed.voxel_column.mass_kg)
    n_cells = 20
    assert rec["accepted"] and rec["voxel_bytes"] == voxel.nbytes and rec["shape"] == list(voxel.shape)
    # the actual bytes of every unique array of the six MAPLE state fields, no estimate
    expected = unique_array_bytes(*(getattr(case, name) for name in PREFLIGHT_CASE_FIELDS))
    assert rec["case_array_bytes"] == expected == sum(rec["case_field_bytes"].values())
    assert set(rec["case_field_bytes"]) == set(PREFLIGHT_CASE_FIELDS)
    ledger = bed.ledger
    n_process = int(ledger.process_totals_kg.shape[0])
    eight = 8 * n_process * n_cells * NC * 8
    assert rec["case_field_bytes"]["sediment_ledger"] >= eight + 2 * n_cells * NC * 8 + 2 * n_process * NC * 8
    assert rec["case_field_bytes"]["sediment_ledger"] == unique_array_bytes(ledger)
    assert rec["case_field_arrays"]["sediment_ledger"] >= 13
    assert rec["case_field_bytes"]["water"] == bed.water.depth_m.nbytes + bed.water.mobile_mass_by_cell_class_kg.nbytes
    # concurrently alive state sets (documented upper bound), working copies, snapshots, phase sets, working set
    assert rec["coexisting_states"] == PREFLIGHT_COEXISTING_STATES == 8
    assert rec["coexisting_state_bytes"] == 8 * rec["case_array_bytes"]
    assert rec["maple_working_bytes"] == 2 * voxel.nbytes
    assert rec["commit_snapshot_bytes"] == 2 * (voxel.nbytes + bed.active_layer.mass_kg.nbytes)
    assert rec["phase_state_sets"] == PREFLIGHT_PHASE_STATE_SETS == 5
    assert rec["phase_state_bytes"] == 2 * n_cells * NC * 32 * 8  # one phase state: fraction + position
    assert rec["phase_bytes"] == 5 * rec["phase_state_bytes"]
    assert rec["transport_working_bytes"] == 12 * n_cells * NC * 33 * 8
    assert rec["estimate_bytes"] == (rec["coexisting_state_bytes"] + rec["maple_working_bytes"] + rec["commit_snapshot_bytes"]
                                     + rec["phase_bytes"] + rec["transport_working_bytes"] + rec["hydrology_bytes"]
                                     + rec["accounting_bytes"])
    assert rec["required_bytes"] == math.ceil(2.0 * rec["estimate_bytes"]) and rec["device_free_bytes"] == 1 << 30
    upwind = cuda_memory_preflight(case, phase_bins=32, device_id=0, transport_scheme="upwind")
    assert upwind["phase_bytes"] == upwind["phase_state_bytes"] == 0
    assert upwind["transport_working_bytes"] == 8 * n_cells * NC * 8
    # the free-memory refusal is exact at the safety-scaled requirement
    monkeypatch.setattr(backend, "cupy_module", lambda: fake_cupy(rec["required_bytes"]))
    assert cuda_memory_preflight(case, phase_bins=32, device_id=0)["accepted"]
    monkeypatch.setattr(backend, "cupy_module", lambda: fake_cupy(rec["required_bytes"] - 1))
    with pytest.raises(SedimentEventError, match="free memory"):
        cuda_memory_preflight(case, phase_bins=32, device_id=0)
    monkeypatch.setattr(backend, "cupy_module", lambda: None)
    with pytest.raises(SedimentEventError, match="CuPy"):
        cuda_memory_preflight(case, phase_bins=32, device_id=0)


def test_memory_preflight_named_estimator_counts_the_documented_concurrent_lifetimes(monkeypatch):
    """The lifetime multipliers are the documented upper bounds, named in the record, and the estimate is at
    least the sum of the lifetimes the root's review identified (not a claim of an exact peak)."""
    from maple.core import backend
    from maple.core.types.topographic_commit import committed_elevation_m

    bed, ctx = make_bed(valley_elevation(4, 5))
    case = preflight_case(bed, ctx)
    monkeypatch.setattr(backend, "cupy_module", lambda: fake_cupy(1 << 30))
    rec = cuda_memory_preflight(case, phase_bins=32, device_id=0)
    one_state = rec["case_array_bytes"]
    # the eight named state lifetimes
    lifetimes = ("retained initial state0", "current accepted state", "previous attempt pickup",
                 "deposit results", "current attempt", "commit candidate bed", "within-call generation")
    assert all(name in rec["coexisting_state_basis"] for name in lifetimes)
    assert "upper bound" in rec["coexisting_state_basis"] and "not an exact peak" in rec["coexisting_state_basis"]
    assert rec["coexisting_state_bytes"] >= 8 * one_state
    # the five named phase lifetimes
    for name in ("retained initial phase", "current accepted phase", "previous kernel prediction",
                 "current kernel prediction", "current reconciled phase"):
        assert name in rec["phase_state_basis"]
    assert rec["phase_bytes"] >= 5 * rec["phase_state_bytes"] and "upper bound" in rec["phase_state_basis"]
    # the estimate is at least every named lifetime plus the separate working / snapshot margin terms
    active = bed.active_layer.mass_kg.nbytes
    voxel = np.asarray(bed.voxel_column.mass_kg).nbytes
    assert rec["estimate_bytes"] >= (8 * one_state + 5 * rec["phase_state_bytes"] + 2 * voxel + 2 * (voxel + active)
                                     + rec["transport_working_bytes"])
    assert rec["required_bytes"] >= 2 * rec["estimate_bytes"]
    assert "not a memory-fit claim" in rec["basis"]
    # the committed topography's arrays are part of every counted state set (they are what the commit candidate holds)
    assert rec["case_field_bytes"]["committed_topography"] >= committed_elevation_m(bed.committed_topography).nbytes
    assert rec["case_field_arrays"]["committed_topography"] >= 4


def test_memory_preflight_ledger_dominated_low_voxel_case_is_refused_where_a_voxel_only_count_would_pass(monkeypatch):
    from maple.core import backend

    bed, ctx = make_bed(valley_elevation(4, 5), n_voxels=1, voxel_dz=0.5)  # one voxel: the ledger dominates
    case = preflight_case(bed, ctx)
    voxel = np.asarray(bed.voxel_column.mass_kg)
    assert voxel.shape[2] == 1
    monkeypatch.setattr(backend, "cupy_module", lambda: fake_cupy(1 << 30))
    rec = cuda_memory_preflight(case, phase_bins=8, device_id=0)
    assert rec["case_field_bytes"]["sediment_ledger"] > 10 * voxel.nbytes
    assert rec["case_array_bytes"] > 12 * voxel.nbytes
    # a voxel-only "8 bed copies + working" count would call this budget sufficient; the actual ledger-inclusive
    # lifetime estimate refuses
    voxel_only = math.ceil(2.0 * (10 * voxel.nbytes + rec["phase_bytes"] + rec["transport_working_bytes"]
                                  + rec["hydrology_bytes"] + rec["accounting_bytes"]))
    assert voxel_only < rec["required_bytes"]
    monkeypatch.setattr(backend, "cupy_module", lambda: fake_cupy(voxel_only))
    with pytest.raises(SedimentEventError, match="free memory"):
        cuda_memory_preflight(case, phase_bins=8, device_id=0)


def test_memory_preflight_validates_arguments_and_the_cap_before_any_device_query(monkeypatch):
    from maple.core import backend

    bed, ctx = make_bed(valley_elevation(4, 5))
    case = preflight_case(bed, ctx)
    voxel = np.asarray(bed.voxel_column.mass_kg)

    def queried():
        raise AssertionError("the device was queried before the arguments / cap were validated")

    monkeypatch.setattr(backend, "cupy_module", queried)
    for kwargs, match in (
        ({"phase_bins": 0}, "phase_bins"), ({"phase_bins": True}, "phase_bins"), ({"phase_bins": 129}, "phase_bins"),
        ({"phase_bins": 32.0}, "phase_bins"), ({"device_id": -1}, "device_id"), ({"device_id": 0.0}, "device_id"),
        ({"voxel_bytes_max": 0}, "voxel_bytes_max"), ({"safety_factor": 0.5}, "safety_factor"),
        ({"safety_factor": math.inf}, "safety_factor"), ({"safety_factor": True}, "safety_factor"),
        ({"transport_scheme": "legacy"}, "transport_scheme"),
        ({"voxel_bytes_max": voxel.nbytes - 1}, "cap"),
    ):
        full = {"phase_bins": 32, "device_id": 0, **kwargs}
        with pytest.raises(SedimentEventError, match=match):
            cuda_memory_preflight(case, **full)
    with pytest.raises(SedimentEventError, match="voxel column"):
        cuda_memory_preflight(SimpleNamespace(config=case.config, voxel_column=SimpleNamespace(mass_kg=voxel[0])),
                              phase_bins=32, device_id=0)
    with pytest.raises(SedimentEventError, match="lacks"):
        cuda_memory_preflight(SimpleNamespace(config=case.config, voxel_column=bed.voxel_column), phase_bins=32,
                              device_id=0)
    # a failing device query is a refusal, never a blind continuation
    def broken():
        return SimpleNamespace(cuda=SimpleNamespace(Device=lambda i: (_ for _ in ()).throw(RuntimeError("no device"))))

    monkeypatch.setattr(backend, "cupy_module", broken)
    with pytest.raises(SedimentEventError, match="could not query"):
        cuda_memory_preflight(case, phase_bins=32, device_id=0)


def test_runner_refuses_mixed_pairings_before_touching_the_case(tmp_path):
    missing = tmp_path / "no_such_case"
    out = tmp_path / "out"
    for kwargs, match in (
        ({"backend": "cupy", "implementation": "array"}, "refused"),
        ({"backend": "cupy", "implementation": "numba"}, "refused"),
        ({"backend": "cupy"}, "refused"),
        ({"backend": "numpy", "implementation": "cuda"}, "refused"),
        ({"backend": "jax", "implementation": "array"}, "refused"),
        ({"backend": "cupy", "implementation": "cuda", "cuda_mode": "bogus"}, "cuda_mode"),
        ({"backend": "cupy", "implementation": "cuda", "transport_implementation": "numba"}, "numba"),
        ({"backend": "numpy", "implementation": "array", "root_solver": "secant"}, "root_solver"),
    ):
        with pytest.raises((SedimentEventError, Exception), match=match):
            run_plot1_sediment_event(missing, out, **kwargs)
    assert not out.exists() and not any(p.name.startswith(".out") for p in tmp_path.iterdir())
    assert main(["--case-dir", str(missing), "--output-dir", str(out), "--implementation", "cuda",
                 "--backend", "numpy"]) == 1
    assert main(["--case-dir", str(missing), "--output-dir", str(out), "--implementation", "array",
                 "--backend", "cupy"]) == 1
    assert not out.exists()


# =====================================================================================================================
# GPU: actual CuPy beds through the MAPLE API with the explicit CUDA hydrology
# =====================================================================================================================
@pytest.mark.parametrize("root_solver", ROOT_SOLVERS)
def test_cuda_event_matches_the_host_sweeps_on_a_committing_valley(gpu, root_solver):
    schedule = constant_rainfall(0.0, 40.0, 72.0)  # rain, then recession; interval commits every 10 s
    ref_inputs, ref_control = commit_valley(np, "array", root_solver)
    ref = run(ref_inputs, schedule, 60.0, ref_control, cadence=20.0)
    _closure, bc = assert_closed_any(ref, ref_inputs["state0"])
    assert ref.n_commits >= 5 and ref.n_graph_changes >= 1 and bc["actual_pickup"].sum() > 0.0
    assert bc["deposition_actual"].sum() > 0.0 and bc["export_actual"].sum() > 0.0
    dev_inputs, dev_control = commit_valley(gpu, "cuda", root_solver)
    assert dev_inputs["state0"].graph.xp is gpu and dev_inputs["state0"].phase.xp is gpu
    hyd = CudaHydraulics()
    dev = run(dev_inputs, schedule, 60.0, dev_control, cadence=20.0, hydraulics=hyd)
    assert_closed_any(dev, dev_inputs["state0"])
    compare_events(ref, dev)
    # lifecycle: one context per graph binding (initial + one per commit), none per attempt
    assert hyd.n_preparations == 1 + dev.n_commits and hyd.n_binding_hits == dev.n_accepted_steps + dev.n_rejected_attempts
    assert [e["reason"] for e in hyd.log] == ["initial"] + ["commit"] * dev.n_commits
    assert hyd.is_bound_to(dev.state.graph, dev_inputs["column"]) and hyd.n_commits_recorded == dev.n_commits
    ctx_summary = hyd.summary()["context"]
    assert ctx_summary["graph_input_sha256"] == dev.state.graph.input_sha256 and ctx_summary["device_resident"]
    assert hyd.commit_device_to_host_bytes >= dev.n_commits * 20 * 8  # one committed-elevation download per commit
    # the whole terrain actually changed and the device graph follows the committed MAPLE surface
    from maple.core.types.topographic_commit import committed_elevation_m

    dz = H(committed_elevation_m(dev.state.bed.committed_topography)) - dev.state.terrain.initial_committed_elevation_m
    assert float(np.abs(dz).max()) > 0.0
    assert not np.array_equal(H(dev.state.graph.slope), H(dev_inputs["state0"].graph.slope))
    if root_solver == "bisection":
        # the generic array reference on CuPy arrays (no CUDA kernels) agrees with the host at the same bounds
        arr_inputs, arr_control = commit_valley(gpu, "array")
        arr = run(arr_inputs, schedule, 60.0, arr_control, cadence=20.0)
        assert_closed_any(arr, arr_inputs["state0"])
        compare_events(ref, arr)
    else:
        with pytest.raises(SedimentEventError, match="newton"):
            arr_inputs, arr_control = commit_valley(gpu, "array", "newton")
            run(arr_inputs, schedule, 60.0, arr_control, cadence=20.0)


@NUMBA_SKIP
@pytest.mark.parametrize("root_solver", ROOT_SOLVERS)
def test_cuda_event_matches_the_numba_host_sweep(gpu, root_solver):
    schedule = constant_rainfall(0.0, 40.0, 72.0)
    nb_inputs, nb_control = commit_valley(np, "numba", root_solver, transport_implementation="array")
    nb = run(nb_inputs, schedule, 60.0, nb_control, cadence=20.0)
    assert_closed_any(nb, nb_inputs["state0"])
    dev_inputs, dev_control = commit_valley(gpu, "cuda", root_solver)
    dev = run(dev_inputs, schedule, 60.0, dev_control, cadence=20.0)
    assert_closed_any(dev, dev_inputs["state0"])
    compare_events(nb, dev)


def test_cuda_limited_supply_and_class_selective_refill_on_the_device(gpu):
    """phi_1 at 0.1 %: MAPLE caps the removal at the available active mass and refills from the column; phi_4
    absent. Same assertions as the host test, plus host agreement at the declared bounds."""
    fractions = (0.001, 0.15, 0.25, 0.0, 0.35, 0.249)
    schedule = constant_rainfall(0.0, 40.0, 200.0)
    results = {}
    for xp, implementation in ((np, "array"), (gpu, "cuda")):
        bed, ctx = make_bed(valley_elevation(4, 5), fractions)
        control = control_for(implementation)
        inputs = setup_xp(bed, ctx, xp, south_ring=valley_elevation(5, 5)[0] - 0.01, control=control)
        results[implementation] = (inputs, run(inputs, schedule, 40.0, control, cadence=40.0))
    inputs, r = results["cuda"]
    state0 = inputs["state0"]
    initial_active = H(state0.bed.active_layer.mass_kg).sum(axis=(0, 1))
    _closure, bc = assert_closed_any(r, state0)
    assert bc["requested_pickup"][3] == 0.0 and bc["actual_pickup"][3] == 0.0
    assert not np.any(H(r.state.bed.water.mobile_mass_by_cell_class_kg)[..., 3])
    assert not np.any(H(r.cumulative_deposition_kg)[..., 3]) and bc["export_actual"][3] == 0.0
    assert bc["requested_pickup"][0] > initial_active[0] and bc["availability_shortfall"][0] > 0.0
    assert bc["holdings_shortfall"][0] == 0.0 and 0.0 < bc["actual_pickup"][0] < bc["requested_pickup"][0]
    assert bc["actual_pickup"][0] + bc["availability_shortfall"][0] == pytest.approx(bc["requested_pickup"][0], rel=1e-9)
    assert np.all(bc["availability_shortfall"][1:] == 0.0)
    np.testing.assert_allclose(bc["actual_pickup"][1:], bc["requested_pickup"][1:], rtol=1e-12)
    final_active = H(r.state.bed.active_layer.mass_kg).sum(axis=(0, 1))
    assert final_active[0] < initial_active[0]
    # refill: every cell that lost mass is back at the active-layer target, fed from the column below
    g = inputs["ctx"].geometry
    target = g.active_layer_thickness_m * g.bulk_density_kg_m3 * g.dx_m * g.dy_m
    net = H(r.cumulative_deposition_kg) - H(r.cumulative_pickup_kg)
    lost = net.sum(axis=-1) < 0.0
    assert lost.any()
    np.testing.assert_allclose(H(r.state.bed.active_layer.mass_kg).sum(axis=-1), target, atol=1e-8)
    column0, column1 = H(state0.bed.voxel_column.mass_kg).sum(axis=2), H(r.state.bed.voxel_column.mass_kg).sum(axis=2)
    assert np.all(column1.sum(axis=-1)[lost] < column0.sum(axis=-1)[lost])
    compare_events(results["array"][1], r)


def test_cuda_drying_chain_settles_dry_cells_and_exports_fines(gpu):
    schedule, end = constant_rainfall(0.0, 60.0, 72.0), 180.0
    results = {}
    for xp, implementation in ((np, "array"), (gpu, "cuda")):
        bed, ctx = make_bed(chain_elevation(6))
        control = control_for(implementation)
        inputs = setup_xp(bed, ctx, xp, ksat=1e-5, control=control)
        results[implementation] = (inputs, run(inputs, schedule, end, control, cadence=60.0))
    inputs, r = results["cuda"]
    closure, bc = assert_closed_any(r, inputs["state0"])
    assert np.all(H(r.state.storm.depth_m) == 0.0) and not np.any(H(r.state.bed.water.mobile_mass_by_cell_class_kg))
    assert int(H(r.regime_cell_steps["dry"])) > 0 and int(H(r.regime_cell_steps["diffuse"])) > 0
    assert np.all(bc["actual_pickup"] > 0.0)
    ratio = bc["export_actual"] / np.where(bc["actual_pickup"] > 0.0, bc["actual_pickup"], 1.0)
    assert np.all(np.diff(ratio) <= 0.0) and ratio[0] > ratio[1] > 0.0 and ratio[5] == 0.0
    exported, picked = np.asarray(closure["export_actual_kg"]), bc["actual_pickup"]
    assert exported[:2].sum() / exported.sum() > picked[:2].sum() / picked.sum()
    np.testing.assert_allclose(H(r.cumulative_deposition_kg)[..., 5], H(r.cumulative_pickup_kg)[..., 5], rtol=1e-3)
    sed = H(r.sediment_hydrograph)
    assert sed[-1, SCOL["mobile_kg"]] == 0.0 and sed[-1, SCOL["cumulative_export_kg"]] == pytest.approx(exported.sum(), rel=1e-12)
    compare_events(results["array"][1], r)


def test_cuda_rejections_retry_from_the_same_state_without_re_preparation(gpu):
    storm = {"max_dt_s": 8.0, "min_dt_s": 1.0 / 64.0, "max_retries": 10}
    schedule = constant_rainfall(0.0, 16.0, 0.0)
    results = {}
    for xp, implementation in ((np, "array"), (gpu, "cuda")):
        bed, ctx = make_bed(chain_elevation(4), depth_m=0.05)
        control = control_for(implementation, storm=storm)
        inputs = setup_xp(bed, ctx, xp, ksat=0.0, control=control)
        hyd = CudaHydraulics() if implementation == "cuda" else None
        results[implementation] = (inputs, run(inputs, schedule, 16.0, control, cadence=16.0, hydraulics=hyd), hyd)
    inputs, r, hyd = results["cuda"]
    assert r.n_rejected_attempts > 0 and r.rejections[0]["kind"] == "RoutingStepRejected" and r.state.t_s == 16.0
    assert_closed_any(r, inputs["state0"])
    assert hyd.n_preparations == 1 + r.n_commits and hyd.n_binding_hits == r.n_accepted_steps + r.n_rejected_attempts
    compare_events(results["array"][1], r)


def test_cuda_failed_maple_call_commit_or_preparation_publishes_nothing_on_the_device(gpu, monkeypatch):
    inputs, control = commit_valley(gpu, "cuda")
    state0 = inputs["state0"]
    saved = snapshot_any(state0)
    schedule = constant_rainfall(0.0, 30.0, 72.0)
    real_commit = sediment_event.commit_bed
    calls = []

    def failing_commit(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RoutingGraphError("unsupported sinks (test: a commit created a pit)")
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(sediment_event, "commit_bed", failing_commit)
    with pytest.raises(RoutingGraphError, match="sinks"):
        run(inputs, schedule, 30.0, control, cadence=30.0)
    compare_any(state0, saved)
    monkeypatch.undo()
    real_demand = sediment_event.apply_bed_demand
    seen = []

    def failing_demand(*args, **kwargs):
        seen.append(1)
        if len(seen) == 4:  # the deposit / export call of the second step
            raise ValueError("MAPLE refused the result (test)")
        return real_demand(*args, **kwargs)

    monkeypatch.setattr(sediment_event, "apply_bed_demand", failing_demand)
    with pytest.raises(ValueError, match="refused"):
        run(inputs, schedule, 30.0, control, cadence=30.0)
    compare_any(state0, saved)
    monkeypatch.undo()
    real_prepare = sediment_event._prepare_cuda_context
    prepared = []

    def failing_prepare(graph, column, *, mode):
        if len(prepared) == 1:  # the first commit's re-preparation
            raise RuntimeError("device lost (test)")
        prepared.append(1)
        return real_prepare(graph, column, mode=mode)

    monkeypatch.setattr(sediment_event, "_prepare_cuda_context", failing_prepare)
    hyd = CudaHydraulics()
    with pytest.raises(RuntimeError, match="device lost"):
        run(inputs, schedule, 30.0, control, cadence=30.0, hydraulics=hyd)
    compare_any(state0, saved)
    assert hyd.n_preparations == 1 and hyd.is_bound_to(state0.graph, inputs["column"])
    monkeypatch.undo()
    # the same inputs still run to completion afterwards (nothing was poisoned)
    r = run(inputs, schedule, 30.0, control, cadence=30.0)
    assert_closed_any(r, state0)
    compare_any(state0, saved)


def test_cuda_refusals_before_any_kernel_or_maple_call_on_the_device(gpu, monkeypatch):
    bed, ctx = make_bed(valley_elevation(4, 5))
    inputs = setup_xp(bed, ctx, gpu, south_ring=valley_elevation(5, 5)[0] - 0.01)
    state0 = inputs["state0"]
    saved = snapshot_any(state0)
    schedule = constant_rainfall(0.0, 30.0, 72.0)

    def boom(*args, **kwargs):
        raise AssertionError("a kernel or MAPLE call ran after a refused binding")

    monkeypatch.setattr(sediment_event, "apply_bed_demand", boom)
    monkeypatch.setattr(sediment_event, "_prepare_cuda_context", boom)
    monkeypatch.setattr(sediment_event, "coupled_step", boom)
    with pytest.raises(SedimentEventError, match="numba"):
        run(inputs, schedule, 10.0, control_for("numba"))
    with pytest.raises(SedimentEventError, match="numba"):
        run(inputs, schedule, 10.0, control_for("array", transport_implementation="numba"))
    with pytest.raises(SedimentEventError, match="newton"):
        run(inputs, schedule, 10.0, control_for("array", "newton"))
    rate = inputs["field"].apply(72.0 / 3.6e6)
    with pytest.raises(SedimentEventError, match="numba"):
        sediment_coupled_step(state0, inputs["ctx"], inputs["column"], rate, inputs["vegetation"], inputs["sediment"],
                              1.0, control_for("numba"))
    compare_any(state0, saved)


def test_cuda_repeat_is_bitwise_and_the_convenience_step_equals_the_lifecycle_step(gpu):
    inputs, control = commit_valley(gpu, "cuda")
    schedule = constant_rainfall(0.0, 40.0, 72.0)
    a = run(inputs, schedule, 60.0, control, cadence=20.0)
    b = run(inputs, schedule, 60.0, control, cadence=20.0, hydraulics=CudaHydraulics())
    compare_bitwise(a, b)
    for name in ("max_transport_cell_residual_kg", "max_phase_reconciliation_residual_kg", "numerical_residual_scalar_abs_kg"):
        assert float(H(getattr(a, name))) == float(H(getattr(b, name)))
    # one accepted attempt: the convenience form (context per invocation) and a held context give the same step
    state = a.state
    rate = inputs["field"].apply(0.0)
    hyd = CudaHydraulics()
    held = sediment_coupled_step(state, inputs["ctx"], inputs["column"], rate, inputs["vegetation"], inputs["sediment"],
                                 1.0, control, hydraulics=hyd)
    loose = sediment_coupled_step(state, inputs["ctx"], inputs["column"], rate, inputs["vegetation"], inputs["sediment"],
                                  1.0, control)
    assert hyd.n_preparations == 1 and hyd.is_bound_to(state.graph, inputs["column"])
    for x, y in ((held.storm.state.depth_m, loose.storm.state.depth_m),
                 (held.storm.state.discharge_m2_s, loose.storm.state.discharge_m2_s),
                 (held.bed.water.mobile_mass_by_cell_class_kg, loose.bed.water.mobile_mass_by_cell_class_kg),
                 (held.bed.active_layer.mass_kg, loose.bed.active_layer.mass_kg)):
        np.testing.assert_array_equal(H(x), H(y))
    assert held.storm.route.implementation == "cuda" and loose.storm.route.implementation == "cuda"


def test_cuda_plot1_short_window_runner_matches_the_host_run(plot1_case, gpu, tmp_path):
    host = run_plot1_sediment_event(plot1_case, tmp_path / "host", max_dt_s=1.0, end_s=60.0, implementation="array",
                                    report_every_s=30.0)
    dev = run_plot1_sediment_event(plot1_case, tmp_path / "gpu", max_dt_s=1.0, end_s=60.0, implementation="cuda",
                                   backend="cupy", report_every_s=30.0)
    assert dev.state.graph.xp is gpu and dev.result.state.bed.active_layer.mass_kg.__class__ is gpu.ndarray
    compare_events(host.result, dev.result)
    s = dev.summary
    assert s["backend"]["backend"] == "cupy" and s["routing"]["implementation"] == "cuda"
    assert s["backend"]["pairing"]["hydraulics"] == "cuda" and s["backend"]["pairing"]["transport_implementation"] == "array"
    g = s["backend"]["gpu"]
    assert g["executed"] is True and g["memory_preflight"]["accepted"] is True
    assert g["cuda_hydraulics"]["n_preparations"] == 1 + s["commits"]["n_commits"]
    assert g["cuda_hydraulics"]["commits"]["n_recorded"] == s["commits"]["n_commits"]
    assert g["cuda_hydraulics"]["context"]["graph_input_sha256"] == dev.state.graph.input_sha256
    assert g["device_memory"]["pool_total_bytes_after_loop"] > 0 and "no speed claim" in g["claims"]
    assert s["resolved_config"]["syrup"]["cuda_mode"] == "auto" and s["resolved_config"]["syrup"]["backend"] == "cupy"
    assert "cuda_mode" not in host.summary["resolved_config"]["syrup"]  # host config hash unchanged
    assert s["backend"]["loop_transfer_counters"]["device_to_host"] > 0  # packet reads, MAPLE flags, commits
    assert any("no speed claim" in item for item in s["limitations"])
    on_disk = json.loads((tmp_path / "gpu" / SUMMARY_NAME).read_text())
    assert on_disk["backend"]["gpu"]["cuda_hydraulics"]["n_preparations"] == g["cuda_hydraulics"]["n_preparations"]
    assert host.summary["backend"]["gpu"] == "not exercised; no GPU claim"
    assert "refused" in host.summary["backend"]["transfer_boundaries"]
    assert s["sediment"]["closure"]["closed"] and abs(s["budget"]["water_residual_m3"]) <= s["budget"]["tolerance_m3"]
