"""Phase 5b coupled event (`maple_syrup.sediment_event`) on small REAL
MAPLE beds built with MAPLE's own constructors: bed exchange, active-supply
capping, refill, sorting with natural drying, junctions and outlets, water +
per-class closure, commit / reroute / depth continuity / post-commit
hydraulic diagnostics, morphology from actual inventories, failure
atomicity, forcing windows, retries, cadence-independent peaks, and a
continuously injected constant-v/L reference (discrete recursion exact,
first-order convergence to the analytic solution). Equation-level
references; nothing here executes MAHLERAN. No Plot 1 case (see
test_sediment_experiment.py)."""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest

pytest.importorskip("maple")

from maple_syrup import sediment_event
from maple_syrup.infiltration import column_parameters, initial_soil_water_m
from maple_syrup.rainfall import constant_rainfall, rainfall_field
from maple_syrup.routing import RoutingGraphError, build_routing_graph
from maple_syrup.sediment_bed import (
    BedContext,
    BedState,
    bed_inventory,
    terrain_reference,
)
from maple_syrup.sediment_event import (
    SedimentEventControl,
    SedimentEventError,
    evolve_sediment_event,
    initial_event_state,
    morphology_summary,
    sediment_coupled_step,
    sediment_hydrograph_columns,
)
from maple_syrup.sediment_physics import (
    REGIME_CODES,
    SedimentPhysicsStep,
    plot1_sediment_parameters,
)
from maple_syrup.sediment_transport import TransportStepRejected
from maple_syrup.storm import HYDROGRAPH_COLUMNS, StormControl, initial_state

EPS = np.finfo(np.float64).eps
NC = 6
CLASS_IDS = tuple(f"phi_{k}" for k in range(1, 7))
FRACTIONS = (0.10, 0.15, 0.25, 0.25, 0.15, 0.10)
COL = {name: i for i, name in enumerate(HYDROGRAPH_COLUMNS)}
SCOL = {name: i for i, name in enumerate(sediment_hydrograph_columns(NC))}


# --- real MAPLE beds through MAPLE's public constructors ------------------------------------------------
def arrays(value):
    if isinstance(value, np.ndarray):
        yield value
    elif is_dataclass(value):
        for f in fields(value):
            yield from arrays(getattr(value, f.name))


def make_bed(elevation_m, fractions=FRACTIONS, *, dx=0.5, n_voxels=3, voxel_dz=0.1, bulk=1250.0,
             active_thickness=0.002, depth_m=0.0, commit_spec=None):
    """Uniform-per-cell mixture bed on `elevation_m` (ny, nx): the same
    MAPLE sequence `maple_syrup.probe.build_minimal_state` uses, plus the
    committed topography, full availability and an empty ledger."""
    from maple.core.boundaries import AxisBoundary, BoundaryKind
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.geometry import GeometrySpec, validate_geometry
    from maple.core.parameters.grain_classes import (
        GrainClass,
        GrainClassSet,
        validate_grain_classes,
    )
    from maple.core.parameters.numerics import DEFAULT_MASS_RESOLUTION_KG
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.parameters.water_coupling import MAHLERAN_1_2_1_CLASS_DIAMETERS_M
    from maple.core.types.sediment_availability import (
        sediment_availability_state_from_active_layer,
    )
    from maple.core.types.sediment_ledger import zeros_sediment_ledger_state
    from maple.core.types.voxel import zeros_voxel_column_state
    from maple.core.types.water import water_state_from_depth
    from maple.surface.active_layer import initialize_active_layer_from_voxels
    from maple.surface.topographic_commit import (
        initial_committed_topography_state_from_physical_state,
    )
    from maple.surface.voxels import deposit_surface_mixture_batch

    z = np.asarray(elevation_m, dtype=np.float64)
    ny, nx = z.shape
    f = np.asarray(fractions, dtype=np.float64)
    f = np.broadcast_to(f, (ny, nx, NC)) if f.ndim == 1 else f
    assert np.allclose(f.sum(axis=-1), 1.0)
    grain_classes = GrainClassSet(classes=tuple(
        GrainClass(class_id=c, diameter_m=float(d), particle_density_kg_m3=2650.0, is_aggregate=False)
        for c, d in zip(CLASS_IDS, MAHLERAN_1_2_1_CLASS_DIAMETERS_M, strict=True)))
    validate_grain_classes(grain_classes)

    def boundary():
        return AxisBoundary(kind=BoundaryKind.PRESCRIBED, inflow_flux_kg_m_s={c: 0.0 for c in CLASS_IDS})

    geometry = GeometrySpec(nx=nx, ny=ny, dx_m=dx, dy_m=dx, boundary_x=boundary(), boundary_y=boundary(),
                            voxel_dz_m=voxel_dz, bulk_density_kg_m3=bulk, active_layer_thickness_m=active_thickness)
    validate_geometry(geometry, grain_classes.ids())
    mres = float(DEFAULT_MASS_RESOLUTION_KG)
    assert float(z.max()) < n_voxels * voxel_dz
    request = (z * bulk * dx * dx)[:, :, None] * f
    column = zeros_voxel_column_state(ny, nx, n_voxels, NC)
    column, _ = deposit_surface_mixture_batch(column, request, geometry, mres)
    column, active, init = initialize_active_layer_from_voxels(column, geometry, mres)
    assert not np.any(init.underfilled_mask)
    depth = np.full((ny, nx), float(depth_m)) if np.ndim(depth_m) == 0 else np.asarray(depth_m, dtype=np.float64)
    water = water_state_from_depth(depth, NC)
    ledger = zeros_sediment_ledger_state(ny, nx, NC)
    committed = initial_committed_topography_state_from_physical_state(column, active, geometry, mres)
    available = sediment_availability_state_from_active_layer(active.mass_kg, np.ones_like(active.mass_kg))
    state = BedState(column, active, water, ledger, committed, available)
    context = BedContext(geometry, grain_classes, mres, commit_spec or TopographicCommitSpec(),
                         AvalanchingSpec(enabled=False), np.ones_like(active.mass_kg))
    return state, context


def chain_elevation(n, *, z0=0.12, dz=0.01):
    return z0 + dz * np.arange(n, dtype=np.float64)[:, None]


def valley_elevation(ny, nx, *, z0=0.12, sy=0.01, sx=0.02):
    rows, cols = np.indices((ny, nx)).astype(np.float64)
    return z0 + sy * rows + sx * np.abs(cols - (nx - 1) / 2)


def graph_for(state, *, ff=1.0, walls=1.0, south_ring=None):
    """Ring around the committed MAPLE terrain: high walls N/E/W, the south
    ring row exports (either the given values or the interior row 0 minus
    0.02 m, which makes every row-0 cell an outlet)."""
    from maple.core.types.topographic_commit import committed_elevation_m

    zc = np.asarray(committed_elevation_m(state.committed_topography))
    ny, nx = zc.shape
    z = np.full((ny + 2, nx + 2), float(zc.max()) + walls)
    z[1:-1, 1:-1] = zc
    z[0, 1:-1] = zc[0] - 0.02 if south_ring is None else south_ring
    exports = np.zeros(z.shape, dtype=bool)
    exports[0, 1:-1] = True
    graph = build_routing_graph(z, exports, np.full((ny, nx), float(ff)), 0.5)
    return graph, terrain_reference(z, exports, state, graph)


def column_for(graph, ksat, *, theta0=0.25, theta_sat=0.4, thickness=0.3):
    def full(v):
        return np.full(graph.shape, float(v))

    params = column_parameters(model="fixed_ksat", ksat_m_per_s=full(ksat), suction_m=full(0.0),
                               drainage_parameter=full(0.0), theta_sat=full(theta_sat), soil_thickness_m=full(thickness))
    return params, initial_soil_water_m(params, full(theta0))


def setup(bed, ctx, *, ff=1.0, ksat=1e-7, south_ring=None):
    graph, terrain = graph_for(bed, ff=ff, south_ring=south_ring)
    column, soil0 = column_for(graph, ksat)
    state0 = initial_event_state(graph, terrain, bed, ctx, soil0)
    return {"state0": state0, "column": column, "field": rainfall_field(*graph.shape),
            "vegetation": np.zeros(graph.shape), "sediment": plot1_sediment_parameters(), "ctx": ctx}


def run(inputs, schedule, end, *, control=None, cadence=10.0, state=None):
    control = control or SedimentEventControl()
    return evolve_sediment_event(state or inputs["state0"], inputs["ctx"], inputs["column"], inputs["field"],
                                 schedule, inputs["vegetation"], inputs["sediment"], end, control,
                                 report_every_s=cadence)


def valley_case(ny=4, nx=5, **kwargs):
    bed, ctx = make_bed(valley_elevation(ny, nx), **kwargs)
    inputs = setup(bed, ctx, south_ring=valley_elevation(ny + 1, nx)[0] - 0.01)
    graph = inputs["state0"].graph
    assert int(graph.outlet.sum()) == 1 and graph.outlet[0, nx // 2]  # a single junction outlet
    return inputs


def drying_chain_case():
    """Codex-verified fixture (fixture_probe.log): 6-cell chain, ksat 1e-5
    m/s, 60 s of 72 mm/h, end 180 s -> every cell drains naturally, final
    mobile mass 0, phi2 export/pickup ~0.054, phi6 ~4e-8."""
    bed, ctx = make_bed(chain_elevation(6))
    return setup(bed, ctx, ksat=1e-5), constant_rainfall(0.0, 60.0, 72.0), 180.0


def within(values, tolerance):
    """Elementwise |values| <= tolerance (vector tolerances; NumPy 2.5's
    assert_allclose cannot format an array atol)."""
    values, tolerance = np.asarray(values, dtype=np.float64), np.asarray(tolerance, dtype=np.float64)
    assert values.shape == tolerance.shape or tolerance.ndim == 0
    ok = np.abs(values) <= tolerance
    assert np.all(ok), f"residual {values} exceeds tolerance {tolerance}"


def assert_closed(result, initial_state_, *, check_dry_settling=True):
    """Water and per-class sediment closure, refusals within resolution,
    depth ownership, transport budget bounds and the independent bed
    inventory. `check_dry_settling` asserts that hydraulically dry cells
    hold no wet-mobile mass (the real laws settle dry cells); the
    constant-law stub deliberately permits it and passes False."""
    closure = result.closure()
    assert closure["closed"] and closure["request_reconciled"]
    within(closure["residual_kg"], closure["tolerance_kg"])
    bc = {k: np.asarray(v) for k, v in result.by_class.items()}
    n = result.n_accepted_steps
    mres = 1e-10
    cells = result.state.graph.shape[0] * result.state.graph.shape[1]
    allowance = 2 * n * cells * mres + 64 * EPS * n * np.maximum(bc["deposition_requested"] + bc["export_requested"], 1.0)
    within(bc["deposition_unmet"], allowance)
    within(bc["export_unmet"], allowance)
    within(bc["transport_budget_residual"], bc["transport_budget_tolerance"])
    assert np.all(bc["actual_pickup"] <= bc["requested_pickup"] * (1 + 4 * EPS))
    st = result.state
    np.testing.assert_array_equal(st.bed.water.depth_m, st.storm.depth_m)
    # water identity (m3)
    area = st.graph.dx_m ** 2
    h0, s0 = float(initial_state_.storm.depth_m.sum()), float(initial_state_.storm.soil_water_m.sum())
    lhs = float(st.storm.depth_m.sum() + st.storm.soil_water_m.sum() + result.cumulative_drainage_m.sum()) * area \
        + float(result.cumulative_export_m3)
    rhs = (h0 + s0 + float(result.cumulative_rain_m.sum())) * area
    assert abs(lhs - rhs) <= 1e-9 * max(rhs, 1.0)
    # independent inventories: with avalanching disabled the actual bed change IS the water exchange
    change = np.asarray(result.bed_change_by_cell_class_kg())
    exchange = np.asarray(result.cumulative_deposition_kg) - np.asarray(result.cumulative_pickup_kg)
    within(change - exchange, 4 * cells * mres * (n + 1) + 64 * EPS * (n + 1) * np.maximum(np.abs(exchange), 1e-3))
    if check_dry_settling:
        dry = st.storm.depth_m == 0.0
        assert not np.any(st.bed.water.mobile_mass_by_cell_class_kg[dry])
    return closure, bc


# --- real bed exchange, closure, supply capping, refill, sorting, junctions -----------------------------------
def test_short_storm_on_a_real_valley_bed_closes_water_and_every_class():
    inputs = valley_case()
    state0 = inputs["state0"]
    before = [a.copy() for a in arrays(state0.bed)] + [state0.storm.depth_m.copy(), state0.sediment_velocity_m_s.copy()]
    schedule = constant_rainfall(0.0, 40.0, 72.0)
    r = run(inputs, schedule, 60.0, cadence=20.0)
    closure, bc = assert_closed(r, state0)
    assert r.n_accepted_steps == 60 and r.n_rejected_attempts == 0 and r.state.t_s == 60.0
    assert bc["requested_pickup"].sum() > 0.0 and bc["actual_pickup"].sum() > 0.0
    assert bc["deposition_actual"].sum() > 0.0 and bc["export_actual"].sum() > 0.0
    assert np.all(bc["raindrop_pickup_requested"] > 0.0) and not np.any(bc["flow_pickup_requested"])  # Re < 500
    # the fixture is wet with rain (diffuse), then wet without rain (recession); it never dries
    assert r.regime_cell_steps["diffuse"] > 0 and r.regime_cell_steps["wet_no_law"] > 0
    assert r.regime_cell_steps["dry"] == 0 and np.all(r.state.storm.depth_m > 0.0)
    # gross ledgers from MAPLE ground truth agree with the per-cell grids
    np.testing.assert_allclose(r.cumulative_pickup_kg.sum(axis=(0, 1)), bc["actual_pickup"], rtol=1e-12)
    np.testing.assert_allclose(r.cumulative_deposition_kg.sum(axis=(0, 1)), bc["deposition_actual"], rtol=1e-12)
    # net bed change from actual inventories equals deposition - pickup within the reported residuals
    net = np.asarray(closure["net_bed_change_kg"])
    within(net - (bc["deposition_actual"] - bc["actual_pickup"]), closure["tolerance_kg"])
    # rain integrates exactly over the simulated window
    assert float(r.cumulative_rain_m.sum()) == pytest.approx(schedule.depth_m(0.0, 60.0) * 20, rel=1e-12)
    # hydrographs: Phase 4 water columns plus sediment columns at every boundary
    hyd, sed = np.asarray(r.hydrograph), np.asarray(r.sediment_hydrograph)
    assert hyd.shape == (3, len(HYDROGRAPH_COLUMNS)) and sed.shape == (3, len(sediment_hydrograph_columns(NC)))
    np.testing.assert_array_equal(sed[:, SCOL["t_s"]], [20.0, 40.0, 60.0])
    assert np.all(np.diff(sed[:, SCOL["cumulative_pickup_kg"]]) >= 0.0)
    assert sed[-1, SCOL["cumulative_export_kg"]] == pytest.approx(bc["export_actual"].sum(), rel=1e-12)
    assert sed[-1, SCOL["mobile_kg"]] == pytest.approx(sum(closure["final_mobile_kg"]), rel=1e-12)
    assert float(r.peak_mobile_kg) >= sed[:, SCOL["mobile_kg"]].max()
    # inputs untouched (frozen state; MAPLE and SYRUP kernels are pure)
    after = [a for a in arrays(state0.bed)] + [state0.storm.depth_m, state0.sediment_velocity_m_s]
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)
    # forced final commit: terrain corresponds to the bed, nothing pending
    assert not np.any(r.state.bed.ledger.pending_bed_mass_change_kg)
    assert r.n_commits >= 1 and r.state.bed.committed_topography.commit_count == r.n_commits


def test_absent_class_gets_nothing_and_active_supply_caps_the_pickup():
    """phi_1 at 0.1 % of the mixture: MAPLE caps each step's removal at the
    AVAILABLE active-layer mass (all holdings are declared available, so the
    shortage is reported as availability shortfall, not holdings shortfall)
    and refills from the column; phi_4 is absent everywhere."""
    fractions = (0.001, 0.15, 0.25, 0.0, 0.35, 0.249)
    bed, ctx = make_bed(valley_elevation(4, 5), fractions)
    inputs = setup(bed, ctx, south_ring=valley_elevation(5, 5)[0] - 0.01)
    state0 = inputs["state0"]
    initial_active = np.asarray(state0.bed.active_layer.mass_kg).sum(axis=(0, 1))
    r = run(inputs, constant_rainfall(0.0, 40.0, 200.0), 40.0, cadence=40.0)
    closure, bc = assert_closed(r, state0)
    # absent class: no demand, no pickup, never mobile
    assert bc["requested_pickup"][3] == 0.0 and bc["actual_pickup"][3] == 0.0
    assert not np.any(r.state.bed.water.mobile_mass_by_cell_class_kg[..., 3])
    assert not np.any(r.cumulative_deposition_kg[..., 3]) and bc["export_actual"][3] == 0.0
    # supply-capped class: demand exceeds what the active layer holds; the refusal is the
    # availability shortfall, the actual pickup is what was held, and the request reconciles
    assert bc["requested_pickup"][0] > initial_active[0] and bc["availability_shortfall"][0] > 0.0
    assert bc["holdings_shortfall"][0] == 0.0  # everything held was available; nothing more was refused
    assert 0.0 < bc["actual_pickup"][0] < bc["requested_pickup"][0]
    assert bc["actual_pickup"][0] + bc["availability_shortfall"][0] == pytest.approx(bc["requested_pickup"][0], rel=1e-9)
    assert closure["request_reconciled"]
    # the other classes were never supply-limited
    assert np.all(bc["availability_shortfall"][1:] == 0.0)
    np.testing.assert_allclose(bc["actual_pickup"][1:], bc["requested_pickup"][1:], rtol=1e-12)
    # the active layer's phi_1 is depleted below its start while the class still leaves / deposits
    final_active = np.asarray(r.state.bed.active_layer.mass_kg).sum(axis=(0, 1))
    assert final_active[0] < initial_active[0] and closure["final_bed_kg"][0] < np.asarray(bed_inventory(state0.bed))[0]


def test_active_layer_is_refilled_from_the_voxel_column():
    inputs = valley_case()
    state0 = inputs["state0"]
    g = inputs["ctx"].geometry
    target = g.active_layer_thickness_m * g.bulk_density_kg_m3 * g.dx_m * g.dy_m
    r = run(inputs, constant_rainfall(0.0, 40.0, 72.0), 40.0, cadence=40.0)
    assert_closed(r, state0)
    bed0, bed1 = state0.bed, r.state.bed
    active1 = np.asarray(bed1.active_layer.mass_kg)
    column0, column1 = np.asarray(bed0.voxel_column.mass_kg).sum(axis=2), np.asarray(bed1.voxel_column.mass_kg).sum(axis=2)
    # every cell that lost mass had its active layer refilled to the target from the column below
    net = np.asarray(r.cumulative_deposition_kg) - np.asarray(r.cumulative_pickup_kg)
    lost = net.sum(axis=-1) < 0.0
    assert lost.any()
    np.testing.assert_allclose(active1.sum(axis=-1), target, atol=1e-8)
    assert np.all(column1.sum(axis=-1)[lost] < column0.sum(axis=-1)[lost])
    # per cell, bed change = deposition - pickup (MAPLE ground truth), within resolution
    change = (column1 + active1) - (column0 + np.asarray(bed0.active_layer.mass_kg))
    np.testing.assert_allclose(change, net, atol=1e-8)
    np.testing.assert_array_equal(change, np.asarray(r.bed_change_by_cell_class_kg()))


def test_sorting_fines_export_and_coarse_classes_settle_locally_when_cells_dry():
    inputs, schedule, end = drying_chain_case()
    state0 = inputs["state0"]
    r = run(inputs, schedule, end, cadence=60.0)
    closure, bc = assert_closed(r, state0)
    # the chain drained naturally: every cell is dry, every pool settled into the MAPLE bed
    assert np.all(r.state.storm.depth_m == 0.0) and not np.any(r.state.bed.water.mobile_mass_by_cell_class_kg)
    assert r.regime_cell_steps["dry"] > 0 and r.regime_cell_steps["diffuse"] > 0
    ratio = bc["export_actual"] / np.where(bc["actual_pickup"] > 0.0, bc["actual_pickup"], 1.0)
    assert np.all(bc["actual_pickup"] > 0.0)
    # export fraction decreases monotonically with grain size: fines leave, coarse grains stay
    assert np.all(np.diff(ratio) < 0.0) and ratio[1] > 0.03 and ratio[5] < 1e-3 * ratio[1]
    exported = np.asarray(closure["export_actual_kg"])
    picked = bc["actual_pickup"]
    assert exported[:2].sum() / exported.sum() > picked[:2].sum() / picked.sum()  # exported load is finer
    # the coarsest class settles in the cell that picked it up (virtual velocity ~1e-7 m/s): per cell,
    # pickup = deposition within a 0.1 % transfer allowance, nothing retained mobile
    dep6 = np.asarray(r.cumulative_deposition_kg)[..., 5]
    pick6 = np.asarray(r.cumulative_pickup_kg)[..., 5]
    np.testing.assert_allclose(dep6, pick6, rtol=1e-3)
    assert bc["export_actual"][5] <= 1e-3 * picked[5]
    # morphology from actual inventories: net erosion of the chain equals the exported mass
    morph = morphology_summary(r.bed_change_by_cell_class_kg(),
                               np.asarray(r.cumulative_deposition_kg) - np.asarray(r.cumulative_pickup_kg),
                               bulk_density_kg_m3=1250.0, cell_area_m2=0.25)
    assert morph["net_erosion_kg"] - morph["net_deposition_kg"] == pytest.approx(exported.sum(), abs=1e-8)
    assert morph["non_water_bed_change_abs_kg"] <= 1e-8


def test_retained_wet_mobile_load_is_reported_not_forced_to_settle():
    """Rain to 60 s, still wet at 90 s (ksat 1e-7): coarse classes remain
    MOBILE in place at their slow virtual speed (physical v/L; no settling
    threshold). The load is retained and reported, not deposited early."""
    bed, ctx = make_bed(chain_elevation(6))
    inputs = setup(bed, ctx)
    state0 = inputs["state0"]
    r = run(inputs, constant_rainfall(0.0, 60.0, 72.0), 90.0, cadence=30.0)
    closure, bc = assert_closed(r, state0)
    assert np.all(r.state.storm.depth_m > 0.0) and r.regime_cell_steps["dry"] == 0
    mobile = np.asarray(r.state.bed.water.mobile_mass_by_cell_class_kg)
    assert mobile[..., 5].sum() > 0.5 * bc["actual_pickup"][5]  # phi_6 mostly still mobile (L / v_s ~ 400 s)
    assert sum(closure["final_mobile_kg"]) == pytest.approx(mobile.sum(), rel=1e-12)
    # per cell the coarse pool never left its cell: pickup = deposition + retained mobile
    dep6, pick6 = np.asarray(r.cumulative_deposition_kg)[..., 5], np.asarray(r.cumulative_pickup_kg)[..., 5]
    np.testing.assert_allclose(dep6 + mobile[..., 5], pick6, rtol=1e-3)


def test_branching_valley_moves_mass_to_the_junction_and_exports_only_at_the_outlet():
    inputs = valley_case(ny=4, nx=5)
    state0 = inputs["state0"]
    graph = state0.graph
    r = run(inputs, constant_rainfall(0.0, 40.0, 72.0), 60.0, cadence=60.0)
    _closure, bc = assert_closed(r, state0)
    export = np.asarray(r.cumulative_export_request_kg)
    assert export[graph.outlet].sum() > 0.0 and export[~graph.outlet].sum() == 0.0
    assert bc["export_actual"].sum() == pytest.approx(export.sum(), rel=1e-12)  # nothing refused
    centre = graph.shape[1] // 2
    deposition = np.asarray(r.cumulative_deposition_kg).sum(axis=-1)
    pickup = np.asarray(r.cumulative_pickup_kg).sum(axis=-1)
    assert deposition[:, centre].sum() > 0.0 and np.all(pickup > 0.0)
    # side cells feed the centre column: the centre receives more than it picks up somewhere
    assert np.any(deposition[:, centre] > pickup[:, centre]) or bc["export_actual"].sum() > 0.0


# --- morphology reporting from actual inventories ----------------------------------------------------------------
def test_morphology_separates_class_exchange_from_cell_mass_change():
    ny, nx, nc = 2, 2, 3
    exchange = np.zeros((ny, nx, nc))
    exchange[0, 0] = [1.0, -1.0, 0.0]  # balanced class exchange: sorting, zero cell mass change
    exchange[1, 1] = [0.0, 0.0, -0.25]  # genuine erosion of one class
    change = exchange.copy()
    avalanche = np.zeros_like(change)
    avalanche[0, 1, 2] = 0.5  # mass moved by a non-water process (avalanche inside a commit) ...
    avalanche[1, 0, 2] = -0.5  # ... from one cell to another, zero total
    change += avalanche
    m = morphology_summary(change, exchange, bulk_density_kg_m3=1250.0, cell_area_m2=0.25)
    assert m["class_sorting_exchange_kg"] == pytest.approx(1.0)
    assert m["water_exchange_net_erosion_kg"] == pytest.approx(0.25) and m["water_exchange_net_deposition_kg"] == 0.0
    assert m["net_erosion_kg"] == pytest.approx(0.75) and m["net_deposition_kg"] == pytest.approx(0.5)
    assert m["net_bed_change_kg"] == pytest.approx(-0.25) and m["n_cells_lowered"] == 2 and m["n_cells_raised"] == 1
    assert m["class_bed_change_kg"] == pytest.approx([1.0, -1.0, -0.25]) and m["class_water_exchange_kg"] == pytest.approx([1.0, -1.0, -0.25])
    assert m["non_water_bed_change_abs_kg"] == pytest.approx(1.0) and m["non_water_bed_change_max_abs_kg"] == pytest.approx(0.5)
    assert m["elevation_change_equivalent_m"]["min"] == pytest.approx(-0.5 / (1250.0 * 0.25))
    # a per-class split would have called the balanced exchange 1 kg of erosion AND 1 kg of deposition
    per_class = np.clip(-exchange, 0.0, None).sum()
    assert per_class == pytest.approx(1.25) and m["net_erosion_kg"] < per_class
    with pytest.raises(SedimentEventError):
        morphology_summary(change, exchange[..., :2], bulk_density_kg_m3=1250.0, cell_area_m2=0.25)


# --- commits: depth continuity, terrain refresh, rerouting, post-commit diagnostics, ledger reset ------------------
def test_commits_keep_depth_reroute_from_committed_terrain_and_refresh_hydraulic_diagnostics():
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.types.topographic_commit import committed_elevation_m

    bed, ctx = make_bed(valley_elevation(4, 5), commit_spec=TopographicCommitSpec(commit_interval_s=10.0))
    inputs = setup(bed, ctx, south_ring=valley_elevation(5, 5)[0] - 0.01)
    state0 = inputs["state0"]
    r = run(inputs, constant_rainfall(0.0, 45.0, 72.0), 45.0, cadence=15.0)  # rain through the end: pending exchange at 45 s
    closure, bc = assert_closed(r, state0)
    assert r.n_commits >= 5 and r.n_graph_changes >= 1 and len(r.commit_log) == r.n_commits
    times = [c["t_s"] for c in r.commit_log]
    assert times == sorted(times) and 10.0 in times and times[-1] == 45.0 and r.commit_log[-1]["forced"]
    assert r.n_forced_commits == 1
    for entry in r.commit_log:
        assert entry["depth_unchanged"] and entry["displaced_water_volume_m3"] == 0.0
        assert entry["graph_changed"] == (entry["max_abs_committed_elevation_change_m"] > 0.0)
        assert entry["n_outlets"] == 1 and entry["outlets_changed"] == 0
        assert any(stage.startswith("water_callback") for stage in entry["commit_stage_log"])
    # the final graph is exactly what the committed terrain implies (fixed ring, interior delta)
    st = r.state
    z = st.terrain.initial_full_elevation_m.copy()
    z[1:-1, 1:-1] += np.asarray(committed_elevation_m(st.bed.committed_topography)) - st.terrain.initial_committed_elevation_m
    rebuilt = build_routing_graph(z, st.terrain.export_receiver_full, st.terrain.friction_factor, st.terrain.dx_m,
                                  active_mask=st.terrain.active_mask)
    for name in ("aspect", "slope", "receiver", "outlet"):
        np.testing.assert_array_equal(getattr(rebuilt, name), getattr(st.graph, name))
    np.testing.assert_array_equal(rebuilt.conveyance, st.graph.conveyance)
    assert st.graph.input_sha256 != state0.graph.input_sha256 and st.network.graph_input_sha256 == st.graph.input_sha256
    assert not np.array_equal(st.graph.slope, state0.graph.slope)  # conveyance genuinely changed
    np.testing.assert_array_equal(st.grid.slope, st.graph.slope)
    # discharge was re-initialised from the new conveyance at the unchanged depth ...
    fresh = initial_state(st.graph, st.storm.depth_m, st.storm.soil_water_m, t_s=st.t_s)
    np.testing.assert_array_equal(st.storm.discharge_m2_s, fresh.discharge_m2_s)
    # ... and the hydraulic diagnostics follow the accepted post-commit state (Codex note 4)
    k_final = np.asarray(st.graph.conveyance).reshape(st.graph.shape)
    np.testing.assert_array_equal(np.asarray(r.last_velocity_m_s), np.sqrt(np.asarray(st.storm.depth_m)) * k_final)
    hyd = np.asarray(r.hydrograph)
    q_final = st.graph.dx_m * float(np.asarray(st.storm.discharge_m2_s)[st.graph.outlet].sum())
    assert hyd[-1, COL["outlet_discharge_m3_s"]] == pytest.approx(q_final, rel=1e-12)  # summation order only
    assert hyd[-1, COL["max_velocity_m_s"]] == float(np.asarray(r.last_velocity_m_s).max())
    assert np.all(np.asarray(r.peak_velocity_m_s) >= np.asarray(r.last_velocity_m_s))
    assert float(r.peak_outlet_discharge_m3_s) >= q_final * (1 - 1e-12)
    assert float(np.abs(committed_elevation_m(st.bed.committed_topography) - state0.terrain.initial_committed_elevation_m).max()) > 0.0
    # ledger totals captured before each reset: the water channel sums to deposition - pickup by class,
    # and the recognised pending mass equals the actual net bed change
    channel = np.asarray(r.ledger_process_totals_reset_kg)[sediment_event.water_channel_index()]
    expected = bc["deposition_actual"] - bc["actual_pickup"]
    within(channel - expected, np.asarray(closure["tolerance_kg"]) + 1e-10 * np.abs(expected))
    within(np.asarray(r.committed_net_bed_change_kg) - np.asarray(closure["net_bed_change_kg"]), closure["tolerance_kg"])
    assert not np.any(st.bed.ledger.pending_bed_mass_change_kg) and not np.any(st.bed.ledger.process_totals_kg)
    sed = np.asarray(r.sediment_hydrograph)
    assert np.all(np.diff(sed[:, SCOL["commit_count"]]) >= 0.0) and sed[-1, SCOL["commit_count"]] == r.n_commits


def test_forced_final_commit_only_when_exchange_is_pending():
    inputs = valley_case()
    state0 = inputs["state0"]
    control = SedimentEventControl(commit=False, force_final_commit=False)
    r = run(inputs, constant_rainfall(0.0, 30.0, 72.0), 30.0, control=control, cadence=30.0)
    assert_closed(r, state0)
    assert r.n_commits == 0 and r.state.graph is state0.graph and r.state.bed.committed_topography.commit_count == 0
    assert np.any(r.state.bed.ledger.pending_bed_mass_change_kg)  # pending, honestly unpublished
    forced = run(inputs, constant_rainfall(0.0, 30.0, 72.0), 30.0,
                 control=SedimentEventControl(commit=False, force_final_commit=True), cadence=30.0)
    assert forced.n_commits == forced.n_forced_commits == 1 and forced.state.graph is not state0.graph
    assert not np.any(forced.state.bed.ledger.pending_bed_mass_change_kg)
    # no rain, dry bed, nothing exchanged: the end commit is NOT forced artificially
    empty = run(inputs, constant_rainfall(0.0, 30.0, 0.0), 30.0,
                control=SedimentEventControl(force_final_commit=True), cadence=30.0)
    assert empty.n_commits == 0 and empty.state.bed.committed_topography.commit_count == 0
    assert empty.state.graph is state0.graph and not np.any(empty.state.bed.ledger.n_physical_touches)
    assert empty.by_class["actual_pickup"].sum() == 0.0 and float(empty.cumulative_export_m3) == 0.0


# --- failure atomicity -----------------------------------------------------------------------------------------------
def snapshot(state):
    return [a.copy() for a in arrays(state.bed)] + [a.copy() for a in (
        state.storm.depth_m, state.storm.soil_water_m, state.storm.discharge_m2_s, state.sediment_velocity_m_s)]


def compare(state, saved):
    current = list(arrays(state.bed)) + [state.storm.depth_m, state.storm.soil_water_m, state.storm.discharge_m2_s,
                                         state.sediment_velocity_m_s]
    for a, b in zip(saved, current, strict=True):
        np.testing.assert_array_equal(a, b)


def test_failed_commit_graph_or_maple_call_publishes_nothing_and_propagates(monkeypatch):
    inputs = valley_case()
    state0 = inputs["state0"]
    saved = snapshot(state0)
    schedule = constant_rainfall(0.0, 30.0, 72.0)
    real_commit = sediment_event.commit_bed
    calls = []

    def failing_commit(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RoutingGraphError("unsupported sinks (test: a commit created a pit)")
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(sediment_event, "commit_bed", failing_commit)
    with pytest.raises(RoutingGraphError, match="sinks") as info:
        run(inputs, schedule, 30.0, cadence=30.0)
    assert not isinstance(info.value, SedimentEventError) and len(calls) == 3
    compare(state0, saved)
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
        run(inputs, schedule, 30.0, cadence=30.0)
    compare(state0, saved)
    monkeypatch.undo()
    # invalid inputs are refused before any step
    bad_ctx = replace(inputs["ctx"], depth_update_rule="constant_free_surface")
    with pytest.raises(SedimentEventError, match="constant_depth"):
        evolve_sediment_event(state0, bad_ctx, inputs["column"], inputs["field"], schedule, inputs["vegetation"],
                              inputs["sediment"], 10.0, SedimentEventControl(), report_every_s=10.0)
    with pytest.raises(SedimentEventError, match="rainfall field"):
        evolve_sediment_event(state0, inputs["ctx"], inputs["column"], rainfall_field(3, 3), schedule,
                              inputs["vegetation"], inputs["sediment"], 10.0, SedimentEventControl(), report_every_s=10.0)
    with pytest.raises(SedimentEventError, match="sediment_courant_max"):
        SedimentEventControl(sediment_courant_max=1.5).validated()
    with pytest.raises(SedimentEventError, match="max_transport_substeps"):
        SedimentEventControl(max_transport_substeps=0).validated()
    assert SedimentEventControl().storm is not SedimentEventControl().storm  # per-instance default
    compare(state0, saved)


# --- forcing windows and the plan ------------------------------------------------------------------------------------
def test_rainfall_integrates_exactly_over_full_and_partial_windows():
    inputs = valley_case()
    schedule = constant_rainfall(0.0, 30.0, 72.0)
    n = inputs["state0"].graph.n_active
    full = run(inputs, schedule, 45.0, cadence=20.0)
    partial = run(inputs, schedule, 20.0, cadence=20.0)
    assert float(full.cumulative_rain_m.sum()) == pytest.approx(schedule.total_depth_m() * n, rel=1e-13)
    assert float(partial.cumulative_rain_m.sum()) == pytest.approx(schedule.depth_m(0.0, 20.0) * n, rel=1e-13)
    assert full.boundaries.tolist() == [20.0, 30.0, 40.0, 45.0] and partial.boundaries.tolist() == [20.0]
    assert full.state.t_s == 45.0 and partial.state.t_s == 20.0 and full.min_accepted_dt_s == 1.0


# --- constant-v/L continuous injection: discrete recursion exact, first order to the analytic solution -----------------
def constant_law(*, rate_kg_s, cell, cls, v, L):
    """Stub for `sediment_physics_step`: a constant pickup rate into one
    cell / class, uniform sediment velocity `v` and travel distance `L`,
    no settling anywhere (hydraulically dry cells included: an artificial
    law for analytic injection / CFL tests, hence `check_dry_settling=False`
    in their closure checks), no memory."""

    def stub(params, grid, depth, velocity, rain, veg, active_mass, prev_v, dt):
        ny, nx = grid.shape
        nc = params.n_classes
        zeros2, zeros3 = np.zeros((ny, nx)), np.zeros((ny, nx, nc))
        falses = np.zeros((ny, nx, nc), dtype=bool)
        requested = zeros3.copy()
        requested[cell + (cls,)] = rate_kg_s * dt
        vel = np.where(np.asarray(grid.active)[..., None], float(v), 0.0) * np.ones((ny, nx, nc))
        return SedimentPhysicsStep(
            dt_s=dt, requested_pickup_kg=requested, raindrop_pickup_kg=requested, flow_pickup_kg=zeros3,
            sediment_velocity_m_s=vel, deposition_rate_per_m=np.full((ny, nx, nc), 1.0 / L), law_applies=~falses,
            settle_mask=falses, regime=np.full((ny, nx, nc), REGIME_CODES["diffuse"], dtype=np.int8),
            d50_m=zeros2, shear_velocity_m_s=zeros2, reynolds_number=zeros2, stream_power_w_m2=zeros2,
            rain_energy_j_m2_mm=zeros2, rain_energy_flux_j_m2_s=zeros2, pickup_probability=zeros3,
            legacy_cap_applied=falses, regime_counts={name: np.int64(0) for name in REGIME_CODES},
        )

    return stub


def test_continuous_injection_matches_the_discrete_recursion_and_converges_to_the_analytic_pool(monkeypatch):
    # 100 cells: with one substep per step the pool can advance at most one cell per step, so
    # after N <= 80 steps nothing has reached the outlet and the export is EXACTLY zero.
    n, R, v, L, T = 100, 1e-3, 0.1, 1.0, 20.0
    k = v / L
    bed, ctx = make_bed(chain_elevation(n, dz=0.0015))
    inputs = setup(bed, ctx)
    monkeypatch.setattr(sediment_event, "sediment_physics_step", constant_law(rate_kg_s=R, cell=(n - 1, 0), cls=1, v=v, L=L))
    schedule = constant_rainfall(0.0, T, 0.0)
    analytic = R / k * (1.0 - math.exp(-k * T))
    errors = []
    for dt in (1.0, 0.5, 0.25):
        control = SedimentEventControl(storm=StormControl(max_dt_s=dt))
        r = run(inputs, schedule, T, control=control, cadence=T)
        _closure, bc = assert_closed(r, inputs["state0"], check_dry_settling=False)
        N = round(T / dt)
        assert r.n_accepted_steps == N and r.n_rejected_attempts == 0 and N < n
        s = math.exp(-k * dt)
        discrete = R * dt * s * (1.0 - s ** N) / (1.0 - s)  # pickup at the step start, exact decay over dt
        mobile = float(r.state.bed.water.mobile_mass_by_cell_class_kg[..., 1].sum())
        assert mobile == pytest.approx(discrete, rel=1e-11)
        assert bc["actual_pickup"][1] == pytest.approx(N * R * dt, rel=1e-12)
        assert bc["export_actual"].sum() == 0.0 and bc["export_requested"].sum() == 0.0
        # Source index is n-1; after N crossings index n-1-N is reachable.
        assert not np.any(r.state.bed.water.mobile_mass_by_cell_class_kg[: n - 1 - N])
        assert bc["deposition_actual"][1] == pytest.approx(N * R * dt - mobile, rel=1e-10)
        assert not np.any(r.state.bed.water.mobile_mass_by_cell_class_kg[..., [0, 2, 3, 4, 5]])
        errors.append(abs(mobile - analytic))
        assert float(r.max_sediment_courant) == pytest.approx(v * dt / 0.5) and r.max_transport_substeps_used == 1
    errors = np.array(errors)
    assert np.all(np.diff(errors) < 0.0) and np.all(errors[:-1] / errors[1:] > 1.8)  # first order in dt
    assert errors[0] / analytic == pytest.approx(0.05, abs=0.01)


def test_sediment_courant_uses_substeps_then_halves_dt_and_guards_refuse_without_mutation(monkeypatch):
    n, v = 12, 2.0  # a = v dt / dx = 4 at dt = 1 s, dx = 0.5 m
    bed, ctx = make_bed(chain_elevation(n))
    inputs = setup(bed, ctx)
    state0 = inputs["state0"]
    saved = snapshot(state0)
    monkeypatch.setattr(sediment_event, "sediment_physics_step", constant_law(rate_kg_s=1e-4, cell=(n - 1, 0), cls=2, v=v, L=2.0))
    schedule = constant_rainfall(0.0, 4.0, 0.0)
    substeps = run(inputs, schedule, 4.0, control=SedimentEventControl(), cadence=4.0)
    assert substeps.n_rejected_attempts == 0 and substeps.max_transport_substeps_used == 4
    assert substeps.n_accepted_steps == 4 and float(substeps.max_sediment_courant) == pytest.approx(1.0)
    assert_closed(substeps, state0, check_dry_settling=False)
    halved = run(inputs, schedule, 4.0, control=SedimentEventControl(max_transport_substeps=2), cadence=4.0)
    assert halved.n_rejected_attempts > 0 and halved.rejections[0]["kind"] == "TransportStepRejected"
    assert halved.state.t_s == 4.0 and halved.min_accepted_dt_s == 0.5 and halved.max_transport_substeps_used == 2
    assert_closed(halved, state0, check_dry_settling=False)
    with pytest.raises(SedimentEventError, match="max_retries") as info:
        run(inputs, schedule, 4.0, control=SedimentEventControl(max_transport_substeps=1,
                                                                storm=StormControl(max_retries=1)), cadence=4.0)
    assert isinstance(info.value.__cause__, TransportStepRejected)
    with pytest.raises(SedimentEventError, match="retry floor"):
        run(inputs, schedule, 4.0, control=SedimentEventControl(max_transport_substeps=1,
                                                                storm=StormControl(min_dt_s=0.5)), cadence=4.0)
    compare(state0, saved)


def test_hydraulic_rejection_halves_the_whole_step_from_the_same_state():
    bed, ctx = make_bed(chain_elevation(4), depth_m=0.05)
    inputs = setup(bed, ctx, ff=1.0, ksat=0.0)
    state0 = inputs["state0"]
    saved = snapshot(state0)
    control = SedimentEventControl(storm=StormControl(max_dt_s=8.0, min_dt_s=1.0 / 64.0, max_retries=10))
    r = run(inputs, constant_rainfall(0.0, 16.0, 0.0), 16.0, control=control, cadence=16.0)
    assert r.n_rejected_attempts > 0 and r.rejections[0]["kind"] == "RoutingStepRejected"
    assert r.state.t_s == 16.0 and r.min_accepted_dt_s <= 1.0
    assert_closed(r, state0)
    compare(state0, saved)


# --- true peaks and cadence independence -------------------------------------------------------------------------------
def test_true_peaks_and_final_state_do_not_depend_on_the_reporting_cadence():
    inputs = valley_case()
    coarse = run(inputs, constant_rainfall(0.0, 30.0, 72.0), 60.0, cadence=60.0)
    fine = run(inputs, constant_rainfall(0.0, 30.0, 72.0), 60.0, cadence=5.0)
    assert coarse.n_accepted_steps == fine.n_accepted_steps == 60
    for name in ("depth_m", "soil_water_m", "discharge_m2_s"):
        np.testing.assert_array_equal(getattr(coarse.state.storm, name), getattr(fine.state.storm, name))
    np.testing.assert_array_equal(coarse.state.bed.active_layer.mass_kg, fine.state.bed.active_layer.mass_kg)
    np.testing.assert_array_equal(coarse.state.bed.water.mobile_mass_by_cell_class_kg,
                                  fine.state.bed.water.mobile_mass_by_cell_class_kg)
    np.testing.assert_array_equal(coarse.last_velocity_m_s, fine.last_velocity_m_s)
    assert float(coarse.peak_mobile_kg) == float(fine.peak_mobile_kg) > 0.0
    assert float(coarse.time_of_peak_mobile_s) == float(fine.time_of_peak_mobile_s)
    assert float(coarse.peak_export_rate_kg_s) == float(fine.peak_export_rate_kg_s)
    assert float(coarse.peak_outlet_discharge_m3_s) == float(fine.peak_outlet_discharge_m3_s)
    # the forcing end (30 s) is always a boundary: the coarse cadence has the rows 30 and 60
    assert coarse.boundaries.tolist() == [30.0, 60.0] and fine.boundaries.size == 12
    sed_c, sed_f = np.asarray(coarse.sediment_hydrograph), np.asarray(fine.sediment_hydrograph)
    assert sed_c.shape[0] == 2 and np.all(sed_c[:, SCOL["mobile_kg"]] <= float(coarse.peak_mobile_kg))
    assert np.all(sed_f[:, SCOL["mobile_kg"]] <= float(fine.peak_mobile_kg))
    np.testing.assert_array_equal(sed_c[0], sed_f[5])  # the 30 s row is the same accepted state
    np.testing.assert_array_equal(sed_c[1], sed_f[-1])
    np.testing.assert_array_equal(np.asarray(coarse.hydrograph)[1], np.asarray(fine.hydrograph)[-1])


# --- one attempt: the two-call transaction on a real bed ------------------------------------------------------------------
def test_one_coupled_attempt_moves_actual_pickup_in_the_same_step_and_publishes_nothing():
    inputs = valley_case()
    state0 = inputs["state0"]
    saved = snapshot(state0)
    # advance once so that the surface is wet, then inspect a single attempt
    r = run(inputs, constant_rainfall(0.0, 30.0, 72.0), 5.0, control=SedimentEventControl(commit=False, force_final_commit=False), cadence=5.0)
    state = r.state
    rate = inputs["field"].apply(72.0 / 3.6e6)
    attempt = sediment_coupled_step(state, inputs["ctx"], inputs["column"], rate, inputs["vegetation"],
                                    inputs["sediment"], 1.0, SedimentEventControl())
    pickup, deposit, tr = attempt.pickup, attempt.deposit, attempt.transport
    assert pickup.adapter_name == deposit.adapter_name == "maple_syrup/phase5"
    assert float(pickup.actual_removal_by_class_kg.sum()) > 0.0 and not np.any(pickup.deposited_mass_by_class_kg)
    # transport worked on the ACTUAL post-pickup pool and the deposit call saw T(M): the same pools,
    # summed by MAPLE's reduction and by the transport's pairwise sum (n_cells roundings, ~1e-16 relative)
    n_cells = state.graph.shape[0] * state.graph.shape[1]
    np.testing.assert_allclose(tr.mobile_before_by_class_kg, pickup.mobile_mass_after_by_class_kg, rtol=n_cells * EPS)
    np.testing.assert_allclose(deposit.mobile_mass_before_by_class_kg, tr.mobile_after_by_class_kg, rtol=n_cells * EPS)
    assert not np.any(deposit.actual_removal_by_class_kg)
    np.testing.assert_allclose(deposit.deposited_mass_by_class_kg, tr.deposition_request_by_class_kg, atol=1e-9)
    np.testing.assert_allclose(deposit.boundary_export_by_class_kg, tr.export_request_by_class_kg, rtol=1e-12)
    # the published-to-be water carries the routed depth and the post-deposit pool; the input state is untouched
    np.testing.assert_array_equal(attempt.bed.water.depth_m, attempt.storm.route.depth_m)
    assert attempt.bed.water.mobile_mass_by_cell_class_kg is deposit.new_water.mobile_mass_by_cell_class_kg
    assert state.bed.water.depth_m is not attempt.bed.water.depth_m
    compare(state0, saved)
    assert attempt.n_substeps == 1 and attempt.n_transport_rejections == 0
