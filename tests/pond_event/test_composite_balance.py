"""C3: composite error bound of the per-step mobile balance. NOT RUN by the author (file-only tools).

The pure tests use only an independently validated `SplashResult` (the analytic operator, all faces closed, extreme finite
distances) and zero other operators: no bed is fabricated. The end-to-end test builds a long thin ACTUAL native bed."""
from dataclasses import replace

import numpy as np
import pytest
from pond_helpers import NC, backend_namespace, host

from maple_syrup.pond_event import composite_mobile_balance
from maple_syrup.pond_transport import operation_count
from maple_syrup.sediment_event import PHASE_RECONCILIATION_OPERATIONS
from maple_syrup.splash import splash_grid, splash_parameters, splash_redistribute

N = 4096
PICKUP = 1.0e-4


@pytest.fixture(params=["numpy", "cupy"])
def xp(request):
    return backend_namespace(request.param).xp


def component(xp, n=N):
    """Closed long row, S = 0.04 east, extreme finite splash distance [1e-8] * 6: the component closes by itself."""
    grid = splash_grid(np.full((1, n), 0.04), np.full((1, n), 2, np.int8), np.ones((1, n), bool), 0.5, xp=xp)
    params = splash_parameters([1.0e-8] * NC, xp=xp)
    pickup = xp.full((1, n, NC), PICKUP)
    return pickup, splash_redistribute(params, grid, pickup)


def stage(res):
    return {"x_net": res.x_face_net_kg, "y_net": res.y_face_net_kg, "x_gross": res.x_face_gross_kg,
            "y_gross": res.y_face_gross_kg, "tolerance": res.cell_balance_tolerance_kg}


def audit(xp, pickup, res, *, dep=None, mobile_new=None):
    zeros = xp.zeros_like(pickup)
    return composite_mobile_balance(
        xp, mobile_old=zeros, mobile_new=zeros if mobile_new is None else mobile_new, pick_wet=zeros,
        pick_splash=pickup, dep_wet=zeros, dep_splash=res.deposition_kg if dep is None else dep, wet=None,
        splash=stage(res))


def old_minimal(xp, pickup, res):
    """The R3 guard: scale from the endpoints / net divergence only, n_ops = 160."""
    from maple.surface.voxels._numerics import summation_error_bound_kg

    div = (res.x_face_net_kg[:, 1:] - res.x_face_net_kg[:, :-1]) + (res.y_face_net_kg[1:] - res.y_face_net_kg[:-1])
    residual = -(pickup - div - res.deposition_kg)
    scale = xp.maximum(xp.maximum(pickup, res.deposition_kg), xp.abs(div))
    n_ops = operation_count(1, 1) + 2 * PHASE_RECONCILIATION_OPERATIONS
    return xp.abs(residual) - summation_error_bound_kg(n_ops, scale)


def test_component_closes_old_minimal_guard_rejects_it_and_composite_accepts_it(xp):
    pickup, res = component(xp)
    own = host(xp.abs(res.cell_balance_residual_kg) - res.cell_balance_tolerance_kg)
    assert (own <= 0.0).all()  # the independent component closes in every cell/class on its own
    assert (host(old_minimal(xp, pickup, res)) > 0.0).any()  # the old operand scale false-rejects (root: 984 at n = 4096)
    out = audit(xp, pickup, res)
    assert (host(out["excess"]) <= 0.0).all()
    assert float(host(out["components"]["splash"]).max()) > 0.0  # the component bound is carried into the composite
    assert float(host(out["components"]["faces"]).max()) > 0.0  # individual face operands set the formation scale


def test_materially_unbalanced_cases_are_refused(xp):
    pickup, res = component(xp, 256)
    dep = host(res.deposition_kg).copy()
    dep[0, 5, 2] += 1.0e-6
    assert (host(audit(xp, pickup, res, dep=xp.asarray(dep))["excess"]) > 0.0).any()
    mobile = np.zeros((1, 256, NC))
    mobile[0, 7, 1] = 1.0e-6
    assert (host(audit(xp, pickup, res, mobile_new=xp.asarray(mobile))["excess"]) > 0.0).any()


def test_zero_input_has_zero_bound_and_a_tiny_imbalance_is_not_masked(xp):
    zeros = xp.zeros((1, 8, NC))
    zero_stage = {"x_net": xp.zeros((1, 9, NC)), "y_net": xp.zeros((2, 8, NC)), "x_gross": xp.zeros((1, 9, NC)),
                  "y_gross": xp.zeros((2, 8, NC)), "tolerance": zeros}
    kw = dict(mobile_old=zeros, pick_wet=zeros, pick_splash=zeros, dep_wet=zeros, dep_splash=zeros, wet=None,
              splash=zero_stage)
    clean = composite_mobile_balance(xp, mobile_new=zeros, **kw)
    assert float(host(clean["bound"]).max()) == 0.0 and float(host(clean["abs_residual"]).max()) == 0.0
    tiny = np.zeros((1, 8, NC))
    tiny[0, 3, 0] = 1.0e-300
    out = composite_mobile_balance(xp, mobile_new=xp.asarray(tiny), **kw)
    assert host(out["excess"])[0, 3, 0] > 0.0  # 1e-300 kg of imbalance with zero faces is refused


def build_long_row():
    """Actual native bed: ny = 1, nx = 4096, three 0.1 m voxels, closed pond, dry splash with spd 1e-8, one phase bin."""
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.geometry import validate_geometry
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.types.sediment_availability import sediment_availability_state_from_active_layer
    from maple.core.types.sediment_ledger import zeros_sediment_ledger_state
    from maple.core.types.voxel import zeros_voxel_column_state
    from maple.core.types.water import water_state_from_depth
    from maple.surface.active_layer import initialize_active_layer_from_voxels
    from maple.surface.topographic_commit import initial_committed_topography_state_from_physical_state
    from maple.surface.voxels import deposit_surface_mixture_batch

    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.infiltration import column_parameters
    from maple_syrup.pond_event import PondEvent, PondEventControl
    from maple_syrup.probe import DEFAULT_CLASS_FRACTIONS, build_minimal_state
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_bed import BedContext, BedState
    from maple_syrup.sediment_physics import plot1_sediment_parameters
    from pond_helpers import initial, make_topology

    small = build_minimal_state()  # native classes / resolution / voxel count from the SMALL probe
    geometry = replace(small.geometry, ny=1, nx=N)
    validate_geometry(geometry, small.grain_classes.ids())
    cols = np.arange(N, dtype=np.float64)[None, :]
    elevation = 0.12 + 0.08 * (1.0 - cols / (N - 1))  # < 0.3 m
    request = (elevation * geometry.bulk_density_kg_m3 * geometry.dx_m * geometry.dy_m)[:, :, None] \
        * np.asarray(DEFAULT_CLASS_FRACTIONS)[None, None, :]
    column = zeros_voxel_column_state(1, N, small.voxel_column.mass_kg.shape[2], NC)
    column, _ = deposit_surface_mixture_batch(column, request, geometry, small.mass_resolution_kg)
    column, layer, _ = initialize_active_layer_from_voxels(column, geometry, small.mass_resolution_kg)
    committed = initial_committed_topography_state_from_physical_state(column, layer, geometry, small.mass_resolution_kg)
    available = sediment_availability_state_from_active_layer(layer.mass_kg, np.ones_like(layer.mass_kg))
    bed = BedState(column, layer, water_state_from_depth(np.zeros((1, N)), NC), zeros_sediment_ledger_state(1, N, NC),
                   committed, available)
    ctx = BedContext(geometry, small.grain_classes, small.mass_resolution_kg, TopographicCommitSpec(),
                     AvalanchingSpec(enabled=False), np.ones_like(layer.mass_kg))
    topology = make_topology(bed, [], allow_closed=True)
    full = lambda v: np.full((1, N), float(v))  # noqa: E731
    cp = column_parameters(model="fixed_ksat", ksat_m_per_s=full(1e-3), suction_m=full(0.01), drainage_parameter=full(0.0),
                           theta_sat=full(0.4), soil_thickness_m=full(0.3), active_mask=np.ones((1, N), bool))
    event = PondEvent(ctx, cp, plot1_sediment_parameters(), constant_rainfall(0.0, 100.0, 60.0), topology,
                      control=PondEventControl(phase_bins=1, time=ExperimentalControl(max_dt_s=0.5)),
                      splash=splash_parameters([1.0e-8] * NC))
    return event, initial(event, bed)


@pytest.fixture(scope="module")
def long_row():
    event, state = build_long_row()
    return event, state, event.step(state, 60.0 / 3.6e6, 0.5)  # the composite guard raises if it rejects the step


def test_long_thin_native_row_step_closes_end_to_end(long_row):
    event, state, out = long_row
    assert out.record.splash_exchange and not np.any(host(out.state.hydraulic.depth_m))
    bal = out.record.balance
    assert bal["max_residual_kg"] <= bal["max_bound_kg"] and bal["component_bounds_kg"]["splash"] > 0.0
    audit = event.audit(out.state)
    s = audit["sediment"]
    assert s["closes"], s  # class closure AND the cumulative mobile identity with its derived bound
    resid, bound = np.abs(np.asarray(s["mobile_identity_residual_kg"])), np.asarray(s["mobile_identity_bound_kg"])
    assert (resid <= bound).all() and (bound > 0.0).all()
    # the wet term uses the EXECUTED substep count (1 on this row), not the configured cap
    assert out.record.n_substeps == 1 and out.state.counters.max_transport_substeps_used == 1
    assert out.state.counters.max_transport_substeps_used < event.control.max_transport_substeps
    terms = s["mobile_identity_bound_terms_kg"]
    assert max(terms["splash_component"]) > 0.0 and max(terms["cell_reductions"]) > 0.0


def test_material_mobile_or_ledger_imbalance_on_the_long_row_is_refused(long_row):
    from maple_syrup.sediment_bed import with_water

    event, _, out = long_row
    s = out.state
    ledger = replace(s.sediment, deposited_splash=s.sediment.deposited_splash + 1.0e-6)
    audit = event.audit(replace(s, sediment=ledger))["sediment"]
    assert not audit["closes"] and max(np.abs(audit["mobile_identity_residual_kg"])) > 5.0e-7
    mobile = host(s.bed.water.mobile_mass_by_cell_class_kg).copy()
    mobile[0, 10, 2] += 1.0e-6
    bed = with_water(s.bed, event.context, s.bed.water.depth_m, mobile)
    altered = event.audit(replace(s, bed=bed))["sediment"]
    assert not altered["closes"]  # both the mobile identity and the independent global class closure refuse it


def test_cell_reduction_bound_covers_a_raw_leading_axis_sum_against_exact_fsum(long_row):
    import math

    from maple_syrup.pond_event import cell_reduction_bound

    event, _, _ = long_row
    ny, nx = event.shape
    values = np.full((ny, nx, NC), 1.0e-16)  # class-last layout, sum over the two leading axes like the event's mobile sum
    values[0, 0, :] = 1.0  # one dominant term followed by many tiny ones: order-dependent rounding
    raw = values.sum(axis=(0, 1))  # whatever order this NumPy build uses
    exact = np.array([math.fsum(values[..., k].ravel().tolist()) for k in range(NC)])
    observed = np.abs(raw - exact)
    bound = cell_reduction_bound(ny * nx, raw)  # genuine endpoint of the raw reduction
    assert (bound >= observed).all()  # holds for a tree or a sequential implementation alike
    assert (bound > 0.0).all()
    # zero genuine inputs: zero bound
    assert not cell_reduction_bound(ny * nx, np.zeros(NC)).any()


def test_substep_counter_is_validated_and_follows_the_executed_count(cpu):
    from pond_helpers import wet_plane

    event, state = wet_plane(cpu)
    assert state.counters.max_transport_substeps_used == 0
    out = event.evolve(state, 0.3)
    c = out.state.counters
    assert out.n_accepted_steps > 0 and 1 <= c.max_transport_substeps_used <= event.control.max_transport_substeps
    step = event.step(out.state, 100.0 / 3.6e6, 0.05)
    assert step.state.counters.max_transport_substeps_used == max(c.max_transport_substeps_used, step.record.n_substeps)
    for bad in (-1, event.control.max_transport_substeps + 1, 0):
        forged = replace(out.state, counters=replace(c, max_transport_substeps_used=bad))
        with pytest.raises(Exception, match="max_transport_substeps_used"):
            event.quiet_metrics(forged)
    forged = replace(state, counters=replace(state.counters, max_transport_substeps_used=1))  # 1 with no accepted step
    with pytest.raises(Exception, match="max_transport_substeps_used"):
        event.quiet_metrics(forged)


def test_zero_genuine_inputs_give_zero_mobile_bound_and_a_tiny_imbalance_fails(cpu):
    from pond_helpers import east_outlets, initial as init, make_bed, make_event as mk, make_topology, plane

    bed, ctx = make_bed(plane(), depth=0.0)
    event = mk(bed, ctx, make_topology(bed, east_outlets(3, 4), drop=0.02))
    state = init(event, bed)
    clean = event.audit(state)["sediment"]
    assert clean["closes"] and not any(clean["mobile_identity_bound_kg"]) and not any(clean["mobile_identity_residual_kg"])
    tiny = np.zeros(NC)
    tiny[0] = 1.0e-300
    ledger = replace(state.sediment, deposited_wet=state.sediment.deposited_wet + tiny)
    bad = event.audit(replace(state, sediment=ledger))["sediment"]
    assert not bad["closes"]
    # exact comparison (pytest.approx's default absolute 1e-12 would hide a 1e-300 expectation)
    assert bad["mobile_identity_residual_kg"][0] == pytest.approx(1.0e-300, rel=1e-6, abs=0.0)
    assert bad["mobile_identity_bound_kg"][0] < 1.0e-300 and state.counters.max_transport_substeps_used == 0
