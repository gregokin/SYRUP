"""The compiled CPU (Numba) forms of the two experimental candidates against the NumPy reference, which stays the independent oracle.

Every public field of `HydraulicStep` (state, face momentum, column stage, velocity, face volumes, fluxes, stage face depths, scalars,
limiter activity) is compared recursively, with the declared bound rtol 2e-12 / atol 1e-14 for floats and exact equality for integers,
flags and counts; where the accepted column is the identity (ksat = 0) every ARRAY must also be bitwise equal, because the lateral
kernels use only `+ - * /`, `sqrt` and comparisons (a 1-ulp difference there is a finding, not a reason to widen a bound). Physics is
checked independently of the code (closed-form isolated cell, lake at rest, conservation, positivity), the error contract by class AND
message against the reference, and the safety contract (owned read-only statics, immutable sealed solver, refusal BEFORE any compiled
call, safe handling of strided/read-only arrays, pure failures). Skipped without Numba. Nothing here was run by its author (file-only
tools); Codex records results.
"""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest
from cand_cases import (
    DX,
    build,
    field,
    host,
    make_params,
    param_arrays,
    rain_rate,
    schedule,
)
from test_routing import make_graph, valley_full

pytest.importorskip("maple")
pytest.importorskip("numba")

from maple_syrup import experimental_numba as en
from maple_syrup import hydrology_numba as hn
from maple_syrup.experimental_hydrology import (
    CpuHydraulicSolver,
    ExperimentalHydrologyError,
    HydraulicControl,
    HydraulicState,
    HydraulicStep,
    HydraulicStepRejected,
    build_local_inertial_geometry,
    stage_face_diagnostics,
)
from maple_syrup.experimental_storm import ExperimentalControl, evolve_experimental
from maple_syrup.infiltration import InfiltrationError, column_step

METHODS = ("explicit", "local_inertial")
RTOL, ATOL = 2.0e-12, 1.0e-14
AREA = DX * DX


def twin(cs):
    """The compiled solver on the same graph/parameters/geometry/control as the reference solver of the case."""
    return en.NumbaHydraulicSolver(cs.method, cs.graph, cs.params, geometry=cs.geometry, control=cs.control)


def same(ref, new, path, *, exact):
    """Recursive comparison of every field. Floats: the declared bound; arrays bitwise when `exact`; integers/strings/None exact;
    the top-level scalar TYPES must match the reference (float vs np.float64 vs int are part of the contract)."""
    if dataclasses.is_dataclass(ref) and not isinstance(ref, type):
        assert type(ref) is type(new), path
        for f in dataclasses.fields(ref):
            if f.name == "implementation":
                assert new.implementation == "numba" and ref.implementation == "numpy", path
                continue
            same(getattr(ref, f.name), getattr(new, f.name), f"{path}.{f.name}", exact=exact)
    elif isinstance(ref, dict):
        assert isinstance(new, dict) and set(ref) == set(new), path
        for key in ref:
            same(ref[key], new[key], f"{path}[{key}]", exact=exact)
    elif isinstance(ref, np.ndarray):
        assert type(new) is np.ndarray and new.shape == ref.shape and new.dtype == ref.dtype, path
        if exact and "column" not in path:
            np.testing.assert_array_equal(new, ref, err_msg=path)
        else:
            np.testing.assert_allclose(new, ref, rtol=RTOL, atol=ATOL, err_msg=path)
    elif ref is None or isinstance(ref, (str, bool)):
        assert new == ref and type(new) is type(ref), path
    elif isinstance(ref, (int, np.integer)) and not isinstance(ref, bool):
        assert int(new) == int(ref), path
    else:
        np.testing.assert_allclose(float(new), float(ref), rtol=RTOL, atol=ATOL, err_msg=path)


def compare_step(ref, new, *, exact=False):
    assert isinstance(new, HydraulicStep) and new.implementation == "numba"
    same(ref, new, "step", exact=exact)
    for name in ("limited_cells", "limited_volume_m3", "cfl_kind", "method", "dt_s"):
        assert type(getattr(new, name)) is type(getattr(ref, name)), name  # contract: the reference's scalar types
    assert new.state.t_s == ref.state.t_s


# --- step differential: both column laws, off/donor, active/inactive, open faces, inactive cells ---------------------------
@pytest.mark.parametrize("limiter", ["off", "donor"])
@pytest.mark.parametrize("kind", ["valley:8x7", "random:9x6", "valley_masked", "plane:129"])
@pytest.mark.parametrize("method", METHODS)
def test_step_matches_the_numpy_reference_on_both_column_laws(method, kind, limiter):
    if method == "explicit" and limiter == "donor":
        pytest.skip("the donor limiter belongs to the local-inertial method")
    for model in ("fixed_ksat", "pavement_hawkins"):
        cs = build(kind, method, control=HydraulicControl(limiter=limiter), ksat=2e-6, soil_fraction=0.5, drainage=0.2,
                   model=model, seed=3)
        solver = twin(cs)
        state = cs.state
        for i, dt in enumerate([0.1, 0.05, 0.1, 0.02, 0.1, 0.1, 0.05, 0.1]):
            rain = rain_rate(cs, 150.0 if i < 5 else 0.0)
            ref = cs.solver.step(rain, state, dt)
            new = solver.step(rain, state, dt)
            compare_step(ref, new)
            state = ref.state


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["valley:8x7", "random:9x6", "valley_masked", "plane:129"])
def test_without_infiltration_every_array_is_bitwise_identical_over_a_trajectory(method, kind):
    """ksat = 0 makes the accepted column the identity. The trajectory is advanced with the COMPILED state (not the reference's) so
    a 1-ulp difference could not hide behind re-synchronisation: both are advanced independently and must stay bit-identical."""
    cs = build(kind, method, seed=4, control=HydraulicControl(limiter="donor") if method == "local_inertial" else None)
    solver = twin(cs)
    ref_state, new_state = cs.state, cs.state
    for i in range(15):
        rain = rain_rate(cs, 120.0 if i < 6 else 0.0)
        ref = cs.solver.step(rain, ref_state, 0.05)
        new = solver.step(rain, new_state, 0.05)
        compare_step(ref, new, exact=True)
        ref_state, new_state = ref.state, new.state


@pytest.mark.parametrize("limiter", ["off", "donor"])
@pytest.mark.parametrize("kind", ["valley:8x7", "random:9x6", "valley_masked"])
@pytest.mark.parametrize("method", METHODS)
def test_the_lateral_stage_is_bitwise_identical_when_fed_identical_column_outputs_and_state(method, kind, limiter):
    """Isolation of the compiled LATERAL stage from the compiled column: every step is given the SAME NumPy `ColumnStep` and the
    SAME state (the NumPy trajectory), with real infiltration (pavement_hawkins), and every public field of the two lateral
    results, scalars included, must be bit-identical. This is the narrow, measurable form of the bitwise claim; it says nothing
    about whole independent trajectories, where the compiled column's ulp-level differences can be amplified."""
    if method == "explicit" and limiter == "donor":
        pytest.skip("the donor limiter belongs to the local-inertial method")
    cs = build(kind, method, control=HydraulicControl(limiter=limiter), ksat=2e-6, soil_fraction=0.5, drainage=0.2,
               model="pavement_hawkins", seed=3)
    solver = twin(cs)
    lateral = "_explicit" if method == "explicit" else "_local_inertial"
    state = cs.state
    for i, dt in enumerate([0.1, 0.05, 0.1, 0.02, 0.1, 0.1, 0.05, 0.1, 0.1, 0.05]):
        rain = rain_rate(cs, 150.0 if i < 6 else 0.0)
        column = column_step(cs.params, state.depth_m, state.soil_water_m, rain, dt)  # the NumPy column, shared by both
        t_new = state.t_s + dt
        ref = getattr(cs.solver, lateral)(column, state, dt, t_new)
        new = getattr(solver, lateral)(column, state, dt, t_new)
        compare_step(ref, new, exact=True)
        for name in ("export_m3", "outlet_discharge_m3_s", "storage_change_m3", "budget_residual_m3", "max_cfl",
                     "max_cell_balance_residual_m", "limited_volume_m3"):
            assert getattr(new, name) == getattr(ref, name), name  # scalars bitwise too (the sums share their order)
        state = ref.state


@pytest.mark.parametrize("method", METHODS)
def test_a_closed_lake_with_a_tilted_surface_matches_the_reference_and_conserves_water(method):
    """Backwater/ponding: the local-inertial domain is closed (no open face) and the explicit one drains through its outlets."""
    cs = build("valley:6x5", method, closed=True, seed=5, depth=np.zeros((6, 5)))
    z = np.asarray(cs.z_full)[1:-1, 1:-1]
    surface = 0.3 + 0.01 * np.arange(5)[None, :] + z * 0.0  # a tilted free surface above the bed everywhere
    depth = np.where(cs.graph.active, np.maximum(surface - z, 0.0), 0.0)
    state = cs.solver.initial_state(depth, cs.state.soil_water_m)
    solver = twin(cs)
    ref_state, new_state = state, state
    for _ in range(20):
        ref = cs.solver.step(np.zeros(cs.shape), ref_state, 0.01)
        new = solver.step(np.zeros(cs.shape), new_state, 0.01)
        compare_step(ref, new, exact=True)
        ref_state, new_state = ref.state, new.state
    if method == "local_inertial":
        total = float(new_state.depth_m.sum()) * AREA
        assert total == pytest.approx(float(state.depth_m.sum()) * AREA, rel=1e-13)  # closed: nothing leaves
        assert float(new_state.depth_m.min()) >= 0.0


def test_water_in_inactive_cells_is_retained_and_exchanges_nothing():
    cs = build("valley_masked", "local_inertial", seed=6, depth=np.zeros((6, 5)))
    inactive = ~np.asarray(cs.graph.active)
    assert inactive.any()
    depth = np.where(inactive, 0.05, 0.001 * np.asarray(cs.graph.active))
    state = cs.solver.initial_state(depth, cs.state.soil_water_m)
    solver = twin(cs)
    new = solver.step(np.zeros(cs.shape), state, 0.02)
    ref = cs.solver.step(np.zeros(cs.shape), state, 0.02)
    compare_step(ref, new, exact=True)
    np.testing.assert_array_equal(new.state.depth_m[inactive], depth[inactive])  # untouched, not clipped, not exchanged
    assert not np.any(new.state.qx_m2_s[cs.geometry.fx_type == 0]) and not np.any(new.state.qy_m2_s[cs.geometry.fy_type == 0])


# --- independent physics (closed forms, not a mirror of the code) -------------------------------------------------------------
def test_explicit_isolated_source_cell_follows_the_closed_form():
    cs = build("valley:8x7", "explicit", seed=7, depth=np.zeros((8, 7)))
    solver = twin(cs)
    donors = cs.solver.donors
    k = np.asarray(cs.graph.conveyance).reshape(cs.shape)
    cell = next(i for i in range(donors.shape[1]) if cs.graph.active.reshape(-1)[i] and np.all(donors[:, i] < 0))
    depth = np.zeros(cs.shape)
    depth.reshape(-1)[cell] = 4e-4  # only this source cell is wet: no inflow, so h' = h - (dt/dx) k h^(3/2)
    state = cs.solver.initial_state(depth, cs.state.soil_water_m)
    dt = 0.05
    step = solver.step(np.zeros(cs.shape), state, dt)
    h = 4e-4
    expected = h - (dt / cs.graph.dx_m) * k.reshape(-1)[cell] * h ** 1.5
    assert step.state.depth_m.reshape(-1)[cell] == pytest.approx(expected, rel=1e-13)
    assert step.max_cfl == pytest.approx(1.5 * k.reshape(-1)[cell] * np.sqrt(h) * dt / cs.graph.dx_m, rel=1e-13)
    assert float(step.state.depth_m.min()) >= 0.0


def test_local_inertial_lake_at_rest_stays_at_rest_on_an_uneven_bed():
    cs = build("random:9x6", "local_inertial", closed=True, seed=8, depth=np.zeros((9, 6)))
    z = np.asarray(cs.z_full)[1:-1, 1:-1]
    depth = np.where(cs.graph.active, (z.max() + 0.05) - z, 0.0)  # a flat water surface 5 cm above the highest bed cell
    state = cs.solver.initial_state(depth, cs.state.soil_water_m)
    solver = twin(cs)
    for _ in range(25):
        step = solver.step(np.zeros(cs.shape), state, 0.002)  # small dt: the gravity-wave CFL stays well below its bound
        state = step.state
    assert float(np.abs(state.qx_m2_s).max()) < 1e-12 and float(np.abs(state.qy_m2_s).max()) < 1e-12  # well balanced
    assert float(np.abs(state.depth_m - depth).max()) < 1e-13
    assert float(state.depth_m.sum()) * AREA == pytest.approx(float(depth.sum()) * AREA, rel=1e-14)


def steep(limiter):
    z = valley_full(5, 5, sy=0.2, sx=0.4)
    graph = make_graph(z, ff=0.1)
    params = make_params(param_arrays(graph.shape), np.asarray(graph.active))
    geometry = build_local_inertial_geometry(z, np.asarray(graph.active), np.asarray(graph.friction_factor), graph.dx_m, [])
    control = HydraulicControl(limiter=limiter)
    return (CpuHydraulicSolver("local_inertial", graph, params, geometry=geometry, control=control),
            en.NumbaHydraulicSolver("local_inertial", graph, params, geometry=geometry, control=control))


@pytest.mark.parametrize("cell", [(2, 2), (2, 1)])
def test_the_donor_limiter_binds_identically_conserves_water_and_never_clips(cell):
    cpu, jit = steep("donor")
    depth = np.zeros((5, 5))
    depth[cell] = 2e-4
    state = cpu.initial_state(depth, np.zeros((5, 5)))
    ref = cpu.step(np.zeros((5, 5)), state, 1.0)
    new = jit.step(np.zeros((5, 5)), state, 1.0)
    assert int(ref.limited_cells) == int(new.limited_cells) == 1 and type(new.limited_cells) is int
    compare_step(ref, new, exact=True)
    assert float(new.state.depth_m.min()) >= 0.0
    assert float(new.state.depth_m.sum()) * AREA == pytest.approx(2e-4 * AREA, rel=1e-13)  # nothing removed
    assert new.limited_volume_m3 > 0.0


def test_the_limiter_off_rejects_with_the_reference_class_and_message_and_modifies_nothing():
    cpu, jit = steep("off")
    depth = np.zeros((5, 5))
    depth[2, 2] = 2e-4
    state = cpu.initial_state(depth, np.zeros((5, 5)))
    before = (state.depth_m.copy(), state.qx_m2_s.copy(), state.qy_m2_s.copy())
    with pytest.raises(HydraulicStepRejected) as ref_err:
        cpu.step(np.zeros((5, 5)), state, 1.0)
    with pytest.raises(HydraulicStepRejected) as new_err:
        jit.step(np.zeros((5, 5)), state, 1.0)
    assert type(new_err.value) is type(ref_err.value) and str(new_err.value) == str(ref_err.value)
    for a, b in zip(before, (state.depth_m, state.qx_m2_s, state.qy_m2_s), strict=True):
        np.testing.assert_array_equal(a, b)


# --- refusals: class AND message, error precedence -------------------------------------------------------------------------
def errors(cs, solver, state, dt, rain=None):
    out = []
    rain = np.zeros(cs.shape) if rain is None else rain
    for s in (cs.solver, solver):
        with pytest.raises(Exception) as info:
            s.step(rain, state, dt)
        out.append(info.value)
    return out


@pytest.mark.parametrize("method", METHODS)
def test_cfl_rejection_column_errors_dt_and_time_refusals_match_the_reference(method):
    cs = build("valley:6x5", method, seed=5, ksat=1e-6, soil_fraction=0.4)
    solver = twin(cs)
    st = cs.state
    ref, new = errors(cs, solver, st, 1e3)  # CFL, recoverable
    assert type(ref) is type(new) is HydraulicStepRejected and str(ref) == str(new)
    nan_soil = st.soil_water_m.copy()
    nan_soil[2, 2] = np.nan
    ref, new = errors(cs, solver, HydraulicState(0.0, st.depth_m, nan_soil, st.qx_m2_s, st.qy_m2_s), 1e3)  # column outranks CFL
    assert type(ref) is type(new) is InfiltrationError and str(ref) == str(new)
    for dt in (0.0, -1.0, float("nan"), float("inf"), True, "1", None):  # dt class and dt = 0, in the reference's precedence
        ref, new = errors(cs, solver, st, dt)
        assert type(ref) is type(new) and str(ref) == str(new), repr(dt)
    for t, dt in ((1.7e308, 1.7e308), (1e20, 1.0), (1.0, 5e-324), (-1.0, 0.1), (float("nan"), 0.1), (True, 0.1), ("1", 0.1)):
        bad = HydraulicState(t, st.depth_m, st.soil_water_m, st.qx_m2_s, st.qy_m2_s)
        ref, new = errors(cs, solver, bad, dt)
        assert type(ref) is type(new) and str(ref) == str(new), (t, dt)
    for cap in (0.0, -1.0, float("nan"), True, "1"):  # malformed continuation metadata
        bad = dataclasses.replace(st, next_dt_cap_s=cap)
        ref, new = errors(cs, solver, bad, 0.1)
        assert type(ref) is type(new) and str(ref) == str(new)


def test_rain_on_an_inactive_cell_and_negative_or_nonfinite_inputs_match_the_reference():
    cs = build("valley_masked", "explicit", seed=9, ksat=1e-6, soil_fraction=0.3)
    solver = twin(cs)
    inactive = ~np.asarray(cs.graph.active)
    rain = np.where(inactive, 1e-6, 0.0)
    ref, new = errors(cs, solver, cs.state, 0.05, rain=rain)
    assert type(ref) is type(new) is InfiltrationError and str(ref) == str(new)
    for bad_rain in (np.full(cs.shape, -1e-6), np.full(cs.shape, np.nan)):
        ref, new = errors(cs, solver, cs.state, 0.05, rain=bad_rain)
        assert type(ref) is type(new) is InfiltrationError and str(ref) == str(new)
    bad_depth = cs.state.depth_m.copy()
    bad_depth[1, 1] = np.nan
    ref, new = errors(cs, solver, HydraulicState(0.0, bad_depth, cs.state.soil_water_m), 0.05)
    assert type(ref) is type(new) is InfiltrationError and str(ref) == str(new)


def test_local_inertial_state_flux_errors_match_the_reference_class_and_message():
    cs = build("valley:6x5", "local_inertial", seed=5)
    solver = twin(cs)
    st, g = cs.state, cs.geometry
    closed = np.argwhere(g.fx_type == 0)[0]
    qx = st.qx_m2_s.copy()
    qx[tuple(closed)] = 0.1
    r, c, dr, _dc, _drop, _k = g.open_faces[0]
    idx = (r + (1 if dr > 0 else 0), c)
    qy = st.qy_m2_s.copy()
    qy[idx] = -g.fy_sign[idx] * 0.1
    nan_x = st.qx_m2_s.copy()
    nan_x[1, 1] = np.nan
    inf_y = st.qy_m2_s.copy()
    inf_y[2, 2] = np.inf
    for bad in (HydraulicState(0.0, st.depth_m, st.soil_water_m, qx, st.qy_m2_s),
                HydraulicState(0.0, st.depth_m, st.soil_water_m, st.qx_m2_s, qy),
                HydraulicState(0.0, st.depth_m, st.soil_water_m, nan_x, st.qy_m2_s),
                HydraulicState(0.0, st.depth_m, st.soil_water_m, st.qx_m2_s, inf_y)):
        ref, new = errors(cs, solver, bad, 0.05)
        assert type(ref) is type(new) is ExperimentalHydrologyError and str(ref) == str(new)
        with pytest.raises(ExperimentalHydrologyError):
            solver.validate_state(bad)
    solver.validate_state(st)


def test_explicit_momentum_and_structure_refusals_match_the_reference():
    cs = build("valley:6x5", "explicit", seed=5)
    solver = twin(cs)
    st = cs.state
    with pytest.raises(ExperimentalHydrologyError, match="no momentum") as ref_err:
        cs.solver.step(np.zeros(cs.shape), HydraulicState(0.0, st.depth_m, st.soil_water_m, np.zeros((6, 6)), np.zeros((7, 5))), 0.05)
    with pytest.raises(ExperimentalHydrologyError, match="no momentum") as new_err:
        solver.step(np.zeros(cs.shape), HydraulicState(0.0, st.depth_m, st.soil_water_m, np.zeros((6, 6)), np.zeros((7, 5))), 0.05)
    assert str(ref_err.value) == str(new_err.value)
    for bad in (HydraulicState(0.0, st.depth_m.astype(np.float32), st.soil_water_m),
                HydraulicState(0.0, np.zeros((2, 3)), st.soil_water_m),
                HydraulicState(0.0, st.depth_m.tolist(), st.soil_water_m),
                HydraulicState(0.0, np.ma.masked_array(st.depth_m), st.soil_water_m)):
        with pytest.raises((ExperimentalHydrologyError, InfiltrationError)) as new_err:
            solver.step(np.zeros(cs.shape), bad, 0.05)  # the compiled form never converts or accepts these
        try:
            cs.solver.step(np.zeros(cs.shape), bad, 0.05)
        except (ExperimentalHydrologyError, InfiltrationError) as ref_exc:  # where the reference refuses too, the class agrees
            assert type(ref_exc) is type(new_err.value)
    with pytest.raises(ExperimentalHydrologyError, match="HydraulicState"):
        solver.step(np.zeros(cs.shape), "not a state", 0.05)


def test_a_nonfinite_depth_produced_by_overflow_is_refused_like_the_reference():
    """An enormous depth overflows the cell update: the reference reports non-finite output (or its CFL rejection first); the
    compiled form must raise the same class and message."""
    cs = build("valley:6x5", "explicit", seed=10, depth=np.full((6, 5), 1e300))
    solver = twin(cs)
    ref, new = errors(cs, solver, cs.state, 1e-300)
    assert type(ref) is type(new) and str(ref) == str(new)


# --- array policy: strided, read-only, wrong types ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_strided_fortran_and_read_only_state_arrays_are_accepted_safely_and_unchanged(method):
    cs = build("valley:8x7", method, seed=11, ksat=1e-6, soil_fraction=0.4)
    solver = twin(cs)
    st = cs.state
    rain = rain_rate(cs, 90.0)
    ref = cs.solver.step(rain, st, 0.05)

    def fortran(a):
        return None if a is None else np.asfortranarray(a)

    def strided(a):
        if a is None:
            return None
        big = np.zeros((a.shape[0] * 2, a.shape[1] * 2))
        big[::2, ::2] = a
        return big[::2, ::2]  # a non-contiguous view with the same values

    def read_only(a):
        if a is None:
            return None
        out = np.array(a, copy=True)
        out.flags.writeable = False
        return out

    for convert in (fortran, strided, read_only):
        variant = HydraulicState(st.t_s, convert(st.depth_m), convert(st.soil_water_m), convert(st.qx_m2_s), convert(st.qy_m2_s))
        before = [None if a is None else np.array(a, copy=True) for a in (variant.depth_m, variant.soil_water_m,
                                                                           variant.qx_m2_s, variant.qy_m2_s)]
        new = solver.step(np.asfortranarray(rain) if convert is fortran else rain, variant, 0.05)
        compare_step(ref, new)
        after = (variant.depth_m, variant.soil_water_m, variant.qx_m2_s, variant.qy_m2_s)
        for b, a in zip(before, after, strict=True):
            if b is not None:
                np.testing.assert_array_equal(a, b)  # inputs never written
        assert new.state.depth_m.flags.c_contiguous and new.state.depth_m.flags.writeable  # fresh C-contiguous outputs


def test_wrong_types_are_refused_before_any_compiled_call(monkeypatch):
    cs = build("valley:6x5", "local_inertial", seed=12)
    solver = twin(cs)
    solver.step(np.zeros(cs.shape), cs.state, 0.05)  # compile first
    monkeypatch.setattr(en, "_kernels", lambda: pytest.fail("a lateral kernel ran on an input that had to be refused"))
    monkeypatch.setattr(hn, "_kernels", lambda: pytest.fail("a column kernel ran on an input that had to be refused"))
    st = cs.state
    for bad in (HydraulicState(0.0, st.depth_m, st.soil_water_m, st.qx_m2_s.astype(np.float32), st.qy_m2_s),
                HydraulicState(0.0, st.depth_m, st.soil_water_m, st.qx_m2_s[:, :-1], st.qy_m2_s),
                HydraulicState(0.0, st.depth_m, st.soil_water_m, np.ma.masked_array(st.qx_m2_s), st.qy_m2_s),
                HydraulicState(0.0, st.depth_m, st.soil_water_m, None, None),
                HydraulicState(0.0, np.ma.masked_array(st.depth_m), st.soil_water_m, st.qx_m2_s, st.qy_m2_s),
                HydraulicState(0.0, st.depth_m.astype(np.float32), st.soil_water_m, st.qx_m2_s, st.qy_m2_s)):
        with pytest.raises((ExperimentalHydrologyError, InfiltrationError)):
            solver.step(np.zeros(cs.shape), bad, 0.05)
    for bad_rain in (np.zeros((2, 2)), np.zeros(cs.shape, dtype=np.float32), [[0.0] * 5] * 6, np.ma.masked_array(np.zeros(cs.shape))):
        with pytest.raises(InfiltrationError):
            solver.step(bad_rain, st, 0.05)


# --- purity and ownership --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_outputs_are_fresh_and_inputs_and_earlier_results_never_change(method):
    cs = build("valley:6x5", method, seed=8, ksat=1e-6, soil_fraction=0.3)
    solver = twin(cs)
    st = cs.state
    rain = rain_rate(cs, 100.0)
    arrays = [st.depth_m, st.soil_water_m] + ([st.qx_m2_s, st.qy_m2_s] if method == "local_inertial" else []) + [rain]
    before = [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
    first = solver.step(rain, st, 0.05)
    snap = {k: v.copy() for k, v in (("depth", first.state.depth_m), ("vel", first.velocity_m_s))}
    solver.step(rain, first.state, 0.05)
    with pytest.raises(HydraulicStepRejected):
        solver.step(rain, first.state, 1e3)
    np.testing.assert_array_equal(first.state.depth_m, snap["depth"])
    np.testing.assert_array_equal(first.velocity_m_s, snap["vel"])
    assert before == [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
    outs = [first.state.depth_m, first.velocity_m_s, *first.face_volume_m3.values(), first.state.soil_water_m, *arrays]
    for i, a in enumerate(outs):
        for b in outs[i + 1:]:
            assert not np.shares_memory(a, b)
    again = solver.step(rain, st, 0.05)
    assert again.state.depth_m is not first.state.depth_m  # a new result every call
    np.testing.assert_array_equal(again.state.depth_m, first.state.depth_m)


@pytest.mark.parametrize("method", METHODS)
def test_the_static_arrays_are_owned_read_only_and_never_alias_the_callers(method):
    cs = build("valley:8x7", method, seed=13)
    solver = twin(cs)
    ctx = solver.context
    callers = [np.asarray(cs.graph.active), np.asarray(cs.graph.outlet), np.asarray(cs.graph.conveyance)]
    if cs.geometry is not None:
        callers += [getattr(cs.geometry, name) for name in cs.geometry.ARRAYS]
    for name, array in ctx.arrays.items():
        assert type(array) is np.ndarray and array.flags.c_contiguous and not array.flags.writeable, name
        assert array.base is None or array.flags.owndata, name
        for other in callers:
            assert not np.shares_memory(array, other), name
    hyd = solver.hydrology_context
    for name in ("active", "outlet", "conveyance", "column_static"):
        assert not getattr(hyd, name).flags.writeable
        for other in callers + [np.asarray(cs.params.ksat_m_per_s)]:
            assert not np.shares_memory(getattr(hyd, name), other), name
    assert ctx.static_bytes == sum(a.nbytes for a in ctx.arrays.values()) and ctx.n_cells == 56 and ctx.shape == (8, 7)
    # a later mutation of the caller's own copies cannot change the compiled path (it owns its data)
    ref_before = solver.step(np.zeros(cs.shape), cs.state, 0.05)
    k = np.asarray(cs.graph.conveyance)
    if k.flags.writeable:
        k[:] = 0.0
    again = solver.step(np.zeros(cs.shape), cs.state, 0.05)
    np.testing.assert_array_equal(again.state.depth_m, ref_before.state.depth_m)


# --- sealed context, immutability: refusal BEFORE any compiled call ---------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_forged_contexts_and_attributes_are_refused_before_any_compiled_call(monkeypatch, method):
    cs = build("valley:6x5", method, seed=9)
    other = build("valley:8x7", method, seed=9)
    solver, other_solver = twin(cs), twin(other)
    st = cs.state
    rain = np.zeros(cs.shape)
    solver.step(rain, st, 0.05)  # valid use first (compiles)
    monkeypatch.setattr(en, "_kernels", lambda: pytest.fail("a lateral kernel ran with forged metadata"))
    monkeypatch.setattr(hn, "_kernels", lambda: pytest.fail("a column kernel ran with forged metadata"))
    ctx, hyd = solver.context, solver.hydrology_context
    some = next(iter(ctx.arrays))
    shrunk = np.zeros(max(ctx.arrays[some].size - 1, 1), dtype=ctx.arrays[some].dtype)
    shrunk.flags.writeable = False
    swapped = dict(ctx.arrays)
    swapped[some] = shrunk
    writable = dict(ctx.arrays)
    writable_copy = np.array(ctx.arrays[some], copy=True)  # same values, another pointer, writable
    writable[some] = writable_copy
    forged = {
        ("solver", "_context"): other_solver.context,
        ("solver", "_hydrology"): other_solver.hydrology_context,
        ("solver", "_seal"): (method, (1, 30), ctx.dx_m, 0.5, "off", id(ctx), id(hyd), (1, 30), 30, ctx.dx_m, 0),
        ("solver", "shape"): (1, 30),
        ("solver", "dx_m"): ctx.dx_m * 2.0,
        ("solver", "method"): "explicit" if method == "local_inertial" else "local_inertial",
        ("solver", "control"): HydraulicControl(cfl_max=0.25),
        ("ctx", "arrays"): swapped,
        ("ctx2", "arrays"): writable,
        ("ctx", "n_cells"): ctx.n_cells * 4,
        ("hyd", "n_cells"): hyd.n_cells * 4,
        ("hyd", "shape"): (1, 30),
        ("hyd", "column_static"): np.zeros((hyd.n_cells, 6)),
        ("hyd", "active"): np.zeros(hyd.n_cells - 1, dtype=bool),
        ("hyd", "conveyance"): np.zeros(hyd.n_cells),
    }
    targets = {"solver": solver, "ctx": ctx, "ctx2": ctx, "hyd": hyd}
    for (owner, attribute), value in forged.items():
        target = targets[owner]
        original = getattr(target, attribute)
        object.__setattr__(target, attribute, value)
        try:
            with pytest.raises(ExperimentalHydrologyError):
                solver.step(rain, st, 0.05)
        finally:
            object.__setattr__(target, attribute, original)
    monkeypatch.undo()
    compare_step(cs.solver.step(rain, st, 0.05), solver.step(rain, st, 0.05))  # restored: still the sealed objects


def test_a_static_array_made_writable_or_resized_in_place_is_detected_and_the_solver_is_immutable():
    cs = build("valley:6x5", "local_inertial", seed=14)
    solver = twin(cs)
    solver.step(np.zeros(cs.shape), cs.state, 0.05)
    array = solver.context.arrays["z"]
    array.flags.writeable = True  # an owned copy may be re-enabled by a caller; that is outside the contract and detected
    try:
        with pytest.raises(ExperimentalHydrologyError, match="sealed"):
            solver.step(np.zeros(cs.shape), cs.state, 0.05)
    finally:
        array.flags.writeable = False
    solver.step(np.zeros(cs.shape), cs.state, 0.05)  # restored
    for name, value in (("shape", (1, 30)), ("method", "explicit"), ("control", HydraulicControl()), ("dx_m", 2.0)):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(solver, name, value)
    with pytest.raises(AttributeError, match="immutable"):
        solver.brand_new_attribute = 1


@pytest.mark.parametrize("method", METHODS)
def test_constructor_refusals_match_the_reference(method):
    cs = build("valley:6x5", method, seed=15)
    with pytest.raises(ExperimentalHydrologyError, match="method"):
        en.NumbaHydraulicSolver("diffusive", cs.graph, cs.params)
    with pytest.raises(ExperimentalHydrologyError, match="RoutingGraph"):
        en.NumbaHydraulicSolver(method, "graph", cs.params)
    if method == "explicit":
        with pytest.raises(ExperimentalHydrologyError, match="donor limiter"):
            en.NumbaHydraulicSolver(method, cs.graph, cs.params, control=HydraulicControl(limiter="donor"))
    else:
        with pytest.raises(ExperimentalHydrologyError, match="LocalInertialGeometry"):
            en.NumbaHydraulicSolver(method, cs.graph, cs.params)
    with pytest.raises(ExperimentalHydrologyError, match="cfl_max"):
        en.NumbaHydraulicSolver(method, cs.graph, cs.params, geometry=cs.geometry, control=HydraulicControl(cfl_max=0.9))


# --- describe / provenance / lazy compilation ------------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_describe_records_the_actual_implementation_compiler_and_context(method):
    cs = build("valley:6x5", method, seed=16)
    solver = twin(cs)
    info = solver.describe()
    ref_info = cs.solver.describe()
    assert info["implementation"] == "numba" and ref_info["implementation"] == "numpy" and info["method"] == method
    assert info["control"] == ref_info["control"] and info["cfl_kind"] == ref_info["cfl_kind"]
    assert info["kernels"]["numba_options"]["fastmath"] is False and info["kernels"]["versions"]["numba"]
    assert len(info["kernels"]["module_sha256"]) == 64 and info["kernels"]["column"].startswith("hydrology_numba")
    assert info["context"]["static_bytes"] > 0 and info["context"]["host_only"] is True
    assert info["hydrology_context"]["host_only"] is True and info["hydrology_context"]["n_cells"] == 30
    assert "Host-only" in info["transfer_scope"] and "NOT" in info["qualification"] and "array_policy" in info


def test_kernels_compile_lazily_on_the_first_step():
    cs = build("valley:6x5", "explicit", seed=17)
    en.reset_compiled()
    solver = twin(cs)  # construction compiles nothing of the lateral kernels
    assert en.kernel_provenance()["compiled_in_process"] is False
    solver.step(np.zeros(cs.shape), cs.state, 0.05)
    assert en.kernel_provenance()["compiled_in_process"] is True


# --- shared driver: whole storms, retries, snapshots, continuation -----------------------------------------------------------
def storm(solver, cs, **kw):
    sched = schedule([0.0, 10.0, 25.0, 40.0], [120.0, 0.0, 60.0])
    return evolve_experimental(solver, field(cs), sched, cs.state, 40.0, ExperimentalControl(max_dt_s=0.25),
                               report_every_s=15.0, snapshot_times_s=(12.5, 30.0), **kw)


@pytest.mark.parametrize("method", METHODS)
def test_a_whole_storm_through_the_shared_driver_matches_the_reference(method):
    cs = build("valley:8x7", method, ksat=2e-6, soil_fraction=0.4, drainage=0.3, seed=18)
    ref = storm(cs.solver, cs)
    new = storm(twin(cs), cs)
    assert new.implementation == "numba" and ref.implementation == "numpy" and new.method == method
    assert new.n_accepted_steps == ref.n_accepted_steps and new.n_rejected_attempts == ref.n_rejected_attempts
    assert [(r["t_s"], r["dt_tried_s"]) for r in new.rejections] == [(r["t_s"], r["dt_tried_s"]) for r in ref.rejections]
    assert new.next_dt_cap_s == ref.next_dt_cap_s
    np.testing.assert_array_equal(new.boundaries, ref.boundaries)
    for name in ("hydrograph", "cumulative_rain_m", "cumulative_intake_m", "cumulative_saturation_return_m",
                 "cumulative_drainage_m", "peak_depth_m", "peak_velocity_m_s", "last_velocity_m_s"):
        np.testing.assert_allclose(getattr(new, name), getattr(ref, name), rtol=RTOL, atol=ATOL, err_msg=name)
    for name in ("cumulative_export_m3", "peak_outlet_discharge_m3_s", "time_of_peak_outlet_s", "max_cfl",
                 "limited_volume_total_m3"):
        np.testing.assert_allclose(float(getattr(new, name)), float(getattr(ref, name)), rtol=RTOL, atol=ATOL, err_msg=name)
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        a, b = getattr(new.state, name), getattr(ref.state, name)
        assert (a is None) == (b is None)
        if a is not None:
            np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, err_msg=name)
    assert set(new.snapshots) == set(ref.snapshots) == {12.5, 30.0}
    for t in ref.snapshots:
        for key, value in ref.snapshots[t].items():
            np.testing.assert_allclose(new.snapshots[t][key], value, rtol=RTOL, atol=ATOL, err_msg=f"{t}:{key}")


CASES = [("explicit", 16.0, 0.05), ("explicit", 4.0, 0.2), ("explicit", 1.0, 1.0),
         ("local_inertial", 4.0, 0.02), ("local_inertial", 1.0, 0.1)]


@pytest.mark.parametrize(("method", "max_dt", "depth"), CASES)
def test_adaptive_continuation_is_exact_and_equals_the_reference_trajectory(method, max_dt, depth):
    """The root's forcing-edge scenario (non-binary edge, zero rain, cadence = max_dt, deep lake so steps really reject): the
    compiled run to a boundary plus a resumed run equals one continuous compiled run AND the NumPy reference, bitwise."""
    cs = build("valley:6x5", method, depth="dry", seed=3)
    solver = twin(cs)
    initial = np.where(cs.graph.active, depth, 0.0)
    ref0 = cs.solver.initial_state(initial, cs.state.soil_water_m)
    new0 = solver.initial_state(initial, cs.state.soil_water_m)
    end = 3.0 * max_dt
    sched = schedule([0.0, 0.451875 * max_dt, end], [0.0, 0.0])
    control = ExperimentalControl(max_dt_s=max_dt)

    def run(s, state, until):
        return evolve_experimental(s, field(cs), sched, state, until, control, report_every_s=max_dt)

    once, first = run(solver, new0, end), run(solver, new0, max_dt)
    assert once.n_rejected_attempts > 0, "the scenario must really exercise the adaptive cap"
    second = run(solver, first.state, end)
    ref_once = run(cs.solver, ref0, end)
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        a, b, c = (getattr(x.state, name) for x in (once, second, ref_once))
        assert (a is None) == (b is None) == (c is None)
        if a is not None:
            np.testing.assert_array_equal(a, b, err_msg=f"resumed {name}")
            np.testing.assert_array_equal(a, c, err_msg=f"reference {name}")
    assert first.n_accepted_steps + second.n_accepted_steps == once.n_accepted_steps == ref_once.n_accepted_steps
    assert first.n_rejected_attempts + second.n_rejected_attempts == once.n_rejected_attempts == ref_once.n_rejected_attempts
    assert second.state.next_dt_cap_s == once.state.next_dt_cap_s == ref_once.state.next_dt_cap_s
    stripped = dataclasses.replace(first.state, next_dt_cap_s=None)  # the old reset-at-checkpoint behaviour is still different
    if method == "explicit":
        naive = run(solver, stripped, end)
        assert (not np.array_equal(naive.state.depth_m, once.state.depth_m)
                or first.n_accepted_steps + naive.n_accepted_steps != once.n_accepted_steps)


def test_the_stage_face_diagnostics_work_on_a_compiled_step_and_match_the_reference():
    cs = build("valley:8x7", "local_inertial", seed=19, ksat=1e-6, soil_fraction=0.4)
    solver = twin(cs)
    state = cs.state
    for _ in range(5):
        ref = cs.solver.step(rain_rate(cs, 120.0), state, 0.1)
        new = solver.step(rain_rate(cs, 120.0), state, 0.1)
        state = ref.state
    d_ref, d_new = stage_face_diagnostics(ref), stage_face_diagnostics(new)
    for group in ("face_flow_depth_m", "face_velocity_m_s", "face_froude"):
        for key in d_ref[group]:
            np.testing.assert_allclose(d_new[group][key], d_ref[group][key], rtol=RTOL, atol=ATOL, err_msg=f"{group}.{key}")
    assert host(d_new["face_froude"]["x"]).shape == d_ref["face_froude"]["x"].shape
