"""Spatial state construction/owner binding, metadata/time/forcing validation before allocation, staged value validation, no mutation of any accepted input or retained map, clocks, no backend
fallback, legacy-object refusals. Author-written."""
from __future__ import annotations

import dataclasses
import hashlib
import tracemalloc

import numpy as np
import pytest
from conftest import BIG, AREA, common_column, make_spec, params, prepare, sat_depth, water_kind

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss


def build(L=3, N=10, mode="constant", bottom="open", seed=1, **kw):
    p = params(np.random.default_rng(seed), L, N, mode=mode)
    return p, prepare(make_spec(p, mode=mode, bottom=bottom, **kw))


def forcing(p, seed=2):
    L, N = p["res"].shape
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1e-3, size=(L, N)) * 0.3, rng.uniform(0, 2e-7, size=N)


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def maps_digest(col):
    return digest(*(col._maps[n] for n in sorted(col._maps)), col._mask)


def st(col, water, t=0.0):
    return sp.spatial_column_state(col, water, t, memory_budget_bytes=BIG)


def step(col, state, added, pet, dt=3600.0, **kw):
    return sp.spatial_step(col, state, dt, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG, **kw)


# ---- state construction -----------------------------------------------------------------------------------------------------------------
def test_state_owns_a_copy_is_read_only_and_bound_to_the_column():
    p, col = build()
    water = water_kind(np.random.default_rng(3), p, "random")
    s = st(col, water, 2.5)
    assert s.component_tag == "spatial_explicit_cpu_state" and s.column_sha256 == col.sha256 and not s.water_m.flags.writeable and not np.shares_memory(s.water_m, water)
    before = s.water_m.copy()
    water[...] = 0.0  # the caller reuses its array
    np.testing.assert_array_equal(s.water_m, before)
    assert s.summary()["shape"] == [3, 10] and s.summary()["t_s"] == 2.5
    with pytest.raises(ValueError):
        s.water_m[0, 0] = 1.0


@pytest.mark.parametrize("bad", [lambda w: w.astype(np.float32), lambda w: w[:-1], lambda w: np.asfortranarray(w), lambda w: w.tolist(), lambda w: np.ma.masked_array(w), lambda w: w[:, :-1],
                                 lambda w: w.reshape(-1)])
def test_state_metadata_refusals_precede_allocation(bad):
    p, col = build()
    water = water_kind(np.random.default_rng(3), p, "random")
    tracemalloc.start()
    with pytest.raises(sp.SpatialLayerError):
        st(col, bad(water))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 200_000


@pytest.mark.parametrize("t", [float("nan"), float("inf"), True, "1", None])
def test_state_time_refusals(t):
    p, col = build()
    with pytest.raises(ss.SubsurfaceError):
        sp.spatial_column_state(col, water_kind(np.random.default_rng(3), p, "random"), t, memory_budget_bytes=BIG)


def test_uniform_state_is_theta_times_the_cells_own_width_and_zero_is_allowed():
    p, col = build(L=3, N=7)
    theta = [0.04, 0.0, 0.1]
    s = sp.spatial_uniform_state(col, theta, 1.5, memory_budget_bytes=BIG)
    dz = np.diff(p["edges"], axis=0)
    np.testing.assert_array_equal(s.water_m, np.array(theta)[:, None] * dz)  # the same product as the common `theta * thickness`
    assert s.t_s == 1.5 and (s.water_m[1] == 0.0).all()


def test_uniform_state_equals_the_common_uniform_state_for_equal_columns():
    L, N = 3, 9
    edges = np.array([0.0, 0.1, 0.3, 0.6])
    lay = ss.soil_layers(edges, theta_residual=0.02, theta_field_capacity=0.2, theta_saturation=0.5, diffusivity_m2_s=[1e-7, 2e-7], theta_evaporation_stress=0.5)
    ccol = ss.prepare_column(lay, np.ones(N, bool), AREA)
    spec = sp.spatial_layer_spec(edges, theta_residual=0.02, theta_field_capacity=0.2, theta_saturation=0.5, theta_evaporation_stress=0.5, diffusivity_m2_s=np.array([1e-7, 2e-7]), n_cells=N)
    scol = prepare(spec)
    a = ss.uniform_state(ccol, [0.05, 0.12, 0.3], 0.0)
    b = sp.spatial_uniform_state(scol, [0.05, 0.12, 0.3], 0.0, memory_budget_bytes=BIG)
    np.testing.assert_array_equal(a.water_m.view(np.uint64), b.water_m.view(np.uint64))


@pytest.mark.parametrize("theta", [[0.04, 0.0, 0.9], [-1e-9, 0.1, 0.1], [0.1, 0.1], [float("nan"), 0.1, 0.1], [True, 0.1, 0.1]])
def test_uniform_state_refusals(theta):
    _, col = build()
    with pytest.raises(ss.SubsurfaceError):
        sp.spatial_uniform_state(col, theta, 0.0, memory_budget_bytes=BIG)


def test_uniform_state_refuses_a_theta_above_saturation_in_any_single_cell():
    p, _ = build(L=2, N=6)
    p["sat"][0, 4] = p["fc"][0, 4] + 0.01  # one cell with a small saturation theta
    col = prepare(make_spec(p))
    with pytest.raises(ss.SubsurfaceError, match=r"\[0\]"):
        sp.spatial_uniform_state(col, [float(p["sat"][0, 4]) + 0.02, 0.01], 0.0, memory_budget_bytes=BIG)


def test_moisture_and_storage_against_hand_formulas():
    p, _ = build(L=2, N=5)
    mask = np.array([True, True, False, True, False])
    col = prepare(make_spec(p), mask=mask)
    water = water_kind(np.random.default_rng(3), p, "random")
    s = st(col, water)
    dz = np.diff(p["edges"], axis=0)
    np.testing.assert_array_equal(sp.spatial_moisture(col, s, memory_budget_bytes=BIG), water / dz)
    want = float(water[:, mask].sum()) * AREA
    assert sp.spatial_storage_m3(col, s, memory_budget_bytes=BIG) == pytest.approx(want, rel=1e-13)


# ---- owner binding -------------------------------------------------------------------------------------------------------------------
def test_a_state_cannot_be_used_with_other_geometry_properties_or_mask():
    p, col = build()
    other_p = {k: v.copy() for k, v in p.items()}
    other_p["sat"] = other_p["sat"] + 0.001
    other = prepare(make_spec(other_p))
    masked = prepare(make_spec(p), mask=np.arange(10) % 2 == 0)
    water = water_kind(np.random.default_rng(3), p, "random")
    added, pet = forcing(p)
    s = st(col, water)
    for foreign in (other, masked):
        for call in (lambda c: step(c, s, added, pet), lambda c: sp.spatial_moisture(c, s, memory_budget_bytes=BIG), lambda c: sp.spatial_storage_m3(c, s, memory_budget_bytes=BIG),
                     lambda c: sp.spatial_step_with_recharge(c, s, 86400.0, recharge_depth_m=1e-3, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)):
            with pytest.raises(sp.SpatialLayerError, match="identity mismatch"):
                call(foreign)
    twin = prepare(make_spec(p))  # an identical parameter set/mask is the same identity: accepted
    assert twin.sha256 == col.sha256 and step(twin, s, added, pet).state.t_s == 3600.0
    forged = dataclasses.replace(s, column_sha256="0" * 64)
    with pytest.raises(sp.SpatialLayerError):
        step(col, forged, added, pet)


def test_legacy_objects_are_refused_by_the_new_functions_and_spatial_objects_by_the_old_ones():
    p, col = build()
    lay = ss.soil_layers(np.array([0.0, 0.1, 0.3, 0.6]), theta_residual=0.02, theta_field_capacity=0.2, theta_saturation=0.5, diffusivity_m2_s=[1e-7, 2e-7], theta_evaporation_stress=0.5)
    ccol = ss.prepare_column(lay, np.ones(10, bool), AREA)
    cstate = ss.uniform_state(ccol, [0.05, 0.05, 0.05])
    s = st(col, water_kind(np.random.default_rng(3), p, "random"))
    added, pet = forcing(p)
    with pytest.raises(sp.SpatialLayerError):
        sp.spatial_step(ccol, s, 1.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError):
        sp.spatial_step(col, cstate, 1.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError):
        sp.spatial_column_state(ccol, s.water_m, 0.0, memory_budget_bytes=BIG)
    with pytest.raises(ss.SubsurfaceError):  # the old API refuses the spatial STATE as well as the spatial column
        ss.moisture(ccol, s)
    with pytest.raises(ss.SubsurfaceError):
        ss.step(ccol, s, 1.0, added_water_m=added, potential_evaporation_m_s=pet)
    with pytest.raises(ss.SubsurfaceError):
        ss.storage_m3(col, cstate)


def test_numba_prepared_owner_evolves_and_a_cuda_tagged_owner_refuses_before_any_copy_and_never_falls_back():
    """S2-A.3 narrowed this A2 control: a numba-prepared owner now genuinely evolves (see tests/phase2_spatial_numba); the typed refusal without a fallback remains for the backend that is still
    unsupported (cuda, S2-A.4). A real cuda owner needs a GPU, so the guard path is exercised on a numba owner whose backend tag is set to "cuda" (a metadata-only substitute for the guard; no kernel is run)."""
    pytest.importorskip("numba")
    p = params(np.random.default_rng(1), 3, 10)
    col = prepare(make_spec(p), backend="numba")
    assert col.backend == "numba"
    water = water_kind(np.random.default_rng(3), p, "random")
    added, pet = forcing(p)
    ok = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
    assert sp.spatial_step(col, ok, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG).state.t_s == 3600.0
    object.__setattr__(col, "backend", "cuda")  # guard-path substitute only
    try:
        fake_state = sp.SpatialSoilState(water, 0.0, col.sha256, 3, 10)
        try:
            sp._require_cuda()  # warm the one-off CuPy import probe so that the traced peak measures the refusals only
        except Exception:  # noqa: BLE001
            pass
        tracemalloc.start()
        for call in (lambda: sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG), lambda: sp.spatial_uniform_state(col, [0.1] * 3, 0.0, memory_budget_bytes=BIG),
                     lambda: sp.spatial_moisture(col, fake_state, memory_budget_bytes=BIG), lambda: sp.spatial_storage_m3(col, fake_state, memory_budget_bytes=BIG),
                     lambda: step(col, fake_state, added, pet),
                     lambda: sp.spatial_step_with_recharge(col, fake_state, 86400.0, recharge_depth_m=1e-3, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)):
            with pytest.raises((sp.SpatialLayerBackendUnavailableError, sp.SpatialLayerError), match="CuPy|cuda|namespace"):  # S2-A.4.2: a typed metadata/setup refusal (no CuPy on this host, or a NumPy state/array for a cuda-tagged owner); never hardware evidence
                call()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 200_000
    finally:
        object.__setattr__(col, "backend", "numba")


# ---- step validation ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,mutate", [
    ("added_f32", lambda a, p: (a.astype(np.float32), p)), ("added_shape", lambda a, p: (a[:-1], p)), ("added_list", lambda a, p: (a.tolist(), p)),
    ("added_fortran", lambda a, p: (np.asfortranarray(a), p)), ("pet_f32", lambda a, p: (a, p.astype(np.float32))), ("pet_shape", lambda a, p: (a, p[:-1])),
    ("pet_scalar", lambda a, p: (a, 1e-7)), ("added_masked", lambda a, p: (np.ma.masked_array(a), p))])
def test_forcing_metadata_refusals_precede_allocation(name, mutate):
    p, col = build()
    s = st(col, water_kind(np.random.default_rng(3), p, "random"))
    added, pet = mutate(*forcing(p))
    tracemalloc.start()
    with pytest.raises(sp.SpatialLayerError):
        step(col, s, added, pet)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 200_000


@pytest.mark.parametrize("dt", [-1.0, float("nan"), float("inf"), True, "1", None, -0.0 - 1e-300])
def test_dt_refusals(dt):
    p, col = build()
    s = st(col, water_kind(np.random.default_rng(3), p, "random"))
    added, pet = forcing(p)
    with pytest.raises(ss.SubsurfaceError):
        step(col, s, added, pet, dt=dt)


def test_candidate_time_must_be_finite_and_the_pulse_must_advance_the_clock():
    p, col = build()
    water = water_kind(np.random.default_rng(3), p, "random")
    added, pet = forcing(p)
    big = st(col, water, 1.7e308)
    with pytest.raises(ss.SubsurfaceError, match="not finite"):
        step(col, big, added, pet, dt=1.7e308)
    huge = st(col, water, 1e30)
    with pytest.raises(ss.SubsurfaceError, match="does not advance"):
        sp.spatial_step_with_recharge(col, huge, 1.0, recharge_depth_m=1e-3, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    s = st(col, water)
    for bad in (dict(dt=0.0), dict(dt=-1.0)):
        with pytest.raises(ss.SubsurfaceError):
            sp.spatial_step_with_recharge(col, s, bad["dt"], recharge_depth_m=1e-3, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    for r in (-1e-9, float("nan"), float("inf"), True, "1"):
        with pytest.raises(ss.SubsurfaceError):
            sp.spatial_step_with_recharge(col, s, 3600.0, recharge_depth_m=r, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)


@pytest.mark.parametrize("name,edit,bit", [
    ("nan_water", lambda w, a, e: w.__setitem__((1, 3), np.nan), 0), ("negative_water", lambda w, a, e: w.__setitem__((0, 2), -1e-3), 1),
    ("water_above_sat", lambda w, a, e: w.__setitem__((2, 4), 9.0), 2), ("nan_added", lambda w, a, e: a.__setitem__((0, 1), np.nan), 3),
    ("negative_added", lambda w, a, e: a.__setitem__((1, 1), -1.0), 4), ("nan_pet", lambda w, a, e: e.__setitem__(5, np.nan), 6), ("negative_pet", lambda w, a, e: e.__setitem__(5, -1.0), 7)])
def test_staged_value_validation_raises_the_existing_typed_error_with_its_bit(name, edit, bit):
    p, col = build()
    water = water_kind(np.random.default_rng(3), p, "random")
    added, pet = forcing(p)
    edit(water, added, pet)
    s = st(col, water)
    with pytest.raises(ss.SubsurfaceStepError) as info:
        step(col, s, added, pet)
    assert info.value.bit == bit


def test_inactive_forcing_and_invalid_shares_use_their_own_bits():
    p, _ = build()
    col = prepare(make_spec(p), mask=np.arange(10) != 4)
    water = water_kind(np.random.default_rng(3), p, "random")
    added, pet = forcing(p)
    added[:, 4], pet[4] = 0.0, 0.0  # valid forcing on the inactive cell is zero
    s = st(col, water)
    a2 = added.copy()
    a2[0, 4] = 1e-3
    with pytest.raises(ss.SubsurfaceStepError) as info:
        step(col, s, a2, pet)
    assert info.value.bit == 5
    for bad, bit in ((np.nan, 15), (1.5, 16)):
        share = np.full(10, 0.5)
        share[7] = bad
        with pytest.raises(ss.SubsurfaceStepError) as info:
            step(col, s, added, pet, evaporation_share=share)
        assert info.value.bit == bit


# ---- no mutation, clocks, aliasing --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("failing", [False, True])
def test_inputs_state_and_retained_maps_are_never_written_and_outputs_are_fresh(failing):
    p, col = build(mode="tietjen_donor_depth", bottom="closed")
    water = water_kind(np.random.default_rng(3), p, "above_fc")
    added, pet = forcing(p)
    share = np.full(10, 0.4)
    if failing:
        added = added.copy()
        added[0, 0] = np.nan
    s = st(col, water, 7.0)
    snap = (digest(s.water_m), digest(added), digest(pet), digest(share), maps_digest(col), col.sha256, s.t_s)
    if failing:
        with pytest.raises(ss.SubsurfaceStepError):
            step(col, s, added, pet, evaporation_share=share)
    else:
        out = step(col, s, added, pet, evaporation_share=share)
        for arr in (out.state.water_m, out.surface_return_m, out.drainage_m, out.evaporation_m, out.evaporation_unmet_m, out.percolation_m, out.diffusion_m):
            assert not any(np.shares_memory(arr, x) for x in (s.water_m, added, pet, share, *col._maps.values()))
        assert not out.state.water_m.flags.writeable
    assert snap == (digest(s.water_m), digest(added), digest(pet), digest(share), maps_digest(col), col.sha256, s.t_s)


def test_clock_advances_by_exactly_dt_including_zero_time():
    p, col = build()
    water = water_kind(np.random.default_rng(3), p, "random")
    added, pet = forcing(p)
    s = st(col, water, 5.0)
    for dt in (1e-6, 3600.0):
        out = step(col, s, added, pet, dt=dt)
        assert out.state.t_s == 5.0 + dt and out.dt_s == dt
    zero = step(col, s, np.zeros_like(added), pet, dt=0.0)  # zero time admits no added water (the accepted rule, flag 9)
    assert zero.state.t_s == 5.0
    with pytest.raises(ss.SubsurfaceStepError) as info:
        step(col, s, added, pet, dt=0.0)
    assert info.value.bit == 9
    np.testing.assert_array_equal(zero.state.water_m, s.water_m)  # a zero-time step changes no water (and counts no flux)
    assert not zero.drainage_m.any() and zero.budget.storage_after_m3 == zero.budget.storage_before_m3


def test_chained_steps_equal_the_common_solver_chained_per_column():
    L, N = 3, 8
    p, col = build(L=L, N=N, bottom="closed")
    water = water_kind(np.random.default_rng(3), p, "random")
    added, pet = forcing(p)
    s = st(col, water)
    cols = [common_column(p, c, bottom="closed") for c in range(N)]
    cs = [ss.column_state(cc, water[:, c : c + 1].copy(), 0.0) for c, cc in enumerate(cols)]
    for _ in range(4):
        s = step(col, s, added, pet, dt=1800.0).state
        cs = [ss.step(cc, x, 1800.0, added_water_m=added[:, c : c + 1].copy(), potential_evaporation_m_s=pet[c : c + 1].copy()).state for c, (cc, x) in enumerate(zip(cols, cs))]
    for c in range(N):
        np.testing.assert_array_equal(s.water_m[:, c], cs[c].water_m[:, 0])
    assert s.t_s == cs[0].t_s == 7200.0


def test_inactive_cells_keep_state_bit_for_bit_and_get_zero_fluxes():
    p, _ = build()
    mask = np.arange(10) % 3 != 0
    col = prepare(make_spec(p), mask=mask)
    water = water_kind(np.random.default_rng(3), p, "above_fc")
    added, pet = forcing(p)
    added[:, ~mask] = 0.0
    pet[~mask] = 0.0
    out = step(col, st(col, water), added, pet)
    np.testing.assert_array_equal(out.state.water_m[:, ~mask].view(np.uint64), water[:, ~mask].view(np.uint64))
    for f in (out.surface_return_m, out.drainage_m, out.evaporation_m, out.evaporation_unmet_m):
        assert not f[~mask].any()
    assert not out.percolation_m[:, ~mask].any() and not out.diffusion_m[:, ~mask].any()
