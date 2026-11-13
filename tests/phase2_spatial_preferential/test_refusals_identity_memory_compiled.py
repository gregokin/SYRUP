"""Typed refusals, immutability, identity, admission/quotas, memory model, owner release and the genuine compiled proof of the spatial preferential operator. Author-written, CPU only, no GPU claim."""
from __future__ import annotations

import copy
import dataclasses
import gc
import pickle
import tracemalloc
import weakref

import numpy as np
import pytest
from conftest import BIG, full, hand_column, hand_op, hand_state, pi, sp, spf, ss, step


def ro(a):
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    return a


def legacy(backend, n=3):
    lay = ss.soil_layers([0.0, 0.5, 1.5], theta_residual=0.01, theta_field_capacity=0.25, theta_saturation=0.9, diffusivity_m2_s=0.0, theta_evaporation_stress=0.05)
    col = ss.prepare_column(lay, np.ones(n, dtype=bool), 1.0, backend=backend)
    op = pi.prepare_preferential(col, a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=np.inf, background_receiver_weights=[1.0, 0.0])
    return col, op, ss.column_state(col, np.full((2, n), 0.05), 0.0)


# ---- cross refusals / sealing -------------------------------------------------------------------------------------------------------------
def test_legacy_and_spatial_objects_refuse_each_other(backend):
    col = hand_column(2, 3, backend)
    op, state = hand_op(col), hand_state(col)
    lcol, lop, lstate = legacy(backend)
    H = full(col, 0.01)
    with pytest.raises(sp.SpatialLayerError):
        spf.prepare_spatial_preferential(lcol, a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=1.0, background_receiver_weights=[1.0, 0.0], memory_budget_bytes=BIG)
    for bad in (lop, object(), None):
        with pytest.raises(sp.SpatialLayerError):
            spf.spatial_preferential_step(bad, H, state, 60.0, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError):
        spf.spatial_preferential_step(op, H, lstate, 60.0, memory_budget_bytes=BIG)
    # the legacy entry points refuse the spatial objects (the old guards are NOT widened)
    with pytest.raises(ss.SubsurfaceError):
        pi.prepare_preferential(col, a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=1.0, background_receiver_weights=[1.0, 0.0])
    with pytest.raises(ss.SubsurfaceError):
        pi.preferential_step(op, H, lstate, 60.0)
    with pytest.raises(ss.SubsurfaceError):
        pi.preferential_step(lop, H, state, 60.0)
    for typed in (sp.SpatialLayerError, pi.PreferentialStepError):
        assert issubclass(typed, ss.SubsurfaceError)


def test_cuda_owner_without_a_usable_device_is_refused_before_any_copy_validation_or_reader(monkeypatch):
    """S2-B2.d: a CUDA owner is supported when CuPy and its device are usable; without them (simulated here, independent of the host) the typed refusal precedes every copy, validation and reader call."""
    def no_cuda():
        raise sp.SpatialLayerBackendUnavailableError("no usable CuPy/device (simulated)")

    monkeypatch.setattr(sp, "_require_cuda", no_cuda)
    col = hand_column(2, 2)
    with sp._constructing():
        fake = dataclasses.replace(col, backend="cuda", device_id=0)
    calls = []
    consumed = []
    with pytest.raises(sp.SpatialLayerBackendUnavailableError):
        spf.prepare_spatial_preferential(fake, a3=object(), max_fraction_per_hour=1.0, max_intake_rate_m_s=object(), background_receiver_weights=[1, 0], species_receiver_weights=[[1, 0]],
                                         species_contributions=iter(consumed), memory_budget_bytes=BIG, available_bytes_fn=lambda: calls.append(1) or BIG)
    assert calls == []


def test_direct_construction_copy_and_pickle_are_refused(backend):
    col = hand_column(2, 2, backend)
    op = hand_op(col)
    with pytest.raises(sp.SpatialLayerError):
        copy.copy(op)
    with pytest.raises(sp.SpatialLayerError):
        copy.deepcopy(op)
    with pytest.raises(sp.SpatialLayerError):
        pickle.dumps(op)
    with pytest.raises(dataclasses.FrozenInstanceError):
        op.n_cells = 5
    with pytest.raises(sp.SpatialLayerError):
        dataclasses.replace(op, backend=backend)


def test_the_maps_are_sealed_and_caller_arrays_are_copied_or_not_retained(backend):
    col = hand_column(2, 4, backend)
    mr = np.full(4, 2e-6)
    contribution = np.full(4, 1.5)
    op = hand_op(col, a3=0.4, maxrate=mr, background=[1.0, 0.0], species=[[0.0, 1.0]], contributions=[contribution])
    ref = weakref.ref(contribution)
    mr[:] = 7.0
    assert float(op._law[1][0]) == 2e-6
    for a in op._law:
        assert not a.flags.writeable
        with pytest.raises(ValueError):
            a.flags.writeable = True
        with pytest.raises(ValueError):
            a.reshape(-1)[0] = 1.0
    del contribution
    gc.collect()
    assert ref() is None  # the streamed contribution is never retained
    s = op.summary()
    assert s["pure_trial"] is True and s["generation_bound"] is False and s["exactly_once"] is False and s["n_species_enabled"] == 1
    assert op.retained_bytes == (4 + 2 * 4 + 4) * 8


# ---- identity -----------------------------------------------------------------------------------------------------------------------------
def test_identity_semantics_twins_parameters_weights_and_cold_verification(backend):
    col = hand_column(2, 5, backend)
    a, b = hand_op(col, a3=0.4, maxrate=1e-6), hand_op(col, a3=0.4, maxrate=1e-6)
    assert a.sha256 == b.sha256 and a is not b
    variants = (dict(a3=0.5, maxrate=1e-6), dict(a3=0.4, maxrate=2e-6), dict(a3=0.4, maxrate=1e-6, maxf=0.9), dict(a3=0.4, maxrate=1e-6, background=[1.0, 1.0]),
                dict(a3=0.4, maxrate=1e-6, background=[1.0, 0.0], species=[[0.0, 1.0]], contributions=[np.ones(5)]))
    assert len({hand_op(col, **v).sha256 for v in variants} | {a.sha256}) == len(variants) + 1
    cell = hand_op(col, a3=np.full(5, 0.4), maxrate=np.full(5, 1e-6))
    assert cell.sha256 != a.sha256 and dict(cell.layouts) == {"a3": "cell", "max_intake_rate_m_s": "cell"}  # same values, other layout: another identity by design
    assert spf.verify_operator_identity(a) is True
    object.__setattr__(a, "_law", (a._law[0], np.frombuffer(np.array([9e-6]).tobytes(), dtype=np.float64), a._law[2]))
    assert spf.verify_operator_identity(a) is False


def test_the_identity_hash_is_streamed_and_never_run_per_step(monkeypatch, backend):
    from maple_syrup import spatial_matrix_infiltration as sm

    col = hand_column(2, 3000, backend)
    mr = np.linspace(1e-7, 4e-6, 3000)
    d1 = hand_op(col, a3=0.5, maxrate=mr).sha256
    monkeypatch.setattr(sm, "_HASH_CHUNK", 64)
    assert hand_op(col, a3=0.5, maxrate=mr).sha256 == d1
    monkeypatch.undo()
    op = hand_op(col, a3=0.5, maxrate=mr)
    monkeypatch.setattr(sm, "_hash_array", lambda *a, **k: (_ for _ in ()).throw(AssertionError("hot rehash")))
    step(op, full(col, 0.01), hand_state(col), 60.0)  # steps never hash
    with pytest.raises(AssertionError):
        spf.verify_operator_identity(op)


# ---- input validation ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,bad", [("a3", -0.1), ("a3", 1.0000001), ("a3", np.nan), ("a3", np.inf), ("max_intake_rate_m_s", -1e-9), ("max_intake_rate_m_s", np.nan)])
def test_invalid_scalar_or_map_inputs_are_refused_on_every_cell_inactive_included(backend, name, bad):
    col = hand_column(2, 4, backend, mask=[True, True, False, True])
    base = {"a3": 0.5, "max_intake_rate_m_s": 1e-6}
    for position in (0, 2):
        arr = np.full(4, base[name])
        arr[position] = bad
        with pytest.raises(sp.SpatialLayerError):
            spf.prepare_spatial_preferential(col, **{**base, name: arr}, max_fraction_per_hour=1.0, background_receiver_weights=[1, 0], memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError):
        spf.prepare_spatial_preferential(col, **{**base, name: bad}, max_fraction_per_hour=1.0, background_receiver_weights=[1, 0], memory_budget_bytes=BIG)


def test_plus_infinity_is_a_valid_max_rate_and_nothing_else_is_coerced(backend):
    col = hand_column(2, 3, backend)
    ok = dict(a3=0.5, max_fraction_per_hour=1.0, background_receiver_weights=[1, 0], memory_budget_bytes=BIG)
    spf.prepare_spatial_preferential(col, max_intake_rate_m_s=np.inf, **ok)
    spf.prepare_spatial_preferential(col, max_intake_rate_m_s=np.array([1.0, np.inf, 0.0]), **ok)
    for bad in (True, "1", [1e-6] * 3, np.full(3, 1e-6, dtype=np.float32), np.full(4, 1e-6), np.full((3, 1), 1e-6), np.full(6, 1e-6)[::2], np.full(3, 1, dtype=np.int64)):
        with pytest.raises(sp.SpatialLayerError):
            spf.prepare_spatial_preferential(col, max_intake_rate_m_s=bad, **ok)
    for mf in (-0.1, 1.1, np.nan, True, "1"):
        with pytest.raises(sp.SpatialLayerError):
            spf.prepare_spatial_preferential(col, max_intake_rate_m_s=1.0, **{**ok, "max_fraction_per_hour": mf})


@pytest.mark.parametrize("weights", [[1.0, -0.1], [1.0, np.nan], [1.0, np.inf], [1.0], [1.0, 1.0, 1.0], 1.0, True, "ab", [[1.0, 0.0]]])
def test_invalid_receiver_weights_are_refused_typed(backend, weights):
    col = hand_column(2, 3, backend)
    with pytest.raises(ss.SubsurfaceError):
        spf.prepare_spatial_preferential(col, a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=1.0, background_receiver_weights=weights, memory_budget_bytes=BIG)
    with pytest.raises(ss.SubsurfaceError):
        spf.prepare_spatial_preferential(col, a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=1.0, background_receiver_weights=[1, 0], species_receiver_weights=[weights],
                                         species_contributions=iter([np.ones(3)]), memory_budget_bytes=BIG)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")  # the (unchanged) common derivation adds the huge contributions before it refuses the overflowed cover
def test_species_contribution_protocol_and_values(backend):
    col = hand_column(2, 4, backend, mask=[True, False, True, True])
    kw = dict(a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=1.0, background_receiver_weights=[1, 0], memory_budget_bytes=BIG)
    one = [[0.0, 1.0]]
    with pytest.raises(sp.SpatialLayerError):
        spf.prepare_spatial_preferential(col, species_receiver_weights=one, **kw)  # species without contributions
    with pytest.raises(sp.SpatialLayerError):
        spf.prepare_spatial_preferential(col, species_contributions=iter([np.ones(4)]), **kw)  # contributions without species
    for streams in ([], [np.ones(4), np.ones(4)]):
        with pytest.raises(ss.SubsurfaceError):
            spf.prepare_spatial_preferential(col, species_receiver_weights=one, species_contributions=iter(streams), **kw)
    for bad in (np.array([1.0, np.nan, 1.0, 1.0]), np.array([1.0, -1e-9, 1.0, 1.0]), np.array([1.0, 1.0, np.inf, 1.0]), np.ones(3), np.ones(4, dtype=np.float32), [1.0] * 4):
        with pytest.raises(ss.SubsurfaceError):
            spf.prepare_spatial_preferential(col, species_receiver_weights=one, species_contributions=iter([bad]), **kw)  # an INACTIVE cell (index 1) is validated too
    with pytest.raises(ss.SubsurfaceError, match="overflow"):
        spf.prepare_spatial_preferential(col, species_receiver_weights=[[0, 1], [0, 1]], species_contributions=iter([np.full(4, 1.7e308), np.full(4, 1.7e308)]), **kw)
    # a (S, N) array iterates by rows and an inactive cell's value contributes nothing
    c = np.array([[1.0, 5.0, 1.0, 1.0]])
    op = spf.prepare_spatial_preferential(col, species_receiver_weights=one, species_contributions=c, **kw)
    assert op.n_species == 1 and op.n_species_enabled == 1
    np.testing.assert_array_equal(op._law[2][:, 1], 0.0)  # no receiver share on the inactive cell


CASES = {0: ("water", 0, np.nan), 1: ("water", 1, -1e-3), 2: ("water", 0, 10.0), 3: ("surface", 0, np.nan), 4: ("surface", 2, -1.0)}


@pytest.mark.parametrize("bit", sorted(CASES))
@pytest.mark.parametrize("cell_kind", ["active", "inactive"])
def test_every_invalid_input_cell_is_refused_with_the_common_reason_and_nothing_is_written(backend, bit, cell_kind):
    col = hand_column(2, 4, backend, mask=[True, True, False, True])
    op = hand_op(col)
    ok = hand_state(col, t_s=5.0)
    water = np.array(ok.water_m)
    surface = np.full(4, 0.01)
    which, cell, value = CASES[bit]
    cell = {"active": 0, "inactive": 2}[cell_kind]
    (water if which == "water" else surface)[(0, cell) if which == "water" else cell] = value
    state = dataclasses.replace(ok, water_m=ro(water)) if which == "water" else ok
    before = [np.array(water, copy=True), np.array(surface, copy=True)]
    with pytest.raises(pi.PreferentialStepError) as info:
        spf.spatial_preferential_step(op, surface, state, 60.0, memory_budget_bytes=BIG)
    assert info.value.bit == bit and info.value.n_flagged >= 1 and pi.FLAG_MESSAGES[bit].split(" ")[0] in str(info.value)
    for a, b in zip((water, surface), before, strict=True):
        np.testing.assert_array_equal(a, b)  # inputs untouched (NaN-aware)


def test_unrepresentable_global_volume_is_a_typed_refusal(backend):
    spec = sp.spatial_layer_spec(np.array([0.0, 0.5, 1.5]), theta_residual=0.01, theta_field_capacity=0.25, theta_saturation=0.9, theta_evaporation_stress=0.05, diffusivity_m2_s=0.0, n_cells=2)
    col = sp.prepare_spatial_column(spec, np.ones(2, bool), 1e308, backend=backend, memory_budget_bytes=BIG)
    op = hand_op(col)
    saturated = np.array([[0.9 * 0.5] * 2, [0.9] * 2])  # a valid nearly full column: before = 2.7 x 1e308 m3 overflows FP64
    with pytest.raises(pi.PreferentialStepError, match="non-finite global water budget"):
        step(op, full(col, 0.01), hand_state(col, water=saturated), 60.0)


def test_metadata_checks_precede_everything(backend):
    col = hand_column(2, 3, backend)
    op = hand_op(col)
    state, H = hand_state(col), full(col, 0.01)
    for dt in (-1.0, float("nan"), True, "60"):
        with pytest.raises(ss.SubsurfaceError):
            spf.spatial_preferential_step(op, H, state, dt, memory_budget_bytes=BIG)
    for bad in (H.astype(np.float32), H[:2], list(H), np.ma.masked_array(H), np.repeat(H, 2)[::2]):
        with pytest.raises(sp.SpatialLayerError):
            spf.spatial_preferential_step(op, bad, state, 60.0, memory_budget_bytes=BIG)
    other = hand_column(2, 4, backend)
    with pytest.raises(sp.SpatialLayerError):
        spf.spatial_preferential_step(op, H, hand_state(other), 60.0, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError):
        spf.spatial_preferential_step(op, H, state, 60.0, memory_budget_bytes=BIG, reserve_bytes=-1)


# ---- admission, quotas, readers, reserves -------------------------------------------------------------------------------------------------
def test_admission_precedes_every_allocation_conversion_weight_record_and_compile(backend):
    N = 60000
    col = hand_column(2, N, backend)
    if backend == "numba":
        from maple_syrup import preferential_infiltration_numba as pn

        pn.reset_compiled()
    stream_touched = []

    def stream():
        stream_touched.append(1)
        yield np.ones(N)

    kw = dict(a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=np.full(N, 1e-6), background_receiver_weights=[1, 0], species_receiver_weights=[[0, 1]])
    tracemalloc.start()
    try:
        with pytest.raises(sp.SpatialLayerMemoryError):
            spf.prepare_spatial_preferential(col, species_contributions=stream(), memory_budget_bytes=1000, **kw)
        with pytest.raises(sp.SpatialLayerMemoryError):
            spf.prepare_spatial_preferential(col, species_contributions=stream(), memory_budget_bytes=BIG, available_bytes_fn=lambda: 1000, **kw)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 100_000 and stream_touched == []  # nothing of N-scale allocated, the species stream never consumed
    if backend == "numba":
        assert pn._SPATIAL_KERNEL is None
    need = sm_est("prepare", N, backend, 1).required_bytes
    with pytest.raises(sp.SpatialLayerMemoryError):
        spf.prepare_spatial_preferential(col, species_contributions=stream(), memory_budget_bytes=need - 1, **kw)
    with pytest.raises(sp.SpatialLayerMemoryError):
        spf.prepare_spatial_preferential(col, species_contributions=stream(), memory_budget_bytes=need, reserve_bytes=1, **kw)
    op = spf.prepare_spatial_preferential(col, species_contributions=iter([np.ones(N)]), memory_budget_bytes=need, **kw)  # exactly the requirement is admitted
    state, H = hand_state(col), full(col, 0.01)
    need = sm_est("step", N, backend, 1).required_bytes
    tracemalloc.start()
    try:
        with pytest.raises(sp.SpatialLayerMemoryError):
            spf.spatial_preferential_step(op, H, state, 60.0, memory_budget_bytes=need - 1)
        with pytest.raises(sp.SpatialLayerMemoryError):
            spf.spatial_preferential_step(op, H, state, 60.0, memory_budget_bytes=need, reserve_bytes=1)
        with pytest.raises(sp.SpatialLayerMemoryError):
            spf.spatial_preferential_step(op, H, state, 60.0, memory_budget_bytes=BIG, available_bytes_fn=lambda: need - 1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 100_000
    spf.spatial_preferential_step(op, H, state, 60.0, memory_budget_bytes=need)
    spf.spatial_preferential_step(op, H, state, 60.0, memory_budget_bytes=need + 5, reserve_bytes=5, available_bytes_fn=lambda: need + 5)
    for bad in (True, -5, 1.5, None):
        with pytest.raises(sp.SpatialLayerError):
            spf.spatial_preferential_step(op, H, state, 60.0, memory_budget_bytes=bad)
    for bad in (True, -1, 2.5):
        with pytest.raises(sp.SpatialLayerError):
            spf.spatial_preferential_step(op, H, state, 60.0, memory_budget_bytes=BIG, available_bytes_fn=lambda b=bad: b)
    with pytest.raises(sp.SpatialLayerError):  # CPU owners refuse the CUDA-only host quota arguments (S2-B2.d; earlier slices raised TypeError because the arguments did not exist)
        spf.spatial_preferential_step(op, H, state, 60.0, memory_budget_bytes=BIG, host_memory_budget_bytes=1)


def sm_est(operation, N, backend, species=0):
    return spf.estimate_spatial_preferential_memory(2, N, backend=backend, operation=operation, n_species=species)


def test_the_estimate_is_pure_arithmetic_and_documented():
    for backend in ("numpy", "numba"):
        e = spf.estimate_spatial_preferential_memory(3, 1000, backend=backend, operation="step", reserve_bytes=7)
        L, N = 3, 1000
        per = pi.NUMPY_SCRATCH_DOUBLES_PER_CELL if backend == "numpy" else pi.KERNEL_SCRATCH_DOUBLES_PER_CELL
        want = (2 * N + 2 * L * N) * 8 + pi.ROW_COUNT * N * 8 + per * N * 8 + pi.SCRATCH_MASK_BYTES_PER_CELL * N + pi._PACKET_SCRATCH_BYTES + (spf.ROW_TEMP_DOUBLES * N * 8 if backend == "numpy" else 2 * L * 8) \
            + sp.PY_ALLOWANCE_BYTES + 7
        assert e.required_bytes == want and e.reserve_bytes == 7 and e.local_bytes == (0 if backend == "numpy" else 48)
        assert "caller's state" in e.scope and e.terms["no_LN_derived_bound_tensor"] is True
        p = spf.estimate_spatial_preferential_memory(3, 1000, backend=backend, operation="prepare", n_species=2)
        assert p.required_bytes == (2 * N + L * N) * 8 + (4 * N + L * N) * 8 + pi.BUILD_SCRATCH_MASK_BYTES_PER_CELL * N + 2 * 3 * L * 8 + sp.PY_ALLOWANCE_BYTES and p.weight_bytes == 2 * 3 * L * 8
    for bad in (dict(backend="opencl"), dict(backend="numpy", operation="nope"), dict(backend="numpy", host_reserve_bytes=1)):
        with pytest.raises(sp.SpatialLayerError):
            spf.estimate_spatial_preferential_memory(2, 4, **bad)
    with pytest.raises(sp.SpatialLayerError):
        spf.estimate_spatial_preferential_memory(0, 4, backend="numpy")


def test_numpy_peaks_of_prepare_and_step_are_within_their_models():
    N = 50000
    col = hand_column(2, N, "numpy")
    kw = dict(a3=np.full(N, 0.4), max_fraction_per_hour=0.9, max_intake_rate_m_s=np.full(N, 1e-6), background_receiver_weights=[1, 1], species_receiver_weights=[[0, 1]])
    gc.collect()
    tracemalloc.start()
    try:
        op = spf.prepare_spatial_preferential(col, species_contributions=iter([np.ones(N)]), memory_budget_bytes=BIG, **kw)
        _, peak_p = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    model_p = op.prepare_memory.required_bytes
    assert peak_p <= model_p, (peak_p, model_p)
    assert peak_p >= 0.3 * model_p
    state, H = hand_state(col), full(col, 0.01)
    model = sm_est("step", N, "numpy").required_bytes
    gc.collect()
    tracemalloc.start()
    try:
        r = spf.spatial_preferential_step(op, H, state, 60.0, memory_budget_bytes=BIG)
        _, peak = tracemalloc.get_traced_memory()
        kept = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    assert peak <= model, (peak, model)
    assert peak >= 0.3 * model
    assert kept >= r.surface_water_m.nbytes + r.intake_total_m.nbytes + r.state.water_m.nbytes + r.intake_m.nbytes


def test_numba_heap_blocks_and_python_side_allocations_follow_the_model():
    pytest.importorskip("numba")
    from numba.core.runtime import _nrt_python as nrt

    N = 40000
    col = hand_column(2, N, "numba")
    op = hand_op(col, a3=0.5, maxrate=1e-6)
    state, H = hand_state(col), full(col, 0.01)
    step(op, H, state, 60.0)
    lcol, lop, lstate = legacy("numba", N)
    pi.preferential_step(lop, H, lstate, 60.0)
    nrt.memsys_enable_stats()
    a0, f0 = nrt.memsys_get_stats_alloc(), nrt.memsys_get_stats_free()
    step(op, H, state, 60.0)
    a1, f1 = nrt.memsys_get_stats_alloc(), nrt.memsys_get_stats_free()
    pi.preferential_step(lop, H, lstate, 60.0)
    a2, f2 = nrt.memsys_get_stats_alloc(), nrt.memsys_get_stats_free()
    nrt.memsys_disable_stats()
    assert f1 - f0 == a1 - a0 and f2 - f1 == a2 - a1
    spatial_blocks, common_blocks = a1 - a0, a2 - a1
    # per call: one NRT adaptation per array argument + the two (L,) local rows. The spatial launch passes 15 arrays (the raw dz and theta_sat maps and two aliased unused common slots) against 13 for
    # the common kernel, so the difference is 2 extra adaptations + exactly the 2 local rows (sat, cap_tol); the count is independent of N and L.
    assert spatial_blocks == 15 + 2 and common_blocks == 13 and spatial_blocks - common_blocks == 2 + 2
    for n2, l2 in ((1000, 2), (1000, 5)):
        col2 = hand_column(l2, n2, "numba")
        op2, st2 = hand_op(col2, a3=0.5, maxrate=1e-6), hand_state(col2)
        step(op2, full(col2, 0.01), st2, 60.0)
        nrt.memsys_enable_stats()
        b0 = nrt.memsys_get_stats_alloc()
        step(op2, full(col2, 0.01), st2, 60.0)
        assert nrt.memsys_get_stats_alloc() - b0 == spatial_blocks  # N- and L-independent
        nrt.memsys_disable_stats()
    model = sm_est("step", N, "numba")
    assert model.local_bytes == 2 * 2 * 8
    gc.collect()
    tracemalloc.start()
    try:
        r = step(op, H, state, 60.0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= model.required_bytes and peak >= model.output_bytes + model.row_bytes
    del r


def test_owner_release_and_function_only_caches(backend):
    col = hand_column(2, 50, backend)
    op = hand_op(col, a3=0.5)
    r = step(op, full(col, 0.01), hand_state(col), 60.0)
    refs = [weakref.ref(x) for x in (op, r.state, r.surface_water_m, r.intake_m)]
    del op, r
    gc.collect()
    assert all(w() is None for w in refs)
    if backend == "numba":
        from maple_syrup import preferential_infiltration_numba as pn

        assert type(pn._SPATIAL_KERNEL).__name__ == "CPUDispatcher"


# ---- genuine compiled proof -------------------------------------------------------------------------------------------------------------
def test_the_numba_kernel_is_a_genuine_nopython_readonly_explicit_signature_body(monkeypatch):
    numba = pytest.importorskip("numba")
    from maple_syrup import preferential_infiltration_numba as pn

    k = pn.compiled_spatial_kernel()
    assert len(k.signatures) == 1 and k.targetoptions.get("nopython") is True and not k.targetoptions.get("fastmath") and not k._can_compile
    sig = k.signatures[0]
    ro_i = [i for i, a in enumerate(sig) if isinstance(a, numba.types.Array) and not a.mutable]
    rw_i = [i for i, a in enumerate(sig) if isinstance(a, numba.types.Array) and a.mutable]
    assert len(sig) == 22 and rw_i == [17, 18, 19, 20, 21] and len(ro_i) == 10  # only the five fresh outputs are writable
    assert pn.compiled_spatial_kernel() is k and pn.compiled_kernel() is not k
    col = hand_column(2, 4, "numba")
    op = hand_op(col, a3=0.5)
    state, H = hand_state(col), full(col, 0.01)
    ref = step(op, H, state, 600.0)

    def boom(*a, **kw):
        raise AssertionError("the NumPy body must not run for a numba owner (no fallback)")

    monkeypatch.setattr(pi, "_reference_step_body", boom)
    again = step(op, H, state, 600.0)
    np.testing.assert_array_equal(again.intake_m, ref.intake_m)


def test_a_failed_or_unavailable_compile_is_typed_and_never_falls_back(monkeypatch):
    pytest.importorskip("numba")
    from maple_syrup import preferential_infiltration_numba as pn

    col = hand_column(2, 3, "numba")
    pn.reset_compiled()
    monkeypatch.setattr(pn, "_build", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("synthetic compiler defect")))
    with pytest.raises(sp.SpatialLayerBackendUnavailableError, match="no NumPy fallback"):
        hand_op(col)
    monkeypatch.undo()
    pn.reset_compiled()
    monkeypatch.setattr(pn, "numba_available", lambda: False)
    with pytest.raises(sp.SpatialLayerBackendUnavailableError):
        hand_op(col)
    monkeypatch.undo()
    pn.reset_compiled()
    assert hand_op(col).backend == "numba"
