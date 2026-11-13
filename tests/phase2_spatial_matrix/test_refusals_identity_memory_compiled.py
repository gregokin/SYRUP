"""Typed refusals, immutability, identity, admission/quotas, memory model, owner release and the genuine compiled proof of the spatial matrix operator. Author-written, CPU only, no GPU claim."""
from __future__ import annotations

import copy
import dataclasses
import gc
import pickle
import tracemalloc
import weakref

import numpy as np
import pytest
from conftest import BIG, cases, full, hand_column, hand_hist, hand_op, hand_state, mi, sm, sp, ss, step


def forcing_ok(col):
    n = col.n_cells
    return full(col, 0.01), full(col, 0.01), full(col, 0.0)  # contact pond: pool, standing, rate


def snapshot(*arrays):
    return [np.array(a, copy=True) for a in arrays]


# ---- cross refusals / sealing -------------------------------------------------------------------------------------------------------------
def test_legacy_and_spatial_objects_refuse_each_other(backend):
    col = hand_column(3, backend)
    op, state = hand_op(col), hand_state(col)
    hist = hand_hist(op)
    lay = cases.layers()
    lcol = cases.column(lay, n_cells=3, backend=backend)
    lop = cases.operator(lcol)
    lstate = cases.state_of(lcol)
    lhist = cases.history_of(lop)
    pool, standing, rate = forcing_ok(col)
    with pytest.raises(sp.SpatialLayerError):
        sm.prepare_spatial_matrix(lcol, conductivity_m_s=1e-5, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG)
    for bad_op in (lop, object(), None):
        with pytest.raises(sp.SpatialLayerError):
            sm.spatial_matrix_step(bad_op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError):
        sm.spatial_matrix_step(op, pool, standing, rate, lstate, hist, 60.0, memory_budget_bytes=BIG)  # legacy state
    with pytest.raises(sp.SpatialLayerError):
        sm.spatial_matrix_step(op, pool, standing, rate, state, lhist, 60.0, memory_budget_bytes=BIG)  # legacy history
    with pytest.raises(sp.SpatialLayerError):
        sm.spatial_matrix_history(lop, wetting_m=0.0, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    # and the legacy entry points refuse the spatial objects (the old guards are NOT widened)
    with pytest.raises(ss.SubsurfaceError):
        mi.prepare_matrix(col, conductivity_m_s=1e-5, suction_m=0.1, vegetation_factor=0.5)
    with pytest.raises(ss.SubsurfaceError):
        mi.matrix_step(op, pool, standing, rate, lstate, lhist, 60.0)
    with pytest.raises(ss.SubsurfaceError):
        mi.matrix_step(lop, pool, standing, rate, state, lhist, 60.0)
    with pytest.raises(ss.SubsurfaceError):
        mi.wetting_history(op, wetting_m=0.0, cumulative_intake_m=0.0)
    for typed in (sp.SpatialLayerError, mi.MatrixStepError):
        assert issubclass(typed, ss.SubsurfaceError)


def test_cuda_owner_without_a_usable_device_is_refused_before_any_copy_validation_or_reader(monkeypatch):
    """S2-B2.c: a CUDA owner is supported when CuPy and its device are usable; without them (simulated here, independent of the host) the typed refusal precedes every copy, validation and reader call."""
    def no_cuda():
        raise sp.SpatialLayerBackendUnavailableError("no usable CuPy/device (simulated)")

    monkeypatch.setattr(sp, "_require_cuda", no_cuda)
    col = hand_column(2)
    with sp._constructing():
        fake = dataclasses.replace(col, backend="cuda", device_id=0)
    calls = []
    with pytest.raises(sp.SpatialLayerBackendUnavailableError):
        sm.prepare_spatial_matrix(fake, conductivity_m_s=object(), suction_m=object(), vegetation_factor=object(), memory_budget_bytes=BIG, available_bytes_fn=lambda: calls.append(1) or BIG)
    assert calls == []  # not even the available-bytes reader was consulted


def test_direct_construction_copy_and_pickle_are_refused(backend):
    col = hand_column(2, backend)
    op = hand_op(col)
    hist = hand_hist(op)
    for obj in (op, hist):
        with pytest.raises(sp.SpatialLayerError):
            copy.copy(obj)
        with pytest.raises(sp.SpatialLayerError):
            copy.deepcopy(obj)
        with pytest.raises(sp.SpatialLayerError):
            pickle.dumps(obj)
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.n_cells = 5
    with pytest.raises(sp.SpatialLayerError):
        dataclasses.replace(op, backend=backend)
    with pytest.raises(sp.SpatialLayerError):
        sm.SpatialMatrixHistory(np.zeros(2), np.zeros(2), op.sha256, 2)
    with pytest.raises(sp.SpatialLayerError):
        sm.SpatialMatrixOperator(col, backend, 2, 2, col.sha256, op.sha256, {}, 0, op.prepare_memory, op._law)


def test_the_law_storage_is_sealed_and_a_caller_array_is_copied(backend):
    col = hand_column(4, backend)
    k = np.full(4, 1e-5)
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=k, suction_m=0.08, vegetation_factor=np.full(4, 0.5), memory_budget_bytes=BIG)
    k[:] = 7.0  # the caller's array may change afterwards: the operator keeps its own copy
    assert float(op._law[0][0]) == 1e-5 and op._law[1].shape == (1,) and op._law[2].shape == (4,)
    for a in op._law:
        assert not a.flags.writeable
        with pytest.raises(ValueError):
            a.flags.writeable = True  # bytes-backed storage cannot be re-opened
        with pytest.raises(ValueError):
            a[0] = 1.0
    assert op.summary()["pure_trial"] is True and op.summary()["generation_bound"] is False and op.summary()["exactly_once"] is False
    h = hand_hist(op, np.full(4, 0.001), 0.0)
    assert not h.wetting_m.flags.writeable and not h.cumulative_intake_m.flags.writeable
    assert h.summary()["boundary_enforced"] is False and h.summary()["generation_bound"] is False


# ---- identity -----------------------------------------------------------------------------------------------------------------------------
def test_identity_semantics_twins_layouts_backends_and_cold_verification(backend):
    col = hand_column(5, backend)
    a = hand_op(col)
    b = hand_op(col)  # a twin from identical inputs
    assert a.sha256 == b.sha256 and a is not b
    ha = hand_hist(a, 0.004)
    pool, standing, rate = forcing_ok(col)
    step(b, pool, standing, rate, hand_state(col), ha, 60.0)  # twin accepts the history: the binding is to INPUTS, not to the owner object
    assert hand_op(col, k=2e-5).sha256 != a.sha256 and hand_op(col, psi=0.09).sha256 != a.sha256 and hand_op(col, g=0.6).sha256 != a.sha256
    cell = sm.prepare_spatial_matrix(col, conductivity_m_s=np.full(5, cases.K), suction_m=cases.PSI, vegetation_factor=cases.G, memory_budget_bytes=BIG)
    assert cell.sha256 != a.sha256  # same values, different layout: a different identity by design
    with pytest.raises(sp.SpatialLayerError):
        step(hand_op(col, k=2e-5), pool, standing, rate, hand_state(col), ha, 60.0)  # a history of other inputs is refused, never reinterpreted
    other = hand_column(5, backend, mask=[True, True, False, True, True])
    with pytest.raises(sp.SpatialLayerError):
        step(hand_op(other), pool, standing, rate, hand_state(other), ha, 60.0)  # other column (mask) -> other identity
    assert sm.verify_operator_identity(a) is True
    other_backend = "numba" if backend == "numpy" else "numpy"
    if other_backend == "numpy" or pytest.importorskip("numba"):
        assert hand_op(hand_column(5, other_backend)).sha256 != a.sha256
    # tamper: replace the sealed arrays with other read-only bytes; the cold verification detects it, the metadata check alone does not claim to
    fake = np.frombuffer(np.array([9e-5]).tobytes(), dtype=np.float64)
    object.__setattr__(a, "_law", (fake, a._law[1], a._law[2]))
    assert sm.verify_operator_identity(a) is False


def test_the_identity_hash_is_streamed_in_bounded_chunks(monkeypatch):
    col = hand_column(3000)
    k = np.linspace(1e-6, 2e-5, 3000)
    full_digest = sm.prepare_spatial_matrix(col, conductivity_m_s=k, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG).sha256
    monkeypatch.setattr(sm, "_HASH_CHUNK", 64)
    assert sm.prepare_spatial_matrix(col, conductivity_m_s=k, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG).sha256 == full_digest


# ---- value / metadata validation ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,bad", [("conductivity_m_s", -1e-9), ("conductivity_m_s", np.nan), ("conductivity_m_s", np.inf), ("suction_m", -0.1), ("suction_m", np.nan),
                                       ("vegetation_factor", 1.0000001), ("vegetation_factor", -0.1), ("vegetation_factor", np.nan)])
def test_invalid_law_values_are_refused_on_every_cell_inactive_included(backend, name, bad):
    mask = [True, True, False, True]
    col = hand_column(4, backend, mask=mask)
    base = {"conductivity_m_s": 1e-5, "suction_m": 0.1, "vegetation_factor": 0.5}
    for position in (0, 2):  # an active and an INACTIVE cell
        arr = np.full(4, base[name])
        arr[position] = bad
        with pytest.raises(sp.SpatialLayerError):
            sm.prepare_spatial_matrix(col, **{**base, name: arr}, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError):
        sm.prepare_spatial_matrix(col, **{**base, name: bad}, memory_budget_bytes=BIG)


@pytest.mark.parametrize("make", [lambda n: True, lambda n: "0.1", lambda n: [0.1] * n, lambda n: np.full(n, 0.1, dtype=np.float32), lambda n: np.full(n + 1, 0.1), lambda n: np.full((n, 1), 0.1),
                                  lambda n: np.full(2 * n, 0.1)[::2], lambda n: np.full(n, 1, dtype=np.int64), lambda n: np.ma.masked_array(np.full(n, 0.1))])
def test_law_input_metadata_is_exact(backend, make):
    col = hand_column(4, backend)
    with pytest.raises(sp.SpatialLayerError):
        sm.prepare_spatial_matrix(col, conductivity_m_s=1e-5, suction_m=make(4), vegetation_factor=0.5, memory_budget_bytes=BIG)


@pytest.mark.parametrize("which,bad", [("wetting_m", -1.0), ("wetting_m", np.nan), ("wetting_m", np.inf), ("cumulative_intake_m", -1e-12), ("cumulative_intake_m", np.nan)])
def test_history_constructor_validates_every_cell_and_requires_explicit_values(backend, which, bad):
    col = hand_column(4, backend, mask=[True, False, True, True])
    op = hand_op(col)
    ok = {"wetting_m": 0.0, "cumulative_intake_m": 0.0}
    for pos in (0, 1):
        arr = np.zeros(4)
        arr[pos] = bad
        with pytest.raises(sp.SpatialLayerError):
            sm.spatial_matrix_history(op, **{**ok, which: arr}, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError):
        sm.spatial_matrix_history(op, **{**ok, which: bad}, memory_budget_bytes=BIG)
    with pytest.raises(TypeError):
        sm.spatial_matrix_history(op, wetting_m=0.0, memory_budget_bytes=BIG)  # nothing defaults to zero
    with pytest.raises(sp.SpatialLayerError):
        sm.spatial_matrix_history(op, wetting_m=np.zeros(5), cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError):
        sm.spatial_matrix_history(op, wetting_m=True, cumulative_intake_m=0.0, memory_budget_bytes=BIG)


CASES = {  # flag bit -> (array to corrupt, cell, value)
    0: ("water", 0, np.nan), 1: ("water", 1, -1e-3), 2: ("water", 0, 10.0), 3: ("pool", 0, np.nan), 4: ("pool", 1, -1.0), 5: ("standing", 0, np.inf), 6: ("standing", 0, -1.0),
    7: ("rate", 0, np.nan), 8: ("rate", 0, -1e-6), 9: ("rate", 2, 1e-6), 10: ("wet", 0, np.nan), 11: ("wet", 2, -1.0), 12: ("cum", 0, np.inf), 13: ("cum", 2, -1.0)}


@pytest.mark.parametrize("bit", sorted(CASES))
def test_every_invalid_input_cell_is_refused_with_the_common_reason_and_nothing_is_written(backend, bit):
    col = hand_column(4, backend, mask=[True, True, False, True])
    op = hand_op(col)
    water = np.array([[0.0625] * 4, [0.1] * 4])
    pool, standing, rate = np.full(4, 0.01), np.full(4, 0.01), np.zeros(4)
    wet, cum = np.full(4, 0.002), np.full(4, 0.003)
    which, cell, value = CASES[bit]
    target = {"water": water, "pool": pool, "standing": standing, "rate": rate, "wet": wet, "cum": cum}[which]
    if which == "water":
        target[0, cell] = value
        # the owner validates the state it builds: construct the state first with valid water, then swap the corrupt copy into a state record
        ok = sp.spatial_column_state(col, np.array([[0.0625] * 4, [0.1] * 4]), 5.0, memory_budget_bytes=BIG)
        bad = water.copy()
        state = dataclasses.replace(ok, water_m=_ro(bad))
    else:
        target[cell] = value
        state = hand_state(col, t_s=5.0)
    if which in ("wet", "cum"):
        base = {"wetting_m": np.full(4, 0.002), "cumulative_intake_m": np.full(4, 0.003)}
        hist = _forge_history(op, wet, cum)
    else:
        hist = hand_hist(op, 0.002, 0.003)
    before = snapshot(water, pool, standing, rate, wet, cum)
    with pytest.raises(mi.MatrixStepError) as info:
        sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=BIG)
    assert info.value.bit == bit and info.value.n_flagged >= 1 and mi.FLAG_MESSAGES[bit].split(" ")[0] in str(info.value)
    for a, b in zip(snapshot(water, pool, standing, rate, wet, cum), before, strict=True):
        np.testing.assert_array_equal(a, b)  # inputs untouched (NaN-aware)


def _ro(a):
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    return a


def _forge_history(op, wet, cum):
    """A history record whose arrays carry invalid values: only a test can build it (through the minting context), as it would otherwise be refused by the validated constructor."""
    with sm._minting():
        return sm.SpatialMatrixHistory(_ro(np.array(wet)), _ro(np.array(cum)), op.sha256, op.n_cells)


def test_inconsistent_split_flag_and_nonconvergence_are_typed(backend, monkeypatch):
    col = hand_column(3, backend)
    op = hand_op(col)
    state = hand_state(col)
    with pytest.raises(mi.MatrixStepError) as info:  # no contact, pool above the supply r dt
        step(op, full(col, 0.5), full(col, 0.0), full(col, 1e-6), state, hand_hist(op), 60.0)
    assert info.value.bit == 14 and info.value.n_flagged == 3
    if backend == "numpy":
        monkeypatch.setattr(mi, "NEWTON_MAX_ITERATIONS", 1)
    else:
        from maple_syrup import matrix_infiltration_numba as mn

        monkeypatch.setattr(mn, "NEWTON_MAX_ITERATIONS", 1)
    with pytest.raises(mi.MatrixStepError) as info:
        step(op, full(col, 1.0), full(col, 1.0), full(col, 0.0), state, hand_hist(op, 0.002), 600.0)
    assert info.value.bit == 15  # non-convergence is flagged and refused, never clipped


def test_unrepresentable_global_volume_is_a_typed_refusal(backend):
    """Finite state values whose global volume (depth sum x area) overflows FP64 are refused with the typed step error BEFORE the strict MAPLE helper (no raw ValueError)."""
    spec = sp.spatial_layer_spec(np.array([0.0, 0.5, 1.5]), theta_residual=0.01, theta_field_capacity=0.25, theta_saturation=0.9, theta_evaporation_stress=0.05, diffusivity_m2_s=0.0, n_cells=2)
    col = sp.prepare_spatial_column(spec, np.ones(2, bool), 1e308, backend=backend, memory_budget_bytes=BIG)
    op = hand_op(col)
    saturated = np.array([[cases.SAT * cases.DZ0] * 2, [cases.SAT * 1.0] * 2])  # a valid, nearly full column: before = 2.7 x 1e308 m3 overflows
    with pytest.raises(mi.MatrixStepError, match="non-finite global water budget"):
        step(op, full(col, 0.01), full(col, 0.01), full(col, 0.0), hand_state(col, water=saturated), hand_hist(op), 60.0)


def test_metadata_checks_precede_everything(backend):
    col = hand_column(3, backend)
    op = hand_op(col)
    state, hist = hand_state(col), hand_hist(op)
    pool, standing, rate = forcing_ok(col)
    for kwargs in (dict(dt_s=-1.0), dict(dt_s=float("nan")), dict(dt_s=True), dict(dt_s="60")):
        with pytest.raises(ss.SubsurfaceError):
            sm.spatial_matrix_step(op, pool, standing, rate, state, hist, kwargs["dt_s"], memory_budget_bytes=BIG)
    for bad in (pool.astype(np.float32), pool[:2], list(pool), np.ma.masked_array(pool), np.repeat(pool, 2)[::2]):
        with pytest.raises(sp.SpatialLayerError):
            sm.spatial_matrix_step(op, bad, standing, rate, state, hist, 60.0, memory_budget_bytes=BIG)
    other = hand_column(4, backend)
    with pytest.raises(sp.SpatialLayerError):
        sm.spatial_matrix_step(op, pool, standing, rate, hand_state(other), hist, 60.0, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError):
        sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=BIG, reserve_bytes=-1)


# ---- admission, quotas, readers, reserves -------------------------------------------------------------------------------------------------
def test_admission_precedes_every_allocation_conversion_and_compile(backend):
    N = 60000
    col = hand_column(N, backend)
    if backend == "numba":
        from maple_syrup import matrix_infiltration_numba as mn

        mn.reset_compiled()
    laws = dict(conductivity_m_s=np.full(N, 1e-5), suction_m=0.1, vegetation_factor=0.5)
    fn = lambda b, **kw: sm.prepare_spatial_matrix(col, **laws, memory_budget_bytes=b, **kw)  # noqa: E731
    tracemalloc.start()
    try:
        with pytest.raises(sp.SpatialLayerMemoryError):
            fn(1000)
        with pytest.raises(sp.SpatialLayerMemoryError):
            fn(BIG, available_bytes_fn=lambda: 1000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 100_000  # nothing of N-scale was allocated before the refusal
    if backend == "numba":
        assert mn._SPATIAL_KERNEL is None  # and the compile never ran
    op = sm.prepare_spatial_matrix(col, **laws, memory_budget_bytes=BIG)
    hist = sm.spatial_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    state = hand_state(col)
    pool, standing, rate = forcing_ok(col)
    need = sm.estimate_spatial_matrix_memory(2, N, backend=backend, operation="step").required_bytes
    tracemalloc.start()
    try:
        with pytest.raises(sp.SpatialLayerMemoryError):
            sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=need - 1)
        with pytest.raises(sp.SpatialLayerMemoryError):
            sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=need, reserve_bytes=1)  # the reserve is part of the requirement
        with pytest.raises(sp.SpatialLayerMemoryError):
            sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=BIG, available_bytes_fn=lambda: need - 1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 100_000
    sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=need)  # exactly the requirement is admitted
    sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=need + 5, reserve_bytes=5, available_bytes_fn=lambda: need + 5)
    for bad in (True, -5, 1.5, None):
        with pytest.raises(sp.SpatialLayerError):
            sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=bad)
    for bad in (True, -1, 2.5):
        with pytest.raises(sp.SpatialLayerError):
            sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=BIG, available_bytes_fn=lambda b=bad: b)
    with pytest.raises(sp.SpatialLayerError):
        sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=BIG, available_bytes_fn=3)
    with pytest.raises(sp.SpatialLayerError):  # CPU owners refuse the CUDA-only host quota/reserve/reader arguments (S2-B2.c; earlier slices raised TypeError because the arguments did not exist)
        sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=BIG, host_memory_budget_bytes=1)


def test_history_and_prepare_admission(backend):
    col = hand_column(1000, backend)
    op = hand_op(col)
    with pytest.raises(sp.SpatialLayerMemoryError):
        sm.spatial_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0, memory_budget_bytes=100)
    est = sm.estimate_spatial_matrix_memory(2, 1000, backend=backend, operation="history")
    sm.spatial_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0, memory_budget_bytes=est.required_bytes)
    with pytest.raises(sp.SpatialLayerMemoryError):
        sm.spatial_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0, memory_budget_bytes=est.required_bytes - 1)


def test_the_estimate_is_pure_arithmetic_and_documented():
    for backend in ("numpy", "numba"):
        e = sm.estimate_spatial_matrix_memory(3, 1000, backend=backend, operation="step", reserve_bytes=7)
        L, N = 3, 1000
        per = mi.NUMPY_SCRATCH_DOUBLES_PER_CELL if backend == "numpy" else mi.KERNEL_SCRATCH_DOUBLES_PER_CELL
        want = (4 * N + L * N) * 8 + mi.ROW_COUNT * N * 8 + per * N * 8 + mi.SCRATCH_MASK_BYTES_PER_CELL * N + mi._PACKET_SCRATCH_BYTES + (sm.ROW_TEMP_DOUBLES * N * 8 if backend == "numpy" else 2 * L * 8) + sp.PY_ALLOWANCE_BYTES + 7
        assert e.required_bytes == want and e.reserve_bytes == 7 and e.local_bytes == (0 if backend == "numpy" else 48)
        assert "caller's state" in e.scope and e.terms["no_LN_derived_bound_tensor"] is True
    for bad in (dict(backend="opencl"), dict(backend="numpy", operation="nope"), dict(backend="numpy", host_reserve_bytes=1)):
        with pytest.raises(sp.SpatialLayerError):
            sm.estimate_spatial_matrix_memory(2, 4, **bad)
    with pytest.raises(sp.SpatialLayerError):
        sm.estimate_spatial_matrix_memory(0, 4, backend="numpy")


def test_numpy_python_side_peak_is_within_the_model_and_the_model_is_not_vacuous():
    N = 50000
    col = hand_column(N, "numpy")
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=np.full(N, 1e-5), suction_m=np.full(N, 0.1), vegetation_factor=0.5, memory_budget_bytes=BIG)
    hist = sm.spatial_matrix_history(op, wetting_m=0.001, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    state = hand_state(col)
    pool, standing, rate = full(col, 0.01), full(col, 0.01), full(col, 0.0)
    model = sm.estimate_spatial_matrix_memory(2, N, backend="numpy", operation="step").required_bytes
    gc.collect()
    tracemalloc.start()
    try:
        r = sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=BIG)
        _, peak = tracemalloc.get_traced_memory()
        kept = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    assert peak <= model, (peak, model)
    assert peak >= 0.3 * model  # a model that is far above the measurement would hide nothing but also prove nothing
    assert kept >= r.surface_pool_m.nbytes + r.intake_m.nbytes + r.state.water_m.nbytes  # the surviving arrays are the outputs (rows and scratch are released)


def test_numba_heap_blocks_and_python_side_allocations_follow_the_model():
    pytest.importorskip("numba")
    from numba.core.runtime import _nrt_python as nrt

    N = 40000
    col = hand_column(N, "numba")
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=1e-5, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG)
    hist = sm.spatial_matrix_history(op, wetting_m=0.001, cumulative_intake_m=0.0, memory_budget_bytes=BIG)
    state = hand_state(col)
    pool, standing, rate = full(col, 0.01), full(col, 0.01), full(col, 0.0)
    sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=BIG)  # warm
    lcol = cases.column(cases.layers(), n_cells=N, backend="numba")  # the accepted common kernel on the same data: the baseline of NRT blocks per call (array adaptation/boxing)
    lop, lstate, lhist = cases.operator(lcol), cases.state_of(lcol), None
    lhist = cases.history_of(lop, 0.001)
    mi.matrix_step(lop, pool, standing, rate, lstate, lhist, 60.0)
    nrt.memsys_enable_stats()
    a0, f0 = nrt.memsys_get_stats_alloc(), nrt.memsys_get_stats_free()
    sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=BIG)
    a1, f1 = nrt.memsys_get_stats_alloc(), nrt.memsys_get_stats_free()
    mi.matrix_step(lop, pool, standing, rate, lstate, lhist, 60.0)
    a2, f2 = nrt.memsys_get_stats_alloc(), nrt.memsys_get_stats_free()
    nrt.memsys_disable_stats()
    spatial_blocks, common_blocks = a1 - a0, a2 - a1
    assert f1 - f0 == spatial_blocks and f2 - f1 == common_blocks  # everything is released within the call
    assert spatial_blocks - common_blocks == 2  # exactly the two (L,) local rows (sat, cap_tol): 2 L doubles, independent of N
    model = sm.estimate_spatial_matrix_memory(2, N, backend="numba", operation="step")
    assert model.local_bytes == 2 * 2 * 8
    gc.collect()
    tracemalloc.start()
    try:
        r = sm.spatial_matrix_step(op, pool, standing, rate, state, hist, 60.0, memory_budget_bytes=BIG)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= model.required_bytes and peak >= model.output_bytes + model.row_bytes  # outputs + dense rows (+ packet), all Python-visible
    del r


# ---- owner release / caches --------------------------------------------------------------------------------------------------------------
def test_owner_release_and_function_only_caches(backend):
    col = hand_column(50, backend)
    op = hand_op(col)
    hist = hand_hist(op)
    r = step(op, full(col, 0.01), full(col, 0.01), full(col, 0.0), hand_state(col), hist, 60.0)
    refs = [weakref.ref(x) for x in (op, hist, r.history, r.state, r.surface_pool_m)]
    del op, hist, r
    gc.collect()
    assert all(w() is None for w in refs)  # nothing in the module retains an operator, a history, a state or an array
    if backend == "numba":
        from maple_syrup import matrix_infiltration_numba as mn

        assert type(mn._SPATIAL_KERNEL).__name__ == "CPUDispatcher"  # the only cache is the process-wide compiled FUNCTION


# ---- genuine compiled proof -------------------------------------------------------------------------------------------------------------
def test_the_numba_kernel_is_a_genuine_nopython_readonly_explicit_signature_body(monkeypatch):
    numba = pytest.importorskip("numba")
    from maple_syrup import matrix_infiltration_numba as mn

    k = mn.compiled_spatial_kernel()
    assert len(k.signatures) == 1 and k.targetoptions.get("nopython") is True and not k.targetoptions.get("fastmath") and not k._can_compile
    sig = k.signatures[0]
    ro = [i for i, a in enumerate(sig) if isinstance(a, numba.types.Array) and not a.mutable]
    rw = [i for i, a in enumerate(sig) if isinstance(a, numba.types.Array) and a.mutable]
    assert len(sig) == 32 and len(rw) == 6 and len(ro) == 14 and rw == [26, 27, 28, 29, 30, 31]  # only the six fresh outputs are writable
    assert mn.compiled_spatial_kernel() is k and mn.compiled_kernel() is not k  # one function per variant, never per owner or call
    col = hand_column(4, "numba")
    op = hand_op(col)
    state, hist = hand_state(col), hand_hist(op, 0.002)
    ref = step(op, full(col, 0.01), full(col, 0.01), full(col, 0.0), state, hist, 60.0)

    def boom(*a, **kw):
        raise AssertionError("the NumPy body must not run for a numba owner (no fallback)")

    monkeypatch.setattr(mi, "_reference_step_body", boom)
    again = step(op, full(col, 0.01), full(col, 0.01), full(col, 0.0), state, hist, 60.0)
    np.testing.assert_array_equal(again.intake_m, ref.intake_m)


def test_a_failed_or_unavailable_compile_is_typed_and_never_falls_back(monkeypatch):
    pytest.importorskip("numba")
    from maple_syrup import matrix_infiltration_numba as mn

    col = hand_column(3, "numba")
    mn.reset_compiled()
    monkeypatch.setattr(mn, "_build", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("synthetic compiler defect")))
    with pytest.raises(sp.SpatialLayerBackendUnavailableError, match="no NumPy fallback"):
        sm.prepare_spatial_matrix(col, conductivity_m_s=1e-5, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG)
    monkeypatch.undo()
    mn.reset_compiled()
    monkeypatch.setattr(mn, "numba_available", lambda: False)
    with pytest.raises(sp.SpatialLayerBackendUnavailableError):
        sm.prepare_spatial_matrix(col, conductivity_m_s=1e-5, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG)
    monkeypatch.undo()
    mn.reset_compiled()
    assert sm.prepare_spatial_matrix(col, conductivity_m_s=1e-5, suction_m=0.1, vegetation_factor=0.5, memory_budget_bytes=BIG).backend == "numba"
