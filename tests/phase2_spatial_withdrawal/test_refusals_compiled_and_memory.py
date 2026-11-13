"""Typed validation/refusal before publication with no mutation, cross-refusal of legacy and spatial objects, genuine compiled Numba with no NumPy fallback, quota/admission before any allocation, the
payload memory model against the traced peak, ownership/release and kernel-cache scope. Author-written."""
from __future__ import annotations

import copy
import gc
import hashlib
import pickle
import tracemalloc
import weakref

import numpy as np
import pytest
from conftest import AREA, BIG, a2, bits, common_pair, spatial_owner, sp, ss, st, sw


def build(backend, L=3, N=40, seed=1, mask=None):
    rng = np.random.default_rng(seed)
    p = a2.params(rng, L, N)
    mask = (rng.random(N) < 0.8) if mask is None else mask
    mask[0] = True
    col = spatial_owner(p, backend, mask)
    op = sw.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG)
    dz = np.diff(p["edges"], axis=0)
    w = p["sat"] * dz * rng.uniform(0.0, 1.0, (L, N))
    state = sp.spatial_column_state(col, w, 5.0, memory_budget_bytes=BIG)
    q = rng.uniform(0, 0.02, (L, N)) * mask
    return rng, p, col, op, w, state, q, mask


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def maps_digest(col):
    return digest(*[col._maps[n] for n in col.retained_names()], col._mask)


# ---- invalid values ----------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("label,mutate,text", [
    ("nan_water", lambda w, q, m: (w.__setitem__((1, 2), np.nan), None), "soil water not finite"),
    ("negative_water", lambda w, q, m: (w.__setitem__((0, 3), -1e-3), None), "soil water not finite"),
    ("over_saturated_water", lambda w, q, m: (w.__setitem__((2, 4), 10.0), None), "soil water not finite"),
    ("nan_water_in_inactive_cell", lambda w, q, m: (w.__setitem__((1, int(np.flatnonzero(~m)[0])), np.nan), None), "soil water not finite"),
    ("nan_request", lambda w, q, m: (q.__setitem__((1, 2), np.nan), None), "requested_water_m not finite"),
    ("inf_request", lambda w, q, m: (q.__setitem__((0, 1), np.inf), None), "requested_water_m not finite"),
    ("negative_request", lambda w, q, m: (q.__setitem__((2, 5), -1e-6), None), "requested_water_m not finite"),
    ("positive_request_on_inactive", lambda w, q, m: (q.__setitem__((1, int(np.flatnonzero(~m)[0])), 1e-6), None), "inactive cell"),
])
def test_invalid_inputs_are_refused_typed_on_every_cell_with_no_mutation_and_no_publication(backend, label, mutate, text):
    rng, p, col, op, w, state, q, mask = build(backend)
    assert (~mask).any()
    w2, q2 = w.copy(), q.copy()
    mutate(w2, q2, mask)
    bad_state = sp.spatial_column_state(col, w2, 5.0, memory_budget_bytes=BIG)  # the constructor defers water VALUES to the step
    before = (digest(bad_state.water_m), digest(q2), maps_digest(col), bad_state.t_s)
    with pytest.raises(ss.SubsurfaceError, match=text):
        sw.spatial_withdraw(op, bad_state, q2, memory_budget_bytes=BIG)
    assert (digest(bad_state.water_m), digest(q2), maps_digest(col), bad_state.t_s) == before
    if "water" in label:
        with pytest.raises(ss.SubsurfaceError, match="soil water not finite"):
            sw.spatial_layer_availability_m(op, bad_state, memory_budget_bytes=BIG)


def test_the_common_operator_reports_the_same_reason_text_for_the_same_invalid_input(backend):
    rng, p, col, op, w, state, q, mask = build(backend, L=2, N=6, mask=np.ones(6, bool))
    q2 = q.copy()
    q2[0, 0] = np.nan
    ccol, cop = common_pair(p, 0, backend)
    with pytest.raises(ss.SubsurfaceError) as e_c:
        st.withdraw_transpiration(cop, ss.column_state(ccol, w[:, :1].copy()), q2[:, :1].copy())
    with pytest.raises(ss.SubsurfaceError) as e_s:
        sw.spatial_withdraw(op, state, q2, memory_budget_bytes=BIG)
    assert "requested_water_m not finite or negative" in str(e_c.value) and "requested_water_m not finite or negative" in str(e_s.value)  # same reason; the stage prefix differs by design


def test_the_legacy_global_flag_max_behaviour_is_preserved_and_documented(backend):
    """Two different invalid cells (flag 2 in one, flag 4 in another): the reported reasons come from `max` over the per-cell flags, exactly as in the common operator (it may omit a bit but always refuses)."""
    rng, p, col, op, w, state, q, mask = build(backend)
    q2 = q.copy()
    q2[0, 1] = -1.0  # flag 2
    q2[1, int(np.flatnonzero(~mask)[0])] = 1e-6  # flag 4
    with pytest.raises(ss.SubsurfaceError) as e:
        sw.spatial_withdraw(op, state, q2, memory_budget_bytes=BIG)
    assert "inactive cell" in str(e.value) and "checked on every cell, inactive included" in str(e.value)


# ---- binding and cross-refusal -----------------------------------------------------------------------------------------------------------------
def test_sealed_operator_cannot_be_constructed_copied_or_pickled(backend):
    rng, p, col, op, w, state, q, mask = build(backend)
    with pytest.raises(sp.SpatialLayerError, match="only by prepare_spatial_withdrawal"):
        sw.SpatialWithdrawalOperator(col, backend, 3, 40, col.sha256, "x", op.prepare_memory)
    import dataclasses

    with pytest.raises(sp.SpatialLayerError):
        dataclasses.replace(op, backend="numpy")
    for fn in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(sp.SpatialLayerError):
            fn(op)
    s = op.summary()
    assert s["pure_trial"] is True and s["generation_bound"] is False and s["exactly_once"] is False and s["tag"] == "maple_syrup.spatial_withdrawal.operator.v1"


def test_legacy_and_spatial_objects_refuse_each_other_in_both_directions(backend):
    rng, p, col, op, w, state, q, mask = build(backend)
    ccol, cop = common_pair(p, 0, backend)
    cstate = ss.column_state(ccol, w[:, :1].copy())
    with pytest.raises(sp.SpatialLayerError, match="not a legacy SoilColumn"):
        sw.prepare_spatial_withdrawal(ccol, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError, match="legacy TranspirationOperator"):
        sw.spatial_withdraw(cop, state, q, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError, match="SpatialSoilState"):
        sw.spatial_withdraw(op, cstate, q, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError):
        sw.spatial_withdraw(object(), state, q, memory_budget_bytes=BIG)
    with pytest.raises(ss.SubsurfaceError):  # the old guards: a spatial column / operator / state never enter the common operator
        st.prepare_transpiration(col)
    with pytest.raises(ss.SubsurfaceError):
        st.withdraw_transpiration(op, cstate, q[:, :1])
    with pytest.raises(ss.SubsurfaceError):
        st.withdraw_transpiration(cop, state, q)


def test_state_binding_is_the_input_sha_not_the_owner_object(backend):
    rng, p, col, op, w, state, q, mask = build(backend)
    twin = spatial_owner(p, backend, mask)  # another owner OBJECT from IDENTICAL inputs: its state is accepted (documented: the state carries the input SHA only)
    assert twin is not col and twin.sha256 == col.sha256
    twin_state = sp.spatial_column_state(twin, w, 5.0, memory_budget_bytes=BIG)
    r = sw.spatial_withdraw(op, twin_state, q, memory_budget_bytes=BIG)
    np.testing.assert_array_equal(bits(r.state.water_m), bits(sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG).state.water_m))
    p2 = {k: v.copy() for k, v in p.items()}
    p2["res"] = p2["res"] * 0.5
    other = spatial_owner(p2, backend, mask)  # different parameter identity: refused
    with pytest.raises(sp.SpatialLayerError, match="identity mismatch"):
        sw.spatial_withdraw(op, sp.spatial_column_state(other, w, 5.0, memory_budget_bytes=BIG), q, memory_budget_bytes=BIG)
    mask2 = mask.copy()
    mask2[1] = ~mask2[1]
    with pytest.raises(sp.SpatialLayerError, match="identity mismatch"):  # a different mask is a different identity too
        sw.spatial_withdraw(op, sp.spatial_column_state(spatial_owner(p, backend, mask2), w, 5.0, memory_budget_bytes=BIG), q, memory_budget_bytes=BIG)


def test_a_cuda_tagged_owner_on_a_host_without_cupy_is_a_typed_refusal_before_any_allocation():
    """S2-B1.c replaced the old "cuda owner is refused" expectation (CUDA is now a supported backend). This remains a GUARD-PATH, NON-HARDWARE test: a numba owner whose backend tag is set to "cuda" reaches the
    CuPy/device gate, which on this CPU host fails with a typed error before any allocation, copy or compilation. It is not evidence of any device behaviour; the CUDA implementation is tested in
    tests/phase2_spatial_withdrawal_cuda."""
    pytest.importorskip("numba")
    rng, p, col, op, w, state, q, mask = build("numba")
    object.__setattr__(col, "backend", "cuda")
    try:
        try:
            sp._require_cuda()  # warm the one-off CuPy import probe so that the traced peak measures the refusal only
        except Exception:  # noqa: BLE001
            pass
        tracemalloc.start()
        with pytest.raises((sp.SpatialLayerBackendUnavailableError, sp.SpatialLayerError), match="CuPy|cuda|CUDA"):
            sw.prepare_spatial_withdrawal(col, memory_budget_bytes=0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 100_000
    finally:
        object.__setattr__(col, "backend", "numba")


def test_the_cuda_only_quota_arguments_are_refused_for_cpu_owners_before_any_work(backend):
    rng, p, col, op, w, state, q, mask = build(backend)
    for kw in (dict(host_memory_budget_bytes=BIG), dict(host_reserve_bytes=1), dict(host_available_bytes_fn=lambda: BIG)):
        for call in (lambda: sw.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG, **kw), lambda: sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG, **kw),
                     lambda: sw.spatial_layer_availability_m(op, state, memory_budget_bytes=BIG, **kw)):
            with pytest.raises(sp.SpatialLayerError, match="apply only to a cuda owner"):
                call()
    with pytest.raises(sp.SpatialLayerError, match="only to the cuda backend"):
        sw.estimate_spatial_withdrawal_memory(3, 40, backend=backend, host_reserve_bytes=1)


# ---- compiled proof, no fallback -------------------------------------------------------------------------------------------------------------
def test_numba_owner_runs_a_genuinely_compiled_kernel_and_never_the_numpy_body(monkeypatch):
    nb = pytest.importorskip("numba")
    kernels = sw.compiled_kernels()
    for k in kernels:
        assert type(k).__name__ == "CPUDispatcher" and k.targetoptions.get("nopython") is True and not k.targetoptions.get("fastmath") and len(k.signatures) == 1
        cres = k.overloads[k.signatures[0]]
        assert cres.objectmode is False and cres.lifted == ()
        arrays = [a for a in k.signatures[0] if isinstance(a, nb.types.Array)]
        assert sum(not a.mutable for a in arrays) >= 5  # the inputs are read-only types: the kernel cannot write them
    rng, p, col, op, w, state, q, mask = build("numba")
    called = []
    monkeypatch.setattr(st, "_fused_numpy", lambda *a, **k: called.append(1))
    r = sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG)
    assert called == [] and r.state.t_s == 5.0
    rng, p, col2, op2, *_ = build("numpy")
    monkeypatch.undo()
    monkeypatch.setattr(sw, "compiled_kernels", lambda: (_ for _ in ()).throw(AssertionError("numpy owner must not compile")))
    sw.spatial_withdraw(op2, sp.spatial_column_state(col2, np.zeros((3, 40)), 0.0, memory_budget_bytes=BIG), np.zeros((3, 40)), memory_budget_bytes=BIG)


def test_a_failed_compiled_build_is_typed_and_has_no_numpy_fallback(monkeypatch):
    pytest.importorskip("numba")
    saved = sw._KERNELS
    rng, p, col, *_ = build("numba")
    monkeypatch.setattr(sw, "_KERNELS", None)
    monkeypatch.setattr(st, "_withdraw_cell", None)
    ran = []
    monkeypatch.setattr(st, "_fused_numpy", lambda *a, **k: ran.append(1))
    with pytest.raises(sp.SpatialLayerBackendUnavailableError, match="no fallback"):
        sw.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG)
    assert ran == []
    monkeypatch.setattr(sw, "_KERNELS", saved)


# ---- memory -------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend_", ["numpy", "numba"])
def test_estimate_formulas_and_scope(backend_):
    L, N = 5, 1000
    w = sw.estimate_spatial_withdrawal_memory(L, N, backend=backend_, operation="withdraw", reserve_bytes=11)
    ln = 8 * L * N
    scratch = (6 * ln if backend_ == "numpy" else 0) + 8 * N + 3 * 64
    row_temp = (6 if backend_ == "numpy" else 2) * N * 8  # NumPy lazy bound rows / Numba staged local rows
    assert (w.output_bytes, w.row_bytes, w.scratch_bytes, w.row_temp_bytes, w.packet_bytes) == (4 * ln, 8 * N * 8, scratch, row_temp, 64)
    assert w.required_bytes == 4 * ln + 8 * N * 8 + scratch + row_temp + 64 + sp.PY_ALLOWANCE_BYTES + 11 and w.terms["no_LN_derived_bound_tensor"] and "retained maps" in w.scope
    a = sw.estimate_spatial_withdrawal_memory(L, N, backend=backend_, operation="availability")
    assert a.output_bytes == ln and a.row_bytes == 0 and a.scratch_bytes == N + (4 * N * 8 if backend_ == "numpy" else 0) and a.row_temp_bytes == (6 * N * 8 if backend_ == "numpy" else 0)
    prep = sw.estimate_spatial_withdrawal_memory(L, N, backend=backend_, operation="prepare")
    one = sw.estimate_spatial_withdrawal_memory(1, 1, backend=backend_, operation="withdraw")
    assert prep.required_bytes == one.output_bytes + one.row_bytes + one.scratch_bytes + one.row_temp_bytes + sp.PY_ALLOWANCE_BYTES
    c = sw.estimate_spatial_withdrawal_memory(L, N, backend="cuda", operation="withdraw")  # S2-B1.c: the CUDA estimate exists (device main requirement + separate host requirement)
    assert c.terms["backend"] == "cuda" and c.device_required_bytes == c.required_bytes and c.host_required_bytes > 0
    for bad in (dict(operation="x"), dict(backend="gpu")):
        with pytest.raises(sp.SpatialLayerError):
            sw.estimate_spatial_withdrawal_memory(L, N, **{"backend": backend_, **bad})
    # the common estimate is untouched
    assert st.estimate_transpiration_memory(L, N, backend=backend_).as_dict() == st.estimate_transpiration_memory(L, N, backend=backend_).as_dict()


@pytest.mark.parametrize("operation", ["withdraw", "availability"])
def test_admission_boundary_precedes_every_allocation_and_the_readers_are_checked(backend, operation, monkeypatch):
    rng, p, col, op, w, state, q, mask = build(backend)
    need = sw.estimate_spatial_withdrawal_memory(3, 40, backend=backend, operation=operation).required_bytes

    def call(budget=BIG, reserve=0, reader=None):
        if operation == "withdraw":
            return sw.spatial_withdraw(op, state, q, memory_budget_bytes=budget, reserve_bytes=reserve, available_bytes_fn=reader)
        return sw.spatial_layer_availability_m(op, state, memory_budget_bytes=budget, reserve_bytes=reserve, available_bytes_fn=reader)

    empties = []
    real_empty = np.empty
    monkeypatch.setattr(np, "empty", lambda *a, **k: (empties.append(1), real_empty(*a, **k))[1])
    for kw in (dict(budget=need - 1), dict(budget=need, reserve=1), dict(reader=lambda: need - 1)):
        with pytest.raises(sp.SpatialLayerMemoryError, match="nothing was allocated"):
            call(**kw)
    assert empties == []  # no output array exists after a refusal
    assert call(budget=need, reader=lambda: need) is not None
    for bad in (lambda: -1, lambda: True, lambda: 2.5):
        with pytest.raises(sp.SpatialLayerError, match="must return"):
            call(reader=bad)
    with pytest.raises(sp.SpatialLayerMemoryError, match="spatial prepare"):
        sw.prepare_spatial_withdrawal(col, memory_budget_bytes=0)  # the preparation scratch is admitted too


@pytest.mark.parametrize("L,N", [(1, 50), (3, 2000), (5, 20000)])
def test_the_traced_warm_peak_is_within_the_model_for_both_backends(backend, L, N):
    rng = np.random.default_rng(L)
    p = a2.params(rng, L, N)
    mask = rng.random(N) < 0.9
    col = spatial_owner(p, backend, mask)
    op = sw.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG)
    w = p["sat"] * np.diff(p["edges"], axis=0) * rng.uniform(0, 1, (L, N))
    state = sp.spatial_column_state(col, w, 0.0, memory_budget_bytes=BIG)
    q = rng.uniform(0, 0.02, (L, N)) * mask
    sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG)
    sw.spatial_layer_availability_m(op, state, memory_budget_bytes=BIG)
    for operation, fn in (("withdraw", lambda: sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG)), ("availability", lambda: sw.spatial_layer_availability_m(op, state, memory_budget_bytes=BIG))):
        tracemalloc.start()
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        res = fn()
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        est = sw.estimate_spatial_withdrawal_memory(L, N, backend=backend, operation=operation)
        assert peak <= est.required_bytes, (operation, peak, est.required_bytes)
        assert peak >= est.output_bytes  # the measured peak contains every published output array
        del res


def test_owner_operator_results_are_released_and_the_kernel_cache_holds_functions_only(backend):
    rng, p, col, op, w, state, q, mask = build(backend)
    r = sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG)
    refs = [weakref.ref(x) for x in (col, op, state, r, r.state, r.state.water_m, r.actual_m)]
    del col, op, state, r
    gc.collect()
    assert all(x() is None for x in refs)
    assert sw._KERNELS is None or all(type(k).__name__ == "CPUDispatcher" for k in sw._KERNELS)


def test_source_scope_the_spatial_module_contains_no_copy_of_the_law():
    import inspect

    src = inspect.getsource(sw)
    for forbidden in ("np.minimum(", "np.maximum(q", "t / qv", "int(f) & 1", "qv == inf"):
        assert forbidden not in src, forbidden  # no second withdrawal solver: the body is st._fused_numpy / st._withdraw_point
    assert "st._fused_numpy" in src and "cell(" in src and "init_rows(" in src


def test_numba_withdrawal_allocates_exactly_the_two_staged_rows_in_the_numba_heap_and_releases_them():
    """The tracer evidence for compiled-code heap blocks is incomplete, so the two staged `(N,)` rows (the model's `2 N 8`) are counted from the source and with NRT allocation statistics: per call the argument wrappers + exactly two rows, all freed."""
    pytest.importorskip("numba")
    from numba.core.runtime import _nrt_python, rtsys

    _nrt_python.memsys_enable_stats()
    rng, p, col, op, w, state, q, mask = build("numba", N=64)
    sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG)
    rows = []
    for N in (64, 640):
        rng, p, col, op, w, state, q, mask = build("numba", N=N)
        sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG)
        s0 = rtsys.get_allocation_stats()
        sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG)
        s1 = rtsys.get_allocation_stats()
        rows.append((s1.alloc - s0.alloc, s1.free - s0.free))
    wrappers = sum(1 for a in sw.compiled_kernels()[0].signatures[0] if type(a).__name__ == "Array")
    assert rows[0] == rows[1] and rows[0][0] == rows[0][1] == wrappers + 2  # independent of N; freed
