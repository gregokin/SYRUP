"""Proof that the spatial "numba" evolution is a genuine compiled nopython CPU kernel with no fallback: dispatcher/option/IR introspection, ONE explicit read-only-input signature, the exact set of per-call
Numba heap allocations, no NumPy/other-backend call on the compiled path, typed failure instead of fallback. Author-written."""
from __future__ import annotations

import re

import numpy as np
import pytest
from conftest import BIG, a2, require_numba, run_spatial, sp, ss, step_args

numba = require_numba()
from maple_syrup import subsurface_numba as sn  # noqa: E402


@pytest.fixture(scope="module", params=["common", "spatial"])
def kernel(request):
    """Both compile-time variants of the ONE body."""
    return sn.compiled_kernel() if request.param == "common" else sn.compiled_spatial_kernel()


def test_the_two_variants_are_distinct_dispatchers_of_the_same_signature():
    a, b = sn.compiled_kernel(), sn.compiled_spatial_kernel()
    assert a is not b and a.signatures == b.signatures and a.py_func.__code__.co_code == b.py_func.__code__.co_code  # one body text, compiled twice


def test_the_kernel_is_a_nopython_cpu_dispatcher_without_fastmath_or_object_mode(kernel):
    assert type(kernel).__name__ == "CPUDispatcher"
    opts = kernel.targetoptions
    assert opts["nopython"] is True and opts["fastmath"] is False and opts["boundscheck"] is False and opts["error_model"] == "numpy"
    assert len(kernel.signatures) == 1 and [s.args for s in kernel.nopython_signatures] == list(kernel.signatures)  # ONE specialisation, compiled in nopython mode
    cres = kernel.overloads[kernel.signatures[0]]
    assert cres.objectmode is False and cres.lifted == ()
    ir = cres.library.get_llvm_str()
    assert "expm1" in ir  # the per-interface exchange factor is compiled in
    assert not re.search(r"\b(fast|reassoc|nnan|ninf|nsz|arcp|contract|afn)\b", ir), "no fast-math flag may appear on any floating-point instruction"


def test_every_input_is_a_readonly_array_type_and_every_output_is_writable(kernel):
    sig = kernel.signatures[0]
    arrays = [a for a in sig if isinstance(a, numba.types.Array)]
    readonly, writable = [a for a in arrays if not a.mutable], [a for a in arrays if a.mutable]
    assert len(arrays) == 21 and len(readonly) == 13 and len(writable) == 8  # 13 inputs (state, forcing, mask, 5 maps, 3 interface, share ...) and the 8 fresh outputs
    assert all(a.dtype == numba.types.float64 or a.dtype == numba.types.uint8 for a in arrays) and all(a.layout == "C" for a in arrays)
    assert [i for i, a in enumerate(sig) if isinstance(a, numba.types.Array) and a.mutable] == list(range(len(sig) - 8, len(sig)))  # the writable ones are exactly the last eight arguments


def test_a_writable_and_a_readonly_caller_array_use_the_same_single_specialisation():
    kernel = sn.compiled_spatial_kernel()
    rng = np.random.default_rng(1)
    p = a2.params(rng, 3, 12)
    col = a2.prepare(a2.make_spec(p), backend="numba")
    water = a2.water_kind(rng, p, "random")
    added, pet = step_args(rng, p, np.ones(12, bool))[1:]
    run_spatial(col, water, 3600.0, added=added, pet=pet)  # the state copy is read-only, `added`/`pet` writable
    added.flags.writeable = False
    pet.flags.writeable = False
    run_spatial(col, water, 3600.0, added=added, pet=pet)
    assert len(kernel.signatures) == 1


def _nrt():
    from numba.core.runtime import _nrt_python

    _nrt_python.memsys_enable_stats()
    from numba.core.runtime import rtsys

    return rtsys


@pytest.mark.parametrize("mode,locals_", [("constant", 7), ("tietjen_donor_depth", 8)])
def test_the_per_call_numba_heap_is_exactly_the_declared_local_arrays_and_is_released(mode, locals_):
    kernel = sn.compiled_spatial_kernel()
    """alloc == free after the call; the allocations are the 21 argument wrappers plus the kernel's own local arrays: dz, res, fc, sat, cap_tol, 1/pair, downward factor and (donor mode only, otherwise
    aliased) the upward factor -- bounded by `local_scratch_doubles(L)` doubles and independent of N."""
    rt = _nrt()
    rng = np.random.default_rng(2)
    L, N = 4, 64
    p = a2.params(rng, L, N, mode=mode)
    col = a2.prepare(a2.make_spec(p, mode=mode), backend="numba")
    water = a2.water_kind(rng, p, "random")
    added, pet = step_args(rng, p, np.ones(N, bool))[1:]
    run_spatial(col, water, 3600.0, added=added, pet=pet)  # warm
    s0 = rt.get_allocation_stats()
    run_spatial(col, water, 3600.0, added=added, pet=pet)
    s1 = rt.get_allocation_stats()
    wrappers = sum(isinstance(a, numba.types.Array) for a in kernel.signatures[0])
    assert s1.alloc - s0.alloc == s1.free - s0.free == wrappers + locals_
    assert sn.local_scratch_doubles(L) >= 5 * L + 3 * (L - 1)  # the declared bound covers the five per-layer and three per-interface arrays
    # independence of N: ten times the cells allocate the same number of NRT blocks
    p2 = a2.params(rng, L, 640, mode=mode)
    col2 = a2.prepare(a2.make_spec(p2, mode=mode), backend="numba")
    w2 = a2.water_kind(rng, p2, "random")
    a2_, pet2 = step_args(rng, p2, np.ones(640, bool))[1:]
    run_spatial(col2, w2, 3600.0, added=a2_, pet=pet2)
    t0 = rt.get_allocation_stats()
    run_spatial(col2, w2, 3600.0, added=a2_, pet=pet2)
    t1 = rt.get_allocation_stats()
    assert t1.alloc - t0.alloc == s1.alloc - s0.alloc


@pytest.mark.parametrize("mode,factors", [("constant", 1), ("tietjen_donor_depth", 2)])
@pytest.mark.parametrize("L", [1, 2, 6, 40])
def test_the_common_variant_allocates_only_the_pre_a3_factor_arrays_whatever_the_stack_height(mode, factors, L):
    """The common column's bound values ALIAS the cached parameter arrays: the per-call Numba heap is exactly the one (constant) or two (donor) per-interface factor arrays the pre-A3 kernel allocated
    and the frozen `estimate_step_memory` counts (`factor_scratch_doubles`), for every L -- including a tall stack with a single cell, where any O(L) local copy would exceed the old model."""
    rt = _nrt()
    edges = np.concatenate([[0.0], np.cumsum(np.full(L, 0.05))])
    kw = dict(diffusivity_m2_s=np.full(max(L - 1, 1) if L > 1 else 1, 2e-7)) if mode == "constant" else dict(interface_conductivity_m_s=np.full(max(L - 1, 1), 1e-6), diffusion_coefficient=0.25)
    if L == 1:  # a single layer accepts a scalar (or an empty) interface input
        kw = dict(diffusivity_m2_s=2e-7) if mode == "constant" else dict(interface_conductivity_m_s=1e-6, diffusion_coefficient=0.25)
    lay = ss.soil_layers(edges, theta_residual=np.full(L, 0.02), theta_field_capacity=np.full(L, 0.2), theta_saturation=np.full(L, 0.4), theta_evaporation_stress=0.5, diffusion_mode=mode, **kw)
    col = ss.prepare_column(lay, np.ones(1, bool), 4.0, backend="numba")
    w = np.full((L, 1), 0.05)
    add = np.zeros((L, 1))
    pet = np.full(1, 1e-8)
    col.runner(col, w, add, pet, 3600.0)  # warm
    s0 = rt.get_allocation_stats()
    col.runner(col, w, add, pet, 3600.0)
    s1 = rt.get_allocation_stats()
    wrappers = sum(isinstance(a, numba.types.Array) for a in sn.compiled_kernel().signatures[0])
    assert s1.alloc - s0.alloc == s1.free - s0.free == wrappers + factors
    assert col.backend_info["kernel_local_scratch_host_bytes"] == col.backend_info["factor_scratch_host_bytes"] == ss.factor_scratch_doubles(L, mode) * 8 == sn.local_scratch_doubles(L, spatial=False, diffusion_mode=mode) * 8


def test_the_compiled_path_never_calls_the_numpy_reference_body(monkeypatch):
    rng = np.random.default_rng(3)
    p = a2.params(rng, 3, 10)
    col = a2.prepare(a2.make_spec(p), backend="numba")
    water = a2.water_kind(rng, p, "random")
    added, pet = step_args(rng, p, np.ones(10, bool))[1:]

    def boom(*a, **k):
        raise AssertionError("the NumPy reference body must not run for a numba-prepared owner")

    monkeypatch.setattr(ss, "_reference_step_body", boom)
    monkeypatch.setattr(sp, "_SpatialStepParams", boom)  # the lazy NumPy parameter reader is not used either
    s = run_spatial(col, water, 3600.0, added=added, pet=pet)
    assert s.state.t_s == 3600.0
    r = run_spatial(col, water, 86400.0, pet=pet, recharge=1e-3)
    assert r.state.t_s == 86400.0


def test_a_failing_compiled_launch_is_a_typed_error_not_a_numpy_fallback(monkeypatch):
    rng = np.random.default_rng(4)
    p = a2.params(rng, 3, 10)
    col = a2.prepare(a2.make_spec(p), backend="numba")
    water = a2.water_kind(rng, p, "random")
    added, pet = step_args(rng, p, np.ones(10, bool))[1:]
    state = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
    called = []

    def broken():
        raise ss.SubsurfaceBackendUnavailableError("numba is unavailable (test double)")

    monkeypatch.setattr(sn, "compiled_spatial_kernel", broken)
    monkeypatch.setattr(ss, "_reference_step_body", lambda *a, **k: called.append("numpy"))
    with pytest.raises(ss.SubsurfaceBackendUnavailableError):
        sp.spatial_step(col, state, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    assert called == []
    with pytest.raises(sp.SpatialLayerBackendUnavailableError):  # preparation of another numba owner reports the same typed failure
        a2.prepare(a2.make_spec(p), backend="numba")


def test_preparation_compiles_and_a_step_afterwards_does_not(monkeypatch):
    """Startup sits at preparation: after `prepare_spatial_column(backend="numba")` the kernel exists and a first step triggers no new compilation (the dispatcher keeps its single signature)."""
    sn.reset_compiled()
    try:
        assert sn._SPATIAL_KERNEL is None and sn._KERNEL is None
        rng = np.random.default_rng(5)
        p = a2.params(rng, 2, 8)
        col = a2.prepare(a2.make_spec(p), backend="numba")
        kernel = sn.compiled_spatial_kernel()
        assert len(kernel.signatures) == 1  # the spatial variant was compiled by the preparation
        assert sn._KERNEL is None  # ... and a spatial preparation does not compile the common variant
        water = a2.water_kind(rng, p, "random")
        added, pet = step_args(rng, p, np.ones(8, bool))[1:]
        run_spatial(col, water, 3600.0, added=added, pet=pet)
        assert sn.compiled_spatial_kernel() is kernel and len(kernel.signatures) == 1 and sn._KERNEL is None
    finally:
        sn.reset_compiled()
        sn.compiled_kernel()
        sn.compiled_spatial_kernel()
