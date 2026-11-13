"""Regressions for the five defects confirmed by Root's S1 review (S1 correction01). Author-written; not an independent oracle.

1 CUDA flag diagnostics without an unsupported reduction (device definitions in test_cuda_definitions.py);
2 failure-path and valid-path host memory measured WARM (compiled backend already built) against the modelled requirement;
3 type-derived non-finite quantities raise typed errors, never a raw ValueError and never a vacuous comparison;
4 signed-zero / subnormal bit preservation for unchanged geometry on every backend;
5 separate host/device admission fields (estimate level here, CUDA calls in the device definitions).
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from conftest import arrays, digest, random_columns, run

from maple_syrup import soil_water_remap as sr


def ident(L=1, C=1, w=0.5):
    e = np.cumsum(np.ones((L + 1, C)), axis=0) - 1.0
    return [e, e.copy(), np.full((L, C), w), np.zeros(C), np.zeros(C)]


# ---- 4 identity bit preservation ---------------------------------------------------------------------------------------------
def test_signed_zero_water_is_preserved_in_an_unchanged_column(cpu_backend):
    a = ident(L=1)
    a[2][0, 0] = -0.0
    out = run(*a, backend=cpu_backend)
    assert out.water_m.view(np.uint64)[0, 0] == a[2].view(np.uint64)[0, 0] == 0x8000000000000000
    for f in (out.removed_above_m, out.removed_below_m, out.added_above_m, out.added_below_m):
        assert f.view(np.uint64)[0] == 0  # exactly +0


def test_subnormals_and_signed_zeros_in_multilayer_identity_and_mixed_columns(cpu_backend):
    rng = np.random.default_rng(3)
    L, C = 4, 12
    eo = np.cumsum(rng.uniform(0.1, 1.0, size=(L + 1, C)), axis=0) - 0.2
    en = eo.copy()
    W = (eo[1:] - eo[:-1]) * rng.uniform(0, 1, size=(L, C))
    W[1, :] = -0.0
    W[2, :6] = 5e-324  # smallest subnormal
    en[:, 6:] += 0.125  # columns 6.. change geometry; 0..5 are unchanged
    res = run(eo, en, W, rng.uniform(0, 1, C), rng.uniform(0, 1, C), cpu_backend)
    np.testing.assert_array_equal(res.water_m[:, :6].view(np.uint64), W[:, :6].view(np.uint64))
    assert not np.any(res.removed_above_m[:6]) and not np.any(res.added_below_m[:6])
    assert np.any(res.removed_above_m[6:] > 0) and np.any(res.added_below_m[6:] > 0)  # the columns shifted DOWN still remap (top slab leaves, bottom slab enters)


def test_equal_edges_with_a_signed_zero_edge_are_still_unchanged_geometry(cpu_backend):
    a = ident(L=1)
    a[0][0, 0], a[1][0, 0] = 0.0, -0.0  # numerically equal edges
    a[2][0, 0] = 0.25
    out = run(*a, backend=cpu_backend)
    assert out.water_m[0, 0] == 0.25 and out.removed_above_m[0] == 0.0 and out.added_above_m[0] == 0.0


# ---- 3 typed finite errors ----------------------------------------------------------------------------------------------------
def test_finite_area_whose_global_scale_overflows_is_a_typed_error_and_preserves_inputs(cpu_backend):
    a = ident(L=1)
    before = digest(*a)
    with pytest.raises(sr.SoilRemapClosureError):
        run(*a, backend=cpu_backend, area=1e308)
    assert digest(*a) == before


def test_column_scale_overflow_is_not_a_vacuous_pass(cpu_backend):
    # valid finite widths whose derived scale (4e308) overflows: the column bound would be inf; it must be refused, never accepted by `<= inf`
    eo = np.array([[-1e308], [0.0], [1e308]])
    a = [eo, eo.copy(), np.zeros((2, 1)), np.zeros(1), np.zeros(1)]
    before = digest(*a)
    with pytest.raises(sr.SoilRemapClosureError) as info:
        run(*a, backend=cpu_backend)
    assert info.value.bit == sr.FLAG_CLOSURE
    assert digest(*a) == before


@pytest.mark.parametrize("area", [1e308, 1.7e308])
def test_huge_finite_areas_are_typed(cpu_backend, area):
    with pytest.raises(sr.SoilRemapError):
        run(*ident(L=2, C=3), backend=cpu_backend, area=area)


# ---- 2 memory: warm, failure path and valid path --------------------------------------------------------------------------------
@pytest.mark.parametrize("L", [1, 3])
@pytest.mark.parametrize("mode", ["invalid_all", "valid"])
def test_warm_peak_is_within_the_modelled_requirement(cpu_backend, L, mode):
    C = 100_000
    a = list(arrays(*random_columns(np.random.default_rng(L), L, C)))
    run(*[x[:, :8] if x.ndim == 2 else x[:8] for x in a], backend=cpu_backend)  # warm: compilation/imports outside the measurement (the caller's inputs already exist)
    if mode == "invalid_all":
        a[1][1] = a[1][0]  # every column has a collapsed new interval
    est = sr.estimate_remap_memory(L, C, backend=cpu_backend)
    tracemalloc.start()
    try:
        run(*a, backend=cpu_backend)
    except sr.SoilRemapValidationError:
        assert mode == "invalid_all"
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= est.required_bytes, (cpu_backend, L, mode, peak, est.required_bytes)


def test_flag_search_is_chunked_and_finds_a_late_first_column(cpu_backend):
    C = sr.FLAG_SEARCH_CHUNK * 2 + 123
    a = list(arrays(*random_columns(np.random.default_rng(5), 2, C)))
    a[2][0, C - 5] = -1.0
    a[1][1, C - 2] = a[1][0, C - 2]
    with pytest.raises(sr.SoilRemapValidationError) as info:
        run(*a, backend=cpu_backend)
    assert info.value.first_column == C - 5 and info.value.n_flagged == 2 and info.value.bit == sr.FLAG_WATER


# ---- 5 host/device admission fields ------------------------------------------------------------------------------------------------
def test_cuda_estimate_separates_host_and_device_sides():
    a = sr.estimate_remap_memory(3, 1000, backend="cuda", reserve_bytes=7, host_reserve_bytes=11)
    b = sr.estimate_remap_memory(3, 100_000, backend="cuda", reserve_bytes=7, host_reserve_bytes=11)
    assert a.host_required_bytes == b.host_required_bytes  # the host side does not scale with the domain: packet + allowance + host reserve only
    assert a.host_required_bytes == sr.PACKET_BYTES + sr.HOST_OBJECT_ALLOWANCE_BYTES + 11
    assert a.device_required_bytes == a.output_bytes + sr.DEVICE_DIAG_BYTES + 5 * 8 * sr.cuda_block_count(1000) + 7 and a.required_bytes == a.device_required_bytes
    assert b.device_required_bytes > 50 * a.device_required_bytes  # the device side scales with C (the host side above does not)
    assert a.terms["no_LxLxC_no_LCK_no_whole_bed"] and "no library reduction workspace" in a.terms["excluded"]
    cpu = sr.estimate_remap_memory(3, 1000, backend="numba", reserve_bytes=7)
    assert cpu.device_required_bytes == 0 and cpu.host_required_bytes == cpu.required_bytes


def test_host_arguments_are_refused_for_the_cpu_backends(cpu_backend):
    for kw in ({"host_reserve_bytes": 1}, {"host_memory_budget_bytes": 10**9}, {"host_available_bytes_fn": lambda: 10**9}):
        with pytest.raises(sr.SoilRemapError):
            run(*ident(), backend=cpu_backend, **kw)
    with pytest.raises(sr.SoilRemapError):
        sr.estimate_remap_memory(2, 5, backend="numpy", host_reserve_bytes=1)


@pytest.mark.parametrize("kw", [{"host_reserve_bytes": -1}, {"host_reserve_bytes": 1.5}, {"host_memory_budget_bytes": -1}, {"host_memory_budget_bytes": True},
                                {"host_available_bytes_fn": 5}])
def test_host_argument_type_refusals(kw):
    with pytest.raises(sr.SoilRemapError):
        run(*ident(), backend="numpy", **kw)
