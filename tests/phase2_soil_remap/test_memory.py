"""Memory admission and the modelled estimate (host lanes). Author-written."""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from conftest import BIG, arrays, random_columns, run

from maple_syrup import soil_water_remap as sr


def test_estimate_terms_scale_linearly_in_columns_and_have_no_LxL_or_LCK_terms():
    a = sr.estimate_remap_memory(3, 1000, backend="numpy")
    b = sr.estimate_remap_memory(3, 2000, backend="numpy")
    fixed = a.terms["host_object_and_flag_search_allowance"]
    assert b.required_bytes - fixed == 2 * (a.required_bytes - fixed)  # the only non-linear term is the declared fixed host allowance
    assert a.output_bytes == 8 * 3 * 1000 + 4 * 8 * 1000
    assert a.input_bytes_excluded == 8 * (4 * 1000 * 2 + 3 * 1000 + 2 * 1000)
    assert a.terms["no_LxLxC_no_LCK_no_whole_bed"] is True
    # doubling L doubles only the (L, C) output term; the scratch is independent of L
    c, d = sr.estimate_remap_memory(3, 500, backend="numba"), sr.estimate_remap_memory(6, 500, backend="numba")
    assert d.scratch_bytes == c.scratch_bytes and d.output_bytes - c.output_bytes == 8 * 3 * 500
    z = sr.estimate_remap_memory(2, 0, backend="numpy")
    assert z.required_bytes == z.terms["host_object_and_flag_search_allowance"] and z.input_bytes_excluded == 0
    assert sr.estimate_remap_memory(2, 10, backend="cuda", reserve_bytes=77).reserve_bytes == 77


@pytest.mark.parametrize("bad", [(0, 3), (2, -1), (True, 3), (2, 1.5)])
def test_estimate_refuses_bad_shapes(bad):
    with pytest.raises(sr.SoilRemapError):
        sr.estimate_remap_memory(*bad)


def test_budget_and_availability_refuse_before_allocation(cpu_backend):
    eo, en, W, up, lo = arrays(*random_columns(np.random.default_rng(3), 3, 200))
    need = sr.estimate_remap_memory(3, 200, backend=cpu_backend).required_bytes
    with pytest.raises(sr.SoilRemapMemoryError):
        run(eo, en, W, up, lo, cpu_backend, budget=need - 1)
    with pytest.raises(sr.SoilRemapMemoryError):
        run(eo, en, W, up, lo, cpu_backend, budget=need + 10, reserve_bytes=11)  # the reserve is inside the requirement
    with pytest.raises(sr.SoilRemapMemoryError):
        run(eo, en, W, up, lo, cpu_backend, available_bytes_fn=lambda: need - 1)
    run(eo, en, W, up, lo, cpu_backend, budget=need, available_bytes_fn=lambda: need)  # exactly enough is admitted


def test_numpy_reference_peak_is_within_the_modelled_requirement():
    eo, en, W, up, lo = arrays(*random_columns(np.random.default_rng(4), 4, 5000))
    est = sr.estimate_remap_memory(4, 5000, backend="numpy")
    tracemalloc.start()
    res = run(eo, en, W, up, lo, "numpy")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert res.n_columns == 5000
    assert peak <= est.required_bytes, (peak, est.required_bytes)
    # and the allocation is linear in C, not in L*L*C: a 4x-deeper profile adds only the (L, C) output term to the peak order of magnitude
    assert peak < 3 * est.required_bytes


def test_failed_call_allocates_no_result_and_keeps_inputs(cpu_backend):
    a = arrays(*random_columns(np.random.default_rng(6), 2, 10))
    from conftest import digest

    before = digest(*a)
    with pytest.raises(sr.SoilRemapMemoryError):
        run(*a, cpu_backend, budget=0)
    assert digest(*a) == before
