"""Correction01: finite accepted parameters whose GLOBAL volumes or closure scale are not representable in FP64 are refused with a typed `SubsurfaceError` BEFORE the strict conservation helper, on both backends,
with the inputs unchanged and no result; valid arithmetic (coefficient, n_terms, every value) is identical to the pre-correction module. The common operator is not touched. Author-written."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import BIG, a2, bits, sp, ss, sw

PRE = Path(__file__).resolve().parents[3] / "s2_b1_b_correction01_pre_sources" / "spatial_transpiration.py"


def owner(backend, N, area, W, q, mask=None):
    spec = sp.spatial_layer_spec(np.array([0.0, 1.0]), theta_residual=0.0, theta_field_capacity=0.25, theta_saturation=0.5, theta_evaporation_stress=0.375, diffusivity_m2_s=0.0, n_cells=N)
    mask = np.ones(N, bool) if mask is None else mask
    col = sp.prepare_spatial_column(spec, mask, area, backend=backend, memory_budget_bytes=BIG)
    op = sw.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG)
    state = sp.spatial_column_state(col, np.full((1, N), W), 3.0, memory_budget_bytes=BIG)
    return col, op, state, np.full((1, N), q)


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def pre():
    if not PRE.is_file():
        pytest.skip("the pre-correction source copy is task evidence only")
    spec = importlib.util.spec_from_file_location("old_spatial_transpiration_pre_c1", PRE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("label,N,area,W,q", [
    ("closure_scale_overflow", 8, 1e308, 0.2, 0.2),  # before and requested are each finite (~1.6e308) but before + requested overflows
    ("direct_before_overflow", 8, 1e308, 0.5, 0.0),  # the global stored volume itself overflows
    ("direct_requested_overflow", 8, 1e308, 0.0, 0.5),  # the global requested volume overflows
])
def test_unrepresentable_global_volumes_or_scale_are_a_typed_refusal_with_no_result_and_unchanged_inputs(backend, label, N, area, W, q):
    col, op, state, req = owner(backend, N, area, W, q)
    before = (digest(state.water_m), digest(req), digest(*[col._maps[n] for n in col.retained_names()], col._mask), state.t_s)
    with pytest.raises(ss.SubsurfaceError, match="not representable in FP64") as e:
        sw.spatial_withdraw(op, state, req, memory_budget_bytes=BIG)
    assert type(e.value) is ss.SubsurfaceError  # typed (SubsurfaceError is a ValueError subclass, but never the conservation helper's raw ValueError)
    assert "refused, nothing was published" in str(e.value)
    assert (digest(state.water_m), digest(req), digest(*[col._maps[n] for n in col.retained_names()], col._mask), state.t_s) == before


def test_the_defect_was_a_raw_value_error_before_and_is_typed_now(backend, pre):
    col, op, state, req = owner(backend, 8, 1e308, 0.2, 0.2)
    old_op = pre.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG)  # the pre-correction module has its own sealed operator class
    with pytest.raises(ValueError, match="volume scale must be finite and nonnegative") as e_old:
        pre.spatial_withdraw(old_op, state, req, memory_budget_bytes=BIG)
    assert type(e_old.value) is ValueError  # the strict helper's raw error escaped
    with pytest.raises(ss.SubsurfaceError) as e_new:
        sw.spatial_withdraw(op, state, req, memory_budget_bytes=BIG)
    assert type(e_new.value) is ss.SubsurfaceError


def test_availability_has_no_volume_and_is_unaffected_by_a_huge_area(backend):
    col, op, state, req = owner(backend, 8, 1e308, 0.2, 0.2)
    av = sw.spatial_layer_availability_m(op, state, memory_budget_bytes=BIG)
    assert np.all(av == 0.2) and av.shape == (1, 8)


@pytest.mark.parametrize("area", [4.0, 1e100, 1e300])
def test_valid_arithmetic_is_identical_to_the_pre_correction_module(backend, pre, area):
    """Representable volumes (even very large) take the unchanged path: every array bit and every budget field equals the pre-correction result, with the same coefficient and n_terms."""
    N = 8
    rng = np.random.default_rng(1)
    mask = np.ones(N, bool)
    mask[3] = False
    for W, q in ((0.2, 0.05), (0.01, 0.3), (0.3, 0.0)):
        col, op, state, req = owner(backend, N, area, W, q, mask)
        req = req * mask
        new_r = sw.spatial_withdraw(op, state, req, memory_budget_bytes=BIG)
        old_r = pre.spatial_withdraw(pre.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG), state, req, memory_budget_bytes=BIG)
        for f in ("actual_m", "unmet_m", "scale"):
            np.testing.assert_array_equal(bits(getattr(new_r, f)), bits(getattr(old_r, f)))
        np.testing.assert_array_equal(bits(new_r.state.water_m), bits(old_r.state.water_m))
        assert new_r.budget.as_dict() == old_r.budget.as_dict() and new_r.budget.n_terms == 3 * 1 * int(mask.sum()) + 8


def test_the_common_operator_source_is_untouched_by_this_correction():
    from maple_syrup import subsurface_transpiration as st

    assert hashlib.sha256(Path(st.__file__).read_bytes()).hexdigest() == "7b2a53dacb3b245ff3b821337f91c385b43b46ce974bf753e7dcb950a0431e8b"
