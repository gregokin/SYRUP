"""Refusal matrix (typed, before any output), metadata checks, layouts/joins, block-edge sizes, backend equality, chunk invariance, ownership. Author-written."""
from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from conftest import AT, BIG, RT, call, close, random_valid, val
from decimal_model import model

from maple_syrup import soil_texture_properties as tp

BAD = [
    (dict(C=0.61, S=0.2, Si=0.19), tp.FLAG_CLAY), (dict(OM=-0.01), tp.FLAG_ORGANIC_MATTER), (dict(OM=8.01), tp.FLAG_ORGANIC_MATTER), (dict(rho=999.0), tp.FLAG_DENSITY),
    (dict(rho=1801.0), tp.FLAG_DENSITY), (dict(S=-0.1, Si=0.9), tp.FLAG_FRACTION), (dict(S=1.1, Si=-0.2, C=0.1), tp.FLAG_FRACTION), (dict(S=0.4, Si=0.4, C=0.19), tp.FLAG_FRACTION),
    (dict(S=0.4, Si=0.4, C=0.2001), tp.FLAG_FRACTION), (dict(S=float("nan")), tp.FLAG_NONFINITE), (dict(rho=float("inf")), tp.FLAG_NONFINITE), (dict(OM=float("nan")), tp.FLAG_NONFINITE),
    (dict(rho=1800.0 * 1.0001), tp.FLAG_DENSITY), (dict(rho=1000.0 * 0.5), tp.FLAG_DENSITY)]


@pytest.mark.parametrize("kw,bit", BAD)
def test_refusals(kw, bit, backend):
    with pytest.raises(tp.TexturePropertyError) as e:
        call(**kw, backend=backend)
    assert e.value.flags & bit and e.value.n_flagged == 1


def test_density_factor_and_gap_refusals(backend):
    rn = 1000.0 / float(model(0.40, 0.40, 0.20, 2.5, 1000.0)[1]["df"])
    for f in (0.85, 0.899, 1.31, 1.4):
        rho = rn * f
        if 1000.0 <= rho <= 1800.0:
            with pytest.raises(tp.TexturePropertyError) as e:
                call(0.40, 0.40, 0.20, 2.5, rho, backend=backend)
            assert e.value.flags & tp.FLAG_DENSITY_FACTOR
    r = call(0.40, 0.40, 0.20, 2.5, rn * 1.0, backend=backend)
    assert val(r) > 0


def test_closure_within_declared_fp_bound(backend):
    """Fractions that sum to 1 only up to FP64 rounding are accepted; a 1e-9 deficit is refused. The bound is MAPLE's own summation_error_bound_kg(3, 1)."""
    assert tp.closure_bound() == pytest.approx(4 * 3 * np.finfo(float).eps)
    S, C = 0.1 + 0.2, 0.1  # 0.1 + 0.2 is 0.30000000000000004: the closing silt fraction absorbs the rounding
    call(S, 1.0 - S - C, C, 2.5, 1400.0, backend=backend)
    with pytest.raises(tp.TexturePropertyError) as e:
        call(0.4, 0.4, 0.2 - 1e-9, 2.5, 1400.0, backend=backend)
    assert e.value.flags & tp.FLAG_FRACTION


@pytest.mark.parametrize("kw", [
    dict(S=[0.4]), dict(S=np.array([0.4], dtype=np.float32)), dict(S=np.ones((2,)), L=3), dict(S=np.ones((1, 2, 3))), dict(S=True), dict(S=np.asfortranarray(np.ones((2, 3))), L=2, N=3),
    dict(S=np.ones((3, 2)), L=2, N=3), dict(S=np.ma.masked_array([0.4])), dict(S="0.4"), dict(S=None)])
def test_metadata_refusals_before_any_work(kw, backend, monkeypatch):
    monkeypatch.setattr(tp, "_derive_numpy", lambda *a, **k: pytest.fail("a kernel ran after a metadata failure"))
    with pytest.raises(tp.TexturePropertyError):
        call(**kw, backend=backend)


@pytest.mark.parametrize("kw", [dict(L=0), dict(N=0), dict(L=True), dict(L=1.5), dict(backend="gpu"), dict(diagnostics=1), dict(chunk_columns=0), dict(hash_chunk_bytes=4), dict(memory_budget_bytes=-1),
                                dict(host_reserve_bytes=1), dict(host_memory_budget_bytes=10), dict(host_available_bytes_fn=lambda: 1), dict(available_bytes_fn=3)])
def test_argument_refusals(kw):
    with pytest.raises(tp.TexturePropertyError):
        call(**kw)


def test_mandatory_inputs_have_no_default():
    for k in tp.INPUTS:
        kw = dict(sand_fraction=0.4, silt_fraction=0.4, clay_fraction=0.2, organic_matter_percent=2.5, bulk_density_kg_m3=1400.0)
        kw[k] = None
        with pytest.raises(tp.TexturePropertyError, match="mandatory"):
            tp.estimate_soil_texture_properties(**kw, n_layers=1, n_cells=1, memory_budget_bytes=BIG)
    with pytest.raises(TypeError):
        tp.estimate_soil_texture_properties(sand_fraction=0.4, silt_fraction=0.4, clay_fraction=0.2, organic_matter_percent=2.5, n_layers=1, n_cells=1, memory_budget_bytes=BIG)  # no density default
    with pytest.raises(TypeError):
        call(density_factor=1.0)  # no caller DF


def test_flag_location_and_counts(backend):
    """First flagged element = lowest (layer, cell) in layer-major order; every flagged element is counted; the whole call is refused."""
    L, N = 3, 5
    S = np.full((L, N), 0.4)
    Si, C, OM, rho = np.full((L, N), 0.4), np.full((L, N), 0.2), np.full((L, N), 2.5), np.full((L, N), 1400.0)
    OM[2, 1], OM[1, 4], OM[1, 3] = 9.0, -1.0, 9.0
    with pytest.raises(tp.TexturePropertyError) as e:
        call(S, Si, C, OM, rho, L=L, N=N, backend=backend)
    assert (e.value.first_layer, e.value.first_cell, e.value.n_flagged) == (1, 3, 3) and e.value.bit == tp.FLAG_ORGANIC_MATTER


def test_failure_publishes_nothing_and_never_mutates_inputs(backend):
    rng = np.random.default_rng(2)
    S, Si, C, OM, rho = random_valid(rng, 6)
    arrs = [a.reshape(2, 3).copy() for a in (S, Si, C, OM, rho)]
    arrs[3][1, 2] = 50.0
    before = [a.copy() for a in arrs]
    with pytest.raises(tp.TexturePropertyError):
        call(*arrs, L=2, N=3, backend=backend)
    for a, b in zip(arrs, before):
        assert np.array_equal(a, b)


# ---- layouts, shapes, equality with single-element calls -----------------------------------------------------------------------------------------------------------------------------------------
def element_inputs(rng, L, N):
    S, Si, C, OM, rho = random_valid(rng, L * N)
    return [a.reshape(L, N).copy() for a in (S, Si, C, OM, rho)]


def single(args, l, c, backend):
    return call(*[float(a[l, c]) for a in args], backend=backend, diagnostics=True)


@pytest.mark.parametrize("L,N", [(1, 1), (3, 1), (2, 127), (2, 128), (3, 129), (2, 257)])
def test_cell_layout_equals_single_element_calls(L, N, backend):
    rng = np.random.default_rng(10 + L + N)
    args = element_inputs(rng, L, N)
    r = call(*args, L=L, N=N, backend=backend, diagnostics=True, chunk_columns=64)
    assert r.layout == "cell" and r.array("theta_saturation").shape == (L, N)
    for l, c in {(0, 0), (L - 1, N - 1), (L // 2, N // 2), (0, N - 1)}:
        s = single(args, l, c, backend)
        for name in ("theta_saturation", "theta_field_capacity_6kpa", "theta_1500", "air_entry_kpa", "provenance"):
            assert r.array(name)[l, c] == s.array(name).reshape(-1)[0], name
        assert r.array("saturated_conductivity_m_s")[l, c] == s.array("saturated_conductivity_m_s").reshape(-1)[0]


def test_layout_joins_and_shared_only_is_O_of_L(backend):
    L, N = 4, 10**6
    rng = np.random.default_rng(3)
    S, Si, C, OM, rho = (a[:, 0].copy() for a in element_inputs(rng, L, 1))
    r = call(S, Si, C, OM, rho, L=L, N=N, backend=backend)
    assert r.layout == "shared" and r.array("theta_saturation").shape == (L,) and r.output_bytes == 3 * 8 * L  # never expanded to N
    assert r.estimate.outputs["theta_saturation"] == 8 * L
    r2 = call(0.4, 0.4, 0.2, 2.5, 1400.0, L=L, N=N, backend=backend)
    assert r2.layout == "shared" and r2.array("theta_saturation").shape == (L,)
    cell_rho = np.tile(rho[:, None], (1, 7))
    r3 = call(S, Si, C, OM, cell_rho, L=L, N=7, backend=backend)
    assert r3.layout == "cell" and r3.array("theta_saturation").shape == (L, 7)
    for c in range(7):
        assert np.array_equal(r3.array("theta_saturation")[:, c], r.array("theta_saturation"))
    assert set(r3.layouts.values()) == {"shared", "cell"}


def test_numpy_numba_agreement():
    pytest.importorskip("numba")
    rng = np.random.default_rng(4)
    L, N = 3, 301
    args = element_inputs(rng, L, N)
    ov = np.where(rng.random((L, N)) < 0.3, rng.uniform(0.5, 20.0, (L, N)), np.nan)
    a = call(*args, L=L, N=N, backend="numpy", diagnostics=True, air_entry_kpa_override=ov)
    b = call(*args, L=L, N=N, backend="numba", diagnostics=True, air_entry_kpa_override=ov)
    for name in ("theta_saturation", "theta_field_capacity_6kpa", "theta_1500", "air_entry_kpa", "provenance"):
        assert np.array_equal(a.array(name), b.array(name)), name  # arithmetic without a library call: bitwise
    ka, kb = a.array("saturated_conductivity_m_s"), b.array("saturated_conductivity_m_s")
    assert np.all(np.abs(ka - kb) <= AT * 1e-4 + RT * np.abs(kb))  # log/pow: declared bound
    assert a.sha256 == b.sha256 and a.n_fc_equals_saturation == b.n_fc_equals_saturation > 0
    # the same typed refusal and the same first element
    bad = [x.copy() for x in args]
    bad[3][2, 17] = 12.0
    bad[3][1, 200] = 12.0
    errs = []
    for bk in ("numpy", "numba"):
        with pytest.raises(tp.TexturePropertyError) as e:
            call(*bad, L=L, N=N, backend=bk)
        errs.append((e.value.first_layer, e.value.first_cell, e.value.n_flagged, e.value.flags))
    assert errs[0] == errs[1] == (1, 200, 2, tp.FLAG_ORGANIC_MATTER)


def test_chunk_and_hash_chunk_invariance(backend):
    rng = np.random.default_rng(5)
    L, N = 2, 130
    args = element_inputs(rng, L, N)
    ref = call(*args, L=L, N=N, backend=backend, diagnostics=True)
    for chunk, hc in ((1, 8), (7, 24), (64, 4096), (130, 1 << 20), (1000, 8)):
        r = call(*args, L=L, N=N, backend=backend, diagnostics=True, chunk_columns=chunk, hash_chunk_bytes=hc)
        assert r.sha256 == ref.sha256
        for name in ref.names():
            assert np.array_equal(r.array(name), ref.array(name)), (chunk, name)


def test_identity_tracks_inputs_options_and_recipe(backend):
    base = call(backend=backend)
    assert call(backend=backend).sha256 == base.sha256
    assert call(OM=2.5000001, backend=backend).sha256 != base.sha256
    assert call(backend=backend, diagnostics=True).sha256 != base.sha256
    assert call(backend=backend, theta_1500_override=0.07).sha256 != base.sha256
    assert call(L=2, backend=backend, S=np.array([0.4, 0.4])).sha256 != call(L=2, backend=backend).sha256  # layout/shape is part of the identity
    assert call(backend=backend).sha256 == base.sha256 and base.summary()["recipe"] == tp.RECIPE_TAG


def test_ownership_sealing_and_readonly(backend):
    S = np.full(1, 0.4)
    r = call(S, backend=backend)
    before = r.array("theta_saturation").copy()
    sha = r.sha256
    S[:] = 0.3  # mutating the caller's input afterwards changes nothing the result owns
    assert np.array_equal(r.array("theta_saturation"), before) and r.sha256 == sha
    with pytest.raises(ValueError):
        r.array("theta_saturation")[0] = 1.0
    d = r.detached("theta_saturation", memory_budget_bytes=BIG)
    d[0] = 5.0
    assert r.array("theta_saturation")[0] == before[0]
    for bad in (lambda: copy.copy(r), lambda: copy.deepcopy(r), lambda: pickle.dumps(r), lambda: tp.SoilTextureProperties(*([None] * 11))):
        with pytest.raises(tp.TexturePropertyError):
            bad()
    with pytest.raises(AttributeError):
        r.n_layers = 3
    with pytest.raises(tp.TexturePropertyError):
        r.array("nope")
    with pytest.raises(tp.TexturePropertyError):
        r.detached("nope", memory_budget_bytes=BIG)
    assert "no native adoption" in r.summary()["scope"] or "no bed/water/clock mutation" in r.summary()["scope"]
    assert set(r.names()) == set(tp.OUTPUTS)


def test_default_outputs_only_three_maps_and_diagnostics_are_optional(backend):
    assert set(call(backend=backend).names()) == set(tp.OUTPUTS)
    assert set(call(backend=backend, diagnostics=True).names()) == set(tp.OUTPUTS) | set(tp.DIAGNOSTICS)


def test_recipe_metadata_is_immutable_and_complete():
    rec = tp.recipe_record()
    assert rec["ks_anchor"].startswith("theta_33") and rec["recipe"] == tp.RECIPE_TAG and rec["fc_pressure_kpa"] == 6.0
    assert set(tp.source_digests()) == {"soil_texture_properties.py", "soil_texture_properties_numba.py", "soil_texture_properties_cuda.py"}
