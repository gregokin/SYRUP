"""G3 S0: the shared E-only law body and the genuine floating adapter (CPU/host; NumPy and Numba). Real columns, real steps, no fake column classes."""
# ruff: noqa: RUF059, C408
from __future__ import annotations

import copy
import dataclasses
import pickle
import tracemalloc

import numpy as np
import pytest
from cases import AREA, BIG, EDGES, MASK, PART, digest, fixed_column, inputs

from maple_syrup import floating_evaporation as fe
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import soil_evaporation as se
from maple_syrup import subsurface as ss

GOLDEN = ["43111f537b822cc7a815471e49a4f39604e7a4d732917379cf1d6e49a62a1c5e", "0692a3ba00aed5f8a0129e4c1270d47d0f2a3c256467987d413ed8d5005381d7",
          "292ccce326328558cebf724d3424efcde6ef2c8328c3363fe01f269cec26ba20"]  # computed from the PRE-extraction soil_evaporation source (fixed column, numpy)
DTS = (0.0, 3600.0, 86400.0)


def _numba_ok():
    try:
        import numba  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


BACKENDS = ["numpy"] + (["numba"] if _numba_ok() else [])


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def floating_column(depth, *, mask=MASK, backend="numpy", factor=1.0, area=AREA, bottom="open"):
    n = len(depth)
    return fsl.prepare_floating_column(np.asarray(EDGES), theta_residual=0.0625, theta_field_capacity=0.25, theta_saturation=0.5, diffusivity_m2_s=np.zeros(len(EDGES) - 2),
                                       theta_evaporation_stress=0.25, diffusion_mode="constant", bottom_drainage=bottom, soil_depth_m=np.asarray(depth, dtype=np.float64),
                                       active_mask=mask[:n].copy(), cell_area_m2=area, backend=backend, evaporation_factor=factor)


FULL = [2.0] * 7  # every layer fully porous (D > H): the fixed full-profile geometry
PARTIAL = [0.0, 0.1, 0.3, 0.75, 1.0, 0.0, 0.4]  # zero, slab (inside layer 1), slab (layer 2), deep partial, exact H, zero, slab


def stage(col, dt, pet=None, cover=None, surf=None, **kw):
    p, c, s = inputs()
    op = fe.prepare_floating_evaporation(col, PART, available_bytes_fn=lambda: BIG)
    return fe.stage_floating_evaporation(op, pet_terrain_m_s=p if pet is None else pet, cover=c if cover is None else cover, surface_water_m=s if surf is None else surf, dt_s=dt,
                                         available_bytes_fn=lambda: BIG, **kw)


# ---- the extraction is bit for bit -----------------------------------------------------------------------------------------------------
def test_fixed_wrapper_is_bit_exact_against_the_pre_extraction_source():
    p, c, s = inputs()
    for dt, gold in zip(DTS, GOLDEN, strict=True):
        op = se.prepare_evaporation(fixed_column(), PART)
        assert digest(se.stage_evaporation(op, pet_terrain_m_s=p, cover=c, surface_water_m=s, dt_s=dt)) == gold


def test_floating_equals_fixed_full_porous_bit_for_bit(backend):
    p, c, s = inputs()
    col = floating_column(FULL, backend=backend)
    for dt, gold in zip(DTS, GOLDEN, strict=True):
        assert digest(stage(col, dt)) == gold  # the floating adapter reproduces the fixed result exactly (same ONE law body)


# ---- independent math -----------------------------------------------------------------------------------------------------------------
def test_independent_partition_and_postcap_share_math(backend):
    p, c, s = inputs()
    dt = 3600.0
    st = stage(floating_column(PARTIAL, backend=backend), dt)
    for i in range(7):
        active = bool(MASK[i])
        canopy = PART.canopy_intercept - PART.canopy_slope * c[i]
        comb = p[i] * canopy * PART.soil_factor if active else 0.0
        share = 1.0 - min(max(PART.transpiration_slope * c[i] - PART.transpiration_offset, 0.0), 1.0)
        demand = p[i] * (1.0 - c[i] / 2.0) * dt
        evap = min(demand, s[i]) if active else 0.0
        assert st.potential_combined_soil_rate_m_s[i] == comb
        assert st.soil_evaporation_share[i] == share  # post-cap share, valid on EVERY cell incl. inactive
        assert st.surface_evaporation_m[i] == evap
        assert st.surface_evaporation_unmet_m[i] == ((demand - evap) if active else 0.0)
        assert st.surface_water_after_m[i] == (s[i] - evap if active else s[i])
    assert 0.0 <= st.soil_evaporation_share.min() and st.soil_evaporation_share.max() <= 1.0
    # cover 0 / 1 endpoints and a heterogeneous field are all present in the inputs
    assert set(inputs()[1].tolist()) >= {0.0, 1.0, 0.5, 0.25, 0.75}


def test_surface_evaporation_is_bounded_by_standing_water_and_ledger_closes(backend):
    p, c, s = inputs()
    st = stage(floating_column(FULL, backend=backend), 10.0**9)  # demand far above the standing water
    assert (st.surface_evaporation_m[MASK[: len(s)]] <= s[MASK[: len(s)]]).all()
    assert (st.surface_water_after_m >= 0.0).all()
    led = st.ledger
    assert abs(led.residual_m3) <= led.bound_m3 and led.n_active == int(MASK.sum())


def test_inactive_cells_are_preserved_and_invalid_inactive_cells_are_refused(backend):
    p, c, s = inputs()
    col = floating_column(FULL, backend=backend)
    st = stage(col, 3600.0)
    off = ~MASK
    assert np.array_equal(st.surface_water_after_m[off], s[off]) and not st.surface_evaporation_m[off].any() and not st.potential_combined_soil_rate_m_s[off].any()
    for name, bad, idx in (("pet", -1.0, 2), ("pet", np.nan, 5), ("cover", 1.5, 2), ("cover", -0.1, 5), ("surf", -1e-3, 2), ("surf", np.inf, 5)):
        pp, cc, ss_ = (a.copy() for a in inputs())
        {"pet": pp, "cover": cc, "surf": ss_}[name][idx] = bad  # idx 2 and 5 are INACTIVE cells: still refused
        with pytest.raises(ss.SubsurfaceError, match="refused"):
            stage(col, 3600.0, pet=pp, cover=cc, surf=ss_)


def test_dt_zero_small_and_overflow(backend):
    col = floating_column(FULL, backend=backend)
    st0 = stage(col, 0.0)
    assert not st0.surface_evaporation_m.any() and not st0.surface_evaporation_unmet_m.any()
    tiny = stage(col, 5e-324)  # smallest positive dt: a valid tiny demand, no spurious refusal
    assert (tiny.surface_evaporation_m >= 0.0).all()
    p, c, s = inputs()
    with pytest.raises(ss.SubsurfaceError, match="non-finite"):
        stage(col, 1e308, pet=np.full(7, 1e10), cover=c, surf=s)  # rate x dt overflows: refused, nothing published
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ss.SubsurfaceError):
            stage(col, bad)


# ---- provenance / factor / classes ---------------------------------------------------------------------------------------------------------
def test_double_factor_and_class_provenance_refusals(backend):
    with pytest.raises(fsl.FloatingSoilError, match="exactly 1.0"):
        fe.prepare_floating_evaporation(floating_column(FULL, backend=backend, factor=0.5), PART, available_bytes_fn=lambda: BIG)
    with pytest.raises(ss.SubsurfaceError, match="exactly 1.0"):
        se.prepare_evaporation(fixed_column(factor=0.5), PART)
    with pytest.raises(fsl.FloatingSoilError, match="FloatingSoilColumn"):
        fe.prepare_floating_evaporation(fixed_column(), PART, available_bytes_fn=lambda: BIG)  # a fixed column is refused by the floating adapter
    with pytest.raises(ss.SubsurfaceError):
        se.prepare_evaporation(floating_column(FULL, backend=backend), PART)  # and a floating column by the fixed one (no loosening)
    with pytest.raises(fsl.FloatingSoilError, match="EvaporationPartition"):
        fe.prepare_floating_evaporation(floating_column(FULL, backend=backend), {"canopy_intercept": 1.0}, available_bytes_fn=lambda: BIG)


def test_operator_is_minted_uncopyable_and_inputs_are_exact_arrays(backend):
    col = floating_column(FULL, backend=backend)
    op = fe.prepare_floating_evaporation(col, PART, available_bytes_fn=lambda: BIG)
    for bad in (lambda: copy.copy(op), lambda: copy.deepcopy(op), lambda: pickle.dumps(op), lambda: dataclasses.replace(op, backend="numpy"),
                lambda: fe.FloatingEvaporationOperator(col, PART, backend, 7)):
        with pytest.raises(fsl.FloatingSoilError):
            bad()
    with pytest.raises(fsl.FloatingSoilError):
        fe.stage_floating_evaporation({"column": col}, pet_terrain_m_s=0, cover=0, surface_water_m=0, dt_s=1.0, available_bytes_fn=lambda: BIG)
    p, c, s = inputs()
    for bad in (p.astype(np.float32), np.asfortranarray(np.stack([p, p]))[0], p[:-1].copy(), list(p), np.broadcast_to(p[:1], (7,)).copy()[::-1], np.ma.masked_array(p)):
        with pytest.raises(fsl.FloatingSoilError):
            fe.stage_floating_evaporation(op, pet_terrain_m_s=bad, cover=c, surface_water_m=s, dt_s=1.0, available_bytes_fn=lambda: BIG)
    wide = np.empty(14)
    wide[::2] = p
    with pytest.raises(fsl.FloatingSoilError):  # non-contiguous view
        fe.stage_floating_evaporation(op, pet_terrain_m_s=wide[::2], cover=c, surface_water_m=s, dt_s=1.0, available_bytes_fn=lambda: BIG)


def test_shared_body_rejects_malformed_explicit_arguments():
    p, c, s = inputs()
    active = MASK.copy()
    good = (np, PART, p, c, s, 1.0, active, 4.0, int(active.sum()))
    se._stage_laws(*good)
    for i, bad in ((2, list(p)), (3, c.astype(np.float32)), (4, s[:-1].copy()), (6, active.astype(np.uint8)), (5, float("nan")), (5, -1.0), (7, 0.0), (7, float("inf")), (8, True), (8, 99), (8, -1)):
        args = list(good)
        args[i] = bad
        with pytest.raises(ss.SubsurfaceError):
            se._stage_laws(*args)
    args = list(good)
    args[8] = None  # an unknown host count reads the active count as a twelfth entry of the same packet
    a = se._stage_laws(*args)
    assert a[5].n_active == int(active.sum()) and digest_of(a) == digest_of(se._stage_laws(*good))


def digest_of(t):
    import hashlib

    h = hashlib.sha256()
    for a in t[:5]:
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr(sorted(t[5].as_dict().items())).encode())
    return h.hexdigest()


# ---- memory model --------------------------------------------------------------------------------------------------------------------------
def test_memory_model_exact_admission_and_strict_quota(backend, monkeypatch):
    col = floating_column(FULL, backend=backend)
    n = 7
    m = fe.estimate_floating_evaporation_memory(n, backend=backend, reserve_bytes=11)
    base = se.estimate_evaporation_memory(n, backend=backend, reserve_bytes=11)
    assert m.host_bytes == base.host_bytes + fe.INTERPRETER_ALLOWANCE_BYTES and m.required_bytes == m.host_bytes + 11
    op = fe.prepare_floating_evaporation(col, PART, available_bytes_fn=lambda: BIG)
    p, c, s = inputs()
    called = []
    class Reached(Exception):
        pass

    def stub(*a, **k):
        called.append(1)
        raise Reached

    monkeypatch.setattr(se, "_stage_laws", stub)
    kw = dict(pet_terrain_m_s=p, cover=c, surface_water_m=s, dt_s=1.0)
    with pytest.raises(fsl.FloatingMemoryError):
        fe.stage_floating_evaporation(op, **kw, reserve_bytes=11, available_bytes_fn=lambda: m.required_bytes - 1)
    with pytest.raises(fsl.FloatingMemoryError):
        fe.stage_floating_evaporation(op, **kw, available_bytes_fn=lambda: 0)
    with pytest.raises(fsl.FloatingMemoryError):
        fe.stage_floating_evaporation(op, **kw, reserve_bytes=11, memory_budget_bytes=m.required_bytes - 1, available_bytes_fn=lambda: BIG)
    assert called == []  # refused BEFORE the law body (no owned allocation)
    with pytest.raises(Reached):  # the exact declared need admits and reaches the law body
        fe.stage_floating_evaporation(op, **kw, reserve_bytes=11, available_bytes_fn=lambda: m.required_bytes)
    assert called == [1]
    monkeypatch.undo()
    for bad in (-1, True, 1.0, "0", None, np.float64(2.0)):
        with pytest.raises(fsl.FloatingSoilError):
            fe.stage_floating_evaporation(op, **kw, reserve_bytes=bad, available_bytes_fn=lambda: BIG)
    with pytest.raises(fsl.FloatingSoilError, match="cuda backend"):
        fe.stage_floating_evaporation(op, **kw, host_reserve_bytes=1, available_bytes_fn=lambda: BIG)  # host_* quota applies only to the CUDA owner


def test_modelled_peak_bounds_the_traced_allocation_of_a_real_staging(backend):
    n = 4096
    rng = np.random.default_rng(1)
    pet, cover, surf = rng.random(n) * 1e-7, rng.random(n), rng.random(n) * 1e-2
    mask = rng.random(n) > 0.2
    col = floating_column([2.0] * n, mask=mask, backend=backend)
    op = fe.prepare_floating_evaporation(col, PART, available_bytes_fn=lambda: BIG)
    fe.stage_floating_evaporation(op, pet_terrain_m_s=pet, cover=cover, surface_water_m=surf, dt_s=60.0, available_bytes_fn=lambda: BIG)  # warm
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = tracemalloc.get_traced_memory()[0]
    st = fe.stage_floating_evaporation(op, pet_terrain_m_s=pet, cover=cover, surface_water_m=surf, dt_s=60.0, available_bytes_fn=lambda: BIG)
    peak = tracemalloc.get_traced_memory()[1] - t0
    tracemalloc.stop()
    model = fe.estimate_floating_evaporation_memory(n, backend=backend)
    assert peak <= model.host_bytes and st.memory == model, (peak, model.host_bytes)
    assert peak >= 5 * n * 8  # the five fresh outputs at least: a non-vacuous bound


# ---- genuine FSL step: E once, drainage once, no T double count -----------------------------------------------------------------------------------
def _water(col, frac):
    sat = np.stack([fsl.porous_thickness(col, k) * 0.5 for k in range(col.n_layers)])
    w = np.ascontiguousarray(frac * sat)
    return w


def test_floating_step_with_staged_forcing_matches_the_fixed_column_on_the_full_profile(backend):
    p, c, s = inputs()
    pet = p * 2.0**17  # a daily-scale rate so the soil branch is visibly active
    fcol = floating_column(FULL, backend=backend)
    xcol = fixed_column(backend=backend)
    water = np.ascontiguousarray(np.tile(np.array([0.9, 0.7, 0.6])[:, None], (1, 7)) * np.array([0.125, 0.125, 0.25])[:, None] * 0.5 * 2.0)
    water[:, ~MASK] = 0.0
    water = np.minimum(water, np.stack([np.full(7, 0.125 * 0.5 * 1.0), np.full(7, 0.125 * 0.5), np.full(7, 0.25 * 0.5)]))
    sf = stage(fcol, 86400.0, pet=pet)
    sx = se.stage_evaporation(se.prepare_evaporation(xcol, PART), pet_terrain_m_s=pet, cover=c, surface_water_m=s, dt_s=86400.0)
    fs = fsl.floating_step(fcol, fsl.floating_continued_state(fcol, water, 0.0), 86400.0, added_water_m=np.zeros((3, 7)), **sf.native_step_kwargs())
    xs = ss.step(xcol, ss.column_state(xcol, water, 0.0), 86400.0, added_water_m=np.zeros((3, 7)), **sx.native_step_kwargs())
    assert np.asarray(fs.state.water_m).tobytes() == np.asarray(xs.state.water_m).tobytes()
    assert np.asarray(fs.evaporation_m).tobytes() == np.asarray(xs.evaporation_m).tobytes()
    assert fs.evaporation_m.sum() > 0.0


def test_partial_zero_rock_soil_closes_with_one_evaporation_and_no_transpiration_double_count(backend):
    p, c, s = inputs()
    pet = p * 2.0**17
    col = floating_column(PARTIAL, backend=backend)
    water = _water(col, 0.8)
    assert (water[:, [0, 5]] == 0.0).all()  # zero-depth columns hold nothing
    sf = stage(col, 86400.0, pet=pet)
    step = fsl.floating_step(col, fsl.floating_continued_state(col, water, 0.0), 86400.0, added_water_m=np.zeros((3, 7)), **sf.native_step_kwargs())
    b = step.budget
    # one E and one drainage: storage change = -(evaporation + drainage + surface return as exported) -- the packet closes within the MAPLE bound and T is not a term of the step
    assert abs(b.residual_m3) <= b.bound_m3 and b.evaporation_m3 > 0.0
    assert step.evaporation_m[[0, 5]].sum() == 0.0  # no soil, no soil evaporation
    after = np.asarray(step.state.water_m)
    removed = (water.sum() - after.sum()) * AREA
    assert removed >= b.evaporation_m3 - b.bound_m3  # water left only through E (and bottom drainage/surface return): nothing else debited


def test_staging_does_not_mutate_inputs_column_or_state(backend):
    p, c, s = inputs()
    before = [a.copy() for a in (p, c, s)]
    col = floating_column(PARTIAL, backend=backend)
    stage(col, 3600.0, pet=p, cover=c, surf=s)
    assert all(np.array_equal(a, b) for a, b in zip((p, c, s), before, strict=True))
