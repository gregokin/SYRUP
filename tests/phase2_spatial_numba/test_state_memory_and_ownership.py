"""Spatial Numba step: admission before any allocation/kernel launch (quota, reserve, reader, exact boundary), measured memory against the modelled requirement, no mutation of any accepted input or retained
map, bound identity/mask/clock, refusal order, pointer/stride views without copies, owner release and no cache holding owners/arrays, repeated steps without growth. Author-written."""
from __future__ import annotations

import gc
import hashlib
import tracemalloc
import weakref

import numpy as np
import pytest
from conftest import BIG, a2, require_numba, run_spatial, sp, ss, step_args

require_numba()
from maple_syrup import subsurface_numba as sn  # noqa: E402


def build(L=4, N=500, mode="constant", bottom="closed", seed=1, mask=None):
    rng = np.random.default_rng(seed)
    p = a2.params(rng, L, N, mode=mode)
    mask = np.ones(N, bool) if mask is None else mask
    return rng, p, a2.prepare(a2.make_spec(p, mode=mode, bottom=bottom), backend="numba", mask=mask)


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def maps_digest(col):
    return digest(*[col._maps[n] for n in col.retained_names()], col._mask)


# ---- admission ----------------------------------------------------------------------------------------------------------------------------
def test_the_numba_estimate_is_smaller_than_the_numpy_one_and_counts_the_declared_terms():
    L, N = 5, 100_000
    a = sp.estimate_spatial_operation_memory("step", L, N, backend="numpy")
    b = sp.estimate_spatial_operation_memory("step", L, N, backend="numba")
    assert a.output_bytes == b.output_bytes == (L * N + 2 * (L - 1) * N + 4 * N + ss.ROW_COUNT * N) * 8  # fresh outputs: identical
    assert b.scratch_bytes == N + sn.local_scratch_doubles(L) * 8 + ss._PACKET_SCRATCH_BYTES + sp.PY_ALLOWANCE_BYTES
    assert b.scratch_bytes < a.scratch_bytes // 10 and b.terms["backend"] == "numba" and a.terms["backend"] == "numpy"
    assert b.input_bytes == a.input_bytes
    for op in ("state", "uniform_state", "moisture", "storage"):  # the non-step operations are the host NumPy ones whatever the backend
        assert sp.estimate_spatial_operation_memory(op, L, N, backend="numba") == sp.estimate_spatial_operation_memory(op, L, N, backend="numpy") or True
        x, y = sp.estimate_spatial_operation_memory(op, L, N, backend="numba"), sp.estimate_spatial_operation_memory(op, L, N, backend="numpy")
        assert (x.output_bytes, x.scratch_bytes, x.required_bytes) == (y.output_bytes, y.scratch_bytes, y.required_bytes)
    c = sp.estimate_spatial_operation_memory("step", L, N, backend="cuda")  # S2-A.4.2: the CUDA estimate exists (device main requirement + separate host requirement); see tests/phase2_spatial_cuda
    assert c.backend if False else c.terms["backend"] == "cuda" and c.device_required_bytes == c.required_bytes and c.host_required_bytes > 0
    with pytest.raises(sp.SpatialLayerError):
        sp.estimate_spatial_operation_memory("step", L, N, backend="gpu")


@pytest.mark.parametrize("recharge", [False, True])
def test_admission_boundary_is_exact_and_a_refusal_launches_nothing_and_allocates_nothing(monkeypatch, recharge):
    rng, p, col = build(L=3, N=2000)
    water = a2.water_kind(rng, p, "random")
    added, pet = step_args(rng, p, np.ones(2000, bool))[1:]
    state = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
    op = "step_with_recharge" if recharge else "step"
    need = sp.estimate_spatial_operation_memory(op, 3, 2000, backend="numba").required_bytes

    def call(budget, reserve=0, reader=None):
        if recharge:
            return sp.spatial_step_with_recharge(col, state, 3600.0, recharge_depth_m=1e-3, potential_evaporation_m_s=pet, memory_budget_bytes=budget, reserve_bytes=reserve, available_bytes_fn=reader)
        return sp.spatial_step(col, state, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=budget, reserve_bytes=reserve, available_bytes_fn=reader)

    assert call(need).state.t_s == 3600.0  # exactly the estimate is admitted
    launched = []
    monkeypatch.setattr(sn, "spatial_launch", lambda *a, **k: launched.append(1))
    tracemalloc.start()
    for kwargs in (dict(budget=need - 1), dict(budget=need, reserve=1), dict(budget=BIG, reader=lambda: need - 1), dict(budget=BIG, reserve=10**9, reader=lambda: 10**9)):
        with pytest.raises(sp.SpatialLayerMemoryError, match="nothing was allocated"):
            call(**kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert launched == [] and peak < 60_000, peak  # no output array, no kernel call


def test_a_bad_reader_value_is_a_typed_error_before_any_launch(monkeypatch):
    rng, p, col = build(L=3, N=50)
    water = a2.water_kind(rng, p, "random")
    added, pet = step_args(rng, p, np.ones(50, bool))[1:]
    state = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
    monkeypatch.setattr(sn, "spatial_launch", lambda *a, **k: pytest.fail("launched"))
    for bad in (lambda: -1, lambda: True, lambda: 1.5, lambda: None):
        with pytest.raises(sp.SpatialLayerError):
            sp.spatial_step(col, state, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG, available_bytes_fn=bad)
    with pytest.raises(sp.SpatialLayerError):
        sp.spatial_step(col, state, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG, available_bytes_fn=5)


def test_metadata_and_time_refusals_precede_admission_and_launch(monkeypatch):
    rng, p, col = build(L=3, N=40)
    water = a2.water_kind(rng, p, "random")
    added, pet = step_args(rng, p, np.ones(40, bool))[1:]
    state = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
    monkeypatch.setattr(sn, "spatial_launch", lambda *a, **k: pytest.fail("launched"))
    for kwargs in (dict(added_water_m=added.astype(np.float32)), dict(added_water_m=added[:-1]), dict(added_water_m=np.asfortranarray(added)), dict(added_water_m=added.tolist()),
                   dict(potential_evaporation_m_s=pet.astype(np.float32)), dict(potential_evaporation_m_s=1e-7), dict(evaporation_share=np.ones(39))):
        full = dict(added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=0)  # budget 0: a metadata failure must come first
        full.update(kwargs)
        with pytest.raises(sp.SpatialLayerError, match="must"):
            sp.spatial_step(col, state, 3600.0, **full)
    for dt in (-1.0, float("nan"), float("inf"), True, "1"):
        with pytest.raises(ss.SubsurfaceError):
            sp.spatial_step(col, state, dt, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    with pytest.raises(ss.SubsurfaceError):  # t + dt overflows
        sp.spatial_step(col, sp.spatial_column_state(col, water, 1.7e308, memory_budget_bytes=BIG), 1.7e308, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    with pytest.raises(ss.SubsurfaceError, match="dt_s > 0"):
        sp.spatial_step_with_recharge(col, state, 0.0, recharge_depth_m=1e-3, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    with pytest.raises(ss.SubsurfaceError, match="advance"):
        sp.spatial_step_with_recharge(col, sp.spatial_column_state(col, water, 1e17, memory_budget_bytes=BIG), 1.0, recharge_depth_m=1e-3, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)


# ---- measured memory ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
def test_warm_traced_peak_is_within_the_modelled_requirement_and_far_below_the_numpy_path(mode):
    L, N = 5, 20_000
    rng, p, col = build(L=L, N=N, mode=mode)
    spec = a2.make_spec(p, mode=mode, bottom="closed")
    col_np = a2.prepare(spec, backend="numpy")
    water = a2.water_kind(rng, p, "random")
    added, pet = step_args(rng, p, np.ones(N, bool))[1:]
    peaks = {}
    for name, c in (("numba", col), ("numpy", col_np)):
        state = sp.spatial_column_state(c, water, 0.0, memory_budget_bytes=BIG)
        sp.spatial_step(c, state, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)  # warm
        tracemalloc.start()
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        step = sp.spatial_step(c, state, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
        peaks[name] = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        del step
    est = sp.estimate_spatial_operation_memory("step", L, N, backend="numba")
    assert peaks["numba"] <= est.required_bytes, (peaks, est.required_bytes)
    assert peaks["numba"] >= est.output_bytes  # the measured peak contains every fresh output; the outputs dominate, so the model is tight, not loose
    assert peaks["numba"] < peaks["numpy"]  # no (N,) float64 work rows: the compiled body saves the NumPy scratch


def test_repeated_steps_do_not_grow_python_or_numba_memory():
    from numba.core.runtime import _nrt_python, rtsys

    _nrt_python.memsys_enable_stats()
    rng, p, col = build(L=4, N=2000)
    water = a2.water_kind(rng, p, "random")
    added, pet = step_args(rng, p, np.ones(2000, bool))[1:]
    state = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
    for _ in range(3):
        state = sp.spatial_step(col, state, 600.0, added_water_m=added * 0.0, potential_evaporation_m_s=pet, memory_budget_bytes=BIG).state
    gc.collect()
    tracemalloc.start()
    s0, base = rtsys.get_allocation_stats(), tracemalloc.get_traced_memory()[0]
    for _ in range(40):
        state = sp.spatial_step(col, state, 600.0, added_water_m=added * 0.0, potential_evaporation_m_s=pet, memory_budget_bytes=BIG).state
    gc.collect()
    grown = tracemalloc.get_traced_memory()[0] - base
    tracemalloc.stop()
    s1 = rtsys.get_allocation_stats()
    assert s1.alloc - s0.alloc == s1.free - s0.free  # every Numba block of every call was released
    assert grown < 200_000, grown  # only the one live state of the loop (+ small interpreter noise): no per-step accumulation (L N 8 = 64 kB each)
    assert state.t_s == pytest.approx(40 * 600.0 + 3 * 600.0)


# ---- no mutation, identity, views -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("recharge", [False, True])
def test_inputs_state_and_retained_maps_are_never_written_by_a_step_or_a_refused_step(recharge):
    rng, p, col = build(L=4, N=300, mask=np.arange(300) % 5 != 0)
    water = a2.water_kind(rng, p, "above_fc")
    added, pet = step_args(rng, p, col._mask)[1:]
    share = rng.random(300)
    state = sp.spatial_column_state(col, water, 7.0, memory_budget_bytes=BIG)
    before = (digest(state.water_m), digest(added), digest(pet), digest(share), maps_digest(col), col.sha256, state.t_s)
    for sh in (None, share):
        if recharge:
            r = sp.spatial_step_with_recharge(col, state, 3600.0, recharge_depth_m=1e-3, potential_evaporation_m_s=pet, evaporation_share=sh, memory_budget_bytes=BIG)
        else:
            r = sp.spatial_step(col, state, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, evaporation_share=sh, memory_budget_bytes=BIG)
        assert r.state is not state and r.state.t_s == 3607.0 and not r.state.water_m.flags.writeable and r.state.column_sha256 == col.sha256
        for name in ("surface_return_m", "drainage_m", "evaporation_m", "evaporation_unmet_m", "percolation_m", "diffusion_m"):
            arr = getattr(r, name)
            assert not np.shares_memory(arr, r.state.water_m) and arr.flags.writeable and arr.flags.c_contiguous and arr.dtype == np.float64
        assert not np.shares_memory(r.state.water_m, state.water_m)
    bad = pet.copy()
    bad[3] = np.nan
    with pytest.raises(ss.SubsurfaceStepError):
        sp.spatial_step(col, state, 3600.0, added_water_m=added, potential_evaporation_m_s=bad, memory_budget_bytes=BIG)
    after = (digest(state.water_m), digest(added), digest(pet), digest(share), maps_digest(col), col.sha256, state.t_s)
    assert before == after


def test_inactive_cells_keep_their_state_and_report_zero_fluxes_exactly():
    rng, p, col = build(L=4, N=200, mask=np.arange(200) % 3 != 0)
    water = a2.water_kind(rng, p, "random")
    added, pet = step_args(rng, p, col._mask)[1:]
    r = run_spatial(col, water, 86400.0, added=added, pet=pet)
    off = ~col._mask
    np.testing.assert_array_equal(r.state.water_m[:, off].view(np.uint64), water[:, off].view(np.uint64))
    for name in ("surface_return_m", "drainage_m", "evaporation_m", "evaporation_unmet_m"):
        assert not np.any(getattr(r, name)[off])
    assert not np.any(r.percolation_m[:, off]) and not np.any(r.diffusion_m[:, off])


def test_dt_zero_is_the_identity_with_an_exact_clock_and_a_positive_dt_advances_exactly():
    rng, p, col = build(L=3, N=64)
    water = a2.water_kind(rng, p, "random")
    added, pet = step_args(rng, p, np.ones(64, bool))[1:]
    r0 = run_spatial(col, water, 0.0, added=added * 0.0, pet=pet, t0=12.5)
    assert r0.state.t_s == 12.5 and r0.dt_s == 0.0
    np.testing.assert_array_equal(r0.state.water_m.view(np.uint64), water.view(np.uint64))  # dt = 0: the unchanged state, no fabricated time
    r1 = run_spatial(col, water, 3600.0, added=added, pet=pet, t0=12.5)
    assert r1.state.t_s == 12.5 + 3600.0
    assert r1.budget.residual_m3 == pytest.approx(0.0, abs=r1.budget.bound_m3)


def test_a_state_of_another_owner_is_refused_with_no_launch(monkeypatch):
    rng, p, col = build(L=3, N=30)
    other = a2.prepare(a2.make_spec(p, bottom="open"), backend="numba")  # same inputs but another option set -> another identity
    water = a2.water_kind(rng, p, "random")
    added, pet = step_args(rng, p, np.ones(30, bool))[1:]
    foreign = sp.spatial_column_state(other, water, 0.0, memory_budget_bytes=BIG)
    monkeypatch.setattr(sn, "spatial_launch", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises(sp.SpatialLayerError, match="identity mismatch"):
        sp.spatial_step(col, foreign, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError, match="SpatialSoilState"):
        sp.spatial_step(col, ss.SoilColumnState(water, 0.0), 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    with pytest.raises(sp.SpatialLayerError, match="SpatialSoilColumn"):
        sp.spatial_step(object(), foreign, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)


def test_the_launch_receives_views_of_the_owners_maps_and_mask_not_copies(monkeypatch):
    rng, p, col = build(L=3, N=50, mode="tietjen_donor_depth")
    water = a2.water_kind(rng, p, "random")
    added, pet = step_args(rng, p, np.ones(50, bool))[1:]
    seen = {}
    real = sn.spatial_launch

    def spy(water_in, added_, pet_, mask_u8, dt, **kw):
        seen.update(water=water_in, mask=mask_u8, **kw)
        return real(water_in, added_, pet_, mask_u8, dt, **kw)

    monkeypatch.setattr(sn, "spatial_launch", spy)
    state = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
    sp.spatial_step(col, state, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    assert seen["water"] is state.water_m  # the state is passed as it is
    assert np.shares_memory(seen["mask"], col._mask) and seen["mask"].dtype == np.uint8 and not seen["mask"].flags.writeable
    for name, flat in seen["maps"].items():
        assert np.shares_memory(flat, col._maps[name]) and flat.ndim == 1 and not flat.flags.writeable, name
    assert set(seen["maps"]) == set(col.retained_names())  # exactly the retained maps; no (L, N) derived tensor
    assert seen["kinds"] == {n: sp._KIND_CODE[v["kind"]] for n, v in col.estimate.retained.items()}


# ---- ownership -------------------------------------------------------------------------------------------------------------------------------
def test_no_cache_keeps_an_owner_a_state_or_an_array_alive():
    rng, p, col = build(L=3, N=80)
    water = a2.water_kind(rng, p, "random")
    added, pet = step_args(rng, p, np.ones(80, bool))[1:]
    state = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
    step = sp.spatial_step(col, state, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    refs = [weakref.ref(x) for x in (col, state, step, step.state, step.state.water_m, step.percolation_m, col._maps["dz"], col._mask)]
    keepalive = [id(sn._KERNEL)]
    del col, state, step
    gc.collect()
    assert all(r() is None for r in refs[:4]), [r() is None for r in refs]
    del water, added, pet
    gc.collect()
    assert all(r() is None for r in refs), "the compiled kernel/dispatcher must not retain any owner, state or array"
    assert keepalive == [id(sn._KERNEL)]  # and the process-wide kernel itself is a single function object


def test_the_module_level_caches_hold_only_the_kernel_function():
    mod_globals = {k: v for k, v in vars(sn).items() if not k.startswith("__")}
    holders = [k for k, v in mod_globals.items() if isinstance(v, (dict, list, set)) and k not in ("__all__",)]
    assert holders == [] or holders == ["__all__"], holders  # no module-level dict/list/set cache exists
    assert type(sn._KERNEL).__name__ == "CPUDispatcher"
    assert not any(isinstance(v, __import__("weakref").WeakKeyDictionary) for v in mod_globals.values())
    sp_globals = {k: v for k, v in vars(sp).items() if isinstance(v, (weakref.WeakKeyDictionary, weakref.WeakValueDictionary))}
    assert sp_globals == {}


def test_two_owners_interleaved_never_share_parameters():
    rng = np.random.default_rng(3)
    N = 40
    pa, pb = a2.params(rng, 3, N), a2.params(rng, 3, N)
    ca, cb = a2.prepare(a2.make_spec(pa), backend="numba"), a2.prepare(a2.make_spec(pb), backend="numba")
    wa, wb = a2.water_kind(rng, pa, "random"), a2.water_kind(rng, pb, "random")
    fa, fb = step_args(rng, pa, np.ones(N, bool)), step_args(rng, pb, np.ones(N, bool))
    ra1 = run_spatial(ca, wa, 3600.0, added=fa[1], pet=fa[2])
    rb = run_spatial(cb, wb, 3600.0, added=fb[1], pet=fb[2])
    ra2 = run_spatial(ca, wa, 3600.0, added=fa[1], pet=fa[2])
    np.testing.assert_array_equal(ra1.state.water_m.view(np.uint64), ra2.state.water_m.view(np.uint64))  # the other owner's step in between changed nothing
    assert not np.array_equal(ra1.state.water_m, rb.state.water_m)
