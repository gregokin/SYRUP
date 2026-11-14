"""Phase 2.5 vegetation lifecycle, CUDA HARDWARE tests: EXECUTABLE, AUTHORED, NOT RUN by their author (no CuPy / GPU / NVRTC / nvidia-smi was touched; the author only compiled and host-collected them). Root runs them on its allocated
device with `SYRUP_VEG_LIFECYCLE_RELEASE_GPU=1` (device id `SYRUP_VEG_LIFECYCLE_GPU_DEVICE`, default 0) and reports the actual errors for correction. Without the flag they are skipped at collection; WITH the flag a missing CuPy / device is
a SETUP FAILURE, never a skip.

Fixtures are the qualified Phase 2 ones (loaded lazily by path in `veg_lc_gpu_support`): the CUDA event<->daily bridge definitions' `matched_pair` (a genuinely completed CUDA closed Boundary, Dense and published Windowed, with a MATCHED CPU
twin) and the CUDA lifecycle definitions' thin closed lake with a real terminal deposit (`opened` / `complete`: typed quiet completion, relative bounded loops, no time cap); the projection is the REAL LEAF P1 + the S0 binding. No fake class,
head, generation, quiet state or solver. The outer fixture selects the device ONCE; the wrong-current-device control uses a `with Device(other)` context around the public call (needs two devices, else that one test skips) and never switches a
device to satisfy an API check.

Declared fixture limits: tiny grids (N = 12 .. 30, L = 3); the device-peak test is a SMALL actual peak measured with MemoryHook (labelled; a moderate-scale peak needs a larger fixture); the allocator-rounded block sizes are compared with the
declared model plus an explicit per-allocation rounding slack."""
from __future__ import annotations

import gc
import os
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import veg_lc_builders as vb
import veg_lc_gpu_support as sup

RELEASE = os.environ.get("SYRUP_VEG_LIFECYCLE_RELEASE_GPU") == "1"
DEVICE = int(os.environ.get("SYRUP_VEG_LIFECYCLE_GPU_DEVICE", "0"))
pytestmark = pytest.mark.skipif(not RELEASE, reason="CUDA hardware tests: set SYRUP_VEG_LIFECYCLE_RELEASE_GPU=1 on the allocated device (authored, not run by the author)")

BIG = 1 << 40
TOL = {"rtol": 1e-9, "atol": 1e-12}  # the declared CPU/GPU float64 tolerance of the Phase 2 GPU definitions (not bitwise)
RUN_ = "ab" * 32


@pytest.fixture(scope="module")
def gpu():
    try:
        import cupy as cp

        cp.cuda.Device(DEVICE).use()  # the ONE outer device selection; production APIs only enforce the current device afterwards
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"SYRUP_VEG_LIFECYCLE_RELEASE_GPU=1 but CuPy / a CUDA device is unusable: {exc}")
    return SimpleNamespace(cp=cp, m=sup.modules())


def hst(a):
    from maple.core.backend import to_host

    return np.asarray(to_host(a))


def gapply(side, b, bd, **over):
    from maple_syrup import vegetation_lifecycle as vl

    kw = {"memory_budget_bytes": BIG, "available_bytes_fn": side.rdr, "host_memory_budget_bytes": BIG, "host_available_bytes_fn": lambda: BIG}
    kw.update(over)
    return vl.apply_vegetation_generation(side.lin, b, bd, **kw)


def lake(gpu, tmp, windowed):
    """OPTIONAL separately labelled fixture (NOT used by the default carry tests: its reopened closed-pond occurrence can take very long to complete): a genuine CUDA closed Boundary of the thin lake with a real terminal deposit, so the
    closing refresh would see a real deposit-driven property change. Kept for a later explicit terminal-deposit control."""
    from maple_syrup import floating_event_lifecycle as lc

    ld = gpu.m.ld
    owner, factory, lin, occ, st, reader = ld.opened(tmp, windowed)
    h = ld.complete(occ, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=reader, host_available_bytes_fn=lambda: BIG)
    ev = occ.event
    return SimpleNamespace(lin=lin, b=b, rdr=reader, ev=ev, owner=owner, factory=factory)


def event_builder(ev, sched=None):
    from maple_syrup.pond_event import PondEvent

    def build(vert, topology, cover):
        return PondEvent(ev.context, None, ev.sediment, ev.schedule if sched is None else sched, topology, control=ev.control, policy=ev.policy, splash=ev.splash, vegetation_cover_fraction=cover, floating_vertical=vert)

    return build


# ----------------------------------------------------------------------------------------------------------------------------------------------
# 1 zero-time apply: nothing physical moves, the NEW device law equals the matched CPU preparation, head LAST
# ----------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_zero_time_apply_preserves_everything_and_matches_the_cpu_twin_law(gpu, leaf, tmp_path, windowed):
    from maple_syrup import floating_event_lifecycle as lc
    from maple_syrup import floating_matrix_infiltration as fm
    from maple_syrup import vegetation_lifecycle as vl

    bd_ = gpu.m.bd
    g, c = bd_.matched_pair(tmp_path, bd_.dry_kwargs(), windowed)
    b = g.b
    s = vb.stack(leaf, b)
    binding = vb.binding(leaf, b, s)
    before = {"W": hst(b.root.water_state.water_m).copy(), "J": hst(b.root.history.wetting_m).copy(), "I": hst(b.root.history.cumulative_intake_m).copy(), "H": hst(b.surface_depth_m).copy(),
              "qx": hst(b.qx_m2_s).copy(), "qy": hst(b.qy_m2_s).copy(), "ledger": dict(b.ledger), "t": b.t_s, "seq": b.seq}
    nb = gapply(g, b, binding)
    assert g.lin.head is nb and nb is not b and nb.role == "boundary"
    assert nb.root.water_state is b.root.water_state and nb.bed is b.bed and nb.surface_depth_m is b.surface_depth_m and nb.routing is b.routing and nb.provenance is b.provenance and nb.cursor is b.cursor
    assert (nb.t_s, nb.seq, nb.root.generation, nb.root.source_witness) == (before["t"], before["seq"], b.root.generation, b.root.source_witness) and nb.ledger == before["ledger"]
    for name, got in (("W", nb.root.water_state.water_m), ("J", nb.root.history.wetting_m), ("I", nb.root.history.cumulative_intake_m), ("H", nb.surface_depth_m), ("qx", nb.qx_m2_s), ("qy", nb.qy_m2_s)):
        assert hst(got).tobytes() == before[name].tobytes(), name  # bit for bit (no water delta, J/I rebound exactly)
    assert nb.root.history is not b.root.history and nb.root.history.operator_sha256 == nb.root.matrix_op.sha256 and nb.root.matrix_op.layouts["vegetation_factor"] == "cell"
    rec = vl.current_record(nb)
    assert rec.generation == 1 and rec.backend == "cuda" and rec.shade_cover_device is not None and rec.splash_cover_device is not None and rec.bound_shade is rec.shade_cover_device
    assert hst(rec.shade_cover_device).tobytes() == np.asarray(rec.shade_cover).tobytes() and hst(rec.splash_cover_device).tobytes() == np.asarray(rec.splash_cover_2d).tobytes()
    # the matched CPU twin prepared from the SAME binding (the same projection on the matched terrain) gives the same law within the declared tolerance
    bc = c.b
    sc = vb.stack(leaf, bc)
    nbc = vl.apply_vegetation_generation(c.lin, bc, vb.binding(leaf, bc, sc), memory_budget_bytes=BIG, available_bytes_fn=c.rdr)
    np.testing.assert_allclose(hst(fm.held_cell_vegetation_factor(nb.root.matrix_op)), np.asarray(fm.held_cell_vegetation_factor(nbc.root.matrix_op)), **TOL)
    for dev_arr, cpu_arr in zip(nb.root.pref_op._law, nbc.root.pref_op._law, strict=True):
        np.testing.assert_allclose(hst(dev_arr), np.asarray(cpu_arr), **TOL)
    with pytest.raises(lc.LifecycleError):
        gapply(g, b, binding)  # the superseded head is stale
    assert gapply(g, nb, binding) is nb  # the same binding is idempotent


# ----------------------------------------------------------------------------------------------------------------------------------------------
# 2 refusal / retry: dual pools (device and host separately), readers, wrong / stale inputs, current device
# ----------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_refusals_leave_the_head_and_allocator_untouched_and_the_exact_need_retries(gpu, leaf, tmp_path, windowed):
    from maple_syrup import floating_soil_layers_cuda as fcu
    from maple_syrup import vegetation_lifecycle as vl
    from maple_syrup import water_vegetation_binding as wvb

    cp = gpu.cp
    g, _c = gpu.m.bd.matched_pair(tmp_path, gpu.m.bd.dry_kwargs(), windowed)
    b = g.b
    s = vb.stack(leaf, b)
    binding = vb.binding(leaf, b, s)
    pool = cp.get_default_memory_pool()
    head0, root0 = g.lin.head, b.root
    est = sup.apply_estimate(b, binding)

    def untouched(used0):
        assert g.lin.head is head0 and b.root is root0 and b.root.vegetation is None
        assert pool.used_bytes() == used0  # nothing was left allocated by the refusal

    cp.cuda.Stream.null.synchronize()
    used0 = pool.used_bytes()
    # inputs: wrong surface, wrong mask, forged / foreign binding
    with pytest.raises(vl.VegetationLifecycleError, match="CURRENT committed surface"):
        gapply(g, b, vb.binding(leaf, b, vb.stack(leaf, b, surface=s["surface"] + 1.0e-3)))
    m = np.ones(s["surface"].shape, dtype=bool)
    m[0, 0] = False
    with pytest.raises(vl.VegetationLifecycleError, match="mask"):
        gapply(g, b, vb.binding(leaf, b, vb.stack(leaf, b, mask=m)))
    with pytest.raises(vl.VegetationLifecycleError):
        gapply(g, b, object())
    with pytest.raises(vl.VegetationLifecycleError):
        gapply(g, b, object.__new__(wvb.VegetationBinding))
    untouched(used0)
    # pools: device and host SEPARATELY at need - 1, readers, the host pool is mandatory
    dev_need, host_need = est["peak_device_bytes"], est["peak_host_bytes"] + fcu.PY_ALLOWANCE_BYTES
    for kw, label in (({"memory_budget_bytes": dev_need - 1}, "device budget"), ({"host_memory_budget_bytes": host_need - 1}, "host budget"), ({"available_bytes_fn": lambda: dev_need - 1}, "device reader"),
                      ({"host_available_bytes_fn": lambda: host_need - 1}, "host reader")):
        with pytest.raises(vl.VegetationLifecycleError, match="nothing was allocated|needs an estimated"):
            gapply(g, b, binding, **kw)
        untouched(used0)
    with pytest.raises(vl.VegetationLifecycleError, match="host pool"):
        vl.apply_vegetation_generation(g.lin, b, binding, memory_budget_bytes=BIG, available_bytes_fn=g.rdr)  # no host pool given
    # the reserve of each pool is counted once: a reserve of 1 byte over the exact need refuses on its OWN pool only
    with pytest.raises(vl.VegetationLifecycleError):
        gapply(g, b, binding, memory_budget_bytes=dev_need, reserve_bytes=1)
    with pytest.raises(vl.VegetationLifecycleError):
        gapply(g, b, binding, host_memory_budget_bytes=host_need, host_reserve_bytes=1)
    untouched(used0)
    # the exact modelled need is accepted (the head was retryable)
    nb = gapply(g, b, binding, memory_budget_bytes=dev_need, host_memory_budget_bytes=host_need)
    assert g.lin.head is nb and vl.current_record(nb).generation == 1


def test_the_current_device_gate_precedes_every_reader_and_hash(gpu, leaf, tmp_path):
    from maple_syrup import vegetation_lifecycle as vl

    cp = gpu.cp
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("the wrong-current-device control needs a second device")
    g, _c = gpu.m.bd.matched_pair(tmp_path, gpu.m.bd.dry_kwargs(), False)
    b = g.b
    binding = vb.binding(leaf, b, vb.stack(leaf, b))
    other = 1 if DEVICE == 0 else 0
    reads = []
    # the CURRENT device is wrong for the boundary's device: the public API must refuse before any reader, copy or hash
    with cp.cuda.Device(other), pytest.raises(vl.VegetationLifecycleError, match="device"):
        gapply(g, b, binding, available_bytes_fn=lambda: reads.append("d") or BIG, host_available_bytes_fn=lambda: reads.append("h") or BIG)
    assert reads == [] and g.lin.head is b and b.root.vegetation is None


# ----------------------------------------------------------------------------------------------------------------------------------------------
# 3 surface hash: bounded chunked D2H, exactly 8 N bytes, host pool admitted first
# ----------------------------------------------------------------------------------------------------------------------------------------------
def test_the_device_surface_hash_is_a_bounded_chunk_stream_of_eight_n_bytes_gated_before_any_transfer(gpu, leaf, tmp_path, monkeypatch):
    import hashlib

    from maple.core import backend as mb
    from maple.core.types.topographic_commit import committed_elevation_m

    from maple_syrup import floating_event_lifecycle_cuda as flc
    from maple_syrup import vegetation_lifecycle_cuda as vlc

    cp = gpu.cp
    g, _c = gpu.m.bd.matched_pair(tmp_path, gpu.m.bd.dry_kwargs(), False)
    b = g.b
    binding = vb.binding(leaf, b, vb.stack(leaf, b))
    surface = committed_elevation_m(b.bed.committed_topography)
    assert hasattr(surface, "__cuda_array_interface__")  # a genuine DEVICE elevation
    calls = []
    real = mb.to_host
    monkeypatch.setattr(mb, "to_host", lambda a, *x, **k: (calls.append(int(getattr(a, "nbytes", 0))), real(a, *x, **k))[1])
    q = flc.quota_of(g.rdr, 0, lambda: BIG, 0, BIG, BIG)
    # a refused host pool: NOTHING is transferred
    qb = flc.quota_of(g.rdr, 0, lambda: 0, 0, BIG, BIG)
    with pytest.raises(Exception, match="nothing was allocated|available"):
        vlc.hash_device_surface(cp, hashlib.sha256(), surface, tuple(binding.shape), qb)
    assert calls == []
    h = hashlib.sha256()
    moved = vlc.hash_device_surface(cp, h, surface, tuple(binding.shape), q)
    n = int(np.prod(binding.shape))
    assert moved == 8 * n == sum(calls) and max(calls) <= vlc.SURFACE_HASH_CHUNK_BYTES and h.hexdigest() == binding.surface_sha256  # the SAME bytes the binding hashed on the host
    assert len(calls) == -(-8 * n // vlc.SURFACE_HASH_CHUNK_BYTES)


# ----------------------------------------------------------------------------------------------------------------------------------------------
# 4 property refresh carry: held device g, no restream, CURRENT ids
# ----------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_a_real_cuda_closed_event_refresh_keeps_the_held_g_without_restream_and_attribution_is_current(gpu, leaf, tmp_path, monkeypatch, windowed):
    """BOUNDED fixture for the LAW carry: the matched DRY thin-rain boundary (it genuinely completes quickly), the vegetation generation applied, then a NEW short wind-free native rain occurrence (0.05 s of 10 mm/h, the fixture's own forcing
    shape) that genuinely completes (typed quiet, relative bounded loop, no time cap) and is closed through the ACTUAL CUDA refresh. WHAT CHANGES: the refresh is a genuine closed-event property refresh -- it prepares a NEW root column and
    operators and the property generation advances by exactly one -- but this fixture has no terminal deposit and no erosion, so the texture-derived properties (hence the prepared column / matrix identities) may be UNCHANGED; the test does
    not claim they differ. A property-changing terminal-deposit control is the separate `lake` fixture, not the default."""
    from maple_syrup import floating_event_lifecycle as lc
    from maple_syrup import floating_matrix_infiltration as fm
    from maple_syrup import floating_preferential_infiltration as fpi
    from maple_syrup import native_floating_soil_cuda as ncuda
    from maple_syrup import vegetation_lifecycle as vl
    from maple_syrup.rainfall import constant_rainfall

    bd_ = gpu.m.bd
    g, _c = bd_.matched_pair(tmp_path, bd_.dry_kwargs(), windowed)
    b = g.b
    nb = gapply(g, b, vb.binding(leaf, b, vb.stack(leaf, b)))
    rec = vl.current_record(nb)
    g0 = hst(fm.held_cell_vegetation_factor(nb.root.matrix_op)).copy()
    law0 = [hst(a).copy() for a in nb.root.pref_op._law]
    kw = {"available_bytes_fn": g.rdr, "host_available_bytes_fn": lambda: BIG}
    # the next native event must be built with the BOUND cover: another cover is refused before the head moves; the bound one is accepted
    build = event_builder(g.ev, constant_rainfall(nb.t_s, nb.t_s + 0.05, 10.0))
    with pytest.raises(lc.LifecycleError, match="bound splash cover"):
        lc.begin_occurrence(g.lin, nb, lambda v, t: build(v, t, np.zeros(nb.surface_depth_m.shape)), **kw)
    assert g.lin.head is nb
    occ, st = lc.begin_occurrence(g.lin, nb, vl.event_factory_for(nb, build), **kw)
    assert occ.vegetation is rec and np.array_equal(hst(occ.event.veg), hst(rec.splash_cover_device))  # the ACTUAL device cover of the native event is the bound one
    fresh, prep = [], []
    real_fresh, real_prep = ncuda._device_op, fpi.prepare_floating_preferential
    monkeypatch.setattr(ncuda, "_device_op", lambda cfg, col, ks, quota, vegetation=None: (fresh.append(vegetation is not None), real_fresh(cfg, col, ks, quota, vegetation=vegetation))[1])
    monkeypatch.setattr(fpi, "prepare_floating_preferential", lambda *a, **k: (prep.append(1), real_prep(*a, **k))[1])
    h = bd_.ld.complete(occ, st)
    b2 = lc.close_occurrence(g.lin, occ, occ.event.completion_capability(h, h.state), h.state, **kw)
    assert b2.root.generation == nb.root.generation + 1  # the actual refresh advanced the property generation (no manual generation)
    assert fresh == [True] and prep == []  # the held device g reached the refreshed operator; NO contribution was re-streamed
    assert b2.root.vegetation is rec and rec.generation == 1
    assert hst(fm.held_cell_vegetation_factor(b2.root.matrix_op)).tobytes() == g0.tobytes()
    assert all(hst(a).tobytes() == c.tobytes() for a, c in zip(b2.root.pref_op._law, law0, strict=True))
    a0, a2 = vl.vegetation_attribution(nb), vl.vegetation_attribution(b2)
    assert a0["recipe_id"] == a2["recipe_id"] == rec.recipe_id and a2["generation"] == 1  # stable identity unchanged, counter not bumped
    assert a2["current_column_sha256"] == b2.root.column.sha256 and a2["current_matrix_op_sha256"] == b2.root.matrix_op.sha256 and a2["current_property_generation"] == b2.root.generation  # CURRENT ids from the actual holder
    assert a2["applied"]["column_sha256"] == nb.root.column.sha256 and a2["applied"]["matrix_op_sha256"] == nb.root.matrix_op.sha256  # apply-time ids stay provenance


# ----------------------------------------------------------------------------------------------------------------------------------------------
# 5 bound E-shading on the GPU day vs a prescribed control; a caller cover is refused
# ----------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_the_gpu_day_uses_the_resident_bound_shading_like_a_prescribed_cover_control(gpu, leaf, tmp_path, windowed):
    from maple_syrup import floating_event_daily_bridge as br
    from maple_syrup import recharge as rc
    from maple_syrup import vegetation_lifecycle as vl

    bd_ = gpu.m.bd
    g, c = bd_.matched_pair(tmp_path, bd_.dry_kwargs(), windowed)
    nb = gapply(g, g.b, vb.binding(leaf, g.b, vb.stack(leaf, g.b)))
    rec = vl.current_record(nb)
    n = nb.root.column.n_cells
    pet = g.xp.full(n, 3e-8)
    pl = rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=bd_.RUN, ordinal=0)
    st = br.open_day(g.lin, nb, partition=bd_.PART, run_digest=bd_.RUN, available_bytes_fn=g.rdr, **g.hk)
    j0, i0 = hst(st.history.wetting_m).copy(), hst(st.history.cumulative_intake_m).copy()
    with pytest.raises(br.BridgeError, match="bound"):
        br.day_action(st, plan=pl, pet_terrain_m_s=pet, cover=g.xp.full(n, 0.3), available_bytes_fn=g.rdr, **g.hk)  # a caller device cover is refused
    res = br.day_action(st, plan=pl, pet_terrain_m_s=pet, cover=None, available_bytes_fn=g.rdr, **g.hk)  # the resident device shading cover
    assert res.outcome == "interevent"
    # the matched CPU control: NO binding, the same live shading cover prescribed explicitly (host canonical copy of the record's cover)
    stc = br.open_day(c.lin, c.b, partition=bd_.PART, run_digest=bd_.RUN, available_bytes_fn=c.rdr)
    resc = br.day_action(stc, plan=pl, pet_terrain_m_s=np.full(n, 3e-8), cover=np.array(rec.shade_cover, copy=True), available_bytes_fn=c.rdr)
    np.testing.assert_allclose(hst(res.state.water.water_m), np.asarray(resc.state.water.water_m), **TOL)
    assert res.state.ledger.evaporation_m3 == pytest.approx(resc.state.ledger.evaporation_m3, rel=1e-9) and res.state.ledger.evaporation_m3 > 0.0
    assert hst(res.state.history.wetting_m).tobytes() == j0.tobytes() and hst(res.state.history.cumulative_intake_m).tobytes() == i0.tobytes()  # zero extra intake in a dry day


# ----------------------------------------------------------------------------------------------------------------------------------------------
# 6 hot loop: no vegetation transfer / hash / projection / allocator growth per hydraulic step
# ----------------------------------------------------------------------------------------------------------------------------------------------
def test_native_gpu_steps_after_apply_perform_no_vegetation_work_and_the_allocator_does_not_grow(gpu, leaf, tmp_path, monkeypatch):
    from maple_syrup import floating_event_lifecycle as lc
    from maple_syrup import native_floating_soil_cuda as ncuda
    from maple_syrup import vegetation_lifecycle as vl
    from maple_syrup import vegetation_lifecycle_cuda as vlc
    from maple_syrup import water_vegetation_binding as wvb
    from maple_syrup.rainfall import constant_rainfall

    cp, ld = gpu.cp, gpu.m.ld
    bd_ = gpu.m.bd
    L, _c = bd_.matched_pair(tmp_path, bd_.dry_kwargs(), False)  # the bounded matched dry boundary (genuinely completes quickly)
    nb = gapply(L, L.b, vb.binding(leaf, L.b, vb.stack(leaf, L.b)))
    build = event_builder(L.ev, constant_rainfall(nb.t_s, nb.t_s + 1.0, ld.LIGHT_RAIN_MM_H))
    occ, st = lc.begin_occurrence(L.lin, nb, vl.event_factory_for(nb, build), available_bytes_fn=L.rdr, host_available_bytes_fn=lambda: BIG)
    spies = []
    for mod, name in ((vlc, "hash_device_surface"), (vlc, "finish_cuda"), (ncuda, "_up"), (wvb, "prepare_vegetation_law"), (vl, "apply_vegetation_generation")):
        real = getattr(mod, name)
        monkeypatch.setattr(mod, name, lambda *a, _n=name, _r=real, **k: (spies.append(_n), _r(*a, **k))[1])
    pool = cp.get_default_memory_pool()
    s = st
    for _ in range(2):  # warm the pool
        s = occ.event.step(s, ld.LIGHT_RAIN_MM_H / 3.6e6, 0.1).state
    cp.cuda.Stream.null.synchronize()
    used, total = pool.used_bytes(), pool.total_bytes()
    for _ in range(4):
        s = occ.event.step(s, ld.LIGHT_RAIN_MM_H / 3.6e6, 0.1).state
    cp.cuda.Stream.null.synchronize()
    assert spies == []  # no surface hash, no cold upload, no law preparation and no re-application per hydraulic step
    assert pool.used_bytes() <= used + 0 and pool.total_bytes() <= total  # no allocator growth in the hot loop (the states of earlier steps are released)


# ----------------------------------------------------------------------------------------------------------------------------------------------
# 7 weak lifetimes
# ----------------------------------------------------------------------------------------------------------------------------------------------
def test_weak_lifetimes_release_the_old_root_operators_and_plants_but_keep_the_borrowed_derived_projection(gpu, leaf, tmp_path):
    from maple_syrup import vegetation_lifecycle as vl

    g, _c = gpu.m.bd.matched_pair(tmp_path, gpu.m.bd.dry_kwargs(), False)
    b = g.b
    s = vb.stack(leaf, b)
    binding = vb.binding(leaf, b, s)
    refs = {"state": weakref.ref(s["state"]), "old_root": weakref.ref(b.root), "old_matrix": weakref.ref(b.root.matrix_op), "old_history": weakref.ref(b.root.history)}
    proj = weakref.ref(s["proj"])
    nb = gapply(g, b, binding)
    del s, binding, b
    g.b = None
    for _ in range(3):
        gc.collect()
    assert {k: r() is None for k, r in refs.items()} == dict.fromkeys(refs, True)
    assert proj() is not None and vl.verify_record(vl.current_record(nb)) is not None  # the borrowed derived projection (no plants) lives through the one record


# ----------------------------------------------------------------------------------------------------------------------------------------------
# 8 a SMALL actual device peak vs the declared model
# ----------------------------------------------------------------------------------------------------------------------------------------------
def test_a_small_actual_device_peak_is_within_the_declared_model_plus_allocator_rounding(gpu, leaf, tmp_path):
    cp = gpu.cp
    from cupy.cuda import memory_hook

    class Peak(memory_hook.MemoryHook):
        name = "veg_lifecycle_peak"

        def __init__(self):
            self.live, self.peak, self.n = 0, 0, 0

        def malloc_postprocess(self, **kw):
            self.live += int(kw["mem_size"])
            self.n += 1
            self.peak = max(self.peak, self.live)

        def free_postprocess(self, **kw):
            self.live -= int(kw["mem_size"])

    g, _c = gpu.m.bd.matched_pair(tmp_path, gpu.m.bd.dry_kwargs(), False)
    b = g.b
    binding = vb.binding(leaf, b, vb.stack(leaf, b))
    est = sup.apply_estimate(b, binding)
    cp.cuda.Stream.null.synchronize()
    hook = Peak()
    with hook:
        gapply(g, b, binding)
        cp.cuda.Stream.null.synchronize()
    slack = 512 * hook.n  # the declared model excludes allocator rounding: an explicit per-allocation slack (512 B blocks)
    assert hook.peak <= est["peak_device_bytes"] + slack, (hook.peak, est["peak_device_bytes"], slack)  # N = 12, L = 3: SMALL; a moderate-scale peak needs a larger fixture (Root)


def test_the_public_device_surface_hash_over_many_chunks_owns_at_most_one_declared_chunk_buffer(gpu):
    """Pure PUBLIC component (no head, no model): a device surface of N = 300000 cells (> 131072 = one 1 MiB chunk, so MANY chunks) allocated BEFORE the tracer; imports / context / kernels warmed OUTSIDE the measurement. Counts 8 N bytes of
    device-to-host traffic and requires the traced OWNED host peak (numpy allocations) to be <= the admitted host budget = the declared chunk buffer + the native allowance (Root's confirmed defect: the previous chunk stayed bound while the
    next copy allocated, 2 chunks = 2 098 724 B > 1 114 112 B)."""
    import hashlib
    import tracemalloc

    from maple_syrup import floating_soil_layers_cuda as fcu
    from maple_syrup import vegetation_lifecycle_cuda as vlc

    cp = gpu.cp
    n = 300000
    surface = cp.arange(n, dtype=cp.float64).reshape(1, n)
    cp.cuda.Stream.null.synchronize()
    admitted = vlc.SURFACE_HASH_CHUNK_BYTES + fcu.PY_ALLOWANCE_BYTES
    q = {"memory_budget_bytes": 0, "reserve_bytes": 0, "available_bytes_fn": lambda: BIG, "host_memory_budget_bytes": admitted, "host_reserve_bytes": 0, "host_available_bytes_fn": lambda: BIG}
    vlc.hash_device_surface(cp, hashlib.sha256(), surface[:, :10], (1, 10), q)  # warm: imports, context, copy path (outside the measurement)
    h = hashlib.sha256()
    tracemalloc.start()
    moved = vlc.hash_device_surface(cp, h, surface, surface.shape, q)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert h.hexdigest() == hashlib.sha256(np.arange(n, dtype=np.float64).tobytes()).hexdigest()
    assert moved == 8 * n and n > vlc.SURFACE_HASH_CHUNK_BYTES // 8  # many chunks
    assert peak <= admitted, (peak, admitted)  # at most ONE declared chunk buffer plus small headers


def test_the_public_device_surface_hash_refuses_below_and_accepts_at_the_exact_host_need_for_a_small_and_a_many_chunk_surface(gpu):
    """Host-cap guard of the PUBLIC component: the admitted host need is `live + surface_hash_chunk_bytes(N)` (+ the native 64 KiB allowance), `min(8 N, 1 MiB)` -- a 12-cell surface needs 96 B, not a fixed 1 MiB. need - 1 refuses with the public
    typed error BEFORE any transfer (counted `to_host` calls = 0); the exact need passes and moves 8 N bytes."""
    import hashlib

    from maple.core import backend as mb

    from maple_syrup import floating_soil_layers_cuda as fcu
    from maple_syrup import vegetation_lifecycle as vl
    from maple_syrup import vegetation_lifecycle_cuda as vlc

    cp = gpu.cp
    for n in (12, 300000):
        surface = cp.arange(n, dtype=cp.float64).reshape(1, n)
        cp.cuda.Stream.null.synchronize()
        live = 1000
        need = live + vlc.surface_hash_chunk_bytes(n) + fcu.PY_ALLOWANCE_BYTES
        assert vlc.surface_hash_chunk_bytes(n) == min(8 * n, vlc.SURFACE_HASH_CHUNK_BYTES)

        def quota(budget):
            return {"memory_budget_bytes": 0, "reserve_bytes": 0, "available_bytes_fn": lambda: BIG, "host_memory_budget_bytes": budget, "host_reserve_bytes": 0, "host_available_bytes_fn": lambda: BIG}

        calls = []
        real = mb.to_host
        mb.to_host = lambda a, *x, _c=calls, _r=real, **k: (_c.append(1), _r(a, *x, **k))[1]
        try:
            with pytest.raises(vl.VegetationLifecycleError, match="nothing was allocated"):
                vlc.hash_device_surface(cp, hashlib.sha256(), surface, surface.shape, quota(need - 1), live)
            assert calls == []
            h = hashlib.sha256()
            assert vlc.hash_device_surface(cp, h, surface, surface.shape, quota(need), live) == 8 * n
        finally:
            mb.to_host = real
        assert h.hexdigest() == hashlib.sha256(np.arange(n, dtype=np.float64).tobytes()).hexdigest()
