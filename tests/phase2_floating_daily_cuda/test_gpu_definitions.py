"""Actual-GPU test DEFINITIONS for the CUDA floating daily lineage and its every-N output (WRITTEN, NOT RUN by their author: no GPU was launched or probed, no CuPy was imported, no NVRTC compile happened, no nvidia-smi).
Opt-in (SYRUP_RELEASE_GPU=1); Root runs them under its device guard with `.venv-cupy` and the pinned native source. CPU/NumPy and the CUDA lineage run the SAME bed, forcing and plans; contracts: rtol 2e-12 / atol 1e-14 for
water (the existing FSL/FT contracts), exact classification/cursor/clock/ledger identities, J/I objects unchanged. Transfers are counted through MAPLE's `to_host`/`to_device`. The second-device branch is honestly untestable
under a one-UUID mask (only the single-device gate is exercised). No full-storm / Pond / Wind / Chastre / day-after-storm claim: pure columns only."""
# ruff: noqa: RUF059
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_T = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e3_builders", _T / "phase2_native_floating_soil" / "builders.py")
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)
_dspec = importlib.util.spec_from_file_location("dbuild_gpu", _T / "phase2_floating_daily" / "dbuild.py")

from maple_syrup import floating_daily as fd
from maple_syrup import floating_daily_cuda as fdc
from maple_syrup import floating_daily_output as fo
from maple_syrup import floating_moisture_frame as g1
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import native_floating_soil as nfs
from maple_syrup import native_floating_soil_cuda as ncuda
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup import soil_moisture_output as g0

RT, AT = 2e-12, 1e-14
RUN = "ab" * 32
PART = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
BIG = 1 << 40
AVAIL = lambda: BIG


@pytest.fixture(scope="module")
def cp():
    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA daily lineage is not exercised (no claim)")
    return backend.cupy_module()


def device_bed(cp, bed):
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    return nfs.NativeBed(VoxelColumnState(mass_kg=cp.asarray(bed.voxel_column.mass_kg)), ActiveLayerState(mass_kg=cp.asarray(bed.active_layer.mass_kg)), bed.geometry, bed.grain_classes,
                         bed.mass_resolution_kg, bed.active_mask)


def host(x):
    return ncuda.inspect_counted(x) if hasattr(x, "device") else np.asarray(x)


def pair(cp, *, layers=b.LAYERS, sat=0.5, bottom="open", shallow=((0, 2),), **cfg):
    """`(cpu_state, cuda_state, device_pet, device_cover, host_pet, host_cover)` on the SAME physical bed/forcing (actual NativeOwner, both backends)."""
    bed = b.stratified(shallow=shallow)
    cpu = nfs.initialize_native_floating_soil(bed, b.config(backend="numpy", layers=layers, bottom=bottom, **cfg), initial_relative_saturation=sat, available_bytes_fn=AVAIL)
    dbed = device_bed(cp, bed)
    gpu = nfs.initialize_native_floating_soil(dbed, b.config(backend="cuda", layers=layers, bottom=bottom, **cfg), initial_relative_saturation=sat)
    sc = fd.start_floating_daily_lineage(cpu, bed, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    sg = fd.start_floating_daily_lineage(gpu, dbed, partition=PART, run_digest=RUN)  # CUDA: device reader/host reader default to the REAL current readings
    n = sc.ctx.n_cells
    hp, hc = np.full(n, 3e-8), np.full(n, 0.3)
    return sc, sg, cp.asarray(hp), cp.asarray(hc), hp, hc


def plan(p, i, t, o):
    return rc.plan_daily_rain(p, i, t, run_digest=RUN, ordinal=o)


def act_pair(sc, sg, dp, dc, hp, hc, p=0.0, i=0.0, t=5.0):
    pl_c, pl_g = plan(p, i, t, sc.cursor.ordinal), plan(p, i, t, sg.cursor.ordinal)
    rc_ = fd.floating_daily_action(sc, expected_parent=fd.expect_parent(sc), plan=pl_c, pet_terrain_m_s=hp, cover=hc, available_bytes_fn=AVAIL)
    rg = fd.floating_daily_action(sg, expected_parent=fd.expect_parent(sg), plan=pl_g, pet_terrain_m_s=dp, cover=dc)
    return rc_, rg


@pytest.mark.parametrize("layers", [(0.16,), (0.01, 0.05, 0.10), tuple([0.16 / 65] * 65)])
@pytest.mark.parametrize("sat", [0.15, 0.6, 0.95])
@pytest.mark.parametrize("bottom", ["open", "closed"])
def test_cpu_and_cuda_days_agree_and_the_lineage_contracts_hold(cp, layers, sat, bottom):
    """Three consecutive occurrences (P0, 0<P<T, P=T) on the SAME physical bed/forcing. Per day the child clock must be the ACCEPTED PARENT's clock + 86400 s (172800 on the second day, 259200 on the third: the model clock is
    cumulative), J/I exact, the cursor consumed once per completed day; a day that needs the full storm leaves both lineages unchanged and identical in outcome."""
    sc, sg, dp, dc, hp, hc = pair(cp, layers=layers, sat=sat, bottom=bottom)
    assert sg.ctx.column.backend != "numpy" and sg.ctx.depth_m.device is not None and sg.ctx.active.device is not None  # W, D and the mask are ONE device's resident arrays
    assert isinstance(sg.ctx.edges_m, np.ndarray) and isinstance(sg.ctx.n_active, int)  # shared edges and the active count stay HOST
    completed = 0
    for p, i, t in ((0.0, 0.0, 5.0), (1.0, 2.0, 5.0), (5.0, 4.0, 5.0)):  # P0, 0<P<T, P=T
        t_c, t_g = sc.t_s, sg.t_s
        assert t_c == t_g  # the lineages start the day at the same accepted clock
        rcpu, rgpu = act_pair(sc, sg, dp, dc, hp, hc, p, i, t)
        assert rcpu.outcome == rgpu.outcome and rcpu.capacity_failure == rgpu.capacity_failure and rcpu.branch == rgpu.branch
        if rcpu.outcome == "requires_full_storm":
            assert rgpu.state is None and rgpu.forcing.p_mm == rcpu.forcing.p_mm and rgpu.forcing.digest == rcpu.forcing.digest  # FULL native forcing, unchanged parent
            assert sg.t_s == t_g and sg.cursor.ordinal == sc.cursor.ordinal  # nothing applied
            continue
        np.testing.assert_allclose(host(rgpu.state.water.water_m), rcpu.state.water.water_m, rtol=RT, atol=AT)
        assert rcpu.state.t_s == t_c + 86400.0 and rgpu.state.t_s == t_g + 86400.0  # accepted parent clock + one day (NOT an absolute 86400 on later days)
        assert rgpu.state.history is sg.history and rcpu.state.history is sc.history  # J/I untouched (matrix/pref not called)
        for f in ("last_day_rain_m3", "last_day_evaporation_m3", "last_day_drainage_m3"):
            assert getattr(rgpu.state.ledger, f) == pytest.approx(getattr(rcpu.state.ledger, f), rel=1e-10, abs=1e-14)
        sc = fd.close_floating_day(rcpu.state, expected_parent=fd.expect_parent(rcpu.state)).state
        sg = fd.close_floating_day(rgpu.state, expected_parent=fd.expect_parent(rgpu.state)).state
        completed += 1
        assert sc.cursor.ordinal == sg.cursor.ordinal == completed and sc.t_s == t_c + 86400.0 and sg.t_s == t_g + 86400.0
    assert sc.cursor.ordinal == sg.cursor.ordinal


def test_storm_branch_and_the_one_heterogeneous_saturated_column_fallback_preserve_the_original(cp):
    sc, sg, dp, dc, hp, hc = pair(cp, sat=0.9, bottom="closed", shallow=((0, 2),))
    w0 = host(sg.water.water_m).copy()
    for p in (20.0, 6.0):  # P > T and a recharge too large for the shallow closed column
        rcpu, rgpu = act_pair(sc, sg, dp, dc, hp, hc, p, 4.0, 5.0 if p > 6.0 else 50.0)
        assert rcpu.outcome == rgpu.outcome and rcpu.branch == rgpu.branch and rgpu.state is None
    assert np.array_equal(host(sg.water.water_m), w0) and sg.t_s == 0.0 and sg.cursor.ordinal == 0 and sg.cursor.pending is None


def test_aggregate_withdrawal_is_resident_proportional_with_no_compensation(cp):
    sc, sg, dp, dc, hp, hc = pair(cp, sat=0.8)
    rcpu, rgpu = act_pair(sc, sg, dp, dc, hp, hc)
    n, L = sg.ctx.n_cells, sg.ctx.n_layers
    from maple_syrup import floating_transpiration as ft

    av = host(ft.floating_layer_availability_m(rgpu.state.ctx.column, rgpu.state.water))
    req = np.zeros((L, n))
    req[0, 1], req[1, 1] = 2.0 * av[0, 1], 0.25 * av[1, 1]
    out_g = fd.request_floating_withdrawal(rgpu.state, expected_parent=fd.expect_parent(rgpu.state), request_m=cp.asarray(req), expected_sequence=0)
    out_c = fd.request_floating_withdrawal(rcpu.state, expected_parent=fd.expect_parent(rcpu.state), request_m=req, expected_sequence=0, available_bytes_fn=AVAIL)
    assert hasattr(out_g.actual_m, "device") and hasattr(out_g.scale, "device")  # LEAF's results stay resident
    for g_, c_ in ((out_g.actual_m, out_c.actual_m), (out_g.unmet_m, out_c.unmet_m), (out_g.scale, out_c.scale)):
        np.testing.assert_allclose(host(g_), c_, rtol=RT, atol=AT)
    assert out_g.state.request_sequence == 1 and out_g.state.t_s == rgpu.state.t_s
    with pytest.raises(fd.DailyRefusal, match="sequence"):
        fd.request_floating_withdrawal(out_g.state, expected_parent=fd.expect_parent(out_g.state), request_m=cp.asarray(req), expected_sequence=0)
    for bad in (req, cp.asarray(req).astype(cp.float32), cp.asarray(req)[:, :-1].copy()):  # host array / dtype / shape: refused before any admission reading or allocation
        with pytest.raises(fd.DailyRefusal):
            fd.request_floating_withdrawal(out_g.state, expected_parent=fd.expect_parent(out_g.state), request_m=bad, expected_sequence=1, available_bytes_fn=lambda: pytest.fail("reader before metadata"))


def test_close_audit_is_resident_counts_exact_packets_and_refuses_changed_water(cp, monkeypatch):
    from maple.core import backend as mb

    sc, sg, dp, dc, hp, hc = pair(cp)
    r = fd.floating_daily_action(sg, expected_parent=fd.expect_parent(sg), plan=plan(0.0, 0.0, 5.0, 0), pet_terrain_m_s=dp, cover=dc)
    moved, up = [], []
    real_d, real_u = mb.to_host, mb.to_device
    monkeypatch.setattr(mb, "to_host", lambda x, *a, **k: moved.append(int(real_d(x, *a, **k).nbytes)) or real_d(x, *a, **k))
    monkeypatch.setattr(mb, "to_device", lambda *a, **k: up.append(1) or real_u(*a, **k))
    fd.close_floating_day(r.state, expected_parent=fd.expect_parent(r.state))
    assert up == [] and sorted(moved) == sorted([8 * fcu.cf.CHECK_WORDS, 8])  # the state-check packet (32 B) and the storage word (8 B): nothing of W is downloaded or hashed
    baseline = list(moved)
    # fault injection (labelled): a finite change of the resident water outside every transaction
    r.state.water.water_m[0, 1] += 1e-6
    with pytest.raises(fd.DailyRefusal, match="final close"):
        fd.close_floating_day(r.state, expected_parent=fd.expect_parent(r.state))
    r.state.water.water_m[0, 1] -= 1e-6
    assert moved[len(baseline):][:2] == baseline  # the refusal read only the same bounded packets


def test_two_pool_admission_zero_need_minus_one_exact_and_reserve_once(cp, monkeypatch):
    sc, sg, dp, dc, hp, hc = pair(cp)
    L, N = sg.ctx.n_layers, sg.ctx.n_cells
    dev, hst = fdc.day_plan(L, N, recharge=False)
    allocated = []
    real_empty = cp.empty
    monkeypatch.setattr(cp, "empty", lambda *a, **k: allocated.append(1) or real_empty(*a, **k))

    def run(dev_avail, host_avail, **kw):
        return fd.floating_daily_action(sg, expected_parent=fd.expect_parent(sg), plan=plan(0.0, 0.0, 5.0, 0), pet_terrain_m_s=dp, cover=dc, available_bytes_fn=lambda: dev_avail,
                                        host_available_bytes_fn=lambda: host_avail, **kw)

    for da, ha in ((0, BIG), (BIG, 0), (dev - 1, BIG), (BIG, hst + fcu.PY_ALLOWANCE_BYTES - 1)):
        with pytest.raises(fd.DailyRefusal) as ex:
            run(da, ha)
        assert ex.value.code == "memory"
    assert allocated == []  # refused BEFORE any device allocation
    run(dev, hst + fcu.PY_ALLOWANCE_BYTES)  # the exact declared needs of both pools
    with pytest.raises(fd.DailyRefusal):
        run(dev, hst + fcu.PY_ALLOWANCE_BYTES, reserve_bytes=1)  # each reserve counted once, in its own pool
    with pytest.raises(fd.DailyRefusal):
        run(dev + 5, hst + fcu.PY_ALLOWANCE_BYTES, reserve_bytes=6)
    run(dev + 5, hst + fcu.PY_ALLOWANCE_BYTES + 7, reserve_bytes=5, host_reserve_bytes=7)


def test_every_n_output_streams_exact_bytes_and_adds_no_traffic_when_not_due(cp, tmp_path, monkeypatch):
    from maple.core import backend as mb

    sc, sg, dp, dc, hp, hc = pair(cp, sat=0.6)
    lim = g0.MoistureOutputLimits(10, 10**8, 8192, 1 << 16, 10**8, 0, 0)
    (tmp_path / "o").mkdir()
    out = fo.FloatingMoistureOutput.create(tmp_path / "o", sg, expected_parent=fd.expect_parent(sg), every_n_days=2, limits=lim)
    moved = []
    real = mb.to_host
    monkeypatch.setattr(mb, "to_host", lambda x, *a, **k: moved.append(int(real(x, *a, **k).nbytes)) or real(x, *a, **k))
    counts, states = [], []
    st = sg
    for d in range(4):
        r = fd.floating_daily_action(st, expected_parent=fd.expect_parent(st), plan=plan(0.0, 0.0, 5.0, st.cursor.ordinal), pet_terrain_m_s=dp, cover=dc)
        base = len(moved)
        res = fd.close_floating_day(r.state, expected_parent=fd.expect_parent(r.state), output=out)
        counts.append((res.notice.action, moved[base:]))
        st = res.state
        states.append(st)
    audit_only = sorted([8 * fcu.cf.CHECK_WORDS, 8])
    assert counts[0][0] == "not_due" and sorted(counts[0][1]) == audit_only and counts[2][0] == "not_due" and sorted(counts[2][1]) == audit_only  # not due: ZERO extra traffic
    L, N = sg.ctx.n_layers, sg.ctx.n_cells
    for k in (1, 3):
        assert counts[k][0] == "saved" and sum(x for x in counts[k][1]) >= sum(audit_only)  # the due day adds the streamed W + D + mask chunks through G1 only
    fr = g1.read_floating_frame(tmp_path / "o" / g1.frame_name(2), max_return_bytes=10**8)
    np.testing.assert_array_equal(fr.water_m, host(states[1].water.water_m))  # the saved frame is the accepted day-2 W bit for bit (the G1 writer also hashes/compares the streamed bytes)
    np.testing.assert_array_equal(fr.depth_m, host(states[1].ctx.depth_m))
    assert fr.water_m.shape == (L, N) and fr.manifest["day"] == 2


def test_wrong_device_arrays_are_refused_before_any_work(cp):
    sc, sg, dp, dc, hp, hc = pair(cp)
    bad = (hp, dp.astype(cp.float32), dp[:-1].copy(), cp.asarray(np.zeros((2, sg.ctx.n_cells)))[0])
    for pet in bad[:3]:
        with pytest.raises(fd.DailyRefusal):
            fd.floating_daily_action(sg, expected_parent=fd.expect_parent(sg), plan=plan(0.0, 0.0, 5.0, 0), pet_terrain_m_s=pet, cover=dc, available_bytes_fn=lambda: pytest.fail("reader before metadata"))
    if cp.cuda.runtime.getDeviceCount() > 1:
        pytest.skip("a second device exists: the second-device branch must be exercised by Root's two-device run; it is untestable under a one-UUID mask")


class PeakAllocator:
    """REAL allocator hook (installed with `cp.cuda.using_allocator`): every device allocation of the daily trial goes through `pool.malloc(size)`; the TRANSIENT peak is the maximum of `pool.used_bytes() - baseline`
    sampled at each allocation and the ACTUAL allocator rounding is the sum of `ptr.mem.size - size`. Nothing here is a bound on the number of live allocations."""

    def __init__(self, cp):
        self.pool = cp.get_default_memory_pool()
        self.baseline = int(self.pool.used_bytes())
        self.peak = 0
        self.rounding = 0
        self.n = 0

    def __call__(self, size):
        ptr = self.pool.malloc(size)
        self.n += 1
        self.peak = max(self.peak, int(self.pool.used_bytes()) - self.baseline)
        self.rounding += int(ptr.mem.size) - int(size)
        return ptr


def moderate_pair(cp):
    """The moderate scalable REAL native fixture: ONE row of N = 50003 cells (`stratified_dense(shallow_cells=(), ny=1, nx=50003)`; NOT the pond-helper tilted default, whose capacity is exceeded on big grids),
    L = 3, the CUDA owner on the device bed, the default REAL device/host readers."""
    bed = b.bed_from(b.fx.stratified_dense(shallow_cells=(), ny=1, nx=50003))
    dbed = device_bed(cp, bed)
    gpu = nfs.initialize_native_floating_soil(dbed, b.config(backend="cuda"), initial_relative_saturation=0.5)
    sg = fd.start_floating_daily_lineage(gpu, dbed, partition=PART, run_digest=RUN)
    n = sg.ctx.n_cells
    assert n == 50003 and sg.ctx.n_layers == 3
    return sg, cp.full(n, 3e-8), cp.full(n, 0.3)


def sync_sample(cp):
    import gc

    cp.cuda.Device().synchronize()
    gc.collect()
    pool = cp.get_default_memory_pool()
    return int(pool.used_bytes()), int(pool.total_bytes())


def test_moderate_n50003_warm_transient_peak_live_end_and_released_parent_plateau(cp):
    """Warm day, then ten more days with every parent/trial dropped and the pool synchronised and collected between samples. THREE different quantities are measured and named as such:
      * `transient_peak` = the REAL allocator peak of one daily action (the `PeakAllocator` hook, parents live), admitted against the composed `fdc.day_plan` device bytes PLUS the ACTUAL allocator rounding the
        same hook recorded (`ptr.mem.size - size`): no arbitrary allowance and no live-allocation-count bound;
      * `live_end` = the pool's used bytes right after the action while the trial references are still live (the retained footprint of the result, NOT the transient peak; it can only be <= the peak);
      * the released-parent PLATEAU = used bytes after the parents/trial are dropped, constant over the days within one retained state (`8 L N`) plus the largest per-day actual rounding.
    No NVRTC/context claim."""
    sg, dp, dc = moderate_pair(cp)
    L, N = sg.ctx.n_layers, sg.ctx.n_cells
    dev_plan, _host_plan = fdc.day_plan(L, N, recharge=False)
    state_bytes = 8 * L * N
    st = sg
    r = fd.floating_daily_action(st, expected_parent=fd.expect_parent(st), plan=plan(0.0, 0.0, 5.0, 0), pet_terrain_m_s=dp, cover=dc)  # WARM day (kernels loaded, pool primed)
    st = fd.close_floating_day(r.state, expected_parent=fd.expect_parent(r.state)).state
    del r
    used_warm, _ = sync_sample(cp)
    used_series, roundings = [], []
    for d in range(1, 11):
        before, _ = sync_sample(cp)
        hook = PeakAllocator(cp)
        with cp.cuda.using_allocator(hook):
            r = fd.floating_daily_action(st, expected_parent=fd.expect_parent(st), plan=plan(0.0, 0.0, 5.0, st.cursor.ordinal), pet_terrain_m_s=dp, cover=dc)
        live_end, _ = sync_sample(cp)  # trial references STILL live: the LIVE-END footprint, not the peak
        assert hook.n > 0 and hook.baseline == before
        assert live_end - before <= hook.peak  # the live end can never exceed the true transient peak
        assert hook.peak <= dev_plan + hook.rounding, (d, hook.peak, dev_plan, hook.rounding)  # the composed plan + the ACTUAL rounding of this very day
        roundings.append(hook.rounding)
        st = fd.close_floating_day(r.state, expected_parent=fd.expect_parent(r.state)).state
        del r
        after, _ = sync_sample(cp)  # parents and trial dropped
        used_series.append(after)
    slack = state_bytes + max(roundings)
    assert max(used_series) - min(used_series) <= slack  # the plateau: no growth with days
    assert used_series[-1] - used_warm <= slack


def test_moderate_every_n_due_frame_is_bit_exact_bounded_and_counted(cp, tmp_path, monkeypatch):
    """N = 50003, N-cadence 2: day 1 not due (zero traffic beyond the audited close: the 32 B state check + the 8 B storage word), day 2 due (the audit + the streamed W + D + mask chunks through G1, no O(L N) readback
    other than G1's counted chunks); the sealed frame equals the accepted post-T W, canonical D and mask bit for bit (the caller's EXPLICIT frame reader is the only full read). Only `mb.to_host`/`mb.to_device` are
    instrumented: implicit backend boolean/scalar syncs are NOT claimed as counted."""
    from maple.core import backend as mb

    sg, dp, dc = moderate_pair(cp)
    L, N = sg.ctx.n_layers, sg.ctx.n_cells
    chunk = 1 << 16
    lim = g0.MoistureOutputLimits(10, 10**9, 8192, chunk, 10**9, 0, 0)
    (tmp_path / "o").mkdir()
    out = fo.FloatingMoistureOutput.create(tmp_path / "o", sg, expected_parent=fd.expect_parent(sg), every_n_days=2, limits=lim)
    moved, up = [], []
    real_d, real_u = mb.to_host, mb.to_device
    monkeypatch.setattr(mb, "to_host", lambda x, *a, **k: moved.append(int(real_d(x, *a, **k).nbytes)) or real_d(x, *a, **k))
    monkeypatch.setattr(mb, "to_device", lambda *a, **k: up.append(1) or real_u(*a, **k))
    audit = sorted([8 * fcu.cf.CHECK_WORDS, 8])
    states = []
    st = sg
    for d in range(2):
        r = fd.floating_daily_action(st, expected_parent=fd.expect_parent(st), plan=plan(0.0, 0.0, 5.0, st.cursor.ordinal), pet_terrain_m_s=dp, cover=dc)
        base = len(moved)
        res = fd.close_floating_day(r.state, expected_parent=fd.expect_parent(r.state), output=out)
        pieces = moved[base:]
        st = res.state
        states.append(st)
        if d == 0:
            assert res.notice.action == "not_due" and sorted(pieces) == audit  # not due: ZERO extra traffic
        else:
            assert res.notice.action == "saved"
            assert sum(pieces) == sum(audit) + 8 * L * N + 8 * N + N  # the audit + exactly the W, D and mask bytes, each streamed in bounded pieces
            assert max(pieces) <= chunk and len(pieces) > len(audit) + 2  # bounded chunks (G1 `chunk_bytes`), never one O(L N) block
    assert up == []  # no host-to-device transfer at all
    fr = g1.read_floating_frame(tmp_path / "o" / g1.frame_name(2), max_return_bytes=10**9)  # the caller's explicit, permitted full read
    final = states[1]
    np.testing.assert_array_equal(fr.water_m, host(final.water.water_m))  # the accepted post-T W of the audited close
    np.testing.assert_array_equal(fr.depth_m, cp.asnumpy(final.ctx.depth_m))
    np.testing.assert_array_equal(fr.active, cp.asnumpy(final.ctx.active))
    # The established contract: notifying the SAME already-notified closed state is an idempotent no-I/O repeat, NOT a verification.
    before_repeat = list(moved)
    again = out.notify_day_closed(final, expected_parent=fd.expect_parent(final))
    assert again.action == "repeat_same_state" and again.bytes_written == 0 and again.due
    assert moved == before_repeat  # the repeat moved nothing (the counted list did not grow)
    # The byte-verifying path is exercised by RESUMING the output at the accepted final boundary and notifying that day: the sealed due frame is verified against the live W/D/mask by G1.
    resumed = fo.FloatingMoistureOutput.resume(tmp_path / "o", final, expected_parent=fd.expect_parent(final), limits=lim)
    replay = resumed.notify_day_closed(final, expected_parent=fd.expect_parent(final))
    assert replay.action == "verified_existing" and replay.due and replay.path is not None
