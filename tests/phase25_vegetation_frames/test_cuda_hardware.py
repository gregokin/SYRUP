"""Phase 2.5 vegetation-aware frame / every-N output, CUDA HARDWARE tests: EXECUTABLE, AUTHORED, NOT RUN by their author (no CuPy / GPU / NVRTC / nvidia-smi touched). Root runs them with `SYRUP_VEG_FRAMES_RELEASE_GPU=1` (device
`SYRUP_VEG_FRAMES_GPU_DEVICE`, default 0); without the flag they skip at collection, with it a missing CuPy / device is a SETUP FAILURE. Fixtures: the qualified CUDA bridge `matched_pair` (a genuinely completed CUDA closed Boundary with a
matched CPU twin, Dense and published Windowed) through the lifecycle's own CUDA apply (real P1 + S0), then the real GPU daily path. The outer fixture selects the device once; no device is switched inside a test."""
# ruff: noqa: RUF059
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

RELEASE = os.environ.get("SYRUP_VEG_FRAMES_RELEASE_GPU") == "1"
DEVICE = int(os.environ.get("SYRUP_VEG_FRAMES_GPU_DEVICE", "0"))
pytestmark = pytest.mark.skipif(not RELEASE, reason="CUDA hardware tests: set SYRUP_VEG_FRAMES_RELEASE_GPU=1 on the allocated device (authored, not run by the author)")
BIG = 1 << 40
TOL = {"rtol": 1e-9, "atol": 1e-12}


@pytest.fixture(scope="module")
def gpu():
    try:
        import cupy as cp

        cp.cuda.Device(DEVICE).use()
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"SYRUP_VEG_FRAMES_RELEASE_GPU=1 but CuPy / a CUDA device is unusable: {exc}")
    import veg_lc_gpu_support as sup

    return SimpleNamespace(cp=cp, m=sup.modules())


def hst(a):
    from maple.core.backend import to_host

    return np.asarray(to_host(a))


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_a_gpu_vegetation_lineage_writes_n_equals_two_frames_from_device_arrays_like_its_cpu_twin(gpu, leaf, tmp_path, windowed):
    import veg_fr_support as S
    import veg_lc_builders as vb

    from maple_syrup import floating_daily_output_vegetation as fov
    from maple_syrup import floating_moisture_frame as g1
    from maple_syrup import floating_moisture_frame_vegetation as g2
    from maple_syrup import soil_moisture_output as g0

    bd_ = gpu.m.bd
    g, c = bd_.matched_pair(tmp_path, bd_.dry_kwargs(), windowed)
    lim = g0.MoistureOutputLimits(max_snapshots=10, max_total_bytes=10**7, max_manifest_bytes=16384, chunk_bytes=4096, host_memory_budget_bytes=10**7, host_reserve_bytes=0, disk_reserve_bytes=0)
    out_dirs = {}
    frames = {}
    for side, label in ((g, "gpu"), (c, "cpu")):
        sess0, _sess1, _cat, surf, mask, ctl, _nfx = S.session_pair(leaf, side.b)
        bind, _p = vb.session_binding(leaf, side.b, sess0, surf, mask, ctl)
        kw = {"memory_budget_bytes": BIG, "available_bytes_fn": side.rdr}
        if label == "gpu":
            kw.update(host_memory_budget_bytes=BIG, host_available_bytes_fn=lambda: BIG)
        nb = S.vl.apply_vegetation_generation(side.lin, side.b, bind, **kw)
        state = S.br.open_day(side.lin, nb, partition=S.PART, run_digest=S.RUN, available_bytes_fn=side.rdr, **side.hk)
        d = tmp_path / f"out_{label}"
        d.mkdir()
        out = fov.VegetationMoistureOutput.create(d, state, expected_parent=S.fd.expect_parent(state), every_n_days=2, limits=lim)
        for k in (1, 2):
            n, L = state.ctx.n_cells, state.ctx.n_layers
            pl = S.rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=S.RUN, ordinal=int(state.cursor.ordinal))
            res = S.br.day_action(state, plan=pl, pet_terrain_m_s=side.xp.full(n, 3e-8), cover=None, available_bytes_fn=side.rdr, **side.hk)
            t = S.fd.request_floating_withdrawal(res.state, expected_parent=S.fd.expect_parent(res.state), request_m=side.xp.zeros((L, n)), expected_sequence=0, available_bytes_fn=side.rdr, **side.hk)
            cr = S.fd.close_floating_day(t.state, expected_parent=S.fd.expect_parent(t.state), output=out, available_bytes_fn=side.rdr, host_available_bytes_fn=lambda: BIG)
            assert cr.notice.due == (k == 2) and cr.notice.action == ("saved" if k == 2 else "not_due")
            state = cr.state
        frames[label] = (g2.read_vegetation_frame(d / g1.frame_name(2), max_return_bytes=BIG), hst(state.water.water_m).copy(), S.vl.vegetation_attribution(state))
        out_dirs[label] = d
    fg, wg, ag = frames["gpu"]
    fc, wc, ac = frames["cpu"]
    assert np.array_equal(fg.water_m, wg)  # the GPU frame is the accepted device state's water (device -> host bounded chunks), bit for bit
    np.testing.assert_allclose(fg.water_m, fc.water_m, **TOL)  # and equals the matched CPU twin within the declared tolerance
    v = fg.manifest["vegetation"]
    assert v["generation"] == 1
    assert v["current_column_sha256"] == ag["current_column_sha256"] == fg.manifest["identity"]["property_sha256"] and v["recipe_id"] == ag["recipe_id"]
    assert fc.manifest["vegetation"]["generation"] == 1


def test_a_not_due_gpu_day_transfers_nothing_and_a_refused_host_pool_writes_nothing(gpu, leaf, tmp_path, monkeypatch):
    import veg_fr_support as S
    import veg_lc_builders as vb

    from maple_syrup import floating_daily_output_vegetation as fov
    from maple_syrup import floating_moisture_frame as g1
    from maple_syrup import soil_moisture_output as g0

    bd_ = gpu.m.bd
    g, _c = bd_.matched_pair(tmp_path, bd_.dry_kwargs(), False)
    sess0, _s1, _cat, surf, mask, ctl, _nfx = S.session_pair(leaf, g.b)
    bind, _p = vb.session_binding(leaf, g.b, sess0, surf, mask, ctl)
    nb = S.vl.apply_vegetation_generation(g.lin, g.b, bind, memory_budget_bytes=BIG, available_bytes_fn=g.rdr, host_memory_budget_bytes=BIG, host_available_bytes_fn=lambda: BIG)
    state = S.br.open_day(g.lin, nb, partition=S.PART, run_digest=S.RUN, available_bytes_fn=g.rdr, **g.hk)
    lim = g0.MoistureOutputLimits(max_snapshots=10, max_total_bytes=10**7, max_manifest_bytes=16384, chunk_bytes=4096, host_memory_budget_bytes=10**7, host_reserve_bytes=0, disk_reserve_bytes=0)
    d = tmp_path / "o"
    d.mkdir()
    out = fov.VegetationMoistureOutput.create(d, state, expected_parent=S.fd.expect_parent(state), every_n_days=2, limits=lim)
    pieces = []
    real = g1._to_host
    monkeypatch.setattr(g1, "_to_host", lambda a: (pieces.append(1), real(a))[1])
    n, L = state.ctx.n_cells, state.ctx.n_layers

    def one(state, **extra):
        pl = S.rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=S.RUN, ordinal=int(state.cursor.ordinal))
        res = S.br.day_action(state, plan=pl, pet_terrain_m_s=g.xp.full(n, 3e-8), cover=None, available_bytes_fn=g.rdr, **g.hk)
        t = S.fd.request_floating_withdrawal(res.state, expected_parent=S.fd.expect_parent(res.state), request_m=g.xp.zeros((L, n)), expected_sequence=0, available_bytes_fn=g.rdr, **g.hk)
        return t, extra

    t, _ = one(state)
    before = len(pieces)
    cr = S.fd.close_floating_day(t.state, expected_parent=S.fd.expect_parent(t.state), output=out, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)
    assert not cr.notice.due and len(pieces) == before  # a not-due GPU day: no array piece, no transfer
    t2, _ = one(cr.state)
    parent2 = S.fd.expect_parent(t2.state)
    # (1) NATIVE refusal: the final close audit itself needs host bytes and refuses (DailyRefusal[memory]) BEFORE the output callback; this is NOT the frame quota
    with pytest.raises(S.fd.DailyRefusal, match="final close audit") as ei:
        S.fd.close_floating_day(t2.state, expected_parent=parent2, output=out, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: 0)
    assert ei.value.code == "memory" and "only 0 B are reported available" in str(ei.value)
    assert len(pieces) == before and not (d / g1.frame_name(2)).exists() and sorted(p.name for p in d.iterdir()) == [fov.OUTPUT_NAME]  # nothing transferred, no frame piece/file
    assert out.last_day == 1  # cadence unchanged (the retry below uses the SAME accepted interevent parent)
    cr2 = S.fd.close_floating_day(t2.state, expected_parent=parent2, output=out, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)  # ONE successful retry with the same parent
    assert cr2.notice.due and cr2.notice.action == "saved" and (d / g1.frame_name(2)).is_dir() and out.last_day == 2
    # (2) OUTPUT-ONLY zero-host admission control on an ACTUAL genuinely closed state (closed without an output; the public writer is then notified directly)
    t3, _ = one(cr2.state)
    cr3 = S.fd.close_floating_day(t3.state, expected_parent=S.fd.expect_parent(t3.state), output=out, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)
    assert not cr3.notice.due
    t4, _ = one(cr3.state)
    closed4 = S.fd.close_floating_day(t4.state, expected_parent=S.fd.expect_parent(t4.state), output=None, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG).state
    mark = len(pieces)
    with pytest.raises(g0.MoistureOutputQuotaError):
        out.notify_day_closed(closed4, expected_parent=S.fd.expect_parent(closed4), host_available_bytes_fn=lambda: 0)
    assert len(pieces) == mark and not (d / g1.frame_name(4)).exists() and out.last_day == 3  # the frame quota itself: no transfer, no file, cadence unchanged
    ok = out.notify_day_closed(closed4, expected_parent=S.fd.expect_parent(closed4), host_available_bytes_fn=lambda: BIG)
    assert ok.due and ok.action == "saved" and (d / g1.frame_name(4)).is_dir() and len(pieces) > mark
