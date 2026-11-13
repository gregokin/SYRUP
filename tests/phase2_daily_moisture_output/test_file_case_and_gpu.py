"""G2a on the GENUINE native file-case factory (real NCP source verification, admit_package_case, prepare_native_dense_case, prepare_layered_case
from a YAML file; see tests/phase15_recharge/ncpcases.py) plus gated CUDA DEFINITIONS (written, not run by their author).

File-case tests are opt-in by EXPLICIT configuration (`SYRUP_NCP_TARGET`); when it IS configured a missing/wrong-origin bundle FAILS setup (the
genuine source guard is never skipped away), when it is unset the tests skip with that reason (no claim, no substitute)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase15_recharge"))
import ncpcases as nc
from dcases import RUN, dw, wl

from maple_syrup import daily_moisture_output as dmo
from maple_syrup import soil_moisture_output as mo

LIMITS = mo.MoistureOutputLimits(max_snapshots=20, max_total_bytes=1 << 30, max_manifest_bytes=8192, chunk_bytes=1 << 16,
                                 host_memory_budget_bytes=1 << 30, host_reserve_bytes=0, disk_reserve_bytes=0)


@pytest.fixture
def bundle():
    if nc.skip_reason() is not None:
        pytest.skip(nc.skip_reason())
    return nc.require_configured_bundle()  # a configured but unusable bundle FAILS here


def day(c, s, p, *, withdraw=0.0):
    plan = dw.next_plan(s, p, 3600.0, 1.0)
    r = dw.daily_water_action(c.lop, s, expected_parent=wl.expect_parent(s), plan=plan, pet_terrain_m_s=np.full(c.n, 1e-8))
    assert r.status == "complete"
    ie = r.state
    if withdraw:
        q = np.zeros((c.L, c.n))
        q[0] = withdraw
        ie, _ = wl.request_withdrawal(c.lop, ie, expected_parent=wl.expect_parent(ie), depth_m=q, identity="leaf")
    return ie


@pytest.mark.parametrize("n_layers", [1, 2])
def test_file_case_every_second_day_saves_post_withdrawal_water(bundle, tmp_path, n_layers):
    c = nc.build_file_case(tmp_path / "case", nc.yaml_for(n_layers))
    assert c.profile.maple_commit.startswith("9f466d2c")
    s = dw.start_daily_lineage(c.lop, c.s0, expected_parent=wl.expect_parent(c.s0), run_digest=RUN)
    (tmp_path / "out").mkdir()
    out = dmo.create_daily_moisture_output(tmp_path / "out", c.lop, s, expected_parent=wl.expect_parent(s), every_n_days=2, limits=LIMITS)
    actions = []
    for d, (p, w) in enumerate(((0.25, 1e-5), (0.0, 0.0), (0.25, 0.0)), 1):
        ie = day(c, s, p, withdraw=w)
        r = dmo.close_final_day(c.lop, ie, expected_parent=wl.expect_parent(ie), output=out)
        s = r.state
        actions.append(r.notice.action)
        assert r.notice.day == d
    assert actions == ["not_due", "saved", "not_due"]
    snap = mo.read_snapshot(tmp_path / "out" / mo.snapshot_name(2), max_bytes=1 << 22)
    assert snap.water_m.shape == (n_layers, c.n) and snap.manifest["day"] == 2 and snap.manifest["not_a_restart"] is True


GPU = pytest.mark.skipif(os.environ.get("SYRUP_RELEASE_GPU") != "1", reason="GPU tests are opt-in (SYRUP_RELEASE_GPU=1); written for Root, not run by the author")


@GPU
def test_cuda_extra_transfers_relative_to_the_direct_native_close(tmp_path):
    """CUDA DEFINITION (written, NOT run by its author). The existing native `close_interevent` performs its own audits and therefore its own small
    device-to-host reads (measured by Root: 64 B in 3 reads, 2 scalar); that baseline is NOT a diagnostic regression and is neither removed nor
    weakened. The contract is zero EXTRA transfers of the disabled/not-due diagnostics RELATIVE to a direct native close of the SAME accepted parent,
    measured with the ACTUAL MAPLE counters (every field compared), and for a due day exactly `8*L*N` extra device-to-host water bytes in exactly
    `L*ceil(N / (chunk_bytes // 8))` extra pieces with no extra host-to-device. Needs the configured native9f CUDA bundle (tests/phase15_recharge/gcases.py)."""
    import dataclasses
    import math

    import gcases as g
    from maple.core.backend import read_transfer_counters

    g.require_cuda_bundle()
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2))
    s = dw.start_daily_lineage(c.lop, c.s0, expected_parent=wl.expect_parent(c.s0), run_digest=RUN)
    (tmp_path / "out").mkdir()
    lim = mo.MoistureOutputLimits(max_snapshots=20, max_total_bytes=1 << 30, max_manifest_bytes=8192, chunk_bytes=8 * 16,
                                  host_memory_budget_bytes=1 << 30, host_reserve_bytes=0, disk_reserve_bytes=0)
    out = dmo.create_daily_moisture_output(tmp_path / "out", c.lop, s, expected_parent=wl.expect_parent(s), every_n_days=2, limits=lim)

    def occurrence(state):
        plan = dw.next_plan(state, 2.0, 10.0, 5.0)
        return dw.daily_water_action(c.lop, state, expected_parent=wl.expect_parent(state), plan=plan, pet_terrain_m_s=g.pet(c, 2.0 ** -24)).state

    def measured(fn):
        t0 = read_transfer_counters()
        res = fn()
        return res, read_transfer_counters().delta(t0)

    def baseline(ie):  # the direct genuine native close of the SAME accepted parent
        return measured(lambda: wl.close_interevent(c.lop, ie, expected_parent=wl.expect_parent(ie)))

    def facade(ie, output):
        return measured(lambda: dmo.close_final_day(c.lop, ie, expected_parent=wl.expect_parent(ie), output=output))

    L, N = c.col.n_layers, c.col.n_cells
    ie1 = occurrence(s)
    _, base1 = baseline(ie1)
    r_off, d_off = facade(ie1, None)
    assert r_off.notice is None and dataclasses.astuple(d_off) == dataclasses.astuple(base1)  # off: identical to the native baseline in every field
    r1, d1 = facade(ie1, out)  # day 1 of N = 2: not due -> still identical (no extra array download of any kind)
    assert r1.notice.action == "not_due" and dataclasses.astuple(d1) == dataclasses.astuple(base1)
    ie2 = occurrence(r1.state)
    _, base2 = baseline(ie2)
    r2, d2 = facade(ie2, out)  # day 2: due
    per = lim.chunk_bytes // 8
    pieces = L * math.ceil(N / per)
    assert r2.notice.action == "saved" and r2.notice.device_chunks == pieces
    assert d2.device_to_host_bytes - base2.device_to_host_bytes == 8 * L * N
    assert d2.device_to_host - base2.device_to_host == pieces
    assert d2.host_to_device == base2.host_to_device and d2.host_to_device_bytes == base2.host_to_device_bytes


def _numba_ok() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(sys.platform != "linux", reason="fresh-process file-case fixtures are Linux")
@pytest.mark.parametrize("backend", ["numpy"] + (["numba"] if _numba_ok() else []))
def test_file_case_fresh_process_v3_resume_replays_days_2_to_4_against_sealed_future_frames(bundle, tmp_path, backend):
    """Actual native file-case factory in TWO fresh processes (own factory_full / factory_resume source-guard paths): the full run seals frames 2 and 4 and
    saves the V3 checkpoint at day 1; the resume process loads it, replays days 2..4 THROUGH THE FACADE and must reproduce the physical/state/water
    hashes while VERIFYING (never rewriting) the sealed future frames. Recorded source identities: the imported MAPLE file and commit, the SYRUP tree digest."""
    import json
    import subprocess

    worker = Path(__file__).resolve().parent / "file_case_worker.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    run = lambda mode: subprocess.run([sys.executable, "-B", str(worker), mode, backend, str(tmp_path)], capture_output=True, text=True,
                                      env=env, timeout=1800, check=False)
    a = run("full")
    assert a.returncode == 0, a.stderr[-3000:]
    full = json.loads((tmp_path / "full.json").read_text())
    sealed = {p.name: (p / "manifest.json").read_bytes() for p in (tmp_path / "moisture").iterdir() if p.name.startswith("day_")}
    assert sorted(sealed) == [mo.snapshot_name(2), mo.snapshot_name(4)]
    b = run("resume")
    assert b.returncode == 0, b.stderr[-3000:]
    res = json.loads((tmp_path / "resume.json").read_text())
    assert [t["action"] for t in full["trace"]] == ["not_due", "saved", "not_due", "saved"]
    assert [t["action"] for t in res["trace"]] == ["verified_existing", "not_due", "verified_existing"]
    assert res["trace"] == [{**t, "action": a_} for t, a_ in zip(full["trace"][1:], [x["action"] for x in res["trace"]], strict=True)]
    assert sealed == {p.name: (p / "manifest.json").read_bytes() for p in (tmp_path / "moisture").iterdir() if p.name.startswith("day_")}
    assert full["maple_file"] == res["maple_file"] and full["syrup_tree_digest"] == res["syrup_tree_digest"] and full["maple_commit"].startswith("9f466d2c")


def test_misconfigured_bundle_fails_instead_of_skipping(monkeypatch, tmp_path):
    monkeypatch.setenv("SYRUP_NCP_TARGET", str(tmp_path / "no_such_bundle"))
    assert nc.skip_reason() is None  # a configured target is never skipped away
    with pytest.raises(nc.ConfiguredBundleError):
        nc.require_configured_bundle()
