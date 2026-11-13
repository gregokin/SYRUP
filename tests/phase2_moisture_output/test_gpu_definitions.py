"""Actual-GPU test DEFINITIONS for the moisture output (WRITTEN, NOT RUN by their author: no GPU launch, probe, CuPy device initialisation or NVRTC). Opt-in like the Phase 1.5 CUDA lane:
SYRUP_RELEASE_GPU=1 and the configured native9f CUDA bundle (see tests/phase15_recharge/gcases.py); GPU0 only under Root's fresh guard. They assert, through the ACTUAL MAPLE transfer counters, that a
not-due day makes zero device-to-host transfers, a due day downloads exactly the bounded pieces (total == the array bytes, each piece <= chunk_bytes), the saved bytes equal the device array, the CURRENT-device
gate fires before any reader/copy, and a CUDA-saved snapshot equals what the CPU writer produces for the same host array bytes (same array bytes -> same snapshot bytes; not a physics identity claim)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase15_recharge"))
import gcases as g  # noqa: E402
from conftest import limits, mo  # noqa: E402
from dcases import RUN  # noqa: E402

from maple_syrup import daily_water as dw  # noqa: E402
from maple_syrup import water_lifecycle as wl  # noqa: E402

pytestmark = pytest.mark.usefixtures("gpu")


@pytest.fixture(autouse=True)
def _bundle(gpu):  # noqa: ARG001
    g.require_cuda_bundle()


def close_day(c, state, p):
    plan = dw.next_plan(state, p, 10.0, 5.0)
    r = dw.daily_water_action(c.lop, state, expected_parent=wl.expect_parent(state), plan=plan, pet_terrain_m_s=g.pet(c, 2.0 ** -24))
    return wl.close_interevent(c.lop, r.state, expected_parent=wl.expect_parent(r.state))


def start(c):
    return dw.start_daily_lineage(c.lop, c.s0, expected_parent=wl.expect_parent(c.s0), run_digest=RUN)


def test_counted_transfers_bounded_pieces_and_exact_bytes(tmp_path):
    from maple.core.backend import read_transfer_counters

    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2))
    (tmp_path / "out").mkdir()
    s0 = start(c)
    L, N = c.col.n_layers, c.col.n_cells
    lim = limits(chunk_bytes=8 * 16)
    out = mo.MoistureOutput.create(tmp_path / "out", c.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=2, limits=lim)
    s1 = close_day(c, s0, 2.0)
    t0 = read_transfer_counters()
    n1 = out.notify_day_closed(c.lop, s1, expected_parent=wl.expect_parent(s1))
    assert n1.action == "not_due" and read_transfer_counters().delta(t0).device_to_host == 0  # zero downloads on a not-due day
    s2 = close_day(c, s1, 0.0)
    t1 = read_transfer_counters()
    n2 = out.notify_day_closed(c.lop, s2, expected_parent=wl.expect_parent(s2))
    d = read_transfer_counters().delta(t1)
    pieces = L * ((N + 15) // 16)
    assert n2.action == "saved" and n2.device_chunks == pieces and d.device_to_host == pieces and d.device_to_host_bytes == 8 * L * N  # bounded pieces, never a second full mirror
    snap = mo.read_snapshot(n2.path, max_bytes=1 << 22)
    assert snap.water_m.tobytes() == s2.payload.soil.water_m.get().tobytes()
    # the same array bytes through the CPU path give the same file bytes
    import hashlib

    head = mo._npy_header((L, N), "<f8")
    assert hashlib.sha256(head + s2.payload.soil.water_m.get().tobytes()).hexdigest() == snap.manifest["water"]["file_sha256"]


def test_current_device_gate_before_reader_or_copy(tmp_path):
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs two visible devices (no claim)")
    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2))
    (tmp_path / "out").mkdir()
    s0 = start(c)
    out = mo.MoistureOutput.create(tmp_path / "out", c.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=1, limits=limits())
    s1 = close_day(c, s0, 2.0)
    reads = []
    with cp.cuda.Device(1):
        with pytest.raises(mo.MoistureOutputError):
            out.notify_day_closed(c.lop, s1, expected_parent=wl.expect_parent(s1), host_available_bytes_fn=lambda: reads.append(1) or (1 << 40))
    assert not reads and out.last_day == 0
    assert out.notify_day_closed(c.lop, s1, expected_parent=wl.expect_parent(s1)).action == "saved"


def test_static_mask_is_a_separate_bounded_cold_transfer(tmp_path):
    """Gated (correction01): `create` streams the mask in chunk_bytes pieces through the counted path (D2H count == ceil(N/chunk), bytes == N) BEFORE any due-day water; unsaved days stay at zero."""
    from maple.core.backend import read_transfer_counters

    c = g.build_cuda_case(tmp_path, g.cuda_yaml(2))
    (tmp_path / "out").mkdir()
    s0 = start(c)
    N = c.col.n_cells
    t0 = read_transfer_counters()
    out = mo.MoistureOutput.create(tmp_path / "out", c.lop, s0, expected_parent=wl.expect_parent(s0), every_n_days=3, limits=limits(chunk_bytes=16))
    d = read_transfer_counters().delta(t0)
    assert d.device_to_host == -(-N // 16) and d.device_to_host_bytes == N
    s1 = close_day(c, s0, 2.0)
    t1 = read_transfer_counters()
    assert out.notify_day_closed(c.lop, s1, expected_parent=wl.expect_parent(s1)).action == "not_due" and read_transfer_counters().delta(t1).device_to_host == 0
