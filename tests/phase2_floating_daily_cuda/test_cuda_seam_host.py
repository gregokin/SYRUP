"""G3 S3 host tests of the CUDA seam of the floating daily lineage (CPU/host: NO GPU, NO CuPy). They pin what can be proven without a device: the pure two-pool plan arithmetic built from the existing
`operation_terms`, the strict quota, the order device-gate-before-reader, that the CPU lineage never touches the CUDA plumbing and rejects host-pool arguments, and that the module needs no CuPy at import. The CUDA behaviour itself
is defined in `test_gpu_definitions.py` for Root's hardware."""
# ruff: noqa: C408
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from dbuild import AVAIL, PART, RUN, close, day, lineage, make, start

from maple_syrup import floating_daily as fd
from maple_syrup import floating_daily_cuda as fdc
from maple_syrup import floating_evaporation as fe
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import floating_transpiration_cuda as ftc


def test_the_cuda_plumbing_imports_and_plans_without_cupy():
    for L, N in ((1, 1), (3, 10), (65, 50003)):
        fdc.start_plan(L, N), fdc.day_plan(L, N, recharge=True), fdc.withdrawal_plan(L, N), fdc.audit_plan(L, N)
    assert "cupy" not in sys.modules


@pytest.mark.parametrize("L,N", [(1, 7), (3, 4096), (65, 50003)])
def test_two_pool_plans_are_exactly_the_existing_component_terms(L, N):
    step, rech = fcu.operation_terms("step", L, N), fcu.operation_terms("step_with_recharge", L, N)
    e = fe.estimate_floating_evaporation_memory(N, backend="cuda")
    d0, h0 = fdc.day_plan(L, N, recharge=False)
    d1, h1 = fdc.day_plan(L, N, recharge=True)
    assert d0 == e.device_bytes + step["device"] and h0 == e.packet_bytes + step["host"]  # live E staging + ONE FSL step; packets on the host
    assert d1 == e.device_bytes + rech["device"] and h1 == e.packet_bytes + rech["host"]
    assert e.packet_bytes == (11 + 1) * 8  # the E packet (+ the device active count): 96 B, not an N-dependent host term
    wd, wh = fdc.withdrawal_plan(L, N)
    t = ftc.operation_terms("withdraw", L, N)
    assert (wd, wh) == (t["outputs"] + t["scratch"], t["host"])
    chk, sto = fcu.operation_terms("state_check", L, N), fcu.operation_terms("storage", L, N)
    assert fdc.audit_plan(L, N) == (chk["scratch"] + sto["device"], chk["host"] + sto["host"])
    sd, sh = fdc.start_plan(L, N)
    assert sd == 8 * L * N + 8 * N + chk["scratch"] + sto["device"] and sh == chk["host"] + sto["host"]
    # no host term of the resident plans depends on N (nothing of the water, maps or inputs is downloaded)
    assert fdc.day_plan(L, 7, recharge=False)[1] == fdc.day_plan(L, 50003, recharge=False)[1] and fdc.audit_plan(L, 7)[1] == fdc.audit_plan(L, 50003)[1]


@pytest.mark.parametrize("bad", [-1, True, 1.5, "0", None, np.float64(2.0)])
def test_quota_is_strict(bad):
    with pytest.raises(fsl.FloatingSoilError):
        fdc.quota(None, bad, None, 0)
    with pytest.raises(fsl.FloatingSoilError):
        fdc.quota(None, 0, None, bad)
    for not_callable in (5, "x", object()):
        with pytest.raises(fsl.FloatingSoilError):
            fdc.quota(not_callable, 0, None, 0)
    q = fdc.quota(AVAIL, 3, AVAIL, 4)
    assert q["reserve_bytes"] == 3 and q["host_reserve_bytes"] == 4
    c = fdc.component_quota(q)
    assert c["reserve_bytes"] == 0 and c["host_reserve_bytes"] == 0 and c["available_bytes_fn"] is AVAIL  # same readers, each reserve counted ONCE (composed admission)


def test_the_device_gate_precedes_every_reader_and_array_operation(monkeypatch):
    """CONTROL of the ORDER only (a stand-in column with the one attribute `_pools` reads; it is never given to a numerical body): the gate is the first operation, so a wrong current device refuses before any reader."""
    reads, gated = [], []

    def refuse(column):
        gated.append(1)
        raise fsl.FloatingSoilError("this object lives on device 0 but the current device is 1")

    monkeypatch.setattr(fdc, "gate", refuse)
    col = SimpleNamespace(backend="cuda", device_id=0)
    with pytest.raises(fd.DailyRefusal, match="device"):
        fd._pools(col, lambda: reads.append(1) or 10**12, 0, lambda: reads.append(1) or 10**12, 0)
    assert gated == [1] and reads == []


def test_a_cpu_lineage_rejects_host_pool_arguments_and_never_touches_the_cuda_plumbing(monkeypatch):
    touched = []
    for name in ("gate", "admit", "day_plan", "audit_plan", "withdrawal_plan", "start_plan", "check_state_resident", "zeros"):
        real = getattr(fdc, name)
        monkeypatch.setattr(fdc, name, lambda *a, _n=name, _r=real, **k: touched.append(_n) or _r(*a, **k))
    owner, bed = make()
    with pytest.raises(fd.DailyRefusal, match="CUDA lineage"):
        fd.start_floating_daily_lineage(owner, bed, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL, host_reserve_bytes=1)
    with pytest.raises(fd.DailyRefusal, match="CUDA lineage"):
        fd.start_floating_daily_lineage(owner, bed, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    assert owner.is_current
    st = start(owner, bed)
    n = st.ctx.n_cells
    z = np.zeros(n)
    kw = dict(plan=__import__("dbuild").plan(0.0, 0.0, 5.0, 0), pet_terrain_m_s=z, cover=z)
    for extra in ({"host_reserve_bytes": 1}, {"host_available_bytes_fn": AVAIL}):
        with pytest.raises(fd.DailyRefusal, match="CUDA lineage"):
            fd.floating_daily_action(st, expected_parent=fd.expect_parent(st), available_bytes_fn=AVAIL, **kw, **extra)
    inter = day(st).state
    with pytest.raises(fd.DailyRefusal, match="CUDA lineage"):
        fd.request_floating_withdrawal(inter, expected_parent=fd.expect_parent(inter), request_m=np.zeros((3, n)), expected_sequence=0, available_bytes_fn=AVAIL, host_reserve_bytes=1)
    with pytest.raises(fd.DailyRefusal, match="CUDA lineage"):
        fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), available_bytes_fn=AVAIL, host_reserve_bytes=1)
    assert close(inter).state.cursor.ordinal == 1 and touched == []  # the CPU daily path never calls the CUDA plumbing


def test_the_context_keeps_the_host_active_count_and_the_cpu_rain_declaration_uses_it(monkeypatch):
    st, bed, *_ = lineage()
    assert st.ctx.n_active == int(np.asarray(bed.active_mask).sum()) == int(st.ctx.active.sum())
    r = day(st, p=2.0, i=4.0, t=5.0)
    assert r.state.ledger.last_day_rain_m3 == pytest.approx(2.0 / 1000.0 * st.ctx.area * st.ctx.n_active)


def test_the_moderate_native_fixture_of_the_device_definitions_is_valid_on_the_cpu_lineage():
    """HOST validation of the fixture the N = 50003 device definitions use (`stratified_dense(shallow_cells=(), ny=1, nx=50003)`): the actual native bed/owner/lineage build and complete a day, and a due output frame on the
    CPU has the exact shapes; so a device failure cannot be a fixture defect (the pond-helper tilted default would exceed its voxels on this grid)."""
    import builders as b

    bed = b.bed_from(b.fx.stratified_dense(shallow_cells=(), ny=1, nx=50003))
    owner, _ = make(bed=bed)
    st = start(owner, bed)
    assert st.ctx.n_cells == 50003 and st.ctx.n_layers == 3 and st.ctx.n_active == 50003
    res = day(st)
    assert res.outcome == "interevent"
    assert close(res.state).state.cursor.ordinal == 1 and res.state.t_s == 86400.0
