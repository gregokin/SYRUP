"""Actual-GPU test DEFINITIONS of the CUDA event<->daily bridge (WRITTEN, NOT RUN by their author: no GPU/CuPy/NVRTC/nvidia-smi was touched). Opt-in (SYRUP_RELEASE_GPU=1); Root runs them on GPU0 with `.venv-cupy`.

FIXTURES (two, reused; every GPU fixture has a genuinely MATCHED CPU twin and `assert_matched` proves the matched deep layered landscape / initial inventory / held FC,SAT / D0 BEFORE any output comparison):
  * PIT   the CPU_C3 isolated shallow closed pit (a genuinely COMPLETED native event that retains 1.5 mm of standing water below its rim: a real WET boundary with real no-flow faces, not a fake H/q), on the qualified deep bed
          (`e5.device_event(**pit_kwargs)` vs `fb.make_pair(backend="numpy", **pit_kwargs)[0]`, the same kwargs). The wet day uses an EXPLICIT `day_control` (max_dt 3600 s, retries/steps declared) so 86400 s honestly completes in
          O(100) fragments; the deep connected-lake case with max_dt 1 s (> 345 000 fragments) is deliberately NOT used.
  * DRY   a native light rain fully absorbed by an unsaturated open-bottom plane (zero standing water), same matched construction.
The reopened-forcing / completion helpers are the relative bounded loops of the lifecycle suite (no absolute time cap, no fake quiet). Declared limits (NOT defined, no claim): a real connected-flow outlet export and a CFL
rejection fixture on device (labelled COMPONENT controls need a steep draining fixture), nonzero carried q (a rest pit carries q = 0 honestly), Windowed variants, a physical GPU bridge file, large catchments, and the
traced allocator peak/pool-reuse medium control (needs the real allocator; Root measures). Nothing is fabricated: no capability, generation, owner, held class or quiet state."""
# ruff: noqa: RUF059
from __future__ import annotations

import importlib.util
from pathlib import Path

import fpond_builders as fb
import numpy as np
import pytest
from lifecycle_builders import AVAIL as CPU_AVAIL
from lifecycle_builders import run_to_complete
from refresh_fixtures import assert_matched

_T = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("lifecycle_gpu_defs_for_bridge", _T / "phase2_floating_event_lifecycle_cuda" / "test_gpu_definitions.py")
ld = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ld)

from fpond_builders import ph

from maple_syrup import floating_daily as fd
from maple_syrup import floating_daily_output as fo
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import native_floating_soil as nfs
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup import soil_moisture_output as g0
from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.experimental_storm import ExperimentalControl
from maple_syrup.pond_event import PondCompletionPolicy, PondEvent

BIG = 1 << 40
MEM = fsl.FloatingMemoryError
cp = ld.cp
host = ld.host
e5 = ld.rd.e5
RUN = "ab" * 32
PART = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
CTL = ExperimentalControl(max_dt_s=3600.0, min_dt_s=1 / 1024, max_retries=30, max_steps=2_000_000)  # explicit declared DAY control (not a global default)
POLICY = PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-4, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True)
TOL = {"rtol": 1e-9, "atol": 1e-12}  # declared CPU/GPU float64 tolerance (not bitwise)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# matched fixtures
# ---------------------------------------------------------------------------------------------------------------------------------------------
def pit_kwargs(level_m=0.0015, **over):
    z = ph.basin() - 0.08
    kw = {"surface": z, "depth": np.maximum(float(z.min()) + level_m - z, 0.0), "closed": True, "bottom": "open", "saturation": 0.6, "splash": False, "rain_mm_h": 0.0, "end_s": 2.0, "max_dt_s": 1.0, "policy": POLICY}
    kw.update(over)
    return kw


def dry_kwargs(**over):
    kw = {"surface": ph.plane(base=0.12, drop=0.004), "depth": 0.0, "bottom": "open", "saturation": 0.5, "splash": False, "rain_mm_h": 10.0, "end_s": 0.05, "max_dt_s": 0.2, "policy": POLICY}  # a 0.05 s native rain: pure contract fixture (genuine quiet, unmodified policy)
    kw.update(over)
    return kw


def factory_of(ev):
    def factory(v, topology=None, schedule=None):
        return PondEvent(ev.context, None, ev.sediment, ev.schedule if schedule is None else schedule, ev._initial_topology if topology is None else topology, control=ev.control, policy=ev.policy, splash=ev.splash,
                         floating_vertical=v)

    return factory


class Side:
    """One backend's genuinely completed lifecycle boundary of a fixture (`dev` True: the CUDA device fixture, False: its matched CPU twin)."""

    def __init__(self, tmp, kw, dev, windowed=False, pref=False):
        self.dev, self.windowed, self.pref = dev, windowed, pref
        if dev:
            self.ev, self.st, self.vert, self.owner = e5.device_event(tmp, windowed=windowed, **kw)
            self.bed, self.rdr, self.hk, self.xp = self.st.bed, fcu.device_reader(e5._cp()), {"host_available_bytes_fn": lambda: BIG}, e5._cp()
        else:
            self.ev, self.st, self.vert, self.owner, self.bed, _c = fb.make_pair(tmp, backend="numpy", **kw)[1 if windowed else 0]
            self.rdr, self.hk, self.xp = CPU_AVAIL, {}, np
        self.factory = factory_of(self.ev)

    def prepared_pref(self):
        """The background preferential operator of the matched pair (same spec on both sides; device a3 on the device column)."""
        from maple_syrup import floating_preferential_infiltration as fpi

        n = self.owner.column.n_cells
        return fpi.prepare_floating_preferential(self.owner.column, a3=self.xp.full(n, 2.0e-3), max_fraction_per_hour=0.5, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([0.0, 1.0, 1.0]),
                                                 available_bytes_fn=self.rdr, **self.hk)

    def boundary(self):
        self.lin, occ, s0 = lc.open_floating_event(self.owner, self.bed, self.factory, available_bytes_fn=self.rdr, preferential=self.prepared_pref() if self.pref else None, **self.hk)
        h = ld.complete(occ, s0) if self.dev else run_to_complete(occ.event, s0)
        self.b = lc.close_occurrence(self.lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=self.rdr, **self.hk)
        return self.b

    def arrays(self, n):
        return self.xp.full(n, 3e-8), self.xp.full(n, 0.3)

    def day(self, boundary, ordinal, plan=None, *, ctl=CTL, request=None, wet=True):
        st = br.open_day(self.lin, boundary, partition=PART, run_digest=RUN, available_bytes_fn=self.rdr, **self.hk)
        n = st.ctx.n_cells
        pet, cov = self.arrays(n)
        pl = rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=ordinal) if plan is None else plan
        kw = {"event_factory": self.factory, "day_control": ctl} if wet else {}
        res = br.day_action(st, plan=pl, pet_terrain_m_s=pet, cover=cov, available_bytes_fn=self.rdr, **self.hk, **kw) if wet else br.day_action(st, plan=pl, pet_terrain_m_s=pet, cover=cov,
                                                                                                                                              available_bytes_fn=self.rdr, **self.hk)
        return st, res

    def finish(self, res, request=None):
        n = res.state.ctx.n_cells
        req = self.xp.zeros((res.state.ctx.n_layers, n)) if request is None else request
        t = fd.request_floating_withdrawal(res.state, expected_parent=fd.expect_parent(res.state), request_m=req, expected_sequence=0, available_bytes_fn=self.rdr, **self.hk)
        closed = fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), available_bytes_fn=self.rdr, **self.hk).state
        return t, closed

    def hand(self, closed):
        return br.hand_back(self.lin, closed, available_bytes_fn=self.rdr, host_available_bytes_fn=(self.hk or {}).get("host_available_bytes_fn")) if self.dev else br.hand_back(self.lin, closed)


def matched_pair(tmp, kw, windowed=False, pref=False):
    g, c = Side(tmp / "g", kw, True, windowed, pref), Side(tmp / "c", kw, False, windowed, pref)
    assert_matched(g.ev, g.st, g.vert, c.ev, c.st, c.vert, windowed)  # same deep strata/inventory/held FC,SAT/D0 BEFORE any comparison
    g.boundary()
    c.boundary()
    return g, c


def w(state):
    return host(state.water.water_m)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# wet pit: matched twin, exact day, partial T once, hand-back, consecutive day, tripwires
# ---------------------------------------------------------------------------------------------------------------------------------------------
def test_wet_pit_day_matches_the_deep_cpu_twin_t_once_partial_roots_close_hand_back_and_consecutive_day(cp, tmp_path):
    g, c = matched_pair(tmp_path, pit_kwargs())
    assert float(host(g.b.surface_depth_m).max()) > 0.0 and float(np.asarray(c.b.surface_depth_m).max()) > 0.0  # a real standing film on both
    sg, rg = g.day(g.b, 0)
    sc, rc_ = c.day(c.b, 0)
    assert rg.outcome == rc_.outcome == "interevent" and float(rg.state.t_s) == float(rc_.state.t_s) == float(g.b.t_s) + fd.DAY_S  # exactly one native day, no restamp
    np.testing.assert_allclose(w(rg.state), np.asarray(rc_.state.water.water_m), **TOL)
    np.testing.assert_allclose(host(rg.state.surface[0]), np.asarray(rc_.state.surface[0]), **TOL)
    # T ONCE, partial (depth-limited / proportional) through the existing aggregate request API: a request larger than the available water in some layers
    n, L = rg.state.ctx.n_cells, rg.state.ctx.n_layers
    req_g, req_c = g.xp.full((L, n), 2e-2), np.full((L, n), 2e-2)
    tg, closed_g = g.finish(rg, req_g)
    tc, closed_c = c.finish(rc_, req_c)
    assert 0.0 < tg.actual_m3 < tg.requested_m3 and tg.unmet_m3 > 0.0
    np.testing.assert_allclose(host(tg.actual_m), np.asarray(tc.actual_m), **TOL)
    with pytest.raises(fd.DailyRefusal):  # the same request sequence cannot debit twice
        fd.request_floating_withdrawal(tg.state, expected_parent=fd.expect_parent(tg.state), request_m=req_g, expected_sequence=0, available_bytes_fn=g.rdr, **g.hk)
    bg = g.hand(closed_g)
    bc_ = c.hand(closed_c)
    assert g.lin.head is bg and abs(bg.ledger["residual_m3"]) <= bg.ledger["residual_bound_m3"] and bg.root.host_active is not None and bg.root.depth_m is g.b.root.depth_m
    # a CONSECUTIVE wet/dry day from the handed-back boundary (carried H/q; a rest pit carries q = 0 honestly)
    s2g, r2g = g.day(bg, 1)
    s2c, r2c = c.day(bc_, 1)
    assert float(r2g.state.t_s) == float(g.b.t_s) + 2 * fd.DAY_S
    np.testing.assert_allclose(w(r2g.state), np.asarray(r2c.state.water.water_m), **TOL)


def test_the_wet_gpu_day_allocates_no_nvk_inventory_no_pond_phase_and_downloads_no_water_shaped_array(cp, tmp_path, monkeypatch):
    """Root-style TRIPWIRES on the ACTUAL native functions, armed only AFTER `open_day` (the HOT wet day is claimed, the cold static metadata of the opening is attributed separately): `pond_event.bed_inventory` is a MODULE
    function (not a `PondEvent` member), `_assemble_initial` / `continued_state` are class members; none may run. Every MAPLE-counted device-to-host read is RECORDED with shape, dtype, nbytes and the stage it belongs to
    and classified independently: a read of an actual `(L, N)`-shaped or flat `L*N` water array (or of any live W/J/I buffer by identity) is FORBIDDEN; bounded diagnostic PACKETS (state-check 4 words, step/storage/E packets,
    scalar sums) are listed as packets and never mistaken for water by a size threshold."""
    from maple.core import backend as mb

    import maple_syrup.pond_event as pe

    g = Side(tmp_path / "g", pit_kwargs(), True)
    g.boundary()
    st = br.open_day(g.lin, g.b, partition=PART, run_digest=RUN, available_bytes_fn=g.rdr, **g.hk)  # the cold opening is NOT under the hot wet-day tripwires
    L, N = st.ctx.n_layers, st.ctx.n_cells
    trips = []
    monkeypatch.setattr(pe, "bed_inventory", lambda *a, **k: trips.append("bed_inventory") or (_ for _ in ()).throw(AssertionError("bed_inventory (full NVK reduction) ran in the wet day")))
    for name in ("_assemble_initial", "continued_state"):
        real = getattr(PondEvent, name)

        def spy(self, *a, _n=name, _r=real, **k):
            trips.append(_n)
            return _r(self, *a, **k)

        monkeypatch.setattr(PondEvent, name, spy)
    live = {id(st.water.water_m)}
    reads = []
    real_to_host = mb.to_host

    def recording(a, *x, **k):
        reads.append({"shape": tuple(getattr(a, "shape", ())), "dtype": str(getattr(a, "dtype", "")), "nbytes": int(getattr(a, "nbytes", 0)), "id": id(a)})
        return real_to_host(a, *x, **k)

    monkeypatch.setattr(mb, "to_host", recording)
    pet, cov = g.arrays(N)
    res = br.day_action(st, plan=rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=0), pet_terrain_m_s=pet, cover=cov, available_bytes_fn=g.rdr, event_factory=g.factory, day_control=CTL, **g.hk)
    assert res.outcome == "interevent" and trips == []
    water_shaped = [r for r in reads if r["shape"] in ((L, N), (L * N,)) or r["id"] in live]
    assert water_shaped == [], water_shaped  # an actual water-shaped / live-W download would be listed here and fails (no size-threshold heuristic)
    packets = [r for r in reads if r["nbytes"] <= 64 * 8]
    others = [r for r in reads if r["nbytes"] > 64 * 8]
    print("ROOT_WET_DAY_READS", {"L": L, "N": N, "n_packets": len(packets), "others_shapes": sorted({(r["shape"], r["dtype"]) for r in others})})  # reported truthfully: packets vs everything else


# ---------------------------------------------------------------------------------------------------------------------------------------------
# quota: the caller's two pools must reach the WET day (Root's confirmed defect) and every stage
# ---------------------------------------------------------------------------------------------------------------------------------------------
def test_wet_day_honours_a_low_host_reader_the_device_reader_reserves_and_invalid_arguments_and_leaves_the_head(cp, tmp_path):
    g = Side(tmp_path / "g", pit_kwargs(), True)
    g.boundary()
    st = br.open_day(g.lin, g.b, partition=PART, run_digest=RUN, available_bytes_fn=g.rdr, **g.hk)
    n = st.ctx.n_cells
    pet, cov = g.arrays(n)
    plan = rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=0)
    calls = []

    def counting(tag, value):
        return lambda: calls.append(tag) or value

    base = {"plan": plan, "pet_terrain_m_s": pet, "cover": cov, "event_factory": g.factory, "day_control": CTL}
    # a host reader that reports (almost) nothing: refused as a typed memory error, AFTER the host reader was actually consulted, and the parent day state unchanged
    with pytest.raises((br.BridgeError, MEM)):
        br.day_action(st, available_bytes_fn=g.rdr, host_available_bytes_fn=counting("host", 1), **base)
    assert "host" in calls and st.ledger.days_applied == 0
    calls.clear()
    # the DEVICE reader and reserve count too (reserve once per pool)
    with pytest.raises((br.BridgeError, MEM)):
        br.day_action(st, available_bytes_fn=counting("dev", 1), host_available_bytes_fn=lambda: BIG, **base)
    assert "dev" in calls
    with pytest.raises((br.BridgeError, MEM)):
        br.day_action(st, available_bytes_fn=g.rdr, reserve_bytes=BIG, host_available_bytes_fn=lambda: BIG, **base)
    with pytest.raises((br.BridgeError, MEM)):
        br.day_action(st, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG, host_reserve_bytes=BIG, **base)
    # invalid argument TYPES are typed refusals (never coerced / defaulted)
    for bad in ({"host_available_bytes_fn": 5}, {"host_reserve_bytes": -1}, {"host_reserve_bytes": True}):
        with pytest.raises((br.BridgeError, fsl.FloatingSoilError, TypeError, ValueError)):
            br.day_action(st, available_bytes_fn=g.rdr, **{"host_available_bytes_fn": lambda: BIG, **bad}, **base)
    assert isinstance(g.lin.head, lc.DailyHold) and st.ledger.days_applied == 0
    ok = br.day_action(st, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG, **base)  # the unchanged parent still runs (retry)
    assert ok.outcome == "interevent"
    # the CPU twin REFUSES the CUDA-only arguments (no silent stripping)
    c = Side(tmp_path / "c", pit_kwargs(), False)
    c.boundary()
    cst = br.open_day(c.lin, c.b, partition=PART, run_digest=RUN, available_bytes_fn=c.rdr)
    with pytest.raises(br.BridgeError, match="only to a CUDA"):
        br.day_action(cst, available_bytes_fn=c.rdr, host_reserve_bytes=3, plan=plan, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), event_factory=c.factory, day_control=CTL)


def test_current_device_gate_precedes_every_reader_in_open_day_and_the_wet_action(cp, tmp_path, monkeypatch):
    g = Side(tmp_path / "g", pit_kwargs(), True)
    g.boundary()
    reads = []
    reader = lambda: reads.append(1) or BIG
    real = fcu.current_device
    monkeypatch.setattr(fcu, "current_device", lambda c: real(c) + 1)
    with pytest.raises((br.BridgeError, fsl.FloatingSoilError), match="device"):
        br.open_day(g.lin, g.b, partition=PART, run_digest=RUN, available_bytes_fn=reader, host_available_bytes_fn=reader)
    assert reads == [] and g.lin.head is g.b
    monkeypatch.setattr(fcu, "current_device", real)
    st = br.open_day(g.lin, g.b, partition=PART, run_digest=RUN, available_bytes_fn=g.rdr, **g.hk)
    n = st.ctx.n_cells
    pet, cov = g.arrays(n)
    monkeypatch.setattr(fcu, "current_device", lambda c: real(c) + 1)
    with pytest.raises((br.BridgeError, fsl.FloatingSoilError), match="device"):
        br.day_action(st, plan=rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=0), pet_terrain_m_s=pet, cover=cov, available_bytes_fn=reader, host_available_bytes_fn=reader, event_factory=g.factory, day_control=CTL)
    assert reads == []


def test_open_day_need_minus_one_exact_reserves_once_and_a_refusal_leaves_the_boundary_head(cp, tmp_path):
    from maple_syrup import floating_event_daily_bridge_cuda as bc

    g = Side(tmp_path / "g", pit_kwargs(), True)
    b = g.boundary()
    d_need, h_need = bc.open_plan_cuda(b.root.column.n_layers, b.root.column.n_cells)
    big = lambda: BIG
    for kw in ({"available_bytes_fn": lambda: d_need + 7 - 1, "host_available_bytes_fn": big}, {"available_bytes_fn": big, "host_available_bytes_fn": lambda: h_need + fcu.PY_ALLOWANCE_BYTES + 11 - 1},
               {"available_bytes_fn": big, "host_available_bytes_fn": big, "memory_budget_bytes": 0}):
        with pytest.raises((br.BridgeError, MEM)):
            br.open_day(g.lin, b, partition=PART, run_digest=RUN, reserve_bytes=7, host_reserve_bytes=11, **kw)
        assert g.lin.head is b
    day = br.open_day(g.lin, b, partition=PART, run_digest=RUN, reserve_bytes=7, host_reserve_bytes=11, available_bytes_fn=lambda: d_need + 7, host_available_bytes_fn=lambda: h_need + fcu.PY_ALLOWANCE_BYTES + 11)
    assert isinstance(g.lin.head, lc.DailyHold) and day.role == "boundary"


# ---------------------------------------------------------------------------------------------------------------------------------------------
# dry fixture: P0, P = T recharge, closed-capacity fallback, full-storm pending (forcing-history proof), N = 3
# ---------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("tag", "p", "i", "t", "branch"), [("P0", 0.0, 0.0, 5.0, "no_rain"), ("P=T", 5.0, 5.0, 5.0, "recharge")])
def test_dry_days_p0_and_p_equal_t_recharge_match_the_cpu_twin(cp, tmp_path, tag, p, i, t, branch):
    g, c = matched_pair(tmp_path, dry_kwargs())
    pl = rc.plan_daily_rain(p, i, t, run_digest=RUN, ordinal=0)
    _sg, rg = g.day(g.b, 0, pl, wet=False)
    _sc, rc_ = c.day(c.b, 0, pl, wet=False)
    assert rg.outcome == rc_.outcome == "interevent" and rg.branch == rc_.branch == branch
    np.testing.assert_allclose(w(rg.state), np.asarray(rc_.state.water.water_m), **TOL)
    assert rg.state.ledger.rain_m3 == pytest.approx(rc_.state.ledger.rain_m3, rel=1e-9, abs=1e-18)
    assert isinstance(g.lin.head, lc.DailyHold)  # the lineage head is the hold until the day is closed and handed back
    _t, closed = g.finish(rg)
    assert abs(g.hand(closed).ledger["residual_m3"]) <= g.lin.head.ledger["residual_bound_m3"]


def test_closed_profile_capacity_failure_falls_back_to_the_full_native_storm_from_the_unchanged_parent(cp, tmp_path):
    g, c = matched_pair(tmp_path, dry_kwargs(bottom="closed", saturation=0.97, rain_mm_h=0.0, end_s=2.0))
    pl = rc.plan_daily_rain(100.0, 50.0, 100.0, run_digest=RUN, ordinal=0)  # P = T = 100 mm: a pulse no near-saturated closed column can hold
    st, rg = g.day(g.b, 0, pl, wet=False)
    _sc, rc_ = c.day(c.b, 0, pl, wet=False)
    assert rg.outcome == rc_.outcome == "requires_full_storm" and rg.branch == rc_.branch == "storm_fallback" and rg.capacity_failure
    assert rg.forcing.p_mm == 100.0 and rg.forcing.digest == rc_.forcing.digest  # the FULL P and the native forcing
    assert st.ledger.days_applied == 0 and float(st.t_s) == float(g.b.t_s)  # the unchanged parent: nothing applied, no clock/cursor movement


def test_pending_full_storm_is_matched_by_the_native_schedule_history_and_the_day_resumes_once(cp, tmp_path):
    g, c = matched_pair(tmp_path, dry_kwargs(max_dt_s=1.0))
    pl = rc.plan_daily_rain(0.01, 3600.0, 0.005, run_digest=RUN, ordinal=0)  # P = 0.01 mm > T = 0.005 mm at 3600 mm/h: a 0.01 s full native storm (pure contract fixture, same forcing/history/head gates)
    st = br.open_day(g.lin, g.b, partition=PART, run_digest=RUN, available_bytes_fn=g.rdr, **g.hk)
    pet, cov = g.arrays(st.ctx.n_cells)
    pend = br.begin_storm_from_day(g.lin, st, pl, pet_terrain_m_s=pet, cover=cov, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)
    f = pend.pending_forcing
    sched = br.storm_schedule(f)
    # a WRONG history is refused BEFORE the head moves: one manual off-native-rate step, then the genuine completion; the same total/end cannot buy the proof
    occ, s = lc.begin_occurrence(g.lin, pend, lambda v, topology=None: g.factory(v, topology, sched), available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)
    s = occ.event.step(s, 2.0 * 3600.0 / 3.6e6, 0.005).state  # twice the native intensity once (a manual off-rate step inside the native piece)
    h = ld.complete(occ, s)
    with pytest.raises(lc.LifecycleError, match="forcing history"):
        lc.close_occurrence(g.lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)
    assert g.lin.head is occ  # refused before the consume and the head
    # the GENUINE canonical native evolution is matched, closes through the shared GPU proof and the day resumes ONCE with the ORIGINAL plan
    g2, c2 = matched_pair(tmp_path / "ok", dry_kwargs(max_dt_s=1.0))
    st2 = br.open_day(g2.lin, g2.b, partition=PART, run_digest=RUN, available_bytes_fn=g2.rdr, **g2.hk)
    pet2, cov2 = g2.arrays(st2.ctx.n_cells)
    pend2 = br.begin_storm_from_day(g2.lin, st2, pl, pet_terrain_m_s=pet2, cover=cov2, available_bytes_fn=g2.rdr, host_available_bytes_fn=lambda: BIG)
    sched2 = br.storm_schedule(pend2.pending_forcing)
    occ2, s2 = lc.begin_occurrence(g2.lin, pend2, lambda v, topology=None: g2.factory(v, topology, sched2), available_bytes_fn=g2.rdr, host_available_bytes_fn=lambda: BIG)
    h2 = ld.complete(occ2, s2)
    b2 = lc.close_occurrence(g2.lin, occ2, occ2.event.completion_capability(h2, h2.state), h2.state, available_bytes_fn=g2.rdr, host_available_bytes_fn=lambda: BIG)
    assert b2.storm_match == pend2.pending_forcing.digest and b2.cursor.pending is not None
    dstate = br.open_day(g2.lin, b2, partition=PART, run_digest=RUN, available_bytes_fn=g2.rdr, **g2.hk)
    res = br.day_after_storm(dstate, plan=pl, pet_terrain_m_s=pet2, cover=cov2, available_bytes_fn=g2.rdr, host_available_bytes_fn=lambda: BIG, event_factory=g2.factory, day_control=CTL)
    _t, closed = g2.finish(res)
    assert closed.cursor.ordinal == 1 and closed.cursor.pending is None  # the original plan consumed ONCE
    # (a positive-rain begin from a pending recertification boundary and the R0 recertification after a matched storm are covered by the CPU C3 controls; the GPU definitions of those two are not repeated here)


def native_event_with_refresh(g, boundary, monkeypatch):
    """A REAL native 30 s / 10 mm/h event on the device, genuinely completed, closed through the GPU selected refresh with the provider values scaled by 0.9 AFTER the completion and BEFORE the close (declared control)."""
    from maple_syrup.rainfall import constant_rainfall

    sched = constant_rainfall(boundary.t_s, boundary.t_s + 30.0, 10.0)

    def factory(v, topology=None):
        return g.factory(v, topology, sched)

    occ, s = lc.begin_occurrence(g.lin, boundary, factory, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)
    h = ld.complete(occ, s)
    real = nfs._batch_properties

    def scaled(b_, c_, k_, p_):
        f, sa, k = real(b_, c_, k_, p_)
        return f * 0.9, sa * 0.9, k

    monkeypatch.setattr(nfs, "_batch_properties", scaled)
    try:
        return lc.close_occurrence(g.lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)
    finally:
        monkeypatch.setattr(nfs, "_batch_properties", real)


def test_n3_output_saves_only_days_3_and_6_over_seven_nominal_days_with_an_intervening_native_event_and_a_real_property_refresh(cp, tmp_path, monkeypatch):
    """Seven completed NOMINAL days; between days 3 and 4 a REAL native event (30 s of 10 mm/h on the device) with the selected property refresh whose provider values are perturbed AFTER the genuine completion and BEFORE
    the close (declared control: the held values are then the originals, so the refresh really changes the properties). Only the post-T audited closures of days 3 and 6 are saved (the storm's physical duration adds no day)."""
    lim = g0.MoistureOutputLimits(max_snapshots=10, max_total_bytes=10**8, max_manifest_bytes=8192, chunk_bytes=1 << 16, host_memory_budget_bytes=10**8, host_reserve_bytes=0, disk_reserve_bytes=0)
    g = Side(tmp_path / "g", dry_kwargs(splash=True), True)  # SPLASH-enabled: the short intervening native rain performs real wet/splash MAPLE exchanges (existing physics) so the gross dirty set is genuinely non-empty
    boundary = g.boundary()
    gen0, sha0 = boundary.root.generation, boundary.root.column.sha256
    fc0 = host(boundary.root.column._maps["theta_field_capacity"]).copy()
    boundary_gen = boundary_sha = None
    (tmp_path / "out").mkdir()
    out = None
    notices = []
    for day in range(7):
        st, res = g.day(boundary, day, wet=False)
        if out is None:
            out = fo.FloatingMoistureOutput.create(tmp_path / "out", st, expected_parent=fd.expect_parent(st), every_n_days=3, limits=lim, host_available_bytes_fn=lambda: BIG, disk_available_bytes_fn=lambda: BIG)
        n = st.ctx.n_cells
        t = fd.request_floating_withdrawal(res.state, expected_parent=fd.expect_parent(res.state), request_m=g.xp.zeros((st.ctx.n_layers, n)), expected_sequence=0, available_bytes_fn=g.rdr, **g.hk)
        r = fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), output=out, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG, disk_available_bytes_fn=lambda: BIG)
        notices.append(r.notice)
        boundary = g.hand(r.state)
        if day == 2:  # the intervening real native event + property refresh
            boundary = native_event_with_refresh(g, boundary, monkeypatch)
            assert boundary.root.column.sha256 != sha0 and boundary.root.generation > gen0  # the properties REALLY changed (a no-op perturbation over an empty dirty set would fail here)
            assert not np.array_equal(host(boundary.root.column._maps["theta_field_capacity"]), fc0)
            boundary_gen, boundary_sha = boundary.root.generation, boundary.root.column.sha256
        if day == 3:
            ident = st.frame_identity()
            assert ident["property_generation"] == boundary_gen and ident["property_sha256"] == boundary_sha  # the frames after the refresh carry the CURRENT property identity
    assert [x.due for x in notices] == [False, False, True, False, False, True, False] and [x.action for x in notices if x.due] == ["saved", "saved"]
    assert boundary.root.generation > gen0


# ---------------------------------------------------------------------------------------------------------------------------------------------
# wet preferential, Windowed water-only day, connected-flow / CFL scope, composed memory on a non-trivial N
# ---------------------------------------------------------------------------------------------------------------------------------------------
def test_wet_pit_day_with_the_background_preferential_operator_matches_the_cpu_twin(cp, tmp_path):
    """The accepted preferential stage of the wet fragment on the device (a genuine preferential operator on both sides, the SAME spec: a3 2e-3, max fraction 0.5 /h, max rate 1e-5 m/s, receivers [0, 1, 1])."""
    g, c = matched_pair(tmp_path, pit_kwargs(), pref=True)
    assert g.b.root.pref_op is not None and c.b.root.pref_op is not None
    _sg, rg = g.day(g.b, 0)
    _sc, rc_ = c.day(c.b, 0)
    np.testing.assert_allclose(w(rg.state), np.asarray(rc_.state.water.water_m), **TOL)
    np.testing.assert_allclose(host(rg.state.surface[0]), np.asarray(rc_.state.surface[0]), **TOL)
    assert abs(rg.state.ledger.surface_m3 - rc_.state.ledger.surface_m3) <= 1e-9 * max(1.0, abs(rc_.state.ledger.surface_m3))
    _t, closed = g.finish(rg)
    assert abs(g.hand(closed).ledger["residual_m3"]) <= g.lin.head.ledger["residual_bound_m3"]


def test_windowed_water_only_wet_day_mask_and_metadata_path_matches_the_cpu_twin(cp, tmp_path):
    """Published native Windowed beds on both sides (`e5.device_event(windowed=True)` vs `fb.make_pair(...)[1]`): the day moves no sediment, so only the water wrapper, the active mask and the sealed host metadata are exercised."""
    g, c = matched_pair(tmp_path, pit_kwargs(), windowed=True)
    assert g.b.root.host_active is not None and g.b.root.host_depth is not None
    _sg, rg = g.day(g.b, 0)
    _sc, rc_ = c.day(c.b, 0)
    assert float(rg.state.t_s) == float(rc_.state.t_s) == float(g.b.t_s) + fd.DAY_S
    np.testing.assert_allclose(w(rg.state), np.asarray(rc_.state.water.water_m), **TOL)
    _t, closed = g.finish(rg)
    bg = g.hand(closed)
    assert bg.bed.voxel_column is g.b.bed.voxel_column and bg.root.depth_m is g.b.root.depth_m  # the Windowed bed is REFERENCED (never copied or republished), the CURRENT D carried


# (superseded by the component definitions below) CONNECTED OUTLET / CFL scope: the fragment's surface stage is the accepted `CudaSurfaceRouter` through `_FloatingSolver._step`; its connected-flow export and CFL-rejection/halving controls are
# qualified by the floating Pond GPU suite (phase2_floating_pond/test_gpu_definitions.py) and the retry contract by the CPU C3 bridge controls. The isolated pit exercises the router with real no-flow faces only; a draining
# connected device fixture (a labelled COMPONENT control of `solver.step_day` with an over-CFL dt over a slope, then the halving) remains for Root and is NOT claimed.


def flat_event(n, level, dev):
    """`(event, state, vertical, owner, bed)` of a genuine FLAT closed standing-film event at N = n x n built from the PUBLIC constructors (`moderate_bed.build`: the native Dense bed on a nearly flat terrain; the closed
    topology, the control, the floating owner and the Pond event are the ordinary public factories; nothing invented). No rain, no mobile pool; `dev` selects the CUDA or the NumPy backend of the same construction."""
    from dataclasses import replace
    from types import SimpleNamespace

    import moderate_bed as mbd
    from maple.core.backend import to_device, to_device_tree

    from maple_syrup import floating_pond as fp
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters

    dense, ctx, _topo, z = mbd.build(n, n, 60, depth=float(level))
    wh = fb._windowed_helpers()
    topo = ph.make_topology(dense, [], allow_closed=True)
    xp = e5._cp() if dev else np
    if dev:
        bed = to_device_tree(dense, xp)
        dctx = replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, xp))
    else:
        bed, dctx = dense, ctx
    cfg = fb.config(backend="cuda" if dev else "numpy", bottom="open")
    owner = nfs.initialize_native_floating_soil(fb.native_bed(bed, dctx), cfg, initial_relative_saturation=0.6, **({} if dev else {"available_bytes_fn": CPU_AVAIL}))
    vert = fp_vertical(fp, owner, dev)
    ctl = wh.control_for(SimpleNamespace(hydraulics="cuda" if dev else "numpy", xp=xp), windowed=False, max_dt_s=0.2)
    ev = PondEvent(dctx, None, plot1_sediment_parameters(xp=xp), constant_rainfall(0.0, 0.05, 0.0), topo, control=ctl, policy=POLICY, splash=None, floating_vertical=vert)
    return ev, ev.initial_state(bed, vert.initial_soil_water()), vert, owner, bed, z


def fp_vertical(fp, owner, dev):
    return fp.FloatingPondVertical(owner, available_bytes_fn=fcu.device_reader(e5._cp())) if dev else fp.FloatingPondVertical(owner, available_bytes_fn=CPU_AVAIL)


class FlatSide(Side):
    def __init__(self, n, level, dev):
        self.dev, self.windowed, self.pref = dev, False, False
        self.ev, self.st, self.vert, self.owner, self.bed, _z = flat_event(n, level, dev)
        if dev:
            self.rdr, self.hk, self.xp = fcu.device_reader(e5._cp()), {"host_available_bytes_fn": lambda: BIG}, e5._cp()
        else:
            self.rdr, self.hk, self.xp = CPU_AVAIL, {}, np
        self.factory = factory_of(self.ev)


@pytest.mark.parametrize("n", [64])
def test_open_day_true_allocator_peak_inside_the_model_pool_reuse_and_counted_traffic_on_a_nontrivial_n(cp, tmp_path, n):
    """COMPONENT control on a genuine flat closed standing-film event (N = n x n = 4096; real native quiets with an explicit policy hold of 1 s; no rain, no mobile) built through the PUBLIC factories: THREE independent genuine
    boundary parents are prepared BEFORE the allocator watch; `open_day` runs once on each (no head rewinds) under the accepted observer (pool sampled at EVERY allocator call + measured rounding) and the opened Day state is
    released after each measurement. SCOPE: the opening operation only (`open_plan_cuda`: zero blocks, storage query, surface/face copies, probes); the caller's inputs (bed, roots, W/J/I) are live and EXCLUDED; the nested
    stages of a wet DAY admit separately and are not summed here; source/JIT/driver memory excluded. A peak above the declared device payload (+ measured rounding) is a model underestimate and FAILS."""
    import gc

    from maple.core import backend as mb

    sides = []
    for i in range(3):
        sd = FlatSide(n, 0.0015, True)
        sd.boundary()
        sides.append(sd)
    pool = cp.get_default_memory_pool()
    L, N = sides[0].b.root.column.n_layers, sides[0].b.root.column.n_cells
    model_dev, model_host = bc_model(L, N)
    plateau, evidence = [], []
    for sd in sides:
        gc.collect()
        cp.cuda.Device().synchronize()
        base = int(pool.used_bytes())
        box = {"peak": 0, "rounding": 0}

        def hook(size, box=box, base=base):
            ptr = pool.malloc(size)
            box["peak"] = max(box["peak"], int(pool.used_bytes()) - base)
            box["rounding"] += int(ptr.mem.size) - int(size)
            return ptr

        mb.reset_transfer_counters()
        with cp.cuda.using_allocator(hook):
            day = br.open_day(sd.lin, sd.b, partition=PART, run_digest=RUN, available_bytes_fn=sd.rdr, **sd.hk)
        c = mb.read_transfer_counters()
        assert box["peak"] <= model_dev + box["rounding"], (box, model_dev)  # TRUE peak vs the declared device payload (+ measured rounding)
        assert c.device_to_host_bytes < 4096 and c.host_to_device_bytes < 4096, c  # counted scalar probes only (no W/J/I/surface mirror, no upload)
        evidence.append({"peak": box["peak"], "model": model_dev, "rounding": box["rounding"], "d2h": c.device_to_host_bytes, "h2d": c.host_to_device_bytes})
        del day
        gc.collect()
        cp.cuda.Device().synchronize()
        plateau.append(int(pool.used_bytes()))
    assert max(plateau) - min(plateau) <= max(e["rounding"] for e in evidence), plateau  # the released pool plateau is stable across the three independent runs
    print("ROOT_OPEN_DAY_ALLOCATOR_EVIDENCE", {"N": N, "L": L, "model_host": model_host, "runs": evidence, "plateau": plateau})


def bc_model(L, N):
    from maple_syrup import floating_event_daily_bridge_cuda as bc

    return bc.open_plan_cuda(L, N)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# correction03: cumulative closure with nonzero E and T through the next occurrence; connected-flow / CFL component controls
# ---------------------------------------------------------------------------------------------------------------------------------------------
LEDGER_KEYS = ("evaporation_m3", "transpiration_m3", "rain_m3", "drainage_m3", "outlet_export_m3", "residual_m3", "residual_bound_m3")


def test_cumulative_closure_with_nonzero_e_and_t_through_hand_back_next_occurrence_and_close_matches_the_cpu_twin(cp, tmp_path):
    """Both closes (CPU and CUDA) subtract the daily E and T from the expected storage AND carry them in the bound's operand scale (the same `volume_roundoff_bound_m3` formula and count): the pit day has nonzero surface+soil
    E and a partial T; the handed-back boundary opens a next native occurrence which is genuinely completed and closed; the cumulative water closure fields are compared with the matched CPU twin."""
    g, c = matched_pair(tmp_path, pit_kwargs())
    out = {}
    for side in (g, c):
        _s, r = side.day(side.b, 0)
        n, L = r.state.ctx.n_cells, r.state.ctx.n_layers
        _t, closed = side.finish(r, side.xp.full((L, n), 2e-2))
        b2 = side.hand(closed)
        assert b2.ledger["evaporation_m3"] > 0.0 and b2.ledger["transpiration_m3"] > 0.0  # nonzero E and T actually entered the lifetime ledger
        occ, s0 = lc.begin_occurrence(side.lin, b2, side.factory, available_bytes_fn=side.rdr, **side.hk)
        h = ld.complete(occ, s0) if side.dev else run_to_complete(occ.event, s0)
        b3 = lc.close_occurrence(side.lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=side.rdr, **side.hk)
        assert abs(b3.ledger["residual_m3"]) <= b3.ledger["residual_bound_m3"]
        out[side.dev] = {k: float(b3.ledger[k]) for k in LEDGER_KEYS}
    for k in LEDGER_KEYS:
        assert out[True][k] == pytest.approx(out[False][k], rel=1e-6, abs=1e-15), k


def conn_kwargs():
    """The opened (NOT completed) draining plane of the CPU C3 router control: a genuine completed fixture with an outlet cannot be produced (a draining film never quiets), so this is a labelled COMPONENT control of the actual
    `_FloatingSolver.step_day` on the actual opened vertical."""
    return {"surface": ph.plane(base=0.12, drop=0.004), "depth": 0.003, "bottom": "open", "saturation": 0.5, "splash": False, "rain_mm_h": 0.0, "end_s": 2.0, "max_dt_s": 0.2, "policy": POLICY}


def test_connected_flow_cfl_rejection_is_pure_and_the_retry_and_outlet_export_match_the_cpu_twin_fragment_by_fragment(cp, tmp_path):
    """Component level (no fabricated completed Boundary / day state; no `wet_day_action`): the opened native plane's ACTUAL solver (`CudaSurfaceRouter` on the device, the CPU router on the twin) over a few fragments:
    a CFL-violating fragment is rejected with `HydraulicStepRejected` BEFORE anything is registered and the clock/state unchanged; the retry from the UNCHANGED state at a valid dt proceeds; per fragment the soil water, J/I
    history, surface depth, face fluxes q (NONZERO and carried by the previous fragment's output) and the outlet export (m3) match the CPU twin within the declared tolerance; and the water conservation closes
    (storage change = -(E + drainage + export))."""
    from maple_syrup import floating_evaporation as fe
    from maple_syrup.experimental_hydrology import HydraulicStepRejected

    rows = {}
    for dev in (True, False):
        side = Side(tmp_path / ("g" if dev else "c"), conn_kwargs(), dev)
        lin, occ, st = lc.open_floating_event(side.owner, side.bed, side.factory, available_bytes_fn=side.rdr, **side.hk)
        vert, ev = occ.vertical, occ.event
        solver = ev._runtime(ev._initial_topology, None).pond.solver
        held = vert.held_template()
        op = fe.prepare_floating_evaporation(held.column, PART, available_bytes_fn=side.rdr, **side.hk)
        n = vert.n_cells
        pet, cov = side.arrays(n)
        hyd = st.hydraulic
        reg, t0 = len(vert._registry), float(hyd.t_s)
        with pytest.raises(HydraulicStepRejected):
            solver.step_day(hyd, 600.0, op, pet, cov)  # a real CFL rejection, pure: ...
        assert len(vert._registry) == reg and float(hyd.t_s) == t0  # ... nothing registered, the state unchanged
        soil0 = float(fsl.floating_storage_m3(held.column, vert.current_payload(hyd.soil_water_m).state, **({"available_bytes_fn": side.rdr, **side.hk})))
        area = float(vert.area)
        frag, es, eso, drain, export = [], 0.0, 0.0, 0.0, 0.0
        acc_bound, acc_terms = 0.0, 0
        surf0 = float(host(hyd.depth_m).sum()) * area
        for _ in range(6):
            step = solver.step_day(hyd, 0.5, op, pet, cov)  # the retry from the UNCHANGED start at a valid dt, then a few carried fragments
            info = solver.last_day
            b = info["soil_budget"]
            es += float(info["surface_evaporation_m3"]) if dev else float(info["surface_evaporation_m"].sum()) * area
            eso += float(b.evaporation_m3)
            drain += float(b.drainage_m3)
            acc_bound += float(b.bound_m3)
            acc_terms += int(b.n_terms)
            export += float(info["export_m3"])
            hyd = step.state
            p = vert.current_payload(hyd.soil_water_m)
            frag.append({"W": host(p.state.water_m).copy(), "J": host(p.history.wetting_m).copy(), "I": host(p.history.cumulative_intake_m).copy(), "H": host(hyd.depth_m).copy(), "qx": host(hyd.qx_m2_s).copy(),
                         "qy": host(hyd.qy_m2_s).copy(), "export": float(info["export_m3"])})
        soil1 = float(fsl.floating_storage_m3(held.column, vert.current_payload(hyd.soil_water_m).state, **({"available_bytes_fn": side.rdr, **side.hk})))
        surf1 = float(host(hyd.depth_m).sum()) * area
        resid = (soil1 + surf1) - (soil0 + surf0) + es + eso + drain + export
        n_active = int(np.count_nonzero(np.asarray(vert._active_host())))
        scale = max(abs(soil0 + surf0), abs(soil1 + surf1), abs(es + eso), abs(drain), abs(export), 1e-30)  # actual flux/storage magnitude scale
        bound = acc_bound + volume_roundoff_bound_m3(acc_terms + 2 * vert.n_layers * n_active + 8, scale)  # the ACTUAL MAPLE bound: accumulated native step budgets + the same-count roundoff term as `wet_day_action`
        assert abs(resid) <= bound, (resid, bound)  # strict volume closure (storage change = -(E + drainage + export)); no arbitrary tolerance
        assert export > 0.0 and max(float(np.abs(f["qx"]).max()) for f in frag) > 0.0  # a draining plane: real outlet export and NONZERO q carried between fragments
        rows[dev] = frag
    for fg, fc_ in zip(rows[True], rows[False], strict=True):
        for k in ("W", "J", "I", "H", "qx", "qy"):
            np.testing.assert_allclose(fg[k], fc_[k], **TOL, err_msg=k)
        assert fg["export"] == pytest.approx(fc_["export"], rel=1e-9, abs=1e-18)
    # NOT claimed: a genuinely completed outlet-full 86400 s connected GPU day (not fixturable: a draining film never quiets).


def test_wet_runtime_composed_admission_refuses_below_the_larger_of_the_setup_and_composed_models_on_each_pool_and_leaves_the_state(cp, tmp_path):
    """The PUBLIC `lc.day_runtime` on a genuine wet day state: the vertical's SETUP guard (`construction_bytes`, host_op) and the corrected COMPOSED two-pool wet model (`wet_runtime_plan_cuda`) are both consulted; the LARGER
    governs each pool and is never relabelled as exact (the router admits separately). Below it (reserves counted once) the call is a typed memory refusal BEFORE any carried copy or initial state, the head is the hold and the
    day state is unchanged; with generous pools the same call succeeds (retry)."""
    from maple_syrup import floating_event_lifecycle_cuda as flc
    from maple_syrup import floating_pond as fp

    g = Side(tmp_path / "g", pit_kwargs(), True)
    g.boundary()
    st = br.open_day(g.lin, g.b, partition=PART, run_digest=RUN, available_bytes_fn=g.rdr, **g.hk)
    probe = fp.FloatingPondVertical.from_boundary(g.b.root, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)
    qn = int(st.surface[1].size) + int(st.surface[2].size)
    wd, wh = flc.wet_runtime_plan_cuda(probe.step_bytes, probe.host_op_bytes, probe.n_cells, qn)
    dev_need, host_need = max(probe.construction_bytes, wd), max(probe.cold_host_bytes, wh) + fcu.PY_ALLOWANCE_BYTES
    del probe
    hold, big = st.ctx.hold, (lambda: BIG)
    for kw in ({"available_bytes_fn": lambda: dev_need + 7 - 1, "host_available_bytes_fn": big}, {"available_bytes_fn": big, "host_available_bytes_fn": lambda: host_need + 11 - 1}):
        with pytest.raises(MEM):
            lc.day_runtime(hold, st, g.factory, reserve_bytes=7, host_reserve_bytes=11, **kw)
        assert g.lin.head is hold and st.ledger.days_applied == 0
    vert, _ev, _h0, _solver = lc.day_runtime(hold, st, g.factory, available_bytes_fn=g.rdr, host_available_bytes_fn=big)  # generous pools: the same call succeeds
    assert vert.n_cells == st.ctx.n_cells
