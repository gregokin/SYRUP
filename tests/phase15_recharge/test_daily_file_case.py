"""Phase 1.5 S2 correction01: the daily lifecycle on the GENUINE native file-case factory (real NCP source verification, `admit_package_case`,
`prepare_native_dense_case`, `prepare_layered_case` from a YAML file). Opt-in by EXPLICIT configuration: set SYRUP_NCP_TARGET to the materialized
MAPLE9f bundle and run with that bundle's `editable` and `source/src` (plus the candidate `src`) on PYTHONPATH and MAPLE_SYRUP_EXPECTED_MAPLE_ROOT
set; otherwise every test skips strictly (no fake classes/guards are ever substituted). The matching SYRUP digest proves development consistency only."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import ncpcases as nc
from dcases import DAY, RUN, act, dw, ev, run_storm, start, wl
from maple_syrup import sediment_bed as sb

pytestmark = [pytest.mark.skipif(nc.skip_reason() is not None, reason=nc.skip_reason() or "")]


@pytest.fixture(autouse=True)
def _configured_bundle_or_fail():
    """When SYRUP_NCP_TARGET is configured a missing/wrong-origin bundle FAILS setup here (the genuine source guard is never skipped away)."""
    if nc.skip_reason() is None:
        nc.require_configured_bundle()


DEFAULT_CONTROL = ev.LayeredEventControl()  # the existing defaults (max_steps 100_000, max_recession_s 86_400): no loosening, no cap raised
# Explicitly labeled TEST INPUT of the completion/fallback/direct/later-storm comparison: an OWNED COPY of the native case with FLAT terrain
# (topography.perturbation.relief_m = 0) and a CLOSED surface domain (layered domain.outlets = []). The ordinary irregular file case keeps the
# 1/2/3-layer recharge, cold-source/geometry and genuine-CFL controls. Scientific defaults, LayeredQuietPolicy() and the model are unchanged.
FLAT_CLOSED = dict(flat_closed=True)


def tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode() + p.read_bytes())
    return h.hexdigest()


def fp(s):
    a = wl._arrays_of(s)
    return (s.digest, s.t_s, s.revision, s.lifetime.as_dict(), {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in {
        "h": a.surface_m, "soil": a.soil.water_m, "qx": a.qx, "qy": a.qy, "j": a.history.wetting_m, "intake": a.history.cumulative_intake_m}.items()})


@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_file_case_daily_recharge_for_1_2_3_layer_profiles_with_originals_preserved(n_layers, tmp_path):
    example_before = tree_hash(nc.EXAMPLE)
    c = nc.build_file_case(tmp_path, nc.yaml_for(n_layers))
    assert c.L == n_layers and c.profile.maple_commit.startswith("9f466d2c")
    s0 = start(c)
    before = fp(s0)
    plan = dw.next_plan(s0, 0.25, 3600.0, 1.0)  # P < T: recharge; 0.25 mm
    r = dw.daily_water_action(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, pet_terrain_m_s=np.full(c.n, 1e-8))
    assert r.status == "complete" and r.state.t_s == DAY and r.receipt.recharge_m3 == pytest.approx(0.25e-3 * c.lop.column.n_active * c.lop.column.cell_area_m2, rel=1e-12)
    audit = wl.audit_lifecycle(c.lop, r.state)
    assert abs(audit["residual_m3"]) <= audit["bound_m3"] and audit["rain_m3"] == pytest.approx(r.receipt.recharge_m3, rel=1e-12)
    assert fp(s0) == before  # the original accepted parent is untouched
    assert tree_hash(nc.EXAMPLE) == example_before  # the public case/YAML files are immutable (inputs were copied into the owned directory)
    assert r.receipt.preferential_intake_m3 == 0.0 and r.receipt.matrix_intake_m3 == 0.0  # dry surface: recharge is not modeled intake


def test_file_case_capacity_fallback_equals_a_direct_full_storm_then_a_later_storm(tmp_path):
    raw = nc.yaml_for(2, bottom="closed", theta=0.5)
    cd, cn = nc.build_file_case(tmp_path / "a", raw, **FLAT_CLOSED), nc.build_file_case(tmp_path / "b", raw, **FLAT_CLOSED)
    s_daily, s_direct = start(cd), cn.s0
    plan = dw.next_plan(s_daily, 0.01, 360.0, 1.0)
    r = dw.daily_water_action(cd.lop, s_daily, expected_parent=wl.expect_parent(s_daily), plan=plan, pet_terrain_m_s=np.full(cd.n, 1e-8))
    assert r.status == "capacity_failure" and r.fallback.failing_columns == cd.n and r.fallback.reason_bit == 17
    before = fp(s_daily)
    st = dw.begin_daily_storm(cd.lop, s_daily, expected_parent=wl.expect_parent(s_daily), plan=plan, fallback_of=r.fallback, control=DEFAULT_CONTROL)
    forcing = plan.bind_forcing(0.0)
    from maple_syrup.rainfall import rainfall_field

    ny, nx = cn.shape
    field = rainfall_field(ny, nx, active_mask=np.asarray(cn.lop.topology.active).reshape(ny, nx), xp=np)
    direct = wl.begin_storm(cn.lop, s_direct, expected_parent=wl.expect_parent(s_direct), schedule=forcing.schedule, field=field, control=DEFAULT_CONTROL)
    a, b = run_storm(cd, st).state, run_storm(cn, direct).state
    assert a.payload.completed and b.payload.completed and a.payload.event_sha256 == b.payload.event_sha256
    fa, fb = fp(a), fp(b)
    assert fa[1] == fb[1] and fa[4] == fb[4] and a.payload.water.ledger == b.payload.water.ledger  # arrays, time, J/intake, full-P ledger
    assert fp(s_daily) == before
    done = dw.finish_daily_storm(cd.lop, a, expected_parent=wl.expect_parent(a), plan=plan, pet_terrain_m_s=np.full(cd.n, 1e-8))
    assert done.state.daily.ordinal == 1 and done.receipt.capacity_failing_columns == cd.n
    closed = wl.close_interevent(cd.lop, done.state, expected_parent=wl.expect_parent(done.state))
    later = dw.next_plan(closed, 0.5, 360.0, 0.1)
    st2 = dw.begin_daily_storm(cd.lop, closed, expected_parent=wl.expect_parent(closed), plan=later, control=DEFAULT_CONTROL)
    res = run_storm(cd, st2)
    assert res.complete


def test_file_case_genuine_quiet_water_cfl_refusal_is_exact_and_credits_no_trial_e(tmp_path):
    """Root's un-injected case: genuine 4x6 file case, closed saturated 2-layer column, 1e-9 m of quiet water, P=0: one 86400 s router trial CFL-refuses."""
    c = nc.build_file_case(tmp_path, nc.yaml_for(2, bottom="closed", theta=0.5))
    lop = c.lop
    bed = sb.with_water(c.s0.payload.bed, lop.context, np.full(lop.shape, 1e-9))
    state = wl.start_lifecycle(lop, soil=c.s0.payload.soil, history=c.s0.payload.history, t_s=0.0, bed=bed)
    parent = dw.start_daily_lineage(lop, state, expected_parent=wl.expect_parent(state), run_digest="cd" * 32)
    before, cursor = fp(parent), parent.daily
    for pet_rate in (0.0, 1e-16):
        plan = dw.next_plan(parent, 0.0, 0.0, 1.0)
        with pytest.raises(dw.DailyRefusal) as e:
            dw.daily_water_action(lop, parent, expected_parent=wl.expect_parent(parent), plan=plan, pet_terrain_m_s=np.full(lop.n_cells, pet_rate))
        assert e.value.code == "router_step_rejected" and "exceeds cfl_max" in str(e.value)
        assert fp(parent) == before and parent.daily is cursor and parent.lifetime.soil_evaporation_m3 == 0.0


def test_file_case_whole_call_admission_precedes_the_rain_field(tmp_path, monkeypatch):
    c = nc.build_file_case(tmp_path, nc.yaml_for(2))
    s0 = start(c)
    plan = dw.next_plan(s0, 1.0, 360.0, 0.5)
    calls = []
    real = dw.rainfall_field
    monkeypatch.setattr(dw, "rainfall_field", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    c.quota.value = 0
    with pytest.raises(dw.DailyRefusal) as e:
        dw.begin_daily_storm(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan)
    assert e.value.code == "memory" and calls == []


def test_file_case_storm_after_a_p0_day_starts_at_86400_with_analytic_native_edges_and_rains_full_p(tmp_path):
    """The confirmed defect trigger on the genuine flat closed file case: after a P0 day + close the physical clock is 86400; the full storm (P=1 mm, I=360 mm/h, T=0.5 mm
    -> D = 10 s, written by hand) must have native absolute edges [86400, 86410], origin 86400, rainfall end 86410, and debit FULL P x active area by then."""
    from maple_syrup.conservation import volume_roundoff_bound_m3

    c = nc.build_file_case(tmp_path, nc.yaml_for(2, bottom="closed", theta=0.4), **FLAT_CLOSED)
    s0 = start(c)
    day = dw.daily_water_action(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=dw.next_plan(s0, 0.0, 0.0, 0.5), pet_terrain_m_s=np.full(c.n, 1e-8))
    closed = wl.close_interevent(c.lop, day.state, expected_parent=wl.expect_parent(day.state))
    assert closed.t_s == 86400.0
    plan = dw.next_plan(closed, 1.0, 360.0, 0.5)
    st = dw.begin_daily_storm(c.lop, closed, expected_parent=wl.expect_parent(closed), plan=plan, control=DEFAULT_CONTROL)
    event = st.binding.event
    assert event.event_origin_s == 86400.0 and [float(x) for x in event.absolute_edges_s] == [86400.0, 86410.0] and event.rainfall_end_s == 86410.0
    res = wl.evolve_storm(c.lop, st, expected_parent=wl.expect_parent(st), until_s=86410.0)
    expected = 1.0e-3 * c.lop.column.n_active * c.lop.column.cell_area_m2
    led = res.state.payload.water.ledger
    assert res.state.payload.water.t_s == 86410.0 and abs(led.rain_m3 - expected) <= volume_roundoff_bound_m3(res.accepted_steps + 2, expected) + led.global_residual_bound_m3
    assert res.state.lifetime.soil_evaporation_m3 == st.lifetime.soil_evaporation_m3  # no E during the storm
    done = run_storm(c, st)
    assert done.complete  # genuine completion under the default quiet/completion rules


# --- CPU metadata controls of the SAME owned geometry/mask test inputs used by the written CUDA controls (genuine CPU NCP factory; no GPU) ---------------------------------
def test_public_example_column_is_refused_for_the_100mm_profiles_and_the_owned_nz8_column_builds_1_2_3_layers(tmp_path):
    from maple_syrup import layered_case as lc

    def raw(n):
        r = nc.yaml_for(n, bottom="closed", theta=[0.2] * n)
        r["soil"]["layers"]["thickness_m"] = [0.1] * n
        r["soil"]["theta_residual"], r["soil"]["theta_field_capacity"], r["soil"]["theta_saturation"] = [0.0] * n, [0.2] * n, [0.4] * n
        r["soil"]["theta_evaporation_stress"] = 0.1
        return r

    for n in (2, 3):  # the genuine native texture factory refuses a profile deeper than the represented column of nz = 4 (not weakened)
        with pytest.raises(lc.LayeredCaseError, match="do not represent enough actual material"):
            nc.build_file_case(tmp_path / f"pub{n}", raw(n))
    for n in (1, 2, 3):
        c = nc.build_file_case(tmp_path / f"deep{n}", raw(n), nz=nc.NZ_DEEP)
        assert c.L == n and abs(float(np.sum(c.lop.column.layers.thickness_m)) - 0.1 * n) < 1e-12


@pytest.mark.parametrize("initial,rain,bottom,final,drain", __import__("test_daily_known_answers").TABLE)
def test_file_case_known_answer_table_incl_drainage_on_the_owned_deeper_native_column(initial, rain, bottom, final, drain, tmp_path):
    n_l = len(initial)
    raw = nc.yaml_for(n_l, bottom=bottom, theta=[x / 100.0 for x in initial])
    raw["soil"]["layers"]["thickness_m"] = [0.1] * n_l
    raw["soil"]["theta_residual"], raw["soil"]["theta_field_capacity"], raw["soil"]["theta_saturation"] = [0.0] * n_l, [0.2] * n_l, [0.4] * n_l
    raw["soil"]["theta_evaporation_stress"] = 0.1
    c = nc.build_file_case(tmp_path, raw, nz=nc.NZ_DEEP)
    s0 = start(c)
    plan = dw.next_plan(s0, float(rain), 10.0, float(rain))
    r = dw.daily_water_action(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan, pet_terrain_m_s=np.zeros(c.n))
    assert r.status == "complete"
    soil = r.state.payload.soil.water_m
    np.testing.assert_allclose(soil / 1e-3, np.array(final)[:, None] * np.ones((1, c.n)), rtol=0, atol=1e-9)
    area = c.lop.column.n_active * c.lop.column.cell_area_m2
    assert r.receipt.drainage_m3 == pytest.approx(drain * 1e-3 * area, abs=1e-12)  # drainage against the known table


def test_file_case_configured_active_mask_leaves_inactive_cells_unchanged(tmp_path):
    raw = nc.with_inactive_cells(tmp_path, nc.yaml_for(2, bottom="closed", theta=[0.1, 0.1]), [(1, 2), (2, 3)])
    c = nc.build_file_case(tmp_path, raw)
    assert c.lop.column.n_active == c.n - 2
    s0 = start(c)
    before = np.array(s0.payload.soil.water_m)
    r = dw.daily_water_action(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=dw.next_plan(s0, 2.0, 10.0, 5.0), pet_terrain_m_s=np.full(c.n, 1e-8))
    active = np.asarray(c.lop.column.active)
    assert r.status == "complete" and np.array(r.state.payload.soil.water_m)[:, ~active].tobytes() == before[:, ~active].tobytes()
    assert r.receipt.recharge_m3 == pytest.approx(2.0e-3 * c.lop.column.n_active * c.lop.column.cell_area_m2, rel=1e-12)
