"""Phase 1.5 S4 (CPU): the V3 daily physical checkpoint on the GENUINE native file case (real NCP verification/admission, Dense preparation, YAML layered factory; flat closed owned
fixture, default quiet/completion rules). FRESH-PROCESS continuation: a child process runs the whole scenario saving a V3 checkpoint after every stage; for every stage a NEW
child rebuilds the fixed components from the genuine factories, loads that checkpoint (with the expected plan/StormSpec) and continues; the physical digests (arrays, native bed,
ledgers, cursor, counters) and key scalars of every later stage must equal the uninterrupted run — real physical execution including a later full storm at a nonzero origin, not
writer=reader equality. Plus the refusal/tamper/budget/old-vs-new codec controls. Needs SYRUP_NCP_TARGET like the file-case lane (skip only if unset; configured-unusable FAILS)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import ncpcases as nc
import numpy as np
import pytest
import s4_worker as w
from dcases import DAY, RUN, dw, ev, rc, wl

from maple_syrup import layered_checkpoint as lc
from maple_syrup import lifecycle_checkpoint as lk
from maple_syrup import lifecycle_checkpoint_v2 as v2
from maple_syrup import lifecycle_checkpoint_v3 as v3
from maple_syrup import pond_checkpoint as pc

pytestmark = [pytest.mark.skipif(nc.skip_reason() is not None, reason=nc.skip_reason() or "")]
HERE = Path(__file__).resolve().parent
LIMITS = lc.CheckpointLimits(1 << 30, 1 << 30, 0, 1 << 30)
ENOUGH = lambda: 1 << 40  # noqa: E731


@pytest.fixture(autouse=True)
def _configured_bundle_or_fail():
    if nc.skip_reason() is None:
        nc.require_configured_bundle()


def child(*args) -> None:
    proc = subprocess.run([sys.executable, "-B", str(HERE / "s4_worker.py"), *map(str, args)], capture_output=True, text=True, env=dict(os.environ), cwd=str(HERE), timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]


@pytest.fixture(scope="module")
def full_runs(tmp_path_factory):
    out = {}
    for scenario in ("storm", "recharge", "fallback", "storm_numba"):
        d = tmp_path_factory.mktemp(f"s4_{scenario}")
        child(scenario, "full", d, d / "full.json")
        out[scenario] = (d, json.loads((d / "full.json").read_text()))
    return out


CASES = [(sc, st) for sc, stages in w.STAGES.items() for st in stages] + [("storm_numba", st) for st in ("PS", "D", "E")]  # + the COMPILED Numba backend at three pause points


@pytest.mark.parametrize("scenario,stage", CASES, ids=[f"{a}-{b}" for a, b in CASES])
def test_fresh_process_restart_continues_exactly_like_the_uninterrupted_run(scenario, stage, full_runs, tmp_path):
    d, full = full_runs[scenario]
    work = tmp_path / "resume"
    work.mkdir()
    for ck in d.glob("ck_*"):  # ONLY the checkpoints cross the process boundary (the fresh process rebuilds the case/compiled outputs itself)
        shutil.copytree(ck, work / ck.name)
    child(scenario, "resume", work, work / "out.json", stage)
    got = json.loads((work / "out.json").read_text())
    assert got[stage]["v3"] == full[stage]["v3"] and got[stage]["state"] == full[stage]["state"]  # the restored state IS the saved one (physical + scalar digests)
    for key, value in got.items():
        assert full[key] == value, (scenario, stage, key)  # every LATER stage (day, T, close, later full storm at a nonzero origin), digest for digest
    assert got["_final_soil_sha"] == full["_final_soil_sha"]


def test_the_scenarios_really_cover_partial_storm_nonzero_origin_fallback_and_one_update(full_runs):
    _, nb = full_runs["storm_numba"]
    assert nb["PS"]["role"] == "storm" and nb["L"]["ordinal"] == 2  # the compiled backend runs the same trajectory
    _, storm = full_runs["storm"]
    assert storm["PS"]["role"] == "storm" and storm["PS"]["pending"] and storm["CS"]["pending"] and storm["PS"]["t_s"] > 86400.0  # an ACTUAL partial storm at a nonzero clock
    assert storm["D"]["role"] == "interevent" and storm["D"]["ordinal"] == 2 and not storm["D"]["pending"]  # P0 day + storm = two consumed occurrences, ONE update each
    assert storm["T"]["seq"] == storm["D"]["seq"] + 1 and storm["E"]["role"] == "boundary" and storm["later_begin"]["pending"] and storm["L"]["ordinal"] == 2 and storm["L"]["pending"]
    assert storm["later_begin"]["t_s"] == storm["E"]["t_s"] > 2 * 86400.0 - 1  # the later storm begins at the nonzero boundary clock
    _, rech = full_runs["recharge"]
    assert rech["R"]["role"] == "interevent" and rech["R"]["rain_m3"] > 0.0 and rech["L"]["ordinal"] == 1 and rech["L"]["pending"]
    _, fb = full_runs["fallback"]
    assert fb["PS"]["role"] == "storm" and fb["D"]["ordinal"] == 2


# --- refusal / tamper / budget / old-vs-new ----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    d = tmp_path_factory.mktemp("s4_parts")
    run = w.Run(d, "storm")
    return run


def load(run, path, role, **kw):
    spec = kw.pop("spec", None)
    return v3.load_lifecycle_checkpoint_v3(path, run.lop, w.IDENT, expected_role=role, limits=LIMITS, host_available_bytes_fn=ENOUGH, reference_bed=run.c.s0.payload.bed,
                                           storm=spec, **kw)


def tamper(src: Path, dst: Path, mutate) -> Path:
    shutil.copytree(src, dst)
    mp = dst / lk.MANIFEST_NAME
    data = json.loads(mp.read_text())
    mutate(data)
    mp.write_text(json.dumps(pc._seal_manifest(data), indent=1, sort_keys=True) + "\n")  # RE-SEALED: only the semantic validators can refuse it
    return dst


def snapshot(path: Path):
    return {p.name: p.read_bytes() for p in sorted(path.iterdir())}


def test_old_writers_refuse_a_daily_state_and_v3_refuses_a_cursor_free_state_before_any_io(full_runs, tmp_path):
    run = w.Run(tmp_path / "r", "storm")
    run.step_start()
    daily = run.state
    free = run.c.s0
    assert free.daily is None and daily.daily is not None
    for save, kw in ((lk.save_lifecycle_checkpoint, {}), (lambda *a, **k: v2.save_lifecycle_checkpoint_v2(*a, profile=v2.NATIVE9F_PROFILE_V2, **k), {})):
        target = tmp_path / "old"
        with pytest.raises(lk.LifecycleCheckpointError, match="daily"):
            save(target, run.lop, daily, w.IDENT, limits=LIMITS, host_available_bytes_fn=ENOUGH, **kw)
        assert not target.exists()
    target = tmp_path / "v3free"
    with pytest.raises(lk.LifecycleCheckpointError, match="no cursor"):
        v3.save_lifecycle_checkpoint_v3(target, run.lop, free, w.IDENT, limits=LIMITS, host_available_bytes_fn=ENOUGH)
    assert not target.exists()


def test_v2_and_v3_directories_never_cross_load_and_there_is_no_migration(full_runs, tmp_path):
    d, _ = full_runs["storm"]
    run = w.Run(tmp_path / "r", "storm")
    with pytest.raises(lk.LifecycleCheckpointError):  # the V2 loader refuses a V3 directory (schema/profile literals)
        v2.load_lifecycle_checkpoint_v2(d / "ck_B0", run.lop, w.IDENT, profile=v2.NATIVE9F_PROFILE_V2, expected_role="boundary", limits=LIMITS, host_available_bytes_fn=ENOUGH,
                                        reference_bed=run.c.s0.payload.bed)
    v2dir = tmp_path / "v2dir"
    v2.save_lifecycle_checkpoint_v2(v2dir, run.lop, run.c.s0, w.IDENT, profile=v2.NATIVE9F_PROFILE_V2, limits=LIMITS, host_available_bytes_fn=ENOUGH)  # an ordinary (cursor-free) V2 archive
    with pytest.raises(lk.LifecycleCheckpointError):  # the V3 loader refuses it too (and nothing converts it)
        load(run, v2dir, "boundary")
    assert load(run, d / "ck_B0", "boundary").state.daily.ordinal == 0


def test_restored_daily_state_rejects_duplicates_old_apis_and_a_stale_parent(full_runs):
    d, _ = full_runs["storm"]
    run = w.Run(d / "again", "storm")
    r = load(run, d / "ck_B1", "boundary")
    s = r.state
    assert s.daily.ordinal == 1 and s.t_s == DAY
    dup = rc.plan_daily_rain(0.0, 0.0, 0.5, run_digest=RUN, ordinal=0)  # the already consumed P0 occurrence
    with pytest.raises(dw.DailyRefusal) as e:
        dw.daily_water_action(r.lop, s, expected_parent=wl.expect_parent(s), plan=dup, pet_terrain_m_s=np.zeros(run.c.n))
    assert e.value.code == "occurrence"
    with pytest.raises(wl.LifecycleError, match="daily lineage"):
        wl.enter_interevent_from_boundary(r.lop, s, expected_parent=wl.expect_parent(s))
    twin = wl._mint(s)
    with pytest.raises(dw.DailyRefusal):
        dw.daily_water_action(r.lop, twin, expected_parent=wl.expect_parent(s), plan=dw.next_plan(s, 0.0, 0.0, 0.5), pet_terrain_m_s=np.zeros(run.c.n))
    with pytest.raises(lk.LifecycleCheckpointError):  # the V1/V2 writers still refuse the restored daily state
        lk.save_lifecycle_checkpoint(d / "nope", r.lop, s, w.IDENT, limits=LIMITS, host_available_bytes_fn=ENOUGH)
    audit = wl.audit_lifecycle(r.lop, s)
    assert abs(audit["residual_m3"]) <= audit["bound_m3"]  # conservative closure after restore


def _set(path, value):
    """A manifest mutator that sets `daily[path...] = value` and PROVES it changed the field (the old value must differ)."""
    def mutate(m):
        node = m["payload"]["scalars"]["daily"]
        for key in path[:-1]:
            node = node[key]
        assert node[path[-1]] != value, f"mutation of {path} would not change the saved field"
        node[path[-1]] = value
    return mutate


def test_tampered_daily_records_are_refused_before_publication_even_when_resealed(full_runs, tmp_path):
    d, _ = full_runs["storm"]
    run = w.Run(tmp_path / "r", "storm")
    spec = run.spec_for(rc.plan_daily_rain(*w.P_STORM, run_digest=RUN, ordinal=1), 86400.0)
    saved = json.loads((d / "ck_PS" / lk.MANIFEST_NAME).read_text())["payload"]["scalars"]["daily"]
    assert saved["pending"]["branch"] == "storm" and saved["ordinal"] == 1 and saved["pending"]["event_sha256"] is not None  # what the mutations below actually change
    bad = {
        "clock_contract": _set(("clock_contract",), "maple_syrup.daily_clock/999"),
        "run_digest": _set(("run_digest",), "0" * 64),
        "ordinal_plus_one": _set(("ordinal",), 2),
        "pending_event_sha": _set(("pending", "event_sha256"), "1" * 64),
        "pending_branch": _set(("pending", "branch"), "storm_fallback"),
        "pending_forcing_digest": _set(("pending", "forcing_digest"), "2" * 64),
        "pending_ordinal": _set(("pending", "ordinal"), 2),
    }
    for name, mutate in bad.items():
        dst = tamper(d / "ck_PS", tmp_path / f"t_{name}", mutate)
        before = snapshot(dst)
        with pytest.raises(lk.LifecycleCheckpointError):
            load(run, dst, "storm", spec=spec)
        assert snapshot(dst) == before  # files unchanged by the refusal
    nopending = tamper(d / "ck_PS", tmp_path / "t_pending_removed", _set(("pending",), None))
    with pytest.raises(lk.LifecycleCheckpointError, match="pending"):
        load(run, nopending, "storm", spec=spec)  # a storm role without its pending occurrence
    pending_on_boundary = tamper(d / "ck_B1", tmp_path / "t_pending_on_boundary", _set(("pending",), saved["pending"]))
    with pytest.raises(lk.LifecycleCheckpointError):
        load(run, pending_on_boundary, "boundary")  # a non-storm role carrying a pending occurrence


def test_wrong_storm_spec_or_expected_plan_is_refused_and_the_right_ones_are_accepted(full_runs, tmp_path):
    d, _ = full_runs["storm"]
    run = w.Run(tmp_path / "r", "storm")
    plan = rc.plan_daily_rain(*w.P_STORM, run_digest=RUN, ordinal=1)
    good = run.spec_for(plan, 86400.0)
    assert load(run, d / "ck_PS", "storm", spec=good, expected_plan=plan).state.daily.pending is not None  # partial storm restored, still pending, not completed
    assert load(run, d / "ck_PS", "storm", spec=good).state.payload.completed is False
    other_origin = run.spec_for(plan, 86400.5)
    with pytest.raises(lk.LifecycleCheckpointError):
        load(run, d / "ck_PS", "storm", spec=other_origin)  # the physical origin is part of the saved forcing identity
    wrong_plan = rc.plan_daily_rain(2.0, 360.0, 0.5, run_digest=RUN, ordinal=1)
    with pytest.raises(lk.LifecycleCheckpointError):
        load(run, d / "ck_PS", "storm", spec=run.spec_for(wrong_plan, 86400.0))  # another storm (2x P): forcing digest differs
    with pytest.raises(lk.LifecycleCheckpointError):
        load(run, d / "ck_PS", "storm", spec=good, expected_plan=rc.plan_daily_rain(*w.P_STORM, run_digest=RUN, ordinal=2))  # wrong ordinal
    with pytest.raises(lk.LifecycleCheckpointError):
        load(run, d / "ck_B1", "boundary", expected_plan=plan)  # a non-pending role: the plan must be the last consumed occurrence
    with pytest.raises(lk.LifecycleCheckpointError):
        load(run, d / "ck_PS", "interevent", expected_source_kind="storm_end")  # wrong role
    with pytest.raises(lk.LifecycleCheckpointError):
        load(run, d / "ck_D", "interevent", expected_source_kind="boundary_origin")  # wrong source kind


def test_budgets_are_admitted_before_io_and_never_raised(full_runs, tmp_path):
    d, _ = full_runs["storm"]
    run = w.Run(tmp_path / "r", "storm")
    run.step_start()
    target = tmp_path / "lowsave"
    with pytest.raises(lk.LifecycleCheckpointQuotaError):
        v3.save_lifecycle_checkpoint_v3(target, run.lop, run.state, w.IDENT, limits=LIMITS, host_available_bytes_fn=lambda: 0)
    assert not target.exists()  # refused BEFORE any directory/I/O
    budget = v3.estimate_lifecycle_budget_v3(run.lop, "boundary", bed=run.state.payload.bed)
    assert budget.host_save_bytes > 0 and budget.serialized_bytes > 0
    with pytest.raises(lk.LifecycleCheckpointQuotaError):  # a low host reading refuses the LOAD before any source audit or file read
        v3.load_lifecycle_checkpoint_v3(d / "ck_B0", run.lop, w.IDENT, expected_role="boundary", limits=LIMITS, host_available_bytes_fn=lambda: 0, reference_bed=run.c.s0.payload.bed)
    tiny = lc.CheckpointLimits(1024, 1 << 30, 0, 1 << 30)
    with pytest.raises(lk.LifecycleCheckpointQuotaError):
        v3.save_lifecycle_checkpoint_v3(tmp_path / "tiny", run.lop, run.state, w.IDENT, limits=tiny, host_available_bytes_fn=ENOUGH)
    assert not (tmp_path / "tiny").exists()


def test_cursor_record_codec_is_strict_and_the_clock_contract_is_versioned():
    c = rc.DailyCursor(RUN)
    rec = rc.cursor_record(c)
    assert rc.cursor_from_record(rec).digest() == c.digest()
    for key, bad in (("ordinal", True), ("ordinal", 1.0), ("clock_contract", "x"), ("run_digest", "z" * 64), ("last_branch", "bogus")):
        with pytest.raises(rc.RechargeError):
            rc.cursor_from_record({**rec, key: bad})
    with pytest.raises(rc.RechargeError):
        rc.cursor_from_record({**rec, "extra": 1})
    with pytest.raises(rc.RechargeError):
        rc.cursor_from_record({k: v for k, v in rec.items() if k != "pending"})
    assert rec["clock_contract"] == rc.CLOCK_CONTRACT and lk.SCHEMA_V3.endswith("/3") and lk.NATIVE9F_PROFILE_V3.endswith("_v3")
