"""Phase 1.5 S4: the CUDA V3 daily physical checkpoint on the GENUINE native9f CUDA file case (WRITTEN, NOT RUN by the author: no GPU launch/probe; Root guards GPU0 and runs it after writer
closure). Opt-in `SYRUP_RELEASE_GPU=1` (the `gpu` fixture) plus `SYRUP_NCP_CUDA_TARGET` (materialized CUDA 9f bundle; configured-but-unusable FAILS setup); run in a FRESH process with that bundle's
`editable`/`source/src` and the candidate `src` on PYTHONPATH. FRESH-PROCESS continuation: a child (`s4_cuda_worker.py`) runs a whole scenario saving a V3 checkpoint at every stage; per stage a NEW
child rebuilds the genuine CUDA factories, loads that checkpoint (expected plan + StormSpec) and continues; the CUDA physical digests (device arrays, native bed, ledgers, cursor, counters) of every
later stage — day, T, close, a later FULL storm at a nonzero origin — must equal the uninterrupted run. Plus wrong-policy/daily-type/source/tamper/plan/clock/counter refusals, low host and low
device readings BEFORE I/O, and conservative closure. No metadata-only surrogate; no tolerance is introduced (exact digests)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import gcases as g
import numpy as np
import pytest
import s4_cuda_worker as cw
import s4_worker as w
from dcases import RUN, rc

from maple_syrup import layered_checkpoint as lc
from maple_syrup import lifecycle_checkpoint as lk
from maple_syrup import lifecycle_checkpoint_cuda as cs
from maple_syrup import lifecycle_checkpoint_cuda_v2 as v2c
from maple_syrup import lifecycle_checkpoint_cuda_v3 as v3c
from maple_syrup import pond_checkpoint as pc
from maple_syrup import water_lifecycle as wl

pytestmark = pytest.mark.usefixtures("gpu")
HERE = Path(__file__).resolve().parent
MANIFEST = "lifecycle_checkpoint_cuda_v3.json"


@pytest.fixture(autouse=True)
def _configured_cuda_bundle(gpu):  # noqa: ARG001  (the `gpu` fixture skips unless SYRUP_RELEASE_GPU=1 and a device exists)
    g.require_cuda_bundle()


def child(*args) -> None:
    proc = subprocess.run([sys.executable, "-B", str(HERE / "s4_cuda_worker.py"), *map(str, args)], capture_output=True, text=True, env=dict(os.environ), cwd=str(HERE), timeout=1800)
    assert proc.returncode == 0, proc.stderr[-3000:]


@pytest.fixture(scope="module")
def full_runs(tmp_path_factory):
    out = {}
    for scenario in ("storm", "recharge", "fallback"):
        d = tmp_path_factory.mktemp(f"s4cuda_{scenario}")
        child(scenario, "full", d, d / "full.json")
        out[scenario] = (d, json.loads((d / "full.json").read_text()))
    return out


CASES = [(sc, st) for sc, stages in w.STAGES.items() for st in stages]


@pytest.mark.parametrize("scenario,stage", CASES, ids=[f"{a}-{b}" for a, b in CASES])
def test_fresh_process_cuda_restart_continues_exactly_like_the_uninterrupted_run(scenario, stage, full_runs, tmp_path):
    d, full = full_runs[scenario]
    work = tmp_path / "resume"
    work.mkdir()
    for ck in d.glob("ck_*"):  # ONLY the checkpoints cross the process boundary
        shutil.copytree(ck, work / ck.name)
    child(scenario, "resume", work, work / "out.json", stage)
    got = json.loads((work / "out.json").read_text())
    assert got[stage]["v3"] == full[stage]["v3"] and got[stage]["state"] == full[stage]["state"]  # the restored device state IS the saved one
    for key, value in got.items():
        assert full[key] == value, (scenario, stage, key)  # every LATER stage, digest for digest (incl. the later full storm at a nonzero origin)


def test_scenarios_cover_partial_storm_nonzero_origin_fallback_and_one_update_on_cuda(full_runs):
    _, storm = full_runs["storm"]
    assert storm["PS"]["role"] == "storm" and storm["PS"]["pending"] and storm["PS"]["t_s"] > 86400.0 and storm["D"]["ordinal"] == 2 and storm["L"]["pending"]
    assert storm["T"]["seq"] == storm["D"]["seq"] + 1 and storm["later_begin"]["t_s"] == storm["E"]["t_s"]
    _, fb = full_runs["fallback"]
    assert fb["PS"]["role"] == "storm" and fb["D"]["ordinal"] == 2


# --- fixed components rebuilt in-process for the refusal controls -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return cw.CudaRun(tmp_path_factory.mktemp("s4cuda_parts"), "storm")


def load(run, path, role, **kw):
    kw.setdefault("limits", cw.LIMITS)
    return v3c.load_lifecycle_checkpoint_cuda_v3(path, run.lop, w.IDENT, expected_role=role, reference_bed=run.c.s0.payload.bed, **cw.RES, **kw)


def tamper(src: Path, dst: Path, mutate) -> Path:
    shutil.copytree(src, dst)
    mp = dst / MANIFEST
    data = json.loads(mp.read_text())
    mutate(data)
    mp.write_text(json.dumps(pc._seal_manifest(data), indent=1, sort_keys=True) + "\n")
    return dst


def snapshot(path: Path):
    return {p.name: p.read_bytes() for p in sorted(path.iterdir())}


def _set(path, value):
    def mutate(m):
        node = m["payload"]["scalars"]["daily"]
        for key in path[:-1]:
            node = node[key]
        assert node[path[-1]] != value
        node[path[-1]] = value
    return mutate


C1_V2 = HERE / "fixtures" / "c1_cuda_v2"  # the immutable GENUINE C1-produced V2 archive (byte copy; provenance in fixtures/c1_cuda_v2_PROVENANCE.json)


def test_wrong_policy_and_daily_type_are_refused_before_io(full_runs, tmp_path):
    d, _ = full_runs["storm"]
    run = cw.CudaRun(tmp_path / "r", "storm")
    run.step_start()
    kw = dict(limits=cw.LIMITS, **cw.RES)
    for target, save in (("v1", lambda p: cs.save_lifecycle_checkpoint_cuda(p, run.lop, run.state, w.IDENT, **kw)),
                         ("v2", lambda p: v2c.save_lifecycle_checkpoint_cuda_v2(p, run.lop, run.state, w.IDENT, **kw))):
        with pytest.raises(cs.LifecycleCheckpointCudaError, match="daily"):
            save(tmp_path / target)
        assert not (tmp_path / target).exists()  # V1/V2 cannot save a DAILY source: refused before any I/O
    with pytest.raises(cs.LifecycleCheckpointCudaError, match="no cursor"):
        v3c.save_lifecycle_checkpoint_cuda_v3(tmp_path / "free", run.lop, run.c.s0, w.IDENT, **kw)  # V3 cannot save a cursor-free state
    assert not (tmp_path / "free").exists()
    with pytest.raises(cs.LifecycleCheckpointCudaError):  # the V2 loader refuses a V3 directory
        v2c.load_lifecycle_checkpoint_cuda_v2(d / "ck_B0", run.lop, w.IDENT, expected_role="boundary", limits=cw.LIMITS, reference_bed=run.c.s0.payload.bed, **cw.RES)
    before = snapshot(C1_V2)
    with pytest.raises(cs.LifecycleCheckpointCudaError):  # the V3 loader refuses the GENUINE historical V2 archive (names/schema/profile/source), reading nothing of its payload
        load(run, C1_V2, "boundary")
    assert snapshot(C1_V2) == before  # the immutable fixture is untouched


def test_v2_saving_an_ordinary_cursor_free_state_under_the_new_sources_refuses_its_c1_helper_pin_before_publication(tmp_path):
    """V2 is NOT repinned: an ordinary (cursor-free) V2 save under the N15 candidate correctly refuses the changed CPU helper before any archive is published (the C1 fixture above is the only V2 archive)."""
    run = cw.CudaRun(tmp_path / "r", "storm")
    target = tmp_path / "ordinary_v2"
    with pytest.raises(cs.LifecycleCheckpointCudaError, match="differs from its pinned digest"):
        v2c.save_lifecycle_checkpoint_cuda_v2(target, run.lop, run.c.s0, w.IDENT, limits=cw.LIMITS, **cw.RES)
    assert not target.exists()


def test_tampered_cursor_plan_clock_and_counters_are_refused_when_resealed(full_runs, tmp_path):
    d, _ = full_runs["storm"]
    run = cw.CudaRun(tmp_path / "r", "storm")
    plan = rc.plan_daily_rain(*w.P_STORM, run_digest=RUN, ordinal=1)
    spec = run.spec_for(plan, 86400.0)
    bad = {"clock_contract": _set(("clock_contract",), "maple_syrup.daily_clock/999"), "run_digest": _set(("run_digest",), "0" * 64), "ordinal": _set(("ordinal",), 2),
           "event": _set(("pending", "event_sha256"), "1" * 64), "branch": _set(("pending", "branch"), "storm_fallback"), "forcing": _set(("pending", "forcing_digest"), "2" * 64),
           "pending_ordinal": _set(("pending", "ordinal"), 2)}
    for name, mutate in bad.items():
        dst = tamper(d / "ck_PS", tmp_path / f"t_{name}", mutate)
        before = snapshot(dst)
        with pytest.raises(cs.LifecycleCheckpointCudaError):
            load(run, dst, "storm", storm=spec)
        assert snapshot(dst) == before
    wrong_source = tamper(d / "ck_B0", tmp_path / "t_source", lambda m: m["source"].__setitem__("maple_syrup_tree_sha256", "0" * 64))
    with pytest.raises(cs.LifecycleCheckpointCudaError, match="source pair mismatch"):
        load(run, wrong_source, "boundary")  # a wrong source pair is refused before any payload


def test_storm_spec_expected_plan_and_forged_plan_types_on_cuda(full_runs, tmp_path):
    d, _ = full_runs["storm"]
    run = cw.CudaRun(tmp_path / "r", "storm")
    plan = rc.plan_daily_rain(*w.P_STORM, run_digest=RUN, ordinal=1)
    good = run.spec_for(plan, 86400.0)
    r = load(run, d / "ck_PS", "storm", storm=good, expected_plan=plan)
    assert r.state.daily.pending is not None and r.state.payload.completed is False  # an ACTUAL partial storm restored on the device, never completed by the loader
    with pytest.raises(cs.LifecycleCheckpointCudaError):
        load(run, d / "ck_PS", "storm", storm=run.spec_for(plan, 86400.5))  # wrong physical origin
    with pytest.raises(cs.LifecycleCheckpointCudaError):
        load(run, d / "ck_PS", "storm", storm=good, expected_plan=rc.plan_daily_rain(*w.P_STORM, run_digest=RUN, ordinal=2))
    import dataclasses
    from types import SimpleNamespace

    for bad in ("plan", SimpleNamespace(digest=plan.digest), dataclasses.replace(plan, depth_m=0.5)):
        with pytest.raises(cs.LifecycleCheckpointCudaError, match="expected_plan"):
            load(run, d / "ck_PS", "storm", storm=good, expected_plan=bad)


class Box:
    def __init__(self, value=1 << 42):
        self.value = value

    def __call__(self):
        return self.value


@pytest.mark.parametrize("which", ["host", "device"])
def test_low_host_and_low_device_readings_refuse_save_and_load_before_io(which, full_runs, tmp_path, monkeypatch):
    d, _ = full_runs["storm"]
    run = cw.CudaRun(tmp_path / "r", "storm")
    run.step_start()
    low = {"host_available_bytes_fn": Box(0)} if which == "host" else {"device_available_bytes_fn": Box(0)}
    target = tmp_path / "lowsave"
    with pytest.raises(cs.LifecycleCheckpointCudaQuotaError):
        v3c.save_lifecycle_checkpoint_cuda_v3(target, run.lop, run.state, w.IDENT, limits=cw.LIMITS, **cw.RES, **low)
    assert not target.exists()  # refused BEFORE any directory/I/O
    spy = []
    real = cs._fresh_source_pair
    monkeypatch.setattr(cs, "_fresh_source_pair", lambda *a, **k: (spy.append(1), real(*a, **k))[1])
    with pytest.raises(cs.LifecycleCheckpointCudaQuotaError):
        load(run, d / "ck_B0", "boundary", **low)
    assert spy == []  # the cold source audit never ran


def test_restored_state_closes_conservatively_and_rejects_duplicates_and_old_apis(full_runs, tmp_path):
    from maple_syrup import daily_water as dw

    d, _ = full_runs["storm"]
    run = cw.CudaRun(tmp_path / "r", "storm")
    r = load(run, d / "ck_B1", "boundary")
    s = r.state
    assert s.daily.ordinal == 1 and s.t_s == 86400.0
    audit = wl.audit_lifecycle(r.lop, s)
    assert abs(audit["residual_m3"]) <= audit["bound_m3"]
    dup = rc.plan_daily_rain(0.0, 0.0, 0.5, run_digest=RUN, ordinal=0)
    with pytest.raises(dw.DailyRefusal) as e:
        dw.daily_water_action(r.lop, s, expected_parent=wl.expect_parent(s), plan=dup, pet_terrain_m_s=run.pet())
    assert e.value.code == "occurrence"
    with pytest.raises(wl.LifecycleError, match="daily lineage"):
        wl.enter_interevent_from_boundary(r.lop, s, expected_parent=wl.expect_parent(s))
