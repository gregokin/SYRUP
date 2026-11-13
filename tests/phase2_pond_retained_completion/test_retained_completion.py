"""E4a: shared terminal settlement + `PondEvent.settle_retaining_water` (retained-water completion), CPU. Native Dense and genuine
Windowed beds through the existing pond helpers; the legacy `finalize`/`complete` default is checked against the pre-edit reference sources."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[0] / "pond_event"), str(HERE.parents[0] / "pond_windowed")]

from pond_helpers import (
    arrays_of,
    assert_unchanged,
    basin,
    host,
    initial,
    make_bed,
    make_event,
    make_topology,
)
from windowed_helpers import (
    control_for,
    dense_deep_bed,
    windowed_counterpart,
)

from maple_syrup.pond_event import (
    PondCompletionPolicy,
    PondEventError,
    PondEventIncomplete,
    PondRetainedHandoff,
)

LOAD = 2.0e-4
POLICY = PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)


def lake(*, eta=0.215, load=LOAD, soil=0.0, windowed=False, tmp_path=None, dt=0.5, policy=POLICY):
    z = basin()
    mobile = np.zeros((3, 5, 6))
    if load:
        mobile[1, 2, 2] = load
    depth = np.maximum(eta - z, 0.0)
    if windowed:
        from types import SimpleNamespace

        dense, ctx = dense_deep_bed(z, depth=depth, mobile=mobile)
        bed, wctx = windowed_counterpart(dense, ctx, tmp_path / "store")
        topo = make_topology(dense, [], allow_closed=True)
        event = make_event(bed, wctx, topo, backend=SimpleNamespace(hydraulics="numpy", xp=np), policy=policy, max_dt_s=dt,
                           control=control_for(SimpleNamespace(hydraulics="numpy", xp=np), windowed=True, max_dt_s=dt))
    else:
        bed, ctx = make_bed(z, depth=depth, mobile=mobile)
        topo = make_topology(bed, [], allow_closed=True)
        event = make_event(bed, ctx, topo, policy=policy, max_dt_s=dt)
    state = event.initial_state(bed, np.full(event.shape, soil))
    return event, state


def quiet(event, state, end=20.0):
    res = event.evolve(state, end, stop_when_quiet=True)
    assert res.quiet_ready
    return res.state


def digest(value) -> str:
    h = hashlib.sha256()
    for path, a in arrays_of(value):
        h.update(path.encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def scalars(value, path=""):
    if is_dataclass(value) and not isinstance(value, type):
        for f in fields(value):
            yield from scalars(getattr(value, f.name), f"{path}.{f.name}")
    elif isinstance(value, (int, float, str, bool)) or value is None:
        yield path, value


@pytest.fixture(params=["dense", "windowed"])
def kind(request):
    return request.param


def build(kind, tmp_path, **kw):
    return lake(windowed=kind == "windowed", tmp_path=tmp_path, **kw)


def test_retains_surface_soil_faces_history_and_settles_the_mobile_pool(kind, tmp_path):
    event, state = build(kind, tmp_path, soil=0.01)
    q = quiet(event, state)
    assert float(host(q.hydraulic.soil_water_m).sum()) > 0.0  # the initial soil is genuinely non-zero
    keep = {n: host(getattr(q.hydraulic, n)).copy() for n in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s")}
    wd = host(q.bed.water.depth_m).copy()
    h = event.settle_retaining_water(q)
    assert isinstance(h, PondRetainedHandoff) and h.complete and h.status == "complete" and h.state.status == "wet"
    for n, a in keep.items():  # nothing removed, zeroed or restored
        assert host(getattr(h.state.hydraulic, n)).tobytes() == a.tobytes(), n
    assert host(h.state.bed.water.depth_m).tobytes() == wd.tobytes() and h.state.t_s == q.t_s == h.t_s
    assert not np.any(host(h.state.bed.water.mobile_mass_by_cell_class_kg))  # settled through MAPLE, nothing erased
    acc = h.accounting
    assert acc["terminal_deposition_by_class_kg"][2] == pytest.approx(LOAD, abs=1e-9)
    assert acc["retained_surface_m3"] > 0.0 and acc["retained_soil_m3"] > 0.0
    assert acc["final_audit"]["water"]["closes"] and acc["final_audit"]["sediment"]["closes"]
    assert acc["final_audit"]["water"].get("reset_removed_m3", 0.0) == 0.0
    assert h.state.water.reset_depth is None and h.state.water.reset_soil is None  # no dry-reset record
    assert h.state.counters.n_final_commits == q.counters.n_final_commits + 1
    assert h.config_sha256 == event.config_sha256 and h.forcing_sha256 == event.forcing_sha256 and h.schema.endswith("pond_retained_handoff/1")
    if kind == "windowed":
        from maple_syrup.pond_bed import pending_journal_records

        assert pending_journal_records(h.state.bed) == 0  # a complete handoff carries no unpersisted journal


def test_retained_sediment_state_equals_the_legacy_terminal_settlement(kind, tmp_path):
    """The shared worker: the settled native bed/ledgers/counters of the retained path equal those of the legacy finalize (same physics), the legacy adds only the reset."""
    e1, s1 = build(kind, tmp_path / "a", soil=0.01) if kind == "dense" else build(kind, _sub(tmp_path, "a"), soil=0.01)
    e2, s2 = build(kind, tmp_path / "b", soil=0.01) if kind == "dense" else build(kind, _sub(tmp_path, "b"), soil=0.01)
    q1, q2 = quiet(e1, s1), quiet(e2, s2)
    kept = e1.settle_retaining_water(q1).state
    dry = e2.finalize(q2).dry_state
    assert kept.sediment.deposited_terminal is not None
    for name in ("deposited_terminal", "deposition_cell"):
        assert host(getattr(kept.sediment, name)).tobytes() == host(getattr(dry.sediment, name)).tobytes()
    assert host(kept.bed.active_layer.mass_kg).tobytes() == host(dry.bed.active_layer.mass_kg).tobytes()
    assert host(kept.phase_fraction).tobytes() == host(dry.phase_fraction).tobytes()
    ks, ds = dict(scalars(kept.counters)), dict(scalars(dry.counters))
    assert {k: v for k, v in ks.items() if "storage" not in k} == {k: v for k, v in ds.items() if "storage" not in k}
    assert dry.status == "dry" and kept.status == "wet"


def _sub(tmp_path, name):
    p = tmp_path / name
    p.mkdir()
    return p


def test_initial_gates_are_required_and_failures_leave_the_parent_unchanged(tmp_path):
    event, state = lake()
    before = arrays_of(state)
    with pytest.raises(PondEventIncomplete, match="quiet interval"):
        event.settle_retaining_water(state)  # t = 0: rain not finished, no hold (resource caps are not completion)
    mid = event.evolve(state, 0.5).state  # rain off but the hold interval is not complete
    before_mid = arrays_of(mid)
    with pytest.raises(PondEventIncomplete):
        event.settle_retaining_water(mid)
    assert_unchanged(before, state)
    assert_unchanged(before_mid, mid)
    q = quiet(event, state)
    dry = event.finalize(q).dry_state
    with pytest.raises(PondEventIncomplete, match="only a wet state"):
        event.settle_retaining_water(dry)


def test_forced_gate_control_returns_a_staged_wet_state_control_flow_only_not_physics_proof(kind, tmp_path):
    """CONTROL of the return/resume flow with the post-settlement gate FORCED false; the physics proof is the real morphology tests below."""
    event, state = build(kind, tmp_path, soil=0.01)
    q = quiet(event, state)
    real = event.quiet_metrics
    calls = {"n": 0}

    def wake(s):  # the gate that holds BEFORE the settlement is real; the post-settlement gate is forced not to hold (morphology woke the flow)
        calls["n"] += 1
        ok, v = real(s)
        return (ok, v) if calls["n"] == 1 else (False, v)

    event.quiet_metrics = wake
    h = event.settle_retaining_water(q)
    event.quiet_metrics = real
    assert h.status == "requires_recession" and h.reason == "quiet_gates_not_held" and not h.complete and h.state.status == "wet"
    assert not h.state.quiet.ready and h.state.quiet.since_s is None  # the monitor only was reset
    for n in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):  # water, faces and clock are untouched
        assert host(getattr(h.state.hydraulic, n)).tobytes() == host(getattr(q.hydraulic, n)).tobytes()
    assert h.state.t_s == q.t_s and not np.any(host(h.state.bed.water.mobile_mass_by_cell_class_kg))
    assert h.accounting["final_audit"]["water"]["closes"] and h.accounting["final_audit"]["sediment"]["closes"]
    with pytest.raises(PondEventIncomplete):
        event.settle_retaining_water(h.state)  # no false finality: the real gates must hold again first
    again = quiet(event, h.state, end=60.0)  # the caller actually evolves rain 0 (real recession) from the RETURNED state
    assert again.t_s > q.t_s
    done = event.settle_retaining_water(again)
    assert done.complete and float(host(done.state.hydraulic.depth_m).sum()) > 0.0


RELAXED = PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True, max_unit_discharge_m2_s=2.0e-6, max_depth_change_m=1.0e-4)


def test_real_terminal_geometry_refresh_invalidates_the_quiet_certificate_and_real_recession_recertifies(tmp_path):
    """REAL morphology (no forced gate): a 1 kg terminal deposit forces a genuine net-elevation routing refresh at the final commit; the closed pond's
    old face fluxes are 0 so the instantaneous gates still pass, which used to return `complete`. One real rain-0 step then shows a non-quiet flow."""
    event, state = lake(load=1.0, soil=0.01, dt=1.0)
    q = quiet(event, state)
    assert not np.any(host(q.hydraulic.qx_m2_s)) and not np.any(host(q.hydraulic.qy_m2_s))
    h = event.settle_retaining_water(q)
    assert h.accounting["geometry_refreshed_at_final"] is True and h.accounting["terminal_deposition_by_class_kg"][2] == pytest.approx(1.0)
    assert h.status == "requires_recession" and h.reason == "routing_geometry_refreshed" and not h.complete
    assert h.accounting["wake_reason"] == "routing_geometry_refreshed" and h.state.topology.generation == q.topology.generation + 1
    for n in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):  # nothing erased, no fake epsilon step, no clock reset
        assert host(getattr(h.state.hydraulic, n)).tobytes() == host(getattr(q.hydraulic, n)).tobytes()
    assert h.state.t_s == q.t_s and not h.state.quiet.ready and h.state.quiet.since_s is None
    assert not np.any(host(h.state.bed.water.mobile_mass_by_cell_class_kg))
    with pytest.raises(PondEventIncomplete):
        event.settle_retaining_water(h.state)  # an immediate retry cannot complete: the hold must be re-established by real steps
    step = event.step(h.state, 0.0, 0.01)  # ONE actual native rain-0 step on the returned state
    qmax = max(float(np.max(np.abs(host(step.state.hydraulic.qx_m2_s)))), float(np.max(np.abs(host(step.state.hydraulic.qy_m2_s)))))
    assert qmax > 1.0e-6 > event.policy.max_unit_discharge_m2_s and not event.quiet_metrics(step.state)[0]  # genuinely flowing again


def test_real_recession_from_the_returned_state_reaches_a_fresh_hold_then_completes_with_an_explicit_policy(tmp_path):
    """Actual recession (Dense): with the explicit completion policy `RELAXED` (max unit discharge 2e-6 m2/s; the DEFAULT 1e-10 is not reached within 900 s
    of real steps on this fixture -- recorded limit), real rain-0 evolution from the RETURNED state establishes a fresh hold and the second call completes."""
    event, state = lake(load=1.0, soil=0.01, dt=1.0, policy=RELAXED)
    q = quiet(event, state)
    h = event.settle_retaining_water(q)
    assert h.status == "requires_recession" and h.reason == "routing_geometry_refreshed"
    res = event.evolve(h.state, h.state.t_s + 900.0, stop_when_quiet=True)
    assert res.quiet_ready and res.state.t_s > h.state.t_s + RELAXED.hold_s  # a REAL fresh continuous hold after real steps
    done = event.settle_retaining_water(res.state)
    assert done.complete and done.reason is None and done.accounting["geometry_refreshed_at_final"] is False
    assert float(host(done.state.hydraulic.depth_m).sum()) > 0.0 and done.accounting["final_audit"]["water"]["closes"]


def test_windowed_step_requires_recession_receipt_equals_the_audit_of_the_returned_state(tmp_path):
    from maple_syrup.pond_bed import pending_journal_records

    event, state = lake(load=1.0, soil=0.01, dt=1.0, windowed=True, tmp_path=tmp_path)
    q = quiet(event, state)
    h = event.settle_retaining_water(q)
    assert h.status == "requires_recession" and h.reason == "routing_geometry_refreshed"
    actual = event.audit(h.state)
    for block in ("storage", "water", "sediment"):
        assert repr(h.accounting["final_audit"][block]) == repr(actual[block]), block  # the returned receipt describes the RETURNED state
    assert h.accounting["final_audit"]["storage"]["pending_journal_records"] == pending_journal_records(h.state.bed) == 0
    assert h.accounting["final_audit"]["storage"]["deep_store_version_id"] == int(h.state.bed.voxel_column.deep_store.version_id)
    published = int(h.state.bed.voxel_column.deep_store.version_id) != int(q.bed.voxel_column.deep_store.version_id)
    assert h.state.counters.n_storage_publications == q.counters.n_storage_publications + int(published)


def test_no_extra_elapsed_time_et_or_drainage(tmp_path):
    event, state = lake(soil=0.01)
    q = quiet(event, state)
    h = event.settle_retaining_water(q)
    assert h.state.t_s == q.t_s
    assert h.state.water.rain.tobytes() == q.water.rain.tobytes() and h.state.water.drain.tobytes() == q.water.drain.tobytes()
    assert h.state.water.intake.tobytes() == q.water.intake.tobytes() and h.state.water.n_steps == q.water.n_steps


def test_native_failure_leaves_the_parent_and_prior_state_unchanged(tmp_path, monkeypatch):
    import maple_syrup.pond_event as pe

    event, state = lake(soil=0.01)
    q = quiet(event, state)
    before = arrays_of(q)
    monkeypatch.setattr(pe, "apply_bed_demand", lambda *a, **k: (_ for _ in ()).throw(ValueError("MAPLE refused")))
    with pytest.raises(ValueError, match="MAPLE refused"):
        event.settle_retaining_water(q)
    monkeypatch.undo()
    assert_unchanged(before, q)
    assert event.settle_retaining_water(q).complete  # the same accepted parent still completes


def test_audit_failure_publishes_nothing_windowed(tmp_path, monkeypatch):
    from maple_syrup.pond_bed import pending_journal_records

    event, state = lake(windowed=True, tmp_path=tmp_path)
    q = quiet(event, state)
    version = q.bed.voxel_column.deep_store.version_id
    real = event.audit
    n = {"i": 0}

    def late_fail(s):
        n["i"] += 1
        out = real(s)
        if n["i"] >= 3:  # the post-settlement audit
            out["sediment"] = {**out["sediment"], "closes": False}
        return out

    event.audit = late_fail
    with pytest.raises(PondEventError):
        event.settle_retaining_water(q)
    event.audit = real
    assert q.bed.voxel_column.deep_store.version_id == version and pending_journal_records(q.bed) == 0
    assert event.settle_retaining_water(q).complete


def test_legacy_default_is_unchanged_against_the_pre_edit_reference_sources(tmp_path):
    """Strong equivalence: the PRE-EDIT `pond_event.py` (archived byte-identical copy) and the edited module, run on separate physically equivalent
    instances, produce identical physical arrays, counters, ledger and audit scalars for `complete`, plus identical exceptions for refused calls."""
    pre = HERE.parents[2] / "completion_program_20261113" / "e4a_pre_sources" / "pond_event.py"
    assert pre.is_file()
    spec = importlib.util.spec_from_file_location("pond_event_pre_edit", pre)
    old = importlib.util.module_from_spec(spec)
    sys.modules["pond_event_pre_edit"] = old
    spec.loader.exec_module(old)
    import maple_syrup.pond_event as new

    results = []
    for mod in (old, new):
        z = basin()
        mobile = np.zeros((3, 5, 6))
        mobile[1, 2, 2] = LOAD
        bed, ctx = make_bed(z, depth=np.maximum(0.215 - z, 0.0), mobile=mobile)
        topo = make_topology(bed, [], allow_closed=True)
        from maple_syrup.experimental_storm import ExperimentalControl
        from maple_syrup.infiltration import column_parameters
        from maple_syrup.rainfall import constant_rainfall
        from maple_syrup.sediment_physics import plot1_sediment_parameters

        shape = topo.shape
        def full(v, shape=shape):
            return np.full(shape, float(v))

        column = column_parameters(model="fixed_ksat", ksat_m_per_s=full(0.0), suction_m=full(0.01), drainage_parameter=full(0.0), theta_sat=full(0.4),
                                   soil_thickness_m=full(0.3), active_mask=np.asarray(topo.active))
        ctl = mod.PondEventControl(backend="numpy", time=ExperimentalControl(max_dt_s=0.5, min_dt_s=1.0 / 1024.0, max_retries=12, max_steps=50_000))
        ev = mod.PondEvent(ctx, column, plot1_sediment_parameters(xp=np), constant_rainfall(0.0, 1.0, 0.0), topo, control=ctl,
                           policy=mod.PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True))
        st = ev.initial_state(bed, np.zeros(ev.shape))
        handoff = ev.complete(st, 20.0)
        dry = handoff.dry_state
        results.append((digest(dry), dict(scalars(dry.counters)), repr(sorted((k, repr(v)) for k, v in handoff.accounting.items() if k != "final_audit")),
                        repr(handoff.accounting["final_audit"]["water"]), repr(handoff.accounting["final_audit"]["sediment"]), ev.config_sha256))
    assert results[0] == results[1]
    ctl_fields = {f.name for f in fields(new.PondEventControl)}
    assert not {"retain_water", "soil_model", "continuous"} & ctl_fields  # no new default control field: repr/config SHA unchanged


def test_source_identity_is_recorded_and_a_configured_wrong_source_fails():
    import maple

    maple_file = Path(maple.__file__).resolve()
    expected = os.environ.get("MAPLE_SYRUP_EXPECTED_MAPLE_ROOT")
    if expected:  # a configured expectation that this process did not satisfy is a FAILURE, never a skip
        assert Path(expected).resolve() in maple_file.parents, f"MAPLE imported from {maple_file}, not from the configured {expected}"
    import maple_syrup.pond_event as pe

    pre = HERE.parents[2] / "completion_program_20261113" / "e4a_pre_sources" / "pond_event.py"
    print({"maple": str(maple_file), "pond_event_sha256": hashlib.sha256(Path(pe.__file__).read_bytes()).hexdigest(),
           "pre_edit_sha256": hashlib.sha256(pre.read_bytes()).hexdigest()})
    assert hashlib.sha256(pre.read_bytes()).hexdigest() != hashlib.sha256(Path(pe.__file__).read_bytes()).hexdigest()


GPU = pytest.mark.skipif(os.environ.get("SYRUP_RELEASE_GPU") != "1", reason="GPU tests are opt-in (SYRUP_RELEASE_GPU=1); definitions for Root, not run by the author")


@GPU
def test_cuda_dense_retained_completion_definition():
    """CUDA DEFINITION (not run): bed AND context come from the existing helper with `xp=cp` (device `initial_available_fraction`), never a CPU context with
    a CuPy bed (MAPLE's namespace guard is not weakened). The retained handoff keeps the device depth/soil/face arrays byte-identical and its audits close."""
    import cupy as cp
    from pond_helpers import backend_namespace

    be = backend_namespace("cuda")
    z = basin()
    mobile = np.zeros((3, 5, 6))
    mobile[1, 2, 2] = LOAD
    bed, ctx = make_bed(z, depth=np.maximum(0.215 - z, 0.0), mobile=mobile, xp=cp)
    topo = make_topology(bed, [], allow_closed=True)
    event = make_event(bed, ctx, topo, backend=be, policy=POLICY, max_dt_s=0.5)
    q = quiet(event, initial(event, bed, cp))
    h = event.settle_retaining_water(q)
    assert h.complete and cp.asnumpy(h.state.hydraulic.depth_m).tobytes() == cp.asnumpy(q.hydraulic.depth_m).tobytes()
    assert h.accounting["final_audit"]["water"]["closes"] and h.accounting["final_audit"]["sediment"]["closes"]


@GPU
def test_cuda_windowed_step_retained_completion_definition(tmp_path):
    """CUDA DEFINITION (not run): genuine device-resident Windowed bed under STEP publication, prepared by the qualified helpers (`dense_deep_bed`,
    `windowed_counterpart(..., xp=cp)`, `initial_available_fraction` taken from the windowed context as `pair` does); NON-ZERO base/material below the
    resident floor; the retained handoff completes (or requires recession) with a receipt equal to `audit(returned state)`."""
    import cupy as cp
    from pond_helpers import backend_namespace

    be = backend_namespace("cuda")
    z = basin()
    mobile = np.zeros((3, 5, 6))
    mobile[1, 2, 2] = LOAD
    dense, ctx = dense_deep_bed(z, depth=np.maximum(0.215 - z, 0.0), mobile=mobile)
    wbed, wctx = windowed_counterpart(dense, ctx, tmp_path / "store", xp=cp)
    assert int(cp.asnumpy(wbed.voxel_column.base).min()) > 0  # material genuinely below the resident floor
    topo = make_topology(dense, [], allow_closed=True)
    event = make_event(wbed, wctx, topo, backend=be, policy=POLICY, max_dt_s=0.5, control=control_for(be, windowed=True, deep_publication="step", max_dt_s=0.5))
    q = quiet(event, event.initial_state(wbed, cp.zeros(event.shape)))
    h = event.settle_retaining_water(q)
    assert h.status in ("complete", "requires_recession")
    for block in ("storage", "water", "sediment"):
        assert repr(h.accounting["final_audit"][block]) == repr(event.audit(h.state)[block])
