# ruff: noqa: RUF059
"""Phase 2.5 wind / vegetation compatibility: WH1 (`floating_wind_handoff`, Dense NumPy, no-LEAF scope) must FAIL CLOSED on a vegetation-bound boundary instead of silently adopting it and losing the generation record.

TRIGGER (confirmed before the guard existed, with a GENUINE applied boundary): `wind_builders.started` (a real initial Boundary from `start_wind_boundary`) -> the real LEAF P1 projection and S0 binding applied by the OWNING
`vegetation_lifecycle.apply_vegetation_generation` -> `stage_wind_handoff` + `adopt_wind_handoff` of an unchanged wind child ADOPTED and the new head's `root.vegetation` was `None` (the generation record, bound covers and recipe were dropped).
The wind result in these tests is the mechanical, typed `WindEventFinalizationResult` of `wind_builders.native_result` (an instance of the actual native class built directly); the genuine MAPLE wind simulation (`finalize_wind_event`) is NOT run
here, and no producer provenance is pretended. LEAF is never faked."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import veg_lc_builders as vb
import wind_builders as wb
from lcp_builders import PART

from maple_syrup import floating_daily as fd
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_wind_handoff as wh
from maple_syrup import recharge as rc
from maple_syrup import vegetation_lifecycle as vl

AVAIL = wb.AVAIL
SRC = Path(wh.__file__).resolve().parent
TASK = Path(__file__).resolve().parents[3]


def vegetated(leaf, **kw):
    """`ns` of a real wind-started lineage whose head is a genuine Boundary carrying ONE applied vegetation generation (owning apply). `ns.plain` is the boundary BEFORE the apply."""
    ns = wb.started(**kw)
    ns.plain = ns.b
    s = vb.stack(leaf, ns.b)
    bd = vb.binding(leaf, ns.b, s)
    ns.b = vl.apply_vegetation_generation(ns.lin, ns.b, bd, memory_budget_bytes=vb.BIG, available_bytes_fn=AVAIL)
    assert ns.b.root.vegetation is not None and ns.lin.head is ns.b
    return ns


def run_bound_day(st, ordinal):
    """One genuine native bridge day on a vegetation-bound lineage (the E-shading cover is the BOUND one: `cover=None`)."""
    n = st.ctx.n_cells
    r = br.day_action(st, plan=rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=wb.RUN, ordinal=ordinal), pet_terrain_m_s=np.full(n, 3e-8), cover=None, available_bytes_fn=AVAIL)
    assert r.outcome == "interevent"
    t = fd.request_floating_withdrawal(r.state, expected_parent=fd.expect_parent(r.state), request_m=np.full((st.ctx.n_layers, n), 1e-6), expected_sequence=0, available_bytes_fn=AVAIL)
    return fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), host_available_bytes_fn=lambda: vb.BIG, disk_available_bytes_fn=lambda: vb.BIG)


def head_state(ns):
    rec = vl.current_record(ns.b)
    return wb.snapshot(ns.b), (rec, rec.generation, rec.recipe_id, ns.lin.head, ns.b.root.vegetation)


def assert_head_unchanged(ns, before):
    snap, (rec, gen, rid, head, veg) = before
    wb.assert_same(snap, wb.snapshot(ns.b))
    assert ns.lin.head is head is ns.b and ns.b.root.vegetation is veg and vl.current_record(ns.b) is rec and (rec.generation, rec.recipe_id) == (gen, rid)
    assert vl.verify_record(rec) is rec


class Counter:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return 10**15


def test_stage_refuses_a_vegetation_bound_boundary_before_any_read_callback_or_array_work(leaf):
    ns = vegetated(leaf, depth=0.003)
    before = head_state(ns)
    reads = Counter()
    # EVERY other argument is a type the later stages would refuse or crash on (no child BedState, no final result, no context, no limits): reaching any of them would raise a different error
    with pytest.raises(wh.WindHandoffError, match="vegetation-aware wind handoff is not qualified"):
        wh.stage_wind_handoff(ns.lin, ns.b, object(), object(), context=object(), occurrence_id="occ-1", start_s=float(ns.b.t_s), limits=object(), available_bytes_fn=reads)
    assert reads.n == 0  # the real reader was never called: no admission, no array allocation
    assert_head_unchanged(ns, before)
    # the same with a fully valid child / result / context: still refused, nothing read
    child = wb.child_same(ns.b.bed)
    final = wb.native_result(child, 120.0)
    with pytest.raises(wh.WindHandoffError, match="vegetation"):
        wh.stage_wind_handoff(ns.lin, ns.b, final, child, context=ns.wctx, occurrence_id="occ-1", start_s=float(ns.b.t_s), limits=wb.LIMITS, available_bytes_fn=reads)
    assert reads.n == 0
    assert_head_unchanged(ns, before)
    assert issubclass(wh.WindHandoffError, lc.LifecycleError)


def test_the_refused_vegetation_boundary_stays_usable_for_a_native_water_day(leaf):
    ns = vegetated(leaf, depth=0.0)  # a DRY boundary: the accepted soil-only native day needs no event factory
    before = head_state(ns)
    child = wb.child_same(ns.b.bed)
    with pytest.raises(wh.WindHandoffError):
        wb.stage(ns, child, end_s=120.0)
    assert_head_unchanged(ns, before)
    st = br.open_day(ns.lin, ns.b, partition=PART, run_digest=wb.RUN, available_bytes_fn=AVAIL)  # the native daily path still works on the untouched vegetated head
    closed = run_bound_day(st, 0).state
    assert closed.ctx.hold is ns.lin.head and closed.cursor.ordinal == 1
    back = br.hand_back(ns.lin, closed)
    assert back.root.vegetation is ns.b.root.vegetation and vl.current_record(back).recipe_id == before[1][2]  # the generation survives the native day


def test_a_staged_candidate_cannot_be_adopted_after_a_vegetation_generation_was_applied_and_the_stale_parent_is_still_refused_natively(leaf):
    ns = wb.started(depth=0.003)
    plain = ns.b
    child = wb.child_same(plain.bed)
    cand = wb.stage(ns, child, end_s=120.0)  # a legitimate staging on the NON-vegetated parent (WH1 behavior preserved)
    s = vb.stack(leaf, plain)
    bd = vb.binding(leaf, plain, s)
    nb = vl.apply_vegetation_generation(ns.lin, plain, bd, memory_budget_bytes=vb.BIG, available_bytes_fn=AVAIL)  # the head becomes the vegetated boundary
    ns.b = nb
    before = head_state(ns)
    with pytest.raises(lc.LifecycleError):  # the old parent is no longer the head: the native stale refusal comes first and is unchanged
        wh.adopt_wind_handoff(ns.lin, plain, cand)
    reads = Counter()
    with pytest.raises(wh.WindHandoffError, match="vegetation"):  # the vegetated head as the "parent": the owning-entry guard, before before_publish / admission / integrity work
        wh.adopt_wind_handoff(ns.lin, nb, cand, available_bytes_fn=reads, before_publish=lambda: pytest.fail("the publish callback must not run"))
    assert reads.n == 0 and not cand.consumed
    assert_head_unchanged(ns, before)


def test_restore_wind_context_refuses_a_vegetation_bound_boundary_and_still_rebuilds_a_plain_one(leaf):
    ns = vegetated(leaf, depth=0.003)
    record = ns.wctx.record()
    again = wh.restore_wind_context(ns.plain, ns.ctx, record)  # a non-vegetated boundary: unchanged behavior
    assert again.record() == record
    with pytest.raises(wh.WindHandoffError, match="vegetation"):
        wh.restore_wind_context(ns.b, ns.ctx, record)


def test_a_non_genuine_or_copied_boundary_never_reaches_the_guard_and_is_refused_by_the_owning_lifecycle(leaf):
    import copy

    ns = vegetated(leaf, depth=0.003)
    for bad in (object(), None, ns.b.root):
        with pytest.raises(lc.LifecycleError):
            wh.stage_wind_handoff(ns.lin, bad, object(), object(), context=ns.wctx, occurrence_id="occ-1", start_s=float(ns.b.t_s), limits=wb.LIMITS, available_bytes_fn=AVAIL)
    with pytest.raises((lc.LifecycleError, TypeError)):  # a copied head is not a minted head (the lifecycle's own copy guard); the vegetation field is never trusted from a record
        wh.stage_wind_handoff(ns.lin, copy.copy(ns.b), object(), object(), context=ns.wctx, occurrence_id="occ-1", start_s=float(ns.b.t_s), limits=wb.LIMITS, available_bytes_fn=AVAIL)


def test_the_non_vegetated_wh1_path_is_preserved_stage_adopt_and_no_vegetation_field():
    ns = wb.started(depth=0.003)
    assert ns.b.root.vegetation is None
    old, new, cand = wb.wind(ns, wb.child_same(ns.b.bed), end_s=120.0)
    assert ns.lin.head is new and new.root.vegetation is None and cand.consumed and new.t_s == 120.0
    again = wh.restore_wind_context(new, ns.ctx, ns.wctx.record())
    assert again.record() == ns.wctx.record()


def _historical_wind_source_scope():
    receipt = TASK / "integration_initial_overlay_receipt.json"
    if not receipt.is_file():
        pytest.fail(f"the immutable integration overlay receipt is missing: {receipt} (setup failure, never a skip)")
    d = json.loads(receipt.read_text())
    base = {e["path"]: e["sha256"] for e in d["entries"]}
    now = {p.relative_to(SRC).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.rglob("*.py")}
    assert set(now) == set(base) and len(base) == 180
    changed = sorted(k for k in base if now[k] != base[k])
    assert changed == ["floating_wind_handoff.py"]
    assert base["floating_wind_handoff.py"] == d["new_main_wind_bytes_preserved"]  # the published main bytes are the recorded baseline of this one guarded delta
    text = (SRC / "floating_wind_handoff.py").read_text()
    assert "import vegetation_lifecycle" not in text and "water_vegetation" not in text  # the module stays LEAF-free (code-level imports)


def test_wind_handoff_import_stays_leaf_free_in_a_fresh_process():
    import subprocess
    import sys

    code = "import sys, maple_syrup.floating_wind_handoff; bad=[k for k in sys.modules if k=='leaf' or k.startswith('leaf.') or k in ('maple_syrup.vegetation_lifecycle','maple_syrup.water_vegetation_binding','maple_syrup.water_vegetation_held')]; print(bad)"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=False, env=env)
    assert out.stdout.strip() == "[]", (out.stdout, out.stderr[-400:])
    assert np is not None


def test_the_source_scope_only_the_wind_handoff_module_differs_from_the_recorded_integration_overlay():
    """Historical wind-only delta keeps its original proof; the explicitly locked release proves all later reviewed owner hooks and the exact same guarded WH1 bytes without requiring a task directory."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase25_source_profile"))
    import p25_source_profile as p25

    branch = p25.require_branch(SRC, historical=("wind_compat_c0",))
    if branch.kind == "historical":
        _historical_wind_source_scope()
        return
    proof = branch.verified
    assert proof.wh1_input_sha256 == "a289e35b329bc33ec918768133ada9aef1d37089b110d4085fb8c97c4c09156b"
    assert proof.wh1_sha256 == "15194445f89919f3706f361980a56aac31f2ba1c460d5caaa2c557fa97c09c33"
    text = (SRC / "floating_wind_handoff.py").read_text()
    assert "import vegetation_lifecycle" not in text and "water_vegetation" not in text
