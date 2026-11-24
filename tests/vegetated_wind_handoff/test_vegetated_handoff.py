"""Control/ownership checks with real applied LEAF and Native sediment APIs.

The finalization carrier is the actual Native class built by the existing
wind_builders helper. These checks do not claim a real finalized wind run;
Root separately qualifies the genuine MAPLE wind integration.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import veg_lc_builders as vb
import wind_builders as wb

from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_wind_handoff as wh
from maple_syrup import floating_wind_handoff_vegetated as vwh
from maple_syrup import vegetation_lifecycle as vl


def _depth_only_started():
    """Synthetic depth-only input through genuine owning property overrides.

    All three slabs lie within the actual bed (base .20 > H .16). The four
    supported final-anchor inputs use the existing property-override test
    values; parent and child run the real provider with these same inputs.
    FC6/SAT therefore stay exact while Native pickup/deposit changes depth.
    No provider, Native algorithm, guard or tolerance is patched.
    """
    overrides = {name: np.full((len(wb.fb.LAYERS), 12), value) for name, value in (
        ("theta_1500_override", 0.08), ("theta_33_override", 0.20),
        ("theta_saturation_override", 0.49), ("air_entry_kpa_override", 2.0))}
    policy = wb.PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-4,
        max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True)
    event, _state, bed, context, _vertical, owner = wb.fb.make(
        surface=wb.ph.plane(base=0.20, drop=0.004), depth=0.003,
        bottom="open", saturation=0.5, backend="numpy", splash=False,
        rain_mm_h=10.0, end_s=60.0, max_dt_s=0.2, policy=policy, overrides=overrides)

    def factory(vertical, topology=None, schedule=None):
        return wb.PondEvent(event.context, None, event.sediment,
            event.schedule if schedule is None else schedule,
            event._initial_topology if topology is None else topology,
            control=event.control, policy=event.policy, splash=event.splash, floating_vertical=vertical)

    lineage, boundary, wind_context = wh.start_wind_boundary(owner, bed, factory,
        bed_context=context, limits=wb.LIMITS, available_bytes_fn=wb.AVAIL)
    return SimpleNamespace(lin=lineage, b=boundary, wctx=wind_context, ctx=context)


def _applied(leaf, *, depth_only=False):
    ns = _depth_only_started() if depth_only else wb.started(depth=0.003)
    stack = vb.stack(leaf, ns.b)
    binding = vb.binding(leaf, ns.b, stack)
    ns.b = vl.apply_vegetation_generation(ns.lin, ns.b, binding,
        memory_budget_bytes=vb.BIG, available_bytes_fn=wb.AVAIL)
    return ns


def _stage(ns, child, **kw):
    args = dict(context=ns.wctx, occurrence_id="vegetated-wind-1", start_s=ns.b.t_s,
                limits=wb.LIMITS, available_bytes_fn=wb.AVAIL)
    args.update(kw)
    return vwh.stage_wind_handoff(ns.lin, ns.b, wb.native_result(child, ns.b.t_s + 120.0), child, **args)


@pytest.mark.parametrize("change", ["unchanged", "depth", "texture"])
def test_applied_record_laws_history_and_water_survive_handoff(leaf, change):
    ns = _applied(leaf, depth_only=change == "depth")
    old = ns.b
    record = vl.current_record(old)
    before = wb.snapshot(old)
    law = fm.held_cell_vegetation_factor(old.root.matrix_op).tobytes()
    covers = (record.splash_cover.tobytes(), record.shade_cover.tobytes())
    if change == "unchanged":
        child = wb.child_same(old.bed)
    else:
        # Both helpers perform genuine Native pickup/deposit and ledger updates.
        # The texture selector changes only the authored demand class.
        child = wb.child_move(old.bed, ns.ctx, klass=0 if change == "texture" else None)
    candidate = _stage(ns, child)
    receipt = candidate.receipt()
    new = vwh.adopt_wind_handoff(ns.lin, old, candidate, available_bytes_fn=wb.AVAIL)
    assert ns.lin.head is new and candidate.consumed
    assert candidate.accepted_boundary_ref() is new
    assert vl.current_record(new) is record and new.root.vegetation is record
    assert (record.splash_cover.tobytes(), record.shade_cover.tobytes()) == covers
    assert fm.held_cell_vegetation_factor(new.root.matrix_op).tobytes() == law
    assert new.root.history.wetting_m.tobytes() == old.root.history.wetting_m.tobytes()
    assert new.root.history.cumulative_intake_m.tobytes() == old.root.history.cumulative_intake_m.tobytes()
    assert new.routing is old.routing and new.qx_m2_s is old.qx_m2_s and new.qy_m2_s is old.qy_m2_s
    assert new.cursor is old.cursor and new.t_s == old.t_s + 120.0 == new.root.water_state.t_s
    assert new.ledger["rain_m3"] == old.ledger["rain_m3"]
    assert abs(new.ledger["residual_m3"]) <= new.ledger["residual_bound_m3"]
    assert receipt["vegetation"]["recipe_id"] == record.recipe_id
    assert receipt["vegetation"]["held_factor_sha256_before"] == receipt["vegetation"]["held_factor_sha256_after"]
    assert receipt["vegetation"]["current_after"]["matrix_op_sha256"] == new.root.matrix_op.sha256
    assert receipt["plan"]["vegetation_extra"]["required_bytes"] > 0
    if change == "unchanged":
        assert new.root.column is old.root.column and new.root.matrix_op is old.root.matrix_op
        assert new.root.water_state.water_m.tobytes() == old.root.water_state.water_m.tobytes()
    else:
        props = receipt["properties"]
        assert props["n_dirty"] > 0
        if change == "texture":
            assert props["n_changed"] > 0
        else:  # depth: equivalent hydraulic properties, changed depth and/or Ks
            assert props["n_changed"] == 0 and (props["n_depth_changed"] > 0 or props["ks_changed"])
        assert props["root_replaced"] and new.root.matrix_op is not old.root.matrix_op
        assert new.root.matrix_op.column_sha256 == new.root.column.sha256
        assert new.root.pref_op.column_sha256 == new.root.column.sha256
        assert new.root.pref_op._law is old.root.pref_op._law
        if new.root.column is not old.root.column:
            assert new.root.pref_op is not old.root.pref_op
    wb.assert_same(before, wb.snapshot(old))
    restored = vwh.restore_wind_context(new, ns.ctx, ns.wctx.record())
    assert restored.record() == ns.wctx.record()
    with pytest.raises(vwh.WindHandoffError, match="already adopted"):
        vwh.adopt_wind_handoff(ns.lin, new, candidate)


def test_legacy_guard_and_v2_bare_refusal_stay_separate(leaf):
    ns = _applied(leaf)
    child = wb.child_same(ns.b.bed)
    with pytest.raises(wh.WindHandoffError, match="vegetation"):
        wh.stage_wind_handoff(ns.lin, ns.b, wb.native_result(child, 120.0), child,
            context=ns.wctx, occurrence_id="old", start_s=ns.b.t_s, limits=wb.LIMITS, available_bytes_fn=wb.AVAIL)
    plain = wb.started()
    with pytest.raises(vwh.WindHandoffError, match="requires an applied"):
        _stage(plain, wb.child_same(plain.b.bed))
    candidate = _stage(ns, wb.child_same(ns.b.bed))
    with pytest.raises(wh.WindHandoffError):
        wh.adopt_wind_handoff(ns.lin, ns.b, candidate, available_bytes_fn=wb.AVAIL)
    assert ns.lin.head is ns.b and not candidate.consumed


def test_quota_and_poststage_tamper_refuse_without_publishing(leaf):
    ns = _applied(leaf)
    old = ns.b
    child = wb.child_same(old.bed)
    before = wb.snapshot(old)
    with pytest.raises(vwh.WindHandoffLimitError, match="candidate"):
        _stage(ns, child, limits=replace(wb.LIMITS, max_candidate_bytes=1))
    assert ns.lin.head is old
    candidate = _stage(ns, child)
    child.water.depth_m[0, 0] += 1e-6
    with pytest.raises(vwh.WindHandoffError, match="integrity"):
        vwh.adopt_wind_handoff(ns.lin, old, candidate, available_bytes_fn=wb.AVAIL)
    assert ns.lin.head is old and not candidate.consumed
    wb.assert_same(before, wb.snapshot(old))


def test_changed_context_record_and_stale_candidate_refuse(leaf):
    ns = _applied(leaf)
    old = ns.b
    record = dict(ns.wctx.record(), config_sha256="0" * 64)
    with pytest.raises(vwh.WindHandoffError):
        vwh.restore_wind_context(old, ns.ctx, record)
    first = _stage(ns, wb.child_same(old.bed))
    second = _stage(ns, wb.child_same(old.bed))
    head = vwh.adopt_wind_handoff(ns.lin, old, first, available_bytes_fn=wb.AVAIL)
    with pytest.raises(vwh.WindHandoffError):
        vwh.adopt_wind_handoff(ns.lin, head, second, available_bytes_fn=wb.AVAIL)
    assert ns.lin.head is head and not second.consumed
