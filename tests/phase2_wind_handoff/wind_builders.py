"""Builders of the wind-handoff suite (authored; NOT executed by their author: no shell was available).

Real native objects only: the thin Dense NumPy beds, owners, static event factories and lifecycle of the existing authored suites. A VALID wind CHILD is a real `BedState`: a `copy.deepcopy` like the MASY executor's staged copy, or the result of the ACTUAL native
sediment exchange + ledger APIs (`child_move`: `apply_bed_demand` pickup and deposit, so the native validators and the ledger reconciliation accept it). Direct native-class construction (`child_class_swap`) and a lone pickup (`child_pickup`) are MECHANICAL /
NEGATIVE fixtures only. The finalization result is an INSTANCE OF THE ACTUAL native `maple.aeolian.scheduler.finalization.WindEventFinalizationResult` (and real `SchedulerState` / mobile pool) constructed directly with real
arrays: genuine MAPLE `finalize_wind_event` execution is NOT performed here (that needs the MASY wind component and belongs to the root tests). No fake physical result class exists.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import fpond_builders as fb
import numpy as np
from fpond_builders import ph

from maple_syrup import floating_wind_handoff as wh
from maple_syrup.pond_event import PondCompletionPolicy, PondEvent

AVAIL = fb.AVAIL
RUN = "ab" * 32
LIMITS = wh.WindHandoffLimits(max_cells=10**6, max_voxels=10**4, max_classes=64, max_batch_cells=10**4, max_scan_bytes=10**13, max_candidate_bytes=10**13, max_receipt_bytes=200_000, max_cohorts=10**6)


def build(*, depth=0.0, bottom="open", saturation=0.5, base=0.12, mobile=None, backend="numpy", surface=None):
    """`SimpleNamespace(owner, bed, factory, ev0, ctx)`: a CURRENT owner on a real thin Dense bed and the caller's static event factory (the original event/vertical only provide the static case). `base=0.20` gives a soil deeper than H = 0.16 m."""
    pol = PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-4, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True)
    kw = {"surface": ph.plane(base=base, drop=0.004) if surface is None else surface, "depth": depth, "bottom": bottom, "saturation": saturation, "backend": backend, "splash": False, "rain_mm_h": 10.0, "end_s": 60.0,
          "max_dt_s": 0.2, "policy": pol, "mobile": mobile}
    ev0, _st0, bed, ctx, _v, owner = fb.make(**kw)

    def factory(vert, topology=None, schedule=None):
        return PondEvent(ev0.context, None, ev0.sediment, ev0.schedule if schedule is None else schedule, ev0._initial_topology if topology is None else topology, control=ev0.control, policy=ev0.policy, splash=ev0.splash,
                         floating_vertical=vert)

    return SimpleNamespace(owner=owner, bed=bed, factory=factory, ev0=ev0, ctx=ctx, original=ev0._initial_topology)


def started(**kw):
    """The case after `start_wind_boundary`: adds `lin`, `b` (the initial Boundary) and `wctx`."""
    ns = build(**kw)
    ns.lin, ns.b, ns.wctx = wh.start_wind_boundary(ns.owner, ns.bed, ns.factory, bed_context=ns.ctx, limits=LIMITS, available_bytes_fn=AVAIL)
    return ns


def child_same(bed):
    """A separate staged copy with identical values (exactly what the MASY executor stages by `copy.deepcopy`)."""
    return copy.deepcopy(bed)


def _pickup_class(bed, ctx, cell, klass, fraction=0.25):
    """ACTUAL native pickup (`apply_bed_demand`) of a fraction of ONE grain class of the cell's active layer."""
    from maple.water import WaterProcessDemand

    from maple_syrup.sediment_bed import apply_bed_demand

    act = ph.host(bed.active_layer.mass_kg)
    demand = np.zeros(act.shape)
    demand[cell[0], cell[1], klass] = fraction * act[cell[0], cell[1], klass]
    return apply_bed_demand(bed, ctx, WaterProcessDemand(np.asarray(demand), np.zeros_like(bed.active_layer.mass_kg)))


def child_move(bed, ctx, src=(1, 1), dst=(1, 2), dz_m=1e-4, klass=None):
    """A GENUINE, native-valid wind-like child: two ACTUAL native MAPLE sediment calls (`pond_helpers.move_mass`: `apply_bed_demand` pickup at `src`, then deposit of exactly the removed mass at the D4-adjacent `dst`) that update the voxels, active layers and the
    sediment LEDGER together, so the actual native checkpoint validators and the ledger reconciliation accept the bed. `klass=None` moves the cell's own mixture (depth change, composition ~ same); an integer moves that ONE class (the composition of both columns
    changes, net elevation of the pair is conserved). The water leaves are the parent's bit for bit (wind is dry) and the result is a fully separate deep copy (no shared native array)."""
    if klass is None:
        moved = ph.move_mass(bed, ctx, src, dst, dz_m)
    else:
        from unittest import mock

        with mock.patch.object(ph, "pickup_cell", lambda b, c, cell, dz, xp=np: _pickup_class(b, c, cell, klass)):  # the same real native pickup, class-specific demand
            moved = ph.move_mass(bed, ctx, src, dst, dz_m)
    return copy.deepcopy(replace(moved, water=bed.water))


def child_class_swap(bed, cell=(1, 1), klass=0):
    """MECHANICAL / NEGATIVE FIXTURE ONLY (never a valid wind child): direct native-class construction. Every voxel of ONE column is overwritten to hold its whole mass in a single grain class: per-voxel totals are unchanged but no native ledger records the
    change, so the actual native bed validity refuses it. It exists to prove that refusal; valid fixtures use `child_move`."""
    ch = copy.deepcopy(bed)
    m = ch.voxel_column.mass_kg
    for k in range(m.shape[2]):
        tot = m[cell[0], cell[1], k].sum()
        if tot > 0.0:
            m[cell[0], cell[1], k, :] = 0.0
            m[cell[0], cell[1], k, klass] = tot
    return ch


def child_pickup(bed, ctx, cell=(1, 1), dz_m=1e-4):
    """MECHANICAL fixture (not a checkpoint-valid wind child): a lone native MAPLE pickup lowering one column whose removed mass stays in the mobile pool; the water leaves are then replaced by the parent's. Valid fixtures use `child_move`."""
    moved, _res = ph.pickup_cell(bed, ctx, cell, dz_m)
    return copy.deepcopy(replace(moved, water=bed.water))  # a fully separate staged copy: no native array is shared with the parent bed


def native_result(child, end_s, *, block_index=1, block_offset_s=0.0, completed=True, status="completed", reasons=(), mobile_state=None, physical_time_s=None, class_kw=None, bounds=(1e-12, 1e-12)):
    """An instance of the ACTUAL native `WindEventFinalizationResult` whose finalized scheduler state shares the child's fields (real `SchedulerState`, real empty cohort pool). NOT produced by `finalize_wind_event`."""
    from maple.aeolian.scheduler.finalization import WindEventFinalizationResult
    from maple.aeolian.scheduler.state import initial_scheduler_state
    from maple.core.types.cohort import zeros_mobile_cohort_state
    from maple.core.types.threshold_modifiers import neutral_threshold_modifier_state

    ny, nx = child.water.depth_m.shape
    nc = child.active_layer.mass_kg.shape[2]
    sched = initial_scheduler_state(child.voxel_column, child.active_layer, zeros_mobile_cohort_state(0) if mobile_state is None else mobile_state, child.ledger, child.committed_topography,
                                    neutral_threshold_modifier_state(ny, nx), time_s=float(end_s if physical_time_s is None else physical_time_s), sediment_availability=child.sediment_availability, water=child.water)
    sched = replace(sched, block_index=block_index, block_offset_s=block_offset_s)
    vec = {n: np.zeros(nc) for n in ("airborne_mass_before_by_class_kg", "airborne_mass_after_by_class_kg", "deposited_mass_by_class_kg", "boundary_export_mass_by_class_kg", "bed_change_by_class_kg",
                                     "transfer_identity_gap_by_class_kg", "reservoir_identity_gap_by_class_kg")}
    vec.update(class_kw or {})
    return WindEventFinalizationResult(completed=completed, status=status, reasons=tuple(reasons), state=sched, time_s=float(end_s), block_index=block_index, block_offset_s=block_offset_s, representation="dense",
                                       placement_rule="test", numerical_residual_scalar_kg=0.0, transfer_identity_bound_kg=bounds[0], reservoir_identity_bound_kg=bounds[1], n_settled=0, n_exported=0, voxel_growth_event=None, **vec)


def stage(ns, child, *, end_s=120.0, final=None, parent=None, **over):
    final = native_result(child, end_s) if final is None else final
    kw = {"context": ns.wctx, "occurrence_id": "occ-1", "start_s": float(ns.b.t_s), "limits": LIMITS, "available_bytes_fn": AVAIL}
    kw.update(over)
    return wh.stage_wind_handoff(ns.lin, ns.b if parent is None else parent, final, child, **kw)


def wind(ns, child, **kw):
    """Stage + adopt; the new head is stored on `ns.b`. Returns `(old_boundary, new_boundary, candidate)`."""
    old = ns.b
    cand = stage(ns, child, **kw)
    ns.b = wh.adopt_wind_handoff(ns.lin, old, cand)
    return old, ns.b, cand


def snapshot(b):
    """Host copies of everything a refused handoff must leave unchanged."""
    r = b.root
    out = {"W": np.array(r.water_state.water_m, copy=True), "t": float(r.water_state.t_s), "J": np.array(r.history.wetting_m, copy=True), "I": np.array(r.history.cumulative_intake_m, copy=True), "D": np.array(r.depth_m, copy=True),
           "H": np.array(b.surface_depth_m, copy=True), "qx": np.array(b.qx_m2_s, copy=True), "qy": np.array(b.qy_m2_s, copy=True), "vox": np.array(b.bed.voxel_column.mass_kg, copy=True),
           "act": np.array(b.bed.active_layer.mass_kg, copy=True), "ledger": dict(b.ledger), "t_s": b.t_s, "seq": b.seq, "col": r.column.sha256, "op": r.matrix_op.sha256, "gen": r.generation}
    return out


def assert_same(a, b):
    for k in a:
        if isinstance(a[k], np.ndarray):
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        else:
            assert a[k] == b[k], k
