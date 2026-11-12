"""Shared eligibility evidence for a genuine initial / closed-boundary INTEREVENT entry (Phase 1, task
phase1_initial_interevent_20261110).

The interevent water owner may begin from an initialized or previously closed physical boundary instead of a just-completed
storm. Whether that instant is hydraulically quiet is decided from the ACTUAL current signed face discharges, the actual
outlet discharge and the actual depression excess, under an explicit existing `LayeredQuietPolicy`. The formulas are the event
gate's (`layered_event._hydraulic_packet` / `_instant_quiet`) restated on a storm operator, so an entry never needs a fabricated
event; a test compares them value for value. This is an INSTANTANEOUS eligibility of an initial condition, never a statement that
a quiet hold was executed. Nothing here drains, rounds or deletes water.

Pure read-only metadata/reduction helpers; no state, registry or physical kernel lives here.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np

from maple_syrup import layered_event as ev
from maple_syrup import layered_storm as ls

__all__ = ["EVIDENCE_NAMES", "POLICY_NAMES", "QUIET_SCRATCH_DOUBLES_PER_CELL", "eligibility", "policy_scalars",
           "quiet_evidence"]

#: Quiet scratch of one evidence evaluation: bed + H, minus escape, clamp, min, where (about six (N,) doubles) plus the face-absolute
#: temporaries (two F doubles, counted separately by the caller); declared as twelve (N,) doubles like the reset evidence.
QUIET_SCRATCH_DOUBLES_PER_CELL = 12

POLICY_NAMES = ("hold_s", "max_depth_change_m", "max_unit_discharge_m2_s", "outlet_discharge_m3_s", "max_excess_depth_m",
                "excess_volume_m3")
EVIDENCE_NAMES = ("max_unit_discharge_m2_s", "max_excess_depth_m", "excess_volume_m3", "outlet_discharge_m3_s",
                  "max_active_surface_depth_m")


def policy_scalars(policy: Any) -> dict[str, float]:
    """Validated plain-float copy of a genuine `LayeredQuietPolicy` (all six fields positive and finite)."""
    if type(policy) is not ev.LayeredQuietPolicy:
        raise ls.LayeredStormError(f"policy must be a genuine LayeredQuietPolicy, got {type(policy).__name__}")
    policy.validated()
    return {name: float(value) for name, value in asdict(policy).items()}


def quiet_evidence(op: ls.LayeredStormOperator, surface_m: Any, qx_m2_s: Any, qy_m2_s: Any) -> dict[str, float]:
    """The actual instantaneous hydraulic evidence of one surface/face state on the storm operator's topology, in the event
    gate's formulas: max |unit discharge| over every signed face, the depression excess `min(max(bed + H - escape, 0), H)` over
    active cells (maximum depth and volume), the outlet discharge `dx (sum |qx| on type-2 x faces + sum |qy| on type-2 y faces)`
    and, for the record, the maximum active surface depth. One batched host read; raises on any non-finite value."""
    from maple.core.backend import to_device

    xp, topo = op.xp, op.topology
    active = op.column.active.reshape(op.shape)
    escape = xp.asarray(to_device(topo.escape_level_m, xp))
    held = xp.asarray(to_device(topo.source_bed_m, xp))
    outlet_x = xp.asarray(to_device(topo.geometry.fx_type == 2, xp))
    outlet_y = xp.asarray(to_device(topo.geometry.fy_type == 2, xp))
    excess = xp.where(active, xp.minimum(xp.maximum(held + surface_m - escape, 0.0), surface_m), 0.0)
    outlet = op.dx_m * (xp.sum(xp.where(outlet_x, xp.abs(qx_m2_s), 0.0)) + xp.sum(xp.where(outlet_y, xp.abs(qy_m2_s), 0.0)))
    packet = ls._packet(xp, [xp.maximum(xp.max(xp.abs(qx_m2_s)), xp.max(xp.abs(qy_m2_s))), xp.max(excess),
                             xp.sum(excess) * op.column.cell_area_m2, outlet, xp.max(xp.where(active, surface_m, 0.0))])
    if not np.all(np.isfinite(packet)):
        raise ls.LayeredStormError("non-finite hydraulic quiet evidence")
    return {name: float(packet[i]) for i, name in enumerate(EVIDENCE_NAMES)}


def eligibility(evidence: dict[str, float], policy: dict[str, float]) -> tuple[bool, str | None]:
    """Instantaneous gate on stored scalars (same comparisons as `layered_event._instant_quiet`). Returns `(ok, first failing
    metric)`; a non-finite or negative scalar is never eligible."""
    for name in EVIDENCE_NAMES:
        v = evidence[name]
        if not (isinstance(v, float) and math.isfinite(v) and v >= 0.0):
            return False, name
    checks = (("max_unit_discharge_m2_s", "max_unit_discharge_m2_s"), ("max_excess_depth_m", "max_excess_depth_m"),
              ("excess_volume_m3", "excess_volume_m3"), ("outlet_discharge_m3_s", "outlet_discharge_m3_s"))
    for ev_name, pol_name in checks:
        if not evidence[ev_name] <= policy[pol_name]:
            return False, ev_name
    return True, None
