"""Builders of the EB3 suite: real native thin-lake storms opened through the genuine lifecycle opener (static event factory, no fakes)."""
from __future__ import annotations

import fpond_builders as fb
import numpy as np
from case import AVAIL
from fpond_builders import ph

from maple_syrup.pond_event import PondCompletionPolicy, PondEvent


def lake_kwargs(backend, *, splash=False, preferential=False, loose=False, bottom="closed", mobile_kg=0.1):
    z = ph.basin() - 0.08
    mob = np.zeros((3, 5, 6))
    mob[1, 2, 2] = mobile_kg
    pol = PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-6, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True) if preferential else PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)
    if loose:  # declared looser quiet thresholds (the default thresholds never quiet a slowly equilibrating returned-water lake within a bounded test)
        pol = PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-4, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True)
    return {"surface": z, "depth": np.maximum(0.135 - z, 0.0), "mobile": mob, "closed": True, "bottom": bottom, "saturation": 0.999999, "backend": backend, "splash": splash, "rain_mm_h": 0.0, "end_s": 2.0,
            "max_dt_s": 1.0, "policy": pol}


def build(tmp, backend, windowed, *, splash=False, preferential=False, loose=False, bottom="closed", mobile_kg=0.1):
    """`(owner, bed, factory, pref, ev0)`: a CURRENT owner on the actual bed and the caller's static event factory (the original event/vertical are only used to read the static case)."""
    kw = lake_kwargs(backend, splash=splash, preferential=False, loose=loose, bottom=bottom, mobile_kg=mobile_kg)
    if windowed:
        ev0, _st0, _vert0, owner, bed, _c = fb.make_pair(tmp, **kw)[1]
    else:
        ev0, _st0, bed, _c, _vert0, owner = fb.make(**kw)
    pref = None
    if preferential:
        from maple_syrup import floating_preferential_infiltration as fpi

        pref = fpi.prepare_floating_preferential(owner.column, a3=np.full(owner.column.n_cells, 2.0e-3), max_fraction_per_hour=0.5, max_intake_rate_m_s=1e-5, background_receiver_weights=[0.0, 1.0, 1.0], available_bytes_fn=AVAIL)

    def factory(vert, topology=None, schedule=None):
        return PondEvent(ev0.context, None, ev0.sediment, ev0.schedule if schedule is None else schedule, ev0._initial_topology if topology is None else topology, control=ev0.control, policy=ev0.policy, splash=ev0.splash, floating_vertical=vert)

    return owner, bed, factory, pref, ev0


def run_to_complete(ev, st, max_rounds=8):
    """Real quiet + terminal settlement + recession loop until a genuine COMPLETE handoff (the bounded loop of the retained-completion tests)."""
    h = None
    state = st
    for _ in range(max_rounds):
        q = ev.evolve(state, float(state.t_s) + 200.0, stop_when_quiet=True)
        state = q.state
        if q.quiet_ready:
            h = ev.settle_retaining_water(state)
            state = h.state
            if h.complete:
                return h
    raise AssertionError("the storm did not reach a genuine complete handoff in the bounded loop")


__all__ = ["AVAIL", "build", "lake_kwargs", "run_to_complete"]
