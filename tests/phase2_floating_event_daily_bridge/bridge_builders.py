"""EB-D builders: a REAL native storm (light rain fully absorbed by an unsaturated open-bottom plane -> a genuine COMPLETE retained boundary with exactly zero surface water) opened through the genuine lifecycle."""
from __future__ import annotations

import fpond_builders as fb
import numpy as np
from case import AVAIL
from fpond_builders import ph

from maple_syrup import floating_event_lifecycle as lc
from maple_syrup.pond_event import PondCompletionPolicy, PondEvent

RUN = "ab" * 32


def build(tmp, backend, windowed, *, rain_mm_h=10.0, end_s=60.0, depth=0.0, mobile_kg=None, bottom="open", saturation=0.5):
    pol = PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-4, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True)
    kw = {"surface": ph.plane(base=0.12, drop=0.004), "depth": depth, "bottom": bottom, "saturation": saturation, "backend": backend, "splash": False, "rain_mm_h": rain_mm_h, "end_s": end_s, "max_dt_s": 0.2, "policy": pol}
    if mobile_kg is not None:  # a real mobile pool: its terminal deposit makes the gross dirty set non-empty (a genuine property refresh)
        mob = np.zeros((3, 4, 6))
        mob[1, 2, 2] = mobile_kg
        kw["mobile"] = mob
    if windowed:
        ev0, _st0, _v, owner, bed, _c = fb.make_pair(tmp, **kw)[1]
    else:
        ev0, _st0, bed, _c, _v, owner = fb.make(**kw)

    def factory(vert, topology=None, schedule=None):
        return PondEvent(ev0.context, None, ev0.sediment, ev0.schedule if schedule is None else schedule, ev0._initial_topology if topology is None else topology, control=ev0.control, policy=ev0.policy,
                         splash=ev0.splash, floating_vertical=vert)

    return owner, bed, factory


def complete(ev, st, rounds=8, span=100.0):
    state = st
    for _ in range(rounds):
        q = ev.evolve(state, float(state.t_s) + span, stop_when_quiet=True)
        state = q.state
        if q.quiet_ready:
            h = ev.settle_retaining_water(state)
            state = h.state
            if h.complete:
                return h
    raise AssertionError("the storm did not reach a genuine complete handoff in the bounded loop")


def dry_boundary(tmp, backend, windowed, **kw):  # kw: rain_mm_h, end_s, depth, bottom, saturation
    """`(lineage, boundary, factory)` after a genuine native storm that left exactly zero standing water."""
    owner, bed, factory = build(tmp, backend, windowed, **kw)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL)
    h = complete(occ.event, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert float(np.asarray(b.surface_depth_m).max()) == 0.0
    return lin, b, factory


def shallow_basin_build(tmp, backend, windowed, *, level_m=0.0015, rain_mm_h=0.0, end_s=2.0, pref=False):
    """A CLOSED basin holding a thin standing film (max depth `level_m`): a genuine COMPLETE retained boundary WITH standing water whose wet-day fragments are tractable."""
    z = ph.basin() - 0.08
    depth = np.maximum(float(z.min()) + level_m - z, 0.0)
    pol = PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-4, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True)
    kw = {"surface": z, "depth": depth, "closed": True, "bottom": "open", "saturation": 0.6, "backend": backend, "splash": False, "rain_mm_h": rain_mm_h, "end_s": end_s, "max_dt_s": 1.0, "policy": pol}
    if windowed:
        ev0, _st0, _v, owner, bed, _c = fb.make_pair(tmp, **kw)[1]
    else:
        ev0, _st0, bed, _c, _v, owner = fb.make(**kw)

    def factory(vert, topology=None, schedule=None):
        return PondEvent(ev0.context, None, ev0.sediment, ev0.schedule if schedule is None else schedule, ev0._initial_topology if topology is None else topology, control=ev0.control, policy=ev0.policy,
                         splash=ev0.splash, floating_vertical=vert)

    if pref:
        from maple_syrup import floating_preferential_infiltration as fpi

        factory.pref = fpi.prepare_floating_preferential(owner.column, a3=np.full(owner.column.n_cells, 2.0e-3), max_fraction_per_hour=0.5, max_intake_rate_m_s=1e-5, background_receiver_weights=[0.0, 1.0, 1.0],
                                                         available_bytes_fn=AVAIL)
    return owner, bed, factory


def wet_boundary(tmp, backend, windowed, **kw):
    owner, bed, factory = shallow_basin_build(tmp, backend, windowed, **kw)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL, preferential=getattr(factory, "pref", None))
    h = complete(occ.event, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    assert float(np.asarray(b.surface_depth_m).max()) > 0.0
    return lin, b, factory


def dry_boundary_changed(tmp, monkeypatch, backend="numpy", windowed=False, mobile_kg=0.1, scale=0.9):
    """`(lineage, boundary, factory, original_root_sha)`: a dry boundary whose FIRST event deposits a real mobile pool; the provider output of that refresh is scaled (DECLARED perturbation) so the properties genuinely change."""
    from maple_syrup import native_floating_soil as nfs

    owner, bed, factory = build(tmp, backend, windowed, mobile_kg=mobile_kg)
    orig = owner.column.sha256
    real = nfs._batch_properties
    monkeypatch.setattr(nfs, "_batch_properties", lambda b_, c_, k_, p_: (lambda f, sa, k: (f * scale, sa * scale, k))(*real(b_, c_, k_, p_)))
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL)
    h = complete(occ.event, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    monkeypatch.undo()
    assert float(np.asarray(b.surface_depth_m).max()) == 0.0
    return lin, b, factory, orig
