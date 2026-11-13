"""S1 builders: REAL native lifecycle states only (a genuine storm -> completion -> selected refresh -> accepted Boundary; a genuine bridge day -> closed state). No fake capability, producer, owner or generation."""
from __future__ import annotations

from types import SimpleNamespace

import fpond_builders as fb
import numpy as np
from bridge_builders import AVAIL, RUN, complete
from bridge_builders import build as _build
from maple_syrup import floating_daily as fd
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_lifecycle_checkpoint as lcp
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup.rainfall import constant_rainfall

IDENT = {"case": "s1", "v": 1}
PART = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
BIG = 10**12


def build(tmp, backend, windowed, *, deep_dense=False, **kw):
    """The bridge builder; `deep_dense=True` selects the DENSE side of the deep layered pair (the exact context / voxel_dz / source of the Windowed counterpart, for the matched fresh-process static recipe)."""
    if not deep_dense:
        return _build(tmp, backend, windowed, **kw)
    from fpond_builders import ph
    from maple_syrup.pond_event import PondCompletionPolicy, PondEvent

    assert not windowed
    pol = PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-4, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True)
    ev0, _st0, _v, owner, bed, _c = fb.make_pair(tmp, surface=ph.plane(base=0.12, drop=0.004), depth=kw.pop("depth", 0.0), bottom="open", saturation=0.5, backend=backend, splash=False, rain_mm_h=10.0, end_s=60.0,
                                                  max_dt_s=0.2, policy=pol, **kw)[0]

    def factory(vert, topology=None, schedule=None):
        return PondEvent(ev0.context, None, ev0.sediment, ev0.schedule if schedule is None else schedule, ev0._initial_topology if topology is None else topology, control=ev0.control, policy=ev0.policy,
                         splash=ev0.splash, floating_vertical=vert)
    return owner, bed, factory


def boundary_ns(tmp, backend, windowed, **kw):
    """A lineage whose head is a genuine COMPLETE Boundary (real storm -> genuine completion -> selected refresh): `lin, b, factory, original` (`original` = the genuine generation-0 PondTopology)."""
    owner, bed, factory = build(tmp, backend, windowed, **kw)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL)
    original = occ.event._initial_topology
    h = complete(occ.event, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    return SimpleNamespace(lin=lin, b=b, factory=factory, original=original, backend=backend, windowed=windowed)


def plan(ordinal, p=0.0, i=0.0, t=5.0):
    return rc.plan_daily_rain(p, i, t, run_digest=RUN, ordinal=ordinal)


def run_day(st, ordinal, *, output=None):
    n = st.ctx.n_cells
    r = br.day_action(st, plan=plan(ordinal), pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    assert r.outcome == "interevent"
    t = fd.request_floating_withdrawal(r.state, expected_parent=fd.expect_parent(r.state), request_m=np.full((st.ctx.n_layers, n), 1e-6), expected_sequence=0, available_bytes_fn=AVAIL)
    return fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), output=output, host_available_bytes_fn=lambda: BIG, disk_available_bytes_fn=lambda: BIG)


def closed_day_ns(tmp, backend, windowed, days=1, **kw):
    """The head is a genuine DailyHold with the closed state of day `days` (nothing handed back)."""
    ns = boundary_ns(tmp, backend, windowed, **kw)
    st = br.open_day(ns.lin, ns.b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    ns.origin = st
    for d in range(days):
        res = run_day(st, d)
        st = res.state
        if d != days - 1:
            ns.b = br.hand_back(ns.lin, st)
            st = br.open_day(ns.lin, ns.b, partition=PART, available_bytes_fn=AVAIL)
    ns.closed = st
    return ns


def save(ns, path, closed=None):
    return lcp.save_floating_lifecycle_checkpoint(path, ns.lin, ns.factory, IDENT, closed=closed, original_topology=ns.original, available_bytes_fn=AVAIL)


def load(ns, path, saved, partition=None, factory=None):
    return lcp.load_floating_lifecycle_checkpoint(path, ns.factory if factory is None else factory, IDENT, saved.manifest_sha256, original_topology=ns.original, partition=partition, available_bytes_fn=AVAIL)


def storm(lin, b, factory, end=30.0):
    sched = constant_rainfall(b.t_s, b.t_s + end, 10.0)
    occ, s = lc.begin_occurrence(lin, b, lambda vert, topology: factory(vert, topology, sched), available_bytes_fn=AVAIL)
    h = complete(occ.event, s)
    return lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
