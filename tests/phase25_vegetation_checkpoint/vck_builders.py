"""Builders of the vegetation physical-checkpoint suite: REAL native lifecycle states (genuine storm -> completion -> accepted Boundary; genuine bridge days -> closed state), the ACTUAL LEAF P1 projection and the S0 binding applied by the
owning `apply_vegetation_generation`. The static event recipe takes the BOUND splash cover (`factory(vertical, topology, cover)`). No fake capability, producer, owner, projection or generation."""
from __future__ import annotations

from types import SimpleNamespace

import fpond_builders as fb
import numpy as np
import veg_lc_builders as vb
from bridge_builders import AVAIL, RUN, complete
from lcp_builders import IDENT, PART, plan

from maple_syrup import floating_daily as fd
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_lifecycle_checkpoint_vegetation as lcv
from maple_syrup import vegetation_lifecycle as vl
from maple_syrup.pond_event import PondCompletionPolicy, PondEvent
from maple_syrup.rainfall import constant_rainfall

BIG = 10**12
__all__ = ["AVAIL", "BIG", "IDENT", "PART", "RUN", "boundary_vns", "closed_day_vns", "load", "run_day", "save", "storm"]


def build(tmp, backend, windowed, *, deep_dense=False, splash=True):
    """`(owner, bed, factory)`: the same plane as the bridge builder, with a cover-aware static event recipe. `deep_dense=True` selects the DENSE side of the deep layered pair (the exact context / voxel_dz / source of the Windowed
    counterpart: the matched fresh-process static recipe)."""
    pol = PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-4, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True)
    ph = fb.ph
    kw = {"surface": ph.plane(base=0.12, drop=0.004), "depth": 0.0, "bottom": "open", "saturation": 0.5, "backend": backend, "splash": splash, "rain_mm_h": 10.0, "end_s": 60.0, "max_dt_s": 0.2, "policy": pol}
    if windowed or deep_dense:
        ev0, _st0, _v, owner, bed, _c = fb.make_pair(tmp, **kw)[1 if windowed else 0]
    else:
        ev0, _st0, bed, _c, _v, owner = fb.make(**kw)

    def factory(vert, topology=None, cover=None, schedule=None):
        return PondEvent(ev0.context, None, ev0.sediment, ev0.schedule if schedule is None else schedule, ev0._initial_topology if topology is None else topology, control=ev0.control, policy=ev0.policy,
                         splash=ev0.splash, vegetation_cover_fraction=cover, floating_vertical=vert)

    return owner, bed, factory


def boundary_vns(tmp, leaf, backend, windowed, *, apply=True, deep_dense=False, splash=True):
    """A lineage whose head is a genuine COMPLETE Boundary carrying ONE applied vegetation generation (real storm -> genuine completion -> refresh -> actual LEAF projection -> S0 binding -> owning apply)."""
    owner, bed, factory = build(tmp, backend, windowed, deep_dense=deep_dense, splash=splash)
    lin, occ, st = lc.open_floating_event(owner, bed, lambda vert, topology=None: factory(vert, topology), available_bytes_fn=AVAIL)
    original = occ.event._initial_topology
    h = complete(occ.event, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    ns = SimpleNamespace(lin=lin, b=b, factory=factory, original=original, backend=backend, windowed=windowed, leaf=leaf)
    if apply:
        s = vb.stack(leaf, b)
        bd = vb.binding(leaf, b, s)
        ns.stack, ns.binding = s, bd
        ns.b = vl.apply_vegetation_generation(lin, b, bd, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    return ns


def closed_day_vns(tmp, leaf, backend, windowed, days=1, **kw):
    ns = boundary_vns(tmp, leaf, backend, windowed, **kw)
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


def run_day(st, ordinal, *, output=None):
    """One genuine interevent bridge day with the BOUND E-shading cover (`cover=None`), one transpiration request, audited close."""
    n = st.ctx.n_cells
    r = br.day_action(st, plan=plan(ordinal), pet_terrain_m_s=np.full(n, 3e-8), cover=None, available_bytes_fn=AVAIL)
    assert r.outcome == "interevent"
    t = fd.request_floating_withdrawal(r.state, expected_parent=fd.expect_parent(r.state), request_m=np.full((st.ctx.n_layers, n), 1e-6), expected_sequence=0, available_bytes_fn=AVAIL)
    return fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), output=output, host_available_bytes_fn=lambda: BIG, disk_available_bytes_fn=lambda: BIG)


def pins(ns):
    return {"leaf_package_sha256": vb.leaf_pin(), "external_helpers": vb.helper_pin()}


def save(ns, path, closed=None, **kw):
    args = {"available_bytes_fn": AVAIL, "memory_budget_bytes": BIG}
    args.update(kw)
    return lcv.save_floating_lifecycle_vegetation_checkpoint(path, ns.lin, ns.factory, IDENT, closed=closed, original_topology=ns.original, **args)


def load(ns, path, saved, partition=None, factory=None, **kw):
    args = {"available_bytes_fn": AVAIL, "memory_budget_bytes": BIG, **pins(ns)}
    args.update(kw)
    return lcv.load_floating_lifecycle_vegetation_checkpoint(path, ns.factory if factory is None else factory, IDENT, saved.manifest_sha256, original_topology=ns.original, partition=partition, **args)


def storm(lin, b, factory, end=30.0):
    sched = constant_rainfall(b.t_s, b.t_s + end, 10.0)
    occ, s = lc.begin_occurrence(lin, b, vl.event_factory_for(b, lambda vert, topology, cover: factory(vert, topology, cover, sched)), available_bytes_fn=AVAIL)
    h = complete(occ.event, s)
    return lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
