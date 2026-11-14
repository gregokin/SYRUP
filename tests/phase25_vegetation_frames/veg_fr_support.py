"""Helpers of the vegetation-frame suite (unique module name): a REAL accepted `PlantSession` generation sequence applied to a REAL lifecycle boundary, and a real daily lineage driven day by day through the bridge. No fake states."""
from __future__ import annotations

import numpy as np
import veg_lc_builders as vb
from bridge_builders import AVAIL, dry_boundary

from maple_syrup import floating_daily as fd
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup import vegetation_lifecycle as vl

PART = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
RUN = "ab" * 32
BIG = vb.BIG


def session_pair(leaf, boundary):
    """`(sess0, sess1, cat, surf, mask, ctl, nfx)`: a genuine accepted session and its NEXT published generation (a dry published day), planted on the boundary's actual grid / surface."""
    nfx = vb.leaf_fixture(leaf)
    surf = vb.surface_of(boundary)
    dx = float(boundary.routing[0].dx_m)
    cat = vb.session_catalog(leaf, nfx, boundary.root.column.n_layers)
    sess0 = nfx.start_session(vb.session_state(leaf, cat, surf, dx), nfx.make_config(cat), nfx.anchor(), bound=True)
    sess1 = nfx.run_day(sess0, nfx.anchor(), 0, t=False, g=False)[0]
    return sess0, sess1, cat, surf, np.ones(surf.shape, dtype=bool), vb.controls(boundary), nfx


def apply_session(lin, boundary, leaf, sess, surf, mask, ctl):
    bd, _p = vb.session_binding(leaf, boundary, sess, surf, mask, ctl)
    return vl.apply_vegetation_generation(lin, boundary, bd, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)


def day(state, *, out=None, bound=True):
    """ONE genuine daily update from a closed bridge state: dry no-rain day, one T request (zero), the final close with the optional output. Returns `(closed, notice)`."""
    n, L = state.ctx.n_cells, state.ctx.n_layers
    pl = rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=int(state.cursor.ordinal))
    cover = None if bound else np.full(n, 0.3)
    res = br.day_action(state, plan=pl, pet_terrain_m_s=np.full(n, 3e-8), cover=cover, available_bytes_fn=AVAIL)
    assert res.outcome == "interevent"
    t = fd.request_floating_withdrawal(res.state, expected_parent=fd.expect_parent(res.state), request_m=np.zeros((L, n)), expected_sequence=0, available_bytes_fn=AVAIL)
    cr = fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), output=out, available_bytes_fn=AVAIL)
    return cr.state, cr.notice


def theta_independent(water, depth, edges):
    """Independent reconstruction theta = W / h with h = max(0, min(D, edge_{l+1}) - edge_l) (0 where h == 0)."""
    L, N = water.shape
    th = np.zeros((L, N))
    for l in range(L):
        h = np.maximum(np.minimum(depth, edges[l + 1]) - edges[l], 0.0)
        np.divide(water[l], h, out=th[l], where=h > 0.0)
    return th


__all__ = ["AVAIL", "BIG", "PART", "RUN", "apply_session", "br", "day", "dry_boundary", "fd", "session_pair", "theta_independent", "vl"]
