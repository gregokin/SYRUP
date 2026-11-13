"""Native-case builders for the Phase 1.5 S2 daily lifecycle controls (CPU). Everything is built through the EXISTING component constructors on the
ACTUAL native Dense MAPLE bed (`storm_cases.native_dense_bed` uses MAPLE's own constructors); adapted from the Phase 1 boundary-entry builders."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from storm_cases import (BIG, as_xp, column_of, history_of, layers, matrix_of, native_dense_bed, native_topology, pref_of, router_of, soil_state,
                         topology_from)

from maple_syrup import layered_event as ev
from maple_syrup import soil_evaporation as se
from maple_syrup import water_lifecycle as wl
from maple_syrup.pond_hydraulics import metadata_routing_graph
from maple_syrup.rainfall import constant_rainfall, rainfall_field

PARTITION = se.EvaporationPartition(canopy_intercept=1.0, canopy_slope=0.5, transpiration_slope=1.0, transpiration_offset=0.0, soil_factor=1.0)
DAY = 86400.0


class Box:
    def __init__(self, value=BIG):
        self.value = value

    def __call__(self):
        return self.value


def host(a):
    return np.asarray(a.get()) if hasattr(a, "get") else np.asarray(a)


def build(interior, outlets=(), *, backend="numpy", dense=True, lay=None, cover=0.5, theta=(0.125, 0.125), depth=0.0, active=None, wetting=0.001,
          cumulative=0.002, native_reserve=1 << 22, extra_reserve=1 << 20, n_classes=3, pref=None, matrix=None, available=BIG, t_s=0.0):
    """Lifecycle + initial closed boundary on the ACTUAL native Dense bed (or bedless)."""
    interior = np.asarray(interior, dtype=np.float64)
    bed = ctx = None
    if dense:
        bed, ctx = native_dense_bed(interior, depth=depth, n_classes=n_classes)
        topo = native_topology(bed, ctx, outlets, active=active)
    else:
        topo = topology_from(interior, outlets, active=active)
    graph = metadata_routing_graph(topo, np)
    router = router_of(topo, graph, backend=backend)
    lay = layers() if lay is None else lay
    col = column_of(topo, lay, backend=backend, xp=np, available=available)
    p = pref_of(col) if pref is None else pref(col)
    m = matrix_of(col) if matrix is None else matrix(col)
    evap = se.prepare_evaporation(col, PARTITION)
    n = col.n_cells
    cov = np.full(n, float(cover))
    lop = wl.prepare_lifecycle(col, p, m, router, graph, topo, evap, cov, bed=bed, context=ctx, native_reserve_bytes=native_reserve if dense else None,
                               extra_reserve_bytes=extra_reserve)
    soil = soil_state(col, theta, t_s)
    hist = history_of(m, wetting, cumulative)
    ny, nx = topo.shape
    if dense:
        s0 = wl.start_lifecycle(lop, bed=bed, soil=soil, history=hist, t_s=t_s)
    else:
        s0 = wl.start_lifecycle(lop, surface_m=np.full((ny, nx), float(depth)), soil=soil, history=hist, t_s=t_s)
    return SimpleNamespace(lop=lop, s0=s0, col=col, topo=topo, n=n, L=col.n_layers, shape=topo.shape, bed=bed, ctx=ctx, dense=dense, pref=p, matrix=m)


def pet(c, rate):
    return np.full(c.n, float(rate))


def field_of(c):
    return rainfall_field(*c.shape, active_mask=np.array(c.topo.active), xp=np)


def raw(s):
    a = wl._arrays_of(s)
    return dict(t=float(a.t_s), surface=host(a.surface_m).copy(), qx=host(a.qx).copy(), qy=host(a.qy).copy(), soil=host(a.soil.water_m).copy(),
                j=host(a.history.wetting_m).copy(), intake=host(a.history.cumulative_intake_m).copy())


def fingerprint(s):
    return (s.digest, s.revision, s.t_s, s.request_sequence, s.reset_count, s.episode, {k: v.tobytes() if isinstance(v, np.ndarray) else v for k, v in raw(s).items()})
