"""Device-free fixture helpers of the CUDA refresh tests (host-importable, so the host suite can verify the logic Root's GPU definitions rely on).

MATCHED landscapes: the CUDA fixture `e5.device_event` ALWAYS builds the qualified deep layered bed (`wh.dense_deep_bed`: V = 300 voxels of 0.001 m, two strata); the earlier CPU twin used `fb.make` (V = 3 voxels of 0.1 m,
a homogeneous bed), which is a DIFFERENT soil (Root's diagnostic: full-column class inventory differs by 5.25 kg and the real stratified deposit moves one column's field capacity on the device bed only). The CPU twin is now
`fb.make_pair(..., backend="numpy", **same kwargs)[0|1]` (the same deep Dense bed / its published-window twin), and `assert_matched` proves the matched geometry/strata/initial inventory BEFORE any backend-parity claim.
The independent COLD ORACLE runs on the SAME completed child of whichever backend is under test (host copies of the held maps/current D/W, the full accepted compact read of every active column, the accepted changed-set
rule and the accepted CPU D1 reconcile on those host copies), so the strict changed set never depends on two different storms."""
from __future__ import annotations

import numpy as np
from fpond_builders import ph

from maple_syrup import native_floating_soil as nfs
from maple_syrup.pond_event import PondCompletionPolicy


def host(a):
    from maple.core.backend import to_host

    return np.asarray(to_host(a))


def lake_kwargs(*, mobile: bool = True) -> dict:
    """The thin closed lake of the EB2 suites; `mobile=True` leaves a real class-2 terminal deposit, `mobile=False` is a genuinely completed zero-mobile event (no bed write at all)."""
    z = ph.basin() - 0.08
    kw = {"surface": z, "depth": np.maximum(0.135 - z, 0.0), "closed": True, "bottom": "closed", "saturation": 0.999999, "splash": False, "rain_mm_h": 0.0, "end_s": 2.0, "max_dt_s": 1.0,
          "policy": PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)}
    if mobile:
        mob = np.zeros((3, 5, 6))
        mob[1, 2, 2] = 1.0e-1
        kw["mobile"] = mob
    return kw


def settle_to_completion(ev, st):
    """The EB2 recipe: evolve to quiet, retained settlement, evolve again, retained settlement; the final handoff must be COMPLETE."""
    q = ev.evolve(st, 60.0, stop_when_quiet=True)
    assert q.quiet_ready
    h = ev.settle_retaining_water(q.state)
    if not h.complete:
        h = ev.settle_retaining_water(ev.evolve(h.state, 120.0, stop_when_quiet=True).state)
    assert h.complete
    return h


def cpu_matched(tmp, windowed: bool, *, mobile: bool = True, backend: str = "numpy"):
    """`(event, completed handoff, vertical, initial state)` of the CPU twin built on the SAME deep layered terrain/forcing as the CUDA fixture (`fb.make_pair` -> Dense [0] or Windowed [1])."""
    import fpond_builders as fb

    tup = fb.make_pair(tmp, backend=backend, **lake_kwargs(mobile=mobile))[1 if windowed else 0]
    ev, st0, vert = tup[0], tup[1], tup[2]
    return ev, settle_to_completion(ev, st0), vert, st0


def bed_arrays(state, windowed: bool) -> dict:
    """The native bed arrays of a state as host arrays (Dense: voxel + active; Windowed: window/base/deep + active), for the matched-landscape proof."""
    b = state.bed
    out = {"active": host(b.active_layer.mass_kg)}
    if windowed:
        c = b.voxel_column
        out.update(window=host(c.window_kg), base=host(c.base), deep_by_class=host(c.deep_by_class_kg), deep_total=host(c.deep_total_kg), n_voxels=np.asarray(int(c.n_voxels)))
    else:
        out["voxel"] = host(b.voxel_column.mass_kg)
    return out


def assert_matched(a_ev, a_st0, a_vert, b_ev, b_st0, b_vert, windowed: bool) -> dict:
    """MATCHED landscapes BEFORE any backend-parity assertion: identical grid/voxel geometry and grain classes, byte-identical INITIAL bed arrays (strata), byte-identical held maps and D0, equal layer geometry. Returns a
    small record of what was compared (for logs)."""
    ga, gb = a_ev.context.geometry, b_ev.context.geometry
    assert (ga.ny, ga.nx, ga.dx_m, ga.voxel_dz_m, ga.bulk_density_kg_m3) == (gb.ny, gb.nx, gb.dx_m, gb.voxel_dz_m, gb.bulk_density_kg_m3), "different landscape geometry"
    assert [c.diameter_m for c in a_ev.context.grain_classes.classes] == [c.diameter_m for c in b_ev.context.grain_classes.classes]
    ba, bb = bed_arrays(a_st0, windowed), bed_arrays(b_st0, windowed)
    assert set(ba) == set(bb)
    for k in ba:
        assert ba[k].shape == bb[k].shape and ba[k].tobytes() == bb[k].tobytes(), f"initial bed array {k} differs (different strata/soil)"
    ta, tb = a_vert.held_template(), b_vert.held_template()
    for name in ("theta_field_capacity", "theta_saturation"):
        assert host(ta.column._maps[name]).tobytes() == host(tb.column._maps[name]).tobytes(), f"held {name} differs"
    assert host(ta.depth0_m).tobytes() == host(tb.depth0_m).tobytes() and a_vert.n_layers == b_vert.n_layers
    assert host(ta.column._maps["edges"]).tobytes() == host(tb.column._maps["edges"]).tobytes()
    return {"voxel_shape": None if windowed else list(ba["voxel"].shape), "windowed": windowed, "n_cells": int(a_vert.n_cells)}


def current_inventory(vert, h) -> dict:
    p = vert.current_payload(h.state.hydraulic.soil_water_m)
    return {"W": host(p.state.water_m), "J": host(p.history.wetting_m), "I": host(p.history.cumulative_intake_m), "D": host(p.depth_m)}


def assert_matched_inventory(a_vert, a_h, b_vert, b_h) -> None:
    """The CURRENT inventories of two completed events within the EXISTING numerical contracts (water 2e-12/1e-14, history 2e-12/1e-14, canonical depth BITWISE) before the refresh is compared."""
    a, b = current_inventory(a_vert, a_h), current_inventory(b_vert, b_h)
    np.testing.assert_allclose(a["W"], b["W"], rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(a["J"], b["J"], rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(a["I"], b["I"], rtol=2e-12, atol=1e-14)
    assert a["D"].tobytes() == b["D"].tobytes()


def cold_oracle(ev, h, vert, cfg_cpu=None, *, batch: int = 64):
    """FULL cold child property oracle on the SAME completed child of the backend under test, driven by the CURRENT payload (never the stale held D0): every active column read by the accepted compact reader; the changed set
    by the accepted comparison; W by the accepted CPU D1 reconcile on HOST copies of exactly those columns. `cfg_cpu` is the numpy-backend twin of the vertical's config (None: the vertical's own, for a CPU vertical).
    Returns `(changed, depth_changed, water, surf, exp_, fc, sat, depth)` (host)."""
    p = vert.current_payload(h.state.hydraulic.soil_water_m)
    held = vert.held_template()
    cfg = vert.config if cfg_cpu is None else cfg_cpu
    ctx = ev.context
    active = np.asarray(vert._active_host()).reshape(-1)
    ny, nx = ev.shape
    child = nfs.NativeBed(h.state.bed.voxel_column, h.state.bed.active_layer, ctx.geometry, ctx.grain_classes, ctx.mass_resolution_kg, active.reshape(ny, nx).copy())
    allc = np.flatnonzero(active).astype(np.int64)
    maps = {"theta_field_capacity": host(held.column._maps["theta_field_capacity"]), "theta_saturation": host(held.column._maps["theta_saturation"])}
    d_cur, w_cur = host(p.depth_m), host(p.state.water_m)
    fc, sat, depth = np.array(maps["theta_field_capacity"], copy=True), np.array(maps["theta_saturation"], copy=True), np.array(d_cur, copy=True)
    read_cfg = vert.config  # the ACTUAL resident bed/backend for the full compact read; host reconciliation below uses cfg_cpu and explicit host copies
    model = nfs._estimate(None, child, read_cfg, batch, 0)
    nfs._read_texture(child, read_cfg, allc, fc, sat, depth, model=model, batch_cells=int(model["batch_cells"]), update_properties=True, ks=None)
    changed, depth_changed = nfs._changed_columns_parts(cfg, maps, d_cur, allc, fc, sat, depth, batch)
    water, surf, exp_, _acc = nfs._reconcile_changed(cfg, maps, d_cur, w_cur, fc, sat, depth, changed, nfs._area(child), batch, float(p.state.t_s))
    return changed, depth_changed, water, surf, exp_, fc, sat, depth
