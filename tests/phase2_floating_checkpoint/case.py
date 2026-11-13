"""Deterministic ORIGINAL-case factory of the R2a suite: native MAPLE bed (dense deep layered, or its published resident-window twin) -> cold native floating owner -> vertical -> PondEvent. Used identically by the saving process
and by the fresh restarting process, so the restart target's HELD template is the genuine original one."""
from __future__ import annotations

from pathlib import Path

import fpond_builders as fb
import numpy as np

from maple_syrup import floating_pond as fp
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import native_floating_soil as nfs
from maple_syrup.pond_event import PondEvent
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_physics import plot1_sediment_parameters
from maple_syrup.splash import splash_parameters

AVAIL = lambda: 1 << 40
IDENT = {"case": "r2a", "version": 1}
RAIN = 300.0 / 3.6e6


def build(workdir: Path, *, backend: str = "numpy", windowed: bool = False, preferential: bool = True, thin: bool = True):
    """`(event, state0, vertical)` of the original case. The windowed twin's deep store lives under `workdir/store`."""
    wh = fb._windowed_helpers()
    ph = fb.ph
    z = ph.plane(base=0.12, drop=0.004) if thin else ph.plane()
    dense, ctx = wh.dense_deep_bed(z, depth=0.004)
    topo = ph.make_topology(dense, ph.east_outlets(*z.shape), drop=0.02)
    if windowed:
        bed, c = wh.windowed_counterpart(dense, ctx, Path(workdir) / "store")
    else:
        bed, c = dense, ctx
    cfg = fb.config(backend=backend, bottom="closed" if thin else "open")
    owner = nfs.initialize_native_floating_soil(fb.native_bed(bed, c), cfg, initial_relative_saturation=0.97 if thin else 0.5, available_bytes_fn=AVAIL)
    pref = None
    if preferential:
        pref = fpi.prepare_floating_preferential(owner.column, a3=np.full(owner.column.n_cells, 2.0e-3), max_fraction_per_hour=0.5, max_intake_rate_m_s=1e-5, background_receiver_weights=[0.0, 1.0, 1.0],
                                                 available_bytes_fn=AVAIL)
    vert = fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, preferential=pref)
    be = type("B", (), {"hydraulics": backend, "xp": np})()
    ctl = wh.control_for(be, windowed=windowed, max_dt_s=0.2)
    ev = PondEvent(c, None, plot1_sediment_parameters(xp=np), constant_rainfall(0.0, 1000.0, 300.0), topo, control=ctl, splash=splash_parameters(np.linspace(0.4, 3.0, fb.NC), xp=np), floating_vertical=vert)
    return ev, ev.initial_state(bed, vert.initial_soil_water()), vert


def run(ev, st, n, rate=RAIN, dt=0.2):
    for _ in range(n):
        st = ev.step(st, rate, dt).state
    return st


def snapshot(ev, st, vert) -> dict:
    """Every physical array of a state as host bytes-comparable arrays (hydraulic, floating inventory/history/depth, bed, phase, ledgers, counters)."""
    import dataclasses

    from maple.core.backend import to_host

    out = {}

    def walk(prefix, value):
        if "window_cache" in prefix:  # a derived, rebuildable accelerator (not physical state; its presence differs between a live and a loaded bed)
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for f in dataclasses.fields(value):
                walk(f"{prefix}.{f.name}", getattr(value, f.name))
        elif isinstance(value, (tuple, list)):
            for i, x in enumerate(value):
                walk(f"{prefix}[{i}]", x)
        elif isinstance(value, dict):
            for k, x in value.items():
                walk(f"{prefix}[{k!r}]", x)
        elif isinstance(value, np.ndarray) or hasattr(value, "shape") and hasattr(value, "dtype"):
            out[prefix] = np.asarray(to_host(value) if not isinstance(value, np.ndarray) else value)
        elif isinstance(value, (int, float, str, bool, type(None))):
            out[prefix] = np.asarray(repr(value))

    for name in ("hydraulic", "bed", "sediment_velocity_m_s", "phase_fraction", "phase_position_m", "held_source_m", "datum", "water", "sediment", "counters", "quiet", "topology"):
        walk(name, getattr(st, name))
    p = vert.current_payload(st.hydraulic.soil_water_m)
    out["fl.W"], out["fl.J"], out["fl.I"], out["fl.D"] = (np.asarray(to_host(p.state.water_m)), np.asarray(to_host(p.history.wetting_m)), np.asarray(to_host(p.history.cumulative_intake_m)), np.asarray(to_host(p.depth_m)))  # counted MAPLE readback (identity for NumPy)
    return out
