"""Builders for the E4b floating-Pond suite (author-written). Real MAPLE beds/topologies from the accepted pond-event helpers (loaded by path, unchanged)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "pond_event" / "pond_helpers.py"
_spec = importlib.util.spec_from_file_location("e4b_pond_helpers", _p)
ph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ph)

from maple_syrup import native_floating_soil as nfs

LAYERS = (0.01, 0.05, 0.10)  # H = 0.16 m
NC = ph.NC
TEXTURE = np.array([[0.05, 0.45, 0.50], [0.10, 0.55, 0.35], [0.15, 0.65, 0.20], [0.30, 0.50, 0.20], [0.55, 0.30, 0.15], [0.25, 0.50, 0.25]])
AVAIL = lambda: 10**12


def config(*, backend="numpy", bottom="open", layers=LAYERS, **kw):
    L = len(layers)
    args = {"layer_thickness_m": layers, "grain_texture": TEXTURE, "organic_matter_percent": np.full(L, 3.0), "bulk_density_kg_m3": 1250.0, "latent_theta_field_capacity": np.full(L, 0.25),
            "latent_theta_saturation": np.full(L, 0.42), "theta_residual": 0.05, "theta_evaporation_stress": 0.2, "diffusivity_m2_s": np.full(L - 1, 1e-8), "bottom_drainage": bottom,
            "matrix_conductivity_m_s": 2e-6, "matrix_suction_m": 0.1, "matrix_vegetation_factor": 0.7, "initial_wetting_m": 0.01, "initial_cumulative_intake_m": 0.02, "overrides": None, "backend": backend}
    args.update(kw)
    return nfs.native_floating_config(**args)


def native_bed(bed, ctx, mask=None):
    ny, nx = int(ctx.geometry.ny), int(ctx.geometry.nx)
    return nfs.NativeBed(bed.voxel_column, bed.active_layer, ctx.geometry, ctx.grain_classes, ctx.mass_resolution_kg, np.ones((ny, nx), bool) if mask is None else mask)


def make(*, surface=None, depth=0.0, backend="numpy", bottom="open", saturation=0.5, rain_mm_h=100.0, end_s=100.0, splash=False, preferential=False, max_dt_s=0.2, closed=False, mobile=None, policy=None, mask=None, **cfg_kw):
    """`(event, state, bed, ctx, vertical, owner)` on a real MAPLE bed (3 x 4 plane falling east unless `surface`)."""
    from maple_syrup import floating_pond as fp
    from maple_syrup import floating_preferential_infiltration as fpi
    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.pond_event import PondEvent, PondEventControl
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters
    from maple_syrup.splash import splash_parameters

    z = ph.plane() if surface is None else surface
    bed, ctx = ph.make_bed(z, depth=depth, mobile=mobile)
    topo = ph.make_topology(bed, [], allow_closed=True) if closed else ph.make_topology(bed, ph.east_outlets(*z.shape), drop=0.02, active=mask)
    cfg = config(backend=backend, bottom=bottom, **cfg_kw)
    owner = nfs.initialize_native_floating_soil(native_bed(bed, ctx, mask), cfg, initial_relative_saturation=saturation, available_bytes_fn=AVAIL)
    pref = None
    if preferential:
        pref = fpi.prepare_floating_preferential(owner.column, a3=np.full(owner.column.n_cells, 2.0e-3), max_fraction_per_hour=0.5, max_intake_rate_m_s=1e-5, background_receiver_weights=[0.0, 1.0, 1.0],
                                                 available_bytes_fn=AVAIL)
    vert = fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, preferential=pref)
    ctl = PondEventControl(backend=backend, time=ExperimentalControl(max_dt_s=max_dt_s, min_dt_s=1.0 / 1024.0, max_retries=12, max_steps=50_000))
    sp = splash_parameters(np.linspace(0.4, 3.0, NC), xp=np) if splash else None
    ev = PondEvent(ctx, None, plot1_sediment_parameters(xp=np), constant_rainfall(0.0, end_s, rain_mm_h), topo, control=ctl, policy=policy, splash=sp, floating_vertical=vert)
    state = ev.initial_state(bed, vert.initial_soil_water())
    return ev, state, bed, ctx, vert, owner


def _windowed_helpers():
    p = Path(__file__).resolve().parents[1] / "pond_windowed" / "windowed_helpers.py"
    import sys

    sys.path[:0] = [str(p.parent), str(p.parent.parent / "pond_event")]
    spec = importlib.util.spec_from_file_location("e4c_windowed_helpers", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_pair(tmp_path, *, surface=None, depth=0.003, backend="numpy", bottom="open", saturation=0.5, rain_mm_h=100.0, end_s=100.0, splash=True, mobile=None, closed=False, policy=None,
              max_dt_s=0.2, deep_publication="step", **cfg_kw):
    """`[(event, state, vertical, owner, bed, ctx)] for (dense, windowed)` on the SAME deep layered terrain/forcing; each side's floating owner is initialised from ITS OWN actual native bed."""
    from types import SimpleNamespace

    from maple_syrup import floating_pond as fp
    from maple_syrup.pond_event import PondEvent
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters
    from maple_syrup.splash import splash_parameters

    wh = _windowed_helpers()
    z = ph.plane(base=0.12, drop=0.004) if surface is None else surface
    dense, ctx = wh.dense_deep_bed(z, depth=depth, mobile=mobile)
    wbed, wctx = wh.windowed_counterpart(dense, ctx, tmp_path / "store")
    topo = ph.make_topology(dense, [], allow_closed=True) if closed else ph.make_topology(dense, ph.east_outlets(*z.shape), drop=0.02)
    be = SimpleNamespace(hydraulics=backend, xp=np)
    out = []
    for bed, c, windowed in ((dense, ctx, False), (wbed, wctx, True)):
        cfg = config(backend=backend, bottom=bottom, **cfg_kw)
        owner = nfs.initialize_native_floating_soil(native_bed(bed, c), cfg, initial_relative_saturation=saturation, available_bytes_fn=AVAIL)
        vert = fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL)
        ctl = wh.control_for(be, windowed=windowed, max_dt_s=max_dt_s, deep_publication=deep_publication)
        sp = splash_parameters(np.linspace(0.4, 3.0, NC), xp=np) if splash else None
        ev = PondEvent(c, None, plot1_sediment_parameters(xp=np), constant_rainfall(0.0, end_s, rain_mm_h), topo, control=ctl, policy=policy, splash=sp, floating_vertical=vert)
        out.append((ev, ev.initial_state(bed, vert.initial_soil_water()), vert, owner, bed, c))
    return out
