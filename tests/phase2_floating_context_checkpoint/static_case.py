"""The caller's STATIC immutable case recipe (example): primitive parameters only -- terrain array, grid, forcing, sediment/splash parameters, control. It never touches the original deep bed, a NativeFloatingSoil, the Saxton-Rawls
provider or the original-case builder; the soil law comes from the verified held context. The MAPLE geometry/grain/context classes are built by the actual public MAPLE constructors."""
from __future__ import annotations

import numpy as np

from maple_syrup import floating_pond as fp
from maple_syrup.pond_event import PondEvent
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_physics import plot1_sediment_parameters
from maple_syrup.splash import splash_parameters

DX, NC, FRICTION = 0.5, 6, 5.0
NV, DZ = 300, 0.001
TERRAIN = {"base": 0.12, "drop": 0.004, "ny": 3, "nx": 4}
RAIN_MM_H, RAIN_END_S = 300.0, 1000.0
CASE = {"z": None, "closed": False, "policy": None, "max_dt_s": 0.2, "splash": True, "rain": (300.0, 1000.0)}  # recorded static options (a closed lake overrides them)


def terrain():
    if CASE["z"] is not None:
        return np.array(CASE["z"], dtype=np.float64)
    t = TERRAIN
    cols = np.arange(t["nx"], dtype=np.float64)[None, :]
    return t["base"] + t["drop"] * (t["nx"] - 1 - cols) + np.zeros((t["ny"], 1))


def bed_context():
    """The MAPLE bed CONTEXT from the actual public constructors (GeometrySpec, GrainClassSet, boundaries, specs) and recorded primitive values -- no probe state, no voxel/active arrays."""
    from maple.core.boundaries import AxisBoundary, BoundaryKind
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.geometry import GeometrySpec, validate_geometry
    from maple.core.parameters.grain_classes import (
        GrainClass,
        GrainClassSet,
        validate_grain_classes,
    )
    from maple.core.parameters.numerics import DEFAULT_MASS_RESOLUTION_KG
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.parameters.water_coupling import MAHLERAN_1_2_1_CLASS_DIAMETERS_M

    from maple_syrup.sediment_bed import BedContext

    t = TERRAIN
    gc = GrainClassSet(classes=tuple(GrainClass(class_id=f"mahleran_{i + 1}", diameter_m=float(d), particle_density_kg_m3=2650.0, is_aggregate=False) for i, d in enumerate(MAHLERAN_1_2_1_CLASS_DIAMETERS_M)))
    validate_grain_classes(gc)

    def boundary():
        return AxisBoundary(kind=BoundaryKind.PRESCRIBED, inflow_flux_kg_m_s={c.class_id: 0.0 for c in gc.classes})

    g = GeometrySpec(nx=t["nx"], ny=t["ny"], dx_m=DX, dy_m=DX, boundary_x=boundary(), boundary_y=boundary(), voxel_dz_m=DZ, bulk_density_kg_m3=1250.0, active_layer_thickness_m=0.002)
    validate_geometry(g, gc.ids())
    return BedContext(g, gc, float(DEFAULT_MASS_RESOLUTION_KG), TopographicCommitSpec(), AvalanchingSpec(enabled=False), np.ones((t["ny"], t["nx"], NC)))


def topology():
    from maple_syrup.pond_network import build_pond_topology

    z = terrain()
    ny, nx = z.shape
    full = np.pad(z, 1, mode="edge")
    outlets = [] if CASE["closed"] else [(r, nx - 1, 0, 1) for r in range(ny)]
    for r, c, dr, dc in outlets:
        full[r + 1 + dr, c + 1 + dc] = z[r, c] - 0.02
    return build_pond_topology(full, np.ones((ny, nx), bool), np.full((ny, nx), FRICTION), DX, outlets, allow_closed_components=CASE["closed"])


def factory(backend: str, windowed: bool):
    """`factory(held_context) -> PondEvent` (floating vertical from the context; no bed, no owner)."""
    def make(ctx):
        from maple_syrup.experimental_storm import ExperimentalControl
        from maple_syrup.pond_event import PondEventControl

        vert = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=lambda: 1 << 40)
        ctl = PondEventControl(backend=backend, time=ExperimentalControl(max_dt_s=CASE["max_dt_s"], min_dt_s=1.0 / 1024.0, max_retries=12, max_steps=50_000), bed_storage="windowed" if windowed else "dense",
                               deep_publication="step")
        return PondEvent(bed_context(), None, plot1_sediment_parameters(xp=np), constant_rainfall(0.0, CASE["rain"][1], CASE["rain"][0]), topology(), control=ctl, policy=CASE["policy"],
                         splash=splash_parameters(np.linspace(0.4, 3.0, NC), xp=np) if CASE["splash"] else None, floating_vertical=vert)
    return make
