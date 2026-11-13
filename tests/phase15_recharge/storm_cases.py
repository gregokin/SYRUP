"""Builders for the layered storm step tests (not a test module).

Everything is built through the EXISTING component constructors: `pond_network.build_pond_topology` (held bed,
geometry, real outlets / closed basins), `pond_hydraulics.metadata_routing_graph` (the metadata graph of the local-
inertial router), `surface_routing.build_surface_router`, `subsurface.prepare_column`, `preferential_infiltration.
prepare_preferential`, `matrix_infiltration.prepare_matrix`, and an ACTUAL dense native MAPLE bed through MAPLE's own
constructors (`native_dense_bed`). Dyadic numbers where exactness is asserted (the repository convention).

Default column (two layers, dyadic): edges (0, 0.25, 1.0); theta_res 0.0625, fc 0.25, sat 0.5 -> top layer
residual 0.015625 m, field capacity 0.0625 m, saturation 0.125 m; bottom residual 0.046875, fc 0.1875, sat 0.375.
Default matrix: K = 2^-17 m/s (7.63e-6, ~27 mm/h), psi = 0.0625 m, g = 0.5. Default preferential: a3 = 0.5, maxFL2 = 0.5,
no rate cap, background to the bottom layer.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

from maple_syrup import layered_storm as lst
from maple_syrup import matrix_infiltration as mi
from maple_syrup import preferential_infiltration as pi
from maple_syrup import subsurface as ss
from maple_syrup.pond_hydraulics import metadata_routing_graph
from maple_syrup.pond_network import OutletFace, build_pond_topology
from maple_syrup.surface_routing import build_surface_router

DX = 0.5
AREA = DX * DX
BIG = 1 << 42
EDGES = (0.0, 0.25, 1.0)
RES, FC, SAT, STRESS = 0.0625, 0.25, 0.5, 0.25
K = 2.0 ** -17
PSI = 0.0625
G = 0.5
FRICTION = 1.0


# --- host helpers ------------------------------------------------------------------------------------------------------
def host(a) -> np.ndarray:
    return np.asarray(a.get()) if hasattr(a, "get") else np.asarray(a)


def digest(*arrays) -> list[str]:
    return [hashlib.sha256(np.ascontiguousarray(host(a)).tobytes()).hexdigest() for a in arrays]


def state_digests(state: lst.LayeredWaterState) -> list[str]:
    return digest(state.surface_m, state.qx_m2_s, state.qy_m2_s, state.soil.water_m, state.history.wetting_m,
                  state.history.cumulative_intake_m)


# --- terrain / topology ------------------------------------------------------------------------------------------------
def topology_from(interior, outlets=(), *, drop=0.02, active=None, friction=FRICTION, allow_closed=None, ring=None, dx=DX):
    """Topology of an interior bed `(ny, nx)`. Each outlet `(row, col, d_row, d_col)` gets a ring cell `drop` m below its
    cell; every other ring cell is higher than the whole bed (closed). `active` defaults to all cells."""
    z = np.asarray(interior, dtype=np.float64)
    ny, nx = z.shape
    full = np.full((ny + 2, nx + 2), float(z.max()) + 10.0)
    full[1:-1, 1:-1] = z
    faces = [o if isinstance(o, OutletFace) else OutletFace(*o) for o in outlets]
    for o in faces:
        full[o.row + o.d_row + 1, o.col + o.d_col + 1] = z[o.row, o.col] - drop
    for (r, c), value in (ring or {}).items():
        full[r, c] = value
    act = np.ones((ny, nx), dtype=bool) if active is None else np.array(active, dtype=bool)
    if allow_closed is None:
        allow_closed = not faces
    return build_pond_topology(full, act, np.full((ny, nx), float(friction)), dx, faces, allow_closed_components=allow_closed)


def plane(ny: int, nx: int, slope: float = 0.02, base: float = 0.2) -> np.ndarray:
    """Plane descending eastward (column index increasing) by `slope * dx` per cell."""
    cols = np.arange(nx, dtype=np.float64)[None, :]
    return np.repeat(base - slope * DX * cols, ny, axis=0)


def bowl(n: int = 3, depth: float = 0.03, base: float = 0.2) -> np.ndarray:
    """Square bowl: centre `depth` below the rim, rim flat (flats + a depression)."""
    z = np.full((n, n), base)
    z[n // 2, n // 2] = base - depth
    return z


def east_outlets(ny: int, nx: int):
    return [(r, nx - 1, 0, 1) for r in range(ny)]


# --- soil / stages ---------------------------------------------------------------------------------------------------
def layers(*, edges=EDGES, res=RES, fc=FC, sat=SAT, stress=STRESS, diffusivity=0.0, gravity=True, bottom="open", factor=1.0):
    return ss.soil_layers(edges, theta_residual=res, theta_field_capacity=fc, theta_saturation=sat,
                          diffusivity_m2_s=diffusivity, theta_evaporation_stress=stress, evaporation_factor=factor,
                          gravity_drainage=gravity, bottom_drainage=bottom)


def column_of(topology, lay, *, backend="numpy", xp=np, available=BIG, reserve=0):
    mask = np.ascontiguousarray(np.asarray(topology.active, dtype=bool).reshape(-1))
    if xp is not np:
        mask = xp.asarray(mask)
    return ss.prepare_column(lay, mask, topology.dx_m * topology.dx_m, backend=backend, reserve_bytes=reserve,
                             available_bytes_fn=lambda: available)


def pref_of(col, *, a3=0.5, max_fraction=0.5, rate=np.inf, background=None, species_weights=(), contributions=None):
    bg = [0.0] * (col.n_layers - 1) + [1.0] if background is None else background
    return pi.prepare_preferential(col, a3=a3, max_fraction_per_hour=max_fraction, max_intake_rate_m_s=rate,
                                   background_receiver_weights=bg, species_receiver_weights=species_weights,
                                   species_contributions=contributions)


def matrix_of(col, *, k=K, psi=PSI, g=G):
    return mi.prepare_matrix(col, conductivity_m_s=k, suction_m=psi, vegetation_factor=g)


def router_of(topology, graph, *, backend="numpy", control=None, available=BIG, reserve=0):
    return build_surface_router("local_inertial", graph, backend=backend, geometry=topology.geometry, control=control,
                                reserve_bytes=reserve, available_bytes_fn=lambda: available)


def soil_state(col, theta_by_layer, t_s=0.0):
    return ss.uniform_state(col, list(theta_by_layer), t_s)


def history_of(mat, wetting=0.0, cumulative=0.0):
    return mi.wetting_history(mat, wetting_m=wetting, cumulative_intake_m=cumulative)


def as_xp(xp, a):
    return a if xp is np else xp.asarray(a)


# --- native dense bed ------------------------------------------------------------------------------------------------
def native_grain_classes(n_classes: int):
    """A genuine validated MAPLE `GrainClassSet` of `n_classes` distinct diameters (wide-class controls)."""
    from maple.core.parameters.grain_classes import GrainClass, GrainClassSet, validate_grain_classes

    diameters = 2e-5 * (2.0 ** (np.arange(n_classes) * (6.0 / max(n_classes - 1, 1))))  # 20 um .. 1.28 mm
    classes = GrainClassSet(classes=tuple(
        GrainClass(class_id=f"class_{k:04d}", diameter_m=float(d), particle_density_kg_m3=2650.0, is_aggregate=False)
        for k, d in enumerate(diameters)))
    validate_grain_classes(classes)
    return classes


def native_dense_bed(elevation, *, depth=0.0, n_voxels=40, voxel_dz=0.01, n_classes=6):
    """ACTUAL dense MAPLE `(BedState, BedContext)` built by MAPLE's own constructors on `elevation (ny, nx)` (must stay
    below `n_voxels * voxel_dz`): `n_classes` genuine grain classes, a uniform mixture deposited by
    `deposit_surface_mixture_batch`, the active layer by `initialize_active_layer_from_voxels`, committed topography,
    availability, an empty ledger and a water state of uniform `depth` with NO mobile sediment."""
    from maple.core.boundaries import AxisBoundary, BoundaryKind
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.geometry import GeometrySpec, validate_geometry
    from maple.core.parameters.numerics import DEFAULT_MASS_RESOLUTION_KG
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.types.sediment_availability import sediment_availability_state_from_active_layer
    from maple.core.types.sediment_ledger import process_index, zeros_sediment_ledger_state
    from maple.core.types.voxel import zeros_voxel_column_state
    from maple.core.types.water import water_state_from_depth
    from maple.surface.active_layer import initialize_active_layer_from_voxels
    from maple.surface.topographic_commit import initial_committed_topography_state_from_physical_state
    from maple.surface.voxels import deposit_surface_mixture_batch

    from maple_syrup.sediment_bed import BedContext, BedState

    elevation = np.asarray(elevation, dtype=np.float64)
    ny, nx = elevation.shape
    gc = native_grain_classes(n_classes)
    nc = len(gc.classes)

    def boundary():
        return AxisBoundary(kind=BoundaryKind.PRESCRIBED, inflow_flux_kg_m_s={c.class_id: 0.0 for c in gc.classes})

    g = GeometrySpec(nx=nx, ny=ny, dx_m=DX, dy_m=DX, boundary_x=boundary(), boundary_y=boundary(), voxel_dz_m=voxel_dz,
                     bulk_density_kg_m3=1250.0, active_layer_thickness_m=0.002)
    validate_geometry(g, gc.ids())
    mres = float(DEFAULT_MASS_RESOLUTION_KG)
    if float(elevation.max()) >= n_voxels * voxel_dz:
        raise ValueError("surface elevation exceeds the allocated voxel column")
    mix = np.full(nc, 1.0 / nc)
    column = zeros_voxel_column_state(ny, nx, n_voxels, nc)
    per_m = g.bulk_density_kg_m3 * g.dx_m * g.dy_m
    column, _ = deposit_surface_mixture_batch(column, (elevation * per_m)[:, :, None] * mix[None, None, :], g, mres)
    column, layer, _ = initialize_active_layer_from_voxels(column, g, mres)
    water = water_state_from_depth(np.full((ny, nx), float(depth)), nc)
    fraction = np.ones_like(layer.mass_kg)
    committed = initial_committed_topography_state_from_physical_state(column, layer, g, mres)
    availability = sediment_availability_state_from_active_layer(layer.mass_kg, fraction)
    ledger = zeros_sediment_ledger_state(ny, nx, nc, present_processes=(process_index("water_erosion_deposition"),))
    state = BedState(column, layer, water, ledger, committed, availability)
    ctx = BedContext(g, gc, mres, TopographicCommitSpec(), AvalanchingSpec(enabled=False), fraction)
    return state, ctx


def native_topology(bed, ctx, outlets=(), *, drop=0.02, active=None, allow_closed=None):
    """The held topology built from the bed's CURRENT routing surface (the fixed-bed binding the operator requires)."""
    from maple_syrup.sediment_bed import current_routing_surface_m

    surface = host(current_routing_surface_m(bed, ctx))
    return topology_from(surface, outlets, drop=drop, active=active, allow_closed=allow_closed)


# --- composed case -------------------------------------------------------------------------------------------------------
def storm_of(interior, outlets=(), *, backend="numpy", xp=np, lay=None, active=None, allow_closed=None, drop=0.02,
             control=None, pref=None, matrix=None, available=BIG, reserve=0, extra_reserve=0, native=False, depth=0.0,
             router_available=None, n_classes=6, native_reserve=0):
    """Everything bound: topology, graph, router, column, preferential, matrix, operator and (optionally) the ACTUAL
    native bed whose water is the surface. `pref` / `matrix` are callables `col -> operator` to vary the laws."""
    bed = ctx = None
    if native:
        bed, ctx = native_dense_bed(interior, depth=depth, n_classes=n_classes)
        if xp is not np:
            from maple.core.backend import to_device, to_device_tree
            from dataclasses import replace

            bed = to_device_tree(bed, xp)
            ctx = replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, xp))
        topo = native_topology(bed, ctx, outlets, drop=drop, active=active, allow_closed=allow_closed)
    else:
        topo = topology_from(interior, outlets, drop=drop, active=active, allow_closed=allow_closed)
    graph = metadata_routing_graph(topo, xp)
    router = router_of(topo, graph, backend=backend, control=control,
                       available=available if router_available is None else router_available, reserve=reserve)
    lay = layers() if lay is None else lay
    col = column_of(topo, lay, backend=backend, xp=xp, available=available, reserve=reserve)
    p = pref_of(col) if pref is None else pref(col)
    m = matrix_of(col) if matrix is None else matrix(col)
    op = lst.prepare_layered_storm(col, p, m, router, graph, topo, bed=bed, context=ctx, extra_reserve_bytes=extra_reserve,
                                   native_reserve_bytes=native_reserve if native else None)
    return SimpleNamespace(topology=topo, graph=graph, router=router, column=col, pref=p, matrix=m, op=op, bed=bed, ctx=ctx,
                           xp=xp, backend=backend, shape=topo.shape, lay=lay)


def initial(cs, *, theta=(0.125, 0.125), depth=0.0, t_s=0.0, wetting=0.0, cumulative=0.0, qx=None, qy=None):
    """Accepted initial aggregate: uniform soil theta per layer, uniform surface depth (or an (ny, nx) array)."""
    soil = soil_state(cs.column, theta, t_s)
    hist = history_of(cs.matrix, wetting, cumulative)
    if cs.bed is not None:
        return lst.initial_layered_state(cs.op, soil=soil, history=hist, t_s=t_s, qx_m2_s=qx, qy_m2_s=qy)
    ny, nx = cs.shape
    surface = np.broadcast_to(np.asarray(depth, dtype=np.float64), (ny, nx)).copy()
    return lst.initial_layered_state(cs.op, soil=soil, history=hist, t_s=t_s, surface_m=as_xp(cs.xp, surface),
                                     qx_m2_s=qx, qy_m2_s=qy)


def rain_of(cs, mm_per_h: float):
    ny, nx = cs.shape
    rate = np.where(np.asarray(cs.topology.active), mm_per_h / 3.6e6, 0.0).astype(np.float64)
    return as_xp(cs.xp, np.ascontiguousarray(rate.reshape(ny, nx)))


def manual_stages(cs, accepted, rain, dt):
    """The documented stage order, called by hand on the SAME components (an ordering/binding control for the
    composed step, not an independent physics oracle)."""
    from maple_syrup.surface_routing import SurfaceState

    xp = cs.xp
    ny, nx = cs.shape
    n = ny * nx
    h0 = accepted.surface_m.reshape(n)
    r = rain.reshape(n)
    pool = r * dt
    xp.add(pool, h0, out=pool)
    p = pi.preferential_step(cs.pref, pool, accepted.soil, dt)
    m = mi.matrix_step(cs.matrix, p.surface_water_m, h0, r, p.state, accepted.history, dt)
    zeros_l = xp.zeros((cs.column.n_layers, n))
    c = ss.step(cs.column, m.state, dt, added_water_m=zeros_l, potential_evaporation_m_s=xp.zeros(n))
    post = m.surface_pool_m + c.surface_return_m
    s = cs.router.step(SurfaceState(accepted.t_s, post.reshape(ny, nx), accepted.qx_m2_s, accepted.qy_m2_s), dt)
    return SimpleNamespace(p=p, m=m, c=c, s=s, pool=pool, post=post)


def global_terms(cand: lst.LayeredStepCandidate) -> dict[str, float]:
    r = cand.receipt
    return {"rain": r.rain_m3, "surface0": r.surface_before_m3, "surface1": r.surface_after_m3, "soil0": r.soil_before_m3,
            "soil1": r.soil_after_m3, "export": r.export_m3, "drain": r.column.drainage_m3,
            "pref": r.preferential.intake_m3, "matrix": r.matrix.intake_m3, "return": r.column.surface_return_m3}
