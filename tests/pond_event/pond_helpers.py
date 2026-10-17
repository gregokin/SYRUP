"""Builders for the pond-event tests (not a test module). Nothing here was run by its author (file-only tools).

The bed is an ACTUAL MAPLE state: geometry / grain classes / mass resolution come from `probe.build_minimal_state` (the
constructor `tests/phase5` uses), the voxel column, active layer, water, ledger, committed topography and availability from
the same public MAPLE constructors with a caller-chosen surface elevation (no fake bed, no monkeypatching). Terrain: cells
`(row, col)`, row 0 = south, dx = 0.5 m; the DEM handed to the topology is MAPLE's own committed elevation.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

DX = 0.5
NC = 6
FRICTION = 5.0


def host(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "get"):
        return np.asarray(x.get())
    return np.asarray(x)


def backends():
    """Parameter ids for the three backends; availability is decided by `skip_unavailable`."""
    return ["numpy", "numba", "cupy"]


def backend_namespace(name):
    import pytest

    if name == "numpy":
        return SimpleNamespace(name=name, hydraulics="numpy", xp=np)
    if name == "numba":
        pytest.importorskip("numba")
        return SimpleNamespace(name=name, hydraulics="numba", xp=np)
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; no GPU claim is made")
    return SimpleNamespace(name=name, hydraulics="cuda", xp=backend.cupy_module())


def make_bed(elevation, *, depth=0.0, mobile=None, xp=np, spec=None):
    """Actual MAPLE `(BedState, BedContext)` on the surface `elevation (ny, nx)` (< 0.30 m), uniform default mixture."""
    from maple.core.backend import to_device, to_device_tree
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.types.sediment_availability import sediment_availability_state_from_active_layer
    from maple.core.types.sediment_ledger import zeros_sediment_ledger_state
    from maple.core.types.voxel import zeros_voxel_column_state
    from maple.core.types.water import WaterState, water_state_from_depth
    from maple.surface.active_layer import initialize_active_layer_from_voxels
    from maple.surface.topographic_commit import initial_committed_topography_state_from_physical_state
    from maple.surface.voxels import deposit_surface_mixture_batch

    from maple_syrup.probe import DEFAULT_CLASS_FRACTIONS, build_minimal_state
    from maple_syrup.sediment_bed import BedContext, BedState

    elevation = np.asarray(elevation, dtype=np.float64)
    ny, nx = elevation.shape
    p = build_minimal_state(ny=ny, nx=nx, dx_m=DX, dy_m=DX)
    g = p.geometry
    fractions = np.asarray(DEFAULT_CLASS_FRACTIONS)
    request = (elevation * g.bulk_density_kg_m3 * g.dx_m * g.dy_m)[:, :, None] * fractions[None, None, :]
    column = zeros_voxel_column_state(ny, nx, p.voxel_column.mass_kg.shape[2], NC)
    column, _ = deposit_surface_mixture_batch(column, request, g, p.mass_resolution_kg)
    column, layer, _ = initialize_active_layer_from_voxels(column, g, p.mass_resolution_kg)
    depth_a = np.broadcast_to(np.asarray(depth, dtype=np.float64), (ny, nx)).copy()
    water = water_state_from_depth(depth_a, NC)
    if mobile is not None:
        water = WaterState(depth_a, np.asarray(mobile, dtype=np.float64))
    committed = initial_committed_topography_state_from_physical_state(column, layer, g, p.mass_resolution_kg)
    available = sediment_availability_state_from_active_layer(layer.mass_kg, np.ones_like(layer.mass_kg))
    state = BedState(column, layer, water, zeros_sediment_ledger_state(ny, nx, NC), committed, available)
    ctx = BedContext(g, p.grain_classes, p.mass_resolution_kg, spec or TopographicCommitSpec(),
                     AvalanchingSpec(enabled=False), np.ones_like(layer.mass_kg))
    if xp is not np:
        state = to_device_tree(state, xp)
        ctx = replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, xp))
    return state, ctx


def plane(ny=3, nx=4, drop=0.02, base=0.20):
    """Surface falling EAST by `drop` per cell (slope drop / dx), constant along rows."""
    cols = np.arange(nx, dtype=np.float64)[None, :]
    return base + drop * (nx - 1 - cols) + np.zeros((ny, 1))


def basin():
    """3 x 5 basin: flat rim 0.22, depression (1, 1..3) = 0.20 / 0.18 / 0.20. Outlet east of (1, 4)."""
    z = np.full((3, 5), 0.22)
    z[1, 1:4] = (0.20, 0.18, 0.20)
    return z


def east_outlets(ny, nx):
    return [(r, nx - 1, 0, 1) for r in range(ny)]


def make_topology(bed, outlets, *, active=None, friction=FRICTION, drop=0.01, allow_closed=False):
    """`PondTopology` on MAPLE's committed elevation; the ring replicates the edge, each outlet's ring cell is `drop`
    lower (a real strictly positive bed drop)."""
    from maple.core.types.topographic_commit import committed_elevation_m

    from maple_syrup.pond_network import build_pond_topology

    z = host(committed_elevation_m(bed.committed_topography)).astype(np.float64)
    ny, nx = z.shape
    full = np.pad(z, 1, mode="edge")
    for r, c, dr, dc in outlets:
        full[r + 1 + dr, c + 1 + dc] = z[r, c] - drop
    act = np.ones((ny, nx), dtype=bool) if active is None else np.asarray(active, dtype=bool)
    return build_pond_topology(full, act, np.full((ny, nx), friction), DX, outlets, allow_closed_components=allow_closed)


def make_event(bed, ctx, topology, *, backend=None, schedule=None, splash=False, ksat=0.0, control=None, policy=None,
               max_dt_s=0.2, **kwargs):
    import numpy as _np

    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.infiltration import column_parameters
    from maple_syrup.pond_event import PondEvent, PondEventControl
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters
    from maple_syrup.splash import splash_parameters

    be = backend or SimpleNamespace(hydraulics="numpy", xp=_np)
    xp = be.xp
    shape = topology.shape
    full = lambda v: xp.full(shape, float(v))  # noqa: E731
    column = column_parameters(model="fixed_ksat", ksat_m_per_s=full(ksat), suction_m=full(0.01),
                               drainage_parameter=full(0.0), theta_sat=full(0.4), soil_thickness_m=full(0.3),
                               active_mask=xp.asarray(topology.active))
    sediment = plot1_sediment_parameters(xp=xp)
    sp = splash_parameters(_np.linspace(0.4, 3.0, NC), xp=xp) if splash else None
    ctl = control or PondEventControl(backend=be.hydraulics, time=ExperimentalControl(
        max_dt_s=max_dt_s, min_dt_s=1.0 / 1024.0, max_retries=12, max_steps=50_000))
    sched = schedule or constant_rainfall(0.0, 1.0, 0.0)
    return PondEvent(ctx, column, sediment, sched, topology, control=ctl, policy=policy, splash=sp, **kwargs)


def initial(event, bed, xp=np):
    return event.initial_state(bed, xp.zeros(event.shape))


def pickup_cell(bed, ctx, cell, dz_m, xp=np):
    """ACTUAL native MAPLE pickup lowering `cell` by about `dz_m` (class mixture of the cell); returns `(bed, result)`."""
    from maple.water import WaterProcessDemand

    from maple_syrup.sediment_bed import apply_bed_demand

    g = ctx.geometry
    mass = dz_m * g.bulk_density_kg_m3 * g.dx_m * g.dy_m
    demand = np.zeros(host(bed.active_layer.mass_kg).shape)
    layer = host(bed.active_layer.mass_kg)[cell]
    demand[cell] = layer / layer.sum() * mass
    zeros = xp.zeros_like(bed.active_layer.mass_kg)
    return apply_bed_demand(bed, ctx, WaterProcessDemand(xp.asarray(demand), zeros))


def move_mass(bed, ctx, src, dst, dz_m, xp=np):
    """Two ACTUAL native MAPLE calls: pick up about `dz_m` of elevation at `src`, then deposit exactly the mass MAPLE
    actually removed into `dst` (the pre-existing mobile pool is untouched). `src == dst` is a same-cell gain and loss
    (net zero); different cells move elevation from one to the other. Returns the bed."""
    from maple.aeolian.flux.face_flux import FaceFluxResult
    from maple.water import WaterProcessDemand

    from maple_syrup.sediment_bed import apply_bed_demand, current_routing_surface_m, with_water

    dr, dc = dst[0] - src[0], dst[1] - src[1]
    if abs(dr) + abs(dc) > 1:
        raise ValueError("move_mass moves mass between the same cell or D4-ADJACENT cells only")
    ny, nx, nc = host(bed.active_layer.mass_kg).shape
    surface0 = host(current_routing_surface_m(bed, ctx))
    mobile0 = host(bed.water.mobile_mass_by_cell_class_kg).copy()
    # call 1: actual native pickup; MAPLE puts the removed mass into the mobile pool of `src`
    picked, result = pickup_cell(bed, ctx, src, dz_m, xp)
    removed = host(result.actual_removal_by_cell_class_kg)[src]
    zeros = xp.zeros_like(bed.active_layer.mass_kg)
    if src == dst:
        deposit = WaterProcessDemand(zeros, xp.asarray(_cell_demand(removed, dst, ny, nx, nc)))
        with_pool = picked
    else:
        # lateral transfer of exactly that mass in the mobile pool (pre-existing pool untouched), with the matching
        # actual D4 face gross/net crossing (+east/+north) so cell balances are consistent
        pool = mobile0.copy()
        pool[dst] += removed
        with_pool = with_water(picked, ctx, picked.water.depth_m, xp.asarray(pool))
        xg, yg = np.zeros((ny, nx + 1, nc)), np.zeros((ny + 1, nx, nc))
        xn, yn = np.zeros_like(xg), np.zeros_like(yg)
        if dr == 0:
            col = max(src[1], dst[1])
            xg[src[0], col], xn[src[0], col] = removed, removed * (1.0 if dc > 0 else -1.0)
        else:
            row = max(src[0], dst[0])
            yg[row, src[1]], yn[row, src[1]] = removed, removed * (1.0 if dr > 0 else -1.0)
        flux = FaceFluxResult(x_face_crossing_mass_kg=xp.asarray(xg), y_face_crossing_mass_kg=xp.asarray(yg),
                              x_face_net_crossing_mass_kg=xp.asarray(xn), y_face_net_crossing_mass_kg=xp.asarray(yn))
        deposit = WaterProcessDemand(zeros, xp.asarray(_cell_demand(removed, dst, ny, nx, nc)), face_flux=flux)
    # call 2: actual native deposition of the relocated mass
    moved, dep = apply_bed_demand(with_pool, ctx, deposit)
    from maple_syrup.conservation import reservoir_bound_kg, volume_roundoff_bound_m3

    deposited = host(dep.deposition_by_cell_class_kg)[dst]
    slack = reservoir_bound_kg(16, mobile0, host(with_pool.water.mobile_mass_by_cell_class_kg), removed, deposited)
    assert np.allclose(deposited, removed, rtol=0.0, atol=slack), "native deposit differs from the actual removal"
    assert np.allclose(host(moved.water.mobile_mass_by_cell_class_kg), mobile0, rtol=0.0, atol=slack), \
        "the pre-existing mobile pool was not preserved"
    surface1 = host(current_routing_surface_m(moved, ctx))
    delta = surface1 - surface0
    area = ctx.geometry.dx_m * ctx.geometry.dy_m
    n_terms = (bed.voxel_column.mass_kg.shape[2] + 1) * NC + 8
    tol = volume_roundoff_bound_m3(n_terms, area * float(max(np.abs(surface0).max(), np.abs(surface1).max()))) / area
    actual_dz = float(removed.sum()) / (ctx.geometry.bulk_density_kg_m3 * area)
    if src == dst:
        assert abs(delta[src]) < tol
    else:
        assert abs(delta[src] + actual_dz) <= tol and abs(delta[dst] - actual_dz) <= tol, delta
    return moved


def _cell_demand(vector, cell, ny, nx, nc):
    demand = np.zeros((ny, nx, nc))
    demand[cell] = vector
    return demand


def wet_plane(be, *, depth=0.003, rain_mm_h=100.0, end_s=100.0, splash=False, **kw):
    """3 x 4 plane falling east (slope 0.04), uniform shallow depth, no infiltration, outlets on the east edge."""
    from maple_syrup.rainfall import constant_rainfall

    xp = be.xp
    bed, ctx = make_bed(plane(), depth=depth, xp=xp)
    topo = make_topology(bed, east_outlets(3, 4), drop=0.02)
    ev = make_event(bed, ctx, topo, backend=be, schedule=constant_rainfall(0.0, end_s, rain_mm_h), splash=splash, **kw)
    return ev, initial(ev, bed, xp)


def dry_plane(be, *, impl="array", rain_mm_h=60.0, **kw):
    """Same plane, DRY, infiltration capacity (1e-3 m/s) above the rain rate so the post-hydrology depth stays 0: the
    dry splash path runs on every cell."""
    from maple_syrup.pond_event import PondEventControl
    from maple_syrup.rainfall import constant_rainfall

    xp = be.xp
    bed, ctx = make_bed(plane(), depth=0.0, xp=xp)
    topo = make_topology(bed, east_outlets(3, 4), drop=0.02)
    ctl = PondEventControl(backend=be.hydraulics, splash_implementation=impl)
    ev = make_event(bed, ctx, topo, backend=be, schedule=constant_rainfall(0.0, 100.0, rain_mm_h), splash=True,
                    ksat=1e-3, control=ctl, **kw)
    return ev, initial(ev, bed, xp)


def leaves(value, path=""):
    """Every dataclass field value reachable from `value` (for the no-live-object check of the state)."""
    if is_dataclass(value) and not isinstance(value, type):
        for f in fields(value):
            yield from leaves(getattr(value, f.name), f"{path}.{f.name}")
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from leaves(v, f"{path}[{k!r}]")
    else:
        yield path, value


def arrays_of(value):
    return [(p, host(v).copy()) for p, v in leaves(value) if hasattr(v, "shape") and hasattr(v, "dtype")]


def assert_unchanged(before, value):
    after = dict(arrays_of(value))
    for path, saved in before:
        np.testing.assert_array_equal(saved, after[path], err_msg=path)


def is_module(x):
    return isinstance(x, ModuleType)
