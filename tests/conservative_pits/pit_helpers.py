"""Small actual-MAPLE beds on opt-in terminal-storage graphs (CW-C2): a donor chain into a strict pit, a flat
basin of two equal-elevation sinks, and a masked-nodata two-column grid. Walls on every ring cell and no export
receiver anywhere, so the only way out of the domain is the external dry reset (as the prepared Chastre
geometry: sinks, no outlets). Nothing here executes MAHLERAN."""

from __future__ import annotations

import numpy as np
from test_coupled_backends import H, device_bed
from test_sediment_event import chain_elevation, make_bed

from maple_syrup.infiltration import column_parameters, initial_soil_water_m
from maple_syrup.rainfall import rainfall_field
from maple_syrup.routing import build_routing_graph
from maple_syrup.sediment_bed import terrain_reference
from maple_syrup.sediment_event import initial_event_state
from maple_syrup.sediment_physics import plot1_sediment_parameters

DX = 0.5
NODATA = -9999.0
FLAT_BASIN = np.array([0.12, 0.12, 0.13, 0.14])[:, None]  # rows 0 and 1: equal-elevation sinks


def walled_dem(bed, *, wall=1.0):
    """The committed MAPLE terrain inside a ring one metre above its highest cell; no export receiver."""
    from maple.core.types.topographic_commit import committed_elevation_m

    zc = H(committed_elevation_m(bed.committed_topography))
    ny, nx = zc.shape
    z = np.full((ny + 2, nx + 2), float(zc.max()) + wall)
    z[1:-1, 1:-1] = zc
    return z, np.zeros(z.shape, dtype=bool)


def pit_graph(bed, xp, *, flat=False, dem=None, active_mask=None, nodata_value=None, masked=False, pit=True):
    z, exports = walled_dem(bed) if dem is None else dem
    ny, nx = z.shape[0] - 2, z.shape[1] - 2
    graph = build_routing_graph(z, exports, np.ones((ny, nx)), DX, active_mask=active_mask, nodata_value=nodata_value,
                                allow_masked_nodata=masked, allow_pit_storage=pit, allow_flat_storage=flat, xp=xp)
    return graph, terrain_reference(z, exports, bed, graph, nodata_value=nodata_value)


def column_for_grid(graph, ksat_grid, xp, *, theta0=0.25, theta_sat=0.4, thickness=0.3):
    """Per-cell ksat column parameters whose active mask is the graph's (required when the grid has inactive
    cells, as the masked-nodata fixture)."""
    from maple.core.backend import to_device

    def full(v):
        return xp.full(graph.shape, float(v), dtype=np.float64)

    active = np.asarray(graph.active, dtype=np.bool_)
    params = column_parameters(model="fixed_ksat", ksat_m_per_s=to_device(np.asarray(ksat_grid, dtype=np.float64), xp),
                               suction_m=full(0.0), drainage_parameter=full(0.0), theta_sat=full(theta_sat),
                               soil_thickness_m=full(thickness),
                               active_mask=None if active.all() else to_device(active, xp))
    return params, initial_soil_water_m(params, full(theta0))


def masked_inputs(xp, *, ksat=1e-7):
    """The masked two-column grid as event inputs: rain only on active cells, columns masked like the graph."""
    from maple.core.backend import to_device

    bed, ctx, graph, terrain, _z, _exports, active = masked_two_column(xp)
    column, soil0 = column_for_grid(graph, np.full(graph.shape, float(ksat)), xp)
    state0 = initial_event_state(graph, terrain, bed, ctx, soil0)
    return {"state0": state0, "column": column,
            "field": rainfall_field(*graph.shape, active_mask=to_device(active, xp), xp=xp),
            "vegetation": xp.zeros(graph.shape, dtype=np.float64), "sediment": plot1_sediment_parameters(xp=xp),
            "ctx": ctx, "xp": xp, "graph": graph, "terrain": terrain, "bed": bed}


def pit_inputs(xp, *, flat=False, n=4, ksat=1e-7, pit_ksat=None, pit_ksat_cells=None, control=None, depth_m=0.0):
    """Donor chain of `n` cells into the lowest cell (a strict pit) or the flat basin; `pit_ksat` (e.g. 0) makes
    the terminal cells' water persist, so completion depends on the explicit policy option. `pit_ksat_cells`
    restricts that override to the listed `(row, col)` cells (default: every initial terminal cell)."""
    bed, ctx = make_bed(FLAT_BASIN if flat else chain_elevation(n), depth_m=depth_m)
    bed, ctx = device_bed(bed, ctx, xp)
    graph, terrain = pit_graph(bed, xp, flat=flat)
    ksat_grid = np.full(graph.shape, float(ksat))
    if pit_ksat is not None:
        if pit_ksat_cells is None:
            ksat_grid[np.asarray(graph.pit_storage, dtype=bool)] = float(pit_ksat)
        else:
            for cell in pit_ksat_cells:
                ksat_grid[cell] = float(pit_ksat)
    column, soil0 = column_for_grid(graph, ksat_grid, xp)
    state0 = initial_event_state(graph, terrain, bed, ctx, soil0, control=control)
    return {"state0": state0, "column": column, "field": rainfall_field(*graph.shape, xp=xp),
            "vegetation": xp.zeros(graph.shape, dtype=np.float64), "sediment": plot1_sediment_parameters(xp=xp),
            "ctx": ctx, "xp": xp, "graph": graph, "terrain": terrain, "bed": bed}


def masked_two_column(xp):
    """Two interior columns: column 0 a walled chain into a pit, column 1 inactive and holding the DEM sentinel
    while the MAPLE bed holds ordinary placeholder elevations there."""
    z_bed = np.concatenate([chain_elevation(4), chain_elevation(4) + 0.005], axis=1)  # (4, 2) bed elevations
    bed, ctx = make_bed(z_bed)
    bed, ctx = device_bed(bed, ctx, xp)
    z, exports = walled_dem(bed)
    z[1:-1, 2] = NODATA  # interior column 1 is masked nodata in the DEM (the bed keeps its placeholder values)
    active = np.zeros((4, 2), dtype=bool)
    active[:, 0] = True
    graph, terrain = pit_graph(bed, xp, dem=(z, exports), active_mask=active, nodata_value=NODATA, masked=True)
    return bed, ctx, graph, terrain, z, exports, active
