"""Native beds and configs for the E3 suite (author-written). The bed fixtures are the E2 ones, loaded by path (unchanged)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "phase2_compact_soil_profile" / "texture_fixtures.py"
_spec = importlib.util.spec_from_file_location("e2_texture_fixtures", _p)
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

from maple_syrup import native_floating_soil as nfs

LAYERS = (0.01, 0.05, 0.10)  # H = 0.16 m
# explicit grain-class -> (sand, silt, clay): fine, medium, coarse, aggregate (never inferred from the diameters)
GRAIN_TEXTURE = np.array([[0.05, 0.45, 0.50], [0.15, 0.65, 0.20], [0.55, 0.30, 0.15], [0.25, 0.50, 0.25]])


def config(*, backend="numpy", layers=LAYERS, bottom="open", overrides=None, density=1250.0, texture=GRAIN_TEXTURE, **kw):
    L = len(layers)
    args = {"layer_thickness_m": layers, "grain_texture": texture, "organic_matter_percent": np.full(L, 3.0), "bulk_density_kg_m3": density,
                "latent_theta_field_capacity": np.full(L, 0.25), "latent_theta_saturation": np.full(L, 0.42), "theta_residual": 0.05, "theta_evaporation_stress": 0.2,
                "diffusivity_m2_s": np.full(L - 1, 1e-8), "bottom_drainage": bottom, "matrix_conductivity_m_s": 2e-6, "matrix_suction_m": 0.1, "matrix_vegetation_factor": 0.7,
                "initial_wetting_m": 0.01, "initial_cumulative_intake_m": 0.02, "overrides": overrides, "backend": backend}
    args.update(kw)
    return nfs.native_floating_config(**args)


def bed_from(parts, mask=None):
    vc, al, g, gc, mres = parts
    ny, nx = int(g.ny), int(g.nx)
    return nfs.NativeBed(vc, al, g, gc, mres, np.ones((ny, nx), bool) if mask is None else mask)


def stratified(shallow=((0, 2),)):
    return bed_from(fx.stratified_dense(shallow_cells=shallow))


def copy_dense(bed):
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    return nfs.NativeBed(VoxelColumnState(mass_kg=bed.voxel_column.mass_kg.copy()), ActiveLayerState(mass_kg=bed.active_layer.mass_kg.copy()), bed.geometry, bed.grain_classes,
                         bed.mass_resolution_kg, bed.active_mask.copy())


def windowed(bed, tmp_path, name="store", window_voxels=3):
    wc = fx.windowed_twin(bed.voxel_column, bed.geometry, bed.mass_resolution_kg, Path(tmp_path) / name, window_voxels=window_voxels)
    return nfs.NativeBed(wc, bed.active_layer, bed.geometry, bed.grain_classes, bed.mass_resolution_kg, bed.active_mask)


def swap_classes(bed, cell, a=0, b=1):
    """A ZERO-NET-ELEVATION composition exchange: every voxel and the active layer of one cell swap class a and b (same totals)."""
    out = copy_dense(bed)
    iy, ix = divmod(cell, int(bed.geometry.nx))
    for arr in (out.voxel_column.mass_kg, out.active_layer.mass_kg):
        sub = arr[iy, ix]
        sub[..., [a, b]] = sub[..., [b, a]]
    return out


def set_cell_from(bed, cell, other_cell):
    out = copy_dense(bed)
    iy, ix = divmod(cell, int(bed.geometry.nx))
    jy, jx = divmod(other_cell, int(bed.geometry.nx))
    out.voxel_column.mass_kg[iy, ix] = bed.voxel_column.mass_kg[jy, jx]
    out.active_layer.mass_kg[iy, ix] = bed.active_layer.mass_kg[jy, jx]
    return out


def shift_top(bed, cell, delta_m):
    """Add (positive) or remove (negative) `delta_m` of material from the TOP FILLED voxel of one cell, keeping its mixture (a pure change of the canonical depth by `delta_m`)."""
    out = copy_dense(bed)
    iy, ix = divmod(cell, int(bed.geometry.nx))
    col = out.voxel_column.mass_kg[iy, ix]
    top = int(np.max(np.flatnonzero(col.sum(axis=1) > 0.0)))
    total = float(col[top].sum())
    col[top] *= (total + fx.PER_M * delta_m) / total
    return out
