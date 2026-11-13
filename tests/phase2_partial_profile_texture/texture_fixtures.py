"""Native MAPLE beds for the soil-profile-texture tests (not a test module).

Two constructions, both through GENUINE MAPLE types and constructors:

* `stratified_dense` -- hand-authored dense column whose every number is chosen so the expected layer masses can be
  worked by hand (see `EXPECTED` below): voxel 0 full of mixture A, voxel 1 full of mixture B, voxel 2 a PARTIAL
  0.02 m of mixture C, voxel 3 empty capacity, and the separate active cap (0.002 m, at MAPLE's target mass) of
  mixture D. Cell (0, 1) is the same column; cell (0, 2) is a deliberately shallow column (used as an inactive cell
  or as the refusal trigger).
* `native_deep_bed` -- a deep column built by MAPLE's own `deposit_surface_mixture_batch` (three strata) and
  `initialize_active_layer_from_voxels`, plus its genuine resident-window twin from `windowed_column_from_dense`
  (window 3 of 40 thin voxels, so `base > 0` and the deep store really holds the lower strata).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DX = 0.5
RHO = 1250.0
AREA = DX * DX
PER_M = RHO * AREA  # 312.5 kg per metre of column
DZ = 0.1
CAPACITY = PER_M * DZ  # 31.25 kg
CAP_M = 0.002
CAP_KG = PER_M * CAP_M  # 0.625 kg
PARTIAL_M = 0.02
PARTIAL_KG = PER_M * PARTIAL_M  # 6.25 kg

MIX_A = np.array([0.5, 0.3, 0.2, 0.0])  # deepest full voxel
MIX_B = np.array([0.1, 0.2, 0.3, 0.4])  # second full voxel
MIX_C = np.array([0.25, 0.25, 0.25, 0.25])  # partial top voxel
MIX_D = np.array([0.0, 0.0, 0.4, 0.6])  # active cap


def grain_classes():
    from maple.core.parameters.grain_classes import (
        GrainClass,
        GrainClassSet,
        validate_grain_classes,
    )

    classes = GrainClassSet(classes=(
        GrainClass(class_id="fine", diameter_m=2e-5, particle_density_kg_m3=2650.0, is_aggregate=False),
        GrainClass(class_id="medium", diameter_m=2e-4, particle_density_kg_m3=2650.0, is_aggregate=False),
        GrainClass(class_id="coarse", diameter_m=1e-3, particle_density_kg_m3=2650.0, is_aggregate=False),
        GrainClass(class_id="aggregate", diameter_m=5e-4, particle_density_kg_m3=1800.0, is_aggregate=True),
    ))
    validate_grain_classes(classes)
    return classes


def geometry(ny: int, nx: int, *, n_voxels_unused: int | None = None, voxel_dz_m: float = DZ):
    from maple.core.boundaries import AxisBoundary, BoundaryKind
    from maple.core.parameters.geometry import GeometrySpec, validate_geometry

    gc = grain_classes()

    def boundary():
        return AxisBoundary(kind=BoundaryKind.PRESCRIBED, inflow_flux_kg_m_s={c.class_id: 0.0 for c in gc.classes})

    g = GeometrySpec(nx=nx, ny=ny, dx_m=DX, dy_m=DX, boundary_x=boundary(), boundary_y=boundary(),
                     voxel_dz_m=voxel_dz_m, bulk_density_kg_m3=RHO, active_layer_thickness_m=CAP_M)
    validate_geometry(g, gc.ids())
    return g


def mass_resolution_kg() -> float:
    from maple.core.parameters.numerics import DEFAULT_MASS_RESOLUTION_KG

    return float(DEFAULT_MASS_RESOLUTION_KG)


def stratified_column() -> tuple[np.ndarray, np.ndarray]:
    """`(voxels (4, 4), active (4,))` of the hand-worked column."""
    voxels = np.zeros((4, 4), dtype=np.float64)
    voxels[0] = CAPACITY * MIX_A
    voxels[1] = CAPACITY * MIX_B
    voxels[2] = PARTIAL_KG * MIX_C
    active = CAP_KG * MIX_D
    return voxels, active


def shallow_column() -> tuple[np.ndarray, np.ndarray]:
    """A column with only a 0.03 m partial voxel 0 below the cap (represented depth 0.032 m)."""
    voxels = np.zeros((4, 4), dtype=np.float64)
    voxels[0] = (PER_M * 0.03) * MIX_B
    active = CAP_KG * MIX_D
    return voxels, active


def stratified_dense(*, shallow_cells=((0, 2),), ny: int = 1, nx: int = 3):
    """Dense `(voxel_column, active_layer, geometry, grain_classes, mass_resolution_kg)`; every cell holds the
    stratified column except `shallow_cells`."""
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    v, a = stratified_column()
    vs, as_ = shallow_column()
    voxels = np.empty((ny, nx, 4, 4), dtype=np.float64)
    active = np.empty((ny, nx, 4), dtype=np.float64)
    for iy in range(ny):
        for ix in range(nx):
            if (iy, ix) in shallow_cells:
                voxels[iy, ix], active[iy, ix] = vs, as_
            else:
                voxels[iy, ix], active[iy, ix] = v, a
    return (VoxelColumnState(mass_kg=voxels), ActiveLayerState(mass_kg=active), geometry(ny, nx), grain_classes(),
            mass_resolution_kg())


# Hand-worked expectations for layers [0.01, 0.05, 0.10] on the stratified column (depth from the material surface):
#   active cap [0, 0.002), voxel 2 [0.002, 0.022), voxel 1 [0.022, 0.122), voxel 0 [0.122, 0.222).
#   layer 1 [0, 0.01):    cap 0.625 kg of D + 0.008/0.02 * 6.25 = 2.5 kg of C                    -> 3.125 kg
#   layer 2 [0.01, 0.06): 0.012/0.02 * 6.25 = 3.75 kg of C + 0.038/0.1 * 31.25 = 11.875 kg of B -> 15.625 kg
#   layer 3 [0.06, 0.16): 0.062/0.1 * 31.25 = 19.375 kg of B + 0.038/0.1 * 31.25 = 11.875 kg of A -> 31.25 kg
LAYERS_3 = (0.01, 0.05, 0.10)
EXPECTED_3 = np.array([
    0.625 * MIX_D + 2.5 * MIX_C,
    3.75 * MIX_C + 11.875 * MIX_B,
    19.375 * MIX_B + 11.875 * MIX_A,
])
DEPTH_3 = 0.222


def native_deep_bed(ny: int = 2, nx: int = 3, *, n_voxels: int = 40, voxel_dz_m: float = 0.01):
    """`(voxel_column, active_layer, geometry, grain_classes, mres)` built by MAPLE's own constructors: three strata
    (A below 0.12 m, B to 0.22 m, C to the surface) on a tilted surface between 0.25 m and ~0.30 m."""
    from maple.core.types.voxel import zeros_voxel_column_state
    from maple.surface.active_layer import initialize_active_layer_from_voxels
    from maple.surface.voxels import deposit_surface_mixture_batch

    g = geometry(ny, nx, voxel_dz_m=voxel_dz_m)
    mres = mass_resolution_kg()
    rows = np.arange(ny, dtype=np.float64)[:, None]
    cols = np.arange(nx, dtype=np.float64)[None, :]
    surface = 0.25 + 0.01 * rows + 0.012 * cols  # < n_voxels * dz = 0.4 m
    column = zeros_voxel_column_state(ny, nx, n_voxels, 4)
    for mix, bottom, top in ((MIX_A, 0.0, 0.12), (MIX_B, 0.12, 0.22), (MIX_C, 0.22, None)):
        upper = surface if top is None else np.minimum(surface, top)
        thickness = np.maximum(upper - bottom, 0.0)
        request = (thickness * PER_M)[:, :, None] * mix[None, None, :]
        column, _ = deposit_surface_mixture_batch(column, request, g, mres)
    column, active, result = initialize_active_layer_from_voxels(column, g, mres)
    assert not bool(np.any(result.underfilled_mask))
    return column, active, g, grain_classes(), mres


def random_native_bed(seed: int, ny: int = 3, nx: int = 5, *, n_voxels: int = 30, voxel_dz_m: float = 0.01):
    """Random strata (random thicknesses and mixtures per stratum) deposited by MAPLE's own batch constructor on a
    random surface; the active layer is initialised natively. Every column is a genuine valid MAPLE state."""
    from maple.core.types.voxel import zeros_voxel_column_state
    from maple.surface.active_layer import initialize_active_layer_from_voxels
    from maple.surface.voxels import deposit_surface_mixture_batch

    rng = np.random.default_rng(seed)
    g = geometry(ny, nx, voxel_dz_m=voxel_dz_m)
    mres = mass_resolution_kg()
    surface = rng.uniform(0.08, 0.28, size=(ny, nx))
    column = zeros_voxel_column_state(ny, nx, n_voxels, 4)
    bottom = np.zeros((ny, nx))
    for _ in range(int(rng.integers(2, 6))):
        mix = rng.dirichlet(np.ones(4))
        thickness = np.minimum(rng.uniform(0.005, 0.08, size=(ny, nx)), surface - bottom)
        request = (thickness * PER_M)[:, :, None] * mix[None, None, :]
        column, _ = deposit_surface_mixture_batch(column, request, g, mres)
        bottom = bottom + thickness
    mix = rng.dirichlet(np.ones(4))
    request = ((surface - bottom) * PER_M)[:, :, None] * mix[None, None, :]
    column, _ = deposit_surface_mixture_batch(column, request, g, mres)
    column, active, result = initialize_active_layer_from_voxels(column, g, mres)
    assert not bool(np.any(result.underfilled_mask))
    return column, active, g, grain_classes(), mres


def windowed_twin(column, g, mres, store_dir: Path, *, window_voxels: int = 3, chunk_columns: int = 64):
    """The genuine resident-window counterpart of a dense column with a NEW deep store at `store_dir`."""
    from maple.surface.voxels.windowed_inventory import windowed_column_from_dense

    return windowed_column_from_dense(column, g, mres, store_dir=Path(store_dir), max_store_bytes=10**9,
                                      class_ids=tuple(c.class_id for c in grain_classes().classes),
                                      window_voxels=window_voxels, chunk_columns=chunk_columns)


def host(x):
    from maple.core.backend import to_host

    return np.asarray(to_host(x))


def texture_bytes(texture) -> dict[str, bytes]:
    """Raw bytes of every physical output (for exact equality between paths)."""
    return {name: np.ascontiguousarray(host(getattr(texture, name))).tobytes() for name in (
        "layer_mass_kg", "layer_fraction", "layer_total_kg", "coverage_m", "represented_depth_m",
        "active_cap_thickness_m", "active_mask")}


def full_mask(ny: int, nx: int) -> np.ndarray:
    return np.ones((ny, nx), dtype=np.bool_)


BUDGET = 10**9
