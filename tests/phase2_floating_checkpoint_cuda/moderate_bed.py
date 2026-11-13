"""Genuine moderate native Dense bed built DIRECTLY from the public MAPLE constructors (no probe, no `build_minimal_state`, no `wh.dense_deep_bed`): the static primitive geometry/grain recipe of the H2 suite (`static_case.bed_context`,
resized to `ny x nx` with the qualified 0.001 m voxels), `zeros_voxel_column_state`, two strata deposited with `deposit_surface_mixture_batch` at the DESIRED physical elevation, `initialize_active_layer_from_voxels`, and the native
committed-topography / availability / water / ledger constructors, assembled into ONE `BedState`. Pattern: Root's `root_h2_valid_wide_probe.py`. Host (NumPy) only here; GPU tests move it with MAPLE's `to_device_tree`."""
from __future__ import annotations

import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import static_case as sc

from maple_syrup import pond_checkpoint as pc


def elevation(ny: int, nx: int, base: float = 0.05, slope: float = 4e-9) -> np.ndarray:
    return base + slope * (nx - 1 - np.arange(nx, dtype=np.float64))[None, :] + np.zeros((ny, 1))


def build(ny: int = 1, nx: int = 50003, n_voxels: int = 60, *, depth: float = 0.003):
    """`(bed, context, topology, z)`: a native Dense `BedState` on `z`, its host `BedContext` and the east-outlet `PondTopology` of the same terrain. The shared `static_case` globals are set only for the construction and restored."""
    from maple.core.types.sediment_availability import (
        sediment_availability_state_from_active_layer,
    )
    from maple.core.types.sediment_ledger import (
        process_index,
        zeros_sediment_ledger_state,
    )
    from maple.core.types.voxel import zeros_voxel_column_state
    from maple.core.types.water import water_state_from_depth
    from maple.surface.active_layer import initialize_active_layer_from_voxels
    from maple.surface.topographic_commit import (
        initial_committed_topography_state_from_physical_state,
    )
    from maple.surface.voxels import deposit_surface_mixture_batch

    from maple_syrup import probe
    from maple_syrup.sediment_bed import BedState

    z = elevation(ny, nx)
    saved = (dict(sc.TERRAIN), sc.CASE["z"], sc.DZ)
    try:
        sc.TERRAIN.update(ny=ny, nx=nx)
        sc.CASE["z"] = z
        context = sc.bed_context()
        topo = sc.topology()
    finally:
        sc.TERRAIN.clear()
        sc.TERRAIN.update(saved[0])
        sc.CASE["z"], sc.DZ = saved[1], saved[2]
    g, nc = context.geometry, sc.NC
    if z.max() >= n_voxels * g.voxel_dz_m:
        raise ValueError("the terrain exceeds the voxel capacity")
    column = zeros_voxel_column_state(ny, nx, n_voxels, nc)
    per_m = g.bulk_density_kg_m3 * g.dx_m * g.dy_m
    lower, upper = np.asarray(sc_lower_fractions()), np.asarray(probe.DEFAULT_CLASS_FRACTIONS)
    column, _ = deposit_surface_mixture_batch(column, (0.6 * z * per_m)[:, :, None] * lower[None, None, :], g, context.mass_resolution_kg)
    column, _ = deposit_surface_mixture_batch(column, (0.4 * z * per_m)[:, :, None] * upper[None, None, :], g, context.mass_resolution_kg)
    column, layer, _ = initialize_active_layer_from_voxels(column, g, context.mass_resolution_kg)
    committed = initial_committed_topography_state_from_physical_state(column, layer, g, context.mass_resolution_kg)
    availability = sediment_availability_state_from_active_layer(layer.mass_kg, np.ones_like(layer.mass_kg))
    ledger = zeros_sediment_ledger_state(ny, nx, nc, present_processes=(process_index("water_erosion_deposition"),))
    bed = BedState(column, layer, water_state_from_depth(np.full((ny, nx), depth), nc), ledger, committed, availability)
    return bed, context, topo, z


def sc_lower_fractions():
    """The qualified lower-stratum fractions of the windowed pond helpers (read from the accepted helper module, not re-typed)."""
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "pond_windowed" / "windowed_helpers.py"
    spec = importlib.util.spec_from_file_location("gf6_windowed_helpers", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LOWER_FRACTIONS


def residency_components(ck, man, ny, nx, nc):
    """Truthful residency components of a Windowed `/8` checkpoint from the REAL native NPZ member HEADERS (no array is read) and file sizes. The whole checkpoint also holds the dense Pond phase/hydraulic/sediment/soil/ledger
    arrays, so its TOTAL can exceed the logical bed; the meaningful claims are about the NATIVE window, exactly W / full-V, and the absence of any full cells x V x classes member."""
    deep = man["deep_storage"]
    logical = ny * nx * deep["n_voxels"] * nc * 8
    members = {}
    with zipfile.ZipFile(ck / pc.PAYLOADS[0]) as z:
        for info in z.infolist():
            with z.open(info) as m:
                shape, _fortran, dtype = pc._read_npy_header(m, info.filename)
            members[info.filename[: -len(".npy")]] = {"shape": [int(n) for n in shape], "dtype": dtype.str, "bytes": int(info.file_size)}
    window = members["window_kg"]
    assert window["shape"] == [ny, nx, deep["window_voxels"], nc] and window["bytes"] >= ny * nx * deep["window_voxels"] * nc * 8
    window_data = ny * nx * deep["window_voxels"] * nc * 8
    assert Fraction(window_data, logical) == Fraction(deep["window_voxels"], deep["n_voxels"])  # EXACT native window / full-V ratio (3/60 = 1/20 here), not an invented saving
    assert all(m["bytes"] < logical for m in members.values()) and not any(m["shape"][-2:] == [deep["n_voxels"], nc] for m in members.values())  # no full cells x V x classes member
    store = Path(deep["store_dir"])
    assert store.is_dir() and ck.resolve() not in store.resolve().parents and {p.name for p in ck.iterdir()} == {"floating_pond_checkpoint.json", *pc.PAYLOADS}  # the deep store is EXTERNAL and not copied into the checkpoint
    native_total = sum(m["bytes"] for m in members.values())
    return {"logical_full_v_bed_bytes": logical, "window_data_bytes": window_data, "window_to_full_v": f"{deep['window_voxels']}/{deep['n_voxels']}", "native_member_bytes": native_total,
            "native_non_window_bytes": native_total - window["bytes"], "native_members": {k: v["bytes"] for k, v in sorted(members.items())}, "companion_bytes": (ck / pc.PAYLOADS[1]).stat().st_size,
            "native_file_bytes": (ck / pc.PAYLOADS[0]).stat().st_size, "total_disk_bytes": sum(p.stat().st_size for p in ck.iterdir()), "deep_store_external_bytes": sum(p.stat().st_size for p in store.rglob("*") if p.is_file())}
