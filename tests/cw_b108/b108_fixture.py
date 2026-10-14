"""CW-B108 small actual-MAPLE resident-window fixture: 12 x 7 cells (two deep chunks of 64 + 20 columns), 40 voxels, 3 classes (the task's size limits).

A valley (south outlet row) on a MAPLE bed with thin voxels (1 mm), so ordinary water deposition / erosion moves the
4-voxel window and writes genuine deep journals. An optional seeded mobile load at an upstream cell deposits through
MAPLE's actual water API during the event (part of the initial inventory; no elevation is set by hand). The routing
threshold is the actual MAPLE default `pending_elevation_trigger_m` (0.00075 m); only the commit interval (2 s) is a
LABELLED fixture setting. Built identically for every encoding; only `BedContext.deep_chunk_encoding` /
`deep_overlay_max_bytes` differ between encodings.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

NY, NX, NV, NC = 12, 7, 40, 3
CLASS_IDS = ("phi_1", "phi_2", "phi_3")
FRACTIONS = (0.3, 0.4, 0.3)
COMMIT_INTERVAL_S = 2.0  # LABELLED fixture setting (the routing threshold is the MAPLE default)
MOBILE_CELL = (NY - 2, NX // 2)


def H(value):
    from maple.core.backend import to_host

    return np.asarray(to_host(value))


def elevation():
    rows, cols = np.indices((NY, NX)).astype(np.float64)
    return 0.012 + 0.002 * rows + 0.001 * np.abs(cols - (NX - 1) / 2)


def dense_bed(*, depth_m=0.001, mobile_kg=0.0, voxel_dz=0.001, mobile_cells=(MOBILE_CELL,)):
    """Host dense MAPLE bed (the `test_sediment_event.make_bed` MAPLE sequence, 3 classes / 40 voxels)."""
    from maple.core.boundaries import AxisBoundary, BoundaryKind
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.geometry import GeometrySpec, validate_geometry
    from maple.core.parameters.grain_classes import GrainClass, GrainClassSet, validate_grain_classes
    from maple.core.parameters.numerics import DEFAULT_MASS_RESOLUTION_KG
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.parameters.water_coupling import MAHLERAN_1_2_1_CLASS_DIAMETERS_M
    from maple.core.types.sediment_availability import sediment_availability_state_from_active_layer
    from maple.core.types.sediment_ledger import zeros_sediment_ledger_state
    from maple.core.types.voxel import zeros_voxel_column_state
    from maple.core.types.water import WaterState
    from maple.surface.active_layer import initialize_active_layer_from_voxels
    from maple.surface.topographic_commit import initial_committed_topography_state_from_physical_state
    from maple.surface.voxels import deposit_surface_mixture_batch

    from maple_syrup.sediment_bed import BedContext, BedState

    z = elevation()
    dx, bulk = 0.5, 1250.0
    classes = GrainClassSet(classes=tuple(
        GrainClass(class_id=c, diameter_m=float(d), particle_density_kg_m3=2650.0, is_aggregate=False)
        for c, d in zip(CLASS_IDS, MAHLERAN_1_2_1_CLASS_DIAMETERS_M[:NC], strict=True)))
    validate_grain_classes(classes)

    def boundary():
        return AxisBoundary(kind=BoundaryKind.PRESCRIBED, inflow_flux_kg_m_s={c: 0.0 for c in CLASS_IDS})

    geometry = GeometrySpec(nx=NX, ny=NY, dx_m=dx, dy_m=dx, boundary_x=boundary(), boundary_y=boundary(),
                            voxel_dz_m=voxel_dz, bulk_density_kg_m3=bulk, active_layer_thickness_m=0.002)
    validate_geometry(geometry, classes.ids())
    mres = float(DEFAULT_MASS_RESOLUTION_KG)
    assert float(z.max()) < NV * voxel_dz
    request = (z * bulk * dx * dx)[:, :, None] * np.asarray(FRACTIONS)
    column = zeros_voxel_column_state(NY, NX, NV, NC)
    column, _ = deposit_surface_mixture_batch(column, request, geometry, mres)
    column, active, init = initialize_active_layer_from_voxels(column, geometry, mres)
    assert not np.any(init.underfilled_mask)
    mobile = np.zeros((NY, NX, NC))
    for cell in mobile_cells:
        mobile[cell] = mobile_kg * np.asarray(FRACTIONS)
    water = WaterState(np.full((NY, NX), float(depth_m)), mobile)
    ledger = zeros_sediment_ledger_state(NY, NX, NC)
    committed = initial_committed_topography_state_from_physical_state(column, active, geometry, mres)
    available = sediment_availability_state_from_active_layer(active.mass_kg, np.ones_like(active.mass_kg))
    bed = BedState(column, active, water, ledger, committed, available)
    ctx = BedContext(geometry, classes, mres, TopographicCommitSpec(commit_interval_s=COMMIT_INTERVAL_S),
                     AvalanchingSpec(enabled=False), np.ones_like(active.mass_kg))
    return bed, ctx


def sediment_parameters(xp):
    """The Plot 1 wet-law constants restricted to the first three classes (same convention as the 6-class fixture)."""
    from maple_syrup.sediment_physics import (
        LEGACY_CLASS_RADII_M,
        LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM,
        PLOT1_ACTIVE_LAYER_SENSITIVITY_MM,
        PLOT1_KE_MODEL,
        PLOT1_PARTICLE_DENSITY_G_CM3,
        PLOT1_RAINDROP_A,
        PLOT1_RAINDROP_B,
        PLOT1_RAINDROP_C,
        PLOT1_RAINDROP_MAX_MM,
        sediment_physics_parameters,
    )

    return sediment_physics_parameters(
        diameter_m=tuple(2.0 * r for r in LEGACY_CLASS_RADII_M[:NC]), raindrop_a=PLOT1_RAINDROP_A[:NC],
        raindrop_b=PLOT1_RAINDROP_B[:NC], raindrop_c=PLOT1_RAINDROP_C[:NC],
        raindrop_max_depth_mm=PLOT1_RAINDROP_MAX_MM[:NC], particle_density_g_cm3=PLOT1_PARTICLE_DENSITY_G_CM3,
        active_layer_sensitivity_mm=PLOT1_ACTIVE_LAYER_SENSITIVITY_MM, ke_model=PLOT1_KE_MODEL,
        raindrop_depth_attenuation_per_cm=LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM[:NC], xp=xp)


def build(xp, store_dir, *, encoding="full", overlay_max_bytes=None, mobile_kg=0.45, deep_publication="step",
          high_water=None, max_journal_records=None, chunk_columns=64, rain_s=20.0, rain_mm_h=100.0,
          mobile_cells=(MOBILE_CELL,)):
    """`(inputs, ctx, control, schedule)` of the windowed fixture in namespace `xp`, with a NEW store at `store_dir`."""
    from maple.core.backend import to_device, to_device_tree
    from maple.core.types.sediment_ledger import process_index, zeros_sediment_ledger_state
    from maple.surface.topographic_commit.windowed_commit import windowed_committed_state_from_dense_baseline
    from maple.surface.voxels.windowed_inventory import windowed_column_from_dense
    from maple.surface.voxels.windowed_prefix import move_prefix_packets
    from test_coupled_backends import column_for_xp, graph_for_xp
    from test_sediment_event import rainfall_field

    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_event import SedimentEventControl, initial_event_state
    from maple_syrup.storm import StormControl

    host, ctx = dense_bed(mobile_kg=mobile_kg, mobile_cells=mobile_cells)
    column = windowed_column_from_dense(host.voxel_column, ctx.geometry, ctx.mass_resolution_kg,
                                        store_dir=Path(store_dir), max_store_bytes=10**8, class_ids=CLASS_IDS,
                                        window_voxels=4, chunk_columns=chunk_columns, with_prefix_packets=True)
    committed, _ = windowed_committed_state_from_dense_baseline(host.committed_topography, column, host.active_layer,
                                                                ctx.geometry)
    ledger = zeros_sediment_ledger_state(NY, NX, NC, present_processes=(process_index("water_erosion_deposition"),))
    bed = replace(host, voxel_column=column, committed_topography=committed, ledger=ledger)
    if xp is not np:
        moved = to_device_tree(bed, xp)
        bed = replace(moved, voxel_column=move_prefix_packets(moved.voxel_column, column, xp))
        ctx = replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, xp))
    options = {} if max_journal_records is None else {"window_max_journal_records": max_journal_records}
    if (encoding, overlay_max_bytes) != ("full", None):  # default: the context exactly as any earlier SYRUP builds it
        options.update(deep_chunk_encoding=encoding, deep_overlay_max_bytes=overlay_max_bytes)
    ctx = replace(ctx, **options)
    implementation = "numba" if xp is np else "cuda"
    control = SedimentEventControl(storm=StormControl(max_dt_s=1.0, implementation=implementation, max_steps=10000),
                                   force_final_commit=False, phase_storage="csr", deep_publication=deep_publication,
                                   deep_publication_high_water_records=high_water)
    graph, terrain = graph_for_xp(bed, xp)
    params, soil0 = column_for_xp(graph, 1e-5, xp)
    state0 = initial_event_state(graph, terrain, bed, ctx, soil0, control=control)
    inputs = {"state0": state0, "column": params, "field": rainfall_field(NY, NX, xp=xp),
              "vegetation": xp.zeros((NY, NX), dtype=np.float64), "sediment": sediment_parameters(xp)}
    return inputs, ctx, control, constant_rainfall(0.0, rain_s, rain_mm_h)


# --- logical / physical comparison helpers ----------------------------------------------------------------------------
def leaves(value, prefix=""):
    """(path, array) of every array reachable through dataclass fields / dicts / tuples (NumPy or CuPy)."""
    import dataclasses

    from maple.core.backend import is_array

    if is_array(value):
        yield prefix, value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            yield from leaves(getattr(value, f.name), f"{prefix}.{f.name}")
    elif isinstance(value, dict):
        for k in sorted(value):
            yield from leaves(value[k], f"{prefix}[{k}]")
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            yield from leaves(v, f"{prefix}[{i}]")


def state_arrays(state):
    """Raw host bytes of every PHYSICAL / logical array of an event state: resident window, prefix packets, deep
    metadata, active layer, ledger, water, availability, committed topography, storm, velocity, phase, routing source.
    The deep version HANDLE (physical files / manifest) is excluded and compared separately as logical identity."""
    out = {}
    skip = {"deep_store", "pending_deep", "window_cache"}  # handles / caches; prefix packets (height_packet) kept
    import dataclasses

    for name in ("bed", "storm", "sediment_velocity_m_s", "phase", "routing_source_elevation_m"):
        value = getattr(state, name)
        if name == "bed":
            for f in dataclasses.fields(value):
                sub = getattr(value, f.name)
                if f.name == "voxel_column":
                    for g in dataclasses.fields(sub):
                        if g.name in skip:
                            continue
                        for path, a in leaves(getattr(sub, g.name), f"bed.voxel_column.{g.name}"):
                            out[path] = H(a)
                    continue
                for path, a in leaves(sub, f"bed.{f.name}"):
                    out[path] = H(a)
        else:
            for path, a in leaves(value, name):
                out[path] = H(a)
    return out


def assert_same_arrays(a, b):
    assert set(a) == set(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
        assert a[k].tobytes() == b[k].tobytes(), k


def logical_identity(column):
    """Logical stored identity of a published windowed column (encoding-independent)."""
    store = column.deep_store
    # A v1 (full) version's logical identity IS its manifest digest (MAPLE `DeepStoreVersion` contract); M94 handles
    # predate the explicit property.
    logical = getattr(store, "logical_manifest_digest", None) or store.manifest_digest
    return {"logical_manifest_digest": logical, "journal_digest": store.journal_digest,
            "version_id": store.version_id, "generation": column.generation}


def dense_bytes(column):
    from maple.surface.voxels.windowed_inventory import windowed_to_dense

    return H(windowed_to_dense(column, max_bytes=10**8).mass_kg).tobytes()


RESULT_FIELDS_EXACT = (
    "n_accepted_steps", "n_rejected_attempts", "n_commits", "n_forced_commits", "n_graph_changes", "n_reroutes",
    "n_net_commits", "n_routing_gate_evaluations", "max_unrouted_net_displacement_m", "routing_source_origin",
    "routing_source_t_s", "rerouted_cells_total", "n_maple_water_calls", "max_transport_substeps_used",
    "n_transport_rejections", "n_phase_canonicalized", "n_phase_rounding_remnants", "phase_steered_cells_total",
    "min_accepted_dt_s", "max_accepted_dt_s", "n_inventory_terms")


def result_arrays(result):
    """Every accumulator / hydrograph array of a SedimentEventResult (host bytes)."""
    import dataclasses

    out = {}
    for f in dataclasses.fields(result):
        if f.name in ("state", "deep_publication", "commit_log", "rejections", "exchange_history"):
            continue
        for path, a in leaves(getattr(result, f.name), f.name):
            out[path] = H(a)
    return out
