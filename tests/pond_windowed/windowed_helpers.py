"""Builders for the windowed pond-event tests (not a test module). Nothing here was run by its author (file-only tools).

Genuine native constructors only: a DENSE MAPLE bed (deep column: `NV` thin voxels, layered grain mixture), its resident-window
counterpart from `windowed_column_from_dense` (window of `WINDOW` voxels, prefix/height packets, an actual deep store on disk)
and `windowed_committed_state_from_dense_baseline`. The column is much deeper than the window, so `base > 0` and refills /
spills really reach the deep store. Voxel capacity is ~0.31 kg (dz = 1 mm), so a heavy mobile load (see `LOAD_KG`) overflows the
active layer and the window and produces real journal records.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from pond_helpers import DX, NC, backend_namespace, east_outlets, host, make_event, make_topology, plane  # noqa: F401

NV = 300
DZ = 0.001
WINDOW = 3  # MAPLE's minimum window
CHUNK_COLUMNS = 64  # MAPLE's DeepStoreHeader accepts 64..1024
LOAD_KG = 0.3  # per class and cell: 1.8 kg > active layer (~0.6 kg) + two voxels (~0.3 kg each), < the 2 cm terrain step
LOWER_FRACTIONS = (0.05, 0.05, 0.10, 0.20, 0.30, 0.30)  # coarse-rich lower stratum (upper = the default mixture)


def class_ids(ctx):
    return tuple(c.class_id for c in ctx.grain_classes.classes)


def dense_deep_bed(elevation, *, depth=0.0, mobile=None, available=None, layered=True):
    """Actual dense MAPLE `(BedState, BedContext)` on `elevation (ny, nx)` (< 0.29 m): `NV` voxels of `DZ`, two strata."""
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.types.sediment_availability import sediment_availability_state_from_active_layer
    from maple.core.types.sediment_ledger import process_index, zeros_sediment_ledger_state
    from maple.core.types.voxel import zeros_voxel_column_state
    from maple.core.types.water import WaterState, water_state_from_depth
    from maple.surface.active_layer import initialize_active_layer_from_voxels
    from maple.surface.topographic_commit import initial_committed_topography_state_from_physical_state
    from maple.surface.voxels import deposit_surface_mixture_batch

    from maple_syrup.probe import DEFAULT_CLASS_FRACTIONS, build_minimal_state
    from maple_syrup.sediment_bed import BedContext, BedState

    elevation = np.asarray(elevation, dtype=np.float64)
    ny, nx = elevation.shape
    p = build_minimal_state(ny=ny, nx=nx, dx_m=DX, dy_m=DX, n_voxels=NV, voxel_dz_m=DZ)
    g = p.geometry
    upper = np.asarray(DEFAULT_CLASS_FRACTIONS)
    column = zeros_voxel_column_state(ny, nx, NV, NC)
    per_m = g.bulk_density_kg_m3 * g.dx_m * g.dy_m
    if layered:
        lower = np.asarray(LOWER_FRACTIONS)
        column, _ = deposit_surface_mixture_batch(column, (0.6 * elevation * per_m)[:, :, None] * lower[None, None, :],
                                                  g, p.mass_resolution_kg)
        column, _ = deposit_surface_mixture_batch(column, (0.4 * elevation * per_m)[:, :, None] * upper[None, None, :],
                                                  g, p.mass_resolution_kg)
    else:
        column, _ = deposit_surface_mixture_batch(column, (elevation * per_m)[:, :, None] * upper[None, None, :],
                                                  g, p.mass_resolution_kg)
    column, layer, _ = initialize_active_layer_from_voxels(column, g, p.mass_resolution_kg)
    depth_a = np.broadcast_to(np.asarray(depth, dtype=np.float64), (ny, nx)).copy()
    water = water_state_from_depth(depth_a, NC)
    if mobile is not None:
        water = WaterState(depth_a, np.asarray(mobile, dtype=np.float64))
    fraction = np.ones_like(layer.mass_kg) if available is None else np.asarray(available, dtype=np.float64)
    committed = initial_committed_topography_state_from_physical_state(column, layer, g, p.mass_resolution_kg)
    availability = sediment_availability_state_from_active_layer(layer.mass_kg, fraction)
    ledger = zeros_sediment_ledger_state(ny, nx, NC, present_processes=(process_index("water_erosion_deposition"),))
    state = BedState(column, layer, water, ledger, committed, availability)
    ctx = BedContext(g, p.grain_classes, p.mass_resolution_kg, TopographicCommitSpec(), AvalanchingSpec(enabled=False),
                     fraction)
    return state, ctx


def windowed_counterpart(dense, ctx, store_dir, *, xp=np, window=WINDOW, **ctx_options):
    """The resident-window twin of `dense` (same logical bed) with a NEW deep store at `store_dir`; `ctx_options` are
    forwarded to the `BedContext` (e.g. `window_max_journal_records`, `deep_chunk_encoding`)."""
    from maple.core.backend import to_device, to_device_tree
    from maple.surface.topographic_commit.windowed_commit import windowed_committed_state_from_dense_baseline
    from maple.surface.voxels.windowed_inventory import windowed_column_from_dense
    from maple.surface.voxels.windowed_prefix import move_prefix_packets

    column = windowed_column_from_dense(dense.voxel_column, ctx.geometry, ctx.mass_resolution_kg,
                                        store_dir=Path(store_dir), max_store_bytes=10**9, class_ids=class_ids(ctx),
                                        window_voxels=window, chunk_columns=CHUNK_COLUMNS, with_prefix_packets=True)
    committed, _ = windowed_committed_state_from_dense_baseline(dense.committed_topography, column, dense.active_layer,
                                                                ctx.geometry)
    bed = replace(dense, voxel_column=column, committed_topography=committed)
    if xp is not np:
        moved = to_device_tree(bed, xp)
        bed = replace(moved, voxel_column=move_prefix_packets(moved.voxel_column, column, xp))
        ctx = replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, xp))
    return bed, replace(ctx, **ctx_options)


def control_for(be, *, windowed, deep_publication="step", high_water=None, max_dt_s=0.2):
    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.pond_event import PondEventControl

    return PondEventControl(
        backend=be.hydraulics, time=ExperimentalControl(max_dt_s=max_dt_s, min_dt_s=1.0 / 1024.0, max_retries=12,
                                                        max_steps=50_000),
        bed_storage="windowed" if windowed else "dense", deep_publication=deep_publication,
        deep_publication_high_water_records=high_water)


def pair(be, tmp_path, *, elevation=None, depth=0.003, rain_mm_h=100.0, end_s=100.0, mobile=None, available=None,
         splash=False, ksat=0.0, deep_publication="step", high_water=None, outlets=None, topology_drop=0.02,
         ctx_options=None, prepare=None, **event_kw):
    """`(dense_event, dense_state, windowed_event, windowed_state)` over the SAME terrain, forcing and parameters."""
    from maple_syrup.rainfall import constant_rainfall

    xp = be.xp
    z = plane() if elevation is None else elevation
    dense, ctx = dense_deep_bed(z, depth=depth, mobile=mobile, available=available)
    wbed, wctx = windowed_counterpart(dense, ctx, tmp_path / "store", xp=xp, **(ctx_options or {}))
    topo = make_topology(dense, outlets or east_outlets(*np.asarray(z).shape), drop=topology_drop)
    sched = constant_rainfall(0.0, end_s, rain_mm_h)
    if xp is not np:
        from maple.core.backend import to_device_tree

        dense = to_device_tree(dense, xp)
        ctx = replace(ctx, initial_available_fraction=wctx.initial_available_fraction)
    d_event = make_event(dense, ctx, topo, backend=be, schedule=sched, splash=splash, ksat=ksat,
                         control=control_for(be, windowed=False), **event_kw)
    w_event = make_event(wbed, wctx, topo, backend=be, schedule=sched, splash=splash, ksat=ksat,
                         control=control_for(be, windowed=True, deep_publication=deep_publication,
                                             high_water=high_water), **event_kw)
    if prepare is not None:  # native bed edits applied identically to both beds BEFORE the event state exists
        dense, wbed = prepare(dense, d_event.context), prepare(wbed, w_event.context)
    zeros = xp.zeros(d_event.shape)
    return d_event, d_event.initial_state(dense, zeros), w_event, w_event.initial_state(wbed, zeros)


PICKUP_STEP_M = 0.0005  # one native pickup is limited by the ~2 mm active layer: excavate in <= 0.5 mm native pickups


def _excavate(bed, ctx, cell, dz_m):
    """ACTUAL native pickups (each at most `PICKUP_STEP_M`, each from the CURRENT holdings, so the active layer refills from
    the window and the window refills from the deep store between them). The surface change is MEASURED with the native
    surface (`current_routing_surface_m`) and must match the request. Returns the bed (removed mass in the mobile pool)."""
    from pond_helpers import pickup_cell

    from maple_syrup.sediment_bed import current_routing_surface_m

    xp = _namespace_of(bed)
    n = max(1, int(np.ceil(dz_m / PICKUP_STEP_M - 1e-9)))
    before = host(current_routing_surface_m(bed, ctx))[cell]
    for _ in range(n):
        bed = pickup_cell(bed, ctx, cell, dz_m / n, xp=xp)[0]
    measured = before - host(current_routing_surface_m(bed, ctx))[cell]
    assert abs(measured - dz_m) <= 0.05 * dz_m, f"native excavation lowered the cell by {measured} m, requested {dz_m} m"
    return bed


def lower_cell(cell, dz_m):
    """`prepare` hook: native excavation of `cell` by `dz_m` (see `_excavate`). The removed mass LEAVES the system (the
    mobile pool is restored to its value before, so it cannot settle back and mask the displacement; the event's totals
    start at `initial_state`). The committed topography is NOT updated, so the first step sees the net displacement."""
    from maple_syrup.sediment_bed import with_water

    def prepare(bed, ctx):
        xp = _namespace_of(bed)
        pool = host(bed.water.mobile_mass_by_cell_class_kg).copy()
        picked = _excavate(bed, ctx, cell, dz_m)
        return with_water(picked, ctx, picked.water.depth_m, xp.asarray(pool))

    return prepare


def excavate_cell(cell, dz_m):
    """`prepare` hook: native excavation of `cell` that KEEPS the removed mass in that cell's mobile pool. The window was
    refilled downwards; when the event's native deposition puts the mass back, the window slides up and spills to the deep
    store: a genuine journal (native erosion alone need not produce one)."""

    def prepare(bed, ctx):
        return _excavate(bed, ctx, cell, dz_m)

    return prepare


def chain(*hooks):
    """Compose `prepare` hooks (applied in order)."""

    def prepare(bed, ctx):
        for hook in hooks:
            bed = hook(bed, ctx)
        return bed

    return prepare


def carry_redundant_journal(n_records):
    """`prepare` hook: a STORAGE FIXTURE, not physics. Attaches (through MAPLE's `attach_pending_journal`, with the height
    packets re-bound as MAPLE's own isolated copy does) a valid canonical native `DeepJournal` of `n_records` records that
    rewrite already-buried values of cell 0 with exactly the values the deep store holds (voxels 0..n-1, below `base`): no
    physical change, but a real carried pending journal of a known size on the initial valid windowed bed. A dense bed is
    returned unchanged. The caller's column object is never mutated (a shallow copy carries the marker)."""

    def prepare(bed, ctx):
        from maple_syrup.sediment_bed import is_windowed_bed

        if not is_windowed_bed(bed):
            return bed
        import dataclasses

        from maple.surface.voxels.deep_store import DeepJournal
        from maple.surface.voxels.windowed_pending import attach_pending_journal
        from maple.surface.voxels.windowed_prefix import attach_prefix_packets

        column = bed.voxel_column
        assert column.pending_deep is None and int(host(column.base).reshape(-1)[0]) > n_records
        deep = column.deep_store.gather_columns(np.zeros(1, dtype=np.int64))[0]  # the stored values of cell 0
        journal = DeepJournal.from_records(np.zeros(n_records, dtype=np.int64), np.arange(n_records, dtype=np.int64),
                                           np.array(deep[:n_records]), header=column.deep_store.header)
        twin = dataclasses.replace(column)
        twin = attach_prefix_packets(twin, column.height_packet.packet.copy())
        twin = attach_pending_journal(twin, journal)
        return replace(bed, voxel_column=twin)

    return prepare


def _namespace_of(bed):
    from maple.core.backend import array_namespace

    return array_namespace(bed.water.depth_m)


def heavy_mobile(shape=(3, 4), cells=((1, 0), (1, 1))):
    """A mobile pool of `LOAD_KG` per class on `cells`: more than the active layer holds, so deposition overflows into
    the voxel column and the window spills to the deep store (real journal records)."""
    mobile = np.zeros((*shape, NC))
    for cell in cells:
        mobile[cell] = LOAD_KG
    return mobile


# --- comparisons ------------------------------------------------------------------------------------
HYD = dict(rtol=2e-12, atol=1e-14)  # the existing strict hydraulic policy
SED = dict(rtol=1e-10, atol=1e-14)  # the existing strict sediment policy


def logical_column(bed):
    """Dense host `(ny, nx, nv, nc)` of a windowed bed's PUBLISHED logical column (small tests only)."""
    from maple.surface.voxels.windowed_inventory import windowed_to_dense

    return host(windowed_to_dense(bed.voxel_column, max_bytes=10**9).mass_kg)


def assert_physical_match(dense_state, windowed_state):
    """Hydraulics (strict hydraulic policy) and bed / mobile / layer inventories (strict sediment policy)."""
    h_d, h_w = dense_state.hydraulic, windowed_state.hydraulic
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        np.testing.assert_allclose(host(getattr(h_w, name)), host(getattr(h_d, name)), err_msg=name, **HYD)
    assert h_w.t_s == h_d.t_s
    bd, bw = dense_state.bed, windowed_state.bed
    np.testing.assert_allclose(host(bw.water.mobile_mass_by_cell_class_kg), host(bd.water.mobile_mass_by_cell_class_kg),
                               **SED)
    np.testing.assert_allclose(host(bw.active_layer.mass_kg), host(bd.active_layer.mass_kg), **SED)
    np.testing.assert_allclose(logical_column(windowed_state.bed), host(bd.voxel_column.mass_kg), **SED)
    for name in ("wet_actual", "splash_actual", "deposited_wet", "deposited_splash", "export_wet", "export_splash"):
        np.testing.assert_allclose(host(getattr(windowed_state.sediment, name)), host(getattr(dense_state.sediment, name)),
                                   err_msg=name, **SED)
    cd, cw = dense_state.counters, windowed_state.counters
    for name in ("n_accepted_steps", "n_commits", "n_refreshes", "n_splash_exchanges"):
        assert getattr(cw, name) == getattr(cd, name), name


def physical_bytes(state):
    """Raw bytes of every PHYSICAL array of a windowed event state (store handles / caches excluded): used for the exact
    equalities between publication policies (identical operations, only deep-file durability differs)."""
    import dataclasses

    from maple.core.backend import is_array

    skip = {"deep_store", "pending_deep", "window_cache", "generation", "manifest_digest"}
    out = {}

    def walk(value, path):
        if is_array(value) or isinstance(value, np.generic):
            # NumPy reductions yield scalar np.generic; checkpoint decodes the same typed bits as a 0-d array.
            out[path] = host(value).tobytes()
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for f in dataclasses.fields(value):
                if f.name not in skip:
                    walk(getattr(value, f.name), f"{path}.{f.name}")
        elif isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")

    walk(state.bed, "bed")
    for name in ("hydraulic", "sediment_velocity_m_s", "phase_fraction", "phase_position_m", "held_source_m", "water",
                 "sediment", "quiet"):
        walk(getattr(state, name), name)
    return out


def store_identity(state):
    store = state.bed.voxel_column.deep_store
    return store.version_id, store.manifest_digest
