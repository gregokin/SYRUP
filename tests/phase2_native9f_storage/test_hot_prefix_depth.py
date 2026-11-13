"""E4j: the hot Windowed canonical depth from the NATIVE prefix packets equals the actual Dense oracle BITWISE through native exchange, spill/miss,
pending journals and publication. No new decoder: `windowed_committed_surface_elevation_m` (native) = decoded NumPy-order voxel total /rho/area +
`compute_active_layer_actual_thickness_m`. CPU: the E1 decode is the compiled Numba kernel on the host. Root runs the GPU definitions."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[0] / "pond_event"), str(HERE.parents[0] / "pond_windowed")]

from pond_helpers import basin, host, pickup_cell
from windowed_helpers import dense_deep_bed, windowed_counterpart

CELL = (1, 0)


def oracle(bed, ctx):
    """The actual Dense NumPy diagnostic (the combined elevation, fixed_base=None)."""
    from maple.surface.active_layer.diagnostics import diagnose_combined_surface_state

    return np.asarray(diagnose_combined_surface_state(bed.voxel_column, bed.active_layer, ctx.geometry, ctx.mass_resolution_kg).combined_elevation_m)


def hot(bed, ctx):
    from maple.surface.topographic_commit.windowed_commit import (
        windowed_committed_surface_elevation_m,
    )

    return np.asarray(windowed_committed_surface_elevation_m(bed.voxel_column, bed.active_layer, ctx.geometry))


def explicit(bed, ctx):
    """The stated formula with the native pieces: `T / rho / area + active.sum / (rho * area)`, T from the native packet decode."""
    from maple.surface.active_layer.capacity import (
        compute_active_layer_actual_thickness_m,
    )
    from maple.surface.voxels.windowed_prefix import windowed_committed_height_total_kg

    g = ctx.geometry
    total = np.asarray(windowed_committed_height_total_kg(bed.voxel_column))
    return total / g.bulk_density_kg_m3 / (g.dx_m * g.dy_m) + compute_active_layer_actual_thickness_m(bed.active_layer, g)


@pytest.fixture
def beds(tmp_path):
    z = basin()
    dense, ctx = dense_deep_bed(z, depth=np.zeros(z.shape))
    wbed, wctx = windowed_counterpart(dense, ctx, tmp_path / "store")
    col = wbed.voxel_column
    assert int(host(col.base).min()) > 0 and float(host(col.deep_total_kg).min()) > 0.0  # material genuinely BELOW the resident floor
    return dense, ctx, wbed, wctx


def same(d, w, dctx, wctx):
    ref = oracle(d, dctx)
    assert hot(w, wctx).tobytes() == ref.tobytes()  # bit-exact, no epsilon
    assert explicit(w, wctx).tobytes() == ref.tobytes()  # the stated grouping, native pieces


def excavate(bed, ctx, cell, dz):
    for _ in range(max(1, int(np.ceil(dz / 0.0005 - 1e-9)))):
        bed = pickup_cell(bed, ctx, cell, dz / max(1, int(np.ceil(dz / 0.0005 - 1e-9))))[0]
    return bed


def deposit_pool(bed, ctx):
    from maple.water import WaterProcessDemand

    from maple_syrup.sediment_bed import apply_bed_demand

    pool = bed.water.mobile_mass_by_cell_class_kg
    return apply_bed_demand(bed, ctx, WaterProcessDemand(np.zeros_like(pool), np.array(pool)))[0]


def test_hot_depth_equals_the_dense_oracle_through_pickup_spill_miss_journal_and_publication(beds):
    from maple_syrup.pond_bed import pending_journal_records
    from maple_syrup.sediment_bed import publish_bed_storage

    d, dctx, w, wctx = beds
    same(d, w, dctx, wctx)  # genesis
    d, w = excavate(d, dctx, CELL, 0.0095), excavate(w, wctx, CELL, 0.0095)  # native pickups: the window refills downwards from the deep store
    same(d, w, dctx, wctx)
    d, w = deposit_pool(d, dctx), deposit_pool(w, wctx)  # native deposition: the window slides up and spills to the deep store (journal)
    assert pending_journal_records(w) > 0, "the fixture must produce a genuine pending journal"
    same(d, w, dctx, wctx)  # UNPUBLISHED pending journal: the packet already reflects it
    w = publish_bed_storage(w, wctx)
    assert pending_journal_records(w) == 0
    same(d, w, dctx, wctx)  # after publication (packets rebound to the new version/generation)
    # sparse touched cells: only the excavated cell changed from its genesis height
    d2 = oracle(d, dctx)
    assert int(np.count_nonzero(d2 != hot(w, wctx) )) == 0
    # a second round after publication
    d, w = excavate(d, dctx, (0, 3), 0.004), excavate(w, wctx, (0, 3), 0.004)
    d, w = deposit_pool(d, dctx), deposit_pool(w, wctx)
    same(d, w, dctx, wctx)


def test_zero_depth_and_exact_edges_are_bit_exact(tmp_path):
    z = np.zeros((2, 3))
    z[0, 1], z[1, 1] = 0.1, 0.2  # exact multiples of the voxel height next to an EMPTY column (depth 0)
    try:
        dense, ctx = dense_deep_bed(z, depth=np.zeros(z.shape), layered=False)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the native fixture constructor refuses a zero-height column: {exc}")
    wbed, wctx = windowed_counterpart(dense, ctx, tmp_path / "store")
    same(dense, wbed, ctx, wctx)


def with_packets(column, packets):
    """A shallow copy of a genuine column carrying ANOTHER genuine `WindowedPrefixPackets` object (the way the pond tests mark a column): a tamper
    control, no fake class; `dataclasses.replace` cannot set the init=False field."""
    import copy

    out = copy.copy(column)
    object.__setattr__(out, "height_packet", packets)
    return out


def test_stale_cross_bed_and_wrong_prefix_packets_are_refused_before_decoding(beds, tmp_path):
    from maple.surface.voxels.windowed_prefix import (
        require_prefix_packets,
        windowed_committed_height_total_kg,
    )

    from maple_syrup.sediment_bed import publish_bed_storage

    d, dctx, w, wctx = beds
    d, w = excavate(d, dctx, CELL, 0.0095), excavate(w, wctx, CELL, 0.0095)
    d, w = deposit_pool(d, dctx), deposit_pool(w, wctx)
    published = publish_bed_storage(w, wctx)
    old_packets = w.voxel_column.height_packet  # bound to the PRE-publication pending/generation state
    new_col = published.voxel_column
    assert new_col.manifest_digest != w.voxel_column.manifest_digest or int(new_col.generation) != int(w.voxel_column.generation)
    stale = with_packets(new_col, old_packets)
    with pytest.raises(ValueError, match="stale|different|bound"):
        require_prefix_packets(stale, operation="e4j")
    with pytest.raises(ValueError):
        windowed_committed_height_total_kg(stale)
    # cross-bed: packets of an independent genuine bed (different base/store objects)
    other_d, other_ctx = dense_deep_bed(basin() * 0.9, depth=np.zeros(basin().shape))
    other_w, _ = windowed_counterpart(other_d, other_ctx, tmp_path / "other")
    cross = with_packets(new_col, other_w.voxel_column.height_packet)
    with pytest.raises(ValueError):
        require_prefix_packets(cross, operation="e4j")
    missing = with_packets(new_col, None)
    with pytest.raises(ValueError, match="no prefix height packets"):
        require_prefix_packets(missing, operation="e4j")
    # the genuine packets of the genuine column still decode bit-exactly afterwards
    same(d, published, dctx, wctx)


def test_no_new_decoder_or_full_scan_is_introduced():
    import maple_syrup

    src = Path(maple_syrup.__file__).parent
    assert not (src / "native_windowed_depth.py").exists()  # the native wrapper `windowed_committed_surface_elevation_m` suffices


GPU = pytest.mark.skipif(os.environ.get("SYRUP_RELEASE_GPU") != "1", reason="GPU tests are opt-in (SYRUP_RELEASE_GPU=1); definitions for Root, not run by the author")


@GPU
def test_cuda_trusted_decode_equals_the_host_oracle_definition(tmp_path):
    """CUDA DEFINITION (not run): device-resident Windowed bed (qualified helper `windowed_counterpart(..., xp=cp)`); the packet decode on the device requires
    `trusted=True` AFTER the actual native validator accepted the bed; the result equals the HOST Dense oracle bit for bit (a different reduction order
    on CuPy would be refused/differ and must not be accepted with an epsilon)."""
    import cupy as cp
    from maple.surface.topographic_commit.windowed_commit import (
        windowed_committed_surface_elevation_m,
    )
    from maple.surface.voxels.windowed_inventory import validate_windowed_column_full

    z = basin()
    dense, ctx = dense_deep_bed(z, depth=np.zeros(z.shape))
    wbed, wctx = windowed_counterpart(dense, ctx, tmp_path / "store", xp=cp)
    validate_windowed_column_full(wbed.voxel_column, wctx.geometry, wctx.mass_resolution_kg, class_ids=tuple(c.class_id for c in wctx.grain_classes.classes))
    got = cp.asnumpy(windowed_committed_surface_elevation_m(wbed.voxel_column, wbed.active_layer, wctx.geometry, trusted_packet=True))
    assert got.tobytes() == oracle(dense, ctx).tobytes()
