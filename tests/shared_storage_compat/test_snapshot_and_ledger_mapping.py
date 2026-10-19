"""Native snapshot continuation / refusals on a SYRUP windowed bed, and the seven-process ledger dimension/mapping contract.

Snapshot part: MAPLE's `save_windowed_state_snapshot` / `load_windowed_state_snapshot` (what SYRUP's pond checkpoint embeds)
on a genuine published bed; this is NOT SYRUP's `pond_checkpoint` (covered by tests/pond_windowed_checkpoint) and not a
restart of a full event. Ledger part: merged MAPLE has seven canonical channels (`surface_creep` last). A compact ledger
compacts ONLY its eight directional spatial arrays (leading axis = stored slot, mapped by `slot_of`); `process_totals_kg` and
its compensation keep the CANONICAL `(N_LEDGER_PROCESSES, n_classes)` shape (`sediment_ledger.py`, class docstring of
`CompactSedimentLedgerState`). So `water_channel_index()` (3) is correct for totals/`ledger_reset` on dense AND compact ledgers,
and `slot_of` is the mapping for directional arrays only; both distinctions are pinned below. (Stage A's finding F2 claimed
otherwise and is withdrawn.) Not executed by its author (file tools only)."""
from __future__ import annotations

import dataclasses
import os
import stat

import numpy as np
import pytest
from compat_support import assert_exact_equal, host, make_transport, physical_arrays

from maple_syrup.pond_bed import pickup_transport_deposit
from maple_syrup.sediment_bed import publish_bed_storage


def _step(bed, ctx):
    request = 0.05 * host(bed.active_layer.mass_kg)
    pool0 = host(bed.water.mobile_mass_by_cell_class_kg).copy()
    return pickup_transport_deposit(bed, ctx, bed.water.depth_m, request, make_transport(pool0, 1.0)[0])[0]


def _save(bed, ctx, path, *, time_s=1.5, step=3):
    from maple.surface.voxels.windowed_snapshot import save_windowed_state_snapshot

    save_windowed_state_snapshot(bed.voxel_column, bed.active_layer, time_s, path, step=step, geometry=ctx.geometry,
                                 mass_resolution_kg=ctx.mass_resolution_kg, sediment_availability=bed.sediment_availability,
                                 water=bed.water)


def _load(path, ctx, max_bytes=10**9):
    from maple.surface.voxels.windowed_snapshot import load_windowed_state_snapshot

    return load_windowed_state_snapshot(path, geometry=ctx.geometry, grain_classes=ctx.grain_classes,
                                        mass_resolution_kg=ctx.mass_resolution_kg, max_bytes=max_bytes)


def test_pending_journal_is_refused_by_the_native_snapshot_and_nothing_is_written(windowed_bed, tmp_path):
    bed, ctx = windowed_bed
    pending = _step(bed, ctx)
    assert pending.voxel_column.pending_deep is not None
    out = tmp_path / "snap"
    out.mkdir()
    with pytest.raises(ValueError):
        _save(pending, ctx, out / "state.npz")
    assert list(out.iterdir()) == []


def test_snapshot_round_trip_continues_exactly_and_refuses_misuse(windowed_bed, tmp_path):
    bed, ctx = windowed_bed
    published = publish_bed_storage(_step(bed, ctx), ctx)
    assert published.voxel_column.pending_deep is None
    out = tmp_path / "snap"
    out.mkdir()
    path = out / "state.npz"
    _save(published, ctx, path)

    loaded = _load(path, ctx)
    assert loaded["restartable"] is True and loaded["time_s"] == 1.5 and loaded["step"] == 3
    restored = dataclasses.replace(published, voxel_column=loaded["voxel_column"], active_layer=loaded["active_layer"],
                                   water=loaded["water"], sediment_availability=loaded["sediment_availability"])
    assert restored.voxel_column.manifest_digest == published.voxel_column.manifest_digest
    assert_exact_equal(physical_arrays(restored), physical_arrays(published), "loaded snapshot")
    # continuation: the same next step on the original and on the restored bed is bit-identical
    assert_exact_equal(physical_arrays(_step(published, ctx)), physical_arrays(_step(restored, ctx)), "continuation")

    with pytest.raises(ValueError, match="already exists"):
        _save(published, ctx, path)
    with pytest.raises(ValueError, match=r"\.npz"):
        _save(published, ctx, out / "state.dat")
    with pytest.raises(ValueError, match="bounded"):
        _load(path, ctx, max_bytes=16)

    # source refusal: the referenced version's manifest no longer hashes to its content address
    store = published.voxel_column.deep_store
    # The immutable version file is read-only by design; this TEST-OWNED tmp store file is corrupted deliberately with a
    # permission change and restored in `finally` (no live or reference store is involved).
    manifest = store.store_dir / "versions" / f"{store.manifest_digest}.json"
    original_mode = stat.S_IMODE(manifest.stat().st_mode)
    original = manifest.read_bytes()
    os.chmod(manifest, original_mode | stat.S_IWUSR)
    try:
        manifest.write_bytes(original + b" ")
        with pytest.raises((ValueError, OSError)):
            _load(path, ctx)
    finally:
        manifest.write_bytes(original)
        os.chmod(manifest, original_mode)
    assert manifest.read_bytes() == original
    assert _load(path, ctx)["step"] == 3  # restored: the genuine snapshot loads again


DIRECTIONAL = ("numerical_residual_positive_kg", "numerical_residual_negative_kg", "actual_mass_positive_kg",
               "actual_mass_negative_kg", "numerical_residual_positive_compensation_kg",
               "numerical_residual_negative_compensation_kg", "actual_mass_positive_compensation_kg",
               "actual_mass_negative_compensation_kg")


def test_canonical_channels_are_seven_with_surface_creep_last_and_only_directional_arrays_compact(windowed_bed):
    from maple.core.types.sediment_ledger import (
        CANONICAL_PRESENT_PROCESSES,
        LEDGER_PROCESS_NAMES,
        N_LEDGER_PROCESSES,
        zeros_sediment_ledger_state,
    )

    bed, ctx = windowed_bed
    assert N_LEDGER_PROCESSES == len(LEDGER_PROCESS_NAMES) == 7 and LEDGER_PROCESS_NAMES[-1] == "surface_creep"
    dense = zeros_sediment_ledger_state(ctx.geometry.ny, ctx.geometry.nx, len(ctx.grain_classes.classes))
    assert tuple(dense.present_processes) == CANONICAL_PRESENT_PROCESSES
    compact = bed.ledger
    assert len(compact.present_processes) == 1
    nc = len(ctx.grain_classes.classes)
    for ledger in (dense, compact):  # totals and their compensation are CANONICAL on both representations
        assert ledger.process_totals_kg.shape == (7, nc)
        assert ledger.process_totals_compensation_kg.shape == (7, nc)
    for name in DIRECTIONAL:  # only the eight directional arrays are compacted to the stored slots
        assert getattr(dense, name).shape[0] == 7, name
        assert getattr(compact, name).shape[0] == 1, name


def test_canonical_totals_agree_exactly_between_compact_and_dense_and_directional_rows_follow_slot_of(windowed_bed):
    from maple.core.types.sediment_ledger import (
        process_index,
        slot_of,
        zeros_sediment_ledger_state,
    )

    bed, ctx = windowed_bed
    water = process_index("water_erosion_deposition")
    assert water == 3
    dense_ledger = zeros_sediment_ledger_state(ctx.geometry.ny, ctx.geometry.nx, len(ctx.grain_classes.classes))
    compact_after = _step(bed, ctx)
    dense_after = _step(dataclasses.replace(bed, ledger=dense_ledger), ctx)
    assert slot_of(dense_ledger, water) == water and slot_of(bed.ledger, water) == 0

    # canonical totals and compensation: the ENTIRE (7, nc) arrays are identical, row 3 is the water channel on both
    for name in ("process_totals_kg", "process_totals_compensation_kg"):
        np.testing.assert_array_equal(host(getattr(compact_after.ledger, name)), host(getattr(dense_after.ledger, name)),
                                      err_msg=name)
    totals = host(dense_after.ledger.process_totals_kg)
    assert np.any(totals[water])
    assert not np.any(totals[[p for p in range(7) if p != water]]), "a non-water channel moved mass"

    # directional arrays: the compact ledger stores the water channel in slot 0, the dense one in row 3
    slot = slot_of(bed.ledger, water)
    for name in DIRECTIONAL:
        compact_rows, dense_rows = host(getattr(compact_after.ledger, name)), host(getattr(dense_after.ledger, name))
        np.testing.assert_array_equal(compact_rows[slot], dense_rows[water], err_msg=name)
        assert not np.any(dense_rows[[p for p in range(7) if p != water]]), name


@pytest.mark.parametrize("ledger_kind", ["dense", "compact"])
def test_report_rows_use_the_canonical_water_index_and_directional_rows_use_the_slot(windowed_bed, ledger_kind):
    """`sediment_event.py` allocates `ledger_reset` with `ledger.process_totals_kg.shape` and
    `sediment_experiment.py` reports `ledger_reset[water_channel_index()]`: valid on BOTH ledgers because the totals axis is
    canonical. `slot_of` is the (different) mapping of the directional arrays and equals 0 only on the compact water ledger."""
    from maple.core.types.sediment_ledger import slot_of, zeros_sediment_ledger_state

    from maple_syrup.sediment_event import water_channel_index

    bed, ctx = windowed_bed
    ledger = bed.ledger if ledger_kind == "compact" else zeros_sediment_ledger_state(
        ctx.geometry.ny, ctx.geometry.nx, len(ctx.grain_classes.classes))
    reset = np.zeros(ledger.process_totals_kg.shape)
    assert reset.shape[0] == 7 and water_channel_index() == 3
    assert reset[water_channel_index()].shape == (len(ctx.grain_classes.classes),)
    assert slot_of(ledger, water_channel_index()) == (0 if ledger_kind == "compact" else 3)
