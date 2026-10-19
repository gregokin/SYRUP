"""SYRUP adapter x merged MAPLE windowed exchange: finite supply, class closure, callback mass vs actual pickup,
paired/public equivalence, exact publication once and idempotence, and late-failure atomicity.

Everything runs through SYRUP's own adapter entries (`pond_bed.pickup_transport_deposit`, `sediment_bed.apply_bed_demand`,
`sediment_bed.publish_bed_storage`) on genuine native windowed beds. The comparisons are EXACT where MAPLE documents raw
identity (paired vs sequential public calls) and use the repository's own `conservation.reservoir_bound_kg` otherwise; no
tolerance is widened and no physical field is exempted. A CPU/NumPy contract only: no GPU, wind, growth, avalanche, storm or
performance claim. Authored with file tools only: not executed by its author."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from compat_support import (
    assert_exact_equal,
    class_inventory_kg,
    host,
    make_transport,
    physical_arrays,
    sequential_reference,
    tree_fingerprint,
)

from maple_syrup.conservation import reservoir_bound_kg
from maple_syrup.pond_bed import pickup_transport_deposit
from maple_syrup.sediment_bed import is_windowed_bed, publish_bed_storage


def _request(bed, factor):
    return factor * host(bed.active_layer.mass_kg)


def _paired(bed, ctx, request, fraction):
    pool0 = host(bed.water.mobile_mass_by_cell_class_kg).copy()
    transport, seen = make_transport(pool0, fraction)
    result = pickup_transport_deposit(bed, ctx, bed.water.depth_m, request, transport)
    return (*result, seen, pool0)


def test_paired_transaction_equals_the_sequential_public_calls_exactly(windowed_bed):
    bed, ctx = windowed_bed
    assert is_windowed_bed(bed)
    request = _request(bed, 0.05)
    before = physical_arrays(bed)
    store_dir = bed.voxel_column.deep_store.store_dir
    files_before = tree_fingerprint(store_dir)

    paired_bed, view, deposit, _seen, pool0 = _paired(bed, ctx, request, 1.0)
    transport, _ = make_transport(pool0, 1.0)
    seq_bed, pickup, seq_deposit = sequential_reference(bed, ctx, bed.water.depth_m, request, transport)

    assert paired_bed.voxel_column.pending_deep is not None, "fixture must produce a real pending journal (not vacuous)"
    assert_exact_equal(physical_arrays(paired_bed), physical_arrays(seq_bed), "paired vs sequential")
    np.testing.assert_array_equal(host(view.actual_removal_by_cell_class_kg), host(pickup.actual_removal_by_cell_class_kg))
    np.testing.assert_array_equal(host(deposit.deposition_by_cell_class_kg), host(seq_deposit.deposition_by_cell_class_kg))
    np.testing.assert_array_equal(host(deposit.deposited_mass_by_class_kg), host(seq_deposit.deposited_mass_by_class_kg))
    # neither path touched the caller's bed or wrote a deep-store file
    assert_exact_equal(physical_arrays(bed), before, "caller bed")
    assert tree_fingerprint(store_dir) == files_before


def test_receipt_reports_the_actual_caps_and_what_the_paired_owner_did(windowed_bed):
    from maple.surface.voxels.deep_store import DEFAULT_MAX_JOURNAL_RECORDS
    from maple.surface.voxels.windowed_transfer import DEFAULT_EXCHANGE_GATHER_COLUMNS

    bed, ctx = windowed_bed
    assert ctx.window_max_journal_records is None  # SYRUP then passes no cap: MAPLE's own default applies
    _, _, deposit, _, _ = _paired(bed, ctx, _request(bed, 0.05), 1.0)
    report = deposit.exchange_receipt.report
    assert report["max_gather_columns"] == DEFAULT_EXCHANGE_GATHER_COLUMNS == 256
    assert report["max_journal_records"] == DEFAULT_MAX_JOURNAL_RECORDS
    assert report["deep_writes_performed_here"] == 0
    assert isinstance(report["transient_bytes_estimate"], int) and report["transient_bytes_estimate"] > 0
    assert {"journal_records", "window_voxels", "resident_window_bytes", "transfer_counters_delta",
            "deep_store_counters_delta"} <= set(report)
    owner = report["paired_transaction"]
    assert set(owner) == {"begin_mode", "first_accumulation", "second_accumulation"}
    assert all(isinstance(v, str) and v for v in owner.values())

    _seq_bed, _, seq_deposit = sequential_reference(bed, ctx, bed.water.depth_m, _request(bed, 0.05),
                                                   make_transport(host(bed.water.mobile_mass_by_cell_class_kg), 1.0)[0])
    assert "paired_transaction" not in seq_deposit.exchange_receipt.report


def test_context_journal_cap_is_forwarded_and_reported(windowed_bed):
    bed, ctx = windowed_bed
    capped = dataclasses.replace(ctx, window_max_journal_records=5000)
    _, _, deposit, _, _ = _paired(bed, capped, _request(bed, 0.05), 1.0)
    assert deposit.exchange_receipt.report["max_journal_records"] == 5000


def test_finite_supply_class_closure_and_callback_mass_equal_the_actual_pickup(windowed_bed):
    bed, ctx = windowed_bed
    request = _request(bed, 5.0)  # asks for five times what each cell's active layer holds
    before = class_inventory_kg(bed)
    pool0 = host(bed.water.mobile_mass_by_cell_class_kg).copy()

    new_bed, view, deposit, seen, _ = _paired(bed, ctx, request, 0.5)
    actual = host(view.actual_removal_by_cell_class_kg)

    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], actual)  # the callback saw exactly the actual pickup, not the request
    slack = reservoir_bound_kg(16, before, class_inventory_kg(new_bed), request.sum(axis=(0, 1)), pool0)
    assert np.all(actual >= 0.0) and np.all(actual <= request + slack)
    assert actual.sum() < request.sum() - slack, "supply is finite: the actual pickup must fall short of the request"
    np.testing.assert_allclose(host(view.requested_removal_by_class_kg), request.sum(axis=(0, 1)), rtol=0.0, atol=slack)
    np.testing.assert_allclose(host(view.actual_removal_by_class_kg), actual.sum(axis=(0, 1)), rtol=0.0, atol=slack)
    # mobile bookkeeping: pool0 + actual pickup - actual deposition
    deposited = host(deposit.deposition_by_cell_class_kg)
    np.testing.assert_allclose(host(new_bed.water.mobile_mass_by_cell_class_kg), pool0 + actual - deposited,
                               rtol=0.0, atol=slack)
    # class closure: bed + mobile is conserved per class (no boundary export in this fixture)
    np.testing.assert_allclose(class_inventory_kg(new_bed), before, rtol=0.0, atol=slack)


def test_public_pickup_moves_exactly_the_reported_mass_into_the_mobile_pool(light_windowed_bed):
    from maple.water import WaterProcessDemand

    from maple_syrup.sediment_bed import apply_bed_demand

    bed, ctx = light_windowed_bed
    before = class_inventory_kg(bed)
    request = _request(bed, 0.2)
    new_bed, result = apply_bed_demand(bed, ctx, WaterProcessDemand(request, np.zeros_like(request)))
    actual = host(result.actual_removal_by_cell_class_kg)
    slack = reservoir_bound_kg(16, before, class_inventory_kg(new_bed), request.sum(axis=(0, 1)))
    assert np.all(actual >= 0.0) and np.all(actual <= request + slack)
    np.testing.assert_allclose(host(new_bed.water.mobile_mass_by_cell_class_kg),
                               host(bed.water.mobile_mass_by_cell_class_kg) + actual, rtol=0.0, atol=slack)
    np.testing.assert_allclose(class_inventory_kg(new_bed), before, rtol=0.0, atol=slack)


def test_publication_is_exactly_once_exact_and_idempotent(windowed_bed):
    from maple.surface.voxels.deep_store import load_version, read_deep_store_counters
    from maple.surface.voxels.windowed_inventory import column_mass_by_class_kg

    bed, ctx = windowed_bed
    old = bed.voxel_column.deep_store
    new_bed = _paired(bed, ctx, _request(bed, 0.05), 1.0)[0]
    column = new_bed.voxel_column
    assert column.pending_deep is not None and column.pending_deep.journal.n_records > 0
    files_pending = tree_fingerprint(old.store_dir)
    inventory = host(column_mass_by_class_kg(column)).copy()
    arrays = physical_arrays(new_bed)
    arrays = {k: v for k, v in arrays.items() if not k.startswith("journal_")}
    published_before = read_deep_store_counters()["versions_published"]

    published = publish_bed_storage(new_bed, ctx)

    assert read_deep_store_counters()["versions_published"] == published_before + 1
    assert published.voxel_column.pending_deep is None
    assert published.voxel_column.manifest_digest != old.manifest_digest
    assert published.voxel_column.generation != old.version_id
    files_published = tree_fingerprint(old.store_dir)
    assert files_published.keys() > files_pending.keys()
    assert all(files_published[name] == digest for name, digest in files_pending.items()), "an old store file changed"
    # publication persists; it never exchanges mass
    np.testing.assert_array_equal(host(column_mass_by_class_kg(published.voxel_column)), inventory)
    assert_exact_equal({k: v for k, v in physical_arrays(published).items()}, arrays, "publication changed a resident array")
    # the older referenced version keeps its own truth and stays loadable
    reopened = load_version(old.store_dir, old.manifest_digest, geometry=ctx.geometry,
                            mass_resolution_kg=ctx.mass_resolution_kg,
                            class_ids=tuple(c.class_id for c in ctx.grain_classes.classes),
                            expected_header_digest=old.header.digest())
    assert reopened.version_id == old.version_id

    again = publish_bed_storage(published, ctx)
    assert again.voxel_column is published.voxel_column
    assert read_deep_store_counters()["versions_published"] == published_before + 1
    assert tree_fingerprint(old.store_dir) == files_published


@pytest.mark.parametrize("failure", ["callback_raises", "negative_mobile", "journal_cap"])
def test_late_failure_leaves_the_input_bed_and_every_store_file_unchanged(windowed_bed, failure):
    from maple.surface.voxels.deep_store import read_deep_store_counters
    from maple.water import WaterProcessDemand

    bed, ctx = windowed_bed
    request = _request(bed, 0.05)
    before = physical_arrays(bed)
    store = bed.voxel_column.deep_store
    files = tree_fingerprint(store.store_dir)
    versions = read_deep_store_counters()["versions_published"]
    pool0 = host(bed.water.mobile_mass_by_cell_class_kg).copy()

    if failure == "callback_raises":
        def transport(view):
            raise RuntimeError("transport rejected the step after the pickup")
        error, local_ctx = RuntimeError, ctx
    elif failure == "negative_mobile":
        def transport(view):
            bad = pool0 + host(view.actual_removal_by_cell_class_kg)
            bad[0, 0, 0] = -1.0
            return bad, WaterProcessDemand(np.zeros_like(bad), np.zeros_like(bad))
        error, local_ctx = (ValueError, TypeError), ctx
    else:  # the deposit spills more records than the explicit cap allows, AFTER a successful pickup phase
        transport = make_transport(pool0, 1.0)[0]
        error, local_ctx = ValueError, dataclasses.replace(ctx, window_max_journal_records=1)

    with pytest.raises(error):
        pickup_transport_deposit(bed, local_ctx, bed.water.depth_m, request, transport)

    assert_exact_equal(physical_arrays(bed), before, "input bed after a late failure")
    assert bed.voxel_column.deep_store is store and bed.voxel_column.pending_deep is None
    assert tree_fingerprint(store.store_dir) == files
    assert read_deep_store_counters()["versions_published"] == versions
