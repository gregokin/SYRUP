"""Source lock and concrete API contract: every MAPLE name and keyword SYRUP's windowed adapter imports or passes exists in the
MERGED MAPLE with that exact signature (no import fallback, no fake type, no skip on the merged tree).

The source-pin tests at the end pin the stage-B QUALIFICATION CANDIDATE: SYRUP selects its append-cap recognizer and the
`overlay` encoding by the exact SHA-256 tree digest of the imported `maple` package (`sediment_bed.JOURNAL_RECOGNIZER_SOURCES`);
the two historical digests are retained and the merged `39bcf79e` digest is added explicitly. They must PASS on the merged
bundle and still REFUSE an unknown digest; nothing is skipped, xfailed or downgraded to a capability probe."""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest


def _params(function):
    return inspect.signature(function).parameters


def test_maple_is_the_merged_tree_not_the_old_reference():
    import maple

    parts = Path(maple.__file__).resolve().parts
    assert "reference_context" not in parts, f"maple resolved to the OLD reference runtime: {maple.__file__}"
    # Capabilities that exist only in the merged shared-storage tree, required by SYRUP's windowed event.
    for module in ("maple.water.windowed_transaction", "maple.surface.voxels.windowed_admission",
                   "maple.surface.voxels.windowed_snapshot", "maple.surface.voxels.windowed_pending"):
        importlib.import_module(module)


def test_paired_entry_signature_matches_the_syrup_call():
    from maple.surface.voxels.deep_store import DEFAULT_MAX_JOURNAL_RECORDS
    from maple.surface.voxels.windowed_transfer import DEFAULT_EXCHANGE_GATHER_COLUMNS

    from maple_syrup.sediment_bed import (
        PAIRED_TRANSACTION_ENTRY,
        PAIRED_TRANSACTION_MODULE,
        paired_transaction_entry,
    )

    entry = paired_transaction_entry()
    assert entry is not None, "merged MAPLE must provide the paired transaction (SYRUP refuses to downgrade)"
    assert entry is getattr(importlib.import_module(PAIRED_TRANSACTION_MODULE), PAIRED_TRANSACTION_ENTRY)
    params = _params(entry)
    positional = [n for n, p in params.items() if p.kind is p.POSITIONAL_OR_KEYWORD]
    assert positional[:8] == ["voxel_column", "active_layer", "water", "ledger", "pickup_holder", "geometry",
                              "grain_classes", "mass_resolution_kg"]
    keyword_only = {n for n, p in params.items() if p.kind is p.KEYWORD_ONLY}
    # every keyword SYRUP's `apply_bed_pickup_transport_deposit` passes
    assert {"between", "sediment_availability", "initial_available_fraction", "adapter_name", "detachment_integration",
            "max_journal_records"} <= keyword_only
    # actual defaults the SYRUP adapter inherits (it never passes max_gather_columns; max_journal_records only when set)
    assert params["max_gather_columns"].default == DEFAULT_EXCHANGE_GATHER_COLUMNS == 256
    assert params["max_journal_records"].default == DEFAULT_MAX_JOURNAL_RECORDS == 1_000_000
    assert params["trust_window_cache"].default is False


def test_public_step_and_publication_signatures_match_the_syrup_calls():
    from maple.surface.voxels.windowed_pending import publish_pending_column
    from maple.water.windowed_step import (
        apply_windowed_water_process_demand,
        publish_windowed_result,
    )

    step = _params(apply_windowed_water_process_demand)
    assert {"sediment_availability", "initial_available_fraction", "adapter_name", "detachment_integration",
            "max_journal_records", "max_gather_columns", "trust_window_cache"} <= set(step)
    assert "process_count" not in step and "optimization" not in step  # old-reference-only spellings
    publish = _params(publish_pending_column)
    assert {"max_records", "encoding", "overlay_max_bytes"} <= {n for n, p in publish.items() if p.kind is p.KEYWORD_ONLY}
    assert {"max_records", "encoding", "overlay_max_bytes"} <= set(_params(publish_windowed_result))


def test_staged_commit_surface_and_callback_signatures_match_the_syrup_commit():
    from maple.surface.topographic_commit import windowed_commit
    from maple.water.windowed_commit_callback import (
        make_windowed_water_depth_commit_callback,
    )

    inner = _params(windowed_commit._commit_windowed_topography)
    assert {"avalanche_callback", "water_callback", "trusted_packet", "record_exchange_history", "staged"} <= set(inner)
    assert {"trusted_packet", "depth_update_rule"} <= set(_params(make_windowed_water_depth_commit_callback))
    assert "trusted_packet" in _params(windowed_commit.windowed_committed_surface_elevation_m)
    assert callable(windowed_commit.publish_staged_windowed_commit)
    assert callable(windowed_commit.is_windowed_already_committed)


def test_pickup_view_carries_exactly_the_fields_the_pond_adapter_documents():
    from maple.water.windowed_transaction import PICKUP_VIEW_FIELDS, PickupView

    assert set(PICKUP_VIEW_FIELDS) == {"actual_removal_by_cell_class_kg", "requested_removal_by_class_kg",
                                      "actual_removal_by_class_kg", "availability_shortfall_by_class_kg",
                                      "shortfall_by_class_kg", "numerical_residual_by_class_kg",
                                      "numerical_residual_scalar_kg"}
    # docs/pond_windowed/core.md: the windowed pickup exposes no independent touched-bed change
    assert not hasattr(PickupView, "bed_change_by_class_kg")
    assert "bed_change_by_class_kg" not in {f for f in PickupView.__dataclass_fields__}


def test_syrup_classifier_binds_the_merged_windowed_and_dense_classes(windowed_bed):
    from maple.core.types.voxel import VoxelColumnState, WindowedVoxelColumnState

    from maple_syrup.sediment_bed import is_windowed_bed

    bed, _ = windowed_bed
    assert type(bed.voxel_column) is WindowedVoxelColumnState
    assert is_windowed_bed(bed) is True
    assert VoxelColumnState is not WindowedVoxelColumnState


def test_typed_journal_capacity_error_is_the_object_windowed_transfer_raises():
    """Structural capability SYRUP's typed recognizer needs (independent of any digest): the merged MAPLE provides it."""
    from maple_syrup.sediment_bed import require_typed_journal_capacity_error

    cls = require_typed_journal_capacity_error()
    assert issubclass(cls, ValueError)
    error = cls("pending journal exceeds max_journal_records=5", current_records=4, new_records=3, max_records=5)
    assert (error.current_records, error.new_records, error.max_records) == (4, 3, 5)


MERGED_DIGEST = "af54626ab432670539ed71480fd884f13581dfdfcfec63d7c85053ca215adfec"  # root's exact merged 39bcf79 package


def test_merged_digest_is_an_explicit_pin_and_historical_pins_are_retained():
    from maple_syrup import sediment_bed as sb

    assert sb.MERGED_39BCF79_MAPLE_DIGEST == MERGED_DIGEST
    assert sb.bound_maple_package_digest() == MERGED_DIGEST, "this suite must run on the exact merged bundle"
    assert sb.JOURNAL_RECOGNIZER_SOURCES == {
        "eb14da66e0ffee1e11cd37cc98d3379b01b9e8f54e562ad0d12fc22afb11abd6": "m94_pinned_message",
        "24c6e1478ed732df22f448eddf8cb16c3b72f2406dc9b37a9414cee9a2d1a9ce": "typed_journal_capacity_error",
        MERGED_DIGEST: "typed_journal_capacity_error"}
    assert sb.journal_recognizer_source() == "typed_journal_capacity_error"
    # the overlay allowlist: both typed sources, never the M94 source
    assert sb.TYPED_JOURNAL_SOURCE_DIGESTS == {sb.TYPED_JOURNAL_RECOGNIZER_MAPLE_DIGEST, MERGED_DIGEST}


def test_merged_source_accepts_overlay_options_and_keeps_default_unchanged():
    from maple_syrup.sediment_bed import check_deep_storage_source

    check_deep_storage_source("full", None)
    check_deep_storage_source("overlay", None)
    check_deep_storage_source("overlay", 4096)


def test_unknown_source_digest_is_still_refused_by_name(monkeypatch):
    from maple_syrup import sediment_bed as sb

    monkeypatch.setitem(sb._MAPLE_DIGEST_CACHE, "digest", "0" * 64)
    assert sb.journal_recognizer_source() is None
    with pytest.raises(sb.BedIntegrationError, match="accepted MAPLE source"):
        sb.check_deep_storage_source("overlay", None)
    sb.check_deep_storage_source("full", None)  # the default needs no pinned source


def test_typed_append_cap_recognition_keeps_its_exact_count_and_class_checks(windowed_bed):
    import dataclasses

    from maple.surface.voxels.deep_store import JournalCapacityError

    from maple_syrup.sediment_bed import is_carried_append_cap_refusal

    _, ctx = windowed_bed
    ctx = dataclasses.replace(ctx, window_max_journal_records=5)
    message = "pending journal exceeds max_journal_records=5"
    good = JournalCapacityError(message, current_records=4, new_records=3, max_records=5)
    assert is_carried_append_cap_refusal(good, ctx, 2) is True
    assert is_carried_append_cap_refusal(good, ctx, 0) is False  # nothing carried
    assert is_carried_append_cap_refusal(ValueError(message), ctx, 2) is False  # the M94 spelling is not this source
    wrong_cap = JournalCapacityError(message, current_records=4, new_records=3, max_records=6)
    assert is_carried_append_cap_refusal(wrong_cap, ctx, 2) is False
    overflow = JournalCapacityError(message, current_records=4, new_records=9, max_records=5)
    assert is_carried_append_cap_refusal(overflow, ctx, 2) is False  # the step alone does not fit a clean journal
