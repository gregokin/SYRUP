"""CW-B108 on the M94 dependency (the active CPU reference's source): the default full path and the pinned M94
message recognizer are unchanged, and the overlay option is refused (source) before any work. Skipped (no claim)
under any other MAPLE source."""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("maple")

from b108_fixture import assert_same_arrays, build, state_arrays

from maple_syrup import sediment_bed
from maple_syrup.sediment_bed import (
    JOURNAL_RECOGNIZER_MAPLE_DIGEST,
    BedIntegrationError,
    bound_maple_package_digest,
    check_deep_storage_source,
    is_carried_append_cap_refusal,
)
from maple_syrup.sediment_event import SedimentEventError, evolve_sediment_event

pytestmark = pytest.mark.skipif(bound_maple_package_digest() != JOURNAL_RECOGNIZER_MAPLE_DIGEST,
                                reason="M94-only compatibility checks")


def files(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_m94_has_no_typed_error_and_keeps_the_pinned_message_path(tmp_path):
    from maple.surface.voxels import deep_store

    assert not hasattr(deep_store, "JournalCapacityError")
    assert sediment_bed.journal_recognizer_source() == "m94_pinned_message"
    with pytest.raises(BedIntegrationError):
        sediment_bed.require_typed_journal_capacity_error()
    s, ctx, control, schedule = build(np, tmp_path / "s", max_journal_records=44)
    good = ValueError("pending journal exceeds max_journal_records=44")
    assert is_carried_append_cap_refusal(good, ctx, 1) and not is_carried_append_cap_refusal(good, ctx, 0)
    assert not is_carried_append_cap_refusal(ValueError("pending journal exceeds max_journal_records=45"), ctx, 1)
    assert not is_carried_append_cap_refusal(RuntimeError(str(good)), ctx, 1)


def test_overlay_is_refused_on_m94_before_any_work(tmp_path):
    check_deep_storage_source("full", None)  # the default needs no source
    with pytest.raises(BedIntegrationError, match="accepted MAPLE source"):
        check_deep_storage_source("overlay", None)
    s, ctx, control, schedule = build(np, tmp_path / "s")
    over = dataclasses.replace(ctx, deep_chunk_encoding="overlay")
    before, store = state_arrays(s["state0"]), files(tmp_path / "s")
    with pytest.raises(SedimentEventError, match="accepted MAPLE source"):
        evolve_sediment_event(s["state0"], over, s["column"], s["field"], schedule, s["vegetation"], s["sediment"],
                              4.0, control, report_every_s=2.0)
    assert files(tmp_path / "s") == store
    assert_same_arrays(state_arrays(s["state0"]), before)


@pytest.mark.parametrize("publication", ["step", "boundary"])
def test_default_full_events_run_on_m94_with_unchanged_records(tmp_path, publication):
    s, ctx, control, schedule = build(np, tmp_path / "s", deep_publication=publication,
                                      high_water=10**5 if publication == "boundary" else None)
    r = evolve_sediment_event(s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"], s["sediment"],
                              30.0, control, report_every_s=2.0)
    assert r.closure()["closed"] and r.state.bed.voxel_column.deep_store.version_id >= 1
    if publication == "boundary":
        assert set(r.deep_publication) == {"policy", "scope", "high_water_records", "journal_cap_records",
                                           "n_publications", "n_boundary_publications", "n_high_water_publications",
                                           "n_return_publications", "n_journal_flushes", "max_carried_records"}
    else:
        assert r.deep_publication is None
    assert json.dumps(sediment_bed.deep_storage_publish_options(ctx)) == "{}"
