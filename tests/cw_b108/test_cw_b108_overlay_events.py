"""CW-B108: full vs exact-overlay deep storage in actual SYRUP complete events on a shared evolving MAPLE window bed.

Same fixture, same physics settings, separate stores; only `BedContext.deep_chunk_encoding` / `deep_overlay_max_bytes`
differ. Within one backend every physical / logical array, packet, inventory, water / sediment accumulator, phase,
counter, commit log and the logical stored identity (logical manifest digest, journal digest, version id, generation)
must be EXACTLY equal; only the physical storage identity (manifest digest / file format) may differ, and is checked to
differ as expected. Conservation uses the existing MAPLE closure bounds of `SedimentEventResult.closure` /
`complete_event.water_budget` unchanged. NumPy always; CuPy only when a device is visible (GPU0 child).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("maple")

from b108_fixture import (
    RESULT_FIELDS_EXACT,
    H,
    assert_same_arrays,
    build,
    dense_bytes,
    logical_identity,
    result_arrays,
    state_arrays,
)

from maple_syrup.checkpoint import load_checkpoint, save_checkpoint
from maple_syrup.complete_event import CompletionPolicy, DryHandoff, EventProgress, complete_event, water_budget
from maple_syrup.sediment_bed import TYPED_JOURNAL_SOURCE_DIGESTS, bound_maple_package_digest

pytestmark = pytest.mark.skipif(
    bound_maple_package_digest() not in TYPED_JOURNAL_SOURCE_DIGESTS,
    reason="overlay events need an explicitly pinned typed MAPLE source (an M94 child claims nothing here)")

#: LABELLED fixture stopping criteria (physical quiet gates, never conservation tolerances): looser hydraulic gates and
#: the explicit CW-B51 flow-end assumption leave a genuine mobile load for the terminal MAPLE settlement and forced commit.
POLICY = CompletionPolicy(hold_s=2.0, max_depth_m=5e-4, surface_volume_m3=1e-2, max_unit_discharge_m2_s=1e-5,
                          outlet_discharge_m3_s=1e-4, deposit_mobile_at_flow_end=True)
MAX_END_S = 300.0
REPORT_S = 2.0
PAUSE_S = 12.0  # an original boundary during rain (wet, mobile load in transit)
IDENTITY = {"case": "cw_b108 12x7x40x3 resident-window valley", "forcing": "20 s of 100 mm/h", "source": "test"}


def store_files(store_dir):
    """Relative path -> sha256 of every file of a store (immutability evidence)."""
    return {p.relative_to(store_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(store_dir.rglob("*")) if p.is_file()}


def run_event(xp, tmp_path, label, encoding, bound=None, *, publication="step", pause=None, **kw):
    store = tmp_path / f"store_{label}"
    s, ctx, control, schedule = build(xp, store, encoding=encoding, overlay_max_bytes=bound,
                                      deep_publication=publication, high_water=10**5 if publication == "boundary"
                                      else None, **kw)
    saved = {}

    def callback(progress):
        if pause is not None and progress.result.state.t_s == pause:
            saved["progress"] = progress
            return True
        return False

    out = complete_event(s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"], s["sediment"],
                         max_end_s=MAX_END_S, control=control, policy=POLICY, report_every_s=REPORT_S,
                         checkpoint_callback=callback)
    return out, s, ctx, control, schedule, store


def compare_handoffs(a, b):
    """Exact within-backend comparison of two dry handoffs (physical + logical); returns the two final stores."""
    ra, rb = a.progress.result, b.progress.result
    for name in RESULT_FIELDS_EXACT:
        assert getattr(ra, name) == getattr(rb, name), name
    assert ra.commit_log == rb.commit_log and ra.rejections == rb.rejections
    assert_same_arrays(result_arrays(ra), result_arrays(rb))
    for sa, sb in ((ra.state, rb.state), (a.dry_state, b.dry_state)):
        assert sa.t_s == sb.t_s
        assert_same_arrays(state_arrays(sa), state_arrays(sb))
        assert logical_identity(sa.bed.voxel_column) == logical_identity(sb.bed.voxel_column)
        assert dense_bytes(sa.bed.voxel_column) == dense_bytes(sb.bed.voxel_column)
    for name in ("pre_reset_surface_m", "pre_reset_soil_m", "terminal_deposition_kg"):
        assert H(getattr(a, name)).tobytes() == H(getattr(b, name)).tobytes(), name
    assert json.dumps(a.accounting, sort_keys=True) == json.dumps(b.accounting, sort_keys=True)
    return a.dry_state.bed.voxel_column.deep_store, b.dry_state.bed.voxel_column.deep_store


def assert_physical_event(out):
    r = out.progress.result
    assert isinstance(out, DryHandoff)
    assert r.closure()["closed"] and r.closure()["request_reconciled"] and water_budget(out.progress)["closed"]
    assert r.n_reroutes >= 1 and r.routing_source_origin == "net_reroute"  # genuine net reroute (default threshold)
    assert r.n_forced_commits >= 1 and float(H(out.terminal_deposition_kg).sum()) > 0.0  # terminal settling + commit
    assert out.dry_state.bed.voxel_column.deep_store.version_id >= 1  # positive deep journals were published
    assert out.dry_state.bed.voxel_column.pending_deep is None


@pytest.mark.parametrize("publication", ["step", "boundary"])
@pytest.mark.parametrize("bound", [None, 1], ids=["overlay_default_bound", "overlay_keyframe_bound_1"])
def test_full_and_overlay_complete_events_are_exact(xp, tmp_path, publication, bound):
    if publication == "boundary" and bound == 1:
        pytest.skip("bounded matrix: the keyframe bound is exercised under the step policy only")
    full = run_event(xp, tmp_path, "full", "full", publication=publication)
    over = run_event(xp, tmp_path, "overlay", "overlay", bound, publication=publication)
    for out in (full[0], over[0]):
        assert_physical_event(out)
    store_full, store_over = compare_handoffs(full[0], over[0])
    # Physical storage identity: v1 chain vs v2 chain with the SAME logical identity, distinct physical manifests.
    assert store_full.format_version == 1 and store_full.manifest_digest == store_full.logical_manifest_digest
    assert store_over.format_version == 2 and store_over.manifest_digest != store_full.manifest_digest
    assert store_over.logical_manifest_digest == store_full.manifest_digest
    if bound is None:
        assert any(store_over.slot_files(i).overlay is not None for i in range(store_over.n_chunks))
    rf, ro = full[0].progress.result, over[0].progress.result
    if publication == "boundary":
        assert rf.deep_publication == ro.deep_publication  # operational record identical between encodings
        assert rf.deep_publication["append_cap_recognizer"] == "typed_journal_capacity_error"
    else:
        assert rf.deep_publication is None and ro.deep_publication is None
    # The overlay version reloads through MAPLE's version abstraction (local integrity + one-level lineage).
    from maple.surface.voxels.deep_store import load_version

    ctx = over[2]
    loaded = load_version(store_over.store_dir, store_over.manifest_digest, geometry=ctx.geometry,
                          mass_resolution_kg=ctx.mass_resolution_kg, check_parent_lineage=True)
    assert loaded.logical_manifest_digest == store_over.logical_manifest_digest and loaded.format_version == 2


@pytest.mark.parametrize("encoding", ["full", "overlay"])
def test_wet_pause_checkpoint_and_fresh_continuation_are_exact(xp, tmp_path, encoding):
    """A genuine wet / mobile pause at an original boundary, a NEW bundle, a fresh load (new objects from disk through
    MAPLE's version abstraction), and the continuation to the dry handoff: exactly the uninterrupted event of the same
    encoding. The saved held routing surface LAGS the committed bed and is restored as is (never the latest bed)."""
    from maple.core.types.topographic_commit import committed_elevation_m

    straight = run_event(xp, tmp_path, "straight", encoding)[0]
    paused, s, ctx, control, schedule, store = run_event(xp, tmp_path, "paused", encoding, pause=PAUSE_S)
    assert isinstance(paused, EventProgress) and paused.result.state.t_s == PAUSE_S
    state = paused.result.state
    assert float(H(state.bed.water.mobile_mass_by_cell_class_kg).sum()) > 0 and float(H(state.storm.depth_m).max()) > 0
    lag = np.abs(H(state.routing_source_elevation_m) - H(committed_elevation_m(state.bed.committed_topography)))
    assert lag.max() > 0.0  # held routing surface lags the committed bed (accounting commits without reroute)
    if encoding == "overlay":
        assert state.bed.voxel_column.deep_store.format_version == 2
    path = tmp_path / "bundle"
    identity = {**IDENTITY, "encoding": encoding}
    save_checkpoint(path, paused, ctx, s["column"], s["sediment"], identity)
    manifest = json.loads((path / "checkpoint.json").read_text())
    if encoding == "full":
        assert "deep_storage" not in manifest  # default bundle metadata unchanged
    else:
        store_v = state.bed.voxel_column.deep_store
        assert manifest["deep_storage"] == {"deep_chunk_encoding": "overlay", "deep_overlay_max_bytes": None,
                                            "store_format_version": 2,
                                            "physical_manifest_digest": store_v.manifest_digest,
                                            "logical_manifest_digest": store_v.logical_manifest_digest}
    before = store_files(store)
    loaded = load_checkpoint(path, ctx, s["column"], s["sediment"], identity)
    assert store_files(store) == before  # loading never writes the store
    lstate = loaded.result.state
    assert lstate is not state
    assert H(lstate.routing_source_elevation_m).tobytes() == H(state.routing_source_elevation_m).tobytes()
    assert_same_arrays(state_arrays(lstate), state_arrays(state))
    assert logical_identity(lstate.bed.voxel_column) == logical_identity(state.bed.voxel_column)
    assert lstate.bed.voxel_column.manifest_digest == state.bed.voxel_column.manifest_digest
    resumed = complete_event(lstate, ctx, s["column"], s["field"], schedule, s["vegetation"], s["sediment"],
                             max_end_s=MAX_END_S, control=control, policy=POLICY, report_every_s=REPORT_S,
                             continuation=loaded)
    assert_physical_event(resumed)
    compare_handoffs(straight, resumed)
    # Every file the bundle references is still present and unchanged (old versions retained, nothing rewritten).
    after = store_files(store)
    assert all(after[k] == v for k, v in before.items())
