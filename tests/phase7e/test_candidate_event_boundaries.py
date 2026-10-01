"""Phase 7e candidate inside the real SYRUP event: commit boundaries drop and rebuild the surface cache, the
frozen-benchmark driver validates it, the actual checkpoint bundle restores a cache-free column with identical
continuation, and all closures hold. Opt-in: skips unless the selected MAPLE is the recorded candidate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maple")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "phase5"))

from maple.surface.voxels import transfer
from phase6.test_checkpoint import IDENTITY, assert_science_equal
from phase6.test_complete_event import fixture as completion_fixture
from test_sediment_event import assert_closed, run, valley_case

from maple_syrup.checkpoint import load_checkpoint, save_checkpoint
from maple_syrup.complete_event import complete_event
from maple_syrup.dependency import resolve_maple_dependency
from maple_syrup.provenance import source_tree_digest
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_event import SedimentEventControl

EXPECTED = (ROOT / "benchmarks/phase7e/candidate_digest.txt").read_text().strip()


@pytest.fixture(scope="module", autouse=True)
def candidate_identity():
    digest = source_tree_digest(resolve_maple_dependency().package_dir).digest_sha256
    if digest != EXPECTED:
        pytest.skip(f"selected MAPLE {digest[:8]} is not the Phase 7e candidate {EXPECTED[:8]}")


def test_commits_invalidate_cache_and_event_closes():
    inputs = valley_case()
    schedule = constant_rainfall(0.0, 40.0, 60.0)
    transfer.reset_selective_statistics()
    end = 50.0
    r = run(inputs, schedule, end, control=SedimentEventControl(commit=True, force_final_commit=True), cadence=10.0)
    assert_closed(r, inputs["state0"])
    assert r.n_commits >= 1
    stats = transfer.selective_statistics
    # A commit publishes a fresh (cache-free) column; the first exchange after it rebuilds the totals once.
    commits_followed_by_exchange = sum(1 for c in r.commit_log if c["t_s"] < end)
    assert stats["cache_rebuilds"] == 1 + commits_followed_by_exchange, (stats["cache_rebuilds"], r.commit_log)
    assert stats["cache_hits"] == 2 * r.n_maple_water_calls - stats["cache_rebuilds"]
    assert r.state.bed.voxel_column.surface_cache is None  # the forced final commit left a fresh column


def test_frozen_event_validates_cache_and_checkpoint_restore_continues_identically(tmp_path):
    from maple_syrup.benchmark_experiment import frozen_control, run_frozen_event

    inputs = valley_case()
    schedule = constant_rainfall(0.0, 30.0, 60.0)
    control = frozen_control()
    transfer.reset_selective_statistics()
    result, checks = run_frozen_event(inputs["state0"], inputs["ctx"], inputs["column"], inputs["field"], schedule,
                                      inputs["vegetation"], inputs["sediment"], 20.0, control, report_every_s=10.0)
    column = result.state.bed.voxel_column
    assert column.surface_cache is not None and transfer.validate_voxel_surface_cache(column)
    assert transfer.selective_statistics["cache_rebuilds"] == 1
    assert checks["maple_end_state"]
    # actual SYRUP checkpoint bundle (MAPLE snapshot) through the wet pause path of complete_event
    args, kw = completion_fixture(wet=True)
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    assert paused.result.state.bed.voxel_column.surface_cache is not None
    path = tmp_path / "checkpoint"
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    restored = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    assert restored.result.state.bed.voxel_column.surface_cache is None  # derived metadata is never persisted
    np.testing.assert_array_equal(restored.result.state.bed.voxel_column.mass_kg, paused.result.state.bed.voxel_column.mass_kg)
    transfer.reset_selective_statistics()
    full = complete_event(*args, **kw)
    rebuilds_full = transfer.selective_statistics["cache_rebuilds"]
    transfer.reset_selective_statistics()
    resumed = complete_event(restored.result.state, *args[1:], **kw, continuation=restored)
    assert transfer.selective_statistics["cache_rebuilds"] >= 1  # the restored column is rebuilt on its first exchange
    assert rebuilds_full >= 1
    assert_science_equal(full, resumed)
