"""CW-B104: windowed exact-height gate, step / boundary deep publication, own wet restart with a deliberate routing
lag, and checkpoint refusals -- actual MAPLE beds, host NumPy always and CuPy when a device is visible (GPU0 only).

Equalities are bitwise; thresholds are the actual MAPLE `pending_elevation_trigger_m` unless a test states an
EXPLICITLY LABELLED fixture value.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("maple")

from test_coupled_backends import H, setup_xp
from test_sediment_event import chain_elevation, make_bed
from test_windowed_storm import window_inputs

from maple_syrup.checkpoint import (
    _graph_digest,
    load_checkpoint,
    save_checkpoint,
)
from maple_syrup.complete_event import (
    CompletionPolicy,
    DryHandoff,
    complete_event,
)
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_bed import current_routing_surface_m
from maple_syrup.sediment_event import (
    SedimentEventControl,
    SedimentEventError,
    evolve_sediment_event,
)
from maple_syrup.storm import StormControl

IDENTITY = {"case": "cw_b104 controlled actual MAPLE bed", "forcing": "fixed", "source": "test"}


def spec(**kw):
    from maple.core.parameters.topographic_commit import TopographicCommitSpec

    return TopographicCommitSpec(**kw)


def committed(bed):
    from maple.core.types.topographic_commit import committed_elevation_m

    return committed_elevation_m(bed.committed_topography)


def leaves_bytes(value):
    from maple.core.backend import array_leaves

    return [(H(a).dtype.str, H(a).shape, H(a).tobytes()) for a in array_leaves(value)]


# --- windowed exact-height gate ---------------------------------------------------------------------------------------
def seeded_window_inputs(xp, tmp_path):
    """The B102 seeded resident-window valley (64 x 32, one-ULP rooms on a checkerboard: genuine deep journals from the
    first step; commits every 2 s) -- `tests/cw_b102/b102_experiment.build`, unchanged."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "cw_b102" / "b102_experiment.py"
    module_spec = importlib.util.spec_from_file_location("_b104_b102_experiment", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    s, ctx, control, schedule, _, _ = module.build(None, xp, tmp_path)
    return s, ctx, control, schedule


@pytest.mark.parametrize("mode", ["step", "boundary"])
def test_windowed_gate_surface_is_maple_exact_height_on_published_and_staged_states(xp, tmp_path, mode):
    """At every accepted step of a resident-window storm the gate's surface equals (bitwise) MAPLE's committed
    elevation right after a commit, including STAGED states carrying a pending deep journal ('boundary'), and equals
    MAPLE's dense combined-surface diagnostic (with the commit's datum arithmetic) on every published state."""
    from maple.core.backend import to_host_tree
    from maple.surface.active_layer.diagnostics import diagnose_combined_surface_state
    from maple.surface.voxels.windowed_inventory import windowed_to_dense

    s, ctx, control, schedule = seeded_window_inputs(xp, tmp_path)
    if mode == "boundary":
        control = replace(control, deep_publication="boundary", deep_publication_high_water_records=10**5)
    seen = {"steps": 0, "after_commit": 0, "after_commit_staged": 0, "dense_checked": 0}
    last = {"count": int(s["state0"].bed.committed_topography.commit_count)}

    def observe(state):
        bed = state.bed
        surface = H(current_routing_surface_m(bed, ctx))
        seen["steps"] += 1
        count = int(bed.committed_topography.commit_count)
        if count != last["count"]:
            last["count"] = count
            seen["after_commit"] += 1
            seen["after_commit_staged"] += int(bed.voxel_column.pending_deep is not None)
            if not np.any(H(bed.ledger.pending_bed_mass_change_kg)):  # nothing exchanged since the commit
                np.testing.assert_array_equal(surface, H(committed(bed)))
        if bed.voxel_column.pending_deep is None:
            dense = windowed_to_dense(bed.voxel_column, max_bytes=10**8)
            combined = diagnose_combined_surface_state(dense, to_host_tree(bed.active_layer), ctx.geometry,
                                                       ctx.mass_resolution_kg).combined_elevation_m
            ref = H(bed.committed_topography.reference_elevation_m)
            np.testing.assert_array_equal(surface, ref + (combined - ref))
            seen["dense_checked"] += 1
        return False

    hook = {"_staged_accepted_observer": observe} if mode == "boundary" else {"on_accepted_step": observe}
    r = evolve_sediment_event(s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"], s["sediment"],
                              6.0, control, report_every_s=2.0, **hook)
    assert seen["steps"] == r.n_accepted_steps and seen["after_commit"] >= 2
    if mode == "boundary":
        assert seen["after_commit_staged"] >= 1  # a commit on a state carrying a pending MAPLE journal
    else:
        assert seen["dense_checked"] == seen["steps"]
    assert r.closure()["closed"]


@pytest.mark.parametrize("threshold", [None, 5e-6])
def test_step_and_boundary_publication_are_physically_identical_under_net_routing(xp, tmp_path, threshold):
    """'boundary' only defers deep-file publication: commits, reroutes and every physical array are identical to
    'step' (default threshold, and a LABELLED 5e-6 m fixture value under which reroutes happen)."""
    from maple.surface.voxels.windowed_inventory import windowed_to_dense

    results = {}
    for mode in ("step", "boundary"):
        s, ctx, control, schedule = window_inputs(xp, tmp_path / mode, "csr")
        if threshold is not None:
            ctx = replace(ctx, topographic_commit_spec=spec(commit_interval_s=2.0, pending_elevation_trigger_m=threshold))
        if mode == "boundary":
            control = replace(control, deep_publication="boundary", deep_publication_high_water_records=10**5)
        results[mode] = evolve_sediment_event(s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"],
                                              s["sediment"], 12.0, control, report_every_s=6.0)
    a, b = results["step"], results["boundary"]
    for name in ("n_commits", "n_reroutes", "n_net_commits", "n_forced_commits", "n_graph_changes",
                 "n_routing_gate_evaluations", "max_unrouted_net_displacement_m", "routing_source_origin",
                 "routing_source_t_s", "n_accepted_steps"):
        assert getattr(a, name) == getattr(b, name), name
    assert a.commit_log == b.commit_log
    if threshold is not None:
        assert a.n_reroutes >= 1
    for name in ("storm", "sediment_velocity_m_s", "routing_source_elevation_m", "phase"):
        assert leaves_bytes(getattr(a.state, name)) == leaves_bytes(getattr(b.state, name)), name
    for name in ("water", "active_layer", "ledger", "committed_topography", "sediment_availability"):
        assert leaves_bytes(getattr(a.state.bed, name)) == leaves_bytes(getattr(b.state.bed, name)), name
    da = windowed_to_dense(a.state.bed.voxel_column, max_bytes=10**8)
    db = windowed_to_dense(b.state.bed.voxel_column, max_bytes=10**8)
    assert da.mass_kg.tobytes() == db.mass_kg.tobytes()
    assert _graph_digest(a.state.graph) == _graph_digest(b.state.graph)


# --- own wet restart with a deliberate routing lag ---------------------------------------------------------------------
def chain(xp, commit_spec):
    bed, ctx = make_bed(chain_elevation(3), depth_m=.001, commit_spec=commit_spec)
    control = SedimentEventControl(storm=StormControl(max_dt_s=1, max_steps=10000))
    s = setup_xp(bed, ctx, xp, ksat=1e-5, control=control)
    return s, s["ctx"], control, constant_rainfall(0, 10, 100)


def complete(s, ctx, control, schedule, **kw):
    return complete_event(s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"], s["sediment"],
                          max_end_s=180, control=control, policy=CompletionPolicy(hold_s=2), report_every_s=2., **kw)


def pause_at(boundary_index):
    calls = {"n": 0}

    def callback(progress):
        calls["n"] += 1
        return calls["n"] == boundary_index

    return callback


@pytest.mark.parametrize("threshold", [None, 4e-6])
def test_own_wet_restart_restores_the_held_routing_source_and_continues_exactly(xp, tmp_path, threshold):
    """Pause where the latest committed surface DIFFERS from the held routing source; the bundle restores that
    source (not the latest bed), the graph of the source with the saved digest, the saved discharge; nothing is
    refreshed on load; the continuation is raw-exact to the uninterrupted run, with the same later commits /
    reroutes. Default threshold, and a LABELLED 4e-6 m fixture value with reroutes before and after the pause."""
    commit_spec = spec(commit_interval_s=2.0) if threshold is None else spec(commit_interval_s=2.0,
                                                                             pending_elevation_trigger_m=threshold)
    s, ctx, control, schedule = chain(xp, commit_spec)
    paused = complete(s, ctx, control, schedule, checkpoint_callback=pause_at(3))
    state = paused.result.state
    lag = float(np.abs(H(committed(state.bed)) - H(state.routing_source_elevation_m)).max())
    assert lag > 0.0  # the deliberate geometry lag exists at the checkpoint
    path = tmp_path / "wet"
    save_checkpoint(path, paused, ctx, s["column"], s["sediment"], IDENTITY)
    restored = load_checkpoint(path, ctx, s["column"], s["sediment"], IDENTITY)
    rs = restored.result.state
    np.testing.assert_array_equal(H(rs.routing_source_elevation_m), H(state.routing_source_elevation_m))
    assert _graph_digest(rs.graph) == _graph_digest(state.graph) and rs.graph.input_sha256 == state.graph.input_sha256
    np.testing.assert_array_equal(H(rs.storm.discharge_m2_s), H(state.storm.discharge_m2_s))
    assert float(np.abs(H(committed(rs.bed)) - H(rs.routing_source_elevation_m)).max()) == lag
    for name in ("n_reroutes", "n_net_commits", "n_routing_gate_evaluations", "max_unrouted_net_displacement_m",
                 "routing_source_origin", "routing_source_t_s"):
        assert getattr(restored.result, name) == getattr(paused.result, name), name
    reference = complete(s, ctx, control, schedule)
    resumed = complete(dict(s, state0=rs), ctx, control, schedule, continuation=restored)
    assert isinstance(reference, DryHandoff) and isinstance(resumed, DryHandoff)
    ra, rb = reference.progress.result, resumed.progress.result
    assert ra.commit_log == rb.commit_log
    for name in ("n_accepted_steps", "n_commits", "n_reroutes", "n_net_commits", "n_forced_commits",
                 "n_graph_changes", "n_routing_gate_evaluations", "max_unrouted_net_displacement_m",
                 "routing_source_origin", "routing_source_t_s"):
        assert getattr(ra, name) == getattr(rb, name), name
    assert leaves_bytes(reference.dry_state) == leaves_bytes(resumed.dry_state)
    assert reference.accounting == resumed.accounting
    if threshold is not None:
        t_pause = state.t_s
        assert any(c["rerouted"] and c["t_s"] <= t_pause for c in ra.commit_log)
        assert any(c["rerouted"] and c["t_s"] > t_pause for c in ra.commit_log)


def test_checkpoint_refuses_a_tampered_or_missing_routing_source(xp, tmp_path):
    s, ctx, control, schedule = chain(xp, spec(commit_interval_s=2.0))
    paused = complete(s, ctx, control, schedule, checkpoint_callback=pause_at(2))
    state = paused.result.state
    # tampered: a different (valid-shape, finite) source that builds another graph
    src = state.routing_source_elevation_m
    xp_ = state.graph.xp
    tampered_src = src + xp_.where(xp_.arange(src.size).reshape(src.shape) == 1, -1e-3, 0.0)
    tampered = replace(paused, result=replace(paused.result, state=replace(state, routing_source_elevation_m=tampered_src)))
    # CW-B104C1: refused BEFORE anything is written (B104 published it and refused only on load); the corrupted-on-disk
    # load refusal is `test_cw_b104c1_integrity.test_corrupted_on_disk_routing_source_is_refused_on_load`.
    with pytest.raises(SedimentEventError, match="not the graph of its held routing source"):
        save_checkpoint(tmp_path / "tampered", tampered, ctx, s["column"], s["sediment"], IDENTITY)
    assert not (tmp_path / "tampered").exists()
    # missing: a bundle whose state node lacks the routing source (as written before net-elevation routing)
    good = tmp_path / "good"
    save_checkpoint(good, paused, ctx, s["column"], s["sediment"], IDENTITY)
    manifest = json.loads((good / "checkpoint.json").read_text())

    def strip(node):
        if isinstance(node, dict):
            if {"state", "graph_digest", "graph_binding", "routing_source"} <= set(node):
                node.pop("routing_source")
            for value in node.values():
                strip(value)
        elif isinstance(node, list):
            for value in node:
                strip(value)

    strip(manifest["payload"])
    (good / "checkpoint.json").write_text(json.dumps(manifest))
    with pytest.raises(SedimentEventError, match="no saved hydraulic routing source"):
        load_checkpoint(good, ctx, s["column"], s["sediment"], IDENTITY)
