"""CW-B104C1: public routing-source integrity, failure atomicity at an ACTUAL net-driven reroute, and one windowed /
CSR deep-publication 'boundary' complete storm with an own wet restart across a held routing lag -- actual MAPLE beds,
host NumPy always and CuPy when a device is visible (GPU0 only; CPU runs hide CUDA).

A "tampered" source below is a finite, valid-shape source changed by -1 mm in ONE cell while the graph is unchanged:
the graph is then not the graph of its source. Every POSITIVE case uses the saved valid-LAGGED source (the committed
surface differs from it), never a source reconstructed from the latest bed. Equalities are bitwise; the 5e-6 m
threshold of the reroute-failure test is an EXPLICITLY LABELLED small-fixture value (the default MAPLE threshold does
not reroute in this short fixture), every other run uses the actual MAPLE `pending_elevation_trigger_m`.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maple")

from test_coupled_backends import H
from test_cw_b104_restart_windowed import IDENTITY, chain, committed, complete, leaves_bytes, pause_at, spec
from test_windowed_storm import window_inputs

from maple_syrup.checkpoint import _graph_digest, load_checkpoint, save_checkpoint
from maple_syrup.complete_event import CompletionPolicy, DryHandoff, complete_event
from maple_syrup.event_backend import item_to_namespace, state_to_namespace
from maple_syrup.sediment_bed import BedIntegrationError, check_graph_matches_routing_source
from maple_syrup.sediment_event import SedimentEventError

MISMATCH = "not the graph of its held routing source"
FIXTURE_REROUTE_THRESHOLD_M = 5e-6  # LABELLED small-fixture value (reroute-failure test only)


def tampered_source(state):
    src = state.routing_source_elevation_m
    xp = state.graph.xp
    return src + xp.where(xp.arange(src.size).reshape(src.shape) == 1, -1e-3, 0.0)


def with_state(progress, state):
    return replace(progress, result=replace(progress.result, state=state))


def lag_m(state):
    return float(np.abs(H(committed(state.bed)) - H(state.routing_source_elevation_m)).max())


def cupy_or_skip():
    from maple.core.backend import BackendUnavailableError, resolve_backend

    try:
        return resolve_backend("cupy").xp
    except BackendUnavailableError as exc:
        pytest.skip(f"cupy unavailable: {exc}")


def files(store):
    return {p.relative_to(store).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(store).rglob("*")) if p.is_file()}


# --- save_checkpoint refuses an inconsistent source / graph BEFORE writing --------------------------------------------
@pytest.mark.parametrize("where", ["wet_progress", "handoff_progress", "handoff_dry_state"])
def test_save_refuses_a_source_graph_mismatch_before_writing_anything(xp, tmp_path, where):
    s, ctx, control, schedule = chain(xp, spec(commit_interval_s=2.0))
    if where == "wet_progress":
        item = complete(s, ctx, control, schedule, checkpoint_callback=pause_at(2))
        bad = with_state(item, replace(item.result.state, routing_source_elevation_m=tampered_source(item.result.state)))
        state = bad.result.state
    else:
        item = complete(s, ctx, control, schedule)
        assert isinstance(item, DryHandoff)
        if where == "handoff_progress":
            st = item.progress.result.state
            bad = replace(item, progress=with_state(item.progress, replace(st, routing_source_elevation_m=tampered_source(st))))
            state = bad.progress.result.state
        else:
            st = item.dry_state
            bad = replace(item, dry_state=replace(st, routing_source_elevation_m=tampered_source(st)))
            state = bad.dry_state
    with pytest.raises(BedIntegrationError, match=MISMATCH):  # precondition: the mismatch is real
        check_graph_matches_routing_source(state.graph, state.terrain, state.routing_source_elevation_m)
    before = leaves_bytes(bad)
    path = tmp_path / "out" / "bad"
    with pytest.raises(SedimentEventError, match=MISMATCH):
        save_checkpoint(path, bad, ctx, s["column"], s["sediment"], IDENTITY)
    assert not path.exists()
    assert not path.parent.exists() or not list(path.parent.glob(".bad.partial-*"))  # no staging left either
    assert leaves_bytes(bad) == before  # caller inputs unchanged
    # the consistent item still saves (the refusal is the mismatch, not the item)
    save_checkpoint(tmp_path / "good", item, ctx, s["column"], s["sediment"], IDENTITY)


# --- namespace conversion refuses, never repairs ----------------------------------------------------------------------
def test_namespace_conversion_refuses_a_source_graph_mismatch_in_both_directions(tmp_path):
    cp = cupy_or_skip()
    for src_xp, dst_xp in ((np, cp), (cp, np)):
        s, ctx, control, schedule = chain(src_xp, spec(commit_interval_s=2.0))
        paused = complete(s, ctx, control, schedule, checkpoint_callback=pause_at(2))
        state = paused.result.state
        bad = replace(state, routing_source_elevation_m=tampered_source(state))
        before = leaves_bytes(bad)
        binding, digest = bad.graph.input_sha256, _graph_digest(bad.graph)
        with pytest.raises(SedimentEventError, match=MISMATCH):
            state_to_namespace(bad, dst_xp)
        with pytest.raises(SedimentEventError, match=MISMATCH):
            item_to_namespace(with_state(paused, bad), dst_xp)  # the public item path reaches the same refusal
        assert leaves_bytes(bad) == before
        assert bad.graph.input_sha256 == binding and _graph_digest(bad.graph) == digest


def test_valid_lagged_state_converts_both_ways_with_the_same_routing():
    cp = cupy_or_skip()
    s, ctx, control, schedule = chain(np, spec(commit_interval_s=2.0))
    paused = complete(s, ctx, control, schedule, checkpoint_callback=pause_at(3))
    state = paused.result.state
    assert lag_m(state) > 0.0  # the held source lags the committed bed (valid)
    moved = state_to_namespace(state, cp)
    assert moved.graph.xp is cp and moved.graph.input_sha256 == state.graph.input_sha256
    assert _graph_digest(moved.graph) == _graph_digest(state.graph)
    np.testing.assert_array_equal(H(moved.routing_source_elevation_m), H(state.routing_source_elevation_m))
    back = state_to_namespace(moved, np)
    assert leaves_bytes(back) == leaves_bytes(state) and _graph_digest(back.graph) == _graph_digest(state.graph)


# --- a corrupted-on-disk source stays refused on load -----------------------------------------------------------------
def test_corrupted_on_disk_routing_source_is_refused_on_load(xp, tmp_path):
    import json

    s, ctx, control, schedule = chain(xp, spec(commit_interval_s=2.0))
    paused = complete(s, ctx, control, schedule, checkpoint_callback=pause_at(3))
    assert lag_m(paused.result.state) > 0.0
    path = tmp_path / "wet"
    save_checkpoint(path, paused, ctx, s["column"], s["sediment"], IDENTITY)  # a VALID lagged bundle first
    manifest = json.loads((path / "checkpoint.json").read_text())
    names = []

    def find(node):
        if isinstance(node, dict):
            if {"state", "graph_digest", "graph_binding", "routing_source"} <= set(node):
                names.append(node["routing_source"]["array"])
            for value in node.values():
                find(value)
        elif isinstance(node, list):
            for value in node:
                find(value)

    find(manifest["payload"])
    assert len(names) == 1
    with np.load(path / "continuation.npz", allow_pickle=False) as archive:
        arrays = {k: np.array(archive[k]) for k in archive.files}
    arrays[names[0]].reshape(-1)[1] -= 1e-3  # narrow tamper of the saved source only
    np.savez(path / "continuation.npz", **arrays)
    # without the checksum update the payload hash refuses first
    with pytest.raises(SedimentEventError, match="payload hash mismatch: continuation.npz"):
        load_checkpoint(path, ctx, s["column"], s["sediment"], IDENTITY)
    # EXPLICIT test checksum handling: the manifest is re-hashed so the routing check itself is reached
    h = hashlib.sha256((path / "continuation.npz").read_bytes()).hexdigest()
    manifest["files"]["continuation.npz"] = h
    (path / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(SedimentEventError, match="does not reproduce checkpoint routing"):
        load_checkpoint(path, ctx, s["column"], s["sediment"], IDENTITY)


# --- failure atomicity at an ACTUAL net-driven reroute, after the MAPLE commit stage ----------------------------------
@pytest.mark.parametrize("kind", ["graph_build", "cuda_prepare"])
def test_failure_at_an_actual_net_reroute_after_the_maple_commit_publishes_nothing(xp, tmp_path, monkeypatch, kind):
    import maple.surface.topographic_commit.windowed_commit as wc

    import maple_syrup.sediment_bed as sb
    import maple_syrup.sediment_event as se

    def run(sub, observe=None):
        s, ctx, control, schedule = window_inputs(xp, tmp_path / sub, "csr")
        if kind == "cuda_prepare" and not control.cuda:
            pytest.skip("CUDA hydraulics preparation exists only on the CuPy / CUDA path")
        ctx = replace(ctx, topographic_commit_spec=spec(commit_interval_s=2.0,
                                                        pending_elevation_trigger_m=FIXTURE_REROUTE_THRESHOLD_M))
        return s, ctx, control, schedule

    def evolve(s, ctx, control, schedule, hook):
        return se.evolve_sediment_event(s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"],
                                        s["sediment"], 12.0, control, report_every_s=6.0, on_accepted_step=hook)

    # reference: the uninterrupted run and its FIRST actual reroute
    s, ctx, control, schedule = run("ref")
    ref_t = []
    ref = evolve(s, ctx, control, schedule, lambda st: ref_t.append(float(st.t_s)) or False)
    first = next(c for c in ref.commit_log if c["rerouted"])
    assert ref.n_reroutes >= 1 and first["t_s"] in ref_t
    expected_t = ref_t[:ref_t.index(first["t_s"])]

    # failing run: same fixture, instrumented
    s, ctx, control, schedule = run("fail")
    store = tmp_path / "fail" / "store"
    inputs_before = leaves_bytes(s["state0"])
    counts = {"stage": 0, "stage_at_accept": 0, "failed_after_stage": None}
    seen_t, last_files = [], {"files": files(store)}
    stage = wc.stage_windowed_topography

    def staged(*args, **kwargs):  # MAPLE's actual windowed commit stage (counted only)
        out = stage(*args, **kwargs)
        counts["stage"] += 1
        return out

    message = f"B104C1 injected {kind} failure at an actual net reroute (after the MAPLE commit stage)"
    if kind == "graph_build":
        build = sb.routing_graph_from_elevation

        def graph(*args, **kwargs):
            if inspect.stack()[1].function == "commit_bed":  # ONLY the reroute's refreshed-graph build
                counts["failed_after_stage"] = counts["stage"]
                raise RuntimeError(message)
            return build(*args, **kwargs)

        monkeypatch.setattr(sb, "routing_graph_from_elevation", graph)
    else:
        prepare = se.CudaHydraulics.prepare

        def prep(self, graph, column, *, t_s, reason):
            if reason == "commit":  # requested only for a REFRESHED graph
                counts["failed_after_stage"] = counts["stage"]
                raise RuntimeError(message)
            return prepare(self, graph, column, t_s=t_s, reason=reason)

        monkeypatch.setattr(se.CudaHydraulics, "prepare", prep)
    monkeypatch.setattr(wc, "stage_windowed_topography", staged)

    def accepted(st):
        seen_t.append(float(st.t_s))
        counts["stage_at_accept"] = counts["stage"]
        last_files["files"] = files(store)
        return False

    with pytest.raises(RuntimeError, match="B104C1 injected"):
        evolve(s, ctx, control, schedule, accepted)
    # the failure happened inside the reroute's commit, AFTER MAPLE's commit stage ran for that commit
    assert counts["failed_after_stage"] is not None and counts["failed_after_stage"] > counts["stage_at_accept"]
    # the failed candidate was never accepted: exactly the reference's accepted steps before the reroute
    assert seen_t == expected_t
    # caller inputs unchanged; every previously published file unchanged; no new immutable deep version published
    assert leaves_bytes(s["state0"]) == inputs_before
    after = files(store)
    assert all(after.get(k) == v for k, v in last_files["files"].items())
    new = sorted(set(after) - set(last_files["files"]))
    assert not [k for k in new if k.startswith("versions/")], new
    print(f"B104C1 {kind}: failed at t={first['t_s']} after {len(seen_t)} accepted steps; new non-version files "
          f"(content-addressed staged chunks, never referenced by a published version): {new}")


# --- ONE windowed / CSR 'boundary' complete storm + own wet restart across a held routing lag -------------------------
def seeded_window_chain(xp, tmp, monkeypatch):
    """`test_windowed_storm.window_inputs` (3-cell chain, CSR, deep resident window, commits every 2 s), with ITS 32
    buried voxels seeded one ULP under capacity in every cell by B102's `b102_experiment.seed_columns` BEFORE the
    baseline and the window are built -- the same seeding B102's `scen_complete` chain applies -- so actual MAPLE
    deep journals are produced and carried under 'boundary'. The seeding happens in place on `window_inputs`' own
    freshly concatenated dense column at its baseline call (MAPLE's
    `initial_committed_topography_state_from_physical_state`, wrapped for the duration of the build and only for
    that caller), so the baseline, the window and the bed are all built from the seeded column; nothing else changes."""
    import importlib.util

    import maple.surface.topographic_commit.initialization as init
    from maple.surface.voxels.capacity import max_voxel_mass_kg

    path = Path(__file__).resolve().parents[1] / "cw_b102" / "b102_experiment.py"
    module_spec = importlib.util.spec_from_file_location("_b104c1_b102_experiment", path)
    b102 = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(b102)
    original = init.initial_committed_topography_state_from_physical_state
    calls = []

    def seeded(column, active_layer, geometry, *args, **kwargs):
        if inspect.stack()[1].function == "window_inputs":
            mass = column.mass_kg
            assert isinstance(mass, np.ndarray) and mass.flags.writeable and mass.shape[2] > 32
            b102.seed_columns(mass, max_voxel_mass_kg(geometry), np.ones(mass.shape[:2], dtype=bool))
            calls.append(mass.shape)
        return original(column, active_layer, geometry, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(init, "initial_committed_topography_state_from_physical_state", seeded)
        out = window_inputs(xp, tmp, "csr")
    assert len(calls) == 1  # exactly the deep column window_inputs builds was seeded
    return out


def test_windowed_csr_boundary_complete_storm_and_own_wet_restart_across_a_routing_lag(xp, tmp_path, monkeypatch):
    """Default MAPLE threshold; deep_publication 'boundary' with actual carried MAPLE journals. The pause is at the
    first report boundary whose committed surface DIFFERS from the held routing source; the bundle restores that
    source; the resumed run reaches the dry handoff raw-exact (all array leaves, dense deep inventory, accounting,
    commit log and routing counters) to the uninterrupted run of the same backend."""
    from maple.surface.voxels.windowed_inventory import windowed_to_dense

    def fixture(sub):
        s, ctx, control, schedule = seeded_window_chain(xp, tmp_path / sub, monkeypatch)
        return s, ctx, replace(control, deep_publication="boundary", deep_publication_high_water_records=10**5), schedule

    def run(s, ctx, control, schedule, **kw):
        return complete_event(s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"], s["sediment"],
                              max_end_s=180, control=control, policy=CompletionPolicy(hold_s=2), report_every_s=2.0,
                              **kw)

    s, ctx, control, schedule = fixture("ref")
    reference = run(s, ctx, control, schedule)
    assert isinstance(reference, DryHandoff)
    ra = reference.progress.result
    assert ra.closure()["closed"] and ra.n_commits >= 2
    assert ra.deep_publication["max_carried_records"] > 0  # 'boundary' actually carried MAPLE journals

    s2, ctx2, control2, schedule2 = fixture("pause")
    paused = run(s2, ctx2, control2, schedule2, checkpoint_callback=lambda p: lag_m(p.result.state) > 0.0)
    assert not isinstance(paused, DryHandoff)
    state = paused.result.state
    lag = lag_m(state)
    assert lag > 0.0 and state.bed.voxel_column.pending_deep is None  # a published bed at the pause, lagged routing
    path = tmp_path / "wet"
    save_checkpoint(path, paused, ctx2, s2["column"], s2["sediment"], IDENTITY)
    restored = load_checkpoint(path, ctx2, s2["column"], s2["sediment"], IDENTITY)
    rs = restored.result.state
    np.testing.assert_array_equal(H(rs.routing_source_elevation_m), H(state.routing_source_elevation_m))
    assert lag_m(rs) == lag and _graph_digest(rs.graph) == _graph_digest(state.graph)
    assert rs.graph.input_sha256 == state.graph.input_sha256
    resumed = run(dict(s2, state0=rs), ctx2, control2, schedule2, continuation=restored)
    assert isinstance(resumed, DryHandoff)
    rb = resumed.progress.result
    assert ra.commit_log == rb.commit_log
    for name in ("n_accepted_steps", "n_commits", "n_reroutes", "n_net_commits", "n_forced_commits",
                 "n_graph_changes", "n_routing_gate_evaluations", "max_unrouted_net_displacement_m",
                 "routing_source_origin", "routing_source_t_s"):
        assert getattr(ra, name) == getattr(rb, name), name
    assert leaves_bytes(reference.dry_state) == leaves_bytes(resumed.dry_state)
    assert leaves_bytes(reference.progress.result.state) == leaves_bytes(resumed.progress.result.state)
    da = windowed_to_dense(reference.dry_state.bed.voxel_column, max_bytes=10**8)
    db = windowed_to_dense(resumed.dry_state.bed.voxel_column, max_bytes=10**8)
    assert da.mass_kg.tobytes() == db.mass_kg.tobytes()
    assert reference.accounting == resumed.accounting
    print(f"B104C1 windowed boundary restart: pause t={state.t_s} lag={lag!r} m; reference commits={ra.n_commits} "
          f"reroutes={ra.n_reroutes} deep_publication={ra.deep_publication}")


# --- CW-B104C2: the save dedup key includes the terrain reference -----------------------------------------------------
@pytest.mark.parametrize("terrain_kind", ["changed_friction", "equal_content_copy"])
def test_save_checks_a_distinct_dry_state_terrain_sharing_graph_and_source(xp, tmp_path, terrain_kind):
    """A completed handoff whose `dry_state` shares the progress state's graph AND source objects but carries a
    DISTINCT TerrainReference: a finite positive friction change (graph no longer reproducible) is refused before any
    write; an equal-content copy (distinct object, same reconstruction) is still accepted and loads."""
    s, ctx, control, schedule = chain(xp, spec(commit_interval_s=2.0))
    item = complete(s, ctx, control, schedule)
    assert isinstance(item, DryHandoff)
    progress_state, dry = item.progress.result.state, item.dry_state
    assert dry.graph is progress_state.graph and dry.routing_source_elevation_m is progress_state.routing_source_elevation_m
    terrain = dry.terrain
    friction = terrain.friction_factor * 1.1 if terrain_kind == "changed_friction" else terrain.friction_factor.copy()
    other = replace(terrain, friction_factor=friction)
    assert other is not terrain and other is not progress_state.terrain
    variant = replace(item, dry_state=replace(dry, terrain=other))
    path = tmp_path / "out" / "dry"
    if terrain_kind == "changed_friction":
        assert float(np.min(H(friction))) > 0.0 and np.isfinite(H(friction)).all()
        with pytest.raises(BedIntegrationError, match=MISMATCH):  # precondition: the reconstruction differs
            check_graph_matches_routing_source(dry.graph, other, dry.routing_source_elevation_m)
        before = leaves_bytes(variant)
        with pytest.raises(SedimentEventError, match=MISMATCH):
            save_checkpoint(path, variant, ctx, s["column"], s["sediment"], IDENTITY)
        assert not path.exists()
        assert not path.parent.exists() or not list(path.parent.glob(".dry.partial-*"))
        assert leaves_bytes(variant) == before
    else:
        save_checkpoint(path, variant, ctx, s["column"], s["sediment"], IDENTITY)
        restored = load_checkpoint(path, ctx, s["column"], s["sediment"], IDENTITY, allow_complete=True)
        assert isinstance(restored, DryHandoff)
        assert _graph_digest(restored.dry_state.graph) == _graph_digest(dry.graph)
