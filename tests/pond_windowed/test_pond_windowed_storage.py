"""Storage / publication contract of the windowed pond event: deferred (boundary) vs per-step publication, storage-only
publication, one-flush capacity retry with the ACTUAL typed MAPLE error, honest refusals, failure atomicity, no deep reads on
the height gate, bounded residency. Nothing here was run by its author (file-only tools); the root executes it.

Journals are REAL native ones: `excavate_cell` removes ~9.5 mm from a cell with native <= 0.5 mm pickups (the window refills
downwards from the deep store) and keeps the mass in that cell's mobile pool; the event's native deposition then slides the
window up and spills to the deep store. Native erosion alone need not produce a journal. Where a test needs a pending journal
it steps the real event until `pending_journal_records > 0` (and fails loudly if that never happens)."""
from dataclasses import replace

import numpy as np
import pytest
from pond_helpers import host, make_event
from windowed_helpers import (
    WINDOW,
    carry_redundant_journal,
    chain,
    control_for,
    excavate_cell,
    heavy_mobile,
    logical_column,
    lower_cell,
    pair,
    physical_bytes,
    store_identity,
)

from maple_syrup.pond_bed import PondBedError, gate_displacement_m, pending_journal_records
from maple_syrup.pond_event import PondEventControl, PondEventError

RATE = 100.0 / 3.6e6  # 100 mm/h in m/s


def counters():
    from maple.surface.voxels.deep_store import read_deep_store_counters

    return read_deep_store_counters()


EXCAVATE = excavate_cell((1, 0), 0.0095)


def pending_state(tmp_path, *, cap=None, high_water=10**5, max_steps=200, **kw):
    """`(event, initial_state, state)` where `state` is an accepted state of the 'boundary' policy that carries a real
    unpublished journal (stepped from the excavated bed until `pending_journal_records > 0`)."""
    options = {} if cap is None else {"window_max_journal_records": cap}
    event, state = pair(_CPU, tmp_path, deep_publication="boundary", high_water=high_water, ctx_options=options,
                        prepare=EXCAVATE, **kw)[2:]
    current = state
    for _ in range(max_steps):
        current = event.step(current, RATE, 0.05).state
        if pending_journal_records(current.bed) > 0:
            return event, state, current
    pytest.fail(f"no pending journal after {max_steps} steps of the excavated bed")


class _Cpu:
    name, hydraulics = "numpy", "numpy"
    xp = np


_CPU = _Cpu()


# --- refusals ----------------------------------------------------------------------------------------
def test_construction_and_bed_refusals(tmp_path):
    d_ev, d0, w_ev, w0 = pair(_CPU, tmp_path)
    zeros = np.zeros(d_ev.shape)
    with pytest.raises(PondBedError, match="windowed"):
        d_ev.initial_state(w0.bed, zeros)  # a dense event never accepts a windowed bed
    with pytest.raises(PondBedError, match="dense"):
        w_ev.initial_state(d0.bed, zeros)  # a windowed event never accepts a dense bed
    with pytest.raises(PondEventError, match="windowed"):
        PondEventControl(deep_publication="boundary", deep_publication_high_water_records=5).validated()
    with pytest.raises(PondEventError, match="high_water"):
        PondEventControl(bed_storage="windowed", deep_publication="boundary").validated()
    with pytest.raises(PondEventError, match="boundary"):
        PondEventControl(bed_storage="windowed", deep_publication_high_water_records=3).validated()
    ctx = replace(w_ev.context, window_max_journal_records=10)
    with pytest.raises(PondBedError, match="journal cap"):
        make_event(w0.bed, ctx, w_ev._initial_topology,
                   control=control_for(_CPU, windowed=True, deep_publication="boundary", high_water=10))
    with pytest.raises(PondBedError, match="windowed"):  # unchanged dense behaviour
        make_event(d0.bed, ctx, d_ev._initial_topology, control=control_for(_CPU, windowed=False))
    assert d_ev.config_sha256 != w_ev.config_sha256


def test_default_bed_check_still_refuses_windowed_for_dense_only_callers(tmp_path):
    from maple_syrup.pond_bed import require_supported_bed

    d_ev, d0, w_ev, w0 = pair(_CPU, tmp_path)
    with pytest.raises(PondBedError, match="windowed"):
        require_supported_bed(w0.bed, w_ev.context)  # e.g. the dense-only pond checkpoint wrapper
    require_supported_bed(w0.bed, w_ev.context, allow_windowed=True)
    with pytest.raises(PondBedError, match="windowed"):  # windowed context options on a dense bed
        require_supported_bed(d0.bed, replace(d_ev.context, window_max_journal_records=5))


# --- deferred vs per-step publication ----------------------------------------------------------------------
def test_boundary_publication_equals_step_publication(be, tmp_path):
    """Both runs use the SAME chunked evolve calls (same snapped dt sequences). The 'step' run is advanced chunk by chunk
    until it has published at least twice (real, phased journals); the boundary run replays the chunks without publishing
    and publishes once at the end."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    kw = dict(mobile=heavy_mobile(), depth=0.02, prepare=EXCAVATE)  # deeper water: deposition is spread over steps
    s_ev, s0 = pair(be, tmp_path / "a", deep_publication="step", **kw)[2:]
    b_ev, b0 = pair(be, tmp_path / "b", deep_publication="boundary", high_water=10**5, **kw)[2:]
    s, ends, n_steps = s0, [], 0
    for k in range(1, 61):
        end = 0.5 * k
        r = s_ev.evolve(s, end)
        s, n_steps = r.state, n_steps + r.n_accepted_steps
        ends.append(end)
        if s.counters.n_storage_publications >= 2:
            break
    assert s.counters.n_storage_publications >= 2, "the step run never published twice within 30 s"
    b, n_b = b0, 0
    for end in ends:
        r = b_ev.evolve(b, end, publish_at_return=False)
        b, n_b = r.state, n_b + r.n_accepted_steps
    assert pending_journal_records(b.bed) > 0  # carried, never published by the boundary policy yet
    b = b_ev.publish_storage(b)
    assert n_b == n_steps > 2
    assert physical_bytes(s) == physical_bytes(b)  # identical operations; only durability differs
    np.testing.assert_array_equal(logical_column(s.bed), logical_column(b.bed))
    assert b.counters.n_storage_publications == 1 < s.counters.n_storage_publications
    assert b.counters.max_carried_records >= 1 and pending_journal_records(b.bed) == 0


def test_high_water_publishes_and_changes_nothing_physical(tmp_path):
    """Two boundary runs from the SAME real pending state (reached by `pending_state`'s identical stepping): a huge mark that
    only publishes at return versus a mark of one record that publishes at the high-water boundary after an accepted step."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ref_ev, _, ref_pending = pending_state(tmp_path / "a")
    hw_ev, _, hw_pending = pending_state(tmp_path / "b", high_water=1)
    assert pending_journal_records(ref_pending.bed) == pending_journal_records(hw_pending.bed) > 0
    end = ref_pending.t_s + 0.5
    ref, hw = ref_ev.evolve(ref_pending, end), hw_ev.evolve(hw_pending, end)
    assert hw.storage["n_high_water_publications"] > 0 and ref.storage["n_high_water_publications"] == 0
    assert physical_bytes(ref.state) == physical_bytes(hw.state)


def test_overlay_encoding_is_logically_equal_to_full(tmp_path):
    from maple_syrup.sediment_bed import TYPED_JOURNAL_RECOGNIZER_MAPLE_DIGEST, bound_maple_package_digest

    if bound_maple_package_digest() != TYPED_JOURNAL_RECOGNIZER_MAPLE_DIGEST:
        pytest.skip("exact overlay publication needs the accepted typed MAPLE source")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    f_ev, f0 = pair(_CPU, tmp_path / "a", prepare=EXCAVATE, deep_publication="boundary", high_water=10**5)[2:]
    o_ev, o0 = pair(_CPU, tmp_path / "b", prepare=EXCAVATE, deep_publication="boundary", high_water=10**5,
                    ctx_options={"deep_chunk_encoding": "overlay"})[2:]
    f, o = f_ev.evolve(f0, 1.0).state, o_ev.evolve(o0, 1.0).state
    assert physical_bytes(f) == physical_bytes(o)
    np.testing.assert_array_equal(logical_column(f.bed), logical_column(o.bed))
    assert o_ev.audit(o)["sediment"]["closes"]


# --- storage-only publication / geometry gate ---------------------------------------------------------------
def test_publish_storage_is_storage_only_and_deterministic(tmp_path):
    event, _, pending = pending_state(tmp_path)
    record_before = store_identity(pending)
    published = event.publish_storage(pending)
    assert published.bed.voxel_column.pending_deep is None and pending.bed.voxel_column.pending_deep is not None
    assert store_identity(published)[0] > record_before[0]
    for name in ("hydraulic", "topology", "held_source_m", "water", "sediment", "quiet", "phase_fraction",
                 "phase_position_m", "sediment_velocity_m_s"):
        assert getattr(published, name) is getattr(pending, name), name  # clocks, ledgers, geometry untouched
    c0 = pending.counters
    assert published.counters == replace(c0, n_storage_publications=c0.n_storage_publications + 1)
    assert event.publish_storage(published) is published  # nothing pending: the same object
    # continuing from the pending and from the published state is the same physics
    a = event.evolve(pending, pending.t_s + 0.5)
    b = event.evolve(published, pending.t_s + 0.5)
    assert physical_bytes(a.state) == physical_bytes(b.state)


@pytest.mark.parametrize("dz,refreshes", [(0.0005, 0), (0.0009, 1)])
def test_only_net_elevation_refreshes_the_geometry(tmp_path, dz, refreshes):
    event, state = pair(_CPU, tmp_path, rain_mm_h=0.0, prepare=lower_cell((1, 0), dz), deep_publication="boundary",
                        high_water=10**5)[2:]
    out = event.step(state, 0.0, 0.05).state
    assert out.counters.n_refreshes == refreshes and out.topology.generation == refreshes
    published = event.publish_storage(out)  # publication / time never refresh
    assert published.topology is out.topology and published.counters.n_refreshes == refreshes
    later = event.evolve(published, 0.3).state
    assert later.counters.n_refreshes == refreshes
    if refreshes:
        # MAPLE's own pending-elevation trigger may commit first (reason 'maple_trigger'); the refresh itself is decided
        # by the actual-surface gate, so either reason accounts for the one commit
        assert out.counters.n_net_elevation_commits + out.counters.n_maple_trigger_commits == 1
        assert out.counters.max_gate_displacement_m >= 0.00075


def test_height_gate_reads_no_deep_columns(tmp_path):
    from maple.surface.voxels.deep_store import reset_deep_store_counters

    event, state = pair(_CPU, tmp_path, mobile=heavy_mobile())[2:]
    rt = event._check_state(state)
    reset_deep_store_counters()
    for _ in range(3):
        gate = gate_displacement_m(state.bed, event.context, state.held_source_m, rt.active)
    c = counters()
    assert c["gather_calls"] == 0 and c["columns_read"] == 0 and c["bytes_read"] == 0 and gate < 1e-12
    (tmp_path / "q").mkdir()
    quiet, qstate = pair(_CPU, tmp_path / "q", rain_mm_h=0.0)[2:]
    reset_deep_store_counters()
    quiet.step(qstate, 0.0, 0.05)  # nothing exchanged: no deep read, no deep write
    c = counters()
    assert c["columns_read"] == 0 and c["versions_published"] == 0


def test_resident_state_is_not_depth_proportional(tmp_path):
    from maple.surface.voxels.windowed_inventory import windowed_storage_bytes

    _, state = pair(_CPU, tmp_path)[2:]
    info = windowed_storage_bytes(state.bed.voxel_column)
    assert info["resident_bytes_total"] < info["logical_dense_bytes"] / 20
    assert state.bed.voxel_column.window_kg.shape[2] == WINDOW


# --- capacity: one flush, honest refusals ---------------------------------------------------------------------
def _typed_error(cap, carried, new=None):
    """The ACTUAL `JournalCapacityError`; by default counts consistent with an attempt started from `carried` records
    whose own records fit the cap (recognisable), or an explicit `new` (e.g. a single-step overflow)."""
    from maple.surface.voxels.deep_store import JournalCapacityError

    return JournalCapacityError(f"pending journal exceeds max_journal_records={cap}", current_records=carried,
                                new_records=cap - carried + 1 if new is None else new, max_records=cap)


def _need_typed_source():
    from maple_syrup.sediment_bed import journal_recognizer_source

    if journal_recognizer_source() != "typed_journal_capacity_error":
        pytest.skip("the typed JournalCapacityError source is not the bound MAPLE")


def test_one_flush_then_same_physical_retry(tmp_path):
    _need_typed_source()
    cap = 10**5
    event, _, pending = pending_state(tmp_path, cap=cap, high_water=cap - 1)
    carried = pending_journal_records(pending.bed)
    reference = event.evolve(pending, pending.t_s + 0.4)
    real_step, calls = event.step, []

    def flaky(state, rate, dt, **kwargs):
        calls.append((state.t_s, rate, dt, kwargs.get("t_end_s"), pending_journal_records(state.bed)))
        if len(calls) == 1:
            raise _typed_error(cap, carried)
        return real_step(state, rate, dt, **kwargs)

    event.step = flaky
    try:
        result = event.evolve(pending, pending.t_s + 0.4)
    finally:
        del event.step
    assert calls[0][:4] == calls[1][:4]  # the SAME physical step: clock, forcing, dt, snap
    assert calls[0][4] == carried and calls[1][4] == 0  # redone from the flushed prior state
    assert result.n_rejected_attempts == 0 and result.n_accepted_steps == reference.n_accepted_steps
    assert result.state.counters.n_journal_flushes == 1 and result.storage["n_flush_publications"] == 1
    assert result.state.t_s == reference.state.t_s and result.state.counters.n_accepted_steps == \
        reference.state.counters.n_accepted_steps  # no duplicated rain / pickup / budget
    assert physical_bytes(result.state) == physical_bytes(reference.state)


def test_genuine_carried_append_cap_is_flushed_once_and_retried(tmp_path):
    """No injected error. HONEST FIXTURE LABEL: the initial valid windowed bed carries a REDUNDANT native pending journal
    (`carry_redundant_journal`: N valid records rewriting already-buried values of cell 0, attached with MAPLE's
    `attach_pending_journal`; a storage fixture, no physical change; it exists before `initial_state`, so the initial bed
    budgets are correct). The cap is chosen from a reference measurement so that the carried N records fit, the next REAL
    wet-deposition spill (m records of its own) fits alone, and N + m exceeds the cap: MAPLE itself raises the typed
    `JournalCapacityError` from the real exchange and the event flushes once and redoes the same step.

    The steps before the spill use `event.step` directly (the boundary policy's high-water mark `cap - 1` would otherwise
    publish the carried state after an earlier accepted `evolve` step and remove the condition under test)."""
    _need_typed_source()
    for name in ("m", "ref", "cap"):
        (tmp_path / name).mkdir()
    dt = 0.05

    def build(directory, n_records, cap=None):
        options = {} if cap is None else {"window_max_journal_records": cap}
        return pair(_CPU, directory, deep_publication="boundary", high_water=10**5 if cap is None else cap - 1,
                    prepare=chain(EXCAVATE, carry_redundant_journal(n_records)), ctx_options=options)[2:]

    # 1. measure m: the records the first real spill appends to a carried journal of one record
    m_ev, cur = build(tmp_path / "m", 1)
    assert pending_journal_records(cur.bed) == 1
    for _ in range(200):
        cur = m_ev.step(cur, RATE, dt).state
        if pending_journal_records(cur.bed) > 1:
            break
    m = pending_journal_records(cur.bed) - 1
    assert m >= 1, "no real spill within 200 steps of the excavated bed"
    # 2. N large enough that the step's own records (application order may repeat keys) fit the cap; N + m - 1 < N + own
    n_carried = 3 * m + 3
    cap = n_carried + m - 1
    # 3. uncapped reference: direct steps up to the spill step k, the spill step itself through the same evolve call
    ref_ev, ref = build(tmp_path / "ref", n_carried)
    assert pending_journal_records(ref.bed) == n_carried
    prior_states = [ref]
    for k in range(1, 201):
        nxt = ref_ev.step(prior_states[-1], RATE, dt).state
        if pending_journal_records(nxt.bed) > n_carried:
            break
        prior_states.append(nxt)
    else:
        pytest.fail("the reference never spilled")
    reference = ref_ev.evolve(prior_states[-1], prior_states[-1].t_s + dt, publish_at_return=False).state
    # 4. capped run: identical direct steps keep the carried journal; MAPLE refuses the spill step's native append
    c_ev, cur = build(tmp_path / "cap", n_carried, cap)
    assert pending_journal_records(cur.bed) == n_carried <= cap
    for _ in range(len(prior_states) - 1):
        cur = c_ev.step(cur, RATE, dt).state
    assert pending_journal_records(cur.bed) == n_carried  # still carried, never auto-published
    with pytest.raises(ValueError) as refusal:  # the public step does not retry: the genuine native refusal
        c_ev.step(cur, RATE, dt)
    assert type(refusal.value).__name__ == "JournalCapacityError"
    flushes_before = cur.counters.n_journal_flushes
    result = c_ev.evolve(cur, cur.t_s + dt, publish_at_return=False)
    assert result.n_rejected_attempts == 0 and result.n_accepted_steps == 1
    assert result.state.counters.n_journal_flushes == flushes_before + 1  # exactly one prior-state flush
    assert result.storage["n_flush_publications"] == 1
    assert physical_bytes(result.state) == physical_bytes(reference)  # exact physics of the clean (uncapped) run


def test_flush_then_genuine_rejection_adapts_normally_without_doubled_physics(tmp_path, monkeypatch):
    """The capacity refusal precedes the transport: after the (dt-preserving) flush the redo meets a REAL
    `HydraulicStepRejected`, which must enter the ordinary bounded dt halving; the flush is not repeated."""
    _need_typed_source()
    import maple_syrup.pond_event as pond_event
    from maple_syrup.experimental_hydrology import HydraulicStepRejected

    cap = 10**5
    event, _, pending = pending_state(tmp_path, cap=cap, high_water=cap - 1)
    carried = pending_journal_records(pending.bed)
    real_step_once = pond_event.step_once

    def one_rejection():
        seen = []

        def wrapped(*args, **kwargs):
            if not seen:
                seen.append(1)
                raise HydraulicStepRejected("injected genuine physical rejection")
            return real_step_once(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(pond_event, "step_once", one_rejection())
    reference = event.evolve(pending, pending.t_s + 0.4)  # rejection only: the ordinary adaptation
    assert reference.n_rejected_attempts == 1
    monkeypatch.setattr(pond_event, "step_once", one_rejection())
    real_step, calls = event.step, []

    def flaky(state, rate, dt, **kwargs):
        calls.append((state.t_s, dt, pending_journal_records(state.bed)))
        if len(calls) == 1:
            raise _typed_error(cap, carried)
        return real_step(state, rate, dt, **kwargs)

    event.step = flaky
    try:
        result = event.evolve(pending, pending.t_s + 0.4)
    finally:
        del event.step
    assert calls[0][0] == calls[1][0] == calls[2][0] and calls[0][1] == calls[1][1]  # flush redo: same clock, SAME dt
    assert calls[0][2] == carried and calls[1][2] == 0
    assert calls[2][1] == calls[1][1] * 0.5  # then the ordinary halving after the genuine rejection
    assert result.n_rejected_attempts == 1 and result.state.counters.n_journal_flushes == 1
    assert result.storage["n_flush_publications"] == 1
    assert result.n_accepted_steps == reference.n_accepted_steps
    assert physical_bytes(result.state) == physical_bytes(reference.state)  # no doubled rain / pickup / budget


def test_no_second_flush_and_no_retry_for_unrelated_refusals(tmp_path):
    _need_typed_source()
    cap = 10**5
    event, _, pending = pending_state(tmp_path, cap=cap, high_water=cap - 1)
    carried = pending_journal_records(pending.bed)
    base = counters()["versions_published"]

    def raiser(error):
        def step(state, rate, dt, **kwargs):
            raise error
        return step

    for error in (ValueError("unrelated refusal"), MemoryError("out of memory"),
                  ValueError(f"pending journal exceeds max_journal_records={cap}"),  # right text, wrong (untyped) class
                  _typed_error(cap, carried, new=cap + 5)):  # typed, but a single-step overflow one flush cannot cure
        event.step = raiser(error)
        try:
            with pytest.raises(type(error)):
                event.evolve(pending, pending.t_s + 0.4)
        finally:
            del event.step
        assert counters()["versions_published"] == base  # no flush happened
    # a recognised refusal is flushed ONCE; a second one from the redo propagates
    event.step = raiser(_typed_error(cap, carried))
    try:
        with pytest.raises(ValueError, match="max_journal_records"):
            event.evolve(pending, pending.t_s + 0.4)
    finally:
        del event.step
    assert counters()["versions_published"] == base + 1
    assert pending.bed.voxel_column.pending_deep is not None  # the caller's state is untouched


def test_a_step_too_large_for_a_clean_journal_fails_honestly(tmp_path):
    event, state = pair(_CPU, tmp_path, mobile=heavy_mobile(), ctx_options={"window_max_journal_records": 1})[2:]
    before, ident, published = physical_bytes(state), store_identity(state), counters()["versions_published"]
    with pytest.raises(ValueError) as caught:
        event.evolve(state, 1.0)
    assert type(caught.value).__name__ == "JournalCapacityError"
    assert physical_bytes(state) == before and store_identity(state) == ident
    assert counters()["versions_published"] == published and state.counters.n_journal_flushes == 0


# --- failure atomicity ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("target", ["native_commit", "pond_transport_step", "update_pond_topology"])
def test_late_failure_leaves_inputs_and_store_unchanged(tmp_path, monkeypatch, target):
    import maple_syrup.pond_event as pond_event

    prepare = lower_cell((1, 0), 0.0009) if target == "update_pond_topology" else None
    event, state = pair(_CPU, tmp_path, mobile=heavy_mobile(), prepare=prepare)[2:]
    before, ident, published = physical_bytes(state), store_identity(state), counters()["versions_published"]
    rows = host(state.bed.voxel_column.window_kg).tobytes()

    def boom(*args, **kwargs):
        raise RuntimeError("late failure")

    monkeypatch.setattr(pond_event, target, boom)
    with pytest.raises(RuntimeError, match="late failure"):
        event.step(state, RATE, 0.05)
    assert physical_bytes(state) == before and store_identity(state) == ident
    assert host(state.bed.voxel_column.window_kg).tobytes() == rows
    assert counters()["versions_published"] == published  # a failed candidate never writes a store version


def test_late_host_bookkeeping_error_publishes_no_speculative_version(tmp_path):
    """Counters supplied through `dataclasses.replace` are unblessed public inputs: a malformed one must fail BEFORE the
    deep-store write of a step ('step' policy) or of `publish_storage` / the flush ('boundary' policy)."""
    event, state = pair(_CPU, tmp_path, mobile=heavy_mobile())[2:]
    bad = replace(state, counters=replace(state.counters, max_carried_records=None))
    ident, written = store_identity(bad), counters()["versions_published"]
    with pytest.raises(TypeError):
        event.step(bad, RATE, 0.05)
    assert store_identity(bad) == ident and counters()["versions_published"] == written
    (tmp_path / "b").mkdir()
    b_event, _, pending = pending_state(tmp_path / "b")
    bad_pending = replace(pending, counters=replace(pending.counters, n_storage_publications=None))
    written = counters()["versions_published"]
    with pytest.raises(TypeError):
        b_event.publish_storage(bad_pending)
    with pytest.raises(TypeError):
        b_event._publish(bad_pending, "flush")
    assert counters()["versions_published"] == written
    assert bad_pending.bed.voxel_column.pending_deep is not None


def test_tiny_changes_are_retained_exactly(tmp_path):
    d_ev, d0, w_ev, w0 = pair(_CPU, tmp_path, prepare=lower_cell((1, 0), 1e-9), mobile=heavy_mobile(),
                              deep_publication="boundary", high_water=10**5)
    np.testing.assert_allclose(host(w0.bed.active_layer.mass_kg), host(d0.bed.active_layer.mass_kg), rtol=1e-10, atol=1e-14)
    d, w = d_ev.evolve(d0, 0.5).state, w_ev.evolve(w0, 0.5).state
    np.testing.assert_allclose(logical_column(w.bed), host(d.bed.voxel_column.mass_kg), rtol=1e-10, atol=1e-14)
