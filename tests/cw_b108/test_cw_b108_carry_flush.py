"""CW-B108: source-qualified typed append-cap recognition, the bounded carried-journal flush / same-t-dt redo, and
failure preservation at publication and reroute -- actual MAPLE windowed beds (accepted typed source only).

The actual refusal: the fixture's first accepted step carries 40 deep-journal records and the second appends 5 more;
with a LABELLED fixture cap of 44 records (high-water 42 < cap) MAPLE's `_require_journal_room` raises its typed
`JournalCapacityError` while the previous accepted state carries 40 records: one flush, the SAME attempt redone, and
every physical result exactly equal to an uncapped run. The cap / high-water values are fixture settings for this test,
not production defaults (the production cap is unchanged).
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("maple")

from b108_fixture import (
    RESULT_FIELDS_EXACT,
    assert_same_arrays,
    build,
    dense_bytes,
    logical_identity,
    result_arrays,
    state_arrays,
)

from maple_syrup import sediment_bed, sediment_event
from maple_syrup.sediment_bed import (
    TYPED_JOURNAL_RECOGNIZER_MAPLE_DIGEST,
    BedIntegrationError,
    bound_maple_package_digest,
    is_carried_append_cap_refusal,
    pending_journal_records,
)
from maple_syrup.sediment_event import SedimentEventError, evolve_sediment_event

pytestmark = pytest.mark.skipif(bound_maple_package_digest() != TYPED_JOURNAL_RECOGNIZER_MAPLE_DIGEST,
                                reason="typed recognizer tests need the accepted typed MAPLE source")

CELLS = ((10, 3), (10, 1), (10, 5), (7, 2), (7, 4))  # step 1 carries 40 records, step 2 appends 5
CAP, HIGH_WATER = 44, 42  # LABELLED fixture values
END_S, REPORT_S = 60.0, 30.0


def store_files(store_dir):
    return {p.relative_to(store_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(store_dir.rglob("*")) if p.is_file()}


def inputs(xp, store, encoding="overlay", *, cap=None, high_water=10**5):
    return build(xp, store, encoding=encoding, mobile_kg=0.3, mobile_cells=CELLS, deep_publication="boundary",
                 high_water=high_water, max_journal_records=cap)


def evolve(s, ctx, control, schedule, **kw):
    return evolve_sediment_event(s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"], s["sediment"],
                                 END_S, control, report_every_s=REPORT_S, **kw)


def assert_same_physics(a, b):
    for name in RESULT_FIELDS_EXACT:
        assert getattr(a, name) == getattr(b, name), name
    assert a.commit_log == b.commit_log and a.rejections == b.rejections
    assert_same_arrays(result_arrays(a), result_arrays(b))
    assert_same_arrays(state_arrays(a.state), state_arrays(b.state))
    assert dense_bytes(a.state.bed.voxel_column) == dense_bytes(b.state.bed.voxel_column)


def typed(message=None, *, current, new, maximum):
    from maple.surface.voxels.deep_store import JournalCapacityError

    message = f"pending journal exceeds max_journal_records={maximum}" if message is None else message
    return JournalCapacityError(message, current_records=current, new_records=new, max_records=maximum)


# --- recognizer contract ----------------------------------------------------------------------------------------------
def test_recognizer_accepts_only_the_validated_typed_refusal(tmp_path):
    from maple.surface.voxels.deep_store import JournalCapacityError

    s, ctx, control, schedule = inputs(np, tmp_path / "s", cap=CAP)
    assert sediment_bed.journal_recognizer_source() == "typed_journal_capacity_error"
    good = typed(current=40, new=5, maximum=CAP)  # 45 > 44; the step's own 5 records fit after a flush
    assert is_carried_append_cap_refusal(good, ctx, 40)
    assert is_carried_append_cap_refusal(typed(current=43, new=2, maximum=CAP), ctx, 40)  # mid-attempt append
    assert not is_carried_append_cap_refusal(good, ctx, 0)  # nothing carried
    assert not is_carried_append_cap_refusal(good, ctx, True)  # a bool is not a record count

    class Sub(JournalCapacityError):
        pass

    refused = [
        ValueError(str(good)),  # the M94-style untyped refusal on the typed source
        RuntimeError(str(good)),
        Sub(str(good), current_records=40, new_records=5, max_records=CAP),
        typed("pending journal exceeds max_journal_records=45", current=40, new=5, maximum=CAP),  # message / cap
        typed(current=40, new=6, maximum=45),  # another cap than the context's actual cap
        typed(current=40, new=3, maximum=43),
        typed(current=39, new=6, maximum=CAP),  # current below the carried journal
        typed(current=40, new=4, maximum=CAP),  # fits: not a refusal at all
        typed(current=40, new=0, maximum=CAP),
        typed(current=45, new=1, maximum=CAP),  # prior journal already above the cap
        typed(current=40, new=45, maximum=CAP),  # single-step overflow: the step alone exceeds the cap
        typed(current=43, new=42, maximum=CAP),  # 3 own + 42 new > 44: single-step overflow after a flush
    ]
    for exc in refused:
        assert not is_carried_append_cap_refusal(exc, ctx, 40), exc
    for name, value in (("current_records", True), ("new_records", 5.0), ("max_records", None),
                        ("new_records", np.int64(5))):
        bad = typed(current=40, new=5, maximum=CAP)
        object.__setattr__(bad, name, value)
        assert not is_carried_append_cap_refusal(bad, ctx, 40), (name, value)


def test_recognizer_is_bound_to_the_source_digest(tmp_path, monkeypatch):
    s, ctx, control, schedule = inputs(np, tmp_path / "s", cap=CAP, high_water=HIGH_WATER)
    good = typed(current=40, new=5, maximum=CAP)
    m94_style = ValueError(f"pending journal exceeds max_journal_records={CAP}")
    monkeypatch.setitem(sediment_bed._MAPLE_DIGEST_CACHE, "digest", sediment_bed.JOURNAL_RECOGNIZER_MAPLE_DIGEST)
    assert sediment_bed.journal_recognizer_source() == "m94_pinned_message"
    assert is_carried_append_cap_refusal(m94_style, ctx, 40) and not is_carried_append_cap_refusal(good, ctx, 40)
    monkeypatch.setitem(sediment_bed._MAPLE_DIGEST_CACHE, "digest", "0" * 64)  # an unrecognized source
    assert sediment_bed.journal_recognizer_source() is None
    assert not is_carried_append_cap_refusal(good, ctx, 40) and not is_carried_append_cap_refusal(m94_style, ctx, 40)
    with pytest.raises(SedimentEventError, match="recognizes only"):
        evolve(s, ctx, control, schedule)


def test_typed_source_startup_requires_the_exact_class(tmp_path, monkeypatch):
    from maple.surface.voxels import windowed_transfer

    s, ctx, control, schedule = inputs(np, tmp_path / "s")
    monkeypatch.setattr(windowed_transfer, "JournalCapacityError", type("Other", (ValueError,), {}))
    with pytest.raises(SedimentEventError, match="typed JournalCapacityError"):
        evolve(s, ctx, control, schedule)


# --- actual MAPLE refusal: one flush, same attempt redone ---------------------------------------------------------------
@pytest.mark.parametrize("encoding", ["full", "overlay"])
def test_actual_typed_refusal_flushes_once_and_redoes_the_same_step(xp, tmp_path, encoding):
    from maple.core.backend import is_array

    s, ctx, control, schedule = inputs(xp, tmp_path / "capped", encoding, cap=CAP, high_water=HIGH_WATER)
    before = state_arrays(s["state0"])
    seen = []
    capped = evolve(s, ctx, control, schedule,
                    _staged_accepted_observer=lambda st: seen.append(pending_journal_records(st.bed)) or False)
    pub = capped.deep_publication
    assert pub["n_journal_flushes"] == 1 and pub["append_cap_recognizer"] == "typed_journal_capacity_error"
    assert pub["journal_cap_records"] == CAP and pub["max_carried_records"] <= CAP
    assert seen[:2] == [40, 5]  # the flushed 40 records were published before the redo of step 2
    assert_same_arrays(state_arrays(s["state0"]), before)  # the caller's state is untouched
    r, ctx_r, control_r, schedule_r = inputs(xp, tmp_path / "uncapped", encoding)
    reference = evolve(r, ctx_r, control_r, schedule_r)
    assert reference.deep_publication["n_journal_flushes"] == 0
    assert_same_physics(capped, reference)
    assert capped.n_rejected_attempts == reference.n_rejected_attempts == 0
    assert all(is_array(v) for v in capped.by_class.values())


def test_flush_redo_is_encoding_independent_in_logical_identity(tmp_path):
    runs = {}
    for encoding in ("full", "overlay"):
        s, ctx, control, schedule = inputs(np, tmp_path / encoding, encoding, cap=CAP, high_water=HIGH_WATER)
        runs[encoding] = evolve(s, ctx, control, schedule)
    a, b = runs["full"], runs["overlay"]
    assert_same_physics(a, b)
    assert a.deep_publication == b.deep_publication
    assert logical_identity(a.state.bed.voxel_column) == logical_identity(b.state.bed.voxel_column)
    assert b.state.bed.voxel_column.deep_store.format_version == 2


def test_single_step_overflow_with_nothing_carried_propagates(tmp_path):
    from maple.surface.voxels.deep_store import JournalCapacityError

    s, ctx, control, schedule = inputs(np, tmp_path / "s", cap=39, high_water=38)  # step 1 alone needs 40 records
    files = store_files(tmp_path / "s")
    with pytest.raises(JournalCapacityError) as info:
        evolve(s, ctx, control, schedule)
    assert info.value.current_records == 0 or info.value.current_records < 40
    assert store_files(tmp_path / "s") == files


# --- injected refusals in the loop ----------------------------------------------------------------------------------
def inject(monkeypatch, make_exc, *, when_carried=True):
    """Wrap the accounting step: raise `make_exc(carried)` ONCE at the first attempt whose input state carries
    (`when_carried`) / does not carry records; every other call is the actual step."""
    real = sediment_event._sediment_accounting_step
    box = {"raised": None, "calls": 0, "t_dt": []}

    def wrapper(state, *args, **kwargs):
        box["calls"] += 1
        box["t_dt"].append((state.t_s, args[5]))  # (t, dt) of every attempt (positional dt)
        carried = pending_journal_records(state.bed)
        if box["raised"] is None and (carried > 0) == when_carried:
            box["raised"] = make_exc(carried)
            raise box["raised"]
        return real(state, *args, **kwargs)

    monkeypatch.setattr(sediment_event, "_sediment_accounting_step", wrapper)
    return box


def test_injected_valid_typed_refusal_flushes_once_and_redoes_same_t_dt(tmp_path, monkeypatch):
    s, ctx, control, schedule = inputs(np, tmp_path / "a")
    reference = evolve(s, ctx, control, schedule)
    cap = sediment_bed.resolved_journal_cap(ctx)
    box = inject(monkeypatch, lambda carried: typed(current=carried, new=cap - carried + 1, maximum=cap))
    s2, ctx2, control2, schedule2 = inputs(np, tmp_path / "b")
    injected = evolve(s2, ctx2, control2, schedule2)
    assert box["raised"] is not None and injected.deep_publication["n_journal_flushes"] == 1
    k = box["t_dt"].index(next(x for x in box["t_dt"] if x[0] > 0))  # the injected (first carried) attempt
    assert box["t_dt"][k] == box["t_dt"][k + 1]  # the SAME t and dt redone, no halving
    assert injected.n_rejected_attempts == reference.n_rejected_attempts and not injected.rejections
    assert_same_physics(injected, reference)


@pytest.mark.parametrize("variant", ["untyped_m94_message", "subclass", "malformed_counts", "other_cap",
                                     "single_step_overflow", "nothing_carried"])
def test_unrecognized_injected_refusals_propagate_unchanged(tmp_path, monkeypatch, variant):
    from maple.surface.voxels.deep_store import JournalCapacityError

    s, ctx, control, schedule = inputs(np, tmp_path / "s")
    cap = sediment_bed.resolved_journal_cap(ctx)

    class Sub(JournalCapacityError):
        pass

    def make(carried):
        if variant == "untyped_m94_message":
            return ValueError(f"pending journal exceeds max_journal_records={cap}")
        if variant == "subclass":
            return Sub(f"pending journal exceeds max_journal_records={cap}", current_records=carried,
                       new_records=cap - carried + 1, max_records=cap)
        if variant == "malformed_counts":
            exc = typed(current=carried, new=cap - carried + 1, maximum=cap)
            object.__setattr__(exc, "new_records", float(cap - carried + 1))
            return exc
        if variant == "other_cap":
            return typed(current=carried, new=cap, maximum=cap - 1)
        if variant == "single_step_overflow":
            return typed(current=carried, new=cap + 1, maximum=cap)
        return typed(current=0, new=cap + 1, maximum=cap)

    box = inject(monkeypatch, make, when_carried=variant != "nothing_carried")
    before = state_arrays(s["state0"])
    files = store_files(tmp_path / "s")
    with pytest.raises(BaseException) as info:
        evolve(s, ctx, control, schedule)
    assert info.value is box["raised"]  # the very exception, not wrapped, not retried
    assert box["calls"] == len(box["t_dt"]) and box["t_dt"].count(box["t_dt"][-1]) == 1  # no redo of that attempt
    assert_same_arrays(state_arrays(s["state0"]), before)
    assert store_files(tmp_path / "s") == files  # no flush publication happened ('boundary': nothing published yet)


# --- failure at publication / reroute keeps every published and retained file ----------------------------------------
@pytest.mark.parametrize("encoding", ["full", "overlay"])
def test_failure_at_publication_leaves_published_state_intact(tmp_path, monkeypatch, encoding):
    from maple.surface.voxels import windowed_pending

    s, ctx, control, schedule = build(np, tmp_path / "s", encoding=encoding)
    real = windowed_pending.publish_pending_column
    box = {"calls": 0, "published": 0, "files": None}

    def failing(column, *args, **kwargs):
        box["calls"] += 1
        if column.pending_deep is not None:
            box["published"] += 1
            if box["published"] == 2:  # the second actual journal publication fails before writing anything
                box["files"] = store_files(tmp_path / "s")
                raise RuntimeError("injected publication failure")
        return real(column, *args, **kwargs)

    monkeypatch.setattr(windowed_pending, "publish_pending_column", failing)
    before = state_arrays(s["state0"])
    with pytest.raises(RuntimeError, match="injected publication failure"):
        evolve_sediment_event(s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"], s["sediment"],
                              300.0, control, report_every_s=2.0)
    assert box["files"] is not None and store_files(tmp_path / "s") == box["files"]
    assert_same_arrays(state_arrays(s["state0"]), before)
    assert s["state0"].bed.voxel_column.deep_store.version_id == 0


def test_failure_at_reroute_publishes_nothing_for_that_step(tmp_path, monkeypatch):
    s, ctx, control, schedule = build(np, tmp_path / "s", encoding="overlay")
    real = sediment_bed.routing_graph_from_elevation
    box = {"calls": 0, "files": None}

    def failing(reference, elevation, *, xp=np):
        box["calls"] += 1
        if box["calls"] == 2:  # call 1: the entry graph check; call 2: the first genuine net reroute
            box["files"] = store_files(tmp_path / "s")
            raise RuntimeError("injected reroute failure")
        return real(reference, elevation, xp=xp)

    monkeypatch.setattr(sediment_bed, "routing_graph_from_elevation", failing)
    before = state_arrays(s["state0"])
    with pytest.raises(RuntimeError, match="injected reroute failure"):
        evolve_sediment_event(s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"], s["sediment"],
                              300.0, control, report_every_s=2.0)
    assert box["files"] is not None and store_files(tmp_path / "s") == box["files"]
    assert_same_arrays(state_arrays(s["state0"]), before)


def test_boundary_policy_rejects_high_water_at_or_above_the_cap(tmp_path):
    s, ctx, control, schedule = inputs(np, tmp_path / "s", cap=CAP, high_water=CAP)
    with pytest.raises(SedimentEventError, match="below the actual MAPLE journal cap"):
        evolve(s, ctx, control, schedule)
    assert isinstance(BedIntegrationError("x"), ValueError)
    assert replace(ctx, window_max_journal_records=None).window_max_journal_records is None
