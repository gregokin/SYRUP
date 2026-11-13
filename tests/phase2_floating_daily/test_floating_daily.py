"""G3 S1: the genuine floating daily lineage from an initialized soil (CPU/host; NumPy and Numba). Real native beds/owners, real plans, real FSL steps; no fake classes."""
# ruff: noqa: RUF059, C408
from __future__ import annotations

import copy
import dataclasses
import importlib.util  # noqa: F401
import pickle

import builders as b
import numpy as np
import pytest
from dbuild import AVAIL, PART, RUN, close, day, full_day, lineage, make, plan, start

from maple_syrup import floating_daily as fd
from maple_syrup import floating_evaporation as fe
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_transpiration as ft
from maple_syrup import native_floating_soil as nfs
from maple_syrup import recharge as rc

BACKENDS = ["numpy"]
if __import__("importlib").util.find_spec("numba") is not None:
    BACKENDS.append("numba")


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def wbytes(st):
    return np.asarray(st.water.water_m).tobytes()


# ---- start: provenance, generation 0, single authority ---------------------------------------------------------------------------------
def test_start_releases_the_owner_and_is_atomic_on_refusal(backend):
    owner, bed = make(backend=backend)
    reads = []
    with pytest.raises(fd.DailyRefusal):
        fd.start_floating_daily_lineage(owner, bed, partition=PART, run_digest=RUN, available_bytes_fn=lambda: reads.append(1) or 0)  # budget 0
    assert owner.is_current  # nothing consumed
    with pytest.raises(fd.DailyRefusal, match="witness"):
        fd.start_floating_daily_lineage(owner, b.swap_classes(bed, 1), partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)  # another source
    assert owner.is_current
    with pytest.raises(fd.DailyRefusal):
        fd.start_floating_daily_lineage(owner, bed, partition=PART, run_digest="xyz", available_bytes_fn=AVAIL)
    with pytest.raises(fd.DailyRefusal):
        fd.start_floating_daily_lineage(owner, bed, partition={"x": 1}, run_digest=RUN, available_bytes_fn=AVAIL)
    assert owner.is_current
    st = start(owner, bed)
    assert not owner.is_current and st.role == "boundary" and st.cursor.ordinal == 0 and st.t_s == 0.0
    with pytest.raises(fd.DailyRefusal):
        start(owner, bed)  # a released owner cannot start a second lineage
    with pytest.raises(nfs.NativeStaleError):
        nfs.stage_native_refresh(owner, bed, bed, available_bytes_fn=AVAIL)  # nor be staged against its stale W0
    assert st.water.water_m is not None and np.array_equal(st.water.water_m, np.asarray(st.water.water_m))


def test_only_a_fresh_generation_zero_owner_starts_a_lineage(backend):
    owner, bed = make(backend=backend)
    child = b.shift_top(bed, 1, 0.003)
    cand = nfs.stage_native_refresh(owner, bed, child, mode="properties", available_bytes_fn=AVAIL)
    owner1 = nfs.adopt_native_refresh(owner, cand)
    with pytest.raises(fd.DailyRefusal, match="generation-0|superseded|current"):
        start(owner1, child)
    with pytest.raises(fd.DailyRefusal):
        start(owner, bed)  # the superseded generation-0 owner


def test_states_are_minted_uncopyable_and_expected_parent_is_required(backend):
    st, *_ = lineage(backend=backend)
    for bad in (lambda: copy.copy(st), lambda: copy.deepcopy(st), lambda: pickle.dumps(st), lambda: dataclasses.replace(st, revision=9)):
        with pytest.raises(fd.FloatingDailyError):
            bad()
    pl = plan(0.0, 0.0, 5.0, 0)
    n = st.ctx.n_cells
    kw = dict(plan=pl, pet_terrain_m_s=np.zeros(n), cover=np.zeros(n), available_bytes_fn=AVAIL)
    for exp in (None, "x", object()):
        with pytest.raises(fd.DailyRefusal):
            fd.floating_daily_action(st, expected_parent=exp, **kw)
    other, *_ = lineage(backend=backend)
    with pytest.raises(fd.DailyRefusal, match="another instance"):
        fd.floating_daily_action(st, expected_parent=fd.expect_parent(other), **kw)
    with pytest.raises(fd.DailyRefusal):
        fd.floating_daily_action({"plain": 1}, expected_parent=fd.expect_parent(st), **kw)


# ---- the no-rain day -----------------------------------------------------------------------------------------------------------------------
def test_no_rain_day_is_one_evaporation_one_drainage_one_clock_and_exact_history(backend):
    owner, bed = make(backend=backend, saturation=0.6)
    water0 = np.array(owner.state.water_m, copy=True)
    hist0 = (np.array(owner.matrix_history.wetting_m), np.array(owner.matrix_history.cumulative_intake_m))
    n = owner.column.n_cells
    pet, cover = np.full(n, 3e-8), np.full(n, 0.3)
    # independent reference: the qualified E staging + ONE floating step on the owner's own column/state (before the lineage takes the water)
    op = fe.prepare_floating_evaporation(owner.column, PART, available_bytes_fn=AVAIL)
    staged = fe.stage_floating_evaporation(op, pet_terrain_m_s=pet, cover=cover, surface_water_m=np.zeros(n), dt_s=86400.0, available_bytes_fn=AVAIL)
    ref = fsl.floating_step(owner.column, owner.state, 86400.0, added_water_m=np.zeros((3, n)), available_bytes_fn=AVAIL, **staged.native_step_kwargs())
    bed_before = (bed.voxel_column.mass_kg.tobytes(), bed.active_layer.mass_kg.tobytes())
    st = start(owner, bed)
    res = day(st, pet=pet, cover=cover)
    assert res.outcome == "interevent" and res.state.role == "interevent"
    ch = res.state
    assert np.array_equal(ch.water.water_m, ref.state.water_m) and ch.t_s == 86400.0  # ONE day, ONE E/drainage, clock += DAY_S exactly once
    assert ch.cursor.ordinal == 0 and ch.cursor is st.cursor  # the nominal day is counted only at the final close
    assert ch.history is st.history and np.array_equal(ch.history.wetting_m, hist0[0]) and np.array_equal(ch.history.cumulative_intake_m, hist0[1])  # J/I exact (matrix/pref not called)
    assert np.array_equal(st.water.water_m, water0) and st.t_s == 0.0  # the parent is untouched
    assert (bed.voxel_column.mass_kg.tobytes(), bed.active_layer.mass_kg.tobytes()) == bed_before  # the bed is referenced, never changed
    led = ch.ledger
    assert led.last_day_evaporation_m3 > 0.0 and led.last_day_rain_m3 == 0.0 and abs(led.last_residual_m3) <= led.last_bound_m3
    closed = close(ch).state
    assert closed.role == "boundary" and closed.cursor.ordinal == 1 and closed.cursor.last_branch == "no_rain" and closed.t_s == 86400.0


def test_multi_day_water_ledger_closes_and_the_lineage_digest_is_deterministic(backend):
    def run():
        st, *_ = lineage(backend=backend, saturation=0.7)
        s0 = float(st.water.water_m.sum())
        for d in range(4):
            st = full_day(st, p=1.0 if d == 1 else 0.0, i=2.0, t=5.0)
        return st, s0

    a, s0 = run()
    b_, _ = run()
    assert a.execution_digest == b_.execution_digest and a.digest == b_.digest  # deterministic execution-receipt lineage (NOT a content hash)
    area, na = a.ctx.area, int(a.ctx.active.sum())
    led = a.ledger
    after = float(a.water.water_m[:, a.ctx.active].sum()) * area
    assert led.days_applied == 4 and a.cursor.ordinal == 4 and a.t_s == 4 * 86400.0
    before = s0 * 0 + float(np.asarray(make(backend=backend, saturation=0.7)[0].state.water_m)[:, a.ctx.active].sum()) * area
    assert abs((before + led.rain_m3 - led.evaporation_m3 - led.drainage_m3 - led.transpiration_m3) - after) < 1e-9 * max(1.0, abs(before)), (before, led, after)
    assert na == int(a.ctx.active.sum())


# ---- recharge / full-storm classification -----------------------------------------------------------------------------------------------
def test_small_recharge_enters_the_top_layer_then_cascades_and_equality_attempts_recharge(backend):
    st, *_ = lineage(backend=backend, saturation=0.3)
    for p, t in ((2.0, 5.0), (5.0, 5.0)):  # 0 < P < T and the user-chosen equality P == T both attempt recharge
        r = day(st, p=p, i=4.0, t=t)
        assert r.outcome == "interevent" and r.branch == "recharge"
        gain = float(r.state.water.water_m.sum() - st.water.water_m.sum())
        assert gain > 0.0 and r.state.ledger.last_day_rain_m3 == pytest.approx(p / 1000.0 * st.ctx.area * int(st.ctx.active.sum()))
        assert r.state.history is st.history  # a simplified recharge is NOT matrix intake: J/I untouched
    closed = close(r.state).state
    assert closed.cursor.last_branch == "recharge"


def test_recharge_that_does_not_fit_a_closed_column_requires_the_full_storm_from_the_unchanged_parent(backend):
    st, *_ = lineage(backend=backend, saturation=0.99, bottom="closed")
    w0, t0, c0 = wbytes(st), st.t_s, st.cursor
    r = day(st, p=500.0, i=10.0, t=1000.0)  # a pulse far above the closed profile's room
    assert r.outcome == "requires_full_storm" and r.state is None and r.capacity_failure and r.branch == "storm_fallback"
    assert r.forcing.p_mm == 500.0 and r.forcing.t0_s == t0 and r.forcing.duration_s == 3600.0 * 500.0 / 10.0  # FULL P and the native forcing at the unchanged clock
    assert wbytes(st) == w0 and st.t_s == t0 and st.cursor is c0  # nothing applied: no E, no time, no cursor


def test_one_active_closed_column_that_cannot_fit_makes_the_whole_occurrence_fall_back(backend):
    """The SAME pulse on the same soil: a bed whose cells are all deep fits, a bed with one shallow closed cell (little capacity above rock) does not -> the whole occurrence needs the full storm."""
    found = None
    for p in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
        deep, bd, *_ = lineage(backend=backend, saturation=0.9, bottom="closed", bed=b.stratified(shallow=()))
        mixed, bm, *_ = lineage(backend=backend, saturation=0.9, bottom="closed", bed=b.stratified(shallow=((0, 2),)))
        rd, rm = day(deep, p=p, i=4.0, t=50.0), day(mixed, p=p, i=4.0, t=50.0)
        if rd.outcome == "interevent" and rm.outcome == "requires_full_storm":
            found = (p, rm)
            break
    assert found is not None, "fixture must contain a pulse that fits the all-deep bed but not the bed with one shallow closed column"
    p, rm = found
    assert rm.capacity_failure and rm.branch == "storm_fallback" and rm.forcing.p_mm == p and rm.state is None and mixed.t_s == 0.0 and mixed.cursor.ordinal == 0


def test_storm_branch_is_classified_before_any_hydrology(backend):
    st, *_ = lineage(backend=backend)
    r = day(st, p=20.0, i=10.0, t=5.0)
    assert r.outcome == "requires_full_storm" and r.branch == "storm" and not r.capacity_failure and r.state is None
    assert r.forcing.p_mm == 20.0 and r.forcing.amount_mm == pytest.approx(20.0, rel=1e-12)
    assert st.t_s == 0.0 and st.cursor.ordinal == 0 and st.cursor.pending is None


def test_input_validation_precedes_all_work(backend):
    st, *_ = lineage(backend=backend)
    n = st.ctx.n_cells
    for args in ((-1.0, 1.0, 5.0), (1.0, 0.0, 5.0), (float("nan"), 1.0, 5.0), (1.0, 1.0, -1.0), (True, 1.0, 5.0)):
        with pytest.raises(rc.RechargeError):
            plan(*args, 0)
    good = dict(pet=np.full(n, 3e-8), cover=np.full(n, 0.3))
    for bad in ({"pet": np.full(n, -1.0)}, {"pet": np.full(n, np.nan)}, {"cover": np.full(n, 1.5)}, {"cover": np.full(n, -0.1)}):
        with pytest.raises(fd.DailyRefusal):
            day(st, **{**good, **bad})
    for bad in (np.zeros(n + 1), np.zeros(n, dtype=np.float32), list(np.zeros(n)), np.zeros(2 * n)[::2]):
        with pytest.raises(fd.DailyRefusal):
            day(st, pet=bad)
    with pytest.raises(fd.DailyRefusal, match="occurrence"):
        fd.floating_daily_action(st, expected_parent=fd.expect_parent(st), plan=plan(0.0, 0.0, 5.0, 3), pet_terrain_m_s=good["pet"], cover=good["cover"], available_bytes_fn=AVAIL)  # a gap
    with pytest.raises(fd.DailyRefusal, match="occurrence"):
        fd.floating_daily_action(st, expected_parent=fd.expect_parent(st), plan=plan(0.0, 0.0, 5.0, 0, run="cd" * 32), pet_terrain_m_s=good["pet"], cover=good["cover"], available_bytes_fn=AVAIL)
    assert st.t_s == 0.0 and st.cursor.ordinal == 0


def test_unexpected_surface_return_refuses_the_candidate_and_memory_precedes_the_trial(backend, monkeypatch):
    st, *_ = lineage(backend=backend)
    real = fsl.floating_step

    def injected(*a, **k):  # FAULT INJECTION (labelled): the real step, then a positive surface return
        s = real(*a, **k)
        r = np.array(s.surface_return_m)
        r[0] = 1e-6
        return dataclasses.replace(s, surface_return_m=r)

    monkeypatch.setattr(fsl, "floating_step", injected)
    with pytest.raises(fd.DailyRefusal, match="surface"):
        day(st)
    monkeypatch.undo()
    reached = []
    monkeypatch.setattr(fe, "stage_floating_evaporation", lambda *a, **k: reached.append(1))
    n = st.ctx.n_cells
    with pytest.raises(fd.DailyRefusal) as ex:
        fd.floating_daily_action(st, expected_parent=fd.expect_parent(st), plan=plan(0.0, 0.0, 5.0, 0), pet_terrain_m_s=np.zeros(n), cover=np.zeros(n), available_bytes_fn=lambda: 0)
    assert ex.value.code == "memory" and reached == []
    for bad in (-1, True, 1.5, "0"):
        with pytest.raises(fd.DailyRefusal):
            fd.floating_daily_action(st, expected_parent=fd.expect_parent(st), plan=plan(0.0, 0.0, 5.0, 0), pet_terrain_m_s=np.zeros(n), cover=np.zeros(n), available_bytes_fn=AVAIL, reserve_bytes=bad)
    assert reached == []


@pytest.mark.parametrize("layers", [(0.16,), (0.01, 0.05, 0.10)])
@pytest.mark.parametrize("sat", [0.15, 0.6, 0.95])
@pytest.mark.parametrize("bottom", ["open", "closed"])
def test_layers_saturation_and_bottom_variants_close_the_daily_water(backend, layers, sat, bottom):
    st, *_ = lineage(backend=backend, layers=layers, saturation=sat, bottom=bottom)
    r = day(st, p=1.0, i=2.0, t=5.0)
    if r.outcome == "requires_full_storm":  # a legitimate physical outcome: the pulse does not fit this (nearly saturated / shallow) profile -> whole occurrence, unchanged parent
        assert r.capacity_failure and r.state is None and r.forcing.p_mm == 1.0 and st.t_s == 0.0
        return
    led = r.state.ledger
    assert abs(led.last_residual_m3) <= led.last_bound_m3 + 1e-12


# ---- LEAF aggregate withdrawal ---------------------------------------------------------------------------------------------------------------------
def test_sparse_aggregation_is_bounded_and_validated():
    out = fd.aggregate_sparse_demand(3, 4, np.array([0, 0, 2]), np.array([1, 1, 0]), np.array([0.1, 0.2, 0.3]), max_links=3, available_bytes_fn=AVAIL)
    assert out.shape == (3, 4) and out[1, 0] == pytest.approx(0.3) and out[0, 2] == 0.3 and out.sum() == pytest.approx(0.6)
    for args in ((np.array([0]), np.array([5]), np.array([1.0])), (np.array([9]), np.array([0]), np.array([1.0])), (np.array([0]), np.array([0]), np.array([-1.0])),
                 (np.array([0]), np.array([0]), np.array([np.nan])), (np.array([0.5]), np.array([0]), np.array([1.0])), (np.array([0, 1]), np.array([0]), np.array([1.0]))):
        with pytest.raises(fd.DailyRefusal):
            fd.aggregate_sparse_demand(3, 4, *args, max_links=3, available_bytes_fn=AVAIL)
    with pytest.raises(fd.DailyRefusal):
        fd.aggregate_sparse_demand(3, 4, np.zeros(4, dtype=int), np.zeros(4, dtype=int), np.ones(4), max_links=3, available_bytes_fn=AVAIL)  # above the explicit cap


def test_withdrawal_debits_once_with_proportional_scale_and_no_compensation(backend):
    st, *_ = lineage(backend=backend, saturation=0.8)
    inter = day(st).state
    n = inter.ctx.n_cells
    av = ft.floating_layer_availability_m(inter.ctx.column, inter.water)
    c = 1
    req = np.zeros((3, n))
    req[0, c], req[1, c] = 2.0 * av[0, c], 0.25 * av[1, c]  # layer 0 over-requested, layer 1 within its availability (the 25 / 200 example's shape)
    out = fd.request_floating_withdrawal(inter, expected_parent=fd.expect_parent(inter), request_m=req, expected_sequence=0, available_bytes_fn=AVAIL)
    assert out.actual_m[0, c] == pytest.approx(av[0, c]) and out.actual_m[1, c] == pytest.approx(0.25 * av[1, c])  # 25 + 50 style: min(request, available) per layer
    assert out.unmet_m[0, c] == pytest.approx(av[0, c]) and out.scale[0, c] == pytest.approx(0.5) and out.scale[1, c] == 1.0  # no reallocation to another depth
    after = np.asarray(out.state.water.water_m)
    assert np.allclose(after[:, c], np.asarray(inter.water.water_m)[:, c] - out.actual_m[:, c], rtol=0, atol=1e-18)
    assert out.state.t_s == inter.t_s and out.state.request_sequence == 1 and out.state.ledger.transpiration_m3 == pytest.approx(out.actual_m3)
    assert np.array_equal(inter.water.water_m, inter.water.water_m) and inter.request_sequence == 0  # the parent is unchanged
    # a duplicate / stale / skipped sequence never debits twice; a new request applies sequentially from the accepted child
    for seq in (0, 2, True, 1.0):
        with pytest.raises(fd.DailyRefusal, match="sequence"):
            fd.request_floating_withdrawal(out.state, expected_parent=fd.expect_parent(out.state), request_m=req, expected_sequence=seq, available_bytes_fn=AVAIL)
    zero = fd.request_floating_withdrawal(out.state, expected_parent=fd.expect_parent(out.state), request_m=np.zeros((3, n)), expected_sequence=1, available_bytes_fn=AVAIL)  # all-zero: legal, accepts 0
    assert zero.actual_m3 == 0.0 and zero.state.request_sequence == 2 and np.array_equal(zero.state.water.water_m, out.state.water.water_m)
    closed = close(zero.state).state  # the day is final ONLY here, after the actual T
    assert closed.cursor.ordinal == 1 and closed.ledger.transpiration_m3 == pytest.approx(out.actual_m3)
    with pytest.raises(fd.DailyRefusal, match="role"):
        fd.request_floating_withdrawal(closed, expected_parent=fd.expect_parent(closed), request_m=req, expected_sequence=0, available_bytes_fn=AVAIL)  # T only on an interevent


def test_final_close_needs_an_interevent_and_never_double_counts(backend):
    st, *_ = lineage(backend=backend)
    with pytest.raises(fd.DailyRefusal, match="role"):
        close(st)
    inter = day(st).state
    c1, c2 = close(inter).state, close(inter).state  # a retry of the pure close returns an equal-content sibling; the caller's accepted state controls the lineage
    assert c1.cursor.ordinal == c2.cursor.ordinal == 1 and c1.digest == c2.digest and c1 is not c2
    again = day(c1)
    assert again.state.cursor.ordinal == 1  # the next occurrence is ordinal 1: a repeated plan for ordinal 0 is refused
    with pytest.raises(fd.DailyRefusal, match="occurrence"):
        fd.floating_daily_action(c1, expected_parent=fd.expect_parent(c1), plan=plan(0.0, 0.0, 5.0, 0), pet_terrain_m_s=np.zeros(c1.ctx.n_cells), cover=np.zeros(c1.ctx.n_cells), available_bytes_fn=AVAIL)
