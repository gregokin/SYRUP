"""Streaming profile initialisation: equals the manual step sequence, consumes records lazily without retaining them,
releases the previous step's result before the next record (weakref-verified), reports partial progress on failure
(including cumulative overflow and time overflow), and counts explicit intake once. Nothing here was run by its
author; Codex records results."""
from __future__ import annotations

import dataclasses
import weakref

import numpy as np
import pytest
from column_cases import AREA, column, forcing, layers, state_of

from maple_syrup import subsurface
from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.subsurface import (
    CumulativeBudget,
    ForcingRecord,
    ProfileInitialization,
    SubsurfaceStreamError,
    initialize_profile,
    prepare_column,
    step,
    storage_m3,
)


def make_records(col, n):
    rng = np.random.default_rng(11)
    out = []
    for k in range(n):
        add, pet = forcing(col)
        add[:] = rng.uniform(0.0, 0.01, add.shape)
        add[:, ~col.active] = 0.0
        pet[:] = rng.uniform(0.0, 2e-7, pet.shape)
        pet[~col.active] = 0.0
        out.append(ForcingRecord(float(2 ** (k % 3)) * 64.0, add, pet))
    return out


def test_streaming_equals_manual_steps_and_sums_budgets():
    col = column(layers(gravity=True, diffusivity=1e-6, factor=1.2), n_cells=3, active=[True, False, True])
    state = state_of(col, [[0.0625, 0.125], [0.03125, 0.25], [0.1, 0.3]], t_s=10.0)
    records = make_records(col, 5)
    manual = state
    added = ret = drain = evap = unmet = 0.0
    for r in records:
        result = step(col, manual, r.dt_s, added_water_m=r.added_water_m, potential_evaporation_m_s=r.potential_evaporation_m_s)
        manual = result.state
        b = result.budget
        added, ret, drain, evap, unmet = (added + b.added_m3, ret + b.surface_return_m3, drain + b.drainage_m3,
                                          evap + b.evaporation_m3, unmet + b.evaporation_unmet_m3)
    profile = initialize_profile(col, state, iter(records))
    assert profile.state.water_m.tobytes() == manual.water_m.tobytes()
    assert profile.state.t_s == manual.t_s == 10.0 + sum(r.dt_s for r in records)
    c = profile.cumulative
    assert (c.added_m3, c.surface_return_m3, c.drainage_m3, c.evaporation_m3, c.evaporation_unmet_m3) == (
        added, ret, drain, evap, unmet)
    assert c.n_steps == 5 and c.elapsed_s == sum(r.dt_s for r in records)
    assert profile.storage_initial_m3 == storage_m3(col, state)
    assert profile.storage_final_m3 == storage_m3(col, manual)
    assert abs(profile.residual_m3) <= profile.bound_m3


def test_records_are_consumed_lazily_and_not_retained():
    col = column(layers(gravity=True), n_cells=2)
    state = state_of(col, [[0.0625, 0.125]] * 2)
    records = make_records(col, 4)
    consumed = []

    def generator():
        for k, r in enumerate(records):
            consumed.append(k)
            yield r

    profile = initialize_profile(col, state, generator())
    assert consumed == [0, 1, 2, 3]
    assert isinstance(profile, ProfileInitialization)
    assert {f.name for f in dataclasses.fields(profile)} == {"state", "cumulative", "storage_initial_m3",
                                                              "storage_final_m3", "residual_m3", "bound_m3", "n_terms"}
    for f in dataclasses.fields(profile):
        value = getattr(profile, f.name)
        assert not isinstance(value, (list, tuple, dict))
    assert all(not isinstance(getattr(profile.cumulative, f.name), (list, tuple, dict))
               for f in dataclasses.fields(CumulativeBudget))


def test_previous_step_result_is_released_before_the_next_record(monkeypatch):
    """Weakref probe: when the stream calls `step` for record k, the SubsurfaceStep of record k-1 and all of its flux
    arrays must already be dead (only the current state array and the cumulative scalars are carried)."""
    col = column(layers(gravity=True, diffusivity=1e-6), n_cells=3)
    state = state_of(col, [[0.0625, 0.125]] * 3)
    records = make_records(col, 4)
    original = subsurface.step
    refs = []
    alive_at_next_call = []

    def observed(*args, **kwargs):
        if refs:
            alive_at_next_call.append([r() is not None for r in refs[-1]])
        out = original(*args, **kwargs)
        refs.append((weakref.ref(out), weakref.ref(out.diffusion_m), weakref.ref(out.percolation_m),
                     weakref.ref(out.surface_return_m), weakref.ref(out.drainage_m), weakref.ref(out.evaporation_m),
                     weakref.ref(out.evaporation_unmet_m)))
        return out

    monkeypatch.setattr(subsurface, "step", observed)
    profile = subsurface.initialize_profile(col, state, records)
    assert len(refs) == 4 and alive_at_next_call == [[False] * 7] * 3
    # after the stream only the final state array is held by the result; the last step object and its fluxes are dead
    assert all(r() is None for r in refs[-1])
    assert profile.cumulative.n_steps == 4


def test_failure_mid_stream_reports_progress_and_preserves_inputs():
    col = column(layers(gravity=True), n_cells=2)
    state = state_of(col, [[0.0625, 0.125]] * 2)
    records = make_records(col, 6)
    records[3].added_water_m[0, 0] = np.nan
    state_bytes = state.water_m.tobytes()
    manual = state
    for r in records[:3]:
        manual = step(col, manual, r.dt_s, added_water_m=r.added_water_m,
                      potential_evaporation_m_s=r.potential_evaporation_m_s).state
    consumed = []

    def generator():
        for k, r in enumerate(records):
            consumed.append(k)
            yield r

    with pytest.raises(SubsurfaceStreamError) as info:
        initialize_profile(col, state, generator())
    err = info.value
    assert err.completed_steps == 3 and err.cumulative.n_steps == 3
    assert err.last_state.water_m.tobytes() == manual.water_m.tobytes()
    assert consumed == [0, 1, 2, 3]
    assert state.water_m.tobytes() == state_bytes


def test_cumulative_overflow_carries_last_accepted_state_and_prior_budget():
    """Each interval refills layer 0 from its residual to saturation with explicit counted soil intake (0.109375 m,
    dyadic) BEFORE the evaporation stage, so every step sees theta0 = 0.5 >= stress (multiplier 1) and a huge but
    FINITE unmet demand (1e308 m3 per step). The second accumulation overflows: the stream must stop with the state
    and budget accepted after step 1, never a partially published step. No law or tolerance is altered."""
    col = prepare_column(layers(gravity=False), np.ones(2, dtype=bool), 1e7, reserve_bytes=0,
                         available_bytes_fn=lambda: 1 << 40)
    state = state_of(col, [[0.015625, 0.1875]] * 2)  # layer 0 at its residual depth, as after a drying step
    add, pet = forcing(col, add_cells=[[0.109375, 0.0]] * 2, pet=[5e150, 5e150])
    records = [ForcingRecord(1e150, add, pet) for _ in range(3)]
    # the first two steps, run explicitly: both must evaporate the full refill and leave the same huge finite unmet
    first = step(col, state, 1e150, added_water_m=add, potential_evaporation_m_s=pet)
    second = step(col, first.state, 1e150, added_water_m=add, potential_evaporation_m_s=pet)
    for result in (first, second):
        b = result.budget
        assert b.added_m3 == 0.109375 * 2 * 1e7 and b.evaporation_m3 == b.added_m3  # refill fully evaporated
        assert b.evaporation_unmet_m3 == pytest.approx(1e308, rel=1e-12) and np.isfinite(b.evaporation_unmet_m3)
        np.testing.assert_array_equal(result.state.water_m[0], [0.015625, 0.015625])  # back at residual
    assert first.budget.evaporation_unmet_m3 + second.budget.evaporation_unmet_m3 == float("inf")
    state_bytes = state.water_m.tobytes()
    with pytest.raises(SubsurfaceStreamError) as info:
        initialize_profile(col, state, records)
    err = info.value
    assert "evaporation_unmet_m3" in str(err)
    assert err.completed_steps == 1 and err.cumulative.n_steps == 1
    assert err.cumulative.evaporation_unmet_m3 == first.budget.evaporation_unmet_m3
    assert err.cumulative.added_m3 == first.budget.added_m3
    assert err.last_state.water_m.tobytes() == first.state.water_m.tobytes() and err.last_state.t_s == 1e150
    assert state.water_m.tobytes() == state_bytes


def test_time_overflow_in_stream_carries_progress():
    col = column(layers(gravity=False), n_cells=1)
    state = state_of(col, [[0.0625, 0.125]])
    add, pet = forcing(col)
    records = [ForcingRecord(1e308, add, pet), ForcingRecord(1e308, add, pet)]
    with pytest.raises(SubsurfaceStreamError) as info:
        initialize_profile(col, state, records)
    err = info.value
    assert err.completed_steps == 1 and err.last_state.t_s == 1e308 and err.cumulative.elapsed_s == 1e308
    assert err.last_state.water_m.tobytes() == state.water_m.tobytes()


def test_non_record_items_are_refused():
    col = column(layers(), n_cells=1)
    state = state_of(col, [[0.0625, 0.125]])
    add, pet = forcing(col)
    with pytest.raises(SubsurfaceStreamError) as info:
        initialize_profile(col, state, [(1.0, add, pet)])
    assert info.value.completed_steps == 0 and info.value.last_state is state


def test_empty_stream_is_the_identity():
    col = column(layers(), n_cells=1)
    state = state_of(col, [[0.0625, 0.25]])
    profile = initialize_profile(col, state, [])
    assert profile.state is state and profile.cumulative == CumulativeBudget()
    assert profile.residual_m3 == 0.0 and profile.n_terms == 4 * 2 * 1 + 7


def test_explicit_intake_is_counted_once():
    """Dyadic forcing (0.015625 m) so every sum below is exact in FP64."""
    col = column(layers(gravity=False), n_cells=3, active=[True, True, False])
    state = state_of(col, [[0.03125, 0.09375]] * 3)
    add, pet = forcing(col, add_cells=[[0.015625, 0.0], [0.015625, 0.0], [0.0, 0.0]])
    zero_add, zero_pet = forcing(col)
    profile = initialize_profile(col, state, [ForcingRecord(60.0, add, pet), ForcingRecord(60.0, zero_add, zero_pet),
                                              ForcingRecord(60.0, zero_add, zero_pet)])
    np.testing.assert_array_equal(profile.state.water_m, state.water_m + add)
    assert profile.cumulative.added_m3 == 2 * 0.015625 * AREA
    assert profile.storage_final_m3 - profile.storage_initial_m3 == profile.cumulative.added_m3
    assert profile.residual_m3 == 0.0
    assert profile.cumulative.surface_return_m3 == profile.cumulative.drainage_m3 == 0.0


def test_stream_closure_uses_the_maple_bound():
    col = column(layers(gravity=True, diffusivity=1e-6), n_cells=2)
    state = state_of(col, [[0.0625, 0.125]] * 2)
    profile = initialize_profile(col, state, make_records(col, 3))
    assert profile.n_terms == 4 * 2 * 2 * 3 + 7
    c = profile.cumulative
    scale = max(abs(profile.storage_initial_m3), abs(profile.storage_final_m3), c.added_m3, c.surface_return_m3,
                c.drainage_m3, c.evaporation_m3)
    assert profile.bound_m3 == volume_roundoff_bound_m3(profile.n_terms, scale)
