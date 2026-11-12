"""Post-moisture-cap soil evaporation share (`subsurface.step(..., evaporation_share=...)`, user decision
2026-11-10): the user's 1 mm / 0.5 example with the real LEAF withdrawal afterwards (E + T conservation, time
preserved, no double debit), dry / wet / stress / residual-limited / zero-share / one-share cases, the detectable
difference from the superseded before-cap partition, masked and invalid share fields (failure without mutation),
all diffusion/bottom modes, streaming, memory accounting before allocation, NumPy/Numba equivalence (bitwise where
no expm1 enters) and real-GPU cases behind the honest `gpu` skip. Equation checks, not a field validation."""
from __future__ import annotations

import numpy as np
import pytest
from tietjen_cases import (
    AREA,
    BIG,
    EDGES3,
    arrays_of,
    column,
    donor_layers,
    forcing,
    host,
    layers,
    random_case,
    state_of,
)

from maple_syrup import soil_evaporation as se
from maple_syrup import subsurface_transpiration as st
from maple_syrup.subsurface import (
    FLAG_MESSAGES,
    ForcingRecord,
    SubsurfaceError,
    SubsurfaceMemoryError,
    SubsurfaceStepError,
    admit_step,
    column_state,
    estimate_step_memory,
    initialize_profile,
    prepare_column,
    soil_layers,
    step,
    step_memory,
)

RTOL, ATOL = 2e-12, 1e-14


def thin_top_layers(*, gravity=False, diffusivity=0.0):
    """Top layer 0.01 m with zero residual and stress threshold 0.05: w0 = 0.001 m (1 mm) is theta 0.1 >= stress, so
    the moisture multiplier is 1 (the user's stress-free example) and the full 1 mm is withdrawable."""
    return soil_layers([0.0, 0.01, 0.4], theta_residual=0.0, theta_field_capacity=0.2, theta_saturation=0.4,
                       diffusivity_m2_s=diffusivity, theta_evaporation_stress=0.05, evaporation_factor=1.0,
                       gravity_drainage=gravity)


def run(col, state, dt, pet, share, add=None):
    added, rate = forcing(col, add, pet)
    if share is not None and not isinstance(share, np.ndarray):
        share = np.array(share, dtype=np.float64)
    return step(col, state, dt, added_water_m=added, potential_evaporation_m_s=rate, evaporation_share=share)


# --- the user's example and the LEAF withdrawal afterwards ------------------------------------------------------------------
def test_user_example_one_mm_withdrawable_half_share():
    col = column(thin_top_layers(), n_cells=2)
    state = state_of(col, [[0.001, 0.05], [0.001, 0.05]], t_s=100.0)  # 1 mm withdrawable in the top layer
    pet = [2e-3 / 3600.0, 0.2e-3 / 3600.0]  # potential combined 2 mm and 0.2 mm over the hour
    r = run(col, state, 3600.0, pet, [0.5, 0.5])
    np.testing.assert_allclose(r.evaporation_m, [0.5e-3, 0.1e-3], rtol=1e-12)  # min(2,1) x .5 ; min(.2,1) x .5
    np.testing.assert_allclose(r.evaporation_unmet_m, [0.5e-3, 0.0], rtol=1e-12, atol=1e-20)  # 2 x .5 - .5 ; .2 x .5 - .1
    np.testing.assert_allclose(r.state.water_m[0], [0.5e-3, 0.9e-3], rtol=1e-12)  # the (1 - s) part stays for LEAF
    assert r.budget.evaporation_m3 == pytest.approx(0.6e-3 * AREA, rel=1e-12) and r.state.t_s == 3700.0
    # LEAF requests 1 mm from the top layer afterwards against the REMAINING water: real withdrawal, time preserved
    op = st.prepare_transpiration(col)
    req = np.zeros((2, 2))
    req[0] = 0.001
    t = st.withdraw_transpiration(op, r.state, req)
    np.testing.assert_allclose(t.actual_m[0], [0.5e-3, 0.9e-3], rtol=1e-12)
    np.testing.assert_allclose(t.unmet_m[0], [0.5e-3, 0.1e-3], rtol=1e-12)
    assert t.state.t_s == 3700.0
    np.testing.assert_allclose(r.evaporation_m + t.actual_m[0], [1e-3, 1e-3], rtol=1e-12)  # E + T = the 1 mm, once
    np.testing.assert_allclose(t.state.water_m[0], [0.0, 0.0], atol=1e-18)
    assert abs(r.budget.residual_m3) <= r.budget.bound_m3 and abs(t.budget.residual_m3) <= t.budget.bound_m3


def test_combined_step_with_transpiration_passes_the_share_and_closes():
    col = column(thin_top_layers(), n_cells=2)
    op = st.prepare_transpiration(col)
    state = state_of(col, [[0.001, 0.05], [0.001, 0.05]], t_s=10.0)
    add, pet = forcing(col, pet=[2e-3 / 3600.0, 2e-3 / 3600.0])
    req = np.zeros((2, 2))
    req[0] = 0.001
    coupled = st.step_with_transpiration(op, state, 3600.0, added_water_m=add, potential_evaporation_m_s=pet,
                                         requested_transpiration_m=req, evaporation_share=np.array([0.5, 0.0]))
    b = coupled.budget
    assert b.evaporation_m3 == pytest.approx(0.5e-3 * AREA, rel=1e-12)  # share 0 on the second cell: no E
    assert b.transpiration_m3 == pytest.approx((0.5e-3 + 1e-3) * AREA, rel=1e-12)  # LEAF takes what remains
    assert b.storage_after_m3 - b.storage_before_m3 == pytest.approx(-(b.evaporation_m3 + b.transpiration_m3), abs=b.bound_m3)
    assert coupled.state.t_s == 3610.0 and abs(b.residual_m3) <= b.bound_m3
    assert coupled.memory.hydro == estimate_step_memory(2, 2, backend="numpy", with_evaporation_share=True)


# --- regimes ---------------------------------------------------------------------------------------------------------
def test_dry_at_or_below_residual_removes_nothing_and_reports_partitioned_unmet():
    col = column(layers(gravity=False), n_cells=2)
    state = state_of(col, [[0.015625, 0.1875], [0.0078125, 0.1875]])  # top at / below residual
    r = run(col, state, 1000.0, [1e-6, 1e-6], [0.5, 0.5])
    np.testing.assert_array_equal(r.evaporation_m, [0.0, 0.0])
    np.testing.assert_array_equal(r.evaporation_unmet_m, [0.0, 0.0])  # multiplier 0: demand 0, so demand x s - 0 = 0
    assert r.state.water_m.tobytes() == state.water_m.tobytes()


def test_stress_regime_applies_the_multiplier_to_the_combined_loss_before_the_cap():
    col = column(layers(gravity=False, factor=1.0), n_cells=1)
    state = state_of(col, [[0.03125, 0.1875]])  # theta0 0.125: multiplier (0.125 / 0.25)^2 = 0.25
    r = run(col, state, 1000.0, [1e-6], [0.5])
    demand = (1e-6 * 1000.0) * 0.25  # combined potential loss after stress, well below the 0.015625 available
    assert r.evaporation_m[0] == demand * 0.5
    assert r.evaporation_unmet_m[0] == demand * 0.5 - demand * 0.5 == 0.0
    assert r.state.water_m[0, 0] == 0.03125 - demand * 0.5


def test_residual_limited_wet_case_caps_the_combined_loss_then_shares():
    col = column(layers(gravity=False, factor=1.0), n_cells=1)
    state = state_of(col, [[0.125, 0.1875]])  # theta0 0.5 >= stress; 0.109375 above residual
    r = run(col, state, 1000.0, [1.0], [0.25])  # demand 1000 m >> available
    assert r.evaporation_m[0] == 0.109375 * 0.25
    assert r.evaporation_unmet_m[0] == 1000.0 * 0.25 - 0.109375 * 0.25  # partitioned potential minus actual
    assert r.state.water_m[0, 0] == pytest.approx(0.125 - 0.109375 * 0.25, rel=1e-15)
    assert r.state.water_m[0, 0] - 0.015625 > 0.0  # water above residual remains for LEAF


def test_zero_share_removes_no_water_and_one_share_is_bitwise_the_historical_law():
    for lay in (layers(gravity=True, diffusivity=2e-7, factor=1.2), donor_layers(edges=EDGES3, k=[1e-6, 5e-7], d=0.05,
                                                                                   bottom="closed", factor=1.2)):
        _, mask, water, add, pet, dt = random_case(71, lay.n_layers, 37, gravity=True, mode="constant", bottom="open")
        col = column(lay, n_cells=37, active=mask)
        water = np.ascontiguousarray(np.minimum(water, lay.saturation_m[:, None]))
        base = step(col, column_state(col, water), dt, added_water_m=add, potential_evaporation_m_s=pet)
        one = step(col, column_state(col, water), dt, added_water_m=add, potential_evaporation_m_s=pet,
                   evaporation_share=np.ones(37))
        zero = step(col, column_state(col, water), dt, added_water_m=add, potential_evaporation_m_s=pet,
                    evaporation_share=np.zeros(37))
        for name, a in arrays_of(base).items():
            assert arrays_of(one)[name].tobytes() == a.tobytes(), name
        assert one.budget == base.budget
        assert np.all(zero.evaporation_m == 0.0) and np.all(zero.evaporation_unmet_m == 0.0)
        assert zero.budget.evaporation_m3 == 0.0 and abs(zero.budget.residual_m3) <= zero.budget.bound_m3
        # with share 0 the hydrology equals the same step at zero potential (no E at all), including the diffusion
        none = step(col, column_state(col, water), dt, added_water_m=add, potential_evaporation_m_s=np.zeros(37))
        assert zero.state.water_m.tobytes() == none.state.water_m.tobytes()


def test_detects_the_superseded_before_cap_partition():
    """Old candidate: feed PET x share as the potential (partition BEFORE the cap). New law: feed the combined potential
    and the share (partition AFTER the cap). Where the cap binds they differ materially: 1 mm withdrawable, combined
    4 mm, share 0.5 -> before-cap E = min(2, 1) = 1 mm; post-cap E = min(4, 1) x 0.5 = 0.5 mm."""
    col = column(thin_top_layers(), n_cells=1)
    state = state_of(col, [[0.001, 0.05]])
    combined = 4e-3 / 3600.0
    before_cap = run(col, state, 3600.0, [combined * 0.5], None)
    post_cap = run(col, state, 3600.0, [combined], [0.5])
    assert before_cap.evaporation_m[0] == pytest.approx(1e-3, rel=1e-12)
    assert post_cap.evaporation_m[0] == pytest.approx(0.5e-3, rel=1e-12)
    assert post_cap.evaporation_m[0] < before_cap.evaporation_m[0]
    # where the cap does NOT bind, the two coincide (share x potential == potential x share)
    loose_before = run(col, state, 3600.0, [0.4e-3 / 3600.0 * 0.5], None)
    loose_post = run(col, state, 3600.0, [0.4e-3 / 3600.0], [0.5])
    assert loose_post.evaporation_m[0] == pytest.approx(loose_before.evaporation_m[0], rel=1e-12)


def test_unmet_means_partitioned_potential_minus_actual_not_a_transpiration_loss():
    col = column(thin_top_layers(), n_cells=1)
    r = run(col, state_of(col, [[0.001, 0.05]]), 3600.0, [4e-3 / 3600.0], [0.25])
    assert r.evaporation_m[0] == pytest.approx(0.25e-3, rel=1e-12)
    assert r.evaporation_unmet_m[0] == pytest.approx(4e-3 * 0.25 - 0.25e-3, rel=1e-12)  # 0.75 mm, not 4 - 0.25
    assert r.budget.evaporation_unmet_m3 == pytest.approx(0.75e-3 * AREA, rel=1e-12)


def test_dt_zero_identity_with_a_share():
    col = column(layers(), n_cells=2)
    state = state_of(col, [[0.0625, 0.25], [0.125, 0.375]], t_s=7.0)
    r = run(col, state, 0.0, [1e-7, 1e-7], [0.3, 0.7])
    assert r.state.water_m.tobytes() == state.water_m.tobytes() and r.state.t_s == 7.0
    assert np.all(r.evaporation_m == 0.0) and np.all(r.evaporation_unmet_m == 0.0)
    with pytest.raises(SubsurfaceStepError) as info:  # an invalid share is flagged even on the identity step
        run(col, state, 0.0, [0.0, 0.0], [0.3, 1.5])
    assert info.value.bit == 16


# --- masks, invalid fields, failure without mutation ---------------------------------------------------------------------
def test_share_must_be_valid_on_inactive_cells_too_and_failure_leaves_inputs_unchanged():
    col = column(layers(gravity=True), n_cells=3, active=[True, False, True])
    state = state_of(col, [[0.0625, 0.125], [0.125, 0.375], [0.0625, 0.125]])
    add, pet = forcing(col, pet=[1e-7, 0.0, 1e-7])
    good = np.array([0.4, 0.0, 1.0])
    r = step(col, state, 60.0, added_water_m=add, potential_evaporation_m_s=pet, evaporation_share=good)
    np.testing.assert_array_equal(r.state.water_m[:, 1], [0.125, 0.375])  # inactive unchanged
    assert r.evaporation_m[1] == 0.0 and r.evaporation_unmet_m[1] == 0.0
    for bad, bit in (([0.4, np.nan, 1.0], 15), ([0.4, -0.1, 1.0], 16), ([0.4, 1.0000001, 1.0], 16),
                     ([np.inf, 0.0, 1.0], 15), ([0.4, 0.0, -1e-300], 16)):
        share = np.array(bad)
        before = (state.water_m.tobytes(), add.tobytes(), pet.tobytes(), share.tobytes())
        with pytest.raises(SubsurfaceStepError) as info:
            step(col, state, 60.0, added_water_m=add, potential_evaporation_m_s=pet, evaporation_share=share)
        assert info.value.bit == bit and info.value.n_flagged == 1 and FLAG_MESSAGES[bit] in str(info.value)
        assert (state.water_m.tobytes(), add.tobytes(), pet.tobytes(), share.tobytes()) == before


def test_share_array_metadata_is_validated_before_any_work():
    col = column(layers(), n_cells=2)
    state = state_of(col, [[0.0625, 0.125]] * 2)
    add, pet = forcing(col)
    for bad in (np.zeros(3), np.zeros(2, np.float32), np.zeros((1, 2)), [0.5, 0.5], np.zeros(4)[::2], 0.5):
        with pytest.raises(SubsurfaceError) as info:
            step(col, state, 1.0, added_water_m=add, potential_evaporation_m_s=pet, evaporation_share=bad)
        assert not isinstance(info.value, SubsurfaceStepError)


def test_lowest_bit_precedence_includes_the_share_bits():
    col = column(layers(), n_cells=2)
    state = state_of(col, [[0.0625, 0.125]] * 2)
    add, pet = forcing(col)
    pet[0] = -1.0  # bit 7 precedes bit 15
    with pytest.raises(SubsurfaceStepError) as info:
        step(col, state, 1.0, added_water_m=add, potential_evaporation_m_s=pet, evaporation_share=np.array([np.nan, 0.5]))
    assert info.value.bit == 7


# --- all modes, streaming, memory ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode, bottom", [("constant", "open"), ("constant", "closed"), ("tietjen_donor_depth", "open"),
                                          ("tietjen_donor_depth", "closed")])
def test_share_closes_in_every_mode_and_only_debits_actual_evaporation(mode, bottom):
    lay, mask, water, add, pet, dt = random_case(80, 4, 61, gravity=True, mode=mode, bottom=bottom)
    rng = np.random.default_rng(81)
    share = rng.uniform(0.0, 1.0, 61)
    col = prepare_column(lay, mask, AREA, backend="numpy", reserve_bytes=0, available_bytes_fn=lambda: BIG)
    r = step(col, column_state(col, water), dt, added_water_m=add, potential_evaporation_m_s=pet, evaporation_share=share)
    b = r.budget
    assert abs(b.residual_m3) <= b.bound_m3
    assert b.storage_after_m3 - b.storage_before_m3 == pytest.approx(
        b.added_m3 - b.surface_return_m3 - b.drainage_m3 - b.evaporation_m3, abs=b.bound_m3)
    full = step(col, column_state(col, water), dt, added_water_m=add, potential_evaporation_m_s=pet)
    assert np.all(r.evaporation_m <= full.evaporation_m * (1.0 + 1e-15) + 1e-20)  # share can only reduce the debit
    assert np.all(r.evaporation_m >= 0.0) and np.all(r.state.water_m >= 0.0)
    np.testing.assert_array_equal(r.state.water_m[:, ~mask], water[:, ~mask])


def test_streaming_records_carry_the_share():
    col = column(thin_top_layers(), n_cells=2)
    state = state_of(col, [[0.001, 0.05], [0.001, 0.05]])
    add, pet = forcing(col, pet=[2e-3 / 3600.0, 2e-3 / 3600.0])
    share = np.array([0.5, 0.5])
    records = [ForcingRecord(1800.0, add, pet, share), ForcingRecord(1800.0, add, pet, share), ForcingRecord(60.0, add, pet)]
    assert ForcingRecord(1.0, add, pet).evaporation_share is None  # old records keep the historical law
    profile = initialize_profile(col, state, records)
    # two half-hours at share .5: 1 mm demand each, limited 1 then .5 -> E .5 then .25 (theta .05 = stress after the
    # first: multiplier still 1); the last minute without a share runs the historical law at theta .025 (multiplier
    # (.025/.05)^2 = .25): demand 2/60 x .25 = 1/120 mm < remaining .25 mm -> E = 1/120 mm
    per_cell = 0.5e-3 + 0.25e-3 + (2e-3 / 60.0) * 0.25
    assert profile.cumulative.evaporation_m3 == pytest.approx(per_cell * 2 * AREA, rel=1e-12)
    np.testing.assert_allclose(profile.state.water_m[0], 1e-3 - per_cell, rtol=1e-12)
    assert abs(profile.residual_m3) <= profile.bound_m3 and profile.cumulative.n_steps == 3


def test_memory_counts_the_share_field_before_allocation():
    L, N = 3, 1000
    for backend in ("numpy", "numba", "cuda"):
        base = estimate_step_memory(L, N, backend=backend)
        with_share = estimate_step_memory(L, N, backend=backend, with_evaporation_share=True)
        assert with_share.input_bytes - base.input_bytes == N * 8
        assert with_share.output_bytes == base.output_bytes and with_share.scratch_bytes == base.scratch_bytes
        assert with_share.parameter_bytes == base.parameter_bytes
        assert with_share.required_bytes - base.required_bytes == N * 8
    with pytest.raises(SubsurfaceError):
        estimate_step_memory(L, N, backend="numpy", with_evaporation_share=1)
    budget = {"bytes": BIG}
    col = prepare_column(layers(), np.ones(5, dtype=bool), AREA, reserve_bytes=0, available_bytes_fn=lambda: budget["bytes"])
    state = state_of(col, [[0.0625, 0.125]] * 5)
    add, pet = forcing(col, pet=[1e-7] * 5)
    share = np.full(5, 0.5)
    assert step_memory(col) == col.memory and step_memory(col, with_evaporation_share=True).required_bytes == col.memory.required_bytes + 40
    budget["bytes"] = col.memory.required_bytes  # exactly the historical requirement
    step(col, state, 1.0, added_water_m=add, potential_evaporation_m_s=pet)  # no share: admitted
    with pytest.raises(SubsurfaceMemoryError):  # the share field is counted, refused before any allocation
        step(col, state, 1.0, added_water_m=add, potential_evaporation_m_s=pet, evaporation_share=share)
    with pytest.raises(SubsurfaceMemoryError):
        admit_step(col, with_evaporation_share=True)
    budget["bytes"] = col.memory.required_bytes + 40
    r = step(col, state, 1.0, added_water_m=add, potential_evaporation_m_s=pet, evaporation_share=share)
    assert r.budget.evaporation_m3 > 0.0
    # the combined transpiration estimator counts it too
    assert (st.estimate_combined_memory(col, query_available=False, with_evaporation_share=True).required_bytes
            - st.estimate_combined_memory(col, query_available=False).required_bytes) == 40


def test_reference_allocations_with_a_share_stay_within_the_declared_bound():
    import gc
    import tracemalloc

    L, N = 3, 100_000
    lay = soil_layers([0.0, 0.1, 0.3, 1.0], theta_residual=0.02, theta_field_capacity=0.2, theta_saturation=0.4,
                      diffusivity_m2_s=1e-7, theta_evaporation_stress=0.15, evaporation_factor=1.0, bottom_drainage="closed")
    col = prepare_column(lay, np.ones(N, dtype=bool), 1.0, backend="numpy", reserve_bytes=0, available_bytes_fn=lambda: BIG)
    rng = np.random.default_rng(9)
    state = column_state(col, np.ascontiguousarray(rng.uniform(0.0, 1.0, (L, N)) * lay.saturation_m[:, None]))
    added = np.zeros((L, N))
    added[0] = 0.03
    pet = np.full(N, 1e-7)
    share = rng.uniform(0.0, 1.0, N)
    gc.collect()
    tracemalloc.start()
    result = step(col, state, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, evaporation_share=share)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    est = step_memory(col, with_evaporation_share=True)
    assert est.output_bytes <= peak <= est.step_allocation_bytes, (est.output_bytes, peak, est.step_allocation_bytes)
    assert result.budget.evaporation_m3 > 0.0
    del result


# --- backends --------------------------------------------------------------------------------------------------------
def test_numba_matches_reference_with_a_share_bitwise_without_diffusion():
    pytest.importorskip("numba")
    lay, mask, water, add, pet, dt = random_case(90, 3, 70, gravity=True, mode="constant", bottom="closed")
    zero_d = layers(edges=np.array(lay.depth_edges_m), res=np.array(lay.theta_residual), fc=np.array(lay.theta_field_capacity),
                    sat=np.array(lay.theta_saturation), stress=float(lay.theta_evaporation_stress), factor=1.2,
                    diffusivity=0.0, bottom="closed")
    share = np.random.default_rng(91).uniform(0.0, 1.0, 70)
    outs = []
    for backend in ("numpy", "numba"):
        col = prepare_column(zero_d, mask, AREA, backend=backend, reserve_bytes=0, available_bytes_fn=lambda: BIG)
        outs.append(step(col, column_state(col, water), dt, added_water_m=add, potential_evaporation_m_s=pet,
                         evaporation_share=share))
    for name, a in arrays_of(outs[0]).items():
        assert arrays_of(outs[1])[name].tobytes() == a.tobytes(), name
    assert outs[0].budget == outs[1].budget and np.any(outs[0].evaporation_m > 0.0)


@pytest.mark.parametrize("mode, bottom", [("tietjen_donor_depth", "open"), ("constant", "closed")])
def test_numba_matches_reference_with_a_share_in_diffusive_modes(mode, bottom):
    pytest.importorskip("numba")
    lay, mask, water, add, pet, dt = random_case(92, 5, 41, gravity=True, mode=mode, bottom=bottom)
    share = np.random.default_rng(93).uniform(0.0, 1.0, 41)
    outs = []
    for backend in ("numpy", "numba"):
        col = prepare_column(lay, mask, AREA, backend=backend, reserve_bytes=0, available_bytes_fn=lambda: BIG)
        outs.append(step(col, column_state(col, water), dt, added_water_m=add, potential_evaporation_m_s=pet,
                         evaporation_share=share))
    for name, a in arrays_of(outs[0]).items():
        np.testing.assert_allclose(arrays_of(outs[1])[name], a, rtol=RTOL, atol=ATOL, err_msg=name)
    bad = share.copy()
    bad[3] = 1.2
    bits = []
    for backend in ("numpy", "numba"):
        col = prepare_column(lay, mask, AREA, backend=backend, reserve_bytes=0, available_bytes_fn=lambda: BIG)
        with pytest.raises(SubsurfaceStepError) as info:
            step(col, column_state(col, water), dt, added_water_m=add, potential_evaporation_m_s=pet, evaporation_share=bad)
        bits.append((info.value.bit, info.value.n_flagged))
    assert bits[0] == bits[1] == (16, 1)


def test_cuda_matches_reference_with_a_share(gpu):
    from maple.core.backend import read_transfer_counters

    for mode, bottom in (("constant", "open"), ("tietjen_donor_depth", "closed")):
        lay, mask, water, add, pet, dt = random_case(94, 4, 301, gravity=True, mode=mode, bottom=bottom)
        share = np.random.default_rng(95).uniform(0.0, 1.0, 301)
        ref = prepare_column(lay, mask, AREA, backend="numpy", reserve_bytes=0, available_bytes_fn=lambda: BIG)
        dev = prepare_column(lay, gpu.asarray(mask), AREA, backend="cuda", reserve_bytes=0, available_bytes_fn=lambda: BIG)
        r = step(ref, column_state(ref, water), dt, added_water_m=add, potential_evaporation_m_s=pet, evaporation_share=share)
        share_d = gpu.asarray(share)
        state_d = column_state(dev, gpu.asarray(water))
        add_d, pet_d = gpu.asarray(add), gpu.asarray(pet)
        before = read_transfer_counters()
        d = step(dev, state_d, dt, added_water_m=add_d, potential_evaporation_m_s=pet_d, evaporation_share=share_d)
        delta = read_transfer_counters().delta(before)
        assert delta.host_to_device == 0 and delta.device_to_host == 1  # the share stays device resident
        for name, a in arrays_of(r).items():
            np.testing.assert_allclose(host(arrays_of(d)[name]), a, rtol=RTOL, atol=ATOL, err_msg=f"{mode} {name}")
        assert d.budget.evaporation_m3 == pytest.approx(r.budget.evaporation_m3, rel=RTOL, abs=ATOL)
        one = step(dev, state_d, dt, added_water_m=add_d, potential_evaporation_m_s=pet_d, evaporation_share=gpu.ones(301))
        none = step(dev, state_d, dt, added_water_m=add_d, potential_evaporation_m_s=pet_d)
        for name, a in arrays_of(none).items():
            assert host(arrays_of(one)[name]).tobytes() == host(a).tobytes(), f"share 1 vs historical {name}"
        with pytest.raises(SubsurfaceError):  # host share for a device column: refused before any work
            step(dev, state_d, dt, added_water_m=add_d, potential_evaporation_m_s=pet_d, evaporation_share=share)
        bad = share_d.copy()
        bad[5] = -0.5
        with pytest.raises(SubsurfaceStepError) as info:
            step(dev, state_d, dt, added_water_m=add_d, potential_evaporation_m_s=pet_d, evaporation_share=bad)
        assert info.value.bit == 16 and info.value.n_flagged == 1


def test_cuda_user_example_and_withdrawal(gpu):
    dev = prepare_column(thin_top_layers(), gpu.ones(2, dtype=bool), AREA, backend="cuda", reserve_bytes=0,
                         available_bytes_fn=lambda: BIG)
    state = column_state(dev, gpu.asarray(np.array([[0.001, 0.001], [0.05, 0.05]])))
    pet = gpu.asarray(np.array([2e-3 / 3600.0, 0.2e-3 / 3600.0]))
    r = step(dev, state, 3600.0, added_water_m=gpu.zeros((2, 2)), potential_evaporation_m_s=pet, evaporation_share=gpu.full(2, 0.5))
    np.testing.assert_allclose(host(r.evaporation_m), [0.5e-3, 0.1e-3], rtol=1e-12)
    op = st.prepare_transpiration(dev)
    req = gpu.zeros((2, 2))
    req[0] = 0.001
    t = st.withdraw_transpiration(op, r.state, req)
    np.testing.assert_allclose(host(r.evaporation_m) + host(t.actual_m)[0], [1e-3, 1e-3], rtol=1e-12)
    assert t.state.t_s == 3600.0


# --- the helper stages the combined potential and the share -----------------------------------------------------------------
def test_helper_stages_unpartitioned_potential_and_share_for_the_native_law():
    part = se.EvaporationPartition(1.2, 0.2, 1.05, 0.04, 0.5)
    assert se.PARTITION_LEVEL == "post_cap_share" and part.summary()["partition_level"] == "post_cap_share"
    assert "AFTER" in part.summary()["partition_status"] and "exact" in part.summary()["soil_law_attribution"]
    assert part.combined_potential_fraction(0.5) == (1.2 - 0.1) * 0.5
    assert part.evaporation_share(0.5) == pytest.approx(1.0 - (1.05 * 0.5 - 0.04))
    assert part.evaporation_share(0.0) == 1.0 and part.evaporation_share(1.0) == 0.0  # Tabernas raw 1.01 -> clipped
    assert part.informational_e_only_fraction(0.5) == part.combined_potential_fraction(0.5) * part.evaporation_share(0.5)
    col = column(thin_top_layers(), n_cells=3, active=[True, True, False])
    op = se.prepare_evaporation(col, part)
    pet = np.array([4e-3 / 3600.0, 4e-3 / 3600.0, 0.0])
    cover = np.array([0.5, 0.0, 0.3])
    staged = se.stage_evaporation(op, pet_terrain_m_s=pet, cover=cover, surface_water_m=np.zeros(3), dt_s=3600.0)
    assert not hasattr(staged, "potential_soil_evaporation_m_s")  # the before-cap field no longer exists
    np.testing.assert_allclose(staged.potential_combined_soil_rate_m_s, [pet[0] * part.combined_potential_fraction(0.5),
                                                                        pet[1] * part.combined_potential_fraction(0.0), 0.0],
                               rtol=1e-14)
    np.testing.assert_allclose(staged.soil_evaporation_share, [part.evaporation_share(0.5), 1.0, part.evaporation_share(0.3)],
                               rtol=1e-14)  # valid on the inactive cell too
    kwargs = staged.native_step_kwargs()
    assert kwargs["potential_evaporation_m_s"] is staged.potential_combined_soil_rate_m_s
    assert kwargs["evaporation_share"] is staged.soil_evaporation_share
    led = staged.ledger
    assert led.soil_combined_potential_m3 == pytest.approx(float(staged.potential_combined_soil_rate_m_s[:2].sum()) * 3600.0 * AREA, rel=1e-13)
    assert led.soil_e_only_informational_m3 == pytest.approx(
        float((staged.potential_combined_soil_rate_m_s[:2] * staged.soil_evaporation_share[:2]).sum()) * 3600.0 * AREA, rel=1e-13)
    assert led.soil_e_only_informational_m3 < led.soil_combined_potential_m3
    # through the native column: 1 mm withdrawable, combined 4 mm x 1.1 x 0.5 = 2.2 mm >= 1 mm -> E = 1 mm x share
    state = state_of(col, [[0.001, 0.05], [0.001, 0.05], [0.001, 0.05]])
    add, _ = forcing(col)
    r = step(col, state, 3600.0, added_water_m=add, **kwargs)
    assert r.evaporation_m[0] == pytest.approx(1e-3 * part.evaporation_share(0.5), rel=1e-12)
    assert r.evaporation_m[1] == pytest.approx(1e-3 * 1.0, rel=1e-12)  # bare soil: all of the capped loss is E
    assert r.evaporation_m[2] == 0.0 and r.state.water_m[0, 2] == 0.001  # inactive
    # memory: five outputs
    assert op.memory.output_bytes == 5 * 3 * 8 and staged.memory == op.memory


def test_helper_cuda_share_and_combined_potential_match_reference(gpu):
    part = se.EvaporationPartition(1.2, 0.2, 1.05, 0.04, 0.9)
    mask = np.ones(200, dtype=bool)
    mask[::9] = False
    lay = layers(gravity=False, factor=1.0)
    ref = prepare_column(lay, mask, AREA, backend="numpy", reserve_bytes=0, available_bytes_fn=lambda: BIG)
    dev = prepare_column(lay, gpu.asarray(mask), AREA, backend="cuda", reserve_bytes=0, available_bytes_fn=lambda: BIG)
    rng = np.random.default_rng(96)
    pet, cover, surface = rng.uniform(0.0, 1e-7, 200), rng.uniform(0.0, 1.0, 200), rng.uniform(0.0, 1e-4, 200)
    r = se.stage_evaporation(se.prepare_evaporation(ref, part), pet_terrain_m_s=pet, cover=cover, surface_water_m=surface, dt_s=900.0)
    d = se.stage_evaporation(se.prepare_evaporation(dev, part), pet_terrain_m_s=gpu.asarray(pet), cover=gpu.asarray(cover),
                             surface_water_m=gpu.asarray(surface), dt_s=900.0)
    for name in ("potential_combined_soil_rate_m_s", "soil_evaporation_share"):
        assert type(getattr(d, name)) is gpu.ndarray
        np.testing.assert_allclose(host(getattr(d, name)), getattr(r, name), rtol=RTOL, atol=ATOL, err_msg=name)
    state = column_state(dev, gpu.asarray(np.full((2, 200), 0.1)))
    out = step(dev, state, 900.0, added_water_m=gpu.zeros((2, 200)), **d.native_step_kwargs())
    assert abs(out.budget.residual_m3) <= out.budget.bound_m3


# --- correction 01: no uncounted no-share allocation; the parameter ledger equals the resident buffers -------------------
@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("n_layers", [1, 3])
def test_numba_parameter_ledger_equals_resident_host_buffers_without_a_share_dummy(mode, n_layers):
    pytest.importorskip("numba")
    from maple_syrup import subsurface_numba
    from maple_syrup.subsurface import kernel_parameter_doubles, kernel_parameters

    assert not hasattr(subsurface_numba, "_NO_SHARE")  # no module-level dummy array
    edges = [0.0, 0.25, 0.5, 1.0][: n_layers + 1]
    if mode == "constant":
        lay = soil_layers(edges, theta_residual=0.0625, theta_field_capacity=0.25, theta_saturation=0.5,
                          theta_evaporation_stress=0.25, diffusivity_m2_s=1e-7)
    else:
        lay = soil_layers(edges, theta_residual=0.0625, theta_field_capacity=0.25, theta_saturation=0.5,
                          theta_evaporation_stress=0.25, diffusion_mode="tietjen_donor_depth",
                          interface_conductivity_m_s=1e-6, diffusion_coefficient=0.05)
    col = column(lay, n_cells=3, backend="numba")
    expected = kernel_parameter_doubles(n_layers, mode) * 8
    assert col.backend_info["parameter_host_bytes"] == expected
    params = kernel_parameters(lay)
    distinct = {id(getattr(params, f)): getattr(params, f) for f in params._fields if hasattr(getattr(params, f), "nbytes")}
    assert sum(a.nbytes for a in distinct.values()) == expected  # the declared count IS the distinct resident buffers
    state = state_of(col, [[0.0625] * n_layers] * 3)
    add, pet = forcing(col, pet=[1e-7] * 3)
    r0 = step(col, state, 60.0, added_water_m=add, potential_evaporation_m_s=pet)  # no share: resident pointer reused
    r1 = step(col, state, 60.0, added_water_m=add, potential_evaporation_m_s=pet, evaporation_share=np.ones(3))
    assert r0.state.water_m.tobytes() == r1.state.water_m.tobytes() and r0.budget == r1.budget


def test_cuda_parameter_ledger_equals_resident_device_buffers_without_a_share_dummy(gpu):
    from maple_syrup.subsurface import kernel_parameter_doubles

    for mode, n_layers in (("constant", 1), ("constant", 3), ("tietjen_donor_depth", 1), ("tietjen_donor_depth", 3)):
        edges = [0.0, 0.25, 0.5, 1.0][: n_layers + 1]
        if mode == "constant":
            lay = soil_layers(edges, theta_residual=0.0625, theta_field_capacity=0.25, theta_saturation=0.5,
                              theta_evaporation_stress=0.25, diffusivity_m2_s=1e-7)
        else:
            lay = soil_layers(edges, theta_residual=0.0625, theta_field_capacity=0.25, theta_saturation=0.5,
                              theta_evaporation_stress=0.25, diffusion_mode="tietjen_donor_depth",
                              interface_conductivity_m_s=1e-6, diffusion_coefficient=0.05)
        dev = prepare_column(lay, gpu.ones(3, dtype=bool), AREA, backend="cuda", reserve_bytes=0, available_bytes_fn=lambda: BIG)
        expected = kernel_parameter_doubles(n_layers, mode) * 8
        assert dev.backend_info["parameter_device_bytes"] == expected, (mode, n_layers)
        assert dev.backend_info["parameter_host_bytes"] == expected
        assert dev.memory.parameter_bytes == 2 * expected  # host + device copies, nothing else
        state = column_state(dev, gpu.full((n_layers, 3), 0.0625))
        pet = gpu.full(3, 1e-7)
        r0 = step(dev, state, 60.0, added_water_m=gpu.zeros((n_layers, 3)), potential_evaporation_m_s=pet)
        r1 = step(dev, state, 60.0, added_water_m=gpu.zeros((n_layers, 3)), potential_evaporation_m_s=pet,
                  evaporation_share=gpu.ones(3))
        assert host(r0.state.water_m).tobytes() == host(r1.state.water_m).tobytes()
