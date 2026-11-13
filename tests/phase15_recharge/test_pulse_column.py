"""Phase 1.5 S1: the ONE pulse-aware daily column movement (`subsurface.step_with_recharge`) on the SHARED column body. Analytic known-answer
controls (Root `root_independent_controls.md`), independent exact-rational oracle, masks, distinct fit failure, exact discarded parent,
historical-step identity and NumPy/Numba parity. CUDA cases skip without GPU0 (no claim)."""
from __future__ import annotations

import numpy as np
import pytest
from pcases import AREA, DAY, MM, BIG, layers, make_column, oracle_cascade, state_mm, zeros_pet

from maple_syrup import subsurface as ss
from maple_syrup.subsurface import (SubsurfaceError, SubsurfaceFitFailure, SubsurfaceMemoryError, SubsurfaceStepError, column_state,
                                    step, step_with_recharge)

BACKENDS = ["numpy", "numba"]
ATOL = 1e-15


def backend_or_skip(name):
    if name == "numba":
        pytest.importorskip("numba")
    return name


def run(col, cells_mm, rain_mm, pet=None, dt=DAY):
    st = state_mm(col, cells_mm)
    return step_with_recharge(col, st, dt, recharge_depth_m=rain_mm * MM, potential_evaporation_m_s=zeros_pet(col) if pet is None else pet), st


# Root table: (initial mm, rain mm, bottom, expected final mm, drainage mm)
TABLE = [
    ([0], 10, "closed", [10], 0),
    ([20], 10, "closed", [30], 0),
    ([20], 10, "open", [20], 10),
    ([0, 0], 30, "closed", [20, 10], 0),
    ([20, 20], 30, "closed", [30, 40], 0),
    ([20, 20], 30, "open", [20, 20], 30),
    ([20, 20, 20], 50, "closed", [30, 40, 40], 0),
    ([20, 20, 20], 50, "open", [20, 20, 20], 50),
    ([40, 40], 1, "open", [20, 20], 41),
    ([40, 40, 0], 10, "closed", [20, 30, 40], 0),  # water passes through an already-full intermediate layer when lower storage exists
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("initial,rain,bottom,final,drain", TABLE)
def test_root_known_answer_table(backend, initial, rain, bottom, final, drain):
    col = make_column(layers(len(initial), bottom=bottom), 1, backend_or_skip(backend))
    out, st = run(col, [initial], rain)
    np.testing.assert_allclose(out.state.water_m[:, 0] / MM, final, rtol=0, atol=1e-9)
    assert out.drainage_m[0] / MM == pytest.approx(drain, abs=1e-9)
    assert out.surface_return_m[0] == 0.0  # NO premature top saturation return, even when layer 0 transiently exceeds SAT
    assert out.state.t_s == DAY and out.dt_s == DAY
    assert abs(out.budget.residual_m3) <= out.budget.bound_m3
    assert out.budget.added_m3 == pytest.approx(rain * MM * AREA, rel=1e-12)  # the pulse is the only addition
    assert out.budget.evaporation_m3 == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("initial,rain,bottom", [([40, 40], 1, "closed"), ([40], 1, "closed"), ([40, 40, 40], 5, "closed")])
def test_saturated_closed_profile_is_a_distinct_fit_failure_and_the_parent_is_untouched(backend, initial, rain, bottom):
    col = make_column(layers(len(initial), bottom=bottom), 1, backend_or_skip(backend))
    st = state_mm(col, [initial])
    before = st.water_m.copy()
    with pytest.raises(SubsurfaceFitFailure) as info:
        step_with_recharge(col, st, DAY, recharge_depth_m=rain * MM, potential_evaporation_m_s=zeros_pet(col))
    assert info.value.bit == ss.RECHARGE_FIT_BIT and info.value.n_flagged == 1
    assert isinstance(info.value, SubsurfaceStepError)
    assert np.array_equal(st.water_m, before) and st.t_s == 0.0  # exact discarded parent: arrays and clock


@pytest.mark.parametrize("backend", BACKENDS)
def test_gravity_disabled_never_exploits_lower_room(backend):
    lay = layers(2, bottom="closed", gravity=False)
    col = make_column(lay, 1, backend_or_skip(backend))
    with pytest.raises(SubsurfaceFitFailure):  # [20, 0] + 30 -> layer 0 would be 50 > SAT 40; no silent gravity
        run(col, [[20, 0]], 30)
    out, _ = run(col, [[20, 0]], 10)
    np.testing.assert_allclose(out.state.water_m[:, 0] / MM, [30, 0], atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_heterogeneous_columns_one_failure_rejects_the_whole_occurrence(backend):
    col = make_column(layers(2, bottom="closed"), 2, backend_or_skip(backend))
    st = state_mm(col, [[40, 40], [0, 0]])
    before = st.water_m.copy()
    with pytest.raises(SubsurfaceFitFailure) as info:
        step_with_recharge(col, st, DAY, recharge_depth_m=1 * MM, potential_evaporation_m_s=zeros_pet(col))
    assert info.value.n_flagged == 1  # only the saturated column fails; no per-column result exists
    assert np.array_equal(st.water_m, before)


@pytest.mark.parametrize("backend", BACKENDS)
def test_masks_inactive_cells_are_bitwise_unchanged_and_get_no_pulse(backend):
    col = make_column(layers(2, bottom="closed"), 3, backend_or_skip(backend), active=[True, False, True])
    cells = [[0, 0], [11.25, 33.5], [20, 20]]  # a nonzero sentinel in the inactive cell
    out, st = run(col, cells, 10)
    assert np.array_equal(out.state.water_m[:, 1], st.water_m[:, 1])
    assert out.drainage_m[1] == 0.0 and out.surface_return_m[1] == 0.0 and out.evaporation_m[1] == 0.0
    assert out.budget.added_m3 == pytest.approx(2 * 10 * MM * AREA, rel=1e-12)  # two active cells only
    # an inactive-cell failure cannot reject the occurrence and an inactive overflow is ignored by the fit test
    col2 = make_column(layers(2, bottom="closed"), 2, backend_or_skip(backend), active=[True, False])
    out2, _ = run(col2, [[0, 0], [40, 40]], 5)
    np.testing.assert_allclose(out2.state.water_m[:, 1] / MM, [40, 40])
    assert float(out2.state.water_m[0, 0]) == pytest.approx(5 * MM)


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_pulse_equals_the_historical_step_bitwise(backend):
    """Default arithmetic unchanged: with P = 0 the pulse-aware step and the historical step (zero added water) share one body and
    produce identical bits, including nonzero evaporation and diffusion."""
    rng = np.random.default_rng(3)
    lay = layers(3, bottom="open", diffusivity=1e-9, factor=1.2, stress=0.12)
    col = make_column(lay, 16, backend_or_skip(backend))
    water = np.ascontiguousarray(rng.uniform(0.0, 0.04, (3, 16)) * np.array([[0.1], [0.1], [0.1]]))
    pet = col.xp.asarray(rng.uniform(0.0, 5e-8, 16))
    st = column_state(col, water)
    a = step(col, st, DAY, added_water_m=col.xp.zeros((3, 16)), potential_evaporation_m_s=pet)
    b = step_with_recharge(col, st, DAY, recharge_depth_m=0.0, potential_evaporation_m_s=pet)
    for name in ("state", ):
        assert np.array_equal(a.state.water_m, b.state.water_m)
    for name in ("surface_return_m", "drainage_m", "evaporation_m", "evaporation_unmet_m", "percolation_m", "diffusion_m"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert float(np.sum(b.evaporation_m)) > 0.0 and float(np.sum(np.abs(b.diffusion_m))) > 0.0  # E and diffusion really ran


@pytest.mark.parametrize("backend", BACKENDS)
def test_evaporation_and_diffusion_run_once_on_the_fitted_column(backend):
    """Independent composition: the pulse-aware step equals (pulse + cascade settle) followed by ONE historical step; the settle is
    obtained with E=0 and D=0 and the exact-rational oracle gives the same settled soil; the existing cascade is then a no-op."""
    lay_d = layers(3, bottom="closed", diffusivity=3e-9, factor=1.2, stress=0.12)
    lay_0 = layers(3, bottom="closed", diffusivity=0.0)
    cells = [[20, 20, 20], [0, 5, 10], [30, 40, 10]]
    rain = 12.0
    col_d = make_column(lay_d, 3, backend_or_skip(backend))
    col_0 = make_column(lay_0, 3, backend_or_skip(backend))
    pet = col_d.xp.asarray(np.array([4e-8, 2e-8, 6e-8]))
    full, st = run(col_d, cells, rain, pet=pet)
    settled, _ = run(col_0, cells, rain)  # E = 0, D = 0: the settled soil
    # exact-rational settled oracle
    for c in range(3):
        w, _drain, fits = oracle_cascade([x * MM for x in cells[c]], [0.02] * 3, [0.04] * 3, True, True, rain * MM)
        assert fits
        np.testing.assert_allclose(settled.state.water_m[:, c], [float(x) for x in w], rtol=0, atol=1e-15)
    one_more = step(col_d, settled.state, DAY, added_water_m=col_d.xp.zeros((3, 3)), potential_evaporation_m_s=pet)
    np.testing.assert_allclose(full.state.water_m, one_more.state.water_m, rtol=0, atol=1e-15)  # same E and diffusion, exactly once
    np.testing.assert_allclose(full.evaporation_m, one_more.evaporation_m, rtol=0, atol=1e-15)
    assert float(np.sum(full.evaporation_m)) > 0.0 and float(np.sum(np.abs(full.diffusion_m))) > 0.0
    assert abs(full.budget.residual_m3) <= full.budget.bound_m3


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_layers", [1, 2, 3, 5])
@pytest.mark.parametrize("bottom", ["open", "closed"])
@pytest.mark.parametrize("gravity", [True, False])
def test_exact_rational_oracle_over_random_dyadic_cases(backend, n_layers, bottom, gravity):
    rng = np.random.default_rng(100 + n_layers)
    dz, fc, sat = 0.125, 0.25, 0.5  # dyadic: every fixture value is exactly representable and every sum is exact
    lay = layers(n_layers, bottom=bottom, gravity=gravity, dz=dz, fc=fc, sat=sat, stress=0.125)
    n = 64
    col = make_column(lay, n, backend_or_skip(backend))
    fc_m, sat_m = fc * dz, sat * dz
    water = np.ascontiguousarray(rng.integers(0, int(sat_m * 4096) + 1, (n_layers, n)).astype(np.float64) / 4096.0)
    pulse = float(rng.integers(0, 600)) / 8192.0
    st = column_state(col, water)
    expected = [oracle_cascade(list(water[:, c]), [fc_m] * n_layers, [sat_m] * n_layers, bottom == "closed", gravity, pulse) for c in range(n)]
    n_fail = sum(1 for _w, _d, fits in expected if not fits)
    pet = zeros_pet(col)
    if n_fail:
        with pytest.raises(SubsurfaceFitFailure) as info:
            step_with_recharge(col, st, DAY, recharge_depth_m=pulse, potential_evaporation_m_s=pet)
        assert info.value.n_flagged == n_fail
        return
    out = step_with_recharge(col, st, DAY, recharge_depth_m=pulse, potential_evaporation_m_s=pet)
    for c in range(n):
        w, d, _ = expected[c]
        assert out.state.water_m[:, c].tolist() == [float(x) for x in w], c  # exact
        assert float(out.drainage_m[c]) == float(d)
    assert float(np.sum(out.surface_return_m)) == 0.0


def test_fit_failure_priority_over_nothing_but_below_invalid_prior_state():
    col = make_column(layers(2, bottom="closed"), 2)
    # cell 0: invalid prior state (above saturation + tolerance); cell 1: a would-be fit failure. The lowest bit (2) names the invalid state.
    st = state_mm(col, [[50, 0], [40, 40]])
    with pytest.raises(SubsurfaceStepError) as info:
        step_with_recharge(col, st, DAY, recharge_depth_m=1 * MM, potential_evaporation_m_s=zeros_pet(col))
    assert not isinstance(info.value, SubsurfaceFitFailure) and info.value.bit == 2


@pytest.mark.parametrize("bad", [-1e-3, float("nan"), float("inf"), True, "1", None, np.array([0.01])])
def test_invalid_pulse_is_refused_before_any_work(bad):
    col = make_column(layers(2), 1)
    st = state_mm(col, [[0, 0]])
    with pytest.raises(SubsurfaceError):
        step_with_recharge(col, st, DAY, recharge_depth_m=bad, potential_evaporation_m_s=zeros_pet(col))


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf"), True])
def test_zero_time_and_bad_dt_are_refused_no_zero_time_rain_step(dt):
    col = make_column(layers(2), 1)
    st = state_mm(col, [[0, 0]])
    with pytest.raises(SubsurfaceError):
        step_with_recharge(col, st, dt, recharge_depth_m=0.001, potential_evaporation_m_s=zeros_pet(col))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("t0,dt", [(1e308, DAY), (1e22, DAY), (1.0, 1e-20), (2.0 ** 60, 1.0)])
def test_positive_dt_that_cannot_advance_the_native_clock_is_refused_before_any_work(backend, t0, dt):
    import dataclasses
    col = make_column(layers(2, bottom="open"), 3, backend_or_skip(backend))
    assert t0 + dt == t0  # the clock cannot represent this interval
    calls = []

    def spy(*a, **k):
        calls.append(1)
        raise AssertionError("the runner must not be reached")

    spied = dataclasses.replace(col, runner=spy, preflight=lambda: calls.append(2))
    st = state_mm(col, [[0, 0]] * 3, t_s=t0)
    before = np.array(st.water_m)
    with pytest.raises(SubsurfaceError, match="advance the native clock"):
        step_with_recharge(spied, st, dt, recharge_depth_m=0.001, potential_evaporation_m_s=zeros_pet(col))
    assert calls == [] and np.array_equal(st.water_m, before) and st.t_s == t0  # no preflight, no admission, no kernel, no mutation
    ok = step_with_recharge(col, state_mm(col, [[0, 0]] * 3, t_s=1e15), DAY, recharge_depth_m=0.001, potential_evaporation_m_s=zeros_pet(col))
    assert ok.state.t_s == 1e15 + DAY  # an advancing interval at a large clock is still accepted


def test_historical_step_clock_behaviour_is_left_as_it_was():
    col = make_column(layers(2, bottom="open"), 1)
    st = state_mm(col, [[0, 0]], t_s=1e308)
    out = step(col, st, DAY, added_water_m=np.zeros((2, 1)), potential_evaporation_m_s=zeros_pet(col))
    assert out.state.t_s == 1e308  # the ordinary step is deliberately not retrofitted (Root R3): this documents, not endorses, it


def test_memory_refusal_precedes_any_allocation_and_is_distinct():
    import dataclasses
    full = make_column(layers(2), 8)
    st = state_mm(full, [[0, 0]] * 8)
    small = dataclasses.replace(full, available_bytes_fn=lambda: 1)
    with pytest.raises(SubsurfaceMemoryError):
        step_with_recharge(small, st, DAY, recharge_depth_m=0.001, potential_evaporation_m_s=zeros_pet(full))
    est = ss.estimate_step_memory(2, 8, backend="numpy", with_recharge=True)
    base = ss.estimate_step_memory(2, 8, backend="numpy")
    assert base.input_bytes - est.input_bytes == 2 * 8 * 8 and est.step_allocation_bytes == base.step_allocation_bytes  # no (L, N) forcing tensor


def test_historical_default_modes_and_signatures_are_unchanged():
    import inspect
    sig = inspect.signature(ss.step)
    assert [p.name for p in sig.parameters.values()] == ["column", "state", "dt_s", "added_water_m", "potential_evaporation_m_s", "evaporation_share"]
    assert ss.cell_closure_coefficient(3) == ss.cell_closure_coefficient(3, False) < ss.cell_closure_coefficient(3, True)
    assert ss.N_FLAGS == 18 and ss.FLAG_MESSAGES[17].startswith("recharge pulse does not fit")
    assert ss.STAGE_ORDER == ("validation", "addition", "gravity_cascade", "evaporation", "diffusion", "output_checks")


def test_numba_matches_numpy_on_random_pulse_cases_within_the_declared_bound():
    pytest.importorskip("numba")
    rng = np.random.default_rng(11)
    for n_layers in (1, 2, 3, 5):
        for bottom in ("open", "closed"):
            for gravity in (True, False):
                lay = layers(n_layers, bottom=bottom, gravity=gravity, diffusivity=2e-9, factor=1.2, stress=0.12)
                n = 40
                ref, jit = make_column(lay, n, "numpy"), make_column(lay, n, "numba")
                water = np.ascontiguousarray(rng.uniform(0.0, 0.04, (n_layers, n)))
                pet = rng.uniform(0.0, 6e-8, n)
                pulse = float(rng.uniform(0.0, 0.03))
                outs = []
                for col in (ref, jit):
                    try:
                        outs.append(step_with_recharge(col, column_state(col, water), DAY, recharge_depth_m=pulse, potential_evaporation_m_s=pet.copy()))
                    except SubsurfaceFitFailure as exc:
                        outs.append(exc.n_flagged)
                assert type(outs[0]) is type(outs[1]), (n_layers, bottom, gravity)
                if isinstance(outs[0], int):
                    assert outs[0] == outs[1]
                    continue
                for name in ("drainage_m", "evaporation_m", "evaporation_unmet_m", "percolation_m", "diffusion_m"):
                    np.testing.assert_allclose(getattr(outs[1], name), getattr(outs[0], name), rtol=2e-12, atol=1e-14, err_msg=name)
                np.testing.assert_allclose(outs[1].state.water_m, outs[0].state.water_m, rtol=2e-12, atol=1e-14)
                assert float(np.sum(outs[0].surface_return_m)) == float(np.sum(outs[1].surface_return_m))


@pytest.mark.parametrize("initial,rain,bottom,final,drain", TABLE[:8])
def test_cuda_known_answers_when_gpu0_is_available(gpu, initial, rain, bottom, final, drain):
    cp = gpu
    col = make_column(layers(len(initial), bottom=bottom), 1, "cuda")
    out, _ = run(col, [initial], rain)
    np.testing.assert_allclose(cp.asnumpy(out.state.water_m)[:, 0] / MM, final, rtol=0, atol=1e-9)
    assert float(cp.asnumpy(out.drainage_m)[0]) / MM == pytest.approx(drain, abs=1e-9)


@pytest.mark.parametrize("share", [False, True])
@pytest.mark.parametrize("n_layers", [1, 3, 6])
def test_numpy_pulse_mode_actual_live_bytes_stay_within_the_historical_mask_allowance(n_layers, share):
    """R6: MEASURED (tracemalloc, which traces NumPy allocations), not structural. The pulse mode's extra simultaneously live masks
    (fit_fail + its NOT + the replacement `checked`, N bytes each, on top of the historical masks) must fit the declared 8 bytes/cell mask
    allowance, and the whole call must stay inside the declared step allocation; the comparison peak is the historical step on the
    same column (caller-owned state/added tensors excluded from both)."""
    import tracemalloc
    n = 200_000
    col = make_column(layers(n_layers, bottom="closed"), n)
    w = np.full((n_layers, n), 0.01)
    pet = np.zeros(n)
    sh = np.full(n, 0.5) if share else None
    added = np.zeros((n_layers, n))

    def peak(fn):
        st = column_state(col, w.copy(), 0.0)
        tracemalloc.start()
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        out = fn(st)
        top = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
        del out
        return top

    hist = peak(lambda st: step(col, st, DAY, added_water_m=added, potential_evaporation_m_s=pet, evaporation_share=sh))
    pulse = peak(lambda st: step_with_recharge(col, st, DAY, recharge_depth_m=0.001, potential_evaporation_m_s=pet, evaporation_share=sh))
    est = ss.estimate_step_memory(n_layers, n, backend="numpy", with_evaporation_share=share, with_recharge=True)
    assert pulse - hist <= ss.SCRATCH_MASK_BYTES_PER_CELL * n  # measured extra live bytes fit the declared per-cell mask bytes
    assert pulse <= est.step_allocation_bytes  # and the whole call fits the declared step allocation
