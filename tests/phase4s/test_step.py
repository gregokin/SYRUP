"""Phase 4S: the prepared CUDA coupled hydrology step on a real device, against the accepted NumPy `array` reference
(`storm.coupled_step`, no Numba needed) and, where Numba exists, the prepared CPU step.

Public fields: rtol 2e-12 / atol 1e-14 (arrays bitwise in the saturated regime, where the column arithmetic does not
depend on libm); counts, bools, strings exact. Failure cases compare exception CLASS and MESSAGE with the reference.
Skipped without a device (explicit skip, no GPU claim). Nothing here was run by its author (file-only tools); Codex
records results. Select the device with CUDA_VISIBLE_DEVICES before starting pytest.
"""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import weakref

import hydro_cases as hcs
import numpy as np
import pytest

pytest.importorskip("maple")

from maple.core import backend as mb

from maple_syrup import hydrology_cuda as hc
from maple_syrup import routing_numba
from maple_syrup.infiltration import ColumnStep, InfiltrationError
from maple_syrup.routing import RoutingError, RoutingStepRejected
from maple_syrup.storm import CoupledStep, StormControl, StormError, StormState

pytestmark = pytest.mark.usefixtures("gpu")

CUDA = StormControl(implementation="cuda")  # deliberately NOT .validated(): StormControl.validated() refuses cuda
MODES = ("fused", "split", "auto")


def cupy():
    import cupy as cp

    return cp


def prepared(case, mode="auto"):
    cp = cupy()
    dev = hcs.device_case(case, cp)
    return dev, hc.prepare_cuda_hydrology(dev.graph, dev.params, mode=mode)


def step_pair(case, dev, ctx, rate_host, rate_dev, state, dt, control=CUDA):
    """Reference step on the host state and CUDA step on its device upload (same input bits)."""
    cp = cupy()
    ref = hcs.ref_step(case, rate_host, state, dt)
    new = hc.prepared_coupled_step(ctx, rate_dev, hcs.upload_state(state, cp), dt, control)
    return ref, new


# --- differential --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["random", "valley", "valley_masked"])
@pytest.mark.parametrize("model", ["pavement_hawkins", "fixed_ksat"])
def test_unsaturated_columns_match_the_reference_in_every_mode(model, kind, mode):
    cp = cupy()
    case = hcs.build_case(3, kind=kind, model=model)
    dev, ctx = prepared(case, mode)
    zero = np.zeros(case.graph.shape)
    zero_d = cp.zeros(case.graph.shape)
    state = case.state
    totals = {"no": 0, "partial": 0, "complete": 0, "drain": 0.0, "intake": 0.0}
    dts = ([1.0, 0.5, 0.25, 1.0 / 1024.0, 1.0, 1.0, 0.5] * 4)[:24]
    for i, dt in enumerate(dts):
        on = i < 14
        ref, new = step_pair(case, dev, ctx, case.rate_on if on else zero, dev.rate_on if on else zero_d, state, dt)
        assert isinstance(new, CoupledStep) and isinstance(new.column, ColumnStep)
        hcs.compare(ref, new)
        assert new.route.implementation == "cuda" and new.route.conservative is True
        totals["no"] += int(ref.n_no_runon)
        totals["partial"] += int(ref.n_partial_runon)
        totals["complete"] += int(ref.n_complete_runon)
        totals["drain"] += float(ref.column.drainage_m.sum())
        totals["intake"] += float(ref.column.intake_m.sum())
        state = ref.state
    assert totals["drain"] > 0.0 and totals["intake"] > 0.0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["valley", "random", "valley_masked"])
def test_saturated_regime_arrays_are_bitwise_in_every_mode(kind, mode):
    """Near-full columns: expm1(-x) is exactly -1 on device and host, capacity is exactly Ksat, every cellwise field is
    IEEE-exact. Rain on, saturation return, then recession. Scalars are checked within the declared bound only (the
    device sums use a fixed tree, not NumPy's pairwise order)."""
    cp = cupy()
    case = hcs.build_case(11, kind=kind, model="fixed_ksat", saturated=True)
    dev, ctx = prepared(case, mode)
    zero, zero_d = np.zeros(case.graph.shape), cp.zeros(case.graph.shape)
    state = case.state
    seen = {"return": 0.0, "intake": 0.0, "no": 0, "partial": 0}
    for i in range(35):
        on = i < 20
        ref, new = step_pair(case, dev, ctx, case.rate_on if on else zero, dev.rate_on if on else zero_d, state, 1.0)
        hcs.compare(ref, new, exact_arrays=True)
        seen["return"] += float(ref.column.saturation_return_m.sum())
        seen["intake"] += float(ref.column.intake_m.sum())
        seen["no"] += int(ref.n_no_runon)
        seen["partial"] += int(ref.n_partial_runon)
        state = ref.state
    assert seen["no"] > 0 and seen["partial"] > 0 and seen["return"] > 0.0 and seen["intake"] > 0.0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("width", [1, 127, 128, 129, 255, 256, 257])
def test_exact_level_widths_match_the_reference(width, mode):
    """Plane graphs: every row is ONE dependency level of exactly `width` cells (block-boundary widths)."""
    cp = cupy()
    case = hcs.build_case(5, kind=f"plane:{width}", model="pavement_hawkins")
    dev, ctx = prepared(case, mode)
    assert ctx.max_level_width == width
    expect = hc.select_mode(width, mode)
    assert ctx.mode == expect and ctx.requested_mode == mode
    state = case.state
    for _ in range(3):
        ref, new = step_pair(case, dev, ctx, case.rate_on, dev.rate_on, state, 0.5)
        hcs.compare(ref, new)
        state = ref.state
    del cp


@pytest.mark.parametrize("dt, courant, iterations, root_tol", [
    (1.0, 1.0, 40, 1e-11), (0.25, 0.5, 60, 1e-12), (1.0 / 1024.0, 2.0, 30, 1e-9), (0.5, 1, np.int64(45), 1e-10),
])
def test_dt_courant_iteration_and_root_controls(dt, courant, iterations, root_tol):
    case = hcs.build_case(6, kind="valley", model="pavement_hawkins")
    dev, ctx = prepared(case)
    kw = {"courant_max": courant, "bisection_iterations": iterations, "root_tolerance_m": root_tol}
    state = case.state
    for _ in range(6):
        ref = hcs.ref_step(case, case.rate_on, state, dt, **kw)
        new = hc.prepared_coupled_step(ctx, dev.rate_on, hcs.upload_state(state, cupy()), dt,
                                       StormControl(implementation="cuda", **kw))
        hcs.compare(ref, new)
        assert new.route.bisection_iterations == int(iterations)
        state = ref.state


def _chain(depth_value, *, rate_value=0.0, ksat=1e-6, n=3):
    """Full (S = Smax) fixed-Ksat columns on a chain: Smith-Parlange capacity is exactly Ksat."""
    from test_routing import chain_full, make_graph

    from maple_syrup.infiltration import column_parameters, initial_soil_water_m
    from maple_syrup.storm import initial_state

    graph = make_graph(chain_full(n), ff=5.0)
    shape = graph.shape

    def full(v):
        return np.full(shape, float(v))

    params = column_parameters(model="fixed_ksat", ksat_m_per_s=full(ksat), suction_m=full(0.01),
                               drainage_parameter=full(0.0), theta_sat=full(0.4), soil_thickness_m=full(0.3),
                               active_mask=graph.active.copy())
    soil = initial_soil_water_m(params, full(0.4))
    spec = {"z": chain_full(n), "ff": 5.0, "active": None, "ksat": full(ksat), "mask": graph.active.copy(),
            "model": "fixed_ksat",
            "fields": {"suction_m": full(0.01), "drainage_parameter": full(0.0), "theta_sat": full(0.4),
                       "soil_thickness_m": full(0.3)}}
    return hcs.Case(graph, params, initial_state(graph, full(depth_value), soil), full(rate_value), spec)


@pytest.mark.parametrize("dt", [1.0, 0.25, 1.0 / 1024.0])
@pytest.mark.parametrize("mode", ["fused", "split"])
def test_near_boundary_capacities_and_runon_branch_ties(dt, mode):
    ksat = 1e-6
    cap = ksat * dt
    cases = {  # name: (depth, rain rate, expected branch of every active cell)
        "complete_tie_J_equals_A": (0.0, ksat, "complete"),
        "no_runon_tie_J_equals_P": (2e-3, ksat, "no_runon"),
        "complete_tie_J_equals_h": (cap, 0.0, "complete"),
        "partial_just_above": (float(np.nextafter(cap, 1.0)), 0.0, "partial"),
        "complete_just_below": (float(np.nextafter(cap, 0.0)), 0.0, "complete"),
    }
    for name, (depth, rate, branch) in cases.items():
        case = _chain(depth, rate_value=rate, ksat=ksat)
        dev, ctx = prepared(case, mode)
        ref, new = step_pair(case, dev, ctx, case.rate_on, dev.rate_on, case.state, dt)
        counts = {"no_runon": int(ref.n_no_runon), "partial": int(ref.n_partial_runon),
                  "complete": int(ref.n_complete_runon)}
        assert counts[branch] == case.graph.n_active and sum(counts.values()) == case.graph.n_active, (name, counts)
        hcs.compare(ref, new, exact_arrays=True, path=name)


@pytest.mark.parametrize("soil_value", [0.0, 1e-320])
def test_limiting_intake_on_a_dry_column_stays_accepted(soil_value):
    from maple_syrup.storm import initial_state

    case = hcs.build_case(7, kind="valley_masked", model="fixed_ksat", saturated=True, adversarial=True)
    soil = np.full(case.graph.shape, soil_value)
    state = initial_state(case.graph, case.state.depth_m, soil)
    dev, ctx = prepared(case)
    ref, new = step_pair(case, dev, ctx, case.rate_on, dev.rate_on, state, 1.0)
    hcs.compare(ref, new, exact_arrays=True)
    available = ref.column.rain_m + state.depth_m
    np.testing.assert_array_equal(hcs.host(new.column.intake_m)[case.graph.active], available[case.graph.active])


def test_masked_cells_keep_their_inventory_and_report_zero_flux():
    case = hcs.build_case(8, kind="valley_masked", model="pavement_hawkins")
    dev, ctx = prepared(case)
    new = hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, CUDA)
    inactive = ~case.graph.active
    assert inactive.any()
    np.testing.assert_array_equal(hcs.host(new.state.depth_m)[inactive], case.state.depth_m[inactive])
    np.testing.assert_array_equal(hcs.host(new.state.soil_water_m)[inactive], case.state.soil_water_m[inactive])
    for array in (new.state.discharge_m2_s, new.route.velocity_m_s, new.route.face_volume_m3, new.route.inflow_m2_s,
                  new.column.intake_m, new.column.drainage_m, new.route.old_discharge_m2_s):
        assert not np.any(hcs.host(array)[inactive])
    hcs.compare(hcs.ref_step(case, case.rate_on, case.state, 1.0), new)


@pytest.mark.skipif(not routing_numba.numba_available(), reason="Numba prepared CPU oracle not installed")
@pytest.mark.parametrize("mode", MODES)
def test_matches_the_prepared_cpu_step_too(mode):
    from maple_syrup import hydrology_numba as hn

    case = hcs.build_case(9, kind="random", model="pavement_hawkins")
    dev, ctx = prepared(case, mode)
    cpu_ctx = hn.prepare_hydrology(case.graph, case.params)
    cpu_control = StormControl(implementation="numba")
    state = case.state
    for _ in range(8):
        cpu = hn.prepared_coupled_step(cpu_ctx, case.rate_on, state, 1.0, cpu_control)
        new = hc.prepared_coupled_step(ctx, dev.rate_on, hcs.upload_state(state, cupy()), 1.0, CUDA)
        hcs.compare(cpu, new)
        state = cpu.state


# --- the reference's own strict roundoff guard must bite identically --------------------------------------------------
def _find_inconsistent_wet_depth(rain_m: float = 1e-5) -> float:
    rng = np.random.default_rng(2024)
    for h in rng.uniform(1e-6, 1e-3, 4000):
        h = np.float64(h)
        available = h + np.float64(rain_m)
        hpre = max(h - max(available - np.float64(rain_m), 0.0), 0.0)
        if hpre > (available - available) + 0.0:
            return float(h)
    raise AssertionError("no inconsistent depth found; the roundoff premise of this reproducer changed")


def _find_partial_positive_rain_depth(rain_m: float = 1e-5, ksat: float = 3e-5) -> float:
    rng = np.random.default_rng(77)
    for h in rng.uniform(1e-3, 3e-3, 4000):
        h = np.float64(h)
        available = h + np.float64(rain_m)
        intake = min(available, np.float64(ksat))
        hpre = max(h - max(intake - np.float64(rain_m), 0.0), 0.0)
        if hpre > (available - intake) + 0.0:
            return float(h)
    raise AssertionError("no inconsistent partial-intake depth found; the roundoff premise changed")


@pytest.mark.parametrize("which", ["dry_full_intake", "partial_positive_rain"])
def test_former_roundoff_triggers_run_the_coherent_branch_with_equivalent_device_state(which):
    from maple_syrup.storm import initial_state

    ksat = 1e-6 if which == "dry_full_intake" else 3e-5
    h = _find_inconsistent_wet_depth() if which == "dry_full_intake" else _find_partial_positive_rain_depth()
    case = _chain(0.0, rate_value=1e-5, ksat=ksat)
    depth = np.zeros(case.graph.shape)
    depth[1, 0] = h
    soil = np.zeros(case.graph.shape) if which == "dry_full_intake" else case.params.storage_max_m - 1e-4
    state = initial_state(case.graph, depth, soil)
    rate = np.full(case.graph.shape, 1e-5)
    _dev, ctx = prepared(case)
    cp = cupy()
    dstate = hcs.upload_state(state, cp)
    ref = hcs.ref_step(case, rate, state, 1.0)  # no old-flow refusal any more
    new = hc.prepared_coupled_step(ctx, cp.asarray(rate), dstate, 1.0, CUDA)
    hcs.compare(ref, new)
    assert abs(float(ref.route.budget_residual_m3)) <= 1e-13 and abs(float(hcs.host(new.route.budget_residual_m3))) <= 1e-13
    np.testing.assert_array_equal(cp.asarray(dstate.depth_m).get(), depth)  # the input state is not mutated
    np.testing.assert_array_equal(state.depth_m, depth)
    if which == "dry_full_intake":
        assert int(ref.n_complete_runon) >= 1
    else:
        assert int(ref.n_partial_runon) >= 1


def test_the_device_routing_guard_still_rejects_a_genuinely_inconsistent_old_depth():
    """Direct route_step on CuPy arrays: old depth one ulp above the start depth is refused (guard unchanged)."""
    from maple_syrup.routing import route_step

    cp = cupy()
    from test_routing import chain_full, make_graph

    dev_graph = make_graph(chain_full(3), ff=5.0, xp=cp)
    start = np.zeros(dev_graph.shape)
    start[1, 0] = 9.949999999999987e-05
    old = start.copy()
    old[1, 0] = np.nextafter(start[1, 0], 1.0)
    with pytest.raises(RoutingError, match="exceeds depth_start_m"):
        route_step(dev_graph, cp.asarray(start), cp.asarray(old), 1.0, implementation="cuda")


# --- failures: class AND message match the reference --------------------------------------------------------------
def poke(array, index, value):
    out = array.copy()
    out[index] = value
    return out


def _adv():
    case = hcs.build_case(41, kind="valley_masked", model="fixed_ksat", saturated=True, adversarial=True)
    d = {"rate": case.rate_on.copy(), "depth": case.state.depth_m.copy(), "soil": case.state.soil_water_m.copy(),
         "q": case.state.discharge_m2_s.copy(), "dt": 1.0, "ctl": {}, "smax": case.params.storage_max_m,
         "active": case.graph.active.copy(),
         "k": np.asarray(case.graph.conveyance).reshape(case.graph.shape).copy()}
    return case, d


def _courant(d):
    """Consistent old flux but a 16 s step on 5 cm of water: the old-flux Courant number is ~5 > 1."""
    depth = np.where(d["active"], 0.05, 0.3)
    d.update(depth=depth, rate=np.zeros_like(d["rate"]), dt=16.0,
             q=np.where(d["active"], (np.sqrt(depth) * depth) * d["k"], 0.0))


def _huge_dry(d):
    d.update(depth=np.where(d["active"], 1e300, 0.3), rate=np.zeros_like(d["rate"]), q=np.zeros_like(d["q"]))


MUTATIONS = {
    "depth_nan": lambda d: d.update(depth=poke(d["depth"], hcs.ACT, np.nan)),
    "depth_inf": lambda d: d.update(depth=poke(d["depth"], hcs.ACT, np.inf)),
    "depth_negative": lambda d: d.update(depth=poke(d["depth"], hcs.ACT, -1e-3)),
    "soil_nan": lambda d: d.update(soil=poke(d["soil"], hcs.ACT, np.nan)),
    "soil_negative": lambda d: d.update(soil=poke(d["soil"], hcs.ACT, -1e-3)),
    "soil_exceeds_capacity": lambda d: d.update(soil=poke(d["soil"], hcs.ACT, float(d["smax"][hcs.ACT]) * 1.01)),
    "rain_nan": lambda d: d.update(rate=poke(d["rate"], hcs.ACT, np.nan)),
    "rain_negative": lambda d: d.update(rate=poke(d["rate"], hcs.ACT, -1e-6)),
    "rain_on_inactive_cell": lambda d: d.update(rate=poke(d["rate"], hcs.INACT, 1e-5)),
    "old_flux_nan_on_a_run_off_cell": lambda d: d.update(q=poke(d["q"], hcs.ACT, np.nan)),
    "old_flux_negative_on_a_run_off_cell": lambda d: d.update(q=poke(d["q"], hcs.ACT, -1e-9)),
    "old_flux_inconsistent_with_depth": lambda d: d.update(q=poke(d["q"], hcs.ACT, float(d["q"][hcs.ACT]) * 2.0)),
    "huge_finite_depth_overflows_the_old_flux": _huge_dry,
    "courant_rejection_is_recoverable": _courant,
    "dt_zero": lambda d: d.update(dt=0.0),
    "dt_negative": lambda d: d.update(dt=-1.0),
    "dt_nan": lambda d: d.update(dt=float("nan")),
    "dt_bool": lambda d: d.update(dt=True),
    "courant_max_out_of_range": lambda d: d["ctl"].update(courant_max=2.5),
    "iterations_zero": lambda d: d["ctl"].update(bisection_iterations=0),
    "iterations_bool": lambda d: d["ctl"].update(bisection_iterations=True),
    "root_tolerance_zero": lambda d: d["ctl"].update(root_tolerance_m=0.0),
    "bisection_not_converged": lambda d: d["ctl"].update(bisection_iterations=5),
    "depth_float32": lambda d: d.update(depth=d["depth"].astype(np.float32)),
    "depth_wrong_shape": lambda d: d.update(depth=np.zeros((2, 3))),
    "rain_wrong_dtype": lambda d: d.update(rate=d["rate"].astype(np.float32)),
}


def _digest(arrays) -> list[str]:
    return [hashlib.sha256(np.ascontiguousarray(a.get()).tobytes()).hexdigest() for a in arrays]


@pytest.mark.parametrize("mode", ["fused", "split"])
@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_failures_match_the_reference_class_and_message_and_mutate_nothing(name, mode):
    cp = cupy()
    case, d = _adv()
    MUTATIONS[name](d)
    _dev, ctx = prepared(case, mode)
    state = StormState(0.0, d["depth"], d["soil"], d["q"])
    dstate = StormState(0.0, cp.asarray(d["depth"]), cp.asarray(d["soil"]), cp.asarray(d["q"]))
    drate = cp.asarray(d["rate"])
    before = _digest([drate, dstate.depth_m, dstate.soil_water_m, dstate.discharge_m2_s])
    with pytest.raises(Exception) as ref_err:
        hcs.ref_step(case, d["rate"], state, d["dt"], **d["ctl"])
    with pytest.raises(Exception) as new_err:
        hc.prepared_coupled_step(ctx, drate, dstate, d["dt"], StormControl(implementation="cuda", **d["ctl"]))
    assert type(new_err.value) is type(ref_err.value), (name, repr(ref_err.value), repr(new_err.value))
    assert str(new_err.value) == str(ref_err.value), name
    assert before == _digest([drate, dstate.depth_m, dstate.soil_water_m, dstate.discharge_m2_s])


def test_unknown_implementation_and_non_cuda_implementations_are_refused():
    cp = cupy()
    case, d = _adv()
    _dev, ctx = prepared(case)
    dstate = hcs.upload_state(StormState(0.0, d["depth"], d["soil"], d["q"]), cp)
    drate = cp.asarray(d["rate"])
    with pytest.raises(RoutingError, match="implementation must be one of"):
        hc.prepared_coupled_step(ctx, drate, dstate, 1.0, StormControl(implementation="fortran"))
    for impl in ("array", "numba"):
        with pytest.raises(RoutingError, match="CUDA kernels only"):
            hc.prepared_coupled_step(ctx, drate, dstate, 1.0, StormControl(implementation=impl))
    with pytest.raises(StormError):
        hc.prepared_coupled_step(ctx, drate, dstate, 1.0, object())
    with pytest.raises(StormError):
        hc.prepared_coupled_step(ctx, drate, object(), 1.0, CUDA)
    with pytest.raises(hc.CudaHydrologyPreparationError):
        hc.prepared_coupled_step(object(), drate, dstate, 1.0, CUDA)


def test_error_categories_precedence_and_recoverability():
    cp = cupy()

    def run(case, ctx, d, control=CUDA):
        return hc.prepared_coupled_step(
            ctx, cp.asarray(d["rate"]), StormState(0.0, cp.asarray(d["depth"]), cp.asarray(d["soil"]),
                                                   cp.asarray(d["q"])), d["dt"], control)

    case, d = _adv()
    _, ctx = prepared(case)
    _courant(d)
    with pytest.raises(RoutingStepRejected, match="Courant"):
        run(case, ctx, d)
    case, d = _adv()
    _, ctx = prepared(case)
    with pytest.raises(RoutingError, match="bisection did not reach") as info:
        run(case, ctx, d, StormControl(bisection_iterations=5, implementation="cuda"))
    assert not isinstance(info.value, RoutingStepRejected)
    MUTATIONS["soil_nan"](d)  # a column failure outranks a routing-option failure (deferred option error)
    with pytest.raises(InfiltrationError):
        run(case, ctx, d, StormControl(courant_max=9.0, implementation="cuda"))
    with pytest.raises(InfiltrationError):  # ... also an invalid iteration count must never be launched
        run(case, ctx, d, StormControl(bisection_iterations=10**9, implementation="cuda"))
    d["soil"] = case.state.soil_water_m.copy()
    with pytest.raises(RoutingError, match="courant_max"):  # options only: raised after the (clean) column stage
        run(case, ctx, d, StormControl(courant_max=9.0, implementation="cuda"))
    d["dt"] = 0.0  # dt = 0 with clean inputs is the option error, with a bad input the column error
    with pytest.raises(RoutingError, match="dt_s must be finite and > 0"):
        run(case, ctx, d)
    d["depth"] = poke(d["depth"], hcs.ACT, np.nan)
    with pytest.raises(InfiltrationError, match="depth_m must be finite"):
        run(case, ctx, d)


def test_dynamic_array_structure_refusals_mention_the_cupy_contract():
    cp = cupy()
    case, d = _adv()
    _, ctx = prepared(case)
    ok = cp.asarray(d["depth"])
    soil, q = cp.asarray(d["soil"]), cp.asarray(d["q"])
    rate = cp.asarray(d["rate"])
    with pytest.raises(InfiltrationError, match="cupy.ndarray"):
        hc.prepared_coupled_step(ctx, rate, StormState(0.0, d["depth"], soil, q), 1.0, CUDA)  # NumPy: no transfer
    transposed_copy = rate.T.copy().T  # a non-C-contiguous VIEW of the right shape (no cuBLAS needed)
    assert transposed_copy.shape == rate.shape and not transposed_copy.flags.c_contiguous
    for bad in (d["rate"].tolist(), cp.zeros((2, 2)), rate.astype(cp.float32), transposed_copy):
        with pytest.raises(InfiltrationError):
            hc.prepared_coupled_step(ctx, bad, StormState(0.0, ok, soil, q), 1.0, CUDA)
    with pytest.raises(RoutingError, match="cupy.ndarray"):
        hc.prepared_coupled_step(ctx, rate, StormState(0.0, ok, soil, d["q"]), 1.0, CUDA)
    with pytest.raises(RoutingError, match="shape"):
        hc.prepared_coupled_step(ctx, rate, StormState(0.0, ok, soil, cp.zeros((2, 2))), 1.0, CUDA)


def test_previous_discharge_is_checked_everywhere_like_the_prepared_cpu_step():
    cp = cupy()
    case, d = _adv()
    _dev, ctx = prepared(case)
    zero_rain = np.zeros_like(d["rate"])  # partial branch everywhere: the reference recomputes q and ignores q_prev
    nan_q = poke(d["q"], hcs.ACT, np.nan)
    state = StormState(0.0, d["depth"], d["soil"], nan_q)
    hcs.ref_step(case, zero_rain, state, 1.0)  # the reference accepts (laundered)
    for q, match in ((nan_q, "state.discharge_m2_s must be finite"), (poke(d["q"], hcs.ACT, -1e-9), ">= 0"),
                     (poke(d["q"], hcs.INACT, 1e-9), "0 on inactive")):
        with pytest.raises(RoutingError, match=match):
            hc.prepared_coupled_step(ctx, cp.asarray(zero_rain),
                                     StormState(0.0, cp.asarray(d["depth"]), cp.asarray(d["soil"]), cp.asarray(q)),
                                     1.0, CUDA)


# --- outputs, determinism, ownership -----------------------------------------------------------------------------
def test_outputs_are_fresh_and_never_alias_inputs_or_each_other():
    cp = cupy()
    case = hcs.build_case(12, kind="valley", model="pavement_hawkins")
    dev, ctx = prepared(case)
    inputs = [dev.rate_on, dev.state.depth_m, dev.state.soil_water_m, dev.state.discharge_m2_s]
    out = hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, CUDA)
    arrays = hcs.all_arrays(out)
    assert out.state.depth_m is out.route.depth_m and out.state.soil_water_m is out.column.soil_water_m
    assert out.state.discharge_m2_s is out.route.discharge_m2_s
    names = list(arrays)
    for name in names:
        a = arrays[name]
        assert type(a) is cp.ndarray and a.dtype == cp.float64 and a.shape == case.graph.shape, name
        assert a.flags.c_contiguous, name
        for inp in inputs:
            assert not cp.shares_memory(a, inp), name
        for owned in (ctx.conveyance, ctx.column_static, ctx.donor_cell):
            assert not cp.shares_memory(a, owned), name
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if arrays[a] is not arrays[b]:
                assert not cp.shares_memory(arrays[a], arrays[b]), (a, b)
    for key, value in hcs.scalar_fields(out).items():
        assert type(value) is cp.ndarray and value.shape == (), key  # 0-d device views of the fresh packet


def test_a_failure_or_a_later_step_never_changes_an_earlier_result():
    cp = cupy()
    case = hcs.build_case(13, kind="random", model="pavement_hawkins")
    dev, ctx = prepared(case)
    first = hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, CUDA)
    snapshot = {k: v.get().copy() for k, v in hcs.all_arrays(first).items()}
    scalars = {k: hcs.host(v).copy() for k, v in hcs.scalar_fields(first).items()}
    bad = dev.state.depth_m.copy()
    bad[hcs.ACT] = cp.nan
    with pytest.raises(InfiltrationError):
        hc.prepared_coupled_step(ctx, dev.rate_on, StormState(0.0, bad, dev.state.soil_water_m,
                                                              dev.state.discharge_m2_s), 1.0, CUDA)
    hc.prepared_coupled_step(ctx, dev.rate_on, first.state, 0.5, CUDA)
    for k, v in hcs.all_arrays(first).items():
        np.testing.assert_array_equal(v.get(), snapshot[k], err_msg=k)
    for k, v in hcs.scalar_fields(first).items():
        np.testing.assert_array_equal(hcs.host(v), scalars[k], err_msg=k)


@pytest.mark.parametrize("mode", ["fused", "split"])
def test_repeated_steps_are_bitwise_deterministic(mode):
    case = hcs.build_case(21, kind="random", model="pavement_hawkins")
    dev, ctx = prepared(case, mode)
    runs = [hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 0.5, CUDA) for _ in range(5)]
    first = {k: v.get().tobytes() for k, v in hcs.all_arrays(runs[0]).items()}
    scal = {k: hcs.host(v).tobytes() for k, v in hcs.scalar_fields(runs[0]).items()}
    for run in runs[1:]:
        assert first == {k: v.get().tobytes() for k, v in hcs.all_arrays(run).items()}
        assert scal == {k: hcs.host(v).tobytes() for k, v in hcs.scalar_fields(run).items()}


def test_launch_structures_agree_bitwise_on_cellwise_arrays_and_within_the_bound_on_sums():
    case = hcs.build_case(22, kind="random:12x130", model="pavement_hawkins")
    dev = hcs.device_case(case, cupy())
    a = hc.prepared_coupled_step(hc.prepare_cuda_hydrology(dev.graph, dev.params, mode="fused"), dev.rate_on,
                                 dev.state, 1.0, CUDA)
    b = hc.prepared_coupled_step(hc.prepare_cuda_hydrology(dev.graph, dev.params, mode="split"), dev.rate_on,
                                 dev.state, 1.0, CUDA)
    for k, v in hcs.all_arrays(a).items():
        np.testing.assert_array_equal(v.get(), hcs.all_arrays(b)[k].get(), err_msg=k)
    hcs.compare(a, b)


# --- launches, transfers, residency -------------------------------------------------------------------------------
class CountingFunction:
    def __init__(self, real, log, name):
        self.real, self.log, self.name = real, log, name

    def __call__(self, grid, block, args):
        self.log.append((self.name, grid, block))
        return self.real(grid, block, args)

    def __getattr__(self, name):
        return getattr(self.real, name)


@pytest.fixture
def launches(monkeypatch):
    log: list = []
    real = hc._function
    monkeypatch.setattr(hc, "_function", lambda name: CountingFunction(real(name), log, name))
    return log


@pytest.mark.parametrize("width, mode, expected", [(86, "auto", "fused"), (128, "auto", "fused"),
                                                   (129, "auto", "split"), (257, "auto", "split"),
                                                   (86, "split", "split"), (257, "fused", "fused")])
def test_launch_structure_threads_and_counted_transfers(launches, width, mode, expected):
    case = hcs.build_case(23, kind=f"plane:{width}", model="pavement_hawkins")
    dev, ctx = prepared(case, mode)
    assert ctx.mode == expected
    hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, CUDA)  # warm
    launches.clear()
    before = mb.read_transfer_counters()
    hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, CUDA)
    delta = mb.read_transfer_counters().delta(before)
    assert len(launches) == hc.launch_count(ctx.level_bounds, expected) == ctx.summary()["launches_per_step"]
    if expected == "fused":
        assert launches == [("maple_syrup_hydro_fused", (1,), (128,))]
    else:
        names = [name for name, _g, _b in launches]
        assert names[0] == "maple_syrup_hydro_pre" and names[-1] == "maple_syrup_hydro_reduce"
        assert names.count("maple_syrup_hydro_solve_level") == ctx.n_levels
        assert launches[-1][2] == (256,) and all(b == (128,) for _n, _g, b in launches[:-1])
    # one counted packet read; no upload, no full-grid download, no extra synchronising scalar read
    assert delta.device_to_host == 1 and delta.device_to_host_bytes == hc.PACKET_WORDS * 8
    assert delta.host_to_device == 0 and delta.host_to_device_bytes == 0 and delta.scalar_reads == 0


def test_invalid_options_launch_only_the_column_stage(launches):
    case, _d = _adv()
    dev, ctx = prepared(case, "split")
    launches.clear()
    with pytest.raises(RoutingError):
        hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, StormControl(bisection_iterations=10**9,
                                                                                implementation="cuda"))
    assert [n for n, _g, _b in launches] == ["maple_syrup_hydro_pre", "maple_syrup_hydro_reduce"]


def test_in_memory_continuation_stays_on_the_device_and_closes_the_water_budget():
    cp = cupy()
    case = hcs.build_case(24, kind="valley", model="pavement_hawkins")
    dev, ctx = prepared(case)
    area = case.graph.dx_m ** 2
    zero = cp.zeros(case.graph.shape)
    ref_state = case.state
    state = dev.state
    before = mb.read_transfer_counters()
    steps, totals = 30, {"rain": 0.0, "export": 0.0, "drain": 0.0}
    for i in range(steps):
        on = i < 20
        step = hc.prepared_coupled_step(ctx, dev.rate_on if on else zero, state, 1.0, CUDA)
        ref = hcs.ref_step(case, case.rate_on if on else np.zeros(case.graph.shape), ref_state, 1.0)
        state, ref_state = step.state, ref.state
        totals["rain"] += float(step.column.rain_m.sum()) * area
        totals["drain"] += float(step.column.drainage_m.sum()) * area
        totals["export"] += float(step.route.export_m3)
    delta = mb.read_transfer_counters().delta(before)
    assert state.depth_m.dtype == cp.float64 and type(state.depth_m) is cp.ndarray
    np.testing.assert_allclose(state.depth_m.get(), ref_state.depth_m, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(state.soil_water_m.get(), ref_state.soil_water_m, rtol=1e-8, atol=1e-12)
    initial = (float(case.state.depth_m.sum()) + float(case.state.soil_water_m.sum())) * area
    final = (float(state.depth_m.sum()) + float(state.soil_water_m.sum())) * area
    residual = final + totals["drain"] + totals["export"] - initial - totals["rain"]
    assert abs(residual) <= 1e-9 * max(initial + totals["rain"], 1e-30), residual  # sanity only; the CLI owns budgets
    assert delta.device_to_host == steps and delta.host_to_device == 0  # one packet per step; the sums above read none


def test_rejected_attempt_then_halved_step_matches_the_reference():
    """The retry pattern of `storm.evolve`: a Courant rejection leaves the state untouched; dt/2 then succeeds."""
    cp = cupy()
    case, d = _adv()
    _courant(d)
    _dev, ctx = prepared(case)
    state = StormState(0.0, cp.asarray(d["depth"]), cp.asarray(d["soil"]), cp.asarray(d["q"]))
    rate = cp.asarray(d["rate"])
    dt = d["dt"]
    rejected = 0
    while True:
        try:
            new = hc.prepared_coupled_step(ctx, rate, state, dt, CUDA)
            break
        except RoutingStepRejected:
            rejected += 1
            dt *= 0.5
    assert rejected >= 1
    ref = hcs.ref_step(case, d["rate"], StormState(0.0, d["depth"], d["soil"], d["q"]), dt)
    hcs.compare(ref, new)
    np.testing.assert_array_equal(state.depth_m.get(), d["depth"])


# --- context: ownership, lifetime, metadata, preparation ---------------------------------------------------------
def test_context_owns_device_copies_and_follows_neither_later_mutation_nor_keeps_the_graph_alive():
    cp = cupy()
    case = hcs.build_case(15, kind="random", model="fixed_ksat")
    dev, ctx = prepared(case)
    for name in hc._OWNED:
        array = getattr(ctx, name)
        assert array.flags.c_contiguous, name
        for source in (dev.graph.conveyance, dev.graph.donor_position, dev.graph.level_order,
                       dev.params.ksat_m_per_s, dev.params.storage_max_m):
            assert not cp.shares_memory(array, source), name
    assert not any(v is dev.graph or v is dev.params for v in vars(ctx).values())
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.n_cells = 1
    expected = hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, CUDA)
    assert float(expected.column.intake_m.sum()) > 0.0
    dev.params.ksat_m_per_s[...] = 0.0  # the caller's device array can be mutated in place (CuPy cannot freeze)
    dev.graph.conveyance[...] = 0.0
    still = hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, CUDA)
    hcs.compare(expected, still, exact_arrays=True)  # the context did not follow the mutated source
    case2 = hcs.build_case(15, kind="random", model="fixed_ksat")
    dev2 = hcs.device_case(case2, cp)
    ctx2 = hc.prepare_cuda_hydrology(dev2.graph, dev2.params)
    graph_ref, params_ref = weakref.ref(dev2.graph), weakref.ref(dev2.params)
    del dev2
    gc.collect()
    assert graph_ref() is None and params_ref() is None, "the context kept the graph or parameters alive"
    assert ctx2.n_active > 0


def test_tampered_context_metadata_is_refused_before_any_launch(launches):
    cp = cupy()
    case = hcs.build_case(16, kind="valley", model="fixed_ksat")
    dev, ctx = prepared(case)
    launches.clear()
    for name in ("donor_cell", "level_order", "level_bounds_device", "column_static", "conveyance", "active"):
        array = getattr(ctx, name)
        forged = dataclasses.replace(ctx, **{name: cp.zeros_like(array)})
        with pytest.raises(RoutingError, match="owned device arrays"):
            hc.prepared_coupled_step(forged, dev.rate_on, dev.state, 1.0, CUDA)
    forged = dataclasses.replace(ctx, conveyance=ctx.conveyance.astype(cp.float32))
    with pytest.raises(RoutingError):
        hc.prepared_coupled_step(forged, dev.rate_on, dev.state, 1.0, CUDA)
    assert launches == []


def test_context_summary_and_provenance_on_device():
    case = hcs.build_case(17, kind="valley", model="fixed_ksat")
    _dev, ctx = prepared(case)
    s = ctx.summary()
    assert s["shape"] == list(case.graph.shape) and s["n_active"] == case.graph.n_active
    assert s["n_levels"] == case.graph.n_levels and s["max_level_width"] == case.graph.max_level_width
    assert s["static_bytes"] == ctx.nbytes() > 0 and s["device_resident"] and s["fastmath"] is False
    assert s["graph_input_sha256"] == case.graph.input_sha256 and s["numba_required"] is False
    assert s["host_to_device_bytes"] >= s["static_bytes"] and s["device_to_host_bytes"] > 0
    assert s["kernel_load_s"] >= 0.0 and ctx.preparation_s > 0.0
    for name, attrs in ctx.kernel_attributes.items():
        assert attrs["max_threads_per_block"] >= 128 and "num_regs" in attrs, name
    info = hc.kernel_provenance()
    assert info["cupy"] and info["device"]["name"] and info["fastmath"] is False


def test_every_prepare_returns_an_independent_context_without_a_hidden_cache():
    case = hcs.build_case(18, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cupy())
    a = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    b = hc.prepare_cuda_hydrology(dev.graph, dev.params)
    assert a is not b and a.donor_cell.data.ptr != b.donor_cell.data.ptr  # explicit contexts, no hidden cache


def _forge_graph(dev, **changes):
    return dataclasses.replace(dev.graph, **changes)


def _forge_params(dev, **changes):
    return dataclasses.replace(dev.params, **changes)


def test_preparation_refuses_wrong_namespaces_types_and_masks(launches):
    cp = cupy()
    case = hcs.build_case(19, kind="valley_masked", model="pavement_hawkins")
    dev = hcs.device_case(case, cp)
    with pytest.raises(hc.CudaHydrologyPreparationError, match="RoutingGraph"):
        hc.prepare_cuda_hydrology(object(), dev.params)
    with pytest.raises(hc.CudaHydrologyPreparationError, match="ColumnParameters"):
        hc.prepare_cuda_hydrology(dev.graph, object())
    with pytest.raises(hc.CudaHydrologyPreparationError, match="CuPy graph"):
        hc.prepare_cuda_hydrology(case.graph, dev.params)  # host graph
    with pytest.raises(hc.CudaHydrologyPreparationError, match="CuPy graph"):
        hc.prepare_cuda_hydrology(dev.graph, case.params)  # host parameters
    other = hcs.device_case(hcs.build_case(19, kind="valley", model="pavement_hawkins"), cp)
    with pytest.raises(hc.CudaHydrologyPreparationError, match="shape"):
        hc.prepare_cuda_hydrology(dev.graph, other.params)
    from maple_syrup.infiltration import column_parameters

    shape = case.graph.shape
    all_true = column_parameters(model="fixed_ksat", ksat_m_per_s=cp.full(shape, 1e-6), suction_m=cp.zeros(shape),
                                 drainage_parameter=cp.zeros(shape), theta_sat=cp.full(shape, 0.4),
                                 soil_thickness_m=cp.full(shape, 0.3))
    with pytest.raises(hc.CudaHydrologyPreparationError, match="active_mask differs"):
        hc.prepare_cuda_hydrology(dev.graph, all_true)
    with pytest.raises(hc.CudaHydrologyPreparationError, match="mode"):
        hc.prepare_cuda_hydrology(dev.graph, dev.params, mode="speculative")
    assert launches == []


def _bad_value(array, index, value):
    out = array.copy()
    out[index] = value
    return out


@pytest.mark.parametrize("name, which, make, match", [
    ("ksat_nan", "p", lambda c, cp: {"ksat_m_per_s": _bad_value(c.params.ksat_m_per_s, hcs.ACT, cp.nan)}, "finite"),
    ("suction_negative", "p", lambda c, cp: {"suction_m": _bad_value(c.params.suction_m, hcs.ACT, -1.0)}, ">= 0"),
    ("theta_zero", "p", lambda c, cp: {"theta_sat": _bad_value(c.params.theta_sat, hcs.ACT, 0.0)}, "theta_sat"),
    ("thickness_zero", "p", lambda c, cp: {"soil_thickness_m": _bad_value(c.params.soil_thickness_m, hcs.ACT, 0.0)},
     "soil_thickness_m"),
    ("smax_not_the_product", "p", lambda c, cp: {"storage_max_m": c.params.storage_max_m * 1.0001},
     "storage_max_m"),
    ("lambda_missing", "p", lambda c, cp: {"lambda_m_per_s": None}, "lambda"),
    ("lambda_zero", "p", lambda c, cp: {"lambda_m_per_s": cp.zeros(c.graph.shape)}, "lambda"),
    ("ksat_float32", "p", lambda c, cp: {"ksat_m_per_s": c.params.ksat_m_per_s.astype(cp.float32)}, "float64"),
    ("conveyance_negative", "g", lambda c, cp: {"conveyance": -c.graph.conveyance - 1.0}, "conveyance"),
    ("conveyance_lo_differs", "g", lambda c, cp: {"conveyance_lo": c.graph.conveyance_lo * 1.5}, "conveyance_lo"),
    ("order_not_a_permutation", "g", lambda c, cp: {"level_order": cp.zeros_like(c.graph.level_order)},
     "permutation"),
    ("donor_index_out_of_range", "g", lambda c, cp: {"donor_position": c.graph.donor_position + c.graph.n_active},
     "outside"),
    ("donor_in_same_or_later_level", "g",
     lambda c, cp: {"donor_position": cp.full_like(c.graph.donor_position, c.graph.n_active - 1),
                    "donor_mask": cp.ones_like(c.graph.donor_mask)}, "earlier"),
    ("donor_mask_incomplete", "g", lambda c, cp: {"donor_mask": cp.zeros_like(c.graph.donor_mask)}, "donors"),
    ("level_bounds_wrong_end", "g", lambda c, cp: {"level_bounds": (*c.graph.level_bounds[:-1],
                                                                    c.graph.level_bounds[-1] + 1)}, "level_bounds"),
    ("level_bounds_not_a_tuple", "g", lambda c, cp: {"level_bounds": list(c.graph.level_bounds)}, "level_bounds"),
    ("outlet_wrong", "g", lambda c, cp: {"outlet_flat": cp.zeros_like(c.graph.outlet_flat)}, "outlet"),
    ("dx_nan", "g", lambda c, cp: {"dx_m": float("nan")}, "dx_m"),
    ("graph_shape_bool", "g", lambda c, cp: {"shape": (True, True)}, "shape"),
])
def test_preparation_refuses_unsafe_static_data_before_any_launch(launches, name, which, make, match):
    cp = cupy()
    case = hcs.build_case(18, kind="valley_masked", model="pavement_hawkins")
    dev = hcs.device_case(case, cp)
    changes = make(dev, cp)
    graph = _forge_graph(dev, **changes) if which == "g" else dev.graph
    params = _forge_params(dev, **changes) if which == "p" else dev.params
    with pytest.raises(hc.CudaHydrologyPreparationError, match=match):
        hc.prepare_cuda_hydrology(graph, params)
    assert launches == [], name


def test_forged_donor_slot_permutation_is_refused_like_the_routing_context():
    cp = cupy()
    case = hcs.build_case(20, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    pos, mask = dev.graph.donor_position.get().copy(), dev.graph.donor_mask.get().copy()
    two = np.flatnonzero(mask.sum(axis=0) >= 2)
    assert two.size, "fixture must contain a receiver with several donors"
    p2 = int(two[0])
    s1, s2 = (int(v) for v in np.flatnonzero(mask[:, p2])[:2])
    pos[[s1, s2], p2] = pos[[s2, s1], p2]
    mask[[s1, s2], p2] = mask[[s2, s1], p2]
    bad = dataclasses.replace(dev.graph, donor_position=cp.asarray(pos), donor_mask=cp.asarray(mask))
    with pytest.raises(hc.CudaHydrologyPreparationError, match="slot"):
        hc.prepare_cuda_hydrology(bad, dev.params)
    hc.prepare_cuda_hydrology(dev.graph, dev.params)  # the unforged builder graph is accepted


def test_compile_failure_is_unavailable_without_fallback_and_before_any_transfer(monkeypatch):
    cp = cupy()
    case = hcs.build_case(18, kind="valley", model="fixed_ksat")
    dev = hcs.device_case(case, cp)
    bad_module = cp.RawModule(code='extern "C" __global__ void x(double* p) { this is not valid CUDA C++ }',
                              options=hc.COMPILE_OPTIONS, backend="nvrtc")
    monkeypatch.setattr(hc, "_MODULE", bad_module)
    monkeypatch.setattr(hc, "_FUNCTIONS", {})
    before = mb.read_transfer_counters()
    with pytest.raises(hc.CudaUnavailableError, match="compile"):
        hc.prepare_cuda_hydrology(dev.graph, dev.params)
    assert mb.read_transfer_counters().delta(before).device_to_host == 0, "validation ran after a failed compile"
    monkeypatch.undo()
    hc.prepare_cuda_hydrology(dev.graph, dev.params)  # recovery with the real module


# --- streams and devices ------------------------------------------------------------------------------------------
def test_nondefault_stream_matches_default_stream():
    cp = cupy()
    case = hcs.build_case(25, kind="random", model="pavement_hawkins")
    dev, ctx = prepared(case)
    ref = hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, CUDA)
    dev2 = hcs.device_case(case, cp)
    with cp.cuda.Stream(non_blocking=True) as stream:
        ctx2 = hc.prepare_cuda_hydrology(dev2.graph, dev2.params)  # prepares on this stream too
        rate = cp.asarray(case.rate_on)
        state = hcs.upload_state(case.state, cp)
        got = hc.prepared_coupled_step(ctx2, rate, state, 1.0, CUDA)
        stream.synchronize()
    for k, v in hcs.all_arrays(ref).items():
        np.testing.assert_array_equal(v.get(), hcs.all_arrays(got)[k].get(), err_msg=k)
    # a context prepared on the default stream is usable on another stream (static data was synchronized)
    with cp.cuda.Stream(non_blocking=True) as stream:
        rate = cp.asarray(case.rate_on)
        state = hcs.upload_state(case.state, cp)
        again = hc.prepared_coupled_step(ctx, rate, state, 1.0, CUDA)
        stream.synchronize()
    hcs.compare(ref, again)


def test_wrong_current_device_and_wrong_device_inputs_are_refused_without_migration(launches):
    cp = cupy()
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("only one CUDA device is visible; the multi-device refusals are not exercised")
    with cp.cuda.Device(0):
        case = hcs.build_case(26, kind="valley", model="fixed_ksat")
        dev, ctx = prepared(case)
        hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, CUDA)
        with cp.cuda.Device(1):
            far_rate = cp.asarray(case.rate_on)
            far_state = hcs.upload_state(case.state, cp)
            with pytest.raises(RoutingError, match="device"):
                hc.prepared_coupled_step(ctx, far_rate, far_state, 1.0, CUDA)
            # preparation reports structural/device problems as CudaHydrologyPreparationError by contract, the step
            # (dynamic arrays / current device) as RoutingError / InfiltrationError
            with pytest.raises(hc.CudaHydrologyPreparationError, match="device"):
                hc.prepare_cuda_hydrology(dev.graph, dev.params)
        launches.clear()
        assert int(cp.cuda.Device().id) == 0
        with pytest.raises(InfiltrationError, match=r"depth_m lives on CUDA device 1"):
            hc.prepared_coupled_step(ctx, dev.rate_on, far_state, 1.0, CUDA)
        assert launches == []
