"""Phase 7h: prepared compiled hydrology against the reference `storm.coupled_step(..., implementation="numba")`.

The reference is the oracle and is never modified: every comparison runs it on the same inputs. Booleans, counts,
strings and dtypes/shapes must be identical. Floats are compared BITWISE where the column arithmetic is
libm-independent (see `test_saturated_regime_is_bitwise`: Smith-Parlange argument >> 40 so expm1 is exactly -1)
and otherwise within the declared water bound (rtol 2e-12, atol 1e-14), which only exists because the prepared
kernels call libm `expm1`/`pow` where the reference uses NumPy's loops. Failure cases compare exception CLASS and
message with the reference. Differences that are deliberately STRICTER than the reference are tested separately.
Nothing here was run by its author (file-only tools); Codex records actual results and any tightening."""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest
from test_routing import chain_full, make_graph, random_full, valley_full

pytest.importorskip("maple")

from maple_syrup import hydrology_numba as hn
from maple_syrup.infiltration import (
    ColumnStep,
    InfiltrationError,
    column_parameters,
    column_step,
    initial_soil_water_m,
)
from maple_syrup.routing import RoutingError, RoutingStepRejected
from maple_syrup.routing_numba import numba_available
from maple_syrup.storm import (
    CoupledStep,
    StormControl,
    StormError,
    StormState,
    coupled_step,
    initial_state,
)

pytestmark = pytest.mark.skipif(not numba_available(), reason="Numba not installed; no claim made")

WATER_RTOL = 2.0e-12
WATER_ATOL = 1.0e-14
CONTROL = StormControl(implementation="numba")
ACT = (2, 2)  # an active cell of the masked 6 x 5 valley
INACT = (5, 0)  # its top-west corner, masked out


# --- builders ----------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Case:
    graph: object
    params: object
    state: StormState
    rate_on: np.ndarray
    ctx: object


def terrain(kind: str, rng):
    if kind == "random":
        return random_full(rng, 11, 8), rng.uniform(5.0, 30.0, (11, 8)), None
    if kind == "valley":
        return valley_full(8, 7), 5.0, None
    if kind == "valley_masked":  # nothing drains into the two top corners, so masking them is a valid graph
        active = np.ones((6, 5), dtype=bool)
        active[-1, 0] = active[-1, -1] = False
        return valley_full(6, 5), 5.0, active
    raise AssertionError(kind)


def build_case(seed: int, *, kind: str = "random", model: str = "pavement_hawkins", saturated: bool = False,
               adversarial: bool = False, rain_mm_h: float = 45.0) -> Case:
    rng = np.random.default_rng(seed)
    z, ff, active = terrain(kind, rng)
    graph = make_graph(z, ff=ff, active=active)
    shape = graph.shape

    def uni(lo, hi):
        return rng.uniform(lo, hi, shape)

    # Reference-accepted unsaturated forcing (see the module note below `build_case`): small suction keeps the
    # Smith-Parlange argument x = S / ((psi + h)(theta_sat - theta)) in ~[0.5, 100] (non-trivial, not saturated).
    fields = {"suction_m": uni(0.0, 1e-3), "drainage_parameter": uni(0.0, 0.5), "theta_sat": uni(0.3, 0.5),
              "soil_thickness_m": uni(0.1, 0.5)}
    ksat = uni(1e-8, 3e-6)
    if saturated:  # near-full columns, no drainage: capacity is exactly Ksat (see the module docstring)
        ksat = np.full(shape, 1e-7)
        fields["suction_m"] = np.full(shape, 0.01)
        fields["drainage_parameter"] = np.zeros(shape)
    if model == "pavement_hawkins":
        fields["pavement_cover_fraction"] = uni(0.8, 1.0)  # lambda ~ 6.8e-6 .. 1.2e-5 m/s: nonlinear Hawkins K(r)
    params = column_parameters(model=model, ksat_m_per_s=ksat, active_mask=graph.active.copy(), **fields)
    if saturated:
        soil = params.storage_max_m - 1e-6
    else:
        soil = initial_soil_water_m(params, uni(0.05, 0.95) * params.theta_sat)
    if adversarial:
        depth = np.where(graph.active, 1e-3, 0.3)
        rate = np.where(graph.active, 1e-5, 0.0)
    else:
        if saturated:  # tiny capacity: intake never equals the available water, sparse wetting is safe
            wet = uni(0.0, 3e-3) * (rng.random(shape) > 0.3)
        else:  # wet floor: intake (capacity ~ K ~ 1e-5 m) stays far below the stored water, see the reproducer test
            wet = uni(1e-3, 3e-3)
        depth = np.where(graph.active, wet, 0.3)
        scale = np.where(graph.active, uni(0.5, 1.5) * (rng.random(shape) > 0.2), 0.0)
        rate = scale * (rain_mm_h / 3.6e6)
        if saturated:
            rate = np.where(graph.active, 1e-5, 0.0)
        else:
            # Positive rain >= 6e-5 m/s (216 mm/h) on ~80% of active cells, exactly zero on the rest. Bound: the
            # Hawkins K <= lambda <= 1.2e-5 m/s (cover >= 0.8) or Ksat <= 3e-6, and capacity = K / (1 - exp(-x))
            # <= ~2.6 K for x >= 0.48 (suction <= 1e-3, h <= 1e-2, thickness >= 0.1, theta/(theta_sat-theta) >=
            # 0.0526), i.e. <= ~3.1e-5 m/s < 6e-5, so J < P on every raining cell: pure no run-on with
            # h* = (h + P - J) + 0 >= h + 2.9e-5 >> ulp, never the one-ulp hpre/h* guard. Zero-rain cells have
            # P = 0, so hpre = h - J and h* = (h + 0) - J are the SAME FP64 expression (partial branch, exact).
            rate = np.where(scale > 0.0, uni(1.0, 1.5) * 6e-5, 0.0)
    state = initial_state(graph, depth, soil)
    return Case(graph, params, state, rate, hn.prepare_hydrology(graph, params))


# --- comparison --------------------------------------------------------------------------------------------
def compare(ref, new, *, exact: bool, path: str = "step") -> None:
    if dataclasses.is_dataclass(ref) and not isinstance(ref, type):
        assert type(new) is type(ref), path
        for f in dataclasses.fields(ref):
            compare(getattr(ref, f.name), getattr(new, f.name), exact=exact, path=f"{path}.{f.name}")
        return
    if ref is None:
        assert new is None, path
        return
    if isinstance(ref, np.ndarray):
        assert type(new) is np.ndarray, path
        assert new.dtype == ref.dtype and new.shape == ref.shape, path
        if exact or ref.dtype.kind != "f":
            np.testing.assert_array_equal(new, ref, err_msg=path)
        else:
            np.testing.assert_allclose(new, ref, rtol=WATER_RTOL, atol=WATER_ATOL, err_msg=path)
        return
    assert type(new) is type(ref), f"{path}: {type(new).__name__} vs {type(ref).__name__}"
    if isinstance(ref, (bool, str, int, np.integer)) or exact:
        assert new == ref, path
    else:
        assert new == pytest.approx(ref, rel=WATER_RTOL, abs=WATER_ATOL), path


def run_pair(case: Case, rates, dts, *, control: StormControl = CONTROL, exact: bool = False) -> dict[str, float]:
    """Both implementations on the SAME input state each step (the reference's state is advanced), all fields
    compared. Returns branch/flux totals of the reference for coverage assertions."""
    state = case.state
    totals = {"no_runon": 0, "partial": 0, "complete": 0, "return_m": 0.0, "drain_m": 0.0, "intake_m": 0.0}
    for rate, dt in zip(rates, dts, strict=True):
        ref = coupled_step(case.graph, case.params, rate, state, dt, control)
        new = hn.prepared_coupled_step(case.ctx, rate, state, dt, control)
        assert isinstance(new, CoupledStep)
        compare(ref, new, exact=exact)
        totals["no_runon"] += int(ref.n_no_runon)
        totals["partial"] += int(ref.n_partial_runon)
        totals["complete"] += int(ref.n_complete_runon)
        totals["return_m"] += float(ref.column.saturation_return_m.sum())
        totals["drain_m"] += float(ref.column.drainage_m.sum())
        totals["intake_m"] += float(ref.column.intake_m.sum())
        state = ref.state
    return totals


# --- differential: wet / dry / recession / run-on / saturation / drainage --------------------------------------
@pytest.mark.parametrize("kind", ["valley", "random", "valley_masked"])
def test_saturated_regime_is_bitwise(kind):
    """Near-full columns: x = S / ((psi + h)(theta_sat - theta)) >> 40, so expm1(-x) is exactly -1 in NumPy and in
    libm, capacity is exactly Ksat, and the whole step is IEEE-exact. Rain on, saturation return, then recession.
    Every field must be BITWISE equal (the only libm call left is the pow of a threshold test)."""
    case = build_case(11, kind=kind, model="fixed_ksat", saturated=True)
    zero = np.zeros(case.graph.shape)
    rates = [case.rate_on] * 20 + [zero] * 15
    totals = run_pair(case, rates, [1.0] * 35, exact=True)
    assert totals["no_runon"] > 0 and totals["partial"] > 0
    assert totals["return_m"] > 0.0 and totals["intake_m"] > 0.0


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("kind", ["random", "valley_masked", "valley"])
def test_hawkins_random_columns_match_within_the_water_bound(seed, kind):
    case = build_case(seed, kind=kind, model="pavement_hawkins")
    zero = np.zeros(case.graph.shape)
    rates = [case.rate_on] * 14 + [zero] * 10
    dts = ([1.0, 0.5, 0.25, 1.0 / 1024.0, 1.0, 1.0, 0.5] * 4)[:24]
    totals = run_pair(case, rates, dts)
    assert totals["drain_m"] > 0.0 and totals["intake_m"] > 0.0


@pytest.mark.parametrize("seed", [4, 5])
def test_fixed_ksat_unsaturated_columns_match_within_the_water_bound(seed):
    case = build_case(seed, kind="random", model="fixed_ksat")
    zero = np.zeros(case.graph.shape)
    run_pair(case, [case.rate_on] * 12 + [zero] * 8, [1.0] * 20)


@pytest.mark.parametrize("dt, courant, iterations, root_tol", [
    (1.0, 1.0, 40, 1e-11), (0.25, 0.5, 60, 1e-12), (1.0 / 1024.0, 2.0, 30, 1e-9), (0.5, 1, np.int64(45), 1e-10),
])
def test_dt_courant_and_root_controls(dt, courant, iterations, root_tol):
    case = build_case(6, kind="valley", model="pavement_hawkins")
    control = StormControl(courant_max=courant, bisection_iterations=iterations, root_tolerance_m=root_tol,
                           implementation="numba")
    run_pair(case, [case.rate_on] * 6, [dt] * 6, control=control)


def _chain_case(depth_value, *, rate_value=0.0, ksat=1e-6, n=3):
    """Full (S = Smax) fixed-Ksat columns on a chain: Smith-Parlange capacity is exactly Ksat."""
    graph = make_graph(chain_full(n), ff=5.0)
    shape = graph.shape
    full = lambda v: np.full(shape, float(v))
    params = column_parameters(model="fixed_ksat", ksat_m_per_s=full(ksat), suction_m=full(0.01),
                               drainage_parameter=full(0.0), theta_sat=full(0.4), soil_thickness_m=full(0.3),
                               active_mask=graph.active.copy())
    soil = initial_soil_water_m(params, full(0.4))
    assert np.array_equal(soil, params.storage_max_m)
    state = initial_state(graph, full(depth_value), soil)
    return Case(graph, params, state, full(rate_value), hn.prepare_hydrology(graph, params))


@pytest.mark.parametrize("dt", [1.0, 0.25, 1.0 / 1024.0])
def test_near_boundary_capacities_and_runon_branch_ties(dt):
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
        case = _chain_case(depth, rate_value=rate, ksat=ksat)
        ref = coupled_step(case.graph, case.params, case.rate_on, case.state, dt, CONTROL)
        counts = {"no_runon": int(ref.n_no_runon), "partial": int(ref.n_partial_runon),
                  "complete": int(ref.n_complete_runon)}
        assert counts[branch] == case.graph.n_active and sum(counts.values()) == case.graph.n_active, (name, counts)
        new = hn.prepared_coupled_step(case.ctx, case.rate_on, case.state, dt, CONTROL)
        compare(ref, new, exact=True, path=name)


@pytest.mark.parametrize("soil_value", [0.0, 1e-320])
def test_limiting_intake_on_a_dry_column_stays_accepted(soil_value):
    """S = 0 (1 - exp(-x) == 0) and S = 1e-320 (capacity = K / 1e-318 overflows to +inf, min() then selects the
    finite available water) are valid limits the reference accepts: so must the prepared step."""
    case = build_case(7, kind="valley_masked", model="fixed_ksat", saturated=True, adversarial=True)
    soil = np.full(case.graph.shape, soil_value)
    state = initial_state(case.graph, case.state.depth_m, soil)
    ref = coupled_step(case.graph, case.params, case.rate_on, state, 1.0, CONTROL)
    new = hn.prepared_coupled_step(case.ctx, case.rate_on, state, 1.0, CONTROL)
    compare(ref, new, exact=True)
    available = ref.column.rain_m + state.depth_m
    np.testing.assert_array_equal(ref.column.intake_m[case.graph.active], available[case.graph.active])


def test_masked_cells_keep_their_inventory_and_report_zero_flux():
    case = build_case(8, kind="valley_masked", model="pavement_hawkins")
    new = hn.prepared_coupled_step(case.ctx, case.rate_on, case.state, 1.0, CONTROL)
    inactive = ~case.graph.active
    assert inactive.any()
    np.testing.assert_array_equal(new.state.depth_m[inactive], case.state.depth_m[inactive])
    np.testing.assert_array_equal(new.state.soil_water_m[inactive], case.state.soil_water_m[inactive])
    for array in (new.state.discharge_m2_s, new.route.velocity_m_s, new.route.face_volume_m3,
                  new.route.inflow_m2_s, new.column.intake_m, new.column.drainage_m, new.route.old_discharge_m2_s):
        assert not np.any(array[inactive])
    compare(coupled_step(case.graph, case.params, case.rate_on, case.state, 1.0, CONTROL), new, exact=False)


# --- former reference roundoff inconsistency (fixed by the coherent branch arithmetic; seeded triggers kept) --------------------------------
def _find_inconsistent_wet_depth(rain_m: float = 1e-5):
    """Deterministic search of a small depth h for which, on a dry column (J = A, no saturation return),
    hpre = h - max(J - P, 0) is a ulp ABOVE h* = (h + P - J) + 0 = 0 in FP64: the reference's own old-flow depth
    then exceeds its own start depth. Pure scalar arithmetic, the same operations as storm/infiltration."""
    rng = np.random.default_rng(2024)
    for h in rng.uniform(1e-6, 1e-3, 4000):
        h = np.float64(h)
        available = h + np.float64(rain_m)
        intake = available  # dry column: capacity is unbounded, J = A
        hpre = max(h - max(intake - np.float64(rain_m), 0.0), 0.0)
        hstar = (available - intake) + 0.0
        if hpre > hstar:
            return float(h)
    raise AssertionError("no inconsistent depth found; the roundoff premise of this reproducer changed")


def _run_coherent_branch(case, depth, soil, rate):
    """Reference and prepared step on the SAME input: both must run (no old-flow refusal), agree at the unchanged bounds, and
    mutate nothing; the unchanged per-cell/global balances (validate=True, route checks) pass inside both."""
    state = initial_state(case.graph, depth, soil)
    before = [a.copy() for a in (depth, soil, rate, state.discharge_m2_s)]
    ref = coupled_step(case.graph, case.params, rate, state, 1.0, CONTROL)
    new = hn.prepared_coupled_step(case.ctx, rate, state, 1.0, CONTROL)
    compare(ref, new, exact=False)
    for a, b in zip(before, (depth, soil, rate, state.discharge_m2_s), strict=True):
        np.testing.assert_array_equal(a, b)
    assert abs(float(ref.route.budget_residual_m3)) <= 1e-13
    return ref, new


def test_former_roundoff_trigger_now_runs_the_coherent_complete_branch_identically_in_both_paths():
    """Formerly refused (hpre one ulp above h*): a fully-infiltrating wet cell. The complete branch sets the old-flow depth to 0
    exactly and the column depth shares that arithmetic, so the step runs; reference and prepared agree."""
    h = _find_inconsistent_wet_depth()
    case = _chain_case(0.0, rate_value=1e-5)
    depth = np.zeros(case.graph.shape)
    depth[1, 0] = h
    ref, _ = _run_coherent_branch(case, depth, np.zeros(case.graph.shape), np.full(case.graph.shape, 1e-5))
    assert int(ref.n_complete_runon) >= 1 and float(ref.state.depth_m[1, 0]) == 0.0


def test_the_routing_guard_still_rejects_a_genuinely_inconsistent_old_depth():
    """The guard is unchanged: an old-flow depth ABOVE the start depth (one ulp is enough) is refused by route_step, with the same
    class and message in the array sweep, and nothing is mutated."""
    from maple_syrup.routing import route_step

    case = _chain_case(0.0, rate_value=1e-5)
    start = np.zeros(case.graph.shape)
    start[1, 0] = 9.949999999999987e-05
    old = start.copy()
    old[1, 0] = np.nextafter(start[1, 0], 1.0)
    before = (start.copy(), old.copy())
    with pytest.raises(RoutingError, match="exceeds depth_start_m") as err:
        route_step(case.graph, start, old, 1.0)
    assert not isinstance(err.value, RoutingStepRejected)
    np.testing.assert_array_equal(start, before[0])
    np.testing.assert_array_equal(old, before[1])


def _find_partial_positive_rain_depth(rain_m: float = 1e-5, ksat: float = 3e-5):
    """Seeded search for the BROADER domain of the same guard: partial intake with positive rain (P < J < A).
    Capacity is exactly Ksat (soil = Smax - 1e-4 so x ~ 1e4 >> 40, no saturation return), J = Ksat dt = 3e-5,
    hpre = h - (J - P) and h* = (h + P) - J are different FP64 expressions and can differ by an ulp for any h."""
    rng = np.random.default_rng(77)
    for h in rng.uniform(1e-3, 3e-3, 4000):
        h = np.float64(h)
        available = h + np.float64(rain_m)
        intake = min(available, np.float64(ksat))
        hpre = max(h - max(intake - np.float64(rain_m), 0.0), 0.0)
        if hpre > (available - intake) + 0.0:
            return float(h)
    raise AssertionError("no inconsistent partial-intake depth found; the roundoff premise changed")


def test_former_partial_roundoff_trigger_now_runs_the_coherent_partial_branch_identically_in_both_paths():
    """Formerly refused for P > 0 partial intake at ordinary depths. Retained depth h - (J - P) is now the single arithmetic
    of the column depth and the old-flow depth: the step runs, reference and prepared agree, nothing is mutated."""
    h = _find_partial_positive_rain_depth()
    case = _chain_case(0.0, rate_value=1e-5, ksat=3e-5)
    depth = np.zeros(case.graph.shape)
    depth[1, 0] = h
    soil = case.params.storage_max_m - 1e-4
    ref, _ = _run_coherent_branch(case, depth, soil, np.full(case.graph.shape, 1e-5))
    assert int(ref.n_partial_runon) >= 1


# --- column alone ------------------------------------------------------------------------------------------
def test_prepared_column_step_matches_column_step_and_dt_zero_is_the_identity():
    case = build_case(9, kind="random", model="pavement_hawkins")
    h, s = case.state.depth_m, case.state.soil_water_m
    ref = column_step(case.params, h, s, case.rate_on, 0.5)
    new = hn.prepared_column_step(case.ctx, h, s, case.rate_on, 0.5)
    assert isinstance(new, ColumnStep)
    compare(ref, new, exact=False)
    zero_ref = column_step(case.params, h, s, case.rate_on, 0.0)
    zero_new = hn.prepared_column_step(case.ctx, h, s, case.rate_on, 0.0)
    compare(zero_ref, zero_new, exact=True)
    assert not np.shares_memory(zero_new.depth_m, h)
    bad = h.copy()
    bad[1, 1] = np.nan
    with pytest.raises(InfiltrationError) as ref_err:
        column_step(case.params, bad, s, case.rate_on, 0.0)  # inputs are validated even for dt = 0
    with pytest.raises(InfiltrationError) as new_err:
        hn.prepared_column_step(case.ctx, bad, s, case.rate_on, 0.0)
    assert str(new_err.value) == str(ref_err.value)


# --- failures: class AND message match the reference ----------------------------------------------------------
def poke(array, index, value):
    out = array.copy()
    out[index] = value
    return out


def _adv():
    case = build_case(41, kind="valley_masked", model="fixed_ksat", saturated=True, adversarial=True)
    return case, {"rate": case.rate_on.copy(), "depth": case.state.depth_m.copy(), "soil": case.state.soil_water_m.copy(),
                  "q": case.state.discharge_m2_s.copy(), "dt": 1.0, "control": CONTROL,
                  "smax": case.params.storage_max_m, "active": case.graph.active.copy(),
                  "k": np.asarray(case.graph.conveyance).reshape(case.graph.shape).copy()}


def _courant(d):
    """Consistent old flux but a 16 s step on 5 cm of water: the old-flux Courant number is ~5 > 1."""
    depth = np.where(d["active"], 0.05, 0.3)
    d.update(depth=depth, rate=np.zeros_like(d["rate"]), dt=16.0,
             q=np.where(d["active"], (np.sqrt(depth) * depth) * d["k"], 0.0))


def _huge_dry(d):
    """Finite 1e300 m of water: k h^{3/2} overflows to +inf in the partial run-on branch."""
    d.update(depth=np.where(d["active"], 1e300, 0.3), rate=np.zeros_like(d["rate"]), q=np.zeros_like(d["q"]))


MUTATIONS = {
    "depth_nan": lambda d: d.update(depth=poke(d["depth"], ACT, np.nan)),
    "depth_inf": lambda d: d.update(depth=poke(d["depth"], ACT, np.inf)),
    "depth_negative": lambda d: d.update(depth=poke(d["depth"], ACT, -1e-3)),
    "soil_nan": lambda d: d.update(soil=poke(d["soil"], ACT, np.nan)),
    "soil_negative": lambda d: d.update(soil=poke(d["soil"], ACT, -1e-3)),
    "soil_exceeds_capacity": lambda d: d.update(soil=poke(d["soil"], ACT, float(d["smax"][ACT]) * 1.01)),
    "rain_nan": lambda d: d.update(rate=poke(d["rate"], ACT, np.nan)),
    "rain_negative": lambda d: d.update(rate=poke(d["rate"], ACT, -1e-6)),
    "rain_on_inactive_cell": lambda d: d.update(rate=poke(d["rate"], INACT, 1e-5)),
    "old_flux_nan_on_a_run_off_cell": lambda d: d.update(q=poke(d["q"], ACT, np.nan)),
    "old_flux_negative_on_a_run_off_cell": lambda d: d.update(q=poke(d["q"], ACT, -1e-9)),
    "old_flux_inconsistent_with_depth": lambda d: d.update(q=poke(d["q"], ACT, float(d["q"][ACT]) * 2.0)),
    "huge_finite_depth_overflows_the_old_flux": _huge_dry,
    "courant_rejection_is_recoverable": _courant,
    "dt_zero": lambda d: d.update(dt=0.0),
    "dt_negative": lambda d: d.update(dt=-1.0),
    "dt_nan": lambda d: d.update(dt=float("nan")),
    "dt_bool": lambda d: d.update(dt=True),
    "courant_max_out_of_range": lambda d: d.update(control=StormControl(courant_max=2.5, implementation="numba")),
    "iterations_zero": lambda d: d.update(control=StormControl(bisection_iterations=0, implementation="numba")),
    "iterations_bool": lambda d: d.update(control=StormControl(bisection_iterations=True, implementation="numba")),
    "root_tolerance_zero": lambda d: d.update(control=StormControl(root_tolerance_m=0.0, implementation="numba")),
    "unknown_implementation": lambda d: d.update(control=StormControl(implementation="fortran")),
    "bisection_not_converged": lambda d: d.update(control=StormControl(bisection_iterations=5, implementation="numba")),
    "depth_float32": lambda d: d.update(depth=d["depth"].astype(np.float32)),
    "depth_wrong_shape": lambda d: d.update(depth=np.zeros((2, 3))),
    "rain_wrong_dtype": lambda d: d.update(rate=d["rate"].astype(np.float32)),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_failures_match_the_reference_class_and_message_and_mutate_nothing(name):
    case, d = _adv()
    MUTATIONS[name](d)
    state = StormState(0.0, d["depth"], d["soil"], d["q"])
    before = [np.array(a, copy=True) for a in (d["rate"], d["depth"], d["soil"], d["q"])]
    with pytest.raises(Exception) as ref_err:
        coupled_step(case.graph, case.params, d["rate"], state, d["dt"], d["control"])
    with pytest.raises(Exception) as new_err:
        hn.prepared_coupled_step(case.ctx, d["rate"], state, d["dt"], d["control"])
    assert type(new_err.value) is type(ref_err.value), (name, repr(ref_err.value), repr(new_err.value))
    assert str(new_err.value) == str(ref_err.value), name
    for a, b in zip(before, (d["rate"], d["depth"], d["soil"], d["q"]), strict=True):
        np.testing.assert_array_equal(a, b)  # no input was modified


def test_error_categories_and_recoverability_are_preserved():
    case, d = _adv()
    MUTATIONS["courant_rejection_is_recoverable"](d)
    state = StormState(0.0, d["depth"], d["soil"], d["q"])
    with pytest.raises(RoutingStepRejected, match="Courant"):
        hn.prepared_coupled_step(case.ctx, d["rate"], state, d["dt"], d["control"])
    case, d = _adv()
    MUTATIONS["bisection_not_converged"](d)
    with pytest.raises(RoutingError, match="bisection did not reach") as info:
        hn.prepared_coupled_step(case.ctx, d["rate"], StormState(0.0, d["depth"], d["soil"], d["q"]), 1.0,
                                 d["control"])
    assert not isinstance(info.value, RoutingStepRejected)
    case, d = _adv()
    MUTATIONS["soil_nan"](d)  # a column failure outranks a routing-option failure, as in the reference
    with pytest.raises(InfiltrationError):
        hn.prepared_coupled_step(case.ctx, d["rate"], StormState(0.0, d["depth"], d["soil"], d["q"]), 1.0,
                                 StormControl(courant_max=9.0, implementation="numba"))


def test_list_input_is_refused_like_the_reference():
    case, d = _adv()
    state = StormState(0.0, d["depth"], d["soil"], d["q"])
    rain = d["rate"].tolist()
    with pytest.raises(InfiltrationError, match="array"):
        coupled_step(case.graph, case.params, rain, state, 1.0, CONTROL)
    with pytest.raises(InfiltrationError, match="array"):
        hn.prepared_coupled_step(case.ctx, rain, state, 1.0, CONTROL)


# --- deliberately stricter than the reference ------------------------------------------------------------------
def test_previous_discharge_is_checked_everywhere_not_only_where_a_branch_reads_it():
    case, d = _adv()
    zero_rain = np.zeros_like(d["rate"])  # partial branch everywhere: the reference recomputes q and ignores q_prev
    nan_q = poke(d["q"], ACT, np.nan)
    state = StormState(0.0, d["depth"], d["soil"], nan_q)
    coupled_step(case.graph, case.params, zero_rain, state, 1.0, CONTROL)  # reference accepts (laundered)
    with pytest.raises(RoutingError, match="state.discharge_m2_s must be finite"):
        hn.prepared_coupled_step(case.ctx, zero_rain, state, 1.0, CONTROL)
    for q, match in ((poke(d["q"], ACT, -1e-9), ">= 0"), (poke(d["q"], INACT, 1e-9), "0 on inactive")):
        with pytest.raises(RoutingError, match=match):
            hn.prepared_coupled_step(case.ctx, zero_rain, StormState(0.0, d["depth"], d["soil"], q), 1.0, CONTROL)


def test_masked_arrays_and_subclasses_cannot_launder_non_finite_values():
    case, d = _adv()
    bad = poke(d["depth"], ACT, np.nan)
    masked = np.ma.masked_array(bad, mask=~np.isfinite(bad))
    with pytest.raises(InfiltrationError, match="host numpy.ndarray"):
        hn.prepared_coupled_step(case.ctx, d["rate"], StormState(0.0, masked, d["soil"], d["q"]), 1.0, CONTROL)

    class Sub(np.ndarray):
        pass

    with pytest.raises(InfiltrationError, match="host numpy.ndarray"):
        hn.prepared_coupled_step(case.ctx, d["rate"], StormState(0.0, d["depth"].view(Sub), d["soil"], d["q"]), 1.0,
                                 CONTROL)
    with pytest.raises(RoutingError, match="host numpy.ndarray"):
        hn.prepared_coupled_step(case.ctx, d["rate"], StormState(0.0, d["depth"], d["soil"], d["q"].view(Sub)), 1.0,
                                 CONTROL)
    with pytest.raises(RoutingError, match="shape"):
        hn.prepared_coupled_step(case.ctx, d["rate"], StormState(0.0, d["depth"], d["soil"], np.zeros((2, 2))), 1.0,
                                 CONTROL)


def test_only_the_numba_sweep_is_accepted_and_the_reference_array_path_is_untouched():
    case, d = _adv()
    state = StormState(0.0, d["depth"], d["soil"], d["q"])
    with pytest.raises(RoutingError, match="numba sweep only"):
        hn.prepared_coupled_step(case.ctx, d["rate"], state, 1.0, StormControl(implementation="array"))
    reference = coupled_step(case.graph, case.params, d["rate"], state, 1.0, StormControl(implementation="array"))
    assert reference.route.implementation == "array"
    with pytest.raises(StormError):
        hn.prepared_coupled_step(case.ctx, d["rate"], state, 1.0, object())
    with pytest.raises(StormError):
        hn.prepared_coupled_step(case.ctx, d["rate"], object(), 1.0, CONTROL)
    with pytest.raises(hn.HydrologyPreparationError):
        hn.prepared_coupled_step(object(), d["rate"], state, 1.0, CONTROL)


# --- outputs, inputs and layouts ---------------------------------------------------------------------------------
def _all_arrays(step):
    c, r, s = step.column, step.route, step.state
    return {"state.depth": s.depth_m, "state.soil": s.soil_water_m, "state.q": s.discharge_m2_s,
            "col.depth": c.depth_m, "col.soil": c.soil_water_m, "col.rain": c.rain_m, "col.intake": c.intake_m,
            "col.return": c.saturation_return_m, "col.drainage": c.drainage_m,
            "route.depth": r.depth_m, "route.flow": r.flow_depth_m, "route.q": r.discharge_m2_s,
            "route.velocity": r.velocity_m_s, "route.inflow": r.inflow_m2_s, "route.old_q": r.old_discharge_m2_s,
            "route.old_inflow": r.old_inflow_m2_s, "route.face": r.face_volume_m3}


def test_outputs_are_fresh_and_never_alias_inputs_or_each_other():
    case = build_case(12, kind="valley", model="pavement_hawkins")
    inputs = [case.rate_on, case.state.depth_m, case.state.soil_water_m, case.state.discharge_m2_s]
    out = hn.prepared_coupled_step(case.ctx, case.rate_on, case.state, 1.0, CONTROL)
    arrays = _all_arrays(out)
    assert out.state.depth_m is out.route.depth_m and out.state.soil_water_m is out.column.soil_water_m
    assert out.state.discharge_m2_s is out.route.discharge_m2_s  # the same sharing as the reference
    names = list(arrays)
    for name in names:
        assert arrays[name].dtype == np.float64 and arrays[name].shape == case.graph.shape and arrays[name].flags.writeable
        for inp in inputs:
            assert not np.shares_memory(arrays[name], inp), name
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if arrays[a] is not arrays[b]:
                assert not np.shares_memory(arrays[a], arrays[b]), (a, b)


def test_a_failure_or_a_later_step_never_changes_an_earlier_result():
    case = build_case(13, kind="random", model="pavement_hawkins")
    first = hn.prepared_coupled_step(case.ctx, case.rate_on, case.state, 1.0, CONTROL)
    snapshot = {k: v.copy() for k, v in _all_arrays(first).items()}
    scalars = {k: getattr(first.route, k) for k in ("export_m3", "outlet_discharge_m3_s", "storage_change_m3")}
    bad = poke(case.state.depth_m, ACT, np.nan)
    with pytest.raises(InfiltrationError):
        hn.prepared_coupled_step(case.ctx, case.rate_on, StormState(0.0, bad, case.state.soil_water_m,
                                                                    case.state.discharge_m2_s), 1.0, CONTROL)
    hn.prepared_coupled_step(case.ctx, case.rate_on, first.state, 0.5, CONTROL)
    for k, v in _all_arrays(first).items():
        np.testing.assert_array_equal(v, snapshot[k], err_msg=k)
    for k, v in scalars.items():
        assert getattr(first.route, k) == v


def test_noncontiguous_fortran_and_readonly_inputs_match_the_reference():
    case = build_case(14, kind="random", model="pavement_hawkins")
    ny, nx = case.graph.shape
    big = np.zeros((2 * ny, 2 * nx))
    big[::2, ::2] = case.rate_on
    rate = big[::2, ::2]
    soil = case.state.soil_water_m.copy()
    soil.flags.writeable = False
    state = StormState(0.0, np.asfortranarray(case.state.depth_m), soil, np.asfortranarray(case.state.discharge_m2_s))
    assert not rate.flags.c_contiguous and not state.depth_m.flags.c_contiguous
    ref = coupled_step(case.graph, case.params, rate, state, 1.0, CONTROL)
    new = hn.prepared_coupled_step(case.ctx, rate, state, 1.0, CONTROL)
    compare(ref, new, exact=False)
    assert not soil.flags.writeable  # still read-only and unchanged
    np.testing.assert_array_equal(soil, case.state.soil_water_m)


# --- preparation ---------------------------------------------------------------------------------------------------
def _unfreeze(array) -> bool:
    try:
        array.flags.writeable = True
    except ValueError:
        return False
    return True


def test_context_owns_contiguous_readonly_copies_and_is_isolated_from_later_source_mutation():
    case = build_case(15, kind="random", model="fixed_ksat")
    ctx = case.ctx
    for name in hn.HydrologyContext._ARRAYS:
        array = getattr(ctx, name)
        assert array.flags.c_contiguous and not array.flags.writeable, name
        for source in (case.graph.conveyance, case.graph.donor_position, case.graph.level_order,
                       case.params.ksat_m_per_s, case.params.storage_max_m):
            assert not np.shares_memory(array, source), name
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.n_cells = 1
    expected = coupled_step(case.graph, case.params, case.rate_on, case.state, 1.0, CONTROL)
    assert float(expected.column.intake_m.sum()) > 0.0
    assert _unfreeze(case.params.ksat_m_per_s)  # the source's own array can be re-enabled and mutated in place
    case.params.ksat_m_per_s[...] = 0.0  # fixed Ksat 0: a fresh context must now take in no water
    still = hn.prepared_coupled_step(ctx, case.rate_on, case.state, 1.0, CONTROL)
    compare(expected, still, exact=False)  # the context did not follow the mutated source
    fresh = hn.prepare_hydrology(case.graph, case.params)  # a new context sees the new parameters
    changed = hn.prepared_coupled_step(fresh, case.rate_on, case.state, 1.0, CONTROL)
    assert not np.array_equal(changed.column.intake_m, expected.column.intake_m)


def test_context_summary_and_kernel_provenance_are_recorded():
    case = build_case(16, kind="valley", model="fixed_ksat")
    summary = case.ctx.summary()
    assert summary["shape"] == list(case.graph.shape) and summary["n_active"] == case.graph.n_active
    assert summary["n_levels"] == case.graph.n_levels and summary["max_level_width"] == case.graph.max_level_width
    assert summary["static_bytes"] == case.ctx.nbytes() > 0 and summary["host_only"] and summary["fastmath"] is False
    assert summary["graph_input_sha256"] == case.graph.input_sha256
    assert isinstance(case.ctx.preparation_s, float) and case.ctx.preparation_s > 0.0
    hn.prepared_coupled_step(case.ctx, case.rate_on, case.state, 1.0, CONTROL)
    prov = hn.kernel_provenance()
    assert prov["compiled_in_process"] is True and prov["numba_options"]["fastmath"] is False
    assert prov["numba_options"]["parallel"] is False and len(prov["module_sha256"]) == 64
    assert set(prov["versions"]) == {"numba", "llvmlite"}
    kernels = hn._kernels()
    for kernel in (kernels.column, kernels.route):
        assert not kernel.targetoptions.get("fastmath", False)
        assert not kernel.targetoptions.get("parallel", False)


def _graph_with(case, **changes):
    return dataclasses.replace(case.graph, **changes)


def _params_with(case, **changes):
    return dataclasses.replace(case.params, **changes)


def test_preparation_refuses_wrong_types_namespaces_shapes_and_inconsistent_masks():
    case = build_case(17, kind="valley_masked", model="pavement_hawkins")
    g, p = case.graph, case.params
    with pytest.raises(hn.HydrologyPreparationError, match="RoutingGraph"):
        hn.prepare_hydrology(object(), p)
    with pytest.raises(hn.HydrologyPreparationError, match="ColumnParameters"):
        hn.prepare_hydrology(g, object())
    with pytest.raises(hn.HydrologyPreparationError, match="host NumPy only"):
        hn.prepare_hydrology(_graph_with(case, xp=types.ModuleType("cupy")), p)
    with pytest.raises(hn.HydrologyPreparationError, match="host NumPy only"):
        hn.prepare_hydrology(g, _params_with(case, xp=types.ModuleType("cupy")))
    other = build_case(17, kind="valley", model="pavement_hawkins")
    with pytest.raises(hn.HydrologyPreparationError, match="shape"):
        hn.prepare_hydrology(g, other.params)
    all_true = column_parameters(model="fixed_ksat", ksat_m_per_s=np.full(g.shape, 1e-6),
                                 suction_m=np.zeros(g.shape), drainage_parameter=np.zeros(g.shape),
                                 theta_sat=np.full(g.shape, 0.4), soil_thickness_m=np.full(g.shape, 0.3))
    with pytest.raises(hn.HydrologyPreparationError, match="active_mask differs"):
        hn.prepare_hydrology(g, all_true)


@pytest.mark.parametrize("name, make, match", [
    ("ksat_nan", lambda c: ("p", {"ksat_m_per_s": poke(np.array(c.params.ksat_m_per_s), ACT, np.nan)}), "finite"),
    ("suction_negative", lambda c: ("p", {"suction_m": poke(np.array(c.params.suction_m), ACT, -1.0)}), ">= 0"),
    ("theta_zero", lambda c: ("p", {"theta_sat": poke(np.array(c.params.theta_sat), ACT, 0.0)}), "theta_sat"),
    ("thickness_zero", lambda c: ("p", {"soil_thickness_m": poke(np.array(c.params.soil_thickness_m), ACT, 0.0)}),
     "soil_thickness_m"),
    ("smax_not_the_product", lambda c: ("p", {"storage_max_m": np.array(c.params.storage_max_m) * 1.0001}),
     "storage_max_m"),
    ("lambda_missing", lambda c: ("p", {"lambda_m_per_s": None}), "lambda"),
    ("lambda_zero", lambda c: ("p", {"lambda_m_per_s": np.zeros(c.graph.shape)}), "lambda"),
    ("ksat_float32", lambda c: ("p", {"ksat_m_per_s": np.array(c.params.ksat_m_per_s, dtype=np.float32)}), "float64"),
    ("conveyance_negative", lambda c: ("g", {"conveyance": -np.asarray(c.graph.conveyance) - 1.0}), "conveyance"),
    ("conveyance_lo_differs", lambda c: ("g", {"conveyance_lo": np.asarray(c.graph.conveyance_lo) * 1.5}),
     "conveyance_lo"),
    ("order_not_a_permutation", lambda c: ("g", {"level_order": np.zeros_like(np.asarray(c.graph.level_order))}),
     "permutation"),
    ("donor_index_out_of_range", lambda c: ("g", {"donor_position": np.asarray(c.graph.donor_position) + c.graph.n_active}),
     "outside"),
    ("donor_in_same_or_later_level",
     lambda c: ("g", {"donor_position": np.full_like(np.asarray(c.graph.donor_position), c.graph.n_active - 1),
                      "donor_mask": np.ones_like(np.asarray(c.graph.donor_mask))}), "earlier"),
    ("donor_mask_incomplete", lambda c: ("g", {"donor_mask": np.zeros_like(np.asarray(c.graph.donor_mask))}), "donors"),
    ("level_bounds_wrong_end", lambda c: ("g", {"level_bounds": (*c.graph.level_bounds[:-1],
                                                                c.graph.level_bounds[-1] + 1)}), "level_bounds"),
    ("level_bounds_not_ints", lambda c: ("g", {"level_bounds": [0, c.graph.n_active]}), "level_bounds"),
    ("outlet_wrong", lambda c: ("g", {"outlet_flat": np.zeros_like(np.asarray(c.graph.outlet_flat))}), "outlet"),
    ("dx_nan", lambda c: ("g", {"dx_m": float("nan")}), "dx_m"),
])
def test_preparation_refuses_unsafe_static_data(name, make, match):
    case = build_case(18, kind="valley_masked", model="pavement_hawkins")
    which, changes = make(case)
    graph = _graph_with(case, **changes) if which == "g" else case.graph
    params = _params_with(case, **changes) if which == "p" else case.params
    with pytest.raises(hn.HydrologyPreparationError, match=match):
        hn.prepare_hydrology(graph, params)
