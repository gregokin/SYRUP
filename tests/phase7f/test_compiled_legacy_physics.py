"""Phase 7f: compiled frozen-composition wet physics against the NumPy reference `sediment_physics_step`.

The reference is the oracle: every comparison builds the reference result from the same inputs. Regimes, masks,
counts and quantities made only of IEEE-exact operations (+ - * / sqrt) must be identical; quantities passing
through exp/log/pow may differ by the libm-versus-NumPy-SIMD rounding of those functions (RTOL below, a bound
chosen for that rounding amplified by the `xs = stream power - Bagnold` cancellation, not a physics tolerance).
Nothing here was run by its author (file-only tools); Codex records actual results and any tightening."""
from __future__ import annotations

import dataclasses
import sys
import types

import numpy as np
import pytest

pytest.importorskip("maple")

from maple_syrup import legacy_experiment as le
from maple_syrup import legacy_physics_numba as lpn
from maple_syrup.routing_numba import numba_available
from maple_syrup.sediment_physics import (
    LEGACY_CLASS_RADII_M,
    LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM,
    PLOT1_RAINDROP_A,
    PLOT1_RAINDROP_B,
    PLOT1_RAINDROP_C,
    PLOT1_RAINDROP_MAX_MM,
    REGIME_CODES,
    SedimentPhysicsError,
    physics_grid,
    plot1_sediment_parameters,
    sediment_physics_parameters,
    sediment_physics_step,
)

NUMBA = pytest.mark.skipif(not numba_available(), reason="Numba not installed; no claim made")

RTOL = 1.0e-10
AREA = 0.25
FLOAT_FIELDS = ("requested_pickup_kg", "raindrop_pickup_kg", "flow_pickup_kg", "sediment_velocity_m_s",
                "deposition_rate_per_m", "rain_energy_j_m2_mm", "rain_energy_flux_j_m2_s", "pickup_probability")
IEEE_EXACT_FIELDS = ("d50_m", "shear_velocity_m_s", "reynolds_number", "stream_power_w_m2")
MASK_FIELDS = ("law_applies", "settle_mask", "regime", "legacy_cap_applied")
SWITCHES = {"ke_model": "wainwright_log", "ke_vegetation_form": "intended",
            "raindrop_composition_scaling": "fraction", "distance_convention": "formula_mean",
            "dstar_convention": "van_rijn", "bagnold_depth_units": "si_m"}


def params_for(n_classes: int = 6, **overrides):
    if n_classes == 6:
        return plot1_sediment_parameters(**overrides)
    n = n_classes
    kwargs = {
        "diameter_m": tuple(2.0 * r for r in LEGACY_CLASS_RADII_M[:n]), "raindrop_a": PLOT1_RAINDROP_A[:n],
        "raindrop_b": PLOT1_RAINDROP_B[:n], "raindrop_c": PLOT1_RAINDROP_C[:n],
        "raindrop_max_depth_mm": PLOT1_RAINDROP_MAX_MM[:n], "particle_density_g_cm3": 2.65,
        "active_layer_sensitivity_mm": 1.52e-6, "ke_model": "verstraeten_exp",
        "raindrop_depth_attenuation_per_cm": LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM[:n],
    }
    kwargs.update(overrides)
    return sediment_physics_parameters(**kwargs)


class Case:
    """Static terrain/vegetation/holdings plus a generator of valid dynamic states."""

    def __init__(self, seed: int, ny: int = 24, nx: int = 31, n_classes: int = 6, **param_overrides):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.ny, self.nx, self.nc = ny, nx, n_classes
        self.params = params_for(n_classes, **param_overrides)
        self.slope = rng.uniform(0.002, 0.6, (ny, nx))
        self.active = rng.random((ny, nx)) > 0.15
        self.active[0, 0] = False  # at least one inactive cell for the inactive-velocity checks
        self.active[3, 3] = True  # and a known active cell for the single-cell edits
        self.grid = physics_grid(self.slope, self.active, AREA)
        self.veg = rng.uniform(0.0, 1.0, (ny, nx)) * (rng.random((ny, nx)) > 0.5)
        self.holdings = rng.random((ny, nx, n_classes)) * rng.integers(0, 2, (ny, nx, n_classes)) * 5.0
        self.holdings[1, 1] = 0.0  # a fully empty cell
        self.holdings[2, 2, :] = 0.0
        self.holdings[2, 2, -1] = 3.0  # a single (coarsest) class

    def state(self) -> dict:
        rng, shape = self.rng, (self.ny, self.nx)
        depth = 10.0 ** rng.uniform(-5.0, -0.7, shape)
        depth[rng.random(shape) < 0.2] = 0.0
        velocity = 10.0 ** rng.uniform(-3.0, 0.6, shape)
        velocity[~self.active] = 0.0
        rain = np.where(rng.random(shape) < 0.5, 10.0 ** rng.uniform(-8.5, -4.3, shape), 0.0)
        prev = rng.random(shape + (self.nc,)) * 0.01 * (rng.random(shape + (self.nc,)) > 0.3)
        return {"depth": depth, "velocity": velocity, "rain": rain, "prev": prev}

    def reference(self, s: dict, dt: float = 1.0, *, holdings=None, veg=None):
        return sediment_physics_step(self.params, self.grid, s["depth"], s["velocity"], s["rain"],
                                     self.veg if veg is None else veg,
                                     self.holdings if holdings is None else holdings, s["prev"], dt)

    def context(self):
        return lpn.prepare_legacy_physics(self.params, self.grid, self.veg, self.holdings)

    def compiled(self, ctx, s: dict, dt: float = 1.0):
        return lpn.legacy_physics_step(ctx, s["depth"], s["velocity"], s["rain"], s["prev"], dt)


def assert_same(got, ref, *, rtol: float = RTOL) -> None:
    assert got.dt_s == ref.dt_s
    for name in MASK_FIELDS:
        a, b = getattr(got, name), getattr(ref, name)
        assert a.dtype == b.dtype and a.shape == b.shape, name
        np.testing.assert_array_equal(a, b, err_msg=name)
    for name in IEEE_EXACT_FIELDS:
        np.testing.assert_array_equal(getattr(got, name), getattr(ref, name), err_msg=name)
    for name in FLOAT_FIELDS:
        a, b = getattr(got, name), getattr(ref, name)
        assert a.dtype == np.float64 and a.shape == b.shape, name
        np.testing.assert_allclose(a, b, rtol=rtol, atol=0.0, err_msg=name)
    assert list(got.regime_counts) == list(ref.regime_counts)
    for key, value in ref.regime_counts.items():
        assert int(got.regime_counts[key]) == int(value), key


def outcome(fn):
    try:
        return fn()
    except SedimentPhysicsError as exc:
        return exc


# --- differential --------------------------------------------------------------------------------
@NUMBA
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_states_match_reference_and_cover_every_regime(seed):
    case = Case(seed)
    ctx = case.context()
    seen = set()
    for _ in range(3):
        s = case.state()
        got, ref = case.compiled(ctx, s), case.reference(s)
        assert_same(got, ref)
        seen |= {code for code in np.unique(ref.regime[case.active])}
        # counts are self-consistent with the returned regime array
        for name, code in REGIME_CODES.items():
            assert int(got.regime_counts[name]) == int(np.sum(case.active[..., None] & (got.regime == code)))
    if seed == 1:  # the generator must exercise all seven regimes, or the comparison proves little
        assert seen == set(REGIME_CODES.values())


@NUMBA
@pytest.mark.parametrize("name", sorted(SWITCHES))
def test_every_scientific_switch_matches_reference(name):
    case = Case(11, **{name: SWITCHES[name]})
    ctx = case.context()
    for _ in range(2):
        s = case.state()
        assert_same(case.compiled(ctx, s), case.reference(s))


@NUMBA
def test_all_switches_together_and_nondefault_parameters():
    case = Case(12, **SWITCHES, particle_density_g_cm3=2.9, kinematic_viscosity_m2_s=1.3e-6,
                recession_factor_per_reference_s=0.8, reference_interval_s=2.0, active_layer_sensitivity_mm=3.0e-3)
    ctx = case.context()
    s = case.state()
    assert_same(case.compiled(ctx, s), case.reference(s))


@NUMBA
@pytest.mark.parametrize("n_classes", [1, 2, 3])
def test_other_class_counts(n_classes):
    case = Case(13, ny=12, nx=13, n_classes=n_classes)
    ctx = case.context()
    s = case.state()
    assert_same(case.compiled(ctx, s), case.reference(s))


@NUMBA
@pytest.mark.parametrize("dt", [1.0, 0.25, 7.5, 1.0e-3, 123.0])
def test_dt_variation_including_memory_decay_and_pickup_scaling(dt):
    case = Case(14)
    ctx = case.context()
    s = case.state()
    got, ref = case.compiled(ctx, s, dt), case.reference(s, dt)
    assert_same(got, ref)
    assert got.dt_s == dt


@NUMBA
def test_same_context_serves_many_dynamic_states_and_is_not_changed_by_use():
    case = Case(15)
    ctx = case.context()
    before = {f.name: getattr(ctx, f.name).copy() for f in dataclasses.fields(ctx)
              if isinstance(getattr(ctx, f.name), np.ndarray)}
    for _ in range(4):
        s = case.state()
        assert_same(case.compiled(ctx, s, 0.5), case.reference(s, 0.5))
    for name, arr in before.items():
        np.testing.assert_array_equal(getattr(ctx, name), arr, err_msg=name)
        assert not getattr(ctx, name).flags.writeable


# --- exact thresholds ---------------------------------------------------------------------------
def _line_case(params, depth, velocity, rain, *, slope=0.1, nc=6, holdings=None):
    n = len(depth)
    grid = physics_grid(np.full((1, n), slope), np.ones((1, n), dtype=bool), AREA)
    veg = np.zeros((1, n))
    hold = np.ones((1, n, nc)) if holdings is None else holdings
    s = {"depth": np.asarray(depth, dtype=float).reshape(1, n), "velocity": np.asarray(velocity, dtype=float).reshape(1, n),
         "rain": np.asarray(rain, dtype=float).reshape(1, n), "prev": np.full((1, n, nc), 0.003)}
    ref = sediment_physics_step(params, grid, s["depth"], s["velocity"], s["rain"], veg, hold, s["prev"], 1.0)
    ctx = lpn.prepare_legacy_physics(params, grid, veg, hold)
    got = lpn.legacy_physics_step(ctx, s["depth"], s["velocity"], s["rain"], s["prev"], 1.0)
    return got, ref


@NUMBA
@pytest.mark.parametrize("reynolds, rain", [(500.0, 0.0), (500.0, 2.0e-5), (2500.0, 0.0), (2500.0, 2.0e-5)])
def test_reynolds_boundaries_are_exact(reynolds, rain):
    nu = 2.0 ** -10  # v * d / nu is then exact in binary: Re = 500 and 2500 are hit, not approximated
    params = plot1_sediment_parameters(kinematic_viscosity_m2_s=nu)
    v_exact = reynolds * nu
    velocities = [np.nextafter(v_exact, 0.0), v_exact, np.nextafter(v_exact, 1.0e9)]
    got, ref = _line_case(params, [1.0] * 3, velocities, [rain] * 3, slope=0.05)
    assert float(ref.reynolds_number[0, 1]) == reynolds
    assert len(set(ref.regime[0, :, 0].tolist())) > 1  # the three points straddle a regime boundary
    assert_same(got, ref)


@NUMBA
def test_suspension_threshold_is_ge_and_matches_at_neighbouring_floats():
    params = plot1_sediment_parameters()
    dstar = params.d_diameter[0, 0] * params.dstar_const_per_m
    crit = np.where(dstar <= 10.0, 4.0 * params.d_settling[0, 0] / dstar, 0.4 * params.d_settling[0, 0])
    slope, g = 0.2, 9.81
    for k in (2, 3, 4):
        d0 = crit[k] ** 2 / (g * slope)
        depths = [d0]
        for _ in range(16):
            depths = [np.nextafter(depths[0], 0.0)] + depths + [np.nextafter(depths[-1], 1.0)]
        got, ref = _line_case(params, depths, [5.0] * len(depths), [0.0] * len(depths), slope=slope)
        assert_same(got, ref)
        flags = ref.regime[0, :, k] == REGIME_CODES["suspended"]
        assert flags.any() and not flags.all()  # the scan really straddles ustar == critical


@NUMBA
@pytest.mark.parametrize("ulps", [-1, 0, 1])
def test_dstar_boundary_at_ten(ulps):
    # 4 ws / D* equals 0.4 ws at D* = 10, so the criterion is continuous there; this checks the branch selection
    # `D* <= 10` on both sides without a result jump.
    base = plot1_sediment_parameters()
    d = np.array(base.diameter_m)
    d[2] = 10.0 / base.dstar_const_per_m
    d[2] = {-1: np.nextafter(d[2], 0.0), 0: d[2], 1: np.nextafter(d[2], 1.0)}[ulps]
    assert d[1] < d[2] < d[3]
    params = plot1_sediment_parameters(diameter_m=tuple(d))
    got, ref = _line_case(params, [0.05, 0.1, 0.2], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0], slope=0.3)
    assert_same(got, ref)


@NUMBA
def test_zero_and_empty_edge_values():
    params = plot1_sediment_parameters()
    zeros = np.zeros((1, 4, 6))
    got, ref = _line_case(params, [0.0, 0.0, 0.01, 0.01], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0e-5],
                          holdings=zeros)
    assert_same(got, ref)
    assert not got.requested_pickup_kg.any()  # empty holdings detach nothing


# --- ownership and mutation ------------------------------------------------------------------
@NUMBA
def test_context_owns_its_static_data_against_caller_mutation():
    case = Case(21)
    ctx = case.context()
    s = case.state()
    expected = case.reference(s)  # reference built from the ORIGINAL holdings and vegetation
    for arr in (case.holdings, case.veg, case.slope, case.active):
        assert not any(np.shares_memory(arr, getattr(ctx, f.name)) for f in dataclasses.fields(ctx)
                       if isinstance(getattr(ctx, f.name), np.ndarray))
    case.holdings[...] = 7.0
    case.veg[...] = 0.9
    case.slope[...] = 0.5
    case.active[...] = False
    assert_same(case.compiled(ctx, s), expected)


@NUMBA
def test_results_are_independently_owned_and_not_changed_by_later_calls():
    case = Case(22)
    ctx = case.context()
    s1, s2 = case.state(), case.state()
    r1 = case.compiled(ctx, s1)
    snapshot = {n: getattr(r1, n).copy() for n in FLOAT_FIELDS + IEEE_EXACT_FIELDS + MASK_FIELDS}
    r2 = case.compiled(ctx, s2)
    for n, arr in snapshot.items():
        np.testing.assert_array_equal(getattr(r1, n), arr, err_msg=n)
        assert not np.shares_memory(getattr(r1, n), getattr(r2, n)), n
        assert not np.shares_memory(getattr(r1, n), ctx.d50), n
    r1.d50_m[...] = -1.0  # a caller scribbling on a result cannot corrupt the context or later results
    assert_same(case.compiled(ctx, s2), r2)


@NUMBA
def test_noncontiguous_and_readonly_inputs_are_accepted_and_untouched():
    case = Case(23)
    ctx = case.context()
    s = case.state()
    expected = case.reference(s)
    fortran = {k: np.asfortranarray(v) for k, v in s.items()}
    for v in fortran.values():
        v.setflags(write=False)
    copies = {k: v.copy() for k, v in fortran.items()}
    assert_same(case.compiled(ctx, fortran), expected)
    for k, v in fortran.items():
        np.testing.assert_array_equal(v, copies[k])


# --- refusals match the reference -------------------------------------------------------------
def _bad_states():
    def edit(key, index, value):
        def apply(s, case):
            arr = s[key]
            arr[index] = value
            return s
        return apply

    return {
        "nan_depth": edit("depth", (3, 3), np.nan),
        "inf_depth": edit("depth", (3, 3), np.inf),
        "negative_depth": edit("depth", (3, 3), -1.0e-3),
        "nan_velocity": edit("velocity", (3, 3), np.nan),
        "negative_velocity": edit("velocity", (3, 3), -0.1),
        "inactive_velocity": edit("velocity", (0, 0), 0.2),
        "nan_rain": edit("rain", (3, 3), np.nan),
        "negative_rain": edit("rain", (3, 3), -1.0e-6),
        "rain_overflow_active": edit("rain", (3, 3), 1.0e305),
        "rain_overflow_inactive": edit("rain", (0, 0), 1.0e305),
        "rain_barely_overflowing": edit("rain", (3, 3), 6.0e301),
        "nan_memory": edit("prev", (3, 3, 2), np.nan),
        "negative_memory": edit("prev", (3, 3, 2), -1.0e-3),
        "stream_power_overflow": lambda s, c: (s["depth"].__setitem__((3, 3), 1.0e200),
                                               s["velocity"].__setitem__((3, 3), 1.0e200), s)[-1],
        "large_but_finite": lambda s, c: (s["depth"].__setitem__((3, 3), 1.0e100),
                                          s["velocity"].__setitem__((3, 3), 1.0e100), s)[-1],
    }


@pytest.mark.parametrize("name", sorted(_bad_states()))
def test_dynamic_refusals_and_acceptances_agree_with_reference(name):
    if not numba_available():
        pytest.skip("Numba not installed; no claim made")
    case = Case(31)
    ctx = case.context()
    s = _bad_states()[name](case.state(), case)
    ref = outcome(lambda: case.reference(s))
    inputs_before = {k: v.copy() for k, v in s.items()}
    got = outcome(lambda: case.compiled(ctx, s))
    for k, v in s.items():
        np.testing.assert_array_equal(v, inputs_before[k], err_msg=f"{k} was modified")
    assert isinstance(got, SedimentPhysicsError) == isinstance(ref, SedimentPhysicsError), (name, got, ref)
    if not isinstance(ref, SedimentPhysicsError):
        assert_same(got, ref)


@NUMBA
@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf, 1.0e308, True, "1", None])
def test_dt_refusals_agree_with_reference(dt):
    case = Case(32)
    ctx = case.context()
    s = case.state()
    ref = outcome(lambda: case.reference(s, dt))
    got = outcome(lambda: case.compiled(ctx, s, dt))
    assert isinstance(ref, SedimentPhysicsError), "the reference is expected to refuse every one of these dt"
    assert isinstance(got, SedimentPhysicsError)


@NUMBA
def test_tiny_dt_is_accepted_by_both():
    case = Case(33)
    ctx = case.context()
    s = case.state()
    assert_same(case.compiled(ctx, s, 1.0e-300), case.reference(s, 1.0e-300))


@NUMBA
@pytest.mark.parametrize("bad", ["float32", "int", "shape", "list", "namespace_fake", "wrong_prev_shape"])
def test_dynamic_shape_dtype_and_type_refusals(bad):
    case = Case(34)
    ctx = case.context()
    s = case.state()
    if bad == "float32":
        s["depth"] = s["depth"].astype(np.float32)
    elif bad == "int":
        s["rain"] = np.zeros((case.ny, case.nx), dtype=np.int64)
    elif bad == "shape":
        s["velocity"] = s["velocity"][:-1]
    elif bad == "list":
        s["depth"] = s["depth"].tolist()
    elif bad == "namespace_fake":
        s["depth"] = types.SimpleNamespace(shape=s["depth"].shape, dtype=np.float64)
    elif bad == "wrong_prev_shape":
        s["prev"] = s["prev"][..., :-1]
    assert isinstance(outcome(lambda: case.reference(s)), SedimentPhysicsError)
    assert isinstance(outcome(lambda: case.compiled(ctx, s)), SedimentPhysicsError)


def _static_bad():
    def holdings(fn):
        return lambda c: (fn(c.holdings), c)[-1]

    return {
        "nan_holdings": holdings(lambda h: h.__setitem__((3, 3, 1), np.nan)),
        "negative_holdings": holdings(lambda h: h.__setitem__((3, 3, 1), -1.0)),
        "inf_holdings": holdings(lambda h: h.__setitem__((3, 3, 1), np.inf)),
        "holdings_total_overflow": holdings(lambda h: (h.__setitem__((3, 3, 0), 1.0e308),
                                                       h.__setitem__((3, 3, 1), 1.0e308))),
        "vegetation_above_one": lambda c: c.veg.__setitem__((3, 3), 1.5),
        "negative_vegetation": lambda c: c.veg.__setitem__((3, 3), -0.1),
        "nan_vegetation": lambda c: c.veg.__setitem__((3, 3), np.nan),
    }


@NUMBA
@pytest.mark.parametrize("name", sorted(_static_bad()))
def test_invalid_static_inputs_are_refused_like_the_reference(name):
    case = Case(35)
    s = case.state()
    _static_bad()[name](case)
    assert isinstance(outcome(lambda: case.reference(s)), SedimentPhysicsError), "reference must refuse these"
    assert isinstance(outcome(case.context), SedimentPhysicsError)


@NUMBA
def test_wrong_static_dtype_and_shape_are_refused():
    case = Case(36)
    with pytest.raises(SedimentPhysicsError):
        lpn.prepare_legacy_physics(case.params, case.grid, case.veg.astype(np.float32), case.holdings)
    with pytest.raises(SedimentPhysicsError):
        lpn.prepare_legacy_physics(case.params, case.grid, case.veg, case.holdings[..., :-1])
    with pytest.raises(SedimentPhysicsError):
        lpn.prepare_legacy_physics(case.params, case.grid, case.veg[:-1], case.holdings)
    with pytest.raises(SedimentPhysicsError):
        lpn.prepare_legacy_physics(case.params, "grid", case.veg, case.holdings)
    with pytest.raises(SedimentPhysicsError):
        lpn.legacy_physics_step("not a context", np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                                np.zeros((2, 2, 6)), 1.0)


@NUMBA
def test_context_rejects_wrong_shaped_dynamic_state_for_its_grid():
    case = Case(37)
    ctx = case.context()
    other = Case(37, ny=5, nx=5)
    with pytest.raises(SedimentPhysicsError):
        case.compiled(ctx, other.state())


@NUMBA
def test_invalid_hand_built_parameters_are_refused():
    case = Case(38)
    bad = dataclasses.replace(case.params, recession_factor_per_reference_s=1.5)
    with pytest.raises(SedimentPhysicsError):
        lpn.prepare_legacy_physics(bad, case.grid, case.veg, case.holdings)
    bad = dataclasses.replace(case.params, ke_model="nonsense")
    with pytest.raises(SedimentPhysicsError):
        lpn.prepare_legacy_physics(bad, case.grid, case.veg, case.holdings)


# --- missing Numba / GPU: explicit refusal, never a fallback ----------------------------------------
def test_missing_numba_is_an_explicit_error_and_never_a_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "numba", None)  # `import numba` now raises ImportError
    monkeypatch.setattr(lpn, "_KERNEL", None)
    assert lpn.numba_available() is False
    with pytest.raises(lpn.LegacyPhysicsNumbaUnavailableError):
        lpn.compiled_kernel()
    case = Case(41, ny=6, nx=7)
    ctx = case.context()  # preparation is plain NumPy and needs no Numba
    with pytest.raises(lpn.LegacyPhysicsNumbaUnavailableError):
        case.compiled(ctx, case.state())
    assert issubclass(lpn.LegacyPhysicsNumbaUnavailableError, SedimentPhysicsError)


def test_gpu_namespace_is_refused_without_transfer():
    case = Case(42, ny=6, nx=7)
    fake_params = dataclasses.replace(case.params, xp=types.SimpleNamespace(__name__="cupy"))
    with pytest.raises(SedimentPhysicsError, match="host NumPy only"):
        lpn.prepare_legacy_physics(fake_params, case.grid, case.veg, case.holdings)
    fake_grid = dataclasses.replace(case.grid, xp=types.SimpleNamespace(__name__="cupy"))
    with pytest.raises(SedimentPhysicsError):
        lpn.prepare_legacy_physics(case.params, fake_grid, case.veg, case.holdings)


def test_actual_cupy_arrays_are_refused_when_a_device_exists():
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)
    except Exception as exc:  # noqa: BLE001 - CuPy raises several runtime types when no device is usable
        pytest.skip(f"no CUDA device: {exc}")
    case = Case(43, ny=6, nx=7)
    with pytest.raises(SedimentPhysicsError):
        lpn.prepare_legacy_physics(case.params, case.grid, cp.asarray(case.veg), case.holdings)
    with pytest.raises(SedimentPhysicsError):
        lpn.prepare_legacy_physics(case.params, case.grid, case.veg, cp.asarray(case.holdings))
    if numba_available():
        ctx = case.context()
        s = case.state()
        with pytest.raises(SedimentPhysicsError):
            lpn.legacy_physics_step(ctx, cp.asarray(s["depth"]), cp.asarray(s["velocity"]), cp.asarray(s["rain"]),
                                    cp.asarray(s["prev"]), 1.0)


# --- preparation record ---------------------------------------------------------------------------
def test_preparation_is_timed_and_summarised():
    case = Case(44, ny=10, nx=11)
    ctx = case.context()
    assert isinstance(ctx.preparation_s, float) and ctx.preparation_s > 0.0
    summary = ctx.summary()
    assert summary["frozen_composition"] is True and summary["fastmath"] is False
    assert summary["static_bytes"] == ctx.nbytes() > 0 and summary["shape"] == [10, 11]
    for name in ("slope", "active", "d50", "fractions", "cap_rate", "slope_power", "class_constants"):
        arr = getattr(ctx, name)
        assert arr.flags["C_CONTIGUOUS"] and not arr.flags.writeable


# --- driver selector --------------------------------------------------------------------------------
@pytest.fixture
def no_run(monkeypatch):
    calls = []
    monkeypatch.setattr(le, "run", lambda args: calls.append(args))
    return calls


def test_physics_selector_defaults_follow_the_implementation_choice():
    parser = le.build_parser()
    ns = parser.parse_args(["--output", "x"])
    assert ns.physics_implementation is None and le.resolve_physics_implementation(ns) == "numba"
    ns = parser.parse_args(["--output", "x", "--implementation", "array"])
    assert le.resolve_physics_implementation(ns) == "array"
    ns = parser.parse_args(["--output", "x", "--physics-implementation", "array"])
    assert ns.implementation == "numba" and le.resolve_physics_implementation(ns) == "array"
    ns = parser.parse_args(["--output", "x", "--implementation", "array", "--physics-implementation", "numba"])
    assert le.resolve_physics_implementation(ns) == "numba"
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "x", "--physics-implementation", "cupy"])


def test_missing_numba_refuses_compiled_physics_but_not_the_array_reference(no_run, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(le, "numba_available", lambda: False)
    # array water + compiled physics: refused, and only for the physics
    assert le.main(["--output", str(tmp_path / "a"), "--implementation", "array",
                    "--physics-implementation", "numba"]) == 2
    err = capsys.readouterr().err
    assert "wet physical laws" in err and "compiled water" not in err and not no_run
    # numba water: both refused
    assert le.main(["--output", str(tmp_path / "b")]) == 2
    err = capsys.readouterr().err
    assert "compiled water" in err and "wet physical laws" in err and not no_run
    # the declared array diagnostic stays runnable
    monkeypatch.setattr(le.L, "KERNEL_IMPLEMENTATION", "python")
    assert le.main(["--output", str(tmp_path / "c"), "--implementation", "array"]) == 0
    assert no_run and le.resolve_physics_implementation(no_run[-1]) == "array"


@NUMBA
def test_array_physics_reference_remains_selectable_on_compiled_water(no_run, tmp_path):
    out = tmp_path / "d"
    assert le.main(["--output", str(out), "--physics-implementation", "array"]) == (
        0 if le.L.KERNEL_IMPLEMENTATION == "numba" else 2)
    if no_run:
        assert no_run[-1].implementation == "numba" and no_run[-1].physics_implementation == "array"
