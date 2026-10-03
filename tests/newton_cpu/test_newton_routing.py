"""`route_step(..., root_solver="newton")`: same equation, D4 donor order, storage identity and checks as the
bisection default, array (NumPy per level) and compiled (Numba) forms. The default stays bitwise the historical
bisection; Newton agrees with it only within the root accuracy and is compared quantitatively, not bitwise."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from rfid_helpers import DX, build, pit_chain
from test_routing import chain_full, make_graph, random_full, valley_full

pytest.importorskip("maple")

from maple_syrup import routing_newton as rn
from maple_syrup.routing import (
    DEFAULT_NEWTON_MAX_ITERATIONS,
    RoutingError,
    RoutingStepRejected,
    route_step,
)
from maple_syrup.routing_numba import numba_available

NUMBA = pytest.mark.skipif(not numba_available(), reason="Numba not installed; compiled form not exercised, no claim made")
IMPLS = ["array", pytest.param("numba", marks=NUMBA)]
FIELDS = ("depth_m", "flow_depth_m", "discharge_m2_s", "velocity_m_s", "inflow_m2_s", "old_discharge_m2_s",
          "old_inflow_m2_s", "face_volume_m3")


def state(graph, rng, *, scale=0.01, extra=0.002, dry_fraction=0.0):
    """(depth_start, old_flow_depth, old_discharge): old flow depth <= start depth, q_old = k h_old^1.5."""
    h_old = rng.uniform(0.0, scale, graph.shape) * (rng.random(graph.shape) >= dry_fraction)
    h_old = np.where(graph.active, h_old, 0.0)
    start = np.where(graph.active, h_old + rng.uniform(0.0, extra, graph.shape), 0.0)
    k = graph.conveyance.reshape(graph.shape)
    q_old = np.where(graph.active, (np.sqrt(h_old) * h_old) * k, 0.0)
    return start, h_old, q_old


def same_bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))


def graphs():
    rng = np.random.default_rng(0)
    return {
        "valley": make_graph(valley_full(8, 7), ff=5.0),
        "random": make_graph(random_full(rng, 11, 8), ff=rng.uniform(0.2, 40.0, (11, 8))),
        "chain": make_graph(chain_full(9), ff=1.0),
    }


GRAPHS = graphs()


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("name", list(GRAPHS))
def test_default_is_the_unchanged_bisection_bitwise(name, impl):
    g = GRAPHS[name]
    start, h_old, q_old = state(g, np.random.default_rng(1))
    default = route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation=impl)
    explicit = route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation=impl,
                          root_solver="bisection", newton_max_iterations=7)  # an unused Newton cap changes nothing
    assert default.root_solver == "bisection" and default.root_stats is None and default.newton_max_iterations == 0
    assert default.bisection_iterations == 40
    for f in dataclasses.fields(default):
        a, b = getattr(default, f.name), getattr(explicit, f.name)
        if isinstance(a, np.ndarray):
            same_bits(a, b)
        else:
            assert a == b or (a is None and b is None), f.name


@pytest.mark.parametrize("name", list(GRAPHS))
def test_array_and_numba_agree_bitwise_and_report_newton_provenance(name):
    pytest.importorskip("numba")
    g = GRAPHS[name]
    start, h_old, q_old = state(g, np.random.default_rng(2), dry_fraction=0.2)
    a = route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation="array", root_solver="newton")
    b = route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation="numba", root_solver="newton")
    for f in FIELDS:
        same_bits(getattr(a, f), getattr(b, f))
    for f in ("export_m3", "storage_change_m3", "budget_residual_m3", "outlet_discharge_m3_s",
              "max_constitutive_residual_m", "max_cell_balance_residual_m", "max_courant_new"):
        assert float(getattr(a, f)) == float(getattr(b, f)), f
    assert a.root_stats == b.root_stats and set(a.root_stats) == set(rn.STAT_NAMES)
    assert (a.root_solver, a.implementation) == ("newton", "array") and b.implementation == "numba"
    assert a.bisection_iterations == 0 and a.newton_max_iterations == DEFAULT_NEWTON_MAX_ITERATIONS
    assert a.root_stats["fallback_cells"] == 0 and 0 < a.root_stats["max_newton_iterations"] <= 8
    assert a.root_stats["iterated_cells"] <= g.n_active and a.conservative


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("name", list(GRAPHS))
def test_newton_roots_agree_with_bisection_within_the_root_accuracy_and_are_tighter(name, impl):
    g = GRAPHS[name]
    rng = np.random.default_rng(3)
    worst = {"flow": 0.0, "depth": 0.0, "q": 0.0}
    for dt in (0.05, 1.0):
        start, h_old, q_old = state(g, rng, dry_fraction=0.1)
        b = route_step(g, start, h_old, dt, old_discharge_m2_s=q_old, implementation=impl)
        n = route_step(g, start, h_old, dt, old_discharge_m2_s=q_old, implementation=impl, root_solver="newton")
        worst["flow"] = max(worst["flow"], float(np.abs(n.flow_depth_m - b.flow_depth_m).max()))
        worst["depth"] = max(worst["depth"], float(np.abs(n.depth_m - b.depth_m).max()))
        worst["q"] = max(worst["q"], float(np.abs(n.discharge_m2_s - b.discharge_m2_s).max()))
        # both pass the unchanged constitutive/water checks; the Newton residual is not larger than the bisection's
        assert float(n.max_constitutive_residual_m) <= max(float(b.max_constitutive_residual_m), 1e-15)
        assert abs(float(n.budget_residual_m3)) <= 1e-12 and abs(float(b.budget_residual_m3)) <= 1e-12
        assert float(n.max_cell_balance_residual_m) <= 1e-12
    # bisection leaves R 2^-40 ~ 1e-12 R of root error; Newton ~ eps R. Deviation is bounded by the bisection's.
    assert worst["flow"] <= 2e-14 and worst["depth"] <= 2e-14 and worst["q"] <= 2e-14, worst


@pytest.mark.parametrize("impl", IMPLS)
def test_dry_state_and_tiny_wet_values(impl):
    g = GRAPHS["valley"]
    zero = np.zeros(g.shape)
    r = route_step(g, zero, zero, 1.0, implementation=impl, root_solver="newton")
    assert not r.flow_depth_m.any() and not r.discharge_m2_s.any() and r.root_stats["iterated_cells"] == 0
    for depth in (5e-324, 1e-300, 1e-200, 1e-30, 1e-12):
        start = np.where(g.active, depth, 0.0)
        n = route_step(g, start, start, 1.0, implementation=impl, root_solver="newton")
        b = route_step(g, start, start, 1.0, implementation=impl)
        assert np.all(n.depth_m >= 0.0) and np.all(np.isfinite(n.depth_m))
        np.testing.assert_allclose(n.depth_m, b.depth_m, rtol=1e-9, atol=0.0)
        assert abs(float(n.budget_residual_m3)) <= 1e-12 * max(1.0, depth)


@pytest.mark.parametrize("impl", IMPLS)
def test_wide_dynamic_range_of_friction_and_depth(impl):
    rng = np.random.default_rng(8)
    ff = 10 ** rng.uniform(-1.0, 4.0, (11, 8))  # friction f >= 0.1 up to 1e4
    g = make_graph(random_full(rng, 11, 8), ff=ff)
    for scale in (1e-9, 1e-4, 1e-2, 0.2):
        start, h_old, q_old = state(g, rng, scale=scale, extra=scale)
        for dt in (1e-6, 0.01, 0.5):
            try:
                b = route_step(g, start, h_old, dt, old_discharge_m2_s=q_old, implementation=impl)
            except RoutingStepRejected:
                with pytest.raises(RoutingStepRejected):  # same recoverable Courant rejection with Newton
                    route_step(g, start, h_old, dt, old_discharge_m2_s=q_old, implementation=impl,
                               root_solver="newton")
                continue
            n = route_step(g, start, h_old, dt, old_discharge_m2_s=q_old, implementation=impl, root_solver="newton")
            # bisection stops R 2^-40 (~9.1e-13 R) below the root; Newton is within rounding of it
            assert float(np.abs(n.flow_depth_m - b.flow_depth_m).max()) <= 1.0e-12 * float(b.flow_depth_m.max())
            assert n.root_stats["fallback_cells"] == 0
            assert abs(float(n.budget_residual_m3)) <= 1e-12 * max(1.0, scale)


@pytest.mark.parametrize("impl", IMPLS)
def test_pit_storage_cell_is_solved_analytically_and_retains_its_water(impl):
    g = build(pit_chain(), allow_pit_storage=True)
    assert g.pit_storage.any() and np.count_nonzero(g.conveyance.reshape(g.shape)[g.pit_storage]) == 0
    rng = np.random.default_rng(9)
    start, h_old, q_old = state(g, rng, dry_fraction=0.0)
    n = route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation=impl, root_solver="newton")
    b = route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation=impl)
    pit = g.pit_storage
    rhs_pit = n.depth_m[pit]  # k = 0: h_new = R = h_flow exactly, nothing leaves the pit
    np.testing.assert_array_equal(n.flow_depth_m[pit], rhs_pit)
    assert not n.discharge_m2_s[pit].any() and not n.face_volume_m3[pit].any()
    np.testing.assert_allclose(n.depth_m, b.depth_m, rtol=0.0, atol=2e-14)
    assert abs(float(n.budget_residual_m3)) <= 1e-13 and n.root_stats["fallback_cells"] == 0


@pytest.mark.parametrize("impl", IMPLS)
def test_small_iteration_cap_uses_the_bracketed_fallback_with_identical_science(impl):
    g = GRAPHS["random"]
    start, h_old, q_old = state(g, np.random.default_rng(4))
    full = route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation=impl, root_solver="newton")
    capped = route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation=impl, root_solver="newton",
                        newton_max_iterations=1)
    assert capped.root_stats["fallback_cells"] > 0 and capped.root_stats["max_newton_iterations"] == 1
    assert capped.newton_max_iterations == 1 and full.root_stats["fallback_cells"] == 0
    np.testing.assert_allclose(capped.flow_depth_m, full.flow_depth_m, rtol=1e-12, atol=1e-18)
    assert float(capped.max_constitutive_residual_m) <= 1e-11 and abs(float(capped.budget_residual_m3)) <= 1e-12


@NUMBA
def test_fallback_path_is_identical_between_the_array_and_numba_forms():
    g = GRAPHS["random"]
    start, h_old, q_old = state(g, np.random.default_rng(6))
    a = route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, root_solver="newton", newton_max_iterations=2)
    b = route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation="numba", root_solver="newton",
                   newton_max_iterations=2)
    for f in FIELDS:
        same_bits(getattr(a, f), getattr(b, f))
    assert a.root_stats == b.root_stats and a.root_stats["fallback_cells"] > 0


# --- guards: invalid options are refused before anything is computed or returned --------------------------------
BAD_OPTIONS = [
    {"root_solver": "secant"}, {"root_solver": "Newton"}, {"root_solver": None}, {"root_solver": 1},
    {"root_solver": "newton", "newton_max_iterations": 0}, {"root_solver": "newton", "newton_max_iterations": -3},
    {"root_solver": "newton", "newton_max_iterations": True}, {"root_solver": "newton", "newton_max_iterations": 2.0},
    {"root_solver": "newton", "newton_max_iterations": "5"},
    {"root_solver": "newton", "newton_max_iterations": rn.MAX_NEWTON_ITERATIONS + 1},
    {"newton_max_iterations": 0},  # validated even while the bisection default is selected
]


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("options", BAD_OPTIONS, ids=lambda o: repr(o))
def test_invalid_root_options_raise_before_mutation(options, impl):
    g = GRAPHS["valley"]
    start, h_old, q_old = state(g, np.random.default_rng(1))
    before = [a.copy() for a in (start, h_old, q_old)]
    with pytest.raises(RoutingError, match="root_solver|newton_max_iterations"):
        route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation=impl, **options)
    for a, b in zip((start, h_old, q_old), before, strict=True):
        np.testing.assert_array_equal(a, b)


def test_cuda_newton_is_refused_explicitly_without_fallback():
    g = GRAPHS["valley"]
    start, h_old, q_old = state(g, np.random.default_rng(1))
    with pytest.raises(RoutingError, match="CPU-only"):
        route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation="cuda", root_solver="newton")


@pytest.mark.parametrize("impl", IMPLS)
def test_nonconvergence_to_a_tight_tolerance_is_an_error_and_is_not_labelled_bisection(impl):
    g = GRAPHS["valley"]
    start, h_old, q_old = state(g, np.random.default_rng(1))
    before = [a.copy() for a in (start, h_old, q_old)]
    with pytest.raises(RoutingError, match="Newton root solver did not reach root_tolerance_m") as info:
        route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation=impl, root_solver="newton",
                   root_tolerance_m=1e-300, newton_max_iterations=9)
    assert "bisection" not in str(info.value) and "newton_max_iterations = 9" in str(info.value)
    with pytest.raises(RoutingError, match="bisection did not reach"):  # the default's message is unchanged
        route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation=impl, root_tolerance_m=1e-300)
    for a, b in zip((start, h_old, q_old), before, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("impl", IMPLS)
def test_shared_step_rejections_are_unchanged_under_newton(impl):
    g = GRAPHS["chain"]
    h = np.where(g.active, 1e-3, 0.0)
    k = g.conveyance.reshape(g.shape)
    hot = np.where(g.active, 1.0, 0.0)  # a deep old flow with a long step violates the Courant bound
    with pytest.raises(RoutingStepRejected):
        route_step(g, hot, hot, 50.0, implementation=impl, root_solver="newton")
    with pytest.raises(RoutingError, match="dt_s"):
        route_step(g, h, h, 0.0, implementation=impl, root_solver="newton")  # dt = 0 stays rejected, not an identity
    bad = h.copy()
    bad[g.active] = np.nan
    with pytest.raises(RoutingError, match="finite"):
        route_step(g, bad, h, 1.0, implementation=impl, root_solver="newton")
    assert np.isfinite(k).all()
    assert DX > 0


def test_array_newton_route_step_never_loops_over_cells(monkeypatch):
    g = GRAPHS["valley"]
    start, h_old, q_old = state(g, np.random.default_rng(1))
    monkeypatch.setattr(rn, "newton_root_scalar", lambda *a, **k: pytest.fail("cell-by-cell Python path used"))
    r = route_step(g, start, h_old, 1.0, old_discharge_m2_s=q_old, root_solver="newton")
    assert r.root_stats["iterated_cells"] > 0
