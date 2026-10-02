"""Phase 7j: `routing_numba.compiled_sweep_batched()` against the ORIGINAL `compiled_sweep()` (the oracle).

Every output (qin, q, flow depth, rhs) is compared BIT FOR BIT through `uint64` views. Where both sides produce a NaN
the NaN positions and signs must agree (payload bits are not part of the contract). The original sweep is never
modified and is run on the same inputs. Nothing here was run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import functools
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_routing import chain_full, make_graph, random_full, valley_full

pytest.importorskip("maple")

from maple_syrup import routing_numba as rn

pytestmark = pytest.mark.skipif(not rn.numba_available(), reason="Numba not installed; no claim made")

SPECIAL_RHS = [0.0, -0.0, -1e-3, np.nan, np.inf, -np.inf, 5e-324, 1e-300, 1e-12, 3e-3, 1e200, 1e308, -5e-324]
ITERATIONS = [1, 2, 5, 40, 200]


# --- helpers ----------------------------------------------------------------------------------------------------
GRAPH_NAMES = ("single_cell", "chain_width1", "valley_branching", "valley_wide_levels", "random")


def single_cell_network():
    """Raw one-cell dependency network (n_active = 1, one level, no donors, positive finite conveyance). The
    production graph builder refuses a one-cell chain (its legacy edge rule reads a ring slope), and that guard is
    not relaxed; the sweep kernels only read these five attributes, so a raw network is a legitimate kernel input."""
    position = np.zeros((4, 1), dtype=np.int64)
    mask = np.zeros((4, 1), dtype=np.bool_)
    # Match the immutable donor arrays owned by production RoutingGraph instances.
    position.flags.writeable = False
    mask.flags.writeable = False
    return SimpleNamespace(n_active=1, level_bounds=(0, 1), conveyance_lo=np.array([0.7], dtype=np.float64),
                           donor_position=position, donor_mask=mask)


@functools.cache
def get_graph(name):
    rng = np.random.default_rng(5)
    if name == "single_cell":
        return single_cell_network()
    if name == "chain_width1":
        return make_graph(chain_full(9), ff=5.0)
    if name == "valley_branching":
        return make_graph(valley_full(6, 5), ff=5.0)
    if name == "valley_wide_levels":
        return make_graph(valley_full(3, 41), ff=5.0)
    if name == "random":
        return make_graph(random_full(rng, 14, 9), ff=rng.uniform(5.0, 30.0, (14, 9)))
    raise AssertionError(name)


def graphs():
    return {name: get_graph(name) for name in GRAPH_NAMES}


def args(graph, base, c, iterations, k=None, sentinel=7.0):
    n = graph.n_active
    outs = tuple(np.full(n, sentinel) for _ in range(4))
    return (np.asarray(graph.level_bounds, dtype=np.int64), np.array(graph.conveyance_lo if k is None else k),
            graph.donor_position, graph.donor_mask, np.array(base, dtype=np.float64), float(c), int(iterations),
            *outs)


def run(dispatcher, graph, base, c, iterations, k=None, sentinel=7.0):
    a = args(graph, base, c, iterations, k, sentinel)
    dispatcher(*a)
    return a[7:]


def assert_same_bits(new, old, label):
    new, old = np.asarray(new), np.asarray(old)
    nan_new, nan_old = np.isnan(new), np.isnan(old)
    np.testing.assert_array_equal(nan_new, nan_old, err_msg=f"{label}: NaN positions")
    np.testing.assert_array_equal(np.signbit(new[nan_new]), np.signbit(old[nan_old]), err_msg=f"{label}: NaN sign")
    keep = ~nan_new
    np.testing.assert_array_equal(new[keep].view(np.uint64), old[keep].view(np.uint64), err_msg=f"{label}: bits")


def compare_all(graph, base, c, iterations, k=None, label=""):
    old = run(rn.compiled_sweep(), graph, base, c, iterations, k)
    new = run(rn.compiled_sweep_batched(), graph, base, c, iterations, k)
    for name, a, b in zip(("qin", "q", "flow", "rhs"), new, old, strict=True):
        assert_same_bits(a, b, f"{label} {name}")
    return old, new


def wet_dry_bases(graph, rng):
    n = graph.n_active
    positive = rng.uniform(0.0, 3e-3, n)
    mixed = positive * (rng.random(n) > 0.5)  # ~half the cells have a zero own base
    donor_wetting = np.zeros(n)  # only the highest-level cells hold water; everything else wets via donors only
    donor_wetting[: max(1, n // 6)] = 2e-3
    special = np.array([SPECIAL_RHS[i % len(SPECIAL_RHS)] for i in range(n)])
    return {"all_dry": np.zeros(n), "wet": positive, "mixed": mixed, "donor_wetting_zero_own_base": donor_wetting,
            "special_values": special}


# --- bitwise parity ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", GRAPH_NAMES)
@pytest.mark.parametrize("iterations", ITERATIONS)
@pytest.mark.parametrize("c", [1.0 / 3.0, 1e-8, 1.0])
def test_batched_sweep_is_bitwise_identical_to_the_original(name, iterations, c):
    graph = graphs()[name]
    rng = np.random.default_rng(11)
    for label, base in wet_dry_bases(graph, rng).items():
        compare_all(graph, base, c, iterations, label=f"{name}/{label}")


@pytest.mark.parametrize("name", ["chain_width1", "valley_branching", "random"])
@pytest.mark.parametrize("k_value", [1e-300, 1e-12, 0.1, 3.0, 1e6, 1e300])
@pytest.mark.parametrize("c", [1e-300, 1e-6, 1e6])
def test_finite_positive_extreme_k_and_c_are_bitwise_identical(name, k_value, c):
    graph = graphs()[name]
    n = graph.n_active
    rng = np.random.default_rng(3)
    for label, base in wet_dry_bases(graph, rng).items():
        compare_all(graph, base, c, 40, k=np.full(n, k_value), label=f"{name}/{label}")


@pytest.mark.parametrize("k_value", [0.0, np.inf, np.nan, -2.0])
def test_raw_kernel_with_invalid_conveyance_keeps_the_same_formula(k_value):
    """The routing graph guarantees finite positive k; the raw kernel still has the same behaviour for invalid k
    (q = (sqrt(lo) * lo) * k even where lo = 0, so 0 * inf is NaN in both)."""
    graph = graphs()["valley_branching"]
    n = graph.n_active
    rng = np.random.default_rng(4)
    for label, base in wet_dry_bases(graph, rng).items():
        compare_all(graph, base, 0.25, 40, k=np.full(n, k_value), label=label)


def test_nonpositive_rhs_leaves_flow_exactly_positive_zero_and_stores_the_raw_rhs():
    graph = graphs()["chain_width1"]
    n = graph.n_active
    base = np.array([-0.0, 0.0, -1e-3, np.nan, -np.inf, 0.0, -5e-324, 0.0, 0.0][:n])
    _old, new = compare_all(graph, base, 0.5, 40)
    _qin, q, flow, rhs = new
    # cell 0 is the level-0 head of the chain (no donors): rhs = base + 0.0 * c
    assert rhs[0] == 0.0
    assert flow[0] == 0.0 and not np.signbit(flow[0])
    assert q[0] == 0.0 and not np.signbit(q[0])
    assert np.isnan(rhs[3])  # raw NaN is stored, the refusal checks stay downstream
    assert flow[3] == 0.0 and q[3] == 0.0


def test_tiny_subnormal_and_infinite_positive_rhs_run_the_unchanged_operations():
    graph = graphs()["single_cell"]
    # Bitwise against the original for every value (including 1e308 and +inf, where all 40 trial depths overflow the
    # flux test and the original leaves lo = 0: an unchanged, separately refused outcome). Only values the original
    # bisection can actually resolve are additionally required to be solved (not skipped): rhs = 5e-324 halves to 0.
    for value in (5e-324, 1e-310, 1e-300, 2.2250738585072014e-308, 1e-12, 3e-3, 1e308, np.inf):
        _old, new = compare_all(graph, np.array([value]), 0.5, 40, label=repr(value))
        if value in (1e-300, 1e-12, 3e-3):
            assert new[2][0] > 0.0  # flow depth really solved, not skipped


def test_zero_base_cells_fed_by_a_wet_donor_are_solved_not_skipped():
    graph = graphs()["chain_width1"]
    n = graph.n_active
    base = np.zeros(n)
    base[np.asarray(graph.level_bounds)[0]] = 3e-3  # head of the chain only
    _old, new = compare_all(graph, base, 0.5, 40)
    flow = new[2]
    assert np.all(flow > 0.0), "every cell downstream of a wet donor has a positive RHS and must be solved"


@pytest.mark.parametrize("sentinel", [0.0, 7.0, -0.0, np.nan, np.inf])
def test_every_output_is_overwritten_whatever_the_sentinel(sentinel):
    graph = graphs()["random"]
    base = wet_dry_bases(graph, np.random.default_rng(8))["mixed"]
    old = run(rn.compiled_sweep(), graph, base, 0.4, 40, sentinel=0.0)
    new = run(rn.compiled_sweep_batched(), graph, base, 0.4, 40, sentinel=sentinel)
    for name, a, b in zip(("qin", "q", "flow", "rhs"), new, old, strict=True):
        assert_same_bits(a, b, name)


def test_repeated_calls_and_reused_output_arrays_are_stable_and_inputs_are_not_modified():
    graph = graphs()["valley_branching"]
    base = wet_dry_bases(graph, np.random.default_rng(9))["wet"]
    a = args(graph, base, 0.3, 40)
    inputs_before = [np.array(x, copy=True) for x in a[:5]]
    dispatcher = rn.compiled_sweep_batched()
    dispatcher(*a)
    first = [np.array(x, copy=True) for x in a[7:]]
    dispatcher(*a)  # outputs now hold the first result; they must be recomputed identically
    for x, y in zip(a[7:], first, strict=True):
        np.testing.assert_array_equal(x.view(np.uint64), y.view(np.uint64))
    for x, y in zip(inputs_before, a[:5], strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("name", GRAPH_NAMES)
def test_solved_flow_satisfies_the_accepted_predicate_without_clipping(name):
    graph = graphs()[name]
    k = np.asarray(graph.conveyance_lo)
    rng = np.random.default_rng(12)
    base = wet_dry_bases(graph, rng)["mixed"]
    c = 0.5 / 3.0
    _qin, q, flow, rhs = run(rn.compiled_sweep_batched(), graph, base, c, 40)
    pos = rhs > 0.0
    assert np.all(flow[pos] >= 0.0) and np.all(flow[pos] <= rhs[pos])
    t = ((np.sqrt(flow) * flow) * k) * c + flow
    assert np.all(t[pos] < rhs[pos])  # the strict bisection test
    assert np.all((rhs - q * c)[pos] >= flow[pos])  # storage identity keeps h_new >= flow


# --- dispatcher contract ----------------------------------------------------------------------------------------
def test_original_dispatcher_is_untouched_and_the_batched_one_is_separate_with_one_signature():
    rn.reset_compiled()
    original = rn.compiled_sweep()
    batched = rn.compiled_sweep_batched()
    assert original is not batched and original is rn.compiled_sweep() and batched is rn.compiled_sweep_batched()
    assert original.py_func is rn._sweep and batched.py_func is rn._sweep_batched
    for name, graph in graphs().items():
        compare_all(graph, np.full(graph.n_active, 1e-3), 0.5, 40, label=name)
    assert len(batched.signatures) == 1 and len(original.signatures) == 1
    for d in (original, batched):
        assert d.targetoptions.get("fastmath", False) is False
        assert not d.targetoptions.get("parallel", False)


def test_reset_compiled_drops_both_dispatchers():
    first = (rn.compiled_sweep(), rn.compiled_sweep_batched())
    rn.reset_compiled()
    assert rn._SWEEP is None and rn._BATCHED is None
    assert rn.compiled_sweep() is not first[0] and rn.compiled_sweep_batched() is not first[1]


def test_missing_numba_is_an_explicit_error_for_both_sweeps(monkeypatch):
    rn.reset_compiled()
    monkeypatch.setitem(sys.modules, "numba", None)
    for factory in (rn.compiled_sweep, rn.compiled_sweep_batched):
        with pytest.raises(rn.NumbaUnavailableError, match="no fallback"):
            factory()
    monkeypatch.undo()
    rn.reset_compiled()
