"""CW-B15 item 5: exact zero-conveyance root in the original bisection solvers.

At a cell with conveyance k == 0 exactly (a recognised terminal-storage cell; strict graphs refuse k == 0 on active
cells) the routing equation h + c k h^{3/2} = rhs reduces to h = rhs. The solvers now return lo = rhs there for finite
rhs > 0 instead of approaching it to rhs 2^-iterations. Every k > 0 cell, rhs <= 0 and non-finite rhs must be
bit-identical to the frozen B12 solvers. The 1e-11 m tolerance and the 40-iteration default are unchanged."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("maple")

from rfid_helpers import build, pit_chain
from test_routing import chain_full

from maple_syrup import routing, routing_cuda
from maple_syrup import routing_numba as rn
from maple_syrup.routing import (
    DEFAULT_BISECTION_ITERATIONS,
    DEFAULT_ROOT_TOLERANCE_M,
    route_step,
)

NUMBA = pytest.mark.skipif(not rn.numba_available(), reason="Numba not installed; compiled path not exercised")
C = 2.0  # dt / dx of a 1 s step on the 0.5 m helper grid
PIT_ROW = 3  # full-grid row of the strict pit in `pit_chain(6, 3)`; interior (2, 0)


def bits(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.uint64)


def old_and_new_bisect(frozen, rhs, k, iterations=DEFAULT_BISECTION_ITERATIONS):
    out = []
    for module in (frozen["routing"], routing):
        n = rhs.size
        lo = np.empty(n)
        bufs = (np.empty(n), np.empty(n), np.empty(n), np.empty(n, dtype=np.bool_))
        module._bisect(lo, rhs.copy(), k.copy(), C, iterations, *bufs, np)
        out.append(lo)
    return out


def test_array_bisect_exact_only_at_zero_conveyance_finite_positive_rhs(frozen_b12):
    rhs = np.array([5.0, 0.0, -0.0, 12.5, 12.5, np.inf, -1.0, np.nan, 5e-320, 1e300, 10.9984820351, 3.0e-3])
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.7])
    old, new = old_and_new_bisect(frozen_b12, rhs, k)
    exact = (k == 0.0) & (rhs > 0.0) & np.isfinite(rhs)
    assert np.array_equal(bits(new[exact]), bits(rhs[exact]))  # h = rhs, exactly
    assert np.array_equal(bits(new[~exact]), bits(old[~exact]))  # everything else bit-identical to frozen B12
    # The old 40-iteration result misses the unchanged 1e-11 m tolerance above ~10.995 m (science review D1).
    deep = exact & (rhs > 1e-11 * 2.0 ** DEFAULT_BISECTION_ITERATIONS)
    assert deep.sum() == 3 and np.all(rhs[deep] - old[deep] > DEFAULT_ROOT_TOLERANCE_M)
    assert np.all(rhs[exact & ~deep] - old[exact & ~deep] <= DEFAULT_ROOT_TOLERANCE_M)


def graphs():
    pit = build(pit_chain(6, PIT_ROW), allow_pit_storage=True)
    strict = build(chain_full(6))
    return {"pit": pit, "strict": strict}


def base_cases(n, seed=11):
    rng = np.random.default_rng(seed)
    wet = rng.uniform(1e-4, 3e-3, n)
    return {"wet": wet, "dry": np.zeros(n), "deep": wet + 11.5, "very_deep": wet * 1e3 + 40.0,
            "mixed": wet * (rng.random(n) < 0.5), "negative": -wet}


def frozen_sweep(frozen, name, graph, base, iterations=DEFAULT_BISECTION_ITERATIONS):
    if name == "array":
        return frozen["routing"]._sweep_array(graph, np.array(base), C, iterations, np)
    import numba

    fn = numba.njit(fastmath=False, nogil=True, boundscheck=False)(
        getattr(frozen["routing_numba"], "_sweep" if name == "serial" else "_sweep_batched"))
    return call(fn, graph, base, iterations)


def call(fn, graph, base, iterations=DEFAULT_BISECTION_ITERATIONS):
    outs = tuple(np.full(graph.n_active, 7.0) for _ in range(4))
    fn(np.asarray(graph.level_bounds, dtype=np.int64), np.array(graph.conveyance_lo), graph.donor_position,
       graph.donor_mask, np.array(base, dtype=np.float64), C, int(iterations), *outs)
    return outs


def candidate_sweep(name, graph, base, iterations=DEFAULT_BISECTION_ITERATIONS):
    if name == "array":
        return routing._sweep_array(graph, np.array(base), C, iterations, np)
    fn = rn.compiled_sweep() if name == "serial" else rn.compiled_sweep_batched()
    return call(fn, graph, base, iterations)


IMPLEMENTATIONS = ["array", pytest.param("serial", marks=NUMBA), pytest.param("batched", marks=NUMBA)]


@pytest.mark.parametrize("name", IMPLEMENTATIONS)
@pytest.mark.parametrize("case", ["wet", "dry", "deep", "very_deep", "mixed", "negative"])
def test_sweeps_match_frozen_b12_except_exact_zero_conveyance_lanes(frozen_b12, name, case):
    for label, graph in graphs().items():
        k = np.asarray(graph.conveyance_lo)
        assert (label == "strict") == (not np.any(k == 0.0))  # strict graphs carry no k == 0 active cell
        base = base_cases(graph.n_active)[case]
        old = frozen_sweep(frozen_b12, name, graph, base)
        new = candidate_sweep(name, graph, base)
        qin, q, flow, rhs = new
        for field, a, b in zip(("qin", "q", "rhs"), old[:2] + old[3:], (qin, q, rhs), strict=True):
            assert np.array_equal(bits(a), bits(b)), (label, field)  # discharge, inflow, RHS: unchanged everywhere
        exact = (k == 0.0) & (rhs > 0.0) & np.isfinite(rhs)
        assert np.array_equal(bits(flow[~exact]), bits(old[2][~exact])), label
        assert np.array_equal(bits(flow[exact]), bits(rhs[exact])), label
        assert np.all(q[k == 0.0] == 0.0)
        if label == "strict":
            assert not exact.any()


@NUMBA
@pytest.mark.parametrize("case", ["wet", "dry", "deep", "very_deep", "mixed", "negative"])
def test_array_serial_and_batched_candidates_agree_bitwise(case):
    for graph in graphs().values():
        base = base_cases(graph.n_active)[case]
        ref = candidate_sweep("array", graph, base)
        for name in ("serial", "batched"):
            for a, b in zip(ref, candidate_sweep(name, graph, base), strict=True):
                assert np.array_equal(bits(a), bits(b)), (name, case)


def deep_pit_state(depth_pit):
    graph = build(pit_chain(6, PIT_ROW), allow_pit_storage=True)
    depth = np.zeros(graph.shape)
    depth[3:, 0] = 3.0e-3
    depth[2, 0] = depth_pit
    return graph, depth


@pytest.mark.parametrize("implementation", ["array", pytest.param("numba", marks=NUMBA)])
@pytest.mark.parametrize("depth_pit", [0.0, 1.0, 10.9984820351, 12.0, 53.9747])
def test_route_step_terminal_pond_is_exact_with_the_default_40_iterations(implementation, depth_pit):
    graph, depth = deep_pit_state(depth_pit)
    step = route_step(graph, depth, np.zeros(graph.shape), 1.0, implementation=implementation)
    assert step.bisection_iterations == DEFAULT_BISECTION_ITERATIONS == 40
    pit = (2, 0)
    assert step.discharge_m2_s[pit] == 0.0 and step.velocity_m_s[pit] == 0.0
    assert step.flow_depth_m[pit] == step.depth_m[pit]  # exact h = rhs
    assert step.depth_m[pit] >= depth_pit
    assert step.max_constitutive_residual_m <= DEFAULT_ROOT_TOLERANCE_M


@NUMBA
@pytest.mark.parametrize("depth_pit", [0.0, 12.0, 53.9747])
def test_route_step_array_and_numba_are_bit_identical_on_terminal_ponds(depth_pit):
    graph, depth = deep_pit_state(depth_pit)
    a = route_step(graph, depth, np.zeros(graph.shape), 1.0, implementation="array")
    b = route_step(graph, depth, np.zeros(graph.shape), 1.0, implementation="numba")
    for name in ("depth_m", "discharge_m2_s", "flow_depth_m", "velocity_m_s"):
        assert np.array_equal(bits(getattr(a, name)), bits(getattr(b, name))), name


def test_frozen_b12_refuses_the_deep_pond_that_the_candidate_solves(frozen_b12):
    """The before state: frozen B12 bisection leaves |h_new - lo| > 1e-11 m at a 12 m terminal pond (40 iterations)."""
    graph, depth = deep_pit_state(12.0)
    base = graph.xp.asarray(depth.reshape(-1)[np.asarray(graph.level_order)])
    _qin, q, flow, rhs = frozen_b12["routing"]._sweep_array(graph, base, C, DEFAULT_BISECTION_ITERATIONS, np)
    k = np.asarray(graph.conveyance_lo)
    h_new = rhs - q * C
    assert np.max(np.abs(h_new - flow)[k == 0.0]) > DEFAULT_ROOT_TOLERANCE_M
    _qin2, q2, flow2, rhs2 = routing._sweep_array(graph, base, C, DEFAULT_BISECTION_ITERATIONS, np)
    assert np.max(np.abs((rhs2 - q2 * C) - flow2)[k == 0.0]) == 0.0


# --- kernel text: the CUDA change is exactly the documented branch (CPU-checkable; GPU execution is the root's) ---
B12_SHA = {  # the accepted baseline kernel texts (tests/gpu_newton/test_sweep.py BASELINE_SHA, unchanged since d5861b4)
    "level": "2189717bbc88f79ddf7f88e08115871c7a644854faaf0c90ea7734810f7e3e49",
    "block": "ad2911e51fa7b610722c5cc6d9c6e6c6913f9c23bd0880e921a3d477c5a44413",
    "hydrology": "4743809f27f9f1daefbd26545a18891a1299c636748060acfc0e81c5c130033f",
}
B15_OLD = "    if (rhs > 0.0) {\n"
B15_NEW = ("    if (rhs > 0.0 && k == 0.0 && rhs <= 1.7976931348623157e+308) {\n"
           "        lo = rhs;\n"
           "    } else if (rhs > 0.0) {\n")


def kernel_texts():
    from maple_syrup import hydrology_cuda as hc

    return {"level": routing_cuda.kernel_source(), "block": routing_cuda.block_kernel_source(),
            "hydrology": hc.kernel_source()}


def test_cuda_kernel_texts_are_the_baseline_plus_exactly_the_zero_conveyance_branch():
    sha = lambda t: hashlib.sha256(t.encode()).hexdigest()
    for name, text in kernel_texts().items():
        assert text.count(B15_NEW) == 1 and B15_OLD not in text, name
        assert sha(text.replace(B15_NEW, B15_OLD)) == B12_SHA[name], name


# --- actual device parity: written for the root's GPU0 run; skipped (no claim) without a device ---
@NUMBA
@pytest.mark.parametrize("mode", ["level", "block"])
@pytest.mark.parametrize("case", ["wet", "deep", "very_deep", "mixed", "negative"])
def test_cuda_sweep_bit_identical_to_numba_on_terminal_and_strict_graphs(gpu, mode, case):
    cp = gpu
    for graph in graphs().values():
        dev = build(np.asarray(graph_full(graph)), xp=cp, **graph_policy(graph))
        base = base_cases(graph.n_active)[case]
        ref = candidate_sweep("serial", graph, base)
        got = routing_cuda.run_sweep(dev, cp.asarray(base), C, DEFAULT_BISECTION_ITERATIONS, mode=mode)
        for a, b in zip(ref, got, strict=True):
            assert np.array_equal(bits(a), bits(cp.asnumpy(b)))


@pytest.mark.parametrize("depth_pit", [0.0, 12.0, 53.9747])
def test_cuda_route_step_terminal_pond_exact_and_matches_array(gpu, depth_pit):
    cp = gpu
    graph, depth = deep_pit_state(depth_pit)
    dev = build(pit_chain(6, PIT_ROW), allow_pit_storage=True, xp=cp)
    ref = route_step(graph, depth, np.zeros(graph.shape), 1.0, implementation="array")
    got = route_step(dev, cp.asarray(depth), cp.zeros(graph.shape), 1.0, implementation="cuda")
    pit = (2, 0)
    flow, h = cp.asnumpy(got.flow_depth_m), cp.asnumpy(got.depth_m)
    assert flow[pit] == h[pit] and float(got.max_constitutive_residual_m) <= DEFAULT_ROOT_TOLERANCE_M
    for name in ("depth_m", "discharge_m2_s", "flow_depth_m"):
        assert np.array_equal(bits(getattr(ref, name)), bits(cp.asnumpy(getattr(got, name)))), name


def graph_full(graph):
    return pit_chain(6, PIT_ROW) if np.any(np.asarray(graph.conveyance_lo) == 0.0) else chain_full(6)


def graph_policy(graph):
    return {"allow_pit_storage": True} if np.any(np.asarray(graph.conveyance_lo) == 0.0) else {}
