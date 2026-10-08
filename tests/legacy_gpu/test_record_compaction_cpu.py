"""CPU-only tests of the B2 compact walk-record strategy: the static eligibility rule and its premise, parameter validation before any
allocation, the memory estimate and the kernel provenance. The actual-GPU equivalence tests are in `test_record_compaction_gpu.py`.
Written without being run (Codex executes them)."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from maple_syrup import legacy_native as N
from maple_syrup import legacy_native_cuda as K
from maple_syrup.legacy_native_cuda import (
    CudaLegacyContext,
    CudaLegacyError,
    build_walk_tables,
    eligible_record_classes,
    estimate_bytes,
    kernel_provenance,
    kernel_source,
    resolve_record_classes,
)

from .helpers import GRAPHS, expected_eligible, fraction_pattern, physics_for

needs_numba = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="Numba is not installed")
LIMITS = np.array([1, 1, 3, 3, 6, 6, 12], dtype=np.int64)


def flat(net, pattern):
    graph_shape = tuple(net.shape)
    return fraction_pattern(pattern, graph_shape).reshape(net.active.size, 6)


@pytest.mark.parametrize("name", sorted(GRAPHS))
@pytest.mark.parametrize("pattern", ["all6", "two", "zero", "hetero"])
def test_eligibility_equals_the_independent_active_cell_rule(name, pattern):
    net = N.native_network(GRAPHS[name]())
    fr = flat(net, pattern)
    got = eligible_record_classes(net, fr)
    assert got.dtype == np.int32 and got.ndim == 1 and np.all(np.diff(got) > 0)
    assert got.tolist() == expected_eligible(net.active, fr)
    if pattern == "all6":
        assert got.tolist() == [0, 1, 2, 3, 4, 5]
    if pattern == "two":
        assert got.tolist() == [3, 4]
    if pattern == "zero":
        assert got.size == 0


def test_a_class_present_only_on_inactive_cells_or_with_unusable_fractions_is_not_eligible():
    net = N.native_network(GRAPHS["inactive_neighbour"]())  # cell 0 is inactive
    assert not net.active[0]
    fr = np.zeros((net.active.size, 6))
    fr[0, 2] = 0.9  # only on the inactive cell
    fr[1, 3] = -0.1  # negative
    fr[1, 4] = np.nan  # not positive
    fr[2, 5] = 1e-300  # a positive subnormal-scale fraction IS eligible
    assert eligible_record_classes(net, fr).tolist() == [5]


@pytest.mark.parametrize("bad", [np.zeros((4, 6), dtype=np.int64), np.zeros(6), np.zeros((1, 6)), np.zeros((4, 0)),
                                 np.zeros((4, 33))])
def test_eligibility_rejects_malformed_fractions(bad):
    net = N.native_network(GRAPHS["converging"]())
    with pytest.raises(CudaLegacyError):
        eligible_record_classes(net, bad)


def test_strategies_and_supplied_record_classes_are_validated():
    net = N.native_network(GRAPHS["converging"]())
    fr = flat(net, "two")
    assert resolve_record_classes(net, fr, "all").tolist() == [0, 1, 2, 3, 4, 5]
    assert resolve_record_classes(net, fr, "compact").tolist() == [3, 4]
    assert resolve_record_classes(net, fr, "compact", np.array([1, 3, 4])).tolist() == [1, 3, 4]  # a superset is safe
    assert resolve_record_classes(net, flat(net, "zero"), "compact").size == 0
    for bad in (np.array([3]), np.array([4, 3, 5]), np.array([3, 3, 4]), np.array([3, 4, 6]), np.array([-1, 3, 4]),
                np.array([3.0, 4.0]), np.array([True, False]), np.zeros((2, 2), dtype=np.int64)):
        with pytest.raises(CudaLegacyError):
            resolve_record_classes(net, fr, "compact", bad)  # omitting an eligible class would silently drop deposition
    with pytest.raises(CudaLegacyError, match="compact"):
        resolve_record_classes(net, fr, "all", np.array([3, 4]))
    for bad_strategy in ("COMPACT", "", None, 1, "skip"):
        with pytest.raises(CudaLegacyError, match="strategy"):
            resolve_record_classes(net, fr, bad_strategy)


@needs_numba
def test_the_context_refuses_bad_record_parameters_before_touching_the_device(monkeypatch):
    graph = GRAPHS["converging"]()
    net = N.native_network(graph)
    physics = physics_for(graph, fraction_pattern("two", graph.shape))
    monkeypatch.setattr(K, "_cupy", lambda: (_ for _ in ()).throw(AssertionError("the device must not be touched")))
    common = {"limits": LIMITS, "dt": 1.0, "n_steps": 1}
    for kwargs in ({"record_strategy": "bogus"}, {"record_classes": [3]}, {"record_strategy": "all", "record_classes": [3, 4]},
                   {"record_classes": np.array([4, 3])}, {"record_classes": np.array([3, 9])}):
        with pytest.raises(CudaLegacyError):
            CudaLegacyContext(net, physics, graph, **common, **kwargs)


@pytest.mark.parametrize("ne", [0, 1, 2, 6])
def test_estimate_scales_the_record_values_only_and_keeps_total_consistent(ne):
    net = N.native_network(GRAPHS["converging"]())
    tb = build_walk_tables(net, LIMITS)

    class Physics:
        shape = tuple(net.shape)

    ref = estimate_bytes(net, Physics(), tb.n_records, 100, 6, tb.nbytes())  # default: every class (the B1 numbers + the index array)
    est = estimate_bytes(net, Physics(), tb.n_records, 100, 6, tb.nbytes(), ne)
    assert ref["walk_values"] == tb.n_records * 6 * 8
    assert est["walk_values"] == (tb.n_records * ne * 8 if ne else 8)  # one placeholder element, never a zero-size array
    for key in ("dynamic", "walk_codes", "partials", "ledger"):
        assert est[key] == ref[key]  # the six physical classes keep every other array
    assert est["total"] == sum(est[k] for k in ("static", "dynamic", "walk_values", "walk_codes", "partials", "ledger"))
    if ne < 6:
        assert est["total"] < ref["total"] or tb.n_records == 0
    for bad in (-1, 7):
        with pytest.raises(CudaLegacyError):
            estimate_bytes(net, Physics(), tb.n_records, 100, 6, tb.nbytes(), bad)


def test_the_record_class_count_is_part_of_the_kernel_source_and_the_provenance():
    a, b, c = kernel_source(6), kernel_source(6, 2), kernel_source(6, 0)
    assert "#define NE 6 " in a and "#define NE 2 " in b and "#define NC 6" in b
    assert "#define NE_RECORD 0 " in c and "#define NE 1 " in c  # the divisor is never a constant zero; the true count is NE_RECORD
    assert kernel_source(6, 6) == a  # the default is the all-class reference
    assert len({a, b, c}) == 3
    pa, pb = kernel_provenance(6), kernel_provenance(6, 2)
    assert pa["n_record_classes_compiled"] == 6 and pb["n_record_classes_compiled"] == 2
    assert pa["source_sha256"] != pb["source_sha256"] and pa["n_classes_compiled"] == pb["n_classes_compiled"] == 6
    for bad in ((0, 0), (33, 1), (6, 7), (6, -1), (True, 1), (6, True), (6.0, 2)):
        with pytest.raises(CudaLegacyError):
            kernel_source(*bad)


@needs_numba
@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_premise_a_class_without_composition_has_zero_detachment_walk_and_deposition_but_its_pool_still_evolves(name):
    """The eligibility proof, checked on the accepted CPU engine: classes with zero fraction in every active cell never detach or
    deposit, while a seeded mobile pool and a seeded recession velocity of such a class keep evolving (nothing is skipped)."""
    from maple_syrup.legacy_native_numba import StepEngine, WetLawRunner

    graph = GRAPHS[name]()
    net = N.native_network(graph)
    shape = tuple(graph.shape)
    cpu = StepEngine(net, WetLawRunner(physics_for(graph, fraction_pattern("two", shape))), dt=1.0)
    excluded = [0, 1, 2, 5]
    cpu.M1.reshape(-1, 6)[:, excluded] = np.where(net.active[:, None], 0.25, 0.0)[:, :len(excluded)]
    # a terminal pit may not carry a velocity (a validated invariant), so the recession velocity is seeded elsewhere only
    cpu.v_prev.reshape(-1, 6)[:, excluded] = np.where((net.active & ~net.terminal)[:, None], 0.02, 0.0)[:, :len(excluded)]
    rng = np.random.default_rng(31)
    ny, nx = shape
    active = net.active.reshape(ny, nx)
    moved = False
    for _ in range(5):
        depth = np.where(active & (rng.random(shape) > 0.2), rng.uniform(5e-5, 8e-3, shape), 0.0)
        vel = np.where(active & (depth > 0.0), rng.uniform(0.001, 0.4, shape), 0.0)
        rain = np.where(active & (rng.random(shape) > 0.2), rng.uniform(1e-6, 3e-5, shape), 0.0)
        before = cpu.M1[:, excluded].copy()
        res = cpu.step(depth, vel, rain)
        assert not cpu.det[:, excluded].any() and not cpu.depos[:, excluded].any()
        assert not res.ledger_row[0, excluded].any() and not res.ledger_row[1, excluded].any()  # pickup, active deposition
        assert not res.ledger_row[3, excluded].any() and not res.ledger_row[4, excluded].any()  # ring, inactive
        moved = moved or not np.array_equal(before, cpu.M1[:, excluded])
    assert moved and not cpu.cum_det[:, excluded].any() and not cpu.cum_dep[:, excluded].any()
    assert cpu.cum_det[:, [3, 4]].any()  # the eligible classes do detach
