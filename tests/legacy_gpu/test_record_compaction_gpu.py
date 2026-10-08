"""Real-GPU equivalence of the compact walk-record strategy against the all-class reference strategy and the accepted CPU engine
(queued for Codex on a verified-idle device). Predeclared bounds: sediment rtol 2e-11 / atol 1e-14; integer tallies, flag words and
counts EXACT. Written without being run."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from .helpers import GRAPHS, expected_eligible, fraction_pattern, gpu_available
from .test_gpu_kernels import ATOL, RTOL, build, cpu_walk, states

pytestmark = [pytest.mark.skipif(not gpu_available(), reason="no CuPy / CUDA device"),
              pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="Numba (the CPU oracle) is not installed")]
PATTERNS = ["all6", "two", "zero", "hetero"]


def seed(cp, cpu, gpus, net, classes):
    """The same seeded mobile pools and recession velocities (a terminal pit may not carry a velocity, so those are seeded on other
    active cells only) on the CPU engine and every device context, for the given classes, written IN PLACE (sealed metadata is
    unchanged). Class k gets a distinct pool so a class mix-up cannot pass."""
    n = net.active.size
    for j, k in enumerate(classes):
        mob = np.where(net.active, 0.25 + 0.05 * j, 0.0)
        vel = np.where(net.active & ~net.terminal, 0.02 + 0.005 * j, 0.0)
        cpu.M1[:, k] = mob
        cpu.v_prev[:, k] = vel
        for gpu in gpus:
            gpu.M1.reshape(n, 6)[:, k] = cp.asarray(mob)
            gpu.v_prev.reshape(n, 6)[:, k] = cp.asarray(vel)


def run_steps(cp, net, cpu, gpu, shape, n_steps, rng_seed):
    rows = []
    for row, (depth, vel, rain) in enumerate(states(net, shape, np.random.default_rng(rng_seed), n_steps)):
        res = cpu.step(depth, vel, rain) if cpu is not None else None
        if res is not None:  # the engine reuses preallocated buffers for these arrays on every step: keep OWNED copies per step
            res = SimpleNamespace(ledger_row=res.ledger_row.copy(), walk_counts=res.walk_counts.copy(),
                                  regime_counts=res.regime_counts.copy())
        gpu.step(row, cp.asarray(depth), cp.asarray(vel), cp.asarray(rain))
        gpu.check_flags(row + 1)
        rows.append(res)
    return rows


@pytest.mark.parametrize("name", sorted(GRAPHS))
@pytest.mark.parametrize("pattern", PATTERNS)
def test_compact_equals_the_cpu_engine_and_the_all_class_reference_with_seeded_excluded_pools(name, pattern):
    n_steps = 6
    shape = tuple(GRAPHS[name]().shape)
    frac = fraction_pattern(pattern, shape)
    cp, net, cpu, compact, _ = build(name, n_steps=n_steps, fractions=frac)  # default strategy: compact
    _, _, _, reference, _ = build(name, n_steps=n_steps, fractions=frac, record_strategy="all")
    eligible = expected_eligible(net.active, frac.reshape(net.active.size, 6))
    assert compact.elig_host.tolist() == eligible and compact.ne == len(eligible) and compact.record_strategy == "compact"
    assert reference.ne == 6 and reference.elig_host.tolist() == list(range(6))
    excluded = [k for k in range(6) if k not in eligible]
    if excluded:  # a seeded pool and recession velocity of an excluded class must evolve exactly as in the references
        seed(cp, cpu, (compact, reference), net, excluded)
    cpu_rows = run_steps(cp, net, cpu, compact, shape, n_steps, 17)
    run_steps(cp, net, None, reference, shape, n_steps, 17)
    for row, res in enumerate(cpu_rows):
        np.testing.assert_allclose(compact.host_ledger[row], res.ledger_row, rtol=RTOL, atol=ATOL, err_msg=f"{name} {pattern} {row}")
        assert np.array_equal(compact.host_counts[row, :9], res.walk_counts[:9]) and np.array_equal(compact.host_counts[row, 10:], res.regime_counts)
    # compact against the all-class reference on the device: integers and flag words exact, sediment at the declared bound
    assert np.array_equal(compact.host_flags, reference.host_flags) and np.array_equal(compact.host_counts, reference.host_counts)
    np.testing.assert_allclose(compact.host_ledger, reference.host_ledger, rtol=RTOL, atol=ATOL)
    mc, mr = compact.download_maps(), reference.download_maps()
    for key in mc:
        np.testing.assert_allclose(mc[key], mr[key], rtol=RTOL, atol=ATOL, err_msg=f"{name} {pattern} {key}")
    for key, ref in (("cum_det", cpu.cum_det), ("cum_dep", cpu.cum_dep), ("cum_clip", cpu.cum_clip), ("mobile", cpu.M1)):
        np.testing.assert_allclose(mc[key], ref, rtol=RTOL, atol=ATOL, err_msg=f"{name} {pattern} {key}")
    for k in excluded:  # exact zeros of the excluded classes' detachment-derived maps; their seeded pools are NOT zero
        assert not mc["cum_det"][:, k].any() and not mc["cum_dep"][:, k].any()
        assert not compact.host_ledger[:, [0, 1, 3, 4], k].any()
    if excluded:
        assert mc["mobile"][:, excluded].any()
    # memory: the record array is exactly records x record classes (one placeholder element when none)
    expect = compact.tables.n_records * compact.ne * 8 if compact.ne else 8
    assert compact.values.nbytes == expect == compact.estimate["walk_values"]
    assert reference.values.nbytes == reference.tables.n_records * 6 * 8


def test_zero_eligible_classes_launch_no_record_kernel_and_still_run_the_physics():
    cp, net, cpu, compact, shape = build("converging", n_steps=3, fractions=fraction_pattern("zero", (3, 3)))
    _, _, _, reference, _ = build("converging", n_steps=3, fractions=fraction_pattern("zero", (3, 3)), record_strategy="all")
    assert compact.ne == 0 and compact.values.size == 1 and compact.record_info["n_record_classes"] == 0
    run_steps(cp, net, cpu, compact, shape, 3, 5)
    run_steps(cp, net, None, reference, shape, 3, 5)
    assert reference.stats["launches"] - compact.stats["launches"] == 3 * 3  # values, gather, ring/inactive skipped per step
    assert np.array_equal(compact.host_counts, reference.host_counts)
    compact.reset()  # the test hook needs a fresh row-0 context; every launch/tally assertion above has been made
    got = compact.walk_only(np.zeros((net.active.size, 6)), np.ones((net.active.size, 6)), np.ones((net.active.size, 6), dtype=bool),
                            np.full((net.active.size, 6), 5, dtype=np.int8))
    assert got["flag"] == 0 and not got["depos"].any() and not got["counts"][:9].any()


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_forced_walk_on_eligible_classes_matches_the_cpu_rules_and_the_reference_strategy(name):
    shape = tuple(GRAPHS[name]().shape)
    frac = fraction_pattern("two", shape)
    _, net, _, compact, _ = build(name, n_steps=1, fractions=frac)
    _, _, _, reference, _ = build(name, n_steps=1, fractions=frac, record_strategy="all")
    n, nc = net.active.size, 6
    rng = np.random.default_rng(9)
    for trial in range(3):
        compact.reset()
        reference.reset()
        det = np.zeros((n, nc))
        det[:, [3, 4]] = np.where(net.active[:, None], rng.uniform(0.1, 1.0, (n, 2)), 0.0)  # source-order / pit / ring / inactive paths
        inv = rng.uniform(0.2, 5.0, (n, nc))
        law = rng.random((n, nc)) > 0.3
        regime = rng.integers(2, 7, (n, nc)).astype(np.int8)
        regime[:, 3] = 2  # diffuse + no law on zero-slope cells: the "no walk, no deposit" branch
        law[:, 3] = False
        a, b = compact.walk_only(det, inv, law, regime), reference.walk_only(det, inv, law, regime)
        depos, ring, inactive, counts = cpu_walk(net, det, inv, law, regime)
        np.testing.assert_allclose(a["depos"][net.active], depos[net.active], rtol=RTOL, atol=ATOL, err_msg=f"{name} {trial}")
        np.testing.assert_allclose(a["ring_kg_per_step"], ring, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(a["inactive_kg_per_step"], inactive, rtol=RTOL, atol=ATOL)
        assert np.array_equal(a["counts"][:9], counts[:9]) and a["flag"] == 0
        assert np.array_equal(a["counts"], b["counts"]) and np.array_equal(a["code"], b["code"])
        np.testing.assert_allclose(a["depos"], b["depos"], rtol=RTOL, atol=ATOL)
        assert not a["depos"][:, [0, 1, 2, 5]].any() and not a["ring_kg_per_step"][[0, 1, 2, 5]].any()


def test_forced_detachment_in_a_non_record_class_is_refused_before_any_mutation_and_runs_under_the_all_strategy():
    from maple_syrup.legacy_native_cuda import CudaLegacyError

    frac = fraction_pattern("two", (3, 3))
    _, net, _, compact, _ = build("converging", n_steps=1, fractions=frac)
    _, _, _, reference, _ = build("converging", n_steps=1, fractions=frac, record_strategy="all")
    n, nc = net.active.size, 6
    det = np.where(net.active[:, None], 0.5, 0.0) * np.ones((n, nc))  # includes the excluded classes 0, 1, 2, 5
    inv, law, regime = np.ones((n, nc)), np.ones((n, nc), dtype=bool), np.full((n, nc), 5, dtype=np.int8)
    launches, uploaded = compact.stats["launches"], compact.stats.get("h2d_test_bytes", 0)
    before = compact.det.copy()
    with pytest.raises(CudaLegacyError, match="non-record"):
        compact.walk_only(det, inv, law, regime)
    assert compact.stats["launches"] == launches and compact.stats.get("h2d_test_bytes", 0) == uploaded
    assert bool((compact.det == before).all())  # nothing was written
    got = reference.walk_only(det, inv, law, regime)  # the declared full-class diagnostic path
    depos, ring, _, counts = cpu_walk(net, det, inv, law, regime)
    np.testing.assert_allclose(got["depos"][net.active], depos[net.active], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(got["ring_kg_per_step"], ring, rtol=RTOL, atol=ATOL)
    assert np.array_equal(got["counts"][:9], counts[:9])


def test_compact_replay_is_bitwise_repeatable_after_reset_and_equals_a_second_context():
    frac = fraction_pattern("two", (3, 3))
    cp, net, _, gpu, shape = build("converging", n_steps=6, fractions=frac)
    _, _, _, other, _ = build("converging", n_steps=6, fractions=frac)
    seq = list(states(net, shape, np.random.default_rng(3), 6))

    def run(ctx):
        for row, (depth, vel, rain) in enumerate(seq):
            ctx.step(row, cp.asarray(depth), cp.asarray(vel), cp.asarray(rain))
        ctx.check_flags()
        return ctx.host_ledger.copy(), ctx.host_counts.copy(), ctx.host_flags.copy(), ctx.download_maps()

    first = run(gpu)
    gpu.reset()
    second = run(gpu)
    third = run(other)
    for x in (second, third):
        assert np.array_equal(first[0], x[0]) and np.array_equal(first[1], x[1]) and np.array_equal(first[2], x[2])
        assert all(np.array_equal(first[3][k], x[3][k]) for k in first[3])


def test_the_seal_covers_the_record_strategy_and_the_class_index_array():
    from maple_syrup.legacy_native_cuda import CudaLegacyError

    cp, _, _, gpu, shape = build("converging", n_steps=2, fractions=fraction_pattern("two", (3, 3)))
    good = [cp.asarray(np.full(shape, v)) for v in (1e-3, 0.05, 1e-5)]
    launches = gpu.stats["launches"]
    for attr, value in (("ne", 6), ("record_strategy", "all")):
        old = getattr(gpu, attr)
        setattr(gpu, attr, value)
        with pytest.raises(CudaLegacyError, match="sealed|changed"):
            gpu.step(0, *good)
        setattr(gpu, attr, old)
    old = gpu.elig_host
    gpu.elig_host = np.array([0, 1], dtype=np.int32)
    with pytest.raises(CudaLegacyError, match="sealed|changed"):
        gpu.step(0, *good)
    gpu.elig_host = old
    original = gpu.d_elig
    gpu.d_elig = cp.zeros_like(original)
    with pytest.raises(CudaLegacyError, match="d_elig"):
        gpu.step(0, *good)
    gpu.d_elig = original
    assert gpu.stats["launches"] == launches


def test_explicit_superset_record_classes_run_identically_and_a_subset_is_refused():
    from maple_syrup.legacy_native_cuda import CudaLegacyError

    frac = fraction_pattern("two", (3, 3))
    cp, net, _, a, shape = build("converging", n_steps=4, fractions=frac)
    _, _, _, b, _ = build("converging", n_steps=4, fractions=frac, record_classes=np.array([1, 3, 4]))
    assert b.ne == 3 and b.elig_host.tolist() == [1, 3, 4]
    for row, (depth, vel, rain) in enumerate(states(net, shape, np.random.default_rng(2), 4)):
        for ctx in (a, b):
            ctx.step(row, cp.asarray(depth), cp.asarray(vel), cp.asarray(rain))
    a.check_flags()
    b.check_flags()
    assert np.array_equal(a.host_ledger, b.host_ledger) and np.array_equal(a.host_counts, b.host_counts)
    with pytest.raises(CudaLegacyError, match="omit"):
        build("converging", n_steps=1, fractions=frac, record_classes=np.array([3]))


def test_the_memory_budget_sees_the_strategy():
    """The compact estimate is below the reference estimate by exactly the dropped record values and the budget refusal uses it."""
    frac = fraction_pattern("two", (3, 3))
    _, _, _, compact, _ = build("converging", n_steps=2, fractions=frac)
    _, _, _, reference, _ = build("converging", n_steps=2, fractions=frac, record_strategy="all")
    n_rec = compact.tables.n_records
    assert reference.estimate["walk_values"] - compact.estimate["walk_values"] == n_rec * 4 * 8
    assert reference.estimate["total"] - compact.estimate["total"] == n_rec * 4 * 8 + (6 - 2) * 4  # + the class-index array
    for ctx in (compact, reference):
        assert ctx.values.nbytes == ctx.estimate["walk_values"]
        assert ctx.record_info["record_values_bytes_estimated"] == ctx.estimate["walk_values"]
