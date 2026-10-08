"""Real-GPU tests of the CUDA legacy replay against the accepted CPU kernels (queued for Codex on a verified-idle device).

Skipped ONLY when CuPy or a CUDA device is genuinely unavailable (or Numba, for the CPU oracle); a compile, launch or scientific
failure FAILS. Predeclared bounds: sediment rtol 2e-11 / atol 1e-14; integer tallies and branch masks exact. Written without being run."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from .helpers import GRAPHS, gpu_available

pytestmark = [pytest.mark.skipif(not gpu_available(), reason="no CuPy / CUDA device"),
              pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="Numba (the CPU oracle) is not installed")]
RTOL, ATOL = 2.0e-11, 1.0e-14
FRACTIONS = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])  # all six classes carry mass


def build(name, n_steps=8, fractions=None, **ctx_kwargs):
    """`fractions`: None (every class carries mass), a (6,) vector, or a full (ny, nx, 6) array of composition fractions. The CPU
    engine and the GPU context share the same physics. `ctx_kwargs` go to the GPU context (e.g. `record_strategy='all'`)."""
    import cupy as cp

    from maple_syrup import legacy_native as N
    from maple_syrup.legacy_native_cuda import CudaLegacyContext
    from maple_syrup.legacy_native_numba import StepEngine, WetLawRunner
    from maple_syrup.legacy_physics_numba import prepare_legacy_physics
    from maple_syrup.sediment_physics import (
        physics_grid_from_graph,
        plot1_sediment_parameters,
    )

    graph = GRAPHS[name]()
    ny, nx = graph.shape
    net = N.native_network(graph)
    frac = FRACTIONS if fractions is None else np.asarray(fractions, dtype=np.float64)
    holdings = (np.broadcast_to(frac * 2.5, (ny, nx, 6)) if frac.ndim == 1 else frac * 2.5).copy()
    physics = prepare_legacy_physics(plot1_sediment_parameters(), physics_grid_from_graph(graph), np.zeros((ny, nx)), holdings)
    cpu = StepEngine(net, WetLawRunner(physics), dt=1.0)
    gpu = CudaLegacyContext(net, physics, graph, limits=N.walk_limits(net.dx_m), dt=1.0, n_steps=n_steps, **ctx_kwargs)
    return cp, net, cpu, gpu, (ny, nx)


def states(net, shape, rng, n_steps):
    ny, nx = shape
    active = net.active.reshape(ny, nx)
    zero_slope = net.slope_zero.reshape(ny, nx)
    for _ in range(n_steps):
        depth = np.where(active & (rng.random(shape) > 0.15), rng.uniform(5e-5, 8e-3, shape), 0.0)
        vel = np.where(active & ~zero_slope & (depth > 0.0), rng.uniform(0.001, 0.4, shape), 0.0)
        rain = np.where(active & (rng.random(shape) > 0.2), rng.uniform(1e-6, 3e-5, shape), 0.0)
        yield depth, vel, rain


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_gpu_steps_match_the_cpu_engine_within_the_declared_sediment_bounds_and_exact_tallies(name):
    cp, net, cpu, gpu, shape = build(name)
    rng = np.random.default_rng(42)
    seq = list(states(net, shape, rng, 8))
    for row, (depth, vel, rain) in enumerate(seq):
        res = cpu.step(depth, vel, rain)
        gpu.step(row, cp.asarray(depth), cp.asarray(vel), cp.asarray(rain))
        gpu.check_flags(row + 1)
        np.testing.assert_allclose(gpu.host_ledger[row], res.ledger_row, rtol=RTOL, atol=ATOL, err_msg=f"{name} step {row}")
        assert np.array_equal(gpu.host_counts[row, :N_WALK], res.walk_counts[:N_WALK]), (name, row)
        assert np.array_equal(gpu.host_counts[row, 10:], res.regime_counts), (name, row)
    maps = gpu.download_maps()
    for key, ref in (("cum_det", cpu.cum_det), ("cum_dep", cpu.cum_dep), ("cum_clip", cpu.cum_clip), ("mobile", cpu.mobile)):
        np.testing.assert_allclose(maps[key], ref, rtol=RTOL, atol=ATOL, err_msg=f"{name} {key}")
    np.testing.assert_allclose(maps["v_prev"], cpu.v_prev, rtol=RTOL, atol=ATOL)
    assert gpu.stats["d2h_flag_reads"] == 8 and gpu.poisoned is None


N_WALK = 9  # walk tallies C_WALKS .. C_ZERO_SLOPE (the erase-cell tally is a separate, regime-based count compared below)


def test_erase_cell_tally_and_regime_counts_cover_the_cpu_definition():
    cp, net, cpu, gpu, shape = build("converging", n_steps=3)
    rng = np.random.default_rng(7)
    for row, (depth, vel, rain) in enumerate(states(net, shape, rng, 3)):
        cpu.step(depth, vel, rain)
        gpu.step(row, cp.asarray(depth), cp.asarray(vel), cp.asarray(rain))
    gpu.check_flags()
    assert gpu.host_counts[:, 9].sum() == cpu.total_counts[9]


def test_the_gpu_replay_is_bitwise_repeatable_after_reset():
    cp, net, _, gpu, shape = build("converging", n_steps=6)
    rng = np.random.default_rng(3)
    seq = list(states(net, shape, rng, 6))

    def run():
        for row, (depth, vel, rain) in enumerate(seq):
            gpu.step(row, cp.asarray(depth), cp.asarray(vel), cp.asarray(rain))
        gpu.check_flags()
        return gpu.host_ledger.copy(), gpu.download_maps()

    first = run()
    gpu.reset()
    second = run()
    assert np.array_equal(first[0], second[0])
    for key in first[1]:
        assert np.array_equal(first[1][key], second[1][key]), key


def test_a_device_failure_poisons_the_context_until_reset():
    from maple_syrup.legacy_native_cuda import CudaLegacyError

    cp, _, _, gpu, shape = build("converging", n_steps=4)
    good = [cp.asarray(np.full(shape, v)) for v in (1e-3, 0.05, 1e-5)]
    gpu.step(0, *good)
    bad_depth = np.full(shape, 1e-3)
    bad_depth[0, 0] = np.nan
    gpu.step(1, cp.asarray(bad_depth), good[1], good[2])
    with pytest.raises(CudaLegacyError, match="poisoned|validation"):
        gpu.check_flags()
    assert gpu.poisoned
    with pytest.raises(CudaLegacyError, match="poisoned"):
        gpu.step(2, *good)
    gpu.reset()
    assert gpu.poisoned is None
    gpu.step(0, *good)
    gpu.check_flags()


def test_inputs_are_strict_and_the_memory_budget_refuses_before_allocation():
    from maple_syrup.legacy_native_cuda import CudaLegacyError

    cp, _, _, gpu, shape = build("channel_to_ring", n_steps=2)
    good = [cp.asarray(np.full(shape, v)) for v in (1e-3, 0.05, 1e-5)]
    with pytest.raises(CudaLegacyError):
        gpu.step(0, good[0].astype(np.float32), good[1], good[2])
    with pytest.raises(CudaLegacyError):
        gpu.step(0, np.full(shape, 1e-3), good[1], good[2])  # a host array: no implicit transfer
    with pytest.raises(CudaLegacyError, match="consecutive"):
        gpu.step(1, *good)
    with pytest.raises(CudaLegacyError, match="exceeds the budget"):
        build("channel_to_ring", memory_budget_bytes=1024)


def cpu_walk(net, det, inv_l, law, regime):
    from maple_syrup import legacy_native as N
    from maple_syrup.legacy_native_numba import get_kernels

    n, nc = det.shape
    depos, ring, inactive, erased = np.zeros((n, nc)), np.zeros(nc), np.zeros(nc), np.zeros(nc)
    counts = np.zeros(N.N_COUNTS, dtype=np.int64)
    get_kernels(False).walk(net.source_order_index, det, inv_l, law, regime, N.walk_limits(net.dx_m), net.slope_zero, net.walk_first,
                            net.walk_next, net.aspect0, net.inactive, np.zeros(n, dtype=bool), False, net.dx_m, 1.0, depos, ring,
                            inactive, erased, counts)
    return depos, ring, inactive, counts


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_forced_walk_bypassing_the_wet_laws_matches_the_cpu_rules_on_every_stop_kind(name):
    """Forced detachment on EVERY active cell, including aspect-0 sources (west start), pits, inactive-neighbour and ring stops, with
    zero-slope diffuse no-law and local deposits: the walk rules proven on the device without the wet laws."""
    from maple_syrup import legacy_native as N

    _, net, _, gpu, _ = build(name, n_steps=1)
    n, nc = net.active.size, 6
    rng = np.random.default_rng(5)
    for trial in range(3):
        gpu.reset()
        det = np.where(net.active[:, None], rng.uniform(0.1, 1.0, (n, nc)), 0.0)
        inv = rng.uniform(0.2, 5.0, (n, nc))
        law = rng.random((n, nc)) > 0.3
        regime = rng.integers(2, 7, (n, nc)).astype(np.int8)
        regime[:, 0] = 2  # diffuse: zero-slope cells without a law take the "no walk, no deposit" branch
        law[:, 0] = False
        got = gpu.walk_only(det, inv, law, regime)
        depos, ring, inactive, counts = cpu_walk(net, det, inv, law, regime)
        assert got["flag"] == 0
        np.testing.assert_allclose(got["depos"][net.active], depos[net.active], rtol=2e-11, atol=1e-14, err_msg=f"{name} {trial}")
        np.testing.assert_allclose(got["ring_kg_per_step"], ring, rtol=2e-11, atol=1e-14)
        np.testing.assert_allclose(got["inactive_kg_per_step"], inactive, rtol=2e-11, atol=1e-14)
        assert np.array_equal(got["counts"][:9], counts[:9]), (name, trial)  # walks, ring, terminal, inactive, limit, vge, aspect0, local, zero slope
    assert N.C_ASPECT0 == 6


def test_aspect0_source_starts_west_and_credits_before_the_terminal_stop_on_the_device():
    from maple_syrup import legacy_native as N

    _, net, _, gpu, _ = build("terminal_pit", n_steps=1)
    n, nc = net.active.size, 6
    det = np.zeros((n, nc))
    det[2] = 0.5  # only the pit (aspect 0) is a source
    inv = np.full((n, nc), 1.0)
    law = np.ones((n, nc), dtype=bool)
    regime = np.full((n, nc), 5, dtype=np.int8)
    got = gpu.walk_only(det, inv, law, regime)
    d = 0.5
    assert got["counts"][N.C_ASPECT0] == nc and got["counts"][N.C_TERMINAL] > 0
    assert got["depos"][1].min() > 0.0  # the west neighbour was credited
    assert got["depos"][3].sum() == 0.0  # nothing went east
    assert abs(got["depos"][2, 0] - (d * (1 - np.exp(-1.0)) + d * (np.exp(-2.0) - np.exp(-3.0)))) < 1e-13


@pytest.mark.parametrize("mode", ["level", "block"])
def test_explicit_cn_modes_match_the_cpu_engine_and_are_repeatable_after_reset(mode):
    cp, net, cpu, gpu, shape = build("converging", n_steps=6, cn_mode=mode)
    assert gpu.cn_mode == mode
    seq = list(states(net, shape, np.random.default_rng(21), 6))
    results = []
    for repeat in range(2):
        gpu.reset()
        cpu.reset()
        for row, (depth, vel, rain) in enumerate(seq):
            res = cpu.step(depth, vel, rain)
            gpu.step(row, cp.asarray(depth), cp.asarray(vel), cp.asarray(rain))
            gpu.check_flags(row + 1)
            np.testing.assert_allclose(gpu.host_ledger[row], res.ledger_row, rtol=RTOL, atol=ATOL, err_msg=f"{mode} {row}")
        results.append((gpu.host_ledger.copy(), gpu.download_maps()))
    assert np.array_equal(results[0][0], results[1][0])
    assert all(np.array_equal(results[0][1][k], results[1][1][k]) for k in results[0][1])


def test_more_than_nine_classes_use_every_resource_and_match_the_cpu():
    """Ten synthetic classes: the final reduction needs 7 x 10 = 70 threads (the earlier 64-thread launch would have missed six)."""
    import cupy as cp

    from maple_syrup import legacy_native as N
    from maple_syrup.legacy_native_cuda import CudaLegacyContext
    from maple_syrup.legacy_native_numba import StepEngine, WetLawRunner
    from maple_syrup.legacy_physics_numba import prepare_legacy_physics
    from maple_syrup.sediment_physics import (
        KE_MODELS,
        physics_grid_from_graph,
        sediment_physics_parameters,
    )

    nc = 10
    graph = GRAPHS["converging"]()
    ny, nx = graph.shape
    net = N.native_network(graph)
    params = sediment_physics_parameters(
        diameter_m=np.geomspace(6e-5, 2.4e-2, nc), raindrop_a=np.linspace(4e-5, 8e-4, nc), raindrop_b=np.linspace(0.5, 1.2, nc),
        raindrop_c=np.linspace(0.1, 1.0, nc), raindrop_max_depth_mm=np.full(nc, 1000.0), particle_density_g_cm3=2.65,
        active_layer_sensitivity_mm=1.52e-6, ke_model=KE_MODELS[1], raindrop_depth_attenuation_per_cm=np.linspace(2.7, 0.3, nc))
    holdings = np.broadcast_to(np.full(nc, 0.25), (ny, nx, nc)).copy()
    physics = prepare_legacy_physics(params, physics_grid_from_graph(graph), np.zeros((ny, nx)), holdings)
    cpu = StepEngine(net, WetLawRunner(physics), dt=1.0)
    gpu = CudaLegacyContext(net, physics, graph, limits=N.walk_limits(net.dx_m), dt=1.0, n_steps=4)
    for row, (depth, vel, rain) in enumerate(states(net, (ny, nx), np.random.default_rng(8), 4)):
        res = cpu.step(depth, vel, rain)
        gpu.step(row, cp.asarray(depth), cp.asarray(vel), cp.asarray(rain))
        gpu.check_flags(row + 1)
        np.testing.assert_allclose(gpu.host_ledger[row], res.ledger_row, rtol=RTOL, atol=ATOL)
        assert gpu.host_ledger[row].shape == (13, nc)
    assert gpu.host_ledger[:4, 7].sum() >= 0.0  # every class row exists (new mobile column), including those beyond the old 9


def test_sealed_context_refuses_replaced_arrays_changed_scalars_and_future_rows_before_any_launch():
    from maple_syrup.legacy_native_cuda import CudaLegacyError

    cp, _, _, gpu, shape = build("converging", n_steps=4)
    good = [cp.asarray(np.full(shape, v)) for v in (1e-3, 0.05, 1e-5)]
    gpu.step(0, *good)
    launches = gpu.stats["launches"]
    for attr, value in (("n_steps", 99), ("nc", 5), ("n", 3), ("dt", 2.0), ("dx", 3.0), ("m", 1), ("cn_mode", "level")):
        old = getattr(gpu, attr)
        setattr(gpu, attr, value)
        with pytest.raises(CudaLegacyError, match="sealed|changed"):
            gpu.step(1, *good)
        setattr(gpu, attr, old)
    original = gpu.d_tgt_rec
    gpu.d_tgt_rec = cp.zeros_like(original)  # a replaced static array
    with pytest.raises(CudaLegacyError, match="d_tgt_rec"):
        gpu.step(1, *good)
    gpu.d_tgt_rec = original
    original = gpu.det
    gpu.det = cp.zeros(original.size + 1)
    with pytest.raises(CudaLegacyError, match="det"):
        gpu.check_flags()
    with pytest.raises(CudaLegacyError, match="det"):
        gpu.reset()
    gpu.det = original
    gpu.M2 = gpu.M1  # the ping-pong pair is no longer the sealed pair
    with pytest.raises(CudaLegacyError, match="ping-pong"):
        gpu.step(1, *good)
    assert gpu.stats["launches"] == launches  # nothing was launched by any refused call


def test_a_context_is_refused_on_another_device_before_any_launch():
    cp, _, _, gpu, shape = build("converging", n_steps=2)
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("a second CUDA device is not visible")
    from maple_syrup.legacy_native_cuda import CudaLegacyError

    with cp.cuda.Device(1):
        other = [cp.asarray(np.full(shape, v)) for v in (1e-3, 0.05, 1e-5)]
        with pytest.raises(CudaLegacyError, match="device"):
            gpu.step(0, *other)
        with pytest.raises(CudaLegacyError, match="device"):
            gpu.check_flags()
    assert gpu.stats["launches"] == 0


@pytest.mark.parametrize("upto", [-1, 99, 1.0, True, "1", np.float64(1.0), 2])
def test_check_flags_upto_is_strictly_an_executed_integer_row_count(upto):
    from maple_syrup.legacy_native_cuda import CudaLegacyError

    cp, _, _, gpu, shape = build("converging", n_steps=4)
    gpu.step(0, *[cp.asarray(np.full(shape, v)) for v in (1e-3, 0.05, 1e-5)])
    reads = gpu.stats["d2h_flag_reads"]
    with pytest.raises(CudaLegacyError):
        gpu.check_flags(upto)  # 2 is a future row (only 1 was executed); the others are not integers or are out of range
    assert gpu.stats["d2h_flag_reads"] == reads and gpu._checked == 0  # nothing was transferred
    for bad_row in (1.0, True, -1, "0", None):
        with pytest.raises(CudaLegacyError):
            gpu.step(bad_row, *[cp.asarray(np.full(shape, v)) for v in (1e-3, 0.05, 1e-5)])
    assert gpu.check_flags() == 1


def test_post_infiltration_refuses_any_overlapping_span_and_malformed_outputs_before_launch():
    from maple_syrup.legacy_native_cuda import CudaLegacyError

    cp, _net, _, gpu, shape = build("channel_to_ring", n_steps=1)  # shape (1, 5)
    big = cp.zeros(16)
    old = big[0:5].reshape(shape)
    rain = cp.zeros(shape)
    intake = cp.zeros(shape)
    launches = gpu.stats["launches"]
    shifted = big[2:7].reshape(shape)  # a C-contiguous shifted view overlapping `old` with a different start pointer
    with pytest.raises(CudaLegacyError, match="overlaps"):
        gpu.post_infiltration_depth(old, rain, intake, shifted)
    with pytest.raises(CudaLegacyError, match="overlaps"):
        gpu.post_infiltration_depth(old, rain, intake, old)
    for bad_out in (cp.zeros(shape, dtype=np.float32), cp.zeros((2, 5)), np.zeros(shape), cp.zeros((2, 10))[::2, ::2][:, :5]):
        with pytest.raises(CudaLegacyError):
            gpu.post_infiltration_depth(old, rain, intake, bad_out)
    assert gpu.stats["launches"] == launches
    out = big[8:13].reshape(shape)  # adjacent but disjoint spans are fine
    gpu.post_infiltration_depth(old, rain, intake, out)
    assert gpu.stats["launches"] == launches + 1


def test_reset_separates_warmup_counters_from_the_measured_run():
    cp, _, _, gpu, shape = build("converging", n_steps=4)
    good = [cp.asarray(np.full(shape, v)) for v in (1e-3, 0.05, 1e-5)]
    gpu.step(0, *good)
    gpu.check_flags()
    warm = dict(gpu.stats)
    gpu.reset()
    assert gpu.stats_before_reset == warm and gpu.stats["launches"] == 0 and gpu.stats["steps"] == 0
    assert gpu.stats["h2d_static_bytes"] == warm["h2d_static_bytes"]  # context-lifetime upload is kept


def test_post_infiltration_kernel_equals_the_cpu_helper_bitwise():
    from maple_syrup import legacy_driver as D

    cp, net, _, gpu, shape = build("converging", n_steps=1)
    rng = np.random.default_rng(11)
    old = rng.uniform(0.0, 0.02, shape)
    rain = rng.uniform(0.0, 0.003, shape)
    intake = rng.uniform(0.0, 0.025, shape)
    out = cp.zeros(shape)
    gpu.post_infiltration_depth(cp.asarray(old), cp.asarray(rain), cp.asarray(intake), out)
    expected = D.post_infiltration_depth(old, rain, intake, net.active.reshape(shape))
    assert np.array_equal(cp.asnumpy(out), expected)
