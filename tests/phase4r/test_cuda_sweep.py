"""Phase 4R: the CUDA ordered sweep and `route_step(implementation="cuda")` on a real device.

Sweep outputs are compared BIT FOR BIT (uint64 views) against the ORIGINAL serial and the level-batched compiled CPU
sweeps on identical inputs. Where both sides are NaN the positions and sign bits must agree (payload bits are not part
of the contract). A mismatch is reported, never absorbed by a tolerance. Whole-step physical fields use the declared
rtol 2e-12 / atol 1e-14. Skipped without a device (explicit skip, no GPU claim). Nothing here was run by its author;
Codex records results. Select the device with CUDA_VISIBLE_DEVICES before starting pytest.
"""
from __future__ import annotations

import functools
import gc
import hashlib

import numpy as np
import pytest
from test_routing import chain_full, make_graph, random_full, valley_full

pytest.importorskip("maple")

from maple.core import backend as mb

from maple_syrup import routing, routing_cuda
from maple_syrup import routing_numba as rn
from maple_syrup.routing import RoutingError

pytestmark = [pytest.mark.usefixtures("gpu"),
              pytest.mark.skipif(not rn.numba_available(), reason="Numba oracle not installed")]

NAMES = ("chain", "valley_branching", "valley_wide", "random_w127", "random_w128", "random_w129", "random_w257")
SPECIAL = [0.0, -0.0, -1e-3, np.nan, -np.nan, np.inf, -np.inf, 5e-324, 2.2250738585072014e-308, 1e-300, 1e-12, 3e-3,
           1e200, 1e308, -5e-324]
ITERATIONS = [1, 2, 5, 40, 200]
C_DEFAULT = 0.5 / 3.0


@functools.cache
def graph_pair(name):
    """(host graph, CuPy graph) built from identical inputs."""
    import cupy as cp

    rng = np.random.default_rng(11)
    if name == "chain":
        z, ff = chain_full(9), 5.0
    elif name == "valley_branching":
        z, ff = valley_full(6, 5), 5.0
    elif name == "valley_wide":
        z, ff = valley_full(3, 41), 5.0
    elif name.startswith("plane_"):
        # Uniform southward plane: equal elevations across columns, 3 rows, every row is ONE level of exactly nx cells.
        nx = int(name.split("_")[1])
        z = np.repeat((np.arange(3 + 2, dtype=np.float64) * 0.015625)[:, None], nx + 2, axis=1)
        ff = 5.0
    else:
        width = int(name.split("_w")[1])
        z = random_full(rng, 6, width)
        ff = rng.uniform(5.0, 30.0, (6, width))
    return make_graph(z, ff=ff), make_graph(z, ff=ff, xp=cp)


def bases(n, seed=3):
    rng = np.random.default_rng(seed)
    wet = rng.uniform(1e-4, 3e-3, n)
    return {
        "wet": wet,
        "dry": np.zeros(n),
        "negzero": np.full(n, -0.0),
        "mixed": wet * (rng.random(n) < 0.45),
        "mostly_dry": wet * (rng.random(n) < 0.1),
        "negative": -wet,
        "subnormal": rng.uniform(1.0, 2.0, n) * 5e-320,
        "special": np.resize(np.array(SPECIAL), n),
    }


def cpu_sweep(fn, host, base, c, it):
    n = host.n_active
    outs = tuple(np.full(n, 7.0) for _ in range(4))
    fn(np.asarray(host.level_bounds, dtype=np.int64), np.array(host.conveyance_lo), host.donor_position,
       host.donor_mask, np.array(base, dtype=np.float64), float(c), int(it), *outs)
    return outs


def assert_bits(ref, got, label):
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(mb.to_host(got), dtype=np.float64)
    assert ref.shape == got.shape
    rb, gb = ref.view(np.uint64), got.view(np.uint64)
    both_nan = np.isnan(ref) & np.isnan(got)
    bad = (rb != gb) & ~both_nan
    bad |= both_nan & ((rb >> np.uint64(63)) != (gb >> np.uint64(63)))
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise AssertionError(f"{label}: {int(bad.sum())} bit mismatches, first at {i}: cpu {ref[i]!r} "
                             f"({int(rb[i]):#018x}) vs cuda {got[i]!r} ({int(gb[i]):#018x})")


def check_sweep(name, base_name, c, it):
    cp = pytest.importorskip("cupy")
    host, dev = graph_pair(name)
    base = bases(host.n_active)[base_name]
    serial = cpu_sweep(rn.compiled_sweep(), host, base, c, it)
    batched = cpu_sweep(rn.compiled_sweep_batched(), host, base, c, it)
    base_d = cp.asarray(base)
    got = routing_cuda.run_sweep(dev, base_d, c, it)
    for label, s, b, g in zip(("qin", "q", "flow", "rhs"), serial, batched, got, strict=True):
        assert_bits(s, g, f"{name}/{base_name}/it={it}/c={c} {label} vs original serial")
        assert_bits(b, g, f"{name}/{base_name}/it={it}/c={c} {label} vs level-batched")
    assert np.array_equal(cp.asnumpy(base_d).view(np.uint64), base.view(np.uint64)), "base_lo was modified"


@pytest.mark.parametrize("it", ITERATIONS)
@pytest.mark.parametrize("base_name", list(bases(4)))
@pytest.mark.parametrize("name", NAMES)
def test_sweep_bitwise_matches_cpu(name, base_name, it):
    check_sweep(name, base_name, C_DEFAULT, it)


PLANE_WIDTHS = [1, 127, 128, 129, 255, 256, 257]


@pytest.mark.parametrize("width", PLANE_WIDTHS)
def test_plane_graphs_have_exact_level_widths(width):
    host, dev = graph_pair(f"plane_{width}")
    widths = [b - a for a, b in zip(dev.level_bounds[:-1], dev.level_bounds[1:], strict=True)]
    assert widths == [width] * 3 and dev.max_level_width == width and host.max_level_width == width


@pytest.mark.parametrize("it", [1, 40, 200])
@pytest.mark.parametrize("base_name", ["wet", "mixed", "negative", "subnormal", "special"])
@pytest.mark.parametrize("width", PLANE_WIDTHS)
def test_sweep_bitwise_at_exact_block_boundary_widths(width, base_name, it):
    """Launch grids of 1, 1, 1, 2, 2, 2, 3 blocks of 128 threads with a partial last block (127, 129, 255, 257)."""
    check_sweep(f"plane_{width}", base_name, C_DEFAULT, it)


@pytest.mark.parametrize("c", [1e-300, 1e-9, 1.0, 1e3])
@pytest.mark.parametrize("base_name", ["wet", "mixed", "subnormal", "special"])
@pytest.mark.parametrize("name", ["valley_branching", "random_w129"])
def test_sweep_bitwise_extreme_c(name, base_name, c):
    check_sweep(name, base_name, c, 40)


# --- whole route_step ---------------------------------------------------------------------------------------------
FIELDS = ("depth_m", "flow_depth_m", "discharge_m2_s", "velocity_m_s", "inflow_m2_s", "old_discharge_m2_s",
          "old_inflow_m2_s", "face_volume_m3")
SCALARS = ("export_m3", "outlet_discharge_m3_s", "storage_change_m3", "budget_residual_m3", "max_courant_old",
           "max_courant_new", "max_constitutive_residual_m", "max_cell_balance_residual_m")


def host_arr(x):
    return np.asarray(mb.to_host(x))


def assert_steps_close(a, b, label):
    for f in FIELDS:
        np.testing.assert_allclose(host_arr(getattr(a, f)), host_arr(getattr(b, f)), rtol=2e-12, atol=1e-14,
                                   err_msg=f"{label}: {f}")
    for f in SCALARS:
        np.testing.assert_allclose(float(host_arr(getattr(a, f))), float(host_arr(getattr(b, f))), rtol=2e-12,
                                   atol=1e-14, err_msg=f"{label}: {f}")
    assert a.conservative == b.conservative and a.bisection_iterations == b.bisection_iterations
    # `implementation` metadata legitimately differs and is excluded from the physical comparison.


def states(n_shape, seed=5, wet_fraction=0.8):
    rng = np.random.default_rng(seed)
    h_old = rng.uniform(1e-4, 2e-3, n_shape) * (rng.random(n_shape) < wet_fraction)
    h_start = h_old + rng.uniform(0.0, 1e-3, n_shape)
    return h_start, h_old


@pytest.mark.parametrize("name", ["chain", "valley_branching", "random_w129"])
@pytest.mark.parametrize("frac", [1.0, 0.5, 0.1, 0.0])
def test_route_step_matches_array_and_numba(name, frac):
    cp = pytest.importorskip("cupy")
    host, dev = graph_pair(name)
    h_start, h_old = states(host.shape, wet_fraction=frac)
    hs, ho = cp.asarray(h_start), cp.asarray(h_old)
    cu = routing.route_step(dev, hs, ho, 1.0, implementation="cuda")
    ar = routing.route_step(dev, hs, ho, 1.0, implementation="array")
    nb = routing.route_step(host, h_start, h_old, 1.0, implementation="numba")
    assert cu.implementation == "cuda" and ar.implementation == "array" and nb.implementation == "numba"
    assert_steps_close(cu, ar, "cuda vs gpu array")
    assert_steps_close(cu, nb, "cuda vs cpu numba")
    assert isinstance(cu.depth_m, cp.ndarray)


def outcome(fn):
    try:
        fn()
    except RoutingError as exc:
        return type(exc), str(exc)
    return None


def test_refusals_match_array_path_class_and_message_and_inputs_unchanged():
    cp = pytest.importorskip("cupy")
    host, dev = graph_pair("valley_branching")
    h_start, h_old = states(host.shape, wet_fraction=1.0)

    def variant(label):
        hs, ho = h_start.copy(), h_old.copy()
        kw = {}
        if label == "nan":
            hs[1, 1] = np.nan
        elif label == "negative":
            hs[1, 1] = -1e-3
        elif label == "old_exceeds":
            ho[1, 1] = hs[1, 1] + 1e-3
        elif label == "courant":
            hs[:] = 100.0
            ho[:] = 100.0
        elif label == "overflow":
            hs[:] = 1e308
            ho[:] = 0.0
        elif label == "nonconvergence":
            kw["bisection_iterations"] = 5
        return hs, ho, kw

    for label in ("ok", "nan", "negative", "old_exceeds", "courant", "overflow", "nonconvergence"):
        hs, ho, kw = variant(label)
        hs_d, ho_d = cp.asarray(hs), cp.asarray(ho)
        before = (hashlib.sha256(hs_d.get().tobytes()).hexdigest(), hashlib.sha256(ho_d.get().tobytes()).hexdigest())
        got = {}
        for impl in ("array", "cuda"):
            got[impl] = outcome(lambda impl=impl, hs_d=hs_d, ho_d=ho_d, kw=kw: routing.route_step(
                dev, hs_d, ho_d, 1.0, implementation=impl, **kw))
        assert got["array"] == got["cuda"], f"{label}: {got}"
        if label != "ok":
            assert got["cuda"] is not None, f"{label}: expected a refusal"
        after = (hashlib.sha256(hs_d.get().tobytes()).hexdigest(), hashlib.sha256(ho_d.get().tobytes()).hexdigest())
        assert before == after, f"{label}: inputs modified"
    assert outcome(lambda: routing.route_step(dev, cp.asarray(h_start), cp.asarray(h_old), 1.0,
                                              implementation="cuda", bisection_iterations=5)) is not None
    rejected = outcome(lambda: routing.route_step(dev, cp.full(host.shape, 100.0), cp.full(host.shape, 100.0), 1.0,
                                                  implementation="cuda"))
    assert rejected[0] is routing.RoutingStepRejected


@pytest.mark.parametrize("it", [0, 201, True, 40.0, None])
def test_route_step_iteration_refusals(it):
    cp = pytest.importorskip("cupy")
    host, dev = graph_pair("chain")
    h_start, h_old = states(host.shape)
    with pytest.raises(RoutingError):
        routing.route_step(dev, cp.asarray(h_start), cp.asarray(h_old), 1.0, implementation="cuda",
                           bisection_iterations=it)


def test_multistep_source_and_recession_tracks_array_and_cpu():
    cp = pytest.importorskip("cupy")
    host, dev = graph_pair("valley_branching")
    shape = host.shape
    n_steps, n_rain = 240, 120
    rain = np.full(shape, 2e-6)
    rain[::2, 0] = 0.0
    h_np = np.zeros(shape)
    h_ar, h_cu = cp.zeros(shape), cp.zeros(shape)
    worst = 0.0
    for step in range(n_steps):
        r = rain if step < n_rain else 0.0 * rain  # prescribed source, then recession
        a = routing.route_step(dev, h_ar + cp.asarray(r), h_ar, 1.0, implementation="array")
        b = routing.route_step(dev, h_cu + cp.asarray(r), h_cu, 1.0, implementation="cuda")
        c = routing.route_step(host, h_np + r, h_np, 1.0, implementation="numba")
        assert_steps_close(b, a, f"step {step} cuda vs array")
        assert_steps_close(b, c, f"step {step} cuda vs numba")
        h_ar, h_cu, h_np = a.depth_m, b.depth_m, c.depth_m
        denom = np.maximum(np.abs(host_arr(h_np)), 1e-14)
        worst = max(worst, float(np.max(np.abs(host_arr(h_cu) - host_arr(h_np)) / denom)))
    assert worst < 1e-8, worst  # recorded for Codex; the per-step comparison above carries the declared bound
    assert float(host_arr(h_cu).sum()) < float(host_arr(h_cu).size) * 1e-3


# --- context, ownership, streams, devices, launches ------------------------------------------------------------
class CountingKernel:
    """Proxy around the real RawKernel (immutable extension type: wrap, do not patch)."""

    def __init__(self, real):
        self.real, self.launches, self.grids = real, 0, []

    def __call__(self, grid, block, args):
        self.launches += 1
        self.grids.append((grid, block))
        return self.real(grid, block, args)

    def __getattr__(self, name):
        return getattr(self.real, name)


@pytest.fixture
def counting(monkeypatch):
    real = routing_cuda._get_kernel()
    proxy = CountingKernel(real)
    monkeypatch.setattr(routing_cuda, "_get_kernel", lambda: proxy)
    return proxy


def fresh_dev_graph(name="valley_branching"):
    import cupy as cp

    rng = np.random.default_rng(11)
    del rng
    host, _ = graph_pair(name)
    z = {"valley_branching": valley_full(6, 5), "chain": chain_full(9)}[name]
    return make_graph(z, ff=5.0, xp=cp), host


def test_one_launch_per_level_block128_and_no_transfer_on_cache_hit(counting):
    cp = pytest.importorskip("cupy")
    dev, host = fresh_dev_graph()
    base = cp.asarray(bases(host.n_active)["wet"])
    routing_cuda.prepare_cuda_routing(dev)
    counting.launches = 0
    before = mb.read_transfer_counters()
    routing_cuda.run_sweep(dev, base, C_DEFAULT, 40, mode="level")  # the per-level comparator (default is now "auto")
    delta = mb.read_transfer_counters().delta(before)
    assert counting.launches == dev.n_levels
    assert all(block == (128,) for _grid, block in counting.grids)
    assert delta.host_to_device == delta.device_to_host == delta.scalar_reads == delta.synchronizations == 0
    assert counting.launches < 40  # the previous array sweep issued ~ n_levels * 40 * 9 array operations


def test_route_step_has_only_the_existing_single_flag_read(counting):
    cp = pytest.importorskip("cupy")
    dev, host = fresh_dev_graph()
    h_start, h_old = states(host.shape)
    hs, ho = cp.asarray(h_start), cp.asarray(h_old)
    routing.route_step(dev, hs, ho, 1.0, implementation="cuda")  # prepares
    before = mb.read_transfer_counters()
    routing.route_step(dev, hs, ho, 1.0, implementation="cuda")
    d = mb.read_transfer_counters().delta(before)
    before = mb.read_transfer_counters()
    routing.route_step(dev, hs, ho, 1.0, implementation="array")
    d_ref = mb.read_transfer_counters().delta(before)
    assert d.host_to_device_bytes == d_ref.host_to_device_bytes
    assert d.device_to_host == d_ref.device_to_host and d.scalar_reads == d_ref.scalar_reads


def test_context_cache_ownership_and_release():
    cp = pytest.importorskip("cupy")
    dev, host = fresh_dev_graph()
    ctx = routing_cuda.prepare_cuda_routing(dev)
    before = mb.read_transfer_counters()
    assert routing_cuda.prepare_cuda_routing(dev) is ctx
    d = mb.read_transfer_counters().delta(before)
    assert d.host_to_device == d.device_to_host == d.synchronizations == 0
    assert ctx.device_id == int(cp.cuda.Device().id)
    assert ctx.donor_position.data.ptr != dev.donor_position.data.ptr  # owned copies, not aliases
    assert ctx.n_active == host.n_active and ctx.level_bounds == tuple(host.level_bounds)
    assert ctx.static_bytes == ctx.conveyance_lo.nbytes + ctx.donor_position.nbytes + ctx.donor_mask.nbytes
    assert not any(v is dev for v in vars(ctx).values())
    s = ctx.summary()
    assert s["device_to_host_bytes"] > 0 and s["host_to_device_bytes"] >= ctx.static_bytes
    assert routing_cuda.release_cuda_routing(dev) is True
    assert routing_cuda.release_cuda_routing(dev) is False
    ctx2 = routing_cuda.prepare_cuda_routing(dev)
    assert ctx2 is not ctx
    base = cp.asarray(bases(host.n_active)["wet"])
    a = routing_cuda.run_sweep(dev, base, C_DEFAULT, 40)
    routing_cuda.release_cuda_routing(dev)
    b = routing_cuda.run_sweep(dev, base, C_DEFAULT, 40)
    for x, y in zip(a, b, strict=True):
        assert_bits(host_arr(x), y, "re-prepared context")


def test_cache_does_not_keep_graph_alive():
    import weakref

    dev, _host = fresh_dev_graph()
    ctx = routing_cuda.prepare_cuda_routing(dev)
    ref = weakref.ref(dev)
    ctx_ref = weakref.ref(ctx)
    assert not any(v is dev for v in vars(ctx).values())
    del dev
    gc.collect()
    assert ref() is None, "the cache (or its context) kept the graph alive"
    del ctx
    gc.collect()
    assert ctx_ref() is None, "the context outlived its graph's cache entry"


def test_outputs_fresh_each_call_and_static_context_unchanged():
    cp = pytest.importorskip("cupy")
    dev, host = fresh_dev_graph()
    ctx = routing_cuda.prepare_cuda_routing(dev)
    static_hash = [hashlib.sha256(a.get().tobytes()).hexdigest()
                   for a in (ctx.conveyance_lo, ctx.donor_position, ctx.donor_mask)]
    base = cp.asarray(bases(host.n_active)["mixed"])
    a = routing_cuda.run_sweep(dev, base, C_DEFAULT, 40)
    b = routing_cuda.run_sweep(dev, base, C_DEFAULT, 40)
    assert len({int(x.data.ptr) for x in (*a, *b)}) == 8
    keep = [host_arr(x).copy() for x in b]
    for x in a:
        x.fill(123.0)  # scribbling on one result must not affect another call or the static data
    c = routing_cuda.run_sweep(dev, base, C_DEFAULT, 40)
    for k_, y in zip(keep, c, strict=True):
        assert_bits(k_, y, "fresh result after mutating an earlier result")
    assert static_hash == [hashlib.sha256(a_.get().tobytes()).hexdigest()
                           for a_ in (ctx.conveyance_lo, ctx.donor_position, ctx.donor_mask)]


def test_nondefault_stream_matches_default_stream():
    cp = pytest.importorskip("cupy")
    host, dev = graph_pair("valley_branching")
    base_h = bases(host.n_active)["mixed"]
    ref = cpu_sweep(rn.compiled_sweep(), host, base_h, C_DEFAULT, 40)
    dev2, _ = fresh_dev_graph()
    with cp.cuda.Stream(non_blocking=True) as stream:
        base = cp.asarray(base_h)
        got = routing_cuda.run_sweep(dev2, base, C_DEFAULT, 40)  # prepares on this stream too
        h_start, h_old = states(host.shape)
        step = routing.route_step(dev2, cp.asarray(h_start), cp.asarray(h_old), 1.0, implementation="cuda")
        stream.synchronize()
    for r, g in zip(ref, got, strict=True):
        assert_bits(r, g, "non-default stream")
    ref_step = routing.route_step(dev, cp.asarray(h_start), cp.asarray(h_old), 1.0, implementation="array")
    assert_steps_close(step, ref_step, "non-default stream route_step")


def test_dynamic_array_structure_refusals(counting):
    cp = pytest.importorskip("cupy")
    dev, host = fresh_dev_graph()
    n = host.n_active
    ok = cp.asarray(bases(n)["wet"])
    routing_cuda.prepare_cuda_routing(dev)
    counting.launches = 0
    bad = {
        "numpy": np.zeros(n), "float32": ok.astype(cp.float32), "short": ok[:-1], "2d": ok.reshape(1, -1),
        "strided": cp.zeros(2 * n)[::2], "list": [0.0] * n,
    }
    for label, arr in bad.items():
        with pytest.raises(RoutingError):
            routing_cuda.run_sweep(dev, arr, C_DEFAULT, 40)
        assert counting.launches == 0, label
    for c, it in ((0.0, 40), (float("nan"), 40), (C_DEFAULT, 0), (C_DEFAULT, 201), (C_DEFAULT, True)):
        with pytest.raises(RoutingError):
            routing_cuda.run_sweep(dev, ok, c, it)
    assert counting.launches == 0


def forged(dev, **changes):
    import dataclasses

    return dataclasses.replace(dev, **changes)


def test_corrupt_static_graph_is_refused_before_any_launch(counting):
    cp = pytest.importorskip("cupy")
    dev, host = fresh_dev_graph()
    n = host.n_active
    pos = dev.donor_position.get().copy()
    mask = dev.donor_mask.get().copy()
    s, p = (int(v[0]) for v in np.nonzero(mask))
    bounds = tuple(host.level_bounds)
    def with_pos(row, col, value):
        t = pos.copy()
        t[row, col] = value
        return cp.asarray(t)

    def with_mask(row, col, value):
        t = mask.copy()
        t[row, col] = value
        return cp.asarray(t)

    cases = {
        "index_out_of_range": {"donor_position": with_pos(s, p, n)},
        "negative_index": {"donor_position": with_pos(s, p, -1)},
        "donor_same_cell": {"donor_position": with_pos(s, p, p)},
        "missing_donor": {"donor_mask": with_mask(s, p, False)},
        "extra_donor": {"donor_mask": with_mask((s + 1) % 4, p, True)},
        "bounds_end": {"level_bounds": bounds[:-1] + (bounds[-1] - 1,)},
        "bounds_decreasing": {"level_bounds": (0, bounds[2], bounds[1], *bounds[3:])},
        "bounds_list": {"level_bounds": list(bounds)},
        "bad_dtype": {"donor_position": dev.donor_position.astype(cp.int32)},
        "noncontiguous": {"donor_mask": cp.asarray(np.asfortranarray(mask))},
        "numpy_runtime_array": {"conveyance_lo": host.conveyance_lo.copy()},
        "k_mismatch": {"conveyance_lo": dev.conveyance_lo * 2.0},
        "zero_active": {"level_order_host": np.zeros(0, dtype=np.int64)},
        "order_host_list": {"level_order_host": host.level_order_host.tolist()},
        "order_host_none": {"level_order_host": None},
        "shape_list_of_floats": {"shape": (2.0, 3.0)},
        "shape_wrong_length": {"shape": (4,)},
        "shape_bool": {"shape": (True, True)},
        "nan_conveyance": {"conveyance": cp.where(cp.arange(dev.conveyance.size) == 0, cp.nan, dev.conveyance)},
    }
    for label, change in cases.items():
        bad = forged(dev, **change)
        with pytest.raises(RoutingError):
            routing_cuda.prepare_cuda_routing(bad)
        assert counting.launches == 0, label
        assert not routing_cuda.release_cuda_routing(bad), f"{label}: a failed preparation must not be cached"


def test_wrong_current_device_is_refused_without_migration():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("only one CUDA device is visible; the multi-device refusal is not exercised")
    dev, host = fresh_dev_graph()
    h_start, h_old = states(host.shape)
    hs, ho = cp.asarray(h_start), cp.asarray(h_old)
    routing.route_step(dev, hs, ho, 1.0, implementation="cuda")
    with cp.cuda.Device(1):
        with pytest.raises(RoutingError, match="device"):
            routing.route_step(dev, hs, ho, 1.0, implementation="cuda")
        with pytest.raises(RoutingError, match="device"):
            routing_cuda.run_sweep(dev, cp.zeros(host.n_active), C_DEFAULT, 40)


def test_wrong_device_dynamic_input_with_matching_graph_and_current_device(counting):
    """Graph, static context and current device are all on device 0; only the dynamic inputs live elsewhere."""
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("only one CUDA device is visible; the dynamic-array device refusal is not exercised")
    with cp.cuda.Device(0):
        dev, host = fresh_dev_graph()
        h_start, h_old = states(host.shape)
        ok_hs, ok_ho = cp.asarray(h_start), cp.asarray(h_old)
        routing.route_step(dev, ok_hs, ok_ho, 1.0, implementation="cuda")  # prepared and working on device 0
        counting.launches = 0
        with cp.cuda.Device(1):
            far_hs, far_ho = cp.asarray(h_start), cp.asarray(h_old)
            far_base = cp.zeros(host.n_active)
        assert int(cp.cuda.Device().id) == 0
        with pytest.raises(RoutingError, match=r"depth_start_m lives on CUDA device 1"):
            routing.route_step(dev, far_hs, ok_ho, 1.0, implementation="cuda")
        with pytest.raises(RoutingError, match=r"old_flow_depth_m lives on CUDA device 1"):
            routing.route_step(dev, ok_hs, far_ho, 1.0, implementation="cuda")
        with pytest.raises(RoutingError, match=r"base_lo lives on CUDA device 1"):
            routing_cuda.run_sweep(dev, far_base, C_DEFAULT, 40)
        assert counting.launches == 0


def test_nvrtc_compile_failure_is_unavailable_without_fallback_or_cached_context(monkeypatch):
    cp = pytest.importorskip("cupy")
    real = routing_cuda._get_kernel()
    bad_source = 'extern "C" __global__ void maple_syrup_route_level(double* x) { this is not valid CUDA C++ }'
    bad = cp.RawKernel(bad_source, routing_cuda.KERNEL_NAME, options=routing_cuda.COMPILE_OPTIONS, backend="nvrtc")
    monkeypatch.setattr(routing_cuda, "_KERNEL", bad)
    monkeypatch.setattr(routing, "_sweep_array", lambda *a, **k: pytest.fail("array fallback was used"))
    dev, host = fresh_dev_graph()
    h_start, h_old = states(host.shape)
    hs, ho = cp.asarray(h_start), cp.asarray(h_old)
    before = mb.read_transfer_counters()
    with pytest.raises(routing_cuda.CudaUnavailableError, match="compile"):
        routing_cuda.prepare_cuda_routing(dev)
    assert mb.read_transfer_counters().delta(before).device_to_host == 0, "validation ran after a failed compile"
    assert routing_cuda.release_cuda_routing(dev) is False, "a failed preparation must not be cached"
    with pytest.raises(routing_cuda.CudaUnavailableError):
        routing.route_step(dev, hs, ho, 1.0, implementation="cuda")
    with pytest.raises(routing_cuda.CudaUnavailableError):
        routing_cuda.run_sweep(dev, cp.asarray(bases(host.n_active)["wet"]), C_DEFAULT, 40)
    assert routing_cuda.release_cuda_routing(dev) is False
    # Restore the real kernel: the same graph now prepares and gives the bitwise CPU result.
    monkeypatch.setattr(routing_cuda, "_KERNEL", real)
    monkeypatch.undo()
    base_h = bases(host.n_active)["wet"]
    got = routing_cuda.run_sweep(dev, cp.asarray(base_h), C_DEFAULT, 40)
    ref = cpu_sweep(rn.compiled_sweep_batched(), host, base_h, C_DEFAULT, 40)
    for label, r, g in zip(("qin", "q", "flow", "rhs"), ref, got, strict=True):
        assert_bits(r, g, f"after compile-failure recovery {label}")
    assert routing_cuda._KERNEL is real


def test_forged_donor_slot_permutation_is_refused(counting):
    """Genuine, unique, earlier-level donors in the WRONG slot would change the legacy summation order."""
    cp = pytest.importorskip("cupy")
    dev, _host = fresh_dev_graph()
    pos = dev.donor_position.get().copy()
    mask = dev.donor_mask.get().copy()
    two = np.flatnonzero(mask.sum(axis=0) >= 2)
    one = np.flatnonzero(mask.sum(axis=0) == 1)
    assert two.size and one.size, "fixture must contain receivers with one and with several donors"
    p2 = int(two[0])
    s1, s2 = (int(v) for v in np.flatnonzero(mask[:, p2])[:2])
    swapped_pos, swapped_mask = pos.copy(), mask.copy()
    swapped_pos[[s1, s2], p2] = swapped_pos[[s2, s1], p2]
    swapped_mask[[s1, s2], p2] = swapped_mask[[s2, s1], p2]
    p1 = int(one[0])
    s_old = int(np.flatnonzero(mask[:, p1])[0])
    s_new = (s_old + 1) % 4
    moved_pos, moved_mask = pos.copy(), mask.copy()
    moved_pos[s_new, p1], moved_pos[s_old, p1] = pos[s_old, p1], pos[s_new, p1]
    moved_mask[s_new, p1], moved_mask[s_old, p1] = True, False
    for label, change in {
        "two_donors_swapped": {"donor_position": cp.asarray(swapped_pos)},
        "single_donor_moved_slot": {"donor_position": cp.asarray(moved_pos), "donor_mask": cp.asarray(moved_mask)},
    }.items():
        bad = forged(dev, **change)
        with pytest.raises(RoutingError, match="slot"):
            routing_cuda.prepare_cuda_routing(bad)
        assert counting.launches == 0, label
    routing_cuda.prepare_cuda_routing(dev)  # the unforged builder graph is still accepted


def test_provenance_on_device():
    info = routing_cuda.kernel_provenance()
    assert info["cupy"] and info["device"]["name"] and info["device"]["compute_capability"]
    assert info["cuda_runtime"] and info["compile_options"][0] == "--fmad=false"
