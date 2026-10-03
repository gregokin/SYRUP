"""Phase 4S: the one-block whole-sweep routing kernel against the per-level kernel and the CPU sweeps, on a real device.

All four raw sweep outputs are compared BIT FOR BIT (uint64 views; NaN positions and sign only, payload bits are not part
of the contract). The per-level comparator (`mode="level"`) stays available and the one-block kernel
(`mode="block"`) is also forced on wide graphs, where the strided cell loop (not the launch count) is what is tested.
Nothing here was run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("maple")

from maple.core import backend as mb
from test_cuda_sweep import (  # Phase 4R helpers (graph builders, bases, CPU sweeps, bit comparison)
    C_DEFAULT,
    ITERATIONS,
    NAMES,
    PLANE_WIDTHS,
    assert_bits,
    bases,
    cpu_sweep,
    graph_pair,
)

from maple_syrup import routing, routing_cuda
from maple_syrup import routing_numba as rn
from maple_syrup.routing import RoutingError

pytestmark = pytest.mark.usefixtures("gpu")
NUMBA = pytest.mark.skipif(not rn.numba_available(), reason="Numba oracle not installed")


def cupy():
    import cupy as cp

    return cp


@pytest.mark.parametrize("it", [1, 40, 200])
@pytest.mark.parametrize("base_name", list(bases(4)))
@pytest.mark.parametrize("name", NAMES)
def test_block_equals_level_bitwise(name, base_name, it):
    cp = cupy()
    _host, dev = graph_pair(name)
    base = bases(dev.n_active)[base_name]
    level = routing_cuda.run_sweep(dev, cp.asarray(base), C_DEFAULT, it, mode="level")
    block = routing_cuda.run_sweep(dev, cp.asarray(base), C_DEFAULT, it, mode="block")
    for label, a, b in zip(("qin", "q", "flow", "rhs"), level, block, strict=True):
        assert_bits(a.get(), b, f"{name}/{base_name}/it={it} {label} block vs level")


@NUMBA
@pytest.mark.parametrize("it", ITERATIONS)
@pytest.mark.parametrize("base_name", ["wet", "mixed", "dry", "negative", "subnormal", "special"])
@pytest.mark.parametrize("name", NAMES)
def test_block_bitwise_matches_both_cpu_sweeps(name, base_name, it):
    cp = cupy()
    host, dev = graph_pair(name)
    base = bases(host.n_active)[base_name]
    serial = cpu_sweep(rn.compiled_sweep(), host, base, C_DEFAULT, it)
    batched = cpu_sweep(rn.compiled_sweep_batched(), host, base, C_DEFAULT, it)
    base_d = cp.asarray(base)
    got = routing_cuda.run_sweep(dev, base_d, C_DEFAULT, it, mode="block")
    for label, s, b, g in zip(("qin", "q", "flow", "rhs"), serial, batched, got, strict=True):
        assert_bits(s, g, f"{name}/{base_name}/it={it} {label} vs original serial")
        assert_bits(b, g, f"{name}/{base_name}/it={it} {label} vs level-batched")
    assert np.array_equal(cp.asnumpy(base_d).view(np.uint64), base.view(np.uint64)), "base_lo was modified"


@pytest.mark.parametrize("it", [1, 40, 200])
@pytest.mark.parametrize("base_name", ["wet", "mixed", "negative", "subnormal", "special"])
@pytest.mark.parametrize("width", PLANE_WIDTHS)
def test_block_at_exact_block_boundary_widths(width, base_name, it):
    cp = cupy()
    _host, dev = graph_pair(f"plane_{width}")
    base = bases(dev.n_active)[base_name]
    level = routing_cuda.run_sweep(dev, cp.asarray(base), C_DEFAULT, it, mode="level")
    block = routing_cuda.run_sweep(dev, cp.asarray(base), C_DEFAULT, it, mode="block")
    auto = routing_cuda.run_sweep(dev, cp.asarray(base), C_DEFAULT, it, mode="auto")
    for label, a, b, c in zip(("qin", "q", "flow", "rhs"), level, block, auto, strict=True):
        assert_bits(a.get(), b, f"plane_{width}/{base_name}/it={it} {label} block")
        assert_bits(a.get(), c, f"plane_{width}/{base_name}/it={it} {label} auto")


@pytest.mark.parametrize("c", [1e-300, 1e-9, 1.0, 1e3])
@pytest.mark.parametrize("base_name", ["wet", "mixed", "subnormal", "special"])
@pytest.mark.parametrize("name", ["valley_branching", "random_w129"])
def test_block_bitwise_for_extreme_c(name, base_name, c):
    cp = cupy()
    _host, dev = graph_pair(name)
    base = bases(dev.n_active)[base_name]
    level = routing_cuda.run_sweep(dev, cp.asarray(base), c, 40, mode="level")
    block = routing_cuda.run_sweep(dev, cp.asarray(base), c, 40, mode="block")
    for label, a, b in zip(("qin", "q", "flow", "rhs"), level, block, strict=True):
        assert_bits(a.get(), b, f"{name}/{base_name}/c={c} {label}")


# --- launch structure ---------------------------------------------------------------------------------------------
class Counting:
    def __init__(self, real):
        self.real, self.launches, self.calls = real, 0, []

    def __call__(self, grid, block, args):
        self.launches += 1
        self.calls.append((grid, block))
        return self.real(grid, block, args)

    def __getattr__(self, name):
        return getattr(self.real, name)


@pytest.fixture
def counting(monkeypatch):
    level, block = Counting(routing_cuda._get_kernel()), Counting(routing_cuda._get_block_kernel())
    monkeypatch.setattr(routing_cuda, "_get_kernel", lambda: level)
    monkeypatch.setattr(routing_cuda, "_get_block_kernel", lambda: block)
    return level, block


def _fresh(name):
    import cupy as cp
    from test_routing import chain_full, make_graph, random_full, valley_full

    width = int(name.split("_w")[1]) if "_w" in name else None
    rng = np.random.default_rng(11)
    z = {"valley_branching": lambda: valley_full(6, 5), "chain": lambda: chain_full(9),
         "plane_86": lambda: np.repeat((np.arange(5.0) * 0.015625)[:, None], 88, axis=1)}.get(name)
    if z is None:
        z = lambda: random_full(rng, 6, width)
        return make_graph(z(), ff=rng.uniform(5.0, 30.0, (6, width)), xp=cp)
    return make_graph(z(), ff=5.0, xp=cp)


@pytest.mark.parametrize("name, mode, expect_block", [
    ("valley_branching", None, True), ("valley_branching", "auto", True), ("valley_branching", "level", False),
    ("valley_branching", "block", True), ("plane_86", "auto", True), ("random_w129", "auto", False),
    ("random_w129", "block", True), ("random_w257", None, False),
])
def test_launch_structure_per_mode(counting, name, mode, expect_block):
    cp = cupy()
    level, block = counting
    dev = _fresh(name)
    base = cp.asarray(bases(dev.n_active)["wet"])
    ctx = routing_cuda.prepare_cuda_routing(dev)
    level.launches = block.launches = 0
    before = mb.read_transfer_counters()
    routing_cuda.run_sweep(dev, base, C_DEFAULT, 40, mode=mode)
    delta = mb.read_transfer_counters().delta(before)
    if expect_block:
        assert block.launches == 1 and level.launches == 0
        assert block.calls[0] == ((1,), (routing_cuda.BLOCK_THREADS,))
    else:
        assert level.launches == dev.n_levels and block.launches == 0
        assert all(b == (routing_cuda.BLOCK_THREADS,) for _g, b in level.calls)
    if mode in (None, "auto"):  # the automatic choice follows the widest level, nothing else
        assert expect_block == (ctx.max_level_width <= routing_cuda.NARROW_MAX_WIDTH)
    assert delta.host_to_device == delta.device_to_host == delta.scalar_reads == delta.synchronizations == 0


def test_invalid_mode_is_refused_before_preparation_or_launch(counting):
    cp = cupy()
    level, block = counting
    dev = _fresh("valley_branching")
    base = cp.asarray(bases(dev.n_active)["wet"])
    for bad in ("one-block", "", "BLOCK", 3, True):
        with pytest.raises(RoutingError, match="sweep mode"):
            routing_cuda.run_sweep(dev, base, C_DEFAULT, 40, mode=bad)
    assert level.launches == block.launches == 0
    assert routing_cuda.release_cuda_routing(dev) is False, "an invalid mode must not prepare/cache a context"


def test_module_default_can_restore_the_old_per_level_behaviour(counting, monkeypatch):
    cp = cupy()
    level, block = counting
    dev = _fresh("valley_branching")
    base = cp.asarray(bases(dev.n_active)["wet"])
    routing_cuda.prepare_cuda_routing(dev)
    monkeypatch.setattr(routing_cuda, "DEFAULT_SWEEP_MODE", "level")
    level.launches = block.launches = 0
    routing_cuda.run_sweep(dev, base, C_DEFAULT, 40)
    assert level.launches == dev.n_levels and block.launches == 0


def test_route_step_is_bitwise_identical_in_both_default_modes(monkeypatch):
    cp = cupy()
    dev = _fresh("valley_branching")
    rng = np.random.default_rng(5)
    h_start = rng.uniform(1e-3, 3e-3, dev.shape)
    h_old = h_start * 0.5
    hs, ho = cp.asarray(h_start), cp.asarray(h_old)
    a = routing.route_step(dev, hs, ho, 1.0, implementation="cuda")
    monkeypatch.setattr(routing_cuda, "DEFAULT_SWEEP_MODE", "level")
    b = routing.route_step(dev, hs, ho, 1.0, implementation="cuda")
    for field in ("depth_m", "flow_depth_m", "discharge_m2_s", "velocity_m_s", "inflow_m2_s", "old_inflow_m2_s",
                  "face_volume_m3"):
        assert_bits(getattr(a, field).get(), getattr(b, field), field)
    assert a.implementation == b.implementation == "cuda"


def test_context_owns_the_level_bounds_and_records_the_modes():
    cp = cupy()
    dev = _fresh("valley_branching")
    ctx = routing_cuda.prepare_cuda_routing(dev)
    assert ctx.level_bounds_device.dtype == cp.int64 and tuple(ctx.level_bounds_device.get().tolist()) == ctx.level_bounds
    assert ctx.level_bounds_device.data.ptr != dev.conveyance.data.ptr
    s = ctx.summary()
    assert s["max_level_width"] == ctx.max_level_width and s["n_levels"] == dev.n_levels
    info = routing_cuda.kernel_provenance()
    assert info["default_sweep_mode"] == "auto" and info["narrow_max_width"] == 128
    assert len(info["block_source_sha256"]) == 64
