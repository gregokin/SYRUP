"""CUDA Newton ordered sweep and `route_step(implementation="cuda", root_solver="newton")` against the CPU Newton forms
on identical inputs: graphs with merges, masks, outlets and pits, all launch structures (level / block / auto), width
boundaries around the block size, special values, forced low caps, conservation/positivity/constitutive residuals,
guards and failure labels. The default bisection kernel sources and results stay unchanged. Needs a device."""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest
from gn_helpers import C_DEFAULT, NAMES, bases, bit_mismatches, graph_pair
from rfid_helpers import build, pit_chain
from test_routing import chain_full, make_graph, valley_full

pytest.importorskip("maple")

from maple_syrup import routing_cuda
from maple_syrup import routing_newton as rn
from maple_syrup import routing_newton_cuda as rnc
from maple_syrup.routing import RoutingError, RoutingStepRejected, route_step
from maple_syrup.routing_numba import numba_available

pytestmark = pytest.mark.usefixtures("gpu")
MODES = ["level", "block", "auto"]
WATER_RTOL, WATER_ATOL = 2.0e-12, 1.0e-14
# SHA-256 of the DEFAULT bisection kernel texts of the accepted baseline d5861b4 (they must not change).
BASELINE_SHA = {
    "level": "2189717bbc88f79ddf7f88e08115871c7a644854faaf0c90ea7734810f7e3e49",
    "block": "ad2911e51fa7b610722c5cc6d9c6e6c6913f9c23bd0880e921a3d477c5a44413",
    "hydrology": "4743809f27f9f1daefbd26545a18891a1299c636748060acfc0e81c5c130033f",
}


def cpu_newton_sweep(host, base, c, cap):
    """The compiled CPU Newton sweep (needs Numba) or, without it, the NumPy per-level form on the same inputs."""
    if numba_available():
        qin, q, flow, rhs, stats = rn.run_sweep(host, np.array(base, dtype=np.float64), c, cap)
        return (qin, q, flow, rhs), rn.stats_dict(stats)
    from maple_syrup.routing import _sweep_array_newton

    qin, q, flow, rhs, stats = _sweep_array_newton(host, np.array(base, dtype=np.float64), c, cap)
    return (qin, q, flow, rhs), stats if isinstance(stats, dict) else rn.stats_dict(stats)


@pytest.mark.parametrize("cap", [50, 2])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("base_name", list(bases(4)))
@pytest.mark.parametrize("name", NAMES)
def test_sweep_is_bitwise_the_cpu_newton_sweep(name, base_name, mode, cap):
    import cupy as cp

    host, dev = graph_pair(name)
    base = bases(host.n_active)[base_name]
    with np.errstate(all="ignore"):
        ref, ref_stats = cpu_newton_sweep(host, base, C_DEFAULT, cap)
    base_d = cp.asarray(base)
    qin, q, flow, rhs, counters = rnc.run_sweep(dev, base_d, C_DEFAULT, cap, mode, stats=True)
    for label, r, g in zip(("qin", "q", "flow", "rhs"), ref, (qin, q, flow, rhs), strict=True):
        assert bit_mismatches(r, g.get()) == 0, f"{name}/{base_name}/{mode}/cap={cap} {label}"
    assert rnc.stats_to_dict(counters) == ref_stats
    assert np.array_equal(base_d.get().view(np.uint64), base.view(np.uint64)), "base_lo was modified"


def test_all_launch_structures_give_identical_outputs_and_production_has_no_counters():
    import cupy as cp

    host, dev = graph_pair("random_w257")
    base = cp.asarray(bases(host.n_active)["deep"])
    runs = [rnc.run_sweep(dev, base, C_DEFAULT, 50, mode) for mode in MODES]
    assert all(len(r) == 4 for r in runs)  # no counter block is allocated unless stats=True
    for other in runs[1:]:
        for a, b in zip(runs[0], other, strict=True):
            assert bit_mismatches(a.get(), b.get()) == 0


def test_deep_water_and_extreme_c_keep_the_bitwise_agreement():
    import cupy as cp

    for c in (1e-300, 1e-9, 1.0, 1e3):
        for name in ("valley_branching", "random_w129"):
            host, dev = graph_pair(name)
            base = bases(host.n_active)["deep"]
            with np.errstate(all="ignore"):
                ref, _ = cpu_newton_sweep(host, base, c, 50)
            got = rnc.run_sweep(dev, cp.asarray(base), c, 50, "auto")
            for r, g in zip(ref, got, strict=True):
                assert bit_mismatches(r, g.get()) == 0, (name, c)


def test_invalid_options_are_refused_before_any_launch_or_allocation(monkeypatch):
    import cupy as cp

    host, dev = graph_pair("valley_branching")
    base = cp.asarray(bases(host.n_active)["wet"])
    launched = []
    real = rnc._function

    class Recorder:
        def __init__(self, fn, name):
            self.fn, self.name = fn, name

        def __call__(self, *args, **kwargs):
            launched.append(self.name)
            return self.fn(*args, **kwargs)

    monkeypatch.setattr(rnc, "_function", lambda name: Recorder(real(name), name))
    for kwargs in ({"max_iterations": 0}, {"max_iterations": 1001}, {"max_iterations": True}, {"max_iterations": 1.5}):
        with pytest.raises(RoutingError, match="newton_max_iterations"):
            rnc.run_sweep(dev, base, C_DEFAULT, kwargs["max_iterations"])
    with pytest.raises(RoutingError, match="sweep mode"):
        rnc.run_sweep(dev, base, C_DEFAULT, 50, "warp")
    with pytest.raises(RoutingError, match="c must"):
        rnc.run_sweep(dev, base, -1.0, 50)
    with pytest.raises(RoutingError, match="base_lo"):
        rnc.run_sweep(dev, cp.asarray(base.get()[:-1]), C_DEFAULT, 50)
    with pytest.raises(RoutingError, match="base_lo"):
        rnc.run_sweep(dev, cp.asarray(base.get().astype(np.float32)), C_DEFAULT, 50)
    assert launched == []
    with pytest.raises(RoutingError, match="CuPy graph|xp=cupy|cuda"):
        rnc.run_sweep(host, base, C_DEFAULT, 50)


# --- route_step -------------------------------------------------------------------------------------------------
def state(graph, rng, *, scale=0.01, extra=0.002, dry_fraction=0.0):
    h_old = rng.uniform(0.0, scale, graph.shape) * (rng.random(graph.shape) >= dry_fraction)
    h_old = np.where(graph.active, h_old, 0.0)
    start = np.where(graph.active, h_old + rng.uniform(0.0, extra, graph.shape), 0.0)
    k = graph.conveyance.reshape(graph.shape)
    q_old = np.where(graph.active, (np.sqrt(h_old) * h_old) * k, 0.0)
    return start, h_old, q_old


def route_graphs():
    rng = np.random.default_rng(0)
    from test_routing import random_full

    out = {"valley": (valley_full(8, 7), 5.0), "chain": (chain_full(9), 1.0),
           "random": (random_full(rng, 11, 8), rng.uniform(0.2, 40.0, (11, 8)))}
    masked = np.ones((6, 5), dtype=bool)
    masked[-1, 0] = masked[-1, -1] = False
    out["masked"] = (valley_full(6, 5), 5.0, masked)
    return out


def both_graphs(name):
    import cupy as cp

    spec = route_graphs()[name]
    z, ff = spec[0], spec[1]
    active = spec[2] if len(spec) > 2 else None
    return make_graph(z, ff=ff, active=active), make_graph(z, ff=ff, active=active, xp=cp)


FIELDS = ("depth_m", "flow_depth_m", "discharge_m2_s", "velocity_m_s", "inflow_m2_s", "old_discharge_m2_s",
          "old_inflow_m2_s", "face_volume_m3")
SCALARS = ("export_m3", "outlet_discharge_m3_s", "storage_change_m3", "budget_residual_m3",
           "max_courant_old", "max_courant_new", "max_constitutive_residual_m", "max_cell_balance_residual_m")


@pytest.mark.parametrize("name", ["valley", "chain", "random", "masked"])
def test_route_step_cuda_newton_matches_cpu_newton_within_the_water_bounds(name):
    import cupy as cp

    host, dev = both_graphs(name)
    rng = np.random.default_rng(4)
    for dt, dry in ((0.05, 0.0), (1.0, 0.25)):
        start, h_old, q_old = state(host, rng, dry_fraction=dry)
        ref = route_step(host, start, h_old, dt, old_discharge_m2_s=q_old, implementation="array", root_solver="newton")
        got = route_step(dev, cp.asarray(start), cp.asarray(h_old), dt, old_discharge_m2_s=cp.asarray(q_old),
                         implementation="cuda", root_solver="newton")
        assert (got.root_solver, got.implementation, got.newton_max_iterations, got.bisection_iterations) == \
            ("newton", "cuda", 50, 0)
        assert got.root_stats is None  # uninstrumented device production: not a lie of zero counts
        assert got.conservative
        for f in FIELDS:
            np.testing.assert_allclose(getattr(got, f).get(), getattr(ref, f), rtol=WATER_RTOL, atol=WATER_ATOL,
                                       err_msg=f)
        for f in SCALARS:
            np.testing.assert_allclose(float(getattr(got, f)), float(getattr(ref, f)), rtol=WATER_RTOL,
                                       atol=WATER_ATOL, err_msg=f)
        # conservation, positivity and the constitutive residual of the CUDA step itself
        assert abs(float(got.budget_residual_m3)) <= 1e-12 and float(got.max_cell_balance_residual_m) <= 1e-12
        assert float(got.max_constitutive_residual_m) <= 1e-12
        for f in ("depth_m", "flow_depth_m", "discharge_m2_s"):
            assert (getattr(got, f).get() >= 0.0).all()


@pytest.mark.parametrize("mode", MODES)
def test_every_launch_structure_runs_the_same_route_step(mode, monkeypatch):
    import cupy as cp

    monkeypatch.setattr(routing_cuda, "DEFAULT_SWEEP_MODE", mode)
    host, dev = both_graphs("random")
    start, h_old, q_old = state(host, np.random.default_rng(2))
    kw = {"old_discharge_m2_s": cp.asarray(q_old), "implementation": "cuda", "root_solver": "newton"}
    got = route_step(dev, cp.asarray(start), cp.asarray(h_old), 0.5, **kw)
    ref = route_step(host, start, h_old, 0.5, old_discharge_m2_s=q_old, implementation="array", root_solver="newton")
    np.testing.assert_allclose(got.flow_depth_m.get(), ref.flow_depth_m, rtol=WATER_RTOL, atol=WATER_ATOL)


def test_pit_storage_and_export_cells_match_the_cpu_newton_step():
    import cupy as cp

    z = pit_chain()
    host = build(z, allow_pit_storage=True)
    dev = build(z, allow_pit_storage=True, xp=cp)
    assert bool(host.pit_storage.any()) and bool(host.outlet_flat.any())
    start, h_old, q_old = state(host, np.random.default_rng(3), scale=0.02, extra=0.004)
    ref = route_step(host, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation="array", root_solver="newton")
    got = route_step(dev, cp.asarray(start), cp.asarray(h_old), 1.0, old_discharge_m2_s=cp.asarray(q_old),
                     implementation="cuda", root_solver="newton")
    for f in FIELDS:
        np.testing.assert_allclose(getattr(got, f).get(), getattr(ref, f), rtol=WATER_RTOL, atol=WATER_ATOL,
                                   err_msg=f)
    pit = host.pit_storage
    assert (got.discharge_m2_s.get()[pit] == 0.0).all()  # k = 0: nothing leaves, h_flow = h (no water deleted)
    assert abs(float(got.budget_residual_m3)) <= 1e-12


def test_default_cuda_bisection_is_unchanged_and_ignores_newton_options():
    import cupy as cp

    host, dev = both_graphs("valley")
    start, h_old, q_old = state(host, np.random.default_rng(1))
    args = (dev, cp.asarray(start), cp.asarray(h_old), 1.0)
    kw = {"old_discharge_m2_s": cp.asarray(q_old), "implementation": "cuda"}
    default = route_step(*args, **kw)
    explicit = route_step(*args, **kw, root_solver="bisection", newton_max_iterations=7)
    assert default.root_solver == "bisection" and default.root_stats is None and default.bisection_iterations == 40
    assert default.newton_max_iterations == 0
    for f in dataclasses.fields(default):
        a, b = getattr(default, f.name), getattr(explicit, f.name)
        if hasattr(a, "get"):
            assert bit_mismatches(a.get(), b.get()) == 0 if a.dtype == np.float64 else (a.get() == b.get()).all()
        else:
            assert a == b, f.name


def test_default_bisection_kernel_sources_are_the_accepted_baseline_text():
    from maple_syrup import hydrology_cuda as hc

    sha = lambda t: hashlib.sha256(t.encode()).hexdigest()
    assert sha(routing_cuda.kernel_source()) == BASELINE_SHA["level"]
    assert sha(routing_cuda.block_kernel_source()) == BASELINE_SHA["block"]
    assert sha(hc.kernel_source()) == BASELINE_SHA["hydrology"]
    assert "newton" not in hc.kernel_source().lower() and "newton" not in routing_cuda.kernel_source().lower()


def test_invalid_pairings_options_and_inputs_are_refused_before_mutation():
    import cupy as cp

    host, dev = both_graphs("valley")
    start, h_old, q_old = state(host, np.random.default_rng(1))
    d_args = (cp.asarray(start), cp.asarray(h_old), 1.0)
    before = [a.get().copy() for a in d_args[:2]]
    d_q = cp.asarray(q_old)
    for bad in ({"newton_max_iterations": 0}, {"newton_max_iterations": 1001}, {"newton_max_iterations": True},
                {"newton_max_iterations": 2.5}, {"root_solver": "brent"}, {"root_solver": None}):
        kw = {"root_solver": "newton"} | bad
        with pytest.raises(RoutingError, match="root_solver|newton_max_iterations"):
            route_step(dev, *d_args, old_discharge_m2_s=d_q, implementation="cuda", **kw)
    # CuPy graph with a host implementation: refused, never transferred
    for impl in ("array", "numba"):
        with pytest.raises(RoutingError):
            route_step(dev, *d_args, old_discharge_m2_s=d_q, implementation=impl, root_solver="newton")
    # host graph with cuda: refused with the CuPy message (no host solve)
    with pytest.raises(RoutingError, match="runs on CuPy graphs"):
        route_step(host, start, h_old, 1.0, old_discharge_m2_s=q_old, implementation="cuda", root_solver="newton")
    # NumPy dynamic arrays on a CuPy graph
    with pytest.raises(RoutingError):
        route_step(dev, start, h_old, 1.0, old_discharge_m2_s=d_q, implementation="cuda", root_solver="newton")
    # the literal-legacy stale-inflow comparison tool has no solver option and is still refused for CUDA
    from maple_syrup.routing import legacy_stale_inflow_step

    with pytest.raises(RoutingError, match="stale-inflow"):
        legacy_stale_inflow_step(dev, *d_args, stale_old_inflow_m2_s=d_q * 0.0, old_discharge_m2_s=d_q,
                                 implementation="cuda")
    for a, b in zip(d_args[:2], before, strict=True):
        np.testing.assert_array_equal(a.get(), b)


def test_nonconvergence_is_labelled_newton_not_bisection_and_mutates_nothing():
    import cupy as cp

    host, dev = both_graphs("valley")
    start, h_old, q_old = state(host, np.random.default_rng(1))
    d = [cp.asarray(a) for a in (start, h_old, q_old)]
    before = [a.get().copy() for a in d]
    with pytest.raises(RoutingError, match="Newton root solver did not reach root_tolerance_m") as info:
        route_step(dev, d[0], d[1], 1.0, old_discharge_m2_s=d[2], implementation="cuda", root_solver="newton",
                   root_tolerance_m=1e-300, newton_max_iterations=9)
    assert "bisection did not reach" not in str(info.value) and "newton_max_iterations = 9" in str(info.value)
    for a, b in zip(d, before, strict=True):
        np.testing.assert_array_equal(a.get(), b)


def test_courant_rejection_is_the_same_recoverable_error():
    import cupy as cp

    host, dev = both_graphs("valley")
    h_old = np.where(host.active, 0.05, 0.0)
    k = host.conveyance.reshape(host.shape)
    q_old = np.where(host.active, (np.sqrt(h_old) * h_old) * k, 0.0)
    for graph, conv in ((host, lambda a: a), (dev, cp.asarray)):
        with pytest.raises(RoutingStepRejected, match="Courant"):
            route_step(graph, conv(h_old.copy()), conv(h_old.copy()), 40.0, old_discharge_m2_s=conv(q_old),
                       implementation="array" if graph is host else "cuda", root_solver="newton")


def test_provenance_records_the_newton_source_and_options():
    info = rnc.kernel_provenance()
    assert info["root_solver"] == "newton" and info["compile_options"] == list(routing_cuda.COMPILE_OPTIONS)
    assert "--fmad=false" in info["compile_options"] and info["fastmath"] is False
    assert info["newton_source_sha256"] == hashlib.sha256(rnc.kernel_source().encode()).hexdigest()
    assert info["newton_constants"]["fallback_limit"] == rn._FALLBACK_LIMIT
    assert info["device"] is not None and info["newton_kernels"] == list(rnc._KERNEL_NAMES)
