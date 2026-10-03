"""Phase 4R CPU-only contracts of the CUDA routing option (no device needed). Nothing here was run by its author."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from test_routing import make_graph, valley_full

pytest.importorskip("maple")

from maple_syrup import routing, routing_cuda
from maple_syrup.routing import RoutingError


def test_implementation_tuples():
    assert routing.IMPLEMENTATIONS == ("array", "numba")
    assert routing.ROUTE_IMPLEMENTATIONS == ("array", "numba", "cuda")
    assert "ROUTE_IMPLEMENTATIONS" in routing.__all__


def test_storm_control_and_legacy_cli_cannot_select_cuda():
    from maple_syrup import legacy_experiment
    from maple_syrup.storm import STORM_IMPLEMENTATIONS, StormControl

    # Task B: the water-only storm accepts "cuda" through its OWN choice list; routing.IMPLEMENTATIONS and the
    # legacy replay CLI still cannot select it.
    assert StormControl(implementation="cuda").validated().implementation == "cuda"
    assert STORM_IMPLEMENTATIONS == ("array", "numba", "cuda") and routing.IMPLEMENTATIONS == ("array", "numba")
    parser = legacy_experiment.build_parser()
    action = next(a for a in parser._actions if "--implementation" in a.option_strings)
    assert "cuda" not in action.choices


def test_numpy_graph_refuses_cuda_without_transfer():
    g = make_graph(valley_full(6, 5), ff=5.0)
    h = np.full(g.shape, 1e-3)
    with pytest.raises(RoutingError, match="CuPy graphs and arrays only"):
        routing.route_step(g, h, h * 0.5, 1.0, implementation="cuda")


def test_cuda_is_refused_for_the_literal_legacy_stale_inflow_step():
    g = make_graph(valley_full(6, 5), ff=5.0)
    h = np.full(g.shape, 1e-3)
    stale = np.zeros(g.shape)
    with pytest.raises(RoutingError, match="stale-inflow"):
        routing.legacy_stale_inflow_step(g, h, h * 0.5, 1.0, stale_old_inflow_m2_s=stale, implementation="cuda")
    # the refusal is specific to cuda: array and numba keep their literal-legacy behaviour
    from maple_syrup import routing_numba

    for impl in ("array", "numba"):
        if impl == "numba" and not routing_numba.numba_available():
            continue
        step = routing.legacy_stale_inflow_step(g, h, h * 0.5, 1.0, stale_old_inflow_m2_s=stale, implementation=impl)
        assert step.conservative is False and step.implementation == impl


def test_coupled_step_refuses_unvalidated_cuda_control_before_any_column_work(monkeypatch):
    from maple_syrup import hydrology_cuda, storm
    from maple_syrup.storm import StormControl

    def column_must_not_run(*args, **kwargs):
        raise AssertionError("column_step ran before the cuda dispatch")

    monkeypatch.setattr(storm, "column_step", column_must_not_run)
    control = StormControl(implementation="cuda")  # deliberately NOT .validated()
    # Task B: "cuda" dispatches to the prepared CUDA hydrology, which refuses unusable graphs/parameters (or a missing
    # CuPy/device) BEFORE any column work; arguments are never touched by the CPU path.
    refusals = (hydrology_cuda.CudaHydrologyPreparationError, hydrology_cuda.CudaUnavailableError)
    with pytest.raises(refusals):
        storm.coupled_step(None, None, None, None, 1.0, control)
    g = make_graph(valley_full(6, 5), ff=5.0)
    with pytest.raises(refusals):
        storm.coupled_step(g, None, None, None, 1.0, control)
    # Other implementations still reach the column step (the guard is not a general validation).
    for impl in ("array", "numba"):
        with pytest.raises(AssertionError, match="column_step ran"):
            storm.coupled_step(g, None, None, storm.StormState(0.0, None, None, None), 1.0,
                               StormControl(implementation=impl))


def test_unknown_implementation_message_lists_cuda():
    g = make_graph(valley_full(6, 5), ff=5.0)
    h = np.full(g.shape, 1e-3)
    with pytest.raises(RoutingError, match="implementation must be one of"):
        routing.route_step(g, h, h * 0.5, 1.0, implementation="cupy")


def test_importing_module_does_not_import_cupy():
    code = ("import sys; import maple_syrup.routing, maple_syrup.routing_cuda; "
            "assert 'cupy' not in sys.modules, 'cupy imported'")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_cupy_is_unavailable_without_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)  # import cupy -> ImportError
    monkeypatch.setattr(routing, "_sweep_array", lambda *a, **k: pytest.fail("array fallback used"))
    g = make_graph(valley_full(6, 5), ff=5.0)
    with pytest.raises(routing_cuda.CudaUnavailableError):
        routing_cuda.prepare_cuda_routing(g)
    assert issubclass(routing_cuda.CudaUnavailableError, RoutingError)
    info = routing_cuda.kernel_provenance()  # must not raise
    assert info["cupy"] is None


def test_source_and_options_are_ieee_strict():
    src = routing_cuda.kernel_source()
    assert "--fmad=false" in routing_cuda.COMPILE_OPTIONS
    assert not any("fast" in o for o in routing_cuda.COMPILE_OPTIONS)
    for banned in ("fma", "fast_math", "rsqrt", "__fdividef", "atomic", "__syncthreads", "volatile"):
        assert banned not in src
    for needed in ("__dadd_rn", "__dmul_rn", "__dsqrt_rn", "if (t < rhs) lo = mid;", "if (rhs > 0.0)"):
        assert needed in src
    assert "/" not in src.replace("//", "")  # the sweep contains no division
    assert routing_cuda.BLOCK_THREADS == 128


def test_provenance_fields_without_requiring_a_device():
    info = routing_cuda.kernel_provenance()
    assert info["fastmath"] is False
    assert info["compile_options"] == list(routing_cuda.COMPILE_OPTIONS)
    assert len(info["source_sha256"]) == 64
    assert info["block_threads"] == 128


@pytest.mark.parametrize("c", [0.0, -1.0, float("nan"), float("inf"), True, "1"])
def test_sweep_option_validation_c(c):
    with pytest.raises(RoutingError):
        routing_cuda._check_sweep_options(c, 40)


@pytest.mark.parametrize("it", [0, 201, -1, True, 40.0, "40", None])
def test_sweep_option_validation_iterations(it):
    with pytest.raises(RoutingError):
        routing_cuda._check_sweep_options(0.1, it)


@pytest.mark.parametrize("it", [1, 2, 40, 200, np.int64(7)])
def test_sweep_option_validation_accepts(it):
    assert routing_cuda._check_sweep_options(np.float64(0.25), it) == (0.25, int(it))
