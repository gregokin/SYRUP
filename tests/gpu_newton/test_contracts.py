"""Device-free contracts of the CUDA Newton option: lazy optional imports, no fallback, one shared device text, the
default bisection sources untouched, option validation before any work. Runs without CuPy or a GPU."""
from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip("maple")

from maple_syrup import hydrology_cuda as hc
from maple_syrup import routing_cuda
from maple_syrup import routing_newton as rn
from maple_syrup import routing_newton_cuda as rnc
from maple_syrup.routing import RoutingError, route_step
from maple_syrup.storm import StormControl


def run_blocked_cupy(code: str) -> str:
    """Run `code` in a fresh interpreter where `import cupy` raises ImportError."""
    prelude = textwrap.dedent("""
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "cupy" or name.startswith("cupy."):
                    raise ImportError("cupy blocked by the test")
        sys.meta_path.insert(0, Block())
    """)
    result = subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code)], capture_output=True, text=True,
                            check=False, timeout=300)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_importing_the_newton_cuda_modules_never_imports_cupy():
    out = run_blocked_cupy("""
        import sys
        import maple_syrup.routing_newton_cuda, maple_syrup.hydrology_cuda, maple_syrup.storm
        print("cupy" in sys.modules)
    """)
    assert out.strip() == "False"


def test_missing_cupy_is_an_explicit_unavailable_error_with_no_fallback():
    out = run_blocked_cupy("""
        import numpy as np
        from maple_syrup import routing_newton_cuda as rnc, hydrology_cuda as hc
        from maple_syrup.routing_cuda import CudaUnavailableError
        for call in (rnc.ensure_loaded, hc.load_newton_kernels):
            try:
                call()
            except CudaUnavailableError as exc:
                print("unavailable", "no fallback" in str(exc))
    """)
    assert out.split() == ["unavailable", "True", "unavailable", "True"]


def test_cuda_newton_with_a_numpy_graph_is_refused_with_no_host_solve():
    from test_routing import make_graph, valley_full

    g = make_graph(valley_full(6, 5), ff=5.0)
    z = np.zeros(g.shape)
    with pytest.raises(RoutingError, match="runs on CuPy graphs"):
        route_step(g, z, z, 1.0, implementation="cuda", root_solver="newton")


@pytest.mark.parametrize("bad", [{"newton_max_iterations": 0}, {"newton_max_iterations": 1001},
                                 {"newton_max_iterations": True}, {"root_solver": "brent"}])
def test_invalid_root_options_are_refused_before_any_cuda_work(bad):
    from test_routing import make_graph, valley_full

    g = make_graph(valley_full(6, 5), ff=5.0)
    z = np.zeros(g.shape)
    kw = {"root_solver": "newton"} | bad
    with pytest.raises(RoutingError, match="root_solver|newton_max_iterations"):
        route_step(g, z, z, 1.0, implementation="cuda", **kw)  # the options fail before the graph-namespace check


def test_option_checker_is_strict():
    assert rnc.check_newton_options(0.3, 50) == (0.3, 50)
    assert rnc.check_newton_options(1, np.int64(1000)) == (1.0, 1000)
    for c in (0.0, -1.0, float("nan"), float("inf"), True, "x", None):
        with pytest.raises(RoutingError, match="c must"):
            rnc.check_newton_options(c, 50)
    for cap in (0, -1, 1001, True, 2.5, "9", None):
        with pytest.raises(RoutingError, match="newton_max_iterations"):
            rnc.check_newton_options(0.3, cap)


def test_cuda_newton_control_validates_and_is_not_refused_by_the_scheduler_control():
    control = StormControl(implementation="cuda", root_solver="newton").validated()
    assert (control.root_solver, control.newton_max_iterations) == ("newton", rn.DEFAULT_NEWTON_MAX_ITERATIONS)


def test_one_shared_device_helper_text_and_cpu_constants():
    device = rnc.newton_device_source()
    assert device in rnc.kernel_source() and device in hc.kernel_source("newton")
    assert "maple_syrup_newton(" in device and "maple_syrup_newton_root(" in device
    for constant in (repr(rn._EPS), repr(rn._CONVERGED_STEP), repr(rn._FALLBACK_STEP), str(rn._FALLBACK_LIMIT)):
        assert constant in device
    assert "__fmad" not in device and "fast" not in device.lower()  # only RN intrinsics, no fast-math forms
    assert routing_cuda.COMPILE_OPTIONS == ("--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")


def test_the_newton_hydrology_variant_differs_from_the_default_only_in_the_root_call():
    default, newton = hc.kernel_source(), hc.kernel_source("newton")
    assert "maple_syrup_bisect(rhs, k, c, iterations)" in default
    assert "maple_syrup_bisect(rhs, k, c, iterations)" not in newton
    assert newton.count("maple_syrup_newton_root(rhs, k, c, iterations)") == 1
    assert default == routing_cuda.bisect_device_source() + hc._HYDRO_TEXT + hc._STORM_BODY
    assert newton == rnc.newton_device_source() + hc._HYDRO_TEXT.replace(
        "maple_syrup_bisect(rhs, k, c, iterations)", "maple_syrup_newton_root(rhs, k, c, iterations)")
    assert "newton" not in default.lower()
    with pytest.raises(hc.CudaHydrologyPreparationError):
        hc.kernel_source("secant")


def test_context_free_provenance_reports_the_newton_variant_without_compiling():
    info = hc.kernel_provenance()
    assert info["newton_step_kernels"] == list(hc._KERNEL_NAMES[:5])
    assert set(info["root_solvers"]) == {"bisection", "newton"} and len(info["newton_hydrology_source_sha256"]) == 64
    assert rnc.kernel_provenance()["newton_stats"].startswith("production: null pointer")


def test_gpu_sediment_and_disk_event_restart_stay_refused_for_cuda_newton():
    from maple_syrup.sediment_event import SedimentEventControl, SedimentEventError

    control = SedimentEventControl(storm=StormControl(implementation="cuda", root_solver="newton"))
    with pytest.raises(SedimentEventError, match="WATER-ONLY CUDA hydrology"):
        control.validated()  # the event control (and therefore the disk checkpoint path) never accepts a CUDA storm


def test_cuda_newton_control_round_trips_through_dataclass_metadata():
    import dataclasses

    control = StormControl(implementation="cuda", root_solver="newton", newton_max_iterations=7).validated()
    clone = StormControl(**{f.name: getattr(control, f.name) for f in dataclasses.fields(control)}).validated()
    assert clone == control and clone.root_solver == "newton" and clone.newton_max_iterations == 7
    assert StormControl(implementation="cuda").root_solver == "bisection"  # default metadata unchanged
