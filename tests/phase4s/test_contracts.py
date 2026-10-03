"""Phase 4S CPU-only contracts of the CUDA hydrology and the one-block routing sweep (no device needed, no Numba needed).
Nothing here was run by its author (file-only tools); Codex records results."""
from __future__ import annotations

import re
import subprocess
import sys

import numpy as np
import pytest
from test_routing import make_graph, valley_full

pytest.importorskip("maple")

from maple_syrup import hydrology_cuda as hc
from maple_syrup import routing, routing_cuda
from maple_syrup.infiltration import column_parameters
from maple_syrup.routing import RoutingError


# --- import / optionality ------------------------------------------------------------------------------------------
def test_importing_modules_needs_neither_cupy_nor_numba():
    code = ("import sys; sys.modules['numba'] = None; sys.modules['cupy'] = None; "
            "import maple_syrup.routing_cuda, maple_syrup.hydrology_cuda as h; "
            "assert h.kernel_source() and h.select_mode(1) == 'fused'")
    subprocess.run([sys.executable, "-c", code], check=True)
    code = "import sys; import maple_syrup.hydrology_cuda; assert 'cupy' not in sys.modules, 'cupy imported'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_cupy_is_unavailable_without_any_fallback(monkeypatch):
    from maple_syrup import hydrology_numba, routing_numba, storm

    monkeypatch.setitem(sys.modules, "cupy", None)  # import cupy -> ImportError
    for owner, name in ((hydrology_numba, "prepare_hydrology"), (hydrology_numba, "prepared_coupled_step"),
                        (storm, "coupled_step"), (routing_numba, "run_sweep"), (routing, "_sweep_array")):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail("a CPU fallback was used"))
    graph = make_graph(valley_full(6, 5), ff=5.0)
    params = column_parameters(model="fixed_ksat", ksat_m_per_s=np.full(graph.shape, 1e-6),
                               suction_m=np.zeros(graph.shape), drainage_parameter=np.zeros(graph.shape),
                               theta_sat=np.full(graph.shape, 0.4), soil_thickness_m=np.full(graph.shape, 0.3),
                               active_mask=graph.active.copy())
    with pytest.raises(hc.CudaUnavailableError):
        hc.prepare_cuda_hydrology(graph, params)
    assert issubclass(hc.CudaUnavailableError, RoutingError)
    assert hc.kernel_provenance()["cupy"] is None  # must not raise
    assert hc.kernel_provenance()["numba_required"] is False


def test_numpy_graph_and_bad_arguments_are_refused_with_preparation_errors_when_cupy_exists(monkeypatch):
    # With a stand-in cupy namespace the structural refusals run before any device work; without CuPy the
    # unavailable error is the (also fallback-free) answer. Either way no CPU path is used.
    graph = make_graph(valley_full(6, 5), ff=5.0)
    with pytest.raises((hc.CudaHydrologyPreparationError, hc.CudaUnavailableError)):
        hc.prepare_cuda_hydrology(graph, object())
    with pytest.raises((hc.CudaHydrologyPreparationError, hc.CudaUnavailableError)):
        hc.prepare_cuda_hydrology(graph, object(), mode="nope")


# --- mode selection ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("width, expected", [(1, "fused"), (86, "fused"), (127, "fused"), (128, "fused"),
                                             (129, "split"), (256, "split"), (257, "split"), (10_000, "split")])
def test_auto_mode_uses_the_block_size_as_the_only_threshold(width, expected):
    assert hc.select_mode(width, "auto") == expected
    assert hc.select_mode(width) == expected
    assert hc.FUSED_MAX_LEVEL_WIDTH == hc.FUSED_THREADS == 128
    assert routing_cuda.NARROW_MAX_WIDTH == routing_cuda.BLOCK_THREADS == 128
    assert routing_cuda.resolve_sweep_mode(width, "auto") == ("block" if expected == "fused" else "level")


@pytest.mark.parametrize("width", [1, 128, 129, 5000])
def test_explicit_modes_are_always_honoured(width):
    assert hc.select_mode(width, "fused") == "fused" and hc.select_mode(width, "split") == "split"
    assert routing_cuda.resolve_sweep_mode(width, "block") == "block"
    assert routing_cuda.resolve_sweep_mode(width, "level") == "level"


def test_invalid_modes_are_refused():
    with pytest.raises(hc.CudaHydrologyPreparationError):
        hc.select_mode(10, "speculative")
    for bad in ("one-block", "", 3):
        with pytest.raises(RoutingError):
            routing_cuda.resolve_sweep_mode(10, bad)


def test_module_default_sweep_mode_is_auto_and_can_force_the_old_comparator(monkeypatch):
    assert routing_cuda.DEFAULT_SWEEP_MODE == "auto" and routing_cuda.resolve_sweep_mode(10) == "block"
    monkeypatch.setattr(routing_cuda, "DEFAULT_SWEEP_MODE", "level")
    assert routing_cuda.resolve_sweep_mode(10) == "level" and routing_cuda.resolve_sweep_mode(10, "block") == "block"


def test_launch_counts():
    bounds = (0, 3, 3, 8, 20)  # one empty level
    assert hc.launch_count(bounds, "fused") == 1
    assert hc.launch_count(bounds, "split") == 3 + 3
    with pytest.raises(hc.CudaHydrologyPreparationError):
        hc.launch_count(bounds, "auto")


# --- source hygiene ------------------------------------------------------------------------------------------------
BANNED = ("fast_math", "fastmath", "rsqrt", "__fdividef", "atomic", "volatile", "__threadfence",
          "cooperative", "cudaDeviceSynchronize", "printf", "__expf", "__powf", "__frcp", "__drcp")


def test_hydrology_source_and_options_are_ieee_strict():
    src = hc.kernel_source()
    assert "--fmad=false" in hc.COMPILE_OPTIONS and not any("fast" in o for o in hc.COMPILE_OPTIONS)
    for banned in BANNED:
        assert banned not in src, banned
    assert not re.search(r"(?<![A-Za-z_])__?(?:d|f)?fma(?:_|\()|(?<![A-Za-z_])fma\(", src), "an explicit FMA"
    assert not re.search(r"(?<![A-Za-z_])f(?:min|max)\(", src), "CUDA fmin/fmax are not NaN-propagating"
    for needed in ("fmax_nan", "fmin_nan", "__dadd_rn", "__dmul_rn", "__dsqrt_rn", "expm1(", "pow(", "isfinite("):
        assert needed in src
    assert routing_cuda.bisect_device_source() in src  # the very same root search as routing_cuda


def test_every_flag_bit_of_the_cpu_tables_is_produced_by_the_kernels():
    src = hc.kernel_source()
    for k in range(19):  # column flag bits recorded by hydrology_numba.column_kernel
        assert re.search(rf"\bf \|= 1u << {k};", src), f"column bit {k}"
    for k in range(3):  # previous-discharge bits
        assert re.search(rf"\bqf \|= 1u << {k};", src), f"qprev bit {k}"
    for k in range(26):  # route bits 0..25
        assert re.search(rf"\br \|= 1u << {k};", src), f"route bit {k}"
    from maple_syrup import hydrology_numba as hn

    assert len(hn._COLUMN_MESSAGES) == 19 and len(hn._ROUTE_STATIC_MESSAGES) == 26 and len(hn._QPREV_MESSAGES) == 3


def test_barriers_are_only_in_block_wide_kernels_and_never_in_the_per_level_source():
    assert "__syncthreads" not in routing_cuda.kernel_source()  # the accepted per-level comparator is unchanged
    block = routing_cuda.block_kernel_source()
    assert "__syncthreads()" in block and "atomic" not in block and "fma(" not in block
    assert routing_cuda.bisect_device_source() in block
    src = hc.kernel_source()
    assert src.count("extern \"C\" __global__") == 7  # 5 step kernels + storm accumulate/report
    accumulate = src[src.index("void maple_syrup_storm_accumulate"):src.index("void maple_syrup_storm_report")]
    assert "__syncthreads" not in accumulate and "atomic" not in accumulate
    fused = src[src.index("maple_syrup_hydro_fused"):src.index("maple_syrup_hydro_pre")]
    assert fused.count("__syncthreads()") == 3
    for name in ("maple_syrup_hydro_pre", "maple_syrup_hydro_solve_level", "maple_syrup_hydro_post"):
        body = src[src.index(f"void {name}"):]
        body = body[:body.index("}")]
        assert "__syncthreads" not in body, name


def test_provenance_and_packet_are_documented_without_a_device():
    info = hc.kernel_provenance()
    assert info["fastmath"] is False and info["compile_options"] == list(hc.COMPILE_OPTIONS)
    assert info["packet_bytes"] == hc.PACKET_WORDS * 8 == 144
    assert len(info["hydrology_source_sha256"]) == 64 and len(info["block_source_sha256"]) == 64
    assert info["modes"] == ["auto", "fused", "split"] and info["sweep_modes"] == ["auto", "level", "block"]
    assert "numba" not in info["kernels"]
    assert hc.prepared_cuda_hydrology is hc.prepare_cuda_hydrology and hc.cuda_coupled_step is hc.prepared_coupled_step


def test_routing_implementation_choice_lists_are_unchanged_by_task_a():
    assert routing.IMPLEMENTATIONS == ("array", "numba")
    assert routing.ROUTE_IMPLEMENTATIONS == ("array", "numba", "cuda")
