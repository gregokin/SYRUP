"""CPU-only contracts of the EXPERIMENTAL hydraulic alternatives (no device, no Numba needed): lazy optional imports, no fallback,
kernel-source hygiene, flag-table agreement, untouched production defaults, one shared driver, CLI refusals. Nothing here was
run by its author (file-only tools); Codex records results."""
from __future__ import annotations

import ast
import inspect
import re
import subprocess
import sys

import numpy as np
import pytest
from cand_cases import build

pytest.importorskip("maple")

from maple_syrup import experimental_cuda as ec
from maple_syrup import experimental_experiment as ee
from maple_syrup import experimental_hydrology as eh
from maple_syrup import experimental_storm as es
from maple_syrup import routing, storm
from maple_syrup.routing_cuda import CudaUnavailableError

MODULES = ("experimental_hydrology", "experimental_cuda", "experimental_storm", "experimental_experiment")


def test_modules_import_without_cupy_and_without_numba():
    code = ("import sys; sys.modules['numba'] = None; sys.modules['cupy'] = None; "
            + "; ".join(f"import maple_syrup.{m}" for m in MODULES)
            + "; assert 'cupy' not in sys.modules or sys.modules['cupy'] is None")
    subprocess.run([sys.executable, "-c", code], check=True)
    code = "import sys; " + "; ".join(f"import maple_syrup.{m}" for m in MODULES) + "; assert 'cupy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_cupy_is_an_explicit_error_without_any_cpu_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(eh.CpuHydraulicSolver, "step", lambda *a, **k: pytest.fail("the CPU solver was used as a fallback"))
    cs = build("valley:6x5", "explicit")
    with pytest.raises(CudaUnavailableError):
        ec.prepare_experimental_cuda("explicit", cs.graph, cs.params)
    with pytest.raises(CudaUnavailableError):
        ec.CudaHydraulicSolver("explicit", cs.graph, cs.params)
    assert ec.kernel_provenance()["numba_required"] is False


def test_kernel_source_is_ieee_strict_and_uses_no_libm_transcendental():
    src = ec.kernel_source()
    assert "--fmad=false" in ec.COMPILE_OPTIONS and not any("fast" in o for o in ec.COMPILE_OPTIONS)
    for banned in ("fast_math", "fastmath", "rsqrt", "__fdividef", "atomic", "volatile", "__threadfence", "printf",
                   "__expf", "__powf", "expm1(", "exp(", "pow(", "log(", "__frcp", "__drcp"):
        assert banned not in src, banned
    assert not re.search(r"(?<![A-Za-z_])__?(?:d|f)?fma(?:_|\()|(?<![A-Za-z_])fma\(", src), "an explicit FMA"
    for needed in ("__dadd_rn", "__dmul_rn", "__dsqrt_rn", "__ddiv_rn", "fmax_nan", "isfinite("):
        assert needed in src
    names = re.findall(r'extern "C" __global__ void (\w+)', src)
    assert names == [*ec._KERNELS["explicit"], *ec._KERNELS["local_inertial"]] and len(names) == 10
    assert src.count("__syncthreads()") == 3 * 4  # only the four block-reduction helpers (three barriers each)


def test_flag_tables_agree_with_the_kernels_and_resolution_order_is_documented_in_code():
    assert eh.EXPLICIT_BITS == {"nonfinite": 1, "negative": 2, "cfl": 4, "balance": 8}
    assert eh.LOCAL_BITS == {"state_nonfinite": 1, "closed_face": 2, "open_inflow": 4, "cfl": 8, "negative": 16,
                             "nonfinite": 32, "balance": 64}
    # CFL (recoverable) is reported before non-finite/negative/balance failures; state errors come first for local inertia
    with pytest.raises(eh.HydraulicStepRejected):
        eh.resolve_flags("explicit", 4 | 1, (False,) * 5, False, 2.0, 0.5, "off")
    with pytest.raises(eh.ExperimentalHydrologyError, match="finite") as info:
        eh.resolve_flags("local_inertial", 1 | 8, (False,) * 5, False, 2.0, 0.5, "off")
    assert type(info.value) is eh.ExperimentalHydrologyError
    with pytest.raises(eh.HydraulicStepRejected, match="negative depth"):
        eh.resolve_flags("local_inertial", 16, (False,) * 5, False, 0.1, 0.5, "off")
    with pytest.raises(eh.ExperimentalHydrologyError) as info:
        eh.resolve_flags("local_inertial", 16, (False,) * 5, False, 0.1, 0.5, "donor")  # with the limiter: an error, not a retry
    assert type(info.value) is eh.ExperimentalHydrologyError
    assert ec.PACKET_WORDS * 8 == 128


def test_production_defaults_and_modules_are_untouched_by_the_experiments():
    assert storm.STORM_IMPLEMENTATIONS == ("array", "numba", "cuda") and routing.IMPLEMENTATIONS == ("array", "numba")
    from maple_syrup import (
        benchmark_experiment,
        hydrology_cuda,
        hydrology_numba,
        legacy_experiment,
        routing_cuda,
        sediment_experiment,
        storm_experiment,
    )

    for module in (storm, routing, routing_cuda, hydrology_cuda, hydrology_numba, storm_experiment, legacy_experiment,
                   benchmark_experiment, sediment_experiment):
        assert "experimental_" not in inspect.getsource(module), module.__name__  # nothing imports the experiments
    parser_actions = []
    import argparse

    original = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = lambda self, *a, **k: (parser_actions.append(self), (_ for _ in ()).throw(SystemExit(0)))
    try:
        with pytest.raises(SystemExit):
            storm_experiment.main(["--case-dir", "x", "--output-dir", "y"])
    finally:
        argparse.ArgumentParser.parse_args = original
    action = next(a for a in parser_actions[0]._actions if "--implementation" in a.option_strings)
    assert "explicit" not in action.choices and "local_inertial" not in action.choices


def test_one_shared_driver_and_no_second_framework():
    assert callable(es.evolve_experimental)
    for module in (eh, ec, ee):
        assert "def evolve" not in inspect.getsource(module)  # the scheduler exists once, in experimental_storm
    assert "HydraulicControl" in inspect.getsource(ee) and "evolve_experimental" in inspect.getsource(ee)


def executable_names(source: str) -> dict:
    """Identifiers that source text DEFINES or EXECUTES, from its AST: defined function/class names, called names/attributes and
    imported names. String constants (docstrings, metadata text, the CUDA kernel source) are not identifiers and are
    deliberately outside this contract (the kernel source has its own hygiene test)."""
    defined, called, imported = set(), set(), set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Call):
            func = node.func
            called.add(func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else "")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imported.update(alias.name for alias in node.names)
    return {"defined": defined, "called": called, "imported": imported}


SECOND_LAW = re.compile(r"expm1|smith|capacity|hawkins|green_?ampt", re.IGNORECASE)


def second_copies(names: dict) -> list:
    """Executable evidence of a second infiltration law: a defined function/class named like the column physics or a law, or a
    call/import of an infiltration primitive."""
    return sorted({n for n in names["defined"] if SECOND_LAW.search(n) or n.lower().startswith("column")}
                  | {n for n in names["called"] | names["imported"] if SECOND_LAW.search(n)})


def test_the_experimental_modules_reuse_the_accepted_column_stage_and_define_no_second_copy():
    cpu, gpu = executable_names(inspect.getsource(eh)), executable_names(inspect.getsource(ec))
    assert second_copies(cpu) == [] and second_copies(gpu) == []
    # the accepted stage is actually the one used: the CPU solver calls infiltration.column_step, the GPU solver the
    # baseline device column stage
    assert "column_step" in cpu["imported"] and "column_step" in cpu["called"]
    assert "prepared_column_step" in gpu["called"] and "column_step" not in gpu["defined"]


def test_the_ast_contract_ignores_explanatory_text_but_detects_definitions_calls_and_imports():
    explanatory = 'def f():\n    """the baseline column uses expm1 and a Smith capacity"""\n    return {"note": "expm1 Smith capacity"}\n'
    assert second_copies(executable_names(explanatory)) == []
    copied = "import numpy as np\ndef column_copy(x):\n    return np.expm1(x)\n"
    assert second_copies(executable_names(copied)) == ["column_copy", "expm1"]
    assert second_copies(executable_names("from m import smith_waterman\nclass GreenAmptLaw: pass\n")) == [
        "GreenAmptLaw", "smith_waterman"]


def test_donor_cells_by_cell_agree_with_the_receivers_in_the_legacy_slot_order():
    cs = build("valley:6x5", "explicit")
    donors = eh.donor_cells_by_cell(cs.graph)
    assert donors.shape == (4, 30) and donors.dtype == np.int64 and donors.min() >= -1
    receiver = np.asarray(cs.graph.receiver).reshape(-1)
    for cell in range(30):  # every donor really drains into its receiver, once, in the legacy slot order
        for slot in range(4):
            if donors[slot, cell] >= 0:
                assert receiver[donors[slot, cell]] == cell
    assert int(np.count_nonzero(donors >= 0)) == int(np.count_nonzero(receiver >= 0))


# --- CLI refusals (nothing is read, nothing is written) ----------------------------------------------------------------
def test_cli_refusals_happen_before_the_case_is_read_and_write_nothing(tmp_path, capsys, monkeypatch):
    out = tmp_path / "out"
    base = ["--case-dir", str(tmp_path / "missing_case"), "--output-dir", str(out)]
    assert ee.main([*base, "--solver", "explicit", "--limiter", "donor"]) == 1
    assert "local-inertial" in capsys.readouterr().err and not out.exists()
    assert ee.main([*base, "--solver", "local_inertial", "--cfl-max", "0.9"]) == 1
    assert "cfl_max" in capsys.readouterr().err
    assert ee.main([*base, "--solver", "explicit", "--snapshot-times-s", "a,b"]) == 1
    assert "snapshot" in capsys.readouterr().err
    from maple.core import backend

    monkeypatch.setattr(backend, "gpu_execution_available", lambda *a, **k: False)
    assert ee.main([*base, "--solver", "explicit", "--backend", "cupy"]) == 1
    assert "no fallback" in capsys.readouterr().err and not out.exists()
    out.mkdir()
    assert ee.main([*base, "--solver", "explicit"]) == 1
    assert "refusing to write into existing path" in capsys.readouterr().err and list(out.iterdir()) == []
    with pytest.raises(SystemExit):
        ee.main([*base, "--solver", "diffusive"])


def test_cli_choice_lists():
    assert eh.METHODS == ("explicit", "local_inertial") and eh.LIMITERS == ("off", "donor")
    assert eh.DEFAULT_CFL_MAX == 0.5 and eh.LIMITER_SAFETY == 1.0 - 16.0 * float(np.finfo(np.float64).eps)
