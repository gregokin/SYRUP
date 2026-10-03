"""Contracts of the compiled CPU candidates that need NO Numba: lazy imports, an explicit error and no fallback when Numba is
missing, the (backend, implementation) resolution table and the CLI refusals (all BEFORE the case is read), the AST contract
(no second column law, the accepted prepared column is used), the harness contender names. Nothing here was run by its author
(file-only tools); Codex records results.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from cand_cases import build

pytest.importorskip("maple")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "hydraulic_candidates"))

import compare_plot1 as cmp

from maple_syrup import experimental_experiment as ee
from maple_syrup import experimental_hydrology as eh
from maple_syrup import experimental_numba as en
from maple_syrup import routing_numba


# --- lazy imports ------------------------------------------------------------------------------------------------------------
def test_the_module_imports_without_numba_and_without_cupy_and_compiles_nothing_at_import():
    code = ("import sys; sys.modules['numba'] = None; sys.modules['cupy'] = None; "
            "import maple_syrup.experimental_numba as en, maple_syrup.experimental_hydrology, maple_syrup.experimental_storm, "
            "maple_syrup.experimental_experiment; assert en._KERNELS is None; "
            "assert en.kernel_provenance()['compiled_in_process'] is False")
    subprocess.run([sys.executable, "-c", code], check=True)
    code = ("import sys; import maple_syrup.experimental_numba, maple_syrup.experimental_cuda, "
            "maple_syrup.experimental_hydrology; assert 'cupy' not in sys.modules")  # importing never loads CuPy
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_numpy_reference_and_cuda_modules_do_not_import_the_compiled_module():
    code = ("import sys; import maple_syrup.experimental_hydrology, maple_syrup.experimental_cuda, "
            "maple_syrup.experimental_storm; assert 'maple_syrup.experimental_numba' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_missing_numba_is_an_explicit_error_with_no_fallback_and_the_reference_still_works(monkeypatch):
    cs = build("valley:6x5", "explicit", seed=1)
    monkeypatch.setattr(en, "numba_available", lambda: False)
    monkeypatch.setattr(routing_numba, "numba_available", lambda: False)
    monkeypatch.setattr(eh.CpuHydraulicSolver, "step", lambda *a, **k: pytest.fail("the reference was used as a fallback"))
    with pytest.raises(en.ExperimentalNumbaUnavailableError, match="no fallback") as info:
        en.NumbaHydraulicSolver("explicit", cs.graph, cs.params)
    assert isinstance(info.value, en.NumbaUnavailableError) and isinstance(info.value, eh.ExperimentalHydrologyError)
    en.reset_compiled()
    with pytest.raises(en.ExperimentalNumbaUnavailableError, match="no fallback"):
        en._kernels()
    monkeypatch.undo()  # the reference itself never needed Numba
    out = cs.solver.step(np.zeros(cs.shape), cs.state, 0.05)
    assert out.implementation == "numpy"


# --- (backend, implementation) resolution ------------------------------------------------------------------------------------
@pytest.mark.parametrize(("backend", "implementation", "expected"), [
    (None, None, ("numpy", "numpy")), ("numpy", None, ("numpy", "numpy")), ("numpy", "numpy", ("numpy", "numpy")),
    ("numpy", "numba", ("numpy", "numba")), (None, "numba", ("numpy", "numba")), ("cupy", None, ("cupy", "cuda")),
    ("cupy", "cuda", ("cupy", "cuda")), (None, "cuda", ("cupy", "cuda")),
])
def test_valid_pairs_resolve_with_the_historical_defaults(backend, implementation, expected):
    assert ee.resolve_implementation(backend, implementation) == expected


@pytest.mark.parametrize(("backend", "implementation"), [
    ("cupy", "numba"), ("cupy", "numpy"), ("numpy", "cuda"), ("numpy", "bogus"), ("bogus", None), ("bogus", "numba"), ("", None),
])
def test_invalid_pairs_are_refused_without_substitution(backend, implementation):
    with pytest.raises(ee.StormError):
        ee.resolve_implementation(backend, implementation)


def fail(*_a, **_k):
    pytest.fail("the case was read (or a run began) before the refusal")


@pytest.mark.parametrize("flags", [["--backend", "cupy", "--implementation", "numba"],
                                   ["--backend", "numpy", "--implementation", "cuda"],
                                   ["--backend", "cupy", "--implementation", "numpy"]])
def test_cli_refuses_invalid_combinations_before_the_case_is_read(flags, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ee, "verify_plot1_case", fail)
    out = tmp_path / "out"
    code = ee.main(["--case-dir", str(tmp_path / "missing"), "--output-dir", str(out), "--solver", "explicit", *flags])
    assert code == 1 and not out.exists()
    assert "implementation" in capsys.readouterr().err


def test_cli_refuses_numba_when_numba_is_missing_before_the_case_is_read(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ee, "verify_plot1_case", fail)
    monkeypatch.setattr(routing_numba, "numba_available", lambda: False)
    out = tmp_path / "out"
    code = ee.main(["--case-dir", str(tmp_path / "missing"), "--output-dir", str(out), "--solver", "local_inertial",
                    "--implementation", "numba"])
    assert code == 1 and not out.exists()
    err = capsys.readouterr().err
    assert "Numba" in err and "no fallback" in err


def test_cli_rejects_an_unknown_implementation_name(tmp_path):
    with pytest.raises(SystemExit):
        ee.main(["--case-dir", str(tmp_path), "--output-dir", str(tmp_path / "o"), "--solver", "explicit",
                 "--implementation", "fortran"])


def test_the_default_cli_selection_is_unchanged(tmp_path, monkeypatch, capsys):
    """Without --implementation the historical selection holds: --backend numpy is the NumPy reference, never Numba."""
    resolved = []
    real = ee.resolve_implementation

    def spy_resolve(backend, implementation):
        out = real(backend, implementation)
        resolved.append(((backend, implementation), out))
        return out

    def stop(*args, **kwargs):
        raise ee.StormError("stop before reading the case")

    monkeypatch.setattr(ee, "resolve_implementation", spy_resolve)
    monkeypatch.setattr(ee, "verify_plot1_case", stop)
    ee.main(["--case-dir", str(tmp_path / "c"), "--output-dir", str(tmp_path / "o"), "--solver", "explicit"])
    assert "stop before reading the case" in capsys.readouterr().err
    assert resolved == [((None, None), ("numpy", "numpy"))]  # nothing requested -> the NumPy reference, never Numba


def test_the_api_defaults_match_the_cli_omitted_values_resolve_and_cuda_alone_implies_cupy(tmp_path, monkeypatch):
    """`run_plot1_experiment` takes `backend=None, implementation=None` like the CLI: both omitted -> the NumPy reference;
    `implementation="cuda"` alone -> cupy (refused without a device, no fallback); explicit numpy + cuda is still refused. All of
    it is decided BEFORE the case is read, so no live case is needed."""
    import inspect

    from maple.core import backend as mb

    assert inspect.signature(ee.run_plot1_experiment).parameters["backend"].default is None
    assert inspect.signature(ee.run_plot1_experiment).parameters["implementation"].default is None
    monkeypatch.setattr(ee, "verify_plot1_case", fail)
    monkeypatch.setattr(mb, "gpu_execution_available", lambda *a, **k: False)
    out = tmp_path / "out"
    with pytest.raises(ee.StormError, match="no fallback") as info:  # backend omitted + cuda -> cupy -> no device
        ee.run_plot1_experiment(tmp_path / "missing", out, solver="explicit", implementation="cuda")
    assert "--backend cupy" in str(info.value) and not out.exists()
    with pytest.raises(ee.StormError, match="needs backend cupy"):  # explicit numpy + cuda stays refused
        ee.run_plot1_experiment(tmp_path / "missing", out, solver="explicit", backend="numpy", implementation="cuda")
    with pytest.raises(ee.StormError, match="needs backend numpy"):
        ee.run_plot1_experiment(tmp_path / "missing", out, solver="explicit", backend="cupy", implementation="numba")
    resolved = []
    real = ee.resolve_implementation

    def spy(backend, implementation):
        result = real(backend, implementation)
        resolved.append(result)
        return result

    class Stop(RuntimeError):
        pass

    def stop(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(ee, "resolve_implementation", spy)
    monkeypatch.setattr(ee, "verify_plot1_case", stop)
    with pytest.raises(Stop):  # both omitted: resolved to the NumPy reference, then the (stubbed) case read is reached
        ee.run_plot1_experiment(tmp_path / "missing", tmp_path / "out2", solver="explicit")
    assert resolved == [("numpy", "numpy")]


# --- AST contract: no second column law, the accepted prepared column is used -------------------------------------------------
def executable_names(path: Path) -> dict:
    defined, called, imported = set(), set(), set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Call):
            func = node.func
            called.add(func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else "")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imported.update(alias.name for alias in node.names)
    return {"defined": defined, "called": called, "imported": imported}


def test_the_compiled_module_defines_no_second_infiltration_law_and_uses_the_prepared_column():
    names = executable_names(Path(en.__file__))
    law = re.compile(r"expm1|smith|capacity|hawkins|green_?ampt", re.IGNORECASE)
    assert not [n for n in names["defined"] if law.search(n) or n.lower().startswith("column")]
    assert not [n for n in names["called"] | names["imported"] if law.search(n)]
    assert "prepared_column_step" in names["called"] and "prepare_hydrology" in names["called"]
    assert "column_step" not in names["called"]  # not even the NumPy column: the accepted COMPILED stage is reused
    # Numba itself is imported lazily inside a function only, never at module level
    top_level = [n for n in ast.parse(Path(en.__file__).read_text()).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    top_modules = {alias.name for n in top_level if isinstance(n, ast.Import) for alias in n.names}
    top_modules |= {n.module for n in top_level if isinstance(n, ast.ImportFrom)}
    assert not [m for m in top_modules if m == "numba" or m.startswith("numba.") or m in ("cupy", "llvmlite")]


def test_the_lateral_kernels_use_no_fastmath_and_no_parallel_options():
    options = en.kernel_provenance()["numba_options"]
    assert options["fastmath"] is False and options["parallel"] is False and options["error_model"] == "numpy"
    source = Path(en.__file__).read_text()
    assert "fastmath=False" in source and "prange" not in source.replace("no prange", "")


# --- harness names --------------------------------------------------------------------------------------------------------
def test_the_harness_accepts_the_numba_contenders_by_name_and_keeps_the_default_list():
    assert cmp.CONTENDERS == ("legacy_numba_prepared", "legacy_cuda", "explicit_numpy", "explicit_cuda",
                              "local_inertial_numpy", "local_inertial_cuda")
    assert cmp.NUMBA_CONTENDERS == ("explicit_numba", "local_inertial_numba")
    assert set(cmp.ALL_CONTENDERS) == set(cmp.CONTENDERS) | set(cmp.NUMBA_CONTENDERS)

    class Args:
        contenders = "explicit_numpy,explicit_numba,local_inertial_numpy,local_inertial_numba"
        dts, end_s, report_every_s, snapshot_times_s = "1", 10.0, 5.0, "5"
        warmup_s, repeats, cfl_max = 1.0, 1, 0.5

    names, dts, snapshots = cmp.validate_controls(Args)
    assert names == ["explicit_numpy", "explicit_numba", "local_inertial_numpy", "local_inertial_numba"]
    assert dts == [1.0] and snapshots == [5.0]
    Args.contenders = "explicit_numba,explicit_numba"
    with pytest.raises(ValueError):
        cmp.validate_controls(Args)
    Args.contenders = "explicit_fortran"
    with pytest.raises(ValueError):
        cmp.validate_controls(Args)
