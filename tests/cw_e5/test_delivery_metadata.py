"""CW-E5: delivery metadata checks (CPU, no install, no network, no storm).

Reads this tree's `pyproject.toml` and source text, and imports only the CLI
modules to read their `--help`. Nothing is installed or written.
"""
from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import io
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
USAGE = (ROOT / "docs" / "conservative_water_usage.md").read_text(encoding="utf-8")

EXISTING_SCRIPTS = {  # unchanged by CW-E5
    "maple-syrup-probe": "maple_syrup.probe:main",
    "maple-syrup-plot1-import": "maple_syrup.case_import:main",
    "maple-syrup-columns": "maple_syrup.column_experiment:main",
    "maple-syrup-storm": "maple_syrup.storm_experiment:main",
    "maple-syrup-sediment": "maple_syrup.sediment_experiment:main",
    "maple-syrup-event": "maple_syrup.event_experiment:main",
    "maple-syrup-benchmark": "maple_syrup.benchmark_experiment:main",
    "maple-syrup-legacy": "maple_syrup.legacy_experiment:main",
}
PHASE_TESTPATHS = ["tests/phase1", "tests/phase2", "tests/phase3", "tests/phase4", "tests/phase5", "tests/phase6",
                   "tests/phase7", "tests/phase7b", "tests/phase7f"]
CONSERVATIVE_TESTPATHS = ["tests/conservative_water", "tests/conservative_csr", "tests/conservative_csr_event",
                          "tests/conservative_pits"]


def _module_path(module: str) -> Path:
    return SRC.joinpath(*module.split(".")).with_suffix(".py")


def _top_level_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_project_identity_and_base_dependencies_unchanged():
    project = PYPROJECT["project"]
    assert project["name"] == "maple-syrup"
    assert project["version"] == "0.0.1.dev0"
    assert project["requires-python"] == ">=3.11"
    assert project["dependencies"] == ["numpy"]  # MAPLE deliberately not a package dependency
    extras = project["optional-dependencies"]
    assert extras["dev"] == ["pytest"]
    assert extras["numba"] == ["numba>=0.61"]
    assert set(extras) == {"dev", "numba", "dependency-delivery"}


def test_dependency_delivery_extra_declares_packaging():
    assert PYPROJECT["project"]["optional-dependencies"]["dependency-delivery"] == ["packaging>=23"]
    requirements = pytest.importorskip("packaging.requirements")  # parse only; nothing is installed
    req = requirements.Requirement("packaging>=23")
    assert req.name == "packaging" and str(req.specifier) == ">=23"


def test_bundle_helper_imports_exactly_the_declared_packaging_apis():
    tree = ast.parse((SRC / "maple_syrup" / "dependency_bundle.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "packaging":
            imported |= {(node.module, alias.name) for alias in node.names}
        elif isinstance(node, ast.Import):
            assert not any(a.name.split(".")[0] == "packaging" for a in node.names)
    assert imported == {("packaging.version", "InvalidVersion"), ("packaging.version", "Version"),
                        ("packaging.requirements", "Requirement")}


def test_console_scripts_existing_unchanged_plus_windowed():
    scripts = PYPROJECT["project"]["scripts"]
    assert {k: v for k, v in scripts.items() if k != "maple-syrup-windowed"} == EXISTING_SCRIPTS
    assert scripts["maple-syrup-windowed"] == "maple_syrup.windowed_experiment:main"
    assert len(scripts) == len(EXISTING_SCRIPTS) + 1


@pytest.mark.parametrize("target", sorted(set(EXISTING_SCRIPTS.values()) | {"maple_syrup.windowed_experiment:main"}))
def test_every_entrypoint_target_is_a_top_level_function_in_this_tree(target):
    module, _, attr = target.partition(":")
    path = _module_path(module)
    assert path.is_file(), path
    assert attr in _top_level_functions(path), target


def _help(module_name: str) -> str:
    module = importlib.import_module(module_name)
    assert Path(module.__file__).resolve() == _module_path(module_name), module.__file__  # this tree, not another
    assert callable(module.main)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as info:
        module.main(["--help"])
    assert info.value.code == 0
    return out.getvalue()


def _section(start: str, end: str) -> str:
    return USAGE[USAGE.index(start):USAGE.index(end)]


def test_windowed_entrypoint_is_callable_and_documented_options_exist():
    text = _help("maple_syrup.windowed_experiment")
    documented = set(re.findall(r"`(--[a-z][a-z0-9-]*)", _section("## Resident-window", "## Dense-bed")))
    documented.discard("--expected-maple-root")  # documented as ABSENT from this CLI (asserted below)
    assert {"--case-kind", "--store-dir", "--backend", "--phase-max-nnz", "--allow-maple-source-change"} <= documented
    missing = sorted(o for o in documented if o not in text)
    assert not missing, missing
    assert "--expected-maple-root" not in text  # the doc says the windowed CLI uses the environment variable


def test_dense_runner_documented_options_exist():
    text = _help("maple_syrup.event_experiment")
    documented = set(re.findall(r"`(--[a-z][a-z0-9-]*)", _section("## Dense-bed", "## Checkpoint, resume")))
    assert documented
    missing = sorted(o for o in documented if o not in text)
    assert not missing, missing


def test_default_legacy_transport_unchanged():
    legacy = importlib.import_module("maple_syrup.legacy_experiment")
    action = legacy.build_parser()._option_string_actions["--transport-scheme"]
    assert action.default == "legacy" and tuple(action.choices) == ("legacy",)
    tree = ast.parse((SRC / "maple_syrup" / "benchmark_experiment.py").read_text(encoding="utf-8"))
    defaults = [kw.value.value for node in ast.walk(tree)
                if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--transport-scheme"
                for kw in node.keywords if kw.arg == "default"]
    assert defaults == ["legacy", "legacy"]  # the scheme probe and the full parser


def test_default_testpaths_add_only_the_general_conservative_suites():
    paths = PYPROJECT["tool"]["pytest"]["ini_options"]["testpaths"]
    # The conservative suites precede phase2: its bare `conftest` must be the last one loaded (see pyproject).
    assert paths == CONSERVATIVE_TESTPATHS + PHASE_TESTPATHS
    bare = [p for p in paths if (ROOT / p / "conftest.py").is_file() and not (ROOT / p / "__init__.py").is_file()]
    assert bare[-1] == "tests/phase2", bare
    for rel in CONSERVATIVE_TESTPATHS:
        directory = ROOT / rel
        assert directory.is_dir(), rel
        for path in directory.rglob("*.py"):
            assert "/home/" not in path.read_text(encoding="utf-8"), path  # no frozen-snapshot oracle paths
    for historical in ("tests/cw_e2", "tests/cw_f2", "tests/cw_b46", "tests/cw_e5"):
        assert historical not in paths


def test_usage_doc_separates_legacy_default_and_keeps_open_items_open():
    assert "maple-syrup-windowed" in USAGE and "python -m maple_syrup.windowed_experiment" in USAGE
    assert "INITIAL case import only" in USAGE and "never** a restart-identity bypass" in USAGE
    assert "has not been installed" in USAGE
    for phrase in ("large Chastre GPU fit", "No result is claimed"):
        assert phrase in USAGE, phrase
