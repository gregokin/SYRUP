"""Phase 7f: the default run workflow is the Python/Numba legacy replay; conservative schemes stay explicit."""
from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maple")

from maple_syrup import benchmark_experiment as bench
from maple_syrup import legacy_experiment as le
from maple_syrup import legacy_transport as L
from maple_syrup import routing_numba

REPO = Path(__file__).resolve().parents[2]
BASELINE_COMMIT = "b350d36"  # accepted pre-promotion script; immutable, unlike HEAD
NUMBA = pytest.mark.skipif(not routing_numba.numba_available(), reason="Numba not installed; no claim made")
REFERENCES = [le.LEGACY_DEFAULT_RAINFALL, le.LEGACY_DEFAULT_LEDGER, le.LEGACY_DEFAULT_REFERENCE_RUN]
NEEDS_REFERENCES = pytest.mark.skipif(not all((REPO / p).exists() for p in REFERENCES),
                                      reason="actual Fortran reference artifacts not available")


@pytest.fixture
def no_run(monkeypatch):
    """Replace the legacy run with a recorder so only dispatch/refusal logic is exercised."""
    calls = []
    monkeypatch.setattr(le, "run", lambda args: calls.append(args))
    return calls


def test_generic_cli_defaults_to_legacy(monkeypatch):
    seen = []
    monkeypatch.setattr(le, "main", lambda argv=None: seen.append(list(argv)) or 0)
    monkeypatch.setattr(bench, "run_plot1_matched_benchmark", lambda *a, **k: pytest.fail("bins path must not run"))
    assert bench.main(["--case-dir", "c", "--output-dir", "o"]) == 0
    assert seen == [["--case-dir", "c", "--output-dir", "o"]]
    seen.clear()
    assert bench.main(["--transport-scheme=legacy", "--case-dir", "c", "--output-dir", "o"]) == 0 and len(seen) == 1


@pytest.mark.parametrize("scheme", ["characteristic", "upwind"])
def test_explicit_scheme_reaches_conservative_benchmark(monkeypatch, scheme):
    captured = {}

    def fake(case, out, **kwargs):
        captured.update(kwargs)
        raise ValueError("stop")

    monkeypatch.setattr(bench, "run_plot1_matched_benchmark", fake)
    monkeypatch.setattr(le, "main", lambda argv=None: pytest.fail("legacy must not run"))
    code = bench.main(["--case-dir", "c", "--output-dir", "o", "--applied-rainfall", "r.csv",
                       "--transport-scheme", scheme, "--phase-bins", "16"])
    assert code == 1 and captured["transport_scheme"] == scheme and captured["phase_bins"] == 16


def test_explicit_scheme_still_requires_applied_rainfall():
    with pytest.raises(SystemExit):
        bench.main(["--case-dir", "c", "--output-dir", "o", "--transport-scheme", "characteristic"])


def test_help_describes_both_models(capsys):
    with pytest.raises(SystemExit):
        bench.main(["--help"])
    text = capsys.readouterr().out
    assert "legacy" in text and "UNLIMITED supply" in text and "characteristic" in text and "Phase 7g" in text
    with pytest.raises(SystemExit):
        le.main(["--help"])
    text = capsys.readouterr().out
    assert "NOT a conservative complete-event" in text and "--transport-scheme characteristic" in text


@pytest.mark.parametrize("flags", [
    ["--phase-bins", "32"], ["--max-dt-s", "0.5"], ["--backend", "cupy"], ["--transport-courant", "1"],
    ["--sediment-courant-max", "0.5"], ["--max-transport-substeps", "8"], ["--transport-implementation", "numba"],
    ["--min-dt-s", "0.01"], ["--max-retries", "3"], ["--report-every-s", "5"], ["--transport-scheme", "upwind"],
])
def test_legacy_refuses_unsupported_controls(no_run, tmp_path, capsys, flags):
    out = tmp_path / "out"
    argv = ["--output-dir", str(out), *flags]
    if flags[0] in ("--transport-courant", "--transport-scheme"):
        with pytest.raises(SystemExit):  # unknown option / not a legacy choice: argparse refuses
            le.main(argv)
    else:
        assert le.main(argv) == 2
        assert "refused" in capsys.readouterr().err
    assert not no_run and not out.exists()


def test_legacy_accepts_supported_spellings(no_run, tmp_path):
    out = tmp_path / "out"
    assert le.main(["--case-dir", "c", "--output-dir", str(out), "--max-dt-s", "1", "--backend", "numpy",
                    "--report-every-s", "1", "--transport-scheme", "legacy"]) == (0 if L.KERNEL_IMPLEMENTATION == "numba"
                                                                                  and routing_numba.numba_available() else 2)


@NUMBA
def test_numba_default_requires_compiled_sediment_kernels(no_run, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(L, "KERNEL_IMPLEMENTATION", "python")
    assert le.main(["--output", str(tmp_path / "o")]) == 2
    assert "pure Python" in capsys.readouterr().err and not no_run


def test_numba_default_requires_compiled_water(no_run, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(le, "numba_available", lambda: False)
    assert le.main(["--output", str(tmp_path / "o")]) == 2
    assert "compiled water" in capsys.readouterr().err and not no_run


def test_array_diagnostic_is_explicit_and_not_blocked_by_missing_numba(no_run, tmp_path, monkeypatch):
    monkeypatch.setattr(le, "numba_available", lambda: False)
    monkeypatch.setattr(L, "KERNEL_IMPLEMENTATION", "python")
    assert le.main(["--output", str(tmp_path / "o"), "--implementation", "array"]) == 0
    assert no_run and no_run[0].implementation == "array"


def test_legacy_wrapper_and_entry_point_declared():
    text = (REPO / "pyproject.toml").read_text()
    assert 'maple-syrup-legacy = "maple_syrup.legacy_experiment:main"' in text
    assert 'maple-syrup-benchmark = "maple_syrup.benchmark_experiment:main"' in text
    wrapper = (REPO / "benchmarks/phase7e/run_legacy_benchmark.py").read_text()
    assert "from maple_syrup.legacy_experiment import main" in wrapper and len(wrapper.splitlines()) < 15


# Arrays that depend on the wet laws' exp/log/pow. The compiled default may differ from the NumPy reference by the
# libm-versus-NumPy rounding of those functions; the bound is declared here, before any result is inspected.
# Water arrays, times, labels and every non-float array stay exact in both paths.
PHYSICS_FLOAT_KEYS = ("ledger", "cumulative_detachment_kg", "cumulative_deposition_kg",
                      "cumulative_clipping_source_kg", "final_mobile_kg", "identity_residual_kg")
COMPILED_RTOL = 2.0e-11
COMPILED_ATOL = 1.0e-14


@NUMBA
@NEEDS_REFERENCES
@pytest.mark.parametrize("physics", ["array", "default"])
def test_short_cli_run_matches_original_script(plot1_case, tmp_path, monkeypatch, physics):
    """`array`: the NumPy-physics path reproduces the pre-promotion script's ledger arrays BITWISE.
    `default`: the compiled-physics default matches within the declared float tolerance on physics-dependent
    arrays and exactly on water, times and non-floats."""
    monkeypatch.chdir(REPO)
    new = tmp_path / "new"
    extra = ["--physics-implementation", "array"] if physics == "array" else []
    assert bench.main(["--case-dir", str(plot1_case), "--output-dir", str(new), "--end-s", "30",
                       "--allow-maple-source-change", *extra]) == 0
    shown = subprocess.run(["git", "show", f"{BASELINE_COMMIT}:benchmarks/phase7e/run_legacy_benchmark.py"], cwd=REPO,
                           capture_output=True, text=True, check=False)
    if shown.returncode != 0 or "def main():" not in shown.stdout:
        pytest.skip(f"accepted baseline commit {BASELINE_COMMIT} is not available in this git repository")
    original = shown.stdout
    script = tmp_path / "original.py"
    script.write_text(original)
    old = tmp_path / "old"
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(old), "--case", str(plot1_case), "--end-s", "30",
                                      "--allow-maple-source-change"])
    runpy.run_path(str(script), run_name="__main__")
    a, b = np.load(new / "legacy_ledger.npz"), np.load(old / "legacy_ledger.npz")
    assert sorted(a.files) == sorted(b.files)
    for key in a.files:
        if physics == "default" and key in PHYSICS_FLOAT_KEYS:
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape, key
            np.testing.assert_allclose(a[key], b[key], rtol=COMPILED_RTOL, atol=COMPILED_ATOL, err_msg=key)
        else:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    assert float(a["ledger"][:, :, 0].sum()) > 0.0
