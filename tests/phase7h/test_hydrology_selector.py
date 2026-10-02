"""Phase 7h: `--hydrology-implementation` selector, refusals (no silent fallback), missing Numba, and a short
end-to-end comparison of the prepared default with the reference hydrology on the actual Plot 1 case.
Nothing here was run by its author (file-only tools); Codex records actual results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from test_routing import chain_full, make_graph

pytest.importorskip("maple")

from maple_syrup import benchmark_experiment as bench
from maple_syrup import hydrology_numba as hn
from maple_syrup import legacy_experiment as le
from maple_syrup import legacy_transport as L
from maple_syrup import routing_numba
from maple_syrup.infiltration import column_parameters
from maple_syrup.routing import RoutingError
from maple_syrup.storm import StormControl, StormState, initial_state

REPO = Path(__file__).resolve().parents[2]
NUMBA = pytest.mark.skipif(not routing_numba.numba_available(), reason="Numba not installed; no claim made")
REFERENCES = [le.LEGACY_DEFAULT_RAINFALL, le.LEGACY_DEFAULT_LEDGER, le.LEGACY_DEFAULT_REFERENCE_RUN]
NEEDS_REFERENCES = pytest.mark.skipif(not all((REPO / p).exists() for p in REFERENCES),
                                      reason="actual Fortran reference artifacts not available")
WATER_FLOAT_KEYS = ("water_outlet_m3_s", "water_export_m3")
WATER_RTOL, WATER_ATOL = 2.0e-12, 1.0e-14
PHYSICS_RTOL, PHYSICS_ATOL = 2.0e-11, 1.0e-14  # the declared compiled-physics bound of Phase 7f, unchanged


@pytest.fixture
def no_run(monkeypatch):
    calls = []
    monkeypatch.setattr(le, "run", lambda args: calls.append(args))
    return calls


def parse(*argv):
    return le.build_parser().parse_args(["--output", "x", *argv])


def test_selector_defaults_follow_the_water_implementation():
    assert parse().hydrology_implementation is None and le.resolve_hydrology_implementation(parse()) == "prepared"
    assert le.resolve_hydrology_implementation(parse("--implementation", "array")) == "reference"
    assert le.resolve_hydrology_implementation(parse("--hydrology-implementation", "reference")) == "reference"
    assert le.resolve_hydrology_implementation(parse("--hydrology-implementation", "prepared")) == "prepared"
    with pytest.raises(SystemExit):
        parse("--hydrology-implementation", "cupy")


def test_require_compiled_explains_each_refusal(monkeypatch):
    monkeypatch.setattr(L, "KERNEL_IMPLEMENTATION", "numba")
    monkeypatch.setattr(le, "numba_available", lambda: True)
    assert le.require_compiled("numba") == []
    assert le.require_compiled("numba", None, "reference") == []
    assert le.require_compiled("array", "array", "reference") == []
    problems = le.require_compiled("array", "array", "prepared")  # explicit mismatch: refused, never downgraded
    assert len(problems) == 1 and "requires --implementation numba" in problems[0] and "no silent fallback" in problems[0]
    monkeypatch.setattr(le, "numba_available", lambda: False)
    problems = le.require_compiled("numba")
    assert any("prepared compiled hydrology" in p for p in problems) and any("compiled water" in p for p in problems)
    assert le.require_compiled("numba", None, "reference") != []  # the compiled sweep itself still needs Numba
    assert not any("prepared" in p for p in le.require_compiled("array"))  # array diagnostic needs nothing


def test_main_refuses_prepared_with_array_water_and_does_not_run(no_run, tmp_path, capsys):
    out = tmp_path / "o"
    assert le.main(["--output", str(out), "--implementation", "array", "--hydrology-implementation", "prepared"]) == 2
    err = capsys.readouterr().err
    assert "prepared requires --implementation numba" in err and not no_run and not out.exists()


def test_main_without_numba_refuses_the_prepared_default_but_not_the_array_diagnostic(no_run, tmp_path, monkeypatch,
                                                                                     capsys):
    monkeypatch.setattr(le, "numba_available", lambda: False)
    assert le.main(["--output", str(tmp_path / "a")]) == 2
    err = capsys.readouterr().err
    assert "prepared compiled hydrology" in err and not no_run
    monkeypatch.setattr(le.L, "KERNEL_IMPLEMENTATION", "python")
    assert le.main(["--output", str(tmp_path / "b"), "--implementation", "array"]) == 0
    assert no_run and le.resolve_hydrology_implementation(no_run[-1]) == "reference"


@NUMBA
def test_reference_hydrology_stays_selectable_on_compiled_water(no_run, tmp_path):
    code = le.main(["--output", str(tmp_path / "r"), "--hydrology-implementation", "reference"])
    assert code == (0 if L.KERNEL_IMPLEMENTATION == "numba" else 2)
    if no_run:
        assert no_run[-1].implementation == "numba" and no_run[-1].hydrology_implementation == "reference"


def _tiny():
    graph = make_graph(chain_full(3), ff=5.0)
    full = lambda v: np.full(graph.shape, float(v))
    params = column_parameters(model="fixed_ksat", ksat_m_per_s=full(1e-6), suction_m=full(0.01),
                               drainage_parameter=full(0.0), theta_sat=full(0.4), soil_thickness_m=full(0.3),
                               active_mask=graph.active.copy())
    return graph, params


def test_missing_numba_is_an_explicit_error_without_fallback(monkeypatch):
    graph, params = _tiny()
    monkeypatch.setattr(routing_numba, "_SWEEP", None)
    monkeypatch.setattr(hn, "_KERNELS", None)
    monkeypatch.setitem(sys.modules, "numba", None)  # `import numba` -> ImportError
    assert not routing_numba.numba_available() and not hn.numba_available()
    with pytest.raises(hn.HydrologyNumbaUnavailableError, match="Numba") as info:
        hn.prepare_hydrology(graph, params)
    assert isinstance(info.value, RoutingError) and isinstance(info.value, routing_numba.NumbaUnavailableError)


@NUMBA
def test_a_prepared_context_refuses_to_step_once_numba_disappears(monkeypatch):
    graph, params = _tiny()
    ctx = hn.prepare_hydrology(graph, params)
    state = initial_state(graph, np.full(graph.shape, 1e-3), np.full(graph.shape, 0.1))
    before = state.depth_m.copy()
    monkeypatch.setattr(hn, "_KERNELS", None)
    monkeypatch.setattr(hn, "numba_available", lambda: False)
    with pytest.raises(hn.HydrologyNumbaUnavailableError):
        hn.prepared_coupled_step(ctx, np.zeros(graph.shape), state, 1.0, StormControl(implementation="numba"))
    np.testing.assert_array_equal(state.depth_m, before)


def test_cupy_graphs_are_refused_without_a_transfer():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the device-refusal path is covered with a stand-in "
                    "namespace in test_hydrology_prepared.py")
    cp = backend.cupy_module()
    graph = make_graph(chain_full(3), xp=cp)
    params = _tiny()[1]
    with pytest.raises(hn.HydrologyPreparationError, match="host NumPy only"):
        hn.prepare_hydrology(graph, params)


@NUMBA
@NEEDS_REFERENCES
def test_short_cli_run_prepared_matches_reference_hydrology(plot1_case, tmp_path, monkeypatch):
    """30 one-second steps of the actual Plot 1 replay: the prepared default against `--hydrology-implementation
    reference` (same compiled physics and legacy transport). Water arrays within 2e-12/1e-14, physics-dependent
    arrays within the declared Phase 7f bound, times/labels/non-floats exact; provenance recorded."""
    monkeypatch.chdir(REPO)
    common = ["--case-dir", str(plot1_case), "--end-s", "30", "--allow-maple-source-change"]
    prepared, reference = tmp_path / "prepared", tmp_path / "reference"
    assert bench.main([*common, "--output-dir", str(prepared)]) == 0
    assert bench.main([*common, "--output-dir", str(reference), "--hydrology-implementation", "reference"]) == 0
    a, b = np.load(prepared / "legacy_ledger.npz"), np.load(reference / "legacy_ledger.npz")
    assert sorted(a.files) == sorted(b.files)
    for key in a.files:
        assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape, key
        if a[key].dtype.kind != "f" or key == "t_s":
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        elif key in WATER_FLOAT_KEYS:
            np.testing.assert_allclose(a[key], b[key], rtol=WATER_RTOL, atol=WATER_ATOL, err_msg=key)
        else:
            np.testing.assert_allclose(a[key], b[key], rtol=PHYSICS_RTOL, atol=PHYSICS_ATOL, err_msg=key)
    assert float(a["ledger"][:, :, 0].sum()) > 0.0
    sa = json.loads((prepared / "legacy_summary.json").read_text())
    sb = json.loads((reference / "legacy_summary.json").read_text())
    assert sa["hydrology_implementation"] == "prepared" and sb["hydrology_implementation"] == "reference"
    perf = sa["performance"]
    assert perf["hydrology_preparation_s"] > 0.0 and perf["hydrology_context"]["n_active"] > 0
    assert perf["hydrology_kernels"]["numba_options"]["fastmath"] is False
    assert perf["hydrology_kernels"]["compiled_in_process"] is True
    assert sb["performance"]["hydrology_context"] is None and sb["performance"]["hydrology_preparation_s"] == 0.0
    assert sa["regime_cell_class_steps"] == sb["regime_cell_class_steps"]
    assert sa["time_of_peak_outlet_flux_s"] == sb["time_of_peak_outlet_flux_s"]


def test_state_type_is_unchanged():
    """The prepared step returns the reference's StormState (not a new type), so downstream code is unchanged."""
    assert hn.StormState is StormState
