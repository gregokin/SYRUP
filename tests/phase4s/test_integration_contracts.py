"""Phase 4S Task B: CPU-only contracts of the CUDA storm integration (no device, no Numba needed): the separate
`STORM_IMPLEMENTATIONS`, the single shared scheduler, the explicit water-only CLI refusals, the sediment refusal, lazy
optional imports and the absence of any fallback. Nothing here was run by its author (file-only tools); Codex records
results."""
from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys

import hydro_cases as hcs
import numpy as np
import pytest

pytest.importorskip("maple")

from maple_syrup import hydrology_cuda as hc
from maple_syrup import routing, storm, storm_experiment
from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule, rainfall_field
from maple_syrup.sediment_event import SedimentEventControl, SedimentEventError
from maple_syrup.storm import STORM_IMPLEMENTATIONS, StormControl, StormError

REFUSALS = (hc.CudaHydrologyPreparationError, hc.CudaUnavailableError)


def test_choice_lists_are_separate_and_the_unknown_message_lists_all_three():
    assert STORM_IMPLEMENTATIONS == ("array", "numba", "cuda") and "STORM_IMPLEMENTATIONS" in storm.__all__
    assert routing.IMPLEMENTATIONS == ("array", "numba")  # unchanged: every other module/CLI keeps this list
    for impl in STORM_IMPLEMENTATIONS:
        assert StormControl(implementation=impl).validated().implementation == impl
    with pytest.raises(StormError, match=r"\('array', 'numba', 'cuda'\)"):
        StormControl(implementation="fortran").validated()


def test_there_is_one_scheduler_and_no_cuda_driver_clone():
    assert importlib.util.find_spec("maple_syrup.storm_cuda") is None
    assert not hasattr(storm, "evolve_cuda") and not hasattr(hc, "evolve_cuda")
    source = inspect.getsource(storm.evolve)
    assert source.count("while t < target") == 1 and source.count("RoutingStepRejected") >= 1
    assert "cuda_step_with_packet" in source and "CudaStormAccumulator" in source
    assert "def evolve" not in inspect.getsource(hc)


def test_sediment_controls_refuse_cuda_before_any_physics_and_keep_the_cpu_choices():
    for impl in ("array", "numba"):
        assert SedimentEventControl(storm=StormControl(implementation=impl)).validated().storm.implementation == impl
    with pytest.raises(SedimentEventError, match="WATER-ONLY"):
        SedimentEventControl(storm=StormControl(implementation="cuda")).validated()
    from maple_syrup import benchmark_experiment, event_experiment, sediment_experiment

    for module in (benchmark_experiment, event_experiment, sediment_experiment):
        assert "cuda" not in getattr(module, "IMPLEMENTATIONS", ("array", "numba"))


def test_storm_cli_offers_cuda_only_as_an_explicit_water_choice():
    parser_actions = []

    def capture(self, *args, **kwargs):  # build the parser without running anything
        parser_actions.append(self)
        raise SystemExit(0)

    import argparse

    original = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = capture
    try:
        with pytest.raises(SystemExit):
            storm_experiment.main(["--case-dir", "x", "--output-dir", "y"])
    finally:
        argparse.ArgumentParser.parse_args = original
    action = next(a for a in parser_actions[0]._actions if "--implementation" in a.option_strings)
    assert tuple(action.choices) == STORM_IMPLEMENTATIONS and action.default == "numba"


def test_cuda_without_a_cupy_backend_is_refused_before_anything_is_read(tmp_path, capsys):
    out = tmp_path / "out"
    code = storm_experiment.main(["--case-dir", str(tmp_path / "missing_case"), "--output-dir", str(out),
                                  "--implementation", "cuda"])  # default backend is numpy
    assert code == 1 and "--backend cupy" in capsys.readouterr().err and not out.exists()


def test_cuda_with_no_device_is_refused_without_fallback(tmp_path, capsys, monkeypatch):
    from maple.core import backend

    monkeypatch.setattr(backend, "gpu_execution_available", lambda *a, **k: False)
    out = tmp_path / "out"
    code = storm_experiment.main(["--case-dir", str(tmp_path / "missing_case"), "--output-dir", str(out),
                                  "--implementation", "cuda", "--backend", "cupy"])
    err = capsys.readouterr().err
    assert code == 1 and "no fallback" in err and not out.exists()


def _numpy_storm_inputs():
    case = hcs.build_case(3, kind="valley", model="fixed_ksat")
    ny, nx = case.graph.shape
    field = rainfall_field(ny, nx, scale=np.where(case.graph.active, 1.0, 0.0))
    schedule = RainfallSchedule(edges_s=[0.0, 20.0], intensity_mm_per_h=[200.0],
                                provenance=RainfallProvenance(kind="constant"))
    return case, field, schedule


def test_evolve_cuda_with_a_numpy_graph_is_refused_before_time_advances(monkeypatch):
    case, field, schedule = _numpy_storm_inputs()
    monkeypatch.setattr(storm, "coupled_step", lambda *a, **k: pytest.fail("a CPU step ran"))
    before = [a.copy() for a in (case.state.depth_m, case.state.soil_water_m, case.state.discharge_m2_s)]
    with pytest.raises(REFUSALS):
        storm.evolve(case.graph, case.params, field, schedule, case.state, 20.0, StormControl(implementation="cuda"),
                     report_every_s=10.0)
    for a, b in zip(before, (case.state.depth_m, case.state.soil_water_m, case.state.discharge_m2_s), strict=True):
        np.testing.assert_array_equal(a, b)


def test_cuda_context_is_refused_for_other_implementations():
    case, field, schedule = _numpy_storm_inputs()
    for impl in ("array", "numba"):
        with pytest.raises(StormError, match="cuda_context"):
            storm.evolve(case.graph, case.params, field, schedule, case.state, 20.0,
                         StormControl(implementation=impl), report_every_s=10.0, cuda_context=object())


def test_a_foreign_cuda_context_object_is_refused_by_the_driver():
    case, field, schedule = _numpy_storm_inputs()
    with pytest.raises((StormError, *REFUSALS)):
        storm.evolve(case.graph, case.params, field, schedule, case.state, 20.0,
                     StormControl(implementation="cuda"), report_every_s=10.0, cuda_context=object())


def test_storm_state_validation_and_cuda_control_need_no_numba():
    code = (
        "import sys; sys.modules['numba'] = None; sys.modules['cupy'] = None; "
        "from maple_syrup import storm, hydrology_cuda; "
        "c = storm.StormControl(implementation='cuda').validated(); assert c.implementation == 'cuda'; "
        "assert 'maple_syrup.storm_cuda' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_new_public_names_and_kernel_inventory():
    for name in ("prepared_column_step", "cuda_step_with_packet", "CudaStormAccumulator", "prepare_cuda_hydrology",
                 "prepared_coupled_step"):
        assert name in hc.__all__ and callable(getattr(hc, name))
    src = hc.kernel_source()
    for kernel in ("maple_syrup_storm_accumulate", "maple_syrup_storm_report"):
        assert kernel in src and kernel in hc._KERNEL_NAMES
    with pytest.raises(REFUSALS):
        hc.prepared_column_step(object(), None, None, None, 1.0)


def test_context_scalar_signature_is_part_of_the_context_contract():
    import dataclasses

    assert "scalar_signature" in {f.name for f in dataclasses.fields(hc.CudaHydrologyContext)}
