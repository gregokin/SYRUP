"""Controls, ordering, comparison and path rejection of `benchmarks/newton_cpu/compare_cases.py` on synthetic data and
mocks (no case, no timing, no Fortran)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("benchmarks/newton_cpu", "benchmarks/hydraulic_candidates", "benchmarks/rfid"):
    sys.path.insert(0, str(ROOT / sub))
pytest.importorskip("maple")
import compare_cases as cc


def ns(**kw):
    base = {"case": "rfid", "contenders": ",".join(cc.SYRUP_CONTENDERS), "end_s": 2700.0, "max_dt_s": 1.0,
            "report_every_s": 60.0, "rounds": 3, "bisection_iterations": 64, "newton_max_iterations": 50,
            "reference": None, "fortran_build_dir": None, "fortran_exe": None}
    base.update(kw)
    return argparse.Namespace(**base)


def test_valid_defaults_and_reference_choice():
    args = ns()
    assert cc.validate_args(args) == list(cc.SYRUP_CONTENDERS) and args.reference == "bisection_numba"
    args = ns(contenders="newton_numpy,newton_numba")
    cc.validate_args(args)
    assert args.reference == "newton_numba"
    args = ns(contenders="fortran_newton", fortran_exe=Path("x"))
    cc.validate_args(args)
    assert args.reference == "fortran_newton"


@pytest.mark.parametrize("changes,match", [
    ({"contenders": "bisection_numba,bisection_numba"}, "duplicate-free"),
    ({"contenders": "secant_numba"}, "duplicate-free"),
    ({"contenders": ""}, "duplicate-free"),
    ({"case": "plot1", "contenders": "fortran_newton", "fortran_exe": Path("x")}, "RFID adapter"),
    ({"contenders": "fortran_newton"}, "need --fortran-build-dir"),
    ({"contenders": "fortran_newton", "fortran_exe": Path("x"), "fortran_build_dir": Path("y")}, "not both"),
    ({"end_s": 0.0}, "--end-s"), ({"end_s": float("nan")}, "--end-s"), ({"max_dt_s": 1e-4}, "retry floor"),
    ({"end_s": 2700.5}, "integer multiple"), ({"max_dt_s": 0.3, "end_s": 3.0}, "integer multiple"),
    ({"report_every_s": 0.5}, "integer multiple"), ({"rounds": 0}, "--rounds"), ({"rounds": True}, "--rounds"),
    ({"bisection_iterations": 0}, "--bisection-iterations"), ({"bisection_iterations": 201}, "--bisection-iterations"),
    ({"newton_max_iterations": 1001}, "--newton-max-iterations"), ({"newton_max_iterations": True}, "--newton-max"),
    ({"reference": "newton_numpy", "contenders": "bisection_numba"}, "--reference"),
])
def test_invalid_controls_are_refused(changes, match):
    with pytest.raises(ValueError, match=match):
        cc.validate_args(ns(**changes))


def test_plot1_fortran_wording_does_not_claim_impossibility():
    with pytest.raises(ValueError) as info:
        cc.validate_args(ns(case="plot1", contenders="fortran_bisection", fortran_exe=Path("x")))
    text = str(info.value) + cc.__doc__
    assert "unsupported harness configuration" in text and "model 2" in text
    assert "requires the whole application" not in text and "heterogeneous captured realization" not in text


def test_round_orders():
    names = ["a", "b", "c"]
    assert cc.round_orders(names, 3, "balanced") == [names, names[::-1], names]
    assert cc.round_orders(names, 2, "forward") == [names, names]
    assert cc.round_orders(names, 2, "reverse") == [names[::-1]] * 2


def test_deviation_and_comparison_report_not_judge():
    a = np.array([1.0, 2.0, 3.0])
    same = cc.deviation(a, a.copy())
    assert same["bitwise_equal"] and same["max_abs"] == 0.0 and same["within_backend_bounds"]
    off = cc.deviation(a + np.array([0, 0, 1e-6]), a)
    assert not off["within_backend_bounds"] and not off["bitwise_equal"] and off["max_abs"] == pytest.approx(1e-6)
    last = {n: {"finals": {k: a.copy() for k in cc.FIELDS}, "export_m3": 2.0, "peak_outlet_m3_s": 1.0,
                "time_of_peak_s": 10.0, "series_q": [0.0, 1.0]} for n in ("x", "y")}
    last["x"]["export_m3"] = 2.2
    out = cc.compare_finals("x", last, "y")
    assert out["reference"] == "y" and out["export_m3"]["rel_diff"] == pytest.approx(0.1)
    assert out["outlet_hydrograph_rel_l2"] == 0.0 and set(out["fields"]) == set(cc.FIELDS)


def test_fortran_grid_maps_legacy_north_first_cells_to_south_first():
    cells = np.array([[3.0, 2.0, 0.5, 0.25, 0.125], [2.0, 3.0, 0.75, 0.0, 0.0]])  # legacy (i, j), ny = 2, nx = 2
    grid = cc.fortran_grid(cells, (2, 2))
    assert grid["depth_m"][0, 0] == 0.5 and grid["soil_water_m"][0, 0] == 0.25 and grid["discharge_m2_s"][0, 0] == 0.125
    assert grid["depth_m"][1, 1] == 0.75 and grid["depth_m"].sum() == 1.25


def test_timed_validated_separates_evolution_and_guard():
    calls = []

    class Guard:
        def validate(self, raw, end_s):
            calls.append(("validate", raw, end_s))
            return {"host": {"accepted": 3, "rejected": 0}}

    out = cc.timed_validated(lambda end, snaps: (calls.append(("run", end, snaps)) or "raw"), Guard(), 5.0)
    assert calls == [("run", 5.0, []), ("validate", "raw", 5.0)]
    assert out["evolution_wall_s"] >= 0.0 and out["guard_wall_s"] >= 0.0 and out["checked"]["host"]["accepted"] == 3


def test_newton_stats_aggregation():
    stats = cc.NewtonStats()
    step = type("S", (), {"route": type("R", (), {"root_stats": {
        "max_newton_iterations": 4, "total_newton_iterations": 30, "iterated_cells": 10,
        "bisection_safeguard_steps": 1, "fallback_cells": 0}})()})()
    stats.add(step)
    stats.add(step)
    rec = stats.record()
    assert rec["max_passes_of_any_cell"] == 4 and rec["mean_passes_per_iterated_cell_step"] == 3.0
    assert rec["iterated_cell_steps"] == 20 and rec["bisection_safeguard_steps"] == 2


def test_cli_rejects_existing_output_missing_args_and_lists_contenders(tmp_path, capsys):
    assert cc.main(["--list-contenders"]) == 0
    assert capsys.readouterr().out.split() == list(cc.ALL_CONTENDERS)
    with pytest.raises(SystemExit):
        cc.main(["--case", "rfid"])
    with pytest.raises(SystemExit):
        cc.main(["--case", "rfid", "--case-dir", str(tmp_path), "--output-dir", str(tmp_path)])  # exists: refused
    with pytest.raises(SystemExit):
        cc.main(["--case", "plot1", "--case-dir", str(tmp_path), "--output-dir", str(tmp_path / "new"),
                 "--contenders", "fortran_newton", "--fortran-exe", "x"])
    assert not (tmp_path / "new").exists()  # nothing created on bad controls


def test_every_selected_compiled_solver_gets_its_own_kernel_provenance_source():
    src = Path(cc.__file__).read_text()
    assert 'for n in ("bisection_numba", "newton_numba")' in src and "selected_root_solver" in src
    assert 'elif "newton_numba" in last' not in src
