"""Timing harness pieces: argument validation, balanced order, the common legacy arrays, the Fortran input file, qualified
sample validation, and (with a working gfortran) short checked pilots of the ORIGINAL routines. No full timing storm runs here.
Nothing here was run by its author (file-only tools)."""
from __future__ import annotations

from types import SimpleNamespace

import fortran_timing as ft
import numpy as np
import pytest
import run_rfid_timing as rt
from rfid_helpers import needs_gfortran, needs_rfid

pytest.importorskip("maple")

from maple_syrup.rfid_case import rfid_inputs


def args(**kw):
    base = {"contenders": "legacy_numba_prepared,explicit_numba", "end_s": 2700.0, "max_dt_s": 1.0, "report_every_s": 60.0,
            "rounds": 3, "bisection_iterations": 64, "cfl_max": 0.5, "fortran_build_dir": None, "fortran_exe": None}
    base.update(kw)
    return SimpleNamespace(**base)


def test_argument_validation_refuses_unsafe_requests():
    assert rt.validate_args(args()) == ["legacy_numba_prepared", "explicit_numba"]
    bad_cases = (
        ({"contenders": "nope"}, "contenders"), ({"contenders": "explicit_numba,explicit_numba"}, "contenders"),
        ({"end_s": 2700.5}, "multiple"), ({"max_dt_s": 0.1, "end_s": 2700.0}, "REAL32"),
        ({"max_dt_s": 2.0**-12, "end_s": 1.0}, "retry floor"), ({"end_s": 2.0e6}, "exceeds the bound"),
        ({"report_every_s": 0.5}, "multiple"), ({"rounds": 0}, "rounds"),
        ({"bisection_iterations": 0}, "bisection"), ({"cfl_max": 0.9}, "cfl"),
        ({"contenders": "fortran_iroute2"}, "Fortran contenders need"),
        ({"contenders": "fortran_iroute2", "fortran_build_dir": "a", "fortran_exe": "b"}, "OR"),
    )
    for bad, match in bad_cases:
        with pytest.raises(ValueError, match=match):
            rt.validate_args(args(**bad))


def test_rounds_are_balanced_forward_reverse_forward():
    names = ["a", "b", "c"]
    assert rt.orders(names, 3) == [names, ["c", "b", "a"], names]
    assert rt.summarize([3.0, 1.0, 2.0]) == {"median_s": 2.0, "min_s": 1.0, "max_s": 3.0, "samples_s": [3.0, 1.0, 2.0]}


def test_request_validation_and_destination_safety(tmp_path):
    ft.validate_run_request(2700, 1.0, 60, 5)
    for bad, match in (((0, 1.0, 60, 5), "n_steps"), ((10, 1.0, 0, 5), "report_every_steps"), ((10, 0.1, 5, 5), "REAL"),
                       ((10, 2.0**-12, 5, 5), "retry floor"), ((10, 1.0, 5, 3), "iroute"),
                       ((ft.MAX_STEPS + 1, 1.0, 5, 5), "bound")):
        with pytest.raises(ValueError, match=match):
            ft.validate_run_request(*bad)
    assert ft.safe_destination(tmp_path / "new") == (tmp_path / "new").resolve()
    for victim in (ft.fr.MAHLERAN_ROOT / "src" / "x", ft.ROOT / "src" / "x", ft.ROOT / "benchmarks", ft.ROOT / "cases" / "rfid" / "y"):
        with pytest.raises(ValueError, match="refusing"):
            ft.safe_destination(victim)
    with pytest.raises(FileExistsError):
        ft.safe_destination(tmp_path)
    with pytest.raises(ValueError, match="bound case"):
        ft.safe_destination(tmp_path / "case" / "sub", {"bound case": tmp_path / "case"})


def test_last_ok_sample_skips_a_failed_last_sample():
    """Regression: the comparison reference was `samples[-1]` even when it failed."""
    samples = [{"status": "ok", "export_m3": 1.0}, {"status": "ok", "export_m3": 2.0}, {"status": "failed", "reason": "x"}]
    assert rt.last_ok_sample(samples)["export_m3"] == 2.0
    with pytest.raises(ValueError, match="no successful"):
        rt.last_ok_sample([{"status": "failed"}])


@needs_rfid
def test_common_legacy_arrays_encode_the_matched_input(rfid_case):
    inputs = rfid_inputs(rfid_case, "numpy", with_geometry=False)
    a = ft.common_arrays(inputs)
    assert a["aspect"].shape == a["rmask"].shape == (106, 60) and a["n_active"] == 5697
    active = a["active"].astype(bool)
    assert int(active.sum()) == 5697 and not active[0].any() and not active[-1].any()  # the ring is never computed
    assert np.all(a["rmask"][~active] == -9999.0) and np.all(a["rmask"][active] > 0.0)  # ring + inactive cells: matched negative mask
    assert int(a["outlet"].sum()) == 1
    order = a["order"]
    assert order.shape == (5697, 3) and np.unique(order[:, :2], axis=0).shape[0] == 5697
    assert np.all(active[order[:, 0] - 1, order[:, 1] - 1])  # legacy 1-based (i, j) names active cells only
    assert np.all(np.diff(order[:, 2]) >= 0)  # upstream first
    np.testing.assert_allclose(a["ksat"][active], 0.028867846354842186, rtol=1e-15)
    np.testing.assert_allclose(a["psi"][active], 23.6, rtol=1e-15)
    np.testing.assert_allclose(a["cum_inf"][active], 0.84, rtol=1e-12)
    np.testing.assert_allclose(a["stmax"][active], 75.6, rtol=1e-12)
    assert np.all(a["cum_inf"][~active] == 0.0)
    pit = np.asarray(inputs.graph.pit_storage)
    assert int(pit.sum()) == 26 and np.all(a["slope"][::-1][1:-1, 1:-1][pit] == 0.0)
    ij = ft.expected_active_ij(a)
    assert ij.shape == (5697, 2) and np.all(active[ij[:, 0] - 1, ij[:, 1] - 1])


@needs_rfid
def test_input_file_layout_refusals_and_no_overwrite(rfid_case, tmp_path):
    inputs = rfid_inputs(rfid_case, "numpy", with_geometry=False)
    a = ft.common_arrays(inputs)
    rates = np.full(10, 0.03836299851536751)
    path = tmp_path / "in.dat"
    digest = ft.write_input(path, a, dt_s=1.0, rates_mm_s=rates, iroute=2, report_every_steps=5)
    lines = path.read_text().splitlines()
    assert len(lines) == 1 + 5697 + 106 * 60 + 10 and len(digest) == 64
    assert lines[0].split()[:4] == ["106", "60", "5697", "10"] and lines[0].split()[6:] == ["2", "5"]
    assert len(lines[1 + 5697].split()) == 15
    with pytest.raises(FileExistsError):
        ft.write_input(path, a, dt_s=1.0, rates_mm_s=rates, iroute=2, report_every_steps=5)
    for kw, match in (({"iroute": 3}, "iroute"), ({"dt_s": 0.1}, "REAL"), ({"rates_mm_s": np.array([-1.0])}, "non-negative"),
                      ({"rates_mm_s": np.array([np.nan])}, "non-negative"), ({"report_every_steps": 0}, "report_every_steps")):
        base = {"dt_s": 1.0, "rates_mm_s": rates, "iroute": 2, "report_every_steps": 5}
        base.update(kw)
        with pytest.raises(ValueError, match=match):
            ft.write_input(tmp_path / f"bad_{match}.dat", a, **base)


def _expected(inputs, arrays, n, dt, cadence, iroute):
    rain = inputs.schedule.depth_m(0.0, n * dt) * float(inputs.host["rainfall_scale"].sum()) * inputs.area
    return {"n_steps": n, "dt_s": dt, "report_every_steps": cadence, "iroute": iroute, "rain_expected_m3": rain,
            "active_ij": ft.expected_active_ij(arrays)}


def _run(rfid_case, base, iroute, n, cadence, build=None):
    inputs = rfid_inputs(rfid_case, "numpy", with_geometry=False)
    arrays = ft.common_arrays(inputs)
    rates = np.array([inputs.schedule.rate_after_m_per_s(k * 1.0) * 1000.0 for k in range(n)])
    path = base / f"in_{iroute}_{n}_{cadence}.dat"
    ft.write_input(path, arrays, dt_s=1.0, rates_mm_s=rates, iroute=iroute, report_every_steps=cadence)
    expected = _expected(inputs, arrays, n, 1.0, cadence, iroute)
    out = ft.run_once(base / "build" / "rfid_water_driver", path, base / f"run_{iroute}_{n}_{cadence}", expected=expected,
                      build_record=build)
    return inputs, expected, out


@pytest.fixture(scope="module")
def checked_build(tmp_path_factory):
    base = tmp_path_factory.mktemp("fortran_checked")
    return base, ft.build(base / "build", variant="checked")


@needs_rfid
@needs_gfortran
@pytest.mark.parametrize("iroute", [2, 5])
def test_checked_original_routine_pilot_qualifies_and_pins_sources(rfid_case, checked_build, iroute):
    base, build = checked_build
    assert build["reference_tree_unchanged"] and build["flags"].count("-fcheck=all") == 1
    for rel in ft.SOURCES:  # every linked original is the hash-pinned one
        assert build["source_sha256"][rel] == ft.fr.REFERENCE_SOURCES[rel]
    with pytest.raises(FileExistsError):
        ft.build(base / "build", variant="checked")
    inputs, expected, out = _run(rfid_case, base, iroute, 30, 10, build)
    assert out["status"] == "complete", out
    assert out["loop_s"] > 0.0 and out["history"].shape == (3, 7) and out["history"][-1, 0] == 30.0
    assert out["final_cells"].shape == (5697, 5) and np.all(out["final_cells"][:, 2:] >= 0.0)
    bud = ft.budget(out, inputs, expected["rain_expected_m3"])
    assert bud["time_s"] == 30.0 and np.isfinite(bud["residual_m3"]) and "not claimed conservative" in bud["note"]


@needs_rfid
@needs_gfortran
@pytest.mark.parametrize("n, cadence, rows", [(90, 60, 2), (30, 60, 1), (120, 60, 2), (7, 7, 1)])
def test_final_history_row_is_always_the_requested_end(rfid_case, checked_build, n, cadence, rows):
    """Regression: floor(n / cadence) rows dropped the final state (end 90 / report 60 ended the history at 60 s)."""
    base, build = checked_build
    _, _, out = _run(rfid_case, base, 5, n, cadence, build)
    assert out["status"] == "complete", out
    assert out["history"].shape[0] == rows and out["history"][-1, 0] == float(n)
    assert len(set(out["history"][:, 0])) == rows  # no duplicate time


@needs_rfid
@needs_gfortran
def test_unqualified_samples_are_failures_not_times(rfid_case, checked_build, tmp_path):
    base, build = checked_build
    inputs = rfid_inputs(rfid_case, "numpy", with_geometry=False)
    arrays = ft.common_arrays(inputs)
    # (a) truncated input: a legacy-style early end must not yield a timing
    path = tmp_path / "trunc.dat"
    ft.write_input(path, arrays, dt_s=1.0, rates_mm_s=np.full(5, 0.03), iroute=5, report_every_steps=5)
    lines = path.read_text().splitlines()
    path.unlink()
    path.write_text("\n".join(lines[:-2]) + "\n")
    out = ft.run_once(base / "build" / "rfid_water_driver", path, tmp_path / "run_a", build_record=build,
                      expected=_expected(inputs, arrays, 5, 1.0, 5, 5))
    assert out["status"] == "failed" and "loop_s" not in out
    # (b) a forcing integral that differs from the common one is refused even though the run itself completed
    path_b = tmp_path / "ok.dat"
    common = np.array([inputs.schedule.rate_after_m_per_s(k * 1.0) * 1000.0 for k in range(5)])
    ft.write_input(path_b, arrays, dt_s=1.0, rates_mm_s=common, iroute=5, report_every_steps=5)
    wrong = _expected(inputs, arrays, 5, 1.0, 5, 5)
    wrong["rain_expected_m3"] *= 1.01
    out = ft.run_once(base / "build" / "rfid_water_driver", path_b, tmp_path / "run_b", expected=wrong, build_record=build)
    assert out["status"] == "failed" and "forcing integral" in out["reason"]
    # (c) a wrong active-cell expectation (coverage) is refused
    bad_ij = _expected(inputs, arrays, 5, 1.0, 5, 5)
    bad_ij["active_ij"] = bad_ij["active_ij"][:-1]
    out = ft.run_once(base / "build" / "rfid_water_driver", path_b, tmp_path / "run_c", expected=bad_ij, build_record=build)
    assert out["status"] == "failed" and "final cells" in out["reason"]
    # (d) a stale / mismatched build record refuses to run at all
    stale = dict(build, executable_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="differ from the build record"):
        ft.run_once(base / "build" / "rfid_water_driver", path_b, tmp_path / "run_d", build_record=stale,
                    expected=_expected(inputs, arrays, 5, 1.0, 5, 5))
