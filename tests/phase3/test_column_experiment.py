"""Phase 3b Plot 1 column diagnostic.

The integration tests generate a fresh Plot 1 case with the Phase 2
importer (so the binding matches the MAPLE in use), run the no-routing
column diagnostic, and check it against hand values (9.652 mm over
300 m^2 = 2.8956 m^3; 0.25 x 0.3 m x 300 m^2 = 22.5 m^3 initial soil water),
an independent millimetre transcription of the legacy column update on
selected cells, and the reloaded MAPLE sediment state.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_infiltration import legacy_infilt

from maple_syrup.column_experiment import (
    FINAL_NAME,
    SUMMARY_NAME,
    ColumnExperimentError,
    _substeps,
    main,
    plot1_parameters,
    run_plot1_columns,
)
from maple_syrup.rainfall import constant_rainfall, parse_legacy_rainfall_text

REPO = Path(__file__).resolve().parents[2]
RECIPE_PATH = REPO / "cases" / "plot1" / "recipe.yaml"
MAHLERAN_ROOT = Path(os.environ.get("MAPLE_SYRUP_MAHLERAN_ROOT", "/home/okin/MAHLERAN"))
PLOT1_DIR = MAHLERAN_ROOT / "Input" / "input_p1"


# --- unit: step plan and parameter source checks -------------------------------------
def test_substeps_split_at_every_knot_and_max_dt():
    s = parse_legacy_rainfall_text("00:00:00\n00:00:10 3.6\n00:00:25 0\n00:00:26 7.2\n00:01:40 36\n")
    plan = _substeps(s, 4.0)
    assert [n for _, n, _ in plan] == [3, 4, 1, 19]
    for (dt, n, rate), span, expected_rate in zip(plan, (10, 15, 1, 74), s.rate_m_per_s):
        assert dt <= 4.0 and dt * n == pytest.approx(span, rel=1e-15) and rate == expected_rate
    lead = _substeps(constant_rainfall(5.0, 10.0, 36.0), 10.0)
    assert lead[0] == (5.0, 1, 0.0)  # zero-rain lead-in before the first knot
    assert lead[1][:2] == (5.0, 1) and lead[1][2] == pytest.approx(1e-5, rel=1e-15)


def _fake_source():
    settings = {
        "infiltration_model": 2, "infiltration_parameter_type": 2, "rain_type": 2, "number_of_surface_types": 1,
        "use_flags": {"use_final_infiltration_map": False, "use_suction_map": False, "use_drainage_map": False,
                      "use_initial_soil_moisture_map": False, "use_saturated_soil_moisture_map": True},
        "distributions": {"wettingFrontSuctionDistribution": "deterministic",
                          "drainageParameterDistribution": "deterministic",
                          "initialSoilMoistureDistribution": "deterministic",
                          "finalInfiltrationRateDistribution": "normal"},
        "by_surface_type": {k: {"type_1": v} for k, v in {
            "final_infiltration_rate_mean": "0.00025", "final_infiltration_std_dev": "0.001",
            "wetting_front_suction_mean": "46.6", "drainage_parameter_mean": "0.05",
            "initial_soil_moisture_mean": "0.25", "soil_thickness": "0.3"}.items()},
    }
    fields = {"saturated_soil_moisture": np.full((2, 3), 0.39), "pavement_cover_fraction": np.zeros((2, 3)),
              "rainfall_scaling": np.ones((2, 3)), "surface_type_resolved": np.ones((2, 3), dtype=np.int16)}
    return {"legacy_options": {"settings": settings}}, fields


def test_plot1_parameters_from_settings():
    report, fields = _fake_source()
    inputs, record = plot1_parameters(report, fields)
    assert inputs["ksat_m_per_s"][0, 0] == pytest.approx(2.5e-7, rel=1e-15)
    assert inputs["suction_m"][0, 0] == pytest.approx(0.0466, rel=1e-15)
    assert inputs["initial_theta"][0, 0] == 0.25 and inputs["soil_thickness_m"][0, 0] == 0.3
    assert "not sampled" in record["ksat_m_per_s"]["decision"]


@pytest.mark.parametrize(
    "path, value",
    [
        (("infiltration_model",), 1),
        (("infiltration_parameter_type",), 1),
        (("use_flags", "use_suction_map"), True),
        (("use_flags", "use_saturated_soil_moisture_map"), False),
        (("distributions", "wettingFrontSuctionDistribution"), "normal"),
    ],
)
def test_plot1_parameters_refuse_other_configurations(path, value):
    report, fields = _fake_source()
    target = report["legacy_options"]["settings"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ColumnExperimentError):
        plot1_parameters(report, fields)


def test_plot1_parameters_refuse_other_surface_types():
    report, fields = _fake_source()
    fields["surface_type_resolved"][0, 0] = 2
    with pytest.raises(ColumnExperimentError, match="surface types"):
        plot1_parameters(report, fields)


# --- integration ---------------------------------------------------------------------
def _generate(tmp_path_factory, name):
    pytest.importorskip("maple")
    if not (MAHLERAN_ROOT / "mahleran_input.xml").is_file() or not PLOT1_DIR.is_dir():
        pytest.skip(f"MAHLERAN reference not available at {MAHLERAN_ROOT}")
    from maple_syrup.case_import import generate_plot1_case, load_recipe

    out = tmp_path_factory.mktemp(name) / "case"
    generate_plot1_case(load_recipe(RECIPE_PATH, mahleran_root=MAHLERAN_ROOT), out)
    return out


@pytest.fixture(scope="module")
def plot1_case(tmp_path_factory):
    return _generate(tmp_path_factory, "plot1_columns_case")


@pytest.fixture(scope="module")
def plot1_run(plot1_case, tmp_path_factory):
    out = tmp_path_factory.mktemp("plot1_columns_run") / "columns"
    return run_plot1_columns(plot1_case, out, max_dt_s=1.0), out


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _interior_maple(path):
    """Independent legacy grid read: interior, MAPLE order (row 0 = south)."""
    lines = Path(path).read_bytes().splitlines()
    nrows = int(dict(line.decode().split() for line in lines[:6])["nrows"])
    grid = np.array([[float(t) for t in line.split()] for line in lines[6:6 + nrows]])
    return grid[1:-1, 1:-1][::-1]


def _plot1_intensities_mm_h():
    records = (PLOT1_DIR / "p1_01_08_06.dat").read_text().split("\n")[1:]
    values = [float(line.split()[1]) for line in records if line.strip()]
    assert len(values) == 27  # one-minute records, 18:00 -> 18:27
    return values


def test_plot1_budget_and_rainfall(plot1_run):
    run, _ = plot1_run
    s = run.summary
    b = s["budget"]
    assert sum(_plot1_intensities_mm_h()) / 60.0 == pytest.approx(9.652, rel=1e-12)
    assert s["domain"]["area_m2"] == 300.0
    assert b["rain_m3"] == pytest.approx(2.8956, rel=1e-12)
    assert b["soil_initial_m3"] == pytest.approx(22.5, rel=1e-12) and b["surface_initial_m3"] == 0.0
    total = b["surface_final_m3"] + b["soil_final_m3"] + b["drainage_m3"]
    assert total == pytest.approx(22.5 + 2.8956, rel=1e-12)
    assert b["surface_final_m3"] == pytest.approx(2.8956 - b["net_infiltration_m3"], rel=1e-9)
    for key in ("water_residual_m3", "surface_residual_m3", "soil_residual_m3"):
        assert abs(b[key]) <= b["tolerance_m3"] < 1e-8
    assert abs(b["rainfall_integral_residual_m3"]) <= b["rainfall_tolerance_m3"]
    assert 0.0 < b["surface_final_m3"] < b["rain_m3"] and b["drainage_m3"] > 0.0
    t = s["time"]
    assert (t["n_steps"], t["end_s"], t["dt_max_s"]) == (1620, 1620.0, 1.0)
    assert s["status"].startswith("rainfall_window_complete") and "NOT an event completion" in s["status"]
    assert s["final_state"]["ponded_cells"] > 0  # ponded water is retained, not discarded


def test_plot1_selected_cells_match_legacy_transcription(plot1_run):
    _run, out = plot1_run
    final = np.load(out / FINAL_NAME)
    cover = _interior_maple(PLOT1_DIR / "p1pavcoverveg.asc")  # percent
    theta_sat = _interior_maple(PLOT1_DIR / "thetasat39.asc")
    cells = {tuple(int(i) for i in np.unravel_index(k, cover.shape))
             for k in (int(np.argmin(cover)), int(np.argmax(cover)), int(np.argmin(np.abs(cover - 45.0))))}
    rates = [v / 3600.0 for v in _plot1_intensities_mm_h()]
    for cell in cells:
        stmax = theta_sat[cell] * 300.0
        d1, cum, drained = 0.0, 0.25 * 300.0, 0.0
        for rate in rates:
            for _ in range(60):
                d1, cum, drain, _ = legacy_infilt(
                    model=2, r=rate, d1=d1, cum_inf=cum, stmax=stmax, theta_sat=theta_sat[cell],
                    ksat=0.00025, psi=46.6, drain_par=0.05, pave=cover[cell] * 1e-4, dt=1.0)
                drained += drain
        assert final["depth_m"][cell] * 1e3 == pytest.approx(d1, rel=1e-9, abs=1e-10)
        assert final["soil_water_m"][cell] * 1e3 == pytest.approx(cum, rel=1e-9)
        assert final["cumulative_drainage_m"][cell] * 1e3 == pytest.approx(drained, rel=1e-9)
    assert len(cells) >= 2  # at least bare (lambda = 0.16 mm/s) and the most paved cell


def test_plot1_maple_state_untouched_and_water_state(plot1_case, plot1_run):
    from maple.case_tools.compilers.case_compiler import load_compiled_case
    from maple.core.types.water import WaterState

    run, out = plot1_run
    fresh = load_compiled_case(plot1_case)
    used = run.verified.case
    pairs = {
        "voxel": (used.voxel_column.mass_kg, fresh.voxel_column.mass_kg),
        "active": (used.active_layer.mass_kg, fresh.active_layer.mass_kg),
        "available": (used.sediment_availability.available_mass_kg, fresh.sediment_availability.available_mass_kg),
        "bound": (used.sediment_availability.bound_mass_kg, fresh.sediment_availability.bound_mass_kg),
        "elevation": (used.topography_result.elevation_m, fresh.topography_result.elevation_m),
        "pending": (used.sediment_ledger.pending_bed_mass_change_kg, fresh.sediment_ledger.pending_bed_mass_change_kg),
        "mobile": (run.water.mobile_mass_by_cell_class_kg, fresh.water.mobile_mass_by_cell_class_kg),
    }
    for name, (a, b) in pairs.items():
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
    assert isinstance(run.water, WaterState)
    assert run.water.mobile_mass_by_cell_class_kg is used.water.mobile_mass_by_cell_class_kg
    final = np.load(out / FINAL_NAME)
    np.testing.assert_array_equal(run.water.depth_m, final["depth_m"])
    assert not np.any(np.asarray(used.water.depth_m))  # the loaded state itself was not overwritten
    assert run.summary["maple_state"]["sediment_digest_before"] == run.summary["maple_state"]["sediment_digest_after"]


def test_plot1_outputs_are_bounded_and_hashed(plot1_run, plot1_case):
    _run, out = plot1_run
    assert sorted(p.name for p in out.iterdir()) == sorted([SUMMARY_NAME, FINAL_NAME])
    on_disk = json.loads((out / SUMMARY_NAME).read_text())
    assert on_disk["outputs"][FINAL_NAME] == _sha(out / FINAL_NAME)
    assert set(np.load(out / FINAL_NAME).files) == {
        "depth_m", "soil_water_m", "cumulative_rain_m", "cumulative_intake_m",
        "cumulative_return_m", "cumulative_drainage_m"}
    assert on_disk["case"]["maple_case_identity_sha256"] == json.loads(
        (plot1_case / "syrup" / "plot1_binding.json").read_text())["maple_case_identity_sha256"]
    with pytest.raises(ColumnExperimentError, match="existing"):
        run_plot1_columns(plot1_case, out, max_dt_s=1.0)
    with pytest.raises(ColumnExperimentError, match="max_dt_s"):
        run_plot1_columns(plot1_case, out.parent / "never", max_dt_s=0.0)
    assert not (out.parent / "never").exists()


def test_cli_splits_rain_knots_with_larger_max_dt(plot1_case, tmp_path, capsys):
    out = tmp_path / "cli"
    code = main(["--case-dir", str(plot1_case), "--max-dt-s", "45", "--output-dir", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    printed = json.loads(text[text.index('{\n  "budget"'):])  # the summary is the last thing printed
    assert (printed["time"]["n_steps"], printed["time"]["dt_max_s"]) == (54, 30.0)
    assert printed["budget"]["rain_m3"] == pytest.approx(2.8956, rel=1e-12)


# --- tampering is refused before anything is written ---------------------------------------
@pytest.fixture(scope="module")
def tamper_case(tmp_path_factory):
    return _generate(tmp_path_factory, "plot1_columns_tamper")


def _tampered(path: Path):
    original = path.read_bytes()
    path.write_bytes(original + b"\0")
    return original


@pytest.mark.parametrize(
    "relpath, error",
    [
        ("syrup/plot1_fields.npz", "plot1_fields.npz"),
        ("syrup/rainfall/p1_01_08_06.dat", "staged rainfall"),
        ("source/mahleran_input_p1/thetasat39.asc", "staged thetasat39.asc"),
        ("processed", None),  # MAPLE's own artifact hash check
    ],
)
def test_modified_inputs_are_refused(tamper_case, tmp_path, relpath, error):
    from maple_syrup.case_import import Plot1ImportError

    path = tamper_case / relpath
    if path.is_dir():
        path = min(path.glob("*.npz"))
    original = _tampered(path)
    out = tmp_path / "out"
    try:
        with pytest.raises((Plot1ImportError, ValueError)) as info:
            run_plot1_columns(tamper_case, out, max_dt_s=1.0)
        if error is not None:
            assert error in str(info.value)
    finally:
        path.write_bytes(original)
    assert not out.exists()


def _mahleran_copy(root: Path) -> Path:
    """Disposable copy of the root XML and Plot 1 inputs; the reference is only read."""
    target = root / "Input" / "input_p1"
    target.mkdir(parents=True)
    shutil.copy2(MAHLERAN_ROOT / "mahleran_input.xml", root / "mahleran_input.xml")
    for source in PLOT1_DIR.iterdir():
        if source.is_file() and source.suffix in (".asc", ".dat"):
            shutil.copy2(source, target / source.name)
    return target


def test_calibration_file_is_refused(tamper_case, tmp_path):
    from maple_syrup.case_import import Plot1ImportError

    root = tmp_path / "mahleran_copy"
    (_mahleran_copy(root) / "calib.dat").write_text("2.0\n1.0\n")
    with pytest.raises(Plot1ImportError, match="calib.dat"):
        run_plot1_columns(tamper_case, tmp_path / "out", max_dt_s=1.0, mahleran_root=root)
    assert not (tmp_path / "out").exists()


# --- output location and code identity -------------------------------------------------------
def test_output_inside_the_mahleran_root_in_use_is_refused(plot1_case, tmp_path):
    from maple_syrup.case_import import Plot1ImportError

    root = tmp_path / "mahleran_copy"
    _mahleran_copy(root)
    out = root / "Output" / "columns"
    with pytest.raises(Plot1ImportError, match="MAHLERAN tree"):
        run_plot1_columns(plot1_case, out, max_dt_s=60.0, mahleran_root=root)
    assert not out.exists() and not out.parent.exists()


@pytest.mark.parametrize("field, label", [("source_root", "MAPLE source"), ("package_dir", "MAPLE package")])
def test_output_inside_the_imported_maple_is_refused(plot1_case, tmp_path, monkeypatch, field, label):
    """The resolved MAPLE roots are simulated by a tmp directory; the real
    MAPLE tree is never a write target."""
    import dataclasses

    from maple_syrup import case_import
    from maple_syrup.case_import import Plot1ImportError

    fake = tmp_path / "fake_maple"
    fake.mkdir()
    (fake / "__init__.py").write_text("")
    real = case_import.resolve_maple_dependency
    monkeypatch.setattr(case_import, "resolve_maple_dependency",
                        lambda *a, **k: dataclasses.replace(real(*a, **k), **{field: fake}))
    out = fake / "columns"
    with pytest.raises(Plot1ImportError, match=f"{label} tree"):
        # a fake package_dir has a different digest, so allow the recorded drift
        run_plot1_columns(plot1_case, out, max_dt_s=60.0, allow_maple_source_change=True)
    assert not out.exists()


def test_run_provenance_identifies_the_code(plot1_run):
    run, out = plot1_run
    on_disk = json.loads((out / SUMMARY_NAME).read_text())
    assert "maple_syrup_version" not in on_disk["sources"]
    prov = on_disk["provenance"]
    syrup = prov["maple_syrup"]
    digest = syrup["package_source_digest"]["digest_sha256"]
    assert len(digest) == 64 and int(digest, 16) >= 0
    assert syrup["git"]["status"] == "ok" and len(syrup["git"]["head_commit"]) == 40
    assert syrup["git"]["scoped_status"]["status"] in ("clean", "dirty")
    assert syrup["git"]["head_describes_source"] == (syrup["git"]["scoped_status"]["status"] == "clean")
    stability = prov["source_stability"]
    assert stability["stable"] is True and stability["before"] == stability["after"]
    assert stability["before"]["maple_syrup"] == digest
    assert (stability["before"]["maple"] == prov["maple"]["package_source_digest"]["digest_sha256"]
            == run.verified.checks["maple_source_digest_now"])
    assert prov["maple_matches_import_binding"] is True
    assert prov["environment"]["python_version"]


def test_source_change_during_the_run_writes_nothing(plot1_case, tmp_path, monkeypatch):
    from maple_syrup import column_experiment

    real = column_experiment._source_digests
    monkeypatch.setattr(column_experiment, "_source_digests",
                        lambda *a: {**real(*a), "maple_syrup": "0" * 64})
    out = tmp_path / "out"
    with pytest.raises(ColumnExperimentError, match="source changed during the run"):
        run_plot1_columns(plot1_case, out, max_dt_s=60.0)
    assert not out.exists()
