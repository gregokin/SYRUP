"""The qualification comparator on a fully synthetic capture run and SYRUP output pair, so every conversion (orientation,
units, precision, sampling) has a known answer. No model is executed; these tests prove arithmetic, targets and
refusals, not agreement between the models.
Nothing here was run by its author (file-only tools); Codex records actual results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import capture_data as cd
import compare_hydrology as ch
import compare_matched as cm
import numpy as np
import pytest
from capture_fixture import full, render_capture, render_steps, synthetic_setup

NR, NC, N = 4, 3, 6
Q_MM2 = np.array([0.0, 1.0, 3.0, 5.0, 4.0, 2.0]) * 0.25  # exact in float32; peak at iteration 4
RVAL = np.full(N, 0.01)
DEPTH = 1.0e-3 * (1.0 + np.arange(6.0).reshape(3, 2))  # SYRUP order (row 0 = south), m
VEL = 1.0e-2 * (1.0 + np.arange(6.0).reshape(3, 2)[::-1])  # m/s
SOIL = 0.07 + 1.0e-3 * np.arange(6.0).reshape(3, 2)  # m
DRAIN = 1.0e-4 * (1.0 + np.arange(6.0).reshape(3, 2))  # m
Q_M3_S = Q_MM2 * 500.0 * 1.0e-9


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_asc(path, interior_si_order, factor):
    arr = full(np.asarray(cm.round_sig(interior_si_order * factor)), 0.0, NR, NC)
    header = "ncols 4\nnrows 5\nxllcorner 0\nyllcorner 0\ncellsize 0.5\nnodata_value -9999\n"
    with path.open("w") as handle:
        handle.write(header)
        np.savetxt(handle, arr, fmt="%.10E")


def build_run(tmp_path):
    setup = synthetic_setup(nr=NR, nc=NC, nit=N)
    run = tmp_path / "run"
    out = run / "Output"
    out.mkdir(parents=True)
    (out / cd.STATIC_NAME).write_text(setup["text"])
    (out / cd.STEPS_NAME).write_text(render_steps(RVAL, q_single=Q_MM2, q_double=Q_MM2))
    zeros = np.zeros((NR + 1, NC + 1))
    arrays = {"d_peak_single_mm": full(DEPTH * 1e3, 0.0, NR, NC), "v_peak_single_mm_s": full(VEL * 1e3, 0.0, NR, NC),
              "d_peak_double_mm": full(DEPTH * 1e3, 0.0, NR, NC), "v_peak_double_mm_s": full(VEL * 1e3, 0.0, NR, NC),
              "model_dmax_mm": full(DEPTH * 1e3, 0.0, NR, NC), "model_vmax_mm_s": full(VEL * 1e3, 0.0, NR, NC),
              "d_final_mm": full(DEPTH * 0.5e3, 0.0, NR, NC), "q_final_mm2_s": zeros.copy(), "v_final_mm_s": zeros.copy(),
              "cum_inf_final_mm": full(SOIL * 1e3, 0.0, NR, NC), "cum_drain_final_mm": full(DRAIN * 1e3, 0.0, NR, NC)}
    scalars = {"n_steps_written": N, "peak_iter_single": 4, "peak_iter_double": 4, "peak_q_single_mm2_s": 1.25,
               "peak_q_double_mm2_s": 1.25, "max_abs_model_dmax_minus_capture_mm": 0.0,
               "max_abs_model_vmax_minus_capture_mm_s": 0.0}
    (out / cd.FINAL_NAME).write_text(render_capture("final", scalars, arrays))
    stock = np.zeros((N, 12))
    stock[:, 0] = np.arange(1, N + 1)
    stock[:, 1] = RVAL
    stock[:, 2] = cm.round_sig((Q_MM2.astype(np.float32) * np.float32(500.0)).astype(np.float64))
    np.savetxt(out / "hydro001.dat", stock, fmt="%.4E")
    write_asc(out / "ksat_001.asc", setup["ksat_mm"], 1.0)
    write_asc(out / "depth001.asc", DEPTH, 1e3)
    write_asc(out / "veloc001.asc", VEL, 1e3)
    outputs = {f"Output/{p.name}": {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir())}
    (run / "mahleran_input.xml").write_text("<xml/>\n")
    record = {"returncode": 0, "completion_marker": True, "input_unchanged": True, "reference_unchanged": True,
              "prepared_unchanged": True, "outputs": outputs, "executable_sha256": "cd" * 32,
              "input_sha256": {"mahleran_input.xml": sha(run / "mahleran_input.xml")}}
    (run / "execution.json").write_text(json.dumps(record))
    return run, record


def build_syrup(tmp_path, record, *, label="dt1", substeps=1, q=Q_M3_S, depth_scale=1.0, flip=False, n_seconds=N,
                capture_exe=None):
    d = tmp_path / label
    d.mkdir()
    def sub(a):  # one value per sub-step; the second's end is the last
        return np.repeat(a, substeps)

    history = {"outlet_m3_s": sub(q), "export_m3": sub(q) / substeps}
    orient = (lambda a: a[::-1]) if flip else (lambda a: a)
    fields = {"own_peak_depth_m": orient(DEPTH * depth_scale), "own_peak_velocity_m_s": orient(VEL),
              "final_depth_m": orient(DEPTH * 0.5), "final_soil_water_m": orient(SOIL), "cum_drainage_m": orient(DRAIN),
              "snapshot_s4_depth_m": orient(DEPTH), "snapshot_s4_velocity_m_s": orient(VEL)}
    np.savez(d / "history.npz", **history)
    np.savez(d / "fields.npz", **fields)
    budget = {"rain": 1.0, "drainage": float(DRAIN.sum()) * 0.25, "intake": 0.5, "return": 0.0,
              "surface_final": float(DEPTH.sum()) * 0.5 * 0.25, "soil_final": float(SOIL.sum()) * 0.25,
              "residual": 0.0, "water_residual_m3": 0.0, "bound_m3": 1.0e-9, "closed": True}
    summary = {"schema": "maple_syrup.phase7i.hydrology.v1", "substeps": substeps, "n_seconds": n_seconds,
               "implementation": "prepared",
               "inputs": {"capture_executable_sha256": capture_exe or record["executable_sha256"],
                          "capture_files_sha256": {name: record["outputs"][name]["sha256"] for name in cd.CAPTURE_FILES}},
               "peak": {"own_peak_time_s": 4.0, "own_outlet_peak_m3_s": float(np.max(q))}, "budget": budget,
               "parity_prepared_vs_reference": {"pass": True, "compared_steps": N * substeps},
               "conductivity": {}, "forcing": {},
               "outputs": {name: sha(d / name) for name in ("history.npz", "fields.npz")}}
    (d / "hydrology_summary.json").write_text(json.dumps(summary))
    return d


# --- units, helpers ----------------------------------------------------------------------------------------------
def test_rounded_map_check_bounds_and_hidden_zeros():
    printed = np.array([[0.1350e-2, 0.2954e-3], [0.0, 0.9735e-4]])
    exact = np.array([[0.13503e-2, 0.29544e-3], [0.0, 0.97351e-4]])
    check = ch.rounded_map_check(exact, printed)
    assert check["consistent"] and check["max_relative_difference"] <= cd.ROUNDED_MAP_RELATIVE_BOUND
    assert not ch.rounded_map_check(exact * 1.01, printed)["consistent"]
    hidden = exact.copy()
    hidden[1, 0] = 1.0e-3  # a printed zero must not hide a material value
    assert ch.rounded_map_check(hidden, printed)["printed_zero_cells_hiding_nonzero_capture"] == 1


def test_original_outputs_identity_is_by_bytes_and_reports_differences(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name / "Output").mkdir(parents=True)
        for f in ch.NUMERIC_OUTPUTS:
            (tmp_path / name / "Output" / f).write_text(f"{f}\n")
    same = ch.compare_original_outputs(tmp_path / "a", tmp_path / "b")
    assert same["all_numeric_outputs_identical"] and len(same["identical"]) == 14 and not same["different"]
    (tmp_path / "b/Output/hydro001.dat").write_text("changed\n")
    (tmp_path / "b/Output/aspct001.asc").unlink()
    diff = ch.compare_original_outputs(tmp_path / "a", tmp_path / "b")
    assert diff["different"] == ["hydro001.dat"] and diff["missing"] == ["aspct001.asc"]
    assert not diff["all_numeric_outputs_identical"] and "OUTPUTS DIFFER" in diff["interpretation"]


def test_spatial_pair_applies_the_predeclared_one_percent_target_and_flags_unevaluable_references():
    ref = np.arange(1.0, 7.0).reshape(3, 2)
    assert ch.spatial_pair(ref, ref)["pass"] and ch.spatial_pair(ref, ref * 1.009)["pass"]
    assert not ch.spatial_pair(ref, ref * 1.02)["pass"]
    zero = ch.spatial_pair(np.zeros((3, 2)), np.ones((3, 2)))
    # an undefined relative error is neither a pass nor a failure: it is flagged as a zero-reference case
    assert zero["evaluable"] is False and zero["pass"] is None and zero["reference_is_zero"] and not zero["zero_match"]
    both_zero = ch.spatial_pair(np.zeros((3, 2)), np.zeros((3, 2)))
    assert both_zero["pass"] is None and both_zero["zero_match"] is True and "zero" in both_zero["note"]
    assert ch.TARGETS["spatial_relative_l2"] == 0.01 and ch.TARGETS["peak_time_tolerance_s"] == 10.0
    assert cm.TARGETS["integrated_outlet_relative_difference"] == 0.01 and cm.TARGETS["peak_outlet_relative_difference"] == 0.01


def test_legacy_budget_units_and_both_export_definitions():
    setup = synthetic_setup(nr=NR, nc=NC, nit=N)
    steps = cd.parse_steps_text(render_steps(RVAL, q_single=Q_MM2, q_double=Q_MM2))
    steps["cn_export_step_m3"] = np.full(N, 1.0e-6)
    budget = ch.legacy_budget(setup["static"], steps, NR, NC)
    c = 500.0 * 500.0 * 1.0e-9  # m3 per mm of depth in one cell
    assert budget["soil_initial"] == pytest.approx(6 * 75.0 * c) and budget["rain"] == pytest.approx(0.04 * N * c)
    assert budget["surface_final"] == pytest.approx(steps["surface_sum_mm"][-1] * c)
    assert budget["export_endpoint_sum_q_dt"] == pytest.approx(float(Q_M3_S.sum()))
    assert budget["export_cn_face"] == pytest.approx(6.0e-6)  # the conservative face export is a DIFFERENT quantity
    stored = budget["surface_final"] + budget["soil_final"] + budget["drainage"] - budget["soil_initial"]
    assert budget["residual_with_cn_export"] == pytest.approx(stored + 6.0e-6 - budget["rain"])
    assert budget["residual_with_endpoint_export"] == pytest.approx(stored + float(Q_M3_S.sum()) - budget["rain"])
    assert budget["net_infiltration"] == pytest.approx(budget["soil_final"] - budget["soil_initial"] + budget["drainage"])


def test_labelled_arguments_are_strict():
    assert ch.parse_labelled(["dt1=/a", "dt.5=/b"]) == {"dt1": Path("/a").resolve(), "dt.5": Path("/b").resolve()}
    assert ch.parse_labelled(None) == {}
    for bad in (["noequals"], ["=/a"], ["x="], ["a=/p", "a=/q"]):
        with pytest.raises(SystemExit):
            ch.parse_labelled(bad)


# --- the whole comparison on synthetic data --------------------------------------------------------------------------
def test_matching_models_pass_every_target_and_conversion(tmp_path):
    run, record = build_run(tmp_path)
    syrup = build_syrup(tmp_path, record)
    report, series = ch.compare(run, {"dt1": syrup}, {}, None)
    s = report["syrup"]["dt1"]
    for ref in ("hydrograph_vs_stock_rounded", "hydrograph_vs_full_precision", "hydrograph_vs_single_precision"):
        block = s[ref]
        assert all(v["pass"] for k, v in block["targets"].items() if k != "policy"), ref
    full_block = s["hydrograph_vs_full_precision"]
    assert full_block["rmse_m3_s"] == 0.0 and full_block["integrated_relative_difference"] == 0.0
    assert full_block["nash_sutcliffe"] == 1.0 and full_block["test_peak_time_s"] == 4.0
    assert report["legacy"]["peak_time_single_precision_logic_s"] == 4.0 and report["legacy"]["argmax_single_equals_model_peak"]
    for name in ("own_peak_depth", "own_peak_velocity", "final_surface_depth", "final_soil_water", "cumulative_drainage",
                 "synchronous_depth_at_legacy_single_peak_step_diagnostic",
                 "synchronous_velocity_at_legacy_double_peak_step_diagnostic"):
        assert s["spatial"][name]["pass"] and s["spatial"][name]["relative_l2"] < 1.0e-14, name  # (x e3) e-3 differs by ulps
    # unit conversions (mm -> m, mm/s -> m/s) and the row flip were exercised: a zero error is only possible if all agree
    consistency = report["stock_consistency"]
    assert consistency["hydro001_outlet_rounded_mismatches"] == 0 and consistency["ksat_map_consistent"]
    assert consistency["peak_maps_consistent"] and consistency["hook_peak_copy_equals_model_maps"]
    assert report["earlier_run_identity"] is None
    assert set(series) >= {"t_s", "mahleran_q_full_double_m3_s", "syrup_dt1_q_m3_s"}
    assert s["endpoint_vs_conservative_export_m3"] == pytest.approx(0.0, abs=1e-18)  # one sub-step: export = Q dt here
    assert report["sensitivity"][0]["label"].startswith("MAHLERAN") and len(report["sensitivity"]) == 2
    json.dumps(report, allow_nan=False, default=float)  # the report is plain finite JSON


def test_a_wrong_row_order_a_scaled_field_or_a_shifted_hydrograph_is_detected(tmp_path):
    run, record = build_run(tmp_path)
    flipped = ch.compare(run, {"dt1": build_syrup(tmp_path, record, label="flip", flip=True)}, {}, None)[0]["syrup"]["dt1"]
    assert not flipped["spatial"]["own_peak_depth"]["pass"] and not flipped["spatial"]["final_soil_water"]["pass"]
    scaled = ch.compare(run, {"dt1": build_syrup(tmp_path, record, label="scaled", depth_scale=1.05)}, {}, None)[0]["syrup"]["dt1"]
    assert scaled["spatial"]["own_peak_depth"]["relative_l2"] == pytest.approx(0.05, rel=1e-9)
    assert not scaled["spatial"]["own_peak_depth"]["pass"] and scaled["spatial"]["final_surface_depth"]["pass"]
    late = ch.compare(run, {"dt1": build_syrup(tmp_path, record, label="late", q=np.roll(Q_M3_S, 3) * 1.2)}, {}, None)[0]["syrup"]["dt1"]
    targets = late["hydrograph_vs_full_precision"]["targets"]
    assert not targets["peak_outlet_relative_difference"]["pass"]  # +20 % peak
    assert not targets["integrated_outlet_relative_difference"]["pass"]  # +20 % runoff
    assert targets["peak_time_within_rounded_plateau"]["offset_s"] == 3.0 and targets["peak_time_within_rounded_plateau"]["pass"]


def test_substeps_sample_the_second_ends_and_provenance_mismatches_are_refused(tmp_path):
    run, record = build_run(tmp_path)
    fine = ch.compare(run, {"dt.25": build_syrup(tmp_path, record, label="dt.25", substeps=4)}, {}, None)[0]["syrup"]["dt.25"]
    assert fine["substeps"] == 4 and fine["hydrograph_vs_full_precision"]["rmse_m3_s"] == 0.0
    assert fine["native_dt_endpoint_integral_m3"] == pytest.approx(float(Q_M3_S.sum()))  # sum Q dt over all 24 steps
    assert fine["one_second_sample_endpoint_integral_m3"] == pytest.approx(float(Q_M3_S.sum()))  # 6 samples, dt = 1 s
    row = ch.compare(run, {"dt.25": build_syrup(tmp_path, record, label="dt.25b", substeps=4)}, {}, None)[0]["sensitivity"][1]
    assert {"one_second_sample_endpoint_integral_m3", "native_dt_endpoint_integral_m3"} <= set(row)
    assert fine["hydrograph_vs_full_precision"]["diagnostic_peak_time_offset_from_reference_argmax_s"] == 0.0
    with pytest.raises(ValueError, match="different capture"):
        ch.compare(run, {"x": build_syrup(tmp_path, record, label="exe", capture_exe="ee" * 32)}, {}, None)
    with pytest.raises(ValueError, match="seconds"):
        ch.compare(run, {"x": build_syrup(tmp_path, record, label="secs", n_seconds=N + 1)}, {}, None)
    tampered = build_syrup(tmp_path, record, label="tampered")
    with (tampered / "fields.npz").open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError, match="recorded hash"):
        ch.compare(run, {"x": tampered}, {}, None)
    (run / "Output" / "hydro001.dat").write_text("tampered\n")
    with pytest.raises(cd.CaptureError, match="no longer matches"):
        ch.compare(run, {"dt1": build_syrup(tmp_path, record, label="after")}, {}, None)


def test_command_line_writes_a_new_directory_only(tmp_path):
    run, record = build_run(tmp_path)
    syrup = build_syrup(tmp_path, record)
    out = tmp_path / "comparison"
    assert ch.main(["--capture-run", str(run), "--syrup", f"dt1={syrup}", "--earlier-run", str(tmp_path / "none"),
                    "--output", str(out)]) == 0
    assert (out / "comparison.json").is_file() and (out / "series.npz").is_file()
    assert json.loads((out / "comparison.json").read_text())["schema"] == ch.SCHEMA
    with pytest.raises(cd.CaptureError, match="existing"):
        ch.main(["--capture-run", str(run), "--syrup", f"dt1={syrup}", "--output", str(out)])
    with pytest.raises(cd.CaptureError, match="inside"):
        ch.main(["--capture-run", str(run), "--syrup", f"dt1={syrup}", "--output", str(run / "sub")])
    with pytest.raises(SystemExit):
        ch.main(["--capture-run", str(run), "--output", str(tmp_path / "nosyrup")])
