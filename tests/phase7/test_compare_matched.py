"""Comparator units, orientation and accounting on synthetic data, plus an
end-to-end run on a synthetic SYRUP/MAHLERAN directory pair. No storm and
no MAHLERAN execution; these tests prove the comparator's arithmetic, not
any agreement between the models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import compare_matched as cm
import numpy as np
import pytest


def test_asc_orientation_crops_ring_and_reverses_rows():
    full = np.full((5, 4), -9999.0)
    full[1:-1, 1:-1] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # north-first interior rows
    interior = cm.mahleran_asc_interior(full, shape=(3, 2))
    np.testing.assert_array_equal(interior, [[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])  # MAPLE row 0 = south
    with pytest.raises(ValueError, match="ring"):
        cm.mahleran_asc_interior(full[:-1], shape=(3, 2))
    bad = full.copy()
    bad[2, 2] = -9999.0
    with pytest.raises(ValueError, match="nodata"):
        cm.mahleran_asc_interior(bad, shape=(3, 2))


def test_rounding_matches_four_significant_digits_and_units():
    values = np.array([0.00011509476271, 1.0e-10 * 9.694, 0.0, 123456.789])
    rounded = cm.round_sig(values)
    assert rounded.tolist() == [0.0001151, 9.694e-10, 0.0, 123500.0]
    # mm3/s -> m3/s conversion used for hydro001 column 3
    assert 0.1151e6 * 1.0e-9 == pytest.approx(0.0001151)


def test_endpoint_integral_differs_from_a_conservative_ledger_and_plateau():
    t = np.arange(1.0, 7.0)
    q = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0])
    assert cm.endpoint_integral_m3(t, q) == 8.0  # right-endpoint sum with dt = 1
    trapezoid = float(np.sum(0.5 * (q[1:] + q[:-1])))  # a face-integrated ledger is a different quantity
    assert trapezoid == 7.0 and trapezoid != cm.endpoint_integral_m3(t, q)
    assert cm.plateau_s(t, q) == (3.0, 4.0)
    with pytest.raises(ValueError, match="increasing"):
        cm.endpoint_integral_m3(np.array([1.0, 1.0]), np.array([1.0, 1.0]))


def test_hydrograph_metrics_and_targets_on_rounded_reference():
    t = np.arange(1.0, 201.0)
    test = 1.0e-4 * np.exp(-((t - 100.0) / 30.0) ** 2) + 1.2345e-6 * (t / 200.0)
    reference = cm.round_sig(test)  # what the stock file would print
    m = cm.hydrograph_metrics(t, reference, test)
    assert abs(m["integrated_relative_difference"]) < 1e-3 and abs(m["peak_relative_difference"]) < 1e-3
    assert m["reference_peak_plateau_s"][0] <= m["test_peak_time_s"] <= m["reference_peak_plateau_s"][1]
    assert m["peak_time_offset_from_plateau_s"] == 0.0 and m["rounding_floor_m3_s"] <= 5e-8
    targets = cm.evaluate_targets(m)
    assert all(targets[k]["pass"] for k in ("integrated_outlet_relative_difference", "peak_outlet_relative_difference",
                                            "peak_time_within_rounded_plateau"))
    worse = cm.hydrograph_metrics(t, reference, 1.05 * np.roll(test, 15))
    bad = cm.evaluate_targets(worse)
    assert not bad["integrated_outlet_relative_difference"]["pass"] and not bad["peak_outlet_relative_difference"]["pass"]
    assert not bad["peak_time_within_rounded_plateau"]["pass"] and worse["peak_time_offset_from_plateau_s"] >= 5.0
    assert "never relaxed" in bad["policy"]
    with pytest.raises(ValueError, match="time axis"):
        cm.hydrograph_metrics(t, reference[:-1], test)


def test_field_metrics_and_applied_expansion():
    ref = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = cm.field_metrics(ref, ref * 2.0)
    assert m["sum_ratio_test_over_reference"] == 2.0 and m["pearson_r"] == pytest.approx(1.0)
    assert m["relative_l2"] == pytest.approx(1.0) and m["cell_of_max_abs_difference_maple_row_col"] == [1, 1]
    forcing = {"edges_s": np.array([0.0, 2.0, 3.0, 5.0]), "intensity_mm_per_h": np.array([15.24, 0.0, 7.62])}
    np.testing.assert_array_equal(cm.syrup_applied_per_second(forcing, 7), [15.24, 15.24, 0.0, 7.62, 7.62, 0.0, 0.0])


# --- end-to-end on a synthetic pair --------------------------------------------------------------------------
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def asc_text(interior_south_first: np.ndarray) -> str:
    full = np.full((62, 22), -9999.0)
    full[1:-1, 1:-1] = interior_south_first[::-1]
    header = "ncols 22\nnrows 62\nxllcorner 0\nyllcorner 0\ncellsize 0.5\nnodata_value -9999\n"
    # 17 significant digits round-trip exactly, so the identical-pair assertions test accounting, not printing
    return header + "\n".join(" ".join(f"{v:.17E}" for v in row) for row in full) + "\n"


def synthetic_pair(root: Path, *, scale=1.0, forcing_ok=True, peak_scale=1.0):
    """`peak_scale` scales ONLY the SYRUP per-cell peak depth/velocity fields, so
    a value other than 1 gives storm maxima that deliberately differ from the
    legacy peak-outlet snapshot: the comparator must not treat them as the
    same quantity whatever the value."""
    n = 6
    t = np.arange(1.0, n + 1)
    q = np.array([1.0e-5, 4.0e-5, 1.151e-4, 1.151e-4, 6.0e-5, 2.0e-5])
    sed = np.array([0.0, 1.0e-4, 3.0e-4, 2.0e-4, 1.0e-4, 0.0])
    classes = np.column_stack([sed * f for f in (0.8, 0.19, 0.01, 0.0, 0.0, 0.0)])
    rain = np.array([15.24, 15.24, 30.48, 0.0, 0.0, 0.0])
    # MAHLERAN
    m = root / "mahleran"
    (m / "Output").mkdir(parents=True)
    hydro = np.zeros((n, 12))
    hydro[:, 0], hydro[:, 2] = t, q * 1.0e9
    np.savetxt(m / "Output" / "hydro001.dat", hydro, fmt="%.4E")
    sedtr = np.zeros((n, 16))
    sedtr[:, 0], sedtr[:, 1] = t, sed
    np.savetxt(m / "Output" / "sedtr001.dat", sedtr, fmt="%.4E")
    np.savetxt(m / "Output" / "seddisch001.dat", np.column_stack((t, classes)), fmt="%.4E")
    rng = np.random.default_rng(1)
    maps = {"depth": rng.uniform(0.0, 5.0, (60, 20)), "veloc": rng.uniform(0.0, 50.0, (60, 20)),
            "detac": rng.uniform(0.0, 1e-3, (60, 20)), "neter": rng.uniform(-1e-3, 1e-3, (60, 20)),
            "dschg": rng.uniform(0.0, 1e-2, (60, 20))}
    for name, grid in maps.items():
        (m / "Output" / f"{name}001.asc").write_text(asc_text(grid))
    (m / "Output" / "ksat_001.asc").write_text(asc_text(np.full((60, 20), 0.00025)))
    (m / "mahleran_input.xml").write_text('<x><finalInfiltrationRateDistribution value="deterministic" /></x>')
    log = "".join(f" Starting iteration {k + 1:6d} rain intensity: {r:7.2f} time step:  1.00 t: 0\n"
                  for k, r in enumerate(rain))
    (m / "stdout.log").write_text(log)
    outputs = {f"Output/{p.name}": {"sha256": sha(p)} for p in (m / "Output").iterdir()}
    (m / "execution.json").write_text(json.dumps({"returncode": 0, "completion_marker": True,
                                                  "executable_sha256": "e" * 64, "outputs": outputs}))
    # SYRUP
    s = root / "syrup"
    s.mkdir()
    q_s = q * scale
    ledger = np.cumsum(q_s) * 1.001  # a face ledger is not the endpoint sum
    hyd = {"t_s": t, "outlet_discharge_m3_s": q_s, "cumulative_export_m3": ledger,
           "sed_cumulative_export_kg": np.cumsum(sed) * scale, "sed_export_rate_kg_s": sed * scale}
    for k in range(6):
        hyd[f"sed_cumulative_export_c{k + 1}_kg"] = np.cumsum(classes[:, k]) * scale
    np.savez(s / "hydrograph.npz", **hyd)
    final = {"peak_depth_m": maps["depth"] * 1e-3 * peak_scale, "peak_velocity_m_s": maps["veloc"] * 1e-3 * peak_scale,
             "cumulative_pickup_kg": np.repeat(maps["detac"][..., None] / 6.0, 6, axis=-1),
             "cumulative_deposition_kg": np.repeat((maps["detac"] - maps["neter"])[..., None] / 6.0, 6, axis=-1),
             "bed_change_kg": np.repeat(-maps["neter"][..., None] / 6.0, 6, axis=-1)}
    np.savez(s / "final_state.npz", **final)
    edges = np.array([0.0, 2.0, 3.0, 6.0]) if forcing_ok else np.array([0.0, 3.0, 6.0])
    intensity = np.array([15.24, 30.48, 0.0]) if forcing_ok else np.array([15.24, 0.0])
    np.savez(s / "forcing.npz", edges_s=edges, intensity_mm_per_h=intensity)
    summary = {
        "schema": "maple_syrup.plot1_matched_benchmark.v1",
        "mode": {"frozen_hydraulic_geometry": True, "commit": False, "force_final_commit": False,
                 "checks": {"checks": {"graph_same_object": True, "n_commits_zero": True}}},
        "outputs": {name: sha(s / name) for name in ("hydrograph.npz", "final_state.npz", "forcing.npz")},
        "time": {"report_every_s": 1.0, "max_dt_s": 1.0},
        "conductivity": {"mm_s": 0.00025},
        "sediment": {"peak_export_rate_kg_s": float(sed.max() * scale), "time_of_peak_export_s": 3.0,
                     "totals": {"actual_pickup": float(maps["detac"].sum()),
                                "deposition_actual": float((maps["detac"] - maps["neter"]).sum())},
                     "closure": {"closed": True}},
        "budget": {"water_residual_m3": 0.0},
        "provenance": {"implementation": "array",
                       "maple_syrup": {"package_source_digest": {"digest_sha256": "a" * 64}},
                       "maple": {"package_source_digest": {"digest_sha256": "b" * 64}}},
    }
    (s / "benchmark_summary.json").write_text(json.dumps(summary))
    return s, m


def test_compare_end_to_end_identical_pair(tmp_path):
    s, m = synthetic_pair(tmp_path)
    report, arrays = cm.compare(s, m)
    cfg = report["matched_configuration"]
    assert cfg["forcing_identical_per_second"] and cfg["reference_conductivity_uniform"]
    assert cfg["reference_distribution"] == "deterministic"
    w = report["water"]
    assert w["integrated_relative_difference"] == pytest.approx(0.0, abs=1e-12)
    assert w["syrup_conservative_export_ledger_m3"] != w["test_endpoint_integral_m3"]
    assert w["reference_peak_plateau_s"] == [3.0, 4.0] and w["test_peak_time_s"] == 3.0
    assert all(report["targets"][k]["pass"] for k in ("integrated_outlet_relative_difference",
                                                       "peak_outlet_relative_difference",
                                                       "peak_time_within_rounded_plateau"))
    sd = report["sediment"]
    assert sd["total_ratio_test_over_reference"] == pytest.approx(1.0)
    assert sd["class_ratio_test_over_reference"][0] == pytest.approx(1.0) and sd["class_ratio_test_over_reference"][3] is None
    assert sd["reference_total_export_kg_class_sum"] == pytest.approx(sd["reference_total_export_kg_endpoint_sum"])
    sp = report["spatial"]
    assert "max_depth_mm" not in sp and "max_velocity_mm_s" not in sp
    snapshot = sp["peak_outlet_snapshot_depth_velocity"]
    assert snapshot["comparable"] is False and "489-505" in snapshot["reference_definition"]
    assert "(mm)" in snapshot["reference_definition"] and "(mm/s)" in snapshot["reference_definition"]
    assert "[3.0, 4.0]" in snapshot["reference_snapshot_time_s"]
    assert not any(key in snapshot for key in ("relative_l2", "rmse", "pearson_r", "max_abs_difference"))
    assert sp["detachment_kg_per_cell"]["sum_ratio_test_over_reference"] == pytest.approx(1.0)
    assert sp["net_erosion_kg_per_cell_water_exchange"]["relative_l2"] == pytest.approx(0.0, abs=1e-9)
    assert sp["net_erosion_kg_per_cell_actual_bed"]["relative_l2"] == pytest.approx(0.0, abs=1e-9)
    assert sp["cumulative_cell_discharge_m3"]["test"] is None
    assert arrays["mahleran_peak_outlet_snapshot_depth_mm"].shape == (60, 20)
    assert "mahleran_max_depth_mm" not in arrays and "syrup_max_depth_mm" not in arrays
    assert report["spatial_panels"] == [label for _, _, label in cm.SPATIAL_PANELS]
    assert not any(word in " ".join(report["spatial_panels"]).lower() for word in ("depth", "velocity"))
    assert any("peak-outlet snapshots" in item for item in report["limitations"])
    assert report["mahleran_outputs_sha256"] and report["syrup_outputs_sha256"]
    json.dumps(report, allow_nan=False)


def test_differing_per_cell_maxima_are_never_scored_against_the_peak_outlet_snapshot(tmp_path):
    """Regression for the withdrawn depth/velocity 'field metrics': SYRUP storm
    maxima three times larger than the legacy snapshot must produce no error
    metric, no pass/fail and no side-by-side panel, only labelled statistics."""
    s, m = synthetic_pair(tmp_path, peak_scale=3.0)
    report, arrays = cm.compare(s, m)
    snapshot = report["spatial"]["peak_outlet_snapshot_depth_velocity"]
    assert snapshot["comparable"] is False
    ref, tst = snapshot["reference_snapshot_stats"], snapshot["test_storm_maxima_stats"]
    assert tst["depth_mm"]["max"] == pytest.approx(3.0 * ref["depth_mm"]["max"])
    assert tst["velocity_mm_s"]["sum"] == pytest.approx(3.0 * ref["velocity_mm_s"]["sum"])
    assert set(snapshot) == {"comparable", "reference_definition", "reference_snapshot_time_s", "test_definition",
                             "matching_quantity", "reference_snapshot_stats", "test_storm_maxima_stats", "note"}
    assert "peak-outlet" in snapshot["matching_quantity"]
    # the water and sediment scoring is untouched by the differing maxima
    assert all(report["targets"][k]["pass"] for k in ("integrated_outlet_relative_difference",
                                                       "peak_outlet_relative_difference",
                                                       "peak_time_within_rounded_plateau"))
    assert report["sediment"]["total_ratio_test_over_reference"] == pytest.approx(1.0)
    # retained diagnostics are labelled by definition and are never co-plotted
    np.testing.assert_allclose(arrays["syrup_storm_max_depth_mm"], 3.0 * arrays["mahleran_peak_outlet_snapshot_depth_mm"])
    assert not any("depth" in a or "veloc" in a or "depth" in b or "veloc" in b for a, b, _ in cm.SPATIAL_PANELS)
    plots = cm.plot(arrays, tmp_path)
    assert plots in ([], ["series.png", "spatial.png"])  # matplotlib optional; panels come only from SPATIAL_PANELS


def test_compare_flags_mismatch_and_forcing_difference(tmp_path):
    s, m = synthetic_pair(tmp_path, scale=1.05, forcing_ok=False)
    report, _ = cm.compare(s, m)
    assert not report["matched_configuration"]["forcing_identical_per_second"]
    assert any("FORCING MISMATCH" in item for item in report["limitations"])
    assert not report["targets"]["integrated_outlet_relative_difference"]["pass"]
    assert report["water"]["integrated_relative_difference"] == pytest.approx(0.05)
    assert report["sediment"]["total_ratio_test_over_reference"] == pytest.approx(1.05)


def test_compare_refuses_tampered_or_wrong_inputs(tmp_path):
    s, m = synthetic_pair(tmp_path)
    (m / "Output" / "hydro001.dat").write_text("1 0 0 0 0 0 0 0 0 0 0 0\n")
    with pytest.raises(ValueError, match="execution manifest"):
        cm.compare(s, m)
    s2, m2 = synthetic_pair(tmp_path / "second")
    summary = json.loads((s2 / "benchmark_summary.json").read_text())
    summary["mode"]["commit"] = True
    (s2 / "benchmark_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="frozen-geometry"):
        cm.compare(s2, m2)
