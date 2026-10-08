"""CPU-only tests of `benchmarks/legacy_gpu/compare_cpu_gpu.py` on synthetic run directories: legitimately matching runs are accepted and
every malformed, incompatible or non-finite case is refused. Written without being run (Codex executes them)."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("compare_cpu_gpu", ROOT / "benchmarks" / "legacy_gpu" / "compare_cpu_gpu.py")
H = importlib.util.module_from_spec(spec)
sys.modules["compare_cpu_gpu"] = H
spec.loader.exec_module(H)

from maple_syrup.legacy_native_numba import LEDGER_COLUMNS

STEPS, NC, NY, NX = 5, 3, 2, 3


def base_arrays(seed=0):
    rng = np.random.default_rng(seed)
    ledger = np.zeros((STEPS, len(LEDGER_COLUMNS), NC))
    col = {name: i for i, name in enumerate(LEDGER_COLUMNS)}
    for name in ("pickup_kg", "deposition_active_kg", "effective_clip_source_kg", "old_mobile_kg", "cn_export_kg", "endpoint_export_kg"):
        ledger[:, col[name]] = rng.uniform(0.1, 1.0, (STEPS, NC))
    ledger[:, col["new_mobile_kg"]] = (ledger[:, col["old_mobile_kg"]] + (ledger[:, col["pickup_kg"]] - ledger[:, col["deposition_active_kg"]])
                                       - ledger[:, col["cn_export_kg"]] + ledger[:, col["effective_clip_source_kg"]])
    arrays = {"t_s": np.arange(1.0, STEPS + 1), "ledger": ledger, "columns": np.array(LEDGER_COLUMNS),
              "walk_counts": rng.integers(0, 50, (STEPS, 10)).astype(np.int64), "regime_counts": rng.integers(0, 9, (STEPS, 7)).astype(np.int64),
              "water_outlet_m3_s": rng.uniform(0, 1, STEPS), "water_export_m3": rng.uniform(0, 1, STEPS),
              "water_budget_residual_m3": rng.uniform(-1e-13, 1e-13, STEPS)}
    for key in H.SEDIMENT_KEYS[1:]:
        arrays[key] = rng.uniform(0.0, 1.0, (NY, NX, NC))
    for key in H.WATER_KEYS[:8]:
        arrays[key] = rng.uniform(0.0, 1.0, (NY, NX))
    arrays["active"] = np.ones((NY, NX), dtype=bool)
    arrays["terminal_storage"] = np.zeros((NY, NX), dtype=bool)
    outlet = np.zeros((NY, NX), dtype=bool)
    outlet[0, 0] = True
    arrays["outlet"] = outlet
    return arrays


def residual_of(ledger):
    from maple_syrup import legacy_driver as D

    return D._check_identity(ledger)[0]


def base_summary(**over):
    s = {"case_kind": "plot1", "steps": STEPS, "dt_s": 1.0, "end_s": 5.0, "depth_time_level": "previous",
         "law_depth": {"selected": "previous"},
         "config": {"source_order": "index", "legacy_depos_erase": False, "snapshot_times": ""},
         "hydrology": {"root_solver": "bisection", "bisection_iterations": 40, "newton_max_iterations": 20},
         "provenance": {"graph_input_sha256": "ab" * 32}, "network": {"n_active": NY * NX},
         "artifact_pins": {"sha256": {"/a": "11" * 32, "/b": "22" * 32}},
         "water": {"whole_storm_budget": {"closed": True}}, "totals": {"pickup_kg": 1.0}, "totals_by_class": {"pickup_kg": [0.1, 0.2, 0.3]},
         "walk_tallies": {"walks": 5}, "wet_regime_cell_class_steps": {"dry": 1}, "onset": {"first_positive_pickup_s": 1.0},
         "identity": {"worst_relative_residual": 1e-16}, "final_mobile_by_class_kg": [1, 2, 3], "maps_total_kg": {"a": [1, 2, 3]},
         "performance": {"loop_wall_s_excluding_progress": 2.0}, "gpu": {"device": {"name": "x"}}}
    for key, value in over.items():
        s[key] = value
    return s


def write_run(directory, arrays, summary, snapshots=None):
    directory.mkdir(parents=True)
    arrays = dict(arrays)
    arrays["identity_residual_kg"] = residual_of(arrays["ledger"])
    np.savez(directory / "legacy_ledger.npz", **arrays)
    (directory / "legacy_summary.json").write_text(json.dumps(summary))
    if snapshots is not None:
        np.savez(directory / "legacy_snapshots.npz", **snapshots)
    return directory


def pin(directory, summary_over=None, names=("legacy_ledger.npz", "legacy_snapshots.npz")):
    """Add `output_pins` (the driver's own function) to the run's summary, as a pinned run would have."""
    from maple_syrup import legacy_gpu_driver as G

    summary = json.loads((directory / "legacy_summary.json").read_text())
    summary["output_pins"] = G._output_pins(directory, names)
    summary.update(summary_over or {})
    (directory / "legacy_summary.json").write_text(json.dumps(summary))
    return summary


def perturbed(arrays, key, factor):
    out = {k: v.copy() for k, v in arrays.items()}
    out[key] = out[key] * factor
    return out


def pair(tmp_path, gpu_arrays=None, gpu_summary=None, cpu_arrays=None, cpu_summary=None):
    cpu = write_run(tmp_path / "cpu", cpu_arrays or base_arrays(), cpu_summary or base_summary())
    gpu = write_run(tmp_path / "gpu", gpu_arrays or base_arrays(), gpu_summary or base_summary())
    return cpu, gpu


def test_identical_runs_are_accepted_and_the_report_is_complete(tmp_path):
    cpu, gpu = pair(tmp_path)
    report = H.compare(cpu, gpu)
    assert report["accepted_by_declared_bounds"] and not report["flags"]
    assert all(v["pass"] for v in report["fields"].values()) and set(H.WATER_KEYS) <= set(report["fields"])
    assert report["identity_diagnostic"]["pass"] and report["receipts"]["root_solver"]["match"]


def test_a_difference_inside_the_declared_bounds_is_accepted_and_outside_is_refused(tmp_path):
    arrays = base_arrays()
    cpu = write_run(tmp_path / "cpu", arrays, base_summary())
    inside = write_run(tmp_path / "in", perturbed(arrays, "cumulative_deposition_kg", 1.0 + 1e-12), base_summary())
    assert H.compare(cpu, inside)["fields"]["cumulative_deposition_kg"]["pass"]
    outside = write_run(tmp_path / "out", perturbed(arrays, "cumulative_deposition_kg", 1.0 + 1e-9), base_summary())
    report = H.compare(cpu, outside)
    assert not report["fields"]["cumulative_deposition_kg"]["pass"] and not report["accepted_by_declared_bounds"]
    water = write_run(tmp_path / "water", perturbed(arrays, "final_depth_m", 1.0 + 1e-10), base_summary())
    assert not H.compare(cpu, water)["fields"]["final_depth_m"]["pass"]  # the water bound (2e-12) is tighter than the sediment bound


@pytest.mark.parametrize("key", ["cumulative_detachment_kg", "final_mobile_kg", "final_depth_m", "water_outlet_m3_s", "ledger"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nan_and_inf_never_pass(tmp_path, key, bad):
    arrays = base_arrays()
    cpu = write_run(tmp_path / "cpu", arrays, base_summary()) if key != "ledger" else None
    broken = {k: v.copy() for k, v in arrays.items()}
    broken[key].reshape(-1)[0] = bad
    if key == "ledger":  # the identity guard sees the non-finite ledger; write the run without recomputing the residual
        directory = tmp_path / "gpu"
        directory.mkdir()
        stored = dict(broken)
        stored["identity_residual_kg"] = residual_of(arrays["ledger"])
        np.savez(directory / "legacy_ledger.npz", **stored)
        (directory / "legacy_summary.json").write_text(json.dumps(base_summary()))
        cpu = write_run(tmp_path / "cpu", arrays, base_summary())
        gpu = directory
    else:
        gpu = write_run(tmp_path / "gpu", broken, base_summary())
    report = H.compare(cpu, gpu)
    assert not report["accepted_by_declared_bounds"] and report["flags"]


def test_a_nan_in_the_reference_also_fails(tmp_path):
    arrays = base_arrays()
    broken = {k: v.copy() for k, v in arrays.items()}
    broken["cumulative_rain_m"].reshape(-1)[2] = np.nan
    cpu = write_run(tmp_path / "cpu", broken, base_summary())
    gpu = write_run(tmp_path / "gpu", broken, base_summary())  # NaN equal to NaN must still not be accepted
    assert not H.compare(cpu, gpu)["accepted_by_declared_bounds"]


def test_field_report_flags_nan_shape_and_dtype_directly():
    a = np.array([1.0, np.nan])
    assert not H.field_report(a, a, 1e-3, 1e-3)["pass"]
    assert not H.field_report(np.zeros(3), np.zeros(4), 1.0, 1.0)["pass"]
    assert not H.field_report(np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64), 1.0, 1.0)["pass"]
    assert H.field_report(np.array([1.0]), np.array([1.0 + 1e-13]), 2e-12, 1e-14)["pass"]
    assert not H.exact_report(np.array([1.0, np.nan]), np.array([1.0, np.nan]))["pass"]


@pytest.mark.parametrize("missing", ["final_soil_water_m", "cumulative_drainage_m", "water_export_m3", "identity_residual_kg", "outlet"])
def test_a_missing_required_field_is_a_failure_not_a_skip(tmp_path, missing):
    arrays = base_arrays()
    cpu = write_run(tmp_path / "cpu", arrays, base_summary())
    gpu = write_run(tmp_path / "gpu", arrays, base_summary())
    with np.load(gpu / "legacy_ledger.npz") as z:
        kept = {k: z[k] for k in z.files if k != missing}
    np.savez(gpu / "legacy_ledger.npz", **kept)
    report = H.compare(cpu, gpu)
    assert not report["accepted_by_declared_bounds"] and any("lacks required arrays" in f for f in report["flags"])


def test_an_unreadable_or_truncated_artifact_is_a_flag(tmp_path):
    cpu, gpu = pair(tmp_path)
    data = (gpu / "legacy_ledger.npz").read_bytes()
    (gpu / "legacy_ledger.npz").write_bytes(data[:len(data) // 2])
    report = H.compare(cpu, gpu)
    assert not report["accepted_by_declared_bounds"] and any("unusable artifact" in f for f in report["flags"])
    assert not H.compare(tmp_path / "nowhere", gpu)["accepted_by_declared_bounds"]


@pytest.mark.parametrize("edit", [
    lambda s: s["hydrology"].update(root_solver="newton"), lambda s: s.update(depth_time_level="current"),
    lambda s: s.update(dt_s=2.0), lambda s: s.update(end_s=6.0), lambda s: s["config"].update(source_order="legacy"),
    lambda s: s["config"].update(legacy_depos_erase=True), lambda s: s.update(case_kind="rfid"),
    lambda s: s["hydrology"].update(bisection_iterations=41), lambda s: s["provenance"].update(graph_input_sha256="cd" * 32),
    lambda s: s["artifact_pins"]["sha256"].update({"/a": "33" * 32}), lambda s: s["config"].update(snapshot_times="2,4"),
    lambda s: s["law_depth"].update(selected="post_infiltration"), lambda s: s.update(network={"n_active": 99}),
    lambda s: s.pop("hydrology"), lambda s: s.update(steps=STEPS + 1),
])
def test_incompatible_receipts_are_refused_before_any_numeric_comparison(tmp_path, edit):
    summary = copy.deepcopy(base_summary())
    edit(summary)
    cpu, gpu = pair(tmp_path, gpu_summary=summary)
    report = H.compare(cpu, gpu)
    assert not report["accepted_by_declared_bounds"] and report["flags"] and not report["fields"]  # no numeric comparison was made


def test_backend_and_code_differences_do_not_matter(tmp_path):
    gpu_summary = base_summary(backend="cuda")
    gpu_summary["config"].update(check_every_steps=60, cuda_mode="auto", cn_mode="auto", record_strategy="compact", output="/elsewhere")
    gpu_summary["performance"]["loop_wall_s_excluding_progress"] = 0.1
    cpu, gpu = pair(tmp_path, gpu_summary=gpu_summary)
    report = H.compare(cpu, gpu)
    assert report["accepted_by_declared_bounds"] and report["timing"]["speedup_loop"] == pytest.approx(20.0)


def snapshots(depth_factor=1.0, mobile_factor=1.0, times=(2.0, 4.0)):
    rng = np.random.default_rng(4)
    depth, mobile = rng.uniform(0, 1, (2, NY, NX)), rng.uniform(0, 1, (2, NY, NX, NC))
    return {"t_s": np.array(times), "depth_m": depth * depth_factor, "mobile_kg": mobile * mobile_factor}


def test_requested_snapshots_are_compared_and_must_exist(tmp_path):
    summary = base_summary()
    summary["config"]["snapshot_times"] = "2,4"
    cpu = write_run(tmp_path / "cpu", base_arrays(), summary, snapshots())
    ok = write_run(tmp_path / "ok", base_arrays(), copy.deepcopy(summary), snapshots())
    assert H.compare(cpu, ok)["accepted_by_declared_bounds"]
    for name, snaps in (("depth", snapshots(depth_factor=1.0 + 1e-9)), ("mobile", snapshots(mobile_factor=1.0 + 1e-9)),
                        ("time", snapshots(times=(2.0, 5.0)))):
        bad = write_run(tmp_path / f"bad_{name}", base_arrays(), copy.deepcopy(summary), snaps)
        report = H.compare(cpu, bad)
        assert not report["accepted_by_declared_bounds"], name
    none = write_run(tmp_path / "none", base_arrays(), copy.deepcopy(summary))
    assert any("snapshot archive is missing" in f for f in H.compare(cpu, none)["flags"])
    plain_cpu = write_run(tmp_path / "plain_cpu", base_arrays(), base_summary())
    unexpected = write_run(tmp_path / "unexpected", base_arrays(), base_summary(), snapshots())
    assert not H.compare(plain_cpu, unexpected)["accepted_by_declared_bounds"]


@pytest.mark.parametrize("key", ["walk_counts", "regime_counts", "active", "outlet", "terminal_storage", "t_s"])
def test_integer_and_control_arrays_are_exact(tmp_path, key):
    arrays = base_arrays()
    cpu = write_run(tmp_path / "cpu", arrays, base_summary())
    other = {k: v.copy() for k, v in arrays.items()}
    flat = other[key].reshape(-1)
    flat[0] = (not flat[0]) if flat.dtype == bool else flat[0] + 1
    gpu = write_run(tmp_path / "gpu", other, base_summary())
    assert not H.compare(cpu, gpu)["accepted_by_declared_bounds"]


def test_identity_residual_is_a_separate_diagnostic_checked_against_the_unchanged_guard(tmp_path):
    arrays = base_arrays()
    cpu = write_run(tmp_path / "cpu", arrays, base_summary())
    # a ledger that violates the 1e-10 identity on one backend is refused on its own
    broken = {k: v.copy() for k, v in arrays.items()}
    broken["ledger"][2, LEDGER_COLUMNS.index("new_mobile_kg"), 1] += 1e-3
    directory = tmp_path / "gpu"
    directory.mkdir()
    np.savez(directory / "legacy_ledger.npz", **broken, identity_residual_kg=residual_of(arrays["ledger"]))
    (directory / "legacy_summary.json").write_text(json.dumps(base_summary()))
    report = H.compare(cpu, directory)
    assert any("identity" in f for f in report["flags"]) and not report["accepted_by_declared_bounds"]
    # a stored residual that does not equal the one recomputed from the run's own ledger is refused
    tampered = {k: v.copy() for k, v in arrays.items()}
    directory2 = tmp_path / "gpu2"
    directory2.mkdir()
    np.savez(directory2 / "legacy_ledger.npz", **tampered, identity_residual_kg=residual_of(arrays["ledger"]) + 1e-20)
    (directory2 / "legacy_summary.json").write_text(json.dumps(base_summary()))
    assert any("recomputed" in f for f in H.compare(cpu, directory2)["flags"])


def test_the_cross_backend_identity_difference_is_bounded_only_by_the_propagated_declared_bounds(tmp_path):
    arrays = base_arrays()
    cpu = write_run(tmp_path / "cpu", arrays, base_summary())
    col = LEDGER_COLUMNS.index("new_mobile_kg")
    within = {k: v.copy() for k, v in arrays.items()}
    within["ledger"][:, col] *= 1.0 + 1e-12  # inside the declared 2e-11 ledger bound: the residual difference is inside the propagated bound
    ok = write_run(tmp_path / "ok", within, base_summary())
    report = H.compare(cpu, ok)
    assert report["identity_diagnostic"].get("pass") is True, report["identity_diagnostic"]
    # a ledger shifted coherently by far more than the declared bound keeps each backend's own identity intact but fails the ledger
    # field: the identity diagnostic never rescues a physical field
    far ={k: v.copy() for k, v in arrays.items()}
    far["ledger"][:, col] *= 1.0 + 1e-8
    far["ledger"][:, LEDGER_COLUMNS.index("old_mobile_kg")] *= 1.0 + 1e-8  # keep each backend's own identity inside the 1e-10 guard
    far["ledger"][:, LEDGER_COLUMNS.index("pickup_kg")] *= 1.0 + 1e-8
    far["ledger"][:, LEDGER_COLUMNS.index("deposition_active_kg")] *= 1.0 + 1e-8
    far["ledger"][:, LEDGER_COLUMNS.index("cn_export_kg")] *= 1.0 + 1e-8
    far["ledger"][:, LEDGER_COLUMNS.index("effective_clip_source_kg")] *= 1.0 + 1e-8
    bad = write_run(tmp_path / "far", far, base_summary())
    rep = H.compare(cpu, bad)
    assert not rep["accepted_by_declared_bounds"] and not rep["fields"]["ledger"]["pass"]


def test_the_gpu_repeat_check_covers_every_saved_array_snapshot_and_counter(tmp_path):
    summary = base_summary()
    summary["config"]["snapshot_times"] = "2,4"
    arrays = base_arrays()
    cpu = write_run(tmp_path / "cpu", arrays, summary, snapshots())
    gpu = write_run(tmp_path / "gpu", arrays, copy.deepcopy(summary), snapshots())
    same = write_run(tmp_path / "same", arrays, copy.deepcopy(summary), snapshots())
    report = H.compare(cpu, gpu, same)
    assert report["accepted_by_declared_bounds"] and all(report["gpu_repeatability_bitwise"]["arrays"].values())
    assert set(report["gpu_repeatability_bitwise"]["arrays"]) >= set(H.WATER_KEYS) | set(H.INTEGER_KEYS) | {"t_s", "walk_counts"}
    for key in ("final_depth_m", "water_budget_residual_m3", "t_s", "walk_counts", "cumulative_rain_m"):
        other = {k: v.copy() for k, v in arrays.items()}
        flat = other[key].reshape(-1)
        flat[0] = np.nextafter(flat[0], np.inf) if flat.dtype.kind == "f" else flat[0] + 1
        rep_dir = write_run(tmp_path / f"rep_{key}", other, copy.deepcopy(summary), snapshots())
        result = H.compare(cpu, gpu, rep_dir)
        assert not result["accepted_by_declared_bounds"] and any(key in f for f in result["flags"]), key
    bad_snap = write_run(tmp_path / "rep_snap", arrays, copy.deepcopy(summary), snapshots(depth_factor=np.nextafter(1.0, 2.0)))
    assert any("snapshot field depth_m" in f for f in H.compare(cpu, gpu, bad_snap)["flags"])
    changed = copy.deepcopy(summary)
    changed["walk_tallies"] = {"walks": 6}
    bad_counter = write_run(tmp_path / "rep_counter", arrays, changed, snapshots())
    assert any("walk_tallies" in f for f in H.compare(cpu, gpu, bad_counter)["flags"])
    incompatible = copy.deepcopy(summary)
    incompatible["hydrology"]["root_solver"] = "newton"
    bad_root = write_run(tmp_path / "rep_root", arrays, incompatible, snapshots())
    assert any("repeat run" in f for f in H.compare(cpu, gpu, bad_root)["flags"])


def test_unpinned_legacy_outputs_are_accepted_but_qualified_unbound_and_pinned_ones_are_verified(tmp_path):
    cpu, gpu = pair(tmp_path)  # no output_pins: older B1/CPU receipts
    report = H.compare(cpu, gpu)
    assert report["accepted_by_declared_bounds"] and report["output_pins"]["gpu"]["verified"] is False
    assert "unbound" in report["output_pins"]["gpu"]["status"] and "unbound" in report["scope"]
    pin(gpu)
    report = H.compare(cpu, gpu)
    assert report["accepted_by_declared_bounds"] and report["output_pins"]["gpu"]["verified"] is True
    assert report["output_pins"]["cpu"]["verified"] is False  # each side is qualified separately


def test_pin_fields_record_size_hash_and_members(tmp_path):
    from maple_syrup import legacy_gpu_driver as G

    gpu = write_run(tmp_path / "gpu", base_arrays(), base_summary(), snapshots())
    pins = G._output_pins(gpu, ("legacy_ledger.npz", "legacy_snapshots.npz", "absent.npz"))
    assert set(pins["files"]) == {"legacy_ledger.npz", "legacy_snapshots.npz"}  # an absent optional archive is simply not listed
    ledger = pins["files"]["legacy_ledger.npz"]
    assert ledger["bytes"] == (gpu / "legacy_ledger.npz").stat().st_size and len(ledger["sha256"]) == 64
    assert "ledger.npy" in ledger["members"] and "summary_file" in pins


@pytest.mark.parametrize("tamper", ["truncate", "flip_byte", "append", "swap_pin_hash", "drop_pin", "unlisted_snapshots", "missing_file"])
def test_tampered_bodies_sizes_and_pins_are_rejected(tmp_path, tamper):
    summary = base_summary()
    cpu = write_run(tmp_path / "cpu", base_arrays(), summary)
    gpu = write_run(tmp_path / "gpu", base_arrays(), copy.deepcopy(summary))
    path = gpu / "legacy_ledger.npz"
    if tamper == "unlisted_snapshots":
        pin(gpu, names=("legacy_ledger.npz",))
        np.savez(gpu / "legacy_snapshots.npz", **snapshots())
    else:
        pinned = pin(gpu)
    if tamper == "truncate":
        data = path.read_bytes()
        path.write_bytes(data[:len(data) - 7])
    elif tamper == "flip_byte":
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0x01  # the size is unchanged: only the hash can see it
        path.write_bytes(bytes(data))
    elif tamper == "append":
        path.write_bytes(path.read_bytes() + b"\0")
    elif tamper == "swap_pin_hash":
        pinned["output_pins"]["files"]["legacy_ledger.npz"]["sha256"] = "0" * 64
        (gpu / "legacy_summary.json").write_text(json.dumps(pinned))
    elif tamper == "drop_pin":
        del pinned["output_pins"]["files"]["legacy_ledger.npz"]
        (gpu / "legacy_summary.json").write_text(json.dumps(pinned))
    elif tamper == "missing_file":
        path.unlink()
    report = H.compare(cpu, gpu)
    assert not report["accepted_by_declared_bounds"] and any("unusable artifact" in f for f in report["flags"]), tamper


def test_a_pinned_snapshot_archive_is_verified_too_and_the_marker_alone_is_never_trusted(tmp_path):
    summary = base_summary()
    summary["config"]["snapshot_times"] = "2,4"
    cpu = write_run(tmp_path / "cpu", base_arrays(), summary, snapshots())
    gpu = write_run(tmp_path / "gpu", base_arrays(), copy.deepcopy(summary), snapshots())
    pin(gpu, summary_over={"output_pins_ok_marker": True, "artifact_pins": {"checked_unchanged_before_publish": True,
                                                                           "sha256": summary["artifact_pins"]["sha256"]}})
    assert H.compare(cpu, gpu)["accepted_by_declared_bounds"]
    snap = gpu / "legacy_snapshots.npz"
    data = bytearray(snap.read_bytes())
    data[len(data) // 2] ^= 0x01
    snap.write_bytes(bytes(data))  # the receipt still carries every 'ok' marker
    report = H.compare(cpu, gpu)
    assert not report["accepted_by_declared_bounds"] and any("SHA-256" in f for f in report["flags"])


def test_the_driver_summary_carries_the_qualification_and_pins_are_taken_after_the_archives_are_closed():
    import inspect

    from maple_syrup import legacy_gpu_driver as G

    source = inspect.getsource(G._execute)
    assert source.index("np.savez(out_dir / \"legacy_snapshots.npz\"") < source.index("_output_pins(out_dir") < source.index("summary = {")
    assert source.index("_output_pins(out_dir") < source.index("write_text(json.dumps(summary")  # the summary is written last
    assert '"qualification"' in source and '"output_pins": output_pins' in source


def test_main_refuses_to_overwrite_and_returns_nonzero_for_a_rejected_run(tmp_path, capsys):
    cpu, gpu = pair(tmp_path)
    out = tmp_path / "report.json"
    assert H.main(["--cpu", str(cpu), "--gpu", str(gpu), "--output", str(out)]) == 0
    assert H.main(["--cpu", str(cpu), "--gpu", str(gpu), "--output", str(out)]) == 1  # never overwrites
    bad = write_run(tmp_path / "bad", perturbed(base_arrays(), "final_depth_m", 1.1), base_summary())
    assert H.main(["--cpu", str(cpu), "--gpu", str(bad), "--output", str(tmp_path / "r2.json")]) == 2
    capsys.readouterr()
