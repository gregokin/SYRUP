"""Capture parsing, orientation/unit conversion, conductivity validation, static-consistency gates, safe outputs and
run-record verification on synthetic data in the exact format of the Fortran hook. No model is executed.
Nothing here was run by its author (file-only tools); Codex records actual results."""

from __future__ import annotations

import hashlib
import json

import capture_data as cd
import numpy as np
import pytest
from capture_fixture import fortran_float, render_capture, render_steps, synthetic_setup


# --- parsing ---------------------------------------------------------------------------------------------------
def test_static_roundtrip_reads_three_digit_exponents_exactly():
    s = synthetic_setup()
    assert "E-004" in s["text"] or "E+000" in s["text"]  # Fortran ES25.16E3 style, not Python's two digits
    parsed = s["static"]
    assert parsed.scalars["nr"] == 4 and isinstance(parsed.scalars["nr"], int)
    assert parsed.scalars["dx_mm"] == 500.0 and isinstance(parsed.scalars["dx_mm"], float)
    for name, array in s["arrays"].items():
        np.testing.assert_array_equal(parsed.arrays[name], array)  # 17 significant digits round-trip bitwise
    assert parsed.arrays["order"].dtype == np.int64 and parsed.arrays["ksat"].dtype == np.float64
    value = 2.5e-4
    assert float(fortran_float(value)) == value


@pytest.mark.parametrize("mutate, message", [
    (lambda t: t.replace(f"{cd.COMPLETE} static\n", ""), "truncated"),
    (lambda t: t.replace(cd.MAGIC, "OTHER", 1), "not a"),
    (lambda t: t.replace("scalar nc 3", "scalar nr 3", 1), "duplicate"),
    (lambda t: t.replace("scalar dx_mm", "scalar dx_mm NaN\nscalar dx_dup", 1), "non-finite|not a number|unrecognised"),
    (lambda t: t.replace("array ksat d 5 4", "array ksat d 6 4", 1), "truncated rows|bad shape|row has|not a number"),
    (lambda t: t.replace("scalar dt_s", "bogus line\nscalar dt_s", 1), "unrecognised"),
])
def test_malformed_static_captures_are_refused(mutate, message):
    with pytest.raises(cd.CaptureError, match=message):
        cd.parse_capture_text(mutate(synthetic_setup()["text"]), "static")


def test_a_row_of_the_wrong_width_and_infinity_are_refused():
    text = synthetic_setup()["text"]
    lines = text.splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith("array ksat")) + 1
    short = lines.copy()
    short[k] = " ".join(short[k].split()[:-1])
    with pytest.raises(cd.CaptureError, match="row has"):
        cd.parse_capture_text("\n".join(short) + "\n", "static")
    bad = lines.copy()
    bad[k] = bad[k].replace(bad[k].split()[0], "Infinity", 1)
    with pytest.raises(cd.CaptureError, match="non-finite"):
        cd.parse_capture_text("\n".join(bad) + "\n", "static")


def test_steps_history_is_validated():
    rval = [0.01, 0.01, 0.02]
    cols = cd.parse_steps_text(render_steps(rval, q_single=[0.0, 1.0, 2.0], q_double=[0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(cols["iter"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(cols["rval_applied_mm_s"], rval)
    assert set(cols) == set(cd.STEP_COLUMNS)
    text = render_steps(rval)
    with pytest.raises(cd.CaptureError, match="truncated"):
        cd.parse_steps_text(text.replace(f"{cd.COMPLETE} steps\n", ""))
    with pytest.raises(cd.CaptureError, match="columns"):
        cd.parse_steps_text(text.replace("rval_applied_mm_s", "rain", 1))
    lines = text.splitlines()
    gap = lines.copy()
    gap[3] = gap[3].replace("2 ", "7 ", 1) if gap[3].split()[0] == "2" else gap[3]
    with pytest.raises(cd.CaptureError, match="iterations"):
        cd.parse_steps_text("\n".join(gap) + "\n")
    with pytest.raises(cd.CaptureError, match="negative"):
        cd.parse_steps_text(render_steps([0.01, -0.01]))
    with pytest.raises(cd.CaptureError, match="increasing"):
        cd.parse_steps_text(render_steps(rval, dt=0.0))


def test_integer_arrays_refuse_fractional_and_exponent_tokens_and_order_indices_are_bounds_checked():
    s = synthetic_setup()
    lines = s["text"].splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith("array aspect")) + 1
    for token in ("3.5", "3.0", "3.0E+000"):
        bad = lines.copy()
        bad[k] = token + " " + " ".join(bad[k].split()[1:])
        with pytest.raises(cd.CaptureError, match="not an integer"):
            cd.parse_capture_text("\n".join(bad) + "\n", "static")
    nr, nc = s["nr"], s["nc"]
    good = s["arrays"]["order"]
    assert cd.validate_order(good, nr, nc) is not None
    for mutate in (lambda o: o.__setitem__((0, 0), 0), lambda o: o.__setitem__((0, 0), nr + 2),
                   lambda o: o.__setitem__((1, 1), nc + 2), lambda o: o.__setitem__((1, 1), -3)):
        bad_order = good.copy()
        mutate(bad_order)
        with pytest.raises(cd.CaptureError, match="outside"):
            cd.validate_order(bad_order, nr, nc)
        arrays = {**s["arrays"], "order": bad_order}
        static = cd.parse_capture_text(render_capture("static", s["scalars"], arrays), "static")
        with pytest.raises(cd.CaptureError, match="outside"):  # refused before any rmask/aspect indexing
            cd.check_static_consistency(static, s["ref"])
    for bad_shape in (good[:, :2], good.astype(np.float64)):
        with pytest.raises(cd.CaptureError, match="integer"):
            cd.validate_order(bad_shape, nr, nc)


# --- orientation and units --------------------------------------------------------------------------------------------
def test_interior_conversion_crops_the_ring_and_reverses_rows_to_south_first():
    nr, nc = 4, 3
    full = np.arange((nr + 1) * (nc + 1), dtype=np.float64).reshape(nr + 1, nc + 1)  # Fortran (i, j), i north -> south
    interior = cd.interior_south_first(full, nr, nc)
    assert interior.shape == (3, 2)
    np.testing.assert_array_equal(interior[0], full[3, 1:3])  # SYRUP row 0 = southernmost interior = Fortran i = nr
    np.testing.assert_array_equal(interior[-1], full[1, 1:3])  # last SYRUP row = Fortran i = 2
    assert interior.base is None or not np.shares_memory(interior, full)  # a copy
    with pytest.raises(cd.CaptureError, match="shape"):
        cd.interior_south_first(full[:-1], nr, nc)


def test_synthetic_fields_convert_back_to_the_syrup_arrays():
    s = synthetic_setup()
    got = cd.interior_south_first(s["static"].arrays["ksat"], s["nr"], s["nc"])
    np.testing.assert_array_equal(got, s["ksat_mm"])  # flip, crop and the 17-digit text are exact


def test_legacy_outlet_mask_matches_the_output_routine_condition_for_all_four_aspects():
    nr, nc = 5, 5  # interior i, j = 2..5 (0-based 1..4)
    rmask = np.ones((nr + 1, nc + 1))
    rmask[0, :] = rmask[-1, :] = rmask[:, 0] = rmask[:, -1] = -9999.0
    aspect = np.zeros(rmask.shape, dtype=np.int64)
    aspect[1:nr, 1:nc] = 3  # south
    aspect[1, 1] = 1  # north-west corner flows north into the ring
    aspect[1, 4] = 2  # north-east corner flows east into the ring
    aspect[4, 1] = 4  # south-west corner flows west into the ring
    aspect[2, 2] = 1  # interior cell flowing north into an active cell: not an outlet
    out = cd.legacy_outlet_mask(aspect, rmask, nr, nc)
    expected = np.zeros(rmask.shape, dtype=bool)
    expected[4, 1:5] = True  # south row flows south into the ring (includes the west corner flowing west)
    expected[1, 1] = expected[1, 4] = True
    np.testing.assert_array_equal(out, expected)
    assert not out[0].any() and not out[-1].any() and not out[:, 0].any() and not out[:, -1].any()
    rmask[3, 2] = -1.0  # an inactive interior cell is never an outlet and a neighbouring receiver may be inactive
    aspect[3, 2] = 3
    assert not cd.legacy_outlet_mask(aspect, rmask, nr, nc)[3, 2]


# --- conductivity ---------------------------------------------------------------------------------------------------------
def test_conductivity_realization_is_validated_and_converted():
    s = synthetic_setup()
    k = cd.validate_ksat_mm_s(s["static"].arrays["ksat"], s["nr"], s["nc"])
    np.testing.assert_array_equal(k, s["ksat_mm"])
    full = s["static"].arrays["ksat"].copy()
    for value, match in ((0.0, "non-positive"), (-1.0e-4, "non-positive"), (np.nan, "non-finite"), (np.inf, "non-finite")):
        bad = full.copy()
        bad[2, 2] = value
        with pytest.raises(cd.CaptureError, match=match):
            cd.validate_ksat_mm_s(bad, s["nr"], s["nc"])
    bad = full.copy()
    bad[0, :] = 0.0  # the exterior ring is not part of the realization
    cd.validate_ksat_mm_s(bad, s["nr"], s["nc"])
    with pytest.raises(cd.CaptureError, match="shape"):
        cd.validate_ksat_mm_s(full[:, :-1], s["nr"], s["nc"])


def test_inject_ksat_converts_units_copies_and_never_mutates_its_inputs():
    s = synthetic_setup()
    host = {"ksat_m_per_s": np.full(s["ksat_mm"].shape, 2.5e-7), "theta_sat": np.full(s["ksat_mm"].shape, 0.4),
            "initial_theta": np.full(s["ksat_mm"].shape, 0.25), "note": "kept"}
    snapshot = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in host.items()}
    k = s["ksat_mm"].copy()
    out = cd.inject_ksat(host, k)
    np.testing.assert_array_equal(out["ksat_m_per_s"], s["ksat_mm"] * 1.0e-3)
    assert out["ksat_m_per_s"].shape == host["ksat_m_per_s"].shape and out["note"] == "kept"
    for name, value in snapshot.items():  # the caller's dict and arrays are unchanged
        assert (np.array_equal(host[name], value) if isinstance(value, np.ndarray) else host[name] == value)
    assert out["theta_sat"] is not host["theta_sat"]
    k[0, 0] = -1.0  # later mutation of the source does not reach the injected copy
    assert out["ksat_m_per_s"][0, 0] > 0.0
    for bad in (k.T, np.where(s["ksat_mm"] > 0.0, 0.0, 1.0), np.full(k.shape, np.nan)):
        with pytest.raises(cd.CaptureError):
            cd.inject_ksat(host, bad)


# --- consistency gates ------------------------------------------------------------------------------------------------------
def test_consistent_setup_passes_every_gate_and_reports_them():
    s = synthetic_setup()
    report = cd.check_static_consistency(s["static"], s["ref"])
    assert all(entry["pass"] for entry in report.values() if isinstance(entry, dict))
    assert report["outlet_cells"] == {"legacy": 2, "syrup": 2, "pass": True}
    assert report["routing_order"]["n_active"] == 6 and report["routing_order"]["pass"]
    assert report["rainfall_scale"]["max_abs_difference"] == 0.0


def _with(s, **changes):
    scalars = {**s["scalars"], **{k: v for k, v in changes.items() if k in s["scalars"]}}
    arrays = {**s["arrays"], **{k: v for k, v in changes.items() if k in s["arrays"]}}
    return cd.parse_capture_text(render_capture("static", scalars, arrays), "static")


@pytest.mark.parametrize("name, mutate, match", [
    ("pave", lambda a: a * 1.001, "pavement_fraction"),
    ("psi", lambda a: a * 1.0001, "suction_m"),
    ("theta_sat", lambda a: a + 1e-6, "theta_sat"),
    ("stmax", lambda a: a * 1.0001, "storage_max_m"),
    ("drain_par", lambda a: a * 1.0001, "drainage_parameter"),
    ("cum_inf", lambda a: a * 1.0001, "initial_soil_water_m"),
    ("theta", lambda a: a * 1.0001, "initial_theta"),
    ("slope", lambda a: a * 1.001, "slope"),
    ("ff", lambda a: a * 1.001, "friction_factor"),
    ("d_initial_mm", lambda a: a + 1.0, "dry start"),
])
def test_inconsistent_setup_is_refused_before_any_simulation(name, mutate, match):
    s = synthetic_setup()
    bad = _with(s, **{name: mutate(s["arrays"][name])})
    with pytest.raises(cd.CaptureError, match=match):
        cd.check_static_consistency(bad, s["ref"])


@pytest.mark.parametrize("scalar, value, match", [
    ("dt_s", 0.5, "dt_s"), ("iroute", 2, "iroute"), ("inf_model", 1, "inf_model"), ("ksat_mod", 0.9, "ksat_mod"),
    ("rain_type", 1, "rain_type"), ("dx_mm", 250.0, "dx_m"),
])
def test_wrong_configuration_scalars_are_refused(scalar, value, match):
    s = synthetic_setup()
    with pytest.raises(cd.CaptureError, match=match):
        cd.check_static_consistency(_with(s, **{scalar: value}), s["ref"])


def test_a_missing_row_flip_or_wrong_routing_is_detected():
    s = synthetic_setup()
    ref = dict(s["ref"])
    ref["slope"] = s["ref"]["slope"][::-1]  # the north-first order would be silently accepted without the conversion
    with pytest.raises(cd.CaptureError, match="slope"):
        cd.check_static_consistency(s["static"], ref)
    ref = dict(s["ref"])
    ref["outlet"] = s["ref"]["outlet"][::-1]
    with pytest.raises(cd.CaptureError, match="outlet"):
        cd.check_static_consistency(s["static"], ref)
    arrays = dict(s["arrays"])
    arrays["order"] = s["arrays"]["order"][:-1]
    scalars = {**s["scalars"], "ncell1": s["scalars"]["ncell1"] - 1}
    with pytest.raises(cd.CaptureError, match="permutation"):
        cd.check_static_consistency(cd.parse_capture_text(render_capture("static", scalars, arrays), "static"), s["ref"])
    arrays = dict(s["arrays"])
    arrays["rmask"] = s["arrays"]["rmask"].copy()
    arrays["rmask"][0, 0] = 1.0  # ring cell that is not an export cell
    with pytest.raises(cd.CaptureError, match="full_rainfall_mask"):
        cd.check_static_consistency(cd.parse_capture_text(render_capture("static", s["scalars"], arrays), "static"),
                                    {**s["ref"], "legacy_full_rmask": s["arrays"]["rmask"].copy()})


# --- safe outputs -------------------------------------------------------------------------------------------------------------
def test_unsafe_output_paths_are_refused(tmp_path):
    protected = tmp_path / "protected"
    (protected / "deep").mkdir(parents=True)
    existing = tmp_path / "existing"
    existing.mkdir()
    link = tmp_path / "link"
    link.symlink_to(protected)
    ok = cd.refuse_output(tmp_path / "new", {"p": protected, "none": None})
    assert ok == (tmp_path / "new").resolve() and not ok.exists()
    for bad, match in ((existing, "existing"), (protected, "inside the p tree"), (protected / "deep" / "x", "inside the p tree"),
                       (link / "x", "inside the p tree"), (link, "inside the p tree")):
        with pytest.raises(cd.CaptureError, match=match):
            cd.refuse_output(bad, {"p": protected})
    # an output that would contain a protected tree (neither exists here, so only the containment rule can fire)
    with pytest.raises(cd.CaptureError, match="contains the p tree"):
        cd.refuse_output(tmp_path / "outer", {"p": tmp_path / "outer" / "inner"})
    assert not (tmp_path / "new").exists() and not (tmp_path / "outer").exists()  # the check itself writes nothing


def _fake_run(root, *, mutate=None):
    out = root / "Output"
    out.mkdir(parents=True)
    outputs = {}
    for name in cd.CAPTURE_FILES:
        path = root / name
        path.write_text(f"content of {name}\n")
        outputs[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    (root / "mahleran_input.xml").write_text("<xml/>\n")
    record = {"returncode": 0, "completion_marker": True, "input_unchanged": True, "reference_unchanged": True,
              "prepared_unchanged": True, "outputs": outputs, "executable_sha256": "ab" * 32,
              "input_sha256": {"mahleran_input.xml": hashlib.sha256(b"<xml/>\n").hexdigest()}}
    if mutate:
        mutate(record, root)
    (root / "execution.json").write_text(json.dumps(record))
    return root


def test_run_record_verification_detects_tampering_and_incomplete_runs(tmp_path):
    ok = _fake_run(tmp_path / "ok")
    assert cd.verify_capture_run(ok)["executable_sha256"] == "ab" * 32
    tampered = _fake_run(tmp_path / "t")
    (tampered / cd.CAPTURE_FILES[0]).write_text("changed\n")
    with pytest.raises(cd.CaptureError, match="no longer matches"):
        cd.verify_capture_run(tampered)
    for key, value in (("returncode", 1), ("completion_marker", False), ("prepared_unchanged", False)):
        run = _fake_run(tmp_path / key, mutate=lambda r, _root, key=key, value=value: r.__setitem__(key, value))
        with pytest.raises(cd.CaptureError):
            cd.verify_capture_run(run)
    inputs = _fake_run(tmp_path / "in")
    (inputs / "mahleran_input.xml").write_text("<changed/>\n")  # a saved run INPUT changed after the run
    with pytest.raises(cd.CaptureError, match="saved run input"):
        cd.verify_capture_run(inputs)
    no_inputs = _fake_run(tmp_path / "noin", mutate=lambda r, _root: r.pop("input_sha256"))
    with pytest.raises(cd.CaptureError, match="no input_sha256"):
        cd.verify_capture_run(no_inputs)
    before = cd.capture_digest(ok)
    assert set(before) == {*cd.CAPTURE_FILES, "execution.json"} and cd.capture_digest(ok) == before
    missing = _fake_run(tmp_path / "m", mutate=lambda r, _root: r["outputs"].pop(cd.CAPTURE_FILES[1]))
    with pytest.raises(cd.CaptureError, match="not among the recorded outputs"):
        cd.verify_capture_run(missing)
    with pytest.raises(cd.CaptureError, match="execution.json"):
        cd.verify_capture_run(tmp_path / "absent")


def test_legacy_soil_binding_matches_real32_thickness_without_mutating_inputs():
    setup = synthetic_setup()
    host = {"soil_thickness_m": setup["ref"]["soil_thickness_m"].copy(),
            "theta_sat": setup["ref"]["theta_sat"].copy(),
            "initial_theta": setup["ref"]["initial_theta"].copy()}
    before = {k: v.copy() for k, v in host.items()}
    arrays = {k: v.copy() for k, v in setup["arrays"].items()}
    arrays["stmax"] = arrays["theta_sat"] * float(np.float32(0.3)) * 1000.
    arrays["cum_inf"] = arrays["theta"] * float(np.float32(0.3)) * 1000.
    static = cd.Capture("static", setup["scalars"], arrays)
    bound, soil, record = cd.bind_legacy_soil_initialization(static, host)
    np.testing.assert_allclose(bound["soil_thickness_m"], float(np.float32(0.3)), rtol=1e-15)
    np.testing.assert_allclose(soil, 0.25 * float(np.float32(0.3)), rtol=1e-15)
    assert record["legacy_storage_initialization"]["pass"]
    for k, value in host.items():
        np.testing.assert_array_equal(value, before[k])
    arrays["cum_inf"][1, 1] *= 1.001
    with pytest.raises(cd.CaptureError, match="initialization mismatch"):
        cd.bind_legacy_soil_initialization(static, host)


def test_closed_zero_ring_cells_are_valid_and_exact_mask_binding_is_enforced():
    setup = synthetic_setup()
    arrays = {k: v.copy() for k, v in setup["arrays"].items()}
    arrays["rmask"][0, 0] = 0.
    ref = dict(setup["ref"], legacy_full_rmask=arrays["rmask"].copy())
    static = cd.Capture("static", setup["scalars"], arrays)
    assert cd.check_static_consistency(static, ref)["ring_cells"]["closed_zero"] == 1
    arrays["rmask"][0, 0] = -9999.
    with pytest.raises(cd.CaptureError, match="full_rainfall_mask"):
        cd.check_static_consistency(static, ref)
