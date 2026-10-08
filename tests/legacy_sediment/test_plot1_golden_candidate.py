"""CPU tests of the isolated candidate `plot1_golden` helper (agent_handoffs/tasks/gpu_sediment/golden_candidate/compare_legacy_sediment.py).

EXPERIMENTAL LOCAL DEPENDENCE: the candidate helper lives under `agent_handoffs/`, which is git-ignored, so it is NOT part of a clean
checkout. It is a local, not-adopted candidate (the committed benchmark helper `benchmarks/legacy_sediment/compare_legacy_sediment.py` is
unchanged; see docs/legacy_gpu/current_status.md). When the candidate file is absent this whole module skips at collection time instead of
failing on the dynamic load; when it is present, every test below runs unchanged.

The candidate is loaded explicitly by path (never imported as the benchmark helper); the benchmark `sources.py` comes from the existing conftest
path. The real saved Plot 1 reference and the published Numba output are for the root to run: the data here are small synthetic ledgers with
distinct class and time values, persistent mobile storage (sum over time != final != peak) and the actual Fortran omitted-E format.
Written without being run."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import compare_legacy_sediment as ORIGINAL  # the frozen benchmark helper, read-only (conftest puts benchmarks/legacy_sediment on sys.path)
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_PATH = ROOT / "agent_handoffs" / "tasks" / "gpu_sediment" / "golden_candidate" / "compare_legacy_sediment.py"

if not CANDIDATE_PATH.is_file():  # git-ignored local candidate: absent in a clean checkout, so skip the module rather than fail to collect
    pytest.skip(f"the experimental local golden-helper candidate is absent (not adopted, git-ignored): {CANDIDATE_PATH}",
                allow_module_level=True)


def _load():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("compare_legacy_sediment_golden_candidate", CANDIDATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C = _load()
SUBNORMAL = 1.9762625833649862e-323  # the token the original `float()` parser failed on: `1.9762625833649862-323`
TIMES = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # distinct, not 1..5: a time-alignment error cannot hide
NEW_PROFILE = np.array([1.0, 3.0, 5.0, 4.0, 2.0])  # persistent stock: final 2 != peak 5 != sum over time 15 (times the class factor)
OLD_PROFILE = np.array([0.0, 1.0, 3.0, 5.0, 4.0])  # the previous step's stock


def fmt(value: float) -> str:
    """The original application's real formatting: exponent with `E`, and the E DROPPED for a three-digit exponent."""
    if value == 0.0:
        return "0.0000000000000000E+00"
    text = f"{value:.16E}"
    return re.sub(r"E([+-]\d{3})$", r"\1", text)


def a1_ledger():
    led = np.zeros((5, len(C.A1_COLUMNS), 6))
    cls = np.arange(1, 7, dtype=np.float64)
    step = np.arange(1, 6, dtype=np.float64)
    led[:, C.A1_COLUMNS.index("pickup_kg"), :] = 0.1 * step[:, None] * cls[None, :]
    led[:, C.A1_COLUMNS.index("deposition_active_kg"), :] = 0.05 * step[:, None] * cls[None, :]
    led[:, C.A1_COLUMNS.index("effective_clip_source_kg"), :] = 0.01 * cls[None, :]
    led[:, C.A1_COLUMNS.index("cn_export_kg"), :] = 0.002 * step[:, None]
    led[:, C.A1_COLUMNS.index("new_mobile_kg"), :] = NEW_PROFILE[:, None] * cls[None, :]
    led[:, C.A1_COLUMNS.index("old_mobile_kg"), :] = OLD_PROFILE[:, None] * cls[None, :]
    return led


def write_dat(path: Path, led: np.ndarray, times=TIMES, mutate=None) -> Path:
    """Rows `iter t class` + 11 columns (pickup dep_active dep_outside clip old new cn endpoint cellflux resid internal)."""
    zero = np.zeros(led.shape[::2])
    cols = [led[:, C.A1_COLUMNS.index("pickup_kg"), :], led[:, C.A1_COLUMNS.index("deposition_active_kg"), :], zero,
            led[:, C.A1_COLUMNS.index("effective_clip_source_kg"), :], led[:, C.A1_COLUMNS.index("old_mobile_kg"), :],
            led[:, C.A1_COLUMNS.index("new_mobile_kg"), :], led[:, C.A1_COLUMNS.index("cn_export_kg"), :], zero, zero,
            np.zeros(led.shape[::2]), zero]
    cols[9][0, 0] = SUBNORMAL  # the residual column of step 1, class 1
    lines = ["# iter t class pickup dep_active dep_outside clip old new cn endpoint cellflux resid internal", ""]
    for i in range(led.shape[0]):
        for k in range(led.shape[2]):
            tokens = [str(i + 1), fmt(float(times[i])), str(k + 1)] + [fmt(float(c[i, k])) for c in cols]
            lines.append(" ".join(tokens))
    if mutate is not None:
        lines = mutate(lines)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_npz(path: Path, led: np.ndarray, t=TIMES, columns=None, drop=(), extra=None) -> Path:
    arrays = {"ledger": led, "t_s": np.asarray(t, dtype=np.float64), "columns": np.array(columns or C.A1_COLUMNS)}
    for key in drop:
        del arrays[key]
    arrays.update(extra or {})
    np.savez(path, **arrays)
    return path


def data_rows(lines):
    return [i for i, ln in enumerate(lines) if ln.strip() and not ln.startswith("#")]


# ---- the parser ----------------------------------------------------------------------------------------------------------------------
def test_the_original_float_parser_fails_on_the_real_token_and_the_candidate_keeps_the_subnormal():
    with pytest.raises(ValueError):
        float("1.9762625833649862-323")  # the confirmed failure of the original helper
    value = C.fortran_number("1.9762625833649862-323")
    assert value == SUBNORMAL and 0.0 < value < 2.2250738585072014e-308  # subnormal, not erased to zero
    assert fmt(SUBNORMAL) == "1.9762625833649862-323"  # the test format is the real one


@pytest.mark.parametrize("token, expected", [
    ("0.5", 0.5), ("-12", -12.0), ("1.0E-05", 1e-5), ("1.5d+02", 150.0), ("1.5e3", 1500.0), (".5", 0.5), ("5.", 5.0), ("+2.5", 2.5),
    ("1.0+100", 1e100), ("-2.5-150", -2.5e-150), ("1.7976931348623157+308", 1.7976931348623157e308), ("-4.9406564584124654-324", -5e-324),
    ("1.2345678901234567E-310", 1.2345678901234567e-310), ("0.0000000000000000E+00", 0.0)])
def test_valid_fortran_real_tokens_including_omitted_e(token, expected):
    assert C.fortran_number(token) == expected


@pytest.mark.parametrize("token", ["", "abc", "1.5-05", "1.5-5", "1.0E", "1.0E+", "NaN", "nan", "Infinity", "+Infinity", "-Infinity", "inf", "********",
                                   "1e999", "1.0+999", "-1.0-999e", "1,5", "1.5.2", "--1", "1.5E+3.0", "1 .5", "1.5-3234", "0x10"])
def test_invalid_or_non_finite_tokens_are_rejected(token):
    with pytest.raises(ValueError):
        C.fortran_number(token)


def test_a_file_in_the_real_format_parses_with_validated_iteration_class_and_time_columns(tmp_path):
    path = write_dat(tmp_path / "ledger.dat", a1_ledger())
    ref = C.parse_fortran_ledger(path)
    assert ref["data"].shape == (5, 6, 14) and ref["steps"] == 5 and ref["classes"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert np.array_equal(ref["t_s"], TIMES)
    assert ref["data"][0, 0, 12] == SUBNORMAL  # the omitted-E token of the residual column survived the parse
    assert ref["data"][4, 5, 8] == 2.0 * 6.0  # new mobile of step 5, class 6
    assert np.array_equal(ref["data"][:, :, 3], a1_ledger()[:, 0, :])  # pickup in column 3, step-major then class


def test_malformed_files_are_refused_not_silently_reshaped(tmp_path):
    led = a1_ledger()

    def swap_rows(lines):
        rows = data_rows(lines)
        lines[rows[0]], lines[rows[1]] = lines[rows[1]], lines[rows[0]]  # classes 1 and 2 of iteration 1 exchanged
        return lines

    def duplicate_iteration(lines):
        rows = data_rows(lines)
        return lines[:rows[6]] + lines[rows[0]:rows[6]] + lines[rows[6]:]

    def drop_last_row(lines):
        return lines[:-1]

    def narrow(lines):
        rows = data_rows(lines)
        lines[rows[3]] = " ".join(lines[rows[3]].split()[:13])
        return lines

    def bad_token(lines):
        rows = data_rows(lines)
        tokens = lines[rows[4]].split()
        tokens[5] = "1.5-05"
        lines[rows[4]] = " ".join(tokens)
        return lines

    def infinite(lines):
        rows = data_rows(lines)
        tokens = lines[rows[4]].split()
        tokens[5] = "Infinity"
        lines[rows[4]] = " ".join(tokens)
        return lines

    def time_differs_in_iteration(lines):
        rows = data_rows(lines)
        tokens = lines[rows[8]].split()
        tokens[1] = fmt(99.0)
        lines[rows[8]] = " ".join(tokens)
        return lines

    def time_not_increasing(lines):
        rows = data_rows(lines)
        for r in rows[12:18]:
            tokens = lines[r].split()
            tokens[1] = fmt(5.0)  # iteration 3 earlier than iteration 2
            lines[r] = " ".join(tokens)
        return lines

    def class_order_changes(lines):
        rows = data_rows(lines)
        for r, k in zip(rows[12:18], [6, 5, 4, 3, 2, 1], strict=True):
            tokens = lines[r].split()
            tokens[2] = str(k)
            lines[r] = " ".join(tokens)
        return lines

    cases = {"swapped classes": (swap_rows, "ascending"), "duplicate iteration": (duplicate_iteration, "iteration|whole number"),
             "dropped row": (drop_last_row, "whole number"), "13 tokens": (narrow, "tokens"), "bad token": (bad_token, "line"),
             "infinite": (infinite, "line"), "time differs in an iteration": (time_differs_in_iteration, "time differs"),
             "time not increasing": (time_not_increasing, "strictly increasing"), "class order changes": (class_order_changes, "ascending")}
    for name, (mutate, match) in cases.items():
        path = write_dat(tmp_path / f"{name.replace(' ', '_')}.dat", led, mutate=mutate)
        with pytest.raises(ValueError, match=match):
            C.parse_fortran_ledger(path)
    empty = tmp_path / "empty.dat"
    empty.write_text("# nothing\n\n")
    with pytest.raises(ValueError, match="no data rows"):
        C.parse_fortran_ledger(empty)
    # iterations 1 and 2 exchanged (the header and blank line come first, then 6 rows per iteration)
    rows_swapped = write_dat(tmp_path / "iter_swapped.dat", led, mutate=lambda ls: [ls[0], ls[1], *ls[8:14], *ls[2:8], *ls[14:]])
    with pytest.raises(ValueError, match="iteration column"):
        C.parse_fortran_ledger(rows_swapped)


# ---- plot1_golden --------------------------------------------------------------------------------------------------------------------
def pair(tmp_path, led=None, a1=None, **kw):
    led = a1_ledger() if led is None else led
    dat = write_dat(tmp_path / "ref.dat", led, **kw.pop("dat_kw", {}))
    npz = write_npz(tmp_path / "a1.npz", led if a1 is None else a1, **kw.pop("npz_kw", {}))
    return npz, dat


def test_transfer_columns_are_summed_and_storage_is_reported_as_final_and_peak_never_summed_over_time(tmp_path):
    npz, dat = pair(tmp_path)
    out = C.plot1_golden(npz, dat)
    assert out["steps_compared"] == 5 and out["partial_comparison"] is False and out["reference_steps"] == out["syrup_steps"] == 5
    assert set(out["columns"]) == {"pickup_kg", "deposition_active_kg", "effective_clip_source_kg", "cn_export_kg"}  # no mobile under a *_total label
    led = a1_ledger()
    pick = out["columns"]["pickup_kg"]
    assert pick["fortran_total"] == pytest.approx(float(led[:, 0, :].sum()), rel=1e-14)
    assert pick["syrup_total"] == pytest.approx(float(led[:, 0, :].sum()), rel=1e-14) and pick["total_relative"] < 1e-15
    assert pick["fortran_total_by_class"] == pytest.approx(led[:, 0, :].sum(axis=0).tolist())
    cls_sum = 21.0  # 1 + 2 + ... + 6
    new, old = out["storage"]["new_mobile_kg"], out["storage"]["old_mobile_kg"]
    assert new["fortran_final_total"] == new["syrup_final_total"] == 2.0 * cls_sum  # the LAST step, not a sum over time
    assert new["fortran_peak_total"] == 5.0 * cls_sum and new["fortran_peak_time_s"] == 30.0 and new["syrup_peak_time_s"] == 30.0
    naive = float(a1_ledger()[:, C.A1_COLUMNS.index("new_mobile_kg"), :].sum())
    assert naive == 15.0 * cls_sum and len({new["fortran_final_total"], new["fortran_peak_total"], naive}) == 3  # final != peak != sum over time
    assert new["fortran_final_by_class"] == [2.0 * c for c in range(1, 7)] and new["fortran_peak_by_class"] == [5.0 * c for c in range(1, 7)]
    assert new["final_time_s"] == 50.0 and new["final_total_relative"] == 0.0 and new["peak_total_relative"] == 0.0
    assert old["fortran_final_total"] == 4.0 * cls_sum and old["fortran_peak_total"] == 5.0 * cls_sum and old["fortran_peak_time_s"] == 40.0
    assert out["max_abs_time_difference_s"] == 0.0 and out["classes"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_storage_differences_are_reported_per_class_and_in_total(tmp_path):
    a1 = a1_ledger()
    a1[:, C.A1_COLUMNS.index("new_mobile_kg"), 2] *= 1.1  # class 3 only
    npz, dat = pair(tmp_path, a1=a1)
    new = C.plot1_golden(npz, dat)["storage"]["new_mobile_kg"]
    assert new["syrup_final_by_class"][2] == pytest.approx(2.0 * 3.0 * 1.1) and new["syrup_final_by_class"][1] == 4.0
    assert new["syrup_final_total"] == pytest.approx(42.0 + 0.6) and new["final_total_relative"] == pytest.approx(0.6 / 42.6)
    assert new["syrup_peak_by_class"][2] == pytest.approx(5.0 * 3.0 * 1.1) and new["peak_total_relative"] > 0.0
    transfer = C.plot1_golden(npz, dat)["columns"]["pickup_kg"]
    assert transfer["total_relative"] < 1e-15  # the transfer columns were not touched


def test_unequal_storms_are_refused_by_default_and_a_declared_partial_comparison_is_labelled(tmp_path):
    led = a1_ledger()
    npz, dat = pair(tmp_path, led=led, a1=led[:4], npz_kw={"t": TIMES[:4]})
    with pytest.raises(ValueError, match="different lengths"):
        C.plot1_golden(npz, dat)
    out = C.plot1_golden(npz, dat, partial_steps=3)
    assert out["partial_comparison"] is True and out["steps_compared"] == 3 and out["reference_steps"] == 5 and out["syrup_steps"] == 4
    new = out["storage"]["new_mobile_kg"]
    assert new["fortran_final_total"] == 5.0 * 21.0 and new["final_time_s"] == 30.0  # the FINAL value is that of the declared step 3
    assert new["fortran_peak_total"] == 5.0 * 21.0
    for bad in (0, 5, True, 2.0, -1, "3"):  # zero, longer than the shorter storm, bool, float, negative, str
        with pytest.raises(ValueError, match="partial_steps"):
            C.plot1_golden(npz, dat, partial_steps=bad)
    equal_dir = tmp_path / "eq"
    equal_dir.mkdir()
    equal_npz, equal_dat = pair(equal_dir)
    assert C.plot1_golden(equal_npz, equal_dat, partial_steps=2)["partial_comparison"] is True  # explicit partial even for equal lengths


def test_schema_class_count_time_and_finiteness_mismatches_are_refused(tmp_path):
    led = a1_ledger()
    sub = tmp_path
    cases = {
        "missing t_s": ({"drop": ("t_s",)}, "lacks"),
        "missing columns": ({"drop": ("columns",)}, "lacks"),
        "column order": ({"columns": tuple(reversed(C.A1_COLUMNS))}, "columns differ"),
        "time shifted": ({"t": TIMES + 1.0}, "time axes disagree"),
        "time length": ({"t": TIMES[:4]}, "one entry per"),
    }
    for name, (npz_kw, match) in cases.items():
        d = sub / name.replace(" ", "_")
        d.mkdir()
        npz, dat = pair(d, npz_kw=npz_kw)
        with pytest.raises(ValueError, match=match):
            C.plot1_golden(npz, dat)
    d = sub / "five_classes"
    d.mkdir()
    npz = write_npz(d / "a1.npz", led[:, :, :5])
    dat = write_dat(d / "ref.dat", led)
    with pytest.raises(ValueError, match="shape"):
        C.plot1_golden(npz, dat)
    d = sub / "nonfinite"
    d.mkdir()
    bad = a1_ledger()
    bad[2, 7, 3] = np.nan
    npz, dat = pair(d, a1=bad)
    with pytest.raises(ValueError, match="non-finite"):
        C.plot1_golden(npz, dat)
    d = sub / "dat_time_other_axis"
    d.mkdir()
    npz, dat = pair(d, dat_kw={"times": TIMES * 2.0})
    with pytest.raises(ValueError, match="time axes disagree"):
        C.plot1_golden(npz, dat)
    d = sub / "explicit_time_tolerance"
    d.mkdir()
    npz, dat = pair(d, npz_kw={"t": TIMES + 5e-7})
    with pytest.raises(ValueError, match="time axes disagree"):  # the DEFAULT tolerance is 0: exact axes, no invented 1e-6
        C.plot1_golden(npz, dat)
    out = C.plot1_golden(npz, dat, time_atol_s=1e-6)  # an explicit positive tolerance is the caller's choice and is labelled
    assert out["max_abs_time_difference_s"] == pytest.approx(5e-7) and out["time_atol_s"] == 1e-6 and "explicitly requested" in out["time_axes"]
    with pytest.raises(ValueError, match="time axes disagree"):
        C.plot1_golden(npz, dat, time_atol_s=1e-9)
    assert C.plot1_golden(npz, dat, time_atol_s=np.float64(1e-6))["time_atol_s"] == 1e-6  # NumPy scalars are numbers too


@pytest.mark.parametrize("bad", [float("nan"), np.nan, float("inf"), float("-inf"), -1.0, -1e-12, True, False, np.bool_(True), "1e-6", None,
                                 [1e-6], np.array([1e-6]), 1j])
def test_a_malformed_time_tolerance_is_refused_before_any_comparison(tmp_path, bad):
    """NaN made `diff > tol` False and accepted a 50 s mismatch; every non-scalar, non-finite, negative or non-numeric tolerance is refused up front."""
    npz, dat = pair(tmp_path, npz_kw={"t": TIMES + 50.0})  # axes that differ by 50 s
    with pytest.raises(ValueError, match="time_atol_s"):
        C.plot1_golden(npz, dat, time_atol_s=bad)
    missing = tmp_path / "does_not_exist.npz"
    with pytest.raises(ValueError, match="time_atol_s"):  # validated BEFORE the files are even read
        C.plot1_golden(missing, tmp_path / "also_missing.dat", time_atol_s=bad)


def test_the_default_and_an_exact_zero_tolerance_demand_identical_axes_and_reject_a_mismatch(tmp_path):
    npz, dat = pair(tmp_path, npz_kw={"t": TIMES + 50.0})
    for kwargs in ({}, {"time_atol_s": 0.0}, {"time_atol_s": 0}):
        with pytest.raises(ValueError, match="time axes disagree"):
            C.plot1_golden(npz, dat, **kwargs)
    ok_dir = tmp_path / "ok"
    ok_dir.mkdir()
    npz, dat = pair(ok_dir)
    out = C.plot1_golden(npz, dat)
    assert out["time_atol_s"] == 0.0 and out["time_axes"] == "exact" and out["max_abs_time_difference_s"] == 0.0


# ---- the class axis ---------------------------------------------------------------------------------------------------------------------
def relabel(fn):
    """A mutate function that rewrites the class token of every data row with `fn(k)`."""
    def mutate(lines):
        for r in data_rows(lines):
            tokens = lines[r].split()
            tokens[2] = fn(int(tokens[2]))
            lines[r] = " ".join(tokens)
        return lines
    return mutate


@pytest.mark.parametrize("label, fn", [("shifted_up_2_to_7", lambda k: str(k + 1)), ("shifted_down_0_to_5", lambda k: str(k - 1)),
                                       ("fractional", lambda k: f"{k}.5"), ("all_equal", lambda k: "1"), ("doubled", lambda k: str(2 * k)),
                                       ("zero_based_and_one", lambda k: str(0 if k == 1 else k))])
def test_class_ids_must_be_exactly_1_to_6_in_every_iteration(tmp_path, label, fn):
    """Arbitrary ascending labels (2..7, 1.5..6.5, 2,4,..12) used to pass the old `strictly ascending` check although the NPZ class axis is 1..6."""
    dat = write_dat(tmp_path / f"{label}.dat", a1_ledger(), mutate=relabel(fn))
    with pytest.raises(ValueError, match="expected class IDs"):
        C.parse_fortran_ledger(dat)
    npz = write_npz(tmp_path / "a1.npz", a1_ledger())
    with pytest.raises(ValueError, match="expected class IDs"):
        C.plot1_golden(npz, dat)


def test_the_exact_class_ids_are_accepted_and_one_shifted_iteration_is_still_refused(tmp_path):
    ok = write_dat(tmp_path / "ok.dat", a1_ledger())
    assert C.parse_fortran_ledger(ok)["classes"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def shift_iteration_three(lines):
        for r in data_rows(lines)[12:18]:
            tokens = lines[r].split()
            tokens[2] = str(int(tokens[2]) + 1)
            lines[r] = " ".join(tokens)
        return lines

    with pytest.raises(ValueError, match="expected class IDs"):
        C.parse_fortran_ledger(write_dat(tmp_path / "shift3.dat", a1_ledger(), mutate=shift_iteration_three))


# ---- per-class peak times ---------------------------------------------------------------------------------------------------------------
def peak_ledger(peak_steps, base=1.0, height=10.0):
    """new/old mobile whose class k peaks at step `peak_steps[k]` (None = flat: the first tie is the peak)."""
    led = np.zeros((5, len(C.A1_COLUMNS), 6))
    for k, step in enumerate(peak_steps):
        profile = np.full(5, base)
        if step is not None:
            profile[step] += height * (k + 1)
        led[:, C.A1_COLUMNS.index("new_mobile_kg"), k] = profile
        led[:, C.A1_COLUMNS.index("old_mobile_kg"), k] = profile
    return led


def test_per_class_peak_times_use_each_axis_first_ties_and_are_aligned_by_class(tmp_path):
    ref = peak_ledger([0, 1, 2, 3, 4, None])  # grain classes peak at different steps; class 6 is flat
    syr = peak_ledger([2, 2, 4, 0, 3, None])
    npz, dat = pair(tmp_path, led=ref, a1=syr, npz_kw={"t": TIMES})
    out = C.plot1_golden(npz, dat)
    for name in ("new_mobile_kg", "old_mobile_kg"):
        s = out["storage"][name]
        assert s["fortran_peak_time_by_class_s"] == [10.0, 20.0, 30.0, 40.0, 50.0, 10.0]  # the flat class: the FIRST step
        assert s["syrup_peak_time_by_class_s"] == [30.0, 30.0, 50.0, 10.0, 40.0, 10.0]
        assert s["fortran_peak_by_class"] == [11.0, 21.0, 31.0, 41.0, 51.0, 1.0]  # the per-class peak VALUES are still reported
        assert "fortran_peak_time_s" in s and "syrup_peak_time_s" in s and "fortran_final_total" in s  # the totals and finals are unchanged
        assert s["fortran_final_by_class"] == [1.0 + 10.0 * (k + 1) * (k == 4) for k in range(5)] + [1.0]  # final = the LAST step, not a peak or a sum


def test_per_class_peak_times_come_from_each_runs_own_time_axis_and_the_compared_window(tmp_path):
    ref = peak_ledger([0, 4, 4, 1, 2, 3])
    npz, dat = pair(tmp_path, led=ref, a1=ref, npz_kw={"t": TIMES})
    full = C.plot1_golden(npz, dat)["storage"]["new_mobile_kg"]
    assert full["fortran_peak_time_by_class_s"] == [10.0, 50.0, 50.0, 20.0, 30.0, 40.0] == full["syrup_peak_time_by_class_s"]
    window = C.plot1_golden(npz, dat, partial_steps=3)["storage"]["new_mobile_kg"]  # classes peaking after step 3 now peak at their in-window max
    assert window["fortran_peak_time_by_class_s"] == [10.0, 10.0, 10.0, 20.0, 30.0, 10.0] == window["syrup_peak_time_by_class_s"]
    assert window["final_time_s"] == 30.0


# ---- nothing else changed -----------------------------------------------------------------------------------------------------------
def test_every_other_function_constant_and_the_main_comparison_are_unchanged_from_the_frozen_helper():
    for name in ("_rel", "map_stats", "compare_runs", "build_engine", "injection_check"):
        assert inspect.getsource(getattr(C, name)) == inspect.getsource(getattr(ORIGINAL, name)), name
    for name in ("FLAG_TOTAL_REL", "FLAG_SIGN_FRACTION", "A1_COLUMNS", "PER_STEP_KG", "N_CLASSES_PLOT"):
        assert getattr(C, name) == getattr(ORIGINAL, name), name
    assert C.S is ORIGINAL.S  # the same read-only sources module
    assert inspect.getsource(C.plot1_golden) != inspect.getsource(ORIGINAL.plot1_golden)  # the one intended change
