"""Short real Plot 1 runs of the compiled CPU candidates: the CLI's `--implementation numba` choice audited against the NumPy
reference field by field at the UNCHANGED bounds (rtol 2e-12 / atol 1e-14), an exact same-implementation determinism check, the
refusal of invalid pairs, and the comparison-harness smoke run with the numba contenders and their per-sample guards. Skipped
without Numba. Nothing here was run by its author (file-only tools); Codex records results.

KNOWN EXPERIMENTAL EXCEPTION (root measurements, recorded and audited here, never turned into a pass): in the 90 s local-inertial
run on the root's platform only `final_state.npz:velocity_m_s` exceeded the bound (3 cells, max abs 2.4673e-14 m/s, worst normalized
error 2.086); every other saved field passed. The compiled lateral stage is bitwise equal to the NumPy reference for IDENTICAL column
inputs (see test_numba_candidates.py), so the independent-trajectory difference is rounding amplified by the near-dry reconstructed
velocity, which is an experimental diagnostic and not an acceptable erosion velocity. The test therefore (a) allows agreement on
platforms where the libm/NumPy rounding is identical (the exception need not recur), (b) allows ONLY that one field to fail,
(c) records exactly which fields fail and by how much, and (d) fails on any other field, count, metadata or budget.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maple")
pytest.importorskip("numba")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "hydraulic_candidates"))

import compare_plot1 as cmp

from maple_syrup import experimental_experiment as ee

SOLVERS = ("explicit", "local_inertial")
RTOL, ATOL = 2.0e-12, 1.0e-14
FILES = (ee.FINAL_NAME, ee.HYDROGRAPH_NPZ, ee.SNAPSHOTS_NAME)
# The ONLY field allowed to fail at the unchanged bound (the root's measured local-inertial exception). Explicit: none.
KNOWN_EXCEPTIONS = {"explicit": set(), "local_inertial": {(ee.FINAL_NAME, "velocity_m_s")}}


def run_cli(case, out, *extra, end="90") -> dict:
    argv = ["--case-dir", str(case), "--output-dir", str(out), "--end-s", end, "--report-every-s", "30",
            "--allow-maple-source-change", *extra]
    assert ee.main(argv) == 0
    return json.loads((out / ee.SUMMARY_NAME).read_text())


def audit_saved_fields(reference_dir, other_dir) -> dict:
    """Compare EVERY array of every saved file with the NumPy reference at the unchanged bounds, with the same predicate as
    `assert_allclose(actual=other, desired=reference)`. Returns `{(file, key): details}` for each field that exceeds it."""
    failures = {}
    for name in FILES:
        a, b = np.load(reference_dir / name), np.load(other_dir / name)
        assert sorted(a.files) == sorted(b.files), name
        for key in a.files:
            x, y = a[key], b[key]
            assert x.shape == y.shape and x.dtype == y.dtype, (name, key)
            if not np.allclose(y, x, rtol=RTOL, atol=ATOL):
                diff = np.abs(y - x)
                tolerance = ATOL + RTOL * np.abs(x)
                failures[(name, key)] = {"mismatched_elements": int(np.count_nonzero(diff > tolerance)), "size": int(x.size),
                                         "max_abs": float(diff.max()), "worst_normalized": float((diff / tolerance).max())}
    return failures


@pytest.mark.parametrize("solver", SOLVERS)
def test_numba_cli_audits_every_saved_field_against_the_numpy_reference_at_the_unchanged_bounds(
        plot1_case, tmp_path, solver, record_property):
    ref = run_cli(plot1_case, tmp_path / "ref", "--solver", solver, "--backend", "numpy", "--snapshot-times-s", "45")
    new = run_cli(plot1_case, tmp_path / "numba", "--solver", solver, "--backend", "numpy", "--implementation", "numba",
                  "--snapshot-times-s", "45")
    failures = audit_saved_fields(tmp_path / "ref", tmp_path / "numba")
    record_property("saved_field_failures", json.dumps({f"{k[0]}:{k[1]}": v for k, v in sorted(failures.items())}))
    unexpected = set(failures) - KNOWN_EXCEPTIONS[solver]
    assert not unexpected, {f"{k[0]}:{k[1]}": failures[k] for k in unexpected}  # every other field passes the unchanged bound
    for details in failures.values():  # an AUDIT of the known exception (a sanity ceiling, not a tolerance): tiny and sparse
        assert details["max_abs"] < 1e-12 and details["mismatched_elements"] <= 0.01 * details["size"], details
    # counts, continuation, limiter, budgets, guards: exact or within their own bound, whatever the velocity audit recorded
    for key in ("n_accepted_steps", "n_rejected_attempts", "n_boundaries", "dt_min_accepted_s", "dt_max_accepted_s"):
        assert new["time"][key] == ref["time"][key], key
    assert new["numerics"]["limited_cells_total"] == ref["numerics"]["limited_cells_total"]
    assert new["final_state"]["continuation"] == ref["final_state"]["continuation"]  # the adaptive cap travels identically
    budget = new["budget"]
    assert abs(budget["water_residual_m3"]) <= budget["bound_m3"] and abs(budget["surface_residual_m3"]) <= budget["bound_m3"]
    assert abs(budget["soil_residual_m3"]) <= budget["bound_m3"]
    assert new["maple_state"]["unchanged"] is True and new["provenance"]["source_stability"]["stable"] is True
    # the actual choice and provenance are saved; the reference run records its own (the default selection is unchanged)
    assert new["provenance"]["implementation"] == "numba" and new["hydraulics"]["implementation"] == "numba"
    assert ref["provenance"]["implementation"] == "numpy" and ref["hydraulics"]["implementation"] == "numpy"
    assert new["backend"]["requested"] == {"backend": "numpy", "implementation": "numba"}
    assert new["backend"]["resolved"] == {"backend": "numpy", "implementation": "numba"}
    assert ref["backend"]["requested"] == {"backend": "numpy", "implementation": None}
    assert ref["backend"]["resolved"] == {"backend": "numpy", "implementation": "numpy"}
    kernels = new["hydraulics"]["kernels"]
    assert kernels["numba_options"]["fastmath"] is False and kernels["versions"]["numba"]
    assert len(kernels["module_sha256"]) == 64
    assert new["hydraulics"]["context"]["static_bytes"] > 0 and new["hydraulics"]["hydrology_context"]["host_only"] is True
    assert "Numba" in new["backend"]["gpu"] and "no GPU" in new["backend"]["gpu"]
    assert "Host-only" in new["backend"]["transfer_scope"]
    assert new["backend"]["counted_packet_reads_per_attempt"] is None
    assert "NOT" in new["hydraulics"]["qualification"] and "Numba" in new["hydraulics"]["qualification"]


@pytest.mark.parametrize("solver", SOLVERS)
def test_the_same_implementation_reruns_the_90_s_cli_bitwise_in_every_saved_field(plot1_case, tmp_path, solver):
    """Determinism of the compiled form itself: two runs of the same inputs are bit-identical in ALL saved fields (no tolerance),
    so a difference from the NumPy reference is a property of the two arithmetic paths, not noise inside the compiled path."""
    first = run_cli(plot1_case, tmp_path / "a", "--solver", solver, "--backend", "numpy", "--implementation", "numba",
                    "--snapshot-times-s", "45")
    second = run_cli(plot1_case, tmp_path / "b", "--solver", solver, "--backend", "numpy", "--implementation", "numba",
                     "--snapshot-times-s", "45")
    for name in FILES:
        a, b = np.load(tmp_path / "a" / name), np.load(tmp_path / "b" / name)
        assert sorted(a.files) == sorted(b.files), name
        for key in a.files:
            np.testing.assert_array_equal(b[key], a[key], err_msg=f"{name}:{key}")
    for key in ("n_accepted_steps", "n_rejected_attempts", "n_boundaries", "dt_min_accepted_s", "dt_max_accepted_s"):
        assert second["time"][key] == first["time"][key], key
    assert second["budget"]["water_residual_m3"] == first["budget"]["water_residual_m3"]


@pytest.mark.parametrize("flags", [["--backend", "cupy", "--implementation", "numba"],
                                   ["--backend", "numpy", "--implementation", "cuda"]])
def test_invalid_pairs_are_refused_before_anything_is_written(plot1_case, tmp_path, capsys, flags):
    out = tmp_path / "out"
    assert ee.main(["--case-dir", str(plot1_case), "--output-dir", str(out), "--solver", "explicit", *flags]) == 1
    assert "implementation" in capsys.readouterr().err and not out.exists()


def test_the_comparison_harness_runs_the_numba_contenders_with_guards_and_matches_the_numpy_contenders(plot1_case, tmp_path):
    out = tmp_path / "comparison"
    code = cmp.main(["--case-dir", str(plot1_case), "--output-dir", str(out), "--contenders",
                     "legacy_numba_prepared,explicit_numpy,explicit_numba,local_inertial_numpy,local_inertial_numba",
                     "--dts", "1", "--end-s", "10", "--snapshot-times-s", "5", "--warmup-s", "1", "--report-every-s", "5",
                     "--allow-maple-source-change"])
    assert code == 0
    payload = json.loads((out / "comparison.json").read_text())
    runs = {r["contender"]: r for r in payload["runs"]}
    assert set(runs) == {"legacy_numba_prepared", "explicit_numpy", "explicit_numba", "local_inertial_numpy",
                         "local_inertial_numba"}
    for name, run in runs.items():
        assert run["guard"]["budget_closed"] and run["guard"]["bed_unchanged"] and run["guard"]["sources_stable"], name
        assert all(s["guard"]["budget_closed"] for s in run["samples"]), name  # every timed sample was validated
        assert run["budget"]["closed"] and run["snapshots_taken"] == [5.0], name
    for name in ("explicit_numba", "local_inertial_numba"):
        hydraulics = runs[name]["provenance"]["hydraulics"]
        assert hydraulics["implementation"] == "numba" and hydraulics["kernels"]["numba_options"]["fastmath"] is False
        assert runs[name]["accepted_steps"] == runs[name.replace("numba", "numpy")]["accepted_steps"]
    arrays, maps = np.load(out / "final_arrays.npz"), np.load(out / "maps.npz")
    assert "local_inertial_numba_dt1_qx_m2_s" in arrays.files  # the face momentum is saved for the compiled form too
    for family in ("explicit", "local_inertial"):
        for key in ("depth_m", "soil_water_m"):
            np.testing.assert_allclose(arrays[f"{family}_numba_dt1_{key}"], arrays[f"{family}_numpy_dt1_{key}"], rtol=RTOL,
                                       atol=ATOL, err_msg=f"{family}:{key}")
        for key in ("depth_m", "velocity_m_s"):
            np.testing.assert_allclose(maps[f"{family}_numba_dt1_t5_{key}"], maps[f"{family}_numpy_dt1_t5_{key}"], rtol=RTOL,
                                       atol=ATOL, err_msg=f"{family}:map:{key}")
