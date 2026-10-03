"""The comparison harness `benchmarks/hydraulic_candidates/compare_plot1.py` must WORK, not merely be designed: the scoped
`storm.coupled_step` redirection is entered fresh for every segment of every run (warm-up, repeated runs and snapshot
segmentation) and the original is restored even on failure; controls and protected output paths are refused before anything
runs; the water budget is really computed and fails on a violation; the timer holds only the evolution. CPU only, no Numba, no
Plot 1 case (a small synthetic valley and fakes stand in). Nothing here was run by its author (file-only tools); Codex records
results.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from cand_cases import build, field, schedule

pytest.importorskip("maple")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "hydraulic_candidates"))

import compare_plot1 as cmp

from maple_syrup import case_import, column_experiment, storm
from maple_syrup.storm import StormControl


def legacy_case():
    cs = build("valley:6x5", "explicit", depth="wet", seed=1)
    return cs, {"graph": cs.graph, "params": cs.params, "field": field(cs), "schedule": schedule([0.0, 6.0], [60.0]),
                "depth0": cs.state.depth_m, "soil0": cs.state.soil_water_m, "control": StormControl(max_dt_s=0.5),
                "cadence": 3.0, "xp": np}


def run(case, end_s, snapshots, **kw):
    return cmp.run_legacy_segments(case["graph"], case["params"], case["field"], case["schedule"], case["depth0"],
                                   case["soil0"], case["control"], end_s, snapshots, case["cadence"], xp=case["xp"], **kw)


# --- the scoped redirection ----------------------------------------------------------------------------------------------
def test_scoped_coupled_step_restores_the_original_after_use_and_after_failure_and_is_created_fresh():
    original = storm.coupled_step
    sentinel = object()
    with cmp.scoped_coupled_step(sentinel):
        assert storm.coupled_step is sentinel
    assert storm.coupled_step is original
    with pytest.raises(RuntimeError, match="boom"), cmp.scoped_coupled_step(sentinel):
        raise RuntimeError("boom")
    assert storm.coupled_step is original
    # a generator context manager cannot be re-entered: this is the defect the first smoke run hit, and why a helper that
    # takes a REPLACEMENT (not a stored manager) is used for every segment
    stored = cmp.scoped_coupled_step(sentinel)
    with stored:
        pass
    with pytest.raises((AttributeError, RuntimeError)), stored:  # the exact error depends on the Python version
        pass
    assert storm.coupled_step is original


def test_warmup_repeated_runs_and_snapshot_segments_each_use_a_fresh_scope_and_restore_the_original():
    _cs, case = legacy_case()
    original = storm.coupled_step
    calls = []

    def counting(*args):
        calls.append(1)
        assert storm.coupled_step is counting  # the scheduler really goes through the override while it is active
        return original(*args)

    previous = 0
    for end, snapshots in ((1.0, []), (6.0, [2.0, 4.0]), (6.0, [2.0, 4.0]), (6.0, [])):  # warm-up, two samples, single event
        raw = run(case, end, snapshots, step_override=counting)
        assert len(calls) > previous and storm.coupled_step is original
        previous = len(calls)
        assert len(raw["segments"]) == len({*snapshots, end})
        assert sorted(raw["maps"]) == sorted(snapshots)


def test_a_failure_in_a_later_segment_still_restores_the_original():
    _cs, case = legacy_case()
    original = storm.coupled_step
    count = []

    def fails_late(*args):
        count.append(1)
        if len(count) > 5:  # inside the second segment
            raise RuntimeError("late failure")
        return original(*args)

    with pytest.raises(RuntimeError, match="late failure"):
        run(case, 6.0, [2.0, 4.0], step_override=fails_late)
    assert storm.coupled_step is original


def test_without_an_override_the_production_step_is_used_and_segmentation_does_not_change_the_result():
    _cs, case = legacy_case()
    original = storm.coupled_step
    whole = run(case, 6.0, [])
    cut = run(case, 6.0, [2.0, 4.0])
    assert storm.coupled_step is original
    np.testing.assert_allclose(cut["state"].depth_m, whole["state"].depth_m, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(cut["state"].soil_water_m, whole["state"].soil_water_m, rtol=1e-12, atol=1e-18)
    export = [sum(float(s.cumulative_export_m3) for s in raw["segments"]) for raw in (whole, cut)]
    assert export[1] == pytest.approx(export[0], rel=1e-12)
    assert np.array_equal(cut["maps"][2.0]["depth_m"].shape, whole["state"].depth_m.shape)


# --- controls and protected paths, before anything runs --------------------------------------------------------------------
GOOD = {"contenders": "explicit_numpy,local_inertial_numpy", "dts": "1,0.5", "end_s": 10.0, "report_every_s": 5.0,
        "snapshot_times_s": "5", "warmup_s": 1.0, "repeats": 1, "cfl_max": 0.5}


def args(**change):
    return SimpleNamespace(**{**GOOD, **change})


def test_valid_controls_are_accepted():
    names, dts, snapshots = cmp.validate_controls(args())
    assert names == ["explicit_numpy", "local_inertial_numpy"] and dts == [1.0, 0.5] and snapshots == [5.0]
    assert cmp.validate_controls(args(warmup_s=0.0, snapshot_times_s=""))[2] == []


@pytest.mark.parametrize("change", [
    {"contenders": "bogus"}, {"contenders": ""}, {"contenders": "explicit_numpy,explicit_numpy"},
    {"dts": "1,nan"}, {"dts": "1,inf"}, {"dts": "0"}, {"dts": "-1"}, {"dts": "0.0001"}, {"dts": ""}, {"dts": "1,1"},
    {"dts": "a"}, {"end_s": 0.0}, {"end_s": float("nan")}, {"end_s": float("inf")}, {"report_every_s": 0.0},
    {"report_every_s": -5.0}, {"warmup_s": -1.0}, {"warmup_s": float("nan")}, {"repeats": 0}, {"repeats": True},
    {"snapshot_times_s": "0"}, {"snapshot_times_s": "11"}, {"snapshot_times_s": "nan"}, {"snapshot_times_s": "x"},
    {"cfl_max": 0.0}, {"cfl_max": 0.9}, {"cfl_max": float("nan")},
])
def test_bad_controls_are_refused(change):
    with pytest.raises(ValueError):
        cmp.validate_controls(args(**change))


def fail(*_a, **_k):
    pytest.fail("a run or a case read happened before the refusal")


@pytest.mark.parametrize("flags", [["--dts", "0.5,nan"], ["--end-s", "-1"], ["--repeats", "0"], ["--cfl-max", "0.9"],
                                   ["--snapshot-times-s", "999999"], ["--contenders", "bogus"], ["--warmup-s", "-1"],
                                   ["--report-every-s", "0"]])
def test_main_refuses_bad_controls_before_reading_the_case_or_running(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(cmp, "build_runner", fail)
    monkeypatch.setattr(case_import, "verify_plot1_case", fail)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as info:
        cmp.main(["--case-dir", str(tmp_path / "case"), "--output-dir", str(out), *flags])
    assert info.value.code == 2 and not out.exists()


def test_main_refuses_an_existing_output_path_and_never_overwrites_it(tmp_path, monkeypatch):
    monkeypatch.setattr(cmp, "build_runner", fail)
    monkeypatch.setattr(case_import, "verify_plot1_case", fail)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("precious")
    with pytest.raises(SystemExit):
        cmp.main(["--case-dir", str(tmp_path / "case"), "--output-dir", str(out)])
    assert [p.name for p in out.iterdir()] == ["keep.txt"] and (out / "keep.txt").read_text() == "precious"


def test_main_refuses_output_inside_protected_trees_with_the_same_rule_as_the_cli(tmp_path, monkeypatch):
    case_dir, maple_src, pkg = tmp_path / "case", tmp_path / "maple", tmp_path / "syrup_pkg"
    for path in (case_dir, maple_src, pkg, tmp_path / "mah", tmp_path / "recipes"):
        path.mkdir()
    fake = SimpleNamespace(
        maple_dependency=SimpleNamespace(source_root=maple_src, package_dir=maple_src / "maple"),
        report={"recipe": {"mahleran_root": str(tmp_path / "mah"), "recipe_path": str(tmp_path / "recipes" / "r.json")}})
    monkeypatch.setattr(case_import, "verify_plot1_case", lambda *a, **k: fake)
    monkeypatch.setattr(column_experiment, "_syrup_provenance", lambda: {"package_dir": str(pkg)})
    monkeypatch.setattr(cmp, "build_runner", fail)
    for protected in (case_dir / "out", maple_src / "x", tmp_path / "mah" / "x", tmp_path / "recipes" / "x", pkg / "o"):
        with pytest.raises(SystemExit) as info:
            cmp.main(["--case-dir", str(case_dir), "--output-dir", str(protected)])
        assert info.value.code == 2 and not protected.exists(), protected


# --- the budget is really computed ----------------------------------------------------------------------------------------
def budget_inputs():
    return SimpleNamespace(ny=2, nx=2, area=1.0, host={"rainfall_scale": np.ones(4)},
                           schedule=SimpleNamespace(depth_m=lambda a, b: 0.75))  # 0.75 m over 4 unit cells = 3 m3


CLOSED = {"surface_initial": 1.0, "soil_initial": 2.0, "rain": 3.0, "intake": 1.0, "saturation_return": 0.0,
          "drainage": 0.5, "export": 0.5, "surface_final": 2.5, "soil_final": 2.5}


def test_a_closed_water_budget_is_reported_with_the_maple_bound():
    result = cmp.water_budget(dict(CLOSED), budget_inputs(), 10, 5.0)
    assert result["closed"] and abs(result["water_residual_m3"]) <= result["bound_m3"] and result["bound_m3"] > 0.0
    assert result["rule"].startswith("conservation.volume_roundoff_bound_m3")


@pytest.mark.parametrize("key", ["export", "surface_final", "soil_final", "intake", "rain", "drainage"])
def test_a_violated_budget_raises_before_any_output(key):
    volumes = dict(CLOSED)
    volumes[key] += 1e-3
    with pytest.raises(RuntimeError, match="no output written"):
        cmp.water_budget(volumes, budget_inputs(), 10, 5.0)


# --- the timer holds only the evolution ----------------------------------------------------------------------------------
def test_the_timed_region_contains_only_the_run_between_two_counter_reads():
    @dataclasses.dataclass
    class Counters:
        n: int = 0

        def delta(self, other):
            return Counters(self.n - other.n)

    events = []

    def reader():
        events.append("read")
        return Counters(len(events))

    def fake_run(end, snapshots):
        events.append("run")
        return {"kind": "candidate"}

    wall, raw, transfers = cmp.time_sample(fake_run, 5.0, [1.0], reader)
    assert events == ["read", "run", "read"] and raw == {"kind": "candidate"} and wall >= 0.0
    assert transfers == {"n": 2}  # second read (3 events seen) minus the first (1 event seen)
