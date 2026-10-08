"""CPU-only orchestration tests of `benchmarks/legacy_gpu/kernel_event_profile.py` with a fake context and fake events: the launch wrapper preserves every
argument and the order, is removed on any exit, the event capacity is bounded, aggregation and controls are checked, and instrumented/plain results must
agree bitwise. They are NOT GPU tests and prove nothing about the real kernels (the root runs the actual equivalence on the device). Written without
being run."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("kernel_event_profile", ROOT / "benchmarks" / "legacy_gpu" / "kernel_event_profile.py")
P = importlib.util.module_from_spec(spec)
sys.modules["kernel_event_profile"] = P
spec.loader.exec_module(P)

SCRIPT = ["sg_laws", "sg_values", "sg_gather", "sg_cn_level", "sg_cn_level", "sg_reduce_partial", "sg_tally"]


def make_ops():
    state = {"t": 0.0}

    class Event:
        def record(self):
            state["t"] += 1.0
            self.t = state["t"]

    return SimpleNamespace(event=Event, elapsed_ms=lambda a, b: b.t - a.t, sync=lambda e: None, clock=lambda: state["t"] * 1e-3)


class FakeCtx:
    """The parts of `CudaLegacyContext` the harness uses: `_launch`, `step`, `reset`, `check_flags`, results and counters."""

    def __init__(self, tamper=False, nonfinite=False, diverge=False):
        self.tamper, self.nonfinite, self.diverge = tamper, nonfinite, diverge
        self.stats = {"launches": 0, "steps": 0, "d2h_flag_reads": 0}
        self.trace: list = []
        self.resets = 0
        self.acc = 0.0
        self.row = 0

    def _launch(self, name, grid, block, args):
        self.trace.append((name, grid, block, args))
        self.stats["launches"] += 1

    def step(self, row, depth, velocity, rain):
        names = SCRIPT + (["sg_tally"] if self.diverge and self.resets >= 3 else [])
        for i, name in enumerate(names):
            self._launch(name, 8 + i, 128, (row, i, depth))
        self.acc += depth[0] + velocity + rain
        self.row = row + 1
        self.stats["steps"] += 1

    def reset(self):
        self.resets += 1
        self.stats.update(launches=0, steps=0, d2h_flag_reads=0)  # like CudaLegacyContext.reset: the measured counters restart
        self.acc, self.row = 0.0, 0

    def _bump(self):
        return 0.5 if self.tamper and "_launch" in self.__dict__ else 0.0

    def check_flags(self, upto=None):
        n = self.row
        self.stats["d2h_flag_reads"] += 1
        self.host_ledger = np.full((n, 2), self.acc + self._bump())
        self.host_counts = np.full((n, 3), 4, dtype=np.int64)
        self.host_flags = np.zeros(n, dtype=np.uint64)
        if self.nonfinite:
            self.host_ledger[0, 0] = np.nan
        return n

    def download_maps(self):
        # the five maps of the real `CudaLegacyContext.download_maps`
        return {"cum_det": np.full((2, 2), self.acc + self._bump()), "cum_dep": np.full((2, 2), 1.0), "cum_clip": np.full((2, 2), 0.25),
                "mobile": np.full((2, 2), 2.0), "v_prev": np.full((2, 2), 0.125)}


DEPTH = [1.0]
INPUTS = [(DEPTH, 0.5, 0.25)]


def test_wrapper_preserves_arguments_order_and_restores_the_method():
    ctx = FakeCtx()
    plain = FakeCtx()
    for row in range(2):
        plain.step(row, DEPTH, 0.5, 0.25)
    with P.profiling(ctx, 14, make_ops()) as rec:
        assert "_launch" in ctx.__dict__ and ctx._launch is not FakeCtx._launch
        for row in range(2):
            ctx.step(row, DEPTH, 0.5, 0.25)
        assert rec.names == SCRIPT * 2
    assert ctx.trace == plain.trace  # same names, grids, blocks, argument tuples and order
    assert all(t[3][2] is DEPTH for t in ctx.trace)  # the very same objects, not copies
    assert "_launch" not in ctx.__dict__ and ctx._launch.__func__ is FakeCtx._launch and ctx.stats["launches"] == 14
    assert len(rec.intervals_ms()) == 14 and all(v > 0 for v in rec.intervals_ms())


def test_the_original_method_is_restored_when_the_body_raises_and_a_preexisting_override_is_kept():
    ctx = FakeCtx()
    with pytest.raises(RuntimeError, match="boom"), P.profiling(ctx, 14, make_ops()):
        ctx.step(0, DEPTH, 0.5, 0.25)
        raise RuntimeError("boom")
    assert "_launch" not in ctx.__dict__
    seen = []

    def custom(name, grid, block, args):
        seen.append(name)
        FakeCtx._launch(ctx, name, grid, block, args)

    ctx.__dict__["_launch"] = custom
    with P.profiling(ctx, 7, make_ops()) as rec:
        ctx.step(0, DEPTH, 0.5, 0.25)
        assert rec.names == SCRIPT and seen == SCRIPT  # the override still runs, wrapped
    assert ctx.__dict__["_launch"] is custom


def test_capacity_is_bounded_and_a_partial_or_overflowing_pass_is_refused():
    ctx = FakeCtx()
    with pytest.raises(P.ProfileError, match="capacity"), P.profiling(ctx, 3, make_ops()):
        ctx.step(0, DEPTH, 0.5, 0.25)  # the 4th launch overflows
    assert "_launch" not in ctx.__dict__ and len(ctx.trace) == 3  # the refused launch was never forwarded
    with P.profiling(ctx, 7, make_ops()) as rec:
        rec.names = ["sg_laws"]
        with pytest.raises(P.ProfileError, match="capacity"):
            rec.intervals_ms()
    for bad in (0, -1, True, 2.0):
        with pytest.raises(P.ProfileError):
            P.LaunchRecorder(ctx, bad, make_ops())
    rec = P.LaunchRecorder(ctx, 2, make_ops())
    rec.install()
    with pytest.raises(P.ProfileError, match="already"):
        rec.install()
    rec.restore()
    rec.restore()  # idempotent
    assert "_launch" not in ctx.__dict__


def test_aggregation_by_kernel_and_group_with_truthful_counts():
    names = ["sg_laws", "sg_values", "sg_cn_level", "sg_cn_level", "sg_ring_inactive", "sg_gather", "sg_new_kernel"]
    ms = [1.0, 2.0, 0.5, 1.5, 0.25, 0.75, 3.0]
    agg = P.aggregate(names, ms)
    assert agg["launches"] == 7 and agg["sum_intervals_ms"] == pytest.approx(9.0)
    assert agg["by_kernel"]["sg_cn_level"] == {"count": 2, "total_ms": 2.0, "max_ms": 1.5, "mean_us": 1000.0}
    assert agg["by_group"]["Crank-Nicolson"]["count"] == 2 and agg["by_group"]["wet laws"]["total_ms"] == 1.0
    assert agg["by_group"]["ordered gather (+ ring/inactive)"]["count"] == 2
    assert agg["by_group"][P.UNCLASSIFIED]["count"] == 1 and agg["unclassified_kernels"] == ["sg_new_kernel"]
    assert sum(g["count"] for g in agg["by_group"].values()) == sum(k["count"] for k in agg["by_kernel"].values()) == 7
    for bad in ([1.0], [1.0] * 6 + [float("nan")], [1.0] * 6 + [-0.1], [1.0] * 6 + [float("inf")]):
        with pytest.raises(P.ProfileError):
            P.aggregate(names, bad)
    assert P.run_length(["a", "a", "b", "a"]) == [["a", 2], ["b", 1], ["a", 1]]
    assert all(P.group_of(k) != P.UNCLASSIFIED for k in ("sg_laws", "sg_post_infiltration", "sg_values", "sg_gather", "sg_ring_inactive",
                                                          "sg_cn_level", "sg_cn_block", "sg_reduce_partial", "sg_reduce_final", "sg_outlet",
                                                          "sg_tally"))


def test_run_profile_warms_up_pairs_alternate_resets_happen_and_launches_are_unchanged():
    ctx = FakeCtx()
    res = P.run_profile(ctx, INPUTS, 3, 3, make_ops())
    assert res["launches_per_pass"] == 21 and res["launches_per_step"] == 7 and res["steps"] == 3 and res["repeats"] == 3
    assert [p["order"] for p in res["pairs"]] == [["plain", "instrumented"], ["instrumented", "plain"], ["plain", "instrumented"]]
    assert [(p["repeat"], p["mode"]) for p in res["passes"]] == [(0, "plain"), (0, "instrumented"), (1, "instrumented"), (1, "plain"),
                                                               (2, "plain"), (2, "instrumented")]
    assert ctx.resets == 2 + 6  # two warm-up passes then six measured passes, each from reset()
    chunk = ctx.trace[:21]
    assert len(ctx.trace) == 8 * 21 and all(ctx.trace[i * 21:(i + 1) * 21] == chunk for i in range(8))  # identical launches, instrumented or not
    assert "_launch" not in ctx.__dict__
    assert res["first_step_launch_sequence"] == [["sg_laws", 1], ["sg_values", 1], ["sg_gather", 1], ["sg_cn_level", 2], ["sg_reduce_partial", 1],
                                                 ["sg_tally", 1]]
    assert res["medians"]["by_kernel"]["sg_cn_level"]["count"] == 6 and res["medians"]["by_group"]["Crank-Nicolson"]["count"] == 6
    assert res["equivalence"]["bitwise_equal_across_all_passes_and_toggle"] is True and "map_cum_det" in res["equivalence"]["fields"]
    assert all(pair["instrumented_over_plain_device"] > 0 and pair["sum_launch_intervals_s"] > 0 for pair in res["pairs"])
    instrumented = [p for p in res["passes"] if p["mode"] == "instrumented"]
    assert all("aggregate" in p and p["aggregate"]["launches"] == 21 for p in instrumented)
    assert all("aggregate" not in p for p in res["passes"] if p["mode"] == "plain")
    assert all(p["diagnostic_transfers"]["d2h_flag_reads"] == 1 for p in res["passes"])  # reads happen after the loop, once per pass
    assert json.dumps(res, default=str)  # serialisable


def test_instrumentation_that_changes_any_result_is_refused():
    with pytest.raises(P.ProfileError, match="differs"):
        P.run_profile(FakeCtx(tamper=True), INPUTS, 2, 1, make_ops())


def test_nonfinite_or_incomplete_captures_and_launch_count_divergence_are_refused():
    with pytest.raises(P.ProfileError, match="non-finite"):
        P.run_profile(FakeCtx(nonfinite=True), INPUTS, 2, 1, make_ops())
    with pytest.raises(P.ProfileError, match="launch"):
        P.run_profile(FakeCtx(diverge=True), INPUTS, 2, 2, make_ops())
    with pytest.raises(P.ProfileError, match="no input"):
        P.run_profile(FakeCtx(), [], 2, 1, make_ops())
    a = {"x": np.ones(2), "y": np.zeros(2)}
    assert P.bitwise_differences(a, {"x": np.ones(2), "y": np.zeros(2)}) == []
    assert P.bitwise_differences(a, {"x": np.ones(2), "y": -np.zeros(2)}) == ["y"]  # -0.0 differs bitwise
    assert P.bitwise_differences(a, {"x": np.ones(2)}) == ["y"]
    assert P.bitwise_differences(a, {"x": np.ones(2, dtype=np.float32), "y": np.zeros(2)}) == ["x"]


def full_capture(steps=2):
    cap = {"ledger": np.ones((steps, 2)), "counts": np.ones((steps, 3), dtype=np.int64), "flags": np.zeros(steps, dtype=np.uint64)}
    cap.update({f"map_{k}": np.ones((2, 2)) for k in ("cum_det", "cum_dep", "cum_clip", "mobile", "v_prev")})
    return cap


def test_a_capture_must_hold_every_required_field_and_the_fake_context_exposes_all_of_them():
    assert set(P.REQUIRED_FIELDS) == set(full_capture())
    ctx = FakeCtx()
    ctx.step(0, DEPTH, 0.5, 0.25)
    ctx.check_flags()
    assert set(P.capture(ctx, 1)) == set(P.REQUIRED_FIELDS)  # what a pass really captures is exactly what is required
    P.check_complete(full_capture(), "ok")
    # the root's reproducer: a capture holding only a zero flag word used to pass
    with pytest.raises(P.ProfileError, match="lacks the required"):
        P.check_complete({"flags": np.zeros(1, dtype=np.uint64)}, "missing-ledger/maps")
    for field in P.REQUIRED_FIELDS:
        cap = full_capture()
        del cap[field]
        with pytest.raises(P.ProfileError, match=field):
            P.check_complete(cap, f"missing {field}")
    with pytest.raises(P.ProfileError, match="empty capture"):
        P.check_complete({}, "x")


def test_present_but_unusable_fields_are_still_refused():
    cap = full_capture()
    cap["flags"][1] = 1
    with pytest.raises(P.ProfileError, match="flag"):
        P.check_complete(cap, "x")
    cap = full_capture()
    cap["ledger"] = np.zeros((0, 2))
    with pytest.raises(P.ProfileError, match="empty"):
        P.check_complete(cap, "x")
    for field in ("ledger", "map_mobile", "map_v_prev"):
        cap = full_capture()
        cap[field][0, 0] = np.inf
        with pytest.raises(P.ProfileError, match="non-finite"):
            P.check_complete(cap, "x")


class IncompleteCtx(FakeCtx):
    def download_maps(self):
        maps = super().download_maps()
        del maps["v_prev"]
        return maps


def test_a_context_that_returns_an_incomplete_capture_is_refused_by_the_whole_profile():
    with pytest.raises(P.ProfileError, match="map_v_prev"):
        P.run_profile(IncompleteCtx(), INPUTS, 2, 1, make_ops())


def test_the_provenance_guard_names_every_changed_item_and_publishes_nothing_on_a_change():
    same = {"a": ({"m": "1"}, {"m": "1"}), "b": ("x", "x")}
    P.require_unchanged(same)
    with pytest.raises(P.ProfileError, match="SYRUP and verified MAPLE package sources") as err:
        P.require_unchanged({**same, "SYRUP and verified MAPLE package sources": ({"f.py": "1"}, {"f.py": "2"}),
                             "state file": ("h", "h2")})
    assert "state file" in str(err.value) and "nothing is published" in str(err.value)
    assert "(a" not in str(err.value) and ", b" not in str(err.value)  # the unchanged items are not named


def wet_state(k=1, ny=2, nx=3):
    return {"depth_m": np.full((k, ny, nx), 1e-3), "velocity_m_s": np.full((k, ny, nx), 0.05), "rain_m_s": np.full((k, ny, nx), 1e-5)}


def ledger_with(columns, steps=3, nc=2, pickup=0.5, deposition=0.25, mobile=(2.0, 3.0)):
    led = np.zeros((steps, len(columns), nc))
    led[:, columns.index("pickup_kg"), :] = pickup
    led[:, columns.index("deposition_active_kg"), :] = deposition
    led[:, columns.index("old_mobile_kg"), :] = 7.0  # a different column: must not be read as the final mobile inventory
    led[:, columns.index("new_mobile_kg"), :] = 1.0
    led[-1, columns.index("new_mobile_kg"), :] = mobile
    return led


def test_nonvacuity_reports_per_class_totals_final_mobile_and_input_statistics_from_the_actual_ledger_columns():
    from maple_syrup.legacy_native_numba import LEDGER_COLUMNS

    columns = tuple(LEDGER_COLUMNS)
    assert columns.index("pickup_kg") == 0 and columns.index("deposition_active_kg") == 1 and columns.index("new_mobile_kg") == 7
    state = wet_state(k=2)
    state["depth_m"][0, 0, 0] = 0.0
    report = P.nonvacuity(ledger_with(columns), columns, state)
    assert report["pickup_kg_total_by_class"] == [1.5, 1.5] and report["pickup_kg_total"] == 3.0  # summed over the 3 steps
    assert report["deposition_active_kg_total_by_class"] == [0.75, 0.75] and report["deposition_active_kg_total"] == 1.5
    assert report["final_mobile_kg_by_class"] == [2.0, 3.0] and report["final_mobile_kg_total"] == 5.0  # the LAST row of new_mobile only
    assert report["columns_used"] == {"pickup_kg": 0, "deposition_active_kg": 1, "new_mobile_kg": 7}
    assert report["inputs_wet"] is True and report["pickup_nonzero"] is True and report["status"].startswith("wet:")
    depth = report["inputs"]["depth_m"]
    assert depth["n_states"] == 2 and depth["positive_cells_per_state"] == [5, 6] and depth["positive_cells_total"] == 11
    assert depth["min"] == 0.0 and depth["max"] == 1e-3 and report["inputs"]["rain_m_s"]["positive_cells_total"] == 12
    assert "no additional device read" in report["source"] and "not a conservation" in report["note"]
    assert json.dumps(report)  # serialisable


def test_a_dry_or_zero_pickup_profile_is_supported_but_identified_not_passed_off_as_wet():
    from maple_syrup.legacy_native_numba import LEDGER_COLUMNS

    columns = tuple(LEDGER_COLUMNS)
    dry = {k: np.zeros_like(v) for k, v in wet_state().items()}
    zero = P.nonvacuity(ledger_with(columns, pickup=0.0, deposition=0.0, mobile=(0.0, 0.0)), columns, dry)
    assert zero["status"].startswith("DRY") and zero["inputs_wet"] is False and zero["pickup_nonzero"] is False and zero["pickup_kg_total"] == 0.0
    wet_but_still = P.nonvacuity(ledger_with(columns, pickup=0.0, deposition=0.0, mobile=(0.0, 0.0)), columns, wet_state())
    assert wet_but_still["status"].startswith("wet inputs but ZERO pickup")
    assert P.nonvacuity(ledger_with(columns), columns, dry)["status"].startswith("DRY")  # sediment moved from a dry input is still flagged dry


def test_nonvacuity_refuses_malformed_ledgers_and_inputs():
    from maple_syrup.legacy_native_numba import LEDGER_COLUMNS

    columns = tuple(LEDGER_COLUMNS)
    good = ledger_with(columns)
    bad_ledgers = [good[:, :5, :], good[0], np.where(np.arange(good.size).reshape(good.shape) == 3, np.nan, good), good[:0]]
    for bad in bad_ledgers:
        with pytest.raises(P.ProfileError):
            P.nonvacuity(bad, columns, wet_state())
    with pytest.raises(P.ProfileError, match="lack"):
        P.nonvacuity(good, tuple(c for c in columns if c != "new_mobile_kg") + ("renamed",), wet_state())
    for key in P.INPUT_KEYS:
        broken = wet_state()
        del broken[key]
        with pytest.raises(P.ProfileError, match=key):
            P.nonvacuity(good, columns, broken)
        nonfinite = wet_state()
        nonfinite[key][0, 0, 0] = np.nan
        with pytest.raises(P.ProfileError):
            P.nonvacuity(good, columns, nonfinite)
    with pytest.raises(P.ProfileError):
        P.nonvacuity(good, columns, {k: v[0] for k, v in wet_state().items()})  # not (K, ny, nx)


def test_the_caveat_acknowledges_cycled_multi_state_inputs():
    text = " ".join(P.CAVEATS)
    assert "K > 1" in text and "cycled" in text and "K = 1" in text


def test_controls_and_output_paths_are_validated_before_any_work(tmp_path):
    P.validate_controls(60, 2, None, "compact")
    P.validate_controls(1, 1, 8.0, "all")
    for steps, repeats, mem, strat in ((0, 2, None, "compact"), (60, 0, None, "compact"), (True, 2, None, "compact"), (60, 2.0, None, "compact"),
                                       (60, 2, 0.0, "compact"), (60, 2, float("nan"), "compact"), (60, 2, -1.0, "compact"),
                                       (60, 2, None, "bogus")):
        with pytest.raises(P.ProfileError):
            P.validate_controls(steps, repeats, mem, strat)
    case = tmp_path / "case"
    case.mkdir()
    state = tmp_path / "state.npz"
    state.write_bytes(b"x")
    forbidden = {"case": case, "state file": state}
    good = tmp_path / "report.json"
    assert P.validate_output(good, forbidden) == good
    for bad in (case / "report.json", case, tmp_path, state, tmp_path / "missing_dir" / "r.json"):  # inside/equal/containing/missing parent
        with pytest.raises(P.OutputRefused):
            P.validate_output(bad, forbidden)
    fresh = tmp_path / "fresh.json"  # a NEW name whose only problem is containing a forbidden tree is covered by `tmp_path` above
    assert P.validate_output(fresh, {"case": case}) == fresh
    good.write_text("{}")
    with pytest.raises(P.OutputRefused, match="exists"):
        P.validate_output(good, forbidden)
    other = tmp_path / "other.json"
    (tmp_path / "other.json.FAILED").write_text("{}")
    with pytest.raises(P.OutputRefused, match="exists"):
        P.validate_output(other, forbidden)
    link = tmp_path / "link.json"
    link.symlink_to(case / "target.json")  # a dangling symlink into the case tree is not a free output name
    with pytest.raises(P.OutputRefused):
        P.validate_output(link, forbidden)


def test_main_writes_a_failure_record_only_for_a_validated_output_path(tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    argv = ["--case-kind", "plot1", "--case", str(tmp_path / "c"), "--state-npz", str(tmp_path / "s.npz"), "--output", str(out)]

    def refused(args):
        args.output_validated = False
        raise P.OutputRefused("no")

    monkeypatch.setattr(P, "_execute", refused)
    assert P.main(argv) == 1 and not out.exists() and not (tmp_path / "report.json.FAILED").exists()

    def failed_after_validation(args):
        args.output_validated = True
        raise P.ProfileError("instrumentation changed outputs")

    monkeypatch.setattr(P, "_execute", failed_after_validation)
    assert P.main(argv) == 1 and not out.exists()
    record = json.loads((tmp_path / "report.json.FAILED").read_text())
    assert "FIXED-WET-INPUT" in record["label"] and "instrumentation changed outputs" in record["error"]
    assert P.main(argv) == 1  # the existing failure record is never overwritten (exclusive create)


def test_the_label_and_caveats_state_what_the_profile_is_not_and_the_file_hash_helper(tmp_path):
    assert "FIXED-WET-INPUT ISOLATED SEDIMENT" in P.LABEL and "not a full storm" in P.LABEL and "not end-to-end" in P.LABEL
    assert any("not isolated kernel times" in c for c in P.CAVEATS) and any("outside the measured loops" in c for c in P.CAVEATS)
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert P.sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
