"""Compare a CPU legacy-replay output directory with a GPU one (same case, same conventions): task gpu_sediment, stages B1/B2.

    python benchmarks/legacy_gpu/compare_cpu_gpu.py --cpu <dir> --gpu <dir> --output <NEW report.json> [--gpu-repeat <dir>]

Predeclared bounds (never fitted, never relaxed to accept a run): sediment rtol 2e-11 / atol 1e-14 for the ledger, the cumulative maps, the
final mobile map and the mobile snapshots; water rtol 2e-12 / atol 1e-14 for the final depth/soil/discharge/velocity, the cumulative water
maps, the outlet/export series and the depth snapshots; integer walk/regime tallies, masks and control arrays EXACT.

Acceptance is strict, in this order, and every failure is a FLAG (the run is not accepted):
  0. when a receipt carries `output_pins` (size + SHA-256 of the closed archives) they are VERIFIED before the archives are read; a
     tampered/truncated body or a pin mismatch is an unusable artifact. Receipts without pins (older B1/CPU outputs) are accepted and
     qualified "unbound" in the report; the summary file itself is never pinned (self-reference) and the marker alone is never trusted;
  1. both directories must be readable and complete: every required array and summary field must exist (a missing field is a failure,
     never a skipped comparison);
  2. the saved receipts (summaries) must declare the SAME case, geometry, forcing pins, hydrology root solver and iteration controls,
     depth time level, time step, end time, source order, erase convention and snapshot request (backend, device and code may differ);
  3. every compared field must be an all-finite floating array of the same shape on both sides (a NaN or Inf is a failure: `nan > tol`
     is False, so non-finiteness is tested explicitly before any tolerance);
  4. time series and the requested snapshots are compared exactly like the end-of-run fields;
  5. the clipped-pool identity residual is a CANCELLATION DIAGNOSTIC derived from six ledger columns, not an independent physical field:
     each backend's stored residual must be finite, must equal the residual recomputed bitwise from its own ledger, and must pass the
     unchanged `legacy_driver._check_identity` guard (1e-10) on its own; the cross-backend difference is bounded ONLY by the already
     declared sediment bounds propagated through the formula (triangle inequality over the six unit-coefficient columns) plus the
     standard floating-point evaluation error of the formula, and is reported; a violation is flagged, never explained away;
  6. `--gpu-repeat` demands byte-identical repeatability of EVERY saved array (ledger archive and snapshots) and of the saved summary
     counters/totals.
A difference is reported with its location; it is never explained away by a CPU root-to-root noise floor.
Nothing here was run by its author.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

SEDIMENT = (2.0e-11, 1.0e-14)
WATER = (2.0e-12, 1.0e-14)
IDENTITY_GUARD_RTOL = 1.0e-10  # documented only: the guard itself is `legacy_driver._check_identity`, unchanged
UNIT_ROUNDOFF = 2.0 ** -53
SEDIMENT_KEYS = ("ledger", "cumulative_detachment_kg", "cumulative_deposition_kg", "cumulative_clipping_source_kg", "final_mobile_kg")
WATER_KEYS = ("final_depth_m", "final_soil_water_m", "final_discharge_m2_s", "final_velocity_m_s", "cumulative_rain_m",
              "cumulative_intake_m", "cumulative_saturation_return_m", "cumulative_drainage_m", "water_outlet_m3_s", "water_export_m3")
#: cancellation diagnostics: finite and individually guarded, never an independent physical field
DIAGNOSTIC_KEYS = ("identity_residual_kg", "water_budget_residual_m3")
INTEGER_KEYS = ("regime_counts", "active", "terminal_storage", "outlet")
CONTROL_KEYS = ("t_s",)
REQUIRED = (*SEDIMENT_KEYS, *WATER_KEYS, *DIAGNOSTIC_KEYS, *INTEGER_KEYS, *CONTROL_KEYS, "walk_counts", "columns")
IDENTITY_COLUMNS = ("pickup_kg", "deposition_active_kg", "effective_clip_source_kg", "old_mobile_kg", "new_mobile_kg", "cn_export_kg")
SUMMARY_COUNTERS = ("totals_by_class", "totals", "walk_tallies", "wet_regime_cell_class_steps", "onset", "identity",
                    "final_mobile_by_class_kg", "maps_total_kg")
MISSING = "<missing>"


class HarnessError(RuntimeError):
    """An artifact is unreadable or incomplete (reported as a flag by `compare`)."""


def field_report(a: Any, b: Any, rtol: float, atol: float) -> dict:
    """`a` = the tested (GPU) array, `b` = the reference (CPU). Both must be all-finite floating arrays of one shape."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind != "f" or b.dtype.kind != "f":
        return {"pass": False, "reason": f"not floating point ({a.dtype}, {b.dtype})"}
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)], "pass": False, "reason": "shape mismatch"}
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    if not (finite_a.all() and finite_b.all()):  # NaN/Inf can never pass: `nan > tol` is False
        return {"pass": False, "reason": "non-finite values", "n_nonfinite_tested": int((~finite_a).sum()),
                "n_nonfinite_reference": int((~finite_b).sum()), "n": int(a.size)}
    a, b = a.astype(np.float64), b.astype(np.float64)
    diff = np.abs(a - b)
    allowed = atol + rtol * np.abs(b)
    bad = diff > allowed
    k = int(np.argmax(diff)) if diff.size else 0
    return {"max_abs": float(diff.max()) if diff.size else 0.0,
            "max_abs_location": [int(v) for v in np.unravel_index(k, a.shape)] if diff.size else [],
            "n_bad": int(bad.sum()), "n": int(diff.size), "pass": bool(not bad.any())}


def exact_report(a: Any, b: Any) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"pass": False, "reason": "shape mismatch", "shape_mismatch": [list(a.shape), list(b.shape)]}
    if (a.dtype.kind in "fc" or b.dtype.kind in "fc") and not (np.isfinite(a).all() and np.isfinite(b).all()):
        return {"pass": False, "reason": "non-finite values"}
    return {"pass": bool(np.array_equal(a, b)), "n": int(a.size)}


PIN_FILES = ("legacy_ledger.npz", "legacy_snapshots.npz")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_output_pins(directory: Path, summary: dict) -> dict:
    """Verify the summary's `output_pins` (archive size and SHA-256, never the marker alone) WHEN present. A summary without pins (older
    B1/CPU outputs) is accepted but qualified `unbound`: its archives are not bound to their receipt. Any mismatch is a HarnessError."""
    pins = summary.get("output_pins")
    if pins is None:
        return {"status": "unbound legacy output (no output_pins in the receipt): archives are not integrity-pinned", "verified": False}
    files = pins.get("files") if isinstance(pins, dict) else None
    if not isinstance(files, dict) or "legacy_ledger.npz" not in files:
        raise HarnessError(f"{directory}: output_pins is malformed or does not pin legacy_ledger.npz")
    for name in PIN_FILES:  # a data archive that exists but is not pinned is as suspicious as a changed one
        if (directory / name).is_file() and name not in files:
            raise HarnessError(f"{directory / name} exists but is not listed in output_pins")
    for name, pin in files.items():
        path = directory / name
        if not isinstance(pin, dict) or not path.is_file():
            raise HarnessError(f"{path} is pinned but missing or its pin is malformed")
        size = path.stat().st_size
        if size != pin.get("bytes"):
            raise HarnessError(f"{path} has {size} bytes, its pin declares {pin.get('bytes')} (truncated or altered after publication)")
        if _sha256_file(path) != pin.get("sha256"):
            raise HarnessError(f"{path} does not match its pinned SHA-256 (altered after publication)")
    return {"status": "pinned and verified (size and SHA-256 of the archives; the summary itself is not pinned)", "verified": True,
            "files": sorted(files)}


def load_run(directory: Path) -> dict:
    """Arrays (materialised), the summary and the optional snapshots of one run directory; anything missing is a HarnessError."""
    ledger_path, summary_path = directory / "legacy_ledger.npz", directory / "legacy_summary.json"
    for path in (ledger_path, summary_path):
        if not path.is_file():
            raise HarnessError(f"{path} is missing")
    try:
        summary = json.loads(summary_path.read_text())
    except Exception as exc:  # a truncated or malformed receipt
        raise HarnessError(f"{summary_path} is unreadable ({type(exc).__name__}: {exc})") from exc
    pin_status = verify_output_pins(directory, summary)  # BEFORE the archives are decompressed: a changed body is named as such
    try:
        with np.load(ledger_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}  # decompression happens here: a truncated archive fails inside this block
        snaps = None
        snap_path = directory / "legacy_snapshots.npz"
        if snap_path.is_file():
            with np.load(snap_path, allow_pickle=False) as z:
                snaps = {k: z[k] for k in z.files}
    except Exception as exc:  # any read failure (truncation, bad zip, bad JSON) means an unusable artifact
        raise HarnessError(f"{directory} is unreadable ({type(exc).__name__}: {exc})") from exc
    missing = [k for k in REQUIRED if k not in arrays]
    if missing:
        raise HarnessError(f"{ledger_path} lacks required arrays {missing}")
    return {"dir": directory, "arrays": arrays, "summary": summary, "snapshots": snaps, "pin_status": pin_status}


def _get(d: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return MISSING
        d = d[key]
    return d


CONVENTIONS = {
    "case_kind": ("case_kind",), "steps": ("steps",), "dt_s": ("dt_s",), "end_s": ("end_s",),
    "depth_time_level": ("depth_time_level",), "law_depth_selected": ("law_depth", "selected"),
    "source_order": ("config", "source_order"), "legacy_depos_erase": ("config", "legacy_depos_erase"),
    "snapshot_times_requested": ("config", "snapshot_times"),
    "root_solver": ("hydrology", "root_solver"), "bisection_iterations": ("hydrology", "bisection_iterations"),
    "newton_max_iterations": ("hydrology", "newton_max_iterations"),
    "graph_input_sha256": ("provenance", "graph_input_sha256"), "network": ("network",),
}


def convention_record(summary: dict) -> dict:
    out = {name: _get(summary, path) for name, path in CONVENTIONS.items()}
    pins = _get(summary, ("artifact_pins", "sha256"))
    out["bound_input_hashes"] = sorted(pins.values()) if isinstance(pins, dict) else MISSING
    return out


def check_receipts(tested: dict, reference: dict) -> tuple[dict, list[str]]:
    """Matched case/geometry/forcing/solver/time-level/step/end/order/erase/snapshot conventions from the saved summaries."""
    a, b = convention_record(tested["summary"]), convention_record(reference["summary"])
    flags, detail = [], {}
    for name in a:
        same = a[name] == b[name] and a[name] != MISSING
        detail[name] = {"tested": a[name], "reference": b[name], "match": bool(same)}
        if not same:
            reason = "missing from a receipt" if MISSING in (a[name], b[name]) else "differs"
            flags.append(f"receipt convention {name} {reason}")
    for label, run in (("tested", tested), ("reference", reference)):
        steps = _get(run["summary"], ("steps",))
        ledger = run["arrays"]["ledger"]
        if isinstance(steps, int) and ledger.ndim >= 1 and ledger.shape[0] != steps:
            flags.append(f"{label} ledger has {ledger.shape[0]} rows but its receipt declares {steps} steps")
    return detail, flags


def identity_diagnostic(tested: dict, reference: dict) -> tuple[dict, list[str]]:
    """The clipped-pool identity residual, separately from the physical fields (see the module docstring, item 5)."""
    from maple_syrup import legacy_driver as D  # the unchanged guard

    flags: list[str] = []
    out: dict = {"guard_rtol": IDENTITY_GUARD_RTOL, "formula": "new - old - (pickup - deposition_active) + cn_export - clip",
                 "bound": "sum over the six unit-coefficient ledger columns of (atol + rtol |reference|) at the declared sediment bound, "
                          "plus 5u/(1-5u) x sum|terms| per backend (u = 2^-53) for the evaluation of the formula"}
    resid = {}
    for label, run in (("tested", tested), ("reference", reference)):
        ledger = run["arrays"]["ledger"]
        columns = [str(c) for c in run["arrays"]["columns"]]
        if ledger.ndim != 3 or ledger.shape[1] != len(columns) or not np.isfinite(ledger).all() \
                or any(name not in columns for name in IDENTITY_COLUMNS):
            flags.append(f"identity: {label} ledger is malformed, non-finite or lacks an identity column")
            return out, flags
        stored = run["arrays"]["identity_residual_kg"]
        if not np.isfinite(stored).all():
            flags.append(f"identity: {label} stored residual is non-finite")
            return out, flags
        try:  # the ledger is (steps, columns, classes); the unchanged guard returns the per-step, per-class residual
            recomputed, worst = D._check_identity(ledger)
        except D.DriverError as exc:
            flags.append(f"identity: {label} violates the unchanged guard ({exc})")
            return out, flags
        resid[label] = recomputed
        out[f"{label}_worst_relative_residual"] = worst
        if recomputed.shape != stored.shape or not np.array_equal(recomputed, stored):
            flags.append(f"identity: {label} stored residual does not equal the residual recomputed from its own ledger")
    ta, ra = tested["arrays"], reference["arrays"]
    cols = [str(c) for c in ta["columns"]]
    if cols != [str(c) for c in ra["columns"]] or ta["ledger"].shape != ra["ledger"].shape:
        flags.append("identity: ledger column names or shapes differ between the runs")
        return out, flags
    terms_t = np.stack([ta["ledger"][:, cols.index(name)] for name in IDENTITY_COLUMNS])  # (6, steps, classes)
    terms_r = np.stack([ra["ledger"][:, cols.index(name)] for name in IDENTITY_COLUMNS])
    propagate = (SEDIMENT[1] + SEDIMENT[0] * np.abs(terms_r)).sum(axis=0)
    gamma = 5.0 * UNIT_ROUNDOFF / (1.0 - 5.0 * UNIT_ROUNDOFF)
    roundoff = gamma * (np.abs(terms_t).sum(axis=0) + np.abs(terms_r).sum(axis=0))
    allowed = propagate + roundoff
    diff = np.abs(resid["tested"] - resid["reference"])
    bad = diff > allowed
    out.update({"max_abs_difference_kg": float(diff.max()) if diff.size else 0.0, "n_values": int(diff.size),
                "n_violating_the_propagated_bound": int(bad.sum()), "pass": bool(not bad.any())})
    if bad.any():
        flags.append("identity residual differs across backends by more than the propagated declared bounds (kept flagged)")
    return out, flags


def snapshot_checks(tested: dict, reference: dict, report: dict, flags: list[str]) -> None:
    requested = _get(tested["summary"], ("config", "snapshot_times"))
    if requested in ("", None, MISSING, []):
        if tested["snapshots"] is not None or reference["snapshots"] is not None:
            flags.append("a snapshot archive exists although no snapshot times were requested")
        return
    if tested["snapshots"] is None or reference["snapshots"] is None:
        flags.append("snapshots were requested but a snapshot archive is missing")
        return
    a, b = tested["snapshots"], reference["snapshots"]
    for key in ("t_s", "depth_m", "mobile_kg"):
        if key not in a or key not in b:
            flags.append(f"snapshot array {key} is missing")
            return
    report["snapshots"] = {"t_s": exact_report(a["t_s"], b["t_s"]),
                           "depth_m": field_report(a["depth_m"], b["depth_m"], *WATER),
                           "mobile_kg": field_report(a["mobile_kg"], b["mobile_kg"], *SEDIMENT)}
    flags.extend(f"snapshot {k} failed ({v.get('reason', 'outside its predeclared bound')})"
                 for k, v in report["snapshots"].items() if not v["pass"])


def repeat_check(tested: dict, repeat: dict) -> tuple[dict, list[str]]:
    """Byte-identical repeatability of every saved array and of the saved summary counters."""
    result: dict = {"arrays": {}, "snapshots": {}, "summary_counters": {}}
    flags: list[str] = []
    a, b = tested["arrays"], repeat["arrays"]
    if set(a) != set(b):
        flags.append("the repeat run saved a different set of arrays")
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        same = x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()
        result["arrays"][key] = bool(same)
        if x.dtype.kind == "f" and not np.isfinite(x).all():
            same = False
            result["arrays"][key] = False
        if not same:
            flags.append(f"the GPU repeat differs (or is non-finite) in {key}")
    sa, sb = tested["snapshots"], repeat["snapshots"]
    if (sa is None) != (sb is None):
        flags.append("the GPU repeat disagrees on the presence of snapshots")
    elif sa is not None:
        for key in sorted(set(sa) | set(sb)):
            same = key in sa and key in sb and sa[key].tobytes() == sb[key].tobytes() and sa[key].shape == sb[key].shape
            result["snapshots"][key] = bool(same)
            if not same:
                flags.append(f"the GPU repeat differs in snapshot field {key}")
    for key in SUMMARY_COUNTERS:
        same = _get(tested["summary"], (key,)) == _get(repeat["summary"], (key,)) != MISSING
        result["summary_counters"][key] = bool(same)
        if not same:
            flags.append(f"the GPU repeat differs in the saved summary counter {key}")
    return result, flags


def compare(cpu: Path, gpu: Path, repeat: Path | None = None) -> dict:
    report: dict = {"bounds": {"sediment": SEDIMENT, "water": WATER}, "fields": {}, "integers": {}, "flags": []}
    try:
        reference, tested = load_run(cpu), load_run(gpu)
        repeat_run = load_run(repeat) if repeat is not None else None
    except HarnessError as exc:
        report["flags"].append(f"unusable artifact: {exc}")
        report["accepted_by_declared_bounds"] = False
        return report
    flags = report["flags"]
    report["output_pins"] = {"cpu": reference["pin_status"], "gpu": tested["pin_status"],
                             **({"gpu_repeat": repeat_run["pin_status"]} if repeat_run is not None else {})}
    report["scope"] = ("legacy replay (fixed composition, unlimited supply, explicit clipping source, no MAPLE bed): a comparison of "
                       "THIS case at the unchanged predeclared bounds; no conservation claim; unpinned (legacy) outputs are qualified "
                       "'unbound', not rejected")
    c, g = reference["arrays"], tested["arrays"]
    report["receipts"], receipt_flags = check_receipts(tested, reference)
    flags.extend(receipt_flags)
    if repeat_run is not None:
        _, repeat_receipt_flags = check_receipts(repeat_run, tested)
        flags.extend(f"repeat run: {f}" for f in repeat_receipt_flags)
    if receipt_flags:  # a numeric comparison of runs with different declared conventions would be meaningless
        report["accepted_by_declared_bounds"] = False
        return report
    for key in SEDIMENT_KEYS:
        report["fields"][key] = field_report(g[key], c[key], *SEDIMENT)
    for key in WATER_KEYS:
        report["fields"][key] = field_report(g[key], c[key], *WATER)
    report["integers"]["t_s"] = exact_report(g["t_s"], c["t_s"])["pass"]
    wc_g, wc_c = g["walk_counts"], c["walk_counts"]
    report["integers"]["walk_counts_without_erase_cells"] = bool(wc_g.shape == wc_c.shape and wc_g.ndim == 2 and wc_g.shape[1] >= 10
                                                                 and np.array_equal(wc_g[:, :9], wc_c[:, :9]))
    report["integers"]["erase_cell_counts"] = bool(wc_g.shape == wc_c.shape and wc_g.ndim == 2 and wc_g.shape[1] >= 10
                                                   and np.array_equal(wc_g[:, 9], wc_c[:, 9]))
    for key in INTEGER_KEYS:
        report["integers"][key] = exact_report(g[key], c[key])["pass"]
    for key, v in report["fields"].items():
        if not v["pass"]:
            flags.append(f"field {key} outside its predeclared bound ({v.get('reason', 'bound exceeded')})")
    flags.extend(f"integer {k} differs" for k, ok in report["integers"].items() if not ok)
    report["identity_diagnostic"], identity_flags = identity_diagnostic(tested, reference)
    flags.extend(identity_flags)
    budget = {label: run["arrays"]["water_budget_residual_m3"] for label, run in (("tested", tested), ("reference", reference))}
    report["water_budget_diagnostic"] = {label: {"finite": bool(np.isfinite(v).all()),
                                                 "max_abs_m3": float(np.abs(v).max()) if v.size and np.isfinite(v).all() else None,
                                                 "whole_storm_closed": _get(run["summary"], ("water", "whole_storm_budget", "closed"))}
                                         for (label, v), run in zip(budget.items(), (tested, reference), strict=True)}
    for label, rec in report["water_budget_diagnostic"].items():
        if not rec["finite"] or rec["whole_storm_closed"] is not True:
            flags.append(f"water budget diagnostic of the {label} run is non-finite or not closed")
    snapshot_checks(tested, reference, report, flags)
    cs, gs = reference["summary"], tested["summary"]
    report["totals"] = {"cpu": cs.get("totals"), "gpu": gs.get("totals")}
    report["onset"] = {"cpu": cs.get("onset"), "gpu": gs.get("onset")}
    cp_, gp_ = cs.get("performance", {}), gs.get("performance", {})
    cl, gl = cp_.get("loop_wall_s_excluding_progress"), gp_.get("loop_wall_s_excluding_progress")
    report["timing"] = {"cpu_loop_s": cl, "gpu_loop_s": gl, "speedup_loop": (cl / gl) if cl and gl else None,
                        "cpu_warmup": cp_.get("warmup"), "gpu_warmup": gp_.get("warmup"),
                        "gpu_cold": {k: gp_.get(k) for k in ("hydrology_prepare_s", "sediment_context_build_s",
                                                             "sediment_kernel_compile_and_tables_s", "first_step_s")},
                        "gpu_event_split": gp_.get("event_split_s"), "gpu_peak_rss_kib": gp_.get("peak_rss_kib"),
                        "cpu_peak_rss_kib": cp_.get("peak_rss_kib"),
                        "note": "compare steady-state loops after a warm-up window; cold costs are listed separately"}
    gpu_block = gs.get("gpu", {})
    report["gpu"] = {"device": gpu_block.get("device"), "memory": gpu_block.get("memory"), "transfers": gpu_block.get("transfers"),
                     "walk_tables": gs.get("walk_tables"), "record_strategy": gpu_block.get("record_strategy"),
                     "launches_per_step_sediment": gpu_block.get("launches_per_step_sediment")}
    if repeat_run is not None:
        report["gpu_repeatability_bitwise"], repeat_flags = repeat_check(tested, repeat_run)
        flags.extend(repeat_flags)
    report["accepted_by_declared_bounds"] = not flags
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cpu", type=Path, required=True)
    p.add_argument("--gpu", type=Path, required=True)
    p.add_argument("--gpu-repeat", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if os.path.lexists(a.output):
        print(f"refusing to overwrite {a.output}", file=sys.stderr)
        return 1
    report = compare(a.cpu, a.gpu, a.gpu_repeat)
    with a.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"accepted_by_declared_bounds": report["accepted_by_declared_bounds"], "flags": report["flags"],
                      "speedup_loop": (report.get("timing") or {}).get("speedup_loop")}, indent=2))
    return 0 if report["accepted_by_declared_bounds"] else 2


if __name__ == "__main__":
    sys.exit(main())
