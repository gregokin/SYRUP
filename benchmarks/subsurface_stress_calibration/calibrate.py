"""Pooled calibration of ONE shared evaporation-stress threshold against the JORN4_A / JORN4_B / MOAB2_C observed dry-downs (stage-1 analysis helper; every window is used for fitting, so a fit is a MECHANISM DIAGNOSTIC,
not an independent field validation).

Every objective evaluation runs the ACTUAL native column (`drydown_driver.run_column`, compiled Numba backend, three prepared windows) with E = 1, T = 0, open bottom, gravity on and the given stress; the inputs are
prepared ONCE per window and the stress evaluations are cached by exact value. Scores compare model CHANGES with observed CHANGES (`scoring`): a constant per-(window, sensor) offset, t0 excluded, the pooled
objective weighting every valid observation equally. The offset never alters the physical state. The calibrated quantity is an EMPIRICAL stress threshold in volumetric fraction, not a fitted measured theta1500/wilting point.

    python -m benchmarks.subsurface_stress_calibration.calibrate --out NEWDIR                 # plan: write + validate the native configs only (no raw data read, no model run)
    python -m benchmarks.subsurface_stress_calibration.calibrate --out NEWDIR --execute [--verify-cuda]   # Root only: prepare, fit, controls, verification (CUDA only with the explicit flag, never probed)
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.subsurface_site_comparison import study as S
from benchmarks.subsurface_validation import drydown_driver as dd

from . import scoring as SC
from . import search as SE

PROTOCOL_ID = "syrup.evaporation_stress_calibration/1"
WINDOWS = ("JORN4_A", "JORN4_B", "MOAB2_C")
E_FACTOR = 1.0
BASELINE_STRESS = 0.18
FITS = {"main": ("harmonic", 0.05), "secondary": ("geometric", 0.05), "sensitivity_d0": ("harmonic", 0.0)}
EXPECTED_SOURCES = {"maple_syrup": {"package_sha256": "37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0", "files": 180},
                    "maple": {"package_sha256": "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e", "files": 412}}
LABEL = ("empirical evaporation-stress threshold (volumetric fraction) calibrated to pooled observed changes; NOT the measured theta1500 / wilting point; all three windows are used in the fit (mechanism diagnostic, no held-out claim)")


class CalibrationError(RuntimeError):
    pass


def scenario(d: float) -> dict:
    return {"name": f"fit_d{d:g}_E{E_FACTOR:g}_open", "diffusion_coefficient": d, "evaporation_factor": E_FACTOR, "potential_et": "weather", "bottom": "open"}


def build_configs(method: str, d: float, stress: float | None = None, receipt: dict | None = None) -> dict:
    """Raw (JSON-able) driver configs per window: `study.make_config` (frozen window blocks, site-derived layers, interface coefficients of `method`) with the single E = 1 scenario of diffusion `d`, the explicit new
    provenance and, when `stress` is given, that threshold for every window (None keeps each site's own theta1500, the site-specific control). Each is validated by the driver and its pinned sources must be the frozen ones."""
    receipt = S.load_receipt() if receipt is None else receipt
    out = {}
    for w in WINDOWS:
        cfg = S.make_config(w, method, [scenario(d)], receipt)
        if cfg["pins"]["expected_sources"] != EXPECTED_SOURCES:
            raise CalibrationError(f"{w}: the template's expected sources differ from the frozen production identities")
        theta1500 = cfg["layers"]["theta_evaporation_stress"]
        if stress is not None:
            cfg["layers"]["theta_evaporation_stress"] = float(stress)
        cfg["name"] = f"{w} pooled evaporation-stress calibration, {method}, d={d:g}, E={E_FACTOR:g}"
        prov = cfg["provenance"]
        prov["historical_template_role"] = prov.get("role")                       # the earlier train/held-out role of the frozen template, history only
        prov["role"] = "calibration_all_windows"
        prov["evaporation_stress"] = ("one shared constant calibrated to the pooled observations of all three windows (empirical threshold, not theta1500)" if stress is not None else
                                      "site-specific predicted top-layer theta1500 (no-refit control)")
        prov["historical_evaporation_stress"] = "top-layer predicted theta1500, fixed within a run (earlier comparison; superseded for this protocol)"
        prov["stress_calibration"] = {"protocol": PROTOCOL_ID, "label": LABEL, "stress": cfg["layers"]["theta_evaporation_stress"], "site_theta1500": theta1500,
                                                   "stress_source": "calibrated shared constant" if stress is not None else "site-specific predicted theta1500 (control)", "E": E_FACTOR, "d": d, "T": 0}
        dd.validate_config(cfg)
        out[w] = cfg
    return out


def with_stress(cfg: dict, stress: float) -> dict:
    """A validated (normalised) config with a different stress; nothing else changes."""
    return {**cfg, "layers": {**cfg["layers"], "theta_evaporation_stress": float(stress)}}


def depth_labels(cfg: dict, prep: dict) -> list:
    return [f"{v}@{z:.3f}m" for v, z in zip(dd.SENSOR_VERS, prep["sensor_depths_m"])]


def window_score_from_run(cfg: dict, prep: dict, run: dict) -> dict:
    """Offset-aligned change score of one native run: the layer CONTAINING each probe (as the driver), initial model value = the integrated initial layer mean, initial observation = the raw point value."""
    probe = dd.probe_layers(cfg, prep)
    lt = np.asarray(run["layer_theta"], dtype=float)
    model = np.array([lt[:, li] for li in probe])
    obs = np.array([prep["obs"][v] for v in dd.SENSOR_VERS], dtype=float)
    return SC.window_score(model, np.asarray(prep["initial"]["layer_theta"], dtype=float)[probe], obs, np.asarray(prep["theta0_points"], dtype=float))


class Pooled:
    """Objective over the three prepared windows with exact-value caching of the native runs."""

    def __init__(self, cfgs: dict, preps: dict, backend: str = "numba"):
        self.cfgs, self.preps, self.backend = cfgs, preps, backend
        self.scen = {w: cfgs[w]["scenarios"][0] for w in cfgs}
        self.runs: dict = {}

    def evaluate(self, stress: float) -> dict:
        key = float(stress)
        if key not in self.runs:
            runs = {w: dd.run_column(with_stress(self.cfgs[w], key), self.preps[w], self.scen[w], self.backend) for w in self.cfgs}
            self.runs[key] = {"runs": runs, "windows": {w: window_score_from_run(self.cfgs[w], self.preps[w], runs[w]) for w in runs}}
        r = self.runs[key]
        return {**SC.pooled_objective(r["windows"]), "stress": key}

    def objective(self, stress: float) -> float:
        return self.evaluate(stress)["mse"]


def run_record(run: dict) -> dict:
    return {"trajectory": dd.jsonable(run["trajectory"]), "layer_theta": dd.jsonable(run["layer_theta"]), "initial_storage_m": run["initial_storage_m"], "final_storage_m": run["final_storage_m"],
            "cumulative_budget": dd.jsonable(dataclasses.asdict(run["cumulative"])), "cumulative_ledger": dd.jsonable(run["ledger"])}


def score_record(cfgs: dict, preps: dict, windows: dict) -> dict:
    """Everything a report needs from a pooled score, with the offsets, aligned model, observations and unshifted RMSE kept beside the aligned residuals."""
    sites = {w: cfgs[w]["site"]["site"] for w in cfgs}
    labels = list(dd.SENSOR_VERS)                                              # aggregate by vertical sensor ID only; actual per-window depths stay in `windows`
    per = {}
    for w, ws in windows.items():
        lab = depth_labels(cfgs[w], preps[w])
        per[w] = {"depth_labels": lab, "sensors": [{k: dd.jsonable(v) for k, v in s.items()} | {"observed": dd.jsonable(preps[w]["obs"][v]), "label": lab[i], "depth_m": float(preps[w]["sensor_depths_m"][i])} for i, (s, v) in enumerate(zip(ws["sensors"], dd.SENSOR_VERS))]}
    return {"pooled": SC.pooled_objective(windows), "breakdown": SC.breakdown(windows, sites, labels), "windows": per, "t0_excluded": True}


def physical_flags(cfgs: dict, stress: float) -> dict:
    """Where the selected threshold sits against each site's top-layer FC and saturation (a statement of position only, no physical meaning invented)."""
    out = {}
    for w, c in cfgs.items():
        fc, sat = float(c["layers"]["theta_field_capacity"][0]), float(c["layers"]["theta_saturation"][0])
        out[w] = {"top_layer_fc": fc, "top_layer_saturation": sat, "above_fc": stress > fc, "above_saturation": stress > sat, "site_theta1500": c["provenance"]["stress_calibration"]["site_theta1500"]}
    return out


def verify_case(cfgs: dict, preps: dict, stress: float | None, *, verify_cuda: bool) -> dict:
    """NumPy/Numba (and, only when asked, CUDA) agreement on whole trajectories and midpoint public-state export/reconstruction on every declared backend. Raises on failure."""
    out = {}
    for w, c in cfgs.items():
        cc = c if stress is None else with_stress(c, stress)
        scen, ctl, prep = cc["scenarios"][0], cc["controls"], preps[w]
        if ctl["restart_after_day"] is None or ctl["restart_after_day"] >= prep["n_days"]:
            raise dd.ConfigError(f"{w}: controls.restart_after_day must be inside the modelled days")
        names = ["numpy", "numba"] + (["cuda"] if verify_cuda else [])
        runs = {b: dd.run_column(cc, prep, scen, b) for b in names}
        bitwise = scen["diffusion_coefficient"] == 0.0
        agree = {f"{a}_vs_{b}": dd.assert_backends_agree(runs[a], runs[b], rtol=ctl["numba_rtol"], atol=ctl["numba_atol"], bitwise=bitwise and "cuda" not in (a, b), what=f"{w} {a} vs {b}")
                 for i, a in enumerate(names) for b in names[i + 1:]}
        for b in names:
            dd.assert_runs_identical(runs[b], dd.run_column(cc, prep, scen, b, restart_after_day=ctl["restart_after_day"]), f"{w} {b} midpoint restart")
        out[w] = {"backends": names, "backend_agreement_max_abs": agree, "restart_after_day": ctl["restart_after_day"], "restart_label": dd.RESTART_LABEL, "ledgers": {b: dd.jsonable(runs[b]["ledger"]) for b in names}}
    return out


def raw_guard(cfgs: dict) -> dict:
    """Re-hash the pinned inventory, selected raw files and air manifests (used before preparation indirectly by `prepare_inputs` and again after every evaluation)."""
    out = {}
    for w, cfg in cfgs.items():
        inv_bytes = Path(dd.P.INVENTORY).read_bytes()
        if hashlib.sha256(inv_bytes).hexdigest() != cfg["pins"]["inventory_sha256"]:
            raise dd.ni.InputError("the inventory file differs from its pin")
        n_raw = dd.ni.verify_inventory_files(json.loads(inv_bytes), dd.P.RAW_ROOT)
        air_root = getattr(dd.P, cfg["air"]["root"])
        if hashlib.sha256((air_root / "manifest.json").read_bytes()).hexdigest() != cfg["pins"]["air_manifest_sha256"]:
            raise dd.ni.InputError("the air manifest differs from its pin")
        out[w] = {"raw_files_checked": n_raw, "air_files_checked": dd.ni.verify_manifest_files(air_root / "manifest.json", air_root)}
    return out


def source_guard(cfgs: dict) -> dict:
    rec = dd.source_receipt()
    for c in cfgs.values():
        dd.assert_expected_sources(c, rec)
    return rec


def _write_new(path: Path, doc: dict) -> str:
    text = json.dumps(dd.jsonable(doc), indent=1, allow_nan=False) + "\n"
    if path.exists():
        raise CalibrationError(f"{path} exists; nothing is overwritten")
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def case_summary(pooled: dict, cfgs: dict, preps: dict, windows: dict) -> dict:
    return {"objective": {k: pooled[k] for k in ("n", "sse", "mse", "rmse", "zero_change_rmse", "skill_vs_zero_change")}, **score_record(cfgs, preps, windows)}


def execute(out: Path, *, verify_cuda: bool = False) -> dict:
    """Root's run: prepare once, source/raw guards, three fits, controls, verification, export. Everything is written under `out` (new, empty)."""
    out = Path(out)
    plan(out)                                                                  # writes configs; refuses a non-empty directory
    prep_dir, fit_dir = out / "prepared_inputs", out / "fits"
    prep_dir.mkdir(); fit_dir.mkdir()
    base = build_configs(*FITS["main"], stress=BASELINE_STRESS)
    cfgs0 = {w: dd.validate_config(c) for w, c in base.items()}
    src_before, raw_before = source_guard(cfgs0), raw_guard(cfgs0)
    preps = {w: dd.prepare_inputs(cfgs0[w]) for w in WINDOWS}                    # once per window
    shas = {"prepared_inputs": {w: _write_new(prep_dir / f"{w}.json", dd.prepared_inputs_export(cfgs0[w], preps[w])) for w in WINDOWS}}
    summary = {"protocol": PROTOCOL_ID, "label": LABEL, "stress_range": [SE.STRESS_MIN, SE.STRESS_MAX], "fits": {}, "sources_before": src_before, "raw_before": raw_before}
    for name, (method, d) in FITS.items():
        cfgs = {w: dd.validate_config(c) for w, c in build_configs(method, d, stress=BASELINE_STRESS).items()}
        pooled = Pooled(cfgs, preps)
        res = SE.search(pooled.objective)
        sel = res["selected"]["stress"]
        near = SE.near_optimal_intervals(res["profile"]["stress"], res["profile"]["objective"])
        chosen = pooled.evaluate(sel)
        controls = {}
        for cname, cs in (("baseline_0.18", BASELINE_STRESS), ("site_theta1500", None)):
            ccfgs = {w: dd.validate_config(c) for w, c in build_configs(method, d, stress=cs).items()}
            cp = Pooled(ccfgs, preps)
            ev = cp.evaluate(ccfgs[WINDOWS[0]]["layers"]["theta_evaporation_stress"]) if cs is not None else None
            if cs is None:                                                    # per-site thresholds differ: run each window with its own value
                runs = {w: dd.run_column(ccfgs[w], preps[w], cp.scen[w], "numba") for w in WINDOWS}
                wins = {w: window_score_from_run(ccfgs[w], preps[w], runs[w]) for w in WINDOWS}
                ev = SC.pooled_objective(wins)
            else:
                wins = cp.runs[float(cs)]["windows"]
            controls[cname] = case_summary(ev, ccfgs, preps, wins)
            controls[cname]["verification"] = verify_case(ccfgs, preps, cs, verify_cuda=verify_cuda)
        verification = verify_case(cfgs, preps, sel, verify_cuda=verify_cuda)
        if d == 0.0:                                                          # at d = 0 the interface conductivity is unused: geometric must reproduce harmonic exactly
            g = {w: dd.validate_config(c) for w, c in build_configs("geometric", 0.0, stress=sel).items()}
            for w in WINDOWS:
                dd.assert_runs_identical(pooled.runs[sel]["runs"][w], dd.run_column(g[w], preps[w], g[w]["scenarios"][0], "numba"), f"{w} d=0 harmonic vs geometric")
            verification["d0_harmonic_geometric_identical"] = True
        fit_doc = {"fit": name, "method": method, "d": d, "E": E_FACTOR, "search": res, "near_optimal": near, "selected_stress": sel, "physical_flags": physical_flags(cfgs, sel),
                   "selected": case_summary(chosen, cfgs, preps, pooled.runs[sel]["windows"]), "selected_native_trajectories": {w: run_record(pooled.runs[sel]["runs"][w]) for w in WINDOWS},
                   "selected_config": {w: build_configs(method, d, stress=sel)[w] for w in WINDOWS}, "controls": controls, "verification": verification, "n_native_evaluations": len(pooled.runs)}
        shas.setdefault("fits", {})[name] = _write_new(fit_dir / f"{name}.json", fit_doc)
        summary["fits"][name] = {"method": method, "d": d, "selected_stress": sel, "pooled_rmse": chosen["rmse"], "zero_change_rmse": chosen["zero_change_rmse"], "n": chosen["n"],
                                 "n_local_minima": res["n_local_minima"], "boundary_solution": res["boundary_solution"], "multiple_minima": res["multiple_minima"], "n_evaluations": res["n_evaluations"]}
    src_after, raw_after = source_guard(cfgs0), raw_guard(cfgs0)
    if {k: (v["package_sha256"], v["files"]) for k, v in src_before.items()} != {k: (v["package_sha256"], v["files"]) for k, v in src_after.items()} or raw_before != raw_after:
        raise CalibrationError("source or raw-input identities changed during the run")
    summary.update({"sources_after": src_after, "raw_after": raw_after, "file_sha256": shas})
    _write_new(out / "summary.json", summary)
    return summary


def plan(out: Path) -> dict:
    """Write and validate the native configs of the three fits (one stress placeholder = the 0.18 baseline) and the frozen protocol; no raw data are read and nothing runs."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise CalibrationError(f"{out} is not empty; nothing is overwritten")
    (out / "configs").mkdir(parents=True, exist_ok=True)
    receipt = S.load_receipt()
    shas = {}
    for name, (method, d) in FITS.items():
        for tag, st in (("baseline_0.18", BASELINE_STRESS), ("site_theta1500", None)):
            for w, cfg in build_configs(method, d, st, receipt).items():
                text = json.dumps(cfg, indent=1, allow_nan=False) + "\n"
                (out / "configs" / f"{name}_{tag}_{w}.json").write_text(text)
                shas[f"{name}_{tag}_{w}.json"] = hashlib.sha256(text.encode()).hexdigest()
    proto = {"protocol": PROTOCOL_ID, "fits": {k: {"method": m, "d": d, "E": E_FACTOR} for k, (m, d) in FITS.items()}, "stress_range": [SE.STRESS_MIN, SE.STRESS_MAX], "n_grid": SE.N_GRID,
             "baseline_stress": BASELINE_STRESS, "expected_sources": EXPECTED_SOURCES, "receipt_sha256": S.RECEIPT_SHA, "config_sha256": shas}
    (out / "protocol.json").write_text(json.dumps(proto, indent=1) + "\n")
    return proto


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="a NEW or empty directory")
    ap.add_argument("--execute", action="store_true", help="run the actual native column (Root only)")
    ap.add_argument("--verify-cuda", action="store_true", help="also run the explicitly requested CUDA backend (no probe, no fallback)")
    a = ap.parse_args(argv)
    if a.verify_cuda and not a.execute:
        print("REFUSED: --verify-cuda needs --execute", file=sys.stderr)
        return 2
    try:
        res = execute(Path(a.out), verify_cuda=a.verify_cuda) if a.execute else plan(Path(a.out))
    except (CalibrationError, SE.SearchError, SC.ScoreError, S.ProtocolError, dd.ConfigError, dd.ControlFailure, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(dd.jsonable(res), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
