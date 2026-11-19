"""JOINT calibration of ONE shared diffusion coefficient d and ONE shared evaporation-stress threshold against the JORN4_A / JORN4_B / MOAB2_C observed dry-downs, E fixed at 1 (stage-1 analysis helper; every window
is used for fitting, so a fit is a MECHANISM DIAGNOSTIC, not an independent field validation). It reuses `benchmarks.subsurface_stress_calibration` (configs, change scoring, source/raw guards, backend verification) and
`benchmarks.subsurface_validation.drydown_driver.run_column` unchanged; no water physics lives here.

Every objective evaluation runs the ACTUAL native column (`dd.run_column`, compiled Numba backend) on all three prepared windows with E = 1, T = 0, open bottom, gravity on, the given d (it reaches the scenario AND the
config) and the given stress (it reaches the soil-layer threshold). Inputs are prepared ONCE per window; only light per-window SSE/count are cached (never the full native histories). Scores compare model CHANGES with
observed CHANGES (constant per-(window, sensor) offset, t0 excluded, equal weight per valid observation); the offset never touches the physical water. E and the stress are not separately identifiable in the dry
quadratic regime and fixing E = 1 does not measure theta1500 or alter retention/Ks: the result is a conditional empirical calibration, not field validation and not a new default.

    python -m benchmarks.subsurface_joint_stress.joint --out NEWDIR                              # plan: write + validate the native configs and protocol (no raw data read, no model run)
    python -m benchmarks.subsurface_joint_stress.joint --out NEWDIR --execute [--verify-cuda]    # Root only: prepare, fit both interface methods, controls, verification (CUDA only with the flag, never probed)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks.subsurface_site_comparison import study as S
from benchmarks.subsurface_stress_calibration import calibrate as C
from benchmarks.subsurface_stress_calibration import scoring as SC
from benchmarks.subsurface_validation import drydown_driver as dd

from . import search2d as S2

PROTOCOL_ID = "syrup.joint_d_stress_calibration/1"
WINDOWS = C.WINDOWS
METHODS = ("harmonic", "geometric")
E_FACTOR = C.E_FACTOR
BASELINE_STRESS, BASELINE_D = C.BASELINE_STRESS, 0.05
N_D_LOG, N_STRESS = 41, 81
EXPECTED_N_D, EXPECTED_N_STRESS = 43, 85                  # frozen: 0 + 41 log-spaced 1e-6..1 + exact 0.05; 81 linear + 0.18 + the three previous fitted stresses
OLD_STRESS = {"main": 0.21717029762566153, "secondary": 0.2171797728919738, "sensitivity_d0": 0.20644317480382438}
OLD_FIT_OF = {"harmonic": ("main", 0.05), "geometric": ("secondary", 0.05)}
OLD_D0_FIT = "sensitivity_d0"
OLD_DATA_REL = Path("docs/subsurface_hydrology/evaporation_stress_calibration")
OLD_MANIFEST_SHA256 = "4af7b08fc2f1234a2130dabdd82e00b9acc9e28e3ed7fbd4c85388dea3504e23"      # the immutable e11 artifact manifest itself; pinned before any listed hash is trusted
ROUNDOFF_REL = 1e-9
LABEL = ("empirical diffusion coefficient d and evaporation-stress threshold (volumetric fraction) JOINTLY calibrated to pooled observed changes with E = 1 fixed; NOT the measured theta1500 / wilting point, "
         "retention or Ks; all three windows are used in the fit (mechanism diagnostic, no held-out claim, not a new default)")


class JointError(RuntimeError):
    pass


def frozen_axes() -> tuple:
    d = S2.d_grid(N_D_LOG, extra=(BASELINE_D,))
    s = S2.stress_grid(N_STRESS, extra=(BASELINE_STRESS, *OLD_STRESS.values()))
    if d.size != EXPECTED_N_D or s.size != EXPECTED_N_STRESS:
        raise JointError(f"frozen grid counts changed: {d.size} d, {s.size} stress")
    return d, s


def incumbents(method: str) -> list:
    """Previous published fixed-d optima that must be evaluated exactly and used as refinement starts: the d = 0 fit and this method's own d = 0.05 fit."""
    name, d = OLD_FIT_OF[method]
    return [(0.0, OLD_STRESS[OLD_D0_FIT]), (d, OLD_STRESS[name])]


# ----------------------------------------------------------------------------------------------------- previous data
def load_previous(root: Path | None = None, manifest_sha256: str = OLD_MANIFEST_SHA256) -> dict:
    """Previous published fixed-d fits. The old artifact manifest ITSELF must match its pinned sha256 (a changed manifest is refused even if internally self-consistent); only then is every `actual_cpu/` file checked
    against the hash it lists. Refuses a missing/changed file."""
    base = (Path(__file__).resolve().parents[2] if root is None else Path(root)) / OLD_DATA_REL
    man_bytes = (base / "artifact_manifest.json").read_bytes()
    if hashlib.sha256(man_bytes).hexdigest() != manifest_sha256:
        raise JointError("the previous artifact manifest differs from its pinned sha256")
    manifest = json.loads(man_bytes)["data"]
    files = {k: v for k, v in manifest.items() if k.startswith("actual_cpu/")}
    need = {"actual_cpu/summary.json", *(f"actual_cpu/fits/{n}.json" for n in OLD_STRESS)}
    if not need <= set(files):
        raise JointError("the old manifest does not list the previous fits")
    for rel, sha in files.items():
        p = base / "data" / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise JointError(f"previous result {rel} is missing or differs from the manifest hash")
    fits = {n: json.loads((base / "data" / f"actual_cpu/fits/{n}.json").read_text()) for n in OLD_STRESS}
    for n, st in OLD_STRESS.items():
        if fits[n]["selected_stress"] != st:
            raise JointError(f"previous fit {n} stress differs from the frozen incumbent")
    return {"n_files_verified": len(files), "fits": fits, "summary": json.loads((base / "data" / "actual_cpu/summary.json").read_text())}


# ---------------------------------------------------------------------------------------------------------- configs
def build_configs(method: str, d: float, stress: float, receipt: dict | None = None) -> dict:
    """Validated raw configs per window for ONE joint (d, stress): the old builder's frozen windows/layers/interfaces/E = 1 scenario, with the provenance rewritten so that it says d AND stress are jointly fitted
    (the earlier fixed-d / site-theta1500 statements are retained only under `historical_*` keys)."""
    d, stress = S2.check_point(d, stress)
    out = C.build_configs(method, d, stress=stress, receipt=receipt)
    for w, cfg in out.items():
        prov = cfg["provenance"]
        prov["historical_fixed_d_stress_calibration"] = prov.pop("stress_calibration")
        prov["historical_evaporation_stress_fixed_d_fit"] = prov["evaporation_stress"]
        prov["evaporation_stress"] = "one shared constant JOINTLY calibrated with d to the pooled observations of all three windows (empirical threshold, not theta1500)"
        prov["diffusion"] = "one shared diffusion coefficient d JOINTLY calibrated with the stress to the pooled observations of all three windows (empirical value, not a measured or physical diffusivity)"
        prov["joint_calibration"] = {"protocol": PROTOCOL_ID, "label": LABEL, "fitted": ["diffusion_coefficient", "theta_evaporation_stress"], "fixed": {"E": E_FACTOR, "T": 0}, "d": d, "stress": stress,
                                     "windows_used": list(WINDOWS), "held_out_windows": [], "method": method}
        cfg["name"] = f"{w} joint d/stress calibration, {method}, d={d:g}, stress={stress:g}, E={E_FACTOR:g}"
        dd.validate_config(cfg)
    return out


def with_params(cfg: dict, d: float, stress: float) -> dict:
    """A config with a different (d, stress); nothing else changes. d reaches the scenario (the actual run input) and the stress the soil-layer threshold. Refuses out-of-range/non-finite values."""
    d, stress = S2.check_point(d, stress)
    scen = {**cfg["scenarios"][0], "name": f"joint_d{d:.17g}_stress{stress:.17g}_E{E_FACTOR:g}_open", "diffusion_coefficient": d}
    return {**cfg, "layers": {**cfg["layers"], "theta_evaporation_stress": stress}, "scenarios": [scen]}


# ------------------------------------------------------------------------------------------------------- objective
class JointPooled:
    """Pooled objective over the prepared windows. Each NEW (d, stress) runs the actual column on every window, scores it, and keeps ONLY the light per-window (n, SSE, zero-change SSE); the native history is
    dropped at once. `runner` / `scorer` are injectable for engineering tests only."""

    def __init__(self, cfgs: dict, preps: dict, backend: str = "numba", runner: Callable | None = None, scorer: Callable | None = None):
        self.cfgs, self.preps, self.backend = cfgs, preps, backend
        self.runner, self.scorer = runner or dd.run_column, scorer or C.window_score_from_run
        self.light: dict = {}
        self.n_objective_runs = 0                                                     # columns run to evaluate NEW cached pairs (= n windows x pairs)
        self.n_full_runs = 0                                                          # columns re-run with complete histories (selected points / controls), separately labelled

    def _run(self, d: float, s: float, *, full: bool = False) -> tuple:
        runs, wins = {}, {}
        for w, c in self.cfgs.items():
            cc = with_params(c, d, s)
            runs[w] = self.runner(cc, self.preps[w], cc["scenarios"][0], self.backend)
            wins[w] = self.scorer(cc, self.preps[w], runs[w])
            if full:
                self.n_full_runs += 1
            else:
                self.n_objective_runs += 1
        return runs, wins

    def evaluate(self, d: float, s: float) -> dict:
        key = S2.check_point(d, s)                                                    # refused BEFORE any column runs
        if key not in self.light:
            _, wins = self._run(*key)
            self.light[key] = {w: {"n": int(v["n"]), "sse": float(v["sse"]), "zero_change_sse": float(v["zero_change_sse"])} for w, v in wins.items()}
        return {**SC.pooled_objective(self.light[key]), "d": key[0], "stress": key[1]}

    def objective(self, d: float, s: float) -> float:
        return self.evaluate(d, s)["mse"]

    def full(self, d: float, s: float) -> tuple:
        """A fresh actual run with complete native histories (selected points / controls only): (runs, full window scores, pooled)."""
        key = S2.check_point(d, s)
        runs, wins = self._run(*key, full=True)
        return runs, wins, SC.pooled_objective(wins)

    def evaluations_table(self) -> list:
        return [[d, s] + [self.light[(d, s)][w][k] for w in self.cfgs for k in ("n", "sse")] for (d, s) in self.light]


# ----------------------------------------------------------------------------------------------------------- fits
def fit_method(method: str, pooled: JointPooled, d_axis: Any, s_axis: Any) -> dict:
    return S2.search_joint(pooled.objective, d_axis, s_axis, incumbents(method))


def case_record(method: str, pooled: JointPooled, d: float, s: float) -> dict:
    """A fresh actual run at (d, s) with the full report record; its objective must equal the cached one exactly (a deterministic engine)."""
    runs, wins, pool = pooled.full(d, s)
    cached = pooled.evaluate(d, s)
    if pool["sse"] != cached["sse"]:
        raise JointError(f"{method}: the re-run at d={d!r}, stress={s!r} differs from the cached evaluation")
    cfgs = {w: with_params(c, d, s) for w, c in pooled.cfgs.items()}
    return {"d": d, "stress": s, "summary": C.case_summary(pool, cfgs, pooled.preps, wins), "native_trajectories": {w: C.run_record(runs[w]) for w in runs}, "physical_flags": physical_flags(cfgs, s)}


def physical_flags(cfgs: dict, stress: float) -> dict:
    """Where the selected threshold sits against each window's top-layer FC and saturation (a position statement only). The site theta1500 is the HISTORICAL no-refit value, read from the explicitly
    historical provenance (the current provenance does not describe a fixed-d or site-theta1500 fit)."""
    out = {}
    for w, c in cfgs.items():
        fc, sat = float(c["layers"]["theta_field_capacity"][0]), float(c["layers"]["theta_saturation"][0])
        out[w] = {"top_layer_fc": fc, "top_layer_saturation": sat, "above_fc": stress > fc, "above_saturation": stress > sat,
                  "historical_site_theta1500": c["provenance"]["historical_fixed_d_stress_calibration"]["site_theta1500"]}
    return out


def compare_previous(method: str, sel: dict, pooled: JointPooled, previous: dict) -> dict:
    """Selected joint fit against the PREVIOUS published fixed-d fits (re-evaluated, never refitted) and the zero-change baseline. The new minimum must be no worse than every previous incumbent within roundoff
    (those exact pairs are grid members)."""
    out = {"selected": {"d": sel["d"], "stress": sel["stress"], "mse": sel["objective"], "rmse": math.sqrt(sel["objective"])}, "previous": {}}
    name, d_old = OLD_FIT_OF[method]
    for label, nm, d in (("previous_fixed_d_fit", name, d_old), ("previous_d0_fit", OLD_D0_FIT, 0.0)):
        old = previous["fits"][nm]["selected"]["objective"]
        now = pooled.evaluate(d, OLD_STRESS[nm])
        if abs(now["mse"] - old["mse"]) > ROUNDOFF_REL * old["mse"]:
            raise JointError(f"{method}: the previous {label} ({d}, {OLD_STRESS[nm]}) no longer reproduces ({now['mse']!r} vs {old['mse']!r})")
        if sel["objective"] > old["mse"] * (1.0 + ROUNDOFF_REL):
            raise JointError(f"{method}: the joint minimum {sel['objective']!r} is worse than the previous {label} {old['mse']!r}")
        out["previous"][label] = {"d": d, "stress": OLD_STRESS[nm], "old_mse": old["mse"], "old_rmse": old["rmse"], "re_evaluated_mse": now["mse"], "n": now["n"], "zero_change_rmse": now["zero_change_rmse"],
                                  "improvement_of_selected_mse": old["mse"] - sel["objective"]}
    b = pooled.evaluate(BASELINE_D, BASELINE_STRESS)
    out["baseline_d0.05_stress0.18"] = {"mse": b["mse"], "rmse": b["rmse"]}
    sel_eval = pooled.evaluate(sel["d"], sel["stress"])
    out["zero_change_baseline"] = {"rmse": sel_eval["zero_change_rmse"], "mse": sel_eval["zero_change_mse"], "skill_of_selected": sel_eval["skill_vs_zero_change"]}
    return out


def source_identities(rec: dict) -> dict:
    return {k: (v["package_sha256"], v["files"]) for k, v in rec.items()}


def _write_new(path: Path, doc: Any) -> str:
    text = json.dumps(dd.jsonable(doc), indent=1, allow_nan=False) + "\n"
    if path.exists():
        raise JointError(f"{path} exists; nothing is overwritten")
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def plan(out: Path, previous_root: Path | None = None) -> dict:
    """Write and validate the native configs (one per method: the previous d = 0.05 / baseline-stress point as the placeholder) and the frozen protocol; verify the old data hashes; no raw data, no model run."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise JointError(f"{out} is not empty; nothing is overwritten")
    prev = load_previous(previous_root)
    d_axis, s_axis = frozen_axes()
    (out / "configs").mkdir(parents=True, exist_ok=True)
    receipt = S.load_receipt()
    shas = {}
    for m in METHODS:
        for w, cfg in build_configs(m, BASELINE_D, BASELINE_STRESS, receipt).items():
            shas[f"{m}_{w}.json"] = _write_new(out / "configs" / f"{m}_{w}.json", cfg)
    proto = {"protocol": PROTOCOL_ID, "label": LABEL, "methods": list(METHODS), "E": E_FACTOR, "T": 0, "windows": list(WINDOWS), "d_bounds": [S2.D_MIN, S2.D_MAX], "stress_bounds": [S2.STRESS_MIN, S2.STRESS_MAX],
             "d_scale_for_z": S2.D_SCALE, "d_axis": d_axis.tolist(), "stress_axis": s_axis.tolist(), "n_d": int(d_axis.size), "n_stress": int(s_axis.size), "n_grid_pairs": int(d_axis.size * s_axis.size),
             "old_incumbents": {m: incumbents(m) for m in METHODS}, "n_separated_refinements": S2.N_REFINE, "separation": S2.SEPARATION, "roundoff_rel": ROUNDOFF_REL,
             "previous_files_verified": prev["n_files_verified"], "expected_sources": C.EXPECTED_SOURCES, "receipt_sha256": S.RECEIPT_SHA, "config_sha256": shas}
    _write_new(out / "protocol.json", proto)
    return proto


def execute(out: Path, *, verify_cuda: bool = False, previous_root: Path | None = None) -> dict:
    """Root's run: guards, prepare once, joint fits (both interface methods), selected reruns with full trajectories, comparison with the previous fits, verification, export, guards again."""
    out = Path(out)
    proto = plan(out, previous_root)
    previous = load_previous(previous_root)
    d_axis, s_axis = frozen_axes()
    prep_dir, fit_dir, ev_dir = out / "prepared_inputs", out / "fits", out / "evaluations"
    for p in (prep_dir, fit_dir, ev_dir):
        p.mkdir()
    cfgs0 = {m: {w: dd.validate_config(c) for w, c in build_configs(m, BASELINE_D, BASELINE_STRESS).items()} for m in METHODS}
    src_before, raw_before = C.source_guard(cfgs0["harmonic"]), C.raw_guard(cfgs0["harmonic"])
    preps = {w: dd.prepare_inputs(cfgs0["harmonic"][w]) for w in WINDOWS}                  # once per window, shared by both methods (identical inputs)
    shas = {"prepared_inputs": {w: _write_new(prep_dir / f"{w}.json", dd.prepared_inputs_export(cfgs0["harmonic"][w], preps[w])) for w in WINDOWS}, "fits": {}, "evaluations": {}}
    summary = {"protocol": PROTOCOL_ID, "label": LABEL, "d_bounds": proto["d_bounds"], "stress_bounds": proto["stress_bounds"], "fits": {}, "sources_before": src_before, "raw_before": raw_before,
               "previous_files_verified": previous["n_files_verified"]}
    pools = {}
    for m in METHODS:
        pooled = pools[m] = JointPooled(cfgs0[m], preps)
        res = fit_method(m, pooled, d_axis, s_axis)
        sel = res["selected"]
        chosen = case_record(m, pooled, sel["d"], sel["stress"])
        comp = compare_previous(m, sel, pooled, previous)
        controls = {"baseline_d0.05_stress0.18": case_record(m, pooled, BASELINE_D, BASELINE_STRESS),
                    **{f"previous_{nm}": case_record(m, pooled, 0.0 if nm == OLD_D0_FIT else OLD_FIT_OF[m][1], OLD_STRESS[nm]) for nm in ((OLD_FIT_OF[m][0], OLD_D0_FIT))}}
        sel_cfgs = {w: dd.validate_config(c) for w, c in build_configs(m, sel["d"], sel["stress"]).items()}
        verification = C.verify_case(sel_cfgs, preps, None, verify_cuda=verify_cuda)
        if sel["d"] == 0.0:                                                              # at d = 0 the interface conductivity is unused: geometric must equal harmonic exactly
            other = "geometric" if m == "harmonic" else "harmonic"
            g = {w: dd.validate_config(c) for w, c in build_configs(other, 0.0, sel["stress"]).items()}
            for w in WINDOWS:
                a, b = (dd.run_column(sel_cfgs[w], preps[w], sel_cfgs[w]["scenarios"][0], "numba"), dd.run_column(g[w], preps[w], g[w]["scenarios"][0], "numba"))
                dd.assert_runs_identical(a, b, f"{w} d=0 harmonic vs geometric")
            verification["d0_harmonic_geometric_identical"] = True
        fit_doc = {"fit": m, "method": m, "E": E_FACTOR, "search": res, "selected": chosen, "comparison": comp, "controls": controls, "verification": verification, "selected_config": build_configs(m, sel["d"], sel["stress"]),
                   "n_pairs_evaluated": len(pooled.light), "n_objective_column_runs": pooled.n_objective_runs, "n_full_rerun_column_runs": pooled.n_full_runs,
                   "n_total_column_runs": pooled.n_objective_runs + pooled.n_full_runs, "n_windows": len(WINDOWS)}
        shas["fits"][m] = _write_new(fit_dir / f"{m}.json", fit_doc)
        shas["evaluations"][m] = _write_new(ev_dir / f"{m}.json", {"columns": ["d", "stress"] + [f"{w}_{k}" for w in WINDOWS for k in ("n", "sse")], "rows": pooled.evaluations_table()})
        obj = chosen["summary"]["objective"]
        summary["fits"][m] = {"d": sel["d"], "stress": sel["stress"], "pooled_mse": obj["mse"], "pooled_rmse": obj["rmse"], "zero_change_rmse": obj["zero_change_rmse"], "n": obj["n"],
                              "boundary_solution": res["boundary_solution"], "requires_root_bounded_extension_decision": res["requires_root_bounded_extension_decision"], "n_pairs_evaluated": res["n_evaluations"],
                              "n_local_refinements": len(res["local_minima"])}
    src_after, raw_after = C.source_guard(cfgs0["harmonic"]), C.raw_guard(cfgs0["harmonic"])
    if source_identities(src_before) != source_identities(src_after) or raw_before != raw_after:
        raise JointError("source or raw-input identities changed during the run")
    summary.update({"sources_after": src_after, "raw_after": raw_after, "file_sha256": shas})
    _write_new(out / "summary.json", summary)
    return summary


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
    except (JointError, S2.JointSearchError, C.CalibrationError, SC.ScoreError, S.ProtocolError, dd.ConfigError, dd.ControlFailure, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(dd.jsonable(res), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
