"""Comparison of driver results between the harmonic (H) and geometric (G) interface-mixing runs, and of site-derived results with the earlier generic-soil results. Pure post-processing of result JSON; nothing is
re-simulated, refit or smoothed. Every difference is GEOMETRIC minus HARMONIC (G - H; a positive RMSE difference means G is worse) with maximum / RMS absolute values and relative values where the reference is non-zero; non-zero differences against a zero reference are COUNTED, not hidden."""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from .study import NAME_RE, PROTOCOL_ID, SERIES, ProtocolError


def _arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def diff_stats(ref: Any, other: Any) -> dict:
    """other - ref: max |d|, RMS |d|, max relative |d|/|ref| over non-zero ref, and the number of points where ref == 0 but d != 0 (relative undefined there)."""
    a, b = _arr(ref), _arr(other)
    if a.shape != b.shape:
        raise ProtocolError(f"shape mismatch {a.shape} vs {b.shape}")
    if a.size == 0:
        return {"n": 0, "max_abs": 0.0, "rms_abs": 0.0, "max_rel": None, "n_zero_reference_nonzero_diff": 0, "identical": True}
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ProtocolError("non-finite values in a compared series")
    d = b - a
    nz = a != 0.0
    rel = np.abs(d[nz]) / np.abs(a[nz])
    return {"n": int(a.size), "max_abs": float(np.max(np.abs(d))), "rms_abs": float(math.sqrt(np.mean(d**2))), "max_rel": float(rel.max()) if rel.size else None,
            "n_zero_reference_nonzero_diff": int(np.sum(~nz & (d != 0.0))), "identical": bool(np.array_equal(a, b))}


def compare_runs(run_h: dict, run_g: dict) -> dict:
    """Layer water / theta / fluxes / evaporation / drainage / storage differences for one scenario (G - H)."""
    out = {"series": {}}
    th, tg = run_h["trajectory"], run_g["trajectory"]
    for k in SERIES:
        out["series"][k] = diff_stats(th[k], tg[k])
    out["series"]["layer_theta"] = diff_stats(run_h["layer_theta"], run_g["layer_theta"])
    out["series"]["total_storage_m"] = diff_stats(_arr(th["water_m"]).sum(axis=1), _arr(tg["water_m"]).sum(axis=1))
    out["final_storage_m"] = {"harmonic": run_h["final_storage_m"], "geometric": run_g["final_storage_m"], "diff": run_g["final_storage_m"] - run_h["final_storage_m"]}
    if run_h["initial_storage_m"] != run_g["initial_storage_m"]:
        raise ProtocolError(f"initial storage differs between the paired runs ({run_h['initial_storage_m']!r} vs {run_g['initial_storage_m']!r})")
    if not np.array_equal(_arr(th["t_s"]), _arr(tg["t_s"])):
        raise ProtocolError("the paired runs do not share the same day-end times t_s")
    out["initial_storage_identical"] = True
    out["orientation"] = "geometric - harmonic"
    return out


def _delta(h: Any, g: Any) -> Any:
    return None if h is None or g is None else g - h


def compare_scores(sc_h: dict, sc_g: dict) -> dict:
    out = {"overall_rmse": {"harmonic": sc_h["overall_rmse"], "geometric": sc_g["overall_rmse"], "diff": _delta(sc_h["overall_rmse"], sc_g["overall_rmse"])},
           "pooled_rmse_alternative": {"harmonic": sc_h["pooled_rmse_alternative"], "geometric": sc_g["pooled_rmse_alternative"], "diff": _delta(sc_h["pooled_rmse_alternative"], sc_g["pooled_rmse_alternative"])}, "per_layer": {}}
    for v, ph in sc_h["per_layer"].items():
        pg = sc_g["per_layer"][v]
        if ph["n"] != pg["n"]:
            raise ProtocolError(f"layer {v}: number of scored targets differs between the paired results ({ph['n']} vs {pg['n']})")
        out["per_layer"][v] = {k: {"harmonic": ph[k], "geometric": pg[k], "diff": _delta(ph[k], pg[k])} for k in ("rmse", "bias", "mse")} | {"n": ph["n"]}
    for k, hv in sc_h.items():
        if k.startswith("skill_vs_"):
            out[k] = {"harmonic": hv, "geometric": sc_g.get(k), "diff": _delta(hv, sc_g.get(k))}
    return out


def _params(name: str) -> tuple | None:
    m = NAME_RE.fullmatch(name)
    return (float(m[1]), float(m[2])) if m else None


def _same_scenarios(rh: dict, rg: dict) -> list:
    if list(rh["runs"]) != list(rg["runs"]) or set(rh["scores"]) != set(rg["scores"]):
        raise ProtocolError("harmonic and geometric results must contain the same scenarios in the same order")
    return list(rh["runs"])


NAME_PARSE = re.compile(r"(JORN4_A|JORN4_B|MOAB2_C) (train|heldout) site-derived hydraulics, (harmonic|geometric) interface mixing")
REQUIRED_KEYS = ("config_sha256", "plan", "sources_before", "sources_after", "persistence", "runs", "scores")


def result_identity(res: dict) -> dict | None:
    """(window, role, method) read from the config name that the driver stored in the result plan; None if the name is not one of this study's."""
    m = NAME_PARSE.match(str(res.get("plan", {}).get("name", "")))
    return {"window": m[1], "role": m[2], "method": m[3]} if m else None


def _packages(res: dict, key: str) -> dict:
    src = res[key]
    if not isinstance(src, dict) or not {"maple_syrup", "maple"} <= set(src):
        raise ProtocolError(f"{key} must record both source packages (maple_syrup, maple)")
    return {n: (v["package_sha256"], v["files"]) for n, v in src.items()}


def check_window(plan: dict, window: str) -> None:
    """The result plan must be the requested window: its start, anchor rule and day count against the pinned template of that window (not just a label)."""
    from .study import load_template

    w = load_template(window)["window"]
    if plan["run_start_utc"] != w["run_start_utc"] or plan["anchor_rule"] != w["anchor"] or plan["model_days"] > (w["max_days"] or plan["model_days"]):
        raise ProtocolError(f"the result plan (start {plan['run_start_utc']}, anchor {plan['anchor_rule']}, {plan['model_days']} days) is not window {window} ({w['run_start_utc']}, {w['anchor']}, max {w['max_days']})")


def check_pair(rh: dict, rg: dict, *, window: str | None = None, role: str | None = None, strict: bool = True) -> dict:
    """Refuse a scientific H-vs-G comparison unless the two results come from the SAME prepared inputs: identical plan (weather/PET, observations, initial point/layer data and inventories, UTC day clock, probe layers,
    scored-target counts) apart from the config name; identical persistence nulls; identical source PACKAGE identities (sha256 + file counts of maple_syrup and maple, before and after, not filesystem paths). With
    `strict` (the CLI default) the plan name must also identify harmonic for `rh`, geometric for `rg`, one common window and the requested role/window; `strict=False` only relaxes that name check for synthetic fixtures."""
    for tag, res in (("harmonic", rh), ("geometric", rg)):
        missing = [k for k in REQUIRED_KEYS if k not in res]
        if missing:
            raise ProtocolError(f"{tag} result lacks {missing}")
    ph, pg = ({k: v for k, v in r["plan"].items() if k != "name"} for r in (rh, rg))
    if ph != pg:
        bad = sorted(k for k in set(ph) | set(pg) if ph.get(k) != pg.get(k))
        raise ProtocolError(f"the paired results were not prepared from identical inputs; plan differs in {bad}")
    if rh["persistence"] != rg["persistence"]:
        raise ProtocolError("the paired results have different persistence nulls (different observations/initial data)")
    pk = _packages(rh, "sources_before")
    for res, key in ((rh, "sources_after"), (rg, "sources_before"), (rg, "sources_after")):
        if _packages(res, key) != pk:
            raise ProtocolError(f"source package identities differ between the paired results or before/after ({key})")
    ident = {"window": None, "role": None}
    if strict:
        ih, ig = result_identity(rh), result_identity(rg)
        if ih is None or ig is None:
            raise ProtocolError("a result plan name is not a site-hydraulics comparison config (no window/role/method)")
        if ih["method"] != "harmonic" or ig["method"] != "geometric":
            raise ProtocolError(f"methods are mislabelled: first result is {ih['method']}, second {ig['method']} (expected harmonic, geometric)")
        if (ih["window"], ih["role"]) != (ig["window"], ig["role"]):
            raise ProtocolError(f"the paired results are different windows/roles: {ih} vs {ig}")
        if window is not None and ih["window"] != window:
            raise ProtocolError(f"requested window {window} but the results are {ih['window']}")
        if role is not None and ih["role"] != role:
            raise ProtocolError(f"requested role {role} but the results are {ih['role']}")
        if window is not None:
            check_window(rh["plan"], window)
        elif ih["window"]:
            check_window(rh["plan"], ih["window"])
        ident = {"window": ih["window"], "role": ih["role"]}
    return {"plan_identical_except_name": True, "persistence_identical": True, "source_packages": {n: {"package_sha256": v[0], "files": v[1]} for n, v in pk.items()}, **ident}


def d0_identity(rh: dict, rg: dict) -> dict:
    """Every diffusion_coefficient == 0 scenario (open or closed) must be IDENTICAL between H and G: the interface conductivity multiplies the diffusion coefficient, so it cannot act. Returns the verdict per
    scenario and raises ProtocolError if any differs. The names are read from the scenario names `d{d}_E{E}_{bottom}`."""
    out = {}
    for name in _same_scenarios(rh, rg):
        m = re.fullmatch(r"d([0-9.eE+-]+)_E([0-9.eE+-]+)_(open|closed)", name)
        if m and float(m[1]) == 0.0:
            st = compare_runs(rh["runs"][name], rg["runs"][name])
            same = all(v["identical"] for v in st["series"].values()) and st["final_storage_m"]["diff"] == 0.0
            out[name] = same
            if not same:
                raise ProtocolError(f"d0 control {name} differs between harmonic and geometric (interface conductivity must not act at d = 0)")
    return out


def compare_train(rh: dict, rg: dict, *, strict: bool = True, expected_config_sha256: dict | None = None) -> dict:
    """Paired comparison on the MATCHING (d, E) tuple over the whole training grid (not only the two separately tuned minima), the d0 identity, the two train-only selections and the score differences."""
    pairing = check_pair(rh, rg, window="JORN4_A" if strict else None, role="train" if strict else None, strict=strict)
    _check_config_shas(rh, rg, expected_config_sha256)
    names = _same_scenarios(rh, rg)
    ident = d0_identity(rh, rg)
    rows = []
    for name in names:
        h, g = rh["scores"][name]["overall_rmse"], rg["scores"][name]["overall_rmse"]
        rows.append({"scenario": name, "harmonic_overall_rmse": h, "geometric_overall_rmse": g, "diff": _delta(h, g)})
    diffs = np.array([r["diff"] for r in rows if r["diff"] is not None])
    from .study import select_best
    bh, bg = select_best(rh["scores"]), select_best(rg["scores"])
    return {"orientation": "geometric - harmonic", "pairing": pairing, "n_scenarios": len(rows), "d0_identity": ident, "rows": rows, "overall_rmse_diff_max_abs": float(np.max(np.abs(diffs))) if diffs.size else None,
            "overall_rmse_diff_rms": float(math.sqrt(np.mean(diffs**2))) if diffs.size else None, "n_undefined": len(rows) - int(diffs.size),
            "harmonic_best": bh, "geometric_best": bg, "same_selected_pair": (bh["d"], bh["E"]) == (bg["d"], bg["E"]),
            "scenario_comparison": {n: {"scores": compare_scores(rh["scores"][n], rg["scores"][n]), "runs": compare_runs(rh["runs"][n], rg["runs"][n])} for n in names}}


def _check_config_shas(rh: dict, rg: dict, expected: dict | None) -> None:
    if expected is not None and (rh["config_sha256"] != expected.get("harmonic") or rg["config_sha256"] != expected.get("geometric")):
        raise ProtocolError("a result's config_sha256 is not the frozen config hash it must be bound to")


def compare_heldout(rh: dict, rg: dict, selection: dict, *, window: str | None = None, strict: bool = True) -> dict:
    """Held-out comparison: the PRIMARY direct comparison is the COMMON H-selected pair under both methods; the G-selected pair is secondary. Every same-pair scenario is also compared. No refit."""
    if selection.get("protocol") != PROTOCOL_ID:
        raise ProtocolError("selection.json is not a protocol selection record")
    if strict and window is None:
        raise ProtocolError("a held-out comparison must name its window (JORN4_B or MOAB2_C)")
    pairing = check_pair(rh, rg, window=window, role="heldout" if strict else None, strict=strict)
    if window is not None:
        shas = selection.get("config_sha256", {})
        _check_config_shas(rh, rg, {"harmonic": shas.get(f"{window}_heldout_harmonic.json"), "geometric": shas.get(f"{window}_heldout_geometric.json")})
    names = _same_scenarios(rh, rg)
    if list(rh["runs"]) != selection.get("heldout_scenarios", list(rh["runs"])):
        raise ProtocolError("the held-out scenario list differs from the frozen selection record")
    ident = d0_identity(rh, rg)
    common, second = selection["harmonic_best"]["name"], selection["geometric_best"]["name"]
    for n in (common, second):
        if n not in names:
            raise ProtocolError(f"selected scenario {n} is absent from the held-out results")
    full = {n: {"scores": compare_scores(rh["scores"][n], rg["scores"][n]), "runs": compare_runs(rh["runs"][n], rg["runs"][n])} for n in names}
    return {"orientation": "geometric - harmonic", "pairing": pairing, "primary_pair": common, "primary": full[common], "secondary_pair_geometric_selected": second, "secondary": full[second], "d0_identity": ident, "scenarios": full}


def property_effect(new: dict, historical: dict) -> dict:
    """Matched-scenario score differences (new site-derived minus earlier generic-soil results; the historical file is the unchanged earlier CPU result, never re-run). Unmatched names are listed."""
    matched, unmatched = {}, []
    for name, sc in new["scores"].items():
        if name in historical["scores"]:
            h = historical["scores"][name]
            matched[name] = {"new_overall_rmse": sc["overall_rmse"], "historical_overall_rmse": h["overall_rmse"], "diff": _delta(h["overall_rmse"], sc["overall_rmse"]),
                             "per_layer_bias": {v: {"new": sc["per_layer"][v]["bias"], "historical": h["per_layer"][v]["bias"]} for v in sc["per_layer"]}}
        else:
            unmatched.append(name)
    return {"matched": matched, "unmatched_new_only": unmatched}
