"""Comparison tools for the original-Fortran legacy sediment reference versus the A1 Numba replay (task gpu_sediment, A2).

Nothing here was executed by its author. Policy (fixed before any result exists):

* No tolerance is fitted to data. Smooth equation-level probes (`injection_check`) use rtol 2e-6 / atol 1e-14 because the original
  keeps default-REAL globals and constants (sources.SMOOTH_RTOL/ATOL); the walk probe uses 1e-12/1e-14; SYRUP backend sediment
  comparisons keep 2e-11/1e-14 and water 2e-12/1e-14 and are NOT applied to Fortran-versus-SYRUP storms.
* Full matched storms are compared as OBSERVATIONS: totals, per-step series, spatial maps. A relative total difference above 1 % or a
  sign-pattern disagreement above 1 % of cells TRIGGERS investigation (law depth time level, REAL kind, water differences,
  order-dependent deposition erasure); it is never hidden by a loosened bound and never "fitted".
* Chastre has zero outlets: the export is vacuously zero; the comparison relies on maps, terminal-storage census and per-class ledgers.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:  # package-relative when imported as benchmarks.legacy_sediment, plain when run from the directory
    from . import sources as S
except ImportError:  # pragma: no cover
    import sources as S

FLAG_TOTAL_REL = 0.01
FLAG_SIGN_FRACTION = 0.01
A1_COLUMNS = ("pickup_kg", "deposition_active_kg", "deposition_pit_kg", "deposition_ring_kg", "deposition_inactive_kg",
              "effective_clip_source_kg", "old_mobile_kg", "new_mobile_kg", "mobile_terminal_kg", "cn_export_kg",
              "endpoint_export_kg", "outlet_flux_kg_s", "erased_deposition_kg")
PER_STEP_KG = ("pickup_kg", "deposition_active_kg", "deposition_pit_kg", "deposition_ring_kg", "deposition_inactive_kg",
               "effective_clip_source_kg", "cn_export_kg", "endpoint_export_kg")


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0.0 else abs(a - b) / scale


def map_stats(fortran: np.ndarray, syrup: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Class-summed `(ny, nx)` maps compared over `mask` (active cells): L1/L2 relative difference, max abs and its location,
    Pearson correlation, sign agreement of the field, quantiles and the overlap of the top 0.1 % / 1 % cells."""
    f, s = np.asarray(fortran, dtype=np.float64), np.asarray(syrup, dtype=np.float64)
    if f.shape != s.shape or mask.shape != f.shape:
        raise ValueError(f"shape mismatch {f.shape} {s.shape} {mask.shape}")
    x, y = f[mask], s[mask]
    diff = x - y
    out: dict[str, Any] = {"n_cells": int(mask.sum()), "fortran_sum": float(x.sum()), "syrup_sum": float(y.sum())}
    l1 = float(np.abs(x).sum() + np.abs(y).sum())
    out["l1_relative"] = float(np.abs(diff).sum() / l1 * 2.0) if l1 > 0.0 else 0.0
    l2 = math.sqrt(float((x * x).sum() * (y * y).sum()))
    out["l2_relative"] = float(math.sqrt(float((diff * diff).sum())) / math.sqrt(max(float((x * x).sum()), float((y * y).sum())))) \
        if l2 > 0.0 else 0.0
    if diff.size:
        k = int(np.argmax(np.abs(diff)))
        out["max_abs_difference"] = float(abs(diff[k]))
        where = np.flatnonzero(mask.reshape(-1))[k]
        out["max_abs_location_yx"] = [int(where // f.shape[1]), int(where % f.shape[1])]
    out["pearson"] = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 and x.std() > 0.0 and y.std() > 0.0 else None
    out["sign_disagreement_fraction"] = float(np.mean(np.sign(x) != np.sign(y))) if x.size else 0.0
    out["quantiles_fortran"] = [float(v) for v in np.quantile(x, [0.01, 0.1, 0.5, 0.9, 0.99])] if x.size else []
    out["quantiles_syrup"] = [float(v) for v in np.quantile(y, [0.01, 0.1, 0.5, 0.9, 0.99])] if y.size else []
    overlaps = {}
    for fraction in (0.001, 0.01):
        k = max(1, math.ceil(fraction * x.size))
        top_f = set(np.argpartition(-x, k - 1)[:k].tolist())
        top_s = set(np.argpartition(-y, k - 1)[:k].tolist())
        overlaps[str(fraction)] = len(top_f & top_s) / k
    out["top_cell_overlap"] = overlaps
    out["flag_investigate"] = bool(out["sign_disagreement_fraction"] > FLAG_SIGN_FRACTION
                                   or _rel(out["fortran_sum"], out["syrup_sum"]) > FLAG_TOTAL_REL)
    return out


def compare_runs(fortran: dict, a1_dir, *, nr2: int, nc2: int) -> dict[str, Any]:
    """A qualified Fortran `run_once` record against an A1 output directory (`legacy_ledger.npz`, `legacy_summary.json`)."""
    a1_dir = Path(a1_dir)
    npz = np.load(a1_dir / "legacy_ledger.npz")
    ledger_a1 = npz["ledger"]  # (steps, 13, nc)
    ledger_f = fortran["ledger"]
    steps = min(ledger_a1.shape[0], ledger_f.shape[0])
    if ledger_a1.shape[0] != ledger_f.shape[0]:
        raise ValueError("the two runs have different numbers of steps")
    report: dict[str, Any] = {"steps": int(steps), "totals_by_class": {}, "flags": []}
    for name in PER_STEP_KG:
        i = A1_COLUMNS.index(name)
        f, a = ledger_f[:, i].sum(axis=0), ledger_a1[:, i].sum(axis=0)
        report["totals_by_class"][name] = {"fortran": f.tolist(), "syrup": a.tolist(),
                                           "relative": [_rel(float(x), float(y)) for x, y in zip(f, a, strict=True)],
                                           "total_relative": _rel(float(f.sum()), float(a.sum()))}
        if report["totals_by_class"][name]["total_relative"] > FLAG_TOTAL_REL and max(abs(f.sum()), abs(a.sum())) > 0.0:
            report["flags"].append(f"{name}: total differs by more than 1 %")
    mob = {}
    for name in ("old_mobile_kg", "new_mobile_kg", "mobile_terminal_kg"):
        i = A1_COLUMNS.index(name)
        f, a = ledger_f[:, i].sum(axis=1), ledger_a1[:, i].sum(axis=1)
        peak_f, peak_a = int(np.argmax(f)), int(np.argmax(a))
        mob[name] = {"fortran_final": float(f[-1]), "syrup_final": float(a[-1]), "fortran_peak": float(f.max()),
                     "syrup_peak": float(a.max()), "peak_step_fortran": peak_f + 1, "peak_step_syrup": peak_a + 1,
                     "max_relative_step_difference": float(max((_rel(float(x), float(y)) for x, y in zip(f, a, strict=True)),
                                                              default=0.0))}
    report["mobile_series"] = mob
    i = A1_COLUMNS.index("pickup_kg")
    fp, ap = ledger_f[:, i].sum(axis=1), ledger_a1[:, i].sum(axis=1)
    report["first_positive_pickup_step"] = {"fortran": int(np.argmax(fp > 0)) + 1 if np.any(fp > 0) else None,
                                            "syrup": int(np.argmax(ap > 0)) + 1 if np.any(ap > 0) else None}
    active = npz["active"]
    maps = {"cumulative_detachment_kg": ("CUMDET  ", "cumulative_detachment_kg"),
            "cumulative_deposition_kg": ("CUMDEP  ", "cumulative_deposition_kg"),
            "cumulative_clipping_source_kg": ("CUMCLIP ", "cumulative_clipping_source_kg"),
            "final_mobile_kg": ("MOBILE  ", "final_mobile_kg")}
    report["maps"] = {}
    for label, (ftag, key) in maps.items():
        fm = S.fortran_grid_to_syrup(fortran["maps"][ftag], nr2, nc2)
        report["maps"][label] = map_stats(fm.sum(axis=-1), np.asarray(npz[key]).sum(axis=-1), active)
        if report["maps"][label]["flag_investigate"]:
            report["flags"].append(f"map {label}: >1 % total difference or sign disagreement")
    net_f = S.fortran_grid_to_syrup(fortran["maps"]["CUMDET  "], nr2, nc2).sum(-1) - \
        S.fortran_grid_to_syrup(fortran["maps"]["CUMDEP  "], nr2, nc2).sum(-1)
    net_a = np.asarray(npz["cumulative_detachment_kg"]).sum(-1) - np.asarray(npz["cumulative_deposition_kg"]).sum(-1)
    report["maps"]["net_erosion_kg"] = map_stats(net_f, net_a, active)
    if report["maps"]["net_erosion_kg"]["flag_investigate"]:
        report["flags"].append("map net_erosion_kg: >1 % total difference or sign disagreement")
    # final hydraulic/soil fields (original mm, mm/s, mm2/s -> SI) and the water volume integrals
    water_maps = {"final_depth_m": ("DEPTH_MM", "final_depth_m", 1.0e-3), "final_soil_water_m": ("SOILW_MM", "final_soil_water_m", 1.0e-3),
                  "final_discharge_m2_s": ("DISCH_MM", "final_discharge_m2_s", 1.0e-6),
                  "final_velocity_m_s": ("VELOC_MM", "final_velocity_m_s", 1.0e-3)}
    report["water_maps"] = {}
    for label, (ftag, key, factor) in water_maps.items():
        fm = S.fortran_grid_to_syrup(fortran["maps"][ftag], nr2, nc2) * factor
        report["water_maps"][label] = map_stats(fm, np.asarray(npz[key], dtype=np.float64), active)
        if report["water_maps"][label]["flag_investigate"]:
            report["flags"].append(f"water map {label}: >1 % total difference or sign disagreement")
    term = npz["terminal_storage"]
    report["terminal_storage"] = {
        "n_cells": int(term.sum()),
        "fortran_final_mobile_kg": float(S.fortran_grid_to_syrup(fortran["maps"]["MOBILE  "], nr2, nc2).sum(-1)[term].sum()),
        "syrup_final_mobile_kg": float(np.asarray(npz["final_mobile_kg"]).sum(-1)[term].sum()),
        "fortran_deposition_kg": float(S.fortran_grid_to_syrup(fortran["maps"]["CUMDEP  "], nr2, nc2).sum(-1)[term].sum()),
        "syrup_deposition_kg": float(np.asarray(npz["cumulative_deposition_kg"]).sum(-1)[term].sum())}
    summary = json.loads((a1_dir / "legacy_summary.json").read_text())
    res = fortran["result"]
    budget = summary["water"]["whole_storm_budget"]
    fortran_water = {k: S._f(res, k) for k in ("RAIN_M3", "EXPORT_M3", "SURFACE_M3", "SOIL_M3", "DRAIN_M3")}
    pairs = {"rain": ("RAIN_M3", "rain_m3"), "export": ("EXPORT_M3", "export_m3"), "surface_final": ("SURFACE_M3", "surface_final_m3"),
             "soil_final": ("SOIL_M3", "soil_final_m3"), "drainage": ("DRAIN_M3", "drainage_m3")}
    report["water_integrals_relative"] = {label: _rel(fortran_water[fk], float(budget[bk])) for label, (fk, bk) in pairs.items()}
    report["water"] = {
        "fortran": fortran_water, "density_g_cm3_original": S._f(res, "DENSITY_G_CM3"),
        "syrup_whole_storm_budget": budget,
        "note": "the original routines keep their stale-inflow/bracket behaviour; their budget is reported, not claimed closed"}
    report["timing_s"] = {"fortran_loop": fortran["loop_s"], "fortran_diagnostics": fortran["diag_s"],
                          "fortran_capture": fortran["capture_s"], "fortran_kernel": fortran["kernel_s"],
                          "fortran_process_wall": fortran["process_wall_s"], "fortran_peak_rss_kib": fortran["max_rss_kib"],
                          "syrup": summary["performance"]}
    report["policy"] = ("observations; >1 % totals or >1 % sign disagreement trigger investigation, not acceptance; "
                        "no tolerance was fitted")
    return report


# --- state injection ---------------------------------------------------------------------------------------------------
def build_engine(graph, params, grid, vegetation, holdings, *, compiled: bool = True):
    """An A1 `StepEngine` (read-only use of the accepted modules) for injection checks."""
    from maple_syrup import legacy_native as N
    from maple_syrup.legacy_native_numba import StepEngine, WetLawRunner
    from maple_syrup.legacy_physics_numba import prepare_legacy_physics

    net = N.native_network(graph)
    ctx = prepare_legacy_physics(params, grid, vegetation, holdings)
    return StepEngine(net, WetLawRunner(ctx), dt=1.0, compiled=compiled), net


def injection_check(cap: dict, engine, *, nr2: int, nc2: int, af: float, dx_mm: float, density_g_cm3: float,
                    depth: str = "post_infiltration") -> dict[str, Any]:
    """Run ONE A1 step on the state the ORIGINAL routines saw (a `capture_NNNNNN.bin`) and compare equation outputs.

    The Fortran pre-step state is converted to SI and SYRUP layout: depth mm->m (`depth` selects the post-infiltration `d(1)` the
    original uses, or `previous`, the SYRUP default convention, to QUANTIFY that departure), velocity and rain mm/s->m/s, mobile mm
    -> kg (`af` kg per mm), unit flux mm2/s -> kg/s (`dx * density * 1e-6`), and the sediment velocity memory: the original holds
    v_soil already multiplied by 0.9 at the end of the previous step, SYRUP holds the undecayed value, so `v_prev = v_soil / 0.9`.
    Smooth quantities use rtol 2e-6 / atol 1e-14 (REAL kind-4 originals); branch masks must match exactly."""
    engine.reset()

    def g(tag):  # the binary tags are EXACTLY 8 characters (space padded)
        return S.fortran_grid_to_syrup(cap[tag.ljust(8)], nr2, nc2)

    n, nc = engine.n, engine.nc
    ny, nx = engine.net.shape
    active = engine.net.active.reshape(ny, nx)
    qf = dx_mm * density_g_cm3 * 1.0e-6
    mask3 = active[..., None]

    def flat(a):
        return np.ascontiguousarray(np.where(mask3, a, 0.0)).reshape(n, nc)

    engine.M1[:] = flat(g("PRE_DS1") * af)
    engine.Q1[:] = flat(g("PRE_QS1") * qf)
    engine.Qin1[:] = flat(g("PRE_QIN1") * qf)
    engine.v_prev[:] = flat(g("PRE_VSED") * 1.0e-3 / 0.9)
    depth_key = {"post_infiltration": "PRE_D1  ", "previous": "PRE_DOLD"}[depth]
    d_m = np.ascontiguousarray(S.fortran_grid_to_syrup(cap[depth_key], nr2, nc2) * 1.0e-3)
    v_m = np.ascontiguousarray(np.where(active, S.fortran_grid_to_syrup(cap["PRE_V   "], nr2, nc2) * 1.0e-3, 0.0))
    r_m = np.ascontiguousarray(S.fortran_grid_to_syrup(cap["PRE_R2  "], nr2, nc2) * 1.0e-3)
    engine.step(d_m, v_m, r_m)
    sh = (ny, nx, nc)
    trial, factor = g("POST_TRL"), g("POST_FAC")
    clip_f = np.where(trial < 0.0, -trial * factor * af, 0.0)
    pairs = {
        "detachment_rate_kg_s": (g("POST_DET") * af, engine.det.reshape(sh)),
        "deposition_kg": (g("POST_DEP") * af, engine.cum_dep.reshape(sh)),
        "mobile_after_kg": (g("POST_DS2") * af, engine.mobile.reshape(sh)),
        "flux_after_kg_s": (g("POST_QS2") * qf, engine.Q1.reshape(sh)),
        "clipping_source_kg": (clip_f, engine.cum_clip.reshape(sh)),
        "sediment_velocity_m_s": (g("POST_VS ") * 1.0e-3, engine.v_prev.reshape(sh)),
    }
    out: dict[str, Any] = {"depth_convention": depth, "rtol": S.SMOOTH_RTOL, "atol": S.SMOOTH_ATOL, "fields": {},
                           "density_g_cm3_used": density_g_cm3,
                           "density_note": "the widened kind-4 value the original routines use (result.txt DENSITY_G_CM3)"}
    for name, (f, s) in pairs.items():
        f = np.where(mask3, f, 0.0)
        s = np.where(mask3, s, 0.0)
        close = np.isclose(f, s, rtol=S.SMOOTH_RTOL, atol=S.SMOOTH_ATOL)
        diff = np.abs(f - s)
        scale = np.maximum(np.abs(f), np.abs(s))
        out["fields"][name] = {"n_bad": int((~close).sum()), "n": int(close.size), "max_abs": float(diff.max()),
                               "max_rel": float(np.max(np.where(scale > 0, diff / np.where(scale > 0, scale, 1.0), 0.0)))}
    det_f = np.where(mask3, g("POST_DET") > 0.0, False)
    det_s = np.where(mask3, engine.det.reshape(sh) > 0.0, False)
    out["detachment_positive_mask_mismatches"] = int((det_f != det_s).sum())
    out["all_close"] = all(v["n_bad"] == 0 for v in out["fields"].values()) and out["detachment_positive_mask_mismatches"] == 0
    return out


def plot1_golden(a1_npz, fortran_ledger_dat) -> dict[str, Any]:
    """A1 Plot 1 `legacy_ledger.npz` versus the existing real-application ledger `syrup_sediment_ledger.dat` (phase7b audit hook:
    columns iter t class pickup dep_active dep_outside clip old new cn endpoint cellflux resid internal)."""
    rows = [ln.split() for ln in Path(fortran_ledger_dat).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    data = np.array([[float(t) for t in r] for r in rows])
    if data.shape[1] != 14:
        raise ValueError("unexpected ledger width")
    steps = int(data[:, 0].max())
    data = data.reshape(steps, N_CLASSES_PLOT, 14)
    npz = np.load(a1_npz)
    led = npz["ledger"]
    m = min(steps, led.shape[0])
    names = {"pickup_kg": 3, "deposition_active_kg": 4, "effective_clip_source_kg": 6, "old_mobile_kg": 7, "new_mobile_kg": 8,
             "cn_export_kg": 9}
    out = {"steps_compared": m, "columns": {}}
    for name, col in names.items():
        f = data[:m, :, col]
        a = led[:m, A1_COLUMNS.index(name), :]
        out["columns"][name] = {"fortran_total": float(f.sum()), "syrup_total": float(a.sum()),
                                "total_relative": _rel(float(f.sum()), float(a.sum()))}
    return out


N_CLASSES_PLOT = 6
