"""JORN 160-day native field-COLUMN driver (stage 1): one 1 m2 three-layer soil column, observed 30-minute rain, Hargreaves PET, matrix-only Green-Ampt intake, zero-residence open surface.

    python -m benchmarks.subsurface_wet_validation.driver --config CONFIG.json [--out DIR] [--execute] [--stop-after-day N --export-state P | --resume-state P]

Without `--execute` only the plan is produced. The driver COMPOSES the actual native APIs (`rainfall.RainfallSchedule`, `matrix_infiltration.matrix_step`, `subsurface.step`, `weather_pet`); it does not reimplement infiltration or column physics.
Experiment profile `syrup.jorn160_wet.daily_weather_with_storm_substeps/1` (a user-approved DAILY-WEATHER APPROXIMATION, not an exact chronological hourly field clock):

  per nominal UTC-7 day k:   [wet day] native storm sub-steps over the day's rain schedule (first wet 30-min row .. last wet row end, zero gaps kept, max_dt_s, split at schedule edges):
                                matrix_step (pool = rain of the sub-step, standing = 0) -> intake credited ONCE by the matrix stage; remaining pool exported as runoff;
                                ss.step(dt, added = 0, E = 0): gravity/diffusion, clock advanced once; soil surface_return exported
                              -> ONE native 86400 s interevent step with the day's PET (E factor from the config, T = 0)
  Storm durations are ADDITIONAL physical time; the nominal date index and the physical clock are kept separately. No E/T/growth during rain; no extra interevent sub-steps.
Matrix history J starts at 0 for the run and is never reset or recovered. Observations after initialisation score the run; they never drive, stop or reset it. Backends: numpy | numba, explicit, never a fallback; no GPU.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.subsurface_validation import drydown_driver as dd
from benchmarks.subsurface_wet_validation import config as C
from benchmarks.subsurface_wet_validation import inputs as I

STATE_SCHEMA = "syrup.jorn160_wet.driver_state/1"
STATE_LABEL = "DRIVER-level midpoint reconstruction (column water, clock, wetting history, ledgers, nominal day, bindings); NOT a production checkpoint qualification"
ControlFailure = dd.ControlFailure
LEDGER_FIELDS = ("rain_m3", "intake_m3", "prepond_intake_m3", "postpond_intake_m3", "runoff_pool_m3", "surface_return_m3", "drainage_m3", "evaporation_m3", "evaporation_unmet_m3", "room_rejected_m3",
                 "n_onsets", "n_ponded_steps", "max_newton", "n_matrix_steps", "n_soil_steps", "n_terms", "scale_m3", "storm_time_s", "n_storm_days")


# ------------------------------------------------------------------ sources
def check_sources(cfg: dict) -> dict:
    receipt = dd.source_receipt()
    for name, want in cfg["expected_sources"].items():
        got = receipt[name]
        if got["package_sha256"] != want["package_sha256"] or got["files"] != want["files"]:
            raise ControlFailure(f"{name}: imported package {got['package_sha256'][:12]}/{got['files']} is not the declared {want['package_sha256'][:12]}/{want['files']}")
    return receipt


def _ident(receipt: dict) -> dict:
    return {k: (v["package_sha256"], v["files"]) for k, v in receipt.items()}


# ------------------------------------------------------------------ plan
def plan_report(cfg: dict, cfg_sha: str, prep: dict) -> dict:
    sched = [info for _, info in prep["schedules"]]
    m = cfg["matrix"]
    return {
        "profile": C.PROFILE, "config_sha256": cfg_sha, "name": cfg["name"], "backend": cfg["controls"]["backend"], "n_days": prep["n_days"], "n_wet_days": prep["n_wet_days"],
        "storm_max_dt_s": cfg["storm"]["max_dt_s"], "pins": prep["pins"], "rain": {k: prep["rain"][k] for k in ("total_mm_decimal", "peak_mm_per_h_decimal", "n_wet_rows")},
        "n_interpolated_air_half_hours": sum(d["n_interpolated"] for d in prep["air"]), "planned_added_rain_hours": sum(i["duration_s"] for i in sched) / 3600.0, "planned_added_rain_s": sum(i["duration_s"] for i in sched),
        "nominal_elapsed_s": prep["n_days"] * float(C.DAY_S), "planned_physical_end_s": prep["n_days"] * float(C.DAY_S) + sum(i["duration_s"] for i in sched),
        "pet_mm_day": {"min": float(prep["pet_mm_day"].min()), "max": float(prep["pet_mm_day"].max()), "sum": float(prep["pet_mm_day"].sum()), "method": "ecohyd_hargreaves via weather_pet", "terrain_factor": 1.0,
                       "air_height_note": cfg["site"]["air_height_note"]},
        "initial": {"end_utc": cfg["initial"]["end_utc"], "point_theta": prep["theta0_points"], "depths_m": prep["depths_m"], "layer_theta": prep["initial"]["layer_theta"], "layer_water_m": prep["initial"]["layer_water_m"], "total_water_m": prep["initial"]["total_water_m"],
                    "offset_model_minus_point": [float(prep["initial"]["layer_theta"][li] - prep["theta0_points"][j]) for j, li in enumerate(prep["probe_layers"])], "support_note": prep["support_note"]},
        "targets": {"valid_per_level": prep["target_counts"], "n_days": prep["n_days"], "rule": "30-minute mean ENDING at each nominal close; absent or failed targets are null, never interpolated, never a stop"},
        "matrix": {"K_mm_h": m["conductivity_m_s"] * 3.6e6, "suction_m": m["suction_m"], "suction_provenance": m["suction_provenance"], "vegetation_factor": _g(cfg),
                   "preferential_flow": "OFF (user decision); no preferential parameter is invented or fitted",
                   "peak_rain_exceeds_K": bool(Decimal_gt(prep["rain"]["peak_mm_per_h_decimal"], m["conductivity_m_s"] * 3.6e6)),
                   "disclosure": "psi and g are UNMEASURED reference inputs (EcoHyD downloaded values); if the ponded law is active (see n_ponded/n_onsets in the result) the matrix result depends on them: a scientific limit"},
        "physics": cfg["physics"], "scope": "single 1 m2 column field benchmark with a zero-residence open-surface boundary (runoff = explicit external sink); not catchment routing, not coupled Pond acceptance, not a chronological hourly clock",
    }


def Decimal_gt(text: str, value: float) -> bool:
    from decimal import Decimal

    return Decimal(text) > Decimal(repr(value))


def _g(cfg: dict) -> float:
    from maple_syrup import matrix_infiltration as mi

    m = cfg["matrix"]
    return float(mi.reference_vegetation_function(m["vegetation_cover_sum"], m["vegetation_a1"], m["vegetation_a2"]))


# ------------------------------------------------------------------ native run
class Run:
    """Mutable driver state: the authoritative column state, the wetting history, ledgers and the per-day rows (scalars only; no per-step arrays are kept)."""

    def __init__(self, cfg: dict, prep: dict) -> None:
        from maple_syrup import matrix_infiltration as mi
        from maple_syrup import subsurface as ss

        self.cfg, self.prep, self.ss, self.mi = cfg, prep, ss, mi
        self.backend = cfg["controls"]["backend"]
        t = time.perf_counter()
        scen = {"evaporation_factor": cfg["physics"]["evaporation_factor"], "diffusion_coefficient": cfg["physics"]["diffusion_coefficient"], "bottom": cfg["physics"]["bottom"]}
        self.layers = dd._layers({"layers": cfg["layers"]}, scen)
        self.column = ss.prepare_column(self.layers, np.array([True]), 1.0, backend=self.backend)
        m = cfg["matrix"]
        self.op = mi.prepare_matrix(self.column, conductivity_m_s=m["conductivity_m_s"], suction_m=m["suction_m"], vegetation_factor=_g(cfg))
        self.prepare_s = time.perf_counter() - t
        self.area = self.column.cell_area_m2
        self.thick = np.diff(cfg["layers"]["edges_m"])
        self.L = self.column.n_layers
        self.zero_add, self.zero_pet, self.zero_standing = np.zeros((self.L, 1)), np.zeros(1), np.zeros(1)
        self.warm_s = None
        self.state = self.history = None
        self.rows: list = []
        self.ledger = {k: 0.0 for k in LEDGER_FIELDS}
        self.storage0 = None

    def warm(self) -> float:
        """Run BOTH native kernels once on a DISPOSABLE control state/history (ponded-regime pool so the onset/ponded branches execute); nothing from it is ever accepted."""
        t = time.perf_counter()
        ss, mi = self.ss, self.mi
        ctl_state = ss.column_state(self.column, self.prep["initial"]["layer_water_m"].reshape(-1, 1).copy(), 0.0)
        ctl_hist = mi.wetting_history(self.op, wetting_m=0.0, cumulative_intake_m=0.0)
        rate = 1e-4                                                              # m/s, far above K: exercises onset + ponded law
        ms = mi.matrix_step(self.op, np.array([rate * 60.0]), self.zero_standing, np.array([rate]), ctl_state, ctl_hist, 60.0)
        ss.step(self.column, ms.state, 60.0, added_water_m=self.zero_add, potential_evaporation_m_s=self.zero_pet)
        ss.step(self.column, ctl_state, float(C.DAY_S), added_water_m=self.zero_add, potential_evaporation_m_s=np.array([1e-8]))
        self.warm_s = time.perf_counter() - t
        return self.warm_s

    def initialise(self) -> None:
        ss, mi = self.ss, self.mi
        self.state = ss.column_state(self.column, self.prep["initial"]["layer_water_m"].reshape(-1, 1).copy(), 0.0)
        self.history = mi.wetting_history(self.op, wetting_m=0.0, cumulative_intake_m=0.0)
        self.storage0 = ss.storage_m3(self.column, self.state)

    # -------------------------------------------------------------- one nominal day
    def advance_day(self, k: int) -> dict:
        ss, mi, cfg, prep, led = self.ss, self.mi, self.cfg, self.prep, self.ledger
        sched, info = prep["schedules"][k]
        storage_start = ss.storage_m3(self.column, self.state)
        a = dict.fromkeys(("rain", "intake", "pre", "post", "runoff", "ret", "drain", "evap", "unmet", "rejected", "n_onsets", "n_ponded", "n_storm_steps", "newton", "n_terms"), 0.0)
        perc, diff = np.zeros(self.L - 1), np.zeros(self.L - 1)
        scale = abs(storage_start)
        t_phys_before = float(self.state.t_s)

        def soil_step(dt: float, pet_m_s: float) -> None:
            nonlocal perc, diff, scale
            res = ss.step(self.column, self.state, dt, added_water_m=self.zero_add, potential_evaporation_m_s=np.array([pet_m_s]))   # intake was credited ONCE by the matrix stage: nothing is added here
            b = res.budget
            self.state = res.state
            a["ret"] += b.surface_return_m3; a["drain"] += b.drainage_m3; a["evap"] += b.evaporation_m3; a["unmet"] += b.evaporation_unmet_m3; a["n_terms"] += b.n_terms
            perc = perc + np.asarray(res.percolation_m, dtype=float)[:, 0] * self.area
            diff = diff + np.asarray(res.diffusion_m, dtype=float)[:, 0] * self.area
            scale = max(scale, b.storage_before_m3, b.storage_after_m3, b.added_m3, b.surface_return_m3, b.drainage_m3, b.evaporation_m3)
            led["n_soil_steps"] += 1

        if sched is not None:                                                      # --- storm: no E/T/growth; native gravity/diffusion every sub-step
            t, T, dtmax = 0.0, sched.end_s, float(cfg["storm"]["max_dt_s"])
            while t < T:
                nxt = min(t + dtmax, sched.next_edge_s(t), T)
                dt = nxt - t
                depth, rate = sched.depth_m(t, dt), sched.rate_after_m_per_s(t)
                ms = mi.matrix_step(self.op, np.array([depth]), self.zero_standing, np.array([rate]), self.state, self.history, dt)
                bm = ms.budget
                if bm.pool_before_m3 != depth * self.area:
                    raise ControlFailure(f"day {k}: the matrix stage was offered {bm.pool_before_m3!r} m3, not the schedule's {depth * self.area!r}")
                self.state, self.history = ms.state, ms.history
                remaining = float(np.asarray(ms.surface_pool_m)[0])
                a["rain"] += depth * self.area; a["intake"] += bm.intake_m3; a["pre"] += bm.prepond_intake_m3; a["post"] += bm.postpond_intake_m3
                a["runoff"] += remaining * self.area; a["rejected"] += bm.room_rejected_m3; a["n_onsets"] += bm.n_onsets; a["n_ponded"] += bm.n_ponded
                a["newton"] = max(a["newton"], bm.max_newton_iterations); a["n_terms"] += bm.n_terms; a["n_storm_steps"] += 1
                scale = max(scale, bm.soil_before_m3, bm.soil_after_m3, bm.pool_before_m3, bm.intake_m3)
                led["n_matrix_steps"] += 1
                soil_step(dt, 0.0)
                t = nxt
            if t != T:
                raise ControlFailure(f"day {k}: the storm schedule did not complete")
            led["storm_time_s"] += T
            led["n_storm_days"] += 1
        soil_step(float(C.DAY_S), prep["pet_mm_day"][k] / 1000.0 / C.DAY_S)       # --- the ONE native interevent step of the nominal day
        storage_end = ss.storage_m3(self.column, self.state)
        scale = max(scale, abs(storage_end))
        from maple_syrup.conservation import volume_roundoff_bound_m3

        residual = (storage_end - storage_start) - (((a["intake"] - a["ret"]) - a["drain"]) - a["evap"])
        n_terms = int(a["n_terms"]) + 4 * (int(a["n_storm_steps"]) + 1) + 8
        bound = volume_roundoff_bound_m3(n_terms, max(scale, a["rain"]))
        rain_residual = a["rain"] - (a["intake"] + a["runoff"])
        rain_bound = volume_roundoff_bound_m3(2 * int(a["n_storm_steps"]) + 2, a["rain"])
        if not math.isfinite(residual) or abs(residual) > bound or not math.isfinite(rain_residual) or abs(rain_residual) > rain_bound:
            raise ControlFailure(f"day {k}: closure violated (soil residual {residual!r} vs {bound!r} m3; rain residual {rain_residual!r} vs {rain_bound!r} m3)")
        led["n_terms"] += n_terms
        led["scale_m3"] = max(led["scale_m3"], scale, a["rain"])
        for src, dst in (("rain", "rain_m3"), ("intake", "intake_m3"), ("pre", "prepond_intake_m3"), ("post", "postpond_intake_m3"), ("runoff", "runoff_pool_m3"), ("ret", "surface_return_m3"), ("drain", "drainage_m3"),
                         ("evap", "evaporation_m3"), ("unmet", "evaporation_unmet_m3"), ("rejected", "room_rejected_m3"), ("n_onsets", "n_onsets"), ("n_ponded", "n_ponded_steps")):
            led[dst] += a[src]
        led["max_newton"] = max(led["max_newton"], a["newton"])
        water = np.asarray(self.state.water_m, dtype=float)[:, 0]
        row = {"day_index": k, "date": prep["air"][k]["date"].isoformat(), "nominal_close_utc": ni_format(prep["end_times_s"][k]), "physical_t_s": float(self.state.t_s), "added_storm_s": float(self.state.t_s) - t_phys_before - C.DAY_S,
               "water_m": water.tolist(), "layer_theta": (water / self.thick).tolist(), "pet_mm_day": float(prep["pet_mm_day"][k]), "observed_rain_mm": info.get("observed_mm", 0.0), "storm_rows": info["n_rows"],
               "first_wet_row": info["first_wet_row"], "last_wet_row": info["last_wet_row"], "n_storm_steps": int(a["n_storm_steps"]), "rain_m3": a["rain"], "intake_m3": a["intake"], "prepond_intake_m3": a["pre"],
               "postpond_intake_m3": a["post"], "runoff_pool_m3": a["runoff"], "surface_return_m3": a["ret"], "drainage_m3": a["drain"], "evaporation_m3": a["evap"], "evaporation_unmet_m3": a["unmet"],
               "room_rejected_m3": a["rejected"], "n_onsets": int(a["n_onsets"]), "n_ponded_steps": int(a["n_ponded"]), "percolation_m3": perc.tolist(), "diffusion_m3": diff.tolist(),
               "wetting_J_m": float(np.asarray(self.history.wetting_m)[0]), "cumulative_matrix_intake_m": float(np.asarray(self.history.cumulative_intake_m)[0]), "storage_m3": storage_end,
               "closure_residual_m3": residual, "closure_bound_m3": bound, "rain_closure_residual_m3": rain_residual, "rain_closure_bound_m3": rain_bound,
               "obs_theta": [None if not math.isfinite(v) else float(v) for v in prep["obs"][k]]}
        self.rows.append(row)
        return row

    def run_days(self, start: int, stop: int) -> None:
        for k in range(start, stop):
            if k != len(self.rows):
                raise ControlFailure(f"day ordinal {k} is not the next day {len(self.rows)} (duplicate or skipped ordinal refused)")
            self.advance_day(k)

    # -------------------------------------------------------------- cumulative ledger
    def cumulative_closure(self) -> dict:
        from maple_syrup.conservation import volume_roundoff_bound_m3

        led = self.ledger
        final = self.ss.storage_m3(self.column, self.state)
        residual = (final - self.storage0) - (((led["intake_m3"] - led["surface_return_m3"]) - led["drainage_m3"]) - led["evaporation_m3"])
        scale = max(abs(self.storage0), abs(final), led["rain_m3"], led["intake_m3"], led["surface_return_m3"], led["drainage_m3"], led["evaporation_m3"], led["scale_m3"])
        bound = volume_roundoff_bound_m3(int(led["n_terms"]) + 4 * len(self.rows) + 8, scale)
        rain_res = led["rain_m3"] - (led["intake_m3"] + led["runoff_pool_m3"])
        rain_bound = volume_roundoff_bound_m3(2 * int(led["n_matrix_steps"]) + 2 + len(self.rows), led["rain_m3"])
        if not math.isfinite(residual) or abs(residual) > bound or not math.isfinite(rain_res) or abs(rain_res) > rain_bound:
            raise ControlFailure(f"cumulative closure violated (soil {residual!r} vs {bound!r}; rain {rain_res!r} vs {rain_bound!r})")
        return {"storage_initial_m3": self.storage0, "storage_final_m3": final, "soil_residual_m3": residual, "soil_bound_m3": bound, "rain_residual_m3": rain_res, "rain_bound_m3": rain_bound,
                "total_runoff_m3": led["runoff_pool_m3"] + led["surface_return_m3"], "native_bound": "conservation.volume_roundoff_bound_m3 with the sum of the native per-step term counts plus the driver's own accumulations"}

    # -------------------------------------------------------------- driver-level export / import
    def export_state(self, cfg_sha: str, sources: dict) -> dict:
        body = {"schema": STATE_SCHEMA, "label": STATE_LABEL, "config_sha256": cfg_sha, "backend": self.backend, "sources": _ident(sources), "next_day": len(self.rows),
                "water_m": np.asarray(self.state.water_m, dtype=float)[:, 0].tolist(), "t_s": float(self.state.t_s), "wetting_m": float(np.asarray(self.history.wetting_m)[0]),
                "cumulative_intake_m": float(np.asarray(self.history.cumulative_intake_m)[0]), "ledger": dict(self.ledger), "storage0_m3": self.storage0, "rows": self.rows}
        body["state_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return body

    def import_state(self, st: dict, cfg_sha: str, sources: dict) -> None:
        """Stage EVERY candidate field and validate it before the first assignment: a refused import leaves this Run unchanged. Bounded contract validation (not physics replay, not a signature): seal, bindings, ordinals,
        rows against the current inputs, water/clock/history against the last row, the initial storage against the declared initial volume, and the foldable ledgers/clock against the rows (no phantom rain/E/time)."""
        body = {k: v for k, v in st.items() if k != "state_sha256"}
        if st.get("schema") != STATE_SCHEMA or hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest() != st.get("state_sha256"):
            raise ControlFailure("the resume state is not a valid, unmodified driver state")
        if st["config_sha256"] != cfg_sha or st["backend"] != self.backend:
            raise ControlFailure("the resume state belongs to a different config or backend (refused)")
        if {k: tuple(v) for k, v in st["sources"].items()} != _ident(sources):
            raise ControlFailure("the resume state was made with different source packages (refused)")
        rows = copy.deepcopy(st["rows"])
        prep = self.prep
        if [r["day_index"] for r in rows] != list(range(len(rows))) or st["next_day"] != len(rows) or not 0 < len(rows) < prep["n_days"]:
            raise ControlFailure("the resume state has duplicate, missing or out-of-range day ordinals")
        for k, r in enumerate(rows):
            info = prep["schedules"][k][1]
            if (r["date"] != prep["air"][k]["date"].isoformat() or r["nominal_close_utc"] != ni_format(prep["end_times_s"][k]) or r["pet_mm_day"] != float(prep["pet_mm_day"][k])
                    or r["storm_rows"] != info["n_rows"] or r["first_wet_row"] != info["first_wet_row"] or r["last_wet_row"] != info["last_wet_row"]):
                raise ControlFailure(f"resume state row {k} does not match the current inputs")
        last = rows[-1]
        ledger = {k: float(st["ledger"][k]) for k in LEDGER_FIELDS}
        fold = dict.fromkeys(LEDGER_FIELDS, 0.0)
        for r in rows:                                                              # the same sequential additions as `advance_day`
            for src, dst in (("rain_m3", "rain_m3"), ("intake_m3", "intake_m3"), ("prepond_intake_m3", "prepond_intake_m3"), ("postpond_intake_m3", "postpond_intake_m3"), ("runoff_pool_m3", "runoff_pool_m3"),
                             ("surface_return_m3", "surface_return_m3"), ("drainage_m3", "drainage_m3"), ("evaporation_m3", "evaporation_m3"), ("evaporation_unmet_m3", "evaporation_unmet_m3"),
                             ("room_rejected_m3", "room_rejected_m3"), ("n_onsets", "n_onsets"), ("n_ponded_steps", "n_ponded_steps")):
                fold[dst] += r[src]
            fold["n_matrix_steps"] += r["n_storm_steps"]
            fold["n_soil_steps"] += r["n_storm_steps"] + 1
            fold["storm_time_s"] += r["added_storm_s"]
            fold["n_storm_days"] += 1 if r["storm_rows"] else 0
        for key in fold:
            if key not in ("max_newton", "n_terms", "scale_m3") and ledger[key] != fold[key]:
                raise ControlFailure(f"resume ledger {key} {ledger[key]!r} is not the fold of the rows {fold[key]!r}")
        if not all(math.isfinite(ledger[k]) and ledger[k] >= 0.0 for k in ("max_newton", "n_terms", "scale_m3")):
            raise ControlFailure("resume ledger counters must be finite and nonnegative")
        if st["water_m"] != last["water_m"] or float(st["t_s"]) != last["physical_t_s"] or float(st["t_s"]) != C.DAY_S * len(rows) + ledger["storm_time_s"]:
            raise ControlFailure("resume water/clock disagree with the last row or the folded time")
        if st["wetting_m"] != last["wetting_J_m"] or st["cumulative_intake_m"] != last["cumulative_matrix_intake_m"]:
            raise ControlFailure("resume wetting history disagrees with the last row")
        initial = self.ss.storage_m3(self.column, self.ss.column_state(self.column, prep["initial"]["layer_water_m"].reshape(-1, 1).copy(), 0.0))
        if float(st["storage0_m3"]) != initial:
            raise ControlFailure("resume initial storage is not the declared initial volume")
        state = self.ss.column_state(self.column, np.array(st["water_m"]).reshape(-1, 1), float(st["t_s"]))
        history = self.mi.wetting_history(self.op, wetting_m=np.array([st["wetting_m"]]), cumulative_intake_m=np.array([st["cumulative_intake_m"]]))
        self.rows, self.ledger, self.storage0, self.state, self.history = rows, ledger, initial, state, history                  # all candidates validated: assign


def ni_format(t_s: Any) -> str:
    from benchmarks.subsurface_validation import neon_inputs as ni

    return ni.format_utc(int(t_s))


# ------------------------------------------------------------------ execute
def execute(cfg: dict, cfg_sha: str, *, stop_after_day: int | None = None, resume_state: dict | None = None) -> tuple:
    """Returns (result, state_or_None, timing). Staged: config/pins/forcing/targets/sources are all validated before the first native field step."""
    timing = {}
    t0 = time.perf_counter()
    before = check_sources(cfg)
    prep = I.prepare_inputs(cfg)
    n = prep["n_days"]
    if stop_after_day is not None and not (isinstance(stop_after_day, int) and 0 < stop_after_day < n):
        raise ControlFailure(f"stop_after_day must be an integer in [1, {n - 1}]")
    timing["source_checks_and_input_preparation_s"] = time.perf_counter() - t0
    run = Run(cfg, prep)
    timing["operator_prepare_s"] = run.prepare_s
    timing["jit_warmup_control_s"] = run.warm()
    if resume_state is not None:
        run.import_state(resume_state, cfg_sha, before)
        start = len(run.rows)
        if stop_after_day is not None and stop_after_day <= start:
            raise ControlFailure("stop_after_day must be beyond the resumed day")
    else:
        run.initialise()
        start = 0
    stop = n if stop_after_day is None else stop_after_day
    steps0 = (run.ledger["n_matrix_steps"], run.ledger["n_soil_steps"])
    t1 = time.perf_counter()
    run.run_days(start, stop)
    timing["integration_s"] = time.perf_counter() - t1
    timing["n_days_integrated_this_process"] = stop - start
    timing["n_matrix_steps_this_process"] = int(run.ledger["n_matrix_steps"] - steps0[0])
    timing["n_soil_steps_this_process"] = int(run.ledger["n_soil_steps"] - steps0[1])
    timing["n_days_cumulative"], timing["n_matrix_steps_cumulative"], timing["n_soil_steps_cumulative"] = len(run.rows), int(run.ledger["n_matrix_steps"]), int(run.ledger["n_soil_steps"])
    timing["notes"] = ("source_checks_and_input_preparation_s starts inside execute AFTER the module imports and excludes interpreter start and those imports; full fresh-process wall time is measured separately by Root. "
                       "*_this_process count only what this invocation integrated; *_cumulative include resumed days.")
    timing["backend"], timing["python"], timing["numba"] = run.backend, sys.version.split()[0], _numba_version()
    timing["clock"] = "time.perf_counter raw seconds"
    after = check_sources(cfg)
    if _ident(before) != _ident(after):
        raise ControlFailure("the source packages changed during the run")
    complete = stop == n
    closure = run.cumulative_closure()
    led = run.ledger
    result = {"profile": C.PROFILE, "config_sha256": cfg_sha, "complete": complete, "days_done": len(run.rows), "backend": run.backend, "sources_before": before, "sources_after": after, "plan": plan_report(cfg, cfg_sha, prep),
              "rows": run.rows, "ledger": led, "cumulative_closure": closure, "added_rain_hours": led["storm_time_s"] / 3600.0, "physical_end_t_s": float(run.state.t_s),
              "nominal_elapsed_s": len(run.rows) * float(C.DAY_S),
              "matrix_regime": {"n_ponded_steps": led["n_ponded_steps"], "n_onsets": led["n_onsets"], "room_rejected_m3": led["room_rejected_m3"], "runoff_matrix_pool_m3": led["runoff_pool_m3"], "surface_return_m3": led["surface_return_m3"]},
              "scores": I.score_run(prep, np.array([r["layer_theta"] for r in run.rows])) if complete else None}
    state = None if complete else run.export_state(cfg_sha, after)
    return result, state, timing


def _numba_version() -> str | None:
    try:
        import numba

        return numba.__version__
    except ImportError:
        return None


# ------------------------------------------------------------------ outputs (immutable, deterministic retry)
def write_immutable(path: Path, text: str) -> str:
    data = text.encode()
    if path.exists():
        if path.read_bytes() == data:
            return "identical_retry"
        raise ControlFailure(f"{path} exists with different content: refused (outputs are immutable)")
    with open(path, "xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    return "written"


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--execute", action="store_true", help="run the native field column (default: plan only)")
    ap.add_argument("--stop-after-day", type=int, default=None)
    ap.add_argument("--export-state", default=None)
    ap.add_argument("--resume-state", default=None)
    a = ap.parse_args(argv)
    try:
        cfg, sha = C.load_config(a.config)
    except (C.ConfigError, json.JSONDecodeError) as exc:
        print(f"CONFIG REFUSED: {exc}", file=sys.stderr)
        return 2
    if (a.stop_after_day or a.export_state or a.resume_state) and not a.execute:
        print("--stop-after-day/--export-state/--resume-state need --execute", file=sys.stderr)
        return 2
    if bool(a.stop_after_day) != bool(a.export_state):
        print("--stop-after-day and --export-state go together", file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else None
    if not a.execute:
        check_sources(cfg)
        text = json.dumps(I.jsonable(plan_report(cfg, sha, I.prepare_inputs(cfg))), indent=1, allow_nan=False)
        if out:
            out.mkdir(parents=True, exist_ok=True)
            write_immutable(out / "plan.json", text + "\n")
        print(text)
        return 0
    resume = json.loads(Path(a.resume_state).read_bytes()) if a.resume_state else None
    result, state, timing = execute(cfg, sha, stop_after_day=a.stop_after_day, resume_state=resume)
    text = json.dumps(I.jsonable(result), indent=1, allow_nan=False) + "\n"
    if out:
        out.mkdir(parents=True, exist_ok=True)
        t = time.perf_counter()
        name = "result.json" if result["complete"] else f"result_partial_day{result['days_done']}.json"
        how = write_immutable(out / name, text)
        if state is not None:
            write_immutable(Path(a.export_state), json.dumps(state, indent=1, allow_nan=False) + "\n")
        timing["output_write_s"] = time.perf_counter() - t
        tname = "timing.json" if how == "written" else f"timing_retry_{int(time.time())}.json"
        write_immutable(out / tname, json.dumps(timing, indent=1) + "\n")
    print(json.dumps({"complete": result["complete"], "days_done": result["days_done"], "timing": timing}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
