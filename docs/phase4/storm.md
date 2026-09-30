# Phase 4c — coupled water-only Plot 1 storm runner

Status: accepted CPU water-only milestone, 2026-09-29, after Claude implementation/corrections and independent Codex verification. Full regression: 366 passed, 5 GPU-dependent skips. The separate controlled original-Fortran storm benchmark also ran and received Claude review. See [measured results, reproduction and limitations](storm_acceptance.md). No sediment, dry reset, restart, ecology or wind is included.

Files: `src/maple_syrup/storm.py` (pure coupled step and evolution), `src/maple_syrup/storm_experiment.py` (actual-MAPLE Plot 1 runner and CLI `maple-syrup-storm`), `tests/phase4/test_storm.py`, `tests/phase4/test_storm_experiment.py`, one entry in `pyproject.toml`. One narrow addition to `routing.py`: `RoutingStepRejected(RoutingError)` (§3).

## 1. Coupling contract (one attempt of length dt, pure)

Order follows the MAHLERAN storm loop (`MAHLERAN_storm_xml.f90` 100–104: `infilt` then `route_water`), on the **original** pre-attempt state `(h, S, q_prev)`:

```
col   = column_step(params, h, S, rain_rate, dt)           accepted Phase 3 kernel, unchanged
h*    = col.depth_m                                        legacy d(1) + excess·dt (rain excess and saturation return included)
hpre  = max(h − max(J − P, 0), 0)  =  min(h, h* − O)        legacy post-infilt d(1)
q_old = 0                   if J ≥ h + P    (complete run-on; infilt.for 106–114, tested FIRST)
      = q_prev              elif J ≤ P      (no run-on; 125–131, legacy q(1) unchanged)
      = k·hpre^{3/2}        otherwise       (partial run-on; 141–156)
route = route_step(graph, h*, hpre, dt, old_discharge = q_old)   accepted Phase 4b kernel, unchanged
state' = (t + dt, route.depth_m, col.soil_water_m, route.discharge_m2_s)
```

with J = intake, P = rain depth, O = saturation return. The branch precedence is the legacy `infilt.for` order (complete, then no run-on, then partial), so the three masks and their counts are mutually exclusive. At the overlap `h = 0, J = P` (a dry cell whose rain infiltrates completely) both branches give `q_old = 0`; it is counted as complete run-on. Inside `route_step` the receiver's old inflow is the donor sum of this same `q_old` (the accepted conservation correction). The legacy stale `qin(1)` mode is **not** used for production. The initial `q` is `k·h0^{3/2}` (`initial_state`), so the relation check `q_old ↔ hpre` holds from the first step. The capacity of the next column step uses the routed depth, as in Phase 3 (pre-step depth in `(ψ + h)`). Both kernels are pure, so a rejected attempt leaves the state untouched.

**State validation at the boundary.** `initial_state` and `evolve` validate the state once (one batched flag read): graph namespace (mixed namespaces refused), shape, float64, finite, non-negative, `t ≥ 0`, zero discharge on inactive cells, discharge consistent with depth through `q = k h^{3/2}` within the routing root tolerance, and the column parameters' `active_mask` equal to the graph's active cells (a mismatch is unsupported and refused). Bad input is therefore refused up front, never masked by the coupled step recomputing `q` in a branch. `evolve` owns every scratch array it writes (there is no caller-supplied buffer), so caller arrays are never modified, even when a later kernel raises.

Per-cell branch counts (complete / no run-on / partial) are accumulated on the device and reported.

## 2. Time, boundaries and forcing

- Real Plot 1 rainfall record, interval-ending semantics, exact per-piece integration (Phase 3 `RainfallSchedule`). The rate for an attempt is `rate_after(t)`; because every rainfall knot inside the window is a hard boundary, an attempt never crosses a rate change.
- Boundaries (`plan_boundaries`) = knots in `(start, end)` ∪ `start + k·report_every_s` ∪ `end`, strictly increasing; the count is fixed in advance and bounded by `max_report_rows` (a strict positive int, default 100000). Two candidate boundaries closer than 16 ε (relative) are the same instant in floating point (for example `0.1·3 = 0.30000000000000004` and a `0.3` forcing edge): report-time coincidences are merged into one boundary that keeps the **exact forcing edge or end**; two distinct exact forcing/end boundaries are always preserved, so no rate change is skipped and no sub-ε slice is created. Genuinely distinct boundaries (any spacing above that noise level) are all kept.
- Within each boundary interval the steps are a planned uniform partition: `n = ceil(span / max_dt_s)` equal substeps of `span / n ≤ max_dt_s`, each landing exactly on its target (the last on the boundary). A 60 s interval at `max_dt_s = 0.25` is therefore 240 identical steps, and two reporting cadences that share the same integer partition produce the same accepted steps (bitwise, tested). A planned target that floating time cannot distinguish from the current time is refused (`StormError`, "floating time does not advance").
- `dt ≤ max_dt_s` (default 1 s, the legacy dt); `end_s` defaults to the legacy `stormlength` (5400 s; Plot 1 rain ends at 1620 s, so the default run includes 3780 s of dry recession). An explicit `end_s` allows controlled partial windows; the rainfall integral check then uses `schedule.depth_m(0, end)` — only the simulated interval.
- `min_dt_s` (default 1/1024 s) is a **retry floor**, not a minimum slice: a forcing/report/end boundary may force a shorter step (an `end_s` of 1e-4 s, a 0.5 ms forcing interval) and it is simply stepped, as long as time advances. Only a halving *after a rejection* that would fall below the floor is refused.
- The run ends at the configured time with **all residual water retained**. The status says explicitly that this is **not** an event completion, whatever the remaining flow. There is no flow-stop criterion and no dry reset.

## 3. Transactional adaptive stepping

- Only `RoutingStepRejected` (a narrow `RoutingError` subclass raised by `route_step` for the two recoverable conditions: old-flux Courant number above `courant_max`, or a negative right-hand side) triggers a retry: dt is halved and the **entire** column + routing attempt is recomputed from the unchanged original state. Failed attempts accumulate nothing — no fluxes, no time, no hydrograph row.
- Every other exception propagates unchanged: non-convergence of the bisection, invalid inputs, non-finite outputs, programming errors. Nothing is clipped and there is no fallback (`routing_numba` missing → error, no array substitute).
- Guards (`StormError`): the retry floor `min_dt_s` (default 1/1024 s, applied to halvings only), `max_retries` per step (default 10), `max_steps` (default 1e7), a refusal when `t + dt == t` in floating point, and a refusal of a planned substep target that is not after the current time. After a retry sequence the next planned substep is attempted at its full length again. All control values are validated strictly on the values given (bools and non-integers are refused, never coerced).
- A failed call leaves inputs and the MAPLE state untouched and publishes nothing.

Change to `routing.py` (necessary, narrow): `RoutingStepRejected` is raised instead of the base class when the first violated flag is the Courant or negative-RHS check. Messages are unchanged; all previous `match="Courant"` tests still hold. The runner classifies by type, never by string.

## 4. Runner (`run_plot1_storm`, `python -m maple_syrup.storm_experiment`)

Reuses, without copying: `verify_plot1_case`, `plot1_parameters`, `_bed_digest`, `_syrup_provenance`, `_source_digests`, `_Clock` from Phase 3; `plot1_routing_graph` (units, row orientation, 10 south outlets, f = 21.45, nodata declared); MAPLE `resolve_backend`, `to_device`, `to_host`, `read_transfer_counters`, `synchronize`, `validate_water_state`.

- Refusals before any work: existing output path, path inside the case, invalid arguments, `implementation="numba"` with a non-numpy backend, Numba requested but not importable (clear error, **no fallback**). After verification: output inside the MAPLE, MAHLERAN or recipe trees.
- Implementation default **numba** (CPU NumPy); `array` is the reference/backend/GPU-compatible path. GPU execution is not exercised or claimed here.
- Loop: arrays stay in the backend namespace; two validating flag reads per attempt (column, routing); the hydrograph is a preallocated device buffer with one row per boundary; cumulative grids, peak-depth/velocity maps and 0-d scalars are device-resident; no grid transfer in the loop. Transfer counters over the loop are recorded.
- **True peaks.** The peak outlet discharge and its time are tracked per accepted step as two 0-d device values (no host read, O(1) memory), and `peak_depth`/`peak_velocity` maps start from the initial state (`v0 = k√h0`), so a pure recession reports its initial values as the maxima. The summary's `peak_outlet_discharge_m3_s` / `time_of_peak_outlet_discharge_s` are these true numerical peaks; the hydrograph rows remain samples at the reporting cadence and their maximum is reported separately as `sampled_peak_outlet_discharge_m3_s`.
- Timing: setup/verification; the **first accepted step** separately (includes lazy import and JIT for numba); the remaining steps; reporting. Not a controlled benchmark.
- After the loop, in this order and before writing anything: one stacked scalar read; final grids to the host once; finiteness and non-negativity of every grid and the hydrograph; `dataclasses.replace(case.water, depth_m=final)` validated by MAPLE; sediment-side digest unchanged; budgets (§5); source-stability digests unchanged. Any failure raises `StormError` and writes nothing.
- Outputs in a NEW directory: `storm_summary.json`; `final_water.npz` (depth, soil water, discharge, velocity, peak depth, peak velocity, cumulative rain/intake/return/drainage); `hydrograph.npz` and `hydrograph.csv` (per boundary: time, cumulative rain/intake/return/drainage/export volumes, surface and soil storage, **instantaneous** outlet discharge = Σ q_new·dx over outlets (legacy `q_plot·dx`), max depth/velocity, accepted steps, rejected attempts, min accepted dt, max routing cell-balance and constitutive residuals; plus interval differences and the per-row water residual). Output SHA-256 values are in the summary.

`export` in the budget is the time-integrated Crank–Nicolson face volume through outlet cells; `outlet_discharge_m3_s` is the instantaneous end-of-step value. They are different quantities and are labelled as such.

## 5. Budgets and tolerances

Depth sums over cells (×area for m³):

- global: `surface_final + soil_final + drainage + export = surface_initial + soil_initial + rain`;
- surface: `surface_final = surface_initial + rain − intake + return − export`;
- soil: `soil_final = soil_initial + intake − drainage − return`;
- forcing: `rain = schedule.depth_m(0, end) · Σ rainfall_scale`, tolerance `4ε (n_steps + n_cells) rain`;
- every hydrograph row satisfies the global identity;
- tolerance for the three balances: `(16 + 32) ε (n_steps + n_cells) (initial + rain + intake + drainage + return + export)`, i.e. the Phase 3 column tolerance plus the Phase 4b routing tolerance, per step and per cell. No negative clipping anywhere; non-negativity is validated by the kernels every step and again at the end.
- Also reported: max Courant (old/new flux), max per-step routing cell-balance residual, max constitutive residual (≤ 1e-11 m or the step would have been rejected), cell-step branch counts, rejections with `(t, dt, reason)` for the first 100.

## 6. Reproduction commands (executed by Codex; results linked above)

```
cd /home/okin/SYRUP
PY=/home/okin/MAPLE/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src:/tmp/syrup-numba NUMBA_CACHE_DIR=/tmp/syrup-numba-cache MAPLE_SYRUP_NUMBA_CACHE=0

# New tests (small graphs, then the actual MAPLE case with short windows):
$PY -m pytest -q -rs -p no:cacheprovider tests/phase4/test_storm.py tests/phase4/test_storm_experiment.py
# Full regression (330 passed / 4 GPU skips was the pre-task baseline), lint, whitespace:
$PY -m pytest -q -rs -p no:cacheprovider
$PY -m ruff check src/maple_syrup tests benchmarks
git diff --check

# Plot 1 storm + recession to 5400 s, dt study, both implementations (same forcing, same case):
for DT in 1 0.5 0.25; do
  $PY -m maple_syrup.storm_experiment --case-dir outputs/plot1 --output-dir outputs/plot1_storm_numba_dt$DT --max-dt-s $DT --implementation numba
done
$PY -m maple_syrup.storm_experiment --case-dir outputs/plot1 --output-dir outputs/plot1_storm_array_dt1 --max-dt-s 1 --implementation array
# Expected (to verify, not asserted): identical final grids/hydrograph between array and numba at dt 1;
# 5400 accepted steps at dt 1 with 0 rejections; residuals within tolerance_m3; export > 0; ponded water retained.
```

`outputs/plot1` is the accepted Phase 2 case (generated data, not in git). Use another new output directory for every run; an existing path is refused.

## 7. Limitations

- Not an event completion: the configured end retains any residual surface water and outlet discharge; Phase 6 defines stopping and the dry handoff.
- No sediment, no terrain refresh (graph fixed), no restart sidecar (state is `(t, h, S, q)` in memory), no GPU run. The separately executed original-routine benchmark is documented in storm_acceptance.md.
- Constant friction 21.45, draining D4 terrain, 10 south outlets, declared restrictions of Phase 4b (no pits, f ≥ 0.1, no nodata).
- A retry sequence restarts from the full planned substep at the next target; repeated rejections are counted and reported but not adapted away.
- Only report-time coincidences are merged; distinct exact forcing/end boundaries are preserved; planned substeps below the time resolution are refused rather than silently coarsened.
- Costs: the array path is roughly 66 levels × 40 bisections of small array operations per step (measured 11.5 ms/step for a synthetic 60×20 routing step in acceptance.md), so a 5400 s array run is minutes; the numba path was ~1 ms/step for routing alone. Whole-storm times are reported by the runner, not predicted here.
