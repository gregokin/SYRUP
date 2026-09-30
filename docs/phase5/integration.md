# Phase 5b — coupled water + wet-sediment event on the actual MAPLE bed

Status: implemented 2026-09-30 for task `phase5b_integration` by Claude (writer of `src/maple_syrup/sediment_event.py`, `src/maple_syrup/sediment_experiment.py`, `tests/phase5/test_sediment_event.py`, `tests/phase5/test_sediment_experiment.py`, this document; `pyproject.toml` gained the CLI entry and `tests/phase5` in `testpaths`). **Nothing in this task was executed by the author** (no shell): every test, timing and Plot 1 result below is a specification for Codex's independent run, not a measurement. The bed seam `src/maple_syrup/sediment_bed.py` is Codex-authored (Claude read-only review: no must-fix) and is reused unchanged. Phase 5a physics/transport (`docs/phase5/physics.md`) is reused; the only Phase 5a edit is the `run_to_completion` test helper (terminates on the TOTAL remaining pool, as the assertion requires).

Not implemented or claimed: splash, ecology, nutrients, an event-stop criterion, the dry reset, wind alternation, restart, GPU execution, a MAHLERAN Fortran storm benchmark.

Subsequent Codex execution and independent review are recorded in [acceptance.md](acceptance.md); that record supersedes the author-only verification status above.

## 1. State ownership (one authority)

| Quantity | Owner | Where |
|---|---|---|
| Voxel column, active layer, availability, ledger, committed topography | MAPLE | `sediment_bed.BedState` (MAPLE objects, changed only through `apply_water_process_demand` and `commit_topography`) |
| Water depth, water-borne mobile mass | MAPLE `WaterState` inside `BedState.water`; depth is SYRUP-owned and PUBLISHED into it before every MAPLE call (`constant_depth`) | `sediment_bed.with_water` |
| Soil water, discharge, time | SYRUP | `storm.StormState` (`SedimentEventState.storm`); `storm.depth_m` equals `bed.water.depth_m` after every accepted step (checked at entry and end) |
| Sediment velocity memory `(ny, nx, nc)` | SYRUP | `SedimentEventState.sediment_velocity_m_s` |
| Routing graph, transport network, physics grid | derived from the committed terrain, rebuilt at every commit | `SedimentEventState.graph / network / grid`; `network.graph_input_sha256` must equal `graph.input_sha256` (checked) |
| Boundary ring, initial datum | immutable evidence | `sediment_bed.TerrainReference` |

There is no SYRUP copy of the bed or of an evolving elevation. Elevation change is MAPLE's (bulk density 1250 kg m⁻³ on solid mass at 2650 kg m⁻³: 2.12 × the legacy `z_change`, reported in every summary).

## 2. One accepted step (all local until published)

```
1  storm.coupled_step(graph, column, rain, storm, dt)            Phase 3 column + Phase 4b method 5; RoutingStepRejected -> retry
2  sediment_physics_step(params, grid, route.depth, route.velocity, rain, veg,
                         bed.active_layer.mass_kg (CURRENT holdings), v_memory, dt)
3  bed = with_water(bed, ctx, route.depth)                          publish the solver depth
   bed, pickup = apply_bed_demand(bed, ctx, Demand(requested_pickup, 0))
                                                                    MAPLE: availability cap, holdings cap, refill, ACTUAL removal -> mobile
4  n_sub = ceil(v_max dt / (dx courant_max))                        sediment Courant; doubled on an FP-edge rejection; > max -> TransportStepRejected -> retry
   transport = transport_step(network, bed.water.mobile, v, 1/L, settle, dt, n_substeps=n_sub)
5  bed = with_water(bed, ctx, route.depth, T(M))                    publish T(M)
   bed, deposit = apply_bed_demand(bed, ctx, water_demand_from_transport(transport, 0))
                                                                    MAPLE: deposition <= mobile, export <= remaining, ledger both gross directions
6  commit_bed(bed, ctx, graph, terrain, t + dt)                     MAPLE triggers; on commit: constant_depth verified (depth array-equal,
                                                                    displaced volume 0), graph rebuilt from the committed surface with the fixed
                                                                    ring, network + grid rebuilt, discharge = k_new h^{3/2} (storm.initial_state)
7  publish t, storm, bed, v_memory = physics.sediment_velocity_m_s, graph objects, accumulators
```

Time levels: the laws see the routed (post-step) depth and velocity and the step's rain rate (constant inside a step: every rainfall knot is a boundary). Pickup demand is evaluated on the pre-pickup holdings, so class fractions and d50 are the CURRENT active-layer composition; the actual (capped) pickup is transported in the same step (no source lag). Dry / no-capacity cells request their whole pool for deposition, so mobile mass reaching a dry cell returns to the MAPLE bed; wet residual mobile mass is retained (Phase 6 completion policy).

Retry: `RoutingStepRejected` and `TransportStepRejected` discard the whole unpublished attempt and halve `dt` from the same state (`max_retries`, retry floor `min_dt_s`; forced slices to forcing / reporting boundaries may be shorter). Every other exception propagates untouched; nothing is published (tested with a failing commit and a failing MAPLE call).

Post-commit diagnostics (correction after Codex note 4): when a commit re-initialises the discharge, the outlet mask, `last_velocity = k_new sqrt(h)`, the instantaneous outlet discharge and the peaks are refreshed from the ACCEPTED post-commit graph and state (`refresh_hydraulics`), after the pre-commit routed values were recorded; peaks therefore cover both. The integrated water export stays the routing step's face volume; nothing is re-integrated from the re-initialised `q`. The same refresh runs after the forced final commit, so `last_velocity_m_s`, `final_state.npz` velocity/discharge and the last hydrograph row describe the returned state.

Commit: `sediment_bed.commit_bed` composes `evaluate_commit_triggers` + `commit_topography` with the case's avalanching spec (Plot 1: enabled, 34°, never active on these slopes; recorded) and the `constant_depth` water callback. The event additionally requires `depth_update.rule == "constant_depth"`, `displaced_volume_m3 == 0`, no clamp, and an unchanged shape / spacing / active set of the refreshed graph. The exporting boundary ring and the active set are fixed; WHICH interior cells drain into the ring (the outlet set) may change with the terrain: a valid reroute is accepted, the transport network is rebuilt on the new outlets, the outlet-based diagnostics are refreshed, and `outlets_changed` / `n_outlets` are logged per commit. A commit that creates a pit, a flat or a lost receiver raises `RoutingGraphError` from `refresh_routing` before anything is published (no infill, no water loss). `graph_changed` (and `n_graph_changes`) means a PHYSICAL change of slope or receiver; `graph_rebound` marks a differing `input_sha256` binding only (the refresh omits the initial Plot 1 graph's nodata metadata, so the first zero-change commit rebinds without changing geometry). At the configured end a commit is forced (real time `t_end`) only when the ledger holds physical activity since the last commit and MAPLE's `is_already_committed` is false, so the returned terrain corresponds to the returned bed without a no-op commit. Per commit the log records the trigger reasons, `max |dz|`, rerouted cells (aspect changes), graph hash, avalanche flag, depth continuity and the pre-reset water-channel pending mass; the ledger's `(n_process, nc)` totals are summed across commits before MAPLE resets them.

## 3. Accounting

Device-resident, bounded (no per-step history): cumulative rain / intake / return / drainage grids and export volume (Phase 4 columns); cumulative actual pickup, actual deposition (MAPLE ground truth) and export request per cell and class; per class: requested / raindrop / flow pickup, actual pickup, availability shortfall, holdings shortfall, pickup numerical residual, deposition requested / actual / unmet, export requested / actual / unmet, transport budget residual and tolerance, deposit numerical residual; class-unaware scalar residual; true per-step peaks (mobile mass, export rate, outlet discharge, depth, velocity) with their times; max sediment Courant, max `v r dt`, max transport cell residual, max substeps, regime cell-class-step counts; commit count, forced commits, graph changes, rerouted cells.

Closure (`SedimentEventResult.closure()`), independent of the ledger: `bed_final + mobile_final + export_actual − (bed_initial + mobile_initial)` per class from `bed_inventory` at both ends, against `64 ε (n_inventory_terms + 2 n_water_calls) max(inventory) + |MAPLE per-class residuals| + |scalar residual|`; and `requested = actual + availability_shortfall + holdings_shortfall + |residual|`. The runner refuses output when either fails, when MAPLE refused deposition / export beyond `2 n_steps n_cells mres` plus FP, or when the accumulated transport residual exceeds its bound.

Morphology (correction after Codex note 7): two grids are kept apart. `water_exchange_net_kg = cumulative_deposition − cumulative_pickup` per cell and class is the WATER class exchange only. `bed_change_kg = (voxel.sum(axis=2) + active)_final − (…)_initial` per cell and class (`SedimentEventResult.bed_change_by_cell_class_kg()`, robust to the voxel count) is the ACTUAL bed inventory change, including what MAPLE avalanching inside commits and sub-resolution placement did. `morphology_summary` derives net erosion / deposition after summing the classes PER CELL (a balanced compositional exchange is zero morphology and is reported as `class_sorting_exchange_kg`), the elevation-equivalent change, the per-class bed change and water exchange, and `non_water_bed_change_abs_kg` (bed change the water exchange does not explain, from independent inventories). The runner's `net_erosion_kg` / `net_deposition_kg` are these morphological values.

Hydrographs: the Phase 4 water columns unchanged, plus `sediment_hydrograph_columns(nc)`: `t_s, mobile_kg, cumulative_pickup_kg, cumulative_deposition_kg, cumulative_export_kg, export_rate_kg_s, commit_count, graph_changes, max_transport_substeps` and per-class blocks `mobile_c*_kg, export_rate_c*_kg_s, cumulative_pickup_c*_kg, cumulative_deposition_c*_kg, cumulative_export_c*_kg`. `export_rate` is the last accepted step's actual export / dt (a sample); the true peak is tracked per step.

## 4. Plot 1 runner (`sediment_experiment.py`)

```
PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src[:/tmp/syrup-numba] \
/home/okin/MAPLE/.venv/bin/python -m maple_syrup.sediment_experiment \
  --case-dir outputs/plot1 --output-dir NEW_DIR --max-dt-s 1 --end-s 5400 --report-every-s 1 --implementation numba
```

- Case re-verified (`verify_plot1_case`); rainfall hash-checked; Phase 3 columns and Phase 4 graph as `storm_experiment`.
- Sediment parameters from the ORIGINAL root XML (`plot1_sediment_parameters_from_xml`): raindrop a/b/c/hs per class (`spa / 1200` applied inside the law), particle density, `active_layer_sensitivity`, `KE_model_type` (2 → `verstraeten_exp`), `time_step` (reference pickup interval); the six diameters and the particle density come from — and must equal — the actual MAPLE grain classes; the result must equal the transcribed Plot 1 constants. Conventions: `ke_vegetation_form=legacy_literal`, `distance_convention=legacy_literal`, `dstar_convention=legacy_sigma_minus_one`, `bagnold_depth_units=legacy_mm`, `raindrop_composition_scaling=legacy_none`. The XML is hashed before and after the loop.
- Bed from the case through `bed_from_case` (declared uniform availability). The imported case has `water_coupling.enabled=false, event_kind=aeolian, depth_update_rule=constant_free_surface`; the run overrides ONLY the depth ownership to `constant_depth` and records it with the resolved configuration (MAPLE `config_to_dict` + SYRUP controls, parameters, conventions, graph and rainfall hashes) and its SHA-256. MAPLE is called with the explicit adapter label `maple_syrup/phase5`; no stock adapter is used or relabelled.
- Outputs, assembled in a temporary sibling and renamed only after every check: `sediment_summary.json`, `final_state.npz` (voxel / active / available / bound masses, committed and initial elevation, pending ledger mass (zero after the forced commit), depth, mobile, soil water, discharge, velocity, sediment velocity memory, cumulative pickup / deposition / export request, net bed change, peaks, initial and final aspect), `hydrograph.npz` / `.csv` (water columns + `sed_*`), `maple_final_snapshot.npz` (MAPLE `save_state_snapshot`, validated by MAPLE, with availability and water). The snapshot is a MAPLE state record, **not a SYRUP restart**; hydraulic and transport memory are in `final_state.npz` for inspection only.
- Refused: existing output path, output inside the case / MAPLE / MAHLERAN / recipe trees, `backend != numpy` (host-only commit path; no hidden fallback), missing Numba with `--implementation numba`, source or XML change during the run, any failed budget / closure / refusal check.

## 5. New parameters and departures (all reported in the summary)

| Item | Value | Note |
|---|---|---|
| `sediment_courant_max` | 1.0 | bound on `v dt / (n_sub dx)` for the transport operator |
| `max_transport_substeps` | 64 | above it the whole step is halved |
| `commit`, `force_final_commit` | True, True | MAPLE triggers per accepted step; final commit at real `t_end` when activity is pending |
| depth ownership | `constant_depth` per run | case file untouched; recorded with the config hash |
| two MAPLE calls per step | pickup, then deposit / export | same-step transport of the actual pickup; ~2 × the per-call cost measured by Codex (0.0185 s zero-demand on Plot 1) |
| hydraulic diagnostics to MAPLE | none | SYRUP diagnostics are in its own outputs (contract G7); avoids extra device reads per call |
| substep choice | one scalar read of `max v` per step | deterministic; FP-edge rejection doubles `n_sub` |
| avalanching in commits | case spec | Plot 1: enabled by MAPLE default; runs inside every commit, recorded (`avalanche_applied` per commit) |

## 6. Observation for Codex (not a defect of this task, a scientific consequence to record)

Under the local-hydraulics policy the deposition timescale is `L / v_s`. For coarse classes in diffuse flow `v_s` is tiny (class 6: ~1e-7 m s⁻¹) while `L` is microscopic (~2.5e-5 m), so `L / v_s` is hundreds of seconds: the class stays "mobile" in its source cell (moving ~0) rather than depositing at once as the legacy instantaneous walk does. Mass is conserved and the sorting test checks the class stays put, but `mobile_final` at the configured end and the "net erosion" diagnostics include this in-place load until the cell dries or a Phase 6 completion policy settles it. Whether the terminal policy or a settle rule for `v_s dt ≪ dx, L ≪ dx` should apply is a Codex/user decision.

## 7. Tests (NOT RUN by the author)

`tests/phase5/test_sediment_event.py` (small real MAPLE beds built with MAPLE's constructors; corrected after Codex's preliminary run): closure of water and every class on a wet/recession valley (diffuse and wet-no-law regimes, never dry) with export; absent class never picked up / mobile, and a 0.1 % class capped by the AVAILABLE active-layer supply (availability shortfall reported, holdings shortfall zero, request reconciled); active-layer refill from the column and per-cell bed change = deposition − pickup = independent inventory difference; sorting on the Codex-verified drying chain (ksat 1e-5, 60 s rain, 180 s: all cells dry, mobile 0, export/pickup decreasing with size, coarse class settles in its cell, morphology = export) and the still-wet variant (coarse class retained mobile, reported, not forced to settle); branching valley with a single junction outlet; `morphology_summary` on synthetic balanced exchange and avalanche contribution; commits every 10 s with rain through the end keep depth, reroute from the committed terrain (independent `build_routing_graph` rebuild equals the event's graph), re-initialise discharge, refresh velocity / outlet discharge / peaks from the post-commit state, and the ledger reset totals equal deposition − pickup; forced final commit only when exchange is pending (no artificial commit on an empty run); failing commit (RoutingGraphError) and failing MAPLE call leave every input array byte-identical and propagate unchanged; control refusals; rainfall integrates exactly over full / partial windows; a constant-v/L continuous-injection stub on a 100-cell chain (longer than the maximum number of crossings, so export is exactly zero) matches the discrete recursion `M_N = R dt s (1 − s^N)/(1 − s)` to 1e-11 and converges first order (ratio > 1.8) to `R/k (1 − e^{−kT})`; sediment Courant handled by substeps, then by halving `dt`, with `max_retries` / retry-floor refusals; hydraulic rejection; cadence-independent true peaks and final state (the forcing end is a row in both cadences); one coupled attempt inspected (adapter label, T(M) seen by the deposit call within an `n_cells ε` summation-order bound, nothing published). Vector tolerances are checked elementwise (NumPy 2.5's `assert_allclose` cannot format an array `atol`).

`tests/phase5/test_sediment_experiment.py` (actual Plot 1, `end_s` 30–180 s, `array` implementation): XML-derived parameters equal the transcribed constants; budgets and closure; morphology and water-exchange fields; override, resolved-config hash (recomputed from the on-disk record), XML and source stability, case identity; outputs (snapshot loads validated, final grids and hydrograph tail consistent with the returned post-commit state, `bed_change_kg` consistent with the committed elevation change); refusals write nothing (including `backend=cupy`); missing Numba; source change; Numba run over the same 180 s coupled window as the array fixture with bed, availability, committed terrain, memory and hydrographs compared (skips without Numba); CLI.

Commands for Codex:

```
cd /home/okin/SYRUP
PY=/home/okin/MAPLE/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 PYTHONPATH=src:/tmp/syrup-numba
$PY -m pytest -q -rs -p no:cacheprovider tests/phase5
$PY -m pytest -q -rs -p no:cacheprovider            # all phases (tests/phase5 is now in testpaths)
$PY -m ruff check src/maple_syrup tests
git diff --check
```

Full storm (`--end-s 5400 --report-every-s 1`) at dt 1, 0.5, 0.25 with `--implementation numba`, plus `--implementation array` at dt 1, are Codex runs; expected diagnostics to compare across dt: export by class, peak mobile / export rate and their times, net erosion / deposition, commit count, `max_sediment_courant`, closure residuals.

## 8. Limitations

- No execution by the author; thresholds in the tests come from hand estimates (see `claude_report.md`).
- Spatial deposition pattern is first order in `dx` (upwind); coarse-class in-place residence (section 6).
- Commit path is host-only; GPU not exercised; the numba hydraulics are CPU-only.
- `evaluate_commit_triggers` validates the full ledger every accepted step and each MAPLE call makes several counted host reads (contract G11): profile before changing cadence.
- Not a MAHLERAN benchmark; no restart; configured end is not event completion.
