# Coupled pond / splash event (`maple_syrup.pond_event`)

Status: implemented and independently exercised by Codex on NumPy, Numba and a real CUDA device. See [qualification](qualification.md) and [acceptance](acceptance.json) for exact source identities, results and limits.

This is the usable, bounded, opt-in event that couples the foundations: pond topology (`pond_network`), the actual
local-inertial hydraulics (`pond_hydraulics`: NumPy / Numba / CUDA), the wet MAHLERAN laws (`sediment_physics`), direct
rain splash (`splash`), the multi-face characteristic transport (`pond_transport`) and MAPLE's native dense bed
(`sediment_bed` APIs, `pond_bed`). It is a **new alternative backend**. The accepted method-5 event, `complete_event`,
`checkpoint`, the legacy replay and every default (no splash) are untouched. Local inertia is an ALTERNATIVE hydraulic
model, not Fortran method 5/6. Coupled face-speed/friction-slope erosion is qualified on the recorded small cases, with face Froude numbers reported and never clipped; this does not establish universal high-Froude or Fortran storm equivalence. Wind is excluded (MASY). Full Chastre is not claimed.

## Files

| file | role |
| --- | --- |
| `src/maple_syrup/pond_event.py` | event object, state, step, evolve, quiet monitor, audit, finalize |
| `src/maple_syrup/pond_bed.py` | native commit adapter without D4 refresh, displacement gate, real-bed D4 slope/aspect, native transaction |
| `src/maple_syrup/sediment_physics.py` | shared rain-law refactor plus two optional slope keywords (below) |
| `tests/pond_event/*` | actual-MAPLE-bed tests (state, forcing, coupled, geometry, completion, native, backends) |

The friction extension also modifies the experimental NumPy/Numba/CUDA modules behind an explicit choice, preserving their previous lagged default. Existing accepted method-5 modules, drivers, workflow defaults and reference sources are unchanged. `sediment_bed.py` was **not** refactored: a
standalone adapter (`pond_bed.native_commit`) composes the same native MAPLE commit APIs `commit_bed` composes
(`evaluate_commit_triggers`, `commit_topography`, availability-aware avalanche callback, water depth callback) and stops
before any graph refresh.

## Public API

```python
event = PondEvent(context, column, sediment, schedule, topology, *, control=None, policy=None,
                  splash=None, vegetation_cover_fraction=None, rain_scale=None)
state = event.initial_state(bed, soil_water_m)            # PondEventState
out   = event.step(state, rain_rate_m_per_s, dt_s)        # PondCoupledStep(state, record): ONE accepted coupled step
res   = event.evolve(state, end_s, stop_when_quiet=False) # PondEvolveResult: forcing-split, bounded adaptive retry
ok, v = event.quiet_metrics(state)                         # instantaneous quiet gates + monitor
aud   = event.audit(state)                                 # water / sediment / geometry audit (host dict)
done  = event.finalize(state)                              # PondDryHandoff, or PondEventIncomplete
done  = event.complete(state, max_end_s)                   # evolve until quiet, then finalize, else PondEventIncomplete
```

All methods are pure: states are immutable, every MAPLE call returns new objects, and an exception (including a late
hydraulic, transport or geometry-refresh failure) leaves every input unchanged. `evolve` is the only place with a loop over
time; no loop runs over cells and no per-step full-state history exists.

### Reuse (imported, not copied)

`PondTopology` / `update_pond_topology`, `pond_hydraulics` (`step_once`, `face_speed_diagnostics`, `PondLedger`,
`rebind_pond`, NumPy / Numba / CUDA solvers), `pond_transport` (`FaceNetwork`, `pond_transport_step`), `PhaseState`,
`sediment_physics_step` and `dry_raindrop_pickup_demand`, `splash.apply_splash_exchange` / `splash_grid` (array and explicit
CUDA sweeps), `sediment_bed` (`with_water`, `apply_bed_demand`, `bed_inventory`, `current_routing_surface_m`,
`net_routing_displacement_m`, `reroute_threshold_m`, `bed_already_committed`), `sediment_event._reconcile_phase_inputs`
(the existing reconcile rules), `RainfallSchedule`, `ExperimentalControl`, `conservation.reservoir_bound_kg` and
`volume_roundoff_bound_m3`. The MAPLE paired windowed transaction is windowed-only, so the dense two-call native
transaction (`pond_bed.pickup_transport_deposit`) is used.

## Accepted coupled step

1. `step_once` (donor limiter off) on the held topology.
2. Wet laws on the post-hydraulic depth `h1` and the erosion velocity (outflow-volume-weighted outward cell speed of the
   stage-consistent face velocities); holdings of the start of the step.
3. Dry splash (only with `splash` given): cells with rain > 0 and `h1 == 0` exactly. Native `apply_splash_exchange` of the
   ACTUAL pickup, explicit per-class inverse distances, real-bed slope/aspect, exact outlet face masks, holes and closed
   faces closed. The wet law owns every `h1 > 0` cell, so nothing is picked up twice. The reference switches on the
   pre-hydrology depth; this module uses the post-hydrology depth (documented difference).
4. Wet transaction: native pickup (supply-limited) -> `pond_transport_step` on the PRE-pickup pool, its phase and the ACTUAL
   pickup with the signed actual face volumes -> native deposit / export. The phase is reconciled to the ACTUAL pool by the
   existing `reconcile_phase` bound check. After a splash exchange the phase is only canonicalised where the pool is
   exactly 0 (no artificial source, no erasure).
5. Geometry gate + native accounting commit (next section).
6. Ledgers, quiet monitor, counters, new state.

Transport substeps come from the sediment Courant number of the cells with outflow; a request above
`max_transport_substeps` is a `TransportStepRejected`, like a native `HydraulicStepRejected` it is retried by `evolve`
with a halved dt from the unchanged state, bounded by `max_retries`, `min_dt_s` and `max_steps`; a bound stop is
INCOMPLETE (`stop_reason`), never silently continued or reset. Rain schedule discontinuities are split exactly with
`RainfallSchedule.next_edge_s` / `rate_after_m_per_s`; rain, intake, saturation return and drain age the physical dt.

## Three slopes, never conflated

* **Topographic** slope/aspect (wet raindrop law, splash rays): the HELD real bed. `pond_bed.steepest_d4_descent` takes the
  steepest strictly downward D4 neighbour among active cells and across explicit outlet faces; flat -> `S = 0`, aspect 1.
  The metadata graph of the hydraulics is never a terrain authority. By the Quansah slope law `(100 S)^c`, flats give
  exactly zero raindrop detachment (wet and dry).
* **Shear**: the legacy `u* = sqrt(g h S)` is a uniform-flow estimate. With `shear_slope="friction"` (default) the event
  supplies `S_f = f u^2 / (8 g h)` (h > 0, else 0) so that `u* = sqrt(f/8) u` (Darcy-Weisbach); `f` is the topology
  friction. `"topographic"` keeps the legacy estimate. Implemented by the optional keywords `shear_slope` and
  `stream_power_slope` of `sediment_physics_step` (defaults `None` = the original arithmetic, unchanged; explicitly passing
  `grid.slope` reproduces the default exactly, tested).
* **Stream power** `rho g h v S`: `stream_power_slope="friction"` (default, `rho f u^3 / 8`) or `"topographic"` (zero on
  flats). No floor, no minimum slope. Froude diagnostics (`max_face_froude`, number above `froude_reference`) are
  accumulated and not clipped. If `h` is so small that `S_f` overflows FP64, the physics refuses (non-finite) rather than
  clipping.

## Geometry gate (net elevation only)

Every accepted step, before any commit, `gate = max over ACTIVE cells |actual current MAPLE elevation - held source|`
(`current_routing_surface_m`, one counted scalar read, device reductions, per-cell signed, different cells never cancel).
`gate >= pending_elevation_trigger_m` (MAPLE's value, default 0.00075 m, `>=`) forces a genuine native commit (reason
`net_elevation`, if MAPLE's own triggers have not already fired) and THEN refreshes the topology from the committed
elevation: source = stable initial datum + actual committed delta on active cells, ring and mask fixed, generation + 1,
exactly one full-surface transfer. `qx, qy`, depth, soil, phase, mobile and the transport memory are carried unchanged
(`rebind_pond` validates them against the refreshed faces). MAPLE-triggered commits and the forced final commit never
reroute by themselves; time, report, checkpoint, force and reset never refresh. A refresh that the topology refuses (for
example an outlet that no longer drops to its ring) raises before anything is published.

Cost note: `current_routing_surface_m` on a dense bed is MAPLE's own full diagnostic each step (the small dense path only).

## Ledgers and audit

Cumulative accepted-step ledgers only (arrays, not history): rain, intake, saturation return, drainage, export, gross and
net signed face volumes (water) and face masses (sediment, `(ny, nx+1, nc)`), per-cell pickup and deposition, per-class
requested / actual / deposited / exported (wet and splash separately), unmet deposition, phase reconciliation residual
and counts. `audit` evaluates the water closure with the existing `PondLedger` closure (`volume_roundoff_bound_m3`,
per-cell and global) and the per-class sediment closure against MAPLE's own inventory (`bed_inventory` + mobile) with
`reservoir_bound_kg`, plus the mobile-pool identity. No absolute floor, no clipping.

## Completion

Quiet = rain finished (`t >= schedule.end_s`) and, for `hold_s` of continuous accepted steps, the per-step max unit flux,
outlet discharge and (conservatively summed) max depth change under the `PondCompletionPolicy` limits (defaults mirror
`CompletionPolicy`), no cell above its static escape level (closed components store at any level; escape levels are
diagnostics, not redistribution), and mobile load below MAPLE's significance threshold unless the explicit
`deposit_mobile_at_flow_end`. Soil may keep draining. A slowly draining pond is not quiet and is not labelled terminal.
`finalize` re-checks the gates, closes the budgets, settles the pool through a genuine MAPLE deposition (refused if
anything stays mobile), makes the forced final commit (gate still applies), checks the MAPLE ledger is empty, performs the
explicit dry reset and records the removed surface AND soil water. Anything unmet raises `PondEventIncomplete`.

## State and checkpoint design

`PondEventState` is frozen and holds only arrays and host records: hydraulic state (with the adaptive cap), `BedState`,
transport memory, phase fraction/position arrays, the held `PondTopology` (host only), the held source array, the stable
`PondDatum`, water and sediment ledgers, counters (bounded commit log), quiet monitor, `config_sha256` and
`forcing_sha256`. No solver, namespace module or callable is a field (tested). The solver, `PhaseState`, physics/splash
grids and the face network are a runtime rebuilt from the state by `PondEvent._runtime`; a later checkpoint module can
serialise the state and rebuild with the same configuration (the event refuses a state with another config/forcing digest
and re-checks that the held source reproduces the topology source exactly).

## C1 corrections (root review of source `event_runtime_r0`)

* **Public guards.** A state object this event built is "blessed" (weak set) and takes the cheap path (digests + cached
  runtime). Every other state object (restart, `dataclasses.replace`, hand-built) gets ONE full validation: the held source
  must reproduce the topology source EXACTLY from the frozen datum (one host transfer of that source object, never per
  step); metadata / namespace / shape / dtype of bed water, mobile, memory, phase; ledger clock == hydraulic clock; bed depth
  == solver depth; inactive cells hold no water, soil or mobile mass (required at initialisation, never silently deleted);
  `validate_phase`. Datum and held-source host arrays are frozen. No mass is repaired. `step` / `evolve` refuse a
  `dry` (completed) state before anything physical; `audit` / `quiet_metrics` accept it.
* **Config binding.** The constructor requires the sediment diameters (values AND order) and the particle density to equal
  the MAPLE grain classes (as the old `_check_event_inputs`); `config_sha256` binds class ids / order / diameters /
  densities / aggregate flags, the full geometry repr (dx, dy, dz, bulk density, active layer, boundaries), commit and
  avalanche specs, depth rule, mass resolution and the initial availability array. The native context source/version
  identity must additionally be locked by the checkpoint author.
* **Splash face ledger.** Cumulative x/y gross/net sediment faces now include the actual splash crossings. Each step checks
  the cell/class mobile balance `dM = P_wet + P_splash - div_wet - div_splash - Dep_wet - Dep_splash` (actual native pickup and
  deposition, full face divergence including exports) against a MAPLE `summation_error_bound_kg` with the transport +
  reconciliation operation count; a violation raises before publication. `record.balance` carries max residual / max bound;
  `record.numerical_residuals` the four native residuals (pickup, deposit, splash pickup, splash deposit), diagnostics only.
* **Completion.** The raw `n_cells_above_escape` is diagnostic only. The criterion is the EXCESS depth
  `clip(eta - escape, 0, h)` (0 on closed components): max under `max_depth_m = 1e-8` AND volume under
  `surface_volume_m3 = 1e-6` (existing `CompletionPolicy` defaults, physical thresholds) plus flux / outlet / depth-change /
  hold / rain-off. A tiny asymptotic film at an outlet can complete; moving over-crest water cannot; ponded water below the
  crest is retained and accounted until the explicit reset. The physical depth is never altered.
* **Performance.** `_water_add` adds directly (identical operand order to `PondLedger.add`; `PondLedger` is built only for
  the audit closure); `initial_state` reuses the cached prepared solver.
* **Friction treatment.** The lagged local-inertial friction (Fr 185 at first wetting, root finding) is a hydraulics-component
  issue: a separate author adds `HydraulicControl.friction_treatment`. This event's hydraulic factory is unchanged in C1; the
  root wires the new pond default after that component is tested. Fr / u are not clipped.

## C6: raw spatial reductions (supersedes the C4 cell-reduction factor)

Initial/final mobile class totals use a raw `xp.sum(axis=(0,1))` over a class-last array. That API does not guarantee the explicit pairwise helper's tree order. The audit therefore uses MAPLE's `sequential_bound_scale_factor(n_cells)` for these spatial reductions (also a valid upper bound for native pairwise sums). Splash/transport component policies keep their actual explicit pairwise factors. A `math.fsum` reference tests a dominant value followed by tiny positive terms. This changes only the diagnostic bound, preserving physical arithmetic, state, clock, counters and checkpoint format.

## C5: realised substep counter (supersedes the C4 wet-term cap)

`PondCounters.max_transport_substeps_used` (host int, initial 0) records `max(existing, n_sub)` of every ACCEPTED step: no
device transfer, history or array. The wet transport term of the mobile-identity bound now uses
`operation_count(max(used, 1), bins)`: the realised maximum is an operation-count upper bound for every accepted step, not the
configured cap (1 on the long row, so the bound stays tight; the 1e-6 kg perturbation tests are restored). Validation:
nonnegative int, `<= control.max_transport_substeps`, 0 iff no accepted step. The state gains this one diagnostic counter; all
other fields, arrays and clocks are unchanged. `pond_checkpoint.py` (read only) discovers int/float `PondCounters` fields from the
dataclass defaults, so the field is encoded and reloaded generically (fresh restarts under this source; historical R3 files
lack the field and are not a reason to loosen the bound).

## C4: cumulative class mobile-identity bound (diagnostic only)

Note on `commit_bed`: it conditionally handles the D4 routing-graph refresh / validation (the refresh is gated on the net
displacement when a routing source is given); the pond adapter `pond_bed.native_commit` omits all graph handling entirely.

`audit()["sediment"]` checks `mobile - (mobile0 + picked - deposited - export) = 0` per class. The previous bound used only
the call count and the class reservoir scale, so the legitimate long-row splash case (closed row, n = 4096, spd 1e-8; mobile
exactly 0, residuals up to 7.5e-15 kg vs bound 7.1e-16) was false-rejected: the class vectors are cell reductions, the ledger
adds them over steps, and the splash component's own class budget carries the large cancelling ray-face operands. The bound
(`PondEvent._mobile_identity_bound`, MAPLE policies, genuine operands, no multiplier/floor/resolution) is now the sum of
(1) cell reductions `pairwise_bound_scale_factor(ny*nx) * (mobile0 + mobile + picked + deposited + export)`; (2) ledger
accumulation `sequential_bound_scale_factor(n_steps + 2) * (picked + deposited + export)`; (3) native bookkeeping
`reservoir_bound_kg(n_calls, ...)` (kept); (4) the splash class-budget policy
(`summation_error_bound_kg(_CLOSURE_OPS, sum_i S_i) + 2 pw (pickup + dep + exp) + 8 eps pickup`) with `S_i` the cumulative
per-cell scale `pickup_cell + deposition_cell + four cumulative gross faces` (`sum_k max_i s_ki <= sum_i S_i`); (5) the wet
transport class-budget policy with `n_sub <= max_transport_substeps`, pool and gross-face operands from the ledger. All
terms are zero for zero inputs. The independent global class bed closure is unchanged and separate. NOT done: storing the
realised substep counts or per-step component tolerances (would broaden `SedimentLedger`/counters; offered to the root).

## C3: composite balance bound (diagnostic only; no physical change)

The C1 per-step balance used `summation_error_bound_kg(operation_count + 2*PHASE_OPS, max(endpoints, |div|))`. It ignored the
large individual face operands that cancel in the net divergence and did not carry the splash component's own bound, so a
valid long-row splash (n = 4096, spd 1e-8) was false-rejected (root: 984 cell/classes, max 2.3e-17 kg). `composite_mobile_balance`
(pure helper, reusable with a bare `SplashResult`) now bounds the residual by the SUM of: (1) the splash component's validated
`cell_balance_tolerance_kg`; (2) the transport identity recomputed as the operator checks it; (3) face formation at the largest
INDIVIDUAL gross/net face operand (3 flops per stage, plus `10 n_sub` actual accumulation adds for the wet stage); (4) the 7
combination flops and the native pickup/deposit/reconcile roundings (`2*PHASE_RECONCILIATION_OPERATIONS`) at the largest cell
operand. Each term is MAPLE `summation_error_bound_kg(n_i, scale_i)`; no multiplier, floor, resolution or percentage. Zero
operands give a zero bound (a 1e-300 kg imbalance is refused) and a 1e-6 kg imbalance is refused. `record.balance` reports
the component maxima. State, physics and budgets are unchanged. Tests: `tests/pond_event/test_composite_balance.py`.

## C2: friction default (NEW pond mode, documented)

`PondEventControl.hydraulic` now defaults to `HydraulicControl(friction_treatment="implicit_quadratic")` (the lagged update
gave face Fr ~185 at first wetting). The global `HydraulicControl()` stays `"lagged"`; `PondEventControl(hydraulic=
HydraulicControl())` selects the original qualified lagged variant explicitly. The treatment is part of `config_sha256`.
Fr / u are not clipped; the wet and splash laws and the method-5 / legacy defaults are unchanged. Test helper `move_mass`
now performs a genuine native exchange (pickup, lateral relocation of the removed mass in the mobile pool with the matching
actual D4 face gross/net, native deposit) and asserts deposit == removal, mobile pool preserved and the real surface change.

## Explicit limits

* Dense is the default. [Windowed native beds](../pond_windowed/README.md) are explicitly supported by `bed_storage="windowed"`; enabled avalanching, a non-`constant_depth` rule and exchange history remain refused.
* Splash/transport are instantaneous redistribution and array transport; Numba is hydraulics only; the CUDA path runs
  hydraulics, physics, transport and the explicit CUDA splash sweeps on the device.
* The legacy graph builder needs friction >= 0.1 (inherited from the metadata graph of `pond_hydraulics`).
* `evolve` checks the retry bounds per step and the step bound per call; a restart is a new call from the returned state.
* The depth-change quiet criterion is a conservative sum of per-step maxima, not a per-cell net history.
* Disk checkpoint/continuation is implemented separately in [pond_checkpoint](pond_checkpoint.md). Not implemented: avalanching, external sediment inflow,
  periodic boundaries, stage-controlled outlets, wind.

## Tests (`tests/pond_event`)

State/API purity and refusals; exact rain-knot split and ledger integral; bounded hydraulic/transport retry and bound stops;
actual dry-splash nonzero path against the independent raindrop formula; wet pickup with actual < requested and class
closure; per-step vs accumulated ledger; native multi-face deposition / overflow export / rejection non-mutation on an actual
bed; gate (below threshold no refresh, crossing before the time commit, net cancellation, different cells, `>=`, source
and flux carry, time never refreshes); quiet completion, safe dry handoff, refusals while flowing / over the crest /
with significant mobile load / slowly draining; real-bed slope and friction-slope shear; NumPy/Numba/CuPy parity and the
explicit CUDA splash (skipped honestly without a device).
