# Phase 5a — wet detachment, travel and conservative mobile transport

Status: implemented 2026-09-30 for task `phase5a_physics`, corrected the same day after Codex's preliminary review (`agent_handoffs/tasks/phase5a_physics/codex_preliminary_review.md`, `correction_report.md`). **Not executed by the author** (no shell in this task; Codex runs the tests). Claude authored `src/maple_syrup/sediment_physics.py`, `src/maple_syrup/sediment_transport.py`, `tests/phase5/` and this document. This is the physics/transport core only: no bed exchange, runner, terrain refresh, restart, dry reset or GPU run is claimed. Every "legacy" statement below cites MAHLERAN 1.2.3 source lines as read on 2026-09-30; every numerical expectation is a test, not a result.

Files:

- `src/maple_syrup/sediment_physics.py` — laws (rain energy, raindrop and flow detachment, diffuse / concentrated / suspended transport, d50, suspension criterion, regime selection, recession memory) → pickup **demand**, virtual velocity, deposition rate, settle mask.
- `src/maple_syrup/sediment_transport.py` — the lateral operator `T` of the Phase 1 contract (§4.1): class-resolved Eulerian pools on the `RoutingGraph`, exact exponential deposition hazard `v/L` with conservative upwind advection (Strang splitting), deposition and export **requests**, face flux and budgets, plus `water_demand_from_transport` for `apply_water_process_demand`.
- `tests/phase5/test_sediment_physics.py`, `tests/phase5/test_sediment_transport.py`.

Everything is SI: kg, m, s. Legacy mm, mm/s, g cm⁻³ and cm conversions are applied once at the parameter/input boundary and named in the code.

## 1. Where the legacy laws sit and what is kept

Per wet cell, `route_sediment_xml.f90` 62-176 selects the regime from the depth `d(1)`, the velocity `v`, the rainfall `r2` and the Reynolds number `Re = v d / ν` (95):

| Condition (legacy lines) | Detachment | Transport | SYRUP regime code |
|---|---|---|---|
| dry, rain (180-182) | raindrop + **splash** | — | 0 `dry`: **deferred**, no demand, pool settles |
| wet, no rain, Re ≤ 500 (161-170) | none | none (pool advects at the decayed `v_soil`) | 1 `wet_no_law` |
| wet, rain, Re ≤ 500 (123-144) | raindrop (128) | diffuse (142-144) | 2 `diffuse` |
| wet, rain, 500 < Re < 2500 (132-141) | raindrop + flow | diffuse | 3 `transitional_rain` |
| wet, no rain, 500 < Re < 2500 (156-160) | flow | concentrated | 4 `transitional_dry` |
| wet, Re ≥ 2500 (98-120) | flow (100) | per class: suspended if `u* ≥ crit` (112-114) else concentrated (117) | 5 `concentrated` / 6 `suspended` |

Rain-assisted wet detachment (regimes 2 and 3) is preserved; direct dry splash redistribution (`splash_transport`) is deferred by user direction, so a dry rained-on cell produces no demand and its regime is reported as `dry`.

Hydraulic variables (SI; legacy mm forms in parentheses): `u* = sqrt(g d S)` (89: `sqrt(9.81e-3 d_mm S)`), `Re = v d / ν` with ν = 1.003e-6 m² s⁻¹ (`shared_data.f90` 180), stream power `ω = ρ_w g d v S` W m⁻² (diffuse 31-32, conc 42-43, susp 22-23: `9.81e-3 d_mm v_mm/s S`). Slope is the routing graph's final (post-edge-rule) slope.

### 1.1 Composition and d50 from the current bed

`route_sediment_xml.f90` 66-85 builds `d50` from `sed_propn`, which the legacy never updates (the renormalisation in `update_sediment_flow.for` 43-48 is commented out). SYRUP takes the class fractions from the **current** MAPLE active layer, `active_layer.mass_kg / Σ_classes`, in `median_diameter_m`: cumulative fraction over classes fine→coarse, first class where it reaches 0.5; `d50 = D₁/(2 dsum)` for class 1, otherwise linear interpolation between `D(φ−1)` and `D(φ)`; `D₆` when 0.5 is never reached (including an empty cell). The legacy `dsumlast` is the exact previous cumulative sum, reproduced with a shifted cumsum. Holdings whose per-cell total overflows FP64 are refused, never turned into empty fractions.

### 1.2 Raindrop detachment (`raindrop_detachment.for`)

Rain kinetic energy `KE` in J m⁻² mm⁻¹ (24-54): model 1 `11.9 + 8.73 log10(I_mm/h)`, model 2 `29 − 20.88 exp(−0.05 I_mm/h)` (= `exp(−180 r2_mm/s)`), with the vegetation factor `(1 − 8.1e-3 · veg%)`. Plot 1 uses model 2 (`KE_model_type` 2). Vegetation cover enters as a fraction 0–1 (Phase 2 sidecar `vegetation_cover_fraction`, file is percent) and is multiplied by 100.

**Vegetation precedence in model 2 (source relationship kept).** The Fortran expression (33-34) is `29 − 20.88 · exp(−180 r2) · (1 − 8.1e-3 veg)`: only the subtracted exponential term is scaled, so vegetation cover *raises* the energy (bare 100 mm/h: 27.7; 80 % cover: 28.5 J m⁻² mm⁻¹). This is counter-intuitive but it is the literal MAHLERAN relationship, and `ke_vegetation_form="legacy_literal"` is the **default** because the matched Plot 1 comparison must use the source as written. `"intended"` scales the whole energy as model 1 does; it is an explicit scientific *variation*, not a verified correction of the empirical relationship, and is never selected silently. Model 1 is unambiguous (factor on the whole energy in both forms).

Detached mass per unit area over the reference interval (69-72 with the `spa/1200` pre-scaling of `initialize_values_xml.f90` 268):

    X = (a/1200) · (1200 · KE · r2_mm/s)^b · (100 S)^c      [kg m⁻²]   (1200 KE r2 = KE I_mm/h / 3, Quansah's scaling)

then `2X / ρ` (82-85: ×2 because the source data are up/downslope halves; ÷density converts to a depth, legacy mm), attenuated by `exp(−spq · d_cm)` under standing water (87-95, `spq` = `shared_data.f90` 178), clipped at 0 (96-98), capped for class 2 only at `f₂ · hs` (103-109; `hs` = 1000 mm in Plot 1, so it never binds), zeroed where the class is absent (110-112); the gravel feedback factor is `exp(0) = 1` (117-120 with `grav_propn = 0`, route_sediment 59).

**Time basis (resolved).** The legacy divides by `dt` (85) and the routing update multiplies by `dt` again (`route_sediment_xml.f90` 217-219, 284-288), so the same depth `2X/ρ` is detached per **step** whatever `dt` is: at dt = 0.5 s the imposed rate doubles. SYRUP declares the legacy reference interval `t_ref = 1 s` (`time_step` 1.0 in the root XML, `reference_interval_s`) and uses the physical rate `2X/(ρ t_ref)`; demand over a step is rate × area × ρ × dt (kg). A refined timestep changes nothing (tested: `4 × demand(0.25 s) = demand(1 s)` exactly). The same treatment applies to the flow law and the `hs` caps.

The raindrop law is **not** scaled by the class fraction in the legacy: the per-class `a, b, c` were calibrated on a given soil. With MAPLE's evolving composition a nearly depleted class keeps its full demand until its holdings run out (MAPLE then reports a holdings shortfall): supply limitation, absence (zero demand for an absent class) and d50 all follow the *current* holdings. `raindrop_composition_scaling="legacy_none"` is the default (Codex resolution); `"fraction"` is a tested explicit alternative, not a calibration.

### 1.3 Flow detachment (`flow_detachment.for` 18-47)

    θ = u*² / (σ g D),  σ = 1.65 (relative excess density, initialize_values 211)
    p_const = ln(0.049 / (0.25 θ)),  p = 0.5 − 0.5 sign(p_const) sqrt(1 − exp(−(2/π)(p_const/0.702)²))
    depth rate = p · hz · f / t_ref,  hz = active_layer_sensitivity = 1.52e-6 mm (→ 1.52e-9 m)

capped at `f · hs / t_ref` for every class, zeroed where absent. `p` is a pickup probability in [0, 1] (tested: 0.5 at θ = 0.196 where the legacy has 0/0, → 0 as θ → 0 without evaluating log(∞), → 1 at high shear, decreasing with grain size). `hz` is a detachment coefficient with the dimension of a depth per reference interval, not an active-layer thickness. The diagnostic `legacy_cap_applied` is true only where the cap bound the law actually selected for that cell (raindrop cells for the phi-2 cap, flow-detachment cells for the flow cap).

**Density and elevation (morphology departure).** Legacy detachment "depths" are solid-volume depths converted with the **particle** density (2.65 g cm⁻³: `uc = dx·density·1e-6·dt`, route_sediment 209; flow_distrib 34-35). The demand returned here is a solid mass computed with the particle density 2650 kg m⁻³, so mass is exact; MAPLE then converts mass to elevation with its **bulk** density (1250 kg m⁻³ in the Plot 1 case, MAPLE-owned), so an equal solid mass produces 2650/1250 = 2.12× the legacy `z_change`. Phase 5b must use MAPLE's bulk density for elevation and report this morphology departure with every bed-change comparison.

### 1.4 Transport laws

All three return a virtual velocity `v_s` (m s⁻¹, capped at the water velocity where the legacy caps it) and an exponential mean travel distance `L` (m), delivered as `deposition_rate_per_m = 1/L`; the transport operator turns this into the per-second hazard `v_s / L`.

- **Diffuse** (`diffuse_flow_transport.for` 26-70; Parsons et al. 1998 reanalysis): rain energy flux `KE_f = (11.9 + 8.73 log10 I) · r2_mm/s` J m⁻² s⁻¹ (model-1 form regardless of `KE_model_type`, no vegetation, 26-27); particle mass `m = ρ_g·1e6·(4/3)π r³` grams (40-41); `v_s = 0.525 KE_f^2.35 ω^0.981 / m` cm min⁻¹ → m s⁻¹; `L = 0.05 KE_f^1.85 ω^0.481 m^−0.425` m (60-61, used directly as the mean); `v_s ≤ v` (68-70).
- **Concentrated** (`conc_flow_transport.for` 26-81): Bagnold threshold `ω₀ = 4.554e-3 (Δρ d50)^1.5 log10(12 d / d50)` clipped at 0 (26-35; Δρ = 1650 kg m⁻³), excess `ω − ω₀`; if ≤ 0 the detached mass is redeposited at once (49-54) → SYRUP **settle**; `L = 2.85e-3 (ω−ω₀)^1.31 D^−0.94` m (Hassan et al. 1991 eq. 2b, mean), `× 0.693` (63), capped at 30 m (64-68); `v_s = 1.92e-2 (ω−ω₀)^1.01` m h⁻¹ → m s⁻¹ (74-78), `≤ v` (79-81).
- **Suspended** (`suspended_transport.for` 22-56): `L = 727.518 exp(min(7.331976e-3 ω, 100)) exp(−6.1268 D_mm)` m `× 0.693 × (Δρ/1650)^−0.5` (28-50); `v_s = v` (54).
- **Suspension criterion** (route_sediment 102-118, initialize_values 219, 298-302): `D* = D ((σ−1) g / ν²)^{1/3}`, settling `w_s` Stokes below 0.1 mm else `1.1 sqrt(σ g D)`, `crit = 4 w_s / D*` for `D* ≤ 10` else `0.4 w_s`; suspended if `u* ≥ crit`.

### 1.5 Recession memory (explicit legacy behaviour)

`update_sediment_flow.for` 69 multiplies `v_soil` by 0.9 once per (1 s) step so movement "declines slowly rather than abruptly" after the laws stop applying. SYRUP keeps `sediment_velocity_m_s` `(ny, nx, nc)` as transport memory (contract §3.1): where a law applies this step it is the law's value, otherwise `v_prev · exp(−dt/τ)` with `τ = −t_ref / ln 0.9 = 9.49 s` (tested: two 0.5 s steps equal one 1 s step equal ×0.9). Settled cells (dry, no capacity) have velocity 0.

The decayed memory is **not capped at the current water velocity** (the legacy caps only inside the transport laws), so when the water slows faster than 10 %/s the sediment memory exceeds the water velocity (tested). Consequently a water step accepted under its Courant limit does **not** imply the sediment Courant condition: the water check uses the old flux, the sediment uses the new memory. `transport_step` validates its own Courant number and rejects with `TransportStepRejected`; the caller must retry the whole unpublished coupled step (smaller dt or more sediment substeps), never publish a partially applied state.

## 2. Departures, quirks and documented boundaries

| Item | Legacy | SYRUP | Switch / evidence |
|---|---|---|---|
| Per-step pickup | `X/dt` then `×dt`: fixed depth per step | physical rate on `t_ref = 1 s` | `reference_interval_s`; `test_pickup_is_a_rate_over_the_reference_interval_not_per_step` |
| Low rain energy | `11.9 + 8.73 log10 I < 0` below 0.04334 mm/h → negative base to a real power (NaN/crash) | energy floored at 0: no rain detachment, no diffuse capacity (pool settles) | boundary; `test_low_rain_energy_is_floored_at_zero_without_nan` |
| KE model 2 with vegetation | Fortran precedence (33-34): cover **raises** energy | `"legacy_literal"` (default) reproduces it; `"intended"` is an explicit variation | `ke_vegetation_form`; `test_vegetation_forms_...` |
| Median / mean | mean ×0.693 called "median", then used as the exponential **mean** (conc 63, susp 44, flow_distrib 27) | `"legacy_literal"` (default) keeps it; `"formula_mean"` uses the formula as the mean; exact ln 2 helpers provided, no hidden conversion | `distance_convention`; `test_concentrated_conventions_...` |
| D* | `(sigma − 1)` with sigma already 1.65 → 0.65 | literal (default) or `"van_rijn"` (1.65) | `dstar_convention` |
| Bagnold `log10(12 d/d50)` | d in mm, d50 in m | literal (default) or `"si_m"` | `bagnold_depth_units` |
| 0/0 at θ = 0.196; log(∞) at θ = 0 | NaN | 0.5; 0 | `test_pickup_probability_limits_and_sign_singularity` |
| Composition | static `sed_propn` | current active-layer fractions | `test_d50_matches_legacy_transcription_including_edges` |
| Raindrop fraction scaling | none | none (default) or `"fraction"` | `raindrop_composition_scaling` |
| Recession ×0.9 per step | dt-dependent; uncapped | `exp(−dt/τ)`; uncapped (kept), sediment Courant validated separately | `test_zero_rain_wet_low_reynolds_...`, `test_recession_velocity_is_not_capped_...` |
| Overflow | undefined | holdings total, every intermediate and every diagnostic must be finite; refused | `test_overflowing_inputs_are_refused_not_masked` |
| Time levels | `d(1)` old depth with new `v` | one consistent (depth, velocity) supplied by the caller (recommended: the routed step's `depth_m`, `velocity_m_s`) | one-step difference, vanishing with dt; the Phase 5b timestep study checks it |
| Dry rained-on cell | raindrop + splash | no demand (splash deferred), pool settles | scope |
| Deposition timing / location | instantaneous along the walk (flow_distrib), pool never deposits, tails beyond the ring / `nmax` / 1e-19 dropped, `d_soil < 0` clipped, dry `d_soil` zeroed | continuum hazard `v/L` at the local hydraulics, conservative, nothing dropped or clipped, dry pools settle via request | §3 |
| Density for elevation | particle 2650 | solid mass demand at 2650; MAPLE bulk 1250 for elevation (2.12× `z_change`) | §1.3, Phase 5b |

## 3. Transport operator (`sediment_transport.py`)

Reference physics (`flow_distrib.for` 44-46, 102-116; `route_sediment_xml.f90` 211-299): an exponential travel distance of mean `L` covered at the virtual velocity `v` is a deposition hazard `v/L` per second, mean lifetime `L/v`, mean distance `L`. The continuum law on the D4 graph is `dW/dt = −(v/L) W − div(v W)`. The legacy discretizes its own way (instant walk at pickup, non-depositing pool); SYRUP discretizes the continuum law directly.

Scheme, per substep `dt` with `v`, `r = 1/L` and `settle_mask` fixed (Strang splitting):

    settle:  Dep += W, W = 0                      where settle_mask (dry / no capacity)
    react:   Dep += W (1 − s_h), W *= s_h,  s_h = exp(−v r dt / 2)      (exact)
    advect:  a = v dt / dx ≤ courant_max ≤ 1;  cross = a W → receiver, or export request E at an outlet
             W' = (1 − a) W + Σ_{j → i} cross_j                          (conservative upwind)
    react:   Dep += W' (1 − s_h), W' *= s_h                             (arrivals decay at their new cell's hazard)
    settle:  Dep += W', W' = 0                    where settle_mask (dry arrivals of the substep)
    T(M) = W' + Dep + E                           (requests stay in the pool of their cell)

The earlier "deposition at crossing completion" scheme was withdrawn: it reproduced the legacy cell bins exactly at unit Courant but its hazard was `(v/dx)(1 − exp(−dx/L))`, so for `L ≪ dx` deposition was delayed from `L/v` to `dx/[v(1 − exp(−dx/L))]` whatever the timestep (Codex's reproducer: 0.82 kg left after 100 s where the physics leaves `exp(−10)` kg).

Properties (all tested in `tests/phase5/test_sediment_transport.py`):

- **Timescale exact.** With uniform `v r` and no boundary export (reaction only: an outlet exports mid-step in addition, so the in-domain total then has an extra loss), each substep multiplies the in-domain pool by exactly `exp(−v r dt)` for any Courant number (`test_uniform_hazard_total_survival_is_exact_for_any_courant_number`, a = 1, 0.5, 0.2, 0.002). Codex's reproducer retains `exp(−10)` kg after 100 s of 1 s steps and the mean deposition time is `L/v` to within the exact-sampling term `k dt²/12` (`test_short_travel_distance_deposits_on_the_physical_timescale`).
- **Advective timing.** Without deposition the mean arrival time along a chain is the physical `n dx / v` (geometric residence per cell), spread `n(1−a)/a² dt²` (upwind numerical diffusion), zero at `a = 1` (`test_mean_arrival_time_without_deposition_...`).
- **Generator reference.** On a branching valley with random per-cell `v`, `r`, the step converges to the dense continuous-time generator integrated with `scipy.linalg.expm` as `n_substeps` doubles, monotonically and first order (explicit upwind advection; the Strang reaction split alone would be second order) (`test_step_converges_to_the_continuous_time_generator_with_substeps`).
- **One-substep matrix.** `T = S A S + (I − S) + (I − S) A S + E₀ S` assembled independently equals the operator column by column on valley and random networks (`test_operator_matches_explicit_matrix_on_branching_network`).
- **Spatial pattern is a discretization, not an exact bin.** With the jump-count distribution `P(J ≥ j) = q (s_h q)^{j−1}`, `q = a s_h / (1 − (1 − a) s_h²)`, the cumulative deposition within 1 m of the source converges to `1 − exp(−1 m/L)` first order in `dx` (errors ≈ 0.11, 0.053, 0.027, 0.013 at dx = 0.5 … 0.0625 m, `test_deposition_pattern_converges_to_exponential_in_distance_with_grid_refinement`). Exact coarse-cell bins are **not** claimed. For `L ≪ dx` mass deposits in its source cell except the upwind leak `q` (0.0197 for the reproducer; tested bound).
- **Conservation, positivity, requests.** Every operation multiplies by a factor in [0, 1] or sums such products: `W ≥ 0`, `Dep + E ≤ T(M)` per cell, `Σ T(M) = Σ M` per class within a declared pairwise FP64 bound `32 ε (n_sub+1)(ΣM + ΣIn + ΣOut) + 2·pairwise(n)(ΣM + ΣT)` (residual and bound returned), `T − M = In − Out` per cell within `32 ε (n_sub+1)(M + In + Out)`.
- **Junctions, outlets, faces.** Two donors into one receiver; export only from outlets; face crossings in MAPLE layout (`x` faces `(ny, nx+1, nc)` indexed (row, face column), +east; `y` faces `(ny+1, nx, nc)` indexed (face row, column), +north) including the boundary crossing at outlets, so `T − M = face divergence + E`. Tested on a south-draining valley, a north-draining 3×5 grid and 8×1 / 2×1 chains (the first version indexed y faces as (cell row, face row) and failed on any non-square grid).
- **Dry / no-capacity cells** request their whole pool, including arrivals of the same substep, and send nothing; an all-dry domain settles everything without erasing. Mobile mass on an inactive cell is refused.
- **Substeps.** `n_substeps` internal equal substeps with fixed `v`, `r`; the Courant limit is checked per substep (`TransportStepRejected`, recoverable); four calls of dt/4 match one call with `n_substeps=4` to 1e-14. `max_decay_exponent = max v r dt/n_sub` is reported for the splitting-error estimate (the reaction itself is exact for any value).
- **MAPLE integration** (`test_transport_step_applies_through_maple_water_step`): on the Phase 1 probe bed, `water_demand_from_transport(step, pickup)` applied through the real `apply_water_process_demand` yields `mobile_after = T(M) + actual removal − Dep − E` bitwise, export recorded as MAPLE boundary export, deposition applied as requested, and `Σ T(M) = Σ M` so no internal transfer appears as a reservoir term.
- **Backend.** Namespace ops, MAPLE `scatter_add` (one per substep plus four face scatters per step, each one device synchronisation on CuPy), one batched flag read; ~14 `(n, nc)` work arrays plus four face arrays, bounded, no cohorts, no Python loop over cells. `test_cupy_backend_matches_numpy_when_a_gpu_is_available` compares CuPy with NumPy and skips without a device.

Not done by the operator: no per-parcel or carried travel-distance memory (removed: a mass-weighted scalar is not an exact mixture of exponential laws; Codex selected the local policy), no instantaneous deposition ahead of the mobile mass, no `nmax` walk limits, no ring deposition.

## 4. Travel-distance policy: local hydraulics

The deposition rate at a cell is `1/L` from **that cell's current** law (`SedimentPhysicsStep.deposition_rate_per_m`); mass arriving from upstream deposits at the local hazard. Cells with no transport law (regime 1, wet without rain and Re ≤ 500) have rate 0: the pool advects at the recession memory velocity without depositing, as the legacy pool does. This is a modelling choice with a documented difference from the legacy's source-assigned walk: where hydraulics weaken downstream, SYRUP deposits sooner; where they strengthen, later. It is the Phase 5 event policy (Codex resolution); a source-assigned distribution is not claimed or approximated.

## 5. API

```python
from maple_syrup.sediment_physics import (plot1_sediment_parameters, sediment_physics_parameters,
    physics_grid_from_graph, sediment_physics_step, median_diameter_m, recession_velocity)
from maple_syrup.sediment_transport import (transport_network, transport_step, water_demand_from_transport,
    reaction_survival, TransportStepRejected)

params  = plot1_sediment_parameters(xp=graph.xp)              # legacy_literal conventions by default
grid    = physics_grid_from_graph(graph)                       # slope, active, dx², once per graph
network = transport_network(graph)                             # receivers, outlets, face tables, once per graph

phys = sediment_physics_step(params, grid, depth_m, velocity_m_s, rain_rate_m_per_s,
                             vegetation_cover_fraction, active_layer.mass_kg, v_memory, dt)
#  .requested_pickup_kg (ny,nx,nc)  .sediment_velocity_m_s  .deposition_rate_per_m  .settle_mask
#  .law_applies  .regime (int8)  .d50_m  .shear_velocity_m_s  .reynolds_number  .stream_power_w_m2 ...
tr = transport_step(network, mobile_kg, phys.sediment_velocity_m_s, phys.deposition_rate_per_m,
                    phys.settle_mask, dt, courant_max=1.0, n_substeps=1)
#  .mobile_after_transfer_kg  .deposition_request_kg  .export_request_kg  .x/y_face_*_kg  .divergence_kg
#  .budget_residual_by_class_kg  .budget_tolerance_by_class_kg  .max_courant  .max_decay_exponent
demand = water_demand_from_transport(tr, requested_pickup_kg_or_zeros)
```

Shapes and dtypes: cell fields `(ny, nx)` FP64, class fields `(ny, nx, nc)` FP64 (settle/law masks bool, regime int8), all in one NumPy/CuPy namespace; parameters carry `(1, 1, nc)` device copies. Both steps are pure, validate inputs, intermediates and outputs through `DeferredChecks` (one batched flag read), and raise before returning anything.

## 6. Integration notes for Phase 5b (bed/runner)

Per accepted coupled step of length `dt`, after `coupled_step` has produced `route.depth_m`, `route.velocity_m_s` and the step's rain rate (post-route consistent depth and velocity, preferred over the legacy old-depth/new-velocity mix; the Phase 5b timestep study measures the split error):

**Recommended initial transaction — two MAPLE calls, same-step pickup, no source lag:**

1. `phys = sediment_physics_step(...)` with the routed depth and velocity, the step's rain rate, the sidecar vegetation fraction, `active_layer.mass_kg` (current holdings) and the velocity memory.
2. **Pickup call:** `apply_water_process_demand` with `requested_removal = phys.requested_pickup_kg`, zero deposition/export, zero face flux. MAPLE caps by availability and holdings and moves the *actual* removal into the mobile pool of the same cell.
3. `tr = transport_step(network, result.new_water.mobile, phys.sediment_velocity_m_s, phys.deposition_rate_per_m, phys.settle_mask, dt, n_substeps=...)` on the actual post-pickup pool. On `TransportStepRejected`, discard the unpublished attempt and retry the whole coupled step (halve dt or raise `n_substeps`); the sediment Courant number is not implied by the water's.
4. Publish `WaterState(depth_m=route.depth_m, mobile=tr.mobile_after_transfer_kg)`.
5. **Deposit/export call:** `apply_water_process_demand(..., water_demand_from_transport(tr, zeros))`. Requests are ≤ the pool by construction; assert MAPLE refused nothing.
6. `v_memory = phys.sediment_velocity_m_s`; `accumulate_water_step_result` for both calls; `maybe_commit_topography` under `constant_depth`; rebuild the graph, `physics_grid_from_graph` and `transport_network` after a commit. Elevation change uses MAPLE's bulk density (1250 kg m⁻³) on the solid mass at 2650 kg m⁻³; report the 2.12× morphology departure from legacy `z_change`.

**Alternative — one MAPLE call per step (source lag):** transport the previous pool, then one call with pickup + deposition + export. Mass picked up in step n first moves in step n+1; the lag vanishes as dt → 0 but at dt = 1 s it is comparable to a cell transit in slow flow. Halves the accounting cost (contract G11). Not implemented in this milestone; any future comparison must measure timing and source/sink error together.

Checks the runner must add (contract §4.1): `tr.budget_residual_by_class_kg` within its tolerance (enforced inside), zero refused deposition/export from MAPLE, per-class `bed + mobile + exported` closure over the event, and the sediment Courant retry path exercised by a test.

## 7. Restrictions and limitations

- Supported inputs: six MAHLERAN classes with strictly increasing diameters (any count with matching per-class parameters is accepted, but the laws are calibrated for the six); one particle density (legacy scalar; Phase 5b must assert equality with `grain_classes`); rain intensity ≥ 0, vegetation fraction in [0, 1]; square cells; the Phase 4 draining D4 graph (no pits or flats; the event driver rebuilds drainage after each terrain commit). Inputs that overflow FP64 anywhere are refused.
- Legacy caps (`hs`) are implemented as written but do not bind for Plot 1 (1000 mm); real supply limits are MAPLE's.
- No splash, no dry-cell detachment, no nutrient/marker paths, no `nmax` walk limits, no ring deposition.
- Spatial deposition pattern accuracy is first order in `dx` (upwind); the deposition timescale and advective mean timing are exact/first order in `dt`. On Plot 1 (dx = 0.5 m) diffuse-flow travel distances of centimetres deposit in their source cell with a leak of order `a s_h/(1 − (1 − a)s_h²)`; Phase 5b should quantify it on the imported case.
- Sediment velocity below the memory decay never reaches exactly zero on wet no-law cells; a residual pool creeps until the Phase 6 terminal policy settles it (set `settle_mask` everywhere for the dry reset — tested as the all-dry case).
- Core-author verification did not include CuPy, timing or a Plot 1 storm. Subsequent integrated CPU tests, actual original-Fortran equation checks, storm runs and timings are recorded in [acceptance.md](acceptance.md). GPU execution remains unverified.

## 8. Integrated verification

Use the fixed MAPLE dependency environment in [dependency.md](dependency.md),
then run the commands recorded in [acceptance.md](acceptance.md).
`pyproject.toml` includes `tests/phase5` in the default test paths. The
bed/event integration is described in [integration.md](integration.md).
