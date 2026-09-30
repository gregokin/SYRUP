# MAPLE-SYRUP implementation phases

Proposal, 2026-09-29. This records the latest user scope; it does not start implementation. SYRUP means Sediment Yield, Runoff, and Uptake by Plants and is an extension dependent on actual MAPLE, not a standalone landscape model or a fork of MAPLE wind physics. MAPLE owns wind and shared infrastructure; MAPLE-SYRUP supplies water-specific processes and invokes MAPLE. Wind and water never transport simultaneously.

## Phase 1 — MAPLE dependency and interface contract

Establish how MAPLE-SYRUP imports and invokes the installed/development MAPLE package. Reuse MAPLE configuration/case compilation, geometry, grain classes, backend, voxel/active-layer state, availability/ledger, topographic commits, diagnostics, and restart infrastructure wherever available. Define water-specific additions through explicit extension points; propose narrow MAPLE interface changes where necessary rather than cloning its infrastructure. Record MAPLE version plus any working-tree delta in run provenance. Aeolian changes remain in MAPLE and reach MAPLE-SYRUP through its dependency; API compatibility still needs checking.

Map shared versus water-owned state and lifecycle. Select one authority for hydraulic depth: MAPLE's existing default topographic depth adjustment reports water-volume changes and is not by itself a conservative hydraulic redistribution. Avoid double updates. Design GPU-resident state, bounded memory, and compiled/GPU treatment of ordered routing now. Define restart fields and backend interfaces before implementation, even though complete validation occurs later.

Explicit interface acceptance checks: MAPLE's water step currently adds local pickup, subtracts local deposition/export, and returns face-flux diagnostics without applying intercell flux divergence. Define and test where conservative lateral mobile-mass transfer occurs; it cannot be represented as fictitious boundary inflow/export. MAPLE's canonical WaterState contains depth and mobile sediment, not complete hydraulic/infiltration history. Define a water-adapter checkpoint extension (or prove deterministic reconstruction) for every additional solver state; do not assume canonical state alone reproduces a mid-storm restart.

Exit: documented invocation/state contract, real MAPLE import/setup smoke test, source provenance, and explicit required extension points; no copied wind physics.

## Phase 2 — Import one MAHLERAN case into actual MAPLE state

Start by auditing Input/input_p1 with the root mahleran_input.xml and p1_01_08_06.dat rainfall series. The DEM file declares 22 columns, 62 rows, and 0.5 m cells. Establish the physical interior versus legacy boundary ring before choosing the MAPLE grid. Translate the DEM, six class-fraction maps, prescribed vegetation/pavement/surface properties, and selected soil parameters into MAPLE's case import/compilation conventions. Respect configured options: a map's existence does not mean legacy code uses it.

Use actual MAPLE voxel and active-layer initialization. State assumed subsurface stratigraphy, erodible depth, density/porosity, vertical datum, and active-layer thickness where legacy data do not determine them. Preserve source provenance; declare units, nodata, row orientation, grain ordering, masks, and open outlets explicitly. MAPLE's importer records geospatial metadata but its configured geometry controls numerical placement; do not rely on implicit resampling or pixel-size inference.

Exit: a compiled MAPLE case with independently checked terrain, class inventories and active-layer/voxel partition, plus an inspectable map/input report. Case suitability remains contingent on boundary, pit, parameter, and file completeness audit.

## Phase 3 — Rainfall and minimal subsurface water

Implement MAHLERAN-compatible constant and time-series rainfall semantics and the selected case's spatial rainfall scaling/mask. Check legacy timestamp interval meaning. Integrate rainfall exactly across timestep/forcing boundaries and convert mm/h to consistent internal units. No silent interpolation or changed storm totals.

Port one case-appropriate MAHLERAN infiltration formulation and only the soil moisture/storage/drainage state it needs. Handle runon infiltration once routing is connected. No storm evapotranspiration or continuous ecohydrology. Distinguish finite infiltration capacity, cumulative infiltration, retained subsurface water, and deep drainage; do not subtract infiltration twice.

Exit: forcing volume matches the input record; column/no-routing tests balance rainfall, surface storage, soil storage and drainage, including saturation and rainfall gaps. GPU/backend checks begin here.

## Phase 4 — Surface flow, water depth and discharge

Port a selected MAHLERAN hydraulic method and roughness relationships, reusing its computational approach where sound. Build routing from the imported terrain with explicit masks, outlets and supported pit behavior. Compute depth, discharge and velocity; couple rainfall and infiltration consistently. Include wetting/drying, stable/adaptive substeps or rejection, and post-rain recession.

Use the existing prototype as numerical evidence, not as a complete solver. Ordered upstream dependencies require deliberate compiled or GPU algorithms (for example dependency levels); blind vectorization must not change the numerical update silently. Retain CPU reference checks and add meaningful device execution tests. Any pit in the selected case must be handled or explicitly resolved in the case definition rather than losing its water.

Exit: conservative water-only storm with depth maps/outlet hydrograph, matched selected legacy hydrology where runnable, independent controlled tests, timestep refinement, and initial runtime/memory evidence.

User refinement, 2026-09-29: include Numba compilation of the same MAHLERAN-derived ordered solver in Phase 4 and use it for direct comparisons with original Fortran routines. Preserve the selected equations and corrected bisection, without fast-math or changing the hydraulic method. Test compiled-versus-array parity and original-Fortran agreement separately; document root-bracket and conservation corrections rather than claiming literal identity. Record dependency versions and separate compilation/startup from steady-state cost. Numba CPU compilation does not establish GPU acceleration.

## Phase 4R — Priority follow-up: alternative routing approaches

Added by user direction, 2026-09-29. This important investigation follows acceptance of the Phase 4 water-only baseline; its suffix preserves existing phase numbers. It is distinct from Phase 4 implementation subtasks. See [the literature review](docs/phase4/solver_alternatives.md). Numba compilation of the baseline is now part of Phase 4, not deferred here.

First investigate safeguarded Newton on the same scalar equation and improved dependency-level/GPU execution. Then compare explicit conservative kinematic routing and local-inertial face-flux methods as separately selectable experiments. Retain MAHLERAN's selected Darcy–Weisbach relationship where applicable and document changed hydraulic terms, time discretization, direction selection, boundaries and additional state. Explicit kinematic timestep limits must account for wave celerity, not only water velocity. Do not adopt the rejected sandpile formulation.

Compare identical forcing, terrain, infiltration and boundary assumptions on Plot 1 and larger networks with varied width/depth, wetting/drying and recession. Set quantitative acceptance tolerances before evaluating candidates. Measure runoff volume, hydrograph peak/time, depth, velocity, constitutive residuals, conservation and timestep/grid sensitivity alongside whole-event runtime, startup, peak CPU/GPU memory and host/device transfers. Depth and velocity fidelity matter for subsequent erosion even when outlet volumes agree. Recheck detachment and sediment timing when sediment coupling becomes available before adopting a different hydraulic solver.

Exit: reproducible comparative benchmarks and an evidence-based recommendation to retain or supplement the baseline, including accuracy/performance tradeoffs, hardware, backend agreement and unsupported cases. No solver is accepted merely because it is faster or conserves total water. Unavailable GPU execution remains unverified. The investigation may conclude that the compiled MAHLERAN approach is sufficient.

## Phase 5 — Conservative detachment, travel and deposition

**CPU milestone accepted 2026-09-30:** actual MAPLE bed integration, wet laws, conservative transport, supply/depletion/sorting, terrain rerouting, controlled distance/timing checks and imported-case erosion/deposition/discharge verified. Full regression:446 passed,6 GPU skips. Original-Fortran equation tests and matched 1/0.5/0.25-second SYRUP storms completed. See [acceptance and limitations](docs/phase5/acceptance.md). GPU equivalence and full original-MAHLERAN sediment-storm fidelity remain carried-forward qualification work; CPU kernel scaling is measured, larger whole-event scaling remains pending. The implementation below was separately authorized by the user's Phase 5 request.

Implement one complete sediment path before widening regime coverage: selected MAHLERAN wet detachment laws, class-specific transport distances and virtual speeds, actual withdrawal from MAPLE's bed, downstream mobile transport, deposition through MAPLE, and measured outlet export. Retain the applicable diffuse/concentrated/suspension distinctions for the case; unsupported classes/conditions must be explicit.

Separate the meanings of rain-assisted detachment into runoff and direct splash redistribution. Splash redistribution is deferred by user direction; preserve rain-assisted wet detachment where the retained MAHLERAN law requires it, or label a deliberately narrower first case. Reconcile legacy rates/time conventions, mean/median distances, and travel-memory choices. Limit pickup to actual available bed mass and deposition to actual mobile supply. Validate source/sink and advection splitting rather than assuming conservation implies fidelity.

Commit bed changes through MAPLE and refresh hydraulic geometry/drainage consistently. Reconcile water volume during topographic changes through the phase-1 ownership contract. No competing terrain or grain inventories.

Exit: per-class bed + mobile + exported mass closes; depletion, sorting, local exchange and downstream transfer pass controlled checks; travel distances/timing are validated; the imported case produces conservative erosion/deposition and sediment discharge. GPU equivalence and memory scaling remain acceptance concerns throughout.

## Phase 6 — Complete storm and dry handoff

Continue runoff after rainfall ends. Define configurable stopping conditions covering rainfall completion, flow/discharge, remaining surface storage and mobile sediment, with a hold period where appropriate and a failure limit that cannot be mistaken for physical completion. A low outlet discharge alone cannot prove a ponded domain is dry.

Once the criteria hold, apply the user's explicit dry-again assumption. Settle remaining sediment conservatively according to the terminal policy; record residual surface/subsurface water removed by this inter-event reset separately from storm runoff, infiltration or numerical error. Restore a declared dry/antecedent soil state without deleting sediment. Preserve pre-reset quantities for audit. This is a simplifying boundary condition, not simulated evapotranspiration.

Use MAPLE-compatible event records/checkpoints with additional required water-state persistence, and test uninterrupted versus resumed storm results. The next event sees the accepted MAPLE bed, availability and terrain.

Exit: repeatable complete water event with justified tolerances, explicit reset budget, restart test and valid dry MAPLE handoff state.

## Phase 7 — MAHLERAN and performance qualification

Assemble comparisons built during every preceding phase into a matched event benchmark: forcing, infiltration/runoff, water depth/discharge, sediment export by class, bed change, distance/timing and timestep/grid sensitivity. Build/run the reference in an isolated location if possible, without modifying legacy source. Establish selectable process parity (especially splash) before asserting like-for-like sediment comparison; if legacy cannot isolate it, use controlled kernel references and clearly label full-case differences.

Document each preserved equation and intentional numerical/physical departure. Do not require agreement with a demonstrated legacy bookkeeping error. If executable reference results are unavailable, label analytic/Python checks as such rather than claiming Fortran validation.

Measure representative larger grids as well as the small import case, backend parity, startup versus steady state, CPU/GPU peak memory and transfers. A tiny case may be slower on GPU; require a measured scaling report, not universal GPU speedup. Profile actual bottlenecks and optimize without weakening physical tests. This phase consolidates performance validation, not defers GPU design until the end.

Phase 5 CPU profiling identifies a concrete performance follow-up: the two actual MAPLE bed-exchange calls consume about 75% of a profiled 600-second event, versus about 10% for the wet laws and lateral transport together. Investigate a shared MAPLE transaction/validation optimization without copying bed physics, weakening checks, or introducing an unvalidated source lag. Measure complete events and preserve source/sink timing as well as conservation; see docs/phase5/acceptance.md. This complements, rather than replaces, Phase 4R's hydraulic/GPU investigation.

Exit: evidence-backed scientific/compatibility/performance report with explicit limitations and an accepted baseline for the first water-only MAPLE-SYRUP experiment.

## Phase 8 — Actual MAPLE wind/water invocation smoke test

After the water-only target is accepted, invoke real MAPLE wind events around MAPLE-SYRUP water events on the shared state. Verify wind-water-wind inheritance of composition, availability and topography, mutual exclusion, budgets and restart. Wind evolves exclusively through MAPLE code; MAPLE-SYRUP does not carry copied wind equations. Dependency changes must pass this compatibility test.

Exit: a short real alternating-event demonstration. This is a later integration milestone, not a prerequisite for declaring the first water-only experiment complete.

## Excluded from this first experiment

Direct splash redistribution, vegetation growth, nutrient/carbon transport, full ecohydrology/ET, all legacy options, and unrestricted long-term landscape prediction. These can be added later without replacing the shared MAPLE bed or duplicating wind physics.

## Local grounding and review status

Inspected MAPLE case_import.py, case_compiler.py entry points, water/depth.py, coupling/water_event.py, and existing project assessments/prototype evidence; inspected MAHLERAN root XML, Plot 1 files/DEM/rainfall and Set_rain_xml.f90. This is planning, not implementation or fresh numerical validation.

The initial review failed authentication; the resumed review completed successfully (Claude session 1900772f-5ff2-46ec-86a8-9638f5810a76). Report: agent_handoffs/tasks/syrup_plan_review_20260929/claude_report.md. Codex independently checked key source findings and incorporated the qualifications below. Peer review is not implementation acceptance; no numerical tests were run for this planning update.

## Peer-review corrections and phase requirements

### Phase 1: concrete integration gaps

MAPLE's existing fluvial outer step calls the accounting dispatcher without a computed demand injection. It therefore cannot simply be selected as the complete SYRUP event runner. Specify a narrow solver callback/extension or a SYRUP driver composed from actual MAPLE APIs. Do not duplicate MAPLE scheduling, bed physics, ledger, or commit logic unnecessarily. Test block-level mobile identities as well as the per-step budget when composing pickup, routing and deposition.

Topographic commits can already apply MAPLE's configured water-depth callback. Explicitly select a volume-consistent policy (for example constant_depth with SYRUP owning hydraulic recomputation), rather than letting the default constant-free-surface adjustment rewrite solver depth unnoticed. Record precisely when routing is refreshed after committed bed changes.

Define solver checkpoint persistence concretely: a versioned sidecar bound to the complete MAPLE checkpoint/case/solver identity, or a small MAPLE checkpoint extension. A water-array hash alone does not identify all relevant bed/configuration state. Save irreconstructible hydraulic, rainfall-clock, infiltration and transport-memory state. Do not reinterpret current MAPLE snapshots as already storing these fields.

### Phase 2: Plot 1 is not a clean reference fixture yet

The root mahleran_input.xml selects use_map_phi=true and maps phi_1 through phi_6 to plot1_phi1.asc. Its alternative authored means are 0.2, 0.2, 0.2, 0.2, 0.1, 0.1. The setup routine subsequently rescales by pavement. Consequently the selected input cannot be assumed to form six closed fractions. Audit the actual raster and post-setup sums, choose the intended composition with evidence, and document any correction or normalization as a departure. Do not silently normalize or use the six differently named files merely because they exist. Benchmark runs must use the same corrected composition where a matched comparison is intended.

Legacy computed interior loops use rows 2..61 and columns 2..21: candidate physical grid 60 by 20 at 0.5 m. Explicitly map the surrounding ring to boundary behavior and measure its contribution to legacy loss/export diagnostics. Do not claim all ring deposition is physically exported without checking paths. A cropped interior with explicit outlets is the candidate, but test terrain connectivity and pits before declaring the outlet map. MAPLE wind boundary configuration is a separate future event choice; its prescribed-boundary vocabulary is not a hydraulic wall/outlet implementation.

Active-layer thickness, erodible depth and subsurface stratigraphy are independent MAPLE initialization assumptions. The legacy active_layer_sensitivity detachment coefficient must not be mistaken for a measured active-layer thickness.

### Phases 3–5: preserve the selected wet physics explicitly

Plot 1 selects infiltration model 2, hydraulic routing method 5, sediment routing method 2, and dt=1 s in the root XML. Start by tracing those choices. In infilt.for the model-2 rainfall branch tests r2(i,2) rather than the current column: resolve this as an explicit legacy behavior/correction and test spatial rainfall, rather than porting the index blindly. Time-stamped rainfall records use interval-ending semantics in Set_rain_xml; define clean boundary integration and distinguish it from legacy discrete-time off-by-one behavior.

Keep rain-assisted wet detachment and water-depth attenuation for the selected shallow-flow/diffuse regime. Defer dry-cell splash_transport redistribution. Otherwise the shallow-flow case can lose its principal wet erosion mechanism. All grain classes must have conservative supply limits even where legacy caps were narrower. Select and document legacy travel-distance, virtual-speed and memory conventions; per-step decay constants need timestep-sensitivity checks.

### Phases 6–7: honest comparison and performance

The flow-cessation dry reset is a deliberate new inter-event policy, not reproduction of MAHLERAN's fixed-duration storm run. MAPLE's evolving surface composition also intentionally differs from legacy fixed proportions. Compare these effects separately from equation or solver errors.

Profile MAPLE validation/reduction synchronization and exchange allocation overhead on GPU. Larger exchange cadence or batching is only a candidate optimization: it can change depletion, composition feedback and physical transport, so it needs its own numerical/conservation tests. Do not prescribe batching as a correctness-neutral shortcut or assume a universal CPU/GPU speedup. Existing backend implementation details and compiler availability must be rechecked during implementation.
