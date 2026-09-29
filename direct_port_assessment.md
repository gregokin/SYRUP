# Directly porting MAHLERAN water processes into MAPLE

Independent Claude review incorporated, 2026-09-28. No implementation requested or performed.

**Assessment:** a native MAPLE water module is feasible and likely a cleaner long-term integration than maintaining two live model runtimes. It is a moderate-to-large scientific software task, not a simple copy of the water directory. Many detachment and friction equations are short and relatively easy to translate, while infiltration needs correction and state-ordering work; hydraulic routing, conservative sediment transport, persistent state and verification dominate the work. This judgment is based on source inspection, not a completed prototype or benchmark.

Here “port” means translating selected Fortran calculations into Python/array operations under MAPLE's architecture, retaining their mathematical behavior where sound, and using MAPLE's existing bed/accounting code. This maximizes reuse of MAPLE code and MAHLERAN algorithms, but does not preserve the Fortran source verbatim. A compiled wrapper would reuse more literal source; it would still need the same conservation, unit and state decisions.

## Component assessment

| Component | Relative difficulty | What can be reused and what must change |
|---|---|---|
| Rainfall input and event forcing | Low–moderate | MAPLE has time/block infrastructure; add rainfall forcing and water-event configuration. Do not carry over XML driver/file I/O. |
| Local infiltration and soil storage | Moderate–high | infilt.for is 217 lines including comments. Much is per-cell arithmetic suitable for arrays; it also mutates depth, discharge and velocity, and contains several infiltration options. Separate input/output state and make a full water budget explicit. The inf_type>=5 branch uses water_in/drain without defining them in that branch; the inf_model=2 rainfall condition reads column 2 rather than the current column. Triage supported options before translation, and reconcile infiltration changes to old depth/discharge with old inflow terms used by the hydraulic solver. |
| Friction, shear and transport-regime criteria | Low–moderate | Reuse equations from route_water, ff_type8 and route_sediment_xml. Separate pure local calculations from routing; supply current grain fractions. Friction can depend on unknown depth inside the hydraulic solve, so not every evaluation is a one-time preprocessing step. |
| Flow directions, masks, outlets, drainage ordering | Moderate–high | Legacy topog_attrib/contrib_area provide reference algorithms, but their boundaries, allocation and refresh behavior require adaptation. MAPLE has a routing callback slot, not an implemented hydraulic drainage solver. |
| Water routing | High | route_water.for is 1,445 lines, including multiple solver and friction options. Select one supported method first, preserving its numerical semantics. Upstream-ordered implicit methods depend on newly computed neighboring discharge; naive vectorization changes the solver. Need convergence handling, nonnegative depth and water conservation. |
| Entrainment and distance laws | Low to translate, high to validate | flow_detachment 50 lines; diffuse 77; concentrated 91; suspended 59. Reuse formulas but resolve dt dependence, rate versus depth, mean versus median, and units. These decisions can change predicted erosion. |
| Wet sediment advection and deposition | High | Reuse legacy continuity/velocity equations and MAPLE's bed exchange. Add mobile-pool routing between cells, constrain transfers by actual mass, and validate per-cell/class continuity. Existing face-flux arrays are diagnostics, not a working water advection implementation. |
| Rain splash | Moderate–high | Port its redistribution separately. It directly redistributes sediment and has wet/dry unit and edge behavior to resolve. It must not masquerade as suspended-load transport. |
| Bed, grain composition and availability | Low–moderate integration effort | Reuse MAPLE active-layer/voxel/availability/ledger operations unchanged where their contracts fit. Do not port legacy sed_temp, z_change and competing bed ownership as a second inventory. |
| Hydrology state, restart and event completion | High | Extend canonical state/checkpoints for the history genuinely required by the chosen scheme. Existing WaterState has only depth/mobile sediment. Define runoff/settling completion and soil-moisture carryover; do not erase sediment at switching. |
| Ordered sweep implementation and CPU performance | High risk until measured | Water bisection and sediment CN use upstream values from the current timestep. Whole-domain simultaneous vectorization is not equivalent. Benchmark a faithful ordered CPU implementation early; drainage-level batching or a small compiled/JIT kernel are candidate solutions, with dependency/build costs if needed. No measured speed estimate or mandatory new dependency is established yet. |
| GPU implementation/performance | Deferred risk | Local array physics fits MAPLE's backend pattern. Ordered routing, iterative solves and downstream path walks may not; existing GPU support does not make them fast automatically. Prove CPU correctness first and profile. |

The six files in Subroutines_Water total 1,993 lines including comments; route_water accounts for 1,445 of those. The sediment routines and driver/initialization/geometry dependencies lie outside that directory. Raw line counts describe where complexity sits; they are not work estimates. The intended first port would select a subset, not translate every historical option.

## How the native water module would fit

At each fluvial timestep:

1. Consume rainfall, bed geometry/composition, current water/soil state and actual dt.
2. Update infiltration/soil storage and solve water depth/discharge, respecting water budgets and the selected numerical order.
3. Calculate entrainment and deposition tendencies. Use MAPLE bed routines to determine actual supply-limited pickup.
4. Advance mobile sediment with the selected face-flux scheme; resolve deposition against the available mobile store. Keep splash as a separate conservative transfer. The exact split/order must be specified against the selected legacy equations, not silently chosen for coding convenience.
5. Return validated new hydraulic/mobile state, boundary fluxes and shared bed/ledger changes. Publish the timestep transactionally.
6. Use the existing commit lifecycle to refresh geometry and hydraulic routing. Choose one water-depth authority so hydrological evolution and the current constant-free-surface callback do not both apply the same bed displacement. MAPLE already supplies constant_depth, which can leave depth unchanged during bed commits while the hydraulic solver owns the water budget; this is reusable policy machinery, not a hydraulic solution by itself.

The existing water entry point would dispatch to this real solver instead of a prescribed demand. The current local apply_water_process_demand cannot become a complete lateral solver just by supplying face-flux diagnostics; refactoring the water step to insert transport is likely necessary. Its shared bed operations and accounting contracts can be retained.

## What makes a limited first port manageable

- A water-only first milestone, followed by wet sediment. The shipped XML selects iroute=5 bisection, constant friction (type 1), sediment CN (method 2), and infiltration model 2. This provides a concrete candidate starting point. Use one corrected/validated infiltration option. Constant friction simplifies each water solve; sediment CN has a closed-form local update but still needs upstream ordering. The header describes iroute=5 as stable; neither that comment nor the shipped defaults proves this is the best method.
- Square cells, the six common grain classes, explicit nonperiodic outlets, and a small set of friction/infiltration options.
- CPU first; ordinary Eulerian sediment rather than marker-in-cell. Explicitly specify which wet transport regimes are supported and add splash as a separate subsequent stage. Until then reject or clearly declare splash-disabled experiments, rather than presenting them as the full rainfall-erosion model.
- Reuse MAPLE's event separation, bed operations, geometry commit framework, conservation reporting and output infrastructure. Extend rather than replace the existing state model.
- Keep continuous vegetation growth, nutrient transport, marker/RNG state and the full interstorm ecosystem out of the initial scope. Still preserve soil-moisture history and define the minimal inter-event drying/drainage behavior needed by the combined model.

## Necessary work beyond translation

Legacy floors and drying resets can destroy mobile mass; unclipped bed deposition can manufacture it. Mixed wet/splash units prevent one universal array conversion. Different geometry updates handle nodata differently. Existing MAPLE water-enabled configurations disable some dry-only fast paths. These issues persist with a wrapper as well as a native port, so avoiding translation would not avoid the main scientific decisions.

Validation should include local formula comparisons against the selected Fortran expressions where their behavior is intentionally preserved, analytic water/sediment budget cases, a sloping small grid, dry/wet interfaces, sources/sinks/outlets, timestep refinement, mixed grain depletion and wind→water→wind restart equivalence. Legacy end-to-end output is useful reference evidence but cannot be the only oracle when deliberately correcting budget or unit errors.

## Recommendation and uncertainty

Prefer a native MAPLE port for the durable combined model, contingent on a small feasibility implementation demonstrating one conservative water-only path. It removes ongoing unit/array conversions and duplicate runtime state, and naturally connects water erosion to MAPLE's changing bed.

Call this **moderate–high overall difficulty**: moderate implementation effort for a narrowly selected CPU subset, high difficulty for a fully validated general replacement covering all MAHLERAN modes. The difficult work is conservation, numerical integration and persistent state, not translating Fortran syntax. Water-only with one selected scheme is moderate difficulty; wet sediment, event transitions and restart make it moderate–high. Treat CPU sweep performance as an early feasibility gate. Keep a thin standalone Fortran comparison harness if feasible, without requiring a second runtime in production. Do not give a precise calendar estimate before selecting routing/options and measuring one prototype. Neither build equivalence nor performance has been demonstrated.

Evidence: integration_findings.md and the first Claude review; additional direct reads of infilt.for, ff_type8.for, route_water.for, route_sediment_xml.f90, MAPLE's water files and topographic routing callback. No additional runtime tests were run for this architectural assessment.

## Independent review and qualifications

Claude reviewed this assessment against the source (read-only, 24 turns, 183083 ms). Full report: [direct_port_review.md](agent_handoffs/direct_port_review.md). Codex checked the shipped XML options, ordered inflow/STOP paths, infiltration branches, and existing depth policy before incorporating the corrections.

We agree on moderate–high overall difficulty, a narrowed water-only milestone, early CPU performance measurement, and native MAPLE ownership. Two review suggestions remain conditional:

- A compiled/JIT sweep may be worthwhile; it is not proven mandatory, and the review's iteration/call-count performance estimates are not benchmarks. No dependency is selected by this assessment.
- Legacy STOPs should become transactional errors or validated timestep-retry logic. Clamping and merely reporting the discrepancy is not an accepted conservative remedy. Likewise, freezing the timestep makes a controlled comparison possible but does not fix timestep-dependent entrainment physics.

No new model code or runtime tests were added for this assessment. The earlier two-cell diagnostic and 12 integration checks validate existing interfaces only, not the proposed port.
