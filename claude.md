# MAPLE-SYRUP: Sediment Yield, Runoff, and Uptake by Plants

## Current scope clarification — 2026-09-29

The project is now named **MAPLE-SYRUP (Sediment Yield, Runoff, and Uptake by Plants)**. It is a water extension dependent on actual MAPLE infrastructure, not a standalone framework or copied/forked aeolian model. Wind physics stays in MAPLE and MAPLE-SYRUP invokes MAPLE. Use its computational, configuration/file, physical-state and other relevant infrastructure. Earlier suggestions of importing/synchronizing copied wind implementations are superseded. Track dependency provenance and compatibility instead.

First target: import a MAHLERAN case into MAPLE voxels/active layer; implement rainfall, necessary infiltration/subsurface state, depth/flow routing, and conservative wet detachment/transport/deposition. **Splash is deferred.** Full storm evapotranspiration, ecohydrology, plant growth and nutrients are excluded initially. After rainfall and flow finish within justified tolerances, apply a documented dry-again assumption with explicit water reset accounting and conserved sediment. Speed, memory and GPU acceleration remain priorities. See syrup_implementation_phases.md for the proposed phases; subsequent user continuation authorized phased implementation. The user requested a commit after Phase 2 passes independent verification; Codex owns that commit, with no push requested.


## Project intent and current direction

Build a combined model in the MAPLE framework that alternates wind transport and water erosion/transport over the same evolving sediment bed. **Wind and water transport do not occur at the same time.** Each process must inherit the sediment composition and topography left by the other.

The agreed direction is **MAHLERAN-like water physics inside MAPLE**, rather than coupling two independent landscape inventories or reproducing the entire MAHLERAN application. Reuse existing code wherever practical. Retain MAHLERAN's exact physical relationships and approaches where possible; when they are infeasible, inefficient, or incompatible with MAPLE's state representation, use a documented approximation that preserves the relevant physical behavior. Numerical and output identity with MAHLERAN is desirable where attainable, but is not a universal requirement.

Start with a simple version without vegetation growth, full ecohydrological feedbacks, or nutrient transport. This is a process-based water erosion model, not a sandpile model. The sandpile connectivity paper discussed below remains background and a possible comparison, not the implementation direction.

## Current workflow and additional priorities

The user has authorized automatic Codex–Claude orchestration for SYRUP until changed. Follow `AGENTS.md` and `agent_handoffs/orchestration_state.md`; the earlier one-off review/manual restrictions are historical. Codex scopes and independently reviews bounded Claude work. Use ordinary permissions; no bypass is authorized. Preserve task prompts, reports, logs, session IDs, and verification. No background process is implied by these files.

MAPLE's aeolian development will continue. Keep shared/aeolian code sufficiently close to upstream MAPLE to bring improvements into SYRUP throughout development. Isolate water-specific modules, minimize common-interface changes, track exact upstream revisions and local patches, and validate each proposed update in an isolated candidate. Do not use a mutable live MAPLE checkout as an unrecorded production baseline. The synchronization mechanism remains to be selected.

Speed, memory performance, and GPU acceleration are first-class design and acceptance priorities. Reuse MAPLE's GPU/backend facilities where appropriate; plan routing dependencies, layouts, batching, transfers, and memory use early. Measure representative end-to-end runtime and peak memory, distinguish startup from steady-state execution, and preserve physical behavior across backends. CPU-only prototypes do not satisfy the eventual GPU requirement.

## Locations and existing work

- `/home/okin/SYRUP`: project assessments, coordination documents, and isolated feasibility experiments; current working directory.
- `/home/okin/MAPLE`: existing Python/NumPy/CuPy wind model and sediment/landscape framework.
- `/home/okin/MAHLERAN`: legacy Fortran water erosion model, version 1.2.3, serving as the physical and benchmarking reference.

The source trees were inspected without changing them during the assessment. MAPLE already contained unrelated working-tree changes; inspect the current state and preserve existing work. Do not assume these notes describe the latest checkout without checking.

Read these documents as needed:

- `water_extension_scope.md`: detailed proposed inclusion/exclusion table and scientific decisions. The user's current simplified scope, summarized here, takes precedence over earlier broader proposals.
- `integration_findings.md`: source-level integration issues, including conservation and missing water routing.
- `direct_port_assessment.md`: feasibility and limitations of a more literal port.
- `port_feasibility/README.md`: implemented small experiments and their explicit limits.
- `agent_handoffs/physics_native_review.md`: Claude's review of the native-physics approach.
- `agent_handoffs/claude_review.md` and `direct_port_review.md`: earlier reviews; some discussion predates the current direction.

## Initial scientific scope

Include the water processes needed for a defensible storm erosion model:

- Rainfall forcing, infiltration/runoff generation, surface-water storage, flow routing, and hydraulic variables needed by the selected erosion laws.
- MAHLERAN-like detachment, sediment transport distance, deposition, and sediment travel/virtual velocity, with grain-class distinctions and applicable transport regimes.
- Rain-impact detachment and splash in the intended storm model. A smaller first milestone may defer them, but must state the resulting limitations.
- Detachment from MAPLE's actual evolving bed, subject to available sediment, and deposition back into that same bed.
- Sediment sorting, topographic change, boundary export, and explicit water and sediment budgets.
- Prescribed vegetation and surface properties where they affect rain energy, infiltration, roughness, or erosion. Omitting vegetation growth does not mean omitting vegetation's physical effects.

Exclude initially:

- Vegetation growth, mortality, recruitment, dispersal, and vegetation–water feedbacks.
- The full continuous ecohydrological/calendar subsystem, including a port of all daily soil-water and evapotranspiration machinery.
- Dissolved nutrients, sediment-associated nutrients/carbon, and chemistry.
- Individual marker/tracer tracking, all legacy numerical options, and legacy application/input/output compatibility unless a concrete need emerges.

Basic hydrology remains necessary despite excluding full ecohydrology. Specify antecedent moisture and inter-event water handling explicitly. A prescribed moisture condition or simpler drainage/drying treatment is acceptable if its assumptions and budget consequences are clear. The precise infiltration formulation and regime coverage remain design choices, not settled implementations.

## How the extension should fit MAPLE

Use MAPLE's active layer, underlying sediment storage, grain classes, bed exchange, and landscape update framework as the shared sediment authority. Do not maintain a separate MAHLERAN bed that can drift out of agreement with MAPLE.

Water detachment should use current sediment composition and availability, including changes from earlier wind events. Active-layer thickness is a reasonable inventory/exposure representation, but it does not by itself define a physical detachment rate or timescale. Do not assume every wind-specific availability restriction also applies to water.

Implement water-specific hydraulic routing and sediment movement. MAPLE's existing water hooks provide useful accounting and exchange machinery, but are not a complete spatial water solver. Reported face flux alone does not move sediment between cells. Wind transport cohorts and their straight paths cannot simply substitute for converging downhill water pathways.

Retain both transport distance and sediment speed: distance determines where material deposits; speed also determines timing and the mobile load remaining during a storm. A conservative Eulerian mobile-pool formulation is a candidate, not a decided replacement for MAHLERAN. For constant exponential mean travel distance `L` and sediment speed `v_s`, survival over `dt` is `exp(-v_s * dt / L)`. Changing local hydraulics, transport memory, spatial routing, and operator splitting require additional decisions and validation.

Use explicit sequential wind and water events. Before switching, resolve mobile sediment through a defined physical completion policy; never erase it to satisfy an event-boundary check. Distinguish cessation of rainfall from completion of runoff and sediment transport. Preserve any required hydrological state across events, and update drainage after the bed changes.

## Fidelity and conservation rules

Prefer this order:

1. Reuse existing MAPLE infrastructure and directly reusable MAHLERAN code where it fits.
2. Implement the same MAHLERAN physical equations with explicit units and compatible state ownership.
3. Change the numerical method or approximate a relationship when needed for correctness, feasibility, or efficiency, and quantify the consequences.

For every material departure, record the reference equation/routine, why it changed, which physical behavior is retained, and how it is tested. Do not silently replace transport-distance physics with an unrelated transport-capacity law.

Legacy code inspection identified concerns requiring care:

- Wet detachment, splash, and bed/mobile updates use differing timestep and unit conventions. Establish the intended dimensions before translating them.
- Legacy negative-mobile clipping and dry/masked-cell handling can conceal budget inconsistencies under some conditions. Their frequency and magnitude in actual Fortran runs are not yet established. Preserve intended physics rather than copying a suspected bookkeeping error.
- Distinguish mean and median travel distance, and choose whether travel properties are assigned at pickup or updated along the path.
- Pickup and deposition must be limited by their actual source inventories; internal transfers, deposition, and external export must remain distinct.
- Periodic boundaries are not inherently nonconservative. Define water outlets and lateral boundaries deliberately; do not automatically inherit the wind boundary configuration.
- Pits, ponding, overtopping, and changing drainage need an explicit supported policy. Restricted draining terrain is acceptable for early tests, not evidence of general landscape support.

Mass conservation is necessary but does not alone establish physical fidelity. Check erosion magnitudes, sorting, travel distances, and timing as well.

## Benchmarking and implementation strategy

Benchmark against MAHLERAN wherever practical. Begin with matched, controlled cases using the same forcing, geometry, grain properties, and selected physical options. Compare individual equations and kernels before full storms. Where methods differ, compare physical observables and convergence rather than requiring bitwise equality.

Priorities include:

- Water and per-class sediment budgets, supply-limited detachment, and positivity without concealed mass creation/loss.
- Rainfall/runoff response, water depth/discharge, and storm outlet hydrographs.
- Detachment/deposition totals, travel-distance distributions, sediment timing, and exported grain-size composition.
- Sensitivity to timestep, grid spacing, rainfall, slope, roughness, and available sediment.
- Wind→water and water→wind inheritance of the same bed, event completion, topographic updates, and restart consistency.

Use analytic solutions and independent numerical references when a runnable MAHLERAN reference is unavailable. Label those as equation-level tests rather than successful benchmarks against Fortran execution. Differences from MAHLERAN require explanation; agreement with legacy output alone is not sufficient if a budget fails.

Compiled CPU kernels, including a suitable Fortran component or Numba implementation, are possible implementation routes. The production backend has not been selected. Profile realistic work before choosing GPU acceleration or drawing performance conclusions from literal Python loops. Maximize existing-code reuse without forcing incompatible representations together.

## What has actually been demonstrated

`port_feasibility/` contains a small fixed-network water/sediment routing experiment. At its initial verification, 13 tests passed, covering selected routing equations, controlled budgets, an independent numerical reference, and actual MAPLE pickup/transfer/deposition plumbing.

It is not the combined model. It does not yet demonstrate physical detachment/deposition-distance laws, infiltration, splash, changing drainage, a full wind–water sequence, or MAPLE production restart. Its small NPZ checkpoint is only a prototype checkpoint. Plain-Python timings are not compiled performance benchmarks. At the assessment date, Numba and `gfortran` were unavailable in the checked environment; recheck before relying on that status.

The hardest remaining scientific/implementation issue is a complete, conservative pickup–transport–deposition system that retains MAHLERAN-like behavior while exchanging sediment with MAPLE's evolving bed. The existing tests reduce some routing risk but do not resolve that issue.

## Collaboration and background

Codex and Claude are collaborating on assessment and review. Distinguish agreed requirements, candidate designs, implemented behavior, and verified results. Do not describe proposals as completed code or source inspection as empirical validation. The user has explicitly authorized sharing relevant project findings and source excerpts with Claude for requested reviews.

The user considered and rejected adopting the sandpile approach in *Modeling soil-erosion connectivity in drylands using a sandpile framework*, DOI `10.22541/essoar.177013829.90321302/v1`. Keep it in mind for connectivity analysis or future benchmarking, while retaining the process-based direction described above.
