# MAPLE-SYRUP: Sediment Yield, Runoff, and Underground Plumbing

## Accepted Phase 1 source foundation — 2026-11-12

The bounded fixed-bed water/LEAF C1 release uses actual code commits SYRUP
`e40a1b8fc56a96920db4ffbd182d4068fa5de3e8`, LEAF
`52f39fe06166a91e768740b19d3c26d4ffceaf6e`, MASY
`a70400230f26d7931624d49173196a1cfa05a90a`, and historical MAPLE
`9f466d2cfcd52991b24496cf67f4562d6268afa9`. Read
`docs/subsurface_hydrology/phase1_release_qualification.md` and JSON for exact
scope, source packages, actual CPU/GPU/restart/refusal controls and limits.
The dated older design/component statuses below are historical; no source lock
or checkpoint is rebound. New phase work must begin from the accepted source
foundation, preserving live dirty benchmark work and jobs. Documentation-only
commits do not change those qualified code pins.

Phase 1.5 is next ONLY after completion of Phase 1 publication; its binding spec
is `docs/subsurface_hydrology/phase15_daily_recharge.md`. User chose recharge
attempt at P=T. Do not skip it or renumber Phases 2, 2.5,3. Continue automatic
Codex–Claude implementation/review/correction, with per-phase validated commit
and push, asking only unresolved material physics questions. No main merge,
reset, unrelated staging or environment replacement is authorized. Existing
MAPLE `.venv`/`.venv-cupy` only; GPU0 only after fresh availability check.
No wind-with-persistent-soil, moving-soil, field or large-domain readiness is
claimed by the small fixed-geometry qualification.

## Phase 1.5 native daily recharge status — 2026-11-12

Implemented as a SEPARATE native source (N15: package SHA-256
`82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107`, 114 files,
with native MAPLE 9f) and bounded-qualified through Root S4; S5 supplies public
docs/examples/manuals. Whole-phase acceptance is RECORDED (bounded, receipt
`docs/subsurface_hydrology/phase15_acceptance.json`, SHA-256 `4ce37b57f4e009caeb8eb99d30df52ce197bb1c3b4a0947f257b21f9db90deb3`,
status ACCEPTED_BOUNDED_NATIVE_DAILY_RECHARGE); commit and push are PENDING.
Spec `docs/subsurface_hydrology/phase15_daily_recharge.md`; record
`docs/subsurface_hydrology/phase15_qualification.md`. Python API only: no CLI
flag, weather CSV, MASY daily scheduler or LEAF redesign. C1 sources, clocks,
recipes, locks and checkpoints above are preserved, never repointed to N15.
Retained-wet CFL refusal and repeated-storm history limits are visible. Phase 2
(evolving soil, texture-to-property), 2.5 and 3 are unchanged and not started.

## Queued Phase 1.5: daily recharge — 2026-11-12 UTC

After Phase 1 is completed, independently validated, committed and pushed,
implement/qualify/commit/push Phase 1.5 BEFORE Phase 2. Preserve Phases 2, 2.5
and 3. Binding specification: docs/subsurface_hydrology/phase15_daily_recharge.md
and development_phases_20261109.md. Zero rain is a no-op; 0<P<=T attempts small
rain recharge (user explicitly chose equality); P>T uses full native storm.
FIELD CAPACITY triggers gravity movement, SATURATION limits storage; any active
closed-column capacity failure falls back from the unchanged original state
with FULL rainfall. Integrate with one daily water update; no duplicated water,
drainage, E/T or intervals. No matrix-history reset. Audit history semantics
before choosing any new scientific rule. New nominal-calendar/physical-time
separation must be explicitly versioned, never rebound into historical profiles.
Native SYRUP capability only: no full daily MASY scheduler/CSV or LEAF redesign.
This is queued scope, not Phase 1 completion or authorization to skip it.

## Approved bounded wetting-history qualification — 2026-11-10

User chose the bounded approach: retain matrix wetting history by default;
allow explicitly requested resets only between controlled, quiescent test
episodes while preserving actual soil moisture and all physical water ledgers.
Do not infer reset from rain-off, a filename, wind, checkpoint or schedule
boundary. Automatic drying/recovery remains deferred and general repeated-storm
infiltration unqualified. Implement explicit reset safeguards and source-bound
receipts before offering that control; approval does not claim it already exists.
Single-episode component, storm and restart work continues automatically.


User accepted both physics recommendations on 2026-11-10: preserve Tietjen's
separate pre-/post-ponding vegetation placements, and apply the soil E share
after the moisture cap. Debit E only, with LEAF T separately accounting for
actual subsequent withdrawals. These former questions are resolved; implementation
still requires independent review and CPU/GPU qualification. Automatic wetting-history
recovery remains deferred under the approved bounded qualification above.

User added development Phase 2.5 on 2026-11-10: after completing and validating
Phase 2, automatically adapt actual LEAF vegetation fields for the existing
SYRUP/MAHLERAN-derived rainfall splash physics, then proceed to Phase 3.
Follow the full Phase 2.5 requirements in
docs/subsurface_hydrology/development_phases_20261109.md. Preserve plant/bed/water
ownership, explicit cover and source-law conventions, CPU/GPU and bounded-memory
requirements, conservation/restart and no E/T/growth during wind/rain/recession.
This is queued scope; do not claim an implemented vegetation-splash adapter.

User decision, 2026-11-10: Phase1 uses explicit user-supplied hydraulic
properties; texture-to-property method belongs to Phase2. Retain background
preferential infiltration with separate soil receiving-layer weights alongside
LEAF species weights, under one aggregate supply/rate/capacity water budget.
See docs/subsurface_hydrology/development_phases_20261109.md.

User clarification, 2026-11-10: preferential-infiltration target weights belong
to LEAF's species specifications, not root-uptake fractions. Nonnegative weights
are normalized only when their sum is positive; all-zero weights disable that
species' pathway. `[0,1,1]` means equal shares to layers 2/3. SYRUP conserves
the shared supply across all contributions. See the current phase plan.

User authorization, 2026-11-10: share whatever material is needed with
Anthropic/Claude for this project collaboration. This ongoing permission
supersedes the earlier exact-payload approval blocker; do not ask again for
already-covered source sharing. Respect execution controls and exclude
unneeded secrets.

## Active shared scheduler and subsurface program — 2026-11-09

User now explicitly authorized BEGIN of Phase 0–3 in automatic Codex–Claude
mode; older discussion-only/wait notes below are superseded. Codex independently
reviews/tests and commits/pushes each completed validated phase, then continues.
Ask only unresolved material physics questions; preserve unrelated dirty work,
jobs, historical defaults and provenance guards. Governing current plan and
depth-dependent proportional root competition/growth contract:
`docs/subsurface_hydrology/development_phases_20261109.md`. Phase 0 starts in an
isolated MASY candidate under `agent_handoffs/tasks/shared_scheduler_phase0_20261109/`.

## Plant withdrawal ownership clarification — 2026-11-09

LEAF calculates plant water withdrawals. SYRUP owns water inventory and
availability checks, accounts for accepted withdrawals exactly once, and tells
LEAF when supply is insufficient and what amounts were accepted. LEAF uses
the feedback for plant calculations; it does not maintain a second soil-water
authority. Earlier wording assigning actual plant withdrawal calculation to
SYRUP is superseded. This records design intent, not a runtime implementation.

## Interevent coordination clarification — 2026-11-09

Wind, precipitation and interevent intervals are sequential and non-overlapping,
in any order. Evaporation, transpiration and growth run only during interevent
intervals, never during wind/rain or storm recession; do not catch up these
processes over excluded event durations. Calendar time advances throughout.
Storm infiltration, routing and physical water/sediment accounting still run.

MASY is a temporary name for the temporal scheduler/integrator coordinating
MAPLE, SYRUP and LEAF, not a separate physics framework. MAPLE owns landscape
initialization/storage and wind; SYRUP owns physical water; LEAF owns plant
initialization, roots, demand and growth. Reuse MAPLE `location.latitude_deg`
when qualifying PET. Existing dry-reset water handoffs and LEAF's daily-only
driver need explicit extensions for continuous moisture/calendar coupling.
The proposed Phase 0–3 implementation remains deferred pending the user's start
instruction. Preserve existing sources, jobs, checkpoints and qualifications.

## Official SYRUP name — 2026-11-09

The user selected **Sediment Yield, Runoff, and Underground Plumbing** as the
SYRUP expansion. It supersedes the earlier “Uptake by Plants” expansion in
historical notes below. Use the new name in future documentation; preserve
historical reports, receipts and source identities. LEAF stands for
**Landscape Ecohydrology with Agent-based Flora**. This is a naming decision,
not a physics, interface or ownership change.

## Required soil remapping and infiltration feedback — 2026-11-08

Required implementation follow-ups: conservative detailed soil-profile
remapping after erosion/deposition, plus persistent layered soil-moisture
feedback on storm infiltration using MAHLERAN's Smith–Parlange approach.
The existing legacy bucket already includes moisture dependence; layered
coupling is the missing interface. Define the effective infiltration zone,
water carried with added/removed material, properties and LEAF root access
explicitly. Preserve corrected `r2(i,j)`, one water authority, conservation,
net-elevation-only rerouting and CPU/GPU/memory contracts. See the design's
required follow-ups. This is a requirements note, not implemented physics.

## Shared weather and weather-to-PET requirement — 2026-11-08

Required follow-up: implement and qualify weather-to-PET calculation, sharing
temperature, RH, VPD and radiation inputs with LEAF's vegetation/growth forcing.
Support documented simplified inputs and one consistent calendar, units,
provenance and RH/VPD derivation. The detailed requirement is in
`docs/subsurface_hydrology/design.md`; no PET formula or new runtime interface
has yet been selected. LEAF calculates plant withdrawals; SYRUP checks supply
and accounts for accepted withdrawals. This chat returns to subsurface validation; the separate fork handles
MAHLERAN surface validation.

## Subsurface hydrology branch — 2026-10-31

The first isolated column milestone is implemented and independently qualified
on `feature/subsurface-hydrology`: configurable layers, streamed soil intake,
conservative redistribution/drainage, and actual soil evaporation, with NumPy,
Numba and CUDA backends. Read `docs/subsurface_hydrology/qualification.md` for
exact sources, tests and limits. Storm/pond/LEAF integration, transpiration,
restart and moving-ground remapping remain later work; old defaults are intact.
No commit, push or merge was performed. This current status supersedes the
historical design-only status below.

User requested `feature/subsurface-hydrology` for eventual integration into main.
SYRUP will own all physical water in MAPLE–SYRUP–LEAF, including configurable
soil layers, post-rain moisture initialization, inter-event redistribution,
evaporation, transpiration and drainage. LEAF owns vegetation and supplies roots
and plant demand; it must not maintain a second authoritative water budget.
Start from Tietjen hydrology, using SOILWAT layered accounting as a reference.
GPU acceleration and bounded memory remain requirements. See
`docs/subsurface_hydrology/design.md` and the task's source/review records.
The continuous mode must preserve soil moisture across events; old dry-reset
assumptions remain for historical benchmark/default modes until explicit opt-in
integration is implemented and qualified. User authorized branch creation and
automatic Codex–Claude collaboration. Current milestone is design/assessment;
no hydrology kernel, storm run, commit, push or merge is implied by this note.
Preserve prior dirty benchmark work and upstream jobs/source identities.

Later on 2026-10-31 the user explicitly authorized implementation after design
if no material physics question blocks it, and confirmed evaporation is included.
The first bounded column implementation is tracked in
`agent_handoffs/tasks/subsurface_column_20261031/`. This supersedes the
design-only milestone restriction above, without authorizing commit/push/merge
or claiming full storm, vegetation, or moving-bed coupling.


## Corrected MAHLERAN reference required — 2026-10-28

The user confirmed with MAHLERAN's author that `r2(i, 2)` at
`src/Subroutines_Water/infilt.for:71` must be `r2(i, j)`. All new native runs
must use corrected sources and rebuilt binaries, with exact provenance and
existing guards. Use the corrected ledger/capture builds under
`outputs/additional_mahleran_cases/wise_local_rain_index_fix/`. Missing fixed
builds must fail clearly; do not fall back to historical uncorrected builds
or rebind their identities. The patch is in
`benchmarks/additional_cases/patches/mahleran_local_rain_index.patch`.

The user authorized deletion of bug-affected outputs to save disk. Preserve
valid SYRUP CPU/GPU runs, shared forcing/inputs, corrected Fortran runs and
compact deletion/provenance records. Old receipts are historical and must
not be edited to conceal missing outputs. The new seven-case audit finds
Abbott (73 cells) and original Wise (87 cells) affected. Wise and Abbott now have corrected native reruns; see
`docs/additional_mahleran_cases/abbott_index_correction.md`.
Existing Plots 1–4 and RFID results
do not need reruns for this bug (no branch-choice difference in the plots;
RFID's infiltration model 1 skips this code). Future runs still use the fix.

## Merged MAPLE shared-storage qualification — 2026-10-19

The accepted uncommitted SYRUP candidate based on `b4ee14e5` qualifies actual
MAPLE `39bcf79e1e67b6ed960f78a923ddfedf1d173e68`. The exact runtime pair and
fixed reconstruction recipe are in `dependencies/maple_shared_storage/qualification.lock.json`;
read `docs/maple39_shared_storage_qualification.md`. Preserve the historical
water recipe and all source/storage/checkpoint guards. New admission helpers
are explicit caller checks requiring real reserves; existing events do not gain
automatic memory protection. No physics/default/threshold/tolerance changes,
checkpoint rebinding, windowed wind or coupled MASY acceptance are implied.
The bounded task leaves changes uncommitted; no new simulation or push is inferred.

## Direct splash and depression overflow — 2026-10-17

User authorized direct rainfall splash and real flats/depressions/filling/overflow, with small Chastre subset testing. The explicit `maple_syrup.pond_event.PondEvent` integrates local-inertial hydraulics with implicit Darcy–Weisbach friction, MAHLERAN-derived splash and wet laws, multi-face characteristic transport, native evolving MAPLE dense bed, conservative completion and source-bound restart on CPU/GPU. This supersedes historical splash deferral for that opt-in path only; legacy replay, method-5 and no-splash defaults are preserved.

Read `docs/splash_and_overflow/README.md`, `qualification.md` and `acceptance.json` for exact coverage, imported source pair and scientific/performance limits. Dense square-cell bed with avalanching disabled; explicit real positive-drop normal-flow outlets. Windowed integration, stage/backwater boundaries and fused multi-face GPU kernels remain follow-ups. A small bounded wet Chastre integration is not full-catchment/storm-completion or speedup proof. Native MAPLE supply/accounting, net-elevation-only geometry refresh, explicit pond/mobile completion and water reset accounting remain mandatory. No new commit/push is authorized by this task.

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

## Fixed-terrain benchmark requirement — 2026-09-30

User selected the initial MAHLERAN/SYRUP comparison with direct dry-cell splash
disabled in an isolated MAHLERAN copy, and elevation AND routing fixed in both
models. Continue conservative MAPLE sediment-inventory/availability changes;
retain rain-assisted wet detachment. This is benchmark-only, not a change to
normal evolving-terrain SYRUP. See docs/phase7/fixed_terrain_benchmark.md.
Subsequent user instructions authorized full benchmarking and optimization.
See docs/phase7b/acceptance.md and docs/phase7c/optimization.md for executed
results and remaining scientific/performance limitations.

## Default run workflow — Phase 7f, 2026-10-01

All upcoming runs default to the Python/Numba MAHLERAN LEGACY replay
(`maple_syrup.legacy_experiment`, `maple-syrup-legacy`, and the generic
`maple-syrup-benchmark` with its new default `--transport-scheme legacy`).
The multi-bin characteristic model is retained and explicit
(`--transport-scheme characteristic`); its convergence and performance work is
Phase 7g. The legacy replay has fixed composition, unlimited supply, explicit
clipping-source accounting and no evolving MAPLE bed: do not present it as a
conservative complete-event, restart or wind-handoff model, and do not integrate
its legacy errors into MAPLE authority. See docs/phase7f/default_workflow.md.

## Conservative water completion — 2026-10-08

User authorization: complete evolving MAPLE bed integration, memory-bounded CPU/GPU exchange, full GPU storm completion/topographic updates/restart, and performance/physics qualification in automatic Codex–Claude mode. See docs/conservative_water/implementation.md and agent_handoffs/tasks/conservative_water/. The Fortran rain-off recession-water surplus is a documented reference limitation only; do not spend this program investigating its cause or copy it into SYRUP. Preserve legacy replay as the default benchmark workflow; conservative production uses actual evolving MAPLE inventory with no artificial clipping source. Wind handoff, direct dry-cell splash, vegetation growth, full ecohydrology/ET and nutrients remain outside the initial water-component acceptance. Use existing MAPLE .venv and .venv-cupy only; GPU0 is the user allocation, check availability before launches and preserve other devices/jobs. This implementation request does not authorize a new commit/push.

## Net-elevation hydraulic routing — 2026-10-13

User direction: water rerouting must be triggered only by net elevation change. Measure per-cell net displacement from the surface used for the last actual hydraulic routing build, with the maximum absolute displacement over active cells. Gains and losses in one cell may cancel; different cells must not cancel. Reuse the actual MAPLE elevation trigger (currently 0.00075 m, comparison `>=`) unless a separately justified configuration changes it. Time, gross turnover, reporting, checkpointing, storage publication and ordinary/final forced accounting commits alone must not reroute. Keep water and sediment exchange at physical timesteps; a held routing surface is not a held sediment inventory. Preserve the routing source explicitly across restart and validate its graph/source/terrain consistency. Net threshold crossing between ordinary accounting commits must still trigger a genuine authoritative bed commit and reroute.

Storage optimization must preserve every exact changed sediment value and one MAPLE bed authority. Separate physical exchange, bed accounting, hydraulic geometry refresh and persistence. Do not drop tiny changes by tolerance, silently raise resource limits, discard old referenced state, or infer full-storm fit from partial-run averages. See docs/conservative_water/net_elevation_routing.md for the qualified candidate and actual experiment status; this instruction does not imply adoption of an isolated candidate.


## Windowed pond storage — 2026-10-17

User requested commit/push of the accepted splash/depression milestone (done at `e519ea62`), then connection to native windowed storage. The opt-in `PondEventControl(bed_storage="windowed")` now connects paired native wet/splash exchange, staged accounting, net-only hydraulic refresh, bounded journal publication and source-bound resident-window checkpoints. See [docs/pond_windowed/README.md](docs/pond_windowed/README.md) and its qualification record for exact executed status and limits. Legacy benchmark/method-5 defaults stay unchanged. The earlier initial splash deferral and dense-only pond limitation are superseded for this explicitly selected event. Phase arrays remain dense; no whole-Chastre coupled memory fit or new wind implementation is claimed. Original limits, actual MAPLE authority and net-elevation-only routing still apply. No post-integration commit/push is inferred from the preceding commit/push request.
