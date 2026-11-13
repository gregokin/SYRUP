## Phase 1.5 completed and published — 2026-11-12

The accepted native daily recharge source and both manuals are committed and pushed on `feature/subsurface-daily-recharge`, source commit `06952dbc2f984af8fb75dbf09a9c9f7fdbba23fc`, package `82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107` (114 files). Read `docs/subsurface_hydrology/phase15_acceptance.json`, `phase15_qualification.md` and `phase15_publication.json` for exact controls/limits. This supersedes older pending-status notes below. Continue Phase 2 automatically, then 2.5 and 3; unresolved material physics decisions require user input, independent engineering continues meanwhile. Historical C1 clocks/locks/checkpoints are unchanged; no main merge, large-domain or field acceptance.

# SYRUP project instructions

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


## Approved subsurface physics choices — 2026-11-10

The user accepted both recommendations: retain Tietjen's separate vegetation
placements before/after ponding in persistent Green–Ampt infiltration, and
apply the soil-evaporation share AFTER limiting the combined potential soil
loss by current withdrawable water. SYRUP debits actual E only; LEAF requests
T separately against the remaining available water. Do not ask these again
or revert to one vegetation-scaled conductivity or potential-level E share.
See the current development_phases document and task physics-decisions record
for scope and independent qualification. These are approved requirements,
not acceptance of an untested implementation or an automatic wetting-history
recovery/reset policy.

## Added subsurface development Phase 2.5 — 2026-11-10

User added Phase 2.5 between evolving-soil Phase 2 and observational-validation
Phase 3. Once Phase 2 is completed and validated, proceed automatically with
Codex–Claude implementation of an adapter from actual LEAF vegetation to the
vegetation inputs of SYRUP's MAHLERAN-derived splash physics. LEAF owns plants,
SYRUP owns splash/water/sediment transport, MAPLE owns the bed. Preserve explicit
cover/overlap conventions, source-law choices, CPU/GPU equivalence, bounded
memory, conservation, generation-bound restart and event nonoverlap; no growth
or E/T during wind/rain/recession. Do not silently change legacy vegetation
kinetic-energy formulas or infer canopy cover from roots. Read the detailed
Phase 2.5 requirements in docs/subsurface_hydrology/development_phases_20261109.md.
Commit and push each completed validated phase under the user's current
program authorization. This phase is queued, not yet implemented.

User decision, 2026-11-10: Phase1 uses explicit user-supplied hydraulic
properties; texture-to-property method belongs to Phase2. Retain background
preferential infiltration with separate soil receiving-layer weights alongside
LEAF species weights, under one aggregate supply/rate/capacity water budget.
See docs/subsurface_hydrology/development_phases_20261109.md.

User clarification, 2026-11-10: LEAF species traits own preferential-infiltration
receiving weights, distinct from root-uptake fractions. Weights are nonnegative
and need not sum to one; normalize positive vectors for allocation, all zeros
disable that species' contribution. `[0,1,1]` divides equally between the lower
two of three layers. Preserve one aggregate supply/capacity-limited SYRUP water
budget; never assume every shrub participates. See the governing phase plan.

User authorization, 2026-11-10: share whatever material is needed with
Anthropic/Claude for this project collaboration. This ongoing permission
supersedes the earlier exact-payload approval blocker; do not ask again for
already-covered source sharing. Respect execution controls and exclude
unneeded secrets.

## Active shared scheduler and subsurface program — 2026-11-09

User said "OK, begin" after approving depth-dependent proportional root uptake.
The prior instruction to wait before implementing Phase 0–3 is superseded.
Proceed in automatic Codex–Claude mode, asking only unresolved material physics
questions and continuing independent work while answers are pending. Commit and
push each completed, independently validated phase, then continue. Preserve all
pre-existing dirty work and active jobs; acceptance/publication is scoped, not
permission to stage unrelated changes. See
`docs/subsurface_hydrology/development_phases_20261109.md` for governing phases,
ownership, event gates and the agreed root competition/growth equations.
Phase 0 starts in an isolated MASY source candidate under
`agent_handoffs/tasks/shared_scheduler_phase0_20261109/`. Existing native source
guards and historical dry-reset profiles remain unchanged until explicit new
continuous-mode qualification. Source-only Claude sharing remains authorized.

## Plant withdrawal ownership clarification — 2026-11-09

LEAF calculates plant water withdrawals using its roots and plant physiology.
SYRUP maintains the authoritative water inventory, checks the proposed
withdrawals against available water, accounts for accepted withdrawals exactly
once, and reports insufficient supply and accepted amounts to LEAF. LEAF uses
that feedback in plant calculations without independently decrementing a second
soil-water store. Do not assign the plant withdrawal calculation to SYRUP.
This clarifies the planned interface; it changes no current runtime behavior.

## Interevent coordination clarification — 2026-11-09

Wind, precipitation and interevent intervals may occur in any order, but never
overlap. Evaporation, transpiration and plant growth occur only in interevent
intervals; they are inactive throughout wind and rain events, including storm
recession. Calendar time still advances during every event. Do not later apply
interevent E/T or growth retroactively to wind/rain durations. Storm infiltration,
routing and water/sediment accounting remain storm processes.

MASY is the temporary name of the shared scheduler/integrator for MAPLE, SYRUP
and LEAF. It coordinates temporal events, adapters and combined restart; native
MAPLE infrastructure initializes the landscape, including imported MAHLERAN
cases, and LEAF infrastructure initializes vegetation. Reuse MAPLE's existing
`case.yaml` `location.latitude_deg` for future PET, with explicit qualification of
its new physical use. SYRUP remains the physical water authority. Continuous
soil-moisture handoffs must be explicitly implemented and qualified; current
dry-reset handoffs do not provide them. The proposed Phase 0–3 development has
not started; the user's instruction to discuss before starting remains in force.

## Official SYRUP name — 2026-11-09

SYRUP stands for **Sediment Yield, Runoff, and Underground Plumbing**, selected
by the user. This supersedes the earlier “Uptake by Plants” expansion wherever
historical scope notes use it. Use the new expansion in future documentation.
The companion vegetation model is **LEAF — Landscape Ecohydrology with
Agent-based Flora**. Preserve historical reports, receipts and source identities;
this naming decision changes no physics, interfaces or ownership.

## Required soil remapping and infiltration feedback — 2026-11-08

User requires future implementation of detailed, conservative soil-profile
remapping after erosion/deposition, and soil-moisture feedback on storm
infiltration using MAHLERAN's Smith–Parlange approach. The legacy bucket already
has moisture feedback; the missing work is connecting the persistent layered
state as the authority. Define its effective infiltration depth explicitly,
retain corrected `r2(i,j)`, saturation return and water accounting, and do not
silently substitute whole-profile moisture for the MAHLERAN bucket. Remapping
must account for layer geometry/properties, stored water, added/removed material
and LEAF root access. See `docs/subsurface_hydrology/design.md`. These are
required follow-ups, superseding any suggestion of permanent deferral; this
request records scope and does not implement or qualify the interfaces.

## Shared weather and weather-to-PET requirement — 2026-11-08

Weather-to-potential-evapotranspiration (PET) calculation is required future work.
SYRUP and LEAF must consume shared weather inputs: temperature, relative humidity
(RH), vapor-pressure deficit (VPD) and radiation, with explicitly documented
simplified forcing supported. Use one source/calendar and consistent units,
sampling, spatial mapping and derived RH/VPD; do not maintain contradictory
weather streams. See the shared-weather requirement in
`docs/subsurface_hydrology/design.md`. This records a requirement, not a selected
PET method or implemented interface. SYRUP retains physical water authority;
LEAF supplies vegetation demand. This chat focuses on subsurface validation;
MAHLERAN surface validation is assigned to the separate chat fork.

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


## Author-confirmed MAHLERAN rainfall index correction — 2026-10-28

The user confirmed with MAHLERAN's author that `r2(i, 2)` in v1.2.3
`src/Subroutines_Water/infilt.for:71` is a typo and must be `r2(i, j)`.
Use the corrected source and rebuilt executable for ALL new MAHLERAN runs.
Do not execute historical uncorrected binaries or silently rebind their
source identities. The exact patch is
`benchmarks/additional_cases/patches/mahleran_local_rain_index.patch`.
Corrected ledger/capture builds and Wise runs are under
`outputs/additional_mahleran_cases/wise_local_rain_index_fix/`; see
`docs/additional_mahleran_cases/wise_index_correction.md` for results and limits.
Keep source and executable guards; missing corrected builds are setup
failures, never a reason to fall back to uncorrected builds.

The user also authorized deleting results affected by the bug to save disk.
Preserve valid SYRUP CPU/GPU results, forcing, inputs and source provenance,
and current corrected Fortran results. Record exactly what was invalidated
and deleted and the space reclaimed. Do not alter immutable receipts to
make deleted reference outputs appear qualified. The seven-case audit
identifies Abbott and original Wise as affected: Wise and Abbott now have corrected native reruns; see
`docs/additional_mahleran_cases/abbott_index_correction.md`.
Plots 1–4 have no differing rainfall branch
choices; RFID uses infiltration model 1 and bypasses this branch. Their
existing results remain usable for this correction. All future executions
nevertheless use the fixed source.

## Merged MAPLE shared-storage qualification — 2026-10-19

The accepted uncommitted SYRUP candidate based on `b4ee14e5` qualifies actual
MAPLE `39bcf79e1e67b6ed960f78a923ddfedf1d173e68`. Use the separate fixed recipe
and exact source pair in `dependencies/maple_shared_storage/qualification.lock.json`;
read `docs/maple39_shared_storage_qualification.md` before adopting this update.
The old water recipe remains historical reproduction, never a silent fallback.
The new storage-admission helper is explicit opt-in, requires caller reserves,
and is not automatic event/CLI protection or a whole-process memory bound.
Existing checkpoint identities must not be rebound. Physics, routing threshold,
conservation tolerances and defaults are unchanged. This qualification does not
accept windowed wind, growth/avalanching/creep, large alternating storms or MASY
adapters. No new commit/push or subsequent simulation is authorized by this task.

## Direct splash and depression overflow — 2026-10-17

User authorized implementing direct rainfall splash and real flats/depressions/filling/overflow with Claude, using a Chastre subset. The explicit `maple_syrup.pond_event.PondEvent` is the integrated alternative: local-inertial hydraulics with implicit quadratic Darcy–Weisbach friction, MAHLERAN-derived splash and wet laws, multi-face characteristic sediment, actual evolving MAPLE dense bed, native conservation and source-bound restart. This supersedes splash deferral for this opt-in path only. Existing legacy benchmark/method-5/no-splash defaults remain unchanged.

Read `docs/splash_and_overflow/README.md`, `qualification.md` and `acceptance.json` for actual test status/source pair and limits. Dense beds, square cells, avalanching disabled; real positive-drop normal-flow outlets only. Windowed beds, stage/backwater boundaries and fused multi-face GPU transport remain follow-ups. Small native Chastre tests are bounded wet integrations, not complete full-catchment storms or performance/scaling proof. Pond suspension/explicit flow-end settlement and post-hydrology splash splitting are documented choices. Preserve net-elevation-only refresh and actual MAPLE supply/accounting. No new commit/push was authorized by this request.

## Current scope clarification — 2026-09-29

The project is now named **MAPLE-SYRUP (Sediment Yield, Runoff, and Uptake by Plants)**. It is a water extension dependent on actual MAPLE infrastructure, not a standalone framework or copied/forked aeolian model. Wind physics stays in MAPLE and MAPLE-SYRUP invokes MAPLE. Use its computational, configuration/file, physical-state and other relevant infrastructure. Earlier suggestions of importing/synchronizing copied wind implementations are superseded. Track dependency provenance and compatibility instead.

First target: import a MAHLERAN case into MAPLE voxels/active layer; implement rainfall, necessary infiltration/subsurface state, depth/flow routing, and conservative wet detachment/transport/deposition. **Splash is deferred.** Full storm evapotranspiration, ecohydrology, plant growth and nutrients are excluded initially. After rainfall and flow finish within justified tolerances, apply a documented dry-again assumption with explicit water reset accounting and conserved sediment. Speed, memory and GPU acceleration remain priorities. See syrup_implementation_phases.md for the proposed phases; subsequent user continuation authorized phased implementation. The user requested a commit after Phase 2 passes independent verification; Codex owns that commit, with no push requested.


## Objective and scope

Build MAHLERAN-like water erosion and transport inside the MAPLE framework, sharing one evolving sediment bed and topography. Wind and water transport occur in separate events, never simultaneously. Each process inherits the sediment composition and landscape left by the other.

Reuse existing code wherever practical. Maintain the exact MAHLERAN physical approach and relationships where possible, feasible, and efficient. Where a departure is necessary, document it, preserve relevant physical behavior, and benchmark the consequences. Exact legacy numerical/output identity is not universally required. Do not copy suspected conservation errors to achieve agreement.

Start without vegetation growth, full ecohydrological feedbacks, or nutrient/carbon transport. Retain basic rainfall/infiltration/runoff hydrology and prescribed vegetation/surface effects needed for storm erosion. Detachment, splash, grain-specific transport distance and virtual velocity, deposition, and export define the intended storm model. Smaller milestones must identify omitted processes and their limits. Define antecedent moisture, inter-event water handling, and event-end mobile-sediment completion explicitly.

The user rejected adopting the sandpile approach. Keep connectivity ideas in mind for diagnostics or comparisons only.

## Locations and context

- Work root: /home/okin/SYRUP.
- MAPLE framework/upstream aeolian development: /home/okin/MAPLE.
- Legacy MAHLERAN reference: /home/okin/MAHLERAN.
- Read claude.md for the scientific brief, water_extension_scope.md for scope qualifications, and port_feasibility/README.md for actual experimental coverage.
- integration_findings.md and direct_port_assessment.md provide source evidence; historical proposals do not override current user requirements.

SYRUP was not a Git repository when this workflow was established. Do not invent a branch or accepted baseline. Before implementation establish an exact revision or immutable scoped pre-edit snapshot/manifest. MAPLE contains unrelated edits and can have active simulations; preserve its working files and jobs. Prefer isolated development here. Changes to reference trees are not part of routine setup or synchronization.

MAPLE's repository instructions apply when working there, but its historical phase numbering and acceptance status are not SYRUP's specifications. User instructions govern this project's code-reuse direction.

## Continuing MAPLE development

MAPLE's aeolian model will continue to improve independently. SYRUP must be structured so those improvements can be brought over throughout development.

- Treat MAPLE as the upstream source of shared/aeolian functionality. Prefer reuse of its modules behind narrow water integration interfaces over copied, diverging wind implementations.
- Record the exact imported MAPLE revision, any included uncommitted patch/snapshot, and SYRUP-specific differences. A HEAD hash alone does not describe a dirty source tree.
- Keep water-specific physics in separate modules and shared-interface changes small and explicit. Avoid unnecessary rewrites of aeolian algorithms or schemas.
- Select a reproducible dependency, checkout, or synchronization arrangement in a bounded design task; the mechanism is not yet chosen. Do not depend on a mutable live checkout for reproducible production runs.
- Before adopting an upstream update, inspect its diff and compatibility with bed ownership, availability, events, diagnostics, restart, CPU/GPU behavior, and performance. Test an isolated candidate against the current accepted version.
- Record local patches/conflicts and resolve them explicitly. Do not overwrite combined water work or unrelated MAPLE changes during synchronization.
- Shared fixes should be organized so they can be proposed back to MAPLE, while adoption there remains a separate scoped action. Do not modify files imported by a running simulation.

## Performance priorities

Speed, memory performance, and GPU acceleration are first-class requirements, not optional later cleanup. Plan array layouts, routing representation, batching, state ownership, backend interfaces, and diagnostics with these requirements in mind from the start.

Reuse MAPLE's backend and GPU infrastructure where suitable. Avoid repeated host/device transfers, hidden synchronizations, full-domain copies, unbounded cohorts/history, and Python loops in production hot paths. Some routing has ordering dependencies: identify these explicitly and evaluate compiled or GPU-compatible algorithms rather than assuming NumPy/CuPy substitution makes them parallel.

Maintain physical equivalence within declared tolerances across supported backends. Benchmark representative grid sizes and class counts for wall time, scaling, CPU/GPU peak memory, transfer costs, and end-to-end event behavior. Separate compilation/startup from steady-state work. Performance improvements must preserve science, conservation, and restart contracts. CPU-only feasibility experiments are useful milestones, not evidence that the GPU requirement has been satisfied. Hardware limitations must be reported honestly.

The production backend and specific acceleration strategy remain design choices. Do not infer full-model speed from literal Python-loop prototypes.

## Scientific invariants and validation

- One sediment authority: MAPLE's active layer and underlying storage, not a separate MAHLERAN bed.
- Pickup uses actual current grain holdings. Requested pickup need not equal actual pickup. Deposition cannot exceed its source mobile mass.
- Conserve water and sediment by class with declared tolerances; distinguish internal transfer, bed exchange, mobile storage, and export.
- Do not erase negative mass, dry-cell inventory, ponded storage, or event-end mobile load to hide failed balances.
- Specify units, physical time basis, class ordering, array shapes, dtype, and state ownership. Reconcile legacy wet/splash timestep conventions explicitly.
- Preserve both transport distance and sediment speed. Distinguish mean/median distances and pickup-time/local hydraulic dependence.
- Real spatial water routing is required; existing MAPLE water bookkeeping and straight wind trajectories are not a hydraulic solver.
- Define boundaries, outlets, pits, terrain rerouting, and wind/water transitions. Periodicity is not inherently nonconservative.
- Validate before mutation; failures must not leave partially committed state. Restart must preserve equivalent continuation.
- Conservation alone does not prove erosion magnitude, sorting, distance, or timing fidelity.

Benchmark against MAHLERAN wherever practical, from selected equations to matched storms. Record source versions, forcing, geometry, options, grain properties, units, tolerances, and calibration differences. Compare runoff, water depth/discharge, erosion/deposition, sediment timing/distance, exported composition, and timestep/grid sensitivity. If Fortran execution is unavailable, use independent analytic/numerical references but do not label those actual Fortran benchmarks.

Existing port_feasibility tests are small fixed-network experiments, not the complete physics, GPU implementation, production restart, or alternating wind–water model. Preserve this distinction.

## Automatic Codex–Claude workflow

User authorization on 2026-09-28: automatic orchestration, scope until_changed, for requested SYRUP work. This supersedes the historical one-off review/manual restrictions. It enables bounded implementation, review, and correction cycles; it does not start an unlimited background job.

Codex scopes tasks, launches Claude, independently reviews changes, verifies results, and records acceptance. Claude implements bounded assignments and corrections and leaves work uncommitted. For substantive Codex-authored implementation, obtain Claude review rather than treating author review as independent.

Only one writer may modify a given implementation tree at a time. Parallel read-only investigation is acceptable. Explicitly assign separate files if delegating nonoverlapping work. Never launch a competing writer because an existing process is slow.

For each task:

1. Read agent_handoffs/orchestration_state.md, current prompt, reports, and governing specifications. Check actual files/processes before resuming.
2. Establish baseline, existing changes, task ID, writable scope, and acceptance criteria. Archive task evidence under agent_handoffs/tasks/<task_id>/.
3. Write current_prompt.md with objective, files, requirements, exclusions, deliverables, and checks.
4. Verify local claude --help options before the first new launch. Launch from SYRUP with explicit tools/directories, ordinary permissions, and separate stdout/stderr. Save command, process identity, session ID, output paths, and status.
5. Claude inspects, implements, verifies, and reports exact results without committing or broadening scope.
6. Codex independently reviews the accumulated diff and relevant callers, runs appropriate verification, and separates confirmed defects from risks/environment limitations.
7. Issue bounded corrections automatically until requirements are met. Ordinary defects or failed tests do not require repeated user permission.
8. Record acceptance and limitations, archive evidence, and update durable state before proceeding.

No commit or push is authorized by the setup request. If separately authorized later, Codex alone commits accepted task files. Do not reset, rebase, destructively clean, overwrite unrelated changes, or silently initialize a repository to fabricate a baseline.

## Permissions and interruption handling

Automatic mode does not authorize permission bypass. Use ordinary permissions and respect execution-platform controls. Sharing relevant findings and source excerpts with Claude was explicitly authorized by the user; do not ask again for already-covered collaboration.

Preserve prompts, reports, raw logs, commands/results, partial work, and session IDs. An explicit usage limit requires recording its real reset time/timezone; resume after reset plus two minutes if execution remains available. Do not call authentication, permission, network, or test failures usage limits. Do not guess reset times or retry indefinitely.

Use short interruptible waits and communicate progress. Resume an exact recorded session, not an ambiguous latest session. Instructions/state files are not a background scheduler. Never promise unattended resumption without a verified live mechanism. If execution stops, record the next action honestly; preserve automatic preference until the user changes it while distinguishing preference from actual process status.

Route Claude questions through Codex first. Ask the user only for material unresolved scientific/scope decisions or genuinely missing authority. Preserve all partial work when switching modes.

## Review standard

Claude's report is context, not proof. Confirm defects using concrete triggering paths, expected/actual behavior, requirements, locations, severity, and preferably reproducers. Inspect units, shapes, boundaries, failure/mutation behavior, conservation, restart, CPU/GPU equivalence, memory scaling, and upstream compatibility as relevant.

Run meaningful checks proportional to the change; documentation setup does not require a full model test run. Report exact commands, outcomes, and unverified requirements. Never present proposals or historical test results as newly verified behavior.

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

User direction: ALL upcoming runs default to the Python/Numba MAHLERAN LEGACY
replay (`maple_syrup.legacy_experiment`, `maple-syrup-legacy`; the generic
`maple-syrup-benchmark` now defaults to `--transport-scheme legacy`). The
multi-bin characteristic model and upwind remain implemented and are selected
explicitly with `--transport-scheme`; their convergence/performance work is
deferred to Phase 7g. This supersedes the earlier default-32-bin selection as a
workflow choice only. Scientific status of the legacy replay is unchanged: fixed
composition, unlimited supply, explicit clipping source, no evolving MAPLE bed.
It is not a conservative complete-event, restart or wind-handoff model, and its
legacy behaviour is not MAPLE authority. No wind integration is claimed. See
docs/phase7f/default_workflow.md and docs/phase7f/performance.md.

## Conservative water completion — 2026-10-08

User authorization: complete evolving MAPLE bed integration, memory-bounded CPU/GPU exchange, full GPU storm completion/topographic updates/restart, and performance/physics qualification in automatic Codex–Claude mode. See docs/conservative_water/implementation.md and agent_handoffs/tasks/conservative_water/. The Fortran rain-off recession-water surplus is a documented reference limitation only; do not spend this program investigating its cause or copy it into SYRUP. Preserve legacy replay as the default benchmark workflow; conservative production uses actual evolving MAPLE inventory with no artificial clipping source. Wind handoff, direct dry-cell splash, vegetation growth, full ecohydrology/ET and nutrients remain outside the initial water-component acceptance. Use existing MAPLE .venv and .venv-cupy only; GPU0 is the user allocation, check availability before launches and preserve other devices/jobs. This implementation request does not authorize a new commit/push.

## Net-elevation hydraulic routing — 2026-10-13

User direction: water rerouting must be triggered only by net elevation change. Measure per-cell net displacement from the surface used for the last actual hydraulic routing build, with the maximum absolute displacement over active cells. Gains and losses in one cell may cancel; different cells must not cancel. Reuse the actual MAPLE elevation trigger (currently 0.00075 m, comparison `>=`) unless a separately justified configuration changes it. Time, gross turnover, reporting, checkpointing, storage publication and ordinary/final forced accounting commits alone must not reroute. Keep water and sediment exchange at physical timesteps; a held routing surface is not a held sediment inventory. Preserve the routing source explicitly across restart and validate its graph/source/terrain consistency. Net threshold crossing between ordinary accounting commits must still trigger a genuine authoritative bed commit and reroute.

Storage optimization must preserve every exact changed sediment value and one MAPLE bed authority. Separate physical exchange, bed accounting, hydraulic geometry refresh and persistence. Do not drop tiny changes by tolerance, silently raise resource limits, discard old referenced state, or infer full-storm fit from partial-run averages. See docs/conservative_water/net_elevation_routing.md for the qualified candidate and actual experiment status; this instruction does not imply adoption of an isolated candidate.


## Windowed pond storage — 2026-10-17

User requested commit/push of the accepted splash/depression milestone (done at `e519ea62`), then connection to native windowed storage. The opt-in `PondEventControl(bed_storage="windowed")` now connects paired native wet/splash exchange, staged accounting, net-only hydraulic refresh, bounded journal publication and source-bound resident-window checkpoints. See [docs/pond_windowed/README.md](docs/pond_windowed/README.md) and its qualification record for exact executed status and limits. Legacy benchmark/method-5 defaults stay unchanged. The earlier initial splash deferral and dense-only pond limitation are superseded for this explicitly selected event. Phase arrays remain dense; no whole-Chastre coupled memory fit or new wind implementation is claimed. Original limits, actual MAPLE authority and net-elevation-only routing still apply. No post-integration commit/push is inferred from the preceding commit/push request.
