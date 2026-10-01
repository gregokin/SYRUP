# SYRUP project instructions

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
