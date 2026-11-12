# Shared scheduler and subsurface development program

Current milestone: bounded Phase 1 water/LEAF qualification on actual C1 commits is recorded in [phase1_release_qualification.md](phase1_release_qualification.md). Phase 1.5 (daily recharge) is implemented as a separate native source and bounded-qualified through Root S4; its public record is [phase15_qualification.md](phase15_qualification.md) (bounded whole-phase acceptance recorded; publication pending). The scientific requirements and numbering below remain governing.

User authorized implementation on 2026-11-09 after the planning discussion.
Work uses automatic Codex–Claude implementation, independent review, correction
and verification. Commit and push each complete validated phase and continue.
Subtasks may establish bounded milestones without claiming an entire phase is
finished. Preserve prior dirty work, active jobs, source locks and historical
scientific qualification limits. Ask the user for unresolved material physics
decisions, while progressing independent tasks.

## Ownership and scheduling

MAPLE owns native landscape initialization and authoritative active-layer/voxel
sediment, including cases imported from MAHLERAN, and runs wind transport.
SYRUP owns all physical water and authoritative water accounting. LEAF owns
vegetation initialization, roots, plant withdrawal calculations and growth.
The coordinator, temporarily named MASY, owns temporal scheduling, event
adapters and combined restart; it does not replace native case initialization
or copy any model's physics. Display branding does not require renaming imports.

Wind, precipitation and interevent periods may occur in any order and never
overlap. Evaporation, transpiration and growth run ONLY during interevent time,
not during wind/rain or storm recession. Calendar time advances throughout;
there is no retrospective E/T or growth over excluded event durations.
Infiltration, necessary storm drainage/redistribution, surface routing and
sediment exchange remain storm processes. Rainfall ending is distinct from
water-event completion. Never silently shift fixed-calendar forcing to hide
unfinished transport or reset persistent soil water at a handoff.

## Plant withdrawals and growth: agreed contract

For one timestep, plant p requests total water volume D_p. LEAF distributes it
as R_pcl = D_p * w_pcl across rooted cells c and soil layers l, where w_pcl is
the plant's normalized root contribution and sum_cl(w_pcl) = 1. Species traits
and each plant's actual spatial roots determine these weights. Use water volume
for competition; native column depths require explicit cell-area conversion.

SYRUP reports available withdrawable water A_cl for each cell/layer, enforcing
its declared water bounds. LEAF computes total local demand Q_cl = sum_p R_pcl.
For Q_cl > 0 the common fulfillment fraction is f_cl = min(1, A_cl / Q_cl),
and each plant receives U_pcl = R_pcl * f_cl. For Q_cl = 0, withdrawal is zero.
There is no species priority or order-dependent first access. All roots within
one cell/layer share the same fulfillment fraction; total delivered water must
never exceed that cell/layer's available water.

SYRUP validates proposed withdrawals, debits the authoritative inventory exactly
once, and reports accepted amounts and shortfalls. LEAF uses accepted amounts
in plant calculations, never a second authoritative soil-water store. A changed
availability snapshot must not allow stale requests to overdraw water.

For D_p > 0, water-supported growth fraction is
S_p = sum_cl(U_pcl) / D_p = sum_cl(w_pcl * f_cl). Apply it to otherwise planned
aboveground growth, subject to other growth constraints. It is not the minimum
fulfillment fraction of any one root location. Zero-demand handling must be
explicit in the plant interface, without division by zero or invented growth.

Assume no root plasticity and no compensatory redistribution of unmet requests
to wetter cells/layers. A shallow-rooted plant can be water-limited while deeper
water remains available to another plant. One plant spanning several cells is
weighted by its roots across all those cells as well as layers.

User example: D=100, layer weights 0.5/0.5, available water 25/200. Requests are
50/50, accepted withdrawals 25/50, remaining water 0/150. Water-supported growth
is 0.75 of planned growth. It cannot take 75 from the second layer to compensate
for the first. These are physical water amounts; water must close independently
of the dimensionless growth factor.

## Phase 0: shared temporal scheduler and weather

Extend actual MASY infrastructure with reusable wind, rain and interevent files
referenced by ordered scenarios. Preserve owning-model wind/rain conventions;
wind file groups follow native MAPLE case records. Add explicit calendar dates,
source-bound forcing, nonoverlap checks, realized recession/settlement time,
event cursor and combined restart semantics. Landscape/plant initialization
remain with MAPLE/LEAF. Preserve old profiles rather than rebinding them.

Provide shared temperature and weather/PET records for SYRUP and LEAF, supporting
documented simplified inputs. Explicit PET still requires temperature for plant
processes. Hargreaves is the first computed PET method, with a hook for richer
methods such as Penman–Monteith. Reuse MAPLE case.yaml location.latitude_deg;
its first physical use needs explicit validation/identity, no equator default.
Terrain adjustment must be traced to reference equations and source versions.

Existing interfaces need explicit bridging: MASY's current dense handoff resets
soil water, while LEAF's coupled interface accepts whole interevent days only.
Do not label stub dispatch, a metadata checkpoint or refusal of missing native
adapters as acceptance of a continuous three-model simulation. Dates continue
through all phases; interevent biology never catches up across excluded events.

### Delivery boundaries identified during the native-interface audit

Phase 0 may publish independently qualified scheduler/weather foundation
submilestones, but those are not acceptance of the entire requested continuous
three-model run. The native interevent qualification must exercise actual
SYRUP column updates against independent direct controls, with real CPU/GPU
accounting and source identities; mechanical dispatch tests alone are not enough.
Existing separately locked wind/rain execution remains a historical dry-reset
profile until a newly qualified adapter explicitly supports the new state.

Persistent rainfall-to-layered-soil handoff depends on Phase 1: current native
SYRUP storm and pond paths integrate a single soil bucket and explicitly remove
soil water at completion. Layered infiltration and moisture feedback must run
inside accepted storm hydraulic steps. Replaying old intake into another store
post hoc, or marking the old handoff continuous, is not an implementation.

Combined physical restart remains open until native soil, plants and process
state have source-bound codecs; the Phase 0 temporal cursor is only metadata.
The delivery record must distinguish scheduler/calendar, shared PET, actual
interevent soil integration, native wind/rain bridges, LEAF calendar/growth
bridge, persistent storm handoff and combined restart. Any item not yet tested
remains explicitly open; a bounded submilestone does not close that item.

## Phase 1: Tietjen-inspired layered hydrology

User decision, 2026-11-10: hydraulic properties are explicitly user-supplied in
Phase 1. Compute and retain the actual overlap/mass-weighted texture from MAPLE,
but do not silently infer conductivity, suction or retention thresholds from
it. A declared texture-to-hydraulic-property method is required Phase 2 work,
alongside conservative erosion/deposition remapping.

User clarification, 2026-11-10: preferential-infiltration receiving weights are
species-specific traits owned by LEAF, separate from root-uptake fractions.
Each species supplies one finite nonnegative weight per configured soil layer;
the weights need not sum to one. A positive vector is normalized for allocation:
`[0,1,1]` sends equal amounts to layers 2 and 3. An all-zero vector such as
`[0,0,0]` disables that species' preferential-infiltration contribution, without
division by zero. Do not infer that all shrubs participate or that a plant's
root-uptake distribution is its preferential-flow distribution. Species without
the pathway contribute no preferential intake. The two-layer reference can use
`[0,1]` for an enabled shrub and zero weights for other species. Shared surface
water, the declared maximum intake and receiving-layer capacity must still
bound the aggregate intake; never allocate the same water independently to
multiple species. SYRUP owns physical intake and water accounting; LEAF owns
the species traits and spatial vegetation contributions. Do not silently invent
a receiving distribution when the trait is missing.

User decision, 2026-11-10: retain the reference background preferential
infiltration pathway, including when vegetation is absent, with its own
explicit user-defined soil receiving-layer weights. These are not LEAF root
weights or species traits. Species weights remain separate, and an all-zero
species vector disables only that species' contribution. Account for background
and vegetation enhancement within one aggregate surface-water supply, maximum
intake and receiving-capacity bound; never spend the background allowance anew
for each species. Document how the reference background-plus-vegetation response
is partitioned among receiver vectors, preserving the two-layer reference limit.

Inspect downloaded GitHub and both Dryad EcoHyD sources alongside Tietjen
papers; these are later descendants, not verified original HydroVeg releases.
Document equation/source differences and classify likely errors versus version
changes before adopting a departure. New mode follows Tietjen Green–Ampt matrix
infiltration with vegetation and upper-layer moisture feedback, shrub-dependent
bounded preferential deep intake, field-capacity drainage, optional bottom
drainage, conductivity/depth/calibrated-coefficient moisture-gradient exchange,
and moisture/vegetation-dependent evaporation without duplicate transpiration.
Historical MAHLERAN Smith–Parlange comparison paths remain unchanged.

User decisions, 2026-11-10: preserve Tietjen's separate vegetation placements
in the pre-ponding onset rule and post-ponding intake, rather than replacing
them with one vegetation-scaled conductivity throughout. Preserve persistent
Green–Ampt history and declare the numerical extension from hourly source
calculations. For soil evaporation, first limit the combined potential soil
loss by current withdrawable soil water, then apply its evaporation share.
Debit only actual evaporation; LEAF requests transpiration separately from the
remaining available water. Never debit the reference combined ET and then
debit LEAF transpiration again. Both choices are approved requirements;
implementation and qualification remain separate milestones.

User decisions, 2026-11-10: production matrix infiltration uses a persistent
cumulative Green–Ampt wetting front rather than repeating the descendant's
hourly Peschke approximation at every hydraulic substep. Retain the Tietjen
vegetation/moisture effects and explicit hourly reference comparisons; qualify
timestep convergence and persist front/reset state for continuation. This is
a declared numerical/physical extension, not exact EcoHyD result identity.

Preferential flow uses the equivalent continuous rate of the reference hourly
fraction beta: k=-log1p(-beta)/3600 s^-1, isolated removal
H*(-expm1(-k*dt)). Retain the shared maximum intake rate, surface supply and
receiving-capacity bounds, with explicit beta=0/1 handling. This preserves the
isolated uncapped one-hour response; concurrent rain, matrix flow and routing
still require timestep qualification. Never apply the same hourly fraction
anew at every shorter hydraulic timestep.

Keep user-defined layer count/thickness and explicit water inventories. Derive
texture from overlap-weighted actual sediment masses in shared MAPLE voxels and
active layer, counting the active layer once in the top soil layer; handle void
and partially filled top material explicitly. Geometry/properties are fixed
within a Phase 1 run. Preserve SYRUP's exact adjacent-pair relaxation applied
sequentially, variable timesteps, bounded conservative transfers and explicit
intake, return, drainage, evaporation and transpiration accounting.

LEAF calculates withdrawals under the contract above, with SYRUP enforcing
availability/accounting. Qualify NumPy, compiled Numba CPU and CUDA GPU paths.
GPU acceleration, bounded memory, device residency, conservation and restart
are requirements, not claims inferred from a small CPU example.

## Approved bounded wetting-history qualification — 2026-11-10

User chose the bounded approach: retain matrix wetting history by default;
allow explicitly requested resets only between controlled, quiescent test
episodes while preserving actual soil moisture and all physical water ledgers.
Do not infer reset from rain-off, a filename, wind, checkpoint or schedule
boundary. Automatic drying/recovery remains deferred and general repeated-storm
infiltration unqualified. Implement explicit reset safeguards and source-bound
receipts before offering that control; approval does not claim it already exists.
Single-episode component, storm and restart work continues automatically.

## Phase 1.5: daily small-rain recharge and full-storm fallback

Added by the user on 2026-11-12 UTC. Complete, independently validate, commit
and push Phase 1 first. Then implement, qualify, commit and push Phase 1.5
before starting Phase 2; numbering and requirements of Phases 2, 2.5 and 3
remain unchanged. Read the full binding
[Phase 1.5 specification](phase15_daily_recharge.md) before work. Status: the
native API is implemented and bounded-qualified (see
[phase15_qualification.md](phase15_qualification.md)); the bounded whole-phase acceptance is recorded in
[phase15_acceptance.json](phase15_acceptance.json); commit and push are pending, and Phase 2 has not started.

Provide a narrow native SYRUP rainfall-selection/recharge interface, not a
complete daily MASY scheduler, CSV format or LEAF redesign. Caller supplies P
(mm), I (mm/hour) and T (mm); no threshold default or extra intensity threshold.
Zero rain is a no-op; **0 < P <= T attempts simplified recharge** (equality
explicitly selected by the user); P > T uses an actual full native storm of
3600 P/I seconds. Positive rain requires positive intensity. Full-storm
fallback uses the unchanged original state and FULL rainfall when any active
closed-bottom column cannot accommodate recharge after downward redistribution.
Gravity movement starts above FIELD CAPACITY, storage is bounded by SATURATION,
and a field-capacity profile still has storage. Coordinate addition/redistribution
before premature upper-layer saturation return; preserve open/closed bottom rules.

Combine recharge with ONE daily water update without extra subdaily interevent
steps, duplicate drainage/E/T/time or invented positive dt. Preserve masks,
actual bed, persistent moisture and non-reset matrix history, accounting,
parent/generation/occurrence binding and explicitly versioned restart/source
checks. Audit cumulative-intake/history semantics and ask only if a new scientific
rule is needed. Qualify NumPy, compiled CPU and CUDA with independent controls,
real native fallback, discarded-trial identity, subsequent storms and restart.

The future daily contract orders listed wind events, then rain, then daily
interevent hydrology/T and daily-or-coarser growth. Wind/full rain/recession add
physical time BETWEEN nominal days; nominal dates and native physical clocks
are separate. E/T/growth never run during transport. Document this approximation
as a NEW explicit contract, preserving historical dated clocks/locks. Implement
only the bounded native capability in this phase; retain full-storm completion
rules, device residency, memory limits and scientific defaults.

## Phase 2: evolving soil geometry and properties

Conservatively remap layers, water, texture and hydraulic properties following
erosion/deposition; connect remaining changes in MAPLE sediment to the water
profile and LEAF root access. Account for material/water added, removed or
returned, avoid duplicate active-layer mass, preserve one authority and native
net-elevation-only hydraulic rerouting. Qualify restart and CPU/GPU behavior.

Implement and qualify the texture-to-hydraulic-property method explicitly
deferred here by the user on 2026-11-10. Declare its scientific source, required
inputs, calibration/domain and treatment of coarse or unmapped grain classes;
do not treat mass-weighted texture as a unique hydraulic parameterization.

## Phase 2.5: LEAF vegetation inputs for rainfall splash

Added by the user on 2026-11-10. After Phase 2 is complete and validated,
begin this phase automatically under the same Codex–Claude workflow; complete,
validate, commit and push it before proceeding to Phase 3. This is a queued
implementation requirement, not an implemented or qualified adapter.

Adapt LEAF's individual-plant/species and spatial vegetation representation
into the vegetation fields required by SYRUP's MAHLERAN-derived splash
detachment/transport laws. LEAF remains the plant authority, SYRUP owns rainfall,
water and sediment exchange, and MAPLE remains the sole sediment-bed authority.
Reuse the existing splash physics and actual MAPLE supply/availability limits.
Do not introduce a copied vegetation model or a second sediment inventory.

First inventory the exact vegetation inputs read by the current splash and
rain-impact laws, including units and cover conventions. The current
`sediment_physics` rain-impact interfaces accept a cell vegetation-cover
fraction; the pond event binds a cover field. Establish how actual LEAF plant
footprints, species contributions and overlapping plants produce these fields,
and which additional traits are genuinely required by the selected source law.
Keep canopy/ground cover, root distributions and preferential-infiltration
receiving weights distinct. Do not sum overlapping cover beyond one and silently
clip it, or assume root weights are splash protection. Ask only unresolved
material physics questions about the mapping or source laws.

The existing kinetic-energy vegetation modes must remain explicit: the legacy
literal expression has counterintuitive cover dependence, already documented
in `sediment_physics.py`. Audit source correspondence and document the selected
mode; this phase does not authorize silently correcting that relationship or
changing historical benchmark defaults. Qualify an adapter against identical
prescribed vegetation fields and source equations before interpreting coupled
vegetation effects.

Prepare bounded per-cell maps from actual LEAF state and refresh them when its
relevant accepted vegetation generation changes. Plants do not grow and E/T do
not run during wind/rain/recession, so event-held vegetation fields must have
an explicit source/generation identity. Preserve calendar nonoverlap, native
conservation and transaction/restart checks. CPU and GPU paths must agree under
declared existing tolerances, with bounded memory and no dense plant-by-cell
tensor or per-hydraulic-step reconstruction of plant footprints. Test bare soil,
isolated plants, mixed species, overlapping footprints, masks and restart;
distinguish field mapping, splash response and complete coupled-event evidence.

## Phase 3: observational validation design

Evaluate shallow observed JORN and MOAB profiles for validation: declared layer
geometry, sensors/quality, initialization, lower boundary, weather/PET,
evaporation and vegetation/root treatment. Plan rain-free intervals first and
later precipitation-driven tests when forcing is adequate. Distinguish fit from
independent validation and physical parameter inference. Ask needed scientific
questions and propose a defensible comparison before interpreting a fit.


## Confirmed first coupled hydrology cover — 2026-11-11

The user chose to retain prescribed hydrology vegetation-cover and species-infiltration maps for the first coupled qualification. LEAF-driven hydrology cover feedback is added to Phase 2.5, alongside the shared vegetation projection for splash. Plant growth does not silently change those prescribed hydrology maps before that follow-up. The subsequent answers resolve plant PET and disabled-growth credit as recorded below.

## Confirmed plant PET and disabled-growth accounting — 2026-11-11

Use terrain-adjusted PET averaged by canopy area for each plant’s potential demand, followed by unchanged root-weight allocation. Extend LEAF’s PET interface with bounded per-plant arrays and sparse canopy geometry; do not change root allocation or create dense plant-by-cell tensors. During interevent periods with transpiration enabled and growth explicitly disabled, withdraw and account for actual transpiration but give those hours no later biomass-growth credit. Evaporation, transpiration and growth remain excluded from wind, rain and their recession periods.
