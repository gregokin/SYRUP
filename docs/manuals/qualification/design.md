# Subsurface hydrology design

> Current status: the bounded native Phase 2 APIs are implemented, independently qualified and integrated into main at `6f8bab2ab045a3e6299b33ad2a3b0fc170ab4f06`. See [phase2_qualification.md](phase2_qualification.md) and [phase2_publication.json](phase2_publication.json). The design and implementation milestones below retain their original historical status wording; they do not describe current release readiness.

## Historical Phase 2 design — 2026-11-13

Soil layers float with the evolving surface (fixed user thicknesses), preserving per-layer water mass/volume through geometry reassignment; there is no geometry-only erosion water sink in this treatment, which supersedes earlier eroded-soil-water-disappearance proposals below (kept as history). The native storm already runs gravity/redistribution every hydraulic step, so there is no rain-off reset. A pure property update can leave new SAT below retained water, which the standard step refuses before its cascade; the user-approved staged treatment reuses the gravity cascade on the NEW field capacity (open bottom exports once; closed bottom returns only unavoidable excess to surface water), as design only, with implementation and qualification pending. User-resolved (2026-11-13): affected columns refresh at the transport-event handoff with properties held during an event; the full voxel-grid bottom is fixed, flat and impermeable (only actual soil stores water; open lower boundary only for a profile ending above rock). See [phase2_completion_architecture.md](phase2_completion_architecture.md). Green–Ampt history is retained and its behavior with refreshed parameters is a qualification item. Full statement: [phase2_floating_profile_design.md](phase2_floating_profile_design.md).

## Current Phase 1 status — 2026-11-12

The opt-in fixed-bed layered water and LEAF calendar is now bounded-qualified
on actual committed sources: SYRUP `e40a1b8`, LEAF `52f39fe`, MASY `a704002`,
and explicitly historical MAPLE `9f466d2`. See
[the release qualification](phase1_release_qualification.md) and the current
[parameter reference](parameter_reference.md). NumPy, Numba and CUDA complete
the small native example with equivalent state and exact same-backend restart.
The original component/design sections below retain their dated scope; their
statements that later integration is pending are historical, not a description
of the qualified release. No wind, evolving-soil, field or scaling acceptance
is implied. Phase 1.5 daily recharge remains the next implementation, after
Phase 1 publication, before Phases 2, 2.5 and 3.

Phase 1.5 daily recharge status, 2026-11-12 UTC: implemented as a **separate native source (N15)** in the candidate and independently accepted by Root through
S4 (bounded native daily lifecycle, CPU/CUDA, V3 physical restart); a bounded whole-phase acceptance is **recorded** (`phase15_acceptance.json`); publication is **complete** (source commit `06952dbc2f984af8fb75dbf09a9c9f7fdbba23fc`). See
[phase15_qualification.md](phase15_qualification.md) for the exact source (package `82e9c011…`, 114 files), controls and limits. The user rule is unchanged: P = 0 is a
no-op, 0 < P <= T attempts simplified recharge (equality included), P > T uses the native full storm, with whole-occurrence fallback from the unchanged parent. It is a
Python API (`maple_syrup.daily_water`), not a CLI option or weather/CSV reader; the nominal ordinal and native physical seconds are an explicit new versioned
contract that does not retrofit historical clocks. The C1 source and its dated clocks, recipes, locks and checkpoints are preserved and are never repointed to N15.
Phases 2, 2.5 and 3 remain queued with their numbering and requirements.

Current status, 2026-11-10: the dated scheduler/shared weather foundation and
actual interevent column checks are implemented and independently qualified.
See [Phase 0 qualification](phase0_qualification.md) and the governing
[Phase 0–3 program](development_phases_20261109.md). The planning-only/wait
statements and unselected-PET notes below are historical and superseded by
that program. Phase 1 selects persistent Green–Ampt, continuous preferential
flow, explicit hydraulic properties and Tietjen-inspired E/redistribution;
these new storm laws are not implemented by the Phase 0 foundation.


SYRUP means **Sediment Yield, Runoff, and Underground Plumbing** (user-selected
2026-11-09). LEAF means **Landscape Ecohydrology with Agent-based Flora**.
The previous SYRUP expansion in historical records is superseded.

## Plant withdrawal ownership clarification — 2026-11-09

LEAF calculates plant water withdrawals from its root connections and plant
physiology. SYRUP maintains the authoritative water inventory, checks those
withdrawals against available water, records accepted withdrawals exactly once,
and reports accepted amounts and any shortfall to LEAF. LEAF uses this feedback
for its plant calculations without independently debiting soil water. Water
availability enforcement belongs to SYRUP; plant withdrawal calculation belongs
to LEAF. This clarifies the planned interface and supersedes earlier ownership
wording; it does not change or qualify the existing coupled implementation.

## Current scheduling clarification — 2026-11-09

The shared coordinator (temporarily named MASY) manages temporal scheduling and
integration of MAPLE, SYRUP and LEAF. Landscape initialization uses native MAPLE
infrastructure, including imported MAHLERAN cases; vegetation initialization
belongs to LEAF. MAPLE already provides optional `location.latitude_deg` in
`case.yaml`; future PET must explicitly consume and qualify it, and must not
treat an absent latitude as zero. It is currently descriptive MAPLE metadata.

Wind, precipitation and interevent intervals can be ordered arbitrarily and
must not overlap. Evaporation, transpiration and plant growth occur exclusively
during interevent intervals. They are inactive during wind and precipitation
events, including storm recession; calendar time nevertheless advances. An
interevent update must not retroactively include excluded event durations.
Storm infiltration, redistribution/drainage required by storm hydrology,
routing and sediment exchange remain storm processes.

The existing coordinator's native case compilation, adapters, elapsed-time
ledger, source guards and restart infrastructure are a suitable foundation.
They do not yet constitute three-model integration: current case schedules
accept wind/rain only, current dense rain handoffs require an explicit dry reset
including soil water, and LEAF's native-SYRUP coupled driver accepts only whole
86400-second interevent days. Continuous soil-moisture handoff, a dated weather
schedule, partial-day treatment and combined plant/soil restart need explicit
extensions and qualification, preserving historical profiles. Do not discard
soil moisture or reconstruct the bed between events. GPU residency and bounded
memory remain requirements for the new adapters as well as physics kernels.

This records the user's scope and an architecture assessment; it does not
start the proposed Phase 0–3 implementation. The latest phase proposal selects
Hargreaves PET first (explicit PET plus temperature also supported; richer PET
hook later), Tietjen-style infiltration for the new mode, fixed soil geometry
in Phase 1, and conservative moving-ground remapping in Phase 2. These choices
supersede earlier statements below that PET is unselected or that the new
infiltration law must be Smith–Parlange. Historical MAHLERAN comparison modes
and their existing equations remain unchanged.

Status: reviewed architecture; first isolated column implemented and qualified on `feature/subsurface-hydrology`, created from
`deb60c29834a4aad0bf8cb4cfbb611fc28ebe090`. The design assessment ran no
models; implementation is tracked separately in
`agent_handoffs/tasks/subsurface_column_20261031/`; see [qualification.md](qualification.md)
for executed status and limits. Existing uncommitted benchmark work is
preserved. No commit, push, merge, or change to existing defaults is implied.

Evaporation is a core requirement of the inter-event model, alongside
redistribution, drainage, and eventual plant transpiration. The first isolated
column includes actual evaporation driven by supplied potential forcing;
weather-to-PET and vegetation demand are later interfaces.

## Required follow-up: shared weather and weather-to-PET

User requirement recorded 2026-11-08: implement and qualify a weather-to-PET
calculation. SYRUP hydrology and LEAF vegetation/growth must use shared weather
inputs, at least in a documented simplified form: temperature, relative humidity
(RH), vapor-pressure deficit (VPD) and radiation. This is required development,
not merely an optional forcing convenience. The PET method and runtime schema
are not selected by this note.

Use one source-bound weather stream and shared interpretation of time, units
and spatial mapping. Specify timestamp/timezone, interval boundaries, mean versus
minimum/maximum temperature, RH as fraction or percent, VPD pressure units, and
radiation type and rate versus interval energy. Where RH or VPD is derived from
temperature and the other humidity quantity, derive it once under a documented
method and temporal convention; if both are supplied, check consistency rather
than allowing the two components to use contradictory atmospheric conditions.
Preserve input provenance, QC, missing-data policy and observed versus estimated
fields. Simplified scenarios must identify their constants, aggregation and
approximations explicitly, rather than treating invented values as observations.

The weather interface should feed both atmospheric evaporative demand in SYRUP
and temperature, humidity/VPD and radiation responses in LEAF. Define reference
PET/ET, potential bare-soil evaporation and potential plant transpiration
separately, including their partition and resistance/stress conventions. Actual
evaporation and uptake remain SYRUP-owned, moisture/supply-limited water losses;
LEAF receives actual uptake. Avoid counting the same demand or soil-moisture
reduction twice. A selected PET law may require additional inputs such as wind
speed or pressure; identify those requirements and any simplified alternatives
before implementation instead of assuming the four requested variables suffice
for every formulation.

Future qualification must cover units/time aggregation and RH/VPD consistency,
an independent reference for the selected PET law, potential-versus-actual flux
accounting, matching weather consumed by SYRUP and LEAF, and CPU/GPU agreement.
Stream forcing with bounded memory; share derived fields and keep reusable
device inputs resident rather than retaining whole-domain weather histories or
duplicating arrays per component. Existing externally prescribed evaporation
experiments remain valid under their stated assumptions; this note changes no
equation, default, source lock or prior acceptance claim.

The user's 2026-10-31 direction assigns all physical water to SYRUP in
MAPLE–SYRUP–LEAF. LEAF supplies vegetation, roots, and potential plant demand.
MAPLE remains the sediment and wind authority; MASY coordinates events and
biological intervals.

## Recommendation and alternatives

Start with a small Tietjen-inspired multilayer storage and flux kernel in
SYRUP, using SOILWAT's layered accounting as an algorithmic reference. Keep a
two-layer reference case: arbitrary user-defined layers extend the original
published Tietjen model and must be documented and tested as such.

At the same layer count, compact implementations of either approach require
O(N L) storage and work per sweep for N cells and L layers. No comparative
kernel benchmark has been run. Tietjen is the simplest initial scope because
it requires fewer processes and parameters; this is an engineering judgment,
not a measured speed ranking. SOILWAT is a useful established reference for
arbitrary layers and richer hydrology. STEPWAT2 delegates hydrology to
SOILWAT2, rather than supplying a third independent soil-column method.

Do not import entire ECOTONE or STEPWAT2 drivers, their demographic machinery,
or a second soil-water store in LEAF. Reuse appropriate SYRUP/MAPLE backend,
conservation, and checkpoint infrastructure with its identity guards intact.

## LEAF evidence and ownership

The inspected LEAF files include `README.md`, `docs/daily-water-30year.md`, and
`benchmarks/daily_water_growth.py`. Its tests retain individual plant positions
and sparse horizontal root connections, with FP64 `[layer, cell]` water arrays
and factored vertical root weights. The 30-year synthetic CPU/GPU workloads
show scheduling and storage feasibility for those workloads. They are not
calibrated infiltration, evapotranspiration, or drainage simulations. Recharge
of 10 mm every 14 days, equal layer capacities and weights, and simplified
growth are performance fixtures, not physical laws to adopt.

LEAF should supply root geometry, layer access, and potential plant demand.
SYRUP should check LEAF-calculated withdrawals against available water, account
for accepted withdrawals, calculate evaporation and drainage, and return
accepted amounts and any shortfall to LEAF. LEAF
must not deduct the same uptake again. The exact root competition and
physiology partition needs a later explicit contract; do not assume aggregate
vegetation cover where that would prevent individual-plant competition.
Soil-water layers are distinct from MAPLE sediment voxels and its active layer.

## User-defined layers and memory

The user supplies increasing depth edges, for example 0, 0.10, 0.30, and 1.00 m,
plus antecedent moisture and hydraulic properties. Initially use a common
layer count and depth stack across the domain, with spatial property maps as
needed. Different layer counts or bedrock depths per cell require a separate
bounded extension. Validate residual moisture, wilting point, field capacity,
saturation, conductivity, suction, and redistribution parameters as required
by the selected processes, with explicit units and admissible bounds.
Physical upper, deeper, and rooting zones must remain well defined when the
user changes layer edges; “layer 2” cannot universally mean all deeper soil.

Store water-equivalent depth in metres per layer per cell. Volumetric moisture
is that depth divided by layer thickness; volume is depth multiplied by cell
area. LEAF's current synthetic arrays use cubic metres, so its interface needs
an explicit conversion. Derived representations need not persist as additional
authoritative state.

Use FP64 `[layer, cell]` arrays. A proposed GPU kernel assigns a thread to each
column and loops over its few layers: columns run in parallel and adjacent
threads access adjacent cells. Use Numba on CPU and the existing CuPy/CUDA
infrastructure on GPU. Avoid Python loops over cells, a launch for every
individual layer, and host transfers at every step. This layout is a proposal,
not a performance qualification.

One FP64 field over one million cells and ten layers occupies 80 MB decimal
(about 76.3 MiB); three layers occupy 24 MB. That is not a process memory bound:
parameters, transaction buffers, flux scratch, roots, existing model state,
checkpoint staging, and the GPU allocator/context add to it. Share common
layer parameters, retain only necessary current buffers, stream summaries,
and checkpoint at selected times. Preserve FP64 unless reduced precision is
separately qualified. Record live arrays, reserved device memory, and host
memory separately; LEAF's synthetic tests do not establish combined-model fit.

## Initial moisture after a precipitation event

A final single-bucket water total cannot uniquely determine a vertical moisture
profile. Initialization needs antecedent moisture, soil properties, accepted
infiltration and its timing, and an explicit choice about preferential flow.
Uniformly splitting rainfall among layers would be an approximation, not a
reproduction of Tietjen's approach.

The requested inter-event model starts after a wind or water event. Its
initialization still needs information about water entering the soil during
the event. Prefer memory-bounded layer allocation at accepted storm steps,
or an integrated layered storm column, so the event hands over a valid
profile without storing a full time history. Biological growth and quiet-period
ET can remain off during events under the initial scheduling policy. Pure
postprocessing instead needs a recorded infiltration history or a declared
reconstruction assumption; cumulative end-of-event maps alone are insufficient.

Use SYRUP's actual accepted soil intake, rather than adding gross rainfall
again. Matrix intake enters the upper soil; any configured Tietjen-like
preferential fraction enters its physically defined receiving zone. Transfers
between layers, capacity overflow, surface return, and bottom drainage must
all be conservative and explicitly accounted for.

For a Tietjen reference, preferential intake is calculated from available
surface water before matrix intake uses the remainder. Splitting an already
computed legacy bucket intake after the fact is a hybrid reconstruction, not
that original intake law. A production implementation must calculate the
combined admitted intake once, limited by available surface water and the
receiving profile, and couple that result back to SYRUP. The required matrix
closure is MAHLERAN's Smith–Parlange moisture-dependent approach, as directed
on 2026-11-08; its mapping to the configurable layers still requires explicit
design. That contract must identify its moisture state and physical depth: the
current Smith–Parlange bucket uses whole-bucket storage and deficit, which
cannot be replaced silently by the sum of a deeper layered profile.

A SOILWAT-like cascade above field capacity is a cheap reference. However,
instantaneous drainage to field capacity and finite-rate storm percolation
are different physical assumptions. Establish the time basis and test the
two-layer reference and timestep response before adopting a law; do not apply
an arbitrary cascade at every hydraulic substep and claim published identity.
Before accepting a reference kernel, specify whether transfers use old storage
with one interface traversed per update or cascade through successive layers
within an update. The inspected MAHLERAN daily adaptation uses old stores;
ECOTONE's high-water routine cascades. Neither timing convention automatically
qualifies the new storm coupling or matches the original paper's time basis.

If the layered column changes accepted intake or saturation return, it must
feed back to the surface solver at the same accepted step. A shadow diagnostic
may observe the existing bucket but cannot silently disagree with its water
capacity or fluxes. Previously exported single-bucket drainage cannot be
reclaimed as deep soil water, or charged a second time. The production layered
column must replace or extend the actual soil-water authority in an explicit
mode, preserving the existing legacy reference mode.

Start with vertical columns on fixed terrain; lateral subsurface flow and
explicit groundwater are not part of this first milestone. Later implement
adjacent-layer redistribution, soil evaporation, and plant uptake on a stated
time basis. The original Tietjen moisture-gradient diffusion assumes similar
retention curves. Contrasting textures need a qualified hydraulic potential
and interface law, rather than blind use of a moisture gradient. Adding layer
interfaces must not inadvertently alter the physical profile's conductance.
Declare flux dimensions, interface distances, and conductance explicitly. The
MAHLERAN adaptation's two-layer expression cancels its thickness sum; copying
that coefficient independently to every new interface does not establish a
convergent multilayer law. Check thin-layer stability and positivity, using
substeps or a suitable implicit update when needed. Transfers must respect
available donor water and receiving capacity; post-update clipping that creates
or discards water is not permitted.

After wind, retain the previous moisture profile. If erosion or deposition
changes soil thickness, conservative remapping and water carried by removed
or added material need an explicit contract. Fixed-bed initialization tests
cannot qualify that interface.

## Required follow-ups: soil remapping and moisture-dependent infiltration

User direction on 2026-11-08 makes both items below required implementation
work. They are not implemented or qualified by this note. Earlier descriptions
of detailed remapping as deferred refer to milestone ordering, not its removal
from the intended model.

1. **Detailed soil remapping after erosion/deposition.** Use actual MAPLE bed
   changes to update the physical soil profile, including layer positions and
   thicknesses, soil properties and water storage. Define whether layer edges
   follow the surface or a fixed elevation datum, how newly exposed/deposited
   material acquires properties and moisture, and where water associated with
   eroded material goes. Remap existing water conservatively; do not reset
   moisture fractions, invent water to fill new layers, or discard water when
   capacity changes. Explicitly account for surface return, lateral/exported
   water and any other admitted destination. Define how LEAF root access changes
   with burial or exposure, retaining plant establishment elevation as needed.
   One MAPLE sediment authority and one SYRUP water authority remain mandatory.
   Storage publication/remapping must preserve the existing net-elevation-only
   hydraulic refresh rule; a remapping call alone must not force rerouting.
2. **Soil-moisture control of infiltration using MAHLERAN's approach.** Retain
   the Smith–Parlange capacity calculation with the evolving moisture deficit
   `theta_sat - theta`, suction/surface-depth dependence, selected conductivity
   or pavement/rainfall closure, and explicit saturation-excess return. The
   existing SYRUP storm bucket already has this dependence; the new work must
   drive it from the authoritative persistent layered state, carrying antecedent
   moisture into storms and updating it during accepted steps. Define the
   effective infiltration zone and the mapping of its water depth, saturation
   and moisture to MAHLERAN's single bucket. Do not use a whole-deep-profile
   average by default or infer a unique profile from one bucket total. Use the
   author-corrected local rain index `r2(i,j)`. Reconcile layered percolation with
   the old bucket's drainage so water is not removed twice; intake and returned
   water must feed back to the surface solver at the same accepted step.

Qualification must cover wet/dry antecedent profiles, rainfall/run-on and
saturation limits, matched corrected MAHLERAN references for the supported
single-bucket limit, and layered storm feedback beyond that limit. Remapping
needs independently checkable erosion/deposition examples, changed capacities
and root access, complete water destinations, restart continuation and failure
without partial mutation. Both follow-ups require CPU/GPU correspondence and
bounded-memory implementations that process affected columns where feasible.
No existing legacy benchmark defaults, source guards, tolerances or numerical
identities are changed by recording these requirements.

## Lifecycle and implementation stages

The new continuous-water mode will retain soil moisture between events. Existing
dry-reset benchmark/default behavior remains unchanged until this mode is
implemented and qualified. Finish surface-water and sediment accounting under
an explicit policy while retaining soil water. In the new continuous mode,
water left in depressions must persist, infiltrate, or evaporate through an
accounted physical flux; it cannot disappear in a dry reset. Surface state must
remain available to the inter-event solver, and permission to start a later
wind event needs an explicit surface-moisture contract. Do not reinterpret existing
`DryHandoff` or checkpoint identities as compatible with new persistent state.
Version the state, calendar, and source contracts explicitly. Pin grid origin,
row/column orientation, flattened cell identity, active masks, and area units
at the LEAF interface. Layer bookkeeping at every accepted storm step adds
work proportional to layer count. Aggregating at reporting intervals can save
work but loses within-interval capacity and return feedback, and is not an
equivalent default optimization.

Suggested bounded stages are:

1. Define layers, antecedent state, and a reference column that initializes a
   post-rain profile, with CPU/GPU water ledgers.
2. Connect accepted storm intake and surface return with MAHLERAN-like
   moisture-dependent infiltration, persistent event handoff, and restart.
3. Add quiet-interval redistribution and evaporation, with LEAF-calculated plant
   withdrawals checked and accounted for by SYRUP.
4. Implement the required detailed conservative soil remapping after
   erosion/deposition, update LEAF root access, and qualify long event sequences.

Checks must cover all water destinations, nonnegative bounded stores, dry and
wet antecedent states, intense and gentle storms, preferential flow, surface
return, bottom loss, timestep and layer refinement, and heterogeneous soils.
Require two-layer references, independent column checks where appropriate,
CPU/GPU agreement, restart continuation, and failure without partial mutation.
Use existing water conservation policy. Measure actual working memory and
cold/warm timings after physics checks; do not extrapolate full-model speed
from a synthetic column benchmark.

## Sources and limits

- [Tietjen, Zehe & Jeltsch (2009), Water Resources Research](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007WR006589),
  particularly its hydrology methods. The configurable multilayer design above
  is a proposed extension, not the published two-layer implementation.
- Supplied ECOTONE: `/home/okin/Ecotone Wind 9/Ecotone Wind 9/Src/watrflow.f`
  and `subwatr.f`, especially `hwdrain` and `lwdrain`. Its wind component is
  excluded. No new ECOTONE build or performance test was run.
- [SOILWAT2 project](https://github.com/DrylandEcology/SOILWAT2) documents its use
  in STEPWAT2. Reviewed STEPWAT2 source `003cb9b4adbf13d64a39d6d62fd95280adac27fc`, linked SOILWAT2
  `b05f410a9e96a5f26857bd37247fe7152e9861d8`; selected source records are in
  `agent_handoffs/tasks/vegetation_model_comparison_20261030/`.
- Actual LEAF selected-source hashes and copied benchmark evidence are in
  `agent_handoffs/tasks/subsurface_hydrology_design_20261031/`.

Recommendations are engineering judgment. Neither these source inspections nor
LEAF's historical tests constitute new numerical qualification of this design.
