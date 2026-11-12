# Phase 1.5: daily small-rain recharge and native storm fallback

Status: user-authorized queued phase, 2026-11-12 UTC (2026-11-11 local).
This specification records the binding requirements. Implementation status and the actual bounded qualification are in [phase15_qualification.md](phase15_qualification.md) (source N15; bounded whole-phase acceptance recorded in phase15_acceptance.json; publication pending).
Finish, validate, commit and push Phase 1 first; then implement, independently
qualify, commit and push Phase 1.5 before starting Phase 2. Preserve Phases 2,
2.5 and 3 and all their existing requirements. Follow automatic Codex–Claude
implementation, independent review and bounded corrections.

## Scope and ownership

Provide a narrow native SYRUP interface for a future weather/scheduler adapter.
It accepts the rainfall descriptor, threshold, accepted parent state and
occurrence identity. SYRUP owns water physics and the single water inventory;
MASY owns scheduling; LEAF owns plants and withdrawal calculations; MAPLE owns
sediment and landscape initialization. Reuse the qualified Phase 1 column,
lifecycle, validation, conservation and transaction infrastructure. No complete
MASY daily scheduler or final CSV formats, LEAF redesign, new infiltration law,
soil-property estimation, moving-soil remapping or multi-year coupled run is
part of this phase. New behavior is explicit opt-in; historical defaults stay.

Before implementation reread AGENTS.md, agent_handoffs/orchestration_state.md,
development_phases_20261109.md, design.md, implementation.md and
parameter_reference.md. Inspect actual Phase 1 acceptance records and unadopted
candidates as well as current Git HEAD. Establish the exact completed/published
compatible Phase 1 baseline and isolated write scope. Today's water109 candidate
is evidence for Phase 1 components, not a substitute for that future baseline.
Preserve unrelated changes, active writers/jobs and historical source locks.

## Rain selection and forcing

Caller supplies daily amount P in mm, intensity I in mm/hour and a user-selected
amount threshold T in mm. No default threshold is selected by this specification
and no extra intensity threshold is authorized. Validate all inputs, including
finiteness and positive I for positive P, before mutation.

| Condition | Selection |
| --- | --- |
| P = 0 | No precipitation operation. |
| 0 < P < T | Attempt simplified recharge. |
| P = T > 0 | Attempt simplified recharge: user decision, 2026-11-12 UTC. |
| P > T | Full native SYRUP storm. |

For positive constant-intensity full rain, duration is 3600 P/I seconds. Build
forcing with actual SYRUP rainfall APIs; preserve its units and source identity.
A failed completion/resource/time limit is not completion. Record the selection
and reason, including any whole-occurrence fallback. Persistent moisture carries
across both branches.

## Simplified addition and conservative downward movement

This is a deliberate approximation bypassing full infiltration/runoff. Apply P
uniformly as depth P/1000 m over active cells, initially entering the upper soil
layer. Coordinate addition and downward redistribution so the existing
addition-stage saturation return cannot prematurely reject water that lower
layers could accommodate. Use the existing gravity-drainage behavior and its
configured layers; do not fabricate a small positive timestep to call a
zero-time-restricted API.

**Field capacity controls gravity movement. Saturation controls maximum
storage.** Water above field capacity can move downward before saturation is
reached. Continue through the configured layers. Open bottoms allow drainage
under the existing rule; closed bottoms allow no escape and transfers must
respect receiving storage. Blocked water may remain above field capacity but
never above saturation. Layers at field capacity still have storage capacity;
a field-capacity profile is not a saturated profile.

For a closed profile that cannot accommodate the precipitation after this
redistribution, use whole-occurrence full-storm fallback if ANY active column
fails. No spatially mixed recharge/storm scheme in this phase. A discarded trial
must leave the accepted original state exactly unchanged; fallback starts from
that same original state with the FULL P. Never apply partial recharge followed
by the full storm rainfall.

Retained surface water and unresolved flow remain explicitly represented;
never erase them to make simplified recharge eligible. Determine eligibility
and the admitted daily update through actual qualified Phase 1 state interfaces.

## Daily update and clocks

The agreed nominal order is wind events in listed order, then that day's rain
selection/treatment, then daily interevent subsurface hydrology, evaporation,
transpiration and growth. Wind and full rain/recession retain native subdaily
behavior. Interevent soil hydrology and transpiration use daily updates;
growth is daily or coarser. Integrate recharge with that daily water update;
no extra subdaily interevent hydrology and no duplicate drainage/E/T or daily
interval. Evaporation, transpiration and growth remain inactive throughout wind
and full rain/recession. Finish the real full water event before daily processes.

A future daily scheduler has a nominal calendar SEPARATE from physical elapsed
time. Wind/full rain/recession occur between nominal days and add physical
duration to daily intervals, so a nominal year can contain more than 365/366
physical days. Document this approximation and keep native clocks unchanged.
Use an explicitly versioned new contract; do not retrofit this convention into
historical clock profiles, existing dated Phase 1 schedules or old locks. This
phase supplies the native capability/contract, not that complete scheduler.

## Accounting, transactions and restart

Stage without mutating accepted state; adopt conservatively or discard and
fall back. Account rainfall, storage, interlayer transfers, drainage and surface
water exactly once; internal transfers are not external sources/sinks. Preserve
active masks, inactive cells and the bed on the simplified path. Bind generation,
parent and occurrence identity; reject duplicate application, support retry and
checkpoint/restart continuity, and preserve persistent moisture across branches.

Small rain MUST NOT reset matrix wetting history. Audit direct recharge against
persistent Green–Ampt J and cumulative matrix intake. Receipts must distinguish
simplified recharge from modeled matrix/preferential infiltration. No new
history-increment/recovery rule is selected here: inspect actual Phase 1 laws
first and ask the precise scientific question if a new rule is needed; continue
independent engineering while awaiting that answer. Retain all source guards,
explicit restart versioning and old restart compatibility; no implicit migration
or rebinding of old runs.

## Independent qualification and deliverables

Use independent controls and the existing numerical/conservation contracts on
NumPy, compiled CPU and authorized available CUDA hardware. Cover zero rain,
below/equal/above T, 1/2/3+ layers, dry/above-field-capacity/saturated layers,
movement before saturation, open drainage/closed retention, field-capacity
versus saturated closed profiles, heterogeneous fallback triggered by one column,
exact unchanged discarded parents, fallback equal to a direct full native storm
from the same parent/forcing, masks, units, invalid inputs, unchanged bed,
no duplicate rainfall/drainage/E/T/time, retries/duplicate occurrences and
restart followed by further full storms.

Keep memory bounded and arrays resident on the selected backend. Avoid deep-bed
copies, full-domain history and per-step host/device transfers. Record genuine
GPU execution versus unavailable/untested paths, and do not infer large-domain
readiness from a small fixture.

Deliver independently reviewed architecture/implementation; examples for
recharge, open-bottom drainage, closed retention and full fallback; exact source
identities, raw verification, restart evidence and limits; and user/theory
documentation of the approximation, capacity distinction and nominal-calendar
convention. Publish only independently accepted work under the existing phase
commit/push authorization.
