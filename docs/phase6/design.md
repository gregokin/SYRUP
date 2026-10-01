# Phase 6: complete water events and restart

Implementation is under validation; independent Claude review is pending. The
accepted Phase 5 remains the baseline until this phase is explicitly accepted.

## Completion and the dry-again assumption

`complete_event` drives the existing Phase 5 solver and actual MAPLE bed
transactions. It observes every accepted step. Completion requires the entire
rainfall schedule to have ended and all of these conditions to hold continuously
for 60 physical seconds:

| Quantity | Default limit |
| --- | ---: |
| Maximum local surface depth | 1e-8 m |
| Total domain surface storage | 1e-6 m³ |
| Maximum local unit discharge | 1e-10 m²/s |
| Total outlet discharge | 1e-9 m³/s |
| Mobile sediment, every cell and class | Actual MAPLE `mass_resolution_kg` |

Plot 1's MAPLE mass resolution is 1e-10 kg per cell/class. This is physical
significance, not a permitted conservation error. Maximum local depth and flow
prevent a domain average from hiding ponded water; total storage limits the
water removed across a large domain. The volume limit is one millilitre. These
absolute SI limits are configurable and intentionally conservative starting
choices; their portability to other geometries requires sensitivity checks.
The recorded Plot 1 check with 100× stricter thresholds and a 120 s hold
preserved the final bed and exported composition bitwise; see acceptance.md.
The hold lasts accepted physical seconds, independent of sampled report rows.
A zero-rain gap does not finish a schedule containing later rain. Reaching a
maximum duration or global step limit raises noncompletion, never dry success.

After the hold, all remaining mobile sediment is deposited locally through
actual MAPLE. Nothing is clipped or deleted, including quantities below mass
resolution. A final actual MAPLE terrain commit flushes all ledger fields,
including compensation/residual-only state. Completion conditions are checked
again after rerouting. Failure leaves the caller's wet state untouched.

Only after these checks does the documented inter-event assumption set surface
water, soil water, discharge and transport velocity memory to zero. The output
retains the pre-reset water grids and records surface and soil removals
separately. They are external removals, not additional runoff, drainage or
simulated evapotranspiration. Both the storm water budget and storm-plus-reset
budget must close. The hydrograph describes physical storm evolution; its last
row includes terminal deposition and commit but precedes external water removal.
The dry handoff and reset record are separate artifacts.

## The same numerical conservation policy as MAPLE

User direction on 2026-09-30 supersedes earlier independent SYRUP mass-error
constants. `conservation.reservoir_bound_kg` calls the actual dependency's
`maple.surface.voxels._numerics.summation_error_bound_kg`, using MAPLE's block
reservoir policy: `4 * number_of_accounting_terms * epsilon64 * operand_scale`.
The operand scale is the largest genuine per-class bed, mobile or boundary
reservoir endpoint, not a small cancelled net exchange and not the sum of all
classes. Event accounting uses the actual MAPLE water-call count. Local
transport and accumulated request identities use their respective operation
counts and the same shared helper; MAPLE's water request validation includes a
one-kilogram scale minimum for that identity. Reservoir closure does not.

Neither `mass_resolution_kg` nor reported numerical residuals pad the reservoir
conservation bound. Small residuals are reported in kilograms and assessed
against the actual arithmetic scale. Tiny masses in terminal-deposition tests
are test inputs, not separately chosen acceptance tolerances. All of them must
remain in actual MAPLE reservoirs. Volume budgets use the same dimensionless
FP64 summation coefficient multiplied by the relevant volume in m³; sediment
mass resolution is never applied to water volume.

These checks replace the Phase 5 code's earlier closure tolerances. Historical
Phase 5 reports remain historical evidence, not retroactively rerun claims.

## Restart contract

The new versioned bundle contains:

- `maple_state.npz`: actual MAPLE snapshot of voxel storage, active layer,
  availability and water. Its provenance explicitly requires the companion.
- `continuation.npz`: pending sediment ledger and compensation, committed
  terrain, hydraulic/soil state, virtual-velocity memory, original inventories,
  cumulative accounting, peaks and bounded diagnostics.
- `checkpoint.json`: schema, file hashes, strict source/case/forcing/config
  identity, field metadata, original reporting/forcing plan and quiet-hold
  progress. No pickle or dynamic type imports.

Checkpoints occur at accepted original reporting/forcing boundaries. A
checkpoint never forces a terrain commit or introduces another physical
boundary. Resume continues the original global step count and time partition;
report accumulators and hold progress do not restart from zero. History storage
is bounded by configured reporting/log limits, with one prefix copy on resume,
not repeated concatenation per timestep. Derived routing is rebuilt from the
saved committed terrain and fixed boundary reference and compared by digest.

Loading validates component hashes, exact array shapes/dtypes, accounting
counters, geometry, class ordering via identity and physics parameters, MAPLE
partition/availability/water/ledger invariants, actual MAPLE ledger-to-bed
reconciliation, and independent cumulative budgets before returning private
objects. Changed source or forcing is refused rather than silently adopted.
The current CPU loader bounds JSON metadata to 32 MiB and uncompressed array
payloads to 1 GiB by default. Larger domains need an explicit memory policy.

A new bundle is published by same-filesystem directory rename after successful
validation and writing; existing paths are not overwritten. This is atomic
publication, not a guarantee of persistence across power loss. Completed bundles
require explicit `allow_complete=True` for inspection/handoff and cannot be
resumed as wet events or repeat the reset. A MAPLE component snapshot on its own
is not a SYRUP continuation. Full case/forcing/source identity is assembled by
the CLI; low-level checkpoint callers must supply an equally complete identity.

## Invocation and limits

With the pinned dependency environment described in Phase 5:

```bash
python -m maple_syrup.event_experiment --case-dir outputs/plot1 \
  --output-dir outputs/my_event --implementation numba
python -m maple_syrup.event_experiment --case-dir outputs/plot1 \
  --output-dir outputs/my_pause --pause-at-s 1200
python -m maple_syrup.event_experiment --case-dir outputs/plot1 \
  --output-dir outputs/my_resume \
  --resume outputs/my_pause_checkpoints/step_000001200
```

Output and checkpoint trees must be new, separate, and outside imported cases,
reference/source/package trees. Default checkpoint cadence is 600 s rounded up
to an original boundary. Default maximum duration is the imported legacy storm
window, 5400 s for Plot 1. A saved checkpoint at that maximum is diagnostic;
changing the maximum duration is a configuration change and currently refused
by strict restart identity.

This phase is a CPU implementation with NumPy state and optional explicitly
selected compiled Numba hydraulics. It does not claim GPU execution or a new
performance improvement. No per-cell Python loops or alternate bed authority
are added. Actual MAPLE transaction validation remains a measured Phase 5 cost;
checkpoint serialization adds bounded explicit I/O/copies. Wind invocation is
still a later phase; no wind physics is copied. Splash, ecology, nutrients,
plant growth, evapotranspiration, general pit handling and alternative hydraulic
methods remain outside this phase. Phase 4R remains an important GPU/alternative
routing follow-up. Original Fortran equation tests are not a full original
MAHLERAN sediment-storm benchmark.

## Independent review qualifications — 2026-09-30

The terminal handoff is committed at the completion time. A final empty-ledger
commit can therefore update the commit timestamp/count and refresh routing even
when elevation is unchanged. This differs from avoiding redundant interval-end
commits during ordinary Phase5 evolution; it does not add a physical timestep.
Conservation uses MAPLE's scalar largest-real-class endpoint scale for all class
residual checks, not a separate per-class arithmetic scale.

Independent review and Codex probes did not reproduce terminal false refusal on
Plot1 with nonzero remaining loads (8 uniform/random probes);56transport stress
cases up to64substeps also passed. General larger-bed reduction sensitivity is
still a qualification, not a demonstrated failure or a reason to relax bounds.
Other follow-ups: distinct CLI noncompletion exit codes; duration-limit restart
usability; cumulative diagnostic/counter cross-checks; binding terrain-reference
payloads independently to case geometry; clean refusal of malformed snapshots
missing availability; and dependency checks for private MAPLE numerical and
reconciliation symbols. Checkpoints are integrity-checked scientific artifacts,
not authenticated protection against deliberately rewritten payloads.
