# Phase 7b — characteristic transport in the event, completion, checkpoint and benchmark drivers

Status: implemented 2026-10-01 (task `phase7b_sediment_fix`, integration
assignment). Verified locally by the author with the pinned environment
(commands and results in
`agent_handoffs/tasks/phase7b_sediment_fix/integration_report.md`); Codex
performs the independent acceptance, the full Plot 1 storms, the bin-count
study and the GPU checks. Storm-scale agreement with MAHLERAN and a
converged default bin count are **not** claimed here.

## 1. Controls

`sediment_event.SedimentEventControl` gains three fields, all part of the
run identity (checkpoint and summaries record them):

| Field | Values | Default | Meaning |
|---|---|---|---|
| `transport_scheme` | `characteristic`, `upwind` | `characteristic` | Phase 7b bounded-phase-bin kernel (`characteristic_transport`) or the unchanged Phase 5 well-mixed operator (`sediment_transport.transport_step`), kept as an explicitly named comparison |
| `phase_bins` | 1..128 | 32 | position bins per cell and class; 32 is a candidate default, not an accepted convergence |
| `transport_implementation` | `auto`, `array`, `numba` | `auto` | characteristic kernel implementation; `auto` follows `storm.implementation`; no fallback either way |

`sediment_courant_max` now defaults to 0.5 and is validated per scheme:
the characteristic kernel needs `v dt/(n_sub dx) ≤ 0.5` (one face crossing
per packet per substep); the upwind operator accepts up to 1. The
substep count is `ceil(v_max dt / (dx · sediment_courant_max))` as before,
so accepted hydraulic timesteps are unchanged (Plot 1 at dt 1 s stays at
one substep: `a ≤ 0.048`).

The three runners (`sediment_experiment`, `event_experiment`,
`benchmark_experiment`) expose `--transport-scheme`, `--phase-bins`,
`--transport-implementation` and `--sediment-courant-max` (default 0.5);
`frozen_control` passes them through. Summaries carry a `transport` record
(scheme, bins, resolved implementation, pickup convention, reconciliation
counters, reroute rule) and `final_state.npz` carries `phase_fraction` and
`phase_position_m` for the characteristic scheme.

## 2. State

`SedimentEventState.phase: PhaseState | None` — `(ny, nx, nc, B)` fractions
and representative positions partitioning `bed.water.mobile_mass_by_cell_class_kg`
(characteristic) or `None` (upwind; nothing is allocated).
`initial_event_state(..., control=)` builds the canonical partition for the
control's scheme and bins; `prepare_verified_sediment_case(..., control=)`
passes the runner's control through. The event driver refuses, before any
MAPLE call, a state whose phase does not match the control (missing phase
for the characteristic scheme, a phase for the upwind scheme, a different
bin count) or is not a valid partition of the actual pool.

The public single-attempt entry `sediment_coupled_step` performs the same
cheap binding checks before any hydraulic or MAPLE work: the control is
validated, the phase must be present exactly for the characteristic
scheme, and its bin count, cell size and namespace must be the control's
and the network's (a direct call with a 16-bin control on a 32-bin state
is refused; Codex's `direct_control_probe.log` found the earlier version
ran it). The full phase-array validation is not repeated there; the kernel
validates the arrays before returning any result and the drivers validate
the partition at their boundaries.

## 3. Transaction per accepted step (characteristic)

1. Water column and routing; wet laws (unchanged).
2. Pickup call through actual MAPLE (unchanged): `M_after = M_pre + P_actual`.
3. `characteristic_step(network, M_pre, phase, P_actual, v, r, settle, dt,
   n_substeps, courant_max, implementation)`: the kernel reconstructs the
   component masses from the PRE-pickup pool and its fractions, appends the
   actual pickup as a packet at the upstream face, and returns the same
   `TransportStep` contract (`T = M_remaining + Dep + E`, requests, faces,
   budgets with `mobile_before = M_pre + P_actual`).
4. Publish `T`; deposit/export call through actual MAPLE (unchanged).
5. **Reconciliation** (`sediment_event.reconcile_phase`): the kernel
   predicted `M_pred = T − Dep − E`; MAPLE published `M_actual`. Per cell
   and class `|M_actual − M_pred|` must be within
   `summation_error_bound_kg(16, max(T, Dep, E, M_pred, M_actual))`
   (a declared rounding count for the same arithmetic on both sides, not a
   mass floor); otherwise the attempt raises and nothing is published.
   Within the bound MAPLE's mass is authoritative and the fractions remain
   the distribution, with two explicit cases: a cell whose actual pool is
   exactly 0 is canonicalised (counted as `n_phase_canonicalized`); a cell
   whose actual pool is positive while the kernel's is 0 keeps the kernel's
   canonical partition (mass at the upstream face) and is counted as
   `n_phase_rounding_remnants`. Nothing is invented silently; the largest
   residual is reported (`max_phase_reconciliation_residual_kg`).
6. Publish only on acceptance: storm, bed, velocity memory and phase
   together. Rejected hydraulic or transport attempts, and any raise, leave
   the caller's state (including its phase) untouched.

## 4. Commits and rerouting

The phase is per cell along the cell's CURRENT outflow direction. A commit
that reroutes a cell keeps its fractions and positions, so the within-cell
progress is carried over to the new receiver direction (same cell size).
This is a documented steering approximation; mass is never reset to the
upstream face at commits. Each commit log entry records
`phase_steered_cells` (rerouted cells carrying mobile mass) and the result
sums them in `phase_steered_cells_total`.

## 5. Completion and checkpoints

Terminal deposition returns the whole pool to the bed; the phase is then
canonical (both in the pre-reset result state and in the dry handoff).
Checkpoint schema is `maple-syrup-checkpoint/2`: the continuation encodes
the phase (fractions, positions, `dx`, bin count) as host NumPy;
`maple-syrup-checkpoint/1` bundles are refused with an explicit message and
no migration. Loading validates the phase arrays (shape, dtype, bin count
against the control in the identity, partition of the snapshot's actual
mobile mass, canonical wherever that mass is 0, canonical in a dry
handoff) and the new counters, before returning anything.

## 6. Verification performed (author, pinned environment)

See the integration report for the exact commands. In summary: the Phase 7b
kernel tests (37 passed, 1 GPU skip), the new integration tests (14
passed), and the full CPU suite (Phase 1–7b: 565 tests, 9 skips) pass;
ruff passes on `src/maple_syrup` and `tests/phase5..7b`. Two existing
bitwise hydraulics-parity tests (numba vs array sweeps) now pin the array
transport kernel on both sides because the numba and array kernels agree
only to round-off; their kernel pairing is tested separately at rtol 1e-10.
A 120 s Plot 1 frozen window (diagnostic, not a benchmark) ran with all
combinations and closed; timings are in the report.

## 7. Open items for Codex

- Full Plot 1 storms at several bin counts and timesteps; the export,
  composition and timing against the MAHLERAN reference (with the legacy
  ledger's clipping source and stranded pool in mind).
- Performance: the array kernel is markedly slower than the compiled one
  on CPU (see report); the numba kernel adds first-use JIT time.
- GPU: the array kernel is namespace-generic and parity-tested per kernel;
  no coupled GPU event exists.
