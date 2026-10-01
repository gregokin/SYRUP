# Phase 7b — characteristic transport with bounded phase bins (candidate kernel)

Status: kernel implemented 2026-09-30 (task `phase7b_sediment_fix`),
corrected after Codex's independent review (§2, logarithmic-mean merge) and
**integrated as the default transport scheme** of the event, completion,
checkpoint and benchmark drivers on 2026-10-01; see
[integration.md](integration.md) for the transaction, reconciliation,
checkpoint schema and controls, and
`agent_handoffs/tasks/phase7b_sediment_fix/integration_report.md` for the
commands run and their results. `sediment_transport.transport_step` is
retained unchanged as the explicitly named `upwind` comparison scheme.
Nothing here changes hydrology, MAPLE, MAHLERAN or any Phase 5–7 artefact.
Storm-scale agreement with MAHLERAN and the bin-count convergence on Plot 1
remain unmeasured until Codex runs the full storms.

## 1. Disposition of the earlier proposal

`agent_handoffs/tasks/phase7b_sediment_fix/design.md` is retained verbatim
as the raw proposal. Codex's disposition, adopted here:

- **Pickup convention: legacy upstream face.** The user's standing
  preference is exact MAHLERAN where feasible; new pickup enters at
  `x = 0` (`flow_distrib.for` 44–46). The uniform-source convention remains
  a documented alternative, not this fix's default, and no new user
  decision was needed. Both conventions converge as `dx → 0`; the earlier
  proposal's claim that only one is grid-convergent is withdrawn.
- **Erlang-8 transit rejected.** At `Pe = dx/L = 16` it transmits 1.5e-4 of
  entering mass against the exact 1.1e-7; an absolute probability error of
  that size cannot be declared negligible when the reference exports 9 g
  from 450 kg of pickup. Its conditional mean transit of surviving mass is
  `dx/v/(1 + Pe/N)`, not `dx/v`, and the proposal's bracketing bound fails
  at `Pe = 0.1`, `b = 1` because the half-substep reaction offset exceeds
  the continuous-time error.
- **Ring-loss `e^{−2Pe}` attribution: unproved**, kept only as a hypothesis
  for the legacy bookkeeping audit.
- **Bounded packets are allowed.** The contract forbids unbounded history,
  not a fixed number of phase slots per cell.

Codex's non-production prototypes (`characteristic_probe.py`,
`phase_bin_probe.py` and their JSON) showed, for a constant-law six-cell
impulse, EXACT `exp(−k dx/L)` face crossings and the correct 300 s first
arrival for `B = 8/16/32/64`, and for a smooth source an exact export
integral with a rate pulsation that shrinks with `B`. This kernel is the
production-shaped implementation of that idea.

## 2. Mathematical contract

Per cell and class, the mobile mass is a set of packets at distances
`x ∈ [0, dx)` from the upstream face along the cell's D4 outflow. Over one
substep of length `dt` with the cell's `v`, `r = 1/L` and `settle` frozen:

- **Settle** (dry / no capacity): existing packets and the pickup packet
  are the deposition request in full; nothing moves.
- **Characteristic:** a packet at `x` travels `v dt` unless `x + v dt ≥ dx`,
  in which case it reaches the face after `τ = (dx − x)/v` having
  travelled `dx − x`. Survival over a travelled distance `s` is
  `exp(−r s)`; `m (1 − exp(−r s))` deposits in the traversed cell.
- **Crossing:** the surviving mass is the exact face crossing. At an
  outlet it is an export request. If the receiver settles, it is the
  receiver's deposition request. Otherwise it continues for `dt − τ` at
  the receiver's `v_j`, `r_j`: travels `d = v_j (dt − τ)`, deposits
  `(1 − exp(−r_j d))` of itself in the receiver and starts a packet at
  `x = d` there. Courant `v dt/dx ≤ 0.5` per substep on every active
  cell/class guarantees `d ≤ dx/2`, so at most one face per packet per
  substep; a violation raises the recoverable `TransportStepRejected`.
- **Pickup:** the actual MAPLE removal of the step is one new packet at
  `x = 0`, injected in the first substep only, never merged with older
  mass before it has travelled.
- **Merge into `B` bins** (bin `b(x) = min(⌊x B/dx⌋, B−1)`): packets landing
  in the same cell, class and bin become one packet with mass `Σ m_i` and
  representative

      x_rep = L · log( Σ m_i e^{x_i/L} / Σ m_i )      (r > 0, shifted log-sum-exp)
      x_rep = Σ m_i x_i / Σ m_i                       (r = 0)

  Evaluated per bin in one of two numerically safe forms, selected by the
  spread `r (x_max − x_min)`: for a spread `≤ NARROW_SPREAD` (0.5) the sum
  is shifted by `x_max` with `expm1`/`log1p` weights in `[expm1(−0.5), 0]`
  (no cancellation, exact mass-weighted mean as `r → 0`); for a wider
  spread it is shifted by the largest log-weight `log m_i + r x_i`, a
  positive sum whose leading term is exactly 1. Codex's reproducer (a
  1e-20 kg packet at `x = 0.009` merged with 1 kg at `x = 0` in a 0.01 m
  cell with `L = 1e-4`) exposed the earlier single `expm1` form: its
  subtractive sum rounded to `−M` and a floor at `−1 + ε` moved the
  representative from the exact 0.0043948298140119085 m to 0.00540 m,
  changing the eventual survival by ~22,000×. The floor is gone; the exact
  value and the eventual survival (4.54e-25 kg) are now regression tests.
  `x_rep` is within `[min x_i, max x_i]` mathematically; only round-off
  containment (a clamp to the constituents' range) is applied, never mass
  clipping. Because `b(·)` is monotone, `x_rep` lies in bin `b`.

**Exact invariants.** Per-class conservation `Σ T = Σ (M + P)`; per-cell
`T − (M + P) = In − Out`; positivity; `Dep + E ≤ T`; face crossings equal
`Out + E` in the MAPLE face layout; deposition along a traversed distance
is exactly `1 − exp(−r s)`; a lone packet's timing is exact (first arrival
`Σ dx/v` along the path); for constant local `L` the eventual first-face
transmission of a merged bin, `Σ m_i e^{−(dx − x_i)/L}`, is preserved
exactly by the logarithmic mean; at `r = 0` the mass-weighted position is
preserved exactly. `v = 0` is a valid no-motion step and `r = 0` a valid
no-deposition law.

**Approximations.** (i) The timing of merged mass: constituents lose their
individual positions, so the export RATE of a continuous source pulses;
the export INTEGRAL is exact and the pulsation shrinks with `B` (§5).
(ii) When `L` or `v` change after a merge, the logarithmic mean is no
longer the exact mixture (it was formed under the old `L`); a bin
refinement study is the qualification, not a proof. (iii) Positions are
measured along the CURRENT outflow direction; a reroute keeps the
coordinate (same order of approximation as the existing pool following a
new receiver; to be handled explicitly at integration). No source-assigned
`L` memory is claimed: deposition uses the local current law (Phase 5
policy).

## 3. Array state and API (`src/maple_syrup/characteristic_transport.py`)

- `PhaseState(fraction, position_m, dx_m, n_bins, xp)`: two FP64 arrays
  `(ny, nx, nc, B)`. `fraction` partitions the authoritative MAPLE mobile
  total of each cell/class (sum 1 where the total is positive);
  `position_m` is the bin representative in `[0, dx)`, 0 on empty bins.
  Canonical empty field: `fraction[..., 0] = 1`, rest 0, positions 0; the
  same canonical form is required wherever the mobile total is 0. MAPLE
  remains the only mass authority; the fractions are a distribution.
- `empty_phase_state(shape, n_classes, n_bins, dx_m, *, xp=None)`,
  `validate_phase_state(phase, mobile_kg, network)`, `bin_index(...)`,
  `fraction_sum_tolerance(B) = summation_error_bound_kg(B + 1, 1)`,
  `operation_count(n_substeps, B) = 12 n_sub (B + 1) + 8` (declared
  rounding count for the MAPLE bound helper; not a mass floor).
- `characteristic_step(network, mobile_before_kg, phase, pickup_kg, v, r,
  settle, dt, *, n_substeps=1, courant_max=0.5, implementation="array")`
  → `CharacteristicStep(transport: TransportStep, phase_after,
  mobile_before_kg = M + P, mobile_remaining_kg = T − Dep − E,
  arrival_deposition_kg, crossing_kg, n_bins, implementation)`.
  `transport` obeys the existing `TransportStep` contract (the event
  driver will still publish `T = M_remaining + Dep + E` before the deposit
  call); `phase_after` describes the pool AFTER the requests.
- Defaults: `B = 32` is a candidate, not accepted; `1..128` accepted so the
  convergence study can run; Courant limit `0.5`; `n_substeps` chosen by
  the caller (`ceil(2 v_max dt/(dx))` will replace the current
  `ceil(v_max dt/dx)` at integration).
- Validation before any arithmetic: shapes, dtypes, one namespace,
  finiteness, sign, zero mass/pickup/velocity on inactive cells, phase
  invariants (sum, canonical empties, bounds, bin containment), Courant;
  one batched flag read. After the step: finiteness, sign, request bounds,
  cell and class identities against the declared bounds, fraction sums,
  position bounds and bin containment, export only at outlets.

## 4. Implementations

**Array (NumPy or CuPy, `implementation="array"`).** One candidate per
slot: `(n, nc, B + 1)` masses, positions, destination cell and bin,
`O(n nc B)` memory, no `O(B²)` work. Receiver properties by gather, arrival
deposits and inflow by MAPLE `scatter_add` over the receiver index (one
scatter each), bin merge by `scatter_add` (masses, log-sum-exp numerators,
first moments) and `scatter_max`/`scatter_min` (per-bin extrema with
finite sentinels, as the MAPLE extrema require finite values). All are the
pinned MAPLE backend helpers, deterministic on CuPy. No host reads beyond
the two batched flag resolutions; no Python loop over cells (the loop is
over substeps). Mixed namespaces are refused, never transferred.

**Compiled host (`implementation="numba"`, `characteristic_numba.py`).**
Three passes mirroring the array path (candidates and extrema; log-sum-exp
numerators; representatives with containment) in one nopython call per
substep; no `fastmath`, no `prange`, no fallback (missing Numba raises
`NumbaUnavailableError`, a `TransportError`). Candidate accumulation order
is the flattened order `np.add.at` uses, so the two host paths agree to
round-off; a parity test (rtol 1e-12) checks it rather than assuming it.
The compiled path accepts read-only inputs (frozen phase arrays) and
compiles a second specialisation when it first sees writable ones.

Memory: state `2 · 8 · ny nx nc B` bytes (Plot 1, `B = 32`: 3.7 MB);
working set of the array path about twelve `(n, nc, B + 1)` FP64 arrays
(Plot 1: ~23 MB). Cost has not been measured by the author.

## 5. Tests (`tests/phase7b/test_characteristic_transport.py`; run by the author on 2026-10-01: 37 passed, 1 GPU skip, plus the merge regression T12b)

| Test | Reference | Pass mark |
|---|---|---|
| T01 impulse on a 6-cell chain, `L = 0.05`, `v ∈ {0.01, 1/64}`, `B ∈ {8, 32}`, both implementations | legacy `exp(−k dx/L)` crossings, bins `e^{−k Pe} − e^{−(k+1) Pe}`, first arrival `n dx/v` | rtol 1e-12; exact step; closure 1e-13 |
| T02 no deposition | front after exactly 32 steps per cell; `Σ m x` advances by `M v dt` under mass-weighted merging | exact / rtol 1e-12 |
| T03 two cells with differing `v`, `r` | closed-form split `1 − s₁`, `s₁`, `s₁ − s₂`, `s₂` at `x = d` | rtol 1e-13 |
| T04 `v → 0`, `r → 0`, settle everywhere | no motion / no deposition / whole pool + pickup settled, canonical phase | exact |
| T05 pickup beside an existing pool | separate bins, exact masses and positions | rtol 1e-14 |
| T06 dry receiver | arrival settled through the receiver's request | rtol 1e-13 |
| T07 valley, pure south, pure west | identities, face signs, export only at outlets, gross = Out + E | 1e-12 |
| T08 random network, 4 classes, 1 and 3 substeps | identities every step, global closure | rtol 1e-12 |
| T09 immutability, refusals, inactive cells, Courant rejection and substep recovery | — | exceptions / equality |
| T10 `n_substeps = 2` vs two calls | — | 1e-13 (masses), 1e-12 m (positions) |
| T11 CuPy vs NumPy on chain, west plane, valley | — | rtol 1e-10; skipped without a device, no claim |
| T12 continuous Gaussian source, `B ∈ {4, 8, 16, 32, 64}` | UNMERGED packet reference (exact delay `dx/v`, every packet separate) | export total rtol 1e-12; rate relative L2 ≤ 7.5 / 3.7 / 1.9 / 1.1 / 1e-9; centroid within 15 / 2.5 / 1.0 / 1.0 / 1e-6 s; first export within 160 / 12 / 3 / 2 / 0 s; L2 non-increasing in `B` |
| T13 Numba vs array | — | rtol 1e-12 |

The T12 limits are predeclared from Codex's probe values (6.73 / 3.34 /
1.73 / 1.000 / 3e-15; 13.1 / 1.6 / 0.50 / 0.50 / 0 s; 203 / 59 / 52 / 51 /
50 s). They document the pulsation explicitly; instantaneous smoothness is
not promised. `B = 64` is exact for this source only because consecutive
packets never share a bin.

## 6. What this kernel does and does not settle

Settled by construction and (once run) by test: the upwind leak is gone
(transmission is `exp(−dx/L)` for a lone packet, not `L/(L+dx)`), speed and
mean distance are physical, conservation is exact to FP64, MAPLE stays the
mass authority. Open: the bin count for Plot 1 (convergence study at
integration), the cost of the array path on CPU versus the compiled path,
the CuPy path's actual timing, reroute handling of positions, checkpoint
persistence of the phase state, and the storm-scale comparison with
MAHLERAN (the ring / pool / clipping ledger remains a separate audit).

## 7. Integration plan (next assignment, not done here)

`sediment_event`: add `mobile_position_fraction` and `mobile_position_m`
(or the `PhaseState`) to `SedimentEventState`; replace step 4 by
`characteristic_step(network, state.bed.water.mobile, phase,
pickup.actual_removal_by_cell_class_kg, ...)`; publish
`transport.mobile_after_transfer_kg` and issue the same deposit/export
call; store `phase_after`; `n_sub = ceil(v_max dt/(dx · 0.5))`.
`checkpoint`: schema bump, encode/validate the two arrays against the
snapshot's mobile mass. `complete_event`: canonical phase after terminal
deposition and in the dry state. Reroute: explicit policy (keep coordinate
along the new direction, record count). Benchmark: `--transit-bins`
option, Plot 1 frozen runs at `B ∈ {8, 16, 32, 64}` with hydrology
bitwise unchanged.

## 8. Caveats for the reviewer

- The kernel tests, the integration tests and the full CPU suite were run
  by the author with the pinned environment (integration report); Codex's
  independent acceptance, the full Plot 1 storms, the bin-count study and
  the GPU runs are still to come.
- `tests/phase7b` is now in `pyproject.toml` `testpaths`.
- Predeclared T12 limits are derived from a probe with the same arithmetic
  and passed; they document the pulsation, they are not proofs of accuracy
  for arbitrary sources.

## 9. Files

- `src/maple_syrup/characteristic_transport.py` (new)
- `src/maple_syrup/characteristic_numba.py` (new)
- `tests/phase7b/__init__.py`, `tests/phase7b/test_characteristic_transport.py` (new)
- `docs/phase7b/characteristic_design.md` (this file)
- `agent_handoffs/tasks/phase7b_sediment_fix/kernel_report.md`
