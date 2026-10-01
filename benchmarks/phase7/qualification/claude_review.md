# Phase 7 final independent review (Claude, read-only)

Date: 2026-09-30. Reviewer role: independent, read-only. This file is the
only artefact written by this review. No source, test, documentation,
output, upstream MAPLE, MAHLERAN or dependency snapshot was modified, and
nothing was executed. All numbers below are read from Codex-produced
artefacts under `agent_handoffs/tasks/phase7_matched_benchmark/` and
`outputs/phase7/`; I did not reproduce them.

Production source under review: `src/maple_syrup` digest
`01709bb52720923aa9b7211409d7ffd19155c4a9bfc87c36d9e1f0180a58d532`
(recorded identically in `distance_resolution.json`,
`gpu_plane_after_fix.json` `source_before`/`source_after`,
`scaling_*.json`, `array_parity_and_spatial.json`). Pinned MAPLE digest
`d3d007024ff65abc2a0ff179f0f03bcd1c3b27f091493136cce849c34f7a4264`.

## 1. Verdict

- **No blocking defect** in the reviewed Codex-authored scripts, the GPU
  parity test correction, or the acceptance draft's factual statements.
- **Measurement baseline: usable as reproducible diagnostic evidence
  only.** Hydrology targets pass, water and per-class sediment closures
  hold within unchanged MAPLE bounds, artefacts are digest-bound, and the
  array/Numba paths agree bitwise. It is not an accepted sediment
  benchmark.
- **Sediment fidelity remains OPEN.** Whole-storm export is ~23.6× the
  MAHLERAN reference, the exported class composition is inverted, and the
  export-rate peak is 120 s earlier. The upwind first-cell leak mechanism
  is proven by the impulse probe; attribution of the exact whole-event
  ratio is not quantified.
- Items that must stay unqualified after this milestone: full Plot 1
  spatial-grid convergence; full coupled GPU event runtime, transfer cost
  and device peak memory.
- Next bounded step, as tracked by Codex: Phase 7b distance-resolved
  conservative transport investigation, separate from Phase 4R hydraulics
  and Phase 8 wind. Nothing in this review proposes calibration or a
  physics change; deciding on a transport redesign is the user's call.

## 2. Corrections to my earlier `spatial_review.md`

The original `spatial_review.md` is retained unchanged as an audit record.
The following assertions in it were overstated and are corrected here.
The corrected statements supersede the originals.

**C1. "`max_decay_exponent` 0.37 with `v ≤ 0.024 m/s` implies `L ≥ 0.065 m`"
(spatial_review.md §3.1, line 135). Withdrawn.** Global maxima of `v` and
of `v·dt/L` need not occur in the same cell, so their ratio bounds nothing.
The measured pickup-weighted histogram (`distance_diagnostics.json`,
`pickup_weight_kg_by_class_and_distance_bin`, log-spaced edges
`10^(-12 + 0.25·i)` m) shows far smaller local `L` for the coarser
classes: class 1's weighted median lies in bin 43 (0.0562–0.1 m) and
class 2's in bin 41 (0.0178–0.0316 m). I re-summed the histogram rows and
confirm both bins. Class 2, which carries 248 of 453 kg of demand, is
therefore mostly at `dx/L` ≈ 16–28, where the legacy exit `exp(−dx/L)` is
10⁻⁷ to 10⁻¹² and the upwind exit `L/(L+dx)` is 0.03–0.06. The table in
§3.1 remains algebraically correct for the `dx/L` values listed, but the
storm's mass is not where the original text placed it.

**C2. "Detachment laws … refuted by the +0.05 % demand agreement"
(§3.1 item 1, §3.3). Overstated.** Agreement of the whole-storm summed
demand (453.081467 kg vs 452.8581 kg) does not exclude compensating
spatial, temporal or per-class detachment differences. The supportable
claim is only that total demand does not explain a 23.6× export ratio.

**C3. "Ring walk deposits … a minor contributor" (§3.2, last bullet).
Withdrawn.** Legacy export is 0.00922 kg out of ~450 kg picked up, i.e.
2×10⁻⁵ of demand. Any bookkeeping difference affecting even 0.1 % of
outlet-adjacent pickup is of the same order as the entire legacy export.
The ring-deposit-versus-face-export classification difference cannot be
ranked without a measured ledger; its direction is still that it raises
SYRUP export relative to legacy bookkeeping, but its magnitude is unknown.

**C4. "The 0.0159 kg gap between legacy net erosion and export is
consistent with stranded pool plus negative-clipping mass … works against
the 24×" (§3.2, first bullet). Withdrawn as an inference.** No mobile-pool,
ring-deposit or clipping ledger was extracted from the Fortran run, so
neither the cause nor the sign of that gap is established. It is an
unexplained legacy residual to be measured, not a known offset.

**C5. Attribution of the whole-event ratio (§3.1 "A 24× total is reached
for dx/L ≈ 5").** The mechanism is proven (see §3.1 below); the whole-event
attribution is not. The per-class prediction formula in §3.1 is a
proposal for Phase 7b, not a result. Composition inversion and timing are
consistent with the mechanism but remain unproven signatures.

Statements in `spatial_review.md` that stand: the comparator defect and
its correction; the observer-script and GPU-test reviews; the identity of
the legacy distance convention (`flow_distrib.for`), the pool routing and
clipping (`route_sediment_xml.f90` 284-292), and the SYRUP operator
(`sediment_transport.py`, physics.md §3); that `dt` refinement cannot
remove a `dx`-controlled exit fraction.

## 3. Review of Codex-authored scripts

### 3.1 `benchmarks/phase7/distance_resolution_probe.py` — sound

- Builds a 5×3 full grid with walls (`z[:, [0, -1]] += 1.0`), south ring
  as outlet, giving a 3×1 active chain; unit impulse in the outlet-adjacent
  cell; `velocity` 0.01 m/s, `rate = 1/L`, no settling; loops the actual
  `transport_step` until the remaining mobile mass is below 1e-14.
- The mobile update `mobile_after_transfer − export_request −
  deposition_request` is consistent with the operator's contract
  (`mobile_after_transfer_kg − pool == divergence_kg`, asserted in
  `tests/phase7/test_face_topology_backends.py:103`); had the requests
  already been removed the conservation assertion at line 50 would fail.
- The independent geometric series `a·s_h/(1−(1−a)·s_h²)` with
  `a = v·dt/dx`, `s_h = exp(−v·dt/(2L))` is the closed form of the
  reaction–advection–reaction split; the actual export matches it to
  `rtol 1e-11` in all six rows (`distance_resolution.json`).
- Evidence (dx 0.5 m): L = 0.05 m exports 0.0915624 (dt 1) / 0.0911050
  (dt 0.25) vs legacy `exp(−dx/L)` = 4.54e-5 and dt→0 limit 0.0909091;
  L = 0.01 m: 0.0189696 / 0.0196029 vs 1.93e-22; L = 0.1 m: 0.167974 /
  0.167009 vs 6.74e-3. Conservation residual ≤ 1e-14 in every row.
- Scope statements are accurate: a constant-law mechanism probe, not a
  Plot 1 run, not a legacy execution, not an attribution.
- Observations, not defects: the friction argument `np.ones((3,1))` is
  irrelevant because velocity is prescribed; the first launch failed on a
  1×1 chain with `RoutingGraphError` (`distance_resolution.log`) and was
  rerun with the 3-cell chain (`distance_resolution2.log`), which is the
  version whose script hash is recorded in the JSON.

### 3.2 `task/observed_array_run.py` — sound (re-confirmed)

- Wraps `sediment_physics_step` and `sediment_coupled_step` read-only,
  restores the originals in `finally`, asserts one physics call per
  accepted step, zero rejections, and that the observed peak time/value
  equal the result's own (`lines 48-51`).
- Histogram weights are `requested_pickup_kg` at the local `L` at pickup
  time; this is correctly labelled as a pickup-time local mean distance,
  not a particle-lifetime measurement.
- Recorded totals: demand by class 25.894 / 247.958 / 26.780 / 149.003 /
  3.357 / 0.0898 kg (sum 453.0815, equal to the dt1 summary demand); all
  raindrop-driven, zero flow pickup. Mass with `L < 0.5 m`: 452.633 kg =
  99.901 % (I recomputed the ratio). Only class 1 has any mass with
  `L ≥ 0.5 m` (0.448 kg, up to bin 48, 1.0–1.78 m).
- Timing from this run is cProfile-instrumented and must not be quoted as
  Numba performance; the script's own scope string says so.

### 3.3 `task/compare_observed.py` — sound, one limitation

- Asserts identical `maple_syrup` provenance between the Numba and array
  runs, compares all 97 arrays with `np.array_equal` (allclose fallback
  never triggered: `all_values_bitwise_equal: true`, every
  `max_abs_difference` 0.0), asserts both closures closed, and binds the
  snapshot to the source digest.
- Uses the corrected `mahleran_asc_interior` and `field_metrics` on the
  synchronous peak-outlet snapshot: relative L2 0.136173 % depth,
  0.121216 % velocity at SYRUP's exact peak time 1334 s vs the legacy
  rounded plateau 1328–1342 s.
- Limitation (declared in the report scope, not a defect): the SYRUP
  snapshot is the post-step routed field; MAHLERAN's `dmax`/`vmax` are the
  in-step fields when `q_plot` first exceeds its running maximum. The two
  can differ by up to one step and by the unrecorded exact legacy peak
  time. This bounds, but does not invalidate, the 0.13 % agreement.

### 3.4 `task/scaling_probe.py` — sound, reporting limitation

- Synthetic initially wet plane, fixed ksat 2.5e-7 m/s,
  `commit=False, force_final_commit=False`, Numba, 1 s warm step then a
  20-step window; asserts closure, zero commits/graph changes and an
  unchanged source digest afterwards.
- The `hasattr` loop at line 38 records only fields that exist on
  `SedimentEventResult`; `maple_wall_s`, `physics_wall_s`,
  `transport_wall_s`, `hydrology_wall_s` do not, so the JSONs carry only
  `n_accepted_steps`, `n_rejected_attempts`, `commit_wall_s`,
  `n_maple_water_calls`. No per-component split is available from these
  runs. This is a reporting gap, not a defect.
- All four JSONs (1,200 / 4,800 / 19,200 / 76,800 cells) exist and report
  `closed: true`, `request_reconciled: true`. Warm 20-step window:
  0.456 / 1.572 / 6.399 / 26.601 s; peak RSS 340,012 / 418,148 / 730,860 /
  2,012,352 KiB. Growth is close to linear in cell count (×3.4, ×4.1,
  ×4.2 per ×4 cells). The 1,200 and 4,800 `time` logs show 210 % and
  177 % CPU, so these are not single-thread numbers. This is a synthetic
  no-rain wet-plane window, not a Plot 1 storm, and not comparable to the
  35.5 ms/step dt1 figure.

### 3.5 `task/gpu_kernels.py` — sound within its scope

- Isolated wet-law and lateral-transport kernels on a converging valley,
  CPU vs CuPy, `rtol 1e-10, atol 1e-18`, five synchronized samples with
  first-call JIT separated, source digest checked before and after.
- Scope string correctly excludes the MAPLE bed, routing, commits and the
  coupled event; memory is CuPy pool reservation, not device peak.
- Not a defect, but worth noting: `atol 1e-18` is effectively pure
  relative tolerance, so the parity claim is a strict one.

### 3.6 GPU parity test correction (`tests/phase7/test_face_topology_backends.py`)

Previously reviewed as sound; re-confirmed. `budget_residual_by_class_kg`
was removed from the field-by-field device/host list (lines 124-130) and
both backends independently satisfy the unchanged MAPLE bound
(`check_single_orientation`, lines 87-88). Physical arrays and totals
keep `rtol 1e-12, atol 1e-15` parity. The face-topology assertions
(absent-orientation diagnostics exactly zero; present orientation carries
every crossing; export on the correct ring face) are unchanged.

## 4. Acceptance draft (`docs/phase7/acceptance.md`) — interpretation check

Every numerical statement I could trace matches its evidence file:
hydrology targets (`comparison_dt1_corrected/comparison.json`: −0.3778 %,
−0.0166 %, peak inside plateau); export ratio 23.6478; demand, actual
pickup and refusal; conservation residuals and bounds; impulse-probe
values; histogram medians; GPU kernel table (`gpu_valley_after_fix.json`);
whole-program 195.58 s vs 9.297/9.398 s. Its interpretation agrees with
the required reading:

- hydrology targets and conservation pass;
- GPU kernel parity is not a full GPU event;
- the 21-fold whole-program contrast is not a water-routing speed ratio;
- the profiled array-run times are not a Numba comparison;
- sediment fidelity is explicitly not established.

Two things the draft should state when Codex finalises it:

1. **dt refinement results are now complete and should be quoted.**
   `dt0p5_stdout.json` and `dt0p25_stdout.json` both report
   `n_accepted_steps` 10,800 / 21,600, zero rejections, `closed: true`,
   frozen geometry true, water residual 1.64e-13 / 2.05e-13 m³ under
   bound. Export 0.2184541 (dt 0.5) and 0.2186214 kg (dt 0.25) vs
   0.2181098 (dt 1): the change is +0.16 % then +0.08 %, i.e. converging
   in `dt` while the ~23.7× ratio persists. `comparison_dt0p25/` exists
   and passes all three hydrology targets (−0.3735 %, −0.0150 %, peak in
   plateau). Peak export time stays 1141 s at all three steps. This is
   direct storm-scale confirmation that the excess is not a temporal
   discretisation artefact, consistent with the probe. dt 0.25 wall time
   is 809.31 s, peak RSS 357,940 KiB (`dt0p25_stderr.log`).
2. **The synthetic coupled scaling table (§3.4 above) is available** and
   should replace "large-domain coupled CPU profiling … in progress", with
   the caveats that it is multi-threaded wall time and no component split
   was recorded.

The draft's "Correct spatial comparisons" section is accurate and now
consistent across `matched_benchmark.md`, `mahleran_reference_run.md`
(line 82) and `audit_mahleran.py` (line 128).

## 5. Confirmed defects vs scope gaps

**Confirmed defects in reviewed material: none.** The previously confirmed
defects (comparator misuse of `depth001`/`veloc001`; CuPy empty-face
scatter; fixture placement; lint) are all corrected and evidenced in
earlier reports and logs; no new defect was found.

**Scope gaps and limitations (not blockers for a diagnostic baseline):**

- G1. Sediment export, composition and timing do not match MAHLERAN
  (23.6×; 24.19/73.12 % vs 80.93/19.07 % classes 1/2; 1141 vs 1261 s).
  Mechanism proven, attribution unquantified (§2 C5). Phase 7b.
- G2. No legacy mobile-pool / ring-deposit / clipping ledger has been
  extracted, so the legacy net-erosion-minus-export gap (0.0159 kg) and
  the ring-versus-face classification difference are unmeasured (C3, C4).
- G3. Plot 1 spatial-grid convergence not run; the operator is first
  order in `dx` and the leak is `dx`-controlled, so this is the relevant
  refinement axis and remains unqualified.
- G4. Full coupled GPU event unsupported: runtime, transfer cost and
  device peak memory unmeasured. Kernel parity and timings only.
- G5. Scaling probe lacks a component-time split and is multi-threaded;
  no single-thread or per-component profile exists for the Numba path.
- G6. Synchronous peak snapshot comparison carries a ≤ 1-step definition
  offset and an unrecorded legacy exact peak time (§3.3).
- G7. Frozen-geometry mode only; no completion, dry reset, commit,
  restart or wind handoff exercised in this benchmark (by design).
- G8. Only rain-assisted detachment is exercised; concentrated-flow and
  suspension regimes remain untested against the reference.

## 6. What the baseline supports and what must remain open

Supported, reproducible, digest-bound:

- Matched forcing, frozen geometry, deterministic ksat 2.5e-7 m/s.
- Hydrology within predeclared targets at dt 1, 0.5, 0.25 s.
- Water and per-class sediment closures within unchanged MAPLE bounds;
  no negative mass erased; ledger reconciled; availability refusal
  reported (42.02 kg) rather than hidden.
- Numba/array bitwise equivalence over 97 arrays; CPU/CuPy kernel parity
  on both topologies; empty-face guard verified on device.
- The upwind first-cell exit fraction of the actual operator, its
  closed-form match, and its `dt`-independence.
- Pickup-weighted local `L` distribution: 99.90 % below `dx`, class 2
  median 0.018–0.032 m.

Must remain open:

- Whether SYRUP's sediment export, sorting and timing are physically
  acceptable relative to MAHLERAN, and by what quantified mechanism they
  differ (G1, G2).
- Grid convergence of the Plot 1 sediment pattern (G3).
- End-to-end GPU acceleration and memory (G4).

No calibration, tolerance relaxation, legacy-clipping adoption, physics
change or wind run is implied or recommended by this review.

## 7. Files read for this review

Scripts: `benchmarks/phase7/distance_resolution_probe.py`,
`task/observed_array_run.py`, `task/compare_observed.py`,
`task/scaling_probe.py`, `task/gpu_kernels.py`,
`tests/phase7/test_face_topology_backends.py`.
Evidence: `distance_resolution.json/.log/2.log`,
`distance_diagnostics.json`, `array_parity_and_spatial.json`,
`gpu_plane_after_fix.json`, `scaling_{1200,4800,19200,76800}.json` and
`_time.log`, `dt0p5_stdout.json`, `dt0p25_stdout.json`,
`dt0p25_stderr.log`, `hardware.json`,
`outputs/phase7/comparison_dt1_corrected/comparison.json`,
`outputs/phase7/comparison_dt0p5/comparison.json`,
`outputs/phase7/comparison_dt0p25/comparison.json`.
Documents: `docs/phase7/acceptance.md`, `final_review_prompt.md`,
`codex_spatial_disposition.md`, my own `spatial_review.md`,
`docs/phase7/mahleran_reference_run.md` (line 82),
`benchmarks/phase7/audit_mahleran.py` (line 128).
`final_review_stdout.json` was empty at read time (this session's own
capture file).
