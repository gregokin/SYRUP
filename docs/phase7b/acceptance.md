# Phase 7b transport correction — bounded acceptance

The coarse-grid transmission defect is corrected. The event now uses bounded
characteristic transport, with the old upwind operator retained as an explicit
comparison option. Actual MAPLE still owns the bed, pickup, mobile totals,
deposition and export. No wind physics or reference-tree code was copied into
the implementation.

This is a bounded numerical correction. It does not establish whole-storm
sediment equivalence to MAHLERAN, convergence of one-second export peaks at the
default 32 bins, a full GPU event, or production performance qualification.
The half-second timestep trial is complete and supports the cumulative metrics,
while confirming that the default-bin instantaneous peaks remain unresolved.

## What changed and why

The old well-mixed cell operator tends toward a first-cell transmission
probability `L/(L+dx)`. The selected MAHLERAN upstream-face distance convention
requires `exp(-dx/L)`. When `L` is much smaller than a cell, these differ by
orders of magnitude; shrinking only the timestep does not remove that error.

The replacement retains a bounded distribution of within-cell positions,
advects at the computed sediment velocity, deposits along the travelled path,
and exports only when material reaches a boundary face. Newly picked-up mass
comes from actual MAPLE removal and starts at the upstream face. Merging
positions within a bin preserves the constant-local-law eventual face-crossing
mass. Changing local hydraulic laws and unresolved within-bin timing remain
approximations and require refinement checks.

Phase state survives retries, terrain rerouting and wet restart. It is cleared
canonically after actual terminal deposition. Checkpoint schema 2 includes the
phase partition and rejects incompatible schema 1 checkpoints. Direct coupled
steps now reject mismatched scheme/bin controls before hydraulic or MAPLE work.

## Full Plot1 measurements

All runs use the same frozen elevation/routing, applied rainfall, deterministic
conductivity, six classes and actual 20-voxel MAPLE bed as Phase 7. Source,
forcing, dependency and reference identities are bound in each run's summary.

| Run | Export (g) | Class 1 / 2 share | Raw peak (g/s) | Raw peak time (s) |
|---|---:|---:|---:|---:|
| Original SYRUP upwind | 218.110 | 24% / 73% | See Phase 7 | 1141 |
| Characteristic, 32 bins | 13.76398 | 69.7% / 30.3% | 0.21809 | 1145 |
| Characteristic, 64 bins | 13.76316 | 69.7% / 30.3% | 0.17674 | 1132 |
| Characteristic, 128 bins | 13.76260 | 69.7% / 30.3% | 0.18332 | 1143 |
| Characteristic, 32 bins, dt=0.5 s | 13.73271 | 69.7% / 30.3% | 0.34747* | 1148 |
| MAHLERAN reference | 9.22325 | 80.9% / 19.1% | 0.06918 | 1261 |

\* This is the true per-half-second peak. Its maximum one-second interval
average is 0.24401 g/s; both measures are retained in the qualification JSON.

The 32-to-128-bin total changes by 0.0100%. All three give the same one-second
10%, 50% and 90% cumulative-export times: 1082, 1143 and 1195 s. Their
endpoint-weighted centroids are 1133.316, 1133.335 and 1133.339 s, respectively.
The reference centroid is approximately 1315.947 s. Thus the substantial
remaining timing difference is present in cumulative timing, not just a
single sampled spike.

At 32 bins, halving dt changes total export by -0.2272%, the endpoint-weighted
centroid by +0.1170 s, and the 90% cumulative time by +1 s; the 10% and 50%
times are unchanged. The raw peak increases substantially as its sampling
interval shrinks. This is further evidence against using the default-bin
instantaneous peak as a converged prediction. The 10-second averaged peak
changes by about -0.15%, but this does not replace the raw-peak qualification.

Raw peaks are more sensitive than totals: 64 and 128 bins differ by 3.73% in
peak magnitude and 11 s in peak time; 32 bins has a larger pulse. Diagnostic
averaging over 5, 10, 30 and 60 seconds is retained alongside the raw peaks,
not substituted for them as an acceptance criterion. Do not interpret a
32-bin one-second peak as a converged storm prediction.

All class budgets and request reconciliations close under the unchanged MAPLE
policy. The saved water histories and synchronous peak water maps are bitwise
unchanged from Phase 7 at dt=1 s. Geometry freeze checks pass. No phase
canonicalization or rounding-remnant events occurred in these storms; the
maximum phase reconciliation difference was about 1.4e-17 kg.

The residual export ratio is about 1.49, and the class composition and timing
remain different. The [legacy ledger audit](../phase7/legacy_sediment_ledger.md)
measures 0.292 kg of clipping-induced source and 0.308 kg of final mobile
inventory in MAHLERAN. Its 9.2 g export is an observed comparison, not a
conservation-correct calibration target. No parameters were tuned to it.

## Verification and provenance

Claude implemented and corrected the kernel/integration; Codex independently
reviewed callers, reproduced and checked the merge and direct-entry defects,
ran analytical probes and executed the full storms. The last broad CPU run,
before the final input guard,
passed 565 tests with 9 skips. After the final input-validation correction,
Codex's independent kernel, coupling, checkpoint and original-Fortran selection
passed 91 tests with 1 GPU skip. The corrected GPU parity test passed on an
actual GTX 1080 Ti, as did an independent wide-dynamic-range merge regression.

The immutable full-storm trial source is recorded in
`benchmarks/phase7b/trial_source_manifest.json`; the source archive is retained
under `agent_handoffs/tasks/phase7b_sediment_fix`. The validated current source
has digest `916746988acd7c55902fdbaf818047ac45e57df223f7d818459f92a62f1c20ce`.
The only subsequent source differences are the public input-validation guard
and a docstring attribution correction, recorded in
`benchmarks/phase7b/trial_to_validated_source.patch`. They do not alter numerical
operations for the validated trial inputs. Exact trial hashes remain in the
run summaries; the benchmark is not relabelled as a different source revision.

## Performance and remaining work

The unchanged Phase 7 profile establishes the original slowdown: MAPLE bed
transactions took 83.7% of the coupled loop; hydrology took 4.1%. See the
[performance diagnosis](../phase7/performance_diagnosis.md).

The corrected method has additional cost. The observational 32-bin trial took
322 s in the coupled loop, with 119 s in characteristic transport and 172 s
in MAPLE exchanges. It overlapped other verification, so it is not a controlled
speedup/slowdown measurement. Larger bin counts cost more. The original 84%
share must not be presented as the profile of the corrected implementation.
At 64 bins the measured components are approximately 262 s for the phase
operator and 184 s for MAPLE; at 128 bins, 470 s and 176 s. The phase operator
becomes the largest component in these larger-bin trials. The half-second,
32-bin run takes 631 s for 10,800 steps; that total must not be compared to a
5,400-step run as a same-timestep speed ratio.
Both shared MAPLE transactions and the new phase operator are optimization
targets; accelerating water routing alone will not resolve the full-model cost.

Keep open: the legacy export/composition/timing differences, instantaneous-rate
resolution, efficient phase processing and GPU memory scaling, full coupled
GPU execution, and broader cases/terrain evolution. Retain analytical and
conservation tests when optimizing; do not emulate negative-mass clipping or
silently change virtual velocities to obtain a matching outlet curve.

Reproducible measurement scripts and condensed results are in
`benchmarks/phase7b`; `characteristic_qualification.json` retains raw peaks,
windowed rates, cumulative timing, conservation, source identities and
observational performance for all four runs. Claude's read-only review and
Codex's disposition are archived under
`agent_handoffs/tasks/phase7b_sediment_fix`. No commit or push was performed.
