Current qualified acceptance (2026-10-14): [initial CPU/GPU water component](acceptance_20261014.md), independently verified on the adopted working tree; uncommitted. Historical dated entries below retain their original scope.

Current consolidated status: [2026-10-14 qualification and active work](current_status.md). The dated entries below retain historical context.

Primary adoption status (2026-10-14, CW-E47; pending root independent verification): the root working tree now carries the exact reviewed SYRUP 5f runtime `5f85a989…` and the CW-E46 MAPLE 24c runtime recipe (package `24c6e147…`, runtime-only bundle `7ee10bd8…`); B55/M44 is historical. See [current status](current_status.md) and [usage](../conservative_water_usage.md). Uncommitted.

Current qualification (2026-10-14 UTC): rotated/resampled Chastre now completes the bounded 300 s wet slice on both CPU/Numba and GPU0 without changing the original resource caps. The GPU fresh-process own-240 s restart matches 184 numeric members exactly. Both runs have 300 accepted steps, no rejected attempts, 24 accounting commits and no hydraulic reroutes; their net displacement stays below the 0.75 mm trigger. The completed [backend comparison](rotated_backend_comparison.md) passes physical-field and decision checks while retaining four/seven strict near-zero accounting/residual failures at 240/300 seconds, without changing bounds. This is not a full Chastre storm or native-terrain Fortran comparison. Exact overlay storage and SYRUP integration passed bounded independent checks; current-source public Plot1 comparisons and large overlay qualification are being completed. See [net-elevation routing](net_elevation_routing.md), [storage](exact_overlay_storage.md) and [current implementation status](implementation.md). The following dated records are historical.

Current recovery status (2026-10-10T00:32:07.502100+00:00): The Chastre wet300 GPU attempt F11 exited1 on timestep2, at deposition availability fraction selection:107,250,176B requested with9,014,264,320B retained under the unchanged9,092,045,209B pool cap. One infiltration-only step returned; no wet240/300 checkpoint or output was produced. Root rehashed sources and INITIAL unchanged; GPU0 released. The earlier B83one-step fit remains valid only for that first second. Claude is implementing B85 solely in a NEW isolated MAPLE candidate, to reuse dead private fraction scratch without changing physics/caps. E24 CPU INITIAL validation and C24 full RFID CPU remain active; the long M82 CPU300 run is held to avoid qualifying a superseded candidate. Final-source Plot1 CPU complete/wetrestart remains209arrays exact. RootB55/M44 and legacy default are unchanged; whole water acceptance remains open. Earlier entries below are historical.

Current recovery status (2026-10-10T00:21:01.828883+00:00): Chastre original-cap GPU trial B83 passed one accepted infiltration-only second, with closed budgets and a verified checkpoint; tracked live peak8.882GB under the unchanged9.092GB pool cap. This is not wet/full evidence. F11 is actually running the reviewed GPU wet300-second reference with own240-second checkpoint; E24 is validating an INITIAL-only reimport of the preserved, disjoint CPU bed before its matched G8 comparison. Final-source Plot1 CPU full storm and fresh wet restart passed209rawnumeric arrays and fullmetadata/reset checks;209numeric members also match the priorM79 full CPU run. G8/G7 preparation and E22/E23 delivery are bounded accepted. Claude is auditing root adoption and remaining gates read-only. Root runtime/dependency and legacy default remain unchanged; final large wet/peak/completion/restart and final adoption gates remain open. Earlier dated entries below are historical.

Current recovery status (2026-10-09T23:49:03.581047+00:00): B82/M82 private availability-lifetime fix is independently accepted and frozen (1667efe2): 73 CPU tests, 20 actual GPU tests, two CPU integrated and two frozen actual GPU complete/wet-restart tests passed. One device allocation-limit fixture skipped because its maximum was outside the target. Device target savings are exact (1C earlier, 2C at each closure); whole small-fixture peak is unchanged. Large Chastre fit remains unproven. INITIAL-only metadata reimport is active, followed by the unchanged original-cap GPU trial. Root runtime/dependency and legacy default are unchanged; final adoption and whole-component acceptance remain open. Earlier dated entries below are historical.

# Resident-window water qualification

Current status — 2026-10-09 17:17 UTC: **items 2–5 remain active.** The latest original-cap Chastre GPU trial B80 on B76/M79 refused an earlier availability-closure allocation (107,250,176 bytes at 9,082,399,232 retained, unchanged 9,092,045,209-byte cap). Tracked live memory was lower than the previous refusal, but the allocator lacked a suitable free block. No accepted step, output or checkpoint was published; source/INITIAL/observer checks stayed unchanged. The B82 private availability-lifetime implementation is underway after independent design review. No large GPU fit is claimed.

The new M79 Plot1 CPU complete event and genuine wet 2400→4739 s restart pass all 209 arrays and physical/reset metadata exactly. The source-only M75→M79 CPU observation also passes all 209 arrays with verified source and INITIAL provenance. The final large CPU M79 run was deliberately stopped with SIGINT before its 300 s qualification to avoid spending hours on a superseded memory candidate; its genuine MAPLE INITIAL, deep store, logs and failure receipt are preserved for a reviewed final-source reimport. This is not a large CPU result.

The intermediate B76/M75 Plot1 and RFID CPU/GPU storms remain qualified as recorded: closed balances; Plot1 strict window difference4.09e-14 kg with separate actual-MAPLE operand correspondence PASS; RFID repeated one-cell terminal depth difference6.23e-14 m versus5.53e-14 m strict bound, with inventory correspondence PASS. Neither inventory result waives the strict failure. Terminal storage lacks spill/backwater and is not flood-valid. Older B55/M44 large wet restart is exact on163 numeric members.

E19 independently reconstructs all341 M79 runtime files; its E20 materialized bundle resolves in both existing MAPLE environments without installation. These remain intermediate delivery evidence. Root runtime/dependency remain B55/M44, legacy remains default, and final-source full storms/restarts/large wet/peak/correspondence/source adoption/final review are open. Native Fortran Chastre recession surplus is documented only.

## Earlier status records (superseded)

Current status — 2026-10-09: **items 2–5 remain active.** Root runtime/dependency are B55/M44; legacy remains the default benchmark. Later candidate results below do not imply primary adoption or complete water-component acceptance.

B71/M67 passes 95 independent CPU checks (49 device-only skips), 129 actual GPU checks, and the full Plot1 CPU comparison: all 209 numerical arrays, budgets and reset accounts raw-exact against B55/M44. Its optimized standard bed releases exactly one post-pickup mobile allocation during transport; subclass/unusual bed types retain the original path. Full B65/M67 CPU/GPU regressions also passed (209/230 arrays). M67 is a candidate dependency, not the root dependency; the M69 scatter candidate is not adopted because its whole-call peak is unchanged.

The latest original-case large GPU probe **B72 refused deposition** after all transport calculations passed. The 107,250,176-byte allocation at 9,073,047,552 retained bytes exceeds the unchanged 9,092,045,209-byte hard cap. No accepted step, checkpoint or output was published; all initial/source/observer checks passed. This now triggers the prior B48 private paired-transaction design. B73 is separately correcting reporting around terminal storage. No large GPU fit is claimed.

Large CPU Chastre has a verified wet 240-second pause with closed budgets, not completion. The F8 reference to360 seconds is running; its candidate has not started, so no large wet-restart comparison is accepted. A fresh genuine INITIAL assembly into a separate GPU deep store is in progress; it uses the existing case APIs and preserves the original CPU store. Large wet/peak GPU fit, wet restart, final-source correspondence and delivery/adoption remain open. The conservative Plot1 export is still **49.57% above native MAHLERAN**, explicitly qualified by the different conservative evolving-bed physics.

The M67 recipe is independently accepted as a delivery candidate: 57 CPU tests, all340 runtime files byte-exact, both MAPLE environments resolve the materialized M67 bundle without installation. Root recipe remains M44 until final adoption.

## Historical execution entries

The entries below retain their original task context. Their pending/current wording is superseded by the status above.


Items 2–5 remain active; root runtime is accepted B55/M44 and the legacy replay remains the default benchmark. Small full storms and same-source wet restarts pass. The following current large-case results supersede pending statements in the historical entries below.

## Current large wet checkpoint and GPU memory result

The B55/M44 Chastre CPU run reaches 240 s with 240 accepted steps, no rejection and 15 terrain commits. Its checked saved state has 541,213 positive-depth cells (maximum 0.0001542253913 m), 14.9037065371 m³ of surface water, and 19,705.9912166 kg of mobile sediment in 1,006,604 cell/classes. The CSR phase has 1,006,604 entries, all currently in bin 0; this does not establish bin convergence. Water and class-sediment/request budgets close under existing MAPLE policy. Output SHA hashes, source identities and unchanged initial import are verified. Total observed wall time is 7,368.39 s and host RSS peak 12,030,722,048 bytes, including preparation/checkpointing and concurrent work; these are not controlled performance comparisons. The wet pause is not completion or restart evidence. A same-source wet continuation from 240 to the original 300/360-second reporting boundaries is active, followed by a planned fresh-process 300-to-360 restart. An initial 242-second pause was correctly refused because it was not an original reporting boundary; the preserved failure led to correcting the experiment, not the model guard.

The B65 post-transport retention candidate passes the corrected actual-device memory tests: eight diagnostic grids plus two class totals are released before deposition, the four faces remain for actual MAPLE validation and are then released, and return-time live bytes match the old implementation exactly. A missing shared zero-demand alias in the test expectation was independently identified and corrected with explicit pointer/liveness proof, without an allowance band. Its full CPU Plot1 reproduces all 209 saved numeric arrays, physical summary fields and reset accounting exactly; C12 verifies the complete snapshot/checkpoint provenance separately against actual source hashes. Actual full GPU regression remains pending.

B66 repeats original Chastre W3/six classes/32 bins/4M capacities/history off at the same hard 85% pool cap. It passes transport and the previous deposit-shortfall failure, then refuses `active_layer/exchange.py::_remove_mass_from_active_layer` at `requested_vec - actual`: 107,250,176 bytes requested with 9,073,047,552 bytes retained against 9,092,045,209. Zero steps are published and all source/initial/observer identities remain unchanged. B67 is a separate, unaccepted MAPLE candidate for late allocation of absent zero boundary terms; no physical calculations, normalization, compensation or supplied-input validation may be skipped. The larger staged-owner design remains a fallback, not implemented.

Evidence: `cw_b55_large_cpu_wet240_receipt.json`, `cw_b55_large_cpu_wet240_observation.json`, `cw_b65_corrected_targeted_gpu.log`, `cw_b66_result_digest.json`, `cw_c12_corrected_cpu_independent_comparison.json`.

## Accepted CUDA scratch update and earlier large probes

The combined B52/B54 CUDA change releases private scratch at its last use and precomputes the original validation flags while preserving their final error priority. One scalar loop alias found by the first device run was corrected without relaxing the exact allocation assertions. The corrected matrix passes 282 actual-device checks, with no skips; coupled GPU wet/restart checks pass two cases and explicit flow-end settlement passes two dense/windowed cases.

The frozen joint B55 GPU Plot1 storm completes at 4,739 s with 4,739 accepted steps, no rejected attempts and 802 commits. All 212 saved arrays and reset accounting are raw-exact against M44/B46. Loop plus checkpoints: 708.31 s; setup: 8.15 s. Concurrent host activity makes these observational costs, not a controlled speedup. Root runtime bytes now match this frozen joint source.

The next history-off Chastre probe uses the same original case, W3, six classes, 32 bins, capacity 4,000,000 and hard 85% pool cap. B57 completed with a resource refusal at the combined-pool validation addition: 107,250,176 bytes requested with 9,055,316,480 bytes retained against the unchanged 9,092,045,209-byte cap. Pickup, the CSR kernels and all face calculations passed; zero steps were accepted and all source/initial identities remained unchanged. A GPU 0 sampler observed a maximum of 9,183 MiB over its finite sampling interval; this is a sampled lower bound, not an exact high-water measurement or proof of fit. B59 passes 370 actual GPU checks and two coupled complete/wet-restart cases. Its capped B61 diagnostic passes the combined-pool check but refuses the last non-outlet export gather of 107,250,176 bytes at the same retained total and hard cap, with zero accepted steps and all source/initial identities unchanged. A new private B63 candidate moves that identical last check before the faces while preserving its error priority. B63 passes 454 actual GPU checks (five explicitly context-specific historical B59 cases deselected) plus two coupled complete/wet-restart checks. Its capped B64 diagnostic passes the entire transport operator and all validations, then refuses the MAPLE deposit call at the availability-shortfall subtraction: 107,250,176 bytes requested at the same retained total and hard cap. Zero steps were accepted; all identities remained unchanged. This measured deposit peak triggers the reviewed B56 accounting-only retention design in new private B65. Eight dead transport diagnostic fields can be released before deposit; all four faces remain for MAPLE validation. B65 has passed bounded differential/device checks and a full CPU Plot1 regression; root runtime is still accepted B55. An independent bounded CPU run to 240 s is also active, with explicit terminal storage and flow-end settlement enabled. Its saved fields must establish wet transport; the requested duration alone does not.

The tracked dependency recipe reproduces all 340 M44 runtime files exactly. Its runtime-only bundle was materialized and resolves in both existing MAPLE environments; no installation or physical GPU execution was involved. See [the recipe and executed delivery record](../maple_water_dependency_recipe.md).

Evidence: `cw_b54_independent_acceptance.json`, `cw_b55_root_adoption_receipt.json`, `plot1_window_gpu_joint_b55_b32_comparison.json`, `cw_e9_independent_acceptance.json`, `cw_e10_runtime_delivery_receipt.json`.

## Explicit flow-end completion option on a supplied CPU storm

C8 runs the complete actual-MAPLE CPU/Numba Plot1 storm with `deposit_mobile_at_flow_end=True`. It finishes at 4,739 s with 4,739 accepted steps, no rejected attempts and 802 commits. All 209 saved arrays (including the actual MAPLE resident snapshot and exchange history) are raw-exact against the earlier default-policy CPU storm. Water, class-sediment, request and physical reset accounting match exactly. This storm already has zero mobile load at hydraulic completion; the earlier constructed case, rather than this case, demonstrates nonzero terminal settling.

The comparison explicitly validates the added policy metadata and independently rehashes the two expected SYRUP source fingerprints before comparing every other MAPLE snapshot metadata value exactly. This is a cross-policy physical comparison, not a same-source/policy restart claim. The first strict metadata comparison failure is retained. Claude's independent review found no defect in the physical comparison and identified a scalar checkpoint coverage gap. Its bounded correction and root rerun now compare the entire checkpoint payload after exactly 14 asserted policy/source/file-fingerprint/timing differences; all 256 stored commit records and all remaining counters are exact. The full 802-commit count is compared separately; only 256 records are stored by the bounded log. Loop plus checkpoints: 127.41 s; setup: 7.50 s, observational under concurrent host activity. Evidence: `cw_c8_full_plot1_cpu_flowend_comparison.json` and `compare_cw_c8_flowend.py`.

## Fresh-process wet restart on the accepted integrated source

C9 (CPU, explicit flow-end policy True) and C10 (GPU, default policy False) both complete from their own uninterrupted storm's 2,400-second checkpoint. Root independently reruns the corrected C11 comparator: 209 CPU and 230 GPU saved arrays are raw-exact, complete MAPLE snapshot provenance/source/configuration metadata matches, and the entire handoff checkpoint payload matches after excluding only five named execution-time measurements. Both pairs finish at 4,739 s with 802 commits; all 256 stored commit records and the physical counters are exact.

The starting checkpoints contain 463 cells with positive surface depth and 2,778 cell-classes with positive mobile sediment. Snapshot depth/mobile arrays exactly match the encoded continuation state and are finite/nonnegative. All checkpoint file checksums verify. Thirteen injected checksum/nonfinite/negative/one-ULP/dtype controls fail as required on each comparison. Ten GPU diagnostic controls additionally enforce retained static context/configuration/graph identity and valid process-local counters.

The first GPU observer comparison failed because it treated process-local preparation logs/counts and end allocator statistics as common full-storm state. The correction explicitly validates and records only named invocation fields, retaining exact device fingerprint, root solver, pairing, final hydraulic context and graph fingerprint, memory preflight and hard cap. Physical checkpoint payloads and arrays have no tolerance exceptions. This qualifies each same-source/policy/backend restart separately, not cross-backend equality or large-case behavior. Evidence: `cw_c11_independent_acceptance.json`, the two `cw_c11_*_wet_restart_independent_comparison.json` receipts, and `compare_complete_windowed_restart_c11.py`.

## Latest follow-up — M44 lifetimes, B46 empty CPU transport and actual large restart

The genuine F2/M40 Chastre restart from 1 to 2 seconds reproduces all 174 saved
fields exactly, including complete snapshot provenance and source/configuration
identity. The loaded checkpoint contains 3,868,449,626 uncompressed bytes. Only
five explicitly named execution-time fields are excluded; source metadata is
not excluded. This interval has no surface flow or mobile sediment, so it
qualifies large CPU loading/continuation rather than wet or complete storms.

M43 full Plot1 CPU/GPU storms reproduce all 191/212 saved arrays and reset
accounting exactly. Its next capped Chastre GPU probe refuses the first pickup
at the owned residual's negative increment: a 107,250,176-byte allocation with
9,089,392,128 bytes retained against the unchanged 9,092,045,209-byte cap. No
step, output, checkpoint or deep version is accepted; initial/source identities
remain unchanged.

M44 releases private exchange temporaries at their last use and reuses a dead
private residual increment buffer. Existing arithmetic, caller protection,
validation, pool caps and physical laws are retained. Independent checks pass
174 CPU cases (9 device skips), 252 actual GPU cases, 22 combined whole-device
exchange/completion/wet-restart cases, and 18 strengthened device differentials
against immutable M43. The latter positively assert source paths, CuPy
residency and the owned ledger path, and compare caller alias/layout contracts.
A stale historical lifetime-name test is corrected explicitly; its original
failure remains archived. The residual-buffer change reduces one allocation
at the gross-residual line, but does not reduce that block's peak by itself.

B46 skips only CPU count/write/row-sum kernels on a validated empty phase with
zero pickup. Compilation availability, self-checks, Courant/refusal handling,
all later arithmetic and all validation still run. Its independent 86-case
CPU differential/refusal/restart suite passes. Combined M44/B46 full CPU Plot1
reproduces all 191 arrays and reset accounts exactly. Loop plus checkpoint
writes takes 113.98 seconds, setup 8.26 seconds, observational with shared host
activity. The matching full GPU storm completed with all 212 arrays/reset accounting raw-exact; genuine M44 initial reimport also completed with unchanged physical bytes. No large GPU fit, large wet/full-storm acceptance or controlled
speedup is claimed.

Evidence: `cw_f4_large_same_source_restart_comparison.json`,
`cw_f4_large_restart_acceptance.json`, `cw_b47_chastre_W3_F2_M43_capped_allocation_site.json`,
`cw_b44_independent_acceptance.json`, `cw_b46_independent_acceptance.json`, and
`plot1_window_cpu_lifetime_empty_b32_comparison.json` in the task archive.

## Latest follow-up — F2 restart bounds and M43 owned scratch

F2 fixes the confirmed large-checkpoint loader limit: the dense default stays
1 GiB; the windowed default is bounded by its explicit host cap (40 GiB),
and callers can tighten it. Resource limits are recorded separately from
scientific identity. Schema, backend, source and configuration checks remain
strict. Independent checks pass 50 CPU cases and 29 metadata negative
controls. The corrected author suite passes 83 checks, including the two
formerly failing summary comparisons; only the explicitly validated added
resource record is tolerated. Original failure evidence is retained.

M40 passes complete Plot1 CPU/GPU comparisons (191/212 saved arrays and reset
accounts exact). Its next Chastre GPU probe passes the old touch-counter site
but refuses the first pickup's owned pending fold: 107,250,176 bytes requested
with 9,075,985,408 bytes retained against the unchanged 9,092,045,209-byte pool
cap. No timestep is accepted. M43 reuses the existing fixed-order private fold
and typed boolean touch addition inside the owned helper only. Independent
checks pass 149 CPU cases (5 device skips) and 154 actual GPU cases, including
allocation negative controls, exact numerical results, alias/refusal and
caller-preservation checks. Combined F2/M43 GPU completion/wet-restart checks
pass four cases. Its full CPU Plot1 storm reproduces all 191 arrays and reset
accounts exactly; the full GPU comparison now reproduces all 212 arrays and reset accounts exactly. The M43 initial-only import is complete, with resident/baseline bytes unchanged. A hard-capped large GPU probe is active.
These are bounded slice acceptances, not large wet-storm acceptance.

The genuine F2/M40 Chastre CPU run reaches a checked pause at two seconds
(two accepted steps, no commits), with closed water, sediment-by-class and
request budgets. Surface flow, mobile sediment and export remain zero.
Its one-second checkpoint contains 3,868,449,626 uncompressed payload bytes,
exceeding the old 1 GiB limit. Fresh same-source continuation now passes all 174 saved fields exactly, including complete snapshot provenance and source/configuration identity.
The comparison requires exact source/configuration identity, all snapshot
provenance and physical arrays; only five named execution-time fields are
excluded. Preparation takes 477.89 seconds and the loop plus two checkpoint
writes takes 609.89 seconds; peak process RSS is 12,774,584,320 bytes. These
are observational costs including validation/I/O and lazy compilation.
The second timestep attempt alone takes 33.81 seconds, including 20.84 seconds
in its two MAPLE bed calls. This short infiltration-only probe does not
establish steady-state wet throughput or completed large-storm behavior.

The original MAPLE package reconstruction has now been independently checked
against local commit `74dd3e79d9a7df85244355439e3a4a762abb9273`: 322 of 324
package files agree, and exactly the two recorded pre-existing dirty files
differ. The exact dirty patch and reconstruction proof are preserved. This
qualifies the original dependency provenance; later water patches and final
complete-source artifact still require their own pins.

Latest evidence: `cw_f2_independent_acceptance.json`,
`cw_b43_independent_acceptance.json`,
`cw_b42_chastre_W3_F2_M40_capped_allocation_site.json`,
`cw_f3_large_checkpoint_size.json`,
`cw_f3_chastre_W3_F2_M40_cpu_pause2.json` and
`cw_e4_upstream_reconstruction_proof.json` under the task archive.

## Current follow-up — 2026-10-09, B29/B30 and dependency delivery

The M29 compact-water ledger candidate passes independent CPU checks
(752 passed, 32 device skips), actual GPU0 checks (381 passed), and 18
additional raw-exact CUDA fold probes. A complete matched Plot1 CPU storm
reproduces all 191 numerical arrays and reset accounting from M24 exactly.
The first comparison used different maximum-duration controls (10000 versus
5400 seconds); its future-boundary-array refusal is retained alongside the
successful matched-control comparison, rather than counted as agreement.

On Chastre, a 60-second CPU pause with M24 reproduces all 173 numerical
arrays from the preceding accepted source exactly. Parsed checkpoint
metadata also agrees after excluding provenance and the five already named
execution-time fields. This window contains infiltration but no surface
flow or sediment transport, so it does not qualify a wet or complete large
storm. The original comparator's Unicode-string-width refusal is retained;
the corrected comparator still requires raw equality for numerical arrays.

The hard-capped S27/M29 Chastre GPU probe still refuses its **first pickup**,
now at a compact Kahan output allocation. No timestep or new deep-store
generation is accepted. The configured pool cap is 9,092,045,209 bytes;
the allocation hook observes a peak of 8,934,679,552 live pooled bytes before
the refusal. This is not a global-device peak or successful fit. The older
planner estimate undercounts; the hard pool cap remains enforced.

B30 releases obsolete initial/restart references without mutating caller
objects. Its author reports small-case complete/paused/fresh-restart byte
identity with S27 and negative-control lifetime failures on the old source.
The full matched Plot1 CPU run on frozen S30/M29 is complete at 4739 seconds,
4739 accepted steps and 802 commits: all 191 numerical arrays and reset
accounting are raw-exact to S27/M29. Its 134.37-second loop timing is
observational. The full matched GPU run also completes at 4739 seconds and
802 commits; all 212 arrays and reset accounting reproduce S20/M24 GPU
raw bytes exactly. Its 741.82-second loop timing includes checkpoints and
is observational. Independent suites pass 165 CPU checks (51 honest device
skips) and 168 actual GPU0 checks with no skips, including complete wet
restart and failure/publication contracts. A requested
one-voxel Chastre experiment was correctly refused before assembly by
MAPLE's minimum-width rule; the next trial uses its supported three-voxel
window. No guard is relaxed, and scientific equivalence across window
widths remains an explicit qualification task.

The supported W3 Chastre CPU trial now passes a checked pause at one
second, with closed water, class-sediment and request budgets. There is no
surface flow or sediment transport yet. Initial preparation takes 983.29
seconds and the step plus checkpoint takes 288.43 seconds in this
observational run; these include substantial source/store validation and
checkpoint I/O, not just the hydraulic kernel. Small independent width
tests pass 17 CPU checks (2 device skips): W3 and W4 physical fields are
byte-identical to the same dense bed, including genuine buried refill,
an upward move and fresh-process wet continuation. Dense inventory-fold
accounting differs by at most 1.14e-13 kg, within the existing MAPLE bound;
W3/W4 accounting agrees exactly on CPU. The original strict GPU width
comparison fails in the initial per-cell inventory/reference-inventory
folds (and the global initial inventory in the crossing case). That
failure is preserved. A separate qualified comparison is being prepared
to measure those derived fields against existing MAPLE operand-qualified
summation bounds, while requiring exact physical bed, hydrology,
transport and exchange arrays. That separate comparison now passes four
actual GPU checks; 20 independent CPU checks confirm that altered physical
fields or derived values above the declared bounds are refused. Every
physical array is byte-exact. The three named inventory diagnostics differ
by at most 1.14e-13 kg and at most 4.56% of their two-side MAPLE bounds.
This is small-case qualified storage equivalence, not strict byte equality
of all diagnostics, large-case fit or MAHLERAN equivalence. Evidence:
`cw_b38_independent_qualified_acceptance.json` and its linked raw logs.

The supported W3/S30/M29 Chastre GPU probe still refuses before its first
accepted timestep, now at the final touch-counter output in the first
pickup ledger accumulation. The pool cap remains 9,092,045,209 bytes;
retained pool bytes are 8,986,610,688 and the hook's live peak is
8,903,037,952 bytes. Source, initial and observer identities remain
unchanged. A private compact-water correction is testing boolean-mask
integer addition to remove the redundant full-domain integer increment.
Twenty-four actual CUDA expression probes reproduce old counts exactly
across integer dtypes, layouts and overflow. The completed B40 candidate
then passes independent production-path checks: 143 CPU checks (10 device
skips) and 131 actual GPU0 checks. The device allocation negative control
confirms one integer-sized allocation in the new touch block versus two
in M29, with identical results and unchanged caller arrays. The only
runtime difference is this compact-water block. Frozen M40 and a reviewed
INITIAL-only W3 transition are complete. The full matched Plot1 CPU
and GPU storms finish at 4739 seconds and 802 commits: all 191 CPU and
212 GPU saved arrays plus reset accounts are raw-exact to S30/M29.
Observational loop/checkpoint times are 121.11 seconds CPU and 727.78
seconds GPU, including lazy startup work and concurrent host validation;
these are not controlled speed comparisons. The next hard-capped
large GPU probe and fresh same-source CPU checkpoint run are active.
Large wet/full fit remains unverified. No memory cap, physics or conservation tolerance
is changed. Evidence: `cw_b39_chastre_W3_S30_M29_capped_allocation_site.json`
and `cw_b40_independent_integer_add_gpu_probe.json`.

The dependency-delivery helper's initial manifest/version-path defect was
reproduced independently, then corrected in a separate private tree.
Independent tests pass 106 checks, including the old exploit as a negative
control, staged validation before publication and simulated short writes.
A local M29 full-source archive and read-only materialization have been
created; both existing MAPLE interpreters resolve the exact materialized
dependency. This proves source delivery, not storm or GPU qualification.
The recorded historical revision is qualified by preserved dirty-source
provenance and the exact net water-patch identity. Integrity checks do not
provide cryptographic authenticity.

Evidence: `cw_b29_full_plot1_cpu_matched_comparison.json`,
`cw_b28_chastre_cpu60_qualified_comparison.json`,
`cw_b33_chastre_M29_capped_allocation_site.json`,
`cw_b30_correction1_report.md`,
`cw_e2_correction1_independent_cpu.log`,
`cw_e3_M29_delivery_receipt.json`, and
`cw_e3_M29_runtime_resolvers_corrected.json` under
`agent_handoffs/tasks/conservative_water/`.

**Still open:** large wet/full GPU fit and performance, final production
integration and accumulated-diff review. Items 2–5 are not yet complete.
The native Fortran Chastre surplus remains documentation-only as directed.

The accumulated-code review additionally found a real large-checkpoint
restart defect: the runner leaves `load_checkpoint` at its 1 GiB default,
whereas the recorded Chastre pause contains 3,975,620,838 uncompressed
payload bytes. The root independently measured the ZIP directories without
loading those arrays. This safely refuses but makes large-case resume
unreachable. A private correction is adding an explicit recorded byte cap
based on the windowed runner's host resource limit; validation will remain
bounded. Small-case restart success does not cover this defect. Evidence:
`cw_f1_accumulated_review_report.md`,
`cw_f1_D1_independent_checkpoint_sizes.json`, and
`cw_f2_large_restart_prompt.md` in the task archive.

## Earlier reviewed-source checks — 2026-10-09

The frozen SYRUP B15 / MAPLE B17 pair completes Plot1 on both backends at
4739 s with 802 commits. Each backend reproduces all 191 saved numerical
arrays and explicit reset accounting byte-for-byte against its preceding
B14 run; original output hashes and both complete source manifests were
checked. This includes the actual GPU pool-limit wrapper and the reviewed
journal and initial-input binding corrections. It is small-case evidence,
not large-domain memory or controlled speed qualification.

The composed B22 MAPLE dependency adds two independently inspected memory
changes: shorter private availability scratch lifetimes and donation of a
fresh private intermediate ledger within the same transaction. The original
caller's ledger is never donated. Actual GPU 0 tests pass **332 checks**,
including window exchanges, failure/publication contracts, availability and
ordered-ledger arithmetic. Complete storms for this composition are pending.
The public dense/wind accumulation APIs are unchanged.

An allocation-site trace of the original hard-capped Chastre attempt locates
the first refusal in `_closed_availability` while allocating
`above = candidate - total`, before ledger accumulation. No step or deep
version was accepted. The availability lifetime change reduces measured
host scratch at that stage; its device effects still require the capped
large-case trial. The earlier uncapped 17.1 GB live-pool observation on an
11.8 GB GPU is **not** successful device-memory qualification. No uncapped
large-case retry is planned.

The exact zero-conveyance bisection branch now returns its algebraic root
for finite positive RHS. The earlier RFID 40-versus-50 iteration discussion
below records the previous source's limitation. Nonzero-conveyance solver
arithmetic, its accuracy requirement, and the 40-iteration default remain
unchanged. Terminal storage still has no spill or backwater physics.

Evidence is archived under `agent_handoffs/tasks/conservative_water`:
`cw_b17_full_plot1_{cpu,gpu}_comparison.json`,
`cw_b17_post_cpu_source_verification.json`,
`cw_b22_independent_actual_gpu.log`, and
`cw_b16_chastre_capped_allocation_site.json`. These private compositions
remain unadopted pending the outstanding whole-event/resource checks.

## Physical correspondence qualifications

The corrected, source-bound six-class constant-law checks cover distance
survival, deposition location, and finite-speed timing separately on CPU
and GPU. A further CPU check covers shorter/longer downstream distances,
slower/faster downstream speeds, and a cell-discretized distance gradient,
at 1 s and 0.5 s steps. All ten rows pass analytic piecewise-cell references;
maximum time-survival error is about 1.6e-15 kg. This does not test bin
merging, a coupled storm, or exact continuous-gradient behavior. The
source-assigned law values in that receipt are analytic contrasts, not
executed MAHLERAN results. Independent Claude review and GPU checks for
the varying-law helper remain pending.

Re-summing the existing actual no-splash fixed-terrain Fortran Plot1 ledger
gives 0.00922325697 kg exported sediment. The current evolving-bed
conservative experiment exports 0.01379526887 kg, **49.57% more**, including
when both are compared over the same first 4739 s. This is a material
physical/output difference, not native agreement. Supply limitation,
evolving composition, local hydraulics, local rather than source-assigned
distance laws, and event-end mobile completion differ together; the
comparison does not attribute the difference to one cause. The native
ledger also records a 0.29196856 kg effective clipping source and
0.30785825 kg final mobile load. Neither is copied into the conservative
model. No new Fortran run or investigation of the excluded Chastre
recession-water surplus was performed.

See `cw_d5_distance_velocity_{cpu,gpu}_qualified.json`,
`cw_d5_distance_velocity_cpu_nonaligned.json`,
`cw_d5_interval_exports_qualified.json`, `cw_d7_variable_laws_cpu.json`, and
`cw_d6_recorded_storm_comparison_qualified.json`. Report-interval means
support coarse sediment-pulse comparisons; 60 s reports do not establish
resolved instantaneous peaks.

## Complete Plot1 storms and restart

The source-bound, four-voxel-window, 32-bin sparse conservative Plot1 experiment
completes at 4739 s on CPU/Numba and GPU/CUDA, with 802 terrain commits and a
60 s quiet interval. Both water and grain-class sediment budgets close using
the existing policies. The explicit dry reset removes and accounts for
25.2200162282 m³ of soil water. This is a conservative evolving MAPLE-bed
experiment, distinct from the unlimited-supply legacy benchmark default.

Both fresh-process restarts from their respective 2400 s wet checkpoints
reproduce **all 191 saved numerical arrays and reset accounts exactly** in
their own backend. Output file hashes were independently verified. The
external immutable deep-store versions must remain available for restart.

The original strict CPU/GPU field comparison passes hydraulic/transport
fields but fails one near-zero resident voxel field. A separate reconstruction
outside the small-case production path passes the operand-qualified MAPLE
inventory comparison for all five physical reservoirs, using actual recorded
exchange counts, gross activity and reconciliation operands, capped by the
unchanged 1e-10 kg MAPLE mass resolution. Maximum voxel differences are about
4.3e-14 kg. The strict failure is retained; the qualified result does not
replace conservation or constitute independent scientific acceptance.

Observed loop times were 131.6 s CPU and 724.7 s GPU for the corrected source.
These include compilation/checkpoint costs and were not controlled timing
trials. GPU acceleration of the complete small model is not established.

## RFID evolving-bed case

The first full RFID trial refused a hydraulic step after its 1200 s
checkpoint because the default 40 bisection iterations could not satisfy the
existing 1e-11 m constitutive residual limit. A same-source wet-checkpoint
replay reproduces the failure without altering any solver output: at a
recognised zero-conveyance terminal cell, RHS depth is 10.9984820351 m and
the 40-iteration binary interval is approximately 1.0003061e-11 m. The
actual residual is 1.0004442e-11 m. A diagnostic 50-iteration sweep on the
same base operands reduces the maximum residual to 1.0658141e-14 m.

An explicit 50-iteration CPU trial completes at 4372 s, with 4687 accepted
steps, 327 rejections and 905 terrain commits. The default remains 40; no
accuracy requirement or conservation threshold was relaxed. Water residual
is -3.61e-13 m³. Class-4 sediment export is 0.0132359315 kg; maximum class
closure residual is 7.28e-12 kg. All completion and reset budgets close.

**Physical limitation:** this opt-in pit-storage experiment reaches
53.9747 m maximum pond depth in a 0.1 m cell. It retains 1.382018 m³
surface water in 32 recognised terminal cells, subsequently removed and
accounted for by the external dry-reset policy, plus 4.187804 m³ soil water.
Conservation does not validate this depth as a physical flood prediction.
The kinematic terminal-storage policy has no spill/overtopping or backwater
mechanism. Its use is explicitly bounded to this diagnostic scope; it cannot
establish unrestricted real-terrain hydraulic adequacy.

The matched full GPU trial also completes at 4372 s with the same accepted,
rejected and commit counts. Its water/class/reset accounts close. The
unchanged cross-backend field gate fails one pre-reset depth: CPU
0.022660912166100694 m versus GPU 0.02266091216603844 m, a 6.2256e-14 m
difference. All other hydraulic and transport fields pass their original
bounds. A separate small-case deep reconstruction passes all five
operand-qualified MAPLE inventory reservoirs and the original strict voxel
field check for RFID. Final routing receiver, level, donor, outlet,
pit and other integer/control fields match exactly; both reconstructed
graphs verify their own saved digest. Some floating conveyance coefficients
fail the original tight relative bound. These failures remain recorded and
require scientific review; no accuracy or conservation threshold is changed.

Subsequent D8 review and root's independent saved-data rerun qualify this **B9-era pair only**. The conveyance formula reproduces both final graphs exactly from their own terrain: 1–5-ULP differences in committed elevation propagate into a maximum coefficient difference of 1.204×10⁻¹³, with one failing cell. The one depth failure is 6.2256×10⁻¹⁴ m against the original 5.5322×10⁻¹⁴ m bound; its precise history is not attributed. The two strict failures remain failed. Closed budgets, exact routing controls and MAPLE operand-qualified inventories support a documented numerical qualification, without proving roundoff-only origin or changing thresholds. Source identities are equal except four explicitly verified backend paths (backend, storm implementation, CUDA mode presence, sediment-parameter namespace). Final-source reruns are still required. Evidence: `cw_d8_root_disposition.json`, `cw_d8_root_rfid_observation.json` and the corrected D8 report in `agent_handoffs/tasks/conservative_water/`.

Observed full loops were 408.4 s CPU and 843.7 s GPU including first-use
compilation/checkpoints under uncontrolled loads. These are not controlled
performance results or evidence of a small-case GPU speedup.

RFID prescribed vegetation is read from the bound original percent cover
map, which is zero on its active cells. XML `<veg>false>` is an output flag
and does not disable the prescribed cover physics. A nonzero synthetic-map
test verifies cover units, orientation and digest refusal.

## Chastre preparation and metadata profiling

The actual 44 compiled MAPLE row tiles were assembled into an immutable
initial deep store and four-voxel resident bed: 1393×1604 cells, 306 levels
and six classes at native 1 m geometry. The complete dense deep-bed shape
represents roughly 32.8 GB and is not allocated on the GPU. Initial
preparation was followed by an adapter refusal for missing cover metadata;
the persisted initial bed is retained and reused, rather than rebuilt.
The corrected cover adapter reads the bound RFID input and requires its
uniform active cover for the current Chastre mapping policy.

A private Numba host-metadata candidate preserves the original sequential
inventory/room folds without reassociation. Tests include ragged prefixes,
noncontiguous data, multiple class counts, invalid ranges and NaN propagation.
Twelve digest-verified actual initial Chastre chunks match every metadata
output byte. Alternating warm microbenchmarks show about 2.25–3.91× faster
complete metadata calculation for occupied prefixes, and no material gain
for zero prefixes; the first call costs 1.25 s compilation. This is a
bounded host microbenchmark, not measured whole-model or GPU speed.

Full larger-domain storms, measured peak memory, outlet/sensitivity
qualification, controlled performance measurements and actual Claude
independent review remain necessary before adoption and completion.
The separate native Fortran recession-water surplus remains documentation
only, per user instruction.

## Accepted-step memory lifetimes (provisional)

An actual-window ownership test reproduces unnecessary retention of the
previous step's pickup voxel window, pickup ledger, transport deposition
and physics demand into the next step's allocations. The private loop now
releases finished attempt/commit containers after accounting and callbacks,
while keeping the accepted state and persistent diagnostics. CPU/GPU
ownership, completion, wet/dry restart and late-refusal checks pass (six CPU
passes with three honest device skips; nine actual-device matrix passes).
The full CPU Plot1 replay preserves all 191 saved numerical arrays and reset
accounts exactly on both CPU and GPU. Large-domain peak-memory
improvement has not yet been measured, and this change awaits Claude review.

Local raw evidence: `agent_handoffs/tasks/conservative_water/` and
`outputs/conservative_water/` (ignored execution archives).

## Sparse scratch lifetimes and large dry trial (provisional)

The second candidate releases completed sparse count-pass buffers, compact
gathers and compact output accounts after their last use. The actual GPU
sparse/dense-oracle and coupled/restart matrix passes all 169 tests. Its full
CPU Plot1 replay preserves all 191 saved numerical arrays and reset accounts
exactly against the first lifetime candidate, with independently checked pins.
Both changes remain subject to actual Claude independent review.

The original resident-window Chastre CPU trial reaches a checked pause at
60 s: 60 accepted steps, zero rejections, one commit, and closed water/class
and request accounts. Rain input is 2642.123222991496 m³; surface water, runoff
export and mobile sediment are all zero. This tests infiltration and dry
coupling, **not large wet sediment transport or complete-event handling**.
Observed setup is 533.5 s and loop plus checkpoint is 2029.3 s under uncontrolled
load. The sparse-lifetime CPU trial is running for a same-backend comparison.

A GPU preparation with an eight-million-entry phase cap refuses before array
placement or stepping: its declared envelope exceeds 85% of the actual
10.697 GB free device memory. This is a correct preflight refusal, not an OOM.
A four-million-entry cap gives an 8,918,989,120-byte planning estimate and
passes the same guard. A fresh one-step profiler is running with that cap;
physics, bins and tolerances are unchanged. Capacity exhaustion remains a
refusal and never truncates material. The estimate is not a measured peak.
Neither profiler nor this dry pause constitutes full Chastre qualification.

## Large resource measurement — updated result

The four-million-entry GPU trial reached a checked 1 s pause with its
own budgets closed, but recorded **17,078,447,104 bytes** peak live pooled
allocations, and 17,451,018,240 bytes retained by the pool. The actual device
total is 11,811,028,992 bytes. The earlier 8.919 GB planning estimate therefore
undercounts the pipeline; this trial does **not** demonstrate a resident fit
or qualify full large GPU storms. These hook measurements exclude contexts,
non-pool allocators and kernel-local memory.

A separate phase profiler enforces an actual 9,092,045,209-byte CuPy pool
ceiling (85% of initial free memory). It refuses during the first MAPLE
pickup, asking for another 107,250,176-byte array with 8,995,555,328 bytes
already retained. No step is accepted and no candidate is published.
Physics takes about0.284 s under instrumentation, using6.037 GB afterward
and peaking8.398 GB. Failed-scope hook live counters retain their value at
the interruption; the authoritative post-refusal pool-used value is zero.
The device is unallocated after process exit. A full Chastre GPU storm is
withheld until the measured pipeline fits; reducing a capacity estimate alone
does not satisfy this requirement.

The sparse-lifetime CPU60 s replay matches all173 numerical arrays, physical
snapshot metadata, budgets and physical checkpoint payload exactly. Five
CPU/wall execution-time measurements differ, as expected under different
loads; the literal-payload failure and its exact differences are retained.
Simulation time and all model controls still match. Measured maximum RSS is
17,666,836 KiB, while the older implementation had an observed sample above
22 GiB. These are observational resource results, not controlled speedup.

## Compiled compact ledger (CW-B14, provisional)

The actual instrumented first Chastre CPU step spends29.22 s in four ledger
transfer calls. A private candidate retains the original dense-ledger
implementation and four ordered compensated-sum operations. Matching
contiguous FP64 compact arrays use optional Numba without fastmath, or one
CUDA elementwise kernel with explicit round-to-nearest add/subtract
intrinsics. Other dtype/layout/broadcast cases use the original function.
For one stored process, newly computed outputs provide the full new channel
without a redundant copy-then-overwrite. Neither sums nor compensation in
the input are mutated.

Checks:43 CPU passes/22 device skips,286 surrounding CPU passes/2 device
skips,102 actual-device ledger/window passes, and9 actual CPU/GPU coupled
window completion/restart/lifetime passes. A GPU ownership test originally
failed because CuPy14.2 reports sharing for distinct0-D allocations, also
with ordinary addition. Exact device byte ranges confirm disjoint storage;
the test uses those ranges for contiguous GPU arrays. The original failure
and pointer probe are retained. Runtime code was unchanged by that correction.

The complete CPU Plot1 replay reproduces all191 numerical arrays and reset
accounts exactly. In a separate alternating-order synthetic ledger benchmark
on Chastre dimensions, warm median is1.612 s for the original primitive
and1.022 s for the candidate (1.58×); all output bytes and input preservation
match. This is a primitive microbenchmark, not actual Chastre storm or Fortran
speed. Whole-event GPU/restart, large memory reduction and independent
Claude review remain required.

## Timestep sensitivity (CPU,32 bins)

| Maximum dt(s) | Complete time(s) | Runoff(m³) | Sediment export(kg) | Gross pickup(kg) |
| --- | --- | --- | --- | --- |
|1|4739|0.1630526221|0.01379526887|411.0340958|
|0.5|4741.5|0.1625652335|0.01376343230|411.0868304|
|0.25|4741.75|0.1625724537|0.01374657255|411.1137731|

All water/class/request and dry-reset budgets close; output pins verify.
Integrated sediment export changes-0.231% then-0.122% under refinement;
gross pickup changes+0.0128% then+0.00655%. Terrain commit counts change
with the time partition. Last-accepted-step export rates sampled every60 s
show much greater peak variation; these are instantaneous sampled values,
not interval averages or well-resolved storm peaks. No convergence threshold
is fitted. This does not replace the deferred bin/time-distribution study.

## CW-B14 review and CW-D5 physical diagnostics — 2026-10-09

All four actual Opus reviews completed after Fable quota refusals (the latter supplied no new reset time). B14 compact Kahan review found no confirmed runtime defect. Full CPU and GPU Plot1 runs each preserve all191 numeric arrays and reset accounts byte-for-byte against their respective pre-Kahan baselines. The GPU restored the new CPU-produced initial store; hashes and result comparisons bind that provenance. This is per-backend identity, not a claim that every strict CPU/GPU field gate passes.

B7 review identified a pending-journal callback-mutation guard gap. B8 identified the underestimated GPU preflight/no enforced pool cap, hard-coded six classes in planning, reused initial operands missing from run identity, and late XML hash checks. D3 identified insufficient fixed40 bisections for depths above about11m and proposed exact zero-conveyance terminal roots. Two bounded Claude correction writers are addressing these in disjoint B15 private trees. Their changes are not yet accepted or production-adopted.

The controlled six-class chain (`cw_d5_distance_velocity.py`) uses six independent known mean distances and velocities, not calibrated grain laws. On actual Numba CPU at1s and0.5s and CUDA GPU0 at1s, face survival agrees with exp(-distance/L), deposition location agrees with differences of consecutive survivals, and first export arrives exactly at384,192,96,48,24,12s, respectively. Largest per-class closure discrepancy is3.11e-15kg. Existing equation-test bounds are unchanged. This separately checks distance attenuation and sediment speed; it is not a changing-hydraulics/evolving-bed or actual MAHLERAN storm validation.

Saved, hash-verified cumulative exports resolve60s interval means, which are preferable to the earlier sparsely sampled last-step rates when comparing these timesteps. For dt1,0.5,0.25s the peak full60s mean sediment exports occur in1140–1200s and change by+0.9837%, then+0.0985%. Quantile times are reporting interpolations only; no resolved instantaneous-peak convergence is claimed. Final partial intervals are excluded from peak selection. `cw_d5_interval_exports.json` records exact values and all output pins.

RFID margins (`cw_d5_correspondence_margins.json`) retain the strict depth/conveyance failures. Largest voxel difference/bound is0.04213; active/available maximum is0.01610. Largest difference/holding is2.624e-13 for voxels and7.170e-12 for active/available. Bound/mobile are exact. Saved report-time accepted/rejected steps, commit counts, graph-change counts, max transport substeps, and per-cell/class physical touch counts are byte-identical. Regime-cell-step counts were not stored by these runs and are not inferred. The per-voxel broadcast of a cell/class MAPLE reconciliation bound remains a heuristic extension; policy-scale correspondence does not prove roundoff-only origin or accuracy below the mass resolution.

The actual large GPU memory blocker remains:17.1GB live pooled peak on11.8GBdevice, and85% hard cap refusal during first pickup. No full large wet GPU qualification or whole-water acceptance follows from the diagnostics above.

## Current qualified sources and large-domain gate — 2026-10-09

These updates supersede pending status in earlier dated entries. They are private-candidate qualification, not production adoption or acceptance of the whole water component.

The combined immutable SYRUP S20 / MAPLE M24 completes both Plot1 storms at 4739 s and 802 terrain commits. All 191 saved numerical arrays and reset accounts are byte-identical to the previous same-backend S15/M17 storms. Independent MAPLE window tests pass 305 CPU tests (32 device skips) and 337 actual GPU tests; accounting/lifetime/restart checks pass 25 CPU tests (21 device skips) and 46 actual GPU tests. Runtime and checkpoint timings are observational and include startup; no controlled acceleration claim follows.

S27 adds the reviewed initial-only source-transition importer and final source guard to S20 without changing storm physics. The initial-only suite passes 38 independent CPU checks. Chastre's genuine M9-produced generation-zero deep store has been validated under M24, including streamed physical/prefix checks. Newly published initial metadata retain exactly the original resident and baseline file hashes. Original initial files and deep generation zero remain unchanged. The transition admits only dry time-zero initial states and the individually reviewed water-source changes; ordinary restoration remains strict and wet checkpoints cannot use this import path.

The capped Chastre GPU profile on S27/M24 refuses the first pickup's first ledger accumulation at the ordered pending-increment fold. Availability processing now passes. Its hard pool limit is 9,092,045,209 bytes; the same-thread hook reaches 8,921,271,296 live bytes before a refused 107,250,176-byte allocation (pool retained 9,084,930,048 bytes). No timestep is accepted and no result checkpoint is published. All source, observer and reused initial identities remain unchanged. The hook is not a global device-memory high-water measurement. This is a diagnosed resource refusal, not a fitted large wet storm or completion. A bounded private ledger/lifetime correction is underway.

The D7 non-aligned six-class diagnostics cover downstream distance and speed changes, including positive remaining time within a substep. Ten CPU cases (1 s/0.5 s) and five GPU cases (1 s) pass the unchanged analytic survival, arrival and class-closure gates. Independent Claude review found no defect. An additional explicit CPU/GPU comparison gives exact exports and arrival steps at 1 s. These single-pulse fixtures do not exercise bin merging, settling, convergent flow or evolving-bed storm physics. Earlier aligned receipts retain their original script identity.

The recorded native Plot1 comparison remains materially different: native Fortran export is 0.009223256975 kg; the conservative evolving-bed experiment exports 0.013795268872 kg, about 49.57% higher, including when restricted to the same 4739 s window. The native clipping source, fixed composition, source-assigned transport law, supply limitation and event completion differ. This is documented disagreement, not physical equivalence or a uniquely attributed error. No native Fortran rerun or surplus investigation was performed for this task.

Large wet/full CPU/GPU storms, measured resource/performance qualification, reproducible MAPLE dependency delivery and production adoption remain open. Legacy replay remains the default workflow; conservative window/CSR cases are explicit experiments.

## CW-B29 ledger follow-up and large CPU checked pause — 2026-10-09

A further private MAPLE change keeps the compact-water pending fold in one newly allocated buffer, preserving every ordered addition, including zero operands. Dense and non-water folds remain unchanged. Two private intermediate mobile arrays are released after their last read. Independent checks pass 752 CPU tests (32 device skips), 381 actual GPU tests and 18 additional GPU fold byte comparisons covering signed zeros, double subnormals, finite extremes and C/F/sliced operands. The source is frozen as M29. A full matched-control CPU Plot1 run preserves all 191 saved numerical arrays and reset accounts exactly against M24. An initial comparison used max_end_s=10000 instead of the reference5400; its differing future boundary schedule and comparison refusal are retained separately. The matched run uses5400.

The 60 s Chastre CPU checked-pause run on S27/M24 completes60 accepted steps and1 commit. All173 numerical arrays, physical snapshot metadata and continuation payload match the preceding S12/M9-derivative run exactly. The literal continuation payload differs in exactly the five separately recorded CPU/wall timing fields; those are excluded only from physical equality. The original comparator refused different NumPy Unicode widths for the metadata JSON before parsing it. Its corrected diagnostic variant permits different text storage widths while requiring exact non-provenance metadata JSON; all numerical dtype/shape/byte checks remain unchanged. Both original refusals and corrected reports are retained.

Water and sediment budgets close, but surface water, mobile sediment and exports remain zero: this is an infiltration-only interval, not wet/full-storm qualification. Observed loop plus checkpoint time is1677.96 s and setup421.16 s, against2134.90 s and538.27 s previously. Concurrent jobs, cache and compilation prevent interpreting this as a controlled speedup. The new M29 initial-only Chastre import retains exactly the original physical resident/baseline hashes and immutable deep generation zero. A new85%-capped GPU allocation probe is active; fit remains unproven.

Evidence: cw_b29_independent_review.json, cw_b29_gpu_fold_probe.json, cw_b29_full_plot1_cpu_matched_comparison.json, cw_b28_chastre_cpu60_qualified_comparison.json and cw_b33_launch.json under the local task archive. These are conservative experimental runs; the legacy replay default is unchanged.


## Reviewed late boundary-default candidate — M67

The isolated M67 change allocates absent default boundary arrays at their original first numeric use. Supplied terms retain their original validation timing; physical expressions, summation order and zero-exchange behavior are unchanged. The actual GPU matrix passes 184 checks, plus two combined B65/M67 complete-event and wet-restart checks. Memory release is measured exactly on the bounded fixtures: erosion saves one class grid per absent term, deposition saves a grid only when export is absent, and return storage is unchanged.

The reviewed INITIAL-only Chastre and Plot1 imports preserve their resident/baseline file hashes and original immutable generation-zero deep-store reference. The new full CPU Plot1 storm completes at 4739 seconds with 4739 accepted steps, no rejection and 802 commits. Budgets/reset and the four source-independent published files match B55/M44 exactly; full metadata/source-qualified comparison is pending. A new original-case W3/six-class/32-bin Chastre GPU allocation probe is running under the same 85% pool limit. No large GPU fit, root adoption or complete component acceptance is claimed.


## M67 large probe result and next bounded work

B68 completed with a hard-cap resource refusal inside `maple.core.backend.scatter._segment_totals`: the stable gather's extra copy requested 27,082,752 bytes with 9,073,191,936 bytes retained against the unchanged 9,092,045,209-byte pool limit. Pickup and transport-entry live bytes match B66; a changed temporary-allocation order can change pool layout near the cap. This is neither a physical failure nor proof that M67 worsens live-memory requirements. No step, checkpoint or output was published; all source, observer and initial identities are unchanged. The B69 candidate investigates removing only the redundant copy of the already-owned advanced-index gather. B70 is a separate read-only design assessment of a post-pickup mobile pool that CSR does not read; it is not implemented.

The corrected C13 comparator independently verifies the complete CPU B55/M44 → B65/M67 regression: 209 raw-exact numeric arrays, all physical summary fields and complete metadata except individually rehashed source/initial operands and named execution records. Its original 53 and added 29 negative controls refuse as required, as do 23 pure GPU-diagnostic fixture controls. The latter are not a device run. The actual full B65/M67 GPU storm is running under the same default completion policy as the accepted B55 GPU reference.


## Full GPU source regression and scatter-copy finding

The actual B65/M67 Plot1 GPU storm completes at 4739 s, with 4739 accepted steps, no rejection and 802 terrain commits. The corrected joint-source comparator verifies **all 230 saved numerical arrays** raw-exact against the accepted B55/M44 GPU storm, along with complete physical summary, reset and source-qualified checkpoint metadata. The default completion policy is identical in this pair. This is a full small-GPU regression, not a large fit proof or a same-source restart pair. Observed setup/loop-plus-checkpoints times are 9.89/745.36 s with concurrent CPU work; no controlled speed comparison is claimed.

The isolated B69 scatter candidate independently passes 1244 CPU checks (71 device prerequisites skipped) and a 1448-check device-enabled matrix without skips, including 71 explicitly CuPy checks. It removes exactly one gather-copy allocation, but actual device measurements confirm **unchanged whole-call peak and return storage**. It is retained as an isolated experiment, not adopted to fix the large-case limit. The primary dependency remains M67 for the next B71 SYRUP lifetime experiment: release the unused post-pickup water pool from accounting characteristic steps while retaining all genuine MAPLE validation/accounting. B71 is under implementation; root remains B55/M44.

Current 2026-10-10T01:05:20.328132+00:00: Frozen M85 (17044c8a) independently passes 72 CPU plus 22 actual-device focused cases, three small complete/wet-restart CPU raw differentials, and 77 integrated CPU/GPU ownership/completion/restart cases. The measured fraction-site scratch reduction is two class arrays; this is not a large-fit or global-peak claim. Previous M82 F11 failed on step 2 at the original pool cap; its refusal is preserved. Genuine Chastre CPU/GPU INITIAL-only validation under M85 is in progress in separate stores. All large fit/wet/full/strict cross-backend/adoption gates remain open. Root runtime remains B55/M44 and legacy replay remains default. No new commit/push.
