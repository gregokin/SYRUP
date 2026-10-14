# Net-elevation-only hydraulic routing

The independently verified candidate is `outputs/conservative_water/cw_b104c2_frozen`, SYRUP package digest `8acabd96c2163cb165adc48ae3f788a28a1e17d9d865b015c3396091b11d40dd`, using the unchanged MAPLE M94 dependency. It has not been adopted into the root production checkout. Rotated Chastre now passes a bounded GPU 300 s wet run and exact own wet restart; full-storm fit and performance remain unqualified. Evidence is in `agent_handoffs/tasks/conservative_water/cw_b104c2_independent_acceptance.json` and `cw_r8n_root_gpu300_acceptance.json` in the same directory.

## What changes

Actual MAPLE pickup, deposition, sediment inventory, composition, and water accounting continue at every accepted physical step. Ordinary MAPLE commit conditions remain in place. Hydraulic routing is now a separate decision: it refreshes only when the largest absolute **net elevation change in an active cell**, measured from the surface used for the last hydraulic routing build, reaches the actual MAPLE `pending_elevation_trigger_m` (0.00075 m by default; comparison `>=`).

Erosion and deposition within a cell can cancel. Changes in different cells cannot cancel each other. The reference survives accounting commits and ledger resets, so several small changes can eventually trigger a refresh. A threshold crossing between ordinary accounting commits forces a real MAPLE commit before building routing from its authoritative geometry.

Time, gross sediment turnover, reporting, checkpointing, deep-storage publication, and a final forced accounting commit do not independently reroute. On an accounting-only commit, the existing graph, transport network, physics grid, discharge, phase binding and CUDA graph context remain in use. Depth and discharge still evolve through the hydraulic solver at physical timesteps.

MAPLE's existing exact windowed height calculation supplies the current surface, including its commit datum arithmetic. It uses height packets, the resident window and the active layer; the windowed hot path does not scan entire voxel columns. The held reference is one surface array, not a second sediment inventory.

## Restart and completion

A checkpoint saves the surface from which routing was built, which can differ from the latest committed bed. Loading and CPU/GPU conversion reconstruct routing from that saved surface and preserve its binding and discharge. Missing or inconsistent source geometry is refused. Saving checks the graph, routing source and terrain reference together before publishing a bundle.

Terminal mobile deposition can trigger a refresh if its net height change reaches the threshold. The fact that the final commit is forced does not do so. The documented dry-again water reset and sediment accounting remain in place. Wind handoff remains outside this milestone.

## Qualification

Independent checks against the final frozen source passed:

- CPU: 33 tests; 42 GPU-only cases explicitly skipped.
- GPU0: 74 tests; one intentional NumPy-only CUDA-preparation case skipped.
- The original checkpoint-save and namespace-conversion defect reproducers now refuse inconsistent input without changing caller arrays.
- Actual Numba/CUDA windowed CSR completion and own wet restart preserve the held routing surface and carried deep journals. Same-backend continuation is exact; CPU/GPU tests retain the existing declared bounds.
- Failure injections at a genuine net-driven reroute check that a failed graph build or CUDA preparation does not accept the candidate or alter previously published state.

These are bounded fixtures. Historical source/cadence and strict benchmark failures remain recorded; this is not a blanket historical-suite pass. Held hydraulic geometry is an intentional coupling approximation, and later flow, sediment movement and commit times can differ from the previous schedule. Threshold sensitivity and representative large-case behavior still require measurement.

## Storage and next experiment

Hydraulic refresh, sediment accounting and file publication are distinct. The earlier optional deferred deep-storage publication remains available; it does not defer physical sediment exchange. It uses actual MAPLE pending journals and retains conservative flush/retry behavior.

The prepared large-case choice is boundary publication with a 3,000,000-record high-water trigger under the unchanged 4,000,000-record journal cap. This is a post-step trigger, not a hard peak or a guarantee that the next attempt fits. Original partial stores are retained.

The first large experiment uses rotated Chastre through 300 seconds, an own wet restart from 240 seconds, and a matched CPU reference. Its source-aware tools passed independent review before launch. Full-storm continuation depends on memory, storage and restart evidence. The legacy replay remains the default benchmark workflow.

## Observed rotated Chastre reference — 2026-10-14 UTC

The actual GPU0 reference exited successfully at the 300 s pause with 300 accepted steps, no rejected attempts, 24 MAPLE accounting commits and zero hydraulic reroutes. The largest recorded net displacement was 0.21335 mm, below the 0.75 mm trigger. Water and class sediment budgets report closure. This is a wet partial storm; approximately 90,738 kg of sediment remains mobile. The fresh-process restart from its own 240 s checkpoint passes all 184 numeric members and parsed physical metadata exactly. Both source trees, original INITIAL files, generation-zero manifest and all 5,346 unique original chunks were verified unchanged. The matched CPU reference has also exited successfully at 300 s with the same step, commit, publication and rerouting counts and closed budgets. The completed [CPU/GPU comparison](rotated_backend_comparison.md) retains four strict failed records at 240 s and seven at 300 s, confined to residual, compensation and near-zero unmet accounting. Recorded physical inventory, hydraulic and transport fields and exact control decisions pass their original checks; each conservation budget closes independently. The original checker refusals and all strict failures remain archived. This is qualified agreement, not a blanket backend PASS. Actual net-driven reroutes were exercised in bounded fixtures, not in this large slice, which stayed below the threshold.

The original 9,092,045,209-byte CuPy pool limit remained enforced. Retained pool allocation after the loop was 9,015,803,392 bytes; this is not a measured peak or a guarantee of full-storm fit. Reported host high-water RSS was 4.69 GB.

Observational times were 313 s for setup and 2,038 s for the loop including checkpoints. The two checkpoint writes took 216 s and 233 s. These times include validation and storage work and are not isolated hydraulic-kernel timings. The reference added 55.25 GB of chunk files, bringing retained chunk bytes to 169.00 GB. MAPLE counters recorded 324.93 GB of logical reads; that is not a measurement of physical disk traffic. The historical store and file deduplication qualify timing and growth comparisons.

The CPU reference took 450 s for setup and 6,800 s for the loop including checkpoints (230 s and 226 s for checkpoint writes). It added 90.66 GB of chunks and reported 9.68 GB host high-water RSS. Its reused store had fewer pre-existing matching chunks than the GPU store. These are actual end-to-end observations with different write and cache histories, not a controlled backend speedup measurement. See `agent_handoffs/tasks/conservative_water/cw_r8n_root_cpu300_observation.json`.

Exact cumulative sparse overlays on raw chunks now pass bounded independent MAPLE CPU/GPU store and publisher tests; [SYRUP integration remains separate](exact_overlay_storage.md). They preserve final replacement values, including one-ULP changes, and keep the default full-chunk path. A small fixture saved chunk bytes with slightly slower publication; no large-case saving or adoption is established. The GPU reference also reused 9,477 of 24,171 sealed chunks from its older history (manifest audit), qualifying comparisons with the CPU store. Reference evidence: `agent_handoffs/tasks/conservative_water/cw_r8n_root_reference300_observation.json` and `cw_r8n_root_gpu_dedup_metadata_audit.json` in the same directory.
