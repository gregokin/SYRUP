# Evolving-bed Chastre: staged full-storm extension

Current outcome, October 16, 2026: the later isolated-source, 18.9M-entry trial stopped incomplete at its 10,000 s horizon, with water and sediment retained. Its last independently accepted whole stage ends at 9,600 s; the last regular saved state is 9,840 s. Seven hydraulic reroutes occurred. No model is currently running. Production code and dependency pins have not adopted the trial changes. See [current status](current_status.md) and [flat/pond limitations](flat_pond_followup.md); the entries below are historical observations, not active launch instructions.

Historical update, 2026-10-16 UTC: the isolated memory candidate has passed independently verified fresh 0→300, own 300→600 and own 600→720 GPU stages, with exact saved physical results versus the accepted 8M reference, closed balances and unchanged source/INITIAL/old storage. Stage wall times were 1857.775, 1440.382 and 1054.727 s. Matched 600→720 live pool peak is 7,129,789,952 B versus 7,394,395,648 B (3.58% lower); retained peak is 9,078,823,424 B versus 9,026,217,984 B (0.58% higher). Production pins have not adopted this candidate. A separately scoped, explicitly documented count-only 8M→16M experiment uses the same candidate source with unchanged hard memory limits and physics. Its first fresh 0→300-second stage launched on idle GPU1 at 2026-10-16 03:14:55 UTC after independent tool verification; no 16M result is accepted yet. Full-storm fit and a genuine large-case reroute remain unproven. See [current status](current_status.md) and [candidate evidence](deposition_memory_candidate.md).

The dated entries below preserve the earlier progression and refusals.

Status, 2026-10-14: the continuation from the accepted 300-second wet checkpoint to **600 seconds passed** independent checks. A full storm remains unproven. This extends the evolving-bed experiment; it does not change the default legacy benchmark workflow.

## Capacity refusal in the next stage

The 600-to1200-second attempt stopped at 20:11:55 UTC on October 14; the supervisor closed at 20:12:07 UTC. Sparse transport required **4,004,041 nonempty entries**, exceeding its configured **4,000,000** cap. It refused the step before write-array allocation, without discarding sediment. No new checkpoint or completion summary was produced; the verified 600-second wet checkpoint remains the last restart point. The precise failed physical time was not persisted and is not inferred.

The run exited 1 and was reaped, the supervisor stopped as designed, and GPU0 was released. Host high-water RSS was 5,324,873,728 bytes. Existing case pins and the entire before/after store metadata manifest are unchanged. Most downstream failed gate fields follow the absence of a completed-stage summary; they do not establish multiple conservation defects. See `root_s1200_failure_acceptance.json` and the original failed receipt. Full-storm fit remains unproven.

A separately labelled larger-phase-capacity candidate is being assessed with unchanged 32-bin physics and hardware memory limits. Capacity is part of restart identity, so an accepted change requires replay from the same INITIAL; existing wet checkpoint identities must not be edited or bypassed. No larger-capacity GPU run has yet started.

## Configuration and state continuity

Use committed SYRUP `9bfe80a` (package `5f85a989…`, actual ROOT imports) and the pinned MAPLE `24c6e147…` package from the `b8fa43a3…` cache. Geometry, forcing and physical options match the accepted reference: rotated Chastre, 886×1780 rectangular cells / 1,140,189 active 1 m cells, six classes, W3 MAPLE resident voxel window, conservative 32-bin characteristic CSR transport, CUDA bisection hydrology and 1 s maximum physical timestep. Rain is 138.1067946553 mm/h through 2641 s; the forcing schedule ends at 2700 s. Terminal-water storage and flow-end mobile deposition remain explicit assumptions. Splash, growth, full ET and wind handoff are excluded.

Resume the reference's OWN 300-second checkpoint and immutable INITIAL. Append new content-addressed files to the same store while retaining every old version. Use NEW output/checkpoint/receipt paths for each stage. No dry reset occurs at an intermediate pause. Physical water/sediment state and the held routing source continue across restart; pause/publication boundaries mean this is a segmented storm, not a newly executed uninterrupted reference.

Retain the original limits: GPU0 only; CuPy default pool 9,092,045,209 bytes, host 40 GiB, store 256 GiB, phase/journal 4 million records, publication high water 3 million. Checkpoint cadence is 240 s, reports 60 s. Net elevation alone triggers routing at 0.00075 m (`>=`); accounting commits, pauses and checkpoints do not trigger routing by themselves.

## Observed first extension

| Quantity | Result at 600 s |
|---|---:|
| Accepted / rejected steps, cumulative | 600 / 0 |
| Bed accounting commits / hydraulic reroutes, cumulative | 73 / 0 |
| Maximum net elevation displacement | 0.000360512614465 m |
| Sparse phase entries / cap | 3,395,369 / 4,000,000 |
| Largest carried publication journal / cap, this invocation | 1,386,552 / 4,000,000 |
| New deep data, this invocation | 326,060,952 bytes in 639 files |
| New published deep versions | 5 |
| Host process high-water RSS | 5,360,799,744 bytes |
| CuPy pool retained after synchronized loop | 8,643,649,536 bytes |
| Setup / loop including checkpoints | 789.056 / 841.711 s |
| Driver wall time | 1644.839 s |

The pool reading is after the loop, not a measured peak. Device-wide sampled allocation also includes memory outside the CuPy pool. Timing is observational, including lazy compilation; the preceding 0-to300-second run used fresh assembly and a different storm interval, so these observations do not establish an optimization speedup or full-storm fit.

Water residual is 3.6379788071e-12 m³ within the unchanged MAPLE-derived 0.0888229913 m³ bound. All six sediment residuals are exactly zero against the unchanged 0.0272177208 kg bound, and pickup requests reconcile. Final mobile sediment is 236,981.392871 kg; it is retained in the wet checkpoint, not discarded. The pause performs no water reset.

Own 480- and 600-second checkpoints were saved. The 480-second checkpoint records 480 accepted / zero rejected steps, 47 commits, no reroutes and 2,432,365 phase entries. At 600 seconds, checkpoint source hashes, physical clock, accounting-call count and physical/logical deep-manifest bindings match the completion summary. The process exited zero and its launcher reaped it; GPU0 returned to zero allocated memory. Sources and INITIAL identities remain unchanged. Existing large store files are checked by deterministic size/mtime/mode manifests, **metadata evidence rather than rereading every historical byte**. The next public continuation loader verifies the saved payloads and state contract.

## Next stage and gates

Advance to 1200 seconds, then bounded stages through rain-off and recession, stopping on genuine completion, a failed balance/state check, a resource refusal or interruption. The phase count is already 85% of its configured cap; do not silently raise that cap, change bins, or infer capacity from partial-run averages. No reroute has occurred yet in this large run, so large-case terrain-triggered graph rebuilding remains unqualified.

The first continuation is complete. After bounded Claude corrections, independent verification passed 95 supervisor tests and the gate over the actual 600-second artifacts. The host supervisor was launched at 19:57:06 UTC and launched the next GPU continuation toward 1200 seconds. That continuation subsequently stopped at the capacity refusal described above; the supervisor is no longer running. It checks each stage before launching another, preserves the original limits, and stops on completion, a failed gate, an unknown/interrupted process, a resource refusal, a stop request or the bounded wall guards (24 h total / 4 h per stage, with bounded cleanup). It does not retry or raise capacities automatically. Exact live status and subsequent receipts are under `agent_handoffs/tasks/chastre_full_storm_20261014/cfs_supervisor.status.json` and `cfs_runs/`; this is an observed launch, not proof the full storm will finish.

Independent first-stage receipt: `root_s0600_acceptance.json`; driver receipt: `cfs_s0600_receipt.json`; actual outputs: `outputs/conservative_water/cfs_chastre_gpu_overlay_s0600/`. All interruption/failure records, original stores and previous qualifications remain preserved. Existing strict CPU/GPU comparison differences remain documented in [current status](current_status.md).

## Separately labelled eight-million-entry trial

At 01:37:37 UTC on October 15 (18:37 PDT October 14), root launched a fresh physical-zero replay from the same persisted INITIAL with `phase_max_nnz=8,000,000`. This changes the nonempty-entry count guard only. Physics, 32 bins, source packages, forcing, journal limits and all hard memory limits remain unchanged. Strict restart identity forbids migrating the four-million-entry wet checkpoints; the new trial instead produces its own checkpoints.

Independent preparation passed all preflight checks and 29 CPU tests. The actual bounded supervisor advances only to 300 and then 600 seconds before stopping for root acceptance of comparisons with the accepted four-million-entry runs. No beyond-600 gate is present. The first stage is in setup at this update; no new physical result or full-storm fit is claimed. Exact live status and evidence are under `agent_handoffs/tasks/chastre_capacity8m_20261014/`. The original refusal and all old artifacts remain preserved.

## Eight-million-entry stages observed complete — October 15

Both stages exited zero and passed the independent stage gate: fresh 0→300 seconds and own 300→600 seconds. The supervisor stopped at 02:36:40 UTC for the absent root-only 1200-second gate. GPU0 was released; this was the planned review stop.

Root subsequently compared each stage with its accepted four-million-entry baseline. All **291 numeric array members per comparison** are byte-exact, including final fields, hydrographs, checkpoint MAPLE state and wet continuation arrays. Summary metadata passes the existing explicit capacity/provenance/timing schema; all payloads bind to their own recorded hashes. Checkpoint snapshot metadata matches apart from the declared 4M→8M count limit. The original comparator reports failure solely because five measured checkpoint duration fields were omitted from its timing schema (`commit_wall_s`, `first_step_cpu_s`, `first_step_wall_s`, `remaining_cpu_s`, `remaining_wall_s`). These failures are retained. A bounded Claude correction must name those five leaves explicitly, preserve exact physical comparisons, and pass root verification before the longer continuation.

At 600 seconds: 600 accepted / zero rejected steps, 73 bed commits, zero hydraulic reroutes, maximum net displacement 0.000360512614465 m, and 3,395,369 sparse entries. Water residual remains 3.6379788071e-12 m³ and all six sediment residuals remain zero under unchanged tolerances. Host high-water RSS was 5,481,058,304 bytes. Retained CuPy pool after the loop was 8,767,983,616 bytes (not a measured peak), below its unchanged 9,092,045,209-byte limit. Setup/loop-with-checkpoints/driver wall were 729.452/780.083/1519.213 seconds; these are observations rather than a controlled speedup claim. Both replay stages added zero deep data chunks, deduplicating against the retained prior content. This does not prove later storm storage or GPU fit.

The next bounded preparation is `agent_handoffs/tasks/chastre_capacity8m_continue_20261015/`: correct only the five timing leaves, verify the actual comparisons, and attach a new controller to a new 600→1200-second child using the accepted 600-second predecessor. Original controller records and all prior failures remain immutable. No new GPU continuation has started at this update; full storm and large-case rerouting remain unproven.

The corrected comparisons subsequently **both passed** under independent root execution (`chastre_capacity8m_continue_20261015/root_baseline_comparison_acceptance.json`). Each comparison retains all 291 exact numeric members and strict typed physical metadata. The correction changes only the five explicitly named measured timing leaves; original failures remain archived. Root authorized the next 1200-second stage after these passes. Actual launch awaits independent controller/startup review; no additional model has started at this update.

## Actual continuation launched — 19:03 UTC October 15

After bounded Claude corrections, root independently passed 42 focused checks and the preflight using the actual 600-second checkpoint and hash-bound comparison reports. The controller now launches its initial child through the original tested supervisor ownership/cleanup path. Test fixtures were corrected to confine delegated gates to temporary directories and register exact-identity child cleanup; the initial fixture scope violation and prior failures remain documented rather than erased.

The new controller and 600→1200-second model are **live**, verified by exact process identities, with a fresh zero-use GPU0 probe before launch. The model is in setup/terrain hashing at this update. If each stage passes, the bounded controller proceeds through rain-off and recession toward genuine completion or the 10000-second ceiling. It stops on any failed gate, resource refusal, stop request or its 4-hour stage / 24-hour total guards. Physics, 32 bins, eight-million entry cap and hard memory limits remain unchanged. Full-storm fit and large-case hydraulic rerouting remain unproven. Live evidence: `chastre_capacity8m_continue_20261015/c8c_supervisor.status.json` and original `chastre_capacity8m_20261014/c8m_runs/c8m_s1200_root_b8.*`.

October 15, 19:24 UTC: the eight-million-entry continuation **stopped**. Its next step required 8,006,654 entries, above its configured guard; the refusal preceded write-array allocation. The model exited 1, was reaped, the controller closed, and GPU0 was released. This was not a reported GPU out-of-memory failure. A 720-second checkpoint was saved (720 accepted / zero rejected steps, 103 commits, no reroutes, 5,177,175 sparse entries); its metadata was inspected, but its payload/public restart has not been newly validated. The last fully accepted stage is still 600 seconds. Exact failure time was not persisted. Pre-existing store metadata is unchanged, with 835 new chunks / 390,981,456 bytes and four new versions retained; there are no temporary leftovers.

A bounded read-only Claude capacity diagnosis is underway under `chastre_capacity_analysis_20261015/`, examining occupancy growth, actual memory lifetimes and planning before another experiment. No new GPU model, count-cap increase, bin change, checkpoint migration or hard-memory-limit increase is authorized by that diagnostic task. Full-storm completion remains pending.

## Bounded memory profile started October 15

Root accepted an exact-physics replay of the already successful 600→720-second interval under the existing eight-million-entry configuration. Independent preparation passed 14 CPU test groups and 18 preflight checks. The profiler measures CuPy pool peaks and separates setup/restore, event-loop/checkpoint, output and subsequent checkpoint-validation allocations. Its wrappers were tested to avoid retaining old bed/state objects. Physical source packages, 32 bins, forcing, MAPLE inventory authority, routing rules and all limits are unchanged.

The worker and guard closed at 20:54 UTC October 15 and released GPU0. The replay reached 720 seconds with 720 cumulative accepted steps, no rejected steps, 103 bed commits and no hydraulic reroutes. Water and all six sediment-class budgets close under the original MAPLE tolerances; mobile sediment remains in the wet state. Maximum net elevation displacement remains 0.000451677120814509 m, below the 0.00075 m routing threshold.

After releasing its physical state (all tracked weak references dead), the worker successfully loaded the **original** saved 720-second checkpoint through the public GPU loader, using the original expected identity and static parameters. Root's subsequent host comparison passes: **107 numeric arrays are byte-exact**, embedded snapshot metadata and typed physical metadata match, and physical/logical deep manifests match. Only three declared measured duration fields differ. The replay adds no deep chunks or published versions; pre-existing store metadata, source packages, case and INITIAL bindings remain unchanged. This validates the saved physical state through 720 seconds; it does not complete the storm.

| Observed profile quantity | Bytes |
| --- | ---: |
| Event-loop/checkpoint peak live CuPy pool | 7,394,395,648 |
| Event-loop/checkpoint peak retained CuPy pool | 9,026,217,984 |
| Unchanged hard pool limit | 9,092,045,209 |
| Separate original-checkpoint public-load peak live pool | 3,496,865,280 |
| Pool live after physical state release | 152,982,016 |
| Host process high-water RSS | 5,649,387,520 |

The live storm peak is observed in actual MAPLE's voxel deposition at `transfer.py:_deposit_surface_mixture_batched_inplace:601`. It is an allocation site, not a breakdown attributing all live bytes to that expression. The profiler recorded all 120 accepted steps from 601 through 720 seconds, without dropped samples. Sparse entries grew from 3,407,100 at 601 seconds to 5,177,175 at 720 seconds. These are thread-local pooled allocation measurements; CUDA context, kernel-local and non-pool memory are excluded. Timings include instrumentation, setup/restore and checkpoint overhead and are not a controlled speed comparison.

**Diagnostic qualification:** the raw receipt and process exit remain failed (`model_pass=false`, exit 1), solely because the profiler classified every null allocation pointer as a failed allocation. A separate root GPU probe reproduces a successful zero-byte allocation with `size=mem_size=mem_ptr=0` on installed CuPy 14.2.0; the frozen hook counts it as failure. The profile's histogram has exactly 137 zero-sized events and 137 alleged failures, with no instrumentation exception. Claude's audit and six independently executed corrected-hook tests pass. The [pinned allocator source](https://github.com/cupy/cupy/blob/v14.2.0/cupy/cuda/memory.pyx#L1239) guarantees that zero-byte pool requests return null pointers. That source fact plus the observed aggregate equality establishes zero positive-size null-pointer events; the counters alone would not suffice for an arbitrary allocator. Failure classification does not change the measured maxima. Root accepts the physical checkpoint and pool measurements separately in `chastre_memory_audit_20261015/root_acceptance.json` (17 evidence checks), preserving the raw failed receipt. No full-profile retry or limit increase has been made.

Evidence: `chastre_memory_profile_20261015/cm_profile_s0720_receipt.json`, `root_profile_guard.final.json`, `root_actual720_comparison.json`, and `chastre_memory_audit_20261015/root_zero_allocation_probe.json`. Full-storm capacity and a genuine large-case hydraulic reroute remain unproven.

Two qualifications to the initial capacity diagnosis: the installed CuPy 14.2 allocator already attempts cache eviction under allocation pressure; SYRUP already releases previous attempt temporaries before the next step. Neither is an established defect or optimization target merely from the after-loop cached-memory reading. Source evidence and independent review are retained in the profile task.

## Next bounded memory candidate

The first isolated MAPLE deposition candidate in `chastre_deposition_memory_20261015/` combines the eager broadcast product and two `where` expressions into an exact FP64 GPU operation. Independent checks pass: 156 existing MAPLE CPU transfer tests, 24 harness checks and actual original/candidate GPU comparison. All tested outputs match exactly. At a synthetic 1,140,189-cell W3 batch, live pool peak falls from 1.009 GB to 0.928 GB and the warm median from 28.51 ms to 26.69 ms. See [component results and limits](deposition_memory_candidate.md).

The measured peak moves to a later allocation while already-used scratch arrays remain referenced. A second bounded task, `chastre_deposition_lifetimes_20261015/`, prepares an isolated last-use release candidate. It may add only `del` statements after confirmed last uses; numerical operations, reduction order, gates, mutation and accounting remain unchanged. Root will review and measure this variant before any production adoption. Component improvements alone cannot establish full-storm fit. Any source change has a new recorded identity and cannot silently resume the existing wet checkpoints; production dependency qualification and a fresh equivalent replay remain separate steps.
