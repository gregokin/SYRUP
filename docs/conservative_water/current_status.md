# Conservative water status — 2026-10-14 UTC

## Completed large-case qualification

The fresh current-source **rotated Chastre GPU0 overlay reference completed its 300-second wet pause**: 300 accepted steps, zero rejected attempts, 24 accounting commits, zero hydraulic reroutes, closed water and class-sediment/request balances, and maximum net elevation displacement 0.0002133491 m. Geometry is 886×1780 with 1,140,189 active cells, W3 and six classes. Original resource caps and options were retained.

It wrote **6,752,179,632 bytes of new deep-bed data** (6,744,659,376 overlay bytes and 7,520,256 keyframe bytes), versus the earlier measured 55,251,320,832 raw bytes. That is 8.18 times less new deep data; initial raw storage and metadata are additional, and old referenced files were retained. The earlier run had historical chunk deduplication. A normalized fresh full-chunk counterfactual from identical logical versions is 90,886,053,888 bytes, not a newly executed run. Loop plus checkpoints took 1,708.8 s versus 2,038.4 s previously; these are observations with different source/storage initialization and cache conditions. Fresh initial assembly took another 964.1 s. Host high-water RSS was 4.43 GB; retained GPU pool after the loop was 8.82 GB under the original 9.09 GB ceiling, not a measured GPU peak.

Independent root comparison reproduces Claude's result: **all 291 numeric members (6,084,176,934 bytes) are raw-exact to the earlier same-GPU reference**, including final state, hydrograph and both 240/300-second checkpoints. Typed metadata is strict except 61 individually validated source/storage/INITIAL/path/timing declarations. All 19 focused comparator tests pass. This establishes preservation for this particular wet slice, not universal equivalence or full-storm fit.

The first own-240-second restart was interrupted without an exit status, completion receipt or output. Its child and launcher are absent on the host, GPU0 is idle and the store still has the reference's 15 published manifests. The cause is unknown; the attempt is preserved and does not count as a model pass or diagnosed model failure. The path-only recovery passed independent Codex and Claude review and completed on GPU0 at 15:10:56 UTC. Its actual own-240-to300-second continuation is now independently accepted: 184 numeric arrays raw-exact, strict physical metadata, validated deep files and the complete 15-manifest lineage to genesis. The saved 240-second state contains flowing water in 543,587 cells and 17,350.6566 kg of mobile sediment; this is a genuine wet restart. Final physical and logical deep identifiers equal the uninterrupted reference. The recovered run took 1,407.1 s (setup 717.1 s, loop/checkpoints 678.6 s); previous same-configuration observations were 1,525.7/718.0/803.7 s. These timings remain observational. All 113 comparator controls pass. The original interrupted attempt remains preserved without an invented exit status.
Source pair: frozen SYRUP B108C1 `5f85a989…` and delivered MAPLE `24c6e147…` (bundle/cache `b8fa43a3…`). These qualification runs imported the frozen 5f tree and the full-project 24c cache, never the root working tree.

**Primary adoption accepted (CW-E48, 2026-10-14).** After this accepted large own-wet restart, the root working tree adopted the exact reviewed SYRUP 5f runtime (`src/maple_syrup`, 55 files, package `5f85a989…`), its tests and docs, the B109 test-only files and the accepted CW-E46 MAPLE 24c runtime recipe (package `24c6e147…`; runtime-only bundle/cache `7ee10bd8…`). The previous root binding, B55/M44, is now historical; its archived baseline is `agent_handoffs/tasks/conservative_water/cw_e47_baseline_archive`. See [usage](../conservative_water_usage.md). The accepted source, tests, dependency recipe and documentation are included in this commit. Existing MAPLE environments are used without installation.

## Independently executed current-source Plot1 evidence

Fresh actual-MAPLE initial beds were used, with no cross-source wet checkpoint migration. Every run completes at 4,819 seconds, with 4,819 accepted steps, zero rejected attempts, 809 commits, no hydraulic reroutes, closed water/class-sediment/request accounts and zero final mobile sediment. Soil reset removes 25.21842049940974 m³ explicitly under the dry-again assumption.

| Check | Exact numeric arrays | Controls |
|---|---:|---:|
| CPU full storage vs overlays | 908 | 59 CPU controls across storage/restart |
| CPU own 2400-second restart | 549 | included above |
| GPU full storage vs overlays | 1089 | 65 |
| GPU own 2400-second restart | 650 | 85 |

Parsed physical metadata also matches. Only explicitly named timings, storage identities and validated per-invocation GPU fields are qualified individually. The corrected reader refuses malformed NPY headers, duplicate ZIP members and invalid storage record types before materialization. Unknown numeric counters remain exact. The own 2400-second continuations publish no new deep version; separate B109 fresh-process CPU/GPU fixtures prove actual post-resume overlay publication and old-file retention.

Observed wall times: CPU full 130.2 s/overlay 136.0 s; GPU full 942.6 s/overlay 825.2 s. These small-case, separately timed observations do not establish controlled acceleration. New overlay file bytes are 235,992 on CPU and 281,288 on GPU. See [exact storage](exact_overlay_storage.md) for scope and evidence.

## Scientific and resource qualifications

Hydraulic rerouting uses only the largest per-cell net elevation displacement since its held routing surface, compared with MAPLE's 0.00075 m threshold. Time, reporting, checkpoints, gross turnover and ordinary or final commits do not trigger it alone. Exchanges continue at accepted physical steps. See [net routing](net_elevation_routing.md).

The earlier rotated Chastre 300-second CPU/GPU comparison used 8ac/M94. Physical-field and exact decision checks pass, but four/seven strict near-zero accounting/residual failed records remain at 240/300 seconds. Both budgets independently close; no tolerance or failed-field waiver was introduced. These results are [qualified agreement](rotated_backend_comparison.md), not a blanket backend pass. The current-source Plot1 comparison independently reproduces 89 strict failed records among 2716 checks, with all 26 controls passing. Water, reconstructed physical columns, inventory, transport and hydraulic/event decisions pass original bounds. In addition to near-zero accounting failures, resident-window base differs in two cells after the 3000-second checkpoint and version generation is 2 on CPU versus 4 on GPU. These storage placement/publication decisions remain exact failures even though reconstructed columns pass. A read-only source/local-state trace confirms valid alternative placements and two additional GPU publications with unchanged logical values (50,504 bytes). Checkpoints do not determine the intervening branch, step or full canonical journal key set. No concrete conservation defect is established, and no model fix or field waiver has been applied.

State-difference replay of historical deep versions cannot recover their unpersisted no-op journal keys. It would understate production overlay costs and cannot guarantee original manifest identity. That large offline experiment is held in favor of an actual fresh current-source run. Old stores remain preserved.

## Qualified completion and remaining limits

The initial water implementation/adoption program is now **independently accepted within the bounded CPU/GPU scope**. Actual-root conservative CPU suites and routing guards pass; recipe reconstruction, both environment imports, GPU storage/lifetime/pit checks and fresh-process CPU/GPU continuation with new deep publication pass. The accepted code is included in this commit. See [acceptance and exact coverage](acceptance_20261014.md).

Full Chastre storm fit is unproven, and the strict CPU/GPU qualifications above remain. Terminal-commit reactivation refuses a handoff instead of automatically restarting recession. The Fortran surplus is documented only. Legacy replay remains the default benchmark workflow; wind handoff, direct dry-cell splash, growth, full ecohydrology/ET and nutrients remain outside this initial component.

Durable execution records: `agent_handoffs/tasks/conservative_water/cw_c39_root_*acceptance.json`, `cw_b111_root_preparation_acceptance.json`, `cw_b114_root_actual_restart_acceptance.json`, `cw_e48_root_primary_adoption_acceptance.json`, `cw_e47_report.md`, `cw_b108c1_root_delivered_runs/root_b111_chastre_gpu_overlay_reference.*`, and `agent_handoffs/orchestration_state.md`.
