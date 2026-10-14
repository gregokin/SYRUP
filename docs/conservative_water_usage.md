Current consolidated status: [2026-10-14 qualification and active work](conservative_water/current_status.md). The dated entries below retain historical context.

# Conservative water: CPU and GPU usage

**Current — 2026-10-14 UTC (CW-E48 independently accepted bounded primary adoption; included in this commit).** The root working tree now carries the exact reviewed SYRUP 5f runtime (`src/maple_syrup`, 55 files, package `5f85a98989d3e9e5b9d59e32a68f5d46d481985f0930961b26c89898bd8776a9`) and the CW-E46 MAPLE 24c runtime recipe (package `24c6e1478ed732df22f448eddf8cb16c3b72f2406dc9b37a9414cee9a2d1a9ce`). Before adoption, the current-source pair completed the rotated Chastre 300 s wet slice on GPU0, and its recovered fresh-process restart from its own 240 s checkpoint is independently accepted: 184 numeric arrays raw-exact, strict physical metadata, validated deep files and the complete 15-manifest lineage (`cw_b114_root_actual_restart_acceptance.json`). Those runs imported the frozen 5f tree and the full-project 24c cache, not the root tree; Root GPU checks of the adopted tree now pass (storage/event, lifetime, pit completion and 140-array fresh-process wet continuation); see [qualified acceptance](conservative_water/acceptance_20261014.md). Routing refreshes only on net per-cell elevation change. Exact overlay storage is an explicit option. Current Plot1 CPU/GPU retains 89 strict failed records and the earlier 8ac/M94 rotated Chastre comparison four/seven, as qualified physical agreement. Full Chastre storm fit and universal backend equivalence remain unproven; initial bounded water qualification is accepted. See [current status](conservative_water/current_status.md).

**Previous summary (2026-10-14, before CW-E47; historical).** The isolated B104C2/M94 candidate completes the rotated Chastre 300 s wet slice on CPU and GPU0; the GPU own-wet restart is exact across 184 numeric members. Full CPU/GPU comparison is pending a measurement-tool representation correction, with unchanged tolerances. Routing refreshes only on net per-cell elevation change. Optional exact overlay storage has passed bounded MAPLE checks and is being integrated separately into SYRUP. These candidates are not adopted into the root runtime. Full-storm fit and final water-component qualification remain open. See [current implementation status](conservative_water/implementation.md). The runtime instructions below describe the root B55/M44 baseline; subsequent candidate results do not change that binding.

**Historical status (root B55/M44 binding before CW-E47).** The reviewed water runtime and delivery helper had been integrated into the SYRUP working tree. Final water-component qualification is still in progress; this is not a completed release and has not been committed or published.
- The current integrated water runtime combines reviewed B46, the B52/B54 CUDA scratch-lifetime changes, the explicit B51 completion option and the E2 dependency-delivery helper (frozen together as B55). Frozen B46 test runs retain their own source identity. A checkpoint from B46 cannot be resumed by substituting these newer sources.
- The MAPLE dependency these runs were qualified with is the accepted, bounded private candidate **M44** (`outputs/dependencies/maple_cw_b44_lifetime_frozen`). M44 is a source candidate: not a final MAPLE version, not a production release and not qualified for a large GPU fit.
- CW-E6 exported and materialized M44 as a CW-E2 local bundle (see [Historical pinned M44 artifact](#historical-pinned-m44-artifact-cw-e6)). This is a content-addressed **integrity** record, not authenticity. Nothing was installed or published.
- Large Chastre has a verified CPU wet 240-second pause and a genuine fresh-process wet 300→360 s restart with 163 numeric members raw-exact (B55/M44). Neither is a completed large storm. The latest B77 GPU trial on private B76/M75 still refuses a 107,250,176-byte proportional-mixture allocation during deposition at the unchanged 9,092,045,209-byte cap, with zero accepted steps. A separate private lifetime correction B79 is under implementation; no large GPU fit is claimed.
- The separate GPU INITIAL assembly has now passed an independent comparison: all 11 resident numeric members and the baseline match the CPU initial data exactly; only the actual deep-store location differs. This provides independent storage, not GPU storm qualification. Both CPU continuations have now completed and the strict fresh wet-restart comparison is accepted on their immutable B55/M44 sources.
- Earlier RFID CPU/GPU runs conserve water and sediment, but their original strict pre-reset-depth and conveyance comparisons retain small failures. Root independently accepts a documented numerical qualification of that B9-era pair: one conveyance failure is propagated from 1–5-ULP terrain differences; one depth failure (6.2256×10⁻¹⁴ m) remains unattributed. Both strict failures stay recorded and no threshold changes. The newer B76/M75 RFID CPU event completed with closed budgets at 4372 s (4687 accepted, 327 rejected, 905 commits); its matched GPU event is running. Neither the old disposition nor inventory correspondence qualifies the 53.97 m no-spill terminal pond as flood hydraulics.
- Private B76/M75 full Plot1 CPU/GPU storms both complete at 4739 s / 802 commits. The CPU fresh wet restart is raw-exact across 209 numeric arrays. CPU/GPU fields retain one strict voxel-window failure (4.0856×10⁻¹⁴kg), while the separate actual-exchange operand-qualified MAPLE inventory comparison passes. Identity/source/INITIAL/output bindings independently pass; no strict failure is silently reclassified. Final primary adoption and the final large GPU checks remain open.
- See [Status of results](#status-of-results) for what has and has not been demonstrated.

## Two workflows: frozen legacy default vs explicit conservative

| | Legacy replay (DEFAULT benchmark workflow) | Conservative water (EXPLICIT) |
|---|---|---|
| Entry | `maple-syrup-legacy`, `maple-syrup-benchmark` (default `--transport-scheme legacy`) | `maple-syrup-event` (dense bed), `maple-syrup-windowed` (resident-window bed), or `python -m` the same modules |
| Sediment | fixed composition, unlimited supply, explicit clipping source | actual evolving MAPLE active layer and voxel bed, supply-limited, no clipping source |
| Restart / wind handoff | not a conservative complete-event, restart or wind-handoff model | conservative complete event, dry reset, same-backend checkpoint/resume; **no wind handoff** |

Nothing in this candidate changes the legacy default. The conservative model runs only when you select one of its entry points.

## Environment and the MAPLE dependency

MAPLE is deliberately not a package dependency (see `pyproject.toml`). SYRUP resolves `maple` from the running environment, and `maple_syrup.dependency` verifies its identity, source origin and required API. Use only the existing MAPLE interpreters:

- CPU: `/home/okin/MAPLE/.venv/bin/python`
- GPU: `/home/okin/MAPLE/.venv-cupy/bin/python`

Pin the dependency to an immutable MAPLE copy, never a live checkout. **Current binding (CW-E47):** the root SYRUP `src` with the CW-E46 **runtime-only** MAPLE 24c bundle/cache `7ee10bd81b0859a891ff8a476cab2c8824bb0964309695ca948d12d5ac4b0c71` (341 runtime files + reconstruction receipt; package `24c6e147…`):

```bash
C=/home/okin/SYRUP/outputs/dependencies/content_cache/7ee10bd81b0859a891ff8a476cab2c8824bb0964309695ca948d12d5ac4b0c71
S=/home/okin/SYRUP/src
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$C/editable:$C/source/src:$S
export MAPLE_SYRUP_EXPECTED_MAPLE_ROOT=$C/source   # the resolver refuses any other resolved MAPLE source
```

**Two 24c artifacts, one package.** The runtime-only bundle `7ee10bd8…` (project digest `4d371597…`: 341 runtime files plus the reconstruction receipt) and the accepted **full-project** qualification artifact `b8fa43a3…` (1,278 project files, project digest `f8aaff8d…`, with MAPLE tests/docs/build files) resolve the same MAPLE package digest `24c6e147…`. They are different artifacts with different bundle ids and project digests. The large qualification runs used the immutable full-project cache `b8fa43a3…` with the frozen 5f tree; either cache satisfies the same run identity, which binds the package digest, not the bundle id.

**After a fresh clone** (no git-ignored `outputs/` copies), rebuild the exact 24c runtime from upstream MAPLE Git with the recipe ([maple_water_dependency_recipe.md](maple_water_dependency_recipe.md): `tools/reconstruct_water_maple.py` + `dependencies/maple_water/`). Then wrap it with the CW-E2 helper (export → materialize → `env`, exact commands in that document). The reconstructed `src` alone on `PYTHONPATH` is refused by the MAPLE resolver by design, because the venvs' own `maple` distribution names the live checkout. The recipe needs a MAPLE repository containing commit `74dd3e79…`, and its provenance evidence is local; see that document's availability limits. These recipe files are present in the working tree but not yet committed.

A CW-E2 materialized bundle works the same way: `python -m maple_syrup.dependency_bundle env <cache>/<bundle_id> --syrup-src $S` prints the `PYTHONPATH` and `MAPLE_SYRUP_EXPECTED_MAPLE_ROOT` to use. See [cw_e2_dependency_delivery.md](cw_e2_dependency_delivery.md). The helper needs the optional `dependency-delivery` extra (`packaging`); the model runtime does not.

**Historical binding (B55/M44, before CW-E47):** `M=/home/okin/SYRUP/outputs/dependencies/maple_cw_b44_lifetime_frozen`, `PYTHONPATH=$M/editable:$M/source/src:$S`, `MAPLE_SYRUP_EXPECTED_MAPLE_ROOT=$M/source`; its runtime-only and CW-E6 full-project artifacts resolve package `44e4f4c6…`. The 5f runtime refuses the exact-overlay option on any MAPLE other than the accepted 24c source; its default full storage is not qualified against M44.

### Historical pinned M44 artifact (CW-E6)

M44 has been exported and materialized by root with the CW-E2 helper. Evidence: `agent_handoffs/tasks/conservative_water/cw_e6_M44_delivery_receipt.json` and the files it names.

| Field | Value |
|---|---|
| `bundle_id` (artifact) | `43a15ee91d799a893398e08269b1bebf76517f8590b19290a253911488f89673` |
| MAPLE package source digest (`src/maple`, 339 files, 8,099,478 bytes) | `44e4f4c62d0262bfed57973640a3378604e9fbeb7923eb53597319838be8630a` |
| Project digest (1,259 selected files, 26,085,818 bytes) | `fabe14778428c9aea77fdbd9136d43d36b94a46d58b1bc1c08e8e0622c5bf508` |
| Archive `maple-project.tar` (1,263 members) | 27,064,320 bytes, SHA-256 `5a85361fa853784d7d1e09217ea9457efa44e720069b75f18bb02df48e5eb509` |
| Exported from | `outputs/dependencies/maple_cw_b44_lifetime_frozen/source`, MAPLE `0.0.1` |
| Tool | CPython 3.12.3, bundle schema `maple_syrup.dependency_bundle.v1` |
| Bundle directory | `outputs/dependencies/bundles/cw_M44_reviewed` |
| Read-only materialized cache | `outputs/dependencies/content_cache/43a15ee9…f89673/` |
| Origin record | `cw_e6_M44_origin.json` |

**Origin record contents.** Local git revision `74dd3e79…` plus a recorded dirty-package snapshot, then the patch chain:
- original dirty files;
- the earlier bed optimisation `72310c49`;
- water M29;
- water M44, which changes only two runtime files relative to M29 (net patch `cw_e6_maple_M29_to_M44.patch`).

**Resolver checks.** Root's recorded runtime-resolver results cover both existing MAPLE environments (`cw_e6_M44_runtime_resolver_0.json` and `_1.json`); per the receipt, no GPU execution and no install was involved. Each resolved `maple` 0.0.1 from the cache's `source/src/maple`, as an editable working tree, with package source digest `44e4f4c6…be8630a`.

To run against the cache, use the environment printed by `env`. The SYRUP `src` there is whichever tree you pass as `--syrup-src`:

```bash
C=/home/okin/SYRUP/outputs/dependencies/content_cache/43a15ee91d799a893398e08269b1bebf76517f8590b19290a253911488f89673
export PYTHONPATH=$C/editable:$C/source/src:$S
export MAPLE_SYRUP_EXPECTED_MAPLE_ROOT=$C/source
```

**Limits:**
- **Integrity, not authenticity.** There is no signature or trust anchor.
- Third-party Python packages are not bundled.
- The artifact is not installed into any environment and not published.
- M44 remains a bounded private source candidate, not a final production MAPLE, and large-GPU-fit qualification remains OPEN.
- Run identities bind the resolved MAPLE package source digest. A run against the cache resolves the same package digest as one against `maple_cw_b44_lifetime_frozen`, but the bundle id is a separate integrity record; it is not part of the run identity.

**Console scripts.** The `[project.scripts]` names, including the new `maple-syrup-windowed`, exist only after an installation of this project. This candidate has not been installed anywhere. Without installation, use `python -m maple_syrup.windowed_experiment` / `python -m maple_syrup.event_experiment` with the `PYTHONPATH` above.

## Resident-window conservative runs (`maple_syrup.windowed_experiment`)

Every option below is the actual CLI (`--help` lists the same).

| Option | Meaning |
|---|---|
| `--case-dir`, `--output-dir`, `--store-dir`, `--case-kind {plot1,rfid,chastre}` | Required. Output and checkpoint directories must be new. |
| `--backend {numpy,cupy}` | Default `numpy`. The hydraulics implementation follows it: `numba` on numpy, `cuda` on cupy. There is no fallback in either direction. |
| `--root-solver {bisection,newton}`, `--bisection-iterations` | Default `bisection`, 40. |
| `--window-voxels` | Resident voxel window width. Default 4. |
| `--phase-max-nnz` | Predeclared capacity of the sparse (CSR) phase bins. Default 1,000,000. Refused before allocation when exceeded; never truncates mass. CSR is this runner's phase storage. |
| `--max-journal-records` | Pending deep-store journal cap. Default 1,000,000. |
| `--max-host-bytes` | Host cap. Default 40 GiB. |
| `--max-store-bytes` | Deep-store disk cap. Default 256 GiB. |
| `--deep-chunk-encoding {full,overlay}` | Deep-chunk storage. Default `full`. `overlay` stores a raw base plus one cumulative exact final-value overlay per chunk (no tolerance dropping, no chains); it is accepted only on the delivered MAPLE 24c source and refused before setup otherwise. |
| `--deep-overlay-max-bytes` | Overlay only: explicit per-publication overlay byte cap forwarded to MAPLE. Default: MAPLE's own rule. |
| `--max-checkpoint-bytes` | Resume bound. Defaults to `--max-host-bytes` and may only tighten it. |
| `--max-end-s`, `--pause-at-s`, `--checkpoint-every-s` (default 600) | Run window, optional pause, checkpoint interval. |
| `--reuse-initial` | Restore the persisted initial window assembly from `--store-dir` instead of re-importing it. |
| `--resume <checkpoint>` | Continue from a checkpoint (see below). |
| `--nodata-value` | **Required** for `rfid` and `chastre`: the case's declared DEM nodata. It is never inferred. |
| `--allow-flat-storage`, `--terminal-storage-water` | Pit policy (see limits). |
| `--record-exchange-history` | Diagnostic per-cell/class ledger history. Default **off** (the production setting, and off in the large qualification runs); no physical effect. It consumes additional memory. |
| `--deposit-mobile-at-flow-end` | Explicit dry-again assumption: after rain ends and every hydraulic quiet gate has held for the required interval, MAPLE deposits the remaining finite mobile load. Default off. |
| `--allow-maple-source-change` | See below. |

The windowed CLI has no `--expected-maple-root` option; it uses `MAPLE_SYRUP_EXPECTED_MAPLE_ROOT`. The dense runner accepts both.

**API-only options** (not on the windowed CLI; pass them to `maple_syrup.windowed_experiment.run_complete_windowed`): `report_every_s` (default 60), and deep publication `deep_publication='step'` (default: publish the pending MAPLE deep journal at every accepted step) or `'boundary'` with an explicit `deep_publication_high_water_records` below `max_journal_records` (carry the staged journal and publish at report / checkpoint / return boundaries, at the high-water mark, and in one bounded flush on the typed append-cap refusal). `'boundary'` needs the characteristic CSR transport and a resident-window bed.

**Routing and accounting.** Hydraulic rerouting is triggered **only** by net elevation change: the largest per-cell |exact committed surface − held routing surface| over active cells, compared with MAPLE's `pending_elevation_trigger_m` (0.00075 m) using `>=`. Within-cell gains and losses cancel; different cells never cancel. Elapsed time, gross turnover, reporting, checkpoints, deep publication and ordinary or final forced accounting commits never reroute on their own. Accounting commits follow the case's MAPLE `topographic_commit` triggers (defaults: 60 s interval, 0.00075 m pending elevation, 0.035 active-layer turnover fraction); a net threshold crossing between them still forces a genuine bed commit and reroute. Water and sediment exchange continue at every accepted physical step; the held routing surface is not a held sediment inventory. See [net routing](conservative_water/net_elevation_routing.md).

**Plot1 (CPU, Numba).** The compiled case is `outputs/plot1`. Example (history off; add `--deep-chunk-encoding overlay` for exact overlay storage):

```bash
/home/okin/MAPLE/.venv/bin/python -m maple_syrup.windowed_experiment \
  --case-kind plot1 --case-dir outputs/plot1 --backend numpy \
  --store-dir <NEW store> --output-dir <NEW output> \
  --window-voxels 4 --phase-max-nnz 1000000 --max-journal-records 1000000 \
  --max-end-s 5400 --checkpoint-every-s 600 --bisection-iterations 40 \
  --allow-maple-source-change
```

Historical B55-era root Plot1 runs used the same command with `--record-exchange-history`.

**Plot1 (GPU).** Use the same command with `/home/okin/MAPLE/.venv-cupy/bin/python`, `--backend cupy` and `--reuse-initial`. A store assembled by a CPU run can be restored, but its provenance is recorded.

**Device selection.** Select the device with `CUDA_VISIBLE_DEVICES=<GPU0 UUID or index>`. The runner uses device 0 of the visible set. GPU0 is the user's allocation: check that it is free first, and never disturb other devices or jobs.

**Mandatory resource caps.**
- A CuPy run installs a hard limit on the CuPy default memory pool, at 0.85 of the device's *current* free memory, before any upload. It honours a stricter existing limit and restores the previous limit afterwards.
- An out-of-memory error inside that limit becomes an explicit **resource refusal**: no step is published, no new checkpoint or output is written, earlier checkpoints stay valid, and nothing is retried or clipped.
- The pre-run memory plan is a known *undercount*. It is a necessary pre-check, not a bound and not a fit claim.
- Host, store, journal, phase and checkpoint caps are always explicit integers.

**RFID / Chastre.** Use `--case-kind chastre --nodata-value -9999 --terminal-storage-water` and larger `--phase-max-nnz` / `--max-journal-records` (root used 4,000,000). The historical Chastre case was `outputs/chastre/case_v2`; the current qualified case is the rotated `outputs/conservative_water/cw_r1_rotated_chastre_case`.
- **Qualified current rotated-Chastre configuration** (B111 / B114 driver, GPU0; exact values): `case_kind='chastre'`, `window_voxels=3`, `backend='cupy'`, `implementation='cuda'`, `max_end_s=10000`, `pause_at_s=300`, `report_every_s=60`, `checkpoint_every_s=240`, `bisection_iterations=50`, `phase_max_nnz=4_000_000`, `max_journal_records=4_000_000`, `nodata_value=-9999`, `allow_maple_source_change=True`, `record_exchange_history=False`, `deep_publication='boundary'`, `deep_publication_high_water_records=3_000_000`, `deep_chunk_encoding='overlay'`, `deep_overlay_max_bytes=None`, `max_host_bytes=40 GiB`, `max_store_bytes=256 GiB`, policy `terminal_storage_water=True`, `deposit_mobile_at_flow_end=True`. Because `deep_publication` is API-only, this configuration is a `run_complete_windowed(...)` call, not a CLI line; the GPU entry applied the original 9,092,045,209-byte pool ceiling.
- Masked nodata and pit storage are enabled for these kinds; flat storage defaults on for Chastre.
- **Large Chastre:** the rotated 300 s wet slice and its own 240 s restart completed on GPU0 under the original caps; a full large storm is not yet known to fit. See status.

## Dense-bed conservative runs (`maple_syrup.event_experiment`)

`python -m maple_syrup.event_experiment --case-dir <compiled plot1 case> --output-dir <NEW>` runs the complete Plot1 event on dense MAPLE bed storage.
- Defaults: `--backend numpy --implementation numba`; `--backend cupy` requires `--implementation cuda`.
- Phase storage `--phase-storage dense|csr`, `--transport-scheme characteristic|upwind`, `--phase-bins`.
- Pit opt-ins `--allow-pit-storage`, `--allow-flat-storage`, `--allow-masked-nodata --nodata-value`, and the completion thresholds `--hold-s` and `--max-depth-m`.
- `--expected-maple-root` / `--mahleran-root`.
- `--max-checkpoint-bytes` defaults to 1 GiB.
- `--deposit-mobile-at-flow-end` selects the same explicit opt-in policy as the windowed runner.

## Checkpoint, resume and identity

- Checkpoints are written only at accepted step boundaries, into `<output>_checkpoints/step_NNNNNNNNN` unless the dense runner is given `--checkpoint-dir`.
- `--resume` requires the checkpoint's recorded run identity to equal the current one exactly. That identity includes:
  - the SYRUP source tree digest and the **resolved MAPLE package source digest**;
  - the case identity, the import binding and the MAPLE configuration;
  - the MAHLERAN XML and rainfall hashes, and the sediment and column parameters;
  - the solver control, the completion policy and the graph policy;
  - `max_end_s`, the report cadence and the report row cap. A resume must use the same `--max-end-s`.
  - the backend (plus `cuda_mode` on CUDA), and for resident-window runs the bed-storage configuration (window, store and journal caps, initial deep manifest).

  The checkpoint byte bound is a resource control and is deliberately excluded from the identity.
- **Resume is same-backend only.** A CPU checkpoint does not resume on CuPy, or the reverse.
- During a run, the SYRUP source, MAPLE source, MAHLERAN XML and rainfall source are re-hashed at every checkpoint. A change aborts the run.

**`--allow-maple-source-change` acknowledges INITIAL case import only.** It accepts that the MAPLE source digest recorded when the compiled case was imported differs from the resolved MAPLE now; the difference is recorded in the summary. It is **never** a restart-identity bypass. A resumed run still refuses any checkpoint whose recorded MAPLE (or SYRUP, case, configuration or backend) identity differs.

## External deep store retention

The resident-window bed keeps deep voxels in an immutable, versioned MAPLE host store under `--store-dir`. Final outputs and checkpoints hold resident arrays plus **references** to store versions; they are **not portable standalone archives**.
- **Retain the store directory** for as long as any output or checkpoint that references it must be readable or resumable.
- Old versions are never rewritten. Unpublished deep updates are refused by commits, exports and checkpoints.
- A CW-E2 bundle carries the MAPLE dependency only, never deep-store data.

## Event completion and dry reset accounting

A run continues after rain until flow and transport cease: the quiet hold, depth, discharge and mobile-mass thresholds of the completion policy. A forced final terrain commit follows. Then an explicit **dry reset** removes the remaining surface water, and soil water, as an *external, reported* removal. Both amounts are kept as arrays and totals in the handoff and summary; they are not hidden in the balance.
- Mobile sediment must be resolved before the handoff; it is never erased.
- By default the mobile-mass significance threshold is part of the quiet criteria. With `--deposit-mobile-at-flow-end`, only that completion criterion is waived: finite/nonnegative sediment validation, rain-off, every hydraulic gate, the hold, forced terrain commit, post-commit quiet checks, and all conservation checks remain mandatory. MAPLE deposits the whole remaining pool, records its pre-settlement mass and actual deposition, and refuses completion if any mobile mass remains. This is an explicit end-of-event modeling assumption, not a new tolerance or a change to wet transport physics. See [the policy specification](cw_b51_flow_end_settlement.md).
- `--terminal-storage-water` lets the event complete with water ponded **only** in terminal-storage pit cells. That water is removed and accounted by the same reset.

## Scope limits

- **Supply and availability.** Pickup is limited by actual MAPLE active-layer holdings and the case's availability state:
  - Plot1 uses its declared uniform 100% availability;
  - availability *builders* are refused;
  - wind-specific availability restrictions are not applied to water.
- **Pits.** Terminal pit storage, with deposition and ponded water retained and accounted, is an explicit opt-in.
  - Pit overtopping is **not** supported.
  - Masked avalanching is refused, and the resident-window runner disables avalanching for this water-only scope.
  - Routing is refreshed **only** when a commit's net per-cell elevation displacement from the held routing surface reaches MAPLE's `pending_elevation_trigger_m` (0.00075 m, `>=`); ordinary accounting commits (time / turnover / pending triggers and the final forced commit) keep the graph. A refreshed graph still refuses pits, flats and nodata receivers outside the opt-ins.
- **Excluded:** wind handoff, direct dry-cell splash, vegetation growth, full ecohydrology/ET and nutrients. The legacy replay remains a separate benchmark workflow, not MAPLE authority.

## Historical results and remaining qualification

Reported from the archived Codex verification receipts; editing this document ran no model.
The dated status at the top and [implementation status](conservative_water/implementation.md) supersede the historical candidate and active-process descriptions below.

**Small cases:**
- With M44 and the B46 runtime, the full CPU Plot1 storm reproduces all 191 saved numerical arrays and the reset accounting raw-exactly.
- Small CPU/GPU coupled complete and wet-restart checks are accepted.
- The full GPU Plot1 storm also completed: all 212 saved numerical arrays and reset accounting are raw-exact against its preceding GPU reference. The CPU comparison contains 191 arrays. These are within-backend regression comparisons, not a blanket CPU/GPU bitwise-equality claim.
- The opt-in B51 completion policy passed 71 CPU tests (two device skips), 32 tests in an actual-device invocation (no skips), and four additional actual GPU disk-restart/history/default-malformed checks. The extra disk reload checks use the same process; they are not a new cross-process qualification.
- The initial M44 Chastre reimport completed with unchanged physical initial-array and deep-store bytes.
- Initial large CPU checkpoint/resume passed for an infiltration-only two-second segment: 174 saved fields and full source/configuration provenance agree exactly after loading a 3,868,449,626-byte checkpoint bundle. This establishes the initial storage/restart path. A separate 240 s run subsequently saved real flowing water and mobile sediment; its fresh wet continuation comparison remains pending.
- CW-E6: the M44 artifact was exported and materialized, and both existing-environment resolvers resolve it (see [Historical pinned M44 artifact](#historical-pinned-m44-artifact-cw-e6)). This is delivery integrity only; it is not a physics, fit or authenticity result.

**Still OPEN:**
- the large Chastre GPU fit;
- full qualification and performance;
- large GPU restart;
- final accumulated review and qualification. The bounded source integration into root is complete; final component acceptance is not.

The latest history-on large GPU attempt (B50, M44/B46) passed first bed pickup, then refused a 54,728,192-byte transport scratch allocation at 9,081,758,720 bytes retained under the unchanged 9,092,045,209-byte pool cap; no step or checkpoint was published. The completed history-off B53 probe passed pickup and CSR count/write, then refused the first face allocation: 107,316,736 bytes requested at 9,004,742,656 bytes retained under the same 9,092,045,209-byte cap. It accepted no step. History off is the production-default diagnostic setting; it is a distinct configuration from B50. The combined CUDA scratch changes passed 282 actual-device checks, two coupled/wet-restart checks and two flow-end settlement checks. The full B55 GPU Plot1 storm reproduces all 212 saved arrays and reset accounting exactly; its loop plus checkpoints took 708.31 seconds, setup 8.15 seconds, under concurrent host work (observational timing). The next capped large probe is running; large fit is still open. Earlier documented Plot1 export disagreement with the native Fortran replay remains (see `docs/conservative_water/windowed_qualification.md` in the root tree).

## Tests

- **Default collection** (`pytest` in this tree): the phase suites plus the four general conservative suites `tests/conservative_water`, `tests/conservative_csr`, `tests/conservative_csr_event` and `tests/conservative_pits`. GPU tests there skip with an explicit "no claim" reason when CuPy or a device is unavailable.
- **Not collected by default:** the historical task families that need frozen oracle snapshots or task evidence under `/home/okin/SYRUP/outputs/...`. Run each one **in its own pytest invocation, never combined with the four conservative suites or with each other.**
  - **Why separate invocations:** `tests/cw_e2` (like `tests/phase2`) does `from conftest import ...`, and its `conftest.py` is a bare module. In one session together with the conservative suites, pytest's prepend import mode binds `conftest` to another suite's file. Root observed exactly this: `cannot import name 'FROZEN_M24' from 'conftest' (.../tests/conservative_pits/conftest.py)`, in `cw_e5_independent_cpu.log`. This is a collection-harness collision, not a runtime defect.
  - The B46 oracle harness remains in `outputs/conservative_water/cw_b46_runtime_frozen/tests/cw_b46`; it assumes that private location and is not copied into the root test tree. Run it from that immutable snapshot with M43, as shown below.
  - **Different MAPLE bindings:** some families deliberately pin a MAPLE other than M44. Those gates are part of the tests and must not be weakened; use the binding each one requires.

**Current (CW-E47) scoped families**, run from the SYRUP root with the current binding, each in its **own** invocation:

```bash
PY=/home/okin/MAPLE/.venv/bin/python
C=/home/okin/SYRUP/outputs/dependencies/content_cache/7ee10bd81b0859a891ff8a476cab2c8824bb0964309695ca948d12d5ac4b0c71
cd /home/okin/SYRUP
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=$C/editable:$C/source/src:$PWD/src MAPLE_SYRUP_EXPECTED_MAPLE_ROOT=$C/source
# B108 / B109 assert the actually imported roots and refuse (usage error) without these:
CW_B108_EXPECT_MAPLE=$C/source/src/maple CW_B108_EXPECT_SYRUP=$PWD/src/maple_syrup $PY -m pytest -c pyproject.toml -p no:cacheprovider -q -rs tests/cw_b108
CW_B108_EXPECT_MAPLE=$C/source/src/maple CW_B108_EXPECT_SYRUP=$PWD/src/maple_syrup $PY -m pytest -c pyproject.toml -p no:cacheprovider -q -rs tests/cw_b109
$PY -m pytest -c pyproject.toml -p no:cacheprovider -q -rs tests/cw_b104
MAPLE_WATER_RECIPE_REFERENCE=$C/source $PY -m pytest -c pyproject.toml -p no:cacheprovider -q -rs tests/test_reconstruct_water_maple.py
```

GPU parametrizations skip on a CPU-hidden run ("no claim"); root runs the GPU twins separately on a freshly idle GPU0 with the original pool ceiling.

**Default-suite contracts (CW-E47C1).** The CW-E47 adoption reproduced 12 CURRENT default-suite failures (`tests/phase7b` and the four conservative families): their tests still expected per-commit rerouting, the pre-fusion pickup objects and the pre-CW-B102 control field set. They were current contract failures, not historical guards, and CW-E47C1 corrects them against the actual runtime without any runtime, threshold or tolerance change; see [current test contracts](conservative_water/current_test_contracts.md). Run each default family in its own invocation (`tests/phase7b`, `tests/conservative_water`, `tests/conservative_csr`, `tests/conservative_csr_event`, `tests/conservative_pits`).

**Off-default historical static guards (not corrected, recorded).** `tests/cw_b54` (4 static AST / diff guards against the B52 CUDA body) and `tests/cw_b20` (4 cases: B15 driver identity and an intermediate-lifetime case) keep guards pinned to earlier frozen runtimes. They fail identically on the accepted frozen 5f tree, are recorded historical checks and are neither edited nor waived (`agent_handoffs/tasks/conservative_water/cw_e47_report.md`).

**Historical (B55/M44-era) families and bindings.** These commands belong to the PREVIOUS frozen B55 runtime with the M44 MAPLE; they are **not runnable against the current root tree** (5f source, 24c recipe). They are therefore pinned to the immutable B55 snapshot `outputs/conservative_water/cw_b55_joint_runtime_frozen` (whose `tests/cw_e2`, `cw_e5`, `cw_f2` and `cw_b51` resolve their tree relative to their own location) and to the frozen MAPLE each family names. CW-E47C1 did not execute them.

```bash
PY=/home/okin/MAPLE/.venv/bin/python
SY=/home/okin/SYRUP/outputs/conservative_water/cw_b55_joint_runtime_frozen   # the historical B55 tree
pin() { export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$1/editable:$1/source/src:$SY/src MAPLE_SYRUP_EXPECTED_MAPLE_ROOT=$1/source; }
D=/home/okin/SYRUP/outputs/dependencies
ROOTSY=/home/okin/SYRUP
B55=$SY
cd "$SY"

# historical M44 environment: delivery/metadata families, one invocation each
pin $D/maple_cw_b44_lifetime_frozen
CUDA_VISIBLE_DEVICES='' $PY -m pytest -c pyproject.toml -p no:cacheprovider -q tests/cw_e2   # CW-E2 dependency bundle
CUDA_VISIBLE_DEVICES='' $PY -m pytest -c pyproject.toml -p no:cacheprovider -q tests/cw_e5   # CW-E5 metadata checks

# B46 empty transport: its conftest requires the original private location and frozen M43 (deliberate pins)
SY=$ROOTSY/outputs/conservative_water/cw_b46_runtime_frozen
cd "$SY"
pin $D/maple_cw_b43_owned_frozen
CUDA_VISIBLE_DEVICES='' $PY -m pytest -c pyproject.toml -p no:cacheprovider -q tests/cw_b46

# F2 restart byte bound: test_resolved_module_paths asserts the frozen M29 MAPLE
SY=$B55
cd "$SY"
pin $D/maple_cw_b29_ledger_frozen
CUDA_VISIBLE_DEVICES='' $PY -m pytest -c pyproject.toml -p no:cacheprovider -q tests/cw_f2
```

  - Under the M44 environment, `tests/cw_b46` refuses at conftest import: `tests/cw_b46 must run on the frozen M43 MAPLE ...`. This is its intended guard, not a failure of the runtime.
  - The `cw_f2` binding is read from its source (`FROZEN_M29`). This document did not execute it.

**B51 flow-end settlement and the historical E9 recipe** (historical M44-era instructions; the current recipe test uses the 24c `MAPLE_WATER_RECIPE_REFERENCE` shown above). Run B51 in its **own** invocation from the B55 snapshot with the M44 binding (`pin` above). `tests/cw_b51` has a bare `conftest.py` and reuses helper modules of `tests/phase5`, `tests/phase6`, `tests/conservative_water` and `tests/conservative_pits` by import, so never combine it with those suites in one session.

```bash
cd "$B55"
SY=$B55
pin $D/maple_cw_b44_lifetime_frozen
# B51: CPU tests run; the cupy parametrizations skip without a device ("no claim").
# The two false-path differentials need the pinned B46 complete_event.py: found at
# outputs/conservative_water/cw_b46_runtime_frozen (or set MAPLE_SYRUP_B46_REFERENCE=<B46 snapshot root>);
# absent -> explicit skip; wrong SHA-256, or an explicitly set but missing reference -> FAIL.
CUDA_VISIBLE_DEVICES='' $PY -m pytest -c pyproject.toml -p no:cacheprovider -q -rs tests/cw_b51

```

*Historical E9 recipe record (not a runnable current command).* The E9 test ran
`MAPLE_WATER_RECIPE_REFERENCE=$D/maple_cw_b44_lifetime_frozen/source pytest tests/test_reconstruct_water_maple.py` against the M44
recipe, tool and test that CW-E47 replaced. Those exact bytes are archived in
`agent_handoffs/tasks/conservative_water/cw_e47_baseline_archive/files/` (`dependencies/maple_water/`, `tools/`, `tests/`);
the B55 snapshot has no `tools/` / `dependencies/`, so the root-tree test now verifies only the current 24c recipe (command above).
