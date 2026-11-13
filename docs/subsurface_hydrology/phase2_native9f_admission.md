# Phase 2 E4k — exact native 9f storage-admission qualification

`maple_syrup.storage_admission` (explicit driver-call helper; unchanged public semantics: NOT automatic, NOT a whole-process
bound, reserves REQUIRED and never invented) was bound to the merged MAPLE `39bcf79e` package only. It now accepts exactly two
packages through `ADMISSION_QUALIFIED_MAPLE_DIGESTS = {MERGED_39BCF79_MAPLE_DIGEST, NATIVE9F_MAPLE_DIGEST}` (the E4j pin
`fbc504bb…`, MAPLE `9f466d2c`). The check is exact set membership, still BEFORE the optional native planner is imported; the
39bc entry and message wording are unchanged. Nothing else changed in the module (16 diff lines: import, set, guard, docstring).

## Source audit (read-only `git diff`/`git show` of /home/okin/MAPLE, 39bcf79e -> 9f466d2c; no repository write)
| module | result |
|---|---|
| `surface/voxels/windowed_admission.py` (`windowed_transaction_bytes`, `admit_windowed_transaction`, `WindowedStepPlan`, `AdmissionDecision`) | FILE-identical, sha256 `d2c87b86…` (also the runtime file) |
| `surface/voxels/deep_store.py` (header `chunk_bytes_max`, `mapping_cache_chunks`, `format_version`, `n_chunks`, `slot_files`, journal cap default) | FILE-identical, sha256 `b03d1503…` (also the runtime file) |
| `core/backend/__init__.py` | identical |
| `surface/voxels/windowed_transfer.py` (`DEFAULT_EXCHANGE_GATHER_COLUMNS = 256`) | changed (+88/-20): private default-`None` taps (`_evidence`, `_avalanche_observation`, `_depth_work`) and `_apply_pending_segments` extraction; no new array on a default path; gather default still 256 |
| `surface/voxels/windowed_inventory.py` | `_check_windowed_column_layout` split; not read by the planner |

The planner reads only column/store attributes, the ledger process count and the context caps; those are unchanged. The
`_native_transaction` / native-avalanche staging in `topographic_commit/windowed_commit.py` is NOT part of the planned public/paired
transactions and is not covered by this qualification (avalanche stays out of scope).

## Memory contract (unchanged)
Plan = MAPLE's `WindowedStepPlan` of the actual bed and context: new native host/device bytes of ONE call (max over stages) plus caller
reserves versus injected or current-device readings under a safety fraction (native 0.5). Excludes publication buffers, commit,
external transport beyond the paired callback's arrays, pool fragmentation, JIT/library workspace, Python overhead. Only
`max_gather_columns` None/256 is accepted (adapters pass no other); journal cap is `resolved_journal_cap(context)`.

## Evidence (CPU, MAPLE `.venv`, pinned 9f; tests/phase2_native9f_admission, 11 passed + 3 gated CUDA definitions)
guard pair exactness (off-by-one/case/prefix refused; 39bc accepted); module hashes; SYRUP plan == direct native planner call
built from the native objects (public and paired, flags, no device bytes); paired >= public and flags enlarge; non-default gather
refused, context journal cap forwarded; admission precedes and never mutates (store files, arrays), exact threshold, `MemoryError`
class, reserve never defaulted; unknown source refused before native planner import/readings; host bed refuses device inputs, never
imports CuPy; plan bounds the traced NumPy allocations of the real native public and paired exchange (small fixture: 44 kB vs 2.97 MB,
792 kB vs 3.21 MB). Eight of the eleven tests fail on the pre-edit module (sensitivity). Historical `tests/shared_storage_compat`: 50 failed/46 passed
before the pin, 1 failed/95 passed after; the one failure, `test_merged_digest_is_an_explicit_pin…`, asserts the RUNTIME digest equals 39bc
(historical, correct to fail on 9f; not rewritten).

## Not claimed
No GPU run (definitions only), no automatic protection, no whole-process or Chastre-scale bound, no non-default gather, no
avalanche/native-transaction coverage, no coupled Pond/daily/restart acceptance.

Root independently accepted this bounded optional helper after 106 CPU checks, three actual CUDA checks and real native public/paired GPU transaction payload controls. See phase2_native9f_admission_acceptance.json; the historical39bc-only assertion remains unaltered and is not a native9f acceptance test.
