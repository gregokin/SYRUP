# Every-N-completed-days soil-moisture diagnostic output (G0)

Status (2026-11-13): independently accepted bounded native component, uncommitted, NOT wired to any CLI, scheduler or MASY driver. Root passed 348 CPU checks and 2 actual-GPU checks; one two-device check was skipped because only authorized GPU0 was visible. NumPy and Numba fresh-process physical restart/replay checks also passed. Exact sources, evidence and limits are in [the acceptance record](soil_moisture_output_acceptance.json).
This is the native writer/cadence capability; it is not a restart checkpoint and changes no physics, bed, history, ledger, cursor or source guard.
Module: `src/maple_syrup/soil_moisture_output.py`; tests `tests/phase2_moisture_output/`. The full contract is in the module docstring.

## API

* `MoistureOutputLimits(max_snapshots, max_total_bytes, max_manifest_bytes, chunk_bytes, host_memory_budget_bytes, host_reserve_bytes, disk_reserve_bytes)` — every field required (correction01 added the explicit host staging budget).
* `MoistureOutput.create(directory, lop, origin_state, *, expected_parent, every_n_days, limits, origin_date=None, caller_identity=None, host_available_bytes_fn=None, disk_available_bytes_fn=None)` — explicit opt-in. `every_n_days` is a positive integer (no default; bool, 0, floats, NaN refused). The origin is a genuine closed daily boundary (day 0, nothing saved). Needs an EMPTY directory.
* `MoistureOutput.resume(directory, lop, state, *, expected_parent, limits, caller_identity=None, host_available_bytes_fn=None)` — re-bind after a restart to the (possibly lagging) physical boundary; verifies every due day <= the state's day is sealed; removes crashed unsealed reservations; keeps snapshots ahead of the state and VERIFIES them on replay.
* `out.notify_day_closed(lop, state, *, expected_parent, host_available_bytes_fn=None, disk_available_bytes_fn=None) -> MoistureNotice(day, ordinal, due, action, path, bytes_written, device_chunks)`; actions `not_due | saved | verified_existing | repeat_same_state`.
* `read_snapshot(day_dir, *, max_bytes) -> MoistureSnapshot` (`water_m`, `active`, `theta()`, `volume_m3()`, `manifest`, `static`), `read_output_record(root)`, `out.verify_all()`.

## Completed day, from the native cursor

A day is final only at the closed `boundary` after `close_interevent` (after any accepted `request_withdrawal`), of a daily lineage with `pending is None`, whose last phase receipt is the `interevent` close, bound to the lifecycle operator
and its exact-instance `ExpectedParent`. Days since origin `d = cursor.ordinal - origin_ordinal`; a storm day is one day whatever its physical length; `daily_water_action`/`finish_daily_storm` results (role `interevent`) are refused. Notification must be sequential
(1, 2, 3, ... including days not due); skips and rewinds are refused; a repeat of the same state instance is idempotent; another instance for the same day conflicts. Call it BEFORE persisting that day's physical checkpoint.

## Files and bytes

`output.json` (sealed, body SHA-256), `active.npy`, `day_<12 digits>/water_m.npy` (genuine `.npy` v1.0, `<f8`, `(L, N_cells)` C order, `m` water-equivalent depth) and `manifest.json` (strict canonical JSON, body SHA-256; day, ordinal, cadence,
`nominal_elapsed_s = d * 86400`, physical `t_s`, optional ISO date, array record, static SHA, run/operator/state/cursor/receipt identities, module SHA). Profile = `fixed_full_porous_common_column` (dz and saturation of the common column; NO floating
porous-depth array: a later floating owner must add it under a new schema). On-disk bytes per snapshot = `len(npy header)` (128) `+ 8 L N + len(manifest)` (about 1.7 KB); static = `output.json + N` bytes. Peak host staging = ONE `chunk_bytes` piece
(`+ max_manifest_bytes + 64 KiB` modelled and admitted); a CPU array is written from views (no copy), a device array downloads one counted piece per call, nothing on a not-due day.

## Publication, quotas, failure

Exclusive `mkdir` reservation, `O_EXCL|O_NOFOLLOW` files relative to descriptors, `fsync`, manifest-LAST `os.link` seal (no-clobber; the `layered_checkpoint` pattern; readers refuse an unsealed day). Count, total-byte, manifest, host-availability and disk quotas
(and reserves) are checked BEFORE any transfer or write. Any failure removes only its own unsealed reservation and leaves the state, cadence and sealed snapshots unchanged; the same day retries. The cadence is a bounded scalar; on disk it is derived
from the sealed snapshots. Corrupt, extra, symlinked or mismatched entries refuse.

## Verified (author, CPU, native 9f, NumPy and Numba columns, `CUDA_VISIBLE_DEVICES=''`)

`pytest -c tests/phase2_moisture_output/pytest.ini tests/phase2_moisture_output`: 106 passed, 2 skipped (gated GPU definitions). Genuine native Dense lifecycle (daily action, withdrawal, close, storm day); N = 1/2/3; invalid N; before-close refusal; post-withdrawal capture;
physical duration adds no days; dates; sequence/branch/foreign-lineage/operator refusals; exact `.npy` reconstruction; inactive entries preserved; tamper/symlink/untrusted-JSON refusals; chunk-size-invariant bytes; quotas before transfers; failure injection at stream/seal/manifest;
no-clobber, lagging resume, verified replay, conflict, crashed reservation, swapped directory; zero transfers when not due and bounded pieces (CuPy-free double); the device gate runs before any reader.

## Remaining integration (not done here)

1. Qualify a real final-day acceptance hook and its configuration. The historical MASY C1/LEAF driver does not create a native `DailyCursor`, so it cannot call this API unchanged. Use an explicit new daily profile without changing historical clocks or source locks. Saves count completed nominal daily intervals after plant withdrawals; extra physical wind/storm duration does not advance this count.
2. Floating/porous-depth extension: a new snapshot schema with the actual porous-depth array for the floating owner.
3. Bind the actual qualified runtime/source pair through sealed `caller_identity`; the native writer alone does not establish that external pair.

## Independent qualification

The accepted writer SHA-256 is `48919d4346ad9c248d7569059e22e10cbe5a6db3a2c8c3c5eb7e6ee9d9273610`, against MAPLE `9f466d2cfcd52991b24496cf67f4562d6268afa9` (package `fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e`). Root independently checked saved-array corruption, failed publication and retry, exact quotas, bounded metadata staging, preservation of unknown files and recovery of the writer's own sealed temporary. A genuine physical checkpoint on day 1 loaded in a new process continued days 2–4 with the same physical/state digests and water bytes as the uninterrupted NumPy and Numba controls; due snapshots on days 2 and 4 were verified rather than overwritten. GPU tests used bounded counted transfers and made no water download on non-due days. This is component qualification, not acceptance of the pending floating-layer driver or a large coupled simulation.

The author-verification and correction notes below record earlier stages. References there to CUDA being unrun describe those earlier stages; the independent results above and acceptance record supersede that status.

## Correction01 (2026-11-13; after Root's independent probes; output engine only, schemas now `.../2`)

Old G0 (schema `/1`) evidence is unqualified and preserved; there is no migration, and `resume` refuses an output written by a different writer module.

1. **Saved-array proof.** `resume` streams EVERY sealed `water_m.npy` (and the mask) against its manifest through one `chunk_bytes` buffer (header, size, regular file, SHA-256): a declared COLD read of all sealed bytes. `notify_day_closed` re-proves the stored file before `verified_existing` and then compares it with the live water. An equal-length corruption is refused.
2. **Publication states.** Before the seal: failures remove only the writer's own unsealed files. At/after the seal nothing is ever removed: a failure raises `MoistureOutputIndeterminate` ("published, durability uncertain": payload and manifest intact, cadence NOT advanced, `path` given); the retry (or `resume`) verifies the sealed identity and advances once. A leftover `manifest.json.tmp` that is a hard link of the sealed manifest is the only thing tidied.
3. **Create/staging.** The mask is streamed in `chunk_bytes` pieces; count, total-byte, host budget/reserve/reading and disk are admitted BEFORE any download or write; the streaming loop holds exactly one host piece (tracemalloc-tested). The mask download on CUDA is a separate cold transfer; unsaved days make zero.
4. **Reader.** `read_snapshot(max_bytes)` caps all returned arrays (water + mask + dz) from the validated record BEFORE any array is read, reading straight into final buffers. Strict JSON refuses duplicate keys, NaN/Infinity, `1e999`, >40-digit integers, deep nesting and non-ASCII; `output.json` and every manifest are validated for exact keys, types, ranges and checksums (typed `MoistureOutputError`).
5. **Restart.** After `resume`, the SAME closed state is idempotent (a due day is re-proved against the sealed bytes, a non-due day repeats the accepted instance); any other branch conflicts. A fresh-process test saves a genuine V3 daily physical checkpoint, loads it in a new process, resumes the output and continues to bytes equal to the uninterrupted run (CPU, the S4 fixtures' codec; the native CLI hook is still later).
6. **No directory enumeration on the hot path** (descriptor-relative no-follow `stat` for existence); the resume audit is a streaming `scandir` with scalar counters, O(1) memory and O(snapshot count) time. Source identity: each snapshot records the closing receipt's `phase_source_digest` (per-phase, changes daily: informational only, not a lineage binding) and the writer module sha; the runtime source PAIR is not hashed by G0: the adapter supplies it as the sealed `caller_identity`, compared on `resume`.

Limits: single writer; a normal-API caller owns the `MoistureOutput` object; restart from a NON-due checkpoint day cannot be proved from disk (only due days are saved), it relies on the caller's state; CUDA behavior remains unrun (gated definitions).

## Correction02 (2026-11-13; output engine only; schema `/2` is still unpublished)

1. **Exactly-once inventory across indeterminate publication.** A snapshot that is sealed but whose notification could not complete is credited to the writer's `saved`/`used` at once (a bounded scalar `_pending_day` receipt) while `last_day` stays pending. The retry (`verified_existing`) and a fresh `resume` scan do not credit it again (resume counts every sealed snapshot once, including ones ahead of the restart state). The next day therefore hits `max_snapshots`/`max_total_bytes` BEFORE any download or write (Root's probe: previously the next day wrote a second file). Verified for root-directory fsync failure, any directory fsync failure, manifest-temp unlink failure and death right after the link, with repeated retries.
2. **Metadata model.** `MoistureOutputLimits.staging_bytes(record_json_bytes) = chunk_bytes + 6 * (max_manifest_bytes + record_json_bytes) + 64 KiB + host_reserve_bytes`. `create` bounds the record from the known layer count (`3072 + len(caller_identity) + 50 L` bytes) and admits it BEFORE building the O(L) lists/JSON (a large-L refusal builds nothing); `resume`/`verify_all` `fstat` the record (<= 1 MiB) first. Measured warm (tracemalloc, planned vs traced): create L=2 133,784 vs 11,131 B; L=128 171,584 vs 41,259; L=1024 440,384 vs 272,089; resume plans 122,174 / 154,622 / 381,986 vs 67,263 / 72,671 / 160,225. The few-layer overhead stays at the ~130 KB of the 64 KiB allowance plus the manifest scratch. Lazy first imports are excluded (not O(L) metadata). `read_snapshot(max_bytes)` caps only the returned arrays; its extra metadata scratch is `<= 6 * record size + 64 KiB` (about 6 MiB at the 1 MiB record ceiling), declared and outside `max_bytes`. The writer keeps the parsed record resident (about 64 B per layer).
3. **Bounded finiteness check** of loaded water in 65,536-element blocks (no `(L, N)` boolean; a 16 MB array's load peak stays within the array + 300 KB).
4. **Dates and day names.** `origin_date + day` beyond year 9999 and a day above `MAX_DAY = 10**12 - 1` (the 12-digit directory name; a representational cap, not physics) are typed refusals before any I/O on every notified day; resealed invalid day/date fields raise only `MoistureOutputError`.

Limits unchanged: single writer; no CLI/driver wiring (the actual C1 driver does not start a `DailyCursor`, so the adapter is later work); a restart from a non-due checkpoint day relies on the caller's state; CUDA unrun.

## Correction03 (2026-11-13; cold scan/recovery only)

1. An unsealed `day_*` directory is cleaned only if EVERY member is one of this writer's two expected REGULAR names (`water_m.npy`, `manifest.json.tmp`; an empty reservation also qualifies). Any unknown, symlinked or non-regular member proves the directory is not this writer's reservation: `resume`/`verify_all` refuse with a typed error and remove nothing (not even a known-named file). Verified byte for byte.
2. `output.json.tmp` (a creation interrupted between the link and the unlink, or a failed final fsync) is tidied ONLY when it is the same regular file (device + inode) as the validated sealed `output.json`; any other regular file, symlink or special file is refused and preserved. The record's module, caller-identity, lifecycle and lineage bindings are verified before the cold scan, so a refused resume tidies nothing. Counts and bytes after recovery equal the real file sizes, exactly once.
