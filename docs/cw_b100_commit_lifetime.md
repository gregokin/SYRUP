# CW-B100 — private event-loop lifetime at the topographic commit (candidate; not adopted)

## Defect (actual R6 record)

**Where it failed.** The R6 rotated-Chastre GPU reference (R1 source, M94 MAPLE) stopped after its 237th candidate
step was constructed, inside the commit:

- `sediment_event.py:2432` → `commit_bed` → M94 `windowed_commit.py:588` → `zeros_sediment_ledger_state`
  (`sediment_ledger.py:896`, the replacement ledger's `n_physical_touches`).

**How much it was over.** It requested 75,700,224 B, which is one 886 × 1780 × 6 eight-byte C-grid at 512-byte
granularity. 9,018,241,536 B were already allocated under the unchanged 9,092,045,209 B pool limit, i.e. 1,896,551 B
over.

**Why.** In `evolve_sediment_event` the loop rebound its local `state` only AFTER `commit_and_reroute`. So the PREVIOUS
accepted state was still alive while MAPLE built the commit, next to the validated attempt bed. That previous state is
the bed (ledger C-grids, resident window, active layer, water, availability, prefix packets), the sediment velocity
C-grid, the CSR phase and the storm.

Nothing reads those objects after the attempt's retry loop:
- the commit reads only `state.graph` and `state.terrain`;
- `publish_commit` replaces bed / graph / network / grid / storm / phase;
- a commit or reroute failure propagates out of the function, which publishes and returns nothing.

**Callers do not pin them either:**
- `complete_event` and `run_complete_case` hand the start state over through single-use holders (CW-B30);
- the quiet monitor keeps floats only;
- the checkpoint callback keeps paths only;
- the R1 / F9 marker observer keeps values only.

## Change (`src/maple_syrup/sediment_event.py`, `evolve_sediment_event` only)

After the attempt has passed every retry and check, the private local is rebound BEFORE the commit:

```
state = replace(state, storm=storm_state, bed=attempt.bed, sediment_velocity_m_s=..., phase=...)
```

These are exactly the fields step 7 published before, with the still-unpublished solver bed. After the commit /
deep-storage publication, only the bed is replaced (`accepted_bed`). The commit receives the same `storm_state` object
the state carries, with values equal to the former separate `replace`.

**What did not change:**
- no array is copied, recomputed or mutated, and no caller object is touched;
- no MAPLE change, no cap, tolerance, bin, class, domain, forcing or commit-frequency change;
- no `gc` call;
- the public `sediment_coupled_step` and the private `_sediment_accounting_step` are byte-identical;
- the AST diff against R1 is `evolve_sediment_event` only, and every other package file is byte-identical.

## Verification (see agent_handoffs/tasks/conservative_water/cw_b100_report.md for the exact records)

**Fixture.** `tests/cw_b100/b100_driver.py` runs an actual MAPLE resident-window bed (deep store in tmp, W4,
interval commits every 2 s) on a 64 × 32 valley. It uses the CSR characteristic transport with history OFF, for 12 s,
which gives 6 commits.

**Tests.** `tests/cw_b100/test_cw_b100_commit_lifetime.py` runs the CPU cases, R1 tree vs this tree. The task
orchestrator also runs the GPU0 cases.

**What they establish:**
- weak references prove that R1 keeps all 9 tracked previous-state arrays alive at every fresh-ledger allocation,
  while B100 keeps none;
- the allocator's in-use bytes at that allocation are deterministically lower: GPU by 2.68–3.15 MB, i.e. 27–32 C-grid
  equivalents of this fixture; CPU (tracemalloc) likewise;
- results are raw-exact against R1 (all non-timing entries) and the deep-store files are byte-identical;
- the caller's inputs and earlier deep files are preserved on success and on injected commit / reroute failures.

## Limits

- **Small fixture only.** The Chastre-scale saving is not measured. Structurally it releases one previous bed, one
  velocity C-grid, one CSR phase and one storm per commit.
- **No full-GPU-fit claim.** Whether the R6 step-237 commit, and every later step, then fits under the unchanged pool
  limit is unverified and requires root's frozen large retry.
- **Paired-limit behaviour is stochastic in the tiny fixture.** At the margin the binding allocation is the transport
  thrust argsort scratch, not the commit, and pass / fail at a fixed CuPy pool limit is not reproducible run to run.

## Deep-store follow-up (documentation only; no GC / prune / cap change here)

The M94 deep store is append-only. `apply_journal` (`deep_store.py:981-1059`) publishes a NEW version for every
non-empty journal. For each touched chunk it:

- copies the whole chunk (256 columns × 306 voxels × 6 classes × 8 B = 3,760,128 B) twice: a validate pass, then a seal
  pass;
- SHA-256s it;
- writes and fsyncs it;
- re-reads it to hash it (`_write_bytes_once`, `:507-529`).

It also scans the whole store directory to enforce `max_store_bytes` (`_scan_store_bytes`, `:484-504`). Nothing is
ever removed; collection is explicitly deferred.

**Measured on the R1 / R6 GPU store `cw_r1_rotated_chastre_W3_initial` after the R6 reference stopped (t ≈ 237 s):**

| Measure | Value |
|---|---|
| Version manifests | 47 |
| Chunk files | 30,252 |
| Total size | 114,844,185,205 B |
| Store cap (`max_store_bytes`) | 274,877,906,944 B |
| Free on the volume | 626,959,589,376 B |
| Generation-0 unique chunks | 5,346 ≈ 20.10 GB |

So about 24,906 new chunk files, about 93.6 GB, were written for 46 published versions: about 541 whole chunks, about
2.0 GB written plus re-read, per version.

**Linear extrapolation from these measured operands only (not a prediction of the storm):** the cap would be reached
after roughly 79 further versions of the same size ((274.88 − 114.84) GB ÷ 2.036 GB per version). A full storm on this store is therefore at risk of an
`apply_journal` cap refusal ("Nothing was written") and is dominated by per-step whole-chunk I/O. Any change to version
retention / collection, chunk size or the journal publication cadence needs its own reviewed task that preserves the
INITIAL, checkpoint and restart contracts.
