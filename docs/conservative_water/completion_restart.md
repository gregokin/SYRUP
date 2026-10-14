# CW-C1 — complete events and same-backend restart on CuPy (small actual-MAPLE beds)

Historical task record; pending statements below apply to that task snapshot. See [the current implementation status](implementation.md) for executed results and remaining gates.

Status: implemented by the sole writer of task CW-C1 (file-only tools; nothing here was executed by its author).
Executed results, acceptance and corrections are recorded by the root under
`agent_handoffs/tasks/conservative_water/`. Builds on the accepted CPU completion / checkpoint layer (Phase 6) and
the CW-A1 CUDA coupling (`backend_integration.md`); the A1 source, the event numerics and the storage kernels are
untouched.

## What changed

| File | Change |
|---|---|
| `src/maple_syrup/event_backend.py` (new) | `host_packet` (one counted host read of a few 0-d reductions), `arrays_equal` (counted equality, mixed pairs refused), `backend_name`, and the explicit reconstruction `item_to_namespace` / `result_to_namespace` / `state_to_namespace` (physical arrays placed; graph, network, grid and phase REBUILT with their `xp`; host objects untouched; identity on NumPy) |
| `src/maple_syrup/complete_event.py` | NumPy refusals removed; `quiet_metrics` and `water_budget` compute in the state's namespace and read ONE compact host packet each; `finalize_event` keeps every physical array in the namespace through the genuine MAPLE terminal deposition, the forced final commit and the dry handoff, with the per-class terminal inventories read once to the host for the MAPLE conservation audit; `complete_event(..., hydraulics=None)` hands a CUDA lifecycle to the solver |
| `src/maple_syrup/checkpoint.py` | host bundle format kept; every device array encoded through an explicit counted `to_host`; `_graph_digest` hashes array BYTES for either namespace (host digest unchanged); manifest `backend` tag (absent = pre-CW-C1 NumPy); `load_checkpoint` decodes to host, checks the MAPLE snapshot against the companion on the host, reconstructs the continuation in the namespace of the caller's column / sediment parameters, validates THERE; cross-backend loads refused; `_validate_accumulators` and the handoff comparisons are namespace-aware (MAPLE counted predicates) |
| `src/maple_syrup/event_experiment.py` | `backend` (numpy / cupy), `implementation` (numba / array / cuda), `root_solver`, `cuda_mode`, `allow_maple_source_change`; the CW-A1 pairing refusals before the case is touched; backend resolution and the CW-A1 memory preflight BEFORE the case is placed on the device; `CudaHydraulics` lifecycle passed to the solver; synchronized loop timing, transfer counters and pool report; host-converted outputs; truthful `backend` block |
| `tests/conservative_water/test_complete_backend.py`, `test_checkpoint_backend.py` (new) | CPU regression and the GPU tests listed below |

## Contracts

* **Namespaces.** Physical and accumulator arrays live in `state.graph.xp` from the first step through the dry
  handoff. Host by contract: `TerrainReference`, the planned and reporting boundaries, the graph's evidence arrays
  (aspect, slope, active, outlet, receiver, friction), the integer counters, the commit / rejection logs, the
  accounting dictionaries, the parameters' host diameter copies. The only host reads of the completion layer are
  the two compact packets (completion criteria, water budget), the per-class terminal inventories of the
  conservation audit, the reported accounting scalars and the checkpoint transfers.
* **Thresholds and policy unchanged.** `CompletionPolicy`, the hold, the rain-finished rule, the MAPLE
  mass-resolution significance threshold, the MAPLE summation bounds, the request / closure checks and the
  terminal conservation bound are the accepted CPU ones. The NumPy path computes exactly the reductions it computed
  before; the packet batches the reads and adds no arithmetic.
* **Terminal transaction.** Nonzero remaining mobile mass is deposited by a genuine `apply_water_process_demand`
  call (refused if MAPLE leaves anything mobile); the final commit is a genuine `commit_topography` through
  `_commit_and_reroute`; the ledger must be empty afterwards; a terminal reroute that reactivates flow refuses the
  handoff; every failure path leaves the wet input unchanged (pure functions, frozen states). The dry reset is an
  external removal of the pre-reset surface and soil water, both reported and both kept as arrays on the handoff.
* **CUDA lifecycle.** `complete_event` passes the caller's `CudaHydraulics` to `evolve_sediment_event`
  (CW-A1 rules: one context per graph binding, rebuilt before each loop commit is published). The terminal forced
  commit in `finalize_event` does not rebuild the context: a continuation's next water step re-prepares it, counted,
  from the committed graph; nothing irreconstructible is discarded.
* **Checkpoint.** Schema `maple-syrup-checkpoint/2` unchanged (the `backend` tag is manifest metadata; no new
  physical state is persisted). The bundle is host-only on disk; it binds the exact source / config / case through
  the caller's identity, every bed / availability / ledger / terrain array, soil water, discharge, the rain clock
  (planned boundaries and reporting position), the velocity memory, the phase partition, every accumulator and the
  quiet monitor. Loading validates before anything usable is returned: malformed payloads, hash mismatches,
  unsupported or foreign backend tags, identity / config mismatches and completed handoffs (without the explicit
  flag) are refused. Same-backend resume reproduces physical arrays, counters and the final handoff bitwise;
  setup / timing counters are not persisted. Cross-backend migration is refused in both directions (not needed, not
  qualified). Strict source identity: no migration of older sources / configs.
* **Runner.** `backend='numpy'` + `array`/`numba` (default unchanged) or `backend='cupy'` + `cuda` only; bisection
  or Newton; optional CUDA mode; preflight before device placement; same output protection, new destination,
  atomic publish, non-finite-free JSON, source guards, checkpoint cadence on the original time partition. The legacy
  benchmark CLI default is untouched.

## Tests (author-unexecuted)

CPU: completion reductions equal the direct host expressions; water budget equals its direct computation; CUDA
control on a NumPy state and a lifecycle with an array control refused; manifest `backend: numpy`; legacy manifest
without the tag loads; foreign / unknown tag and extra key refused; host graph digest equals the pre-CW-C1
algorithm; namespace reconstruction is the identity on NumPy; mixed parameter namespaces refused. The existing
`tests/phase6` suites are the CPU regression.

GPU (skip without a device): quiescent storm and wet storm (bisection, Newton) complete with a dry handoff on the
device and match the host array event at the declared bounds (and the numba host event when Numba is present);
nonzero terminal settlement is a genuine MAPLE deposition; failed terminal deposit / commit / reactivated flow
leaves the wet input unchanged; duration and step limits are not handoffs; lifecycle preparations equal one plus the
loop commits; wet pause with pending ledger and non-trivial phase / mobile saved from and resumed on the device is
bitwise with the uninterrupted event on every science array; completed handoff round-trips and cannot replay;
cross-backend loads refused both ways; corrupted hash / tag / payload refused; device and host graph digests agree.

```bash
cd /home/okin/SYRUP
export PATH=/home/okin/MAPLE/.venv/bin:$PATH && source benchmarks/phase7e/candidate_env.sh
ruff check src/maple_syrup/complete_event.py src/maple_syrup/checkpoint.py src/maple_syrup/event_experiment.py \
    src/maple_syrup/event_backend.py tests/conservative_water
python -m pytest tests/phase6 tests/conservative_water -q
export PATH=/home/okin/MAPLE/.venv-cupy/bin:$PATH && source benchmarks/phase7e/candidate_env.sh
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/conservative_water -q
CUDA_VISIBLE_DEVICES=0 python -m maple_syrup.event_experiment --case-dir outputs/plot1 \
    --output-dir outputs/conservative_water/plot1_complete_cuda --backend cupy --implementation cuda \
    --checkpoint-every-s 600 --pause-at-s 1200
CUDA_VISIBLE_DEVICES=0 python -m maple_syrup.event_experiment --case-dir outputs/plot1 \
    --output-dir outputs/conservative_water/plot1_complete_cuda_resumed --backend cupy --implementation cuda \
    --resume outputs/conservative_water/plot1_complete_cuda_checkpoints/step_000001200
```

## Historical first-run findings (correction 2)

* `item_to_namespace` now returns the very object for a supported item already in the target namespace
  (previously it rebuilt the wrapper around identity-converted parts); the namespace validation and the
  unsupported-type refusal precede both paths.
* The root's first actual GPU 0 completion run (interrupted by device availability after one failure and four
  passes) reported `n_graph_changes` host 0 versus device 1 for the quiescent complete event, i.e. the device's
  terminal forced commit flagged a slope / aspect change where the host did not, with no bed exchange in that
  storm. The counter assertion is kept as written, no tolerance was introduced and no A1 / shared numerics were
  changed in this correction; the attribution (a separate read-only CW-A2 review of the long-storm CPU/GPU
  divergence is in progress) and the corrective assignment are the root's next bounded steps. The full CPU and GPU
  storms each closed their own budgets but their physical comparison failed; no qualification is claimed.

## Not claimed

No Chastre-scale or windowed bed, no pit storage, no cross-backend resume, no wind handoff, splash, ecology or
ET, no speed claim, no native-surplus investigation. The Plot 1 device complete event, its repeats and the
checkpoint / resume of the full storm are the root's runs.

## Independent execution, 2026-10-08

After three bounded corrections, CPU Phase6 and new coupled/completion tests passed **56 tests**, with26 device skips. Actual GPU wet checkpoint/resume passed1test (10deselected) in20.71s: checked physical arrays, dtype/shape/raw bytes and accounting match uninterrupted continuation. Other previously unexecuted device checks passed13tests, with a derived-cache test defect corrected before the wet resume rerun. Ruff passed. A quiescent GPU event has a host/device graph-change-counter mismatch (0 versus1); this remains open. Full evolving fixed-duration GPU Plot1 conserves water/sediment, but fails declared long-event CPU/GPU physical parity. These tests do **not** accept the complete water component or full imported storm restart/performance. Large memory-bounded integration remains pending. Evidence is in `agent_handoffs/tasks/conservative_water/`.

## Full imported complete-event qualification in progress

The isolated ordered-height dependency resolves the quiet graph-counter discrepancy and keeps the full Plot1 routing controls identical on CPU and GPU. Both complete events finish at4739s/802commits with water and class balances closed and an explicitly accounted25.2200162282m³ soil-water reset. CPU repeat and fresh disk continuation from2400s reproduce all173 saved numeric arrays and final accounting byte-for-byte. The full GPU original comparison retains one tiny voxel-field failure at the original strict bounds; repeat/restart and MAPLE operand-scaled qualification are pending. Neither these small-domain runs nor the implemented checkpoint layer prove bounded large-domain integration.
