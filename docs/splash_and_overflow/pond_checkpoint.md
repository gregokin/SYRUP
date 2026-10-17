# Pond event checkpoint / continuation (`maple_syrup.pond_checkpoint`)

Status: implemented and independently exercised by Codex on CPU and GPU. See [qualification](qualification.md) and
[acceptance](acceptance.json) for exact source versions, results and limits; the component descriptions below are not whole-catchment qualification.

## What it is

An opt-in, **new-schema** save/load pair for the coupled pond event:

```python
from maple_syrup.pond_checkpoint import save_pond_checkpoint, load_pond_checkpoint

path = save_pond_checkpoint("run/ck_0600s", state_or_handoff, event, identity)
state = load_pond_checkpoint("run/ck_0600s", fresh_event, identity)                      # wet PondEventState
handoff = load_pond_checkpoint("run/ck_done", fresh_event, identity, allow_complete=True)  # PondDryHandoff (data)
```

* `state_or_handoff`: a **wet** `PondEventState` or a completed `PondDryHandoff`. A bare dry state is refused.
* `event`: the static `PondEvent` (context, column, sediment, schedule, **original** initial topology, control, policy,
  splash, vegetation, rain scale). On load the caller builds a **fresh** event from the ORIGINAL topology, not from the
  evolved one; the checkpoint proves it is the same event and carries the evolved source itself.
* `identity`: a caller dict (strict JSON; no NaN/Infinity). It is stored verbatim and must equal `expected_identity` on load.
* `max_bytes`: default `DEFAULT_MAX_CHECKPOINT_BYTES`, **the existing 1 GiB** of `maple_syrup.checkpoint`. It bounds the summed
  uncompressed bytes of both payload files, is checked from file sizes and zip directories before any array is expanded, and
  is never raised automatically. A bigger value is a caller-explicit resource decision. `save` applies the same bound to what
  it would write.

The accepted `maple_syrup.checkpoint` (`maple-syrup-checkpoint/2`, `checkpoint.json`) and all its defaults are untouched. The
pond format has its own schema `maple-syrup-pond-checkpoint/1` and manifest `pond_checkpoint.json`; the old decoder cannot read
it (there is no `checkpoint.json`) and this loader names and refuses an old-format directory.

## Layout

```
<path>/                          created as a hidden sibling staging dir, then renamed; never overwrites
  pond_checkpoint.json           hash-bound manifest
  maple_state.npz                actual MAPLE save_state_snapshot: voxel column, active layer, availability, water
  pond_continuation.npz          companion arrays (whitelisted names / dtypes / shapes)
```

Manifest keys (exact set): `schema, identity, status ("wet"|"complete"), array_backend ("numpy"|"cupy"),
hydraulic_backend ("numpy"|"numba"|"cuda"), source, binding, binding_sha256, files, payload, manifest_sha256`.
`manifest_sha256` covers all other keys (canonical JSON); `files` holds the SHA-256 of the two payload files.

### One bed authority

The dense bed (voxel column, active layer, availability, **water depth and mobile pool**) lives only in the native snapshot,
written by MAPLE's `save_state_snapshot(voxel_column, active_layer, time_s, path, step=..., reason=..., geometry=...,
mass_resolution_kg=..., sediment_availability=..., water=..., provenance=...)` and read by `load_state_snapshot`. The
snapshot's provenance carries the schema, the caller identity, the binding digest, status and backends; the loader refuses a
snapshot not bound to the manifest it sits next to, a snapshot that is not `validated`/`restartable`/non-diagnostic, one
that carries foreign optional state, and one whose clock/step differ from the companion. The companion does **not** repeat
the voxel/active inventory.

### Companion (everything else that continues the event)

| Group | Content |
|---|---|
| hydraulic | `depth`, `soil`, `qx`, `qy`, clock `t_s`, `next_dt_cap_s` (the adaptive cap) |
| transport | `sediment_velocity`, phase `fraction`/`position` (retained unchanged) |
| held topology | the held **source** (full `(ny+2,nx+2)` elevation incl. ring), `generation`, mask, friction, outlets, escape arrays (finite mask + values, no Infinity), digests |
| datum | initial full bed and initial committed elevation (host) |
| water ledger | initial anchors, rain/intake/saturation-return/drainage, per-face positive/negative/net volumes, export, steps, `t0`, `t_end`, reset removal (dry) |
| sediment ledger | host class anchors `inventory0`/`mobile0`, per-class requested/actual/shortfall/deposited/export vectors, per-cell pickup/deposition, per-face gross/net, residuals, phase counters, native-call count |
| native | the MAPLE pending **ledger** and **committed topography** states (fixed type whitelist) |
| counters / quiet | all `PondCounters` incl. commit log; `PondQuiet` (since, last time, window change, ready) |
| handoff only | `terminal_deposition`, reset surface/soil (the pre-reset arrays), and the accounting tree |
| identity | state config/forcing digests, rain-forcing digest through the binding |

Nothing serialised is a solver, face network, physics/splash grid, namespace module, callable, or pickled/dynamically
imported type. The only reconstructed dataclass types are the two MAPLE ledger types and the committed-topography state (by
name from a fixed registry, with an exact field set).

## Identity and the source pair

Checked **before any payload file is hashed or opened**:

1. exact manifest keys, schema, the manifest's own SHA-256;
2. caller identity equality (JSON-canonical);
3. **source pair**: the content digest of the *imported* `maple_syrup` package tree (`provenance.source_tree_digest`) and the
   digest of the *imported* `maple` package (`sediment_bed.bound_maple_package_digest`) — not a repository HEAD, so an edited
   or dirty checkout cannot masquerade as a clean commit; any mismatch is refused (`pond_checkpoint_source_pair()` shows the pair);
4. array backend and hydraulic backend (exact strings; `numpy` ≠ `numba`; CPU ≠ CuPy);
5. status, `allow_complete`;
6. **binding of the event**, independent of `event.config_sha256` (which may omit native bed-context fields): digests of the
   bed context (geometry incl. bulk density and boundaries, grain class order/diameters/densities, mass resolution, commit
   spec, avalanching spec, initial availability array, depth-update rule, window/deep options), the column parameter arrays,
   sediment (and splash) parameters, rain schedule and provenance, vegetation cover, rain scale, control and policy; the event's own
   config/forcing digests; shape, class count, phase bins; and the original topology's identity/source/topology digests.
   The failure message names which binding keys differ.

## Rebuild from the saved held source

Loading never touches the latest bed to choose geometry. The saved held source is rebuilt into a `PondTopology` with the
original mask/friction/outlets/nodata/closed-component policy taken from the event's original topology and the **saved
generation**; every field (source, mask, friction, outlets, escape level/next/connected/label arrays, closed count, generation,
identity/source/topology and geometry digests) is compared with the saved descriptor, and the saved datum must equal the
event's original source. The runtime (solver, grids, face network) is then built by the event from that verified state.

## Independent full validation (shared by save and load)

`_validate_state` never calls `step`, and never reads the event's runtime cache for its own conclusions:

* arrays: exact namespace/ndarray type, dtype, shape, finite, sign (non-negative depth/soil/phase/ledger totals; signed `q`, net
  fluxes, source, `deposit_unmet`); count fields integral;
* a freshly built solver validates the hydraulic state against the rebuilt faces (closed/outlet face rules, finiteness);
* held source and datum reproduce the topology source exactly; the committed-datum gate and the MAPLE gate on the current
  surface are both strictly below `pending_elevation_trigger_m` (an accepted state is never lagging the threshold; an
  un-stepped initial state whose pending pickup already crosses it is therefore refused — step it first);
* phase fractions/positions partition the **actual** mobile pool (`pond_transport.validate_phase`);
* native MAPLE validators on the bed: voxel/active partition, availability, ledger, committed topography, water, ledger
  reconciliation at the state clock, plus water depth equals hydraulic depth;
* clock, counter and source consistency (steps, refreshes = generation, commit log length = `min(commits, cap)`, native call
  count = `2·steps + 2·splash (+1 at terminal)`, quiet interval/ready vs the hold time and rain end, ledger clock = state clock);
* `event.audit(state)`: native-ledger-based water and sediment closure with MAPLE's bounds must both close;
* handoff: dry/empty/mobile-free state, empty and committed MAPLE ledger, reset arrays = pre-reset arrays, accounting keys,
  removed volumes and terminal deposition consistent with the arrays/ledger, both embedded audits closed.

Malformed payloads (unknown keys, wrong types, unknown native type or extra/missing fields, aliased/unused arrays, non-stored
or oddly named zip members, member size not matching the declared dtype/shape, object dtypes) are rejected **before** the
object they describe is constructed. No `inf`/`nan` is ever written (escape levels use a finite mask + values; the manifest
is strict JSON on write and read).

## Exact continuation

Same backend and same inputs ⇒ the restarted run is bitwise the uninterrupted run from the pause boundary: hydraulic
state and its adaptive cap, phase memory, velocity memory, ledgers, counters, quiet history, held source and generation are
all restored; the forcing is a pure function of time and the (bound) schedule, so no knot, rain edge or export is duplicated.
Wet and dry: a wet state at the quiet boundary finalises to the same handoff; a completed handoff loads only with
`allow_complete=True`, is data, and its reset can never run again (`finalize` refuses a dry state).

Save and load **never** commit, reroute, reset, finalise or step. A failed save removes only its own staging directory; a
refused load changes no file and no input.

## Backends

CPU (NumPy) and CuPy checkpoints are not interchangeable; neither are the `numpy` and `numba` hydraulic backends (the
hydraulic backend is recorded and compared). Device arrays cross the host only through MAPLE's counted `to_host`/`to_device`
(and the native snapshot writer's own counted conversion); the topology, datum and host class anchors stay on the host;
no unqualified `np.asarray(cupy_array)` and no copying of solver objects. Cross-backend migration is refused until qualified.

## C1 corrections

* **0-d payloads.** `np.ascontiguousarray` promotes a 0-d array to shape `(1,)`; the writer, the native-node encoder and the
  identity digest now copy with `np.array(..., copy=True, order="C")`, which keeps the exact shape. `add_host` additionally
  refuses any array whose host copy changed shape. Scalars such as `sl_n_phase_canonicalized` are written, tabled and read as `()`.
* **Bounded `.npy` preflight** (`_npy_preflight`) on every member of BOTH archives (native snapshot included) after the
  manifest/source/identity/binding/hash/zip-directory checks and before `load_state_snapshot` / `np.load` expand anything.
  Only the magic and a header of at most 4096 bytes are read through numpy's literal-only parser (no pickle, no `eval`); then
  `header end + prod(shape)·itemsize` must equal the archive's declared member size exactly (so a header demanding a huge
  array cannot allocate; the declared size is already bounded by `max_bytes`, default 1 GiB, never raised). Object/void
  dtypes are refused; 0-d, empty and string-metadata members use the same arithmetic. The companion header must equal the
  manifest table (dtype, shape, C order). The writer runs the same audit on its staging files.
* **Reset/terminal receipts use MAPLE's policy, not a floor.** The recorded surface/soil removal volumes are compared with
  `conservation.volume_roundoff_bound_m3(cells + 2, area·Σ|x|)` (doubled: two independent reductions); terminal deposition with
  `reservoir_bound_kg(cells, class totals)` (doubled). A tiny water reset therefore gets a tiny bound; zero must be exactly
  zero. The measured water budget (`event.audit`'s `reset_removed_m3`) and the declared receipts (accounting and embedded
  `final_audit`) must also agree to the same bound. The event's completion-precision fields (`max_depth`, excess surface volume)
  are physical stop precision and are not used here.
* **Identity vocabulary.** Every dict key at every depth must be a `str`; values None/bool/int/finite float/str/list/dict.
  `{1: "a"}` is refused instead of being coerced and colliding with `"1"`.
* **Counters follow the actual `PondCounters` fields** (int/float by default type; `commit_log` separate). A new int/float
  counter is carried automatically; any other kind of field, or a changed `PondQuiet`, is refused with an adaptation message
  instead of being dropped.
* A bare dry state is refused with an explicit "not a PondDryHandoff" message.
* **Tests** compare with a walker that skips exactly the native non-physical `surface_cache` field and `ArrayBinding` objects
  (`NONPHYSICAL_FIELDS/TYPES` in `pond_fixture.py`); everything physical, every control/time/counter, is compared. A closed-lake
  fixture (positive depth, no infiltration/rain/outlet, retained mobile load, explicit flow-end settlement) exercises the nonzero
  terminal transaction; the infiltrating dry fixture is kept for the soil reset.

## Internal API dependencies (maintain when updating the source pair)

Reads from `pond_event`: `PondEvent` attributes `context, column, sediment, schedule, splash, veg, rain_scale, control, policy,
xp, nc, shape, config_sha256, forcing_sha256`, **private** `_initial_topology`, and methods `audit`; classes `PondEventState`,
`PondDryHandoff`, `PondDatum`, `PondQuiet`, `PondCounters`, `WaterLedger`, `SedimentLedger` (importable names but not in
`__all__`), `SCHEMA`. If the writer tree renames or changes the fields of any of these, only `_extract`, `_build_state`, `_specs`
and `_binding` need updating. `PondEvent._check_state` is *not* used directly; `event.audit` (public) binds the runtime.

## Tests (`tests/pond_checkpoint`, root executes)

`test_roundtrip.py` (pending-ledger below threshold with source/generation 0 preserved; actual threshold crossing with the
saved generation; adaptive cap and rain knots through `evolve`; audits equal; no commit/reset on load; layout), `test_complete.py`
(dry handoff data-only load, no second finalisation, wet-at-quiet then finalise), `test_refusals.py` (manifest, source pair
before payload read, identity, backend, event binding independent of `config_sha256`, byte bound, every typed corruption
kind, swapped snapshot, no overwrite, failed-write cleanup, non-mutation), `test_cuda.py` (CuPy restart; honest skip without a
device; cross-backend refusal both ways).


## Windowed extension

The preceding schema `/1` is the dense format. [Windowed pond checkpoints](../pond_windowed/checkpoint.md) use distinct schema `maple-syrup-pond-checkpoint/2`, native resident snapshots and exact immutable external deep-store references, with the same source/backend guards and original 1 GiB resident-plus-companion bound. Pending journals must be explicitly published before save; save never publishes or reroutes.
