# ruff: noqa: BLE001  (the verbatim validator copy and the origin rebuild convert any native failure into one typed refusal, as the historical codec does)
"""CC1: explicit CONTINUED-ORIGIN CPU physical checkpoint of a floating Pond state (NumPy/Numba x Dense / published Windowed).

WHY A NEW PROFILE. A Pond state that continues a lifecycle boundary (`floating_event_lifecycle.begin_occurrence` -> `PondEvent.continued_state`) carries the LAST actual routing topology (possibly generation > 0), the held source and the GLOBAL
ORIGINAL datum, with truthful fresh EVENT-LOCAL counters/ledgers (`n_refreshes == 0`). The historical floating profiles (`maple-syrup-floating-pond-checkpoint/1-6`) require a state whose datum equals the event's own initial topology source and
whose local `n_refreshes` equals the topology generation; a genuine continued state therefore fails the datum check FIRST and the counter rule second. Neither is relaxed: the historical validators, schemas and loaders are byte-unchanged in behaviour
(`original_topology=None`), and a continued file is refused by them (and an old file by this profile).

THE PROFILE (`maple-syrup-floating-continued-pond-checkpoint/1` Dense, `/2` published Windowed; profile `floating-vertical-continued-origin/1`; layout, payloads, native snapshot, held template, W/J/I/D, terminal-exchange and retained quiet-reset
records identical to `floating_checkpoint` /5-/6). Declared differences, each explicit and distinct:
  * the caller supplies `original_topology` (the genuine generation-0 `PondTopology` of the ORIGINAL case; the datum is attested against it, never against the carried routing origin) and the event's own `_initial_topology` is the CARRIED
    routing origin (any generation); both are bound in the manifest (`binding.continued_origin` + `binding.initial_topology`) and the carried routing source is stored (`co_routing_source`) so the origin can be read back before an event exists;
  * the state's current topology may be a LATER reroute of the carried origin; the local counter rule is `n_refreshes == topology.generation - origin.generation` (and `n_refreshes <= n_commits`); a state at the origin (no local refresh)
    must carry exactly the carried topology (same source, same digest) -- the carried routing is never stale and never rerouted by a save or a load;
  * nothing else changes: the independent floating validation (copy of the historical one, AST-guarded), the native bed validators, geometry gate against the CURRENT bed, phase partition, event audit, canonical-depth check, held template checks and
    the capability chain (`VerifiedFloatingInputs` -> `prepare_from_verified_inputs` -> `resume_scope` -> `adopt_resumed` LAST) are the unchanged ones.

NOT claimed: a quiet/completion proof (the fresh monitor of a continued state is not one), a boundary/daily cursor, a scheduler, GPU, or an outer H0 container for this profile (the historical outer `floating_context_checkpoint` calls a one-argument
factory; a continued outer needs `factory(held_context, carried_topology)` and is the next slice). The restart TARGET is the caller's event built on a vertical that carries the refreshed held root (e.g. an H0 reconstruction of the boundary root)
and on the carried routing origin read from the file by `read_continued_origin`. The `datum_committed` elevation is carried from the file (as in the historical profile; only its shape is attested).
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup import floating_checkpoint as fc
from maple_syrup import pond_checkpoint as pc
from maple_syrup.characteristic_transport import PhaseState
from maple_syrup.checkpoint import DEFAULT_MAX_CHECKPOINT_BYTES
from maple_syrup.floating_checkpoint import (
    CONTINUED_SCHEMAS,
    ORIGIN_ARRAY,
    FloatingCheckpointError,
)
from maple_syrup.pond_bed import (
    committed_displacement_m,
    gate_displacement_m,
    physical_full_source,
)
from maple_syrup.pond_checkpoint import (
    PondCheckpointError,
    _check_arrays,
    _extract,
    _fields_mismatch,
    _initial_topology,
    _int,
    _native_checks,
    _real,
    _rebuild_topology,
    _refuse,
    _require_published,
    _reset_receipts,
    _specs,
    _topology_fields,
    _validate_commit_log,
    _validate_handoff,
)
from maple_syrup.pond_event import SCHEMA as POND_EVENT_SCHEMA
from maple_syrup.pond_event import PondDryHandoff, PondEvent, PondEventState
from maple_syrup.pond_network import PondTopology, build_pond_topology
from maple_syrup.pond_transport import face_network_from_pond_topology, validate_phase
from maple_syrup.sediment_bed import BedState, is_windowed_bed, reroute_threshold_m

__all__ = ["ContinuedOrigin", "load_floating_continued_checkpoint", "origin_topology", "read_continued_origin", "save_floating_continued_checkpoint", "validate_continued_state"]


def _fail(message: str) -> None:
    raise FloatingCheckpointError(message)


def _check_continued_origin(fresh: PondTopology, arrays: dict, init: PondTopology, origin: Any) -> None:
    """The two DECLARED replacements of the historical datum check: the state belongs to the event's static case AND to the caller-attested ORIGINAL case, its datum is the ORIGINAL datum (not the carried routing source), and the
    topology is the carried routing origin or a later reroute of it (an unrefreshed state carries exactly the carried topology)."""
    if type(origin) is not PondTopology or int(origin.generation) != 0:
        _refuse("the continued-origin profile needs the genuine generation-0 original topology")
    if fresh.identity_sha256 != init.identity_sha256 or fresh.identity_sha256 != origin.identity_sha256:
        _refuse("the state does not belong to the event's / original case's static topology")
    if not np.array_equal(arrays["datum_full"], origin.source_z_full):
        _refuse("the state's datum is not the ORIGINAL case datum (the carried routing source is not a datum)")
    if fresh.generation < init.generation:
        _refuse("the state's topology generation precedes the carried routing origin")
    if fresh.generation == init.generation and (fresh.topology_sha256 != init.topology_sha256 or fresh.source_sha256 != init.source_sha256 or not np.array_equal(arrays["topo_source"], init.source_z_full)):
        _refuse("a state without a local reroute must carry exactly the carried routing topology")


def origin_record(event: PondEvent, arrays: dict) -> dict:
    """The manifest's `continued_origin` block (strict JSON): the carried routing generation and the hash of the stored ORIGINAL datum array."""
    init = _initial_topology(event)
    return {"routing_generation": int(init.generation), "datum_full_sha256": hashlib.sha256(np.ascontiguousarray(arrays["datum_full"], dtype=np.float64).tobytes()).hexdigest()}


def check_origin_arrays(event: PondEvent, rec: Any, host_arrays: dict, original: Any) -> None:
    """Loader check (before any state is built): record keys/types, the stored carried routing source and datum against the caller's event and original topology, bit exact."""
    init = _initial_topology(event)
    if not isinstance(rec, dict) or set(rec) != {"routing_generation", "datum_full_sha256"} or type(rec["routing_generation"]) is not int or not isinstance(rec["datum_full_sha256"], str):
        _fail("continued_origin record is malformed")
    if rec["routing_generation"] != int(init.generation):
        _fail("the file's carried routing generation differs from the event's")
    if not np.array_equal(host_arrays[ORIGIN_ARRAY], init.source_z_full):
        _fail("the stored carried routing source differs from the event's carried routing origin")
    datum = host_arrays["datum_full"]
    if rec["datum_full_sha256"] != hashlib.sha256(np.ascontiguousarray(datum, dtype=np.float64).tobytes()).hexdigest() or not np.array_equal(datum, original.source_z_full):
        _fail("the stored datum is not the ORIGINAL case datum the caller attests")


def validate_continued_state(event: PondEvent, state: Any, handoff: PondDryHandoff | None,
                    storage: str, origin: Any) -> tuple[dict, dict, PondTopology]:
    """CONTINUED-ORIGIN copy of `floating_checkpoint._validate_floating_state` (identical except the extra `origin` argument and the TWO declared statements: the carried-origin / datum check and the local-counter scalar rule; drift-guarded by test). FULL independent validation of a state (or the dry state of a handoff). Never calls `event.step`, never consults the
    event's runtime cache for its own checks. Returns `(arrays, scalars, fresh_topology)`. `storage` is the checkpoint's bed
    storage ("dense" or "windowed"); a bed of the other kind is refused (no cross-storage conversion)."""
    from maple.core.backend import array_namespace, to_bool, to_host

    xp, ctx = event.xp, event.context
    if not isinstance(state, PondEventState):
        _refuse("state must be a PondEventState")
    if state.schema != POND_EVENT_SCHEMA:
        _refuse(f"state schema {state.schema!r} is not {POND_EVENT_SCHEMA!r}")
    if state.config_sha256 != event.config_sha256 or state.forcing_sha256 != event.forcing_sha256:
        _refuse("state was built under a different event configuration or forcing")
    if state.status != ("wet" if handoff is None else "dry"):
        _refuse(f"state status {state.status!r} does not match the checkpoint kind")
    arrays, scalars = _extract(state, handoff, xp)
    _check_arrays(arrays, _specs(event, dry=handoff is not None), xp)

    # topology: an independent fresh build from the HELD source, every field and digest compared (generation preserved)
    init = _initial_topology(event)
    top = state.topology
    if not isinstance(top, PondTopology):
        _refuse("state.topology must be a PondTopology")
    generation = _int(top.generation, "topology generation")
    fresh = _rebuild_topology(event, arrays["topo_source"], generation)
    bad = _fields_mismatch(_topology_fields(top), _topology_fields(fresh))
    if bad:
        _refuse(f"saved topology differs from an independent build of its held source in {bad}")
    _check_continued_origin(fresh, arrays, init, origin)
    if arrays["datum_committed"].shape != tuple(event.shape):
        _refuse("datum committed elevation shape mismatch")

    # hydraulics against the faces of the fresh topology (a freshly built solver, not the event cache)
    # FLOATING hydraulic replacement (the ONLY declared difference from the historical `_validate_state`): the genuine floating faces/solver of the event's vertical on the rebuilt topology, validating the
    # hydraulic state against the REGISTERED (or provisionally resolved) floating payload -- never a bucket `ColumnParameters` (None for a floating event) and never a fake column
    try:
        pond = event.vertical.build_hydraulics(fresh, control=event.control.hydraulic)
        pond.solver.validate_state(state.hydraulic)
    except Exception as exc:
        raise PondCheckpointError(f"floating hydraulic state is invalid for the rebuilt faces: {exc}") from None
    hyd = state.hydraulic
    if hyd.next_dt_cap_s is not None and _real(hyd.next_dt_cap_s, "next_dt_cap_s") <= 0.0:
        _refuse("next_dt_cap_s must be None or positive")

    # held source / datum identity and the geometry gate
    held_host = np.asarray(to_host(state.held_source_m))
    try:
        expected = physical_full_source(state.datum.initial_full_m, state.datum.initial_committed_m, held_host, fresh.active)
    except Exception as exc:
        raise PondCheckpointError(f"held source cannot be combined with the datum: {exc}") from None
    if not np.array_equal(expected, fresh.source_z_full):
        _refuse("held source and topology source differ (source identity / generation mismatch)")
    bed = state.bed
    if not isinstance(bed, BedState) or array_namespace(bed.water.depth_m) is not xp:
        _refuse("the bed is not an actual MAPLE BedState in the event namespace")
    if is_windowed_bed(bed) != (storage == "windowed"):
        _refuse(f"the bed storage ({'windowed' if is_windowed_bed(bed) else 'dense'}) is not the checkpoint's ({storage}); "
                "no cross-storage conversion exists")
    if getattr(event.control, "bed_storage", "dense") != storage:
        _refuse(f"the event control's bed_storage ({getattr(event.control, 'bed_storage', 'dense')!r}) is not the "
                f"checkpoint's ({storage!r}); a windowed checkpoint needs a PondEventControl(bed_storage='windowed')")
    if storage == "windowed":
        _require_published(bed)
    _native_checks(bed, event, scalars["t_s"], storage)
    if not to_bool(xp.array_equal(bed.water.depth_m, hyd.depth_m)):
        _refuse("native water depth differs from the hydraulic depth")
    active_dev = xp.asarray(fresh.active)
    try:
        threshold = reroute_threshold_m(ctx)
        gate = gate_displacement_m(bed, ctx, state.held_source_m, active_dev)
        net = committed_displacement_m(bed, state.held_source_m, active_dev)
    except Exception as exc:
        raise PondCheckpointError(f"geometry gate cannot be evaluated: {exc}") from None
    if not (gate < threshold and net < threshold):
        _refuse(f"the held source lags the bed by {max(gate, net)} m >= the commit threshold {threshold} m: not an accepted state")

    # phase partition of the ACTUAL mobile pool
    try:
        network = face_network_from_pond_topology(fresh, xp=xp)
        phase = PhaseState(fraction=state.phase_fraction, position_m=state.phase_position_m, dx_m=fresh.dx_m,
                           n_bins=event.control.phase_bins, xp=xp)
        validate_phase(network, bed.water.mobile_mass_by_cell_class_kg, phase)
    except Exception as exc:
        raise PondCheckpointError(f"phase state does not partition the mobile pool: {exc}") from None

    _validate_continued_scalars(event, state, scalars, handoff)
    if handoff is not None:
        _validate_handoff(event, state, handoff, arrays)

    # native ledger audit (water and sediment closure); also binds the event runtime lazily to the verified topology
    try:
        audit = event.audit(state)
    except Exception as exc:
        raise PondCheckpointError(f"event audit failed: {exc}") from None
    if audit["water"]["closes"] is not True or audit["sediment"]["closes"] is not True:
        _refuse("water / sediment budgets of the state do not close")
    if handoff is not None:
        # measured global budget (the live audit's reset removal) and the declared receipts must agree to MAPLE's bound
        receipts = _reset_receipts(state, event)
        measured = float(audit["water"]["reset_removed_m3"])
        bound = receipts["surface"][1] + receipts["soil"][1]
        declared = [("accounting", handoff.accounting["surface_removed_m3"] + handoff.accounting["soil_removed_m3"])]
        final = handoff.accounting["final_audit"]["water"].get("reset_removed_m3")
        if final is not None:
            declared.append(("final_audit", final))
        for label, value in declared:
            if abs(_real(value, label) - measured) > 2.0 * bound:
                _refuse(f"declared reset removal ({label}: {value!r}) disagrees with the measured water budget "
                        f"({measured!r}, bound {2.0 * bound!r})")
    return arrays, scalars, fresh


def _validate_continued_scalars(event: PondEvent, state: PondEventState, scalars: dict, handoff: PondDryHandoff | None) -> None:
    """CONTINUED-ORIGIN copy of `floating_checkpoint._validate_scalars` (identical except the declared local refresh-counter rule; drift-guarded by test). Counter / clock / source consistency (every relation is an invariant of the accepted step of `PondEvent`)."""
    c, wl, q, top = scalars["counters"], scalars["water"], scalars["quiet"], state.topology
    t = scalars["t_s"]
    n_steps = c["n_accepted_steps"]
    if n_steps != wl["n_steps"]:
        _refuse("accepted-step counter disagrees with the water ledger step count")
    if wl["t_end_s"] != t or wl["t0_s"] > t or (n_steps == 0) != (wl["t0_s"] == t):
        _refuse("water ledger clock disagrees with the state clock")
    if top.generation < _initial_topology(event).generation or c["n_refreshes"] != top.generation - _initial_topology(event).generation or c["n_refreshes"] > c["n_commits"]:
        _refuse("local refresh counter disagrees with the topology generation advance since the carried routing origin / commits")
    if c["n_maple_trigger_commits"] + c["n_net_elevation_commits"] + c["n_final_commits"] > c["n_commits"]:
        _refuse("commit reason counters exceed the commit counter")
    if c["n_splash_exchanges"] > n_steps or (event.splash is None and c["n_splash_exchanges"]):
        _refuse("splash exchange counter inconsistent")
    base = 2 * n_steps + 2 * c["n_splash_exchanges"]
    calls = scalars["sediment"]["n_native_calls"]
    terminal = state.sediment.n_terminal_native_calls  # PRODUCER-maintained by `_settle_terminal` (S4-1); 0 for every state of a /1 or /2 file
    if type(terminal) is not int or terminal < 0:
        _refuse("terminal native-call count must be an integer >= 0")
    from maple.core.backend import to_host

    if (terminal > 0) != bool(np.any(np.asarray(to_host(state.sediment.deposited_terminal)) > 0.0)):  # counted per-class readback, identity for NumPy
        _refuse("terminal native-call count disagrees with the terminal deposition ledger")
    if (calls != base + terminal) if handoff is None else (calls not in (base + terminal, base + terminal + 1)):
        _refuse("native call counter disagrees with the accepted steps and terminal exchanges")
    _validate_commit_log(state.counters.commit_log, event, c, top.generation)
    if handoff is not None:
        if q != {"since_s": None, "last_t_s": None, "window_depth_change_m": 0.0, "ready": False}:
            _refuse("a completed state must carry the reset quiet monitor")
        return
    # a RETAINED terminal settlement that woke the flow returns the FRESH quiet monitor of a wet state with accepted steps (`settle_retaining_water` -> `PondQuiet()`); it is accepted ONLY at the CURRENT accepted step for which the
    # producer recorded its own reset receipt (an ordinary later step keeps the older receipt and clears the fresh monitor), never as a general relaxation or from historical totals
    reset_step = state.sediment.retained_quiet_reset_step  # PRODUCER receipt (`settle_retaining_water` wake branch) of the accepted-step count at the quiet reset; None for a /1-/4 file or when no reset happened
    if reset_step is not None and (type(reset_step) is not int or not 0 <= reset_step <= n_steps):
        _refuse("retained quiet-reset step must be None or an integer in [0, accepted steps]")
    fresh = q == {"since_s": None, "last_t_s": None, "window_depth_change_m": 0.0, "ready": False}
    woken = fresh and n_steps > 0 and reset_step == n_steps
    if not woken and ((q["last_t_s"] is None) != (n_steps == 0) or (q["last_t_s"] is not None and q["last_t_s"] != t)):
        _refuse("quiet monitor time differs from the accepted state")
    pol = event.policy
    since = q["since_s"]
    if since is None:
        if q["ready"] or q["window_depth_change_m"] != 0.0:
            _refuse("quiet monitor without an interval cannot be ready or carry a depth change")
    else:
        if not (event.schedule.end_s <= since <= q["last_t_s"]) or q["window_depth_change_m"] > pol.max_depth_change_m:
            _refuse("quiet interval is outside the rain end, the state time or the depth-change bound")
        if q["ready"] != bool(q["last_t_s"] - since >= pol.hold_s):
            _refuse("quiet ready flag disagrees with the hold time")


@dataclasses.dataclass(frozen=True, eq=False)
class ContinuedOrigin:
    """The CARRIED routing origin read back from a continued-origin FILE, before any event exists (metadata + one small array; hashes verified). It is a pure value: it claims NO acceptance and authorises nothing -- the loader rebuilds nothing from it
    that it does not re-verify against the event the caller builds from it."""

    schema: str
    storage: str
    routing_generation: int
    routing_source_full: np.ndarray
    routing_identity_sha256: str
    routing_source_sha256: str
    routing_topology_sha256: str
    manifest_sha256: str


def read_continued_origin(path: Any, expected_identity: dict, *, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0, max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES) -> ContinuedOrigin:
    """CPU continued-origin file reader (module docstring); the shared core is `read_origin_core` (also used by the CUDA continued profile with ITS schemas, keys, profile and kernel check)."""
    return read_origin_core(path, expected_identity, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, max_bytes=max_bytes, schemas=CONTINUED_SCHEMAS, dense_keys=fc.MANIFEST_KEYS_CONTINUED,
                            windowed_keys=fc.WINDOWED_MANIFEST_KEYS_CONTINUED, profile=fc.CONTINUED_PROFILE, windowed_schemas=(fc.CONTINUED_WINDOWED_SCHEMA,), extra_check=None)


def read_origin_core(path: Any, expected_identity: dict, *, available_bytes_fn: Callable[[], int], reserve_bytes: int, max_bytes: int, schemas: tuple, dense_keys: Any, windowed_keys: Any, profile: str,
                     windowed_schemas: tuple, extra_check: Callable[[dict], None] | None) -> ContinuedOrigin:
    """Bounded metadata read of a continued-origin file: strict manifest, seal, schema/profile/keys, identity, source pair, and the hash of the companion file (streamed, bounded buffer); then ONE array (the carried routing source) is read
    after its table record and member header are cross-checked. No payload array beyond that one is expanded; nothing is registered or changed."""
    limit = pc._max_bytes(max_bytes)
    reserve = fc._reserve(reserve_bytes)
    expected = pc._canonical_identity(expected_identity)
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        _fail(f"checkpoint path is not a plain directory: {root}")
    root = root.resolve()
    if {p.name for p in root.iterdir()} != {fc.MANIFEST_NAME, *fc.PAYLOADS}:
        _fail("floating checkpoint directory must contain exactly the manifest and the two payload files")
    try:
        msize = os.lstat(root / fc.MANIFEST_NAME).st_size
    except OSError as exc:
        _fail(f"floating checkpoint manifest is missing: {exc}")
    fc._read(available_bytes_fn, 3 * int(msize) + fc.HASH_BUFFER_BYTES + fc.HEADER_AUDIT_BYTES + reserve, "continued-origin manifest read")
    data = fc._read_manifest(root)
    schema = data.get("schema")
    if schema not in schemas:
        _fail(f"checkpoint schema {schema!r} is not a continued-origin floating pond checkpoint schema")
    windowed = schema in windowed_schemas
    if set(data) != (windowed_keys if windowed else dense_keys):
        _fail("manifest keys mismatch")
    if pc._hex(data["manifest_sha256"], "manifest_sha256") != fc._seal(data)["manifest_sha256"]:
        _fail("manifest hash mismatch (the manifest was modified)")
    if data["profile"] != profile or data["identity"] != expected:
        _fail("profile or caller identity mismatch")
    if data["source"] != pc.pond_checkpoint_source_pair():
        _fail("source pair mismatch: refused before any payload is read")
    if extra_check is not None:
        extra_check(data)  # profile-specific closed checks from the manifest alone (CUDA: the kernel/ABI descriptor), BEFORE any payload is touched
    binding, files, payload = data["binding"], data["files"], data["payload"]
    if not isinstance(binding, dict) or not isinstance(files, dict) or set(files) != set(fc.PAYLOADS) or not isinstance(payload, dict) or not isinstance(payload.get("arrays"), dict):
        _fail("manifest structure mismatch")
    init = binding.get("initial_topology")
    shape = binding.get("shape")
    rec = payload["arrays"].get(ORIGIN_ARRAY)
    if (not isinstance(init, dict) or set(init) != {"identity_sha256", "source_sha256", "topology_sha256", "generation"} or type(init["generation"]) is not int or init["generation"] < 0
            or not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape) or rec != {"dtype": "<f8", "shape": [shape[0] + 2, shape[1] + 2]}):
        _fail("carried routing origin record is malformed")
    for key in ("identity_sha256", "source_sha256", "topology_sha256"):
        pc._hex(init[key], f"initial_topology.{key}")  # strict 64-hex digests (typed refusal otherwise)
    n_bytes = (shape[0] + 2) * (shape[1] + 2) * 8
    fc._read(available_bytes_fn, 3 * n_bytes + fc.HASH_BUFFER_BYTES + fc.HEADER_AUDIT_BYTES + reserve, "continued-origin routing read")
    companion = fc.PAYLOADS[1]
    if pc._hash_file(root / companion, limit)[0] != pc._hex(files[companion], f"files[{companion}]"):
        _fail(f"checkpoint payload hash mismatch: {companion}")
    sizes = pc._zip_audit(root / companion, limit)
    if ORIGIN_ARRAY + ".npy" not in sizes:
        _fail("the companion lacks the carried routing origin array")
    table = payload["arrays"]
    pc._audit_array_table(table, {}, sizes, limit)  # the historical strict table audit (types, dtype/shape, summed bytes, member sizes, member set): metadata only
    pc._npy_preflight(root / companion, sizes, table)  # the historical bounded header audit of EVERY member against the table BEFORE np.load expands anything
    try:
        with np.load(root / companion, allow_pickle=False) as archive:
            a = np.array(archive[ORIGIN_ARRAY], copy=True, order="C")
    except (ValueError, OSError, zipfile.BadZipFile, EOFError, KeyError) as exc:
        _fail(f"the carried routing origin cannot be read: {exc}")
    if a.dtype.str != "<f8" or list(a.shape) != rec["shape"] or not np.isfinite(a).all():
        _fail("the carried routing origin array is malformed")
    a.flags.writeable = False
    return ContinuedOrigin(schema, "windowed" if windowed else "dense", int(init["generation"]), a, init["identity_sha256"], init["source_sha256"], init["topology_sha256"], data["manifest_sha256"])


def origin_topology(origin: ContinuedOrigin, static_topology: PondTopology) -> PondTopology:
    """The carried routing topology a continued event is built on: an independent fresh build from the stored source with the static case of `static_topology` (active, friction, spacing, outlets, closed-component rule) at the file's
    generation, compared in identity / source / topology digests with the file's record. A different static case or source is refused."""
    if type(origin) is not ContinuedOrigin or type(static_topology) is not PondTopology:
        _fail("origin_topology needs a ContinuedOrigin and a PondTopology")
    try:
        fresh = build_pond_topology(np.array(origin.routing_source_full, dtype=np.float64, copy=True), static_topology.active, static_topology.friction, static_topology.dx_m, static_topology.outlets,
                                    nodata_value=static_topology.nodata_value, allow_closed_components=static_topology.n_closed_components > 0)
    except Exception as exc:
        raise FloatingCheckpointError(f"the stored carried routing source does not rebuild a valid topology: {exc}") from None
    top = dataclasses.replace(fresh, generation=origin.routing_generation)
    if (top.identity_sha256, top.source_sha256, top.topology_sha256) != (origin.routing_identity_sha256, origin.routing_source_sha256, origin.routing_topology_sha256):
        _fail("the rebuilt carried routing topology differs from the file's record (a different static case or source)")
    return top


def save_floating_continued_checkpoint(path: Any, state: Any, event: PondEvent, identity: dict, *, original_topology: PondTopology, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0,
                                       max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES) -> Path:
    """Write a continued-origin checkpoint (module docstring): `original_topology` is REQUIRED (the caller-attested generation-0 original case). Nothing is advanced, committed, rerouted, reset or consumed."""
    if original_topology is None:
        _fail("original_topology is required by the continued-origin profile")
    return fc.save_floating_pond_checkpoint(path, state, event, identity, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, max_bytes=max_bytes, original_topology=original_topology)


def load_floating_continued_checkpoint(path: Any, event: PondEvent, expected_identity: dict, *, original_topology: PondTopology, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0,
                                       max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES, expected_bed_storage: str | None = None) -> PondEventState:
    """Restore a continued-origin checkpoint into `event` (built on the carried routing origin of `read_continued_origin` / `origin_topology` and a vertical carrying the refreshed held root). Same validation, admission and LAST adoption as
    the historical loader."""
    if original_topology is None:
        _fail("original_topology is required by the continued-origin profile")
    return fc.load_floating_pond_checkpoint(path, event, expected_identity, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, max_bytes=max_bytes, expected_bed_storage=expected_bed_storage,
                                            original_topology=original_topology)
