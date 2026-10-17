"""POND EVENT CHECKPOINT / CONTINUATION (opt-in, new schema; dense bed first). Documentation: docs/splash_and_overflow/pond_checkpoint.md.

Saves and restores the continuation of the coupled pond event of `maple_syrup.pond_event` (`PondEventState`, or the completed
`PondDryHandoff`) so that a same-backend restart continues EXACTLY: the same hydraulics, phase memory, ledgers, counters,
quiet monitor, adaptive step cap and held source as the uninterrupted run. It is a NEW directory format with its own schema
(`SCHEMA`); the accepted `maple_syrup.checkpoint` (schema `maple-syrup-checkpoint/2`, `checkpoint.json`) is untouched, its
defaults are not changed, and it does not (and must not) decode this format.

PUBLIC API
    save_pond_checkpoint(path, state_or_handoff, event, identity, *, max_bytes=DEFAULT_MAX_CHECKPOINT_BYTES) -> Path
    load_pond_checkpoint(path, event, expected_identity, *, allow_complete=False, max_bytes=DEFAULT_MAX_CHECKPOINT_BYTES)
        -> PondEventState (wet) | PondDryHandoff (complete; only with `allow_complete=True`)
    pond_checkpoint_source_pair() -> dict   (the package digests recorded in, and required of, every checkpoint)

LAYOUT (a NEW sibling directory, never an overwrite)
    pond_checkpoint.json    hash-bound manifest: schema, caller identity, status, backends, source pair, binding, payload
                            hashes, the typed payload tree and the manifest's own SHA-256
    maple_state.npz         ONE dense bed authority written by the actual MAPLE `save_state_snapshot`
                            (voxel column, active layer, availability, WATER state)
    pond_continuation.npz   the companion: hydraulic state, phase, held topology SOURCE, datum, water / sediment ledgers, the
                            native pending ledger and committed topography, counters, quiet, and for a handoff the pre-reset
                            arrays and accounting. The bed voxel / active inventory is NOT repeated here.

WHAT IS NOT SERIALISED: the solver, the face network, physics / splash grids, the event runtime cache, namespaces, callables,
dataclass type names of anything but the fixed whitelist (the two MAPLE ledger types and the committed-topography state) and
no pickle. The caller supplies the static `PondEvent` (column, sediment, schedule, rain scale, control, policy, splash,
vegetation, ORIGINAL initial topology); the checkpoint only proves it is the same one.

LOAD ORDER (nothing is trusted before the step that proves it; a refusal leaves every input and file unchanged)
 1. argument and directory checks (no symlink, exactly the three files); manifest size bound, strict JSON (no NaN / Infinity,
    no duplicate keys), exact key set, the manifest's own SHA-256, schema.
 2. BEFORE any payload is opened: caller identity (strict, JSON-canonical), the source pair (the digest of the imported
    `maple_syrup` package tree AND of the imported `maple` package, not a checkout HEAD), array backend, hydraulic backend
    (`numpy` and `numba` are NOT interchangeable), status / `allow_complete`, and the binding of the event: the native bed
    context (geometry, bulk density, grain class order / diameters / densities, mass resolution, commit spec, availability,
    avalanching, depth rule), the column / sediment / splash arrays, rain schedule, vegetation, rain scale, control, policy
    and the ORIGINAL topology identity. `event.config_sha256` alone may omit context fields, so the binding is independent.
 3. payload file hashes and zip central-directory audit (stored members only, names, declared sizes) against `max_bytes`
    (default the existing 1 GiB; never inferred or raised) BEFORE a single array is expanded.
 4. native snapshot via MAPLE `load_state_snapshot` (validated, restartable, not diagnostic, provenance bound to this
    manifest, no foreign optional state); companion decode through a fixed whitelist (types, names, dtypes, shapes).
 5. an independent full validation (the same function the writer runs before it writes): arrays (namespace, dtype, shape,
    finite, sign), a FRESH topology build from the saved held source compared in every field and digest (escape arrays
    included) with the saved generation preserved, a freshly built solver validating the hydraulic state against those faces
    (the event's cache is not consulted), datum / held source identity, geometry gate, phase partition against the actual
    mobile pool, native MAPLE validators (partition, availability, ledger, committed topography, water, reconciliation at the
    state clock), counter / clock / source consistency, and finally `event.audit` (native ledger and water / sediment closure).
    `state.step` is never executed to validate.
 6. the verified state is returned; the event rebinds its runtime lazily from the saved held source on first use.

NO commit, no reroute, no reset and no finalisation happens on save or load. A completed handoff cannot be resumed and its
dry reset cannot be repeated: it loads only with `allow_complete=True`, as data.

BACKENDS. CPU NumPy arrays and CuPy arrays are different backends and are never converted into each other; a checkpoint is
loaded only by an event of the same array backend AND the same hydraulic backend string. Device arrays cross the host only
through MAPLE's counted `to_host` / `to_device` (and the native snapshot writer's own conversion). The topology, datum and the
host class-total anchors of the sediment ledger stay on the host.

Nothing here was run by its author (file-only tools); the root executes `tests/pond_checkpoint`.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from maple_syrup.characteristic_transport import PhaseState
from maple_syrup.checkpoint import DEFAULT_MAX_CHECKPOINT_BYTES
from maple_syrup.pond_bed import (
    committed_displacement_m,
    gate_displacement_m,
    physical_full_source,
    require_supported_bed,
)
from maple_syrup.pond_event import SCHEMA as POND_EVENT_SCHEMA
from maple_syrup.pond_event import (
    PondCounters,
    PondDatum,
    PondDryHandoff,
    PondEvent,
    PondEventState,
    PondQuiet,
    SedimentLedger,
    WaterLedger,
)
from maple_syrup.pond_hydraulics import build_pond_hydraulics
from maple_syrup.pond_network import PondTopology, build_pond_topology
from maple_syrup.pond_transport import face_network_from_pond_topology, validate_phase
from maple_syrup.sediment_bed import BedState, bound_maple_package_digest, reroute_threshold_m

__all__ = [
    "DEFAULT_MAX_CHECKPOINT_BYTES",
    "MANIFEST_NAME",
    "PAYLOADS",
    "SCHEMA",
    "PondCheckpointError",
    "load_pond_checkpoint",
    "pond_checkpoint_source_pair",
    "save_pond_checkpoint",
]

SCHEMA = "maple-syrup-pond-checkpoint/1"
MANIFEST_NAME = "pond_checkpoint.json"
PAYLOADS = ("maple_state.npz", "pond_continuation.npz")
MAX_MANIFEST_BYTES = 32 * 1024**2
MANIFEST_KEYS = frozenset({"schema", "identity", "status", "array_backend", "hydraulic_backend", "source", "binding",
                           "binding_sha256", "files", "payload", "manifest_sha256"})
PAYLOAD_KEYS = frozenset({"scalars", "arrays", "native", "accounting"})
STATUSES = ("wet", "complete")
BACKENDS = ("numpy", "cupy")
_MAX_TREE_DEPTH = 24
_MAX_TREE_NODES = 200_000
_MAX_NDIM = 5
_FOREIGN_SNAPSHOT_FIELDS = ("mobile_state", "grain_borne_stress", "threshold_modifier_state", "vegetation",
                            "topographic_wind_source", "scheduler_block_index", "scheduler_last_wind")
_COMMIT_REASONS = ("maple_trigger", "net_elevation", "final_forced")
_COMMIT_KEYS = frozenset({"t_s", "reason", "gate_before_m", "net_after_m", "threshold_m", "committed", "refreshed",
                          "generation"})
_ACCOUNTING_KEYS = frozenset({"policy", "surface_removed_m3", "soil_removed_m3", "completion_before_terminal",
                              "completion_after_terminal", "terminal_deposition_by_class_kg", "pre_terminal_audit",
                              "final_audit", "final_commit", "geometry_refreshed_at_final"})
_SOURCE_CACHE: dict[str, dict] = {}


class PondCheckpointError(RuntimeError):
    """A checkpoint that is invalid, corrupt, incompatible or unsupported, or a state that cannot be saved. Raised before
    anything is published; the inputs, the event and every existing file are unchanged."""


def _refuse(message: str) -> None:
    raise PondCheckpointError(message)


def _dtype_ok(text: Any) -> bool:
    """Numeric / bool little-endian (or byte-order-free) dtype strings of at most 8 bytes: never object, string or void."""
    if not isinstance(text, str) or not text or text[0] not in "<|":
        return False
    try:
        dt = np.dtype(text)
    except TypeError:
        return False
    return dt.kind in "fiub" and dt.itemsize <= 8 and dt.str == text


# --- identity / source ----------------------------------------------------------------------------------------------
def pond_checkpoint_source_pair() -> dict:
    """The digests of the ACTUAL imported code: the content digest of the `maple_syrup` package tree and the digest of the
    imported `maple` package (`sediment_bed.bound_maple_package_digest`). Computed once per process. Never a repository HEAD."""
    if "pair" not in _SOURCE_CACHE:
        import maple_syrup
        from maple_syrup.provenance import ProvenanceScopeError, source_tree_digest

        try:
            tree = source_tree_digest(Path(maple_syrup.__file__).resolve().parent)
            maple_digest = bound_maple_package_digest()
        except (ProvenanceScopeError, OSError, ImportError) as exc:
            raise PondCheckpointError(f"the source pair of the imported packages cannot be computed: {exc}") from None
        _SOURCE_CACHE["pair"] = {"maple_syrup_tree_sha256": tree.digest_sha256, "maple_syrup_n_files": int(tree.file_count),
                                 "maple_package_sha256": maple_digest, "algorithm": str(tree.algorithm)}
    return dict(_SOURCE_CACHE["pair"])


def _strict_json_tree(value: Any, depth: int = 0) -> None:
    """Exact JSON vocabulary of the caller identity: every dict key at every depth is a `str` (`json.dumps` would silently
    coerce numeric / bool / None keys and could collide `1` with `"1"`), values are None / bool / int / finite float / str /
    list / tuple / dict. Nothing is coerced."""
    if depth > _MAX_TREE_DEPTH:
        _refuse("identity nests too deeply")
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _refuse("identity must be JSON-serialisable with finite numbers: non-finite float")
        return
    if isinstance(value, (list, tuple)):
        for v in value:
            _strict_json_tree(v, depth + 1)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if type(k) is not str:
                _refuse(f"identity must have string keys at every depth (found {type(k).__name__} key {k!r}); "
                        "numeric keys would be coerced and could collide with string keys")
            _strict_json_tree(v, depth + 1)
        return
    _refuse(f"identity must be JSON-serialisable: unsupported {type(value).__name__}")


def _canonical_identity(identity: Any) -> dict:
    if not isinstance(identity, dict):
        _refuse(f"identity must be a dict, got {type(identity).__name__}")
    _strict_json_tree(identity)
    try:
        return json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise PondCheckpointError(f"identity must be JSON-serialisable with finite numbers and string keys: {exc}") from None


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seal_manifest(manifest: dict) -> dict:
    """Return `manifest` with its own `manifest_sha256` (canonical JSON of every other key)."""
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return {**body, "manifest_sha256": _sha256_text(_json_dumps(body))}


def _canon_into(h: Any, value: Any, depth: int = 0) -> None:
    """Hash any value of the fixed vocabulary (scalars, arrays in any namespace through the counted `to_host`, dataclasses,
    enums, tuples / lists / dicts, module names, paths). Anything else is refused, never `repr`-ed."""
    from decimal import Decimal

    from maple.core.backend import is_array, to_host

    if depth > _MAX_TREE_DEPTH:
        _refuse("identity digest: value nesting too deep")

    def tag(text: str) -> None:
        h.update(text.encode("utf-8") + b"\x00")

    if value is None:
        tag("N")
    elif isinstance(value, (bool, np.bool_)):
        tag("B1" if value else "B0")
    elif isinstance(value, (int, np.integer)):
        tag("I" + str(int(value)))
    elif isinstance(value, (float, np.floating)):
        tag("F" + float(value).hex())
    elif isinstance(value, str):
        tag("S" + str(len(value)))
        h.update(value.encode("utf-8"))
    elif isinstance(value, ModuleType):
        tag("M" + value.__name__)
    elif isinstance(value, enum.Enum):
        tag("E" + type(value).__qualname__ + "." + str(value.name))
    elif isinstance(value, Decimal):
        tag("C" + str(value))
    elif isinstance(value, os.PathLike):
        tag("P" + os.fspath(value))
    elif isinstance(value, np.ndarray) or is_array(value):
        # np.array(copy=True, order="C") keeps a 0-d array 0-d (np.ascontiguousarray promotes it to shape (1,))
        host = np.array(np.asarray(to_host(value)), copy=True, order="C")
        tag(f"A{host.dtype.str}{host.shape}")
        h.update(host.tobytes())
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        tag("D" + type(value).__qualname__)
        for f in dataclasses.fields(value):
            tag("f" + f.name)
            _canon_into(h, getattr(value, f.name), depth + 1)
    elif isinstance(value, (tuple, list)):
        tag(("T" if isinstance(value, tuple) else "L") + str(len(value)))
        for item in value:
            _canon_into(h, item, depth + 1)
    elif isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            _refuse("identity digest: mapping keys must be strings")
        tag("Y" + str(len(value)))
        for key in sorted(value):
            tag("k" + key)
            _canon_into(h, value[key], depth + 1)
    else:
        _refuse(f"identity digest: unsupported value type {type(value).__qualname__}")


def _digest_of(*values: Any) -> str:
    h = hashlib.sha256(SCHEMA.encode("utf-8"))
    for value in values:
        _canon_into(h, value)
    return h.hexdigest()


def _event_checks(event: Any) -> None:
    if not isinstance(event, PondEvent):
        _refuse("event must be a PondEvent")
    if getattr(event.xp, "__name__", None) not in BACKENDS:
        _refuse(f"unsupported array namespace {event.xp!r}")


def _initial_topology(event: PondEvent) -> PondTopology:
    top = getattr(event, "_initial_topology", None)
    if not isinstance(top, PondTopology):
        _refuse("the event does not expose its initial topology (provisional PondEvent API changed)")
    return top


def _binding(event: PondEvent) -> dict:
    """Everything the continuation depends on that is NOT in the checkpoint, bound independently of
    `event.config_sha256` (which may omit native bed-context fields). All arrays hashed by value."""
    top = _initial_topology(event)
    return {
        "schema": SCHEMA,
        "event_state_schema": POND_EVENT_SCHEMA,
        "array_backend": event.xp.__name__,
        "hydraulic_backend": event.control.backend,
        "shape": [int(event.shape[0]), int(event.shape[1])],
        "n_classes": int(event.nc),
        "phase_bins": int(event.control.phase_bins),
        "event_config_sha256": event.config_sha256,
        "event_forcing_sha256": event.forcing_sha256,
        "context_sha256": _digest_of(event.context),
        "column_sha256": _digest_of(event.column),
        "sediment_sha256": _digest_of(event.sediment),
        "splash_sha256": None if event.splash is None else _digest_of(event.splash),
        "schedule_sha256": _digest_of(event.schedule),
        "options_sha256": _digest_of(event.veg, event.rain_scale, event.control, event.policy),
        "initial_topology": {"identity_sha256": top.identity_sha256, "source_sha256": top.source_sha256,
                             "topology_sha256": top.topology_sha256, "generation": int(top.generation)},
    }


# --- array specification --------------------------------------------------------------------------------------------
# name -> (shape, dtype string, sign, location). sign: "nn" >= 0, "s" signed, "b" bool, "i" integer, "count" integral >= 0.
def _specs(event: PondEvent, *, dry: bool) -> dict[str, tuple]:
    ny, nx = event.shape
    nc, nb = event.nc, event.control.phase_bins
    cell, fx, fy, full = (ny, nx), (ny, nx + 1), (ny + 1, nx), (ny + 2, nx + 2)
    cc, phase = (ny, nx, nc), (ny, nx, nc, nb)
    s = {}

    def add(name, shape, dtype, sign, loc="dev"):
        s[name] = (shape, dtype, sign, loc)

    add("hyd_depth", cell, "<f8", "nn")
    add("hyd_soil", cell, "<f8", "nn")
    add("hyd_qx", fx, "<f8", "s")
    add("hyd_qy", fy, "<f8", "s")
    add("sediment_velocity", cc, "<f8", "nn")
    add("phase_fraction", phase, "<f8", "nn")
    add("phase_position", phase, "<f8", "nn")
    add("held_source", cell, "<f8", "s")
    add("datum_full", full, "<f8", "s", "host")
    add("datum_committed", cell, "<f8", "s", "host")
    add("topo_source", full, "<f8", "s", "host")
    add("topo_active", cell, "|b1", "b", "host")
    add("topo_friction", cell, "<f8", "s", "host")
    add("topo_escape_value", cell, "<f8", "s", "host")
    add("topo_escape_finite", cell, "|b1", "b", "host")
    add("topo_escape_next", cell, "<i8", "i", "host")
    add("topo_outlet_connected", cell, "|b1", "b", "host")
    add("topo_closed_label", cell, "<i4", "i", "host")
    add("topo_outlets", (None, 4), "<i8", "i", "host")
    for name in ("wl_depth0", "wl_soil0", "wl_rain", "wl_intake", "wl_sat_return", "wl_drain"):
        add(name, cell, "<f8", "nn")
    for name in ("wl_pos_x", "wl_neg_x"):
        add(name, fx, "<f8", "nn")
    for name in ("wl_pos_y", "wl_neg_y"):
        add(name, fy, "<f8", "nn")
    add("wl_net_x", fx, "<f8", "s")
    add("wl_net_y", fy, "<f8", "s")
    add("wl_export", (), "<f8", "s")
    add("sl_inventory0", (nc,), "<f8", "nn", "host")
    add("sl_mobile0", (nc,), "<f8", "nn", "host")
    for name in ("wet_requested", "wet_actual", "wet_shortfall", "splash_requested", "splash_actual", "deposited_wet",
                 "deposited_splash", "export_wet", "export_splash", "deposited_terminal"):
        add("sl_" + name, (nc,), "<f8", "nn")
    add("sl_deposit_unmet", (nc,), "<f8", "s")
    add("sl_pickup_cell", cc, "<f8", "nn")
    add("sl_deposition_cell", cc, "<f8", "nn")
    add("sl_x_gross", (ny, nx + 1, nc), "<f8", "nn")
    add("sl_y_gross", (ny + 1, nx, nc), "<f8", "nn")
    add("sl_x_net", (ny, nx + 1, nc), "<f8", "s")
    add("sl_y_net", (ny + 1, nx, nc), "<f8", "s")
    add("sl_numerical_residual", (), "<f8", "nn")
    add("sl_phase_residual_max", (), "<f8", "nn")
    add("sl_n_phase_canonicalized", (), "<f8", "count")
    add("sl_n_phase_remnants", (), "<f8", "count")
    if dry:
        add("wl_reset_depth", cell, "<f8", "nn")
        add("wl_reset_soil", cell, "<f8", "nn")
        add("terminal_deposition", cc, "<f8", "nn")
    return s


def _check_arrays(arrays: dict, specs: dict, xp: ModuleType) -> None:
    """Exact name set, namespace (host arrays NumPy, device arrays `xp`), exact ndarray type, shape, dtype, finite and sign,
    in one batched flag read."""
    from maple.core.backend import DeferredChecks, array_namespace, finite_flag, is_array, negative_flag, true_flag

    if set(arrays) != set(specs):
        _refuse(f"array set mismatch: missing {sorted(set(specs) - set(arrays))}, unexpected {sorted(set(arrays) - set(specs))}")
    checks = DeferredChecks()
    for name, (shape, dtype, sign, loc) in specs.items():
        a = arrays[name]
        ns = np if loc == "host" else xp
        if not is_array(a) or type(a) is not ns.ndarray or array_namespace(a) is not ns:
            _refuse(f"{name} must be an exact {ns.__name__}.ndarray ({loc} array), got {type(a).__name__}")
        if a.dtype.str != dtype:
            _refuse(f"{name} must have dtype {dtype}, got {a.dtype.str}")
        got = tuple(a.shape)
        if len(got) != len(shape) or any(w is not None and w != g for w, g in zip(shape, got, strict=True)):
            _refuse(f"{name} must have shape {shape}, got {got}")
        if dtype == "<f8":
            checks.require(finite_flag(a), f"{name} must be finite everywhere")
            if sign in ("nn", "count"):
                checks.forbid(negative_flag(a), f"{name} must be >= 0 everywhere")
            if sign == "count":
                checks.forbid(true_flag(ns.floor(a) != a), f"{name} must be integral")
    try:
        checks.resolve()
    except ValueError as exc:
        raise PondCheckpointError(str(exc)) from None


# --- scalar helpers ---------------------------------------------------------------------------------------------------
def _int(value: Any, name: str, lo: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < lo:
        _refuse(f"{name} must be an integer >= {lo}, got {value!r}")
    return int(value)


def _real(value: Any, name: str, lo: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        _refuse(f"{name} must be a real number, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or (lo is not None and v < lo):
        _refuse(f"{name} must be finite" + ("" if lo is None else f" and >= {lo}") + f", got {value!r}")
    return v


def _exact_int(value: Any, name: str, lo: int = 0) -> int:
    if type(value) is not int or value < lo:
        _refuse(f"{name} must be an int >= {lo}, got {value!r}")
    return value


def _exact_float(value: Any, name: str, lo: float | None = None) -> float:
    if type(value) is not float or not math.isfinite(value) or (lo is not None and value < lo):
        _refuse(f"{name} must be a finite float" + ("" if lo is None else f" >= {lo}") + f", got {value!r}")
    return value


def _hex(value: Any, name: str) -> str:
    if not (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)):
        _refuse(f"{name} must be a 64-hex SHA-256 string")
    return value


def _as_array(value: Any, xp: ModuleType) -> Any:
    """A ledger scalar reduction is a NumPy scalar (host) or 0-d array; a 0-d ndarray is what the specification checks."""
    if isinstance(value, (np.generic, int, float)) and not isinstance(value, (bool, np.bool_)):
        if xp is not np:
            _refuse("a host scalar appears in a device ledger")
        return np.asarray(value, dtype=np.float64)
    return value


# --- extraction from a state ------------------------------------------------------------------------------------------
def _topology_fields(top: PondTopology) -> dict:
    return {"shape": tuple(top.shape), "dx_m": top.dx_m, "source_z_full": top.source_z_full, "active": top.active,
            "friction": top.friction, "nodata_value": top.nodata_value, "outlets": tuple(top.outlets),
            "escape_level_m": top.escape_level_m, "escape_path_next": top.escape_path_next,
            "outlet_connected": top.outlet_connected, "closed_component_label": top.closed_component_label,
            "n_closed_components": top.n_closed_components, "generation": top.generation,
            "identity_sha256": top.identity_sha256, "source_sha256": top.source_sha256,
            "topology_sha256": top.topology_sha256, "geometry_input_sha256": top.geometry.input_sha256}


def _fields_mismatch(a: dict, b: dict) -> list[str]:
    bad = []
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            same = (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.dtype == y.dtype and x.shape == y.shape
                    and bool(np.array_equal(x, y)))
        else:
            same = x == y
        if not same:
            bad.append(key)
    return bad


def _extract(state: PondEventState, handoff: PondDryHandoff | None, xp: ModuleType) -> tuple[dict, dict]:
    """Logical array map and JSON scalars of a state (no copy, no conversion beyond 0-d normalisation)."""
    if {f.name for f in dataclasses.fields(PondQuiet)} != {"since_s", "last_t_s", "window_depth_change_m", "ready"}:
        _refuse("PondQuiet fields changed; this checkpoint schema must be adapted (no field is silently dropped)")
    hyd, top, wl, sl = state.hydraulic, state.topology, state.water, state.sediment
    finite =np.isfinite(top.escape_level_m)
    arrays = {
        "hyd_depth": hyd.depth_m, "hyd_soil": hyd.soil_water_m, "hyd_qx": hyd.qx_m2_s, "hyd_qy": hyd.qy_m2_s,
        "sediment_velocity": state.sediment_velocity_m_s, "phase_fraction": state.phase_fraction,
        "phase_position": state.phase_position_m, "held_source": state.held_source_m,
        "datum_full": state.datum.initial_full_m, "datum_committed": state.datum.initial_committed_m,
        "topo_source": top.source_z_full, "topo_active": top.active, "topo_friction": top.friction,
        "topo_escape_value": np.where(finite, top.escape_level_m, 0.0), "topo_escape_finite": finite,
        "topo_escape_next": top.escape_path_next, "topo_outlet_connected": top.outlet_connected,
        "topo_closed_label": top.closed_component_label,
        "topo_outlets": np.array([o.as_tuple() for o in top.outlets], dtype=np.int64).reshape(-1, 4),
        "wl_depth0": wl.depth0, "wl_soil0": wl.soil0, "wl_rain": wl.rain, "wl_intake": wl.intake,
        "wl_sat_return": wl.sat_return, "wl_drain": wl.drain, "wl_pos_x": wl.pos_x, "wl_neg_x": wl.neg_x,
        "wl_net_x": wl.net_x, "wl_pos_y": wl.pos_y, "wl_neg_y": wl.neg_y, "wl_net_y": wl.net_y,
        "wl_export": _as_array(wl.export, xp), "sl_inventory0": sl.inventory0, "sl_mobile0": sl.mobile0,
        "sl_numerical_residual": _as_array(sl.numerical_residual_kg, xp),
        "sl_phase_residual_max": _as_array(sl.phase_residual_max_kg, xp),
        "sl_n_phase_canonicalized": _as_array(sl.n_phase_canonicalized, xp),
        "sl_n_phase_remnants": _as_array(sl.n_phase_remnants, xp),
    }
    for name in ("wet_requested", "wet_actual", "wet_shortfall", "splash_requested", "splash_actual", "deposited_wet",
                 "deposited_splash", "export_wet", "export_splash", "deposited_terminal", "deposit_unmet", "pickup_cell",
                 "deposition_cell", "x_gross", "y_gross", "x_net", "y_net"):
        arrays["sl_" + name] = getattr(sl, name)
    if handoff is not None:
        arrays["wl_reset_depth"], arrays["wl_reset_soil"] = wl.reset_depth, wl.reset_soil
        arrays["terminal_deposition"] = handoff.terminal_deposition_kg
    c, q = state.counters, state.quiet
    scalars = {
        "t_s": _real(hyd.t_s, "t_s", 0.0), "next_dt_cap_s": None if hyd.next_dt_cap_s is None else _real(hyd.next_dt_cap_s, "next_dt_cap_s"),
        "status": state.status, "state_schema": state.schema, "config_sha256": state.config_sha256,
        "forcing_sha256": state.forcing_sha256,
        "topology": {"generation": _int(top.generation, "generation"), "identity_sha256": top.identity_sha256,
                     "source_sha256": top.source_sha256, "topology_sha256": top.topology_sha256,
                     "n_closed_components": _int(top.n_closed_components, "n_closed_components"),
                     "dx_m": _real(top.dx_m, "dx_m"),
                     "nodata_value": None if top.nodata_value is None else _real(top.nodata_value, "nodata_value"),
                     "shape": [int(top.shape[0]), int(top.shape[1])]},
        "water": {"t0_s": _real(wl.t0_s, "water.t0_s", 0.0), "t_end_s": _real(wl.t_end_s, "water.t_end_s", 0.0),
                  "n_steps": _int(wl.n_steps, "water.n_steps")},
        "sediment": {"n_native_calls": _int(sl.n_native_calls, "n_native_calls")},
        "counters": {k: _int(getattr(c, k), k) if kind is int else _real(getattr(c, k), k, 0.0)
                     for k, kind in _counter_kinds().items()},
        "commit_log": _plain_encode(tuple(c.commit_log)),
        "quiet": {"since_s": None if q.since_s is None else _real(q.since_s, "quiet.since_s"),
                  "last_t_s": None if q.last_t_s is None else _real(q.last_t_s, "quiet.last_t_s"),
                  "window_depth_change_m": _real(q.window_depth_change_m, "quiet.window_depth_change_m", 0.0),
                  "ready": _bool(q.ready, "quiet.ready")},
    }
    return arrays, scalars


def _counter_kinds() -> dict[str, type]:
    """Counter fields of the ACTUAL `PondCounters` (so a field the event author adds is carried, not silently dropped): `int`
    or `float` by the type of the field's default; `commit_log` is handled separately; any other kind is refused."""
    kinds: dict[str, type] = {}
    for f in dataclasses.fields(PondCounters):
        if f.name == "commit_log":
            continue
        default = f.default
        if type(default) is int:
            kinds[f.name] = int
        elif type(default) is float:
            kinds[f.name] = float
        else:
            _refuse(f"PondCounters.{f.name} has a default of type {type(default).__name__}; this checkpoint schema carries "
                    "only int and float counters (adaptation needed)")
    required = {"n_accepted_steps", "n_commits", "n_maple_trigger_commits", "n_net_elevation_commits", "n_final_commits",
                "n_refreshes", "n_splash_exchanges", "n_rejected_attempts"}
    if not required <= set(kinds):
        _refuse(f"PondCounters lacks fields this module validates: {sorted(required - set(kinds))}")
    return kinds


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        _refuse(f"{name} must be a bool, got {value!r}")
    return bool(value)


# --- plain (array-free) tree codec: accounting and commit log -------------------------------------------------------
def _plain_encode(value: Any, depth: int = 0, budget: list | None = None) -> Any:
    """Whitelisted vocabulary only: None, bool, int, finite float, str, tuple, list, str-keyed dict, `PondQuiet`. NumPy scalars
    become Python scalars; no array, no other object, no non-finite number."""
    budget = [0] if budget is None else budget
    budget[0] += 1
    if depth > _MAX_TREE_DEPTH or budget[0] > _MAX_TREE_NODES:
        _refuse("accounting tree too deep or too large")
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _refuse("non-finite number in the accounting (JSON Infinity / NaN is not allowed)")
        return value
    if isinstance(value, tuple):
        return {"tuple": [_plain_encode(v, depth + 1, budget) for v in value]}
    if isinstance(value, list):
        return {"list": [_plain_encode(v, depth + 1, budget) for v in value]}
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            _refuse("accounting mapping keys must be strings")
        return {"dict": {k: _plain_encode(v, depth + 1, budget) for k, v in value.items()}}
    if isinstance(value, PondQuiet):
        return {"quiet": {f.name: _plain_encode(getattr(value, f.name), depth + 1, budget)
                          for f in dataclasses.fields(PondQuiet)}}
    _refuse(f"unsupported accounting value {type(value).__qualname__}")
    return None


def _plain_decode(value: Any, depth: int = 0, budget: list | None = None) -> Any:
    budget = [0] if budget is None else budget
    budget[0] += 1
    if depth > _MAX_TREE_DEPTH or budget[0] > _MAX_TREE_NODES:
        _refuse("accounting tree too deep or too large")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _refuse("non-finite number in the accounting")
        return value
    if not isinstance(value, dict) or len(value) != 1:
        _refuse("malformed accounting node")
    (key, body), = value.items()
    if key in ("tuple", "list"):
        if not isinstance(body, list):
            _refuse("malformed accounting sequence")
        items = [_plain_decode(v, depth + 1, budget) for v in body]
        return tuple(items) if key == "tuple" else items
    if key == "dict":
        if not isinstance(body, dict):
            _refuse("malformed accounting mapping")
        return {k: _plain_decode(v, depth + 1, budget) for k, v in body.items()}
    if key == "quiet":
        names = {f.name for f in dataclasses.fields(PondQuiet)}
        if not isinstance(body, dict) or set(body) != names:
            _refuse("malformed quiet record")
        v = {k: _plain_decode(x, depth + 1, budget) for k, x in body.items()}
        for k in ("since_s", "last_t_s"):
            if v[k] is not None and (type(v[k]) is not float):
                _refuse(f"quiet.{k} must be None or a float")
        if type(v["window_depth_change_m"]) is not float or type(v["ready"]) is not bool:
            _refuse("quiet record types invalid")
        return PondQuiet(**v)
    _refuse(f"unknown accounting node {key!r}")
    return None


# --- native MAPLE node codec (ledger / committed topography): fixed type whitelist -----------------------------------
def _native_registry() -> dict:
    from maple.core.types.sediment_ledger import CompactSedimentLedgerState, SedimentLedgerState
    from maple.core.types.topographic_commit import CommittedTopographyState

    return {t.__name__: t for t in (SedimentLedgerState, CompactSedimentLedgerState, CommittedTopographyState)}


def _native_encode(value: Any, add_array: Any, depth: int = 0) -> Any:
    from maple.core.backend import is_array, to_host

    if depth > _MAX_TREE_DEPTH:
        _refuse("native node too deep")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray) or is_array(value):
        host = np.array(np.asarray(to_host(value)), copy=True, order="C")  # exact shape: 0-d stays 0-d
        if not _dtype_ok(host.dtype.str) or host.ndim > _MAX_NDIM:
            _refuse(f"native arrays must be little-endian numeric / bool data of at most 5 dimensions, got {host.dtype.str}")
        if host.dtype.kind == "f" and not np.isfinite(host).all():
            _refuse("native arrays must be finite")
        return {"array": add_array(host)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if _native_registry().get(name) is not type(value):
            _refuse(f"unregistered native type {name}")
        return {"type": name, "fields": {f.name: _native_encode(getattr(value, f.name), add_array, depth + 1)
                                         for f in dataclasses.fields(value)}}
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            _refuse("native mapping keys must be strings")
        return {"dict": {k: _native_encode(v, add_array, depth + 1) for k, v in value.items()}}
    if isinstance(value, (tuple, list)):
        return {("tuple" if isinstance(value, tuple) else "list"): [_native_encode(v, add_array, depth + 1) for v in value]}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _refuse("non-finite native scalar")
        return value
    _refuse(f"unsupported native field {type(value).__qualname__}")
    return None


def _native_decode(value: Any, get_array: Any, depth: int = 0) -> Any:
    if depth > _MAX_TREE_DEPTH:
        _refuse("native node too deep")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _refuse("non-finite native scalar")
        return value
    if not isinstance(value, dict):
        _refuse("malformed native node")
    keys = set(value)
    if keys == {"array"}:
        return get_array(value["array"])
    if keys == {"type", "fields"}:
        cls = _native_registry().get(value["type"]) if isinstance(value["type"], str) else None
        if cls is None or not isinstance(value["fields"], dict):
            _refuse("unknown native type or fields")
        if set(value["fields"]) != {f.name for f in dataclasses.fields(cls)}:
            _refuse(f"native fields of {value['type']} do not match the bound MAPLE type")
        try:
            return cls(**{k: _native_decode(v, get_array, depth + 1) for k, v in value["fields"].items()})
        except PondCheckpointError:
            raise
        except Exception as exc:  # the MAPLE constructor's own validation
            raise PondCheckpointError(f"native {value['type']} is invalid: {exc}") from None
    if keys == {"dict"} and isinstance(value["dict"], dict):
        return {k: _native_decode(v, get_array, depth + 1) for k, v in value["dict"].items()}
    if keys in ({"tuple"}, {"list"}):
        key = next(iter(keys))
        if not isinstance(value[key], list):
            _refuse("malformed native sequence")
        items = [_native_decode(v, get_array, depth + 1) for v in value[key]]
        return tuple(items) if key == "tuple" else items
    _refuse("unknown native value encoding")
    return None


# --- independent validation (shared by save and load) ---------------------------------------------------------------
def _rebuild_topology(event: PondEvent, source_full: np.ndarray, generation: int) -> PondTopology:
    init = _initial_topology(event)
    try:
        fresh = build_pond_topology(np.array(source_full, dtype=np.float64, copy=True), init.active, init.friction, init.dx_m,
                                    init.outlets, nodata_value=init.nodata_value,
                                    allow_closed_components=init.n_closed_components > 0)
    except Exception as exc:
        raise PondCheckpointError(f"the saved held source does not rebuild a valid pond topology: {exc}") from None
    return dataclasses.replace(fresh, generation=generation)


def _native_checks(bed: BedState, event: PondEvent, t_s: float) -> None:
    """The native MAPLE validators on the bed (in its namespace), including the ledger reconciliation at the state clock."""
    from maple.core.types.topographic_commit import validate_committed_topography_state
    from maple.coupling.sediment_ledger.validation import validate_ledger_state
    from maple.surface.active_layer.validation import check_active_layer_voxel_partition
    from maple.surface.availability.validation import check_sediment_availability_state
    from maple.surface.topographic_commit.commit import (
        _reconcile_ledger_with_physical_state,
        resolve_reconciliation_reference_inventory_kg,
        validate_reconciliation_context,
    )
    from maple.water import validate_water_state

    ctx = event.context
    shape = (ctx.geometry.ny, ctx.geometry.nx, len(ctx.grain_classes.classes))
    try:
        require_supported_bed(bed, ctx)
        check_active_layer_voxel_partition(bed.active_layer, bed.voxel_column, ctx.geometry, ctx.mass_resolution_kg)
        check_sediment_availability_state(bed.sediment_availability, bed.active_layer, ctx.mass_resolution_kg)
        if validate_ledger_state(bed.ledger) != shape or validate_committed_topography_state(bed.committed_topography) != shape:
            _refuse("native ledger / committed topography geometry mismatch")
        validate_water_state(bed.water, *shape)
        validate_reconciliation_context(
            bed.ledger, bed.voxel_column, bed.active_layer, ctx.geometry, committed_state=bed.committed_topography,
            reconciliation_baseline=None, current_time_s=t_s, operation="SYRUP pond checkpoint", ledger_shape=shape)
        reference, label = resolve_reconciliation_reference_inventory_kg(bed.committed_topography, None)
        _reconcile_ledger_with_physical_state(bed.ledger, reference, bed.voxel_column, bed.active_layer,
                                              ctx.mass_resolution_kg, reference_label=label, ledger_shape=shape)
    except PondCheckpointError:
        raise
    except Exception as exc:
        raise PondCheckpointError(f"native MAPLE bed validation failed: {exc}") from None


def _validate_commit_log(log: Any, event: PondEvent, counters: dict, generation: int) -> None:
    if not isinstance(log, tuple):
        _refuse("commit_log must be a tuple")
    cap = event.control.max_commit_log
    if len(log) != min(counters["n_commits"], cap):
        _refuse("commit_log length disagrees with the commit counter and the log cap")
    refreshed = 0
    for i, rec in enumerate(log):
        if not isinstance(rec, dict) or set(rec) != _COMMIT_KEYS:
            _refuse(f"commit_log[{i}] keys mismatch")
        _real(rec["t_s"], "commit.t_s", 0.0)
        for k in ("gate_before_m", "net_after_m", "threshold_m"):
            _real(rec[k], "commit." + k, 0.0)
        if (rec["reason"] not in _COMMIT_REASONS or not isinstance(rec["committed"], (bool, np.bool_)) or not rec["committed"]
                or not isinstance(rec["refreshed"], (bool, np.bool_))):
            _refuse(f"commit_log[{i}] reason / flags invalid")
        if _int(rec["generation"], "commit.generation") > generation:
            _refuse(f"commit_log[{i}] generation exceeds the state's")
        refreshed += int(bool(rec["refreshed"]))
    if refreshed > counters["n_refreshes"]:
        _refuse("commit_log records more refreshes than the counter")


def _validate_scalars(event: PondEvent, state: PondEventState, scalars: dict, handoff: PondDryHandoff | None) -> None:
    """Counter / clock / source consistency (every relation is an invariant of the accepted step of `PondEvent`)."""
    c, wl, q, top = scalars["counters"], scalars["water"], scalars["quiet"], state.topology
    t = scalars["t_s"]
    n_steps = c["n_accepted_steps"]
    if n_steps != wl["n_steps"]:
        _refuse("accepted-step counter disagrees with the water ledger step count")
    if wl["t_end_s"] != t or wl["t0_s"] > t or (n_steps == 0) != (wl["t0_s"] == t):
        _refuse("water ledger clock disagrees with the state clock")
    if c["n_refreshes"] != top.generation or c["n_refreshes"] > c["n_commits"]:
        _refuse("refresh counter disagrees with the topology generation / commits")
    if c["n_maple_trigger_commits"] + c["n_net_elevation_commits"] + c["n_final_commits"] > c["n_commits"]:
        _refuse("commit reason counters exceed the commit counter")
    if c["n_splash_exchanges"] > n_steps or (event.splash is None and c["n_splash_exchanges"]):
        _refuse("splash exchange counter inconsistent")
    base = 2 * n_steps + 2 * c["n_splash_exchanges"]
    calls = scalars["sediment"]["n_native_calls"]
    if (calls != base) if handoff is None else (calls not in (base, base + 1)):
        _refuse("native call counter disagrees with the accepted steps")
    _validate_commit_log(state.counters.commit_log, event, c, top.generation)
    if handoff is not None:
        if q != {"since_s": None, "last_t_s": None, "window_depth_change_m": 0.0, "ready": False}:
            _refuse("a completed state must carry the reset quiet monitor")
        return
    if (q["last_t_s"] is None) != (n_steps == 0) or (q["last_t_s"] is not None and q["last_t_s"] != t):
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


def _reset_receipts(state: PondEventState, event: PondEvent) -> dict:
    """`{"surface"|"soil": (volume_m3, bound_m3)}` of the recorded dry-reset removal, recomputed on the host from the saved
    arrays. The bound is MAPLE's `volume_roundoff_bound_m3` for the cell-reduction term count (+ the area product and the
    final combination) at the real operand scale `area * sum|x|`, doubled because it compares two independent reductions
    (the event's device sum and this host sum)."""
    from maple.core.backend import to_host

    from maple_syrup.conservation import volume_roundoff_bound_m3

    area = state.topology.dx_m ** 2
    out = {}
    for name, array in (("surface", state.water.reset_depth), ("soil", state.water.reset_soil)):
        a = np.asarray(to_host(array), dtype=np.float64)
        scale = float(area * np.abs(a).sum())
        out[name] = (float(area * a.sum()), 2.0 * volume_roundoff_bound_m3(int(a.size) + 2, scale))
    return out


def _validate_handoff(event: PondEvent, state: PondEventState, handoff: PondDryHandoff, arrays: dict) -> None:
    from maple.core.backend import array_namespace, to_bool, to_host
    from maple.surface.topographic_commit.commit import ledger_is_empty

    from maple_syrup.sediment_bed import bed_already_committed

    xp, ctx = event.xp, event.context
    if handoff.status != "complete" or state.status != "dry":
        _refuse("a handoff must be complete with a dry state")
    wl = state.water
    for pre, reset, name in ((handoff.pre_reset_surface_m, wl.reset_depth, "surface"), (handoff.pre_reset_soil_m, wl.reset_soil, "soil")):
        if array_namespace(pre) is not xp or not to_bool(xp.array_equal(pre, reset)):
            _refuse(f"pre-reset {name} arrays disagree with the recorded dry-reset removal")
    h = state.hydraulic
    if any(to_bool(xp.any(a != 0)) for a in (h.depth_m, h.soil_water_m, h.qx_m2_s, h.qy_m2_s, state.sediment_velocity_m_s,
                                             state.bed.water.depth_m, state.bed.water.mobile_mass_by_cell_class_kg)):
        _refuse("the completed state is not dry and free of mobile sediment")
    if not ledger_is_empty(state.bed.ledger) or not bed_already_committed(state.bed, ctx, state.t_s):
        _refuse("the completed state's MAPLE ledger is not empty / committed at the state clock")
    acc = handoff.accounting
    if not isinstance(acc, dict) or set(acc) != _ACCOUNTING_KEYS:
        _refuse("handoff accounting keys mismatch")
    if not isinstance(acc["policy"], str) or not isinstance(acc["geometry_refreshed_at_final"], (bool, np.bool_)):
        _refuse("handoff accounting types invalid")
    # Receipts are compared with MAPLE's FP64 summation policy (`conservation`): the bound comes from the real operands and
    # the honest reduction term count of the two independent sums being compared. There is no absolute or relative floor,
    # so a corruption of a tiny reset volume is not hidden by a tolerance larger than the quantity.
    receipts = _reset_receipts(state, event)
    for key, name in (("surface_removed_m3", "surface"), ("soil_removed_m3", "soil")):
        got = _real(acc[key], key, 0.0)
        value, bound = receipts[name]
        if abs(got - value) > bound:
            _refuse(f"accounting {key} = {got!r} disagrees with the recorded reset depths ({value!r}, summation bound {bound!r})")
    from maple_syrup.conservation import reservoir_bound_kg

    terminal_cells = np.asarray(to_host(handoff.terminal_deposition_kg), dtype=np.float64)
    terminal = terminal_cells.sum(axis=(0, 1))
    listed = acc["terminal_deposition_by_class_kg"]
    ledger_terminal = np.asarray(to_host(state.sediment.deposited_terminal), dtype=np.float64)
    n_cells = int(terminal_cells.shape[0] * terminal_cells.shape[1])
    bound = 2.0 * reservoir_bound_kg(n_cells, terminal)  # the per-class totals dominate every partial sum (non-negative cells)
    if (not isinstance(listed, list) or len(listed) != event.nc
            or np.any(np.abs(np.asarray([_real(v, "terminal deposition") for v in listed]) - terminal) > bound)
            or np.any(np.abs(ledger_terminal - terminal) > bound)):
        _refuse("accounting terminal deposition disagrees with the terminal deposition array / ledger "
                f"(MAPLE summation bound {bound!r} kg)")
    for key in ("pre_terminal_audit", "final_audit"):
        a = acc[key]
        if not isinstance(a, dict) or a.get("water", {}).get("closes") is not True or a.get("sediment", {}).get("closes") is not True:
            _refuse(f"accounting {key} does not record closed budgets")
    del arrays


def _validate_state(event: PondEvent, state: Any, handoff: PondDryHandoff | None) -> tuple[dict, dict, PondTopology]:
    """FULL independent validation of a state (or the dry state of a handoff). Never calls `event.step`, never consults the
    event's runtime cache for its own checks. Returns `(arrays, scalars, fresh_topology)`."""
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
    if fresh.identity_sha256 != init.identity_sha256 or not np.array_equal(arrays["datum_full"], init.source_z_full):
        _refuse("the state does not belong to the event's ORIGINAL initial topology / source datum")
    if arrays["datum_committed"].shape != tuple(event.shape):
        _refuse("datum committed elevation shape mismatch")

    # hydraulics against the faces of the fresh topology (a freshly built solver, not the event cache)
    try:
        pond = build_pond_hydraulics(fresh, event.column, backend=event.control.backend, control=event.control.hydraulic)
        pond.solver.validate_state(state.hydraulic)
    except Exception as exc:
        raise PondCheckpointError(f"hydraulic state is invalid for the rebuilt faces: {exc}") from None
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
    _native_checks(bed, event, scalars["t_s"])
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

    _validate_scalars(event, state, scalars, handoff)
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


# --- files ---------------------------------------------------------------------------------------------------------------
def _hash_file(path: Path, max_bytes: int) -> tuple[str, int]:
    """SHA-256 of a regular, non-symlink file, reading at most `max_bytes + 1` bytes."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PondCheckpointError(f"cannot open {path.name}: {exc}") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            _refuse(f"{path.name} is not a regular file")
        if info.st_size > max_bytes:
            _refuse(f"{path.name} ({info.st_size} bytes) exceeds the checkpoint size bound {max_bytes}")
        hasher, read = hashlib.sha256(), 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            read += len(chunk)
            if read > max_bytes:
                _refuse(f"{path.name} grew beyond the checkpoint size bound")
            hasher.update(chunk)
    finally:
        os.close(fd)
    return hasher.hexdigest(), read


def _zip_audit(path: Path, max_bytes: int, *, budget_used: int = 0) -> dict[str, int]:
    """Central-directory audit WITHOUT expanding anything: stored members only, plain `*.npy` names, no duplicates, no
    encryption; the summed declared uncompressed bytes (plus `budget_used`) must fit `max_bytes`."""
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
    except (zipfile.BadZipFile, OSError) as exc:
        raise PondCheckpointError(f"{path.name} is not a valid archive: {exc}") from None
    sizes: dict[str, int] = {}
    total = budget_used
    for info in infos:
        name = info.filename
        if (info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1 or not name.endswith(".npy")
                or "/" in name or "\\" in name or name.startswith(".") or name in sizes):
            _refuse(f"{path.name} member {name!r} is not an allowed stored .npy array")
        sizes[name] = int(info.file_size)
        total += int(info.file_size)
        if total > max_bytes:
            _refuse(f"checkpoint exceeds the bounded uncompressed size {max_bytes} bytes (checked from the archive directory, "
                    "nothing was expanded)")
    return sizes


_MAX_NPY_HEADER = 4096
_SNAPSHOT_KINDS = "fiubSU"  # the native snapshot also stores string metadata; never object / void (structured) data


def _npy_preflight(path: Path, sizes: dict[str, int], table: dict | None) -> None:
    """Bounded header audit of EVERY `.npy` member BEFORE anything is expanded (`np.load` / `load_state_snapshot`).

    Reads only the magic and the header (at most `_MAX_NPY_HEADER` bytes, through numpy's literal-only header parser; no
    pickle, no `eval`), then requires `header end + shape-product * itemsize == the archive's declared member size` exactly,
    so a header that demands a huge array cannot allocate: the declared size has already been bounded by `max_bytes`. For the
    companion (`table` given) the header dtype / shape must also equal the manifest table and the order must be C; object and
    void dtypes are refused everywhere. 0-d, empty and string-metadata members are handled by the same arithmetic."""
    from numpy.lib import format as npf

    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                with archive.open(info) as member:
                    version = npf.read_magic(member)
                    if version == (1, 0):
                        shape, fortran, dtype = npf.read_array_header_1_0(member, max_header_size=_MAX_NPY_HEADER)
                    elif version in ((2, 0), (3, 0)):
                        shape, fortran, dtype = npf.read_array_header_2_0(member, max_header_size=_MAX_NPY_HEADER)
                    else:
                        _refuse(f"{path.name}:{info.filename}: unsupported .npy version {version}")
                    offset = member.tell()
                if dtype.hasobject or dtype.kind not in _SNAPSHOT_KINDS:
                    _refuse(f"{path.name}:{info.filename}: dtype {dtype} is not allowed (no object / structured data)")
                if not isinstance(shape, tuple) or len(shape) > 8 or any(type(n) is not int or n < 0 for n in shape):
                    _refuse(f"{path.name}:{info.filename}: malformed .npy shape {shape!r}")
                expected = offset + math.prod(shape) * dtype.itemsize
                if expected != sizes[info.filename]:
                    _refuse(f"{path.name}:{info.filename}: .npy header (shape {shape}, dtype {dtype}) needs {expected} bytes but the "
                            f"archive member holds {sizes[info.filename]}; refused before any allocation")
                if table is not None:
                    rec = table.get(info.filename[: -len(".npy")])
                    if rec is None or fortran or dtype.str != rec["dtype"] or list(shape) != rec["shape"]:
                        _refuse(f"{path.name}:{info.filename}: .npy header differs from the manifest table or is not C-ordered")
    except PondCheckpointError:
        raise
    except (ValueError, EOFError, OSError, zipfile.BadZipFile, SyntaxError, TypeError, KeyError) as exc:
        raise PondCheckpointError(f"{path.name}: malformed .npy header or archive: {exc}") from None


def _max_bytes(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        _refuse(f"max_bytes must be a positive integer byte bound, got {value!r}")
    return int(value)


def _tree_nbytes(value: Any) -> int:
    from maple.core.backend import is_array

    if isinstance(value, np.ndarray) or is_array(value):
        return int(value.nbytes)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sum(_tree_nbytes(getattr(value, f.name)) for f in dataclasses.fields(value))
    return 0


def _read_manifest(path: Path) -> dict:
    meta = path / MANIFEST_NAME
    try:
        info = os.lstat(meta)
    except OSError as exc:
        raise PondCheckpointError(f"checkpoint manifest is missing: {exc}") from None
    if not stat.S_ISREG(info.st_mode):
        _refuse("checkpoint manifest is not a regular file")
    if info.st_size > MAX_MANIFEST_BYTES:
        _refuse("checkpoint manifest exceeds the bounded size")

    def no_constants(token: str) -> Any:
        raise ValueError(f"non-finite JSON constant {token}")

    def no_duplicates(pairs: list) -> dict:
        out: dict = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"duplicate manifest key {k!r}")
            out[k] = v
        return out

    try:
        with meta.open("rb") as handle:
            data = json.loads(handle.read(MAX_MANIFEST_BYTES + 1).decode("utf-8"), parse_constant=no_constants,
                              object_pairs_hook=no_duplicates)
    except (ValueError, UnicodeDecodeError, RecursionError, OSError) as exc:
        raise PondCheckpointError(f"checkpoint manifest is not strict JSON: {exc}") from None
    if not isinstance(data, dict):
        _refuse("checkpoint manifest must be a JSON object")
    return data


# --- save ----------------------------------------------------------------------------------------------------------------
def save_pond_checkpoint(path: Any, state_or_handoff: Any, event: PondEvent, identity: dict, *,
                         max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES) -> Path:
    """Write a NEW checkpoint directory `path` for a wet `PondEventState` or a completed `PondDryHandoff`.

    Everything is validated BEFORE the first byte is written (the same independent validation `load_pond_checkpoint` runs on
    what it reads); nothing is committed, rerouted, reset or finalised. The directory is staged as a hidden sibling and
    renamed into place; on any failure only that staging directory is removed. `path` must not exist and its parent must."""
    from maple.core.backend import to_host
    from maple.io.outputs.snapshot import save_state_snapshot

    _event_checks(event)
    limit = _max_bytes(max_bytes)
    ident = _canonical_identity(identity)
    target = Path(path)
    if os.path.lexists(target):
        _refuse(f"checkpoint path already exists (no overwrite): {target}")
    parent = target.resolve().parent
    if not parent.is_dir():
        _refuse(f"the parent directory of the checkpoint does not exist: {parent}")
    target = parent / target.name

    if isinstance(state_or_handoff, PondDryHandoff):
        handoff, state, status = state_or_handoff, state_or_handoff.dry_state, "complete"
    elif isinstance(state_or_handoff, PondEventState) and state_or_handoff.status != "wet":
        _refuse(f"a bare {state_or_handoff.status!r} state is not a PondDryHandoff: the pre-reset arrays, terminal deposition and "
                "accounting of the completion are lost without it; save the handoff returned by finalize / complete")
    elif isinstance(state_or_handoff, PondEventState):
        handoff, state, status = None, state_or_handoff, "wet"
    else:
        _refuse("save needs a PondEventState or a PondDryHandoff (a bare dry state is not a handoff and is refused)")
    arrays, scalars, _fresh = _validate_state(event, state, handoff)

    table: dict[str, dict] = {}
    host: dict[str, np.ndarray] = {}

    def add_host(name: str, array: Any) -> str:
        h = np.array(np.asarray(to_host(array) if not isinstance(array, np.ndarray) else array), copy=True, order="C")
        if h.shape != tuple(arrays[name].shape if name in arrays else h.shape):
            _refuse(f"{name}: host copy changed the array shape")  # a 0-d ledger scalar must stay 0-d
        if not _dtype_ok(h.dtype.str) or h.ndim > _MAX_NDIM:
            _refuse(f"{name}: dtype {h.dtype.str} / ndim {h.ndim} not allowed")
        host[name] = h
        table[name] = {"dtype": h.dtype.str, "shape": [int(n) for n in h.shape]}
        return name

    for name, array in arrays.items():
        add_host(name, array)

    def add_native(array: np.ndarray) -> str:
        name = f"native_{len(host):04d}"
        host[name] = array
        table[name] = {"dtype": array.dtype.str, "shape": [int(n) for n in array.shape]}
        return name

    native = {"ledger": _native_encode(state.bed.ledger, add_native),
              "committed_topography": _native_encode(state.bed.committed_topography, add_native)}
    payload = {"scalars": scalars, "arrays": table, "native": native,
               "accounting": None if handoff is None else _plain_encode(handoff.accounting)}
    binding = _binding(event)
    manifest = {"schema": SCHEMA, "identity": ident, "status": status, "array_backend": event.xp.__name__,
                "hydraulic_backend": event.control.backend, "source": pond_checkpoint_source_pair(), "binding": binding,
                "binding_sha256": _sha256_text(_json_dumps(binding)), "files": {}, "payload": payload}
    try:
        _json_dumps(manifest)
    except (TypeError, ValueError) as exc:
        raise PondCheckpointError(f"manifest is not strict JSON: {exc}") from None
    bed = state.bed
    estimate = (sum(a.nbytes for a in host.values()) + _tree_nbytes(bed.voxel_column) + _tree_nbytes(bed.active_layer)
                + _tree_nbytes(bed.sediment_availability) + _tree_nbytes(bed.water))
    if estimate > limit:
        _refuse(f"checkpoint needs about {estimate} uncompressed bytes, above max_bytes = {limit}; nothing was written "
                "(the bound is a caller-explicit resource control and is never raised automatically)")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=parent))
    try:
        provenance = {"syrup_pond_checkpoint_schema": SCHEMA, "requires_companion": True, "identity": ident,
                      "binding_sha256": manifest["binding_sha256"], "array_backend": event.xp.__name__,
                      "hydraulic_backend": event.control.backend, "status": status}
        save_state_snapshot(bed.voxel_column, bed.active_layer, scalars["t_s"], staging / PAYLOADS[0],
                            step=scalars["counters"]["n_accepted_steps"],
                            reason="SYRUP pond checkpoint component: companion continuation file required",
                            geometry=event.context.geometry, mass_resolution_kg=event.context.mass_resolution_kg,
                            sediment_availability=bed.sediment_availability, water=bed.water, provenance=provenance)
        np.savez(staging / PAYLOADS[1], **host)
        total = 0
        audited = {}
        for name in PAYLOADS:
            audited[name] = _zip_audit(staging / name, limit, budget_used=total)
            total += sum(audited[name].values())
        _npy_preflight(staging / PAYLOADS[1], audited[PAYLOADS[1]], table)
        _npy_preflight(staging / PAYLOADS[0], audited[PAYLOADS[0]], None)
        files = {name: _hash_file(staging / name, limit)[0] for name in PAYLOADS}
        sealed = _seal_manifest({**manifest, "files": files})
        _write_manifest(staging / MANIFEST_NAME, sealed)
        if os.path.lexists(target):
            _refuse("checkpoint destination appeared during the write; nothing was overwritten")
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def _write_manifest(path: Path, manifest: dict) -> None:
    text = json.dumps(manifest, indent=1, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


# --- load ----------------------------------------------------------------------------------------------------------------
def load_pond_checkpoint(path: Any, event: PondEvent, expected_identity: dict, *, allow_complete: bool = False,
                         max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES):
    """Restore a checkpoint into the namespace of `event` (see the module docstring for the order of checks). Returns a wet
    `PondEventState`, or a `PondDryHandoff` for a completed checkpoint when `allow_complete=True` (it is data: its dry reset
    is never replayed). `event` must be a fresh `PondEvent` built from the ORIGINAL initial topology and the same static
    inputs; the state is bound to its held source on first use. Nothing is committed, rerouted, reset or finalised."""
    from maple.core.backend import to_device
    from maple.io.outputs.snapshot import load_state_snapshot

    _event_checks(event)
    limit = _max_bytes(max_bytes)
    if not isinstance(allow_complete, bool):
        _refuse("allow_complete must be a bool")
    expected = _canonical_identity(expected_identity)
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        _refuse(f"checkpoint path is not a plain directory: {root}")
    root = root.resolve()
    if {p.name for p in root.iterdir()} != {MANIFEST_NAME, *PAYLOADS}:
        _refuse("checkpoint directory must contain exactly the manifest and the two payload files")

    # -- 1/2: manifest and every binding, BEFORE any payload is opened --
    data = _read_manifest(root)
    if data.get("schema") in ("maple-syrup-checkpoint/1", "maple-syrup-checkpoint/2"):
        _refuse("this is an event checkpoint of the accepted checkpoint module, not a pond checkpoint")
    if set(data) != MANIFEST_KEYS:
        _refuse(f"manifest keys mismatch: missing {sorted(MANIFEST_KEYS - set(data))}, unknown {sorted(set(data) - MANIFEST_KEYS)}")
    if data["schema"] != SCHEMA:
        _refuse(f"checkpoint schema {data['schema']!r} is not {SCHEMA!r}")
    if _hex(data["manifest_sha256"], "manifest_sha256") != _seal_manifest(data)["manifest_sha256"]:
        _refuse("manifest hash mismatch (the manifest was modified)")
    if data["identity"] != expected:
        _refuse("checkpoint case / source / config identity differs from the expected identity")
    if data["source"] != pond_checkpoint_source_pair():
        _refuse("source pair mismatch: the imported maple_syrup / maple packages differ from the ones that wrote the "
                "checkpoint (refused before any payload is read)")
    if data["array_backend"] not in BACKENDS or data["hydraulic_backend"] not in ("numpy", "numba", "cuda"):
        _refuse("manifest records an unsupported backend")
    if data["array_backend"] != event.xp.__name__:
        _refuse(f"cross-backend load refused: written by the {data['array_backend']!r} array backend, event is "
                f"{event.xp.__name__!r} (cross-backend migration is not qualified)")
    if data["hydraulic_backend"] != event.control.backend:
        _refuse(f"hydraulic backend mismatch: written with {data['hydraulic_backend']!r}, event uses "
                f"{event.control.backend!r} (numpy and numba are not interchangeable here)")
    if data["status"] not in STATUSES:
        _refuse(f"unknown checkpoint status {data['status']!r}")
    if data["status"] == "complete" and not allow_complete:
        _refuse("a completed event cannot resume or repeat its dry reset (load with allow_complete=True to read the handoff)")
    binding = _binding(event)
    if data["binding"] != binding or data["binding_sha256"] != _sha256_text(_json_dumps(binding)):
        diff = sorted(k for k in binding if not isinstance(data["binding"], dict) or data["binding"].get(k) != binding[k])
        _refuse(f"event binding mismatch (static inputs differ from the ones that wrote the checkpoint): {diff}")
    payload = data["payload"]
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        _refuse("payload keys mismatch")
    files = data["files"]
    if not isinstance(files, dict) or set(files) != set(PAYLOADS):
        _refuse("payload file set mismatch")

    # -- 3: payload hashes, archive directory, declared array table vs the bound --
    for name in PAYLOADS:
        if _hash_file(root / name, limit)[0] != _hex(files[name], f"files[{name}]"):
            _refuse(f"checkpoint payload hash mismatch: {name}")
    total, member_sizes = 0, {}
    for name in PAYLOADS:
        sizes = _zip_audit(root / name, limit, budget_used=total)
        total += sum(sizes.values())
        member_sizes[name] = sizes
    specs = _specs(event, dry=data["status"] == "complete")
    table = payload["arrays"]
    if not isinstance(table, dict) or not set(specs) <= set(table):
        _refuse("array table is missing required arrays")
    declared = 0
    for name, rec in table.items():
        if not isinstance(name, str) or not isinstance(rec, dict) or set(rec) != {"dtype", "shape"}:
            _refuse("malformed array table entry")
        if not _dtype_ok(rec["dtype"]) or not isinstance(rec["shape"], list) or len(rec["shape"]) > _MAX_NDIM \
                or any(type(n) is not int or n < 0 for n in rec["shape"]):
            _refuse(f"array {name}: dtype / shape not allowed")
        count = math.prod(rec["shape"])
        nbytes = count * np.dtype(rec["dtype"]).itemsize
        declared += nbytes
        if declared > limit:
            _refuse("declared array bytes exceed the bounded uncompressed size")
        size = member_sizes[PAYLOADS[1]].get(name + ".npy")
        if size is None or not (nbytes < size <= nbytes + 4096):
            _refuse(f"array {name}: archive member size does not match the declared dtype / shape")
        if name in specs:
            shape, dtype, _sign, _loc = specs[name]
            got = tuple(rec["shape"])
            if rec["dtype"] != dtype or len(got) != len(shape) or any(w is not None and w != g for w, g in zip(shape, got, strict=True)):
                _refuse(f"array {name}: declared dtype / shape differ from the schema (expected {dtype} {shape}, got {rec['dtype']} {got})")
    if set(member_sizes[PAYLOADS[1]]) != {n + ".npy" for n in table}:
        _refuse("companion archive members differ from the array table")
    # bounded header audit of every member of BOTH archives (native snapshot included) before anything is expanded
    _npy_preflight(root / PAYLOADS[1], member_sizes[PAYLOADS[1]], table)
    _npy_preflight(root / PAYLOADS[0], member_sizes[PAYLOADS[0]], None)

    # -- 4: native snapshot, then the companion through the fixed whitelist --
    try:
        snapshot = load_state_snapshot(root / PAYLOADS[0])
    except Exception as exc:
        raise PondCheckpointError(f"MAPLE snapshot cannot be loaded: {exc}") from None
    if not snapshot["validated"] or not snapshot["restartable"] or snapshot["is_diagnostic"]:
        _refuse("invalid MAPLE component snapshot (not validated / restartable, or diagnostic)")
    try:
        prov = json.loads(snapshot["provenance_json"])
    except ValueError:
        prov = None
    if (not isinstance(prov, dict) or prov.get("syrup_pond_checkpoint_schema") != SCHEMA
            or prov.get("binding_sha256") != data["binding_sha256"] or prov.get("identity") != data["identity"]
            or prov.get("status") != data["status"] or prov.get("array_backend") != data["array_backend"]
            or prov.get("hydraulic_backend") != data["hydraulic_backend"] or prov.get("requires_companion") is not True):
        _refuse("MAPLE snapshot provenance is not bound to this manifest")
    if (snapshot["water"] is None or snapshot["sediment_availability"] is None or snapshot["history_store_path"]
            or any(snapshot.get(k) is not None for k in _FOREIGN_SNAPSHOT_FIELDS)):
        _refuse("MAPLE snapshot lacks water / availability or carries foreign optional state")

    xp = event.xp
    used: set[str] = set()
    try:
        archive = np.load(root / PAYLOADS[1], allow_pickle=False)
    except Exception as exc:
        raise PondCheckpointError(f"companion cannot be opened: {exc}") from None
    with archive:
        if set(archive.files) != set(table):
            _refuse("companion arrays differ from the array table")

        def get_array(name: Any) -> np.ndarray:
            if not isinstance(name, str) or name not in table:
                _refuse(f"unknown array reference {name!r}")
            if name in used:
                _refuse(f"aliased array {name}")
            used.add(name)
            rec = table[name]
            try:
                a = np.array(archive[name], copy=True, order="C")
            except (ValueError, OSError, zipfile.BadZipFile, EOFError) as exc:
                raise PondCheckpointError(f"array {name} cannot be read: {exc}") from None
            if a.dtype.str != rec["dtype"] or list(a.shape) != rec["shape"]:
                _refuse(f"array {name}: stored dtype / shape differ from the table")
            if a.dtype.kind == "f" and not np.isfinite(a).all():
                _refuse(f"array {name} holds non-finite values")
            a.flags.writeable = False
            return a

        if not isinstance(payload["native"], dict) or set(payload["native"]) != {"ledger", "committed_topography"}:
            _refuse("native companion keys mismatch")
        host_arrays = {name: get_array(name) for name in specs}
        ledger = _native_decode(payload["native"]["ledger"], get_array)
        committed = _native_decode(payload["native"]["committed_topography"], get_array)
    if used != set(table):
        _refuse("unused or missing companion arrays")

    # -- 5: reconstruction in the event namespace, then the SAME independent validation the writer runs --
    def place(name: str) -> Any:
        a = host_arrays[name]
        return to_device(a, xp) if xp.__name__ == "cupy" else np.array(a, copy=True)

    def host(name: str) -> np.ndarray:
        return np.array(host_arrays[name], copy=True)

    scalars = payload["scalars"]
    state = _build_state(event, scalars, host_arrays, place, host, snapshot, ledger, committed, data["status"])
    handoff = None
    if data["status"] == "complete":
        handoff = PondDryHandoff(
            dry_state=state, pre_reset_surface_m=place("wl_reset_depth"), pre_reset_soil_m=place("wl_reset_soil"),
            terminal_deposition_kg=place("terminal_deposition"), accounting=_plain_decode(payload["accounting"]),
            status="complete")
    elif payload["accounting"] is not None:
        _refuse("a wet checkpoint carries no accounting")
    _arrays, check_scalars, _fresh = _validate_state(event, state, handoff)
    if _json_dumps(check_scalars) != _json_dumps(scalars):
        _refuse("scalars of the reconstructed state differ from the saved payload")
    return handoff if handoff is not None else state


def _build_state(event: PondEvent, scalars: Any, host_arrays: dict, place: Any, host: Any, snapshot: dict, ledger: Any,
                 committed: Any, status: str) -> PondEventState:
    """Typed reconstruction. Every scalar is checked for exact keys and types before an object is constructed."""
    from maple.core.backend import to_device_tree

    from maple_syrup.experimental_hydrology import HydraulicState

    xp = event.xp
    keys = {"t_s", "next_dt_cap_s", "status", "state_schema", "config_sha256", "forcing_sha256", "topology", "water",
            "sediment", "counters", "commit_log", "quiet"}
    if not isinstance(scalars, dict) or set(scalars) != keys:
        _refuse("scalar keys mismatch")
    t_s = _exact_float(scalars["t_s"], "t_s", 0.0)
    cap = scalars["next_dt_cap_s"]
    if cap is not None:
        cap = _exact_float(cap, "next_dt_cap_s")
        if cap <= 0.0:
            _refuse("next_dt_cap_s must be positive")
    if scalars["status"] != ("wet" if status == "wet" else "dry") or scalars["state_schema"] != POND_EVENT_SCHEMA:
        _refuse("state status / schema mismatch")
    if _hex(scalars["config_sha256"], "config_sha256") != event.config_sha256 \
            or _hex(scalars["forcing_sha256"], "forcing_sha256") != event.forcing_sha256:
        _refuse("state digests differ from the event's")

    tp = scalars["topology"]
    if not isinstance(tp, dict) or set(tp) != {"generation", "identity_sha256", "source_sha256", "topology_sha256",
                                               "n_closed_components", "dx_m", "nodata_value", "shape"}:
        _refuse("topology scalar keys mismatch")
    generation = _exact_int(tp["generation"], "generation")
    saved = {
        "shape": tuple(tp["shape"]) if isinstance(tp["shape"], list) else None, "dx_m": _exact_float(tp["dx_m"], "dx_m"),
        "source_z_full": host_arrays["topo_source"], "active": host_arrays["topo_active"],
        "friction": host_arrays["topo_friction"],
        "nodata_value": None if tp["nodata_value"] is None else _exact_float(tp["nodata_value"], "nodata_value"),
        "outlets": None, "escape_level_m": np.where(host_arrays["topo_escape_finite"], host_arrays["topo_escape_value"], np.inf),
        "escape_path_next": host_arrays["topo_escape_next"], "outlet_connected": host_arrays["topo_outlet_connected"],
        "closed_component_label": host_arrays["topo_closed_label"],
        "n_closed_components": _exact_int(tp["n_closed_components"], "n_closed_components"), "generation": generation,
        "identity_sha256": _hex(tp["identity_sha256"], "identity_sha256"), "source_sha256": _hex(tp["source_sha256"], "source_sha256"),
        "topology_sha256": _hex(tp["topology_sha256"], "topology_sha256"), "geometry_input_sha256": None,
    }
    fresh = _rebuild_topology(event, host_arrays["topo_source"], generation)
    fresh_fields = _topology_fields(fresh)
    saved["outlets"] = fresh_fields["outlets"]  # compared through the identity digest and the saved outlet array below
    saved["geometry_input_sha256"] = fresh_fields["geometry_input_sha256"]
    bad = _fields_mismatch(saved, fresh_fields)
    outlets = np.array([o.as_tuple() for o in fresh.outlets], dtype=np.int64).reshape(-1, 4)
    if bad or not np.array_equal(outlets, host_arrays["topo_outlets"]):
        _refuse(f"saved topology differs from an independent build of the saved held source in {bad or ['outlets']}")

    wl, sd, cn, qt = scalars["water"], scalars["sediment"], scalars["counters"], scalars["quiet"]
    if not isinstance(wl, dict) or set(wl) != {"t0_s", "t_end_s", "n_steps"}:
        _refuse("water scalar keys mismatch")
    if not isinstance(sd, dict) or set(sd) != {"n_native_calls"}:
        _refuse("sediment scalar keys mismatch")
    kinds = _counter_kinds()
    counter_ints = tuple(k for k, kind in kinds.items() if kind is int)
    counter_floats = tuple(k for k, kind in kinds.items() if kind is float)
    if not isinstance(cn, dict) or set(cn) != set(kinds):
        _refuse("counter keys mismatch")
    if not isinstance(qt, dict) or set(qt) != {"since_s", "last_t_s", "window_depth_change_m", "ready"}:
        _refuse("quiet keys mismatch")
    for k in ("since_s", "last_t_s"):
        if qt[k] is not None:
            _exact_float(qt[k], "quiet." + k)
    _exact_float(qt["window_depth_change_m"], "quiet.window_depth_change_m", 0.0)
    if type(qt["ready"]) is not bool:
        _refuse("quiet.ready must be a bool")
    log = _plain_decode(scalars["commit_log"])
    counters = PondCounters(
        **{k: _exact_int(cn[k], k) for k in counter_ints}, **{k: _exact_float(cn[k], k, 0.0) for k in counter_floats},
        commit_log=log)
    if not isinstance(log, tuple):
        _refuse("commit_log must be a tuple")
    quiet = PondQuiet(since_s=qt["since_s"], last_t_s=qt["last_t_s"], window_depth_change_m=qt["window_depth_change_m"],
                      ready=qt["ready"])

    hydraulic = HydraulicState(t_s, place("hyd_depth"), place("hyd_soil"), place("hyd_qx"), place("hyd_qy"), cap)
    water = WaterLedger(
        t0_s=_exact_float(wl["t0_s"], "water.t0_s", 0.0), t_end_s=_exact_float(wl["t_end_s"], "water.t_end_s", 0.0),
        depth0=place("wl_depth0"), soil0=place("wl_soil0"), rain=place("wl_rain"), intake=place("wl_intake"),
        sat_return=place("wl_sat_return"), drain=place("wl_drain"), pos_x=place("wl_pos_x"), pos_y=place("wl_pos_y"),
        neg_x=place("wl_neg_x"), neg_y=place("wl_neg_y"), net_x=place("wl_net_x"), net_y=place("wl_net_y"),
        export=place("wl_export"), n_steps=_exact_int(wl["n_steps"], "water.n_steps"),
        reset_depth=place("wl_reset_depth") if status == "complete" else None,
        reset_soil=place("wl_reset_soil") if status == "complete" else None)
    vec = ("wet_requested", "wet_actual", "wet_shortfall", "splash_requested", "splash_actual", "deposited_wet",
           "deposited_splash", "export_wet", "export_splash", "deposited_terminal", "deposit_unmet", "pickup_cell",
           "deposition_cell", "x_gross", "y_gross", "x_net", "y_net")
    sediment = SedimentLedger(
        inventory0=host("sl_inventory0"), mobile0=host("sl_mobile0"), **{k: place("sl_" + k) for k in vec},
        numerical_residual_kg=place("sl_numerical_residual"), phase_residual_max_kg=place("sl_phase_residual_max"),
        n_phase_canonicalized=place("sl_n_phase_canonicalized"), n_phase_remnants=place("sl_n_phase_remnants"),
        n_native_calls=_exact_int(sd["n_native_calls"], "n_native_calls"))

    # the bed: ONE dense authority from the native snapshot + the companion's pending ledger / committed topography
    bed = BedState(voxel_column=snapshot["voxel_column"], active_layer=snapshot["active_layer"], water=snapshot["water"],
                   ledger=ledger, committed_topography=committed, sediment_availability=snapshot["sediment_availability"])
    if xp.__name__ == "cupy":
        bed = to_device_tree(bed, xp)  # counted MAPLE transfer of the whole bed tree
    if snapshot["time_s"] != t_s or snapshot["step"] != counters.n_accepted_steps:
        _refuse("MAPLE snapshot clock / step differ from the companion")

    datum = PondDatum(initial_full_m=host("datum_full"), initial_committed_m=host("datum_committed"))
    return PondEventState(
        hydraulic=hydraulic, bed=bed, sediment_velocity_m_s=place("sediment_velocity"),
        phase_fraction=place("phase_fraction"), phase_position_m=place("phase_position"), topology=fresh,
        held_source_m=place("held_source"), datum=datum, water=water, sediment=sediment, counters=counters, quiet=quiet,
        config_sha256=event.config_sha256, forcing_sha256=event.forcing_sha256, status=scalars["status"],
        schema=POND_EVENT_SCHEMA)
