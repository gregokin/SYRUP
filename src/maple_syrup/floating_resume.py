"""R1a: genuine IN-MEMORY capture / restore of a continuing FLOATING Pond state (CPU/host; NumPy and Numba owners; Dense and published genuine Windowed beds). NOT a file format, NOT a physical checkpoint.

    cap   = capture_floating_pond(event, state, available_bytes_fn=...)        # from the ACTUAL validated parent of this event/vertical
    state2 = restore_floating_pond(other_event, cap, available_bytes_fn=...)   # into ANOTHER actual PondEvent with the identical static config/source

What is captured (sealed copies, host `(L, N)` / `(N,)` float64): the CURRENT water `W`, the persistent Green-Ampt history `J` and the lifetime matrix intake `I`, the CURRENT canonical depth `D`, the soil clock, and the
HELD law context facts (the prepared field-capacity / saturation maps, the depth `D0` they were prepared with, the layer-0 conductivity record, the stable root column / matrix operator / preferential identities, the
property-source facts: owner generation, native source witness, config identity). The parent `PondEventState` (hydraulic state, native bed, topology, held source, datum, ledgers, counters, quiet monitor, phase memory) is
referenced, not copied: Pond states are immutable values (every step returns a new state) and a test proves a continued restored state leaves the captured parent bit for bit unchanged.

What restore does (all refusable work first, ONE registration last): verifies the producer/source binding; verifies that the target vertical's HELD template equals the captured held facts bit for bit (it never
reads a texture, never calls the property provider and never derives a property from the current, eroded/deposited bed); builds the CURRENT payload on the public floating factories: an official geometry view of the held
root column at the CURRENT depth `D` (`fsl.floating_geometry_view`; none when `D == D0`), `fsl.floating_continued_state(view, W, t)` -- so `W` is validated against the CURRENT geometry, never against the original `D0`
capacity (after a deposition `W` may legitimately exceed what the original depth could hold) --, a history created on the held operator and rebound (`fmi.rebind_matrix_geometry_view`), and the preferential operator
rebound to the view. The target vertical registers that payload only through `FloatingPondVertical.adopt_resumed`, which demands a genuinely minted capture. The restored `PondEventState` is validated by the event's full
`_check_state` and `audit`. No owner is minted, no water is zeroed, no clock advances, no E/T/drainage/diffusion is added.

Integrity (correction01). The capture's containers are read-only mappings of read-only array copies and its identity is bound to an INTEGRITY DIGEST of every captured value (all scalars, identities, the source pair, every array's dtype/shape/bytes and the identity of the referenced parent state), held in a module-private weak table (never a field of the record); it is re-verified before every use, so replacing a field, a mapping entry or an array's bytes through ordinary or low-level API is refused. The parent `PondEventState` is referenced, not deeply sealed: Pond states are immutable VALUES by the code-base convention (honest bounded limit, proven only by the hash tests).
Adoption. A resumed payload reaches the vertical only through a minted, one-shot, vertical-bound `_PreparedResume` record whose parts are re-verified exactly against the capture and the held template (foreign column/law/history/state are refused); the event validates the candidate through `FloatingPondVertical.resume_scope` (provisional, no registry mutation) and `adopt_resumed` registers LAST.

Trust boundary. A capture is minted ONLY here, from a validated parent; it is an opaque, owned, uncopyable (no `copy`, `deepcopy`, pickle, `dataclasses.replace`) in-process record with a private construction window. A later
file codec must own its own array/source verification and mint through a legitimate factory; nothing here is a `VerifiedCheckpoint`, a signature or durable. `summary()` is descriptive metadata, never proof.
Stable (content) identities: the root column, matrix operator and preferential `sha256`; NOT durable: the current view / operator generation tokens (never compared or hashed into any configuration).
Limits: R1b dirty tracking is not here (the source facts a dirty set needs are recorded: generation and source witness); Windowed requires a PUBLISHED column (a pending deep journal is refused: publish through the event's
own storage policy first); CUDA record unsupported: capture/restore serve the CPU (NumPy/Numba) owners only and a CUDA vertical is refused explicitly (the floating Pond itself has a CUDA backend, but this in-memory record, its host arrays and its hashes do not). Explicit `available_bytes_fn` is required for both calls (no unlimited reader).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from maple_syrup import floating_matrix_infiltration as fmi
from maple_syrup import floating_pond as fp
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers as fsl

SCHEMA = "maple_syrup.floating_resume_capture/1"
PRODUCER = "maple_syrup.floating_pond.FloatingPondVertical"
CPU_BACKENDS = ("numpy", "numba")  # the only owners this in-memory record supports
HASH_ALLOWANCE_BYTES = 1 << 16  # the incremental integrity digest keeps no copy of any array (memoryview over validated C-contiguous FP64 bytes); this is the declared interpreter allowance
# COLD source-provenance staging (correction02). The FIRST use in a process hashes the imported `maple_syrup` and `maple` package trees (`provenance.source_tree_digest`): one read buffer of up to
# `provenance._READ_CHUNK_BYTES` (1 MiB, an unmodelled `bytes` object) plus the walked file lists/paths/metadata. Measured (tracemalloc, caches empty, 146 + native files): 1.48 MB, i.e. 1 MiB of buffer plus
# ~0.43 MB of metadata (~0.8 kB per file). The allowance is the chunk size (read from `provenance`, so it follows a change) plus 2 MiB of metadata headroom (about 5x the measured walk). It is charged ONLY when a
# process-local source cache is still empty (a warm process is not charged, nothing is charged twice) and it is part of the SAME admission as the owned arrays; nothing is pre-warmed and the source guard is unchanged.
COLD_METADATA_ALLOWANCE_BYTES = 2 << 20

__all__ = ["FloatingPondCapture", "FloatingResumeError", "capture_floating_pond", "mark_consumed", "prepared_parts", "require_capture", "restore_floating_pond"]


class FloatingResumeError(RuntimeError):
    """Typed refusal of a floating capture/restore (nothing is mutated; nothing is downgraded)."""


_MINT = threading.local()
_MINTED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # capture -> integrity digest (private; never a field of the record)
_PREPARED: weakref.WeakSet = weakref.WeakSet()


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


def _cold_source_bytes() -> int:
    """Conservative bytes of the first-use source hash (module comment), 0 when both process-local source caches are already populated."""
    from maple_syrup import provenance
    from maple_syrup import sediment_bed as sb

    try:
        from maple_syrup import pond_checkpoint as pc

        warm = "pair" in pc._SOURCE_CACHE and "digest" in sb._MAPLE_DIGEST_CACHE
    except AttributeError:  # a changed cache layout is treated as cold (conservative)
        warm = False
    return 0 if warm else int(provenance._READ_CHUNK_BYTES) + COLD_METADATA_ALLOWANCE_BYTES


def _seal(a: Any) -> np.ndarray:
    out = np.array(a, dtype=np.float64, copy=True, order="C")
    out.flags.writeable = False
    return out


def _reserve(value: Any) -> int:
    """Strict non-Boolean integer >= 0 (the existing `_count` rule), BEFORE any availability read, value check or copy."""
    try:
        return fsl._count(value, "reserve_bytes")
    except fsl.FloatingSoilError as exc:
        raise FloatingResumeError(str(exc)) from None


_FIELDS = ("schema", "producer", "vertical_identity", "event_config_sha256", "forcing_sha256", "backend", "bed_storage", "shape", "n_layers", "soil_t_s", "root_column_sha256", "root_matrix_op_sha256",
           "root_pref_sha256", "owner_config_sha256", "generation", "source_witness")
_CURRENT = ("water_m", "wetting_m", "cumulative_intake_m", "depth_m")


_HELD_NAMES = ("theta_field_capacity", "theta_saturation", "depth0_m")


def _check_layout(capture: Any) -> None:
    """EXACT metadata of every captured container/array BEFORE any byte is read, hashed or viewed: types, key sets, dtype, shape, C-contiguity. Nothing is converted or copied and no `__array__` callback can run;
    a malformed capture gets a typed refusal."""
    try:
        ny, nx = capture.shape
        L, N = int(capture.n_layers), int(ny) * int(nx)
    except (TypeError, ValueError):
        raise FloatingResumeError("the capture's shape/layers are malformed (integrity)") from None
    scalars = (("schema", str), ("producer", str), ("vertical_identity", str), ("event_config_sha256", str), ("forcing_sha256", str), ("backend", str), ("bed_storage", str), ("root_column_sha256", str),
               ("root_matrix_op_sha256", str), ("owner_config_sha256", str), ("source_witness", str), ("generation", int), ("soil_t_s", float), ("n_layers", int))
    for name, kind in scalars:
        if type(getattr(capture, name)) is not kind:
            raise FloatingResumeError(f"capture field {name} is malformed (integrity)")
    if capture.root_pref_sha256 is not None and type(capture.root_pref_sha256) is not str:
        raise FloatingResumeError("capture field root_pref_sha256 is malformed (integrity)")
    if type(capture.shape) is not tuple or len(capture.shape) != 2 or any(type(x) is not int for x in capture.shape):
        raise FloatingResumeError("the capture's shape is malformed (integrity)")
    pair = capture.source_pair
    if not isinstance(pair, MappingProxyType) or any(type(k) is not str or type(v) not in (str, int) for k, v in pair.items()):
        raise FloatingResumeError("the capture's source pair is malformed (integrity)")
    if set(capture.held) not in (set(_HELD_NAMES), set(_HELD_NAMES) | {"ks_m_s"}) or set(capture.current) != set(_CURRENT):
        raise FloatingResumeError("the capture's array sets are malformed (integrity)")
    expected = {("held", "theta_field_capacity"): (L, N), ("held", "theta_saturation"): (L, N), ("held", "depth0_m"): (N,), ("held", "ks_m_s"): (N,), ("current", "water_m"): (L, N),
                ("current", "wetting_m"): (N,), ("current", "cumulative_intake_m"): (N,), ("current", "depth_m"): (N,)}
    for group in ("held", "current"):
        for name, a in getattr(capture, group).items():
            if type(a) is not np.ndarray or a.dtype != np.float64 or a.shape != expected[(group, name)] or not a.flags.c_contiguous:
                raise FloatingResumeError(f"capture array {group}.{name} is not an exact C-contiguous float64 ndarray of the declared shape (integrity)")


def _digest(capture: Any) -> str:
    """Integrity digest of EVERY captured value (scalars, identities, source pair, the held/current arrays' dtype/shape/bytes in a fixed order and the identity of the referenced parent). Requires `_check_layout` first:
    every array is then an exact C-contiguous FP64 ndarray and is hashed through a direct `memoryview` (no `ascontiguousarray`, no copy)."""
    h = hashlib.sha256()
    meta = {name: getattr(capture, name) for name in _FIELDS if name != "soil_t_s"}
    meta["soil_t_s"] = float(capture.soil_t_s).hex()
    meta["source_pair"] = {k: capture.source_pair[k] for k in sorted(capture.source_pair)}
    meta["parent_id"] = id(capture.parent)
    meta["held_keys"] = sorted(capture.held)
    meta["current_keys"] = sorted(capture.current)
    h.update(json.dumps(meta, sort_keys=True, default=repr).encode())
    for group, names in (("held", sorted(capture.held)), ("current", list(_CURRENT))):
        for name in names:
            a = getattr(capture, group)[name]
            h.update(f"{group}.{name}|{a.dtype}|{a.shape}".encode())
            h.update(memoryview(a).cast("B"))
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class FloatingPondCapture:
    """Opaque owned capture (module docstring). Created only by `capture_floating_pond`."""

    schema: str
    producer: str
    vertical_identity: str
    event_config_sha256: str
    forcing_sha256: str
    backend: str
    bed_storage: str
    source_pair: Any
    shape: tuple
    n_layers: int
    soil_t_s: float
    # held law context facts (durable identities and the held values proving the template)
    root_column_sha256: str
    root_matrix_op_sha256: str
    root_pref_sha256: str | None
    owner_config_sha256: str
    generation: int
    source_witness: str
    held: Any
    # current physical payload (sealed copies)
    current: Any
    parent: Any  # the validated parent PondEventState (immutable VALUE by convention; referenced, not deeply sealed)

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise FloatingResumeError("a FloatingPondCapture is minted only by capture_floating_pond (no direct construction or replace())")

    def __copy__(self):
        raise FloatingResumeError("a FloatingPondCapture is an owned capability and cannot be copied")

    def __deepcopy__(self, memo):
        raise FloatingResumeError("a FloatingPondCapture is an owned capability and cannot be copied")

    def __reduce_ex__(self, protocol):
        raise FloatingResumeError("a FloatingPondCapture is not serialisable (no file format in R1a)")

    def summary(self) -> dict:
        return {"schema": self.schema, "producer": self.producer, "vertical_identity": self.vertical_identity, "backend": self.backend, "bed_storage": self.bed_storage, "shape": self.shape,
                "n_layers": self.n_layers, "generation": self.generation, "t_s": self.soil_t_s, "descriptive_only": True}


def require_capture(capture: Any) -> None:
    """Genuinely minted AND intact: the recomputed integrity digest must equal the one bound at minting (refused before any source check, use or adoption)."""
    if type(capture) is not FloatingPondCapture or capture not in _MINTED:
        raise FloatingResumeError("not a genuinely minted FloatingPondCapture (a copy, plain record or reconstruction does not carry the authority)")
    if not (isinstance(capture.held, MappingProxyType) and isinstance(capture.current, MappingProxyType) and isinstance(capture.source_pair, MappingProxyType)):
        raise FloatingResumeError("the capture's containers were replaced (integrity)")
    _check_layout(capture)  # exact metadata BEFORE any data read/hash/view (no conversion, no copy)
    try:
        digest = _digest(capture)
    except (TypeError, ValueError, BufferError) as exc:
        raise FloatingResumeError(f"the capture could not be digested ({type(exc).__name__}); integrity refused") from None
    if digest != _MINTED[capture]:
        raise FloatingResumeError("the capture's contents no longer match the integrity digest bound at capture time (altered)")


def _read(fn: Callable[[], int] | None, need: int, what: str) -> None:
    if not callable(fn):
        raise FloatingResumeError("an explicit available_bytes_fn is required (no unlimited reader)")
    avail = fn()
    if isinstance(avail, (bool, np.bool_)) or not isinstance(avail, (int, np.integer)) or int(avail) < 0:
        raise FloatingResumeError(f"the available-bytes reader must return a non-Boolean integer >= 0, got {avail!r}")
    if need > int(avail):
        raise fp.nfs.NativeFloatingMemoryError(f"{what} needs an estimated {need} B but only {int(avail)} B are reported available; nothing was changed")


def _check_event(event: Any) -> Any:
    from maple_syrup.pond_event import PondEvent

    if type(event) is not PondEvent or event.vertical is None or type(event.vertical) is not fp.FloatingPondVertical:
        raise FloatingResumeError("event must be a PondEvent with a FloatingPondVertical")
    v = event.vertical
    if v.backend not in CPU_BACKENDS:  # explicit CPU tuple: `fp.SUPPORTED_BACKENDS` now also lists "cuda" (E5), which this record does not support
        raise FloatingResumeError(f"unsupported backend {v.backend!r}: CUDA record unsupported (CPU in-memory only)")
    return v


def _source_pair() -> dict:
    from maple_syrup.pond_checkpoint import pond_checkpoint_source_pair

    return pond_checkpoint_source_pair()


def capture_floating_pond(event: Any, state: Any, *, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0) -> FloatingPondCapture:
    """Capture the continuing floating state `state` of `event` (a genuine wet `PondEventState` this event's vertical issued). Pure: nothing of the event, vertical or state is mutated."""
    from maple_syrup.pond_event import PondEventError, PondEventState
    from maple_syrup.sediment_bed import is_windowed_bed, pending_journal_records

    reserve = _reserve(reserve_bytes)
    v = _check_event(event)
    if type(state) is not PondEventState or state.status != "wet":
        raise FloatingResumeError("state must be a genuine wet PondEventState")
    try:
        event._check_state(state)  # config/forcing/topology identity, namespace/shape/clock/phase and the vertical's own registry check
    except (PondEventError, fp.FloatingPondError) as exc:
        raise FloatingResumeError(f"the parent state is not valid for this event: {exc}") from None
    storage = "windowed" if is_windowed_bed(state.bed) else "dense"
    if storage != event.control.bed_storage:
        raise FloatingResumeError("the bed storage differs from the event's configured storage")
    if storage == "windowed" and pending_journal_records(state.bed) != 0:
        raise FloatingResumeError("a Windowed bed with a pending deep journal is not capturable: publish through the event's own storage policy first (no arbitrary pending trust)")
    try:
        payload = v.current_payload(state.hydraulic.soil_water_m)
    except fp.FloatingPondError as exc:
        raise FloatingResumeError(f"the state's soil water was not issued by this vertical: {exc}") from None
    L, N = v.n_layers, v.n_cells
    if float(state.hydraulic.t_s) != float(payload.state.t_s):
        raise FloatingResumeError("the hydraulic clock differs from the soil clock")
    _read(available_bytes_fn, (3 * L + 5) * N * 8 + HASH_ALLOWANCE_BYTES + _cold_source_bytes() + reserve, "floating capture (sealed copies, integrity digest and first-use source hash)")
    t = v.held_template()
    maps = t.column._maps
    held = {"theta_field_capacity": _seal(maps["theta_field_capacity"]), "theta_saturation": _seal(maps["theta_saturation"]), "depth0_m": _seal(t.depth0_m)}
    if t.ks_m_s is not None:
        held["ks_m_s"] = _seal(t.ks_m_s)
    current = {"water_m": _seal(payload.state.water_m), "wetting_m": _seal(payload.history.wetting_m), "cumulative_intake_m": _seal(payload.history.cumulative_intake_m), "depth_m": _seal(payload.depth_m)}
    # EVERY refusable computation (source hashes, imports) happens before the tight construction window; the constructor receives plain values only
    pair = MappingProxyType(dict(_source_pair()))
    held_ro, current_ro = MappingProxyType(held), MappingProxyType(current)
    pref_sha = None if t.preferential is None else t.preferential.sha256
    soil_t = float(payload.state.t_s)
    shape = tuple(v.shape)
    with _minting():
        cap = FloatingPondCapture(SCHEMA, PRODUCER, v.identity, event.config_sha256, event.forcing_sha256, v.backend, storage, pair, shape, L, soil_t, t.column.sha256, t.matrix_op.sha256, pref_sha,
                                  t.config_sha256, t.generation, t.source_witness, held_ro, current_ro, state)
    _MINTED[cap] = _digest(cap)
    return cap


@dataclass(frozen=True, eq=False)
class _PreparedParts:
    soil: np.ndarray
    column: Any
    matrix_op: Any
    history: Any
    preferential: Any
    state: Any
    depth_m: np.ndarray


@dataclass(frozen=True, eq=False)
class _PreparedResume:
    """One-shot, vertical-bound candidate payload minted ONLY by `restore_floating_pond`; the vertical accepts nothing else."""

    capture: Any
    target: Any  # weakref to the target vertical
    parts: _PreparedParts
    consumed: list

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise FloatingResumeError("a prepared resume record is minted only by restore_floating_pond")
        _PREPARED.add(self)

    def __copy__(self):
        raise FloatingResumeError("a prepared resume record cannot be copied")

    def __deepcopy__(self, memo):
        raise FloatingResumeError("a prepared resume record cannot be copied")

    def __reduce_ex__(self, protocol):
        raise FloatingResumeError("a prepared resume record is not serialisable")


def prepared_parts(prepared: Any, vertical: Any) -> _PreparedParts:
    """Exact re-verification of a prepared record against the capture and the target's held template (called by the vertical before a provisional scope and before the registration). No state is changed."""
    if type(prepared) is not _PreparedResume or prepared not in _PREPARED:
        raise FloatingResumeError("not a genuinely minted prepared resume record")
    if prepared.consumed[0]:
        raise FloatingResumeError("the prepared resume record was already adopted")
    if prepared.target() is not vertical:
        raise FloatingResumeError("the prepared resume record belongs to another vertical")
    cap = prepared.capture
    require_capture(cap)
    if cap.vertical_identity != vertical.identity:
        raise FloatingResumeError("the capture belongs to another floating vertical identity")
    p, t = prepared.parts, vertical.held_template()
    cur = cap.current
    column = p.column
    root_ok = column is t.column or (column.parent_sha256 == t.column.sha256 and column.identity_kind == "geometry_view_token")
    geom_ok = (root_ok and column.backend == t.column.backend and (column.n_layers, column.n_cells) == (t.column.n_layers, t.column.n_cells) and column.cell_area_m2 == t.column.cell_area_m2
               and column.mask_sha256 == t.column.mask_sha256 and column.edges_sha256 == t.column.edges_sha256 and np.array_equal(column._maps["soil_depth"], cur["depth_m"])
               and np.array_equal(column._maps["theta_field_capacity"], cap.held["theta_field_capacity"]) and np.array_equal(column._maps["theta_saturation"], cap.held["theta_saturation"]))
    op_ok = (p.matrix_op.column is column and p.matrix_op.column_sha256 == column.sha256 and (p.matrix_op is t.matrix_op or p.matrix_op.layouts == t.matrix_op.layouts)
             and p.history.operator_sha256 == p.matrix_op.sha256 and (column is not t.column or p.matrix_op is t.matrix_op)
             and (column is t.column or p.matrix_op._law is t.matrix_op._law))
    pref_ok = (p.preferential is None) == (t.preferential is None) and (p.preferential is None or (p.preferential.column is column and (column is t.column) == (p.preferential is t.preferential)))
    state_ok = (p.state.column_sha256 == column.sha256 and float(p.state.t_s) == cap.soil_t_s and np.array_equal(p.state.water_m, cur["water_m"]) and np.array_equal(p.history.wetting_m, cur["wetting_m"])
                and np.array_equal(p.history.cumulative_intake_m, cur["cumulative_intake_m"]) and np.array_equal(p.depth_m, cur["depth_m"])
                and np.array_equal(p.soil, np.asarray(p.state.water_m).sum(axis=0).reshape(tuple(vertical.shape))))
    if not (geom_ok and op_ok and pref_ok and state_ok):
        raise FloatingResumeError("the prepared resume parts differ from the capture / held template (foreign column, law, history, state or geometry)")
    return p


def mark_consumed(prepared: Any) -> None:
    prepared.consumed[0] = True


def prepare_model_bytes(v: Any) -> int:
    """Declared cold host model of the R1a prepare path of vertical `v` (current payload, view, hydraulic copies, integrity digest and first-use source hash); shared by restore and the file loader."""
    return prepare_model_bytes_dims(v.n_layers, v.n_cells)


def prepare_model_bytes_dims(L: int, N: int) -> int:
    """The same arithmetic from plain dimensions (no vertical needed); `prepare_model_bytes(v)` delegates here, values unchanged."""
    return (2 * L + 8) * N * 8 + 16 * N + HASH_ALLOWANCE_BYTES + _cold_source_bytes()


def _prepare(event: Any, v: Any, capture: FloatingPondCapture, available_bytes_fn: Callable[[], int], reserve: int) -> tuple[Any, Any]:
    """The ONE shared body of restore preparation (R1a, factored for the file path): verifies the target's HELD template against the capture, builds the CURRENT payload on the public floating factories (official view, continued
    state on the CURRENT geometry, history rebind), derives the soil sum, mints the one-shot target-bound prepared record and the candidate state. Registers NOTHING: the caller validates inside `resume_scope` and adopts last."""
    # held template: the target's prepared context must equal the captured held facts bit for bit (never re-derived from the current bed)
    t = v.held_template()
    maps = t.column._maps
    held = capture.held
    ok = (t.column.sha256 == capture.root_column_sha256 and t.matrix_op.sha256 == capture.root_matrix_op_sha256 and (None if t.preferential is None else t.preferential.sha256) == capture.root_pref_sha256
          and t.config_sha256 == capture.owner_config_sha256 and t.generation == capture.generation and t.source_witness == capture.source_witness
          and np.array_equal(maps["theta_field_capacity"], held["theta_field_capacity"]) and np.array_equal(maps["theta_saturation"], held["theta_saturation"])
          and np.array_equal(t.depth0_m, held["depth0_m"]) and (t.ks_m_s is None) == ("ks_m_s" not in held) and (t.ks_m_s is None or np.array_equal(t.ks_m_s, held["ks_m_s"])))
    if not ok:
        raise FloatingResumeError("the target's HELD template differs from the capture's held maps/identities/source facts (property source, generation or config)")
    cur = capture.current
    depth, w, j, i = cur["depth_m"], cur["water_m"], cur["wetting_m"], cur["cumulative_intake_m"]
    try:
        if np.array_equal(depth, t.depth0_m):
            column = t.column
        else:
            column = fsl.floating_geometry_view(t.column, np.array(depth, copy=True), available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
        state = fsl.floating_continued_state(column, w, capture.soil_t_s)  # W against the CURRENT geometry (never the original D0 capacity)
        hist0 = fmi.floating_matrix_history(t.matrix_op, wetting_m=np.array(j, copy=True), cumulative_intake_m=np.array(i, copy=True))
        if column is t.column:
            op, hist = t.matrix_op, hist0
            pref = t.preferential
        else:
            op, hist = fmi.rebind_matrix_geometry_view(t.matrix_op, column, hist0)
            pref = None if t.preferential is None else fpi.rebind_preferential_geometry_view(t.preferential, column)
    except fsl.FloatingSoilError as exc:
        raise FloatingResumeError(f"the captured water/history is not valid on the CURRENT geometry: {exc}") from None
    parent = capture.parent
    hyd = parent.hydraulic
    soil = np.asarray(state.water_m).sum(axis=0).reshape(tuple(v.shape))
    if not np.array_equal(soil, np.asarray(hyd.soil_water_m)):
        raise FloatingResumeError("the captured inventory does not reproduce the parent's derived soil water")
    new_hyd = replace(hyd, depth_m=np.array(hyd.depth_m, copy=True), soil_water_m=soil, qx_m2_s=np.array(hyd.qx_m2_s, copy=True), qy_m2_s=np.array(hyd.qy_m2_s, copy=True))
    new_state = replace(parent, hydraulic=new_hyd)
    parts = _PreparedParts(soil, column, op, hist, pref, state, _seal(depth))
    target = weakref.ref(v)
    consumed = [False]
    with _minting():  # tight window: plain, precomputed values only
        prepared = _PreparedResume(capture, target, parts, consumed)
    return prepared, new_state


def restore_floating_pond(event: Any, capture: FloatingPondCapture, *, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0) -> Any:
    """Restore `capture` into `event` (another actual `PondEvent` with the identical static configuration and held source). Returns a genuine wet `PondEventState` whose soil water is registered with `event.vertical`.
    The candidate is prepared and validated (full event `_check_state`, native water/sediment audits) through a PROVISIONAL scope; the single registration is the LAST operation, so every refusal leaves the vertical's
    registry, stats and accepted payloads and the capture untouched."""
    from maple_syrup.pond_event import PondEventError

    reserve = _reserve(reserve_bytes)
    require_capture(capture)  # minted AND intact, before any source check, use or adoption
    v = _check_event(event)
    if capture.schema != SCHEMA or capture.producer != PRODUCER:
        raise FloatingResumeError("unknown capture schema/producer")
    if (event.config_sha256 != capture.event_config_sha256 or event.forcing_sha256 != capture.forcing_sha256 or v.identity != capture.vertical_identity or v.backend != capture.backend
            or event.control.bed_storage != capture.bed_storage or tuple(v.shape) != capture.shape or v.n_layers != capture.n_layers):
        raise FloatingResumeError("the target event/vertical is not the capture's static configuration, held source or backend")
    L, N = v.n_layers, v.n_cells
    _read(available_bytes_fn, (2 * L + 8) * N * 8 + 16 * N + HASH_ALLOWANCE_BYTES + _cold_source_bytes() + reserve, "floating restore (current payload, view, hydraulic copies, integrity digest and first-use source hash)")
    if dict(_source_pair()) != dict(capture.source_pair):  # admitted ABOVE: the first-use source hash I/O is inside the declared budget
        raise FloatingResumeError("the imported source pair differs from the capture's (a configured wrong source fails)")
    prepared, new_state = _prepare(event, v, capture, available_bytes_fn, reserve)
    with v.resume_scope(prepared):  # provisional resolution: the registry is NOT touched
        try:
            event._check_state(new_state)  # full validation of a non-blessed state (registry + derived sum + bed depth == solver depth ...)
            audit = event.audit(new_state)
        except (PondEventError, fp.FloatingPondError) as exc:
            raise FloatingResumeError(f"the restored state failed the event's validation: {exc}") from None
        if not (audit["water"]["closes"] and audit["sediment"]["closes"]):
            raise FloatingResumeError("the restored state's native water/sediment audits do not close")
    v.adopt_resumed(prepared)  # the ONE registration, LAST
    return new_state


def _mint_capture_from_inputs(event: Any, inputs: Any) -> FloatingPondCapture:
    """Mint a capture from genuine verified FILE inputs (NOT from a validated live parent): the physical validation of the file's state is still PENDING at this point (it runs inside `resume_scope`, after the payload is
    provisionally resolvable). The arrays are the ALREADY-OWNED sealed (read-only, C-contiguous, integrity-digested) buffers of the verified inputs: no second copy is made, so the declared admission of the caller covers the
    only new allocations (current state, view, conversions, integrity digest, first-use source hash)."""
    from maple_syrup import floating_checkpoint as fc

    fc.require_inputs(inputs)  # genuine, unconsumed and intact (exact metadata, then the integrity digest)
    v = _check_event(event)
    held = MappingProxyType(dict(inputs.held))
    current = MappingProxyType(dict(inputs.current))
    pair = MappingProxyType(dict(inputs.source_pair))
    shape = tuple(v.shape)
    with _minting():
        cap = FloatingPondCapture(SCHEMA, PRODUCER, inputs.vertical_identity, inputs.event_config_sha256, inputs.forcing_sha256, inputs.backend, inputs.storage, pair, shape, v.n_layers, inputs.soil_t_s,
                                  inputs.root_column_sha256, inputs.root_matrix_op_sha256, inputs.root_pref_sha256, inputs.owner_config_sha256, inputs.generation, inputs.source_witness, held, current, inputs.parent)
    _MINTED[cap] = _digest(cap)
    return cap


def prepare_from_verified_inputs(event: Any, inputs: Any, *, available_bytes_fn: Callable[[], int], reserve_bytes: int = 0) -> tuple[Any, Any]:
    """The file path of the shared preparation: `(prepared, candidate_state)` from genuine verified file inputs. Order: strict reserve -> genuine/unconsumed/intact inputs (exact metadata, then the integrity digest: a streamed
    hash, no copy) -> static identity checks -> the composed PUBLIC admission from this call's own reader (the R1a prepare working set incl. the first-use source hash, one reserve) BEFORE any new allocation, the capture mint
    or the source hash -> capture -> shared preparation -> one-shot consumption (private registry) only after every refusable step succeeded. Registers nothing; the loader validates the candidate inside
    `vertical.resume_scope(prepared)` and adopts last."""
    from maple_syrup import floating_checkpoint as fc

    reserve = _reserve(reserve_bytes)
    fc.require_inputs(inputs)
    v = _check_event(event)
    if (event.config_sha256 != inputs.event_config_sha256 or event.forcing_sha256 != inputs.forcing_sha256 or v.identity != inputs.vertical_identity or v.backend != inputs.backend
            or event.control.bed_storage != inputs.storage or v.n_layers != inputs.current["water_m"].shape[0]):
        raise FloatingResumeError("the target event/vertical is not the verified file's static configuration, held source or backend")
    _read(available_bytes_fn, prepare_model_bytes(v) + reserve, "floating verified-input preparation (current payload, view, conversions, integrity digest and first-use source hash)")
    if dict(_source_pair()) != dict(inputs.source_pair):
        raise FloatingResumeError("the imported source pair differs from the file's")
    cap = _mint_capture_from_inputs(event, inputs)
    prepared, new_state = _prepare(event, v, cap, available_bytes_fn, reserve)
    fc.consume_inputs(inputs)
    return prepared, new_state
