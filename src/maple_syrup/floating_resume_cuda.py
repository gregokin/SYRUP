"""G-F2: CUDA preparation and adoption of the CURRENT floating payload from genuine verified CUDA FILE inputs (`floating_checkpoint_cuda`). AUTHOR-WRITTEN, NOT RUN ON A DEVICE (Root qualifies it on hardware). Importing this module
never imports CuPy.

The CPU `floating_resume` record (sealed HOST arrays, NumPy compares) is left byte-identical and serves the CPU owners only; this module is the separate GPU preparation. It does no physics and adds no numerical body: the CURRENT
payload is built on the SAME public, hardware-qualified factories the E5 hot path uses --

    D (current canonical depth)    uploaded in bounded counted chunks; an official geometry view of the held ROOT column at D (`fsl.floating_geometry_view`, device, current device first) when D differs from the held D0
    W (current water)              `fsl.floating_continued_state(column, W, t)`: validated against the CURRENT geometry on the device (never the original D0 capacity: after a deposition W may legitimately exceed it)
    J, I (wetting history/intake)  `fmi.floating_matrix_history(t.matrix_op, ...)` then `fmi.rebind_matrix_geometry_view` / `fpi.rebind_preferential_geometry_view` (zero-copy)

-- on top of the vertical's HELD template (the original law context: caller-built original owner vertical or `FloatingPondVertical.from_held_context`). No owner is minted, no water is zeroed, nothing is derived from the loaded
bed, no E/T/drainage/diffusion is added, no clock advances. The derived soil sum is the vertical's own `W.sum(axis=0)` and must equal the file's stored one BITWISE (same device class/CuPy build).

Trust boundary. `prepare_from_verified_inputs` consumes a genuinely minted, unconsumed, intact `VerifiedFloatingInputsCuda`, and mints a one-shot, vertical-bound `_PreparedResumeCuda` (module-private weak registry + the integrity digest
of its inputs bound at minting). `FloatingPondVertical.resume_scope/adopt_resumed` dispatch to `prepared_parts` here for a CUDA vertical: every use re-verifies the record, the intact inputs and the device parts against the sealed host
values (column/operator/history/preferential identities, shared held maps by object identity, the view depth, W, J, I and the derived soil by bounded counted chunks). The candidate is provisional during the loader's full
validation and registered LAST; a refusal anywhere leaves the vertical untouched.

Transfers (counted): H2D of W, J, I, D (chunks <= HASH_CHUNK_BYTES) once per preparation; per `prepared_parts` call (scope entry and adoption) a bounded D2H comparison of W, J, I, D (the same bytes) and one 8-byte resident
count for the soil sum; the held-template comparison streams the held maps once. Hardware peaks and device numbers are unverified."""
from __future__ import annotations

import contextlib
import threading
import weakref
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from maple_syrup import floating_matrix_infiltration as fmi
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_resume as fr
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu

__all__ = ["FloatingResumeCudaError", "mark_consumed", "prepare_from_verified_inputs", "prepare_models", "prepared_parts"]


class FloatingResumeCudaError(fr.FloatingResumeError):
    """Typed refusal of the CUDA preparation/adoption (nothing is mutated; nothing is downgraded)."""


_MINT = threading.local()
_PREPARED: weakref.WeakSet = weakref.WeakSet()
_BOUND: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # prepared -> integrity digest of its verified inputs (private; never a field of the record)


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


def _fail(message: str) -> None:
    raise FloatingResumeCudaError(message)


def prepare_models(L: int, N: int) -> tuple[int, int]:
    """`(device, host)` bytes of one preparation from plain dimensions (reserves excluded). DEVICE: the four uploads alive until the factories copied them, the retained payload (state W, derived soil, depth D), the
    history J/I, the view's private depth and flags, resident comparison temporaries, the state-check scratch and one chunk. HOST: the integrity-digest/comparison allowance, the first-use source hash, two chunks and the
    check packet."""
    from maple_syrup import floating_checkpoint_cuda as fcc

    chk = fcu.operation_terms("state_check", L, N)
    device = (L + 3) * N * 8 + (L + 2) * N * 8 + 2 * N * 8 + N * 8 + N + 2 * N * 8 + chk["scratch"] + fcc.CHUNK_BYTES
    host = fr.HASH_ALLOWANCE_BYTES + fr._cold_source_bytes() + 2 * fcc.CHUNK_BYTES + chk["host"]
    return int(device), int(host)


@dataclass(frozen=True, eq=False)
class _PreparedParts:
    soil: Any
    column: Any
    matrix_op: Any
    history: Any
    preferential: Any
    state: Any
    depth_m: Any


@dataclass(frozen=True, eq=False)
class _PreparedResumeCuda:
    """One-shot, vertical-bound candidate payload minted ONLY by `prepare_from_verified_inputs`; the CUDA vertical accepts nothing else."""

    inputs: Any
    target: Any  # weakref to the target vertical
    parts: _PreparedParts
    consumed: list

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            _fail("a CUDA prepared resume record is minted only by prepare_from_verified_inputs")
        _PREPARED.add(self)

    def __copy__(self):
        _fail("a CUDA prepared resume record cannot be copied")

    def __deepcopy__(self, memo):
        _fail("a CUDA prepared resume record cannot be copied")

    def __reduce_ex__(self, protocol):
        _fail("a CUDA prepared resume record is not serialisable")


def _is_device(a: Any, cp: Any) -> bool:
    from maple.core.backend import is_device_array

    return is_device_array(a) and type(a) is cp.ndarray


def prepared_parts(prepared: Any, vertical: Any) -> _PreparedParts:
    """Exact re-verification of a prepared record against its intact inputs and the target's HELD template (called by the vertical before a provisional scope and before the registration). No state is changed."""
    from maple_syrup import floating_checkpoint_cuda as fcc

    if type(prepared) is not _PreparedResumeCuda or prepared not in _PREPARED or prepared not in _BOUND:
        _fail("not a genuinely minted CUDA prepared resume record")
    if prepared.consumed[0]:
        _fail("the CUDA prepared resume record was already adopted")
    if prepared.target() is not vertical or vertical.backend != "cuda":
        _fail("the CUDA prepared resume record belongs to another vertical")
    inputs = prepared.inputs
    cp, _dev = fcc.gate(vertical.held_template().column)  # the CURRENT device FIRST: before any integrity hash, reader, transfer or comparison
    try:
        fcc.verify_intact(inputs)  # exact metadata, then the digest of the sealed host parts
    except fcc.FloatingCheckpointCudaError as exc:
        _fail(f"the prepared record's inputs are not intact: {exc}")
    if fcc.input_digest(inputs) != _BOUND[prepared]:
        _fail("the prepared record's inputs differ from the ones it was prepared from (altered)")
    if inputs.vertical_identity != vertical.identity:
        _fail("the verified inputs belong to another floating vertical identity")
    p, t = prepared.parts, vertical.held_template()
    cur = inputs.current
    column = p.column
    root_ok = column is t.column or (column.parent_sha256 == t.column.sha256 and column.identity_kind == "geometry_view_token")
    shared_ok = column is t.column or all(column._maps[k] is t.column._maps[k] for k in column._maps if k not in ("soil_depth", "bottom_open"))  # a view shares every static map by object identity
    geom_ok = (root_ok and shared_ok and column.backend == t.column.backend == "cuda" and (column.n_layers, column.n_cells) == (t.column.n_layers, t.column.n_cells) and column.cell_area_m2 == t.column.cell_area_m2
               and column.mask_sha256 == t.column.mask_sha256 and column.edges_sha256 == t.column.edges_sha256 and column.device_id == t.column.device_id)
    op_ok = (p.matrix_op.column is column and p.matrix_op.column_sha256 == column.sha256 and (p.matrix_op is t.matrix_op or p.matrix_op.layouts == t.matrix_op.layouts)
             and p.history.operator_sha256 == p.matrix_op.sha256 and (column is not t.column or p.matrix_op is t.matrix_op) and (column is t.column or p.matrix_op._law is t.matrix_op._law))
    pref_ok = (p.preferential is None) == (t.preferential is None) and (p.preferential is None or (p.preferential.column is column and (column is t.column) == (p.preferential is t.preferential)))
    types_ok = all(_is_device(a, cp) for a in (p.soil, p.depth_m, p.state.water_m, p.history.wetting_m, p.history.cumulative_intake_m, column._maps["soil_depth"]))
    if not (geom_ok and op_ok and pref_ok and types_ok and p.state.column_sha256 == column.sha256 and float(p.state.t_s) == inputs.soil_t_s):
        _fail("the prepared parts differ from the verified inputs / held template (foreign column, law, history, state or geometry)")
    values_ok = (fcc.stream_equal(column._maps["soil_depth"], cur["depth_m"]) and fcc.stream_equal(p.depth_m, cur["depth_m"]) and fcc.stream_equal(p.state.water_m, cur["water_m"])
                 and fcc.stream_equal(p.history.wetting_m, cur["wetting_m"]) and fcc.stream_equal(p.history.cumulative_intake_m, cur["cumulative_intake_m"]))
    if not values_ok:
        _fail("the prepared device payload differs bit for bit from the verified inputs (W, J, I, D or the view depth)")
    from maple_syrup import floating_pond_cuda as fpc

    if not fpc.scalar(cp.array_equal(p.soil, vertical._derived(p.state)).astype(np.int64)):
        _fail("the prepared derived soil water is not the layer sum of the prepared inventory")
    return p


def mark_consumed(prepared: Any) -> None:
    prepared.consumed[0] = True


def prepare_from_verified_inputs(event: Any, inputs: Any, *, quota: dict) -> tuple[Any, Any]:
    """`(prepared, candidate_state)` from genuine verified CUDA file inputs. Order: event metadata -> the CURRENT device gate -> genuine/unconsumed/intact inputs -> static identity, kernel and held-template checks -> one two-pool
    admission of `prepare_models` from the caller's quota (the factories then use the SAME readers with zero reserves) -> source-pair check -> bounded counted uploads -> public factories -> derived soil -> mint -> one-shot
    consumption (private registry) only after every refusable step. Registers nothing; the loader validates the candidate inside `vertical.resume_scope(prepared)` and adopts last."""
    from maple_syrup import floating_checkpoint_cuda as fcc
    from maple_syrup import floating_pond_cuda as fpc

    v = fcc.check_event(event)
    t = v.held_template()
    cp, dev = fcc.gate(t.column)  # FIRST, before any reader, array or transfer
    fcc.require_inputs(inputs)
    if (event.config_sha256 != inputs.event_config_sha256 or event.forcing_sha256 != inputs.forcing_sha256 or v.identity != inputs.vertical_identity or v.backend != inputs.backend
            or event.control.bed_storage != inputs.storage or v.n_layers != inputs.current["water_m"].shape[0] or v.n_cells != inputs.current["depth_m"].shape[0]):
        _fail("the target event/vertical is not the verified file's static configuration, held source or backend")
    L, N = v.n_layers, v.n_cells
    d_need, h_need = prepare_models(L, N)
    fcc.admit(cp, dev, d_need, h_need, quota, "the CUDA floating current-payload preparation (uploads, public factories, derived soil and comparison)")
    comp = fcc.component_quota(quota)
    if dict(fr._source_pair()) != dict(inputs.source_pair):
        _fail("the imported source pair differs from the file's (a configured wrong source fails)")
    if dict(inputs.kernel) != fcc._kernel():
        _fail("the kernel/ABI descriptor differs from the file's")
    facts = {"root_column_sha256": inputs.root_column_sha256, "root_matrix_op_sha256": inputs.root_matrix_op_sha256, "root_pref_sha256": inputs.root_pref_sha256, "owner_config_sha256": inputs.owner_config_sha256,
             "generation": inputs.generation, "source_witness": inputs.source_witness, "has_ks": "ks_m_s" in inputs.held, "held_layouts": dict(inputs.held_layouts)}
    fcc.check_held_ids(v, facts)
    fcc.check_held_values(v, inputs.held)  # the held bytes vs the REAL device held template (never re-derived from the bed)
    cur = inputs.current
    soil_t = inputs.soil_t_s
    try:
        d_dev = fcc.h2d(cp, cur["depth_m"])
        same = bool(fpc.scalar(cp.array_equal(d_dev, t.depth0_m).astype(np.int64)))
        column = t.column if same else fsl.floating_geometry_view(t.column, d_dev, **comp)
        w_dev = fcc.h2d(cp, cur["water_m"])
        state = fsl.floating_continued_state(column, w_dev, soil_t, **comp)  # W against the CURRENT geometry (never the original D0 capacity)
        del w_dev
        j_dev, i_dev = fcc.h2d(cp, cur["wetting_m"]), fcc.h2d(cp, cur["cumulative_intake_m"])
        hist0 = fmi.floating_matrix_history(t.matrix_op, wetting_m=j_dev, cumulative_intake_m=i_dev, **comp)
        del j_dev, i_dev
        if column is t.column:
            op, hist, pref = t.matrix_op, hist0, t.preferential
        else:
            op, hist = fmi.rebind_matrix_geometry_view(t.matrix_op, column, hist0)
            pref = None if t.preferential is None else fpi.rebind_preferential_geometry_view(t.preferential, column)
    except fsl.FloatingSoilError as exc:
        _fail(f"the file's water/history is not valid on the CURRENT geometry: {exc}")
    parent = inputs.parent
    hyd = parent.hydraulic
    soil = v._derived(state)
    if not fpc.scalar(cp.array_equal(soil, hyd.soil_water_m).astype(np.int64)):
        _fail("the stored inventory does not reproduce the file's derived soil water (bitwise; same device class and CuPy build are required)")
    new_state = replace(parent, hydraulic=replace(hyd, soil_water_m=soil))
    parts = _PreparedParts(soil, column, op, hist, pref, state, d_dev)
    target = weakref.ref(v)
    consumed = [False]
    digest = fcc.input_digest(inputs)
    with _minting():  # tight window: plain, precomputed values only
        prepared = _PreparedResumeCuda(inputs, target, parts, consumed)
    _BOUND[prepared] = digest
    fcc.consume_inputs(inputs)
    return prepared, new_state

