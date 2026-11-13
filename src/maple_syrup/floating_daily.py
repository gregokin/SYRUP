"""G3 S1: the genuine native FLOATING DAILY lineage of an INITIALIZED soil (CPU/host; NumPy and Numba owners; Dense/Windowed bed REFERENCED, never copied or changed).

SCOPE (explicit, honest). The lineage starts from a genuinely freshly initialized, generation-0 `native_floating_soil.NativeFloatingSoil` and the actual `NativeBed` it was initialized from (cold witness equality), with NO
surface-water carry (H0 = 0: a soil-only start; no Pond/wind state is accepted and no storm is run). It covers COMPLETED no-rain and small-recharge days. `P > T`, and a recharge whose pulse does not fit an active closed
column, return the explicit `requires_full_storm` outcome from the UNCHANGED parent with the FULL native forcing (`recharge.DailyRainPlan.bind_forcing`): nothing is applied, no E/T/time/cursor moves, and no storm is
executed here. The event-end refresh / current-W handoff, the Pond/wind/day-after-storm coupling and the physical (file) checkpoint are later slices (S4-S7); this module is NOT a Pond-continuous model.

Single water authority. The state's `FloatingSoilState` (W) and `FloatingMatrixHistory` (J, lifetime intake I) are the ONLY current water/history; the owner is a HISTORICAL initial context and is retired LAST through
`native_floating_soil.retire_native_owner_to_lineage` (after every refusable construction) so it can never be staged/adopted against its stale W0. The held template (root column, evaporation operator, sealed zero arrays) is
water-free and shared by every state; no `NativeFloatingSoil` is manufactured per day.

Transactions (pure; the caller's accepted state controls the lineage; `expected_parent` is REQUIRED on all of them, minted by `expect_parent(state)` for the exact instance):
  start_floating_daily_lineage(owner, bed, ...)           -> boundary state (a fresh `recharge.DailyCursor(run_digest)`; the new profile's own nominal clock; no historical cursor)
  floating_daily_action(boundary, plan, PET, cover, ...)   -> `DailyResult`: `interevent` (ONE 86400 s interevent: qualified `floating_evaporation` E-only staging, then `fsl.floating_step` / `floating_step_with_recharge` once; no
                                                              extra E/drainage/diffusion/dt; matrix/preferential are NOT called (zero pool: J/I stay EXACT); a direct recharge is a simplified pulse, never matrix intake)
                                                              or `requires_full_storm`
  request_floating_withdrawal(interevent, request, expected_sequence) -> accepted/unmet/scale from the actual `floating_transpiration.floating_withdraw` (aggregate (L,N), no compensation, zero-time)
  close_floating_day(interevent, output=None)              -> closed boundary (the day is final ONLY here, after the actual T, consuming the cursor once) and, if configured, the every-N output notice
Physical clock advances `DAY_S` once per day (daily action); the nominal ordinal counts one completed day per final close. Unexpected surface storage (any positive surface return) refuses the candidate before acceptance
(nothing is rounded to zero). Identities: `execution_digest` is a deterministic EXECUTION-RECEIPT lineage digest chained over accepted transactions (explicitly NOT a water content hash); run-local capabilities are never durable.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import threading
import weakref
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from maple_syrup import floating_daily_cuda as fdc
from maple_syrup import floating_evaporation as fe
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_transpiration as ft
from maple_syrup import native_floating_soil as nfs
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = ["DailyCloseResult", "DailyLedger", "DailyRefusal", "DailyResult", "ExpectedParent", "FloatingDailyError", "FloatingDailyState", "WithdrawalOutcome", "aggregate_sparse_demand",
           "close_floating_day", "expect_parent", "floating_daily_action", "request_floating_withdrawal", "start_floating_daily_lineage"]

SCHEMA = "maple_syrup.floating_daily/1"
PROFILE = "native_floating_daily_initialized_soil/1"
DAY_S = rc.DAY_S
_ROLES = ("boundary", "interevent")


class FloatingDailyError(RuntimeError):
    """A floating daily-lineage contract violation."""


class DailyRefusal(FloatingDailyError):
    """Typed refusal of one transaction: the parent, its clock, water, history, ledger and cursor are exactly as they were. `code`: occurrence, forcing, state, memory, surface_return, step_refused, sequence, parent, input."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code


_MINT = threading.local()
_MINTED: weakref.WeakSet = weakref.WeakSet()
_CAPS: weakref.WeakSet = weakref.WeakSet()
_PRODUCER: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # live state -> (previous execution digest, canonical receipt JSON): O(1) scalars only (no arrays, no history); minted with the state, never caller-supplied


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


def _sha(tag: str, body: Any) -> str:
    return hashlib.sha256((tag + "\0" + json.dumps(body, sort_keys=True, default=repr)).encode()).hexdigest()


@dataclass(frozen=True)
class DailyLedger:
    """Bounded scalar water ledger (m3 over active cells; area multiplied once): lifetime totals and the last day's own terms. `water = before + rain - after - bottom export - E - T` holds per day within the
    MAPLE bound (`last_residual_m3`/`last_bound_m3`)."""

    days_applied: int = 0
    rain_m3: float = 0.0
    evaporation_m3: float = 0.0
    drainage_m3: float = 0.0
    transpiration_m3: float = 0.0
    last_day_rain_m3: float = 0.0
    last_day_evaporation_m3: float = 0.0
    last_day_drainage_m3: float = 0.0
    last_day_transpiration_m3: float = 0.0
    last_residual_m3: float = 0.0
    last_bound_m3: float = 0.0
    initial_storage_m3: float = 0.0  # native storage at the lineage start (the owner's initial W)
    bound_m3: float = 0.0  # accumulated MAPLE component bounds of every accepted daily step and withdrawal
    n_terms: int = 0
    expected_t_s: float = 0.0  # the physical clock the accepted transactions imply (start + DAY_S per daily trial)
    day_storage_before_m3: float = 0.0  # the CURRENT daily window: native storage at the day's start (the step budget's `storage_before_m3`)
    day_bound_m3: float = 0.0  # the window's own MAPLE bounds: the day's step bound plus every withdrawal bound of this day (NOT the lifetime sum)
    day_n_terms: int = 0
    # event<->daily bridge profile (standing surface water on the day): zero for the historical soil-only profile (and then ABSENT from the receipt digest body)
    surface_m3: float = 0.0  # the CURRENT surface storage over the active cells (the audit adds it to the soil storage)
    outlet_m3: float = 0.0  # cumulative outlet export of the day segments (surface router)
    last_day_outlet_m3: float = 0.0


_BRIDGE_LEDGER_FIELDS = ("surface_m3", "outlet_m3", "last_day_outlet_m3")


@dataclass(frozen=True, eq=False)
class _Context:
    """Static, water-free, shared by every state of one lineage (held template; no NativeFloatingSoil)."""

    config_sha256: str
    column: Any
    evaporation: Any
    partition: Any
    zero_added: np.ndarray
    zero_surface: np.ndarray
    shape: tuple
    n_layers: int
    n_cells: int
    area: float
    depth_m: np.ndarray
    active: np.ndarray
    edges_m: np.ndarray
    source_witness: str
    property_generation: int
    root_column_sha256: str
    run_digest: str
    bed: Any
    physics_config_sha256: str  # COMPLETE daily physics configuration (partition, backend, clock profile, native config, root column/operator, property source)
    source_pair: Any  # sealed ACTUAL imported maple_syrup / maple package digests (cold, admitted once at start)
    config: Any  # the held water-free native configuration (reference)
    matrix_op: Any  # the held root matrix operator (reference; carries the Ks/law inputs)
    ks_m_s: Any
    n_active: int  # HOST count of active cells (from the bed's host mask pattern; never a device reduction)
    bed_generation: int = 0  # the event<->daily bridge profile carries its lifecycle execution generations (the historical generation-0 profile keeps 0)
    depth_generation: int = 0
    hold: Any = None  # the lifecycle DailyHold the bridge profile belongs to (None for the historical profile)
    profile: str = ""  # "" = the historical generation-0 profile (its digest bodies are unchanged); the bridge profile name otherwise
    stable_sha256: str = ""  # bridge profile: the digest of the case-STABLE physics identity (partition, backend, clock, raw native config, source anchor, grid) WITHOUT the evolving property root; "" for the historical profile


@dataclass(frozen=True, eq=False)
class FloatingDailyState:
    """Genuine accepted state of the lineage. Minted ONLY by this module; not copyable, not serialisable. `role`: boundary (closed day / origin) or interevent (a completed daily trial before its final close)."""

    ctx: Any
    role: str
    water: Any  # fsl.FloatingSoilState: the SINGLE current water authority
    history: Any  # fm.FloatingMatrixHistory: the ONE J/I
    cursor: Any
    occurrence: Any  # (plan, branch) of the interevent, else None
    origin_digest: str | None  # digest of the boundary that started the occurrence (interevent only)
    revision: int
    request_sequence: int
    ledger: DailyLedger
    execution_digest: str
    parent_digest: str | None
    digest: str
    surface: Any = None  # bridge profile: `(depth_m (ny, nx), qx, qy)` the CURRENT surface state of a wet day (None: soil-only dry day / historical profile)

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise FloatingDailyError("a FloatingDailyState is minted only by this module's transactions (no direct construction or replace())")
        _MINTED.add(self)

    def __copy__(self):
        raise FloatingDailyError("a FloatingDailyState is not copyable")

    def __deepcopy__(self, memo):
        raise FloatingDailyError("a FloatingDailyState is not copyable")

    def __reduce_ex__(self, protocol):
        raise FloatingDailyError("a FloatingDailyState is not serialisable (no physical checkpoint in this slice)")

    @property
    def t_s(self) -> float:
        return float(self.water.t_s)

    @property
    def depth_m(self) -> np.ndarray:
        return self.ctx.depth_m

    def frame_identity(self) -> dict:
        """The attribution identity of this ACCEPTED state for the G1 frame writer, derived from the state (never caller-authored): the property/bed/depth generations of this slice are the generation-0 owner's (the bed and
        depth do not change); `lineage_sha256` is the execution-receipt lineage digest (not a content hash)."""
        c = self.ctx
        return {"run_digest": self.cursor.run_digest, "lineage_sha256": self.execution_digest,
                "source_sha256": _sha("source", [sorted(c.source_pair.items()), c.physics_config_sha256, c.root_column_sha256, c.source_witness]) if not c.profile
                else _sha("source/evolving", [sorted(c.source_pair.items()), c.stable_sha256, c.source_witness]), "property_generation": c.property_generation, "property_sha256": c.root_column_sha256,
                "bed_generation": c.bed_generation, "depth_generation": c.depth_generation, "parent_digest": self.parent_digest or ("0" * 64), "state_digest": self.digest, "state_revision": self.revision}


@dataclass(frozen=True, eq=False)
class ExpectedParent:
    """Capability naming the exact accepted parent instance the caller means (never auto-minted for a transition's own argument)."""

    state: Any

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise FloatingDailyError("an ExpectedParent is minted only by expect_parent(state)")
        _CAPS.add(self)


def expect_parent(state: Any) -> ExpectedParent:
    _need_state(state)
    with _minting():
        return ExpectedParent(state)


def _need_state(state: Any) -> None:
    if type(state) is not FloatingDailyState or state not in _MINTED:
        raise DailyRefusal("state", "not a genuinely minted FloatingDailyState (a copy, plain record or reconstruction carries no authority)")


def _check_parent(parent: Any, expected: Any, role: str) -> None:
    if type(expected) is not ExpectedParent or expected not in _CAPS:
        raise DailyRefusal("parent", "expected_parent is REQUIRED and must be minted by expect_parent() for the accepted state the caller means")
    _need_state(parent)
    if expected.state is not parent:
        raise DailyRefusal("parent", "expected_parent names another instance (stale or alternate branch); refused before any work")
    if parent.role != role:
        raise DailyRefusal("state", f"this transaction needs role {role!r}, the parent is {parent.role!r}")


@dataclass(frozen=True)
class DailyResult:
    """`outcome`: `interevent` (state = the completed daily trial awaiting T/close) or `requires_full_storm` (state None; `forcing` = the FULL native forcing at the unchanged parent's clock; nothing applied)."""

    outcome: str
    state: FloatingDailyState | None
    plan_digest: str
    branch: str
    forcing: Any = None
    reason: str | None = None
    capacity_failure: bool = False


@dataclass(frozen=True, eq=False)
class WithdrawalOutcome:
    state: FloatingDailyState
    actual_m: np.ndarray
    unmet_m: np.ndarray
    scale: np.ndarray
    requested_m3: float
    actual_m3: float
    unmet_m3: float
    sequence: int


@dataclass(frozen=True, eq=False)
class DailyCloseResult:
    state: FloatingDailyState
    notice: Any


def _read(fn: Any, need: int, what: str) -> None:
    if not callable(fn):
        raise DailyRefusal("memory", "an explicit available_bytes_fn is required (no unlimited reader)")
    avail = fn()
    if isinstance(avail, (bool, np.bool_)) or not isinstance(avail, (int, np.integer)) or int(avail) < 0:
        raise DailyRefusal("memory", f"the available-bytes reader must return a non-Boolean integer >= 0, got {avail!r}")
    if need > int(avail):
        raise DailyRefusal("memory", f"{what} needs an estimated {need} B but only {int(avail)} B are reported available; nothing was changed")


def _pools(column: Any, available_bytes_fn: Any, reserve_bytes: Any, host_available_bytes_fn: Any, host_reserve_bytes: Any) -> dict | None:
    """CUDA: the strict two-pool quota (`floating_daily_cuda.quota`; the device gate FIRST, before any reader/hash/array operation). CPU: None, and every host-pool argument is refused as meaningless there."""
    if column.backend == "cuda":
        try:
            fdc.gate(column)
            return fdc.quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes)
        except (fsl.FloatingSoilError, ss.SubsurfaceError, RuntimeError) as exc:
            raise DailyRefusal("memory", f"the CUDA quota/device gate was refused: {exc}") from None
    if host_available_bytes_fn is not None or host_reserve_bytes:
        raise DailyRefusal("memory", "host_available_bytes_fn / host_reserve_bytes apply only to the CUDA lineage (a CPU lineage has one host pool)")
    return None


def _admit_cuda(column: Any, plan: tuple, q: dict, what: str) -> None:
    try:
        fdc.admit(column, plan[0], plan[1], q, what)
    except (fsl.FloatingSoilError, ss.SubsurfaceError, MemoryError) as exc:
        raise DailyRefusal("memory", f"{what}: {exc}") from None


def _reserve(value: Any) -> int:
    try:
        return fsl._count(value, "reserve_bytes")
    except fsl.FloatingSoilError as exc:
        raise DailyRefusal("memory", str(exc)) from None


def _step_need(ctx: _Context, *, recharge: bool, reserve: int) -> int:
    """Composed host model of ONE day: the E-only staging outputs/scratch (live while the step runs), the FSL step, the ledger scalars and the caller's reserve, counted once each pool."""
    e = fe.estimate_floating_evaporation_memory(ctx.n_cells, backend=ctx.column.backend)
    s = fsl.estimate_floating_memory(ctx.n_layers, ctx.n_cells, layouts=dict(ctx.column.layouts), diffusion_mode=ctx.column.diffusion_mode, backend=ctx.column.backend, with_recharge=recharge)
    return int(e.host_bytes) + int(s.step_bytes) + 4096 + reserve


def _mint_state(ctx: _Context, role: str, water: Any, history: Any, cursor: Any, occurrence: Any, origin_digest: str | None, revision: int, seq: int, ledger: DailyLedger,
                prev_exec: str | None, parent_digest: str | None, receipt: dict, surface: Any = None) -> FloatingDailyState:
    items = [(k, v) for k, v in ledger.__dict__.items() if k not in _BRIDGE_LEDGER_FIELDS or v != 0.0]  # a zero bridge field is absent: the historical digest bodies are byte-identical
    body = {"schema": SCHEMA, "run": cursor.run_digest, "role": role, "revision": revision, "seq": seq, "t": float(water.t_s).hex(), "cursor": cursor.digest(), "ledger": sorted(items),
            "occurrence": None if occurrence is None else [occurrence[0].digest, occurrence[1]], "receipt": receipt}
    execution = _sha("execution_receipt_lineage/1", [prev_exec, body])  # deterministic receipt-chain digest, NOT a water content hash
    digest = _sha("state/1", [execution, role, revision, ctx.config_sha256])
    with _minting():
        state = FloatingDailyState(ctx, role, water, history, cursor, occurrence, origin_digest, revision, seq, ledger, execution, parent_digest, digest, surface)
    _PRODUCER[state] = (prev_exec, json.dumps(receipt, sort_keys=True))  # the digest preimage the hash discards: a verified file restore recomputes the SAME execution digest from it
    return state


COLD_METADATA_ALLOWANCE_BYTES = 2 << 20  # first-use source-pair hash: walked-file metadata headroom (measured ~0.43 MB for ~560 files; ~5x), on top of the provenance read chunk
CONSTRUCTION_ALLOWANCE_BYTES = 1 << 16  # declared interpreter/metadata allowance of the lineage construction


def _cold_source_bytes() -> int:
    """Conservative bytes of the FIRST-use source-pair hash in this process (read buffer = `provenance._READ_CHUNK_BYTES` plus walked-file metadata): 0 when both process-local source caches are already populated."""
    from maple_syrup import provenance
    from maple_syrup import sediment_bed as sb

    try:
        from maple_syrup import pond_checkpoint as pc

        warm = "pair" in pc._SOURCE_CACHE and "digest" in sb._MAPLE_DIGEST_CACHE
    except AttributeError:  # a changed cache layout counts as cold (conservative)
        warm = False
    return 0 if warm else int(provenance._READ_CHUNK_BYTES) + COLD_METADATA_ALLOWANCE_BYTES


def _construction_need(owner: Any, bed: Any, batch_cells: int, reserve: int) -> int:
    """Composed HOST plan of the whole constructor, dimension-aware, admitted BEFORE any owned array, cold hash or witness scan: the cold source witness scan, the first-use source-pair hash (when the caches are empty),
    the evaporation operator model, every retained HOST array and the interpreter allowance; the reserve once. CPU/Numba: the `(L, N)` zero block, `N` surface/depth, `L+1` edges, `N` mask and one `(L, N)` storage-query
    block are host arrays. CUDA: ONLY the `L+1` HOST edges and the scalar/packet host terms count here (the zero blocks, device depth/mask references and the initial-storage query are DEVICE terms planned by
    `floating_daily_cuda.start_plan` against the device pool)."""
    column = owner.column
    L, N = int(column.n_layers), int(column.n_cells)
    scan = int(nfs._estimate(None, bed, owner.config, batch_cells, 0)["terms"]["scan"])
    evap = int(fe.estimate_floating_evaporation_memory(N, backend=column.backend).host_bytes)
    if column.backend == "cuda":
        return scan + _cold_source_bytes() + evap + (L + 1) * 8 + CONSTRUCTION_ALLOWANCE_BYTES + reserve
    retained = (L * N + 3 * N + (L + 1)) * 8 + N
    return scan + _cold_source_bytes() + evap + retained + L * N * 8 + CONSTRUCTION_ALLOWANCE_BYTES + reserve


def _physics_config_sha_parts(partition: Any, backend: str, witness: str, shape: tuple, *, config_sha256: str, root_column_sha256: str, matrix_op_sha256: str, generation: int, area: float, profile: str = PROFILE) -> str:
    """The body of `_physics_config_sha` from plain parts (the event<->daily bridge has a boundary root, not an owner); with the default profile the digest is byte-identical to the historical one."""
    return _sha("physics_config/1", {"partition": [float(getattr(partition, k)).hex() for k in ("canopy_intercept", "canopy_slope", "transpiration_slope", "transpiration_offset", "soil_factor")],
                                      "backend": backend, "profile": profile, "schema": SCHEMA, "clock": rc.CLOCK_CONTRACT, "day_s": float(DAY_S).hex(), "native_config_sha256": config_sha256,
                                      "root_column_sha256": root_column_sha256, "matrix_op_sha256": matrix_op_sha256, "source_witness": witness, "property_generation": int(generation),
                                      "area": float(area).hex(), "shape": list(shape)})


def _stable_physics_sha(partition: Any, backend: str, witness: str, shape: tuple, *, config_sha256: str, area: float, profile: str) -> str:
    """The case-STABLE part of the bridge profile's physics identity: it deliberately excludes the root column / matrix operator / property generation (those EVOLVE with every event-boundary property refresh and are
    recorded per frame). The output writer binds its run to this digest; any change of partition, backend, clock contract, raw native configuration, source anchor or grid is a different case and is refused."""
    return _sha("stable_physics/1", {"partition": [float(getattr(partition, k)).hex() for k in ("canopy_intercept", "canopy_slope", "transpiration_slope", "transpiration_offset", "soil_factor")],
                                      "backend": backend, "profile": profile, "schema": SCHEMA, "clock": rc.CLOCK_CONTRACT, "day_s": float(DAY_S).hex(), "native_config_sha256": config_sha256,
                                      "source_witness": witness, "area": float(area).hex(), "shape": list(shape)})


def _physics_config_sha(owner: Any, partition: Any, backend: str, witness: str, shape: tuple) -> str:
    """COMPLETE daily physics configuration (never a code-source hash): the evaporation partition (all five coefficients), backend, versioned clock profile, the native config identity, the root column and matrix operator
    identities (the operator digests the Ks/law inputs), the property source (witness/generation), area and grid. Source code is a separate field (`source_pair`)."""
    return _physics_config_sha_parts(partition, backend, witness, shape, config_sha256=owner.config.sha256, root_column_sha256=owner.column.sha256, matrix_op_sha256=owner.matrix_op.sha256,
                                     generation=owner.generation, area=owner.column.cell_area_m2)


def start_floating_daily_lineage(owner: Any, bed: Any, *, partition: Any, run_digest: Any, available_bytes_fn: Any = None, reserve_bytes: Any = 0, batch_cells: int = nfs.DEFAULT_BATCH_CELLS,
                                 host_available_bytes_fn: Any = None, host_reserve_bytes: Any = 0) -> FloatingDailyState:
    """New-profile start from a genuinely freshly initialized, generation-0, CURRENT owner and the actual `NativeBed` it was initialized from. Order: types/generation/freshness -> (CUDA: the current device gate FIRST)
    -> ONE composed, dimension-aware construction admission (before any owned array, cold hash or scan; CUDA: the HOST pool for the scan/source hash/host scalars and the DEVICE pool for the zero blocks and the initial
    storage query, each reserve once) -> cold source witness equals the owner's -> the sealed source pair (first-use cold hash, admitted above) -> evaporation operator -> every state/context construction and the initial
    native storage -> LAST the owner is released (`retire_native_owner_to_lineage`). Any failure before that leaves the owner current and untouched. CPU/Numba: `available_bytes_fn` (REQUIRED) is the host reader;
    CUDA: it is the DEVICE reader (default: the current device's real reading) and `host_available_bytes_fn`/`host_reserve_bytes` the host pool (default: the real host reader)."""
    if type(owner) is not nfs.NativeFloatingSoil or not owner.is_current:
        raise DailyRefusal("state", "owner must be a genuine CURRENT NativeFloatingSoil")
    column = owner.column
    if column.backend not in ("numpy", "numba", "cuda"):
        raise DailyRefusal("state", f"unsupported backend {column.backend!r}")
    cuda = column.backend == "cuda"
    q = _pools(column, available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes)  # CUDA: the device gate is the FIRST operation of the construction
    reserve = _reserve(reserve_bytes) if not cuda else q["host_reserve_bytes"]  # the host-side reserve of the scan/hash plan (CUDA: the host pool's; CPU: the single pool's)
    if owner.generation != 0 or owner.receipts or owner.totals["refreshes"] != 0:
        raise DailyRefusal("state", "only a freshly initialized generation-0 owner can start a daily lineage (a refreshed or depth-only owner needs a separate freshness proof or a cold full refresh: later slice)")
    if type(bed) is not nfs.NativeBed:
        raise DailyRefusal("state", "bed must be the NativeBed the owner was initialized from")
    try:
        run = rc._hex64(run_digest, "run_digest")
    except rc.RechargeError as exc:
        raise DailyRefusal("occurrence", str(exc)) from None
    if not isinstance(partition, se.EvaporationPartition):
        raise DailyRefusal("state", f"partition must be an EvaporationPartition, got {type(partition).__name__}")
    L, N = int(column.n_layers), int(column.n_cells)
    try:
        need = _construction_need(owner, bed, batch_cells, reserve)
    except (nfs.NativeFloatingError, ss.SubsurfaceError, KeyError) as exc:
        raise DailyRefusal("state", f"construction plan refused: {exc}") from None
    if cuda:
        d_dev, d_host = fdc.start_plan(L, N)
        _admit_cuda(column, (d_dev, need - reserve + d_host), q, "the CUDA floating daily lineage construction")  # host plan WITHOUT its reserve here: fcu.admit adds the host reserve once
        host_fn = q["host_available_bytes_fn"] or ss.default_host_available_bytes
        comp = fdc.component_quota(q)
    else:
        _read(available_bytes_fn, need, "the floating daily lineage construction (scan, first-use source hash, retained arrays)")
        host_fn, comp = available_bytes_fn, None
    try:
        witness = nfs.native_source_witness(bed, owner.config, batch_cells=batch_cells, available_bytes_fn=host_fn, reserve_bytes=0 if cuda else reserve)  # HOST work: the host reader
    except (nfs.NativeFloatingError, ss.SubsurfaceError) as exc:
        raise DailyRefusal("state", f"cold source witness refused: {exc}") from None
    if witness != owner.binding.source_witness:
        raise DailyRefusal("state", "the bed is not the native source this owner was initialized against (cold witness mismatch)")
    if int(bed.active_mask.shape[0]) * int(bed.active_mask.shape[1]) != column.n_cells:
        raise DailyRefusal("state", "the bed's grid does not match the owner's cell count")
    try:
        from maple_syrup.pond_checkpoint import (
            PondCheckpointError,
            pond_checkpoint_source_pair,
        )

        pair = MappingProxyType(dict(pond_checkpoint_source_pair()))  # the ACTUAL imported model source pair, cold once (HOST hash, admitted above), sealed in the context
    except (ImportError, PondCheckpointError) as exc:
        raise DailyRefusal("state", f"the imported source pair cannot be computed: {exc}") from None
    try:
        if cuda:
            evap = fe.prepare_floating_evaporation(column, partition, **comp)
            storage0 = float(fsl.floating_storage_m3(column, owner.state, **comp))
        else:
            evap = fe.prepare_floating_evaporation(column, partition, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve)
            storage0 = float(fsl.floating_storage_m3(column, owner.state))
    except (fsl.FloatingSoilError, ss.SubsurfaceError) as exc:
        raise DailyRefusal("state", f"evaporation operator / initial native storage refused: {exc}") from None
    shape = tuple(int(x) for x in bed.active_mask.shape)
    physics = _physics_config_sha(owner, partition, column.backend, witness, shape)
    n_active = int(np.asarray(bed.active_mask).sum())  # HOST mask pattern (never a device reduction)
    if cuda:
        try:
            zero_added, zero_surface = fdc.zeros(column, (L, N)), fdc.zeros(column, (N,))
        except (fsl.FloatingSoilError, ss.SubsurfaceError) as exc:
            raise DailyRefusal("state", f"device zero blocks refused: {exc}") from None
        depth, active = column._maps["soil_depth"], column._maps["active"]  # the owner's resident device maps (referenced, not copied): ONE device for W, D and the mask
        edges = np.array(owner.config.nominal_edges_m, dtype=np.float64, copy=True)  # shared O(L) edges stay HOST
    else:
        zero_added, zero_surface = np.zeros((L, N)), np.zeros(N)
        zero_added.flags.writeable = False
        zero_surface.flags.writeable = False
        depth = np.array(owner.depth_m, copy=True)
        depth.flags.writeable = False
        edges = np.array(column._maps["edges"], copy=True)
        active = np.array(column._maps["active"], copy=True)
        active.flags.writeable = False
    edges.flags.writeable = False
    ctx = _Context(owner.config.sha256, column, evap, partition, zero_added, zero_surface, shape, L, N, float(column.cell_area_m2), depth, active, edges, witness, int(owner.generation), column.sha256, run, bed,
                   physics, pair, owner.config, owner.matrix_op, owner.ks_m_s, n_active)
    cursor = rc.DailyCursor(run)
    led0 = DailyLedger(initial_storage_m3=storage0, expected_t_s=float(owner.state.t_s))
    state = _mint_state(ctx, "boundary", owner.state, owner.matrix_history, cursor, None, None, 0, 0, led0, None, None, {"kind": "start", "witness": witness, "physics": physics})
    nfs.retire_native_owner_to_lineage(owner)  # LAST: the lineage state is now the only current water authority
    return state


def _validate_inputs(ctx: _Context, pet: Any, cover: Any) -> None:
    if ctx.column.backend == "cuda":  # exact namespace (cupy.ndarray), the column's device, FP64, C-contiguity and shape: metadata only, no coercion, no upload
        for name, a in (("pet_terrain_m_s", pet), ("cover", cover)):
            try:
                fsl._check_vec(ctx.column, name, a, (ctx.n_cells,))
            except fsl.FloatingSoilError as exc:
                raise DailyRefusal("input", str(exc)) from None
        return
    for name, a in (("pet_terrain_m_s", pet), ("cover", cover)):
        if type(a) is not np.ndarray or a.dtype != np.float64 or a.shape != (ctx.n_cells,) or not a.flags.c_contiguous:
            raise DailyRefusal("input", f"{name} must be a C-contiguous float64 ({ctx.n_cells},) numpy array")


def floating_daily_action(parent: FloatingDailyState, *, expected_parent: ExpectedParent, plan: Any, pet_terrain_m_s: Any, cover: Any, available_bytes_fn: Any = None, reserve_bytes: Any = 0,
                          host_available_bytes_fn: Any = None, host_reserve_bytes: Any = 0, after_storm: bool = False) -> DailyResult:
    """ONE daily occurrence from a closed boundary (module docstring). Validation, the re-derived plan, the occurrence check and the composed memory admission precede any trial; the parent is never mutated.
    `after_storm=True` (event<->daily bridge only): the cursor holds the PENDING storm of exactly this plan (the genuinely completed full native storm); the day's soil update is the ordinary rain-0 daily step (the storm's
    rain was the native forcing) and the ORIGINAL plan/digest/branch is what the final close consumes ONCE. The default False keeps every historical behaviour."""
    _check_parent(parent, expected_parent, "boundary")
    if parent.surface is not None:
        raise DailyRefusal("surface", "a state with standing surface water needs the wet-interevent fragments (the event<->daily bridge's wet_day_action); the soil-only action refuses it")
    ctx = parent.ctx
    q = _pools(ctx.column, available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes)  # CUDA: the current device gate precedes every reader/array operation
    cuda = q is not None
    reserve = _reserve(reserve_bytes)
    try:
        rc.verify_plan(plan)
        if after_storm:
            pend = parent.cursor.pending
            if pend is None:
                raise rc.RechargeError("after_storm needs a pending storm occurrence in the cursor")
            hold = parent.ctx.hold
            if hold is None or getattr(hold, "storm_match", None) != pend.forcing_digest:
                raise rc.RechargeError("after_storm needs the pending storm to have been completed as a MATCHED full native storm (the lifecycle's recorded forcing match); a cursor alone is not proof")
            parent.cursor.check_for_consume(plan, pend.branch)
        else:
            parent.cursor.check(plan)
    except (rc.RechargeError, AttributeError) as exc:
        raise DailyRefusal("occurrence", str(exc)) from None
    t0 = parent.t_s
    if plan.selection == "storm" and not after_storm:  # P > T: before E, time, cursor or any water operation
        try:
            forcing = plan.bind_forcing(t0)
        except rc.RechargeError as exc:
            raise DailyRefusal("forcing", str(exc)) from None
        return DailyResult("requires_full_storm", None, plan.digest, "storm", forcing, "P > T: the full native storm is required (not run in this slice)")
    _validate_inputs(ctx, pet_terrain_m_s, cover)
    recharge = plan.selection == "recharge" and not after_storm
    if cuda:
        _admit_cuda(ctx.column, fdc.day_plan(ctx.n_layers, ctx.n_cells, recharge=recharge), q, "one CUDA floating daily interevent")  # composed DEVICE + HOST admission before any array operation
        comp = fdc.component_quota(q)  # the same readers, zero reserves (counted once above)
    else:
        _read(available_bytes_fn, _step_need(ctx, recharge=recharge, reserve=reserve), "one floating daily interevent")
        comp = {"available_bytes_fn": available_bytes_fn}
    try:
        staged = fe.stage_floating_evaporation(ctx.evaporation, pet_terrain_m_s=pet_terrain_m_s, cover=cover, surface_water_m=ctx.zero_surface, dt_s=DAY_S, **comp)
    except (fsl.FloatingSoilError, ss.SubsurfaceError) as exc:
        raise DailyRefusal("input", f"evaporation staging refused: {exc}") from None
    kw = staged.native_step_kwargs()
    try:
        if recharge:
            step = fsl.floating_step_with_recharge(ctx.column, parent.water, DAY_S, recharge_depth_m=plan.depth_m, **comp, **kw)
        else:
            step = fsl.floating_step(ctx.column, parent.water, DAY_S, added_water_m=ctx.zero_added, **comp, **kw)
    except ss.SubsurfaceFitFailure:  # a pulse that does not fit an active closed column: whole occurrence from the UNCHANGED parent, FULL forcing
        try:
            forcing = plan.bind_forcing(t0)
        except rc.RechargeError as exc:
            raise DailyRefusal("forcing", str(exc)) from None
        return DailyResult("requires_full_storm", None, plan.digest, "storm_fallback", forcing, "recharge pulse does not fit an active closed column: full native storm from the unchanged parent", True)
    except (fsl.FloatingSoilError, ss.SubsurfaceError) as exc:
        raise DailyRefusal("step_refused", f"the daily column step was refused: {exc}") from None
    if (float(step.budget.surface_return_m3) > 0.0) if cuda else (float(np.asarray(step.surface_return_m).max(initial=0.0)) > 0.0):  # CUDA: the step's bounded packet scalar (no array download)
        raise DailyRefusal("surface_return", "the daily step returned water to the surface (this soil-only slice has no surface store/router; the candidate is refused, nothing is rounded to zero: needs the native flow of a later slice)")
    b = step.budget
    # the FSL step budget is authoritative: `added_m3` is the rain actually added, `bound_m3` the MAPLE closure bound of that step (no extra tolerance); the declared recharge must agree within that same bound
    rain = float(b.added_m3)
    declared = float(plan.depth_m) * ctx.area * ctx.n_active if recharge else 0.0
    if abs(rain - declared) > float(b.bound_m3):
        raise DailyRefusal("step_refused", f"the step's added water {rain!r} m3 differs from the declared recharge {declared!r} m3 beyond the MAPLE bound {b.bound_m3!r} m3")
    resid = (float(b.storage_after_m3) - float(b.storage_before_m3)) - rain + float(b.evaporation_m3) + float(b.drainage_m3)
    if abs(resid) > float(b.bound_m3):
        raise DailyRefusal("step_refused", f"daily water closure violated: residual {resid!r} m3 exceeds the MAPLE bound {b.bound_m3!r} m3")
    expected_t = float(parent.ledger.expected_t_s) + DAY_S
    if float(step.state.t_s) != expected_t:
        raise DailyRefusal("step_refused", f"the step ended at {step.state.t_s!r} s, not the accepted clock + one day {expected_t!r} s")
    led = parent.ledger
    new_led = replace(DailyLedger(led.days_applied + 1, led.rain_m3 + rain, led.evaporation_m3 + float(b.evaporation_m3), led.drainage_m3 + float(b.drainage_m3), led.transpiration_m3, rain, float(b.evaporation_m3),
                                  float(b.drainage_m3), 0.0, resid, float(b.bound_m3), led.initial_storage_m3, led.bound_m3 + float(b.bound_m3), led.n_terms + int(b.n_terms), expected_t,
                                  float(b.storage_before_m3), float(b.bound_m3), int(b.n_terms)), surface_m3=led.surface_m3, outlet_m3=led.outlet_m3)
    branch = parent.cursor.pending.branch if after_storm else plan.selection
    child = _mint_state(ctx, "interevent", step.state, parent.history, parent.cursor, (plan, branch), parent.digest, parent.revision + 1, 0, new_led, parent.execution_digest, parent.digest,
                        {"kind": "daily", "plan": plan.digest, "branch": branch, "dt": DAY_S})
    return DailyResult("interevent", child, plan.digest, branch)


def aggregate_sparse_demand(n_layers: int, n_cells: int, cells: Any, layers: Any, amounts_m: Any, *, max_links: int, available_bytes_fn: Any, reserve_bytes: Any = 0) -> np.ndarray:
    """Bounded aggregation of LEAF's sparse root links (cell, layer, requested depth in m) into ONE `(L, N)` aggregate request (duplicates sum). The only dense array is the aggregate itself (no `P x L x N`).
    Order: strict metadata (types, 1-D, dtype kind, equal length <= the explicit `max_links`) -> the composed host plan `8 L N` (aggregate) + `O(K)` index/amount temporaries (casts, validation masks) + allowance +
    reserve, admitted against the explicit reader BEFORE any cast, validation temporary or the dense allocation -> range/finite validation -> aggregation. Nothing is allocated on a refusal."""
    L, N = fsl._count(n_layers, "n_layers", minimum=1), fsl._count(n_cells, "n_cells", minimum=1)
    cap = fsl._count(max_links, "max_links", minimum=1)
    reserve = _reserve(reserve_bytes)
    arrs = []
    for name, a, kinds in (("cells", cells, "iu"), ("layers", layers, "iu"), ("amounts_m", amounts_m, "f")):
        if type(a) is not np.ndarray or a.ndim != 1 or a.dtype.kind not in kinds:
            raise DailyRefusal("input", f"{name} must be a 1-D numpy array of {'integers' if kinds == 'iu' else 'floats'}")
        arrs.append(a)
    k = int(arrs[0].shape[0])
    if not (arrs[1].shape[0] == arrs[2].shape[0] == k) or k > cap:
        raise DailyRefusal("input", f"links must have equal length <= max_links ({cap})")
    need = 8 * L * N + k * (3 * 8 + 2) + CONSTRUCTION_ALLOWANCE_BYTES + reserve  # aggregate + int64/float64 casts + finite and sign masks
    _read(available_bytes_fn, need, "one sparse-link aggregation")
    c, l, v = arrs
    if k and (c.min() < 0 or c.max() >= N or l.min() < 0 or l.max() >= L):
        raise DailyRefusal("input", "a root link index is out of range")
    if not np.isfinite(v).all() or (v < 0.0).any():
        raise DailyRefusal("input", "amounts must be finite and >= 0")
    out = np.zeros((L, N), dtype=np.float64)
    np.add.at(out, (l.astype(np.int64, copy=False), c.astype(np.int64, copy=False)), v.astype(np.float64, copy=False))
    return out


def request_floating_withdrawal(parent: FloatingDailyState, *, expected_parent: ExpectedParent, request_m: Any, expected_sequence: Any, available_bytes_fn: Any = None, reserve_bytes: Any = 0,
                                host_available_bytes_fn: Any = None, host_reserve_bytes: Any = 0) -> WithdrawalOutcome:
    """ZERO-time aggregate LEAF withdrawal on the interevent (after the day's hydrology, before the final close): the actual `floating_withdraw` law (no compensation); the child consumes one request sequence number (a
    repeated or stale `expected_sequence` is refused, so a request never debits twice); an all-zero request is legal and accepts 0. SYRUP debits ONCE; LEAF receives actual/unmet/scale."""
    _check_parent(parent, expected_parent, "interevent")
    q = _pools(parent.ctx.column, available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes)  # CUDA: device gate first
    reserve = _reserve(reserve_bytes)
    if isinstance(expected_sequence, (bool, np.bool_)) or not isinstance(expected_sequence, (int, np.integer)) or int(expected_sequence) != parent.request_sequence:
        raise DailyRefusal("sequence", f"the request sequence must be exactly {parent.request_sequence} (a duplicate, stale or skipped request is refused)")
    ctx = parent.ctx
    if q is not None:  # the aggregate request must already be device resident (caller-owned); no upload, no P*L*N
        try:  # genuine CUDA request METADATA (exact cupy.ndarray of the column's namespace and DEVICE, FP64, C-contiguity, (L, N) shape) BEFORE any reader/admission: no coercion, no upload, no array read
            fsl._check_vec(ctx.column, "requested_water_m", request_m, (ctx.n_layers, ctx.n_cells))
        except fsl.FloatingSoilError as exc:
            raise DailyRefusal("input", f"withdrawal request refused: {exc}") from None
        _admit_cuda(ctx.column, fdc.withdrawal_plan(ctx.n_layers, ctx.n_cells), q, "one CUDA floating withdrawal")
        comp = fdc.component_quota(q)
    else:
        est = ft.estimate_floating_withdrawal_memory(ctx.n_layers, ctx.n_cells, backend=ctx.column.backend)
        _read(available_bytes_fn, int(est.required_bytes) + 4096 + reserve, "one floating withdrawal")
        comp = {"available_bytes_fn": available_bytes_fn}
    try:
        res = ft.floating_withdraw(ctx.column, parent.water, request_m, **comp)
    except (fsl.FloatingSoilError, ss.SubsurfaceError) as exc:
        raise DailyRefusal("input", f"withdrawal refused: {exc}") from None
    b = res.budget
    req, act, unm = float(b.requested_m3), float(b.actual_m3), float(b.unmet_m3)
    led = parent.ledger
    new_led = replace(DailyLedger(led.days_applied, led.rain_m3, led.evaporation_m3, led.drainage_m3, led.transpiration_m3 + act, led.last_day_rain_m3, led.last_day_evaporation_m3, led.last_day_drainage_m3,
                                  led.last_day_transpiration_m3 + act, led.last_residual_m3, led.last_bound_m3, led.initial_storage_m3, led.bound_m3 + float(b.bound_m3), led.n_terms + int(b.n_terms), led.expected_t_s,
                                  led.day_storage_before_m3, led.day_bound_m3 + float(b.bound_m3), led.day_n_terms + int(b.n_terms)), surface_m3=led.surface_m3, outlet_m3=led.outlet_m3, last_day_outlet_m3=led.last_day_outlet_m3)
    child = _mint_state(ctx, "interevent", res.state, parent.history, parent.cursor, parent.occurrence, parent.origin_digest, parent.revision + 1, parent.request_sequence + 1, new_led, parent.execution_digest,
                        parent.digest, {"kind": "withdrawal", "seq": parent.request_sequence, "requested": req.hex(), "actual": act.hex()}, parent.surface)
    return WithdrawalOutcome(child, res.actual_m, res.unmet_m, res.scale, req, act, unm, parent.request_sequence)


AUDIT_DOUBLES_PER_CELL = 10  # declared host float rows the audit may hold per cell: the per-layer porous thickness / saturation / tolerance rows of `_check_water_values` and the boolean-indexed copy of `floating_storage_m3`
AUDIT_MASK_BYTES_PER_CELL = 8  # declared boolean masks per cell (finite/sign/capacity/active-index)


def _audit_need(ctx: _Context, reserve: int) -> int:
    """Dimension-aware host plan of the final close audit (O(N), independent of L: the checks run layer by layer): the float rows and masks above, the ledger scalars and the interpreter allowance; the reserve once."""
    return AUDIT_DOUBLES_PER_CELL * 8 * ctx.n_cells + AUDIT_MASK_BYTES_PER_CELL * ctx.n_cells + CONSTRUCTION_ALLOWANCE_BYTES + reserve


def _finite_scalars(led: DailyLedger) -> None:
    for name, v in led.__dict__.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise DailyRefusal("state", f"the ledger scalar {name} is not a finite number")
    for name in ("bound_m3", "day_bound_m3", "last_bound_m3"):
        if getattr(led, name) < 0.0:
            raise DailyRefusal("state", f"the ledger bound {name} is negative")
    if led.n_terms < 0 or led.day_n_terms < 0:
        raise DailyRefusal("state", "a ledger term count is negative")


def _audit_final(parent: FloatingDailyState, available_bytes_fn: Any, reserve_raw: Any, host_available_bytes_fn: Any = None, host_reserve_raw: Any = 0) -> None:
    """The FINAL close audit. ORDER: strict reserve/reader metadata -> the dimension-aware host plan admitted against the REAL reader (the default host reader when none is given) BEFORE any private value check, array
    temporary or native storage query -> FSL state/column binding and values (finite, non-negative, capacity, rock) -> the accepted clock -> TWO closures, each within MAPLE bounds only (`conservation.volume_roundoff_bound_m3`
    plus the actual component bounds; no invented tolerance):
      * the DAILY WINDOW: `storage_now = storage at the day start (StepBudget.storage_before_m3) + the day's rain (StepBudget.added_m3) - the day's E - drainage - T`, within the window's OWN bound (the day's step bound
        plus its withdrawal bounds and the counted terms of those components, and the storage query);
      * the LIFETIME: `storage_now = initial + all rain - all E - all drainage - all T` within the ACCUMULATED component bounds.
    A long history therefore never loosens the current day's bound. Nothing is mutated; the caller's accepted parent, cursor, expected parent and output are unchanged and a retry works."""
    ctx, led, w = parent.ctx, parent.ledger, parent.water
    q = _pools(ctx.column, available_bytes_fn, reserve_raw, host_available_bytes_fn, host_reserve_raw)  # CUDA: device gate first
    reserve = _reserve(reserve_raw)
    if q is not None:  # CUDA: composed device/host plan of the resident state check + storage packet, admitted BEFORE any kernel or readback; the CPU private value checks never see a CuPy array
        _admit_cuda(ctx.column, fdc.audit_plan(ctx.n_layers, ctx.n_cells), q, "the CUDA final close audit")
    else:
        fn = ss.default_host_available_bytes if available_bytes_fn is None else available_bytes_fn
        _read(fn, _audit_need(ctx, reserve), "the final close audit")
    _finite_scalars(led)
    try:
        fsl._check_state(ctx.column, w)
        if q is not None:
            fdc.check_state_resident(ctx.column, w.water_m)
            storage = float(fsl.floating_storage_m3(ctx.column, w, **fdc.component_quota(q)))
        else:
            fsl._check_water_values(ctx.column, np.asarray(w.water_m), False, who="final close: ")
            storage = float(fsl.floating_storage_m3(ctx.column, w))
    except (fsl.FloatingSoilError, ss.SubsurfaceError) as exc:
        raise DailyRefusal("state", f"the current native water failed the final close audit: {exc}") from None
    if not math.isfinite(storage):
        raise DailyRefusal("state", "the current native storage is not finite")
    storage += float(led.surface_m3)  # bridge profile: the CURRENT surface storage (0 for the historical soil-only profile)
    if float(w.t_s) != float(led.expected_t_s):
        raise DailyRefusal("state", f"the water clock {w.t_s!r} s is not the accepted clock {led.expected_t_s!r} s")
    n_act = ctx.n_active
    query_terms = ctx.n_layers * n_act + 8
    # daily window
    w_expected = led.day_storage_before_m3 + led.last_day_rain_m3 - led.last_day_evaporation_m3 - led.last_day_drainage_m3 - led.last_day_transpiration_m3 - led.last_day_outlet_m3
    w_scale = max(abs(storage), abs(led.day_storage_before_m3), abs(led.last_day_rain_m3), abs(led.last_day_evaporation_m3), abs(led.last_day_drainage_m3), abs(led.last_day_transpiration_m3))
    w_bound = led.day_bound_m3 + volume_roundoff_bound_m3(led.day_n_terms + query_terms, w_scale)
    w_res = storage - w_expected
    if not math.isfinite(w_res) or abs(w_res) > w_bound:
        raise DailyRefusal("state", f"final close daily-window audit failed: native storage {storage!r} m3 differs from the day's start + rain - E - drainage - T = {w_expected!r} m3 by {w_res!r} m3 "
                                    f"(the day's own MAPLE bound {w_bound!r} m3): the water was changed outside the accepted transactions")
    # lifetime
    expected = led.initial_storage_m3 + led.rain_m3 - led.evaporation_m3 - led.drainage_m3 - led.transpiration_m3 - led.outlet_m3
    scale = max(abs(storage), abs(led.initial_storage_m3), abs(led.rain_m3), abs(led.evaporation_m3), abs(led.drainage_m3), abs(led.transpiration_m3))
    bound = led.bound_m3 + volume_roundoff_bound_m3(led.n_terms + 2 * query_terms, scale)
    residual = storage - expected
    if not math.isfinite(residual) or abs(residual) > bound:
        raise DailyRefusal("state", f"final close lifetime audit failed: native storage {storage!r} m3 differs from initial + rain - E - drainage - T = {expected!r} m3 by {residual!r} m3 "
                                    f"(accumulated MAPLE bound {bound!r} m3): the water was changed outside the accepted transactions")


def close_floating_day(parent: FloatingDailyState, *, expected_parent: ExpectedParent, output: Any = None, host_available_bytes_fn: Any = None, disk_available_bytes_fn: Any = None,
                       available_bytes_fn: Any = None, reserve_bytes: Any = 0, host_reserve_bytes: Any = 0) -> DailyCloseResult:
    """The FINAL close of the day, after the actual T: the closed boundary consumes the cursor ordinal ONCE (nominal day = one completed day) and, if an output is configured, the every-N writer is notified for exactly
    that accepted closed state. If the output step refuses or fails its exception propagates and NO closed state is returned: the caller still holds the accepted interevent parent (never mutated) and may retry."""
    from maple_syrup import floating_daily_output as fo

    _check_parent(parent, expected_parent, "interevent")
    if output is not None and type(output) is not fo.FloatingMoistureOutput:
        raise DailyRefusal("input", f"output must be None or a FloatingMoistureOutput, got {type(output).__name__} (no duck-typed substitutes)")
    cuda_close = parent.ctx.column.backend == "cuda"  # CPU: `host_available_bytes_fn` belongs to the OUTPUT only; CUDA: it is also the host pool of the audit
    _audit_final(parent, available_bytes_fn, reserve_bytes, host_available_bytes_fn if cuda_close else None, host_reserve_bytes)  # admission, then the actual window + lifetime water audit, precede the mint and any output I/O
    plan, branch = parent.occurrence
    try:
        cursor = parent.cursor.consume(plan, branch, parent.origin_digest)
    except rc.RechargeError as exc:
        raise DailyRefusal("occurrence", str(exc)) from None
    closed = _mint_state(parent.ctx, "boundary", parent.water, parent.history, cursor, None, None, parent.revision + 1, 0, parent.ledger, parent.execution_digest, parent.digest,
                         {"kind": "close", "plan": plan.digest, "branch": branch}, parent.surface)
    if output is None:
        return DailyCloseResult(closed, None)
    notice = output.notify_day_closed(closed, expected_parent=expect_parent(closed), host_available_bytes_fn=host_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn)
    return DailyCloseResult(closed, notice)




# ----------------------------------------------------------------------------------------------------------------------------------------------
# physical-file hooks (closed day): the producer's own scalar record and its verified restore (no caller-blessed hash, no re-mint as a new start)
# ----------------------------------------------------------------------------------------------------------------------------------------------
_LEDGER_FIELDS = tuple(DailyLedger.__dataclass_fields__)
_RECORD_KEYS = frozenset({"role", "revision", "request_sequence", "execution_digest", "digest", "parent_digest", "prev_execution", "receipt", "ledger", "cursor", "t_s", "has_surface"})


def closed_state_record(state: Any) -> dict:
    """The bounded closed SCALAR record of a genuinely minted CLOSED daily state (role boundary, no occurrence), from the state's own producer receipt. Refuses a copy/plain record, an interevent trial or a state whose
    producer receipt is absent. Pure; changes nothing."""
    _need_state(state)
    if state.role != "boundary" or state.occurrence is not None or state.cursor.pending is not None:
        raise DailyRefusal("state", "only a genuine CLOSED daily state (role boundary, no occurrence, no pending storm) has a physical record")
    prod = _PRODUCER.get(state)
    if prod is None:
        raise DailyRefusal("state", "the state carries no producer receipt")
    prev, receipt = prod
    return {"role": state.role, "revision": int(state.revision), "request_sequence": int(state.request_sequence), "execution_digest": state.execution_digest, "digest": state.digest, "parent_digest": state.parent_digest,
            "prev_execution": prev, "receipt": json.loads(receipt), "ledger": {k: getattr(state.ledger, k) for k in _LEDGER_FIELDS}, "cursor": rc.cursor_record(state.cursor), "t_s": float(state.water.t_s),
            "has_surface": state.surface is not None}


def restore_closed_state(ctx: Any, water: Any, history: Any, record: Any, surface: Any = None) -> FloatingDailyState:
    """Mint the closed state of a VERIFIED file record on the restored context: the strict closed record (exact types), the cursor's own validators, the clock/ledger/surface coherence with the ACTUAL restored water, and the
    RECOMPUTED execution and state digests (from the stored previous execution digest and receipt) must equal the stored ones -- the chain, revision, request sequence, parent digest and cursor are preserved EXACTLY (never
    re-minted as a new start). A mismatch of any kind is a typed refusal before a state exists."""
    if type(ctx) is not _Context or type(record) is not dict or set(record) != _RECORD_KEYS:
        raise DailyRefusal("state", "closed-state record keys mismatch (closed key set)")
    r = record
    for key in ("revision", "request_sequence"):
        if type(r[key]) is not int or r[key] < 0:
            raise DailyRefusal("state", f"closed-state record {key} must be an exact non-negative integer")
    if r["role"] != "boundary" or r["request_sequence"] != 0 or type(r["receipt"]) is not dict or type(r["has_surface"]) is not bool:
        raise DailyRefusal("state", "closed-state record role/sequence/receipt is not a closed day")
    for key in ("execution_digest", "digest"):
        if type(r[key]) is not str or len(r[key]) != 64:
            raise DailyRefusal("state", f"closed-state record {key} is not a 64-hex digest")
    for key in ("parent_digest", "prev_execution"):
        if r[key] is not None and (type(r[key]) is not str or len(r[key]) != 64):
            raise DailyRefusal("state", f"closed-state record {key} is not a 64-hex digest or null")
    led = r["ledger"]
    if type(led) is not dict or set(led) != set(_LEDGER_FIELDS):
        raise DailyRefusal("state", "closed-state ledger keys mismatch (closed key set)")
    kw = {}
    for k in _LEDGER_FIELDS:
        v = led[k]
        want = int if k in ("days_applied", "n_terms", "day_n_terms") else float
        if type(v) is not want and not (want is float and type(v) is int) or isinstance(v, bool):
            raise DailyRefusal("state", f"closed-state ledger {k} has the wrong type")
        if not math.isfinite(float(v)):
            raise DailyRefusal("state", f"closed-state ledger {k} is not finite")
        kw[k] = want(v)
    ledger = DailyLedger(**kw)
    try:
        cursor = rc.cursor_from_record(r["cursor"])
    except rc.RechargeError as exc:
        raise DailyRefusal("state", f"closed-state cursor record refused: {exc}") from None
    if cursor.pending is not None or cursor.run_digest != ctx.run_digest:
        raise DailyRefusal("state", "the restored cursor must have nothing pending and the context's run")
    t = float(water.t_s)
    if type(r["t_s"]) is not float or r["t_s"] != t or ledger.expected_t_s != t:
        raise DailyRefusal("state", "the record's clock is not the restored water's clock / the ledger's accepted clock")
    if (surface is not None) != r["has_surface"]:
        raise DailyRefusal("state", "the record's surface flag differs from the restored surface")
    kind = r["receipt"].get("kind")
    if kind == "close":  # the parent is the interevent trial the close finalised: its state digest is recomputable from the chain (so a tampered parent digest cannot hide behind a valid-looking file)
        if r["prev_execution"] is None or r["revision"] < 1 or r["parent_digest"] != _sha("state/1", [r["prev_execution"], "interevent", r["revision"] - 1, ctx.config_sha256]):
            raise DailyRefusal("state", "the closed-state record's parent digest is not the digest of the interevent trial its close finalised")
    elif kind in ("start", "bridge_start"):
        if r["prev_execution"] is not None or r["parent_digest"] is not None or r["revision"] != 0:
            raise DailyRefusal("state", "an origin closed state has no previous execution, no parent and revision 0")
    else:
        raise DailyRefusal("state", f"closed-state receipt kind {kind!r} is not a closed-day producer receipt")
    state = _mint_state(ctx, "boundary", water, history, cursor, None, None, r["revision"], 0, ledger, r["prev_execution"], r["parent_digest"], r["receipt"], surface)
    if state.execution_digest != r["execution_digest"] or state.digest != r["digest"]:
        raise DailyRefusal("state", "the recomputed execution/state digest differs from the file's (the chain, receipt, ledger, cursor or context was altered)")
    return state
