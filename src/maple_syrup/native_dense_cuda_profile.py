"""Opt-in native MAPLE 9f466d2c ORDINARY DENSE bed preparation ON CUDA (CANDIDATE A3b, new file; nothing existing is edited).

STATUS: UNQUALIFIED. Written without any GPU/device/CuPy-array execution; only Root hardware qualification can change that.

SCOPE. One cold physical-setup capability, profile ``native_maple_9f_dense_cuda_v1``: from a genuine `NativeCaseAdmission` admitted under
the device-runtime SOURCE profile (`native_case_profile.CUDA_SOURCE_PROFILE_NAME`; a CPU admission is refused) derive the actual device
`(BedState, BedContext)` through the UNCHANGED `sediment_bed.bed_from_case(compiled, xp=cupy)` (ONE cold MAPLE backend transfer, no copied
wind/bed physics), after explicit bounded HOST/DEVICE admission, the A1 source association (`dense_cuda_source`, native9f profile), the
unchanged CPU validation suite on the caller's host leaves, an exact streamed content fingerprint of host vs device, and the native
namespace-aware validators on the device bed. It is NOT layered-soil setup (A4), hydrology, the GPU owner (A2), a codec (A6), calendar/
LEAF/Windowed/evolving-bed/field/scale qualification, and it never returns or relabels a `NativeDensePreparation`.

Ownership after success: the DEVICE bed's ``water.depth_m`` is the single future H authority; the caller's host compiled H is stale after any
event and no second water store exists. The record BORROWS the admission/compiled case only for the COLD initial `recheck` and for upstream
(A4) host texture work; callers that evolve state should drop the record once physical setup owns the device bed, and must never keep the
initial bed/soil alive for later reporting.

Memory (see `plan_native_dense_cuda`): host and device are planned and admitted separately against fresh readings plus explicit caller reserves
BEFORE the first recheck/hash/transfer; named terms; sequential stages take their max, retained records are added. Exclusions: CUDA context,
NVRTC/JIT, CuPy import, allocator pool growth/fragmentation, pinned-pool retention, native compile/load, OS page cache; not a whole-process
bound, and a reading is not a reservation. Transfer counters are process-global: the contract is single-threaded around the call.
"""
from __future__ import annotations

import dataclasses
import hashlib
import types
import weakref
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# Everything below is imported at MODULE load (normal setup cost) so the first plan call lazy-imports no source file.
import maple.core.backend as _backend
from maple.core.backend.arrays import array_leaf_inventory as _array_leaf_inventory
from maple.coupling.sediment_ledger.validation import validate_ledger_state as _validate_ledger_state
from maple.core.types.topographic_commit import validate_committed_topography_state as _validate_committed
from maple.surface.active_layer.validation import check_active_layer_voxel_partition as _check_partition
from maple.surface.availability.validation import check_sediment_availability_state as _check_availability
from maple.water import validate_water_state as _validate_water

from maple_syrup import dense_cuda_source as dcs
from maple_syrup import layered_storm as _ls
from maple_syrup import native_case_profile as ncp
from maple_syrup import native_dense_profile as ndp
from maple_syrup import sediment_bed as sb
from maple_syrup import subsurface as ss

__all__ = ["PROFILE_NAME", "H_AUTHORITY", "NativeDenseCudaPlan", "NativeDenseCudaPreparation", "NativeDenseCudaProfileError",
           "NativeDenseCudaMemoryError", "plan_native_dense_cuda", "prepare_native_dense_cuda_case"]

PROFILE_NAME = "native_maple_9f_dense_cuda_v1"
H_AUTHORITY = "device_bed.water.depth_m"
OWN_RECORD_RETAINED_BYTES = 8192  # sealed payload mapping (~24 entries), counters mapping and tuples; independent of N, K, V, L
CUPY_PINNED_STAGING = "largest_transferred_leaf"  # conservative host staging allowance (unverified; Root measures)
ALLOCATOR_ROUNDING_BYTES = 512  # CuPy rounds each device allocation up to 512 B (per leaf occurrence, conservative)
_PACKET_BYTES = 1 << 16  # fixed shells/flag packets/hashlib state (not scaled)


def _r(value, limit: int = 60) -> str:
    return ndp._r(value, limit)


class NativeDenseCudaProfileError(RuntimeError):
    """Unsupported source/case/storage/feature/backend/device request, failed validation, or counter/fingerprint disagreement."""


class NativeDenseCudaMemoryError(NativeDenseCudaProfileError):
    """A host or device admission was refused BEFORE the stage it protects (nothing was transferred, hashed or rechecked)."""


# --- argument gates (typed, before any other work) -------------------------------------------------------------------------------
def _nonneg_int(v, name: str) -> int:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 0:
        raise NativeDenseCudaProfileError(f"{name} must be a non-Boolean integer >= 0, got {_r(v)}")
    return int(v)


def _check_args(admission, device_id, host_reserve, device_reserve, backend, limits, host_fn=None, device_fn=None):
    if type(backend) is not str or backend != "cuda":
        raise NativeDenseCudaProfileError(f"backend must be exactly 'cuda' (this capability never falls back), got {_r(backend)}")
    dev = _nonneg_int(device_id, "device_id")
    hr = _nonneg_int(host_reserve, "caller_host_reserve_bytes")
    dr = _nonneg_int(device_reserve, "caller_device_reserve_bytes")
    if limits is None:
        limits = ndp.DenseLimits()
    if type(limits) is not ndp.DenseLimits:
        raise NativeDenseCudaProfileError("limits must be a DenseLimits")
    for name, fn in (("host_available_bytes_fn", host_fn), ("device_available_bytes_fn", device_fn)):
        if fn is not None and not callable(fn):
            raise NativeDenseCudaProfileError(f"{name} must be callable or None")
    if type(admission) is not ncp.NativeCaseAdmission:
        raise NativeDenseCudaProfileError("a genuine NativeCaseAdmission from native_case_profile is required")
    try:
        compiled = admission.compiled  # authenticates the sealed record
        payload = admission._payload("admission")
    except ncp.NativeCaseProfileError as exc:
        raise NativeDenseCudaProfileError(f"unauthenticated admission: {exc}") from exc
    if payload["profile_name"] != ncp.CUDA_SOURCE_PROFILE_NAME:
        raise NativeDenseCudaProfileError(
            f"the device bed preparation requires a {ncp.CUDA_SOURCE_PROFILE_NAME} admission, got {payload['profile_name']!r} "
            f"(a {ncp.PROFILE_NAME} CPU admission is never accepted here)")
    return dev, hr, dr, limits, compiled, payload


# --- host/device structural metadata (no array work beyond reading shapes/flags) -----------------------------------------------------
def _host_shell(compiled):
    """Dataclass shell over the caller's host leaves (NO leaf copy)."""
    state = sb.BedState(compiled.voxel_column, compiled.active_layer, compiled.water, compiled.sediment_ledger,
                        compiled.committed_topography, compiled.sediment_availability)
    return state


def _walk(obj, path=""):
    """Yield (path, leaf) for every array and (path, scalar) for every non-array leaf, in the traversal order of MAPLE's `_map_tree`
    (tuples, lists, dicts, `init=True` dataclass fields)."""
    if _backend.is_array(obj):
        yield path, obj
    elif isinstance(obj, tuple):
        for i, item in enumerate(obj):
            yield from _walk(item, f"{path}[{i}]")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from _walk(item, f"{path}[{i}]")
    elif isinstance(obj, dict):
        for k, item in obj.items():
            yield from _walk(item, f"{path}[{k!r}]")
    elif hasattr(obj, "__dataclass_fields__"):
        for f in dataclasses.fields(obj):
            if f.init:
                yield from _walk(getattr(obj, f.name), f"{path}.{f.name}")
    else:
        yield path, obj


def _host_leaves_meta(state, fraction_shape):
    """(path, shape, dtype str, nbytes) of every transferred array occurrence in transfer order (state first, then the fraction) and a
    refusal of anything the cold fingerprint/transfer contract cannot cover. Metadata only."""
    out = []
    for path, leaf in _walk(state, "bed"):
        if not isinstance(leaf, np.ndarray):
            if _backend.is_array(leaf):
                raise NativeDenseCudaProfileError(f"{path} is already a device array; the compiled case must be host data")
            continue
        if leaf.dtype.hasobject or leaf.dtype.kind not in "fiub" or not leaf.dtype.isnative:
            raise NativeDenseCudaProfileError(f"{path} has unsupported dtype {leaf.dtype}")
        if not leaf.flags.c_contiguous:
            raise NativeDenseCudaProfileError(f"{path} is not C-contiguous (the exact streamed fingerprint contract needs C order)")
        out.append((path, tuple(leaf.shape), leaf.dtype.str, int(leaf.nbytes)))
    f = ("context.initial_available_fraction", tuple(fraction_shape), np.dtype(np.float64).str, 8 * int(np.prod(fraction_shape)))
    out.append(f)
    return out


@dataclass(frozen=True)
class NativeDenseCudaPlan:
    """Pure-metadata plan: named host and device terms (bytes) and the transfer leaf inventory. No CuPy, array, device or hash was used."""

    profile_name: str
    shape: tuple
    n_voxels: int
    n_classes: int
    device_id: int
    n_leaf_occurrences: int
    n_aliased_leaves: int
    transfer_bytes: int
    context_fraction_bytes: int
    host_terms: tuple
    device_terms: tuple
    host_required_bytes: int
    device_required_bytes: int
    caller_host_reserve_bytes: int
    caller_device_reserve_bytes: int
    resident_bytes_borrowed: int
    source_cold: bool
    fraction_bytes: int = 0
    leaves: tuple = ()

    def host_total(self) -> int:
        return self.host_required_bytes + self.caller_host_reserve_bytes

    def device_total(self) -> int:
        return self.device_required_bytes + self.caller_device_reserve_bytes


def _plan(compiled, payload, meta, dev, hr, dr) -> NativeDenseCudaPlan:
    ny, nx = meta["shape"]
    N, V, K = ny * nx, meta["n_voxels"], meta["n_classes"]
    state = _host_shell(compiled)
    leaves = _host_leaves_meta(state, (ny, nx, K))
    inv = _array_leaf_inventory(state)  # what to_device_tree(state) moves: alias-counted, metadata only
    state_leaves = leaves[:-1]
    if inv["n_leaves"] != len(state_leaves) or inv["transfer_bytes"] != sum(x[3] for x in state_leaves):
        raise NativeDenseCudaProfileError("the planned leaf inventory differs from MAPLE's alias-counted transfer inventory")
    frac_bytes = leaves[-1][3]
    transfer_bytes = int(inv["transfer_bytes"]) + frac_bytes
    n_occ = int(inv["n_leaves"]) + 1
    largest = max(x[3] for x in leaves)
    dense = meta["dense"]
    src = dcs.plan_source_verification(dcs.PROFILE_NATIVE9F)
    chunk = min(_ls.HASH_CHUNK_BYTES, max(largest, 8))
    host_stages = {
        "stage_source_audit": meta["audit"]["total"],
        "stage_source_verify": src.stage_bytes,
        "stage_host_validation": dense["total"],
        "stage_transfer_host": frac_bytes + largest + _PACKET_BYTES,  # fractions made inside bed_from_case + pinned staging allowance
        "stage_fingerprint": 2 * chunk + _PACKET_BYTES,
    }
    host_retained = {"retained_source_record": dcs.RECORD_RETAINED_BYTES, "retained_own_record": OWN_RECORD_RETAINED_BYTES}
    host_peak = max(host_stages.values())
    host_terms = (*host_stages.items(), ("stages_max", host_peak), *host_retained.items(),
                  ("resident_bytes_borrowed_caller_compiled", meta["resident"]))
    host_required = host_peak + sum(host_retained.values())
    scratch = {k: dense[k] for k in ("voxel_totals", "voxel_masks", "partition_active", "ledger", "fixed")}
    scratch["native_flags"] = _ls.native_validation_bytes(N, K, "cuda")
    device_terms = (("transferred_leaves_alias_counted", int(inv["transfer_bytes"])), ("context_fraction", frac_bytes),
                    ("allocator_rounding", ALLOCATOR_ROUNDING_BYTES * n_occ), *(("validation_" + k, v) for k, v in scratch.items()),
                    ("validation_scratch", sum(scratch.values())))
    device_required = transfer_bytes + ALLOCATOR_ROUNDING_BYTES * n_occ + sum(scratch.values())
    return NativeDenseCudaPlan(PROFILE_NAME, (ny, nx), V, K, dev, n_occ, int(inv["n_aliased_leaves"]), transfer_bytes, frac_bytes,
                               host_terms, device_terms, host_required, device_required, hr, dr, meta["resident"], src.cold,
                               frac_bytes, tuple(leaves))


def _meta(admission, compiled, payload, limits):
    t = ndp._types()
    if type(compiled) is not t.compiled:
        raise NativeDenseCudaProfileError("admission does not carry the genuine native CompiledCase")
    try:
        return ndp._scope_metadata(admission, compiled, t, limits, payload)
    except ndp.NativeDenseProfileError as exc:
        raise NativeDenseCudaProfileError(str(exc)) from exc
    except (AttributeError, TypeError, KeyError, IndexError) as exc:
        raise NativeDenseCudaProfileError(f"malformed native case binding: {type(exc).__name__}: {str(exc)[:120]}") from exc


def plan_native_dense_cuda(admission, *, device_id: int, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int,
                           backend: str = "cuda", limits: ndp.DenseLimits | None = None) -> NativeDenseCudaPlan:
    """PURE metadata plan. Imports no CuPy, creates no array, queries no device, reads/hashes no source body and does not warm the source
    cache. It performs the same bounded filesystem METADATA census (`os.scandir`/`lstat`, no file body) as the qualified CPU planner for the
    source-audit byte estimate; zero filesystem I/O is NOT claimed."""
    dev, hr, dr, limits, compiled, payload = _check_args(admission, device_id, caller_host_reserve_bytes, caller_device_reserve_bytes,
                                                          backend, limits)
    return _plan(compiled, payload, _meta(admission, compiled, payload, limits), dev, hr, dr)


# --- sealed record ----------------------------------------------------------------------------------------------------------------
_SEALED: "weakref.WeakKeyDictionary[Any, types.MappingProxyType]" = weakref.WeakKeyDictionary()
_FIELDS = ("profile_name", "source_profile", "bundle_id", "package_digest", "syrup_package_digest", "case_identity_sha256",
           "latitude_deg", "route", "shape", "n_voxels", "n_classes", "class_ids", "ledger_kind", "device_id", "h_authority",
           "transfer_evidence", "fingerprint", "memory", "caller_host_reserve_bytes", "caller_device_reserve_bytes", "source",
           "bed", "context", "admission")


class NativeDenseCudaPreparation:
    """Sealed result of `prepare_native_dense_cuda_case` (factory-only; payload in a private read-only registry)."""

    __slots__ = ("__weakref__",)

    def __init__(self, *a, **k):
        raise NativeDenseCudaProfileError("NativeDenseCudaPreparation is created only by prepare_native_dense_cuda_case")

    def _p(self):
        p = _SEALED.get(self) if type(self) is NativeDenseCudaPreparation else None
        if p is None:
            raise NativeDenseCudaProfileError("unauthenticated NativeDenseCudaPreparation (not factory-produced)")
        return p

    def __setattr__(self, name, value):
        raise NativeDenseCudaProfileError("NativeDenseCudaPreparation is immutable")

    def __delattr__(self, name):
        raise NativeDenseCudaProfileError("NativeDenseCudaPreparation is immutable")

    def __copy__(self):
        raise NativeDenseCudaProfileError("NativeDenseCudaPreparation cannot be copied; re-run the factory")

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __reduce_ex__(self, protocol):
        return self.__copy__()

    def __repr__(self):
        return f"<NativeDenseCudaPreparation {'sealed' if _SEALED.get(self) is not None else 'UNAUTHENTICATED'}>"

    def __dir__(self):
        return list(_FIELDS)

    def latitude_binding(self) -> dict[str, Any]:
        return self._p()["admission"].latitude_binding()

    def recheck(self, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int,
                host_available_bytes_fn: Callable[[], int] | None = None,
                device_available_bytes_fn: Callable[[], int] | None = None) -> None:
        """COLD recheck of the INITIAL preparation only (never a step-time guard, never a later evolved bed): same gates, plan, fresh
        readings and nested reserves, fresh admission/source verification, host validation + fresh host fingerprint, device validation +
        fresh device fingerprint, and equality with the sealed fingerprint."""
        p = self._p()
        _recheck(p, caller_host_reserve_bytes, caller_device_reserve_bytes, host_available_bytes_fn, device_available_bytes_fn)


for _n in _FIELDS:
    setattr(NativeDenseCudaPreparation, _n, property(lambda self, _n=_n: self._p()[_n]))
del _n


# --- device/runtime helpers -----------------------------------------------------------------------------------------------------
def _device(dev):
    """CuPy module + current-device gate. Never selects or touches another device."""
    from maple_syrup import subsurface_cuda

    try:
        cp = subsurface_cuda.require_cuda()
        cur = subsurface_cuda.current_device_id(cp)
    except ss.SubsurfaceBackendUnavailableError as exc:
        raise NativeDenseCudaProfileError(f"CUDA unavailable: {exc}") from exc
    if cur != dev:
        raise NativeDenseCudaProfileError(f"the current CUDA device is {cur}, not the requested device {dev}; select the device first "
                                          "(this capability never selects another device)")
    return cp


def _reading(fn, what):
    v = fn()
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 0:
        raise NativeDenseCudaProfileError(f"{what} must return a non-Boolean integer >= 0, got {_r(v)}")
    return int(v)


def _admit(plan, host_fn, device_fn, *, host_extra=0, host_required=None, device_required=None, stage="call"):
    """Fresh host + device readings (host first, then device) against required + reserve; typed memory refusal before the protected stage."""
    hneed = (plan.host_required_bytes if host_required is None else host_required) + plan.caller_host_reserve_bytes + host_extra
    dneed = (plan.device_required_bytes if device_required is None else device_required) + plan.caller_device_reserve_bytes
    h = _reading(host_fn, "host_available_bytes_fn")
    if hneed > h:
        raise NativeDenseCudaMemoryError(
            f"host admission ({stage}) needs {hneed} B (required {hneed - plan.caller_host_reserve_bytes - host_extra}, reserve "
            f"{plan.caller_host_reserve_bytes}) but {h} B are reported; nothing was transferred, hashed or rechecked")
    d = _reading(device_fn, "device_available_bytes_fn")
    if dneed > d:
        raise NativeDenseCudaMemoryError(
            f"device admission ({stage}) needs {dneed} B (required {dneed - plan.caller_device_reserve_bytes}, reserve "
            f"{plan.caller_device_reserve_bytes}) but {d} B are reported; nothing was transferred, hashed or rechecked")
    return h, d


# --- exact content fingerprint --------------------------------------------------------------------------------------------------
def _scalar_token(path, v) -> bytes:
    if v is None or isinstance(v, (bool, str)):
        return repr(v).encode()
    if isinstance(v, (int, np.integer)):
        return repr(int(v)).encode()
    if isinstance(v, (float, np.floating)):
        return float(v).hex().encode()
    raise NativeDenseCudaProfileError(f"{path} holds an unsupported non-array leaf {type(v).__name__}")


def _fingerprint(state, fraction, *, device: bool) -> tuple[str, int, int]:
    """SHA-256 over every physical leaf (state traversal order, then the initial available fraction): path, dtype, shape header then RAW
    bytes streamed in <= HASH_CHUNK_BYTES chunks (host: memoryview, no copy; device: counted bounded D2H per chunk). Returns
    (hexdigest, n_chunks_downloaded, bytes_downloaded). Float reductions are never used as byte identity."""
    h = hashlib.sha256(b"maple_syrup.native_dense_cuda_profile.fingerprint/1")
    n_chunks = n_bytes = 0
    for path, leaf in (*_walk(state, "bed"), ("context.initial_available_fraction", fraction)):
        if not _backend.is_array(leaf):
            h.update(path.encode() + b"=" + _scalar_token(path, leaf) + b"\0")
            continue
        h.update(f"{path}|{np.dtype(leaf.dtype).str}|{tuple(int(s) for s in leaf.shape)}\0".encode())
        flat = leaf.reshape(-1)
        item = max(1, int(np.dtype(leaf.dtype).itemsize))
        step = max(1, _ls.HASH_CHUNK_BYTES // item)
        for a in range(0, int(flat.shape[0]), step):
            part = flat[a:a + step]
            host = _backend.to_host(part) if device else part
            if device:
                n_chunks += 1
                n_bytes += int(host.nbytes)
            h.update(memoryview(np.ascontiguousarray(host)).cast("B"))
    return h.hexdigest(), n_chunks, n_bytes


def _host_fraction(compiled):
    from maple.core.parameters.case_initialization import resolve_initial_available_fraction_array

    cfg = compiled.config
    return resolve_initial_available_fraction_array(compiled.sediment_availability_spec, cfg.geometry,
                                                    [g.class_id for g in cfg.grain_classes.classes])


def _host_stage(compiled, meta):
    """Unchanged CPU validation suite on the caller's host leaves (shell, no copy) + the host fingerprint; releases the host fraction."""
    frac = _host_fraction(compiled)
    ctx = sb.BedContext(compiled.config.geometry, compiled.config.grain_classes, compiled.config.numerics.mass_resolution_kg,
                        compiled.config.topographic_commit, compiled.config.avalanching, frac)
    state = _host_shell(compiled)
    try:
        ndp._validate_bed(state, ctx, compiled, meta)
    except ndp.NativeDenseProfileError as exc:
        raise NativeDenseCudaProfileError(f"host validation refused the compiled case: {exc}") from exc
    digest, _c, _b = _fingerprint(state, frac, device=False)
    return digest


# --- device validation ------------------------------------------------------------------------------------------------------------
def _device_validate(state, ctx, compiled, host_leaves, meta, dev, cp):
    """Structural per-leaf checks + native namespace-aware validators (bounded flag packets; no full-array D2H shadow)."""
    t = ndp._types()
    for name, cls in (("voxel_column", t.dense), ("active_layer", t.active), ("water", t.water),
                      ("sediment_availability", t.availability), ("committed_topography", t.committed)):
        if type(getattr(state, name)) is not cls:
            raise NativeDenseCudaProfileError(f"device {name} is not exactly the native {cls.__name__}")
    if type(state) is not sb.BedState or type(ctx) is not sb.BedContext or sb.is_windowed_bed(state):
        raise NativeDenseCudaProfileError("bed_from_case did not return a genuine Dense BedState/BedContext")
    if type(state.ledger) is not type(compiled.sediment_ledger):
        raise NativeDenseCudaProfileError("device ledger class differs from the compiled case ledger class")
    if ctx.depth_update_rule != "constant_depth":
        raise NativeDenseCudaProfileError("constant_depth is required for SYRUP hydraulic ownership")
    leaves = [(p, a) for p, a in _walk(state, "bed") if _backend.is_array(a)] + [("context.initial_available_fraction",
                                                                                      ctx.initial_available_fraction)]
    if len(leaves) != len(host_leaves):
        raise NativeDenseCudaProfileError("device leaf count differs from the host leaf inventory")
    for (path, arr), (hp, shape, dtype, nbytes) in zip(leaves, host_leaves):
        if path != hp:
            raise NativeDenseCudaProfileError(f"device leaf order differs at {path} vs {hp}")
        if isinstance(arr, np.ndarray) or not _backend.is_device_array(arr):
            raise NativeDenseCudaProfileError(f"{path} is not a device array")
        if arr.dtype != np.dtype(dtype) or tuple(arr.shape) != shape or not arr.flags.c_contiguous or int(arr.nbytes) != nbytes:
            raise NativeDenseCudaProfileError(f"{path} device dtype/shape/layout differs from the host leaf ({arr.dtype} {tuple(arr.shape)})")
        if int(arr.device.id) != dev:
            raise NativeDenseCudaProfileError(f"{path} lives on device {int(arr.device.id)}, not {dev}")
    ny, nx = meta["shape"]
    axes = (ny, nx, meta["n_classes"])
    g, mres = ctx.geometry, ctx.mass_resolution_kg
    try:
        _check_partition(state.active_layer, state.voxel_column, g, mres)
        if tuple(_validate_ledger_state(state.ledger)) != axes:
            raise ValueError("ledger axes differ from the bed")
        _check_availability(state.sediment_availability, state.active_layer, mres)
        _validate_water(state.water, ny, nx, meta["n_classes"])
        if tuple(_validate_committed(state.committed_topography)) != axes:
            raise ValueError("committed topography axes differ from the bed")
        inv = np.asarray(_backend.to_host(sb.bed_inventory(state)), dtype=np.float64)
    except Exception as exc:
        raise NativeDenseCudaProfileError(f"native validator refused the device bed: {type(exc).__name__}: {str(exc)[:200]}") from exc
    from maple_syrup.conservation import reservoir_bound_kg

    want = compiled.expected_total_mass_by_class_kg
    if inv.shape != want.shape or np.max(np.abs(inv - want), initial=0.0) > reservoir_bound_kg(ny * nx * meta["n_voxels"], inv, want):
        raise NativeDenseCudaProfileError("device bed class totals differ from the native expected class totals")


def _zero_mobile_and_pending(state) -> None:
    """Initial airborne/mobile load and pending ledger change must be exactly zero (the CPU scope); fused device predicates, no mask."""
    xp = _backend.array_namespace(state.water.mobile_mass_by_cell_class_kg)
    if int(_backend.to_int(xp.count_nonzero(state.water.mobile_mass_by_cell_class_kg))) \
            or int(_backend.to_int(xp.count_nonzero(state.ledger.pending_bed_mass_change_kg))):
        raise NativeDenseCudaProfileError("an initial airborne/mobile load or pending ledger change is outside this capability")


# --- factory / recheck ------------------------------------------------------------------------------------------------------------
def _gates(admission, device_id, host_reserve, device_reserve, host_fn, device_fn, backend, limits):
    """Arguments -> admission scope -> pure plan -> current-device gate -> (host reader, device reader). No reading/hash/transfer yet."""
    dev, hr, dr, limits, compiled, payload = _check_args(admission, device_id, host_reserve, device_reserve, backend, limits,
                                                          host_fn, device_fn)
    meta = _meta(admission, compiled, payload, limits)
    plan = _plan(compiled, payload, meta, dev, hr, dr)
    cp = _device(dev)  # current-device gate precedes every reading, hash and transfer
    if host_fn is None:  # explicit: a valid callable that is falsy (__bool__ False) must be preserved
        host_fn = ss.default_host_available_bytes
    if device_fn is None:
        from maple_syrup import subsurface_cuda

        device_fn = subsurface_cuda.default_device_available_bytes_fn(cp, dev)
    return compiled, meta, plan, cp, host_fn, device_fn


def _cold_checks(admission, plan, compiled, meta, host_fn, hr):
    """Fresh admission recheck, A1 source verification (own admission + cold hash), host validation + host fingerprint."""
    try:
        admission.recheck()
    except ncp.NativeCaseProfileError as exc:
        raise NativeDenseCudaProfileError(f"cold source/case recheck refused: {exc}") from exc
    try:
        src = dcs.verify_dense_cuda_source(dcs.PROFILE_NATIVE9F, available_bytes_fn=host_fn,
                                           reserve_bytes=hr + dcs.RECORD_RETAINED_BYTES + OWN_RECORD_RETAINED_BYTES)
    except dcs.DenseCudaSourceError as exc:
        raise NativeDenseCudaProfileError(f"dense-CUDA source verification refused: {exc}") from exc
    return src, _host_stage(compiled, meta)


def prepare_native_dense_cuda_case(admission, *, device_id: int, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int,
                                   host_available_bytes_fn: Callable[[], int] | None = None,
                                   device_available_bytes_fn: Callable[[], int] | None = None, backend: str = "cuda",
                                   limits: ndp.DenseLimits | None = None) -> NativeDenseCudaPreparation:
    """Genuine device-runtime-admitted native9f case -> sealed device Dense `(BedState, BedContext)` preparation (see module notes).

    Order: argument/admission/scope refusals -> pure plan -> current-device gate -> fresh host+device readings (whole call) -> cold
    admission recheck -> A1 source verification -> host validation + host fingerprint -> fresh readings (transfer stage) -> ONE
    `bed_from_case(compiled, xp=cupy)` with bracketed counters -> device validation -> device fingerprint == host fingerprint -> seal.
    A failure before the transfer touches nothing on the device; every failure leaves the admission and compiled arrays unmodified, and
    an exact retry is valid."""
    compiled, meta, plan, cp, host_fn, device_fn = _gates(admission, device_id, caller_host_reserve_bytes, caller_device_reserve_bytes,
                                                          host_available_bytes_fn, device_available_bytes_fn, backend, limits)
    hr = plan.caller_host_reserve_bytes
    whole = _admit(plan, host_fn, device_fn, stage="whole call")
    src, host_digest = _cold_checks(admission, plan, compiled, meta, host_fn, hr)
    terms = dict(plan.host_terms)
    retained = dcs.RECORD_RETAINED_BYTES + OWN_RECORD_RETAINED_BYTES
    pre_transfer = _admit(plan, host_fn, device_fn, host_required=terms["stage_transfer_host"] + retained, stage="transfer")
    before = _backend.read_transfer_counters()
    try:
        state, ctx = sb.bed_from_case(compiled, xp=cp)
    except sb.BedIntegrationError as exc:
        raise NativeDenseCudaProfileError(f"bed_from_case refused: {exc}") from exc
    d = _backend.read_transfer_counters().delta(before)
    if d.host_to_device != plan.n_leaf_occurrences or d.host_to_device_bytes != plan.transfer_bytes:
        raise NativeDenseCudaProfileError(
            f"actual H2D counters ({d.host_to_device} transfers, {d.host_to_device_bytes} B) differ from the planned leaf occurrences "
            f"({plan.n_leaf_occurrences}, {plan.transfer_bytes} B); a concurrent transfer occurred or the inventory is wrong")
    _device_validate(state, ctx, compiled, plan.leaves, meta, plan.device_id, cp)
    _zero_mobile_and_pending(state)
    pre = _backend.read_transfer_counters()
    dev_digest, n_chunks, n_bytes = _fingerprint(state, ctx.initial_available_fraction, device=True)
    fp = _backend.read_transfer_counters().delta(pre)
    if dev_digest != host_digest:
        raise NativeDenseCudaProfileError("the device bed content fingerprint differs from the host compiled case")
    evidence = types.MappingProxyType(dict(
        n_leaf_occurrences=plan.n_leaf_occurrences, n_aliased_leaves=plan.n_aliased_leaves, transfer_bytes=plan.transfer_bytes,
        context_fraction_bytes=plan.context_fraction_bytes, h2d_transfers=d.host_to_device, h2d_bytes=d.host_to_device_bytes,
        fingerprint_d2h_chunks=n_chunks, fingerprint_d2h_bytes=n_bytes, fingerprint_d2h_counter_transfers=fp.device_to_host,
        single_cold_transfer=True))
    memory = (("whole_call_host_reading_bytes", whole[0]), ("whole_call_device_reading_bytes", whole[1]),
              ("transfer_stage_host_reading_bytes", pre_transfer[0]), ("transfer_stage_device_reading_bytes", pre_transfer[1]),
              ("host_required_bytes", plan.host_required_bytes), ("device_required_bytes", plan.device_required_bytes),
              *(("host_" + k, v) for k, v in plan.host_terms), *(("device_" + k, v) for k, v in plan.device_terms))
    obj = object.__new__(NativeDenseCudaPreparation)
    _SEALED[obj] = types.MappingProxyType(dict(
        profile_name=PROFILE_NAME, source_profile=ncp.CUDA_SOURCE_PROFILE_NAME, bundle_id=meta["bundle_id"],
        package_digest=admission.package_digest, syrup_package_digest=admission.syrup_package_digest,
        case_identity_sha256=admission.case_identity_sha256, latitude_deg=admission.latitude_deg, route=admission.route,
        shape=meta["shape"], n_voxels=meta["n_voxels"], n_classes=meta["n_classes"], class_ids=meta["class_ids"],
        ledger_kind=meta["ledger_kind"], device_id=plan.device_id, h_authority=H_AUTHORITY, transfer_evidence=evidence,
        fingerprint=dev_digest, memory=memory, caller_host_reserve_bytes=hr,
        caller_device_reserve_bytes=plan.caller_device_reserve_bytes, source=src, bed=state, context=ctx, admission=admission))
    return obj


def _recheck(p, host_reserve, device_reserve, host_fn, device_fn) -> None:
    """COLD initial-only recheck of a sealed record against fresh source/case/device/content (no new transfer, no step-time use)."""
    admission = p["admission"]
    compiled, meta, plan, cp, host_fn, device_fn = _gates(admission, p["device_id"], host_reserve, device_reserve, host_fn, device_fn,
                                                          "cuda", None)
    # whole-call readings: a recheck makes no transfer, so the device side is validation scratch only (the bed is already resident)
    scratch = dict(plan.device_terms)["validation_scratch"]
    _admit(plan, host_fn, device_fn, device_required=scratch, stage="recheck")
    src, host_digest = _cold_checks(admission, plan, compiled, meta, host_fn, plan.caller_host_reserve_bytes)
    dcs.check_dense_cuda_source_hot(p["source"], dcs.PROFILE_NATIVE9F)
    if p["source"].package_sha256 != src.package_sha256:
        raise NativeDenseCudaProfileError("recheck: the source association changed since preparation")
    state, ctx = p["bed"], p["context"]
    _device_validate(state, ctx, compiled, plan.leaves, meta, plan.device_id, cp)
    _zero_mobile_and_pending(state)
    digest, _c, _b = _fingerprint(state, ctx.initial_available_fraction, device=True)
    if digest != p["fingerprint"] or digest != host_digest:
        raise NativeDenseCudaProfileError("recheck: the device bed is no longer the initial bed (content fingerprint differs from the "
                                          "sealed record or the host compiled case)")
