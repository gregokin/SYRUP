"""Read-only per-layer water AVAILABILITY of an accepted lifecycle INTEREVENT state (Phase 1 Step 0; design: DesignAvailability.md).

SYRUP owns all physical water. A later consumer (the plant-demand bridge) needs to know how much of the stored water a requested-
transpiration withdrawal can actually take. The native law is `available = max(w - residual_m[l], 0)` on active cells (inactive cells carry
zero requests); it lived only inside the fused withdrawal kernel. This module exposes THE SAME arithmetic, from THE SAME bound, as a pure read:

    avail = layer_availability_m(lop, parent, expected_parent=expect_parent(parent))      # (L, N) float64 depth in m

Nothing is advanced or stored: no time, ledger, request sequence, revision, state or cache changes, no payload is handed out, and the
result is a fresh detached C-contiguous float64 array in the column namespace (host results are read-only; a CuPy array cannot be marked so).
The floor is the residual of the general owner bound to `parent` (`parent.binding.gop.t_op.residual_xp`, the SAME owner `request_transpiration`
uses), proved equal to the column's canonical layer property by the owner's own value check; soil properties are never re-derived here.
Interevent role only. Volume = depth x `lop.column.cell_area_m2` is the caller's single conversion.

Memory (see `AvailabilityPlan` / DesignAvailability.md section 3): fresh bytes only (the borrowed resident state/operator are reported, not charged),
admitted BEFORE any array read, reduction or copy; one `(L, N)` output is the sole grid-sized array (every stage uses `out=`).
Exclusions: interpreter objects, allocator pools, backend reduction scratch beyond the named allowance, JIT, the CUDA context and the whole process.
This is NOT a transpiration, a water transfer, a LEAF bridge, a checkpoint or a field qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import interevent_transpiration as it
from maple_syrup import layered_storm as ls
from maple_syrup import subsurface as ss
from maple_syrup import water_lifecycle as wl

from maple.core.backend import to_host as _to_host  # the genuine native transfer (counted by `read_transfer_counters`); a missing API is a setup failure, never a fallback

__all__ = ["AvailabilityPlan", "layer_availability_m", "required_availability_bytes", "REDUCTION_WORKSPACE_ALLOWANCE_BYTES", "VERSION"]

VERSION = "0.2.0"
_F8 = 8
PACKET_BYTES = 2 * _F8  # the helper's OWN host packet: [min(w), max(w - (saturation + tolerance))]
#: Declared allowance for backend reduction scratch (NumPy: none; CuPy: reduction scratch). An ESTIMATE, not a measured or reserved figure.
REDUCTION_WORKSPACE_ALLOWANCE_BYTES = 4096
SCOPE = ("fresh arrays of one availability read: the owner's bound check (canonical host vectors 16L + comparison L [+ CUDA: device concat 16L and host packet 16L]) "
         "followed by the value pass ((L,N) output + (N,) bool inactive mask + 16 B packet + a named reduction allowance); the stages are sequential so the admitted-side "
         "peak is their MAX; reserves once; borrowed resident state/operator/mask/floor are reported, not charged; excluded: interpreter objects, allocator pools, "
         "backend reduction/concat scratch beyond the allowance, JIT, CUDA context, whole process. A reading is not a reservation.")


def _plan_terms(n_layers: int, n_cells: int, *, cuda: bool, host_reserve: int = 0) -> dict[str, int]:
    """PURE arithmetic of the derived terms (numbers in, numbers out; no operator). Derivation: DesignC1.md / `it._check_t_parameters`, `it._check_cuda_values`.
    Owner bound check: canonical host vectors (<= 2 fresh L doubles) + a bool comparison temporary (L) [CUDA: + the device concat 16L and its host packet 16L, with
    the canonical vectors still live => host peak 33L]. Then the value pass. The stages are sequential, so the peak is a MAX."""
    L, N = int(n_layers), int(n_cells)
    canonical, compare = 2 * _F8 * L, L
    packet_host = 2 * _F8 * L if cuda else 0
    packet_dev = 2 * _F8 * L if cuda else 0
    bound_host = canonical + compare + packet_host
    value_pass = _F8 * L * N + N + PACKET_BYTES + REDUCTION_WORKSPACE_ALLOWANCE_BYTES
    fresh = max(packet_dev if cuda else bound_host, value_pass)  # admitted side: device for CUDA, host for CPU
    host_staging = (max(bound_host, PACKET_BYTES) + int(host_reserve)) if cuda else 0
    return dict(bound_canonical_host_bytes=canonical, bound_comparison_bytes=compare, bound_packet_host_bytes=packet_host,
                bound_packet_device_bytes=packet_dev, bound_check_host_bytes=bound_host, value_pass_bytes=value_pass, fresh_bytes=fresh,
                host_staging_bytes=host_staging)


@dataclass(frozen=True)
class AvailabilityPlan:
    """Pure metadata arithmetic of one `layer_availability_m` call (admitted side: host for numpy/numba, device for cuda)."""

    backend: str
    n_layers: int
    n_cells: int
    output_bytes: int
    mask_bytes: int
    packet_bytes: int  # the helper's own 16 B packet
    workspace_allowance_bytes: int
    bound_canonical_host_bytes: int
    bound_comparison_bytes: int
    bound_packet_host_bytes: int  # CUDA only: the owner's 2L D2H packet
    bound_packet_device_bytes: int  # CUDA only: the owner's 2L device concat
    bound_check_host_bytes: int  # host peak of the owner's value check
    value_pass_bytes: int
    fresh_bytes: int  # admitted-side peak = max(bound stage, value pass)
    reserves_bytes: int  # lop.memory.reserves_bytes (column reserve + lifecycle extra + native reserve, declared once at prepare)
    extra_reserve_bytes: int  # the caller's
    required_bytes: int  # fresh + reserves + extra: what is compared with the column's reported availability
    host_staging_bytes: int  # CUDA: max(bound_check_host, 16) + the declared host reserve once; 0 on CPU (host is the admitted side)
    expected_d2h_bytes: int  # CUDA: 16L (owner packet) + 16 (helper packet) in exactly two transfers; 0 on CPU
    borrowed_resident_bytes: int  # REPORTED only: accepted arrays + statics already resident (never charged against free memory)
    scope: str = SCOPE

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def required_availability_bytes(lop: Any, *, extra_reserve_bytes: Any = 0) -> AvailabilityPlan:
    """PURE metadata plan: only `wl._check_lop(preflight=False)` (no CuPy import, device query, allocation, I/O or source hash)."""
    wl._check_lop(lop, preflight=False)
    extra = wl._count(extra_reserve_bytes, "extra_reserve_bytes")
    L, N = int(lop.n_layers), int(lop.n_cells)
    cuda = lop._cuda is not None
    t = _plan_terms(L, N, cuda=cuda, host_reserve=int(lop._cuda.host_reserve_bytes) if cuda else 0)
    m = lop.memory
    reserves = int(m.reserves_bytes)
    borrowed = int(m.accepted_bytes) + int(m.static_bytes) + int(m.owned_fixed_bytes)
    return AvailabilityPlan(lop.backend, L, N, _F8 * L * N, N, PACKET_BYTES, REDUCTION_WORKSPACE_ALLOWANCE_BYTES, t["bound_canonical_host_bytes"],
                            t["bound_comparison_bytes"], t["bound_packet_host_bytes"], t["bound_packet_device_bytes"], t["bound_check_host_bytes"],
                            t["value_pass_bytes"], t["fresh_bytes"], reserves, extra, t["fresh_bytes"] + reserves + extra, t["host_staging_bytes"],
                            (2 * _F8 * L + PACKET_BYTES) if cuda else 0, borrowed)


def _check_parent(lop: Any, parent: Any, expected_parent: Any) -> None:
    """Metadata/identity only (no array read): genuine minted interevent state of THIS lifecycle operator + the exact expected instance."""
    if type(parent) is not wl.LifecycleState or parent not in wl._MINTED:
        raise wl.LifecycleError(f"parent must be a LifecycleState minted by the lifecycle module, got {type(parent).__name__}")
    if parent.role != "interevent":
        raise wl.LifecycleError(f"layer availability needs an 'interevent' parent (plant demand exists only there), got {parent.role!r}")
    if parent.lop_sha256 != lop.sha256:
        raise wl.LifecycleError("the parent is bound to a different lifecycle operator; refused")
    wl._require_parent(parent, expected_parent)  # exact instance + digest/revision/role; expect_parent alone proves only a minted state
    # Component association (object identity, not just the operator hash: two independently built lifecycles of equal configuration share a
    # hash but not a column): the state's owners must be built on THIS lifecycle's column.
    b = parent.binding
    gop = getattr(b, "gop", None)
    if (type(b) is not wl._Binding or type(gop) is not it.GeneralIntereventOperator or gop.t_op is None or gop.base.op.column is not lop.column
            or gop.t_op.column is not lop.column):
        raise wl.LifecycleError("the parent's general owner is not bound to this lifecycle's column/operator; refused")


def layer_availability_m(lop: Any, parent: Any, *, expected_parent: Any, extra_reserve_bytes: Any = 0) -> Any:
    """Fresh `(L, N)` float64 availability depth (m) = max(stock - native bound residual, 0) on active cells, 0 on inactive cells, of the accepted
    interevent `parent`. Pure read; nothing in `lop`/`parent`/the column is written. Refusals happen before the output exists."""
    plan = required_availability_bytes(lop, extra_reserve_bytes=extra_reserve_bytes)  # metadata + strict scalar, no arrays read
    _check_parent(lop, parent, expected_parent)
    column, xp = lop.column, lop.xp
    # --- admission BEFORE any array read, reduction, copy or allocation ------------------------------------------------------
    if lop._cuda is not None:
        try:
            if callable(getattr(column, "preflight", None)):
                column.preflight()  # the ACTUAL current-device gate before the first host/device availability read (no switch, no fallback)
        except ss.SubsurfaceError as exc:
            raise wl.LifecycleError(str(exc)) from None
        try:
            ls._admit_host_build(plan.host_staging_bytes, lop._cuda.host_available_bytes_fn)
        except ls.LayeredMemoryError as exc:
            raise ls.LayeredMemoryError(f"layer availability host staging: {exc}") from None
        except ls.LayeredStormError as exc:
            raise wl.LifecycleError(str(exc)) from None
    ls._admit(column, lop.shape, plan.required_bytes, "layer availability")
    # --- full (value-touching) validation, still before the output ----------------------------------------------------------
    wl._check_state(lop, parent)  # includes the full _check_lop (CUDA: current device) and the genuine general-owner state checks
    gop = parent.binding.gop
    try:
        it._check_gop_meta(gop)
        it._check_gop_values(gop)  # owner bound == column canonical layer properties (L is tiny); CUDA: packet vs sealed host copy
    except it.GeneralIntereventError as exc:
        raise wl.LifecycleError(f"the bound transpiration owner failed its checks: {exc}") from None
    t = gop.t_op
    if lop.t_op is not None:  # CPU: the lifecycle's own bound and the owner's detached copy must still be the same physics
        if not (np.array_equal(lop.t_op.residual_xp, t.residual_xp) and np.array_equal(lop.t_op.saturation_xp, t.saturation_xp)):
            raise wl.LifecycleError("the owner's transpiration bound differs from the lifecycle's; refused")
    water = parent.payload.soil.water_m
    L, N = int(lop.n_layers), int(lop.n_cells)
    if (water.shape != (L, N) or water.dtype != np.float64 or not water.flags.c_contiguous
            or t.residual_xp.shape != (L,) or t.saturation_xp.shape != (L,) or column.active_u8.shape != (N,)):
        raise wl.LifecycleError("soil/floor/mask extents disagree with the lifecycle shape; refused")
    res, satb = t.residual_xp[:, None], t.saturation_xp[:, None]
    # --- value pass: ONE (L,N) output, every stage in place (no further grid-sized temporary) --------------------------------
    out = xp.empty((L, N), dtype=np.float64)
    xp.subtract(water, satb, out=out)
    hi = out.max()  # max(w - (saturation + tolerance)) must be <= 0 on EVERY cell (inactive included, as the native kernel)
    lo = water.min()  # min(w) must be >= 0 and finite on every cell
    packet = np.asarray(_to_host(xp.stack([lo, hi])), dtype=np.float64)  # the helper's own readback: 16 B (the owner's bound check adds 16L before this)
    if not (np.isfinite(packet[0]) and packet[0] >= 0.0 and np.isfinite(packet[1]) and packet[1] <= 0.0):
        raise wl.LifecycleError("the soil water is not finite, is negative, or exceeds saturation plus tolerance on some cell; refused")
    xp.subtract(water, res, out=out)
    xp.maximum(out, 0.0, out=out)
    inactive = column.active_u8 == 0
    xp.copyto(out, 0.0, where=inactive[None, :])
    del inactive
    if xp is np:
        out.flags.writeable = False
    return out
