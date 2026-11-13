"""CUDA EXECUTION HELPER of the floating porous column (`maple_syrup.floating_soil_layers`, backend "cuda"). Component only; AUTHOR-WRITTEN, NOT RUN ON A DEVICE (Root qualifies it on hardware). No physical source lives here:
the kernels are `subsurface_cuda_floating.render_resident_source()` (ONE shared body `subsurface_cuda._BODY` through the floating geometry readers; reconciliation from its anchored slices).

Residency/traffic contract: every raw map, the mask, the bottom-open flags, the soil depth and the water state live on ONE device and are private to the owner/state (logical immutability by contract, NOT external tamper-proofing).
Cold factory: the caller's device arrays are copied device-to-device, the O(L) edges and nominal separations are uploaded ONCE (counted), the resident validation kernel checks every raw value and reports ONE 32-byte packet, and the cold identity hash
is streamed through one bounded host buffer (separate, declared traffic). HOT step: no host-to-device copy, one 72-byte device-to-host packet read (raw `PACKET_WORDS x 8` B and the rebuilt nine-value float packet are both counted in the host
memory model). HOT reconcile: one 48-byte packet; NO content hash, no download of water or properties (the receipt carries capability/generation identity, never a content SHA). The reduction is the fixed all-lane block tree + one-block finalize
(no CUB, no CuPy reduction, no full-width rows). The current device is gated BEFORE any quota reader, copy or accounting. Importing this module never imports CuPy or touches a device.
"""

from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np

from maple_syrup import subsurface as ss
from maple_syrup import subsurface_cuda_floating as cf
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = ["HASH_CHUNK_BYTES", "check_launch_ranges", "current_device", "device_reader", "load_kernels", "moisture", "n_blocks", "operation_terms", "prepare_device", "run_reconcile", "run_step", "geometry_view_maps", "selection_counts", "state_check", "storage_depth",
           "stream_hash"]

THREADS = cf.THREADS
FINAL_THREADS = cf.FINAL_THREADS
MAX_GRID = 2**31 - 1
HASH_CHUNK_BYTES = 1 << 20
STEP_HOST_PACKET_BYTES = 8 * cf.STEP_PACKET_WORDS + 8 * ss.PACKET_LENGTH  # raw uint64 packet + rebuilt float packet, both live in `run_step`
RECONCILE_HOST_PACKET_BYTES = 8 * cf.RECONCILE_PACKET_WORDS + 8 * cf.RECONCILE_PACKET_WORDS
CHECK_HOST_BYTES = 8 * cf.CHECK_WORDS
_LOCK = threading.Lock()
_FUNCTIONS: dict[int, SimpleNamespace] = {}


def _rc():
    from maple_syrup import soil_water_remap_cuda as rc  # the S1 device probes (require_cuda, current device id, device available-bytes reading)

    return rc


def current_device(cp) -> int:
    return _rc().current_device_id(cp)


def device_reader(cp):
    return _rc().default_device_available_bytes_fn(cp)


def n_blocks(n_cells: int) -> int:
    return (int(n_cells) + THREADS - 1) // THREADS


def check_launch_ranges(n_layers: int, n_cells: int) -> None:
    if not (1 <= n_layers <= 2**31 - 1):
        raise ss.SubsurfaceError(f"n_layers = {n_layers} does not fit the 32-bit kernel argument")
    if not (1 <= n_cells and n_blocks(n_cells) <= MAX_GRID):
        raise ss.SubsurfaceError(f"n_cells = {n_cells} exceeds the CUDA launch grid limit of {MAX_GRID} blocks of {THREADS} threads")


def load_kernels(cp, device_id: int) -> SimpleNamespace:
    """Compile/load once per process and device (functions only, never an owner/state/array). A failure is a typed setup error, never a fallback."""
    with _LOCK:
        fns = _FUNCTIONS.get(device_id)
        if fns is not None:
            return fns
        try:
            module = cp.RawModule(code=cf.render_resident_source(), options=cf.COMPILE_OPTIONS, backend="nvrtc")
            got = {name: module.get_function(name) for name in cf.RESIDENT_KERNELS}
            for fn in got.values():
                if int(dict(fn.attributes).get("max_threads_per_block", 0)) < THREADS:
                    raise ss.SubsurfaceBackendUnavailableError(f"a floating kernel supports fewer than {THREADS} threads per block")
        except ss.SubsurfaceBackendUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ss.SubsurfaceBackendUnavailableError(f"the CUDA floating kernels failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
        ns = SimpleNamespace(**{n.replace("maple_syrup_floating_", ""): got[n] for n in cf.RESIDENT_KERNELS})
        _FUNCTIONS[device_id] = ns
        return ns


# ---- memory model --------------------------------------------------------------------------------------------------------------------------------------------------------------------
def kind_bytes(kind: str, rows: int, n: int) -> int:
    return 0 if rows <= 0 else {"scalar": 8, "shared": 8 * rows, "cell": 8 * rows * n}[kind]


def retained_terms(L: int, N: int, layouts: dict) -> dict:
    """Device bytes retained by the owner (exact payload): raw theta/interface maps by layout, edges (L+1), nominal separations (max(L-1,1)), soil depth (N), mask (N bool), bottom-open flags (N uint8). No (L, N) derived map."""
    f8 = 8
    t = {"theta_maps": sum(kind_bytes(layouts[k], L, N) for k in ("theta_residual", "theta_field_capacity", "theta_saturation")), "interface": kind_bytes(layouts["interface"], L - 1, N),
         "edges": (L + 1) * f8, "sep_nominal": max(L - 1, 1) * f8, "soil_depth": N * f8, "mask": N, "bottom_open": N}
    t["total"] = sum(t.values())
    return t


def operation_terms(operation: str, n_layers: int, n_cells: int) -> dict:
    """Exact payload byte terms of one CUDA operation (allocator rounding, NVRTC, the context and the interpreter are excluded and declared). `outputs` survive the call, `scratch` is released after the synchronising read,
    `host` is the call's own host allocations (without the Python allowance); `shared` is static shared memory per block (not a global allocation).
      state_copy   device 8LN (private copy of the caller's water); host 0
      state_check  device 8*CHECK_WORDS diag; host CHECK_HOST_BYTES
      prepare      device 8*CHECK_WORDS diag (+ the retained terms, counted by the factory); host CHECK_HOST_BYTES + the cold hash buffer
      moisture     device 8LN out; host 0
      storage      device 8*nb partials + 8 diag; host 8
      step         device outputs (LN + 2(L-1)N + 4N)*8 + scratch (7*8*nb partials + PACKET); host STEP_HOST_PACKET_BYTES (raw 72 + rebuilt 72)
      reconcile    device outputs (LN + (L-1)N + 2N)*8 + scratch (RECONCILE_ROWS*8*nb partials + 8*RECONCILE_PACKET_WORDS = 56); host RECONCILE_HOST_PACKET_BYTES"""
    L, N = int(n_layers), int(n_cells)
    nb, f8 = n_blocks(N), 8
    if operation == "state_copy":
        t = dict(outputs=f8 * L * N, scratch=0, host=0, shared=0)
    elif operation == "state_check":
        t = dict(outputs=0, scratch=f8 * cf.CHECK_WORDS, host=CHECK_HOST_BYTES, shared=0)
    elif operation == "prepare":
        t = dict(outputs=0, scratch=f8 * cf.CHECK_WORDS, host=CHECK_HOST_BYTES + HASH_CHUNK_BYTES, shared=0)
    elif operation == "moisture":
        t = dict(outputs=f8 * L * N, scratch=0, host=0, shared=0)
    elif operation == "storage":
        t = dict(outputs=0, scratch=f8 * nb + f8, host=f8, shared=f8 * THREADS)
    elif operation in ("step", "step_with_recharge"):
        t = dict(outputs=(L * N + 2 * (L - 1) * N + 4 * N) * f8, scratch=cf.STEP_ROWS * f8 * nb + 8 * cf.STEP_PACKET_WORDS, host=STEP_HOST_PACKET_BYTES, shared=cf.STEP_ROWS * f8 * THREADS)
    elif operation == "reconcile":
        t = dict(outputs=(L * N + (L - 1) * N + 2 * N) * f8, scratch=cf.RECONCILE_ROWS * f8 * nb + 8 * cf.RECONCILE_PACKET_WORDS, host=RECONCILE_HOST_PACKET_BYTES, shared=cf.RECONCILE_ROWS * f8 * THREADS)
    else:
        raise ss.SubsurfaceError(f"unknown CUDA operation {operation!r}")
    t["n_blocks"] = nb
    t["device"] = t["outputs"] + t["scratch"]
    return t


# ---- argument binding ------------------------------------------------------------------------------------------------------------------------------------------------------------------
_KIND_STRIDES = {"scalar": lambda n: (0, 0), "shared": lambda n: (1, 0), "cell": lambda n: (n, 1)}


def _st(kind: str, n: int) -> tuple:
    a, b = _KIND_STRIDES[kind](int(n))
    return np.int64(a), np.int64(b)


def _maps_args(column) -> tuple:
    m, k, N = column._maps, column.layouts, column.n_cells
    return (m["edges"], m["soil_depth"], m["sep_nominal"], m["theta_residual"], *_st(k["theta_residual"], N), m["theta_field_capacity"], *_st(k["theta_field_capacity"], N),
            m["theta_saturation"], *_st(k["theta_saturation"], N))


def _kif_args(column) -> tuple:
    return (column._maps["interface"], *_st(column.layouts["interface"], column.n_cells), np.float64(0.0 if column.diffusion_coefficient is None else column.diffusion_coefficient),
            np.int32(0 if column.diffusion_mode == "constant" else 1))


def _coef() -> float:
    return volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)


def _read_words(diag, count: int) -> np.ndarray:
    from maple.core.backend import to_host

    return np.asarray(to_host(diag), dtype=np.uint64)[:count]


def _flag_report(words: np.ndarray) -> tuple[int, int, int | None, int]:
    orf, count = int(words[0]), int(words[1])
    if not count:
        return 0, 0, None, 0
    packed = int(words[2])
    return orf, count, packed >> 8, packed & 0xFF


def run_step(cp, column, water_in, added, pet, share, dt: float, pulse: float | None):
    """One resident floating step (admission, device gate, type/time checks done by the caller). Returns `(StepArrays with rows=None, nine-value float packet)`; fresh outputs; inputs and retained maps are only read."""
    L, N = int(column.n_layers), int(column.n_cells)
    check_launch_ranges(L, N)
    fns = load_kernels(cp, column.device_id)
    pulse_on = 0 if pulse is None else 1
    n_if = L - 1
    water_out = cp.empty((L, N), dtype=np.float64)
    percolation = cp.empty((n_if, N), dtype=np.float64)
    diffusion = cp.empty((n_if, N), dtype=np.float64)
    ret, drain, evap, unmet = (cp.empty(N, dtype=np.float64) for _ in range(4))
    nb = n_blocks(N)
    partials = cp.empty(cf.STEP_ROWS * nb, dtype=np.float64)
    diag = cp.zeros(cf.STEP_PACKET_WORDS, dtype=np.uint64)
    share_arr, has_share = (pet, 0) if share is None else (share, 1)
    fns.resident_step((nb,), (THREADS,), (np.int64(N), np.int32(L), water_in, water_in if pulse is not None else added, pet, column._maps["active"], column._maps["bottom_open"], *_maps_args(column), *_kif_args(column),
                                          np.float64(_coef()), np.float64(dt), np.float64(column.theta_evaporation_stress), np.float64(column.evaporation_factor), np.int32(1), np.float64(ss.cell_closure_coefficient(L, pulse_on == 1)),
                                          share_arr, np.int32(has_share), np.float64(0.0 if pulse is None else pulse), np.int32(pulse_on), water_out, percolation, diffusion, ret, drain, evap, unmet, partials, diag))
    fns.resident_step_finalize((1,), (FINAL_THREADS,), (partials, np.int64(nb), diag))
    words = _read_words(diag, cf.STEP_PACKET_WORDS)  # the ONE counted 72-byte read; synchronises the stream: the partials are released only afterwards
    orf, count = int(words[0]), int(words[1])
    packet = np.empty(ss.PACKET_LENGTH, dtype=np.float64)
    packet[:7] = words[2:9].view(np.float64)
    packet[7] = float(orf & -orf) if orf else ss.NO_FLAG
    packet[8] = float(count)
    return ss.StepArrays(water_out, percolation, diffusion, ret, drain, evap, unmet, None), packet


def run_reconcile(cp, column, old_column, water_in, active=None):
    """Resident staged reconciliation of `water_in` (valid for `old_column`) against `column` (the NEW column). The OLD owner's soil depth and saturation map are read in the same per-cell walk to refuse an invalid old state
    (bits 20/21 of the flag word). Returns `(w_out, perc, surface, export, totals(before, after, surface, export, moved percolation), orf, flagged)`; ONE 56-byte read; no content hash."""
    L, N = int(column.n_layers), int(column.n_cells)
    check_launch_ranges(L, N)
    fns = load_kernels(cp, column.device_id)
    w_out = cp.empty((L, N), dtype=np.float64)
    perc = cp.empty((L - 1, N), dtype=np.float64)
    surface, export = cp.empty(N, dtype=np.float64), cp.empty(N, dtype=np.float64)
    nb = n_blocks(N)
    partials = cp.empty(cf.RECONCILE_ROWS * nb, dtype=np.float64)
    diag = cp.zeros(cf.RECONCILE_PACKET_WORDS, dtype=np.uint64)
    om = old_column._maps
    fns.resident_reconcile((nb,), (THREADS,), (np.int64(N), np.int32(L), water_in, column._maps["active"] if active is None else active, column._maps["bottom_open"], *_maps_args(column), om["soil_depth"], om["theta_saturation"],
                                               *_st(old_column.layouts["theta_saturation"], N), np.float64(_coef()), w_out, perc, surface, export, partials, diag))
    fns.resident_reconcile_finalize((1,), (FINAL_THREADS,), (partials, np.int64(nb), diag))
    words = _read_words(diag, cf.RECONCILE_PACKET_WORDS)
    return w_out, perc, surface, export, tuple(float(v) for v in words[2:2 + cf.RECONCILE_ROWS].view(np.float64)), int(words[0]), int(words[1])


def state_check(cp, column, water, *, initial: bool):
    """Resident state validation: ONE 32-byte packet. Returns `(flag OR, flagged cells, first cell or None, its flags)`."""
    L, N = int(column.n_layers), int(column.n_cells)
    check_launch_ranges(L, N)
    fns = load_kernels(cp, column.device_id)
    m, k = column._maps, column.layouts
    diag = cp.zeros(cf.CHECK_WORDS, dtype=np.uint64)
    diag[2] = np.uint64(0xFFFFFFFFFFFFFFFF)
    fns.state_check((n_blocks(N),), (THREADS,), (np.int64(N), np.int32(L), water, m["active"], m["edges"], m["soil_depth"], m["theta_saturation"], *_st(k["theta_saturation"], N), np.float64(_coef()),
                                                 np.int32(0 if initial else 1), diag))
    return _flag_report(_read_words(diag, cf.CHECK_WORDS))


def prepare_device(cp, owned: dict, layouts: dict, L: int, N: int, *, directional: bool, dcoef: float, stress: float, config_open: bool):
    """Resident validation of the owner's freshly copied RAW maps + the bottom-open flags (written into `owned["bottom_open"]`). ONE 32-byte packet; returns the flag report."""
    check_launch_ranges(L, N)
    device_id = current_device(cp)
    fns = load_kernels(cp, device_id)
    diag = cp.zeros(cf.CHECK_WORDS, dtype=np.uint64)
    diag[2] = np.uint64(0xFFFFFFFFFFFFFFFF)
    fns.prepare((n_blocks(N),), (THREADS,), (np.int64(N), np.int32(L), owned["active"], owned["edges"], owned["soil_depth"], owned["sep_nominal"], owned["theta_residual"], *_st(layouts["theta_residual"], N),
                                              owned["theta_field_capacity"], *_st(layouts["theta_field_capacity"], N), owned["theta_saturation"], *_st(layouts["theta_saturation"], N),
                                              owned["interface"], *_st(layouts["interface"], N), np.float64(dcoef), np.int32(1 if directional else 0), np.float64(_coef()), np.float64(stress),
                                              np.int32(1 if config_open else 0), owned["bottom_open"], diag))
    return _flag_report(_read_words(diag, cf.CHECK_WORDS))


def moisture(cp, column, water):
    L, N = int(column.n_layers), int(column.n_cells)
    check_launch_ranges(L, N)
    fns = load_kernels(cp, column.device_id)
    out = cp.empty((L, N), dtype=np.float64)
    fns.moisture((n_blocks(N),), (THREADS,), (np.int64(N), np.int32(L), water, column._maps["edges"], column._maps["soil_depth"], out))
    return out


def storage_depth(cp, column, water) -> float:
    """Sum over ACTIVE cells of the layer-sequential per-cell depth (m): block tree + one-block finalize + one 8-byte read."""
    L, N = int(column.n_layers), int(column.n_cells)
    check_launch_ranges(L, N)
    fns = load_kernels(cp, column.device_id)
    nb = n_blocks(N)
    partials = cp.empty(nb, dtype=np.float64)
    diag = cp.zeros(1, dtype=np.uint64)
    fns.storage((nb,), (THREADS,), (np.int64(N), np.int32(L), water, column._maps["active"], partials))
    fns.storage_finalize((1,), (FINAL_THREADS,), (partials, np.int64(nb), diag))
    return float(_read_words(diag, 1).view(np.float64)[0])


def stream_hash(cp, arrays: list[tuple[str, Any]], header: bytes) -> str:
    """COLD identity hash: every array streamed through ONE bounded host buffer of `HASH_CHUNK_BYTES` (device-to-host copies counted through MAPLE's helper); never called by a step or a reconcile."""
    from maple.core.backend import to_host

    h = hashlib.sha256(header)
    for name, a in arrays:
        h.update(f"{name}|{a.dtype}|{tuple(a.shape)}\0".encode())
        flat = a.reshape(-1)
        per = max(1, HASH_CHUNK_BYTES // a.dtype.itemsize)
        for i in range(0, int(flat.size), per):
            h.update(memoryview(np.ascontiguousarray(to_host(flat[i:i + per]))).cast("B"))
    return h.hexdigest()


# ---- device gate and admission (current device FIRST, then quotas/readers; host and device separately) -----------------------------------------------------------------------------------------
PY_ALLOWANCE_BYTES = 65536
QUOTA_DEFAULTS = {"memory_budget_bytes": None, "reserve_bytes": 0, "available_bytes_fn": None, "host_memory_budget_bytes": None, "host_reserve_bytes": 0, "host_available_bytes_fn": None}


def gate_device(cp, device_id: int | None) -> int:
    """The CURRENT device must be the owner's device BEFORE any quota reader, copy, accounting or kernel (an explicit typed refusal; this module never selects or changes a device). `device_id=None` (the factory) binds to the current one."""
    from maple_syrup.floating_soil_layers import FloatingSoilError

    current = current_device(cp)
    if device_id is not None and current != device_id:
        raise FloatingSoilError(f"this object lives on device {device_id} but the current device is {current}; no implicit cross-device use (nothing was read, copied or allocated)")
    return current


def _count(v: Any, name: str) -> int:
    from maple_syrup.floating_soil_layers import FloatingSoilError

    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 0:
        raise FloatingSoilError(f"{name} must be an integer >= 0, got {v!r}")
    return int(v)


def admit(cp, device_id: int | None, device_bytes: int, host_bytes: int, quota: dict, what: str) -> int:
    """Gate the device, then admit the DEVICE requirement against `memory_budget_bytes` + `reserve_bytes` + a truthful device reading and the HOST requirement (+ 64 KiB allowance) against `host_memory_budget_bytes` + `host_reserve_bytes`
    + a host reading. Nothing is allocated before this returns. Returns the device id."""
    from maple_syrup import spatial_soil_layers as sp
    from maple_syrup.floating_soil_layers import FloatingMemoryError, FloatingSoilError

    dev = gate_device(cp, device_id)
    q = {**QUOTA_DEFAULTS, **quota}
    d_need = int(device_bytes) + _count(q["reserve_bytes"], "reserve_bytes")
    h_need = int(host_bytes) + PY_ALLOWANCE_BYTES + _count(q["host_reserve_bytes"], "host_reserve_bytes")
    for need, budget, reader, side, bname in ((d_need, q["memory_budget_bytes"], q["available_bytes_fn"], "device", "memory_budget_bytes"),
                                              (h_need, q["host_memory_budget_bytes"], q["host_available_bytes_fn"], "host", "host_memory_budget_bytes")):
        if budget is not None and need > _count(budget, bname):
            raise FloatingMemoryError(f"{what} needs an estimated {need} B on the {side} side but the declared {bname} is {budget}; nothing was allocated")
        fn = reader if reader is not None else (device_reader(cp) if side == "device" else sp.default_host_available_bytes)
        avail = fn()
        if isinstance(avail, (bool, np.bool_)) or not isinstance(avail, (int, np.integer)) or int(avail) < 0:
            raise FloatingSoilError(f"the {side} available-bytes reader must return a non-Boolean integer >= 0, got {avail!r}")
        if need > int(avail):
            raise FloatingMemoryError(f"{what} needs an estimated {need} B on the {side} side but only {int(avail)} B are reported available; nothing was allocated")
    return dev


FLAG_TEXT = {1: "a theta input is not finite", 2: "theta values violate 0 <= residual <= field capacity <= saturation <= 1 with residual < saturation", 4: "theta_evaporation_stress must exceed the layer-0 residual",
             8: "the interface input must be finite and >= 0", 16: "a porous slab or pair is not representable (capacity tolerance / derived pair quantities)", 32: "soil depth must be finite and >= 0"}
STATE_FLAG_TEXT = {1: "water must be finite and >= 0 on every cell (inactive placeholders included)", 2: "water exceeds the porous capacity / is not strictly unsaturated (initial)", 4: "a rock slot holds water (must be exactly 0)"}


def raise_flags(report: tuple, texts: dict, what: str) -> None:
    from maple_syrup.floating_soil_layers import FloatingSoilError

    orf, count, first, flags = report
    if not count:
        return
    low = flags & -flags
    raise FloatingSoilError(f"{what}: {texts[low]} ({count} cell(s) flagged, first cell {first}, its flags {flags}; OR {orf})")


# ---- D3: official geometry view maps and the selected-operation helpers ---------------------------------------------------------------------------------------------------------------------------------
def geometry_view_maps(cp, parent, depth_in, quota: dict) -> dict:
    """Owned maps of an official geometry VIEW of a prepared CUDA column: every static map is SHARED (the same device arrays); the depth `(N,)` (a private device-to-device copy of the caller's CuPy array) and the strict
    bottom-open flags are owned. Order: current device FIRST (before any reader, allocation or copy) -> metadata of the caller's depth (exact CuPy type, float64, C-contiguous, shape, same device) -> admission (DEVICE: the
    owned depth 8 N + flags N + the 8 CHECK_WORDS packet; HOST: the 32-byte packet; the reserves are counted ONCE by `admit`) -> copies -> the existing resident validation kernel (finite/non-negative depth, every porous slab/pair
    derived quantity of the new depth; ONE 32-byte packet; it also writes the flags) -> typed refusal BEFORE anything is published. No download of D, water, maps or history."""
    from maple_syrup.floating_soil_layers import FloatingSoilError

    dev = gate_device(cp, parent.device_id)
    L, N = int(parent.n_layers), int(parent.n_cells)
    check_launch_ranges(L, N)
    ns = cp.ndarray
    if type(depth_in) is not ns or depth_in.ndim != 1 or depth_in.dtype != np.float64 or depth_in.shape != (N,) or not depth_in.flags.c_contiguous:
        raise FloatingSoilError(f"soil_depth_m must be a C-contiguous float64 cupy array of shape ({N},) of the owner's device")
    if int(depth_in.device.id) != dev:
        raise FloatingSoilError(f"soil_depth_m lives on device {int(depth_in.device.id)} but the current device is {dev}; no implicit transfer")
    admit(cp, dev, 8 * N + N + 8 * cf.CHECK_WORDS, CHECK_HOST_BYTES, quota, "floating geometry view")
    owned = dict(parent._maps)
    owned["soil_depth"] = cp.array(depth_in, dtype=cp.float64, copy=True)
    owned["bottom_open"] = cp.zeros(N, dtype=cp.uint8)
    report = prepare_device(cp, owned, parent.layouts, L, N, directional=parent.diffusion_mode != "constant", dcoef=0.0 if parent.diffusion_coefficient is None else parent.diffusion_coefficient,
                            stress=parent.theta_evaporation_stress, config_open=parent.bottom_drainage == "open")
    raise_flags(report, FLAG_TEXT, "floating geometry view")
    return owned


def selection_counts(cp, column, selection) -> tuple[int, int]:
    """Validate an operation-selection mask against the column's ACTUAL domain and count it: `selection` must be an exact CuPy bool `(N,)` C-contiguous array of the column's device; returns `(n_selected, n_outside_actual)`
    through two counted 8-byte scalar reads (the caller refuses `n_outside_actual > 0`). Allocates two N-byte temporaries (declared in the operation model)."""
    from maple.core.backend import to_host

    from maple_syrup.floating_soil_layers import FloatingSoilError

    N = int(column.n_cells)
    if type(selection) is not cp.ndarray or selection.dtype != np.bool_ or selection.shape != (N,) or not selection.flags.c_contiguous:
        raise FloatingSoilError(f"selected_mask must be a C-contiguous bool cupy array of shape ({N},)")
    if int(selection.device.id) != int(column.device_id):
        raise FloatingSoilError(f"selected_mask lives on device {int(selection.device.id)} but the column on {column.device_id}; no implicit transfer")
    sel = selection.view(np.uint8)
    outside = sel > column._maps["active"].view(np.uint8)
    n_sel = int(np.asarray(to_host(sel.sum(dtype=np.int64))))
    n_out = int(np.asarray(to_host(outside.sum(dtype=np.int64))))
    return n_sel, n_out
