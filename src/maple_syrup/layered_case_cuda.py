"""File-driven layered-water setup on the native MAPLE 9f ORDINARY DENSE bed ON CUDA (CANDIDATE A4, new file).

STATUS: UNQUALIFIED. Written without any GPU/device/CuPy-array execution; only Root hardware qualification can change that.

    cfg   = load_layered_cuda_config("layered_cuda.yaml")          # compute.backend: cuda (the CPU loader still refuses it)
    prep  = native_dense_cuda_profile.prepare_native_dense_cuda_case(admission, device_id=0, ...)      # ONE cold device transfer (A3b)
    plan  = plan_layered_cuda_setup(prep, cfg, caller_host_reserve_bytes=..., caller_device_reserve_bytes=..., lifecycle_extra_reserve_bytes=...,
                                    native_reserve_bytes=...)          # PURE metadata: no CuPy, no device, no array
    setup = prepare_layered_cuda_case(prep, cfg, <same keywords>)
    setup.state            # the genuine initial BOUNDARY lifecycle state on the device bed (H = prep.bed.water.depth_m: the ONE surface-water authority)

It composes the EXISTING production constructors on the CUDA column/owners with the same scientific parameter schema as the CPU setup (`layered_case`) and
adds no water physics, no texture-to-property method, no scheduler/CLI/restart/LEAF bridge, no Windowed support. MAPLE initializes terrain/grains/voxels/active
layer/native water and the latitude (`prep.latitude_binding()` of the case the A3b preparation was built from); LEAF initializes plants separately.

HYBRID cold setup (declared, not an all-GPU initializer): the retained soil-profile texture and the routing surface/pond topology are reduced on the HOST from the
caller's ALIASED host compiled bed (streaming batches; no copy of the voxels, no second permanent host bed), after the A3b cold recheck proved (exact byte
fingerprint) that the device bed is that same initial bed. The texture stays on the HOST and is retained (read-only by convention: the production record is a frozen dataclass whose arrays are not write-locked; not a device texture). The routing graph is the existing HYBRID `RoutingGraph` built from the
host topology with `xp=cupy`: its static metadata (aspect, slope, friction, active, outlet, receiver, level, level_order_host, pit_storage, flat_storage) stays read-only HOST NumPy, only its runtime arrays (conveyance,
active_flat, outlet_flat, level_order, conveyance_lo, donor_position, donor_mask) live on the device. This setup itself downloads no device voxels; the A3b cold `recheck` it calls first streams
EVERY native leaf to the host in <= HASH_CHUNK_BYTES chunks for the exact content fingerprint (cold bytes, not a per-step cost).

Ownership after success: the returned state aliases `prep.bed.water.depth_m` (one authority, asserted by identity); the NativeDenseCudaPreparation, its admission
and the host compiled case are NOT retained (weakref-testable). A failure at any stage leaves the preparation, the device bed, the host compiled arrays and all
files unchanged; an exact retry is valid. Memory: see `plan_layered_cuda_setup` (host and device separate; reading is not a reservation).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from maple_syrup import layered_case as lc
from maple_syrup import layered_checkpoint as lcp
from maple_syrup import layered_storm as ls
from maple_syrup import matrix_infiltration as mi
from maple_syrup import native_dense_cuda_profile as ndc
from maple_syrup import pond_network as pn
from maple_syrup import preferential_infiltration as pi
from maple_syrup import sediment_bed as sb
from maple_syrup import soil_evaporation as se
from maple_syrup import soil_profile_texture as st
from maple_syrup import subsurface as ss
from maple_syrup import water_lifecycle as wl
from maple_syrup import water_lifecycle_cuda as wlc
from maple_syrup.experimental_hydrology import HydraulicControl
from maple_syrup.pond_hydraulics import metadata_routing_graph
from maple_syrup.surface_routing import build_surface_router

__all__ = ["LIFECYCLE_PROFILE", "SETUP_PROFILE", "LayeredCudaSetup", "plan_layered_cuda_setup", "plan_layered_cuda_setup_from_admission",
           "prepare_layered_cuda_case"]

SETUP_PROFILE = "maple_syrup.layered_case_cuda/native9f_dense_file_driven_hybrid_UNQUALIFIED"
LIFECYCLE_PROFILE = wl.CUDA9F_BOUNDARY_PROFILE  # genuine BOUNDARY start; storm-first, carry, interevent E/T and the controlled J reset are the owners'
_F8 = 8
ALLOCATOR_ROUNDING_BYTES = 512  # CuPy rounds each device allocation up to 512 B
GRAPH_DEVICE_LEAVES_ALLOWANCE = 64  # upper bound of leaf occurrences of one routing graph (rounding allowance only)
_Error = lc.LayeredCaseError


# --- argument gates (typed, before any array/hash/device work) ----------------------------------------------------------------------------------------
def _reader(fn: Any, name: str) -> Callable[[], int] | None:
    if fn is not None and not callable(fn):  # explicit: a valid callable that is FALSE-valued (`__bool__` False) is honoured, never replaced
        raise _Error(f"{name} must be callable or None")
    return fn


def _read(fn: Callable[[], int], what: str) -> int:
    v = fn()
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 0:
        raise _Error(f"{what} must return a non-Boolean integer >= 0, got {v!r}")
    return int(v)


def _check_common(config: Any, hr: Any, dr: Any, extra: Any, native: Any, host_fn: Any, device_fn: Any) -> dict:
    hr = lc._checked_count(hr, "caller_host_reserve_bytes")
    dr = lc._checked_count(dr, "caller_device_reserve_bytes")
    extra = lc._checked_count(extra, "lifecycle_extra_reserve_bytes")
    native = lc._checked_count(native, "native_reserve_bytes")
    host_fn = _reader(host_fn, "host_available_bytes_fn")
    device_fn = _reader(device_fn, "device_available_bytes_fn")
    lc._verify_config(config)
    if config.backend != lc.CUDA_BACKEND:
        raise lc.LayeredCaseUnsupportedError(f"the CUDA layered setup needs a config parsed by load_layered_cuda_config (backend 'cuda'), got {config.backend!r}; "
                                             "the CPU setup is layered_case.prepare_layered_case (the config's backend is never relabelled)")
    return dict(hr=hr, dr=dr, extra=extra, native=native, host_fn=host_fn, device_fn=device_fn)


def _check(prep: Any, config: Any, hr: Any, dr: Any, extra: Any, native: Any, host_fn: Any, device_fn: Any) -> dict:
    g = _check_common(config, hr, dr, extra, native, host_fn, device_fn)
    if type(prep) is not ndc.NativeDenseCudaPreparation:
        raise _Error("prep must be a genuine native_dense_cuda_profile.NativeDenseCudaPreparation (a CPU NativeDensePreparation or any other object is refused)")
    try:
        p = prep._p()
    except ndc.NativeDenseCudaProfileError as exc:
        raise _Error(f"unauthenticated device preparation: {exc}") from None
    ny, nx = (int(v) for v in p["shape"])
    lc._check_domain_metadata(config, ny, nx)
    return {**g, "ny": ny, "nx": nx, "V": int(p["n_voxels"]), "K": int(p["n_classes"]), "dev": int(p["device_id"]), "admission": p["admission"]}


def _leaf_split(obj: Any) -> tuple[int, int]:
    """(host bytes, non-host bytes) of every array leaf of a (nested) dataclass/tuple/dict record."""
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes), 0
    if hasattr(obj, "nbytes") and hasattr(obj, "dtype") and hasattr(obj, "shape"):
        return 0, int(obj.nbytes)
    h = d = 0
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        parts = [getattr(obj, f.name) for f in dataclasses.fields(obj)]
    elif isinstance(obj, (tuple, list)):
        parts = list(obj)
    elif isinstance(obj, dict):
        parts = list(obj.values())
    else:
        return 0, 0
    for x in parts:
        a, b = _leaf_split(x)
        h, d = h + a, d + b
    return h, d


# --- memory plan (metadata only) -----------------------------------------------------------------------------------------------------------------------
def plan_layered_cuda_setup(prep: Any, config: Any, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, lifecycle_extra_reserve_bytes: int,
                            native_reserve_bytes: int, host_available_bytes_fn: Callable[[], int] | None = None,
                            device_available_bytes_fn: Callable[[], int] | None = None) -> dict:
    """Named, SEPARATE host and device requirement of `prepare_layered_cuda_case` from METADATA only: no CuPy import, device query, N-sized array, source audit,
    hash, material read or owner. Sequential stages take their max; retained items add. Active counts/shapes come from the certified config (`all` = N).

    Host stages: (R) the A3b cold recheck (`plan_native_dense_cuda`: source audit/verify/host validation/fingerprint; the caller's compiled bytes are borrowed and
    only LISTED); (Z) the routing surface from the host compiled bed (staged mask, expanded scalar friction, host initial fraction 8NK for the real BedContext,
    the `layered_checkpoint` scan constants, surface 8N, ring 8R); (B) the streamed host texture (`estimate_texture_memory`, output retained on the HOST);
    (T) pond-topology + graph CONSTRUCTION (`layered_case.topology_graph_terms`, a code-derived bound of the real builders; the topology is retained on the host,
    the graph's static metadata on the HOST and its runtime arrays on the DEVICE) beside the retained texture; (O) owners: retained texture + topology + the CUDA owner/lifecycle host figures. Device stages: (R)
    validation scratch only (the resident prep bed is BORROWED, not charged); (G) the graph's device runtime arrays; (O) graph + owners (`estimate_layered_memory(cuda).required_bytes`
    - it CONSERVATIVELY includes the native mobile array the borrowed bed already holds, because the lifecycle's own admission counts it - plus the evaporation
    operator, the lifecycle cold figure with the declared extra/native reserves, the device cover/map transients and the initial soil/J/intake state).
    Each nested stage admits itself with the caller reserve against a FRESH reading that already reflects the bytes retained so far; the single up-front admission
    compares the maximum of the stages. A reading is not a reservation. Excluded: CuPy import, NVRTC/JIT, context, allocator pool growth/fragmentation, pinned-pool
    retention, native compile, OS cache and the whole process. Parameter-map files are admitted by the loader (YAML and each map header/size before its body).
    When a reader is supplied the plan compares (host reading, then device reading) and raises `LayeredCaseMemoryError`; the plan itself never queries a device."""
    g = _check(prep, config, caller_host_reserve_bytes, caller_device_reserve_bytes, lifecycle_extra_reserve_bytes, native_reserve_bytes,
               host_available_bytes_fn, device_available_bytes_fn)
    return _plan(g["admission"], config, g, host_available_bytes_fn, device_available_bytes_fn)


def plan_layered_cuda_setup_from_admission(admission: Any, config: Any, *, device_id: int, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int,
                                           lifecycle_extra_reserve_bytes: int, native_reserve_bytes: int, host_available_bytes_fn: Callable[[], int] | None = None,
                                           device_available_bytes_fn: Callable[[], int] | None = None) -> dict:
    """The same PURE metadata plan BEFORE any device transfer: from a genuine device-runtime `NativeCaseAdmission` (native_case_profile.CUDA_SOURCE_PROFILE_NAME;
    a CPU-profile admission is refused by the A3b planner) and the intended device id. It prepares nothing, imports no CuPy, queries no device and equals
    `plan_layered_cuda_setup(prep, ...)` for the preparation later built from the same admission."""
    g = _check_common(config, caller_host_reserve_bytes, caller_device_reserve_bytes, lifecycle_extra_reserve_bytes, native_reserve_bytes,
                      host_available_bytes_fn, device_available_bytes_fn)
    dev = lc._checked_count(device_id, "device_id")
    try:
        r = ndc.plan_native_dense_cuda(admission, device_id=dev, caller_host_reserve_bytes=0, caller_device_reserve_bytes=0)
    except ndc.NativeDenseCudaProfileError as exc:
        raise _Error(f"native device preparation plan refused: {exc}") from None
    ny, nx = (int(v) for v in r.shape)
    lc._check_domain_metadata(config, ny, nx)
    g.update(ny=ny, nx=nx, V=int(r.n_voxels), K=int(r.n_classes), dev=dev, admission=admission)
    return _plan(admission, config, g, host_available_bytes_fn, device_available_bytes_fn)


def _plan(admission: Any, config: Any, g: dict, host_available_bytes_fn: Any, device_available_bytes_fn: Any) -> dict:
    ny, nx, V, K = g["ny"], g["nx"], g["V"], g["K"]
    N, L, S, R = ny * nx, config.n_layers, config.n_species, (ny + 2) * (nx + 2)
    n_active = config.n_active(N)
    hr, dr, extra, native = g["hr"], g["dr"], g["extra"], g["native"]
    try:
        rplan = ndc.plan_native_dense_cuda(admission, device_id=g["dev"], caller_host_reserve_bytes=0, caller_device_reserve_bytes=0)
    except ndc.NativeDenseCudaProfileError as exc:
        raise _Error(f"native device preparation plan refused: {exc}") from None
    compiled = admission.compiled
    borrowed_host = lc._tree_nbytes(compiled.voxel_column) + lc._tree_nbytes(compiled.active_layer)
    maps = config.maps()
    largest_map = max([int(m.array.nbytes) for m in maps], default=0)
    scalar_friction = not lc._is_map(config.friction_factor)
    friction_exp = _F8 * N if scalar_friction else 0
    batch = min(st.DEFAULT_BATCH_CELLS, N)
    tex = st.estimate_texture_memory(representation="dense", backend="numpy", output_namespace="numpy", n_cells=N, n_active=n_active, n_layers=L, n_classes=K,
                                     n_voxels=V, batch_cells=batch, reserve_bytes=0, input_bytes_excluded=borrowed_host)
    tex_retained = int(tex.output_bytes) + N + _F8 * (2 * L + 1)  # + the host active mask and the layer-geometry vectors (edges L+1, thickness L) the record also retains
    scan = lcp.VOXEL_SCAN_MASKS * N * V * K + lcp.VOXEL_SCAN_NV_TEMPORARIES * _F8 * N * V + lcp.CLASS_SCAN_NK_TEMPORARIES * _F8 * N * K + 6 * _F8 * N
    tg = lc.topology_graph_terms(ny, nx, len(config.outlets))
    graph_dev = int(tg["graph_retained_outputs"])
    topo_host_retained = int(tg["retained_bytes"])  # the full CPU bound: conservative for the host (the graph is counted on the host too)
    control = HydraulicControl()
    own = ls.estimate_layered_memory(L, (ny, nx), backend="cuda", limiter=control.limiter, reserve_bytes=0, extra_reserve_bytes=extra, n_species=S,
                                     diffusion_mode=config.layers.diffusion_mode, native_bed=True, n_classes=K, native_reserve_bytes=native)
    evap = se.estimate_evaporation_memory(N, backend="cuda", reserve_bytes=0, extra_reserve_bytes=0)
    rec = wl._cuda_source_record_bytes(LIFECYCLE_PROFILE)
    cm = wlc.estimate_cuda_lifecycle_memory(L, N, host_reserve_bytes=hr, source_cold=True, source_record_bytes=rec)
    life = wlc.compose_entry_plan(L, N, "prepare", device_raw_bytes=cm.cold_device_bytes + extra + native, host_reserve_bytes=0, source_cold=True,
                                  source_record_bytes=rec)
    state_dev = L * N * _F8 + 2 * N * _F8 + _F8 * N + int(own.initialization_bytes)  # soil + J + intake + one map transient + start validation
    maps_dev = 3 * _F8 * N + _F8 * N  # matrix/preferential map transients (sequential) + the lifecycle's device cover copy
    h_stage = {
        "R": int(rplan.host_required_bytes),
        "Z": N + friction_exp + _F8 * N * K + scan + _F8 * N + _F8 * R + largest_map,
        "B": N + friction_exp + _F8 * N + int(tex.required_bytes),
        "T": N + friction_exp + _F8 * N + tex_retained + _F8 * R + int(tg["construction_peak_bytes"]),
        "O": tex_retained + topo_host_retained + int(own.host_bytes) + int(evap.host_bytes) + int(life["host_bytes"]) + largest_map,
    }
    d_stage = {
        "R": int(dict(rplan.device_terms)["validation_scratch"]),
        "G": graph_dev + ALLOCATOR_ROUNDING_BYTES * GRAPH_DEVICE_LEAVES_ALLOWANCE,
        "O": graph_dev + int(own.required_bytes) + int(own.initialization_bytes) + int(own.build_bytes) + int(evap.device_bytes) + int(life["device_bytes"])
        + maps_dev + state_dev,
    }
    host_required, device_required = max(h_stage.values()), max(d_stage.values())
    plan = {
        "schema": lc.SCHEMA, "profile": SETUP_PROFILE, "lifecycle_profile": LIFECYCLE_PROFILE, "backend": "cuda", "device_id": g["dev"], "shape": (ny, nx),
        "n_cells": N, "n_active": n_active, "n_layers": L, "n_classes": K, "n_voxels": V, "n_species": S,
        "host_terms": {"R_cold_recheck": h_stage["R"], "Z_surface_stage": h_stage["Z"], "B_texture_stage": h_stage["B"], "B_texture_required": int(tex.required_bytes),
                       "B_texture_retained_output": int(tex.output_bytes), "B_texture_retained_total": tex_retained, "T_topology_graph_construction_peak": int(tg["construction_peak_bytes"]),
                       "T_topology_host_retained_bound": topo_host_retained, "O_owners_host": int(own.host_bytes), "O_evaporation_host": int(evap.host_bytes),
                       "O_lifecycle_prepare_host": int(life["host_bytes"]), "O_stage": h_stage["O"], "texture_batch_cells": batch,
                       "scalar_friction_expansion": friction_exp, "largest_map_staging_allowance": largest_map,
                       "caller_compiled_host_bytes_borrowed_not_counted": borrowed_host,
                       "loaded_config_map_bytes_already_allocated_not_counted": sum(int(m.array.nbytes) for m in maps)},
        "device_terms": {"R_recheck_validation_scratch": d_stage["R"], "G_graph_retained": graph_dev, "G_stage": d_stage["G"], "O_owners_required": int(own.required_bytes),
                         "O_owners_initialization": int(own.initialization_bytes), "O_owners_build": int(own.build_bytes), "O_evaporation": int(evap.device_bytes),
                         "O_lifecycle_prepare": int(life["device_bytes"]), "O_map_and_cover_transients": maps_dev, "O_initial_state": state_dev, "O_stage": d_stage["O"],
                         "resident_prep_bed_bytes_borrowed_not_counted": int(rplan.transfer_bytes), "extra_reserve_bytes": extra, "native_reserve_bytes": native},
        "topology_graph_terms": dict(tg), "stage_bytes": {"host": dict(h_stage), "device": dict(d_stage)},
        "host_required_bytes": host_required, "device_required_bytes": device_required, "caller_host_reserve_bytes": hr, "caller_device_reserve_bytes": dr,
        "host_required_with_reserve": host_required + hr, "device_required_with_reserve": device_required + dr,
        "texture_location": "host (numpy), retained; arrays are not write-locked by the production builder", "topology_location": "host", "graph_location": "hybrid: host read-only NumPy static metadata (aspect, slope, slope_before_edge_rule, edge_rule_applied, friction_factor, active, outlet, receiver, level, level_order_host, pit_storage, flat_storage) + device runtime arrays (conveyance, active_flat, outlet_flat, level_order, conveyance_lo, donor_position, donor_mask)",
        "actual": {"texture_host_bytes": None, "topology_host_bytes": None, "graph_host_bytes": None, "graph_device_bytes": None},
        "scope": "metadata-only; host and device separate; sequential stages take the max, retained items add; each nested stage admits itself against a fresh reading "
                 "with the caller reserve; a reading is not a reservation; excludes CuPy import, NVRTC/JIT, context, allocator pools/fragmentation, pinned-pool retention, "
                 "native compile, OS cache and the whole process; the resident prep bed and the caller's compiled host case are borrowed and only listed",
    }
    if host_available_bytes_fn is not None:
        h = _read(host_available_bytes_fn, "host_available_bytes_fn")
        if plan["host_required_with_reserve"] > h:
            raise lc.LayeredCaseMemoryError(f"layered CUDA setup needs an estimated {plan['host_required_with_reserve']} B on the HOST (stage plan {host_required} B + "
                                            f"reserve {hr} B) but {h} B are reported; nothing was rechecked, hashed, allocated or built")
    if device_available_bytes_fn is not None:
        d = _read(device_available_bytes_fn, "device_available_bytes_fn")
        if plan["device_required_with_reserve"] > d:
            raise lc.LayeredCaseMemoryError(f"layered CUDA setup needs an estimated {plan['device_required_with_reserve']} B on the DEVICE (stage plan {device_required} B + "
                                            f"reserve {dr} B) but {d} B are reported; nothing was rechecked, hashed, allocated or built")
    return plan


# --- result --------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class LayeredCudaSetup:
    """The genuine initial boundary lifecycle on the device bed plus every retained setup product. `state` aliases `prep.bed.water.depth_m` (ONE surface authority)
    and holds the device `BedState`; the preparation / admission / compiled host case, the host initial fraction, the parameter maps' expansions and the initial
    host H are NOT retained. `texture` and `topology` are HOST records (retained; treat as read-only), `graph` is HYBRID (static metadata host NumPy, runtime arrays device) and the owners' arrays live on the device."""

    lifecycle_operator: Any
    state: Any
    texture: Any
    column: Any
    preferential: Any
    matrix: Any
    evaporation: Any
    router: Any
    graph: Any
    topology: Any
    config: lc.LayeredCaseConfig
    identities: Any
    plan: dict


def _wrap(stage: str, exc: BaseException) -> lc.LayeredCaseError:
    cls = lc.LayeredCaseMemoryError if isinstance(exc, MemoryError) else lc.LayeredCaseError
    return cls(f"{stage} refused: {type(exc).__name__}: {exc}")


def _device_gate(dev: int) -> Any:
    """CuPy module + CURRENT-device gate; never selects or touches another device. Precedes every reading, hash and copy."""
    from maple_syrup import subsurface_cuda

    try:
        cp = subsurface_cuda.require_cuda()
        cur = subsurface_cuda.current_device_id(cp)
    except ss.SubsurfaceBackendUnavailableError as exc:
        raise _Error(f"CUDA unavailable: {exc}") from None
    if cur != dev:
        raise _Error(f"the current CUDA device is {cur}, not the preparation's device {dev}; select that device first (this setup never selects another device)")
    return cp


def _admit_stage(plan: dict, host_fn: Callable[[], int], device_fn: Callable[[], int], host_need: int, device_need: int, stage: str) -> None:
    hr, dr = plan["caller_host_reserve_bytes"], plan["caller_device_reserve_bytes"]
    h = _read(host_fn, "host_available_bytes_fn")
    if host_need + hr > h:
        raise lc.LayeredCaseMemoryError(f"layered CUDA setup ({stage}) needs {host_need + hr} B on the HOST (stage {host_need} B + reserve {hr} B) but {h} B are reported; "
                                        "nothing protected was done")
    d = _read(device_fn, "device_available_bytes_fn")
    if device_need + dr > d:
        raise lc.LayeredCaseMemoryError(f"layered CUDA setup ({stage}) needs {device_need + dr} B on the DEVICE (stage {device_need} B + reserve {dr} B) but {d} B are "
                                        "reported; nothing protected was done")



def _host_stage(compiled: Any, config: lc.LayeredCaseConfig, plan: dict, cp: Any) -> tuple[Any, Any, Any]:
    """The HYBRID cold host stage on the ALIASED host compiled bed: routing surface (the real `current_routing_surface_m` with a host `BedContext` from the same public
    MAPLE calls the A3b preparation uses) -> retained HOST texture (streaming batches; mass/overlap weighted once; no PTF) -> pond topology (host) -> routing graph
    (`xp=cp`: host construction; the static metadata stays host NumPy and only the runtime arrays move to the device; with `cp=numpy` it is the CPU graph, used only by the CPU equivalence control). Construction-only arrays are released
    before return. Writes nothing to the caller's arrays."""
    ny, nx = (int(v) for v in plan["shape"])
    mask = np.ones((ny, nx), dtype=bool) if config.active_mask is None else np.array(config.active_mask.array, dtype=bool)
    friction2d = config.friction_factor.array if lc._is_map(config.friction_factor) else np.full((ny, nx), float(config.friction_factor))
    thickness = np.array(config.layers.thickness_m)
    try:
        from maple.core.parameters.case_initialization import resolve_initial_available_fraction_array

        cfg = compiled.config
        frac = resolve_initial_available_fraction_array(compiled.sediment_availability_spec, cfg.geometry, [c.class_id for c in cfg.grain_classes.classes])
        hctx = sb.BedContext(cfg.geometry, cfg.grain_classes, cfg.numerics.mass_resolution_kg, cfg.topographic_commit, cfg.avalanching, frac)
        shell = sb.BedState(compiled.voxel_column, compiled.active_layer, compiled.water, compiled.sediment_ledger, compiled.committed_topography,
                            compiled.sediment_availability)
        z = np.array(sb.current_routing_surface_m(shell, hctx), dtype=np.float64)
        dx = float(hctx.geometry.dx_m)
        geometry, grains, mres = cfg.geometry, cfg.grain_classes, cfg.numerics.mass_resolution_kg
    except (sb.BedIntegrationError, ValueError, KeyError, AttributeError, TypeError) as exc:
        raise _wrap("host routing surface of the compiled native bed", exc) from None
    del frac, hctx, shell
    try:
        texture = st.build_soil_profile_texture(compiled.voxel_column, compiled.active_layer, geometry, grains, mres, tuple(float(t) for t in thickness),
                                                active_mask=mask, memory_budget_bytes=int(plan["host_terms"]["B_texture_required"]), backend="numpy",
                                                batch_cells=int(plan["host_terms"]["texture_batch_cells"]))
    except (st.SoilProfileTextureError, st.SoilProfileMemoryError, sb.BedIntegrationError) as exc:
        raise _wrap("retained actual soil-profile texture", exc) from None
    try:
        full = np.full((ny + 2, nx + 2), float(z.max()) + config.closed_ring_rise_m)
        full[1:-1, 1:-1] = z
        faces = []
        for (row, col, dr_, dc_, drop) in config.outlets:
            full[row + dr_ + 1, col + dc_ + 1] = z[row, col] - drop
            faces.append(pn.OutletFace(row, col, dr_, dc_))
        topo = pn.build_pond_topology(full, mask, friction2d, dx, faces, allow_closed_components=not faces)
        graph = metadata_routing_graph(topo, cp)
    except (pn.PondNetworkError, sb.BedIntegrationError, ValueError) as exc:
        raise _wrap("pond topology / routing graph", exc) from None
    return texture, topo, graph


def prepare_layered_cuda_case(prep: Any, config: Any, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, lifecycle_extra_reserve_bytes: int,
                              native_reserve_bytes: int, host_available_bytes_fn: Callable[[], int] | None = None,
                              device_available_bytes_fn: Callable[[], int] | None = None) -> LayeredCudaSetup:
    """Cold, initial-only setup. Order: typed arguments + config/prep authentication + metadata refusals (no N-sized array) -> pure metadata plan -> CURRENT-device
    gate (before any reading, hash or copy) -> fresh HOST then DEVICE readings and the whole-call admission -> A3b cold `recheck` (fresh source/case/device
    fingerprint; never on an evolved bed) -> host routing surface -> retained host texture -> pond topology (host) + hybrid routing graph (host static + device runtime) -> verified retained bytes
    and the owner-stage readings -> column / preferential / matrix / evaporation / router / `prepare_lifecycle(CUDA9F_BOUNDARY_PROFILE)` -> initial soil state and
    history -> `start_lifecycle` on the device bed. Failures leave the preparation, device bed, host compiled arrays and files unchanged."""
    g = _check(prep, config, caller_host_reserve_bytes, caller_device_reserve_bytes, lifecycle_extra_reserve_bytes, native_reserve_bytes,
               host_available_bytes_fn, device_available_bytes_fn)
    try:
        plan = plan_layered_cuda_setup(prep, config, caller_host_reserve_bytes=g["hr"], caller_device_reserve_bytes=g["dr"],
                                       lifecycle_extra_reserve_bytes=g["extra"], native_reserve_bytes=g["native"])
    except lc.LayeredCaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _wrap("memory plan", exc) from None
    cp = _device_gate(g["dev"])
    host_fn = ss.default_host_available_bytes if g["host_fn"] is None else g["host_fn"]
    if g["device_fn"] is None:
        from maple_syrup import subsurface_cuda

        device_fn = subsurface_cuda.default_device_available_bytes_fn(cp, g["dev"])
    else:
        device_fn = g["device_fn"]
    hr, dr, extra, native = g["hr"], g["dr"], g["extra"], g["native"]
    ny, nx = g["ny"], g["nx"]
    N, L = ny * nx, config.n_layers
    _admit_stage(plan, host_fn, device_fn, plan["host_required_bytes"], plan["device_required_bytes"], "whole call")

    def net_device() -> int:
        return max(_read(device_fn, "device_available_bytes_fn") - dr, 0)

    try:
        prep.recheck(caller_host_reserve_bytes=hr, caller_device_reserve_bytes=dr, host_available_bytes_fn=host_fn, device_available_bytes_fn=device_fn)
    except ndc.NativeDenseCudaProfileError as exc:
        raise _wrap("cold native device bed recheck", exc) from None
    bed, ctx = prep.bed, prep.context
    admission = g["admission"]
    compiled = admission.compiled
    thickness = np.array(config.layers.thickness_m)
    by_layer = np.array(config.moisture_by_layer)
    depth0 = by_layer * thickness if config.moisture_units == "theta" else by_layer.copy()
    texture, topo, graph = _host_stage(compiled, config, plan, cp)
    th, td = _leaf_split(texture)
    gh_t, gd_t = _leaf_split(topo)
    gh_g, gd_g = _leaf_split(graph)
    planned_host, planned_graph_dev = int(plan["host_terms"]["T_topology_host_retained_bound"]), int(plan["device_terms"]["G_graph_retained"])
    if th > int(plan["host_terms"]["B_texture_retained_total"]) or td:
        raise lc.LayeredCaseMemoryError(f"internal: the retained texture holds host {th} B / device {td} B but {plan['host_terms']['B_texture_retained_total']} B host were planned "
                                        "(plan underestimated); no owner was built")
    if gh_t + gh_g > planned_host or gd_g + gd_t > planned_graph_dev:
        raise lc.LayeredCaseMemoryError(f"internal: the topology/graph retained host {gh_t + gh_g} B / device {gd_g + gd_t} B exceed the planned {planned_host} B / "
                                        f"{planned_graph_dev} B (plan underestimated); no owner was built")
    h2 = max(plan["stage_bytes"]["host"]["O"] - planned_host + gh_t + gh_g, 0)
    d2 = max(plan["stage_bytes"]["device"]["O"] - planned_graph_dev + gd_g + gd_t, 0)
    _admit_stage(plan, host_fn, device_fn, h2, d2, "owners")
    plan = {**plan, "actual": {"texture_host_bytes": th, "topology_host_bytes": gh_t, "graph_host_bytes": gh_g, "graph_device_bytes": gd_g + gd_t}}
    # --- device owners ---------------------------------------------------------------------------------------------------------------------------------
    from maple.core.backend import to_device

    def dflat(v: Any) -> Any:
        return to_device(np.ascontiguousarray(v.array.reshape(-1)), cp) if lc._is_map(v) else float(v)

    def dfull(v: Any) -> Any:
        return to_device(np.ascontiguousarray(v.array.reshape(-1)), cp) if lc._is_map(v) else cp.full(N, float(v), dtype=np.float64)

    try:
        flat_mask = to_device(np.ascontiguousarray(np.asarray(topo.active, dtype=bool).reshape(-1)), cp)
        column = ss.prepare_column(config.layers, flat_mask, topo.dx_m * topo.dx_m, backend="cuda", reserve_bytes=0, available_bytes_fn=net_device)
        pref = config.preferential
        species = pref["species"]
        rate = np.inf if pref["max_intake_rate_m_s"] == "unlimited" else dflat(pref["max_intake_rate_m_s"])
        preferential = pi.prepare_preferential(
            column, a3=dflat(pref["a3"]), max_fraction_per_hour=pref["max_fraction_per_hour"], max_intake_rate_m_s=rate,
            background_receiver_weights=list(pref["background_receiver_weights"]), species_receiver_weights=[list(sp["receiver_weights"]) for sp in species],
            species_contributions=(dfull(sp["contribution"]) for sp in species) if species else None)  # lazily: one (N,) device array at a time
        del rate
        matrix = mi.prepare_matrix(column, conductivity_m_s=dflat(config.matrix[0]), suction_m=dflat(config.matrix[1]), vegetation_factor=dflat(config.matrix[2]))
        evaporation = se.prepare_evaporation(column, config.partition)
        router = build_surface_router("local_inertial", graph, backend="cuda", geometry=topo.geometry, control=HydraulicControl(), reserve_bytes=0,
                                      available_bytes_fn=net_device)
        cover = dfull(config.cover)
        lop = wl.prepare_lifecycle(column, preferential, matrix, router, graph, topo, evaporation, cover, bed=bed, context=ctx, native_reserve_bytes=native,
                                   extra_reserve_bytes=extra, cuda_profile=LIFECYCLE_PROFILE, host_available_bytes_fn=host_fn, host_reserve_bytes=hr)
        del cover
        if config.moisture_units == "theta":
            soil = ss.uniform_state(column, list(by_layer), 0.0)
        else:
            d1 = to_device(np.ascontiguousarray(depth0), cp)
            soil = ss.column_state(column, cp.ascontiguousarray(cp.broadcast_to(d1[:, None], (L, N))), 0.0)
            del d1
        history = mi.wetting_history(matrix, wetting_m=dflat(config.wetting_history_m), cumulative_intake_m=dflat(config.cumulative_intake_m))
        state = wl.start_lifecycle(lop, bed=bed, soil=soil, history=history, t_s=0.0)
    except (ss.SubsurfaceError, ss.SubsurfaceMemoryError, ls.LayeredStormError, wl.LifecycleError, MemoryError, ValueError) as exc:
        raise _wrap("water owner construction", exc) from None
    if state.payload.surface_m is not bed.water.depth_m:
        raise lc.LayeredCaseError("internal: the lifecycle surface does not alias the device bed's water depth (one authority required)")
    tex_src = json.dumps(texture.source, sort_keys=True, default=str, separators=(",", ":"))
    identities = MappingProxyType({
        "config": config.identity, "layered_case_schema": lc.SCHEMA, "setup_profile": SETUP_PROFILE, "lifecycle_profile": LIFECYCLE_PROFILE,
        "native_profile": prep.profile_name, "native_source_profile": prep.source_profile, "native_bundle_id": prep.bundle_id,
        "native_package_digest": prep.package_digest, "syrup_package_digest": prep.syrup_package_digest, "case_identity_sha256": prep.case_identity_sha256,
        "latitude_deg": prep.latitude_deg, "latitude_binding": prep.latitude_binding(), "route": prep.route, "device_id": prep.device_id,
        "h_authority": prep.h_authority, "bed_fingerprint_sha256": prep.fingerprint, "lifecycle_operator_sha256": lop.sha256,
        "texture_source_sha256": hashlib.sha256(tex_src.encode()).hexdigest(), "backend": "cuda"})
    return LayeredCudaSetup(lop, state, texture, column, preferential, matrix, evaporation, router, graph, topo, config, identities, plan)
