"""Phase 2.5 vegetation lifecycle, CUDA branch (AUTHORED, NOT RUN: no hardware execution by the author; Root executes it on its allocated device).

`vegetation_lifecycle.apply_vegetation_generation` dispatches here for a genuine CUDA closed `Boundary` AFTER the same eligibility / live-LEAF / binding / mask / current-bed-surface / generation / idempotency contracts as the CPU path; the CURRENT
device is gated FIRST (before any reader, hash, array or transfer). Everything below reuses the qualified native pieces:

    S0 `prepare_vegetation_law(column, binding, ..., host_active_mask=)`  (the qualified CUDA law preparation: g and per-cell maps admitted cold H2D, LEAF contributions streamed one at a time, the device mask identity verified with the
                                                                         column's own `stream_hash` recipe against the SEALED host mask -- never by comparing the device identity with LEAF's byte hash)
    `fm.rebind_floating_matrix_history`                                  (the qualified CUDA history copy: J/I bit for bit, nothing downloaded, no reset)
    two sealed device covers                                             (splash (ny, nx) and E-shading (N,): prepared COLD ONCE here by the MAPLE counted `to_device`; reused by every GPU event check and day)

Two pools, literal and composed from metadata BEFORE any allocation: DEVICE = the held (old) operators + S0's device peak + the new operators + the rebound history + the two device covers; HOST = S0's host peak + the borrowed binding +
the rebinding staging. The caller reserve of each pool is counted ONCE per stage by `fcu.admit` (current device first, then the budgets and the real readers; a refusal allocates nothing). The cold surface hash streams the bed's committed device
elevation to the host in bounded chunks (a declared 1 MiB buffer, admitted in the host pool; counted by MAPLE's `to_host`); there is no whole-N host surface, no per-hydraulic-step hash / transfer / re-projection and no implicit NumPy conversion of a device array.

What this slice does NOT do: restore, files, frames (the old profiles still fail closed), CPU/GPU numerical equivalence evidence, device peak measurements. Those are Root hardware definitions (`tests/phase25_vegetation_lifecycle/test_cuda_hardware.py`).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import vegetation_lifecycle as vl
from maple_syrup import water_vegetation_binding as wvb

__all__ = ["estimate_apply_bytes_cuda", "finish_cuda", "hash_device_surface"]

_F8 = 8
SURFACE_HASH_CHUNK_BYTES = 1 << 20  # declared host buffer of the cold device-to-host surface hash stream


def surface_hash_chunk_bytes(n_cells: int) -> int:
    """The host chunk buffer of the cold surface hash: `min(8 N, 1 MiB)` (at least one element), the SAME number the admission and the model use."""
    return max(_F8, min(_F8 * int(n_cells), SURFACE_HASH_CHUNK_BYTES))


def hash_device_surface(cp: Any, h: Any, surface: Any, shape: tuple, q: dict, live_host_bytes: int = 0) -> int:
    """Chunked cold hash of the bed's committed DEVICE elevation (float64, C-contiguous, `shape`): raw bytes in C order, i.e. the SAME bytes the binding's surface hash covers. The current device is gated and the host chunk buffer admitted
    (host pool, `live_host_bytes` = the borrowed binding that is live meanwhile + the chunk buffer `surface_hash_chunk_bytes(N)`) BEFORE the first transfer. Returns the counted device-to-host bytes (8 N). Each hashed chunk is released before the next device-to-host copy allocates, so the owned host peak is one chunk buffer."""
    from maple.core.backend import to_host

    if str(surface.dtype) != "float64" or tuple(surface.shape) != tuple(shape) or not surface.flags.c_contiguous:
        raise vl.VegetationLifecycleError("the bed's committed device elevation is not a C-contiguous float64 (ny, nx) array (refused; no hidden conversion)")
    try:
        fcu.admit(cp, None, 0, int(live_host_bytes) + surface_hash_chunk_bytes(int(np.prod(shape))), q, "the cold vegetation surface hash stream (host chunk buffer)")
    except (fsl.FloatingSoilError, fsl.FloatingMemoryError, wvb.VegetationBindingError) as exc:  # the native admission's typed memory refusal is part of the public error family
        raise vl.VegetationLifecycleError(str(exc)) from None
    flat = surface.reshape(-1)
    n, step, moved = int(flat.shape[0]), surface_hash_chunk_bytes(int(flat.shape[0])) // _F8, 0
    for lo in range(0, n, step):
        chunk = np.ascontiguousarray(to_host(flat[lo:lo + step]), dtype=np.float64)
        moved += int(chunk.nbytes)
        h.update(memoryview(chunk).cast("B"))
        del chunk  # release the hashed chunk BEFORE the next transfer allocates: at most ONE chunk buffer is ever owned (the declared 1 MiB), not two
    return moved  # the documented counted device-to-host bytes (8 N); the public return contract


def estimate_apply_bytes_cuda(n_layers: int, n_cells: int, n_species_enabled: int, *, binding_resident_bytes: int, borrowed_bytes: int, old_retained_device_bytes: int, stream_per_next_held_bytes: int,
                              stream_per_next_free_bytes: int, stream_weights_stage_bytes: int = 0, k_suction_cell_arrays: int = 0, a3_cell_map: bool = False,
                              max_rate_cell_map: bool = False) -> dict:
    """Declared DEVICE and HOST bytes of ONE CUDA application, stage by stage from metadata (maximum stage = peak; never a padded sum). The S0 estimator supplies the law stages; the matrix rebind terms are the native
    `operation_terms('rebind')`; the new operators' retained device bytes are the matrix (g + K/psi cell arrays: `8 N (1 + kc)`) plus the preferential `(L N + 2 N) 8` (receiver map + rate + at most a per-cell max rate: the same retained
    formula as `estimate_floating_preferential_memory`, whose estimator serves only the CPU owners)."""
    from maple_syrup import floating_matrix_infiltration_cuda as fmc

    L, N, S = (wvb._count(v, k, mn) for v, k, mn in ((n_layers, "n_layers", 1), (n_cells, "n_cells", 1), (n_species_enabled, "n_species_enabled", 0)))
    br, bo, old = wvb._count(binding_resident_bytes, "binding_resident_bytes"), wvb._count(borrowed_bytes, "borrowed_bytes"), wvb._count(old_retained_device_bytes, "old_retained_device_bytes")
    law = wvb.estimate_law_bytes(L, N, S, binding_resident_bytes=br + bo, stream_per_next_held_bytes=stream_per_next_held_bytes, stream_per_next_free_bytes=stream_per_next_free_bytes, stream_weights_stage_bytes=stream_weights_stage_bytes, backend="cuda",
                                 k_suction_cell_arrays=k_suction_cell_arrays, a3_cell_map=a3_cell_map, max_rate_cell_map=max_rate_cell_map)
    rb = fmc.operation_terms("rebind", L, N)
    new_dev = _F8 * N * (1 + k_suction_cell_arrays) + (L * N + 2 * N) * _F8
    covers_dev = _F8 * N + _F8 * N  # the splash (ny, nx) and the E-shading (N,) device covers
    dev = {"law": old + int(law["peak_bytes"]), "rebind": old + new_dev + int(rb["outputs"]) + int(rb["scratch"]), "seal": old + new_dev + 2 * _F8 * N + covers_dev}
    hash_chunk = surface_hash_chunk_bytes(N)
    host = {"hash": br + bo + hash_chunk, "law": int(law["peak_host_bytes"]), "rebind": br + bo + int(rb["host"]), "seal": br + bo + vl.RECORD_ALLOWANCE_BYTES}  # the cold surface hash (borrowed binding live + ONE chunk buffer) precedes the law
    return {"device_stages": dev, "host_stages": host, "peak_device_bytes": int(max(dev.values())), "peak_host_bytes": int(max(host.values())), "new_operator_device_bytes": int(new_dev),
            "record_device_bytes": int(covers_dev), "surface_hash_host_buffer_bytes": hash_chunk, "surface_hash_d2h_bytes": _F8 * N,
            "scope": "declared; excludes the CUDA context and allocator rounding; the borrowed binding / projection is counted as live, not as new; cold once per application, no per-hydraulic-step transfer"}


def finish_cuda(lineage: Any, expected: Any, binding: Any, cur: Any, cp: Any, q: dict, mb: int, rb: int) -> Any:
    """The CUDA tail of `apply_vegetation_generation` (the caller verified everything up to idempotency): composed two-pool admission, S0 law, history rebinding, device covers, then the shared mint + head LAST."""
    from maple_syrup import native_floating_soil_cuda as ncuda

    root, column = expected.root, expected.root.column
    L, N = int(column.n_layers), int(column.n_cells)
    cfg = root.config
    S = len(binding.enabled_species)
    held, free, wstage = vl.stream_numbers(binding)
    old_dev = vl._retained(root.matrix_op) + vl._retained(root.pref_op)
    est = estimate_apply_bytes_cuda(L, N, S, binding_resident_bytes=binding.resident_bytes, borrowed_bytes=binding.borrowed_bytes, old_retained_device_bytes=old_dev, stream_per_next_held_bytes=held,
                                    stream_per_next_free_bytes=free, stream_weights_stage_bytes=wstage, k_suction_cell_arrays=int(cfg.matrix_conductivity_mode == "texture_ks"), a3_cell_map=binding.controls.a3_layout == "cell",
                                    max_rate_cell_map=binding.controls.max_rate_layout == "cell")
    for stage in ("law", "rebind", "seal"):  # every stage admitted from metadata BEFORE any allocation; each pool's reserve is counted once per stage by `fcu.admit`
        try:
            fcu.admit(cp, column.device_id, est["device_stages"][stage], est["host_stages"][stage], q, f"apply_vegetation_generation/{stage} (CUDA)")
        except (fsl.FloatingSoilError, fsl.FloatingMemoryError, wvb.VegetationBindingError) as exc:
            raise vl.VegetationLifecycleError(str(exc)) from None
    mask = root.host_active  # the SEALED host mask of the genuine root, reshaped metadata-only (a view): S0 hashes it and requires it to reproduce the device column's identity
    if type(mask) is not np.ndarray or mask.dtype != np.bool_ or mask.size != N or not mask.flags.c_contiguous:
        raise vl.VegetationLifecycleError("the CUDA boundary root carries no canonical sealed host active mask (refused)")
    host_mask = mask.reshape(tuple(binding.shape))
    cond = root.ks_m_s if cfg.matrix_conductivity_mode == "texture_ks" else cfg.matrix_conductivity_m_s
    try:
        law = wvb.prepare_vegetation_law(column, binding, conductivity_m_s=cond, suction_m=cfg.matrix_suction_m, memory_budget_bytes=mb, reserve_bytes=rb + old_dev, available_bytes_fn=q["available_bytes_fn"],
                                         host_memory_budget_bytes=q["host_memory_budget_bytes"], host_reserve_bytes=q["host_reserve_bytes"], host_available_bytes_fn=q["host_available_bytes_fn"],
                                         host_active_mask=host_mask)
        wvb.verify_prepared_law(law)
        if law.binding_sha256 != binding.sha256 or law.column_sha256 != column.sha256 or law.backend != "cuda":
            raise vl.VegetationLifecycleError("the prepared law is not this binding's / this CUDA column's")
        new_ret = vl._retained(law.matrix_op) + vl._retained(law.preferential_op)
        comp = {**q, "reserve_bytes": rb + old_dev + new_ret}
        reb = fm.rebind_floating_matrix_history(root.matrix_op, root.history, law.matrix_op, memory_budget_bytes=comp["memory_budget_bytes"], reserve_bytes=comp["reserve_bytes"],
                                                available_bytes_fn=comp["available_bytes_fn"], host_memory_budget_bytes=comp["host_memory_budget_bytes"], host_reserve_bytes=comp["host_reserve_bytes"],
                                                host_available_bytes_fn=comp["host_available_bytes_fn"])
    except (wvb.VegetationBindingError, fsl.FloatingSoilError, fsl.FloatingMemoryError) as exc:  # the NATIVE typed memory refusal (a SubsurfaceMemoryError, not a FloatingSoilError) joins the public family; a real MemoryError (OOM) is never caught
        raise vl.VegetationLifecycleError(f"the vegetation law could not be prepared / the history rebound: {exc}") from None
    if reb.reset or reb.history.operator_sha256 != law.matrix_sha256 or reb.history.namespace != "cupy" or reb.history.device_id != column.device_id:
        raise vl.VegetationLifecycleError("the persistent J/I were not rebound on the device bit for bit (refused)")  # values are copied by the native device rebinding (nothing downloaded); Root's hardware tests diagnose them explicitly
    splash, shade = vl.host_covers(binding, N)
    splash_dev = ncuda._up(cp, np.ascontiguousarray(splash).reshape(tuple(binding.shape)))  # cold ONCE: the resident covers every GPU event check and day reuse (admitted in the 'seal' stage)
    shade_dev = ncuda._up(cp, np.ascontiguousarray(shade))
    return vl.mint_and_install(lineage, expected, binding, cur, law, reb.history, splash_dev, shade_dev, int(splash_dev.nbytes + shade_dev.nbytes))
