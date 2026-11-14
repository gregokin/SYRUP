"""H0-VEG-CUDA: the dynamic-vegetation held law record `/4` for a CUDA floating Pond vertical (export from a device vertical, re-preparation on the device). Nothing here was run on a GPU by its author.

Same scope as `floating_held_context_vegetation` (`/3`): water-free law context + the ONE compact sealed cached recipe, no plants / session / bed / per-species N contribution. Record `/4` carries the SAME sealed HOST parts
as `/3` with `backend == "cuda"` and the `/2` `kernel` descriptor (common ABI + resident / matrix / preferential kernel SOURCE hashes). The recorded root identities are the CUDA content hashes of the original device
objects (they differ from the NumPy ones by construction). The cached recipe is HOST data and is staged on the HOST (cold, `stage_cached_inputs`, `backend="numpy"`): the CUDA work is the column / matrix / preferential
preparation through the actual qualified public factories and the actual S0 CUDA law preparation (`prepare_vegetation_law` on the device column: one cold admitted upload of `g`, the per-cell `a3` / maximum-rate maps and each
contribution; there is no per-step transfer).

Order of EVERY GPU transaction: the current device gate FIRST (`fcu.gate_device`, before any reader, array or copy) -> strict host metadata -> composed TWO-POOL admission (DEVICE: reader / reserve / optional budget; HOST: the
mandatory `host_*` quota; each reserve ONCE via `fcu.admit`) -> cold bounded counted transfers -> the public factories only -> identity comparison -> plain parts returned to
`floating_held_context.reconstruct_vegetation_held_context`, which re-checks and MINTS the context (tight window, no callbacks). Both pools and the host active mask are REQUIRED (no implicit NumPy conversion of a device array,
no fallback). The model below is declared arithmetic over the public CUDA estimators (allocator rounding, NVRTC and the CUDA context are excluded). Executable GPU definitions: `tests/phase25_held_context/test_gpu_definitions.py`.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np

from maple_syrup import floating_held_context as fh
from maple_syrup import floating_held_context_cuda as fhc
from maple_syrup import floating_held_context_vegetation as fhv
from maple_syrup import floating_pond as fp
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import native_floating_soil as nfs

HeldContextError = fh.HeldContextError
__all__ = ["export_held_context_vegetation_cuda", "plan_cuda", "prepare_parts_cuda"]


def _need_host_pool(q: dict) -> None:
    if q["host_memory_budget_bytes"] is None:
        raise HeldContextError("a CUDA vegetation held context requires an explicit host_memory_budget_bytes (and host reserve / reader): two separate pools, no default")


def plan_cuda(meta: dict) -> tuple[int, int, int]:
    """`(device_bytes, host_bytes, export_host_bytes)` of the reconstruction / export from a VALIDATED `/4` metadata dict by arithmetic over the public estimators (reserves excluded: `fcu.admit` adds each once)."""
    wh, wvb = fhv._wh(), fhv._wvb()
    view, sz = fhv._check_meta(meta, 1 << 30)
    L, N = meta["n_layers"], meta["n_cells"]
    _count, record = fhv._analytic(view, sz)
    mb = fhv._meta_bytes(meta, sz)
    base_dev, base_host, base_export = fhc.plan(view)
    stage = wh.estimate_cached_stage_bytes_from_meta(meta["vegetation"]["cached"], backend="numpy", source_cold=True, with_mask=True)   # the cache's FINAL public model (HOST: the recipe is staged on the host)
    lw = wvb._load_leaf()
    held = lw.estimate_stream_bytes(sz["N"], sz["M"], sz["n_en"], sz["L"], previous_held=True)["per_next_bytes"]
    free = lw.estimate_stream_bytes(sz["N"], sz["M"], sz["n_en"], sz["L"], previous_held=False)["per_next_bytes"]
    wstage = lw.estimate_stream_bytes(sz["N"], sz["M"], sz["n_en"], sz["L"])["weights_stage_bytes"]
    law = wvb.estimate_law_bytes(L, N, sz["n_en"], binding_resident_bytes=stage["resident_bytes"] + wvb.PY_RECORD_ALLOWANCE_BYTES, stream_per_next_held_bytes=held, stream_per_next_free_bytes=free,
                                 stream_weights_stage_bytes=wstage, backend="cuda", k_suction_cell_arrays=int(meta["mode"] == "texture_ks"), a3_cell_map=sz["ctl"]["a3_layout"] == "cell",
                                 max_rate_cell_map=sz["ctl"]["max_rate_layout"] == "cell")
    retained = fcu.retained_terms(L, N, dict(meta["layouts"]["all"]))["total"] + (8 * N if meta["mode"] == "texture_ks" else 0)
    device = max(base_dev, retained + int(law["peak_bytes"]))
    host = max(base_host + record + 2 * mb, record + 2 * mb + int(stage["peak_bytes"]), record + 2 * mb + int(law["peak_host_bytes"]) + int(stage["resident_bytes"]))
    return int(device), int(host), int(base_export + record + 2 * mb)


def export_held_context_vegetation_cuda(vertical: Any, cached: Any, pins: Any, *, memory_budget_bytes: Any = None, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0,
                                        verify: bool = True, host_memory_budget_bytes: Any = None, host_available_bytes_fn: Callable[[], int] | None = None,
                                        host_reserve_bytes: Any = 0) -> Any:
    """Export the dynamic-vegetation held law context of a genuine CUDA `FloatingPondVertical` as a minted `/4` record (never W/J/I): gate FIRST; metadata / references; ONE two-pool admission; cold chunked counted
    D2H of the device maps; the canonical law re-prepared from the cached recipe on the device column (the vertical's CURRENT operator identities must ALWAYS equal it); with `verify` a device
    reconstruction proves the identities."""
    wvb = fhv._wvb()
    if type(vertical) is not fp.FloatingPondVertical or vertical.backend != "cuda":
        raise HeldContextError("vertical must be a genuine CUDA FloatingPondVertical")
    pins = fhv._check_pins(pins)
    q = fhc._quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    t = vertical.held_template()
    cp, dev = fhc._gate(t.column)  # the CURRENT device FIRST, before any reader / array / copy
    _need_host_pool(q)
    cfg = vertical.config
    if type(cfg) is not nfs.NativeFloatingConfig or cfg.sha256 != t.config_sha256:
        raise HeldContextError("the vertical's native config does not match its held template identity")
    if t.preferential is None:
        raise HeldContextError("the vertical holds no preferential operator: it cannot hold the dynamic vegetation law")
    rec = fhv._cached_check(cached, pins)
    fhc._admit(cp, dev, 0, fhv._cached_export_need(rec), q, "exporting the cached recipe metadata (detached copies)")
    cmeta, carrays = fhv._cached_export(rec)
    meta, refs = fhv._snapshot_base(vertical)
    meta = {k: v for k, v in meta.items() if k != "pref"}
    meta["schema"] = fhv.SCHEMA4
    meta["kernel"] = fhc.kernel_descriptor()
    meta["vegetation"] = {"cached": cmeta, "law": {"g_sha256": "0" * 64, "law_sha256": "0" * 64}}
    meta["identities"] = {**meta["identities"], "pref_sha256": "0" * 64}
    fhv._check_meta(meta, 1 << 30)
    d_need, h_need, export_host = plan_cuda(meta)
    fhc._admit(cp, dev, d_need if verify else 0, export_host + (h_need if verify else 0), q, "exporting the CUDA vegetation held law record (owned host arrays, the cached recipe and the law preparation)")
    out: dict = {}
    for name, a in refs.items():
        out[name] = fhc._d2h(cp, a) if fhc._is_device(a) else np.array(a, copy=True, order="C")
    shape = tuple(cmeta["compact"]["shape"])
    mask2d = out["active"].reshape(shape)
    owned = int(sum(v.nbytes for v in out.values()) + sum(v.nbytes for v in carrays.values()))
    try:
        binding = fhv._wh().cached_binding(cached, expected_binding_sha256=pins.binding_sha256, max_bytes=q["host_memory_budget_bytes"], reserve_bytes=q["host_reserve_bytes"] + owned,
                                           available_bytes_fn=q["host_available_bytes_fn"])
        law = wvb.prepare_vegetation_law(t.column, binding, conductivity_m_s=fhv._law_conductivity(t, cfg), suction_m=cfg.matrix_suction_m, memory_budget_bytes=q["memory_budget_bytes"],
                                         reserve_bytes=q["reserve_bytes"], available_bytes_fn=q["available_bytes_fn"], host_memory_budget_bytes=q["host_memory_budget_bytes"],
                                         host_reserve_bytes=q["host_reserve_bytes"] + owned, host_available_bytes_fn=q["host_available_bytes_fn"], host_active_mask=mask2d,
                                         expected_matrix_op_sha256=t.matrix_op.sha256, expected_preferential_op_sha256=t.preferential.sha256)
    except (wvb.VegetationBindingError, *fhv._ERRORS) as exc:
        raise HeldContextError(f"the vertical's CURRENT CUDA operators are not the canonical law of the cached recipe on its column: {exc}") from None
    meta["identities"] = {**meta["identities"], "matrix_op_sha256": law.matrix_sha256, "pref_sha256": law.preferential_sha256}
    meta["vegetation"]["law"] = {"g_sha256": law.g_sha256, "law_sha256": law.law_sha256}
    out.update({fhv.VEG + k: fh._seal(v) for k, v in carrays.items()})
    for v in out.values():
        v.flags.writeable = False
    record = fhv._mint(json.loads(json.dumps(meta)), out)
    if verify:
        ident = meta["identities"]
        fh.reconstruct_vegetation_held_context(record, fh.ExpectedHeldIdentities(ident["config_sha256"], ident["root_column_sha256"], ident["matrix_op_sha256"], ident["pref_sha256"]), pins,
                                               memory_budget_bytes=q["memory_budget_bytes"], available_bytes_fn=q["available_bytes_fn"], reserve_bytes=q["reserve_bytes"],
                                               host_memory_budget_bytes=q["host_memory_budget_bytes"], host_available_bytes_fn=q["host_available_bytes_fn"], host_reserve_bytes=q["host_reserve_bytes"])
    return record


def prepare_parts_cuda(record: Any, pins: Any, *, memory_budget_bytes: Any = None, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0, host_memory_budget_bytes: Any = None,
                       host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0) -> dict:
    """Re-prepare the CUDA objects from an intact `/4` record (called ONLY by `reconstruct_vegetation_held_context`). Gate FIRST; two pools mandatory; host cold staging of the cached recipe; device column / matrix / law
    through the public factories; plain parts returned (the hook re-checks and mints). Nothing is mutated on failure (device uploads are local and released)."""
    wh, wvb = fhv._wh(), fhv._wvb()
    q = fhc._quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes, memory_budget_bytes, host_memory_budget_bytes)
    cp, dev = fhc._gate()  # FIRST
    _need_host_pool(q)
    meta = record._meta if fhv.record_schema(record) == fhv.SCHEMA4 else None  # the hook verified the record AFTER the gate; no second digest
    if meta is None:
        raise HeldContextError("prepare_parts_cuda serves /4 records only")
    pins = fhv._check_pins(pins)
    arr = record._arrays
    base_arr, veg_arr = fhv._split(arr)
    d_need, h_need, _e = plan_cuda(meta)
    fhc._admit(cp, dev, d_need, h_need, q, "reconstructing the CUDA vegetation held law context (uploads kept alive while the public preparations copy them)")
    comp = fhc._component(q)
    ident, sc = meta["identities"], {k: (None if v is None else fh._unhex(v, k)) for k, v in meta["scalars"].items()}
    overrides = None if meta["overrides"] is None else {k: base_arr[fh._OVERRIDE_PREFIX + k] for k in meta["overrides"]}
    L = meta["n_layers"]
    lay = meta["layouts"]
    cm = meta["vegetation"]["cached"]
    shape = tuple(cm["compact"]["shape"])
    owned = int(sum(a.nbytes for a in arr.values()))
    try:
        cfg = nfs.native_floating_config(layer_thickness_m=base_arr["thickness_m"], grain_texture=base_arr["grain_texture"], organic_matter_percent=base_arr["organic_matter_percent"],
                                         bulk_density_kg_m3=sc["bulk_density_kg_m3"], latent_theta_field_capacity=base_arr["latent_theta_field_capacity"],
                                         latent_theta_saturation=base_arr["latent_theta_saturation"], theta_residual=sc["theta_residual"], theta_evaporation_stress=sc["theta_evaporation_stress"],
                                         evaporation_factor=sc["evaporation_factor"], diffusivity_m2_s=base_arr["diffusivity_m2_s"], bottom_drainage=meta["bottom_drainage"],
                                         matrix_conductivity_m_s=sc["matrix_conductivity_m_s"], matrix_suction_m=sc["matrix_suction_m"], matrix_vegetation_factor=sc["matrix_vegetation_factor"],
                                         initial_wetting_m=sc["initial_wetting_m"], initial_cumulative_intake_m=sc["initial_cumulative_intake_m"], overrides=overrides, backend="cuda",
                                         matrix_conductivity_mode=meta["mode"], latent_saturated_conductivity_m_s=sc["latent_saturated_conductivity_m_s"])
        if cfg.sha256 != ident["config_sha256"]:
            raise HeldContextError("the reconstructed native config does not reproduce the recorded config sha256")

        def raw(name: str, kind: str) -> Any:
            return float(base_arr[name][0]) if kind == "scalar" else fhc._h2d(cp, base_arr[name])  # a scalar is a PYTHON float (a 0-d array would be promoted to (1,))

        fc, sat = raw("theta_field_capacity", lay["theta_field_capacity"]), raw("theta_saturation", lay["theta_saturation"])
        diff = fhc._h2d(cp, base_arr["diffusivity_m2_s"]) if L > 1 else 0.0
        d0, mask = fhc._h2d(cp, base_arr["depth0_m"]), fhc._h2d(cp, base_arr["active"])
        col = fsl.prepare_floating_column(cfg.nominal_edges_m, theta_residual=cfg.theta_residual, theta_field_capacity=fc, theta_saturation=sat, diffusivity_m2_s=diff,
                                          theta_evaporation_stress=cfg.theta_evaporation_stress, evaporation_factor=cfg.evaporation_factor, diffusion_mode="constant", bottom_drainage=cfg.bottom_drainage,
                                          soil_depth_m=d0, active_mask=mask, cell_area_m2=sc["cell_area_m2"], backend="cuda", **comp)
        del fc, sat, diff, d0, mask
        if col.sha256 != ident["root_column_sha256"]:
            raise HeldContextError("the re-prepared CUDA root column does not reproduce the recorded root column sha256")
        ks = fhc._h2d(cp, base_arr["ks_m_s"]) if meta["mode"] == "texture_ks" else None
        mask2d = base_arr["active"].reshape(shape)
        cached = wh.stage_cached_inputs(json.loads(json.dumps(cm)), dict(veg_arr), expected_record_sha256=pins.record_sha256, expected_binding_sha256=pins.binding_sha256,
                                        expected_projection_sha256=pins.projection_sha256, expected_leaf_package_sha256=pins.leaf_package_sha256, expected_external_helpers=pins.external_helpers,
                                        expected_source=pins.source, active_mask=mask2d, backend="numpy", max_bytes=q["host_memory_budget_bytes"], reserve_bytes=q["host_reserve_bytes"] + owned,
                                        available_bytes_fn=q["host_available_bytes_fn"])
        binding = wh.cached_binding(cached, expected_binding_sha256=pins.binding_sha256, max_bytes=q["host_memory_budget_bytes"], reserve_bytes=q["host_reserve_bytes"] + owned,
                                    available_bytes_fn=q["host_available_bytes_fn"])
        law = wvb.prepare_vegetation_law(col, binding, conductivity_m_s=ks if ks is not None else cfg.matrix_conductivity_m_s, suction_m=cfg.matrix_suction_m, memory_budget_bytes=q["memory_budget_bytes"],
                                         reserve_bytes=q["reserve_bytes"], available_bytes_fn=q["available_bytes_fn"], host_memory_budget_bytes=q["host_memory_budget_bytes"],
                                         host_reserve_bytes=q["host_reserve_bytes"] + owned, host_available_bytes_fn=q["host_available_bytes_fn"], host_active_mask=mask2d,
                                         expected_matrix_op_sha256=ident["matrix_op_sha256"], expected_preferential_op_sha256=ident["pref_sha256"])
    except HeldContextError:
        raise
    except (wvb.VegetationBindingError, *fhv._ERRORS) as exc:
        raise HeldContextError(f"a public CUDA factory or the cached recipe refused the record: {exc}") from None
    vm = meta["vegetation"]
    if law.law_sha256 != vm["law"]["law_sha256"] or law.g_sha256 != vm["law"]["g_sha256"]:
        raise HeldContextError("the re-prepared CUDA vegetation law does not reproduce the recorded law identities")
    return {"config": cfg, "column": col, "matrix_op": law.matrix_op, "preferential": law.preferential_op, "depth0": col._maps["soil_depth"], "ks": ks, "active": col._maps["active"],
            "area": sc["cell_area_m2"], "edges": cfg.nominal_edges_m, "host_depth0": base_arr["depth0_m"], "host_active": base_arr["active"], "host_ks": base_arr["ks_m_s"] if ks is not None else None,
            "cached": cached, "law": law}
