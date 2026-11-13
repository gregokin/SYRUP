"""H0-CUDA: the water-free held LAW record for a CUDA floating Pond vertical (record `/2`; export from a device vertical, re-preparation on the device). Nothing here was run on a GPU by its author.

Scope. A pure law context (as `floating_held_context`): no W/J/I, no owner, no bed, no Saxton-Rawls, no file/source proof. Record `/2` carries the SAME sealed HOST parts as `/1` (original thicknesses, true
`overrides` None vs `{}`, texture Ks vector, raw FC/SAT in their layouts, `D0`, mask, explicit `PreferentialSpec` inputs) with `backend == "cuda"` and a separate `kernel` descriptor (the common CUDA ABI/source hash and the
FSL-resident, matrix and preferential kernel SOURCE hashes: stable values computed from rendered source text, never a runtime compile result, device id or architecture). The three recorded root identities are the CUDA
content hashes of the ORIGINAL device objects (column: `fcu.stream_hash` over the device arrays with the `cuda` header; matrix: `fmi._identity_cuda`; preferential: the streamed CUDA hash); they differ from the NumPy
ones by construction (their headers contain `cuda`) and contain no device id; "device/architecture independence" is a code fact of those hashes, NOT a multi-device qualification (only GPU0 exists for Root).

Order of EVERY GPU transaction: the current device gate FIRST (`fcu.gate_device`, before any reader, array or copy) -> strict host metadata (no arrays touched) -> composed TWO-POOL admission (DEVICE: `available_bytes_fn` /
`reserve_bytes` / optional budget, default the CURRENT device's real reading; HOST: `host_*`, default the real host reader; each reserve ONCE via `fcu.admit`) -> cold bounded counted transfers (`HASH_CHUNK_BYTES` chunks
through MAPLE's `to_host`/`to_device`) -> the ACTUAL qualified public factories only (`native_floating_config`, `fsl.prepare_floating_column` with HOST nominal edges, Python-float scalars (a 0-d array would be promoted to
`(1,)`), device FC/SAT/D0/mask/shared vectors; `fmi.prepare_floating_matrix` with the device Ks vector or the explicit scalar; `fpi.prepare_floating_preferential` from the host spec uploaded under the same quota) ->
identity comparison -> plain parts returned to `floating_held_context`, which checks and MINTS the context (tight window, no callbacks). No W/J/I traffic exists anywhere.

Model (declared, modelled payload bytes; allocator rounding, NVRTC and the CUDA context are excluded; see `plan`): DEVICE = the uploaded inputs held ALIVE while the public preparations copy them + the retained
prepared maps (`fcu.retained_terms`) + the matrix and preferential prepare outputs/scratch (`fmc/fpc.operation_terms("prepare")`) + the preferential derivation live set + one transfer chunk; HOST = the config copies
`native_floating_config` makes + the preferential host weights + the cold hash/staging buffers (`fcu` prepare host, `STAGE_ELEMS`) + one transfer chunk. Export: HOST = the owned record arrays + one chunk (device 0: slices).

Constraint (documented): the preferential maps are the PRESCRIBED Phase 1/2 inputs supplied by the user as an explicit host `PreferentialSpec`; a live LEAF-GPU raw-map path is a Phase 2.5 hook and would be a separate input
source (no mirrored live plant water is created here).
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np

from maple_syrup import floating_held_context as fh
from maple_syrup import floating_matrix_infiltration as fmi
from maple_syrup import floating_pond as fp
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import native_floating_soil as nfs
from maple_syrup import subsurface as ss

__all__ = ["export_held_context_cuda", "kernel_descriptor", "plan", "prepare_parts"]

_ERRORS = (fsl.FloatingSoilError, ss.SubsurfaceError, nfs.NativeFloatingError, ValueError)
FloatingHeldError = fh.HeldContextError


def kernel_descriptor() -> dict:
    """STABLE source/ABI descriptor recorded beside (never inside) the physical content ids: the common CUDA source hash and the SHA-256 of the rendered kernel SOURCE text of the floating resident, matrix and
    preferential kernels. Pure host computation (no CuPy, no compile, no device query)."""
    from maple_syrup import floating_matrix_infiltration_cuda as fmc
    from maple_syrup import floating_preferential_infiltration_cuda as fpc
    from maple_syrup import preferential_infiltration_cuda as pic
    from maple_syrup import subsurface_cuda_floating as scf

    return {"common_abi_sha256": pic.COMMON_SOURCE_SHA256, "fsl_resident_source_sha256": hashlib.sha256(scf.render_resident_source().encode()).hexdigest(),
            "fmc_source_sha256": hashlib.sha256(fmc.render_source().encode()).hexdigest(), "fpc_source_sha256": hashlib.sha256(fpc.render_source().encode()).hexdigest()}


def _quota(available_bytes_fn: Any, reserve_bytes: Any, host_available_bytes_fn: Any, host_reserve_bytes: Any, memory_budget_bytes: Any = None, host_memory_budget_bytes: Any = None) -> dict:
    for name, fn in (("available_bytes_fn", available_bytes_fn), ("host_available_bytes_fn", host_available_bytes_fn)):
        if fn is not None and not callable(fn):
            raise FloatingHeldError(f"{name} must be callable or None (the real default reader)")
    try:
        return {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": fsl._count(reserve_bytes, "reserve_bytes"), "available_bytes_fn": available_bytes_fn,
                "host_memory_budget_bytes": host_memory_budget_bytes, "host_reserve_bytes": fsl._count(host_reserve_bytes, "host_reserve_bytes"), "host_available_bytes_fn": host_available_bytes_fn}
    except fsl.FloatingSoilError as exc:
        raise FloatingHeldError(str(exc)) from None


def _component(q: dict) -> dict:
    """The SAME readers/budgets with ZERO reserves: the composed admission already counted each pool's reserve once."""
    return {**q, "reserve_bytes": 0, "host_reserve_bytes": 0}


def _gate(column: Any = None) -> tuple[Any, int]:
    try:
        cp = fcu._rc().require_cuda()
        return cp, fcu.gate_device(cp, None if column is None else column.device_id)
    except (fsl.FloatingSoilError, ss.SubsurfaceError, RuntimeError) as exc:
        raise FloatingHeldError(f"the CUDA device gate refused: {exc}") from None


def _admit(cp: Any, dev: int, device_bytes: int, host_bytes: int, q: dict, what: str) -> None:
    try:
        fcu.admit(cp, dev, int(device_bytes), int(host_bytes), q, what)
    except (fsl.FloatingSoilError, ss.SubsurfaceError, MemoryError) as exc:
        raise FloatingHeldError(f"{what}: {exc}") from None


def _is_device(a: Any) -> bool:
    """ACTUAL namespace classification through MAPLE's backend (`is_device_array`: a CuPy array); a host array is EXACTLY `numpy.ndarray`. Anything else (list, subclass, foreign object) is refused: no duck attribute
    (NumPy 2.x arrays also have `.device == "cpu"`) and no fallback."""
    from maple.core.backend import is_device_array

    if is_device_array(a):
        return True
    if type(a) is np.ndarray:
        return False
    raise FloatingHeldError(f"an exported array must be a CuPy device array or exactly a numpy.ndarray, got {type(a).__module__}.{type(a).__name__}")


def _chunk_elems(itemsize: int) -> int:
    return max(1, fcu.HASH_CHUNK_BYTES // int(itemsize))


# ---------------------------------------------------------------------------------------------------------------------------------------------
# the composed two-pool model
# ---------------------------------------------------------------------------------------------------------------------------------------------
def plan(meta: dict) -> tuple[int, int, int]:
    """`(device_bytes, host_bytes, export_host_bytes)` of the reconstruction / export, from a VALIDATED /2 metadata dict by arithmetic only (module docstring). The reserves are NOT included (`fcu.admit` adds each once)."""
    from maple_syrup import floating_matrix_infiltration_cuda as fmc
    from maple_syrup import floating_preferential_infiltration_cuda as fpc

    L, N, K = meta["n_layers"], meta["n_cells"], meta["n_classes"]
    f8 = 8
    lay = meta["layouts"]
    kinds = {"scalar": 0, "shared": L * f8, "cell": L * N * f8}
    ks = meta["mode"] == "texture_ks"
    p = meta["pref"]
    S = 0 if p is None else p["n_species"]
    nov = len(meta["overrides"] or ())
    uploads = kinds[lay["theta_field_capacity"]] + kinds[lay["theta_saturation"]] + N * f8 + N + max(L - 1, 0) * f8 + (N * f8 if ks else 0)
    if p is not None:
        uploads += (N * f8 if p["a3_kind"] == "cell" else 0) + (N * f8 if p["max_rate_kind"] == "cell" else 0) + S * N * f8
    retained = fcu.retained_terms(L, N, dict(lay["all"]))["total"]
    chk = fcu.operation_terms("prepare", L, N)
    mat = fmc.operation_terms("prepare", L, N)
    mat_dev = mat["outputs"] + mat["scratch"]
    pref_dev = 0
    pref_host = 0
    if p is not None:
        pt = fpc.operation_terms("prepare", L, N)
        pref_dev = pt["outputs"] + pt["scratch"] + (L * N + 6 * N + 4 * N) * f8 + (L * N + 2 * N) * f8  # derivation live set + the retained derived maps (declared conservative, as the CPU model)
        pref_host = pt["host"] + (S * L * f8 * 2 + L * f8) + (2 * S + 2) * fh.SPEC_ARRAY_OVERHEAD_BYTES
    ks_keep = N * f8 if ks else 0  # the uploaded Ks vector the context keeps
    max_up = max([kinds[lay["theta_field_capacity"]], kinds[lay["theta_saturation"]], N * f8, N * f8 if ks else 0, (S and N * f8) or 0])
    chunk = min(fcu.HASH_CHUNK_BYTES, max(max_up, 8))
    device = uploads + retained + chk["scratch"] + mat_dev + pref_dev + ks_keep + chunk
    cfg_copies = (L + K * 3 + L + 2 * L + max(L - 1, 0) + nov * L * N) * f8 + 2 * (L + 1) * f8
    host = cfg_copies + pref_host + chk["host"] + mat["host"] + chunk
    record_bytes = fh._analytic(meta)[1]
    export_host = record_bytes + (S * L * f8 if S else 0) + chunk + (2 * S + 2) * fh.SPEC_ARRAY_OVERHEAD_BYTES
    return int(device), int(host), int(export_host)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# export (device vertical -> sealed host record /2)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _d2h(cp: Any, dev_arr: Any) -> np.ndarray:
    """COLD counted device-to-host copy into a fresh owned host array in `HASH_CHUNK_BYTES` chunks (never an unbounded `np.asarray(device)`)."""
    from maple.core.backend import to_host

    out = np.empty(tuple(int(x) for x in dev_arr.shape), dtype=dev_arr.dtype)
    flat, dst = dev_arr.reshape(-1), out.reshape(-1)
    per = _chunk_elems(out.dtype.itemsize)
    for i in range(0, int(flat.size), per):
        dst[i:i + per] = np.asarray(to_host(flat[i:i + per]))
    return out


def export_held_context_cuda(vertical: Any, pref_spec: Any = None, *, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0, verify: bool = True,
                             host_available_bytes_fn: Callable[[], int] | None = None, host_reserve_bytes: Any = 0) -> Any:
    """Export the held law context of a genuine CUDA `FloatingPondVertical` (never W/J/I) as a minted `/2` record. Gate first; metadata/references; ONE two-pool admission (host = the owned record arrays + one chunk,
    device = 0: slices only); cold chunked counted D2H of the device maps; plain host parts adopted by `floating_held_context.adopt_parts`; with `verify` a full device reconstruction proves the identities."""
    q = _quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes)
    if type(vertical) is not fp.FloatingPondVertical or vertical.backend != "cuda":
        raise FloatingHeldError("vertical must be a genuine CUDA FloatingPondVertical")
    t = vertical.held_template()
    cp, dev = _gate(t.column)  # the CURRENT device FIRST, before any reader/array/copy
    meta, refs = fh._snapshot(vertical, pref_spec)  # metadata and REFERENCES only
    meta["schema"] = fh.SCHEMA2
    meta["kernel"] = kernel_descriptor()
    fh._check_meta(meta, len(refs) + (1 if pref_spec is not None and pref_spec.species_receiver_weights else 0))  # strict self-check before any allocation
    _d, _h, export_host = plan(meta)
    _admit(cp, dev, 0, export_host, q, "exporting the CUDA held law record (owned host arrays and one transfer chunk)")
    out: dict = {}
    for name, a in refs.items():
        out[name] = _d2h(cp, a) if _is_device(a) else np.array(a, copy=True, order="C")
    if pref_spec is not None and pref_spec.species_receiver_weights:
        out["pref_species_weights"] = np.stack(pref_spec.species_receiver_weights)
    try:
        record = fh.adopt_parts(meta, out)
    except _ERRORS as exc:
        raise FloatingHeldError(f"the exported parts were refused: {exc}") from None
    if verify:
        ident = meta["identities"]
        fh.reconstruct_held_context(record, fh.ExpectedHeldIdentities(ident["config_sha256"], ident["root_column_sha256"], ident["matrix_op_sha256"], ident["pref_sha256"]),
                                    available_bytes_fn=available_bytes_fn, reserve_bytes=q["reserve_bytes"], host_available_bytes_fn=host_available_bytes_fn, host_reserve_bytes=q["host_reserve_bytes"])
    return record


# ---------------------------------------------------------------------------------------------------------------------------------------------
# reconstruction (sealed host record /2 -> device objects through the public factories)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _h2d(cp: Any, host: np.ndarray) -> Any:
    """COLD counted host-to-device copy into a preallocated device array in `HASH_CHUNK_BYTES` chunks (the per-chunk device temporary is inside the plan's chunk term)."""
    from maple.core.backend import to_device

    out = cp.empty(tuple(int(x) for x in host.shape), dtype=host.dtype)
    flat, src = out.reshape(-1), host.reshape(-1)
    per = _chunk_elems(host.dtype.itemsize)
    for i in range(0, int(src.size), per):
        flat[i:i + per] = to_device(np.ascontiguousarray(src[i:i + per]), cp)
    return out


def prepare_parts(record: Any, *, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None,
                  host_reserve_bytes: Any = 0) -> dict:
    """Re-prepare the CUDA objects from an intact `/2` record (called by `floating_held_context.reconstruct_held_context`, which then checks and mints). Returns plain parts; mutates nothing on failure (the device
    uploads are local and released)."""
    q = _quota(available_bytes_fn, reserve_bytes, host_available_bytes_fn, host_reserve_bytes)
    cp, dev = _gate()  # FIRST
    meta, arr = record._meta, record._arrays
    dev_need, host_need, _e = plan(meta)
    _admit(cp, dev, dev_need, host_need, q, "reconstructing the CUDA held law context (uploads kept alive while the public preparations copy them)")
    comp = _component(q)
    ident, sc = meta["identities"], {k: (None if v is None else fh._unhex(v, k)) for k, v in meta["scalars"].items()}
    overrides = None if meta["overrides"] is None else {k: arr[fh._OVERRIDE_PREFIX + k] for k in meta["overrides"]}
    L = meta["n_layers"]
    lay = meta["layouts"]
    try:
        cfg = nfs.native_floating_config(layer_thickness_m=arr["thickness_m"], grain_texture=arr["grain_texture"], organic_matter_percent=arr["organic_matter_percent"], bulk_density_kg_m3=sc["bulk_density_kg_m3"],
                                         latent_theta_field_capacity=arr["latent_theta_field_capacity"], latent_theta_saturation=arr["latent_theta_saturation"], theta_residual=sc["theta_residual"],
                                         theta_evaporation_stress=sc["theta_evaporation_stress"], evaporation_factor=sc["evaporation_factor"], diffusivity_m2_s=arr["diffusivity_m2_s"],
                                         bottom_drainage=meta["bottom_drainage"], matrix_conductivity_m_s=sc["matrix_conductivity_m_s"], matrix_suction_m=sc["matrix_suction_m"],
                                         matrix_vegetation_factor=sc["matrix_vegetation_factor"], initial_wetting_m=sc["initial_wetting_m"], initial_cumulative_intake_m=sc["initial_cumulative_intake_m"],
                                         overrides=overrides, backend="cuda", matrix_conductivity_mode=meta["mode"], latent_saturated_conductivity_m_s=sc["latent_saturated_conductivity_m_s"])
        if cfg.sha256 != ident["config_sha256"]:
            raise FloatingHeldError("the reconstructed native config does not reproduce the recorded config sha256")

        def raw(name: str, kind: str) -> Any:
            return float(arr[name][0]) if kind == "scalar" else _h2d(cp, arr[name])  # a scalar is a PYTHON float (a 0-d array would be promoted to (1,))

        fc, sat = raw("theta_field_capacity", lay["theta_field_capacity"]), raw("theta_saturation", lay["theta_saturation"])
        diff = _h2d(cp, arr["diffusivity_m2_s"]) if L > 1 else 0.0
        d0, mask = _h2d(cp, arr["depth0_m"]), _h2d(cp, arr["active"])
        col = fsl.prepare_floating_column(cfg.nominal_edges_m, theta_residual=cfg.theta_residual, theta_field_capacity=fc, theta_saturation=sat, diffusivity_m2_s=diff,
                                          theta_evaporation_stress=cfg.theta_evaporation_stress, evaporation_factor=cfg.evaporation_factor, diffusion_mode="constant", bottom_drainage=cfg.bottom_drainage,
                                          soil_depth_m=d0, active_mask=mask, cell_area_m2=sc["cell_area_m2"], backend="cuda", **comp)
        del fc, sat, diff, d0, mask
        if col.sha256 != ident["root_column_sha256"]:
            raise FloatingHeldError("the re-prepared CUDA root column does not reproduce the recorded root column sha256")
        ks = _h2d(cp, arr["ks_m_s"]) if meta["mode"] == "texture_ks" else None
        op = fmi.prepare_floating_matrix(col, conductivity_m_s=ks if ks is not None else cfg.matrix_conductivity_m_s, suction_m=cfg.matrix_suction_m, vegetation_factor=cfg.matrix_vegetation_factor, **comp)
        if op.sha256 != ident["matrix_op_sha256"]:
            raise FloatingHeldError("the re-prepared CUDA matrix operator does not reproduce the recorded sha256")
        pref = None
        p = meta["pref"]
        if p is not None:
            S = p["n_species"]
            sw = tuple(arr["pref_species_weights"][i] for i in range(S)) if S else ()
            contrib = tuple(_h2d(cp, arr[f"{fh._CONTRIB_PREFIX}{i}"]) for i in range(S)) if S else None
            a3 = _h2d(cp, arr["pref_a3"]) if p["a3_kind"] == "cell" else fh._unhex(p["a3_value"], "a3")
            mr = _h2d(cp, arr["pref_max_rate"]) if p["max_rate_kind"] == "cell" else fh._unhex(p["max_rate_value"], "max_rate", allow_inf=True)
            pref = fpi.prepare_floating_preferential(col, a3=a3, max_fraction_per_hour=fh._unhex(p["max_fraction_per_hour"], "max_fraction"), max_intake_rate_m_s=mr,
                                                     background_receiver_weights=arr["pref_background"] if p["has_background"] else None, species_receiver_weights=sw,
                                                     species_contributions=contrib, **comp)
            del a3, mr, contrib
            if pref.sha256 != ident["pref_sha256"]:
                raise FloatingHeldError("the re-prepared CUDA preferential operator does not reproduce the recorded sha256")
    except FloatingHeldError:
        raise
    except (*_ERRORS, MemoryError) as exc:
        raise FloatingHeldError(f"a public CUDA factory refused the record: {exc}") from None
    return {"config": cfg, "column": col, "matrix_op": op, "preferential": pref, "depth0": col._maps["soil_depth"], "ks": ks, "active": col._maps["active"], "edges": cfg.nominal_edges_m,
            "host_depth0": arr["depth0_m"], "host_active": arr["active"], "host_ks": arr["ks_m_s"] if ks is not None else None, "area": sc["cell_area_m2"]}
