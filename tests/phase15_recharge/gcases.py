"""GPU helpers of the Phase 1.5 S3 daily CUDA controls (WRITTEN, NOT RUN: the author never launches or probes a GPU; Root guards GPU0 and runs them). Strictly separate from
the CPU fixtures: the CUDA NCP profile and the CPU NCP profile live in SEPARATE processes (the CPU profile refuses a CuPy import). Every positive control uses the GENUINE
chain `ncp.verify_source_profile(profile=CUDA_SOURCE_PROFILE_NAME)` -> `admit_package_case` -> `native_dense_cuda_profile.prepare_native_dense_cuda_case` ->
`layered_case.load_layered_cuda_config` -> `layered_case_cuda.prepare_layered_cuda_case` (no stubs, no fake classes, no relabeling). Configuration: SYRUP_RELEASE_GPU=1 (opt-in,
the `gpu` fixture) AND SYRUP_NCP_CUDA_TARGET (the materialized CUDA 9f bundle); a configured-but-unusable target FAILS setup (never skipped)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

import ncpcases as nc

MiB = 1 << 20
RES = dict(caller_host_reserve_bytes=MiB, caller_device_reserve_bytes=MiB, lifecycle_extra_reserve_bytes=MiB, native_reserve_bytes=4 * MiB)


class ConfiguredCudaBundleError(RuntimeError):
    """SYRUP_NCP_CUDA_TARGET unset/missing/wrong origin although the GPU lane was opted in: a setup failure, never a skip."""


def require_cuda_bundle() -> Path:
    target = os.environ.get("SYRUP_NCP_CUDA_TARGET")
    if not target:
        raise ConfiguredCudaBundleError("SYRUP_NCP_CUDA_TARGET is unset although SYRUP_RELEASE_GPU=1 opted the GPU lane in")
    path = Path(target)
    if not path.is_dir():
        raise ConfiguredCudaBundleError(f"SYRUP_NCP_CUDA_TARGET {target!r} is missing or not a directory")
    import maple

    if path.resolve() not in Path(maple.__file__).resolve().parents:
        raise ConfiguredCudaBundleError(f"this process imported MAPLE from {maple.__file__}, not from the configured CUDA bundle {target}")
    return path.resolve()


def cuda_yaml(n_layers: int, *, bottom: str = "open", theta=0.25, root_table: bool = False, gravity: bool = True) -> dict:
    raw = nc.yaml_for(n_layers, bottom=bottom, theta=theta)
    raw["compute"]["backend"] = "cuda"
    if root_table:  # Root's known-answer profile: 100 mm layers, FC 20 mm, SAT 40 mm, residual 0
        soil = raw["soil"]
        soil["layers"]["thickness_m"] = [0.1] * n_layers
        soil["theta_residual"], soil["theta_field_capacity"], soil["theta_saturation"] = [0.0] * n_layers, [0.2] * n_layers, [0.4] * n_layers
        soil["theta_evaporation_stress"] = 0.1
    raw["soil"]["gravity_drainage"] = gravity
    return raw


def build_cuda_case(tmp: Path, raw: dict, *, flat_closed: bool = False, nz: int | None = None, host_fn=None, device_fn=None) -> SimpleNamespace:
    """Genuine CUDA native9f setup from a YAML FILE in the owned `tmp` (public example inputs copied; `flat_closed` is the same explicitly labeled test-input
    selector as the CPU lane: relief_m = 0 on the owned case copy and domain.outlets = [] in the owned YAML, before the real admission). `nz` is the explicit owned
    GEOMETRY selector (`topography.nz` of the owned case copy, before the real admission): the 100 mm-layer known-answer fixtures need `nz = nc.NZ_DEEP` (8) because the
    genuine native texture factory refuses a profile deeper than the represented column (the public example's nz = 4 represents ~0.19 m). Nothing is truncated or relabeled."""
    import cupy as cp
    from maple_syrup import layered_case as lc
    from maple_syrup import layered_case_cuda as lcc
    from maple_syrup import native_case_profile as ncp
    from maple_syrup import native_dense_cuda_profile as ndc

    target = require_cuda_bundle()
    profile = ncp.verify_source_profile(target, profile=ncp.CUDA_SOURCE_PROFILE_NAME)
    case_dir = tmp / "case"
    if not case_dir.exists():
        shutil.copytree(nc.EXAMPLE / "case", case_dir)
        if flat_closed or nz is not None:
            case = yaml.safe_load((case_dir / "case.yaml").read_text())
            if flat_closed:
                case["topography"]["perturbation"]["relief_m"] = 0.0
            if nz is not None:
                case["topography"]["nz"] = int(nz)
            (case_dir / "case.yaml").write_text(yaml.safe_dump(case))
    if flat_closed:
        raw = {**raw, "domain": {**raw["domain"], "outlets": []}}
    admission = ncp.admit_package_case(profile, case_dir)
    prep = ndc.prepare_native_dense_cuda_case(admission, device_id=0, caller_host_reserve_bytes=MiB, caller_device_reserve_bytes=MiB)
    path = tmp / f"layered_cuda_{abs(hash(yaml.safe_dump(raw))) % 10**8}.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = lc.load_layered_cuda_config(path)
    setup = lcc.prepare_layered_cuda_case(prep, cfg, host_available_bytes_fn=host_fn, device_available_bytes_fn=device_fn, **RES)
    lop = setup.lifecycle_operator
    return SimpleNamespace(setup=setup, lop=lop, s0=setup.state, col=setup.column, n=lop.n_cells, L=lop.n_layers, shape=lop.shape, cp=cp, case_dir=case_dir)


def host(cp, a):
    return np.asarray(cp.asnumpy(a))


def pet(c, rate: float):
    return c.cp.full(c.n, float(rate), dtype=np.float64)  # strictly device float64 (no host upload inside the daily API)


def fingerprint(c, state):
    from maple_syrup import water_lifecycle as wl

    a = wl._arrays_of(state)
    arrays = {k: host(c.cp, v).tobytes() for k, v in {"h": a.surface_m, "soil": a.soil.water_m, "qx": a.qx, "qy": a.qy, "j": a.history.wetting_m,
                                                       "intake": a.history.cumulative_intake_m}.items()}
    return (state.digest, state.t_s, state.revision, state.daily.digest() if state.daily is not None else None, state.lifetime.as_dict(), arrays)


def at_clock(c, state, t_s: float):
    """A NEW closed boundary on the same lifecycle/bed/device soil+history at the physical clock `t_s` (the soil state clock is re-stamped with it; arrays are shared)."""
    from maple_syrup import subsurface as ss
    from maple_syrup import water_lifecycle as wl

    p = state.payload
    soil = ss.column_state(c.col, p.soil.water_m, float(t_s))
    return wl.start_lifecycle(c.lop, bed=p.bed, soil=soil, history=p.history, t_s=float(t_s))
