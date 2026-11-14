"""Helpers of the cached-inputs suite (distinct module name)."""
from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path

import numpy as np
import veg_binding_support
from veg_binding_support import (
    A1,
    A2,
    BIG,
    catalog,
    full_source,
    helper_pin,
    leaf_pin,
    make_column,
    plants_state,
    project,
    two_species,
)

from maple_syrup import water_vegetation_binding as wvb
from maple_syrup import water_vegetation_held as wh

K, PSI = 2e-6, 0.3
_STORE = "/home/okin/SYRUP/agent_handoffs/tasks/phase1_publication_c1_20261112/runs"
CPU_ROOT = Path(f"{_STORE}/runtime-store-cpu/objects/maple-materialized/5d850789f9e82dc456227cb1eb4dfcc255b3ee4dfb0ed74883813c93e98d0ab3")
CUDA_ROOT = Path(f"{_STORE}/runtime-store-cuda/objects/maple-materialized/53a3e569dff6fd9d62a73eec1f35627645b93401a4510d873d9f0bf24cfd4366")
QUALIFIED_MAPLE_ROOTS = (CPU_ROOT, CUDA_ROOT)  # the TWO exact qualified immutable materializations of the same MAPLE 9f package (fbc412...)
ROOT_ENV = "CACHED_INPUTS_EXPECTED_MAPLE_ROOT"
GPU_FLAG = "SYRUP_VEG_BINDING_RELEASE_GPU"
NATIVE = CPU_ROOT  # the CPU lane's explicit constant (kept for the CPU fresh children)
MAPLE_PACKAGE = "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"
LEAF_PACKAGE = "38009872f8969f86b56c720adc1b5d9ffefc554fb9cacf90855d999db604530f"  # the FINAL frozen H3 compact-helpers LEAF (28 files)


def configured_maple_root(environ=None) -> Path:
    """The EXPECTED MAPLE materialization of this run, from EXPLICIT configuration only. CPU lane: `ROOT_ENV` unset => the CPU constant. GPU lane (`GPU_FLAG` set): `ROOT_ENV` MUST be set
    (no default). In every case the value must equal one of the TWO exact qualified immutable roots (resolved; a live / mutable / prefix / sibling path, an empty value or a missing source file is
    a hard failure, never a fallthrough) and its `source/src/maple/__init__.py` must exist. The caller then asserts that the IMPORTED `maple` lies under it and that its FULL package digest is
    the qualified one before and after the tests."""
    env = os.environ if environ is None else environ
    raw = env.get(ROOT_ENV)
    if raw is None:
        if env.get(GPU_FLAG):
            raise ValueError(f"{GPU_FLAG} is set but {ROOT_ENV} is not: the GPU lane must configure its expected MAPLE materialization explicitly (no default)")
        root = CPU_ROOT
    else:
        if not raw.strip():
            raise ValueError(f"{ROOT_ENV} is empty")
        root = Path(raw)
    resolved = root.resolve()
    if resolved not in {q.resolve() for q in QUALIFIED_MAPLE_ROOTS}:
        raise ValueError(f"{root} is not one of the two qualified immutable MAPLE materializations {[str(q) for q in QUALIFIED_MAPLE_ROOTS]}")
    if not (resolved / "source" / "src" / "maple" / "__init__.py").is_file():
        raise ValueError(f"the configured MAPLE source does not exist under {resolved}")
    return resolved


def native_env(extra_path: str = "", *, gpu: bool = False) -> dict:
    """The environment of EVERY child process: the explicit pinned prefix bootstrap (no ambient fallback). CPU children (default) ALWAYS use the CPU materialization; a GPU child (`gpu=True`) uses the
    root the parent GPU lane configured explicitly (`configured_maple_root`)."""
    root = configured_maple_root() if gpu else CPU_ROOT
    cand = str(Path(__file__).resolve().parents[2])
    path = [str(root / "editable"), str(root / "source" / "src"), f"{cand}/src", str(Path(__file__).resolve().parents[1] / "phase25_vegetation_binding"), str(Path(__file__).resolve().parent)]
    if extra_path:
        path.append(extra_path)
    return {**os.environ, "PYTHONPATH": os.pathsep.join(path), "MAPLE_SYRUP_EXPECTED_MAPLE_ROOT": str(root / "source"), "E3_PINNED_MAPLE_SRC": str(root / "source" / "src"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "") if gpu else "", "NUMBA_NUM_THREADS": "1",
            "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR", "/tmp/cached_inputs_c03_numba"), "PYTHONDONTWRITEBYTECODE": "1", "LEAF_CANDIDATE_ROOT": str(veg_binding_support.LEAF_ROOT)}


ARR = ("splash_cover", "shading_cover", "matrix_root_index", "cell_offsets", "entry_cell", "entry_species", "entry_area_all_m2", "entry_area_live_m2", "root", "lower", "weights", "has_w",
       "omitted_ids")


def controls(L=3, n=9, **kw):
    base = {"a1": A1, "a2": A2, "a3": 0.4, "max_fraction_per_hour": 0.7, "max_intake_rate_m_s": 1e-5, "background_receiver_weights": [0.0, 1.0, 1.0] if L == 3 else [1.0] * L,
            "n_layers": L, "n_cells": n, "max_bytes": BIG}
    base.update(kw)
    return wvb.site_controls(**base)


def live(leaf, *, layers=3, grass=(0, 1, 1), shrub=(0, 1, 1), mask=None, **ckw):
    _cat, _st, p, surf = two_species(leaf, layers, grass_w=list(grass), shrub_w=list(shrub), mask=mask)
    m = np.ones((3, 3), dtype=bool) if mask is None else mask
    b = wvb.bind_projection(p, controls(L=layers, **ckw), expected_leaf_package_sha256=leaf_pin(), expected_projection_sha256=p.identity_record()["content_sha256"], expected_source=full_source(p),
                            expected_external_helpers=helper_pin(), active_mask=m, current_surface=surf, max_bytes=BIG)
    return p, b, surf, m


def expected(p, b, rec):
    return {"expected_record_sha256": rec.record_sha256, "expected_binding_sha256": b.sha256, "expected_projection_sha256": p.identity_record()["content_sha256"],
            "expected_leaf_package_sha256": leaf_pin(), "expected_external_helpers": helper_pin(), "expected_source": full_source(p)}


def loadable(rec):
    """A closed `{meta, arrays}` pair as a codec would read it: MUTABLE, writable copies (untrusted load arrays)."""
    e = wh.export_cached_record(rec)
    return copy.deepcopy(e["meta"]), {k: np.array(v) for k, v in e["arrays"].items()}


def fingerprint(meta, arrays):
    return json.dumps(meta, sort_keys=True), {k: (a.tobytes(), str(a.dtype), a.shape, bool(a.flags.writeable)) for k, a in arrays.items()}


def stage(rec, p, b, meta, arrays, mask, **kw):
    args = {**expected(p, b, rec), "active_mask": mask, "max_bytes": BIG}
    args.update(kw)
    return wh.stage_cached_inputs(meta, arrays, **args)


def prepare(col, b, **kw):
    args = {"conductivity_m_s": K, "suction_m": PSI, "memory_budget_bytes": BIG}
    args.update(kw)
    return wvb.prepare_vegetation_law(col, b, **args)


def collect():
    gc.collect()


__all__ = ["ARR", "CPU_ROOT", "CUDA_ROOT", "GPU_FLAG", "LEAF_PACKAGE", "MAPLE_PACKAGE", "NATIVE", "PSI", "QUALIFIED_MAPLE_ROOTS", "ROOT_ENV", "K", "catalog", "collect", "configured_maple_root", "controls", "expected", "fingerprint", "live", "loadable", "make_column", "native_env", "plants_state", "prepare", "project", "stage", "wh", "wvb"]
