"""CC2 NO-ORIGINAL-BED static recipe of the exact deep-pair lake (device-free module; CuPy only inside `factory` for the CUDA backend). The LOAD side builds the static `PondEvent` from (a) a bounded, closed-schema RECIPE record written
by the SAVE side, (b) the verified H0 context and (c) the carried routing origin read from the checkpoint file -- with NO original bed, owner, probe, `device_event`, `fb.make`/`make_pair` or texture provider anywhere.

The recipe holds ONLY static case inputs (fixture-only; JSON; closed key set; no pickle/eval): grid, spacing, friction, the ORIGINAL padded datum terrain `original_source_full` (an O(N padded) static input -- NOT a bed; it is recorded because
the original committed elevation is a roundoff-exact result of the original bed construction and is never inferred from the current bed), forcing, the declared completion policy, control and storage, plus the SourceBinding digests of the
original event (`context`, `sediment`, `schedule`, `options`, `config`, `forcing`, original topology identity/source/topology). The BedContext is built from the PUBLIC MAPLE constructors (GeometrySpec, GrainClassSet, AxisBoundary,
TopographicCommitSpec, AvalanchingSpec) exactly like the accepted H2 `static_case.py` (parametrised by the grid); the soil law comes from the verified H0 record, never from the Saxton-Rawls provider. SAVE refuses (`verify_event`) when the
event built by this recipe does not reproduce the original event's bindings."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import static_case as sc

RECIPE_SCHEMA = "cc2-static-lake-recipe/1"
KEYS = frozenset({"schema", "ny", "nx", "dx_m", "friction", "closed", "original_source_full", "rain_mm_h", "rain_end_s", "policy", "control", "bed_storage", "digests", "original_topology"})
CONTROL_KEYS = frozenset({"max_dt_s", "min_dt_s", "max_retries", "max_steps", "deep_publication"})
DIGEST_KEYS = frozenset({"context", "sediment", "schedule", "options", "config", "forcing", "splash"})
TOPO_KEYS = frozenset({"identity_sha256", "source_sha256", "topology_sha256", "generation"})
MAX_CELLS = 1 << 20


class RecipeError(RuntimeError):
    pass


def _fail(msg: str) -> None:
    raise RecipeError(msg)


def _f(v: Any, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        _fail(f"{name} must be a finite number")
    return float(v)


def _hex(v: Any, name: str) -> str:
    if not (isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)):
        _fail(f"{name} must be a 64-hex digest")
    return v


def bed_context(ny: int, nx: int):
    """The MAPLE bed CONTEXT of the deep pair from the actual public constructors (as `tests/phase2_floating_context_checkpoint/static_case.bed_context`, with the grid as a parameter): no probe state, no voxel/active arrays."""
    from maple.core.boundaries import AxisBoundary, BoundaryKind
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.geometry import GeometrySpec, validate_geometry
    from maple.core.parameters.grain_classes import (
        GrainClass,
        GrainClassSet,
        validate_grain_classes,
    )
    from maple.core.parameters.numerics import DEFAULT_MASS_RESOLUTION_KG
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.parameters.water_coupling import MAHLERAN_1_2_1_CLASS_DIAMETERS_M

    from maple_syrup.sediment_bed import BedContext

    gc = GrainClassSet(classes=tuple(GrainClass(class_id=f"mahleran_{i + 1}", diameter_m=float(d), particle_density_kg_m3=2650.0, is_aggregate=False) for i, d in enumerate(MAHLERAN_1_2_1_CLASS_DIAMETERS_M)))
    validate_grain_classes(gc)

    def boundary():
        return AxisBoundary(kind=BoundaryKind.PRESCRIBED, inflow_flux_kg_m_s={c.class_id: 0.0 for c in gc.classes})

    g = GeometrySpec(nx=nx, ny=ny, dx_m=sc.DX, dy_m=sc.DX, boundary_x=boundary(), boundary_y=boundary(), voxel_dz_m=sc.DZ, bulk_density_kg_m3=1250.0, active_layer_thickness_m=0.002)
    validate_geometry(g, gc.ids())
    return BedContext(g, gc, float(DEFAULT_MASS_RESOLUTION_KG), TopographicCommitSpec(), AvalanchingSpec(enabled=False), np.ones((ny, nx, sc.NC)))


def make_recipe(event: Any, original_topology: Any) -> dict:
    """The recipe of the ORIGINAL event (SAVE side): static inputs only, plus the SourceBinding digests of that event. The original topology is the caller-attested generation-0 one."""
    from maple_syrup import pond_checkpoint as pc

    ny, nx = (int(n) for n in event.shape)
    sched = event.schedule
    edges, inten = np.asarray(sched.edges_s), np.asarray(sched.intensity_mm_per_h)
    if edges.shape != (2,) or inten.shape != (1,) or float(edges[0]) != 0.0:
        _fail("the recipe serves a single constant-rain piece starting at 0")
    t = event.control.time
    top = original_topology
    return {"schema": RECIPE_SCHEMA, "ny": ny, "nx": nx, "dx_m": float(top.dx_m), "friction": float(np.asarray(top.friction).flat[0]), "closed": bool(int(top.n_closed_components) > 0),
            "original_source_full": np.asarray(top.source_z_full, dtype=np.float64).tolist(), "rain_mm_h": float(inten[0]), "rain_end_s": float(edges[1]), "policy": dataclasses.asdict(event.policy),
            "control": {"max_dt_s": float(t.max_dt_s), "min_dt_s": float(t.min_dt_s), "max_retries": int(t.max_retries), "max_steps": int(t.max_steps), "deep_publication": str(event.control.deep_publication)},
            "bed_storage": str(event.control.bed_storage),
            "digests": {"context": pc._digest_of(event.context), "sediment": pc._digest_of(event.sediment), "schedule": pc._digest_of(event.schedule), "options": pc._digest_of(event.veg, event.rain_scale, event.control, event.policy),
                        "config": event.config_sha256, "forcing": event.forcing_sha256, "splash": None if event.splash is None else pc._digest_of(event.splash)},
            "original_topology": {"identity_sha256": top.identity_sha256, "source_sha256": top.source_sha256, "topology_sha256": top.topology_sha256, "generation": int(top.generation)}}


def write_recipe(path: Any, recipe: dict) -> None:
    Path(path).write_text(json.dumps(check_recipe(recipe), allow_nan=False, sort_keys=True))


def read_recipe(path: Any) -> dict:
    p = Path(path)
    if p.stat().st_size > 8 << 20:
        _fail("recipe file too large")

    def no_const(tok):
        raise ValueError(tok)

    return check_recipe(json.loads(p.read_text(), parse_constant=no_const))


def check_recipe(r: Any) -> dict:
    """Closed-schema validation of a recipe record (types, finiteness, shapes, digests) BEFORE it is used."""
    from maple_syrup.pond_event import PondCompletionPolicy

    if not isinstance(r, dict) or set(r) != KEYS or r["schema"] != RECIPE_SCHEMA:
        _fail("recipe keys/schema mismatch")
    for k in ("ny", "nx"):
        if type(r[k]) is not int or not 1 <= r[k] <= 4096:
            _fail(f"{k} must be a bounded positive int")
    if r["ny"] * r["nx"] > MAX_CELLS:
        _fail("recipe grid too large")
    _f(r["dx_m"], "dx_m"), _f(r["friction"], "friction"), _f(r["rain_mm_h"], "rain_mm_h"), _f(r["rain_end_s"], "rain_end_s")
    if type(r["closed"]) is not bool or r["bed_storage"] not in ("dense", "windowed"):
        _fail("closed/bed_storage malformed")
    d = r["original_source_full"]
    if not isinstance(d, list) or len(d) != r["ny"] + 2 or any(not isinstance(row, list) or len(row) != r["nx"] + 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row) for row in d):
        _fail("original_source_full is malformed")
    pol = r["policy"]
    if not isinstance(pol, dict) or set(pol) != {f.name for f in dataclasses.fields(PondCompletionPolicy)}:
        _fail("policy keys mismatch")
    if not isinstance(r["control"], dict) or set(r["control"]) != CONTROL_KEYS or r["control"]["deep_publication"] != "step":
        _fail("control record malformed")
    if not isinstance(r["digests"], dict) or set(r["digests"]) != DIGEST_KEYS:
        _fail("digest record malformed")
    for k, v in r["digests"].items():
        if v is not None:
            _hex(v, k)
    o = r["original_topology"]
    if not isinstance(o, dict) or set(o) != TOPO_KEYS or o["generation"] != 0:
        _fail("original_topology record malformed")
    for k in ("identity_sha256", "source_sha256", "topology_sha256"):
        _hex(o[k], k)
    return r


def original_topology(r: dict):
    """The generation-0 ORIGINAL topology rebuilt from the recorded datum terrain and static inputs; its three digests must equal the recorded ones (a different datum/static case is refused)."""
    from maple_syrup.pond_network import build_pond_topology

    ny, nx = r["ny"], r["nx"]
    full = np.asarray(r["original_source_full"], dtype=np.float64)
    outlets = [] if r["closed"] else None
    if outlets is None:
        _fail("the recipe serves the closed lake only")
    top = build_pond_topology(full, np.ones((ny, nx), dtype=bool), np.full((ny, nx), r["friction"]), r["dx_m"], outlets, allow_closed_components=True)
    o = r["original_topology"]
    if (top.identity_sha256, top.source_sha256, top.topology_sha256) != (o["identity_sha256"], o["source_sha256"], o["topology_sha256"]):
        _fail("the rebuilt original topology differs from the recorded SourceBinding digests")
    return top


def factory(recipe: dict, backend: str):
    """`make(held_context, carried_topology=None) -> PondEvent`: the vertical from `FloatingPondVertical.from_held_context(ctx)` (the verified H0 record) and the event from the recipe on the carried routing origin (the original topology
    when `None`). `backend` is "numpy" | "numba" | "cuda". Nothing here constructs a bed, owner or probe."""
    r = check_recipe(recipe)

    def make(ctx, topology=None, *, available_bytes_fn=None, host_available_bytes_fn=None):
        from dataclasses import replace

        from maple_syrup import floating_pond as fp
        from maple_syrup.experimental_storm import ExperimentalControl
        from maple_syrup.pond_event import (
            PondCompletionPolicy,
            PondEvent,
            PondEventControl,
        )
        from maple_syrup.rainfall import constant_rainfall
        from maple_syrup.sediment_physics import plot1_sediment_parameters

        bed_ctx = bed_context(r["ny"], r["nx"])
        kw = {"available_bytes_fn": available_bytes_fn if available_bytes_fn is not None else (lambda: 1 << 40)}
        if backend == "cuda":
            from maple.core.backend import cupy_module, to_device

            from maple_syrup import floating_soil_layers_cuda as fcu

            cp = cupy_module()
            xp = cp
            bed_ctx = replace(bed_ctx, initial_available_fraction=to_device(bed_ctx.initial_available_fraction, cp))
            if available_bytes_fn is None:
                kw["available_bytes_fn"] = fcu.device_reader(cp)
            if host_available_bytes_fn is not None:
                kw["host_available_bytes_fn"] = host_available_bytes_fn
        else:
            xp = np
        vert = fp.FloatingPondVertical.from_held_context(ctx, **kw)
        c = r["control"]
        ctl = PondEventControl(backend=backend, time=ExperimentalControl(max_dt_s=c["max_dt_s"], min_dt_s=c["min_dt_s"], max_retries=c["max_retries"], max_steps=c["max_steps"]), bed_storage=r["bed_storage"],
                               deep_publication=c["deep_publication"])
        top = original_topology(r) if topology is None else topology
        return PondEvent(bed_ctx, None, plot1_sediment_parameters(xp=xp), constant_rainfall(0.0, r["rain_end_s"], r["rain_mm_h"]), top, control=ctl, policy=PondCompletionPolicy(**r["policy"]), splash=None,
                         floating_vertical=vert)

    return make


def verify_event(recipe: dict, event: Any) -> None:
    """SAVE-side SourceBinding check: the event this recipe builds must reproduce the ORIGINAL event's binding digests (context, sediment, schedule, options, config, forcing); a mismatch is refused (no silent drift)."""
    from maple_syrup import pond_checkpoint as pc

    got = {"context": pc._digest_of(event.context), "sediment": pc._digest_of(event.sediment), "schedule": pc._digest_of(event.schedule), "options": pc._digest_of(event.veg, event.rain_scale, event.control, event.policy),
           "config": event.config_sha256, "forcing": event.forcing_sha256, "splash": None if event.splash is None else pc._digest_of(event.splash)}
    bad = sorted(k for k in got if got[k] != recipe["digests"][k])
    if bad:
        _fail(f"the static-recipe event does not reproduce the original event's bindings: {bad}")
