"""NEW static recipe of the OPEN-PLANE vegetation fixture (correction01 C2). The old qualified `cc2_static_case` helper deliberately serves the CLOSED lake only and is left untouched; this helper records the primitive static inputs of the
actual bridge `dry_kwargs` plane (bottom open, a real positive-drop outlet on every east cell): grid, spacing, friction, the ORIGINAL padded datum terrain (a recorded static input, never inferred from the bed), the east-outlet rule, rain, policy,
control, storage and the SourceBinding digests of the ORIGINAL event (built WITH the bound splash cover). The generation-0 topology is rebuilt through the actual `build_pond_topology` API and ALL of its identity / source / topology digests are
compared; the static event rebuilt from the recipe must reproduce ALL of the original event's binding digests (`verify_open_event`), exactly like `cc2_static_case.verify_event`. No bed, owner, probe or provider is touched; the
BedContext comes from the PUBLIC MAPLE constructors through the old helper's `bed_context`. Device code is lazy (CuPy only inside the `cuda` branch)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "phase2_floating_continued_checkpoint_cuda"), str(Path(__file__).resolve().parents[1] / "phase2_floating_context_checkpoint")]
import cc2_static_case as sc2

SCHEMA = "veg-open-plane-recipe/1"
OUTLET_RULES = ("east",)


class OpenRecipeError(RuntimeError):
    pass


def _fail(msg: str) -> None:
    raise OpenRecipeError(msg)


def make_open_recipe(event: Any, original_topology: Any) -> dict:
    """Recipe of the ORIGINAL open-plane event (an event built through the lifecycle factory WITH its bound cover): the closed-lake base record of the old helper plus the explicit outlet rule."""
    if int(original_topology.n_closed_components) != 0:
        _fail("this helper serves the OPEN plane (no closed component); the closed lake keeps its own qualified helper")
    base = sc2.make_recipe(event, original_topology)
    return {"schema": SCHEMA, "outlets": "east", "base": sc2.check_recipe(base)}


def check_open_recipe(r: Any) -> dict:
    if not isinstance(r, dict) or set(r) != {"schema", "outlets", "base"} or r["schema"] != SCHEMA or r["outlets"] not in OUTLET_RULES:
        _fail("open-plane recipe keys / schema / outlet rule mismatch")
    base = sc2.check_recipe(r["base"])
    if base["closed"]:
        _fail("an open-plane recipe records a non-closed base")
    return r


def write_open_recipe(path: Any, recipe: dict) -> None:
    Path(path).write_text(json.dumps(check_open_recipe(recipe), allow_nan=False, sort_keys=True))


def read_open_recipe(path: Any) -> dict:
    p = Path(path)
    if p.stat().st_size > 8 << 20:
        _fail("recipe file too large")

    def no_const(tok):
        raise ValueError(tok)

    return check_open_recipe(json.loads(p.read_text(), parse_constant=no_const))


def original_topology(recipe: dict) -> Any:
    """The generation-0 open-plane topology rebuilt through the ACTUAL API from the recorded datum terrain and static inputs; its identity / source / topology digests must equal the recorded ones."""
    from maple_syrup.pond_network import build_pond_topology

    r = check_open_recipe(recipe)["base"]
    ny, nx = r["ny"], r["nx"]
    full = np.asarray(r["original_source_full"], dtype=np.float64)
    outlets = [(row, nx - 1, 0, 1) for row in range(ny)]  # the east rule: every east cell has a real positive-drop normal-flow outlet (the drop is inside the recorded padded datum)
    top = build_pond_topology(full, np.ones((ny, nx), dtype=bool), np.full((ny, nx), r["friction"]), r["dx_m"], outlets, allow_closed_components=False)
    o = r["original_topology"]
    if (top.identity_sha256, top.source_sha256, top.topology_sha256) != (o["identity_sha256"], o["source_sha256"], o["topology_sha256"]):
        _fail("the rebuilt original open-plane topology differs from the recorded SourceBinding digests")
    return top


def verify_open_event(recipe: dict, event: Any) -> None:
    """The static event rebuilt from the recipe must reproduce ALL of the original event's binding digests (context, sediment, schedule, options incl. the BOUND cover, config, forcing, splash)."""
    try:
        sc2.verify_event(check_open_recipe(recipe)["base"], event)
    except sc2.RecipeError as exc:
        _fail(str(exc))


def static_event(recipe: dict, vert: Any, topology: Any, cover: Any, backend: str, schedule: Any = None) -> Any:
    """`PondEvent` from the recipe only (no bed, owner, probe or provider), built with the BOUND cover it is handed; the original binding digests are verified for the recipe's own schedule."""
    from dataclasses import replace

    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.pond_event import PondCompletionPolicy, PondEvent, PondEventControl
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters

    r = check_open_recipe(recipe)["base"]
    ctx = sc2.bed_context(r["ny"], r["nx"])
    if backend == "cuda":
        from maple.core.backend import cupy_module, to_device

        cp = cupy_module()
        ctx = replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, cp))
        xp = cp
    else:
        xp = np
    c = r["control"]
    ctl = PondEventControl(backend=backend, time=ExperimentalControl(max_dt_s=c["max_dt_s"], min_dt_s=c["min_dt_s"], max_retries=c["max_retries"], max_steps=c["max_steps"]), bed_storage=r["bed_storage"],
                           deep_publication=c["deep_publication"])
    top = original_topology(recipe) if topology is None else topology
    ev = PondEvent(ctx, None, plot1_sediment_parameters(xp=xp), constant_rainfall(0.0, r["rain_end_s"], r["rain_mm_h"]) if schedule is None else schedule, top, control=ctl, policy=PondCompletionPolicy(**r["policy"]),
                   splash=None, vegetation_cover_fraction=cover, floating_vertical=vert)
    if schedule is None:
        verify_open_event(recipe, ev)
    return ev
