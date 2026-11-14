"""Builders of the Phase 2.5 vegetation-lifecycle suite: ACTUAL LEAF plants / catalog / P1 projection and the S0 binding on top of REAL native floating lifecycle boundaries. No fake classes, projectors or boundaries."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from maple_syrup import water_vegetation_binding as wvb

# The LEAF root is resolved HERE from the environment / the task directory (never imported from a conftest: with several conftest.py files in one collection the last one wins and a helper that imports from `conftest` breaks)
LEAF_ROOT = Path(os.environ.get("LEAF_CANDIDATE_ROOT", str(Path(__file__).resolve().parents[3] / "leaf_candidate")))
BIG = 1 << 30
A1, A2 = 0.013, 0.067  # the Tietjen reference pair, passed EXPLICITLY
FR3 = {"grass": [0.8, 0.1, 0.1], "shrub": [0.25, 0.35, 0.4]}
GRASS_W, SHRUB_W = [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]  # shrub receiving weights all zero: that species' pathway is disabled


def surface_of(boundary):
    """The AUTHORITATIVE current bed surface through the existing public MAPLE read (a fresh C-contiguous float64 HOST copy owned by the test; a device elevation goes through MAPLE's counted `to_host`)."""
    from maple.core.backend import to_host
    from maple.core.types.topographic_commit import committed_elevation_m

    return np.ascontiguousarray(np.array(to_host(committed_elevation_m(boundary.bed.committed_topography)), dtype=np.float64))


def catalog(leaf, layers=3, grass=GRASS_W, shrub=SHRUB_W):
    doc = json.loads((LEAF_ROOT / "examples" / "phase0" / "species.json").read_text())
    doc["root_layers"] = layers
    for sid, rec in doc["species"].items():
        rec["roots"]["layer_fractions"] = list(FR3[sid])
    doc["schema_version"] = 2
    for sid, w in (("grass", grass), ("shrub", shrub)):
        if w is not None:
            doc["species"][sid]["hydrology"] = {leaf["traits"].RECEIVER_WEIGHTS_KEY: list(w)}
    return leaf["traits"].SpeciesCatalog(doc)


def plant_layout(ny, nx, dx):
    """(id, species, x, y, radius, status): a grass plant alone in cell (0,0) (r .4 dx), a shrub (r .35 dx) and a second grass (r .3 dx) sharing cell (1,1); every circle lies inside its cell."""
    return [(1, "grass", 0.5 * dx, 0.5 * dx, 0.4 * dx, "live"), (2, "shrub", 1.5 * dx, 1.5 * dx, 0.35 * dx, "live"), (3, "grass", 1.5 * dx, 1.5 * dx, 0.3 * dx, "live")]


def plants_state(leaf, cat, surf, dx, plants=None):
    st = leaf["state"]
    ny, nx = surf.shape
    dom = st.Domain(nx, ny, dx, origin_x_m=0.0, origin_y_m=0.0, periodic=(False, False))
    recs = []
    for pid, sp, x, y, r, status in plants or plant_layout(ny, nx, dx):
        est = float(surf[min(int(y // dx), ny - 1), min(int(x // dx), nx - 1)])  # anchored at the CURRENT ground: no burial
        recs.append({"plant_id": pid, "species": sp, "x_m": x, "y_m": y, "establishment_ground_m": est, "agb_dry_kg": float(cat.biomass_from_diameter(sp, 2.0 * r)), "seasonal_agb_dry_kg": 0.0,
                     "age_days": 10.0, "status": status})
    return st.PlantState.from_records(cat, dom, recs)


def leaf_pin(root=LEAF_ROOT):
    """The expected FULL LEAF package digest computed INDEPENDENTLY (sorted 'relpath NUL file sha256 LF' over every non-bytecode file of the package directory)."""
    base = Path(root) / "leaf"
    rows = sorted((p.relative_to(base).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo"))
    top = hashlib.sha256()
    for rel, h in rows:
        top.update(rel.encode() + b"\0" + h.encode() + b"\n")
    return top.hexdigest()


def helper_pin(root=LEAF_ROOT):
    """The expected external helper pins computed INDEPENDENTLY from the helper file of the LEAF root (`benchmarks/spatial_mapping.py`)."""
    data = (Path(root) / "benchmarks" / "spatial_mapping.py").read_bytes()
    return [{"module": "benchmarks.spatial_mapping", "sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "bytes": len(data)}]


def controls(boundary, **kw):
    L, N = boundary.root.column.n_layers, boundary.root.column.n_cells
    base = {"a1": A1, "a2": A2, "a3": 0.4, "max_fraction_per_hour": 0.7, "max_intake_rate_m_s": 1e-5, "background_receiver_weights": [0.0, 1.0, 1.0], "n_layers": L, "n_cells": N, "max_bytes": BIG}
    base.update(kw)
    return wvb.site_controls(**base)


def stack(leaf, boundary, *, surface=None, mask=None, plants=None, cat=None, state=None):
    """Real LEAF catalog / PlantState / P1 projection on the boundary's actual grid and CURRENT bed surface. Returns a dict."""
    surf = surface_of(boundary) if surface is None else surface
    dx = float(boundary.routing[0].dx_m)
    cat = catalog(leaf, boundary.root.column.n_layers) if cat is None else cat
    st = plants_state(leaf, cat, surf, dx, plants) if state is None else state
    m = np.ones(surf.shape, dtype=bool) if mask is None else mask
    proj = leaf["wv"].project_from_initial_state(st, cat, surf, active_mask=m, max_edges=1 << 16, max_bytes=BIG)
    return {"cat": cat, "state": st, "surface": surf, "mask": m, "proj": proj, "dx": dx}


def binding(leaf, boundary, s, *, ctl=None, **kw):
    ctl = controls(boundary) if ctl is None else ctl
    p = s["proj"]
    ident = p.identity_record()
    args = {"expected_leaf_package_sha256": leaf_pin(), "expected_projection_sha256": ident["content_sha256"],
            "expected_source": {k: ident["source"][k] for k in ("kind", "state_sha256", "catalog_hash")}, "expected_external_helpers": helper_pin(), "active_mask": s["mask"], "current_surface": s["surface"], "max_bytes": BIG}
    args.update(kw)
    return wvb.bind_projection(p, ctl, **args)


def closed_form_r1(dx, ny, nx):
    """R1 = sum_s f_live_s * rootfrac_s[0] for `plant_layout` (independent closed form: the circles lie inside their cells, area pi r^2 / dx^2)."""
    r1 = np.zeros(ny * nx)
    f = lambda r: math.pi * (r / dx) ** 2
    r1[0] = f(0.4 * dx) * FR3["grass"][0]
    r1[nx + 1] = f(0.3 * dx) * FR3["grass"][0] + f(0.35 * dx) * FR3["shrub"][0]
    return r1


def closed_form_pref_grass(dx, ny, nx):
    """Preferential contribution of the (enabled) grass species: f_live * sum(rootfrac[1:]); the shrub receiving vector is all zero (disabled)."""
    c = np.zeros(ny * nx)
    lower = sum(FR3["grass"][1:])
    f = lambda r: math.pi * (r / dx) ** 2
    c[0] = f(0.4 * dx) * lower
    c[nx + 1] = f(0.3 * dx) * lower
    return c


def leaf_fixture(leaf):
    """The LEAF candidate's own real session fixtures (`tests/native_leaf_fixture.py`), loaded by PATH under a unique module name (no collision with this repository's `tests` package)."""
    import importlib.util

    path = LEAF_ROOT / "tests" / "native_leaf_fixture.py"
    if not path.is_file():
        import pytest

        pytest.skip("the LEAF candidate's native session fixture is not present (no claim)")
    spec = importlib.util.spec_from_file_location("veg_lc_leaf_fixture", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def session_catalog(leaf, nfx, layers=3):
    doc = json.loads(nfx.FIXTURES["phase1"].read_text())
    nfx.simple_edit(doc)
    doc["root_layers"] = layers
    doc["schema_version"] = 2
    for sid, rec in doc["species"].items():
        rec["roots"]["layer_fractions"] = [0.8, 0.1, 0.1] if sid == "grass" else [0.25, 0.35, 0.4]
        rec["hydrology"] = {leaf["traits"].RECEIVER_WEIGHTS_KEY: list(GRASS_W)}
    return leaf["traits"].SpeciesCatalog(doc)


def session_state(leaf, cat, surf, dx):
    st = leaf["state"]
    ny, nx = surf.shape
    dom = st.Domain(nx, ny, dx, origin_x_m=0.0, origin_y_m=0.0, periodic=(False, False))
    recs = [{"plant_id": pid, "species": "grass", "x_m": x, "y_m": y, "establishment_ground_m": float(surf[int(y // dx), int(x // dx)]), "agb_dry_kg": float(cat.biomass_from_diameter("grass", 2.0 * r)),
             "seasonal_agb_dry_kg": 0.0, "age_days": 100.0, "status": "live"} for pid, x, y, r in ((11, 0.5 * dx, 0.5 * dx, 0.4 * dx), (12, 1.5 * dx, 1.5 * dx, 0.3 * dx))]
    return st.PlantState.from_records(cat, dom, recs)


def session_binding(leaf, boundary, sess, surf, mask, ctl):
    p = leaf["wv"].project_from_plant_session(sess, surf, active_mask=mask, max_edges=1 << 16, max_bytes=BIG)
    ident = p.identity_record()
    src = {k: ident["source"][k] for k in ("kind", "token", "sequence", "pop_sha256", "geometry_sha256", "session_id", "lineage", "catalog_hash")}
    return wvb.bind_projection(p, ctl, expected_leaf_package_sha256=leaf_pin(), expected_projection_sha256=ident["content_sha256"], expected_source=src, expected_external_helpers=helper_pin(), active_mask=mask,
                               current_surface=surf, session=sess, max_bytes=BIG), p
