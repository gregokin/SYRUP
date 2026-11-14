"""Support of the S0 vegetation-binding suite (distinct module name: no collision with other suites' `builders` / `conftest`).

LEAF is the ACTUAL package imported by path from LEAF_CANDIDATE_ROOT (default: Root's immutable frozen H3 compact-helpers snapshot (final public models), P1 V3 + H1/H2/H3; never the live writer): real PlantState, real catalog, real P1
projection, real PlantSession. The frozen C4 (V3) is used for independent controls only; the consumer stays UNQUALIFIED until a compatible P1 is accepted."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[3]
CANDIDATE = Path(__file__).resolve().parents[2]
_FROZEN = TASK / "root_compact_helpers_h3_snapshot"
LEAF_ROOT = Path(os.environ.get("LEAF_CANDIDATE_ROOT", str(_FROZEN)))
if (LEAF_ROOT / "leaf" / "water_vegetation.py").is_file() and str(LEAF_ROOT) not in sys.path:
    sys.path.insert(0, str(LEAF_ROOT))

_p = CANDIDATE / "tests" / "phase2_floating_column" / "conftest.py"
_spec = importlib.util.spec_from_file_location("d1_floating_builders_s0", _p)
_d1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d1)
column = _d1.column
numba_ok = _d1.numba_ok

BIG = 1 << 30
EDGES = [0.0, 0.1, 0.3, 0.6]
A1, A2 = 0.013, 0.067  # the Tietjen reference pair, passed EXPLICITLY (never a default of the module)
INF = math.inf
KEY = "preferential_infiltration_receiver_weights"


def fractions(layers):
    return {1: {"grass": [1.0], "shrub": [1.0]}, 2: {"grass": [0.8, 0.2], "shrub": [0.25, 0.75]}, 3: {"grass": [0.8, 0.1, 0.1], "shrub": [0.25, 0.35, 0.4]}}[layers]


def species_doc(layers, grass, shrub):
    doc = json.loads((LEAF_ROOT / "examples" / "phase0" / "species.json").read_text())
    doc["root_layers"] = layers
    for sid, rec in doc["species"].items():
        rec["roots"]["layer_fractions"] = list(fractions(layers)[sid])
    doc["schema_version"] = 2
    for sid, w in (("grass", grass), ("shrub", shrub)):
        if w is not None:
            doc["species"][sid]["hydrology"] = {KEY: list(w)}
    return doc


def catalog(leaf, layers=3, grass=None, shrub=None):
    return leaf["traits"].SpeciesCatalog(species_doc(layers, grass, shrub))


def plants_state(leaf, cat, plants, *, nx=3, ny=3, dx=1.0):
    """plants: (id, species, x, y, radius_m, status). Real `leaf.state.PlantState` records (no fake classes)."""
    st = leaf["state"]
    dom = st.Domain(nx, ny, dx, origin_x_m=0.0, origin_y_m=0.0, periodic=(False, False))
    recs = [{"plant_id": pid, "species": sp, "x_m": x, "y_m": y, "establishment_ground_m": 0.0, "agb_dry_kg": float(cat.biomass_from_diameter(sp, 2.0 * r)),
             "seasonal_agb_dry_kg": 0.0, "age_days": 10.0, "status": status} for pid, sp, x, y, r, status in plants]
    return st.PlantState.from_records(cat, dom, recs)


def project(leaf, st, cat, mask=None, surface=None):
    surf = np.zeros(st.domain.shape) if surface is None else surface
    p = leaf["wv"].project_from_initial_state(st, cat, surf, active_mask=mask, max_edges=1 << 16, max_bytes=BIG)
    return p, surf


def leaf_pin(root=None):
    """The expected FULL LEAF package digest, computed INDEPENDENTLY here (sorted 'relpath NUL file sha256 LF' over every non-bytecode file of the imported package directory)."""
    import leaf as pkg
    base = Path(pkg.__file__).resolve().parent
    rows = []
    for path in sorted(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")):
        rows.append((path.relative_to(base).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
    rows.sort()
    top = hashlib.sha256()
    for rel, h in rows:
        top.update(rel.encode() + b"\0" + h.encode() + b"\n")
    return top.hexdigest()


def helper_pin(root=None):
    """The expected external helper pins, computed INDEPENDENTLY here from the helper file of the LEAF root (`benchmarks/spatial_mapping.py`)."""
    f = Path(LEAF_ROOT if root is None else root) / "benchmarks" / "spatial_mapping.py"
    data = f.read_bytes()
    return [{"module": "benchmarks.spatial_mapping", "sha256": "sha256:" + hashlib.sha256(data).hexdigest(), "bytes": len(data)}]


def full_source(p):
    """The COMPLETE expected source record of a projection (every mandatory key; the free-text note excluded)."""
    from maple_syrup import water_vegetation_binding as wvb
    src = p.identity_record()["source"]
    return {k: src[k] for k in wvb.SOURCE_KEYS[src["kind"]]}


def two_species(leaf, layers=3, *, grass_w, shrub_w, mask=None):
    """3x3 domain: a grass plant (r 0.4) centred in cell (0,0), a shrub (r 0.35) in cell (1,1) and a second grass plant (r 0.3) in cell (1,1)."""
    cat = catalog(leaf, layers, grass_w, shrub_w)
    st = plants_state(leaf, cat, [(1, "grass", 0.5, 0.5, 0.4, "live"), (2, "shrub", 1.5, 1.5, 0.35, "live"), (3, "grass", 1.5, 1.5, 0.3, "live")])
    p, surf = project(leaf, st, cat, mask)
    return cat, st, p, surf


def make_column(layers=3, n=9, active=None, backend="numpy"):
    return column(EDGES[:layers + 1], [EDGES[layers]] * n, tr=0.1, tf=0.3, ts=0.5, backend=backend, active=active, stress=0.2)


def load_leaf_fixture():
    """LEAF's own real-session fixture (tests/native_leaf_fixture.py of the imported LEAF root), loaded by path under a distinct module name."""
    path = LEAF_ROOT / "tests" / "native_leaf_fixture.py"
    spec = importlib.util.spec_from_file_location("leaf_native_fixture_s0", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
