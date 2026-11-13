"""Builders of the S2-C texture-property suite (author-written). Root's independent Decimal80 controls are portable copies under ./oracles."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from maple_syrup import soil_texture_properties as tp  # noqa: E402

ORACLES = Path(__file__).resolve().parent / "oracles"
BIG = 1 << 33
RT, AT = 2e-12, 1e-14


def numba_ok() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


BACKENDS = ["numpy"] + (["numba"] if numba_ok() else [])


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def normal_controls():
    return json.loads((ORACLES / "root_sr2006_normal_density_oracles.json").read_text())["controls"]


def density_controls():
    return json.loads((ORACLES / "root_sr2006_density_oracles.json").read_text())["controls"]


def call(S=0.4, Si=0.4, C=0.2, OM=2.5, rho=1400.0, *, L=1, N=1, backend="numpy", **kw):
    """Provider call with explicit inputs (scalars by default) and a generous quota."""
    kw.setdefault("memory_budget_bytes", BIG)
    return tp.estimate_soil_texture_properties(sand_fraction=S, silt_fraction=Si, clay_fraction=C, organic_matter_percent=OM, bulk_density_kg_m3=rho, n_layers=L, n_cells=N, backend=backend, **kw)


def val(r, name="theta_saturation"):
    return float(r.array(name).reshape(-1)[0])


def close(a, b, rt=RT, at=AT):
    return abs(a - b) <= at + rt * abs(b)


def valid_random(rng, L, N):
    """Random inputs that pass every check: texture (S,Si,C) with C<=.5, OM 0.5-6, a density factor inside (0.93, 1.27) of the normal density of each element."""
    S = rng.uniform(0.15, 0.6, (L, N))
    C = rng.uniform(0.08, 0.4, (L, N))
    Si = 1.0 - S - C
    OM = rng.uniform(0.5, 6.0, (L, N))
    return S, Si, C, OM


def random_valid(rng, n, *, om_range=(0.5, 6.0)):
    """`n` elements (S, Si, C, OM, rho_kg) that the Decimal model accepts (density 1000-1800 kg/m3, DF in range, air entry in [0, 33), curve order, gap), by rejection sampling."""
    from decimal_model import model

    out = []
    while len(out) < n:
        S, C = rng.uniform(0.1, 0.7), rng.uniform(0.06, 0.5)
        if S + C > 0.97:
            continue
        Si = 1.0 - S - C
        OM = rng.uniform(*om_range)
        flags, vals = model(S, Si, C, OM, 1000.0)  # rho = 1 g/cm3 only to obtain the normal density: rho_N = 1/DF
        rho_n = 1.0 / float(vals["df"])
        rho = 1000.0 * rho_n * rng.uniform(0.93, 1.27)
        flags, vals = model(S, Si, C, OM, rho)
        if flags or not (1000.0 <= rho <= 1800.0):
            continue
        out.append((S, Si, C, OM, rho))
    return np.array(out).T
