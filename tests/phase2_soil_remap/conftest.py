"""Shared helpers of the Phase 2 S1 remap suite. AUTHOR-written known answers and invariants only: nothing here is an independent oracle (Root writes a separate exact-rational
oracle in its own test file). The helpers deliberately build inputs only; they never reimplement the remap."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from maple_syrup import soil_water_remap as sr

BIG = 1 << 30


def numba_ok() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.fixture(params=["numpy", "numba"])
def cpu_backend(request):
    if request.param == "numba" and not numba_ok():
        pytest.skip("Numba is not importable (no claim made for the compiled backend)")
    return request.param


def arrays(eo, en, W, up, lo):
    """Columns are the second axis: eo/en `(L+1, C)`, W `(L, C)`, up/lo `(C,)`; everything fresh float64 C-contiguous."""
    eo, en, W = (np.ascontiguousarray(np.asarray(a, dtype=np.float64)) for a in (eo, en, W))
    up, lo = (np.ascontiguousarray(np.asarray(a, dtype=np.float64).reshape(-1)) for a in (up, lo))
    return eo, en, W, up, lo


def run(eo, en, W, up, lo, backend="numpy", area=1.0, budget=BIG, **kw):
    eo, en, W, up, lo = arrays(eo, en, W, up, lo)
    return sr.remap_intervals(eo, en, W, upper_theta=up, lower_theta=lo, cell_area_m2=area, backend=backend, memory_budget_bytes=budget, **kw)


def digest(*arrays_):
    h = hashlib.sha256()
    for a in arrays_:
        h.update(np.ascontiguousarray(a).tobytes())
        h.update(str(a.shape).encode())
    return h.hexdigest()


def random_columns(rng, L, C, shift_scale=0.4, widen=True, theta_scale=1.0):
    """Valid random columns: positive random widths, a random shift/stretch of the new edges (which may be disjoint), water fractions in [0,1]."""
    w_old = rng.uniform(0.05, 1.0, size=(L, C))
    eo = np.vstack([np.zeros((1, C)), np.cumsum(w_old, axis=0)]) + rng.uniform(-1.0, 1.0, size=(1, C))
    w_new = rng.uniform(0.05, 1.0, size=(L, C)) if widen else w_old
    en = np.vstack([np.zeros((1, C)), np.cumsum(w_new, axis=0)]) + eo[:1] + rng.uniform(-shift_scale, shift_scale, size=(1, C)) * 2.0 * w_old.sum(axis=0)
    W = w_old * rng.uniform(0.0, theta_scale, size=(L, C))
    return eo, en, W, rng.uniform(0, 1, C), rng.uniform(0, 1, C)
