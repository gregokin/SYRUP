"""Optional Numba-compiled ordered sweep for `route_step(..., implementation="numba")`.

Same discrete equations, same proven `[0, R]` bracket, same bisection
multiply sequence and same donor summation order as the array sweep in
`routing.py`. Only the Python-level loops over dependency levels and
bisection iterations move into ONE nopython call that walks every active
cell in level order. Nothing else changes: the graph preprocessing, input
validation, Courant/right-hand-side rejection, storage identity, face
volumes, budgets and `RouteStep` are shared with the array path.

Constraints, by user direction:

- no `fastmath` (floating-point behaviour must match the array path);
- no `prange` (downstream cells depend on upstream results);
- CPU NumPy only, no host/device transfer, no silent fallback -- a missing
  Numba raises `NumbaUnavailableError` (a `RoutingError`).

Optional dependency: `pip install "maple-syrup[numba]"`. Codex tests with
Numba 0.67.0 / llvmlite 0.49.0 on NumPy 2.5.2 / Python 3.12.3 in an
isolated tree; nothing here was run by its author. On-disk caching is off
unless `MAPLE_SYRUP_NUMBA_CACHE=1`, in which case `NUMBA_CACHE_DIR` must
point outside every reference tree (Numba otherwise writes beside this
file). Compilation happens on the first call ("cold"); later calls with the
same array types reuse it ("warm"). Numba CPU compilation is not GPU
acceleration and no speed is claimed here.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from maple_syrup.routing import RoutingError

__all__ = [
    "NumbaUnavailableError",
    "compiled_sweep",
    "numba_available",
    "numba_versions",
    "reset_compiled",
    "run_sweep",
]

_SWEEP: Any = None


class NumbaUnavailableError(RoutingError):
    """`implementation="numba"` was requested but Numba cannot be imported.
    There is no fallback to the array path."""


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def numba_versions() -> dict[str, str] | None:
    """Installed Numba/llvmlite versions for provenance, or None."""
    try:
        import llvmlite
        import numba
    except ImportError:
        return None
    return {"numba": numba.__version__, "llvmlite": llvmlite.__version__}


def _sweep(bounds, k_lo, donor_position, donor_mask, base_lo, c, iterations,
           qin_new_lo, q_new_lo, flow_lo, rhs_lo):
    """Level-ordered method-5 sweep. Mirrors routing._route/_bisect exactly:
    qin = ((0 + d0) + d1) + d2) + d3 with non-donors adding 0.0; rhs = base +
    qin*c; bisection on [0, rhs] keeping lo where mid + ((sqrt(mid)*mid)*k)*c
    < rhs; q = (sqrt(lo)*lo)*k. np.sqrt of a negative rhs yields NaN (never
    an exception), leaving lo = 0 for the shared R < 0 check."""
    for lev in range(bounds.shape[0] - 1):
        for p in range(bounds[lev], bounds[lev + 1]):
            qin = 0.0
            for s in range(4):
                if donor_mask[s, p]:
                    qin = qin + q_new_lo[donor_position[s, p]]
                else:
                    qin = qin + 0.0
            rhs = base_lo[p] + qin * c
            k = k_lo[p]
            lo = 0.0
            w = rhs
            for _ in range(iterations):
                w = w * 0.5
                mid = lo + w
                t = np.sqrt(mid)
                t = t * mid
                t = t * k
                t = t * c
                t = t + mid
                if t < rhs:
                    lo = mid
            q = np.sqrt(lo)
            q = q * lo
            q = q * k
            qin_new_lo[p] = qin
            q_new_lo[p] = q
            flow_lo[p] = lo
            rhs_lo[p] = rhs


def compiled_sweep():
    """The nopython-compiled `_sweep` (compiled lazily, once per process)."""
    global _SWEEP
    if _SWEEP is None:
        try:
            import numba
        except ImportError as exc:
            raise NumbaUnavailableError(
                "implementation 'numba' requested but Numba is not installed (optional extra "
                "maple-syrup[numba]); there is no fallback to the array implementation"
            ) from exc
        cache = os.environ.get("MAPLE_SYRUP_NUMBA_CACHE", "0") == "1"
        _SWEEP = numba.njit(cache=cache, fastmath=False, nogil=True, boundscheck=False)(_sweep)
    return _SWEEP


def reset_compiled() -> None:
    """Drop the compiled dispatcher (tests: cold-start and missing-Numba paths)."""
    global _SWEEP
    _SWEEP = None


def run_sweep(graph, base_lo: np.ndarray, c: float, iterations: int):
    """Run the compiled sweep on host arrays. Returns
    (qin_new_lo, q_new_lo, flow_lo, rhs_lo) in level order, all new arrays."""
    if graph.xp is not np:
        raise RoutingError(
            f"implementation 'numba' runs on host NumPy arrays only; the graph lives in "
            f"{graph.xp.__name__!r} and no host/device transfer is performed"
        )
    sweep = compiled_sweep()
    n = graph.n_active
    outputs = tuple(np.zeros(n, dtype=np.float64) for _ in range(4))
    bounds = np.asarray(graph.level_bounds, dtype=np.int64)
    sweep(bounds, graph.conveyance_lo, graph.donor_position, graph.donor_mask,
          np.ascontiguousarray(base_lo, dtype=np.float64), float(c), int(iterations), *outputs)
    return outputs
