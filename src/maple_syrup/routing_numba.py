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
    "compiled_sweep_batched",
    "numba_available",
    "numba_versions",
    "reset_compiled",
    "run_sweep",
]

_SWEEP: Any = None
_BATCHED: Any = None


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


def _sweep_batched(bounds, k_lo, donor_position, donor_mask, base_lo, c, iterations,
                   qin_new_lo, q_new_lo, flow_lo, rhs_lo):
    """Level-batched method-5 sweep, bit-identical to `_sweep` (same signature, same outputs).

    Per level, in series: (1) for every cell the donor sum `((0 + d0) + d1) + d2) + d3` (non-donors add
    0.0, donors are always in earlier levels) and `rhs = base + qin * c`, stored RAW; (2) the root
    iterations OUTER and the independent cells INNER over contiguous scratch, with exactly the `_sweep`
    multiply sequence and the strict `<` test; (3) `q = (sqrt(lo) * lo) * k` for every cell, always the
    same formula.

    Only cells with `rhs > 0.0` enter the root iterations. For `not (rhs > 0.0)` (+-0, negative, -inf,
    NaN) `_sweep` provably leaves `lo = +0.0`: rhs = +-0 gives mid = +0, t = +0 and `0 < rhs` is False;
    rhs < 0 gives a negative mid, NaN t (or +0 after underflow of w) and a False comparison; NaN
    propagates to a False comparison. Positive tiny, subnormal and +inf rhs run the unchanged operations.
    No threshold, no clipping. All four outputs are written for every active cell. Scratch is allocated
    per call (never shared). No fastmath, no prange; one level's cells are independent, which is the
    structure a device kernel (one launch/barrier per level, predicated lanes) would use."""
    n_levels = bounds.shape[0] - 1
    width = 1
    for lev in range(n_levels):
        width = max(width, bounds[lev + 1] - bounds[lev])
    idx = np.empty(width, dtype=np.int64)
    r_s = np.empty(width, dtype=np.float64)
    k_s = np.empty(width, dtype=np.float64)
    w_s = np.empty(width, dtype=np.float64)
    l_s = np.empty(width, dtype=np.float64)
    for lev in range(n_levels):
        b0 = bounds[lev]
        b1 = bounds[lev + 1]
        m = 0
        for p in range(b0, b1):
            qin = 0.0
            for s in range(4):
                if donor_mask[s, p]:
                    qin = qin + q_new_lo[donor_position[s, p]]
                else:
                    qin = qin + 0.0
            rhs = base_lo[p] + qin * c
            qin_new_lo[p] = qin
            rhs_lo[p] = rhs
            flow_lo[p] = 0.0
            if rhs > 0.0:
                idx[m] = p
                r_s[m] = rhs
                k_s[m] = k_lo[p]
                w_s[m] = rhs
                l_s[m] = 0.0
                m += 1
        for _ in range(iterations):
            for j in range(m):
                w = w_s[j] * 0.5
                w_s[j] = w
                lo = l_s[j]
                mid = lo + w
                t = np.sqrt(mid)
                t = t * mid
                t = t * k_s[j]
                t = t * c
                t = t + mid
                if t < r_s[j]:
                    l_s[j] = mid
        for j in range(m):
            flow_lo[idx[j]] = l_s[j]
        for p in range(b0, b1):
            lo = flow_lo[p]
            q = np.sqrt(lo)
            q = q * lo
            q = q * k_lo[p]
            q_new_lo[p] = q


def _require_numba():
    try:
        import numba
    except ImportError as exc:
        raise NumbaUnavailableError(
            "implementation 'numba' requested but Numba is not installed (optional extra "
            "maple-syrup[numba]); there is no fallback to the array implementation"
        ) from exc
    return numba


def compiled_sweep():
    """The nopython-compiled `_sweep` (compiled lazily, once per process)."""
    global _SWEEP
    if _SWEEP is None:
        numba = _require_numba()
        cache = os.environ.get("MAPLE_SYRUP_NUMBA_CACHE", "0") == "1"
        _SWEEP = numba.njit(cache=cache, fastmath=False, nogil=True, boundscheck=False)(_sweep)
    return _SWEEP


def compiled_sweep_batched():
    """The nopython-compiled level-batched `_sweep_batched` (CPU only; host NumPy arrays; same arguments and
    bit-identical results as `compiled_sweep()`; compiled lazily, once per process; a missing Numba raises
    `NumbaUnavailableError`, no fallback)."""
    global _BATCHED
    if _BATCHED is None:
        numba = _require_numba()
        cache = os.environ.get("MAPLE_SYRUP_NUMBA_CACHE", "0") == "1"
        _BATCHED = numba.njit(cache=cache, fastmath=False, nogil=True, boundscheck=False)(_sweep_batched)
    return _BATCHED


def reset_compiled() -> None:
    """Drop the compiled dispatchers (tests: cold-start and missing-Numba paths)."""
    global _SWEEP, _BATCHED
    _SWEEP = None
    _BATCHED = None


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
