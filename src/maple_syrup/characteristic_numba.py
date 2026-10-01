"""Optional Numba-compiled host kernel for `characteristic_transport` (`implementation="numba"`).

Same scalar expressions as `_substep_array` (candidate
formation with exact time-to-face and receiver continuation, per-bin
extrema, shifted log-sum-exp merge, containment), with the Python-level
work over `(cell, class, slot)` moved into ONE nopython call. A bounded
compact candidate list avoids revisiting empty slots. Narrow/wide/zero-rate
bins share numerator scratch, and expensive log weights are evaluated only
for wide bins after their extrema are known. The
candidate accumulation order (cell, class, slot) is the flattened order
`np.add.at` uses on the array path, so the two host implementations agree
to round-off; they are compared by test, never assumed identical.

Constraints, by user direction (as `routing_numba.py`): no `fastmath`, no
`prange`, host NumPy only, no host/device transfer, no silent fallback --
a missing Numba raises `NumbaUnavailableError` (a `TransportError`).
On-disk caching is off unless `MAPLE_SYRUP_NUMBA_CACHE=1`.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from maple_syrup.sediment_transport import TransportError

__all__ = [
    "NumbaUnavailableError",
    "compiled_substep",
    "numba_available",
    "reset_compiled",
    "run_substep",
]

_SUBSTEP: Any = None


class NumbaUnavailableError(TransportError):
    """`implementation="numba"` was requested but Numba cannot be imported.
    There is no fallback to the array path."""


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def _substep(W, X, P, V, R, S, receiver, outlet, dx, dt, nb, narrow_spread):
    # `receiver` is the network's `receiver_index` (the cell itself at
    # outlets and inactive cells); `outlet` its `outlet_flat`.
    n, nc, _ = W.shape
    W_out = np.zeros((n, nc, nb))
    X_out = np.zeros((n, nc, nb))
    dep_decay = np.zeros((n, nc))
    settled = np.zeros((n, nc))
    export = np.zeros((n, nc))
    out_internal = np.zeros((n, nc))
    in_internal = np.zeros((n, nc))
    arrival_decay = np.zeros((n, nc))
    arrival_settled = np.zeros((n, nc))
    # A nonzero source count bounds surviving candidates even with merging,
    # settling, export and decay. This is an exact zero test, not a mass cutoff.
    capacity = np.count_nonzero(W) + np.count_nonzero(P)
    if capacity == 0:
        return W_out, X_out, dep_decay, settled, export, out_internal, in_internal, arrival_decay, arrival_settled
    xmax = np.full((n, nc, nb), -1.0)
    xmin = np.full((n, nc, nb), 2.0 * dx)
    smax = np.full((n, nc, nb), -1.0e300)
    # Each destination bin uses exactly one of the narrow, wide or zero-rate
    # numerators. Share that scratch storage. Compact candidates retain source
    # traversal order, hence the same summation order as the dense reference.
    numerator = np.zeros((n, nc, nb))
    cand_dest = np.empty(capacity, dtype=np.int64)
    cand_x = np.empty(capacity)
    cand_mass = np.empty(capacity)
    cand_logw = np.empty(capacity)
    count = 0
    # Pass 1: characteristics, bookkeeping, candidate destinations, extrema.
    for i in range(n):
        for c in range(nc):
            for k in range(nb + 1):
                if k == nb:
                    m = P[i, c]
                    x = 0.0
                else:
                    m = W[i, c, k]
                    x = X[i, c, k]
                if m == 0.0:
                    continue
                if S[i, c]:
                    settled[i, c] += m
                    continue
                vel = V[i, c]
                r = R[i, c]
                travel = vel * dt
                cross = x + travel >= dx
                if cross:
                    travel = dx - x
                lost = m * (-math.expm1(-r * travel))
                surv = m * math.exp(-r * travel)
                dep_decay[i, c] += lost
                if not cross:
                    dest = i
                    xnew = x + travel
                    mass = surv
                else:
                    if outlet[i]:
                        export[i, c] += surv
                        continue
                    j = receiver[i]
                    out_internal[i, c] += surv
                    in_internal[j, c] += surv
                    if S[j, c]:
                        arrival_settled[j, c] += surv
                        continue
                    rem = dt - travel / vel
                    rem = max(rem, 0.0)
                    d = V[j, c] * rem
                    rj = R[j, c]
                    lost2 = surv * (-math.expm1(-rj * d))
                    arrival_decay[j, c] += lost2
                    dest = j
                    xnew = d
                    mass = surv * math.exp(-rj * d)
                if mass <= 0.0:
                    continue
                # Identical expression to `characteristic_transport.bin_index`.
                b = math.floor(xnew / dx * nb)
                b = min(b, nb - 1)
                cand_dest[count] = (dest * nc + c) * nb + b
                cand_x[count] = xnew
                cand_mass[count] = mass
                count += 1
                W_out[dest, c, b] += mass
                xmax[dest, c, b] = max(xmax[dest, c, b], xnew)
                xmin[dest, c, b] = min(xmin[dest, c, b], xnew)
    # Once extrema are known, compute logarithms only for WIDE bins.
    # Singleton/equal-position positive-rate bins have an exactly zero narrow
    # numerator; skipping expm1(0) preserves the reference representative.
    flat_num = numerator.reshape(-1)
    flat_hi = xmax.reshape(-1)
    flat_lo = xmin.reshape(-1)
    flat_smax = smax.reshape(-1)
    rates = R.reshape(-1)
    for k in range(count):
        dest = cand_dest[k]
        mass = cand_mass[k]
        x = cand_x[k]
        r = rates[dest // nb]
        if r > 0.0:
            if r * (flat_hi[dest] - flat_lo[dest]) <= narrow_spread:
                if flat_hi[dest] != flat_lo[dest]:
                    flat_num[dest] += mass * math.expm1(r * (x - flat_hi[dest]))
            else:
                lw = math.log(mass) + r * x
                cand_logw[k] = lw
                flat_smax[dest] = max(flat_smax[dest], lw)
        else:
            flat_num[dest] += mass * x
    for k in range(count):
        dest = cand_dest[k]
        r = rates[dest // nb]
        if r > 0.0 and r * (flat_hi[dest] - flat_lo[dest]) > narrow_spread:
            flat_num[dest] += math.exp(cand_logw[k] - flat_smax[dest])
    # Pass 3: representatives with round-off containment.
    for i in range(n):
        for c in range(nc):
            r = R[i, c]
            for b in range(nb):
                mass = W_out[i, c, b]
                if mass <= 0.0:
                    X_out[i, c, b] = 0.0
                    continue
                if r > 0.0:
                    if r * (xmax[i, c, b] - xmin[i, c, b]) <= narrow_spread:
                        xr = xmax[i, c, b] + math.log1p(numerator[i, c, b] / mass) / r
                    else:
                        xr = (smax[i, c, b] + math.log(numerator[i, c, b]) - math.log(mass)) / r
                else:
                    xr = numerator[i, c, b] / mass
                lo = xmin[i, c, b]
                hi = xmax[i, c, b]
                xr = max(xr, lo)
                xr = min(xr, hi)
                X_out[i, c, b] = xr
    return W_out, X_out, dep_decay, settled, export, out_internal, in_internal, arrival_decay, arrival_settled


def compiled_substep():
    """The nopython-compiled `_substep` (compiled lazily, once per process)."""
    global _SUBSTEP
    if _SUBSTEP is None:
        try:
            import numba
        except ImportError as exc:
            raise NumbaUnavailableError(
                "implementation 'numba' requested but Numba is not installed (optional extra "
                "maple-syrup[numba]); there is no fallback to the array implementation"
            ) from exc
        cache = os.environ.get("MAPLE_SYRUP_NUMBA_CACHE", "0") == "1"
        _SUBSTEP = numba.njit(cache=cache, fastmath=False, nogil=True, boundscheck=False)(_substep)
    return _SUBSTEP


def reset_compiled() -> None:
    """Drop the compiled dispatcher (tests: cold-start and missing-Numba paths)."""
    global _SUBSTEP
    _SUBSTEP = None


def run_substep(network, W, X, P, V, R, S, dx: float, dt: float, nb: int):
    """One substep on host arrays (`W`, `X` `(n, nc, B)`; `P`, `V`, `R` `(n, nc)`
    FP64; `S` bool). Returns nine new arrays in the order
    `_substep_array` uses. No input is modified."""
    if network.xp is not np:
        raise TransportError("implementation 'numba' runs on host NumPy arrays only")
    from maple_syrup.characteristic_transport import NARROW_SPREAD

    kernel = compiled_substep()
    return kernel(np.ascontiguousarray(W, dtype=np.float64), np.ascontiguousarray(X, dtype=np.float64),
                  np.ascontiguousarray(P, dtype=np.float64), np.ascontiguousarray(V, dtype=np.float64),
                  np.ascontiguousarray(R, dtype=np.float64), np.ascontiguousarray(S, dtype=np.bool_),
                  np.ascontiguousarray(network.receiver_index, dtype=np.int64),
                  np.ascontiguousarray(network.outlet_flat, dtype=np.bool_),
                  float(dx), float(dt), int(nb), float(NARROW_SPREAD))
