"""Frozen pre-optimization Numba arithmetic, for bitwise regression only.

Extracted from Phase 7b characteristic_numba.py before Phase 7c edits.
Original complete module SHA256: 433ad523c51fb52f22b8357d8683f6f70ae534b29d603c8a7d02b6a516ce7403
Never imported by production code. Changes to the optimized implementation
must not update this reference to make a mismatch pass.
"""
import math

import numpy as np


def _substep(W, X, P, V, R, S, receiver, outlet, dx, dt, nb, narrow_spread):
    # `receiver` is the network's `receiver_index` (the cell itself at
    # outlets and inactive cells); `outlet` its `outlet_flat`.
    n, nc, _ = W.shape
    W_out = np.zeros((n, nc, nb))
    X_out = np.zeros((n, nc, nb))
    xmax = np.full((n, nc, nb), -1.0)
    xmin = np.full((n, nc, nb), 2.0 * dx)
    smax = np.full((n, nc, nb), -1.0e300)
    dsum = np.zeros((n, nc, nb))
    esum = np.zeros((n, nc, nb))
    msum = np.zeros((n, nc, nb))
    cand_cell = np.zeros((n, nc, nb + 1), dtype=np.int64)
    cand_bin = np.zeros((n, nc, nb + 1), dtype=np.int64)
    cand_x = np.zeros((n, nc, nb + 1))
    cand_mass = np.zeros((n, nc, nb + 1))
    dep_decay = np.zeros((n, nc))
    settled = np.zeros((n, nc))
    export = np.zeros((n, nc))
    out_internal = np.zeros((n, nc))
    in_internal = np.zeros((n, nc))
    arrival_decay = np.zeros((n, nc))
    arrival_settled = np.zeros((n, nc))
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
                cand_cell[i, c, k] = dest
                cand_bin[i, c, k] = b
                cand_x[i, c, k] = xnew
                cand_mass[i, c, k] = mass
                W_out[dest, c, b] += mass
                xmax[dest, c, b] = max(xmax[dest, c, b], xnew)
                xmin[dest, c, b] = min(xmin[dest, c, b], xnew)
                lw = math.log(mass) + R[dest, c] * xnew
                smax[dest, c, b] = max(smax[dest, c, b], lw)
    # Pass 2: log-mean numerators. NARROW bins (r (xmax - xmin) <= narrow_spread)
    # use the expm1 form shifted by xmax; WIDE bins the positive sum shifted by
    # the largest log-weight; r = 0 bins the mass-weighted position.
    for i in range(n):
        for c in range(nc):
            for k in range(nb + 1):
                mass = cand_mass[i, c, k]
                if mass <= 0.0:
                    continue
                dest = cand_cell[i, c, k]
                b = cand_bin[i, c, k]
                x = cand_x[i, c, k]
                r = R[dest, c]
                if r > 0.0:
                    if r * (xmax[dest, c, b] - xmin[dest, c, b]) <= narrow_spread:
                        dsum[dest, c, b] += mass * math.expm1(r * (x - xmax[dest, c, b]))
                    else:
                        esum[dest, c, b] += math.exp(math.log(mass) + r * x - smax[dest, c, b])
                else:
                    msum[dest, c, b] += mass * x
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
                        xr = xmax[i, c, b] + math.log1p(dsum[i, c, b] / mass) / r
                    else:
                        xr = (smax[i, c, b] + math.log(esum[i, c, b]) - math.log(mass)) / r
                else:
                    xr = msum[i, c, b] / mass
                lo = xmin[i, c, b]
                hi = xmax[i, c, b]
                xr = max(xr, lo)
                xr = min(xr, hi)
                X_out[i, c, b] = xr
    return W_out, X_out, dep_decay, settled, export, out_internal, in_internal, arrival_decay, arrival_settled
