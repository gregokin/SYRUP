"""Compiled host kernels for the sparse nonempty-bin (CSR) characteristic transport (CW-B2 candidate).

Count / scan / write reference for `characteristic_sparse.sparse_characteristic_step`.
It is the CPU reference the later CUDA kernels of the corrected CW-B0 design
(agent_handoffs/tasks/conservative_water/cw_b_design_corrected_report.md section
2.3-2.4) must reproduce; nothing here runs on a device.

State between substeps is the UNNORMALISED packet-mass CSR: `row_ptr`
`(n_rows + 1,)` int64 with `row = slot * nc + class` over the sorted active
cells, `bins` `(nnz,)` uint8 strictly increasing within a row, `mass`
`(nnz,)` FP64 kilograms and `pos` `(nnz,)` FP64 metres. Fractions appear only
in the public state (`characteristic_sparse`), exactly as the dense kernel
`characteristic_numba._substep` carries `W` across substeps and the caller
normalises once.

Order contract (the reason the dense Numba kernel is expected to be matched
BITWISE; tested, never assumed): the dense kernel accumulates every
destination quantity in the global `(source cell ascending, class, source bin
ascending, pickup packet last)` order. For one destination row `(j, c)` the
sources that can contribute are its D4 donors and the cell itself, i.e. a
subset of `{j - nx, j - 1, j, j + 1, j + nx}`, already in ascending flat
order. Each destination row therefore visits those sources in that order and
their packets in bin order with the pickup last, and every floating-point
accumulation (`in_internal`, arrival deposition, bin mass, narrow and wide
log-mean numerators) is the same sequence of additions as the dense kernel's.
Per-source sums (`dep_decay`, `settled`, `export`, `out_internal`) only
involve one row's packets in bin order. Extrema (`xmax`, `xmin`, `smax`) are
order-independent. The scalar expressions are copied from
`characteristic_numba._substep` lines 94-191 unchanged.

Row sums: the per-row remaining mass that normalises the final fractions is
`W.sum(axis=-1)` on the dense path. `_row_sums` here is the compiled EMULATION
of that reduction and the PRODUCTION path of `sparse_characteristic_step`
(CW-B2 correction: selected after the root's independent raw-byte probes on
every supported bin count and a ~6x local microbenchmark). It reproduces
NumPy's `pairwise_sum` (`numpy/core/src/umath/loops_utils.h.src`: sequential
for `n < 8`, eight accumulators combined as `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))`
plus a sequential tail for `8 <= n <= 128`) over a local `B`-length scratch
holding exact zeros in the empty bins, with the reduction's identity start
`0.0 +`. `MAX_N_BINS = 128 = PW_BLOCKSIZE`, so the recursive branch never
applies. Bitwise agreement with NumPy is NOT assumed across NumPy versions:
`characteristic_sparse.row_sum_emulation_self_check` compares this kernel with
the bounded NumPy reference `characteristic_sparse.row_sums_numpy` (raw bytes,
every supported bin edge, signed wide magnitudes, empty/single/full rows) on
the running NumPy before the first production use in a process and refuses
(`RowSumEmulationMismatch`, no fallback) on any difference. It is also the
reference the later CUDA kernel must reproduce.

Memory: outputs are `(n_rows,)`, `(n_active, nc)` and `(nnz,)` arrays; the
only `B`-sized objects are the sequential kernel's local scratch (`B`
masses / extrema / numerators / flags and `5 (B + 1)` candidate slots, the
bound on candidates per destination row: own non-crossing packets plus the
crossing packets of at most four donors, each at most `B` bins plus one
pickup). No `(n, nc, B)` array exists.

Constraints, by user direction (as `routing_numba.py`, `characteristic_numba.py`):
no `fastmath`, no `prange` (fixed summation order), host NumPy only, no
host/device transfer, no silent fallback -- a missing Numba raises
`NumbaUnavailableError` (the existing `TransportError` subclass of
`characteristic_numba`). On-disk caching is off unless
`MAPLE_SYRUP_NUMBA_CACHE=1`.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from maple_syrup.characteristic_numba import NumbaUnavailableError
from maple_syrup.sediment_transport import TransportError

__all__ = [
    "PW_BLOCKSIZE",
    "NumbaUnavailableError",
    "candidate_scratch_length",
    "compiled_kernels",
    "numba_available",
    "reset_compiled",
    "run_count_pass",
    "run_row_sums",
    "run_write_pass",
]

# NumPy's pairwise-summation block size; MAX_N_BINS (128) never exceeds it.
PW_BLOCKSIZE = 128
# Sentinels identical to characteristic_transport / characteristic_numba.
_BELOW_ANY_POSITION = -1.0
_BELOW_ANY_LOG_WEIGHT = -1.0e300
# Sources that can land in one destination row: the cell itself and at most
# four D4 donors, each with at most B bins plus the pickup packet.
_SOURCES_PER_ROW = 5

_COMPILED: Any = None


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def candidate_scratch_length(n_bins: int) -> int:
    """Bound on candidates per destination row: `5 (B + 1)`."""
    return _SOURCES_PER_ROW * (int(n_bins) + 1)


# --- scalar helpers (identical expressions to characteristic_numba._substep) ----------------------
def _bin(xnew, dx, nb):
    # Identical expression to `characteristic_transport.bin_index`.
    b = math.floor(xnew / dx * nb)
    return min(b, nb - 1)


def _source_row(si, c, row_ptr, mass, pos, P, V, R, S, outlet, nc, dx, dt):
    """Per-source bookkeeping of row `(si, c)`: own packets in bin order, the
    pickup packet last. Returns `(dep_decay, settled, export, out_internal)`
    accumulated from 0.0 in the dense kernel's order."""
    dep_decay = 0.0
    settled = 0.0
    export = 0.0
    out_internal = 0.0
    row = si * nc + c
    e0 = row_ptr[row]
    e1 = row_ptr[row + 1]
    n_pk = e1 - e0 + 1
    s_src = S[si, c]
    vel = V[si, c]
    r = R[si, c]
    is_outlet = outlet[si]
    for k in range(n_pk):
        if k == n_pk - 1:
            m = P[si, c]
            x = 0.0
        else:
            m = mass[e0 + k]
            x = pos[e0 + k]
        if m == 0.0:
            continue
        if s_src:
            settled += m
            continue
        travel = vel * dt
        cross = x + travel >= dx
        if cross:
            travel = dx - x
        lost = m * (-math.expm1(-r * travel))
        surv = m * math.exp(-r * travel)
        dep_decay += lost
        if cross:
            if is_outlet:
                export += surv
            else:
                out_internal += surv
    return dep_decay, settled, export, out_internal


def _row_candidates(js, c, row_ptr, mass, pos, P, V, R, S, cell_of_slot, slot_of_cell, receiver_slot,
                    ny, nx, nc, dx, dt, nb, cand_b, cand_x, cand_m):
    """Candidates landing in destination row `(js, c)` this substep, in the
    dense kernel's order (source cell ascending; own packets in bin order with
    the pickup last; donors' crossing packets likewise). Fills `cand_b`,
    `cand_x`, `cand_m` and returns `(count, in_internal, arrival_decay,
    arrival_settled)` for the row, each accumulated from 0.0 in that order."""
    j = cell_of_slot[js]
    rj = j // nx
    cj = j - rj * nx
    count = 0
    in_sum = 0.0
    arr_decay = 0.0
    arr_settled = 0.0
    s_dest = S[js, c]
    v_dest = V[js, c]
    r_dest = R[js, c]
    for q in range(_SOURCES_PER_ROW):
        if q == 0:
            if rj == 0:
                continue
            i = j - nx
        elif q == 1:
            if cj == 0:
                continue
            i = j - 1
        elif q == 2:
            i = j
        elif q == 3:
            if cj == nx - 1:
                continue
            i = j + 1
        else:
            if rj == ny - 1:
                continue
            i = j + nx
        si = slot_of_cell[i]
        if si < 0:
            continue
        own = i == j
        if (not own) and receiver_slot[si] != js:
            continue
        if S[si, c]:
            continue  # settled at its source: neither moves nor arrives
        vel = V[si, c]
        r = R[si, c]
        row = si * nc + c
        e0 = row_ptr[row]
        e1 = row_ptr[row + 1]
        n_pk = e1 - e0 + 1
        for k in range(n_pk):
            if k == n_pk - 1:
                m = P[si, c]
                x = 0.0
            else:
                m = mass[e0 + k]
                x = pos[e0 + k]
            if m == 0.0:
                continue
            travel = vel * dt
            cross = x + travel >= dx
            if cross:
                travel = dx - x
            surv = m * math.exp(-r * travel)
            if not cross:
                if not own:
                    continue  # stays in its own cell, which is another row
                xnew = x + travel
                mnew = surv
            else:
                if own:
                    continue  # leaves this cell (export or receiver)
                in_sum += surv
                if s_dest:
                    arr_settled += surv
                    continue
                rem = dt - travel / vel
                rem = max(rem, 0.0)
                d = v_dest * rem
                lost2 = surv * (-math.expm1(-r_dest * d))
                arr_decay += lost2
                xnew = d
                mnew = surv * math.exp(-r_dest * d)
            if mnew <= 0.0:
                continue
            b = _bin(xnew, dx, nb)
            cand_b[count] = b
            cand_x[count] = xnew
            cand_m[count] = mnew
            count += 1
    return count, in_sum, arr_decay, arr_settled


def _pairwise_sum(a, n):
    """NumPy `pairwise_sum` for `n <= PW_BLOCKSIZE` on a contiguous FP64
    vector, including the reduction's identity start (`0.0 + result`)."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[i]
        return res
    r0 = a[0]
    r1 = a[1]
    r2 = a[2]
    r3 = a[3]
    r4 = a[4]
    r5 = a[5]
    r6 = a[6]
    r7 = a[7]
    i = 8
    stop = n - (n % 8)
    while i < stop:
        r0 += a[i]
        r1 += a[i + 1]
        r2 += a[i + 2]
        r3 += a[i + 3]
        r4 += a[i + 4]
        r5 += a[i + 5]
        r6 += a[i + 6]
        r7 += a[i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[i]
        i += 1
    return 0.0 + res


# --- passes ---------------------------------------------------------------------------------------
def _count_pass(row_ptr, mass, pos, P, V, R, S, cell_of_slot, slot_of_cell, receiver_slot, outlet,
                ny, nx, nb, dx, dt):
    """Pass 1 of a substep: per-source bookkeeping, per-destination arrivals,
    and the number of occupied destination bins of every row (`counts`, for
    the scan). Returns `(counts, dep_decay, settled, export, out_internal,
    in_internal, arrival_decay, arrival_settled)`; the seven FP64 arrays are
    `(n_active, nc)`."""
    n_active, nc = P.shape
    n_rows = n_active * nc
    counts = np.zeros(n_rows, dtype=np.int64)
    dep_decay = np.zeros((n_active, nc))
    settled = np.zeros((n_active, nc))
    export = np.zeros((n_active, nc))
    out_internal = np.zeros((n_active, nc))
    in_internal = np.zeros((n_active, nc))
    arrival_decay = np.zeros((n_active, nc))
    arrival_settled = np.zeros((n_active, nc))
    cap = _SOURCES_PER_ROW * (nb + 1)
    cand_b = np.empty(cap, dtype=np.int64)
    cand_x = np.empty(cap)
    cand_m = np.empty(cap)
    occupied = np.zeros(nb, dtype=np.bool_)
    for si in range(n_active):
        for c in range(nc):
            d, s, e, o = _source_row(si, c, row_ptr, mass, pos, P, V, R, S, outlet, nc, dx, dt)
            dep_decay[si, c] = d
            settled[si, c] = s
            export[si, c] = e
            out_internal[si, c] = o
            cnt, in_sum, ad, ast = _row_candidates(si, c, row_ptr, mass, pos, P, V, R, S, cell_of_slot,
                                                   slot_of_cell, receiver_slot, ny, nx, nc, dx, dt, nb,
                                                   cand_b, cand_x, cand_m)
            in_internal[si, c] = in_sum
            arrival_decay[si, c] = ad
            arrival_settled[si, c] = ast
            n_occ = 0
            for k in range(cnt):
                b = cand_b[k]
                if not occupied[b]:
                    occupied[b] = True
                    n_occ += 1
            for k in range(cnt):
                occupied[cand_b[k]] = False
            counts[si * nc + c] = n_occ
    return counts, dep_decay, settled, export, out_internal, in_internal, arrival_decay, arrival_settled


def _write_pass(row_ptr, mass, pos, P, V, R, S, cell_of_slot, slot_of_cell, receiver_slot,
                ny, nx, nb, dx, dt, narrow_spread, row_ptr_new, bins_new, mass_new, pos_new):
    """Pass 3 of a substep (after the host scan): regenerate each row's
    candidates, merge them with the three dense-kernel sweeps (mass + extrema;
    narrow numerator or wide `smax`; wide numerator), form the representatives
    with round-off containment and write the occupied bins in ascending bin
    order at `row_ptr_new[row] + rank`. Writes only into the preallocated
    `bins_new`, `mass_new`, `pos_new`."""
    n_active, nc = P.shape
    cap = _SOURCES_PER_ROW * (nb + 1)
    cand_b = np.empty(cap, dtype=np.int64)
    cand_x = np.empty(cap)
    cand_m = np.empty(cap)
    mass_b = np.zeros(nb)
    xmax_b = np.full(nb, _BELOW_ANY_POSITION)
    xmin_b = np.full(nb, 2.0 * dx)
    num_b = np.zeros(nb)
    smax_b = np.full(nb, _BELOW_ANY_LOG_WEIGHT)
    occupied = np.zeros(nb, dtype=np.bool_)
    for si in range(n_active):
        for c in range(nc):
            row = si * nc + c
            start = row_ptr_new[row]
            if row_ptr_new[row + 1] == start:
                continue  # the count pass found no candidate for this row
            cnt, _in_sum, _ad, _ast = _row_candidates(si, c, row_ptr, mass, pos, P, V, R, S, cell_of_slot,
                                                      slot_of_cell, receiver_slot, ny, nx, nc, dx, dt, nb,
                                                      cand_b, cand_x, cand_m)
            for k in range(cnt):
                b = cand_b[k]
                mass_b[b] = 0.0
                xmax_b[b] = _BELOW_ANY_POSITION
                xmin_b[b] = 2.0 * dx
                num_b[b] = 0.0
                smax_b[b] = _BELOW_ANY_LOG_WEIGHT
                occupied[b] = True
            # Sweep 1: masses and extrema (dense pass 1 order).
            for k in range(cnt):
                b = cand_b[k]
                x = cand_x[k]
                mass_b[b] += cand_m[k]
                xmax_b[b] = max(xmax_b[b], x)
                xmin_b[b] = min(xmin_b[b], x)
            r = R[si, c]
            # Sweep 2: narrow numerators or wide log-weight maxima; r = 0 moments.
            for k in range(cnt):
                b = cand_b[k]
                m = cand_m[k]
                x = cand_x[k]
                if r > 0.0:
                    if r * (xmax_b[b] - xmin_b[b]) <= narrow_spread:
                        if xmax_b[b] != xmin_b[b]:
                            num_b[b] += m * math.expm1(r * (x - xmax_b[b]))
                    else:
                        lw = math.log(m) + r * x
                        smax_b[b] = max(smax_b[b], lw)
                else:
                    num_b[b] += m * x
            # Sweep 3: wide numerators once the maxima are known.
            if r > 0.0:
                for k in range(cnt):
                    b = cand_b[k]
                    if r * (xmax_b[b] - xmin_b[b]) > narrow_spread:
                        lw = math.log(cand_m[k]) + r * cand_x[k]
                        num_b[b] += math.exp(lw - smax_b[b])
            # Representatives with round-off containment, ascending bin order.
            out = start
            for b in range(nb):
                if not occupied[b]:
                    continue
                occupied[b] = False
                mw = mass_b[b]
                if r > 0.0:
                    if r * (xmax_b[b] - xmin_b[b]) <= narrow_spread:
                        xr = xmax_b[b] + math.log1p(num_b[b] / mw) / r
                    else:
                        xr = (smax_b[b] + math.log(num_b[b]) - math.log(mw)) / r
                else:
                    xr = num_b[b] / mw
                lo = xmin_b[b]
                hi = xmax_b[b]
                xr = max(xr, lo)
                xr = min(xr, hi)
                bins_new[out] = np.uint8(b)
                mass_new[out] = mw
                pos_new[out] = xr
                out += 1


def _row_sums(row_ptr, bins, mass, nb, n_rows):
    """Remaining mass per row as NumPy's `sum(axis=-1)` of the dense `(..., B)`
    layout computes it (module docstring); production path, self-checked
    against the running NumPy by `characteristic_sparse`."""
    out = np.zeros(n_rows)
    buf = np.zeros(nb)
    for row in range(n_rows):
        e0 = row_ptr[row]
        e1 = row_ptr[row + 1]
        if e1 == e0:
            continue  # 0.0 + pairwise(zeros) = 0.0
        for e in range(e0, e1):
            buf[bins[e]] = mass[e]
        out[row] = _pairwise_sum(buf, nb)
        for e in range(e0, e1):
            buf[bins[e]] = 0.0
    return out


# Pure-Python originals, restored by `reset_compiled` after the globals were
# rebound to dispatchers (the outer kernels resolve the helpers by global
# name at compile time).
_PY_HELPERS = {"_bin": _bin, "_source_row": _source_row, "_row_candidates": _row_candidates,
               "_pairwise_sum": _pairwise_sum}


def compiled_kernels():
    """`(count_pass, write_pass, row_sums)` nopython dispatchers, compiled
    lazily once per process. Raises `NumbaUnavailableError` without Numba."""
    global _COMPILED
    if _COMPILED is None:
        try:
            import numba
        except ImportError as exc:
            raise NumbaUnavailableError(
                "the sparse CSR characteristic kernels require Numba (optional extra maple-syrup[numba]); "
                "there is no array or Python fallback"
            ) from exc
        cache = os.environ.get("MAPLE_SYRUP_NUMBA_CACHE", "0") == "1"
        jit = numba.njit(cache=cache, fastmath=False, nogil=True, boundscheck=False)
        for name, function in _PY_HELPERS.items():
            globals()[name] = jit(function)
        _COMPILED = (jit(_count_pass), jit(_write_pass), jit(_row_sums))
    return _COMPILED


def reset_compiled() -> None:
    """Drop the compiled dispatchers and restore the Python helpers (tests:
    cold-start and missing-Numba paths)."""
    global _COMPILED
    _COMPILED = None
    globals().update(_PY_HELPERS)


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _i64(a):
    return np.ascontiguousarray(a, dtype=np.int64)


def run_count_pass(row_ptr, mass, pos, P, V, R, S, cell_of_slot, slot_of_cell, receiver_slot, outlet,
                   ny: int, nx: int, nb: int, dx: float, dt: float):
    """Compiled pass 1 on host arrays (see `_count_pass`). No input is modified."""
    count, _write, _sums = compiled_kernels()
    return count(_i64(row_ptr), _f64(mass), _f64(pos), _f64(P), _f64(V), _f64(R),
                 np.ascontiguousarray(S, dtype=np.bool_), _i64(cell_of_slot), _i64(slot_of_cell),
                 _i64(receiver_slot), np.ascontiguousarray(outlet, dtype=np.bool_),
                 int(ny), int(nx), int(nb), float(dx), float(dt))


def run_write_pass(row_ptr, mass, pos, P, V, R, S, cell_of_slot, slot_of_cell, receiver_slot,
                   ny: int, nx: int, nb: int, dx: float, dt: float, narrow_spread: float,
                   row_ptr_new, bins_new, mass_new, pos_new) -> None:
    """Compiled pass 3 (see `_write_pass`); writes only the three preallocated
    output arrays, whose length is the scanned `nnz`."""
    nnz = int(row_ptr_new[-1])
    for name, array, dtype in (("bins_new", bins_new, np.uint8), ("mass_new", mass_new, np.float64),
                               ("pos_new", pos_new, np.float64)):
        if not isinstance(array, np.ndarray) or array.ndim != 1 or array.size != nnz or array.dtype != dtype \
                or not array.flags.writeable or not array.flags.c_contiguous:
            raise TransportError(f"{name} must be a writeable contiguous {np.dtype(dtype)} array of length {nnz}")
    _count, write, _sums = compiled_kernels()
    write(_i64(row_ptr), _f64(mass), _f64(pos), _f64(P), _f64(V), _f64(R),
          np.ascontiguousarray(S, dtype=np.bool_), _i64(cell_of_slot), _i64(slot_of_cell), _i64(receiver_slot),
          int(ny), int(nx), int(nb), float(dx), float(dt), float(narrow_spread),
          _i64(row_ptr_new), bins_new, mass_new, pos_new)


def run_row_sums(row_ptr, bins, mass, nb: int):
    """Compiled per-row remaining mass (see `_row_sums`), `(n_rows,)` FP64."""
    _count, _write, sums = compiled_kernels()
    n_rows = int(row_ptr.shape[0]) - 1
    return sums(_i64(row_ptr), np.ascontiguousarray(bins, dtype=np.uint8), _f64(mass), int(nb), n_rows)
