"""CUDA sparse nonempty-bin (CSR) characteristic transport: count / scan / write (CW-B4 candidate).

`sparse_characteristic_step(..., implementation="cuda")` on a CuPy network with a CuPy `SparsePhaseState`
and CuPy inputs. Same physics, quantization (`B` bins, 1..128, default 32), merge, request semantics,
`TransportStep` shapes and `SparseCharacteristicStep` semantics as the accepted CPU CSR reference
(`characteristic_sparse` / `characteristic_numba_csr`), whose Numba kernels are NOT modified here.

Order contract (identical to the CPU CSR kernels): one thread handles one destination row `(slot, class)`
and visits its possible sources `{j-nx, j-1, j, j+1, j+nx}` (own cell and D4 donors) in ascending flat
order, each source's packets in bin order with the pickup packet last. Every floating-point accumulation
(`in_internal`, arrival deposition, bin mass, narrow/wide log-mean numerators, the eight-lane row sum) is
the same sequence of additions as `characteristic_numba_csr`; per-source sums (`dep_decay`, `settled`,
`export`, `out_internal`) involve one row's packets in bin order. No floating-point atomics, no `fastmath`,
no FMA contraction (`--fmad=false`, round-to-nearest intrinsics `__dadd_rn`/`__dmul_rn`/`__ddiv_rn` for
every `+ - * /`). The transcendentals `exp`, `expm1`, `log`, `log1p` are the precise libdevice double
functions: their last bits are NOT the host libm's, so the GPU operator is compared with the CPU reference
at the existing isolated GPU characteristic bounds (rtol 1e-10 / atol 1e-15 on masses and transport
fields, atol 1e-9 m on positions; `tests/phase7b` T11), never bitwise. Integer controls (`counts`,
`row_ptr`, `nnz`, bins), the budgets' MAPLE bounds and same-device repeats (raw bytes) are exact.

Passes per substep (all rows in ONE launch each, clamped persistent grid-stride; no Python loop over rows
or cells):
  1. `ms_csr_count`  -- per row: source bookkeeping (4 sums), destination candidates regenerated from the
                        source CSR rows, arrivals (3 sums), occupied-bin bitmask -> `counts[row]`.
  2. scan            -- `cupy.cumsum(counts)` -> `row_ptr_new`; ONE counted scalar read of `nnz`
                        (`maple.core.backend.to_int`) enforces `max_nnz` BEFORE any write array exists.
  3. `ms_csr_write`  -- per row: candidates into a `5 (B + 1)` local scratch, the three dense-kernel sweeps
                        (mass + extrema; narrow numerator or wide `smax`; wide numerator), representatives
                        with containment, written in ascending bin order at `row_ptr_new[row] + rank`.
  Final: `ms_csr_rowsum_normalize` -- per row the NumPy pairwise reduction (sequential below 8 lanes, eight
  accumulators `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))` plus tail, identity start `0.0 +`) over a `B`-length
  local buffer with exact zeros, then `fraction = mass / remaining`. The unnormalised packet-mass CSR is
  carried across substeps on the device; fractions are formed once.

Memory: device arrays are `(n_rows + 1,)` pointers, `(nnz,)` entries, `(n_active, nc)` bookkeeping,
`(n, nc)` MAPLE-contract fields and faces. No `rows x B` array exists. The only `B`-sized storage is each
thread's local scratch (`MS_CAP = 5 (B + 1)` candidate slots and five `B`-length bin arrays), bounded by
the launched-thread cap (`DEFAULT_MAX_LAUNCHED_THREADS`, grid-stride over rows), so local/spill memory is
`threads x local_size_bytes`, independent of the row count. Kernels are specialised per `B` at compile
time (`MS_NB`), so the default B = 32 does not carry a 128-bin stack. The compiled `local_size_bytes`,
`num_regs`, `shared_size_bytes` and `max_threads_per_block` attributes are recorded per kernel beside the
analytic estimate (`estimate_local_scratch_bytes`), and reported separately as estimate vs compiled.

Capacity: `max_nnz` is enforced after the scan and before the write arrays are allocated
(`SparseCapacityError` naming required / cap / substep); the old phase and every input are untouched. For a
case whose full occupancy `n_rows x B x 17 B` exceeds `DEFAULT_UNBOUNDED_NNZ_BYTES` an explicit `max_nnz`
is REQUIRED (refused before any launch); the GPU never silently sizes a large case for all `B` bins.
`cuda_sparse_preflight` gives the byte estimates a caller needs to choose the cap.

Binding: the active-slot maps are derived ONCE from the network's HOST evidence (`active_host`,
`outlet_host`, `receiver_host` and the optional `terminal_host`, see
`characteristic_sparse.host_network_evidence`), uploaded once through counted `to_device` and cached by
network identity. What is and is not detected, truthfully: the cached metadata fingerprints (pointer,
shape, dtype, layout, device) detect a REPLACED or MOVED runtime array, not an in-place value change. For
the connectivity arrays the operator depends on (`receiver_index`, `active_flat`, `outlet_flat`,
`terminal_flat`), a one-packet device consistency check compares the owned maps with the runtime arrays at
binding and AGAIN in every step's input flag packet (gathers over `n_active`, no physical download), so an
in-place mutation of those arrays is refused before any launch or publication. The face tables (`cells_x`,
`rows_x`, ...) are read from the network at every step, never cached. Callers must still not mutate a
network: the contract is detection and refusal, not tolerance. `release_binding` drops a binding. A phase
state built for another graph is refused by its digest; `characteristic_sparse.rebind_sparse_phase_state` is
the explicit rebind.

Terminal storage (optional network fields `terminal_host` / `terminal_flat`, both present or both absent):
a terminal is an active non-outlet cell with the `PIT_STORAGE` receiver code whose runtime receiver is
itself. The operator applies the same explicit all-classes rule as the CPU step and the C2 operators: the
caller's settle mask must be True for every grain class at a terminal (the whole mobile load -- own pool,
pickup, arrivals -- settles through the deposition request and nothing is sent); a mask missing any class
is refused in the input flag packet before any launch, never coerced. Networks without the fields keep the
strict rule (every active non-outlet cell has an active receiver).

Validation: structural refusals (exact `cupy.ndarray` on the current device, dtype, shape, binding) are
host metadata; value refusals use MAPLE flags resolved in ONE batched read (`DeferredChecks`) before any
kernel launch, and the outputs are validated the same way before publication. Physical arrays are never
downloaded; the per-step device->host traffic is the `nnz` scalar per substep plus the one flag packet
before and after. `sparse_to_dense` / `sparse_from_dense` on CuPy are the same SMALL-domain, cap-first
helpers as on the host (device arrays, cap decided from metadata before any allocation or scan).

No fallback anywhere: missing CuPy / device / compile failure -> `CudaUnavailableError`
(`TransportError`); a NumPy network, a host array, a mismatched device or namespace -> `TransportError`.
Nothing here was executed by its author (file-only tools); the root records every result.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np
from maple.surface.voxels._numerics import summation_error_bound_kg

from maple_syrup.characteristic_transport import (
    DEFAULT_COURANT_MAX,
    MAX_N_BINS,
    MAX_SUBSTEPS,
    NARROW_SPREAD,
    PhaseState,
    bin_index,
    fraction_sum_tolerance,
    operation_count,
    validate_phase_state,
)
from maple_syrup.sediment_transport import (
    TransportError,
    TransportNetwork,
    TransportStep,
    TransportStepRejected,
)

__all__ = [
    "BLOCK_THREADS",
    "COMPILE_OPTIONS",
    "DEFAULT_MAX_LAUNCHED_THREADS",
    "DEFAULT_UNBOUNDED_NNZ_BYTES",
    "KERNEL_NAMES",
    "CudaCsrBinding",
    "CudaUnavailableError",
    "bind_network",
    "cuda_sparse_preflight",
    "cuda_sparse_step",
    "empty_cuda_sparse_phase_state",
    "enqueue_runtime_consistency",
    "estimate_local_scratch_bytes",
    "kernel_provenance",
    "kernel_source",
    "launch_shape",
    "load_kernels",
    "release_binding",
    "row_sum_device_self_check",
    "sparse_from_dense_cuda",
    "sparse_to_dense_cuda",
    "validate_cuda_sparse_phase_state",
]

_EPS = float(np.finfo(np.float64).eps)
BLOCK_THREADS = 128
# Clamped persistent grid: at most this many threads are launched, each striding over rows, so the
# per-thread local scratch of the write kernel is bounded independently of the row count.
DEFAULT_MAX_LAUNCHED_THREADS = 8192
# Above this full-occupancy entry volume (n_rows * B * ENTRY_BYTES) the CUDA path requires an explicit
# `max_nnz`; it never sizes a large case for every bin by default.
DEFAULT_UNBOUNDED_NNZ_BYTES = 1 << 26
COMPILE_OPTIONS = ("--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
KERNEL_NAMES = ("ms_csr_count", "ms_csr_write", "ms_csr_rowsum_normalize")
_ENTRY_BYTES = 8 + 8 + 1
_COURANT_REJECTION = "characteristic sediment Courant number v dt / (n_substeps dx) exceeds"
_PW_BLOCKSIZE = 128


class CudaUnavailableError(TransportError):
    """`implementation="cuda"` was requested but CuPy, a CUDA device or the compiled kernels are unavailable.
    There is no fallback to the Numba or array paths."""


# --- kernel source (specialised per B) -------------------------------------------------------------------
_SOURCE_TEMPLATE = r"""
#define MS_NB @NB@
#define MS_CAP (5 * (MS_NB + 1))
#define MS_OCC_WORDS ((MS_NB + 31) / 32)
#define MS_BELOW_ANY_POSITION (-1.0)
#define MS_BELOW_ANY_LOG_WEIGHT (-1.0e300)

/* Identical expression to characteristic_transport.bin_index: min(floor(x / dx * B), B - 1). */
__device__ __forceinline__ int ms_bin(const double xnew, const double dx)
{
    const double q = __dmul_rn(__ddiv_rn(xnew, dx), (double)MS_NB);
    const long long b = (long long)floor(q);
    return (int)(b < (long long)(MS_NB - 1) ? b : (long long)(MS_NB - 1));
}

/* Per-source bookkeeping of row (si, c): own packets in bin order, pickup last (characteristic_numba_csr._source_row). */
__device__ void ms_source_row(const long long si, const int c, const long long* __restrict__ row_ptr,
                              const double* __restrict__ mass, const double* __restrict__ pos,
                              const double* __restrict__ P, const double* __restrict__ V,
                              const double* __restrict__ R, const unsigned char* __restrict__ S,
                              const unsigned char* __restrict__ outlet, const int nc, const double dx,
                              const double dt, double* out4)
{
    double dep_decay = 0.0, settled = 0.0, export_ = 0.0, out_internal = 0.0;
    const long long row = si * (long long)nc + (long long)c;
    const long long e0 = row_ptr[row];
    const long long e1 = row_ptr[row + 1];
    const long long n_pk = e1 - e0 + 1;
    const bool s_src = S[row] != 0;
    const double vel = V[row];
    const double r = R[row];
    const bool is_outlet = outlet[si] != 0;
    for (long long k = 0; k < n_pk; ++k) {
        double m, x;
        if (k == n_pk - 1) { m = P[row]; x = 0.0; } else { m = mass[e0 + k]; x = pos[e0 + k]; }
        if (m == 0.0) continue;
        if (s_src) { settled = __dadd_rn(settled, m); continue; }
        double travel = __dmul_rn(vel, dt);
        const bool cross = __dadd_rn(x, travel) >= dx;
        if (cross) travel = __dadd_rn(dx, -x);
        const double lost = __dmul_rn(m, -expm1(__dmul_rn(-r, travel)));
        const double surv = __dmul_rn(m, exp(__dmul_rn(-r, travel)));
        dep_decay = __dadd_rn(dep_decay, lost);
        if (cross) {
            if (is_outlet) export_ = __dadd_rn(export_, surv);
            else out_internal = __dadd_rn(out_internal, surv);
        }
    }
    out4[0] = dep_decay; out4[1] = settled; out4[2] = export_; out4[3] = out_internal;
}

/* Candidates landing in destination row (js, c) in the dense/CSR kernel order
   (characteristic_numba_csr._row_candidates). STORE=false only marks occupied bins. */
template <bool STORE>
__device__ int ms_row_candidates(const long long js, const int c, const long long* __restrict__ row_ptr,
                                 const double* __restrict__ mass, const double* __restrict__ pos,
                                 const double* __restrict__ P, const double* __restrict__ V,
                                 const double* __restrict__ R, const unsigned char* __restrict__ S,
                                 const long long* __restrict__ cell_of_slot,
                                 const long long* __restrict__ slot_of_cell,
                                 const long long* __restrict__ receiver_slot,
                                 const long long ny, const long long nx, const int nc,
                                 const double dx, const double dt,
                                 int* cand_b, double* cand_x, double* cand_m, unsigned int* occ,
                                 double* in_sum_out, double* arr_decay_out, double* arr_settled_out)
{
    const long long j = cell_of_slot[js];
    const long long rj = j / nx;
    const long long cj = j - rj * nx;
    int count = 0;
    double in_sum = 0.0, arr_decay = 0.0, arr_settled = 0.0;
    const long long drow = js * (long long)nc + (long long)c;
    const bool s_dest = S[drow] != 0;
    const double v_dest = V[drow];
    const double r_dest = R[drow];
    for (int q = 0; q < 5; ++q) {
        long long i;
        if (q == 0) { if (rj == 0) continue; i = j - nx; }
        else if (q == 1) { if (cj == 0) continue; i = j - 1; }
        else if (q == 2) { i = j; }
        else if (q == 3) { if (cj == nx - 1) continue; i = j + 1; }
        else { if (rj == ny - 1) continue; i = j + nx; }
        const long long si = slot_of_cell[i];
        if (si < 0) continue;
        const bool own = (i == j);
        if (!own && receiver_slot[si] != js) continue;
        const long long srow = si * (long long)nc + (long long)c;
        if (S[srow] != 0) continue;  /* settled at its source: neither moves nor arrives */
        const double vel = V[srow];
        const double r = R[srow];
        const long long e0 = row_ptr[srow];
        const long long e1 = row_ptr[srow + 1];
        const long long n_pk = e1 - e0 + 1;
        for (long long k = 0; k < n_pk; ++k) {
            double m, x;
            if (k == n_pk - 1) { m = P[srow]; x = 0.0; } else { m = mass[e0 + k]; x = pos[e0 + k]; }
            if (m == 0.0) continue;
            double travel = __dmul_rn(vel, dt);
            const bool cross = __dadd_rn(x, travel) >= dx;
            if (cross) travel = __dadd_rn(dx, -x);
            const double surv = __dmul_rn(m, exp(__dmul_rn(-r, travel)));
            double xnew, mnew;
            if (!cross) {
                if (!own) continue;              /* stays in its own cell, another row */
                xnew = __dadd_rn(x, travel);
                mnew = surv;
            } else {
                if (own) continue;               /* leaves this cell (export or receiver) */
                in_sum = __dadd_rn(in_sum, surv);
                if (s_dest) { arr_settled = __dadd_rn(arr_settled, surv); continue; }
                double rem = __dadd_rn(dt, -__ddiv_rn(travel, vel));
                if (rem < 0.0) rem = 0.0;        /* Python max(rem, 0.0) */
                const double d = __dmul_rn(v_dest, rem);
                const double lost2 = __dmul_rn(surv, -expm1(__dmul_rn(-r_dest, d)));
                arr_decay = __dadd_rn(arr_decay, lost2);
                xnew = d;
                mnew = __dmul_rn(surv, exp(__dmul_rn(-r_dest, d)));
            }
            if (mnew <= 0.0) continue;
            const int b = ms_bin(xnew, dx);
            if (STORE) { cand_b[count] = b; cand_x[count] = xnew; cand_m[count] = mnew; }
            else { occ[b >> 5] |= (1u << (b & 31)); }
            ++count;
        }
    }
    *in_sum_out = in_sum; *arr_decay_out = arr_decay; *arr_settled_out = arr_settled;
    return count;
}

extern "C" __global__ void ms_csr_count(
    const long long n_active, const int nc, const long long ny, const long long nx, const double dx, const double dt,
    const long long* __restrict__ row_ptr, const double* __restrict__ mass, const double* __restrict__ pos,
    const double* __restrict__ P, const double* __restrict__ V, const double* __restrict__ R,
    const unsigned char* __restrict__ S, const long long* __restrict__ cell_of_slot,
    const long long* __restrict__ slot_of_cell, const long long* __restrict__ receiver_slot,
    const unsigned char* __restrict__ outlet,
    long long* counts, double* dep_decay, double* settled, double* export_, double* out_internal,
    double* in_internal, double* arrival_decay, double* arrival_settled)
{
    const long long n_rows = n_active * (long long)nc;
    const long long stride = (long long)gridDim.x * (long long)blockDim.x;
    for (long long row = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x; row < n_rows;
         row += stride) {
        const long long si = row / nc;
        const int c = (int)(row - si * nc);
        double s4[4];
        ms_source_row(si, c, row_ptr, mass, pos, P, V, R, S, outlet, nc, dx, dt, s4);
        dep_decay[row] = s4[0]; settled[row] = s4[1]; export_[row] = s4[2]; out_internal[row] = s4[3];
        unsigned int occ[MS_OCC_WORDS];
        for (int w = 0; w < MS_OCC_WORDS; ++w) occ[w] = 0u;
        double in_sum, ad, ast;
        ms_row_candidates<false>(si, c, row_ptr, mass, pos, P, V, R, S, cell_of_slot, slot_of_cell, receiver_slot,
                                 ny, nx, nc, dx, dt, (int*)0, (double*)0, (double*)0, occ, &in_sum, &ad, &ast);
        int n_occ = 0;
        for (int w = 0; w < MS_OCC_WORDS; ++w) n_occ += __popc(occ[w]);
        counts[row] = (long long)n_occ;
        in_internal[row] = in_sum; arrival_decay[row] = ad; arrival_settled[row] = ast;
    }
}

extern "C" __global__ void ms_csr_write(
    const long long n_active, const int nc, const long long ny, const long long nx, const double dx, const double dt,
    const double narrow_spread,
    const long long* __restrict__ row_ptr, const double* __restrict__ mass, const double* __restrict__ pos,
    const double* __restrict__ P, const double* __restrict__ V, const double* __restrict__ R,
    const unsigned char* __restrict__ S, const long long* __restrict__ cell_of_slot,
    const long long* __restrict__ slot_of_cell, const long long* __restrict__ receiver_slot,
    const long long* __restrict__ row_ptr_new, unsigned char* bins_new, double* mass_new, double* pos_new)
{
    const long long n_rows = n_active * (long long)nc;
    const long long stride = (long long)gridDim.x * (long long)blockDim.x;
    for (long long row = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x; row < n_rows;
         row += stride) {
        const long long start = row_ptr_new[row];
        if (row_ptr_new[row + 1] == start) continue;   /* the count pass found no candidate here */
        const long long si = row / nc;
        const int c = (int)(row - si * nc);
        int cand_b[MS_CAP];
        double cand_x[MS_CAP];
        double cand_m[MS_CAP];
        double mass_b[MS_NB], xmax_b[MS_NB], xmin_b[MS_NB], num_b[MS_NB], smax_b[MS_NB];
        unsigned int occ[MS_OCC_WORDS];
        for (int w = 0; w < MS_OCC_WORDS; ++w) occ[w] = 0u;
        double in_sum, ad, ast;
        const int cnt = ms_row_candidates<true>(si, c, row_ptr, mass, pos, P, V, R, S, cell_of_slot, slot_of_cell,
                                                receiver_slot, ny, nx, nc, dx, dt, cand_b, cand_x, cand_m, occ,
                                                &in_sum, &ad, &ast);
        for (int k = 0; k < cnt; ++k) {
            const int b = cand_b[k];
            mass_b[b] = 0.0; xmax_b[b] = MS_BELOW_ANY_POSITION; xmin_b[b] = __dmul_rn(2.0, dx);
            num_b[b] = 0.0; smax_b[b] = MS_BELOW_ANY_LOG_WEIGHT;
            occ[b >> 5] |= (1u << (b & 31));
        }
        /* Sweep 1: masses and extrema (dense pass-1 order). */
        for (int k = 0; k < cnt; ++k) {
            const int b = cand_b[k];
            const double x = cand_x[k];
            mass_b[b] = __dadd_rn(mass_b[b], cand_m[k]);
            if (x > xmax_b[b]) xmax_b[b] = x;    /* Python max(xmax, x) */
            if (x < xmin_b[b]) xmin_b[b] = x;    /* Python min(xmin, x) */
        }
        const double r = R[row];
        /* Sweep 2: narrow numerators or wide log-weight maxima; r = 0 moments. */
        for (int k = 0; k < cnt; ++k) {
            const int b = cand_b[k];
            const double m = cand_m[k];
            const double x = cand_x[k];
            if (r > 0.0) {
                if (__dmul_rn(r, __dadd_rn(xmax_b[b], -xmin_b[b])) <= narrow_spread) {
                    if (xmax_b[b] != xmin_b[b]) {
                        num_b[b] = __dadd_rn(num_b[b], __dmul_rn(m, expm1(__dmul_rn(r, __dadd_rn(x, -xmax_b[b])))));
                    }
                } else {
                    const double lw = __dadd_rn(log(m), __dmul_rn(r, x));
                    if (lw > smax_b[b]) smax_b[b] = lw;
                }
            } else {
                num_b[b] = __dadd_rn(num_b[b], __dmul_rn(m, x));
            }
        }
        /* Sweep 3: wide numerators once the maxima are known. */
        if (r > 0.0) {
            for (int k = 0; k < cnt; ++k) {
                const int b = cand_b[k];
                if (__dmul_rn(r, __dadd_rn(xmax_b[b], -xmin_b[b])) > narrow_spread) {
                    const double lw = __dadd_rn(log(cand_m[k]), __dmul_rn(r, cand_x[k]));
                    num_b[b] = __dadd_rn(num_b[b], exp(__dadd_rn(lw, -smax_b[b])));
                }
            }
        }
        /* Representatives with round-off containment, ascending bin order. */
        long long out = start;
        for (int b = 0; b < MS_NB; ++b) {
            if ((occ[b >> 5] & (1u << (b & 31))) == 0u) continue;
            const double mw = mass_b[b];
            double xr;
            if (r > 0.0) {
                if (__dmul_rn(r, __dadd_rn(xmax_b[b], -xmin_b[b])) <= narrow_spread) {
                    xr = __dadd_rn(xmax_b[b], __ddiv_rn(log1p(__ddiv_rn(num_b[b], mw)), r));
                } else {
                    xr = __ddiv_rn(__dadd_rn(__dadd_rn(smax_b[b], log(num_b[b])), -log(mw)), r);
                }
            } else {
                xr = __ddiv_rn(num_b[b], mw);
            }
            const double lo = xmin_b[b];
            const double hi = xmax_b[b];
            if (lo > xr) xr = lo;   /* Python max(xr, lo) */
            if (hi < xr) xr = hi;   /* Python min(xr, hi) */
            bins_new[out] = (unsigned char)b;
            mass_new[out] = mw;
            pos_new[out] = xr;
            ++out;
        }
    }
}

/* NumPy pairwise_sum over the B-length buffer (n <= 128): sequential below 8, else eight accumulators,
   identity start 0.0 + result (characteristic_numba_csr._pairwise_sum). */
__device__ __forceinline__ double ms_pairwise(const double* buf)
{
#if MS_NB < 8
    double res = 0.0;
    for (int i = 0; i < MS_NB; ++i) res = __dadd_rn(res, buf[i]);
    return res;
#else
    double r0 = buf[0], r1 = buf[1], r2 = buf[2], r3 = buf[3], r4 = buf[4], r5 = buf[5], r6 = buf[6], r7 = buf[7];
    int i = 8;
    const int stop = MS_NB - (MS_NB % 8);
    for (; i < stop; i += 8) {
        r0 = __dadd_rn(r0, buf[i]);     r1 = __dadd_rn(r1, buf[i + 1]);
        r2 = __dadd_rn(r2, buf[i + 2]); r3 = __dadd_rn(r3, buf[i + 3]);
        r4 = __dadd_rn(r4, buf[i + 4]); r5 = __dadd_rn(r5, buf[i + 5]);
        r6 = __dadd_rn(r6, buf[i + 6]); r7 = __dadd_rn(r7, buf[i + 7]);
    }
    double res = __dadd_rn(__dadd_rn(__dadd_rn(r0, r1), __dadd_rn(r2, r3)), __dadd_rn(__dadd_rn(r4, r5), __dadd_rn(r6, r7)));
    for (; i < MS_NB; ++i) res = __dadd_rn(res, buf[i]);
    return __dadd_rn(0.0, res);
#endif
}

extern "C" __global__ void ms_csr_rowsum_normalize(
    const long long n_rows, const long long* __restrict__ row_ptr, const unsigned char* __restrict__ bins,
    const double* __restrict__ mass, double* remaining, double* fraction)
{
    const long long stride = (long long)gridDim.x * (long long)blockDim.x;
    for (long long row = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x; row < n_rows;
         row += stride) {
        const long long e0 = row_ptr[row];
        const long long e1 = row_ptr[row + 1];
        if (e1 == e0) { remaining[row] = 0.0; continue; }
        double buf[MS_NB];
        for (int b = 0; b < MS_NB; ++b) buf[b] = 0.0;
        for (long long e = e0; e < e1; ++e) buf[bins[e]] = mass[e];
        const double res = ms_pairwise(buf);
        remaining[row] = res;
        for (long long e = e0; e < e1; ++e) fraction[e] = __ddiv_rn(mass[e], res);
    }
}
"""


def _strict_bins(n_bins: Any) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or not (1 <= int(n_bins) <= MAX_N_BINS):
        raise TransportError(f"n_bins must be an int in [1, {MAX_N_BINS}], got {n_bins!r}")
    return int(n_bins)


def kernel_source(n_bins: int) -> str:
    """CUDA C++ source specialised for `n_bins` (`MS_NB`), host-only text."""
    return _SOURCE_TEMPLATE.replace("@NB@", str(_strict_bins(n_bins)))


def estimate_local_scratch_bytes(n_bins: int) -> dict[str, int]:
    """Analytic per-thread local scratch of each kernel (candidate slots `5 (B + 1)` x (int + 2 doubles), five
    `B` double arrays, the occupancy words, the row-sum buffer). An ESTIMATE; the compiled `local_size_bytes`
    attribute is the measurement and is reported beside it."""
    nb = _strict_bins(n_bins)
    cap = 5 * (nb + 1)
    occ = 4 * ((nb + 31) // 32)
    return {
        "ms_csr_count": occ + 4 * 8,
        "ms_csr_write": cap * (4 + 8 + 8) + 5 * nb * 8 + occ,
        "ms_csr_rowsum_normalize": nb * 8,
    }


def launch_shape(n_rows: int, *, max_launched_threads: int = DEFAULT_MAX_LAUNCHED_THREADS,
                 block_threads: int = BLOCK_THREADS) -> tuple[int, int, int]:
    """`(grid_blocks, block_threads, launched_threads)` of the clamped persistent grid-stride launch: never more
    than `max_launched_threads` (rounded down to whole blocks, at least one block), never more blocks than rows need."""
    if isinstance(block_threads, bool) or not isinstance(block_threads, (int, np.integer)) \
            or not (32 <= int(block_threads) <= 1024) or int(block_threads) % 32:
        raise TransportError(f"block_threads must be a multiple of 32 in [32, 1024], got {block_threads!r}")
    if isinstance(max_launched_threads, bool) or not isinstance(max_launched_threads, (int, np.integer)) \
            or int(max_launched_threads) < int(block_threads):
        raise TransportError(f"max_launched_threads must be an int >= block_threads ({block_threads}), got "
                             f"{max_launched_threads!r}")
    block = int(block_threads)
    cap_blocks = max(1, int(max_launched_threads) // block)
    need_blocks = max(1, (int(n_rows) + block - 1) // block)
    grid = min(cap_blocks, need_blocks)
    return grid, block, grid * block


def cuda_sparse_preflight(shape: tuple[int, int], n_active: int, n_classes: int, n_bins: int, *,
                          max_nnz: int | None = None,
                          max_launched_threads: int = DEFAULT_MAX_LAUNCHED_THREADS,
                          block_threads: int = BLOCK_THREADS,
                          unbounded_nnz_bytes: int = DEFAULT_UNBOUNDED_NNZ_BYTES) -> dict[str, Any]:
    """Host-only byte estimates of one CUDA sparse step (no device work): CSR pointers/counts, bookkeeping,
    compact inputs, `(n, nc)` MAPLE-contract fields, faces, the two CSR generations at `max_nnz` (or at full
    occupancy when unbounded), the launched-thread local scratch ESTIMATE, and whether an explicit `max_nnz`
    is required for this case. Compiled attributes are not available here (`load_kernels` reports them)."""
    ny, nx = int(shape[0]), int(shape[1])
    n = ny * nx
    na = int(n_active)
    nc = int(n_classes)
    nb = _strict_bins(n_bins)
    if na < 1 or nc < 1 or ny < 1 or nx < 1:
        raise TransportError("preflight needs ny, nx, n_active, n_classes >= 1")
    n_rows = na * nc
    full_entries = n_rows * nb
    cap = None if max_nnz is None else int(max_nnz)
    entries = full_entries if cap is None else min(cap, full_entries)
    grid, block, threads = launch_shape(n_rows, max_launched_threads=max_launched_threads, block_threads=block_threads)
    local = estimate_local_scratch_bytes(nb)
    requires_cap = full_entries * _ENTRY_BYTES > int(unbounded_nnz_bytes)
    return {
        "n_cells": n, "n_active": na, "n_classes": nc, "n_bins": nb, "n_rows": n_rows,
        "row_ptr_bytes": 2 * (n_rows + 1) * 8,  # old + new generation
        "counts_bytes": n_rows * 8,
        "compact_input_bytes": 4 * n_rows * 8 + n_rows,  # M, P, V, R compact + settle mask
        "bookkeeping_bytes": (7 + 6) * n_rows * 8,  # per-substep returns + accumulators
        "remaining_bytes": n_rows * 8,
        "full_field_bytes": 12 * n * nc * 8,  # T, deposition, export, in/out, divergence, balance, scale, ... (estimate)
        "face_bytes": 2 * (ny * (nx + 1) + (ny + 1) * nx) * nc * 8,
        "csr_entry_bytes_per_generation": entries * _ENTRY_BYTES,
        "csr_entry_bytes_two_generations": 2 * entries * _ENTRY_BYTES,
        "csr_entries_sized_for": "max_nnz" if cap is not None else "full occupancy (unbounded)",
        "full_occupancy_entry_bytes": full_entries * _ENTRY_BYTES,
        "requires_explicit_max_nnz": requires_cap,
        "unbounded_nnz_bytes_threshold": int(unbounded_nnz_bytes),
        "launch": {"grid_blocks": grid, "block_threads": block, "launched_threads": threads},
        "local_scratch_estimate_bytes_per_thread": local,
        "local_scratch_estimate_bytes_launched": {k: v * threads for k, v in local.items()},
        "dense_equivalent_bytes": 2 * n * nc * nb * 8,
    }


# --- CuPy / kernels -----------------------------------------------------------------------------------------
_MODULES: dict[tuple[int, int], Any] = {}
_ATTRIBUTES: dict[tuple[int, int], dict[str, dict[str, int]]] = {}
_ROW_SUM_CHECKS: dict[tuple[int, int], dict[str, Any]] = {}
_LOCK = threading.Lock()


def _cupy():
    try:
        import cupy
    except Exception as exc:  # any import-time failure means "unavailable"
        raise CudaUnavailableError(
            f"implementation 'cuda' requested but CuPy is not importable ({type(exc).__name__}: {exc}); there is no "
            "fallback to the Numba implementation") from exc
    return cupy


def _current_device_id(cp) -> int:
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise CudaUnavailableError("implementation 'cuda' requested but no CUDA device is visible; no fallback")
        return int(cp.cuda.Device().id)
    except CudaUnavailableError:
        raise
    except Exception as exc:
        raise CudaUnavailableError(f"CUDA runtime/device query failed ({type(exc).__name__}: {exc}); no fallback") from exc


def load_kernels(n_bins: int) -> dict[str, dict[str, int]]:
    """Compile/load the three kernels specialised for `n_bins` on the CURRENT device (idempotent per device and
    bin count; startup work) and return the compiled attributes per kernel (`local_size_bytes`, `num_regs`,
    `shared_size_bytes`, `max_threads_per_block`, ...). A compile/load failure is `CudaUnavailableError`."""
    cp = _cupy()
    device_id = _current_device_id(cp)
    nb = _strict_bins(n_bins)
    key = (device_id, nb)
    cached = _ATTRIBUTES.get(key)
    if cached is not None:
        return dict(cached)
    with _LOCK:
        try:
            module = _MODULES.get(key)
            if module is None:
                module = cp.RawModule(code=kernel_source(nb), options=COMPILE_OPTIONS, backend="nvrtc")
                _MODULES[key] = module
            attributes = {}
            for name in KERNEL_NAMES:
                fn = module.get_function(name)
                attrs = dict(fn.attributes)  # forces compile + load for this device now
                if int(attrs.get("max_threads_per_block", 0)) < BLOCK_THREADS:
                    raise CudaUnavailableError(
                        f"kernel {name} (B = {nb}) supports only {attrs.get('max_threads_per_block')} threads per "
                        f"block, {BLOCK_THREADS} are needed; no fallback")
                attributes[name] = {k: int(v) for k, v in attrs.items() if isinstance(v, (int, np.integer))}
        except CudaUnavailableError:
            raise
        except Exception as exc:
            raise CudaUnavailableError(
                f"CUDA sparse CSR kernels (B = {nb}) failed to compile/load ({type(exc).__name__}: {exc}); "
                "no fallback") from exc
        _ATTRIBUTES[key] = attributes
    return dict(attributes)


def _function(name: str, n_bins: int):
    cp = _cupy()
    key = (_current_device_id(cp), int(n_bins))
    if key not in _ATTRIBUTES:
        load_kernels(n_bins)
    return _MODULES[key].get_function(name)


def kernel_provenance(n_bins: int = 32) -> dict[str, Any]:
    """Source hash for `n_bins`, compile options, launch policy and toolchain/device (no compilation)."""
    info: dict[str, Any] = {
        "kernels": list(KERNEL_NAMES), "n_bins": _strict_bins(n_bins),
        "source_sha256": hashlib.sha256(kernel_source(n_bins).encode()).hexdigest(),
        "template_sha256": hashlib.sha256(_SOURCE_TEMPLATE.encode()).hexdigest(),
        "compile_options": list(COMPILE_OPTIONS), "fastmath": False, "fma": False, "float_atomics": False,
        "intrinsics": ["__dadd_rn", "__dmul_rn", "__ddiv_rn"],
        "transcendentals": "libdevice exp/expm1/log/log1p (precise double; not bitwise with host libm)",
        "launch_structure": "one launch per pass, clamped persistent grid-stride over rows "
                            f"(<= {DEFAULT_MAX_LAUNCHED_THREADS} threads by default, {BLOCK_THREADS} per block)",
        "local_scratch_estimate_bytes_per_thread": estimate_local_scratch_bytes(n_bins),
        "cupy": None, "cuda_runtime": None, "device": None,
    }
    try:
        import cupy as cp

        info["cupy"] = cp.__version__
        info["cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
        if cp.cuda.runtime.getDeviceCount() > 0:
            dev = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(dev.id)
            name = props["name"]
            info["device"] = {"visible_id": int(dev.id), "name": name.decode() if isinstance(name, bytes) else str(name),
                              "compute_capability": str(dev.compute_capability),
                              "total_memory_bytes": int(props["totalGlobalMem"])}
    except (ImportError, OSError, RuntimeError, AttributeError, ValueError) as exc:
        info["provenance_incomplete"] = True
        info["provenance_error"] = f"{type(exc).__name__}: {exc}"
    return info


# --- device row-sum qualification -----------------------------------------------------------------------------
def row_sum_device_self_check(n_bins: int, *, force: bool = False) -> dict[str, Any]:
    """Qualify the device row-sum kernel for `n_bins` against the running NumPy ONCE per device and bin count:
    the small deterministic fixture of `characteristic_sparse._self_check_rows` (96 rows) is uploaded (counted),
    reduced on the device, downloaded (96 doubles, counted) and compared RAW BYTES with the bounded NumPy
    reference `row_sums_numpy`. A mismatch is a `TransportError` refusal (no fallback). Returns the record."""
    from maple.core.backend import to_device, to_host

    from maple_syrup.characteristic_sparse import (
        RowSumEmulationMismatch,
        _self_check_rows,
        row_sums_numpy,
    )

    cp = _cupy()
    nb = _strict_bins(n_bins)
    key = (_current_device_id(cp), nb)
    cached = _ROW_SUM_CHECKS.get(key)
    if cached is not None and not force:
        return dict(cached)
    rng = np.random.default_rng(20261008 + nb)
    _dense, row_ptr, bins, values = _self_check_rows(nb, rng)
    reference = row_sums_numpy(row_ptr, bins, values, nb)
    n_rows = int(row_ptr.shape[0]) - 1
    d_row_ptr = to_device(np.ascontiguousarray(row_ptr), cp)
    d_bins = to_device(np.ascontiguousarray(bins), cp)
    d_values = to_device(np.ascontiguousarray(values), cp)
    remaining = cp.empty(n_rows, dtype=np.float64)
    fraction = cp.empty(int(values.shape[0]), dtype=np.float64)
    grid, block, _threads = launch_shape(n_rows)
    fn = _function("ms_csr_rowsum_normalize", nb)
    try:
        fn((grid,), (block,), (np.int64(n_rows), d_row_ptr, d_bins, d_values, remaining, fraction))
    except Exception as exc:
        raise CudaUnavailableError(f"row-sum kernel launch failed ({type(exc).__name__}: {exc}); no fallback") from exc
    got = to_host(remaining)
    if got.tobytes() != reference.tobytes():
        raise RowSumEmulationMismatch(np.__version__, nb, "device pairwise emulation differs from the bounded NumPy "
                                                           "reference (raw bytes)")
    record = {"numpy": str(np.__version__), "cupy": str(cp.__version__), "device_id": key[0], "n_bins": nb,
              "rows": n_rows, "byte_exact": True}
    _ROW_SUM_CHECKS[key] = record
    return dict(record)


# --- network binding ------------------------------------------------------------------------------------------
def _fingerprint(array: Any) -> tuple:
    return (int(array.data.ptr), tuple(array.shape), str(array.dtype), bool(array.flags.c_contiguous),
            int(array.device.id))


_NETWORK_RUNTIME_FIELDS = ("receiver_index", "outlet_flat", "active_flat", "cells_x", "rows_x", "faces_x", "sign_x",
                           "cells_y", "faces_y", "cols_y", "sign_y")


@dataclass(frozen=True, eq=False)
class CudaCsrBinding:
    """Owned device slot maps of one CuPy network (built once from host evidence). Holds no reference to the
    network; cached by network identity. Do not mutate the arrays. `receiver_cell` is the flat receiver cell of
    every slot (itself at outlets/terminals) used by the runtime consistency check; `terminal` is the compact
    terminal mask (all False for a strict network, `has_terminal` False)."""

    device_id: int
    shape: tuple[int, int]
    n_cells: int
    n_active: int
    dx_m: float
    cell_of_slot: Any  # (n_active,) int64 device
    slot_of_cell: Any  # (n,) int64 device
    receiver_slot: Any  # (n_active,) int64 device
    outlet: Any  # (n_active,) bool device
    receiver_cell: Any  # (n_active,) int64 device
    terminal: Any  # (n_active,) bool device
    has_terminal: bool
    graph_input_sha256: str
    network_fingerprints: tuple
    owned_fingerprints: tuple
    host_to_device_bytes: int
    preparation_s: float

    def summary(self) -> dict[str, Any]:
        return {"device_id": self.device_id, "shape": list(self.shape), "n_cells": self.n_cells,
                "n_active": self.n_active, "has_terminal": self.has_terminal,
                "graph_input_sha256": self.graph_input_sha256,
                "slot_map_bytes": int(self.cell_of_slot.nbytes + self.slot_of_cell.nbytes + self.receiver_slot.nbytes
                                      + self.outlet.nbytes + self.receiver_cell.nbytes + self.terminal.nbytes),
                "host_to_device_bytes": self.host_to_device_bytes, "preparation_s": self.preparation_s,
                "mutation_detection": "metadata fingerprints (replacement/device/layout) + per-step device "
                                      "consistency flags for receiver_index/active_flat/outlet_flat/terminal_flat"}

    def owned_arrays(self) -> tuple:
        return (self.cell_of_slot, self.slot_of_cell, self.receiver_slot, self.outlet, self.receiver_cell, self.terminal)


def _runtime_terminal(cp, network, device_id: int, has_terminal: bool):
    """The network's optional device `terminal_flat` (exact cupy bool `(n,)`), required iff the host side declares
    terminals; None for a strict network."""
    terminal_flat = getattr(network, "terminal_flat", None)
    if has_terminal != (terminal_flat is not None):
        raise TransportError("network.terminal_host and network.terminal_flat must both be present or both absent")
    if terminal_flat is not None:
        _check_device_array(cp, terminal_flat, "network.terminal_flat", np.bool_, (int(network.n_cells),), device_id)
    return terminal_flat


def enqueue_runtime_consistency(checks, network, binding: CudaCsrBinding) -> None:
    """Flags (one batched read by the caller) that the runtime connectivity arrays still agree with the owned maps:
    `active_flat == (slot_of_cell >= 0)`, `outlet_flat[cell_of_slot] == outlet`, `receiver_index[cell_of_slot] ==
    receiver_cell`, and for terminal networks `terminal_flat[cell_of_slot] == terminal`. Gathers over `n_active`
    only; no physical download. Catches in-place value mutation that metadata fingerprints cannot."""
    from maple.core.backend import true_flag

    cp = _cupy()
    checks.forbid(true_flag(network.active_flat != (binding.slot_of_cell >= 0)),
                  "network.active_flat changed in place since binding (runtime arrays must not be mutated)")
    checks.forbid(true_flag(network.outlet_flat[binding.cell_of_slot] != binding.outlet),
                  "network.outlet_flat changed in place since binding (runtime arrays must not be mutated)")
    checks.forbid(true_flag(network.receiver_index[binding.cell_of_slot] != binding.receiver_cell),
                  "network.receiver_index disagrees with the bound connectivity (in-place mutation or host/device "
                  "evidence mismatch); refusing before any launch")
    terminal_flat = _runtime_terminal(cp, network, binding.device_id, binding.has_terminal)
    if terminal_flat is not None:
        checks.forbid(true_flag(terminal_flat[binding.cell_of_slot] != binding.terminal),
                      "network.terminal_flat disagrees with the bound terminal evidence")
        checks.forbid(true_flag(terminal_flat & ~network.active_flat), "a terminal-storage cell must be active")


_BINDINGS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_BINDING_LOCK = threading.Lock()


def _check_device_array(cp, array: Any, name: str, dtype, shape: tuple[int, ...] | None, device_id: int) -> None:
    if type(array) is not cp.ndarray:
        raise TransportError(f"{name} must be an exact cupy.ndarray for implementation 'cuda' (no transfer or "
                             f"conversion is made), got {type(array).__name__}")
    if array.dtype != dtype:
        raise TransportError(f"{name} must have dtype {np.dtype(dtype)}, got {array.dtype}")
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise TransportError(f"{name} shape {tuple(array.shape)} != {tuple(shape)}")
    if not array.flags.c_contiguous:
        raise TransportError(f"{name} must be C-contiguous")
    if int(array.device.id) != device_id:
        raise TransportError(f"{name} lives on CUDA device {int(array.device.id)} but the current device is "
                             f"{device_id}; no peer access or migration is performed")


def _require_cupy_network(cp, network: Any, device_id: int) -> None:
    if not isinstance(network, TransportNetwork):
        raise TransportError("network must be a TransportNetwork (use transport_network)")
    if network.xp is not cp:
        raise TransportError(f"implementation 'cuda' needs a network built with xp=cupy; the network lives in "
                             f"{getattr(network.xp, '__name__', network.xp)!r} and no transfer is performed")
    n = int(network.n_cells)
    _check_device_array(cp, network.receiver_index, "network.receiver_index", np.int64, (n,), device_id)
    _check_device_array(cp, network.outlet_flat, "network.outlet_flat", np.bool_, (n,), device_id)
    _check_device_array(cp, network.active_flat, "network.active_flat", np.bool_, (n,), device_id)
    for name in _NETWORK_RUNTIME_FIELDS[3:]:
        array = getattr(network, name)
        if type(array) is not cp.ndarray or int(array.device.id) != device_id:
            raise TransportError(f"network.{name} must be a cupy.ndarray on the current device {device_id}")


def _network_fingerprints(network) -> tuple:
    return tuple(_fingerprint(getattr(network, name)) for name in _NETWORK_RUNTIME_FIELDS)


def bind_network(network: TransportNetwork) -> CudaCsrBinding:
    """Validate a CuPy network for the CURRENT device and return its (cached) owned slot-map binding. The first
    call derives the maps from the validated HOST evidence (`characteristic_sparse.host_network_evidence`, with
    optional terminal storage), uploads them once through counted `to_device`, synchronises the current stream
    once and then resolves ONE device flag packet proving the runtime arrays agree with that evidence
    (`enqueue_runtime_consistency`). A cache hit compares metadata fingerprints only (replacement / device /
    layout); in-place value changes are caught by the same consistency flags, which every step re-enqueues."""
    from maple.core.backend import DeferredChecks, read_transfer_counters, to_device

    from maple_syrup.characteristic_sparse import (
        host_network_evidence,
        slot_maps_from_host_evidence,
    )

    cp = _cupy()
    device_id = _current_device_id(cp)
    _require_cupy_network(cp, network, device_id)
    with _BINDING_LOCK:
        binding = _BINDINGS.get(network)
    if binding is not None:
        if binding.device_id != device_id:
            raise TransportError(f"the network was bound on CUDA device {binding.device_id} but the current device is "
                                 f"{device_id}; release_binding(network) and bind again on purpose")
        if binding.network_fingerprints != _network_fingerprints(network):
            raise TransportError("the network's runtime arrays were replaced or moved after binding (networks must "
                                 "not be mutated or have fields replaced); release_binding(network) and bind again")
        if binding.owned_fingerprints != tuple(_fingerprint(a) for a in binding.owned_arrays()):
            raise TransportError("the owned slot-map device arrays no longer match the binding metadata")
        return binding
    t0 = time.perf_counter()
    ny, nx = int(network.shape[0]), int(network.shape[1])
    n = ny * nx
    evidence = host_network_evidence(network)
    has_terminal = evidence.terminal is not None
    _runtime_terminal(cp, network, device_id, has_terminal)  # both fields or neither, exact device array
    maps = slot_maps_from_host_evidence(evidence.active, evidence.receiver_index, evidence.outlet)
    if maps.cell_of_slot.size != int(network.n_active):
        raise TransportError("network.n_active disagrees with its active_host mask")
    receiver_cell = maps.cell_of_slot[maps.receiver_slot]
    terminal_c = (np.zeros(maps.cell_of_slot.size, dtype=np.bool_) if evidence.terminal is None
                  else evidence.terminal[maps.cell_of_slot])
    before = read_transfer_counters()
    owned = tuple(to_device(np.ascontiguousarray(a), cp) for a in (
        maps.cell_of_slot, maps.slot_of_cell, maps.receiver_slot, maps.outlet, receiver_cell, terminal_c))
    cp.cuda.get_current_stream().synchronize()
    delta = read_transfer_counters().delta(before)
    binding = CudaCsrBinding(
        device_id=device_id, shape=(ny, nx), n_cells=n, n_active=int(maps.cell_of_slot.size), dx_m=float(network.dx_m),
        cell_of_slot=owned[0], slot_of_cell=owned[1], receiver_slot=owned[2], outlet=owned[3], receiver_cell=owned[4],
        terminal=owned[5], has_terminal=has_terminal,
        graph_input_sha256=str(network.graph_input_sha256), network_fingerprints=_network_fingerprints(network),
        owned_fingerprints=tuple(_fingerprint(a) for a in owned),
        host_to_device_bytes=int(delta.host_to_device_bytes), preparation_s=time.perf_counter() - t0)
    # Initial consistency packet: the device arrays the operator relies on must agree with the host evidence
    # the maps were built from (one read; nothing physical is downloaded). Refused before caching.
    checks = DeferredChecks()
    enqueue_runtime_consistency(checks, network, binding)
    try:
        checks.resolve()
    except ValueError as exc:
        raise TransportError(f"binding refused: {exc}") from None
    with _BINDING_LOCK:
        _BINDINGS[network] = binding
    return binding


def release_binding(network: TransportNetwork) -> bool:
    """Drop the cached binding of `network` (frees the owned device maps once unreferenced). True if one existed."""
    with _BINDING_LOCK:
        return _BINDINGS.pop(network, None) is not None


# --- state helpers ----------------------------------------------------------------------------------------------
def _state_class():
    from maple_syrup.characteristic_sparse import SparsePhaseState

    return SparsePhaseState


def _row_of_entry(cp, row_ptr, nnz: int):
    """Row id of every entry without a host read: `searchsorted(row_ptr, arange(nnz), side="right") - 1`."""
    if nnz == 0:
        return cp.zeros(0, dtype=np.int64)
    return cp.searchsorted(row_ptr, cp.arange(nnz, dtype=np.int64), side="right") - 1


def empty_cuda_sparse_phase_state(network: TransportNetwork, n_classes: int, n_bins: int):
    """Canonical empty CuPy state (`nnz = 0`) bound to the network's owned device slot maps."""
    binding = bind_network(network)
    cp = _cupy()
    if isinstance(n_classes, bool) or not isinstance(n_classes, (int, np.integer)) or not (1 <= int(n_classes) <= 1 << 30):
        raise TransportError(f"n_classes must be an int in [1, {1 << 30}], got {n_classes!r}")
    nc = int(n_classes)
    nb = _strict_bins(n_bins)
    n_rows = binding.n_active * nc
    return _state_class()(
        row_ptr=cp.zeros(n_rows + 1, dtype=np.int64), bin=cp.zeros(0, dtype=np.uint8),
        fraction=cp.zeros(0, dtype=np.float64), position_m=cp.zeros(0, dtype=np.float64),
        dx_m=binding.dx_m, n_bins=nb, n_classes=nc, shape=binding.shape,
        cell_of_slot=binding.cell_of_slot, slot_of_cell=binding.slot_of_cell,
        network_sha256=binding.graph_input_sha256, xp=cp)


def _check_state_structure(cp, phase: Any, binding: CudaCsrBinding, n_classes: int) -> int:
    """Host-metadata refusals; returns nnz (from the entry arrays' shapes, no device read)."""
    if not isinstance(phase, _state_class()):
        raise TransportError("phase must be a SparsePhaseState (use empty_sparse_phase_state)")
    if phase.xp is not cp:
        raise TransportError(f"sparse phase namespace {getattr(phase.xp, '__name__', phase.xp)!r} is not 'cupy'; "
                             "no transfer is performed")
    _strict_bins(phase.n_bins)
    if isinstance(phase.n_classes, bool) or not isinstance(phase.n_classes, (int, np.integer)) \
            or int(phase.n_classes) != n_classes:
        raise TransportError(f"phase n_classes {phase.n_classes!r} differs from the mobile field's {n_classes}")
    if not isinstance(phase.dx_m, (int, float, np.floating)) or float(phase.dx_m) != binding.dx_m:
        raise TransportError(f"phase dx_m {phase.dx_m!r} differs from the network dx_m {binding.dx_m}")
    if tuple(phase.shape) != tuple(binding.shape):
        raise TransportError(f"phase shape {tuple(phase.shape)} differs from the network shape {binding.shape}")
    if phase.network_sha256 != binding.graph_input_sha256:
        raise TransportError("phase state was built for a different routing graph (graph digest mismatch); use "
                             "rebind_sparse_phase_state for an explicit rebind")
    n_rows = binding.n_active * n_classes
    _check_device_array(cp, phase.cell_of_slot, "phase.cell_of_slot", np.int64, (binding.n_active,), binding.device_id)
    _check_device_array(cp, phase.slot_of_cell, "phase.slot_of_cell", np.int64, (binding.n_cells,), binding.device_id)
    _check_device_array(cp, phase.row_ptr, "phase.row_ptr", np.int64, (n_rows + 1,), binding.device_id)
    nnz = int(phase.fraction.shape[0]) if type(phase.fraction) is cp.ndarray and phase.fraction.ndim == 1 else -1
    if nnz < 0:
        raise TransportError("phase.fraction must be a 1-D cupy.ndarray")
    _check_device_array(cp, phase.bin, "phase.bin", np.uint8, (nnz,), binding.device_id)
    _check_device_array(cp, phase.fraction, "phase.fraction", np.float64, (nnz,), binding.device_id)
    _check_device_array(cp, phase.position_m, "phase.position_m", np.float64, (nnz,), binding.device_id)
    return nnz


def _enqueue_state_values(cp, checks, phase: Any, binding: CudaCsrBinding, mobile_compact: Any, nnz: int) -> None:
    """Value invariants of a structurally valid device state as unread flags (one batched read by the caller)."""
    from maple.core.backend import finite_flag, negative_flag, scatter_add, true_flag

    nb = int(phase.n_bins)
    dx = float(phase.dx_m)
    row_ptr = phase.row_ptr
    n_rows = int(row_ptr.shape[0]) - 1
    if phase.cell_of_slot is not binding.cell_of_slot:
        checks.forbid(true_flag(phase.cell_of_slot != binding.cell_of_slot), "phase slot maps differ from the network's")
    if phase.slot_of_cell is not binding.slot_of_cell:
        checks.forbid(true_flag(phase.slot_of_cell != binding.slot_of_cell), "phase slot maps differ from the network's")
    checks.forbid(true_flag(row_ptr[0] != 0), "phase.row_ptr[0] must be 0")
    counts = cp.diff(row_ptr)
    checks.forbid(negative_flag(counts), "phase.row_ptr must be non-decreasing")
    checks.forbid(true_flag(row_ptr[-1] != nnz), "phase.row_ptr[-1] must equal the number of stored bins")
    occupied_rows = counts > 0
    positive = mobile_compact.reshape(-1) > 0.0
    checks.forbid(true_flag(occupied_rows & ~positive),
                  "phase state stores bins on a row whose mobile total is 0 (canonical empty rows store nothing)")
    checks.forbid(true_flag(positive & ~occupied_rows), "phase state has no bins on a row whose mobile total is positive")
    if nnz == 0:
        return
    bins = phase.bin
    f = phase.fraction
    x = phase.position_m
    rows = _row_of_entry(cp, row_ptr, nnz)
    checks.forbid(true_flag(bins >= nb), f"phase.bin must be < n_bins = {nb}")
    if nnz > 1:
        same_row = rows[1:] == rows[:-1]
        checks.forbid(true_flag(same_row & ~(bins[1:] > bins[:-1])), "phase.bin must be strictly increasing within each row")
    checks.require(finite_flag(f), "phase.fraction must be finite everywhere")
    checks.require(finite_flag(x), "phase.position_m must be finite everywhere")
    checks.forbid(true_flag(f <= 0.0), "phase.fraction must be > 0 on every stored bin (empty bins are not stored)")
    checks.forbid(negative_flag(x), "phase.position_m must be >= 0 everywhere")
    checks.forbid(true_flag(x >= dx), "phase.position_m must be < dx everywhere")
    checks.forbid(true_flag(bin_index(x, dx, nb, cp) != bins.astype(np.int64)), "phase.position_m must lie inside its own bin")
    sums = cp.zeros(n_rows, dtype=np.float64)
    scatter_add(sums, rows, f)
    checks.forbid(true_flag(occupied_rows & (cp.abs(sums - 1.0) > fraction_sum_tolerance(nb))),
                  "phase.fraction must sum to 1 within the declared bound on every nonempty row")


def _check_mobile_structure(cp, mobile_kg: Any, binding: CudaCsrBinding) -> int:
    if type(mobile_kg) is not cp.ndarray or mobile_kg.ndim != 3 or mobile_kg.shape[-1] < 1:
        raise TransportError("mobile_kg must be a (ny, nx, nc) cupy.ndarray for implementation 'cuda'")
    nc = int(mobile_kg.shape[-1])
    _check_device_array(cp, mobile_kg, "mobile_kg", np.float64, (*binding.shape, nc), binding.device_id)
    return nc


def validate_cuda_sparse_phase_state(phase: Any, mobile_kg: Any, network: Any) -> None:
    """Device counterpart of `validate_sparse_phase_state`: structural refusals on host metadata, value
    invariants as flags resolved in ONE read; physical arrays are never downloaded."""
    from maple.core.backend import DeferredChecks, finite_flag, negative_flag, true_flag

    cp = _cupy()
    binding = bind_network(network)
    nc = _check_mobile_structure(cp, mobile_kg, binding)
    nnz = _check_state_structure(cp, phase, binding, nc)
    checks = DeferredChecks()
    enqueue_runtime_consistency(checks, network, binding)
    checks.require(finite_flag(mobile_kg), "mobile_kg must be finite everywhere")
    checks.forbid(negative_flag(mobile_kg), "mobile_kg must be >= 0 everywhere")
    M = mobile_kg.reshape(binding.n_cells, nc)
    inactive = ~network.active_flat
    checks.forbid(true_flag(inactive[:, None] & (M != 0.0)), "mobile_kg must be 0 on inactive cells")
    _enqueue_state_values(cp, checks, phase, binding, M[binding.cell_of_slot], nnz)
    try:
        checks.resolve()
    except ValueError as exc:
        raise TransportError(str(exc)) from None


# --- small dense conversions (cap decided before any allocation or scan) ---------------------------------------
def sparse_to_dense_cuda(phase: Any, *, max_bytes: int) -> PhaseState:
    from maple_syrup.characteristic_sparse import (
        DenseConversionRefused,
        dense_phase_bytes,
    )

    cp = _cupy()
    if not isinstance(phase, _state_class()) or phase.xp is not cp:
        raise TransportError("phase must be a CuPy SparsePhaseState")
    ny, nx = int(phase.shape[0]), int(phase.shape[1])
    nc, nb = int(phase.n_classes), _strict_bins(phase.n_bins)
    required = dense_phase_bytes(phase.shape, nc, nb)
    if required > int(max_bytes):
        raise DenseConversionRefused(required, int(max_bytes))
    n = ny * nx
    nnz = int(phase.fraction.shape[0])
    fraction = cp.zeros((n, nc, nb), dtype=np.float64)
    fraction[:, :, 0] = 1.0
    position = cp.zeros((n, nc, nb), dtype=np.float64)
    counts = cp.diff(phase.row_ptr)
    nonempty = cp.flatnonzero(counts > 0)  # device-side; its size is read implicitly by the indexing below
    if nnz:
        fraction[phase.cell_of_slot[nonempty // nc], nonempty % nc, 0] = 0.0
        rows = _row_of_entry(cp, phase.row_ptr, nnz)
        cells = phase.cell_of_slot[rows // nc]
        classes = rows % nc
        bins = phase.bin.astype(np.int64)
        fraction[cells, classes, bins] = phase.fraction
        position[cells, classes, bins] = phase.position_m
    return PhaseState(fraction=fraction.reshape(ny, nx, nc, nb), position_m=position.reshape(ny, nx, nc, nb),
                      dx_m=float(phase.dx_m), n_bins=nb, xp=cp)


def sparse_from_dense_cuda(phase: PhaseState, mobile_kg: Any, network: TransportNetwork, *, max_bytes: int):
    """CSR state of a dense CuPy `PhaseState`. Cap decided from metadata BEFORE the dense validator (which
    allocates full-size masks); then the dense `validate_phase_state`, then the conversion; `nnz` read once."""
    from maple.core.backend import DeferredChecks, to_int

    from maple_syrup.characteristic_sparse import (
        DenseConversionRefused,
        dense_phase_bytes,
    )

    cp = _cupy()
    binding = bind_network(network)
    if not isinstance(phase, PhaseState):
        raise TransportError("phase must be a dense PhaseState")
    if type(mobile_kg) is not cp.ndarray or mobile_kg.ndim != 3 or mobile_kg.shape[-1] < 1:
        raise TransportError("mobile_kg must be a (ny, nx, nc) cupy.ndarray")
    nc = int(mobile_kg.shape[-1])
    nb = _strict_bins(phase.n_bins)
    required = dense_phase_bytes(binding.shape, nc, nb)
    if required > int(max_bytes):
        raise DenseConversionRefused(required, int(max_bytes))
    validate_phase_state(phase, mobile_kg, network)
    n = binding.n_cells
    F = phase.fraction.reshape(n, nc, nb)[binding.cell_of_slot]
    X = phase.position_m.reshape(n, nc, nb)[binding.cell_of_slot]
    M = mobile_kg.reshape(n, nc)[binding.cell_of_slot]
    stored = (M > 0.0)[..., None] & (F > 0.0)
    slots, classes, bins = cp.nonzero(stored)  # C order: rows ascending, bins ascending within a row (one sync)
    rows = slots * nc + classes
    n_rows = binding.n_active * nc
    counts = cp.bincount(rows, minlength=n_rows)
    row_ptr = cp.zeros(n_rows + 1, dtype=np.int64)
    cp.cumsum(counts, out=row_ptr[1:])
    state = _state_class()(
        row_ptr=row_ptr, bin=bins.astype(np.uint8), fraction=cp.ascontiguousarray(F[slots, classes, bins]),
        position_m=cp.ascontiguousarray(X[slots, classes, bins]), dx_m=binding.dx_m, n_bins=nb, n_classes=nc,
        shape=binding.shape, cell_of_slot=binding.cell_of_slot, slot_of_cell=binding.slot_of_cell,
        network_sha256=binding.graph_input_sha256, xp=cp)
    checks = DeferredChecks()
    _enqueue_state_values(cp, checks, state, binding, M, to_int(row_ptr[-1]))
    try:
        checks.resolve()
    except ValueError as exc:
        raise TransportError(str(exc)) from None
    return state


# --- step -----------------------------------------------------------------------------------------------------
def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TransportError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _strict_int(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not (low <= int(value) <= high):
        raise TransportError(f"{name} must be an int in [{low}, {high}], got {value!r}")
    return int(value)


def cuda_sparse_step(network, mobile_before_kg, phase, pickup_kg, sediment_velocity_m_s, deposition_rate_per_m,
                     settle_mask, dt_s, *, n_substeps: int = 1, courant_max: float = DEFAULT_COURANT_MAX,
                     max_nnz: int | None = None, max_launched_threads: int = DEFAULT_MAX_LAUNCHED_THREADS,
                     block_threads: int = BLOCK_THREADS, unbounded_nnz_bytes: int = DEFAULT_UNBOUNDED_NNZ_BYTES):
    """The CUDA sparse step (module docstring); called by `sparse_characteristic_step(..., implementation="cuda")`."""
    return _cuda_sparse_core(network, mobile_before_kg, phase, pickup_kg, sediment_velocity_m_s, deposition_rate_per_m,
                             settle_mask, dt_s, n_substeps=n_substeps, courant_max=courant_max, max_nnz=max_nnz,
                             max_launched_threads=max_launched_threads, block_threads=block_threads,
                             unbounded_nnz_bytes=unbounded_nnz_bytes, retain="full")


def _cuda_sparse_accounting_step(network, mobile_before_kg, phase, pickup_kg, sediment_velocity_m_s,
                                 deposition_rate_per_m, settle_mask, dt_s, *, n_substeps: int = 1,
                                 courant_max: float = DEFAULT_COURANT_MAX, max_nnz: int | None = None):
    """CW-B93 PRIVATE: the same step with private accounting retention (`characteristic_sparse._SparseAccountingStep`);
    reached only through `characteristic_sparse._sparse_characteristic_accounting_step` (default launch options)."""
    return _cuda_sparse_core(network, mobile_before_kg, phase, pickup_kg, sediment_velocity_m_s, deposition_rate_per_m,
                             settle_mask, dt_s, n_substeps=n_substeps, courant_max=courant_max, max_nnz=max_nnz,
                             max_launched_threads=DEFAULT_MAX_LAUNCHED_THREADS, block_threads=BLOCK_THREADS,
                             unbounded_nnz_bytes=DEFAULT_UNBOUNDED_NNZ_BYTES, retain="accounting")


_RETENTION = ("full", "accounting")


def _cuda_sparse_core(network, mobile_before_kg, phase, pickup_kg, sediment_velocity_m_s, deposition_rate_per_m,
                      settle_mask, dt_s, *, n_substeps, courant_max, max_nnz, max_launched_threads, block_threads,
                      unbounded_nnz_bytes, retain):
    """The ONE CUDA sparse step implementation. `retain="full"` returns the public `SparseCharacteristicStep`;
    `retain="accounting"` (CW-B93, private) runs the identical validation, kernels, expressions, flag packet order and
    batched reads, releases the return-only class grids before the four face arrays are allocated and returns
    `_SparseAccountingStep` (lifetimes and result type only; no value, order, check or kernel differs)."""
    if retain not in _RETENTION:
        raise TransportError(f"retain must be one of {_RETENTION}, got {retain!r}")
    accounting = retain == "accounting"
    from maple.core.backend import (
        DeferredChecks,
        finite_flag,
        negative_flag,
        pairwise_bound_scale_factor,
        pairwise_sum_over_leading_axes,
        read_transfer_counters,
        scatter_add,
        to_int,
        true_flag,
    )

    from maple_syrup.characteristic_sparse import (
        SparseCapacityError,
        SparseCharacteristicStep,
        _SparseAccountingStep,
        _TransportAccountingStep,
    )

    cp = _cupy()
    binding = bind_network(network)
    ny, nx = binding.shape
    n = binding.n_cells
    dt = _real(dt_s, "dt_s")
    if not (math.isfinite(dt) and dt > 0.0):
        raise TransportError(f"dt_s must be finite and > 0, got {dt_s!r}")
    cr_max = _real(courant_max, "courant_max")
    if not (0.0 < cr_max <= DEFAULT_COURANT_MAX):
        raise TransportError(f"courant_max must lie in (0, {DEFAULT_COURANT_MAX}] (one face crossing per packet per "
                             f"substep), got {courant_max!r}")
    n_sub = _strict_int(n_substeps, "n_substeps", 1, MAX_SUBSTEPS)
    cap = None if max_nnz is None else _strict_int(max_nnz, "max_nnz", 0, 1 << 62)
    nc = _check_mobile_structure(cp, mobile_before_kg, binding)
    shape3 = (ny, nx, nc)
    floats = {"mobile_before_kg": mobile_before_kg, "pickup_kg": pickup_kg,
              "sediment_velocity_m_s": sediment_velocity_m_s, "deposition_rate_per_m": deposition_rate_per_m}
    for name, array in floats.items():
        _check_device_array(cp, array, name, np.float64, shape3, binding.device_id)
    _check_device_array(cp, settle_mask, "settle_mask", np.bool_, shape3, binding.device_id)
    nnz0 = _check_state_structure(cp, phase, binding, nc)
    nb = int(phase.n_bins)
    n_active = binding.n_active
    n_rows = n_active * nc
    full_entries = n_rows * nb
    if cap is None and full_entries * _ENTRY_BYTES > int(unbounded_nnz_bytes):
        raise TransportError(
            f"implementation 'cuda' requires an explicit max_nnz for this case: full occupancy would be {full_entries} "
            f"entries ({full_entries * _ENTRY_BYTES} bytes) > unbounded_nnz_bytes = {int(unbounded_nnz_bytes)}; the "
            "GPU path never sizes a large case for every bin by default (see cuda_sparse_preflight)")
    grid, block, launched = launch_shape(n_rows, max_launched_threads=max_launched_threads, block_threads=block_threads)
    attributes = load_kernels(nb)  # compile/load before any flag read or launch
    row_sum_check = row_sum_device_self_check(nb)  # cached per device/B; refusal on mismatch
    counters_before = read_transfer_counters()

    # --- input validation: one batched flag read, before any kernel launch ---
    checks = DeferredChecks()
    enqueue_runtime_consistency(checks, network, binding)  # cached maps still agree with the runtime arrays
    for name, array in floats.items():
        checks.require(finite_flag(array), f"{name} must be finite everywhere")
        checks.forbid(negative_flag(array), f"{name} must be >= 0 everywhere")
    active = network.active_flat
    outlet = network.outlet_flat
    inactive3 = (~active)[:, None]
    M = mobile_before_kg.reshape(n, nc)
    P = pickup_kg.reshape(n, nc)
    V = sediment_velocity_m_s.reshape(n, nc)
    R = deposition_rate_per_m.reshape(n, nc)
    S = settle_mask.reshape(n, nc)
    checks.forbid(true_flag(inactive3 & (M != 0.0)), "mobile_before_kg must be 0 on inactive cells")
    checks.forbid(true_flag(inactive3 & (P != 0.0)), "pickup_kg must be 0 on inactive cells")
    checks.forbid(true_flag(inactive3 & (V != 0.0)), "sediment_velocity_m_s must be 0 on inactive cells")
    idx = binding.cell_of_slot
    Mc = cp.ascontiguousarray(M[idx])
    if binding.has_terminal:
        # Terminal rule, explicit refusal (never coerced): every class must settle at a terminal cell.
        checks.forbid(true_flag(binding.terminal[:, None] & ~S[idx]),
                      "settle_mask must be True for every grain class at terminal-storage cells (the mobile load "
                      "settles there and nothing is sent); a mask missing a class is refused, not coerced")
    _enqueue_state_values(cp, checks, phase, binding, Mc, nnz0)
    dx = binding.dx_m
    dt_sub = dt / n_sub
    a = V * (dt_sub / dx)
    max_courant = cp.max(cp.where(active[:, None], a, 0.0))
    checks.require(finite_flag(a), "characteristic sediment Courant number overflowed FP64")
    checks.forbid(true_flag(active[:, None] & (a > cr_max)),
                  f"{_COURANT_REJECTION} courant_max = {cr_max}; step rejected (retry with a smaller dt or more substeps)")
    exponent = (V * R) * dt_sub
    max_exponent = cp.max(cp.where(active[:, None], exponent, 0.0))
    checks.require(finite_flag(exponent), "deposition hazard v r dt overflowed FP64")
    try:
        checks.resolve()
    except ValueError as exc:
        message = str(exc)
        raise (TransportStepRejected if message.startswith(_COURANT_REJECTION) else TransportError)(message) from None
    # CW-B52: every input flag is resolved. The full-domain Courant and hazard arrays and the inactive mask were read
    # only by those flags and by the two 0-d maxima (kept for the result), so their storage is released here, before
    # any substep allocation. Names only; no value or order changes.
    del a, exponent, inactive3

    # --- substeps on the device ---
    Pc = cp.ascontiguousarray(P[idx])
    Vc = cp.ascontiguousarray(V[idx])
    Rc = cp.ascontiguousarray(R[idx])
    Sc = cp.ascontiguousarray(S[idx])  # the terminal all-classes rule was enforced in the flag packet above
    # CW-B52: the zero packet of substeps k >= 1 is allocated only when such a substep exists (below); a one-substep
    # step never needs it.
    no_pickup = None
    row_ptr = phase.row_ptr
    bins = phase.bin
    masses = Mc.reshape(-1)[_row_of_entry(cp, row_ptr, nnz0)] * phase.fraction  # dense: W = M[..., None] * F
    del Mc  # CW-B52: its last reads were the state-value flags (resolved above) and the line above
    positions = phase.position_m
    # CW-B52: the six accumulators are NOT pre-allocated as zeros. Substep 0 adopts its own private count-kernel
    # outputs (every row of all seven is written by the kernel) and adds +0.0 in place, which is bitwise the original
    # `zeros + x` (IEEE addition is commutative, signed zeros included); later substeps add into them in place.
    acc = None
    count_fn = _function("ms_csr_count", nb)
    write_fn = _function("ms_csr_write", nb)
    norm_fn = _function("ms_csr_rowsum_normalize", nb)
    i64, i32, f64 = np.int64, np.int32, np.float64
    peak_nnz = 0
    nnz_reads = 0
    for k in range(n_sub):
        if k == 1:
            # CW-B52: the pickup packet's last read was substep 0's write kernel; the zero packet replaces it.
            packet = None  # drops substep 0's reference to the pickup packet
            del Pc
            no_pickup = cp.zeros((n_active, nc), dtype=np.float64)
        packet = Pc if k == 0 else no_pickup  # pickup injected ONCE, at x = 0
        counts = cp.empty(n_rows, dtype=np.int64)
        parts = [cp.empty((n_active, nc), dtype=np.float64) for _ in range(7)]
        try:
            count_fn((grid,), (block,), (i64(n_active), i32(nc), i64(ny), i64(nx), f64(dx), f64(dt_sub), row_ptr, masses,
                                         positions, packet, Vc, Rc, Sc, binding.cell_of_slot, binding.slot_of_cell,
                                         binding.receiver_slot, binding.outlet, counts, *parts))
        except Exception as exc:
            raise TransportError(f"CUDA sparse count kernel launch failed ({type(exc).__name__}: {exc}); no fallback") from exc
        d_decay, d_settled, d_export, d_out, d_in, d_arr_decay, d_arr_settled = parts
        del parts
        row_ptr_new = cp.zeros(n_rows + 1, dtype=np.int64)
        cp.cumsum(counts, out=row_ptr_new[1:])
        del counts  # CW-B52: last read was the prefix sum; released before the write arrays are allocated
        nnz_new = to_int(row_ptr_new[-1])  # the ONE scalar read of the substep
        nnz_reads += 1
        peak_nnz = max(peak_nnz, nnz_new)
        if cap is not None and nnz_new > cap:
            raise SparseCapacityError(nnz_new, cap, k, n_sub)  # before any write array exists
        bins_new = cp.empty(nnz_new, dtype=np.uint8)
        mass_new = cp.empty(nnz_new, dtype=np.float64)
        pos_new = cp.empty(nnz_new, dtype=np.float64)
        try:
            write_fn((grid,), (block,), (i64(n_active), i32(nc), i64(ny), i64(nx), f64(dx), f64(dt_sub), f64(NARROW_SPREAD),
                                         row_ptr, masses, positions, packet, Vc, Rc, Sc, binding.cell_of_slot,
                                         binding.slot_of_cell, binding.receiver_slot, row_ptr_new, bins_new, mass_new,
                                         pos_new))
        except Exception as exc:
            raise TransportError(f"CUDA sparse write kernel launch failed ({type(exc).__name__}: {exc}); no fallback") from exc
        # CW-B52: the same six sums of the same operands as the original
        #   acc[dep_decay] += d_decay + d_arr_decay; acc[settled] += d_settled + d_arr_settled; acc[export] += d_export
        #   acc[out_internal] += d_out; acc[in_internal] += d_in; acc[arrival] += d_arr_decay + d_arr_settled
        # with each two-operand sum written into this substep's own (private, now dead) left operand instead of a new
        # temporary; d_arr_decay is overwritten only after both of its other reads. Substep 0 finishes each sum with
        # `+ 0.0`, i.e. the original `0.0 + sum` of a zero-initialised accumulator.
        cp.add(d_decay, d_arr_decay, out=d_decay)
        cp.add(d_settled, d_arr_settled, out=d_settled)
        cp.add(d_arr_decay, d_arr_settled, out=d_arr_decay)
        del d_arr_settled
        step_sums = {"dep_decay": d_decay, "settled": d_settled, "export": d_export, "out_internal": d_out,
                     "in_internal": d_in, "arrival": d_arr_decay}
        del d_decay, d_settled, d_export, d_out, d_in, d_arr_decay
        if acc is None:
            for value in step_sums.values():
                value += 0.0
            acc = step_sums
        else:
            for name, value in step_sums.items():
                acc[name] += value
        del step_sums, value
        row_ptr, bins, masses, positions = row_ptr_new, bins_new, mass_new, pos_new
        # The accumulated bookkeeping owns its values now. Keep neither
        # this count pass's seven arrays nor the prior pass alive while
        # allocating the next substep or expanding final domain fields.
    del Vc, Rc, Sc, packet
    if n_sub == 1:
        del Pc
    else:
        del no_pickup
    nnz_after = int(masses.shape[0])
    remaining_c = cp.empty(n_rows, dtype=np.float64)
    fraction_new = cp.empty(nnz_after, dtype=np.float64)
    try:
        norm_fn((grid,), (block,), (i64(n_rows), row_ptr, bins, masses, remaining_c, fraction_new))
    except Exception as exc:
        raise TransportError(f"CUDA sparse row-sum kernel launch failed ({type(exc).__name__}: {exc}); no fallback") from exc
    remaining_c = remaining_c.reshape(n_active, nc)

    def full(compact):
        out = cp.zeros((n, nc), dtype=np.float64)
        out[idx] = compact
        return out

    remaining = full(remaining_c)
    dep_decay_f = full(acc.pop("dep_decay"))
    settled_f = full(acc.pop("settled"))
    export = full(acc.pop("export"))
    out_internal_f = full(acc.pop("out_internal"))
    in_internal_f = full(acc.pop("in_internal"))
    if accounting:
        del acc["arrival"]  # CW-B93: return-only (no check reads it); never expanded on the private path
    else:
        arrival_f = full(acc.pop("arrival"))
    deposition = dep_decay_f + settled_f
    finite_pre, negative_pre = {}, {}
    if accounting:
        # CW-B93 private retention (measured H4 refusal at the first face allocation): every class grid that is not
        # part of the private result is released at its LAST read, never later. The finite / negative OUTPUT flags of
        # the four internal-accounting grids are therefore evaluated here -- each the EXACT original expression on the
        # same never-again-written array -- and enqueued below at their ORIGINAL packet positions (packet order, the
        # one batched read and first-violation priority unchanged). The public path evaluates them at enqueue, as
        # before. Released here: decay / settled deposition (last reads: `deposition` and these flags); below:
        # divergence, M + P, internal in and internal out.
        for name, array in (("internal_transfer_in", in_internal_f), ("internal_transfer_out", out_internal_f),
                            ("decay_deposition", dep_decay_f), ("settled", settled_f)):
            finite_pre[name] = finite_flag(array)
            negative_pre[name] = negative_flag(array)
        del name, array, dep_decay_f, settled_f  # `array` would otherwise keep `settled_f` alive
    T = (remaining + deposition) + export
    before = M + P
    divergence = in_internal_f - out_internal_f
    cell_balance = (T - before) - divergence
    if accounting:
        del divergence  # last read: the per-cell balance above
    cell_scale = cp.maximum(cp.maximum(before, T), cp.maximum(in_internal_f, out_internal_f))
    n_ops = operation_count(n_sub, nb)
    cell_tol = summation_error_bound_kg(n_ops, cell_scale)
    del cell_scale  # CW-B52: its only read was the tolerance above
    before_total = pairwise_sum_over_leading_axes(before)
    if accounting:
        del before  # last reads: the balance, its scale and the per-class total above
    after_total = pairwise_sum_over_leading_axes(T)
    dep_total = pairwise_sum_over_leading_axes(deposition)
    exp_total = pairwise_sum_over_leading_axes(export)
    in_total = pairwise_sum_over_leading_axes(in_internal_f)
    if accounting:
        del in_internal_f  # last reads: the balance scale, the flags and the per-class total
    out_total = pairwise_sum_over_leading_axes(out_internal_f)
    residual = after_total - before_total
    pairwise = pairwise_bound_scale_factor(n)
    tolerance = summation_error_bound_kg(
        n_ops, cp.maximum(cp.maximum(before_total, after_total), cp.maximum(in_total, out_total))) \
        + 2.0 * pairwise * (before_total + after_total)
    # CW-B54: validation-only flags of PRIVATE scratch, computed BEFORE the face arrays exist, then the scratch is
    # released. Each flag is the original expression on the same (never again written) arrays; it is enqueued below at
    # its ORIGINAL position, so the packet order, the one batched read and the first-violation priority are unchanged.
    # `max_cell_balance` is the original `cp.max(cp.abs(cell_balance))` reduction of the same array.
    mass_finite_flag = finite_flag(masses)
    mass_negative_flag = negative_flag(masses)
    abs_balance = cp.abs(cell_balance)
    balance_flag = true_flag(abs_balance > cell_tol)
    max_cell_balance = cp.max(abs_balance)
    del abs_balance, cell_balance, cell_tol
    counts_after = cp.diff(row_ptr)
    phase_flags = []
    if nnz_after:
        rows_after = _row_of_entry(cp, row_ptr, nnz_after)
        phase_flags.append((true_flag(masses <= 0.0), "sparse characteristic transport stored a bin without mass"))
        sums = cp.zeros(n_rows, dtype=np.float64)
        scatter_add(sums, rows_after, fraction_new)
        phase_flags.append((true_flag((counts_after > 0) & (cp.abs(sums - 1.0) > fraction_sum_tolerance(nb))),
                            "phase fractions after the step do not sum to 1 within the declared bound"))
        del sums
        phase_flags.append((true_flag(positions >= dx), "phase position after the step reached dx"))
        phase_flags.append((true_flag(bin_index(positions, dx, nb, cp) != bins.astype(np.int64)),
                            "phase position after the step left its bin"))
        if nnz_after > 1:
            same_row = rows_after[1:] == rows_after[:-1]
            phase_flags.append((true_flag(same_row & ~(bins[1:] > bins[:-1])),
                                "sparse characteristic transport wrote bins out of order"))
            del same_row
        del rows_after
    row_flag = true_flag((counts_after > 0) != (remaining_c.reshape(-1) > 0.0))
    # The remaining per-bin masses are owned scratch (the published phase holds fractions, positions, bins and row
    # pointers; `nnz_after` is already a host int), as are the compact remaining pool and the row counts.
    del counts_after, remaining_c, masses, mass_new
    # CW-B59 (measured B57 refusal at the combined-pool check's `deposition + export` temporary): the EXACT original
    # expression, evaluated before the face arrays exist on the same never-again-written T / deposition / export, so
    # its two full-domain temporaries and the boolean mask are freed before the faces. Only the 0-d flag is retained,
    # enqueued below at its original packet position.
    pool_flag = true_flag((deposition + export) > T * (1.0 + 4.0 * _EPS))
    # CW-B63 (measured B61 refusal at the non-outlet check's `export[~outlet]` boolean-index gather): the EXACT
    # original expression on the same never-again-written `export` and the caller's network `outlet` mask, evaluated
    # before the face arrays exist; its mask, gathered copy and comparison are freed here. Only the 0-d flag is
    # retained, enqueued below at its original (last) packet position. Its implicit selected-count read is the same
    # single call, only earlier.
    nonoutlet_export_flag = true_flag(export[~outlet] != 0.0)
    finite_pre["remaining mass"] = mass_finite_flag  # CW-B54 precomputed flags, same packet positions
    negative_pre["remaining mass"] = mass_negative_flag
    del mass_finite_flag, mass_negative_flag
    crossing = out_internal_f + export
    if accounting:
        # CW-B93: last reads of internal out were the balance scale, its flags, its total and `crossing`; together
        # with the releases above, the private path holds only remaining, deposition, T, export and crossing when the
        # four face arrays are allocated (the public path keeps all twelve grids it returns).
        del out_internal_f
    x_gross = cp.zeros((ny, nx + 1, nc), dtype=np.float64)
    y_gross = cp.zeros((ny + 1, nx, nc), dtype=np.float64)
    x_net = cp.zeros((ny, nx + 1, nc), dtype=np.float64)
    y_net = cp.zeros((ny + 1, nx, nc), dtype=np.float64)
    if network.cells_x.size:
        vx = crossing[network.cells_x]
        scatter_add(x_gross, (network.rows_x, network.faces_x), vx)
        scatter_add(x_net, (network.rows_x, network.faces_x), vx * network.sign_x[:, None])
        del vx  # CW-B52: gathered face crossings, last read by the two scatters above
    if network.cells_y.size:
        vy = crossing[network.cells_y]
        scatter_add(y_gross, (network.faces_y, network.cols_y), vy)
        scatter_add(y_net, (network.faces_y, network.cols_y), vy * network.sign_y[:, None])
        del vy
    if accounting:
        del crossing  # CW-B93: its last read was the face scatters; return-only otherwise

    # --- output validation: one batched flag read before publication ---
    checks = DeferredChecks()
    # CW-B54 / CW-B93: `None` marks a precomputed flag (remaining mass; on the private path also the four
    # internal-accounting grids), enqueued at the original position of each list.
    for name, array in (("mobile_after_transfer", T), ("deposition_request", deposition), ("export_request", export),
                        ("internal_transfer_in", None if accounting else in_internal_f),
                        ("internal_transfer_out", None if accounting else out_internal_f),
                        ("x_face_gross", x_gross), ("y_face_gross", y_gross), ("x_face_net", x_net),
                        ("y_face_net", y_net), ("decay_deposition", None if accounting else dep_decay_f),
                        ("settled", None if accounting else settled_f),
                        ("phase fraction", fraction_new), ("phase position", positions), ("remaining mass", None)):
        checks.require(finite_pre[name] if array is None else finite_flag(array),
                       f"sparse characteristic transport produced non-finite {name}")
    for name, array in (("mobile_after_transfer", T), ("deposition_request", deposition), ("export_request", export),
                        ("internal_transfer_in", None if accounting else in_internal_f),
                        ("internal_transfer_out", None if accounting else out_internal_f),
                        ("x_face_gross", x_gross), ("y_face_gross", y_gross),
                        ("decay_deposition", None if accounting else dep_decay_f),
                        ("settled", None if accounting else settled_f), ("phase fraction", fraction_new),
                        ("phase position", positions), ("remaining mass", None)):
        checks.forbid(negative_pre[name] if array is None else negative_flag(array),
                      f"sparse characteristic transport produced negative {name}")
    del finite_pre, negative_pre, array
    checks.forbid(true_flag(export > T), "export request exceeds the post-transfer pool")
    checks.forbid(true_flag(deposition > T), "deposition request exceeds the post-transfer pool")
    checks.forbid(pool_flag,
                  "deposition plus export requests exceed the post-transfer pool")
    checks.forbid(balance_flag,
                  "per-cell mobile balance T - (M + P) = in - out violated beyond FP64 tolerance")
    checks.require(finite_flag(residual), "per-class budget residual is non-finite")
    checks.forbid(true_flag(cp.abs(residual) > tolerance),
                  "per-class mobile mass not conserved by the transfer beyond the declared FP64 bound")
    for flag, message in phase_flags:  # the original nnz-dependent phase checks, in their original order
        checks.forbid(flag, message)
    if phase_flags:
        # CW-B54 correction 1 (root MemoryHook: one retained 0-d block at the result when nnz > 0): the loop variables
        # still named the LAST phase flag after the loop; drop them with the other precomputed flag names.
        del flag, message
    checks.forbid(row_flag, "sparse characteristic transport left a row inconsistent with its remaining mass")
    checks.forbid(nonoutlet_export_flag, "export requested at a non-outlet cell")
    # CW-B54: the packet now holds the only references to the precomputed flags (as to every other flag).
    del balance_flag, phase_flags, row_flag, pool_flag, nonoutlet_export_flag
    try:
        checks.resolve()
    except ValueError as exc:
        raise TransportError(str(exc)) from None
    transfers = read_transfer_counters().delta(counters_before)

    def grid3(array):
        return array.reshape(shape3)

    def phase_and_report():
        # The published CSR phase and the device report (host metadata); built after the transport record on the
        # public path, as before.
        phase_after = _state_class()(
            row_ptr=row_ptr, bin=bins, fraction=fraction_new, position_m=positions, dx_m=dx, n_bins=nb, n_classes=nc,
            shape=(ny, nx), cell_of_slot=binding.cell_of_slot, slot_of_cell=binding.slot_of_cell,
            network_sha256=binding.graph_input_sha256, xp=cp)
        local_estimate = estimate_local_scratch_bytes(nb)
        device_report = {
            "device_id": binding.device_id, "n_bins": nb, "n_rows": n_rows,
            "launch": {"grid_blocks": grid, "block_threads": block, "launched_threads": launched},
            "kernel_attributes": attributes,
            "local_scratch_estimate_bytes_per_thread": local_estimate,
            "local_scratch_estimate_bytes_launched": {k: v * launched for k, v in local_estimate.items()},
            "local_scratch_compiled_bytes_launched": {k: int(v.get("local_size_bytes", -1)) * launched
                                                      for k, v in attributes.items()},
            "row_sum_self_check": row_sum_check,
            "scalar_reads_nnz": nnz_reads,
            "transfer_counters_step": {"device_to_host_bytes": int(transfers.device_to_host_bytes),
                                       "host_to_device_bytes": int(transfers.host_to_device_bytes),
                                       "scalar_reads": int(getattr(transfers, "scalar_reads", 0))},
        }
        return phase_after, device_report

    if accounting:
        phase_after, device_report = phase_and_report()
        # CW-B93: the SAME objects the public `TransportStep` would carry for these fields (no copy).
        private = _TransportAccountingStep(
            dt_s=dt, n_substeps=n_sub,
            mobile_after_transfer_kg=grid3(T), deposition_request_kg=grid3(deposition),
            export_request_kg=grid3(export),
            x_face_gross_kg=x_gross, y_face_gross_kg=y_gross, x_face_net_kg=x_net, y_face_net_kg=y_net,
            deposition_request_by_class_kg=dep_total, export_request_by_class_kg=exp_total,
            budget_residual_by_class_kg=residual, budget_tolerance_by_class_kg=tolerance,
            max_courant=max_courant, max_decay_exponent=max_exponent,
            max_cell_balance_residual_kg=max_cell_balance)
        return _SparseAccountingStep(
            transport=private, phase_after=phase_after, mobile_remaining_kg=grid3(remaining), n_bins=nb,
            implementation="cuda", nnz_after=nnz_after, peak_nnz=int(peak_nnz), device_report=device_report)

    transport = TransportStep(
        dt_s=dt, n_substeps=n_sub,
        mobile_after_transfer_kg=grid3(T), deposition_request_kg=grid3(deposition), export_request_kg=grid3(export),
        decay_deposition_kg=grid3(dep_decay_f), settled_kg=grid3(settled_f),
        internal_transfer_in_kg=grid3(in_internal_f), internal_transfer_out_kg=grid3(out_internal_f),
        divergence_kg=grid3(divergence),
        x_face_gross_kg=x_gross, y_face_gross_kg=y_gross, x_face_net_kg=x_net, y_face_net_kg=y_net,
        mobile_before_by_class_kg=before_total, mobile_after_by_class_kg=after_total,
        deposition_request_by_class_kg=dep_total, export_request_by_class_kg=exp_total,
        budget_residual_by_class_kg=residual, budget_tolerance_by_class_kg=tolerance,
        max_courant=max_courant, max_decay_exponent=max_exponent,
        max_cell_balance_residual_kg=max_cell_balance,
    )
    phase_after, device_report = phase_and_report()
    return SparseCharacteristicStep(
        transport=transport, phase_after=phase_after, mobile_before_kg=grid3(before), mobile_remaining_kg=grid3(remaining),
        arrival_deposition_kg=grid3(arrival_f), crossing_kg=grid3(crossing), n_bins=nb, implementation="cuda",
        nnz_after=nnz_after, peak_nnz=int(peak_nnz), device_report=device_report)
