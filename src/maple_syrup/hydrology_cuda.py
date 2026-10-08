"""Prepared, fused CUDA hydrology for the frozen-terrain water-only storm (Phase 4R, task phase4r_gpu_storm, task A).

`hydrology_numba.prepared_coupled_step` stays the CPU oracle and is not touched. This module evaluates the SAME
equations on a CuPy device, term for term in the same expression order, after a one-time preparation of everything
that is constant in a fixed-terrain replay:

    prepare_cuda_hydrology(graph, params, mode="auto") -> CudaHydrologyContext   # once; validated, owned device copies
    prepared_coupled_step(ctx, rain, state, dt, control) -> CoupledStep          # every step; same CoupledStep fields
                                                                                 # (alias: cuda_coupled_step)

Both column laws (`fixed_ksat`, `pavement_hawkins`: Smith-Parlange capacity, linear drainage, saturation return), the
legacy old-flux branches (complete > no run-on > partial), the coherent donor sums, the Courant/old-flux checks, the
ordered `[0, R]` bisection (the very device root search of `routing_cuda`; `control.root_solver == "newton"` selects the
safeguarded Newton variant of the same step kernels, `routing_newton_cuda`, same equation and checks), the storage identity and every per-cell and
global balance check of `routing._route` are evaluated on the device. Nothing here imports Numba or CuPy at import
time; `prepare_cuda_hydrology` needs CuPy and a device and never falls back to NumPy/Numba (`CudaUnavailableError`).

Device functions, one copy each, shared by every launch structure:

  pre_cell    cellwise: input checks, column physics, branch, old flux q_old, hpre, route input checks, Courant number.
  solve_cell  level position p: coherent old inflow, base RHS, new donor sum, RHS, bisection, q, storage identity.
  post_cell   cellwise: velocity, face volume, balances, finiteness/sign checks, per-cell reduction rows.
  reduce      ONE block: OR of flag words, branch counts, maxima, the four scalar sums, the scalar algebra of
              `_route_phase`, and the compact result packet (PACKET_WORDS 64-bit words = 144 bytes).

Launch structures (selected ONCE at preparation, recorded in `ctx.mode`; they differ only in launch layout):

  "fused"  one launch of one block of FUSED_THREADS (=128) threads: strided cell loops, an unconditional
           `__syncthreads()` after every dependency level (a level's cells are independent, donors lie in earlier
           levels, the barrier makes their results visible inside the block), then post and reduce in the same kernel.
  "split"  pre-cells (grid), one launch per dependency level (grid; stream order is the barrier), post-cells (grid),
           reduce (one block of REDUCE_THREADS threads). Parallel across cells for wide graphs.
  "auto"   "fused" when the widest dependency level has at most FUSED_MAX_LEVEL_WIDTH (=128, the block size) cells,
           else "split". An explicit "fused"/"split" is honoured for any width (the cell loops are strided).
There are no atomics, no inter-block communication, no CUDA graphs and no speculative root search.

Beyond the coupled step (Task B): `prepared_column_step` (the column stage alone, reusing `pre_cell`, no routing
guards) and the driver helpers `cuda_step_with_packet` / `CudaStormAccumulator`, used by the single host scheduler
`storm.evolve` when `control.implementation == "cuda"`: one accumulate launch per accepted step and one report launch
per hydrograph row, no read-back. The context's launch-dependent host metadata is sealed at preparation
(`scalar_signature`) and compared before every launch, because the raw kernels do no bounds checking.

Host traffic per attempted step: ONE counted device-to-host read of the packet (PACKET_WORDS * 8 bytes); no static
upload, no full-grid download, no host synchronization other than that read. Every public output array is freshly
allocated per call; there is no reusable workspace and no mutable shared buffer, so a context may be shared by threads.
A failure raises after the packet read, before any result exists, and never writes an input or an earlier result.

Error precedence (identical to `prepared_coupled_step` on the CPU): context/state/control type -> `dt` -> dynamic array
structure (depth, soil, rain, previous discharge) -> column failures (`InfiltrationError`, lowest recorded bit) ->
scalar-option failures -> routing failures in `hydrology_numba._resolve_route` order -> previous-discharge failures.
Because the device is launched before the host reads anything, option errors are DEFERRED: with invalid options only
the cellwise column stage runs (stage 0, never an unvalidated iteration count) so a column failure still outranks them.

Floating point: compiled with `--fmad=false`, `--prec-div=true`, `--prec-sqrt=true`, `--ftz=false`, no fast math; the
route arithmetic and the root search use the round-to-nearest intrinsics `__dadd_rn/__dmul_rn/__dsub_rn/__dsqrt_rn`
exactly as `routing_cuda`; min/max are the NaN-propagating forms of the CPU kernels (CUDA `fmin/fmax` are NOT). `expm1`
and `pow` are the device libm functions: they may differ from NumPy/libm by an ulp and affect only the Smith-Parlange
capacity near its limits and one threshold test; the declared bound against the CPU is rtol 2e-12 / atol 1e-14 and
nothing here claims bitwise equality of the whole step. Scalar sums use thread partials in index order followed by a
fixed shared-memory pairwise tree: deterministic run to run, NOT NumPy's summation order (differences are rounding noise
far below the declared tolerances and the balance tolerances, which are unchanged).

Result scalars (`export_m3`, branch counts, maxima, ...) are 0-d CuPy views of the fresh per-call packet (the same
convention as the reference array path), not host scalars. The context's owned device arrays are immutable by contract;
only pointer/shape/dtype/contiguity/device metadata is re-checked per step, arbitrary content mutation is not detected.
Dynamic CuPy inputs must be on the current device and already ordered with respect to the current stream (the caller
orders producers on other streams). Asynchronous device faults surface at the packet read as CuPy/CUDA errors.

Nothing here was run by its author (file-only tools); Codex records results.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import threading
import time
import weakref
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from maple_syrup import routing_cuda, routing_newton_cuda
from maple_syrup.hydrology_numba import (
    _COLUMN_INPUT_MASK,
    _COLUMN_MESSAGES,
    _EPS64,
    _TWO_THIRDS,
    HydrologyPreparationError,
    _lowest_bit,
    _resolve_route,
)
from maple_syrup.infiltration import (
    INFILTRATION_MODELS,
    LOCAL_BALANCE_RTOL,
    ColumnStep,
    InfiltrationError,
    _check_dt,
)
from maple_syrup.routing import (
    BALANCE_RTOL,
    RouteStep,
    RoutingError,
    _check_root_solver,
    _check_step_options,
)
from maple_syrup.routing_cuda import CudaUnavailableError
from maple_syrup.storm import CoupledStep, StormControl, StormError, StormState

__all__ = [
    "COMPILE_OPTIONS",
    "FUSED_MAX_LEVEL_WIDTH",
    "FUSED_THREADS",
    "MODES",
    "PACKET_WORDS",
    "REDUCE_THREADS",
    "CudaHydrologyContext",
    "CudaHydrologyPreparationError",
    "CudaStormAccumulator",
    "CudaUnavailableError",
    "cuda_coupled_step",
    "cuda_step_with_packet",
    "kernel_provenance",
    "kernel_source",
    "launch_count",
    "load_newton_kernels",
    "prepare_cuda_hydrology",
    "prepared_column_step",
    "prepared_coupled_step",
    "prepared_cuda_hydrology",
    "select_mode",
]

COMPILE_OPTIONS = routing_cuda.COMPILE_OPTIONS
FUSED_THREADS = 128
CELL_THREADS = 128
REDUCE_THREADS = 256  # a power of two <= 256 (shared tree size)
FUSED_MAX_LEVEL_WIDTH = FUSED_THREADS
MODES = ("auto", "fused", "split")
PACKET_WORDS = 18
_MODEL_CODES = {"fixed_ksat": 0, "pavement_hawkins": 1}
_KERNEL_NAMES = ("maple_syrup_hydro_fused", "maple_syrup_hydro_pre", "maple_syrup_hydro_solve_level",
                 "maple_syrup_hydro_post", "maple_syrup_hydro_reduce", "maple_syrup_storm_accumulate",
                 "maple_syrup_storm_report")

# Packet word layout (uint64; doubles are stored as raw bits):
#  0 OR of column flag bits (bits 0..18)   1 OR of previous-discharge flag bits   2 OR of route flag bits (0..25)
#  3 n_no_runon  4 n_partial_runon  5 n_complete_runon (int64)
#  6 max Courant old  7 max |h_new - h_flow|  8 max |cell balance|  9 max velocity
# 10 storage change  11 export  12 budget residual  13 global balance tolerance  14 outlet discharge
# 15 bit mask of non-finite scalars (storage, export, residual, tolerance, outlet)  16 residual failed  17 max Courant new
_P_COL, _P_Q, _P_ROUTE, _P_NNO, _P_NPARTIAL, _P_NCOMPLETE = 0, 1, 2, 3, 4, 5
_P_CR_OLD, _P_CONS, _P_BAL, _P_VEL = 6, 7, 8, 9
_P_STORAGE, _P_EXPORT, _P_RESIDUAL, _P_GTOL, _P_OUTLET, _P_NONFINITE, _P_RESFAIL, _P_CR_NEW = 10, 11, 12, 13, 14, 15, 16, 17

_STEP_ARGS = r"""
    const long long n, const long long na, const int nlev, \
    const unsigned char* active, const unsigned char* outlet, const double* kcell, const double* col, \
    const long long* order, const long long* donor_cell, const long long* bounds, \
    const double* h_in, const double* s_in, const double* r_in, const double* qprev, \
    double* depth_new, double* soil_new, double* rain_o, double* intake_o, double* overflow_o, \
    double* drainage_o, double* hpre_o, double* q_old_o, double* h_new, double* flow, double* q_new, \
    double* qin_new, double* qin_old, double* velocity, double* face, \
    unsigned int* cflag, unsigned int* rflag, unsigned char* br, double* cr_row, double* cons_row, \
    double* bal_row, double* sc_row, unsigned long long* packet, \
    const int hawkins, const int iterations, const int stage, \
    const double dt, const double local_rtol, const double c, const double dt_over_dx, const double area, \
    const double dx, const double cr_max, const double root_tol, const double bal_rtol, const double p23, \
    const double eps64, const double tol_pref
"""

_STEP_FWD = r"""
    n, na, nlev, active, outlet, kcell, col, order, donor_cell, bounds, h_in, s_in, r_in, qprev, \
    depth_new, soil_new, rain_o, intake_o, overflow_o, drainage_o, hpre_o, q_old_o, h_new, flow, q_new, \
    qin_new, qin_old, velocity, face, cflag, rflag, br, cr_row, cons_row, bal_row, sc_row, packet, \
    hawkins, iterations, stage, dt, local_rtol, c, dt_over_dx, area, dx, cr_max, root_tol, bal_rtol, p23, \
    eps64, tol_pref
"""

_HYDRO_BODY = r"""
#define STEP_ARGS __STEP_ARGS__
#define STEP_FWD __STEP_FWD__

__device__ __forceinline__ double nan_value() { return __longlong_as_double(0x7ff8000000000000LL); }

// np.maximum / np.minimum of the CPU kernels: NaN propagates (CUDA fmax/fmin return the non-NaN operand).
__device__ __forceinline__ double fmax_nan(const double a, const double b)
{
    if (a != a || b != b) return nan_value();
    return a > b ? a : b;
}
__device__ __forceinline__ double fmin_nan(const double a, const double b)
{
    if (a != a || b != b) return nan_value();
    return a < b ? a : b;
}

// Phase A + B1: column physics, old flux, route input checks. Mirrors hydrology_numba column_kernel / route_kernel B1.
__device__ __forceinline__ void pre_cell(STEP_ARGS, const long long i)
{
    const double hi = h_in[i];
    const double si = s_in[i];
    const double ri = r_in[i];
    const bool a = active[i] != 0;
    const double ksat = col[i * 6 + 0];
    const double psi = col[i * 6 + 1];
    const double drn = col[i * 6 + 2];
    const double thick = col[i * 6 + 3];
    const double smax = col[i * 6 + 4];
    const double lam = col[i * 6 + 5];
    unsigned int f = 0u;

    if (!isfinite(hi)) f |= 1u << 0;
    if (hi < 0.0) f |= 1u << 1;
    if (!isfinite(si)) f |= 1u << 2;
    if (si < 0.0) f |= 1u << 3;
    if (!isfinite(ri)) f |= 1u << 4;
    if (ri < 0.0) f |= 1u << 5;
    if (si > smax) f |= 1u << 6;
    if (ri > 0.0 && !a) f |= 1u << 7;

    const double rain = ri * dt;
    const double avail = hi + rain;
    double kf;
    if (hawkins == 1 && ri > 0.0) {
        kf = (-lam) * expm1((-ri) / lam);
    } else {
        kf = ksat;
    }
    const double deficit = (smax - si) / thick;
    const double scale = (psi + hi) * deficit;
    const bool capillary = scale > 0.0;
    const double x = si / (capillary ? scale : 1.0);
    const double denom = -expm1(-x);
    const bool fin = denom > 0.0;
    double capacity;
    if (capillary) {
        capacity = kf / (fin ? denom : 1.0);
    } else {
        capacity = kf;
    }
    double intake_raw;
    if (capillary && (!fin) && kf > 0.0) {
        intake_raw = avail;
    } else {
        intake_raw = fmin_nan(avail, capacity * dt);
    }
    const double intake = a ? intake_raw : 0.0;
    const double wetted = si + intake;
    const double demand = (si / smax) * ksat * drn * dt;
    const double drainage = a ? fmin_nan(demand, wetted) : 0.0;
    const double retained = wetted - drainage;
    const double overflow = fmax_nan(retained - smax, 0.0);
    const double soil = fmin_nan(retained, smax);
    // coherent branch form of h + P - J (same retained arithmetic as hp below); see infiltration.column_step
    const double ret_s = (a && intake >= avail) ? 0.0 : fmax_nan(hi - fmax_nan(intake - rain, 0.0), 0.0);
    const double depth = (ret_s + fmax_nan(rain - intake, 0.0)) + overflow;

    depth_new[i] = depth;
    soil_new[i] = soil;
    rain_o[i] = rain;
    intake_o[i] = intake;
    overflow_o[i] = overflow;
    drainage_o[i] = drainage;

    if (!isfinite(depth)) f |= 1u << 8;
    if (depth < 0.0) f |= 1u << 9;
    if (!isfinite(soil)) f |= 1u << 10;
    if (soil < 0.0) f |= 1u << 11;
    if (!isfinite(intake)) f |= 1u << 12;
    if (intake < 0.0) f |= 1u << 13;
    if (!isfinite(drainage)) f |= 1u << 14;
    if (drainage < 0.0) f |= 1u << 15;
    if (!isfinite(overflow)) f |= 1u << 16;
    if (overflow < 0.0) f |= 1u << 17;
    const double residual = (depth + soil + drainage) - (hi + si + rain);
    const double bscale = hi + rain + si + intake + drainage + overflow;
    if (fabs(residual) > local_rtol * bscale) f |= 1u << 18;

    // coupled_step: hpre, branch masks (legacy precedence), old flux
    const bool complete = a && intake >= hi + rain;
    // infilt.for 112-115 sets d(1) = 0 in the complete branch; a tiny hi absorbed in hi + rain must not leave a residue
    const double hp = complete ? 0.0 : fmax_nan(hi - fmax_nan(intake - rain, 0.0), 0.0);
    const bool no_runon = a && (!complete) && intake <= rain;
    const bool partial = a && (!complete) && (!no_runon);
    const double qp = qprev[i];
    double qo;
    unsigned char code;
    if (complete) {
        qo = 0.0;
        code = 3;
    } else if (no_runon) {
        qo = qp;
        code = 1;
    } else if (partial) {
        qo = (__dsqrt_rn(hp) * hp) * kcell[i];
        code = 2;
    } else {
        qo = 0.0;
        code = 0;
    }
    hpre_o[i] = hp;
    q_old_o[i] = qo;
    br[i] = code;

    unsigned int qf = 0u;
    if (!isfinite(qp)) qf |= 1u << 0;
    if (qp < 0.0) qf |= 1u << 1;
    if ((!a) && qp != 0.0) qf |= 1u << 2;
    cflag[i] = f | (qf << 24);

    // B1: routing input checks (h_start = column depth, h_old = hpre) and the old-flux Courant number
    unsigned int r = 0u;
    const double hs = depth;
    const double ho = hp;
    if (!isfinite(hs)) r |= 1u << 0;
    if (hs < 0.0) r |= 1u << 1;
    if (!isfinite(ho)) r |= 1u << 2;
    if (ho < 0.0) r |= 1u << 3;
    if (!isfinite(qo)) r |= 1u << 4;
    if (qo < 0.0) r |= 1u << 5;
    if (a && ho > hs) r |= 1u << 6;
    if ((!a) && qo != 0.0) r |= 1u << 7;
    const double kk = a ? kcell[i] : 1.0;
    const double ratio = qo / kk;
    double implied;
    if (ratio >= 0.0) {
        implied = pow(ratio, p23);
    } else {
        implied = nan_value();
    }
    if (a && fabs(implied - ho) > root_tol + eps64 * ho) r |= 1u << 8;
    double cr;
    if (ho > 0.0) {
        cr = qo / ho;
    } else if (qo > 0.0) {
        cr = __longlong_as_double(0x7ff0000000000000LL);
    } else {
        cr = 0.0;
    }
    cr = cr * dt_over_dx;
    if (!a) cr = 0.0;
    if (!isfinite(qo)) r |= 1u << 9;
    if (cr > cr_max) r |= 1u << 10;
    rflag[i] = r;
    cr_row[i] = cr;
    if (!a) {  // inactive cells keep their depth and carry no flow (the arrays are not pre-zeroed)
        h_new[i] = depth;
        flow[i] = 0.0;
        q_new[i] = 0.0;
        qin_new[i] = 0.0;
        qin_old[i] = 0.0;
    }
}

// Phases B2 + sweep for the cell at level position p: coherent old inflow, base RHS, donors' new flux, RHS, root.
__device__ __forceinline__ void solve_cell(STEP_ARGS, const long long p)
{
    const long long i = order[p];
    const long long d0 = donor_cell[p];
    double tot = d0 >= 0 ? q_old_o[d0] : 0.0;
    for (int s = 1; s < 4; ++s) {
        const long long d = donor_cell[(long long)s * na + p];
        tot = tot + (d >= 0 ? q_old_o[d] : 0.0);
    }
    const double qo = q_old_o[i];
    const double base = depth_new[i] + c * (tot - qo);
    double qin = 0.0;
    for (int s = 0; s < 4; ++s) {
        const long long d = donor_cell[(long long)s * na + p];
        if (d >= 0) {
            qin = __dadd_rn(qin, q_new[d]);
        } else {
            qin = __dadd_rn(qin, 0.0);
        }
    }
    const double rhs = __dadd_rn(base, __dmul_rn(qin, c));
    const double k = kcell[i];
    const double lo = maple_syrup_bisect(rhs, k, c, iterations);
    double q = __dsqrt_rn(lo);
    q = __dmul_rn(q, lo);
    q = __dmul_rn(q, k);
    unsigned int r = 0u;
    if (!isfinite(rhs)) r |= 1u << 11;
    if (rhs < 0.0) r |= 1u << 12;
    if (r != 0u) rflag[i] |= r;
    h_new[i] = rhs - q * c;
    q_new[i] = q;
    flow[i] = lo;
    qin_new[i] = qin;
    qin_old[i] = tot;
}

// Phase B3: per-cell outputs and checks (all cells; inactive cells were initialised by pre_cell).
__device__ __forceinline__ void post_cell(STEP_ARGS, const long long i)
{
    const bool a = active[i] != 0;
    const double hs = depth_new[i];
    const double hn = h_new[i];
    const double qn = q_new[i];
    const double fl = flow[i];
    const double qo = q_old_o[i];
    const double qio = qin_old[i];
    const double qi = qin_new[i];
    double vel, fc;
    if (a) {
        vel = __dsqrt_rn(fl) * kcell[i];
        fc = area * (c * (qo + qn));
    } else {
        vel = 0.0;
        fc = 0.0;
    }
    double bal = (hn - hs) - c * ((qio + qi) - (qo + qn));
    const double sc = hs + hn + c * (qio + qi + qo + qn);
    if (!a) bal = 0.0;
    const double cons = a ? (hn - fl) : 0.0;
    velocity[i] = vel;
    face[i] = fc;
    unsigned int r = 0u;
    if (!isfinite(hn)) r |= 1u << 13;
    if (!isfinite(qn)) r |= 1u << 14;
    if (!isfinite(fl)) r |= 1u << 15;
    if (!isfinite(vel)) r |= 1u << 16;
    if (!isfinite(qi)) r |= 1u << 17;
    if (!isfinite(qio)) r |= 1u << 18;
    if (!isfinite(fc)) r |= 1u << 19;
    if (!isfinite(sc)) r |= 1u << 20;
    if (hn < 0.0) r |= 1u << 21;
    if (qn < 0.0) r |= 1u << 22;
    if (fl < 0.0) r |= 1u << 23;
    const double ac = fabs(cons);
    if (ac > root_tol) r |= 1u << 24;
    const double ab = fabs(bal);
    if (ab > bal_rtol * sc) r |= 1u << 25;
    rflag[i] |= r;
    cons_row[i] = ac;
    bal_row[i] = ab;
    sc_row[i] = sc;
}

// Block reductions: fixed trees (deterministic), every thread of the block calls them uniformly.
__device__ double block_sum(const double v)
{
    __shared__ double sh[256];
    const int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = __dadd_rn(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    const double r = sh[0];
    __syncthreads();
    return r;
}
__device__ double block_max(const double v)
{
    __shared__ double sh[256];
    const int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && sh[tid + s] > sh[tid]) sh[tid] = sh[tid + s];
        __syncthreads();
    }
    const double r = sh[0];
    __syncthreads();
    return r;
}
__device__ unsigned int block_or(const unsigned int v)
{
    __shared__ unsigned int sh[256];
    const int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] |= sh[tid + s];
        __syncthreads();
    }
    const unsigned int r = sh[0];
    __syncthreads();
    return r;
}
__device__ long long block_isum(const long long v)
{
    __shared__ long long sh[256];
    const int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    const long long r = sh[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ void put_double(unsigned long long* packet, const int k, const double v)
{
    packet[k] = (unsigned long long)__double_as_longlong(v);
}

__device__ __forceinline__ void reduce_block(STEP_ARGS)
{
    const int tid = threadIdx.x;
    const int bd = blockDim.x;
    unsigned int cf = 0u, qf = 0u, rf = 0u;
    long long n_no = 0, n_partial = 0, n_complete = 0;
    double m_cr = 0.0, m_cons = 0.0, m_bal = 0.0, m_vel = 0.0;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (long long i = tid; i < n; i += bd) {
        const unsigned int w = cflag[i];
        cf |= (w & 0xFFFFFFu);
        qf |= (w >> 24);
        if (stage == 0) continue;
        rf |= rflag[i];
        const unsigned char code = br[i];
        if (code == 1) n_no += 1;
        if (code == 2) n_partial += 1;
        if (code == 3) n_complete += 1;
        const double cr = cr_row[i];
        const double cons = cons_row[i];
        const double bal = bal_row[i];
        const double vel = velocity[i];
        if (cr > m_cr) m_cr = cr;
        if (cons > m_cons) m_cons = cons;
        if (bal > m_bal) m_bal = bal;
        if (vel > m_vel) m_vel = vel;
        const bool out = outlet[i] != 0;
        s0 = s0 + (h_new[i] - depth_new[i]);
        s1 = s1 + (out ? face[i] : 0.0);
        s2 = s2 + (active[i] != 0 ? sc_row[i] : 0.0);
        s3 = s3 + (out ? q_new[i] : 0.0);
    }
    const unsigned int cf_all = block_or(cf);
    const unsigned int qf_all = block_or(qf);
    if (stage == 0) {
        if (tid == 0) {
            packet[0] = cf_all;
            packet[1] = qf_all;
        }
        return;
    }
    const unsigned int rf_all = block_or(rf);
    const long long t_no = block_isum(n_no);
    const long long t_partial = block_isum(n_partial);
    const long long t_complete = block_isum(n_complete);
    const double g_cr = block_max(m_cr);
    const double g_cons = block_max(m_cons);
    const double g_bal = block_max(m_bal);
    const double g_vel = block_max(m_vel);
    const double t0 = block_sum(s0);
    const double t1 = block_sum(s1);
    const double t2 = block_sum(s2);
    const double t3 = block_sum(s3);
    if (tid == 0) {
        const double storage = area * t0;
        const double exportv = t1;
        const double residual = storage + exportv;
        const double gtol = tol_pref * t2;
        const double outq = dx * t3;
        unsigned long long nonfinite = 0ull;
        if (!isfinite(storage)) nonfinite |= 1ull << 0;
        if (!isfinite(exportv)) nonfinite |= 1ull << 1;
        if (!isfinite(residual)) nonfinite |= 1ull << 2;
        if (!isfinite(gtol)) nonfinite |= 1ull << 3;
        if (!isfinite(outq)) nonfinite |= 1ull << 4;
        packet[0] = cf_all;
        packet[1] = qf_all;
        packet[2] = rf_all;
        packet[3] = (unsigned long long)t_no;
        packet[4] = (unsigned long long)t_partial;
        packet[5] = (unsigned long long)t_complete;
        put_double(packet, 6, g_cr);
        put_double(packet, 7, g_cons);
        put_double(packet, 8, g_bal);
        put_double(packet, 9, g_vel);
        put_double(packet, 10, storage);
        put_double(packet, 11, exportv);
        put_double(packet, 12, residual);
        put_double(packet, 13, gtol);
        put_double(packet, 14, outq);
        packet[15] = nonfinite;
        packet[16] = fabs(residual) > gtol ? 1ull : 0ull;
        put_double(packet, 17, g_vel * dt_over_dx);
    }
}

// ONE launch, ONE block. Every barrier is reached by every thread: loop bounds are uniform, the strided cell loops
// sit between barriers and never around one, and `stage` is a kernel argument (uniform).
extern "C" __global__ void maple_syrup_hydro_fused(STEP_ARGS)
{
    const int tid = threadIdx.x;
    const int bd = blockDim.x;
    for (long long i = tid; i < n; i += bd) pre_cell(STEP_FWD, i);
    __syncthreads();
    if (stage > 0) {
        for (int lev = 0; lev < nlev; ++lev) {
            const long long b0 = bounds[lev];
            const long long m = bounds[lev + 1] - b0;
            for (long long j = tid; j < m; j += bd) solve_cell(STEP_FWD, b0 + j);
            __syncthreads();
        }
        for (long long i = tid; i < n; i += bd) post_cell(STEP_FWD, i);
        __syncthreads();
    }
    reduce_block(STEP_FWD);
}

extern "C" __global__ void maple_syrup_hydro_pre(STEP_ARGS)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i < n) pre_cell(STEP_FWD, i);
}

extern "C" __global__ void maple_syrup_hydro_solve_level(STEP_ARGS, const long long b0, const long long m)
{
    const long long j = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (j < m) solve_cell(STEP_FWD, b0 + j);
}

extern "C" __global__ void maple_syrup_hydro_post(STEP_ARGS)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i < n) post_cell(STEP_FWD, i);
}

extern "C" __global__ void maple_syrup_hydro_reduce(STEP_ARGS)
{
    reduce_block(STEP_FWD);
}
"""

# Driver-owned accumulation and reporting (Task B). The host scheduler stays the single authority for boundaries,
# retries and guards; these kernels only replace the ~20 small CuPy operations per accepted step and ~30 per row of
# `storm.evolve`, reading the scalars from the step's device packet so nothing is read back. Same IEEE operations
# as the CPU loop: `x = x + y` adds, NaN-propagating maxima, strict `>` peak test; sums use the fixed block tree.
_STORM_BODY = r"""
__device__ __forceinline__ double packet_double(const unsigned long long* packet, const int k)
{
    return __longlong_as_double((long long)packet[k]);
}

// acc layout: 0 cum_export, 1 peak_q, 2 peak_t, 3 max_balance, 4 max_constitutive, 5 max_cr_old, 6 max_cr_new;
// acc_i layout: 0 no-run-on, 1 partial, 2 complete cell-steps.
extern "C" __global__ void maple_syrup_storm_accumulate(
    const long long n,
    double* cum_rain, double* cum_intake, double* cum_return, double* cum_drain,
    double* peak_depth, double* peak_velocity,
    const double* rain, const double* intake, const double* overflow, const double* drainage,
    const double* depth, const double* velocity,
    double* acc, long long* acc_i, const unsigned long long* packet, const double t)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i < n) {
        cum_rain[i] = cum_rain[i] + rain[i];
        cum_intake[i] = cum_intake[i] + intake[i];
        cum_return[i] = cum_return[i] + overflow[i];
        cum_drain[i] = cum_drain[i] + drainage[i];
        peak_depth[i] = fmax_nan(peak_depth[i], depth[i]);
        peak_velocity[i] = fmax_nan(peak_velocity[i], velocity[i]);
    }
    if (i == 0) {
        acc[0] = acc[0] + packet_double(packet, 11);
        const double outq = packet_double(packet, 14);
        if (outq > acc[1]) {
            acc[2] = t;
            acc[1] = outq;
        }
        acc[3] = fmax_nan(acc[3], packet_double(packet, 8));
        acc[4] = fmax_nan(acc[4], packet_double(packet, 7));
        acc[5] = fmax_nan(acc[5], packet_double(packet, 6));
        acc[6] = fmax_nan(acc[6], packet_double(packet, 17));
        acc_i[0] += (long long)packet[3];
        acc_i[1] += (long long)packet[4];
        acc_i[2] += (long long)packet[5];
    }
}

// One hydrograph row (the 16 HYDROGRAPH_COLUMNS) from device state; host scalars arrive as arguments. ONE block.
extern "C" __global__ void maple_syrup_storm_report(
    const long long n, const double* cum_rain, const double* cum_intake, const double* cum_return,
    const double* cum_drain, const double* depth, const double* soil, const double* last_velocity,
    const double* acc, const unsigned long long* packet, double* hydro, const long long row,
    const double t, const double area, const double n_steps, const double n_rejected, const double dt_min)
{
    const int tid = threadIdx.x;
    const int bd = blockDim.x;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0, m_depth = 0.0, m_vel = 0.0;
    for (long long i = tid; i < n; i += bd) {
        s0 = s0 + cum_rain[i];
        s1 = s1 + cum_intake[i];
        s2 = s2 + cum_return[i];
        s3 = s3 + cum_drain[i];
        s4 = s4 + depth[i];
        s5 = s5 + soil[i];
        if (depth[i] > m_depth) m_depth = depth[i];
        if (last_velocity[i] > m_vel) m_vel = last_velocity[i];
    }
    const double t0 = block_sum(s0);
    const double t1 = block_sum(s1);
    const double t2 = block_sum(s2);
    const double t3 = block_sum(s3);
    const double t4 = block_sum(s4);
    const double t5 = block_sum(s5);
    const double g_depth = block_max(m_depth);
    const double g_vel = block_max(m_vel);
    if (tid == 0) {
        double* out = hydro + row * 16;
        out[0] = t;
        out[1] = area * t0;
        out[2] = area * t1;
        out[3] = area * t2;
        out[4] = area * t3;
        out[5] = acc[0];
        out[6] = area * t4;
        out[7] = area * t5;
        out[8] = packet_double(packet, 14);
        out[9] = g_depth;
        out[10] = g_vel;
        out[11] = n_steps;
        out[12] = n_rejected;
        out[13] = dt_min;
        out[14] = acc[3];
        out[15] = acc[4];
    }
}
"""

_HYDRO_TEXT = _HYDRO_BODY.replace("__STEP_ARGS__", _STEP_ARGS.strip()).replace("__STEP_FWD__", _STEP_FWD.strip())
_SOURCE = routing_cuda.bisect_device_source() + _HYDRO_TEXT + _STORM_BODY  # the default (bisection) module: unchanged

# The Newton variant (explicit, `control.root_solver == "newton"`): the SAME step kernels with the root call replaced
# by the shared device helper of `routing_newton_cuda` (`iterations` then carries the Newton pass cap). Separate module,
# so the default carries no extra branch; the accumulate/report kernels are the default module's.
_BISECT_CALL = "maple_syrup_bisect(rhs, k, c, iterations)"
_NEWTON_CALL = "maple_syrup_newton_root(rhs, k, c, iterations)"
if _HYDRO_TEXT.count(_BISECT_CALL) != 1:  # pragma: no cover - guards the text substitution
    raise RuntimeError("the hydrology kernel text must contain exactly one root call")
_STEP_KERNEL_NAMES = _KERNEL_NAMES[:5]
_SOURCE_NEWTON = routing_newton_cuda.newton_device_source() + _HYDRO_TEXT.replace(_BISECT_CALL, _NEWTON_CALL)


class CudaHydrologyPreparationError(HydrologyPreparationError):
    """The graph/column parameters cannot be prepared for the CUDA hydrology (wrong type/namespace/dtype/shape/device,
    inconsistent masks, out-of-range or unsafe static data). Raised before any context exists."""


def kernel_source(root_solver: str = "bisection") -> str:
    """The default (bisection) module source, or with `root_solver="newton"` the Newton step-kernel variant."""
    if root_solver not in ("bisection", "newton"):
        raise CudaHydrologyPreparationError(f"root_solver must be 'bisection' or 'newton', got {root_solver!r}")
    return _SOURCE_NEWTON if root_solver == "newton" else _SOURCE


def select_mode(max_level_width: int, mode: str = "auto") -> str:
    """Pure host selection of the launch structure: "fused" or "split". "auto" picks "fused" only when the widest
    dependency level has at most FUSED_MAX_LEVEL_WIDTH cells; explicit modes are honoured at any width."""
    if mode not in MODES:
        raise CudaHydrologyPreparationError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "auto":
        return "fused" if int(max_level_width) <= FUSED_MAX_LEVEL_WIDTH else "split"
    return mode


def launch_count(level_bounds, mode: str) -> int:
    """Kernel launches per attempted step of a RESOLVED mode ("fused" or "split") on a graph with these bounds."""
    if mode == "fused":
        return 1
    if mode == "split":
        return 3 + sum(1 for a, b in itertools.pairwise(level_bounds) if b > a)
    raise CudaHydrologyPreparationError(f"launch_count needs a resolved mode ('fused' or 'split'), got {mode!r}")


# --- lazy optional CuPy / kernels -----------------------------------------------------------------------------------
_MODULE: Any = None
_MODULE_NEWTON: Any = None
_FUNCTIONS: dict[tuple[int, str, bool], Any] = {}
_LOCK = threading.Lock()


def _function(name: str, newton: bool = False):
    """The compiled kernel `name` for the CURRENT device (module created lazily; compilation at first load). `newton`
    selects the Newton step-kernel variant (separate module); the default is the unchanged bisection module."""
    global _MODULE, _MODULE_NEWTON
    cp = routing_cuda._cupy()
    device_id = routing_cuda._current_device_id(cp)
    key = (device_id, name, newton)
    fn = _FUNCTIONS.get(key)
    if fn is not None:
        return fn
    with _LOCK:
        try:
            if newton:
                if name not in _STEP_KERNEL_NAMES:
                    raise KeyError(f"{name!r} has no Newton variant")
                if _MODULE_NEWTON is None:
                    _MODULE_NEWTON = cp.RawModule(code=_SOURCE_NEWTON, options=COMPILE_OPTIONS, backend="nvrtc")
                module = _MODULE_NEWTON
            else:
                if _MODULE is None:
                    _MODULE = cp.RawModule(code=_SOURCE, options=COMPILE_OPTIONS, backend="nvrtc")
                module = _MODULE
            fn = module.get_function(name)
            fn.attributes  # noqa: B018 - forces the module load for this device so failures surface now
        except Exception as exc:
            raise CudaUnavailableError(
                f"CUDA hydrology kernel {name!r} failed to compile/load ({type(exc).__name__}: {exc}); "
                "no fallback") from exc
        _FUNCTIONS[key] = fn
    return fn


def _load_all(newton: bool = False) -> dict[str, dict[str, int]]:
    attributes = {}
    for name in (_STEP_KERNEL_NAMES if newton else _KERNEL_NAMES):
        fn = _function(name, True) if newton else _function(name)  # the default call shape is unchanged
        attrs = dict(fn.attributes)
        limit = int(attrs.get("max_threads_per_block", 0))
        need = REDUCE_THREADS if name.endswith(("reduce", "report")) else max(FUSED_THREADS, CELL_THREADS)
        if limit < need:
            raise CudaUnavailableError(f"kernel {name} supports only {limit} threads per block, {need} are needed "
                                       f"(register pressure); no fallback")
        attributes[name] = {k: int(v) for k, v in attrs.items() if isinstance(v, (int, np.integer))}
    return attributes


_NEWTON_ATTRIBUTES: dict[int, dict[str, dict[str, int]]] = {}


def load_newton_kernels() -> dict[str, Any]:
    """Compile/load the Newton step-kernel variant for the CURRENT device (idempotent; startup, outside any timed
    evolution) and return `{"seconds", "attributes"}`. Missing CuPy/device/compile failure: `CudaUnavailableError`,
    including a kernel that cannot run the fused block size."""
    cp = routing_cuda._cupy()
    device_id = routing_cuda._current_device_id(cp)
    t0 = time.perf_counter()
    with _LOCK:
        cached = _NEWTON_ATTRIBUTES.get(device_id)
    if cached is None:
        cached = _load_all(True)
        with _LOCK:
            _NEWTON_ATTRIBUTES[device_id] = cached
    return {"seconds": time.perf_counter() - t0, "attributes": cached}


def kernel_provenance() -> dict[str, Any]:
    """Source hash, options, launch structures and toolchain/device (reuses the routing provenance; no compilation)."""
    info = routing_cuda.kernel_provenance()
    info.update({
        "newton_hydrology_source_sha256": hashlib.sha256(_SOURCE_NEWTON.encode()).hexdigest(),
        "newton_device_source_sha256": hashlib.sha256(routing_newton_cuda.newton_device_source().encode()).hexdigest(),
        "newton_step_kernels": list(_STEP_KERNEL_NAMES),
        "root_solvers": {"bisection": "default module (unchanged source)",
                         "newton": "explicit step-kernel variant; same device helper as routing_newton_cuda; "
                                   "uninstrumented (no counters, no extra transfer)"},
        "module": "maple_syrup.hydrology_cuda",
        "hydrology_source_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest(),
        "kernels": list(_KERNEL_NAMES),
        "modes": list(MODES),
        "fused_threads": FUSED_THREADS, "cell_threads": CELL_THREADS, "reduce_threads": REDUCE_THREADS,
        "fused_max_level_width": FUSED_MAX_LEVEL_WIDTH,
        "packet_words": PACKET_WORDS, "packet_bytes": PACKET_WORDS * 8,
        "reductions": "thread partials in index order + fixed shared-memory pairwise tree; deterministic, not "
                      "NumPy's summation order; no atomics",
        "libm": "device expm1/pow (may differ from NumPy by an ulp; declared bound rtol 2e-12 / atol 1e-14)",
        "numba_required": False,
    })
    return info


# --- context ------------------------------------------------------------------------------------------------------
_OWNED = ("active", "outlet", "conveyance", "level_order", "donor_cell", "level_bounds_device", "column_static")


@dataclass(frozen=True, eq=False)
class CudaHydrologyContext:
    """Owned, validated static DEVICE data of a fixed-terrain replay. Build with `prepare_cuda_hydrology`. Holds no
    reference to the caller's graph or parameters; the device arrays are immutable by contract (do not write them)."""

    device_id: int
    shape: tuple[int, int]
    n_cells: int
    n_active: int
    dx_m: float
    model: str
    model_code: int
    requested_mode: str
    mode: str  # resolved: "fused" or "split"
    level_bounds: tuple[int, ...]
    max_level_width: int
    active: Any  # (n_cells,) uint8
    outlet: Any  # (n_cells,) uint8
    conveyance: Any  # (n_cells,) float64
    level_order: Any  # (n_active,) int64
    donor_cell: Any  # (4, n_active) int64: flat cell index of the donor in DONOR_SLOTS order, -1 = none
    level_bounds_device: Any  # (n_levels + 1,) int64
    column_static: Any  # (n_cells, 6) float64: ksat, suction, drainage, thickness, Smax, lambda
    graph_input_sha256: str
    static_bytes: int
    preparation_s: float
    kernel_load_s: float
    host_to_device_bytes: int
    device_to_host_bytes: int
    kernel_attributes: dict
    owned_fingerprints: tuple
    scalar_signature: tuple  # launch-dependent host metadata sealed at preparation (see `_scalar_signature`)
    source_refs: tuple  # (weakref to the graph, weakref to the params) PREPARED from; never strong references
    source_fingerprints: tuple  # pointer/shape/dtype/contiguity/device of the source arrays at preparation

    @property
    def n_levels(self) -> int:
        return len(self.level_bounds) - 1

    def is_bound_to(self, graph: Any, params: Any) -> bool:
        """True only if `graph` and `params` are the very objects this context was prepared from (identity, through
        weak references, so a freed-and-recycled address cannot pass) and the metadata (pointer, shape, dtype,
        contiguity, device) of their arrays is unchanged. Cheap, no device access, no content hash: an in-place
        content change of the source arrays after preparation is NOT detected (the context owns static copies and is
        immutable by contract; prepare a new one). Does not extend the lifetime of the graph or the parameters."""
        graph_ref, params_ref = self.source_refs
        if graph_ref() is not graph or params_ref() is not params:
            return False
        try:
            graph_fp = tuple(_fingerprint_attr(graph, name) for name in routing_cuda._GRAPH_RUNTIME_FIELDS)
            params_fp = tuple((name, _fingerprint_attr(params, name)) for name, _ in self.source_fingerprints[1])
        except (AttributeError, TypeError, ValueError):
            return False
        return (graph_fp, params_fp) == self.source_fingerprints

    def nbytes(self) -> int:
        return int(self.static_bytes)

    def summary(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape), "n_cells": self.n_cells, "n_active": self.n_active, "dx_m": self.dx_m,
            "infiltration_model": self.model, "n_levels": self.n_levels, "max_level_width": self.max_level_width,
            "requested_mode": self.requested_mode, "mode": self.mode,
            "launches_per_step": launch_count(self.level_bounds, self.mode),
            "packet_bytes": PACKET_WORDS * 8,
            "device_id": self.device_id, "static_bytes": self.static_bytes,
            "preparation_s": self.preparation_s, "kernel_load_s": self.kernel_load_s,
            "host_to_device_bytes": self.host_to_device_bytes, "device_to_host_bytes": self.device_to_host_bytes,
            "kernel_attributes": self.kernel_attributes,
            "graph_input_sha256": self.graph_input_sha256,
            "ownership": "owned device copies; no alias of the caller's graph/parameter arrays; content mutation "
                         "of the owned arrays is not detected",
            "layout": "flat cell-major; column_static (n_cells, 6) = ksat, suction, drainage, thickness, Smax, "
                      "lambda; donor_cell (4, n_active) flat cell indices in level order",
            "device_resident": True, "fastmath": False, "numba_required": False,
        }


def _arrays_of(ctx: CudaHydrologyContext) -> tuple:
    return tuple(getattr(ctx, name) for name in _OWNED)


def _fingerprint_attr(owner: Any, name: str) -> tuple:
    array = getattr(owner, name)
    if not hasattr(array, "data") or not hasattr(array.data, "ptr"):
        raise TypeError(f"{name} is not a device array")
    return routing_cuda._fingerprint(array)


def _scalar_signature(ctx: CudaHydrologyContext) -> tuple:
    """Every host-side field that reaches a kernel launch as a count, extent, scale, model code or launch structure.
    The raw kernels do no bounds checking, so a context whose `n_cells`/`n_active`/`level_bounds`/... were forged with
    `dataclasses.replace` (the owned device arrays alone would still look valid) must be refused BEFORE any launch.
    `shape` and `level_bounds` must be TUPLES (immutable context metadata: a list that merely equals the tuple is
    refused, it could be mutated after the check). The tuple is compared by value (a few hundred ints at most, no
    device access, no hashing of device content). Raises `TypeError` for non-tuples (the caller turns it into a refusal)."""
    if not isinstance(ctx.shape, tuple) or not isinstance(ctx.level_bounds, tuple):
        raise TypeError("shape and level_bounds must be tuples")
    return (ctx.device_id, ctx.shape, ctx.n_cells, ctx.n_active, ctx.dx_m, ctx.model, ctx.model_code,
            ctx.requested_mode, ctx.mode, ctx.level_bounds, ctx.max_level_width)


def _expected_shapes(ctx: CudaHydrologyContext) -> tuple:
    return ((ctx.n_cells,), (ctx.n_cells,), (ctx.n_cells,), (ctx.n_active,), (4, ctx.n_active),
            (len(ctx.level_bounds),), (ctx.n_cells, 6))


def _consistent(ctx: CudaHydrologyContext) -> bool:
    """The sealed counts must also agree with each other and with the extents of the owned device arrays (a forger who
    recomputes the signature still cannot make forged counts fit arrays whose fingerprints are fixed)."""
    bounds = ctx.level_bounds
    return bool(
        len(ctx.shape) == 2 and ctx.shape[0] >= 1 and ctx.shape[1] >= 1 and ctx.shape[0] * ctx.shape[1] == ctx.n_cells
        and 1 <= ctx.n_active <= ctx.n_cells and len(bounds) >= 2 and bounds[0] == 0 and bounds[-1] == ctx.n_active
        and all(a <= b for a, b in itertools.pairwise(bounds))
        and ctx.max_level_width == max(b - a for a, b in itertools.pairwise(bounds))
        and ctx.mode in ("fused", "split") and ctx.requested_mode in MODES
        and _MODEL_CODES.get(ctx.model) == ctx.model_code
        and isinstance(ctx.dx_m, float) and math.isfinite(ctx.dx_m) and ctx.dx_m > 0.0
        and tuple(tuple(a.shape) for a in _arrays_of(ctx)) == _expected_shapes(ctx))


def _prep_error(exc: BaseException) -> CudaHydrologyPreparationError:
    return CudaHydrologyPreparationError(str(exc))


def prepare_cuda_hydrology(graph, params, *, mode: str = "auto") -> CudaHydrologyContext:
    """Validate the CuPy graph and column parameters once and build the owned device context for the CURRENT device.
    Static arrays are downloaded ONCE through counted transfers for validation (graph checks are
    `routing_cuda._validate_host_graph`, including the donor-slot geometry; the column checks are those of
    `hydrology_numba.prepare_hydrology`), owned copies are uploaded, kernels are compiled/loaded, and the current
    stream is synchronized once. Raises `CudaUnavailableError` (no CuPy/device/compiler) or
    `CudaHydrologyPreparationError` before anything is returned; there is no NumPy/Numba fallback."""
    from maple.core.backend import read_transfer_counters, to_device, to_host

    from maple_syrup.infiltration import ColumnParameters
    from maple_syrup.routing import RoutingGraph

    t0 = time.perf_counter()
    if mode not in MODES:
        raise CudaHydrologyPreparationError(f"mode must be one of {MODES}, got {mode!r}")
    cp = routing_cuda._cupy()
    device_id = routing_cuda._current_device_id(cp)
    if not isinstance(graph, RoutingGraph):
        raise CudaHydrologyPreparationError("graph must be a RoutingGraph (use build_routing_graph)")
    if not isinstance(params, ColumnParameters):
        raise CudaHydrologyPreparationError("params must be a ColumnParameters (use column_parameters)")
    if graph.xp is not cp or params.xp is not cp:
        raise CudaHydrologyPreparationError(
            f"the CUDA hydrology needs a CuPy graph and CuPy column parameters; the graph is in "
            f"{getattr(graph.xp, '__name__', graph.xp)!r} and the parameters in "
            f"{getattr(params.xp, '__name__', params.xp)!r}; no host/device transfer is made")
    shape = tuple(graph.shape)
    if len(shape) != 2 or min(shape) < 1 or not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                                                     for v in shape):
        raise CudaHydrologyPreparationError(f"graph shape must be a positive (ny, nx), got {graph.shape!r}")
    ny, nx = int(shape[0]), int(shape[1])
    n = ny * nx
    if tuple(params.shape) != shape:
        raise CudaHydrologyPreparationError(f"column parameter shape {tuple(params.shape)} != graph shape {shape}")
    dx = graph.dx_m
    if isinstance(dx, bool) or not isinstance(dx, (int, float, np.integer, np.floating)):
        raise CudaHydrologyPreparationError(f"graph.dx_m must be a real number, got {type(dx).__name__}")
    dx = float(dx)
    if not (math.isfinite(dx) and dx > 0.0 and math.isfinite(dx * dx) and dx * dx > 0.0):
        raise CudaHydrologyPreparationError(f"graph.dx_m must be finite with a finite positive cell area, got {dx!r}")
    if params.model not in INFILTRATION_MODELS:
        raise CudaHydrologyPreparationError(f"unsupported infiltration model {params.model!r}")

    # structure, dtype, shape, contiguity and device of every runtime array BEFORE any transfer
    try:
        graph_arrays = routing_cuda._graph_arrays(cp, graph, device_id)
        names = {"ksat_m_per_s": np.float64, "suction_m": np.float64, "drainage_parameter": np.float64,
                 "theta_sat": np.float64, "soil_thickness_m": np.float64, "storage_max_m": np.float64,
                 "active_mask": np.bool_}
        if params.model == "pavement_hawkins":
            if params.lambda_m_per_s is None:
                raise CudaHydrologyPreparationError("model 'pavement_hawkins' needs params.lambda_m_per_s")
            names["lambda_m_per_s"] = np.float64
        for name, dtype in names.items():
            routing_cuda._check_device_array(cp, getattr(params, name), f"params.{name}", dtype, shape, device_id)
    except CudaUnavailableError:
        raise
    except RoutingError as exc:
        raise _prep_error(exc) from exc

    t_load = time.perf_counter()
    attributes = _load_all()  # compile/load now: a compiler failure precedes every transfer
    kernel_load_s = time.perf_counter() - t_load
    before = read_transfer_counters()
    host = {nm: to_host(graph_arrays[nm]) for nm in routing_cuda._GRAPH_RUNTIME_FIELDS}
    try:
        bounds = routing_cuda._validate_host_graph(
            graph, host["active_flat"], host["outlet_flat"], host["conveyance"], host["level_order"],
            host["conveyance_lo"], host["donor_position"], host["donor_mask"])
    except RoutingError as exc:
        raise _prep_error(exc) from exc
    active = host["active_flat"]
    n_active = int(np.count_nonzero(active))
    ph = {name: to_host(getattr(params, name)).reshape(-1) for name in names}
    for name in ("ksat_m_per_s", "suction_m", "drainage_parameter"):
        if not (np.all(np.isfinite(ph[name])) and np.all(ph[name] >= 0.0)):
            raise CudaHydrologyPreparationError(f"params.{name} must be finite and >= 0 everywhere")
    theta, thick, smax = ph["theta_sat"], ph["soil_thickness_m"], ph["storage_max_m"]
    if not (np.all(np.isfinite(theta)) and np.all(theta > 0.0) and np.all(theta <= 1.0)):
        raise CudaHydrologyPreparationError("params.theta_sat must lie in (0, 1] everywhere")
    if not (np.all(np.isfinite(thick)) and np.all(thick > 0.0)):
        raise CudaHydrologyPreparationError("params.soil_thickness_m must be finite and > 0 everywhere")
    if not (np.all(np.isfinite(smax)) and np.all(smax > 0.0) and np.array_equal(smax, theta * thick)):
        raise CudaHydrologyPreparationError(
            "params.storage_max_m must be finite, > 0 and equal theta_sat * soil_thickness")
    if params.model == "pavement_hawkins":
        lam = ph["lambda_m_per_s"]
        if not (np.all(np.isfinite(lam)) and np.all(lam > 0.0)):
            raise CudaHydrologyPreparationError("params.lambda_m_per_s must be finite and > 0 everywhere")
    else:
        lam = np.zeros(n, dtype=np.float64)
    if not np.array_equal(ph["active_mask"], active):
        raise CudaHydrologyPreparationError(
            "column active_mask differs from the routing graph's active cells; unsupported")

    order = np.ascontiguousarray(host["level_order"], dtype=np.int64)
    donor_cell = np.where(host["donor_mask"], order[host["donor_position"]], -1).astype(np.int64)
    column_static = np.empty((n, 6), dtype=np.float64)
    for j, array in enumerate((ph["ksat_m_per_s"], ph["suction_m"], ph["drainage_parameter"], thick, smax, lam)):
        column_static[:, j] = array
    owned = {
        "active": to_device(np.ascontiguousarray(active.astype(np.uint8)), cp),
        "outlet": to_device(np.ascontiguousarray(host["outlet_flat"].astype(np.uint8)), cp),
        "conveyance": to_device(np.ascontiguousarray(host["conveyance"], dtype=np.float64), cp),
        "level_order": to_device(order, cp),
        "donor_cell": to_device(np.ascontiguousarray(donor_cell), cp),
        "level_bounds_device": to_device(np.array(bounds, dtype=np.int64), cp),
        "column_static": to_device(np.ascontiguousarray(column_static), cp),
    }
    cp.cuda.get_current_stream().synchronize()  # uploads complete before any other stream may use them
    delta = read_transfer_counters().delta(before)
    widths = [b - a for a, b in itertools.pairwise(bounds)]
    max_width = int(max(widths))
    resolved = select_mode(max_width, mode)
    fields = {
        "device_id": device_id, "shape": (ny, nx), "n_cells": n, "n_active": n_active, "dx_m": dx,
        "model": params.model, "model_code": _MODEL_CODES[params.model], "requested_mode": mode, "mode": resolved,
        "level_bounds": tuple(bounds), "max_level_width": max_width,
    }
    signature = (device_id, (ny, nx), n, n_active, dx, params.model, _MODEL_CODES[params.model], mode, resolved,
                 tuple(bounds), max_width)
    return CudaHydrologyContext(
        **fields, graph_input_sha256=str(graph.input_sha256),
        static_bytes=int(sum(a.nbytes for a in owned.values())), preparation_s=time.perf_counter() - t0,
        kernel_load_s=kernel_load_s, host_to_device_bytes=int(delta.host_to_device_bytes),
        device_to_host_bytes=int(delta.device_to_host_bytes), kernel_attributes=attributes,
        owned_fingerprints=tuple(routing_cuda._fingerprint(a) for a in owned.values()),
        scalar_signature=signature,
        source_refs=(weakref.ref(graph), weakref.ref(params)),
        source_fingerprints=(
            tuple(routing_cuda._fingerprint(graph_arrays[nm]) for nm in routing_cuda._GRAPH_RUNTIME_FIELDS),
            tuple((name, routing_cuda._fingerprint(getattr(params, name))) for name in names)),
        **owned,
    )


prepared_cuda_hydrology = prepare_cuda_hydrology


# --- one coupled step -----------------------------------------------------------------------------------------
def _check_context(cp, ctx: Any) -> None:
    if not isinstance(ctx, CudaHydrologyContext):
        raise CudaHydrologyPreparationError("ctx must be a CudaHydrologyContext (use prepare_cuda_hydrology)")
    device_id = routing_cuda._current_device_id(cp)
    if ctx.device_id != device_id:
        raise RoutingError(f"the context was prepared on CUDA device {ctx.device_id} but the current device is "
                           f"{device_id}; prepare again on purpose (no peer access or migration)")
    if ctx.owned_fingerprints != tuple(routing_cuda._fingerprint(a) for a in _arrays_of(ctx)):
        raise RoutingError("the context's owned device arrays no longer match the prepared metadata")
    try:
        current = _scalar_signature(ctx)
    except (TypeError, ValueError):
        current = None  # a forged non-sequence shape/level_bounds cannot equal the sealed tuple
    if current is None or ctx.scalar_signature != current:
        raise RoutingError("the context's scalar metadata (counts, extents, dx, model, launch structure, level "
                           "bounds) no longer matches the values sealed at preparation; refusing before any launch")
    if not _consistent(ctx):
        raise RoutingError("the context's owned array extents are inconsistent with its counts, shape, model and "
                           "level bounds")


def _dynamic(cp, array: Any, name: str, shape: tuple[int, int], error: type[Exception], device_id: int):
    if type(array) is not cp.ndarray:
        raise error(f"{name} must be an exact cupy.ndarray (the CUDA hydrology never transfers, converts or accepts "
                    f"NumPy arrays, lists or subclasses), got {type(array).__name__}")
    if int(array.device.id) != device_id:
        raise error(f"{name} lives on CUDA device {int(array.device.id)} but the current device is {device_id}; "
                    "no peer access or migration is performed")
    if tuple(array.shape) != shape:
        raise error(f"{name} shape {tuple(array.shape)} != {shape}")
    if array.dtype != np.float64:
        raise error(f"{name} must be float64, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise error(f"{name} must be C-contiguous (no copy is made)")
    return array


def _launch(name: str, grid: int, block: int, args: tuple, newton: bool = False) -> None:
    try:
        fn = _function(name, True) if newton else _function(name)  # the default call shape is unchanged
        fn((grid,), (block,), args)
    except CudaUnavailableError:
        raise
    except Exception as exc:  # enqueue-time failure; asynchronous faults surface at the packet read
        raise RoutingError(f"CUDA hydrology kernel {name} launch failed ({type(exc).__name__}: {exc}); "
                           "no fallback") from exc


_F64_ROWS = ("depth_new", "soil_new", "rain", "intake", "overflow", "drainage", "hpre", "q_old", "h_new", "flow",
             "q_new", "qin_new", "qin_old", "velocity", "face", "cr_row", "cons_row", "bal_row", "sc_row")


def _make_args(ctx: CudaHydrologyContext, h, s, r, qprev, b, *, iterations: int, stage: int, dt: float, dt_r: float,
               cr_max: float, root_tol: float) -> tuple:
    """The ONE place where the kernel argument list (the `STEP_ARGS` order) is assembled. `b` holds the per-call
    buffers (`_F64_ROWS`, `cflag`, `rflag`, `br`, `packet`); `dt` feeds the column, `dt_r` the routing scales."""
    f64 = np.float64
    dx = ctx.dx_m
    area = dx * dx
    return (
        np.int64(ctx.n_cells), np.int64(ctx.n_active), np.int32(ctx.n_levels),
        ctx.active, ctx.outlet, ctx.conveyance, ctx.column_static, ctx.level_order, ctx.donor_cell,
        ctx.level_bounds_device, h, s, r, qprev,
        b.depth_new, b.soil_new, b.rain, b.intake, b.overflow, b.drainage, b.hpre, b.q_old, b.h_new, b.flow,
        b.q_new, b.qin_new, b.qin_old, b.velocity, b.face, b.cflag, b.rflag, b.br, b.cr_row, b.cons_row,
        b.bal_row, b.sc_row, b.packet,
        np.int32(ctx.model_code), np.int32(iterations), np.int32(stage),
        f64(dt), f64(LOCAL_BALANCE_RTOL), f64(dt_r / (2.0 * dx)), f64(dt_r / dx), f64(area), f64(dx), f64(cr_max),
        f64(root_tol), f64(BALANCE_RTOL), f64(_TWO_THIRDS), f64(_EPS64),
        f64(BALANCE_RTOL * (ctx.n_active + 2) * area),
    )


def _dispatch(ctx: CudaHydrologyContext, args: tuple, stage: int, newton: bool = False) -> None:
    """Enqueue the kernels of the context's launch structure. `stage` 0 = column stage + column-flag reduction only
    (no sweep, no post), 1 = the whole coupled step."""
    grid_cells = (ctx.n_cells + CELL_THREADS - 1) // CELL_THREADS
    if ctx.mode == "fused":
        _launch("maple_syrup_hydro_fused", 1, FUSED_THREADS, args, newton)
        return
    _launch("maple_syrup_hydro_pre", grid_cells, CELL_THREADS, args, newton)
    if stage == 1:
        for b0, b1 in itertools.pairwise(ctx.level_bounds):
            m = b1 - b0
            if m:
                _launch("maple_syrup_hydro_solve_level", (m + CELL_THREADS - 1) // CELL_THREADS, CELL_THREADS,
                        (*args, np.int64(b0), np.int64(m)), newton)
        _launch("maple_syrup_hydro_post", grid_cells, CELL_THREADS, args, newton)
    _launch("maple_syrup_hydro_reduce", 1, REDUCE_THREADS, args, newton)


def prepared_coupled_step(ctx: CudaHydrologyContext, rain_rate_m_per_s: Any, state: StormState, dt_s: float,
                          control: StormControl) -> CoupledStep:
    """`storm.coupled_step` / `hydrology_numba.prepared_coupled_step` on the prepared device context. Inputs are exact
    CuPy float64 C-contiguous `(ny, nx)` arrays on the current device. Same `CoupledStep` fields, error classes,
    messages and precedence as the CPU prepared step; pure: raises before returning anything and never writes an
    input. `RoutingStepRejected` still means "retry the SAME state with a smaller dt". The control may be a plain
    (not `.validated()`) `StormControl`; only `implementation == "cuda"` is accepted here. Result grids are fresh
    CuPy arrays; result scalars are 0-d CuPy views of the fresh packet. One counted packet read per call."""
    return cuda_step_with_packet(ctx, rain_rate_m_per_s, state, dt_s, control)[0]


def cuda_step_with_packet(ctx: CudaHydrologyContext, rain_rate_m_per_s: Any, state: StormState, dt_s: float,
                          control: StormControl) -> tuple[CoupledStep, Any]:
    """`prepared_coupled_step` that also returns the step's device packet (the fresh `uint64[PACKET_WORDS]` array the
    scalar results are views of). The storm driver passes it to the accumulate/report kernels so no scalar is ever
    read back; nothing else differs."""
    from maple.core.backend import to_host

    cp = routing_cuda._cupy()
    _check_context(cp, ctx)
    if not isinstance(state, StormState):
        raise StormError(f"state must be a StormState, got {type(state).__name__}")
    if not isinstance(control, StormControl):
        raise StormError(f"control must be a StormControl, got {type(control).__name__}")
    dt = _check_dt(dt_s)
    shape = ctx.shape
    device_id = ctx.device_id
    h = _dynamic(cp, state.depth_m, "depth_m", shape, InfiltrationError, device_id)
    s = _dynamic(cp, state.soil_water_m, "soil_water_m", shape, InfiltrationError, device_id)
    r = _dynamic(cp, rain_rate_m_per_s, "rain_rate_m_per_s", shape, InfiltrationError, device_id)
    qprev = _dynamic(cp, state.discharge_m2_s, "state.discharge_m2_s", shape, RoutingError, device_id)

    # Scalar options are checked now but raised only AFTER the column stage (a column failure outranks them).
    option_error: RoutingError | None = None
    cr_max, iterations, root_tol = 1.0, 1, 1.0
    root_solver, newton_cap, kernel_iterations = "bisection", 0, 1
    try:
        dt_r, cr_max, iterations, root_tol = _check_step_options(
            dt_s, control.courant_max, control.bisection_iterations, control.root_tolerance_m,
            control.implementation)
        if control.implementation != "cuda":
            raise RoutingError(f"the prepared CUDA hydrology runs the CUDA kernels only; control.implementation is "
                               f"{control.implementation!r} (select the reference or the numba hydrology for it)")
        root_solver, newton_cap = _check_root_solver(control.root_solver, control.newton_max_iterations,
                                                     control.implementation, cp)
        kernel_iterations = newton_cap if root_solver == "newton" else iterations
    except RoutingError as exc:
        option_error = exc
        dt_r = dt
    stage = 0 if option_error is not None else 1
    newton = root_solver == "newton" and option_error is None
    if newton:  # explicit variant; already loaded by `load_newton_kernels` in a storm, otherwise compiled here
        load_newton_kernels()

    n = ctx.n_cells
    f64 = np.float64
    b = SimpleNamespace(**{name: cp.empty(n, dtype=f64) for name in _F64_ROWS},
                        cflag=cp.empty(n, dtype=np.uint32), rflag=cp.empty(n, dtype=np.uint32),
                        br=cp.empty(n, dtype=np.uint8), packet=cp.empty(PACKET_WORDS, dtype=np.uint64))
    args = _make_args(ctx, h, s, r, qprev, b, iterations=kernel_iterations if stage else 1, stage=stage, dt=dt,
                      dt_r=dt_r, cr_max=cr_max, root_tol=root_tol)
    _dispatch(ctx, args, stage, newton)
    packet = b.packet
    depth_new, soil_new, rain, intake, overflow, drainage = (b.depth_new, b.soil_new, b.rain, b.intake, b.overflow,
                                                              b.drainage)
    q_old, h_new, velocity, face, flow, q_new, qin_new, qin_old = (b.q_old, b.h_new, b.velocity, b.face, b.flow,
                                                                   b.q_new, b.qin_new, b.qin_old)

    words = to_host(packet)  # the ONE counted host read of this attempt
    column_flags = int(words[_P_COL])
    if dt == 0.0:
        column_flags &= _COLUMN_INPUT_MASK
    if column_flags:
        raise InfiltrationError(_COLUMN_MESSAGES[_lowest_bit(column_flags)])
    if option_error is not None:
        raise option_error
    nonfinite = int(words[_P_NONFINITE])
    scalar_nonfinite = tuple(bool((nonfinite >> k) & 1) for k in range(5))
    _resolve_route(int(words[_P_ROUTE]), scalar_nonfinite, bool(words[_P_RESFAIL]), int(words[_P_Q]), cr_max,
                   root_tol, iterations, newton_cap if newton else 0)

    grid = shape
    as_f64, as_i64 = packet.view(np.float64), packet.view(np.int64)
    route = RouteStep(
        dt_s=dt_r,
        depth_m=h_new.reshape(grid),
        flow_depth_m=flow.reshape(grid),
        discharge_m2_s=q_new.reshape(grid),
        velocity_m_s=velocity.reshape(grid),
        inflow_m2_s=qin_new.reshape(grid),
        old_discharge_m2_s=q_old.reshape(grid),
        old_inflow_m2_s=qin_old.reshape(grid),
        face_volume_m3=face.reshape(grid),
        export_m3=as_f64[_P_EXPORT],
        outlet_discharge_m3_s=as_f64[_P_OUTLET],
        storage_change_m3=as_f64[_P_STORAGE],
        budget_residual_m3=as_f64[_P_RESIDUAL],
        stale_inflow_gain_m3=None,
        max_courant_old=as_f64[_P_CR_OLD],
        max_courant_new=as_f64[_P_CR_NEW],
        max_constitutive_residual_m=as_f64[_P_CONS],
        max_cell_balance_residual_m=as_f64[_P_BAL],
        conservative=True,
        bisection_iterations=0 if newton else iterations,
        implementation="cuda",
        root_solver=root_solver,
        newton_max_iterations=newton_cap if newton else 0,
        root_stats=None,  # uninstrumented device production: no counters are read back (documented, not zeros)
    )
    col = ColumnStep(dt, depth_new.reshape(grid), soil_new.reshape(grid), rain.reshape(grid), intake.reshape(grid),
                     overflow.reshape(grid), drainage.reshape(grid))
    new_state = StormState(state.t_s + float(dt_s), route.depth_m, col.soil_water_m, route.discharge_m2_s)
    return CoupledStep(
        dt_s=float(dt_s), state=new_state, column=col, route=route,
        n_no_runon=as_i64[_P_NNO], n_partial_runon=as_i64[_P_NPARTIAL], n_complete_runon=as_i64[_P_NCOMPLETE],
    ), packet


cuda_coupled_step = prepared_coupled_step


# --- column alone ---------------------------------------------------------------------------------------------------
def prepared_column_step(ctx: CudaHydrologyContext, depth_m: Any, soil_water_m: Any, rain_rate_m_per_s: Any,
                         dt_s: float) -> ColumnStep:
    """`infiltration.column_step(validate=True)` on the prepared device columns: the very `pre_cell` device function
    of the coupled step (no second copy of the Smith-Parlange / drainage / saturation-return formulas), the column
    stage only. Same `ColumnStep`, same `InfiltrationError` conditions, messages and precedence; `dt_s = 0` is the
    identity (inputs still validated). It applies NO routing check: no old-flux, `hpre`/`h*`, Courant or
    previous-discharge guard (those belong to the coupled step), so it is the reusable column physics for solvers
    other than the Crank-Nicolson sweep. Pure: inputs and the context are never written; every output is fresh.
    Exact CuPy float64 C-contiguous `(ny, nx)` inputs on the current device. One counted 144-byte packet read."""
    from maple.core.backend import to_host

    cp = routing_cuda._cupy()
    _check_context(cp, ctx)
    dt = _check_dt(dt_s)
    shape = ctx.shape
    h = _dynamic(cp, depth_m, "depth_m", shape, InfiltrationError, ctx.device_id)
    s = _dynamic(cp, soil_water_m, "soil_water_m", shape, InfiltrationError, ctx.device_id)
    r = _dynamic(cp, rain_rate_m_per_s, "rain_rate_m_per_s", shape, InfiltrationError, ctx.device_id)
    n = ctx.n_cells
    f64 = np.float64
    keep = {name: cp.empty(n, dtype=f64) for name in ("depth_new", "soil_new", "rain", "intake", "overflow",
                                                       "drainage")}
    junk = cp.empty(n, dtype=f64)  # route-side outputs of the shared cell function: written, never read (stage 0)
    junk_flags = cp.empty(n, dtype=np.uint32)
    scratch = {name: junk for name in _F64_ROWS if name not in keep}
    b = SimpleNamespace(**keep, **scratch, cflag=cp.empty(n, dtype=np.uint32), rflag=junk_flags,
                        br=cp.empty(n, dtype=np.uint8), packet=cp.empty(PACKET_WORDS, dtype=np.uint64))
    qprev = cp.zeros(n, dtype=f64)
    args = _make_args(ctx, h, s, r, qprev, b, iterations=1, stage=0, dt=dt, dt_r=dt, cr_max=1.0, root_tol=1.0)
    _dispatch(ctx, args, 0)
    flags = int(to_host(b.packet)[_P_COL])  # the ONE counted host read
    if dt == 0.0:
        flags &= _COLUMN_INPUT_MASK
    if flags:
        raise InfiltrationError(_COLUMN_MESSAGES[_lowest_bit(flags)])
    if dt == 0.0:
        return ColumnStep(0.0, cp.array(h, copy=True), cp.array(s, copy=True), cp.zeros(shape, dtype=f64),
                          cp.zeros(shape, dtype=f64), cp.zeros(shape, dtype=f64), cp.zeros(shape, dtype=f64))
    return ColumnStep(dt, keep["depth_new"].reshape(shape), keep["soil_new"].reshape(shape),
                      keep["rain"].reshape(shape), keep["intake"].reshape(shape), keep["overflow"].reshape(shape),
                      keep["drainage"].reshape(shape))


# --- driver-owned accumulation / reporting (used by storm.evolve with control.implementation == "cuda") ----------------
_ACC_CUM_EXPORT, _ACC_PEAK_Q, _ACC_PEAK_T, _ACC_BAL, _ACC_CONS, _ACC_CR_OLD, _ACC_CR_NEW = range(7)


class CudaStormAccumulator:
    """Device-resident accumulation of one `storm.evolve` call. The host scheduler (boundaries, retries, guards)
    stays in `storm.evolve`; this object owns only a small scalar block (`7` doubles + `3` int64 on the device) and
    updates the driver's own cumulative/peak GRIDS in place with ONE launch per accepted step (`accept`) and writes a
    hydrograph row with ONE launch (`report`). Neither reads anything back. The grids are created by `evolve` for this
    call, so no caller array is ever written and results stay fresh public outputs; after `evolve` returns this
    object is discarded and `scalars()` views stay valid (they keep the scalar block alive)."""

    def __init__(self, ctx: CudaHydrologyContext, *, cum_rain, cum_intake, cum_return, cum_drain, peak_depth,
                 peak_velocity, peak_q, peak_t):
        cp = routing_cuda._cupy()
        _check_context(cp, ctx)
        self._ctx = ctx
        self._cp = cp
        self._grids = tuple(_dynamic(cp, g, name, ctx.shape, StormError, ctx.device_id) for name, g in (
            ("cumulative_rain", cum_rain), ("cumulative_intake", cum_intake), ("cumulative_return", cum_return),
            ("cumulative_drainage", cum_drain), ("peak_depth", peak_depth), ("peak_velocity", peak_velocity)))
        for name, value in (("peak_q", peak_q), ("peak_t", peak_t)):  # initial 0-d device scalars (no host read)
            if type(value) is not cp.ndarray or value.shape != () or value.dtype != np.float64 \
                    or int(value.device.id) != ctx.device_id:
                raise StormError(f"{name} must be a 0-d float64 CuPy array on the context's device")
        self._acc = cp.zeros(7, dtype=np.float64)
        self._acc_i = cp.zeros(3, dtype=np.int64)
        self._acc[_ACC_PEAK_Q] = peak_q
        self._acc[_ACC_PEAK_T] = peak_t
        # metadata of everything a launch will read or write through a raw pointer, re-verified on every public entry
        self._fingerprints = tuple(routing_cuda._fingerprint(a) for a in (*self._grids, self._acc, self._acc_i))
        self.n_accumulations = 0
        self.n_reports = 0

    def _guard(self) -> None:
        """Every public entry: current device and sealed context metadata (`_check_context`), and the pointer/shape/
        dtype/contiguity/device of the driver's own grids and scalar blocks, BEFORE any enqueue. Content is not read."""
        _check_context(self._cp, self._ctx)
        arrays = (*self._grids, self._acc, self._acc_i)
        if self._fingerprints != tuple(routing_cuda._fingerprint(a) for a in arrays):
            raise StormError("the accumulator's device arrays no longer match their metadata; refusing to launch")

    def _packet(self, packet: Any) -> Any:
        cp = self._cp
        if type(packet) is not cp.ndarray or packet.dtype != np.uint64 or tuple(packet.shape) != (PACKET_WORDS,) \
                or not packet.flags.c_contiguous or int(packet.device.id) != self._ctx.device_id:
            raise StormError(f"packet must be the step's C-contiguous uint64[{PACKET_WORDS}] CuPy array on the "
                             "context's device")
        return packet

    @staticmethod
    def _real(value: Any, name: str, *, minimum: float | None = None) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise StormError(f"{name} must be a real host number, got {type(value).__name__}")
        value = float(value)
        if not math.isfinite(value) or (minimum is not None and value < minimum):
            raise StormError(f"{name} must be finite" + ("" if minimum is None else f" and >= {minimum}")
                             + f", got {value!r}")
        return value

    @staticmethod
    def _count(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise StormError(f"{name} must be a non-negative int, got {value!r}")
        return int(value)

    def accept(self, step: CoupledStep, packet: Any, t_s: float) -> None:
        """Fold one ACCEPTED step in (rejected attempts are never passed). `packet` is the step's device packet."""
        self._guard()
        cp, ctx = self._cp, self._ctx
        col, route = step.column, step.route
        arrays = [_dynamic(cp, a, name, ctx.shape, StormError, ctx.device_id) for name, a in (
            ("rain", col.rain_m), ("intake", col.intake_m), ("saturation_return", col.saturation_return_m),
            ("drainage", col.drainage_m), ("depth", route.depth_m), ("velocity", route.velocity_m_s))]
        self._packet(packet)
        t_s = self._real(t_s, "t_s", minimum=0.0)
        n = ctx.n_cells
        _launch("maple_syrup_storm_accumulate", (n + CELL_THREADS - 1) // CELL_THREADS, CELL_THREADS,
                (np.int64(n), *self._grids, *arrays, self._acc, self._acc_i, packet, np.float64(t_s)))
        self.n_accumulations += 1

    def report(self, hydrograph: Any, row: int, state: StormState, last_velocity: Any, packet: Any, t_s: float,
               n_steps: int, n_rejected: int, dt_min: float) -> None:
        """Write hydrograph row `row` (all `len(HYDROGRAPH_COLUMNS)` columns) with one single-block launch."""
        from maple_syrup.storm import HYDROGRAPH_COLUMNS

        self._guard()
        cp, ctx = self._cp, self._ctx
        if type(hydrograph) is not cp.ndarray or hydrograph.dtype != np.float64 or hydrograph.ndim != 2 \
                or hydrograph.shape[1] != len(HYDROGRAPH_COLUMNS) or not hydrograph.flags.c_contiguous \
                or int(hydrograph.device.id) != ctx.device_id:
            raise StormError("hydrograph must be a C-contiguous float64 (rows, columns) CuPy array on the device")
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)) or not 0 <= int(row) < hydrograph.shape[0]:
            raise StormError(f"hydrograph row {row!r} must be an int in [0, {hydrograph.shape[0]})")
        row = int(row)
        self._packet(packet)
        t_s = self._real(t_s, "t_s", minimum=0.0)
        n_steps = self._count(n_steps, "n_steps")
        n_rejected = self._count(n_rejected, "n_rejected")
        dt_min = self._real(dt_min, "dt_min", minimum=0.0)
        depth = _dynamic(cp, state.depth_m, "state.depth_m", ctx.shape, StormError, ctx.device_id)
        soil = _dynamic(cp, state.soil_water_m, "state.soil_water_m", ctx.shape, StormError, ctx.device_id)
        velocity = _dynamic(cp, last_velocity, "last_velocity", ctx.shape, StormError, ctx.device_id)
        f64 = np.float64
        _launch("maple_syrup_storm_report", 1, REDUCE_THREADS,
                (np.int64(ctx.n_cells), *self._grids[:4], depth, soil, velocity, self._acc, packet, hydrograph,
                 np.int64(row), f64(t_s), f64(ctx.dx_m * ctx.dx_m), f64(n_steps), f64(n_rejected), f64(dt_min)))
        self.n_reports += 1

    def scalars(self) -> dict[str, Any]:
        """0-d device views of the accumulated scalars (cumulative export, true peak outlet discharge and its time,
        maxima of the per-step diagnostics, branch cell-step totals)."""
        a, i = self._acc, self._acc_i
        return {"cumulative_export_m3": a[_ACC_CUM_EXPORT], "peak_outlet_discharge_m3_s": a[_ACC_PEAK_Q],
                "time_of_peak_outlet_s": a[_ACC_PEAK_T], "max_balance_residual_m": a[_ACC_BAL],
                "max_constitutive_residual_m": a[_ACC_CONS], "max_courant_old": a[_ACC_CR_OLD],
                "max_courant_new": a[_ACC_CR_NEW], "cell_steps_no_runon": i[0], "cell_steps_partial_runon": i[1],
                "cell_steps_complete_runon": i[2]}
