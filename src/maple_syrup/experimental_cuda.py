"""CUDA forms of the two EXPERIMENTAL hydraulic alternatives (`experimental_hydrology` documents the equations, units,
CFL/positivity bounds, boundary and limiter; this module evaluates the SAME expressions in the SAME order on a CuPy device).

    ctx    = prepare_experimental_cuda(method, graph, params, geometry=None)   # once; validated owned device copies
    solver = CudaHydraulicSolver(method, graph, params, geometry=geometry, control=HydraulicControl(...))
    step   = solver.step(rain, state, dt)                                      # pure; fresh outputs; same HydraulicStep

The column physics is NOT repeated here: every step first calls the accepted `hydrology_cuda.prepared_column_step` (so
infiltration, drainage and saturation return are the baseline device formulas) and the routing context it needs
(`CudaHydrologyContext`) is the baseline one, prepared once, sealed and bound exactly as there. This module adds only the lateral
redistribution kernels, whose launch count is CONSTANT per step (no per-cell Python loop, no ordered dependency sweep, no
bisection): explicit = face kernel, cell kernel, one-block reduction; local inertia = x-face, y-face (+ limiter phi, x/y scale when
`limiter="donor"`), cell kernel, one-block reduction. Every kernel is cell/face parallel; the reduction is a fixed tree
(deterministic, not NumPy's order). FP64 throughout, compiled with `--fmad=false`, no fast math, round-to-nearest intrinsics
only (+ - * / sqrt: no libm transcendental is used here, so apart from the baseline column's `expm1` the device and the CPU
reference agree operation for operation; sums are the only order difference).

Host traffic per attempted step: ONE counted packet read of the column stage (baseline) and ONE counted packet read of the
lateral stage (`PACKET_WORDS` words); no upload, no full-grid download. Static data are downloaded once for validation and
uploaded once as owned copies at preparation. Result scalars are 0-d device views of the fresh per-call packet; every array is fresh. Failures raise
after the packet read and before a result exists; state-input problems (non-finite or invalid face fluxes) are detected on the
device and reported with the CPU messages. Context metadata that reaches a launch (counts, extents, dx, method, names, level
structure of the baseline context) is sealed at preparation and checked, together with the owned arrays' pointer/shape/dtype/device
fingerprints, BEFORE every launch; a forged context is refused before any enqueue. The context is immutable by contract: owned content
mutation is not detected, and `is_bound_to` ties an external context to the very graph/params/geometry objects prepared from.
No NumPy/Numba fallback: a missing CuPy/device/compiler raises `CudaUnavailableError`; Numba is never imported.

Nothing here was run by its author (file-only tools); Codex records results.
"""

from __future__ import annotations

import hashlib
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import hydrology_cuda as hc
from maple_syrup import routing_cuda
from maple_syrup.experimental_hydrology import (
    CFL_NAMES,
    EXPLICIT_BITS,
    LIMITER_SAFETY,
    LOCAL_BITS,
    METHODS,
    QUALIFICATION_STATUS,
    TRANSFER_SCOPE,
    ExperimentalHydrologyError,
    HydraulicControl,
    HydraulicState,
    HydraulicStep,
    LocalInertialGeometry,
    advance_time,
    check_dt_cap,
    check_state_time,
    resolve_flags,
    validate_hydraulic_state,
)
from maple_syrup.infiltration import _check_dt
from maple_syrup.routing import BALANCE_RTOL, GRAVITY_M_S2, RoutingGraph
from maple_syrup.routing_cuda import CudaUnavailableError

__all__ = [
    "COMPILE_OPTIONS",
    "ELEMENT_THREADS",
    "PACKET_WORDS",
    "REDUCE_THREADS",
    "CudaHydraulicSolver",
    "ExperimentalCudaContext",
    "kernel_provenance",
    "kernel_source",
    "prepare_experimental_cuda",
]

COMPILE_OPTIONS = routing_cuda.COMPILE_OPTIONS
ELEMENT_THREADS = 128
REDUCE_THREADS = 256
PACKET_WORDS = 16
_KERNELS = {
    "explicit": ("maple_syrup_exp_face", "maple_syrup_exp_cell", "maple_syrup_exp_reduce"),
    "local_inertial": ("maple_syrup_li_face_x", "maple_syrup_li_face_y", "maple_syrup_li_phi", "maple_syrup_li_scale_x",
                       "maple_syrup_li_scale_y", "maple_syrup_li_cell", "maple_syrup_li_reduce"),
}
_ALL_KERNELS = _KERNELS["explicit"] + _KERNELS["local_inertial"]

_SOURCE = r"""
__device__ __forceinline__ double nan_value() { return __longlong_as_double(0x7ff8000000000000LL); }
__device__ __forceinline__ double fmax_nan(const double a, const double b)
{
    if (a != a || b != b) return nan_value();
    return a > b ? a : b;
}
__device__ __forceinline__ double pos_part(const double x) { return x > 0.0 ? x : 0.0; }
__device__ __forceinline__ void put_double(unsigned long long* packet, const int k, const double v)
{
    packet[k] = (unsigned long long)__double_as_longlong(v);
}

// Block reductions over a fixed tree (deterministic); every thread of the block calls them uniformly (<= 256 threads).
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

// ===================== explicit kinematic wave =====================================================================
extern "C" __global__ void maple_syrup_exp_face(
    const long long n, const unsigned char* active, const double* kcell, const double* h_c,
    double* f_out, double* q_used, double* cfl_row, const double dtdx_area, const double dt_over_dx)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i >= n) return;
    double f = 0.0, q = 0.0, cfl = 0.0;
    if (active[i] != 0) {
        const double h = h_c[i];
        const double k = kcell[i];
        const double sh = __dsqrt_rn(h);
        q = __dmul_rn(__dmul_rn(sh, h), k);
        cfl = __dmul_rn(__dmul_rn(__dmul_rn(1.5, k), sh), dt_over_dx);
        f = __dmul_rn(dtdx_area, q);
    }
    f_out[i] = f;
    q_used[i] = q;
    cfl_row[i] = cfl;
}

extern "C" __global__ void maple_syrup_exp_cell(
    const long long n, const unsigned char* active, const double* kcell, const long long* donors, const double* h_c,
    const double* f_out, double* h_new, double* q_inst, double* vel, double* bal_row, double* sc_row, const double area)
{
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i >= n) return;
    long long d = donors[i];
    double total = d >= 0 ? f_out[d] : 0.0;
    for (int s = 1; s < 4; ++s) {
        d = donors[(long long)s * n + i];
        total = __dadd_rn(total, d >= 0 ? f_out[d] : 0.0);
    }
    const double out = f_out[i];
    const double hc = h_c[i];
    const double net = __ddiv_rn(__dsub_rn(total, out), area);
    const double hn = __dadd_rn(hc, net);
    h_new[i] = hn;
    const bool a = active[i] != 0;
    const double q = a ? __dmul_rn(__dmul_rn(__dsqrt_rn(hn), hn), kcell[i]) : 0.0;
    q_inst[i] = q;
    vel[i] = hn > 0.0 ? __ddiv_rn(q, hn) : 0.0;
    double bal = 0.0, sc = 0.0;
    if (a) {
        bal = fabs(__dsub_rn(__dsub_rn(hn, hc), net));
        sc = __dadd_rn(__dadd_rn(hc, hn), __ddiv_rn(__dadd_rn(total, out), area));
    }
    bal_row[i] = bal;
    sc_row[i] = sc;
}

// packet: 0 flags (1 nonfinite, 2 negative, 4 cfl, 8 balance) 1 max cfl 2 max cell balance 3 storage change 4 export
// 5 residual 6 global tolerance 7 outlet discharge 8 nonfinite-scalar mask 9 residual failed 10 max velocity
extern "C" __global__ void maple_syrup_exp_reduce(
    const long long n, const unsigned char* outlet, const double* h_c, const double* h_new, const double* f_out,
    const double* q_used, const double* q_inst, const double* vel, const double* cfl_row, const double* bal_row,
    const double* sc_row, unsigned long long* packet, const double area, const double dx, const double cfl_max,
    const double bal_rtol, const double tol_pref)
{
    const int tid = threadIdx.x;
    const int bd = blockDim.x;
    unsigned int f_nonfinite = 0u, f_negative = 0u, f_balance = 0u;
    double m_cfl = 0.0, m_bal = 0.0, m_vel = 0.0, s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (long long i = tid; i < n; i += bd) {
        const double hn = h_new[i];
        if (!isfinite(hn) || !isfinite(f_out[i]) || !isfinite(q_used[i]) || !isfinite(q_inst[i]) || !isfinite(vel[i])
            || !isfinite(sc_row[i]) || !isfinite(bal_row[i])) f_nonfinite = 1u;
        if (hn < 0.0) f_negative = 1u;
        if (bal_row[i] > bal_rtol * sc_row[i]) f_balance = 1u;
        if (cfl_row[i] > m_cfl) m_cfl = cfl_row[i];
        if (bal_row[i] > m_bal) m_bal = bal_row[i];
        if (vel[i] > m_vel) m_vel = vel[i];
        const bool out = outlet[i] != 0;
        s0 = __dadd_rn(s0, __dsub_rn(hn, h_c[i]));
        s1 = __dadd_rn(s1, out ? f_out[i] : 0.0);
        s2 = __dadd_rn(s2, sc_row[i]);
        s3 = __dadd_rn(s3, out ? q_inst[i] : 0.0);
    }
    const unsigned int g_nonfinite = block_or(f_nonfinite);
    const unsigned int g_negative = block_or(f_negative);
    const unsigned int g_balance = block_or(f_balance);
    const double g_cfl = block_max(m_cfl);
    const double g_bal = block_max(m_bal);
    const double g_vel = block_max(m_vel);
    const double t0 = block_sum(s0);
    const double t1 = block_sum(s1);
    const double t2 = block_sum(s2);
    const double t3 = block_sum(s3);
    if (tid == 0) {
        const double storage = __dmul_rn(area, t0);
        const double residual = __dadd_rn(storage, t1);
        const double gtol = __dmul_rn(tol_pref, t2);
        const double outq = __dmul_rn(dx, t3);
        unsigned long long mask = 0ull;
        if (!isfinite(storage)) mask |= 1ull << 0;
        if (!isfinite(t1)) mask |= 1ull << 1;
        if (!isfinite(residual)) mask |= 1ull << 2;
        if (!isfinite(gtol)) mask |= 1ull << 3;
        if (!isfinite(outq)) mask |= 1ull << 4;
        unsigned long long flags = 0ull;
        if (g_nonfinite != 0u) flags |= 1ull;
        if (g_negative != 0u) flags |= 2ull;
        if (g_cfl > cfl_max) flags |= 4ull;
        if (g_balance != 0u) flags |= 8ull;
        packet[0] = flags;
        put_double(packet, 1, g_cfl);
        put_double(packet, 2, g_bal);
        put_double(packet, 3, storage);
        put_double(packet, 4, t1);
        put_double(packet, 5, residual);
        put_double(packet, 6, gtol);
        put_double(packet, 7, outq);
        packet[8] = mask;
        packet[9] = fabs(residual) > gtol ? 1ull : 0ull;
        put_double(packet, 10, g_vel);
    }
}

// ===================== local inertia ================================================================================
// One face of the staggered grid. type 0 closed, 1 interior, 2 open outlet (sign = outward direction, donor = interior side).
// flag bits written per face: 1 non-finite old flux, 2 non-zero old flux on a closed face, 4 old flux into the domain on an
// open face.
__device__ __forceinline__ void li_face(
    const int type, const long long A, const long long B, const int sign, const double zmax, const double fric,
    const double kb, const double q_old, const double* z, const double* h, const double dt, const double dx,
    const double g, double* q_out, double* hf_out, unsigned char* flag_out)
{
    unsigned char flag = 0;
    if (!isfinite(q_old)) flag |= 1;
    if (type == 0 && q_old != 0.0) flag |= 2;
    if (type == 2 && __dmul_rn(q_old, (double)sign) < 0.0) flag |= 4;
    double q = 0.0, hf = 0.0;
    if (type == 1) {
        const double etaA = __dadd_rn(z[A], h[A]);
        const double etaB = __dadd_rn(z[B], h[B]);
        const double top = etaA > etaB ? etaA : etaB;
        const double hf1 = fmax_nan(__dsub_rn(top, zmax), 0.0);
        hf = hf1;
        if (hf1 > 0.0) {
            const double grad = __ddiv_rn(__dsub_rn(etaB, etaA), dx);
            const double num = __dsub_rn(q_old, __dmul_rn(__dmul_rn(__dmul_rn(g, hf1), dt), grad));
            double fr = 0.0;
            if (q_old != 0.0) {
                fr = __ddiv_rn(__dmul_rn(__dmul_rn(dt, __ddiv_rn(fric, 8.0)), fabs(q_old)), __dmul_rn(hf1, hf1));
            }
            q = __ddiv_rn(num, __dadd_rn(1.0, fr));
        }
    } else if (type == 2) {
        const double hd = sign > 0 ? h[A] : h[B];
        const double qo = __dmul_rn(__dmul_rn(__dsqrt_rn(hd), hd), kb);
        q = sign > 0 ? qo : -qo;
        hf = hd;
    }
    *q_out = q;
    *hf_out = hf;
    *flag_out = flag;
}

extern "C" __global__ void maple_syrup_li_face_x(
    const long long ny, const long long nx, const double* z, const double* h_c, const signed char* ftype,
    const double* zmax, const double* fric, const double* kb, const signed char* fsign, const double* q_old,
    double* q_new, double* hf_row, unsigned char* fflag, const double dt, const double dx, const double g)
{
    const long long f = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (f >= ny * (nx + 1)) return;
    const long long i = f / (nx + 1);
    const long long j = f - i * (nx + 1);
    const long long A = j >= 1 ? i * nx + (j - 1) : -1;
    const long long B = j < nx ? i * nx + j : -1;
    li_face((int)ftype[f], A, B, (int)fsign[f], zmax[f], fric[f], kb[f], q_old[f], z, h_c, dt, dx, g,
            &q_new[f], &hf_row[f], &fflag[f]);
}

extern "C" __global__ void maple_syrup_li_face_y(
    const long long ny, const long long nx, const double* z, const double* h_c, const signed char* ftype,
    const double* zmax, const double* fric, const double* kb, const signed char* fsign, const double* q_old,
    double* q_new, double* hf_row, unsigned char* fflag, const double dt, const double dx, const double g)
{
    const long long f = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (f >= (ny + 1) * nx) return;
    const long long i = f / nx;
    const long long j = f - i * nx;
    const long long A = i >= 1 ? (i - 1) * nx + j : -1;
    const long long B = i < ny ? i * nx + j : -1;
    li_face((int)ftype[f], A, B, (int)fsign[f], zmax[f], fric[f], kb[f], q_old[f], z, h_c, dt, dx, g,
            &q_new[f], &hf_row[f], &fflag[f]);
}

// Donor limiter: phi per cell (outflow volume vs available water), then every face leaving a cell is scaled ONCE by that
// cell's phi (the same face value is read by both ends afterwards).
extern "C" __global__ void maple_syrup_li_phi(
    const long long ny, const long long nx, const double* qx, const double* qy, const double* h_c, double* phi,
    double* outv_row, const double dtdx_area, const double area, const double safety)
{
    const long long c = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (c >= ny * nx) return;
    const long long i = c / nx;
    const long long j = c - i * nx;
    const double qw = qx[i * (nx + 1) + j];
    const double qe = qx[i * (nx + 1) + j + 1];
    const double qs = qy[i * nx + j];
    const double qn = qy[(i + 1) * nx + j];
    const double out = __dadd_rn(__dadd_rn(pos_part(-qw), pos_part(qe)), __dadd_rn(pos_part(-qs), pos_part(qn)));
    const double outv = __dmul_rn(dtdx_area, out);
    const double avail = __dmul_rn(h_c[c], area);
    outv_row[c] = outv;
    phi[c] = outv > avail ? __dmul_rn(__ddiv_rn(avail, outv), safety) : 1.0;
}

extern "C" __global__ void maple_syrup_li_scale_x(const long long ny, const long long nx, const double* phi, double* q)
{
    const long long f = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (f >= ny * (nx + 1)) return;
    const long long i = f / (nx + 1);
    const long long j = f - i * (nx + 1);
    double v = q[f];
    if (v > 0.0) {
        if (j >= 1) v = __dmul_rn(v, phi[i * nx + (j - 1)]);
    } else if (v < 0.0) {
        if (j < nx) v = __dmul_rn(v, phi[i * nx + j]);
    }
    q[f] = v;
}

extern "C" __global__ void maple_syrup_li_scale_y(const long long ny, const long long nx, const double* phi, double* q)
{
    const long long f = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (f >= (ny + 1) * nx) return;
    const long long i = f / nx;
    const long long j = f - i * nx;
    double v = q[f];
    if (v > 0.0) {
        if (i >= 1) v = __dmul_rn(v, phi[(i - 1) * nx + j]);
    } else if (v < 0.0) {
        if (i < ny) v = __dmul_rn(v, phi[i * nx + j]);
    }
    q[f] = v;
}

extern "C" __global__ void maple_syrup_li_cell(
    const long long ny, const long long nx, const unsigned char* active, const double* qx, const double* qy,
    const double* h_c, double* h_new, double* speed, double* bal_row, double* sc_row, const double dt_over_dx)
{
    const long long c = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (c >= ny * nx) return;
    const long long i = c / nx;
    const long long j = c - i * nx;
    const double qw = qx[i * (nx + 1) + j];
    const double qe = qx[i * (nx + 1) + j + 1];
    const double qs = qy[i * nx + j];
    const double qn = qy[(i + 1) * nx + j];
    const double div = __dadd_rn(__dsub_rn(qe, qw), __dsub_rn(qn, qs));
    const double hc = h_c[c];
    const double hn = __dsub_rn(hc, __dmul_rn(dt_over_dx, div));
    h_new[c] = hn;
    const bool wet = hn > 0.0;
    const double ux = wet ? __ddiv_rn(__dmul_rn(0.5, __dadd_rn(qw, qe)), hn) : 0.0;
    const double uy = wet ? __ddiv_rn(__dmul_rn(0.5, __dadd_rn(qs, qn)), hn) : 0.0;
    speed[c] = __dsqrt_rn(__dadd_rn(__dmul_rn(ux, ux), __dmul_rn(uy, uy)));
    double bal = 0.0, sc = 0.0;
    if (active[c] != 0) {
        bal = fabs(__dadd_rn(__dsub_rn(hn, hc), __dmul_rn(dt_over_dx, div)));
        const double faces = __dadd_rn(__dadd_rn(fabs(qw), fabs(qe)), __dadd_rn(fabs(qs), fabs(qn)));
        sc = __dadd_rn(__dadd_rn(hc, hn), __dmul_rn(dt_over_dx, faces));
    }
    bal_row[c] = bal;
    sc_row[c] = sc;
}

// packet: 0 flags (1 state nonfinite, 2 closed-face flux, 4 open-face inflow, 8 cfl, 16 negative, 32 nonfinite, 64 balance)
// 1 max wave cfl 2 max cell balance 3 storage change 4 export 5 residual 6 global tolerance 7 outlet discharge
// 8 nonfinite-scalar mask 9 residual failed 10 max speed 11 limited cells 12 limited volume 13 max face depth
extern "C" __global__ void maple_syrup_li_reduce(
    const long long ny, const long long nx, const signed char* ftype_x, const signed char* fsign_x,
    const signed char* ftype_y, const signed char* fsign_y, const double* h_c, const double* h_new, const double* qx,
    const double* qy, const double* speed, const double* bal_row, const double* sc_row, const double* hfx,
    const double* hfy, const unsigned char* fflag_x, const unsigned char* fflag_y, const double* phi,
    const double* outv_row, unsigned long long* packet, const int limiter, const double area, const double dx,
    const double dtdx_area, const double dt_over_dx, const double g, const double cfl_max, const double bal_rtol,
    const double tol_pref)
{
    const int tid = threadIdx.x;
    const int bd = blockDim.x;
    unsigned int f_input = 0u, f_negative = 0u, f_nonfinite = 0u, f_balance = 0u;
    double m_bal = 0.0, m_speed = 0.0, m_hf = 0.0, s0 = 0.0, s2 = 0.0, sx = 0.0, sy = 0.0, lim_volume = 0.0;
    long long n_limited = 0;
    for (long long c = tid; c < ny * nx; c += bd) {
        const double hn = h_new[c];
        if (!isfinite(hn) || !isfinite(speed[c]) || !isfinite(bal_row[c]) || !isfinite(sc_row[c])) f_nonfinite = 1u;
        if (hn < 0.0) f_negative = 1u;
        if (bal_row[c] > bal_rtol * sc_row[c]) f_balance = 1u;
        if (bal_row[c] > m_bal) m_bal = bal_row[c];
        if (speed[c] > m_speed) m_speed = speed[c];
        s0 = __dadd_rn(s0, __dsub_rn(hn, h_c[c]));
        s2 = __dadd_rn(s2, sc_row[c]);
        if (limiter != 0 && phi[c] < 1.0) {
            n_limited += 1;
            lim_volume = __dadd_rn(lim_volume, __dmul_rn(outv_row[c], __dsub_rn(1.0, phi[c])));
        }
    }
    for (long long f = tid; f < ny * (nx + 1); f += bd) {
        f_input |= (unsigned int)fflag_x[f];
        if (!isfinite(qx[f])) f_nonfinite = 1u;
        if (hfx[f] > m_hf) m_hf = hfx[f];
        if (ftype_x[f] == 2) sx = __dadd_rn(sx, __dmul_rn(qx[f], (double)fsign_x[f]));
    }
    for (long long f = tid; f < (ny + 1) * nx; f += bd) {
        f_input |= (unsigned int)fflag_y[f];
        if (!isfinite(qy[f])) f_nonfinite = 1u;
        if (hfy[f] > m_hf) m_hf = hfy[f];
        if (ftype_y[f] == 2) sy = __dadd_rn(sy, __dmul_rn(qy[f], (double)fsign_y[f]));
    }
    const unsigned int g_input = block_or(f_input);
    const unsigned int g_negative = block_or(f_negative);
    const unsigned int g_nonfinite = block_or(f_nonfinite);
    const unsigned int g_balance = block_or(f_balance);
    const double g_bal = block_max(m_bal);
    const double g_speed = block_max(m_speed);
    const double g_hf = block_max(m_hf);
    const double t0 = block_sum(s0);
    const double t2 = block_sum(s2);
    const double tx = block_sum(sx);
    const double ty = block_sum(sy);
    const double lim_total = block_sum(lim_volume);
    const long long lim_count = block_isum(n_limited);
    if (tid == 0) {
        const double outward = __dadd_rn(tx, ty);
        const double storage = __dmul_rn(area, t0);
        const double exportv = __dmul_rn(dtdx_area, outward);
        const double residual = __dadd_rn(storage, exportv);
        const double gtol = __dmul_rn(tol_pref, t2);
        const double outq = __dmul_rn(dx, outward);
        const double wave = __dmul_rn(dt_over_dx, __dsqrt_rn(__dmul_rn(__dmul_rn(2.0, g), g_hf)));
        unsigned long long mask = 0ull;
        if (!isfinite(storage)) mask |= 1ull << 0;
        if (!isfinite(exportv)) mask |= 1ull << 1;
        if (!isfinite(residual)) mask |= 1ull << 2;
        if (!isfinite(gtol)) mask |= 1ull << 3;
        if (!isfinite(outq)) mask |= 1ull << 4;
        unsigned long long flags = (unsigned long long)g_input;
        if (wave > cfl_max) flags |= 8ull;
        if (g_negative != 0u) flags |= 16ull;
        if (g_nonfinite != 0u) flags |= 32ull;
        if (g_balance != 0u) flags |= 64ull;
        packet[0] = flags;
        put_double(packet, 1, wave);
        put_double(packet, 2, g_bal);
        put_double(packet, 3, storage);
        put_double(packet, 4, exportv);
        put_double(packet, 5, residual);
        put_double(packet, 6, gtol);
        put_double(packet, 7, outq);
        packet[8] = mask;
        packet[9] = fabs(residual) > gtol ? 1ull : 0ull;
        put_double(packet, 10, g_speed);
        packet[11] = (unsigned long long)lim_count;
        put_double(packet, 12, lim_total);
        put_double(packet, 13, g_hf);
    }
}
"""


def kernel_source() -> str:
    return _SOURCE


# --- lazy optional CuPy / kernels -----------------------------------------------------------------------------------
_MODULE: Any = None
_FUNCTIONS: dict[tuple[int, str], Any] = {}
_LOCK = threading.Lock()


def _function(name: str):
    """The compiled kernel `name` for the CURRENT device (module created lazily; compilation at first load)."""
    global _MODULE
    cp = routing_cuda._cupy()
    key = (routing_cuda._current_device_id(cp), name)
    fn = _FUNCTIONS.get(key)
    if fn is not None:
        return fn
    with _LOCK:
        try:
            if _MODULE is None:
                _MODULE = cp.RawModule(code=_SOURCE, options=COMPILE_OPTIONS, backend="nvrtc")
            fn = _MODULE.get_function(name)
            fn.attributes  # noqa: B018 - forces the module load for this device so failures surface now
        except Exception as exc:
            raise CudaUnavailableError(
                f"experimental CUDA kernel {name!r} failed to compile/load ({type(exc).__name__}: {exc}); no fallback") from exc
        _FUNCTIONS[key] = fn
    return fn


def _load_all(method: str) -> dict[str, dict[str, int]]:
    attributes = {}
    for name in _KERNELS[method]:
        attrs = dict(_function(name).attributes)
        need = REDUCE_THREADS if name.endswith("reduce") else ELEMENT_THREADS
        if int(attrs.get("max_threads_per_block", 0)) < need:
            raise CudaUnavailableError(f"kernel {name} supports only {attrs.get('max_threads_per_block')} threads per block, "
                                       f"{need} are needed; no fallback")
        attributes[name] = {k: int(v) for k, v in attrs.items() if isinstance(v, (int, np.integer))}
    return attributes


def kernel_provenance() -> dict[str, Any]:
    info = routing_cuda.kernel_provenance()
    info.update({
        "module": "maple_syrup.experimental_cuda", "experimental_source_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest(),
        "kernels": {m: list(v) for m, v in _KERNELS.items()}, "element_threads": ELEMENT_THREADS,
        "reduce_threads": REDUCE_THREADS, "packet_words": PACKET_WORDS, "packet_bytes": PACKET_WORDS * 8,
        "launches_per_step": {"explicit": 3, "local_inertial_limiter_off": 4, "local_inertial_limiter_donor": 7},
        "column": "hydrology_cuda.prepared_column_step (baseline, one more counted packet read per step)",
        "numba_required": False, "libm": "none in the lateral kernels (+ - * / sqrt only); the baseline column uses expm1",
    })
    return info


# --- context --------------------------------------------------------------------------------------------------------
def _static_names(method: str) -> tuple[str, ...]:
    if method == "explicit":
        return ("donors",)
    return ("z", "fx_type", "fx_zmax", "fx_fric", "fx_kb", "fx_sign", "fy_type", "fy_zmax", "fy_fric", "fy_kb", "fy_sign")


def _expected_shapes(method: str, shape: tuple[int, int], n_cells: int) -> tuple:
    ny, nx = shape
    if method == "explicit":
        return ((4, n_cells),)
    nxf, nyf = ny * (nx + 1), (ny + 1) * nx
    return ((n_cells,), (nxf,), (nxf,), (nxf,), (nxf,), (nxf,), (nyf,), (nyf,), (nyf,), (nyf,), (nyf,))


@dataclass(frozen=True, eq=False)
class ExperimentalCudaContext:
    """Owned, validated static device data of one experimental solver on one graph. Wraps the baseline
    `CudaHydrologyContext` (column physics, active/outlet/conveyance, its own seal and binding) and adds the method-specific
    arrays. Immutable by contract; build with `prepare_experimental_cuda`."""

    method: str
    hydrology: Any
    device_id: int
    shape: tuple[int, int]
    n_cells: int
    n_active: int
    dx_m: float
    names: tuple[str, ...]
    arrays: dict
    owned_fingerprints: tuple
    scalar_signature: tuple
    geometry_ref: Any
    geometry_sha256: Any
    static_bytes: int
    preparation_s: float
    kernel_load_s: float
    host_to_device_bytes: int
    device_to_host_bytes: int
    kernel_attributes: dict

    def array(self, name: str):
        return self.arrays[name]

    def is_bound_to(self, graph: Any, params: Any, geometry: Any = None) -> bool:
        """The very graph/params (and, for local inertia, geometry) objects prepared from, metadata unchanged; weak
        identity, no strong reference, no content hash (an in-place content change of the sources is not detected)."""
        if not self.hydrology.is_bound_to(graph, params):
            return False
        if self.method == "local_inertial":
            return self.geometry_ref is not None and self.geometry_ref() is geometry and geometry is not None
        return geometry is None

    def summary(self) -> dict[str, Any]:
        return {"method": self.method, "shape": list(self.shape), "n_cells": self.n_cells, "n_active": self.n_active,
                "dx_m": self.dx_m, "device_id": self.device_id, "static_names": list(self.names),
                "static_bytes": self.static_bytes, "hydrology_static_bytes": self.hydrology.static_bytes,
                "preparation_s": self.preparation_s, "kernel_load_s": self.kernel_load_s,
                "host_to_device_bytes": self.host_to_device_bytes, "device_to_host_bytes": self.device_to_host_bytes,
                "kernel_attributes": self.kernel_attributes, "geometry_sha256": self.geometry_sha256,
                "column_context": self.hydrology.summary(), "packet_bytes": PACKET_WORDS * 8,
                "numba_required": False, "fastmath": False, "device_resident": True}


def _signature(ctx: ExperimentalCudaContext) -> tuple:
    if not isinstance(ctx.shape, tuple) or not isinstance(ctx.names, tuple):
        raise TypeError("shape and names must be tuples")
    return (ctx.method, ctx.device_id, ctx.shape, ctx.n_cells, ctx.n_active, ctx.dx_m, ctx.names)


def _check_context(cp: Any, ctx: Any) -> None:
    """Every public entry, BEFORE any launch: type, the baseline context's own checks (current device, fingerprints, sealed
    metadata), this context's sealed scalars, array extents and fingerprints."""
    if not isinstance(ctx, ExperimentalCudaContext):
        raise ExperimentalHydrologyError("ctx must be an ExperimentalCudaContext (use prepare_experimental_cuda)")
    hc._check_context(cp, ctx.hydrology)
    try:
        current = _signature(ctx)
    except (TypeError, ValueError):
        current = None
    if current is None or ctx.scalar_signature != current:
        raise ExperimentalHydrologyError("the experimental context's scalar metadata no longer matches the values sealed "
                                         "at preparation; refusing before any launch")
    ok = (ctx.method in METHODS and ctx.names == _static_names(ctx.method)
          and len(ctx.shape) == 2 and ctx.shape[0] >= 1 and ctx.shape[1] >= 1 and ctx.shape[0] * ctx.shape[1] == ctx.n_cells
          and ctx.n_cells == ctx.hydrology.n_cells and ctx.n_active == ctx.hydrology.n_active
          and ctx.shape == ctx.hydrology.shape and ctx.dx_m == ctx.hydrology.dx_m and ctx.device_id == ctx.hydrology.device_id
          and ctx.dx_m > 0.0 and set(ctx.arrays) == set(ctx.names))
    arrays = tuple(ctx.arrays[name] for name in ctx.names) if ok else ()
    if not ok or tuple(tuple(a.shape) for a in arrays) != _expected_shapes(ctx.method, ctx.shape, ctx.n_cells) \
            or ctx.owned_fingerprints != tuple(routing_cuda._fingerprint(a) for a in arrays):
        raise ExperimentalHydrologyError("the experimental context's owned device arrays no longer match their prepared "
                                         "extents/metadata; refusing before any launch")


def prepare_experimental_cuda(method: str, graph: RoutingGraph, params: Any, *,
                              geometry: LocalInertialGeometry | None = None, mode: str = "auto") -> ExperimentalCudaContext:
    """Validate and upload once. The baseline context is prepared first (graph/column validation, counted download, owned
    copies, sealed and bound; `mode` selects its launch structure and is irrelevant to this module's kernels); local inertia
    additionally validates `geometry` (a host `LocalInertialGeometry`) against the graph and uploads its static face arrays.
    Raises `CudaUnavailableError` (no CuPy/device/compiler) or a preparation error before anything is returned."""
    from maple.core.backend import read_transfer_counters, to_device, to_host

    t0 = time.perf_counter()
    if method not in METHODS:
        raise ExperimentalHydrologyError(f"method must be one of {METHODS}, got {method!r}")
    cp = routing_cuda._cupy()
    routing_cuda._current_device_id(cp)
    if method == "local_inertial":
        if not isinstance(geometry, LocalInertialGeometry):
            raise ExperimentalHydrologyError("the local-inertial method needs a LocalInertialGeometry")
    elif geometry is not None:
        raise ExperimentalHydrologyError("the explicit method takes no geometry")
    t_load = time.perf_counter()
    attributes = _load_all(method)  # compile/load before any transfer
    kernel_load_s = time.perf_counter() - t_load
    hydrology = hc.prepare_cuda_hydrology(graph, params, mode=mode)  # validation, counted transfers, seal, binding
    before = read_transfer_counters()
    n, shape = hydrology.n_cells, hydrology.shape
    host: dict[str, np.ndarray]
    if method == "explicit":
        order = to_host(hydrology.level_order)
        donor_cell = to_host(hydrology.donor_cell)
        donors = np.full((4, n), -1, dtype=np.int64)
        donors[:, order] = donor_cell
        host = {"donors": donors}
        geometry_ref = geometry_sha = None
    else:
        g = geometry
        if (tuple(g.shape) != tuple(shape) or g.dx_m != hydrology.dx_m or not np.array_equal(g.active, np.asarray(graph.active))
                or not np.array_equal(g.friction, np.asarray(graph.friction_factor))):
            raise ExperimentalHydrologyError("the geometry does not match the graph (shape, dx, active mask, friction)")
        host = {"z": g.z.reshape(-1), "fx_type": g.fx_type.reshape(-1), "fx_zmax": g.fx_zmax.reshape(-1),
                "fx_fric": g.fx_fric.reshape(-1), "fx_kb": g.fx_kb.reshape(-1), "fx_sign": g.fx_sign.reshape(-1),
                "fy_type": g.fy_type.reshape(-1), "fy_zmax": g.fy_zmax.reshape(-1), "fy_fric": g.fy_fric.reshape(-1),
                "fy_kb": g.fy_kb.reshape(-1), "fy_sign": g.fy_sign.reshape(-1)}
        geometry_ref, geometry_sha = weakref.ref(g), g.input_sha256
    names = _static_names(method)
    arrays = {name: to_device(np.ascontiguousarray(host[name]), cp) for name in names}
    cp.cuda.get_current_stream().synchronize()
    delta = read_transfer_counters().delta(before)
    ctx = ExperimentalCudaContext(
        method=method, hydrology=hydrology, device_id=hydrology.device_id, shape=tuple(shape), n_cells=n,
        n_active=hydrology.n_active, dx_m=hydrology.dx_m, names=names, arrays=arrays,
        owned_fingerprints=tuple(routing_cuda._fingerprint(arrays[name]) for name in names), scalar_signature=(),
        geometry_ref=geometry_ref, geometry_sha256=geometry_sha, static_bytes=int(sum(a.nbytes for a in arrays.values())),
        preparation_s=time.perf_counter() - t0, kernel_load_s=kernel_load_s,
        host_to_device_bytes=int(delta.host_to_device_bytes) + hydrology.host_to_device_bytes,
        device_to_host_bytes=int(delta.device_to_host_bytes) + hydrology.device_to_host_bytes,
        kernel_attributes=attributes)
    object.__setattr__(ctx, "scalar_signature", _signature(ctx))  # sealed once, here; frozen afterwards
    return ctx


# --- launches ---------------------------------------------------------------------------------------------------------
def _launch(name: str, grid: int, block: int, args: tuple) -> None:
    try:
        _function(name)((grid,), (block,), args)
    except CudaUnavailableError:
        raise
    except Exception as exc:  # enqueue-time failure; asynchronous faults surface at the packet read
        raise ExperimentalHydrologyError(
            f"experimental CUDA kernel {name} launch failed ({type(exc).__name__}: {exc}); no fallback") from exc


def _blocks(count: int) -> int:
    return (int(count) + ELEMENT_THREADS - 1) // ELEMENT_THREADS


class CudaHydraulicSolver:
    """The CUDA form of `CpuHydraulicSolver` (same `step`/`initial_state`/`validate_state`/`describe` interface, so one driver
    `experimental_storm.evolve_experimental` serves both). Pure, fresh outputs, no hidden transfer."""

    implementation = "cuda"

    def __init__(self, method: str, graph: RoutingGraph, params: Any, *, geometry: LocalInertialGeometry | None = None,
                 control: HydraulicControl | None = None, mode: str = "auto", context: ExperimentalCudaContext | None = None):
        if method not in METHODS:
            raise ExperimentalHydrologyError(f"method must be one of {METHODS}, got {method!r}")
        control = (HydraulicControl() if control is None else control).validated()
        if method == "explicit" and control.limiter != "off":
            raise ExperimentalHydrologyError("the donor limiter belongs to the local-inertial method; the explicit "
                                             "method is positive by its CFL bound")
        routing_cuda._cupy()  # CuPy must be importable (no fallback)
        if context is None:
            context = prepare_experimental_cuda(method, graph, params, geometry=geometry, mode=mode)
        elif (not isinstance(context, ExperimentalCudaContext) or context.method != method
              or not context.is_bound_to(graph, params, geometry)):
            raise ExperimentalHydrologyError("context must be the ExperimentalCudaContext prepared for exactly this "
                                             "method, graph, parameters and geometry")
        # Canonical metadata lives ONLY in the sealed context and the frozen control; method/shape/dx/control/context are
        # read-only properties and every public entry re-checks them against this seal BEFORE any raw enqueue.
        self._context, self._control = context, control
        self._seal = self._current_seal(context, control)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_sealed"):
            raise AttributeError(f"CudaHydraulicSolver is immutable after construction (cannot set {name!r}); build a new "
                                 "solver")
        object.__setattr__(self, name, value)

    @staticmethod
    def _current_seal(ctx: Any, control: Any) -> tuple:
        return (ctx.method, ctx.shape, ctx.dx_m, control.cfl_max, control.limiter, id(ctx))

    @property
    def xp(self):
        return routing_cuda._cupy()

    @property
    def context(self) -> ExperimentalCudaContext:
        return self._context

    @property
    def control(self) -> HydraulicControl:
        return self._control

    @property
    def method(self) -> str:
        return self._context.method

    @property
    def shape(self) -> tuple[int, int]:
        return self._context.shape

    @property
    def dx_m(self) -> float:
        return self._context.dx_m

    def _guard(self) -> tuple[Any, ExperimentalCudaContext, HydraulicControl]:
        """BEFORE any launch: the context (baseline + experimental seals, fingerprints, current device) and that the solver's
        method/shape/dx/control/context still equal the values sealed at construction. Returns `(cupy, context, control)`
        from which every launch parameter must be taken (never from mutable solver attributes)."""
        cp = routing_cuda._cupy()
        ctx, control = self._context, self._control
        _check_context(cp, ctx)
        if not isinstance(control, HydraulicControl):
            raise ExperimentalHydrologyError("the solver's control is not a HydraulicControl; refusing before any launch")
        control.validated()
        if self._current_seal(ctx, control) != self._seal:
            raise ExperimentalHydrologyError("the solver's method, shape, dx, control or context no longer match the values "
                                             "sealed at construction; refusing before any launch")
        if ctx.method == "explicit" and control.limiter != "off":
            raise ExperimentalHydrologyError("the donor limiter belongs to the local-inertial method")
        return cp, ctx, control

    # -- state ------------------------------------------------------------------------------------------------------
    def _faces(self, ctx: ExperimentalCudaContext):
        ny, nx = ctx.shape
        if ctx.method == "explicit":
            return None
        return (ctx.array("fx_type").reshape(ny, nx + 1), ctx.array("fx_sign").reshape(ny, nx + 1),
                ctx.array("fy_type").reshape(ny + 1, nx), ctx.array("fy_sign").reshape(ny + 1, nx))

    def validate_state(self, state: HydraulicState) -> None:
        cp, ctx, _control = self._guard()
        validate_hydraulic_state(cp, state, method=ctx.method, shape=ctx.shape, faces=self._faces(ctx))

    def initial_state(self, depth_m: Any, soil_water_m: Any, *, t_s: float = 0.0) -> HydraulicState:
        """Validated fresh copy of a state at rest; `t_s` must be a real (not bool/str) finite number >= 0."""
        cp, ctx, _control = self._guard()
        time_s = check_state_time(t_s, "t_s")
        for name, array in (("depth_m", depth_m), ("soil_water_m", soil_water_m)):
            hc._dynamic(cp, array, name, ctx.shape, ExperimentalHydrologyError, ctx.device_id)
        ny, nx = ctx.shape
        faces = (None, None) if ctx.method == "explicit" else (cp.zeros((ny, nx + 1)), cp.zeros((ny + 1, nx)))
        state = HydraulicState(time_s, cp.array(depth_m, copy=True), cp.array(soil_water_m, copy=True), *faces)
        self.validate_state(state)
        return state

    def describe(self) -> dict[str, Any]:
        return {"method": self.method, "implementation": self.implementation,
                "control": {"cfl_max": self.control.cfl_max, "limiter": self.control.limiter},
                "cfl_kind": CFL_NAMES[self.method], "context": self.context.summary(),
                "kernels": kernel_provenance(), "transfer_scope": TRANSFER_SCOPE, "qualification": QUALIFICATION_STATUS}

    # -- one step ---------------------------------------------------------------------------------------------------
    def step(self, rain_rate_m_per_s: Any, state: HydraulicState, dt_s: float) -> HydraulicStep:
        from maple.core.backend import to_host

        cp, ctx, control = self._guard()
        if not isinstance(state, HydraulicState):
            raise ExperimentalHydrologyError(f"state must be a HydraulicState, got {type(state).__name__}")
        ny, nx = ctx.shape
        # strict time contract, BEFORE any launch: finite non-negative real state time, then (for dt > 0) the end time must be
        # finite and advance; dt itself is checked by the accepted column stage with its own error class
        t_state = check_state_time(state.t_s, "state.t_s")
        check_dt_cap(state.next_dt_cap_s)  # continuation metadata is never read by a step, but a malformed one is refused
        dt_checked = _check_dt(dt_s)
        t_new = advance_time(t_state, dt_checked) if dt_checked > 0.0 else None
        if ctx.method == "explicit":
            if state.qx_m2_s is not None or state.qy_m2_s is not None:
                raise ExperimentalHydrologyError("the explicit method has no momentum: qx_m2_s and qy_m2_s must be None")
        else:
            for name, array, expected in (("qx_m2_s", state.qx_m2_s, (ny, nx + 1)), ("qy_m2_s", state.qy_m2_s, (ny + 1, nx))):
                hc._dynamic(cp, array, f"state.{name}", expected, ExperimentalHydrologyError, ctx.device_id)
        # the accepted device column physics first (InfiltrationError for any invalid input/dt; dt = 0 is its identity)
        col = hc.prepared_column_step(ctx.hydrology, state.depth_m, state.soil_water_m, rain_rate_m_per_s, dt_s)
        dt = dt_checked
        if not dt > 0.0:
            raise ExperimentalHydrologyError(f"dt_s must be finite and > 0 (dt = 0 is rejected, not an identity), got {dt_s!r}")
        if ctx.method == "explicit":
            return self._explicit(cp, ctx, control, col, dt, t_new, to_host)
        return self._local_inertial(cp, ctx, control, col, state, dt, t_new, to_host)

    @staticmethod
    def _resolve(words: np.ndarray, method: str, control: HydraulicControl) -> None:
        """Raise the first failure recorded in the lateral packet (the CPU reference's order and messages)."""
        nonfinite = int(words[8])
        scalars = tuple(bool((nonfinite >> k) & 1) for k in range(5))
        resolve_flags(method, int(words[0]), scalars, bool(words[9]), float(words.view(np.float64)[1]),
                      control.cfl_max, control.limiter if method == "local_inertial" else "off")

    def _explicit(self, cp, ctx, control, col, dt, t_new, to_host) -> HydraulicStep:
        n, dx = ctx.n_cells, ctx.dx_m
        hyd = ctx.hydrology
        area, dtdx_area, dt_over_dx = dx * dx, dt * dx, dt / dx
        h_c = col.depth_m
        f_out, q_used, cfl_row, h_new, q_inst, vel, bal_row, sc_row = (cp.empty(n, dtype=np.float64) for _ in range(8))
        packet = cp.empty(PACKET_WORDS, dtype=np.uint64)
        f64, i64 = np.float64, np.int64
        _launch("maple_syrup_exp_face", _blocks(n), ELEMENT_THREADS,
                (i64(n), hyd.active, hyd.conveyance, h_c, f_out, q_used, cfl_row, f64(dtdx_area), f64(dt_over_dx)))
        _launch("maple_syrup_exp_cell", _blocks(n), ELEMENT_THREADS,
                (i64(n), hyd.active, hyd.conveyance, ctx.array("donors"), h_c, f_out, h_new, q_inst, vel, bal_row, sc_row,
                 f64(area)))
        _launch("maple_syrup_exp_reduce", 1, REDUCE_THREADS,
                (i64(n), hyd.outlet, h_c, h_new, f_out, q_used, q_inst, vel, cfl_row, bal_row, sc_row, packet, f64(area),
                 f64(dx), f64(control.cfl_max), f64(BALANCE_RTOL),
                 f64(BALANCE_RTOL * (ctx.n_active + 2) * area)))
        self._resolve(to_host(packet), "explicit", control)  # the ONE counted read of the lateral stage
        view_f = packet.view(np.float64)
        grid = ctx.shape
        new_state = HydraulicState(t_new, h_new.reshape(grid), col.soil_water_m)
        return HydraulicStep(
            method="explicit", implementation=self.implementation, dt_s=dt, state=new_state, column=col,
            velocity_m_s=vel.reshape(grid), face_volume_m3={"out": f_out.reshape(grid)},
            used_flux_m2_s={"out": q_used.reshape(grid)}, export_m3=view_f[4], outlet_discharge_m3_s=view_f[7],
            storage_change_m3=view_f[3], budget_residual_m3=view_f[5], max_cfl=view_f[1],
            max_cell_balance_residual_m=view_f[2], limited_cells=0, limited_volume_m3=0.0, cfl_kind=CFL_NAMES["explicit"],
            face_flow_depth_m={"out": h_c.reshape(grid)})

    def _local_inertial(self, cp, ctx, control, col, state, dt, t_new, to_host) -> HydraulicStep:
        ny, nx = ctx.shape
        n, dx = ctx.n_cells, ctx.dx_m
        hyd = ctx.hydrology
        area, dtdx_area, dt_over_dx = dx * dx, dt * dx, dt / dx
        h_c = col.depth_m
        z = ctx.array("z")
        nxf, nyf = ny * (nx + 1), (ny + 1) * nx
        qx_new, hfx = cp.empty(nxf, dtype=np.float64), cp.empty(nxf, dtype=np.float64)
        qy_new, hfy = cp.empty(nyf, dtype=np.float64), cp.empty(nyf, dtype=np.float64)
        fflag_x, fflag_y = cp.empty(nxf, dtype=np.uint8), cp.empty(nyf, dtype=np.uint8)
        h_new, speed, bal_row, sc_row, phi, outv_row = (cp.empty(n, dtype=np.float64) for _ in range(6))
        packet = cp.empty(PACKET_WORDS, dtype=np.uint64)
        f64, i64, i32 = np.float64, np.int64, np.int32
        limiter = control.limiter == "donor"
        _launch("maple_syrup_li_face_x", _blocks(nxf), ELEMENT_THREADS,
                (i64(ny), i64(nx), z, h_c, ctx.array("fx_type"), ctx.array("fx_zmax"), ctx.array("fx_fric"),
                 ctx.array("fx_kb"), ctx.array("fx_sign"), state.qx_m2_s, qx_new, hfx, fflag_x, f64(dt), f64(dx),
                 f64(GRAVITY_M_S2)))
        _launch("maple_syrup_li_face_y", _blocks(nyf), ELEMENT_THREADS,
                (i64(ny), i64(nx), z, h_c, ctx.array("fy_type"), ctx.array("fy_zmax"), ctx.array("fy_fric"),
                 ctx.array("fy_kb"), ctx.array("fy_sign"), state.qy_m2_s, qy_new, hfy, fflag_y, f64(dt), f64(dx),
                 f64(GRAVITY_M_S2)))
        if limiter:
            _launch("maple_syrup_li_phi", _blocks(n), ELEMENT_THREADS,
                    (i64(ny), i64(nx), qx_new, qy_new, h_c, phi, outv_row, f64(dtdx_area), f64(area),
                     f64(LIMITER_SAFETY)))
            _launch("maple_syrup_li_scale_x", _blocks(nxf), ELEMENT_THREADS, (i64(ny), i64(nx), phi, qx_new))
            _launch("maple_syrup_li_scale_y", _blocks(nyf), ELEMENT_THREADS, (i64(ny), i64(nx), phi, qy_new))
        _launch("maple_syrup_li_cell", _blocks(n), ELEMENT_THREADS,
                (i64(ny), i64(nx), hyd.active, qx_new, qy_new, h_c, h_new, speed, bal_row, sc_row, f64(dt_over_dx)))
        _launch("maple_syrup_li_reduce", 1, REDUCE_THREADS,
                (i64(ny), i64(nx), ctx.array("fx_type"), ctx.array("fx_sign"), ctx.array("fy_type"), ctx.array("fy_sign"),
                 h_c, h_new, qx_new, qy_new, speed, bal_row, sc_row, hfx, hfy, fflag_x, fflag_y, phi, outv_row, packet,
                 i32(1 if limiter else 0), f64(area), f64(dx), f64(dtdx_area), f64(dt_over_dx), f64(GRAVITY_M_S2),
                 f64(control.cfl_max), f64(BALANCE_RTOL), f64(BALANCE_RTOL * (ctx.n_active + 2) * area)))
        self._resolve(to_host(packet), "local_inertial", control)  # the ONE counted read of the lateral stage
        view_f, view_i = packet.view(np.float64), packet.view(np.int64)
        grid = ctx.shape
        qx_state, qy_state = qx_new.reshape(ny, nx + 1), qy_new.reshape(ny + 1, nx)
        new_state = HydraulicState(t_new, h_new.reshape(grid), col.soil_water_m, qx_state, qy_state)
        return HydraulicStep(
            method="local_inertial", implementation=self.implementation, dt_s=dt, state=new_state, column=col,
            velocity_m_s=speed.reshape(grid), face_volume_m3={"x": dtdx_area * qx_state, "y": dtdx_area * qy_state},
            used_flux_m2_s={"x": qx_state, "y": qy_state}, export_m3=view_f[4], outlet_discharge_m3_s=view_f[7],
            storage_change_m3=view_f[3], budget_residual_m3=view_f[5], max_cfl=view_f[1],
            max_cell_balance_residual_m=view_f[2], limited_cells=view_i[11], limited_volume_m3=view_f[12],
            cfl_kind=CFL_NAMES["local_inertial"],
            face_flow_depth_m={"x": hfx.reshape(ny, nx + 1), "y": hfy.reshape(ny + 1, nx)})


# the flag-bit tables are shared with the CPU reference; the kernels above hard-code the same numbers
if EXPLICIT_BITS != {"nonfinite": 1, "negative": 2, "cfl": 4, "balance": 8} or LOCAL_BITS != {
        "state_nonfinite": 1, "closed_face": 2, "open_inflow": 4, "cfl": 8, "negative": 16, "nonfinite": 32,
        "balance": 64}:
    raise ImportError("experimental flag-bit tables disagree with the CUDA kernels")
