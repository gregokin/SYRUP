"""Compiled CPU (Numba) form of the two EXPERIMENTAL hydraulic alternatives (task candidate_cpu_numba).

`experimental_hydrology` documents the equations, units, CFL/positivity bounds, boundary and donor limiter and holds the NumPy
reference (`CpuHydraulicSolver`, the independent oracle, unchanged); `experimental_cuda` is the CUDA form. This module evaluates
the SAME expressions in the SAME order in serial Numba kernels, so the lateral hot loops make no NumPy temporary chains:

    solver = NumbaHydraulicSolver("explicit" | "local_inertial", graph, params, geometry=geometry, control=HydraulicControl(...))
    step   = solver.step(rain, state, dt)            # pure; fresh outputs; the same HydraulicStep as the NumPy reference

`NumbaHydraulicSolver` SUBCLASSES the reference: `step`, `initial_state`, `validate_state` and every public check (types, times,
dt class, continuation cap, momentum and shape rules, error precedence, dt = 0) are inherited unchanged; only three private hooks
are replaced: `_column` (the accepted column stage, see below), `_explicit` and `_local_inertial` (the lateral stage, compiled).
The shared driver `experimental_storm.evolve_experimental` serves it unchanged (adaptive retry, snapshots, continuation cap).

Column stage: NOT repeated here. Every step calls the accepted `hydrology_numba.prepared_column_step` on a
`hydrology_numba.HydrologyContext` prepared ONCE in the constructor (no infiltration formula appears in this module).

Lateral stage, term by term as in the reference: explicit q = k h_c^(3/2), F = dt dx q used once, donors gathered in the fixed
`DONOR_SLOTS` order with missing donors adding 0.0, h_new = h_c + (inflow - out)/dx^2, characteristic CFL (3/2) k sqrt(h_c) dt/dx,
`h_new < 0` and per-cell balance flags; local inertia face update (`h_f`, semi-implicit Darcy friction, normal-flow outlet),
donor limiter `phi = min(1, avail/out)(1 - 16 eps)` computed from the UNLIMITED fluxes of all cells before any face is scaled,
continuity, cell speed, balances, wave CFL. NO fastmath, no FMA contraction, no prange, no clipping: only `+ - * /`, `sqrt` and
comparisons (identical IEEE operations in identical order give identical bits). That bitwise statement is NARROW: it holds for the
LATERAL stage given IDENTICAL column outputs and state (root trial1: all 7440 accepted local steps of the 5400 s storm, fed the
NumPy column and state, match every public field bitwise) and in the tests where the column is the identity (ksat = 0). It is NOT a
claim for a whole run: the compiled column (LLVM/libm `expm1`) may differ from NumPy's by an ulp, and independent evolving
trajectories amplify such rounding where the local-inertial dynamics are poorly conditioned. Measured (root, not by the author):
the explicit 5400 s storm passes every public step and driver field at the unchanged bound (rtol 2e-12 / atol 1e-14); the local
5400 s storm closes water with equal accepted/rejected counts (7440/2040) but some fields exceed that bound (soil water, cumulative
intake, hydrograph, peak velocity ~6e-4 m/s, three snapshot velocity maps); in the 90 s CLI run only final_state.npz velocity_m_s
fails (3 cells, max abs 2.4673e-14 m/s, worst normalized 2.086) and every other saved field passes. The bound is not widened and
this is not an erosion-qualified velocity. The scalar sums
(storage change, export, balance scale, outlet discharge, limited volume) are `numpy.sum` on kernel-produced 1-D operand rows,
so the pairwise order is the reference's. Everything the reference derives from a NumPy reduction that propagates NaN
(`max`) is derived the same way.

Errors: the kernels record the same flag bits as the reference (`EXPLICIT_BITS` / `LOCAL_BITS` values are taken from
`experimental_hydrology`) and `resolve_flags` raises the first failure in the reference's precedence with its messages;
`HydraulicStepRejected` is the only recoverable class. Failures raise before a result exists, every output is a fresh array and
no input is written. Documented difference: the column stage errors come from the prepared compiled column (`InfiltrationError`,
same conditions and precedence as `column_step`), whose messages for a wrong TYPE/dtype/shape of depth, soil or rain differ from
the NumPy column's wording (both are `InfiltrationError`).

Array policy (CPU): state and column arrays must be exact host `numpy.ndarray` float64 of the right shape (the inherited public
checks and the prepared column refuse subclasses, masked arrays, other dtypes and CuPy: no conversion, no transfer, no mask can
launder a non-finite value). A strided/Fortran-ordered or READ-ONLY array is accepted and COPIED ONCE per step to a C-contiguous
writable flat array (never written, never retained), so results are identical to the contiguous case; nothing is refused for
layout. Outputs are freshly allocated C-contiguous arrays.

Static data and guards: the solver owns contiguous READ-ONLY flat copies of everything the kernels read (explicit: active,
outlet, k, donors; local inertia: bed z and every face table) plus the prepared column context, built once and sealed with
pointer/shape/dtype/flag fingerprints. The solver is immutable after construction (`__setattr__` refuses) and BEFORE every compiled
call the seal is re-checked (method, shape, dx, CFL, limiter, identity of both contexts, every array's fingerprint and read-only
flag, the column context's scalar metadata and the arrays the column kernel indexes); a forged context or attribute is refused with
`ExperimentalHydrologyError` before any kernel runs. No static array is hashed per step. The contexts are immutable BY CONTRACT:
in-place content mutation of an owned array (possible only by re-enabling its write flag) is not detected; the column context's own
limitations (`hydrology_numba` module docstring) apply. A new graph, parameters, geometry or control requires a NEW solver.

Scratch: each step allocates its own bounded operand rows and outputs (a few `n_cells` arrays); nothing grows and nothing is
shared between calls, so a solver may be used from several threads. The kernels are compiled lazily on first use (`cache=False`,
`nogil=True`, `boundscheck=False`, `error_model="numpy"`): importing this module imports neither Numba nor CuPy, and a missing Numba
raises `ExperimentalNumbaUnavailableError` (there is no fallback to the NumPy reference; select it explicitly).

Qualification: the compiled forms are NEW and unqualified beyond the tests that accompany them. They compile unchanged science;
the local-inertial long-storm conditioning, velocity and Froude limitations documented in `experimental_hydrology` and the README
are NOT addressed here.

Nothing here was run by its author (file-only tools); Codex records results.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from maple_syrup import hydrology_numba as hn
from maple_syrup.experimental_hydrology import (
    CFL_NAMES,
    EXPLICIT_BITS,
    LIMITER_SAFETY,
    LOCAL_BITS,
    QUALIFICATION_STATUS,
    CpuHydraulicSolver,
    ExperimentalHydrologyError,
    HydraulicControl,
    HydraulicState,
    HydraulicStep,
    LocalInertialGeometry,
    donor_cells_by_cell,
    resolve_flags,
)
from maple_syrup.infiltration import ColumnParameters, ColumnStep
from maple_syrup.routing import BALANCE_RTOL, GRAVITY_M_S2, RoutingGraph
from maple_syrup.routing_numba import (
    NumbaUnavailableError,
    numba_available,
    numba_versions,
)

__all__ = [
    "NUMBA_TRANSFER_SCOPE",
    "ExperimentalNumbaUnavailableError",
    "NumbaExperimentalContext",
    "NumbaHydraulicSolver",
    "kernel_provenance",
    "prepare_experimental_numba",
    "reset_compiled",
]

NUMBA_TRANSFER_SCOPE = (
    "Host-only CPU implementation: no device, no host/device transfer and no MAPLE transfer counter is involved. Each step "
    "allocates its own bounded outputs and operand rows; a strided or read-only state array is copied once per step to a "
    "C-contiguous writable flat array (never written, never retained)."
)
_KERNELS: Any = None


class ExperimentalNumbaUnavailableError(NumbaUnavailableError, ExperimentalHydrologyError):
    """The compiled CPU hydraulics were requested but Numba cannot be imported. No fallback to the NumPy reference."""


# --- kernels (Numba is imported lazily by the caller) -------------------------------------------------------------------
def _build_kernels(numba: Any) -> SimpleNamespace:
    jit = numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")
    grav = float(GRAVITY_M_S2)
    safety = float(LIMITER_SAFETY)
    e_nonfinite, e_negative, e_balance = (int(EXPLICIT_BITS[k]) for k in ("nonfinite", "negative", "balance"))
    b_state, b_closed, b_inflow = (int(LOCAL_BITS[k]) for k in ("state_nonfinite", "closed_face", "open_inflow"))
    b_negative, b_nonfinite, b_balance = (int(LOCAL_BITS[k]) for k in ("negative", "nonfinite", "balance"))

    @jit
    def explicit_kernel(hc, active, outlet, k, donors, dt_over_dx, dtdx_area, area, rtol,
                        q_used, out_vol, h_new, velocity, scratch, maxima):
        """Explicit kinematic wave on flat cell arrays. `scratch` rows (the operands of the reference sums): 0 h_new - h_c,
        1 outlet face volume, 2 balance scale, 3 outlet instantaneous flux. maxima: characteristic CFL, balance residual."""
        n = hc.shape[0]
        flags = np.int64(0)
        max_cfl = -np.inf
        for i in range(n):
            h = hc[i]
            sq = np.sqrt(h)
            if active[i]:
                kk = k[i]
                qu = (sq * h) * kk
                cfl = ((1.5 * kk) * sq) * dt_over_dx
                ov = dtdx_area * qu
            else:
                qu = 0.0
                cfl = 0.0
                ov = 0.0
            q_used[i] = qu
            out_vol[i] = ov
            if cfl > max_cfl or cfl != cfl:  # noqa: PLR0124  (np.max: NaN propagates)
                max_cfl = cfl
        max_bal = -np.inf
        for i in range(n):
            h = hc[i]
            ov = out_vol[i]
            d = donors[0, i]
            inflow = out_vol[d] if d >= 0 else 0.0
            d = donors[1, i]
            inflow = inflow + (out_vol[d] if d >= 0 else 0.0)
            d = donors[2, i]
            inflow = inflow + (out_vol[d] if d >= 0 else 0.0)
            d = donors[3, i]
            inflow = inflow + (out_vol[d] if d >= 0 else 0.0)
            net = (inflow - ov) / area
            hn = h + net
            if active[i]:
                qi = (np.sqrt(hn) * hn) * k[i]
                bal = abs((hn - h) - net)
                sc = (h + hn) + ((inflow + ov) / area)
            else:
                qi = 0.0
                bal = 0.0
                sc = 0.0
            vel = qi / hn if hn > 0.0 else 0.0
            if not (np.isfinite(hn) and np.isfinite(ov) and np.isfinite(q_used[i]) and np.isfinite(qi)
                    and np.isfinite(vel) and np.isfinite(sc) and np.isfinite(bal)):
                flags |= e_nonfinite
            if hn < 0.0:
                flags |= e_negative
            if bal > rtol * sc:
                flags |= e_balance
            if bal > max_bal or bal != bal:  # noqa: PLR0124
                max_bal = bal
            h_new[i] = hn
            velocity[i] = vel
            scratch[0, i] = hn - h
            scratch[1, i] = ov if outlet[i] else 0.0
            scratch[2, i] = sc
            scratch[3, i] = qi if outlet[i] else 0.0
        maxima[0] = max_cfl
        maxima[1] = max_bal
        return flags

    @jit
    def local_kernel(ny, nx, hc, z, active, fx_type, fx_zmax, fx_fric, fx_kb, fx_sign, fy_type, fy_zmax, fy_fric, fy_kb,
                     fy_sign, qx_old, qy_old, dt, dx, dt_over_dx, dtdx_area, area, rtol, donor_limiter,
                     qx, qy, hfx, hfy, h_new, speed, fvx, fvy, ox, oy, scratch, maxima, counts):
        """Local inertia on flat arrays: x faces (ny, nx+1), y faces (ny+1, nx), cells (ny, nx), all row-major. `scratch`
        rows: 0 h_new - h_c, 1 balance scale, 2 limiter phi, 3 limited-volume operand. maxima: balance residual, max h_f on the
        x faces, max h_f on the y faces. counts[0] = limited cells."""
        flags = np.int64(0)
        nxf = nx + 1
        max_hfx = -np.inf
        for r in range(ny):  # ---- x faces: A = west cell, B = east cell (the ring side is the zero pad of the reference)
            for j in range(nxf):
                f = r * nxf + j
                qo = qx_old[f]
                ft = fx_type[f]
                if not np.isfinite(qo):
                    flags |= b_state
                if ft == 0 and qo != 0.0:
                    flags |= b_closed
                if ft == 2 and qo * float(fx_sign[f]) < 0.0:
                    flags |= b_inflow
                q = 0.0
                hf = 0.0
                if ft == 1 or ft == 2:
                    if j > 0:
                        ca = r * nx + j - 1
                        h_a = hc[ca]
                        eta_a = z[ca] + h_a
                    else:
                        h_a = 0.0
                        eta_a = 0.0
                    if j < nx:
                        cb = r * nx + j
                        h_b = hc[cb]
                        eta_b = z[cb] + h_b
                    else:
                        h_b = 0.0
                        eta_b = 0.0
                    if ft == 1:
                        top = eta_a if eta_a > eta_b else eta_b  # noqa: FURB136  (np.where(eta_a > eta_b, ..): NaN selects eta_b)
                        xx = top - fx_zmax[f]
                        hf1 = xx if (xx >= 0.0 or xx != xx) else 0.0  # noqa: PLR0124  (np.maximum(x, 0.0))
                        grad = (eta_b - eta_a) / dx
                        numerator = qo - (((grav * hf1) * dt) * grad)
                        if qo != 0.0:
                            fr = ((dt * (fx_fric[f] / 8.0)) * abs(qo)) / (hf1 * hf1)
                        else:
                            fr = 0.0
                        if hf1 > 0.0:
                            q = numerator / (1.0 + fr)
                        hf = hf1
                    else:
                        donor = h_a if fx_sign[f] > 0 else h_b
                        qout = (np.sqrt(donor) * donor) * fx_kb[f]
                        q = qout if fx_sign[f] > 0 else -qout
                        hf = donor
                qx[f] = q
                hfx[f] = hf
                if hf > max_hfx or hf != hf:  # noqa: PLR0124
                    max_hfx = hf
        max_hfy = -np.inf
        for i in range(ny + 1):  # ---- y faces: A = south cell, B = north cell
            for c in range(nx):
                g = i * nx + c
                qo = qy_old[g]
                ft = fy_type[g]
                if not np.isfinite(qo):
                    flags |= b_state
                if ft == 0 and qo != 0.0:
                    flags |= b_closed
                if ft == 2 and qo * float(fy_sign[g]) < 0.0:
                    flags |= b_inflow
                q = 0.0
                hf = 0.0
                if ft == 1 or ft == 2:
                    if i > 0:
                        ca = (i - 1) * nx + c
                        h_a = hc[ca]
                        eta_a = z[ca] + h_a
                    else:
                        h_a = 0.0
                        eta_a = 0.0
                    if i < ny:
                        cb = i * nx + c
                        h_b = hc[cb]
                        eta_b = z[cb] + h_b
                    else:
                        h_b = 0.0
                        eta_b = 0.0
                    if ft == 1:
                        top = eta_a if eta_a > eta_b else eta_b  # noqa: FURB136  (same NaN-selecting np.where semantics)
                        xx = top - fy_zmax[g]
                        hf1 = xx if (xx >= 0.0 or xx != xx) else 0.0  # noqa: PLR0124
                        grad = (eta_b - eta_a) / dx
                        numerator = qo - (((grav * hf1) * dt) * grad)
                        if qo != 0.0:
                            fr = ((dt * (fy_fric[g] / 8.0)) * abs(qo)) / (hf1 * hf1)
                        else:
                            fr = 0.0
                        if hf1 > 0.0:
                            q = numerator / (1.0 + fr)
                        hf = hf1
                    else:
                        donor = h_a if fy_sign[g] > 0 else h_b
                        qout = (np.sqrt(donor) * donor) * fy_kb[g]
                        q = qout if fy_sign[g] > 0 else -qout
                        hf = donor
                qy[g] = q
                hfy[g] = hf
                if hf > max_hfy or hf != hf:  # noqa: PLR0124
                    max_hfy = hf
        if donor_limiter:  # ---- donor limiter: phi of EVERY cell from the unlimited fluxes, then the faces are scaled
            n_limited = np.int64(0)
            for r in range(ny):
                for c in range(nx):
                    idx = r * nx + c
                    west = qx[r * nxf + c]
                    east = qx[r * nxf + c + 1]
                    south = qy[r * nx + c]
                    north = qy[(r + 1) * nx + c]
                    # np.where(x > 0, x, 0.0) of the reference (a NaN flux gives 0.0, which max() would not reproduce)
                    wv = -west if -west > 0.0 else 0.0  # noqa: FURB136
                    ev = east if east > 0.0 else 0.0  # noqa: FURB136
                    sv = -south if -south > 0.0 else 0.0  # noqa: FURB136
                    nv = north if north > 0.0 else 0.0  # noqa: FURB136
                    out = (wv + ev) + (sv + nv)
                    out_v = dtdx_area * out
                    avail = hc[idx] * area
                    if out_v > avail:
                        phi = (avail / out_v) * safety
                        n_limited += 1
                        scratch[3, idx] = out_v * (1.0 - phi)
                    else:
                        phi = 1.0
                        scratch[3, idx] = 0.0
                    scratch[2, idx] = phi
            counts[0] = n_limited
            for r in range(ny):
                for j in range(nxf):
                    f = r * nxf + j
                    q = qx[f]
                    if q > 0.0:
                        qx[f] = q * (scratch[2, r * nx + j - 1] if j > 0 else 1.0)  # phi of A (the donor side)
                    elif q < 0.0:
                        qx[f] = q * (scratch[2, r * nx + j] if j < nx else 1.0)  # phi of B
            for i in range(ny + 1):
                for c in range(nx):
                    g = i * nx + c
                    q = qy[g]
                    if q > 0.0:
                        qy[g] = q * (scratch[2, (i - 1) * nx + c] if i > 0 else 1.0)
                    elif q < 0.0:
                        qy[g] = q * (scratch[2, i * nx + c] if i < ny else 1.0)
        else:
            counts[0] = 0
        max_bal = -np.inf
        for r in range(ny):  # ---- continuity, cell speed, checks
            for c in range(nx):
                idx = r * nx + c
                west = qx[r * nxf + c]
                east = qx[r * nxf + c + 1]
                south = qy[r * nx + c]
                north = qy[(r + 1) * nx + c]
                h = hc[idx]
                div = (east - west) + (north - south)
                hn = h - (dt_over_dx * div)
                if hn > 0.0:
                    ux = (0.5 * (west + east)) / hn
                    uy = (0.5 * (south + north)) / hn
                else:
                    ux = 0.0
                    uy = 0.0
                sp = np.sqrt(ux * ux + uy * uy)
                if active[idx]:
                    bal = abs((hn - h) + (dt_over_dx * div))
                    sc = (h + hn) + dt_over_dx * ((abs(west) + abs(east)) + (abs(south) + abs(north)))
                else:
                    bal = 0.0
                    sc = 0.0
                if hn < 0.0:
                    flags |= b_negative
                if not (np.isfinite(west) and np.isfinite(east) and np.isfinite(south) and np.isfinite(north)
                        and np.isfinite(hn) and np.isfinite(sp) and np.isfinite(sc) and np.isfinite(bal)):
                    flags |= b_nonfinite
                if bal > rtol * sc:
                    flags |= b_balance
                if bal > max_bal or bal != bal:  # noqa: PLR0124
                    max_bal = bal
                h_new[idx] = hn
                speed[idx] = sp
                scratch[0, idx] = hn - h
                scratch[1, idx] = sc
        for f in range(qx.shape[0]):  # ---- face volumes and the outward (outlet) flux operands
            q = qx[f]
            fvx[f] = dtdx_area * q
            ox[f] = q * float(fx_sign[f]) if fx_type[f] == 2 else 0.0
        for g in range(qy.shape[0]):
            q = qy[g]
            fvy[g] = dtdx_area * q
            oy[g] = q * float(fy_sign[g]) if fy_type[g] == 2 else 0.0
        maxima[0] = max_bal
        maxima[1] = max_hfx
        maxima[2] = max_hfy
        return flags

    return SimpleNamespace(explicit=explicit_kernel, local=local_kernel)


def _kernels() -> SimpleNamespace:
    global _KERNELS
    if _KERNELS is None:
        if not numba_available():
            raise ExperimentalNumbaUnavailableError(
                "the compiled CPU hydraulics require Numba (optional extra maple-syrup[numba]), which is not importable; "
                "there is no fallback to the NumPy reference (select --implementation numpy / CpuHydraulicSolver explicitly)")
        import numba

        _KERNELS = _build_kernels(numba)
    return _KERNELS


def reset_compiled() -> None:
    """Drop the compiled dispatchers (tests: cold-start and missing-Numba paths)."""
    global _KERNELS
    _KERNELS = None


def kernel_provenance() -> dict[str, Any]:
    """Host-side description of the compiled kernels for reports (no compilation)."""
    here = Path(__file__).resolve()
    return {
        "implementation": "numba",
        "module": "maple_syrup.experimental_numba",
        "module_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        "kernels": ["explicit_kernel", "local_kernel"],
        "column": "hydrology_numba.prepared_column_step on a HydrologyContext prepared once (no infiltration formula here)",
        "hydrology_numba_sha256": hashlib.sha256(here.with_name("hydrology_numba.py").read_bytes()).hexdigest(),
        "numba_options": {"fastmath": False, "parallel": False, "nogil": True, "boundscheck": False,
                          "error_model": "numpy", "cache": False},
        "scalar_reductions": "numpy.sum on kernel-produced 1-D operand rows (the reference's pairwise order)",
        "elementwise": "+ - * /, sqrt and comparisons only, no FMA contraction: the lateral stage is bitwise equal to the NumPy "
                       "reference for identical column inputs and state (root-measured); the compiled column (libm expm1) may "
                       "differ by an ulp and independent trajectories amplify it where local inertia is ill-conditioned",
        "versions": numba_versions(),
        "compiled_in_process": _KERNELS is not None,
        "compilation": "lazy: the first step includes JIT compilation of the lateral kernel (and of the column kernels)",
    }


# --- static context -------------------------------------------------------------------------------------------------
def _owned_flat(array: Any, name: str, dtype: Any) -> np.ndarray:
    """Contiguous READ-ONLY flat owned copy of a host array (never aliases the caller's array)."""
    if type(array) is not np.ndarray:
        raise ExperimentalHydrologyError(f"{name} must be a host numpy.ndarray, got {type(array).__name__}")
    out = np.ascontiguousarray(array, dtype=dtype).reshape(-1).copy()
    out.flags.writeable = False
    return out


def _fingerprint(name: str, array: np.ndarray) -> tuple:
    return (name, int(array.__array_interface__["data"][0]), tuple(array.shape), array.dtype.str)


def _check_array(name: str, array: Any, fingerprint: tuple, *, read_only: bool) -> None:
    if (type(array) is not np.ndarray or not array.flags.c_contiguous or (read_only and array.flags.writeable)
            or _fingerprint(name, array) != fingerprint):
        raise ExperimentalHydrologyError(
            f"static array {name!r} no longer matches the values sealed at preparation (replaced, resized, re-typed or made "
            "writable); refusing before any compiled call")


@dataclass(frozen=True, eq=False)
class NumbaExperimentalContext:
    """Owned, read-only static data of one compiled solver. Immutable by contract; build with `prepare_experimental_numba`."""

    method: str
    shape: tuple[int, int]
    n_cells: int
    n_active: int
    dx_m: float
    arrays: dict
    fingerprints: tuple
    graph_input_sha256: str
    geometry_sha256: Any
    static_bytes: int
    preparation_s: float

    def array(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def summary(self) -> dict[str, Any]:
        return {"method": self.method, "shape": list(self.shape), "n_cells": self.n_cells, "n_active": self.n_active,
                "dx_m": self.dx_m, "arrays": sorted(self.arrays), "static_bytes": self.static_bytes,
                "preparation_s": self.preparation_s, "graph_input_sha256": self.graph_input_sha256,
                "geometry_sha256": self.geometry_sha256, "host_only": True, "fastmath": False,
                "ownership": "contiguous read-only flat copies; no alias of the caller's graph/geometry arrays; the solver "
                             "is sealed (pointer/shape/dtype fingerprints) and re-checks the seal before every compiled call",
                "immutability": "by contract: in-place content mutation of an owned array is not detected"}


def prepare_experimental_numba(method: str, graph: RoutingGraph, geometry: LocalInertialGeometry | None = None,
                               ) -> NumbaExperimentalContext:
    """Validate and copy the lateral static data once. Raises `ExperimentalHydrologyError` before anything is returned."""
    t0 = time.perf_counter()
    if method not in ("explicit", "local_inertial"):
        raise ExperimentalHydrologyError(f"method must be 'explicit' or 'local_inertial', got {method!r}")
    ny, nx = (int(v) for v in graph.shape)
    n = ny * nx
    dx = float(graph.dx_m)
    arrays: dict[str, np.ndarray] = {"active": _owned_flat(np.asarray(graph.active), "graph.active", np.bool_)}
    if arrays["active"].size != n:
        raise ExperimentalHydrologyError("graph.active does not have shape (ny, nx)")
    if method == "explicit":
        donors = np.ascontiguousarray(donor_cells_by_cell(graph), dtype=np.int64)
        if donors.shape != (4, n) or donors.min() < -1 or donors.max() >= n:
            raise ExperimentalHydrologyError("donor table outside [-1, n_cells) or of the wrong shape")
        donors = donors.copy()
        donors.flags.writeable = False
        arrays["donors"] = donors
        arrays["outlet"] = _owned_flat(np.asarray(graph.outlet), "graph.outlet", np.bool_)
        arrays["k"] = _owned_flat(np.asarray(graph.conveyance), "graph.conveyance", np.float64)
        for name in ("outlet", "k"):
            if arrays[name].size != n:
                raise ExperimentalHydrologyError(f"graph.{name} does not have n_cells entries")
    else:
        if not isinstance(geometry, LocalInertialGeometry):
            raise ExperimentalHydrologyError("the local-inertial method needs a LocalInertialGeometry")
        nxf, nyf = ny * (nx + 1), (ny + 1) * nx
        spec = (("z", geometry.z, np.float64, n), ("fx_type", geometry.fx_type, np.int8, nxf),
                ("fx_zmax", geometry.fx_zmax, np.float64, nxf), ("fx_fric", geometry.fx_fric, np.float64, nxf),
                ("fx_kb", geometry.fx_kb, np.float64, nxf), ("fx_sign", geometry.fx_sign, np.int8, nxf),
                ("fy_type", geometry.fy_type, np.int8, nyf), ("fy_zmax", geometry.fy_zmax, np.float64, nyf),
                ("fy_fric", geometry.fy_fric, np.float64, nyf), ("fy_kb", geometry.fy_kb, np.float64, nyf),
                ("fy_sign", geometry.fy_sign, np.int8, nyf))
        for name, source, dtype, size in spec:
            if np.asarray(source).dtype != np.dtype(dtype):
                raise ExperimentalHydrologyError(f"geometry.{name} must have dtype {np.dtype(dtype)}, got {source.dtype}")
            arrays[name] = _owned_flat(source, f"geometry.{name}", dtype)
            if arrays[name].size != size:
                raise ExperimentalHydrologyError(f"geometry.{name} has {arrays[name].size} entries, expected {size}")
    fingerprints = tuple(_fingerprint(name, arrays[name]) for name in sorted(arrays))
    return NumbaExperimentalContext(
        method=method, shape=(ny, nx), n_cells=n, n_active=int(np.count_nonzero(arrays["active"])), dx_m=dx, arrays=arrays,
        fingerprints=fingerprints, graph_input_sha256=str(graph.input_sha256),
        geometry_sha256=None if geometry is None else geometry.input_sha256,
        static_bytes=int(sum(a.nbytes for a in arrays.values())), preparation_s=time.perf_counter() - t0)


# --- the solver -----------------------------------------------------------------------------------------------------
def _flat64(array: Any) -> np.ndarray:
    """Flat C-contiguous WRITABLE view of a float64 array (a copy only if the layout is strided or the array read-only: Numba
    types read-only arrays differently, so one specialisation is kept). The kernels never write it."""
    flat = np.ascontiguousarray(array).reshape(-1)
    return flat if flat.flags.writeable else flat.copy()


_HYDROLOGY_ARRAYS = (("active", np.bool_), ("conveyance", np.float64), ("column_static", np.float64))


class NumbaHydraulicSolver(CpuHydraulicSolver):
    """The compiled CPU form of `CpuHydraulicSolver` (see the module docstring). Same constructor arguments, same public
    methods, same `HydraulicStep`; `implementation == "numba"`. Immutable after construction."""

    implementation = "numba"

    def __init__(self, method: str, graph: RoutingGraph, params: ColumnParameters, *,
                 geometry: LocalInertialGeometry | None = None, control: HydraulicControl | None = None):
        if not numba_available():
            raise ExperimentalNumbaUnavailableError(
                "the compiled CPU hydraulics require Numba (optional extra maple-syrup[numba]), which is not importable; "
                "there is no fallback to the NumPy reference (select it explicitly)")
        super().__init__(method, graph, params, geometry=geometry, control=control)  # all public validation, as the reference
        self._context = prepare_experimental_numba(method, graph, geometry)
        self._hydrology = hn.prepare_hydrology(graph, params)  # the accepted column context, prepared ONCE
        hyd = self._hydrology
        if (hyd.shape != self._context.shape or hyd.n_cells != self._context.n_cells or hyd.dx_m != self._context.dx_m):
            raise ExperimentalHydrologyError("the prepared column context does not match the lateral context")
        self._hydrology_fingerprints = tuple(_fingerprint(name, getattr(hyd, name)) for name, _ in _HYDROLOGY_ARRAYS)
        self._seal = self._current_seal()
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_sealed"):
            raise AttributeError(f"NumbaHydraulicSolver is immutable after construction (cannot set {name!r}); build a new "
                                 "solver")
        object.__setattr__(self, name, value)

    @property
    def context(self) -> NumbaExperimentalContext:
        return self._context

    @property
    def hydrology_context(self) -> hn.HydrologyContext:
        return self._hydrology

    def _current_seal(self) -> tuple:
        ctx, hyd = self._context, self._hydrology
        ctl = self.control
        return (ctx.method, ctx.shape, ctx.dx_m, ctx.n_cells, ctx.n_active, ctl.cfl_max, ctl.limiter, id(ctx), id(hyd),
                hyd.shape, hyd.n_cells, hyd.dx_m, hyd.model_code)

    def _guard(self, method: str) -> tuple[NumbaExperimentalContext, hn.HydrologyContext, float, str]:
        """BEFORE any compiled call: type, identity and fingerprint checks of everything a kernel indexes, and that the public
        attributes still equal the sealed values. Returns `(context, column context, cfl_max, limiter)` taken from the SEAL,
        never from mutable attributes."""
        ctx, hyd = self._context, self._hydrology
        if not isinstance(ctx, NumbaExperimentalContext) or not isinstance(hyd, hn.HydrologyContext):
            raise ExperimentalHydrologyError("the solver's contexts are not the prepared context types; refusing before any "
                                             "compiled call")
        if not isinstance(self.control, HydraulicControl):
            raise ExperimentalHydrologyError("the solver's control is not a HydraulicControl; refusing before any compiled call")
        if self._current_seal() != self._seal or (self.method, self.shape, self.dx_m) != (ctx.method, ctx.shape, ctx.dx_m):
            raise ExperimentalHydrologyError("the solver's method, shape, dx, control or contexts no longer match the values "
                                             "sealed at construction; refusing before any compiled call")
        if ctx.method != method:
            raise ExperimentalHydrologyError(f"the solver was prepared for {ctx.method!r}, not {method!r}")
        if sorted(ctx.arrays) != [fingerprint[0] for fingerprint in ctx.fingerprints]:
            raise ExperimentalHydrologyError("the static array set differs from the sealed one; refusing")
        for fingerprint in ctx.fingerprints:
            _check_array(fingerprint[0], ctx.arrays.get(fingerprint[0]), fingerprint, read_only=True)
        for (name, _dtype), fingerprint in zip(_HYDROLOGY_ARRAYS, self._hydrology_fingerprints, strict=True):
            _check_array(f"hydrology.{name}", getattr(hyd, name), (f"hydrology.{name}", *fingerprint[1:]), read_only=True)
        n = ctx.n_cells
        if (hyd.active.shape != (n,) or hyd.conveyance.shape != (n,) or hyd.column_static.shape != (n, 6)
                or hyd.n_cells != n or hyd.shape != ctx.shape):
            raise ExperimentalHydrologyError("the column context's arrays do not match the lateral context; refusing")
        return ctx, hyd, float(self.control.cfl_max), str(self.control.limiter)

    def _column(self, rain_rate_m_per_s: Any, state: HydraulicState, dt_s: float) -> ColumnStep:
        _ctx, hyd, _cfl, _limiter = self._guard(self.method)
        return hn.prepared_column_step(hyd, state.depth_m, state.soil_water_m, rain_rate_m_per_s, dt_s)

    @staticmethod
    def _cell_input(depth: Any, n: int) -> np.ndarray:
        flat = _flat64(depth)
        if flat.shape != (n,) or flat.dtype != np.float64:
            raise ExperimentalHydrologyError("column depth does not match the prepared context; refusing")
        return flat

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update({
            "context": self._context.summary(), "hydrology_context": self._hydrology.summary(),
            "kernels": kernel_provenance(), "transfer_scope": NUMBA_TRANSFER_SCOPE,
            "array_policy": "exact host float64 ndarrays; strided or read-only state arrays are copied once per step to "
                            "C-contiguous writable flat arrays (never written); outputs are fresh C-contiguous arrays",
            "qualification": QUALIFICATION_STATUS,
        })
        return info

    # -- explicit -----------------------------------------------------------------------------------------------------
    def _explicit(self, col: ColumnStep, state: HydraulicState, dt: float, t_new: float) -> HydraulicStep:
        ctx, _hyd, cfl_limit, _limiter = self._guard("explicit")
        n, dx = ctx.n_cells, ctx.dx_m
        shape = ctx.shape
        area, dtdx_area, dt_over_dx = dx * dx, dt * dx, dt / dx
        hc = self._cell_input(col.depth_m, n)
        q_used, out_vol, h_new, velocity = (np.empty(n, dtype=np.float64) for _ in range(4))
        scratch = np.empty((4, n), dtype=np.float64)  # operand rows of the four reference sums
        maxima = np.empty(2, dtype=np.float64)
        flags = int(_kernels().explicit(
            hc, ctx.array("active"), ctx.array("outlet"), ctx.array("k"), ctx.array("donors"), dt_over_dx, dtdx_area, area,
            BALANCE_RTOL, q_used, out_vol, h_new, velocity, scratch, maxima))
        with np.errstate(all="ignore"):
            storage = area * scratch[0].sum()
            export = scratch[1].sum()
            residual = storage + export
            gtol = BALANCE_RTOL * (ctx.n_active + 2) * area * scratch[2].sum()
            outlet_q = dx * scratch[3].sum()
        max_cfl = float(maxima[0])
        if max_cfl > cfl_limit:
            flags |= EXPLICIT_BITS["cfl"]
        scalars = tuple(not math.isfinite(float(v)) for v in (storage, export, residual, gtol, outlet_q))
        resolve_flags("explicit", flags, scalars, bool(abs(residual) > gtol), max_cfl, cfl_limit, "off")
        new_state = HydraulicState(t_new, h_new.reshape(shape), col.soil_water_m)
        return HydraulicStep(
            method="explicit", implementation=self.implementation, dt_s=dt, state=new_state, column=col,
            velocity_m_s=velocity.reshape(shape), face_volume_m3={"out": out_vol.reshape(shape)},
            used_flux_m2_s={"out": q_used.reshape(shape)}, export_m3=export, outlet_discharge_m3_s=outlet_q,
            storage_change_m3=storage, budget_residual_m3=residual, max_cfl=max_cfl,
            max_cell_balance_residual_m=float(maxima[1]), limited_cells=0, limited_volume_m3=0.0,
            cfl_kind=CFL_NAMES["explicit"], face_flow_depth_m={"out": col.depth_m})

    # -- local inertia ------------------------------------------------------------------------------------------------
    def _local_inertial(self, col: ColumnStep, state: HydraulicState, dt: float, t_new: float) -> HydraulicStep:
        ctx, _hyd, cfl_limit, limiter = self._guard("local_inertial")
        ny, nx = ctx.shape
        n, dx = ctx.n_cells, ctx.dx_m
        nxf, nyf = ny * (nx + 1), (ny + 1) * nx
        area, dtdx_area, dt_over_dx = dx * dx, dt * dx, dt / dx
        hc = self._cell_input(col.depth_m, n)
        qx_old, qy_old = _flat64(state.qx_m2_s), _flat64(state.qy_m2_s)  # exact (ny, nx+1) / (ny+1, nx), checked by `step`
        if qx_old.shape != (nxf,) or qy_old.shape != (nyf,):
            raise ExperimentalHydrologyError("face fluxes do not match the prepared context; refusing")
        qx, hfx, fvx, ox = (np.empty(nxf, dtype=np.float64) for _ in range(4))
        qy, hfy, fvy, oy = (np.empty(nyf, dtype=np.float64) for _ in range(4))
        h_new, speed = (np.empty(n, dtype=np.float64) for _ in range(2))
        scratch = np.empty((4, n), dtype=np.float64)  # rows: h_new - h_c, balance scale, limiter phi, limited-volume operand
        maxima = np.empty(3, dtype=np.float64)
        counts = np.zeros(1, dtype=np.int64)
        donor = limiter == "donor"
        a = ctx.array
        flags = int(_kernels().local(
            ny, nx, hc, a("z"), a("active"), a("fx_type"), a("fx_zmax"), a("fx_fric"), a("fx_kb"), a("fx_sign"), a("fy_type"),
            a("fy_zmax"), a("fy_fric"), a("fy_kb"), a("fy_sign"), qx_old, qy_old, dt, dx, dt_over_dx, dtdx_area, area,
            BALANCE_RTOL, donor, qx, qy, hfx, hfy, h_new, speed, fvx, fvy, ox, oy, scratch, maxima, counts))
        limited_cells, limited_volume = 0, 0.0
        with np.errstate(all="ignore"):
            if donor:
                limited_cells = int(counts[0])
                limited_volume = float(scratch[3].sum())
            storage = area * scratch[0].sum()
            outward = ox.sum() + oy.sum()
            export = dtdx_area * outward
            residual = storage + export
            gtol = BALANCE_RTOL * (ctx.n_active + 2) * area * scratch[1].sum()
            outlet_q = dx * outward
            hf_max = max(float(maxima[1]), float(maxima[2]))
            max_cfl = float(dt_over_dx * np.sqrt(2.0 * GRAVITY_M_S2 * hf_max))
        if max_cfl > cfl_limit:
            flags |= LOCAL_BITS["cfl"]
        scalars = tuple(not math.isfinite(float(v)) for v in (storage, export, residual, gtol, outlet_q))
        resolve_flags("local_inertial", flags, scalars, bool(abs(residual) > gtol), max_cfl, cfl_limit, limiter)
        shape_x, shape_y, shape = (ny, nx + 1), (ny + 1, nx), (ny, nx)
        qx2, qy2 = qx.reshape(shape_x), qy.reshape(shape_y)
        new_state = HydraulicState(t_new, h_new.reshape(shape), col.soil_water_m, qx2, qy2)
        return HydraulicStep(
            method="local_inertial", implementation=self.implementation, dt_s=dt, state=new_state, column=col,
            velocity_m_s=speed.reshape(shape), face_volume_m3={"x": fvx.reshape(shape_x), "y": fvy.reshape(shape_y)},
            used_flux_m2_s={"x": qx2, "y": qy2}, export_m3=export, outlet_discharge_m3_s=outlet_q, storage_change_m3=storage,
            budget_residual_m3=residual, max_cfl=max_cfl, max_cell_balance_residual_m=float(maxima[0]),
            limited_cells=limited_cells, limited_volume_m3=limited_volume, cfl_kind=CFL_NAMES["local_inertial"],
            face_flow_depth_m={"x": hfx.reshape(shape_x), "y": hfy.reshape(shape_y)})
