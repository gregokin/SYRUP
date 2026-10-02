"""Prepared, fused CPU hydrology for the frozen-terrain legacy replay (Phase 7h).

`storm.coupled_step` (column -> branch -> routing, NumPy/CuPy) with `implementation="numba"` stays the
reference and is not touched. This module evaluates the SAME equations in two serial Numba kernels after a
one-time preparation of everything that is constant in a fixed-terrain replay (graph, column parameters):

    prepare_hydrology(graph, params) -> HydrologyContext        # once; validated, owned, read-only copies
    prepared_coupled_step(ctx, rain, state, dt, control)        # every step; same CoupledStep as coupled_step
    prepared_column_step(ctx, depth, soil, rain, dt)            # the column part alone; same ColumnStep

What is preserved (term by term, in the reference's expression and evaluation order): the Smith-Parlange
capacity and its limits, linear drainage and saturation return (`infiltration.column_step`); the legacy
branch precedence complete > no run-on > partial and the old-flux choice (`storm.coupled_step`); the coherent
donor sum, the Courant/old-flux checks, the `[0, R]` bisection (executed by the existing compiled
`routing_numba._sweep`, called from inside the kernel, not re-implemented), the storage identity and every
per-cell/global balance check (`routing._route`). No fastmath, no prange, no changed timestep, no relaxed
tolerance, no clipping. The scalar sums (storage change, export, balance scale, outlet discharge) are formed
by `numpy.sum` on kernel-produced operand rows so their pairwise summation order is exactly the reference's.

Phases (what a later device implementation would map one-to-one):

  A. `column_kernel`  cellwise, no cross-cell dependency: input checks, intake/drainage/return, output
     checks, branch masks, old flux q_old, branch counts, previous-discharge checks. -> one thread per cell.
  B1. `route_kernel` cellwise: routing input checks, implied-depth consistency, Courant number.
  B2. level-ordered: gather q_old into level order, coherent donor sum, base right-hand side; then the
     existing ordered sweep (levels in series, cells of one level independent, so one launch per level with
     same-level candidates in parallel, donors always at earlier levels).
  B3. cellwise scatter + outputs + checks: depth, flow, velocity, face volume, balances, maxima, operand rows.
  C.  host (Python): four `numpy.sum`, the scalar finiteness/global-balance checks and error resolution
     (lowest recorded failure first, the reference's precedence). On a device this is a small reduction
     plus one flag read per step.

State ownership and boundaries: the context owns contiguous read-only copies of all static data (flat,
cell-major; `column_static` is one `(n_cells, 6)` array = ksat, suction, drainage, thickness, Smax, lambda),
so a later mutation of the caller's graph/parameters cannot make it stale and a new graph (rerouting) or new
parameters (soil, model) require a NEW context. Dynamic inputs (depth, soil water, previous discharge, rain
rate) are read from the caller's arrays every step and never written, never retained, never converted: a
non-host-NumPy array (CuPy, masked array, other ndarray subclass) is REFUSED, so no hidden host/device
transfer and no mask can launder a non-finite value. Every output array is freshly allocated per call; the
step owns no mutable shared buffer (a context may be used from several threads). A failure raises before a
result exists and leaves every input untouched.

Error categories follow the reference: `InfiltrationError` for column-stage failures, `RoutingError` for
routing failures, `RoutingStepRejected` (subclass of `RoutingError`) only for the Courant and negative-RHS
rejections, `StormError` for a wrong control/state type. Precedence follows the reference too (column
failures first, then scalar-option failures, then routing failures, each in the reference's recorded order).
Documented differences, all STRICTER only for states the reference's own drivers never produce: the whole
previous-discharge array must be finite, non-negative and zero on inactive cells (the reference reads it only
through the branch and could be laundered by a skipped branch); `Smax = theta_sat * thickness` must be
positive and finite and equal that product exactly; static graph arrays are checked at preparation
(permutation, donor levels, receiver consistency) instead of trusted; only `StormControl.implementation
== "numba"` is accepted (the prepared path IS the numba sweep).

Floating point: identical IEEE operations (`+ - * /`, `sqrt`, comparisons) in identical order give identical
bits; `expm1` and `pow` (implied-depth check only) come from LLVM/libm instead of NumPy's loops and may
differ by an ulp. They affect only `capacity` near the Smith-Parlange limits and a threshold test, so water
agrees to a few ulp at most; the declared bound is rtol 2e-12 / atol 1e-14 and nothing here claims more.

Nothing here was run by its author (file-only tools); Codex records results. See docs/phase7h/hydrology_design.md.
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

from maple_syrup.infiltration import (
    INFILTRATION_MODELS,
    LOCAL_BALANCE_RTOL,
    ColumnParameters,
    ColumnStep,
    InfiltrationError,
    _check_dt,
)
from maple_syrup.routing import (
    _COURANT_REJECTION,
    _NEGATIVE_RHS_REJECTION,
    BALANCE_RTOL,
    EXPORT,
    RouteStep,
    RoutingError,
    RoutingGraph,
    RoutingStepRejected,
    _check_step_options,
)
from maple_syrup.routing_numba import (
    NumbaUnavailableError,
    compiled_sweep,
    numba_available,
    numba_versions,
)
from maple_syrup.storm import CoupledStep, StormControl, StormError, StormState

__all__ = [
    "HydrologyContext",
    "HydrologyNumbaUnavailableError",
    "HydrologyPreparationError",
    "kernel_provenance",
    "numba_available",
    "prepare_hydrology",
    "prepared_column_step",
    "prepared_coupled_step",
    "reset_compiled",
]

_KERNELS: Any = None
_EPS = float(np.finfo(np.float64).eps)
_EPS64 = 64.0 * _EPS  # routing._route: root_tol + 64 eps h_old
_TWO_THIRDS = 2.0 / 3.0
_MODEL_CODES = {"fixed_ksat": 0, "pavement_hawkins": 1}


class HydrologyPreparationError(ValueError):
    """The graph/column parameters cannot be prepared (wrong type/dtype/shape/namespace, inconsistent masks,
    out-of-range or unsafe static data). Raised before any context exists."""


class HydrologyNumbaUnavailableError(NumbaUnavailableError):
    """The prepared compiled hydrology was requested but Numba cannot be imported. No fallback to the
    NumPy/CuPy reference."""


# --- bit tables ------------------------------------------------------------------------------------------
# Column kernel flag bits, in the order `column_step` records its checks (the lowest set bit is reported).
_COLUMN_MESSAGES = (
    "depth_m must be finite everywhere",  # 0
    "depth_m must be >= 0 everywhere",  # 1
    "soil_water_m must be finite everywhere",  # 2
    "soil_water_m must be >= 0 everywhere",  # 3
    "rain_rate_m_per_s must be finite everywhere",  # 4
    "rain_rate_m_per_s must be >= 0 everywhere",  # 5
    "soil_water_m exceeds theta_sat * soil_thickness",  # 6
    "rain on an inactive (masked) cell",  # 7
    "step produced non-finite depth",  # 8   (bits >= 8 are output checks, skipped for dt == 0)
    "step produced negative depth",  # 9
    "step produced non-finite soil water",  # 10
    "step produced negative soil water",  # 11
    "step produced non-finite intake",  # 12
    "step produced negative intake",  # 13
    "step produced non-finite drainage",  # 14
    "step produced negative drainage",  # 15
    "step produced non-finite saturation return",  # 16
    "step produced negative saturation return",  # 17
    "local water balance h' + S' + D = h + S + P violated beyond FP64 tolerance",  # 18
)
_COLUMN_INPUT_MASK = (1 << 8) - 1

# Previous-discharge checks of the prepared step (stricter than the reference; reported last).
_QPREV_MESSAGES = (
    (
        "state.discharge_m2_s must be finite everywhere (the prepared step checks the whole array, also cells "
        "whose branch does not read it)"
    ),  # 0
    "state.discharge_m2_s must be >= 0 everywhere",  # 1
    "state.discharge_m2_s must be 0 on inactive cells",  # 2
)

# Route kernel flag bits, in the order `_route` records its checks. Bits 0..23 precede the host scalar
# checks, bits 24 and 25 follow them (see `_resolve_route`).
_ROUTE_NAMES = ("depth", "discharge", "flow depth", "velocity", "inflow", "old inflow", "face volume",
                "balance scale")
_ROUTE_STATIC_MESSAGES = (
    "depth_start_m must be finite everywhere",  # 0
    "depth_start_m must be >= 0 everywhere",  # 1
    "old_flow_depth_m must be finite everywhere",  # 2
    "old_flow_depth_m must be >= 0 everywhere",  # 3
    "old_discharge_m2_s must be finite everywhere",  # 4
    "old_discharge_m2_s must be >= 0 everywhere",  # 5
    "old_flow_depth_m (legacy post-infiltration d(1)) exceeds depth_start_m on an active cell",  # 6
    "old_discharge_m2_s must be 0 on inactive cells",  # 7
    "old_discharge_m2_s does not satisfy q = k h^{3/2} at old_flow_depth_m within root_tolerance_m",  # 8
    "old discharge q_old = k h_old^{3/2} overflowed FP64",  # 9
    None,  # 10 Courant (needs courant_max)
    "right-hand side overflowed FP64 (inflow or storage too large)",  # 11
    f"{_NEGATIVE_RHS_REJECTION} (legacy STOP condition); step rejected",  # 12
    *(f"step produced non-finite {name}" for name in _ROUTE_NAMES),  # 13..20
    *(f"step produced negative {name}" for name in _ROUTE_NAMES[:3]),  # 21..23
    None,  # 24 bisection (needs root_tol, iterations)
    "per-cell water balance violated beyond FP64 tolerance",  # 25
)
_SCALAR_NAMES = ("storage change", "export", "budget residual", "balance tolerance", "outlet discharge")
_LAST_PRE_SCALAR_BIT = 23


def _lowest_bit(flags: int) -> int:
    return (flags & -flags).bit_length() - 1


# --- kernels -----------------------------------------------------------------------------------------------
def _build_kernels(numba: Any) -> SimpleNamespace:
    """Define the nopython functions (Numba is imported lazily by the caller). `routing_numba._sweep` is
    reused through its own dispatcher, so the bisection/donor-sum arithmetic is literally the reference's."""
    sweep = compiled_sweep()  # raises NumbaUnavailableError if Numba is missing
    p23 = _TWO_THIRDS
    eps64 = _EPS64
    jit = numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False, error_model="numpy")

    @jit
    def fmax(a, b):  # np.maximum: NaN propagates
        if a != a or b != b:  # noqa: PLR0124
            return np.nan
        return a if a > b else b  # noqa: FURB136

    @jit
    def fmin(a, b):  # np.minimum: NaN propagates
        if a != a or b != b:  # noqa: PLR0124
            return np.nan
        return a if a < b else b  # noqa: FURB136

    @jit
    def column_kernel(h, s, r, qprev, active, k, col, hawkins, dt, local_rtol,
                      depth_new, soil_new, rain_o, intake_o, overflow_o, drainage_o, hpre_o, q_old_o, counts):
        """Phase A. Mirrors infiltration.column_step (validate=True) and the branch of storm.coupled_step."""
        n = h.shape[0]
        flags = np.int64(0)
        qflags = np.int64(0)
        n_no = np.int64(0)
        n_partial = np.int64(0)
        n_complete = np.int64(0)
        for i in range(n):
            hi = h[i]
            si = s[i]
            ri = r[i]
            a = active[i]
            ksat = col[i, 0]
            psi = col[i, 1]
            drn = col[i, 2]
            thick = col[i, 3]
            smax = col[i, 4]
            lam = col[i, 5]

            # inputs (column_step checks, in recorded order)
            if not np.isfinite(hi):
                flags |= np.int64(1) << 0
            if hi < 0.0:
                flags |= np.int64(1) << 1
            if not np.isfinite(si):
                flags |= np.int64(1) << 2
            if si < 0.0:
                flags |= np.int64(1) << 3
            if not np.isfinite(ri):
                flags |= np.int64(1) << 4
            if ri < 0.0:
                flags |= np.int64(1) << 5
            if si > smax:
                flags |= np.int64(1) << 6
            if ri > 0.0 and not a:
                flags |= np.int64(1) << 7

            # _final_infiltration / _intake / column_step, same expressions and order
            rain = ri * dt
            avail = hi + rain
            if hawkins == 1 and ri > 0.0:
                kf = (-lam) * np.expm1((-ri) / lam)
            else:
                kf = ksat
            deficit = (smax - si) / thick
            scale = (psi + hi) * deficit
            capillary = scale > 0.0
            x = si / (scale if capillary else 1.0)
            denom = -np.expm1(-x)
            fin = denom > 0.0
            if capillary:
                capacity = kf / (denom if fin else 1.0)
            else:
                capacity = kf
            if capillary and (not fin) and kf > 0.0:
                intake_raw = avail
            else:
                intake_raw = fmin(avail, capacity * dt)
            intake = intake_raw if a else 0.0
            wetted = si + intake
            demand = (si / smax) * ksat * drn * dt
            drainage = fmin(demand, wetted) if a else 0.0
            retained = wetted - drainage
            overflow = fmax(retained - smax, 0.0)
            soil = fmin(retained, smax)
            depth = (avail - intake) + overflow

            depth_new[i] = depth
            soil_new[i] = soil
            rain_o[i] = rain
            intake_o[i] = intake
            overflow_o[i] = overflow
            drainage_o[i] = drainage

            # outputs (recorded after the inputs)
            if not np.isfinite(depth):
                flags |= np.int64(1) << 8
            if depth < 0.0:
                flags |= np.int64(1) << 9
            if not np.isfinite(soil):
                flags |= np.int64(1) << 10
            if soil < 0.0:
                flags |= np.int64(1) << 11
            if not np.isfinite(intake):
                flags |= np.int64(1) << 12
            if intake < 0.0:
                flags |= np.int64(1) << 13
            if not np.isfinite(drainage):
                flags |= np.int64(1) << 14
            if drainage < 0.0:
                flags |= np.int64(1) << 15
            if not np.isfinite(overflow):
                flags |= np.int64(1) << 16
            if overflow < 0.0:
                flags |= np.int64(1) << 17
            residual = (depth + soil + drainage) - (hi + si + rain)
            bscale = hi + rain + si + intake + drainage + overflow
            if abs(residual) > local_rtol * bscale:
                flags |= np.int64(1) << 18

            # coupled_step: hpre, branch masks (legacy precedence), old flux
            hp = fmax(hi - fmax(intake - rain, 0.0), 0.0)
            complete = a and intake >= hi + rain
            no_runon = a and (not complete) and intake <= rain
            partial = a and (not complete) and (not no_runon)
            qp = qprev[i]
            if complete:
                qo = 0.0
                n_complete += 1
            elif no_runon:
                qo = qp
                n_no += 1
            elif partial:
                qo = (np.sqrt(hp) * hp) * k[i]
                n_partial += 1
            else:
                qo = 0.0
            hpre_o[i] = hp
            q_old_o[i] = qo

            # whole previous-discharge array (prepared-step strictness)
            if not np.isfinite(qp):
                qflags |= np.int64(1) << 0
            if qp < 0.0:
                qflags |= np.int64(1) << 1
            if (not a) and qp != 0.0:
                qflags |= np.int64(1) << 2
        counts[0] = n_no
        counts[1] = n_partial
        counts[2] = n_complete
        return flags, qflags

    @jit
    def route_kernel(order, k, k_lo, active, outlet, donor_position, donor_mask, bounds,
                     h_start, h_old, q_old, c, dt_over_dx, area, cr_max, root_tol, bal_rtol, iterations,
                     h_new, flow, q_new, qin_new, qin_old, velocity, face, scratch, maxima):
        """Phases B1-B3. Mirrors routing._route with old_discharge supplied (the coupled path)."""
        n = h_start.shape[0]
        na = order.shape[0]
        flags = np.int64(0)
        max_cr = 0.0

        # B1: cellwise input checks and the old-flux Courant number
        for i in range(n):
            hs = h_start[i]
            ho = h_old[i]
            qo = q_old[i]
            a = active[i]
            if not np.isfinite(hs):
                flags |= np.int64(1) << 0
            if hs < 0.0:
                flags |= np.int64(1) << 1
            if not np.isfinite(ho):
                flags |= np.int64(1) << 2
            if ho < 0.0:
                flags |= np.int64(1) << 3
            if not np.isfinite(qo):
                flags |= np.int64(1) << 4
            if qo < 0.0:
                flags |= np.int64(1) << 5
            if a and ho > hs:
                flags |= np.int64(1) << 6
            if (not a) and qo != 0.0:
                flags |= np.int64(1) << 7
            kk = k[i] if a else 1.0
            ratio = qo / kk
            if ratio >= 0.0:
                implied = ratio ** p23
            else:
                implied = np.nan
            if a and abs(implied - ho) > root_tol + eps64 * ho:
                flags |= np.int64(1) << 8
            if ho > 0.0:
                cr = qo / ho
            elif qo > 0.0:
                cr = np.inf
            else:
                cr = 0.0
            cr = cr * dt_over_dx
            if not a:
                cr = 0.0
            if not np.isfinite(qo):
                flags |= np.int64(1) << 9
            if cr > cr_max:
                flags |= np.int64(1) << 10
            if cr > max_cr:  # noqa: PLR1730 - explicit comparison: a NaN never replaces the running maximum
                max_cr = cr

        # B2: level-ordered coherent donor sum (0 + a + b + c + d, non-donors add 0.0) and base RHS
        q_old_lo = np.empty(na, dtype=np.float64)
        qin_old_lo = np.empty(na, dtype=np.float64)
        base_lo = np.empty(na, dtype=np.float64)
        for p in range(na):
            q_old_lo[p] = q_old[order[p]]
        for p in range(na):
            tot = q_old_lo[donor_position[0, p]] if donor_mask[0, p] else 0.0
            for sl in range(1, 4):
                tot = tot + (q_old_lo[donor_position[sl, p]] if donor_mask[sl, p] else 0.0)
            qin_old_lo[p] = tot
            base_lo[p] = h_start[order[p]] + c * (tot - q_old_lo[p])
        qin_new_lo = np.zeros(na, dtype=np.float64)
        q_new_lo = np.zeros(na, dtype=np.float64)
        flow_lo = np.zeros(na, dtype=np.float64)
        rhs_lo = np.zeros(na, dtype=np.float64)
        sweep(bounds, k_lo, donor_position, donor_mask, base_lo, c, iterations,
              qin_new_lo, q_new_lo, flow_lo, rhs_lo)

        # B3: scatter to cell order (inactive cells keep h_start, zero elsewhere) and the RHS checks
        for i in range(n):
            h_new[i] = h_start[i]
        for p in range(na):
            i = order[p]
            rhs = rhs_lo[p]
            if not np.isfinite(rhs):
                flags |= np.int64(1) << 11
            if rhs < 0.0:
                flags |= np.int64(1) << 12
            h_new[i] = rhs - q_new_lo[p] * c
            q_new[i] = q_new_lo[p]
            flow[i] = flow_lo[p]
            qin_new[i] = qin_new_lo[p]
            qin_old[i] = qin_old_lo[p]

        max_cons = 0.0
        max_bal = 0.0
        max_vel = 0.0
        for i in range(n):
            a = active[i]
            hs = h_start[i]
            hn = h_new[i]
            qn = q_new[i]
            fl = flow[i]
            qo = q_old[i]
            qio = qin_old[i]
            qi = qin_new[i]
            if a:
                vel = np.sqrt(fl) * k[i]
                fc = area * (c * (qo + qn))
            else:
                vel = 0.0
                fc = 0.0
            bal = (hn - hs) - c * ((qio + qi) - (qo + qn))
            sc = hs + hn + c * (qio + qi + qo + qn)
            if not a:
                bal = 0.0
            cons = (hn - fl) if a else 0.0
            velocity[i] = vel
            face[i] = fc
            if not np.isfinite(hn):
                flags |= np.int64(1) << 13
            if not np.isfinite(qn):
                flags |= np.int64(1) << 14
            if not np.isfinite(fl):
                flags |= np.int64(1) << 15
            if not np.isfinite(vel):
                flags |= np.int64(1) << 16
            if not np.isfinite(qi):
                flags |= np.int64(1) << 17
            if not np.isfinite(qio):
                flags |= np.int64(1) << 18
            if not np.isfinite(fc):
                flags |= np.int64(1) << 19
            if not np.isfinite(sc):
                flags |= np.int64(1) << 20
            if hn < 0.0:
                flags |= np.int64(1) << 21
            if qn < 0.0:
                flags |= np.int64(1) << 22
            if fl < 0.0:
                flags |= np.int64(1) << 23
            ac = abs(cons)
            if ac > root_tol:
                flags |= np.int64(1) << 24
            ab = abs(bal)
            if ab > bal_rtol * sc:
                flags |= np.int64(1) << 25
            # explicit comparisons (not max()): NaN never replaces a running maximum; NaN is flagged separately
            if ac > max_cons:  # noqa: PLR1730
                max_cons = ac
            if ab > max_bal:  # noqa: PLR1730
                max_bal = ab
            if vel > max_vel:  # noqa: PLR1730
                max_vel = vel
            scratch[0, i] = hn - hs
            scratch[1, i] = fc if outlet[i] else 0.0
            scratch[2, i] = sc if a else 0.0
            scratch[3, i] = qn if outlet[i] else 0.0
        maxima[0] = max_cr
        maxima[1] = max_cons
        maxima[2] = max_bal
        maxima[3] = max_vel
        return flags

    return SimpleNamespace(column=column_kernel, route=route_kernel, sweep=sweep)


def _kernels() -> SimpleNamespace:
    global _KERNELS
    if _KERNELS is None:
        if not numba_available():
            raise HydrologyNumbaUnavailableError(
                "the prepared compiled hydrology requires Numba (optional extra maple-syrup[numba]), which is "
                "not importable; there is no fallback to the NumPy/CuPy reference (use the reference "
                "implementation explicitly)")
        import numba

        _KERNELS = _build_kernels(numba)
    return _KERNELS


def reset_compiled() -> None:
    """Drop the compiled dispatchers (tests: cold-start and missing-Numba paths)."""
    global _KERNELS
    _KERNELS = None


def kernel_provenance() -> dict[str, Any]:
    """Host-side description of the compiled hydrology for reports (no compilation, no device reads)."""
    here = Path(__file__).resolve()
    sweep_file = here.with_name("routing_numba.py")
    return {
        "implementation": "numba",
        "module": "maple_syrup.hydrology_numba",
        "module_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        "kernels": ["column_kernel (phase A)", "route_kernel (phases B1-B3)"],
        "ordered_sweep": "maple_syrup.routing_numba._sweep via compiled_sweep(), called from inside route_kernel",
        "routing_numba_sha256": hashlib.sha256(sweep_file.read_bytes()).hexdigest(),
        "numba_options": {"fastmath": False, "parallel": False, "nogil": True, "boundscheck": False,
                          "error_model": "numpy", "cache": False},
        "scalar_reductions": "numpy.sum on kernel-produced operand rows (reference pairwise order)",
        "versions": numba_versions(),
        "compiled_in_process": _KERNELS is not None,
        "compilation": "lazy: the first prepared step includes JIT compilation of both kernels and the sweep",
    }


# --- context -----------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class HydrologyContext:
    """Static data of a fixed-terrain hydrology replay: contiguous, read-only, owned copies (flat, cell-major
    unless noted). Build with `prepare_hydrology`; never mutated and never aliased to a caller array."""

    shape: tuple[int, int]
    n_cells: int
    n_active: int
    dx_m: float
    model: str
    model_code: int
    active: np.ndarray  # (n_cells,) bool
    outlet: np.ndarray  # (n_cells,) bool
    conveyance: np.ndarray  # (n_cells,) float64, k = sqrt(8 g S / f)
    level_order: np.ndarray  # (n_active,) int64, flat cell index by (level, index)
    conveyance_lo: np.ndarray  # (n_active,) k in level order
    donor_position: np.ndarray  # (4, n_active) int64 into level-ordered arrays, DONOR_SLOTS order
    donor_mask: np.ndarray  # (4, n_active) bool
    level_bounds: np.ndarray  # (n_levels + 1,) int64
    column_static: np.ndarray  # (n_cells, 6) float64: ksat, suction, drainage, thickness, Smax, lambda
    graph_input_sha256: str
    preparation_s: float

    _ARRAYS = ("active", "outlet", "conveyance", "level_order", "conveyance_lo", "donor_position", "donor_mask",
               "level_bounds", "column_static")

    def nbytes(self) -> int:
        return int(sum(getattr(self, name).nbytes for name in self._ARRAYS))

    @property
    def n_levels(self) -> int:
        return int(self.level_bounds.size) - 1

    @property
    def max_level_width(self) -> int:
        return int(np.diff(self.level_bounds).max())

    def summary(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape), "n_cells": self.n_cells, "n_active": self.n_active, "dx_m": self.dx_m,
            "infiltration_model": self.model, "n_levels": self.n_levels, "max_level_width": self.max_level_width,
            "static_bytes": self.nbytes(), "preparation_s": self.preparation_s,
            "graph_input_sha256": self.graph_input_sha256,
            "ownership": "contiguous read-only copies; no alias of the caller's graph/parameter arrays",
            "layout": "flat cell-major; column_static (n_cells, 6) = ksat, suction, drainage, thickness, Smax, "
                      "lambda; donors/levels in level order",
            "host_only": True, "fastmath": False,
        }


def _own(array: Any, name: str, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
    if type(array) is not np.ndarray:
        raise HydrologyPreparationError(f"{name} must be a host numpy.ndarray (the prepared hydrology is "
                                        f"host-only and never transfers), got {type(array).__name__}")
    if array.dtype != dtype:
        raise HydrologyPreparationError(f"{name} must have dtype {np.dtype(dtype)}, got {array.dtype}")
    if tuple(array.shape) != shape:
        raise HydrologyPreparationError(f"{name} shape {tuple(array.shape)} != {shape}")
    owned = np.array(array, dtype=dtype, order="C", copy=True)
    owned.flags.writeable = False
    return owned


def _frozen(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def prepare_hydrology(graph: RoutingGraph, params: ColumnParameters) -> HydrologyContext:
    """Validate and copy the static graph and column data once. Raises `HydrologyPreparationError` (or
    `HydrologyNumbaUnavailableError`) before anything is returned. Checks, beyond the types, dtypes and
    shapes: matching graph/column active masks; finite non-negative conveyance and Ksat/suction/drainage,
    0 < theta_sat <= 1, thickness > 0, Smax == theta_sat * thickness > 0, lambda > 0 for the Hawkins model;
    the level order is a permutation of the active cells consistent with the levels and bounds; every donor
    position is in range and, where used, lies in an EARLIER level and is a genuine donor of that cell
    (receiver consistency, uniqueness, completeness); the outlets are exactly the exporting active cells.
    Together these make the unchecked kernels memory-safe and the level-ordered sweep dependency-safe."""
    t0 = time.perf_counter()
    if not numba_available():
        raise HydrologyNumbaUnavailableError(
            "the prepared compiled hydrology requires Numba (optional extra maple-syrup[numba]), which is not "
            "importable; there is no fallback (use the reference implementation explicitly)")
    if not isinstance(graph, RoutingGraph):
        raise HydrologyPreparationError("graph must be a RoutingGraph (use build_routing_graph)")
    if not isinstance(params, ColumnParameters):
        raise HydrologyPreparationError("params must be a ColumnParameters (use column_parameters)")
    if graph.xp is not np or params.xp is not np:
        raise HydrologyPreparationError(
            f"the prepared hydrology is host NumPy only; the graph is in {graph.xp.__name__!r} and the column "
            f"parameters in {params.xp.__name__!r}; no host/device transfer is made (use the reference "
            "implementation for CuPy)")
    shape = tuple(graph.shape)
    if len(shape) != 2 or min(shape) < 1 or not all(isinstance(v, (int, np.integer)) for v in shape):
        raise HydrologyPreparationError(f"graph shape must be a positive (ny, nx), got {graph.shape!r}")
    if tuple(params.shape) != shape:
        raise HydrologyPreparationError(f"column parameter shape {tuple(params.shape)} != graph shape {shape}")
    ny, nx = int(shape[0]), int(shape[1])
    n = ny * nx
    dx = graph.dx_m
    if isinstance(dx, bool) or not isinstance(dx, (int, float, np.integer, np.floating)):
        raise HydrologyPreparationError(f"graph.dx_m must be a real number, got {type(dx).__name__}")
    dx = float(dx)
    if not (math.isfinite(dx) and dx > 0.0 and math.isfinite(dx * dx) and dx * dx > 0.0):
        raise HydrologyPreparationError(f"graph.dx_m must be finite with a finite positive cell area, got {dx!r}")

    # --- graph ---
    active = _own(graph.active_flat, "graph.active_flat", np.bool_, (n,))
    n_active = int(np.count_nonzero(active))
    if n_active < 1:
        raise HydrologyPreparationError("the graph has no active cell")
    outlet = _own(graph.outlet_flat, "graph.outlet_flat", np.bool_, (n,))
    k = _own(graph.conveyance, "graph.conveyance", np.float64, (n,))
    order = _own(graph.level_order, "graph.level_order", np.int64, (n_active,))
    k_lo = _own(graph.conveyance_lo, "graph.conveyance_lo", np.float64, (n_active,))
    donor_position = _own(graph.donor_position, "graph.donor_position", np.int64, (4, n_active))
    donor_mask = _own(graph.donor_mask, "graph.donor_mask", np.bool_, (4, n_active))
    host_active = _own(graph.active, "graph.active", np.bool_, shape)
    level = _own(graph.level, "graph.level", np.int32, shape).reshape(-1)
    receiver = _own(graph.receiver, "graph.receiver", np.int64, shape).reshape(-1)
    if not np.array_equal(host_active.reshape(-1), active):
        raise HydrologyPreparationError("graph.active and graph.active_flat disagree")
    if not (np.all(np.isfinite(k)) and np.all(k >= 0.0)):
        raise HydrologyPreparationError("graph.conveyance must be finite and >= 0 everywhere")
    if order.min() < 0 or order.max() >= n or np.unique(order).size != n_active or not np.all(active[order]):
        raise HydrologyPreparationError("graph.level_order must be a permutation of the active flat cell indices")
    if not np.array_equal(k_lo, k[order]):
        raise HydrologyPreparationError("graph.conveyance_lo differs from graph.conveyance[level_order]")
    raw_bounds = graph.level_bounds
    if not isinstance(raw_bounds, tuple) or len(raw_bounds) < 2 or not all(
            isinstance(b, (int, np.integer)) and not isinstance(b, bool) for b in raw_bounds):
        raise HydrologyPreparationError("graph.level_bounds must be a tuple of at least two ints")
    bounds = np.array(raw_bounds, dtype=np.int64)
    if bounds[0] != 0 or bounds[-1] != n_active or np.any(np.diff(bounds) < 0):
        raise HydrologyPreparationError("graph.level_bounds must be non-decreasing from 0 to n_active")
    level_of_p = np.repeat(np.arange(bounds.size - 1, dtype=np.int64), np.diff(bounds))
    if not np.array_equal(level[order].astype(np.int64), level_of_p):
        raise HydrologyPreparationError("graph.level_order is not consistent with graph.level and level_bounds")
    if donor_position.min() < 0 or donor_position.max() >= n_active:
        raise HydrologyPreparationError("graph.donor_position holds an index outside [0, n_active)")
    level_start = bounds[level_of_p]
    if np.any(donor_mask & (donor_position >= level_start[None, :])):
        raise HydrologyPreparationError("a donor is not in an earlier dependency level than its receiver")
    receiving_cell = np.broadcast_to(order[None, :], donor_position.shape)[donor_mask]
    donor_cell = order[donor_position[donor_mask]]
    if np.any(receiver[donor_cell] != receiving_cell):
        raise HydrologyPreparationError("graph.donor_position names a cell that does not drain into the receiver")
    n_internal = int(np.count_nonzero(receiver[active] >= 0))
    if donor_cell.size != n_internal or np.unique(donor_cell).size != n_internal:
        raise HydrologyPreparationError("graph donors are not exactly the active cells with an internal receiver")
    if np.any(receiver[active] < EXPORT):
        raise HydrologyPreparationError("an active cell has no receiver (neither a cell nor EXPORT)")
    if not np.array_equal(outlet, active & (receiver == EXPORT)):
        raise HydrologyPreparationError("graph.outlet is not exactly the exporting active cells")

    # --- column parameters ---
    if params.model not in INFILTRATION_MODELS:
        raise HydrologyPreparationError(f"unsupported infiltration model {params.model!r}")
    ksat = _own(params.ksat_m_per_s, "params.ksat_m_per_s", np.float64, shape).reshape(-1)
    suction = _own(params.suction_m, "params.suction_m", np.float64, shape).reshape(-1)
    drain = _own(params.drainage_parameter, "params.drainage_parameter", np.float64, shape).reshape(-1)
    theta = _own(params.theta_sat, "params.theta_sat", np.float64, shape).reshape(-1)
    thick = _own(params.soil_thickness_m, "params.soil_thickness_m", np.float64, shape).reshape(-1)
    smax = _own(params.storage_max_m, "params.storage_max_m", np.float64, shape).reshape(-1)
    col_active = _own(params.active_mask, "params.active_mask", np.bool_, shape).reshape(-1)
    for name, array in (("ksat_m_per_s", ksat), ("suction_m", suction), ("drainage_parameter", drain)):
        if not (np.all(np.isfinite(array)) and np.all(array >= 0.0)):
            raise HydrologyPreparationError(f"params.{name} must be finite and >= 0 everywhere")
    if not (np.all(np.isfinite(theta)) and np.all(theta > 0.0) and np.all(theta <= 1.0)):
        raise HydrologyPreparationError("params.theta_sat must lie in (0, 1] everywhere")
    if not (np.all(np.isfinite(thick)) and np.all(thick > 0.0)):
        raise HydrologyPreparationError("params.soil_thickness_m must be finite and > 0 everywhere")
    if not (np.all(np.isfinite(smax)) and np.all(smax > 0.0) and np.array_equal(smax, theta * thick)):
        raise HydrologyPreparationError("params.storage_max_m must be finite, > 0 and equal theta_sat * soil_thickness")
    if params.model == "pavement_hawkins":
        if params.lambda_m_per_s is None:
            raise HydrologyPreparationError("model 'pavement_hawkins' needs params.lambda_m_per_s")
        lam = _own(params.lambda_m_per_s, "params.lambda_m_per_s", np.float64, shape).reshape(-1)
        if not (np.all(np.isfinite(lam)) and np.all(lam > 0.0)):
            raise HydrologyPreparationError("params.lambda_m_per_s must be finite and > 0 everywhere")
    else:
        lam = np.zeros(n, dtype=np.float64)
    if not np.array_equal(col_active, active):
        raise HydrologyPreparationError("column active_mask differs from the routing graph's active cells; unsupported")
    column_static = np.empty((n, 6), dtype=np.float64)
    for j, array in enumerate((ksat, suction, drain, thick, smax, lam)):
        column_static[:, j] = array

    return HydrologyContext(
        shape=(ny, nx), n_cells=n, n_active=n_active, dx_m=dx, model=params.model,
        model_code=_MODEL_CODES[params.model],
        active=active, outlet=outlet, conveyance=k, level_order=order, conveyance_lo=k_lo,
        donor_position=donor_position, donor_mask=donor_mask, level_bounds=_frozen(bounds),
        column_static=_frozen(column_static), graph_input_sha256=str(graph.input_sha256), preparation_s=time.perf_counter() - t0,
    )


# --- dynamic inputs ----------------------------------------------------------------------------------------
def _dynamic(array: Any, name: str, shape: tuple[int, int], error: type[Exception]) -> np.ndarray:
    """Flat C-contiguous writable-typed float64 READ view. A copy is made only if the caller's layout is not
    C-contiguous or the array is read-only: Numba types read-only and writable arrays differently, so
    normalising here keeps one compiled specialisation instead of one per flag combination. The kernels never
    write these arrays."""
    if type(array) is not np.ndarray:
        raise error(f"{name} must be a host numpy.ndarray (the prepared hydrology never transfers, converts or "
                    f"accepts subclasses/masked arrays), got {type(array).__name__}")
    if tuple(array.shape) != shape:
        raise error(f"{name} shape {tuple(array.shape)} != {shape}")
    if array.dtype != np.float64:
        raise error(f"{name} must be float64, got {array.dtype}")
    flat = np.ascontiguousarray(array).reshape(-1)
    return flat if flat.flags.writeable else flat.copy()


def _check_context(ctx: Any) -> None:
    if not isinstance(ctx, HydrologyContext):
        raise HydrologyPreparationError("ctx must be a HydrologyContext (use prepare_hydrology)")


def _column_phase(ctx: HydrologyContext, h: np.ndarray, s: np.ndarray, r: np.ndarray, qprev: np.ndarray, dt: float):
    n = ctx.n_cells
    depth_new, soil_new, rain, intake, overflow, drainage, hpre, q_old = (np.empty(n, dtype=np.float64)
                                                                          for _ in range(8))
    counts = np.zeros(3, dtype=np.int64)
    flags, qflags = _kernels().column(
        h, s, r, qprev, ctx.active, ctx.conveyance, ctx.column_static, ctx.model_code, dt, LOCAL_BALANCE_RTOL,
        depth_new, soil_new, rain, intake, overflow, drainage, hpre, q_old, counts)
    return int(flags), int(qflags), (depth_new, soil_new, rain, intake, overflow, drainage, q_old), hpre, counts


def _column_step_from(ctx: HydrologyContext, dt: float, arrays: tuple) -> ColumnStep:
    depth_new, soil_new, rain, intake, overflow, drainage, _q_old = arrays
    grid = ctx.shape
    return ColumnStep(dt, depth_new.reshape(grid), soil_new.reshape(grid), rain.reshape(grid),
                      intake.reshape(grid), overflow.reshape(grid), drainage.reshape(grid))


def prepared_column_step(ctx: HydrologyContext, depth_m: Any, soil_water_m: Any, rain_rate_m_per_s: Any,
                         dt_s: float) -> ColumnStep:
    """`infiltration.column_step(validate=True)` on the prepared columns. Same `ColumnStep`, same
    `InfiltrationError` conditions and precedence; `dt_s = 0` is the identity (inputs still validated)."""
    _check_context(ctx)
    dt = _check_dt(dt_s)
    shape = ctx.shape
    h = _dynamic(depth_m, "depth_m", shape, InfiltrationError)
    s = _dynamic(soil_water_m, "soil_water_m", shape, InfiltrationError)
    r = _dynamic(rain_rate_m_per_s, "rain_rate_m_per_s", shape, InfiltrationError)
    flags, _qflags, arrays, _hpre, _counts = _column_phase(ctx, h, s, r, np.zeros(ctx.n_cells), dt)
    if dt == 0.0:
        flags &= _COLUMN_INPUT_MASK
    if flags:
        raise InfiltrationError(_COLUMN_MESSAGES[_lowest_bit(flags)])
    if dt == 0.0:
        zero = np.zeros(shape, dtype=np.float64)
        return ColumnStep(0.0, np.array(depth_m, copy=True), np.array(soil_water_m, copy=True),
                          zero, zero.copy(), zero.copy(), zero.copy())
    return _column_step_from(ctx, dt, arrays)


# --- one coupled step --------------------------------------------------------------------------------------
def _resolve_route(flags: int, scalar_nonfinite: tuple[bool, ...], residual_failed: bool, qflags: int,
                   cr_max: float, root_tol: float, iterations: int) -> None:
    """Raise the FIRST failure in the reference's recorded order: kernel bits 0..23, the host scalar finiteness
    checks, kernel bits 24-25, the global balance, then the prepared-step previous-discharge checks."""
    def message(bit: int) -> str:
        if bit == 10:
            return f"{_COURANT_REJECTION} {cr_max}; step rejected (retry with a smaller dt)"
        if bit == 24:
            return (f"bisection did not reach root_tolerance_m = {root_tol} m after {iterations} iterations "
                    "(increase bisection_iterations); nothing is clipped")
        return _ROUTE_STATIC_MESSAGES[bit]

    def fail(text: str) -> None:
        recoverable = text.startswith((_COURANT_REJECTION, _NEGATIVE_RHS_REJECTION))
        raise (RoutingStepRejected if recoverable else RoutingError)(text)

    early = flags & ((1 << (_LAST_PRE_SCALAR_BIT + 1)) - 1)
    if early:
        fail(message(_lowest_bit(early)))
    for name, bad in zip(_SCALAR_NAMES, scalar_nonfinite, strict=True):
        if bad:
            fail(f"step produced non-finite {name} (volume overflow)")
    late = flags >> (_LAST_PRE_SCALAR_BIT + 1) << (_LAST_PRE_SCALAR_BIT + 1)
    if late:
        fail(message(_lowest_bit(late)))
    if residual_failed:
        fail("global water balance (storage change + export) violated beyond FP64 tolerance")
    if qflags:
        fail(_QPREV_MESSAGES[_lowest_bit(qflags)])


def prepared_coupled_step(ctx: HydrologyContext, rain_rate_m_per_s: Any, state: StormState, dt_s: float,
                          control: StormControl) -> CoupledStep:
    """`storm.coupled_step(graph, params, rain, state, dt, control)` with `control.implementation == "numba"`,
    on the prepared context. Same `CoupledStep` (every field, scalar types included), same error classes and
    precedence; pure: raises before returning anything and never modifies an input. `RoutingStepRejected`
    still means "retry the SAME state with a smaller dt"."""
    _check_context(ctx)
    if not isinstance(state, StormState):
        raise StormError(f"state must be a StormState, got {type(state).__name__}")
    if not isinstance(control, StormControl):
        raise StormError(f"control must be a StormControl, got {type(control).__name__}")
    dt = _check_dt(dt_s)
    shape = ctx.shape
    h = _dynamic(state.depth_m, "depth_m", shape, InfiltrationError)
    s = _dynamic(state.soil_water_m, "soil_water_m", shape, InfiltrationError)
    r = _dynamic(rain_rate_m_per_s, "rain_rate_m_per_s", shape, InfiltrationError)
    qprev = _dynamic(state.discharge_m2_s, "state.discharge_m2_s", shape, RoutingError)

    # A. column + branch (cellwise)
    flags, qflags, arrays, hpre, counts = _column_phase(ctx, h, s, r, qprev, dt)
    if dt == 0.0:
        flags &= _COLUMN_INPUT_MASK
    if flags:
        raise InfiltrationError(_COLUMN_MESSAGES[_lowest_bit(flags)])

    # scalar options (RoutingError), after the column stage exactly as in route_step (dt == 0 ends here)
    dt_r, cr_max, iterations, root_tol = _check_step_options(
        dt_s, control.courant_max, control.bisection_iterations, control.root_tolerance_m, control.implementation)
    if control.implementation != "numba":
        raise RoutingError(f"the prepared hydrology runs the compiled numba sweep only; control.implementation is "
                           f"{control.implementation!r} (select the reference hydrology for the array sweep)")

    # B. routing (cellwise checks, level-ordered sweep, cellwise outputs) and C. host reductions
    route = _route_phase(ctx, arrays[0], arrays[6], hpre, dt_r, cr_max, iterations, root_tol, qflags)
    col = _column_step_from(ctx, dt, arrays)
    new_state = StormState(state.t_s + float(dt_s), route.depth_m, col.soil_water_m, route.discharge_m2_s)
    return CoupledStep(
        dt_s=float(dt_s), state=new_state, column=col, route=route,
        n_no_runon=np.int64(counts[0]), n_partial_runon=np.int64(counts[1]), n_complete_runon=np.int64(counts[2]),
    )


def _route_phase(ctx: HydrologyContext, depth_start: np.ndarray, q_old: np.ndarray, hpre: np.ndarray, dt: float,
                 cr_max: float, iterations: int, root_tol: float, qflags: int) -> RouteStep:
    """`routing._route` for the coupled path (h_start = column depth, h_old = hpre, q_old supplied)."""
    n = ctx.n_cells
    dx = ctx.dx_m
    area = dx * dx
    c = dt / (2.0 * dx)
    dt_over_dx = dt / dx
    h_new, velocity, face = (np.empty(n, dtype=np.float64) for _ in range(3))
    flow, q_new, qin_new, qin_old = (np.zeros(n, dtype=np.float64) for _ in range(4))
    scratch = np.empty((4, n), dtype=np.float64)  # operand rows of the four reference sums
    maxima = np.zeros(4, dtype=np.float64)  # Courant old, constitutive, balance, velocity
    flags = int(_kernels().route(
        ctx.level_order, ctx.conveyance, ctx.conveyance_lo, ctx.active, ctx.outlet, ctx.donor_position,
        ctx.donor_mask, ctx.level_bounds, depth_start, hpre, q_old, c, dt_over_dx, area, cr_max, root_tol,
        BALANCE_RTOL, iterations, h_new, flow, q_new, qin_new, qin_old, velocity, face, scratch, maxima))
    with np.errstate(all="ignore"):
        storage_change = area * scratch[0].sum()
        export = scratch[1].sum()
        residual = storage_change + export
        global_tol = BALANCE_RTOL * (ctx.n_active + 2) * area * scratch[2].sum()
        outlet_discharge = dx * scratch[3].sum()
        residual_failed = bool(abs(residual) > global_tol)
        max_courant_new = maxima[3] * dt_over_dx
    scalar_nonfinite = tuple(not math.isfinite(float(v)) for v in
                             (storage_change, export, residual, global_tol, outlet_discharge))
    _resolve_route(flags, scalar_nonfinite, residual_failed, qflags, cr_max, root_tol, iterations)

    grid = ctx.shape
    return RouteStep(
        dt_s=dt,
        depth_m=h_new.reshape(grid),
        flow_depth_m=flow.reshape(grid),
        discharge_m2_s=q_new.reshape(grid),
        velocity_m_s=velocity.reshape(grid),
        inflow_m2_s=qin_new.reshape(grid),
        old_discharge_m2_s=q_old.reshape(grid),
        old_inflow_m2_s=qin_old.reshape(grid),
        face_volume_m3=face.reshape(grid),
        export_m3=export,
        outlet_discharge_m3_s=outlet_discharge,
        storage_change_m3=storage_change,
        budget_residual_m3=residual,
        stale_inflow_gain_m3=None,
        max_courant_old=maxima[0],
        max_courant_new=max_courant_new,
        max_constitutive_residual_m=maxima[1],
        max_cell_balance_residual_m=maxima[2],
        conservative=True,
        bisection_iterations=iterations,
        implementation="numba",
    )
