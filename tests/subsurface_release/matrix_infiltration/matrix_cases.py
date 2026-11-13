"""Shared fixtures of the matrix infiltration tests.

Default column: top layer 0.5 m thick with theta = 0.125 (w0 = 0.0625 m), theta_fc = 0.25 (fc_m = 0.125 m),
theta_sat = 0.9 (room 0.3875 m: ample), psi = 0.08 m -> b = psi x (0.125 - 0.0625)/0.5 = 0.08 x 0.125 = 0.01 m
EXACTLY equal to the double nearest 0.01 (dyadic deficit, scaling by a power of two commutes with rounding). The
hand/Decimal oracles use b = 0.01, K = 1e-5 m/s, g = 0.5 (the audit's values). Area 1 m2 so depths are volumes.

Independent controls of the implicit equation x - b log1p(x/(J + b)) = K tau: `decimal_increment` (60-digit
bisection on the exact elapsed-time expression), `float_bisection` (pure float64 bracketing), SciPy `brentq`
(in the tests), and the root-owned 90-digit `root_decimal_oracles.json`."""
from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from maple_syrup import matrix_infiltration as mi
from maple_syrup import subsurface as ss

DZ0 = 0.5
THETA0 = 0.125
FC = 0.25
SAT = 0.9
PSI = 0.01 / (FC - THETA0)  # 0.08: b = PSI x (FC - THETA0) = 0.01
K = 1e-5
G = 0.5
B = PSI * ((FC * DZ0 - THETA0 * DZ0) / DZ0)  # the very FP64 b the kernels form
AREA = 1.0
BIG = 1 << 40  # a declared UNIT-TEST budget for tiny arrays: NOT a free-memory reading, admission or scaling/RSS evidence
ROOT_ORACLES = Path(__file__).resolve().parent / "root_decimal_oracles.json"  # PORTABLE DELTA: fixture shipped beside this helper (was a task-directory path)


def layers(*, fc=FC, sat=SAT, dz0=DZ0, deeper=(1.0,), res=0.01, stress=0.05):
    edges = [0.0, dz0]
    for t in deeper:
        edges.append(edges[-1] + t)
    return ss.soil_layers(edges, theta_residual=res, theta_field_capacity=fc, theta_saturation=sat,
                          diffusivity_m2_s=0.0, theta_evaporation_stress=stress, evaporation_factor=1.0,
                          gravity_drainage=True)


def column(lay=None, n_cells=4, active=None, backend="numpy", area=AREA, reserve=0, available=BIG, xp=np):
    lay = layers() if lay is None else lay
    mask = np.ones(n_cells, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if xp is not np:
        mask = xp.asarray(mask)
    return ss.prepare_column(lay, mask, area, backend=backend, reserve_bytes=reserve, available_bytes_fn=lambda: available)


def operator(col, *, k=K, psi=PSI, g=G, extra_reserve=0):
    xp = col.xp
    conv = (lambda v: xp.asarray(v) if isinstance(v, np.ndarray) and xp is not np else v)
    return mi.prepare_matrix(col, conductivity_m_s=conv(k), suction_m=conv(psi), vegetation_factor=conv(g),
                             extra_reserve_bytes=extra_reserve)


def state_of(col, theta0=THETA0, deeper_theta=0.1, t_s=0.0, cells=None):
    """Uniform state (theta0 in the top layer) or explicit per-cell layer lists."""
    if cells is None:
        thick = np.array(col.layers.thickness_m)
        theta = np.array([theta0] + [deeper_theta] * (col.n_layers - 1))
        water = np.ascontiguousarray(np.repeat((theta * thick)[:, None], col.n_cells, axis=1))
    else:
        water = np.ascontiguousarray(np.array(cells, dtype=np.float64).T)
    if col.xp is not np:
        water = col.xp.asarray(water)
    return ss.column_state(col, water, t_s)


def history_of(op, wetting=0.0, cumulative=0.0):
    xp = op.xp
    conv = (lambda v: xp.asarray(v) if isinstance(v, np.ndarray) and xp is not np else v)
    return mi.wetting_history(op, wetting_m=conv(wetting), cumulative_intake_m=conv(cumulative))


def full(col, value):
    return col.xp.full(col.n_cells, float(value), dtype=np.float64)


def host(array) -> np.ndarray:
    return np.asarray(array.get()) if hasattr(array, "get") else np.asarray(array)


def arrays_of(result):
    return {"pool": result.surface_pool_m, "water": result.state.water_m, "wetting": result.history.wetting_m,
            "cumulative": result.history.cumulative_intake_m, "intake": result.intake_m}


def deficit_b(col, state, psi=PSI):
    """The b the kernels form from the CURRENT accepted state: psi x max(fc_m0 - w0, 0) / dz0 (host, (N,))."""
    w0 = host(state.water_m)[0]
    deficit = np.maximum(float(col.layers.field_capacity_m[0]) - w0, 0.0)
    return psi * (deficit / float(col.layers.thickness_m[0]))


def load_root_oracles() -> dict:
    return json.loads(ROOT_ORACLES.read_text())


def decimal_elapsed(j: Decimal, delta: Decimal, b: Decimal, k: Decimal) -> Decimal:
    if delta == 0:
        return Decimal(0)
    if k == 0:
        return Decimal("Infinity")
    return (delta if b == 0 else delta - b * ((j + b + delta) / (j + b)).ln()) / k


def decimal_increment(j, b, k, dt, prec: int = 60, iterations: int = 300) -> Decimal:
    """Independent high-precision bisection of x - b ln((J + b + x)/(J + b)) = K dt (not the root's file)."""
    with localcontext() as ctx:
        ctx.prec = prec
        j, b, k, dt = (Decimal(repr(float(v))) for v in (j, b, k, dt))
        if k == 0 or dt == 0:
            return Decimal(0)
        if b == 0:
            return k * dt
        lo, hi = Decimal(0), max(k * dt, b, j, Decimal("1e-100"))
        while decimal_elapsed(j, hi, b, k) < dt:
            hi *= 2
        for _ in range(iterations):
            mid = (lo + hi) / 2
            if decimal_elapsed(j, mid, b, k) < dt:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


def float_bisection(j: float, b: float, k: float, dt: float, iterations: int = 200) -> float:
    """Pure float64 bracketing control of the same equation (no Newton, no shared code)."""
    r = k * dt
    if r == 0.0 or b == 0.0:
        return r
    lo, hi = r, r + math.sqrt(r * r + 2.0 * r * b)

    def f(x: float) -> float:
        return (x - r) - b * math.log1p(x / (j + b))

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
