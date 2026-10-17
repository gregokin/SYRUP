"""Shared builders and INDEPENDENT references for the implicit-friction tests. Nothing here was run by its author (file-only
tools); Codex records results.

References: `exact_root` / `exact_residual` evaluate the friction equation `q + alpha |q| q = q*` in 80-digit `decimal`
arithmetic from the exact FP64 inputs (never through the library helper), `original_face_flux` is the VERBATIM pre-change
lagged face function.
"""
from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np

from maple_syrup.experimental_hydrology import local_inertial_face_flux
from maple_syrup.infiltration import column_parameters
from maple_syrup.pond_network import OutletFace, build_pond_topology
from maple_syrup.routing import GRAVITY_M_S2

getcontext().prec = 80
EPS = float(np.finfo(np.float64).eps)
# declared FP64 roundoff of the closed-form root (about 8 operations, each <= 0.5 ulp; sensitivity of the root to each
# relative perturbation is <= 1): a generous 32 eps relative; the residual of the whole equation is bounded by 64 eps |q*|.
ROOT_REL = 32.0 * EPS
RESIDUAL_REL = 64.0 * EPS


def exact_root(q_star: float, h: float, fric: float, dt: float) -> Decimal:
    qs, hh, ff, tt = Decimal(q_star), Decimal(h), Decimal(fric), Decimal(dt)
    alpha = (tt * ff) / (Decimal(8) * hh * hh)
    return Decimal(2) * qs / (Decimal(1) + (Decimal(1) + Decimal(4) * alpha * abs(qs)).sqrt())


def exact_residual(q: float, q_star: float, h: float, fric: float, dt: float) -> Decimal:
    """`q + alpha |q| q - q*` evaluated exactly (in 80 digits) for the FP64 `q` actually returned."""
    qq, qs, hh, ff, tt = Decimal(q), Decimal(q_star), Decimal(h), Decimal(fric), Decimal(dt)
    alpha = (tt * ff) / (Decimal(8) * hh * hh)
    return qq + alpha * abs(qq) * qq - qs


def rel_err(value: float, reference: Decimal) -> float:
    if reference == 0:
        return abs(float(value))
    return float(abs(Decimal(value) - reference) / abs(reference))


def face(q_old, z_a, z_b, h_a, h_b, fric, dt, treatment=None, dx=1.0):
    """One interior face through the real `local_inertial_face_flux`; returns `(q, h_face)` as floats."""
    arr = lambda v: np.array([v], dtype=np.float64)  # noqa: E731
    args = (np.array([1], dtype=np.int8), arr(max(z_a, z_b)), arr(fric), arr(0.0), np.array([0], dtype=np.int8), arr(q_old),
            arr(h_a), arr(h_b), arr(z_a + h_a), arr(z_b + h_b), dt, dx)
    q, hf = local_inertial_face_flux(*args) if treatment is None else local_inertial_face_flux(*args, treatment)
    return float(q[0]), float(hf[0])


def original_face_flux(ftype, zmax, fric, kb, sign, q_old, h_a, h_b, eta_a, eta_b, dt, dx):
    """VERBATIM copy of `local_inertial_face_flux` before the friction option existed (the old-default oracle)."""
    with np.errstate(all="ignore"):
        top = np.where(eta_a > eta_b, eta_a, eta_b)
        hf1 = np.maximum(top - zmax, 0.0)
        grad = (eta_b - eta_a) / dx
        numerator = q_old - (((GRAVITY_M_S2 * hf1) * dt) * grad)
        friction = np.where(q_old != 0.0, ((dt * (fric / 8.0)) * np.abs(q_old)) / (hf1 * hf1), 0.0)
        q1 = np.where(hf1 > 0.0, numerator / (1.0 + friction), 0.0)
        donor = np.where(sign > 0, h_a, h_b)
        q_out = (np.sqrt(donor) * donor) * kb
        q2 = np.where(sign > 0, q_out, -q_out)
        q = np.where(ftype == 1, q1, np.where(ftype == 2, q2, 0.0))
        hf = np.where(ftype == 1, hf1, np.where(ftype == 2, donor, 0.0))
    return q, hf


def topo(interior, outlets=(), *, drop=1.0, friction=1.0, dx=0.5):
    """Pond topology of an interior bed (each outlet `(row, col, d_row, d_col)` gets a ring cell `drop` m below its cell)."""
    z = np.asarray(interior, dtype=np.float64)
    ny, nx = z.shape
    full = np.full((ny + 2, nx + 2), float(z.max()) + 10.0)
    full[1:-1, 1:-1] = z
    faces = [OutletFace(*o) for o in outlets]
    for o in faces:
        full[o.row + o.d_row + 1, o.col + o.d_col + 1] = z[o.row, o.col] - drop
    return build_pond_topology(full, np.ones((ny, nx), dtype=bool), np.full((ny, nx), float(friction)), dx, faces,
                               allow_closed_components=not faces)


def col_params(active, xp=np, *, ksat=0.0):
    shape = active.shape

    def full(v):
        return xp.asarray(np.full(shape, float(v)))

    return column_parameters(model="fixed_ksat", ksat_m_per_s=full(ksat), suction_m=full(0.01), drainage_parameter=full(0.0),
                             theta_sat=full(0.4), soil_thickness_m=full(0.3), active_mask=xp.asarray(active))
