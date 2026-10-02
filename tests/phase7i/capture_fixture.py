"""Synthetic capture files in the exact text format the Fortran hook writes (3-digit exponents, one array row per
line), built from SYRUP-order interior arrays so the tests know the answer of every orientation/unit conversion."""

from __future__ import annotations

import capture_data as cd
import numpy as np


def fortran_float(value: float) -> str:
    mantissa, exponent = f"{value:.16E}".split("E")
    return f"{mantissa}E{exponent[0]}{exponent[1:].zfill(3)}"


def render_capture(kind: str, scalars: dict, arrays: dict) -> str:
    lines = [f"{cd.MAGIC} {kind}"]
    for name, value in scalars.items():
        lines.append(f"scalar {name} {value if isinstance(value, int) else fortran_float(value)}")
    for name, array in arrays.items():
        is_float = array.dtype.kind == "f"
        lines.append(f"array {name} {'d' if is_float else 'i'} {array.shape[0]} {array.shape[1]}")
        for row in array:
            lines.append(" ".join(fortran_float(float(v)) if is_float else str(int(v)) for v in row))
    lines.append(f"{cd.COMPLETE} {kind}")
    return "\n".join(lines) + "\n"


def render_steps(rval_mm_s, *, q_single=None, q_double=None, dt=1.0) -> str:
    n = len(rval_mm_s)
    lines = [f"{cd.MAGIC} steps", "columns " + " ".join(cd.STEP_COLUMNS)]
    for k in range(n):
        row = [float(k + 1), dt * (k + 1), rval_mm_s[k], 4.0 * rval_mm_s[k], rval_mm_s[k],
               0.0 if q_single is None else q_single[k], 0.0 if q_double is None else q_double[k], 0.0,
               0.1 * (k + 1), 75.0 * 6, 0.01 * (k + 1), 0.0, 0.1, 1.0]
        lines.append(f"{k + 1:8d} " + " ".join(fortran_float(v) for v in row[1:]))
    lines.append(f"{cd.COMPLETE} steps")
    return "\n".join(lines) + "\n"


def full(interior_south_first: np.ndarray, ring, nr: int, nc: int) -> np.ndarray:
    """SYRUP interior -> Fortran (nr + 1, nc + 1) with the ring filled and rows north-first."""
    a = np.full((nr + 1, nc + 1), ring, dtype=interior_south_first.dtype)
    a[1:nr, 1:nc] = interior_south_first[::-1]
    return a


def synthetic_setup(nr: int = 4, nc: int = 3, nit: int = 5) -> dict:
    """Distinct values per row/column so that a missing flip, a transposition or a wrong unit is visible.
    Returns the parsed `static` capture, its text and the SYRUP-side `ref` dict of check_static_consistency."""
    ny, nx = nr - 1, nc - 1
    rows = np.arange(ny, dtype=np.float64)[:, None] * np.ones((1, nx))
    cols = np.ones((ny, 1)) * np.arange(nx, dtype=np.float64)[None, :]
    ksat_mm = 1.0e-4 * (1.0 + rows + 0.1 * cols)
    slope = 0.05 + 0.01 * rows + 0.001 * cols
    theta_sat = 0.40 + 0.01 * cols
    pave_fraction = 0.2 + 0.1 * rows
    scale = np.ones((ny, nx))
    scale[1, 1] = 0.5
    aspect = np.full((ny, nx), 3, dtype=np.int64)
    outlet = np.zeros((ny, nx), dtype=bool)
    outlet[0, :] = True
    soil0_m = np.full((ny, nx), 0.25 * 0.3)
    ref = {
        "aspect": aspect.astype(np.int8), "slope": slope, "friction": np.full((ny, nx), 21.45),
        "active": np.ones((ny, nx), dtype=bool), "outlet": outlet, "rainfall_scale": scale,
        "theta_sat": theta_sat, "suction_m": np.full((ny, nx), 0.0466), "drainage_parameter": np.full((ny, nx), 0.05),
        "soil_thickness_m": np.full((ny, nx), 0.3), "pavement_fraction": pave_fraction,
        "initial_theta": np.full((ny, nx), 0.25), "initial_soil_m": soil0_m, "dx_m": 0.5,
    }
    rmask = full(scale, -9999.0, nr, nc)
    # Fortran order: north (small i) first; i, j are 1-based Fortran indices
    order = np.array([[i, j, 1] for i in range(2, nr + 1) for j in range(2, nc + 1)], dtype=np.int64)
    zeros = np.zeros((nr + 1, nc + 1))
    arrays = {
        "ksat": full(ksat_mm, 0.0, nr, nc), "psi": np.full((nr + 1, nc + 1), 46.6),
        "theta_sat": full(theta_sat, 0.0, nr, nc), "theta": full(np.full((ny, nx), 0.25), 0.0, nr, nc),
        "cum_inf": full(np.full((ny, nx), 75.0), 0.0, nr, nc), "cum_drain_initial_mm": zeros.copy(),
        "stmax": full(theta_sat * 300.0, 0.0, nr, nc), "drain_par": np.full((nr + 1, nc + 1), 0.05),
        "pave": full(pave_fraction * 1.0e-2, 0.0, nr, nc), "slope": full(slope, 0.0, nr, nc),
        "ff": np.full((nr + 1, nc + 1), 21.45), "rmask": rmask, "aspect": full(aspect, 0, nr, nc), "order": order,
        "d_initial_mm": zeros.copy(), "q_initial_mm2_s": zeros.copy(),
    }
    scalars = {"nr": nr, "nc": nc, "nr1": nr, "nc1": nc, "nr2": nr + 1, "nc2": nc + 1, "ncell1": int(order.shape[0]),
               "nit": nit, "ndirn": 4, "iroute": 5, "ff_type": 1, "inf_type": 2, "inf_model": 2, "rain_type": 2,
               "dt_s": 1.0, "dx_mm": 500.0, "dy_mm": 500.0, "ksat_mod": 1.0, "psi_mod": 1.0,
               "rval_initial_mm_s": 0.01, "stormlength_s": 5.0}
    text = render_capture("static", scalars, arrays)
    return {"static": cd.parse_capture_text(text, "static"), "text": text, "ref": ref, "scalars": scalars,
            "arrays": arrays, "ksat_mm": ksat_mm, "nr": nr, "nc": nc}
