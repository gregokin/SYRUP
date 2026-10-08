"""Synthetic-terrain helpers of the Chastre tests (importable; conftest.py holds only fixtures)."""
import hashlib
from pathlib import Path

import numpy as np

NODATA = -9999.0
NY, NX = 10, 8


def synthetic_interior() -> np.ndarray:
    """South-first (NY, NX) elevations: a tilted plane draining toward row/col 0 with a 2 x 2 flat plateau at the low corner
    (4 FLAT sinks), one strict pit at (7, 6) and two interior nodata cells. No cell can reach the (nodata) ring: ZERO outlets.
    The tilt is DYADIC (0.125 r + 0.0625 c, exact in binary and in the 6-decimal ESRI text), so the arrays built here and the
    values read back from the written file are bit-identical (`write_dtm` verifies it)."""
    r, c = np.indices((NY, NX), dtype=np.float64)
    z = 10.0 + 0.125 * r + 0.0625 * c
    z[0:2, 0:2] = 10.0
    z[7, 6] = 10.5
    z[5, 4] = NODATA
    z[5, 5] = NODATA
    return z


def write_dtm(path: Path, interior: np.ndarray) -> None:
    """ESRI ASCII (north-first file) of `interior` (south-first) inside a one-cell nodata ring."""
    ny, nx = interior.shape
    full = np.full((ny + 2, nx + 2), NODATA)
    full[1:-1, 1:-1] = interior
    lines = [f"ncols {nx + 2}", f"nrows {ny + 2}", "xllcorner 0", "yllcorner 0", "cellsize 1", "NODATA_value -9999"]
    lines += [" ".join(f"{v:.6f}" if v != NODATA else "-9999" for v in row) for row in full[::-1]]
    path.write_text("\n".join(lines) + "\n")
    if not np.array_equal(read_dtm_interior(path), interior):
        raise AssertionError(f"{path}: the ESRI text does not round-trip the synthetic elevations exactly")


def read_dtm_interior(path: Path) -> np.ndarray:
    """South-first interior values exactly as the case audit reads them (`read_legacy_grid`, north-first file, ring removed)."""
    from maple_syrup.rfid_case import read_legacy_grid

    _header, body = read_legacy_grid(path)
    return np.ascontiguousarray(body[::-1][1:-1, 1:-1])


def recipe_dict(dtm: Path, source_dir: Path, *, n_active, n_sinks, nz, rows=4, sha256=None):
    return {
        "schema": "maple_syrup.chastre_recipe.v1", "case_name": "chastre_synthetic",
        "source_rfid": {"case_dir": str(source_dir)},
        "terrain": {"path": str(dtm), "sha256": sha256 or hashlib.sha256(dtm.read_bytes()).hexdigest(), "nrows": NY + 2,
                    "ncols": NX + 2, "cellsize_m": 1.0, "nodata_value": -9999, "expected_n_active": n_active,
                    "expected_n_sinks": n_sinks, "expected_n_outlets": 0, "expected_nz": nz},
        "tiles": {"rows": rows, "max_rss_gib": 64},
        "rainfall_scaling": {"resize": "nearest_pixel_centre", "gap_fill": "nearest_valid_euclid_first_row_major"},
    }
