"""Shared Phase 2 test helpers.

`read_legacy_grid` is a deliberately independent, test-only parse of a
legacy ESRI ASCII file (six header records, then `nrows` rows, as the
Fortran reader consumes them). The code under test reads through MAPLE's
importer instead, so agreement cross-checks MAPLE's orientation and crop.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
RECIPE_PATH = REPO / "cases" / "plot1" / "recipe.yaml"
MAHLERAN_ROOT = Path(os.environ.get("MAPLE_SYRUP_MAHLERAN_ROOT", "/home/okin/MAHLERAN"))
PLOT1_DIR = MAHLERAN_ROOT / "Input" / "input_p1"


def read_legacy_grid(path: Path) -> np.ndarray:
    """North-first (file order) float grid; ignores bytes after the rows."""
    lines = Path(path).read_bytes().splitlines()
    header = {}
    for line in lines[:6]:
        key, value = line.decode("ascii").split()
        header[key.lower()] = value
    nrows, ncols = int(header["nrows"]), int(header["ncols"])
    rows = [[float(t) for t in line.decode("ascii").split()] for line in lines[6:6 + nrows]]
    grid = np.array(rows, dtype=np.float64)
    assert grid.shape == (nrows, ncols)
    return grid


def interior_maple(grid_north_first: np.ndarray) -> np.ndarray:
    """Legacy interior (rows 2..61, cols 2..21) in MAPLE order (row 0 = south)."""
    return np.ascontiguousarray(grid_north_first[1:-1, 1:-1][::-1])


def independent_fractions(plot1_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(raw, roundoff-normalized, pavement-rescaled) interior fractions in
    MAPLE order, from the six distinct maps and a per-cell transcription of
    MAHLERAN_storm_setting_xml.f90 352-369."""
    raw = np.stack(
        [interior_maple(read_legacy_grid(plot1_dir / f"plot1_phi{k}.asc")) for k in range(1, 7)],
        axis=-1,
    )
    normalized = raw / raw.sum(axis=-1, keepdims=True)
    pave = interior_maple(read_legacy_grid(plot1_dir / "p1pavcoverveg.asc"))
    final = normalized.copy()
    for (r, c), percent in np.ndenumerate(pave):
        ps = normalized[r, c]
        grav = ps[4] + ps[5]
        p = percent / 100.0
        if p <= 0.0 or grav == 0.0:
            continue
        final[r, c, :4] = ps[:4] * ((1.0 - p) / (1.0 - grav))
        final[r, c, 4:] = ps[4:] * (p / grav)
    return raw, normalized, final


def write_legacy_grid(path: Path, grid: np.ndarray, *, cellsize: str = "0.5") -> None:
    nrows, ncols = grid.shape
    header = (
        f"ncols         {ncols}\nnrows         {nrows}\nxllcorner     0.00000000\n"
        f"yllcorner     0.00000000\ncellsize      {cellsize}\nnodata_value  -9999\n"
    )
    body = "".join(" " + " ".join(repr(float(v)) for v in row) + "\n" for row in grid)
    path.write_text(header + body, encoding="ascii")


@pytest.fixture(scope="session")
def mahleran_root() -> Path:
    if not (MAHLERAN_ROOT / "mahleran_input.xml").is_file() or not PLOT1_DIR.is_dir():
        pytest.skip(f"MAHLERAN reference not available at {MAHLERAN_ROOT}")
    return MAHLERAN_ROOT


@pytest.fixture
def fake_mahleran(tmp_path: Path, mahleran_root: Path) -> Path:
    """A disposable copy of the root XML and the top-level Plot 1 inputs,
    for tests that corrupt a file. The reference tree is only read."""
    root = tmp_path / "mahleran_copy"
    target = root / "Input" / "input_p1"
    target.mkdir(parents=True)
    shutil.copy2(mahleran_root / "mahleran_input.xml", root / "mahleran_input.xml")
    for source in PLOT1_DIR.iterdir():
        if source.is_file() and source.suffix in (".asc", ".dat"):
            shutil.copy2(source, target / source.name)
    return root
