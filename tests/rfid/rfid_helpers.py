"""Synthetic builders and markers shared by the RFID tests (not a test module). Imports of the Phase 4 helpers are lazy so this
module imports without MAPLE. Nothing here was run by its author (file-only tools)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

DZ = 0.015625
DX = 0.5
NODATA = -9999.0
MAHLERAN_ROOT = Path("/home/okin/MAHLERAN")
RFID_INPUT = MAHLERAN_ROOT / "Input" / "RFID_2014"

needs_rfid = pytest.mark.skipif(not (RFID_INPUT / "mahleran_input.xml").is_file(),
                                reason="MAHLERAN RFID_2014 input not available")


def _configured_gfortran_works() -> bool:
    """The compiler the harness will really use (`fortran_reference.locate_toolchain`: MAPLE_SYRUP_GFORTRAN or PATH) answers
    `--version`; a PATH-only `shutil.which` would wrongly skip an env-configured toolchain."""
    try:
        import fortran_reference as fr
    except ImportError:
        return False
    toolchain = fr.locate_toolchain()
    if toolchain is None:
        return False
    try:
        return subprocess.run([toolchain.compiler, "--version"], capture_output=True, timeout=60, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


needs_gfortran = pytest.mark.skipif(not _configured_gfortran_works(),
                                    reason="no working configured gfortran (MAPLE_SYRUP_GFORTRAN or PATH); no Fortran claim")


def chain_full(n: int, **kwargs) -> np.ndarray:
    from test_routing import chain_full as _chain

    return _chain(n, **kwargs)


def south_export(z: np.ndarray) -> np.ndarray:
    from test_routing import south_export as _export

    return _export(z)


def pit_chain(n: int = 6, pit_row: int = 3) -> np.ndarray:
    """Chain draining south (`chain_full`); the cell at full-grid row `pit_row` is a strict pit: the cell south of it is raised
    ABOVE it, so none of its D4 neighbours is lower. The cells north of it drain into it."""
    z = chain_full(n)
    z[pit_row - 1, 1] = z[pit_row, 1] + DZ
    return z


def build(z, **kwargs):
    """Strict-by-default graph on `z` (full grid with ring), friction 1, south ring as the export set."""
    from maple_syrup.routing import build_routing_graph

    ny, nx = z.shape[0] - 2, z.shape[1] - 2
    return build_routing_graph(z, south_export(z), np.full((ny, nx), 1.0), DX, **kwargs)


def synthetic_capture(n_on: int = 2641, n: int = 2700, rate: str = "0.03836299851536751") -> str:
    """A `syrup_hydro_steps.txt` look-alike with a constant rate for `n_on` steps, then zero."""
    lines = ["SYRUP_HYDRO_CAPTURE_V1 steps",
             "columns iter t_s rval_applied_mm_s sum_r2_active_mm_s max_r2_active_mm_s q_plot_single_mm2_s"]
    lines += [f"{i} {float(i)!r} {rate if i <= n_on else '0.0'} 0 0 0" for i in range(1, n + 1)]
    lines.append("SYRUP_HYDRO_CAPTURE_COMPLETE steps")
    return "\n".join(lines) + "\n"
