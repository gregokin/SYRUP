"""Fixtures of tests/shared_storage_compat: the SYRUP adapter contract against the MERGED MAPLE shared windowed storage.

Every fixture is a few cells, built by the genuine native constructors through the existing windowed builders of
`tests/pond_windowed` (real file-backed deep store, real prefix packets, real committed baseline). No terrain, forcing,
checkpoint arrays or private data are read; all data are generated here. Nothing in this directory is skipped because the
bound MAPLE is the merged one; a missing capability is a failure. Authored with file tools only: not executed by its author.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parents[1]
_SRC = _TESTS.parent / "src"
sys.path.insert(0, str(_SRC))
for _sub in ("pond_event", "pond_windowed"):
    sys.path.insert(1, str(_TESTS / _sub))
sys.path.insert(1, str(Path(__file__).resolve().parent))

import maple_syrup

_PACKAGE_DIR = Path(maple_syrup.__file__).resolve().parent
if _PACKAGE_DIR != (_SRC / "maple_syrup").resolve():
    raise RuntimeError(f"tests/shared_storage_compat must exercise {_SRC}, but maple_syrup came from {_PACKAGE_DIR}")


def pytest_report_header(config):
    try:
        import maple
    except Exception as exc:  # noqa: BLE001 - reported verbatim; the tests then fail on their own
        return [f"shared_storage_compat: MAPLE import failed ({type(exc).__name__}: {exc})"]
    return [f"shared_storage_compat: maple_syrup={_PACKAGE_DIR}", f"shared_storage_compat: maple={Path(maple.__file__).resolve().parent}"]


@pytest.fixture
def windowed_bed(tmp_path):
    """`(bed, ctx)`: a 3x4 resident-window bed (window 3, 300 x 1 mm voxels, six classes, compact water-only ledger) with a
    heavy mobile load on two cells, so that the native deposition overflows the active layer and the window into a real
    pending journal. Fresh store per test."""
    from pond_helpers import plane
    from windowed_helpers import dense_deep_bed, heavy_mobile, windowed_counterpart

    dense, ctx = dense_deep_bed(plane(), depth=0.003, mobile=heavy_mobile())
    return windowed_counterpart(dense, ctx, tmp_path / "store")


@pytest.fixture
def light_windowed_bed(tmp_path):
    """Same bed without a mobile load (a pure pickup does not spill)."""
    from pond_helpers import plane
    from windowed_helpers import dense_deep_bed, windowed_counterpart

    dense, ctx = dense_deep_bed(plane(), depth=0.003)
    return windowed_counterpart(dense, ctx, tmp_path / "store")
