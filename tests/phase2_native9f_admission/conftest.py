from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parents[1]
for _sub in ("pond_event", "pond_windowed", "shared_storage_compat"):
    sys.path.insert(1, str(_TESTS / _sub))


@pytest.fixture
def windowed_bed(tmp_path):
    """3x4 resident-window bed, six classes, heavy mobile load on two cells (real pending journal on deposition)."""
    from pond_helpers import plane
    from windowed_helpers import dense_deep_bed, heavy_mobile, windowed_counterpart

    dense, ctx = dense_deep_bed(plane(), depth=0.003, mobile=heavy_mobile())
    return windowed_counterpart(dense, ctx, tmp_path / "store")
