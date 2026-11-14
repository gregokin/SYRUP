"""Opt-in materialized current Native bundle, never a substitute runtime."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TESTS / "phase15_recharge"))


@pytest.fixture
def current_bundle():
    target = os.environ.get("SYRUP_NCP_TARGET")
    if not target:
        pytest.skip("SYRUP_NCP_TARGET is not configured; current Native source admission is unissued")
    path = Path(target).resolve()
    if not path.is_dir():
        pytest.fail(f"configured current Native bundle is absent: {path}")
    return path
