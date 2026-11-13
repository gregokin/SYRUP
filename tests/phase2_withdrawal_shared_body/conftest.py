"""S2-B1.a common withdrawal reader refactor (author-written, CPU only). The reference is the pre-B1.a `subsurface_transpiration.py` (task evidence `s2_b1_a_pre_sources/`, byte-identical to the frozen
A4.2 file), loaded as a separate module."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TASK = Path(__file__).resolve().parents[3]
OLD_PATH = TASK / "s2_b1_a_pre_sources" / "subsurface_transpiration.py"
NEW_PATH = Path(__file__).resolve().parents[2] / "src" / "maple_syrup" / "subsurface_transpiration.py"


@pytest.fixture(scope="session")
def old():
    if not OLD_PATH.is_file():
        pytest.skip("the pre-B1.a source copy is task evidence and is not present in this tree")
    spec = importlib.util.spec_from_file_location("old_subsurface_transpiration_pre_b1a", OLD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def new():
    from maple_syrup import subsurface_transpiration

    return subsurface_transpiration
