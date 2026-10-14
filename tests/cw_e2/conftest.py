"""CW-E2 fixtures: a small genuine-shaped MAPLE project and its dist-info, built in tmp_path."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: Frozen, immutable actual-MAPLE M24 copy (read-only; never written by these tests).
FROZEN_M24 = Path("/home/okin/SYRUP/outputs/dependencies/maple_cw_b24_metadata_frozen")

PYPROJECT = """[project]
name = "maple"
version = "0.0.1"
description = "MAPLE fixture"
requires-python = ">=3.11"
dependencies = ["numpy", "xarray>=2024.10"]

[project.optional-dependencies]
dev = ["pytest"]
"""

METADATA = """Metadata-Version: 2.4
Name: maple
Version: 0.0.1
Summary: MAPLE fixture
Requires-Python: >=3.11
Requires-Dist: numpy
Requires-Dist: xarray>=2024.10
Provides-Extra: dev
Requires-Dist: pytest; extra == "dev"
"""


def write_origin(path: Path, **override) -> Path:
    record = {
        "schema": "maple_syrup.dependency_origin.v1",
        "description": "test fixture project",
        "original_source": {"kind": "synthetic fixture", "revision": None,
                            "revision_absence_reason": "fixture has no version-control history",
                            "dirty_snapshot_note": None},
        "patch_chain": [],
        "patch_chain_absence_reason": "no patches applied",
    }
    record.update(override)
    path.write_text(json.dumps(record))
    return path


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "maple_project"
    (root / "src" / "maple" / "core").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pyproject.toml").write_text(PYPROJECT)
    (root / "src" / "maple" / "__init__.py").write_text('"""fixture maple"""\nVALUE = 1\n')
    (root / "src" / "maple" / "core" / "physics.py").write_text("def f(x):\n    return 2 * x\n")
    (root / "src" / "maple" / "data.bin").write_bytes(bytes(range(256)) * 64)
    (root / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
    deep = root / "src" / "maple" / ("d" * 60) / ("e" * 60)
    deep.mkdir(parents=True)
    (deep / "long_name_module.py").write_text("X = 3\n")  # > 100-char tar path: pax header
    (root / "src" / "maple" / "__pycache__").mkdir()
    (root / "src" / "maple" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\0" * 10)  # excluded
    dist = tmp_path / "maple-0.0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(METADATA)
    (dist / "WHEEL").write_text("Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
    (dist / "top_level.txt").write_text("maple\n")
    (dist / "RECORD").write_text("ignored\n")
    origin = write_origin(tmp_path / "origin.json")
    return {"root": root, "dist": dist, "origin": origin, "tmp": tmp_path}
