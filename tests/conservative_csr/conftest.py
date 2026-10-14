"""Bind the CW-B2 candidate tree's `maple_syrup` package, never the production one.

The candidate copy lives beside these tests (outputs/conservative_water/
csr_candidate_source/src). It is placed FIRST on `sys.path` so that an
installed or editable production `maple_syrup` cannot be collected by
mistake; the import is then checked to come from this tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SRC = CANDIDATE_ROOT / "src"
sys.path.insert(0, str(CANDIDATE_SRC))

import maple_syrup

_PACKAGE_DIR = Path(maple_syrup.__file__).resolve().parent
if _PACKAGE_DIR != (CANDIDATE_SRC / "maple_syrup").resolve():
    raise RuntimeError(
        f"tests/conservative_csr must exercise the candidate tree {CANDIDATE_SRC}, but maple_syrup was imported "
        f"from {_PACKAGE_DIR}; a production or installed copy shadows the candidate (check PYTHONPATH / editable "
        "installs)"
    )
