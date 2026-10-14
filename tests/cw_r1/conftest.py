"""Bind the CW-R1 candidate tree's `maple_syrup` for tests/cw_r1 and expose the R1 preprocessing script (read-only)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SRC = CANDIDATE_ROOT / "src"
sys.path.insert(0, str(CANDIDATE_SRC))

import maple_syrup  # noqa: E402

_PACKAGE_DIR = Path(maple_syrup.__file__).resolve().parent
if _PACKAGE_DIR != (CANDIDATE_SRC / "maple_syrup").resolve():
    raise RuntimeError(f"tests/cw_r1 must exercise {CANDIDATE_SRC}, but maple_syrup was imported from {_PACKAGE_DIR}")

SYRUP = Path("/home/okin/SYRUP")
PREP_SCRIPT = SYRUP / "agent_handoffs/tasks/conservative_water/run_cw_r1_prepare_inputs.py"
B93_CHASTRE = SYRUP / "outputs/conservative_water/cw_b93_transport_accounting_frozen/src/maple_syrup/chastre_case.py"


def load_prep():
    spec = importlib.util.spec_from_file_location("_cw_r1_prepare_inputs", PREP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
