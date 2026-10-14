"""CW-B15 review corrections: shared paths and fixtures."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tests/phase4", "tests/phase4s", "tests/hydraulic_candidates", "tests", "tests/rfid", "tests/phase5",
            "tests/conservative_water", "tests/cw_b15"):
    sys.path.insert(0, str(ROOT / sub))

#: The frozen B12 source this correction tree started from (cw_b15_syrup_baseline_manifest.json). Used ONLY as the
#: read-only "before" reference of the solver differential tests; never imported as `maple_syrup`.
FROZEN_B12_SRC = Path("/home/okin/SYRUP/outputs/conservative_water/cw_b12_scratch_frozen/src/maple_syrup")


@pytest.fixture(scope="session")
def gpu():
    """The CuPy module, or an explicit skip (a skipped test makes no GPU claim)."""
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA path is not exercised (no claim)")
    return backend.cupy_module()


@pytest.fixture(scope="session")
def frozen_b12():
    """The frozen B12 `routing` and `routing_numba` modules loaded by path under private names (read-only)."""
    import importlib.util

    if not FROZEN_B12_SRC.is_dir():
        pytest.skip("frozen B12 reference source unavailable; no before/after claim")
    modules = {}
    for name in ("routing", "routing_numba"):
        spec = importlib.util.spec_from_file_location(f"_cw_b15_frozen_b12_{name}", FROZEN_B12_SRC / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses resolve their module by name; private name, never maple_syrup
        spec.loader.exec_module(module)
        modules[name] = module
    return modules

# Genuine compiled synthetic Chastre / RFID cases (copied CW-B6 fixtures; tiny grids, actual MAPLE compile).
from cw_b15_case_fixtures import (  # noqa: F401
    chastre_case_dir,
    dtm_path,
    rfid_source,
    synthetic_recipe_file,
)
