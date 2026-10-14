"""CW-B108: every child asserts the ACTUAL imported MAPLE / SYRUP module roots (B107c03 pitfall: a test file's rootdir
may otherwise import another tree). The launcher exports CW_B108_EXPECT_{MAPLE,SYRUP}; without them (an ad-hoc run)
the session refuses rather than guessing."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Reuse (by import) the actual-MAPLE bed helpers of the Phase 5 / conservative-water tests, as cw_b104 does.
sys.path.insert(0, str(ROOT / "tests" / "phase5"))
sys.path.insert(0, str(ROOT / "tests" / "conservative_water"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _roots():
    import maple
    import maple_syrup

    return Path(maple.__file__).resolve().parent, Path(maple_syrup.__file__).resolve().parent


def pytest_sessionstart(session):
    expect_maple, expect_syrup = os.environ.get("CW_B108_EXPECT_MAPLE"), os.environ.get("CW_B108_EXPECT_SYRUP")
    if not expect_maple or not expect_syrup:
        raise pytest.UsageError("CW-B108 tests need CW_B108_EXPECT_MAPLE / CW_B108_EXPECT_SYRUP (run via cw_b108_run.py)")
    maple_dir, syrup_dir = _roots()
    if maple_dir != Path(expect_maple).resolve() or syrup_dir != Path(expect_syrup).resolve():
        raise pytest.UsageError(f"imported maple {maple_dir} / maple_syrup {syrup_dir} are not the expected "
                                f"{expect_maple} / {expect_syrup}")
    if ROOT.resolve() != syrup_dir.parents[1]:
        raise pytest.UsageError(f"test tree {ROOT} is not the imported SYRUP tree {syrup_dir.parents[1]}")


@pytest.fixture(scope="session", autouse=True)
def _record_roots(record_testsuite_property):
    maple_dir, syrup_dir = _roots()
    record_testsuite_property("maple_package", str(maple_dir))
    record_testsuite_property("syrup_package", str(syrup_dir))
    record_testsuite_property("cupy_imported_at_start", str("cupy" in sys.modules))


@pytest.fixture(params=["numpy", "cupy"])
def xp(request):
    """NumPy always; CuPy only with a device (an explicit skip claims nothing). CPU children hide CUDA."""
    from maple.core.backend import BackendUnavailableError, resolve_backend

    try:
        return resolve_backend(request.param).xp
    except BackendUnavailableError as exc:
        pytest.skip(f"{request.param} unavailable: {exc}")
