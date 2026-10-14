"""CW-B109: assert the actual imported MAPLE / SYRUP roots (launcher env CW_B108_EXPECT_*) and record them."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_sessionstart(session):
    import maple
    import maple_syrup

    expect_maple, expect_syrup = os.environ.get("CW_B108_EXPECT_MAPLE"), os.environ.get("CW_B108_EXPECT_SYRUP")
    if not expect_maple or not expect_syrup:
        raise pytest.UsageError("CW-B109 tests need CW_B108_EXPECT_MAPLE / CW_B108_EXPECT_SYRUP (run via cw_b109_run.py)")
    if (Path(maple.__file__).resolve().parent != Path(expect_maple).resolve()
            or Path(maple_syrup.__file__).resolve().parent != Path(expect_syrup).resolve()):
        raise pytest.UsageError("imported maple / maple_syrup are not the expected roots")


@pytest.fixture(scope="session", autouse=True)
def _record_roots(record_testsuite_property):
    import maple
    import maple_syrup

    from maple_syrup.sediment_bed import bound_maple_package_digest

    record_testsuite_property("maple_package", str(Path(maple.__file__).resolve().parent))
    record_testsuite_property("syrup_package", str(Path(maple_syrup.__file__).resolve().parent))
    record_testsuite_property("maple_package_digest", bound_maple_package_digest())
