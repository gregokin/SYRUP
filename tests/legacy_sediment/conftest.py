"""Fixtures for the original-Fortran sediment harness tests.

A genuinely unconfigured toolchain (no `MAPLE_SYRUP_GFORTRAN` and no `gfortran` on PATH) skips the Fortran tests. A CONFIGURED
toolchain that fails to build or run is a test FAILURE, never a skip (the build helpers raise)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "legacy_sediment"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import sources as S


@pytest.fixture(scope="session")
def toolchain():
    tc = S.require_toolchain()  # raises when a configured compiler does not run
    if tc is None:
        pytest.skip("no Fortran toolchain is configured")
    if not (S.fr.MAHLERAN_ROOT / "src").is_dir():
        pytest.skip("the MAHLERAN reference tree is not present")
    return tc


@pytest.fixture(scope="session")
def hooked_build(toolchain, tmp_path_factory):
    return S.build(tmp_path_factory.mktemp("fortran") / "hooked", hooked=True, variant="checked")


@pytest.fixture(scope="session")
def nohook_build(toolchain, tmp_path_factory):
    return S.build(tmp_path_factory.mktemp("fortran") / "nohook", hooked=False, variant="checked")


@pytest.fixture(scope="session")
def walk_probe_build(toolchain, tmp_path_factory):
    return S.build_walk_probe(tmp_path_factory.mktemp("fortran") / "walk")
