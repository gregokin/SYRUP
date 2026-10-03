"""Shared fixtures of the RFID tests. Nothing here was run by its author (file-only tools); Codex records results."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tests/phase4", "tests/phase4s", "tests/hydraulic_candidates", "tests", "benchmarks/rfid",
            "benchmarks/hydraulic_candidates", "tests/rfid"):
    sys.path.insert(0, str(ROOT / sub))

RECIPE = ROOT / "cases" / "rfid" / "recipe.yaml"


@pytest.fixture(scope="session")
def gpu():
    """The CuPy module, or an explicit skip (a skipped test makes no GPU claim). Select the device with CUDA_VISIBLE_DEVICES
    BEFORE the process starts."""
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA paths are not exercised")
    return backend.cupy_module()


@pytest.fixture(scope="session")
def rfid_recipe():
    pytest.importorskip("maple")
    pytest.importorskip("yaml")
    from maple_syrup.rfid_case import load_rfid_recipe

    return load_rfid_recipe(RECIPE)


@pytest.fixture(scope="session")
def synthetic_recipe(tmp_path_factory):
    """The committed recipe copied to a temporary file whose capture pin is the digest of the SYNTHETIC capture; the committed
    recipe keeps its real pin. The copy is what verify_rfid_case re-reads."""
    import hashlib

    from rfid_helpers import synthetic_capture

    from maple_syrup.rfid_case import load_rfid_recipe

    digest = hashlib.sha256(synthetic_capture().encode("ascii")).hexdigest()
    text = RECIPE.read_text()
    real = load_rfid_recipe(RECIPE).raw["forcing"]["capture_expected_sha256"]
    assert real in text
    path = tmp_path_factory.mktemp("synthetic_recipe") / "recipe.yaml"
    path.write_text(text.replace(real, digest))
    return load_rfid_recipe(path)


@pytest.fixture(scope="session")
def rfid_audit(rfid_recipe, tmp_path_factory):
    """The audit only (reads the MAHLERAN inputs, stages copies into a temporary directory; no MAPLE compile)."""
    from rfid_helpers import RFID_INPUT

    if not (RFID_INPUT / "mahleran_input.xml").is_file():
        pytest.skip("MAHLERAN RFID_2014 input not available")
    from maple_syrup.rfid_case import audit_rfid

    return audit_rfid(rfid_recipe, tmp_path_factory.mktemp("rfid_audit") / "case")


@pytest.fixture(scope="session")
def rfid_case(synthetic_recipe, tmp_path_factory):
    """A fully generated, compiled and verified RFID case (synthetic applied-forcing capture)."""
    from rfid_helpers import RFID_INPUT, synthetic_capture

    if not (RFID_INPUT / "mahleran_input.xml").is_file():
        pytest.skip("MAHLERAN RFID_2014 input not available")
    from maple_syrup.rfid_case import generate_rfid_case, verify_rfid_case

    base = tmp_path_factory.mktemp("rfid_case")
    capture = base / "syrup_hydro_steps.txt"
    capture.write_text(synthetic_capture())
    out = base / "case"
    generate_rfid_case(synthetic_recipe, out, applied_forcing_capture=capture)
    return verify_rfid_case(out, allow_maple_source_change=True)
