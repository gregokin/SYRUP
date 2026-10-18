"""Bind THIS tree's `maple_syrup` for tests/checkpoint_compat (MASY-03 prerequisite A) and record which MAPLE the run bound.

The physical fixtures are the existing small authored native fixtures (`tests/phase6/test_complete_event.fixture`,
`tests/pond_checkpoint/pond_fixture.make_case`), imported from their own directories. The only binding manipulation is the
`ledger_binding` fixture: it installs a SEPARATE synthetic module object (or a broken import) as
`sys.modules['maple.core.types.sediment_ledger']` for pure registry / refusal tests, never edits the real module, and
proves at teardown that the real module object and every one of its bindings are exactly restored."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

CANDIDATE_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_SRC = CANDIDATE_ROOT / "src"
_TESTS = CANDIDATE_ROOT / "tests"
sys.path.insert(0, str(CANDIDATE_SRC))
# checkpoint_compat_support; phase6 (a package whose modules import phase5 top-level modules); pond_checkpoint/pond_fixture
for _entry in (Path(__file__).resolve().parent, _TESTS, _TESTS / "phase5", _TESTS / "pond_checkpoint"):
    if str(_entry) not in sys.path:
        sys.path.insert(1, str(_entry))

import maple_syrup  # noqa: E402

_PACKAGE_DIR = Path(maple_syrup.__file__).resolve().parent
if _PACKAGE_DIR != (CANDIDATE_SRC / "maple_syrup").resolve():
    raise RuntimeError(f"tests/checkpoint_compat must exercise {CANDIDATE_SRC}, but maple_syrup was imported from {_PACKAGE_DIR}")

from checkpoint_compat_support import LEDGER_MODULE, LedgerBinding  # noqa: E402


def pytest_report_header(config):
    """Record the bound MAPLE and whether it literally binds the optional compact ledger class."""
    try:
        import maple
        ledger = importlib.import_module(LEDGER_MODULE)
    except Exception as exc:  # noqa: BLE001 - reported verbatim, the tests then fail on their own
        return [f"checkpoint_compat: MAPLE import failed ({type(exc).__name__}: {exc})"]
    compact = "CompactSedimentLedgerState" in vars(ledger)
    return [f"checkpoint_compat: maple_syrup={_PACKAGE_DIR}",
            f"checkpoint_compat: maple={Path(maple.__file__).resolve().parent} CompactSedimentLedgerState present={compact}"]


@pytest.fixture
def ledger_binding():
    """Synthetic-binding installer for PURE registry / refusal tests. Teardown proves exact restoration of the real module
    object in `sys.modules` and of every binding in its namespace (identity, not equality)."""
    real = importlib.import_module(LEDGER_MODULE)
    before = dict(vars(real))
    patch = pytest.MonkeyPatch()
    try:
        yield LedgerBinding(patch, real)
    finally:
        patch.undo()
    assert sys.modules[LEDGER_MODULE] is real, "the real MAPLE ledger module was not restored"
    after = vars(real)
    assert after.keys() == before.keys() and all(after[k] is before[k] for k in before), \
        "a binding of the real MAPLE ledger module changed"
