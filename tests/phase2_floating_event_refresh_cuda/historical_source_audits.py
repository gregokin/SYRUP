"""HISTORICAL author-refresh-tree SOURCE AUDIT (preserved VERBATIM; NOT collected by default: the filename is not `test_*.py`). It asserts the diff against the refresh baseline manifest is exactly the refresh tree's own three
files, so it holds only in the author-refresh tree and is EXPECTED to differ in a tree built on the lifecycle baseline. Run explicitly only against the matching frozen refresh snapshot:
`pytest tests/phase2_floating_event_refresh_cuda/historical_source_audits.py`. The per-interface replacement for the lifecycle tree is `tests/phase2_floating_event_lifecycle_cuda/test_cuda_lifecycle_host.py::test_production_diff_against_the_lifecycle_baseline`.
Production provenance guards (kernel-preservation, source-pair, dependency tests) are never disabled."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
TASK = HERE.parents[3]
SRC = HERE.parents[2] / "src" / "maple_syrup"
MANIFEST = TASK / "completion_program_20261113" / "event_refresh_cuda_baseline.json"
CHANGED_PROD = {"native_floating_soil_cuda.py", "floating_preferential_infiltration.py"}
NEW_PROD = {"floating_event_refresh_cuda.py"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.skipif(not MANIFEST.exists(), reason="baseline manifest not present")
def test_production_diff_is_exactly_the_three_declared_files():
    manifest = json.loads(MANIFEST.read_text())
    rel = {k.split("/")[-1]: v for k, v in manifest.items() if k.startswith("src/maple_syrup/") and k.endswith(".py")}
    differing = {n for n, h in rel.items() if sha(SRC / n) != h}
    present = {p.name for p in SRC.glob("*.py")}
    assert differing == CHANGED_PROD and present - set(rel) == NEW_PROD and set(rel) <= present
    for name in ("floating_event_refresh.py", "floating_completion.py", "floating_pond.py", "floating_pond_cuda.py", "floating_soil_layers.py", "floating_soil_layers_cuda.py", "floating_matrix_infiltration.py",
                 "native_floating_soil.py", "floating_checkpoint_cuda.py", "floating_context_checkpoint_cuda.py" if False else "floating_checkpoint.py"):
        assert sha(SRC / name) == rel[name]  # the CPU refresh/EB1/FP/D-layer/nfs modules and kernels are byte-identical
