"""HISTORICAL author-candidate SOURCE AUDIT of the G-F4 context-outer task, preserved VERBATIM as task evidence and NOT collected by default (the filename is not `test_*.py`).

QUALIFICATION RECORD: it compares the candidate with `gpu_context_outer_pre_source_manifest.json`, the pre-edit manifest of the G-F4 task. It is STALE by source for every later tree: five of its six compared modules (`floating_pond`,
`floating_checkpoint`, `native_floating_soil_cuda`, `pond_event`, `floating_preferential_infiltration`) were legitimately changed by later accepted/qualified tasks before the CC2 baseline, and `floating_checkpoint_cuda.py` carries the CC2
continued-origin hooks. It passed only against its own task's tree. Run it explicitly against that tree if its history is ever re-audited; it must not be read as a statement about any later tree."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
TASK = HERE.parents[3]
SRC = HERE.parents[2] / "src" / "maple_syrup"
MANIFEST = TASK / "completion_program_20261113" / "gpu_context_outer_pre_source_manifest.json"

@pytest.mark.skipif(not MANIFEST.exists(), reason="pre-edit manifest not present")
def test_every_baseline_production_module_is_byte_identical_and_only_the_one_new_module_was_added():
    manifest = json.loads(MANIFEST.read_text())  # {module name: sha256}
    changed = sorted(n for n, sha in manifest.items() if hashlib.sha256((SRC / n).read_bytes()).hexdigest() != sha)
    present = {p.name for p in SRC.glob("*.py")}
    assert changed == [] and present - set(manifest) == {"floating_context_checkpoint_cuda.py"}  # an identity check of THIS task's baseline, not a whole-tree count invariant
    for name in ("floating_context_checkpoint.py", "floating_checkpoint.py", "floating_checkpoint_cuda.py", "floating_resume_cuda.py", "floating_held_context.py", "floating_held_context_cuda.py", "floating_pond.py", "pond_checkpoint.py"):
        assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == manifest[name]
