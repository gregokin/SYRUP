"""Source scope of the vegetation physical-checkpoint slice against the IMMUTABLE pre-edit manifest (`checkpoint_preedit_manifest.json`: base commit 02388ad + the explicit overlay, 177 modules). Never ambient HEAD / `git diff`: the
manifest hashes are the baseline. Exactly the declared owning hooks changed, exactly the new modules were added, every other production byte is identical, and the OLD held / lifecycle-checkpoint modules stay LEAF-free."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase25_source_profile"))
import p25_source_profile as p25

TASK = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[2] / "src" / "maple_syrup"
MANIFEST = TASK / "checkpoint_preedit_manifest.json"
CHANGED = {"floating_event_lifecycle.py", "vegetation_lifecycle.py"}
ADDED = {"floating_lifecycle_checkpoint_vegetation.py", "floating_lifecycle_checkpoint_vegetation_cuda.py"}


def _historical_scope_guard():
    assert MANIFEST.is_file(), f"the immutable pre-edit manifest is missing: {MANIFEST}"   # the historical branch never skips
    d = json.loads(MANIFEST.read_text())
    assert d["files"] == 177 and d["base_commit"].startswith("02388ad")
    base = {e["path"]: e["sha256"] for e in d["entries"]}
    assert len(base) == 177
    now = {p.relative_to(SRC).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.rglob("*.py")}
    assert {k for k in base if now.get(k) != base[k]} == CHANGED
    assert set(now) - set(base) == ADDED
    assert set(base) <= set(now)


def test_only_the_owning_hooks_changed_and_only_the_new_modules_were_added():
    """Branch chosen ONLY by the literal-pinned package: the historical checkpoint-C0 package runs the original 177-module pre-edit proof; the Root-locked final package runs the portable whole-source proof against the
    169-file base, where this slice's hook modules must be locked changes / additions (`vegetation_lifecycle.py` is itself a locked addition to the 169) and its two new modules locked additions."""
    branch = p25.require_branch(SRC, historical=("checkpoint_c0",))
    if branch.kind == "historical":
        _historical_scope_guard()
        return
    v = branch.verified
    assert ADDED <= v.added and not ADDED & set(v.base)
    assert {"floating_event_lifecycle.py"} <= v.changed and {"vegetation_lifecycle.py"} <= v.added
    assert {k for k in v.base if v.files[k] != v.base[k]} == v.changed and set(v.base) <= set(v.files)


@pytest.mark.parametrize("module", ["maple_syrup.floating_held_context", "maple_syrup.floating_held_context_cuda", "maple_syrup.floating_lifecycle_checkpoint", "maple_syrup.floating_event_lifecycle",
                                    "maple_syrup.floating_continued_checkpoint"])
def test_old_modules_import_without_leaf_or_the_new_profile(module):
    code = f"import sys, {module}; assert 'leaf' not in sys.modules and 'maple_syrup.floating_lifecycle_checkpoint_vegetation' not in sys.modules; print('ok')"
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=dict(os.environ), check=False)
    assert out.stdout.strip() == "ok", out.stderr
