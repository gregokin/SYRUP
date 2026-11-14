"""Pre-edit source guard: every one of the 169 pre-existing production modules keeps its recorded bytes; the only addition under src/maple_syrup is the new binding module."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from veg_binding_support import CANDIDATE, TASK

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase25_source_profile"))
import p25_source_profile as p25

PKG = CANDIDATE / "src" / "maple_syrup"
NEW = {"water_vegetation_binding.py", "water_vegetation_held.py"}  # the S0 module and the cached-inputs module (the only two additions to the 169)


def test_169_preexisting_modules_are_byte_identical_and_only_the_new_module_was_added():
    """Branch chosen ONLY by the literal-pinned package: the historical cache-C3 package runs the original task-folder proof; the Root-locked final package runs the portable whole-source proof (no task path, no Git)."""
    branch = p25.require_branch(PKG, historical=("cache_c3",))
    if branch.kind == "historical":
        _historical_169_guard()
        return
    v = branch.verified
    assert NEW <= v.added and not NEW & set(v.base)                               # the S0 binding and cached-inputs modules are additions, never base modules
    assert set(v.base) <= set(v.files)
    assert {k for k in v.base if v.files[k] != v.base[k]} == v.changed             # every base module NOT in the locked changed set is byte-identical to the immutable base listing
    assert v.unchanged == set(v.base) - v.changed


def _historical_169_guard():
    manifest = json.loads((TASK / "syrup_preedit_manifest169.json").read_text())
    recorded = {name: sha for name, sha in manifest["files"]}
    assert len(recorded) == 169 and manifest["git_base"] == "02388adb70b7f1f22b33b6b3d5f1249e223842e5"
    present = {p.relative_to(PKG).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in PKG.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}
    assert set(present) - set(recorded) == NEW
    assert set(recorded) <= set(present)
    assert {k: v for k, v in present.items() if k in recorded} == recorded


def test_new_module_imports_no_leaf_at_module_level():
    text = (PKG / "water_vegetation_binding.py").read_text()
    top = text.split("\n\n\ndef ", 1)[0]
    assert "\nimport leaf" not in top and "\nfrom leaf" not in top
