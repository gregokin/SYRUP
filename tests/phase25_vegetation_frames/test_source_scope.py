"""Phase 2.5 vegetation frames: the production-source scope of this slice against the immutable pre-edit manifest of the 169-file package (no whole-manifest rewrite): exactly the declared owning modules changed and exactly two modules
were added; every other production byte is identical. The historical package (exact literal digest) runs this original proof; the Root-locked final package runs the portable whole-source proof (a missing manifest or an unfilled final pin FAILS, never skips)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase25_source_profile"))
import p25_source_profile as p25

TASK = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[2] / "src" / "maple_syrup"
MANIFEST = TASK / "syrup_preedit_manifest169.json"
CHANGED = {"floating_matrix_infiltration.py", "native_floating_soil.py", "native_floating_soil_cuda.py", "floating_event_refresh.py", "floating_event_refresh_cuda.py", "floating_event_lifecycle.py",
           "floating_daily.py", "floating_daily_output.py", "floating_event_daily_bridge.py", "floating_held_context.py", "floating_event_lifecycle_cuda.py"}
ADDED = {"vegetation_lifecycle.py", "vegetation_lifecycle_cuda.py", "water_vegetation_binding.py", "water_vegetation_held.py", "floating_moisture_frame_vegetation.py", "floating_daily_output_vegetation.py"}  # + the v2 frame codec and output of THIS task  # the S0 binding is the UNACCEPTED dependency overlay (read-only here); vegetation_lifecycle.py is this slice's module


def _historical_scope_guard():
    assert MANIFEST.is_file(), f"the immutable pre-edit manifest is missing: {MANIFEST}"   # the historical branch never skips
    base = {rel: sha for rel, sha in json.loads(MANIFEST.read_text())["files"]}
    now = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.glob("*.py")}
    assert {k for k in base if now.get(k) != base[k]} == CHANGED
    assert set(now) - set(base) == ADDED
    assert set(base) <= set(now)


def test_only_the_declared_owning_modules_changed_and_two_modules_were_added():
    """Branch chosen ONLY by the literal-pinned package: the historical frames_c2 package runs the original task-folder proof; the Root-locked final package runs the portable whole-source proof, where this slice's declared
    changed / added modules must be members of the locked changed / added sets (the final set is the whole composition's, so equality with this slice's own sets is not asserted)."""
    branch = p25.require_branch(SRC, historical=("frames_c2",))
    if branch.kind == "historical":
        _historical_scope_guard()
        return
    v = branch.verified
    assert CHANGED <= v.changed and ADDED <= v.added and not ADDED & set(v.base)
    assert {k for k in v.base if v.files[k] != v.base[k]} == v.changed and set(v.base) <= set(v.files)
