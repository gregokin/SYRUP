"""Source-guard PROVENANCE of the combined candidate (175 production modules), PORTABLE: it never reads an ambient `git HEAD` or a working-tree path. The immutable references are
(1) the base commit `02388adb70b7f1f22b33b6b3d5f1249e223842e5`, through the archived `syrup_preedit_manifest169.json` (the 169 base file hashes; its `git_base` must be that commit) and, for the additive proof of the old held-context
module, the immutable BASE BLOB `git --git-dir=<explicit> show BASE:src/maple_syrup/floating_held_context.py` (blob hash checked against the manifest; a missing object store / base is a clear hard failure, never a fallback);
(2) the EXACT Root overlay receipt `root_held_context_c1_overlay_receipt.json` (the LAST record per file = its final bytes). Every unowned file is verified byte-equal to the manifest or to the receipt; the added module names are
exact; the old module differs from the BASE blob only by insertions (a `difflib` unified diff of the actual file against the base blob, no removed line); the new modules are imported lazily. The paths are derived from this file's
location (a package copy / Root's frozen snapshot works): the task directory is the package root's parent. The object store is `HELD_CONTEXT_GIT_DIR` (default: the repository's `.git`).

PHASE 2.5 FINAL-SOURCE BRANCH: each source test first asks `p25_source_profile.require_branch` which package this is. ONLY the exact literal-pinned historical held-context-C3 package runs the original proof above; ONLY the exact literal-pinned
Root-locked final package runs the portable whole-source proof (no task path, no Git object store: the base blob is the shipped byte-exact fixture); anything else, or an unfilled final pin, fails."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase25_source_profile"))
import p25_source_profile as p25

BASE = "02388adb70b7f1f22b33b6b3d5f1249e223842e5"
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "maple_syrup"
TASK = ROOT.parent
RECEIPT = TASK / "root_held_context_c1_overlay_receipt.json"
MANIFEST = TASK / "syrup_preedit_manifest169.json"
GIT_DIR = Path(os.environ.get("HELD_CONTEXT_GIT_DIR", "/home/okin/SYRUP/.git"))
OWN = {"floating_held_context.py"}
OWN_NEW = {"floating_held_context_vegetation.py", "floating_held_context_vegetation_cuda.py"}
RECEIPT_NEW = {"vegetation_lifecycle.py", "vegetation_lifecycle_cuda.py", "water_vegetation_binding.py", "water_vegetation_held.py"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _base_blob(rel: str) -> bytes:
    """The immutable BASE blob (explicit git dir, explicit commit): a missing object store / commit is a HARD failure with a clear message."""
    if not (GIT_DIR / "objects").is_dir():
        pytest.fail(f"the immutable base object store is missing: {GIT_DIR} (set HELD_CONTEXT_GIT_DIR); the guard never falls back to an ambient HEAD")
    done = subprocess.run(["git", f"--git-dir={GIT_DIR}", "show", f"{BASE}:{rel}"], capture_output=True, check=False)
    if done.returncode != 0:
        pytest.fail(f"the base commit {BASE} / {rel} is not readable from {GIT_DIR}: {done.stderr.decode()[:200]}")
    return done.stdout


@pytest.fixture(scope="module")
def base() -> dict:
    assert RECEIPT.is_file() and MANIFEST.is_file(), f"the immutable records are missing: {RECEIPT} / {MANIFEST}"
    m = json.loads(MANIFEST.read_text())
    assert m["git_base"] == BASE and m["file_count"] == 169 == len(m["files"]), "the archived 169-file manifest is not the base commit's"
    return {name: sha for name, sha in m["files"]}


@pytest.fixture(scope="module")
def receipt() -> dict:
    d = json.loads(RECEIPT.read_text())
    final = {}
    for c in d["changes"]:
        final[Path(c["file"]).name] = c["new"]  # the LAST record of a file is its FINAL overlaid bytes
    return final


def _historical_provenance(base, receipt):
    on_disk = {p.name for p in SRC.rglob("*.py")}
    assert len(base) == 169 and len(on_disk) == 175
    assert on_disk - set(base) == RECEIPT_NEW | OWN_NEW and set(base) <= on_disk                                 # exact added names; nothing of the base is missing
    for name in sorted(set(base) - OWN):
        want = receipt.get(name, base[name])                                                                     # the receipt's final bytes win for an overlaid file, else the immutable base hash
        assert _sha(SRC / name) == want, f"{name} differs from the {'Root overlay receipt' if name in receipt else 'base manifest'}"
    assert OWN.isdisjoint(receipt) and OWN <= set(base)                                                          # the owning hook module is not in the Root receipt
    for name in RECEIPT_NEW:
        assert name in receipt and _sha(SRC / name) == receipt[name], f"{name} differs from the Root overlay receipt"


def test_the_combined_source_is_exactly_the_recorded_provenance(request):
    branch = p25.require_branch(SRC, historical=("held_context_c3",))
    if branch.kind == "historical":
        _historical_provenance(request.getfixturevalue("base"), request.getfixturevalue("receipt"))
        return
    v = branch.verified
    assert RECEIPT_NEW | OWN_NEW <= v.added and not (RECEIPT_NEW | OWN_NEW) & set(v.base)                       # the held-context additions are locked additions, never base modules
    assert OWN <= v.changed and set(v.base) <= set(v.files)                                                      # the owning hook module is a locked change of a base module
    assert {k for k in v.base if v.files[k] != v.base[k]} == v.changed and v.unchanged == set(v.base) - v.changed  # every other base module is byte-identical to the immutable base listing


def test_the_old_held_context_module_is_touched_only_additively(request):
    branch = p25.require_branch(SRC, historical=("held_context_c3",))
    if branch.kind == "historical":
        blob, want = _base_blob("src/maple_syrup/floating_held_context.py"), request.getfixturevalue("base")["floating_held_context.py"]
    else:
        blob, want = p25.read_fixture("base_blob_floating_held_context.blob"), branch.verified.base["floating_held_context.py"]   # the shipped byte-exact base blob (no Git object store)
    assert hashlib.sha256(blob).hexdigest() == want, "the base blob is not the manifest's hash"
    old = blob.decode().splitlines()
    now = (SRC / "floating_held_context.py").read_text().splitlines()
    diff = list(difflib.unified_diff(old, now, "BASE", "ACTUAL", n=0, lineterm=""))
    removed = [ln for ln in diff if ln.startswith("-") and not ln.startswith("---")]
    assert removed == [], removed                                                                               # purely ADDITIVE against the immutable base blob
    added = "\n".join(ln for ln in diff if ln.startswith("+") and not ln.startswith("+++"))
    for token in ("reconstruct_vegetation_held_context", "per-cell vegetation factor", "vegetation: Any = None", "record_schema(record)", "fhcuda._gate()"):
        assert token in added, token
    assert not [ln for ln in added.splitlines() if ln.startswith(("+from maple_syrup import floating_held_context_vegetation", "+import maple_syrup.floating_held_context_vegetation"))]   # lazy imports only
    for ln in added.splitlines():                                                                               # every module-level import line of the new code is inside a function (indented)
        assert not ln.startswith("+from maple_syrup import floating_held_context_vegetation")


def test_the_guard_reads_only_explicit_immutable_references():
    """Portability proof: every git invocation in this guard names an explicit `--git-dir=` and the explicit base commit (no ambient HEAD, `ls-tree`, working-tree diff or cwd-relative `-C` form); the 169 base hashes come from
    the archived manifest."""
    whole = Path(__file__).read_text()
    code = whole[: whole.index("def test_the_guard_reads_only_explicit_immutable_references")]
    git_calls = [ln for ln in code.splitlines() if '["git"' in ln]
    assert len(git_calls) == 1 and "--git-dir=" in git_calls[0] and '"show", f"{BASE}:{rel}"' in git_calls[0], git_calls
    assert '"-C"' not in code and "--" + "full-tree" not in code and "git_base" in code and BASE in code


@pytest.mark.parametrize("module", ["maple_syrup.floating_held_context", "maple_syrup.floating_held_context_cuda"])
def test_old_modules_import_without_leaf(module):
    import sys
    out = subprocess.run([sys.executable, "-B", "-c", f"import sys, {module}; assert 'leaf' not in sys.modules; print('ok')"], capture_output=True, text=True, env=dict(os.environ), check=False)
    assert out.stdout.strip() == "ok", out.stderr
