"""HISTORICAL author-candidate SOURCE AUDITS of the Dense GPU file task (preserved VERBATIM as task evidence; NOT collected by default: the filename is not `test_*.py`). They compare the candidate against the 155-file baseline,
the pre-H2-C4 `floating_checkpoint.py` and the 157-module Root snapshots, so they are EXPECTED to differ after the legitimate H2-C4 + EB1 reconciliation (158 modules). Run them explicitly only against the matching frozen
snapshots (original targets: the 155-file `gpu_dense_file_pre_source_manifest.json` baseline and `root_h_cuda_factory_snapshot155` for the `floating_pond.py` and manifest audits, `root_gpu_dense_file_snapshot157` for the `floating_checkpoint.py` audit, `root_gpu_dense_file_c1/c2_snapshot157` for the 157-module identity audit): `pytest tests/phase2_floating_checkpoint_cuda/historical_source_audits.py`. The public regressions in `test_cuda_file_host.py` replace them with per-interface behaviour/AST checks. Production provenance guards
(the kernel-preservation, core source-pair and dependency tests) are never disabled."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from maple_syrup import pond_checkpoint as pc

HERE = Path(__file__).resolve()
TASK = HERE.parents[3]
SRC = HERE.parents[2] / "src" / "maple_syrup"
MANIFEST = TASK / "completion_program_20261113" / "gpu_dense_file_pre_source_manifest.json"
PROD_CHANGED = {"floating_pond.py", "floating_checkpoint.py"}
PROD_NEW = {"floating_checkpoint_cuda.py", "floating_resume_cuda.py"}


@pytest.mark.skipif(not MANIFEST.exists(), reason="pre-edit manifest not present")
def test_only_floating_pond_and_the_one_scalar_conversion_line_differ_from_the_155_file_baseline_and_exactly_two_modules_are_new():
    manifest = json.loads(MANIFEST.read_text())["files"]
    differing = {n for n, sha in manifest.items() if hashlib.sha256((SRC / n).read_bytes()).hexdigest() != sha}
    present = {p.name for p in SRC.glob("*.py")}
    assert differing == PROD_CHANGED and present - set(manifest) == PROD_NEW and set(manifest) <= present
    for name in ("floating_resume.py", "pond_checkpoint.py", "floating_context_checkpoint.py", "floating_held_context.py", "floating_held_context_cuda.py", "floating_pond_cuda.py"):
        assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == manifest[name]  # the CPU child, the historical codec and the H0/E5 modules are byte-identical


def test_floating_pond_differs_only_in_the_prose_and_the_backend_dispatch():
    """The only code change is `_resume_module` + two call sites; every other definition is AST-identical to the baseline."""
    base = TASK / "root_h_cuda_factory_snapshot155" / "src" / "maple_syrup" / "floating_pond.py"
    if not base.exists():
        pytest.skip("baseline snapshot not present")

    def defs(path):
        tree = ast.parse(path.read_text())
        out = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef):
                        body = list(sub.body)
                        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                            body = body[1:]  # docstrings are prose
                        out[f"{node.name}.{sub.name}"] = ast.dump(ast.Module(body=body, type_ignores=[]))
            elif isinstance(node, ast.FunctionDef):
                out[node.name] = ast.dump(node)
        return out

    new, old = defs(SRC / "floating_pond.py"), defs(base)
    changed = {k for k in set(new) | set(old) if new.get(k) != old.get(k)}
    assert changed == {"FloatingPondVertical._resume_module", "FloatingPondVertical.resume_scope", "FloatingPondVertical.adopt_resumed"}
    assert pc is not None


def test_floating_checkpoint_differs_from_the_baseline_only_by_the_counted_host_conversion_in_validate_scalars():
    """Exact AST/line equivalence: `fc._validate_scalars` is the baseline text plus ONE import line and the `to_host` wrapper of the deposited_terminal read; every other definition is AST-identical."""
    import difflib

    base = TASK / "root_gpu_dense_file_snapshot157" / "src" / "maple_syrup" / "floating_checkpoint.py"
    if not base.exists():
        pytest.skip("baseline snapshot not present")
    new_text, old_text = (SRC / "floating_checkpoint.py").read_text(), base.read_text()

    def defs(text):
        return {n.name: ast.dump(n) for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    a, b = defs(new_text), defs(old_text)
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"_validate_scalars"}
    changed = [ln for ln in difflib.unified_diff(old_text.splitlines(), new_text.splitlines(), lineterm="", n=0) if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    assert [ln for ln in changed if ln.startswith("-")] == ["-    if (terminal > 0) != bool(np.any(np.asarray(state.sediment.deposited_terminal) > 0.0)):"]
    adds = [ln for ln in changed if ln.startswith("+") and ln[1:].strip()]
    assert len(adds) == 2 and adds[0] == "+    from maple.core.backend import to_host" and "np.asarray(to_host(state.sediment.deposited_terminal))" in adds[1]


@pytest.mark.parametrize("snapshot_name", ["root_gpu_dense_file_c1_snapshot157", "root_gpu_dense_file_c2_snapshot157"])
def test_all_157_production_modules_are_byte_identical_to_the_root_snapshots(snapshot_name):
    snap = TASK / snapshot_name / "src" / "maple_syrup"
    if not snap.exists():
        pytest.skip("snapshot not present")
    names = sorted(p.name for p in snap.glob("*.py"))
    assert names == sorted(p.name for p in SRC.glob("*.py")) and len(names) == 157
    assert [n for n in names if (snap / n).read_bytes() != (SRC / n).read_bytes()] == []
