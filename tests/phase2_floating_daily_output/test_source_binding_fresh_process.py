"""G3 S1/S2 correction01: a model-source change (even comment-only) in a FRESH process must refuse the output of the original source; the writer module bytes are identical. The genuine original state is built in each
process (no fake cursor, no physical-restart claim); the variant is a byte copy of the package with one comment line appended to `floating_daily.py`."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import maple_syrup

SRC = Path(maple_syrup.__file__).resolve().parent
TESTS = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("source_worker.py")


def run(mode, out_dir, package_parent):
    env = dict(os.environ)
    parts = [str(package_parent)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and Path(p).resolve() != SRC.parent.resolve()]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MAPLE_SYRUP_EXPECTED_MAPLE_ROOT"] = env.get("MAPLE_SYRUP_EXPECTED_MAPLE_ROOT", "")
    proc = subprocess.run([sys.executable, str(WORKER), mode, str(out_dir), str(TESTS)], capture_output=True, text=True, env=env, timeout=240, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_fresh_process_with_a_changed_model_source_refuses_the_original_output(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    variant = tmp_path / "variant"
    shutil.copytree(SRC, variant / "maple_syrup", ignore=shutil.ignore_patterns("__pycache__"))
    target = variant / "maple_syrup" / "floating_daily.py"
    target.write_text(target.read_text() + "\n# comment-only source change\n")
    writer_a, writer_b = (SRC / "floating_daily_output.py").read_bytes(), (variant / "maple_syrup" / "floating_daily_output.py").read_bytes()
    assert writer_a == writer_b  # the output writer module bytes are IDENTICAL; only the model source differs
    first = run("create", out, SRC.parent)
    assert first["created"] is True
    again = run("resume", out, SRC.parent)  # a fresh process of the SAME source accepts its own output
    assert again["accepted"] is True and again["pair"] == first["pair"]
    other = run("resume", out, variant)
    assert Path(other["source"]).resolve() == (variant / "maple_syrup").resolve()
    assert other["pair"]["maple_syrup_tree_sha256"] != first["pair"]["maple_syrup_tree_sha256"]
    assert other["accepted"] is False and "binding" in other["error"]  # different model source: refused, no migration/rebind
