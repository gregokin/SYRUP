"""H0: reconstruction in a FRESH process from ONLY the stored record parts and static identities (no original bed or owner exists there; the property provider and every bed reader are disabled)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("fresh_worker.py")


def run(mode, out, backend):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, str(WORKER), mode, str(out), str(TESTS), backend], capture_output=True, text=True, env=env, timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_fresh_process_rebuilds_the_exact_identities_with_no_bed_and_no_owner(tmp_path, backend):
    if backend == "numba":
        pytest.importorskip("numba")
    run("export", tmp_path / "rec", backend)
    ex = json.loads((tmp_path / "rec" / "expected.json").read_text())
    got = run("restore", tmp_path / "rec", backend)
    assert (got["config"], got["column"], got["matrix"], got["pref"]) == (ex["config_sha256"], ex["root_column_sha256"], ex["matrix_op_sha256"], ex["pref_sha256"])
    assert got["overrides"] == ["theta_33_override"] and got["has_water"] is False
    # the true override bytes (NaN pattern included) are exact
    meta = json.loads((tmp_path / "rec" / "meta.json").read_text())
    assert meta["overrides"] == ["theta_33_override"]
    assert bytes.fromhex(got["override_bytes"]) == (tmp_path / "rec" / "override_theta_33_override.bin").read_bytes()
