"""S1: FRESH-PROCESS restart of a CLOSED daily state (Windowed bed): a new interpreter with only the static case recipe, no original bed/owner/probe/provider, restores with the writer's sealed digest and runs the next day bit-exactly."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

WORKER = Path(__file__).resolve().parent / "fresh_worker.py"


def _run(mode, work, backend, storage):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NUMBA_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-B", str(WORKER), mode, str(work), backend, storage], capture_output=True, text=True, env=env, timeout=1200, check=False)
    assert r.returncode == 0, r.stderr[-4000:]


@pytest.mark.parametrize("storage", ["windowed", "dense"])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_fresh_process_closed_day_restore_is_bit_exact(tmp_path, backend, storage):
    if backend == "numba":
        pytest.importorskip("numba")
    _run("save", tmp_path, backend, storage)
    _run("load", tmp_path, backend, storage)
    a, b = np.load(tmp_path / "save_final.npz"), np.load(tmp_path / "load_final.npz")
    assert set(a.files) == set(b.files) and len(a.files) == 7
    for k in a.files:
        assert a[k].tobytes() == b[k].tobytes(), k
