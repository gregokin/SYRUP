"""REAL SEPARATE-INTERPRETER seven-day N = 3 integration (correction01 C5; NumPy and Numba x Dense and published Windowed). Process A: actual LEAF session generation 1 -> days 1-3 (frame 3) -> accepted generation 2 -> days 4-5 -> pause at the CLOSED
day 5 (physical checkpoint + the N = 3 directory holding only frame 3) -> days 6-7. Process B (a NEW interpreter, original-bed / owner / provider / LEAF-plant tripwires armed before loading): load from files + pins + static case + the external
directory, resume the writer, days 6-7. Compared byte for byte: arrays, cursor, digests, physical t, current attributions, frame identity and the whole frame directory tree. Frame modules are a READONLY diagnostics dependency, not restart state."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from maple_syrup import floating_moisture_frame as g1

WORKER = Path(__file__).resolve().parent / "fresh_n3_worker.py"


def _run(mode, work, backend, storage):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NUMBA_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-B", str(WORKER), mode, str(work), backend, storage], capture_output=True, text=True, env=env, timeout=2400, check=False)
    assert r.returncode == 0, r.stderr[-4000:]


@pytest.mark.parametrize("storage", ["windowed", "dense"])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_separate_interpreter_pause_at_closed_day_5_and_continue_days_6_7_equals_the_uninterrupted_run(tmp_path, leaf, backend, storage):
    if backend == "numba":
        pytest.importorskip("numba")
    _run("save", tmp_path, backend, storage)
    _run("load", tmp_path, backend, storage)
    a, b = np.load(tmp_path / "save_final.npz"), np.load(tmp_path / "load_final.npz")
    assert set(a.files) == set(b.files) and len(a.files) == 6
    for k in a.files:
        assert a[k].tobytes() == b[k].tobytes(), k
    ma, mb = json.loads((tmp_path / "save_meta.json").read_text()), json.loads((tmp_path / "load_meta.json").read_text())
    assert ma["days"] == mb["days"]  # digests, cursor, physical t, current vegetation attributions (stable + current ids), frame identity
    assert ma["tree"] == mb["tree"] and {k.split("/")[0] for k in ma["tree"] if k.startswith("frame_")} == {g1.frame_name(3), g1.frame_name(6)}  # every sealed frame byte equals
