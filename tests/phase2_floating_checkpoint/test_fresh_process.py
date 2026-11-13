"""R2a: FRESH-PROCESS restart. Process A builds the original case from the factory, runs K steps, saves the native files + the floating checkpoint, continues M steps and dumps its final state; process B (a new interpreter)
rebuilds the ORIGINAL case, loads the checkpoint and continues M steps. Bit-exact equality of every physical array (hydraulic, W/J/I/D, native bed, phase, ledgers, counters, quiet, clocks). Neither process holds the other's
Pond state."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

WORKER = Path(__file__).resolve().parent / "fresh_worker.py"


def _run(mode, work, backend, windowed):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NUMBA_NUM_THREADS="1")
    r = subprocess.run([sys.executable, str(WORKER), mode, str(work), backend, "1" if windowed else "0", "8", "6"], capture_output=True, text=True, env=env, timeout=600, check=False)
    assert r.returncode == 0, r.stderr[-3000:]


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("windowed", [False, True])
def test_fresh_process_restart_is_bit_exact(tmp_path, backend, windowed):
    if backend == "numba":
        pytest.importorskip("numba")
    _run("save", tmp_path, backend, windowed)
    assert (tmp_path / "ck" / "floating_pond_checkpoint.json").is_file()
    _run("load", tmp_path, backend, windowed)
    a, b = np.load(tmp_path / "save_final.npz"), np.load(tmp_path / "load_final.npz")
    assert set(a.files) == set(b.files) and len(a.files) > 40
    for k in a.files:
        assert a[k].tobytes() == b[k].tobytes(), k
