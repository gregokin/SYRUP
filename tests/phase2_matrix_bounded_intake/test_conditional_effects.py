"""Conditional numerical effects of the bounded combined intake, accounted on IDENTICAL multi-cell data: the PRE-review02 sources (`s2_b2_c_review02_pre_sources`, copied into a temporary tree with the rest of the
current package) against the current sources, spatial NumPy and Numba. Where the bound is inactive every output array is BYTE-IDENTICAL; where it is active (the old recombination exceeded the pool or the room) the
cell changes and is accounted for explicitly. Author-written."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from conftest import CANDIDATE, PRE, backend  # noqa: F401

RUNNER = Path(__file__).resolve().parent / "_variant_runner.py"
needs_pre = pytest.mark.skipif(not PRE.exists(), reason="pre-review02 sources not present")


def _env(src: Path) -> dict:
    cur = str(CANDIDATE / "src")
    parts = [str(src) if p == cur else p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if str(src) not in parts:
        parts.insert(0, str(src))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(parts), PYTHONDONTWRITEBYTECODE="1", NUMBA_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    return env


@pytest.fixture(scope="module")
def old_tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("old_src")
    shutil.copytree(CANDIDATE / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("matrix_infiltration.py", "matrix_infiltration_numba.py", "matrix_infiltration_cuda.py"):
        shutil.copy(PRE / f, root / "src" / "maple_syrup" / f)
    return root / "src"


def _run(src: Path, backend: str, tmp: Path, tag: str) -> dict:
    out = tmp / f"{tag}_{backend}.npz"
    subprocess.run([sys.executable, str(RUNNER), str(out), backend], check=True, env=_env(src), capture_output=True, text=True, timeout=900)
    return dict(np.load(out))


@needs_pre
def test_where_the_bound_is_inactive_the_results_are_byte_identical_and_where_active_the_cell_is_accounted(backend, old_tree, tmp_path):
    new = _run(CANDIDATE / "src", backend, tmp_path, "new")
    old = _run(old_tree, backend, tmp_path, "old")
    total_active = 0
    for scen in range(4):
        g = lambda d, n: d[f"s{scen}_{n}"]  # noqa: E731
        pool, room, mask = g(old, "pool_in"), g(old, "room"), g(old, "mask")
        for key in ("pool_in", "room", "mask"):
            assert g(old, key).tobytes() == g(new, key).tobytes()  # identical inputs
        old_intake = g(old, "intake")
        active = mask & ((old_intake > pool) | (old_intake > room))  # the bound is active exactly where the OLD recombination exceeded the pool or the room
        total_active += int(active.sum())
        for name in ("water_out", "pool_out", "wet_out", "cum_out", "intake", "rows"):
            a, b = g(old, name), g(new, name)
            ne = (a != b) & ~(np.isnan(a) & np.isnan(b))
            diff = ne if ne.ndim == 1 else ne.any(axis=0)
            assert not (diff & ~active).any(), f"scenario {scen} {name}: a cell with an inactive bound changed"  # byte-equivalent where the bound is inactive
        new_intake, new_pool = g(new, "intake"), g(new, "pool_out")
        assert (new_intake[mask] <= pool[mask]).all() and (new_intake[mask] <= room[mask]).all()  # the new bound holds on EVERY cell
        assert (new_pool[mask] >= 0.0).all() and np.isfinite(new_pool).all()
        if active.any():
            assert (new_intake[active] <= old_intake[active]).all()  # the bound only ever REDUCES the intake
            assert (old_intake[active] - new_intake[active] <= 4 * np.spacing(old_intake[active] + 1e-300) + 4 * np.spacing(pool[active] + room[active])).all()  # by rounding-level amounts
            # cumulative ledger and pool debit use the SAME bounded intake
            delta_cum = g(new, "cum_out")[active] - g(old, "cum_out")[active]
            assert np.allclose(delta_cum, new_intake[active] - old_intake[active], rtol=0.0, atol=8 * float(np.spacing(g(old, "cum_out")[active].max() + 1.0)))
    assert total_active > 0  # the scenarios do exercise the active bound (room- and/or pool-limited recombination)
    print(f"{backend}: {total_active} cells with an active bound over 4 x 60000 cells")
