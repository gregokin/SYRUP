"""Executed-routine equivalence: the ORIGINAL MAHLERAN `flow_distrib` versus the Python/Numba walk kernel.

Builds the unmodified `shared_data.f90` + `flow_distrib.for` with a driver that only sets state and calls
the routine (benchmarks/phase7e/flow_distrib_driver.f90), on a synthetic legacy grid with its boundary
ring. Skips when the isolated Fortran toolchain is not configured (tests/phase4/fortran_reference.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "phase4"))
from fortran_reference import MAHLERAN_ROOT, locate_toolchain

from maple_syrup import legacy_transport as L

SOURCES = ("src/Program_Control/shared_data.f90", "src/Subroutines_Sediment/flow_distrib.for")
DRIVER = ROOT / "benchmarks/phase7e/flow_distrib_driver.f90"
# Legacy aspect codes on the north-first grid: 1 N (im-1), 2 E (jm+1), 3 S (im+1), 4 W (jm-1).
STEP = {1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}


def build(tmp_path):
    toolchain = locate_toolchain()
    if toolchain is None:
        pytest.skip("no Fortran toolchain configured")
    objs = []
    for src in [MAHLERAN_ROOT / s for s in SOURCES] + [DRIVER]:
        obj = tmp_path / (src.stem + ".o")
        cmd = [toolchain.compiler, *toolchain.flags, "-std=legacy", "-fcheck=all", "-O0", "-fno-fast-math",
               "-ffree-line-length-none", "-ffixed-line-length-none", f"-J{tmp_path}", f"-I{tmp_path}", "-c", str(src), "-o", str(obj)]
        done = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, check=False, timeout=120)
        if done.returncode:
            pytest.skip(f"Fortran compile failed: {done.stderr[-800:]}")
        objs.append(str(obj))
    exe = tmp_path / "flow_distrib_driver"
    done = subprocess.run([toolchain.compiler, *toolchain.flags, *objs, *toolchain.link_flags, "-o", str(exe)],
                          cwd=tmp_path, capture_output=True, text=True, check=False, timeout=120)
    if done.returncode:
        pytest.skip(f"Fortran link failed: {done.stderr[-800:]}")
    return exe, toolchain


@pytest.mark.parametrize("seed", [1, 2])
def test_python_walk_matches_original_flow_distrib(tmp_path, seed):
    exe, toolchain = build(tmp_path)
    rng = np.random.default_rng(seed)
    nr2, nc2, nclass = 9, 7, 3  # legacy grid WITH the one-cell ring; interior rows/cols 2..nr2-1 / 2..nc2-1
    dx_m, dt = 0.5, 1.0
    # Interior aspects: a plane draining north (1) with some east/west meanders, acyclic by construction.
    aspect = np.zeros((nr2, nc2), dtype=np.int64)
    for i in range(1, nr2 - 1):
        for j in range(1, nc2 - 1):
            aspect[i, j] = 1 if (i == 1 or rng.random() < 0.6) else (2 if j < nc2 - 2 else 4)
    # receivers in legacy coordinates: ring -> EXPORT
    n = nr2 * nc2
    receiver = np.full(n, L.EXPORT, dtype=np.int64)
    interior = np.zeros(n, dtype=bool)
    for i in range(1, nr2 - 1):
        for j in range(1, nc2 - 1):
            di, dj = STEP[int(aspect[i, j])]
            ii, jj = i + di, j + dj
            idx = i * nc2 + j
            interior[idx] = True
            receiver[idx] = L.EXPORT if aspect[ii, jj] == 0 else ii * nc2 + jj
    calls = []
    det = np.zeros((n, nclass)); inv = np.zeros((n, nclass)); nsteps = np.zeros((n, nclass), dtype=np.int64)
    law = np.zeros((n, nclass), dtype=bool)
    for idx in np.flatnonzero(interior):
        for k in range(nclass):
            if rng.random() < 0.5:
                d = float(rng.uniform(0.1, 2.0)); Lm = float(rng.choice([0.05, 0.3, 1.0, 4.0, 40.0])); ns = int(rng.choice([2, 5, 20, 200]))
                det[idx, k] = d; inv[idx, k] = 1.0 / Lm; nsteps[idx, k] = ns; law[idx, k] = True
                calls.append((idx // nc2 + 1, idx % nc2 + 1, k + 1, d, Lm, ns))  # 1-based legacy indices
    lines = [f"{nr2} {nc2} {nclass} {dx_m!r} {dt!r}"] + [" ".join(str(int(a)) for a in row) for row in aspect]
    lines.append(str(len(calls)))
    lines += [f"{im} {jm} {phi} {d!r} {Lm!r} {ns}" for im, jm, phi, d, Lm, ns in calls]
    (tmp_path / "input.dat").write_text("\n".join(lines) + "\n")
    env = dict(os.environ)
    if toolchain.run_library_path:
        env["LD_LIBRARY_PATH"] = os.pathsep.join((*toolchain.run_library_path, env.get("LD_LIBRARY_PATH", "")))
    done = subprocess.run([str(exe), str(tmp_path / "input.dat"), str(tmp_path / "output.dat")], cwd=tmp_path,
                          env=env, capture_output=True, text=True, check=False, timeout=60)
    assert done.returncode == 0, done.stderr
    out_lines = (tmp_path / "output.dat").read_text().splitlines()
    assert out_lines[-1] == "SYRUP_FLOW_DISTRIB_DRIVER_COMPLETE"
    fortran = np.loadtxt(out_lines[:-1]).reshape(nr2, nc2, nclass)
    depos = np.zeros((n, nclass)); ring = np.zeros(nclass)
    L._walk_py(det, inv, law, nsteps, receiver, dx_m, dt, depos, ring)
    depos_grid = depos.reshape(nr2, nc2, nclass)
    ring_mask = ~interior.reshape(nr2, nc2)
    # interior cells: identical bins (exp of the same arguments); ring: the Fortran deposits INTO ring cells
    np.testing.assert_allclose(depos_grid[~ring_mask], fortran[~ring_mask], rtol=4e-15, atol=1e-300)
    np.testing.assert_allclose(fortran[ring_mask].sum(axis=0), ring, rtol=4e-15, atol=1e-300)
    assert det.sum() > 0 and fortran.sum() > 0
    record = {"compiler": toolchain.compiler, "sources": {s: hashlib.sha256((MAHLERAN_ROOT / s).read_bytes()).hexdigest() for s in SOURCES},
              "driver_sha256": hashlib.sha256(DRIVER.read_bytes()).hexdigest(), "calls": len(calls),
              "max_abs_difference_interior": float(np.abs(depos_grid[~ring_mask] - fortran[~ring_mask]).max())}
    (tmp_path / "record.json").write_text(json.dumps(record, indent=2))
