"""Test harness: build and run the ORIGINAL MAHLERAN `route_water` routine.

The unmodified sources `shared_data.f90`, `route_water.for` and
`ff_type8.for` (linked, not selected) are compiled from the reference tree
with `-std=legacy -ffixed-line-length-none -fcheck=all` into a scratch
build directory, together with `benchmarks/phase4/reference_driver.f90`,
which contains no routing equations. Each run gets its own working
directory: route_water can write `fort.51` on one legacy path, and no file
is ever written into the reference tree.

This is an executed-ROUTINE comparison for one routing step at a time. It
is not a MAHLERAN model run: no infiltration, sediment, rainfall or storm
loop is executed.

Toolchain (all optional; the tests skip when no compiler is found):
  MAPLE_SYRUP_GFORTRAN          compiler path (default: gfortran on PATH)
  MAPLE_SYRUP_GFORTRAN_FLAGS    extra flags for every compile and the link
  MAPLE_SYRUP_GFORTRAN_LDFLAGS  extra flags for the link only
  MAPLE_SYRUP_GFORTRAN_RUNPATH  directories prepended to LD_LIBRARY_PATH to run
  MAPLE_SYRUP_MAHLERAN_ROOT     reference tree (default /home/okin/MAHLERAN)
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DRIVER_PATH = REPO / "benchmarks" / "phase4" / "reference_driver.f90"
MAHLERAN_ROOT = Path(os.environ.get("MAPLE_SYRUP_MAHLERAN_ROOT", "/home/okin/MAHLERAN"))
COMPLETION_MARKER = "SYRUP_ROUTE_WATER_DRIVER_COMPLETE"

# SHA-256 recorded by Codex (agent_handoffs/tasks/phase4a_hydraulic_design/
# reference_sources.json). A difference means the reference changed.
REFERENCE_SOURCES = {
    "src/Program_Control/shared_data.f90": "0d05a3b38bad24b40833c530900dc1e061cd90cef187b4e6bce2e1e1d233d36f",
    "src/Subroutines_Water/route_water.for": "cd3c906ec35f181d108ed09fd3636be72578e759192a9f08a61b6f3395462188",
    "src/Subroutines_Water/ff_type8.for": "0dc87ac00acc28a7b04b4e4a57e4251569a9dd127db06880f3df3fbe5b50b796",
    "src/Subroutines_Water/infilt.for": "f67ae7a741be20d80ce041a4510cf65abe66993409e2b6974b857f713f279b8e",
    "src/Subroutines_Water/accumulate_flow.for": "66f0ae6f0a9e4075c939caf17b8c6b613d69cc60399e10081336ef79dcf08d96",
    "src/Subroutines_Water/update_water_flow.for": "fca3dd601611333865df996d7d73bd62da0592810726073ccd61945ae333ed92",
    "src/Subroutines_In_out/topog_attrib.for": "bd44539b98ff213efd68b7c6b79df13f39867a9187ed377a7031a1f7d2ce57f5",
}
# Compiled and linked, in dependency order (shared_data provides the module).
COMPILED_SOURCES = (
    "src/Program_Control/shared_data.f90",
    "src/Subroutines_Water/route_water.for",
    "src/Subroutines_Water/ff_type8.for",
)
ORIGINAL_FLAGS = ("-std=legacy", "-ffixed-line-length-none", "-fcheck=all")
# The driver USEs the original shared_data module, whose initialized COMMON
# variables (viscosity, spq, sma, smb, radius) are rejected under -std=f2008
# (Codex build check). The driver is therefore compiled to the same legacy
# standard as the originals; gfortran still accepts its F2008 statements.
DRIVER_FLAGS = ("-std=legacy", "-fcheck=all")
# Directories whose listings must not change (no .mod, .o or fort.* output).
WATCHED_DIRS = ("", "src/Program_Control", "src/Subroutines_Water", "src/Subroutines_In_out")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_hashes(root: Path = MAHLERAN_ROOT) -> dict[str, str]:
    return {rel: sha256_file(root / rel) for rel in REFERENCE_SOURCES}


def watched_listing(root: Path = MAHLERAN_ROOT) -> dict[str, list[str]]:
    return {rel: sorted(p.name for p in (root / rel).iterdir()) for rel in WATCHED_DIRS}


@dataclass(frozen=True)
class Toolchain:
    compiler: str
    flags: tuple[str, ...]
    link_flags: tuple[str, ...]
    run_library_path: tuple[str, ...]


def locate_toolchain() -> Toolchain | None:
    compiler = os.environ.get("MAPLE_SYRUP_GFORTRAN") or shutil.which("gfortran")
    if not compiler or not Path(compiler).is_file():
        return None
    runpath = os.environ.get("MAPLE_SYRUP_GFORTRAN_RUNPATH", "")
    return Toolchain(
        compiler=str(compiler),
        flags=tuple(shlex.split(os.environ.get("MAPLE_SYRUP_GFORTRAN_FLAGS", ""))),
        link_flags=tuple(shlex.split(os.environ.get("MAPLE_SYRUP_GFORTRAN_LDFLAGS", ""))),
        run_library_path=tuple(p for p in runpath.split(os.pathsep) if p),
    )


@dataclass(frozen=True)
class Build:
    executable: Path
    commands: tuple[tuple[str, ...], ...]
    compiler_version: str
    source_sha256: dict[str, str]
    driver_sha256: str
    toolchain: Toolchain

    def record(self) -> dict[str, Any]:
        return {
            "executable": str(self.executable),
            "commands": [list(c) for c in self.commands],
            "compiler_version": self.compiler_version,
            "source_sha256": self.source_sha256,
            "driver_sha256": self.driver_sha256,
        }


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=300, check=False)


def build_reference(toolchain: Toolchain, build_dir: Path, root: Path = MAHLERAN_ROOT) -> Build:
    """Compile the unchanged originals and the driver into `build_dir`
    (created empty). Raises RuntimeError with the compiler output on failure,
    and if any reference source hash differs from REFERENCE_SOURCES."""
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=False)
    hashes = reference_hashes(root)
    changed = {rel: h for rel, h in hashes.items() if h != REFERENCE_SOURCES[rel]}
    if changed:
        raise RuntimeError(f"reference sources differ from the recorded revision: {changed}")
    base = [toolchain.compiler, *toolchain.flags, f"-J{build_dir}", f"-I{build_dir}"]
    commands, objects = [], []
    for rel in COMPILED_SOURCES:
        obj = build_dir / (Path(rel).stem + ".o")
        commands.append([*base, *ORIGINAL_FLAGS, "-c", str(root / rel), "-o", str(obj)])
        objects.append(str(obj))
    driver_obj = build_dir / "reference_driver.o"
    commands.append([*base, *DRIVER_FLAGS, "-c", str(DRIVER_PATH), "-o", str(driver_obj)])
    executable = build_dir / "syrup_route_water_driver"
    commands.append([toolchain.compiler, *toolchain.flags, *toolchain.link_flags, str(driver_obj), *objects,
                     "-o", str(executable)])
    for command in commands:
        done = _run(command, build_dir)
        if done.returncode != 0:
            raise RuntimeError(f"command failed ({done.returncode}): {shlex.join(command)}\n"
                               f"stdout:\n{done.stdout}\nstderr:\n{done.stderr}")
    version = _run([toolchain.compiler, "--version"], build_dir).stdout.splitlines()
    return Build(
        executable=executable,
        commands=tuple(tuple(c) for c in commands),
        compiler_version=version[0] if version else "",
        source_sha256=hashes,
        driver_sha256=sha256_file(DRIVER_PATH),
        toolchain=toolchain,
    )


# --- legacy one-step state ---------------------------------------------------------
@dataclass(frozen=True)
class LegacyStep:
    """Full legacy grid `(nr2, nc2)`, north-first, 1-based = array index + 1,
    millimetre units. `order` rows are (i, j, level), upstream first."""

    dt_s: float
    dx_mm: float
    order: np.ndarray
    aspect: np.ndarray
    rmask: np.ndarray
    slope: np.ndarray
    ff: np.ndarray
    d1_mm: np.ndarray
    q1_mm2_s: np.ndarray
    qin1_mm2_s: np.ndarray
    excess_mm_s: np.ndarray


def legacy_index(ny: int, r: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MAPLE interior (r, c), row 0 = south -> legacy 1-based (i, j)."""
    return ny + 1 - np.asarray(r), np.asarray(c) + 2


def _to_full(interior: np.ndarray, fill: float) -> np.ndarray:
    """MAPLE interior (south-first) -> legacy full grid (north-first) with a ring."""
    ny, nx = interior.shape
    full = np.full((ny + 2, nx + 2), fill, dtype=np.float64)
    full[1:-1, 1:-1] = interior[::-1]
    return full


def legacy_step(graph, *, old_flow_depth_m, depth_start_m, old_discharge_m2_s, old_inflow_m2_s, dt_s,
                ring_export_full=None) -> LegacyStep:
    """Legacy inputs equivalent to a SYRUP step: d(1) = old flow depth,
    excess = (depth_start - d(1)) / dt, q(1) = old discharge, qin(1) = the
    supplied old inflow (coherent donor sum, or a stale value). Host NumPy.
    `ring_export_full` (south-first full bool grid) sets ring rmask = -9999
    where True; interior inactive cells get rmask -1."""
    ny, nx = graph.shape
    act = graph.active
    rmask_interior = np.where(act, 1.0, -1.0)
    rmask = _to_full(rmask_interior, 1.0)
    if ring_export_full is not None:
        ring = np.zeros((ny + 2, nx + 2), dtype=bool)
        ring[[0, -1], :] = True
        ring[:, [0, -1]] = True
        export_nf = np.asarray(ring_export_full, dtype=bool)[::-1]
        rmask = np.where(ring & export_nf, -9999.0, rmask)
    aspect = np.zeros((ny + 2, nx + 2), dtype=np.int64)
    aspect[1:-1, 1:-1] = graph.aspect[::-1]
    flat = graph.level_order_host
    r, c = np.divmod(flat, nx)
    i, j = legacy_index(ny, r, c)
    order = np.stack([i, j, graph.level.reshape(-1)[flat]], axis=1)
    d1 = np.asarray(old_flow_depth_m, dtype=np.float64)
    return LegacyStep(
        dt_s=float(dt_s),
        dx_mm=graph.dx_m * 1000.0,
        order=order,
        aspect=aspect,
        rmask=rmask,
        slope=_to_full(graph.slope, 0.0),
        ff=_to_full(np.where(act, graph.friction_factor, 1.0), 1.0),
        d1_mm=_to_full(d1 * 1000.0, 0.0),
        q1_mm2_s=_to_full(np.asarray(old_discharge_m2_s) * 1.0e6, 0.0),
        qin1_mm2_s=_to_full(np.asarray(old_inflow_m2_s) * 1.0e6, 0.0),
        excess_mm_s=_to_full((np.asarray(depth_start_m) - d1) * 1000.0 / float(dt_s), 0.0),
    )


def write_input(path: Path, step: LegacyStep) -> None:
    nr2, nc2 = step.aspect.shape
    lines = [f"{nr2} {nc2} {step.order.shape[0]} 5 1", f"{step.dt_s:.17e} {step.dx_mm:.17e}"]
    lines += [f"{int(i)} {int(j)} {int(lev)}" for i, j, lev in step.order]
    for i in range(nr2):
        for j in range(nc2):
            values = (step.rmask[i, j], step.slope[i, j], step.ff[i, j], step.d1_mm[i, j],
                      step.q1_mm2_s[i, j], step.qin1_mm2_s[i, j], step.excess_mm_s[i, j])
            lines.append(f"{int(step.aspect[i, j])} " + " ".join(f"{float(v):.17e}" for v in values))
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


@dataclass(frozen=True)
class RunResult:
    completed: bool
    returncode: int
    stdout: str
    stderr: str
    real_bits: tuple[int, int] | None
    d2_mm: np.ndarray | None
    q2_mm2_s: np.ndarray | None
    qin2_mm2_s: np.ndarray | None
    v_mm_s: np.ndarray | None
    scratch_files: tuple[str, ...]


def run_route_water(build: Build, step: LegacyStep, run_dir: Path) -> RunResult:
    """Run one step in a fresh `run_dir`. Outputs are parsed only when the
    completion marker is the last line; otherwise `completed` is False and
    the arrays are None (e.g. a legacy STOP)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    in_path, out_path = run_dir / "route_water_in.txt", run_dir / "route_water_out.txt"
    write_input(in_path, step)
    env = dict(os.environ)
    if build.toolchain.run_library_path:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*build.toolchain.run_library_path, *filter(None, [env.get("LD_LIBRARY_PATH")])])
    done = subprocess.run([str(build.executable), str(in_path), str(out_path)], cwd=run_dir,
                          capture_output=True, text=True, timeout=300, check=False, env=env)
    scratch = tuple(sorted(p.name for p in run_dir.iterdir() if p.name not in (in_path.name, out_path.name)))
    lines = out_path.read_text(encoding="ascii").splitlines() if out_path.is_file() else []
    if done.returncode != 0 or not lines or lines[-1].strip() != COMPLETION_MARKER:
        return RunResult(False, done.returncode, done.stdout, done.stderr, None, None, None, None, None, scratch)
    nr2, nc2 = step.aspect.shape
    bits_tokens = lines[0].split()
    grid_tokens = lines[1].split()
    if bits_tokens[0] != "REAL_STORAGE_BITS_DT_DX" or grid_tokens[0] != "GRID" \
            or (int(grid_tokens[1]), int(grid_tokens[2])) != (nr2, nc2):
        raise RuntimeError(f"unexpected driver output header: {lines[:2]}")
    arrays = np.full((4, nr2, nc2), np.nan)
    seen = np.zeros((nr2, nc2), dtype=bool)
    for line in lines[2:-1]:
        tokens = line.split()
        i, j = int(tokens[0]), int(tokens[1])
        arrays[:, i - 1, j - 1] = [float(t) for t in tokens[2:6]]
        seen[i - 1, j - 1] = True
    if not seen.all():
        raise RuntimeError("driver output is missing cells")
    return RunResult(True, done.returncode, done.stdout, done.stderr,
                     (int(bits_tokens[1]), int(bits_tokens[2])), *arrays, scratch)


def interior_si(full_north_first: np.ndarray, scale: float) -> np.ndarray:
    """Legacy full grid (north-first) -> MAPLE interior (south-first) x scale."""
    return np.ascontiguousarray(full_north_first[1:-1, 1:-1][::-1]) * scale
