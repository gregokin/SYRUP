"""Original sediment equation harness; flow_distrib records its input only."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse the existing isolated reference-toolchain helper even when this
# directory alone is selected by pytest. No reference tree is imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase4"))
from fortran_reference import locate_toolchain, sha256_file

ROOT = Path('/home/okin/MAHLERAN')
REPO = Path(__file__).resolve().parents[2]
SOURCES = ['src/Program_Control/shared_data.f90', 'src/Program_Control/parameters_from_xml.f90'] + [
    'src/Subroutines_Sediment/' + name + '.for' for name in
    ['raindrop_detachment', 'flow_detachment', 'diffuse_flow_transport', 'conc_flow_transport', 'suspended_transport']]
DRIVER = REPO / 'benchmarks/phase5/reference_physics_driver.f90'
MANIFEST = REPO / 'benchmarks/phase5/reference_sources.json'


def build(directory):
    toolchain = locate_toolchain()
    if toolchain is None:
        raise RuntimeError('No Fortran compiler configured')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    expected = json.loads(MANIFEST.read_text())
    before = {rel: sha256_file(ROOT / rel) for rel in SOURCES}
    if before != expected:
        raise RuntimeError('Original sediment sources changed since audit')
    commands, objects = [], []
    # Compile driver first so its recording module exists; only shared_data
    # and parameters_from_xml must precede it.
    sources = [ROOT / rel for rel in SOURCES[:2]] + [DRIVER] + [ROOT / rel for rel in SOURCES[2:]]
    for source in sources:
        obj = directory / (source.stem + '.o')
        cmd = [toolchain.compiler, *toolchain.flags, '-std=legacy', '-fcheck=all',
               '-O0', '-fno-fast-math', '-ffree-line-length-none', '-ffixed-line-length-none', f'-J{directory}',
               f'-I{directory}', '-c', str(source), '-o', str(obj)]
        commands.append(cmd)
        subprocess.run(cmd, cwd=directory, check=True, capture_output=True, text=True, timeout=60)
        objects.append(str(obj))
    exe = directory / 'reference_physics'
    cmd = [toolchain.compiler, *toolchain.flags, *objects, *toolchain.link_flags, '-o', str(exe)]
    commands.append(cmd)
    subprocess.run(cmd, cwd=directory, check=True, capture_output=True, text=True, timeout=60)
    assert before == {rel: sha256_file(ROOT / rel) for rel in SOURCES}
    version = subprocess.run([toolchain.compiler, '--version'], capture_output=True, text=True,
                             timeout=30, check=True).stdout.splitlines()[0]
    record = {'compiler_version': version, 'xml_sha256': sha256_file(ROOT / 'mahleran_input.xml'),
              'source_sha256': before, 'driver_sha256': sha256_file(DRIVER), 'commands': commands,
              'scope': 'Original equation calls; flow_distrib argument capture, no routing benchmark'}
    (directory / 'build_record.json').write_text(json.dumps(record, indent=2) + '\n')
    return exe


@dataclass(frozen=True)
class EquationResult:
    """Rates kg/m2/s (n,6); distance m and speed m/s (n,3,6).

    Law order diffuse, concentrated, suspended. Inapplicable laws have
    both distance and speed NaN and law_applies=False, never stale values.
    Zero length is a valid original underflow, not a missing law.
    """
    rain_rate: np.ndarray
    flow_rate: np.ndarray
    distance: np.ndarray
    speed: np.ndarray
    law_applies: np.ndarray


def run(executable, cases, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    data = np.asarray(cases, dtype=np.float64)
    np.savetxt(directory / 'input.dat', data, header=str(len(data)), comments='',
               fmt=['%.17e'] * 6 + ['%d'] + ['%.17e'] * 6)
    toolchain = locate_toolchain()
    env = dict(os.environ)
    if toolchain.run_library_path:
        env['LD_LIBRARY_PATH'] = os.pathsep.join((*toolchain.run_library_path, env.get('LD_LIBRARY_PATH', '')))
    done = subprocess.run([str(executable), str(directory.resolve() / 'input.dat'),
                           str(directory.resolve() / 'output.dat')], cwd=directory,
                          env=env, capture_output=True, text=True, timeout=60, check=False)
    (directory / 'stdout.txt').write_text(done.stdout)
    (directory / 'stderr.txt').write_text(done.stderr)
    done.check_returncode()
    lines = (directory / 'output.dat').read_text().splitlines()
    if lines[-1] != 'SYRUP_SEDIMENT_PHYSICS_COMPLETE':
        raise RuntimeError('Legacy STOP or incomplete sediment equation driver')
    values = np.loadtxt(lines[:-1], ndmin=2)
    if values.shape != (len(data), 48) or not np.isfinite(values).all():
        raise RuntimeError(f'Invalid original-equation output: shape {values.shape}, '
                           f'nonfinite indices {np.argwhere(~np.isfinite(values)).tolist()}')
    distance = values[:, 12:30].reshape(-1, 3, 6)
    speed = values[:, 30:48].reshape(-1, 3, 6)
    applies = distance >= 0
    if not np.array_equal(applies, speed >= 0):
        raise RuntimeError('Original law left inconsistent distance/speed sentinels')
    return EquationResult(values[:, :6], values[:, 6:12],
                          np.where(applies, distance, np.nan), np.where(applies, speed, np.nan), applies)
