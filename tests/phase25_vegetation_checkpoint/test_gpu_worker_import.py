"""CPU host control of the GPU fresh worker's own-definitions loader: it must resolve THIS directory's `test_gpu_definitions.py` (the one defining `vegetated` / `day`), by path and under a unique module name, even when sibling test
directories that also contain a `test_gpu_definitions.py` come first on `sys.path`. No GPU, CuPy or GPU fixture is touched (the module's import is lazy; the release flag keeps its setup-failure semantics)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_the_worker_resolves_its_own_definitions_even_with_colliding_sibling_basenames():
    code = f"""
import sys
from pathlib import Path
here = Path({str(HERE)!r})
sys.path[:0] = [str(here)]
sys.path[:0] = [str(p.parent) for p in sorted(here.parent.glob('*/test_gpu_definitions.py')) if p.parent != here]  # colliding siblings FIRST
import importlib, importlib.util
spec = importlib.util.spec_from_file_location('veg_gpu_fresh_worker_under_test', here / 'gpu_fresh_worker.py')
w = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(here))  # the worker script's own first entry; the collision is exercised by the colliding siblings below
sys.modules['veg_gpu_fresh_worker_under_test'] = w
spec.loader.exec_module(w)
sys.path[:0] = [str(p.parent) for p in sorted(here.parent.glob('*/test_gpu_definitions.py')) if p.parent != here]  # colliding siblings FIRST again (after the worker's own path setup)
plain = importlib.import_module('test_gpu_definitions')
assert not hasattr(plain, 'vegetated'), 'the plain basename should resolve a sibling here'
mod = w.load_own_definitions()
assert Path(mod.__file__).resolve() == here / 'test_gpu_definitions.py', mod.__file__
assert hasattr(mod, 'vegetated') and hasattr(mod, 'day') and mod.__name__ == w.DEFS_MODULE_NAME
assert w.load_own_definitions() is mod and sys.modules[w.DEFS_MODULE_NAME] is mod
assert 'cupy' not in sys.modules
print('ok')
"""
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), check=False)
    assert out.stdout.strip().endswith("ok"), out.stdout[-2000:] + out.stderr[-3000:]
