"""G-F4 correction01 actual-GPU SAVE admission DEFINITIONS (WRITTEN, NOT RUN: no GPU/CuPy/NVRTC/nvidia-smi was touched; Root runs them with SYRUP_RELEASE_GPU=1 on a free device). They reuse the qualified fixtures and fixtures-in-module of
`test_gpu_definitions.py` (Dense and genuine native Windowed originals, the static recipe, the helpers)."""
# ruff: noqa: RUF059
from __future__ import annotations

import importlib.util
import os
import tracemalloc
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SYRUP_RELEASE_GPU") != "1", reason="GPU definitions: set SYRUP_RELEASE_GPU=1 on the hardware qualification host (free device)")

_D = Path(__file__).resolve().parent / "test_gpu_definitions.py"
_spec = importlib.util.spec_from_file_location("ctx_gpu_defs_for_save", _D)
defs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(defs)
ccc, fcu, IDENT, BIG, MEM = defs.ccc, defs.fcu, defs.IDENT, defs.BIG, defs.MEM


@pytest.fixture(scope="module")
def cp():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("no CUDA device")
    return backend.cupy_module()


@pytest.mark.parametrize("windowed", [False, True])
def test_save_budget_zero_and_two_pool_need_minus_one_exact_reserve_once_before_the_actual_h0_export(cp, tmp_path, windowed):
    tmp = tmp_path
    ev, st, vert = defs.original(cp, tmp / "o", windowed)
    n = ccc.save_need_bytes_cuda(st, ev, IDENT)
    dev_ok, host_ok = n["device_bytes"], n["host_bytes"] + fcu.PY_ALLOWANCE_BYTES
    entered = []
    real = ccc.fhc.export_held_context_cuda
    ccc.fhc.export_held_context_cuda = lambda *a, **k: (entered.append(1), real(*a, **k))[1]
    try:
        cases = ({"memory_budget_bytes": 0}, {"host_memory_budget_bytes": 0}, {"available_bytes_fn": lambda: dev_ok - 1}, {"host_available_bytes_fn": lambda: host_ok - 1},
                 {"available_bytes_fn": lambda: dev_ok, "reserve_bytes": 1}, {"host_available_bytes_fn": lambda: host_ok, "host_reserve_bytes": 1})
        for i, kw in enumerate(cases):
            kw = {"available_bytes_fn": lambda: BIG, "host_available_bytes_fn": lambda: BIG, **kw}
            with pytest.raises(MEM):
                ccc.save_floating_context_checkpoint_cuda(tmp / f"bad{i}", st, ev, IDENT, **kw)
            assert entered == [] and not (tmp / f"bad{i}").exists() and not list(tmp.glob(f".bad{i}.partial-*"))  # before any export / array / hash / staging
        tracemalloc.start()
        ccc.save_floating_context_checkpoint_cuda(tmp / "exact", st, ev, IDENT, available_bytes_fn=lambda: dev_ok, host_available_bytes_fn=lambda: host_ok)  # exactly the published need saves
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    finally:
        ccc.fhc.export_held_context_cuda = real
    assert entered and (tmp / "exact").is_dir() and peak <= n["host_bytes"] + fcu.PY_ALLOWANCE_BYTES  # traced HOST allocations stay inside the declared host need (device peaks: Root's allocator probe)
    # the container loads exactly at its published load need (fresh static factory: no original bed/owner/probe)
    d = defs.dev_id(cp)
    ln = ccc.context_load_need_bytes_cuda(tmp / "exact", IDENT, expected_device_id=d)
    ccc.load_floating_context_checkpoint_cuda(tmp / "exact", defs.gsc.factory(windowed), IDENT, expected_device_id=d, available_bytes_fn=lambda: ln.device_total_bytes, host_available_bytes_fn=lambda: ln.host_total_bytes)
