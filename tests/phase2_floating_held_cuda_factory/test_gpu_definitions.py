"""Actual-GPU test DEFINITIONS for `FloatingPondVertical.from_held_context` on a CUDA H0 context (WRITTEN, NOT RUN by their author: no GPU/CuPy/NVRTC/nvidia-smi). Opt-in (SYRUP_RELEASE_GPU=1); Root runs them on GPU0.
Every context is produced through the accepted H0 API (original CUDA vertical -> `export_held_context` -> parse -> `reconstruct_held_context`): no forged context, payload or record. The factory alone is a pure law
context: no initial W/J/I exists, physical adoption and GPU file continuation are NOT qualified here. Fixtures are reused from the H0-CUDA GPU definitions (Dense four-grain and the genuine native Windowed fixture)."""
# ruff: noqa: RUF059
from __future__ import annotations

import gc
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_T = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("h0_gpu_defs", _T / "phase2_floating_held_context_cuda" / "test_gpu_definitions.py")
h0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h0)

from maple_syrup import floating_held_context as fh
from maple_syrup import floating_pond as fp
from maple_syrup import floating_soil_layers_cuda as fcu

BIG = 1 << 40
AVAIL = lambda: BIG


@pytest.fixture(scope="module")
def cp():
    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA held factory is not exercised (no claim)")
    return backend.cupy_module()



def origin(cp, tmp_path, L, pref, kind="dense", mode="explicit"):
    spec = None if pref == "none" else h0.spec_for(cp, L, 3, "cell", 2, "scalar", "zero" if pref == "disabled" else "enabled")
    if kind == "windowed":
        v, owner, _bed = h0.windowed_cuda_vertical(cp, tmp_path, mode=mode)
        return v, None
    v, owner = h0.cuda_vertical(cp, L=L, mode=mode, spec=spec)
    return v, spec


def export_context(cp, v, spec):
    rec = fh.export_held_context(v, spec, available_bytes_fn=AVAIL)
    return fh.reconstruct_held_context(rec, h0.expected_of(v), available_bytes_fn=AVAIL)


@pytest.mark.parametrize("L", [1, 3, 65])
@pytest.mark.parametrize("pref", ["none", "enabled", "disabled"])
@pytest.mark.parametrize("mode", ["explicit", "texture_ks"])
def test_dense_origin_factory_vertical_has_the_original_identity_models_and_no_initial_payload(cp, tmp_path, L, pref, mode):
    if pref != "none" and L == 65:
        pytest.skip("65-layer preferential definitions are exercised by the H0 suite; keep this matrix bounded")
    v, spec = origin(cp, tmp_path, L, pref, mode=mode)
    ctx = export_context(cp, v, spec)
    h = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL)
    assert h.backend == "cuda" and h.identity == v.identity and h._owner is None and h._held is ctx and h._initial is None and h._registry == {} and h._provisional is None
    t, o = h.held_template(), v.held_template()
    assert (t.column.sha256, t.matrix_op.sha256, None if t.preferential is None else t.preferential.sha256) == (o.column.sha256, o.matrix_op.sha256, None if o.preferential is None else o.preferential.sha256)
    assert t.generation == o.generation and t.source_witness == o.source_witness and t.config_sha256 == o.config_sha256 and (t.ks_m_s is None) == (o.ks_m_s is None)
    # the water-free models are the owner constructor's, except the sealed initial depth copy (D0 is referenced)
    assert (h.stage_step_bytes, h.exchange_bytes, h.payload_bytes, h.step_bytes) == (v.stage_step_bytes, v.exchange_bytes, v.payload_bytes, v.step_bytes)
    assert h.retained_bytes == v.retained_bytes - v.n_cells * 8 and h.construction_bytes == v.construction_bytes - v.n_cells * 8
    assert np.array_equal(h._active_host(), v._active_host()) and np.array_equal(h._edges_host, v._edges_host)
    for call in (h.initial_soil_water, lambda: h.check_initial(None, None, 0.0)):
        with pytest.raises(fp.FloatingPondError, match="no initial payload"):
            call()
    assert float(h._zero_added.sum()) == 0.0 and h._zero_added.shape == (L, h.n_cells) and h._zero_pet.shape == (h.n_cells,)


@pytest.mark.parametrize("mode", ["explicit", "texture_ks"])
def test_genuine_windowed_origin_factory_vertical(cp, tmp_path, mode):
    v, owner, wbed = h0.windowed_cuda_vertical(cp, tmp_path, mode=mode)
    ctx = export_context(cp, v, None)
    h = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL)
    assert type(wbed.voxel_column).__name__ == "WindowedVoxelColumnState" and h.identity == v.identity and h._initial is None


def test_construction_shares_the_context_with_zero_counted_transfers_and_only_the_two_zero_rows(cp, tmp_path):
    from maple.core import backend as mb

    v, spec = origin(cp, tmp_path, 3, "enabled")
    ctx = export_context(cp, v, spec)
    sizes = []

    class Hook:
        def __init__(self):
            self.pool = cp.get_default_memory_pool()

        def __call__(self, size):
            ptr = self.pool.malloc(size)
            sizes.append((int(size), int(ptr.mem.size)))
            return ptr

    mb.reset_transfer_counters()
    with cp.cuda.using_allocator(Hook()):
        h = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL)
    c = mb.read_transfer_counters()
    assert (c.host_to_device, c.device_to_host, c.host_to_device_bytes, c.device_to_host_bytes) == (0, 0, 0, 0)  # the context is shared: no D2H of the mask, no H2D of D0, no cold payload copy
    L, N = h.n_layers, h.n_cells
    assert sorted(s for s, _ in sizes) == sorted([L * N * 8, N * 8])  # ONLY the zero rows are allocated
    assert h._edges_host is ctx.edges_m and h._host_active is ctx.host_active  # referenced, not copied


def test_gate_before_readers_two_pools_zero_need_minus_one_exact_and_reserve_once(cp, tmp_path, monkeypatch):
    v, spec = origin(cp, tmp_path, 3, "enabled")
    ctx = export_context(cp, v, spec)
    probe = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL)
    dev_need, host_need = probe.construction_bytes, probe.cold_host_bytes + fcu.PY_ALLOWANCE_BYTES
    order = []
    real_gate = fcu.gate_device
    monkeypatch.setattr(fcu, "gate_device", lambda *a, **k: order.append("gate") or real_gate(*a, **k))

    def run(dev_a, host_a, **kw):
        return fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=lambda: order.append("dev") or dev_a, host_available_bytes_fn=lambda: order.append("host") or host_a, **kw)

    allocs = []
    real_zeros = cp.zeros
    monkeypatch.setattr(cp, "zeros", lambda *a, **k: allocs.append(1) or real_zeros(*a, **k))
    for dev_a, host_a in ((0, BIG), (dev_need - 1, BIG), (BIG, 0), (BIG, host_need - 1)):
        order.clear()
        with pytest.raises((fp.FloatingPondError, MemoryError, Exception)):
            run(dev_a, host_a)
        assert order[0] == "gate"  # the CURRENT device is checked before any reader
    assert allocs == []  # refused BEFORE any owned array
    run(dev_need, host_need)  # exact needs of both pools
    with pytest.raises(Exception):  # noqa: B017 - FloatingMemoryError from fcu.admit
        run(dev_need, host_need, reserve_bytes=1)  # each reserve counted once, in its own pool
    run(dev_need + 3, host_need + 5, reserve_bytes=3, host_reserve_bytes=5)
    with pytest.raises(Exception):  # noqa: B017
        run(dev_need + 3, host_need + 4, reserve_bytes=3, host_reserve_bytes=5)


def test_zero_budget_and_foreign_context_refuse_before_owned_work(cp, tmp_path):
    v, spec = origin(cp, tmp_path, 3, "none")
    ctx = export_context(cp, v, spec)
    with pytest.raises(Exception):  # noqa: B017
        fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL, memory_budget_bytes=0)
    with pytest.raises(Exception):  # noqa: B017
        fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL, host_memory_budget_bytes=0)
    with pytest.raises(fh.HeldContextError):
        fp.FloatingPondVertical.from_held_context(object(), available_bytes_fn=AVAIL)
    if cp.cuda.runtime.getDeviceCount() > 1:
        pytest.skip("a second device exists: the wrong-device branch must be exercised by Root's two-device run; untestable under a one-UUID mask")


def test_moderate_n50003_true_allocator_peak_and_released_plateau(cp, tmp_path):
    import importlib.util as iu

    bed = h0.b.bed_from(h0.b.fx.stratified_dense(shallow_cells=(), ny=1, nx=50003))
    v, owner = h0.cuda_vertical(cp, L=3, mode="texture_ks", bed=bed)
    ctx = export_context(cp, v, None)
    pool = cp.get_default_memory_pool()
    fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL)  # warm

    class Peak:
        def __init__(self):
            self.base, self.peak, self.rounding = int(pool.used_bytes()), 0, 0

        def __call__(self, size):
            ptr = pool.malloc(size)
            self.peak = max(self.peak, int(pool.used_bytes()) - self.base)
            self.rounding += int(ptr.mem.size) - int(size)
            return ptr

    used = []
    for _ in range(5):
        gc.collect()
        cp.cuda.Device().synchronize()
        hook = Peak()
        with cp.cuda.using_allocator(hook):
            h = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL)
        model = h.retained_bytes  # device allocations of the held constructor are exactly the retained zero rows
        assert hook.peak <= model + hook.rounding, (hook.peak, model, hook.rounding)  # TRUE transient peak vs the declared model + the ACTUAL rounding
        del h
        gc.collect()
        cp.cuda.Device().synchronize()
        used.append(int(pool.used_bytes()))
    assert max(used) - min(used) <= hook.rounding  # released plateau
    del iu
