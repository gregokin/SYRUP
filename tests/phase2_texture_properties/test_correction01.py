"""Correction C1: owner-device guard of detached(), counted 32-byte packet read, standalone planner layout validation, CUDA grid limit. CPU lane (metadata mocks / CuPy double); no GPU."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import BIG, call
from hostcuda import FakeArr, FakeCupy, build, fake, have_toolchain

from maple_syrup import soil_texture_properties as tp
from maple_syrup import soil_texture_properties_cuda as cu
from maple_syrup import soil_water_remap_cuda as rc

LAY = {k: "cell" for k in tp.INPUTS}


def test_planner_validates_optional_layouts():
    for bad in ("not_a_layout", None, 3, ""):
        with pytest.raises(tp.TexturePropertyError, match="air_entry_kpa_override"):
            tp.estimate_property_memory(2, 5, layouts={**{k: "shared" for k in tp.INPUTS}, "air_entry_kpa_override": bad})
    with pytest.raises(tp.TexturePropertyError, match="mandatory"):
        tp.estimate_property_memory(2, 5, layouts={k: "cell" for k in tp.INPUTS[:4]})
    ok = tp.estimate_property_memory(2, 5, layouts={**{k: "shared" for k in tp.INPUTS}, "air_entry_kpa_override": "cell"})
    assert ok.output_layout == "cell"


def test_cuda_grid_limit_before_allocation():
    lim = (2**31 - 1) * tp.CUDA_BLOCK
    assert tp.MAX_CUDA_GRID_CELLS == lim
    tp.estimate_property_memory(1, lim, layouts=LAY, backend="cuda")  # exactly the limit is planned
    with pytest.raises(tp.TexturePropertyError, match="launches at most"):
        tp.estimate_property_memory(1, lim + 1, layouts=LAY, backend="cuda")
    shared = {k: "shared" for k in tp.INPUTS}
    big = tp.estimate_property_memory(3, lim * 50, layouts=shared, backend="cuda")  # shared-only: one thread, logical N never expands
    assert big.output_bytes == 3 * 8 * 3
    tp.estimate_property_memory(1, lim + 1, layouts=LAY, backend="numpy")  # the CPU planner has no grid limit
    with pytest.raises(tp.TexturePropertyError, match="packed|launches"):
        tp.estimate_property_memory(2**20, 2**31, layouts=LAY, backend="cuda")


@pytest.mark.skipif(not have_toolchain(), reason="no g++")
class TestDeviceGuardAndCounter:
    @pytest.fixture()
    def cupy(self, monkeypatch):
        cp = FakeCupy(build(cu.render_source()))
        self.dev = {"id": 0}
        monkeypatch.setattr(tp, "_namespace_type", lambda b: FakeArr)
        monkeypatch.setattr(rc, "require_cuda", lambda: cp)
        monkeypatch.setattr(rc, "current_device_id", lambda c: self.dev["id"])
        with cu._LOCK:
            saved = dict(cu._KERNELS)
            cu._KERNELS.clear()
        yield cp
        with cu._LOCK:
            cu._KERNELS.clear()
            cu._KERNELS.update(saved)

    def result(self, cp):
        return call(fake(np.full((1, 2), 0.4)), 0.4, 0.2, 2.5, 1400.0, L=1, N=2, backend="cuda", available_bytes_fn=lambda: 1 << 40, host_available_bytes_fn=lambda: 1 << 40)

    def test_detached_is_bound_to_the_owner_device(self, cupy):
        r = self.result(cupy)
        assert r.device_id == 0
        calls = []
        self.dev["id"] = 1  # metadata mock: the current device changed; no real GPU is selected
        with pytest.raises(tp.TexturePropertyError, match="device 0 .* current device is 1"):
            r.detached("theta_saturation", memory_budget_bytes=BIG, available_bytes_fn=lambda: calls.append(1) or (1 << 40))
        assert not calls  # the availability reader was never consulted
        self.dev["id"] = 0
        d = r.detached("theta_saturation", memory_budget_bytes=BIG, available_bytes_fn=lambda: 1 << 40)
        assert np.asarray(d).shape == (1, 2)

    def test_numpy_result_has_no_device_guard(self):
        r = call(backend="numpy")
        assert r.detached("theta_saturation", memory_budget_bytes=BIG).shape == (1,)

    def test_packet_read_uses_the_counted_to_host_once(self, cupy, monkeypatch):
        import maple.core.backend as mb

        seen = []
        real = mb.to_host
        monkeypatch.setattr(mb, "to_host", lambda a: seen.append((tuple(a.shape), a.dtype.name, int(a.nbytes))) or real(a))
        self.result(cupy)
        assert seen == [((tp.DIAG_WORDS,), "uint64", 32)]  # exactly one 32-byte validation packet; the identity chunks are separate `.get(out=)` reads
