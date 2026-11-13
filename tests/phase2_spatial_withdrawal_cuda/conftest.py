"""S2-B1.c spatial CUDA withdrawal/availability suite (author-written; CPU lane = host-compiled source execution + plumbing doubles, NOT hardware evidence). Builders are loaded from the sibling suites."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, TESTS / rel / "conftest.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


a2 = _load("a2_builders_b1c", "phase2_spatial_water")
a3 = _load("a3_builders_b1c", "phase2_spatial_numba")

import hostb1c  # noqa: E402
import hostgpu  # noqa: E402

from maple_syrup import spatial_soil_layers as sp  # noqa: E402
from maple_syrup import spatial_transpiration as sw  # noqa: E402
from maple_syrup import spatial_transpiration_cuda as tc  # noqa: E402
from maple_syrup import subsurface as ss  # noqa: E402
from maple_syrup import subsurface_transpiration as st  # noqa: E402

BIG, AREA = a2.BIG, a2.AREA
needs_host = pytest.mark.skipif(not hostgpu.have_toolchain(), reason="no C++20 toolchain with <barrier>: the host execution of the CUDA source is not run (no claim)")


@pytest.fixture(scope="session")
def so():
    if not hostgpu.have_toolchain():
        pytest.skip("no C++20 toolchain")
    return hostb1c.build(tc.render_source())


@pytest.fixture()
def cupy(so, monkeypatch):
    """A fresh plumbing double wired into the spatial modules (CuPy probe, device id, device reading); the kernel cache is cleared before and after."""
    cp = hostb1c.FakeCupy(so)

    class RC:
        current = 0

        @staticmethod
        def current_device_id(c):
            return RC.current

        @staticmethod
        def default_device_available_bytes_fn(c):
            return lambda: 1 << 40

    monkeypatch.setattr(sp, "_require_cuda", lambda: cp)
    monkeypatch.setattr(sp, "_cuda_module", lambda: RC)
    with tc._LOCK:
        saved = dict(tc._FUNCTIONS)
        tc._FUNCTIONS.clear()
    cp.rc = RC
    yield cp
    with tc._LOCK:
        tc._FUNCTIONS.clear()
        tc._FUNCTIONS.update(saved)


def device_owner(np_owner, device=0):
    """PLUMBING-DOUBLE owner: the retained maps of a prepared numpy owner held as fake device arrays, tagged backend='cuda' (built through the owner's private construction context purely so that the REAL launcher/API
    code can run on host). Evidence of API plumbing and kernel text only."""
    maps = {k: hostgpu.fake(v, device) for k, v in np_owner._maps.items()}
    mask = hostgpu.fake(np_owner._mask.astype(np.bool_), device)
    with sp._constructing():
        return sp.SpatialSoilColumn("cuda", np_owner.n_layers, np_owner.n_cells, np_owner.n_active, np_owner.cell_area_m2, np_owner.diffusion_mode, np_owner.bottom_drainage, np_owner.gravity_drainage,
                                    np_owner.diffusion_coefficient, np_owner.theta_evaporation_stress, np_owner.evaporation_factor, np_owner.layouts, np_owner.parameters_sha256, np_owner.mask_sha256,
                                    np_owner.estimate, device, maps, mask)


def fx(a, device=0):
    return hostgpu.fake(a, device)


def bits(a):
    return np.ascontiguousarray(np.asarray(a)).view(np.uint64)


@pytest.fixture(params=[1, 7, 127, 128, 129, 257])
def ncells(request):
    return request.param
