"""S2-A.4.2 spatial CUDA suite (author-written; CPU lane = host-compiled source execution + plumbing doubles, NOT hardware evidence). The S2-A.2 builders are loaded from the sibling suite."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_A2 = Path(__file__).resolve().parents[1] / "phase2_spatial_water" / "conftest.py"
_spec = importlib.util.spec_from_file_location("a2_builders_cuda", _A2)
a2 = importlib.util.module_from_spec(_spec)
sys.modules["a2_builders_cuda"] = a2
_spec.loader.exec_module(a2)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hostgpu  # noqa: E402

from maple_syrup import spatial_soil_layers as sp  # noqa: E402
from maple_syrup import subsurface as ss  # noqa: E402
from maple_syrup import subsurface_cuda_spatial as cs  # noqa: E402

BIG = a2.BIG
TASK = Path(__file__).resolve().parents[3]
PRE_A42_SPATIAL = TASK / "s2_a4_2_pre_sources" / "spatial_soil_layers.py"

needs_host = pytest.mark.skipif(not hostgpu.have_toolchain(), reason="no C++20 toolchain with <barrier>: the host execution of the CUDA source is not run (no claim)")


@pytest.fixture(scope="session")
def so():
    if not hostgpu.have_toolchain():
        pytest.skip("no C++20 toolchain")
    return hostgpu.build(cs.render_source())


@pytest.fixture()
def cupy(so, monkeypatch):
    """A fresh plumbing double wired into the spatial module (CuPy probe, device id, device reading) and the kernel cache cleared before and after."""
    cp = hostgpu.FakeCupy(so)

    class RC:
        @staticmethod
        def current_device_id(c):
            return RC.current

        @staticmethod
        def default_device_available_bytes_fn(c):
            return lambda: 1 << 40

        current = 0

    monkeypatch.setattr(sp, "_require_cuda", lambda: cp)
    monkeypatch.setattr(sp, "_cuda_module", lambda: RC)
    with cs._LOCK:
        saved = dict(cs._FUNCTIONS)
        cs._FUNCTIONS.clear()
    cp.rc = RC
    yield cp
    with cs._LOCK:
        cs._FUNCTIONS.clear()
        cs._FUNCTIONS.update(saved)


def device_owner(np_owner, cp, device=0):
    """PLUMBING-DOUBLE owner: the same retained maps as a prepared numpy owner, held as fake device arrays, tagged backend='cuda'. Built through the owner's private construction context purely so that the
    REAL launcher/API code can run on host; it is evidence of API plumbing and kernel text only."""
    maps = {k: hostgpu.fake(v, device) for k, v in np_owner._maps.items()}
    mask = hostgpu.fake(np_owner._mask.astype(np.bool_), device)
    with sp._constructing():
        return sp.SpatialSoilColumn("cuda", np_owner.n_layers, np_owner.n_cells, np_owner.n_active, np_owner.cell_area_m2, np_owner.diffusion_mode, np_owner.bottom_drainage, np_owner.gravity_drainage,
                                    np_owner.diffusion_coefficient, np_owner.theta_evaporation_stress, np_owner.evaporation_factor, np_owner.layouts, np_owner.parameters_sha256, np_owner.mask_sha256,
                                    np_owner.estimate, device, maps, mask)


@pytest.fixture(params=["constant", "tietjen_donor_depth"])
def mode(request):
    return request.param


@pytest.fixture(params=["open", "closed"])
def bottom(request):
    return request.param
