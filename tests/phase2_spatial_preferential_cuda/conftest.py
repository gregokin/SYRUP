"""S2-B2.d spatial CUDA preferential suite (author-written; CPU lane = host-compiled source execution + plumbing doubles, NOT hardware evidence; GPU lane = gated definitions for Root). Builders are loaded from
the sibling suites. Every required trigger/data of this suite is generated inside it (no dependence on ignored Root task files)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parents[1]
CANDIDATE = TESTS.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


a2 = _load("a2_builders_b2d", TESTS / "phase2_spatial_water" / "conftest.py")
a3 = _load("a3_builders_b2d", TESTS / "phase2_spatial_numba" / "conftest.py")

import hostb2d  # noqa: E402
import hostgpu  # noqa: E402

from maple_syrup import preferential_infiltration as pi  # noqa: E402
from maple_syrup import preferential_infiltration_cuda as pc  # noqa: E402
from maple_syrup import spatial_preferential_infiltration as spf  # noqa: E402
from maple_syrup import spatial_preferential_infiltration_cuda as tc  # noqa: E402
from maple_syrup import spatial_soil_layers as sp  # noqa: E402
from maple_syrup import subsurface as ss  # noqa: E402

BIG, AREA = a2.BIG, a2.AREA


def sm_hash(h, name, kind, arr):
    """The CPU canonical stream of one law map (`spatial_matrix_infiltration._hash_array`, reused by the CPU preferential identity)."""
    from maple_syrup import spatial_matrix_infiltration as sm

    sm._hash_array(h, name, kind, arr)

RTOL, ATOL = 2e-12, 1e-14
needs_host = pytest.mark.skipif(not hostgpu.have_toolchain(), reason="no C++20 toolchain with <barrier>: the host execution of the CUDA source is not run (no claim)")
LAYOUT_SETS = {
    "all_cell": dict(edges="cell", res="cell", fc="cell", sat="cell", iface="cell"),
    "all_shared": dict(edges="shared", res="shared", fc="shared", sat="shared", iface="shared"),
    "scalar_thetas": dict(edges="shared", res="scalar", fc="scalar", sat="scalar", iface="scalar"),
    "mixed": dict(edges="cell", res="shared", fc="scalar", sat="cell", iface="shared"),
}


def bits(a):
    return np.ascontiguousarray(np.asarray(a)).view(np.uint64)


@pytest.fixture(scope="session")
def so():
    if not hostgpu.have_toolchain():
        pytest.skip("no C++20 toolchain")
    return hostb2d.build(tc.render_source())


@pytest.fixture()
def cupy(so, monkeypatch):
    """A fresh plumbing double wired into the spatial modules (CuPy probe, device id, device reading); the kernel cache is cleared before and after."""
    cp = hostb2d.FakeCupy(so)

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
    """PLUMBING-DOUBLE owner: the retained maps of a prepared numpy owner held as fake device arrays, tagged backend='cuda' (built through the owner's private construction context purely so that the REAL
    launcher/API code can run on host). Evidence of API plumbing and kernel text only."""
    maps = {k: hostb2d.fake(v, device) for k, v in np_owner._maps.items()}
    mask = hostb2d.fake(np_owner._mask.astype(np.bool_), device)
    with sp._constructing():
        return sp.SpatialSoilColumn("cuda", np_owner.n_layers, np_owner.n_cells, np_owner.n_active, np_owner.cell_area_m2, np_owner.diffusion_mode, np_owner.bottom_drainage, np_owner.gravity_drainage,
                                    np_owner.diffusion_coefficient, np_owner.theta_evaporation_stress, np_owner.evaporation_factor, np_owner.layouts, np_owner.parameters_sha256, np_owner.mask_sha256,
                                    np_owner.estimate, device, maps, mask)


def fx(a, device=0):
    return hostb2d.fake(a, device)


def dev_state(column, water, t_s=0.0):
    return sp._new_state(column, fx(water, column.device_id), t_s)


def weight_vec(rng, L, kind):
    if kind == "zero":
        return [0.0] * L
    if kind == "top":
        return [1.0] + [0.0] * (L - 1)
    if kind == "lower":
        return [0.0] + [1.0] * (L - 1)
    return list(rng.uniform(0.0, 3.0, L))


def weight_case(rng, L, kind):
    return {
        "no_species": dict(background=weight_vec(rng, L, "top"), species=[]),
        "no_background": dict(background=None, species=[weight_vec(rng, L, "lower")]),
        "zero_background_one_species": dict(background=weight_vec(rng, L, "zero"), species=[weight_vec(rng, L, "random")]),
        "multi_overlap": dict(background=weight_vec(rng, L, "random"), species=[weight_vec(rng, L, "random"), weight_vec(rng, L, "lower"), weight_vec(rng, L, "random")]),
        "disabled_species_with_cover": dict(background=weight_vec(rng, L, "lower"), species=[weight_vec(rng, L, "zero"), weight_vec(rng, L, "random")]),
        "all_zero": dict(background=weight_vec(rng, L, "zero"), species=[weight_vec(rng, L, "zero")]),
    }[kind]


def numpy_owner(inputs, N, mask, area=AREA):
    return sp.prepare_spatial_column(a3.spec_from(inputs, N), mask, area, backend="numpy", memory_budget_bytes=BIG)


def build_pair(cupy, np_col, kwargs, contributions, *, device=0, budget=BIG, **quota):
    """The CPU spatial operator and the CUDA-double operator of the SAME explicit inputs. `kwargs` hold host arrays/scalars; per-cell arrays are made device arrays for the CUDA operator."""
    conv = lambda v: fx(v, device) if isinstance(v, np.ndarray) else v  # noqa: E731
    cop = spf.prepare_spatial_preferential(np_col, **kwargs, species_contributions=None if contributions is None else iter([c.copy() for c in contributions]), memory_budget_bytes=BIG)
    col = device_owner(np_col, device)
    gkw = {k: (conv(v) if k in ("a3", "max_intake_rate_m_s") else v) for k, v in kwargs.items()}
    gop = spf.prepare_spatial_preferential(col, **gkw, species_contributions=None if contributions is None else iter([fx(c, device) for c in contributions]), memory_budget_bytes=budget, **quota)
    return cop, col, gop
