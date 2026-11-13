"""S2-B2.c spatial CUDA matrix infiltration suite (author-written; CPU lane = host-compiled source execution + plumbing doubles, NOT hardware evidence; GPU lane = gated definitions for Root). Builders are loaded
from the sibling suites."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parents[1]
CANDIDATE = TESTS.parent
TASK = CANDIDATE.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


a2 = _load("a2_builders_b2c", TESTS / "phase2_spatial_water" / "conftest.py")
a3 = _load("a3_builders_b2c", TESTS / "phase2_spatial_numba" / "conftest.py")

import hostb2c  # noqa: E402
import hostgpu  # noqa: E402

from maple_syrup import matrix_infiltration as mi  # noqa: E402
from maple_syrup import matrix_infiltration_cuda as mc  # noqa: E402
from maple_syrup import spatial_matrix_infiltration as sm  # noqa: E402
from maple_syrup import spatial_matrix_infiltration_cuda as tc  # noqa: E402
from maple_syrup import spatial_soil_layers as sp  # noqa: E402
from maple_syrup import subsurface as ss  # noqa: E402

BIG, AREA = a2.BIG, a2.AREA
RTOL, ATOL = 2e-12, 1e-14
needs_host = pytest.mark.skipif(not hostgpu.have_toolchain(), reason="no C++20 toolchain with <barrier>: the host execution of the CUDA source is not run (no claim)")


def bits(a):
    return np.ascontiguousarray(np.asarray(a)).view(np.uint64)


@pytest.fixture(scope="session")
def so():
    if not hostgpu.have_toolchain():
        pytest.skip("no C++20 toolchain")
    return hostb2c.build(tc.render_source())


@pytest.fixture()
def cupy(so, monkeypatch):
    """A fresh plumbing double wired into the spatial modules (CuPy probe, device id, device reading); the kernel cache is cleared before and after."""
    cp = hostb2c.FakeCupy(so)

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
    maps = {k: hostb2c.fake(v, device) for k, v in np_owner._maps.items()}
    mask = hostb2c.fake(np_owner._mask.astype(np.bool_), device)
    with sp._constructing():
        return sp.SpatialSoilColumn("cuda", np_owner.n_layers, np_owner.n_cells, np_owner.n_active, np_owner.cell_area_m2, np_owner.diffusion_mode, np_owner.bottom_drainage, np_owner.gravity_drainage,
                                    np_owner.diffusion_coefficient, np_owner.theta_evaporation_stress, np_owner.evaporation_factor, np_owner.layouts, np_owner.parameters_sha256, np_owner.mask_sha256,
                                    np_owner.estimate, device, maps, mask)


def fx(a, device=0):
    return hostb2c.fake(a, device)


def dev_state(column, water, t_s=0.0):
    """A CUDA-tagged state record of fake device water (the owner's own constructor is the NumPy one; this is the same record type the real CUDA path publishes)."""
    return sp._new_state(column, fx(water, column.device_id), t_s)


LAYOUT_SETS = {
    "all_cell": dict(edges="cell", res="cell", fc="cell", sat="cell", iface="cell"),
    "all_shared": dict(edges="shared", res="shared", fc="shared", sat="shared", iface="shared"),
    "scalar_thetas": dict(edges="shared", res="scalar", fc="scalar", sat="scalar", iface="scalar"),
    "mixed": dict(edges="cell", res="shared", fc="scalar", sat="cell", iface="shared"),
}


def forcing(rng, mask, dt, N):
    """Per-cell regimes: zero rain, low rate (unponded), high rate (onset inside/already), contact ponded, exhausted pool; exact K = 0 / psi = 0 / g = 0 / g = 1 cells."""
    K = rng.uniform(1e-8, 2e-5, N)
    psi = rng.uniform(0.0, 0.3, N)
    g = rng.uniform(0.05, 1.0, N)
    K[0::11], psi[1::7], g[2::9], g[3::13] = 0.0, 0.0, 0.0, 1.0
    regime = rng.integers(0, 6, N)
    rate = np.where(regime == 0, 0.0, np.where(regime == 1, K * rng.uniform(0.1, 1.0, N), np.where(regime == 2, K * rng.uniform(1.5, 6.0, N) + 1e-7, rng.uniform(0.0, 5e-5, N))))
    contact = (regime == 3) | (regime == 4)
    standing = np.where(contact, rng.uniform(1e-4, 5e-3, N), 0.0)
    pool = standing + rate * dt
    pool = np.where(regime == 4, pool * 0.2, pool)
    pool = np.where((~contact) & (regime == 5), rate * dt * rng.uniform(0.3, 1.0, N), pool)
    rate, standing, pool = (np.where(mask, a, 0.0) for a in (rate, standing, pool))
    pool = np.where(mask, pool, rng.uniform(0.0, 1e-3, N))
    return K, psi, g, np.ascontiguousarray(pool), np.ascontiguousarray(standing), np.ascontiguousarray(rate)


def water_for(rng, p, N):
    L = p["res"].shape[0]
    dz = np.diff(p["edges"], axis=0)
    fc, sat = p["fc"] * dz, p["sat"] * dz
    w = fc * rng.uniform(0.1, 1.0, (L, N))
    above = rng.random(N) < 0.35
    w[:, above] = (fc + (sat - fc) * rng.uniform(0.0, 1.0, (L, N)))[:, above]
    full = rng.random(N) < 0.15
    w[0, full] = sat[0, full] * (1.0 - rng.uniform(0.0, 1e-3, full.sum()))
    zero = rng.random(N) < 0.1
    w[:, zero] = 0.0
    return np.ascontiguousarray(w)


# ---- S2-B2.c review01/review02: the history-sequence data and the HISTORICAL pre-review02 refusal evidence ---------------------------------------------------
from types import SimpleNamespace  # noqa: E402

STEP_DTS = (60.0, 3600.0, 0.0, 600.0)
#: HISTORICAL evidence (review01): before the review02 bounded-combined-intake correction, the second step (dt = 3600 s) of these two valid-input sequences was refused by the STRICT contract (bit 17, one cell: a
#: rounding-level negative pool from the `pre + post` recombination) on the spatial NumPy path, the spatial CUDA path and the accepted common CUDA operator alike. They are kept as the recorded cells/pools so the
#: tests can show that EXACTLY these cells now integrate (intake bounded by the pool, pool-out exactly 0). It is NOT an expectation of refusal: every test now requires success on all four stages.
HISTORICAL_PRE_REVIEW02_REFUSALS = {
    (3, 128, "all_shared"): dict(step=1, bit=17, n_flagged=1, cell=103, pool=0.04838095732241791, old_pool_out=-6.938893903907228e-18),
    (5, 257, "scalar_thetas"): dict(step=1, bit=17, n_flagged=1, cell=248, pool=0.029264621197886812, old_pool_out=-3.469446951953614e-18),
}


def sequence_case(L: int, N: int, layout: str) -> SimpleNamespace:
    """The data of the history-sequence definition, drawn in EXACTLY the order of the original gated test (layout case, mask, law K/psi/g at dt = 60, water, J0, then one forcing per step) so that a seed
    reproduces the same inputs; nothing is altered, skipped or erased."""
    rng = np.random.default_rng(1000 * L + N + len(layout))
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS[layout])
    mask = rng.random(N) < 0.8
    mask[0] = True
    K, psi, g, *_ = forcing(rng, mask, 60.0, N)
    water = water_for(rng, p, N)
    j0 = np.where(rng.random(N) < 0.5, 0.0, rng.uniform(0.0, 0.05, N))
    steps = []
    for dt in STEP_DTS:
        _, _, _, pool, standing, rate = forcing(rng, mask, dt, N)
        steps.append((dt, pool, standing, rate))
    return SimpleNamespace(L=L, N=N, layout=layout, inputs=inputs, p=p, mask=mask, K=K, psi=psi, g=g, water=water, j0=j0, steps=steps)
