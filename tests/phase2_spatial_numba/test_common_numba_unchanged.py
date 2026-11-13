"""Moving the compiled body behind indexed readers must leave the accepted COMMON Numba column bit-identical to the pre-A3 kernel (task evidence copy `s2_a3_pre_sources/subsurface_numba.py`, skipped if absent
from this tree) over a wide scenario set, and must keep its parameter ledger and public surface. Author-written."""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest
from conftest import DTS, PRE_A3, require_numba, ss

require_numba()
from maple_syrup import subsurface_numba as new  # noqa: E402


@pytest.fixture(scope="module")
def old():
    if not PRE_A3.is_file():
        pytest.skip("the pre-A3 source copy is not available in this tree (task evidence only)")
    spec = importlib.util.spec_from_file_location("old_subsurface_numba_pre_a3", PRE_A3)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def layers(L, mode, bottom, gravity):
    edges = np.concatenate([[0.0], np.cumsum(np.linspace(0.08, 0.3, L))])
    if L == 1:
        kw = dict(diffusivity_m2_s=1e-7) if mode == "constant" else dict(interface_conductivity_m_s=1e-6, diffusion_coefficient=0.25)
    elif mode == "constant":
        kw = dict(diffusivity_m2_s=np.linspace(2e-7, 9e-7, L - 1))
    else:
        kw = dict(interface_conductivity_m_s=np.linspace(1e-6, 8e-6, L - 1), diffusion_coefficient=0.25)
    return ss.soil_layers(edges, theta_residual=np.linspace(0.01, 0.04, L), theta_field_capacity=np.linspace(0.18, 0.25, L), theta_saturation=np.linspace(0.4, 0.5, L),
                          theta_evaporation_stress=0.5, diffusion_mode=mode, bottom_drainage=bottom, gravity_drainage=gravity, **kw)


def same(a, b, label):
    """Bit identity of every output field. The ONE tolerated difference is the sign/payload bit of a NaN produced by an invalid input (e.g. inf - inf in a cell the step flags and refuses): the hardware
    default NaN of `inf - inf` is sign-negative while a folded constant is positive, so the two compiled bodies may order two such operations differently. Both are NaN; no finite value, infinity,
    signed zero or flag differs."""
    for name in ss.StepArrays._fields[:-1]:
        x, y = np.ascontiguousarray(getattr(a, name)), np.ascontiguousarray(getattr(b, name))
        both_nan = np.isnan(x) & np.isnan(y)
        np.testing.assert_array_equal(np.where(both_nan, 0.0, x).view(np.uint64), np.where(both_nan, 0.0, y).view(np.uint64), err_msg=f"{label} {name}")


@pytest.mark.parametrize("L", [1, 2, 3, 5])
@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("bottom", ["open", "closed"])
def test_new_common_numba_kernel_equals_the_pre_a3_kernel_bit_for_bit(old, L, mode, bottom):
    rng = np.random.default_rng(L * 13)
    N = 60
    for gravity in (True, False):
        lay = layers(L, mode, bottom, gravity)
        mask = rng.random(N) < 0.8
        col = ss.prepare_column(lay, mask, 4.0, backend="numba")
        old_run, _ = old.prepare_runner(lay)
        sat = lay.saturation_m[:, None]
        for kind in ("random", "above", "invalid"):
            w = sat * rng.random((L, N))
            if kind == "above":
                w = lay.field_capacity_m[:, None] + (sat - lay.field_capacity_m[:, None]) * rng.random((L, N))
            if kind == "invalid":
                w[0, 3], w[-1, 7] = -1e-3, np.nan
            add = rng.random((L, N)) * 1e-3 * mask
            pet = rng.random(N) * 1e-7 * mask
            if kind == "invalid":
                add[0, 11], pet[13] = np.inf, -1.0
            for dt in DTS:
                for share in (None, rng.random(N), np.where(np.arange(N) == 5, 1.5, 0.5)):
                    for recharge in (None, 2e-3):
                        if recharge is not None and not dt > 0.0:
                            continue
                        a = col.runner(col, w, add if recharge is None else None, pet, dt, share, recharge)
                        b = old_run(col, w, add if recharge is None else None, pet, dt, share, recharge)
                        same(a, b, f"L{L} {mode} {bottom} g{gravity} {kind} dt{dt}")


def test_the_common_column_ledger_local_scratch_and_public_surface_are_unchanged(old):
    for L, mode in ((1, "constant"), (3, "constant"), (3, "tietjen_donor_depth"), (5, "tietjen_donor_depth"), (40, "constant")):
        lay = layers(L, mode, "open", True) if L <= 5 else layers_tall(L)
        col = ss.prepare_column(lay, np.ones(4, bool), 4.0, backend="numba")
        _, old_info = old.prepare_runner(lay)
        assert col.backend_info["parameter_host_bytes"] == old_info["parameter_host_bytes"] == ss.kernel_parameter_doubles(L, mode) * 8
        assert col.backend_info["factor_scratch_host_bytes"] == old_info["factor_scratch_host_bytes"] == ss.factor_scratch_doubles(L, mode) * 8
        # the common variant's bound values alias the cached arrays: its per-call locals ARE the frozen estimate's factor term, not more
        assert col.backend_info["kernel_local_scratch_host_bytes"] == ss.factor_scratch_doubles(L, mode) * 8
    for name in old.__all__:
        assert hasattr(new, name), name  # every public name of the pre-A3 module still exists


def layers_tall(L):
    edges = np.concatenate([[0.0], np.cumsum(np.full(L, 0.05))])
    return ss.soil_layers(edges, theta_residual=np.full(L, 0.02), theta_field_capacity=np.full(L, 0.2), theta_saturation=np.full(L, 0.4), theta_evaporation_stress=0.5,
                          diffusivity_m2_s=np.full(L - 1, 2e-7))
