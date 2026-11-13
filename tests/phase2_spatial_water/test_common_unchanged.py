"""The factoring of `reference_step` into one private body must leave the COMMON path bit-identical: every StepArrays field of the new sources equals the pre-A2 source on a wide scenario set
(both modes, both bottoms, gravity off, shares, pulse, masks, flagged invalid states, dt boundaries). The pre-A2 source copy is task evidence (skipped if absent from this tree). Author-written."""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import DTS

from maple_syrup import subsurface as ss

OLD = Path(__file__).resolve().parents[3] / "s2_a2_pre_sources" / "subsurface.py"


@pytest.fixture(scope="module")
def old():
    if not OLD.is_file():
        pytest.skip("the pre-A2 source copy is not available in this tree (task evidence only)")
    spec = importlib.util.spec_from_file_location("old_subsurface_pre_a2", OLD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def layers(mod, L, mode, bottom, gravity):
    edges = np.concatenate([[0.0], np.cumsum(np.linspace(0.08, 0.3, L))])
    kw = dict(diffusivity_m2_s=np.linspace(2e-7, 9e-7, max(L - 1, 0))) if mode == "constant" else dict(interface_conductivity_m_s=np.linspace(1e-6, 8e-6, max(L - 1, 0)), diffusion_coefficient=0.25)
    if L == 1:
        kw = dict(diffusivity_m2_s=1e-7) if mode == "constant" else dict(interface_conductivity_m_s=1e-6, diffusion_coefficient=0.25)
    return mod.soil_layers(edges, theta_residual=np.linspace(0.01, 0.04, L), theta_field_capacity=np.linspace(0.18, 0.25, L), theta_saturation=np.linspace(0.4, 0.5, L),
                           theta_evaporation_stress=0.5, diffusion_mode=mode, bottom_drainage=bottom, gravity_drainage=gravity, **kw)


def same(a, b):
    for name in ss.StepArrays._fields:
        if getattr(a, name) is None or getattr(b, name) is None:
            assert getattr(a, name) is None and getattr(b, name) is None, name  # the CUDA keep-alive slot is None on the NumPy reference
            continue
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        assert x.shape == y.shape and x.dtype == y.dtype, name
        np.testing.assert_array_equal(np.ascontiguousarray(x).view(np.uint8), np.ascontiguousarray(y).view(np.uint8), err_msg=name)


@pytest.mark.parametrize("L", [1, 2, 3, 5])
@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("bottom", ["open", "closed"])
@pytest.mark.parametrize("gravity", [True, False])
def test_reference_step_is_bit_identical_to_the_pre_a2_source(old, L, mode, bottom, gravity):
    N = 30
    rng = np.random.default_rng(1000 * L + (mode == "constant") + 2 * (bottom == "open") + 4 * gravity)
    lo, ln = layers(old, L, mode, bottom, gravity), layers(ss, L, mode, bottom, gravity)
    mask = rng.uniform(size=N) > 0.2
    co, cn = old.prepare_column(lo, mask, 4.0), ss.prepare_column(ln, mask.copy(), 4.0)
    sat = np.asarray(ln.saturation_m)[:, None]
    states = {"random": sat * rng.uniform(0.0, 1.0, size=(L, N)), "dry": np.zeros((L, N)), "full": np.repeat(sat, N, axis=1) * 1.0,
              "invalid": np.where(rng.uniform(size=(L, N)) > 0.9, np.nan, sat * rng.uniform(-0.2, 1.4, size=(L, N)))}
    for kind, water in states.items():
        added = rng.uniform(0.0, 2e-3, size=(L, N)) * (rng.uniform(size=(L, N)) > 0.3)
        pet = rng.uniform(0.0, 3e-7, size=N)
        added[:, ~mask] = 0.0
        pet[~mask] = 0.0
        share = rng.uniform(0.0, 1.0, size=N)
        for dt in DTS:
            for sh in (None, share):
                same(old.reference_step(co, water, added, pet, dt, sh), ss.reference_step(cn, water, added, pet, dt, sh))
                same(old.reference_runner(co, water, added, pet, dt, sh), ss.reference_runner(cn, water, added, pet, dt, sh))
            if dt > 0:
                for pulse in (0.0, 1e-4, 3e-3, 0.5):
                    same(old.reference_runner(co, water, None, pet, dt, None, pulse), ss.reference_runner(cn, water, None, pet, dt, None, pulse))
                    same(old.reference_runner(co, water, None, pet, dt, share, pulse), ss.reference_runner(cn, water, None, pet, dt, share, pulse))


@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
def test_public_step_and_step_with_recharge_are_bit_identical_including_budgets(old, mode):
    L, N = 3, 20
    rng = np.random.default_rng(7)
    co, cn = old.prepare_column(layers(old, L, mode, "closed", True), np.ones(N, bool), 4.0), ss.prepare_column(layers(ss, L, mode, "closed", True), np.ones(N, bool), 4.0)
    water = np.asarray(cn.layers.saturation_m)[:, None] * rng.uniform(0.1, 0.9, size=(L, N))
    added, pet = rng.uniform(0, 1e-3, size=(L, N)), rng.uniform(0, 2e-7, size=N)
    so, sn = old.column_state(co, water.copy(), 5.0), ss.column_state(cn, water.copy(), 5.0)
    a, b = old.step(co, so, 3600.0, added_water_m=added, potential_evaporation_m_s=pet), ss.step(cn, sn, 3600.0, added_water_m=added, potential_evaporation_m_s=pet)
    np.testing.assert_array_equal(a.state.water_m, b.state.water_m)
    assert a.state.t_s == b.state.t_s and dataclasses.asdict(a.budget) == dataclasses.asdict(b.budget)  # distinct (equal-shaped) classes in the two modules
    a, b = old.step_with_recharge(co, so, 86400.0, recharge_depth_m=1e-4, potential_evaporation_m_s=pet), ss.step_with_recharge(cn, sn, 86400.0, recharge_depth_m=1e-4, potential_evaporation_m_s=pet)
    np.testing.assert_array_equal(a.state.water_m, b.state.water_m)
    assert dataclasses.asdict(a.budget) == dataclasses.asdict(b.budget)


def test_the_body_text_outside_its_header_is_unchanged():
    """The factoring moved only the parameter reads: the executable body after the header lines is byte-identical to the pre-A2 `reference_step` body (guards against a silent arithmetic edit)."""
    if not OLD.is_file():
        pytest.skip("the pre-A2 source copy is not available in this tree")
    old_src, new_src = OLD.read_text(), Path(ss.__file__).read_text()
    marker = "    pulse = recharge is not None\n"
    old_tail = old_src[old_src.index("def reference_step(") :]
    old_tail = old_tail[old_tail.index(marker) : old_tail.index("def reference_runner(")]
    new_tail = new_src[new_src.index("def _reference_step_body(") :]
    new_tail = new_tail[new_tail.index(marker) : new_tail.index("def reference_runner(")]
    expected = old_tail.replace("            if lay.gravity_drainage:", "            if params.gravity_drainage:")
    assert new_tail == expected
