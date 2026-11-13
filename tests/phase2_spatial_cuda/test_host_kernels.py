"""The rendered spatial CUDA source EXECUTED on the host (real threads + barrier, see hostgpu.py) through the REAL Python launchers and public API, compared with the accepted explicit NumPy spatial path:
declared bound rtol 2e-12 / atol 1e-14 (bitwise at zero diffusivity), identical typed failures / flag bits / flagged counts, padded lanes and inactive cells at every barrier (N = 1/127/128/129/257,
all-inactive), mixed layouts, uniform init, moisture, storage. NOT hardware evidence; a plumbing double stands in for CuPy. Author-written."""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import BIG, a2, device_owner, hostgpu, needs_host, sp, ss

pytestmark = needs_host
RT, AT = 2e-12, 1e-14
FIELDS = ("surface_return_m", "drainage_m", "evaporation_m", "evaporation_unmet_m", "percolation_m", "diffusion_m")


def forcing(rng, p, mask):
    L, N = p["res"].shape
    return rng.uniform(0, 1e-3, (L, N)) * 0.3 * mask, rng.uniform(0, 2e-7, N) * mask


def run(owner, to_arr, water, dt, added, pet, share, pulse, t0=0.0):
    state = sp.spatial_column_state(owner, to_arr(water), t0, memory_budget_bytes=BIG)
    try:
        if pulse is None:
            return sp.spatial_step(owner, state, dt, added_water_m=to_arr(added), potential_evaporation_m_s=to_arr(pet), evaporation_share=None if share is None else to_arr(share), memory_budget_bytes=BIG), state
        return sp.spatial_step_with_recharge(owner, state, dt, recharge_depth_m=pulse, potential_evaporation_m_s=to_arr(pet), evaporation_share=None if share is None else to_arr(share), memory_budget_bytes=BIG), state
    except ss.SubsurfaceError as exc:
        return (type(exc).__name__, getattr(exc, "bit", None), getattr(exc, "n_flagged", None)), state


def compare(ref, got, *, bitwise=False):
    if isinstance(ref, tuple):
        assert got == ref, (ref, got)
        return
    assert not isinstance(got, tuple), got
    kw = dict(rtol=0, atol=0) if bitwise else dict(rtol=RT, atol=AT)
    for f in FIELDS:
        np.testing.assert_allclose(np.asarray(getattr(got, f)), getattr(ref, f), err_msg=f, **kw)
    np.testing.assert_allclose(np.asarray(got.state.water_m), ref.state.water_m, err_msg="water", **kw)
    assert got.state.t_s == ref.state.t_s and got.dt_s == ref.dt_s
    for name in ("storage_before_m3", "storage_after_m3", "added_m3", "surface_return_m3", "drainage_m3", "evaporation_m3", "evaporation_unmet_m3"):
        np.testing.assert_allclose(getattr(got.budget, name), getattr(ref.budget, name), rtol=1e-11, atol=1e-13, err_msg=name)
    assert got.budget.n_terms == ref.budget.n_terms and got.budget.bound_m3 == pytest.approx(ref.budget.bound_m3, rel=1e-11)


@pytest.mark.parametrize("N", [1, 127, 128, 129, 257])
@pytest.mark.parametrize("L", [1, 3])
def test_every_padded_lane_and_inactive_cell_reaches_the_barriers_for_each_block_count(cupy, N, L, mode, bottom):
    rng = np.random.default_rng(N + 10 * L)
    p = a2.params(rng, L, N, mode=mode)
    mask = rng.random(N) < 0.7
    mask[0] = True
    np_owner = a2.prepare(a2.make_spec(p, mode=mode, bottom=bottom), mask=mask)
    dev = device_owner(np_owner, cupy)
    water = a2.water_kind(rng, p, "random")
    added, pet = forcing(rng, p, mask)
    for share, pulse in ((None, None), (rng.random(N), None), (None, 2e-3)):
        ref, _ = run(np_owner, lambda a: a, water, 3600.0, added, pet, share, pulse)
        got, _ = run(dev, hostgpu.fake, water, 3600.0, added, pet, share, pulse)
        compare(ref, got)


@pytest.mark.parametrize("N", [1, 128, 129])
def test_an_all_inactive_domain_reports_zero_totals_and_keeps_the_state(cupy, N):
    rng = np.random.default_rng(N)
    p = a2.params(rng, 3, N)
    mask = np.zeros(N, bool)
    np_owner = a2.prepare(a2.make_spec(p), mask=mask)
    dev = device_owner(np_owner, cupy)
    water = a2.water_kind(rng, p, "random")
    ref, _ = run(np_owner, lambda a: a, water, 3600.0, np.zeros((3, N)), np.zeros(N), None, None)
    got, _ = run(dev, hostgpu.fake, water, 3600.0, np.zeros((3, N)), np.zeros(N), None, None)
    compare(ref, got, bitwise=True)
    np.testing.assert_array_equal(np.asarray(got.state.water_m).view(np.uint64), water.view(np.uint64))
    assert got.budget.storage_before_m3 == got.budget.storage_after_m3 == got.budget.added_m3 == 0.0


@pytest.mark.parametrize("L", [1, 2, 3, 5])
def test_states_and_flags_across_stack_heights_modes_bottoms_dt_share_and_pulse(cupy, L, mode, bottom):
    rng = np.random.default_rng(100 + L)
    N = 130
    p = a2.params(rng, L, N, mode=mode)
    mask = rng.random(N) < 0.8
    mask[0], mask[1] = True, False
    np_owner = a2.prepare(a2.make_spec(p, mode=mode, bottom=bottom, gravity=True), mask=mask)
    dev = device_owner(np_owner, cupy)
    for kind in ("dry", "above_fc", "mixed"):
        water = a2.water_kind(rng, p, kind)
        added, pet = forcing(rng, p, mask)
        for dt in (0.0, 86400.0):
            for share, pulse in ((None, None), (rng.random(N), None), (None, 2.5e-3), (rng.random(N), 40.0)):
                if pulse is not None and not dt > 0.0:
                    continue
                ref, _ = run(np_owner, lambda a: a, water, dt, added, pet, share, pulse)
                got, _ = run(dev, hostgpu.fake, water, dt, added, pet, share, pulse)
                compare(ref, got)


def test_zero_diffusivity_is_bitwise_identical_to_the_numpy_path(cupy, mode, bottom):
    rng = np.random.default_rng(7)
    L, N = 4, 129
    p = a2.params(rng, L, N, mode=mode)
    p["iface"] = np.zeros_like(p["iface"])
    mask = rng.random(N) < 0.8
    np_owner = a2.prepare(a2.make_spec(p, mode=mode, bottom=bottom), mask=mask)
    dev = device_owner(np_owner, cupy)
    for kind in ("above_fc", "random"):
        water = a2.water_kind(rng, p, kind)
        added, pet = forcing(rng, p, mask)
        share = rng.random(N)
        ref, _ = run(np_owner, lambda a: a, water, 86400.0, added, pet, share, None)
        got, _ = run(dev, hostgpu.fake, water, 86400.0, added, pet, share, None)
        compare(ref, got, bitwise=True)


def _layout_cases(rng, n):
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "phase2_spatial_numba" / "conftest.py"
    spec = importlib.util.spec_from_file_location("a3_builders_cuda", path)
    a3 = importlib.util.module_from_spec(spec)
    sys.modules["a3_builders_cuda"] = a3
    spec.loader.exec_module(a3)
    return a3


@pytest.mark.parametrize("mode_", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("L", [1, 2, 3, 5])
def test_mixed_scalar_shared_and_cell_layouts_incl_hoisted_and_per_cell_interface_factors(cupy, mode_, L):
    a3 = _layout_cases(None, 0)
    rng = np.random.default_rng(300 + L)
    N = 130
    mask = rng.random(N) < 0.85
    layouts = ("scalar", "shared", "cell")
    combos = list(itertools.product(("shared", "cell"), layouts, layouts, layouts, layouts))
    for idx in rng.choice(len(combos), size=6, replace=False):
        edges, res, fc, sat, iface = combos[int(idx)]
        inputs, p = a3.layout_case(rng, L, N, edges=edges, res=res, fc=fc, sat=sat, iface=iface, mode=mode_)
        np_owner = a2.prepare(a3.spec_from(inputs, N, mode=mode_, bottom="closed"), mask=mask)
        dev = device_owner(np_owner, cupy)
        water = a2.water_kind(rng, p, "above_fc")
        added, pet = forcing(rng, p, mask)
        share = rng.random(N)
        for dt, pulse in ((3600.0, None), (86400.0, 2e-3)):
            ref, _ = run(np_owner, lambda a: a, water, dt, added, pet, share, pulse)
            got, _ = run(dev, hostgpu.fake, water, dt, added, pet, share, pulse)
            compare(ref, got)


def _set(arr, idx, value):
    if arr.ndim == 2:
        idx = (idx[0] % arr.shape[0], idx[1])
    arr[idx] = value
    return arr


@pytest.mark.parametrize("name,mutate,kw", [
    ("nonfinite_water", lambda w, a, e, s: (_set(w, (1, 2), np.nan), a, e, s), {}),
    ("negative_water", lambda w, a, e, s: (_set(w, (0, 4), -1e-3), a, e, s), {}),
    ("over_saturated_water", lambda w, a, e, s: (_set(w, (1, 5), 10.0), a, e, s), {}),
    ("nonfinite_added", lambda w, a, e, s: (w, _set(a, (0, 1), np.inf), e, s), {}),
    ("negative_added", lambda w, a, e, s: (w, _set(a, (1, 2), -1e-6), e, s), {}),
    ("added_on_inactive", lambda w, a, e, s: (w, _set(a, (0, 3), 1e-6), e, s), {}),
    ("nonfinite_pet", lambda w, a, e, s: (w, a, _set(e, 2, np.nan), s), {}),
    ("negative_pet", lambda w, a, e, s: (w, a, _set(e, 2, -1e-9), s), {}),
    ("pet_on_inactive", lambda w, a, e, s: (w, a, _set(e, 3, 1e-9), s), {}),
    ("added_at_dt0", lambda w, a, e, s: (w, _set(a, (0, 0), 1e-6), e, s), {"dt": 0.0}),
    ("share_nonfinite", lambda w, a, e, s: (w, a, e, _set(s, 4, np.nan)), {"share": True}),
    ("share_out_of_range", lambda w, a, e, s: (w, a, e, _set(s, 3, 1.5)), {"share": True}),
    ("two_flag_cells_count", lambda w, a, e, s: (_set(_set(w, (0, 1), np.nan), (0, 8), -1.0), a, e, s), {}),
])
@pytest.mark.parametrize("L", [1, 3])
def test_invalid_inputs_raise_the_identical_typed_error_with_the_same_lowest_bit_and_flagged_count(cupy, name, mutate, kw, L):
    rng = np.random.default_rng(11)
    N = 140
    p = a2.params(rng, L, N)
    mask = np.ones(N, bool)
    mask[3] = False
    np_owner = a2.prepare(a2.make_spec(p), mask=mask)
    dev = device_owner(np_owner, cupy)
    water = a2.water_kind(rng, p, "random")
    added, pet = forcing(rng, p, mask)
    share = rng.random(N) if kw.get("share") else None
    water, added, pet, share = mutate(water.copy(), added.copy(), pet.copy(), share)
    dt = kw.get("dt", 3600.0)
    ref, rs = run(np_owner, lambda a: a, water, dt, added, pet, share, None)
    got, gs = run(dev, hostgpu.fake, water, dt, added, pet, share, None)
    assert isinstance(ref, tuple) and ref[0] == "SubsurfaceStepError"
    assert got == ref
    assert gs.t_s == 0.0 and np.array_equal(np.asarray(gs.water_m).view(np.uint64), np.array(rs.water_m).view(np.uint64))  # a refusal leaves the accepted state untouched


@pytest.mark.parametrize("bottom_", ["open", "closed"])
def test_the_pulse_fit_failure_is_the_same_typed_trial_outcome(cupy, bottom_, mode):
    rng = np.random.default_rng(3)
    N = 129
    p = a2.params(rng, 3, N, mode=mode)
    np_owner = a2.prepare(a2.make_spec(p, mode=mode, bottom=bottom_))
    dev = device_owner(np_owner, cupy)
    water = a2.water_kind(rng, p, "random")
    pet = np.zeros(N)
    ref, _ = run(np_owner, lambda a: a, water, 86400.0, None, pet, None, 50.0)
    got, _ = run(dev, hostgpu.fake, water, 86400.0, None, pet, None, 50.0)
    if bottom_ == "closed":
        assert ref[0] == "SubsurfaceFitFailure" and ref[1] == ss.RECHARGE_FIT_BIT
    compare(ref, got)


# ---- uniform init, moisture, storage -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1, 127, 129, 257])
def test_uniform_state_moisture_and_storage_match_the_numpy_helpers(cupy, N):
    rng = np.random.default_rng(N)
    L = 3
    p = a2.params(rng, L, N)
    mask = rng.random(N) < 0.6
    mask[0] = True
    np_owner = a2.prepare(a2.make_spec(p), mask=mask)
    dev = device_owner(np_owner, cupy)
    theta = [0.0, 0.03, 0.05]  # zero residual-floor moisture is allowed
    ref = sp.spatial_uniform_state(np_owner, theta, 5.0, memory_budget_bytes=BIG)
    got = sp.spatial_uniform_state(dev, theta, 5.0, memory_budget_bytes=BIG)
    np.testing.assert_array_equal(np.asarray(got.water_m).view(np.uint64), ref.water_m.view(np.uint64))  # the same IEEE product theta * dz
    assert got.t_s == 5.0 and got.component_tag == sp.STATE_COMPONENT_TAG_CUDA and got.namespace == "cupy" and got.device_id == 0 and ref.component_tag == sp.STATE_COMPONENT_TAG
    np.testing.assert_array_equal(np.asarray(sp.spatial_moisture(dev, got, memory_budget_bytes=BIG)).view(np.uint64), sp.spatial_moisture(np_owner, ref, memory_budget_bytes=BIG).view(np.uint64))
    s_ref, s_got = sp.spatial_storage_m3(np_owner, ref, memory_budget_bytes=BIG), sp.spatial_storage_m3(dev, got, memory_budget_bytes=BIG)
    assert s_got == pytest.approx(s_ref, rel=1e-12)


def test_uniform_threshold_is_checked_in_every_cell_including_inactive_placeholders(cupy):
    rng = np.random.default_rng(1)
    N = 150
    p = a2.params(rng, 3, N)
    sat = a2.sat_depth(p)  # noqa: F841
    mask = np.ones(N, bool)
    mask[149] = False  # an INACTIVE cell with the lowest saturation decides the refusal
    p["sat"][1, 149] = 0.2
    p["fc"][1, 149] = 0.15
    np_owner = a2.prepare(a2.make_spec(p), mask=mask)
    dev = device_owner(np_owner, cupy)
    theta = [0.05, 0.3, 0.05]
    with pytest.raises(ss.SubsurfaceError) as e_ref:
        sp.spatial_uniform_state(np_owner, theta, 0.0, memory_budget_bytes=BIG)
    with pytest.raises(ss.SubsurfaceError) as e_got:
        sp.spatial_uniform_state(dev, theta, 0.0, memory_budget_bytes=BIG)
    assert str(e_ref.value) == str(e_got.value) and "theta_by_layer[1]" in str(e_got.value)
    with pytest.raises(ss.SubsurfaceError) as e2:  # the FIRST offending layer is named even when a later one also fails
        sp.spatial_uniform_state(dev, [-0.1, 0.3, 0.9], 0.0, memory_budget_bytes=BIG)
    assert "theta_by_layer[0]" in str(e2.value)


def test_storage_excludes_inactive_cells_never_clips_negatives_and_refuses_nonfinite_or_unrepresentable(cupy):
    rng = np.random.default_rng(2)
    N = 129
    p = a2.params(rng, 2, N)
    mask = rng.random(N) < 0.5
    mask[0] = True
    np_owner = a2.prepare(a2.make_spec(p), mask=mask)
    dev = device_owner(np_owner, cupy)
    water = a2.water_kind(rng, p, "random")
    water[0, 0] = -0.01  # a finite negative value is NOT silently clipped
    water[:, ~mask] = 1e300  # inactive cells are zeroed, not counted (and cannot overflow the total)
    s_ref = sp.spatial_storage_m3(np_owner, sp.spatial_column_state(np_owner, water, 0.0, memory_budget_bytes=BIG), memory_budget_bytes=BIG)
    s_got = sp.spatial_storage_m3(dev, sp.spatial_column_state(dev, hostgpu.fake(water), 0.0, memory_budget_bytes=BIG), memory_budget_bytes=BIG)
    assert s_got == pytest.approx(s_ref, rel=1e-12) and np.isfinite(s_got)
    bad = water.copy()
    bad[1, 0] = np.nan
    with pytest.raises(sp.SpatialLayerError, match="not representable"):
        sp.spatial_storage_m3(dev, sp.spatial_column_state(dev, hostgpu.fake(bad), 0.0, memory_budget_bytes=BIG), memory_budget_bytes=BIG)
