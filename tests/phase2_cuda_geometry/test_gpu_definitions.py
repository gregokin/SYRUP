"""D3 REAL GPU definitions (not placeholders, not xfail): run only on Root's hardware with SYRUP_RELEASE_GPU=1 and a free device. They execute the actual CUDA view/rebind/selected-reconcile path on real CuPy arrays and compare with the
CPU oracle (2e-12 / 1e-14), the unselected bit exactness, the one-packet traffic and the large L / N payload plateau. Author did not run them."""
from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SYRUP_RELEASE_GPU") != "1", reason="GPU definitions: set SYRUP_RELEASE_GPU=1 on the hardware qualification host (free device)")


def _cp():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("no CUDA device")
    return backend.cupy_module()


def _pair(cp, edges, depth, bottom="open"):
    from maple_syrup import floating_soil_layers as fl

    edges = np.asarray(edges, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    N, L = depth.size, len(edges) - 1
    act = np.ones(N, bool)
    common = {"theta_evaporation_stress": 0.2, "diffusion_mode": "constant", "bottom_drainage": bottom, "cell_area_m2": 1.0}
    cpu = fl.prepare_floating_column(edges, theta_residual=0.05, theta_field_capacity=0.2, theta_saturation=0.4, soil_depth_m=depth, active_mask=act, diffusivity_m2_s=np.zeros(L - 1), **common)
    gpu = fl.prepare_floating_column(edges, theta_residual=0.05, theta_field_capacity=0.2, theta_saturation=0.4, soil_depth_m=cp.asarray(depth), active_mask=cp.asarray(act), backend="cuda",
                                     diffusivity_m2_s=cp.zeros(L - 1), **common)
    return cpu, gpu


def _water(col, frac, rng):
    from maple_syrup import floating_soil_layers as fl

    h = np.stack([fl.porous_thickness(col, k) for k in range(col.n_layers)])
    return h * 0.4 * frac * rng.uniform(0.9, 1.0, h.shape)


@pytest.mark.parametrize("bottom", ["open", "closed"])
def test_selected_reconcile_matches_cpu_oracle_on_a_real_device(bottom):
    from maple_syrup import floating_soil_layers as fl

    cp = _cp()
    H = 0.6
    old_d = np.array([0.0, 0.05, 0.3, 0.45, H, float(np.nextafter(H, 1.0)), 0.9, 0.12])
    new_d = np.array([0.0, 0.05, 0.25, 0.45, float(np.nextafter(H, 1.0)), H, 0.9, 0.2])
    cpu, gpu = _pair(cp, [0.0, 0.1, 0.3, H], old_d, bottom)
    cv, gv = fl.floating_geometry_view(cpu, new_d), fl.floating_geometry_view(gpu, cp.asarray(new_d))
    assert gv.identity_kind == "geometry_view_token" and gv.parent_sha256 == gpu.sha256
    w = _water(cpu, 0.97, np.random.default_rng(4))
    sel = np.asarray(cpu._maps["bottom_open"]) != np.asarray(cv._maps["bottom_open"])
    for k in range(3):
        sel |= fl.porous_thickness(cpu, k) != fl.porous_thickness(cv, k)
    want = fl.reconcile_floating_selected(cpu, fl.floating_continued_state(cpu, w), cv, np.ascontiguousarray(sel))
    got = fl.reconcile_floating_selected(gpu, fl.floating_continued_state(gpu, cp.asarray(w)), gv, cp.asarray(np.ascontiguousarray(sel)))
    gw = got.state.water_m.get()
    np.testing.assert_allclose(gw, np.asarray(want.state.water_m), rtol=2e-12, atol=1e-14)
    assert gw[:, ~sel].tobytes() == w[:, ~sel].tobytes()
    assert got.receipt.proves_complete_geometry_refresh is False


def test_view_payload_plateau_and_one_packet_at_large_L_and_N():
    from maple_syrup import floating_soil_layers as fl

    cp = _cp()
    L, N = 65, 50003
    edges = np.concatenate([[0.0], np.cumsum(np.full(L, 0.01))])
    depth = np.random.default_rng(0).uniform(0.0, 0.9, N)
    _cpu, gpu = _pair(cp, edges, depth)
    pool = cp.get_default_memory_pool()
    views = []
    used = []
    for i in range(8):
        views.append(fl.floating_geometry_view(gpu, cp.asarray(np.clip(depth + 0.001 * i, 0, 0.95))))
        views = views[-1:]  # no retained chain
        used.append(pool.used_bytes())
    assert max(used) - min(used) <= 16 * N  # the live payload plateaus (about one owned depth, the flag array and the packet) for L = 65, N = 50003


def test_matrix_and_preferential_rebind_share_device_arrays():
    from maple_syrup import floating_matrix_infiltration as fm
    from maple_syrup import floating_preferential_infiltration as fpi
    from maple_syrup import floating_soil_layers as fl

    cp = _cp()
    _cpu, gpu = _pair(cp, [0.0, 0.1, 0.3, 0.6], np.full(16, 0.45))
    op = fm.prepare_floating_matrix(gpu, conductivity_m_s=2e-6, suction_m=0.1, vegetation_factor=0.7)
    hist = fm.floating_matrix_history(op, wetting_m=0.01, cumulative_intake_m=0.02)
    pref = fpi.prepare_floating_preferential(gpu, a3=0.5, max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([0.0, 1.0, 1.0]))
    view = fl.floating_geometry_view(gpu, cp.asarray(np.full(16, 0.40)))
    op2, h2 = fm.rebind_matrix_geometry_view(op, view, hist)
    p2 = fpi.rebind_preferential_geometry_view(pref, view)
    assert h2.wetting_m is hist.wetting_m and all(a is b for a, b in zip(op2._law, op._law, strict=True)) and all(a is b for a, b in zip(p2._law, pref._law, strict=True))


def test_invalid_old_unselected_state_with_relaxed_new_capacity_is_refused_on_a_real_device():
    from maple_syrup import floating_soil_layers as fl

    cp = _cp()
    edges = np.array([0.0, 0.1, 0.3])
    common = {"theta_residual": 0.05, "theta_field_capacity": 0.3, "theta_saturation": 0.5, "diffusivity_m2_s": cp.zeros(1), "theta_evaporation_stress": 0.2, "cell_area_m2": 1.0, "backend": "cuda"}
    old = fl.prepare_floating_column(edges, soil_depth_m=cp.asarray(np.array([0.1, 0.1])), active_mask=cp.ones(2, dtype=bool), **common)  # old capacity 0.05 m
    new = fl.floating_geometry_view(old, cp.asarray(np.array([0.2, 0.1])))  # new capacity 0.1 m at cell 0
    sel = cp.asarray(np.array([True, False]))
    ok = fl.reconcile_floating_selected(old, fl.floating_continued_state(old, cp.asarray(np.array([[0.04, 0.04], [0.0, 0.0]]))), new, sel)
    assert ok.receipt.n_selected == 1
    st = fl.floating_continued_state(old, cp.asarray(np.array([[0.04, 0.04], [0.0, 0.0]])))
    st.water_m[0, 0] = 0.08  # device water raised beyond the OLD capacity at the UNSELECTED cell
    before = st.water_m.get().copy()
    with pytest.raises(fl.FloatingSoilError, match="OLD state violates"):
        fl.reconcile_floating_selected(old, st, new, sel)
    assert st.water_m.get().tobytes() == before.tobytes()
    # a valid old parent, incomplete mask, unselected cell invalid under the NEW capacity: still refused
    sh = fl.prepare_floating_column(edges, soil_depth_m=cp.asarray(np.array([0.2, 0.2])), active_mask=cp.ones(2, dtype=bool), **common)
    sv = fl.floating_geometry_view(sh, cp.asarray(np.array([0.15, 0.2])))
    with pytest.raises(fl.FloatingSoilError):
        fl.reconcile_floating_selected(sh, fl.floating_continued_state(sh, cp.asarray(np.array([[0.02, 0.02], [0.045, 0.0]]))), sv, cp.asarray(np.array([False, False])))
