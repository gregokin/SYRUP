"""D3: CUDA official geometry view / matrix and preferential rebinding / SELECTED zero-time reconciliation. CPU oracle tests plus HOST execution of the rendered CUDA source through the plumbing double: NOT hardware
evidence (the gated real definitions are in test_gpu_definitions.py). Author-written."""
from __future__ import annotations

import numpy as np
import pytest
from builders import pair
from conftest import needs_host_cuda
from hostd2 import fake

from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers as fl
from maple_syrup import subsurface_cuda_floating as cf

EDGES = [0.0, 0.1, 0.3, 0.6]
H = 0.6
ND = float(np.nextafter(H, 1.0))
FEW = [0.0, 0.05, 0.3, 0.45, H, ND, 0.9, 0.12]  # rock, partial, mid, partial, D == H, D just above H, deep, thin
NEW = [0.0, 0.05, 0.25, 0.45, ND, H, 0.9, 0.2]  # slabs change at cells 2 (below H), 4/5 (open flag flips), 7; others equal


def water_for(col, frac, rng):
    h = np.stack([fl.porous_thickness(col, k) for k in range(col.n_layers)])
    ts = np.broadcast_to(np.asarray(col._maps["theta_saturation"]).reshape(-1, 1) if np.asarray(col._maps["theta_saturation"]).ndim == 1 else col._maps["theta_saturation"], h.shape)
    return h * ts * frac * rng.uniform(0.9, 1.0, h.shape)


def changed_cells(old, new):
    """The owner's complete selection rule of this suite (slab thicknesses or strict bottom flag differ)."""
    sel = np.asarray(old._maps["bottom_open"]) != np.asarray(new._maps["bottom_open"])
    for k in range(old.n_layers):
        sel |= fl.porous_thickness(old, k) != fl.porous_thickness(new, k)
    return sel & np.asarray(old._maps["active"])


def cpu_pair(bottom="open", edges=EDGES):
    e = np.asarray(edges)
    old = fl.prepare_floating_column(e, theta_residual=0.05, theta_field_capacity=0.2, theta_saturation=0.4, soil_depth_m=np.array(FEW), active_mask=np.ones(len(FEW), bool), diffusivity_m2_s=np.zeros(len(e) - 2),
                                     theta_evaporation_stress=0.2, bottom_drainage=bottom, cell_area_m2=1.0)
    return old, fl.floating_geometry_view(old, np.array(NEW))


# ---- CPU: the selected operation against the full D1 body ----------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bottom", ["open", "closed"])
def test_selected_equals_full_reconcile_on_the_selected_columns_and_leaves_the_rest_exact(bottom):
    old, new = cpu_pair(bottom)
    rng = np.random.default_rng(3)
    w = water_for(old, 0.97, rng)  # near saturation: any shrink of capacity returns/export water
    w[:, 6] = old._maps["theta_field_capacity"] * 0 + 1.2 * np.stack([fl.porous_thickness(old, k) for k in range(3)])[:, 6] * 0.2  # an UNSELECTED above-FC column (deep, unchanged)
    state = fl.floating_continued_state(old, w)
    sel = changed_cells(old, new)
    assert sel.any() and not sel[6]
    full = fl.reconcile_floating(old, state, new)
    got = fl.reconcile_floating_selected(old, state, new, np.ascontiguousarray(sel))
    gw, fw = np.asarray(got.state.water_m), np.asarray(full.state.water_m)
    assert gw[:, sel].tobytes() == fw[:, sel].tobytes()  # the SAME D1 body on the selected columns
    assert gw[:, ~sel].tobytes() == w[:, ~sel].tobytes()  # unselected: bit for bit
    for a in (got.percolation_m, got.surface_return_m, got.bottom_export_m):
        assert not np.asarray(a)[..., ~sel].any()  # zero flux outside the selection
    assert not np.array_equal(fw[:, 6], w[:, 6])  # control: the FULL reconcile does drain that unselected above-FC column; the selected one must not
    r = got.receipt
    assert r.operation == fl.SELECTED_RECONCILE_TAG and r.n_selected == int(sel.sum()) and r.proves_complete_geometry_refresh is False and r.selection_identity.startswith("selection_token:")
    assert abs(r.residual_m3) <= r.bound_m3 and r.n_terms == 4 * 3 * r.n_selected + 4
    assert np.asarray(got.surface_return_m).sum() + np.asarray(got.bottom_export_m).sum() > 0.0  # a real physical case, not a no-op
    # the old full API is unchanged (the identical call twice)
    assert np.asarray(fl.reconcile_floating(old, state, new).state.water_m).tobytes() == fw.tobytes()


def test_selected_refusals_are_typed_and_publish_nothing():
    old, new = cpu_pair()
    state = fl.floating_continued_state(old, water_for(old, 0.5, np.random.default_rng(1)))
    n = len(FEW)
    with pytest.raises(fl.FloatingSoilError, match="bool"):
        fl.reconcile_floating_selected(old, state, new, np.ones(n, np.uint8))
    with pytest.raises(fl.FloatingSoilError, match="shape"):
        fl.reconcile_floating_selected(old, state, new, np.ones(n + 1, bool))
    inactive = fl.prepare_floating_column(np.asarray(EDGES), theta_residual=0.05, theta_field_capacity=0.2, theta_saturation=0.4, soil_depth_m=np.array(FEW), active_mask=np.arange(n) != 3,
                                          diffusivity_m2_s=np.zeros(2), theta_evaporation_stress=0.2, cell_area_m2=1.0)
    st2 = fl.floating_continued_state(inactive, water_for(inactive, 0.5, np.random.default_rng(1)) * (np.arange(n) != 3))
    with pytest.raises(fl.FloatingSoilError, match="subset"):
        fl.reconcile_floating_selected(inactive, st2, fl.floating_geometry_view(inactive, np.array(NEW)), np.ones(n, bool))
    other = fl.prepare_floating_column(np.asarray(EDGES), theta_residual=0.05, theta_field_capacity=0.2, theta_saturation=0.41, soil_depth_m=np.array(FEW), active_mask=np.ones(n, bool),
                                       diffusivity_m2_s=np.zeros(2), theta_evaporation_stress=0.2, cell_area_m2=1.0)  # different content => a different prepared root
    with pytest.raises(fl.FloatingSoilError, match="SAME prepared root"):
        fl.reconcile_floating_selected(old, state, fl.floating_geometry_view(other, np.array(NEW)), np.zeros(n, bool))
    # an UNSELECTED column that is invalid under the new geometry is a typed refusal of the whole candidate (the selection was incomplete)
    w = water_for(old, 0.99, np.random.default_rng(2))
    with pytest.raises(fl.FloatingSoilError):
        fl.reconcile_floating_selected(old, fl.floating_continued_state(old, w), new, np.zeros(n, bool))


def test_default_zero_reader_refusal_and_old_form_stay():
    old, _ = cpu_pair()
    with pytest.raises(fl.FloatingMemoryError):
        fl.floating_geometry_view(old, np.array(NEW), available_bytes_fn=lambda: 0)
    assert fl.floating_geometry_view(old, np.array(NEW), available_bytes_fn=lambda: 10**12).identity_kind == "geometry_view_token"


# ---- CUDA through the host-compiled rendered source ----------------------------------------------------------------------------------------------------------------------------------------------------------
pytestmark_cuda = needs_host_cuda


def cuda_cols(cp, bottom="open"):
    edges = np.asarray(EDGES)
    return pair(cp, edges, np.array(FEW), tr=0.05, tf=0.2, ts=0.4, bottom=bottom, stress=0.2)


@pytestmark_cuda
def test_cuda_view_shares_static_maps_owns_depth_and_flags_and_reads_one_packet(cp):
    cpu, gpu = cuda_cols(cp)
    cp.count.__init__()
    cp.allocs.clear()
    view = fl.floating_geometry_view(gpu, fake(np.array(NEW)))
    assert (cp.count.d2h, cp.count.d2h_sizes, cp.count.h2d) == (1, [8 * cf.CHECK_WORDS], 0)  # ONE bounded validation packet; no download of D, W, maps or history; no upload
    assert sorted(cp.allocs) == sorted([8 * len(FEW), len(FEW), 8 * cf.CHECK_WORDS])  # exactly the owned depth, the flags and the packet
    for k in ("theta_residual", "theta_field_capacity", "theta_saturation", "interface", "active", "edges", "sep_nominal"):
        assert view._maps[k] is gpu._maps[k]
    assert view._maps["soil_depth"] is not gpu._maps["soil_depth"] and view._maps["bottom_open"] is not gpu._maps["bottom_open"]
    cpu_view = fl.floating_geometry_view(cpu, np.array(NEW))
    assert np.array_equal(np.asarray(view._maps["bottom_open"]).astype(bool), np.asarray(cpu_view._maps["bottom_open"]))  # strict D > H, [D == H, nextafter] included
    assert np.array_equal(np.asarray(view._maps["soil_depth"]), np.array(NEW))
    assert view.identity_kind == "geometry_view_token" and view.parent_sha256 == gpu.sha256 and view.device_id == gpu.device_id and view.sha256 != gpu.sha256
    again = fl.floating_geometry_view(view, fake(np.array(FEW)))
    assert again.parent_sha256 == gpu.sha256  # a view of a view binds the ROOT: no parent chain


@pytestmark_cuda
def test_cuda_view_refusals_precede_reads_and_allocations(cp):
    _cpu, gpu = cuda_cols(cp)
    calls = []
    n_alloc = len(cp.allocs)
    cp.dev["id"] = 1
    with pytest.raises(fl.FloatingSoilError, match="current device"):
        fl.floating_geometry_view(gpu, fake(np.array(NEW)), available_bytes_fn=lambda: (calls.append(1), 1 << 40)[1])
    cp.dev["id"] = 0
    assert not calls and len(cp.allocs) == n_alloc  # the device gate preceded the reader and every allocation
    with pytest.raises(fl.FloatingMemoryError):
        fl.floating_geometry_view(gpu, fake(np.array(NEW)), available_bytes_fn=lambda: 0)
    assert len(cp.allocs) == n_alloc  # memory refusal before the first owned allocation
    need = 8 * len(FEW) + len(FEW) + 8 * cf.CHECK_WORDS
    assert fl.floating_geometry_view(gpu, fake(np.array(NEW)), available_bytes_fn=lambda: need + 7, reserve_bytes=7)  # the reserve is counted ONCE
    with pytest.raises(fl.FloatingMemoryError):
        fl.floating_geometry_view(gpu, fake(np.array(NEW)), available_bytes_fn=lambda: need + 6, reserve_bytes=7)
    bad = np.array(NEW)
    bad[2] = -1.0
    with pytest.raises(fl.FloatingSoilError, match="soil depth"):
        fl.floating_geometry_view(gpu, fake(bad))
    bad[2] = np.nan
    with pytest.raises(fl.FloatingSoilError):
        fl.floating_geometry_view(gpu, fake(bad))
    with pytest.raises(fl.FloatingSoilError, match="cupy"):
        fl.floating_geometry_view(gpu, np.array(NEW))  # a host array is never coerced
    with pytest.raises(fl.FloatingSoilError, match="device"):
        fl.floating_geometry_view(gpu, fake(np.array(NEW), device=1))
    with pytest.raises(fl.FloatingSoilError):
        fl.floating_geometry_view(gpu, fake(np.array(NEW)[:-1]))


@pytestmark_cuda
def test_cuda_view_payload_plateaus_at_large_L_and_N(cp):
    L, N = 65, 5003
    edges = np.concatenate([[0.0], np.cumsum(np.full(L, 0.01))])
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.0, 0.9, N)
    _cpu, gpu = pair(cp, edges, depth, tr=0.05, tf=0.2, ts=0.4)
    sizes = []
    for i in range(6):
        cp.allocs.clear()
        v = fl.floating_geometry_view(gpu, fake(np.clip(depth + 0.001 * i, 0, 0.95)))
        sizes.append(sum(cp.allocs))
        del v
    assert len(set(sizes)) == 1 and sizes[0] == 8 * N + N + 8 * cf.CHECK_WORDS  # O(N), independent of L: no L*N map is copied or re-prepared, no L cap, no growth


@pytestmark_cuda
def test_cuda_matrix_and_preferential_rebind_share_device_arrays_and_transfer_nothing(cp):
    cpu, gpu = cuda_cols(cp)
    n = len(FEW)
    op = fm.prepare_floating_matrix(gpu, conductivity_m_s=fake(np.full(n, 2e-6)), suction_m=0.1, vegetation_factor=0.7)
    hist = fm.floating_matrix_history(op, wetting_m=fake(np.linspace(0.0, 0.02, n)), cumulative_intake_m=fake(np.linspace(0.0, 0.05, n)))
    pref = fpi.prepare_floating_preferential(gpu, a3=0.5, max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([0.0, 1.0, 1.0]))
    view = fl.floating_geometry_view(gpu, fake(np.array(NEW)))
    cp.count.__init__()
    cp.allocs.clear()
    op2, hist2 = fm.rebind_matrix_geometry_view(op, view, hist)
    pref2 = fpi.rebind_preferential_geometry_view(pref, view)
    assert (cp.count.d2h, cp.count.h2d, cp.allocs) == (0, 0, [])  # no copy, download, upload or allocation
    assert all(a is b for a, b in zip(op2._law, op._law, strict=True)) and all(a is b for a, b in zip(pref2._law, pref._law, strict=True))
    assert hist2.wetting_m is hist.wetting_m and hist2.cumulative_intake_m is hist.cumulative_intake_m  # J and the cumulative intake bit for bit (shared)
    assert (hist2.namespace, hist2.device_id, op2.device_id, pref2.device_id) == ("cupy", 0, 0, 0) and op2.column is view and pref2.column is view
    assert op2.sha256 != op.sha256 and hist2.operator_sha256 == op2.sha256 and pref2.n_layers == pref.n_layers
    # wrong device / foreign root are refused before any identity work
    cp.dev["id"] = 1
    with pytest.raises(fl.FloatingSoilError, match="current device"):
        fm.rebind_matrix_geometry_view(op, view, hist)
    with pytest.raises(fl.FloatingSoilError, match="current device"):
        fpi.rebind_preferential_geometry_view(pref, view)
    cp.dev["id"] = 0
    _c2, gpu2 = pair(cp, np.asarray(EDGES), np.array(FEW), tr=0.05, tf=0.2, ts=0.41, stress=0.2)
    foreign = fl.floating_geometry_view(gpu2, fake(np.array(NEW)))
    with pytest.raises(fl.FloatingSoilError, match="geometry view"):
        fm.rebind_matrix_geometry_view(op, foreign, hist)
    with pytest.raises(fl.FloatingSoilError, match="geometry view"):
        fpi.rebind_preferential_geometry_view(pref, foreign)
    # the rebound operator runs the SAME step as the CPU rebound operator
    rng = np.random.default_rng(5)
    cview = fl.floating_geometry_view(cpu, np.array(NEW))
    cop = fm.prepare_floating_matrix(cpu, conductivity_m_s=np.full(n, 2e-6), suction_m=0.1, vegetation_factor=0.7)
    chist = fm.floating_matrix_history(cop, wetting_m=np.linspace(0.0, 0.02, n), cumulative_intake_m=np.linspace(0.0, 0.05, n))
    cop2, chist2 = fm.rebind_matrix_geometry_view(cop, cview, chist)
    w = water_for(cview, 0.3, rng)
    pool, standing = rng.uniform(0, 0.02, n), np.zeros(n)
    rate = pool / 600.0
    rc = fm.floating_matrix_step(cop2, pool, standing, rate, fl.floating_continued_state(cview, w), chist2, 600.0)
    rg = fm.floating_matrix_step(op2, fake(pool), fake(standing), fake(rate), fl.floating_continued_state(view, fake(w)), hist2, 600.0)
    for a, b in ((rc.state.water_m, rg.state.water_m), (rc.history.wetting_m, rg.history.wetting_m), (rc.intake_m, rg.intake_m)):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=2e-12, atol=1e-14)


@pytestmark_cuda
@pytest.mark.parametrize("bottom", ["open", "closed"])
def test_cuda_selected_reconcile_equals_cpu_and_keeps_unselected_bits_with_bounded_traffic(cp, bottom):
    cpu, gpu = cuda_cols(cp, bottom)
    cview, gview = fl.floating_geometry_view(cpu, np.array(NEW)), fl.floating_geometry_view(gpu, fake(np.array(NEW)))
    rng = np.random.default_rng(11)
    w = water_for(cpu, 0.97, rng)
    w[:, 6] = 1.2 * 0.2 * np.stack([fl.porous_thickness(cpu, k) for k in range(3)])[:, 6]  # unselected above-FC column
    sel = changed_cells(cpu, cview)
    c_state, g_state = fl.floating_continued_state(cpu, w), fl.floating_continued_state(gpu, fake(w))
    want = fl.reconcile_floating_selected(cpu, c_state, cview, np.ascontiguousarray(sel))
    cp.count.__init__()
    cp.allocs.clear()
    got = fl.reconcile_floating_selected(gpu, g_state, gview, fake(np.ascontiguousarray(sel)))
    assert cp.count.d2h_sizes == [8, 8, 8 * cf.CHECK_WORDS, 8 * cf.RECONCILE_PACKET_WORDS, 8 * cf.CHECK_WORDS] and cp.count.h2d == 0  # two selection counts, the OLD full-domain check, the reconcile packet, the output state check: nothing else (v2 traffic)
    gw = np.asarray(got.state.water_m)
    np.testing.assert_allclose(gw, np.asarray(want.state.water_m), rtol=2e-12, atol=1e-14)
    assert gw[:, ~sel].tobytes() == w[:, ~sel].tobytes()  # unselected bit exact on the device
    for a in (got.percolation_m, got.surface_return_m, got.bottom_export_m):
        assert not np.asarray(a)[..., ~sel].any()
    for name in ("storage_before_m3", "storage_after_m3", "surface_return_m3", "bottom_export_m3", "moved_percolation_m3"):
        assert getattr(got.receipt, name) == pytest.approx(getattr(want.receipt, name), rel=2e-12, abs=1e-14)
    assert got.receipt.identity_kind == "capability_token" and got.receipt.proves_complete_geometry_refresh is False and got.receipt.n_selected == int(sel.sum())
    # the old full CUDA reconcile keeps its single 56-byte packet
    cp.count.__init__()
    fl.reconcile_floating(gpu, g_state, gview)
    assert cp.count.d2h_sizes == [8 * cf.RECONCILE_PACKET_WORDS]


@pytestmark_cuda
def test_cuda_selected_refusals_precede_allocation_and_kernel(cp):
    _cpu, gpu = cuda_cols(cp)
    gview = fl.floating_geometry_view(gpu, fake(np.array(NEW)))
    rng = np.random.default_rng(2)
    n = len(FEW)
    st = fl.floating_continued_state(gpu, fake(water_for(_cpu, 0.5, rng)))
    n_alloc, n_launch = len(cp.allocs), len(cp.launches)
    with pytest.raises(fl.FloatingMemoryError):
        fl.reconcile_floating_selected(gpu, st, gview, fake(np.ones(n, bool)), available_bytes_fn=lambda: 0)
    assert (len(cp.allocs), len(cp.launches)) == (n_alloc, n_launch)
    with pytest.raises(fl.FloatingSoilError, match="bool cupy"):
        fl.reconcile_floating_selected(gpu, st, gview, np.ones(n, bool))
    cp.dev["id"] = 1
    with pytest.raises(fl.FloatingSoilError, match="current device"):
        fl.reconcile_floating_selected(gpu, st, gview, fake(np.ones(n, bool)))
    cp.dev["id"] = 0
    _c2, g2 = pair(cp, np.asarray(EDGES), np.array(FEW), tr=0.05, tf=0.2, ts=0.41, stress=0.2)
    with pytest.raises(fl.FloatingSoilError, match="SAME prepared root"):
        fl.reconcile_floating_selected(gpu, st, fl.floating_geometry_view(g2, fake(np.array(NEW))), fake(np.zeros(n, bool)))
    # an invalid old parent is refused (device bits 20/21), nothing minted
    bad_w = water_for(_cpu, 0.5, rng)
    bad_w[0, 2] = -1.0
    with pytest.raises(fl.FloatingSoilError):
        fl.reconcile_floating_selected(gpu, fl.floating_continued_state(gpu, fake(bad_w)), gview, fake(np.ones(n, bool)))
    # unselected column invalid under the new geometry: refused by the full-domain output check
    w = water_for(_cpu, 0.99, rng)
    with pytest.raises(fl.FloatingSoilError):
        fl.reconcile_floating_selected(gpu, fl.floating_continued_state(gpu, fake(w)), gview, fake(np.zeros(n, bool)))


def _relaxed_pair(cp=None):
    """old D = 0.1 (capacity 0.05 m with theta_sat 0.5 over h = 0.1), new D = 0.2: the NEW capacity is larger, so an old water of 0.08 m is invalid for its OLD owner but fits the new geometry; the cell is UNSELECTED."""
    edges = np.array([0.0, 0.1, 0.3])

    def build(dev):
        mk = (lambda x: fake(x)) if dev else (lambda x: x)
        col = fl.prepare_floating_column(edges, theta_residual=mk(np.array(0.05)) if False else 0.05, theta_field_capacity=0.3, theta_saturation=0.5, soil_depth_m=mk(np.array([0.1, 0.1])),
                                         active_mask=mk(np.ones(2, bool)), diffusivity_m2_s=mk(np.zeros(1)) if dev else np.zeros(1), theta_evaporation_stress=0.2, cell_area_m2=1.0, **({"backend": "cuda"} if dev else {}))
        return col, fl.floating_geometry_view(col, mk(np.array([0.2, 0.1])))

    return build


def test_cpu_positive_control_invalid_unselected_old_state_is_refused():
    old, new = _relaxed_pair()(False)
    good = fl.floating_continued_state(old, np.array([[0.04, 0.04], [0.0, 0.0]]))
    assert fl.reconcile_floating_selected(old, good, new, np.array([True, False])).receipt.n_selected == 1  # a valid old parent works
    # the invalid-old water 0.08 > old capacity 0.05 cannot even form a continued state of the OLD owner on the CPU; the CPU refusal is at the state constructor
    with pytest.raises(fl.FloatingSoilError):
        fl.floating_continued_state(old, np.array([[0.08, 0.04], [0.0, 0.0]]))


@pytestmark_cuda
def test_cuda_invalid_old_unselected_cell_with_relaxed_new_capacity_is_refused_and_publishes_nothing(cp):
    old, new = _relaxed_pair()(True)
    good_w = np.array([[0.04, 0.04], [0.0, 0.0]])
    ok = fl.reconcile_floating_selected(old, fl.floating_continued_state(old, fake(good_w)), new, fake(np.array([True, False])))
    assert ok.receipt.n_selected == 1  # the valid old parent with an incomplete mask is accepted
    # a GENUINE continued state of the old owner, then the device water is later raised (0.08 > old capacity 0.05) at the UNSELECTED cell: invalid OLD input
    st = fl.floating_continued_state(old, fake(good_w.copy()))
    st.water_m[0, 0] = 0.08
    before = np.array(st.water_m)
    n_launch = len(cp.launches)
    cp.count.__init__()
    with pytest.raises(fl.FloatingSoilError, match="OLD state violates"):
        fl.reconcile_floating_selected(old, st, new, fake(np.array([True, False])))
    assert cp.count.d2h_sizes == [8, 8, 8 * cf.CHECK_WORDS]  # the refusal happens right after the old-domain check: no reconcile kernel, packet or output check
    assert [k[0] for k in cp.launches[n_launch:]] == ["maple_syrup_floating_state_check"]
    assert np.array_equal(np.asarray(st.water_m), before)  # the input is unchanged
    # a valid old parent with an INCOMPLETE mask whose unselected cell is invalid under the NEW capacity is still refused (output full-domain check)
    edges = np.array([0.0, 0.1, 0.3])
    shrink = fl.prepare_floating_column(edges, theta_residual=0.05, theta_field_capacity=0.3, theta_saturation=0.5, soil_depth_m=fake(np.array([0.2, 0.2])), active_mask=fake(np.ones(2, bool)),
                                        diffusivity_m2_s=fake(np.zeros(1)), theta_evaporation_stress=0.2, cell_area_m2=1.0, backend="cuda")
    sv = fl.floating_geometry_view(shrink, fake(np.array([0.15, 0.2])))  # cell 0: layer-1 slab 0.1 -> 0.05 m (capacity 0.05 -> 0.025 m)
    w = np.array([[0.02, 0.02], [0.045, 0.0]])
    with pytest.raises(fl.FloatingSoilError, match="unselected|selected candidate"):
        fl.reconcile_floating_selected(shrink, fl.floating_continued_state(shrink, fake(w)), sv, fake(np.array([False, False])))
    # guard ordering: the wrong device refuses before any read; a host mask refuses before any kernel
    cp.dev["id"] = 1
    n_launch = len(cp.launches)
    with pytest.raises(fl.FloatingSoilError, match="current device"):
        fl.reconcile_floating_selected(old, st, new, fake(np.array([True, False])))
    cp.dev["id"] = 0
    with pytest.raises(fl.FloatingSoilError, match="bool cupy"):
        fl.reconcile_floating_selected(old, st, new, np.array([True, False]))
    assert len(cp.launches) == n_launch
