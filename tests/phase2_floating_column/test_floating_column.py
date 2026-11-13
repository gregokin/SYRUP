"""D1 floating porous column: geometry, full-soil equivalence with the accepted spatial owner (matching the lower-boundary flag), per-cell open/closed/all-rock mixtures, state policies, inactive placeholders,
pulse step, reconciliation identities, typed refusals, memory. Author-written; NumPy and Numba."""
from __future__ import annotations

import copy
import pickle
import tracemalloc

import numpy as np
import pytest
from conftest import BACKENDS, BIG, backend, column, zero_forcing  # noqa: F401

from maple_syrup import floating_soil_layers as fl
from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss

EDGES = np.array([0.0, 0.1, 0.3, 0.6])


def rand_params(rng, L, N):
    tf = rng.uniform(0.1, 0.25, (L, N))
    ts = tf + rng.uniform(0.05, 0.2, (L, N))
    tr = rng.uniform(0.0, 0.05, (L, N))
    return tr, tf, ts


def make(edges, depth, *, rng=None, bottom="open", mode="tietjen_donor_depth", backend="numpy", cells=None, active=None):
    L, N = len(edges) - 1, len(depth)
    rng = rng or np.random.default_rng(3)
    tr, tf, ts = rand_params(rng, L, N)
    kw = dict(interface_conductivity_m_s=rng.uniform(1e-7, 1e-5, (L - 1, N)), diffusion_coefficient=0.7) if mode != "constant" else dict(diffusivity_m2_s=rng.uniform(1e-8, 1e-6, (L - 1, N)))
    col = fl.prepare_floating_column(np.asarray(edges), theta_residual=tr, theta_field_capacity=tf, theta_saturation=ts, theta_evaporation_stress=0.3, diffusion_mode=mode, bottom_drainage=bottom,
                                     soil_depth_m=np.asarray(depth, float), active_mask=np.ones(N, bool) if active is None else active, cell_area_m2=2.0, backend=backend, **kw)
    return col, (tr, tf, ts, kw)


def same(x, y, backend, what=None):
    """NumPy runs the unchanged shared body: bitwise. Numba differs only through libm expm1: the declared bound."""
    if backend == "numpy":
        assert np.array_equal(x, y), what
    else:
        np.testing.assert_allclose(y, x, rtol=2e-12, atol=1e-14, err_msg=str(what))


def fill(col, rng, frac=0.8):
    h = np.stack([fl.porous_thickness(col, k) for k in range(col.n_layers)])
    ts = col._maps["theta_saturation"]
    ts = np.broadcast_to(ts.reshape(-1, 1) if ts.ndim == 1 and ts.shape[0] == col.n_layers else ts, h.shape)
    return h * ts * rng.uniform(0.05, frac, h.shape)


def fill_fc(col, rng):
    h = np.stack([fl.porous_thickness(col, k) for k in range(col.n_layers)])
    tf = col._maps["theta_field_capacity"]
    return h * np.broadcast_to(tf.reshape(-1, 1) if tf.ndim == 1 else tf, h.shape) * rng.uniform(0.1, 1.0, h.shape)


# ---- geometry --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("depth", [0.0, 1e-12, 0.05, 0.1, 0.1000000001, 0.3, 0.45, 0.6, 0.6000000001, 5.0])
def test_porous_thickness_formula_exact_edges_and_slivers(depth):
    col, _ = make(EDGES, [depth])
    h = [float(fl.porous_thickness(col, k)[0]) for k in range(3)]
    want = [max(0.0, min(depth, EDGES[k + 1]) - EDGES[k]) for k in range(3)]
    assert h == want
    full = [EDGES[k + 1] - EDGES[k] for k in range(3)]
    for k in range(3):
        if depth >= EDGES[k + 1]:
            assert h[k] == full[k]  # a full slab has exactly the nominal expression (bit-identical dz)


def test_no_permanent_LN_maps_and_depth_is_owned_O_of_N():
    col, _ = make(EDGES, np.linspace(0, 1, 50))
    big = [n for n, a in col._maps.items() if a.size >= col.n_layers * col.n_cells and a.ndim == 2]
    assert set(big) <= {"theta_residual", "theta_field_capacity", "theta_saturation", "interface"}  # only RAW property maps are (L, N); no thickness/capacity/conductance map
    assert col._maps["soil_depth"].shape == (50,) and col._maps["edges"].shape == (4,) and col._maps["dz"].shape == (3,)
    assert all(not a.flags.writeable for a in col._maps.values())


def test_bottom_open_is_strict_above_rock():
    col, _ = make(EDGES, [0.0, 0.5, 0.6, 0.6000000001, 1.0], bottom="open")
    assert col._maps["bottom_open"].tolist() == [False, False, False, True, True]  # D == H is physically closed rock
    closed, _ = make(EDGES, [1.0], bottom="closed")
    assert not closed._maps["bottom_open"][0]


# ---- full soil == the accepted spatial owner (matching the lower-boundary flag) ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("bottom,depth,spatial_bottom", [("open", 5.0, "open"), ("closed", 5.0, "closed"), ("open", 0.6, "closed"), ("closed", 0.6, "closed")])
def test_full_soil_is_bit_identical_to_the_spatial_owner(mode, bottom, depth, spatial_bottom, backend):
    rng = np.random.default_rng(11)
    N, L = 37, 3
    col, (tr, tf, ts, kw) = make(EDGES, np.full(N, depth), rng=rng, bottom=bottom, mode=mode, backend=backend)
    spec = sp.spatial_layer_spec(EDGES, theta_residual=tr, theta_field_capacity=tf, theta_saturation=ts, theta_evaporation_stress=0.3, diffusion_mode=mode, bottom_drainage=spatial_bottom, n_cells=N, **kw)
    old = sp.prepare_spatial_column(spec, np.ones(N, bool), 2.0, backend="numpy", memory_budget_bytes=BIG)
    w = fill(col, rng, 1.0)
    added = rng.uniform(0, 2e-3, (L, N)) * 0.3
    pet = rng.uniform(0, 3e-7, N)
    for dt in (0.0, 600.0, 86400.0):
        add = added if dt > 0.0 else np.zeros_like(added)  # both owners refuse added water at dt == 0
        a = ss_step = sp.spatial_step(old, sp.spatial_column_state(old, w, 0.0, memory_budget_bytes=BIG), dt, added_water_m=add, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
        b = fl.floating_step(col, fl.floating_continued_state(col, w), dt, added_water_m=add, potential_evaporation_m_s=pet)
        for f in ("surface_return_m", "drainage_m", "evaporation_m", "evaporation_unmet_m", "percolation_m"):
            same(getattr(a, f), getattr(b, f), backend, (f, dt))
        same(a.state.water_m, b.state.water_m, backend, dt)
        same(a.diffusion_m, b.diffusion_m, backend, dt)
    del ss_step


def test_full_soil_pulse_step_equals_the_spatial_owner():
    rng = np.random.default_rng(12)
    N = 20
    col, (tr, tf, ts, kw) = make(EDGES, np.full(N, 9.0), rng=rng, bottom="open")
    spec = sp.spatial_layer_spec(EDGES, theta_residual=tr, theta_field_capacity=tf, theta_saturation=ts, theta_evaporation_stress=0.3, diffusion_mode="tietjen_donor_depth", bottom_drainage="open", n_cells=N, **kw)
    old = sp.prepare_spatial_column(spec, np.ones(N, bool), 2.0, backend="numpy", memory_budget_bytes=BIG)
    w = fill(col, rng, 0.4)
    pet = rng.uniform(0, 3e-7, N)
    a = sp.spatial_step_with_recharge(old, sp.spatial_column_state(old, w, 0.0, memory_budget_bytes=BIG), 86400.0, recharge_depth_m=1e-3, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    b = fl.floating_step_with_recharge(col, fl.floating_continued_state(col, w), 86400.0, recharge_depth_m=1e-3, potential_evaporation_m_s=pet)
    assert np.array_equal(a.state.water_m, b.state.water_m) and np.array_equal(a.drainage_m, b.drainage_m)


# ---- per-cell mixtures ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_mixed_open_closed_rock_cells_equal_single_cell_columns(backend):
    rng = np.random.default_rng(5)
    depth = np.array([0.0, 0.04, 0.1, 0.25, 0.45, 0.6, 0.7, 2.0, 0.3, 0.6000001])
    N, L = depth.size, 3
    tr, tf, ts = rand_params(rng, L, N)
    iface = rng.uniform(1e-7, 1e-5, (L - 1, N))
    cc = np.ascontiguousarray
    mk = lambda cols: fl.prepare_floating_column(EDGES, theta_residual=cc(tr[:, cols]), theta_field_capacity=cc(tf[:, cols]), theta_saturation=cc(ts[:, cols]), interface_conductivity_m_s=cc(iface[:, cols]), diffusion_coefficient=0.7,  # noqa: E731
                                                  theta_evaporation_stress=0.3, diffusion_mode="tietjen_donor_depth", bottom_drainage="open", soil_depth_m=depth[cols].copy(), active_mask=np.ones(len(depth[cols]), bool),
                                                  cell_area_m2=2.0, backend=backend)
    all_cols = mk(slice(None))
    w = fill(all_cols, rng, 1.0)
    added = rng.uniform(0, 3e-3, (L, N)) * 0.2
    pet = rng.uniform(0, 3e-7, N)
    big = fl.floating_step(all_cols, fl.floating_continued_state(all_cols, w), 3600.0, added_water_m=added, potential_evaporation_m_s=pet)
    for c in range(N):
        one = mk(slice(c, c + 1))
        r = fl.floating_step(one, fl.floating_continued_state(one, np.ascontiguousarray(w[:, c:c + 1])), 3600.0, added_water_m=np.ascontiguousarray(added[:, c:c + 1]), potential_evaporation_m_s=pet[c:c + 1].copy())
        assert np.array_equal(big.state.water_m[:, c], r.state.water_m[:, 0]) and big.drainage_m[c] == r.drainage_m[0] and big.surface_return_m[c] == r.surface_return_m[0], c
    exported = big.drainage_m[all_cols._maps["bottom_open"]]
    assert (big.drainage_m[~all_cols._maps["bottom_open"]] == 0.0).all()  # nothing leaves through rock or a closed bottom
    assert exported.size == 3


def test_numba_equals_numpy_on_a_heterogeneous_domain():
    pytest.importorskip("numba")
    rng = np.random.default_rng(6)
    N = 150
    depth = rng.choice([0.0, 0.03, 0.1, 0.2, 0.45, 0.6, 0.9], size=N)
    act = rng.random(N) > 0.15
    cn, _ = make(EDGES, depth, rng=np.random.default_rng(1), backend="numpy", active=act)
    cb, _ = make(EDGES, depth, rng=np.random.default_rng(1), backend="numba", active=act)
    w = fill(cn, rng, 1.0)
    w[:, ~act] = 7.0  # inactive placeholders far above any capacity
    added = rng.uniform(0, 4e-3, (3, N)) * act * 0.2
    pet = rng.uniform(0, 3e-7, N) * act
    a = fl.floating_step(cn, fl.floating_continued_state(cn, w), 7200.0, added_water_m=added, potential_evaporation_m_s=pet)
    b = fl.floating_step(cb, fl.floating_continued_state(cb, w), 7200.0, added_water_m=added, potential_evaporation_m_s=pet)
    for f in ("surface_return_m", "drainage_m", "evaporation_m", "evaporation_unmet_m", "percolation_m"):
        np.testing.assert_allclose(getattr(a, f), getattr(b, f), rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(a.state.water_m, b.state.water_m, rtol=2e-12, atol=1e-14)
    assert np.array_equal(a.state.water_m[:, ~act], w[:, ~act]) and np.array_equal(b.state.water_m[:, ~act], w[:, ~act])  # placeholders preserved bit for bit on both backends


# ---- rock/zero slots, E, query, states -----------------------------------------------------------------------------------------------------------------------------------------------------
def test_all_rock_column_has_zero_E_zero_capacity_and_no_division_warnings(backend):
    col, _ = make(EDGES, [0.0, 0.0], backend=backend)
    st = fl.floating_initial_state(col, np.zeros((3, 2)))  # a wholly-rock initial state is legal
    with np.errstate(all="raise"):
        r = fl.floating_step(col, st, 86400.0, added_water_m=np.zeros((3, 2)), potential_evaporation_m_s=np.full(2, 5e-7))
        theta = fl.floating_moisture(col, r.state)
    assert (r.state.water_m == 0.0).all() and (r.evaporation_m == 0.0).all() and (r.evaporation_unmet_m == 0.0).all() and (theta == 0.0).all()
    assert r.budget.storage_before_m3 == 0.0 == r.budget.storage_after_m3


def test_initial_state_policy_is_strict_and_continuation_allows_saturation(backend):
    col, _ = make(EDGES, [0.2], backend=backend)  # layers: 0.1, 0.1, rock
    ts = col._maps["theta_saturation"]
    sat = np.array([[ts[0, 0] * 0.1], [ts[1, 0] * 0.1], [0.0]])
    with pytest.raises(fl.FloatingSoilError, match="strictly unsaturated"):
        fl.floating_initial_state(col, sat.copy())  # W == saturation is not an unsaturated initial state
    fl.floating_continued_state(col, sat.copy())  # but it is a legal continuation
    below = sat * 0.9
    fl.floating_initial_state(col, below)
    bad = below.copy()
    bad[2, 0] = 1e-12  # rock slots hold exactly 0: no epsilon, no hidden water
    for ctor in (fl.floating_initial_state, fl.floating_continued_state):
        with pytest.raises(fl.FloatingSoilError):
            ctor(col, bad)
    over = sat * 1.01
    with pytest.raises(fl.FloatingSoilError, match="exceeds"):
        fl.floating_continued_state(col, over)


def test_inactive_placeholders_are_preserved_and_never_counted(backend):
    act = np.array([True, False, True])
    col, _ = make(EDGES, [0.5, 0.2, 5.0], backend=backend, active=act)
    rng = np.random.default_rng(9)
    w = fill(col, rng, 0.8)
    w[:, 1] = [3.0, 2.0, 1.0]  # a placeholder far above any capacity and nonzero in a rock slot
    st = fl.floating_continued_state(col, w)
    assert fl.INACTIVE_POLICY in col.summary()["inactive_policy"]
    want_storage = float(w[:, act].sum()) * 2.0
    assert abs(fl.floating_storage_m3(col, st) - want_storage) <= 1e-12
    r = fl.floating_step(col, st, 3600.0, added_water_m=np.zeros((3, 3)), potential_evaporation_m_s=np.zeros(3))
    assert np.array_equal(r.state.water_m[:, 1], w[:, 1]) and r.drainage_m[1] == 0.0 and (r.percolation_m[:, 1] == 0.0).all()
    assert abs(r.budget.storage_before_m3 - want_storage) <= 1e-12  # the placeholder is not in the active budget
    for bad in (-1.0, float("nan")):
        w2 = w.copy()
        w2[0, 1] = bad
        with pytest.raises(fl.FloatingSoilError, match="inactive placeholders"):
            fl.floating_continued_state(col, w2)


def test_moisture_is_actual_porous_theta_and_zero_in_rock():
    col, _ = make(EDGES, [0.15])
    w = np.array([[0.002], [0.002], [0.0]])
    th = fl.floating_moisture(col, fl.floating_continued_state(col, w))
    assert th[0, 0] == 0.002 / 0.1 and th[1, 0] == 0.002 / (0.15 - 0.1) and th[2, 0] == 0.0


def test_pulse_step_fit_failure_is_the_distinct_typed_outcome(backend):
    col, _ = make(EDGES, [0.2], bottom="closed", backend=backend)
    sat = np.array([[col._maps["theta_saturation"][0, 0] * 0.1 * 0.99], [col._maps["theta_saturation"][1, 0] * 0.1 * 0.99], [0.0]])
    st = fl.floating_continued_state(col, sat)
    with pytest.raises(ss.SubsurfaceFitFailure):
        fl.floating_step_with_recharge(col, st, 86400.0, recharge_depth_m=0.5, potential_evaporation_m_s=np.zeros(1))
    ok = fl.floating_step_with_recharge(col, st, 86400.0, recharge_depth_m=1e-6, potential_evaporation_m_s=np.zeros(1))
    assert ok.state.t_s == 86400.0 and abs(ok.budget.residual_m3) <= ok.budget.bound_m3
    with pytest.raises(fl.FloatingSoilError, match="dt_s > 0"):
        fl.floating_step_with_recharge(col, st, 0.0, recharge_depth_m=1e-3, potential_evaporation_m_s=np.zeros(1))


# ---- reconciliation properties -----------------------------------------------------------------------------------------------------------------------------------------------------------
def test_reconciling_to_an_identical_column_moves_nothing_and_is_pure(backend):
    rng = np.random.default_rng(14)
    depth = np.array([0.2, 0.45, 0.6, 2.0])
    col, _ = make(EDGES, depth, rng=np.random.default_rng(2), backend=backend)
    twin, _ = make(EDGES, depth, rng=np.random.default_rng(2), backend=backend)
    w = fill_fc(col, rng)  # a post-gravity state: at/below field capacity (reconciliation moves only water above the NEW FC or above SAT)
    st = fl.floating_continued_state(col, w)
    snap = st.water_m.copy()
    c1 = fl.reconcile_floating(col, st, twin)
    c2 = fl.reconcile_floating(col, st, twin)
    assert np.array_equal(st.water_m, snap)  # inputs never written
    assert (c1.surface_return_m == 0.0).all() and (c1.bottom_export_m == 0.0).all() and np.array_equal(c1.state.water_m, snap) and np.array_equal(c1.state.water_m, c2.state.water_m)
    import dataclasses
    assert dataclasses.replace(c1.receipt, new_state_token="") == dataclasses.replace(c2.receipt, new_state_token="") and c1.receipt.new_state_token != c2.receipt.new_state_token  # a fresh minted generation token per candidate
    assert c1.receipt.identity_kind == "content_sha256" and c1.receipt.old_state_token == st.token and c1.receipt.old_column_sha256 == col.sha256 == c1.receipt.new_column_sha256
    assert c1.state.column_sha256 == twin.sha256 and c1.state.water_m is not st.water_m


def test_reconcile_inactive_cells_are_copied_and_excluded(backend):
    act = np.array([True, False])
    old, _ = make(EDGES, [5.0, 5.0], backend=backend, active=act)
    new, _ = make(EDGES, [0.0, 0.0], backend=backend, active=act, rng=np.random.default_rng(3))
    w = fill(old, np.random.default_rng(4), 0.9)
    w[:, 1] = 9.0
    cand = fl.reconcile_floating(old, fl.floating_continued_state(old, w), new)
    assert np.array_equal(cand.state.water_m[:, 1], w[:, 1]) and cand.surface_return_m[1] == 0.0
    assert abs(cand.surface_return_m[0] * 2.0 - w[:, 0].sum() * 2.0) <= 1e-12 and (cand.state.water_m[:, 0] == 0.0).all()  # all soil eroded: every drop returns to the surface
    assert abs(cand.receipt.surface_return_m3 - w[:, 0].sum() * 2.0) <= 1e-12


def test_reconcile_refuses_mismatched_bindings(backend):
    a, _ = make(EDGES, [1.0, 1.0], backend=backend)
    b, _ = make(EDGES, [1.0, 1.0], backend=backend, active=np.array([True, False]))
    c, _ = make(np.array([0.0, 0.1, 0.3, 0.7]), [1.0, 1.0], backend=backend)
    st = fl.floating_continued_state(a, fill(a, np.random.default_rng(1)))
    for new in (b, c):
        with pytest.raises(fl.FloatingSoilError, match="same layer count"):
            fl.reconcile_floating(a, st, new)
    other, _ = make(EDGES, [1.0, 1.0], backend=backend, rng=np.random.default_rng(99))
    with pytest.raises(fl.FloatingSoilError, match="different floating column"):
        fl.reconcile_floating(other, st, a)  # the state belongs to `a`, not to the supplied old column


# ---- typed refusals, ownership, backends ---------------------------------------------------------------------------------------------------------------------------------------------------
def test_legacy_spatial_and_floating_owners_refuse_each_other(backend):
    col, _ = make(EDGES, [1.0, 1.0], backend=backend)
    st = fl.floating_continued_state(col, fill(col, np.random.default_rng(1)))
    lay = ss.soil_layers(EDGES, theta_residual=[0.0] * 3, theta_field_capacity=[0.15] * 3, theta_saturation=[0.4] * 3, diffusivity_m2_s=[1e-7] * 2, theta_evaporation_stress=0.3)
    legacy = ss.prepare_column(lay, np.ones(2, bool), 1.0)
    with pytest.raises(fl.FloatingSoilError, match="FloatingSoilColumn"):
        fl.floating_step(legacy, st, 1.0, **zero_forcing(col))
    with pytest.raises(ss.SubsurfaceError):
        ss.step(col, ss.column_state(legacy, np.full((3, 2), 0.01)), 1.0, **zero_forcing(col))
    with pytest.raises(Exception):
        sp.spatial_step(col, st, 1.0, **zero_forcing(col), memory_budget_bytes=BIG)
    with pytest.raises(fl.FloatingSoilError, match="FloatingSoilState"):
        fl.floating_step(col, ss.column_state(legacy, np.full((3, 2), 0.01)), 1.0, **zero_forcing(col))
    for bad in (lambda: copy.copy(col), lambda: copy.deepcopy(col), lambda: pickle.dumps(col), lambda: fl.FloatingSoilColumn(*([None] * 18))):
        with pytest.raises(fl.FloatingSoilError):
            bad()


def test_cuda_backend_without_cupy_is_a_typed_setup_error_and_the_model_exists():
    try:
        import cupy  # noqa: F401
        pytest.skip("CuPy is importable here; the unavailable-backend refusal is not exercised (no claim)")
    except ImportError:
        pass
    with pytest.raises(fl.FloatingBackendUnavailableError, match="no fallback"):
        column(EDGES, [1.0], backend="cuda")
    est = fl.estimate_floating_memory(3, 4, layouts=dict(theta_residual="scalar", theta_field_capacity="scalar", theta_saturation="scalar", interface="shared"), backend="cuda")
    assert est.backend == "cuda" and est.terms["retained_device"] > 0 and est.terms["step_host"] == 144


@pytest.mark.parametrize("kw", [
    dict(depth=[-1e-12]), dict(depth=[float("nan")]), dict(depth=[float("inf")]), dict(tf=0.5, ts=0.4), dict(tr=0.3, tf=0.2), dict(ts=1.2), dict(stress=0.0), dict(iface=np.array([-1.0, 1.0])),
    dict(edges=[0.0, 0.0, 0.3, 0.6]), dict(edges=[0.1, 0.2, 0.3, 0.6]), dict(edges=[0.0, 0.3, 0.2, 0.6]), dict(depth=[1e-310], ts=0.4)])
def test_invalid_inputs_are_refused_before_anything_is_published(kw):
    args = dict(edges=EDGES, depth=[0.5], tr=0.0, tf=0.2, ts=0.4, stress=0.1)
    args.update(kw)
    with pytest.raises(fl.FloatingSoilError):
        column(args.pop("edges"), args.pop("depth"), **args)


@pytest.mark.parametrize("bad", [[0.5], np.array([0.5], dtype=np.float32), np.array(0.5), np.ones((1, 1)), True])
def test_metadata_refusals(bad):
    with pytest.raises(fl.FloatingSoilError):
        fl.prepare_floating_column(EDGES, theta_residual=0.0, theta_field_capacity=0.2, theta_saturation=0.4, diffusivity_m2_s=np.zeros(2), theta_evaporation_stress=0.1, soil_depth_m=bad,
                                   active_mask=np.ones(1, bool), cell_area_m2=1.0)


def test_gravity_is_not_a_parameter_and_modes_are_explicit():
    import inspect

    assert "gravity_drainage" not in inspect.signature(fl.prepare_floating_column).parameters and fl.COMPONENT_TAG == "floating_porous_column_pure"
    with pytest.raises(fl.FloatingSoilError, match="refuses the donor"):
        fl.prepare_floating_column(EDGES, theta_residual=0.0, theta_field_capacity=0.2, theta_saturation=0.4, diffusivity_m2_s=np.zeros(2), interface_conductivity_m_s=np.ones(2), diffusion_coefficient=1.0,
                                   theta_evaporation_stress=0.1, soil_depth_m=np.ones(1), active_mask=np.ones(1, bool), cell_area_m2=1.0)


@pytest.mark.parametrize("L", [1, 2, 5])
def test_one_two_and_five_layers_run_and_close(L, backend):
    edges = np.linspace(0.0, 0.5, L + 1)
    col = column(edges, [0.0, edges[-1] * 0.5, edges[-1], edges[-1] + 1.0], tf=0.2, ts=0.4, mode="tietjen_donor_depth", backend=backend)
    w = fill(col, np.random.default_rng(L), 0.9)
    r = fl.floating_step(col, fl.floating_continued_state(col, w), 600.0, added_water_m=np.zeros((L, 4)), potential_evaporation_m_s=np.full(4, 1e-7))
    assert abs(r.budget.residual_m3) <= r.budget.bound_m3 and r.state.water_m.shape == (L, 4)


# ---- memory ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_admission_precedes_allocation_and_the_model_bounds_a_warm_step(backend):
    rng = np.random.default_rng(21)
    N = 20000
    col, _ = make(EDGES, rng.choice([0.0, 0.2, 0.6, 2.0], size=N), rng=rng, backend=backend)
    est = col.estimate
    st = fl.floating_continued_state(col, fill(col, rng, 0.8))
    f = dict(added_water_m=np.zeros((3, N)), potential_evaporation_m_s=np.zeros(N))
    with pytest.raises(fl.FloatingMemoryError):
        fl.floating_step(col, st, 60.0, memory_budget_bytes=1000, **f)
    with pytest.raises(fl.FloatingMemoryError):
        fl.floating_step(col, st, 60.0, available_bytes_fn=lambda: 1000, **f)
    fl.floating_step(col, st, 60.0, **f)  # warm (compilation, imports)
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    fl.floating_step(col, st, 60.0, **f)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    est_step = fl.estimate_floating_memory(3, N, layouts=col.layouts, diffusion_mode=col.diffusion_mode, backend=backend).step_bytes
    print(f"{backend}: measured warm peak {peak} B, modelled {est_step} B; retained model {est.retained_bytes} B")
    assert peak <= est_step


def test_retained_bytes_model_covers_the_owned_maps():
    col, _ = make(EDGES, np.linspace(0, 1, 100))
    owned = sum(a.nbytes for a in col._maps.values())
    assert owned <= col.estimate.retained_bytes + 100 * 8 + 8 * 20  # the model counts every owned array (bottom_open/active as bytes)


# ---- D1b correction02: the old-owner value check runs only AFTER the memory admission --------------------------------------------------------------------------------------------------------------------------------
def _recon_pair(backend, N=64):
    old, _ = make(EDGES, np.full(N, 2.0), backend=backend, rng=np.random.default_rng(1))
    new, _ = make(EDGES, np.full(N, 0.3), backend=backend, rng=np.random.default_rng(1))
    return old, new, fl.floating_continued_state(old, fill(old, np.random.default_rng(2), 0.5))


def test_reconcile_admits_memory_before_the_old_value_scratch_is_allocated(backend, monkeypatch):
    old, new, st = _recon_pair(backend)
    calls = []
    real = fl._check_water_values
    monkeypatch.setattr(fl, "_check_water_values", lambda col, *a, **k: calls.append(col is old) or real(col, *a, **k))
    snap = st.water_m.copy()
    for kw in (dict(memory_budget_bytes=0), dict(available_bytes_fn=lambda: 0), dict(memory_budget_bytes=1000)):
        with pytest.raises(fl.FloatingMemoryError, match="nothing was allocated"):
            fl.reconcile_floating(old, st, new, **kw)
    assert not calls and np.array_equal(st.water_m, snap)  # refused on the metadata-only model: no old-value row was ever built; the input is unchanged
    fl.reconcile_floating(old, st, new)  # admitted: the OLD-owner validation then runs exactly once (the minted candidate is checked against the NEW owner)
    assert calls == [True, False]


def test_reconcile_area_and_binding_refusals_still_precede_the_quota_reader_and_value_scratch(backend, monkeypatch):
    old, _, st = _recon_pair(backend)
    new_other_area, _ = make(EDGES, np.full(64, 0.3), backend=backend, rng=np.random.default_rng(1))
    object.__setattr__(new_other_area, "cell_area_m2", 3.0)  # same columns, a different area
    reads, calls = [], []
    real = fl._check_water_values
    monkeypatch.setattr(fl, "_check_water_values", lambda *a, **k: calls.append(1) or real(*a, **k))
    with pytest.raises(fl.FloatingSoilError, match="identical cell_area_m2"):
        fl.reconcile_floating(old, st, new_other_area, available_bytes_fn=lambda: reads.append(1) or (1 << 40))
    assert not reads and not calls


def test_invalid_old_state_is_still_refused_after_a_successful_admission_and_the_bound_covers_the_scratch(backend):
    import tracemalloc

    old, new, st = _recon_pair(backend, N=4000)
    bad = st.water_m.copy()
    bad[0, 0] = 1.0  # far above the OLD capacity
    forged = fl.FloatingSoilState(bad, 0.0, old.sha256, "arbitrary")
    with pytest.raises(fl.FloatingSoilError, match="OLD state violates"):
        fl.reconcile_floating(old, forged, new)
    assert np.isclose(bad[0, 0], 1.0)  # unchanged
    est = fl.estimate_floating_memory(old.n_layers, old.n_cells, layouts=new.layouts, diffusion_mode=new.diffusion_mode, backend=backend)
    fl._check_water_values(old, st.water_m, False)  # warm
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    fl._check_water_values(old, st.water_m, False)
    scratch = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert 0 < scratch <= est.step_bytes  # the value-check scratch is inside the modelled (and admitted) step bytes
    fl.reconcile_floating(old, st, new, memory_budget_bytes=est.step_bytes)  # exactly the admitted bound suffices
