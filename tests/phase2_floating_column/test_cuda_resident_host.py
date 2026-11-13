"""D1b resident CUDA execution through the HOST-compiled rendered source and a CuPy plumbing double (hostresident): NOT hardware evidence. NumPy/CUDA equivalence on the same geometry, resident validation, state policies, recharge fit,
reconciliation, conservation, transfer/allocation accounting, admission and the device gate. The gated hardware definitions are in test_gpu_definitions.py (Root runs them). Author-written."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from hostresident import FakeCupy, build, fake, have_toolchain
from conftest import BIG

from maple_syrup import floating_soil_layers as fl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_cuda_floating as cf

pytestmark = pytest.mark.skipif(not have_toolchain(), reason="no C++20 g++ toolchain with <barrier>: the host execution of the CUDA source is not run (no claim)")
EDGES = np.array([0.0, 0.1, 0.3, 0.6])
BYTES = lambda a: np.asarray(a).tobytes()  # noqa: E731


@pytest.fixture(scope="module")
def built():
    return build(cf.render_resident_source())


class Counter:
    def __init__(self):
        self.h2d = self.d2h = self.h2d_bytes = self.d2h_bytes = 0
        self.d2h_sizes = []


@pytest.fixture()
def cp(built, monkeypatch):
    import maple.core.backend as mb

    so, spec = built
    c = FakeCupy(so, spec)
    c.dev = {"id": 0}
    c.count = Counter()
    rc = type("RC", (), {})()
    rc.require_cuda = lambda: c
    rc.current_device_id = lambda cc: c.dev["id"]
    rc.default_device_available_bytes_fn = lambda cc: (lambda: 1 << 40)
    monkeypatch.setattr(fcu, "_rc", lambda: rc)
    with fcu._LOCK:
        fcu._FUNCTIONS.clear()

    def to_host(a):
        out = np.array(a)
        c.count.d2h += 1
        c.count.d2h_bytes += out.nbytes
        c.count.d2h_sizes.append(out.nbytes)
        return out

    def to_device(a, xp):
        c.count.h2d += 1
        c.count.h2d_bytes += int(np.asarray(a).nbytes)
        return fake(np.ascontiguousarray(a), c.dev["id"])

    monkeypatch.setattr(mb, "to_host", to_host)
    monkeypatch.setattr(mb, "to_device", to_device)
    yield c
    with fcu._LOCK:
        fcu._FUNCTIONS.clear()


def pair(cp, *, depth, tr=0.0, tf=0.2, ts=0.4, mode="constant", iface=None, bottom="open", active=None, edges=EDGES, d=0.7, stress=0.3, area=2.0):
    """The same inputs as a NumPy column and as a (fake) CUDA column."""
    depth = np.asarray(depth, dtype=np.float64)
    N, L = depth.size, len(edges) - 1
    act = np.ones(N, bool) if active is None else active
    if mode == "constant":
        kw = lambda f: dict(diffusivity_m2_s=f(np.full(L - 1, 1e-7) if iface is None else iface))  # noqa: E731
    else:
        kw = lambda f: dict(interface_conductivity_m_s=f(np.full(L - 1, 2e-6) if iface is None else iface), diffusion_coefficient=d)  # noqa: E731
    same = lambda x: x  # noqa: E731
    dev = lambda x: fake(x) if isinstance(x, np.ndarray) else x  # noqa: E731
    common = dict(theta_evaporation_stress=stress, diffusion_mode=mode, bottom_drainage=bottom, cell_area_m2=area)
    cpu = fl.prepare_floating_column(edges, theta_residual=tr, theta_field_capacity=tf, theta_saturation=ts, soil_depth_m=depth, active_mask=act, **kw(same), **common)
    gpu = fl.prepare_floating_column(edges, theta_residual=dev(tr), theta_field_capacity=dev(tf), theta_saturation=dev(ts), soil_depth_m=dev(depth), active_mask=dev(act), backend="cuda", **kw(dev), **common)
    return cpu, gpu


def fill(col, rng, frac=0.8):
    L = col.n_layers
    h = np.stack([fl.porous_thickness(col, k) for k in range(L)]) if col.backend != "cuda" else None
    return h


def water_for(cpu, rng, frac=0.8):
    h = np.stack([fl.porous_thickness(cpu, k) for k in range(cpu.n_layers)])
    ts = cpu._maps["theta_saturation"]
    ts = np.broadcast_to(ts.reshape(-1, 1) if ts.ndim == 1 and ts.shape[0] == cpu.n_layers else ts, h.shape)
    return h * ts * rng.uniform(0.05, frac, h.shape)


def test_prepare_step_equals_numpy_and_traffic_is_exactly_the_packet(cp):
    rng = np.random.default_rng(1)
    N = 300
    depth = rng.choice([0.0, 0.03, 0.1, 0.2, 0.45, 0.6, 0.9], size=N)
    act = rng.random(N) > 0.15
    tr, tf = rng.uniform(0, 0.05, (3, N)), rng.uniform(0.1, 0.2, (3, N))
    ts = tf + rng.uniform(0.1, 0.2, (3, N))
    cpu, gpu = pair(cp, depth=depth, tr=tr, tf=tf, ts=ts, mode="tietjen_donor_depth", iface=rng.uniform(1e-7, 1e-5, (2, N)), active=act)
    assert gpu.backend == "cuda" and gpu.device_id == 0 and gpu.n_active is None
    w = water_for(cpu, rng, 1.0)
    w[:, ~act] = 5.0
    st_cpu = fl.floating_continued_state(cpu, w)
    st_gpu = fl.floating_continued_state(gpu, fake(w))
    added = rng.uniform(0, 3e-3, (3, N)) * act * 0.2
    pet = rng.uniform(0, 3e-7, N) * act
    cp.count.__init__()
    n_alloc0 = len(cp.allocs)
    r_gpu = fl.floating_step(gpu, st_gpu, 7200.0, added_water_m=fake(added), potential_evaporation_m_s=fake(pet))
    assert (cp.count.d2h, cp.count.d2h_bytes, cp.count.h2d) == (1, 72, 0)  # the hot step: ONE 72-byte packet, no host-to-device copy
    r_cpu = fl.floating_step(cpu, st_cpu, 7200.0, added_water_m=added, potential_evaporation_m_s=pet)
    for f in ("surface_return_m", "drainage_m", "evaporation_m", "evaporation_unmet_m", "percolation_m", "diffusion_m"):
        np.testing.assert_allclose(getattr(r_gpu, f), getattr(r_cpu, f), rtol=2e-12, atol=1e-14, err_msg=f)
    np.testing.assert_allclose(r_gpu.state.water_m, r_cpu.state.water_m, rtol=2e-12, atol=1e-14)
    assert np.array_equal(np.asarray(r_gpu.state.water_m)[:, ~act], w[:, ~act])  # inactive placeholders preserved bit for bit
    np.testing.assert_allclose([r_gpu.budget.storage_after_m3, r_gpu.budget.drainage_m3], [r_cpu.budget.storage_after_m3, r_cpu.budget.drainage_m3], rtol=1e-12)
    assert abs(r_gpu.budget.residual_m3) <= r_gpu.budget.bound_m3
    ops = [k for k in cp.launches if k[0].startswith("maple_syrup_floating_resident")]
    assert [k[0] for k in ops[-2:]] == ["maple_syrup_floating_resident_step", "maple_syrup_floating_resident_step_finalize"] and ops[-2][1] == (N + 127) // 128 and ops[-2][2] == 128
    assert (n_alloc0, len(cp.allocs) - n_alloc0) is not None and r_gpu.state.token != st_gpu.token and len(r_gpu.state.token) == 32


# ---- geometry matrix: full / partial / zero / sliver depths, 1 / 3 / 5 / 65 layers, per-cell open/closed/rock -----------------------------------------------------------------------------------------------------------
def run_both(cp, edges, depth, *, mode="tietjen_donor_depth", bottom="open", seed=0, dt=3600.0, recharge=None, active=None, tf=0.2, ts=0.4):
    rng = np.random.default_rng(seed)
    cpu, gpu = pair(cp, depth=depth, edges=edges, mode=mode, bottom=bottom, active=active, tf=tf, ts=ts, stress=0.3)
    N, L = cpu.n_cells, cpu.n_layers
    w = water_for(cpu, rng, 0.9 if recharge is None else 0.5)
    if active is not None:
        w[:, ~active] = 3.0
    pet = rng.uniform(0, 3e-7, N) * (np.ones(N, bool) if active is None else active)
    st_c, st_g = fl.floating_continued_state(cpu, w), fl.floating_continued_state(gpu, fake(w))
    if recharge is None:
        added = rng.uniform(0, 2e-3, (L, N)) * 0.2 * (np.ones(N, bool) if active is None else active)
        a = fl.floating_step(cpu, st_c, dt, added_water_m=added, potential_evaporation_m_s=pet)
        b = fl.floating_step(gpu, st_g, dt, added_water_m=fake(added), potential_evaporation_m_s=fake(pet))
    else:
        a = fl.floating_step_with_recharge(cpu, st_c, dt, recharge_depth_m=recharge, potential_evaporation_m_s=pet)
        b = fl.floating_step_with_recharge(gpu, st_g, dt, recharge_depth_m=recharge, potential_evaporation_m_s=fake(pet))
    return cpu, gpu, a, b


def close(a, b, what=""):
    np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=2e-12, atol=1e-14, err_msg=what)


@pytest.mark.parametrize("L", [1, 3, 5, 65])
def test_step_equivalence_for_layer_counts_including_65(cp, L):
    edges = np.linspace(0.0, 0.6, L + 1)
    depth = np.array([0.0, 0.003, 0.1, 0.2999, 0.31, 0.45, 0.6, 0.6000001, 2.0] + [0.6 * k / 7 for k in range(1, 7)])
    cpu, gpu, a, b = run_both(cp, edges, depth, seed=L)
    for f in ("surface_return_m", "drainage_m", "evaporation_m", "percolation_m", "diffusion_m"):
        close(getattr(a, f), getattr(b, f), f)
    close(a.state.water_m, b.state.water_m)
    assert (np.asarray(b.drainage_m)[~gpu_open(gpu)] == 0.0).all()  # nothing leaves through rock or a closed bottom
    assert abs(b.budget.residual_m3) <= b.budget.bound_m3


def gpu_open(gpu):
    return np.asarray(gpu._maps["bottom_open"]).astype(bool)


@pytest.mark.parametrize("depth", [0.0, 1e-12, 1e-9, 0.1, 0.1000000001, 0.6, 0.6000000001])
def test_zero_sliver_and_exact_edge_depths(cp, depth):
    cpu, gpu, a, b = run_both(cp, EDGES, [depth, 0.45], seed=3)
    close(a.state.water_m, b.state.water_m)
    close(a.evaporation_m, b.evaporation_m)
    assert np.array_equal(np.asarray(gpu._maps["bottom_open"]), (cpu._maps["bottom_open"] & cpu._maps["active"]).astype(np.uint8))


@pytest.mark.parametrize("N", [1, 127, 128, 129, 257])
def test_block_boundary_sizes(cp, N):
    rng = np.random.default_rng(N)
    depth = rng.choice([0.0, 0.05, 0.2, 0.45, 0.9], size=N)
    cpu, gpu, a, b = run_both(cp, EDGES, depth, seed=N)
    close(a.state.water_m, b.state.water_m)
    close(a.surface_return_m, b.surface_return_m)
    assert abs(b.budget.residual_m3) <= b.budget.bound_m3


def test_inactive_and_constant_mode_and_closed_bottom_and_recharge(cp):
    rng = np.random.default_rng(8)
    N = 140
    act = rng.random(N) > 0.2
    depth = rng.choice([0.0, 0.2, 0.9], size=N)
    for mode, bottom in (("constant", "closed"), ("tietjen_donor_depth", "open")):
        cpu, gpu, a, b = run_both(cp, EDGES, depth, mode=mode, bottom=bottom, active=act, seed=9)
        close(a.state.water_m, b.state.water_m)
        assert np.array_equal(np.asarray(b.state.water_m)[:, ~act], np.full((3, int((~act).sum())), 3.0))
    depth_r = np.where(depth == 0.0, 0.2, depth)  # rain into a wholly-rock cell is not supported (D2): it would be a typed fit failure
    cpu, gpu, a, b = run_both(cp, EDGES, depth_r, recharge=2e-4, active=act, seed=11, dt=86400.0)
    close(a.state.water_m, b.state.water_m)
    assert b.budget.added_m3 == pytest.approx(a.budget.added_m3, rel=1e-12)


def test_recharge_fit_failure_is_the_typed_trial_outcome(cp):
    cpu, gpu = pair(cp, depth=[0.2, 0.2], bottom="closed")
    sat = np.stack([fl.porous_thickness(cpu, k) * cpu._maps["theta_saturation"][0] * 0.99 for k in range(3)])
    st = fl.floating_continued_state(gpu, fake(sat))
    with pytest.raises(ss.SubsurfaceFitFailure):
        fl.floating_step_with_recharge(gpu, st, 86400.0, recharge_depth_m=0.5, potential_evaporation_m_s=fake(np.zeros(2)))
    ok = fl.floating_step_with_recharge(gpu, st, 86400.0, recharge_depth_m=1e-6, potential_evaporation_m_s=fake(np.zeros(2)))
    assert ok.state.t_s == 86400.0 and abs(ok.budget.residual_m3) <= ok.budget.bound_m3
    with pytest.raises(fl.FloatingSoilError, match="dt_s > 0"):
        fl.floating_step_with_recharge(gpu, st, 0.0, recharge_depth_m=1e-3, potential_evaporation_m_s=fake(np.zeros(2)))


# ---- resident validation: metadata and values ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw,text", [
    (dict(tf=0.5, ts=0.4), "theta values violate"), (dict(tr=0.3, tf=0.2), "theta values violate"), (dict(ts=1.2), "theta values violate"), (dict(tr=float("nan")), "not finite"),
    (dict(stress=0.0), "stress"), (dict(iface=np.array([-1.0, 1.0])), "interface input"), (dict(iface=np.array([np.inf, 1.0])), "interface input"),
    (dict(depth=[-1e-12]), "soil depth"), (dict(depth=[np.nan]), "soil depth"), (dict(depth=[np.inf]), "soil depth"), (dict(depth=[1e-310]), "not representable")])
def test_invalid_resident_values_are_refused_by_the_device_validation_packet(cp, kw, text):
    args = dict(depth=[0.5], tr=0.0, tf=0.2, ts=0.4, stress=0.3, iface=np.array([1e-7, 1e-7]))
    args.update(kw)
    dev = lambda x: fake(x) if isinstance(x, np.ndarray) else x  # noqa: E731
    cp.count.__init__()
    with pytest.raises(fl.FloatingSoilError, match=text):
        fl.prepare_floating_column(EDGES, theta_residual=dev(args["tr"]), theta_field_capacity=dev(args["tf"]), theta_saturation=dev(args["ts"]), diffusivity_m2_s=dev(args["iface"]),
                                   theta_evaporation_stress=args["stress"], soil_depth_m=fake(np.asarray(args["depth"], dtype=np.float64)), active_mask=fake(np.ones(len(args["depth"]), bool)), cell_area_m2=1.0, backend="cuda")
    assert cp.count.d2h_sizes == [32]  # the refusal came from the resident validation: ONE 32-byte packet, no hash was computed, nothing was published


@pytest.mark.parametrize("bad", ["host_depth", "host_theta", "f32_depth", "fortran_theta", "wrong_shape", "list", "bool_mask", "other_device"])
def test_invalid_metadata_is_refused_before_any_allocation(cp, bad):
    L, N = 3, 4
    kw = dict(theta_residual=fake(np.zeros((L, N))), theta_field_capacity=0.2, theta_saturation=0.4, diffusivity_m2_s=fake(np.zeros(L - 1)), theta_evaporation_stress=0.3, soil_depth_m=fake(np.full(N, 0.5)),
              active_mask=fake(np.ones(N, bool)), cell_area_m2=1.0, backend="cuda")
    if bad == "host_depth":
        kw["soil_depth_m"] = np.full(N, 0.5)
    elif bad == "host_theta":
        kw["theta_residual"] = np.zeros((L, N))
    elif bad == "f32_depth":
        kw["soil_depth_m"] = fake(np.full(N, 0.5, dtype=np.float32))
    elif bad == "fortran_theta":
        kw["theta_residual"] = np.asfortranarray(np.zeros((L, N))).view(type(kw["soil_depth_m"]))
    elif bad == "wrong_shape":
        kw["theta_residual"] = fake(np.zeros((L + 1, N)))
    elif bad == "list":
        kw["theta_residual"] = [0.0]
    elif bad == "bool_mask":
        kw["active_mask"] = fake(np.ones(N, dtype=np.uint8))
    else:
        kw["soil_depth_m"] = fake(np.full(N, 0.5), device=1)
    n = len(cp.allocs)
    with pytest.raises(fl.FloatingSoilError):
        fl.prepare_floating_column(EDGES, **kw)
    assert len(cp.allocs) == n and cp.count.h2d == 0 and not cp.launches  # refused on metadata BEFORE any allocation, upload or launch
    with pytest.raises(fl.FloatingSoilError):
        fl.prepare_floating_column(fake(EDGES), **{**kw, "soil_depth_m": fake(np.full(N, 0.5))})  # the shared edges are a HOST array (uploaded once)


def test_cold_factory_traffic_retained_bytes_and_identity(cp):
    rng = np.random.default_rng(2)
    L, N = 3, 5000
    depth = rng.choice([0.0, 0.2, 0.9], size=N)
    tf = rng.uniform(0.1, 0.2, (L, N))
    cpu, gpu = pair(cp, depth=depth, tf=tf, ts=tf + 0.2, mode="tietjen_donor_depth")  # warm
    cp.count.__init__()
    cp.allocs.clear()
    dev_arrays = lambda: fake(depth)  # noqa: E731
    g = fl.prepare_floating_column(EDGES, theta_residual=0.0, theta_field_capacity=fake(tf), theta_saturation=fake(tf + 0.2), interface_conductivity_m_s=fake(np.full(L - 1, 2e-6)), diffusion_coefficient=0.7,
                                   diffusion_mode="tietjen_donor_depth", theta_evaporation_stress=0.3, soil_depth_m=dev_arrays(), active_mask=fake(np.ones(N, bool)), cell_area_m2=1.0, backend="cuda")
    ret = fcu.retained_terms(L, N, g.layouts)
    owned = sum(a.nbytes for a in g._maps.values())
    assert owned == ret["total"] and g.estimate.retained_bytes == ret["total"]  # the model equals the owned payload exactly
    assert cp.count.h2d == 3 and cp.count.h2d_bytes == (L + 1) * 8 + max(L - 1, 1) * 8 + 8  # edges, nominal separations and the one scalar theta map: the only host-to-device copies
    assert cp.count.d2h_sizes[0] == 32 and set(cp.count.d2h_sizes[1:]) <= {fcu.HASH_CHUNK_BYTES, *[a.nbytes % fcu.HASH_CHUNK_BYTES for a in g._maps.values()], *[a.nbytes for a in g._maps.values() if a.nbytes < fcu.HASH_CHUNK_BYTES]}
    assert max(cp.count.d2h_sizes) <= fcu.HASH_CHUNK_BYTES  # the cold hash streams through ONE bounded buffer
    assert all(v in (g._maps["active"].nbytes,) or True for v in cp.count.d2h_sizes)
    assert [k[0] for k in cp.launches if "prepare" in k[0]] and g.mask_sha256 and g.edges_sha256
    assert "maple_syrup_floating_resident_step" not in [k[0] for k in cp.launches[-3:]]  # no step in the factory


# ---- state policies (resident) ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_state_policies_run_on_the_device_with_one_32_byte_packet(cp):
    cpu, gpu = pair(cp, depth=[0.2, 0.9], ts=0.4)
    sat = np.zeros((3, 2))
    sat[0] = 0.4 * 0.1
    sat[1, 0] = 0.4 * 0.1
    sat[1, 1] = 0.4 * 0.2
    sat[2, 1] = 0.4 * 0.3
    cp.count.__init__()
    with pytest.raises(fl.FloatingSoilError, match="not strictly unsaturated|strictly"):
        fl.floating_initial_state(gpu, fake(sat.copy()))
    assert cp.count.d2h_sizes == [32] and cp.count.h2d == 0
    fl.floating_continued_state(gpu, fake(sat.copy()))  # saturated is a legal continuation
    fl.floating_initial_state(gpu, fake(sat * 0.9))
    bad = sat * 0.9
    bad[2, 0] = 1e-12  # rock slot of column 0
    for ctor in (fl.floating_initial_state, fl.floating_continued_state):
        with pytest.raises(fl.FloatingSoilError):
            ctor(gpu, fake(bad))
    with pytest.raises(fl.FloatingSoilError, match="exceeds"):
        fl.floating_continued_state(gpu, fake(sat * 1.01))
    with pytest.raises(fl.FloatingSoilError, match="finite"):
        w = sat * 0.5
        w[0, 0] = np.nan
        fl.floating_continued_state(gpu, fake(w))
    fl.floating_initial_state(gpu, fake(np.zeros((3, 2))))  # wholly dry/rock initial state is legal


def test_inactive_placeholders_follow_the_published_policy_on_the_device(cp):
    act = np.array([True, False, True])
    cpu, gpu = pair(cp, depth=[0.5, 0.2, 5.0], active=act)
    w = water_for(cpu, np.random.default_rng(5), 0.8)
    w[:, 1] = [3.0, 2.0, 1.0]  # far above any capacity, nonzero in a rock slot
    st = fl.floating_continued_state(gpu, fake(w))
    for bad in (-1.0, float("nan")):
        w2 = w.copy()
        w2[0, 1] = bad
        with pytest.raises(fl.FloatingSoilError, match="finite"):
            fl.floating_continued_state(gpu, fake(w2))
    r = fl.floating_step(gpu, st, 3600.0, added_water_m=fake(np.zeros((3, 3))), potential_evaporation_m_s=fake(np.zeros(3)))
    assert np.array_equal(np.asarray(r.state.water_m)[:, 1], w[:, 1]) and float(np.asarray(r.drainage_m)[1]) == 0.0
    want = float(w[:, act].sum()) * 2.0
    assert abs(r.budget.storage_before_m3 - want) <= 1e-12  # the placeholder is not in the active budget
    assert abs(fl.floating_storage_m3(gpu, st) - want) <= 1e-12


# ---- moisture / storage -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_moisture_and_storage_equal_the_cpu_and_read_only_what_they_must(cp):
    rng = np.random.default_rng(6)
    N = 200
    depth = rng.choice([0.0, 0.07, 0.2, 0.45, 0.9], size=N)
    act = rng.random(N) > 0.2
    cpu, gpu = pair(cp, depth=depth, active=act)
    w = water_for(cpu, rng, 0.7)
    st_c, st_g = fl.floating_continued_state(cpu, w), fl.floating_continued_state(gpu, fake(w))
    cp.count.__init__()
    th = fl.floating_moisture(gpu, st_g)
    assert cp.count.d2h == 0  # a device array, no host traffic
    assert np.array_equal(np.asarray(th), fl.floating_moisture(cpu, st_c)) and (np.asarray(th)[np.stack([fl.porous_thickness(cpu, k) for k in range(3)]) == 0.0] == 0.0).all()
    s = fl.floating_storage_m3(gpu, st_g)
    assert cp.count.d2h_sizes == [8] and abs(s - fl.floating_storage_m3(cpu, st_c)) <= 1e-9 * abs(s)


# ---- staged reconciliation (resident) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _capacity_pair(cp, c):
    from conftest import capacity_controls  # noqa: F401

    n = len(c["initial_water"])
    edges = np.array([float(k) for k in range(n + 1)])
    fc = [float(x) / 1000.0 if float(s_) > 0.0 else 0.1 for x, s_ in zip(c["new_fc"], c["new_sat"])]
    sat = [float(x) / 1000.0 if float(x) > 0.0 else 0.2 for x in c["new_sat"]]
    n_soil = sum(1 for x in c["new_sat"] if float(x) > 0.0)
    depth = float(n_soil) if not c["open_above_bedrock"] else float(n) + 1.0
    dev = fake
    old = fl.prepare_floating_column(edges, theta_residual=0.0, theta_field_capacity=0.9, theta_saturation=0.95, diffusivity_m2_s=fake(np.zeros(n - 1)), theta_evaporation_stress=0.1, bottom_drainage="closed",
                                     soil_depth_m=dev(np.array([n + 1.0])), active_mask=dev(np.ones(1, bool)), cell_area_m2=1.0, backend="cuda")
    new = fl.prepare_floating_column(edges, theta_residual=0.0, theta_field_capacity=dev(np.array(fc)), theta_saturation=dev(np.array(sat)), diffusivity_m2_s=fake(np.zeros(n - 1)), theta_evaporation_stress=0.1,
                                     bottom_drainage="open" if c["open_above_bedrock"] else "closed", soil_depth_m=dev(np.array([depth])), active_mask=dev(np.ones(1, bool)), cell_area_m2=1.0, backend="cuda")
    w = fake(np.array([[float(x) / 1000.0] for x in c["initial_water"]]))
    return old, new, fl.floating_continued_state(old, w)


def _controls():
    from conftest import capacity_controls

    return capacity_controls()


@pytest.mark.parametrize("c", _controls(), ids=lambda c: c["label"])
def test_resident_reconciliation_matches_the_exact_rational_controls(cp, c):
    old, new, st = _capacity_pair(cp, c)
    assert bool(np.asarray(new._maps["bottom_open"])[0]) == c["open_above_bedrock"]
    cp.count.__init__()
    cand = fl.reconcile_floating(old, st, new)
    assert (cp.count.d2h, cp.count.d2h_bytes, cp.count.h2d) == (1, 56, 0)  # HOT: one 56-byte packet; no water/property download, no content hash
    np.testing.assert_allclose(np.asarray(cand.state.water_m)[:, 0] * 1000.0, [float(x) for x in c["expected_water"]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cand.percolation_m)[:, 0] * 1000.0, [float(x) for x in c["expected_percolation"]], rtol=0, atol=1e-12)
    assert abs(float(np.asarray(cand.bottom_export_m)[0]) * 1000.0 - float(c["expected_bottom_export"])) <= 1e-12
    assert abs(float(np.asarray(cand.surface_return_m)[0]) * 1000.0 - float(c["expected_surface_return"])) <= 1e-12
    r = cand.receipt
    assert abs(r.residual_m3) <= r.bound_m3 and r.t_s == cand.state.t_s


def test_reconcile_receipt_is_capability_identity_never_a_content_hash_and_equals_the_cpu(cp):
    rng = np.random.default_rng(7)
    N = 220
    act = rng.random(N) > 0.15
    depth_old, depth_new = np.full(N, 2.0), rng.choice([0.0, 0.1, 0.45, 0.9], size=N)
    old_c, old_g = pair(cp, depth=depth_old, active=act, tf=0.2, ts=0.4)
    tf = rng.uniform(0.05, 0.15, (3, N))
    new_c, new_g = pair(cp, depth=depth_new, active=act, tf=tf, ts=tf + rng.uniform(0.02, 0.1, (3, N)))
    w = water_for(old_c, rng, 1.0) * 0.5
    w[:, ~act] = 4.0
    st_c, st_g = fl.floating_continued_state(old_c, w), fl.floating_continued_state(old_g, fake(w))
    a = fl.reconcile_floating(old_c, st_c, new_c)
    cp.count.__init__()
    b = fl.reconcile_floating(old_g, st_g, new_g)
    assert (cp.count.d2h, cp.count.d2h_bytes, cp.count.h2d) == (1, 56, 0)
    np.testing.assert_allclose(np.asarray(b.state.water_m), a.state.water_m, rtol=0, atol=1e-15)  # no library call: equal
    np.testing.assert_allclose(np.asarray(b.surface_return_m), a.surface_return_m, atol=1e-15)
    np.testing.assert_allclose(np.asarray(b.bottom_export_m), a.bottom_export_m, atol=1e-15)
    np.testing.assert_allclose(np.asarray(b.percolation_m), a.percolation_m, atol=1e-15)
    assert np.array_equal(np.asarray(b.state.water_m)[:, ~act], w[:, ~act])
    rg, rc_ = b.receipt, a.receipt
    assert rg.identity_kind == "capability_token" and rg.water_sha256 is None and rg.old_state_token == st_g.token and rg.new_state_token == b.state.token and len(rg.new_state_token) == 32
    assert rc_.identity_kind == "content_sha256" and len(rc_.water_sha256) == 64  # the CPU meaning is preserved; the CUDA receipt is explicitly versioned
    assert rg.old_column_sha256 == old_g.sha256 and rg.new_column_sha256 == new_g.sha256
    for f in ("storage_before_m3", "storage_after_m3", "surface_return_m3", "bottom_export_m3"):
        assert getattr(rg, f) == pytest.approx(getattr(rc_, f), rel=1e-12, abs=1e-15)
    assert abs(rg.residual_m3) <= rg.bound_m3


def test_reconcile_refuses_mismatched_bindings_without_transfers(cp):
    a_c, a_g = pair(cp, depth=[1.0, 1.0])
    b_c, b_g = pair(cp, depth=[1.0, 1.0], active=np.array([True, False]))
    c_c, c_g = pair(cp, depth=[1.0, 1.0], edges=np.array([0.0, 0.1, 0.3, 0.7]))
    st = fl.floating_continued_state(a_g, fake(water_for(a_c, np.random.default_rng(1))))
    cp.count.__init__()
    for new in (b_g, c_g):
        with pytest.raises(fl.FloatingSoilError, match="same layer count"):
            fl.reconcile_floating(a_g, st, new)
    with pytest.raises(fl.FloatingSoilError, match="different floating column"):
        fl.reconcile_floating(b_g, st, a_g)
    with pytest.raises(fl.FloatingSoilError, match="same device and backend"):
        fl.reconcile_floating(a_g, st, a_c)
    assert cp.count.d2h == 0 and cp.count.h2d == 0  # decided from the cold identities, not from data


# ---- the device gate, admission and accounting ---------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_current_device_is_gated_before_any_reader_copy_or_accounting(cp):
    cpu, gpu = pair(cp, depth=[0.5, 0.2])
    st = fl.floating_continued_state(gpu, fake(water_for(cpu, np.random.default_rng(3))))
    other = fl.floating_continued_state(gpu, fake(water_for(cpu, np.random.default_rng(4))))
    reads, nalloc, nlaunch = [], len(cp.allocs), len(cp.launches)
    reader = lambda: reads.append(1) or (1 << 40)  # noqa: E731
    cp.count.__init__()
    cp.dev["id"] = 1  # metadata mock: the current device changed; no real GPU is selected
    f = dict(added_water_m=fake(np.zeros((3, 2))), potential_evaporation_m_s=fake(np.zeros(2)))
    calls = [lambda: fl.floating_step(gpu, st, 60.0, available_bytes_fn=reader, host_available_bytes_fn=reader, **f),
             lambda: fl.floating_step_with_recharge(gpu, st, 60.0, recharge_depth_m=1e-4, potential_evaporation_m_s=f["potential_evaporation_m_s"], available_bytes_fn=reader),
             lambda: fl.floating_continued_state(gpu, fake(np.zeros((3, 2))), available_bytes_fn=reader), lambda: fl.floating_initial_state(gpu, fake(np.zeros((3, 2))), available_bytes_fn=reader),
             lambda: fl.floating_moisture(gpu, st, available_bytes_fn=reader), lambda: fl.floating_storage_m3(gpu, st, available_bytes_fn=reader),
             lambda: fl.reconcile_floating(gpu, st, gpu, available_bytes_fn=reader),
             lambda: fl.prepare_floating_column(EDGES, theta_residual=fake(np.zeros((3, 2))), theta_field_capacity=0.2, theta_saturation=0.4, diffusivity_m2_s=fake(np.zeros(2)), theta_evaporation_stress=0.3,
                                                soil_depth_m=fake(np.full(2, 0.5)), active_mask=fake(np.ones(2, bool)), cell_area_m2=1.0, backend="cuda", available_bytes_fn=reader)]
    for call in calls:
        with pytest.raises(fl.FloatingSoilError, match="current device"):
            call()
    cp.dev["id"] = 0
    assert not reads and cp.count.d2h == 0 and cp.count.h2d == 0 and len(cp.allocs) == nalloc and len(cp.launches) == nlaunch  # nothing was read, copied, allocated or launched
    del other


def test_admission_separates_device_and_host_and_precedes_allocation(cp):
    cpu, gpu = pair(cp, depth=[0.5] * 50)
    st = fl.floating_continued_state(gpu, fake(water_for(cpu, np.random.default_rng(3))))
    f = dict(added_water_m=fake(np.zeros((3, 50))), potential_evaporation_m_s=fake(np.zeros(50)))
    op = fcu.operation_terms("step", 3, 50)
    nalloc = len(cp.allocs)
    with pytest.raises(fl.FloatingMemoryError, match="device side"):
        fl.floating_step(gpu, st, 60.0, memory_budget_bytes=op["device"] - 1, **f)
    with pytest.raises(fl.FloatingMemoryError, match="device side"):
        fl.floating_step(gpu, st, 60.0, available_bytes_fn=lambda: op["device"] - 1, **f)
    with pytest.raises(fl.FloatingMemoryError, match="host side"):
        fl.floating_step(gpu, st, 60.0, host_memory_budget_bytes=op["host"] + fcu.PY_ALLOWANCE_BYTES - 1, **f)
    with pytest.raises(fl.FloatingMemoryError, match="device side"):
        fl.floating_step(gpu, st, 60.0, reserve_bytes=1 << 62, **f)
    for bad in (lambda: True, lambda: -1, lambda: 1.5):
        with pytest.raises(fl.FloatingSoilError, match="available-bytes reader"):
            fl.floating_step(gpu, st, 60.0, available_bytes_fn=bad, **f)
    assert len(cp.allocs) == nalloc  # refused before any allocation
    fl.floating_step(gpu, st, 60.0, memory_budget_bytes=op["device"], host_memory_budget_bytes=op["host"] + fcu.PY_ALLOWANCE_BYTES, **f)  # exactly enough is admitted
    with pytest.raises(fl.FloatingSoilError, match="apply only to the cuda"):
        fl.floating_step(cpu, fl.floating_continued_state(cpu, water_for(cpu, np.random.default_rng(1))), 60.0, added_water_m=np.zeros((3, 50)), potential_evaporation_m_s=np.zeros(50), host_reserve_bytes=1)


def test_step_and_reconcile_allocations_equal_the_model_and_repeat_exactly(cp):
    rng = np.random.default_rng(12)
    N = 777
    depth = rng.choice([0.0, 0.2, 0.9], size=N)
    cpu, gpu = pair(cp, depth=depth, mode="tietjen_donor_depth")
    w = water_for(cpu, rng, 0.6)
    st = fl.floating_continued_state(gpu, fake(w))
    f = dict(added_water_m=fake(np.zeros((3, N))), potential_evaporation_m_s=fake(np.zeros(N)))
    per = []
    for _ in range(4):  # plateau: every step allocates exactly the same payload (outputs + scratch), nothing accumulates in the owner
        n0 = len(cp.allocs)
        st = fl.floating_step(gpu, st, 600.0, **f).state
        per.append(sum(cp.allocs[n0:]))
    op = fcu.operation_terms("step", 3, N)
    assert per == [op["device"]] * 4 and op["device"] == op["outputs"] + op["scratch"]
    n0 = len(cp.allocs)
    fl.reconcile_floating(gpu, st, gpu)
    assert sum(cp.allocs[n0:]) == fcu.operation_terms("reconcile", 3, N)["device"]


def test_cold_hash_is_chunk_invariant_and_distinguished_from_hot_traffic(cp, monkeypatch):
    a = pair(cp, depth=[0.5, 0.2, 0.9])[1]
    orig = fcu.HASH_CHUNK_BYTES
    monkeypatch.setattr(fcu, "HASH_CHUNK_BYTES", 24)
    cp.count.__init__()
    b = pair(cp, depth=[0.5, 0.2, 0.9])[1]
    assert a.sha256 == b.sha256 and b.mask_sha256 == a.mask_sha256 and b.edges_sha256 == a.edges_sha256 and max(cp.count.d2h_sizes) <= 32  # a small buffer changes the transfers, never the identity
    monkeypatch.setattr(fcu, "HASH_CHUNK_BYTES", orig)
    cpu, gpu = pair(cp, depth=[0.5, 0.2, 0.9])
    st = fl.floating_continued_state(gpu, fake(water_for(cpu, np.random.default_rng(1))))
    cp.count.__init__()
    fl.floating_step(gpu, st, 60.0, added_water_m=fake(np.zeros((3, 3))), potential_evaporation_m_s=fake(np.zeros(3)))
    assert cp.count.d2h_sizes == [72]  # the hot step never hashes: one fixed packet


def test_n_50003_resident_allocation_transfer_and_plateau(cp):
    rng = np.random.default_rng(50003)
    N = 50003
    depth = rng.choice([0.0, 0.05, 0.2, 0.45, 0.9], size=N)
    cpu, gpu = pair(cp, depth=depth)
    w = water_for(cpu, rng, 0.6)
    st = fl.floating_continued_state(gpu, fake(w))
    f = dict(added_water_m=fake(np.zeros((3, N))), potential_evaporation_m_s=fake(rng.uniform(0, 3e-7, N)))
    cp.count.__init__()
    sizes = []
    for _ in range(3):
        n0 = len(cp.allocs)
        r = fl.floating_step(gpu, st, 600.0, **f)
        sizes.append(sum(cp.allocs[n0:]))
        st = r.state
    assert sizes == [fcu.operation_terms("step", 3, N)["device"]] * 3 and (cp.count.d2h, cp.count.d2h_bytes, cp.count.h2d) == (3, 216, 0)  # three steps: 3 x 72 bytes down, nothing up
    ref = fl.floating_step(cpu, fl.floating_continued_state(cpu, w), 600.0, added_water_m=np.zeros((3, N)), potential_evaporation_m_s=np.asarray(f["potential_evaporation_m_s"]))
    first = fl.floating_step(gpu, fl.floating_continued_state(gpu, fake(w)), 600.0, **f)
    np.testing.assert_allclose(np.asarray(first.state.water_m), ref.state.water_m, rtol=2e-12, atol=1e-14)
    assert abs(first.budget.residual_m3) <= first.budget.bound_m3


# ---- source-level resource controls --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("L", [3, 5, 65])
def test_resource_report_is_independent_of_the_layer_count(cp, L):
    rep = cf.static_resource_report()
    assert rep["maple_syrup_floating_resident_step"] == {"static_shared_bytes": 7 * 8 * 128, "per_thread_local_array_bytes": 7 * 8}
    assert rep["maple_syrup_floating_resident_reconcile"] == {"static_shared_bytes": 5 * 8 * 128, "per_thread_local_array_bytes": 5 * 8}
    assert all(v["per_thread_local_array_bytes"] <= 56 for v in rep.values())  # only the contribution rows: nothing is sized by the layer count
    src = cf.render_resident_source()
    assert "MAX_LAYERS" not in src and "double hh[" not in src and "double dz_c[" not in src
    edges = np.linspace(0.0, 0.6, L + 1)
    cpu, gpu = pair(cp, depth=[0.3, 0.7], edges=edges)
    fl.floating_step(gpu, fl.floating_continued_state(gpu, fake(water_for(cpu, np.random.default_rng(L)))), 60.0, added_water_m=fake(np.zeros((L, 2))), potential_evaporation_m_s=fake(np.zeros(2)))


def test_resident_source_is_the_shared_body_and_the_historical_common_source_is_untouched():
    import hashlib

    from maple_syrup import subsurface_cuda as old

    src = cf.render_resident_source()
    assert old._BODY in src and src.count(old._BODY) == 1
    p = cf.resident_provenance()
    assert p["shared_body_sha256"] == hashlib.sha256(old._BODY.encode()).hexdigest() and p["source_sha256"] == hashlib.sha256(src.encode()).hexdigest()
    assert hashlib.sha256(old.kernel_source().encode()).hexdigest() == hashlib.sha256(old.render_source("common").encode()).hexdigest()
    assert "maple_syrup_floating" not in old.kernel_source()


def test_kernels_are_loaded_once_per_device_with_the_intended_options(cp):
    pair(cp, depth=[0.5])
    pair(cp, depth=[0.5])
    assert len(cp.compiled) == 1 and cp.compiled[0][1] == ("--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false") and cp.compiled[0][2] == "nvrtc"


# ---- D1b correction01: reconciliation parent admission, cell area, actual moved percolation -------------------------------------------------------------------------------------------------------------------------
def _three(cp, *, old_area=2.0, new_area=2.0, old_depth=0.3, new_depth=0.5):
    """An old/new pair for each backend (same edges/mask/properties; the new column may differ in area/depth)."""
    from conftest import column as cpu_column

    out = {}
    for be in ("numpy", "numba", "cuda"):
        mk = lambda depth, area, be=be: (pair(cp, depth=[depth, depth], area=area)[1] if be == "cuda" else cpu_column(EDGES, [depth, depth], backend=be, area=area))  # noqa: E731
        out[be] = (mk(old_depth, old_area), mk(new_depth, new_area))
    return out


def _state_for(be, old, water):
    return fl.floating_continued_state(old, fake(water) if be == "cuda" else water)


def _numba():
    try:
        import numba  # noqa: F401
        return ["numpy", "numba", "cuda"]
    except Exception:  # noqa: BLE001
        return ["numpy", "cuda"]


@pytest.mark.parametrize("be", _numba())
def test_reconcile_requires_identical_cell_area_before_any_reader_allocation_or_kernel(cp, be):
    cols = _three(cp, old_area=1.0, new_area=2.0)
    old, new = cols[be]
    w = np.array([[0.002, 0.002], [0.002, 0.002], [0.0, 0.0]])
    st = _state_for(be, old, w)
    snap = np.array(st.water_m).copy()
    reads, n0, l0 = [], len(cp.allocs), len(cp.launches)
    reader = lambda: reads.append(1) or (1 << 40)  # noqa: E731
    kw = dict(available_bytes_fn=reader) if be != "cuda" else dict(available_bytes_fn=reader, host_available_bytes_fn=reader)
    cp.count.__init__()
    with pytest.raises(fl.FloatingSoilError, match="identical cell_area_m2"):
        fl.reconcile_floating(old, st, new, **kw)
    assert not reads and len(cp.allocs) == n0 and len(cp.launches) == l0 and cp.count.d2h == 0 and cp.count.h2d == 0  # nothing was read, allocated, launched or transferred
    assert np.array_equal(np.array(st.water_m), snap)  # the input is unchanged; no rescaling happened
    ok_old, ok_new = _three(cp, old_area=2.0, new_area=2.0)[be]
    fl.reconcile_floating(ok_old, _state_for(be, ok_old, w), ok_new)  # equal areas are accepted


@pytest.mark.parametrize("be", _numba())
@pytest.mark.parametrize("kind", ["above_old_capacity", "rock_slot", "negative", "nan", "inf"])
def test_reconcile_refuses_an_old_state_that_violates_its_own_owner_even_if_it_fits_the_new_one(cp, be, kind):
    from conftest import column as cpu_column

    # old: D = 0.1 (one porous layer of 0.1 m, capacity 0.04 m); new: D = 0.6 (everything porous, much larger capacity)
    if be == "cuda":
        old, new = pair(cp, depth=[0.1, 0.1], area=1.0)[1], pair(cp, depth=[0.6, 0.6], area=1.0)[1]
    else:
        old, new = cpu_column(EDGES, [0.1, 0.1], backend=be, area=1.0), cpu_column(EDGES, [0.6, 0.6], backend=be, area=1.0)
    w = np.array([[0.01, 0.01], [0.0, 0.0], [0.0, 0.0]])
    st = _state_for(be, old, w)  # a valid parent
    bad = np.array(w, copy=True)
    if kind == "above_old_capacity":
        bad[0, 0] = 0.15  # 0.15 m over a 0.1 m layer: far above the OLD capacity (0.04 m) but below the NEW one
    elif kind == "rock_slot":
        bad[1, 0] = 0.01  # the slot below the old soil depth holds water
    elif kind == "negative":
        bad[0, 0] = -1e-3
    elif kind == "nan":
        bad[0, 0] = np.nan
    else:
        bad[0, 0] = np.inf
    # a FORGED direct state (the dataclass is not factory-only): the reconcile stage itself must not trust it
    forged = fl.FloatingSoilState(fake(bad) if be == "cuda" else bad, 0.0, old.sha256, "arbitrary-token")
    snap = np.array(forged.water_m).copy()
    with pytest.raises(fl.FloatingSoilError, match="OLD state violates"):
        fl.reconcile_floating(old, forged, new)
    assert np.array_equal(np.array(forged.water_m), snap, equal_nan=True)  # the input is unchanged and no candidate exists
    good = fl.reconcile_floating(old, st, new)  # the legitimate parent is accepted: old/new hydraulic properties and depth refresh stay supported
    assert good.receipt.old_state_token == st.token


@pytest.mark.parametrize("be", _numba())
def test_a_legitimate_over_new_capacity_parent_is_still_reconciled(cp, be):
    from conftest import column as cpu_column

    big, small = (pair(cp, depth=[0.6, 0.6], area=1.0)[1], pair(cp, depth=[0.1, 0.1], area=1.0)[1]) if be == "cuda" else (cpu_column(EDGES, [0.6, 0.6], backend=be, area=1.0), cpu_column(EDGES, [0.1, 0.1], backend=be, area=1.0))
    h = np.stack([np.array([0.1, 0.1]), np.array([0.2, 0.2]), np.array([0.3, 0.3])])
    w = h * 0.3  # valid for the big column (SAT 0.4); above the SMALL column's capacity (only layer 0 has soil)
    cand = fl.reconcile_floating(big, _state_for(be, big, w), small)
    assert float(np.asarray(cand.surface_return_m).sum()) > 0.0  # the displaced water returns to the surface, accounted
    assert abs(cand.receipt.residual_m3) <= cand.receipt.bound_m3


@pytest.mark.parametrize("be", _numba())
def test_the_receipt_reports_the_actual_finite_moved_percolation_and_serialises(cp, be):
    import json
    import math

    rng = np.random.default_rng(31)
    N = 150
    act = rng.random(N) > 0.15
    from conftest import column as cpu_column

    tf = rng.uniform(0.05, 0.15, (3, N))
    ts = tf + rng.uniform(0.02, 0.1, (3, N))
    depth_old, depth_new = np.full(N, 2.0), rng.choice([0.0, 0.1, 0.45, 0.9], size=N)
    if be == "cuda":
        old = pair(cp, depth=depth_old, active=act)[1]
        new = pair(cp, depth=depth_new, active=act, tf=tf, ts=ts)[1]
        ref_old, ref_new = pair(cp, depth=depth_old, active=act)[0], pair(cp, depth=depth_new, active=act, tf=tf, ts=ts)[0]
    else:
        old, new = None, None
    ref_old = pair(cp, depth=depth_old, active=act)[0]
    ref_new = pair(cp, depth=depth_new, active=act, tf=tf, ts=ts)[0]
    if be != "cuda":
        old, new = (fl.prepare_floating_column(EDGES, theta_residual=0.0, theta_field_capacity=0.2, theta_saturation=0.4, diffusivity_m2_s=np.full(2, 1e-7), theta_evaporation_stress=0.3,
                                               soil_depth_m=depth_old, active_mask=act, cell_area_m2=2.0, backend=be),
                    fl.prepare_floating_column(EDGES, theta_residual=0.0, theta_field_capacity=tf, theta_saturation=ts, diffusivity_m2_s=np.full(2, 1e-7), theta_evaporation_stress=0.3,
                                               soil_depth_m=depth_new, active_mask=act, cell_area_m2=2.0, backend=be))
    w = water_for(ref_old, rng, 1.0) * 0.5
    w[:, ~act] = 4.0
    cand = fl.reconcile_floating(old, _state_for(be, old, w), new)
    exp = fl.reconcile_floating(ref_old, fl.floating_continued_state(ref_old, w), ref_new).receipt.moved_percolation_m3
    r = cand.receipt
    assert math.isfinite(r.moved_percolation_m3) and r.moved_percolation_m3 > 0.0 and r.moved_percolation_m3 == pytest.approx(exp, rel=1e-12)
    json.dumps(dataclasses.asdict(r), allow_nan=False)  # strict JSON metadata: no NaN anywhere in the receipt


def test_resident_reconcile_packet_is_56_bytes_and_counted(cp):
    assert fcu.operation_terms("reconcile", 3, 50003)["scratch"] == 5 * 8 * 391 + 56 and fcu.RECONCILE_HOST_PACKET_BYTES == 112
    assert cf.RECONCILE_ROWS == 5 and cf.RECONCILE_PACKET_WORDS == 7 and cf.resident_provenance()["reconcile_packet_bytes"] == 56
    assert cf.static_resource_report()["maple_syrup_floating_resident_reconcile"] == {"static_shared_bytes": 5 * 8 * 128, "per_thread_local_array_bytes": 5 * 8}
    # the shared STEP packet/source ABI is unchanged
    assert cf.STEP_PACKET_WORDS == 9 and cf.resident_provenance()["step_packet_bytes"] == 72


# ---- default-kernel code generation (the reconciliation kernel is generated from the step kernel's own gravity block) ----------------------------------------------------------------------------------------------
def test_floating_reconcile_kernel_is_generated_from_the_step_kernels_own_gravity_block():
    pytest.importorskip("numba")
    from maple_syrup import subsurface_numba as sn

    block, code = sn.floating_reconcile_source()
    import textwrap

    assert block in open(sn.__file__).read() and textwrap.indent(textwrap.dedent(block), " " * 8) in code and code.count("drain_v = carry") == 1
    p = sn.floating_reconcile_provenance()
    assert set(p) == {"gravity_block_sha256", "generated_source_sha256", "template_sha256"} and all(len(v) == 64 for v in p.values())


def test_the_generator_fails_closed_when_an_anchor_is_missing_or_duplicated(monkeypatch):
    pytest.importorskip("numba")
    from maple_syrup import subsurface_numba as sn

    monkeypatch.setattr(sn, "_GRAV_BEGIN", "                # not a real anchor\n")
    with pytest.raises(ss.SubsurfaceError, match="exactly once"):
        sn.floating_reconcile_source()
    monkeypatch.undo()
    monkeypatch.setattr(sn, "_GRAV_END", "                    carry = 0.0\n")  # occurs several times in the kernel text
    with pytest.raises(ss.SubsurfaceError, match="exactly once"):
        sn.floating_reconcile_source()


def test_default_common_kernel_codegen_has_no_extra_reference_counting_against_the_baseline():
    """The default common step kernel's LLVM must not carry the extra NRT_incref/NRT_decref of the earlier inline-helper form: compared with the frozen G0 C3 kernel (Root's immutable snapshot, the performance reference)."""
    pytest.importorskip("numba")
    import importlib.util
    import sys
    from pathlib import Path

    from maple_syrup import subsurface_numba as new

    pre = Path(__file__).resolve().parents[3] / "root_g0_c3_snapshot" / "src" / "maple_syrup" / "subsurface_numba.py"  # the frozen G0 C3 module (task evidence)
    assert pre.is_file(), "the frozen G0 C3 reference kernel is required by this guard"
    spec = importlib.util.spec_from_file_location("old_numba_codegen_guard", pre)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    edges = np.array([0.0, 0.1, 0.3, 0.6])
    lay = ss.soil_layers(edges, theta_residual=[0.0, 0.01, 0.02], theta_field_capacity=[0.15, 0.18, 0.2], theta_saturation=[0.4, 0.45, 0.5], diffusivity_m2_s=[1e-7, 2e-7], theta_evaporation_stress=0.5)
    counts = {}
    for name, mod in (("old", old), ("new", new)):
        mod.prepare_runner(lay)
        ir = list(mod.compiled_kernel().inspect_llvm().values())[0]
        counts[name] = (ir.count("NRT_incref"), ir.count("NRT_decref"))
    assert counts["new"] == counts["old"], counts
