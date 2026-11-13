"""The rendered spatial CUDA withdrawal/availability EXECUTED on the host (real threads + barrier, hostb1c.py) through the REAL Python launchers and public API on a numpy-backed CuPy plumbing double, compared with
the accepted NumPy spatial path (arrays bit for bit: the law has no libm term; budget fields within the roundoff of a fixed-tree sum), plus the plumbing contracts (allocations equal the model, one 64-byte read, admission
before any allocation/compile on both sides, device/type refusals, typed OOM, no mutation, release). NOT hardware evidence; no fake CUDA-tagged owner is offered as proof of device behaviour. Author-written."""
from __future__ import annotations

import gc
import hashlib
import itertools
import weakref

import numpy as np
import pytest
from conftest import BIG, a2, a3, bits, device_owner, fx, hostb1c, hostgpu, needs_host, sp, ss, st, sw, tc

import maple.core.backend as backend

pytestmark = needs_host
SIZES = (1, 7, 127, 128, 129, 257)


def make(p, mask, mode="constant", **kw):
    return a2.prepare(a2.make_spec(p, mode=mode, **kw), mask=mask)


def wkind(rng, p, kind):
    L, N = p["res"].shape
    dz = np.diff(p["edges"], axis=0)
    res, sat = p["res"] * dz, p["sat"] * dz
    return {"zero": np.zeros((L, N)), "at_floor": res.copy(), "below_floor": res * rng.uniform(0, 1, (L, N)), "saturated": sat.copy(), "random": sat * rng.uniform(0, 1, (L, N)),
            "mixed": np.where(rng.random((L, N)) < 0.3, 0.0, sat * rng.uniform(0, 1, (L, N)))}[kind]


def requests(rng, w, p, mask, regime):
    res = p["res"] * np.diff(p["edges"], axis=0)
    avail = np.maximum(w - res, 0.0)
    return {"zero": np.zeros_like(w), "under": avail * 0.5, "equal": avail.copy(), "over": avail + 0.01, "random": rng.uniform(0, 0.05, w.shape)}[regime] * mask


def both(p, mask, w, q, t=7.0, **kw):
    np_owner = make(p, mask, **kw)
    cpu_op = sw.prepare_spatial_withdrawal(np_owner, memory_budget_bytes=BIG)
    ref = sw.spatial_withdraw(cpu_op, sp.spatial_column_state(np_owner, w, t, memory_budget_bytes=BIG), q, memory_budget_bytes=BIG)
    return np_owner, ref


def gpu_run(cupy, np_owner, w, q, t=7.0):
    dev = device_owner(np_owner)
    op = sw.prepare_spatial_withdrawal(dev, memory_budget_bytes=BIG)
    return dev, op, sw.spatial_withdraw(op, sp.spatial_column_state(dev, fx(w), t, memory_budget_bytes=BIG), fx(q), memory_budget_bytes=BIG)


def assert_equal(ref, got):
    for f in ("actual_m", "unmet_m", "scale"):
        np.testing.assert_array_equal(bits(getattr(got, f)), bits(getattr(ref, f)), err_msg=f)
    np.testing.assert_array_equal(bits(got.state.water_m), bits(ref.state.water_m))
    assert got.state.t_s == ref.state.t_s and got.state.namespace == "cupy" and got.state.component_tag == "spatial_explicit_cuda_state"
    for f in ("storage_before_m3", "storage_after_m3", "requested_m3", "actual_m3", "unmet_m3"):
        assert getattr(got.budget, f) == pytest.approx(getattr(ref.budget, f), rel=1e-11, abs=1e-15), f
    assert got.budget.n_terms == ref.budget.n_terms and got.budget.min_water_after_m == ref.budget.min_water_after_m and got.budget.max_requested_m == ref.budget.max_requested_m
    assert abs(got.budget.residual_m3) <= got.budget.bound_m3 and abs(got.budget.request_residual_m3) <= got.budget.bound_m3


@pytest.mark.parametrize("L", [1, 3])
def test_block_boundary_sizes_and_masks_match_the_cpu_spatial_path_bit_for_bit(cupy, ncells, L):
    rng = np.random.default_rng(ncells + L)
    N = ncells
    p = a2.params(rng, L, N)
    mask = rng.random(N) < 0.7
    mask[0] = True
    for kind, regime in (("random", "random"), ("mixed", "over"), ("below_floor", "under"), ("zero", "equal")):
        w = wkind(rng, p, kind)
        q = requests(rng, w, p, mask, regime)
        np_owner, ref = both(p, mask, w, q)
        _, _, got = gpu_run(cupy, np_owner, w, q)
        assert_equal(ref, got)


@pytest.mark.parametrize("N", [1, 128, 129])
def test_all_inactive_domain_contributes_nothing_and_keeps_the_state(cupy, N):
    rng = np.random.default_rng(N)
    p = a2.params(rng, 3, N)
    mask = np.zeros(N, bool)
    w = wkind(rng, p, "random")
    np_owner, ref = both(p, mask, w, np.zeros((3, N)))
    _, _, got = gpu_run(cupy, np_owner, w, np.zeros((3, N)))
    assert_equal(ref, got)
    np.testing.assert_array_equal(bits(got.state.water_m), bits(w))
    assert got.budget.storage_before_m3 == got.budget.actual_m3 == got.budget.requested_m3 == 0.0


@pytest.mark.parametrize("L", [1, 2, 3, 5])
@pytest.mark.parametrize("kind", ["zero", "at_floor", "below_floor", "saturated", "mixed", "random"])
def test_regimes_and_stack_heights_match(cupy, L, kind):
    rng = np.random.default_rng(L * 11 + len(kind))
    N = 130
    p = a2.params(rng, L, N)
    mask = rng.random(N) < 0.8
    w = wkind(rng, p, kind)
    np_owner = make(p, mask)
    for regime in ("zero", "under", "equal", "over", "random"):
        q = requests(rng, w, p, mask, regime)
        _, ref = both(p, mask, w, q)
        _, _, got = gpu_run(cupy, np_owner, w, q)
        assert_equal(ref, got)


@pytest.mark.parametrize("L", [1, 2, 5])
def test_scalar_shared_and_cell_raw_map_layouts_match(cupy, L):
    rng = np.random.default_rng(80 + L)
    N = 129
    mask = rng.random(N) < 0.85
    mask[0] = True
    combos = [("shared", "scalar", "scalar", "scalar", "shared"), ("cell", "cell", "cell", "cell", "cell"), ("shared", "cell", "shared", "scalar", "shared"), ("cell", "scalar", "cell", "shared", "scalar"),
              ("shared", "shared", "shared", "cell", "cell")]
    for edges, res, fc, sat, iface in combos:
        inputs, p = a3.layout_case(rng, L, N, edges=edges, res=res, fc=fc, sat=sat, iface=iface)
        np_owner = a2.prepare(a3.spec_from(inputs, N), mask=mask)
        w = wkind(rng, p, "mixed")
        q = requests(rng, w, p, mask, "random")
        cpu_op = sw.prepare_spatial_withdrawal(np_owner, memory_budget_bytes=BIG)
        ref = sw.spatial_withdraw(cpu_op, sp.spatial_column_state(np_owner, w, 0.0, memory_budget_bytes=BIG), q, memory_budget_bytes=BIG)
        _, _, got = gpu_run(cupy, np_owner, w, q, 0.0)
        assert_equal(ref, got)


def test_dyadic_delivery_example_is_exact(cupy):
    spec = sp.spatial_layer_spec(np.array([0.0, 1.0, 2.0]), theta_residual=0.0, theta_field_capacity=0.25, theta_saturation=np.array([0.5, 0.5]), theta_evaporation_stress=0.5, diffusivity_m2_s=0.0, n_cells=1)
    np_owner = sp.prepare_spatial_column(spec, np.ones(1, bool), 4.0, backend="numpy", memory_budget_bytes=BIG)
    w, q = np.array([[25.0], [200.0]]) / 1024.0, np.array([[50.0], [50.0]]) / 1024.0
    _, _, r = gpu_run(cupy, np_owner, w, q)
    np.testing.assert_array_equal(np.asarray(r.actual_m), np.array([[25.0], [50.0]]) / 1024.0)
    np.testing.assert_array_equal(np.asarray(r.state.water_m), np.array([[0.0], [150.0]]) / 1024.0)
    np.testing.assert_array_equal(np.asarray(r.scale), np.array([[0.5], [1.0]]))
    assert r.budget.actual_m3 / r.budget.requested_m3 == 0.75 and r.state.t_s == 7.0


# ---- invalid values ------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("label,mut", [
    ("nan_water", lambda w, q, m: w.__setitem__((1, 2), np.nan)), ("negative_water", lambda w, q, m: w.__setitem__((0, 3), -1e-3)), ("over_saturated", lambda w, q, m: w.__setitem__((2, 4), 10.0)),
    ("nan_water_inactive", lambda w, q, m: w.__setitem__((1, int(np.flatnonzero(~m)[0])), np.nan)), ("nan_request", lambda w, q, m: q.__setitem__((1, 2), np.nan)),
    ("inf_request", lambda w, q, m: q.__setitem__((0, 1), np.inf)), ("negative_request", lambda w, q, m: q.__setitem__((2, 5), -1e-6)),
    ("positive_request_inactive", lambda w, q, m: q.__setitem__((1, int(np.flatnonzero(~m)[0])), 1e-6)),
    ("two_kinds", lambda w, q, m: (q.__setitem__((0, 1), -1.0), q.__setitem__((1, int(np.flatnonzero(~m)[0])), 1e-6)))])
def test_invalid_inputs_raise_the_identical_typed_error_on_every_cell_with_no_mutation(cupy, label, mut):
    rng = np.random.default_rng(3)
    N = 140
    p = a2.params(rng, 3, N)
    mask = rng.random(N) < 0.7
    mask[0] = True
    mask[-1] = False
    w = wkind(rng, p, "random")
    q = requests(rng, w, p, mask, "random")
    mut(w, q, mask)
    np_owner = make(p, mask)
    cpu_op = sw.prepare_spatial_withdrawal(np_owner, memory_budget_bytes=BIG)
    with pytest.raises(ss.SubsurfaceError) as e_cpu:
        sw.spatial_withdraw(cpu_op, sp.spatial_column_state(np_owner, w, 7.0, memory_budget_bytes=BIG), q, memory_budget_bytes=BIG)
    dev = device_owner(np_owner)
    op = sw.prepare_spatial_withdrawal(dev, memory_budget_bytes=BIG)
    state, req = sp.spatial_column_state(dev, fx(w), 7.0, memory_budget_bytes=BIG), fx(q)
    before = (hashlib.sha256(np.asarray(state.water_m).tobytes()).hexdigest(), hashlib.sha256(np.asarray(req).tobytes()).hexdigest(), state.t_s)
    with pytest.raises(ss.SubsurfaceError) as e_dev:
        sw.spatial_withdraw(op, state, req, memory_budget_bytes=BIG)
    assert str(e_dev.value) == str(e_cpu.value)  # same reason text (the flag max aggregation is the common one)
    assert (hashlib.sha256(np.asarray(state.water_m).tobytes()).hexdigest(), hashlib.sha256(np.asarray(req).tobytes()).hexdigest(), state.t_s) == before
    if "water" in label:
        with pytest.raises(ss.SubsurfaceError, match="soil water not finite"):
            sw.spatial_layer_availability_m(op, state, memory_budget_bytes=BIG)


def test_unrepresentable_global_volumes_are_a_typed_refusal_before_the_conservation_helper(cupy):
    spec = sp.spatial_layer_spec(np.array([0.0, 1.0]), theta_residual=0.0, theta_field_capacity=0.25, theta_saturation=0.5, theta_evaporation_stress=0.375, diffusivity_m2_s=0.0, n_cells=8)
    np_owner = sp.prepare_spatial_column(spec, np.ones(8, bool), 1e308, backend="numpy", memory_budget_bytes=BIG)
    dev = device_owner(np_owner)
    op = sw.prepare_spatial_withdrawal(dev, memory_budget_bytes=BIG)
    state = sp.spatial_column_state(dev, fx(np.full((1, 8), 0.2)), 3.0, memory_budget_bytes=BIG)
    with pytest.raises(ss.SubsurfaceError, match="not representable in FP64") as e:
        sw.spatial_withdraw(op, state, fx(np.full((1, 8), 0.2)), memory_budget_bytes=BIG)
    assert type(e.value) is ss.SubsurfaceError and "nothing was published" in str(e.value)


# ---- availability ---------------------------------------------------------------------------------------------------------------------------------
def test_availability_equals_the_cpu_definition_bitwise_and_zeroes_inactive_cells(cupy, ncells):
    rng = np.random.default_rng(ncells)
    p = a2.params(rng, 3, ncells)
    mask = rng.random(ncells) < 0.6
    mask[0] = True
    np_owner = make(p, mask)
    dev = device_owner(np_owner)
    cpu_op = sw.prepare_spatial_withdrawal(np_owner, memory_budget_bytes=BIG)
    op = sw.prepare_spatial_withdrawal(dev, memory_budget_bytes=BIG)
    for kind in ("zero", "at_floor", "below_floor", "saturated", "random"):
        w = wkind(rng, p, kind)
        ref = sw.spatial_layer_availability_m(cpu_op, sp.spatial_column_state(np_owner, w, 0.0, memory_budget_bytes=BIG), memory_budget_bytes=BIG)
        got = sw.spatial_layer_availability_m(op, sp.spatial_column_state(dev, fx(w), 0.0, memory_budget_bytes=BIG), memory_budget_bytes=BIG)
        np.testing.assert_array_equal(bits(got), bits(ref))
        assert type(got) is hostgpu.FakeArray and not np.any(np.asarray(got)[:, ~mask])


# ---- the no-fusion requirement ----------------------------------------------------------------------------------------------------------------------
def fma_available():
    try:
        return " fma" in open("/proc/cpuinfo").read().split("flags")[1].split("\n")[0] + " "
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not fma_available(), reason="the host CPU has no FMA: the fusion hazard cannot be demonstrated here (no claim)")
def test_contracting_the_raw_products_would_change_bits_so_the_spatial_module_forbids_it():
    """Compile the SAME rendered text with contraction ALLOWED (`-mfma -ffp-contract=fast`): at least one output bit differs from the unfused result. This is why the spatial module is compiled with
    `--fmad=false` (tested in test_source_and_render) and the products are rounded before use."""
    rng = np.random.default_rng(1)
    N = 4096
    p = a2.params(rng, 3, N)
    mask = np.ones(N, bool)
    np_owner = make(p, mask)
    w = wkind(rng, p, "random")
    q = requests(rng, w, p, mask, "over")
    outs = []
    for flags in (("-ffp-contract=off",), ("-O2", "-mfma", "-ffp-contract=fast")):  # GCC fuses at -O2 and above
        so_ = hostb1c.build(tc.render_source(), flags)
        cp = hostb1c.FakeCupy(so_)
        dev = device_owner(np_owner)
        from maple_syrup.conservation import volume_roundoff_bound_m3

        with tc._LOCK:
            tc._FUNCTIONS.clear()
        out_w, actual, unmet, scale, packet = tc.withdraw(cp, dev, fx(w), fx(q), float(volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)))
        outs.append((np.asarray(out_w).copy(), np.asarray(actual).copy()))
    with tc._LOCK:
        tc._FUNCTIONS.clear()
    assert not (np.array_equal(bits(outs[0][0]), bits(outs[1][0])) and np.array_equal(bits(outs[0][1]), bits(outs[1][1])))


# ---- plumbing: allocations, transfers, lifetimes, quotas ------------------------------------------------------------------------------------------------
def test_withdraw_makes_exactly_the_modelled_allocations_one_64_byte_read_and_no_upload(cupy, monkeypatch):
    rng = np.random.default_rng(1)
    L, N = 4, 300
    p = a2.params(rng, L, N)
    mask = rng.random(N) < 0.8
    np_owner = make(p, mask)
    dev = device_owner(np_owner)
    op = sw.prepare_spatial_withdrawal(dev, memory_budget_bytes=BIG)
    assert [m for m in cupy.modules] and not any(f for f in [])  # compiled at preparation
    w = wkind(rng, p, "random")
    state = sp.spatial_column_state(dev, fx(w), 0.0, memory_budget_bytes=BIG)
    q = fx(requests(rng, w, p, mask, "random"))
    cupy.allocations.clear()
    seen = {"to_host": [], "to_device": []}
    real_h, real_d = backend.to_host, backend.to_device
    monkeypatch.setattr(backend, "to_host", lambda a: (seen["to_host"].append((a.dtype.name, a.nbytes)), real_h(a))[1])
    monkeypatch.setattr(backend, "to_device", lambda a, xp: (seen["to_device"].append(a.nbytes), real_d(a, xp))[1])
    r = sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG)
    t = tc.operation_terms("withdraw", L, N)
    assert sum(a[3] for a in cupy.allocations) == t["outputs"] + t["scratch"]  # every device byte requested is in the model, nothing else
    nb = (N + 127) // 128
    assert sorted((a[1], a[2]) for a in cupy.allocations) == sorted([((L, N), "float64")] * 4 + [((8 * nb,), "float64"), ((8,), "float64")])
    assert seen["to_host"] == [("float64", 64)] and seen["to_device"] == []  # ONE counted 64-byte read, zero uploads
    assert type(r.state.water_m) is hostgpu.FakeArray and r.state.device_id == 0
    cupy.allocations.clear()
    seen["to_host"].clear()
    sw.spatial_layer_availability_m(op, state, memory_budget_bytes=BIG)
    a = tc.operation_terms("availability", L, N)
    assert sum(x[3] for x in cupy.allocations) == a["outputs"] + a["scratch"] and seen["to_host"] == [("uint64", 8)] and seen["to_device"] == []


def test_preparation_compiles_once_without_a_physical_dummy_launch_and_functions_only_are_cached(cupy):
    rng = np.random.default_rng(2)
    p = a2.params(rng, 3, 50)
    np_owner = make(p, np.ones(50, bool))
    dev = device_owner(np_owner)
    cupy.allocations.clear()
    op = sw.prepare_spatial_withdrawal(dev, memory_budget_bytes=BIG)
    assert len(cupy.modules) == 1 and cupy.modules[0] == (tc.render_source(), tuple(tc.COMPILE_OPTIONS), "nvrtc")
    assert cupy.allocations == []  # no dummy array, no copy
    fns = tc._FUNCTIONS[0]
    assert all(getattr(fns, n).launches == [] for n in ("withdraw", "finalize", "availability"))
    sw.prepare_spatial_withdrawal(dev, memory_budget_bytes=BIG)
    assert len(cupy.modules) == 1  # cached per device
    assert op.prepare_memory.device_required_bytes == 0 and op.prepare_memory.host_required_bytes == sp.PY_ALLOWANCE_BYTES


@pytest.mark.parametrize("operation", ["prepare", "withdraw", "availability"])
def test_admission_boundaries_on_both_sides_precede_any_allocation_launch_or_compile(cupy, operation):
    rng = np.random.default_rng(5)
    L, N = 3, 130
    p = a2.params(rng, L, N)
    np_owner = make(p, np.ones(N, bool))
    dev = device_owner(np_owner)
    base_op = sw.prepare_spatial_withdrawal(dev, memory_budget_bytes=BIG)
    w = wkind(rng, p, "random")
    state = sp.spatial_column_state(dev, fx(w), 0.0, memory_budget_bytes=BIG)
    q = fx(np.zeros((L, N)))
    est = sw.estimate_spatial_withdrawal_memory(L, N, backend="cuda", operation=operation)
    dn, hn = est.device_required_bytes, est.host_required_bytes

    def call(**kw):
        full = dict(memory_budget_bytes=BIG, reserve_bytes=0, available_bytes_fn=None, host_memory_budget_bytes=None, host_reserve_bytes=0, host_available_bytes_fn=None)
        full.update(kw)
        if operation == "prepare":
            return sw.prepare_spatial_withdrawal(dev, **full)
        if operation == "withdraw":
            return sw.spatial_withdraw(base_op, state, q, **full)
        return sw.spatial_layer_availability_m(base_op, state, **full)

    with tc._LOCK:
        tc._FUNCTIONS.clear()
    cupy.modules.clear()
    cupy.allocations.clear()
    cases = [dict(host_memory_budget_bytes=hn - 1), dict(host_memory_budget_bytes=hn, host_reserve_bytes=1), dict(host_available_bytes_fn=lambda: hn - 1)]
    if dn > 0:  # the preparation needs no device bytes: its device-side boundary is trivially zero (a negative bound is not a valid quota)
        cases += [dict(memory_budget_bytes=dn - 1), dict(memory_budget_bytes=dn, reserve_bytes=1), dict(available_bytes_fn=lambda: dn - 1)]
    else:
        cases += [dict(memory_budget_bytes=0, reserve_bytes=1)]
    for kw in cases:
        with pytest.raises(sp.SpatialLayerMemoryError, match="nothing was allocated"):
            call(**kw)
    assert cupy.allocations == [] and cupy.modules == []  # nothing allocated, compiled or launched by any refusal
    assert call(memory_budget_bytes=dn, host_memory_budget_bytes=hn, available_bytes_fn=lambda: dn, host_available_bytes_fn=lambda: hn) is not None  # exactly the estimate is admitted
    for bad in (lambda: -1, lambda: True, lambda: 2.5):
        with pytest.raises(sp.SpatialLayerError, match="reader must return"):
            call(available_bytes_fn=bad)
        with pytest.raises(sp.SpatialLayerError, match="reader must return"):
            call(host_available_bytes_fn=bad)


def test_cuda_estimate_formulas_are_exact_and_both_sides_are_separate():
    for L, N in ((1, 1), (3, 129), (5, 1000)):
        nb = (N + 127) // 128
        w = sw.estimate_spatial_withdrawal_memory(L, N, backend="cuda", operation="withdraw", reserve_bytes=1000, host_reserve_bytes=77)
        assert (w.output_bytes, w.scratch_bytes, w.row_bytes, w.packet_bytes) == (4 * L * N * 8, 8 * 8 * nb + 64, 0, 64)
        assert w.device_required_bytes == w.required_bytes == 4 * L * N * 8 + 8 * 8 * nb + 64 + 1000 and w.host_required_bytes == 64 + sp.PY_ALLOWANCE_BYTES + 77 and w.terms["no_rows_array"]
        a = sw.estimate_spatial_withdrawal_memory(L, N, backend="cuda", operation="availability")
        assert (a.device_required_bytes, a.host_required_bytes) == (L * N * 8 + 8, 8 + sp.PY_ALLOWANCE_BYTES)
        pr = sw.estimate_spatial_withdrawal_memory(L, N, backend="cuda", operation="prepare")
        assert (pr.device_required_bytes, pr.host_required_bytes) == (0, sp.PY_ALLOWANCE_BYTES)
    cpu = sw.estimate_spatial_withdrawal_memory(3, 129, backend="numpy")
    assert cpu.device_required_bytes == 0 and cpu.host_required_bytes == cpu.required_bytes  # CPU records are unchanged


def test_device_namespace_and_type_checks_precede_any_work(cupy):
    rng = np.random.default_rng(6)
    L, N = 3, 64
    p = a2.params(rng, L, N)
    np_owner = make(p, np.ones(N, bool))
    dev = device_owner(np_owner)
    op = sw.prepare_spatial_withdrawal(dev, memory_budget_bytes=BIG)
    w = wkind(rng, p, "random")
    state = sp.spatial_column_state(dev, fx(w), 0.0, memory_budget_bytes=BIG)
    cpu_op = sw.prepare_spatial_withdrawal(np_owner, memory_budget_bytes=BIG)
    cpu_state = sp.spatial_column_state(np_owner, w, 0.0, memory_budget_bytes=BIG)
    z = np.zeros((L, N))
    cupy.allocations.clear()
    cases = [("numpy request", lambda: sw.spatial_withdraw(op, state, z, memory_budget_bytes=0), "cupy.ndarray"), ("float32", lambda: sw.spatial_withdraw(op, state, fx(z.astype(np.float32)), memory_budget_bytes=0), "float64"),
             ("shape", lambda: sw.spatial_withdraw(op, state, fx(z[:-1]), memory_budget_bytes=0), "shape"), ("other device", lambda: sw.spatial_withdraw(op, state, fx(z, device=1), memory_budget_bytes=0), "device 1"),
             ("cpu state, cuda op", lambda: sw.spatial_withdraw(op, cpu_state, fx(z), memory_budget_bytes=0), "namespace"),
             ("cuda state, cpu op", lambda: sw.spatial_withdraw(cpu_op, state, z, memory_budget_bytes=BIG), "namespace"),
             ("numpy request, availability state", lambda: sw.spatial_layer_availability_m(op, cpu_state, memory_budget_bytes=0), "namespace")]
    for label, fn, text in cases:
        with pytest.raises(sp.SpatialLayerError, match=text):
            fn()
    cupy.rc.current = 1
    for fn in (lambda: sw.spatial_withdraw(op, state, fx(z), memory_budget_bytes=BIG), lambda: sw.spatial_layer_availability_m(op, state, memory_budget_bytes=BIG), lambda: sw.prepare_spatial_withdrawal(dev, memory_budget_bytes=BIG)):
        with pytest.raises(sp.SpatialLayerError, match="current device"):
            fn()
    cupy.rc.current = 0
    assert cupy.allocations == []


def test_scratch_lives_until_the_synchronising_read_and_is_released_afterwards(cupy, monkeypatch):
    rng = np.random.default_rng(7)
    L, N = 3, 130
    p = a2.params(rng, L, N)
    np_owner = make(p, np.ones(N, bool))
    dev = device_owner(np_owner)
    op = sw.prepare_spatial_withdrawal(dev, memory_budget_bytes=BIG)
    w = wkind(rng, p, "random")
    state = sp.spatial_column_state(dev, fx(w), 0.0, memory_budget_bytes=BIG)
    q = fx(np.zeros((L, N)))
    refs = []
    real_empty = cupy.empty
    cupy.empty = lambda shape, dtype=np.float64: (lambda a: (refs.append((tuple(int(x) for x in np.atleast_1d(shape)), weakref.ref(a))), a)[1])(real_empty(shape, dtype))
    alive = []
    real_h = backend.to_host
    monkeypatch.setattr(backend, "to_host", lambda a: (alive.append([r() is not None for _, r in refs]), real_h(a))[1])
    r = sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG)
    assert all(alive[0])  # partials and the packet were still referenced when the packet was read back
    gc.collect()
    nb = (N + 127) // 128
    dead = {s for s, ref in refs if ref() is None}
    assert {(8 * nb,), (8,)} <= dead and (L, N) not in dead  # scratch released after the read; the four outputs survive
    assert r.actual_m is not None


def test_typed_oom_publishes_nothing_and_no_state_or_request_is_written(cupy):
    rng = np.random.default_rng(8)
    L, N = 3, 130
    p = a2.params(rng, L, N)
    np_owner = make(p, np.ones(N, bool))
    dev = device_owner(np_owner)
    op = sw.prepare_spatial_withdrawal(dev, memory_budget_bytes=BIG)
    w = wkind(rng, p, "random")
    state = sp.spatial_column_state(dev, fx(w), 2.0, memory_budget_bytes=BIG)
    q = fx(rng.uniform(0, 0.01, (L, N)))
    before = (hashlib.sha256(np.asarray(state.water_m).tobytes()).hexdigest(), hashlib.sha256(np.asarray(q).tobytes()).hexdigest())
    real_empty, n = cupy.empty, {"n": 0}

    def failing(shape, dtype=np.float64):
        n["n"] += 1
        if n["n"] == 3:
            raise MemoryError("out of memory (test double)")
        return real_empty(shape, dtype)

    cupy.empty = failing
    with pytest.raises(sp.SpatialLayerMemoryError, match="nothing was published"):
        sw.spatial_withdraw(op, state, q, memory_budget_bytes=BIG)
    assert (hashlib.sha256(np.asarray(state.water_m).tobytes()).hexdigest(), hashlib.sha256(np.asarray(q).tobytes()).hexdigest()) == before and state.t_s == 2.0


def test_owner_operator_and_results_are_released_and_the_kernel_cache_holds_functions_only(cupy):
    rng = np.random.default_rng(9)
    L, N = 3, 64
    p = a2.params(rng, L, N)
    np_owner = make(p, np.ones(N, bool))
    dev = device_owner(np_owner)
    op = sw.prepare_spatial_withdrawal(dev, memory_budget_bytes=BIG)
    state = sp.spatial_column_state(dev, fx(wkind(rng, p, "random")), 0.0, memory_budget_bytes=BIG)
    r = sw.spatial_withdraw(op, state, fx(np.zeros((L, N))), memory_budget_bytes=BIG)
    refs = [weakref.ref(x) for x in (dev, op, state, r, r.state, r.state.water_m, r.actual_m, dev._maps["dz"])]
    del dev, op, state, r
    gc.collect()
    assert all(x() is None for x in refs)
    assert all(type(getattr(f, n)).__name__ == "FakeFn" for f in tc._FUNCTIONS.values() for n in ("withdraw", "finalize", "availability"))
