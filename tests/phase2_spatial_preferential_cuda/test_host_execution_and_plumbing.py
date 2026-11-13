"""The rendered spatial CUDA PREFERENTIAL source EXECUTED on the host (real threads + barrier, hostb2d.py) through the REAL Python launchers, the REAL shared derivation and the public API on a numpy-backed
CuPy plumbing double whose library reductions raise, compared with the accepted CPU spatial path (maps and arrays; declared rtol 2e-12 / atol 1e-14 for host libm vs NumPy, bitwise where no `expm1` is evaluated),
plus the plumbing contracts (allocations equal the model, ONE 72-byte read, zero hot H2D, counted cold reducer traffic, admission before any allocation/weight record/species consumption/compile on both sides,
type/device/legacy refusals, typed errors, bounded cold identity hash, release). NOT hardware evidence. Author-written."""
from __future__ import annotations

import gc
import hashlib
import tracemalloc
import weakref

import numpy as np
import pytest
from conftest import (ATOL, AREA, BIG, LAYOUT_SETS, RTOL, a3, bits, build_pair, dev_state, device_owner, fx, needs_host, numpy_owner, pi, sm_hash, sp, spf, ss, tc, weight_case)

import maple.core.backend as backend

# The numpy-backed double evaluates log1p(-1) = -inf (beta = 1) with NumPy's warning machinery; real CuPy emits no such warning (`errstate` is a null context there). Harness artefact only.
pytestmark = [needs_host, pytest.mark.filterwarnings("ignore:divide by zero encountered in log1p:RuntimeWarning"), pytest.mark.filterwarnings("ignore:overflow encountered in add:RuntimeWarning")]
KINDS = ["no_species", "no_background", "zero_background_one_species", "multi_overlap", "disabled_species_with_cover", "all_zero"]


def case_data(rng, L, N, layout, kind, *, scalar_law=False, cap="mixed"):
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS[layout])
    mask = ((rng.random(N) < 0.8)).copy()
    mask[0] = True
    if N > 1:
        mask[1] = False
    w = weight_case(rng, L, kind)
    contributions = [rng.uniform(0.0, 2.0, N) for _ in w["species"]] if w["species"] else None
    a3_in = rng.uniform(0.0, 1.0, N)
    a3_in[0::5], a3_in[1::7] = 0.0, 1.0  # exact beta endpoints on some cells
    mr = {"mixed": np.where(rng.random(N) < 0.3, np.inf, rng.uniform(1e-8, 5e-6, N)), "inf": np.full(N, np.inf), "zero": np.zeros(N), "finite": rng.uniform(1e-8, 5e-6, N)}[cap]
    kwargs = dict(a3=0.6 if scalar_law else a3_in, max_fraction_per_hour=0.85, max_intake_rate_m_s={"inf": np.inf, "zero": 0.0}.get(cap, 2e-6) if scalar_law else mr, background_receiver_weights=w["background"],
                  species_receiver_weights=w["species"])
    dz = np.diff(p["edges"], axis=0)
    sat = p["sat"] * dz
    water = np.ascontiguousarray(sat * rng.uniform(0.0, 0.95, (L, N)))
    near = rng.random(N) < 0.25
    water[:, near] = (sat * (1.0 - rng.uniform(0.0, 5e-3, (L, N))))[:, near]  # room-limited receivers
    return inputs, p, mask, kwargs, contributions, water


def _grid():
    """Full L x N x layout grid for the two principal weight sets; every other weight case on a reduced but boundary-covering grid (host execution runs real threads, so the grid is bounded)."""
    out = []
    for kind in ("multi_overlap", "no_species"):
        out += [(kind, layout, L, N) for layout in ("all_cell", "mixed", "all_shared", "scalar_thetas") for L in (1, 2, 3, 5) for N in (1, 7, 127, 128, 129, 257)]
    for kind in KINDS:
        if kind not in ("multi_overlap", "no_species"):
            out += [(kind, layout, L, N) for layout in ("all_cell", "mixed") for L in (1, 3) for N in (7, 129)]
    return out


@pytest.mark.parametrize("kind,layout,L,N", _grid())
def test_device_maps_and_steps_match_the_cpu_spatial_path_over_repeated_stages(cupy, kind, layout, L, N):
    rng = np.random.default_rng(100 * L + N + len(kind) + len(layout))
    inputs, p, mask, kwargs, contributions, water = case_data(rng, L, N, layout, kind)
    np_col = numpy_owner(inputs, N, mask)
    cop, col, gop = build_pair(cupy, np_col, kwargs, contributions)
    for a, b in zip(gop._law, cop._law, strict=True):
        np.testing.assert_array_equal(bits(np.asarray(a)), bits(b))  # the derived maps equal the CPU maps bit for bit (the same shared derivation; elementwise IEEE)
    assert gop.n_species == cop.n_species and gop.n_species_enabled == cop.n_species_enabled and dict(gop.layouts) == dict(cop.layouts)
    surface = np.where(mask, rng.uniform(0.0, 0.02, N), rng.uniform(0.0, 0.01, N))
    gst, cst = dev_state(col, water, 777.25), sp.spatial_column_state(np_col, water, 777.25, memory_budget_bytes=BIG)
    for dt in (3600.0, 0.0, 600.0, 60.0, 7200.0):
        g = spf.spatial_preferential_step(gop, fx(surface), gst, dt, memory_budget_bytes=BIG)
        c = spf.spatial_preferential_step(cop, surface, cst, dt, memory_budget_bytes=BIG)
        assert g.state.t_s == 777.25 and g.state.namespace == "cupy" and g.dt_s == dt
        for x, y in ((g.state.water_m, c.state.water_m), (g.surface_water_m, c.surface_water_m), (g.intake_m, c.intake_m), (g.intake_total_m, c.intake_total_m)):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=RTOL, atol=ATOL)
        for f in ("soil_before_m3", "soil_after_m3", "surface_before_m3", "surface_after_m3", "isolated_attempt_m3", "attempted_m3", "intake_m3", "rate_capped_m3", "capacity_rejected_m3"):
            assert getattr(g.budget, f) == pytest.approx(getattr(c.budget, f), rel=1e-11, abs=1e-15), f
        assert g.budget.n_terms == c.budget.n_terms and abs(g.budget.soil_residual_m3) <= g.budget.bound_m3 and abs(g.budget.surface_residual_m3) <= g.budget.bound_m3
        gi = np.asarray(g.intake_m)
        np.testing.assert_array_equal(gi[:, ~mask], 0.0)  # nothing allocated to inactive cells
        np.testing.assert_array_equal(bits(np.asarray(g.state.water_m)[:, ~mask]), bits(np.asarray(gst.water_m)[:, ~mask]))  # inactive water preserved
        np.testing.assert_array_equal(bits(np.asarray(g.surface_water_m)[~mask]), bits(surface[~mask]))  # inactive surface preserved
        gst, cst, surface = g.state, c.state, np.asarray(g.surface_water_m).copy()


@pytest.mark.parametrize("cap", ["inf", "zero", "finite"])
@pytest.mark.parametrize("scalar_law", [False, True])
def test_beta_endpoints_caps_and_scalar_or_cell_laws(cupy, cap, scalar_law):
    rng = np.random.default_rng(31 + len(cap) + scalar_law)
    L, N = 3, 40
    inputs, p, mask, kwargs, contributions, water = case_data(rng, L, N, "mixed", "multi_overlap", scalar_law=scalar_law, cap=cap)
    np_col = numpy_owner(inputs, N, mask)
    cop, col, gop = build_pair(cupy, np_col, kwargs, contributions)
    assert dict(gop.layouts) == {"a3": "scalar" if scalar_law else "cell", "max_intake_rate_m_s": "scalar" if scalar_law else "cell"}
    assert gop.retained_bytes == (N + L * N) * 8 + (8 if scalar_law else N * 8)  # rate N + receiver L N + max rate (one value or N): no bound tensor, no host mirror
    surface = np.where(mask, 0.05, 0.02)
    g = spf.spatial_preferential_step(gop, fx(surface), dev_state(col, water), 3600.0, memory_budget_bytes=BIG)
    c = spf.spatial_preferential_step(cop, surface, sp.spatial_column_state(np_col, water, 0.0, memory_budget_bytes=BIG), 3600.0, memory_budget_bytes=BIG)
    np.testing.assert_allclose(np.asarray(g.intake_m), c.intake_m, rtol=RTOL, atol=ATOL)
    if cap == "zero":
        assert g.budget.intake_m3 == 0.0  # a zero cap admits nothing; the isolated attempt is reported, never lost
    if cap == "inf":
        assert g.budget.rate_capped_m3 == 0.0


def test_beta_zero_beta_one_and_all_zero_weights_and_dt_zero(cupy):
    rng = np.random.default_rng(3)
    L, N = 2, 9
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS["all_cell"])
    mask = np.ones(N, bool)
    np_col = numpy_owner(inputs, N, mask)
    water = np.ascontiguousarray(p["sat"] * np.diff(p["edges"], axis=0) * 0.3)
    surface = np.full(N, 1e-4)  # far below every top-layer room: beta 1 admits ALL of H
    for a3v, maxf, bg, expect in ((0.0, 1.0, [1.0, 0.0], "zero"), (1.0, 1.0, [1.0, 0.0], "all"), (0.5, 1.0, [0.0, 0.0], "zero")):
        kw = dict(a3=a3v, max_fraction_per_hour=maxf, max_intake_rate_m_s=np.inf, background_receiver_weights=bg, species_receiver_weights=[])
        cop, col, gop = build_pair(cupy, np_col, kw, None)
        g = spf.spatial_preferential_step(gop, fx(surface), dev_state(col, water), 1e-3, memory_budget_bytes=BIG)
        if expect == "zero":
            assert g.budget.intake_m3 == 0.0 and np.asarray(g.surface_water_m).tobytes() == surface.tobytes()  # beta 0 / all weights zero: nothing moves
        else:
            assert (np.asarray(g.intake_total_m) == 1e-4).all() and (np.asarray(g.surface_water_m) == 0.0).all()  # beta 1: k = +inf, ALL of H for any dt > 0
        ident = spf.spatial_preferential_step(gop, fx(surface), dev_state(col, water, 3.0), 0.0, memory_budget_bytes=BIG)  # dt = 0: the identity
        np.testing.assert_array_equal(bits(np.asarray(ident.state.water_m)), bits(water))
        assert ident.state.t_s == 3.0 and ident.budget.attempted_m3 == 0.0


def test_a_saturating_receiver_keeps_its_share_on_the_surface_without_remapping(cupy):
    inputs, p = a3.layout_case(np.random.default_rng(1), 2, 3, **LAYOUT_SETS["all_shared"])
    np_col = numpy_owner(inputs, 3, np.ones(3, bool))
    dz = np.diff(p["edges"], axis=0)
    sat = p["sat"] * dz
    water = np.ascontiguousarray(sat.copy())
    water[1] = sat[1] * 0.2  # the TOP layer is exactly full: zero room; the second receiver has room
    kw = dict(a3=1.0, max_fraction_per_hour=1.0, max_intake_rate_m_s=np.inf, background_receiver_weights=[1.0, 1.0], species_receiver_weights=[])
    cop, col, gop = build_pair(cupy, np_col, kw, None)
    H = 0.2
    g = spf.spatial_preferential_step(gop, fx(np.full(3, H)), dev_state(col, water), 1e-3, memory_budget_bytes=BIG)
    gi = np.asarray(g.intake_m)
    np.testing.assert_array_equal(gi[0], 0.0)  # the full receiver takes nothing
    np.testing.assert_allclose(gi[1], H / 2, rtol=1e-14)  # the other receiver gets its own half ONLY: the rejected half is NOT remapped to it
    np.testing.assert_allclose(np.asarray(g.surface_water_m), H - H / 2, rtol=1e-14)  # the rejected share stays on the surface
    assert g.budget.capacity_rejected_m3 > 0.0


def test_invalid_input_flags_are_reported_on_every_cell_inactive_included_and_nothing_is_written(cupy):
    rng = np.random.default_rng(12)
    L, N = 2, 20
    inputs, p, mask, kwargs, contributions, water0 = case_data(rng, L, N, "all_cell", "no_species")
    mask = np.array([True, True, False, True] * 5)
    np_col = numpy_owner(inputs, N, mask)
    cop, col, gop = build_pair(cupy, np_col, kwargs, contributions)
    base_surface = np.full(N, 0.01)
    cases = {0: ("water", (0, 0), np.nan), 1: ("water", (1, 2), -1e-3), 2: ("water", (0, 3), 50.0), 3: ("surface", 0, np.nan), 4: ("surface", 2, -1.0)}
    for bit, (which, cell, value) in cases.items():
        water, surface = water0.copy(), base_surface.copy()
        if which == "water":
            water[cell] = value
        else:
            surface[cell] = value
        snap = (water.copy(), surface.copy())
        with pytest.raises(pi.PreferentialStepError) as info:
            spf.spatial_preferential_step(gop, fx(surface), dev_state(col, water), 60.0, memory_budget_bytes=BIG)
        with pytest.raises(pi.PreferentialStepError) as ref:
            spf.spatial_preferential_step(cop, surface, sp._new_state(np_col, water, 0.0), 60.0, memory_budget_bytes=BIG)
        assert (info.value.bit, info.value.n_flagged) == (ref.value.bit, ref.value.n_flagged) == (bit, info.value.n_flagged) and pi.FLAG_MESSAGES[bit].split(" ")[0] in str(info.value)
        np.testing.assert_array_equal(water, snap[0])
        np.testing.assert_array_equal(surface, snap[1])
    water = water0.copy()
    water[0, [1, 4, 150 % N]] = -1.0
    water[1, 150 % N] = -2.0
    with pytest.raises(pi.PreferentialStepError) as info:
        spf.spatial_preferential_step(gop, fx(base_surface), dev_state(col, water), 60.0, memory_budget_bytes=BIG)
    assert info.value.bit == 1 and info.value.n_flagged == len({1, 4, 150 % N})  # each invalid cell is counted ONCE, an inactive cell included


def test_unrepresentable_global_volume_is_a_typed_refusal(cupy):
    spec = sp.spatial_layer_spec(np.array([0.0, 0.5, 1.5]), theta_residual=0.01, theta_field_capacity=0.25, theta_saturation=0.9, theta_evaporation_stress=0.05, diffusivity_m2_s=0.0, n_cells=2)
    np_col = sp.prepare_spatial_column(spec, np.ones(2, bool), 1e308, backend="numpy", memory_budget_bytes=BIG)
    col = device_owner(np_col)
    gop = spf.prepare_spatial_preferential(col, a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=np.inf, background_receiver_weights=[1.0, 0.0], memory_budget_bytes=BIG)
    with pytest.raises(pi.PreferentialStepError, match="non-finite global water budget"):
        spf.spatial_preferential_step(gop, fx(np.full(2, 0.01)), dev_state(col, np.array([[0.45, 0.45], [0.9, 0.9]])), 60.0, memory_budget_bytes=BIG)


# ---- the cold reducer ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 7, 127, 128, 129, 257, 1000])
def test_the_cold_reducer_equals_numpy_on_edge_arrays_with_exact_traffic(cupy, n):
    red = tc.ColdReducer(cupy, 0)
    rng = np.random.default_rng(n)
    arrays = {"plain": rng.normal(size=n), "positive": rng.uniform(0.0, 3.0, n), "all_equal": np.full(n, 2.5), "neg_zero": np.where(rng.random(n) < 0.5, -0.0, 0.0), "has_inf": np.where(rng.random(n) < 0.2, np.inf, 1.0),
              "has_ninf": np.where(rng.random(n) < 0.2, -np.inf, -1.0), "has_nan": np.where(rng.random(n) < 0.3, np.nan, rng.uniform(-1, 1, n)), "all_nan": np.full(n, np.nan), "tiny": rng.uniform(1e-320, 1e-300, n)}
    for name, a in arrays.items():
        d = fx(a)
        assert red("count_nonfinite", d) == float(np.count_nonzero(~np.isfinite(a))) and red("count_nan", d) == float(np.count_nonzero(np.isnan(a))), name
        for kind, fn in (("min", np.min), ("max", np.max)):
            got, want = red(kind, d), float(fn(a))
            assert (np.isnan(got) and np.isnan(want)) or (got == want and np.signbit(got) == np.signbit(want) or (got == 0.0 and want == 0.0)), (name, kind, got, want)  # NaN propagates; +-inf exact
    assert red.calls == 4 * len(arrays) and red.h2d_bytes == 32 * red.calls and red.d2h_bytes == 32 * red.calls  # one 32-byte COLD seed and one 32-byte COLD read per call
    assert cupy.h2d == [32] * red.calls  # exactly the seeds


def test_the_cold_reducer_refuses_unknown_kinds_and_bad_arrays_before_any_launch(cupy):
    red = tc.ColdReducer(cupy, 0)
    good = fx(np.ones(4))
    with pytest.raises(pi.PreferentialError, match="unknown cold reduction kind"):
        red("sum", good)
    n_launch = len(cupy.launches)
    for bad in (np.ones(4), fx(np.ones(4, dtype=np.float32)), fx(np.ones((2, 2))), fx(np.ones(8))[::2], fx(np.ones(4), 1), [1.0]):
        with pytest.raises(sp.SpatialLayerError):
            red("min", bad)
    assert len(cupy.launches) == n_launch and red.calls == 0 and cupy.h2d == []


def test_the_derivation_uses_only_the_counted_reducer_and_never_a_library_reduction(cupy):
    """The double's `sum/min/max/isfinite/isnan` raise: reaching prepare() proves the cold derivation never touched them. Count: 2 per species (non-finite + min, disabled species included) + cover + closed form +
    beta min/max + one per receiver layer + the rate NaN count."""
    rng = np.random.default_rng(5)
    for L, S in ((1, 0), (3, 2), (5, 3)):
        N = 33
        inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS["all_cell"])
        np_col = numpy_owner(inputs, N, np.ones(N, bool))
        w = [weight_vec for weight_vec in []]
        kw = dict(a3=fx(rng.uniform(0, 1, N)), max_fraction_per_hour=0.9, max_intake_rate_m_s=1e-6, background_receiver_weights=[1.0] + [0.0] * (L - 1),
                  species_receiver_weights=[[0.0] * (L - 1) + [1.0]] * S)
        col = device_owner(np_col)
        reducers = []
        real = tc.ColdReducer

        class Spy(real):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                reducers.append(self)

        tc.ColdReducer = Spy
        try:
            op = spf.prepare_spatial_preferential(col, **kw, species_contributions=iter([fx(rng.uniform(0, 1, N)) for _ in range(S)]) if S else None, memory_budget_bytes=BIG)
        finally:
            tc.ColdReducer = real
        assert reducers[0].calls == 2 * S + L + 5 and op.n_species == S


# ---- plumbing: allocations, reads, transfers, launches ---------------------------------------------------------------------------------
def make_op(cupy, L=3, N=300, species=True, seed=6):
    rng = np.random.default_rng(seed)
    inputs, p, mask, kwargs, contributions, water = case_data(rng, L, N, "mixed", "multi_overlap" if species else "no_species")
    np_col = numpy_owner(inputs, N, mask)
    cop, col, gop = build_pair(cupy, np_col, kwargs, contributions)
    return rng, p, mask, water, np_col, cop, col, gop


def test_step_allocations_reads_and_launches_equal_the_model(cupy, monkeypatch):
    rng, p, mask, water, np_col, cop, col, gop = make_op(cupy)
    L, N = 3, 300
    surface, state = fx(np.where(mask, 0.03, 0.01)), dev_state(col, water)
    reads = []
    real = backend.to_host
    monkeypatch.setattr(backend, "to_host", lambda a: reads.append((a.dtype.name, int(a.nbytes))) or real(a))
    cupy.allocations.clear(), cupy.h2d.clear(), cupy.launches.clear()
    r = spf.spatial_preferential_step(gop, surface, state, 3600.0, memory_budget_bytes=BIG)
    nb = tc.n_blocks(N)
    assert [(k, shape, dt) for k, shape, dt, _ in cupy.allocations] == [("empty", (N,), "float64")] * 2 + [("empty", (L, N), "float64")] * 2 + [("empty", (7 * nb,), "float64"), ("zeros", (9,), "uint64")]
    est = r.memory
    assert cupy.allocated_bytes == est.output_bytes + est.scratch_bytes == (2 * N + 2 * L * N) * 8 + 7 * 8 * nb + 72
    assert est.required_bytes == est.device_required_bytes == est.output_bytes + est.scratch_bytes and est.host_required_bytes == 144 + sp.PY_ALLOWANCE_BYTES
    assert cupy.h2d == []  # ZERO host-to-device copies in the hot step
    assert reads == [("uint64", 72)]  # ONE constant-size read: the raw packet
    assert cupy.launches == [("maple_syrup_spatial_pref_step", (nb,), (128,)), ("maple_syrup_spatial_pref_finalize", (1,), (256,))]
    assert est.terms["no_dense_rows"] and est.terms["host_packet_buffers"] == 2 and est.terms["static_shared_bytes_per_block"] == 7 * 8 * 128 + 8 * 128 + 4 * 128


def test_prepare_cold_traffic_and_device_allocations_stay_within_the_model(cupy):
    rng = np.random.default_rng(9)
    L, N = 3, 500
    inputs, p, mask, kwargs, contributions, water = case_data(rng, L, N, "all_cell", "multi_overlap")
    S = len(kwargs["species_receiver_weights"])
    np_col = numpy_owner(inputs, N, mask)
    col = device_owner(np_col)
    cupy.allocations.clear(), cupy.h2d.clear()
    gkw = {k: (fx(v) if k in ("a3", "max_intake_rate_m_s") else v) for k, v in kwargs.items()}
    op = spf.prepare_spatial_preferential(col, **gkw, species_contributions=iter([fx(c) for c in contributions]), memory_budget_bytes=BIG)
    est = op.prepare_memory
    allocated = cupy.allocated_bytes  # every device array of the call, including the ones released before return
    assert allocated <= est.device_required_bytes, (allocated, est.device_required_bytes)  # the model is an upper bound of the allocation requests
    assert est.device_required_bytes == (L * N + 6 * N) * 8 + 3 * N + 64 and op.retained_bytes == (N + L * N + N) * 8 <= est.output_bytes  # rate N + receiver L N + per-cell max rate N
    # cold H2D: one 32-byte seed per reducer call; a per-cell a3/max rate is passed as a device array (no copy of the caller array is a host transfer)
    assert set(cupy.h2d) == {32} and len(cupy.h2d) == 2 * S + L + 5
    assert est.host_required_bytes == 64 + min(L * N, tc.STAGE_ELEMS) * 8 + 2 * (S + 1) * L * 8 + 2 * L * 8 + sp.PY_ALLOWANCE_BYTES


@pytest.mark.parametrize("L", [4096, 8192])
def test_large_l_single_cell_host_weight_peaks_are_within_the_modelled_host_requirement(cupy, L):
    """N = 1 and a large number of layers: the host build holds the retained weight records AND the record under construction (raw + normalised copies and the sealing copies); measured with tracemalloc on
    the real `ReceiverWeights` construction against the modelled `2 (S + 1) L + 2 L` doubles."""
    S = 2
    vecs = [np.linspace(0.1, 2.0, L) for _ in range(S + 1)]
    gc.collect()
    tracemalloc.start()
    try:
        recs = [pi.receiver_weights(v, L, "w") for v in vecs]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    est = spf.estimate_spatial_preferential_memory(L, 1, backend="cuda", operation="prepare", n_species=S)
    assert peak <= est.weight_bytes + 2 * L * 8 + sp.PY_ALLOWANCE_BYTES, (peak, est.weight_bytes)
    assert est.host_required_bytes == 64 + min(L, tc.STAGE_ELEMS) * 8 + est.weight_bytes + 2 * L * 8 + sp.PY_ALLOWANCE_BYTES
    assert len(recs) == S + 1


def test_admission_precedes_every_allocation_weight_record_species_consumption_and_compile_on_both_sides(cupy):
    rng = np.random.default_rng(1)
    L, N = 2, 400
    inputs, p, mask, kwargs, contributions, water = case_data(rng, L, N, "all_cell", "no_species")
    np_col = numpy_owner(inputs, N, mask)
    col = device_owner(np_col)
    touched = []

    def stream():
        touched.append(1)
        yield fx(np.ones(N))

    kw = dict(a3=fx(np.full(N, 0.5)), max_fraction_per_hour=1.0, max_intake_rate_m_s=fx(np.full(N, 1e-6)), background_receiver_weights=[1.0, 0.0], species_receiver_weights=[[0.0, 1.0]])
    est = spf.estimate_spatial_preferential_memory(L, N, backend="cuda", operation="prepare", n_species=1)
    for budget, reader, hb, hrd in ((est.device_required_bytes - 1, None, None, None), (BIG, lambda: est.device_required_bytes - 1, None, None), (BIG, None, est.host_required_bytes - 1, None),
                                    (BIG, None, None, lambda: est.host_required_bytes - 1)):
        before = len(cupy.allocations)
        with pytest.raises(sp.SpatialLayerMemoryError):
            spf.prepare_spatial_preferential(col, **kw, species_contributions=stream(), memory_budget_bytes=budget, available_bytes_fn=reader, host_memory_budget_bytes=hb, host_available_bytes_fn=hrd)
        assert len(cupy.allocations) == before and cupy.modules == [] and cupy.launches == [] and cupy.h2d == [] and touched == []
    for kws in (dict(memory_budget_bytes=est.device_required_bytes, reserve_bytes=1), dict(memory_budget_bytes=BIG, host_memory_budget_bytes=est.host_required_bytes, host_reserve_bytes=1)):
        with pytest.raises(sp.SpatialLayerMemoryError):
            spf.prepare_spatial_preferential(col, **kw, species_contributions=stream(), **kws)
    assert touched == [] and cupy.modules == []
    op = spf.prepare_spatial_preferential(col, **kw, species_contributions=iter([fx(np.ones(N))]), memory_budget_bytes=est.device_required_bytes, host_memory_budget_bytes=est.host_required_bytes)
    sest = spf.estimate_spatial_preferential_memory(L, N, backend="cuda", operation="step")
    state, H = dev_state(col, np.zeros((L, N))), fx(np.full(N, 0.01))
    n_alloc, n_launch = len(cupy.allocations), len(cupy.launches)
    for kws in (dict(memory_budget_bytes=sest.device_required_bytes - 1), dict(memory_budget_bytes=BIG, host_memory_budget_bytes=sest.host_required_bytes - 1),
                dict(memory_budget_bytes=BIG, available_bytes_fn=lambda: sest.device_required_bytes - 1), dict(memory_budget_bytes=BIG, host_available_bytes_fn=lambda: sest.host_required_bytes - 1),
                dict(memory_budget_bytes=sest.device_required_bytes, reserve_bytes=1)):
        with pytest.raises(sp.SpatialLayerMemoryError):
            spf.spatial_preferential_step(op, H, state, 60.0, **kws)
        assert len(cupy.allocations) == n_alloc and len(cupy.launches) == n_launch
    spf.spatial_preferential_step(op, H, state, 60.0, memory_budget_bytes=sest.device_required_bytes, host_memory_budget_bytes=sest.host_required_bytes)
    for bad in (True, -1, 2.5):
        with pytest.raises(sp.SpatialLayerError):
            spf.spatial_preferential_step(op, H, state, 60.0, memory_budget_bytes=BIG, host_available_bytes_fn=lambda b=bad: b)


def test_cpu_owners_refuse_the_cuda_only_arguments_and_the_cpu_model_is_unchanged():
    rng = np.random.default_rng(1)
    inputs, p = a3.layout_case(rng, 2, 5, **LAYOUT_SETS["all_cell"])
    np_col = numpy_owner(inputs, 5, np.ones(5, bool))
    kw = dict(a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=1.0, background_receiver_weights=[1.0, 0.0], memory_budget_bytes=BIG)
    for extra in (dict(host_memory_budget_bytes=1), dict(host_reserve_bytes=1), dict(host_available_bytes_fn=lambda: 1)):
        with pytest.raises(sp.SpatialLayerError):
            spf.prepare_spatial_preferential(np_col, **kw, **extra)
    op = spf.prepare_spatial_preferential(np_col, **kw)
    with pytest.raises(sp.SpatialLayerError):
        spf.spatial_preferential_step(op, np.zeros(5), sp.spatial_column_state(np_col, np.zeros((2, 5)), 0.0, memory_budget_bytes=BIG), 1.0, memory_budget_bytes=BIG, host_memory_budget_bytes=1)
    with pytest.raises(sp.SpatialLayerError):
        spf.verify_operator_identity(op, host_memory_budget_bytes=1)
    with pytest.raises(sp.SpatialLayerError):
        spf.estimate_spatial_preferential_memory(2, 5, backend="numpy", host_reserve_bytes=1)
    e = spf.estimate_spatial_preferential_memory(3, 1000, backend="numba", operation="step", reserve_bytes=7)
    assert e.host_required_bytes == e.required_bytes and e.device_required_bytes == 0
    assert spf.estimate_spatial_preferential_memory(3, 1000, backend="numpy", operation="prepare", n_species=2).required_bytes == (2 * 1000 + 3000) * 8 + (4 * 1000 + 3000) * 8 + pi.BUILD_SCRATCH_MASK_BYTES_PER_CELL * 1000 + 2 * 3 * 3 * 8 + sp.PY_ALLOWANCE_BYTES


def test_type_namespace_device_legacy_refusals_precede_every_allocation(cupy):
    rng, p, mask, water, np_col, cop, col, gop = make_op(cupy, L=2, N=20)
    N = 20
    state, H = dev_state(col, np.zeros((2, N))), fx(np.zeros(N))
    n_alloc = len(cupy.allocations)

    def fails(*a, **k):
        with pytest.raises(sp.SpatialLayerError):
            spf.spatial_preferential_step(*a, memory_budget_bytes=BIG, **k)

    fails(gop, np.zeros(N), state, 60.0)  # a NumPy array for a CUDA owner: no implicit transfer
    fails(gop, fx(np.zeros(N, dtype=np.float32)), state, 60.0)
    fails(gop, fx(np.zeros(N + 1)), state, 60.0)
    fails(gop, fx(np.zeros(2 * N))[::2], state, 60.0)
    fails(gop, H, sp.spatial_column_state(np_col, np.zeros((2, N)), 0.0, memory_budget_bytes=BIG), 60.0)  # a CPU state for a CUDA owner
    fails(cop, np.zeros(N), state, 60.0)  # a CUDA state for a CPU operator
    lay = ss.soil_layers([0.0, 0.5, 1.5], theta_residual=0.01, theta_field_capacity=0.25, theta_saturation=0.9, diffusivity_m2_s=0.0, theta_evaporation_stress=0.05)
    lcol = ss.prepare_column(lay, np.ones(3, bool), 1.0)
    lop = pi.prepare_preferential(lcol, a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=np.inf, background_receiver_weights=[1.0, 0.0])
    fails(lop, H, state, 60.0)
    with pytest.raises(ss.SubsurfaceError):
        pi.preferential_step(gop, H, ss.column_state(lcol, np.zeros((2, 3)), 0.0), 60.0)  # the legacy entry point still refuses the spatial CUDA operator
    with pytest.raises(sp.SpatialLayerError):
        spf.prepare_spatial_preferential(lcol, a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=1.0, background_receiver_weights=[1.0, 0.0], memory_budget_bytes=BIG)
    for dt in (-1.0, float("nan"), True, "60"):
        with pytest.raises(ss.SubsurfaceError):
            spf.spatial_preferential_step(gop, H, state, dt, memory_budget_bytes=BIG)
    assert len(cupy.allocations) == n_alloc
    cupy.rc.current = 1
    for call in (lambda: spf.spatial_preferential_step(gop, H, state, 60.0, memory_budget_bytes=BIG),
                 lambda: spf.prepare_spatial_preferential(col, a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=1.0, background_receiver_weights=[1.0, 0.0], memory_budget_bytes=BIG)):
        with pytest.raises(sp.SpatialLayerError, match="device"):
            call()
    cupy.rc.current = 0
    with pytest.raises(sp.SpatialLayerError):
        spf.spatial_preferential_step(gop, fx(np.zeros(N), 1), state, 60.0, memory_budget_bytes=BIG)  # a wrong-device array
    assert len(cupy.allocations) == n_alloc


def test_prepare_validation_covers_every_cell_inactive_included_and_streams_contributions(cupy):
    rng = np.random.default_rng(4)
    L, N = 2, 40
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS["all_cell"])
    mask = np.array([True, False] * 20)
    np_col = numpy_owner(inputs, N, mask)
    col = device_owner(np_col)
    ok = dict(a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=1e-6, background_receiver_weights=[1.0, 0.0], memory_budget_bytes=BIG)
    for name, bad in (("a3", -0.1), ("a3", 1.0000001), ("a3", np.nan), ("a3", np.inf), ("max_intake_rate_m_s", -1e-9), ("max_intake_rate_m_s", np.nan), ("max_intake_rate_m_s", -np.inf)):
        for pos in (0, 1):  # an active and an INACTIVE cell
            arr = np.full(N, ok[name])
            arr[pos] = bad
            snap = arr.copy()
            with pytest.raises(sp.SpatialLayerError):
                spf.prepare_spatial_preferential(col, **{**ok, name: fx(arr)})
            np.testing.assert_array_equal(arr, snap)
        with pytest.raises(sp.SpatialLayerError):
            spf.prepare_spatial_preferential(col, **{**ok, name: bad})
    spf.prepare_spatial_preferential(col, **{**ok, "max_intake_rate_m_s": fx(np.where(mask, np.inf, 1.0))})  # +inf is a valid max rate
    one = [[0.0, 1.0]]
    for bad in (np.where(mask, 1.0, np.nan), np.where(mask, 1.0, -1e-9), np.where(mask, 1.0, np.inf)):
        with pytest.raises(ss.SubsurfaceError):
            spf.prepare_spatial_preferential(col, **ok, species_receiver_weights=one, species_contributions=iter([fx(bad)]))  # an INACTIVE cell is validated too
    with pytest.raises(ss.SubsurfaceError):
        spf.prepare_spatial_preferential(col, **ok, species_receiver_weights=[[0.0, 0.0]], species_contributions=iter([fx(np.where(mask, 1.0, np.nan))]))  # a DISABLED species is validated too
    with pytest.raises(ss.SubsurfaceError, match="overflow"):
        spf.prepare_spatial_preferential(col, **ok, species_receiver_weights=[[0, 1], [0, 1]], species_contributions=iter([fx(np.full(N, 1.7e308)), fx(np.full(N, 1.7e308))]))
    for bad in (np.ones(N), fx(np.ones(N, dtype=np.float32)), fx(np.ones(N + 1))):
        with pytest.raises(sp.SpatialLayerError):
            spf.prepare_spatial_preferential(col, **ok, species_receiver_weights=one, species_contributions=iter([bad]))
    spf.prepare_spatial_preferential(col, **ok, species_receiver_weights=one, species_contributions=iter([fx(np.ones(N))]))
    # nothing of the caller's per-cell arrays is aliased by the operator: a later change of the caller's array cannot alter it
    mr = np.full(N, 2e-6)
    dmr = fx(mr)
    op = spf.prepare_spatial_preferential(col, **{**ok, "max_intake_rate_m_s": dmr})
    dmr[:] = 9.0
    assert float(np.asarray(op._law[1])[0]) == 2e-6 and op._law[1] is not dmr


# ---- identity, caches, errors, release --------------------------------------------------------------------------------------------------
def test_the_device_identity_recipe_equals_the_cpu_stream_for_1d_and_2d_maps_and_is_bounded(cupy):
    rng = np.random.default_rng(7)
    for shape in ((1000,), (3, 500), (1, 1), (5, 1), (8192, 1)):
        arr = np.ascontiguousarray(rng.uniform(1e-7, 4e-6, shape))
        cpu = hashlib.sha256()
        sm_hash(cpu, "receiver", "map", arr)
        for size in (1, 7, 64, 1000, 4096, 10**6):
            h = hashlib.sha256()
            tc.hash_device_array(h, "receiver", "map", fx(arr), np.empty(size))
            assert h.hexdigest() == cpu.hexdigest(), (shape, size)  # ORIGINAL shape in the header + flat C bytes, any staging size
    # the staged reads of a real prepare are bounded by the counted staging buffer
    inputs, p = a3.layout_case(rng, 3, 3000, **LAYOUT_SETS["all_cell"])
    np_col = numpy_owner(inputs, 3000, np.ones(3000, bool))
    col = device_owner(np_col)
    sizes = []
    base_get = type(fx(arr)).get

    def counting_get(self, out=None):
        sizes.append(0 if out is None else out.size)
        return base_get(self, out)

    type(fx(arr)).get = counting_get
    try:
        op = spf.prepare_spatial_preferential(col, a3=fx(rng.uniform(0, 1, 3000)), max_fraction_per_hour=0.9, max_intake_rate_m_s=fx(rng.uniform(1e-7, 1e-5, 3000)), background_receiver_weights=[1, 1, 1], memory_budget_bytes=BIG)
        assert sizes and max(sizes) <= tc.STAGE_ELEMS
        sizes.clear()
        spf.spatial_preferential_step(op, fx(np.zeros(3000)), dev_state(col, np.zeros((3, 3000))), 60.0, memory_budget_bytes=BIG)
        assert sizes == []  # a step never hashes and never mirrors an array to the host
    finally:
        type(fx(arr)).get = base_get


def test_cold_verification_streams_through_the_admitted_host_buffer_and_detects_tampering(cupy):
    rng, p, mask, water, np_col, cop, col, gop = make_op(cupy, L=2, N=500)
    need = min(2 * 500, tc.STAGE_ELEMS) * 8 + sp.PY_ALLOWANCE_BYTES
    assert spf.verify_operator_identity(gop, host_memory_budget_bytes=need) is True and spf.verify_operator_identity(gop) is True
    for kws in (dict(host_memory_budget_bytes=need - 1), dict(host_available_bytes_fn=lambda: need - 1), dict(host_memory_budget_bytes=need, host_reserve_bytes=1)):
        with pytest.raises(sp.SpatialLayerMemoryError):
            spf.verify_operator_identity(gop, **kws)
    object.__setattr__(gop, "_law", (fx(np.asarray(gop._law[0]) * 1.0000001), gop._law[1], gop._law[2]))
    assert spf.verify_operator_identity(gop) is False


def test_identity_sensitivity_and_the_weight_stream_is_the_unchanged_canonical_one(cupy):
    rng = np.random.default_rng(2)
    inputs, p = a3.layout_case(rng, 3, 20, **LAYOUT_SETS["all_cell"])
    np_col = numpy_owner(inputs, 20, np.ones(20, bool))
    col = device_owner(np_col)
    base = dict(a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=1e-6, background_receiver_weights=[1.0, 2.0, 0.0])
    a = spf.prepare_spatial_preferential(col, **base, memory_budget_bytes=BIG)
    b = spf.prepare_spatial_preferential(col, **base, memory_budget_bytes=BIG)
    assert a.sha256 == b.sha256 and spf.verify_operator_identity(a)
    for change in (dict(a3=0.6), dict(max_intake_rate_m_s=2e-6), dict(max_fraction_per_hour=0.9), dict(background_receiver_weights=[1.0, 2.0, 1.0])):
        assert spf.prepare_spatial_preferential(col, **{**base, **change}, memory_budget_bytes=BIG).sha256 != a.sha256
    # the weight part of the digest is exactly what the CPU operator hashes (the bounded 4096-byte hex stream), for any chunk size
    class Rec:
        def __init__(self, raw):
            self.raw = raw

    raw = np.linspace(0.0, 3.0, 9000)
    for chunk in (1, 8, 4096):
        spf.WEIGHT_HASH_CHUNK_BYTES = chunk
        h = hashlib.sha256()
        spf._hash_weights(h, "background", Rec(raw))
        assert h.hexdigest() == hashlib.sha256(b"background|" + raw.tobytes().hex().encode() + b"\0").hexdigest()
    spf.WEIGHT_HASH_CHUNK_BYTES = 4096


def test_the_kernel_cache_holds_functions_only_per_device_and_compiles_once(cupy):
    rng = np.random.default_rng(2)
    inputs, p = a3.layout_case(rng, 2, 10, **LAYOUT_SETS["all_cell"])
    np_col = numpy_owner(inputs, 10, np.ones(10, bool))
    c0, c1 = device_owner(np_col, 0), device_owner(np_col, 1)
    kw = dict(a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=1e-6, background_receiver_weights=[1.0, 0.0], memory_budget_bytes=BIG)
    cupy.rc.current = 0
    spf.prepare_spatial_preferential(c0, **kw)
    spf.prepare_spatial_preferential(c0, **{**kw, "a3": 0.3})
    assert len(cupy.modules) == 1 and set(tc._FUNCTIONS) == {0}
    cupy.rc.current = 1
    spf.prepare_spatial_preferential(c1, **kw)
    assert len(cupy.modules) == 2 and set(tc._FUNCTIONS) == {0, 1}
    for fns in tc._FUNCTIONS.values():
        assert sorted(vars(fns)) == ["finalize", "reduce", "step", "validate"]
    code, options, backend_name = cupy.modules[0]
    assert options == tc.COMPILE_OPTIONS and "--fmad=false" in options and backend_name == "nvrtc" and code == tc.render_source()


def test_failed_compile_and_device_errors_are_typed_and_publish_nothing(cupy):
    rng, p, mask, water, np_col, cop, col, gop = make_op(cupy, L=2, N=20)
    N = 20

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("synthetic NVRTC failure")

    with tc._LOCK:
        tc._FUNCTIONS.clear()
    real_module = cupy.RawModule
    cupy.RawModule = Boom
    with pytest.raises(sp.SpatialLayerBackendUnavailableError, match="no fallback"):
        spf.prepare_spatial_preferential(col, a3=0.5, max_fraction_per_hour=1.0, max_intake_rate_m_s=1.0, background_receiver_weights=[1.0, 0.0], memory_budget_bytes=BIG)
    cupy.RawModule = real_module
    state, H = dev_state(col, np.zeros((2, N))), fx(np.zeros(N))
    real_empty = cupy.empty
    cupy.empty = lambda shape, dtype=np.float64: (_ for _ in ()).throw(MemoryError("synthetic device out of memory"))
    with pytest.raises(sp.SpatialLayerMemoryError):
        spf.spatial_preferential_step(gop, H, state, 60.0, memory_budget_bytes=BIG)
    cupy.empty = real_empty

    class CUDARuntimeError(Exception):
        pass

    spf.spatial_preferential_step(gop, H, state, 60.0, memory_budget_bytes=BIG)  # loads nothing new: the cache holds the functions
    fns = tc._FUNCTIONS[0]
    real_step = fns.step
    fns.step = lambda *a, **k: (_ for _ in ()).throw(CUDARuntimeError("synthetic launch failure"))
    with pytest.raises(sp.SpatialLayerBackendUnavailableError):
        spf.spatial_preferential_step(gop, H, state, 60.0, memory_budget_bytes=BIG)
    fns.step = real_step
    np.testing.assert_array_equal(np.asarray(state.water_m), 0.0)  # the accepted state is untouched
    assert spf.spatial_preferential_step(gop, H, state, 60.0, memory_budget_bytes=BIG).budget.intake_m3 == 0.0


def test_owner_release_and_no_module_level_retention(cupy):
    rng, p, mask, water, np_col, cop, col, gop = make_op(cupy, L=2, N=20)
    z = fx(np.zeros(20))
    r = spf.spatial_preferential_step(gop, z, dev_state(col, np.zeros((2, 20))), 60.0, memory_budget_bytes=BIG)
    refs = [weakref.ref(x) for x in (gop, r.state, r.surface_water_m, r.intake_m, gop._law[0], gop._law[2])]
    del gop, r
    gc.collect()
    assert all(w() is None for w in refs)


def test_estimates_are_pure_arithmetic():
    for L, N, S in ((1, 1, 0), (3, 1000, 2), (5, 257, 1)):
        e = spf.estimate_spatial_preferential_memory(L, N, backend="cuda", operation="step", reserve_bytes=7, host_reserve_bytes=11)
        nb = (N + 127) // 128
        assert e.device_required_bytes == e.required_bytes == (2 * N + 2 * L * N) * 8 + 7 * 8 * nb + 72 + 7 and e.host_required_bytes == 144 + sp.PY_ALLOWANCE_BYTES + 11 and e.backend == "cuda"
        p = spf.estimate_spatial_preferential_memory(L, N, backend="cuda", operation="prepare", n_species=S)
        assert p.device_required_bytes == (L * N + 6 * N) * 8 + 3 * N + 64
        assert p.host_required_bytes == 64 + min(L * N, tc.STAGE_ELEMS) * 8 + 2 * (S + 1) * L * 8 + 2 * L * 8 + sp.PY_ALLOWANCE_BYTES
    with pytest.raises(Exception):
        spf.estimate_spatial_preferential_memory(2, 2**60, backend="cuda", operation="step")
