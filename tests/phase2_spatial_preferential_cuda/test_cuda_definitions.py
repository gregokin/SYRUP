"""Actual-GPU test DEFINITIONS for the spatial CUDA preferential infiltration (WRITTEN, NOT RUN by their author: no GPU was launched or probed). Skipped without a usable CuPy device; Root runs them under its device
guard (GPU0 UUID, fresh 0 MiB and 0 %, `.venv-cupy`, genuine 9f CUDA materialisation). They compare the device operator with the accepted CPU spatial path (maps and arrays within the declared rtol 2e-12 /
atol 1e-14; bitwise where no `expm1`/`log1p` is evaluated), with the accepted COMMON CUDA preferential operator column by column on the SAME backend (bit for bit: same body, same device math), flags/typed
errors on every cell, sizes around the block boundary, ONE 72-byte read and zero H2D per hot step, the cold reducer against NumPy on the device, the bounded cold identity, and release. Self-contained data."""
from __future__ import annotations

import gc

import numpy as np
import pytest
from conftest import ATOL, AREA, BIG, LAYOUT_SETS, RTOL, a3, bits, numpy_owner, pi, sp, spf, ss, tc, weight_case


@pytest.fixture(scope="module")
def cp():
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the spatial CUDA preferential is not exercised (no claim)")
    return backend.cupy_module()


def dev(cp, a):
    return cp.asarray(np.ascontiguousarray(a))


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def gpu_owner(cp, inputs, N, mask, area=AREA):
    d = {k: (v if np.isscalar(v) else dev(cp, v)) for k, v in inputs.items()}
    spec = sp.spatial_layer_spec(d["edges"], theta_residual=d["res"], theta_field_capacity=d["fc"], theta_saturation=d["sat"], theta_evaporation_stress=0.5, diffusivity_m2_s=d["iface"], n_cells=N)
    return sp.prepare_spatial_column(spec, dev(cp, mask), area, backend="cuda", memory_budget_bytes=BIG, host_memory_budget_bytes=BIG)


def make_case(cp, rng, L, N, layout, kind):
    inputs, p = a3.layout_case(rng, L, N, **LAYOUT_SETS[layout])
    mask = rng.random(N) < 0.8
    mask[0] = True
    w = weight_case(rng, L, kind)
    contributions = [rng.uniform(0.0, 2.0, N) for _ in w["species"]] if w["species"] else None
    a3_in = rng.uniform(0.0, 1.0, N)
    a3_in[0::5], a3_in[1::7] = 0.0, 1.0
    mr = np.where(rng.random(N) < 0.3, np.inf, rng.uniform(1e-8, 5e-6, N))
    kw = dict(max_fraction_per_hour=0.85, background_receiver_weights=w["background"], species_receiver_weights=w["species"])
    gcol = gpu_owner(cp, inputs, N, mask)
    ccol = numpy_owner(inputs, N, mask)
    gop = spf.prepare_spatial_preferential(gcol, a3=dev(cp, a3_in), max_intake_rate_m_s=dev(cp, mr), **kw, species_contributions=None if contributions is None else iter([dev(cp, c) for c in contributions]),
                                           memory_budget_bytes=BIG, host_memory_budget_bytes=BIG)
    cop = spf.prepare_spatial_preferential(ccol, a3=a3_in, max_intake_rate_m_s=mr, **kw, species_contributions=None if contributions is None else iter([c.copy() for c in contributions]), memory_budget_bytes=BIG)
    dz = np.diff(p["edges"], axis=0)
    water = np.ascontiguousarray(p["sat"] * dz * rng.uniform(0.0, 0.95, (L, N)))
    return p, mask, gcol, ccol, gop, cop, water, a3_in, mr, contributions, kw


@pytest.mark.parametrize("kind", ["no_species", "multi_overlap", "disabled_species_with_cover", "all_zero"])
@pytest.mark.parametrize("layout", ["all_cell", "mixed", "all_shared", "scalar_thetas"])
@pytest.mark.parametrize("L", [1, 2, 3, 5])
@pytest.mark.parametrize("N", [1, 7, 127, 128, 129, 257, 1001])
def test_device_maps_and_steps_match_the_cpu_spatial_path(cp, kind, layout, L, N):
    rng = np.random.default_rng(100 * L + N + len(kind) + len(layout))
    p, mask, gcol, ccol, gop, cop, water, *_ = make_case(cp, rng, L, N, layout, kind)
    for a, b in zip(gop._law, cop._law, strict=True):
        np.testing.assert_allclose(host(a), b, rtol=RTOL, atol=ATOL)  # device log1p/elementwise vs NumPy: declared bound
    surface = np.where(mask, rng.uniform(0.0, 0.02, N), 0.01)
    gst, cst = sp._new_state(gcol, dev(cp, water), 777.25), sp.spatial_column_state(ccol, water, 777.25, memory_budget_bytes=BIG)
    for dt in (3600.0, 0.0, 600.0, 7200.0):
        g = spf.spatial_preferential_step(gop, dev(cp, surface), gst, dt, memory_budget_bytes=BIG, host_memory_budget_bytes=BIG)
        c = spf.spatial_preferential_step(cop, surface, cst, dt, memory_budget_bytes=BIG)
        assert g.state.t_s == 777.25 and g.state.namespace == "cupy"
        for x, y in ((g.state.water_m, c.state.water_m), (g.surface_water_m, c.surface_water_m), (g.intake_m, c.intake_m), (g.intake_total_m, c.intake_total_m)):
            np.testing.assert_allclose(host(x), y, rtol=RTOL, atol=ATOL)
        assert g.budget.n_terms == c.budget.n_terms and abs(g.budget.soil_residual_m3) <= g.budget.bound_m3 and abs(g.budget.surface_residual_m3) <= g.budget.bound_m3
        np.testing.assert_array_equal(host(g.intake_m)[:, ~mask], 0.0)
        gst, cst, surface = g.state, c.state, host(g.surface_water_m).copy()


def test_each_column_equals_the_accepted_common_cuda_preferential_operator_bit_for_bit_on_the_same_backend(cp):
    rng = np.random.default_rng(3)
    L, N = 3, 24
    p, mask, gcol, ccol, gop, cop, water, a3_in, mr, contributions, kw = make_case(cp, rng, L, N, "all_cell", "multi_overlap")
    surface = np.where(mask, rng.uniform(0.0, 0.02, N), 0.01)
    got = spf.spatial_preferential_step(gop, dev(cp, surface), sp._new_state(gcol, dev(cp, water), 1.0), 1800.0, memory_budget_bytes=BIG, host_memory_budget_bytes=BIG)
    for c in range(N):
        lay = ss.soil_layers(p["edges"][:, c], theta_residual=p["res"][:, c], theta_field_capacity=p["fc"][:, c], theta_saturation=p["sat"][:, c], theta_evaporation_stress=0.5, diffusivity_m2_s=p["iface"][:, c])
        col = ss.prepare_column(lay, cp.asarray(np.array([bool(mask[c])])), AREA, backend="cuda")  # a REAL cupy bool mask (the common API refuses a NumPy mask)
        op = pi.prepare_preferential(col, a3=float(a3_in[c]), max_fraction_per_hour=0.85, max_intake_rate_m_s=float(mr[c]), background_receiver_weights=kw["background_receiver_weights"],
                                     species_receiver_weights=kw["species_receiver_weights"], species_contributions=iter([dev(cp, x[c:c + 1]) for x in contributions]))
        ref = pi.preferential_step(op, dev(cp, surface[c:c + 1]), ss.column_state(col, dev(cp, water[:, c:c + 1]), 1.0), 1800.0)
        for a, b in ((got.state.water_m[:, c:c + 1], ref.state.water_m), (got.surface_water_m[c:c + 1], ref.surface_water_m), (got.intake_m[:, c:c + 1], ref.intake_m), (got.intake_total_m[c:c + 1], ref.intake_total_m)):
            np.testing.assert_array_equal(bits(host(a)), bits(host(b)), err_msg=f"cell {c}")


def test_the_cold_reducer_equals_numpy_on_the_device(cp):
    red = tc.ColdReducer(cp, int(cp.cuda.Device().id))
    rng = np.random.default_rng(5)
    for n in (1, 127, 128, 129, 1001, 100003):
        for a in (rng.normal(size=n), np.where(rng.random(n) < 0.2, np.inf, 1.0), np.where(rng.random(n) < 0.3, np.nan, rng.uniform(-1, 1, n)), np.where(rng.random(n) < 0.5, -0.0, 0.0)):
            d = dev(cp, a)
            assert red("count_nonfinite", d) == float(np.count_nonzero(~np.isfinite(a))) and red("count_nan", d) == float(np.count_nonzero(np.isnan(a)))
            for kind, fn in (("min", np.min), ("max", np.max)):
                got, want = red(kind, d), float(fn(a))
                assert (np.isnan(got) and np.isnan(want)) or got == want
    assert red.h2d_bytes == 32 * red.calls and red.d2h_bytes == 32 * red.calls


def test_flags_are_reported_for_every_cell_inactive_included_and_nothing_is_written(cp):
    rng = np.random.default_rng(12)
    L, N = 2, 300
    p, mask, gcol, ccol, gop, cop, water, *_ = make_case(cp, rng, L, N, "all_cell", "no_species")
    water = water * 0.5
    water[0, [1, 4, 150, 299]] = -1.0
    water[1, 150] = -2.0
    z = np.zeros(N)
    with pytest.raises(pi.PreferentialStepError) as info:
        spf.spatial_preferential_step(gop, dev(cp, z), sp._new_state(gcol, dev(cp, water), 0.0), 60.0, memory_budget_bytes=BIG, host_memory_budget_bytes=BIG)
    assert info.value.bit == 1 and info.value.n_flagged == 4  # each invalid cell once, inactive included
    surface = np.full(N, 0.01)
    surface[3] = -1.0
    d_surface = dev(cp, surface)
    with pytest.raises(pi.PreferentialStepError) as info:
        spf.spatial_preferential_step(gop, d_surface, sp._new_state(gcol, dev(cp, water * 0 + 0.01), 0.0), 60.0, memory_budget_bytes=BIG, host_memory_budget_bytes=BIG)
    assert info.value.bit == 4
    np.testing.assert_array_equal(host(d_surface), surface)


def test_one_72_byte_read_and_zero_h2d_per_step_and_release(cp):
    backend = pytest.importorskip("maple.core.backend")
    rng = np.random.default_rng(6)
    L, N = 3, 1001
    p, mask, gcol, ccol, gop, cop, water, *_ = make_case(cp, rng, L, N, "mixed", "multi_overlap")
    args = (dev(cp, np.where(mask, 0.03, 0.01)), sp._new_state(gcol, dev(cp, water), 0.0), 3600.0)
    spf.spatial_preferential_step(gop, *args, memory_budget_bytes=BIG, host_memory_budget_bytes=BIG)  # warm
    before = dict(backend.transfer_counters()) if hasattr(backend, "transfer_counters") else None
    r = spf.spatial_preferential_step(gop, *args, memory_budget_bytes=BIG, host_memory_budget_bytes=BIG)
    if before is not None:
        after = backend.transfer_counters()
        assert after.get("device_to_host", 0) - before.get("device_to_host", 0) == 1
        assert after.get("device_to_host_bytes", 0) - before.get("device_to_host_bytes", 0) == 72
        assert after.get("host_to_device", 0) - before.get("host_to_device", 0) == 0
    pool_before = cp.get_default_memory_pool().used_bytes()
    del r
    gc.collect()
    assert cp.get_default_memory_pool().used_bytes() <= pool_before


def test_cold_identity_and_verification_on_the_device(cp):
    rng = np.random.default_rng(2)
    p, mask, gcol, ccol, gop, cop, *_ = make_case(cp, rng, 3, 3000, "all_cell", "multi_overlap")
    assert spf.verify_operator_identity(gop, host_memory_budget_bytes=BIG) is True
    need = min(3 * 3000, tc.STAGE_ELEMS) * 8 + 65536
    with pytest.raises(sp.SpatialLayerMemoryError):
        spf.verify_operator_identity(gop, host_memory_budget_bytes=need - 1)


def test_wrong_device_and_numpy_arrays_are_refused_before_any_launch(cp):
    rng = np.random.default_rng(1)
    p, mask, gcol, ccol, gop, cop, water, *_ = make_case(cp, rng, 2, 20, "all_cell", "no_species")
    state = sp._new_state(gcol, dev(cp, np.zeros((2, 20))), 0.0)
    with pytest.raises(sp.SpatialLayerError):
        spf.spatial_preferential_step(gop, np.zeros(20), state, 60.0, memory_budget_bytes=BIG, host_memory_budget_bytes=BIG)
    if cp.cuda.runtime.getDeviceCount() > 1:
        with cp.cuda.Device(1):
            with pytest.raises(sp.SpatialLayerError):
                spf.spatial_preferential_step(gop, dev(cp, np.zeros(20)), state, 60.0, memory_budget_bytes=BIG, host_memory_budget_bytes=BIG)
