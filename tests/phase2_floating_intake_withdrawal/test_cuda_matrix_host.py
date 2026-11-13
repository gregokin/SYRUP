"""Floating CUDA matrix intake through the HOST-compiled rendered source and the CuPy plumbing double: NOT hardware evidence. NumPy/CUDA equivalence (partial/zero/sliver depths, inactive placeholders, block edges,
1/2/3/65 layers), Root's controls, explicit device history rebinding, resident traffic/allocation accounting, the device gate and admission order. Author-written; hardware definitions: test_gpu_definitions.py."""
from __future__ import annotations

import numpy as np
import pytest
from builders import REFUSALS, pair
from conftest import needs_host_cuda
from hostd2 import fake

from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_matrix_infiltration_cuda as fmc
from maple_syrup import floating_soil_layers as fl
from maple_syrup import matrix_infiltration_cuda as mc

pytestmark = needs_host_cuda


def edges_for(L):
    return np.concatenate([[0.0], np.cumsum(np.full(L, 0.6 / L))])


def case(cp, L, N, seed):
    rng = np.random.default_rng(seed)
    depth = rng.choice([0.0, 1e-6, 0.03, 0.1, 0.2, 0.45, 0.6, 0.9], size=N)
    act = rng.random(N) > 0.2
    cpu, gpu = pair(cp, edges_for(L), depth, tr=0.05, tf=0.3, ts=0.45, active=act)
    h = np.stack([fl.porous_thickness(cpu, k) for k in range(L)])
    w = h * 0.45 * rng.uniform(0.05, 0.95, h.shape)
    w[:, ~act] = rng.uniform(0.0, 2.0, (L, int((~act).sum())))
    pool = rng.uniform(0.0, 0.02, N) * act
    standing = np.where(rng.random(N) > 0.5, rng.uniform(0.0, 0.01, N), 0.0) * act
    rate = np.where(standing > 0.0, rng.uniform(0.0, 1e-5, N), pool / 600.0 * rng.uniform(1.0, 3.0, N)) * act
    return cpu, gpu, w, pool, standing, rate, act, rng


def ops(cpu, gpu, k, psi, g):
    a = fm.prepare_floating_matrix(cpu, conductivity_m_s=k, suction_m=psi, vegetation_factor=g)
    b = fm.prepare_floating_matrix(gpu, conductivity_m_s=fake(k) if isinstance(k, np.ndarray) else k, suction_m=fake(psi) if isinstance(psi, np.ndarray) else psi,
                                   vegetation_factor=fake(g) if isinstance(g, np.ndarray) else g)
    return a, b


@pytest.mark.parametrize("L,N", [(1, 5), (2, 127), (3, 128), (3, 129), (65, 260), (3, 1000)])
def test_step_matches_numpy(cp, L, N):
    cpu, gpu, w, pool, standing, rate, act, rng = case(cp, L, N, L * 777 + N)
    k, psi, g = rng.uniform(1e-7, 1e-5, N), rng.uniform(0.0, 0.3, N), rng.uniform(0.2, 1.0, N)
    a, b = ops(cpu, gpu, k, psi, g)
    j0, c0 = rng.uniform(0.0, 0.02, N), rng.uniform(0.0, 0.5, N)
    ha = fm.floating_matrix_history(a, wetting_m=j0, cumulative_intake_m=c0)
    hb = fm.floating_matrix_history(b, wetting_m=fake(j0), cumulative_intake_m=fake(c0))
    ra = fm.floating_matrix_step(a, pool, standing, rate, fl.floating_continued_state(cpu, w), ha, 600.0)
    rb = fm.floating_matrix_step(b, fake(pool), fake(standing), fake(rate), fl.floating_continued_state(gpu, fake(w)), hb, 600.0)
    pairs = ((ra.state.water_m, rb.state.water_m), (ra.surface_pool_m, rb.surface_pool_m), (ra.history.wetting_m, rb.history.wetting_m), (ra.history.cumulative_intake_m, rb.history.cumulative_intake_m),
             (ra.intake_m, rb.intake_m))
    for x, y in pairs:
        assert np.allclose(np.asarray(x), np.asarray(y), rtol=2e-12, atol=1e-14)
    assert np.array_equal(np.asarray(rb.state.water_m)[:, ~act], w[:, ~act])  # placeholder bits
    assert np.array_equal(np.asarray(rb.history.wetting_m)[~act], j0[~act]) and np.array_equal(np.asarray(rb.history.cumulative_intake_m)[~act], c0[~act])
    assert rb.budget.intake_m3 == pytest.approx(ra.budget.intake_m3, rel=2e-12, abs=1e-14)


def test_root_controls_on_the_device(cp):
    _cpu, gpu = pair(cp, [0.0, 0.2, 0.5], [0.1])
    op = fm.prepare_floating_matrix(gpu, conductivity_m_s=2e-6, suction_m=0.0, vegetation_factor=0.5)
    h = fm.floating_matrix_history(op, wetting_m=0.01, cumulative_intake_m=0.02)
    one = lambda x: fake(np.array([x]))
    r = fm.floating_matrix_step(op, one(0.01), one(0.01), one(0.0), fl.floating_continued_state(gpu, fake(np.array([[0.01], [0.0]]))), h, 1000.0)
    assert np.allclose(np.asarray(r.state.water_m)[:, 0], [0.011, 0.0], atol=1e-15)
    assert float(np.asarray(r.surface_pool_m)[0]) == pytest.approx(0.009, abs=1e-15)
    assert float(np.asarray(r.history.wetting_m)[0]) == pytest.approx(0.012, abs=1e-15) and float(np.asarray(r.history.cumulative_intake_m)[0]) == pytest.approx(0.021, abs=1e-15)
    _cpu0, gpu0 = pair(cp, [0.0, 0.2, 0.5], [0.0])
    op0 = fm.prepare_floating_matrix(gpu0, conductivity_m_s=2e-6, suction_m=0.3, vegetation_factor=0.5)
    h0 = fm.floating_matrix_history(op0, wetting_m=0.01, cumulative_intake_m=0.02)
    r0 = fm.floating_matrix_step(op0, one(0.01), one(0.01), one(0.0), fl.floating_continued_state(gpu0, fake(np.zeros((2, 1)))), h0, 1000.0)
    assert np.all(np.asarray(r0.state.water_m) == 0.0) and float(np.asarray(r0.intake_m)[0]) == 0.0 and float(np.asarray(r0.surface_pool_m)[0]) == 0.01
    assert float(np.asarray(r0.history.wetting_m)[0]) == 0.01 and float(np.asarray(r0.history.cumulative_intake_m)[0]) == 0.02


def test_rebinding_on_the_device_retains_exact_history(cp):
    N = 200
    rng = np.random.default_rng(4)
    depth = rng.choice([0.0, 0.05, 0.15, 0.6], size=N)
    edges = [0.0, 0.2, 0.5]
    c1 = pair(cp, edges, depth)[1]
    c2 = pair(cp, edges, np.minimum(depth + 0.02, 0.6), ts=0.42)[1]  # changed depth/properties, same mask/edges/area
    o1 = fm.prepare_floating_matrix(c1, conductivity_m_s=2e-6, suction_m=0.1, vegetation_factor=0.5)
    o2 = fm.prepare_floating_matrix(c2, conductivity_m_s=fake(rng.uniform(1e-6, 1e-5, N)), suction_m=0.2, vegetation_factor=0.7)
    j, cum = rng.uniform(0.0, 0.3, N), rng.uniform(0.0, 2.0, N)
    h1 = fm.floating_matrix_history(o1, wetting_m=fake(j), cumulative_intake_m=fake(cum))
    cp.count.__init__()
    cp.allocs.clear()
    rb = fm.rebind_floating_matrix_history(o1, h1, o2)
    assert rb.reset is False and rb.wetting_sum_m is None and rb.history.operator_sha256 == o2.sha256
    assert np.array_equal(np.asarray(rb.history.wetting_m), j) and np.array_equal(np.asarray(rb.history.cumulative_intake_m), cum)
    assert rb.history.wetting_m is not h1.wetting_m
    assert cp.count.d2h == 2 and cp.count.d2h_sizes == [8, 8] and cp.count.h2d == 0  # only the two 8-byte validation diagnostics
    assert sorted(cp.allocs) == sorted([8, 8, N * 8, N * 8])
    # refusals: other mask, other area, bad old history
    bad_mask = pair(cp, edges, depth, active=np.arange(N) % 2 == 0)[1]
    bad_area = pair(cp, edges, depth, area=2.0)[1]
    for bad in (bad_mask, bad_area):
        ob = fm.prepare_floating_matrix(bad, conductivity_m_s=2e-6, suction_m=0.1, vegetation_factor=0.5)
        with pytest.raises(REFUSALS):
            fm.rebind_floating_matrix_history(o1, h1, ob)
    with pytest.raises(REFUSALS, match="old the history is bound"):
        fm.rebind_floating_matrix_history(o2, h1, o1)
    nanj = j.copy()
    nanj[3] = np.nan
    with pytest.raises(REFUSALS, match="finite and >= 0"):
        fm.floating_matrix_history(o1, wetting_m=fake(nanj), cumulative_intake_m=fake(cum))


def test_traffic_and_allocation_of_one_step(cp):
    L, N = 3, 300
    _cpu, gpu, w, pool, standing, rate, _act, _rng = case(cp, L, N, 5)
    op = fm.prepare_floating_matrix(gpu, conductivity_m_s=2e-6, suction_m=0.1, vegetation_factor=0.5)
    h = fm.floating_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0)
    s = fl.floating_continued_state(gpu, fake(w))
    cp.count.__init__()
    cp.allocs.clear()
    fm.floating_matrix_step(op, fake(pool), fake(standing), fake(rate), s, h, 600.0)
    nb = fmc.n_blocks(N)
    assert cp.count.d2h == 1 and cp.count.d2h_sizes == [fmc.RAW_PACKET_BYTES] and cp.count.h2d == 0
    assert sorted(cp.allocs) == sorted([N * 8] * 4 + [L * N * 8, fmc.ROWS * 8 * nb, fmc.RAW_PACKET_BYTES])
    t = fmc.operation_terms("step", L, N)
    assert t["outputs"] == (4 * N + L * N) * 8 and t["host"] == 2 * fmc.RAW_PACKET_BYTES  # raw + rebuilt packet are both counted


def test_gate_and_admission_order(cp):
    L, N = 3, 40
    _cpu, gpu, w, pool, standing, rate, _act, _rng = case(cp, L, N, 6)
    op = fm.prepare_floating_matrix(gpu, conductivity_m_s=2e-6, suction_m=0.1, vegetation_factor=0.5)
    h = fm.floating_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0)
    s = fl.floating_continued_state(gpu, fake(w))
    cp.allocs.clear()
    with pytest.raises(REFUSALS, match="nothing"):
        fm.floating_matrix_step(op, fake(pool), fake(standing), fake(rate), s, h, 600.0, memory_budget_bytes=64)
    with pytest.raises(REFUSALS):
        fm.floating_matrix_step(op, fake(pool), fake(standing), fake(rate), s, h, 600.0, host_memory_budget_bytes=100)  # the raw + rebuilt packets need 272 B
    assert cp.allocs == []
    cp.dev["id"] = 1
    cp.count.__init__()
    n = len(cp.launches)
    for call in (lambda: fm.floating_matrix_step(op, fake(pool), fake(standing), fake(rate), s, h, 600.0), lambda: fm.floating_matrix_history(op, wetting_m=0.0, cumulative_intake_m=0.0),
                 lambda: fm.prepare_floating_matrix(gpu, conductivity_m_s=1e-6, suction_m=0.0, vegetation_factor=0.5)):
        with pytest.raises(REFUSALS):
            call()
    assert cp.count.d2h == 0 and len(cp.launches) == n and cp.allocs == []


def test_source_is_the_shared_body_rendered_with_floating_readers():
    src = fmc.render_source()
    assert "dz_p" not in src and "spatial" not in src and "fh_at(edges_p, depth_p[i], l)" in src
    assert fmc.kernel_provenance()["common_source_sha256"] == mc.COMMON_SOURCE_SHA256
