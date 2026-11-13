"""Floating CUDA preferential intake through the HOST-compiled rendered source and the CuPy plumbing double: NOT hardware evidence. NumPy/CUDA equivalence (partial/zero/sliver depths, inactive placeholders, block
edges, 1/2/3/65 layers, species weights), Root's rejected-share control, the bounded cold derivation (no library reduction), resident traffic/allocation accounting, the device gate and admission order."""
from __future__ import annotations

import numpy as np
import pytest
from builders import REFUSALS, pair
from conftest import needs_host_cuda
from hostd2 import fake

from maple_syrup import floating_preferential_infiltration as fp
from maple_syrup import floating_preferential_infiltration_cuda as fpc
from maple_syrup import floating_soil_layers as fl
from maple_syrup import preferential_infiltration_cuda as pc

pytestmark = needs_host_cuda
INF = float("inf")


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
    surf = rng.uniform(0.0, 0.02, N) * act + (~act) * 0.3
    return cpu, gpu, w, surf, act, rng


@pytest.mark.parametrize("L,N", [(1, 5), (2, 127), (3, 128), (3, 129), (65, 260), (3, 1000)])
def test_step_matches_numpy(cp, L, N):
    cpu, gpu, w, surf, act, rng = case(cp, L, N, L * 313 + N)
    weights = rng.uniform(0.0, 1.0, L)
    weights[0] = 0.0
    a3 = rng.uniform(0.1, 0.9, N)
    cover = rng.uniform(0.0, 2.0, N)
    kw = {"max_fraction_per_hour": 0.8, "max_intake_rate_m_s": 1e-5}
    a = fp.prepare_floating_preferential(cpu, a3=a3, background_receiver_weights=weights, species_receiver_weights=[weights[::-1].copy()], species_contributions=[cover], **kw)
    reducer_calls = []
    b = fp.prepare_floating_preferential(gpu, a3=fake(a3), background_receiver_weights=weights, species_receiver_weights=[weights[::-1].copy()], species_contributions=[fake(cover)], **kw)
    assert np.allclose(np.asarray(b._law[2]), a._law[2], rtol=1e-12, atol=1e-15) and np.allclose(np.asarray(b._law[0]), a._law[0], rtol=1e-12, atol=1e-18)
    ra = fp.floating_preferential_step(a, surf, fl.floating_continued_state(cpu, w), 1800.0)
    rb = fp.floating_preferential_step(b, fake(surf), fl.floating_continued_state(gpu, fake(w)), 1800.0)
    for x, y in ((ra.state.water_m, rb.state.water_m), (ra.surface_water_m, rb.surface_water_m), (ra.intake_m, rb.intake_m), (ra.intake_total_m, rb.intake_total_m)):
        assert np.allclose(np.asarray(x), np.asarray(y), rtol=2e-12, atol=1e-14)
    assert np.array_equal(np.asarray(rb.state.water_m)[:, ~act], w[:, ~act]) and np.array_equal(np.asarray(rb.surface_water_m)[~act], surf[~act])
    assert rb.budget.intake_m3 == pytest.approx(ra.budget.intake_m3, rel=2e-12, abs=1e-14)
    del reducer_calls


def test_root_control_rejected_share_on_the_device(cp):
    _cpu, gpu = pair(cp, [0.0, 0.1, 0.3, 0.6], [0.15], tr=0.1, tf=0.3, ts=0.5)
    op = fp.prepare_floating_preferential(gpu, a3=1.0, max_fraction_per_hour=1.0, max_intake_rate_m_s=INF, background_receiver_weights=np.array([0.0, 1.0, 1.0]))
    s = fl.floating_continued_state(gpu, fake(np.array([[0.025], [0.020], [0.0]])))
    r = fp.floating_preferential_step(op, fake(np.array([0.008])), s, 100.0)
    assert np.allclose(np.asarray(r.intake_m)[:, 0], [0.0, 0.004, 0.0], atol=1e-15)
    assert np.allclose(np.asarray(r.state.water_m)[:, 0], [0.025, 0.024, 0.0], atol=1e-15)
    assert float(np.asarray(r.surface_water_m)[0]) == pytest.approx(0.004)


def test_cold_derivation_is_bounded_and_counted(cp):
    _cpu, gpu, _w, _surf, _act, _rng = case(cp, 3, 300, 8)
    cp.count.__init__()
    cp.allocs.clear()
    fp.prepare_floating_preferential(gpu, a3=0.5, max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([1.0, 1.0, 0.0]))
    # every device->host read of the derivation is a 32-byte reducer result or a hash staging slice; no full-array download
    assert cp.count.d2h_sizes and set(cp.count.d2h_sizes) == {32}
    assert max(cp.allocs) <= 3 * 300 * 8 + 4096


def test_traffic_and_allocation_of_one_step(cp):
    L, N = 3, 300
    _cpu, gpu, w, surf, _act, _rng = case(cp, L, N, 5)
    op = fp.prepare_floating_preferential(gpu, a3=0.5, max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([1.0, 1.0, 0.0]))
    s = fl.floating_continued_state(gpu, fake(w))
    cp.count.__init__()
    cp.allocs.clear()
    fp.floating_preferential_step(op, fake(surf), s, 600.0)
    nb = fpc.n_blocks(N)
    assert cp.count.d2h == 1 and cp.count.d2h_sizes == [fpc.RAW_PACKET_BYTES] and cp.count.h2d == 0
    assert sorted(cp.allocs) == sorted([N * 8] * 2 + [L * N * 8] * 2 + [fpc.ROWS * 8 * nb, fpc.RAW_PACKET_BYTES])
    t = fpc.operation_terms("step", L, N)
    assert t["host"] == 2 * fpc.RAW_PACKET_BYTES  # raw + rebuilt packet


def test_gate_and_admission_order(cp):
    L, N = 3, 40
    _cpu, gpu, w, surf, _act, _rng = case(cp, L, N, 6)
    op = fp.prepare_floating_preferential(gpu, a3=0.5, max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([1.0, 1.0, 0.0]))
    s = fl.floating_continued_state(gpu, fake(w))
    cp.allocs.clear()
    with pytest.raises(REFUSALS, match="nothing"):
        fp.floating_preferential_step(op, fake(surf), s, 600.0, memory_budget_bytes=64)
    with pytest.raises(REFUSALS):
        fp.floating_preferential_step(op, fake(surf), s, 600.0, host_memory_budget_bytes=100)
    with pytest.raises(REFUSALS, match="nothing"):
        fp.prepare_floating_preferential(gpu, a3=0.5, max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([1.0, 1.0, 0.0]), memory_budget_bytes=16)
    assert cp.allocs == []
    cp.dev["id"] = 1
    cp.count.__init__()
    n = len(cp.launches)
    for call in (lambda: fp.floating_preferential_step(op, fake(surf), s, 600.0), lambda: fp.prepare_floating_preferential(gpu, a3=0.5, max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5,
                                                                                                                         background_receiver_weights=np.array([1.0, 1.0, 0.0]))):
        with pytest.raises(REFUSALS):
            call()
    assert cp.count.d2h == 0 and len(cp.launches) == n and cp.allocs == []


def test_invalid_cell_inputs_refused_on_the_device(cp):
    _cpu, gpu, _w, _surf, _act, _rng = case(cp, 3, 40, 2)
    bad = np.full(40, 0.5)
    bad[7] = np.nan
    with pytest.raises(REFUSALS, match="every cell"):
        fp.prepare_floating_preferential(gpu, a3=fake(bad), max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([1.0, 1.0, 0.0]))
    over = np.full(40, 0.5)
    over[3] = 1.5
    with pytest.raises(REFUSALS, match="every cell"):
        fp.prepare_floating_preferential(gpu, a3=fake(over), max_fraction_per_hour=0.8, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([1.0, 1.0, 0.0]))


def test_source_is_the_shared_body_rendered_with_floating_readers():
    src = fpc.render_source()
    assert "dz_p" not in src and "maple_syrup_spatial" not in src and "fh_at(edges_p, depth_p[i], l)" in src
    assert fpc.kernel_provenance()["common_source_sha256"] == pc.COMMON_SOURCE_SHA256


def test_species_weight_records_are_counted_in_the_host_budget_before_any_record_is_built(cp, monkeypatch):
    from maple_syrup import preferential_infiltration as pi

    _, gpu = pair(cp, edges_for(3), np.array([0.2]), tr=0.05, tf=0.3, ts=0.45)
    built = []
    real = pi.receiver_weights
    monkeypatch.setattr(pi, "receiver_weights", lambda *a, **k: built.append(1) or real(*a, **k))
    S = 50
    kw = {"a3": 0.0, "max_fraction_per_hour": 1.0, "max_intake_rate_m_s": INF, "background_receiver_weights": None, "species_receiver_weights": [np.array([0.0, 1.0, 1.0])] * S,
              "species_contributions": [fake(np.array([1.0]))] * S}
    t = fpc.operation_terms("prepare", 3, 1)
    from maple_syrup import floating_soil_layers_cuda as fcu

    need = t["host"] + fp.weight_record_bytes(3, S) + fcu.PY_ALLOWANCE_BYTES  # + the declared interpreter allowance of the shared admission
    assert need - t["host"] >= S * 5 * 3 * 8
    cp.allocs.clear()
    with pytest.raises(Exception, match="host side"):
        fp.prepare_floating_preferential(gpu, host_memory_budget_bytes=need - 1, **kw)
    assert built == [] and cp.allocs == []
    fp.prepare_floating_preferential(gpu, host_memory_budget_bytes=need, **kw)
    assert len(built) == S
