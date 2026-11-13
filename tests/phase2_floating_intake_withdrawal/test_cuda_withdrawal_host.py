"""Floating CUDA withdrawal through the HOST-compiled rendered source and the CuPy plumbing double: NOT hardware evidence. NumPy/CUDA equivalence on partial/zero/sliver depths, inactive placeholders, block edges, layer
counts (1/2/3/65), plant proportionality, resident traffic accounting and the device gate. Author-written; the gated hardware definitions are in test_gpu_definitions.py (Root runs them)."""
from __future__ import annotations

import numpy as np
import pytest
from builders import REFUSALS, pair
from conftest import needs_host_cuda
from hostd2 import fake

from maple_syrup import floating_soil_layers as fl
from maple_syrup import floating_transpiration as ft
from maple_syrup import floating_transpiration_cuda as ftc
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_transpiration as st

pytestmark = needs_host_cuda


def edges_for(L):
    return np.concatenate([[0.0], np.cumsum(np.full(L, 0.6 / L))])


def case(cp, L, N, seed, *, active=True):
    rng = np.random.default_rng(seed)
    depth = rng.choice([0.0, 1e-6, 0.03, 0.1, 0.2, 0.45, 0.6, 0.9], size=N)
    act = rng.random(N) > 0.2 if active else np.ones(N, bool)
    cpu, gpu = pair(cp, edges_for(L), depth, tr=0.05, tf=0.3, ts=0.45, active=act)
    h = np.stack([fl.porous_thickness(cpu, k) for k in range(L)])
    w = h * 0.45 * rng.uniform(0.05, 0.95, h.shape)
    w[:, ~act] = rng.uniform(0.0, 2.0, (L, int((~act).sum())))  # placeholders: no capacity law
    q = rng.uniform(0.0, 0.02, h.shape) * act[None, :]
    return cpu, gpu, w, q, act


@pytest.mark.parametrize("L,N", [(1, 5), (2, 127), (3, 128), (3, 129), (65, 300), (3, 1000)])
def test_withdraw_matches_numpy(cp, L, N):
    cpu, gpu, w, q, act = case(cp, L, N, L * 1000 + N)
    s_cpu, s_gpu = fl.floating_continued_state(cpu, w), fl.floating_continued_state(gpu, fake(w))
    a = ft.floating_withdraw(cpu, s_cpu, q)
    b = ft.floating_withdraw(gpu, s_gpu, fake(q))
    for x, y in ((a.state.water_m, b.state.water_m), (a.actual_m, b.actual_m), (a.unmet_m, b.unmet_m), (a.scale, b.scale)):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    assert np.array_equal(np.asarray(b.state.water_m)[:, ~act], w[:, ~act])  # placeholder bits
    for f in ("storage_before_m3", "storage_after_m3", "requested_m3", "actual_m3", "unmet_m3"):
        assert getattr(b.budget, f) == pytest.approx(getattr(a.budget, f), rel=1e-12, abs=1e-15)
    av_cpu, av_gpu = ft.floating_layer_availability_m(cpu, s_cpu), ft.floating_layer_availability_m(gpu, s_gpu)
    assert np.array_equal(av_cpu, np.asarray(av_gpu))


def test_root_control_on_the_device(cp):
    _cpu, gpu = pair(cp, [0.0, 0.1, 0.3, 0.6], [0.15], tr=0.1, tf=0.3, ts=0.5)
    s = fl.floating_continued_state(gpu, fake(np.array([[0.025], [0.020], [0.0]])))
    r = ft.floating_withdraw(gpu, s, fake(np.array([[0.020], [0.010], [0.010]])))
    assert np.allclose(np.asarray(r.actual_m)[:, 0], [0.015, 0.010, 0.0], atol=1e-15)
    assert np.allclose(np.asarray(r.unmet_m)[:, 0], [0.005, 0.0, 0.010], atol=1e-15)
    assert np.allclose(np.asarray(r.state.water_m)[:, 0], [0.010, 0.010, 0.0], atol=1e-15)


def test_invalid_inputs_refused_unchanged_on_the_device(cp):
    _cpu, gpu = pair(cp, [0.0, 0.1, 0.3, 0.6], [0.15, 0.15], tr=0.1, tf=0.3, ts=0.5, active=np.array([True, False]))
    w = np.array([[0.025, 0.7], [0.020, 0.7], [0.0, 0.7]])
    s = fl.floating_continued_state(gpu, fake(w))
    bad = np.zeros((3, 2))
    bad[0, 1] = 0.001  # a positive request on an inactive cell
    with pytest.raises(ss.SubsurfaceError):
        ft.floating_withdraw(gpu, s, fake(bad))
    nanq = np.zeros((3, 2))
    nanq[1, 0] = np.nan
    with pytest.raises(ss.SubsurfaceError):
        ft.floating_withdraw(gpu, s, fake(nanq))
    assert np.array_equal(np.asarray(s.water_m), w)


def test_traffic_is_one_packet_and_no_state_download(cp):
    _cpu, gpu, w, q, _act = case(cp, 3, 300, 9)
    s = fl.floating_continued_state(gpu, fake(w))
    cp.count.__init__()
    cp.allocs.clear()
    ft.floating_withdraw(gpu, s, fake(q))
    assert cp.count.d2h == 1 and cp.count.d2h_sizes == [64] and cp.count.h2d == 0
    L, N = 3, 300
    nb = ftc.n_blocks(N)
    assert sorted(cp.allocs) == sorted([L * N * 8] * 4 + [8 * nb * 8, 64])  # four outputs, partials, packet; no mirror, no hashing
    cp.count.__init__()
    ft.floating_layer_availability_m(gpu, s)
    assert cp.count.d2h_sizes == [8] and cp.count.h2d == 0


def test_device_gate_precedes_everything(cp):
    _cpu, gpu, w, q, _act = case(cp, 3, 40, 3)
    s = fl.floating_continued_state(gpu, fake(w))
    cp.dev["id"] = 1
    cp.count.__init__()
    n_launch = len(cp.launches)
    with pytest.raises(REFUSALS):
        ft.floating_withdraw(gpu, s, fake(q))
    with pytest.raises(REFUSALS):
        ft.floating_layer_availability_m(gpu, s)
    assert cp.count.d2h == 0 and len(cp.launches) == n_launch


def test_admission_precedes_scratch_and_counts_host(cp):
    _cpu, gpu, w, q, _act = case(cp, 3, 40, 3)
    s = fl.floating_continued_state(gpu, fake(w))
    cp.allocs.clear()
    with pytest.raises(REFUSALS, match="nothing"):
        ft.floating_withdraw(gpu, s, fake(q), memory_budget_bytes=16)
    with pytest.raises(REFUSALS):
        ft.floating_withdraw(gpu, s, fake(q), host_memory_budget_bytes=8)  # the 64-byte packet exceeds the host budget
    assert cp.allocs == []


def test_source_is_the_shared_body_rendered_with_floating_readers():
    src = ftc.render_source()
    assert st._CUDA_BODY in src and "#define WITHDRAW_RESIDUAL" in src and "dz_p" not in src and "spatial" not in src
    assert ftc.kernel_provenance()["shared_body_sha256"]
