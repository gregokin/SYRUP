"""CC2 correction01: the genuine matched CPU generation-1 CONTROL on the EXACT deep pair the GPU definitions use (`fb.make_pair` == `e5.device_event` geometry, forcing, mobile pool, policy). Host-run: it proves the fixture really crosses the
net-elevation trigger (generation >= 1, local counters 0, datum != carried routing) and that the CPU continued profile (CC1) round-trips it, BEFORE any device fixture claim."""
from __future__ import annotations

import cc2_fixtures as f
import numpy as np
import pytest
from cont_builders import AVAIL, IDENT, cc, fc, restart_target, save, snapshot

MATRIX = [("numpy", False), ("numpy", True), ("numba", False), ("numba", True)]


@pytest.fixture(params=MATRIX, ids=lambda p: f"{p[0]}-{'windowed' if p[1] else 'dense'}")
def ctl(request, tmp_path):
    backend, windowed = request.param
    if backend == "numba":
        pytest.importorskip("numba")
    return f.cpu_control(tmp_path / "a", windowed, backend)


def test_the_old_fixture_deposit_is_below_the_trigger_and_the_new_one_is_above_it(ctl):
    from refresh_fixtures import lake_kwargs as old_kwargs

    g = ctl.ev0.context.geometry
    assert f.deposit_rise_m(g) > f.THRESHOLD_M > 0.1 / (g.dx_m * g.dy_m * g.bulk_density_kg_m3)  # 1 kg crosses 0.00075 m, the previous 0.1 kg (0.00032 m) does not
    assert old_kwargs()["mobile"].max() == pytest.approx(0.1) and f.lake_kwargs()["mobile"].max() == f.MOBILE_KG


def test_the_control_genuinely_completes_reroutes_to_generation_1_and_continues_with_local_counters_0(ctl):
    s = ctl.state
    assert s.topology.generation >= 1 and s.counters.n_refreshes == 0 and ctl.event._initial_topology.generation == s.topology.generation
    assert not np.array_equal(s.datum.initial_full_m, ctl.event._initial_topology.source_z_full)  # the GLOBAL datum differs from the carried routing source
    assert np.array_equal(s.datum.initial_full_m, ctl.original.source_z_full)
    assert ctl.boundary.ledger["n_refreshes"] == 1 and ctl.boundary.role == "boundary"  # a genuine refresh boundary, not a manufactured generation
    s2 = s
    for _ in range(4):
        s2 = ctl.event.step(s2, 30.0 / 3.6e6, 0.1).state
    assert s2.topology.topology_sha256 == s.topology.topology_sha256 and s2.counters.n_refreshes == 0  # below threshold: the carried routing stands


def test_the_cpu_continued_profile_round_trips_the_exact_pair_control_bit_exactly(ctl, tmp_path):
    ns = ctl
    ns.windowed = ctl.windowed
    ck = save(ns, tmp_path / "ck")
    ev2, v2, _o = restart_target(ns, ck)
    s2 = cc.load_floating_continued_checkpoint(ck, ev2, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)
    a, b = ns.state, s2
    for _ in range(3):
        a = ns.event.step(a, 30.0 / 3.6e6, 0.1).state
        b = ev2.step(b, 30.0 / 3.6e6, 0.1).state
    sa, sb = snapshot(ns.event, a, ns.occ.vertical), snapshot(ev2, b, v2)
    assert set(sa) == set(sb) and all(sa[k].tobytes() == sb[k].tobytes() for k in sa if "window_cache" not in k)
    assert fc.CONTINUED_SCHEMAS


def test_the_static_recipe_is_source_bound():
    assert f.recipe_digest() == f.recipe_digest() and len(f.recipe_digest()) == 64
    assert f.recipe_digest() == "f0750748f673ca1fe032f940b8d3b2d9212d54d8e53242707b8e8432e18912dd"
