"""Floating preferential intake: Root's rejected-share control, common full-soil reference, inactive placeholders, species, failure behaviour."""
from __future__ import annotations

import numpy as np
import pytest
from builders import REFUSALS, column, state

from maple_syrup import floating_preferential_infiltration as fp
from maple_syrup import preferential_infiltration as pi
from maple_syrup import subsurface as ss

EDGES = [0.0, 0.1, 0.3, 0.6]
INF = float("inf")


def col(backend, depth=(0.15,), active=None, ts=0.5):
    return column(EDGES, list(depth), tr=0.1, tf=0.3, ts=ts, backend=backend, active=active, stress=0.2)


def op_for(c, weights=(0.0, 1.0, 1.0), a3=1.0, **kw):
    return fp.prepare_floating_preferential(c, a3=a3, max_fraction_per_hour=1.0, max_intake_rate_m_s=INF, background_receiver_weights=np.array(weights), **kw)


def test_root_control_rejected_rock_share_stays_on_surface(backend):
    c = col(backend)
    s = state(c, [[0.025], [0.020], [0.0]])
    r = fp.floating_preferential_step(op_for(c), np.array([0.008]), s, 100.0)
    assert np.allclose(r.intake_m[:, 0], [0.0, 0.004, 0.0], atol=1e-15)
    assert r.intake_total_m[0] == pytest.approx(0.004)
    assert np.allclose(r.state.water_m[:, 0], [0.025, 0.024, 0.0], atol=1e-15)
    assert r.surface_water_m[0] == pytest.approx(0.004)  # the rejected rock share is NOT renormalised into another layer
    assert r.state.t_s == s.t_s


def test_zero_dt_is_identity(backend):
    c = col(backend)
    s = state(c, [[0.025], [0.020], [0.0]])
    r = fp.floating_preferential_step(op_for(c), np.array([0.008]), s, 0.0)
    assert np.array_equal(r.state.water_m, s.water_m) and r.surface_water_m[0] == 0.008 and r.intake_total_m[0] == 0.0


def test_inactive_placeholder_preserved_and_ignored(backend):
    c = col(backend, depth=(0.15, 0.15), active=np.array([True, False]))
    w = np.array([[0.025, 0.9], [0.020, 0.9], [0.0, 0.9]])
    s = state(c, w)
    surf = np.array([0.008, 0.5])
    r = fp.floating_preferential_step(op_for(c), surf, s, 100.0)
    assert np.array_equal(r.state.water_m[:, 1], w[:, 1]) and r.surface_water_m[1] == 0.5 and r.intake_total_m[1] == 0.0
    assert r.budget.intake_m3 == pytest.approx(0.004)


def test_common_full_soil_reference_bitwise(backend):
    """Full soil reproduces the common per-cell body (bit for bit; the same shared function) for scalar and per-cell properties."""
    c = column(EDGES, [0.6, 0.9, 0.7], tr=0.1, tf=0.3, ts=0.5, backend=backend, stress=0.2)
    rng = np.random.default_rng(5)
    dz = np.diff(EDGES)
    w = rng.uniform(0.0, 0.3, (3, 3)) * dz[:, None]
    surf = rng.uniform(0.0, 0.01, 3)
    op = fp.prepare_floating_preferential(c, a3=0.4, max_fraction_per_hour=0.7, max_intake_rate_m_s=1e-5, background_receiver_weights=np.array([1.0, 2.0, 1.0]))
    r = fp.floating_preferential_step(op, surf, state(c, w), 1800.0)
    rate, mr, recv = op._law
    coef = fp.volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    ctx = pi._StepContext(3, 3, np.ones(3, dtype=bool), pi.cell_closure_coefficient(3))
    ref = pi._reference_step_body(ctx, rate, np.broadcast_to(mr, (3,)), recv, 0.5 * dz, coef * (0.5 * dz), surf, w, 1800.0)
    if backend == "numpy":
        assert np.array_equal(r.state.water_m, ref.water_m) and np.array_equal(r.surface_water_m, ref.surface_m)
    else:  # expm1 may differ by an ulp between libraries (declared bound of the common Numba kernel)
        assert np.allclose(r.state.water_m, ref.water_m, rtol=2e-12, atol=1e-14) and np.allclose(r.surface_water_m, ref.surface_m, rtol=2e-12, atol=1e-14)


def test_species_weights_fixed_and_zero_capacity_rejects(backend):
    c = col(backend)
    s = state(c, [[0.0], [0.0], [0.0]])
    op = fp.prepare_floating_preferential(c, a3=0.0, max_fraction_per_hour=1.0, max_intake_rate_m_s=INF, background_receiver_weights=None,
                                          species_receiver_weights=[np.array([0.0, 0.0, 1.0])], species_contributions=[np.array([1.0])])
    r = fp.floating_preferential_step(op, np.array([0.004]), s, 100.0)
    assert r.intake_total_m[0] == 0.0 and r.surface_water_m[0] == 0.004 and np.array_equal(r.state.water_m, s.water_m)


def test_invalid_inputs_refused_unchanged(backend):
    c = col(backend)
    s = state(c, [[0.025], [0.020], [0.0]])
    op = op_for(c)
    bad = np.array([np.nan])
    with pytest.raises(REFUSALS):
        fp.floating_preferential_step(op, bad, s, 10.0)
    with pytest.raises(REFUSALS, match="nothing was allocated"):
        fp.floating_preferential_step(op, np.array([0.001]), s, 10.0, memory_budget_bytes=0)
    with pytest.raises(REFUSALS):
        fp.prepare_floating_preferential(c, a3=2.0, max_fraction_per_hour=1.0, max_intake_rate_m_s=INF, background_receiver_weights=np.array([1.0, 1.0, 1.0]))
    assert np.array_equal(s.water_m, np.array([[0.025], [0.020], [0.0]]))


def test_old_class_refuses_floating_owner(backend):
    from maple_syrup import preferential_infiltration as old
    c = col(backend)
    with pytest.raises(REFUSALS):
        old.prepare_preferential(c, a3=1.0, max_fraction_per_hour=1.0, max_intake_rate_m_s=INF, background_receiver_weights=np.array([1.0, 1.0, 1.0]))


def test_species_count_enters_the_host_estimate_and_admission_precedes_record_construction(backend, monkeypatch):
    """Small N / large S isolates the weight-record cost: it is counted (>= 5 L 8 B per record), n_species is validated, and a budget below it refuses BEFORE any record is built."""
    c = col(backend)
    L, N = 3, 1
    e0 = fp.estimate_floating_preferential_memory(L, N, backend=backend, operation="prepare", n_species=0)
    e100 = fp.estimate_floating_preferential_memory(L, N, backend=backend, operation="prepare", n_species=100)
    assert e100.required_bytes - e0.required_bytes == 100 * (5 * L * 8 + fp.PY_RECORD_ALLOWANCE_BYTES) == fp.weight_record_bytes(L, 100) - fp.weight_record_bytes(L, 0)
    for bad in (-1, 1.5, True, "3"):
        with pytest.raises(REFUSALS):
            fp.estimate_floating_preferential_memory(L, N, backend=backend, operation="prepare", n_species=bad)
    from maple_syrup import preferential_infiltration as pi
    built = []
    real = pi.receiver_weights
    monkeypatch.setattr(pi, "receiver_weights", lambda *a, **k: built.append(1) or real(*a, **k))
    S = 100
    need = fp.estimate_floating_preferential_memory(L, N, backend=backend, operation="prepare", n_species=S).required_bytes
    kw = {"a3": 0.0, "max_fraction_per_hour": 1.0, "max_intake_rate_m_s": INF, "background_receiver_weights": None, "species_receiver_weights": [np.array([0.0, 1.0, 1.0])] * S,
              "species_contributions": [np.array([1.0])] * S}
    with pytest.raises(REFUSALS, match="nothing was allocated"):
        fp.prepare_floating_preferential(c, memory_budget_bytes=need - 1, **kw)
    assert built == []
    op = fp.prepare_floating_preferential(c, memory_budget_bytes=need, **kw)
    assert op.n_species == S and len(built) == S
