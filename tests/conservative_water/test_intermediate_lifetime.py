"""Accepted-step scratch must die before the next full attempt allocates.

CW-B20: the event loop calls the PRIVATE `_sediment_accounting_step`, whose returned record no longer exposes the
MAPLE pickup result. The intermediates are therefore captured where they are created instead of through the
returned attempt, and they must be dead when the next attempt starts. The original public step is not used by the
loop any more, so patching it would observe nothing; the injection point follows the loop.

CW-E47C1: the resident-window accounting step has TWO actual paths (`sediment_event._sediment_accounting_step`):

* dense phase storage (and every non-CSR / non-windowed / full-form case): the original two-call path -- a MAPLE
  pickup `apply_bed_demand`, the transport, then a deposit `apply_bed_demand`. The post-pickup MAPLE window and
  ledger arrays exist and must die (the original four intermediates, unchanged).
* CSR phase storage on a windowed bed (CW-B76): ONE fused call, MAPLE's private paired windowed transaction through
  `sediment_bed.apply_bed_pickup_transport_deposit`. The post-pickup MAPLE state stays PRIVATE to MAPLE (there is no
  standalone pickup window / ledger result to observe), and the step itself owns three transients: the MAPLE
  `PickupView` (accounting fields copied out, then `del view`), the paired deposit result (copied into
  `DepositAccounting`, then `del deposit`) and the deposit demand returned by the transport callback. Those owners,
  plus the physics demand and (host transport record) the transport deposition request, must die before the next
  attempt. They are observed with transparent wrappers / weakrefs on the REAL path; a deliberate retention is a
  negative control that the detector must catch."""
from __future__ import annotations

import weakref

import numpy as np
import pytest
from test_exchange_history import run
from test_windowed_storm import window_inputs

import maple_syrup.sediment_event as event_module

DENSE_KEYS = {'pickup_window', 'pickup_ledger', 'physics_demand', 'transport_deposition'}
# fused CSR path: the paired-transaction owners; `transport_deposition` is produced by `water_demand_from_transport`
# for the host transport record (NumPy) and by `_water_demand_from_accounting_transport` for the private CUDA CSR
# accounting record (CuPy, CW-B93), whose owner is the returned `paired_demand` either way
FUSED_KEYS = {'paired_view', 'paired_result', 'paired_demand', 'physics_demand'}


def instrument(monkeypatch, *, retain=None):
    """Transparent wrappers on the actual loop. `retain` (negative control) names one captured owner that a test-only
    list keeps alive deliberately, so the next attempt must see it live."""
    original_step = event_module._sediment_accounting_step
    original_apply = event_module.apply_bed_demand
    original_paired = event_module.apply_bed_pickup_transport_deposit
    original_physics = event_module.sediment_physics_step
    original_demand = event_module.water_demand_from_transport
    rec = {'current': {}, 'old': {}, 'calls': 0, 'paired_calls': 0, 'apply_calls': 0, 'seen': [], 'kept': [],
           'leaks': []}
    current, old = rec['current'], rec['old']

    def keep(name, obj):
        current[name] = weakref.ref(obj)
        if name == retain:
            rec['kept'].append(obj)  # the deliberate leak of the negative control

    def apply(*args, **kwargs):
        bed, result = original_apply(*args, **kwargs)
        rec['apply_calls'] += 1
        if 'pickup_window' not in current:  # the first MAPLE call of a two-call attempt is the pickup
            keep('pickup_window', result.new_voxel_column.window_kg)
            keep('pickup_ledger', result.new_ledger.actual_mass_positive_kg)
        return bed, result

    def paired(state, context, holder, between, *, entry):
        rec['paired_calls'] += 1

        def observed_between(view):
            keep('paired_view', view)
            depth, mobile, demand = between(view)
            keep('paired_demand', demand)
            return depth, mobile, demand
        view, bed, result = original_paired(state, context, holder, observed_between, entry=entry)
        assert current['paired_view']() is view  # the callback saw exactly the view the step receives
        keep('paired_result', result)
        return view, bed, result

    def physics(*args, **kwargs):
        out = original_physics(*args, **kwargs)
        keep('physics_demand', out.requested_pickup_kg)
        return out

    def demand(transport, *args, **kwargs):
        keep('transport_deposition', transport.deposition_request_kg)
        return original_demand(transport, *args, **kwargs)

    def observed(*args, **kwargs):
        if old:
            live = sorted(name for name, ref in old.items() if ref() is not None)
            if live:
                rec['leaks'].append(live)
                if retain is None:
                    raise AssertionError(f'prior-step intermediates retained into next allocation: {live}')
        current.clear()
        attempt = original_step(*args, **kwargs)
        rec['seen'].append(frozenset(current))
        old.clear()
        old.update(current)
        rec['calls'] += 1
        return attempt

    monkeypatch.setattr(event_module, 'apply_bed_demand', apply)
    monkeypatch.setattr(event_module, 'apply_bed_pickup_transport_deposit', paired)
    monkeypatch.setattr(event_module, 'sediment_physics_step', physics)
    monkeypatch.setattr(event_module, 'water_demand_from_transport', demand)
    monkeypatch.setattr(event_module, '_sediment_accounting_step', observed)
    return rec


def backend_xp(backend):
    if backend == 'numpy':
        return np
    from maple.core.backend import cupy_module, gpu_execution_available
    if not gpu_execution_available():
        pytest.skip('actual CuPy device unavailable; no device lifetime claim')
    return cupy_module()


@pytest.mark.parametrize('storage', ['dense', 'csr'])
@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
def test_prior_attempt_scratch_released_before_next_step(tmp_path, monkeypatch, backend, storage):
    xp = backend_xp(backend)
    s, ctx, ctl, schedule = window_inputs(xp, tmp_path, storage=storage)
    rec = instrument(monkeypatch)
    result = run(s, ctx, ctl, schedule, 3., cadence=1.)
    assert rec['calls'] >= 3 and result.closure()['closed'] and not rec['leaks']
    if storage == 'dense':
        # the original two-call path: two MAPLE calls per attempt, no fused transaction
        assert rec['paired_calls'] == 0 and rec['apply_calls'] == 2 * rec['calls']
        assert all(keys == DENSE_KEYS for keys in rec['seen']), rec['seen']
    else:
        # the actual fused path: one paired transaction per attempt and NO standalone pickup apply
        assert rec['paired_calls'] == rec['calls'] and rec['apply_calls'] == 0
        expected = FUSED_KEYS | ({'transport_deposition'} if backend == 'numpy' else set())
        assert all(keys == expected for keys in rec['seen']), rec['seen']


@pytest.mark.parametrize('retain', ['paired_view', 'paired_result', 'paired_demand'])
def test_retained_fused_owner_is_detected_negative_control(tmp_path, monkeypatch, retain):
    """The detector is meaningful: a deliberately retained fused-path owner is reported live at the next attempt."""
    s, ctx, ctl, schedule = window_inputs(np, tmp_path, storage='csr')
    rec = instrument(monkeypatch, retain=retain)
    run(s, ctx, ctl, schedule, 3., cadence=1.)
    assert rec['calls'] >= 3 and rec['leaks']
    # exactly the retained owner and what it OWNS: the deposit demand is built from the transport record's
    # deposition request (`water_demand_from_transport`), so retaining the demand keeps that array alive too
    owned = {'paired_view': ['paired_view'], 'paired_result': ['paired_result'],
             'paired_demand': ['paired_demand', 'transport_deposition']}[retain]
    assert all(live == owned for live in rec['leaks']), rec['leaks']
