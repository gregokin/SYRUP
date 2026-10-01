from dataclasses import replace

import numpy as np
import pytest
from test_sediment_event import chain_elevation, make_bed, setup

from maple_syrup.complete_event import (
    CompletionPolicy,
    DryHandoff,
    EventProgress,
    IncompleteEventError,
    QuietProgress,
    complete_event,
    finalize_event,
    quiet_metrics,
)
from maple_syrup.rainfall import RainfallSchedule, constant_rainfall
from maple_syrup.sediment_bed import bed_inventory, with_water
from maple_syrup.sediment_event import SedimentEventControl
from maple_syrup.storm import StormControl, initial_state


def fixture(*, wet=False, implementation='array', max_steps=10000, transport_implementation='auto', **control_kw):
    bed, ctx = make_bed(chain_elevation(3), depth_m=.001 if wet else 0.)
    control = SedimentEventControl(storm=StormControl(max_dt_s=1, max_steps=max_steps, implementation=implementation),
                                   transport_implementation=transport_implementation, **control_kw)
    s = setup(bed, ctx, ksat=1e-5, control=control)
    schedule = constant_rainfall(0, 10 if wet else 2, 100 if wet else 0)
    args = (s['state0'], ctx, s['column'], s['field'], schedule, s['vegetation'], s['sediment'])
    kw = {'max_end_s': 180 if wet else 12, 'control': control, 'policy': CompletionPolicy(hold_s=2),
          'report_every_s': 2.}
    return args, kw


def at_time(state, t, *, h=None, mobile=None):
    h = state.storm.depth_m if h is None else np.full(state.graph.shape, h)
    pool = state.bed.water.mobile_mass_by_cell_class_kg if mobile is None else np.full_like(state.bed.active_layer.mass_kg, mobile)
    bed = with_water(state.bed, fixture()[0][1], h, pool)
    return replace(state, bed=bed, storm=initial_state(state.graph, h, state.storm.soil_water_m, t_s=t))


def test_completion_accounts_dry_reset_without_sediment_loss():
    args, kw = fixture()
    before = bed_inventory(args[0].bed).copy()
    out = complete_event(*args, **kw)
    assert isinstance(out, DryHandoff)
    assert out.dry_state.t_s == 4
    for a in (out.dry_state.storm.depth_m, out.dry_state.storm.soil_water_m,
              out.dry_state.storm.discharge_m2_s, out.dry_state.bed.water.mobile_mass_by_cell_class_kg,
              out.dry_state.sediment_velocity_m_s):
        assert np.all(a == 0)
    np.testing.assert_array_equal(before, bed_inventory(out.dry_state.bed))
    assert out.accounting['soil_removed_m3'] == pytest.approx(.25 * .3 * .25 * 3)
    assert out.accounting['surface_removed_m3'] == 0
    assert out.accounting['combined_water_residual_m3'] == 0
    assert np.any(args[0].storm.soil_water_m > 0)  # caller still wet, no mutation


def test_rainfall_gap_is_not_end_of_forcing():
    args, kw = fixture()
    old = args[4]
    schedule = RainfallSchedule(np.array([0., 2., 10., 12.]), np.array([1., 0., 1.]), old.provenance)
    args = (*args[:4], schedule, *args[5:])
    result = complete_event(*args, **{**kw, 'max_end_s': 20})
    assert result.dry_state.t_s >= 14
    assert result.accounting['storm_water_budget']['rain_m3'] == pytest.approx(4 / 3.6e6 * .75)


def test_quiet_requires_domain_storage_and_mobile_not_only_outlet():
    args, kw = fixture()
    state, ctx, *_ = args
    policy = kw['policy']
    ponded = replace(at_time(state, 3, h=.01), storm=replace(at_time(state, 3, h=.01).storm, discharge_m2_s=np.zeros(state.graph.shape)))
    assert not quiet_metrics(ponded, ctx, policy, 2)[0]
    loaded = at_time(state, 3, mobile=ctx.mass_resolution_kg * 2)
    assert not quiet_metrics(loaded, ctx, policy, 2)[0]
    tiny = at_time(state, 3, mobile=ctx.mass_resolution_kg / 2)
    assert quiet_metrics(tiny, ctx, policy, 2)[0]
    assert quiet_metrics(tiny, ctx, policy, 2)[1]['mobile_per_element_limit_kg'] == ctx.mass_resolution_kg


def test_hold_resets_on_an_exceedance_and_is_measured_in_seconds():
    args, kw = fixture()
    state, ctx = args[:2]
    monitor = QuietProgress()
    for t, mobile, ready in [(2, 0, False), (3, 1e-3, False), (4, 0, False), (5.5, 0, False), (6, 0, True)]:
        monitor = monitor.observe(at_time(state, t, mobile=mobile), ctx, kw['policy'], 2)
        assert monitor.ready is ready
    assert monitor.since_s == 4


def test_duration_and_global_step_limits_do_not_dry_state():
    args, kw = fixture(wet=True)
    before = args[0].bed.water.depth_m.copy()
    with pytest.raises(IncompleteEventError, match='duration'):
        complete_event(*args, **{**kw, 'max_end_s': 4})
    np.testing.assert_array_equal(before, args[0].bed.water.depth_m)
    args, kw = fixture(max_steps=3)
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    assert isinstance(paused, EventProgress)
    with pytest.raises(IncompleteEventError, match='max_steps'):
        complete_event(paused.result.state, *args[1:], **kw, continuation=paused)


def test_nonzero_terminal_deposition_and_surface_removal_are_accounted():
    args, kw = fixture()
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    r = paused.result
    pool = np.full_like(r.state.bed.active_layer.mass_kg, 1e-12)
    h = np.full(r.state.graph.shape, 1e-10)
    bed = with_water(r.state.bed, args[1], h, pool)
    state = replace(r.state, bed=bed, storm=initial_state(r.state.graph, h, r.state.storm.soil_water_m, t_s=2))
    r = replace(r, state=state, initial_mobile_kg=pool.sum(axis=(0, 1)))
    policy = CompletionPolicy(hold_s=1)
    p = replace(paused, result=r, policy=policy, initial_surface_m3=float(h.sum() * .25),
                rainfall_end_s=0., quiet=QuietProgress(since_s=0., last_t_s=2., ready=True))
    initial_bed = bed_inventory(state.bed)
    out = finalize_event(p, args[1], args[2])
    np.testing.assert_array_equal(out.terminal_deposition_kg, pool)
    np.testing.assert_allclose(bed_inventory(out.dry_state.bed) - initial_bed,
                               pool.sum(axis=(0, 1)), atol=out.accounting['terminal_sediment_tolerance_kg'], rtol=0)
    assert out.accounting['surface_removed_m3'] == float(h.sum() * .25)
    assert out.accounting['sediment_closure']['closed']
    np.testing.assert_array_equal(p.result.state.bed.water.mobile_mass_by_cell_class_kg, pool)


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True])
def test_invalid_stopping_policy_refused(value):
    with pytest.raises(RuntimeError):
        CompletionPolicy(hold_s=value).validated()


def test_failed_terminal_commit_preserves_wet_caller(monkeypatch):
    from importlib import import_module

    module = import_module('maple_syrup.complete_event')
    args, kw = fixture()
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    ready = replace(paused, rainfall_end_s=0., policy=CompletionPolicy(hold_s=1),
                    quiet=QuietProgress(since_s=0., last_t_s=2., ready=True))
    before_bed = ready.result.state.bed.voxel_column.mass_kg.copy()
    before_soil = ready.result.state.storm.soil_water_m.copy()
    monkeypatch.setattr(module, '_commit_and_reroute', lambda *a, **k: None)
    with pytest.raises(RuntimeError, match='commit was not performed'):
        finalize_event(ready, args[1], args[2])
    np.testing.assert_array_equal(ready.result.state.bed.voxel_column.mass_kg, before_bed)
    np.testing.assert_array_equal(ready.result.state.storm.soil_water_m, before_soil)
    assert np.any(before_soil > 0)


def test_residual_only_ledger_at_same_commit_time_is_flushed():
    from maple.core.types.sediment_ledger import process_index
    from maple.coupling.sediment_ledger import accumulate_process_transfer
    from maple.surface.topographic_commit.commit import ledger_is_empty

    args, kw = fixture()
    completed = complete_event(*args, **kw)
    progress = completed.progress
    state = progress.result.state
    zeros = np.zeros_like(state.bed.active_layer.mass_kg)
    residual = zeros.copy()
    residual.flat[0] = args[1].mass_resolution_kg / 100
    ledger = accumulate_process_transfer(state.bed.ledger, process_index('water_erosion_deposition'),
                                         zeros, residual, args[1].mass_resolution_kg)
    assert not ledger_is_empty(ledger)
    assert np.all(ledger.pending_bed_mass_change_kg == 0)
    assert state.bed.committed_topography.last_commit_time_s == state.t_s
    state = replace(state, bed=replace(state.bed, ledger=ledger))
    progress = replace(progress, result=replace(progress.result, state=state))
    repeated = finalize_event(progress, args[1], args[2])
    assert ledger_is_empty(repeated.dry_state.bed.ledger)
    assert repeated.progress.result.n_commits == progress.result.n_commits + 1
    np.testing.assert_array_equal(bed_inventory(repeated.dry_state.bed), bed_inventory(state.bed))
    assert not ledger_is_empty(state.bed.ledger)  # original diagnostics preserved


def test_final_rerouting_that_reactivates_flow_refuses_dry_handoff(monkeypatch):
    from importlib import import_module

    module = import_module('maple_syrup.complete_event')
    args, kw = fixture()
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    ready = replace(paused, rainfall_end_s=0., policy=CompletionPolicy(hold_s=1),
                    quiet=QuietProgress(since_s=0., last_t_s=2., ready=True))
    original_commit = module._commit_and_reroute

    def commit_with_reactivated_flow(*a, **k):
        result = original_commit(*a, **k)
        # Simulate a post-commit routing result outside the stopping criterion.
        # Exercise the terminal transaction guard, not an alternative solver.
        q = result.storm.discharge_m2_s.copy()
        q.flat[0] = ready.policy.max_unit_discharge_m2_s * 2
        return replace(result, storm=replace(result.storm, discharge_m2_s=q))

    monkeypatch.setattr(module, '_commit_and_reroute', commit_with_reactivated_flow)
    before_bed = ready.result.state.bed.voxel_column.mass_kg.copy()
    before_soil = ready.result.state.storm.soil_water_m.copy()
    before_count = ready.result.state.bed.committed_topography.commit_count
    with pytest.raises(RuntimeError, match='invalidated completion'):
        finalize_event(ready, args[1], args[2])
    np.testing.assert_array_equal(ready.result.state.bed.voxel_column.mass_kg, before_bed)
    np.testing.assert_array_equal(ready.result.state.storm.soil_water_m, before_soil)
    assert ready.result.state.bed.committed_topography.commit_count == before_count
