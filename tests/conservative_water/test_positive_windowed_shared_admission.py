"""Focused admission controls; synthetic metadata is never a physical pass.

Root executes these under the genuine immutable runtime. The author does not
import this file or execute any test. No test fabricates Native stage evidence.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from maple_syrup import windowed_shared_bed as shared
from maple_syrup.sediment_bed import BedIntegrationError
from maple_syrup.sediment_event import SedimentEventControl
from maple_syrup.storm import StormControl


def _limits(root):
    return shared.PositiveWindowedWaterBridgeLimits(
        max_host_bytes=4 << 30, max_mapped_bytes=128 << 20,
        max_logical_read_bytes=(1 << 63) - 1, max_work=(1 << 63) - 1,
        max_store_bytes=64 << 20, max_run_bytes=128 << 20,
        max_new_versions=1000000, max_entries=4096,
        max_external_backing_bytes=64 << 20, max_event_steps=10000,
        max_retries=10, max_transport_substeps=64, max_report_rows=4096,
        reserve_host_bytes=64 << 20, run_dir=root)


def _context(root):
    return shared.WindowedAvalancheContext(
        geometry=None, grain_classes=None, mass_resolution_kg=1e-12,
        topographic_commit_spec=None, avalanching_spec=None,
        initial_available_fraction=None, window_max_journal_records=320,
        native_iteration_limits=None, native_evidence_limits=None,
        native_solve_limits=None, native_validation_limits=None,
        native_publication_limits=None, native_qualification_limits=None,
        source_identity='0' * 64, bridge_limits=_limits(root))


def _control():
    return SedimentEventControl(storm=StormControl(max_steps=10000),
                                transport_scheme='upwind')


def _bed():
    # Exact holder only; deliberately no Native payload or physical authority.
    return shared.WindowedSharedBedState(None, None, None, None, None, None, None)


def _owner(context, control):
    import threading
    return shared._PositiveOwner(context, control, (123, 456),
                                 threading.get_ident(),
                                 shared._positive_control_projection(context, control))


def _stat(pid=123, start=456):
    fields = ['0'] * 50
    fields[0], fields[19] = 'S', str(start)
    return f'{pid} (comm with ) embedded) ' + ' '.join(fields) + '\n'


def test_exact_new_limits_keep_original_physical_counts(tmp_path):
    limits = _limits(tmp_path)
    assert (limits.max_event_steps, limits.max_retries,
            limits.max_transport_substeps, limits.max_report_rows) == (10000, 10, 64, 4096)
    assert limits.max_host_bytes == 4 << 30
    assert type(limits) is shared.PositiveWindowedWaterBridgeLimits


@pytest.mark.parametrize('field,value', [
    ('max_host_bytes', (4 << 30) + 1), ('max_report_rows', 4095),
    ('max_event_steps', 9999), ('max_retries', True),
    ('max_transport_substeps', 32), ('max_mapped_bytes', 256 << 20),
])
def test_closed_profile_refuses_tuned_or_nonscalar_limits(tmp_path, field, value):
    with pytest.raises(BedIntegrationError):
        replace(_limits(tmp_path), **{field: value})


def test_stat_parser_uses_final_comm_delimiter_and_real_start_field():
    assert shared._positive_parse_stat(_stat()) == (123, 456)


@pytest.mark.parametrize('raw', ['0', '-1', '01', '1e3', '９', str(1 << 63)])
def test_proc_decimal_refuses_ambiguous_or_unbounded_fields(raw):
    with pytest.raises(BedIntegrationError):
        shared._positive_decimal(raw)


def test_stat_parser_refuses_missing_fields_and_zero_start():
    for raw in ('123 (x) S 1 2', _stat(start=0), '123 (x) ' + '0 ' * 51):
        with pytest.raises(BedIntegrationError):
            shared._positive_parse_stat(raw)


def test_mapping_is_vsize_pages_not_resident_bytes():
    assert shared._positive_parse_statm('1024 2 0 0 0 0 0\n', 4096) == 4194304


@pytest.mark.parametrize('raw,page', [
    ('0 0 0 0 0 0 0', 4096), ('1 2 0 0 0 0 0', 4096),
    ('1 0 0 0 0 0', 4096), ('1 0 0 0 0 0 0', 3000),
    ('1 0 0 0 0 0 0', True),
])
def test_mapping_unknown_domain_refuses(raw, page):
    with pytest.raises(BedIntegrationError):
        shared._positive_parse_statm(raw, page)


def test_new_family_cannot_gain_native_authority_from_readonly_holder(tmp_path, monkeypatch):
    context = _context(tmp_path)
    monkeypatch.setattr(shared, '_POSITIVE_OWNER', None)
    assert shared._positive_context(context)
    with pytest.raises(BedIntegrationError, match='owned_entry_required'):
        shared._positive_require_action(_bed(), context)


def test_active_owner_refuses_dense_pair_or_a_different_context(tmp_path, monkeypatch):
    context, control = _context(tmp_path), _control()
    monkeypatch.setattr(shared, '_POSITIVE_OWNER', _owner(context, control))
    with pytest.raises(BedIntegrationError, match='exact_active_pair'):
        shared._positive_pair_guard(shared.BedState(None, None, None, None, None, None),
                                    shared.BedContext(None, None, 1e-12, None, None, None))
    with pytest.raises(BedIntegrationError, match='exact_active_pair'):
        shared._positive_pair_guard(_bed(), _context(tmp_path))


def test_only_original_derived_force_false_control_can_bind_once(tmp_path, monkeypatch):
    context, original = _context(tmp_path), _control()
    owner = _owner(context, original)
    monkeypatch.setattr(shared, '_POSITIVE_OWNER', owner)
    inner = replace(original, force_final_commit=False)
    assert shared._positive_owner(context, inner) is owner
    assert shared._positive_owner(context, inner) is owner
    assert original.force_final_commit is True
    with pytest.raises(BedIntegrationError, match='inner_control_identity'):
        shared._positive_owner(context, replace(original, force_final_commit=False))


def test_equal_storm_clone_is_not_original_storm_authority(tmp_path, monkeypatch):
    context, original = _context(tmp_path), _control()
    monkeypatch.setattr(shared, '_POSITIVE_OWNER', _owner(context, original))
    changed = replace(original, storm=replace(original.storm), force_final_commit=False)
    with pytest.raises(BedIntegrationError, match='storm_identity'):
        shared._positive_owner(context, changed)


def test_counter_failure_is_atomic_and_never_wraps(tmp_path):
    context, original = _context(tmp_path), _control()
    owner = _owner(context, original)
    shared._positive_debit(owner, context, 7, 11, 2)
    before = (owner.work, owner.reads, owner.versions)
    with pytest.raises(BedIntegrationError, match='cumulative_ceiling'):
        shared._positive_debit(owner, context, (1 << 63) - 1, 1, 1)
    assert (owner.work, owner.reads, owner.versions) == before


def test_no_root_or_file_is_touched_by_unowned_publication(tmp_path, monkeypatch):
    context = _context(tmp_path)
    monkeypatch.setattr(shared, '_POSITIVE_OWNER', None)
    with pytest.raises(BedIntegrationError, match='owned_entry_required'):
        shared._positive_publish(_bed(), context, 0.0)
    assert list(tmp_path.iterdir()) == []


def test_escaping_original_error_poison_and_close_do_not_replace_error(tmp_path, monkeypatch):
    from importlib import import_module
    original_module = import_module('maple_syrup.complete_event')
    context, control = _context(tmp_path), _control()
    error, retained = RuntimeError('unchanged original failure'), []
    monkeypatch.setattr(shared, '_POSITIVE_OWNER', None)
    monkeypatch.setattr(shared, '_positive_current_mapped', lambda *args: ((123, 456), 1000, 4 << 30))

    def original(*args, **kwargs):
        retained.append(shared._POSITIVE_OWNER)
        assert kwargs['max_report_rows'] == 4096
        assert kwargs['continuation'] is kwargs['checkpoint_callback'] is kwargs['hydraulics'] is None
        raise error

    monkeypatch.setattr(original_module, 'complete_event', original)
    state = SimpleNamespace(bed=_bed())
    with pytest.raises(RuntimeError) as caught:
        shared.complete_positive_windowed_event(state, context, None, None, None, None, None,
            max_end_s=2.0, control=control, policy=original_module.CompletionPolicy(),
            report_every_s=1.0)
    assert caught.value is error
    assert shared._POSITIVE_OWNER is None
    assert len(retained) == 1 and retained[0].poisoned and retained[0].closed
    assert state.bed.voxel_column is None


def test_caught_new_admission_refusal_stays_poisoned(tmp_path, monkeypatch):
    context, control = _context(tmp_path), _control()
    owner = _owner(context, control)
    monkeypatch.setattr(shared, '_POSITIVE_OWNER', owner)
    try:
        shared._positive_integer(True, 'not_an_integer')
    except BedIntegrationError:
        pass  # A simulated old retry catch must grant no further operation.
    assert owner.poisoned
    with pytest.raises(BedIntegrationError, match='owned_entry_required'):
        shared._positive_owner(context, control)


def test_root_with_parent_components_cannot_pass_inventory(tmp_path, monkeypatch):
    context = _context(tmp_path)
    store = tmp_path / '..' / tmp_path.name / 'store'
    with pytest.raises(BedIntegrationError, match='store_outside_root'):
        shared._positive_inventory(context, store)
