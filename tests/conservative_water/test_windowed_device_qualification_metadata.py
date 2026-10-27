"""Focused real qualification validators; synthetic metadata, no model run.

The unissued paths/identities below are DTO operands only. No test supplies an
Original11 resource contract, queries a provider, allocates arrays or claims
that a selected runtime, device, budget or physical result was qualified.
"""
from dataclasses import replace
import json

import pytest

from maple_syrup import windowed_shared_bed_device as d
from maple_syrup.sediment_bed import BedIntegrationError


def _ref(name, *, size=1, sha='a'*64):
    return d.SourceReference('/unissued/'+name, size, sha)


def _budget():
    return d.DeviceWaterQualificationBudget(
        native_source_identity='b'*64, current_runtime_ref=_ref('runtime.json'),
        factory_source=_ref('test_windowed_shared_device_physical.py', size=20771,
            sha='8ab9ad38bfd434989f8c5e3c0727b301df41184aaef01557fb4678cc9e504eaf'),
        grant=_ref('product_budget.json'),
        source_bindings=tuple((name, _ref(name.replace('.', '/')+'.py')) for name in d._SOURCE_MODULES),
        gpu_uuid='GPU-00000000-0000-0000-0000-000000000000', source_manifest_sha256='c'*64)


def test_budget_codec_retains_all_opaque_rows_and_fixed_scope():
    budget = _budget()
    assert budget.validated() is budget
    raw = d.qualification_budget_bytes(budget)
    value = json.loads(raw)
    assert value['schema'] == 'maple_syrup.windowed_device_water.qualification_budget.v1'
    assert value['profile'] == 'small_C16_K1_V6_W3_complete_device_water_v1'
    assert value['opaque_external_bounds'] == [[name, None] for name in d._EXTERNAL_ROLES]
    assert value['opaque_primitive_units'] == [[name, None] for name in d._UNIT_NAMES]
    assert value['opaque_constructor_accounting'] is None
    assert value['execution_budgets']['leader_rss_abort_bytes'] == 5 << 30
    assert value['execution_budgets']['active_pool_limit_bytes'] == 2 << 30
    assert value['execution_budgets']['native_safety_fraction'] == 0.5
    assert not raw.endswith(b'\n')
    assert json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii') == raw


def test_budget_containing_reference_is_acyclic_but_identities_are_bound():
    budget = _budget()
    raw = d.qualification_budget_bytes(budget)
    assert d.qualification_budget_bytes(replace(budget, grant=_ref('other.json', size=123))) == raw
    for changed in (replace(budget, native_source_identity='d'*64),
                    replace(budget, current_runtime_ref=_ref('different_runtime.json')),
                    replace(budget, source_manifest_sha256='e'*64),
                    replace(budget, gpu_uuid='GPU-11111111-1111-1111-1111-111111111111')):
        assert d.qualification_budget_bytes(changed) != raw


def test_budget_rejects_missing_reordered_duplicate_and_foreign_sources():
    budget = _budget()
    rows = budget.source_bindings
    for bad in (rows[:-1], tuple(reversed(rows)), (rows[0],)*len(rows),
                ((rows[0][0], object()), *rows[1:])):
        with pytest.raises(BedIntegrationError):
            replace(budget, source_bindings=bad).validated()


def test_budget_refuses_changed_prescribed_factory():
    budget = _budget()
    for factory in (replace(budget.factory_source, size=20772),
                    replace(budget.factory_source, sha256='f'*64),
                    replace(budget.factory_source, path='/unissued/other.py')):
        with pytest.raises(BedIntegrationError):
            replace(budget, factory_source=factory).validated()


def test_budget_refuses_numeric_opaque_injection_extra_fields_and_subclass():
    budget = _budget()
    for name, value in (('external_bounds', ((d._EXTERNAL_ROLES[0], 1),)),
                        ('primitive_units', ((d._UNIT_NAMES[0], 0),)),
                        ('constructor_accounting_v2', 1), ('profile', 'other')):
        changed = replace(budget)
        object.__setattr__(changed, name, value)
        with pytest.raises(BedIntegrationError):
            changed.validated()
    class ForeignBudget(d.DeviceWaterQualificationBudget):
        pass
    with pytest.raises(BedIntegrationError):
        ForeignBudget(**vars(budget)).validated()


def test_budget_refuses_bad_plain_identity_and_reference_operands():
    budget = _budget()
    for changed in (replace(budget, gpu_uuid='GPU-unknown'),
                    replace(budget, source_manifest_sha256='C'*64),
                    replace(budget, native_source_identity=True),
                    replace(budget, current_runtime_ref=replace(budget.current_runtime_ref, size=True))):
        with pytest.raises(BedIntegrationError):
            changed.validated()


def test_original_proof_encoder_and_reservation_reject_qualification_kind():
    budget = _budget()
    # Both exact-holder refusals precede any file/provider/Native query.
    with pytest.raises(BedIntegrationError, match='exact_holder'):
        d.runtime_contract_bytes(budget)
    with pytest.raises(BedIntegrationError, match='exact_holder'):
        d.reserve_device_water_oracle_frontier(budget, device_id=0)


def test_qualification_summary_cannot_advertise_numeric_opaque_costs():
    args = (1, 1, 1, 0, 0, 0, 0, 1, 1, {'synthetic_metadata_only': True})
    summary = d.QualificationAdmissionSummary(*args)
    assert summary.known_subset_only is True
    assert all(value is None for _name, value in summary.opaque_terms)
    bad = ((summary.opaque_terms[0][0], 1), *summary.opaque_terms[1:])
    for extras in ({'opaque_terms': bad}, {'known_subset_only': False}, {'claim': 'production proof'}):
        with pytest.raises(BedIntegrationError):
            d.QualificationAdmissionSummary(*args, **extras)


def test_qualification_dispatch_does_not_touch_foreign_context_hooks():
    class Foreign:
        def __getattribute__(self, name):
            raise AssertionError('foreign hook was evaluated: '+name)
    assert d._qualification_dispatch(Foreign()) is False


# Entry protocol tests use genuine SYRUP metadata holders with opaque tokens.
# They stop before geometry/Native/provider/file admission and claim no bed.
def _entry_metadata_pair():
    limits = d.PositiveWindowedDeviceWaterLimits(
        max_host_bytes=5 << 30, max_device_bytes=2 << 30, max_mapped_bytes=128 << 20,
        max_logical_read_bytes=(1 << 63)-1, max_work=(1 << 63)-1,
        max_store_bytes=64 << 20, max_run_bytes=512 << 20, max_new_versions=1000000,
        max_entries=4096, max_external_backing_bytes=64 << 20, max_event_steps=10000,
        max_retries=10, max_transport_substeps=64, max_report_rows=4096,
        reserve_host_bytes=64 << 20, reserve_device_bytes=256 << 20,
        safety_fraction=0.5, run_dir='/unissued/entry')
    state = d.WindowedSharedBedState(*(object() for _ in d._BED_FIELDS))
    context = d.WindowedDeviceWaterContext(
        geometry=object(), grain_classes=object(), mass_resolution_kg=1.0,
        topographic_commit_spec=object(), avalanching_spec=object(),
        initial_available_fraction=object(), native_solve_limits=object(),
        native_validation_limits=object(), native_commit_limits=object(),
        source_identity='b'*64, bridge_limits=limits, runtime_bounds=_budget())
    return state, context


def test_entry_signature_accepts_bed_operands_and_requires_keyword_control():
    import inspect
    signature = inspect.signature(d.commit_qualification_device_entry)
    assert tuple(signature.parameters) == ('state', 'context', 'graph', 'reference',
        'current_time_s', 'routing_source_m', 'control')
    assert signature.parameters['control'].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        d.commit_qualification_device_entry(None, None, None, None, 0.0, None)


def test_entry_foreign_operands_do_not_dereference_event_bed_or_hooks():
    calls = []
    class Foreign:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError('foreign hook: '+name)
    with pytest.raises(BedIntegrationError, match='qualification_entry_requires'):
        d.commit_qualification_device_entry(Foreign(), Foreign(), Foreign(), Foreign(),
            0.0, Foreign(), control=Foreign())
    assert calls == [] and d._OWNER is None


def test_entry_nonreentrant_refusal_leaves_existing_authority_identity():
    original = d._OWNER
    marker = object()  # Explicit synthetic metadata marker, not a Native owner.
    try:
        d._OWNER = marker
        with pytest.raises(BedIntegrationError, match='nonreentrant_device_operation'):
            d.commit_qualification_device_entry(None, None, None, None, 0.0, None, control=None)
        assert d._OWNER is marker
    finally:
        d._OWNER = original


def test_entry_rejects_nonqualification_and_subclass_pairs_before_admission():
    state, context = _entry_metadata_pair()
    for bad in (None, object()):
        with pytest.raises(BedIntegrationError, match='qualification_entry_requires'):
            d.commit_qualification_device_entry(state, replace(context, runtime_bounds=bad),
                None, None, 0.0, None, control=None)
    class ForeignState(d.WindowedSharedBedState):
        pass
    with pytest.raises(BedIntegrationError, match='exact_active_process_thread_pair'):
        d.commit_qualification_device_entry(ForeignState(**vars(state)), context,
            None, None, 0.0, None, control=None)
    assert d._OWNER is None


def test_entry_time_source_and_control_refuse_before_native_or_owner():
    state, context = _entry_metadata_pair()
    for time in (False, -1.0, float('inf'), float('nan')):
        with pytest.raises(BedIntegrationError, match='finite_plain_time'):
            d.commit_qualification_device_entry(state, context, None, None,
                time, None, control=None)
    for control, source in ((None, object()), (object(), None)):
        with pytest.raises(BedIntegrationError, match='explicit_original_control'):
            d.commit_qualification_device_entry(state, context, None, None,
                0.0, source, control=control)
    calls = []
    class ForeignControl:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError('foreign control hook')
    with pytest.raises(BedIntegrationError, match='exact_holder'):
        d.commit_qualification_device_entry(state, context, None, None,
            0.0, object(), control=ForeignControl())
    assert calls == [] and d._OWNER is None


def test_entry_commit_only_summary_has_no_hydraulics_or_numeric_opaque_terms():
    summary = d.QualificationAdmissionSummary(4, 10, 20, 1, 0, 0, 0, 30, 40, None)
    assert summary.hydraulics is None and summary.known_subset_only is True
    assert len(summary.opaque_terms) == 20
    assert all(value is None for _name, value in summary.opaque_terms)
    for changed in ({'known_subset_only': False}, {'claim': 'complete allocation proof'},
                    {'opaque_terms': ((*summary.opaque_terms[0][:-1], 1), *summary.opaque_terms[1:])}):
        with pytest.raises(BedIntegrationError):
            d.QualificationAdmissionSummary(4, 10, 20, 1, 0, 0, 0, 30, 40, None, **changed)
