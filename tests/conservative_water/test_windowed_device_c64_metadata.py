"""Focused C64 metadata/refusal checks; no Native/provider/model operation.

Synthetic path, UUID and holder operands test validators only. They issue no
runtime, device, Source-pair, resource-unit or physical qualification fact.
The author has neither imported nor executed this module.
"""
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

import pytest

from maple_syrup import windowed_shared_bed as generic
from maple_syrup import windowed_shared_bed_device as d
from maple_syrup.sediment_bed import BedIntegrationError
from maple_syrup.sediment_event import SedimentEventControl, SedimentEventResult, SedimentEventState
from maple_syrup.storm import StormControl


def _ref(name, *, size=1, sha='a'*64):
    return d.SourceReference('/unissued/'+name, size, sha)


def _budget():
    return d.DeviceWaterC64QualificationBudget(
        native_source_identity='b'*64, current_runtime_ref=_ref('C64-runtime.json'),
        factory_source=_ref('test_windowed_shared_device_c64_qualification.py',
            size=d._C64_FACTORY_SIZE, sha=d._C64_FACTORY_SHA256),
        grant=_ref('C64-budget.json'),
        source_bindings=tuple((name, _ref(name.replace('.', '/')+'.py')) for name in d._SOURCE_MODULES),
        gpu_uuid='GPU-00000000-0000-0000-0000-000000000000', source_manifest_sha256='c'*64)


def _limits():
    return d.PositiveWindowedDeviceWaterLimits(
        max_host_bytes=5 << 30, max_device_bytes=2 << 30, max_mapped_bytes=128 << 20,
        max_logical_read_bytes=(1 << 63)-1, max_work=(1 << 63)-1,
        max_store_bytes=64 << 20, max_run_bytes=512 << 20, max_new_versions=1000000,
        max_entries=4096, max_external_backing_bytes=64 << 20, max_event_steps=10000,
        max_retries=10, max_transport_substeps=64, max_report_rows=4096,
        reserve_host_bytes=64 << 20, reserve_device_bytes=256 << 20,
        safety_fraction=0.5, run_dir='/unissued/C64')


def _pair():
    # These exact SYRUP metadata holders contain opaque tokens, not a Native
    # state/capability. Every action must still refuse before any such access.
    state = d.WindowedSharedBedState(*(object() for _ in d._BED_FIELDS))
    context = d.WindowedDeviceWaterC64Context(
        geometry=object(), grain_classes=object(), mass_resolution_kg=1.0,
        topographic_commit_spec=object(), avalanching_spec=object(),
        initial_available_fraction=object(), native_solve_limits=object(),
        native_validation_limits=object(), native_commit_limits=object(),
        native_dimension_profile=object(), source_identity='b'*64,
        bridge_limits=_limits(), runtime_bounds=_budget())
    return state, context


def _event(phase=None):
    return SedimentEventState(storm=object(), bed=object(), sediment_velocity_m_s=object(),
        graph=object(), network=object(), grid=object(), terrain=object(), phase=phase,
        routing_source_elevation_m=object())


def _control():
    return SedimentEventControl(storm=StormControl(implementation='cuda', root_solver='bisection',
        max_steps=10000, max_dt_s=1.0, max_retries=10), transport_scheme='upwind',
        transport_implementation='array')


def test_c64_budget_is_a_distinct_exact_kind_and_schema():
    budget = _budget()
    assert budget.validated() is budget
    assert not isinstance(budget, d.DeviceWaterQualificationBudget)
    assert not isinstance(budget, d.DeviceWaterRuntimeBounds)
    value = json.loads(d.c64_qualification_budget_bytes(budget))
    assert value['schema'] == 'maple_syrup.windowed_device_water.c64_qualification_budget.v1'
    assert value['profile'] == 'C64_K1_V20_W3_complete_device_water_v1'
    assert value['opaque_external_bounds'] == [[name, None] for name in d._EXTERNAL_ROLES]
    assert value['opaque_primitive_units'] == [[name, None] for name in d._UNIT_NAMES]
    assert value['opaque_constructor_accounting'] is None
    assert value['execution_budgets'] == dict(d._QUALIFICATION_EXECUTION_BUDGETS)
    assert value['claim'] == d._QUALIFICATION_CLAIM


def test_c64_budget_bytes_are_canonical_acyclic_and_bind_all_provenance():
    budget = _budget()
    raw = d.c64_qualification_budget_bytes(budget)
    value = json.loads(raw)
    assert json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii') == raw
    assert not raw.endswith(b'\n') and 'grant' not in value
    assert d.c64_qualification_budget_bytes(replace(budget, grant=_ref('other.json', size=123))) == raw
    for changed in (replace(budget, native_source_identity='d'*64),
                    replace(budget, current_runtime_ref=_ref('other-runtime.json')),
                    replace(budget, source_manifest_sha256='e'*64),
                    replace(budget, gpu_uuid='GPU-11111111-1111-1111-1111-111111111111'),
                    replace(budget, source_bindings=((budget.source_bindings[0][0], _ref('other.py')),
                                                   *budget.source_bindings[1:]))):
        assert d.c64_qualification_budget_bytes(changed) != raw


def test_c64_factory_pin_matches_actual_new_source_and_refuses_small_factory():
    raw = Path(__file__).with_name('test_windowed_shared_device_c64_qualification.py').read_bytes()
    assert (len(raw), hashlib.sha256(raw).hexdigest()) == (d._C64_FACTORY_SIZE, d._C64_FACTORY_SHA256)
    budget = _budget()
    for ref in (replace(budget.factory_source, size=budget.factory_source.size+1),
                replace(budget.factory_source, sha256='f'*64),
                replace(budget.factory_source, path='/unissued/test_windowed_shared_device_physical.py'),
                _ref('test_windowed_shared_device_physical.py', size=20771,
                     sha='8ab9ad38bfd434989f8c5e3c0727b301df41184aaef01557fb4678cc9e504eaf')):
        with pytest.raises(BedIntegrationError, match='C64_exact_authored_factory'):
            replace(budget, factory_source=ref).validated()


@pytest.mark.parametrize('name,value', [
    ('gpu_uuid', 'GPU-unknown'), ('source_manifest_sha256', 'C'*64),
    ('native_source_identity', True), ('source_bindings', []),
    ('current_runtime_ref', _ref('bad.json', size=True)),
])
def test_c64_budget_rejects_wrong_plain_metadata_domains(name, value):
    with pytest.raises(BedIntegrationError):
        replace(_budget(), **{name: value}).validated()


def test_c64_budget_requires_complete_ordered_source_roster():
    budget = _budget()
    rows = budget.source_bindings
    for bad in (rows[:-1], tuple(reversed(rows)), (rows[0],)*len(rows),
                ((rows[0][0], object()), *rows[1:])):
        with pytest.raises(BedIntegrationError):
            replace(budget, source_bindings=bad).validated()


@pytest.mark.parametrize('name,value', [('external_bounds', (('query', 1),)),
    ('primitive_units', (('int', 0),)), ('constructor_accounting_v2', 1), ('profile', 'Small')])
def test_c64_budget_cannot_inject_issued_opaque_cost_or_extra_field(name, value):
    budget = _budget()
    object.__setattr__(budget, name, value)
    with pytest.raises(BedIntegrationError, match='closed_holder'):
        budget.validated()


def test_c64_budget_subclass_is_not_admitted():
    class Foreign(d.DeviceWaterC64QualificationBudget):
        pass
    with pytest.raises(BedIntegrationError, match='exact_holder'):
        Foreign(**vars(_budget())).validated()


@pytest.mark.parametrize('name,value', [('_C64_SCHEMA', 'other'), ('_C64_PROFILE', 'Small'),
    ('_C64_FACTORY_SIZE', 20771), ('_C64_FACTORY_SHA256', 'a'*64)])
def test_c64_identity_constants_cannot_be_rebound_into_new_authority(name, value):
    old = getattr(d, name)
    try:
        setattr(d, name, value)
        with pytest.raises(BedIntegrationError):
            _budget().validated()
    finally:
        setattr(d, name, old)


def test_small_and_strict_budget_receivers_reject_c64_before_io_or_native():
    budget = _budget()
    for receiver in (d.qualification_budget_record, d.qualification_budget_bytes,
                     d.runtime_contract_bytes):
        with pytest.raises(BedIntegrationError, match='exact_holder'):
            receiver(budget)
    for receiver in (d.reserve_qualification_device_water_oracle_frontier,
                     d.reserve_device_water_oracle_frontier):
        with pytest.raises(BedIntegrationError, match='exact_holder'):
            receiver(budget, device_id=0)


def test_c64_context_is_disjoint_and_requires_explicit_profile():
    state, context = _pair()
    assert not isinstance(context, d.WindowedDeviceWaterContext)
    assert not isinstance(context, d.WindowedAvalancheContext)
    assert d._device_tag(context) and d.is_shared_pair(state, context)
    assert d._positive_context(context)
    values = vars(context).copy()
    del values['native_dimension_profile']
    with pytest.raises(TypeError):
        d.WindowedDeviceWaterC64Context(**values)


def test_c64_pair_rejects_subclasses_foreign_budget_and_extra_fields():
    state, context = _pair()
    class Foreign(d.WindowedDeviceWaterC64Context):
        pass
    bad = Foreign(**vars(context))
    assert d._device_tag(bad)
    for other in (bad, replace(context, runtime_bounds=None), replace(context, runtime_bounds=object())):
        with pytest.raises(BedIntegrationError):
            d._positive_pair_guard(state, other)
    extra = replace(context)
    object.__setattr__(extra, 'additional_profile', object())
    with pytest.raises(BedIntegrationError, match='closed_holder'):
        d._positive_pair_guard(state, extra)


def test_c64_foreign_kind_checks_do_not_evaluate_instance_hooks():
    calls = []
    class Foreign:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError('foreign instance hook: '+name)
    assert not d._c64_tag(Foreign())
    with pytest.raises(BedIntegrationError, match='C64_exact_state_context_pair'):
        d._c64_pair_guard(Foreign(), Foreign())
    with pytest.raises(BedIntegrationError, match='exact_holder'):
        d.complete_c64_qualification_device_event(Foreign(), Foreign(), None, None, None,
            None, None, max_end_s=1.0, control=None, policy=None, report_every_s=1.0)
    assert calls == [] and d._OWNER is None


def test_c64_strict_public_paths_refuse_before_native_or_bed_access():
    state, context = _pair()
    with pytest.raises(BedIntegrationError, match='C64_requires_separate_qualification_producer'):
        d.produce_device_water_phase(state, context, 'initial')
    with pytest.raises(BedIntegrationError, match='C64_requires_separate_qualification_entry'):
        d.complete_positive_windowed_device_event(object(), context, None, None, None, None,
            None, max_end_s=1.0, control=None, policy=None, report_every_s=1.0)
    assert not d._qualification_dispatch(context) and d._OWNER is None


def test_c64_old_small_entry_does_not_accept_c64_context():
    state, context = _pair()
    with pytest.raises(BedIntegrationError, match='qualification_entry_requires'):
        d.commit_qualification_device_entry(state, context, None, None, 0.0, None, control=None)
    assert d._OWNER is None


def test_c64_entry_nonreentrant_refusal_retains_existing_owner_identity():
    old, marker = d._OWNER, object()
    try:
        d._OWNER = marker
        with pytest.raises(BedIntegrationError, match='nonreentrant_device_operation'):
            d.commit_c64_qualification_device_entry(None, None, None, None, 0.0, None, control=None)
        assert d._OWNER is marker
    finally:
        d._OWNER = old


def test_c64_entry_refuses_bad_time_and_missing_original_operands_before_native():
    state, context = _pair()
    for t in (False, -1.0, float('inf'), float('nan')):
        with pytest.raises(BedIntegrationError, match='finite_plain_time'):
            d.commit_c64_qualification_device_entry(state, context, None, None, t, None, control=None)
    for control, source in ((None, object()), (object(), None)):
        with pytest.raises(BedIntegrationError, match='explicit_original_control'):
            d.commit_c64_qualification_device_entry(state, context, None, None, 0.0, source, control=control)
    assert d._OWNER is None


@pytest.mark.parametrize('name,value', [('implementation', 'array'), ('root_solver', 'newton'),
    ('max_dt_s', 2.0), ('max_steps', 9999), ('max_retries', 9), ('bisection_iterations', 201)])
def test_c64_projection_exclusions_require_exact_original_hydraulic_controls(name, value):
    control = _control()
    changed = replace(control, storm=replace(control.storm, **{name: value}))
    with pytest.raises(BedIntegrationError, match='original_cuda_bisection_upwind_step_profile'):
        d._control_projection(None, changed)


@pytest.mark.parametrize('name,value', [('transport_scheme', 'characteristic'),
    ('max_transport_substeps', 63), ('max_commit_log', 255), ('max_rejections_recorded', 99),
    ('commit', False), ('record_exchange_history', True)])
def test_c64_projection_exclusions_require_exact_original_transport_controls(name, value):
    with pytest.raises(BedIntegrationError, match='original_cuda_bisection_upwind_step_profile'):
        d._control_projection(None, replace(_control(), **{name: value}))


def test_c64_canonical_controls_stay_unchanged_and_bind_all_scalar_fields():
    control = _control()
    before = (vars(control).copy(), vars(control.storm).copy())
    projection = d._control_projection(None, control)
    assert dict(projection[0])['transport_scheme'] == 'upwind'
    assert dict(projection[1])['implementation'] == 'cuda'
    assert projection != d._control_projection(None, replace(control, force_final_commit=False))
    assert (vars(control), vars(control.storm)) == before


def test_c64_phase_none_guard_repeats_for_current_and_terminal_prior():
    state = _event()
    prior = object.__new__(SedimentEventResult)
    for f in fields(SedimentEventResult):
        object.__setattr__(prior, f.name, state if f.name == 'state' else object())
    d._c64_phase_roots((('state', state), ('prior', prior)))
    for rows in ((('state', replace(state, phase=object())),),
                 (('prior', replace(prior, state=replace(state, phase=object()))),)):
        with pytest.raises(BedIntegrationError, match='C64_upwind_phase_None'):
            d._c64_phase_roots(rows)


def test_c64_phase_root_foreign_state_and_nonplain_names_are_refused():
    for rows in ([('state', _event())], ((1, _event()),), (('state', object()),), (('prior', object()),)):
        with pytest.raises(BedIntegrationError):
            d._c64_phase_roots(rows)


def test_c64_source_projection_retains_full_host_and_work_while_selecting_device_outputs():
    assert d._c64_interval_units() == (295791, 398619)
    rows, work = d._c64_source_frontier()
    assert sum(value for _name, value in rows) == 3867120251
    assert work == 8972632403
    assert all(type(value) is int and value > 0 for _name, value in rows)
    assert dict(rows)['typed_backend_flags_lists_text'] == 295791*2440
    assert d._c64_visible_device_payload() == 305694720
    # Original generic Small receiver and all its counts remain distinct.
    assert generic._positive_interval_units() == (271839, 297579)


@pytest.mark.parametrize('name', ['_POSITIVE_WORD', '_POSITIVE_HEADER', '_POSITIVE_INTERFACE',
    '_POSITIVE_GRID', '_POSITIVE_FLAG', '_POSITIVE_REPORT_ROWS', '_POSITIVE_PLANNER_EXTENT',
    '_REGIME_COUNTER_LOGICAL_HOST_BYTES'])
def test_c64_projection_refuses_changed_original_generic_control(name):
    old = getattr(generic, name)
    try:
        setattr(generic, name, old+1)
        with pytest.raises(BedIntegrationError, match='C64_original_generic_controls'):
            d._c64_interval_units()
    finally:
        setattr(generic, name, old)


def test_c64_projection_refuses_census_reordering_wrong_kind_and_changed_coefficient():
    old_names, old_census = generic._POSITIVE_CENSUS_NAMES, generic._VISIBLE_REQUEST_CENSUS
    try:
        generic._POSITIVE_CENSUS_NAMES = tuple(reversed(old_names))
        with pytest.raises(BedIntegrationError, match='C64_original_census_roster'):
            d._c64_interval_units()
        generic._POSITIVE_CENSUS_NAMES = old_names
        generic._VISIBLE_REQUEST_CENSUS = dict(old_census)
        name = old_names[0]
        both = old_census[name]
        coefficient, powers = both[0][0]
        generic._VISIBLE_REQUEST_CENSUS[name] = (((coefficient+1, powers), *both[0][1:]), both[1])
        with pytest.raises(BedIntegrationError, match='C64_original_census_hash'):
            d._c64_interval_units()
        generic._VISIBLE_REQUEST_CENSUS = {object(): None}
        with pytest.raises(BedIntegrationError, match='C64_original_census_roster'):
            d._c64_interval_units()
    finally:
        generic._POSITIVE_CENSUS_NAMES, generic._VISIBLE_REQUEST_CENSUS = old_names, old_census


def test_c64_new_helpers_and_generated_methods_are_in_actual_code_binding_forests():
    d._check_code()
    bound = {(id(namespace), name) for namespace, name, _fn, _code in d._OWN_BINDINGS}
    for name in ('_c64_qualification_runtime', '_c64_profile', '_c64_phase_roots',
                 'produce_c64_qualification_device_water_phase', 'complete_c64_qualification_device_event',
                 'commit_c64_qualification_device_entry'):
        assert (id(vars(d)), name) in bound
    for kind in (d.DeviceWaterC64QualificationBudget, d.WindowedDeviceWaterC64Context):
        for name, value in vars(kind).items():
            if callable(value) and hasattr(value, '__code__'):
                assert any(fn is value and code is value.__code__ for fn, code in d._CODE_BANK)
    assert d._c64_code_charge() > 0


def test_c64_exact_context_cannot_cross_an_active_owner_or_process_boundary():
    state, context = _pair()
    owner, old = object.__new__(d._Owner), d._OWNER
    owner.context, owner.pid, owner.thread = context, d.os.getpid(), d.threading.get_ident()
    owner.closed = owner.poisoned = False
    try:
        d._OWNER = owner
        d._positive_pair_guard(state, context)
        with pytest.raises(BedIntegrationError, match='C64_exact_active_owner'):
            d._positive_pair_guard(state, replace(context))
        assert owner.poisoned is True
        owner.poisoned = False
        owner.pid += 1
        with pytest.raises(BedIntegrationError, match='C64_exact_active_owner'):
            d._positive_context(context)
    finally:
        d._OWNER = old
