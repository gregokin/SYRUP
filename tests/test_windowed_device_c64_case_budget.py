"""Case-budget metadata regressions; no physical/model/provider operation.

Every /unissued path, UUID and digest below is synthetic validator input,
never a runtime/grant/device/Source-pair or capacity fact. No producer module,
Native state, device context or model run is constructed. Root alone executes
these tests in its genuine source-pinned bootstrap; the author has not run or
imported them.
"""
from dataclasses import replace
import hashlib
import json

import pytest

from maple_syrup import windowed_shared_bed_device as d
from maple_syrup.sediment_bed import BedIntegrationError


OLD_CANONICAL_SHA256 = '149fefc119e49c75449a2f77b1636c4af10b5f7f5d636c0b5d9191ac52a60059'
OLD_CANONICAL_SIZE = 5295
FACTORY_SHA256 = 'df967a08ac421a2caf76ba63f281a10af2190b37c7b2c3dce55184a8d263e7d5'


def _ref(name, *, size=1, sha='a' * 64):
    return d.SourceReference('/unissued/' + name, size, sha)


def _rows():
    return tuple((name, _ref(f'source/{i}.py')) for i, name in enumerate(d._SOURCE_MODULES))


def _old_budget():
    return d.DeviceWaterC64QualificationBudget(
        native_source_identity='b' * 64, current_runtime_ref=_ref('runtime.json'),
        factory_source=_ref('test_windowed_shared_device_c64_qualification.py', size=34469, sha=FACTORY_SHA256),
        grant=_ref('budget.json'), source_bindings=_rows(),
        gpu_uuid='GPU-00000000-0000-0000-0000-000000000000', source_manifest_sha256='c' * 64)


def _case_budget():
    return d.DeviceWaterC64CaseBudget(
        native_source_identity='b' * 64, current_runtime_ref=_ref('runtime.json'),
        producer_module='masy.case_device_c64', producer_source=_ref('masy/case_device_c64.py', size=123, sha='d' * 64),
        grant=_ref('case-budget.json'), source_bindings=_rows(),
        gpu_uuid='GPU-00000000-0000-0000-0000-000000000000', source_manifest_sha256='c' * 64,
        case_request_binding_sha256='e' * 64)


def test_old_c64_bytes_and_factory_guard_keep_original_golden():
    budget = _old_budget()
    raw = d.c64_qualification_budget_bytes(budget)
    # Derived from the pre-change explicit record literals, without executing
    # the candidate encoder: includes all22 sources, old caps/None/claim.
    assert (len(raw), hashlib.sha256(raw).hexdigest()) == (OLD_CANONICAL_SIZE, OLD_CANONICAL_SHA256)
    assert d._c64_budget(budget) is budget
    assert d._c64_refs(budget) == (budget.current_runtime_ref, budget.factory_source,
                                  budget.grant, *(ref for _name, ref in budget.source_bindings))
    for wrong in (replace(budget.factory_source, size=34470),
                  replace(budget.factory_source, sha256='f' * 64),
                  replace(budget.factory_source, path='/unissued/case_device_c64.py')):
        with pytest.raises(BedIntegrationError, match='C64_exact_authored_factory'):
            replace(budget, factory_source=wrong).validated()


def test_case_budget_has_distinct_complete_record_with_original_caps_and_opaque_none():
    budget = _case_budget()
    assert budget.validated() is budget
    assert d._c64_budget(budget) is budget
    assert not isinstance(budget, d.DeviceWaterC64QualificationBudget)
    assert not isinstance(budget, d.DeviceWaterRuntimeBounds)
    value = d.c64_case_budget_record(budget)
    assert set(value) == {'schema', 'profile', 'native_source_identity', 'current_runtime_ref',
        'producer_module', 'producer_source', 'source_bindings', 'gpu_uuid', 'source_manifest_sha256',
        'case_request_binding_sha256', 'execution_budgets', 'opaque_external_bounds',
        'opaque_primitive_units', 'opaque_constructor_accounting', 'claim'}
    assert value['schema'] == 'maple_syrup.windowed_device_water.c64_case_budget.v1'
    assert value['profile'] == 'C64_K1_V20_W3_complete_device_water_v1'
    assert value['producer_module'] == budget.producer_module
    assert value['producer_source'] == vars(budget.producer_source)
    assert value['case_request_binding_sha256'] == budget.case_request_binding_sha256
    assert value['execution_budgets'] == dict(d._QUALIFICATION_EXECUTION_BUDGETS)
    assert value['opaque_external_bounds'] == tuple((name, None) for name in d._EXTERNAL_ROLES)
    assert value['opaque_primitive_units'] == tuple((name, None) for name in d._UNIT_NAMES)
    assert value['opaque_constructor_accounting'] is None
    assert value['claim'] == d._QUALIFICATION_CLAIM
    raw = d.c64_case_budget_bytes(budget)
    assert raw == json.dumps(value, sort_keys=True, separators=(',', ':'),
                            ensure_ascii=True, allow_nan=False).encode('ascii')
    assert not raw.endswith(b'\n') and 'grant' not in value and 'factory_source' not in value
    assert d._c64_refs(budget) == (budget.current_runtime_ref, budget.producer_source,
                                  budget.grant, *(ref for _name, ref in budget.source_bindings))


def test_case_bytes_bind_request_source_and_runtime_without_grant_cycle():
    budget = _case_budget()
    raw = d.c64_case_budget_bytes(budget)
    assert d.c64_case_budget_bytes(replace(budget, grant=_ref('other-grant.json', size=42))) == raw
    for changed in (replace(budget, native_source_identity='f' * 64),
                    replace(budget, current_runtime_ref=_ref('other-runtime.json')),
                    replace(budget, producer_source=_ref('other/case_device_c64.py', sha='f' * 64)),
                    replace(budget, source_manifest_sha256='f' * 64),
                    replace(budget, case_request_binding_sha256='f' * 64),
                    replace(budget, gpu_uuid='GPU-11111111-1111-1111-1111-111111111111'),
                    replace(budget, source_bindings=((budget.source_bindings[0][0], _ref('other.py')),
                                                    *budget.source_bindings[1:]))):
        assert d.c64_case_budget_bytes(changed) != raw


@pytest.mark.parametrize('module', ['masy.other', 'case_device_c64', None])
def test_case_budget_requires_exact_producer_module(module):
    with pytest.raises(BedIntegrationError, match='C64_case_exact_producer'):
        replace(_case_budget(), producer_module=module).validated()


def test_case_budget_refuses_factory_hash_or_wrong_source_basename():
    budget = _case_budget()
    for source in (_ref('masy/case_device_c64.py', sha=FACTORY_SHA256),
                   _ref('test_windowed_shared_device_c64_qualification.py', sha='d' * 64),
                   _ref('other.py', sha='d' * 64)):
        with pytest.raises(BedIntegrationError, match='C64_case_not_synthetic_factory|C64_case_exact_producer'):
            replace(budget, producer_source=source).validated()


@pytest.mark.parametrize('name,value', [
    ('native_source_identity', True), ('source_manifest_sha256', 'C' * 64),
    ('case_request_binding_sha256', 'short'), ('gpu_uuid', 'GPU-unknown'),
    ('gpu_uuid', 'GPU-00000000-0000-0000-0000-00000000000G'),
    ('producer_source', _ref('masy/case_device_c64.py', size=True)),
    ('current_runtime_ref', _ref('runtime.json', sha='A' * 64)),
])
def test_case_budget_requires_plain_valid_reference_identity_and_uuid(name, value):
    with pytest.raises(BedIntegrationError):
        replace(_case_budget(), **{name: value}).validated()


def test_case_budget_requires_complete_ordered_bounded_source_roster():
    budget = _case_budget()
    rows = budget.source_bindings
    for bad in (rows[:-1], tuple(reversed(rows)), (rows[0],) * len(rows), list(rows),
                ((rows[0][0], object()), *rows[1:]),
                tuple((name, replace(ref, size=2 << 20)) for name, ref in rows)):
        with pytest.raises(BedIntegrationError):
            replace(budget, source_bindings=bad).validated()


def test_old_new_budget_kinds_and_synthetic_reserve_stay_separate():
    case, old = _case_budget(), _old_budget()
    with pytest.raises(BedIntegrationError, match='exact_holder'):
        d.c64_qualification_budget_bytes(case)
    with pytest.raises(BedIntegrationError, match='exact_holder'):
        d.c64_case_budget_bytes(old)
    # The synthetic reserve refuses the new descriptor before any Native
    # import/query/physical or device operation. This does not admit a case.
    with pytest.raises(BedIntegrationError, match='exact_holder'):
        d.reserve_c64_qualification_device_water_oracle_frontier(case, device_id=0)
    class Foreign(d.DeviceWaterC64CaseBudget):
        pass
    for foreign in (object(), Foreign(**vars(case))):
        with pytest.raises(BedIntegrationError, match='C64_budget_kind'):
            d._c64_budget(foreign)
    with pytest.raises(BedIntegrationError, match='exact_holder'):
        Foreign(**vars(case)).validated()


def test_case_descriptor_rejects_unpriced_extra_field():
    case = _case_budget()
    object.__setattr__(case, 'external_bounds', (('query', 1),))
    with pytest.raises(BedIntegrationError, match='closed_holder'):
        case.validated()


def test_case_metadata_preflight_uses_genuine_adapter_code_currentness(monkeypatch):
    case = _case_budget()
    # verify_files=False checks metadata/code only and performs no file or
    # Native/device preflight. Its return is not authority for physical work.
    assert d._c64_case_budget_runtime(case, verify_files=False) is case
    original = d.c64_case_budget_record
    with monkeypatch.context() as patch:
        patch.setattr(d, 'c64_case_budget_record', lambda _budget: {})
        with pytest.raises(BedIntegrationError, match='adapter_function_or_method_rebound'):
            d._c64_case_budget_runtime(case, verify_files=False)
    assert d.c64_case_budget_record is original
    assert d._c64_case_budget_runtime(case, verify_files=False) is case


def test_case_file_preflight_binds_complete_grant_and_actual_producer_origin(tmp_path, monkeypatch):
    """Real file/Source joins refuse in order; synthetic metadata never admits a run."""
    import sys
    import types
    from pathlib import Path

    def frozen(name, raw):
        path = tmp_path / name
        path.write_bytes(raw)
        path.chmod(0o444)
        return d.SourceReference(str(path), len(raw), hashlib.sha256(raw).hexdigest())

    base = replace(_case_budget(), current_runtime_ref=frozen('runtime.json', b'{}'),
                   producer_source=frozen('case_device_c64.py', b'# synthetic Source metadata only\n'))
    stale = d.c64_case_budget_bytes(replace(base, case_request_binding_sha256='f' * 64))
    with pytest.raises(BedIntegrationError, match='C64_case_complete_grant_value_join'):
        d._c64_case_budget_runtime(replace(base, grant=frozen('stale.json', stale)), verify_files=True)
    case = replace(base, grant=frozen('grant.json', d.c64_case_budget_bytes(base)))
    # No adapter function/check is replaced. Only the explicit synthetic
    # producer module metadata is installed, with automatic complete teardown.
    with monkeypatch.context() as patch:
        patch.delitem(sys.modules, case.producer_module, raising=False)
        with pytest.raises(BedIntegrationError, match='C64_case_loaded_producer_module'):
            d._c64_case_budget_runtime(case, verify_files=True)
        producer = types.ModuleType(case.producer_module)
        producer.__file__ = str(tmp_path / 'runtime.json')
        patch.setitem(sys.modules, case.producer_module, producer)
        with pytest.raises(BedIntegrationError, match='C64_case_actual_loaded_producer_origin'):
            d._c64_case_budget_runtime(case, verify_files=True)
        producer.__file__ = case.producer_source.path
        # The FIRST orderedSource module really is loaded; this final guard
        # is an origin mismatch, not a missing-module refusal in disguise.
        first_name, first_ref = case.source_bindings[0]
        assert first_name == d.__name__ and sys.modules.get(first_name) is d
        assert Path(d.__file__).resolve() != Path(first_ref.path).resolve()
        with pytest.raises(BedIntegrationError, match='C64_actual_loaded_Source_origin'):
            d._c64_case_budget_runtime(case, verify_files=True)
