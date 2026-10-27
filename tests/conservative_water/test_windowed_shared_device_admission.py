"""Affected metadata/refusal controls only; Root executes, author did not.

Synthetic unit=1 records below exercise scalar schema arithmetic. They are
explicitly NOT runtime/ABI certificates and cannot authorize a physical call.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import threading

import pytest

from maple_syrup import windowed_shared_bed as cpu
from maple_syrup import windowed_shared_bed_device as device
from maple_syrup.sediment_bed import BedContext, BedIntegrationError, BedState


def _limits(tmp_path):
    return device.PositiveWindowedDeviceWaterLimits(
        max_host_bytes=5 << 30, max_device_bytes=2 << 30, max_mapped_bytes=128 << 20,
        max_logical_read_bytes=(1 << 63)-1, max_work=(1 << 63)-1,
        max_store_bytes=64 << 20, max_run_bytes=512 << 20,
        max_new_versions=1000000, max_entries=4096, max_external_backing_bytes=64 << 20,
        max_event_steps=10000, max_retries=10, max_transport_substeps=64, max_report_rows=4096,
        reserve_host_bytes=64 << 20, reserve_device_bytes=256 << 20,
        safety_fraction=0.5, run_dir=tmp_path)


def _context(tmp_path, bounds=None):
    # Exact new context type with deliberately unusable physical operands.
    # No Native owner/array is constructed or claimed by these unit controls.
    return device.WindowedDeviceWaterContext(
        geometry=None, grain_classes=None, mass_resolution_kg=1.0,
        topographic_commit_spec=None, avalanching_spec=None, initial_available_fraction=None,
        native_solve_limits=None, native_validation_limits=None, native_commit_limits=None,
        source_identity='0'*64, bridge_limits=_limits(tmp_path), runtime_bounds=bounds)


def _reference():
    return device.SourceReference('/UNISSUED_SOURCE_ONLY_METADATA', 1, '0'*64)


def _bounds():
    ref = _reference()
    return device.DeviceWaterRuntimeBounds('0'*64, ref, ref, ref,
        tuple((name, 1) for name in device._UNIT_NAMES),
        tuple(device.ExternalRuntimeBound(role, 1, 1, 1, 1, 1, 1, 1, ref)
              for role in device._EXTERNAL_ROLES),
        tuple((name, ref) for name in device._SOURCE_MODULES))


def test_device_limits_are_independent_and_keep_original_ceilings(tmp_path):
    limits = _limits(tmp_path)
    assert not isinstance(limits, cpu.WindowedWaterBridgeLimits)
    assert not isinstance(limits, cpu.PositiveWindowedWaterBridgeLimits)
    for name, wrong in (('max_device_bytes', 3 << 30), ('max_host_bytes', 6 << 30),
                        ('max_event_steps', 9999), ('safety_fraction', 1.0)):
        with pytest.raises(BedIntegrationError):
            dataclasses.replace(limits, **{name: wrong})


def test_unissued_runtime_refuses_before_native_profile(tmp_path, monkeypatch):
    def unexpected():
        raise AssertionError('Native metadata must not run on an absent runtime contract')
    monkeypatch.setattr(device, '_native', unexpected)
    with pytest.raises(BedIntegrationError, match='unissued_runtime_ABI_provider_contract'):
        device._runtime_bounds(_context(tmp_path))


def test_all_external_roles_are_positive_and_closed():
    bounds = _bounds()
    assert bounds.validated() is bounds  # Shape only: the file does not exist.
    for name in ('host_bytes', 'device_bytes', 'work', 'logical_read_bytes', 'h2d_bytes', 'd2h_bytes', 'scalar_reads'):
        bad = dataclasses.replace(bounds.external_bounds[0], **{name: 0})
        with pytest.raises(BedIntegrationError):
            bad.validated()
    with pytest.raises(BedIntegrationError):
        dataclasses.replace(bounds, external_bounds=bounds.external_bounds[:-1]).validated()


def test_primitive_domain_rejects_missing_extra_bool_and_unknown():
    bounds = _bounds()
    for rows in (bounds.primitive_units[:-1], bounds.primitive_units+(('unknown', 1),),
                 ((bounds.primitive_units[0][0], True),)+bounds.primitive_units[1:]):
        with pytest.raises(BedIntegrationError):
            dataclasses.replace(bounds, primitive_units=rows).validated()


def test_source_roster_rejects_extra_duplicate_and_omitted():
    bounds = _bounds()
    for rows in (bounds.source_bindings[:-1], bounds.source_bindings+(('unknown', _reference()),),
                 (bounds.source_bindings[0],)*len(bounds.source_bindings)):
        with pytest.raises(BedIntegrationError):
            dataclasses.replace(bounds, source_bindings=rows).validated()


def test_reference_shape_is_not_evidence_or_a_path_fallback(tmp_path):
    for ref in (device.SourceReference('relative', 1, '0'*64),
                device.SourceReference('/bad\x00path', 1, '0'*64),
                device.SourceReference('/x', True, '0'*64),
                device.SourceReference('/x', 1, 'g'*64)):
        with pytest.raises(BedIntegrationError):
            ref.validated()
    # A valid shape still must pass genuine immutable bytes/currentness.
    p = tmp_path/'metadata'; p.write_bytes(b'x')
    with pytest.raises(BedIntegrationError, match='immutable_reference_file'):
        device._read_reference(device.SourceReference(str(p), 1, '0'*64))
    p.chmod(0o444)
    assert device._read_reference(device.SourceReference(str(p), 1, hashlib.sha256(b'x').hexdigest())) == b'x'
    link = tmp_path/'metadata-link'
    link.symlink_to(p)
    with pytest.raises(BedIntegrationError, match='immutable_reference_file'):
        device._read_reference(device.SourceReference(str(link), 1, hashlib.sha256(b'x').hexdigest()))


def test_constructor_projection_preserves_every_positive_component():
    bounds = _bounds()
    base = device._constructor_frontier(bounds, rows=2, arrays=3, text=5, frames=7, errors=6)
    assert base > 0
    for key in ('rows', 'arrays', 'text', 'frames', 'errors'):
        args = dict(rows=2, arrays=3, text=5, frames=7, errors=6)
        args[key] += 1
        assert device._constructor_frontier(bounds, **args) > base
    for value in (True, -1, 1 << 63):
        with pytest.raises(BedIntegrationError):
            device._integer(value, 'planned_counter')


def test_inactive_ordinary_pair_does_not_select_or_import_new_lane(tmp_path):
    bed = BedState(None, None, None, None, None, None)
    context = BedContext(None, None, 1.0, None, None, None)
    assert device._OWNER is None
    assert device._positive_pair_guard(bed, context) is None
    assert device.is_shared_pair(bed, context) is False
    assert cpu._positive_pair_guard(bed, context) is None
    assert not isinstance(_context(tmp_path).bridge_limits, cpu.WindowedWaterBridgeLimits)


def _synthetic_guard_owner(context):
    # ONLY the guard's metadata fields; it cannot pass device._owner or a
    # Native action. No Native authority is fabricated by this negative test.
    owner = object.__new__(device._Owner)
    owner.context, owner.pid, owner.thread = context, os.getpid(), threading.get_ident()
    owner.poisoned = owner.closed = False
    return owner


def test_active_device_guard_excludes_cpu_and_foreign_context(tmp_path, monkeypatch):
    context = _context(tmp_path)
    owner = _synthetic_guard_owner(context)
    monkeypatch.setattr(device, '_OWNER', owner)
    bed = cpu.WindowedSharedBedState(None, None, None, None, None, None, None)
    with pytest.raises(BedIntegrationError, match='exact_active_process_thread_pair'):
        cpu._positive_pair_guard(bed, BedContext(None, None, 1.0, None, None, None))
    assert owner.poisoned is True


def test_active_device_guard_excludes_wrong_thread_before_any_action(tmp_path, monkeypatch):
    context = _context(tmp_path)
    owner = _synthetic_guard_owner(context)
    owner.thread += 1
    monkeypatch.setattr(device, '_OWNER', owner)
    bed = cpu.WindowedSharedBedState(None, None, None, None, None, None, None)
    with pytest.raises(BedIntegrationError, match='exact_active_process_thread_pair'):
        device._positive_pair_guard(bed, context)
    assert owner.poisoned is True


def test_subclass_context_and_unknown_phase_refuse(tmp_path):
    class Foreign(device.WindowedDeviceWaterContext):
        pass
    context = _context(tmp_path)
    foreign = Foreign(**vars(context))
    bed = cpu.WindowedSharedBedState(None, None, None, None, None, None, None)
    with pytest.raises(BedIntegrationError):
        device.is_shared_pair(bed, foreign)
    with pytest.raises(BedIntegrationError, match='phase_domain'):
        device.produce_device_water_phase(None, context, 'unknown')


def test_native_public_plan_is_allocating_and_requires_complete_scratch():
    # Genuine metadata-only Native planner, not an observed model result.
    from maple.surface.voxels.windowed_admission import windowed_transaction_bytes
    from maple.surface.voxels.windowed_prefix import packet_terms_for
    from maple.core.types.sediment_ledger import N_LEDGER_PROCESSES
    plan = windowed_transaction_bytes(ny=4, nx=4, n_classes=1, n_voxels=6, window_voxels=3,
        packet_terms=packet_terms_for(6, 1), base_itemsize=4, ledger_present_processes=N_LEDGER_PROCESSES,
        has_availability=True, has_boundary_input=True, has_boundary_export=True, has_face_flux=True,
        max_gather_columns=4, max_journal_records=320, pending_records=320,
        chunk_bytes_max=16*6*8, overlay_bytes_max=0, mapping_cache_chunks=1,
        namespace='cupy', path='public')
    assert plan.path == 'public' and plan.namespace == 'cupy'
    assert plan.advisory_owned_host_bytes_peak is None and plan.advisory_owned_device_bytes_peak is None
    assert plan.new_host_bytes_peak > 0 and plan.new_device_bytes_peak > 0
    assert plan.admissible and not plan.unbounded_terms


def test_complete_issued_contract_encoding_joins_values_and_is_acyclic():
    bounds = _bounds()
    raw = device.runtime_contract_bytes(bounds)
    assert b'current_runtime_ref' in raw and b'external_bounds' in raw and b'source_bindings' in raw
    assert device.runtime_contract_bytes(dataclasses.replace(bounds,
        runtime_evidence=device.SourceReference('/OTHER_CONTAINING_FILE', 5, '1'*64))) == raw
    changed = dataclasses.replace(bounds, primitive_units=((bounds.primitive_units[0][0], 2),)+bounds.primitive_units[1:])
    assert device.runtime_contract_bytes(changed) != raw
    external = dataclasses.replace(bounds.external_bounds[0], work=2)
    changed = dataclasses.replace(bounds, external_bounds=(external,)+bounds.external_bounds[1:])
    assert device.runtime_contract_bytes(changed) != raw


def _constructor_v2_bank(name, **counts):
    values = {key: 0 for key in device._CONSTRUCTOR_V2_COUNTS}
    assert set(counts) <= set(values)
    values.update(counts)
    return (name, *(values[key] for key in device._CONSTRUCTOR_V2_COUNTS))


def _constructor_v2(**changes):
    # Synthetic SCHEMA values only. Shell2 means one positive placeholder plus
    # one cache terminator, slope8 includes both character storage families.
    # None of these values is a genuine selected ABI or runtime policy.
    values = dict(schema=device._CONSTRUCTOR_V2_SCHEMA, source_census_sha256='a'*64,
        domain=device._CONSTRUCTOR_V2_DOMAIN, unicode_shell_bytes=2, unicode_character_bytes=8,
        bytes_shell_bytes=1, bytes_payload_bytes=1, list_header_bytes=1, list_slot_bytes=1,
        tuple_header_bytes=1, tuple_slot_bytes=1, dict_header_bytes=1, dict_entry_bytes=1,
        set_header_bytes=1, set_entry_bytes=1,
        banks=tuple(_constructor_v2_bank(name) for name in device._CONSTRUCTOR_V2_BANK_NAMES))
    values.update(changes)
    return device.DeviceWaterConstructorAccountingV2(**values)


def test_v2_none_preserves_complete_v1_encoding_and_constructor_dispatch():
    import json
    bounds = _bounds()
    assert bounds.constructor_accounting_v2 is None
    def ref(value):
        return dict(path=value.path, size=value.size, sha256=value.sha256)
    # Independent original6c schema: the new optional field must be absent.
    expected = dict(schema='maple_syrup.windowed_device_water.runtime_contract.v1',
        native_source_identity=bounds.native_source_identity,
        current_runtime_ref=ref(bounds.current_runtime_ref), primitive_evidence=ref(bounds.primitive_evidence),
        primitive_units=bounds.primitive_units,
        external_bounds=tuple(dict(role=b.role, host_bytes=b.host_bytes, device_bytes=b.device_bytes,
            work=b.work, logical_read_bytes=b.logical_read_bytes, h2d_bytes=b.h2d_bytes,
            d2h_bytes=b.d2h_bytes, scalar_reads=b.scalar_reads, evidence=ref(b.evidence)) for b in bounds.external_bounds),
        source_bindings=tuple(dict(module=name, **ref(r)) for name, r in bounds.source_bindings))
    assert device.runtime_contract_bytes(bounds) == json.dumps(expected, sort_keys=True,
        separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')
    args = dict(rows=23, arrays=7, text=311, frames=101, errors=6)
    for name in device._CONSTRUCTOR_V2_BANK_NAMES:
        assert device._constructor_projection(bounds, name, **args) == device._constructor_frontier(bounds, **args)


def test_v2_schema_domain_banks_and_utf8_units_are_closed():
    profile = _constructor_v2()
    assert profile.validated() is profile
    assert '__repr__' not in vars(type(profile))  # No second hidden repr closure.
    bad = [dict(schema='v1'), dict(source_census_sha256='G'*64),
        dict(domain=list(profile.domain)), dict(domain=(True,)+profile.domain[1:]),
        dict(domain=(17,)+profile.domain[1:]), dict(unicode_shell_bytes=1),
        dict(unicode_character_bytes=7), dict(banks=profile.banks[:-1]),
        dict(banks=(profile.banks[0],)*4)]
    for change in bad:
        with pytest.raises(BedIntegrationError):
            dataclasses.replace(profile, **change).validated()
    for name in device._CONSTRUCTOR_V2_UNITS:
        for value in (True, 0, 1 << 31):
            with pytest.raises(BedIntegrationError):
                dataclasses.replace(profile, **{name: value}).validated()
    for name in ('call_rows_max', 'call_text_max', 'unicode_objects'):
        for count in (True, -1, 1 << 63):
            row = _constructor_v2_bank('reference_bootstrap', **{name: count})
            with pytest.raises(BedIntegrationError):
                dataclasses.replace(profile, banks=(row,)+profile.banks[1:]).validated()
    row = _constructor_v2_bank('reference_bootstrap', unicode_characters=1)
    with pytest.raises(BedIntegrationError, match='missing_typed_owner'):
        dataclasses.replace(profile, banks=(row,)+profile.banks[1:]).validated()
    for counts in (dict(unrefined_rows=1), dict(remaining_v1_text_characters=1)):
        row = _constructor_v2_bank('reference_bootstrap', **counts)
        with pytest.raises(BedIntegrationError, match='residual_exceeds_call_max'):
            dataclasses.replace(profile, banks=(row,)+profile.banks[1:]).validated()


def test_v2_encoder_binds_every_field_and_remains_acyclic():
    import json
    profile = _constructor_v2()
    bounds = dataclasses.replace(_bounds(), constructor_accounting_v2=profile)
    raw = device.runtime_contract_bytes(bounds)
    decoded = json.loads(raw)
    assert decoded['schema'] == 'maple_syrup.windowed_device_water.runtime_contract.v2'
    assert set(decoded['constructor_accounting_v2']) == {f.name for f in dataclasses.fields(profile)}
    assert decoded['constructor_accounting_v2']['banks'] == [list(row) for row in profile.banks]
    for name in device._CONSTRUCTOR_V2_UNITS:
        changed = dataclasses.replace(profile, **{name: getattr(profile, name)+1})
        assert device.runtime_contract_bytes(dataclasses.replace(bounds, constructor_accounting_v2=changed)) != raw
    for changed in (dataclasses.replace(profile, source_census_sha256='b'*64),
                    *(dataclasses.replace(profile, banks=(_constructor_v2_bank('reference_bootstrap',
                        **counts),)+profile.banks[1:]) for counts in (
                            dict(unicode_objects=1, unicode_characters=1),
                            dict(call_rows_max=1), dict(call_text_max=1)))):
        assert device.runtime_contract_bytes(dataclasses.replace(bounds, constructor_accounting_v2=changed)) != raw
    assert device.runtime_contract_bytes(dataclasses.replace(bounds,
        runtime_evidence=device.SourceReference('/DIFFERENT_CONTAINER', 777, 'c'*64))) == raw


def test_v2_degenerate_identity_at_all_four_source_call_arguments():
    bounds = _bounds()
    refs = (bounds.runtime_evidence, bounds.current_runtime_ref, bounds.primitive_evidence,
        *(b.evidence for b in bounds.external_bounds), *(r for _n, r in bounds.source_bindings))
    raw, source = sum(r.size for r in refs), sum(r.size for _n, r in bounds.source_bindings)
    assert len(refs) == 36
    # Real pure code census; no Native plans or provider operations. Phase's
    # additional immediate-operator slots may take any genuine nonnegative
    # value: this algebraic identity holds for the same actual frames argument.
    _codes, own_slots, own_text = device._code_frontier()
    cases = ((25600, 512, 4*raw+36*4*4096+16384*2048+2305),
             (16384, 1024, source+4*4096*4096),
             (4580, 512, source+own_text+5*4096*4),
             (16512, 576, 16384*2048+2305+4*4096+256*128))
    banks = tuple(_constructor_v2_bank(name, call_rows_max=rows, call_text_max=text,
                                      unrefined_rows=rows, remaining_v1_text_characters=text)
                  for name, (rows, _arrays, text) in zip(device._CONSTRUCTOR_V2_BANK_NAMES, cases))
    bounds = dataclasses.replace(bounds, constructor_accounting_v2=_constructor_v2(banks=banks))
    for name, (rows, arrays, text) in zip(device._CONSTRUCTOR_V2_BANK_NAMES, cases):
        args = dict(rows=rows, arrays=arrays, text=text, frames=90*own_slots, errors=6)
        assert device._constructor_projection(bounds, name, **args) == device._constructor_frontier(bounds, **args)
        with pytest.raises(BedIntegrationError, match='current_call_caps'):
            device._constructor_projection(bounds, name, **{**args, 'rows': rows-1})
        with pytest.raises(BedIntegrationError, match='current_call_caps'):
            device._constructor_projection(bounds, name, **{**args, 'text': text-1})
        with pytest.raises(BedIntegrationError, match='current_call_caps'):
            device._constructor_projection(bounds, name, **{**args, 'rows': rows+1})
        with pytest.raises(BedIntegrationError, match='current_call_caps'):
            device._constructor_projection(bounds, name, **{**args, 'text': text+1})


def test_v2_typed_costs_remain_positive_and_distinct():
    counts = {name: 1 for name in device._CONSTRUCTOR_V2_COUNTS}
    counts.update(call_rows_max=2, call_text_max=5)
    bank = _constructor_v2_bank('reference_bootstrap', **counts)
    profile = _constructor_v2(banks=(bank,)+_constructor_v2().banks[1:])
    bounds = dataclasses.replace(_bounds(), constructor_accounting_v2=profile)
    args = dict(rows=2, arrays=3, text=5, frames=7, errors=6)
    base = device._constructor_frontier_v2(bounds, 'reference_bootstrap', **args)
    for name in device._CONSTRUCTOR_V2_UNITS:
        raised = dataclasses.replace(profile, **{name: getattr(profile, name)+1})
        value = device._constructor_frontier_v2(dataclasses.replace(bounds, constructor_accounting_v2=raised),
            'reference_bootstrap', **args)
        assert value == base+1
    for name in device._CONSTRUCTOR_V2_COUNTS[2:]:
        raised_counts = {**counts, name: 2}
        raised = dataclasses.replace(profile, banks=(_constructor_v2_bank('reference_bootstrap', **raised_counts),)+profile.banks[1:])
        assert device._constructor_frontier_v2(dataclasses.replace(bounds, constructor_accounting_v2=raised),
            'reference_bootstrap', **args) > base


def test_v2_current_call_types_bank_and_int63_overflow_refuse():
    profile = _constructor_v2()
    bounds = dataclasses.replace(_bounds(), constructor_accounting_v2=profile)
    args = dict(rows=2, arrays=3, text=5, frames=7, errors=6)
    for bank in ('unknown', True):
        with pytest.raises(BedIntegrationError, match='constructor_v2_bank'):
            device._constructor_frontier_v2(bounds, bank, **args)
    for name in args:
        for wrong in (True, -1, 1 << 63):
            with pytest.raises(BedIntegrationError):
                device._constructor_frontier_v2(bounds, 'reference_bootstrap', **{**args, name: wrong})
    huge = _constructor_v2_bank('reference_bootstrap', call_rows_max=2, call_text_max=5,
                               unicode_objects=(1 << 63)-1)
    bounds = dataclasses.replace(bounds, constructor_accounting_v2=dataclasses.replace(profile, banks=(huge,)+profile.banks[1:]))
    with pytest.raises(BedIntegrationError, match='constructor_v2_projection'):
        device._constructor_frontier_v2(bounds, 'reference_bootstrap', **args)


def test_v2_frozen_and_revalidated_metadata_mutations_refuse():
    profile = _constructor_v2()
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.unicode_character_bytes = 1
    bounds = dataclasses.replace(_bounds(), constructor_accounting_v2=profile)
    object.__setattr__(profile, 'unicode_character_bytes', 7)
    with pytest.raises(BedIntegrationError):
        device._constructor_frontier_v2(bounds, 'reference_bootstrap', rows=2, arrays=3, text=5, frames=7)


def test_v2_generated_repr_user_function_code_and_bindings_are_witnessed(monkeypatch):
    witness = device._RUNTIME_BOUNDS_REPR_WITNESS
    wrapped = device.DeviceWaterRuntimeBounds.__repr__
    assert witness[0] is wrapped and witness[5] is wrapped.__wrapped__
    assert any(fn is witness[5] and code is witness[6] for fn, code in device._CODE_BANK)
    for fn in (device._constructor_frontier_v2, device._constructor_projection,
               device.DeviceWaterConstructorAccountingV2.validated):
        assert any(owner is fn and code is fn.__code__ for owner, code in device._CODE_BANK)
    device._check_code()
    def substitute(_self):
        return 'synthetic replacement'
    monkeypatch.setattr(wrapped, '__wrapped__', substitute)
    with pytest.raises(BedIntegrationError, match='runtime_bounds_repr_closure_binding'):
        device._check_code()


def test_v2_generated_repr_cell_and_hidden_code_mutations_refuse(monkeypatch):
    witness = device._RUNTIME_BOUNDS_REPR_WITNESS
    def substitute(_self):
        return 'synthetic replacement'
    with monkeypatch.context() as patch:
        patch.setattr(witness[4], 'cell_contents', substitute)
        with pytest.raises(BedIntegrationError, match='runtime_bounds_repr_closure_binding'):
            device._check_code()
    with monkeypatch.context() as patch:
        patch.setattr(witness[5], '__code__', substitute.__code__)
        with pytest.raises(BedIntegrationError, match='adapter_code_identity_changed'):
            device._check_code()
    device._check_code()
