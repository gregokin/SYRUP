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
