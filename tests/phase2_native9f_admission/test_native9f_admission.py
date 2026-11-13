"""E4k: `storage_admission` on the exact native 9f MAPLE package (CPU/host, genuine native windowed beds).

Independent controls: source identity of the planner modules, one explicit two-digest source guard, the SYRUP plan equals a
direct native planner call built from the native objects, native defaults (gather 256, journal cap), pickup/paired
options, admission before any mutation, refusal classes, unknown source refused before any native import, device guard
(pure numbers), and the plan bounding the real host allocations of a native exchange."""
from __future__ import annotations

import builtins
import hashlib
import os
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from compat_support import assert_exact_equal, host, physical_arrays

from maple_syrup import sediment_bed as sb
from maple_syrup import storage_admission as sa
from maple_syrup.storage_admission import DemandFeatures, StorageAdmissionError

BIG = 10**18
PIN9F = "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"
# sha256 of the planner/deep-store modules: equal in MAPLE 39bcf79e and 9f466d2c (audited read-only, see the doc)
SOURCE_HASHES = {
    "maple.surface.voxels.windowed_admission": "d2c87b863f0c62aa8809c790c36e24273f193bc78478bc3ccd59b05f55b26bc1",
    "maple.surface.voxels.deep_store": "b03d1503fd20331432de9e225e604e8049d6b71a69d992aff59e739b5d5b3a5a",
}


def _demand(bed, **extra):
    from maple.water import WaterProcessDemand

    request = 0.05 * host(bed.active_layer.mass_kg)
    return WaterProcessDemand(request, np.zeros_like(request), **extra)


def test_runtime_is_native9f_and_the_guard_is_exactly_the_two_qualified_packages():
    import maple

    expected = os.environ.get("MAPLE_SYRUP_EXPECTED_MAPLE_ROOT")
    if expected:  # a configured wrong source FAILS
        assert Path(expected).resolve() in Path(maple.__file__).resolve().parents
    assert sb.bound_maple_package_digest() == PIN9F == sb.NATIVE9F_MAPLE_DIGEST
    assert sa.ADMISSION_QUALIFIED_MAPLE_DIGESTS == frozenset({sb.MERGED_39BCF79_MAPLE_DIGEST, PIN9F})
    sa._require_source()


def test_planner_and_deep_store_modules_are_file_identical_to_the_39bcf79_sources():
    import importlib

    for name, digest in SOURCE_HASHES.items():
        module = importlib.import_module(name)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == digest, name


def test_unqualified_package_is_refused_before_any_native_planner_import_or_reading(windowed_bed, monkeypatch):
    bed, ctx = windowed_bed
    seen, called = [], []
    real_import = builtins.__import__

    def recorder(name, *args, **kwargs):
        seen.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sb._MAPLE_DIGEST_CACHE, "digest", "0" * 64)
    monkeypatch.delitem(sys.modules, "maple.surface.voxels.windowed_admission", raising=False)
    monkeypatch.setattr(builtins, "__import__", recorder)
    try:
        for call in (lambda: sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=0,
                                             host_available_bytes_fn=lambda: called.append(1) or BIG),
                     lambda: sa.admit_paired(bed, ctx, features=DemandFeatures(), reserve_host_bytes=0,
                                             host_available_bytes_fn=lambda: called.append(1) or BIG),
                     lambda: sa.plan_for_bed(bed, ctx, path="public", features=DemandFeatures())):
            with pytest.raises(StorageAdmissionError, match="merged MAPLE package"):
                call()
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
    assert not any(n.startswith("maple.surface.voxels.windowed_admission") for n in seen)
    assert called == []
    # one digest off the pair is still refused: no prefix/range matching
    for digest in (PIN9F[:-1] + "0", PIN9F.upper(), ""):
        monkeypatch.setitem(sb._MAPLE_DIGEST_CACHE, "digest", digest)
        with pytest.raises(StorageAdmissionError):
            sa._require_source()
    monkeypatch.setitem(sb._MAPLE_DIGEST_CACHE, "digest", sb.MERGED_39BCF79_MAPLE_DIGEST)
    sa._require_source()  # the historical pin is unchanged


@pytest.mark.parametrize("path", ["public", "paired"])
def test_syrup_plan_equals_a_direct_native_planner_call_from_native_objects(windowed_bed, path):
    from maple.surface.voxels.deep_store import DEFAULT_MAX_JOURNAL_RECORDS
    from maple.surface.voxels.windowed_admission import windowed_transaction_bytes
    from maple.surface.voxels.windowed_transfer import DEFAULT_EXCHANGE_GATHER_COLUMNS

    bed, ctx = windowed_bed
    col, store = bed.voxel_column, bed.voxel_column.deep_store
    assert DEFAULT_EXCHANGE_GATHER_COLUMNS == 256 and ctx.window_max_journal_records is None
    flags = DemandFeatures(has_boundary_input=True, has_face_flux=True)
    direct = windowed_transaction_bytes(
        ny=bed.water.depth_m.shape[0], nx=bed.water.depth_m.shape[1], n_classes=bed.water.mobile_mass_by_cell_class_kg.shape[-1],
        n_voxels=col.n_voxels, window_voxels=col.window_voxels, packet_terms=col.height_packet.packet_terms,
        base_itemsize=col.base.dtype.itemsize, ledger_present_processes=len(bed.ledger.present_processes),
        has_availability=True, has_boundary_input=True, has_boundary_export=False, has_face_flux=True,
        max_gather_columns=256, max_journal_records=DEFAULT_MAX_JOURNAL_RECORDS, pending_records=0,
        chunk_bytes_max=store.header.chunk_bytes_max, overlay_bytes_max=0, mapping_cache_chunks=store.mapping_cache_chunks,
        namespace="numpy", path=path)
    mine = sa.plan_for_bed(bed, ctx, path=path, features=flags)
    for field in ("new_host_bytes_peak", "new_device_bytes_peak", "peak_stage_host", "mapped_bytes_max", "stages", "terms",
                  "unbounded_terms", "excluded"):
        assert getattr(mine, field) == getattr(direct, field), field
    assert mine.new_device_bytes_peak == 0 and mine.namespace == "numpy"


def test_paired_plan_covers_the_allocating_fallback_and_enlarges_with_flags(windowed_bed):
    bed, ctx = windowed_bed
    public = sa.plan_for_bed(bed, ctx, path="public", features=DemandFeatures())
    paired = sa.plan_for_bed(bed, ctx, path="paired", features=DemandFeatures())
    flagged = sa.plan_for_bed(bed, ctx, path="paired", features=DemandFeatures(True, True, True))
    assert paired.new_host_bytes_peak >= public.new_host_bytes_peak
    assert flagged.new_host_bytes_peak > paired.new_host_bytes_peak


def test_non_default_gather_and_context_journal_cap_follow_the_adapter_contract(windowed_bed):
    import dataclasses

    bed, ctx = windowed_bed
    for bad in (1, 255, 257, 512, True, 256.0, "256"):
        with pytest.raises(StorageAdmissionError):
            sa.plan_for_bed(bed, ctx, path="public", features=DemandFeatures(), max_gather_columns=bad)
    assert sa.plan_for_bed(bed, ctx, path="public", features=DemandFeatures(),
                           max_gather_columns=np.int64(256)).inputs["max_gather_columns"] == 256
    capped = dataclasses.replace(ctx, window_max_journal_records=77)
    record = sa.admit_public(bed, capped, _demand(bed), reserve_host_bytes=0, host_available_bytes_fn=lambda: BIG)
    assert record.max_journal_records == 77 == sb.resolved_journal_cap(capped)


@pytest.mark.parametrize("entry", ["public", "paired"])
def test_admission_precedes_and_never_mutates_and_refusal_is_memory_error(windowed_bed, entry):
    bed, ctx = windowed_bed
    store_dir = bed.voxel_column.deep_store.store_dir
    files_before = sorted(p.name for p in Path(store_dir).rglob("*"))
    before = physical_arrays(bed)
    kwargs = {"reserve_host_bytes": 123, "host_available_bytes_fn": lambda: BIG}
    admit = (lambda **k: sa.admit_public(bed, ctx, _demand(bed), **k)) if entry == "public" else (
        lambda **k: sa.admit_paired(bed, ctx, features=DemandFeatures(), **k))
    record = admit(**kwargs)
    required = record.decision.host_required_bytes
    assert record.reserve_host_bytes == 123 and record.decision.admitted
    with pytest.raises(MemoryError, match="insufficient host memory"):
        admit(reserve_host_bytes=123, host_available_bytes_fn=lambda: 2 * required - 1)
    assert admit(reserve_host_bytes=123, host_available_bytes_fn=lambda: 2 * required).decision.admitted
    with pytest.raises(TypeError):
        admit(host_available_bytes_fn=lambda: BIG)  # the reserve is never defaulted
    assert_exact_equal(physical_arrays(bed), before, "admission mutated the bed")
    assert sorted(p.name for p in Path(store_dir).rglob("*")) == files_before


def test_host_bed_refuses_device_inputs_before_any_query_and_device_guard_is_numeric(windowed_bed):
    bed, ctx = windowed_bed
    called = []
    with pytest.raises(StorageAdmissionError):
        sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=0, reserve_device_bytes=5,
                        host_available_bytes_fn=lambda: called.append(1) or BIG)
    with pytest.raises(StorageAdmissionError):
        sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=0, host_available_bytes_fn=lambda: called.append(1) or BIG,
                        device_available_bytes_fn=lambda: called.append(2) or BIG)
    assert called == [] and not any(m.startswith("cupy") for m in sys.modules)
    sa.require_current_device(0, {"a": 0})
    with pytest.raises(StorageAdmissionError, match="current device"):
        sa.require_current_device(0, {"a": 1})
    assert sa.effective_free_device_bytes(100, 20, 90, 30) == 60


def test_plan_bounds_the_real_host_allocations_of_native_public_and_paired_exchange(windowed_bed):
    """The planned NEW host bytes (an upper estimate, excluding Python overhead) bound the traced NumPy peak of the REAL
    native pending-journal exchange on this bed."""
    from compat_support import make_transport

    from maple_syrup.pond_bed import pickup_transport_deposit

    bed, ctx = windowed_bed
    demand = _demand(bed)
    public = sa.admit_public(bed, ctx, demand, reserve_host_bytes=0, host_available_bytes_fn=lambda: BIG)
    paired = sa.admit_paired(bed, ctx, features=DemandFeatures(), reserve_host_bytes=0, host_available_bytes_fn=lambda: BIG)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        _new_bed, _ = sb.apply_bed_demand(bed, ctx, demand)
        public_peak = tracemalloc.get_traced_memory()[1] - base
        pool0 = host(bed.water.mobile_mass_by_cell_class_kg).copy()
        transport, _seen = make_transport(pool0, 1.0)
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        pickup_transport_deposit(bed, ctx, bed.water.depth_m, 0.05 * host(bed.active_layer.mass_kg), transport)
        paired_peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    assert 0 < public_peak <= public.plan.new_host_bytes_peak
    assert 0 < paired_peak <= paired.plan.new_host_bytes_peak
