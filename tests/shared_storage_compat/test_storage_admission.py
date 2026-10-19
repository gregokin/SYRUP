"""`maple_syrup.storage_admission` (explicit driver-call helper) on genuine native windowed state.

CPU/NumPy only. Readings are injected numbers; the only synthetic object is a plain number-returning callable, never a MAPLE
type. The actual device path (CuPy bed, real pool) is exercised by root separately; here only its pure arithmetic
(`effective_free_device_bytes`) is pinned. A decision is an estimate against an injected reading, not a reservation. Not
executed by its author (file tools only)."""
from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest
from compat_support import assert_exact_equal, host, make_transport, physical_arrays

from maple_syrup import storage_admission as sa
from maple_syrup.pond_bed import pickup_transport_deposit
from maple_syrup.storage_admission import DemandFeatures, StorageAdmissionError

BIG = 10**18


def _demand(bed, **extra):
    from maple.water import WaterProcessDemand

    request = 0.05 * host(bed.active_layer.mass_kg)
    return WaterProcessDemand(request, np.zeros_like(request), **extra)


def test_public_admission_uses_actual_state_demand_flags_and_native_options(windowed_bed):
    from maple.water import WaterProcessDemand

    bed, ctx = windowed_bed
    before = physical_arrays(bed)
    record = sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=1000, host_available_bytes_fn=lambda: BIG)
    plan = record.plan
    assert record.path == "public" and record.namespace == "numpy" and record.reserve_device_bytes == 0
    assert record.max_gather_columns == 256 and record.max_journal_records == 1_000_000
    assert plan.inputs["ledger_present_processes"] == len(bed.ledger.present_processes) == 1
    assert plan.inputs["window_voxels"] == bed.voxel_column.window_voxels
    assert plan.inputs["packet_terms"] == bed.voxel_column.height_packet.packet_terms
    assert plan.inputs["base_itemsize"] == bed.voxel_column.base.dtype.itemsize
    assert plan.inputs["overlay_bytes_max"] == 0 and plan.inputs["pending_records"] == 0
    assert (plan.inputs["has_boundary_input"], plan.inputs["has_boundary_export"], plan.inputs["has_face_flux"]) == (
        False, False, False)
    assert_exact_equal(physical_arrays(bed), before, "admission mutated the bed")

    zeros = np.zeros_like(host(bed.active_layer.mass_kg))
    with_export = WaterProcessDemand(zeros, zeros, boundary_export_by_cell_class_kg=zeros)
    exported = sa.admit_public(bed, ctx, with_export, reserve_host_bytes=0, host_available_bytes_fn=lambda: BIG)
    assert exported.plan.inputs["has_boundary_export"] is True
    assert exported.plan.new_host_bytes_peak >= plan.new_host_bytes_peak


def test_exact_threshold_and_named_memory_refusal_before_any_exchange(windowed_bed):
    bed, ctx = windowed_bed
    demand = _demand(bed)
    peak = sa.plan_for_bed(bed, ctx, path="public", features=DemandFeatures()).new_host_bytes_peak
    reserve = 777
    required = peak + reserve
    ok = sa.admit_public(bed, ctx, demand, reserve_host_bytes=reserve, host_available_bytes_fn=lambda: 2 * required)
    assert ok.decision.admitted and ok.decision.host_margin_bytes == 0 and ok.decision.host_required_bytes == required
    with pytest.raises(MemoryError, match="insufficient host memory"):
        sa.admit_public(bed, ctx, demand, reserve_host_bytes=reserve, host_available_bytes_fn=lambda: 2 * required - 1)
    with pytest.raises(MemoryError, match="availability"):
        sa.admit_public(bed, ctx, demand, reserve_host_bytes=reserve, host_available_bytes_fn=lambda: None)
    # an explicit smaller safety fraction is honoured exactly
    with pytest.raises(MemoryError):
        sa.admit_public(bed, ctx, demand, reserve_host_bytes=reserve, safety_fraction=0.25,
                        host_available_bytes_fn=lambda: 2 * required)


def test_reserves_are_required_and_never_defaulted(windowed_bed):
    bed, ctx = windowed_bed
    with pytest.raises(TypeError):
        sa.admit_public(bed, ctx, _demand(bed), host_available_bytes_fn=lambda: BIG)
    with pytest.raises(TypeError):
        sa.admit_paired(bed, ctx, features=DemandFeatures(), host_available_bytes_fn=lambda: BIG)
    with pytest.raises(StorageAdmissionError, match="host bed"):
        sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=0, reserve_device_bytes=5,
                        host_available_bytes_fn=lambda: BIG)
    assert sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=0, reserve_device_bytes=0,
                           host_available_bytes_fn=lambda: BIG).reserve_device_bytes == 0
    with pytest.raises(Exception, match="reserve_host_bytes"):
        sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=-1, host_available_bytes_fn=lambda: BIG)


def test_paired_flags_must_be_declared_explicitly_and_enlarge_the_plan(windowed_bed):
    bed, ctx = windowed_bed
    with pytest.raises(TypeError):  # the keyword is required: nothing is inferred from a future callback
        sa.admit_paired(bed, ctx, reserve_host_bytes=0, host_available_bytes_fn=lambda: BIG)
    with pytest.raises(StorageAdmissionError, match="DemandFeatures"):
        sa.admit_paired(bed, ctx, features={"has_face_flux": True}, reserve_host_bytes=0,
                        host_available_bytes_fn=lambda: BIG)
    with pytest.raises(StorageAdmissionError, match="bool"):
        DemandFeatures(has_face_flux=1)
    plain = sa.admit_paired(bed, ctx, features=DemandFeatures(), reserve_host_bytes=0, host_available_bytes_fn=lambda: BIG)
    full = sa.admit_paired(bed, ctx, features=DemandFeatures(True, True, True), reserve_host_bytes=0,
                           host_available_bytes_fn=lambda: BIG)
    assert plain.path == full.path == "paired"
    assert (full.plan.inputs["has_boundary_input"], full.plan.inputs["has_boundary_export"],
            full.plan.inputs["has_face_flux"]) == (True, True, True)
    assert full.plan.new_host_bytes_peak >= plain.plan.new_host_bytes_peak
    assert plain.plan.advisory_owned_host_bytes_peak is not None  # paired plans carry the advisory figure (never admitted)


def test_cap_options_match_what_the_adapters_actually_execute(windowed_bed):
    """The adapters pass `max_journal_records` only when the context sets one and NEVER pass `max_gather_columns`
    (`sediment_bed.apply_bed_demand` / `apply_bed_pickup_transport_deposit`), so MAPLE's native 256 always runs."""
    from maple.surface.voxels.windowed_transfer import DEFAULT_EXCHANGE_GATHER_COLUMNS

    assert DEFAULT_EXCHANGE_GATHER_COLUMNS == 256
    bed, ctx = windowed_bed
    default = sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=0, host_available_bytes_fn=lambda: BIG)
    assert (default.max_journal_records, default.max_gather_columns) == (1_000_000, 256)
    capped = dataclasses.replace(ctx, window_max_journal_records=5000)
    record = sa.admit_public(bed, capped, _demand(bed), reserve_host_bytes=0, max_gather_columns=256,
                             host_available_bytes_fn=lambda: BIG)
    assert (record.max_journal_records, record.max_gather_columns) == (5000, 256)
    assert record.plan.inputs["max_journal_records"] == 5000 and record.plan.inputs["max_gather_columns"] == 256


@pytest.mark.parametrize("gather", [1, 64, 128, 255, 257, 1024, 0, 1025, True, 256.0, "256"])
@pytest.mark.parametrize("entry", ["public", "paired"])
def test_any_non_default_gather_option_is_refused_before_any_reading(windowed_bed, gather, entry):
    bed, ctx = windowed_bed
    readings = []

    def reading():
        readings.append(1)
        return BIG

    common = {"reserve_host_bytes": 0, "max_gather_columns": gather, "host_available_bytes_fn": reading}
    with pytest.raises(StorageAdmissionError, match="max_gather_columns"):
        if entry == "public":
            sa.admit_public(bed, ctx, _demand(bed), **common)
        else:
            sa.admit_paired(bed, ctx, features=DemandFeatures(), **common)
    assert readings == []


@pytest.mark.parametrize("name", ["reserve_host_bytes", "reserve_device_bytes"])
@pytest.mark.parametrize("bad", [False, True, 1.0, "1", -1, np.array(1), np.bool_(False)])
def test_reserves_are_validated_without_coercion(windowed_bed, name, bad):
    bed, ctx = windowed_bed
    readings = []
    args = {"reserve_host_bytes": 0, "host_available_bytes_fn": lambda: readings.append(1) or BIG}
    args[name] = bad
    with pytest.raises(StorageAdmissionError, match=name):
        sa.admit_public(bed, ctx, _demand(bed), **args)
    assert readings == []


def test_host_bed_accepts_numpy_integer_zero_and_none_for_the_device_reserve_only(windowed_bed):
    bed, ctx = windowed_bed
    for value in (None, 0, np.int64(0)):
        record = sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=np.int64(5), reserve_device_bytes=value,
                                 host_available_bytes_fn=lambda: BIG)
        assert record.reserve_device_bytes == 0 and record.reserve_host_bytes == 5
        assert type(record.reserve_host_bytes) is int and type(record.reserve_device_bytes) is int


@pytest.mark.parametrize("entry", ["public", "paired"])
def test_host_bed_refuses_a_device_query_function_before_any_query(windowed_bed, entry):
    bed, ctx = windowed_bed
    queries = []
    common = {"reserve_host_bytes": 0, "host_available_bytes_fn": lambda: queries.append("host") or BIG,
              "device_available_bytes_fn": lambda: queries.append("device") or 123}
    with pytest.raises(StorageAdmissionError, match="device_available_bytes_fn"):
        if entry == "public":
            sa.admit_public(bed, ctx, _demand(bed), **common)
        else:
            sa.admit_paired(bed, ctx, features=DemandFeatures(), **common)
    assert queries == []


def test_unsupported_source_is_refused_before_the_optional_native_module_is_imported(windowed_bed, monkeypatch):
    """An older tree has no `maple.surface.voxels.windowed_admission`: it must get the named refusal, not ModuleNotFoundError.
    Proven with an unsupported digest and an import recorder (the native module is dropped from `sys.modules` so that a
    premature import would be observed)."""
    import builtins

    from maple_syrup import sediment_bed as sb

    bed, ctx = windowed_bed
    native = "maple.surface.voxels.windowed_admission"
    seen = []
    real_import = builtins.__import__

    def recorder(name, *args, **kwargs):
        seen.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sb._MAPLE_DIGEST_CACHE, "digest", "0" * 64)
    monkeypatch.delitem(sys.modules, native, raising=False)
    monkeypatch.setattr(builtins, "__import__", recorder)
    try:
        with pytest.raises(StorageAdmissionError, match="merged MAPLE package"):
            sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=0, host_available_bytes_fn=lambda: BIG)
        with pytest.raises(StorageAdmissionError, match="merged MAPLE package"):
            sa.admit_paired(bed, ctx, features=DemandFeatures(), reserve_host_bytes=0)
        with pytest.raises(StorageAdmissionError, match="merged MAPLE package"):
            sa.plan_for_bed(bed, ctx, path="public", features=DemandFeatures())
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
    assert native not in seen and native not in sys.modules


def test_device_mismatch_is_refused_numerically_before_any_reading():
    """Pure device-id arithmetic of the default query's guard (a real CuPy bed is root's separate GPU check)."""
    sa.require_current_device(0, {"water.depth_m": 0, "voxel_column.window_kg": 0})
    with pytest.raises(StorageAdmissionError, match=r"water\.depth_m lives on device 1 .* current device is 0"):
        sa.require_current_device(0, {"water.depth_m": 1, "voxel_column.window_kg": 0})
    with pytest.raises(StorageAdmissionError, match=r"window_kg lives on device 2"):
        sa.require_current_device(0, {"water.depth_m": 0, "voxel_column.window_kg": 2})


def test_default_device_query_is_never_built_for_a_host_bed(windowed_bed):
    """A host bed with the default (None) device function never reaches the device query code or imports CuPy."""
    present = "cupy" in sys.modules
    bed, ctx = windowed_bed
    sa.admit_paired(bed, ctx, features=DemandFeatures(), reserve_host_bytes=0, host_available_bytes_fn=lambda: BIG)
    if not present:
        assert "cupy" not in sys.modules


def test_actual_pending_count_is_planned(windowed_bed):
    bed, ctx = windowed_bed
    request = 0.05 * host(bed.active_layer.mass_kg)
    pool0 = host(bed.water.mobile_mass_by_cell_class_kg).copy()
    pending = pickup_transport_deposit(bed, ctx, bed.water.depth_m, request, make_transport(pool0, 1.0)[0])[0]
    n = pending.voxel_column.pending_deep.journal.n_records
    assert n > 0
    plan = sa.plan_for_bed(pending, ctx, path="public", features=DemandFeatures())
    assert plan.inputs["pending_records"] == n
    tight = dataclasses.replace(ctx, window_max_journal_records=n - 1)  # a carried journal over the cap is refused
    with pytest.raises(Exception, match="pending_records"):
        sa.plan_for_bed(pending, tight, path="public", features=DemandFeatures())


def test_overlay_map_bytes_come_from_the_real_slot_table_and_indeterminate_is_refused(windowed_bed):
    from maple_syrup.sediment_bed import publish_bed_storage

    bed, ctx = windowed_bed
    ctx = dataclasses.replace(ctx, deep_chunk_encoding="overlay")
    request = 0.05 * host(bed.active_layer.mass_kg)
    pool0 = host(bed.water.mobile_mass_by_cell_class_kg).copy()
    pending = pickup_transport_deposit(bed, ctx, bed.water.depth_m, request, make_transport(pool0, 1.0)[0])[0]
    published = publish_bed_storage(pending, ctx)  # exact overlay publication (allowed on the merged pin)
    store = published.voxel_column.deep_store
    assert store.format_version == 2
    expected = max([0] + [store.slot_files(i).overlay.nbytes for i in range(store.n_chunks)
                          if store.slot_files(i).overlay is not None])
    plan = sa.plan_for_bed(published, ctx, path="public", features=DemandFeatures())
    assert plan.inputs["overlay_bytes_max"] == expected
    assert plan.mapped_bytes_max == store.mapping_cache_chunks * (store.header.chunk_bytes_max + expected)
    # a store whose format or slot table cannot be read is refused, never planned as zero (plain object, no MAPLE type)
    with pytest.raises(StorageAdmissionError, match="indeterminate"):
        sa._largest_overlay_bytes(object())


def test_unsupported_source_and_dense_bed_are_refused_before_any_query(windowed_bed, monkeypatch):
    from maple_syrup import sediment_bed as sb

    bed, ctx = windowed_bed
    called = []

    def reading():
        called.append(1)
        return BIG

    monkeypatch.setitem(sb._MAPLE_DIGEST_CACHE, "digest", "0" * 64)
    with pytest.raises(StorageAdmissionError, match="merged MAPLE package"):
        sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=0, host_available_bytes_fn=reading)
    assert called == []
    monkeypatch.undo()

    from pond_helpers import plane
    from windowed_helpers import dense_deep_bed

    dense, dctx = dense_deep_bed(plane(), depth=0.003)
    with pytest.raises(StorageAdmissionError, match="resident-window"):
        sa.admit_public(dense, dctx, _demand(dense), reserve_host_bytes=0, host_available_bytes_fn=reading)
    assert called == []


def test_host_admission_never_imports_or_queries_a_device(windowed_bed):
    present = "cupy" in sys.modules
    bed, ctx = windowed_bed
    sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=0, host_available_bytes_fn=lambda: BIG)
    if not present:
        assert "cupy" not in sys.modules


@pytest.mark.parametrize("args,expected", [
    ((100, 50, 0, 0), 150),            # no pool limit: driver free + retained pool blocks
    ((100, 50, 120, 30), 90),          # positive limit binds: limit - used
    ((100, 50, 1000, 30), 150),        # limit looser than reachable memory
    ((100, 50, 20, 30), 0),            # used above the limit: nothing, never negative
])
def test_effective_free_device_bytes_arithmetic(args, expected):
    assert sa.effective_free_device_bytes(*args) == expected


@pytest.mark.parametrize("bad", [(-1, 0, 0, 0), (0, 1.5, 0, 0), (0, 0, True, 0), (0, 0, 0, None)])
def test_effective_free_device_bytes_refuses_malformed_numbers(bad):
    with pytest.raises(StorageAdmissionError):
        sa.effective_free_device_bytes(*bad)
