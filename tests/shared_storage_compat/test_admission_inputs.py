"""Handoff precondition for the DRIVER-boundary admission (`windowed_transaction_bytes` / `admit_windowed_transaction`).

SYRUP does not call the planner yet (production change not authorized in this stage). These tests only pin that the merged
planner accepts every input read from a genuine SYRUP/MAPLE windowed state, that its figure is an upper bound of the
new native bytes MEASURED (tracemalloc, NumPy) for the same warm call on the tiny fixture, and that an admission DECISION is
exact arithmetic on injected readings -- not a reservation and not a whole-process bound. `_plan` below is a test-only reading
of the documented driver pattern, not a SYRUP function and not a design of record. CPU/NumPy only; no GPU admission is
exercised (a `cupy` plan is a pure count and is only checked for its refusal rules). Not executed by its author."""
from __future__ import annotations

import tracemalloc
from fractions import Fraction

import numpy as np
import pytest
from compat_support import host, make_transport

from maple_syrup.pond_bed import pickup_transport_deposit
from maple_syrup.sediment_bed import apply_bed_demand


def _plan(bed, ctx, *, path="public", namespace="numpy", max_gather_columns=256, max_journal_records=None):
    from maple.surface.voxels.deep_store import DEFAULT_MAX_JOURNAL_RECORDS
    from maple.surface.voxels.windowed_admission import windowed_transaction_bytes

    column, g = bed.voxel_column, ctx.geometry
    cap = (ctx.window_max_journal_records if ctx.window_max_journal_records is not None
           else DEFAULT_MAX_JOURNAL_RECORDS) if max_journal_records is None else max_journal_records
    return windowed_transaction_bytes(
        ny=g.ny, nx=g.nx, n_classes=len(ctx.grain_classes.classes), n_voxels=column.n_voxels,
        window_voxels=column.window_voxels, packet_terms=0 if column.height_packet is None else column.height_packet.packet_terms,
        base_itemsize=column.base.dtype.itemsize, ledger_present_processes=len(bed.ledger.present_processes),
        has_availability=bed.sediment_availability is not None, has_boundary_input=False, has_boundary_export=False,
        has_face_flux=False, max_gather_columns=max_gather_columns, max_journal_records=cap,
        pending_records=0 if column.pending_deep is None else column.pending_deep.journal.n_records,
        chunk_bytes_max=column.deep_store.header.chunk_bytes_max, overlay_bytes_max=0,
        mapping_cache_chunks=column.deep_store.mapping_cache_chunks, namespace=namespace, path=path)


def test_plan_accepts_genuine_state_fields_and_records_them(windowed_bed):
    bed, ctx = windowed_bed
    plan = _plan(bed, ctx)
    assert plan.admissible and plan.unbounded_terms == ()
    assert plan.inputs["ledger_present_processes"] == len(bed.ledger.present_processes) == 1  # compact water-only ledger
    assert plan.inputs["window_voxels"] == bed.voxel_column.window_voxels
    assert plan.inputs["n_voxels"] == bed.voxel_column.n_voxels
    assert plan.inputs["max_gather_columns"] == 256  # what SYRUP's adapter leaves to MAPLE's default
    assert plan.mapped_bytes_max == (bed.voxel_column.deep_store.mapping_cache_chunks
                                     * bed.voxel_column.deep_store.header.chunk_bytes_max)
    # what the figure does NOT cover is declared, not hidden
    text = " ".join(plan.excluded)
    for word in ("publication", "commit", "external transport", "CuPy", "JIT"):
        assert word in text
    assert plan.new_device_bytes_peak == 0


def test_dense_seven_channel_ledger_enlarges_the_ledger_terms_and_never_shrinks_the_peak(windowed_bed):
    from dataclasses import replace

    from maple.core.types.sediment_ledger import (
        N_LEDGER_PROCESSES,
        zeros_sediment_ledger_state,
    )

    bed, ctx = windowed_bed
    dense_ledger = zeros_sediment_ledger_state(ctx.geometry.ny, ctx.geometry.nx, len(ctx.grain_classes.classes))
    assert len(dense_ledger.present_processes) == N_LEDGER_PROCESSES == 7
    compact = _plan(bed, ctx)
    dense = _plan(replace(bed, ledger=dense_ledger), ctx)
    assert dense.inputs["ledger_present_processes"] == 7
    # A gather-dominated stage may set the maximum for both, so the overall peak need not grow strictly; what must hold
    # is that the ledger-live bytes and the ledger stage grow with the stored channel count and no stage shrinks.
    assert dense.terms["ledger.one_ledger.host"] > compact.terms["ledger.one_ledger.host"]
    assert dense.stages["ledger"][0] > compact.stages["ledger"][0]
    assert dense.caller_live_host_bytes > compact.caller_live_host_bytes
    assert dense.new_host_bytes_peak >= compact.new_host_bytes_peak
    assert all(dense.stages[name][0] >= compact.stages[name][0] for name in compact.stages)


def test_paired_figure_covers_the_allocating_fallback_and_advisory_never_exceeds_it(windowed_bed):
    bed, ctx = windowed_bed
    public, paired = _plan(bed, ctx), _plan(bed, ctx, path="paired")
    assert paired.new_host_bytes_peak >= public.new_host_bytes_peak
    assert public.advisory_owned_host_bytes_peak is None
    assert paired.advisory_owned_host_bytes_peak <= paired.new_host_bytes_peak
    assert any(name.startswith("phase_d.") for name in paired.stages)


def test_planner_peak_bounds_the_measured_new_native_bytes_of_the_same_warm_calls(windowed_bed):
    from maple.water import WaterProcessDemand

    bed, ctx = windowed_bed
    request = 0.05 * host(bed.active_layer.mass_kg)
    zeros = np.zeros_like(request)
    pool0 = host(bed.water.mobile_mass_by_cell_class_kg).copy()

    def public():
        return apply_bed_demand(bed, ctx, WaterProcessDemand(request, zeros))

    def paired():
        return pickup_transport_deposit(bed, ctx, bed.water.depth_m, request, make_transport(pool0, 1.0)[0])

    for call, path in ((public, "public"), (paired, "paired")):
        call()  # warm: imports, caches, first-use tables
        tracemalloc.start()
        try:
            base, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            result = call()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        plan = _plan(bed, ctx, path=path)
        assert peak - base <= plan.new_host_bytes_peak, (
            f"{path}: measured new bytes {peak - base} exceed the planner's {plan.new_host_bytes_peak} "
            f"(peak stage {plan.peak_stage_host!r})")
        del result


def test_receipt_transient_estimate_is_a_diagnostic_not_the_admission_figure(windowed_bed):
    bed, ctx = windowed_bed
    request = 0.05 * host(bed.active_layer.mass_kg)
    pool0 = host(bed.water.mobile_mass_by_cell_class_kg).copy()
    deposit = pickup_transport_deposit(bed, ctx, bed.water.depth_m, request, make_transport(pool0, 1.0)[0])[2]
    estimate = deposit.exchange_receipt.report["transient_bytes_estimate"]
    assert estimate < _plan(bed, ctx, path="paired").new_host_bytes_peak


def test_admission_is_exact_rational_arithmetic_on_injected_readings(windowed_bed):
    from maple.surface.voxels.windowed_admission import admit_windowed_transaction

    bed, ctx = windowed_bed
    plan = _plan(bed, ctx)
    reserve = 12_345  # a driver's own figure for what the plan excludes; any nonnegative int
    required = plan.new_host_bytes_peak + reserve
    admitted = admit_windowed_transaction(plan, host_available_bytes_fn=lambda: 2 * required, reserve_host_bytes=reserve)
    refused = admit_windowed_transaction(plan, host_available_bytes_fn=lambda: 2 * required - 1, reserve_host_bytes=reserve)
    assert admitted.admitted and admitted.host_required_bytes == required and admitted.host_margin_bytes == 0
    assert "not a reservation" in admitted.reason
    assert not refused.admitted and refused.host_margin_bytes == -1 and "insufficient host memory" in refused.reason
    third = admit_windowed_transaction(plan, host_available_bytes_fn=lambda: 3 * required, reserve_host_bytes=reserve,
                                       safety_fraction=Fraction(1, 3))
    assert third.admitted and third.safety_fraction == "1/3"


@pytest.mark.parametrize("reading", [None, -1, 1.5, True])
def test_unknown_or_malformed_reading_is_a_named_refusal(windowed_bed, reading):
    from maple.surface.voxels.windowed_admission import admit_windowed_transaction

    bed, ctx = windowed_bed
    decision = admit_windowed_transaction(_plan(bed, ctx), host_available_bytes_fn=lambda: reading)
    assert not decision.admitted and "host availability" in decision.reason


def test_host_plan_refuses_device_inputs_and_device_plan_needs_an_effective_free_query(windowed_bed):
    from maple.surface.voxels.windowed_admission import (
        WindowedAdmissionError,
        admit_windowed_transaction,
    )

    bed, ctx = windowed_bed
    host_plan = _plan(bed, ctx)
    with pytest.raises(WindowedAdmissionError, match="host-only"):
        admit_windowed_transaction(host_plan, host_available_bytes_fn=lambda: 10**15, reserve_device_bytes=1)
    with pytest.raises(WindowedAdmissionError, match="host-only"):
        admit_windowed_transaction(host_plan, host_available_bytes_fn=lambda: 10**15,
                                   device_available_bytes_fn=lambda: 10**15)
    device_plan = _plan(bed, ctx, namespace="cupy")  # a pure count; no device is touched
    assert device_plan.new_device_bytes_peak > 0
    no_query = admit_windowed_transaction(device_plan, host_available_bytes_fn=lambda: 10**15)
    assert not no_query.admitted and "device" in no_query.reason
    unknown = admit_windowed_transaction(device_plan, host_available_bytes_fn=lambda: 10**15,
                                         device_available_bytes_fn=lambda: None)
    assert not unknown.admitted
    ok = admit_windowed_transaction(device_plan, host_available_bytes_fn=lambda: 10**15,
                                    device_available_bytes_fn=lambda: 10**15, reserve_device_bytes=7)
    assert ok.admitted and ok.device_required_bytes == device_plan.new_device_bytes_peak + 7
