"""E4k CUDA DEFINITIONS for Root (NOT run by the author; opt-in `SYRUP_RELEASE_GPU=1`). CuPy is imported only inside the tests."""
from __future__ import annotations

import os

import numpy as np
import pytest
from compat_support import host

from maple_syrup import storage_admission as sa
from maple_syrup.storage_admission import DemandFeatures, StorageAdmissionError

GPU = pytest.mark.skipif(os.environ.get("SYRUP_RELEASE_GPU") != "1",
                         reason="GPU tests are opt-in (SYRUP_RELEASE_GPU=1); definitions for Root, not run by the author")
BIG = 10**18


def _device_bed(tmp_path):
    import cupy as cp
    from pond_helpers import plane
    from windowed_helpers import dense_deep_bed, heavy_mobile, windowed_counterpart

    dense, ctx = dense_deep_bed(plane(), depth=0.003, mobile=heavy_mobile())
    return windowed_counterpart(dense, ctx, tmp_path / "store", xp=cp)


def _demand(bed):
    import cupy as cp
    from maple.water import WaterProcessDemand

    request = 0.05 * bed.active_layer.mass_kg
    return WaterProcessDemand(request, cp.zeros_like(request))


@GPU
def test_device_bed_requires_an_explicit_device_reserve_and_plans_device_bytes(tmp_path):
    bed, ctx = _device_bed(tmp_path)
    with pytest.raises(StorageAdmissionError, match="explicit reserve_device_bytes"):
        sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=0, host_available_bytes_fn=lambda: BIG)
    record = sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=0, reserve_device_bytes=0,
                             host_available_bytes_fn=lambda: BIG, device_available_bytes_fn=lambda: BIG)
    assert record.namespace == "cupy" and record.plan.new_device_bytes_peak > 0


@GPU
def test_default_query_reads_only_the_current_device_and_refuses_before_reading_on_another(tmp_path):
    import cupy as cp

    bed, ctx = _device_bed(tmp_path)
    free_before = cp.cuda.Device().mem_info[0]
    record = sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=0, reserve_device_bytes=0,
                             host_available_bytes_fn=lambda: BIG)
    assert record.decision.admitted and cp.cuda.Device().id == int(bed.water.depth_m.device.id)
    assert cp.cuda.Device().mem_info[0] <= free_before + record.plan.new_device_bytes_peak  # no allocation by admission
    # a device-mismatch refusal is numeric and precedes any reading (only when >1 device exists)
    if cp.cuda.runtime.getDeviceCount() > 1:
        other = 1 - int(bed.water.depth_m.device.id)
        with cp.cuda.Device(other), pytest.raises(StorageAdmissionError, match="current device"):
            sa.admit_public(bed, ctx, _demand(bed), reserve_host_bytes=0, reserve_device_bytes=0,
                            host_available_bytes_fn=lambda: BIG)


@GPU
def test_exact_device_threshold_refusal_before_any_exchange_and_bed_unchanged(tmp_path):
    bed, ctx = _device_bed(tmp_path)
    before = host(bed.water.depth_m).copy()
    peak = sa.plan_for_bed(bed, ctx, path="paired", features=DemandFeatures()).new_device_bytes_peak
    ok = sa.admit_paired(bed, ctx, features=DemandFeatures(), reserve_host_bytes=0, reserve_device_bytes=11,
                         host_available_bytes_fn=lambda: BIG, device_available_bytes_fn=lambda: 2 * (peak + 11))
    assert ok.decision.admitted
    with pytest.raises(MemoryError, match="device"):
        sa.admit_paired(bed, ctx, features=DemandFeatures(), reserve_host_bytes=0, reserve_device_bytes=11,
                        host_available_bytes_fn=lambda: BIG, device_available_bytes_fn=lambda: 2 * (peak + 11) - 1)
    np.testing.assert_array_equal(host(bed.water.depth_m), before)
