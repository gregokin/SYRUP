"""HOST rainfall metadata census only; no Native action or CUDA provider.

The real RainfallSchedule constructor supplies the arrays. The synthetic
context/unused device type below exercise only the existing metadata walker;
they cannot authorize a physical water operation. Root executes these tests.
"""
from __future__ import annotations

from dataclasses import make_dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from maple_syrup import windowed_shared_bed_device as device
from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule
from maple_syrup.sediment_bed import BedIntegrationError


@pytest.fixture
def census(monkeypatch):
    class UnusedDeviceArray:
        pass
    bed = SimpleNamespace(_cupy_ndarray=lambda: UnusedDeviceArray)
    monkeypatch.setattr(device, '_native', lambda: (bed,))
    assert device._OWNER is None
    def run(roots, *, backing_limit=64 << 20):
        context = SimpleNamespace(
            initial_available_fraction=SimpleNamespace(device=SimpleNamespace(id=0)),
            bridge_limits=SimpleNamespace(max_external_backing_bytes=backing_limit,
                                          max_report_rows=4096))
        return device._backing_census(roots, context)
    return run


def _schedule(n=1):
    return RainfallSchedule(np.arange(n + 1, dtype=np.float64),
                            np.ones(n, dtype=np.float64), RainfallProvenance(kind='constant'))


def _readonly(array):
    array.flags.writeable = False
    return array


def test_native_schedule_metadata_preserves_arrays_and_charges_every_byte(census):
    schedule = _schedule()
    names = ('edges_s', 'intensity_mm_per_h', 'rate_m_per_s')
    before = tuple((getattr(schedule, name), getattr(schedule, name).tobytes()) for name in names)
    host, device_bytes, nodes, leaves = census((('schedule', schedule),))
    assert host == sum(array.nbytes for array, _raw in before)
    assert device_bytes == 0 and leaves == 3 and nodes == 15
    for name, (array, raw) in zip(names, before):
        assert getattr(schedule, name) is array and array.tobytes() == raw
        assert not array.flags.writeable and array.flags.owndata and array.base is None


def test_schedule_intervals_are_not_limited_by_the_report_clock_cap(census):
    schedule = _schedule(5000)
    host, device_bytes, _nodes, leaves = census((('schedule', schedule),))
    assert host == (5001 + 5000 + 5000) * 8 and device_bytes == 0 and leaves == 3


def test_repeated_schedule_occurrences_keep_the_original_whole_backing_cap(census):
    schedule = _schedule()
    host, device_bytes, _nodes, leaves = census((('schedule', schedule),) * 2, backing_limit=16)
    assert host == 64 and device_bytes == 0 and leaves == 6  # No alias credit.
    with pytest.raises(BedIntegrationError, match='whole_external_backing_occurrences'):
        census((('schedule', schedule),) * 3, backing_limit=16)


def test_schedule_owner_must_be_the_exact_native_class(census):
    schedule = _schedule()
    values = vars(schedule)
    forged = make_dataclass('ForgedSchedule', [(name, object) for name in values])(**values)
    class ScheduleSubclass(RainfallSchedule):
        pass
    subclass = ScheduleSubclass(schedule.edges_s, schedule.intensity_mm_per_h, schedule.provenance)
    for wrong in (forged, subclass, values.copy()):
        with pytest.raises(BedIntegrationError, match='exact_holder'):
            census((('schedule', wrong),))


def test_native_owner_with_an_extra_field_refuses(census):
    schedule = _schedule()
    object.__setattr__(schedule, 'physical_payload', np.zeros(1))
    with pytest.raises(BedIntegrationError, match='closed_holder'):
        census((('schedule', schedule),))


def test_mutated_rainfall_array_metadata_refuses(census):
    wrong_arrays = (
        _readonly(np.ones(1, dtype=np.float32)),
        _readonly(np.ones((1, 1), dtype=np.float64)),
        _readonly(np.arange(4, dtype=np.float64)[::2]),
        np.ones(1, dtype=np.float64),
        _readonly(np.ones(1, dtype=np.float64).view()),
        _readonly(np.array([object()], dtype=object)),
        [1.0],
    )
    for name in ('edges_s', 'intensity_mm_per_h', 'rate_m_per_s'):
        for wrong in wrong_arrays:
            schedule = _schedule()
            object.__setattr__(schedule, name, wrong)
            with pytest.raises(BedIntegrationError, match='native_HOST_rainfall_metadata'):
                census((('schedule', schedule),))


def test_native_rainfall_length_relationships_are_required(census):
    for name, size in (('edges_s', 1), ('intensity_mm_per_h', 0), ('rate_m_per_s', 2)):
        schedule = _schedule()
        object.__setattr__(schedule, name, _readonly(np.zeros(size, dtype=np.float64)))
        with pytest.raises(BedIntegrationError, match='native_HOST_rainfall_lengths'):
            census((('schedule', schedule),))


def test_schedule_field_paths_without_an_authenticated_owner_refuse(census):
    schedule = _schedule()
    for name in ('edges_s', 'intensity_mm_per_h', 'rate_m_per_s'):
        with pytest.raises(BedIntegrationError, match='unexpected_HOST_payload_schedule'):
            census((('schedule.' + name, getattr(schedule, name)),))


def test_rainfall_does_not_open_other_host_payload_paths(census):
    schedule = _schedule()
    for roots in ((('other_schedule', schedule),),
                  (('schedule', schedule), ('soil_water_m', schedule.rate_m_per_s)),
                  (('schedule', schedule), ('schedule.payload', schedule.rate_m_per_s))):
        with pytest.raises(BedIntegrationError, match='unexpected_HOST_payload_'):
            census(roots)


def test_rainfall_arrays_still_obey_the_original_individual_backing_cap(census):
    with pytest.raises(BedIntegrationError, match='external_array_domain'):
        census((('schedule', _schedule()),), backing_limit=8)


def test_original_null_metadata_and_combined_report_clock_remain_eligible(census):
    # This is the walker's old primitive None case, not a dry-forcing API.
    # The original complete_event still needs schedule.edges_s/end_s.
    assert census((('schedule', None),)) == (0, 0, 1, 0)
    schedule = _schedule()
    boundaries = np.array([0.5, 1.0], dtype=np.float64)
    host, device_bytes, nodes, leaves = census(
        (('state', None), ('schedule', schedule), ('planned_boundaries', boundaries)))
    assert host == 48 and device_bytes == 0 and nodes == 17 and leaves == 4
