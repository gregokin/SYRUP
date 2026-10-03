"""The experimental driver on a real device must avoid raw host conversions and repeated scalar-array allocation (raw `cupy.asarray(host_float)` bypasses
MAPLE's transfer counters, so a zero counter alone proves nothing). After solver/field/state preparation a trap on
`cupy.asarray` / `cupy.array` refuses any call whose argument is not already a device array; the storm must still complete and
agree with the NumPy reference; the counted loop transfers are exactly the two packet reads per attempt and no upload.
Skipped without a device. Nothing here was run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from cand_cases import build, device_twin, field, host, schedule

pytest.importorskip("maple")

from maple.core import backend as mb

from maple_syrup import experimental_cuda as ec
from maple_syrup import hydrology_cuda as hc
from maple_syrup.experimental_hydrology import TRANSFER_SCOPE
from maple_syrup.experimental_storm import ExperimentalControl, evolve_experimental
from maple_syrup.storm import StormError

pytestmark = pytest.mark.usefixtures("gpu")
METHODS = ("explicit", "local_inertial")


def cupy():
    import cupy as cp

    return cp


@pytest.fixture
def host_conversion_trap(monkeypatch):
    """Install AFTER every valid preparation. Records and refuses `cp.asarray` / `cp.array` of anything that is not already a
    CuPy array (Python/NumPy scalars, NumPy arrays, lists): scalar conversions allocate/fill and host arrays can incur copies."""
    cp = cupy()
    log: list = []

    def guard(name, original):
        def wrapper(obj, *args, **kwargs):
            if not isinstance(obj, cp.ndarray):
                log.append((name, type(obj).__name__, repr(obj)[:40]))
                raise TypeError(f"cupy.{name} of a host {type(obj).__name__} would be an uncounted host conversion")
            return original(obj, *args, **kwargs)

        return wrapper

    def install():
        monkeypatch.setattr(cp, "asarray", guard("asarray", cp.asarray))
        monkeypatch.setattr(cp, "array", guard("array", cp.array))

    return log, install


def prepared(method, seed, **kw):
    cp = cupy()
    cs = build("valley:8x7", method, depth="wet", seed=seed, **kw)
    dev = device_twin(cs, cp)
    return cp, cs, dev, field(cs, cp)


@pytest.mark.parametrize("method", METHODS)
def test_a_whole_storm_creates_no_device_array_from_a_host_value_and_matches_the_reference(method, host_conversion_trap):
    _cp, cs, dev, fld = prepared(method, 1, ksat=2e-6, soil_fraction=0.4, drainage=0.3)
    sched = schedule([0.0, 10.0, 25.0, 40.0], [120.0, 0.0, 60.0])  # forcing edges, a recession and a reporting cadence
    control = ExperimentalControl(max_dt_s=0.25)
    ref = evolve_experimental(cs.solver, field(cs), sched, cs.state, 40.0, control, report_every_s=15.0,
                              snapshot_times_s=(12.5,))
    log, install = host_conversion_trap
    install()

    def counted(end):
        before = mb.read_transfer_counters()
        result = evolve_experimental(dev.solver, fld, sched, dev.state, end, control, report_every_s=15.0,
                                     snapshot_times_s=(12.5,))
        return result, mb.read_transfer_counters().delta(before)

    short, d_short = counted(20.0)
    new, delta = counted(40.0)
    assert log == [], log  # no raw host conversion or extra scalar-array creation
    # whatever the (constant) entry validation costs, the extra steps cost exactly the two packets each and no upload
    extra = (new.n_accepted_steps + new.n_rejected_attempts) - (short.n_accepted_steps + short.n_rejected_attempts)
    assert extra > 0
    assert delta.host_to_device == d_short.host_to_device and delta.host_to_device_bytes == d_short.host_to_device_bytes
    assert delta.scalar_reads == d_short.scalar_reads
    assert delta.device_to_host - d_short.device_to_host == 2 * extra  # the column packet and the lateral packet
    assert delta.device_to_host_bytes - d_short.device_to_host_bytes == extra * (hc.PACKET_WORDS + ec.PACKET_WORDS) * 8
    # the by-value numbers still land in the right places
    assert new.n_accepted_steps == ref.n_accepted_steps and new.n_rejected_attempts == ref.n_rejected_attempts
    np.testing.assert_array_equal(host(new.hydrograph[:, 0]), ref.hydrograph[:, 0])  # row times
    for column in (11, 12, 13):  # accepted steps, rejected attempts, smallest accepted dt
        np.testing.assert_array_equal(host(new.hydrograph[:, column]), ref.hydrograph[:, column])
    np.testing.assert_allclose(host(new.hydrograph), ref.hydrograph, rtol=2e-12, atol=1e-14)
    assert float(host(new.time_of_peak_outlet_s)) == float(ref.time_of_peak_outlet_s)
    np.testing.assert_allclose(float(host(new.peak_outlet_discharge_m3_s)), float(ref.peak_outlet_discharge_m3_s),
                               rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize("method", METHODS)
def test_the_trap_would_catch_the_old_per_step_scalar_conversion(method, host_conversion_trap):
    """Self-check of the trap itself: the pattern the driver used to run on every accepted step is refused."""
    cp, _cs, _dev, _fld = prepared(method, 2)
    log, install = host_conversion_trap
    install()
    with pytest.raises(TypeError, match="uncounted host conversion"):
        cp.asarray(1.0, dtype=np.float64)
    with pytest.raises(TypeError, match="uncounted host conversion"):
        cp.array(np.zeros(3))
    assert [entry[0] for entry in log] == ["asarray", "array"]
    device = cp.zeros(2)
    assert cp.asarray(device) is device  # a device array passes through untouched


def test_a_host_scalar_from_a_solver_is_refused_instead_of_implicitly_converted(host_conversion_trap):
    """If a step ever handed the driver a host float where a device value is required, the driver must say so (a stack/where of
    it could add an implicit allocation or copy) rather than convert it."""
    _cp, _cs, dev, fld = prepared("explicit", 3)

    class HostExport:
        def __init__(self, solver):
            self._solver = solver

        def __getattr__(self, name):
            return getattr(self._solver, name)

        def step(self, rain, state, dt):
            step = self._solver.step(rain, state, dt)
            return dataclasses.replace(step, outlet_discharge_m3_s=0.0)  # a Python float on a CUDA solver

    log, install = host_conversion_trap
    install()
    with pytest.raises(StormError, match="device-resident"):
        evolve_experimental(HostExport(dev.solver), fld, schedule([0.0, 5.0], [60.0]), dev.state, 5.0,
                            ExperimentalControl(max_dt_s=0.5), report_every_s=2.5)
    assert log == []


@pytest.mark.parametrize("method", METHODS)
def test_the_describe_record_states_the_transfer_scope(method):
    _cp, cs, dev, _fld = prepared(method, 4)
    assert dev.solver.describe()["transfer_scope"] == TRANSFER_SCOPE == cs.solver.describe()["transfer_scope"]
    assert "uncounted" in TRANSFER_SCOPE and "by value" in TRANSFER_SCOPE and "packets" in TRANSFER_SCOPE
