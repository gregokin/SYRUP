"""CPU NumPy / Numba and real CuPy parity of the pond event. Availability is decided in `backend_namespace`: a missing
Numba / CuPy / CUDA device is an honest skip, never a fallback and never a GPU claim without a device. Nothing here was run by
its author (file-only tools)."""
import numpy as np
import pytest
from pond_helpers import backend_namespace, dry_plane, host, wet_plane

RTOL, ATOL = 2e-12, 1e-14  # existing hydraulic-candidate policy
SED_RTOL, SED_ATOL = 1e-10, 1e-14  # existing conservative sediment policy


def run_wet(be, end=0.6):
    event, state = wet_plane(be)
    return event, event.evolve(state, end)


def test_backend_matches_the_numpy_reference_on_the_wet_scenario(be):
    reference_event, reference = run_wet(backend_namespace("numpy"))
    event, result = run_wet(be)
    assert result.reached_target and result.n_accepted_steps == reference.n_accepted_steps
    for name in ("depth_m", "soil_water_m", "qx_m2_s", "qy_m2_s"):
        np.testing.assert_allclose(host(getattr(result.state.hydraulic, name)),
                                   host(getattr(reference.state.hydraulic, name)), rtol=RTOL, atol=ATOL, err_msg=name)
    np.testing.assert_allclose(host(result.state.bed.water.mobile_mass_by_cell_class_kg),
                               host(reference.state.bed.water.mobile_mass_by_cell_class_kg), rtol=SED_RTOL, atol=SED_ATOL)
    a, b = event.audit(result.state), reference_event.audit(reference.state)
    assert a["water"]["closes"] and a["sediment"]["closes"]
    assert a["water"]["rain_m3"] == pytest.approx(b["water"]["rain_m3"], rel=1e-12)
    np.testing.assert_allclose(a["sediment"]["wet_actual_kg"], b["sediment"]["wet_actual_kg"], rtol=SED_RTOL,
                               atol=SED_ATOL)
    # the executed transport substep maximum is recorded identically and is consistent with the accepted steps
    used = result.state.counters.max_transport_substeps_used
    assert used == reference.state.counters.max_transport_substeps_used and 1 <= used <= event.control.max_transport_substeps


def test_cupy_state_and_ledgers_stay_on_the_device():
    be = backend_namespace("cupy")
    event, state = wet_plane(be)
    result = event.evolve(state, 0.3)
    cp = be.xp
    for array in (result.state.hydraulic.depth_m, result.state.hydraulic.qx_m2_s, result.state.sediment_velocity_m_s,
                  result.state.phase_fraction, result.state.held_source_m, result.state.water.rain,
                  result.state.sediment.pickup_cell, result.state.bed.water.mobile_mass_by_cell_class_kg):
        assert isinstance(array, cp.ndarray)
    assert result.state.topology.source_z_full.__class__ is np.ndarray  # topology is host-only by design
    assert event.audit(result.state)["sediment"]["closes"]


def test_explicit_cuda_splash_path_matches_the_array_path_on_the_dry_scenario():
    gpu = backend_namespace("cupy")
    cpu = backend_namespace("numpy")
    cpu_event, cpu_state = dry_plane(cpu)
    gpu_event, gpu_state = dry_plane(gpu, impl="cuda")  # the tested compiled sweeps, explicitly requested
    assert gpu_event.control.splash_implementation == "cuda"
    a = cpu_event.evolve(cpu_state, 1.5)
    b = gpu_event.evolve(gpu_state, 1.5)
    assert a.state.counters.n_splash_exchanges == b.state.counters.n_splash_exchanges > 0
    ca, cb = cpu_event.audit(a.state)["sediment"], gpu_event.audit(b.state)["sediment"]
    for key in ("splash_requested_kg", "splash_actual_kg", "deposited_splash_kg", "export_splash_kg"):
        np.testing.assert_allclose(cb[key], ca[key], rtol=1e-10, atol=1e-15, err_msg=key)
    assert cb["closes"]


def test_backend_namespace_mismatch_is_refused_without_fallback():
    from pond_helpers import make_bed, make_event, make_topology, east_outlets, plane

    from maple_syrup.pond_event import PondEventControl, PondEventError

    bed, ctx = make_bed(plane(), depth=0.003)
    topo = make_topology(bed, east_outlets(3, 4), drop=0.02)
    with pytest.raises(PondEventError, match="no fallback"):
        make_event(bed, ctx, topo, control=PondEventControl(backend="cuda"))  # NumPy parameters, cuda control
