"""Characteristic transport inside the real event, completion, checkpoint
and frozen-benchmark drivers on small actual-MAPLE beds.

Covers: scheme / bin-count consistency between state and control (refused
before mutation), the phase as a partition of the ACTUAL MAPLE pool after
every accepted step, atomicity of rejected attempts, terminal dry handoff
canonicalisation and its restart, wet restart with a non-empty mobile pool,
checkpoint refusals (bad fractions / positions / bin count / superseded
schema), rerouting that keeps within-cell progress, two-way transport
implementation invariance of the frozen benchmark, and the explicitly named
upwind comparison scheme. No Plot 1 storm-scale claim is made here.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

pytest.importorskip("maple")

from test_complete_event import fixture as completion_fixture
from test_sediment_event import (
    assert_closed,
    chain_elevation,
    make_bed,
    run,
    setup,
    valley_case,
    valley_elevation,
)

from maple_syrup import sediment_event
from maple_syrup.benchmark_experiment import frozen_control, run_frozen_event
from maple_syrup.characteristic_transport import validate_phase_state
from maple_syrup.checkpoint import SCHEMA, load_checkpoint, save_checkpoint, sha256_file
from maple_syrup.complete_event import DryHandoff, complete_event
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.routing_numba import numba_available
from maple_syrup.sediment_event import (
    SedimentEventControl,
    SedimentEventError,
    evolve_sediment_event,
    initial_event_state,
    sediment_coupled_step,
)
from maple_syrup.storm import StormControl

IDENTITY = {"case": "controlled actual MAPLE bed", "forcing": "fixed", "source": "test", "options": {"dt": 1}}
NUMBA_SKIP = pytest.mark.skipif(not numba_available(), reason="Numba not installed; no claim made")


def phase_arrays(state):
    return state.phase.fraction.copy(), state.phase.position_m.copy()


def mass_weighted_position(state):
    w = state.bed.water.mobile_mass_by_cell_class_kg[..., None] * state.phase.fraction
    return (w * state.phase.position_m).sum(axis=-1), w.sum(axis=-1)


# --- scheme / bins consistency ---------------------------------------------------------------------------------
def test_state_and_control_must_agree_on_scheme_and_bins_before_anything_moves():
    bed, ctx = make_bed(chain_elevation(3))
    default = setup(bed, ctx)
    state = default["state0"]
    assert state.phase is not None and state.phase.n_bins == SedimentEventControl().phase_bins
    assert np.all(state.phase.fraction[..., 0] == 1.0) and not np.any(state.phase.position_m)
    upwind = setup(bed, ctx, control=SedimentEventControl(transport_scheme="upwind", sediment_courant_max=1.0))
    assert upwind["state0"].phase is None
    schedule = constant_rainfall(0.0, 10.0, 72.0)
    before = state.bed.active_layer.mass_kg.copy()
    with pytest.raises(SedimentEventError, match="bins"):
        run(default, schedule, 10.0, control=SedimentEventControl(phase_bins=16), cadence=10.0)
    with pytest.raises(SedimentEventError, match="upwind"):
        run(default, schedule, 10.0, control=SedimentEventControl(transport_scheme="upwind", sediment_courant_max=1.0),
            cadence=10.0)
    with pytest.raises(SedimentEventError, match="characteristic"):
        run(upwind, schedule, 10.0, control=SedimentEventControl(), cadence=10.0)
    np.testing.assert_array_equal(state.bed.active_layer.mass_kg, before)
    # a corrupted phase on the state is refused by the driver before any MAPLE call
    bad = dataclasses.replace(state, phase=dataclasses.replace(state.phase, position_m=state.phase.position_m + 0.1))
    with pytest.raises(SedimentEventError, match="phase state invalid"):
        run(default, schedule, 10.0, cadence=10.0, state=bad)


def test_direct_coupled_step_refuses_mismatched_or_invalid_controls_before_any_maple_call(monkeypatch):
    """Codex reproducer (direct_control_probe.log): a direct
    `sediment_coupled_step` call with a 16-bin control on a 32-bin state
    returned 32 bins without refusal. Every public entry must refuse an
    invalid control or a control that does not bind to the state's phase
    BEFORE hydraulics or MAPLE run."""
    bed, ctx = make_bed(chain_elevation(3))
    default = setup(bed, ctx)
    upwind = setup(bed, ctx, control=SedimentEventControl(transport_scheme="upwind", sediment_courant_max=1.0))
    rate = default["field"].apply(72.0 / 3.6e6)
    calls = {"coupled_step": 0, "apply_bed_demand": 0, "characteristic_step": 0, "transport_step": 0}

    def counting(name):
        original = getattr(sediment_event, name)

        def wrapped(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        return wrapped

    for name in calls:
        monkeypatch.setattr(sediment_event, name, counting(name))

    def attempt(state, control):
        return sediment_coupled_step(state, ctx, default["column"], rate, default["vegetation"], default["sediment"],
                                     1.0, control)

    bad_controls = {
        "bins": (default["state0"], SedimentEventControl(phase_bins=16), "16"),
        "invalid scheme": (default["state0"], dataclasses.replace(SedimentEventControl(), transport_scheme="lagrangian"),
                           "transport_scheme"),
        "invalid bins": (default["state0"], dataclasses.replace(SedimentEventControl(), phase_bins=0), "phase_bins"),
        "invalid implementation": (default["state0"],
                                   dataclasses.replace(SedimentEventControl(), transport_implementation="cuda"),
                                   "transport_implementation"),
        "courant too large for characteristic": (default["state0"],
                                                 dataclasses.replace(SedimentEventControl(), sediment_courant_max=1.0),
                                                 "sediment_courant_max"),
        "upwind control on a phase state": (default["state0"],
                                            SedimentEventControl(transport_scheme="upwind", sediment_courant_max=1.0),
                                            "upwind"),
        "characteristic control without a phase": (upwind["state0"], SedimentEventControl(), "phase state"),
        "not a control": (default["state0"], {"transport_scheme": "characteristic"}, "SedimentEventControl"),
    }
    for state, control, match in bad_controls.values():
        with pytest.raises(SedimentEventError, match=match):
            attempt(state, control)
    assert all(count == 0 for count in calls.values()), calls  # nothing ran: no hydraulics, no MAPLE call
    ok = attempt(default["state0"], SedimentEventControl())
    assert ok.phase.n_bins == 32 and calls["coupled_step"] == 1 and calls["apply_bed_demand"] == 2
    assert calls["characteristic_step"] == 1 and calls["transport_step"] == 0
    ok_upwind = attempt(upwind["state0"], SedimentEventControl(transport_scheme="upwind", sediment_courant_max=1.0))
    assert ok_upwind.phase is None and calls["transport_step"] == 1 and calls["characteristic_step"] == 1


# --- the phase partitions the ACTUAL pool after every accepted step ---------------------------------------------
def test_phase_partitions_the_actual_pool_and_the_input_state_is_untouched():
    inputs = valley_case()
    state0 = inputs["state0"]
    f0, x0 = phase_arrays(state0)
    r = run(inputs, constant_rainfall(0.0, 30.0, 72.0), 40.0, cadence=10.0)
    assert_closed(r, state0)
    st = r.state
    validate_phase_state(st.phase, st.bed.water.mobile_mass_by_cell_class_kg, st.network)
    assert np.any(st.bed.water.mobile_mass_by_cell_class_kg > 0.0)
    moment, _mass = mass_weighted_position(st)
    assert np.any(moment > 0.0)  # mass has advanced from the upstream face
    assert float(r.max_phase_reconciliation_residual_kg) <= 1e-12
    assert r.n_phase_canonicalized >= 0 and r.n_phase_rounding_remnants >= 0
    np.testing.assert_array_equal(state0.phase.fraction, f0)
    np.testing.assert_array_equal(state0.phase.position_m, x0)
    # one attempt on the wet state: its phase partitions ITS bed, the caller's state keeps its own
    rate = inputs["field"].apply(72.0 / 3.6e6)
    attempt = sediment_coupled_step(st, inputs["ctx"], inputs["column"], rate, inputs["vegetation"],
                                    inputs["sediment"], 1.0, SedimentEventControl())
    validate_phase_state(attempt.phase, attempt.bed.water.mobile_mass_by_cell_class_kg, st.network)
    assert attempt.characteristic is not None and attempt.phase is not st.phase
    assert float(attempt.phase_reconciliation["residual_max_kg"]) <= 1e-12
    validate_phase_state(st.phase, st.bed.water.mobile_mass_by_cell_class_kg, st.network)
    # the kernel saw the PRE-pickup pool and the ACTUAL pickup
    np.testing.assert_allclose(attempt.characteristic.mobile_before_kg,
                               st.bed.water.mobile_mass_by_cell_class_kg + attempt.pickup.actual_removal_by_cell_class_kg,
                               rtol=1e-15)


def test_reconciliation_failure_publishes_nothing(monkeypatch):
    inputs = valley_case()
    schedule = constant_rainfall(0.0, 30.0, 72.0)
    r = run(inputs, schedule, 5.0, control=SedimentEventControl(commit=False, force_final_commit=False), cadence=5.0)
    wet = r.state
    f, x = phase_arrays(wet)
    mobile = wet.bed.water.mobile_mass_by_cell_class_kg.copy()
    active = wet.bed.active_layer.mass_kg.copy()

    def refuse(step, actual, xp):
        raise SedimentEventError("forced reconciliation refusal")

    monkeypatch.setattr(sediment_event, "reconcile_phase", refuse)
    with pytest.raises(SedimentEventError, match="forced reconciliation refusal"):
        run(inputs, schedule, 10.0, cadence=10.0, state=wet)
    np.testing.assert_array_equal(wet.phase.fraction, f)
    np.testing.assert_array_equal(wet.phase.position_m, x)
    np.testing.assert_array_equal(wet.bed.water.mobile_mass_by_cell_class_kg, mobile)
    np.testing.assert_array_equal(wet.bed.active_layer.mass_kg, active)
    monkeypatch.undo()
    # a genuine reconciliation refusal: the actual pool differs from the kernel's beyond the FP bound
    from maple_syrup.sediment_event import reconcile_phase

    rate = inputs["field"].apply(72.0 / 3.6e6)
    attempt = sediment_coupled_step(wet, inputs["ctx"], inputs["column"], rate, inputs["vegetation"],
                                    inputs["sediment"], 1.0, SedimentEventControl())
    tampered = attempt.bed.water.mobile_mass_by_cell_class_kg.copy()
    tampered[0, 2, 1] += 1e-3
    with pytest.raises(SedimentEventError, match="beyond the declared FP64 bound"):
        reconcile_phase(attempt.characteristic, tampered, np)
    same, diagnostics = reconcile_phase(attempt.characteristic, attempt.bed.water.mobile_mass_by_cell_class_kg, np)
    np.testing.assert_array_equal(same.fraction, attempt.phase.fraction)
    assert float(diagnostics["residual_max_kg"]) <= 1e-12


# --- dry completion, wet restart, checkpoint refusals ----------------------------------------------------------
def test_dry_handoff_canonicalises_phase_and_reloads(tmp_path):
    args, kw = completion_fixture(wet=True)
    out = complete_event(*args, **kw)
    assert isinstance(out, DryHandoff)
    dry = out.dry_state
    assert not np.any(dry.bed.water.mobile_mass_by_cell_class_kg)
    assert np.all(dry.phase.fraction[..., 0] == 1.0) and not np.any(dry.phase.fraction[..., 1:])
    assert not np.any(dry.phase.position_m)
    # the pre-reset wet result also carries a canonical phase after the terminal deposition
    assert np.all(out.progress.result.state.phase.fraction[..., 0] == 1.0)
    assert out.progress.result.n_phase_canonicalized >= 0
    path = tmp_path / "handoff"
    save_checkpoint(path, out, args[1], args[2], args[6], IDENTITY)
    loaded = load_checkpoint(path, args[1], args[2], args[6], IDENTITY, allow_complete=True)
    np.testing.assert_array_equal(loaded.dry_state.phase.fraction, dry.phase.fraction)
    np.testing.assert_array_equal(loaded.dry_state.phase.position_m, dry.phase.position_m)
    assert loaded.dry_state.phase.n_bins == dry.phase.n_bins


def test_wet_checkpoint_with_mobile_mass_round_trips_and_resumes_identically(tmp_path):
    args, kw = completion_fixture(wet=True)
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    st = paused.result.state
    assert np.any(st.bed.water.mobile_mass_by_cell_class_kg > 0.0)
    assert np.any(st.phase.position_m > 0.0)  # a genuinely non-canonical partition is persisted
    path = tmp_path / "wet"
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    restored = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    np.testing.assert_array_equal(restored.result.state.phase.fraction, st.phase.fraction)
    np.testing.assert_array_equal(restored.result.state.phase.position_m, st.phase.position_m)
    assert not restored.result.state.phase.fraction.flags.writeable
    full = complete_event(*args, **kw)
    resumed = complete_event(restored.result.state, *args[1:], **kw, continuation=restored)
    for name in ("mobile_mass_by_cell_class_kg",):
        np.testing.assert_array_equal(getattr(full.dry_state.bed.water, name), getattr(resumed.dry_state.bed.water, name))
    np.testing.assert_array_equal(full.dry_state.bed.active_layer.mass_kg, resumed.dry_state.bed.active_layer.mass_kg)
    np.testing.assert_array_equal(full.progress.result.sediment_hydrograph, resumed.progress.result.sediment_hydrograph)
    assert full.progress.result.n_phase_canonicalized == resumed.progress.result.n_phase_canonicalized
    assert full.progress.result.n_phase_rounding_remnants == resumed.progress.result.n_phase_rounding_remnants
    assert float(full.progress.result.max_phase_reconciliation_residual_kg) == float(
        resumed.progress.result.max_phase_reconciliation_residual_kg)


@pytest.mark.parametrize("kind", ["fraction", "position", "bins", "schema1", "scheme"])
def test_checkpoint_refuses_bad_phase_and_superseded_schema(tmp_path, kind):
    args, _kw = completion_fixture(wet=True)
    paused = complete_event(*args, **_kw, checkpoint_callback=lambda p: True)
    path = tmp_path / "wet"
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    metadata = path / "checkpoint.json"
    m = json.loads(metadata.read_text())
    expected = json.loads(json.dumps(IDENTITY))
    node = m["payload"]["fields"]["result"]["fields"]["state"]["state"]["phase"]["phase"]
    with np.load(path / "continuation.npz") as archive:
        data = {k: archive[k].copy() for k in archive.files}
    if kind == "fraction":
        data[node["fraction"]["array"]][0, 0, 0, 0] += 0.5
    elif kind == "position":
        data[node["position_m"]["array"]][0, 0, 0, 0] = 0.5  # == dx
    elif kind == "bins":
        node["n_bins"] = 16
    elif kind == "schema1":
        m["schema"] = "maple-syrup-checkpoint/1"
    else:  # the identity binds the control (scheme and bins): a changed scheme is a changed run
        expected = dict(expected, transport="changed")
    np.savez(path / "continuation.npz", **data)
    m["files"]["continuation.npz"] = sha256_file(path / "continuation.npz")
    metadata.write_text(json.dumps(m))
    with pytest.raises(SedimentEventError) as info:
        load_checkpoint(path, args[1], args[2], args[6], expected)
    if kind == "schema1":
        assert "predates" in str(info.value) and SCHEMA in str(info.value)


# --- rerouting keeps the within-cell progress -------------------------------------------------------------------
def test_reroute_keeps_phase_progress_and_counts_steered_cells():
    from maple.core.parameters.topographic_commit import TopographicCommitSpec

    bed, ctx = make_bed(valley_elevation(4, 5), commit_spec=TopographicCommitSpec(commit_interval_s=10.0))
    inputs = setup(bed, ctx, south_ring=valley_elevation(5, 5)[0] - 0.01)
    state0 = inputs["state0"]
    r = run(inputs, constant_rainfall(0.0, 45.0, 72.0), 45.0, cadence=15.0)
    assert_closed(r, state0)
    assert r.n_commits >= 1 and r.n_graph_changes >= 1
    assert all("phase_steered_cells" in entry for entry in r.commit_log)
    assert 0 <= r.phase_steered_cells_total <= r.rerouted_cells_total
    st = r.state
    validate_phase_state(st.phase, st.bed.water.mobile_mass_by_cell_class_kg, st.network)
    # the last commit was forced at 45 s with mobile mass present: positions were NOT reset to the face
    mobile = st.bed.water.mobile_mass_by_cell_class_kg
    moment, _ = mass_weighted_position(st)
    assert np.any(mobile > 0.0) and np.any(moment[mobile.sum(-1) > 0.0] > 0.0)


# --- frozen benchmark: two-way transport implementation invariance ---------------------------------------------
@NUMBA_SKIP
def test_frozen_benchmark_transport_implementations_agree_to_roundoff():
    from maple.core.parameters.topographic_commit import TopographicCommitSpec

    bed, ctx = make_bed(valley_elevation(4, 5), commit_spec=TopographicCommitSpec(commit_interval_s=10.0))
    schedule = constant_rainfall(0.0, 30.0, 72.0)
    results = {}
    for impl in ("array", "numba"):
        control = frozen_control(StormControl(implementation="array"), transport_implementation=impl)
        inputs = setup(bed, ctx, south_ring=valley_elevation(5, 5)[0] - 0.01, control=control)
        results[impl], _ = run_frozen_event(inputs["state0"], ctx, inputs["column"], inputs["field"], schedule,
                                            inputs["vegetation"], inputs["sediment"], 30.0, control,
                                            report_every_s=10.0)
    a, b = results["array"], results["numba"]
    for name in ("depth_m", "soil_water_m", "discharge_m2_s"):  # hydrology is untouched by the transport kernel
        np.testing.assert_array_equal(getattr(a.state.storm, name), getattr(b.state.storm, name), err_msg=name)
    np.testing.assert_allclose(a.state.bed.active_layer.mass_kg, b.state.bed.active_layer.mass_kg, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(a.state.bed.water.mobile_mass_by_cell_class_kg,
                               b.state.bed.water.mobile_mass_by_cell_class_kg, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(a.sediment_hydrograph, b.sediment_hydrograph, rtol=1e-10, atol=1e-15)
    wa = a.state.bed.water.mobile_mass_by_cell_class_kg[..., None] * a.state.phase.fraction
    wb = b.state.bed.water.mobile_mass_by_cell_class_kg[..., None] * b.state.phase.fraction
    np.testing.assert_allclose(wa, wb, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(a.state.phase.position_m, b.state.phase.position_m, rtol=0, atol=1e-11)
    assert a.closure()["closed"] and b.closure()["closed"]


# --- the named upwind comparison scheme ------------------------------------------------------------------------
def test_upwind_scheme_is_explicit_and_differs_from_the_corrected_default():
    bed, ctx = make_bed(chain_elevation(6))
    schedule = constant_rainfall(0.0, 60.0, 72.0)
    upwind_control = SedimentEventControl(transport_scheme="upwind", sediment_courant_max=1.0)
    upwind_inputs = setup(bed, ctx, ksat=1e-5, control=upwind_control)
    default_inputs = setup(bed, ctx, ksat=1e-5)
    up = run(upwind_inputs, schedule, 180.0, control=upwind_control, cadence=60.0)
    ch = run(default_inputs, schedule, 180.0, cadence=60.0)
    assert up.state.phase is None and ch.state.phase is not None
    assert up.n_phase_canonicalized == up.n_phase_rounding_remnants == up.phase_steered_cells_total == 0
    assert float(up.max_phase_reconciliation_residual_kg) == 0.0
    assert_closed(up, upwind_inputs["state0"])
    assert_closed(ch, default_inputs["state0"])
    up_export = np.asarray(up.by_class["export_actual"])
    ch_export = np.asarray(ch.by_class["export_actual"])
    # the well-mixed operator leaks every class; the characteristic kernel exports none of the coarse classes
    assert np.all(up_export > 0.0) and np.all(ch_export[3:] == 0.0) and ch_export[0] > 0.0
    assert ch_export.sum() < up_export.sum()


def test_initial_mobile_load_is_a_canonical_partition_and_moves():
    bed, ctx = make_bed(chain_elevation(3), depth_m=0.002)
    inputs = setup(bed, ctx, ksat=0.0)
    state0 = inputs["state0"]
    pool = np.zeros_like(state0.bed.active_layer.mass_kg)
    pool[2, 0, 0] = 0.01
    from maple_syrup.sediment_bed import with_water

    wet_bed = with_water(state0.bed, ctx, state0.storm.depth_m, pool)
    state = initial_event_state(state0.graph, state0.terrain, wet_bed, ctx, state0.storm.soil_water_m)
    validate_phase_state(state.phase, pool, state.network)
    r = evolve_sediment_event(state, ctx, inputs["column"], inputs["field"], constant_rainfall(0.0, 5.0, 36.0),
                              inputs["vegetation"], inputs["sediment"], 5.0, SedimentEventControl(commit=False,
                                                                                                    force_final_commit=False),
                              report_every_s=5.0)
    assert_closed(r, state, check_dry_settling=False)
    moment, _ = mass_weighted_position(r.state)
    assert np.any(moment > 0.0)
