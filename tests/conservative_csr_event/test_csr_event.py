"""CW-B5: the accepted sparse CSR phase state inside the coupled actual-MAPLE event, completion, checkpoint and
runner layers (`phase_storage="csr"`), against the unchanged dense default.

CPU tests (always run; the CSR operator needs Numba and skips honestly without it): control / state pairing
refusals before any MAPLE call, dense-vs-CSR coupled evolving events on real MAPLE beds with EXACT arrays and
controls where the CPU arithmetic is identical (the accepted CSR Numba kernel is raw-byte equal to the dense Numba
kernel; the dense ARRAY kernel is compared at the T13 bound), closure and supply caps on both storages, a nonzero
initial pool as a canonical CSR partition, zero-actual row removal and rounding-remnant canonical rows through the
unchanged reconciliation bound, rerouting with explicit CSR rebind and conserved partition, pits / terminal masks,
pause-wet-disk-resume and the dry handoff with the EMPTY canonical CSR state, invalid / mismatched checkpoint
refusals with inputs untouched, an allocation guard that fails on ANY full `(.., B)` phase array during init /
event / checkpoint / completion on the CSR path, and byte-exact regression of the dense default's results and
checkpoint schema.

GPU tests (skip without a device, no claim): the CSR coupled event on CUDA against the CSR host event at the
declared coupled-event bounds (water rtol 2e-12 / atol 1e-14, sediment rtol 1e-10 / atol 1e-14), closure on the
device, same-device raw-byte repeat, wet CuPy checkpoint resume in the same backend.

Nothing here executes MAHLERAN or a full storm; nothing here was run by its author (file-only tools).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("maple")

from maple.core.backend import to_host
from maple_syrup.characteristic_sparse import (
    SparsePhaseState,
    canonical_sparse_phase_state,
    empty_sparse_phase_state,
    reconcile_sparse_rows,
    sparse_to_dense,
    validate_sparse_phase_state,
)
from test_complete_event import fixture as completion_fixture
from test_coupled_backends import (
    H,
    assert_closed_any,
    compare_bitwise,
    compare_events,
    setup_xp,
)
from test_sediment_event import (
    assert_closed,
    chain_elevation,
    make_bed,
    run,
    setup,
    valley_case,
    valley_elevation,
)

from maple_syrup import characteristic_numba_csr, sediment_event
from maple_syrup.characteristic_transport import (
    PhaseState,
    empty_phase_state,
    validate_phase_state,
)
from maple_syrup.checkpoint import SCHEMA, load_checkpoint, save_checkpoint, sha256_file
from maple_syrup.complete_event import DryHandoff, complete_event
from maple_syrup.event_backend import state_to_namespace
from maple_syrup.event_experiment import control_identity
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_bed import with_water
from maple_syrup.sediment_event import (
    PHASE_STORAGES,
    TRANSPORT_IMPLEMENTATIONS,
    SedimentEventControl,
    SedimentEventError,
    canonical_phase_for,
    evolve_sediment_event,
    initial_event_state,
    phase_for_network,
    reconcile_phase,
    sediment_coupled_step,
)
from maple_syrup.sediment_experiment import cuda_memory_preflight, phase_summary
from maple_syrup.storm import StormControl

IDENTITY ={"case": "controlled actual MAPLE bed", "forcing": "fixed", "source": "test", "options": {"storage": "csr"}}
# CW-E47C1: the control fields a DEFAULT dense control omits from its identity / checkpoint payload (each is an opt-in
# encoded only when set): CW-B5 storage, CW-D2 exchange history, CW-B102 operational deep publication
DEFAULT_OMITTED_CONTROL_KEYS = {"phase_storage", "phase_max_nnz", "record_exchange_history", "deep_publication",
                                "deep_publication_high_water_records"}
NUMBA = characteristic_numba_csr.numba_available()
needs_numba = pytest.mark.skipif(not NUMBA, reason="Numba not installed; the CSR operator cannot run (no fallback)")


def csr_control(**kwargs):
    kwargs.setdefault("storm", StormControl(implementation="numba" if NUMBA else "array"))
    return SedimentEventControl(phase_storage="csr", **kwargs)


def dense_control(**kwargs):
    kwargs.setdefault("storm", StormControl(implementation="numba" if NUMBA else "array"))
    return SedimentEventControl(**kwargs)


def with_dense_phase(result):
    """The result with a CSR phase replaced by its explicit small-domain dense expansion, so the shared comparators
    (which read `phase.fraction` / `phase.position_m` as `(ny, nx, nc, B)` arrays) can be reused. A comparison tool
    only: the event path itself never densifies."""
    phase = result.state.phase
    if isinstance(phase, SparsePhaseState):
        return replace(result, state=replace(result.state, phase=sparse_to_dense(phase)))
    return result


def _row_mobile(phase, state):
    """A mobile field consistent with the ROWS of `phase`: 1 kg on every stored row, 0 elsewhere (a validation
    fixture for row surgery; the physical pool is untouched)."""
    counts = np.diff(phase.row_ptr).reshape(phase.n_active, phase.n_classes)
    n = state.graph.shape[0] * state.graph.shape[1]
    field = np.zeros((n, phase.n_classes), dtype=np.float64)
    field[phase.cell_of_slot] = (counts > 0).astype(np.float64)
    return field.reshape(state.graph.shape[0], state.graph.shape[1], phase.n_classes)


def expanded_masses(state):
    """Phase masses `mobile * fraction` as a dense `(ny, nx, nc, B)` HOST array for comparison only."""
    phase = state.phase
    dense = sparse_to_dense(phase) if isinstance(phase, SparsePhaseState) else phase
    return H(state.bed.water.mobile_mass_by_cell_class_kg)[..., None] * H(dense.fraction), H(dense.position_m)


def assert_same_physics_exact(dense_result, csr_result):
    """Dense Numba vs CSR Numba on the SAME CPU inputs: every physical array, integer control and the expanded phase
    partition must be raw-byte EQUAL (the accepted raw-byte correspondence of the two kernels), and the CSR state
    must be the typed CSR."""
    assert isinstance(dense_result.state.phase, PhaseState) and isinstance(csr_result.state.phase, SparsePhaseState)
    compare_bitwise(dense_result, with_dense_phase(csr_result))
    wd, xd = expanded_masses(dense_result.state)
    wc, xc = expanded_masses(csr_result.state)
    assert wd.tobytes() == wc.tobytes() and xd.tobytes() == xc.tobytes()
    for name in ("n_phase_canonicalized", "n_phase_rounding_remnants", "phase_steered_cells_total"):
        assert getattr(dense_result, name) == getattr(csr_result, name), name
    assert float(dense_result.max_phase_reconciliation_residual_kg) == float(csr_result.max_phase_reconciliation_residual_kg)


class DenseAllocationGuard:
    """Fail if ANY array with the full phase shape `(ny, nx, nc, B)` or `(n, nc, B)` (or `(n_rows, B)`) is requested
    from NumPy (`zeros`/`empty`/`full`/`ones`/`zeros_like`/`empty_like`/`full_like`/`ones_like`) during the guarded
    region. New-module allocations of row width (`n_rows + 1`, `nnz`) and the `(n, nc)` contract fields pass."""

    def __init__(self, monkeypatch, shape, nc, nb):
        self.forbidden = {(shape[0], shape[1], nc, nb), (shape[0] * shape[1], nc, nb), (shape[0] * shape[1] * nc, nb)}
        self.requests: list[tuple] = []
        for name in ("zeros", "empty", "full", "ones"):
            real = getattr(np, name)

            def wrapped(shape_, *args, _real=real, **kwargs):
                self.check(shape_)
                return _real(shape_, *args, **kwargs)

            monkeypatch.setattr(np, name, wrapped)
        for name in ("zeros_like", "empty_like", "full_like", "ones_like"):
            real = getattr(np, name)

            def wrapped_like(a, *args, _real=real, **kwargs):
                self.check(getattr(a, "shape", ()))
                return _real(a, *args, **kwargs)

            monkeypatch.setattr(np, name, wrapped_like)

    def check(self, shape_):
        shape_ = tuple(np.atleast_1d(shape_).tolist()) if not isinstance(shape_, int) else (shape_,)
        self.requests.append(shape_)
        if shape_ in self.forbidden:
            raise AssertionError(f"a full dense phase array {shape_} was allocated on the CSR path")


# =====================================================================================================================
# control / state pairing
# =====================================================================================================================
def test_control_storage_option_defaults_and_refusals():
    assert PHASE_STORAGES == ("dense", "csr") and TRANSPORT_IMPLEMENTATIONS == ("auto", "array", "numba", "cuda")
    default = SedimentEventControl()
    assert default.phase_storage == "dense" and default.phase_max_nnz is None and not default.csr
    assert default.validated().resolved_transport_implementation() == "array"
    assert "phase_storage" not in default.transport_record() and "phase_storage" not in default.backend_record()
    # CW-E47C1: the identity omits EXACTLY the opt-in keys holding their defaults -- the CW-B5 storage pair, the CW-D2
    # history flag and the CW-B102 operational deep-publication pair ('step', None) -- so every default identity is
    # unchanged; every other field is present with its exact value (no catch-all)
    assert control_identity(default) == {k: v for k, v in dataclasses.asdict(default).items()
                                         if k not in DEFAULT_OMITTED_CONTROL_KEYS}
    assert set(dataclasses.asdict(default)) - set(control_identity(default)) == DEFAULT_OMITTED_CONTROL_KEYS
    assert default.deep_publication == "step" and default.deep_publication_high_water_records is None
    # a non-default publication is part of the identity, with both keys and exact types
    boundary = SedimentEventControl(phase_storage="csr", deep_publication="boundary",
                                    deep_publication_high_water_records=3_000_000).validated()
    ident = control_identity(boundary)
    assert ident["deep_publication"] == "boundary" and ident["deep_publication_high_water_records"] == 3_000_000
    assert type(ident["deep_publication_high_water_records"]) is int
    assert set(dataclasses.asdict(boundary)) - set(ident) == {"record_exchange_history"}
    assert ident != control_identity(SedimentEventControl(phase_storage="csr").validated())
    csr_numba = SedimentEventControl(phase_storage="csr", storm=StormControl(implementation="numba")).validated()
    assert csr_numba.csr and csr_numba.resolved_transport_implementation() == "numba"
    csr_array_sweep = SedimentEventControl(phase_storage="csr").validated()  # array hydraulics on a host state
    assert csr_array_sweep.resolved_transport_implementation() == "numba"  # the CSR operator has no array kernel
    csr_cuda = SedimentEventControl(phase_storage="csr", storm=StormControl(implementation="cuda")).validated()
    assert csr_cuda.resolved_transport_implementation() == "cuda" and csr_cuda.backend_record()["required_namespace"] == "cupy"
    assert csr_numba.transport_record()["phase_storage"] == "csr" and "storage_rule" in csr_numba.transport_record()
    assert "phase_storage" in control_identity(csr_numba) and control_identity(csr_numba)["phase_max_nnz"] is None
    explicit = SedimentEventControl(phase_storage="csr", transport_implementation="cuda",
                                    storm=StormControl(implementation="array")).validated()
    assert explicit.resolved_transport_implementation() == "cuda"  # explicit device operator on a CuPy state
    SedimentEventControl(phase_storage="csr", phase_max_nnz=0).validated()
    bad = {
        "unknown storage": {"phase_storage": "coo"},
        "upwind with csr": {"phase_storage": "csr", "transport_scheme": "upwind", "sediment_courant_max": 1.0},
        "csr with array kernel": {"phase_storage": "csr", "transport_implementation": "array"},
        "dense with cuda kernel": {"transport_implementation": "cuda"},
        "capacity without csr": {"phase_max_nnz": 10},
        "negative capacity": {"phase_storage": "csr", "phase_max_nnz": -1},
        "bool capacity": {"phase_storage": "csr", "phase_max_nnz": True},
        "csr numba under cuda sweep": {"phase_storage": "csr", "transport_implementation": "numba",
                                       "storm": StormControl(implementation="cuda")},
        # CW-B102 deep publication: strict values / types and pairing
        "unknown publication": {"phase_storage": "csr", "deep_publication": "eventually"},
        "boundary without high water": {"phase_storage": "csr", "deep_publication": "boundary"},
        "boundary on dense": {"deep_publication": "boundary", "deep_publication_high_water_records": 10},
        "bool high water": {"phase_storage": "csr", "deep_publication": "boundary",
                            "deep_publication_high_water_records": True},
        "float high water": {"phase_storage": "csr", "deep_publication": "boundary",
                             "deep_publication_high_water_records": 10.0},
        "high water under step": {"phase_storage": "csr", "deep_publication_high_water_records": 10},
    }
    for kwargs in bad.values():
        with pytest.raises(SedimentEventError):
            SedimentEventControl(**kwargs).validated()


@needs_numba
def test_state_and_control_storage_must_agree_before_any_maple_call(monkeypatch):
    bed, ctx = make_bed(chain_elevation(3))
    dense_inputs = setup(bed, ctx, control=dense_control())
    csr_inputs = setup(bed, ctx, control=csr_control())
    dense_state, csr_state = dense_inputs["state0"], csr_inputs["state0"]
    assert isinstance(dense_state.phase, PhaseState) and isinstance(csr_state.phase, SparsePhaseState)
    assert csr_state.phase.nnz == 0 and csr_state.phase.n_bins == 32 and csr_state.phase.network_sha256 == csr_state.network.graph_input_sha256
    calls = {"coupled_step": 0, "apply_bed_demand": 0, "characteristic_step": 0, "sparse_characteristic_step": 0}

    def counting(name):
        original = getattr(sediment_event, name)

        def wrapped(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        return wrapped

    for name in calls:
        monkeypatch.setattr(sediment_event, name, counting(name))
    rate = dense_inputs["field"].apply(72.0 / 3.6e6)

    def attempt(inputs, state, control):
        return sediment_coupled_step(state, ctx, inputs["column"], rate, inputs["vegetation"], inputs["sediment"], 1.0,
                                     control)

    with pytest.raises(SedimentEventError, match="SparsePhaseState"):
        attempt(dense_inputs, dense_state, csr_control())  # dense state under a CSR control: refused, not converted
    with pytest.raises(SedimentEventError, match="PhaseState"):
        attempt(csr_inputs, csr_state, dense_control())  # CSR state under a dense control
    with pytest.raises(SedimentEventError, match="different routing graph"):
        attempt(csr_inputs, replace(csr_state, phase=replace(csr_state.phase, network_sha256="0" * 64)), csr_control())
    with pytest.raises(SedimentEventError, match="bins"):
        attempt(csr_inputs, csr_state, csr_control(phase_bins=16))
    schedule = constant_rainfall(0.0, 10.0, 72.0)
    with pytest.raises(SedimentEventError, match="SparsePhaseState"):
        run(dense_inputs, schedule, 10.0, control=csr_control(), cadence=10.0)
    with pytest.raises(SedimentEventError, match="PhaseState"):
        run(csr_inputs, schedule, 10.0, control=dense_control(), cadence=10.0)
    assert all(count == 0 for count in calls.values()), calls
    # a corrupted CSR phase on the state is refused by the driver before any MAPLE call
    bad_phase = replace(csr_state.phase, bin=np.array(csr_state.phase.bin, copy=True))  # structurally fine...
    bad_phase = replace(bad_phase, row_ptr=np.array(csr_state.phase.row_ptr, copy=True) + 1)  # ... pointer corrupt
    with pytest.raises(SedimentEventError, match="phase state invalid"):
        run(csr_inputs, schedule, 10.0, control=csr_control(), cadence=10.0, state=replace(csr_state, phase=bad_phase))
    assert all(count == 0 for count in calls.values()), calls
    ok = attempt(csr_inputs, csr_state, csr_control())
    assert calls["sparse_characteristic_step"] == 1 and calls["characteristic_step"] == 0 and calls["apply_bed_demand"] == 2
    assert isinstance(ok.phase, SparsePhaseState) and ok.characteristic.implementation == "numba"
    validate_sparse_phase_state(ok.phase, ok.bed.water.mobile_mass_by_cell_class_kg, csr_state.network)


# =====================================================================================================================
# dense vs CSR coupled events on real MAPLE beds (CPU): exact where the arithmetic is identical
# =====================================================================================================================
@needs_numba
def test_dense_numba_and_csr_numba_coupled_valley_events_are_raw_byte_equal():
    inputs_dense = valley_case()
    bed, ctx = make_bed(valley_elevation(4, 5))
    inputs_csr = setup(bed, ctx, south_ring=valley_elevation(5, 5)[0] - 0.01, control=csr_control())
    schedule = constant_rainfall(0.0, 30.0, 72.0)
    dense = run(inputs_dense, schedule, 40.0, control=dense_control(), cadence=10.0)
    csr = run(inputs_csr, schedule, 40.0, control=csr_control(), cadence=10.0)
    assert_closed(dense, inputs_dense["state0"])
    assert_closed(csr, inputs_csr["state0"])
    assert_same_physics_exact(dense, csr)
    st = csr.state
    validate_sparse_phase_state(st.phase, st.bed.water.mobile_mass_by_cell_class_kg, st.network)
    assert st.phase.nnz > 0 and np.any(H(st.bed.water.mobile_mass_by_cell_class_kg) > 0.0)
    bc = {k: np.asarray(v) for k, v in csr.by_class.items()}
    assert np.all(bc["actual_pickup"] <= bc["requested_pickup"] * (1 + 4 * np.finfo(np.float64).eps))
    assert bc["deposition_actual"].sum() > 0.0 and bc["export_actual"].sum() > 0.0
    summary = phase_summary(csr)
    assert summary["phase_storage"] == "csr" and summary["phase_nnz_final"] == st.phase.nnz
    assert "phase_storage" not in phase_summary(dense)


@needs_numba
def test_csr_event_with_commits_reroutes_by_explicit_rebind_and_conserves_the_partition():
    from maple.core.parameters.topographic_commit import TopographicCommitSpec

    bed, ctx = make_bed(valley_elevation(4, 5), commit_spec=TopographicCommitSpec(commit_interval_s=10.0))
    assert ctx.topographic_commit_spec.pending_elevation_trigger_m == 0.00075  # MAPLE's default, unchanged

    def pair(rain_mm_h, end_s):
        dense_in = setup(bed, ctx, south_ring=valley_elevation(5, 5)[0] - 0.01, control=dense_control())
        csr_in = setup(bed, ctx, south_ring=valley_elevation(5, 5)[0] - 0.01, control=csr_control())
        schedule = constant_rainfall(0.0, end_s, rain_mm_h)
        d = run(dense_in, schedule, end_s, control=dense_control(), cadence=15.0)
        c = run(csr_in, schedule, end_s, control=csr_control(), cadence=15.0)
        assert_closed(c, csr_in["state0"])
        assert_same_physics_exact(d, c)
        return c, csr_in
    # CW-E47C1 NEGATIVE control (the original 45 s / 72 mm/h storm): accounting commits below MAPLE's net threshold
    # never reroute, so the CSR phase keeps its INITIAL network binding (no rebind is needed or made)
    below, below_in = pair(72.0, 45.0)
    assert below.n_commits >= 1 and below.n_reroutes == 0 and below.n_graph_changes == 0
    assert all(not e["rerouted"] and e["net_displacement_m"] < e["reroute_threshold_m"] for e in below.commit_log)
    assert below.state.network is below_in["state0"].network
    assert below.state.phase.network_sha256 == below_in["state0"].network.graph_input_sha256
    # a GENUINE net crossing (same actual valley, 600 mm/h for 90 s, default threshold): the refreshed network is
    # adopted and the CSR phase is explicitly rebound to it
    csr, csr_inputs = pair(600.0, 90.0)
    rerouted = [e for e in csr.commit_log if e["rerouted"]]
    assert rerouted and csr.n_reroutes == len(rerouted) and csr.n_commits > len(rerouted)
    assert all(e["net_displacement_m"] >= e["reroute_threshold_m"] == 0.00075 for e in rerouted)
    assert all(e["net_displacement_m"] < e["reroute_threshold_m"] for e in csr.commit_log if not e["rerouted"])
    st = csr.state
    assert st.phase.network_sha256 == st.network.graph_input_sha256 == st.graph.input_sha256  # rebound at the reroute
    assert st.graph.input_sha256 == rerouted[-1]["graph_input_sha256"] != csr_inputs["state0"].graph.input_sha256
    validate_sparse_phase_state(st.phase, st.bed.water.mobile_mass_by_cell_class_kg, st.network)
    # the explicit rebind keeps every stored bin / fraction / position; only the binding changes
    moved = phase_for_network(st.phase, st.network)
    for name in ("row_ptr", "bin", "fraction", "position_m"):
        np.testing.assert_array_equal(getattr(moved, name), getattr(st.phase, name))
    other = csr_inputs["state0"].network  # the initial graph: a different digest, same active set
    rebound = phase_for_network(st.phase, other)
    assert rebound.network_sha256 == other.graph_input_sha256 and rebound.nnz == st.phase.nnz
    with pytest.raises(SedimentEventError, match="different routing graph"):
        sediment_event._bind_phase_to_control(replace(st, phase=rebound), csr_control())


@needs_numba
def test_csr_initial_nonzero_pool_is_a_direct_canonical_partition_and_moves(monkeypatch):
    bed, ctx = make_bed(chain_elevation(3), depth_m=0.002)
    inputs = setup(bed, ctx, ksat=0.0, control=csr_control())
    state0 = inputs["state0"]
    pool = np.zeros_like(state0.bed.active_layer.mass_kg)
    pool[2, 0, 0] = 0.01
    pool[1, 0, 3] = 0.002
    wet_bed = with_water(state0.bed, ctx, state0.storm.depth_m, pool)
    guard = DenseAllocationGuard(monkeypatch, state0.graph.shape, pool.shape[-1], 32)
    state = initial_event_state(state0.graph, state0.terrain, wet_bed, ctx, state0.storm.soil_water_m, control=csr_control())
    monkeypatch.undo()
    assert state.phase.nnz == 2 and state.phase.bin.tolist() == [0, 0] and state.phase.fraction.tolist() == [1.0, 1.0]
    assert state.phase.position_m.tolist() == [0.0, 0.0]
    validate_sparse_phase_state(state.phase, pool, state.network)
    direct = canonical_sparse_phase_state(state.network, pool, 32)
    for name in ("row_ptr", "bin", "fraction", "position_m"):
        np.testing.assert_array_equal(getattr(direct, name), getattr(state.phase, name))
    assert all(len(s) < 3 or s[-1] != 32 for s in guard.requests)  # no (.., B) array during the CSR init
    r = evolve_sediment_event(state, ctx, inputs["column"], inputs["field"], constant_rainfall(0.0, 5.0, 36.0),
                              inputs["vegetation"], inputs["sediment"], 5.0,
                              csr_control(commit=False, force_final_commit=False), report_every_s=5.0)
    assert_closed(r, state, check_dry_settling=False)
    w, x = expanded_masses(r.state)
    assert np.any((w * x).sum(axis=-1) > 0.0)  # the initial load advanced from the upstream face
    # the dense default on the same initial pool is the same physics, raw-byte
    dense_state = initial_event_state(state0.graph, state0.terrain, wet_bed, ctx, state0.storm.soil_water_m,
                                      control=dense_control())
    d = evolve_sediment_event(dense_state, ctx, inputs["column"], inputs["field"], constant_rainfall(0.0, 5.0, 36.0),
                              inputs["vegetation"], inputs["sediment"], 5.0,
                              dense_control(commit=False, force_final_commit=False), report_every_s=5.0)
    assert_same_physics_exact(d, r)


# =====================================================================================================================
# reconciliation on rows: zero actual removes rows; rounding remnants add canonical rows; beyond the bound refused
# =====================================================================================================================
@needs_numba
def test_csr_reconciliation_rules_match_the_dense_rules_and_refuse_beyond_the_bound():
    bed, ctx = make_bed(valley_elevation(4, 5))
    inputs = setup(bed, ctx, south_ring=valley_elevation(5, 5)[0] - 0.01, control=csr_control())
    r = run(inputs, constant_rainfall(0.0, 30.0, 72.0), 5.0, control=csr_control(commit=False, force_final_commit=False),
            cadence=5.0)
    wet = r.state
    rate = inputs["field"].apply(72.0 / 3.6e6)
    attempt = sediment_coupled_step(wet, ctx, inputs["column"], rate, inputs["vegetation"], inputs["sediment"], 1.0,
                                    csr_control())
    actual = attempt.bed.water.mobile_mass_by_cell_class_kg
    same, diagnostics = reconcile_phase(attempt.characteristic, actual, np)
    for name in ("row_ptr", "bin", "fraction", "position_m"):
        np.testing.assert_array_equal(getattr(same, name), getattr(attempt.phase, name))
    assert float(diagnostics["residual_max_kg"]) <= 1e-12 and int(diagnostics["n_canonicalized"]) >= 0
    # beyond the FP bound: refused, nothing changed
    tampered = actual.copy()
    occupied = np.argwhere(actual > 0.0)[0]
    tampered[tuple(occupied)] += 1e-3
    with pytest.raises(SedimentEventError, match="beyond the declared FP64 bound"):
        reconcile_phase(attempt.characteristic, tampered, np)
    # zero-actual rows are removed; a rounding remnant gets the canonical bin: exercised through controlled row
    # surgery. A wet storm on this bed occupies EVERY row (root run: 120/120), so the empty row is created by the
    # first operation (drop row a), and the second operation drops a DIFFERENT row b while inserting the canonical
    # bin into the now-empty row a (distinct drops and adds, as the event-level rules can produce them).
    phase = attempt.phase
    counts = np.diff(phase.row_ptr)
    occupied_rows = np.flatnonzero(counts > 0)
    assert occupied_rows.size >= 2
    a, b = int(occupied_rows[0]), int(occupied_rows[1])
    before = {name: np.array(getattr(phase, name), copy=True) for name in ("row_ptr", "bin", "fraction", "position_m")}
    none = np.zeros(phase.n_rows, bool)
    drop_a = none.copy()
    drop_a[a] = True
    emptied = reconcile_sparse_rows(phase, drop_a, none)  # the zero-actual rule on row a
    c1 = np.diff(emptied.row_ptr)
    assert c1[a] == 0 and emptied.nnz == phase.nnz - counts[a]
    np.testing.assert_array_equal(np.delete(c1, a), np.delete(counts, a))
    drop_b = none.copy()
    drop_b[b] = True
    add_a = none.copy()
    add_a[a] = True
    changed = reconcile_sparse_rows(emptied, drop_b, add_a)  # drop b, canonical bin into the emptied row a
    c2 = np.diff(changed.row_ptr)
    assert c2[a] == 1 and c2[b] == 0 and changed.nnz == emptied.nnz - counts[b] + 1
    e = int(changed.row_ptr[a])
    assert changed.bin[e] == 0 and changed.fraction[e] == 1.0 and changed.position_m[e] == 0.0
    # every other row keeps its bins in order and its values through both operations
    for row in occupied_rows[2:]:
        a0, a1 = phase.row_ptr[row], phase.row_ptr[row + 1]
        b0, b1 = changed.row_ptr[row], changed.row_ptr[row + 1]
        np.testing.assert_array_equal(phase.bin[a0:a1], changed.bin[b0:b1])
        np.testing.assert_array_equal(phase.fraction[a0:a1], changed.fraction[b0:b1])
        np.testing.assert_array_equal(phase.position_m[a0:a1], changed.position_m[b0:b1])
    rows_changed = np.repeat(np.arange(changed.n_rows), c2)
    same_row = rows_changed[1:] == rows_changed[:-1]
    assert np.all(changed.bin[1:][same_row] > changed.bin[:-1][same_row])  # bins stay strictly increasing per row
    validate_sparse_phase_state(changed, _row_mobile(changed, wet), wet.network)  # a valid CSR partition of its rows
    # refusals: a row both dropped and added; a canonical bin onto a row that still stores bins
    with pytest.raises(Exception, match="both dropped"):
        reconcile_sparse_rows(phase, drop_a, drop_a)
    with pytest.raises(Exception, match="still stores bins"):
        reconcile_sparse_rows(phase, none, add_a)  # row a is occupied in `phase`
    # the inputs of every call are untouched
    for name, arr in before.items():
        np.testing.assert_array_equal(getattr(phase, name), arr)
    assert not changed.fraction.flags.writeable
    # the dense default makes the same decisions on the same step (counts and residual are identical)
    dense_inputs = setup(bed, ctx, south_ring=valley_elevation(5, 5)[0] - 0.01, control=dense_control())
    rd = run(dense_inputs, constant_rainfall(0.0, 30.0, 72.0), 5.0,
             control=dense_control(commit=False, force_final_commit=False), cadence=5.0)
    assert rd.n_phase_canonicalized == r.n_phase_canonicalized and rd.n_phase_rounding_remnants == r.n_phase_rounding_remnants


# =====================================================================================================================
# completion, checkpoint, restart on the CSR storage
# =====================================================================================================================
@needs_numba
def test_csr_dry_handoff_is_the_empty_canonical_state_and_round_trips(tmp_path, monkeypatch):
    args, kw = completion_fixture(wet=True, implementation="numba", phase_storage="csr")
    state0 = args[0]
    assert isinstance(state0.phase, SparsePhaseState)
    guard = DenseAllocationGuard(monkeypatch, state0.graph.shape, state0.bed.active_layer.mass_kg.shape[-1], 32)
    out = complete_event(*args, **kw)
    assert isinstance(out, DryHandoff)
    dry = out.dry_state
    assert isinstance(dry.phase, SparsePhaseState) and dry.phase.nnz == 0 and not np.any(dry.phase.row_ptr)
    assert dry.phase.network_sha256 == dry.network.graph_input_sha256
    assert not np.any(dry.bed.water.mobile_mass_by_cell_class_kg)
    assert out.progress.result.state.phase.nnz == 0  # canonical after the terminal deposition
    path = tmp_path / "handoff"
    save_checkpoint(path, out, args[1], args[2], args[6], IDENTITY)
    loaded = load_checkpoint(path, args[1], args[2], args[6], IDENTITY, allow_complete=True)
    monkeypatch.undo()
    assert isinstance(loaded.dry_state.phase, SparsePhaseState) and loaded.dry_state.phase.nnz == 0
    manifest = json.loads((path / "checkpoint.json").read_text())
    assert manifest["phase_storage"] == "csr" and manifest["schema"] == SCHEMA
    node = manifest["payload"]["fields"]["dry_state"]["state"]["phase"]
    assert set(node) == {"csr_phase"} and node["csr_phase"]["n_bins"] == 32
    assert not any(len(s) >= 3 and s[-1] == 32 for s in guard.requests)


@needs_numba
def test_csr_wet_checkpoint_resumes_identically_and_the_dense_default_is_unchanged(tmp_path):
    args, kw = completion_fixture(wet=True, implementation="numba", phase_storage="csr")
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    st = paused.result.state
    assert isinstance(st.phase, SparsePhaseState) and st.phase.nnz > 0 and np.any(st.phase.position_m > 0.0)
    path = tmp_path / "wet"
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    restored = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    rp = restored.result.state.phase
    assert isinstance(rp, SparsePhaseState) and not rp.fraction.flags.writeable
    for name in ("row_ptr", "bin", "fraction", "position_m", "cell_of_slot", "slot_of_cell"):
        np.testing.assert_array_equal(getattr(rp, name), getattr(st.phase, name))
    assert rp.network_sha256 == st.phase.network_sha256
    full = complete_event(*args, **kw)
    resumed = complete_event(restored.result.state, *args[1:], **kw, continuation=restored)
    np.testing.assert_array_equal(full.dry_state.bed.water.mobile_mass_by_cell_class_kg,
                                  resumed.dry_state.bed.water.mobile_mass_by_cell_class_kg)
    np.testing.assert_array_equal(full.dry_state.bed.active_layer.mass_kg, resumed.dry_state.bed.active_layer.mass_kg)
    np.testing.assert_array_equal(full.progress.result.sediment_hydrograph, resumed.progress.result.sediment_hydrograph)
    assert full.progress.result.n_phase_canonicalized == resumed.progress.result.n_phase_canonicalized
    # the dense default: same fixture, dense Numba, raw-byte equal physics and the UNCHANGED bundle encoding
    dargs, dkw = completion_fixture(wet=True, implementation="numba")
    dense_paused = complete_event(*dargs, **dkw, checkpoint_callback=lambda p: True)
    assert_same_physics_exact(dense_paused.result, paused.result)
    dpath = tmp_path / "dense"
    save_checkpoint(dpath, dense_paused, dargs[1], dargs[2], dargs[6], IDENTITY)
    dense_manifest = json.loads((dpath / "checkpoint.json").read_text())
    assert "phase_storage" not in dense_manifest
    control_node = dense_manifest["payload"]["fields"]["control"]["fields"]
    assert "phase_storage" not in control_node and "phase_max_nnz" not in control_node  # pre-CW-B5 encoding
    csr_manifest = json.loads((path / "checkpoint.json").read_text())
    csr_control_node = csr_manifest["payload"]["fields"]["control"]["fields"]
    assert csr_control_node["phase_storage"] == "csr" and csr_control_node["phase_max_nnz"] is None
    assert set(dense_manifest["payload"]["fields"]["result"]["fields"]["state"]["state"]["phase"]) == {"phase"}


@needs_numba
@pytest.mark.parametrize("kind", ["row_ptr", "bin", "fraction", "slot_map", "shape", "digest", "tag", "dense_node"])
def test_csr_checkpoint_refuses_malformed_nodes_without_partial_mutation(tmp_path, kind):
    args, kw = completion_fixture(wet=True, implementation="numba", phase_storage="csr")
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    path = tmp_path / "wet"
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    metadata = path / "checkpoint.json"
    m = json.loads(metadata.read_text())
    node = m["payload"]["fields"]["result"]["fields"]["state"]["state"]["phase"]["csr_phase"]
    with np.load(path / "continuation.npz") as archive:
        data = {k: archive[k].copy() for k in archive.files}
    if kind == "row_ptr":
        data[node["row_ptr"]["array"]][-1] += 1  # end pointer no longer the entry count
    elif kind == "bin":
        data[node["bin"]["array"]][0] = 32  # bin >= B
    elif kind == "fraction":
        data[node["fraction"]["array"]][0] *= 0.5  # row no longer sums to 1
    elif kind == "slot_map":
        data[node["cell_of_slot"]["array"]] = data[node["cell_of_slot"]["array"]][::-1].copy()
    elif kind == "shape":
        node["shape"] = [node["shape"][0] + 1, node["shape"][1]]
    elif kind == "digest":
        node["graph_digest"] = "0" * 64
    elif kind == "tag":
        m["phase_storage"] = "dense"  # tag disagrees with the typed node
    else:
        m["payload"]["fields"]["result"]["fields"]["state"]["state"]["phase"] = {"phase": {
            "fraction": node["fraction"], "position_m": node["position_m"], "dx_m": node["dx_m"], "n_bins": node["n_bins"]}}
    np.savez(path / "continuation.npz", **data)
    m["files"]["continuation.npz"] = sha256_file(path / "continuation.npz")
    metadata.write_text(json.dumps(m))
    before = paused.result.state.phase.fraction.copy()
    with pytest.raises(SedimentEventError):
        load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    np.testing.assert_array_equal(paused.result.state.phase.fraction, before)


@needs_numba
def test_dense_checkpoint_identity_and_schema_are_byte_identical_to_the_pre_csr_encoding(tmp_path):
    """The default (dense) path must regress byte-exact: the control node, the phase node, the manifest keys and the
    identity dict carry no CW-B5 key, and a bundle written with the dense default loads with a plain dense control."""
    args, kw = completion_fixture(wet=True, implementation="numba")
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    path = tmp_path / "dense"
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    m = json.loads((path / "checkpoint.json").read_text())
    assert set(m) == {"schema", "identity", "status", "files", "payload", "backend"}
    control_fields = m["payload"]["fields"]["control"]["fields"]
    # CW-E47C1: exactly the default-valued opt-in control fields are omitted (CW-B5 storage pair, CW-D2 history, CW-B102
    # deep-publication pair), as `checkpoint._OPTIONAL_DEFAULT_FIELDS` / `_CSR_CONTROL_FIELDS` declare
    assert set(control_fields) == {f.name for f in dataclasses.fields(SedimentEventControl)} - DEFAULT_OMITTED_CONTROL_KEYS
    assert paused.control.deep_publication == "step" and paused.control.deep_publication_high_water_records is None
    from maple_syrup.checkpoint import _CSR_CONTROL_FIELDS, _OPTIONAL_DEFAULT_FIELDS
    assert DEFAULT_OMITTED_CONTROL_KEYS == set(_CSR_CONTROL_FIELDS) | set(_OPTIONAL_DEFAULT_FIELDS["SedimentEventControl"])
    restored = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    assert restored.control == paused.control and restored.control.phase_storage == "dense"
    assert restored.control.deep_publication == "step" and restored.control.deep_publication_high_water_records is None
    assert isinstance(restored.result.state.phase, PhaseState)
    assert control_identity(paused.control) == {k: v for k, v in dataclasses.asdict(paused.control).items()
                                                if k not in DEFAULT_OMITTED_CONTROL_KEYS}
    # strict decoding is kept: an unknown control field, or a wrongly typed explicit publication, is refused
    def edited(name, key, value):
        bad = tmp_path / name
        save_checkpoint(bad, paused, args[1], args[2], args[6], IDENTITY)
        doc = json.loads((bad / "checkpoint.json").read_text())
        doc["payload"]["fields"]["control"]["fields"][key] = value
        (bad / "checkpoint.json").write_text(json.dumps(doc))
        return bad
    with pytest.raises(SedimentEventError, match="unknown checkpoint type or fields"):
        load_checkpoint(edited("unknown_key", "not_a_field", 1), args[1], args[2], args[6], IDENTITY)
    with pytest.raises(SedimentEventError, match="deep_publication must be one of"):
        load_checkpoint(edited("typed_publication", "deep_publication", 7), args[1], args[2], args[6], IDENTITY)
    # an explicitly PRESENT default value is decoded and validated like any field (absent == default)
    explicit = load_checkpoint(edited("explicit_step", "deep_publication", "step"), args[1], args[2], args[6], IDENTITY)
    assert explicit.control == paused.control


# =====================================================================================================================
# pits / terminal masks on the CSR storage
# =====================================================================================================================
@needs_numba
def test_csr_pit_chain_deposits_every_grain_at_the_terminal_and_matches_dense():
    from pit_helpers import pit_inputs as pits

    control_d = SedimentEventControl(storm=StormControl(implementation="numba"))
    control_c = SedimentEventControl(storm=StormControl(implementation="numba"), phase_storage="csr")
    dense_in = pits(np, control=control_d)
    csr_in = pits(np, control=control_c)
    assert csr_in["state0"].network.terminal_flat is not None and isinstance(csr_in["state0"].phase, SparsePhaseState)
    schedule = constant_rainfall(0.0, 30.0, 72.0)
    d = evolve_sediment_event(dense_in["state0"], dense_in["ctx"], dense_in["column"], dense_in["field"], schedule,
                              dense_in["vegetation"], dense_in["sediment"], 40.0, control_d, report_every_s=10.0)
    c = evolve_sediment_event(csr_in["state0"], csr_in["ctx"], csr_in["column"], csr_in["field"], schedule,
                              csr_in["vegetation"], csr_in["sediment"], 40.0, control_c, report_every_s=10.0)
    assert_closed_any(c, csr_in["state0"], check_dry_settling=False)
    assert_same_physics_exact(d, c)
    pit = np.asarray(csr_in["graph"].pit_storage, dtype=bool)
    assert float(np.asarray(c.by_class["export_actual"]).sum()) == 0.0
    assert np.all(H(c.state.bed.water.mobile_mass_by_cell_class_kg)[pit] == 0.0)  # the terminal never holds load
    counts = np.diff(c.state.phase.row_ptr).reshape(-1, c.state.phase.n_classes)
    pit_slot = np.flatnonzero(pit.reshape(-1)[c.state.phase.cell_of_slot])
    assert np.all(counts[pit_slot] == 0)  # no stored bins on terminal rows


# =====================================================================================================================
# allocation guard over init / event / checkpoint / completion on the CSR path
# =====================================================================================================================
@needs_numba
def test_csr_path_never_allocates_a_full_phase_array(monkeypatch, tmp_path):
    bed, ctx = make_bed(valley_elevation(4, 5))
    nb = 32
    guard = DenseAllocationGuard(monkeypatch, bed.active_layer.mass_kg.shape[:2], bed.active_layer.mass_kg.shape[-1], nb)
    inputs = setup(bed, ctx, south_ring=valley_elevation(5, 5)[0] - 0.01, control=csr_control(phase_bins=nb))
    r = run(inputs, constant_rainfall(0.0, 20.0, 72.0), 30.0, control=csr_control(phase_bins=nb), cadence=10.0)
    assert r.state.phase.nnz > 0
    args, kw = completion_fixture(wet=True, implementation="numba", phase_storage="csr")
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    path = tmp_path / "wet"
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    restored = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    out = complete_event(restored.result.state, *args[1:], **kw, continuation=restored)
    assert isinstance(out, DryHandoff) and out.dry_state.phase.nnz == 0
    monkeypatch.undo()
    assert len(guard.requests) > 0 and not any(s in guard.forbidden for s in guard.requests)
    # the dense default DOES allocate the full phase array (the guard is meaningful)
    guard2 = DenseAllocationGuard(monkeypatch, bed.active_layer.mass_kg.shape[:2], bed.active_layer.mass_kg.shape[-1], nb)
    with pytest.raises(AssertionError, match="full dense phase array"):
        setup(bed, ctx, south_ring=valley_elevation(5, 5)[0] - 0.01, control=dense_control(phase_bins=nb))
    monkeypatch.undo()
    assert guard2.requests


def test_csr_preflight_reports_capacity_and_full_occupancy_separately(monkeypatch):
    """Host-only: the device preflight distinguishes the CSR capacity / full-occupancy bytes from the dense B-bin
    allocation and reports the operator's capacity rule; the dense record is unchanged."""
    from maple.core import backend
    from test_coupled_backends import fake_cupy, preflight_case

    bed, ctx = make_bed(chain_elevation(4))
    case = preflight_case(bed, ctx)
    monkeypatch.setattr(backend, "cupy_module", lambda: fake_cupy(1 << 40))
    dense = cuda_memory_preflight(case, phase_bins=32, device_id=0)
    assert "phase_storage" not in dense and dense["phase_state_bytes"] == 2 * 4 * 6 * 32 * 8
    csr = cuda_memory_preflight(case, phase_bins=32, device_id=0, phase_storage="csr")
    assert csr["phase_storage"] == "csr" and csr["csr_full_occupancy_entries"] == 4 * 6 * 32
    assert csr["csr_capacity_basis"].startswith("full occupancy") and csr["csr_capacity_required_by_operator"] is False
    capped = cuda_memory_preflight(case, phase_bins=32, device_id=0, phase_storage="csr", phase_max_nnz=10)
    assert capped["csr_capacity_entries"] == 10 and capped["csr_capacity_basis"] == "phase_max_nnz"
    assert capped["phase_bytes"] < csr["phase_bytes"] < dense["phase_bytes"]
    for kwargs in ({"phase_storage": "coo"}, {"phase_storage": "csr", "transport_scheme": "upwind"},
                   {"phase_max_nnz": 3}, {"phase_storage": "csr", "phase_max_nnz": -1}):
        with pytest.raises(SedimentEventError):
            cuda_memory_preflight(case, phase_bins=32, device_id=0, **kwargs)


# =====================================================================================================================
# GPU: the CSR coupled event on CUDA against the CSR host event
# =====================================================================================================================
@needs_numba
def test_csr_cuda_event_matches_the_csr_host_event_and_repeats_raw_byte(gpu):
    cp = gpu
    bed, ctx = make_bed(valley_elevation(4, 5))
    host_control = SedimentEventControl(storm=StormControl(implementation="numba"), phase_storage="csr")
    dev_control = SedimentEventControl(storm=StormControl(implementation="cuda"), phase_storage="csr")
    host_in = setup_xp(bed, ctx, np, south_ring=valley_elevation(5, 5)[0] - 0.01, control=host_control)
    dev_in = setup_xp(bed, ctx, cp, south_ring=valley_elevation(5, 5)[0] - 0.01, control=dev_control)
    assert isinstance(dev_in["state0"].phase, SparsePhaseState) and dev_in["state0"].phase.xp is cp
    schedule = constant_rainfall(0.0, 30.0, 72.0)
    ref = evolve_sediment_event(host_in["state0"], host_in["ctx"], host_in["column"], host_in["field"], schedule,
                                host_in["vegetation"], host_in["sediment"], 40.0, host_control, report_every_s=10.0)
    cand = evolve_sediment_event(dev_in["state0"], dev_in["ctx"], dev_in["column"], dev_in["field"], schedule,
                                 dev_in["vegetation"], dev_in["sediment"], 40.0, dev_control, report_every_s=10.0)
    assert_closed_any(cand, dev_in["state0"])
    compare_events(with_dense_phase(ref), with_dense_phase(cand))
    assert cand.state.phase.xp is cp and type(cand.state.phase.fraction) is cp.ndarray
    assert cand.state.phase.nnz == ref.state.phase.nnz and to_host(cand.state.phase.bin).tobytes() == ref.state.phase.bin.tobytes()
    again = evolve_sediment_event(dev_in["state0"], dev_in["ctx"], dev_in["column"], dev_in["field"], schedule,
                                  dev_in["vegetation"], dev_in["sediment"], 40.0, dev_control, report_every_s=10.0)
    compare_bitwise(with_dense_phase(cand), with_dense_phase(again))
    assert to_host(again.state.phase.fraction).tobytes() == to_host(cand.state.phase.fraction).tobytes()
    # mixed pairings are refused before any kernel: CSR numba on the device state, CSR cuda on the host state
    with pytest.raises(SedimentEventError):
        evolve_sediment_event(dev_in["state0"], dev_in["ctx"], dev_in["column"], dev_in["field"], schedule,
                              dev_in["vegetation"], dev_in["sediment"], 40.0, host_control, report_every_s=10.0)
    with pytest.raises(SedimentEventError):
        evolve_sediment_event(host_in["state0"], host_in["ctx"], host_in["column"], host_in["field"], schedule,
                              host_in["vegetation"], host_in["sediment"], 40.0,
                              SedimentEventControl(phase_storage="csr", transport_implementation="cuda"), report_every_s=10.0)


@needs_numba
def test_csr_cuda_wet_checkpoint_resumes_in_the_same_backend(gpu, tmp_path):
    from test_complete_backend import fixture_xp

    cp = gpu
    args, kw = fixture_xp(cp, wet=True, implementation="cuda", phase_storage="csr")
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    st = paused.result.state
    assert isinstance(st.phase, SparsePhaseState) and st.phase.xp is cp and st.phase.nnz > 0
    path = tmp_path / "wet_cupy"
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    manifest = json.loads((path / "checkpoint.json").read_text())
    assert manifest["backend"] == "cupy" and manifest["phase_storage"] == "csr"
    restored = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    rp = restored.result.state.phase
    assert isinstance(rp, SparsePhaseState) and rp.xp is cp
    for name in ("row_ptr", "bin", "fraction", "position_m"):
        assert to_host(getattr(rp, name)).tobytes() == to_host(getattr(st.phase, name)).tobytes()
    full = complete_event(*args, **kw)
    resumed = complete_event(restored.result.state, *args[1:], **kw, continuation=restored)
    assert to_host(full.dry_state.bed.water.mobile_mass_by_cell_class_kg).tobytes() == to_host(
        resumed.dry_state.bed.water.mobile_mass_by_cell_class_kg).tobytes()
    assert resumed.dry_state.phase.nnz == 0 and full.dry_state.phase.nnz == 0
    # the explicit boundary conversion moves the CSR arrays with counted transfers and rebinds to the host network
    host_state = state_to_namespace(restored.result.state, np)
    assert isinstance(host_state.phase, SparsePhaseState) and host_state.phase.xp is np
    assert host_state.phase.network_sha256 == host_state.network.graph_input_sha256
    validate_sparse_phase_state(host_state.phase, host_state.bed.water.mobile_mass_by_cell_class_kg, host_state.network)
    validate_phase_state(sparse_to_dense(host_state.phase), host_state.bed.water.mobile_mass_by_cell_class_kg,
                         host_state.network)  # the small explicit conversion is a test tool only


def test_dry_canonical_phase_helpers_dispatch_by_type():
    bed, ctx = make_bed(chain_elevation(3))
    dense_state = setup(bed, ctx, control=dense_control())["state0"]
    csr_state = setup(bed, ctx, control=csr_control())["state0"] if NUMBA else None
    assert isinstance(canonical_phase_for(dense_state), PhaseState)
    if csr_state is not None:
        canon = canonical_phase_for(csr_state)
        assert isinstance(canon, SparsePhaseState) and canon.nnz == 0
        assert phase_for_network(csr_state.phase, csr_state.network) is not None
    upwind = setup(bed, ctx, control=SedimentEventControl(transport_scheme="upwind", sediment_courant_max=1.0))["state0"]
    assert canonical_phase_for(upwind) is None and phase_for_network(None, upwind.network) is None
    assert empty_sparse_phase_state(dense_state.network, 6, 32).nnz == 0
    assert isinstance(empty_phase_state(dense_state.graph.shape, 6, 32, 0.5), PhaseState)
