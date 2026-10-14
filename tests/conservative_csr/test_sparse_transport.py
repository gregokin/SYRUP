"""CW-B2 candidate: sparse nonempty-bin CSR characteristic transport (CPU/Numba) against the
immutable dense kernels and the Phase 7b equation-level references.

Comparator policy (cw_b_design_corrected_report.md section 6.1): the CSR Numba
reference is EXPECTED to equal the dense Numba kernel BITWISE on identical
inputs. The assertion is bitwise and fails on any difference; the failure
message then carries, as a diagnostic only, whether the difference would have
been inside the predeclared T13 bounds (masses rtol 1e-12 / atol 1e-16,
positions atol 1e-12). The assertion is never relaxed here; a failure is
reported to the root for disposition. The dense ARRAY path is compared at the
T13 bounds, as T13 itself does.

Equation-level references (T01-T12b of tests/phase7b) are re-stated here for
the sparse state; they are not MAHLERAN executions. No event, MAPLE bed
transaction, checkpoint, GPU or production module is exercised.
"""

from __future__ import annotations

import builtins
import dataclasses
import math
import tracemalloc

import numpy as np
import pytest

pytest.importorskip("maple")

from maple_syrup.characteristic_sparse import (
    DEFAULT_DENSE_MAX_BYTES,
    DEFAULT_IMPLEMENTATION,
    ROW_SUM_CHUNK_ROWS,
    ROW_SUM_EMULATION_VERIFIED_NUMPY_VERSIONS,
    ROW_SUM_SELF_CHECK_BINS,
    DenseConversionRefused,
    RowSumEmulationMismatch,
    SparseCapacityError,
    SparsePhaseState,
    dense_phase_bytes,
    empty_sparse_phase_state,
    row_sum_emulation_self_check,
    row_sum_policy,
    row_sums_numpy,
    slot_maps,
    sparse_characteristic_step,
    sparse_from_dense,
    sparse_state_bytes,
    sparse_to_dense,
    validate_sparse_phase_state,
)

from maple_syrup import characteristic_numba_csr as csr_kernels
from maple_syrup.characteristic_transport import (
    DEFAULT_COURANT_MAX,
    DEFAULT_N_BINS,
    bin_index,
    characteristic_step,
    empty_phase_state,
    fraction_sum_tolerance,
)
from maple_syrup.routing import build_routing_graph
from maple_syrup.sediment_transport import (
    TransportError,
    TransportNetwork,
    TransportStep,
    TransportStepRejected,
    transport_network,
)

EPS = np.finfo(np.float64).eps
DX = 0.5
needs_numba = pytest.mark.skipif(not csr_kernels.numba_available(),
                                 reason="Numba is not installed; the compiled CSR reference cannot run (no fallback)")

T13_FIELDS = ("mobile_after_transfer_kg", "deposition_request_kg", "export_request_kg", "settled_kg",
              "internal_transfer_in_kg", "internal_transfer_out_kg", "y_face_net_kg", "x_face_net_kg")
TRANSPORT_FIELDS = tuple(f.name for f in dataclasses.fields(TransportStep) if f.name not in ("dt_s", "n_substeps"))


# --- terrain (same constructions as tests/phase7b) --------------------------------------------------
def chain_full(n: int, dz: float = 0.015625, walls: float = 1.0) -> np.ndarray:
    z = np.repeat(np.arange(n + 2, dtype=np.float64)[:, None] * dz, 3, axis=1)
    z[:, 0] += walls
    z[:, 2] += walls
    return z


def valley_full(ny: int, nx: int, sy: float = 0.015625, sx: float = 0.03125) -> np.ndarray:
    rows, cols = np.indices((ny + 2, nx + 2)).astype(np.float64)
    return rows * sy + np.abs(cols - (nx + 1) / 2) * sx


def west_plane_full(ny: int, nx: int, dz: float = 0.015, wall: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    z = np.repeat(np.arange(nx + 2, dtype=np.float64)[None, :] * dz, ny + 2, axis=0)
    z[[0, -1], :] += wall
    export = np.zeros(z.shape, dtype=bool)
    export[:, 0] = True
    return z, export


def random_full(rng, ny: int, nx: int) -> np.ndarray:
    rows = np.indices((ny + 2, nx + 2))[0].astype(np.float64)
    z = 0.025 * rows + rng.uniform(0.0, 0.02, rows.shape)
    z[:, 0] += 10.0
    z[:, -1] += 10.0
    return z


def south_export(z: np.ndarray) -> np.ndarray:
    export = np.zeros(z.shape, dtype=bool)
    export[0, :] = True
    return export


def make_graph(z, *, export=None, dx=DX, active=None):
    ny, nx = z.shape[0] - 2, z.shape[1] - 2
    export = south_export(z) if export is None else export
    return build_routing_graph(z, export, np.full((ny, nx), 1.0), dx, active_mask=active)


def chain(n, *, dx=DX):
    graph = make_graph(chain_full(n), dx=dx)
    return graph, transport_network(graph)


def fields(shape, nc, *, v=0.0, rate=0.0, settle=False):
    ny, nx = shape
    return (np.full((ny, nx, nc), float(v)), np.full((ny, nx, nc), float(rate)),
            np.full((ny, nx, nc), bool(settle)))


def zeros(shape, nc):
    return np.zeros((*shape, nc), dtype=np.float64)


# --- comparator ----------------------------------------------------------------------------------------
def _within_t13(a, b, *, position=False) -> bool:
    if position:
        return bool(np.all(np.abs(a - b) <= 1e-12))
    return bool(np.all(np.abs(a - b) <= 1e-16 + 1e-12 * np.abs(b)))


def assert_matches_dense_bitwise(sparse, dense) -> None:
    """Sparse CSR step versus the dense Numba step on the same inputs: every
    transport field, the remaining pool, the expanded phase masses, fractions
    and positions must be IDENTICAL. The T13 bound is reported as a diagnostic
    in the failure message only."""
    mismatches = []

    def compare(name, a, b, *, position=False):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            mismatches.append(f"{name}: shape {a.shape} != {b.shape}")
        elif not np.array_equal(a, b):
            diff = float(np.max(np.abs(a - b)))
            mismatches.append(f"{name}: max|sparse - dense| = {diff:.3e}; within the T13 bound: "
                              f"{_within_t13(a, b, position=position)}")

    for name in TRANSPORT_FIELDS:
        compare(name, getattr(sparse.transport, name), getattr(dense.transport, name))
    assert sparse.transport.n_substeps == dense.transport.n_substeps
    assert sparse.transport.dt_s == dense.transport.dt_s
    compare("mobile_remaining_kg", sparse.mobile_remaining_kg, dense.mobile_remaining_kg)
    compare("mobile_before_kg", sparse.mobile_before_kg, dense.mobile_before_kg)
    compare("arrival_deposition_kg", sparse.arrival_deposition_kg, dense.arrival_deposition_kg)
    compare("crossing_kg", sparse.crossing_kg, dense.crossing_kg)
    expanded = sparse_to_dense(sparse.phase_after)
    compare("phase fraction", expanded.fraction, dense.phase_after.fraction)
    compare("phase masses", sparse.mobile_remaining_kg[..., None] * expanded.fraction,
            dense.mobile_remaining_kg[..., None] * dense.phase_after.fraction)
    compare("phase position_m", expanded.position_m, dense.phase_after.position_m, position=True)
    assert not mismatches, ("sparse CSR Numba differs from dense Numba (bitwise expectation, root disposition "
                            "required):\n" + "\n".join(mismatches))


class Pair:
    """Advance the sparse and the dense Numba steps side by side from the same
    inputs and assert bitwise agreement after every step."""

    def __init__(self, network, n_bins, nc):
        self.net = network
        self.shape = network.shape
        self.nc = nc
        self.sp_mobile = zeros(self.shape, nc)
        self.sp_phase = empty_sparse_phase_state(network, nc, n_bins)
        self.de_mobile = zeros(self.shape, nc)
        self.de_phase = empty_phase_state(self.shape, nc, n_bins, network.dx_m)
        self.dep = zeros(self.shape, nc)
        self.export = zeros(self.shape, nc)
        self.crossing = zeros(self.shape, nc)
        self.nnz_history: list[int] = []

    def step(self, pickup, v, rate, settle, dt, **kwargs):
        sp = sparse_characteristic_step(self.net, self.sp_mobile, self.sp_phase, pickup, v, rate, settle, dt,
                                        **kwargs)
        de = characteristic_step(self.net, self.de_mobile, self.de_phase, pickup, v, rate, settle, dt,
                                 implementation="numba", **kwargs)
        assert_matches_dense_bitwise(sp, de)
        self.sp_mobile, self.sp_phase = sp.mobile_remaining_kg, sp.phase_after
        self.de_mobile, self.de_phase = de.mobile_remaining_kg, de.phase_after
        self.dep = self.dep + sp.transport.deposition_request_kg
        self.export = self.export + sp.transport.export_request_kg
        self.crossing = self.crossing + sp.crossing_kg
        self.nnz_history.append(sp.nnz_after)
        return sp

    @property
    def mobile(self):
        return self.sp_mobile

    @property
    def phase(self):
        return self.sp_phase

    def total(self):
        return float(self.dep.sum() + self.export.sum() + self.sp_mobile.sum())


def _sparse_identity_checks(result, network, before, dx):
    tr = result.transport
    ny, nx = network.shape
    assert tr.x_face_gross_kg.shape == (ny, nx + 1, before.shape[-1])
    assert tr.y_face_gross_kg.shape == (ny + 1, nx, before.shape[-1])
    np.testing.assert_allclose(tr.mobile_after_transfer_kg - before, tr.divergence_kg, rtol=0,
                               atol=1e-12 * max(1.0, float(before.max())))
    assert np.all(tr.export_request_kg[~network.outlet_host] == 0.0)
    assert np.all(tr.deposition_request_kg + tr.export_request_kg <= tr.mobile_after_transfer_kg * (1 + 4 * EPS))
    gross = tr.x_face_gross_kg.sum(axis=(0, 1)) + tr.y_face_gross_kg.sum(axis=(0, 1))
    np.testing.assert_allclose(gross, (tr.internal_transfer_out_kg + tr.export_request_kg).sum(axis=(0, 1)),
                               rtol=1e-12, atol=1e-15)
    assert np.all(np.abs(tr.budget_residual_by_class_kg) <= tr.budget_tolerance_by_class_kg)
    np.testing.assert_allclose(tr.mobile_before_by_class_kg, before.sum(axis=(0, 1)), rtol=1e-13)
    ph = result.phase_after
    validate_sparse_phase_state(ph, result.mobile_remaining_kg, network)
    counts = np.diff(ph.row_ptr)
    remaining = result.mobile_remaining_kg.reshape(-1, ph.n_classes)[ph.cell_of_slot].reshape(-1)
    assert np.array_equal(counts > 0, remaining > 0.0)
    if ph.nnz:
        rows = np.repeat(np.arange(ph.n_rows), counts)
        sums = np.bincount(rows, weights=ph.fraction, minlength=ph.n_rows)
        assert np.all(np.abs(sums[counts > 0] - 1.0) <= fraction_sum_tolerance(ph.n_bins))
        assert np.all(ph.position_m >= 0.0) and np.all(ph.position_m < dx)
        assert np.all(bin_index(ph.position_m, dx, ph.n_bins, np) == ph.bin.astype(np.int64))
        same = rows[1:] == rows[:-1]
        assert np.all(ph.bin[1:][same] > ph.bin[:-1][same])


# --- reduction-order probes for the row sums -------------------------------------------------------------
def _csr_rows(n_bins, n_rows=1200, seed=0):
    """Random sparse rows (35 % occupancy, 33 decades of magnitude), one full
    row, one empty row, plus the CSR arrays of the nonzero entries."""
    rng = np.random.default_rng(seed + n_bins)
    dense = np.where(rng.uniform(size=(n_rows, n_bins)) < 0.35, 10.0 ** rng.uniform(-30, 3, (n_rows, n_bins)), 0.0)
    dense[0] = 10.0 ** rng.uniform(-30, 3, n_bins)
    dense[1] = 0.0
    rows, bins = np.nonzero(dense > 0.0)
    counts = np.bincount(rows, minlength=n_rows)
    row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    return dense, row_ptr, bins.astype(np.uint8), np.ascontiguousarray(dense[rows, bins])


@pytest.mark.parametrize("n_bins", [1, 3, 7, 8, 9, 16, 32, 100, 128])
def test_row_sums_numpy_match_the_dense_reduction(n_bins):
    """The bounded NumPy REFERENCE (`row_sums_numpy`, NumPy on a chunked
    scratch) must equal `W.sum(axis=-1)` of the dense `(n, nc, B)` layout
    bitwise, for row counts below, equal to and above the chunk, and for a
    class axis of length 1 (the chain fixtures)."""
    assert ROW_SUM_CHUNK_ROWS == 512
    for n_rows in (6, 510, 516, 1200):  # below, just below, straddling and spanning several chunks
        dense, row_ptr, bins, masses = _csr_rows(n_bins, n_rows=n_rows)
        sums = row_sums_numpy(row_ptr, bins, masses, n_bins)
        for shape in ((n_rows // 3, 3, n_bins), (n_rows, 1, n_bins), (n_rows // 2, 2, n_bins)):
            reference = dense.reshape(shape).sum(axis=-1).reshape(-1)
            assert np.array_equal(sums, reference), (n_rows, shape, np.max(np.abs(sums - reference)))


@needs_numba
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("n_bins", ROW_SUM_SELF_CHECK_BINS)
def test_compiled_row_sum_emulation_matches_numpy_reduction(n_bins, signed):
    """The compiled `_row_sums` (PRODUCTION row sums) emulates NumPy's pairwise
    reduction. Raw bytes must equal the real dense `(rows/k, k, B).sum(axis=-1)`
    for 1..6 classes and the bounded NumPy reference, on every supported bin
    edge (powers of two 1..128 and their neighbours, 100), positive and signed
    values spanning 600 decades, with full, empty and single-entry rows. The
    running NumPy version is part of any failure message: equality is a
    property of this NumPy, not a universal assumption."""
    rng = np.random.default_rng(1000 + 2 * n_bins + int(signed))
    n_rows = 600  # divisible by 1, 2, 3, 4, 5, 6
    magnitude = 10.0 ** rng.uniform(-300.0, 300.0, (n_rows, n_bins))
    sign = np.where(rng.uniform(size=(n_rows, n_bins)) < 0.5, -1.0, 1.0) if signed else 1.0
    values = magnitude * sign
    dense = np.where(rng.uniform(size=(n_rows, n_bins)) < 0.35, values, 0.0)
    dense[0] = values[0]  # full row
    dense[1] = 0.0  # empty row
    dense[2] = 0.0
    dense[2, n_bins - 1] = values[2, n_bins - 1]  # single entry in the last bin
    dense[3] = 0.0
    dense[3, 0] = values[3, 0]  # single entry in the first bin
    rows, bins = np.nonzero(dense != 0.0)
    counts = np.bincount(rows, minlength=n_rows)
    row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    csr_values = np.ascontiguousarray(dense[rows, bins])
    compiled = np.asarray(csr_kernels.run_row_sums(row_ptr, bins.astype(np.uint8), csr_values, n_bins))
    assert compiled.shape == (n_rows,) and compiled.dtype == np.float64
    assert np.all(np.isfinite(compiled))
    assert compiled[1] == 0.0 and compiled[1].tobytes() == np.float64(0.0).tobytes()
    for k in (1, 2, 3, 4, 5, 6):
        reference = dense.reshape(n_rows // k, k, n_bins).sum(axis=-1).reshape(-1)
        assert compiled.tobytes() == reference.tobytes(), (
            f"NumPy {np.__version__}: compiled row-sum emulation differs from the dense reduction with {k} "
            f"classes at B = {n_bins} (signed = {signed}); max |diff| = {np.max(np.abs(compiled - reference)):.3e}")
    bounded = row_sums_numpy(row_ptr, bins.astype(np.uint8), csr_values, n_bins)
    assert compiled.tobytes() == bounded.tobytes(), f"NumPy {np.__version__}: differs from row_sums_numpy at B = {n_bins}"


@needs_numba
def test_row_sum_self_check_record_refusal_and_production_selection(monkeypatch):
    """The production step uses the compiled row sums only after the runtime
    self-check reproduced the running NumPy's bytes; the NumPy reference is
    never called by production, and a mismatching emulation is refused rather
    than replaced."""
    import maple_syrup.characteristic_sparse as cs

    record = row_sum_emulation_self_check(force=True)
    assert record["byte_exact"] is True and record["numpy"] == np.__version__
    assert tuple(record["bins"]) == ROW_SUM_SELF_CHECK_BINS and record["rows_per_bin_count"] == 96
    assert {1, 2, 8, 16, 32, 64, 100, 127, 128} <= set(ROW_SUM_SELF_CHECK_BINS)
    assert all(1 <= b <= 128 for b in ROW_SUM_SELF_CHECK_BINS)
    assert ROW_SUM_EMULATION_VERIFIED_NUMPY_VERSIONS == ("2.5.3",)  # root-recorded provenance, not a runtime rule
    assert record["running_numpy_root_verified"] == (np.__version__ in ROW_SUM_EMULATION_VERIFIED_NUMPY_VERSIONS)
    policy = row_sum_policy()
    assert policy["production"] == "compiled" and policy["reference"] == "row_sums_numpy"
    assert policy["self_check"]["byte_exact"] is True and policy["running_numpy"] == np.__version__
    graph, net = chain(3)
    vel, rate, settle = fields(graph.shape, 2, v=0.05, rate=2.0)
    pickup = zeros(graph.shape, 2)
    pickup[2, 0, :] = [1.0, 0.5]
    phase = empty_sparse_phase_state(net, 2, 32)
    mobile = zeros(graph.shape, 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("row_sums_numpy is the reference; production must not call it")

    monkeypatch.setattr(cs, "row_sums_numpy", forbidden)
    result = sparse_characteristic_step(net, mobile, phase, pickup, vel, rate, settle, 1.0, n_substeps=2)
    assert result.phase_after.nnz == 2
    dense = characteristic_step(net, mobile, empty_phase_state(graph.shape, 2, 32, DX), pickup, vel, rate, settle, 1.0,
                                n_substeps=2, implementation="numba")
    assert_matches_dense_bitwise(result, dense)
    monkeypatch.undo()
    # a mismatching emulation: refused by the forced self-check and by the step, never replaced
    real_run = csr_kernels.run_row_sums

    def perturbed(row_ptr, bins, mass, nb):
        out = np.array(real_run(row_ptr, bins, mass, nb), copy=True)
        out[0] = np.nextafter(out[0], np.inf)  # row 0 of the fixture is the full row
        return out

    monkeypatch.setattr(csr_kernels, "run_row_sums", perturbed)
    with pytest.raises(RowSumEmulationMismatch) as info:
        row_sum_emulation_self_check(force=True)
    assert isinstance(info.value, TransportError)
    assert info.value.numpy_version == np.__version__ and info.value.n_bins == ROW_SUM_SELF_CHECK_BINS[0]
    monkeypatch.setattr(cs, "_ROW_SUM_SELF_CHECK", None)
    with pytest.raises(RowSumEmulationMismatch):
        sparse_characteristic_step(net, mobile, phase, pickup, vel, rate, settle, 1.0)
    assert row_sum_policy()["self_check"] is None
    monkeypatch.undo()
    assert row_sum_emulation_self_check(force=True)["byte_exact"] is True
    again = sparse_characteristic_step(net, mobile, phase, pickup, vel, rate, settle, 1.0, n_substeps=2)
    np.testing.assert_array_equal(again.phase_after.fraction, result.phase_after.fraction)


# --- state construction and validation ---------------------------------------------------------------------
def test_empty_state_is_canonical_and_validates():
    graph, net = chain(3)
    state = empty_sparse_phase_state(net, 2, 16)
    assert isinstance(state, SparsePhaseState)
    assert state.nnz == 0 and state.n_rows == 6 and state.n_nonempty_rows == 0 and state.n_bins == 16
    assert state.n_classes == 2 and state.shape == graph.shape and state.n_active == 3
    assert state.row_ptr.dtype == np.int64 and state.row_ptr.shape == (7,) and not np.any(state.row_ptr)
    assert state.bin.dtype == np.uint8 and state.fraction.dtype == np.float64 and state.position_m.dtype == np.float64
    assert not state.row_ptr.flags.writeable and not state.fraction.flags.writeable
    assert np.array_equal(state.cell_of_slot, np.arange(3))
    validate_sparse_phase_state(state, zeros(graph.shape, 2), net)
    with pytest.raises(TransportError):  # a positive pool needs a stored row
        validate_sparse_phase_state(state, zeros(graph.shape, 2) + 1.0, net)
    for bad in (zeros(graph.shape, 2) - 1.0, zeros(graph.shape, 2) + np.nan, zeros(graph.shape, 2)[..., :1],
                zeros(graph.shape, 2).astype(np.float32), list(zeros(graph.shape, 2))):
        with pytest.raises(TransportError):
            validate_sparse_phase_state(state, bad, net)
    with pytest.raises(TransportError):
        empty_sparse_phase_state(net, 2, 0)
    with pytest.raises(TransportError):
        empty_sparse_phase_state(net, 2, 129)
    with pytest.raises(TransportError):
        empty_sparse_phase_state(net, 0, 8)
    occupancy = state.occupancy()
    assert occupancy["nnz"] == 0 and occupancy["csr_bytes"] == sparse_state_bytes(6, 0) == 7 * 8
    assert occupancy["dense_equivalent_bytes"] == dense_phase_bytes(graph.shape, 2, 16) == 2 * 3 * 2 * 16 * 8
    assert state.nbytes == 7 * 8 + 3 * 8 + 3 * 8
    assert DEFAULT_IMPLEMENTATION == "numba"
    assert empty_sparse_phase_state(net, 1).n_bins == DEFAULT_N_BINS == 32


@needs_numba
def test_validation_refusals_cover_every_structural_and_value_invariant():
    rng = np.random.default_rng(2)
    graph = make_graph(valley_full(4, 3))
    net = transport_network(graph)
    ny, nx = graph.shape
    nc = 2
    pair = Pair(net, 8, nc)
    settle = np.zeros((ny, nx, nc), bool)
    for _ in range(3):  # several packets per row: the pickup of each step plus moved older mass
        pickup = rng.uniform(0.2, 1.0, (ny, nx, nc))
        pickup[..., 1] = 0.0  # class 1 never receives mass: its rows stay empty
        pair.step(pickup, rng.uniform(0.02, 0.1, (ny, nx, nc)), rng.uniform(0, 2, (ny, nx, nc)), settle, 1.0)
    state, mobile = pair.phase, pair.mobile
    validate_sparse_phase_state(state, mobile, net)
    counts = np.diff(state.row_ptr)
    assert counts.max() >= 2 and np.all(counts.reshape(-1, nc)[:, 1] == 0)
    multi = int(np.flatnonzero(counts >= 2)[0])
    e0 = int(state.row_ptr[multi])

    def replaced(**changes):
        return dataclasses.replace(state, **changes)

    def array_with(name, index, value):
        arr = np.array(getattr(state, name), copy=True)
        arr[index] = value
        return arr

    bad_states = {
        "row_ptr not starting at 0": replaced(row_ptr=array_with("row_ptr", 0, 1)),
        "row_ptr decreasing": replaced(row_ptr=array_with("row_ptr", multi + 1, state.row_ptr[multi] - 1)),
        "row_ptr end != nnz": replaced(row_ptr=array_with("row_ptr", -1, state.nnz + 1)),
        "row_ptr dtype": replaced(row_ptr=state.row_ptr.astype(np.int32)),
        "row_ptr length": replaced(row_ptr=state.row_ptr[:-1]),
        "bin dtype": replaced(bin=state.bin.astype(np.int64)),
        "bin duplicate": replaced(bin=array_with("bin", e0 + 1, state.bin[e0])),
        "bin decreasing": replaced(bin=array_with("bin", e0, 7), position_m=array_with(
            "position_m", e0, 7.5 * DX / 8)),
        "bin >= n_bins": replaced(bin=array_with("bin", e0, 8)),
        "fraction negative": replaced(fraction=array_with("fraction", e0, -state.fraction[e0])),
        "fraction zero": replaced(fraction=array_with("fraction", e0, 0.0)),
        "fraction nan": replaced(fraction=array_with("fraction", e0, np.nan)),
        "fraction sum": replaced(fraction=array_with("fraction", e0, state.fraction[e0] * 0.5)),
        "fraction dtype": replaced(fraction=state.fraction.astype(np.float32)),
        "position >= dx": replaced(position_m=array_with("position_m", e0, DX)),
        "position negative": replaced(position_m=array_with("position_m", e0, -1e-3)),
        "position nan": replaced(position_m=array_with("position_m", e0, np.nan)),
        "position outside its bin": replaced(position_m=array_with(
            "position_m", e0, ((int(state.bin[e0]) + 1) % 8 + 0.5) * DX / 8)),
        "position length": replaced(position_m=state.position_m[:-1]),
        "wrong dx": replaced(dx_m=0.25),
        "wrong n_bins": replaced(n_bins=129),
        "wrong n_classes": replaced(n_classes=3),
        "wrong shape": replaced(shape=(ny + 1, nx)),
        "namespace": replaced(xp=math),
        "graph digest": replaced(network_sha256="0" * 64),
        "slot map": replaced(cell_of_slot=np.array(state.cell_of_slot[::-1], copy=True)),
    }
    for bad in bad_states.values():
        with pytest.raises(TransportError):
            validate_sparse_phase_state(bad, mobile, net)
        with pytest.raises(TransportError):
            sparse_characteristic_step(net, mobile, bad, zeros(graph.shape, nc), zeros(graph.shape, nc) + 0.05,
                                       zeros(graph.shape, nc) + 1.0, settle, 1.0)
    # mobile relation: stored bins on a zero row / positive mass on an empty row
    with pytest.raises(TransportError, match="mobile total is 0"):
        validate_sparse_phase_state(state, mobile * 0.0, net)
    with_class1 = mobile.copy()
    with_class1[..., 1] = 1.0
    with pytest.raises(TransportError, match="positive"):
        validate_sparse_phase_state(state, with_class1, net)
    # a state built for another graph of the same shape is refused (digest), and a device-less
    # network of a different active set is refused before any arithmetic
    other = transport_network(make_graph(valley_full(4, 3, sy=0.03125)))
    with pytest.raises(TransportError):
        validate_sparse_phase_state(state, mobile, other)
    assert isinstance(net, TransportNetwork)


# --- T01: constant-law impulse on a chain -----------------------------------------------------------------
@needs_numba
@pytest.mark.parametrize("n_bins", [8, 32])
@pytest.mark.parametrize("v", [0.01, 1.0 / 64.0])
def test_impulse_face_crossings_follow_exp_minus_k_dx_over_L(n_bins, v):
    n, L = 6, 0.05
    graph, net = chain(n)
    vel, rate, settle = fields(graph.shape, 1, v=v, rate=1.0 / L)
    pair = Pair(net, n_bins, 1)
    pickup = zeros(graph.shape, 1)
    pickup[n - 1, 0, 0] = 1.0
    first = None
    for t in range(1, 601):
        result = pair.step(pickup, vel, rate, settle, 1.0)
        pickup = zeros(graph.shape, 1)
        if first is None and float(result.transport.export_request_kg.sum()) > 0.0:
            first = t
    target = np.exp(-np.arange(1, n + 1) * DX / L)
    np.testing.assert_allclose(pair.crossing[::-1, 0, 0], target, rtol=1e-12, atol=1e-35)
    assert first == round(n * DX / v)
    assert abs(pair.total() - 1.0) <= 1e-13
    assert float(pair.mobile.sum()) <= 1e-30
    exact_dep = np.exp(-np.arange(0, n) * DX / L) - np.exp(-np.arange(1, n + 1) * DX / L)
    np.testing.assert_allclose(pair.dep[::-1, 0, 0], exact_dep, rtol=1e-12, atol=1e-35)
    assert max(pair.nnz_history) <= 2  # one packet, never more than two bins while it crosses a face


# --- T02: no deposition: finite front, exact first moment --------------------------------------------------
@needs_numba
def test_no_deposition_front_and_first_moment():
    v = 1.0 / 64.0
    graph, net = chain(3)
    vel, rate, settle = fields(graph.shape, 1, v=v, rate=0.0)
    pair = Pair(net, 8, 1)
    pickup = zeros(graph.shape, 1)
    pickup[2, 0, 0] = 2.0
    exported_at = None
    for t in range(1, 200):
        result = pair.step(pickup, vel, rate, settle, 1.0)
        pickup = zeros(graph.shape, 1)
        if exported_at is None and float(result.transport.export_request_kg.sum()) > 0.0:
            exported_at = t
        assert float(result.transport.deposition_request_kg.sum()) == 0.0
    assert exported_at == 3 * 32
    assert pair.export[0, 0, 0] == 2.0 and float(pair.mobile.sum()) == 0.0 and pair.phase.nnz == 0
    pair = Pair(net, 4, 1)
    moment = 0.0
    mass = 0.0
    for t in range(1, 25):
        pickup = zeros(graph.shape, 1)
        pickup[2, 0, 0] = 0.1 * t
        pair.step(pickup, vel, rate, settle, 1.0)
        mass += 0.1 * t
        moment += mass * v
        ph = pair.phase
        rows = np.repeat(np.arange(ph.n_rows), np.diff(ph.row_ptr))
        remaining = pair.mobile.reshape(-1, 1)[ph.cell_of_slot].reshape(-1)[rows]
        w = remaining * ph.fraction
        np.testing.assert_allclose(float((w * ph.position_m).sum()), moment, rtol=1e-12)
        np.testing.assert_allclose(float(w.sum()), mass, rtol=1e-13)


# --- T03: differing receiver v and r -------------------------------------------------------------------------
@needs_numba
def test_two_cell_path_with_differing_receiver_laws_is_exact():
    graph, net = chain(2)
    ny, nx = graph.shape
    v1, r1, v2, r2 = 0.2, 3.0, 0.1, 7.0
    dt = 1.0
    vel = zeros(graph.shape, 1)
    rate = zeros(graph.shape, 1)
    vel[1, 0, 0], rate[1, 0, 0] = v1, r1
    vel[0, 0, 0], rate[0, 0, 0] = v2, r2
    settle = np.zeros((ny, nx, 1), dtype=bool)
    x0 = 0.375
    pair = Pair(net, 8, 1)
    pickup = zeros(graph.shape, 1)
    pickup[1, 0, 0] = 1.0
    place = zeros(graph.shape, 1)
    place[1, 0, 0] = 0.1875
    pair.step(pickup, place, zeros(graph.shape, 1), settle, dt)
    pair.step(zeros(graph.shape, 1), place, zeros(graph.shape, 1), settle, dt)
    ph = pair.phase
    assert ph.nnz == 1 and ph.position_m[0] == x0 and int(ph.bin[0]) == int(bin_index(np.array(x0), DX, 8, np))
    result = pair.step(zeros(graph.shape, 1), vel, rate, settle, dt)
    tau = (DX - x0) / v1
    s1 = math.exp(-r1 * (DX - x0))
    d = v2 * (dt - tau)
    s2 = s1 * math.exp(-r2 * d)
    tr = result.transport
    np.testing.assert_allclose(tr.deposition_request_kg[1, 0, 0], 1.0 - s1, rtol=1e-13)
    np.testing.assert_allclose(tr.internal_transfer_out_kg[1, 0, 0], s1, rtol=1e-13)
    np.testing.assert_allclose(tr.internal_transfer_in_kg[0, 0, 0], s1, rtol=1e-13)
    np.testing.assert_allclose(tr.deposition_request_kg[0, 0, 0], s1 - s2, rtol=1e-13)
    np.testing.assert_allclose(result.arrival_deposition_kg[0, 0, 0], s1 - s2, rtol=1e-13)
    np.testing.assert_allclose(result.mobile_remaining_kg[0, 0, 0], s2, rtol=1e-13)
    assert result.mobile_remaining_kg[1, 0, 0] == 0.0
    ph = result.phase_after
    assert ph.nnz == 1 and int(ph.bin[0]) == int(bin_index(np.array(d), DX, 8, np)) and ph.fraction[0] == 1.0
    np.testing.assert_allclose(ph.position_m[0], d, rtol=1e-13)
    assert np.diff(ph.row_ptr).tolist() == [1, 0]  # row 0 (outlet, slot 0) holds the packet; row 1 is empty
    assert float(tr.export_request_kg.sum()) == 0.0
    np.testing.assert_allclose(tr.mobile_after_transfer_kg.sum(), 1.0, rtol=1e-14)


# --- T04: v -> 0, r -> 0, settle toggles; rows disappear and reappear ---------------------------------------
@needs_numba
def test_zero_velocity_zero_rate_settle_and_reappearing_rows():
    graph, net = chain(3)
    vel, rate, settle = fields(graph.shape, 2, v=0.05, rate=4.0)
    pair = Pair(net, 8, 2)
    pickup = zeros(graph.shape, 2)
    pickup[2, 0, :] = [1.0, 3.0]
    pair.step(pickup, vel, rate, settle, 1.0)
    before_phase = pair.phase
    before_mobile = pair.mobile.copy()
    assert before_phase.nnz == 2 and before_phase.n_nonempty_rows == 2
    result = pair.step(zeros(graph.shape, 2), vel * 0.0, rate, settle, 1.0)
    assert float(result.transport.deposition_request_kg.sum()) == 0.0
    np.testing.assert_allclose(pair.mobile, before_mobile, rtol=4 * EPS, atol=0)
    np.testing.assert_array_equal(pair.phase.row_ptr, before_phase.row_ptr)
    np.testing.assert_array_equal(pair.phase.bin, before_phase.bin)
    np.testing.assert_allclose(pair.phase.fraction, before_phase.fraction, rtol=4 * EPS, atol=0)
    np.testing.assert_array_equal(pair.phase.position_m, before_phase.position_m)
    result = pair.step(zeros(graph.shape, 2), vel, rate * 0.0, settle, 1.0)
    assert float(result.transport.deposition_request_kg.sum()) == 0.0
    np.testing.assert_allclose(pair.mobile.sum(), before_mobile.sum(), rtol=1e-14)
    assert float((pair.phase.position_m * pair.phase.fraction).sum()) > float(
        (before_phase.position_m * before_phase.fraction).sum())
    # settle everywhere: whole pool plus pickup deposits; the state is the canonical EMPTY CSR
    result = pair.step(pickup, vel, rate, ~settle, 1.0)
    np.testing.assert_allclose(result.transport.settled_kg.sum(), before_mobile.sum() + pickup.sum(), rtol=1e-14)
    assert float(pair.mobile.sum()) == 0.0 and pair.phase.nnz == 0 and pair.phase.n_nonempty_rows == 0
    # rows reappear with new pickup in one class only, then vanish again through export
    only_class1 = zeros(graph.shape, 2)
    only_class1[1, 0, 1] = 0.5
    pair.step(only_class1, vel, rate * 0.0, settle, 1.0)
    assert pair.phase.nnz == 1 and np.diff(pair.phase.row_ptr).reshape(-1, 2)[:, 0].sum() == 0
    for _ in range(40):  # 0.5 m at 0.05 m/s: the packet exports from the outlet cell
        pair.step(zeros(graph.shape, 2), vel, rate * 0.0, settle, 1.0)
    assert pair.phase.nnz == 0 and float(pair.export[..., 1].sum()) == 0.5
    assert pair.nnz_history[-1] == 0 and max(pair.nnz_history) <= 3


# --- T05: pickup is a new packet at the upstream face ---------------------------------------------------------
@needs_numba
def test_pickup_is_a_new_packet_at_the_upstream_face_not_merged_with_older_mass():
    graph, net = chain(2)
    nb = 8
    pair = Pair(net, nb, 1)
    vel, rate, settle = fields(graph.shape, 1, v=0.1, rate=0.0)
    pickup = zeros(graph.shape, 1)
    pickup[1, 0, 0] = 1.0
    for _ in range(3):
        pair.step(pickup, vel, rate, settle, 1.0)
        pickup = zeros(graph.shape, 1)
    assert pair.mobile[1, 0, 0] == 1.0 and pair.phase.nnz == 1
    pickup[1, 0, 0] = 2.0
    result = pair.step(pickup, vel, rate * 0.0 + 2.0, settle, 1.0)
    ph = pair.phase
    assert ph.nnz == 2
    row = 1 * 1 + 0  # slot of cell (1, 0) is 1 (flat index 1), class 0
    e0, e1 = int(ph.row_ptr[row]), int(ph.row_ptr[row + 1])
    assert e1 - e0 == 2
    w = pair.mobile[1, 0, 0] * ph.fraction[e0:e1]
    x = ph.position_m[e0:e1]
    s = math.exp(-2.0 * 0.1)
    old_bin = int(bin_index(np.array(0.4), DX, nb, np))
    new_bin = int(bin_index(np.array(0.1), DX, nb, np))
    assert old_bin != new_bin and ph.bin[e0:e1].tolist() == sorted([new_bin, old_bin])
    by_bin = dict(zip(ph.bin[e0:e1].tolist(), range(2), strict=True))
    np.testing.assert_allclose(w[by_bin[old_bin]], 1.0 * s, rtol=1e-14)
    np.testing.assert_allclose(w[by_bin[new_bin]], 2.0 * s, rtol=1e-14)
    np.testing.assert_allclose(x[by_bin[old_bin]], 0.4, rtol=1e-14)
    np.testing.assert_allclose(x[by_bin[new_bin]], 0.1, rtol=1e-14)
    np.testing.assert_allclose(result.transport.deposition_request_kg[1, 0, 0], 3.0 * (1.0 - s), rtol=1e-14)


# --- T06: dry receiver settles arrivals ---------------------------------------------------------------------
@needs_numba
def test_dry_receiver_settles_arrivals_immediately():
    graph, net = chain(2)
    vel, rate, settle = fields(graph.shape, 1, v=0.2, rate=1.0)
    settle[0] = True
    pair = Pair(net, 8, 1)
    pickup = zeros(graph.shape, 1)
    pickup[1, 0, 0] = 1.0
    pair.step(pickup, vel, rate, settle, 1.0)
    pair.step(zeros(graph.shape, 1), vel, rate, settle, 1.0)
    result = pair.step(zeros(graph.shape, 1), vel, rate, settle, 1.0)
    s = math.exp(-1.0 * DX)
    tr = result.transport
    np.testing.assert_allclose(tr.settled_kg[0, 0, 0], s, rtol=1e-13)
    np.testing.assert_allclose(result.arrival_deposition_kg[0, 0, 0], s, rtol=1e-13)
    np.testing.assert_allclose(tr.internal_transfer_in_kg[0, 0, 0], s, rtol=1e-13)
    assert float(tr.export_request_kg.sum()) == 0.0 and float(result.mobile_remaining_kg.sum()) == 0.0
    assert result.phase_after.nnz == 0
    np.testing.assert_allclose(pair.dep.sum(), 1.0, rtol=1e-14)


# --- T07: converging graph, pure-orientation planes, face signs ----------------------------------------------
@needs_numba
def test_converging_valley_and_pure_planes_keep_identities_and_face_signs():
    rng = np.random.default_rng(11)
    nc = 3
    cases = [("valley", make_graph(valley_full(6, 5))), ("south", make_graph(chain_full(6)))]
    z, export = west_plane_full(5, 6)
    cases.append(("west", make_graph(z, export=export)))
    for name, graph in cases:
        net = transport_network(graph)
        ny, nx = graph.shape
        mobile = rng.uniform(0.0, 2.0, (ny, nx, nc))
        pickup = rng.uniform(0.0, 1.0, (ny, nx, nc))
        v = rng.uniform(0.0, 0.2, (ny, nx, nc))
        rate = rng.uniform(0.0, 4.0, (ny, nx, nc))
        settle = np.zeros((ny, nx, nc), dtype=bool)
        settle[ny // 2, nx // 2, 0] = True
        pair = Pair(net, 8, nc)
        pair.step(mobile, v, rate, settle, 1.0)
        crossed = zeros(graph.shape, nc)
        arrived = zeros(graph.shape, nc)
        for _ in range(5):
            before = pair.mobile + pickup
            result = pair.step(pickup, v, rate, settle, 1.0)
            _sparse_identity_checks(result, net, before, DX)
            tr = result.transport
            crossed += tr.internal_transfer_out_kg + tr.export_request_kg
            arrived += tr.internal_transfer_in_kg
            if name == "south":
                assert not np.any(tr.x_face_gross_kg) and np.all(tr.y_face_net_kg <= 0.0)
                np.testing.assert_array_equal(tr.y_face_gross_kg[0], tr.export_request_kg[0])
            if name == "west":
                assert not np.any(tr.y_face_gross_kg) and np.all(tr.x_face_net_kg <= 0.0)
                np.testing.assert_array_equal(tr.x_face_gross_kg[:, 0], tr.export_request_kg[:, 0])
        assert float(crossed.sum()) > 0.0 and float(pair.export.sum()) > 0.0
        if name == "valley":
            assert float(arrived[:, nx // 2].sum()) > 0.0
            assert float(arrived[:, 0].sum()) == 0.0 and float(arrived[:, nx - 1].sum()) == 0.0


# --- T08: conservation on a random network, classes, substeps, bin counts ----------------------------------
@needs_numba
@pytest.mark.parametrize("n_substeps", [1, 3])
@pytest.mark.parametrize("n_bins", [1, 8, 32, 128])
def test_random_network_conserves_and_matches_dense_bitwise(n_bins, n_substeps):
    rng = np.random.default_rng(5)
    graph = make_graph(random_full(rng, 7, 6))
    net = transport_network(graph)
    ny, nx = graph.shape
    nc = 4
    pair = Pair(net, n_bins, nc)
    total_pickup = 0.0
    for _ in range(12):
        pickup = rng.uniform(0.0, 1.0, (ny, nx, nc))
        v = rng.uniform(0.0, 0.25 * n_substeps, (ny, nx, nc))
        rate = rng.uniform(0.0, 20.0, (ny, nx, nc)) * (rng.uniform(size=(ny, nx, nc)) > 0.2)
        settle = rng.uniform(size=(ny, nx, nc)) < 0.1
        before = pair.mobile + pickup
        total_pickup += float(pickup.sum())
        result = pair.step(pickup, v, rate, settle, 1.0, n_substeps=n_substeps)
        _sparse_identity_checks(result, net, before, DX)
        tr = result.transport
        assert float(tr.max_courant) <= DEFAULT_COURANT_MAX
        assert tr.n_substeps == n_substeps
        assert result.peak_nnz >= result.nnz_after
        assert result.nnz_after <= result.phase_after.n_rows * n_bins
    np.testing.assert_allclose(pair.dep.sum() + pair.export.sum() + pair.mobile.sum(), total_pickup, rtol=1e-12)
    assert pair.phase.occupancy()["nnz"] == pair.phase.nnz


# --- T13 fixture: compiled kernels, bitwise versus dense Numba and bounded versus the array path -------------
@needs_numba
def test_t13_fixture_sparse_bitwise_with_dense_numba_and_bounded_with_array_path():
    rng = np.random.default_rng(31)
    graph = make_graph(random_full(rng, 6, 5))
    net = transport_network(graph)
    ny, nx = graph.shape
    nc = 3
    pair = Pair(net, 16, nc)
    arr_mobile = zeros(graph.shape, nc)
    arr_phase = empty_phase_state(graph.shape, nc, 16, DX)
    for _ in range(8):
        pickup = rng.uniform(0.0, 1.0, (ny, nx, nc))
        v = rng.uniform(0.0, 0.2, (ny, nx, nc))
        rate = rng.uniform(0.0, 8.0, (ny, nx, nc)) * (rng.uniform(size=(ny, nx, nc)) > 0.25)
        settle = rng.uniform(size=(ny, nx, nc)) < 0.1
        sp = pair.step(pickup, v, rate, settle, 1.0, n_substeps=2)  # bitwise against dense Numba inside
        a = characteristic_step(net, arr_mobile, arr_phase, pickup, v, rate, settle, 1.0, n_substeps=2,
                                implementation="array")
        arr_mobile, arr_phase = a.mobile_remaining_kg, a.phase_after
        for name in T13_FIELDS:
            np.testing.assert_allclose(getattr(sp.transport, name), getattr(a.transport, name), rtol=1e-12,
                                       atol=1e-16, err_msg=name)
        expanded = sparse_to_dense(sp.phase_after)
        np.testing.assert_allclose(sp.mobile_remaining_kg[..., None] * expanded.fraction,
                                   a.mobile_remaining_kg[..., None] * a.phase_after.fraction, rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(expanded.position_m, a.phase_after.position_m, rtol=0, atol=1e-12)
        for step in (sp.transport, a.transport):  # each path independently meets the unchanged MAPLE bound
            assert np.all(np.abs(step.budget_residual_by_class_kg) <= step.budget_tolerance_by_class_kg)


# --- T09: immutability and refusals -------------------------------------------------------------------------
@needs_numba
def test_inputs_are_not_modified_and_outputs_are_frozen():
    rng = np.random.default_rng(2)
    graph = make_graph(valley_full(4, 3))
    net = transport_network(graph)
    ny, nx = graph.shape
    nc = 2
    phase = empty_sparse_phase_state(net, nc, 8)
    first = sparse_characteristic_step(net, zeros(graph.shape, nc), phase, rng.uniform(0, 1, (ny, nx, nc)),
                                       rng.uniform(0, 0.2, (ny, nx, nc)), rng.uniform(0, 3, (ny, nx, nc)),
                                       np.zeros((ny, nx, nc), bool), 1.0)
    inputs = {"mobile": first.mobile_remaining_kg.copy(), "pickup": rng.uniform(0, 1, (ny, nx, nc)),
              "v": rng.uniform(0, 0.2, (ny, nx, nc)), "rate": rng.uniform(0, 3, (ny, nx, nc)),
              "settle": rng.uniform(size=(ny, nx, nc)) < 0.2}
    copies = {k: v.copy() for k, v in inputs.items()}
    state = first.phase_after
    arrays = {name: np.array(getattr(state, name), copy=True) for name in ("row_ptr", "bin", "fraction", "position_m")}
    result = sparse_characteristic_step(net, inputs["mobile"], state, inputs["pickup"], inputs["v"], inputs["rate"],
                                        inputs["settle"], 1.0, n_substeps=2)
    for k, value in inputs.items():
        np.testing.assert_array_equal(value, copies[k])
    for name, value in arrays.items():
        np.testing.assert_array_equal(getattr(state, name), value)
    for obj in (result.phase_after, state, phase):
        for name in ("row_ptr", "bin", "fraction", "position_m", "cell_of_slot", "slot_of_cell"):
            assert not getattr(obj, name).flags.writeable, name
    assert result.implementation == "numba" and result.n_bins == 8


@needs_numba
def test_refusals_leave_nothing_behind_and_inactive_cells_are_refused_not_erased():
    graph = make_graph(valley_full(4, 3))
    net = transport_network(graph)
    ny, nx = graph.shape
    nc = 2
    good = {"mobile": np.full((ny, nx, nc), 0.5), "pickup": np.full((ny, nx, nc), 0.1),
            "v": np.full((ny, nx, nc), 0.1), "rate": np.full((ny, nx, nc), 2.0), "settle": np.zeros((ny, nx, nc), bool)}
    phase = empty_sparse_phase_state(net, nc, 8)
    ok = sparse_characteristic_step(net, good["mobile"] * 0.0, phase, good["mobile"], good["v"], good["rate"],
                                    good["settle"], 1.0)
    base = {"mobile": ok.mobile_remaining_kg, "phase": ok.phase_after,
            **{k: good[k] for k in ("pickup", "v", "rate", "settle")}}

    def call(**changes):
        args = dict(base)
        args.update(changes)
        return sparse_characteristic_step(net, args["mobile"], args["phase"], args["pickup"], args["v"], args["rate"],
                                          args["settle"], args.get("dt", 1.0), n_substeps=args.get("n_substeps", 1),
                                          courant_max=args.get("courant_max", DEFAULT_COURANT_MAX),
                                          implementation=args.get("implementation", DEFAULT_IMPLEMENTATION),
                                          max_nnz=args.get("max_nnz"))

    call()
    bad_cases = {
        "nonfinite mobile": {"mobile": np.where(np.arange(nc) == 0, np.nan, base["mobile"])},
        "negative pickup": {"pickup": -base["pickup"]},
        "negative velocity": {"v": -base["v"]},
        "nonfinite rate": {"rate": base["rate"] * np.inf},
        "settle dtype": {"settle": base["settle"].astype(np.float64)},
        "shape": {"pickup": base["pickup"][:, :, :1]},
        "not an array": {"pickup": list(base["pickup"])},
        "mobile float32": {"mobile": base["mobile"].astype(np.float32)},
        "dense phase object": {"phase": empty_phase_state(graph.shape, nc, 8, DX)},
        "non-canonical empty cell": {"mobile": base["mobile"] * 0.0},
        "wrong dx": {"phase": dataclasses.replace(base["phase"], dx_m=0.25)},
        "dt": {"dt": 0.0},
        "substeps": {"n_substeps": 0},
        "courant_max": {"courant_max": 0.75},
        "implementation array (no array path exists)": {"implementation": "array"},
        "implementation loop": {"implementation": "loop"},
        "max_nnz negative": {"max_nnz": -1},
        "max_nnz float": {"max_nnz": 1.5},
        "max_nnz bool": {"max_nnz": True},
    }
    for changes in bad_cases.values():
        with pytest.raises(TransportError):
            call(**changes)
    with pytest.raises(TransportStepRejected):
        call(v=np.full((ny, nx, nc), 0.3))  # a = 0.6 > 0.5
    call(v=np.full((ny, nx, nc), 0.3), n_substeps=2)
    # inactive cells: mass, pickup or velocity there is refused, never erased
    active = np.ones((4, 1), dtype=bool)
    active[3, 0] = False
    graph2 = make_graph(chain_full(4), active=active)
    net2 = transport_network(graph2)
    assert net2.n_active == 3
    phase2 = empty_sparse_phase_state(net2, 1, 8)
    assert phase2.n_rows == 3 and np.array_equal(phase2.cell_of_slot, [0, 1, 2]) and phase2.slot_of_cell[3] == -1
    empty = np.zeros((4, 1, 1))
    for field in ("mobile", "pickup", "v"):
        bad = {"mobile": empty, "pickup": empty, "v": empty}
        bad[field] = np.where(np.arange(4)[:, None, None] == 3, 1.0, 0.0)
        with pytest.raises(TransportError, match="inactive"):
            sparse_characteristic_step(net2, bad["mobile"], phase2, bad["pickup"], bad["v"], empty + 1.0,
                                       np.zeros((4, 1, 1), bool), 1.0)
    pickup2 = empty + np.where(np.arange(4)[:, None, None] == 2, 1.0, 0.0)
    v2 = empty + np.where(np.arange(4)[:, None, None] < 3, 0.1, 0.0)
    ok2 = sparse_characteristic_step(net2, empty, phase2, pickup2, v2, empty + 1.0, np.zeros((4, 1, 1), bool), 1.0)
    de2 = characteristic_step(net2, empty, empty_phase_state(graph2.shape, 1, 8, DX), pickup2, v2, empty + 1.0,
                              np.zeros((4, 1, 1), bool), 1.0, implementation="numba")
    assert_matches_dense_bitwise(ok2, de2)
    assert ok2.mobile_remaining_kg[3, 0, 0] == 0.0 and ok2.transport.mobile_after_transfer_kg[3, 0, 0] == 0.0
    assert ok2.phase_after.nnz == 1


# --- T10: internal substeps equal repeated calls --------------------------------------------------------------
@needs_numba
def test_internal_substeps_match_separate_calls():
    rng = np.random.default_rng(9)
    graph = make_graph(valley_full(5, 3))
    net = transport_network(graph)
    ny, nx = graph.shape
    nc = 2
    v = rng.uniform(0.0, 0.1, (ny, nx, nc))
    rate = rng.uniform(0.0, 5.0, (ny, nx, nc))
    settle = rng.uniform(size=(ny, nx, nc)) < 0.1
    pickup = rng.uniform(0.0, 1.0, (ny, nx, nc))
    phase = empty_sparse_phase_state(net, nc, 8)
    prior = sparse_characteristic_step(net, zeros(graph.shape, nc), phase, rng.uniform(0, 1, (ny, nx, nc)), v, rate,
                                       settle, 0.7)
    one = sparse_characteristic_step(net, prior.mobile_remaining_kg, prior.phase_after, pickup, v, rate, settle, 1.0,
                                     n_substeps=2)
    a = sparse_characteristic_step(net, prior.mobile_remaining_kg, prior.phase_after, pickup, v, rate, settle, 0.5)
    b = sparse_characteristic_step(net, a.mobile_remaining_kg, a.phase_after, zeros(graph.shape, nc), v, rate, settle,
                                   0.5)
    np.testing.assert_allclose(one.mobile_remaining_kg, b.mobile_remaining_kg, rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(one.transport.deposition_request_kg,
                               a.transport.deposition_request_kg + b.transport.deposition_request_kg, rtol=1e-13,
                               atol=1e-16)
    np.testing.assert_allclose(one.transport.export_request_kg,
                               a.transport.export_request_kg + b.transport.export_request_kg, rtol=1e-13, atol=1e-16)
    d1, d2 = sparse_to_dense(one.phase_after), sparse_to_dense(b.phase_after)
    np.testing.assert_allclose(one.mobile_remaining_kg[..., None] * d1.fraction,
                               b.mobile_remaining_kg[..., None] * d2.fraction, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(d1.position_m, d2.position_m, rtol=0, atol=1e-12)
    assert one.transport.n_substeps == 2 and one.peak_nnz >= one.nnz_after


# --- T12: continuous source, bounded-bin merge, bitwise with dense ------------------------------------------
@needs_numba
@pytest.mark.parametrize("n_bins", [8, 32])
def test_continuous_source_total_exact_and_bitwise_with_dense(n_bins):
    v, L, dt, end = 0.01, 0.05, 1.0, 800
    survival = math.exp(-DX / L)
    graph, net = chain(2)
    vel, rate, settle = fields(graph.shape, 1, v=v, rate=1.0 / L)
    births = np.array([dt * math.exp(-((t * dt - 250.0) / 100.0) ** 2) if t * dt < 600.0 else 0.0
                       for t in range(end)])
    pair = Pair(net, n_bins, 1)
    rates = np.zeros(end)
    for t in range(end):
        pickup = zeros(graph.shape, 1)
        pickup[0, 0, 0] = births[t]
        result = pair.step(pickup, vel, rate, settle, dt)
        rates[t] = float(result.transport.export_request_kg.sum())
    np.testing.assert_allclose(pair.export.sum(), births.sum() * survival, rtol=1e-12)
    assert abs(pair.total() - births.sum()) <= 1e-12 * births.sum()
    assert float(np.flatnonzero(rates)[0] + 1) >= 50.0  # no early leak: at least dx / v steps
    assert max(pair.nnz_history) <= n_bins  # one cell, one class: never more than B bins


# --- T12b: logarithmic-mean merge under extreme dynamic range -----------------------------------------------
@needs_numba
def test_log_mean_merge_survives_extreme_mass_and_position_spread():
    dx, L = 0.01, 1.0e-4
    graph = make_graph(chain_full(2), dx=dx)
    net = transport_network(graph)
    old_mass, old_x, new_mass = 1.0e-20, 0.009, 1.0
    mobile = zeros(graph.shape, 1)
    mobile[1, 0, 0] = old_mass
    fraction = np.zeros((*graph.shape, 1, 1))
    fraction[..., 0] = 1.0
    position = np.zeros((*graph.shape, 1, 1))
    position[1, 0, 0, 0] = old_x
    dense_phase = dataclasses.replace(empty_phase_state(graph.shape, 1, 1, dx), fraction=fraction, position_m=position)
    phase = sparse_from_dense(dense_phase, mobile, net)
    assert phase.nnz == 1 and phase.position_m[0] == old_x and phase.fraction[0] == 1.0
    validate_sparse_phase_state(phase, mobile, net)
    pickup = zeros(graph.shape, 1)
    pickup[1, 0, 0] = new_mass
    vel, rate, settle = fields(graph.shape, 1, v=0.0, rate=1.0 / L)
    merged = sparse_characteristic_step(net, mobile, phase, pickup, vel, rate, settle, 1.0)
    dense_merged = characteristic_step(net, mobile, dense_phase, pickup, vel, rate, settle, 1.0, implementation="numba")
    assert_matches_dense_bitwise(merged, dense_merged)
    exact = L * math.log((old_mass * math.exp(old_x / L) + new_mass) / (old_mass + new_mass))
    np.testing.assert_allclose(exact, 0.0043948298140119085, rtol=1e-15)
    assert merged.phase_after.nnz == 1
    np.testing.assert_allclose(merged.phase_after.position_m[0], exact, rtol=1e-12)
    assert float(merged.transport.deposition_request_kg.sum()) == 0.0
    eventual = old_mass * math.exp(-(dx - old_x) / L) + new_mass * math.exp(-dx / L)
    pair = Pair(net, 1, 1)
    pair.sp_mobile, pair.sp_phase = merged.mobile_remaining_kg, merged.phase_after
    pair.de_mobile, pair.de_phase = dense_merged.mobile_remaining_kg, dense_merged.phase_after
    vel, rate, settle = fields(graph.shape, 1, v=0.005, rate=1.0 / L)
    for _ in range(4):
        pair.step(zeros(graph.shape, 1), vel, rate, settle, 1.0)
    np.testing.assert_allclose(pair.crossing[1, 0, 0], eventual, rtol=1e-9)
    assert pair.mobile[1, 0, 0] == 0.0
    small_dense = dataclasses.replace(dense_phase, position_m=position * 0.5)
    small_phase = sparse_from_dense(small_dense, mobile * 0.0 + 1.0, net)
    small = sparse_characteristic_step(net, mobile * 0.0 + 1.0, small_phase, pickup, vel * 0.0, rate * 0.0 + 1e-6,
                                       settle, 1.0)
    expected_mean = (1.0 * 0.0045 + 1.0 * 0.0) / 2.0
    row = 1  # slot 1 (cell (1, 0)), class 0
    e0 = int(small.phase_after.row_ptr[row])
    np.testing.assert_allclose(small.phase_after.position_m[e0], expected_mean, rtol=0, atol=1e-11)


# --- capacity: predeclared nnz, refused before allocation, nothing discarded ----------------------------------
@needs_numba
def test_capacity_refusal_names_the_required_nnz_and_leaves_inputs_unchanged():
    graph, net = chain(6)
    ny, nx = graph.shape
    vel, rate, settle = fields(graph.shape, 1, v=0.1, rate=1.0)
    phase = empty_sparse_phase_state(net, 1, 8)
    pickup = np.ones((ny, nx, 1))
    mobile = zeros(graph.shape, 1)
    copies = (pickup.copy(), mobile.copy(), np.array(phase.row_ptr, copy=True))
    with pytest.raises(SparseCapacityError) as info:
        sparse_characteristic_step(net, mobile, phase, pickup, vel, rate, settle, 1.0, max_nnz=5)
    err = info.value
    assert isinstance(err, TransportError)
    assert err.required_nnz == 6 and err.max_nnz == 5 and err.substep == 0 and err.n_substeps == 1
    assert "6" in str(err) and "5" in str(err)
    np.testing.assert_array_equal(pickup, copies[0])
    np.testing.assert_array_equal(mobile, copies[1])
    np.testing.assert_array_equal(phase.row_ptr, copies[2])
    ok = sparse_characteristic_step(net, mobile, phase, pickup, vel, rate, settle, 1.0, max_nnz=6)
    assert ok.nnz_after == 6 == ok.peak_nnz
    same = sparse_characteristic_step(net, mobile, phase, pickup, vel, rate, settle, 1.0)  # unlimited capacity
    assert_matches_dense_bitwise(ok, characteristic_step(net, mobile, empty_phase_state(graph.shape, 1, 8, DX), pickup,
                                                         vel, rate, settle, 1.0, implementation="numba"))
    np.testing.assert_array_equal(ok.phase_after.fraction, same.phase_after.fraction)
    # with substeps the capacity is the PEAK over substeps and the failing substep is named
    with pytest.raises(SparseCapacityError) as info2:
        sparse_characteristic_step(net, ok.mobile_remaining_kg, ok.phase_after, pickup, vel, rate, settle, 1.0,
                                   n_substeps=2, max_nnz=6)
    assert info2.value.required_nnz > 6 and info2.value.n_substeps == 2
    two = sparse_characteristic_step(net, ok.mobile_remaining_kg, ok.phase_after, pickup, vel, rate, settle, 1.0,
                                     n_substeps=2, max_nnz=info2.value.required_nnz)
    assert two.peak_nnz == info2.value.required_nnz >= two.nnz_after
    # an empty step fits a zero capacity
    none = sparse_characteristic_step(net, mobile, phase, mobile, vel, rate, settle, 1.0, max_nnz=0)
    assert none.nnz_after == 0 and none.peak_nnz == 0


# --- structure: small nnz with a high bin count never touches a dense array -------------------------------------
@needs_numba
def test_small_nnz_high_bin_count_is_structurally_sparse_and_dense_conversion_refuses_before_allocating():
    """1000 cells, 6 classes, B = 128: a dense `(n, nc, B)` FP64 array is 6.144 MB.
    Two packets give nnz <= 2. The Python layer's peak traced allocation
    (tracemalloc; NumPy registers its allocations, Numba's runtime
    allocations are not visible to it) must stay below ONE dense array, and
    the dense conversion must refuse before allocating."""
    rng = np.random.default_rng(77)
    graph = make_graph(random_full(rng, 50, 20))
    net = transport_network(graph)
    ny, nx = graph.shape
    assert ny * nx == 1000 and net.n_active == 1000
    nc, nb = 6, 128
    dense_array_bytes = ny * nx * nc * nb * 8
    assert dense_array_bytes == 6_144_000
    vel, rate, settle = fields(graph.shape, nc, v=0.1, rate=2.0)
    pickup = zeros(graph.shape, nc)
    pickup[40, 7, 0] = 1.0
    pickup[25, 12, 3] = 2.0

    def run():
        phase = empty_sparse_phase_state(net, nc, nb)
        mobile = zeros(graph.shape, nc)
        results = []
        for k in range(3):
            res = sparse_characteristic_step(net, mobile, phase, pickup if k == 0 else zeros(graph.shape, nc), vel,
                                             rate, settle, 1.0, n_substeps=2 if k == 1 else 1)
            mobile, phase = res.mobile_remaining_kg, res.phase_after
            results.append(res)
        return results

    warm = run()  # compiles every kernel specialisation used below (outside the measurement)
    tracemalloc.start()
    try:
        results = run()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < dense_array_bytes, f"peak traced allocation {peak} bytes reaches one dense (n, nc, B) array"
    for res, ref in zip(results, warm, strict=True):
        assert res.nnz_after <= 2 and res.peak_nnz <= 2
        np.testing.assert_array_equal(res.phase_after.fraction, ref.phase_after.fraction)
    state = results[-1].phase_after
    report = state.occupancy()
    assert report["n_rows"] == 6000 and report["nnz"] == state.nnz <= 2 and report["n_nonempty_rows"] <= 2
    assert report["dense_equivalent_bytes"] == 2 * dense_array_bytes
    assert report["csr_bytes"] == sparse_state_bytes(6000, state.nnz) < 60_000
    assert state.nbytes < 60_000 + 8 * 1000 * 2
    # dense conversion: refused BEFORE allocation below the budget, accepted above it
    tracemalloc.start()
    try:
        with pytest.raises(DenseConversionRefused) as info:
            sparse_to_dense(state, max_bytes=2 * dense_array_bytes - 1)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert info.value.required_bytes == 2 * dense_array_bytes and info.value.max_bytes == 2 * dense_array_bytes - 1
    assert isinstance(info.value, TransportError) and peak < 1_000_000
    # This six-class example fits the default 64-MiB budget. Use a valid
    # empty state with more classes to test the default refusal honestly,
    # while retaining sparse storage and the explicit byte-cap checks above.
    too_wide = empty_sparse_phase_state(net, 64, nb)
    with pytest.raises(DenseConversionRefused) as default_info:
        sparse_to_dense(too_wide)
    assert default_info.value.required_bytes == dense_phase_bytes(graph.shape, 64, nb)
    assert default_info.value.max_bytes == DEFAULT_DENSE_MAX_BYTES
    assert default_info.value.required_bytes > DEFAULT_DENSE_MAX_BYTES
    dense = sparse_to_dense(state, max_bytes=2 * dense_array_bytes)
    assert dense.fraction.shape == (ny, nx, nc, nb)
    np.testing.assert_array_equal(sparse_from_dense(dense, results[-1].mobile_remaining_kg, net,
                                                    max_bytes=2 * dense_array_bytes).fraction, state.fraction)
    # the same storm on the dense Numba path is bitwise identical
    de_mobile, de_phase = zeros(graph.shape, nc), empty_phase_state(graph.shape, nc, nb, DX)
    for k, res in enumerate(results):
        de = characteristic_step(net, de_mobile, de_phase, pickup if k == 0 else zeros(graph.shape, nc), vel, rate,
                                 settle, 1.0, n_substeps=2 if k == 1 else 1, implementation="numba")
        assert_matches_dense_bitwise(res, de)
        de_mobile, de_phase = de.mobile_remaining_kg, de.phase_after


# --- dense <-> sparse conversions ---------------------------------------------------------------------------
@needs_numba
def test_dense_sparse_round_trip_and_small_domain_budget():
    rng = np.random.default_rng(13)
    graph = make_graph(random_full(rng, 5, 4))
    net = transport_network(graph)
    ny, nx = graph.shape
    nc, nb = 3, 8
    pair = Pair(net, nb, nc)
    for _ in range(4):
        pickup = rng.uniform(0.0, 1.0, (ny, nx, nc)) * (rng.uniform(size=(ny, nx, nc)) > 0.5)
        pair.step(pickup, rng.uniform(0, 0.2, (ny, nx, nc)), rng.uniform(0, 3, (ny, nx, nc)),
                  rng.uniform(size=(ny, nx, nc)) < 0.1, 1.0)
    state, mobile = pair.phase, pair.mobile
    dense = sparse_to_dense(state)
    np.testing.assert_array_equal(dense.fraction, pair.de_phase.fraction)
    np.testing.assert_array_equal(dense.position_m, pair.de_phase.position_m)
    back = sparse_from_dense(dense, mobile, net)
    for name in ("row_ptr", "bin", "fraction", "position_m", "cell_of_slot", "slot_of_cell"):
        np.testing.assert_array_equal(getattr(back, name), getattr(state, name))
    assert back.network_sha256 == state.network_sha256 and back.n_bins == nb and back.n_classes == nc
    assert not dense.fraction.flags.writeable and not back.fraction.flags.writeable
    budget = dense_phase_bytes(graph.shape, nc, nb)
    with pytest.raises(DenseConversionRefused):
        sparse_to_dense(state, max_bytes=budget - 1)
    with pytest.raises(DenseConversionRefused):
        sparse_from_dense(dense, mobile, net, max_bytes=budget - 1)
    with pytest.raises(TransportError):
        sparse_from_dense(dense, mobile * 0.0, net)  # the dense validator refuses the non-canonical pool
    with pytest.raises(TransportError):
        sparse_from_dense(state, mobile, net)  # type confusion is refused
    with pytest.raises(TransportError):
        sparse_to_dense(dense)


@needs_numba
def test_sparse_from_dense_refuses_over_cap_before_validation_or_allocation(monkeypatch):
    """Resource contract (root review 2): the byte cap of `sparse_from_dense`
    must be decided from metadata BEFORE the dense value validator (which
    creates full-size masks and bin-index temporaries) or any conversion
    allocation runs. Ordinary in-budget inputs are still fully validated
    exactly once before conversion, and invalid values are still refused."""
    import maple_syrup.characteristic_sparse as cs

    rng = np.random.default_rng(41)
    graph = make_graph(valley_full(4, 3))
    net = transport_network(graph)
    ny, nx = graph.shape
    nc, nb = 2, 8
    pair = Pair(net, nb, nc)
    for _ in range(3):
        pair.step(rng.uniform(0.0, 1.0, (ny, nx, nc)), rng.uniform(0, 0.2, (ny, nx, nc)),
                  rng.uniform(0, 3, (ny, nx, nc)), np.zeros((ny, nx, nc), bool), 1.0)
    state, mobile = pair.phase, pair.mobile
    dense = sparse_to_dense(state)
    budget = dense_phase_bytes(graph.shape, nc, nb)
    calls: list[str] = []

    def forbidden_validator(*args, **kwargs):
        calls.append("validate_phase_state")
        raise AssertionError("the dense validator ran before the byte-cap refusal")

    def forbidden_maps(*args, **kwargs):
        calls.append("slot_maps")
        raise AssertionError("conversion work ran before the byte-cap refusal")

    monkeypatch.setattr(cs, "validate_phase_state", forbidden_validator)
    monkeypatch.setattr(cs, "slot_maps", forbidden_maps)
    tracemalloc.start()
    try:
        with pytest.raises(DenseConversionRefused) as info:
            sparse_from_dense(dense, mobile, net, max_bytes=budget - 1)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert calls == []
    assert info.value.required_bytes == budget and info.value.max_bytes == budget - 1
    assert peak < 100_000  # metadata-only path: no dense mask, no gather
    # an INVALID dense phase over the cap is refused by the cap first (the validator still never runs)
    invalid = dataclasses.replace(dense, fraction=np.asarray(dense.fraction) * 0.5)
    with pytest.raises(DenseConversionRefused):
        sparse_from_dense(invalid, mobile, net, max_bytes=budget - 1)
    assert calls == []
    monkeypatch.undo()
    # in budget: the real validator runs exactly once before conversion, and still refuses bad values
    real_validator = cs.validate_phase_state
    seen: list[int] = []

    def recording_validator(*args, **kwargs):
        seen.append(1)
        return real_validator(*args, **kwargs)

    monkeypatch.setattr(cs, "validate_phase_state", recording_validator)
    back = sparse_from_dense(dense, mobile, net, max_bytes=budget)
    assert len(seen) == 1
    np.testing.assert_array_equal(back.fraction, state.fraction)
    np.testing.assert_array_equal(back.position_m, state.position_m)
    with pytest.raises(TransportError):
        sparse_from_dense(invalid, mobile, net, max_bytes=budget)
    assert len(seen) == 2
    # metadata/type refusals still precede the cap (no cap decision on garbage)
    with pytest.raises(TransportError):
        sparse_from_dense(dense, list(mobile), net, max_bytes=0)
    with pytest.raises(TransportError):
        sparse_from_dense(state, mobile, net, max_bytes=0)
    assert len(seen) == 2


# --- determinism ---------------------------------------------------------------------------------------------
@needs_numba
def test_repeated_calls_are_bitwise_identical():
    rng = np.random.default_rng(23)
    graph = make_graph(random_full(rng, 6, 4))
    net = transport_network(graph)
    ny, nx = graph.shape
    nc = 3
    phase = empty_sparse_phase_state(net, nc, 32)
    args = (rng.uniform(0, 1, (ny, nx, nc)), rng.uniform(0, 0.2, (ny, nx, nc)), rng.uniform(0, 5, (ny, nx, nc)),
            rng.uniform(size=(ny, nx, nc)) < 0.1)
    first = sparse_characteristic_step(net, zeros(graph.shape, nc), phase, *args, 1.0, n_substeps=2)
    second = sparse_characteristic_step(net, zeros(graph.shape, nc), phase, *args, 1.0, n_substeps=2)
    for name in TRANSPORT_FIELDS:
        np.testing.assert_array_equal(getattr(first.transport, name), getattr(second.transport, name), err_msg=name)
    for name in ("row_ptr", "bin", "fraction", "position_m"):
        np.testing.assert_array_equal(getattr(first.phase_after, name), getattr(second.phase_after, name))
    third = sparse_characteristic_step(net, first.mobile_remaining_kg, first.phase_after, *args, 1.0, n_substeps=2)
    fourth = sparse_characteristic_step(net, second.mobile_remaining_kg, second.phase_after, *args, 1.0, n_substeps=2)
    np.testing.assert_array_equal(third.phase_after.fraction, fourth.phase_after.fraction)
    np.testing.assert_array_equal(third.transport.mobile_after_transfer_kg, fourth.transport.mobile_after_transfer_kg)


# --- dependency and namespace refusals -----------------------------------------------------------------------
def test_numba_unavailable_is_an_explicit_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "numba":
            raise ImportError("no numba")
        return real_import(name, *args, **kwargs)

    csr_kernels.reset_compiled()
    monkeypatch.setattr(builtins, "__import__", fake_import)
    graph, net = chain(2)
    assert not csr_kernels.numba_available()
    with pytest.raises(csr_kernels.NumbaUnavailableError):
        sparse_characteristic_step(net, zeros(graph.shape, 1), empty_sparse_phase_state(net, 1, 8),
                                   zeros(graph.shape, 1), zeros(graph.shape, 1), zeros(graph.shape, 1),
                                   np.zeros((*graph.shape, 1), bool), 1.0)
    with pytest.raises(TransportError):
        csr_kernels.compiled_kernels()
    monkeypatch.undo()
    csr_kernels.reset_compiled()


def test_cupy_network_is_refused_explicitly_without_transfer():
    from maple.core.backend import cupy_module, gpu_execution_available

    if not gpu_execution_available():
        pytest.skip("CuPy with a CUDA device is not available; the explicit device refusal is not exercised and "
                    "no GPU claim is made")
    cp = cupy_module()
    z = chain_full(3)
    graph = build_routing_graph(z, south_export(z), np.full((3, 1), 1.0), DX, xp=cp)
    net = transport_network(graph)
    # CW-B4: creating a CuPy state is legitimate (device arrays, metadata only checked here; the one-time
    # binding upload of the slot maps is counted by MAPLE and is NOT claimed to be zero)
    device_state = empty_sparse_phase_state(net, 1, 8)
    assert device_state.xp is cp and device_state.nnz == 0 and device_state.n_rows == 3 and device_state.n_bins == 8
    for name, dtype in (("row_ptr", np.int64), ("bin", np.uint8), ("fraction", np.float64), ("position_m", np.float64),
                        ("cell_of_slot", np.int64), ("slot_of_cell", np.int64)):
        array = getattr(device_state, name)
        assert type(array) is cp.ndarray and array.dtype == dtype, name
    assert device_state.row_ptr.shape == (4,) and device_state.network_sha256 == net.graph_input_sha256
    with pytest.raises(TransportError, match="host NumPy"):
        slot_maps(net)  # the host slot-map helper stays host-only
    _host_graph, host_net = chain(3)
    phase = empty_sparse_phase_state(host_net, 1, 8)
    with pytest.raises(TransportError):
        sparse_characteristic_step(net, cp.zeros((3, 1, 1)), phase, cp.zeros((3, 1, 1)), cp.zeros((3, 1, 1)),
                                   cp.zeros((3, 1, 1)), cp.zeros((3, 1, 1), dtype=bool), 1.0)
    with pytest.raises(TransportError):  # device arrays on a host network are refused, never transferred
        sparse_characteristic_step(host_net, cp.zeros((3, 1, 1)), phase, cp.zeros((3, 1, 1)), cp.zeros((3, 1, 1)),
                                   cp.zeros((3, 1, 1)), cp.zeros((3, 1, 1), dtype=bool), 1.0)
