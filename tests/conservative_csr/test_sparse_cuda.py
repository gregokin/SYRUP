"""CW-B4 candidate: CUDA sparse CSR characteristic transport (`implementation="cuda"`).

Host-only tests run everywhere: namespace/option refusals, kernel-source specialisation, launch clamping,
scratch estimates, preflight capacity policy, explicit CuPy-unavailable error, slot-map evidence, rebind.
GPU tests skip with the stated reason when `maple.core.backend.gpu_execution_available()` is False and
make no GPU claim; when a device is available they compare the CUDA operator with the accepted CPU CSR
reference (Numba, host copies of the same inputs) at the existing isolated GPU characteristic bounds
(rtol 1e-10 / atol 1e-15 on masses and transport fields, atol 1e-9 m on positions; `tests/phase7b` T11),
check same-device repeats raw-byte exact, immutability, refusals before any launch, capacity refusal
before the write allocation, the no-`rows x B` allocation watch, small cap-first conversions and the
device row-sum qualification. Nothing about full coupled events, Chastre, checkpoints or MAPLE exchange
is claimed by this suite.
"""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
import math

import numpy as np
import pytest

pytest.importorskip("maple")

from maple.core.backend import (
    cupy_module,
    freeze,
    gpu_execution_available,
    to_device,
    to_host,
)
from maple_syrup.characteristic_sparse import (
    DEFAULT_IMPLEMENTATION,
    IMPLEMENTATIONS,
    DenseConversionRefused,
    SparseCapacityError,
    SparsePhaseState,
    dense_phase_bytes,
    empty_sparse_phase_state,
    host_network_evidence,
    rebind_sparse_phase_state,
    slot_maps,
    slot_maps_from_host_evidence,
    sparse_characteristic_step,
    sparse_from_dense,
    sparse_to_dense,
    validate_sparse_phase_state,
)

from maple_syrup import characteristic_cuda_csr as cuda_csr
from maple_syrup import characteristic_numba_csr as csr_kernels
from maple_syrup.characteristic_transport import (
    DEFAULT_COURANT_MAX,
    MAX_N_BINS,
    PhaseState,
    bin_index,
    characteristic_step,
    empty_phase_state,
)
from maple_syrup.routing import PIT_STORAGE, build_routing_graph
from maple_syrup.sediment_transport import (
    TransportError,
    TransportNetwork,
    TransportStep,
    TransportStepRejected,
    transport_network,
)

DX = 0.5
HAVE_GPU = gpu_execution_available()
cp = cupy_module() if HAVE_GPU else None
needs_gpu = pytest.mark.skipif(not HAVE_GPU, reason="CuPy with a CUDA device is not available; the CUDA sparse "
                                                     "operator is not exercised and no GPU claim is made")
needs_numba = pytest.mark.skipif(not csr_kernels.numba_available(),
                                 reason="Numba is not installed; the CPU CSR reference comparator cannot run")
TRANSPORT_FIELDS = tuple(f.name for f in dataclasses.fields(TransportStep) if f.name not in ("dt_s", "n_substeps"))
RTOL, ATOL, POS_ATOL = 1e-10, 1e-15, 1e-9  # existing isolated GPU characteristic bounds (T11)
# Field classes of a TransportStep for cross-backend comparison (the T11 list of tests/phase7b and the
# conservative compare_events practice): PHYSICAL fields are compared at the T11 bounds; EXACT controls are
# round-to-nearest products of identical FP64 inputs and must be bitwise; ROUND-OFF DIAGNOSTICS (`after - before`
# residuals and their MAPLE bounds) are NOT compared across backends -- each backend must satisfy its OWN unchanged
# `summation_error_bound_kg` policy. A 1e-15 residual difference is round-off, not physics.
PHYSICAL_FIELDS = ("mobile_after_transfer_kg", "deposition_request_kg", "export_request_kg", "decay_deposition_kg",
                   "settled_kg", "internal_transfer_in_kg", "internal_transfer_out_kg", "divergence_kg",
                   "x_face_gross_kg", "y_face_gross_kg", "x_face_net_kg", "y_face_net_kg", "mobile_before_by_class_kg",
                   "mobile_after_by_class_kg", "deposition_request_by_class_kg", "export_request_by_class_kg")
EXACT_CONTROL_FIELDS = ("max_courant", "max_decay_exponent")
ROUNDOFF_DIAGNOSTIC_FIELDS = ("budget_residual_by_class_kg", "budget_tolerance_by_class_kg",
                              "max_cell_balance_residual_kg")
assert set(PHYSICAL_FIELDS) | set(EXACT_CONTROL_FIELDS) | set(ROUNDOFF_DIAGNOSTIC_FIELDS) == set(TRANSPORT_FIELDS)


# --- terrain (same constructions as tests/phase7b) --------------------------------------------------------------
def chain_full(n, dz=0.015625, walls=1.0):
    z = np.repeat(np.arange(n + 2, dtype=np.float64)[:, None] * dz, 3, axis=1)
    z[:, 0] += walls
    z[:, 2] += walls
    return z


def valley_full(ny, nx, sy=0.015625, sx=0.03125):
    rows, cols = np.indices((ny + 2, nx + 2)).astype(np.float64)
    return rows * sy + np.abs(cols - (nx + 1) / 2) * sx


def random_full(rng, ny, nx):
    rows = np.indices((ny + 2, nx + 2))[0].astype(np.float64)
    z = 0.025 * rows + rng.uniform(0.0, 0.02, rows.shape)
    z[:, 0] += 10.0
    z[:, -1] += 10.0
    return z


def south_export(z):
    export = np.zeros(z.shape, dtype=bool)
    export[0, :] = True
    return export


def make_graph(z, *, dx=DX, xp=None, active=None):
    ny, nx = z.shape[0] - 2, z.shape[1] - 2
    return build_routing_graph(z, south_export(z), np.full((ny, nx), 1.0), dx, xp=xp, active_mask=active)


def networks(z, *, dx=DX, active=None):
    """`(host network, device network)` of the same terrain."""
    host = transport_network(make_graph(z, dx=dx, active=active))
    dev = transport_network(make_graph(z, dx=dx, xp=cp, active=active))
    return host, dev


def zeros(shape, nc):
    return np.zeros((*shape, nc), dtype=np.float64)


def fields(shape, nc, *, v=0.0, rate=0.0, settle=False):
    ny, nx = shape
    return (np.full((ny, nx, nc), float(v)), np.full((ny, nx, nc), float(rate)), np.full((ny, nx, nc), bool(settle)))


def dev(array):
    return cp.asarray(array)


# --- comparator: CUDA versus the accepted CPU CSR reference ---------------------------------------------------------
def assert_own_roundoff_bounds(step) -> None:
    """A step's round-off diagnostics against ITS OWN unchanged MAPLE bounds (never against another backend):
    per-class residual within the step's own `summation_error_bound_kg` tolerance, finite non-negative tolerance,
    finite non-negative per-cell balance maximum (its own per-cell bound was enforced before publication)."""
    tr = step.transport
    residual = to_host(tr.budget_residual_by_class_kg)
    tolerance = to_host(tr.budget_tolerance_by_class_kg)
    assert np.all(np.isfinite(residual)) and np.all(np.isfinite(tolerance)) and np.all(tolerance >= 0.0)
    assert np.all(np.abs(residual) <= tolerance), (step.implementation, residual, tolerance)
    cell = float(to_host(tr.max_cell_balance_residual_kg))
    assert math.isfinite(cell) and cell >= 0.0


def assert_within_gpu_bounds(gpu, cpu, *, name=""):
    """PHYSICAL transport fields, the remaining pool and the expanded phase masses within rtol 1e-10 / atol 1e-15;
    positions within 1e-9 m; EXACT controls bitwise (Courant/exponent maxima, integer nnz/peak/substeps, row
    pointers, bins); ROUND-OFF DIAGNOSTICS checked per backend against their own MAPLE bounds only."""
    for field in PHYSICAL_FIELDS:
        np.testing.assert_allclose(to_host(getattr(gpu.transport, field)), getattr(cpu.transport, field), rtol=RTOL,
                                   atol=ATOL, err_msg=f"{name}{field}")
    for field in EXACT_CONTROL_FIELDS:
        np.testing.assert_array_equal(to_host(getattr(gpu.transport, field)), getattr(cpu.transport, field),
                                      err_msg=f"{name}{field}")
    assert_own_roundoff_bounds(gpu)
    assert_own_roundoff_bounds(cpu)
    assert gpu.transport.n_substeps == cpu.transport.n_substeps and gpu.transport.dt_s == cpu.transport.dt_s
    np.testing.assert_allclose(to_host(gpu.mobile_remaining_kg), cpu.mobile_remaining_kg, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(to_host(gpu.arrival_deposition_kg), cpu.arrival_deposition_kg, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(to_host(gpu.crossing_kg), cpu.crossing_kg, rtol=RTOL, atol=ATOL)
    assert gpu.nnz_after == cpu.nnz_after and gpu.peak_nnz == cpu.peak_nnz, (name, "integer controls")
    g = sparse_to_dense(gpu.phase_after)
    c = sparse_to_dense(cpu.phase_after)
    np.testing.assert_allclose(to_host(gpu.mobile_remaining_kg)[..., None] * to_host(g.fraction),
                               cpu.mobile_remaining_kg[..., None] * c.fraction, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(to_host(g.position_m), c.position_m, rtol=0, atol=POS_ATOL)
    np.testing.assert_array_equal(to_host(gpu.phase_after.row_ptr), cpu.phase_after.row_ptr)
    np.testing.assert_array_equal(to_host(gpu.phase_after.bin), cpu.phase_after.bin)
    assert gpu.implementation == "cuda" and cpu.implementation == "numba"


class Twin:
    """Advance the CUDA operator and the CPU CSR reference side by side from the same host-generated inputs."""

    def __init__(self, host_net, dev_net, n_bins, nc):
        self.host_net, self.dev_net, self.nc = host_net, dev_net, nc
        self.shape = host_net.shape
        self.c_mobile = zeros(self.shape, nc)
        self.c_phase = empty_sparse_phase_state(host_net, nc, n_bins)
        self.g_mobile = cp.zeros((*self.shape, nc), dtype=np.float64)
        self.g_phase = empty_sparse_phase_state(dev_net, nc, n_bins)
        self.dep = zeros(self.shape, nc)
        self.export = zeros(self.shape, nc)
        self.crossing = zeros(self.shape, nc)

    def step(self, pickup, v, rate, settle, dt, **kwargs):
        c = sparse_characteristic_step(self.host_net, self.c_mobile, self.c_phase, pickup, v, rate, settle, dt, **kwargs)
        g = sparse_characteristic_step(self.dev_net, self.g_mobile, self.g_phase, dev(pickup), dev(v), dev(rate),
                                       dev(settle), dt, implementation="cuda", **kwargs)
        assert_within_gpu_bounds(g, c)
        self.c_mobile, self.c_phase = c.mobile_remaining_kg, c.phase_after
        self.g_mobile, self.g_phase = g.mobile_remaining_kg, g.phase_after
        self.dep = self.dep + to_host(g.transport.deposition_request_kg)
        self.export = self.export + to_host(g.transport.export_request_kg)
        self.crossing = self.crossing + to_host(g.crossing_kg)
        return g, c


# =============================================================================================================
# Host-only tests (no device required)
# =============================================================================================================
def test_cuda_requested_on_a_host_network_is_refused_and_cuda_options_are_cuda_only():
    graph = make_graph(chain_full(3))
    net = transport_network(graph)
    assert IMPLEMENTATIONS == ("numba", "cuda") and DEFAULT_IMPLEMENTATION == "numba"
    phase = empty_sparse_phase_state(net, 1, 8)
    args = (zeros(graph.shape, 1), phase, zeros(graph.shape, 1), zeros(graph.shape, 1), zeros(graph.shape, 1),
            np.zeros((*graph.shape, 1), bool), 1.0)
    with pytest.raises(TransportError):  # CudaUnavailableError without CuPy, namespace refusal with it; never a run
        sparse_characteristic_step(net, *args, implementation="cuda")
    for option in ({"max_launched_threads": 1024}, {"block_threads": 64}):
        with pytest.raises(TransportError, match="CUDA launch options"):
            sparse_characteristic_step(net, *args, **option)
    with pytest.raises(TransportError):
        sparse_characteristic_step(net, *args, implementation="array")


def test_kernel_source_is_specialised_per_bin_count_and_estimates_are_explicit():
    s32 = cuda_csr.kernel_source(32)
    assert "#define MS_NB 32" in s32 and "__dadd_rn" in s32 and "atomicAdd" not in s32 and "fma(" not in s32
    assert "ms_csr_count" in s32 and "ms_csr_write" in s32 and "ms_csr_rowsum_normalize" in s32
    assert cuda_csr.kernel_source(128) != s32 and "#define MS_NB 128" in cuda_csr.kernel_source(128)
    assert "#define MS_NB 1\n" in cuda_csr.kernel_source(1)
    for bad in (0, MAX_N_BINS + 1, 1.5, True):
        with pytest.raises(TransportError):
            cuda_csr.kernel_source(bad)
    est32 = cuda_csr.estimate_local_scratch_bytes(32)
    assert est32["ms_csr_write"] == 5 * 33 * (4 + 8 + 8) + 5 * 32 * 8 + 4
    assert est32["ms_csr_rowsum_normalize"] == 32 * 8 and est32["ms_csr_count"] == 4 + 32
    est128 = cuda_csr.estimate_local_scratch_bytes(128)
    assert est128["ms_csr_write"] == 5 * 129 * 20 + 5 * 128 * 8 + 16 > est32["ms_csr_write"]
    prov = cuda_csr.kernel_provenance(32)
    assert prov["source_sha256"] == hashlib.sha256(s32.encode()).hexdigest()
    assert prov["fastmath"] is False and prov["fma"] is False and prov["float_atomics"] is False
    assert "--fmad=false" in prov["compile_options"] and prov["n_bins"] == 32


def test_launch_shape_is_clamped_persistent_grid_stride():
    assert cuda_csr.launch_shape(10) == (1, 128, 128)
    assert cuda_csr.launch_shape(1000) == (8, 128, 1024)
    assert cuda_csr.launch_shape(10_000_000) == (64, 128, 8192)  # clamped: never n_rows threads
    assert cuda_csr.launch_shape(10_000_000, max_launched_threads=4096) == (32, 128, 4096)
    assert cuda_csr.launch_shape(10_000_000, max_launched_threads=8192, block_threads=256) == (32, 256, 8192)
    assert cuda_csr.DEFAULT_MAX_LAUNCHED_THREADS == 8192 and cuda_csr.BLOCK_THREADS == 128
    for kwargs in ({"block_threads": 100}, {"block_threads": 16}, {"block_threads": 2048},
                   {"max_launched_threads": 64}, {"max_launched_threads": True}, {"block_threads": 128.0}):
        with pytest.raises(TransportError):
            cuda_csr.launch_shape(1000, **kwargs)


def test_preflight_reports_bytes_and_requires_an_explicit_cap_for_large_cases():
    small = cuda_csr.cuda_sparse_preflight((50, 20), 1000, 6, 128)
    assert small["n_rows"] == 6000 and small["requires_explicit_max_nnz"] is False
    assert small["full_occupancy_entry_bytes"] == 6000 * 128 * 17 and small["csr_entries_sized_for"].startswith("full")
    assert small["launch"] == {"grid_blocks": 47, "block_threads": 128, "launched_threads": 6016}
    assert small["local_scratch_estimate_bytes_launched"]["ms_csr_write"] == 6016 * small[
        "local_scratch_estimate_bytes_per_thread"]["ms_csr_write"]
    assert small["dense_equivalent_bytes"] == dense_phase_bytes((50, 20), 6, 128)
    chastre = cuda_csr.cuda_sparse_preflight((1394, 1603), 1_140_169, 6, 32)
    assert chastre["requires_explicit_max_nnz"] is True and chastre["n_rows"] == 6_841_014
    assert chastre["full_occupancy_entry_bytes"] == 6_841_014 * 32 * 17
    assert chastre["launch"]["launched_threads"] == 8192  # bounded local scratch independent of 6.8 M rows
    capped = cuda_csr.cuda_sparse_preflight((1394, 1603), 1_140_169, 6, 32, max_nnz=4 * 6_841_014)
    assert capped["csr_entries_sized_for"] == "max_nnz"
    assert capped["csr_entry_bytes_two_generations"] == 2 * 4 * 6_841_014 * 17
    assert capped["requires_explicit_max_nnz"] is True  # the requirement is about the case, satisfied by the caller
    with pytest.raises(TransportError):
        cuda_csr.cuda_sparse_preflight((50, 20), 0, 6, 32)
    with pytest.raises(TransportError):
        cuda_csr.cuda_sparse_preflight((50, 20), 1000, 6, 129)


def test_cupy_unavailable_is_an_explicit_typed_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("no cupy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(cuda_csr.CudaUnavailableError) as info:
        cuda_csr._cupy()
    assert isinstance(info.value, TransportError) and "no fallback" in str(info.value)
    with pytest.raises(cuda_csr.CudaUnavailableError):
        cuda_csr.load_kernels(32)
    graph = make_graph(chain_full(3))
    net = transport_network(graph)
    phase = empty_sparse_phase_state(net, 1, 8)
    with pytest.raises(TransportError):
        sparse_characteristic_step(net, zeros(graph.shape, 1), phase, zeros(graph.shape, 1), zeros(graph.shape, 1),
                                   zeros(graph.shape, 1), np.zeros((*graph.shape, 1), bool), 1.0, implementation="cuda")


def test_slot_maps_from_host_evidence_matches_the_host_helper_and_rebind_rules():
    rng = np.random.default_rng(3)
    graph = make_graph(random_full(rng, 5, 4))
    net = transport_network(graph)
    maps = slot_maps(net)
    evidence = slot_maps_from_host_evidence(net.active_host, net.receiver_index, net.outlet_flat)
    for name in ("cell_of_slot", "slot_of_cell", "receiver_slot", "outlet"):
        np.testing.assert_array_equal(getattr(maps, name), getattr(evidence, name))
    with pytest.raises(TransportError):
        slot_maps_from_host_evidence(net.active_host, np.full(net.n_cells, -5, dtype=np.int64), net.outlet_flat)
    phase = empty_sparse_phase_state(net, 2, 8)
    same = rebind_sparse_phase_state(phase, net)
    assert same.network_sha256 == net.graph_input_sha256 and same.nnz == 0
    other = transport_network(make_graph(random_full(np.random.default_rng(4), 5, 4)))  # same shape, all active
    rebound = rebind_sparse_phase_state(phase, other)
    assert rebound.network_sha256 == other.graph_input_sha256 != phase.network_sha256
    np.testing.assert_array_equal(rebound.cell_of_slot, phase.cell_of_slot)
    validate_sparse_phase_state(rebound, zeros(graph.shape, 2), other)
    with pytest.raises(TransportError):
        validate_sparse_phase_state(phase, zeros(graph.shape, 2), other)  # without the explicit rebind: digest refusal
    active = np.ones((5, 4), dtype=bool)
    active[4, 1] = False  # top row: no cell drains into it, so the graph stays valid with one fewer active cell
    fewer = transport_network(make_graph(random_full(np.random.default_rng(4), 5, 4), active=active))
    assert fewer.n_active == 19
    with pytest.raises(TransportError, match="active"):
        rebind_sparse_phase_state(phase, fewer)
    with pytest.raises(TransportError, match="dx"):
        rebind_sparse_phase_state(phase, transport_network(make_graph(random_full(np.random.default_rng(4), 5, 4), dx=0.25)))
    with pytest.raises(TransportError):
        rebind_sparse_phase_state(empty_phase_state(graph.shape, 2, 8, DX), net)


# =============================================================================================================
# GPU tests (skip without a device)
# =============================================================================================================
@needs_gpu
def test_cuda_state_lifecycle_binding_and_device_validation():
    host_net, dev_net = networks(valley_full(4, 3))
    state = empty_sparse_phase_state(dev_net, 2, 8)
    assert isinstance(state, SparsePhaseState) and state.xp is cp and state.nnz == 0 and state.n_rows == 24
    for name in ("row_ptr", "bin", "fraction", "position_m", "cell_of_slot", "slot_of_cell"):
        assert type(getattr(state, name)) is cp.ndarray, name
    assert state.row_ptr.dtype == np.int64 and state.bin.dtype == np.uint8 and state.cell_of_slot.dtype == np.int64
    assert state.n_nonempty_rows == 0 and state.occupancy()["nnz"] == 0
    np.testing.assert_array_equal(to_host(state.cell_of_slot), slot_maps(host_net).cell_of_slot)
    binding = cuda_csr.bind_network(dev_net)
    assert state.cell_of_slot is binding.cell_of_slot  # bound once, shared, no per-step upload
    assert cuda_csr.bind_network(dev_net) is binding and binding.summary()["n_active"] == 12
    assert binding.host_to_device_bytes >= binding.cell_of_slot.nbytes + binding.slot_of_cell.nbytes
    assert cuda_csr.release_binding(dev_net) is True and cuda_csr.release_binding(dev_net) is False
    assert cuda_csr.bind_network(dev_net) is not binding
    validate_sparse_phase_state(state, cp.zeros((4, 3, 2)), dev_net)
    with pytest.raises(TransportError):
        validate_sparse_phase_state(state, cp.zeros((4, 3, 2)) + 1.0, dev_net)  # positive pool needs stored rows
    for bad in (zeros((4, 3), 2), cp.zeros((4, 3, 2), dtype=np.float32), cp.zeros((4, 3, 1)), cp.zeros((4, 3, 2)) - 1.0,
                cp.zeros((4, 3, 2)) + cp.nan):
        with pytest.raises(TransportError):
            validate_sparse_phase_state(state, bad, dev_net)
    with pytest.raises(TransportError):
        validate_sparse_phase_state(empty_sparse_phase_state(host_net, 2, 8), cp.zeros((4, 3, 2)), dev_net)
    with pytest.raises(TransportError):
        validate_sparse_phase_state(state, zeros((4, 3), 2), host_net)  # CuPy phase on a host network
    with pytest.raises(TransportError, match="host NumPy"):
        slot_maps(dev_net)  # the host helper stays host-only
    # corrupted device states are refused by the device validator (flags), nothing downloaded
    rng = np.random.default_rng(1)
    twin = Twin(host_net, dev_net, 8, 2)
    for _ in range(3):
        twin.step(rng.uniform(0.2, 1.0, (4, 3, 2)), rng.uniform(0.02, 0.1, (4, 3, 2)), rng.uniform(0, 2, (4, 3, 2)),
                  np.zeros((4, 3, 2), bool), 1.0)
    good, mobile = twin.g_phase, twin.g_mobile
    validate_sparse_phase_state(good, mobile, dev_net)
    e0 = int(to_host(good.row_ptr)[1])

    def with_value(name, index, value):
        arr = cp.array(getattr(good, name), copy=True)
        arr[index] = value
        return arr

    bad_states = {
        "row_ptr[0]": dataclasses.replace(good, row_ptr=with_value("row_ptr", 0, 1)),
        "row_ptr end": dataclasses.replace(good, row_ptr=with_value("row_ptr", -1, good.nnz + 1)),
        "row_ptr dtype": dataclasses.replace(good, row_ptr=good.row_ptr.astype(np.int32)),
        "bin >= B": dataclasses.replace(good, bin=with_value("bin", e0, 8)),
        "bin dtype": dataclasses.replace(good, bin=good.bin.astype(np.int64)),
        "fraction zero": dataclasses.replace(good, fraction=with_value("fraction", e0, 0.0)),
        "fraction sum": dataclasses.replace(good, fraction=with_value("fraction", e0, 0.5 * to_host(good.fraction)[e0])),
        "position >= dx": dataclasses.replace(good, position_m=with_value("position_m", e0, DX)),
        "position negative": dataclasses.replace(good, position_m=with_value("position_m", e0, -1e-3)),
        "position bin": dataclasses.replace(good, position_m=with_value(
            "position_m", e0, ((int(to_host(good.bin)[e0]) + 1) % 8 + 0.5) * DX / 8)),
        "host arrays": dataclasses.replace(good, fraction=to_host(good.fraction)),
        "digest": dataclasses.replace(good, network_sha256="0" * 64),
        "n_bins": dataclasses.replace(good, n_bins=129),
        "namespace": dataclasses.replace(good, xp=np),
    }
    for name, bad in bad_states.items():
        with pytest.raises(TransportError):
            validate_sparse_phase_state(bad, mobile, dev_net)
        with pytest.raises(TransportError):
            sparse_characteristic_step(dev_net, mobile, bad, cp.zeros_like(mobile), cp.zeros_like(mobile) + 0.05,
                                       cp.zeros_like(mobile) + 1.0, cp.zeros(mobile.shape, bool), 1.0,
                                       implementation="cuda")
    with pytest.raises(TransportError):
        validate_sparse_phase_state(good, mobile * 0.0, dev_net)


@needs_gpu
@needs_numba
@pytest.mark.parametrize("n_substeps", [1, 2])
@pytest.mark.parametrize("n_bins", [1, 8, 32, 128])
@pytest.mark.parametrize("nc", [1, 3, 6, 7])
def test_cuda_matches_cpu_csr_reference_within_bounds(nc, n_bins, n_substeps):
    rng = np.random.default_rng(100 + nc + n_bins + n_substeps)
    host_net, dev_net = networks(random_full(rng, 7, 6))
    ny, nx = host_net.shape
    twin = Twin(host_net, dev_net, n_bins, nc)
    total_pickup = 0.0
    for _ in range(8):
        pickup = rng.uniform(0.0, 1.0, (ny, nx, nc)) * (rng.uniform(size=(ny, nx, nc)) > 0.3)
        v = rng.uniform(0.0, 0.25 * n_substeps, (ny, nx, nc))
        rate = rng.uniform(0.0, 20.0, (ny, nx, nc)) * (rng.uniform(size=(ny, nx, nc)) > 0.2)
        settle = rng.uniform(size=(ny, nx, nc)) < 0.1
        total_pickup += float(pickup.sum())
        g, _c = twin.step(pickup, v, rate, settle, 1.0, n_substeps=n_substeps)
        assert g.device_report["n_bins"] == n_bins and g.device_report["launch"]["launched_threads"] <= 8192
        assert g.device_report["scalar_reads_nnz"] == n_substeps
    np.testing.assert_allclose(twin.dep.sum() + twin.export.sum() + float(to_host(twin.g_mobile).sum()), total_pickup,
                               rtol=1e-12)
    assert twin.g_phase.n_classes == nc  # every grain class stays dynamic, none dropped


@needs_gpu
@needs_numba
def test_cuda_analytic_impulse_two_cell_dry_arrival_export_and_log_spread():
    # T01: constant-law impulse on a chain, B = 32
    n, L, v = 6, 0.05, 0.01
    host_net, dev_net = networks(chain_full(n))
    vel, rate, settle = fields(host_net.shape, 1, v=v, rate=1.0 / L)
    twin = Twin(host_net, dev_net, 32, 1)
    pickup = zeros(host_net.shape, 1)
    pickup[n - 1, 0, 0] = 1.0
    first = None
    for t in range(1, 601):
        g, _c = twin.step(pickup, vel, rate, settle, 1.0)
        pickup = zeros(host_net.shape, 1)
        if first is None and float(to_host(g.transport.export_request_kg).sum()) > 0.0:
            first = t
    target = np.exp(-np.arange(1, n + 1) * DX / L)
    np.testing.assert_allclose(twin.crossing[::-1, 0, 0], target, rtol=1e-12, atol=1e-35)
    assert first == round(n * DX / v) and float(to_host(twin.g_mobile).sum()) <= 1e-30
    assert abs(twin.dep.sum() + twin.export.sum() - 1.0) <= 1e-13
    # T03: differing receiver laws, exact two-cell split
    host_net, dev_net = networks(chain_full(2))
    v1, r1, v2, r2 = 0.2, 3.0, 0.1, 7.0
    velf = zeros(host_net.shape, 1)
    ratef = zeros(host_net.shape, 1)
    velf[1, 0, 0], ratef[1, 0, 0] = v1, r1
    velf[0, 0, 0], ratef[0, 0, 0] = v2, r2
    settle = np.zeros((*host_net.shape, 1), bool)
    twin = Twin(host_net, dev_net, 8, 1)
    pickup = zeros(host_net.shape, 1)
    pickup[1, 0, 0] = 1.0
    place = zeros(host_net.shape, 1)
    place[1, 0, 0] = 0.1875
    twin.step(pickup, place, zeros(host_net.shape, 1), settle, 1.0)
    twin.step(zeros(host_net.shape, 1), place, zeros(host_net.shape, 1), settle, 1.0)
    assert float(to_host(twin.g_phase.position_m)[0]) == 0.375  # RN-only arithmetic: exact on both backends
    g, _c = twin.step(zeros(host_net.shape, 1), velf, ratef, settle, 1.0)
    x0 = 0.375
    tau = (DX - x0) / v1
    s1 = math.exp(-r1 * (DX - x0))
    d = v2 * (1.0 - tau)
    s2 = s1 * math.exp(-r2 * d)
    tr = g.transport
    np.testing.assert_allclose(to_host(tr.deposition_request_kg)[1, 0, 0], 1.0 - s1, rtol=1e-12)
    np.testing.assert_allclose(to_host(tr.internal_transfer_in_kg)[0, 0, 0], s1, rtol=1e-12)
    np.testing.assert_allclose(to_host(g.mobile_remaining_kg)[0, 0, 0], s2, rtol=1e-12)
    np.testing.assert_allclose(to_host(g.phase_after.position_m)[0], d, rtol=1e-12)
    assert int(to_host(g.phase_after.bin)[0]) == int(bin_index(np.array(d), DX, 8, np)) and g.nnz_after == 1
    # T06: dry receiver settles arrivals; export only at the outlet
    vel, rate, settle = fields(host_net.shape, 1, v=0.2, rate=1.0)
    settle[0] = True
    twin = Twin(host_net, dev_net, 8, 1)
    pickup = zeros(host_net.shape, 1)
    pickup[1, 0, 0] = 1.0
    twin.step(pickup, vel, rate, settle, 1.0)
    twin.step(zeros(host_net.shape, 1), vel, rate, settle, 1.0)
    g, _c = twin.step(zeros(host_net.shape, 1), vel, rate, settle, 1.0)
    np.testing.assert_allclose(to_host(g.transport.settled_kg)[0, 0, 0], math.exp(-DX), rtol=1e-12)
    assert g.nnz_after == 0 and float(to_host(g.transport.export_request_kg).sum()) == 0.0
    vel, rate, settle = fields(host_net.shape, 1, v=0.1, rate=0.0)
    twin = Twin(host_net, dev_net, 8, 1)
    pickup = zeros(host_net.shape, 1)
    pickup[0, 0, 0] = 1.0  # outlet cell: exports after dx / v steps, nowhere else
    exported = 0.0
    for _ in range(6):
        g, _c = twin.step(pickup, vel, rate, settle, 1.0)
        pickup = zeros(host_net.shape, 1)
        ex = to_host(g.transport.export_request_kg)
        assert ex[1].sum() == 0.0
        exported += ex.sum()
    assert exported == 1.0 and twin.g_phase.nnz == 0
    # T12b: extreme mass/position spread in the logarithmic mean (wide branch), dx = 0.01
    dx, L = 0.01, 1.0e-4
    host_net, dev_net = networks(chain_full(2), dx=dx)
    old_mass, old_x, new_mass = 1.0e-20, 0.009, 1.0
    mobile = zeros(host_net.shape, 1)
    mobile[1, 0, 0] = old_mass
    fraction = np.zeros((*host_net.shape, 1, 1))
    fraction[..., 0] = 1.0
    position = np.zeros((*host_net.shape, 1, 1))
    position[1, 0, 0, 0] = old_x
    dense_host = dataclasses.replace(empty_phase_state(host_net.shape, 1, 1, dx), fraction=fraction, position_m=position)
    dense_dev = PhaseState(fraction=dev(fraction), position_m=dev(position), dx_m=dx, n_bins=1, xp=cp)
    c_phase = sparse_from_dense(dense_host, mobile, host_net)
    g_phase = sparse_from_dense(dense_dev, dev(mobile), dev_net)
    pickup = zeros(host_net.shape, 1)
    pickup[1, 0, 0] = new_mass
    vel, rate, settle = fields(host_net.shape, 1, v=0.0, rate=1.0 / L)
    c = sparse_characteristic_step(host_net, mobile, c_phase, pickup, vel, rate, settle, 1.0)
    g = sparse_characteristic_step(dev_net, dev(mobile), g_phase, dev(pickup), dev(vel), dev(rate), dev(settle), 1.0,
                                   implementation="cuda")
    assert_within_gpu_bounds(g, c)
    exact = L * math.log((old_mass * math.exp(old_x / L) + new_mass) / (old_mass + new_mass))
    np.testing.assert_allclose(to_host(g.phase_after.position_m)[0], exact, rtol=1e-12)
    eventual = old_mass * math.exp(-(dx - old_x) / L) + new_mass * math.exp(-dx / L)
    twin = Twin(host_net, dev_net, 1, 1)
    twin.c_mobile, twin.c_phase, twin.g_mobile, twin.g_phase = c.mobile_remaining_kg, c.phase_after, g.mobile_remaining_kg, g.phase_after
    vel, rate, settle = fields(host_net.shape, 1, v=0.005, rate=1.0 / L)
    for _ in range(4):
        twin.step(zeros(host_net.shape, 1), vel, rate, settle, 1.0)
    np.testing.assert_allclose(twin.crossing[1, 0, 0], eventual, rtol=1e-9)
    assert float(to_host(twin.g_mobile)[1, 0, 0]) == 0.0


@needs_gpu
def test_cuda_repeats_are_raw_byte_exact_and_inputs_are_immutable():
    rng = np.random.default_rng(23)
    _host_net, dev_net = networks(random_full(rng, 6, 4))
    ny, nx = dev_net.shape
    nc = 3
    phase = empty_sparse_phase_state(dev_net, nc, 32)
    inputs = [dev(a) for a in (rng.uniform(0, 1, (ny, nx, nc)), rng.uniform(0, 0.2, (ny, nx, nc)),
                               rng.uniform(0, 5, (ny, nx, nc)), rng.uniform(size=(ny, nx, nc)) < 0.1)]
    mobile = cp.zeros((ny, nx, nc), dtype=np.float64)
    snapshot = [to_host(a).tobytes() for a in (mobile, *inputs)]
    first = sparse_characteristic_step(dev_net, mobile, phase, *inputs, 1.0, n_substeps=2, implementation="cuda")
    second = sparse_characteristic_step(dev_net, mobile, phase, *inputs, 1.0, n_substeps=2, implementation="cuda")
    assert first.nnz_after == second.nnz_after > 0
    for name in TRANSPORT_FIELDS:
        assert to_host(getattr(first.transport, name)).tobytes() == to_host(getattr(second.transport, name)).tobytes(), name
    for name in ("row_ptr", "bin", "fraction", "position_m"):
        assert to_host(getattr(first.phase_after, name)).tobytes() == to_host(getattr(second.phase_after, name)).tobytes()
    assert to_host(first.mobile_remaining_kg).tobytes() == to_host(second.mobile_remaining_kg).tobytes()
    third = sparse_characteristic_step(dev_net, first.mobile_remaining_kg, first.phase_after, *inputs, 1.0, n_substeps=2,
                                       implementation="cuda")
    fourth = sparse_characteristic_step(dev_net, second.mobile_remaining_kg, second.phase_after, *inputs, 1.0,
                                        n_substeps=2, implementation="cuda")
    assert to_host(third.phase_after.fraction).tobytes() == to_host(fourth.phase_after.fraction).tobytes()
    assert [to_host(a).tobytes() for a in (mobile, *inputs)] == snapshot  # inputs untouched
    assert to_host(phase.row_ptr).tobytes() == to_host(empty_sparse_phase_state(dev_net, nc, 32).row_ptr).tobytes()
    assert first.device_report["kernel_attributes"]["ms_csr_write"]["local_size_bytes"] >= 0
    assert first.device_report["row_sum_self_check"]["byte_exact"] is True
    assert first.device_report["local_scratch_compiled_bytes_launched"]["ms_csr_write"] == first.device_report[
        "kernel_attributes"]["ms_csr_write"]["local_size_bytes"] * first.device_report["launch"]["launched_threads"]


@needs_gpu
def test_cuda_refusals_happen_before_any_kernel_launch(monkeypatch):
    host_net, dev_net = networks(valley_full(4, 3))
    ny, nx = dev_net.shape
    nc = 2
    good = {"mobile": cp.full((ny, nx, nc), 0.5), "pickup": cp.full((ny, nx, nc), 0.1), "v": cp.full((ny, nx, nc), 0.1),
            "rate": cp.full((ny, nx, nc), 2.0), "settle": cp.zeros((ny, nx, nc), bool)}
    phase0 = empty_sparse_phase_state(dev_net, nc, 8)
    ok = sparse_characteristic_step(dev_net, good["mobile"] * 0.0, phase0, good["mobile"], good["v"], good["rate"],
                                    good["settle"], 1.0, implementation="cuda")
    base = {"mobile": ok.mobile_remaining_kg, "phase": ok.phase_after, **{k: good[k] for k in ("pickup", "v", "rate", "settle")}}
    snapshot = {k: to_host(v).tobytes() for k, v in base.items() if k != "phase"}
    phase_bytes = to_host(base["phase"].fraction).tobytes()

    def call(**changes):
        args = dict(base)
        args.update(changes)
        return sparse_characteristic_step(args.get("network", dev_net), args["mobile"], args["phase"],
                                          args["pickup"], args["v"], args["rate"], args["settle"], args.get("dt", 1.0),
                                          n_substeps=args.get("n_substeps", 1),
                                          courant_max=args.get("courant_max", DEFAULT_COURANT_MAX),
                                          implementation="cuda", max_nnz=args.get("max_nnz"),
                                          max_launched_threads=args.get("max_launched_threads"))

    call()

    def no_launch(*args, **kwargs):
        raise AssertionError("a kernel was requested before the refusal")

    monkeypatch.setattr(cuda_csr, "_function", no_launch)
    bad_cases = {
        "nonfinite mobile": {"mobile": cp.where(cp.arange(nc) == 0, cp.nan, base["mobile"])},
        "negative pickup": {"pickup": -base["pickup"]},
        "negative velocity": {"v": -base["v"]},
        "nonfinite rate": {"rate": base["rate"] * cp.inf},
        "settle dtype": {"settle": base["settle"].astype(np.float64)},
        "shape": {"pickup": base["pickup"][:, :, :1]},
        "host array": {"pickup": to_host(base["pickup"])},
        "float32": {"mobile": base["mobile"].astype(np.float32)},
        "non-contiguous": {"v": cp.asfortranarray(base["v"]) if nc > 1 else base["v"][:, ::-1]},
        "dense phase": {"phase": empty_phase_state(dev_net.shape, nc, 8, DX, xp=cp)},
        "host phase": {"phase": empty_sparse_phase_state(host_net, nc, 8)},
        "zero pool with stored bins": {"mobile": base["mobile"] * 0.0},
        "dt": {"dt": 0.0},
        "substeps": {"n_substeps": 0},
        "courant_max": {"courant_max": 0.75},
        "max_nnz float": {"max_nnz": 1.5},
        "launch threads too small": {"max_launched_threads": 64},
        "host network": {"network": host_net},
    }
    for changes in bad_cases.values():
        with pytest.raises(TransportError):
            call(**changes)
    with pytest.raises(TransportStepRejected):
        call(v=cp.full((ny, nx, nc), 0.3))  # a = 0.6 > 0.5: recoverable rejection, no launch
    monkeypatch.undo()
    call(v=cp.full((ny, nx, nc), 0.3), n_substeps=2)
    with pytest.raises(TransportError):  # a CuPy network on the default Numba path is refused, never transferred
        sparse_characteristic_step(dev_net, base["mobile"], base["phase"], base["pickup"], base["v"], base["rate"],
                                   base["settle"], 1.0)
    assert {k: to_host(v).tobytes() for k, v in base.items() if k != "phase"} == snapshot
    assert to_host(base["phase"].fraction).tobytes() == phase_bytes


@needs_gpu
def test_cuda_capacity_refusal_precedes_the_write_allocation_and_names_the_requirement(monkeypatch):
    _host_net, dev_net = networks(chain_full(6))
    ny, nx = dev_net.shape
    vel, rate, settle = (dev(a) for a in fields(dev_net.shape, 1, v=0.1, rate=1.0))
    phase = empty_sparse_phase_state(dev_net, 1, 8)
    pickup = cp.ones((ny, nx, 1))
    mobile = cp.zeros((ny, nx, 1))
    before = (to_host(pickup).tobytes(), to_host(phase.row_ptr).tobytes())
    uint8_requests: list[int] = []
    real_empty = cp.empty

    def recording_empty(shape, dtype=float, *args, **kwargs):
        if np.dtype(dtype) == np.uint8:
            uint8_requests.append(int(np.prod(np.atleast_1d(shape))))
        return real_empty(shape, dtype, *args, **kwargs)

    monkeypatch.setattr(cp, "empty", recording_empty)
    with pytest.raises(SparseCapacityError) as info:
        sparse_characteristic_step(dev_net, mobile, phase, pickup, vel, rate, settle, 1.0, implementation="cuda", max_nnz=5)
    assert uint8_requests == []  # the bins array (the first write allocation) was never requested
    err = info.value
    assert err.required_nnz == 6 and err.max_nnz == 5 and err.substep == 0 and err.n_substeps == 1
    assert (to_host(pickup).tobytes(), to_host(phase.row_ptr).tobytes()) == before
    ok = sparse_characteristic_step(dev_net, mobile, phase, pickup, vel, rate, settle, 1.0, implementation="cuda", max_nnz=6)
    assert uint8_requests == [6] and ok.nnz_after == 6 == ok.peak_nnz
    monkeypatch.undo()
    with pytest.raises(SparseCapacityError) as info2:
        sparse_characteristic_step(dev_net, ok.mobile_remaining_kg, ok.phase_after, pickup, vel, rate, settle, 1.0,
                                   n_substeps=2, max_nnz=6, implementation="cuda")
    assert info2.value.required_nnz > 6 and info2.value.n_substeps == 2
    two = sparse_characteristic_step(dev_net, ok.mobile_remaining_kg, ok.phase_after, pickup, vel, rate, settle, 1.0,
                                     n_substeps=2, max_nnz=info2.value.required_nnz, implementation="cuda")
    assert two.peak_nnz == info2.value.required_nnz >= two.nnz_after
    # a large full-occupancy case needs an explicit cap (the threshold is lowered here to make the rule testable)
    with pytest.raises(TransportError, match="explicit max_nnz"):
        cuda_csr.cuda_sparse_step(dev_net, mobile, phase, pickup, vel, rate, settle, 1.0, unbounded_nnz_bytes=1)
    capped = cuda_csr.cuda_sparse_step(dev_net, mobile, phase, pickup, vel, rate, settle, 1.0, unbounded_nnz_bytes=1,
                                       max_nnz=6)
    assert capped.nnz_after == 6


@needs_gpu
def test_cuda_small_nnz_high_bin_count_allocates_no_rows_times_bins_array(monkeypatch):
    rng = np.random.default_rng(77)
    _host_net, dev_net = networks(random_full(rng, 50, 20))
    ny, nx = dev_net.shape
    nc, nb = 6, 128
    n_rows = ny * nx * nc
    vel, rate, settle = (dev(a) for a in fields(dev_net.shape, nc, v=0.1, rate=2.0))
    pickup = zeros(dev_net.shape, nc)
    pickup[40, 7, 0] = 1.0
    pickup[25, 12, 3] = 2.0
    pickup = dev(pickup)
    phase = empty_sparse_phase_state(dev_net, nc, nb)
    mobile = cp.zeros((ny, nx, nc), dtype=np.float64)
    warm = sparse_characteristic_step(dev_net, mobile, phase, pickup, vel, rate, settle, 1.0, implementation="cuda")
    sizes: list[int] = []
    originals = {name: getattr(cp, name) for name in ("zeros", "empty", "full", "ones")}

    def recorder(name):
        real = originals[name]

        def wrapped(shape, *args, **kwargs):
            sizes.append(int(np.prod(np.atleast_1d(shape))))
            return real(shape, *args, **kwargs)

        return wrapped

    for name in originals:
        monkeypatch.setattr(cp, name, recorder(name))
    res = sparse_characteristic_step(dev_net, mobile, phase, pickup, vel, rate, settle, 1.0, implementation="cuda")
    monkeypatch.undo()
    assert res.nnz_after == 2 and res.peak_nnz == 2
    assert max(sizes) < n_rows * nb // 8, f"an allocation of {max(sizes)} elements approaches a rows x B array"
    assert max(sizes) <= max(2 * ny * nx * nc, (ny + 1) * nx * nc, ny * (nx + 1) * nc) + n_rows + 1
    report = res.device_report
    assert report["launch"]["launched_threads"] == min(8192, 128 * ((n_rows + 127) // 128)) == 6016
    assert report["local_scratch_estimate_bytes_launched"]["ms_csr_write"] == 6016 * cuda_csr.estimate_local_scratch_bytes(nb)["ms_csr_write"]
    assert set(report["kernel_attributes"]) == set(cuda_csr.KERNEL_NAMES)
    assert to_host(res.phase_after.fraction).tobytes() == to_host(warm.phase_after.fraction).tobytes()
    occupancy = res.phase_after.occupancy()
    assert occupancy["nnz"] == 2 and occupancy["n_nonempty_rows"] == 2 and occupancy["dense_equivalent_bytes"] == 2 * n_rows * nb * 8
    with pytest.raises(DenseConversionRefused):
        sparse_to_dense(res.phase_after, max_bytes=2 * n_rows * nb * 8 - 1)


@needs_gpu
@needs_numba
def test_cuda_small_conversions_are_cap_first_and_agree_with_host_and_dense_cupy_paths(monkeypatch):
    rng = np.random.default_rng(13)
    host_net, dev_net = networks(random_full(rng, 5, 4))
    ny, nx = host_net.shape
    nc, nb = 3, 8
    twin = Twin(host_net, dev_net, nb, nc)
    dense_mobile = cp.zeros((ny, nx, nc), dtype=np.float64)
    dense_phase = empty_phase_state(dev_net.shape, nc, nb, DX, xp=cp)
    for _ in range(4):
        pickup = rng.uniform(0.0, 1.0, (ny, nx, nc)) * (rng.uniform(size=(ny, nx, nc)) > 0.5)
        v, rate, settle = rng.uniform(0, 0.2, (ny, nx, nc)), rng.uniform(0, 3, (ny, nx, nc)), rng.uniform(size=(ny, nx, nc)) < 0.1
        g, _c = twin.step(pickup, v, rate, settle, 1.0)
        d = characteristic_step(dev_net, dense_mobile, dense_phase, dev(pickup), dev(v), dev(rate), dev(settle), 1.0,
                                implementation="array")  # dense CuPy array path, same device
        dense_mobile, dense_phase = d.mobile_remaining_kg, d.phase_after
        for name in PHYSICAL_FIELDS:
            np.testing.assert_allclose(to_host(getattr(g.transport, name)), to_host(getattr(d.transport, name)),
                                       rtol=RTOL, atol=ATOL, err_msg=name)
        residual, tolerance = to_host(d.transport.budget_residual_by_class_kg), to_host(d.transport.budget_tolerance_by_class_kg)
        assert np.all(np.abs(residual) <= tolerance)  # the dense path meets its own MAPLE bound too
        expanded = sparse_to_dense(g.phase_after)
        np.testing.assert_allclose(to_host(g.mobile_remaining_kg)[..., None] * to_host(expanded.fraction),
                                   to_host(d.mobile_remaining_kg)[..., None] * to_host(d.phase_after.fraction), rtol=RTOL,
                                   atol=ATOL)
        np.testing.assert_allclose(to_host(expanded.position_m), to_host(d.phase_after.position_m), rtol=0, atol=POS_ATOL)
    state, mobile = twin.g_phase, twin.g_mobile
    dense = sparse_to_dense(state)
    assert dense.xp is cp and type(dense.fraction) is cp.ndarray
    back = sparse_from_dense(dense, mobile, dev_net)
    for name in ("row_ptr", "bin", "fraction", "position_m"):
        assert to_host(getattr(back, name)).tobytes() == to_host(getattr(state, name)).tobytes(), name
    assert back.cell_of_slot is state.cell_of_slot and back.network_sha256 == state.network_sha256
    budget = dense_phase_bytes(host_net.shape, nc, nb)
    calls: list[str] = []
    monkeypatch.setattr(cuda_csr, "validate_phase_state", lambda *a, **k: calls.append("validate") or (_ for _ in ()).throw(
        AssertionError("validator before the cap")))
    with pytest.raises(DenseConversionRefused):
        sparse_from_dense(dense, mobile, dev_net, max_bytes=budget - 1)
    with pytest.raises(DenseConversionRefused):
        sparse_to_dense(state, max_bytes=budget - 1)
    assert calls == []
    monkeypatch.undo()
    with pytest.raises(TransportError):
        sparse_from_dense(dense, mobile * 0.0, dev_net)  # the dense validator still refuses in budget
    with pytest.raises(TransportError):
        sparse_from_dense(dense, to_host(mobile), dev_net)  # host array on the device path
    with pytest.raises(TransportError):
        sparse_from_dense(sparse_to_dense(twin.c_phase), to_host(mobile), dev_net)  # host dense phase on a CuPy network
    rebound = rebind_sparse_phase_state(state, dev_net)
    assert rebound.cell_of_slot is state.cell_of_slot and rebound.network_sha256 == state.network_sha256
    with pytest.raises(TransportError):
        rebind_sparse_phase_state(state, host_net)


# --- terminal storage compatibility (optional network fields of the upcoming C2 contract) --------------------------
@dataclasses.dataclass(frozen=True, eq=False)
class TerminalNetwork(TransportNetwork):
    """Test double of the C2 contract: a `TransportNetwork` with the optional `terminal_host` (host bool `(ny, nx)`)
    and `terminal_flat` (runtime bool `(n,)`) fields. Only this suite builds it; production networks are unchanged."""

    terminal_host: object = None
    terminal_flat: object = None


def terminal_network(base: TransportNetwork, pit_cells, *, with_fields=True, receiver_code=PIT_STORAGE,
                     runtime_self=True, flag_cells=None):
    """`base` with `pit_cells` turned into terminal storage: host receiver code `receiver_code`, runtime receiver
    the cell itself (`runtime_self`), terminal masks on `flag_cells` (default `pit_cells`), the pits removed
    from the out-face tables, and a distinct digest. Knobs produce the invalid encodings the refusal tests need."""
    xp = base.xp
    n = base.n_cells
    pits = np.asarray(pit_cells, dtype=np.int64)
    flags = pits if flag_cells is None else np.asarray(flag_cells, dtype=np.int64)
    receiver_host = np.array(base.receiver_host, copy=True)
    receiver_host.reshape(-1)[pits] = receiver_code
    receiver_index = np.array(to_host(base.receiver_index), copy=True).reshape(-1)
    if runtime_self:
        receiver_index[pits] = pits
    terminal_host = np.zeros(base.shape, dtype=np.bool_)
    terminal_host.reshape(-1)[flags] = True
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(TransportNetwork)}
    for cells_name, companions in (("cells_x", ("rows_x", "faces_x", "sign_x")), ("cells_y", ("faces_y", "cols_y", "sign_y"))):
        cells = to_host(getattr(base, cells_name))
        keep = ~np.isin(cells, pits)
        fields[cells_name] = to_device(np.ascontiguousarray(cells[keep]), xp)
        for name in companions:
            fields[name] = to_device(np.ascontiguousarray(to_host(getattr(base, name))[keep]), xp)
    fields["receiver_index"] = to_device(np.ascontiguousarray(receiver_index), xp)
    fields["receiver_host"] = freeze(receiver_host)
    fields["graph_input_sha256"] = base.graph_input_sha256 + ":terminal"
    extra = {}
    if with_fields:
        extra = {"terminal_host": freeze(terminal_host), "terminal_flat": to_device(terminal_host.reshape(-1).copy(), xp)}
    assert n == receiver_index.size
    # C2's integrated base now includes terminal fields. Replace their
    # values rather than passing the same keywords twice; absence remains
    # represented by the original None fields in the refusal fixtures.
    fields.update(extra)
    fields["n_terminal"] = int(flags.size) if with_fields else 0
    return TerminalNetwork(**fields)


@needs_numba
def test_host_evidence_accepts_declared_terminals_and_refuses_undeclared_codes():
    net = transport_network(make_graph(chain_full(4)))
    strict = host_network_evidence(net)
    assert strict.terminal is None and np.array_equal(strict.receiver_index, net.receiver_index)
    pit = terminal_network(net, [2])  # cell 2 is a terminal: cell 3 drains into it, nothing leaves it
    evidence = host_network_evidence(pit)
    assert evidence.terminal is not None and evidence.terminal.tolist() == [False, False, True, False]
    assert evidence.receiver_index.tolist() == [0, 0, 2, 2]
    maps = slot_maps(pit)
    assert maps.receiver_slot.tolist() == [0, 0, 2, 2]
    # undeclared negative receiver without the terminal field: strict refusal (unchanged behaviour)
    with pytest.raises(TransportError, match="negative receiver"):
        host_network_evidence(terminal_network(net, [2], with_fields=False))
    # terminal flag on a cell whose code is not PIT_STORAGE; PIT_STORAGE on an unflagged cell; terminal at an outlet
    with pytest.raises(TransportError, match="PIT_STORAGE"):
        host_network_evidence(terminal_network(net, [2], receiver_code=-7))
    with pytest.raises(TransportError, match="not flagged"):
        host_network_evidence(terminal_network(net, [2], flag_cells=[]))  # PIT_STORAGE code, empty terminal mask
    with pytest.raises(TransportError, match="outlet"):
        host_network_evidence(terminal_network(net, [0]))
    # runtime receiver not normalised to self: the host path refuses the disagreement
    with pytest.raises(TransportError, match="disagrees"):
        slot_maps(terminal_network(net, [2], runtime_self=False))


@needs_numba
def test_cpu_terminal_rule_requires_every_class_to_settle_and_settles_arrivals(monkeypatch):
    """C2 rule on the CPU sparse step: a settle mask that is True for every class at the terminal is the valid
    physics input (mobile load settles, nothing is sent); a mask missing ANY class is refused before any kernel
    work, never coerced."""
    net = transport_network(make_graph(chain_full(4)))
    pit_net = terminal_network(net, [2])
    nc = 3
    vel, rate, strict_mask = fields(net.shape, nc, v=0.2, rate=1.0)
    valid = strict_mask.copy()
    valid[2, 0, :] = True  # the C2 physics mask: every class settles at the terminal
    pickup = zeros(net.shape, nc)
    pickup[3, 0, :] = [1.0, 2.0, 3.0]  # above the terminal: crosses into it after 0.5 m
    pickup[2, 0, 1] = 0.5  # directly on the terminal: settles at once
    phase = empty_sparse_phase_state(pit_net, nc, 8)
    mobile = zeros(net.shape, nc)
    first = sparse_characteristic_step(pit_net, mobile, phase, pickup, vel, rate, valid, 1.0)
    tr = first.transport
    assert tr.settled_kg[2, 0, 1] == 0.5 and tr.export_request_kg.sum() == 0.0
    assert first.mobile_remaining_kg[2].sum() == 0.0 and first.phase_after.nnz == nc  # only cell 3 holds mass
    # refusal: any class not settling at the terminal, before any kernel work, inputs untouched
    copies = (pickup.copy(), np.array(phase.row_ptr, copy=True))
    monkeypatch.setattr(csr_kernels, "run_count_pass", lambda *a, **k: (_ for _ in ()).throw(AssertionError("kernel ran")))
    for c in range(nc):
        missing = valid.copy()
        missing[2, 0, c] = False
        with pytest.raises(TransportError, match="terminal-storage"):
            sparse_characteristic_step(pit_net, mobile, phase, pickup, vel, rate, missing, 1.0)
    with pytest.raises(TransportError, match="terminal-storage"):
        sparse_characteristic_step(pit_net, mobile, phase, pickup, vel, rate, strict_mask, 1.0)
    monkeypatch.undo()
    np.testing.assert_array_equal(pickup, copies[0])
    np.testing.assert_array_equal(phase.row_ptr, copies[1])
    # arrivals settle at the terminal and nothing ever leaves it; mass is conserved
    state, mob = first.phase_after, first.mobile_remaining_kg
    dep = tr.deposition_request_kg.copy()
    for _ in range(4):
        res = sparse_characteristic_step(pit_net, mob, state, zeros(net.shape, nc), vel, rate, valid, 1.0)
        assert res.transport.internal_transfer_out_kg[2].sum() == 0.0 and res.transport.export_request_kg.sum() == 0.0
        dep += res.transport.deposition_request_kg
        state, mob = res.phase_after, res.mobile_remaining_kg
    np.testing.assert_allclose(dep.sum() + mob.sum(), pickup.sum(), rtol=1e-12)
    assert mob.sum() == 0.0 and dep[2].sum() > 0.0 and state.nnz == 0
    # a strict network with the same geometry has no terminal rule: the strict mask is accepted there
    strict_phase = empty_sparse_phase_state(net, nc, 8)
    assert host_network_evidence(net).terminal is None
    sparse_characteristic_step(net, mobile, strict_phase, pickup, vel, rate, strict_mask, 1.0)


@needs_gpu
@needs_numba
def test_cuda_terminal_rule_matches_cpu_refuses_missing_classes_and_inconsistent_fields(monkeypatch):
    host_net, dev_net = networks(chain_full(4))
    host_pit = terminal_network(host_net, [2])
    dev_pit = terminal_network(dev_net, [2])
    binding = cuda_csr.bind_network(dev_pit)
    assert binding.has_terminal is True and to_host(binding.terminal).tolist() == [False, False, True, False]
    assert to_host(binding.receiver_cell).tolist() == [0, 0, 2, 2]
    nc = 3
    vel, rate, strict_mask = fields(host_net.shape, nc, v=0.2, rate=1.0)
    valid = strict_mask.copy()
    valid[2, 0, :] = True
    pickup = zeros(host_net.shape, nc)
    pickup[3, 0, :] = [1.0, 2.0, 3.0]
    pickup[2, 0, 1] = 0.5
    twin = Twin(host_pit, dev_pit, 8, nc)
    g, _c = twin.step(pickup, vel, rate, valid, 1.0)  # CUDA vs CPU within bounds, same valid mask
    assert float(to_host(g.transport.settled_kg)[2, 0, 1]) == 0.5
    repeat = sparse_characteristic_step(dev_pit, cp.zeros((*host_net.shape, nc)), empty_sparse_phase_state(dev_pit, nc, 8),
                                        dev(pickup), dev(vel), dev(rate), dev(valid), 1.0, implementation="cuda")
    assert to_host(repeat.phase_after.fraction).tobytes() == to_host(g.phase_after.fraction).tobytes()
    assert to_host(repeat.transport.settled_kg).tobytes() == to_host(g.transport.settled_kg).tobytes()
    # refusal in the input flag packet, before any launch, for a mask missing any class at the terminal
    monkeypatch.setattr(cuda_csr, "_function", lambda *a, **k: (_ for _ in ()).throw(AssertionError("launched")))
    for c in range(nc):
        missing = valid.copy()
        missing[2, 0, c] = False
        with pytest.raises(TransportError, match="terminal-storage"):
            sparse_characteristic_step(dev_pit, twin.g_mobile, twin.g_phase, dev(zeros(host_net.shape, nc)), dev(vel),
                                       dev(rate), dev(missing), 1.0, implementation="cuda")
    with pytest.raises(TransportError, match="terminal-storage"):
        sparse_characteristic_step(dev_pit, twin.g_mobile, twin.g_phase, dev(zeros(host_net.shape, nc)), dev(vel),
                                   dev(rate), dev(strict_mask), 1.0, implementation="cuda")
    monkeypatch.undo()
    for _ in range(4):  # arrivals settle at the terminal; nothing leaves it
        g, _c = twin.step(zeros(host_net.shape, nc), vel, rate, valid, 1.0)
        assert float(to_host(g.transport.internal_transfer_out_kg)[2].sum()) == 0.0
    assert twin.g_phase.nnz == 0
    # refusals: terminal_host without terminal_flat, flat disagreeing with host, runtime receiver not self
    half = dataclasses.replace(dev_pit, terminal_flat=None)
    with pytest.raises(TransportError, match="must both be present or both absent"):
        cuda_csr.bind_network(half)
    wrong_flat = dataclasses.replace(dev_pit, terminal_flat=cp.zeros(dev_pit.n_cells, dtype=bool))
    with pytest.raises(TransportError, match="terminal"):
        cuda_csr.bind_network(wrong_flat)
    with pytest.raises(TransportError):
        cuda_csr.bind_network(terminal_network(dev_net, [2], runtime_self=False))  # device receiver disagrees with host


@needs_gpu
def test_cuda_runtime_consistency_flags_catch_in_place_receiver_mutation(monkeypatch):
    rng = np.random.default_rng(5)
    _host_net, dev_net = networks(random_full(rng, 5, 4))
    ny, nx = dev_net.shape
    nc = 2
    phase = empty_sparse_phase_state(dev_net, nc, 8)
    inputs = [dev(a) for a in (rng.uniform(0, 1, (ny, nx, nc)), rng.uniform(0, 0.1, (ny, nx, nc)),
                               rng.uniform(0, 2, (ny, nx, nc)), np.zeros((ny, nx, nc), bool))]
    mobile = cp.zeros((ny, nx, nc), dtype=np.float64)
    ok = sparse_characteristic_step(dev_net, mobile, phase, *inputs, 1.0, implementation="cuda")
    binding = cuda_csr.bind_network(dev_net)
    assert "per-step device consistency flags" in binding.summary()["mutation_detection"]
    # in-place mutation of a runtime connectivity array: metadata fingerprints cannot see it ...
    original = int(to_host(dev_net.receiver_index[1]))
    dev_net.receiver_index[1] = 0 if original != 0 else 2  # CuPy arrays cannot be frozen; this is the hazard
    assert cuda_csr.bind_network(dev_net) is binding  # fingerprints unchanged, as documented
    # ... but the step's input flag packet does, before any kernel launch or publication
    monkeypatch.setattr(cuda_csr, "_function", lambda *a, **k: (_ for _ in ()).throw(AssertionError("launched")))
    with pytest.raises(TransportError, match="receiver_index"):
        sparse_characteristic_step(dev_net, ok.mobile_remaining_kg, ok.phase_after, *inputs, 1.0, implementation="cuda")
    with pytest.raises(TransportError, match="receiver_index"):
        validate_sparse_phase_state(ok.phase_after, ok.mobile_remaining_kg, dev_net)
    monkeypatch.undo()
    dev_net.receiver_index[1] = original  # restored: the same inputs run again and reproduce the earlier bytes
    again = sparse_characteristic_step(dev_net, mobile, phase, *inputs, 1.0, implementation="cuda")
    assert to_host(again.phase_after.fraction).tobytes() == to_host(ok.phase_after.fraction).tobytes()
    # a fresh network whose device receiver was mutated BEFORE binding is refused by the initial packet
    _h2, dev2 = networks(random_full(np.random.default_rng(6), 5, 4))
    dev2.receiver_index[1] = 0 if int(to_host(dev2.receiver_index[1])) != 0 else 2
    with pytest.raises(TransportError, match="binding refused"):
        cuda_csr.bind_network(dev2)
    dev2.outlet_flat[3] = ~dev2.outlet_flat[3]
    with pytest.raises(TransportError, match="binding refused"):
        cuda_csr.bind_network(dev2)


@needs_gpu
@pytest.mark.parametrize("n_bins", [1, 7, 8, 32, 100, 128])
def test_cuda_row_sum_kernel_is_qualified_against_the_running_numpy(n_bins):
    record = cuda_csr.row_sum_device_self_check(n_bins, force=True)
    assert record["byte_exact"] is True and record["n_bins"] == n_bins and record["rows"] == 96
    assert record["numpy"] == np.__version__ and record["cupy"] == cp.__version__
    attrs = cuda_csr.load_kernels(n_bins)
    assert set(attrs) == set(cuda_csr.KERNEL_NAMES)
    assert all(a["max_threads_per_block"] >= cuda_csr.BLOCK_THREADS for a in attrs.values())
