"""E5: the CUDA touched-column exchange stage (`floating_pond_cuda.after_exchange_cuda`) executed through the HOST-compiled D1/D2/D3 CUDA sources and the CuPy plumbing double: NOT hardware evidence. The same actual native
fixture is run through the CPU `FloatingPondVertical.after_exchange` and the CUDA one and compared (depth bit for bit, water within the existing contracts, receipts, returned water); the Dense NumPy-order reduction runs on the
host because the double's arrays are NumPy-backed. The real device Pond (Dense and Windowed) lives in test_gpu_definitions.py (unrun). Author-written."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_host_cuda
from hostd2 import fake

from maple_syrup import floating_pond as fp
from maple_syrup import floating_pond_cuda as fpc
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs
from maple_syrup.experimental_hydrology import HydraulicState

pytestmark = needs_host_cuda

_spec = importlib.util.spec_from_file_location("e3_builders_e5", Path(__file__).resolve().parents[1] / "phase2_native_floating_soil" / "builders.py")
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)
AVAIL = lambda: 1 << 40

import maple_syrup.compact_soil_profile as csp

_REAL_ITER = csp.iter_compact_soil_profile


def host_feed(monkeypatch, host_bed):
    """Feed the CUDA cold initialization with the REAL E2 compact batches computed on the host twin of the device bed (the device reader itself is exercised only on hardware)."""

    def feed(vc, al, g, gc, mres, layers, **k):
        k = dict(k, backend="numpy", memory_budget_bytes=1 << 30)
        return _REAL_ITER(host_bed.voxel_column, host_bed.active_layer, g, gc, mres, layers, **k)

    monkeypatch.setattr(csp, "iter_compact_soil_profile", feed)


def gpu_owner(dbed, host_bed, monkeypatch, sat=0.9):
    host_feed(monkeypatch, host_bed)
    owner = nfs.initialize_native_floating_soil(dbed, _b.config(backend="cuda"), initial_relative_saturation=sat, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    monkeypatch.setattr(csp, "iter_compact_soil_profile", _REAL_ITER)  # the feed was only the stand-in for the COLD device reader; the hot path never uses the compact reader on a device bed
    return owner


@pytest.fixture(autouse=True)
def _plumbing(cp):
    """The few CuPy functions the exchange stage uses beyond the shared double (plumbing only: each returns a NumPy-backed 'device' array)."""
    cp.flatnonzero = lambda a: cp._rec(fake(np.flatnonzero(np.asarray(a))))
    cp.concatenate = lambda parts: cp._rec(fake(np.concatenate([np.asarray(p) for p in parts])))
    yield


def device_bed(bed, dev=0):
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    return nfs.NativeBed(VoxelColumnState(mass_kg=fake(bed.voxel_column.mass_kg, dev)), ActiveLayerState(mass_kg=fake(bed.active_layer.mass_kg, dev)), bed.geometry, bed.grain_classes,
                         bed.mass_resolution_kg, bed.active_mask)


def bed_state(nb):
    return SimpleNamespace(voxel_column=nb.voxel_column, active_layer=nb.active_layer)


def context(nb):
    return SimpleNamespace(geometry=nb.geometry, grain_classes=nb.grain_classes, mass_resolution_kg=nb.mass_resolution_kg)


def hydraulic(v, xp_zeros):
    ny, nx = v.shape
    soil = v._derived(v._initial.state)
    v._register(soil, v._initial)
    return HydraulicState(0.0, xp_zeros((ny, nx)), soil, xp_zeros((ny, nx + 1)), xp_zeros((ny + 1, nx)))


def erode(nb, cells, factor):
    """A native-valid bed edit: the top `factor` (1 or 2) filled voxels of every cell of the flat `cells` are removed (a real depth change; every other cell untouched, the active layer kept)."""
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState

    vox, act = np.array(nb.voxel_column.mass_kg, copy=True), np.array(nb.active_layer.mass_kg, copy=True)
    nx = vox.shape[1]
    for c in cells:
        iy, ix = divmod(c, nx)
        filled = np.flatnonzero(vox[iy, ix].sum(axis=-1) > 0.0)
        for v_ in filled[::-1][:int(factor)]:
            vox[iy, ix, v_] = 0.0
    return nb.__class__(VoxelColumnState(mass_kg=vox), ActiveLayerState(mass_kg=act), nb.geometry, nb.grain_classes, nb.mass_resolution_kg, nb.active_mask)


@pytest.mark.parametrize("factor", [1, 2])
def test_cuda_exchange_stage_matches_the_cpu_stage_on_the_same_native_bed(cp, monkeypatch, factor):
    bed = _b.stratified(shallow=((0, 2),))
    ny, nx = bed.active_layer.mass_kg.shape[:2]
    owner_c = nfs.initialize_native_floating_soil(bed, _b.config(), initial_relative_saturation=0.9, available_bytes_fn=AVAIL)
    dbed = device_bed(bed)
    owner_g = gpu_owner(dbed, bed, monkeypatch)
    vc = fp.FloatingPondVertical(owner_c, available_bytes_fn=AVAIL)
    vg = fp.FloatingPondVertical(owner_g, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    vc.bind_shape((ny, nx))
    vg.bind_shape((ny, nx))
    hc = hydraulic(vc, np.zeros)
    hg = hydraulic(vg, lambda s: fake(np.zeros(s)))
    cells = [0, 2]
    touched = np.zeros(ny * nx, bool)
    touched[cells] = True
    touched = touched.reshape(ny, nx)
    active = np.ones((ny, nx), bool)
    cb = erode(bed, cells, factor)
    gb = device_bed(cb)
    ctx_c, ctx_g = context(bed), context(dbed)
    cp.count.__init__()
    want, rec_c = vc.after_exchange(bed_state(cb), ctx_c, hc, touched, active)
    got, rec_g = vg.after_exchange(bed_state(gb), ctx_g, hg, fake(touched), fake(active))
    pc, pg = vc.current_payload(want.soil_water_m), vg.current_payload(got.soil_water_m)
    assert np.asarray(pg.depth_m).tobytes() == pc.depth_m.tobytes()  # the device canonical depth is the cold native depth bit for bit
    for k in ("n_touched", "n_depth_changed", "n_reconciled"):
        assert rec_g[k] == rec_c[k], k
    assert rec_c["n_depth_changed"] == len(cells)
    np.testing.assert_allclose(np.asarray(pg.state.water_m), np.asarray(pc.state.water_m), rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(got.depth_m), np.asarray(want.depth_m), rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(got.soil_water_m), np.asarray(want.soil_water_m), rtol=2e-12, atol=1e-14)
    for k in ("surface_return_m3", "bottom_export_m3"):
        assert rec_g[k] == pytest.approx(rec_c[k], rel=2e-12, abs=1e-14)
    assert pg.history.wetting_m is not None and pg.history.operator_sha256 == pg.matrix_op.sha256
    assert pg.column.identity_kind == "geometry_view_token" and pg.column.parent_sha256 == owner_g.column.sha256  # root-bound token identity, no chain
    # the accepted payload of the input state is untouched (pure)
    assert np.asarray(vg.current_payload(hg.soil_water_m).state.water_m).tobytes() == np.asarray(owner_g.state.water_m).tobytes()


def test_cuda_net_zero_and_d_equivalent_exchanges_drain_nothing(cp, monkeypatch):
    bed = _b.stratified(shallow=((0, 2),))
    ny, nx = bed.active_layer.mass_kg.shape[:2]
    dbed = device_bed(bed)
    owner = gpu_owner(dbed, bed, monkeypatch)
    v = fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    v.bind_shape((ny, nx))
    h = hydraulic(v, lambda s: fake(np.zeros(s)))
    touched = np.zeros((ny, nx), bool)
    touched[0, 1] = True
    same, rec = v.after_exchange(bed_state(dbed), context(dbed), h, fake(touched), fake(np.ones((ny, nx), bool)))
    assert rec["n_touched"] == 1 and rec["n_depth_changed"] == 0 and same is h  # net-zero turnover touches the column but changes nothing
    none, rec0 = v.after_exchange(bed_state(dbed), context(dbed), h, fake(np.zeros((ny, nx), bool)), fake(np.ones((ny, nx), bool)))
    assert rec0["n_touched"] == 0 and none is h


def test_cuda_vertical_admission_and_cpu_argument_refusals(cp, monkeypatch):
    bed = _b.stratified(shallow=((0, 2),))
    dbed = device_bed(bed)
    owner = gpu_owner(dbed, bed, monkeypatch)
    calls = []
    n_alloc = len(cp.allocs)
    with pytest.raises((nfs.NativeFloatingMemoryError, fsl.FloatingMemoryError)):
        fp.FloatingPondVertical(owner, available_bytes_fn=lambda: (calls.append(1), 0)[1], host_available_bytes_fn=AVAIL)
    assert calls and len(cp.allocs) == n_alloc  # the device reader refused before any owned allocation
    with pytest.raises((nfs.NativeFloatingMemoryError, fsl.FloatingMemoryError)):
        fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, host_available_bytes_fn=lambda: 0)  # the HOST pool is separate
    cp.dev["id"] = 1
    reads = []
    with pytest.raises(Exception, match="current device"):
        fp.FloatingPondVertical(owner, available_bytes_fn=lambda: (reads.append(1), 1 << 40)[1], host_available_bytes_fn=AVAIL)
    cp.dev["id"] = 0
    assert not reads  # the device gate preceded every reader
    good = fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    need_dev = good.construction_bytes
    assert fp.FloatingPondVertical(owner, available_bytes_fn=lambda: need_dev, host_available_bytes_fn=AVAIL, reserve_bytes=0).construction_bytes == need_dev
    with pytest.raises((nfs.NativeFloatingMemoryError, fsl.FloatingMemoryError)):
        fp.FloatingPondVertical(owner, available_bytes_fn=lambda: need_dev - 1, host_available_bytes_fn=AVAIL)
    # CPU owner: the cuda-only host arguments are refused
    cpu_owner = nfs.initialize_native_floating_soil(bed, _b.config(), initial_relative_saturation=0.9, available_bytes_fn=AVAIL)
    with pytest.raises(fp.FloatingPondError, match="cuda backend"):
        fp.FloatingPondVertical(cpu_owner, available_bytes_fn=AVAIL, host_reserve_bytes=1)


def test_dense_preflight_refuses_an_unqualified_extent(cp, monkeypatch):
    bed = _b.stratified(shallow=((0, 2),))
    dbed = device_bed(bed)
    owner = gpu_owner(dbed, bed, monkeypatch)
    v = fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    from maple.core.types.voxel import VoxelColumnState

    fat = SimpleNamespace(voxel_column=VoxelColumnState(mass_kg=np.zeros((1, 1, 40000, 1))), active_layer=dbed.active_layer)
    with pytest.raises(fp.FloatingPondError, match="32768"):
        fpc.preflight_dense(v, fat)


def test_r1a_reconciliation_cuda_vertical_has_the_provisional_field_and_the_resume_record_refuses_cuda(cp, monkeypatch):
    """R1a was written against the CPU constructor only; the CUDA constructor returns early, so `_provisional` must exist on it too (else `_payload` raised AttributeError), and the CPU in-memory record must refuse a CUDA vertical."""
    from maple_syrup import floating_resume as fr

    bed = _b.stratified(shallow=((0, 2),))
    owner = gpu_owner(device_bed(bed), bed, monkeypatch)
    v = fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    assert v._provisional is None
    with pytest.raises(fp.FloatingPondError, match="not issued"):
        v._payload(fake(np.zeros((1, 3))))  # a typed refusal, never an AttributeError
    from maple_syrup.pond_event import PondEvent

    ev = object.__new__(PondEvent)
    ev.vertical = v
    with pytest.raises(fr.FloatingResumeError, match="CUDA record unsupported"):
        fr._check_event(ev)
