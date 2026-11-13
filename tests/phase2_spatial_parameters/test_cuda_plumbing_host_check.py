"""CUDA PLUMBING and kernel-source check WITHOUT a GPU (no CuPy, no device, nothing here is a hardware claim).

* The CUDA C text of `spatial_soil_layers._cuda_source()` is compiled by `g++` against small stand-ins for the CUDA keywords/atomics/thread indices (`-ffp-contract=off`, the host analogue of
  `--fmad=false`). The plain kernel body is then run for every cell (blockIdx/threadIdx emulated; the kernel uses no shared memory and no barrier), so the whole preparation kernel -- strided layouts,
  flags, atomics, derived values -- is compared with the NumPy reference bit for bit.
* A FAKE `cupy` object (clearly a plumbing double: numpy arrays with a `.get()` method; every CuPy attribute other than `zeros/float64/uint64` raises) lets the REAL `_derive_cuda` argument assembly run: it
  checks the strides, the auxiliary array, the allocation list (only the 4-word diagnostic and the 4-element auxiliary array), the one 32-byte diagnostic read, and the device-style buffered identity route.
It does NOT execute NVRTC or any launch: Root runs the device tests. Skipped when no C++ compiler is available. Author-written.
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from conftest import make_spec, params

from maple_syrup import spatial_soil_layers as sp
from maple_syrup import subsurface as ss

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="no C++ compiler: the CUDA source host check is not run (no claim)")

STUB = r'''
#include <cmath>
#include <cstdint>
#define __device__
#define __global__
#define __forceinline__ inline
#define __restrict__
struct dim3_ { unsigned x; };
static dim3_ threadIdx, blockIdx, blockDim;
inline unsigned long long atomicOr(unsigned long long* a, unsigned long long v){ auto o=*a; *a|=v; return o; }
inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long v){ auto o=*a; *a+=v; return o; }
inline unsigned long long atomicMin(unsigned long long* a, unsigned long long v){ auto o=*a; if(v<*a)*a=v; return o; }
using std::isfinite;
'''
WRAP = r'''
extern "C" void host_run(const double* eo, long long eL, long long eC, const double* tr, long long rL, long long rC, const double* tf, long long fL, long long fC,
                         const double* ts, long long sL, long long sC, const double* ic, long long iL, long long iC,
                         double* dz, long long dL, long long dC, double* pv, long long pL, long long pC, double* gd, double* gu, long long gL, long long gC,
                         const unsigned char* mask, unsigned long long* diag, double coef, double dcoef, double stress, long long L, long long C, int donor, int block)
{
    for (long long c = 0; c < C; ++c) {
        blockDim.x = block; blockIdx.x = (unsigned)(c / block); threadIdx.x = (unsigned)(c % block);
        maple_syrup_spatial_prepare(eo, eL, eC, tr, rL, rC, tf, fL, fC, ts, sL, sC, ic, iL, iC, dz, dL, dC, pv, pL, pC, gd, gu, gL, gC, mask, diag, coef, dcoef, stress, L, C, donor);
    }
}
'''


@pytest.fixture(scope="module")
def lib():
    d = Path(tempfile.mkdtemp())
    (d / "k.cpp").write_text(STUB + sp._cuda_source() + WRAP)
    subprocess.run(["g++", "-std=c++17", "-O1", "-ffp-contract=off", "-fPIC", "-shared", "-o", str(d / "k.so"), str(d / "k.cpp")], check=True)
    so = ctypes.CDLL(str(d / "k.so"))
    P, LL, D, I = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_double, ctypes.c_int
    so.host_run.argtypes = [P, LL, LL] * 5 + [P, LL, LL, P, LL, LL, P, P, LL, LL, P, P, D, D, D, LL, LL, I, I]  # explicit: Python ints would be passed as 32-bit
    so.host_run.restype = None
    return so


class FakeArray(np.ndarray):
    def get(self, out=None):
        if out is None:
            return np.array(self)
        out[...] = np.asarray(self).reshape(out.shape)
        return out


class FakeCupy:
    """PLUMBING DOUBLE (not CuPy): numpy-backed arrays; only zeros/float64/uint64 exist; allocations are recorded."""

    float64, uint64, uint8, bool_ = np.float64, np.uint64, np.uint8, np.bool_

    def __init__(self):
        self.allocations = []

    def zeros(self, shape, dtype):
        a = np.zeros(shape, dtype=dtype).view(FakeArray)
        self.allocations.append((tuple(np.atleast_1d(shape)), np.dtype(dtype).name, a.nbytes))
        return a


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


@pytest.fixture
def fake(monkeypatch, lib):
    class Stub:
        @staticmethod
        def current_device_id(cp):
            return 0

    monkeypatch.setattr(sp, "_cuda_module", lambda: Stub)
    launches = []

    def kernel(grid, block, args):
        launches.append((grid, block))
        # args: eo,eL,eC, tr,rL,rC, tf,fL,fC, ts,sL,sC, ic,iL,iC, dz,dL,dC, pv,pL,pC, gd,gu,gL,gC, mask, diag, coef, dcoef, stress, L, C, donor
        a = list(args)
        lib.host_run(ptr(a[0]), int(a[1]), int(a[2]), ptr(a[3]), int(a[4]), int(a[5]), ptr(a[6]), int(a[7]), int(a[8]), ptr(a[9]), int(a[10]), int(a[11]),
                     ptr(a[12]), int(a[13]), int(a[14]), ptr(a[15]), int(a[16]), int(a[17]), ptr(a[18]), int(a[19]), int(a[20]), ptr(a[21]), ptr(a[22]), int(a[23]), int(a[24]),
                     ctypes.c_void_p(np.ascontiguousarray(a[25]).ctypes.data), ptr(a[26]), ctypes.c_double(float(a[27])), ctypes.c_double(float(a[28])), ctypes.c_double(float(a[29])),
                     int(a[30]), int(a[31]), int(a[32]), int(sp.CUDA_BLOCK))

    monkeypatch.setattr(sp, "_load_kernel", lambda cp, dev: kernel)
    return FakeCupy(), launches


def numpy_reference(spec, coef):
    out = {}
    from maple_syrup.spatial_soil_layers import _retained_layouts, LAYOUT_CELL

    for name, (kind, rows) in _retained_layouts(spec.n_layers, spec.layouts(), spec.diffusion_mode).items():
        out[name] = np.zeros((rows, spec.n_cells) if kind == LAYOUT_CELL else (rows,))
    info = sp._derive_numpy(spec, out, coef, spec.n_cells)
    return out, info


def device_style(fake_cp, spec, mask):
    """Allocate the retained maps like `prepare_spatial_column` does (FakeArray 'device' arrays), then run the REAL `_derive_cuda`."""
    out = {}
    for key in ("theta_residual", "theta_field_capacity", "theta_saturation"):
        i = spec.inputs[key]
        out[key] = (np.full(1, i.scalar) if i.kind == "scalar" else np.array(i.array)).view(FakeArray)
    for name, (kind, rows) in sp._retained_layouts(spec.n_layers, spec.layouts(), spec.diffusion_mode).items():
        if name not in out:
            out[name] = np.zeros((rows, spec.n_cells) if kind == "cell" else (rows,)).view(FakeArray)
    from maple_syrup.conservation import volume_roundoff_bound_m3

    coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    info, n_active = sp._derive_cuda(fake_cp, spec, np.ascontiguousarray(mask), out, coef)
    return out, info, n_active, coef


LAYOUT_CASES = {
    "all_cell": dict(), "shared_edges": dict(shared_edges=True), "scalar_thetas": dict(scalar_thetas=True), "scalar_iface": dict(scalar_iface=True),
    "shared_iface": dict(shared_iface=True), "all_shared": dict(shared_edges=True, scalar_thetas=True, shared_iface=True)}


def build(L, N, mode, case, seed=11):
    p = params(np.random.default_rng(seed), L, N, mode=mode)
    kw = LAYOUT_CASES[case]
    edges = p["edges"][:, 0].copy() if kw.get("shared_edges") else p["edges"]
    res, fc, sat = (0.03, 0.2, 0.45) if kw.get("scalar_thetas") else (p["res"], p["fc"], p["sat"])
    if kw.get("scalar_iface"):
        iface = 3e-7 if L > 1 else 3e-7
    elif kw.get("shared_iface"):
        iface = p["iface"][:, 0].copy() if L > 1 else np.zeros(0)
    else:
        iface = p["iface"]
    return make_spec({"edges": edges, "res": res, "fc": fc, "sat": sat, "iface": iface}, mode=mode, N=N, iface=iface, edges=edges, res=res, fc=fc, sat=sat)


@pytest.mark.parametrize("case", list(LAYOUT_CASES))
@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
@pytest.mark.parametrize("L", [1, 2, 3, 5])
def test_kernel_source_and_argument_assembly_equal_the_numpy_reference_bitwise(fake, case, mode, L):
    N = 300  # more than one emulated block of 128 threads
    spec = build(L, N, mode, case)
    cp, launches = fake
    mask = np.zeros(N, dtype=bool)
    mask[::3] = True
    got, info, n_active, coef = device_style(cp, spec, mask)
    want, want_info = numpy_reference(spec, coef)
    assert info == (0, 0, None, 0) == (want_info[0], want_info[1], want_info[2], want_info[3]) and n_active == int(mask.sum())
    for name in want:
        if name.startswith("theta_"):
            continue  # the theta copies are made by prepare_spatial_column (xp.array copy), not by the derivation
        np.testing.assert_array_equal(np.asarray(got[name]).view(np.uint64), want[name].view(np.uint64), err_msg=f"{name} {case} {mode} L={L}")
    assert launches == [(((N + sp.CUDA_BLOCK - 1) // sp.CUDA_BLOCK,), (sp.CUDA_BLOCK,))]


def test_only_the_modelled_small_device_arrays_are_allocated_and_one_packet_is_read(fake):
    spec = build(3, 200, "constant", "all_cell")
    cp, _ = fake
    device_style(cp, spec, np.ones(200, dtype=bool))
    assert sorted(cp.allocations) == sorted([((sp.DIAG_WORDS,), "uint64", sp.PACKET_BYTES), ((4,), "float64", sp.AUX_BYTES)])


@pytest.mark.parametrize("mode", ["constant", "tietjen_donor_depth"])
def test_flags_counts_and_first_cell_match_the_numpy_classification(fake, mode):
    L, N = 3, 400
    p = params(np.random.default_rng(3), L, N, mode=mode)
    p["edges"][2, 350] = p["edges"][1, 350]  # edges flag, late
    p["res"][0, 130] = -1.0  # relation flag
    p["fc"][1, 5] = np.nan  # non-finite theta, first
    p["iface"][0, 200] = -1e-9  # interface flag
    p["edges"][:, 77] = np.array([0.0, 1e-320, 2e-320, 3e-320])  # derived flag (input checks pass)
    spec = make_spec(p, mode=mode)
    cp, _ = fake
    got, info, _, coef = device_style(cp, spec, np.ones(N, dtype=bool))
    _, want = numpy_reference(spec, coef)
    assert info == want and info[2] == 5 and info[1] == 5 and info[0] == (sp.FLAG_EDGES | sp.FLAG_THETA | sp.FLAG_REL | sp.FLAG_IFACE | sp.FLAG_DERIVED)


def test_stress_flag_and_hierarchy_match(fake):
    L, N = 2, 20
    p = params(np.random.default_rng(4), L, N)
    p["res"][0, 3] = 0.55  # stress 0.5 is NOT above this layer-0 residual
    p["fc"][0, 3], p["sat"][0, 3] = 0.6, 0.8
    p["res"][1, 7] = -1.0
    p["edges"][:, 7] = np.array([0.0, 1e-320, 2e-320])  # input flag only (the derived check is not evaluated for a cell that failed an input check)
    spec = make_spec(p)
    cp, _ = fake
    _, info, _, coef = device_style(cp, spec, np.ones(N, dtype=bool))
    _, want = numpy_reference(spec, coef)
    assert info == want and info[0] == (sp.FLAG_STRESS | sp.FLAG_REL) and info[2] == 3


@pytest.mark.parametrize("hash_chunk", [8, 24, 1 << 10, 1 << 20])
@pytest.mark.parametrize("case", ["all_cell", "all_shared", "scalar_iface"])
def test_the_buffered_device_identity_route_equals_the_numpy_route(case, hash_chunk):
    L, N = 3, 301
    spec = build(L, N, "tietjen_donor_depth", case)
    mask = (np.arange(N) % 5 != 0)
    ref = sp._identity(spec, mask, None)
    buf = np.empty(max(1, hash_chunk // 8))
    fa = type("S", (), {})()
    # emulate the device arrays with FakeArray (their `.get(out=...)` copies exactly like cupy's) for every array input and the mask
    inputs = {k: (sp._Input(v.name, v.kind, v.array.view(FakeArray), v.scalar, v.nbytes, v.rows) if v.array is not None else v) for k, v in spec.inputs.items()}
    fspec = sp.SpatialLayerSpec(spec.n_layers, spec.n_cells, spec.namespace, inputs, spec.diffusion_mode, spec.bottom_drainage, spec.gravity_drainage, spec.diffusion_coefficient,
                                spec.theta_evaporation_stress, spec.evaporation_factor)
    got = sp._identity(fspec, mask.view(FakeArray), buf)
    assert got == ref
