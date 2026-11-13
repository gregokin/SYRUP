"""Host-compiler execution of the CUDA kernel SOURCE (no GPU, no CuPy, no device, no NVRTC) plus a plumbing double of CuPy that runs the compiled host function when `cu.run` 'launches'. This checks the kernel TEXT
(arithmetic, flags, packed diagnostics, strides/layouts, thread/block indexing as emulated) and the Python argument assembly. It is NOT hardware evidence: atomics, real launches, the device libm and CUDA compile flags are
not exercised (`-ffp-contract=off` is only the host analogue of `--fmad=false`). Author-written."""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from maple_syrup import soil_texture_properties as tp
from maple_syrup import soil_texture_properties_cuda as cu

STUB = r'''
#include <cmath>
#include <cstdint>
#define __device__
#define __global__
#define __forceinline__ inline
struct dim3_ { unsigned x; };
static dim3_ threadIdx, blockIdx, blockDim;
inline unsigned long long atomicOr(unsigned long long* a, unsigned long long v){ auto o=*a; *a|=v; return o; }
inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long v){ auto o=*a; *a+=v; return o; }
inline unsigned long long atomicMin(unsigned long long* a, unsigned long long v){ auto o=*a; if(v<*a)*a=v; return o; }
using std::isfinite; using std::isinf; using std::isnan; using std::fabs; using std::log; using std::pow;
'''
WRAP = r'''
extern "C" void host_run(const double* const* a, const long long* st, double* ofc, double* osat, double* oks, double* owp, double* oair, unsigned char* oprov, long long osl, long long osc,
                         unsigned long long* diag, double closure, long long L, long long cells, int want_diag, int block)
{
    const long long grid = (cells + block - 1) / block;
    blockDim.x = (unsigned)block;
    for (long long b = 0; b < grid; ++b) {
        blockIdx.x = (unsigned)b;
        for (int t = 0; t < block; ++t) {
            threadIdx.x = (unsigned)t;
            maple_syrup_texture_properties(a[0], st[0], st[1], a[1], st[2], st[3], a[2], st[4], st[5], a[3], st[6], st[7], a[4], st[8], st[9], a[5], st[10], st[11], a[6], st[12], st[13],
                                           a[7], st[14], st[15], a[8], st[16], st[17], a[9], st[18], st[19], ofc, osat, oks, owp, oair, oprov, osl, osc, diag, closure, L, cells, want_diag);
        }
    }
}
'''


def have_toolchain() -> bool:
    return shutil.which("g++") is not None


def build(source: str):
    d = Path(tempfile.mkdtemp())
    (d / "k.cpp").write_text(STUB + source + WRAP)
    subprocess.run(["g++", "-std=c++17", "-O0", "-ffp-contract=off", "-shared", "-fPIC", "-o", str(d / "k.so"), str(d / "k.cpp")], check=True, capture_output=True)
    lib = ctypes.CDLL(str(d / "k.so"))
    lib.host_run.restype = None
    return lib


class FakeArr(np.ndarray):
    """A 'device' array double: a NumPy subclass with the CuPy surface the provider uses (`get`, `device.id`)."""

    def get(self, out=None):
        if out is None:
            return np.array(self, copy=True).view(np.ndarray)
        out[...] = np.asarray(self)
        return out

    @property
    def device(self):
        return SimpleNamespace(id=0)


def fake(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(FakeArr)


class FakeFn:
    def __init__(self, lib, log):
        self.lib, self.log = lib, log

    def __call__(self, grid, block, args):
        self.log.append((grid, block, len(args)))
        arrs = [np.asarray(args[3 * j]) for j in range(10)]
        for a in arrs:
            assert a.dtype == np.float64
        st = np.array([int(args[3 * j + k]) for j in range(10) for k in (1, 2)], dtype=np.int64)
        ptrs = (ctypes.c_void_p * 10)(*[a.ctypes.data for a in arrs])
        outs = [np.asarray(x) for x in args[30:36]]
        osl, osc, diag, closure, L, cells, want = args[36:43]
        d = np.asarray(diag)
        self.lib.host_run(ptrs, st.ctypes.data_as(ctypes.c_void_p), *[o.ctypes.data_as(ctypes.c_void_p) for o in outs], ctypes.c_longlong(int(osl)), ctypes.c_longlong(int(osc)),
                          d.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(float(closure)), ctypes.c_longlong(int(L)), ctypes.c_longlong(int(cells)), ctypes.c_int(int(want)), ctypes.c_int(int(block[0])))


class FakeCupy:
    """Plumbing double of the CuPy module: records the RawModule request and runs the host-compiled kernel for each launch."""

    ndarray = FakeArr
    float64, uint64, uint8 = np.float64, np.uint64, np.uint8

    def __init__(self, lib):
        self.lib, self.launches, self.modules = lib, [], []

    def full(self, n, value, dtype):
        return np.full(n, value, dtype=dtype).view(FakeArr)

    def zeros(self, shape, dtype):
        return np.zeros(shape, dtype=dtype).view(FakeArr)

    def RawModule(self, code, options, backend):
        self.modules.append((code, tuple(options), backend))
        outer = self
        return SimpleNamespace(get_function=lambda name: (_ for _ in ()).throw(AssertionError(name)) if name != cu.KERNEL_NAME else FakeFn(outer.lib, outer.launches))
