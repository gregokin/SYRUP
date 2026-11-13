"""HOST execution of the rendered spatial CUDA source (NOT a GPU, NOT hardware evidence): g++ compiles the C text against stand-ins for the CUDA keywords; every kernel is launched by REAL threads with a REAL
barrier (`__syncthreads` = `std::barrier`, `__shared__` = a function-static array of the running block), so padded lanes, inactive cells and the block/finalize reductions execute their actual code. The fake
CuPy below is a numpy-backed PLUMBING DOUBLE for the Python launchers (allocation recording, argument conversion); it executes nothing by itself. Author-written."""
from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

STUB = r"""
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <barrier>
#include <vector>
#include <functional>
#define __device__
#define __global__
#define __forceinline__ inline
#define __restrict__
#define __shared__ static
struct dim3_ { unsigned x; };
thread_local dim3_ threadIdx, blockIdx, blockDim, gridDim;
static std::barrier<>* g_bar = nullptr;
inline void __syncthreads() { g_bar->arrive_and_wait(); }
using std::isfinite;
inline unsigned long long atomicOr(unsigned long long* a, unsigned long long v) { return __atomic_fetch_or(a, v, __ATOMIC_SEQ_CST); }
inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long v) { return __atomic_fetch_add(a, v, __ATOMIC_SEQ_CST); }
inline unsigned long long atomicMax(unsigned long long* a, unsigned long long v) { unsigned long long o = __atomic_load_n(a, __ATOMIC_SEQ_CST); while (o < v && !__atomic_compare_exchange_n(a, &o, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {} return o; }
inline long long __double_as_longlong(double x) { long long r; std::memcpy(&r, &x, 8); return r; }
static void launch(int grid, int block, const std::function<void()>& body)
{
    for (int b = 0; b < grid; ++b) {
        std::barrier<> bar(block);
        g_bar = &bar;
        std::vector<std::thread> th;
        for (int t = 0; t < block; ++t) th.emplace_back([&, b, t] { blockIdx.x = b; threadIdx.x = t; blockDim.x = block; gridDim.x = grid; body(); });
        for (auto& x : th) x.join();
    }
}
"""

C = {"P": "void*", "LL": "long long", "I": "int", "D": "double"}
SPEC = {
    "maple_syrup_subsurface_factors": ("const int", "const double*", "const double*", "const double*", "const double", "const int", "double*", "double*"),
    "maple_syrup_spatial_step": ("const long long", "const int") + ("const double*",) * 3 + ("const unsigned char*",) + ("const double*", "const long long", "const long long") * 5
    + ("const double*", "const double*", "const long long", "const long long", "const double*", "const double*", "const int") + ("const double",) * 4 + ("const int", "const double", "const int", "const double*", "const int", "const double", "const int")
    + ("double*",) * 7 + ("double*", "unsigned long long*"),
    "maple_syrup_spatial_step_finalize": ("const double*", "const long long", "unsigned long long*"),
    "maple_syrup_spatial_storage": ("const long long", "const int", "const double*", "const unsigned char*", "double*"),
    "maple_syrup_spatial_storage_finalize": ("const double*", "const long long", "unsigned long long*"),
    "maple_syrup_spatial_moisture": ("const long long", "const int", "const double*", "const double*", "const long long", "const long long", "double*"),
    "maple_syrup_spatial_uniform": ("const long long", "const int", "const double*", "const double*", "const long long", "const long long", "const double*", "const long long", "const long long", "double*", "unsigned long long*"),
}


def _ctype(c: str):
    if c.endswith("*"):
        return ctypes.c_void_p
    return {"const long long": ctypes.c_longlong, "const int": ctypes.c_int, "const double": ctypes.c_double}[c]


def wrappers() -> str:
    out = []
    for name, types in SPEC.items():
        params = ", ".join(f"{t} a{i}" for i, t in enumerate(types))
        call = ", ".join(f"({t})a{i}" if t.endswith("*") else f"a{i}" for i, t in enumerate(types))
        out.append(f'extern "C" void host_{name}(int grid, int block, {params}) {{ launch(grid, block, [=] {{ {name}({call}); }}); }}')
    return "\n".join(out)


def build(source: str):
    d = Path(tempfile.mkdtemp())
    (d / "k.cpp").write_text(STUB + source + wrappers())
    subprocess.run(["g++", "-std=c++20", "-O1", "-pthread", "-ffp-contract=off", "-fPIC", "-shared", "-o", str(d / "k.so"), str(d / "k.cpp")], check=True, capture_output=True, text=True)
    so = ctypes.CDLL(str(d / "k.so"))
    for name, types in SPEC.items():
        fn = getattr(so, f"host_{name}")
        fn.argtypes = [ctypes.c_int, ctypes.c_int] + [_ctype(t) for t in types]
        fn.restype = None
    return so


def have_toolchain() -> bool:
    if shutil.which("g++") is None:
        return False
    r = subprocess.run(["g++", "-std=c++20", "-x", "c++", "-fsyntax-only", "-"], input="#include <barrier>\nint main(){}", capture_output=True, text=True)
    return r.returncode == 0


class FakeDevice:
    def __init__(self, i):
        self.id = i


class FakeArray(np.ndarray):
    """numpy-backed 'device' array (plumbing double): `.device`, `.get()`."""

    device = FakeDevice(0)

    def get(self, out=None):
        return np.array(self)


def fake(a, device=0):
    out = np.ascontiguousarray(a).view(FakeArray)
    out.device = FakeDevice(device)  # type: ignore[attr-defined]
    return out


class FakeFn:
    def __init__(self, so, name):
        self.name, self.attributes, self._fn, self._types = name, {"max_threads_per_block": 1024}, getattr(so, f"host_{name}"), SPEC[name]
        self.launches = []

    def __call__(self, grid, block, args):
        self.launches.append((grid, block))
        conv = []
        for a, t in zip(args, self._types):
            if t.endswith("*"):
                conv.append(a.ctypes.data if isinstance(a, np.ndarray) else None)
            elif t == "const double":
                conv.append(float(a))
            else:
                conv.append(int(a))
        self._fn(int(grid[0]), int(block[0]), *conv)


class FakeCupy:
    """PLUMBING DOUBLE of the CuPy surface the spatial launchers use; `RawModule` maps the kernels to the host-compiled library. Records every allocation."""

    ndarray = FakeArray
    float64, uint64, uint8, bool_ = np.float64, np.uint64, np.uint8, np.bool_

    def __init__(self, so):
        self.so, self.allocations, self.modules = so, [], []
        outer = self

        class RawModule:
            def __init__(self, code, options, backend):
                outer.modules.append((code, tuple(options), backend))
                self.code = code

            def get_function(self, name):
                return FakeFn(outer.so, name)

        self.RawModule = RawModule

    def _rec(self, kind, shape, dtype):
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.allocations.append((kind, tuple(np.atleast_1d(shape)), np.dtype(dtype).name, n))

    def empty(self, shape, dtype=np.float64):
        self._rec("empty", shape, dtype)
        return fake(np.empty(shape, dtype=dtype))

    def zeros(self, shape, dtype=np.float64):
        self._rec("zeros", shape, dtype)
        return fake(np.zeros(shape, dtype=dtype))

    def array(self, a, dtype=None, copy=True, order="C"):
        out = np.array(a, dtype=dtype, copy=True, order=order)
        self._rec("array", out.shape, out.dtype)
        return fake(out, getattr(getattr(a, "device", None), "id", 0))

    def asarray(self, a):
        out = np.asarray(a)
        self._rec("asarray", out.shape, out.dtype)
        return fake(out)
