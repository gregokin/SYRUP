"""HOST execution of the RESIDENT floating CUDA source (NOT a GPU, NOT hardware evidence): g++ compiles `subsurface_cuda_floating.render_resident_source()` against stand-ins for the CUDA keywords and every kernel is launched by REAL
threads with a barrier (`__syncthreads` = `std::barrier`, `__shared__` = a function-static array of the running block), so padded lanes, inactive cells, the block trees and the finalize kernels execute their actual code. The CuPy below is a
numpy-backed PLUMBING DOUBLE for the Python launchers (allocation recording, argument conversion, counted transfers); it executes nothing by itself. The kernel parameter lists are PARSED from the rendered source (no hand-kept spec). Author-written."""
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
using std::isfinite; using std::fmin; using std::fmax; using std::fabs; using std::expm1;
inline unsigned long long atomicOr(unsigned long long* a, unsigned long long v) { return __atomic_fetch_or(a, v, __ATOMIC_SEQ_CST); }
inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long v) { return __atomic_fetch_add(a, v, __ATOMIC_SEQ_CST); }
inline unsigned long long atomicMin(unsigned long long* a, unsigned long long v) { unsigned long long o = __atomic_load_n(a, __ATOMIC_SEQ_CST); while (o > v && !__atomic_compare_exchange_n(a, &o, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {} return o; }
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


def have_toolchain() -> bool:
    if shutil.which("g++") is None:
        return False
    r = subprocess.run(["g++", "-std=c++20", "-x", "c++", "-fsyntax-only", "-"], input="#include <barrier>\nint main(){}", capture_output=True, text=True)
    return r.returncode == 0


def parse_spec(source: str) -> dict[str, list[str]]:
    out = {}
    for m in re.finditer(r'extern "C" __global__ void (\w+)\((.*?)\)\s*\{', source, re.S):
        params = [p.strip() for p in m.group(2).split(",")]
        out[m.group(1)] = [re.sub(r"\s*\w+$", "", p).strip() for p in params]
    return out


def _ctype(t: str):
    if t.endswith("*"):
        return ctypes.c_void_p
    return {"const long long": ctypes.c_longlong, "const int": ctypes.c_int, "const double": ctypes.c_double}[t]


def build(source: str):
    spec = parse_spec(source)
    wrappers = []
    for name, types in spec.items():
        params = ", ".join(f"{t} a{i}" for i, t in enumerate(types))
        call = ", ".join(f"({t})a{i}" if t.endswith("*") else f"a{i}" for i, t in enumerate(types))
        wrappers.append(f'extern "C" void host_{name}(int grid, int block, {params}) {{ launch(grid, block, [=] {{ {name}({call}); }}); }}')
    d = Path(tempfile.mkdtemp())
    (d / "k.cpp").write_text(STUB + source + "\n".join(wrappers))
    subprocess.run(["g++", "-std=c++20", "-O1", "-pthread", "-ffp-contract=off", "-fPIC", "-shared", "-o", str(d / "k.so"), str(d / "k.cpp")], check=True, capture_output=True, text=True)
    so = ctypes.CDLL(str(d / "k.so"))
    for name, types in spec.items():
        fn = getattr(so, f"host_{name}")
        fn.argtypes = [ctypes.c_int, ctypes.c_int] + [_ctype(t) for t in types]
        fn.restype = None
    return so, spec


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
    def __init__(self, so, name, spec, log):
        self.name, self.attributes, self._fn, self._types, self.log = name, {"max_threads_per_block": 1024}, getattr(so, f"host_{name}"), spec[name], log

    def __call__(self, grid, block, args):
        self.log.append((self.name, int(grid[0]), int(block[0])))
        conv = []
        assert len(args) == len(self._types), (self.name, len(args), len(self._types))
        for a, t in zip(args, self._types):
            if t.endswith("*"):
                assert isinstance(a, np.ndarray), (self.name, "pointer argument is not an array")
                conv.append(a.ctypes.data)
            elif t == "const double":
                conv.append(float(a))
            else:
                conv.append(int(a))
        self._fn(int(grid[0]), int(block[0]), *conv)


class FakeCupy:
    """PLUMBING DOUBLE of the CuPy surface the floating launchers use. Records every allocation (bytes by call kind) and every launch; `RawModule` maps the kernels to the host-compiled library."""

    ndarray = FakeArray
    float64, uint64, uint8, bool_ = np.float64, np.uint64, np.uint8, np.bool_

    def __init__(self, so, spec):
        self.so, self.spec = so, spec
        self.launches: list = []
        self.allocs: list[int] = []
        self.compiled: list = []

    def _rec(self, a):
        self.allocs.append(int(a.nbytes))
        return a

    def empty(self, shape, dtype=np.float64):
        return self._rec(fake(np.empty(shape, dtype=dtype)))

    def zeros(self, shape, dtype=np.float64):
        return self._rec(fake(np.zeros(shape, dtype=dtype)))

    def full(self, shape, value, dtype=np.float64):
        return self._rec(fake(np.full(shape, value, dtype=dtype)))

    def array(self, a, dtype=None, copy=True):
        return self._rec(fake(np.array(a, dtype=dtype, copy=True)))

    def RawModule(self, code, options, backend):
        self.compiled.append((hash(code), tuple(options), backend))
        outer = self
        return SimpleNamespace(get_function=lambda name: FakeFn(outer.so, name, outer.spec, outer.launches))
