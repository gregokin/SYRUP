"""HOST execution of the rendered spatial CUDA PREFERENTIAL source (NOT a GPU, NOT hardware evidence). g++ compiles the C text against stand-ins for the CUDA keywords (`-ffp-contract=off`, the host analogue of
`--fmad=false`); every kernel is launched by REAL threads with a REAL barrier (`__syncthreads` = `std::barrier`, `__shared__` = a function-static array of the running block, `atomicOr/Add/Min/Max` on unsigned
long long = `__atomic_*`/compare-exchange). The fake CuPy is a numpy-backed PLUMBING DOUBLE for the Python launchers: it records every allocation, host-to-device copy, kernel launch and read, executes nothing by
itself, and its library reductions (`sum`, `min`, `max`, `isfinite`, `isnan`) RAISE so that any use of them by the cold derivation is detected. Author-written."""
from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2_spatial_cuda"))
import hostgpu  # noqa: E402

EXTRA = r"""
inline unsigned long long atomicMin(unsigned long long* a, unsigned long long v) { unsigned long long o = __atomic_load_n(a, __ATOMIC_SEQ_CST); while (o > v && !__atomic_compare_exchange_n(a, &o, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {} return o; }
"""
MAPS2 = ("const double*", "const long long", "const long long") * 2
STEP_TYPES = (("const long long", "const int") + ("const double*",) * 2 + ("const unsigned char*", "const double") + ("const double*", "const double*", "const long long", "const double*") + MAPS2 + ("const double", "const double")
              + ("double*",) * 4 + ("double*", "unsigned long long*"))
SPEC = {
    "maple_syrup_spatial_pref_step": STEP_TYPES,
    "maple_syrup_spatial_pref_finalize": ("const double*", "const long long", "unsigned long long*"),
    "maple_syrup_spatial_pref_reduce": ("const long long", "const double*", "unsigned long long*"),
    "maple_syrup_spatial_pref_validate": ("const long long", "const double*", "const double", "const double", "unsigned long long*"),
}


def wrappers() -> str:
    out = []
    for name, types in SPEC.items():
        params = ", ".join(f"{t} a{i}" for i, t in enumerate(types))
        call = ", ".join(f"({t})a{i}" if t.endswith("*") else f"a{i}" for i, t in enumerate(types))
        out.append(f'extern "C" void host_{name}(int grid, int block, {params}) {{ launch(grid, block, [=] {{ {name}({call}); }}); }}')
    return "\n".join(out)


def build(source: str, flags=("-ffp-contract=off",)):
    d = Path(tempfile.mkdtemp())
    (d / "k.cpp").write_text(hostgpu.STUB + EXTRA + source + wrappers())
    subprocess.run(["g++", "-std=c++20", "-O1", "-pthread", *flags, "-fPIC", "-shared", "-o", str(d / "k.so"), str(d / "k.cpp")], check=True, capture_output=True, text=True)
    so = ctypes.CDLL(str(d / "k.so"))
    for name, types in SPEC.items():
        fn = getattr(so, f"host_{name}")
        fn.argtypes = [ctypes.c_int, ctypes.c_int] + [hostgpu._ctype(t) for t in types]
        fn.restype = None
    return so


class FakeDevice:
    def __init__(self, i):
        self.id = i


class FakeArray(np.ndarray):
    """numpy-backed 'device' array (plumbing double): `.device`, `.get(out=)` (honours `out`, as CuPy does), `.reshape` keeps the type."""

    device = FakeDevice(0)

    def __array_finalize__(self, obj):
        if obj is not None and hasattr(obj, "device"):  # views and results keep the device of their base, as CuPy views do
            self.device = obj.device

    def get(self, out=None):
        if out is None:
            return np.array(self)
        out[...] = np.asarray(self)
        return out


def fake(a, device=0):
    out = np.ascontiguousarray(a).view(FakeArray)
    out.device = FakeDevice(device)  # type: ignore[attr-defined]
    return out


class FakeFn:
    def __init__(self, cp, so, name):
        self.cp, self.name, self.attributes, self._fn, self._types = cp, name, {"max_threads_per_block": 1024}, getattr(so, f"host_{name}"), SPEC[name]

    def __call__(self, grid, block, args):
        self.cp.launches.append((self.name, tuple(grid), tuple(block)))
        conv = []
        for a, t in zip(args, self._types, strict=True):
            if t.endswith("*"):
                conv.append(a.ctypes.data if isinstance(a, np.ndarray) else None)
            elif t == "const double":
                conv.append(float(a))
            else:
                conv.append(int(a))
        self._fn(int(grid[0]), int(block[0]), *conv)


def _forbidden(name):
    def raiser(*a, **k):
        raise AssertionError(f"library reduction cupy.{name} must not be used by the bounded cold derivation")
    return raiser


class FakeCupy:
    """PLUMBING DOUBLE of the CuPy surface the spatial preferential launchers and the shared derivation use. Elementwise functions delegate to NumPy; the library reductions raise."""

    ndarray = FakeArray
    float64, uint64, uint8, bool_ = np.float64, np.uint64, np.uint8, np.bool_
    sum, min, max, isfinite, isnan = (staticmethod(_forbidden(n)) for n in ("sum", "min", "max", "isfinite", "isnan"))
    add, subtract, multiply, divide, negative, log1p, abs, copyto = (staticmethod(f) for f in (np.add, np.subtract, np.multiply, np.divide, np.negative, np.log1p, np.abs, np.copyto))

    def __init__(self, so):
        self.so, self.allocations, self.h2d, self.modules, self.launches = so, [], [], [], []
        self.rc = None  # the conftest wires the device-id source here; arrays are created on the CURRENT device, as CuPy does
        outer = self

        class RawModule:
            def __init__(self, code, options, backend):
                outer.modules.append((code, tuple(options), backend))

            def get_function(self, name):
                return FakeFn(outer, outer.so, name)

        self.RawModule = RawModule

    def _rec(self, kind, shape, dtype):
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.allocations.append((kind, tuple(np.atleast_1d(shape)), np.dtype(dtype).name, n))

    def _dev(self):
        return 0 if self.rc is None else int(self.rc.current)

    def empty(self, shape, dtype=np.float64):
        self._rec("empty", shape, dtype)
        return fake(np.empty(shape, dtype=dtype), self._dev())

    def zeros(self, shape, dtype=np.float64):
        self._rec("zeros", shape, dtype)
        return fake(np.zeros(shape, dtype=dtype), self._dev())

    def array(self, a, dtype=None, copy=True, order="C"):
        out = np.array(a, dtype=dtype, copy=True, order=order)
        self._rec("array", out.shape, out.dtype)
        if not isinstance(a, FakeArray):
            self.h2d.append(int(out.nbytes))
        return fake(out, getattr(getattr(a, "device", None), "id", self._dev()))

    def asarray(self, a):
        out = np.asarray(a)
        self._rec("asarray", out.shape, out.dtype)
        if not isinstance(a, FakeArray):
            self.h2d.append(int(out.nbytes))
        return fake(out, self._dev())

    def broadcast_to(self, a, shape):
        return np.broadcast_to(np.asarray(a), shape)  # a zero-copy read-only view (no allocation recorded)

    @property
    def allocated_bytes(self):
        return sum(n for *_, n in self.allocations)
