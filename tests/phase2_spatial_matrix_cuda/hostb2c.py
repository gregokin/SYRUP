"""HOST execution of the rendered spatial CUDA MATRIX source (NOT a GPU, NOT hardware evidence). g++ compiles the C text against stand-ins for the CUDA keywords (`-ffp-contract=off`, the host analogue of
`--fmad=false`); every kernel is launched by REAL threads with a REAL barrier (the A4.2 harness stand-ins; `__shared__` = a function-static array of the running block, `atomicOr/Add/Max` on unsigned long long
= `__atomic_*`). The fake CuPy below is a numpy-backed PLUMBING DOUBLE for the Python launchers: it records every allocation, host-to-device copy, kernel launch and read, and executes nothing by itself. Author-written."""
from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2_spatial_cuda"))
import hostgpu  # noqa: E402

EXTRA = "\nusing std::isinf;  /* CUDA device code has a global isinf */\n"
MAPS3 = ("const double*", "const long long", "const long long") * 3
LAW3 = ("const double*", "const long long") * 3
STEP_TYPES = (("const long long", "const int") + ("const double*",) * 6 + ("const unsigned char*", "const double") + MAPS3 + ("const double",) + LAW3 + ("const double", "const double", "const int")
              + ("double*",) * 5 + ("double*", "unsigned long long*"))
SPEC = {
    "maple_syrup_spatial_matrix_step": STEP_TYPES,
    "maple_syrup_spatial_matrix_finalize": ("const double*", "const long long", "unsigned long long*"),
    "maple_syrup_spatial_matrix_validate": ("const long long", "const double*", "const double", "const double", "unsigned long long*"),
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
    """numpy-backed 'device' array (plumbing double): `.device`, `.get(out=)` (honours `out`, as CuPy does), `.fill`."""

    device = FakeDevice(0)

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


class FakeCupy:
    """PLUMBING DOUBLE of the CuPy surface the spatial matrix launchers use; `RawModule` maps the kernels to the host-compiled library. Records allocations, host-to-device copies and launches."""

    ndarray = FakeArray
    float64, uint64, uint8, bool_ = np.float64, np.uint64, np.uint8, np.bool_

    def __init__(self, so):
        self.so, self.allocations, self.h2d, self.modules, self.launches = so, [], [], [], []
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

    def empty(self, shape, dtype=np.float64):
        self._rec("empty", shape, dtype)
        return fake(np.empty(shape, dtype=dtype))

    def zeros(self, shape, dtype=np.float64):
        self._rec("zeros", shape, dtype)
        return fake(np.zeros(shape, dtype=dtype))

    def array(self, a, dtype=None, copy=True, order="C"):
        out = np.array(a, dtype=dtype, copy=True, order=order)
        self._rec("array", out.shape, out.dtype)
        if not isinstance(a, FakeArray):
            self.h2d.append(int(out.nbytes))
        return fake(out, getattr(getattr(a, "device", None), "id", 0))

    def asarray(self, a):
        out = np.asarray(a)
        self._rec("asarray", out.shape, out.dtype)
        if not isinstance(a, FakeArray):
            self.h2d.append(int(out.nbytes))
        return fake(out)

    @property
    def allocated_bytes(self):
        return sum(n for *_, n in self.allocations)
