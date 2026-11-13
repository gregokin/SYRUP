"""HOST execution of the rendered D2 floating CUDA sources (NOT a GPU, NOT hardware evidence): g++ compiles each source against the D1 stand-ins (`hostresident`: CUDA keywords as plain C++, threads with a
`std::barrier` for `__syncthreads`, a function-static array for `__shared__`) plus the few intrinsics the D2 sources add; a numpy-backed CuPy PLUMBING DOUBLE serves the Python launchers (allocation recording,
argument conversion, counted transfers). The double executes nothing by itself. Author-written."""
from __future__ import annotations

import ctypes
import importlib.util
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_p = Path(__file__).resolve().parents[1] / "phase2_floating_column" / "hostresident.py"
_spec = importlib.util.spec_from_file_location("d1_hostresident", _p)
hr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hr)
have_toolchain, fake, FakeArray, FakeFn = hr.have_toolchain, hr.fake, hr.FakeArray, hr.FakeFn

EXTRA = r"""
using std::isinf; using std::isnan;
inline double __longlong_as_double(long long x) { double r; std::memcpy(&r, &x, 8); return r; }
inline unsigned long long atomicMax(unsigned long long* a, unsigned long long v) { unsigned long long o = __atomic_load_n(a, __ATOMIC_SEQ_CST); while (o < v && !__atomic_compare_exchange_n(a, &o, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {} return o; }
"""


def build(source: str):
    spec = hr.parse_spec(source)
    wrappers = []
    for name, types in spec.items():
        params = ", ".join(f"{t} a{i}" for i, t in enumerate(types))
        call = ", ".join(f"({t})a{i}" if t.endswith("*") else f"a{i}" for i, t in enumerate(types))
        wrappers.append(f'extern "C" void host_{name}(int grid, int block, {params}) {{ launch(grid, block, [=] {{ {name}({call}); }}); }}')
    d = Path(tempfile.mkdtemp())
    (d / "k.cpp").write_text(hr.STUB + EXTRA + source + "\n".join(wrappers))
    subprocess.run(["g++", "-std=c++20", "-O1", "-pthread", "-ffp-contract=off", "-fPIC", "-shared", "-o", str(d / "k.so"), str(d / "k.cpp")], check=True, capture_output=True, text=True)
    so = ctypes.CDLL(str(d / "k.so"))
    for name, types in spec.items():
        fn = getattr(so, f"host_{name}")
        fn.argtypes = [ctypes.c_int, ctypes.c_int] + [hr._ctype(t) for t in types]
        fn.restype = None
    return so, spec


def _forbidden(name):
    def raiser(*a, **k):
        raise AssertionError(f"library reduction cupy.{name} must not be used by the cold derivation")

    return raiser


class MultiFakeCupy(hr.FakeCupy):
    """A FakeCupy over several host-compiled libraries: `RawModule(code)` serves the library whose marker kernel name occurs in the code."""

    add, subtract, multiply, divide, negative, log1p, abs, copyto = (staticmethod(f) for f in (np.add, np.subtract, np.multiply, np.divide, np.negative, np.log1p, np.abs, np.copyto))
    sum, min, max, isfinite, isnan = (staticmethod(_forbidden(n)) for n in ("sum", "min", "max", "isfinite", "isnan"))  # the bounded cold reducer must be used instead of any library reduction

    maximum, minimum, array_equal = (staticmethod(f) for f in (np.maximum, np.minimum, np.array_equal))

    def ascontiguousarray(self, a):
        return self._rec(hr.fake(np.ascontiguousarray(a)))

    def ones(self, shape, dtype=np.float64):
        return self._rec(hr.fake(np.ones(shape, dtype=dtype)))

    def broadcast_to(self, a, shape):
        return np.broadcast_to(np.asarray(a), shape)  # a zero-copy read-only view

    def __init__(self, libs):
        so, spec = libs[0][1]
        super().__init__(so, spec)
        self.libs = libs

    def RawModule(self, code, options, backend):
        self.compiled.append((hash(code), tuple(options), backend))
        for marker, (so, spec) in self.libs:
            if marker in code:
                return SimpleNamespace(get_function=lambda name, so=so, spec=spec: FakeFn(so, name, spec, self.launches))
        raise AssertionError("no host-compiled library for this source")
