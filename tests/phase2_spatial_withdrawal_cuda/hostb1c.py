"""HOST execution of the rendered spatial CUDA WITHDRAWAL source (NOT a GPU, NOT hardware evidence). g++ compiles the C text against stand-ins for the CUDA keywords (`-ffp-contract=off`, the host analogue of
`--fmad=false`); every kernel is launched by REAL threads with a REAL barrier (the A4.2 harness stand-ins). The fake CuPy is a numpy-backed PLUMBING DOUBLE for the Python launchers. Author-written."""
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
inline double __longlong_as_double(long long v) { double d; std::memcpy(&d, &v, 8); return d; }
"""
MAPS = ("const double*", "const long long", "const long long") * 3
SPEC = {
    "maple_syrup_spatial_withdraw": ("const long long", "const int", "const double*", "const double*", "const unsigned char*") + MAPS + ("const double",) + ("double*",) * 5,
    "maple_syrup_spatial_withdraw_finalize": ("const double*", "const long long", "double*"),
    "maple_syrup_spatial_availability": ("const long long", "const int", "const double*", "const unsigned char*") + MAPS + ("const double", "double*", "unsigned long long*"),
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


class FakeFn(hostgpu.FakeFn):
    def __init__(self, so, name):
        self.name, self.attributes, self._fn, self._types = name, {"max_threads_per_block": 1024}, getattr(so, f"host_{name}"), SPEC[name]
        self.launches = []


class FakeCupy(hostgpu.FakeCupy):
    def __init__(self, so):
        super().__init__(so)
        outer = self

        class RawModule:
            def __init__(self, code, options, backend):
                outer.modules.append((code, tuple(options), backend))

            def get_function(self, name):
                return FakeFn(outer.so, name)

        self.RawModule = RawModule
