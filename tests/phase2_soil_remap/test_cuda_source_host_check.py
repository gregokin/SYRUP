"""Host-compiler check of the CUDA C SOURCE (no GPU, no CuPy, no device). The text of `soil_water_remap_cuda._source()` is compiled by `g++` against small stand-ins for the CUDA keywords
and atomics, and the plain device function `column_work` (the per-column arithmetic, flag logic and contribution values of the kernel) is called column by column from Python via `ctypes`.
It therefore checks the kernel's per-column math, identity branch, flags and diagnostics against the NumPy reference BIT FOR BIT (compiled with -ffp-contract=off, the host analogue of --fmad=false).
It does NOT execute the kernels' block/shared-memory reductions or any launch: those need Root's hardware run. Skipped when no C++ compiler is available. Author-written."""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from conftest import arrays, random_columns

from maple_syrup import soil_water_remap as sr
from maple_syrup import soil_water_remap_cuda as rc

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="no C++ compiler: the CUDA source host check is not run (no claim)")

STUB = r'''
#include <cmath>
#include <cstdint>
#define __device__
#define __global__
#define __forceinline__ inline
#define __restrict__
#define __shared__ static
struct dim3_ { unsigned x; };
static dim3_ threadIdx, blockIdx, gridDim;
inline void __syncthreads() {}
inline unsigned long long atomicOr(unsigned long long* a, unsigned long long v){ auto o=*a; *a|=v; return o; }
inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long v){ auto o=*a; *a+=v; return o; }
inline unsigned long long atomicMin(unsigned long long* a, unsigned long long v){ auto o=*a; if(v<*a)*a=v; return o; }
inline unsigned long long atomicMax(unsigned long long* a, unsigned long long v){ auto o=*a; if(v>*a)*a=v; return o; }
inline long long __double_as_longlong(double d){ long long r; __builtin_memcpy(&r,&d,8); return r; }
using std::isfinite; using std::fmin; using std::fmax; using std::fabs;
'''
WRAP = r'''
extern "C" int host_column(const double* eo, const double* en, const double* W, const double* up, const double* lo, double* out, double* ra, double* rb, double* aa, double* ab,
                           unsigned long long* diag, double* contrib, double coef_col, double coef_out, long long L, long long C, long long c)
{ return column_work(eo, en, W, up, lo, out, ra, rb, aa, ab, diag, contrib, coef_col, coef_out, L, C, c); }
'''


@pytest.fixture(scope="module")
def lib():
    d = Path(tempfile.mkdtemp())
    (d / "k.cpp").write_text(STUB + rc._source() + WRAP)
    subprocess.run(["g++", "-std=c++17", "-O1", "-ffp-contract=off", "-fPIC", "-shared", "-o", str(d / "k.so"), str(d / "k.cpp")], check=True)
    so = ctypes.CDLL(str(d / "k.so"))
    so.host_column.restype = ctypes.c_int
    return so


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def host_run(lib, a, coef_col, coef_out):
    eo, en, W, up, lo = a
    L, C = W.shape
    out = np.zeros((L, C)); ra, rb, aa, ab = (np.zeros(C) for _ in range(4))
    diag = np.zeros(sr.DIAG_WORDS, dtype=np.uint64)
    diag[2] = np.uint64(0xFFFFFFFFFFFFFFFF)
    contrib = np.zeros((C, 5))
    ok = np.zeros(C, dtype=int)
    for c in range(C):
        ok[c] = lib.host_column(ptr(eo), ptr(en), ptr(W), ptr(up), ptr(lo), ptr(out), ptr(ra), ptr(rb), ptr(aa), ptr(ab), ptr(diag), ptr(contrib[c]),
                                ctypes.c_double(coef_col), ctypes.c_double(coef_out), ctypes.c_longlong(L), ctypes.c_longlong(C), ctypes.c_longlong(c))
    return out, ra, rb, aa, ab, diag, contrib, ok


@pytest.mark.parametrize("L", [1, 2, 3, 5])
def test_cuda_c_column_arithmetic_equals_the_numpy_reference_bitwise(lib, L):
    a = arrays(*random_columns(np.random.default_rng(77 + L), L, 200))
    coef_col, coef_out = sr._coefficients(L)
    out, ra, rb, aa, ab, diag, contrib, ok = host_run(lib, a, coef_col, coef_out)
    ref = sr.remap_intervals(*a[:3], upper_theta=a[3], lower_theta=a[4], cell_area_m2=1.0, backend="numpy", memory_budget_bytes=1 << 30)
    assert ok.all() and diag[1] == 0 and diag[0] == 0
    np.testing.assert_array_equal(out.view(np.uint64), ref.water_m.view(np.uint64))
    for got, want in ((ra, ref.removed_above_m), (rb, ref.removed_below_m)):
        np.testing.assert_array_equal(got.view(np.uint64), want.view(np.uint64))
    np.testing.assert_array_equal((aa).view(np.uint64), ref.added_above_m.view(np.uint64))
    np.testing.assert_array_equal((ab).view(np.uint64), ref.added_below_m.view(np.uint64))
    assert np.all(contrib[:, 4] > 0)


def test_cuda_c_identity_preserves_signed_zero_and_flags_report_exact_integers(lib):
    e = np.cumsum(np.ones((3, 4)), axis=0) - 1.0
    W = np.full((2, 4), 0.5)
    W[1, :] = -0.0
    a = [e, e.copy(), W, np.zeros(4), np.zeros(4)]
    coef_col, coef_out = sr._coefficients(2)
    out, ra, rb, aa, ab, diag, contrib, ok = host_run(lib, a, coef_col, coef_out)
    np.testing.assert_array_equal(out.view(np.uint64), W.view(np.uint64))
    assert not ra.any() and not aa.any() and ok.all()
    bad = [x.copy() for x in a]
    bad[1][2, 3] = bad[1][1, 3]  # new edges collapsed in column 3
    bad[2][0, 1] = -1.0  # water in column 1
    out2, *_rest = host_run(lib, bad, coef_col, coef_out)
    diag2 = _rest[4]
    packed = int(diag2[2])
    assert (int(diag2[0]), int(diag2[1]), packed >> 8, packed & 0xFF) == (sr.FLAG_NEW_EDGES | sr.FLAG_WATER, 2, 1, sr.FLAG_WATER)
    assert (sr.MAX_CUDA_COLUMNS << 8) < 2**64  # the explicit gate keeps `column << 8 | flags` exactly representable in the 64-bit word
