"""CUDA withdrawal source: the macro-expanded kernel is token-identical to the pre-B1.a kernel, the host-compiled C text (g++, CUDA keywords stubbed, `-ffp-contract=off`) EXECUTES bit-identically, the rendering is pure
and refuses incomplete readers, the Python surface is AST-identical except the two intended changes, compile flags/kernel name/launch arguments are unchanged (CuPy PLUMBING DOUBLE, no GPU/NVRTC), the
Numba function is genuinely compiled nopython without fastmath and allocates nothing, and the common operator passes its cached arrays through (alias proof). NOT hardware evidence. Author-written."""
from __future__ import annotations

import ast
import ctypes
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from conftest import NEW_PATH, OLD_PATH

from maple_syrup import subsurface as ss

needs_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="no C++ compiler (host checks not run; no claim)")
STUB = r"""
#include <cmath>
#include <cstdint>
#include <cstring>
#define __global__
#define __restrict__
struct dim3_ { unsigned x; };
static dim3_ threadIdx, blockIdx, blockDim;
inline double __longlong_as_double(long long v) { double d; std::memcpy(&d, &v, 8); return d; }
"""
WRAP = r"""
extern "C" void host_withdraw(const double* w, const double* q, const double* residual, const double* satb, const unsigned char* active_u8, double* out_w, double* actual, double* unmet,
                              double* scale, double* rows, int L, long long N)
{
    for (long long j = 0; j < N + 3; ++j) {  // three padded lanes beyond N exercise the lane guard
        blockDim.x = 256; blockIdx.x = (unsigned)(j / 256); threadIdx.x = (unsigned)(j % 256);
        withdraw_fused(w, q, residual, satb, active_u8, out_w, actual, unmet, scale, rows, L, N);
    }
}
"""


def build(source):
    d = Path(tempfile.mkdtemp())
    (d / "k.cpp").write_text(STUB + source + WRAP)
    subprocess.run(["g++", "-std=c++17", "-O1", "-ffp-contract=off", "-fPIC", "-shared", "-o", str(d / "k.so"), str(d / "k.cpp")], check=True, capture_output=True, text=True)
    so = ctypes.CDLL(str(d / "k.so"))
    P = ctypes.c_void_p
    so.host_withdraw.argtypes = [P] * 10 + [ctypes.c_int, ctypes.c_longlong]
    so.host_withdraw.restype = None
    return so


def preprocess(source):
    d = Path(tempfile.mkdtemp())
    (d / "k.cpp").write_text(source)
    return subprocess.run(["g++", "-E", "-P", "-x", "c++", str(d / "k.cpp")], check=True, capture_output=True, text=True).stdout


def tokens(text):
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*(?:[eE][-+]?\d+)?[uUlLfF]*|\S", text)


@needs_gxx
def test_macro_expanded_cuda_kernel_is_token_identical_to_the_pre_b1a_kernel(old, new):
    a, b = tokens(preprocess(old._CUDA_SRC)), tokens(preprocess(new._CUDA_SRC))
    assert len(a) > 250 and a == b


@needs_gxx
def test_host_compiled_cuda_text_executes_bit_identically(old, new):
    so_old, so_new = build(old._CUDA_SRC), build(new._CUDA_SRC)
    P = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
    rng = np.random.default_rng(2)
    for L in (1, 2, 5, 9):
        for N in (1, 255, 257):
            res = np.sort(rng.uniform(0, 0.01, L))
            satb = res + rng.uniform(0.05, 0.2, L)
            act = (rng.random(N) < 0.7).astype(np.uint8)
            w = satb[:, None] * rng.random((L, N))
            q = rng.uniform(0, 0.1, (L, N)) * act[None, :]
            if N > 3:
                w[0, 0], w[-1, 1], q[0, 2], w[0, 3] = np.nan, -1e-3, -1.0, -0.0
                q[-1, 3 % N] = np.inf
            outs = []
            for so in (so_old, so_new):
                o = [np.empty((L, N)) for _ in range(4)] + [np.empty((8, N))]
                so.host_withdraw(P(w), P(q), P(res), P(satb), P(act), *[P(x) for x in o], L, N)
                outs.append(o)
            for x, y in zip(*outs):
                both_nan = np.isnan(x) & np.isnan(y)
                np.testing.assert_array_equal(np.where(both_nan, 0.0, x).view(np.uint64), np.where(both_nan, 0.0, y).view(np.uint64))


def test_rendering_is_pure_complete_and_the_body_occurs_once(new):
    assert new.render_withdrawal_source(new._CUDA_COMMON_READERS, new._CUDA_COMMON_HEAD) == new._CUDA_SRC
    assert new._CUDA_SRC.count(new._CUDA_BODY) == 1 and new._CUDA_SRC.count("withdraw_fused(") == 1
    assert not hasattr(new, "register_variant") and not any(isinstance(v, (dict, list, set)) and k.startswith("_CUDA") for k, v in vars(new).items())
    for raw in (r"\bresidual\[", r"\bsatb\[", r"\brows\[", r"\breturn;"):
        assert re.search(raw, new._CUDA_BODY) is None, raw  # every parameter read, the guard and the epilogue go through the macros
    for m in new.WITHDRAW_MACROS:
        assert f"#define {m}" in new._CUDA_COMMON_READERS and re.search(rf"\b{m}\b", new._CUDA_BODY), m
    with pytest.raises(ss.SubsurfaceError, match="do not define"):
        new.render_withdrawal_source("#define WITHDRAW_GUARD\n", "head")
    with pytest.raises(ss.SubsurfaceError, match="must be strings"):
        new.render_withdrawal_source(None, "head")
    probe = new.render_withdrawal_source("".join(f"#define {m}\n" for m in new.WITHDRAW_MACROS), "HEAD\n", "TAIL\n")
    assert probe.endswith(new._CUDA_BODY + "TAIL\n") and new._CUDA_SRC == new.render_withdrawal_source(new._CUDA_COMMON_READERS, new._CUDA_COMMON_HEAD)  # nothing was stored


def functions(path):
    return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}


def test_python_surface_is_ast_identical_except_the_two_intended_changes(old, new):
    a, b = functions(OLD_PATH), functions(NEW_PATH)
    assert set(b) - set(a) == {"render_withdrawal_source"} and set(a) <= set(b)
    changed = {n for n in a if a[n] != b[n]}
    assert changed == {"_fused_numpy"}, changed  # prepare/_launch/_run_withdrawal/withdraw/estimate/admission/guards are byte-for-byte the same code
    for name in ("VERSION", "THREADS", "_N_ROWS", "_PACKET"):
        assert getattr(old, name) == getattr(new, name)
    assert old.estimate_transpiration_memory(5, 1000, backend="numpy").as_dict() == new.estimate_transpiration_memory(5, 1000, backend="numpy").as_dict()
    for be in ("numba", "cuda"):
        assert old.estimate_transpiration_memory(7, 123, backend=be, reserve_bytes=3).as_dict() == new.estimate_transpiration_memory(7, 123, backend=be, reserve_bytes=3).as_dict()


class FakeFn:
    calls: list = []

    def __init__(self, source, name, options):
        self.source, self.name, self.options = source, name, options

    def compile(self):
        FakeFn.calls.append(("compile", self.name, tuple(self.options)))

    def __call__(self, grid, block, args):
        FakeFn.calls.append(("launch", grid, block, len(args)))


class FakeXp:
    """CuPy PLUMBING DOUBLE: only what `prepare_transpiration` touches for a cuda column; numpy-backed, no GPU."""

    float64, uint8 = np.float64, np.uint8

    class cuda:  # noqa: N801
        class Device:
            id = 0

            def synchronize(self):
                pass

    RawKernel = FakeFn

    def asarray(self, a, dtype=None):
        return np.asarray(a, dtype=dtype)

    def ascontiguousarray(self, a):
        return np.ascontiguousarray(a)

    def ones(self, s, dtype=None):
        return np.ones(s, dtype=dtype)

    def zeros(self, s, dtype=None):
        return np.zeros(s, dtype=dtype)

    def empty(self, s, dtype=None):
        return np.empty(s, dtype=dtype)


def test_cuda_prepare_and_launch_use_the_unchanged_flags_name_and_arguments(old, new):
    """The CUDA branch of `prepare_transpiration`/`_launch` is exercised through a plumbing double: same kernel name, same compile options, same argument count and grid, source = the rendered source."""
    seen = {}
    for label, mod in (("old", old), ("new", new)):
        FakeFn.calls = []
        xp = FakeXp()
        lay = ss.soil_layers(np.array([0.0, 0.1, 0.25]), theta_residual=0.02, theta_field_capacity=0.2, theta_saturation=0.4, theta_evaporation_stress=0.5, diffusivity_m2_s=1e-7)
        col = ss.prepare_column(lay, np.ones(4, bool), 1.0, backend="numpy")
        fake_col = type("C", (), {})()
        for k in ("layers", "n_layers", "n_cells", "n_active", "cell_area_m2", "active_u8", "reserve_bytes", "device_id"):
            setattr(fake_col, k, getattr(col, k))
        fake_col.backend, fake_col.xp = "cuda", xp
        fake_col.available_bytes_fn = None
        import maple_syrup.subsurface as _ss

        orig = _ss._check_column
        _ss._check_column = lambda c: None  # the plumbing column is not a SoilColumn
        try:
            op = mod.prepare_transpiration(fake_col)
        finally:
            _ss._check_column = orig
        seen[label] = (list(FakeFn.calls), op.kernel.source, op.kernel.name, tuple(op.kernel.options), op.residual_xp.tobytes(), op.saturation_xp.tobytes())
    assert seen["old"][0] == seen["new"][0] and seen["old"][2:] == seen["new"][2:] and seen["new"][2] == "withdraw_fused" and seen["new"][3] == ("-std=c++11",)
    assert seen["new"][1] == new._CUDA_SRC and seen["old"][1] == old._CUDA_SRC


def test_numba_body_is_genuinely_compiled_without_fastmath_and_allocates_nothing(new):
    nb = pytest.importorskip("numba")
    k = new._fused_numba
    assert type(k).__name__ == "CPUDispatcher" and k.targetoptions.get("nopython") is True and not k.targetoptions.get("fastmath")
    w, q, res, satb = np.ones((3, 5)) * 0.02, np.zeros((3, 5)), np.full(3, 0.005), np.full(3, 0.4)
    k(w, q, res, satb, np.ones(5, np.uint8), *[np.empty((3, 5)) for _ in range(4)], np.empty((8, 5)))
    assert len(k.signatures) >= 1 and all(not (s and "pyobject" in str(s).lower()) for s in k.signatures)
    cres = k.overloads[k.signatures[0]]
    assert cres.objectmode is False and cres.lifted == ()
    src = ast.dump(ast.parse(NEW_PATH.read_text()))
    body = [n for n in ast.walk(ast.parse(NEW_PATH.read_text())) if isinstance(n, ast.FunctionDef) and n.name == "_fused_numba"][0]
    calls = {c.func.attr for c in ast.walk(body) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)}
    assert not ({"empty", "zeros", "ones", "full", "copy"} & calls)  # the body allocates no local array: the readers alias the operator's cached vectors
    assert "_read_residual" in src and "_read_satb" in src


def test_the_common_operator_passes_its_cached_vectors_to_the_body_unchanged(new, monkeypatch):
    """Memory-alias proof: the kernel receives the operator's OWN `residual_xp`/`saturation_xp` objects (no copy, no extra `(L,)` or `(L, N)` array), for NumPy and Numba."""
    lay = ss.soil_layers(np.array([0.0, 0.1, 0.25, 0.5]), theta_residual=0.02, theta_field_capacity=0.2, theta_saturation=0.4, theta_evaporation_stress=0.5, diffusivity_m2_s=np.array([1e-7, 2e-7]))
    for backend in ("numpy", "numba"):
        col = ss.prepare_column(lay, np.ones(6, bool), 1.0, backend=backend)
        op = new.prepare_transpiration(col)
        got = {}
        real = new._launch

        def spy(o, w, q, residual, satb, *rest):
            got["res"], got["sat"] = residual, satb
            return real(o, w, q, residual, satb, *rest)

        monkeypatch.setattr(new, "_launch", spy)
        new.withdraw_transpiration(op, ss.column_state(col, np.full((3, 6), 0.02)), np.zeros((3, 6)))
        monkeypatch.setattr(new, "_launch", real)
        assert got["res"] is op.residual_xp and got["sat"] is op.saturation_xp and op.residual_xp.shape == (3,) == op.saturation_xp.shape
