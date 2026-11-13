"""The rendered common CUDA source, compiled by g++ against stand-ins for the CUDA keywords (`-ffp-contract=off`, the host analogue of `--fmad=false`), is EXECUTED for every cell and compared bit for bit with
the pre-A4.1 source compiled the same way: factor kernel and step kernel, every mode/bottom/gravity/share/pulse combination, inactive cells, flagged invalid inputs. Also a token-stream comparison of the
macro-expanded step kernel. No GPU is involved; this is host arithmetic of the same C text, not a device claim."""
from __future__ import annotations

import ctypes
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from conftest import STUB, have_gxx, preprocess

pytestmark = pytest.mark.skipif(not have_gxx(), reason="no C++ compiler: the host-compiled source checks are not run (no claim)")

WRAP = r"""
extern "C" void host_factors(int n_if, const double* cond, const double* cond_up, const double* pair_inv, double dt, int directional, double* factor, double* factor_up)
{
    for (int k = 0; k < n_if; ++k) {
        blockDim.x = 128; blockIdx.x = (unsigned)(k / 128); threadIdx.x = (unsigned)(k % 128);
        maple_syrup_subsurface_factors(n_if, cond, cond_up, pair_inv, dt, directional, factor, factor_up);
    }
}
extern "C" void host_step(long long n, int nl, const double* w_in, const double* add, const double* pet, const unsigned char* active,
    const double* dz, const double* res_m, const double* fc_m, const double* sat_m, const double* cap_tol, const double* factor, const double* pair_inv,
    double dt, double res_theta0, double stress, double fct, int gravity, double cell_tol, double no_flag,
    const double* factor_up, int bottom, const double* share, int has_share, double pulse, int pulse_on,
    double* w_out, double* perc, double* diff, double* ret, double* drain, double* evap, double* unmet, double* rows)
{
    for (long long i = 0; i < n + 3; ++i) {  // three padded lanes beyond n exercise the invalid-lane guard
        blockDim.x = 128; blockIdx.x = (unsigned)(i / 128); threadIdx.x = (unsigned)(i % 128);
        maple_syrup_subsurface_step(n, nl, w_in, add, pet, active, dz, res_m, fc_m, sat_m, cap_tol, factor, pair_inv, dt, res_theta0, stress, fct, gravity, cell_tol, no_flag,
                                    factor_up, bottom, share, has_share, pulse, pulse_on, w_out, perc, diff, ret, drain, evap, unmet, rows);
    }
}
"""


def build(source: str):
    d = Path(tempfile.mkdtemp())
    (d / "k.cpp").write_text(STUB + source + WRAP)
    subprocess.run(["g++", "-std=c++17", "-O1", "-ffp-contract=off", "-fPIC", "-shared", "-o", str(d / "k.so"), str(d / "k.cpp")], check=True)
    so = ctypes.CDLL(str(d / "k.so"))
    P, LL, D, I = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_double, ctypes.c_int
    so.host_factors.argtypes = [I, P, P, P, D, I, P, P]
    so.host_factors.restype = None
    so.host_step.argtypes = [LL, I, P, P, P, P, P, P, P, P, P, P, P, D, D, D, D, I, D, D, P, I, P, I, D, I] + [P] * 8
    so.host_step.restype = None
    return so


@pytest.fixture(scope="module")
def libs(old, new):
    return build(old._SOURCE), build(new._SOURCE)


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def make_inputs(rng, L, N, donor):
    thick = np.linspace(0.08, 0.3, L)
    res_m = thick * np.linspace(0.01, 0.04, L)
    fc_m = thick * np.linspace(0.18, 0.25, L)
    sat_m = thick * np.linspace(0.4, 0.5, L)
    cap_tol = sat_m * 1e-15
    pair = np.zeros(max(L - 1, 1))
    cond = np.zeros(max(L - 1, 1))
    cond_up = np.zeros(max(L - 1, 1))
    if L > 1:
        pair[: L - 1] = 1.0 / thick[:-1] + 1.0 / thick[1:]
        cond[: L - 1] = np.linspace(2e-7, 9e-7, L - 1)
        cond_up[: L - 1] = cond[: L - 1] * 1.7 if donor else cond[: L - 1]
    water = sat_m[:, None] * rng.random((L, N))
    water[:, 0] = np.nan if N > 3 else water[:, 0]
    if N > 5:
        water[0, 4] = -1e-3
        water[-1, 5] = sat_m[-1] * 3
    add = rng.random((L, N)) * 1e-3
    pet = rng.random(N) * 1e-7
    active = (rng.random(N) < 0.8).astype(np.uint8)
    add[:, active == 0] = 0.0
    pet[active == 0] = 0.0
    if N > 8:
        pet[7] = np.inf
    return dict(thick=thick, res_m=res_m, fc_m=fc_m, sat_m=sat_m, cap_tol=cap_tol, pair=pair, cond=cond, cond_up=cond_up, water=water, add=add, pet=pet, active=active)


def run(so, x, L, N, dt, donor, bottom, gravity, share, pulse):
    n_if = L - 1
    m = max(n_if, 1)
    factor, factor_up = np.zeros(m), np.zeros(m) if donor else None
    if factor_up is None:
        factor_up = factor
    if n_if > 0:
        so.host_factors(n_if, ptr(x["cond"]), ptr(x["cond_up"]), ptr(x["pair"]), dt, int(donor), ptr(factor), ptr(factor_up))
    out = {k: np.zeros(s) for k, s in dict(w=(L, N), perc=(n_if, N), diff=(n_if, N), ret=N, drain=N, evap=N, unmet=N, rows=(4, N)).items()}
    cell_tol = 1e-13
    shr = share if share is not None else x["thick"]
    so.host_step(N, L, ptr(x["water"]), ptr(x["add"] if pulse is None else x["thick"]), ptr(x["pet"]), ptr(x["active"]), ptr(x["thick"]), ptr(x["res_m"]), ptr(x["fc_m"]), ptr(x["sat_m"]),
                 ptr(x["cap_tol"]), ptr(factor), ptr(x["pair"]), dt, float(x["res_m"][0] / x["thick"][0]), 0.5, 1.0, int(gravity), cell_tol, float(2**40), ptr(factor_up),
                 0 if bottom == "closed" else 1, ptr(shr), 0 if share is None else 1, 0.0 if pulse is None else pulse, 0 if pulse is None else 1,
                 *[ptr(out[k]) for k in ("w", "perc", "diff", "ret", "drain", "evap", "unmet", "rows")])
    return out, factor, factor_up


@pytest.mark.parametrize("L", [1, 2, 3, 5])
@pytest.mark.parametrize("donor", [False, True])
@pytest.mark.parametrize("bottom", ["open", "closed"])
def test_rendered_common_source_executes_bit_identically_to_the_pre_a41_source(libs, L, donor, bottom):
    so_old, so_new = libs
    rng = np.random.default_rng(10 * L + donor)
    N = 40
    x = make_inputs(rng, L, N, donor)
    x["water"][:, 0] = x["sat_m"] * 0.5  # one finite column; the NaN/negative/over-saturated cells stay flagged elsewhere
    for dt in (0.0, 1e-6, 3600.0, 86400.0, 1e7):
        for gravity in (True, False):
            for share in (None, rng.random(N)):
                for pulse in (None, 2.5e-3):
                    if pulse is not None and not dt > 0.0:
                        continue
                    a, fa, fua = run(so_old, x, L, N, dt, donor, bottom, gravity, share, pulse)
                    b, fb, fub = run(so_new, x, L, N, dt, donor, bottom, gravity, share, pulse)
                    np.testing.assert_array_equal(fa.view(np.uint64), fb.view(np.uint64))
                    np.testing.assert_array_equal(fua.view(np.uint64), fub.view(np.uint64))
                    for k in a:
                        both_nan = np.isnan(a[k]) & np.isnan(b[k])  # the sign bit of an invalid-input NaN is not a contract
                        np.testing.assert_array_equal(np.where(both_nan, 0.0, a[k]).view(np.uint64), np.where(both_nan, 0.0, b[k]).view(np.uint64), err_msg=f"{k} L{L} dt{dt}")


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*(?:[eE][-+]?\d+)?[uUlLfF]*|\S", text)


def step_kernel_text(source: str) -> str:
    i = source.index('extern "C" __global__ void maple_syrup_subsurface_step(')
    return source[i:]


def test_the_macro_expanded_step_kernel_is_token_identical_to_the_pre_a41_step_kernel(old, new):
    a = tokens(step_kernel_text(preprocess(old._SOURCE)))
    b = tokens(step_kernel_text(preprocess(new._SOURCE)))
    assert len(a) > 1500 and a == b  # every parameter read, the guard and the row epilogue expand to the old tokens


def test_the_factor_kernel_differs_only_by_the_shared_relaxation_function(old, new):
    pa, pb = preprocess(old._SOURCE), preprocess(new._SOURCE)
    fa = pa[pa.index("maple_syrup_subsurface_factors"): pa.index('extern "C" __global__ void maple_syrup_subsurface_step(')]
    fb = pb[pb.index("double relax_factor"): pb.index('extern "C" __global__ void maple_syrup_subsurface_step(')]
    assert "expm1" in fa and "relax_factor" not in fa
    assert tokens(pb[pb.index("double relax_factor"): pb.index("}", pb.index("double relax_factor")) + 1]) == tokens(
        "double relax_factor(const double g, const double s, const double dt) { const double rate = g * s; const double x = rate * dt; return -expm1(-x); }")
    assert "relax_factor(cond[k], s, dt)" in fb.replace("  ", " ") and "relax_factor(cond_up[k], s, dt)" in fb.replace("  ", " ")
