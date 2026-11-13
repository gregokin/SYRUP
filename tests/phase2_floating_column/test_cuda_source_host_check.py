"""Host-compiler execution of the FLOATING CUDA kernel SOURCE (no GPU, no CuPy, no NVRTC): the step and reconciliation kernels are compiled by g++ with stand-ins for the CUDA keywords and executed over a grid emulated on the host,
then compared with the NumPy reference / compiled Numba port. Checks the kernel TEXT (arithmetic, flags, per-cell lower boundary, zero slots, inactive cells, pulse, share, layouts/strides); NOT a launch, atomics, the device libm or the CUDA
flags (`-ffp-contract=off` is only the host analogue of `--fmad=false`). Skipped without g++. Author-written."""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from conftest import BACKENDS, numba_ok  # noqa: F401

from maple_syrup import floating_soil_layers as fl
from maple_syrup import subsurface as ss
from maple_syrup import subsurface_cuda_floating as cf
from maple_syrup.conservation import volume_roundoff_bound_m3

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="no g++: the host execution of the CUDA source is not run (no claim)")
STUB = r'''
#include <cmath>
#include <cstdint>
#define __device__
#define __global__
#define __forceinline__ inline
struct dim3_ { unsigned x; };
static dim3_ threadIdx, blockIdx, blockDim;
using std::isfinite; using std::fmin; using std::fmax; using std::fabs; using std::expm1;
'''
WRAP = r'''
extern "C" void host_step(const long long n, const int nl, const double* w_in, const double* add, const double* pet, const unsigned char* active, const unsigned char* bopen,
    const double* edges_p, const double* depth_p, const double* sepn, const double* tr_p, const long long tr_sl, const long long tr_sc, const double* tf_p, const long long tf_sl, const long long tf_sc,
    const double* ts_p, const long long ts_sl, const long long ts_sc, const double* kif_p, const long long kif_sl, const long long kif_sc, const double dcoef, const int directional,
    const double coef, const double dt, const double stress, const double fct, const int gravity, const double cell_tol, const double no_flag, const double* share, const int has_share,
    const double pulse, const int pulse_on, double* w_out, double* perc, double* diff, double* ret, double* drain, double* evap, double* unmet, double* rows)
{
    blockDim.x = 128;
    for (long long b = 0; b < (n + 127) / 128; ++b) { blockIdx.x = (unsigned)b; for (unsigned t = 0; t < 128; ++t) { threadIdx.x = t;
        maple_syrup_floating_step(n, nl, w_in, add, pet, active, bopen, edges_p, depth_p, sepn, tr_p, tr_sl, tr_sc, tf_p, tf_sl, tf_sc, ts_p, ts_sl, ts_sc, kif_p, kif_sl, kif_sc, dcoef, directional, coef, dt,
                                  stress, fct, gravity, cell_tol, no_flag, share, has_share, pulse, pulse_on, w_out, perc, diff, ret, drain, evap, unmet, rows); } }
}
extern "C" void host_reconcile(const long long n, const int nl, const double* w_in, const unsigned char* active, const unsigned char* bopen, const double* edges_p, const double* depth_p, const double* sepn,
    const double* tr_p, const long long tr_sl, const long long tr_sc, const double* tf_p, const long long tf_sl, const long long tf_sc, const double* ts_p, const long long ts_sl, const long long ts_sc,
    const double coef, double* w_out, double* perc, double* surface, double* export_m)
{
    blockDim.x = 128;
    for (long long b = 0; b < (n + 127) / 128; ++b) { blockIdx.x = (unsigned)b; for (unsigned t = 0; t < 128; ++t) { threadIdx.x = t;
        maple_syrup_floating_reconcile(n, nl, w_in, active, bopen, edges_p, depth_p, sepn, tr_p, tr_sl, tr_sc, tf_p, tf_sl, tf_sc, ts_p, ts_sl, ts_sc, coef, w_out, perc, surface, export_m); } }
}
'''
EDGES = np.array([0.0, 0.1, 0.3, 0.6])
P = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
ll = ctypes.c_longlong


@pytest.fixture(scope="module")
def lib():
    d = Path(tempfile.mkdtemp())
    (d / "k.cpp").write_text(STUB + cf.render_source() + WRAP)
    subprocess.run(["g++", "-std=c++17", "-O0", "-ffp-contract=off", "-shared", "-fPIC", "-o", str(d / "k.so"), str(d / "k.cpp")], check=True, capture_output=True)
    return ctypes.CDLL(str(d / "k.so"))


def case(N, seed, mode="tietjen_donor_depth", bottom="open", layout="cell"):
    rng = np.random.default_rng(seed)
    L = 3
    depth = rng.choice([0.0, 0.03, 0.1, 0.2, 0.45, 0.6, 0.9], size=N)
    act = rng.random(N) > 0.15
    if layout == "cell":
        tf = rng.uniform(0.1, 0.2, (L, N)); ts = tf + rng.uniform(0.1, 0.2, (L, N)); tr = rng.uniform(0, 0.05, (L, N)); k = rng.uniform(1e-7, 1e-5, (L - 1, N)); kind = 2
    else:
        tf = np.array([0.12, 0.15, 0.18]); ts = tf + 0.15; tr = np.array([0.0, 0.02, 0.03]); k = np.array([2e-6, 4e-6]); kind = 1
    kw = dict(interface_conductivity_m_s=k, diffusion_coefficient=0.7) if mode != "constant" else dict(diffusivity_m2_s=k)
    col = fl.prepare_floating_column(EDGES, theta_residual=tr, theta_field_capacity=tf, theta_saturation=ts, theta_evaporation_stress=0.3, diffusion_mode=mode, bottom_drainage=bottom, soil_depth_m=depth,
                                     active_mask=act, cell_area_m2=2.0, **kw)
    h = np.stack([fl.porous_thickness(col, kk) for kk in range(L)])
    tsb = np.broadcast_to(ts.reshape(-1, 1) if ts.ndim == 1 else ts, h.shape)
    w = h * tsb * rng.uniform(0.05, 1.0, h.shape)
    w[:, ~act] = 5.0
    return rng, col, w, act, kind


def strides(kind, N):
    return {"scalar": (0, 0), "shared": (1, 0), "cell": (N, 1)}[kind]


def run_step(lib, col, w, added, pet, dt, share=None, pulse=None):
    m, L, N = col._maps, col.n_layers, col.n_cells
    out = dict(w=np.empty((L, N)), perc=np.empty((L - 1, N)), diff=np.empty((L - 1, N)), ret=np.empty(N), drain=np.empty(N), evap=np.empty(N), unmet=np.empty(N), rows=np.empty((4, N)))
    flat = lambda a: np.ascontiguousarray(a.reshape(-1))  # noqa: E731
    add = w if pulse is not None else added
    sh = pet if share is None else share
    bo = np.ascontiguousarray((m["bottom_open"] & m["active"]).view(np.uint8))
    ac = np.ascontiguousarray(m["active"].view(np.uint8))
    coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    tr, tf, ts, kif = (flat(m[n]) for n in ("theta_residual", "theta_field_capacity", "theta_saturation", "interface"))
    st = {n: strides(col.layouts[n], N) for n in ("theta_residual", "theta_field_capacity", "theta_saturation", "interface")}
    lib.host_step(ctypes.c_longlong(N), L, P(w), P(add), P(pet), P(ac), P(bo), P(m["edges"]), P(m["soil_depth"]), P(m["sep_nominal"]),
                  P(tr), ctypes.c_longlong(st["theta_residual"][0]), ctypes.c_longlong(st["theta_residual"][1]), P(tf), ctypes.c_longlong(st["theta_field_capacity"][0]), ctypes.c_longlong(st["theta_field_capacity"][1]),
                  P(ts), ctypes.c_longlong(st["theta_saturation"][0]), ctypes.c_longlong(st["theta_saturation"][1]), P(kif), ctypes.c_longlong(st["interface"][0]), ctypes.c_longlong(st["interface"][1]),
                  ctypes.c_double(col.diffusion_coefficient or 0.0), 0 if col.diffusion_mode == "constant" else 1, ctypes.c_double(coef), ctypes.c_double(dt), ctypes.c_double(col.theta_evaporation_stress),
                  ctypes.c_double(col.evaporation_factor), 1, ctypes.c_double(ss.cell_closure_coefficient(L, pulse is not None)), ctypes.c_double(ss.NO_FLAG), P(sh), 0 if share is None else 1,
                  ctypes.c_double(0.0 if pulse is None else pulse), 0 if pulse is None else 1, P(out["w"]), P(out["perc"]), P(out["diff"]), P(out["ret"]), P(out["drain"]), P(out["evap"]), P(out["unmet"]), P(out["rows"]))
    return out


@pytest.mark.parametrize("N", [1, 127, 128, 129, 300])
@pytest.mark.parametrize("mode,bottom,layout", [("tietjen_donor_depth", "open", "cell"), ("constant", "closed", "shared"), ("tietjen_donor_depth", "open", "shared")])
def test_step_source_equals_the_numpy_reference(lib, N, mode, bottom, layout):
    rng, col, w, act, _ = case(N, 7 + N, mode, bottom, layout)
    added = rng.uniform(0, 3e-3, (3, N)) * act * 0.2
    pet = rng.uniform(0, 3e-7, N) * act
    ref = fl.floating_step(col, fl.floating_continued_state(col, w), 3600.0, added_water_m=added, potential_evaporation_m_s=pet)
    out = run_step(lib, col, w, added, pet, 3600.0)
    inact = ~act
    assert np.array_equal(out["w"][:, act], ref.state.water_m[:, act]) or np.allclose(out["w"][:, act], ref.state.water_m[:, act], rtol=2e-12, atol=1e-14)
    assert np.array_equal(out["w"][:, inact], w[:, inact])  # placeholders preserved bit for bit
    for k, name in (("drain", "drainage_m"), ("ret", "surface_return_m"), ("evap", "evaporation_m"), ("unmet", "evaporation_unmet_m")):
        np.testing.assert_allclose(out[k], getattr(ref, name), rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(out["perc"], ref.percolation_m, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(out["diff"], ref.diffusion_m, rtol=2e-12, atol=1e-14)
    assert (out["rows"][3] == ss.NO_FLAG).all()  # no failure flag anywhere


def test_step_source_flags_pulse_and_share(lib):
    rng, col, w, act, _ = case(40, 3, "tietjen_donor_depth", "open", "cell")
    pet = np.zeros(40)
    share = rng.uniform(0, 1, 40)
    ref = fl.floating_step(col, fl.floating_continued_state(col, w), 86400.0, added_water_m=np.zeros((3, 40)), potential_evaporation_m_s=np.full(40, 2e-7) * act, evaporation_share=share)
    out = run_step(lib, col, w, np.zeros((3, 40)), np.full(40, 2e-7) * act, 86400.0, share=share)
    np.testing.assert_allclose(out["w"][:, act], ref.state.water_m[:, act], rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(out["evap"], ref.evaporation_m, rtol=2e-12, atol=1e-14)
    out = run_step(lib, col, w, np.zeros((3, 40)), pet, 86400.0, pulse=1e-5)
    assert (out["rows"][3] == ss.NO_FLAG).all() or (out["rows"][3][act] >= 0).all()
    bad = w.copy()
    bad[0, np.flatnonzero(act)[0]] = -1.0
    flagged = run_step(lib, col, bad, np.zeros((3, 40)), pet, 600.0)
    assert flagged["rows"][3][np.flatnonzero(act)[0]] == 2.0  # bit 1: negative water


@pytest.mark.parametrize("N", [1, 128, 257])
def test_reconcile_source_equals_the_reference(lib, N):
    rng, col, w, act, kind = case(N, 21 + N, "constant", "open", "cell")
    new_rng = np.random.default_rng(99 + N)
    tf = new_rng.uniform(0.05, 0.15, (3, N)); ts = tf + new_rng.uniform(0.02, 0.1, (3, N))
    new = fl.prepare_floating_column(EDGES, theta_residual=np.zeros((3, N)), theta_field_capacity=tf, theta_saturation=ts, diffusivity_m2_s=np.zeros(2), theta_evaporation_stress=0.3,
                                     bottom_drainage="open", soil_depth_m=np.where(rng.random(N) > 0.5, 0.2, 0.9), active_mask=act, cell_area_m2=2.0)
    old_w = np.minimum(w, 0.9 * np.stack([fl.porous_thickness(col, k) for k in range(3)]) * 0.35)
    old_w[:, ~act] = 5.0
    st = fl.floating_continued_state(col, old_w)
    ref_w, ref_perc, ref_surf, ref_exp = fl._reconcile_numpy(new, st.water_m)
    m = new._maps
    out_w, perc, surf, exp = np.empty((3, N)), np.zeros((2, N)), np.empty(N), np.empty(N)
    flat = lambda a: np.ascontiguousarray(a.reshape(-1))  # noqa: E731
    tf_f, ts_f = flat(m["theta_field_capacity"]), flat(m["theta_saturation"])
    tr_f = flat(m["theta_residual"])
    lib.host_reconcile(ctypes.c_longlong(N), 3, P(st.water_m), P(np.ascontiguousarray(m["active"].view(np.uint8))), P(np.ascontiguousarray((m["bottom_open"] & m["active"]).view(np.uint8))), P(m["edges"]),
                       P(m["soil_depth"]), P(m["sep_nominal"]), P(tr_f), ll(N), ll(1), P(tf_f), ll(N), ll(1), P(ts_f), ll(N), ll(1), ctypes.c_double(volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)), P(out_w), P(perc), P(surf), P(exp))
    assert np.array_equal(out_w, ref_w) and np.array_equal(perc, ref_perc) and np.array_equal(surf, ref_surf) and np.array_equal(exp, ref_exp)  # no library call: bitwise


def test_source_is_rendered_from_the_one_shared_body_and_the_common_source_is_untouched():
    import hashlib

    from maple_syrup import subsurface_cuda as old
    from maple_syrup import subsurface_cuda_spatial as spatial

    src = cf.render_step_source()
    assert old._BODY in src and "fma(" not in src  # the shared body text is embedded whole: no second copy of the stage equations
    assert hashlib.sha256(old.kernel_source().encode()).hexdigest() == hashlib.sha256(old.render_source("common").encode()).hexdigest()  # the common render is unchanged by this module
    assert "maple_syrup_floating_step" not in old.kernel_source() and "maple_syrup_floating_step" not in spatial.render_source()
    p = cf.provenance()
    assert p["shared_body_sha256"] == hashlib.sha256(old._BODY.encode()).hexdigest() and p["source_sha256"] == hashlib.sha256(cf.render_source().encode()).hexdigest()
    assert "[MAX_LAYERS" not in cf.render_source() and "double hh[" not in cf.render_source()  # no fixed layer bound and no per-thread layer arrays


def test_the_renderer_fails_closed_when_an_anchor_is_missing_or_duplicated(monkeypatch):
    from maple_syrup import subsurface_cuda as old

    for broken in (old._BODY.replace(cf._GRAV_BEGIN, "// removed\n"), old._BODY + cf._SAT_BEGIN):
        monkeypatch.setattr(old, "_BODY", broken)
        with pytest.raises(cf.FloatingSourceError, match="exactly once"):
            cf.render_reconcile_source()
    monkeypatch.undo()
    cf.render_reconcile_source()  # the genuine body renders


def test_reconcile_source_reuses_the_gravity_and_saturation_slices_verbatim():
    from maple_syrup import subsurface_cuda as old

    grav, sat = cf._slices()
    assert grav in old._BODY and sat in old._BODY and cf.render_reconcile_source().count(grav) == 1 and cf.render_reconcile_source().count(sat) == 1
    assert cf.render_reconcile_source().index(grav) < cf.render_reconcile_source().index(sat)  # gravity THEN saturation return (the reverse of the step)
