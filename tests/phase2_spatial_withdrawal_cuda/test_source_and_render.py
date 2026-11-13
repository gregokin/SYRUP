"""Spatial CUDA withdrawal SOURCE structure (host checks, no GPU/CuPy/NVRTC): one shared body rendered through the pure renderer, common CUDA text/options byte-exact, the NEW spatial module compiled WITHOUT fusion
(`--fmad=false`), barrier safety of the wrappers, no 8N rows / CuPy / CUB reductions, cold import without CuPy, bounded kernel cache. Author-written."""
from __future__ import annotations

import hashlib
import inspect
import re
import subprocess
import sys

import pytest
from conftest import hostgpu, ss, st, tc

COMMON_CUDA_SHA = "4bc236cf88100672278dfcca74de252251a3c71e42672ba2eaa457e486bd30c7"  # the B1.a-accepted common withdrawal source


def test_common_cuda_source_and_options_are_byte_exact_and_not_broadened():
    assert hashlib.sha256(st._CUDA_SRC.encode()).hexdigest() == COMMON_CUDA_SHA
    src = inspect.getsource(st)
    assert "options=('-std=c++11',)" in src and "--fmad" not in src  # the common kernel keeps its own options; fmad is NOT silently broadened to it


def test_spatial_options_forbid_fusion_and_the_body_is_the_one_shared_text():
    assert tuple(tc.COMPILE_OPTIONS) == ("--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")
    text = tc.render_source()
    assert text.count(st._CUDA_BODY) == 1 and text == st.render_withdrawal_source(tc.SPATIAL_READERS, tc.SPATIAL_HEAD, tc.SPATIAL_TAIL) == tc.kernel_source()
    for marker in ("if (!(wv >= 0.0 && wv <= WITHDRAW_SATB(l))) flags |= 1;", "scale[i] = (qv > 0.0) ? (t / qv) : 0.0;", "double positive_inf = "):
        assert text.count(marker) <= 1 or marker.startswith("double positive_inf"), marker  # no second copy of the withdrawal law (flags, min, unmet, scale) in the rendered source
    # the only repeated statement is the two-line AVAILABILITY definition `a = wv - res; if (a < 0) a = 0` in the pure-read kernel (availability is a different operation than the withdrawal)
    assert text.count("double a = wv - WITHDRAW_RESIDUAL(l);") == 2 and tc.SPATIAL_TAIL.count("double a = wv - WITHDRAW_RESIDUAL(l);") == 1 and text.count("double t = (qv < a) ? qv : a;") == 1
    prov = tc.kernel_provenance()
    assert prov["source_sha256"] == hashlib.sha256(text.encode()).hexdigest() and prov["shared_body_sha256"] == hashlib.sha256(st._CUDA_BODY.encode()).hexdigest()
    assert prov["packet_bytes"] == 64 and prov["fastmath"] is False


def test_the_raw_products_are_named_rounded_values_in_the_documented_order():
    r = tc.SPATIAL_READERS
    assert "const double res = theta_res * dz;" in r and "const double sat = theta_sat * dz;" in r and "return sat + coef * sat;" in r
    assert "#define WITHDRAW_RESIDUAL(l) res_at(" in r and "#define WITHDRAW_SATB(l) satb_at(" in r
    assert "fma(" not in r.lower() and "__fma" not in r.lower()  # no explicit fused intrinsic either


def test_wrapper_kernels_reach_every_barrier_and_use_no_library_reduction():
    tail = re.sub(r"/\*.*?\*/", "", tc.SPATIAL_TAIL, flags=re.S)
    for name in (tc.WITHDRAW_KERNEL, tc.FINAL_KERNEL):
        body = tail[tail.index(f"void {name}("):]
        body = body[: body.index("\n}\n") + 3]
        assert re.search(r"\breturn\b", body) is None and "__syncthreads()" in body, name  # no early return before a barrier
    avail = tail[tail.index(f"void {tc.AVAIL_KERNEL}("):]
    assert "atomicMax" in avail
    src = inspect.getsource(tc)
    for forbidden in (".sum(", "cub", "cupy.cub", ".max(", ".min(", "rows = ", "(8 * N"):
        assert forbidden not in src.replace("`", ""), forbidden
    assert tc.operation_terms("withdraw", 5, 1000)["host"] == 64  # ONE host array: the packet that to_host returns


def test_the_body_reaches_the_neutral_lane_values_and_the_reduction_ops_per_row():
    t = tc.SPATIAL_TAIL
    assert "double contrib[8] = {0.0, 0.0, 0.0, 0.0, 0.0, positive_inf, 0.0, 0.0}" in t  # sums 0, min +inf, flags 0, max request 0
    assert t.count("sh[5][tid] = (sh[5][tid + s] < sh[5][tid])") == 2 and \
         "sh[6][tid + s] > sh[6][tid]" in t and "sh[7][tid + s] > sh[7][tid]" in t


def test_cold_import_touches_no_cupy_and_compiles_nothing():
    code = "import sys; import maple_syrup.spatial_transpiration as s; import maple_syrup.spatial_transpiration_cuda as c; assert 'cupy' not in sys.modules and c._FUNCTIONS == {}; assert c.render_source(); print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_load_kernels_compiles_once_per_device_with_the_spatial_options_and_fails_typed():
    made = []

    class Fn:
        def __init__(self, name, threads=1024):
            self.name, self.attributes = name, {"max_threads_per_block": threads}

    class Cp:
        class RawModule:
            def __init__(self, code, options, backend):
                made.append((code, tuple(options), backend))

            def get_function(self, name):
                return Fn(name)

    with tc._LOCK:
        saved = dict(tc._FUNCTIONS)
        tc._FUNCTIONS.clear()
    try:
        a, b = tc.load_kernels(Cp, 0), tc.load_kernels(Cp, 0)
        assert a is b and len(made) == 1 and made[0] == (tc.render_source(), tc.COMPILE_OPTIONS, "nvrtc")
        tc.load_kernels(Cp, 1)
        assert set(tc._FUNCTIONS) == {0, 1}  # functions only, one entry per device
        tc._FUNCTIONS.clear()

        class Bad:
            class RawModule:
                def __init__(self, *a, **k):
                    raise RuntimeError("nvrtc boom")

        class Slim(Cp):
            class RawModule:
                def __init__(self, *a, **k):
                    pass

                def get_function(self, name):
                    return Fn(name, 64)

        with pytest.raises(ss.SubsurfaceBackendUnavailableError, match="failed to compile/load"):
            tc.load_kernels(Bad, 0)
        with pytest.raises(ss.SubsurfaceBackendUnavailableError, match="fewer than"):
            tc.load_kernels(Slim, 0)
        assert tc._FUNCTIONS == {}
    finally:
        with tc._LOCK:
            tc._FUNCTIONS.clear()
            tc._FUNCTIONS.update(saved)


def test_launch_ranges_are_validated_before_any_cast():
    with pytest.raises(ss.SubsurfaceError, match="grid limit"):
        tc.check_launch_ranges(3, 128 * (2**31 - 1) + 1)
    with pytest.raises(ss.SubsurfaceError, match="32-bit"):
        tc.check_launch_ranges(2**31, 10)
