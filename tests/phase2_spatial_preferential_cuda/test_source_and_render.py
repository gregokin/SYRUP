"""Source/render contracts of the spatial CUDA preferential: the common kernel text, options, name, ABI and provenance are BYTE-identical to the pre-factoring kernel, the render is pure and fails closed, the
spatial source shares the ONE body and carries the structural properties the design requires, the cold-reducer key order and the raw packet reconstruction are exact, and the derivation hook keeps the default
path expression for expression. CPU only; pure text/arithmetic checks. Self-contained: the pre-factoring digest and length are constants of this suite. Author-written."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
from conftest import CANDIDATE, pc, pi, sp, tc

COMMON_SHA = "0d454f1ad0fc4c6d23252687c311802788e3ee2a411f714969a2a08e9c9f72e2"
PRE = CANDIDATE.parent / "s2_b2_d_pre_sources" / "preferential_infiltration_cuda.py"


def test_the_common_text_options_abi_and_provenance_are_byte_identical_to_the_pre_factoring_kernel():
    assert hashlib.sha256(pc._SOURCE.encode()).hexdigest() == pc.COMMON_SOURCE_SHA256 == COMMON_SHA and len(pc._SOURCE) == 3457
    assert pc.kernel_source() == pc._SOURCE == pc.render_source()
    assert pc.COMPILE_OPTIONS == ("--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false") and pc.KERNEL_NAME == "maple_syrup_preferential_step"
    assert (pc.THREADS, pc.LAUNCHES_PER_STEP) == (128, 1)
    assert pc.kernel_provenance()["source_sha256"] == COMMON_SHA
    assert 'extern "C" __global__ void maple_syrup_preferential_step(' in pc._SOURCE and "const double* rate, const double* max_rate, const double* recv," in pc._SOURCE  # the launch ABI


@pytest.mark.skipif(not PRE.exists(), reason="pre-edit source is a task-only artefact and is not distributed with this checkout")
def test_the_common_rendering_equals_the_task_archived_pre_edit_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("old_pref_cuda_b2d", PRE)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    assert old._SOURCE == pc._SOURCE and old.kernel_provenance() == pc.kernel_provenance() and old.COMPILE_OPTIONS == pc.COMPILE_OPTIONS


def sentinels():
    return {t: f"<<{t.strip('@')}>>" for t in pc.BODY_TOKENS}


def test_the_render_is_pure_and_deterministic():
    assert tc.render_source() == tc.render_source() and tc.kernel_source() == tc.render_source()


def test_the_render_fails_closed_on_every_contract_drift(monkeypatch):
    base = dict(pc.COMMON_SUBSTITUTIONS)
    with pytest.raises(pi.PreferentialError):
        pc.render_source(substitutions={k: v for k, v in base.items() if k != "@@RATE@@"})
    with pytest.raises(pi.PreferentialError):
        pc.render_source(substitutions={**base, "@@EXTRA@@": "x"})
    with pytest.raises(pi.PreferentialError):
        pc.render_source(substitutions={**base, "@@RATE@@": 3})
    with pytest.raises(pi.PreferentialError, match="token marker"):
        pc.render_source(substitutions={**base, "@@RATE@@": "rate[i] + @@MAXRATE@@"})
    for bad in (dict(readers=1), dict(head=2), dict(tail=3)):
        with pytest.raises(pi.PreferentialError):
            pc.render_source(**bad)
    body = pc._BODY
    monkeypatch.setattr(pc, "_BODY", body.replace("@@SAT_L@@", "sat_m[l]"))
    with pytest.raises(pi.PreferentialError, match="occurs 0 times"):
        pc.render_source()
    monkeypatch.setattr(pc, "_BODY", body + "\n    double extra = @@RATE@@;\n")
    with pytest.raises(pi.PreferentialError, match="occurs 2 times"):
        pc.render_source()
    monkeypatch.setattr(pc, "_BODY", body + "\n    // @@UNKNOWN@@\n")
    with pytest.raises(pi.PreferentialError, match="unreplaced token"):
        pc.render_source()


def test_every_declared_token_site_is_present_exactly_as_declared():
    for token, count in pc.BODY_TOKENS.items():
        assert pc._BODY.count(token) == count, token
    assert set(pc.COMMON_SUBSTITUTIONS) == set(pc.BODY_TOKENS) == set(tc.SPATIAL_SUBSTITUTIONS)
    assert "@@" not in pc._COMMON_HEAD


def test_the_spatial_source_shares_the_one_body_and_reads_nothing_from_the_common_maps():
    common, spatial = pc.render_source(substitutions=sentinels()), pc.render_source(readers=tc.SPATIAL_READERS, head=tc.SPATIAL_HEAD, substitutions=sentinels(), tail=tc.SPATIAL_TAIL)
    body = pc._BODY
    for token in pc.BODY_TOKENS:
        body = body.replace(token, f"<<{token.strip('@')}>>")
    assert body in common and body in spatial
    real = tc.render_source()
    for text in tc.SPATIAL_SUBSTITUTIONS.values():
        assert text in real
    assert "sat_m[" not in real and "cap_tol[" not in real and "rows[" not in real and "maple_syrup_preferential_step" not in real


def test_the_spatial_source_structure():
    src = tc.render_source()
    step = src[src.index('extern "C" __global__ void maple_syrup_spatial_pref_step'):src.index('extern "C" __global__ void maple_syrup_spatial_pref_finalize')]
    final = src[src.index('extern "C" __global__ void maple_syrup_spatial_pref_finalize'):src.index('extern "C" __global__ void maple_syrup_spatial_pref_reduce')]
    reduce_k = src[src.index('extern "C" __global__ void maple_syrup_spatial_pref_reduce'):src.index('extern "C" __global__ void maple_syrup_spatial_pref_validate')]
    for k in (step, final, reduce_k):
        assert "return" not in k  # every lane of a block reaches every barrier: no early return outside the __device__ function
    assert step.count("__syncthreads()") == 2 and final.count("__syncthreads()") == 2 and reduce_k.count("__syncthreads()") == 2
    assert "__shared__ double sh[7][BLOCK];" in step and "__shared__ double sh[7][256];" in final
    assert "contrib[" not in src and "double rows" not in src and "cub" not in src.lower() and "thrust" not in src.lower() and not re.search(r"double\s+\w+\[\s*(nl|L)\s*\]", src)
    for r, expr in enumerate(tc._ROW_NAMES):
        assert f"sh[{r}][tid] = {expr};" in src  # literal offsets
    assert "if (i >= n) return;" in src[:src.index('extern "C"')]  # the padded-lane guard lives in the __device__ function
    assert src.count("atomicOr(&packet[0]") == 1 and src.count("atomicAdd(&packet[1]") == 1 and src.count("*flag_out = f;") == 1
    assert "atomicMin(&diag[2]" in reduce_k and "atomicMax(&diag[3]" in reduce_k and "atomicAdd(&diag[0]" in reduce_k and "atomicAdd(&diag[1]" in reduce_k


def test_the_products_are_unfused_and_the_law_is_the_shared_body():
    assert tc.COMPILE_OPTIONS is pc.COMPILE_OPTIONS and "--fmad=false" in tc.COMPILE_OPTIONS
    r = tc.SPATIAL_READERS
    assert "const double sat = theta_sat * dz;" in r and "return sat + coef * sat;" in r
    body = pc._BODY
    for needle in ("const double x = @@RATE@@ * dt;", "const double fr = -expm1(-x);", "const double cap = @@MAXRATE@@ * dt;", "if (cap < attempted) attempted = cap;", "double alloc = attempted * recv[p];",
                   "double room = @@SAT_L@@ - w_out[p];", "if (room < alloc) alloc = room;", "if (left < alloc) alloc = left;"):
        assert needle in body, needle  # ONE aggregate continuous path, ONE cap, top layer first, bounded by room and the remaining surface, no remapping loop


def test_raw_packet_layout_and_reconstruction():
    assert tc.PACKET_WORDS == 9 and tc.RAW_PACKET_BYTES == 72 and pi.PACKET_LENGTH == 9 and tc.ROWS == 7
    sums = np.arange(1, 8, dtype=np.float64) * 1.25
    raw = np.zeros(9, dtype=np.uint64)
    raw[2:] = sums.view(np.uint64)
    p = tc.build_packet(raw)
    assert p.shape == (9,) and p.dtype == np.float64 and (p[:7] == sums).all() and p[7] == pi.NO_FLAG and p[8] == 0.0
    raw[0], raw[1] = (1 << 3) | (1 << 8), 4
    p = tc.build_packet(raw)
    assert p[7] == float(1 << 3) and p[8] == 4.0 and p[7] < pi.NO_FLAG
    for bad in (np.zeros(9, dtype=np.float64), np.zeros(8, dtype=np.uint64), np.zeros((9, 1), dtype=np.uint64)):
        with pytest.raises(pi.PreferentialError):
            tc.build_packet(bad)


def test_operation_terms_and_launch_ranges_are_exact():
    t = tc.operation_terms("step", 3, 1000)
    nb = (1000 + 127) // 128
    assert t["outputs"] == (2 * 1000 + 2 * 3000) * 8 and t["scratch"] == 7 * 8 * nb + 72 and t["host"] == 144 and t["n_blocks"] == nb
    assert t["shared"] == 7 * 8 * 128 + 8 * 128 + 4 * 128 and t["shared_finalize"] == 7 * 8 * 256
    p = tc.operation_terms("prepare", 3, 1000)
    assert p["outputs"] == (3000 + 2000) * 8 and p["scratch"] == 4 * 1000 * 8 + 3 * 1000 + 64 and p["host"] == 64 + 3000 * 8
    assert p["outputs"] + p["scratch"] == (3 * 1000 + 6 * 1000) * 8 + 3 * 1000 + 64
    assert tc.operation_terms("prepare", 8192, 1)["host"] == 64 + 8192 * 8  # N = 1, large L: the staging buffer is bounded by the map size
    with pytest.raises(Exception, match="unknown CUDA operation"):
        tc.operation_terms("nope", 1, 1)
    for L, N in ((0, 5), (2**31, 5), (1, 0), (1, 128 * (2**31 - 1) + 1)):
        with pytest.raises(Exception):
            tc.check_launch_ranges(L, N)
    tc.check_launch_ranges(2**31 - 1, 128 * (2**31 - 1))


def test_the_order_preserving_key_decodes_every_double_and_orders_like_the_doubles():
    vals = np.array([-np.inf, -1e300, -1.0, -5e-324, -0.0, 0.0, 5e-324, 1e-300, 1.0, 1e300, np.inf])
    bits_ = vals.view(np.uint64)
    keys = [int(b) ^ 0 for b in bits_]
    key = lambda b: (~b & tc._MAX_KEY) if (b >> 63) else (b | tc._SIGN)  # noqa: E731
    ks = [key(int(b)) for b in bits_]
    assert ks == sorted(ks) and len(set(ks)) == len(ks)  # strictly increasing, -0.0 < +0.0
    for v, k in zip(vals, ks, strict=True):
        got = tc._decode_key(k)
        assert got == v and np.signbit(got) == np.signbit(v)
    assert keys  # (keep the raw bit patterns referenced)


# ---- the derivation hook (the 4th file) ---------------------------------------------------------------------------------------------------
def test_the_default_derivation_executes_the_same_reductions_and_unknown_kinds_fail_closed():
    ctx = pi._DeriveContext(np, 1, 3, np.ones(3, bool), lambda s, c: None)
    a = np.array([1.0, np.nan, np.inf])
    assert pi._reduce(ctx, "count_nonfinite", a) == 2.0 and pi._reduce(ctx, "count_nan", a) == 1.0 and np.isnan(pi._reduce(ctx, "min", a)) and np.isnan(pi._reduce(ctx, "max", a))
    b = np.array([-0.0, 0.0, 2.0])
    assert pi._reduce(ctx, "min", b) == 0.0 and pi._reduce(ctx, "max", b) == 2.0
    for kind in ("count", "sum", "isnan", "", "COUNT_NAN", None):
        with pytest.raises(pi.PreferentialError, match="unknown cold reduction kind"):
            pi._reduce(ctx, kind, a)
    hooked = pi._DeriveContext(np, 1, 3, np.ones(3, bool), lambda s, c: None, lambda kind, arr: 41.0)
    assert pi._reduce(hooked, "min", a) == 41.0
    with pytest.raises(pi.PreferentialError):
        pi._reduce(hooked, "bogus", a)  # the hook is never reached for an unknown kind


def test_the_derivation_routes_exactly_its_eight_reductions_through_the_hook():
    import inspect

    src = inspect.getsource(pi._derive_pathway)
    assert src.count("_reduce(ctx,") == 8 and "to_float" not in src and "xp.sum" not in src and "xp.min" not in src and "xp.max" not in src
    assert [m for m in re.findall(r'_reduce\(ctx, "(\w+)"', src)] == ["count_nonfinite", "min", "count_nonfinite", "max", "min", "max", "count_nonfinite", "count_nan"]
    assert pi.REDUCE_KINDS == ("count_nonfinite", "min", "max", "count_nan") and pi._DeriveContext._fields[-1] == "reduce" and pi._DeriveContext._field_defaults == {"reduce": None}
