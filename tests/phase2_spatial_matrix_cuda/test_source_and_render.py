"""Source/render contracts of the spatial CUDA matrix: the common kernel text and options are BYTE-identical to the pre-factoring text, the render is pure and fails closed, the spatial source shares the ONE
Newton helper text and the ONE body, and carries the structural properties the design requires (no dense rows, no per-thread contribution array, every lane reaches every barrier, literal shared offsets, unfused
products). CPU only; pure text checks. Author-written."""
from __future__ import annotations

import hashlib
import importlib.util
import re

import numpy as np
import pytest
from conftest import TASK, mc, mi, tc

PRE = TASK / "s2_b2_c_pre_sources" / "matrix_infiltration_cuda.py"


def _old_module():
    spec = importlib.util.spec_from_file_location("old_matrix_cuda_b2c", PRE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NEW_SHA = "b9e131308266af692f9a40b3edb09a21df349c67406b6b6927820e1607b4f3d2"  # the CURRENT common text (S2-B2.c review02 bounded combined intake)
OLD_SHA = "b58178dfc303f48364f7bfdcd744a08bf821da481ae61d8f347981c3bee5c05f"  # the HISTORICAL text (before the B2.c factoring and before review02): evidence only, never rebound


@pytest.mark.skipif(not PRE.exists(), reason="pre-edit source not present")
def test_the_common_text_options_abi_and_provenance_differ_from_the_historical_kernel_only_by_the_approved_arithmetic():
    old = _old_module()
    assert hashlib.sha256(old._SOURCE.encode()).hexdigest() == mc.HISTORICAL_COMMON_SOURCE_SHA256 == OLD_SHA  # the historical digest is preserved as evidence
    assert hashlib.sha256(mc._SOURCE.encode()).hexdigest() == mc.COMMON_SOURCE_SHA256 == NEW_SHA and mc._SOURCE != old._SOURCE
    assert mc.kernel_source() == mc._SOURCE == mc.render_source()  # the default rendering IS the current common text
    assert mc.COMPILE_OPTIONS == old.COMPILE_OPTIONS == ("--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")
    assert (mc.KERNEL_NAME, mc.THREADS, mc.LAUNCHES_PER_STEP) == (old.KERNEL_NAME, old.THREADS, old.LAUNCHES_PER_STEP)  # name, block size and launch ABI unchanged
    pn, po = mc.kernel_provenance(), old.kernel_provenance()
    assert {k: v for k, v in pn.items() if k not in ("source_sha256", "historical_source_sha256", "numerical_revision")} == {k: v for k, v in po.items() if k != "source_sha256"}
    assert pn["source_sha256"] == NEW_SHA and po["source_sha256"] == OLD_SHA and pn["historical_source_sha256"] == OLD_SHA and "bounded" in pn["numerical_revision"]
    # the ONLY textual difference from the historical kernel is the combined-intake block (and nothing in the Newton helpers, the signature or the flag logic)
    import difflib

    changed = [ln[1:].strip() for ln in difflib.unified_diff(old._SOURCE.splitlines(), mc._SOURCE.splitlines(), lineterm="", n=0) if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
    markers = ("adv", "j_o", "intake", "reduced", "run_post", "room", "ONE combined", "re-derived", "share the SAME", "byte-identical", "}", "else", "Inactive bound", "/*", "*/")
    assert changed and all(any(m in ln for m in markers) or ln == "" for ln in changed), changed
    head_end = old._SOURCE.index("{\n", old._SOURCE.index('extern "C" __global__ void maple_syrup_matrix_step(')) + 2
    assert mc._SOURCE.startswith(old._SOURCE[:head_end])  # the Newton helpers and the kernel head/signature (up to the opening brace) are byte-identical to the historical text


def sentinels():
    return {t: f"<<{t.strip('@')}>>" for t in mc.BODY_TOKENS}


def test_the_render_is_pure_and_deterministic():
    a, b = tc.render_source(), tc.render_source()
    assert a == b and mc.render_source() == mc.render_source()
    assert tc.kernel_source() == a


def test_the_render_fails_closed_on_every_contract_drift(monkeypatch):
    base = dict(mc.COMMON_SUBSTITUTIONS)
    with pytest.raises(mi.MatrixError):
        mc.render_source(substitutions={k: v for k, v in base.items() if k != "@@K@@"})  # a token is not supplied
    with pytest.raises(mi.MatrixError):
        mc.render_source(substitutions={**base, "@@EXTRA@@": "x"})  # an unknown token
    with pytest.raises(mi.MatrixError):
        mc.render_source(substitutions={**base, "@@K@@": 3})  # not a string
    with pytest.raises(mi.MatrixError, match="token marker"):
        mc.render_source(substitutions={**base, "@@K@@": "kmap[i] + @@PSI@@"})  # a substitution that re-introduces a token (would be double-substituted)
    for bad in (dict(readers=1), dict(head=2), dict(tail=3)):
        with pytest.raises(mi.MatrixError):
            mc.render_source(**bad)
    body = mc._BODY
    monkeypatch.setattr(mc, "_BODY", body.replace("@@FC0@@", "fc_m[0]"))  # a read site silently hard-coded in the body
    with pytest.raises(mi.MatrixError, match="occurs 0 times"):
        mc.render_source()
    monkeypatch.setattr(mc, "_BODY", body + "\n    double extra = @@K@@;\n")  # a duplicated site
    with pytest.raises(mi.MatrixError, match="occurs 2 times"):
        mc.render_source()
    monkeypatch.setattr(mc, "_BODY", body + "\n    // @@UNKNOWN@@\n")  # an undeclared token
    with pytest.raises(mi.MatrixError, match="unreplaced token"):
        mc.render_source()


def test_every_declared_token_site_is_present_exactly_as_declared():
    for token, count in mc.BODY_TOKENS.items():
        assert mc._BODY.count(token) == count, token
    assert set(mc.COMMON_SUBSTITUTIONS) == set(mc.BODY_TOKENS) == set(tc.SPATIAL_SUBSTITUTIONS)
    assert "@@" not in mc._NEWTON and "@@" not in mc._COMMON_HEAD


def test_the_spatial_source_shares_the_one_newton_text_and_the_one_body():
    common, spatial = mc.render_source(substitutions=sentinels()), mc.render_source(readers=tc.SPATIAL_READERS, head=tc.SPATIAL_HEAD, substitutions=sentinels(), tail=tc.SPATIAL_TAIL)
    newton = mc._NEWTON.replace("__LOG1PMX_TERMS__", str(mi.LOG1PMX_TERMS))
    for src in (common, spatial, tc.render_source(), mc._SOURCE):
        assert src.startswith(newton) and src.count("double log1pmx(") == 1 and src.count("double ga_increment(") == 1
    body = mc._BODY
    for token in mc.BODY_TOKENS:
        body = body.replace(token, f"<<{token.strip('@')}>>")
    assert body in common and body in spatial  # the sentinel-rendered body is the SAME text in both renderings
    # the real spatial source differs from the common one ONLY at the designated sites, the head, the readers and the tail
    real = tc.render_source()
    for token, text in tc.SPATIAL_SUBSTITUTIONS.items():
        assert text in real
    assert "sat_m[" not in real and "cap_tol[" not in real and "fc_m[" not in real and "kmap[" not in real and "rows[" not in real  # no common read left in the spatial text
    assert "maple_syrup_matrix_step" not in real


def test_the_spatial_source_structure():
    src = tc.render_source()
    step = src[src.index('extern "C" __global__ void maple_syrup_spatial_matrix_step'):src.index('extern "C" __global__ void maple_syrup_spatial_matrix_finalize')]
    final = src[src.index('extern "C" __global__ void maple_syrup_spatial_matrix_finalize'):src.index('extern "C" __global__ void maple_syrup_spatial_matrix_validate')]
    assert "return" not in step and "return" not in final  # every lane of a block reaches every barrier: no early return outside the device function
    assert step.count("__syncthreads()") == 2 and final.count("__syncthreads()") == 2
    assert "__shared__ double sh[15][BLOCK];" in step and "__shared__ double sh[15][256];" in final
    assert "double contrib[" not in src and "contrib[" not in src  # no per-thread contribution array
    assert "double rows" not in src and "rows[" not in src and "cub" not in src.lower() and "thrust" not in src.lower()
    for r, name in enumerate(tc._ROW_NAMES):
        assert f"sh[{r}][tid] = {name};" in src  # literal offsets, no dynamic row index in the epilogue
        assert f"sh[{r}][tid] = 0.0;" in src
    assert "if (i >= n) return;" in src[:src.index('extern "C"')]  # the padded-lane guard is inside the __device__ function
    assert src.count("atomicOr(&packet[0]") == 1 and src.count("atomicAdd(&packet[1]") == 1 and src.count("__double_as_longlong") == 1
    assert "(1u << 14)" not in src and src.count("*flag_out = f;") == 1
    assert "L]" not in re.sub(r"\[\s*L\s*\]", "[L]", "") and not re.search(r"double\s+\w+\[\s*(nl|L)\s*\]", src)  # never an L-sized per-thread array


def test_the_products_are_unfused_by_options_and_written_as_rounded_values():
    assert tc.COMPILE_OPTIONS is mc.COMPILE_OPTIONS and "--fmad=false" in tc.COMPILE_OPTIONS
    r = tc.SPATIAL_READERS
    assert "const double fc = theta_fc * dz;" in r and "const double sat = theta_sat * dz;" in r and "return sat + coef * sat;" in r
    assert tc.SPATIAL_SUBSTITUTIONS["@@SAT0@@"].startswith("sat_at(") and tc.SPATIAL_SUBSTITUTIONS["@@FC0@@"].startswith("fc_at(")
    # the deficit keeps its exact evaluation order in the shared body: fc0 - w0, clipped at 0, then psi * (deficit / dz0)
    body = mc._BODY
    assert "double deficit = @@FC0@@ - w0;" in body and "const double b = psi * (deficit / @@DZ0@@);" in body and "double room = @@SAT0@@ - w0;" in body


def test_raw_packet_layout_and_reconstruction():
    assert tc.PACKET_WORDS == 17 and tc.RAW_PACKET_BYTES == 136 and mi.PACKET_LENGTH == 17 and tc.ROWS == 15
    sums = np.arange(1, 16, dtype=np.float64) * 1.5
    raw = np.zeros(17, dtype=np.uint64)
    raw[2:] = sums.view(np.uint64)
    p = tc.build_packet(raw)
    assert p.shape == (17,) and p.dtype == np.float64 and (p[:15] == sums).all() and p[15] == mi.NO_FLAG and p[16] == 0.0  # no flag: the NO_FLAG sentinel
    raw[0], raw[1] = (1 << 4) | (1 << 9) | (1 << 17), 5
    p = tc.build_packet(raw)
    assert p[15] == float(1 << 4) and p[16] == 5.0 and p[15] < mi.NO_FLAG  # the LOWEST set bit of the OR = the minimum over cells of each cell's lowest bit
    for bad in (np.zeros(17, dtype=np.float64), np.zeros(16, dtype=np.uint64), np.zeros((17, 1), dtype=np.uint64)):
        with pytest.raises(mi.MatrixError):
            tc.build_packet(bad)


def test_the_lowest_bit_of_the_or_equals_the_minimum_over_cells():
    rng = np.random.default_rng(0)
    for _ in range(200):
        flags = [int(rng.integers(0, 1 << 23)) if rng.random() < 0.5 else 0 for _ in range(int(rng.integers(1, 30)))]
        per_cell_low = [f & -f for f in flags if f]
        union = 0
        for f in flags:
            union |= f
        assert (union & -union if union else 0) == (min(per_cell_low) if per_cell_low else 0)


def test_operation_terms_and_launch_ranges_are_exact():
    t = tc.operation_terms("step", 3, 1000)
    nb = (1000 + 127) // 128
    assert t["outputs"] == (4 * 1000 + 3 * 1000) * 8 and t["scratch"] == 15 * 8 * nb + 136 and t["host"] == 272 and t["n_blocks"] == nb
    assert t["shared"] == 15 * 8 * 128 + 8 * 128 + 4 * 128 and t["shared_finalize"] == 15 * 8 * 256
    assert tc.operation_terms("history", 3, 10)["outputs"] == 160 and tc.operation_terms("prepare", 3, 10)["outputs"] == 240
    assert tc.operation_terms("prepare", 3, 10 ** 6)["host"] == tc.STAGE_ELEMS * 8 + 8
    with pytest.raises(Exception, match="unknown CUDA operation"):
        tc.operation_terms("nope", 1, 1)
    for L, N in ((0, 5), (2**31, 5), (1, 0), (1, 128 * (2**31 - 1) + 1)):
        with pytest.raises(Exception):
            tc.check_launch_ranges(L, N)
    tc.check_launch_ranges(2**31 - 1, 128 * (2**31 - 1))
