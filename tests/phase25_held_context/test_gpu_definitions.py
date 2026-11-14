"""Actual-GPU test DEFINITIONS for the dynamic-vegetation CUDA held context /4 (WRITTEN, NOT RUN by their author: no GPU / CuPy / NVRTC / nvidia-smi was touched; collected and compiled on the host only). Opt-in
(`SYRUP_RELEASE_GPU=1`; device `SYRUP_HELD_CONTEXT_GPU_DEVICE`, default 0): without the flag they are skipped; WITH the flag a missing CuPy / device is a SETUP FAILURE, never a skip. Root runs them on its allocated device
with the actual-GPU MAPLE pin selected through `HELD_CONTEXT_MAPLE_ROOT` / `HELD_CONTEXT_MAPLE_PACKAGE` (see conftest).

Fixtures are the qualified C4 ones: the CUDA event<->daily bridge definitions' `matched_pair` (a genuinely completed CUDA closed Boundary, Dense and published Windowed, with a MATCHED CPU twin) through
`veg_lc_gpu_support`, the lifecycle's own `apply_vegetation_generation` (ACTUAL LEAF projection + S0 binding, device law), `FloatingPondVertical.from_boundary(new root)` and the strict export / parse / reconstruct /
`from_held_context` / strict re-export chain. No private minting, no fake class, no CPU numerical fallback; the CPU twin is a CONTROL for the law identity (g) only."""
from __future__ import annotations

import copy
import gc
import os
from types import SimpleNamespace

import numpy as np
import pytest
import veg_lc_builders as vb
import veg_lc_gpu_support as sup

from maple_syrup import floating_held_context as fh
from maple_syrup import floating_held_context_cuda as fhc
from maple_syrup import floating_held_context_vegetation as fhv
from maple_syrup import floating_held_context_vegetation_cuda as fvc
from maple_syrup import floating_pond as fp
from maple_syrup import vegetation_lifecycle as vl
from maple_syrup import water_vegetation_held as wh

RELEASE = os.environ.get("SYRUP_RELEASE_GPU") == "1"
DEVICE = int(os.environ.get("SYRUP_HELD_CONTEXT_GPU_DEVICE", "0"))
pytestmark = pytest.mark.skipif(not RELEASE, reason="CUDA hardware tests: set SYRUP_RELEASE_GPU=1 on the allocated device (authored, not run by the author)")
BIG = 1 << 40
HOSTQ = {"host_memory_budget_bytes": BIG, "host_available_bytes_fn": lambda: BIG, "host_reserve_bytes": 0}


@pytest.fixture(scope="module")
def gpu():
    try:
        import cupy as cp

        cp.cuda.Device(DEVICE).use()  # the ONE outer device selection; production APIs only enforce the current device afterwards
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"SYRUP_RELEASE_GPU=1 but CuPy / a CUDA device is unusable: {exc}")
    return SimpleNamespace(cp=cp, m=sup.modules())


def world(gpu, leaf, tmp_path, windowed):
    bd_ = gpu.m.bd
    g, c = bd_.matched_pair(tmp_path, bd_.dry_kwargs(), windowed)
    b = g.b
    s = vb.stack(leaf, b)
    binding = vb.binding(leaf, b, s)
    nb = vl.apply_vegetation_generation(g.lin, b, binding, memory_budget_bytes=BIG, available_bytes_fn=g.rdr, **{k: v for k, v in HOSTQ.items() if k != "host_reserve_bytes"})
    v = fp.FloatingPondVertical.from_boundary(nb.root, available_bytes_fn=g.rdr, memory_budget_bytes=BIG, **HOSTQ)
    rec = wh.capture_cached_inputs(binding, max_bytes=BIG)
    ident = s["proj"].identity_record()
    pins = fhv.ExpectedVegetationPins(rec.record_sha256, binding.sha256, ident["content_sha256"], vb.leaf_pin(), vb.helper_pin(), {k: ident["source"][k] for k in ("kind", "state_sha256", "catalog_hash")})
    return SimpleNamespace(g=g, c=c, b=b, nb=nb, v=v, rec=rec, pins=pins, binding=binding, stack=s)


def dev(w):
    return {"memory_budget_bytes": BIG, "available_bytes_fn": w.g.rdr, "reserve_bytes": 0}


def expected_of(r):
    i = r.meta()["identities"]
    return fh.ExpectedHeldIdentities(i["config_sha256"], i["root_column_sha256"], i["matrix_op_sha256"], i["pref_sha256"])


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_dynamic_cuda_chain_reproduces_the_held_device_identities_and_re_exports_strictly(gpu, leaf, tmp_path, windowed):
    w = world(gpu, leaf, tmp_path, windowed)
    t0 = w.v.held_template()
    assert t0.matrix_op.layouts["vegetation_factor"] == "cell"                                                  # the CUDA vertical ACTUALLY holds the law (lifecycle apply)
    r = fhv.export_held_context_vegetation(w.v, w.rec, w.pins, **dev(w), **HOSTQ)
    m = r.meta()
    assert m["schema"] == fhv.SCHEMA4 and m["backend"] == "cuda" and m["kernel"] == fhc.kernel_descriptor()
    assert m["identities"]["matrix_op_sha256"] == t0.matrix_op.sha256 and m["identities"]["pref_sha256"] == t0.preferential.sha256 and m["identities"]["root_column_sha256"] == t0.column.sha256
    parsed = fhv.record_from_parts_vegetation(copy.deepcopy(m), {k: np.array(a) for k, a in r.arrays().items()}, available_bytes_fn=lambda: BIG)
    ctx = fh.reconstruct_vegetation_held_context(parsed, expected_of(parsed), w.pins, **dev(w), **HOSTQ)
    assert ctx.backend == "cuda" and ctx.matrix_op.layouts["vegetation_factor"] == "cell" and ctx.vegetation.cached.record_sha256 == w.rec.record_sha256
    assert ctx.matrix_op.sha256 == t0.matrix_op.sha256 and ctx.preferential.sha256 == t0.preferential.sha256
    v1 = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=w.g.rdr, memory_budget_bytes=BIG, **HOSTQ)
    r2 = fhv.export_held_context_vegetation(v1, w.rec, w.pins, **dev(w), **HOSTQ)                             # STRICT re-export from the restored CUDA vertical
    assert r2.meta()["identities"] == m["identities"] and r2.meta()["vegetation"] == m["vegetation"]
    # CONTROL on the matched CPU twin: the same recipe gives the same g identity (g is formed once on the host); the backend-headed content ids differ by construction
    cb = w.c.b
    sc = vb.stack(leaf, cb)
    nbc = vl.apply_vegetation_generation(w.c.lin, cb, vb.binding(leaf, cb, sc), memory_budget_bytes=BIG, available_bytes_fn=w.c.rdr)
    vc = fp.FloatingPondVertical.from_boundary(nbc.root, available_bytes_fn=w.c.rdr)
    recc = wh.capture_cached_inputs(vb.binding(leaf, cb, sc), max_bytes=BIG)
    ident = sc["proj"].identity_record()
    pinsc = fhv.ExpectedVegetationPins(recc.record_sha256, vb.binding(leaf, cb, sc).sha256, ident["content_sha256"], vb.leaf_pin(), vb.helper_pin(), {k: ident["source"][k] for k in ("kind", "state_sha256", "catalog_hash")})
    rc = fhv.export_held_context_vegetation(vc, recc, pinsc, memory_budget_bytes=BIG, available_bytes_fn=w.c.rdr)
    assert rc.meta()["vegetation"]["law"]["g_sha256"] == m["vegetation"]["law"]["g_sha256"]
    assert rc.meta()["identities"]["matrix_op_sha256"] != m["identities"]["matrix_op_sha256"]


def test_one_cold_upload_set_and_no_per_step_transfer_in_the_reconstruction(gpu, leaf, tmp_path, monkeypatch):
    w = world(gpu, leaf, tmp_path, False)
    r = fhv.export_held_context_vegetation(w.v, w.rec, w.pins, **dev(w), **HOSTQ)
    calls = []
    orig = fhc._h2d
    monkeypatch.setattr(fhc, "_h2d", lambda cp, host: (calls.append(host.shape), orig(cp, host))[1])
    ctx = fh.reconstruct_vegetation_held_context(r, expected_of(r), w.pins, **dev(w), **HOSTQ)
    n1 = len(calls)
    assert 0 < n1 <= 6 + len(r.meta()["overrides"] or ())                                                        # the held maps (FC, SAT, diffusivity, D0, mask, Ks) are uploaded ONCE each, cold
    fh.reconstruct_vegetation_held_context(r, expected_of(r), w.pins, **dev(w), **HOSTQ)
    assert len(calls) == 2 * n1                                                                                 # constant per reconstruction; nothing per hydraulic step exists in this component
    assert ctx.backend == "cuda"


def test_refusals_both_pools_current_device_pins_and_the_retry(gpu, leaf, tmp_path):
    w = world(gpu, leaf, tmp_path, False)
    r = fhv.export_held_context_vegetation(w.v, w.rec, w.pins, **dev(w), **HOSTQ)
    good = expected_of(r)
    pool = gpu.cp.get_default_memory_pool()
    gpu.cp.cuda.Stream.null.synchronize()
    used0 = pool.used_bytes()
    d_need, h_need, _e = fvc.plan_cuda(r.meta())
    with pytest.raises(fh.HeldContextError):
        fh.reconstruct_vegetation_held_context(r, good, w.pins, memory_budget_bytes=d_need - 1, available_bytes_fn=w.g.rdr, **HOSTQ)
    with pytest.raises(fh.HeldContextError):
        fh.reconstruct_vegetation_held_context(r, good, w.pins, **dev(w), host_memory_budget_bytes=h_need - 1, host_available_bytes_fn=lambda: BIG, host_reserve_bytes=0)
    with pytest.raises(fh.HeldContextError):
        fh.reconstruct_vegetation_held_context(r, good, w.pins, **dev(w))                                       # the host pool is mandatory
    for fld, val in (("record_sha256", "0" * 64), ("binding_sha256", "1" * 64), ("leaf_package_sha256", "3" * 64)):
        with pytest.raises(fh.HeldContextError):
            fh.reconstruct_vegetation_held_context(r, good, fhv.ExpectedVegetationPins(**{**w.pins.__dict__, fld: val}), **dev(w), **HOSTQ)
    with pytest.raises(fh.HeldContextError):
        fh.reconstruct_vegetation_held_context(r, fh.ExpectedHeldIdentities(good.config_sha256, good.root_column_sha256, "0" * 64, good.pref_sha256), w.pins, **dev(w), **HOSTQ)
    gc.collect()
    gpu.cp.cuda.Stream.null.synchronize()
    assert pool.used_bytes() == used0                                                                           # no refusal left a device allocation
    fh.reconstruct_vegetation_held_context(r, good, w.pins, **dev(w), **HOSTQ)                                  # the retry succeeds
    # the export of a vertical that does not hold the law (the pre-law CUDA boundary) is refused by the always-on comparison
    old = fp.FloatingPondVertical.from_boundary(w.b.root, available_bytes_fn=w.g.rdr, memory_budget_bytes=BIG, **HOSTQ)
    with pytest.raises(fh.HeldContextError):
        fhv.export_held_context_vegetation(old, w.rec, w.pins, **dev(w), **HOSTQ)


def test_the_old_profiles_refuse_the_dynamic_cuda_vertical_and_record(gpu, leaf, tmp_path):
    w = world(gpu, leaf, tmp_path, False)
    for verify in (False, True):                                                                                # the OLD /2 export fails closed for a per-cell g, with verify False too
        with pytest.raises(fh.HeldContextError, match="per-cell vegetation"):
            fhc.export_held_context_cuda(w.v, None, verify=verify, available_bytes_fn=w.g.rdr, host_available_bytes_fn=lambda: BIG)
    r = fhv.export_held_context_vegetation(w.v, w.rec, w.pins, **dev(w), **HOSTQ)
    with pytest.raises(fh.HeldContextError):
        fh.reconstruct_held_context(r, expected_of(r), available_bytes_fn=w.g.rdr)                              # the old reconstruct refuses /4
    with pytest.raises(fh.HeldContextError):
        fh.record_from_parts(r.meta(), {k: np.array(a) for k, a in r.arrays().items()})


def test_a_wrong_current_device_is_refused_by_the_gate_before_any_reader_or_hash(gpu, leaf, tmp_path):
    cp = gpu.cp
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("two devices are required for the wrong-current-device control (no claim)")
    w = world(gpu, leaf, tmp_path, False)
    r = fhv.export_held_context_vegetation(w.v, w.rec, w.pins, **dev(w), **HOSTQ)
    other = 1 if DEVICE == 0 else 0
    seen = []
    with cp.cuda.Device(other), pytest.raises(fh.HeldContextError):
        fh.reconstruct_vegetation_held_context(r, expected_of(r), w.pins, memory_budget_bytes=BIG, available_bytes_fn=lambda: seen.append(1) or BIG, **HOSTQ)
    assert seen == []                                                                                           # the gate came first: no reader ran
