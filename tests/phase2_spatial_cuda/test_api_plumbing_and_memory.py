"""Spatial CUDA API plumbing, admission, checks, lifetimes and memory terms through a numpy-backed CuPy PLUMBING DOUBLE (+ the host-compiled kernels). Nothing here is hardware evidence and no fake
CUDA-tagged owner is offered as proof of device behaviour; these tests pin the Python side: argument assembly, admission before any allocation/launch, exact allocation lists, transfers, typed refusals/errors,
ownership. CPU estimates/semantics are compared with the pre-A4.2 module. Author-written."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import re
import subprocess
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
from conftest import BIG, PRE_A42_SPATIAL, a2, device_owner, hostgpu, needs_host, sp, ss

import maple.core.backend as backend
from maple_syrup import subsurface_cuda as sc
from maple_syrup import subsurface_cuda_spatial as cs

AFTER = Path(__file__).resolve().parent


def build(L=3, N=130, mode="constant", bottom="closed", seed=1, mask=None, hoist=True):
    rng = np.random.default_rng(seed)
    p = a2.params(rng, L, N, mode=mode)
    mask = np.ones(N, bool) if mask is None else mask
    if hoist:  # SHARED edges and interface input (1-D arrays -> shared layouts) -> shared conductance maps -> the hoisted factor kernel is used; p keeps the equivalent per-cell copy
        for k in ("edges", "iface"):
            p[k] = np.repeat(p[k][:, :1], N, axis=1)
        spec = a2.make_spec(p, mode=mode, bottom=bottom, edges=p["edges"][:, 0].copy(), iface=p["iface"][:, 0].copy())
    else:
        spec = a2.make_spec(p, mode=mode, bottom=bottom)
    return rng, p, a2.prepare(spec, mask=mask)


def fx(a):
    return hostgpu.fake(a)


# ---- source structure ----------------------------------------------------------------------------------------------------------------------
def test_the_common_source_is_the_accepted_a41_source_and_the_spatial_source_uses_the_one_body():
    assert hashlib.sha256(sc._SOURCE.encode()).hexdigest() == "46f1307ebfaca150974ce9a111feb21888f1a415b1600963aa00dd1adff6cda8"  # the A4.1-accepted common source, byte for byte
    src = cs.render_source()
    assert src == cs.kernel_source() and src.count(sc._BODY) == 1
    for marker in ("// addition with saturation return", "// gravity cascade, top to bottom", "// evaporation from layer 0", "// sequential pair diffusion from the top", "// output checks",
                   "// fit test right after the ONE movement", "closed bottom: deeper pore capacity"):
        assert src.count(marker) == 1, marker  # no second solver
    assert src.count("return -expm1(-x);") == 1
    for macro in sc.BODY_MACROS:
        assert f"#define {macro}" in cs.SPATIAL_READERS
    head, body = cs.SPATIAL_HEAD, sc._BODY
    assert head.rstrip().endswith("{") and "__device__" in head and body.lstrip().startswith("GUARD_INVALID_LANE")


def test_the_wrapper_kernels_have_no_early_return_before_a_barrier():
    tail = cs.SPATIAL_TAIL
    for name in (cs.STEP_KERNEL, cs.STORAGE_KERNEL):
        text = tail[tail.index(f"void {name}("):]
        text = text[: text.index("\n}\n") + 3]
        code = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        assert re.search(r"\breturn\b", code) is None and "__syncthreads()" in code, name  # every lane, padded or inactive, reaches every barrier
    fin = cs._finalize("x", 7, 2)
    assert re.search(r"\breturn\b", fin) is None and fin.count("__syncthreads()") == 2


def test_rendering_is_stateless_and_the_cache_holds_functions_only(monkeypatch):
    assert cs.render_source() == cs.render_source()
    assert not hasattr(sc, "register_variant") and set(vars(cs)) >= {"_FUNCTIONS"} and cs._FUNCTIONS is not None
    prov = cs.kernel_provenance()
    assert prov["source_sha256"] == hashlib.sha256(cs.render_source().encode()).hexdigest() and prov["shared_body_sha256"] == hashlib.sha256(sc._BODY.encode()).hexdigest()
    assert prov["packet_bytes"] == 72 and prov["fastmath"] is False and tuple(prov["compile_options"]) == ("--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")


def test_cold_import_touches_no_cupy_and_compiles_nothing():
    code = "import sys; import maple_syrup.subsurface_cuda_spatial as c; assert 'cupy' not in sys.modules and c._FUNCTIONS == {}; assert c.render_source(); print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


class RawFail:
    def __init__(self, *a, **k):
        raise RuntimeError("nvrtc boom")


class Slim:
    class RawModule:
        def __init__(self, code, options, backend):
            pass

        def get_function(self, name):
            return type("F", (), {"attributes": {"max_threads_per_block": 64}})()


def test_load_kernels_compiles_once_per_device_with_the_unchanged_flags_and_fails_typed():
    with cs._LOCK:
        saved = dict(cs._FUNCTIONS)
        cs._FUNCTIONS.clear()
    try:
        made = []

        class Cp:
            class RawModule:
                def __init__(self, code, options, backend):
                    made.append((code, tuple(options), backend))

                def get_function(self, name):
                    return type("F", (), {"attributes": {"max_threads_per_block": 1024}, "name": name})()

        a, b = cs.load_kernels(Cp, 0), cs.load_kernels(Cp, 0)
        assert a is b and len(made) == 1 and made[0] == (cs.render_source(), sc.COMPILE_OPTIONS, "nvrtc")
        assert {a.factor.name, a.step.name} == {sc.FACTOR_KERNEL_NAME, cs.STEP_KERNEL}
        cs.load_kernels(Cp, 1)
        assert set(cs._FUNCTIONS) == {0, 1}  # bounded by the number of devices
        cs._FUNCTIONS.clear()
        with pytest.raises(ss.SubsurfaceBackendUnavailableError, match="failed to compile/load"):
            cs.load_kernels(type("C", (), {"RawModule": RawFail}), 0)
        with pytest.raises(ss.SubsurfaceBackendUnavailableError, match="fewer than"):
            cs.load_kernels(Slim, 0)
        assert cs._FUNCTIONS == {}  # a failed load leaves nothing behind
    finally:
        with cs._LOCK:
            cs._FUNCTIONS.clear()
            cs._FUNCTIONS.update(saved)


# ---- memory terms ----------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("L,N", [(1, 1), (3, 129), (5, 1000)])
def test_cuda_estimate_formulas_device_and_host_are_separate_and_exact(L, N):
    nb = (N + 127) // 128
    py = sp.PY_ALLOWANCE_BYTES
    for share in (False, True):
        for op, outputs, scratch, host in (
            ("state", 8 * L * N, 0, 0), ("moisture", 8 * L * N, 0, 0),
            ("uniform_state", 8 * L * N, 8 * L + 32, 24 * L + 32), ("storage", 0, 8 * nb + 8, 8),
            ("step", (L * N + 2 * (L - 1) * N + 4 * N) * 8, 7 * 8 * nb + 72 + 2 * max(L - 1, 1) * 8, 144),
            ("step_with_recharge", (L * N + 2 * (L - 1) * N + 4 * N) * 8, 7 * 8 * nb + 72 + 2 * max(L - 1, 1) * 8, 144)):
            e = sp.estimate_spatial_operation_memory(op, L, N, backend="cuda", reserve_bytes=1000, host_reserve_bytes=77, with_evaporation_share=share,
                                                     hoisted_factor_arrays=2 if op.startswith("step") else 0)
            assert (e.output_bytes, e.scratch_bytes) == (outputs, scratch)
            assert e.device_required_bytes == e.required_bytes == outputs + scratch + 1000
            assert e.host_required_bytes == host + py + 77 and e.host_reserve_bytes == 77
            assert e.terms["backend"] == "cuda" and e.terms["no_rows_array"] and e.terms["n_blocks"] == nb
    assert sp.estimate_spatial_operation_memory("step", L, N, backend="cuda").scratch_bytes == 7 * 8 * nb + 72  # no factor arrays when the factors are evaluated per cell
    assert sp.estimate_spatial_operation_memory("step", L, N, backend="cuda").terms["static_shared_bytes_per_block"] == 7 * 8 * 128


def test_cuda_estimate_validates_before_arithmetic_and_cpu_arguments_stay_cpu_only():
    with pytest.raises(sp.SpatialLayerError, match="only to the cuda backend"):
        sp.estimate_spatial_operation_memory("step", 3, 10, backend="numpy", host_reserve_bytes=1)
    with pytest.raises(sp.SpatialLayerError, match="only to the cuda backend"):
        sp.estimate_spatial_operation_memory("step", 3, 10, backend="numba", hoisted_factor_arrays=1)
    with pytest.raises(sp.SpatialLayerError):
        sp.estimate_spatial_operation_memory("step", 3, 10, backend="gpu")
    with pytest.raises(ss.SubsurfaceError, match="launch grid limit"):  # integer launch range checked BEFORE any cast or allocation
        sp.estimate_spatial_operation_memory("step", 3, 128 * (2**31 - 1) + 1, backend="cuda")
    with pytest.raises(ss.SubsurfaceError, match="32-bit"):
        sp.estimate_spatial_operation_memory("step", 2**31, 10, backend="cuda")


def test_cpu_estimates_are_unchanged_from_the_pre_a42_module():
    if not PRE_A42_SPATIAL.is_file():
        pytest.skip("pre-A4.2 source copy is task evidence only")
    spec = importlib.util.spec_from_file_location("old_spatial_pre_a42", PRE_A42_SPATIAL)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    for backend_ in ("numpy", "numba"):
        for op in ("state", "uniform_state", "moisture", "storage", "step", "step_with_recharge"):
            for L, N in ((1, 1), (3, 129), (5, 1000)):
                for share in (False, True):
                    a = old.estimate_spatial_operation_memory(op, L, N, backend=backend_, with_evaporation_share=share, reserve_bytes=5)
                    b = sp.estimate_spatial_operation_memory(op, L, N, backend=backend_, with_evaporation_share=share, reserve_bytes=5)
                    assert (a.output_bytes, a.scratch_bytes, a.required_bytes, dict(a.input_bytes)) == (b.output_bytes, b.scratch_bytes, b.required_bytes, dict(b.input_bytes))
                    assert b.host_required_bytes == b.required_bytes and b.device_required_bytes == 0


# ---- CPU owners ------------------------------------------------------------------------------------------------------------------------------
def test_a_cpu_owner_refuses_the_cuda_only_quota_arguments_in_every_helper_before_any_work():
    rng, p, col = build()
    water = a2.water_kind(rng, p, "random")
    state = sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG)
    added, pet = np.zeros((3, 130)), np.zeros(130)
    for kw in (dict(host_memory_budget_bytes=BIG), dict(host_reserve_bytes=1), dict(host_available_bytes_fn=lambda: BIG)):
        for call in (lambda: sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG, **kw), lambda: sp.spatial_uniform_state(col, [0.1] * 3, 0.0, memory_budget_bytes=BIG, **kw),
                     lambda: sp.spatial_moisture(col, state, memory_budget_bytes=BIG, **kw), lambda: sp.spatial_storage_m3(col, state, memory_budget_bytes=BIG, **kw),
                     lambda: sp.spatial_step(col, state, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG, **kw),
                     lambda: sp.spatial_step_with_recharge(col, state, 3600.0, recharge_depth_m=1e-3, potential_evaporation_m_s=pet, memory_budget_bytes=BIG, **kw)):
            with pytest.raises(sp.SpatialLayerError, match="apply only to a cuda owner"):
                call()
    assert state.component_tag == sp.STATE_COMPONENT_TAG and state.namespace == "numpy" and state.device_id is None and not state.water_m.flags.writeable  # the CPU state is unchanged by the new fields
    assert sp.STATE_COMPONENT_TAGS == ("spatial_explicit_cpu_state", "spatial_explicit_cuda_state")


# ---- CUDA plumbing ---------------------------------------------------------------------------------------------------------------------------
@needs_host
def test_device_and_namespace_checks_precede_any_copy_launch_or_allocation(cupy):
    rng, p, col = build()
    dev = device_owner(col, cupy)
    water = a2.water_kind(rng, p, "random")
    added, pet = np.zeros((3, 130)), np.zeros(130)
    good = sp.spatial_column_state(dev, fx(water), 0.0, memory_budget_bytes=BIG)
    cupy.allocations.clear()
    cases = [
        ("numpy water", lambda: sp.spatial_column_state(dev, water, 0.0, memory_budget_bytes=0), "cupy.ndarray"),
        ("float32", lambda: sp.spatial_column_state(dev, fx(water.astype(np.float32)), 0.0, memory_budget_bytes=0), "float64"),
        ("shape", lambda: sp.spatial_column_state(dev, fx(water[:-1]), 0.0, memory_budget_bytes=0), "shape"),
        ("fortran", lambda: sp.spatial_column_state(dev, fx(water).T.copy().T, 0.0, memory_budget_bytes=0), "C-contiguous"),
        ("other device array", lambda: sp.spatial_column_state(dev, hostgpu.fake(water, device=1), 0.0, memory_budget_bytes=0), "device 1"),
        ("numpy forcing", lambda: sp.spatial_step(dev, good, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=0), "cupy.ndarray"),
        ("numpy pet", lambda: sp.spatial_step(dev, good, 3600.0, added_water_m=fx(added), potential_evaporation_m_s=pet, memory_budget_bytes=0), "cupy.ndarray"),
        ("numpy share", lambda: sp.spatial_step(dev, good, 3600.0, added_water_m=fx(added), potential_evaporation_m_s=fx(pet), evaporation_share=np.ones(130), memory_budget_bytes=0), "cupy.ndarray"),
        ("cpu state", lambda: sp.spatial_step(dev, sp.spatial_column_state(col, water, 0.0, memory_budget_bytes=BIG), 3600.0, added_water_m=fx(added), potential_evaporation_m_s=fx(pet), memory_budget_bytes=0), "namespace"),
        ("numpy owner, cuda state", lambda: sp.spatial_step(col, good, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG), "namespace"),
    ]
    for label, call, text in cases:
        with pytest.raises(sp.SpatialLayerError, match=text):
            call()
    cupy.rc.current = 1  # the CURRENT device is not the owner's
    for call in (lambda: sp.spatial_column_state(dev, fx(water), 0.0, memory_budget_bytes=BIG), lambda: sp.spatial_moisture(dev, good, memory_budget_bytes=BIG),
                 lambda: sp.spatial_step(dev, good, 3600.0, added_water_m=fx(added), potential_evaporation_m_s=fx(pet), memory_budget_bytes=BIG)):
        with pytest.raises(sp.SpatialLayerError, match="current device"):
            call()
    cupy.rc.current = 0
    assert cupy.allocations == [] and cupy.modules == []  # nothing allocated, nothing compiled/launched by any refusal


@needs_host
@pytest.mark.parametrize("op", ["state", "uniform_state", "moisture", "storage", "step", "step_with_recharge"])
def test_admission_boundaries_are_exact_on_both_sides_and_precede_every_allocation(cupy, op):
    rng, p, col = build()
    dev = device_owner(col, cupy)
    water = fx(a2.water_kind(rng, p, "random"))
    st = sp.spatial_column_state(dev, water, 0.0, memory_budget_bytes=BIG)
    added, pet = fx(np.zeros((3, 130))), fx(np.zeros(130))
    est = sp.estimate_spatial_operation_memory(op, 3, 130, backend="cuda", hoisted_factor_arrays=1 if op.startswith("step") else 0)
    dev_need, host_need = est.device_required_bytes, est.host_required_bytes

    def call(budget=BIG, host=None, reserve=0, host_reserve=0, reader=None, host_reader=None):
        kw = dict(memory_budget_bytes=budget, reserve_bytes=reserve, available_bytes_fn=reader, host_memory_budget_bytes=host, host_reserve_bytes=host_reserve, host_available_bytes_fn=host_reader)
        if op == "state":
            return sp.spatial_column_state(dev, water, 0.0, **kw)
        if op == "uniform_state":
            return sp.spatial_uniform_state(dev, [0.0, 0.01, 0.02], 0.0, **kw)
        if op == "moisture":
            return sp.spatial_moisture(dev, st, **kw)
        if op == "storage":
            return sp.spatial_storage_m3(dev, st, **kw)
        if op == "step":
            return sp.spatial_step(dev, st, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, **kw)
        return sp.spatial_step_with_recharge(dev, st, 3600.0, recharge_depth_m=1e-3, potential_evaporation_m_s=pet, **kw)

    cupy.allocations.clear()
    for kw in (dict(budget=dev_need - 1), dict(budget=dev_need, reserve=1), dict(host=host_need - 1), dict(host=host_need, host_reserve=1), dict(reader=lambda: dev_need - 1),
               dict(host_reader=lambda: host_need - 1), dict(budget=BIG, reserve=BIG, reader=lambda: BIG)):
        with pytest.raises(sp.SpatialLayerMemoryError, match="nothing was allocated"):
            call(**kw)
    assert cupy.allocations == [] and cupy.modules == []
    assert call(budget=dev_need, host=host_need, reader=lambda: dev_need, host_reader=lambda: host_need) is not None  # exactly the estimate is admitted on both sides
    for bad in (lambda: -1, lambda: True, lambda: 1.5):
        with pytest.raises(sp.SpatialLayerError, match="reader must return"):
            call(reader=bad)
        with pytest.raises(sp.SpatialLayerError, match="reader must return"):
            call(host_reader=bad)


@needs_host
@pytest.mark.parametrize("mode_,hoist,factors", [("constant", True, 1), ("tietjen_donor_depth", True, 2), ("constant", False, 0), ("tietjen_donor_depth", False, 0)])
def test_the_step_makes_exactly_the_modelled_allocations_one_packet_read_and_no_upload(cupy, monkeypatch, mode_, hoist, factors):
    L, N = 4, 300
    rng, p, col = build(L=L, N=N, mode=mode_, hoist=hoist)
    dev = device_owner(col, cupy)
    water = fx(a2.water_kind(rng, p, "random"))
    st = sp.spatial_column_state(dev, water, 0.0, memory_budget_bytes=BIG)
    added, pet = fx(rng.random((L, N)) * 1e-4), fx(rng.random(N) * 1e-8)
    hoisted = hoist
    cupy.allocations.clear()
    seen = {"to_host": [], "to_device": []}
    real_h, real_d = backend.to_host, backend.to_device
    monkeypatch.setattr(backend, "to_host", lambda a: (seen["to_host"].append((a.dtype.name, a.nbytes)), real_h(a))[1])
    monkeypatch.setattr(backend, "to_device", lambda a, xp: (seen["to_device"].append(a.nbytes), real_d(a, xp))[1])
    step = sp.spatial_step(dev, st, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    t = cs.operation_terms("step", L, N, factors=factors)
    total = sum(a[3] for a in cupy.allocations)
    assert total == t["outputs"] + t["scratch"], (cupy.allocations, t)  # every device byte the call requests is in the model, and nothing else
    names = sorted((a[1], a[2]) for a in cupy.allocations)
    nb = (N + 127) // 128
    expect = [((L, N), "float64"), ((L - 1, N), "float64"), ((L - 1, N), "float64")] + [((N,), "float64")] * 4 + [((7 * nb,), "float64"), ((9,), "uint64")]
    expect += [((L - 1,), "float64")] * factors
    assert names == sorted(expect)
    assert seen["to_host"] == [("uint64", 72)] and seen["to_device"] == []  # ONE counted 72-byte read, zero uploads
    assert [m for m in cupy.modules] and cupy.modules[0][0] == cs.render_source()
    assert step.state.namespace == "cupy" and step.state.device_id == 0 and step.state.component_tag == sp.STATE_COMPONENT_TAG_CUDA and step.state.t_s == 3600.0
    assert type(step.state.water_m) is hostgpu.FakeArray and step.percolation_m.shape == (L - 1, N)


@needs_host
def test_scratch_lives_until_the_synchronising_read_and_is_released_afterwards(cupy, monkeypatch):
    rng, p, col = build()
    dev = device_owner(col, cupy)
    st = sp.spatial_column_state(dev, fx(a2.water_kind(rng, p, "random")), 0.0, memory_budget_bytes=BIG)
    added, pet = fx(np.zeros((3, 130))), fx(np.zeros(130))
    refs = []
    real_empty = cupy.empty
    cupy.empty = lambda shape, dtype=np.float64: (lambda a: (refs.append((tuple(np.atleast_1d(shape)), weakref.ref(a))), a)[1])(real_empty(shape, dtype))
    real_zeros = cupy.zeros
    cupy.zeros = lambda shape, dtype=np.float64: (lambda a: (refs.append((("diag",), weakref.ref(a))), a)[1])(real_zeros(shape, dtype))
    alive_at_read = []
    real_h = backend.to_host
    monkeypatch.setattr(backend, "to_host", lambda a: (alive_at_read.append([r() is not None for _, r in refs]), real_h(a))[1])
    step = sp.spatial_step(dev, st, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    assert all(alive_at_read[0])  # every temporary (partials, factors, packet) was still referenced when the packet was read back
    gc.collect()
    nb = (130 + 127) // 128
    dead = {tuple(int(x) if not isinstance(x, str) else x for x in shape) for shape, r in refs if r() is None}
    kept = {tuple(int(x) if not isinstance(x, str) else x for x in shape) for shape, r in refs if r() is not None}
    assert {(7 * nb,), (2,), ("diag",)} <= dead  # the partials, the (L-1,) factor array and the packet were released after the read
    assert (3, 130) in kept and (130,) in kept  # the published outputs survive
    assert step.state.water_m is not st.water_m


@needs_host
def test_uniform_init_has_one_counted_upload_of_8L_and_one_32_byte_read(cupy, monkeypatch):
    rng, p, col = build()
    dev = device_owner(col, cupy)
    seen = {"to_host": [], "to_device": []}
    real_h, real_d = backend.to_host, backend.to_device
    monkeypatch.setattr(backend, "to_host", lambda a: (seen["to_host"].append(a.nbytes), real_h(a))[1])
    monkeypatch.setattr(backend, "to_device", lambda a, xp: (seen["to_device"].append(a.nbytes), real_d(a, xp))[1])
    cupy.allocations.clear()
    sp.spatial_uniform_state(dev, [0.0, 0.01, 0.02], 0.0, memory_budget_bytes=BIG)
    assert seen == {"to_host": [32], "to_device": [24]}
    t = cs.operation_terms("uniform_state", 3, 130)
    assert sum(a[3] for a in cupy.allocations) == t["outputs"] + t["scratch"]  # 8LN result + 8L theta + 32 B diagnostic


@needs_host
def test_a_device_allocation_failure_is_typed_publishes_nothing_and_leaves_the_state(cupy):
    rng, p, col = build()
    dev = device_owner(col, cupy)
    st = sp.spatial_column_state(dev, fx(a2.water_kind(rng, p, "random")), 0.0, memory_budget_bytes=BIG)
    before = hashlib.sha256(np.asarray(st.water_m).tobytes()).hexdigest()
    added, pet = fx(np.zeros((3, 130))), fx(np.zeros(130))
    real_empty = cupy.empty
    count = {"n": 0}

    def failing(shape, dtype=np.float64):
        count["n"] += 1
        if count["n"] == 3:
            raise MemoryError("out of memory (test double)")
        return real_empty(shape, dtype)

    cupy.empty = failing
    with pytest.raises(sp.SpatialLayerMemoryError, match="nothing was published"):
        sp.spatial_step(dev, st, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    assert hashlib.sha256(np.asarray(st.water_m).tobytes()).hexdigest() == before and st.t_s == 0.0


@needs_host
def test_a_failed_step_never_writes_state_forcing_or_retained_maps(cupy):
    rng, p, col = build(mask=np.arange(130) % 4 != 0)
    dev = device_owner(col, cupy)
    water = fx(a2.water_kind(rng, p, "above_fc"))
    st = sp.spatial_column_state(dev, water, 1.0, memory_budget_bytes=BIG)
    added, pet = rng.random((3, 130)) * 1e-4 * col._mask, rng.random(130) * 1e-8 * col._mask
    pet[5] = np.nan
    pet = fx(pet)
    added = fx(added)

    def digest():
        h = hashlib.sha256()
        for a in [st.water_m, added, pet] + [dev._maps[n] for n in dev.retained_names()] + [dev._mask]:
            h.update(np.asarray(a).tobytes())
        return h.hexdigest()

    before = digest()
    with pytest.raises(ss.SubsurfaceStepError):
        sp.spatial_step(dev, st, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    assert digest() == before and st.t_s == 1.0


@needs_host
def test_owner_and_states_are_released_and_no_cache_keeps_them(cupy):
    rng, p, col = build()
    dev = device_owner(col, cupy)
    st = sp.spatial_column_state(dev, fx(a2.water_kind(rng, p, "random")), 0.0, memory_budget_bytes=BIG)
    step = sp.spatial_step(dev, st, 3600.0, added_water_m=fx(np.zeros((3, 130))), potential_evaporation_m_s=fx(np.zeros(130)), memory_budget_bytes=BIG)
    refs = [weakref.ref(x) for x in (dev, st, step, step.state, step.state.water_m, dev._maps["dz"])]
    del dev, st, step
    gc.collect()
    assert all(r() is None for r in refs)  # the kernel cache holds functions only


@needs_host
def test_the_cuda_summary_and_state_metadata_state_the_logical_immutability_honestly(cupy):
    rng, p, col = build()
    dev = device_owner(col, cupy)
    st = sp.spatial_column_state(dev, fx(a2.water_kind(rng, p, "random")), 0.0, memory_budget_bytes=BIG)
    s = st.summary()
    assert s["namespace"] == "cupy" and s["device_id"] == 0 and s["component_tag"] == "spatial_explicit_cuda_state"
    assert "LOGICAL" in sp.SpatialSoilState.__doc__ and "nothing detects an external write" in sp.SpatialSoilState.__doc__
    ev = dev.summary()["water_evolution"]
    assert dev.summary()["hydrology_step_supported"] is False and ev["unsupported_backends"] == [] and "asserts no qualification" in ev["cuda_trial_status"] and "receipts" in ev["cuda_trial_status"]


@needs_host
def test_the_step_host_payload_is_the_raw_read_plus_the_rebuilt_float_packet_144_bytes_not_the_72_byte_wire(cupy, monkeypatch):
    """Root correction 1: `run_step` holds the 72-byte raw uint64 array returned by the counted read AND the 72-byte float packet rebuilt for `_resolve` at the same time. The model's host payload is their sum
    (144 B); the wire traffic stays ONE 72-byte read. Both sizes are taken from the actual arrays (spies), not from the constants under test."""
    L, N = 3, 130
    rng, p, col = build(L=L, N=N)
    dev = device_owner(col, cupy)
    st = sp.spatial_column_state(dev, fx(a2.water_kind(rng, p, "random")), 0.0, memory_budget_bytes=BIG)
    added, pet = fx(np.zeros((L, N))), fx(np.zeros(N))
    wire, rebuilt = [], []
    real_h, real_r = backend.to_host, ss._resolve
    monkeypatch.setattr(backend, "to_host", lambda a: (wire.append(a.nbytes), real_h(a))[1])
    monkeypatch.setattr(ss, "_resolve", lambda column, packet, recharge=False: (rebuilt.append(packet.nbytes), real_r(column, packet, recharge))[1])
    sp.spatial_step(dev, st, 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG)
    assert wire == [72] and rebuilt == [72]  # one wire read; the rebuilt float packet is a SECOND 72-byte host array
    t = cs.operation_terms("step", L, N)
    assert t["host"] == wire[0] + rebuilt[0] == cs.STEP_HOST_PACKET_BYTES == 144 and cs.PACKET_BYTES == 72
    e = sp.estimate_spatial_operation_memory("step", L, N, backend="cuda", host_reserve_bytes=9)
    assert e.host_required_bytes == 144 + sp.PY_ALLOWANCE_BYTES + 9  # the known arrays are counted as payload, not absorbed by the Python allowance
    assert sp.estimate_spatial_operation_memory("step_with_recharge", L, N, backend="cuda").host_required_bytes == 144 + sp.PY_ALLOWANCE_BYTES
    # the other operations' host payloads are unchanged: uniform 24L + 32, storage 8, state/moisture 0
    assert cs.operation_terms("uniform_state", L, N)["host"] == 24 * L + 32 and cs.operation_terms("storage", L, N)["host"] == 8 and cs.operation_terms("state", L, N)["host"] == 0


def test_the_gpu_definitions_have_no_unparenthesised_shift_after_an_addition_and_no_total_growth_lower_bound():
    """Root correction 2 (static guard on the written-not-run GPU file): `a <= b + 1 << 20` parses as `a <= (b + 1) << 20` (a huge bound), and `total_bytes` growth is not a lower bound of the outputs."""
    import ast

    src = (AFTER / "test_cuda_definitions.py").read_text()
    tree = ast.parse(src)
    bad = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.LShift) and isinstance(n.left, ast.BinOp) and isinstance(n.left.op, ast.Add)]
    assert bad == [], bad
    assert "est.output_bytes <= growth" not in src and "using_allocator" in src and "data.mem.size" in src and "used_bytes() - before" not in src.replace("after[\"used\"] - before[\"used\"]", "")
