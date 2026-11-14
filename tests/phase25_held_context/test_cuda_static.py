# ruff: noqa: RUF059
"""CUDA (/4) host-only controls of the dynamic-vegetation held context: no GPU, CuPy, NVRTC or nvidia-smi is touched. They prove the versioned schema, that the PUBLIC owning entry gates the CURRENT device before any reader,
array hash or preparation (spies), the mandatory two pools, the closed /4 metadata, the old per-cell-g guard on the CUDA export path and the absence of implicit NumPy conversion of device arrays. The device behaviour is
defined in `test_gpu_definitions.py` (never run by the author)."""
from __future__ import annotations

import copy
import inspect

import pytest
from hc_support import AVAIL, BIG, World, export, fh, fhv, serialise

from maple_syrup import floating_held_context_cuda as fhc
from maple_syrup import floating_held_context_vegetation_cuda as fvc

ERR = fh.HeldContextError


@pytest.fixture(scope="module")
def parts(leaf, tmp_path_factory):
    w = World(leaf, tmp_path_factory.mktemp("cuda_static"), "numpy", False)
    r = export(w.vertical, w.rec, w.pins)
    meta, arrays = serialise(r)
    m4 = copy.deepcopy(meta)
    m4["schema"], m4["backend"], m4["kernel"] = fhv.SCHEMA4, "cuda", fhc.kernel_descriptor()
    return w, meta, m4, arrays


def expected4(r4):
    i = r4.meta()["identities"]
    return fh.ExpectedHeldIdentities(i["config_sha256"], i["root_column_sha256"], i["matrix_op_sha256"], i["pref_sha256"])


def test_the_cuda_record_is_a_distinct_closed_schema_and_the_backend_versions_never_mix(parts):
    w, m3, m4, arrays = parts
    r4 = fhv.record_from_parts_vegetation(m4, arrays, available_bytes_fn=AVAIL)             # a HOST-data parser: no GPU needed
    assert r4.meta()["schema"] == fhv.SCHEMA4 and r4.meta()["kernel"] == fhc.kernel_descriptor() and fhv.record_schema(r4) == fhv.SCHEMA4
    for edit in (lambda m: m.__setitem__("backend", "cuda") if m["schema"] == fhv.SCHEMA3 else None, lambda m: m.__setitem__("backend", "numpy") if m["schema"] == fhv.SCHEMA4 else None,
                 lambda m: m.pop("kernel", None), lambda m: m["kernel"].__setitem__("fmc_source_sha256", "x") if "kernel" in m else None):
        for base in (m3, m4):
            bad = copy.deepcopy(base)
            edit(bad)
            if bad == base or (bad["schema"] == fhv.SCHEMA3 and "kernel" not in bad and bad["backend"] == m3["backend"]):
                continue
            with pytest.raises(ERR):
                fhv.record_from_parts_vegetation(bad, arrays)
    with pytest.raises(ERR):
        fh.record_from_parts(m4, arrays)
    assert fhv.SCHEMA4 not in (fh.SCHEMA, fh.SCHEMA2) and fhv.SCHEMA3 not in (fh.SCHEMA, fh.SCHEMA2)


def test_the_public_entry_gates_the_current_device_before_any_reader_hash_or_preparation(parts, monkeypatch):
    w, m3, m4, arrays = parts
    r4 = fhv.record_from_parts_vegetation(m4, arrays, available_bytes_fn=AVAIL)
    e4 = expected4(r4)
    r3 = fhv.record_from_parts_vegetation(m3, arrays, available_bytes_fn=AVAIL)
    e3 = expected4(r3)                                                                                      # (`.meta()` verifies: build the expectations BEFORE the spies)
    old = fh.export_held_context(w.old_vertical(), available_bytes_fn=AVAIL, verify=False) if w.old_vertical().held_template().preferential is None else None
    calls = []

    def reader():
        calls.append("reader")
        raise AssertionError("a reader was consulted BEFORE the device gate")

    # the REAL gate on a host without CuPy / a device: the typed refusal comes first, BEFORE the readers and before verify_record hashes the 13 arrays
    spy_verify = []
    orig_verify = fhv.verify_record
    monkeypatch.setattr(fhv, "verify_record", lambda rec: (spy_verify.append(1), orig_verify(rec))[1])
    for kw in ({}, {"host_memory_budget_bytes": BIG, "host_available_bytes_fn": reader}):
        with pytest.raises(ERR, match="CUDA device gate refused"):
            fh.reconstruct_vegetation_held_context(r4, e4, w.pins, memory_budget_bytes=None, available_bytes_fn=reader, **kw)
    assert calls == [] and spy_verify == []                                                                # neither a reader nor the array hash ran
    order = []

    def sentinel_gate(*a, **k):
        order.append("gate")
        raise ERR("gate sentinel")

    monkeypatch.setattr(fhc, "_gate", sentinel_gate)
    with pytest.raises(ERR, match="gate sentinel"):
        fh.reconstruct_vegetation_held_context(r4, e4, w.pins, memory_budget_bytes=None, available_bytes_fn=reader, host_memory_budget_bytes=BIG)
    assert order == ["gate"] and spy_verify == [] and calls == []
    # a CPU record: strict CUDA-argument rejection happens BEFORE the digest / any reader too
    for kw in ({"host_memory_budget_bytes": 1}, {"host_available_bytes_fn": reader}, {"host_reserve_bytes": 1}):
        with pytest.raises(ERR, match="host_"):
            fh.reconstruct_vegetation_held_context(r3, e3, w.pins, memory_budget_bytes=BIG, available_bytes_fn=reader, **kw)
    assert spy_verify == [] and calls == []
    # an old /1 record or a foreign object never reaches a gate or a reader
    for foreign in (object(), None, old):
        if foreign is None and old is None:
            continue
        with pytest.raises(ERR):
            fh.reconstruct_vegetation_held_context(foreign, e3, w.pins, memory_budget_bytes=BIG, available_bytes_fn=reader)
    assert calls == [] and order == ["gate"]


def test_two_pools_are_mandatory_and_the_export_gates_the_device_first(parts, monkeypatch):
    w, m3, m4, arrays = parts
    r4 = fhv.record_from_parts_vegetation(m4, arrays, available_bytes_fn=AVAIL)
    monkeypatch.setattr(fhc, "_gate", lambda *a, **k: (object(), 0))
    with pytest.raises(ERR, match="host_memory_budget_bytes"):
        fvc.prepare_parts_cuda(r4, w.pins, memory_budget_bytes=None)
    with pytest.raises(ERR, match="host_memory_budget_bytes"):
        fvc._need_host_pool(fhc._quota(None, 0, None, 0, None, None))
    src = inspect.getsource(fvc.export_held_context_vegetation_cuda)
    assert src.index("fhc._gate(t.column)") < src.index("fhv._cached_check") < src.index("fhc._admit") < src.index("fhc._d2h")


def test_the_old_cuda_export_path_keeps_the_per_cell_g_guard_before_any_transfer():
    cuda_src = inspect.getsource(fhc.export_held_context_cuda)
    snap = inspect.getsource(fh._snapshot)
    assert "fh._snapshot(vertical, pref_spec)" in cuda_src and cuda_src.index("fh._snapshot(") < cuda_src.index("fh.adopt_parts") < cuda_src.index("if verify")   # the shared guard runs in the snapshot, verify or not
    assert snap.index('layouts["vegetation_factor"] != "scalar"') < snap.index("lay = ") and "per-cell vegetation factor" in snap
    assert cuda_src.index("_gate(t.column)") < cuda_src.index("fh._snapshot(")                                            # the device gate first, then the guard, before any admission / D2H
    assert cuda_src.index("fh._snapshot(") < cuda_src.index("_admit(") < cuda_src.index("_d2h(")


def test_host_arguments_are_refused_by_the_cpu_export_and_plan_is_pure_arithmetic(parts):
    w, m3, m4, arrays = parts
    for kw in ({"host_memory_budget_bytes": 1}, {"host_available_bytes_fn": AVAIL}, {"host_reserve_bytes": 1}):
        with pytest.raises(ERR):
            export(w.vertical, w.rec, w.pins, **kw)
    dev, host, export_host = fvc.plan_cuda(m4)
    assert dev > 0 and host > 0 and export_host > 0 and fvc.plan_cuda(m4) == (dev, host, export_host)
    assert fhv.estimate_vegetation_held_bytes(m3)["reconstruct_bytes"] > 0
    assert host >= fhv.estimate_vegetation_held_bytes(m3)["stages"]["cache"] - fhv.estimate_vegetation_held_bytes(m3)["live_outside_bytes"]       # the host pool carries the cache stage


def test_source_has_no_implicit_numpy_conversion_of_device_arrays_and_uses_the_qualified_transfers():
    src = inspect.getsource(fvc)
    assert "np.asarray(" not in src and "cp.asarray(" not in src and "cupy" not in src.replace("CuPy", "")
    assert "fhc._h2d" in src and "fhc._d2h" in src and "fhc._is_device" in src and "fhc._admit" in src
    prep = inspect.getsource(fvc.prepare_parts_cuda)
    assert prep.index("fhc._gate()") < prep.index("fhc._admit(cp") and "NotImplemented" not in src
    hook = inspect.getsource(fh.reconstruct_vegetation_held_context)
    assert hook.index("fhv.record_schema(record)") < hook.index("fhcuda._gate()") < hook.index("fhv.verify_record(record)") < hook.index("fhv.prepare_parts(")
