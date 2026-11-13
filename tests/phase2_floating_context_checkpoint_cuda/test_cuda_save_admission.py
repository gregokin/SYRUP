"""G-F4 correction01 HOST tests: the CUDA outer SAVE is admitted from header/metadata dimensions BEFORE the H0 export, any array/hash/copy or the staging directory (root_gpu_context_outer_budget_probe). CPU/host: NO GPU, NO CuPy. A CPU vertical
stands in for the CUDA one ONLY for the arithmetic/ordering controls (the real `fcu.admit` runs, with the device gate replaced and the actual H0 export replaced by a recording sentinel); the physical device behaviour is defined in
`test_gpu_save_definitions.py` for Root."""
# ruff: noqa: RUF059
from __future__ import annotations

import numpy as np
import pytest
from case import AVAIL, IDENT, build

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_checkpoint_cuda as fcc
from maple_syrup import floating_context_checkpoint as cc
from maple_syrup import floating_context_checkpoint_cuda as ccc
from maple_syrup import floating_held_context as fh
from maple_syrup import floating_held_context_cuda as fhc
from maple_syrup import floating_resume as fr
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu


class Entered(RuntimeError):
    pass


@pytest.fixture(params=[False, True], ids=["dense", "windowed"])
def stand_in(request, tmp_path, monkeypatch):
    ev, st, vert = build(tmp_path / "o", backend="numpy", windowed=request.param)
    spec = fh.PreferentialSpec(np.full(vert.n_cells, 2.0e-3), 0.5, 1e-5, np.array([0.0, 1.0, 1.0]), (), None)
    entered = []
    real_snapshot = fh._snapshot

    def cuda_labelled(vertical, pref_spec):  # the CPU stand-in's metadata labelled as the CUDA vertical's (ordering/arithmetic control only)
        meta, refs = real_snapshot(vertical, pref_spec)
        meta["backend"] = "cuda"
        return meta, refs

    monkeypatch.setattr(fh, "_snapshot", cuda_labelled)
    monkeypatch.setattr(ccc.fcc, "check_event", lambda e: vert)
    monkeypatch.setattr(ccc.fcc, "gate", lambda column: (object(), 0))
    monkeypatch.setattr(ccc.fcu, "gate_device", lambda cp, dev: 0 if dev is None else int(dev))

    def export(*a, **k):
        entered.append(k)
        raise Entered("actual H0 export entered")

    monkeypatch.setattr(ccc.fhc, "export_held_context_cuda", export)
    return ev, st, vert, spec, entered, tmp_path, request.param


def save(ev, st, spec, tmp, name, **kw):
    kw.setdefault("available_bytes_fn", lambda: 1 << 40)
    kw.setdefault("host_available_bytes_fn", lambda: 1 << 40)
    return ccc.save_floating_context_checkpoint_cuda(tmp / name, st, ev, IDENT, pref_spec=spec, **kw)


def test_zero_budgets_and_zero_readers_are_refused_before_the_actual_h0_export_and_nothing_is_created(stand_in):
    ev, st, vert, spec, entered, tmp, windowed = stand_in
    for name, kw in (("dev_budget", {"memory_budget_bytes": 0}), ("host_budget", {"host_memory_budget_bytes": 0}), ("dev_reader", {"available_bytes_fn": lambda: 0}), ("host_reader", {"host_available_bytes_fn": lambda: 0})):
        with pytest.raises(fsl.FloatingMemoryError):
            save(ev, st, spec, tmp, name, **kw)
        assert entered == [] and not (tmp / name).exists() and not list(tmp.glob(f".{name}.partial-*"))  # Root's reproducer: the export was entered before; now nothing is allocated, staged or published


def test_need_minus_one_exact_and_the_reserves_counted_once_on_both_pools(stand_in):
    ev, st, vert, spec, entered, tmp, windowed = stand_in
    n = ccc.save_need_bytes_cuda(st, ev, IDENT, pref_spec=spec)
    dev_total, host_total = n["device_bytes"] + 7, n["host_bytes"] + fcu.PY_ALLOWANCE_BYTES + 11
    for side, kw_ok, kw_bad in (("device", {"available_bytes_fn": lambda: dev_total}, {"available_bytes_fn": lambda: dev_total - 1}),
                                ("host", {"host_available_bytes_fn": lambda: host_total}, {"host_available_bytes_fn": lambda: host_total - 1})):
        with pytest.raises(fsl.FloatingMemoryError, match=side):
            save(ev, st, spec, tmp, f"bad_{side}", reserve_bytes=7, host_reserve_bytes=11, **kw_bad)
        assert entered == []
        with pytest.raises(Entered):
            save(ev, st, spec, tmp, f"ok_{side}", reserve_bytes=7, host_reserve_bytes=11, **kw_ok)  # exactly the published need passes the admission and reaches the export
        assert not list(tmp.glob(".ok_*.partial-*"))  # the staging directory (created only AFTER the admission) is cleaned
        entered.clear()
    with pytest.raises(Entered):
        save(ev, st, spec, tmp, "budget_ok", memory_budget_bytes=dev_total, host_memory_budget_bytes=host_total, reserve_bytes=7, host_reserve_bytes=11)
    with pytest.raises(fsl.FloatingMemoryError):
        save(ev, st, spec, tmp, "budget_bad", memory_budget_bytes=dev_total - 1, host_memory_budget_bytes=host_total, reserve_bytes=7, host_reserve_bytes=11)


def test_the_nested_export_readers_are_capped_at_the_explicit_budgets_without_subtracting_the_reserve_twice(stand_in):
    ev, st, vert, spec, entered, tmp, windowed = stand_in
    n = ccc.save_need_bytes_cuda(st, ev, IDENT, pref_spec=spec)
    big_dev, big_host = n["device_bytes"] + 10_000, n["host_bytes"] + fcu.PY_ALLOWANCE_BYTES + 10_000
    with pytest.raises(Entered):
        save(ev, st, spec, tmp, "cap", memory_budget_bytes=big_dev, host_memory_budget_bytes=big_host, reserve_bytes=5, host_reserve_bytes=9)
    k = entered[0]
    assert k["available_bytes_fn"]() == big_dev and k["host_available_bytes_fn"]() == big_host  # min(real 2**40, budget): the budget caps the nested reader
    assert k["reserve_bytes"] == 5 and k["host_reserve_bytes"] == 9  # the reserve is passed through ONCE (the nested helper adds it itself; it is not pre-subtracted)


def test_the_composed_save_need_is_the_documented_stage_composition(stand_in):
    ev, st, vert, spec, entered, tmp, windowed = stand_in
    n = ccc.save_need_bytes_cuda(st, ev, IDENT, pref_spec=spec)
    s = n["stages"]
    meta, refs = fh._snapshot(vert, spec)
    meta["schema"], meta["kernel"] = fh.SCHEMA2, fhc.kernel_descriptor()
    plan_dev, plan_host, export_host = fhc.plan(meta)
    rec = int(fh._analytic(meta)[1])
    assert s["A_export_verify"] == {"device": plan_dev, "host": plan_host + export_host} and s["A2_arrays"]["host"] == 2 * rec + fh.ALLOWANCE_BYTES and s["record_bytes"] == rec
    cold = max(int(fr._cold_source_bytes()), cc.COLD_ALLOWANCE_BYTES)
    assert s["cold_source_bytes"] == cold
    assert n["device_bytes"] == max(plan_dev, s["B_child_save"]["device"])
    assert n["host_bytes"] == max(plan_host + export_host, 2 * rec + fh.ALLOWANCE_BYTES, s["B_child_save"]["host"], cc.BUFFER_BYTES) + cold
    if windowed:
        col = st.bed.voxel_column
        assert s["B_child_save"]["windowed"] and s["B_child_save"]["cache_bytes"] >= int(col.deep_store.mapped_bytes_bound) and s["B_child_save"]["V"] == 0  # the REAL mapping bound, not the resident archive
    else:
        assert not s["B_child_save"]["windowed"] and s["B_child_save"]["V"] == int(st.bed.voxel_column.mass_kg.shape[2])
    assert fc is not None


def test_a_non_wet_or_foreign_state_is_refused_typed_before_any_admission_or_staging(stand_in, monkeypatch):
    ev, st, vert, spec, entered, tmp, windowed = stand_in
    reads = []
    with pytest.raises(ccc.ContextCheckpointCudaError, match="genuine wet PondEventState"):
        ccc.save_floating_context_checkpoint_cuda(tmp / "foreign", object(), ev, IDENT, pref_spec=spec, available_bytes_fn=lambda: reads.append(1) or 1 << 40)
    assert reads == [] and entered == [] and not list(tmp.glob(".foreign*"))


def test_the_outer_load_child_models_dominate_the_childs_own_load_inputs(tmp_path, monkeypatch):
    """LOAD audit (arithmetic): the outer composed child stage uses the verified header dimensions with every uncertain term at its upper bound (all table bytes device-placed, all cells active, stored file size >= member bytes),
    so it never under-admits the child's own `fcc.load_models` for the same file; the strict child checks stay (no compression multiplier, nothing weakened)."""
    import json

    from test_cuda_context_host import forge

    ck = forge(tmp_path / "f")
    cman = json.loads((ck / cc.CHILD_DIR / fc.MANIFEST_NAME).read_text())
    data = json.loads((ck / cc.MANIFEST_NAME).read_text())
    cfiles = data["child"]["files"]
    dev, host, det = ccc._child_models(cman, data["context_meta"], cfiles, cc.nfs_default_batch())
    L, N = data["context_meta"]["n_layers"], data["context_meta"]["n_cells"]
    nat = cman["native_dense"]
    own_dev, own_host = fcc.load_models(L, N, cman["binding"]["n_classes"], cman["binding"]["phase_bins"], table_bytes=det["table_bytes"], dev_table_bytes=det["table_bytes"] // 2, native_bytes=nat["native_member_bytes"],
                                        V=nat["voxel_shape"][2], K=nat["voxel_shape"][3], n_active=N // 2, batch_cells=cc.nfs_default_batch())
    assert dev >= own_dev and host >= own_host
    assert AVAIL is not None


def test_bad_or_missing_pref_spec_is_a_typed_container_refusal_at_both_apis_before_export_and_staging(stand_in):
    """C3 (D1): `fh.HeldContextError` from the metadata snapshot/check no longer escapes raw from `save` or `save_need_bytes_cuda`."""
    ev, st, vert, spec, entered, tmp, windowed = stand_in
    has_pref = vert.held_template().preferential is not None
    bad = None if has_pref else spec  # missing raw spec for a preferential operator, or a spec for a vertical without one
    reads = []
    with pytest.raises(ccc.ContextCheckpointCudaError):
        ccc.save_need_bytes_cuda(st, ev, IDENT, pref_spec=bad)
    with pytest.raises(ccc.ContextCheckpointCudaError):
        save(ev, st, bad, tmp, "badspec", available_bytes_fn=lambda: reads.append(1) or 1 << 40)
    assert reads == [] and entered == [] and not (tmp / "badspec").exists() and not list(tmp.glob(".badspec*"))


def test_memory_class_hold_through_the_typed_conversion(stand_in, monkeypatch):
    ev, st, vert, spec, entered, tmp, windowed = stand_in

    def boom(*a, **k):
        raise fh.HeldContextError("needs 5 B but only 1 B reported available")

    monkeypatch.setattr(ccc.fhc, "plan", boom)
    with pytest.raises(fsl.FloatingMemoryError):
        ccc.save_need_bytes_cuda(st, ev, IDENT, pref_spec=spec)


def test_windowed_publication_guard_runs_before_any_deep_metadata(stand_in, monkeypatch):
    ev, st, vert, spec, entered, tmp, windowed = stand_in
    if not windowed:
        pytest.skip("Windowed only")
    calls = []
    monkeypatch.setattr(ccc.pc, "_require_published", lambda bed: (_ for _ in ()).throw(ccc.pc.PondCheckpointError("pending journal: publish first")))
    monkeypatch.setattr(ccc.pc, "_deep_record", lambda *a, **k: calls.append("deep"))
    reads = []
    with pytest.raises(ccc.pc.PondCheckpointError, match="publish first"):
        save(ev, st, spec, tmp, "pend", available_bytes_fn=lambda: reads.append(1) or 1 << 40)
    assert calls == [] and reads == [] and entered == [] and not list(tmp.glob(".pend*"))
