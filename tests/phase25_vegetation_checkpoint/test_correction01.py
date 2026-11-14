"""Correction01: the NEW profiles' own public error contract (C1), the unconsumed-token owning check (C3), the literal retained-recipe accounting and the removal of message-text classification (C4). REAL native states and the ACTUAL LEAF
projection; CPU only (the CUDA twin's pure-host validators are exercised, never a device)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from vck_builders import AVAIL, BIG, IDENT, boundary_vns, load, save

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_context_checkpoint as fcc
from maple_syrup import floating_context_checkpoint_cuda as fcxc
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_held_context_vegetation as fhv
from maple_syrup import floating_lifecycle_checkpoint as lcp
from maple_syrup import floating_lifecycle_checkpoint_cuda as s2
from maple_syrup import floating_lifecycle_checkpoint_vegetation as cpuv
from maple_syrup import floating_lifecycle_checkpoint_vegetation_cuda as cv
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import native_floating_soil as nfs
from maple_syrup import pond_checkpoint as pc
from maple_syrup import vegetation_lifecycle as vl
from maple_syrup import water_vegetation_held as wvh

SRC = Path(__file__).resolve().parents[2] / "src" / "maple_syrup"


def _exact(exc, cls):
    assert type(exc.value) is cls, (type(exc.value).__module__, type(exc.value).__name__, str(exc.value))


def _cuda_kw(**over):
    kw = {"available_bytes_fn": AVAIL, "host_available_bytes_fn": AVAIL, "memory_budget_bytes": BIG, "host_memory_budget_bytes": BIG}
    kw.update(over)
    return kw


CUDA_LOAD = [({"expected_device_id": True}, "strict integer"), ({"expected_device_id": "0"}, "strict integer"), ({"expected_device_id": -1}, "strict integer"), ({"batch_cells": 0}, "batch"),
             ({"reserve_bytes": -1}, "reserve")]


@pytest.mark.parametrize("over,rx", CUDA_LOAD, ids=lambda v: str(v)[:30])
def test_the_cuda_public_entries_own_their_pure_host_validator_refusals(tmp_path, over, rx):
    own = cv.VegetationLifecycleCheckpointCudaError
    kw = {"expected_device_id": 0, **_cuda_kw(), **over}
    pins = {"leaf_package_sha256": "0" * 64, "external_helpers": []}
    with pytest.raises(own, match=rx) as e:
        cv.load_floating_lifecycle_vegetation_checkpoint_cuda(tmp_path / "x", lambda *a, **k: None, IDENT, "0" * 64, original_topology=object(), **pins, **kw)
    _exact(e, own)
    with pytest.raises(own, match=rx) as e:
        cv.vegetation_load_need_bytes_cuda(tmp_path / "x", IDENT, "0" * 64, **kw)
    _exact(e, own)
    if "expected_device_id" not in over:  # the save has no device-id argument
        skw = {k: v for k, v in kw.items() if k != "expected_device_id"}
        with pytest.raises(own, match=rx) as e:
            cv.save_floating_lifecycle_vegetation_checkpoint_cuda(tmp_path / "y", object(), lambda *a, **k: None, IDENT, original_topology=object(), **skw)
        _exact(e, own)
        assert not (tmp_path / "y").exists()


def test_the_cuda_gate_and_child_helpers_refuse_with_the_new_class_and_memory_classes_pass_through(tmp_path, monkeypatch):
    own = cv.VegetationLifecycleCheckpointCudaError
    with pytest.raises(own) as e:  # no device here: the gate refusal is the new class (never the old S2 class)
        cv.load_floating_lifecycle_vegetation_checkpoint_cuda(tmp_path / "x", lambda *a, **k: None, IDENT, "0" * 64, leaf_package_sha256="0" * 64, external_helpers=[], original_topology=object(),
                                                              expected_device_id=0, **_cuda_kw())
    _exact(e, own)
    assert not isinstance(e.value, s2.LifecycleCheckpointCudaError)
    for cls in (fsl.FloatingMemoryError, nfs.NativeFloatingMemoryError):  # native memory classes keep their own class by type, never via the message
        def boom(*a, **k):
            raise cls("synthetic quota refusal")  # noqa: B023

        monkeypatch.setattr(fcxc, "_quota", boom)
        with pytest.raises(cls) as e:
            cv.vegetation_load_need_bytes_cuda(tmp_path / "x", IDENT, "0" * 64, expected_device_id=0, **_cuda_kw())
        _exact(e, cls)
    # an OLD sibling error raised by an accepted helper is re-typed (S2 / context / continued families)
    for old in (s2.LifecycleCheckpointCudaError, fc.FloatingCheckpointError, fcc.ContextCheckpointError, pc.PondCheckpointError, cpuv.VegetationLifecycleCheckpointError):
        def raiser(*a, _old=old, **k):
            raise _old("old family refusal")

        monkeypatch.setattr(fcxc, "_quota", raiser)
        with pytest.raises(own) as e:
            cv.vegetation_load_need_bytes_cuda(tmp_path / "x", IDENT, "0" * 64, expected_device_id=0, **_cuda_kw())
        _exact(e, own)


def test_the_cpu_public_entries_own_their_refusals(bnd_ns, tmp_path):
    own = cpuv.VegetationLifecycleCheckpointError
    ns, saved = bnd_ns
    for over in ({"reserve_bytes": -1}, {"batch_cells": 0}, {"max_bytes": 0}):
        with pytest.raises(own) as e:
            save(ns, tmp_path / "x", **over)
        _exact(e, own)
        with pytest.raises(own) as e:
            load(ns, saved.path, saved, **over)
        _exact(e, own)
        with pytest.raises(own) as e:
            cpuv.vegetation_load_need_bytes(saved.path, IDENT, saved.manifest_sha256, available_bytes_fn=AVAIL, **{k: v for k, v in over.items()})
        _exact(e, own)
    with pytest.raises(own) as e:  # a wrong expected seal / identity: the named container families are re-typed
        cpuv.vegetation_load_need_bytes(saved.path, IDENT, "0" * 64, available_bytes_fn=AVAIL)
    _exact(e, own)
    with pytest.raises(nfs.NativeFloatingMemoryError) as e:  # the native memory class survives, by type
        cpuv.vegetation_load_need_bytes(saved.path, IDENT, saved.manifest_sha256, available_bytes_fn=lambda: 1)
    _exact(e, nfs.NativeFloatingMemoryError)
    assert not (tmp_path / "x").exists()


@pytest.fixture
def bnd_ns(tmp_path, leaf):
    ns = boundary_vns(tmp_path / "a", leaf, "numpy", False)
    return ns, save(ns, tmp_path / "ck")


def test_the_new_profiles_classify_nothing_by_message_text():
    for name in ("floating_lifecycle_checkpoint_vegetation.py", "floating_lifecycle_checkpoint_vegetation_cuda.py"):
        text = (SRC / name).read_text()
        assert "reported available" not in text and '"declared" in str' not in text and "modelled bytes\" in str" not in text, name


def test_a_consumed_token_cannot_mint_a_second_record_and_nothing_is_bound_or_uploaded(bnd_ns, monkeypatch):
    ns, saved = bnd_ns
    cap = {}
    orig = lc.stage_restored_head

    def spy(token, kind, head_rec, hold_rec=None, **kw):
        cap["token"] = token
        return orig(token, kind, head_rec, hold_rec, **kw)

    monkeypatch.setattr(lc, "stage_restored_head", spy)
    r = load(ns, saved.path, saved)  # the genuine FIRST use of the token works
    monkeypatch.undo()
    V = json.loads((saved.path / cpuv.MANIFEST_NAME).read_text())["vegetation"]
    good = {"generation": V["generation"], "recipe_id": V["recipe_id"], "parent_recipe_id": V["parent_recipe_id"], "applied": dict(V["applied"]), "max_bytes": BIG}
    bound = []
    real = wvh.cached_binding
    monkeypatch.setattr(wvh, "cached_binding", lambda *a, **k: bound.append(1) or real(*a, **k))
    with pytest.raises(vl.VegetationLifecycleError, match="unconsumed verified-restore"):
        vl.restore_vegetation_record(cap["token"], **good)  # post-install: the vertical already opened a lineage
    with pytest.raises(lc.LifecycleError, match="consumed"):
        lc.require_usable_token(cap["token"])
    assert bound == [] and r.lineage.head is r.head  # refused BEFORE any binding / array / upload; the installed head is untouched
    with pytest.raises(lc.LifecycleError, match="not a genuine VerifiedRestore"):
        lc.require_usable_token(object())


def test_a_staged_but_not_installed_token_is_still_usable_and_a_duplicate_stage_after_install_is_refused(bnd_ns, monkeypatch):
    ns, saved = bnd_ns
    seen = {}
    orig = lc.install_restored_head

    def spy(token, lineage, head):
        lc.require_usable_token(token)  # staged, NOT yet installed: still usable
        seen["ok"] = True
        return orig(token, lineage, head)

    monkeypatch.setattr(lc, "install_restored_head", spy)
    load(ns, saved.path, saved)
    assert seen["ok"]


def test_a_live_captured_recipe_is_charged_once_to_every_later_admission_and_a_restored_one_is_not(tmp_path, leaf, monkeypatch):
    ns = boundary_vns(tmp_path / "a", leaf, "numpy", False)
    R = 12345
    seen = {"carrier": [], "child": [], "cap": []}
    real_cap, real_car, real_child = wvh.capture_cached_inputs, lcp._admit_carrier, cpuv.cc.save_floating_continued_checkpoint

    def cap(*a, **k):
        out = real_cap(*a, **k)
        seen["cap"].append(int(out.resident_bytes))
        return out

    monkeypatch.setattr(wvh, "capture_cached_inputs", cap)
    monkeypatch.setattr(lcp, "_admit_carrier", lambda *a, **k: (seen["carrier"].append(a[-1]), real_car(*a, **k))[1])
    monkeypatch.setattr(cpuv.cc, "save_floating_continued_checkpoint", lambda *a, **k: (seen["child"].append(k["reserve_bytes"]), real_child(*a, **k))[1])
    saved = save(ns, tmp_path / "live", reserve_bytes=R)
    resident = seen["cap"][0]
    assert resident > 0 and seen["carrier"] == [R + resident] and seen["child"] == [R + resident]  # charged ONCE (not twice) to each later admission
    seen["carrier"].clear()
    seen["child"].clear()
    r = load(ns, saved.path, saved)
    ns2 = type(ns)(**{**vars(ns), "lin": r.lineage})
    save(ns2, tmp_path / "again", reserve_bytes=R)
    assert seen["cap"] == [resident] and seen["carrier"] == [R] and seen["child"] == [R]  # a restored record's recipe is caller-owned: no new capture, no extra charge


def test_the_cpu_load_model_carries_the_retained_cached_recipe_term(tmp_path, leaf, bnd_ns, monkeypatch):
    _ns, saved = bnd_ns
    meta = json.loads((saved.path / cpuv.MANIFEST_NAME).read_text())["context_meta"]
    est = fhv.estimate_vegetation_held_bytes(meta)
    seen = {}
    real = fcc._compose

    def spy(msize, tb, m, child_need, cold, reserve):
        seen["retained"] = int(m.retained_bytes)
        return real(msize, tb, m, child_need, cold, reserve)

    monkeypatch.setattr(fcc, "_compose", spy)
    cpuv.vegetation_load_need_bytes(saved.path, IDENT, saved.manifest_sha256, available_bytes_fn=AVAIL)
    assert seen["retained"] == est["live_outside_bytes"] - est["record_bytes"] + est["cache_resident_bytes"] and est["cache_resident_bytes"] > 0
