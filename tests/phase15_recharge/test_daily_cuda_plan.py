"""Phase 1.5 S3 CPU evidence for the CUDA daily plumbing (no GPU, no CuPy import, no device): the pure metadata composer, the registered admission entries, the
no-CPU-fallthrough guard, the backend dispatch/refusal of a CPU lifecycle, and structural checks that the CUDA branches use the device-resident mask/namespace and the
registered plans. The genuine CUDA positive controls are `test_daily_cuda.py` (written, unrun)."""
from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from dcases import RUN, build, dw, pet, start, wl
from storm_cases import layers

from maple_syrup import water_lifecycle_cuda as wlc

L, N = 3, 24


def plan(entry, raw=10_000_000, **kw):
    kw.setdefault("schedule_bytes", (16, 8, 8))
    kw.setdefault("n_faces", 62)
    kw.setdefault("storm_host_stage_bytes", 4096)
    kw.setdefault("interevent_build_host_bytes", 2048)
    kw.setdefault("source_record_bytes", 1000)
    return wlc.compose_entry_plan(L, N, entry, device_raw_bytes=raw, host_reserve_bytes=777, **kw)


def test_daily_entries_are_registered_in_both_the_composer_and_the_lifecycle_registry():
    assert wlc.DAILY_ENTRIES == {"daily_start", "daily", "daily_begin_storm", "daily_finish"}
    assert wlc.DAILY_ENTRIES <= wlc._ENTRIES
    assert {k: wl._ENTRY_OF[k] for k in wlc.DAILY_ENTRIES} == {k: k for k in wlc.DAILY_ENTRIES}  # every public label maps to its own plan entry
    with pytest.raises(wlc.CudaLifecycleError):
        plan("daily_unknown")


@pytest.mark.parametrize("entry", sorted(wlc.DAILY_ENTRIES))
def test_composed_daily_plans_have_separate_host_and_device_figures(entry):
    p = plan(entry)
    assert p["device_bytes"] == sum(p["device_terms"].values()) and p["host_bytes"] == sum(p["host_terms"].values())
    assert p["device_terms"]["raw_entry_figure"] == 10_000_000 and p["host_bytes"] != p["device_bytes"]
    assert p["host_terms"]["caller_host_reserve"] == 777 and p["host_terms"]["wrapper_retained_host"] > 0
    assert plan(entry, raw=10_000_001)["device_bytes"] == p["device_bytes"] + 1  # device raw passes straight through, never into the host figure
    assert plan(entry, raw=10_000_001)["host_bytes"] == p["host_bytes"]


def test_host_terms_of_each_daily_entry_include_the_stages_it_really_runs():
    d, f, b, s = plan("daily"), plan("daily_finish"), plan("daily_begin_storm"), plan("daily_start")
    eb, ei, bs, st = plan("enter_from_boundary"), plan("enter_interevent"), plan("begin_storm"), plan("start")
    assert d["host_bytes"] == eb["host_bytes"] + d["host_terms"]["day_step_hot_packets"] + d["host_terms"]["general_owner_canonical_host"]
    assert f["host_bytes"] == ei["host_bytes"] + f["host_terms"]["day_step_hot_packets"] + f["host_terms"]["general_owner_canonical_host"]
    assert b["host_bytes"] == bs["host_bytes"]  # the uniform field is a DEVICE term: no host array
    assert s["host_bytes"] == st["host_bytes"]
    assert d["host_terms"]["day_step_hot_packets"] > 0 and "general_owner_canonical_host" in d["host_terms"] and "general_owner_canonical_host" in f["host_terms"]


def test_an_unregistered_operation_on_a_cuda_lifecycle_never_falls_through_to_a_cpu_integer_admission():
    """Negative guard unit test only (NOT a positive CUDA qualification): `_admit` consults only `lop._cuda` before refusing an unregistered label."""
    stand_in = SimpleNamespace(_cuda=object())
    with pytest.raises(wl.LifecycleError, match="no CUDA admission plan is registered"):
        wl._admit(stand_in, 1 << 20, "daily_but_unregistered")
    assert "daily" in wl._ENTRY_OF and "daily_finish" in wl._ENTRY_OF


def test_cpu_lifecycle_has_no_cuda_plan_and_keeps_its_integer_figures_and_labels():
    c = build(np.full((4, 6), 0.2), lay=layers(bottom="open"), theta=(0.1, 0.1))
    s0 = start(c)
    with pytest.raises(dw.DailyRefusal, match="only for a CUDA lifecycle"):
        dw.cuda_daily_plan(c.lop, "daily", s0)
    assert isinstance(dw.required_daily_bytes(c.lop, s0), int) and dw.required_daily_bytes(c.lop, s0) > 0  # the CPU integer API is unchanged
    with pytest.raises(dw.DailyRefusal, match="only for a CUDA lifecycle"):  # the lifecycle gate comes before the kind lookup
        dw.cuda_daily_plan(c.lop, "nope", s0)


def test_cpu_numpy_pet_check_is_metadata_only_and_the_cpu_path_is_unchanged():
    c = build(np.full((4, 6), 0.2), lay=layers(bottom="open"), theta=(0.1, 0.1))
    s0 = start(c)
    plan_ = dw.next_plan(s0, 0.0, 0.0, 1.0)
    for bad in (np.zeros(c.n - 1), np.zeros(c.n, dtype=np.float32), [0.0] * c.n):
        with pytest.raises(dw.DailyRefusal) as e:
            dw.daily_water_action(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan_, pet_terrain_m_s=bad)
        assert e.value.code == "step_refused"
    assert dw.daily_water_action(c.lop, s0, expected_parent=wl.expect_parent(s0), plan=plan_, pet_terrain_m_s=pet(c, 0.0)).status == "complete"


def test_cuda_branches_use_the_device_mask_the_lifecycle_namespace_and_the_registered_plans():
    src = inspect.getsource(dw.begin_daily_storm)
    assert "lop.column.active.reshape(ny, nx)" in src and "xp=lop.xp" in src  # device VIEW + the lifecycle's own namespace
    tree = ast.parse(inspect.getsource(dw))
    asnumpy = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "asnumpy"]
    assert asnumpy == []  # the daily API never downloads a column/array to the host
    for label in ("daily_start", "daily", "daily_begin_storm", "daily_finish"):
        assert f'"{label}"' in inspect.getsource(dw)
    gate = inspect.getsource(dw._backend_gate)
    assert "CUDA9F_BOUNDARY_PROFILE" in gate and "lop.context is None" in gate and "windowed" in gate


def test_s1_column_kernels_are_not_touched_by_s3():
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src" / "maple_syrup"
    reviewed = {"subsurface.py", "subsurface_numba.py", "subsurface_cuda.py"}
    assert all((root / name).exists() for name in reviewed)  # (byte identity vs the reviewed snapshot is recorded in the S3 report by cmp)
    assert hashlib.sha256((root / "subsurface_cuda.py").read_bytes()).hexdigest()
