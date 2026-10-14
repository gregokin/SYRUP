"""CW-B15 items 1 and 2: honest window memory planning with the actual class count, and the enforced CuPy pool hard
limit of a windowed CuPy run. Everything here runs on the CPU with a CuPy stand-in; actual-device behaviour (the real
pool, a real OutOfMemoryError, the Chastre first-pickup refusal) is the root's GPU0 verification."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("maple")

import maple_syrup.windowed_experiment as we
from maple_syrup.sediment_event import SedimentEventError


# --- item 2: the plan ----------------------------------------------------------------------------------------------
def expected_estimate(ny, nx, nv, nc, w, nnz):
    from maple.surface.voxels.windowed_prefix import packet_terms_for

    n = ny * nx
    window = n * w * nc * 8
    prefix = n * packet_terms_for(nv, nc) * 8
    return 4 * window + 4 * prefix + 40 * n * nc * 8 + 8 * (n * nc + 1) * 8 + 6 * nnz * 32 + n * 192, window, prefix


@pytest.mark.parametrize("nc", [1, 3, 6, 9, 13])
def test_plan_uses_the_actual_class_count(nc):
    g = SimpleNamespace(ny=7, nx=5)
    plan = we.window_memory_plan(g, 11, n_classes=nc, window_voxels=4, phase_max_nnz=1000, max_journal_records=50)
    estimate, window, prefix = expected_estimate(7, 5, 11, nc, 4, 1000)
    assert plan["estimated_device_or_array_peak_bytes"] == estimate
    assert plan["resident_window_bytes"] == window and plan["prefix_bytes"] == prefix
    assert plan["dense_voxel_bytes_not_allocated"] == 7 * 5 * 11 * nc * 8
    assert plan["journal_host_scratch_envelope_bytes"] == 50 * (nc * 8 + 16) * 12
    assert plan["n_classes"] == nc
    assert "not a conservative envelope" in plan["note"].lower() or "NOT a conservative envelope" in plan["note"]


def test_three_and_nine_classes_differ_from_the_old_hard_coded_six():
    g = SimpleNamespace(ny=4, nx=4)
    kw = {"window_voxels": 4, "phase_max_nnz": 10, "max_journal_records": 10}
    six = we.window_memory_plan(g, 8, n_classes=6, **kw)["estimated_device_or_array_peak_bytes"]
    three = we.window_memory_plan(g, 8, n_classes=3, **kw)["estimated_device_or_array_peak_bytes"]
    nine = we.window_memory_plan(g, 8, n_classes=9, **kw)["estimated_device_or_array_peak_bytes"]
    assert three < six < nine
    assert six == expected_estimate(4, 4, 8, 6, 4, 10)[0]  # six classes: the pre-B15 numbers are unchanged


@pytest.mark.parametrize("bad", [
    {"n_classes": 0}, {"n_classes": True}, {"n_classes": 2.0}, {"n_classes": None},
    {"window_voxels": 0}, {"window_voxels": 2}, {"window_voxels": 12}, {"n_voxels": 0},
    {"phase_max_nnz": 0}, {"max_journal_records": -1}, {"geometry": SimpleNamespace(ny=0, nx=3)},
    {"geometry": SimpleNamespace(ny=3, nx=True)},
])
def test_plan_refuses_invalid_dimensions(bad):
    kw = {"geometry": SimpleNamespace(ny=3, nx=3), "n_voxels": 10, "n_classes": 6, "window_voxels": 4,
          "phase_max_nnz": 10, "max_journal_records": 10}
    kw.update(bad)
    geometry, nv = kw.pop("geometry"), kw.pop("n_voxels")
    with pytest.raises(SedimentEventError):
        we.window_memory_plan(geometry, nv, **kw)


def test_plan_device_precheck_is_a_necessary_condition_only():
    g = SimpleNamespace(ny=4, nx=4)
    kw = {"n_classes": 6, "window_voxels": 4, "phase_max_nnz": 10, "max_journal_records": 10, "device_id": 0}
    estimate = expected_estimate(4, 4, 8, 6, 4, 10)[0]
    ok = we.window_memory_plan(g, 8, mem_info=lambda: (int(estimate / 0.85) + 8, 10**12), **kw)
    assert ok["device_budget_fraction"] == 0.85
    with pytest.raises(SedimentEventError, match="planning estimate alone exceeds 85%"):
        we.window_memory_plan(g, 8, mem_info=lambda: (estimate, 10**12), **kw)


# --- item 1: the pool limit (CuPy stand-in) --------------------------------------------------------------------------
class FakeOOM(MemoryError):
    """Stand-in of cupy.cuda.memory.OutOfMemoryError (a MemoryError subclass with (size, total, limit))."""


class FakePool:
    def __init__(self, limit=0, used=0, held=0):
        self.limit, self.used, self.held, self.calls = limit, used, held, []

    def get_limit(self):
        return self.limit

    def set_limit(self, size=None, fraction=None):
        assert fraction is None
        self.calls.append(size)
        self.limit = size

    def used_bytes(self):
        return self.used

    def total_bytes(self):
        return self.held

    def malloc(self, n):  # the default-pool allocator hook
        raise AssertionError("never allocates in these tests")


def fake_cupy(free=1_000_000, total=2_000_000, prior=0, allocator=None):
    pool = FakePool(limit=prior)
    current = []

    class Device:
        def __init__(self, i):
            self.i = i

        def __enter__(self):
            current.append(self.i)
            return self

        def __exit__(self, *exc):
            current.pop()

        @property
        def mem_info(self):
            return (free, total)

    cuda = SimpleNamespace(Device=Device, get_allocator=lambda: pool.malloc if allocator is None else allocator,
                           memory=SimpleNamespace(OutOfMemoryError=FakeOOM))
    return SimpleNamespace(cuda=cuda, get_default_memory_pool=lambda: pool, _pool=pool, _current=current)


def test_pool_limit_is_85_percent_of_current_free_and_restored():
    cp = fake_cupy(free=1_000_000, prior=0)
    guard = we.DevicePoolLimit(cupy_module=cp)
    record = guard.install(0)
    assert cp._pool.limit == 850_000 == record["effective_limit_bytes"] == record["computed_limit_bytes"]
    assert record["prior_limit_bytes"] == 0 and record["limit_source"].startswith("85%")
    assert "not a measured peak" in record["scope"] and record["installed"] == "before any resident array upload"
    with pytest.raises(SedimentEventError, match="already installed"):
        guard.install(0)
    assert guard.restore() == 0 and cp._pool.limit == 0 and not guard.installed
    assert guard.restore() is None  # idempotent


@pytest.mark.parametrize("prior,expected,source", [(300_000, 300_000, "stricter"), (900_000, 850_000, "85%"),
                                                   (850_000, 850_000, "85%")])
def test_pool_limit_honours_a_stricter_prior_and_never_loosens(prior, expected, source):
    cp = fake_cupy(free=1_000_000, prior=prior)
    guard = we.DevicePoolLimit(cupy_module=cp)
    record = guard.install(0)
    assert cp._pool.limit == expected and record["limit_source"].startswith(source)
    guard.restore()
    assert cp._pool.limit == prior


def test_pool_limit_refuses_a_non_default_allocator_and_changes_nothing():
    cp = fake_cupy(allocator=lambda n: None)
    guard = we.DevicePoolLimit(cupy_module=cp)
    with pytest.raises(SedimentEventError, match="not the default memory pool"):
        guard.install(0)
    assert cp._pool.calls == [] and not guard.installed


def test_pool_limit_refuses_a_read_back_mismatch_and_restores_the_prior():
    cp = fake_cupy(prior=0)
    cp._pool.get_limit = (lambda pool=cp._pool: 0 if not pool.calls else pool.limit + 1)
    guard = we.DevicePoolLimit(cupy_module=cp)
    with pytest.raises(SedimentEventError, match="reads back"):
        guard.install(0)
    assert cp._pool.calls[-1] == 0 and not guard.installed


@pytest.mark.parametrize("bad", [0.0, 1.5, True, "0.85"])
def test_pool_limit_fraction_is_validated(bad):
    with pytest.raises(SedimentEventError):
        we.DevicePoolLimit(bad)


def test_refusal_names_the_limit_and_finds_oom_anywhere_in_the_chain():
    cp = fake_cupy()
    guard = we.DevicePoolLimit(cupy_module=cp)
    guard.install(0)
    try:
        try:
            raise FakeOOM("Out of memory allocating 107,250,176 bytes")
        except FakeOOM as inner:
            raise RuntimeError("wrapped by a caller") from inner
    except RuntimeError as outer:
        refusal = guard.refusal_for(outer)
    assert isinstance(refusal, SedimentEventError) and "850000 bytes" in str(refusal)
    assert "resource refusal" in str(refusal) and "no accepted step" in str(refusal)
    assert guard.refusal_for(ValueError("other")) is None
    guard.restore()


def wrapper_harness(monkeypatch, tmp_path, cp, failure):
    """`run_complete_windowed` with the setup replaced by: install the limit as the real setup does (before upload),
    then optionally fail. `run_complete_case` is replaced by a stand-in that calls the setup."""
    seen = {}

    def factory():
        return we.DevicePoolLimit(cupy_module=cp)

    def fake_setup(**kw):
        seen["limit_during_setup"] = kw["pool_limit"].install(0)["effective_limit_bytes"]
        seen["pool_limit_seen"] = cp._pool.limit
        if failure is not None:
            raise failure()
        return {}

    def fake_run(case_dir, output_dir, *, case_setup, **options):
        case_setup(backend=options["backend"])
        return "ran"

    monkeypatch.setattr(we, "_new_pool_limit", factory)
    monkeypatch.setattr(we, "_setup_windowed", fake_setup)
    monkeypatch.setattr(we, "run_complete_case", fake_run)
    return seen


def test_wrapper_converts_oom_to_a_refusal_and_restores_the_prior_limit(monkeypatch, tmp_path):
    cp = fake_cupy(free=1_000_000, prior=123_456_789)
    seen = wrapper_harness(monkeypatch, tmp_path, cp, lambda: FakeOOM("first pickup"))
    with pytest.raises(SedimentEventError, match="resource refusal") as info:
        we.run_complete_windowed("case", tmp_path / "out", case_kind="plot1", store_dir=tmp_path / "s",
                                 backend="cupy")
    assert isinstance(info.value.__cause__, FakeOOM)
    assert seen["pool_limit_seen"] == 850_000  # a looser prior was tightened while the run was live
    assert cp._pool.limit == 123_456_789  # restored after the failed invocation
    assert not (tmp_path / "out").exists()


def test_wrapper_restores_after_success_and_after_a_non_memory_failure(monkeypatch, tmp_path):
    cp = fake_cupy(free=1_000_000, prior=0)
    wrapper_harness(monkeypatch, tmp_path, cp, None)
    assert we.run_complete_windowed("case", tmp_path / "o1", case_kind="plot1", store_dir=tmp_path / "s",
                                    backend="cupy") == "ran"
    assert cp._pool.limit == 0
    wrapper_harness(monkeypatch, tmp_path, cp, lambda: KeyError("unrelated"))
    with pytest.raises(KeyError):
        we.run_complete_windowed("case", tmp_path / "o2", case_kind="plot1", store_dir=tmp_path / "s",
                                 backend="cupy")
    assert cp._pool.limit == 0


def test_numpy_runs_create_no_pool_limit(monkeypatch, tmp_path):
    created = []
    monkeypatch.setattr(we, "_new_pool_limit", lambda: created.append(1))
    monkeypatch.setattr(we, "run_complete_case", lambda *a, case_setup, **k: "ran")
    assert we.run_complete_windowed("case", tmp_path / "o", case_kind="plot1", store_dir=tmp_path / "s",
                                    backend="numpy") == "ran"
    assert created == []


def test_cupy_setup_without_the_enforced_limit_is_refused():
    from maple_syrup.sediment_event import SedimentEventControl

    control = SedimentEventControl(phase_storage="csr", phase_max_nnz=10).validated()
    with pytest.raises(SedimentEventError, match="enforced device pool limit"):
        we._setup_windowed("case", case_kind="plot1", store_dir="s", window_voxels=4, max_store_bytes=1,
                           max_host_bytes=1, max_journal_records=1, backend="cupy", control=control,
                           expected_maple_root=None, allow_maple_source_change=True, pool_limit=None)


# --- item 1: an out-of-memory attempt publishes nothing (actual windowed MAPLE event loop, CPU) --------------------
def test_out_of_memory_attempt_publishes_no_step_no_checkpoint_and_no_deep_version(tmp_path, monkeypatch):
    from test_exchange_history import complete
    from test_windowed_storm import window_inputs

    import maple_syrup.sediment_event as event_module

    s, ctx, ctl, schedule = window_inputs(np, tmp_path, "csr")
    store = tmp_path / "store" / "versions"
    # CW-B20: the event loop calls the PRIVATE accounting step; the injection point follows it.
    original = event_module._sediment_accounting_step
    calls = {"n": 0, "versions_before_failure": None}
    boundaries = []

    def failing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:
            calls["versions_before_failure"] = sorted(p.name for p in store.iterdir())
            calls["t_before_failure"] = args[0].t_s
            raise FakeOOM("Out of memory allocating 107,250,176 bytes (limit 9,092,045,209)")
        return original(*args, **kwargs)

    monkeypatch.setattr(event_module, "_sediment_accounting_step", failing)
    with pytest.raises(FakeOOM):
        complete(s, ctx, ctl, schedule, checkpoint_callback=lambda p: boundaries.append(p.result.state.t_s) or False)
    assert sorted(p.name for p in store.iterdir()) == calls["versions_before_failure"]  # nothing published after
    assert all(t <= calls["t_before_failure"] for t in boundaries)  # no boundary/checkpoint for the failed attempt
    guard = we.DevicePoolLimit(cupy_module=fake_cupy())
    guard.install(0)
    try:
        raise FakeOOM("x")
    except FakeOOM as exc:
        assert "resource refusal" in str(guard.refusal_for(exc))
    guard.restore()
