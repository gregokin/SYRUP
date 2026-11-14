"""Vegetation physical-checkpoint CUDA HARDWARE definitions: EXECUTABLE, AUTHORED, NOT RUN by their author (no CuPy / GPU / NVRTC / nvidia-smi was touched; the author only compiled and host-collected them). Root runs them on its allocated
device with `SYRUP_VEG_CHECKPOINT_RELEASE_GPU=1` (device id `SYRUP_VEG_CHECKPOINT_GPU_DEVICE`, default 0; the CUDA MAPLE pin through `VEG_CHECKPOINT_MAPLE_ROOT`). Without the flag they are skipped; WITH the flag a missing CuPy / device is
a SETUP FAILURE, never a skip. No test body is a NotImplemented placeholder.

Fixtures are the qualified Phase 2 ones loaded lazily by path in `veg_lc_gpu_support` (the CUDA event<->daily bridge `matched_pair`: a genuinely completed CUDA closed Boundary, Dense and published Windowed, with a MATCHED CPU twin) plus the
REAL LEAF P1 projection and the S0 binding applied by the owning `apply_vegetation_generation`. They cover: Dense / Windowed boundary and CLOSED-day round trips (exact W / J / I / D / H / q, covers resident on the device, stable
generation counter / recipe id, cached-derived origin, bitwise continuation against the uninterrupted device run), the current-device gate before any reader / IO, the exact published two-pool load need (need - 1 / zero / reserve once), resealed
tampers and mutual refusal with the old CUDA profile, and the same-GPU FRESH-PROCESS restore of a closed day with every original constructor and every LEAF plant / projection constructor DISABLED before the static recipe
(`gpu_fresh_worker.py`). Declared limits (not claimed): pending storm / recession files, large N, a whole-wet-day allocator trace, GPU numerics beyond the declared float64 tolerance."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import veg_lc_builders as vb
import veg_lc_gpu_support as sup

RELEASE = os.environ.get("SYRUP_VEG_CHECKPOINT_RELEASE_GPU") == "1"
DEVICE = int(os.environ.get("SYRUP_VEG_CHECKPOINT_GPU_DEVICE", "0"))
pytestmark = pytest.mark.skipif(not RELEASE, reason="CUDA hardware tests: set SYRUP_VEG_CHECKPOINT_RELEASE_GPU=1 on the allocated device (authored, not run by the author)")

BIG = 1 << 40
IDENT = {"case": "s2vcuda", "v": 1}
TOL = {"rtol": 1e-9, "atol": 1e-12}
RUN_ = "ab" * 32


@pytest.fixture(scope="module")
def gpu():
    try:
        import cupy as cp

        cp.cuda.Device(DEVICE).use()  # the ONE outer device selection; production APIs only enforce the current device afterwards
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"SYRUP_VEG_CHECKPOINT_RELEASE_GPU=1 but CuPy / a CUDA device is unusable: {exc}")
    return SimpleNamespace(cp=cp, m=sup.modules())


def hst(a):
    from maple.core.backend import to_host

    return np.asarray(to_host(a))


def factory_of(ev):
    """The static cover-aware event recipe of a device fixture: `factory(vertical, topology, vegetation_cover_fraction[, schedule])`."""
    from maple_syrup.pond_event import PondEvent

    def make(vert, topology=None, cover=None, schedule=None):
        return PondEvent(ev.context, None, ev.sediment, ev.schedule if schedule is None else schedule, ev._initial_topology if topology is None else topology, control=ev.control, policy=ev.policy,
                         splash=ev.splash, vegetation_cover_fraction=cover, floating_vertical=vert)

    return make


def vegetated(gpu, leaf, tmp, windowed, days=0):
    """A genuine CUDA lineage: matched pair -> closed Boundary -> actual LEAF projection -> S0 binding -> owning apply (both pools) [-> `days` genuine bridge days]. Returns a namespace (g = the device side, c = the CPU twin)."""
    from maple_syrup import vegetation_lifecycle as vl

    bd_ = gpu.m.bd
    g, c = bd_.matched_pair(tmp, bd_.dry_kwargs(), windowed)
    s = vb.stack(leaf, g.b)
    binding = vb.binding(leaf, g.b, s)
    nb = vl.apply_vegetation_generation(g.lin, g.b, binding, memory_budget_bytes=BIG, available_bytes_fn=g.rdr, host_memory_budget_bytes=BIG, host_available_bytes_fn=lambda: BIG)
    ns = SimpleNamespace(g=g, c=c, b=nb, lin=g.lin, factory=factory_of(g.ev), original=g.ev._initial_topology, stack=s, binding=binding, closed=None, lib=bd_)
    for d in range(days):
        ns.closed = day(ns, ns.b, d)
        if d != days - 1:
            ns.b = hand(ns, ns.closed)
    return ns


def day(ns, boundary, ordinal):
    """One genuine CUDA bridge day with the BOUND E-shading cover (`cover=None`)."""
    from maple_syrup import floating_daily as fd
    from maple_syrup import floating_event_daily_bridge as br
    from maple_syrup import recharge as rc

    g, bd_ = ns.g, ns.lib
    st = br.open_day(ns.lin, boundary, partition=bd_.PART, run_digest=bd_.RUN, available_bytes_fn=g.rdr, **g.hk)
    n = st.ctx.n_cells
    res = br.day_action(st, plan=rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=bd_.RUN, ordinal=ordinal), pet_terrain_m_s=g.xp.full(n, 3e-8), cover=None, available_bytes_fn=g.rdr, **g.hk)
    t = fd.request_floating_withdrawal(res.state, expected_parent=fd.expect_parent(res.state), request_m=g.xp.full((st.ctx.n_layers, n), 1e-6), expected_sequence=0, available_bytes_fn=g.rdr, **g.hk)
    return fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), available_bytes_fn=g.rdr, **g.hk).state


def hand(ns, closed):
    from maple_syrup import floating_event_daily_bridge as br

    return br.hand_back(ns.lin, closed, available_bytes_fn=ns.g.rdr, host_available_bytes_fn=lambda: BIG)


def pools(g, **over):
    q = {"available_bytes_fn": g.rdr, "host_available_bytes_fn": lambda: BIG, "memory_budget_bytes": BIG, "host_memory_budget_bytes": BIG}
    q.update(over)
    return q


def save(ns, path, closed=None, **q):
    from maple_syrup import floating_lifecycle_checkpoint_vegetation_cuda as cv

    return cv.save_floating_lifecycle_vegetation_checkpoint_cuda(path, ns.lin, ns.factory, IDENT, closed=closed, original_topology=ns.original, **pools(ns.g, **q))


def load(gpu, ns, path, sv, partition=None, **q):
    from maple_syrup import floating_lifecycle_checkpoint_vegetation_cuda as cv

    return cv.load_floating_lifecycle_vegetation_checkpoint_cuda(path, ns.factory, IDENT, sv.manifest_sha256, leaf_package_sha256=vb.leaf_pin(), external_helpers=vb.helper_pin(), original_topology=ns.original,
                                                                 expected_device_id=int(gpu.cp.cuda.Device().id), partition=partition, **pools(ns.g, **q))


def pack(b):
    r = b.root
    return {"W": hst(r.water_state.water_m), "J": hst(r.history.wetting_m), "I": hst(r.history.cumulative_intake_m), "D": hst(r.depth_m), "H": hst(b.surface_depth_m), "qx": hst(b.qx_m2_s), "qy": hst(b.qy_m2_s)}


def same(a, b):
    assert set(a) == set(b) and all(a[k].tobytes() == b[k].tobytes() for k in a), [k for k in a if a[k].tobytes() != b[k].tobytes()]


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_boundary_round_trip_restores_current_state_covers_and_identity_and_continues_bitwise(gpu, leaf, tmp_path, windowed):
    from maple_syrup import vegetation_lifecycle as vl

    ns = vegetated(gpu, leaf, tmp_path, windowed)
    before, head0, rec0 = pack(ns.b), ns.lin.head, vl.current_record(ns.b)
    sv = save(ns, tmp_path / "ck")
    assert ns.lin.head is head0 and sv.kind == "boundary" and sv.generation == 1 and sv.recipe_id == rec0.recipe_id
    same(before, pack(ns.b))  # a save moves nothing
    r = load(gpu, ns, tmp_path / "ck", sv)
    rr = vl.current_record(r.head)
    assert r.kind == "boundary" and r.lineage is not ns.lin and r.lineage.head is r.head and (rr.generation, rr.recipe_id, rr.parent_recipe_id) == (1, rec0.recipe_id, None)
    assert rr.binding.origin == "cached_derived" and rr.backend == "cuda" and rr.splash_cover_device is not None and rr.shade_cover_device is not None
    assert hst(rr.splash_cover_device).tobytes() == np.asarray(rr.splash_cover_2d).tobytes() == hst(rec0.splash_cover_device).tobytes()
    assert r.head.root.host_active is not None and r.head.root.edges_host is not None
    same(before, pack(r.head))
    assert r.head.root.matrix_op.sha256 == ns.b.root.matrix_op.sha256 and r.head.root.pref_op.sha256 == ns.b.root.pref_op.sha256
    a, b = day(ns, ns.b, 0), day(SimpleNamespace(**{**vars(ns), "lin": r.lineage}), r.head, 0)
    assert a.execution_digest == b.execution_digest and a.digest == b.digest and hst(a.water.water_m).tobytes() == hst(b.water.water_m).tobytes()  # real native day: original vs restored (the same device)


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_closed_day_round_trip_keeps_identity_cursor_ledger_and_hands_back_equal(gpu, leaf, tmp_path, windowed):
    from maple_syrup import vegetation_lifecycle as vl

    ns = vegetated(gpu, leaf, tmp_path, windowed, days=2)
    closed = ns.closed
    before = (closed.execution_digest, closed.digest, closed.parent_digest, closed.revision, closed.request_sequence, closed.cursor, closed.ledger, closed.frame_identity())
    sv = save(ns, tmp_path / "ck", closed)
    r = load(gpu, ns, tmp_path / "ck", sv, partition=ns.lib.PART)
    c2 = r.closed
    assert (c2.execution_digest, c2.digest, c2.parent_digest, c2.revision, c2.request_sequence, c2.cursor, c2.ledger, c2.frame_identity()) == before
    assert c2.ctx.hold is r.lineage.head and vl.current_record(r.lineage.head).recipe_id == vl.current_record(ns.lin.head).recipe_id
    assert hst(c2.water.water_m).tobytes() == hst(closed.water.water_m).tobytes() and hst(c2.history.wetting_m).tobytes() == hst(closed.history.wetting_m).tobytes()
    b1, b2 = hand(ns, closed), hand(SimpleNamespace(**{**vars(ns), "lin": r.lineage}), c2)
    assert b1.ledger == b2.ledger and b1.t_s == b2.t_s and b1.cursor == b2.cursor and b1.seq == b2.seq
    d1, d2 = day(ns, b1, 2), day(SimpleNamespace(**{**vars(ns), "lin": r.lineage}), b2, 2)
    assert d1.execution_digest == d2.execution_digest and hst(d1.water.water_m).tobytes() == hst(d2.water.water_m).tobytes()


def test_current_device_gate_precedes_every_reader_path_check_and_factory(gpu, leaf, tmp_path, monkeypatch):
    from maple_syrup import floating_lifecycle_checkpoint_vegetation_cuda as cv
    from maple_syrup import floating_soil_layers_cuda as fcu

    ns = vegetated(gpu, leaf, tmp_path, False)
    sv = save(ns, tmp_path / "ck")
    reads, made = [], []

    def reader():
        reads.append(1)
        return BIG

    real = fcu.current_device
    monkeypatch.setattr(fcu, "current_device", lambda c: real(c) + 1)  # the current device is NOT the expected one: the real gate, an ordering control
    with pytest.raises(cv.VegetationLifecycleCheckpointCudaError, match="device gate"):
        cv.save_floating_lifecycle_vegetation_checkpoint_cuda(tmp_path / "never", ns.lin, ns.factory, IDENT, original_topology=ns.original, **pools(ns.g, available_bytes_fn=reader, host_available_bytes_fn=reader))
    with pytest.raises(cv.VegetationLifecycleCheckpointCudaError, match="device gate"):
        cv.load_floating_lifecycle_vegetation_checkpoint_cuda(tmp_path / "ck", lambda *a, **k: made.append(1), IDENT, sv.manifest_sha256, leaf_package_sha256=vb.leaf_pin(), external_helpers=vb.helper_pin(),
                                                              original_topology=ns.original, expected_device_id=int(gpu.cp.cuda.Device().id), **pools(ns.g, available_bytes_fn=reader, host_available_bytes_fn=reader))
    assert reads == [] and made == [] and not (tmp_path / "never").exists()


def test_two_pool_admission_need_minus_one_zero_reserve_once_and_the_published_need(gpu, leaf, tmp_path, monkeypatch):
    from maple_syrup import floating_lifecycle_checkpoint_vegetation_cuda as cv
    from maple_syrup import floating_soil_layers as fsl

    MEM = fsl.FloatingMemoryError
    ns = vegetated(gpu, leaf, tmp_path, False)
    sv = save(ns, tmp_path / "ck")
    for q, label in (({"available_bytes_fn": lambda: 0}, "device zero"), ({"host_available_bytes_fn": lambda: 0}, "host zero"), ({"memory_budget_bytes": 0}, "zero device budget"),
                     ({"host_memory_budget_bytes": 0}, "zero host budget")):
        t = tmp_path / f"r_{label.replace(' ', '_')}"
        with pytest.raises((MEM, cv.VegetationLifecycleCheckpointCudaError)):
            save(ns, t, **q)
        assert not t.exists() and not list(tmp_path.glob(f".{t.name}*")) and ns.lin.head is ns.b, label
    need = cv.vegetation_load_need_bytes_cuda(tmp_path / "ck", IDENT, sv.manifest_sha256, expected_device_id=int(gpu.cp.cuda.Device().id), **pools(ns.g))
    assert need.device_total_bytes > 0 and need.host_total_bytes > 0
    reads, real_load = [], np.load

    class Reached(BaseException):
        pass

    def trip(*a, **k):
        reads.append(1)
        raise Reached

    monkeypatch.setattr(np, "load", trip)
    for q, label in (({"available_bytes_fn": lambda: need.device_total_bytes - 1}, "device need-1"), ({"host_available_bytes_fn": lambda: need.host_total_bytes - 1}, "host need-1"),
                     ({"available_bytes_fn": lambda: 0}, "device zero"), ({"host_available_bytes_fn": lambda: 0}, "host zero")):
        with pytest.raises(MEM):
            load(gpu, ns, tmp_path / "ck", sv, **q)
        assert reads == [], label  # refused before any payload array
    with pytest.raises(Reached):  # the exact published need passes the stage admission and reaches the first array read
        load(gpu, ns, tmp_path / "ck", sv, available_bytes_fn=lambda: need.device_total_bytes, host_available_bytes_fn=lambda: need.host_total_bytes)
    monkeypatch.setattr(np, "load", real_load)


def _reseal(src, tmp, name, outer):
    from maple_syrup import floating_context_checkpoint as cc
    from maple_syrup import floating_lifecycle_checkpoint_vegetation_cuda as cv
    from maple_syrup import pond_checkpoint as pc

    d = tmp / name
    shutil.copytree(src, d)
    om = json.loads((d / cv.MANIFEST_NAME).read_text())
    outer(om)
    sealed = cc._seal(om)
    (d / cv.MANIFEST_NAME).write_text(pc._json_dumps(sealed))
    return d, sealed["manifest_sha256"]


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_resealed_tampers_wrong_pins_and_old_profile_mutual_refusal_leave_the_head_and_the_target_retryable(gpu, leaf, tmp_path, windowed):
    from maple_syrup import floating_checkpoint as fc
    from maple_syrup import floating_lifecycle_checkpoint_cuda as s2
    from maple_syrup import floating_soil_layers as fsl
    from maple_syrup import pond_checkpoint as pc

    refuse = (fc.FloatingCheckpointError, pc.PondCheckpointError, fsl.FloatingMemoryError)
    ns = vegetated(gpu, leaf, tmp_path, windowed)
    sv = save(ns, tmp_path / "ck")
    head = ns.lin.head
    cases = {
        "gen": (lambda m: m["vegetation"].update(generation=2), "recipe id|parent"),
        "recipe": (lambda m: m["vegetation"].update(recipe_id="1" * 64), "recipe id"),
        "binding": (lambda m: m["vegetation"].update(binding_sha256="3" * 64), "differs from the recipe|recipe id"),
        "kernel": (lambda m: m["context_meta"].update(kernel={"x": 1}), "kernel|invalid|metadata"),
        "profile_cpu": (lambda m: m.update(profile="maple-syrup-floating-lifecycle-checkpoint-vegetation/1"), "unknown vegetation lifecycle profile"),
        "profile_old": (lambda m: m.update(profile=s2.PROFILE), "unknown vegetation lifecycle profile"),
        "schema_cpu": (lambda m: m.update(context_schema="maple_syrup.floating_held_context/3"), "unsupported"),
        "backend": (lambda m: m.update(array_backend="numpy"), "unsupported"),
    }
    for name, (edit, rx) in cases.items():
        d, seal = _reseal(sv.path, tmp_path, name, edit)
        with pytest.raises(refuse, match=rx):
            load(gpu, ns, d, type(sv)(d, seal, sv.kind, 1, sv.recipe_id))
    with pytest.raises(refuse, match="EXPECTED"):
        load(gpu, ns, d, sv)  # the writer's seal is expected: a re-sealed file is a different file
    wrong = vb.helper_pin()
    wrong = [{**wrong[0], "sha256": "sha256:" + "0" * 64}]
    from maple_syrup import floating_lifecycle_checkpoint_vegetation_cuda as cv

    with pytest.raises(refuse, match="pins differ"):
        cv.load_floating_lifecycle_vegetation_checkpoint_cuda(sv.path, ns.factory, IDENT, sv.manifest_sha256, leaf_package_sha256=vb.leaf_pin(), external_helpers=wrong, original_topology=ns.original,
                                                              expected_device_id=int(gpu.cp.cuda.Device().id), **pools(ns.g))
    with pytest.raises(refuse, match="already exists"):
        save(ns, sv.path)  # no overwrite
    with pytest.raises(s2.LifecycleCheckpointCudaError, match="vegetation-bound head is not saveable"):
        s2.save_floating_lifecycle_checkpoint_cuda(tmp_path / "old", ns.lin, ns.factory, IDENT, original_topology=ns.original, available_bytes_fn=ns.g.rdr, host_available_bytes_fn=lambda: BIG)
    assert ns.lin.head is head and not (tmp_path / "old").exists() and not list(tmp_path.glob(".old.partial-*"))
    r = load(gpu, ns, sv.path, sv)  # the untouched file is still restorable (retry)
    assert r.kind == "boundary"


def _worker(mode, work, windowed):
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve().parent / "gpu_fresh_worker.py"), mode, str(work), "1" if windowed else "0"], capture_output=True, text=True, env=env, timeout=3600, check=False)
    assert r.returncode == 0, r.stderr[-4000:]


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_same_gpu_fresh_process_restore_without_original_bed_owner_probe_provider_or_leaf_plants(gpu, leaf, tmp_path, windowed):
    """Process A saves a vegetation-bound CLOSED day and continues one day; process B (a NEW interpreter) disables every original constructor AND every LEAF plant / projection constructor BEFORE it reads the recipe, restores from the file +
    recipe + the recorded seal + the independent LEAF pins alone, runs the same day and dumps: bitwise equal. The claim is limited to the worker's tripwires (listed in `tripwires.json`)."""
    _worker("save", tmp_path, windowed)
    _worker("load", tmp_path, windowed)
    a, b = np.load(tmp_path / "save_final.npz"), np.load(tmp_path / "load_final.npz")
    assert set(a.files) == set(b.files) and all(a[k].tobytes() == b[k].tobytes() for k in a.files)
    trip = json.loads((tmp_path / "tripwires.json").read_text())["disabled_before_recipe"]
    assert "make_pair" in trip and "project_from_initial_state" in trip
