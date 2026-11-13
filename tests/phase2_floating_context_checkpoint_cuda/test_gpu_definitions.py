"""Actual-GPU test DEFINITIONS of the CUDA general held-context container (WRITTEN, NOT RUN by their author: no GPU/CuPy/NVRTC/nvidia-smi was touched). Opt-in (SYRUP_RELEASE_GPU=1); Root runs them on GPU0 with `.venv-cupy`.
Originals are the qualified E5 device events (Dense and genuine native Windowed); every restore uses the caller's STATIC recipe (`gpu_static_case.factory`, primitive parameters only) on the verified H0 `/2` context with the
original bed/owner/probe/provider builders DISABLED in the fresh process (the ACTUAL loaded native bed of the child is allowed). Child physical contracts (current W/J/I/D, provisional scope, adopt-last) are the Dense/Windowed
child's own, qualified separately; this file adds the container contracts. Windowed beds follow the corrected residency claim: exact window/full-V and external deep references, no false total-checkpoint reduction."""
# ruff: noqa: RUF059
from __future__ import annotations

import dataclasses
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from case import snapshot

_T = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("child_gpu_defs_for_context", _T / "phase2_floating_checkpoint_cuda" / "test_gpu_definitions.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

import gpu_static_case as gsc
import moderate_bed as mbd
import static_case as sc

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_checkpoint_cuda as fcc
from maple_syrup import floating_context_checkpoint as cc
from maple_syrup import floating_context_checkpoint_cuda as ccc
from maple_syrup import floating_pond as fp
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import pond_checkpoint as pc

BIG = 1 << 40
IDENT = {"case": "gf4", "version": 1}
MEM = fsl.FloatingMemoryError
CONTAINER_ERRORS = (ccc.ContextCheckpointCudaError, fc.PondCheckpointError, fcc.FloatingCheckpointCudaError, fp.FloatingPondError, fsl.FloatingSoilError)


@pytest.fixture(scope="module")
def cp():
    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA context container is not exercised (no claim)")
    return backend.cupy_module()


def dev_id(cp):
    return int(cp.cuda.Device().id)


def case_kwargs(windowed):
    return {"surface": sc.terrain(), "depth": 0.004, "bottom": "closed", "saturation": 0.97, "splash": True, "rain_mm_h": 300.0, "end_s": 1000.0, "max_dt_s": 0.2, "windowed": windowed}


def original(cp, tmp, windowed):
    ev, st, vert, _o = base.original(cp, tmp, **case_kwargs(windowed))
    return ev, st, vert


def save_container(cp, tmp, windowed, k=10, name="ctx", **qkw):
    ev, st, vert = original(cp, tmp / "o", windowed)
    st = base.run(ev, st, k)
    ck = tmp / name
    ccc.save_floating_context_checkpoint_cuda(ck, st, ev, IDENT, **qkw)
    return ev, st, vert, ck


def load(cp, ck, windowed, **kw):
    return ccc.load_floating_context_checkpoint_cuda(ck, gsc.factory(windowed), IDENT, expected_device_id=dev_id(cp), **kw)


def reseal_container(ck, *, arrays=None, meta=None, outer=None):
    """Edit the H0 arrays / H0 meta / outer manifest of a REAL container and RECOMPUTE the arrays-file hash/size and the outer seal (the child is untouched)."""
    data = json.loads((ck / cc.MANIFEST_NAME).read_text())
    if arrays:
        with np.load(ck / cc.ARRAYS_NAME, allow_pickle=False) as z:
            held = {k: np.array(z[k]) for k in z.files}
        for name, fn in arrays.items():
            held[name] = fn(held[name])
        (ck / cc.ARRAYS_NAME).unlink()
        np.savez(ck / cc.ARRAYS_NAME, **held)
    if meta:
        meta(data["context_meta"])
    if outer:
        outer(data)
    data["files"] = cc._file_table(ck, (cc.ARRAYS_NAME,), BIG)
    (ck / cc.MANIFEST_NAME).write_text(pc._json_dumps(cc._seal(data)))


def reseal_child(ck, *, arrays=None, native=None):
    """Edit the child's companion/native members, RECOMPUTE the child file table, the child manifest hashes and the outer seal."""
    cdir = ck / cc.CHILD_DIR
    cman = json.loads((cdir / fc.MANIFEST_NAME).read_text())
    for fname, edits in ((pc.PAYLOADS[1], arrays), (pc.PAYLOADS[0], native)):
        if edits:
            with np.load(cdir / fname, allow_pickle=False) as z:
                data = {k: np.array(z[k]) for k in z.files}
            for member, fn in edits.items():
                data[member] = fn(data[member])
            (cdir / fname).unlink()
            np.savez(cdir / fname, **data)
    cman["files"] = {n: pc._hash_file(cdir / n, BIG)[0] for n in pc.PAYLOADS}
    (cdir / fc.MANIFEST_NAME).write_text(pc._json_dumps(fc._seal(cman)))
    outer = json.loads((ck / cc.MANIFEST_NAME).read_text())
    cfiles = cc._file_table(cdir, tuple(sorted(cc.CHILD_NAMES)), BIG)
    outer["child"] = {"schema": cman["schema"], "manifest_sha256": cfiles[fc.MANIFEST_NAME]["sha256"], "files": cfiles}
    (ck / cc.MANIFEST_NAME).write_text(pc._json_dumps(cc._seal(outer)))


def untouched_ctx(factory_vertices):
    return all(not v._registry and v._provisional is None for v in factory_vertices)


class Recorder:
    """A static factory wrapper recording every call and the verticals it produced (the container must not call it before the preflight/hash/H0 identity stages succeed)."""

    def __init__(self, windowed):
        self.calls, self.verticals, self.inner = 0, [], gsc.factory(windowed)

    def __call__(self, ctx):
        self.calls += 1
        ev = self.inner(ctx)
        self.verticals.append(ev.vertical)
        return ev


# ---- round trip, bit-exact continuation -----------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("windowed", [False, True])
def test_container_roundtrip_with_the_static_factory_restores_the_current_payload_and_continues_bit_exactly(cp, tmp_path, windowed):
    ev, st, vert, ck = save_container(cp, tmp_path, windowed)
    man = json.loads((ck / cc.MANIFEST_NAME).read_text())
    assert man["profile"] == ccc.PROFILE and set(man) == ccc.MANIFEST_KEYS and man["array_backend"] == "cupy" and man["hydraulic_backend"] == "cuda" and man["context_meta"]["schema"] == "maple_syrup.floating_held_context/2"
    assert {p.name for p in ck.iterdir()} == {cc.MANIFEST_NAME, cc.ARRAYS_NAME, cc.CHILD_DIR} and man["child"]["schema"] == (fcc.GPU_WINDOWED_SCHEMA if windowed else fcc.GPU_SCHEMA)
    need = ccc.context_load_need_bytes_cuda(ck, IDENT, expected_device_id=dev_id(cp))
    ev2, got = load(cp, ck, windowed)
    assert ev2.vertical._owner is None and ev2.vertical._initial is None and len(ev2.vertical._registry) == 1 and need.device_bytes > 0 and need.host_bytes > 0
    base.same(snapshot(ev2, got, ev2.vertical), snapshot(ev, st, vert))
    live, rest = base.run(ev, st, 6), base.run(ev2, got, 6)
    base.same(snapshot(ev2, rest, ev2.vertical), snapshot(ev, live, vert))  # restored vs uninterrupted: bitwise
    a = ev2.audit(rest)
    assert a["water"]["closes"] and a["sediment"]["closes"]
    if windowed:  # corrected residency claim: exact window / full-V on the CHILD's native member headers, external deep store, true component bytes recorded (no false total-checkpoint saving)
        child_man = json.loads((ck / cc.CHILD_DIR / fc.MANIFEST_NAME).read_text())
        comp = mbd.residency_components(ck / cc.CHILD_DIR, child_man, *sc.terrain().shape, 6)
        print("CONTAINER_WINDOWED_RESIDENCY_COMPONENTS", json.dumps(comp, sort_keys=True))
        assert need.stages["child"]["cache_bytes"] == fc.deep_cache_bound_bytes(child_man["deep_storage"], 6)


@pytest.mark.parametrize("windowed", [False, True])
def test_same_gpu_fresh_process_restores_from_the_container_without_the_original_bed_owner_or_probe(cp, tmp_path, windowed):
    ev, st, vert, ck = save_container(cp, tmp_path, windowed)
    (tmp_path / "case_identity.json").write_text(json.dumps(IDENT))
    live = base.run(ev, st, 8)
    np.savez(tmp_path / "live_final.npz", **snapshot(ev, live, vert))
    worker = Path(__file__).resolve().parent / "gpu_fresh_worker.py"
    proc = subprocess.run([sys.executable, str(worker), str(tmp_path), "8", str(dev_id(cp)), *(["windowed"] if windowed else [])], capture_output=True, text=True, timeout=1800, check=False)
    assert proc.returncode == 0, proc.stderr[-4000:]
    with np.load(tmp_path / "live_final.npz") as a, np.load(tmp_path / "restored_final.npz") as b:
        base.same({k: a[k] for k in a.files}, {k: b[k] for k in b.files})


# ---- two pools, gate first, exact budgets ---------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("windowed", [False, True])
def test_two_pool_zero_need_minus_one_exact_and_reserve_once_before_any_array_or_factory(cp, tmp_path, windowed):
    ev, st, vert, ck = save_container(cp, tmp_path, windowed, k=4)
    d = dev_id(cp)
    n = ccc.context_load_need_bytes_cuda(ck, IDENT, expected_device_id=d, reserve_bytes=0, host_reserve_bytes=0)
    assert n.device_total_bytes == n.device_bytes and n.host_total_bytes == n.host_bytes + fcu.PY_ALLOWANCE_BYTES  # the 64 KiB allowance is explicit
    n2 = ccc.context_load_need_bytes_cuda(ck, IDENT, expected_device_id=d, reserve_bytes=7, host_reserve_bytes=11)
    assert (n2.device_total_bytes, n2.host_total_bytes) == (n.device_total_bytes + 7, n.host_total_bytes + 11)  # each reserve ONCE
    dev_ok, host_ok = n.device_total_bytes, n.host_total_bytes
    for kw, label in (({"available_bytes_fn": lambda: 0, "host_available_bytes_fn": lambda: BIG}, "device zero"), ({"available_bytes_fn": lambda: BIG, "host_available_bytes_fn": lambda: 0}, "host zero"),
                      ({"available_bytes_fn": lambda: dev_ok - 1, "host_available_bytes_fn": lambda: BIG}, "device need-1"), ({"available_bytes_fn": lambda: BIG, "host_available_bytes_fn": lambda: host_ok - 1}, "host need-1"),
                      ({"available_bytes_fn": lambda: dev_ok, "host_available_bytes_fn": lambda: BIG, "reserve_bytes": 1}, "device reserve once"),
                      ({"available_bytes_fn": lambda: BIG, "host_available_bytes_fn": lambda: host_ok, "host_reserve_bytes": 1}, "host reserve once")):
        rec = Recorder(windowed)
        with pytest.raises(MEM):
            ccc.load_floating_context_checkpoint_cuda(ck, rec, IDENT, expected_device_id=d, **kw)
        assert rec.calls == 0, label  # refused BEFORE any array, H0 reconstruct or static-factory call
    rec = Recorder(windowed)
    ccc.load_floating_context_checkpoint_cuda(ck, rec, IDENT, expected_device_id=d, available_bytes_fn=lambda: dev_ok, host_available_bytes_fn=lambda: host_ok)  # exactly the reported need loads
    assert rec.calls == 1


def test_the_current_device_gate_precedes_every_reader_hash_and_callback(cp, tmp_path, monkeypatch):
    ev, st, vert, ck = save_container(cp, tmp_path, False, k=2)
    reads = []
    rec = Recorder(False)
    real = fcu.current_device
    monkeypatch.setattr(fcu, "current_device", lambda c: real(c) + 1)  # ORDERING control with the real gate on one UUID (not a second-device qualification)
    with pytest.raises(ccc.ContextCheckpointCudaError, match="CUDA device gate refused"):
        ccc.load_floating_context_checkpoint_cuda(ck, rec, IDENT, expected_device_id=real(cp), available_bytes_fn=lambda: reads.append(1) or BIG, host_available_bytes_fn=lambda: reads.append(2) or BIG)
    with pytest.raises(ccc.ContextCheckpointCudaError, match="CUDA device gate refused"):
        ccc.context_load_need_bytes_cuda(ck, IDENT, expected_device_id=real(cp), available_bytes_fn=lambda: reads.append(3) or BIG)
    assert reads == [] and rec.calls == 0
    monkeypatch.undo()
    with pytest.raises(ccc.ContextCheckpointCudaError, match="CUDA device gate refused"):  # a wrong EXPLICIT expected device is refused by the real gate
        ccc.load_floating_context_checkpoint_cuda(ck, rec, IDENT, expected_device_id=real(cp) + 1)
    assert rec.calls == 0


# ---- late failures, retry, tamper ---------------------------------------------------------------------------------------------------------------------------------------------------
def test_a_late_factory_failure_or_a_late_validator_failure_leaves_the_source_untouched_and_the_load_retryable(cp, tmp_path):
    ev, st, vert, ck = save_container(cp, tmp_path, False, k=6)
    d = dev_id(cp)

    def failing_factory(ctx):
        gsc.factory(False)(ctx)
        raise RuntimeError("injected late factory failure")

    with pytest.raises(RuntimeError, match="injected late factory failure"):
        ccc.load_floating_context_checkpoint_cuda(ck, failing_factory, IDENT, expected_device_id=d)
    holder = {}

    def audit_failing_factory(ctx):
        event = gsc.factory(False)(ctx)
        holder["v"] = event.vertical
        event.audit = lambda state: (_ for _ in ()).throw(fp.FloatingPondError("injected late validator failure"))
        return event

    with pytest.raises(fc.PondCheckpointError, match="independent validation"):
        ccc.load_floating_context_checkpoint_cuda(ck, audit_failing_factory, IDENT, expected_device_id=d)
    assert untouched_ctx([holder["v"]])
    ev2, got = load(cp, ck, False)  # the SAME container loads after both failures
    base.same(snapshot(ev2, got, ev2.vertical), snapshot(ev, st, vert))


@pytest.mark.parametrize("what", ["held_fc", "scalar_vegetation_factor", "scalar_suction", "config_identity"])
def test_raw_h0_property_tamper_with_recomputed_outer_hashes_fails_identity_reconstruction_before_the_factory(cp, tmp_path, what):
    ev, st, vert, ck = save_container(cp, tmp_path, False, k=4)
    if what == "held_fc":
        reseal_container(ck, arrays={"theta_field_capacity": lambda a: np.nextafter(a, np.inf)})
    elif what == "scalar_vegetation_factor":
        reseal_container(ck, meta=lambda m: m["scalars"].update(matrix_vegetation_factor=0.71.hex()))
    elif what == "scalar_suction":
        reseal_container(ck, meta=lambda m: m["scalars"].update(matrix_suction_m=0.11.hex()))
    else:
        reseal_container(ck, meta=lambda m: m["identities"].update(config_sha256="9" * 64))
    rec = Recorder(False)
    with pytest.raises(CONTAINER_ERRORS):
        ccc.load_floating_context_checkpoint_cuda(ck, rec, IDENT, expected_device_id=dev_id(cp))
    assert rec.calls == 0  # refused by the H0 re-preparation identity (or the cross-binding) BEFORE the static factory ran


@pytest.mark.parametrize("what", ["source", "kernel", "layout", "storage", "child_schema"])
def test_source_kernel_layout_storage_tamper_with_recomputed_outer_hashes_is_refused_before_any_array(cp, tmp_path, what):
    ev, st, vert, ck = save_container(cp, tmp_path, False, k=3)
    edits = {"source": {"outer": lambda d: d["source"].update(maple_package_sha256="0" * 64)}, "kernel": {"meta": lambda m: m["kernel"].update(fmc_source_sha256="0" * 64)},
             "layout": {"meta": lambda m: m["layouts"].update(theta_saturation="scalar" if m["layouts"]["theta_saturation"] != "scalar" else "cell")}, "storage": {"outer": lambda d: d.update(bed_storage="windowed")},
             "child_schema": {"outer": lambda d: d["child"].update(schema=fcc.GPU_WINDOWED_SCHEMA)}}[what]
    reseal_container(ck, **edits)
    rec = Recorder(False)
    with pytest.raises(CONTAINER_ERRORS):
        ccc.load_floating_context_checkpoint_cuda(ck, rec, IDENT, expected_device_id=dev_id(cp))
    assert rec.calls == 0


def test_a_missing_external_deep_store_and_a_prefix_corruption_are_refused_with_an_untouched_retryable_target(cp, tmp_path, monkeypatch):
    from maple.core import backend as mb

    ev, st, vert, ck = save_container(cp, tmp_path, True, k=6)
    store = Path(json.loads((ck / cc.CHILD_DIR / fc.MANIFEST_NAME).read_text())["deep_storage"]["store_dir"])
    moved = store.with_name(store.name + "_moved")
    store.rename(moved)
    rec = Recorder(True)
    try:
        with pytest.raises(CONTAINER_ERRORS):
            ccc.load_floating_context_checkpoint_cuda(ck, rec, IDENT, expected_device_id=dev_id(cp))
        assert untouched_ctx(rec.verticals)
    finally:
        moved.rename(store)
    bad = tmp_path / "bad"
    shutil.copytree(ck, bad)
    reseal_child(bad, native={"prefix_packet": lambda a: a * 1.5 + 1.0})  # recomputed child + outer hashes: only the native HOST load can refuse it
    placed = []
    real = mb.to_device_tree
    monkeypatch.setattr(mb, "to_device_tree", lambda *a, **k: placed.append(1) or real(*a, **k))
    rec2 = Recorder(True)
    with pytest.raises(CONTAINER_ERRORS):
        ccc.load_floating_context_checkpoint_cuda(bad, rec2, IDENT, expected_device_id=dev_id(cp))
    assert placed == [] and untouched_ctx(rec2.verticals)  # nothing was placed on the device
    monkeypatch.undo()
    ev2, got = load(cp, ck, True)  # the genuine container still loads
    base.same(snapshot(ev2, got, ev2.vertical), snapshot(ev, st, vert))


def test_save_is_atomic_no_clobber_and_a_parent_fsync_failure_after_publication_is_indeterminate_and_preserved(cp, tmp_path, monkeypatch):
    ev, st, vert = original(cp, tmp_path / "o", False)
    st = base.run(ev, st, 3)
    ck = tmp_path / "ctx"
    ccc.save_floating_context_checkpoint_cuda(ck, st, ev, IDENT)
    with pytest.raises(ccc.ContextCheckpointCudaError, match="already exists"):
        ccc.save_floating_context_checkpoint_cuda(ck, st, ev, IDENT)
    before = sorted(p.name for p in ck.iterdir())
    calls = {"n": 0}
    real = cc._fsync_dir

    def flaky(path):
        calls["n"] += 1
        if Path(path) == tmp_path:  # the PARENT directory fsync after publication
            raise OSError("injected parent fsync failure")
        return real(path)

    monkeypatch.setattr(cc, "_fsync_dir", flaky)
    ck2 = tmp_path / "ctx2"
    with pytest.raises(ccc.ContextCheckpointCudaError, match="INDETERMINATE"):
        ccc.save_floating_context_checkpoint_cuda(ck2, st, ev, IDENT)
    assert ck2.is_dir() and {p.name for p in ck2.iterdir()} == {cc.MANIFEST_NAME, cc.ARRAYS_NAME, cc.CHILD_DIR} and sorted(p.name for p in ck.iterdir()) == before  # published directory preserved
    assert list(tmp_path.glob(".ctx*")) == []


# ---- moderate N = 50003: true allocator peak, released plateau, host cold peak (scoped) -------------------------------------------------------------------------------------------
def test_moderate_n50003_true_allocator_peak_inside_the_published_need_and_the_plateau_is_released(cp, tmp_path, monkeypatch):
    """Genuine native Dense bed (`moderate_bed.build`: public constructors, NV = 60, 0.05 m terrain) saved through the container; the load uses a static factory with the SAME primitive terrain. Observer = the accepted one: pool
    sampled at EVERY allocator call + the measured rounding, NO slack; model = the published DEVICE need (`context_load_need_bytes_cuda`). The caller's factory runs INSIDE the measured region, so its device allocations (the vertical's two zero rows,
    which the model includes, and the small static context) are measured; its remaining setup is admitted by its own quota and is a declared exclusion of the model. Host cold peak = tracemalloc (Python/NumPy allocations only; interpreter/JIT/driver excluded). A peak above the model is
    a model underestimate and fails; nothing is absorbed. Fixture arithmetic is unrun."""
    import fpond_builders as fb
    from maple.core.backend import to_device, to_device_tree

    from maple_syrup import floating_held_context_cuda as fhc
    from maple_syrup import native_floating_soil as nfs
    from maple_syrup.pond_event import PondEvent
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters
    from maple_syrup.splash import splash_parameters

    wh = fb._windowed_helpers()
    dense, ctx, topo, _z = mbd.build(1, 50003, 60)
    bed = to_device_tree(dense, cp)
    dctx = dataclasses.replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, cp))
    owner = nfs.initialize_native_floating_soil(fb.native_bed(bed, dctx), fb.config(backend="cuda", bottom="open"), initial_relative_saturation=0.5)
    vert = fp.FloatingPondVertical(owner, available_bytes_fn=fcu.device_reader(cp))
    be = type("B", (), {"hydraulics": "cuda", "xp": cp})()
    ev = PondEvent(dctx, None, plot1_sediment_parameters(xp=cp), constant_rainfall(0.0, 1000.0, 300.0), topo, control=wh.control_for(be, windowed=False, max_dt_s=0.2),
                   splash=splash_parameters(np.linspace(0.4, 3.0, fb.NC), xp=cp), floating_vertical=vert)
    st = base.run(ev, ev.initial_state(bed, vert.initial_soil_water()), 2, rate=100.0 / 3.6e6)
    ck = tmp_path / "ctx"
    ccc.save_floating_context_checkpoint_cuda(ck, st, ev, IDENT)
    del fhc

    class Static:
        """The static recipe for the moderate terrain: the shared primitive globals are set only while the factory runs (as in `moderate_bed.build`)."""

        def __init__(self):
            self.inner = gsc.factory(False)
            self.saved = None

        def __call__(self, ctx_):
            saved = (dict(sc.TERRAIN), sc.CASE["z"])
            try:
                sc.TERRAIN.update(ny=1, nx=50003)
                sc.CASE["z"] = mbd.elevation(1, 50003)
                return self.inner(ctx_)
            finally:
                sc.TERRAIN.clear()
                sc.TERRAIN.update(saved[0])
                sc.CASE["z"] = saved[1]

    pool = cp.get_default_memory_pool()
    n = ccc.context_load_need_bytes_cuda(ck, IDENT, expected_device_id=dev_id(cp))

    class Peak:
        def __init__(self):
            self.base, self.peak, self.rounding = int(pool.used_bytes()), 0, 0

        def __call__(self, size):
            ptr = pool.malloc(size)
            self.peak = max(self.peak, int(pool.used_bytes()) - self.base)
            self.rounding += int(ptr.mem.size) - int(size)
            return ptr

    plateau = []
    for rep in range(3):
        gc.collect()
        cp.cuda.Device().synchronize()
        hook = Peak()
        if rep == 0:
            tracemalloc.start()
        with cp.cuda.using_allocator(hook):
            ev2, got = ccc.load_floating_context_checkpoint_cuda(ck, Static(), IDENT, expected_device_id=dev_id(cp))
        if rep == 0:
            _cur, host_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert host_peak <= n.host_bytes, (host_peak, n.host_bytes)  # cold host peak (Python/NumPy allocations only) vs the declared host need
        assert hook.peak <= n.device_bytes + hook.rounding, (hook.peak, n.device_bytes, hook.rounding)
        del got, ev2
        gc.collect()
        cp.cuda.Device().synchronize()
        plateau.append(int(pool.used_bytes()))
    assert max(plateau) - min(plateau) <= hook.rounding, plateau
