"""Actual-GPU test DEFINITIONS of the CUDA continued-origin child profile (WRITTEN, NOT RUN by their author: no GPU/CuPy/NVRTC/nvidia-smi was touched; no device pass is claimed). Opt-in (SYRUP_RELEASE_GPU=1).

DEPENDENCY, flagged: the fixtures need the GENUINE CUDA lifecycle (`floating_event_lifecycle_cuda`, Root's frozen C1/C5 component) and the GPU selected refresh, which are NOT in this candidate's baseline. Every test therefore skips with
that explicit reason until Root composes them (nothing is copied or faked).

FIXTURE: the EXACT deep pair (`e5.device_event` == `fb.make_pair` geometry) with the ABOVE-threshold 1 kg terminal deposit and the declared component policy of `cc2_fixtures` (the previous 0.1 kg deposit rose the bed 0.00032 m, BELOW the
0.00075 m trigger: generation 0, no crossing). Before ANY device claim, `Lin` builds the genuine matched CPU generation-1 control (`cc2_fixtures.cpu_control`, host-verified in `test_cpu_control.py`), proves the matched landscape
(`refresh_fixtures.assert_matched`), the recipe digest, the deposit rise above the trigger, and that the device lifecycle reaches the same generation with the same global datum. Old GPU helpers (`retamper`, `retamper_native`, `untouched`,
admission recorder, `deep_dir`) are REUSED from the accepted `phase2_floating_checkpoint_cuda` definitions, not re-implemented."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

_T = Path(__file__).resolve().parents[1]
BIG = 1 << 40
IDENT = {"case": "cc2", "v": 1}
RAIN = 30.0 / 3.6e6
NO_LC = importlib.util.find_spec("maple_syrup.floating_event_lifecycle_cuda") is None


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _T / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cp():
    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA continued profile is not exercised (no claim)")
    if NO_LC:
        pytest.skip("UNAVAILABLE DEPENDENCY: maple_syrup.floating_event_lifecycle_cuda (and the GPU selected refresh) are not in this tree; Root must compose the qualified GPU lifecycle first (no device pass claimed)")
    return backend.cupy_module()


def _old():
    return _load("old_cuda_file_defs_cc2", "phase2_floating_checkpoint_cuda/test_gpu_definitions.py")


class Lin:
    """A genuine CUDA continued state through the PUBLIC lifecycle on the exact deep pair, proven matched to the genuine CPU generation-1 control BEFORE use."""

    def __init__(self, cp, tmp, windowed):
        import cc2_fixtures as f

        from maple_syrup import floating_event_lifecycle as lc
        from maple_syrup import floating_soil_layers_cuda as fcu

        self.cp, self.lc, self.windowed = cp, lc, windowed
        self.rd = _load("refresh_gpu_defs_cc2", "phase2_floating_event_refresh_cuda/test_gpu_definitions.py")
        self.cpu = f.cpu_control(Path(tmp) / "cpu", windowed)  # the genuine matched CPU generation-1 control FIRST
        ev, st0, vert0, owner = self.rd.e5.device_event(Path(tmp) / "dev", windowed=windowed, **f.lake_kwargs())
        self.ev0, self.original = ev, ev._initial_topology
        self.rd.rf.assert_matched(self.cpu.ev0, self.cpu.st0, self.cpu.vert0, ev, st0, vert0, windowed)  # matched geometry / strata / held maps / D0 BEFORE any claim
        assert f.deposit_rise_m(ev.context.geometry) > f.THRESHOLD_M and f.recipe_digest() == f.recipe_digest()
        self.factory = f.make_factory(ev)
        self.reader = fcu.device_reader(self.rd.e5._cp())
        kw = {"available_bytes_fn": self.reader, "host_available_bytes_fn": lambda: BIG}
        lin, occ, state = lc.open_floating_event(owner, st0.bed, self.factory, **kw)
        from maple_syrup import floating_held_context_cuda as fhc

        self.record0 = fhc.export_held_context_cuda(occ.vertical, None, available_bytes_fn=self.reader)  # the ORIGINAL-root H0 record (before any refresh): SAVE-side recipe verification
        h = f.settle(occ.event, state)
        self.boundary = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, **kw)
        self.record = lc.export_boundary_held_context(self.boundary, None, **kw)
        self.occ, self.state = lc.begin_occurrence(lin, self.boundary, self.factory, **kw)
        self.lin, self.event = lin, self.occ.event
        c = self.cpu.state
        assert self.state.topology.generation == c.topology.generation >= 1 and self.state.counters.n_refreshes == 0 == c.counters.n_refreshes  # the device reaches the CPU control's generation
        assert self.rd.host(self.state.datum.initial_full_m).tobytes() == np.asarray(c.datum.initial_full_m).tobytes()  # same GLOBAL datum
        assert not np.array_equal(np.asarray(c.datum.initial_full_m), self.event._initial_topology.source_z_full)  # datum != carried routing source


def target(L, ck, ccc, record=None):
    from maple_syrup import floating_context_checkpoint as fcx
    from maple_syrup import floating_held_context as fh
    from maple_syrup import floating_pond as fp

    rec = L.record if record is None else record
    ctx = fh.reconstruct_held_context(rec, fcx._expected_from(rec.meta()), available_bytes_fn=L.reader, host_available_bytes_fn=lambda: BIG)
    vert = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=L.reader, host_available_bytes_fn=lambda: BIG)
    origin = ccc.read_continued_origin_cuda(ck, IDENT, available_bytes_fn=lambda: BIG)
    return L.factory(vert, ccc.origin_topology(origin, L.original)), vert


def save(L, path, **q):
    from maple_syrup import floating_continued_checkpoint_cuda as ccc

    q = {"available_bytes_fn": L.reader, "host_available_bytes_fn": lambda: BIG, **q}
    return ccc.save_floating_continued_checkpoint_cuda(path, L.state, L.event, IDENT, original_topology=L.original, **q)


def load(L, ck, ev, **q):
    from maple_syrup import floating_continued_checkpoint_cuda as ccc

    q = {"available_bytes_fn": L.reader, "host_available_bytes_fn": lambda: BIG, **q}
    return ccc.load_floating_continued_checkpoint_cuda(ck, ev, IDENT, original_topology=L.original, **q)


def same(a, b):
    assert set(a) == set(b)
    for k in a:
        if "window_cache" not in k:
            assert a[k].tobytes() == b[k].tobytes(), k


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_gen1_local0_roundtrip_schema_and_real_steps_bitwise_and_close_to_the_cpu_control(cp, tmp_path, windowed):
    from case import snapshot

    from maple_syrup import floating_continued_checkpoint_cuda as ccc

    L = Lin(cp, tmp_path, windowed)
    ck = save(L, tmp_path / "ck")
    assert L.lin.head is L.occ  # a save neither advances nor consumes the head
    man = json.loads((ck / "floating_pond_checkpoint.json").read_text())
    assert man["schema"] == (ccc.CONT_GPU_WINDOWED_SCHEMA if windowed else ccc.CONT_GPU_SCHEMA) and man["schema"] in (ccc.CONT_GPU_SCHEMA, ccc.CONT_GPU_WINDOWED_SCHEMA)  # parenthesised: BOTH branches are really tested
    assert man["schema"] != (ccc.CONT_GPU_SCHEMA if windowed else ccc.CONT_GPU_WINDOWED_SCHEMA) and man["profile"] == ccc.CONT_GPU_PROFILE
    assert man["continued_origin"]["routing_generation"] == L.state.topology.generation
    ev2, v2 = target(L, ck, ccc)
    s2 = load(L, ck, ev2)
    assert s2.topology.generation == L.state.topology.generation and s2.counters.n_refreshes == 0 and v2._owner is None
    a, b, c = L.state, s2, L.cpu.state
    for _ in range(4):
        a = L.event.step(a, RAIN, 0.1).state
        b = ev2.step(b, RAIN, 0.1).state
        c = L.cpu.event.step(c, RAIN, 0.1).state
    same(snapshot(L.event, a, L.occ.vertical), snapshot(ev2, b, v2))  # GPU-restored vs uninterrupted-GPU continuation: bitwise
    np.testing.assert_allclose(L.rd.host(a.hydraulic.depth_m), np.asarray(c.hydraulic.depth_m), rtol=1e-9, atol=1e-12)  # the matched CPU control continues to the same fields (existing contract)
    assert a.counters.n_refreshes == c.counters.n_refreshes == 0


def test_current_w_j_i_d_cross_in_bounded_counted_chunks_and_only_the_attributed_origin_array_is_new(cp, tmp_path, monkeypatch):
    from maple.core import backend as mb

    from maple_syrup import floating_checkpoint_cuda as fcc
    from maple_syrup import floating_continued_checkpoint_cuda as ccc
    from maple_syrup import floating_soil_layers_cuda as fcu

    L = Lin(cp, tmp_path, False)
    Ls, N = L.occ.vertical.n_layers, L.occ.vertical.n_cells
    chunks, flag = [], {"on": False}
    real_d2h, real_to_host = fcc.d2h, mb.to_host

    def d2h(x):
        flag["on"] = True
        try:
            return real_d2h(x)
        finally:
            flag["on"] = False

    def to_host(x, *a, **k):
        out = real_to_host(x, *a, **k)
        if flag["on"]:
            chunks.append(int(np.asarray(out).nbytes))
        return out

    monkeypatch.setattr(fcc, "d2h", d2h)
    monkeypatch.setattr(mb, "to_host", to_host)
    mb.reset_transfer_counters()
    ck = save(L, tmp_path / "ck")
    c = mb.read_transfer_counters()
    floating = (Ls * N + 3 * N) * 8  # W, J, I, D
    assert sum(chunks) >= floating and max(chunks) <= fcu.HASH_CHUNK_BYTES and c.device_to_host_bytes >= sum(chunks)  # W/J/I/D move in bounded counted chunks; the origin array is HOST data (no D2H)
    table = json.loads((ck / "floating_pond_checkpoint.json").read_text())["payload"]["arrays"]
    assert "co_routing_source" in table and ccc.CONT_GPU_SCHEMA


def test_the_device_gate_precedes_every_reader_and_path_check_for_save_and_load(cp, tmp_path, monkeypatch):
    from maple_syrup import floating_checkpoint_cuda as fcc
    from maple_syrup import floating_continued_checkpoint_cuda as ccc
    from maple_syrup import floating_soil_layers_cuda as fcu

    L = Lin(cp, tmp_path, False)
    ck = save(L, tmp_path / "ck")
    ev2, _v2 = target(L, ck, ccc)
    reads = []

    def reader():
        reads.append(1)
        return BIG

    real = fcu.current_device
    monkeypatch.setattr(fcu, "current_device", lambda c: real(c) + 1)  # the current device is NOT the vertical's device (the real gate, ordering control)
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="CUDA device gate refused"):
        ccc.save_floating_continued_checkpoint_cuda(tmp_path / "never", L.state, L.event, IDENT, original_topology=L.original, available_bytes_fn=reader, host_available_bytes_fn=reader)
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="CUDA device gate refused"):
        ccc.load_floating_continued_checkpoint_cuda(ck, ev2, IDENT, original_topology=L.original, available_bytes_fn=reader, host_available_bytes_fn=reader)
    assert reads == [] and not (tmp_path / "never").exists()


def test_two_pool_admission_zero_need_minus_one_exact_reserve_once_and_header_derived_needs(cp, tmp_path, monkeypatch):
    from maple_syrup import floating_checkpoint_cuda as fcc
    from maple_syrup import floating_soil_layers as fsl
    from maple_syrup import floating_soil_layers_cuda as fcu

    old = _old()
    L = Lin(cp, tmp_path, False)
    calls = old._record_admissions(monkeypatch)
    ck = save(L, tmp_path / "ck", available_bytes_fn=lambda: BIG)
    d_save, h_save, _w = calls[-1]
    man = json.loads((ck / "floating_pond_checkpoint.json").read_text())
    table_bytes = sum(int(np.prod(r["shape"], dtype=np.int64)) * np.dtype(r["dtype"]).itemsize for r in man["payload"]["arrays"].values())  # independent: from the FILE's header table (includes the origin array)
    assert h_save >= table_bytes and d_save > 0
    for kw, label in (({"available_bytes_fn": lambda: 0}, "device zero"), ({"host_available_bytes_fn": lambda: 0}, "host zero"), ({"available_bytes_fn": lambda: d_save - 1}, "device need-1"),
                      ({"host_available_bytes_fn": lambda: h_save + fcu.PY_ALLOWANCE_BYTES - 1}, "host need-1"), ({"available_bytes_fn": lambda: d_save, "reserve_bytes": 1}, "device reserve once"),
                      ({"memory_budget_bytes": 0}, "zero device budget"), ({"host_memory_budget_bytes": 0}, "zero host budget")):
        t = tmp_path / f"r_{label.replace(' ', '_')}"
        with pytest.raises(fsl.FloatingMemoryError):
            save(L, t, **kw)
        assert not t.exists() and not list(tmp_path.glob(f".{t.name}*")), label
    save(L, tmp_path / "exact", available_bytes_fn=lambda: d_save, host_available_bytes_fn=lambda: h_save + fcu.PY_ALLOWANCE_BYTES)
    ev2, v2 = target(L, ck, __import__("maple_syrup.floating_continued_checkpoint_cuda", fromlist=["x"]))
    calls.clear()
    load(L, ck, ev2)
    stage0, stage1 = calls[0], calls[1]
    assert stage0[0] == 0 and "manifest" in stage0[2] and "load" in stage1[2] and stage1[1] >= table_bytes
    for kw, label in (({"available_bytes_fn": lambda: stage1[0] - 1}, "device need-1"), ({"host_available_bytes_fn": lambda: stage1[1] + fcu.PY_ALLOWANCE_BYTES - 1}, "host need-1"),
                      ({"available_bytes_fn": lambda: 0}, "device zero"), ({"host_available_bytes_fn": lambda: 0}, "host zero")):
        e3, v3 = target(L, ck, __import__("maple_syrup.floating_continued_checkpoint_cuda", fromlist=["x"]))
        with pytest.raises(fsl.FloatingMemoryError):
            load(L, ck, e3, **kw)
        assert old.untouched(v3), label
    assert fcc is not None and v2 is not None


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_resealed_origin_datum_counter_record_source_kernel_schema_tampers_are_refused_before_adoption_and_the_target_is_retryable(cp, tmp_path, windowed):
    from maple_syrup import floating_checkpoint as fc
    from maple_syrup import floating_checkpoint_cuda as fcc
    from maple_syrup import floating_continued_checkpoint_cuda as ccc
    from maple_syrup import floating_pond as fp
    from maple_syrup import floating_resume_cuda as frc
    from maple_syrup import floating_soil_layers as fsl
    from maple_syrup import pond_checkpoint as pc

    old = _old()
    L = Lin(cp, tmp_path, windowed)
    good = save(L, tmp_path / "good")
    ev2, v2 = target(L, good, ccc)
    g = L.state.topology.generation
    cases = {
        "datum": {"arrays": {"datum_full": lambda a: np.nextafter(a, np.inf)}},
        "routing_source": {"arrays": {"co_routing_source": lambda a: np.nextafter(a, np.inf)}},
        "depth": {"arrays": {"fl_depth": lambda a: np.nextafter(a, np.inf)}},
        "held_fc": {"arrays": {"fl_held_fc": lambda a: np.nextafter(a, np.inf)}},
        "origin_generation": {"manifest": lambda m: m["continued_origin"].update(routing_generation=g + 1)},
        "binding_generation": {"manifest": lambda m: m["binding"]["initial_topology"].update(generation=g + 1)},
        "binding_original": {"manifest": lambda m: m["binding"]["continued_origin"].update(original_source_sha256="0" * 64)},
        "counter_n_refreshes": {"manifest": lambda m: m["payload"]["scalars"]["counters"].update(n_refreshes=g)},
        "topology_generation": {"manifest": lambda m: m["payload"]["scalars"]["topology"].update(generation=g + 1)},
        "kernel": {"manifest": lambda m: m["kernel"].update(fmc_source_sha256="0" * 64)},
        "source": {"manifest": lambda m: m["source"].update(maple_package_sha256="0" * 64)},
        "schema_old_cuda": {"manifest": lambda m: m.update(schema=fcc.GPU_WINDOWED_SCHEMA if windowed else fcc.GPU_SCHEMA, profile=fcc.GPU_PROFILE)},
        "schema_cpu_continued": {"manifest": lambda m: m.update(schema=fc.CONTINUED_SCHEMA, profile=fc.CONTINUED_PROFILE)},
        "profile": {"manifest": lambda m: m.update(profile=fcc.GPU_PROFILE)},
        "backend": {"manifest": lambda m: m.update(array_backend="numpy")},
        "unknown_key": {"manifest": lambda m: m.update(extra="x")},
        "drop_origin_record": {"manifest": lambda m: m.pop("continued_origin")},
    }
    typed = (fc.PondCheckpointError, fcc.FloatingCheckpointCudaError, frc.FloatingResumeCudaError, fp.FloatingPondError, fsl.FloatingSoilError)
    for name, kw in cases.items():
        bad = tmp_path / f"bad_{name}"
        shutil.copytree(good, bad)
        old.retamper(bad, **kw)
        with pytest.raises(typed):
            load(L, bad, ev2)
        assert old.untouched(v2), name
    if windowed:  # the native prefix and the EXTERNAL deep store (genuine native Windowed protocol)
        bad = tmp_path / "bad_prefix"
        shutil.copytree(good, bad)
        old.retamper_native(bad, "prefix_packet", lambda a: a * 1.5 + 1.0)
        with pytest.raises(fcc.FloatingCheckpointCudaError, match="windowed snapshot cannot be loaded"):
            load(L, bad, ev2)
        assert old.untouched(v2)
        store = old.deep_dir(good)
        moved = store.with_name(store.name + "_moved")
        store.rename(moved)
        try:
            with pytest.raises(fcc.FloatingCheckpointCudaError, match="windowed snapshot cannot be loaded"):
                load(L, good, ev2)
            assert old.untouched(v2)
        finally:
            moved.rename(store)
        for name, fn in {"generation": lambda m: m["deep_storage"].update(generation=m["deep_storage"]["generation"] + 1), "digest": lambda m: m["deep_storage"].update(physical_manifest_digest="0" * 64)}.items():
            bad = tmp_path / f"bad_deep_{name}"
            shutil.copytree(good, bad)
            old.retamper(bad, manifest=fn)
            with pytest.raises(typed):
                load(L, bad, ev2)
            assert old.untouched(v2), name
    flip = tmp_path / "flip"
    shutil.copytree(good, flip)
    with (flip / pc.PAYLOADS[1]).open("r+b") as h:  # a plain bit flip WITHOUT recomputing the hash
        h.seek(200)
        b = h.read(1)
        h.seek(200)
        h.write(bytes([b[0] ^ 1]))
    with pytest.raises(fc.PondCheckpointError, match="hash mismatch"):
        load(L, flip, ev2)
    assert old.untouched(v2)
    got = load(L, good, ev2)  # the SAME target restores after every refusal
    assert got.counters.n_refreshes == 0


def test_old_cuda_and_cpu_profiles_and_the_continued_family_refuse_each_other_on_device(cp, tmp_path):
    from maple_syrup import floating_checkpoint as fc
    from maple_syrup import floating_checkpoint_cuda as fcc
    from maple_syrup import floating_continued_checkpoint_cuda as ccc

    L = Lin(cp, tmp_path, False)
    ck = save(L, tmp_path / "ck")
    ev2, _v2 = target(L, ck, ccc)
    with pytest.raises(fc.FloatingCheckpointError, match="not a CUDA floating schema"):
        fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT, available_bytes_fn=L.reader, host_available_bytes_fn=lambda: BIG)  # the historical /7 /8 loader refuses the continued family
    from maple_syrup import pond_checkpoint as pc

    head, state, t0 = L.lin.head, L.state, float(L.state.hydraulic.t_s)
    # the ACTUAL typed refusal of the unchanged historical validator is the NATIVE `pc.PondCheckpointError` (not a `fc.FloatingCheckpointError`: `FloatingCheckpointError` is a SUBCLASS of it, not the other way round)
    with pytest.raises(pc.PondCheckpointError, match="ORIGINAL initial topology / source datum"):
        fcc.save_floating_pond_checkpoint_cuda(tmp_path / "old", L.state, L.event, IDENT, available_bytes_fn=L.reader, host_available_bytes_fn=lambda: BIG)  # the historical writer refuses the continued state
    assert not (tmp_path / "old").exists() and not list(tmp_path.glob(".old.partial-*")) and not list(tmp_path.glob(".old*"))  # nothing published, no partial staging
    assert L.lin.head is head and L.state is state and float(L.state.hydraulic.t_s) == t0  # same accepted head, same current state


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_same_gpu_fresh_process_restores_from_the_files_alone(cp, tmp_path, windowed):
    """Process A saves (continued file + recipe.json + the refreshed-root H0 record parts; the recipe's event is verified against the ORIGINAL event's bindings first) and dumps; process B (NEW interpreter) disables every original
    bed/owner/probe/provider/pair constructor BEFORE reading the recipe, then restores from recipe + H0 record + checkpoint alone and continues. CLAIM (no original bed allocation) is limited to what the worker enforces: the tripwires
    listed in `tripwires.json` were installed before any recipe preparation and the load never imports the original-case fixtures; the SAVE process of course builds the original case."""
    import subprocess
    import sys

    worker = Path(__file__).resolve().parent / "gpu_continued_worker.py"
    env = dict(os.environ)
    for mode in ("save", "load"):
        r = subprocess.run([sys.executable, "-B", str(worker), mode, str(tmp_path), "1" if windowed else "0", "4"], capture_output=True, text=True, env=env, timeout=1800, check=False)
        assert r.returncode == 0, r.stderr[-3000:]
    a, b = np.load(tmp_path / "save_final.npz"), np.load(tmp_path / "load_final.npz")
    assert set(a.files) == set(b.files) and all(a[k].tobytes() == b[k].tobytes() for k in a.files if "window_cache" not in k)
    assert "make_pair" in json.loads((tmp_path / "tripwires.json").read_text())["disabled_before_recipe"]


def test_generation_2_local_advance_needs_a_genuine_second_reroute(cp):
    pytest.skip("no bounded genuine generation-2 fixture exists (a second real net-elevation reroute inside a continued event); nothing is faked")
