"""G-F4 HOST tests of the CUDA general held-context container (CPU/host: NO GPU, NO CuPy). The physical device behaviour is defined in `test_gpu_definitions.py` for Root.

CONTROLS ARE EXPLICIT: (a) the refusals that need no device are exercised with the real functions; (b) to reach the strict manifest/cross-binding/model code BEYOND the device gate, `no_device` replaces ONLY the gate and the
admission recorder (ordering/arithmetic control, never a numerical body); (c) the FORGED GPU containers are built from a genuine CPU `/1` container (real H0 record, real child manifest, real files) by editing metadata and
RESEALING every hash, so they exercise exactly the cross-binding refusals; they are never loaded as physical state (no array is read in any forged-container test: `np.load`, `record_from_parts` and the factory are made to
fail if reached)."""
# ruff: noqa: RUF059
from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from case import AVAIL, IDENT, build

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_checkpoint_cuda as fcc
from maple_syrup import floating_context_checkpoint as cc
from maple_syrup import floating_context_checkpoint_cuda as ccc
from maple_syrup import floating_held_context as fh
from maple_syrup import floating_held_context_cuda as fhc
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import pond_checkpoint as pc

HERE = Path(__file__).resolve()
TASK = HERE.parents[3]
SRC = HERE.parents[2] / "src" / "maple_syrup"
MANIFEST = TASK / "completion_program_20261113" / "gpu_context_outer_pre_source_manifest.json"
LIMIT = 1 << 40


def cpu_container(tmp, windowed=False):
    tmp.mkdir(parents=True, exist_ok=True)
    ev, st, vert = build(tmp / "o", backend="numpy", windowed=windowed)
    spec = fh.PreferentialSpec(np.full(vert.n_cells, 2.0e-3), 0.5, 1e-5, np.array([0.0, 1.0, 1.0]), (), None)
    ck = tmp / "cpu_ck"
    cc.save_floating_context_checkpoint(ck, st, ev, IDENT, pref_spec=spec, available_bytes_fn=AVAIL)
    return ck


def reseal(ck: Path, *, outer=None, child=None, meta=None):
    """Edit the outer manifest / child manifest / H0 meta and RECOMPUTE every hash, size and seal."""
    data = json.loads((ck / cc.MANIFEST_NAME).read_text())
    cdir = ck / cc.CHILD_DIR
    cman = json.loads((cdir / fc.MANIFEST_NAME).read_text())
    if meta:
        meta(data["context_meta"])
    if child:
        child(cman)
    (cdir / fc.MANIFEST_NAME).write_text(pc._json_dumps(fc._seal(cman)))
    cfiles = cc._file_table(cdir, tuple(sorted(cc.CHILD_NAMES)), LIMIT)
    data["child"] = {"schema": cman["schema"], "manifest_sha256": cfiles[fc.MANIFEST_NAME]["sha256"], "files": cfiles}
    data["files"] = cc._file_table(ck, (cc.ARRAYS_NAME,), LIMIT)
    if outer:
        outer(data)
    (ck / cc.MANIFEST_NAME).write_text(pc._json_dumps(cc._seal(data)))


def forge(tmp, windowed=False) -> Path:
    """A GPU-profile container forged from a genuine CPU one (module docstring): GPU profile/backends, H0 `/2` meta with the live kernel descriptor, child manifest relabelled `/7`/`/8` with kernel and layouts."""
    ck = cpu_container(tmp, windowed)
    kern = fhc.kernel_descriptor()
    schema = fcc.GPU_WINDOWED_SCHEMA if windowed else fcc.GPU_SCHEMA

    def meta(m):
        m["schema"], m["backend"], m["kernel"] = fh.SCHEMA2, "cuda", dict(kern)

    def child(c):
        lay = json.loads((ck / cc.MANIFEST_NAME).read_text())["context_meta"]["layouts"]
        c.update(schema=schema, profile=fcc.GPU_PROFILE, array_backend="cupy", hydraulic_backend="cuda", kernel=dict(kern))
        c["floating"]["held_layouts"] = {"theta_field_capacity": lay["theta_field_capacity"], "theta_saturation": lay["theta_saturation"]}

    def outer(d):
        d.update(profile=ccc.PROFILE, array_backend="cupy", hydraulic_backend="cuda", context_schema=fh.SCHEMA2)

    reseal(ck, outer=outer, child=child, meta=meta)
    return ck


@pytest.fixture
def no_device(monkeypatch):
    """ORDERING/ARITHMETIC control: replaces ONLY the device gate and the admission (recorded); everything else is the real code."""
    calls = []
    monkeypatch.setattr(ccc, "gate_expected", lambda device_id: (object(), int(device_id)))
    monkeypatch.setattr(ccc.fcu, "admit", lambda cp, dev, d, h, q, what: calls.append((int(d), int(h), what)) or dev)
    return calls


@pytest.fixture
def tripwires(monkeypatch):
    """Nothing beyond the preflight may run FROM THE CONTAINER MODULE: no array load, no H0 record parse, no reconstruct, no factory (the tripwires fire only when the caller is the container, so the test's own fixture code is unaffected)."""
    import sys

    def make(real):
        def guarded(*a, **k):
            if sys._getframe(1).f_globals.get("__name__") == ccc.__name__:
                raise AssertionError("an array/record/factory operation ran before the cross-binding checks finished")
            return real(*a, **k)

        return guarded

    monkeypatch.setattr(np, "load", make(np.load))
    monkeypatch.setattr(ccc.fh, "record_from_parts", make(fh.record_from_parts))
    monkeypatch.setattr(ccc.fh, "reconstruct_held_context", make(fh.reconstruct_held_context))


def need(ck, **kw):
    return ccc.context_load_need_bytes_cuda(ck, IDENT, expected_device_id=0, **kw)


# ---- closed profile, mutual refusal ----------------------------------------------------------------------------------------------------------------------------------------------
def test_the_module_imports_without_cupy_and_the_profile_is_a_separate_closed_version():
    import sys

    assert "cupy" not in sys.modules
    assert ccc.PROFILE == "maple-syrup-floating-context-checkpoint/2-cuda" and ccc.PROFILE != cc.PROFILE and cc.PROFILE.endswith("/1")
    assert ccc.MANIFEST_KEYS == cc.MANIFEST_KEYS and ccc.CHILD_SCHEMAS == {"dense": fcc.GPU_SCHEMA, "windowed": fcc.GPU_WINDOWED_SCHEMA}
    src = inspect.getsource(ccc)
    assert "load_floating_pond_checkpoint(" not in src.replace("load_floating_pond_checkpoint_cuda(", "") and "save_floating_pond_checkpoint(" not in src.replace("save_floating_pond_checkpoint_cuda(", "")  # no CPU child fallback


def test_the_cpu_container_refuses_the_gpu_profile_and_the_gpu_loader_refuses_the_cpu_container(tmp_path, no_device, tripwires):
    cpu = cpu_container(tmp_path)
    with pytest.raises(ccc.ContextCheckpointCudaError, match="unknown context profile"):
        need(cpu)
    gpu = forge(tmp_path / "g")
    with pytest.raises(cc.ContextCheckpointError, match="unknown context profile"):
        cc.context_load_need_bytes(gpu, IDENT, available_bytes_fn=AVAIL)  # the unchanged CPU `/1` loader refuses the new profile (closed profile, no migration)


# ---- arguments and gate order ----------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [True, False, -1, 1.0, "0", None, np.bool_(True)])
def test_expected_device_id_is_a_strict_integer_refused_before_the_gate(tmp_path, monkeypatch, bad):
    gate = []
    monkeypatch.setattr(ccc, "gate_expected", lambda d: gate.append(d))
    with pytest.raises(ccc.ContextCheckpointCudaError, match="strict integer"):
        ccc.load_floating_context_checkpoint_cuda(tmp_path / "x", lambda c: None, IDENT, expected_device_id=bad)
    with pytest.raises(ccc.ContextCheckpointCudaError, match="strict integer"):
        ccc.context_load_need_bytes_cuda(tmp_path / "x", IDENT, expected_device_id=bad)
    assert gate == []


@pytest.mark.parametrize("which", ["load", "need"])
def test_the_current_device_gate_precedes_every_reader_path_check_hash_and_callback(tmp_path, which):
    reads, calls = [], []

    def reader():
        reads.append(1)
        return 1 << 40

    kw = {"available_bytes_fn": reader, "host_available_bytes_fn": reader}
    with pytest.raises(ccc.ContextCheckpointCudaError, match="CUDA device gate refused"):  # no CUDA here: the REAL gate is the first operation after pure metadata validation
        if which == "load":
            ccc.load_floating_context_checkpoint_cuda(tmp_path / "missing", lambda c: calls.append(1), IDENT, expected_device_id=0, **kw)
        else:
            ccc.context_load_need_bytes_cuda(tmp_path / "missing", IDENT, expected_device_id=0, **kw)
    assert reads == [] and calls == [] and not (tmp_path / "missing").exists()


@pytest.mark.parametrize("bad", [-1, 1.5, True, "1", None])
def test_reserves_budgets_and_readers_are_validated_before_the_gate(tmp_path, monkeypatch, bad):
    gate = []
    monkeypatch.setattr(ccc, "gate_expected", lambda d: gate.append(d))
    with pytest.raises(ccc.ContextCheckpointCudaError, match="reserve_bytes must be an integer >= 0"):
        need(tmp_path / "x", reserve_bytes=bad)
    with pytest.raises(ccc.ContextCheckpointCudaError, match="host_reserve_bytes must be an integer >= 0"):
        ccc.load_floating_context_checkpoint_cuda(tmp_path / "x", lambda c: None, IDENT, expected_device_id=0, host_reserve_bytes=bad)
    with pytest.raises(ccc.ContextCheckpointCudaError, match="must be callable or None"):
        need(tmp_path / "x", host_available_bytes_fn=3)
    with pytest.raises(ccc.ContextCheckpointCudaError, match="factory must be callable"):
        ccc.load_floating_context_checkpoint_cuda(tmp_path / "x", "not callable", IDENT, expected_device_id=0)
    assert gate == []


def test_save_refuses_a_cpu_event_typed_before_any_reader_gate_or_io(tmp_path):
    ev, st, vert = build(tmp_path / "o", backend="numpy")
    reads = []
    with pytest.raises(ccc.ContextCheckpointCudaError, match="needs a CUDA vertical on a CuPy event"):
        ccc.save_floating_context_checkpoint_cuda(tmp_path / "ck", st, ev, IDENT, available_bytes_fn=lambda: reads.append(1) or 1 << 40)
    assert reads == [] and not (tmp_path / "ck").exists() and list(tmp_path.glob(".ck*")) == []


def test_save_gates_before_the_path_check_and_refuses_an_existing_target_after_it(tmp_path, monkeypatch):
    ev, st, vert = build(tmp_path / "o", backend="numpy")
    monkeypatch.setattr(ccc.fcc, "check_event", lambda e: vert)  # ordering control only: a CPU vertical stands in for the CUDA one
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(ccc.ContextCheckpointCudaError, match="CUDA device gate refused"):
        ccc.save_floating_context_checkpoint_cuda(existing, st, ev, IDENT)  # the gate wins over the "already exists" refusal
    monkeypatch.setattr(ccc.fcc, "gate", lambda column: (None, 0))
    with pytest.raises(ccc.ContextCheckpointCudaError, match="already exists"):
        ccc.save_floating_context_checkpoint_cuda(existing, st, ev, IDENT)  # nothing was exported or written
    assert list(tmp_path.glob(".exists*")) == []


# ---- preflight arithmetic: composed two-pool models from header dimensions ------------------------------------------------------------------------------------------------
def test_the_dense_need_is_the_documented_composition_and_includes_allowance_and_reserves_once(tmp_path, no_device, tripwires):
    ck = forge(tmp_path)
    n = need(ck, reserve_bytes=1000, host_reserve_bytes=2000)
    data = json.loads((ck / cc.MANIFEST_NAME).read_text())
    cman = json.loads((ck / cc.CHILD_DIR / fc.MANIFEST_NAME).read_text())
    meta = data["context_meta"]
    msize = (ck / cc.MANIFEST_NAME).stat().st_size
    table = {k: v for k, v in data["context_arrays"].items()}
    h0 = sum(int(np.prod(r["shape"], dtype=np.int64)) * np.dtype(r["dtype"]).itemsize for r in table.values())
    tb = sum(int(np.prod(r["shape"], dtype=np.int64)) * np.dtype(r["dtype"]).itemsize for r in cman["payload"]["arrays"].values())
    nat = cman["native_dense"]
    L, N = meta["n_layers"], meta["n_cells"]
    cd, ch = fcc.load_models(L, N, cman["binding"]["n_classes"], cman["binding"]["phase_bins"], table_bytes=tb, dev_table_bytes=tb, native_bytes=nat["native_member_bytes"], V=nat["voxel_shape"][2],
                             K=nat["voxel_shape"][3], n_active=N, batch_cells=cc.nfs_default_batch())
    plan_dev, plan_host, _e = fhc.plan(meta)
    from maple_syrup import floating_resume as fr

    cold = max(int(fr._cold_source_bytes()), cc.COLD_ALLOWANCE_BYTES)
    ks = meta["mode"] == "texture_ks"
    ret_dev = fcu.retained_terms(L, N, dict(meta["layouts"]["all"]))["total"] + N * 8 + N + (N * 8 if ks else 0) + (L * N + N) * 8
    ret_host = N * 8 + N + (N * 8 if ks else 0) + 2 * (L + 1) * 8
    host = 3 * msize + cc.BUFFER_BYTES + cold + max(2 * h0 + fh.ALLOWANCE_BYTES + plan_host, ret_host + ch)
    assert (n.device_bytes, n.host_bytes) == (max(plan_dev, ret_dev + cd), host)
    assert n.device_total_bytes == n.device_bytes + 1000 and n.host_total_bytes == n.host_bytes + fcu.PY_ALLOWANCE_BYTES + 2000  # reserves ONCE, the 64 KiB allowance explicit
    assert no_device[0][:2] == (0, 3 * msize + cc.BUFFER_BYTES + cold) and "manifest" in no_device[0][2]  # stage 0 is admitted from the manifest size alone
    assert n.stages["child"]["windowed"] is False and n.stages["child"]["V"] == nat["voxel_shape"][2] and n.stages["child"]["cache_bytes"] == 0


def test_the_windowed_need_carries_the_deep_store_mapping_cache_bound_not_the_resident_archive(tmp_path, no_device, tripwires):
    ck = forge(tmp_path, windowed=True)
    n = need(ck)
    cman = json.loads((ck / cc.CHILD_DIR / fc.MANIFEST_NAME).read_text())
    meta = json.loads((ck / cc.MANIFEST_NAME).read_text())["context_meta"]
    cache = fc.deep_cache_bound_bytes(cman["deep_storage"], meta["n_classes"])
    archive = (ck / cc.CHILD_DIR / pc.PAYLOADS[0]).stat().st_size
    assert n.stages["child"]["windowed"] and n.stages["child"]["cache_bytes"] == cache and n.stages["child"]["V"] == 0 and n.stages["child"]["native_bytes"] == archive
    assert cache > archive and n.stages["child_host"] >= cache  # the real MiB-scale mapping bound, never the KiB-scale resident archive


def test_the_need_is_monotone_in_the_declared_dimensions(tmp_path, no_device, tripwires):
    ck = forge(tmp_path)
    base = need(ck)
    bigger = need(ck, batch_cells=cc.nfs_default_batch() * 2)
    assert bigger.device_bytes >= base.device_bytes and bigger.host_bytes >= base.host_bytes
    meta = json.loads((ck / cc.MANIFEST_NAME).read_text())["context_meta"]
    a = ccc.compose_need(meta, 1000, 5000, 10, 20, 0)
    for args in ((2000, 5000, 10, 20, 0), (1000, 9000, 10, 20, 0), (1000, 5000, 10**9, 20, 0), (1000, 5000, 10, 10**9, 0), (1000, 5000, 10, 20, 10**6)):
        b = ccc.compose_need(meta, *args)
        assert b[0] >= a[0] and b[1] >= a[1] and (b[0], b[1]) != (a[0], a[1])


def test_stage_one_admits_exactly_the_published_need_before_any_array_and_a_refusal_stops_everything(tmp_path, no_device, tripwires, monkeypatch):
    ck = forge(tmp_path)
    n = need(ck)
    no_device.clear()
    factory_calls = []
    with pytest.raises(AssertionError, match="before the cross-binding checks finished|an array/record"):  # the tripwires fire AFTER stage-1 admission (the first operation past the preflight/hash/header audits)
        ccc.load_floating_context_checkpoint_cuda(ck, lambda c: factory_calls.append(1), IDENT, expected_device_id=0)
    assert [c[:2] for c in no_device][1] == (n.device_bytes, n.host_bytes) and "composed" in no_device[1][2] and factory_calls == []

    def refuse(cp, dev, d, h, q, what):
        if "composed" in what:
            raise fsl.FloatingMemoryError("refused (control)")
        return dev

    monkeypatch.setattr(ccc.fcu, "admit", refuse)
    with pytest.raises(fsl.FloatingMemoryError):
        ccc.load_floating_context_checkpoint_cuda(ck, lambda c: factory_calls.append(1), IDENT, expected_device_id=0)  # typed memory refusal BEFORE any hash, array or callback
    assert factory_calls == []


# ---- cross-binding / strict-manifest refusals (recomputed outer hashes; nothing past the preflight may run) ---------------------------------------------------------------
def mut_child(**kv):
    return lambda c: c.update(kv)


CASES = {
    "source": {"outer": lambda d: d["source"].update(maple_package_sha256="0" * 64)},
    "outer_kernel_in_meta": {"meta": lambda m: m["kernel"].update(fmc_source_sha256="0" * 64)},
    "child_kernel": {"child": lambda c: c["kernel"].update(fpc_source_sha256="0" * 64)},
    "root_column": {"meta": lambda m: m["identities"].update(root_column_sha256="1" * 64)},
    "config": {"meta": lambda m: m["identities"].update(config_sha256="2" * 64)},
    "matrix_op": {"meta": lambda m: m["identities"].update(matrix_op_sha256="3" * 64)},
    "pref": {"meta": lambda m: m["identities"].update(pref_sha256="4" * 64)},
    "generation": {"meta": lambda m: m["facts"].update(generation=m["facts"]["generation"] + 1)},
    "witness": {"meta": lambda m: m["facts"].update(source_witness="5" * 64)},
    "layouts": {"child": lambda c: c["floating"]["held_layouts"].update(theta_saturation="scalar" if c["floating"]["held_layouts"]["theta_saturation"] != "scalar" else "cell")},
    "ks": {"child": lambda c: c["floating"].update(has_ks=not c["floating"]["has_ks"])},
    "shape": {"child": lambda c: c["floating"].update(shape=[c["floating"]["shape"][0] + 1, c["floating"]["shape"][1]])},
    "child_schema_cpu": {"child": mut_child(schema=fc.SCHEMA_V3)},
    "child_schema_windowed": {"child": mut_child(schema=fcc.GPU_WINDOWED_SCHEMA)},
    "child_backend": {"child": mut_child(array_backend="numpy")},
    "child_identity": {"child": mut_child(identity={"case": "other"})},
    "child_source": {"child": lambda c: c["source"].update(maple_syrup_tree_sha256="0" * 64)},
    "storage": {"outer": lambda d: d.update(bed_storage="windowed")},
    "context_schema_v1": {"outer": lambda d: d.update(context_schema=fh.SCHEMA)},
    "backend": {"outer": lambda d: d.update(array_backend="numpy")},
    "hydraulic_backend": {"outer": lambda d: d.update(hydraulic_backend="numpy")},
    "identity": {"outer": lambda d: d.update(identity={"case": "other", "version": 1})},
    "unknown_key": {"outer": lambda d: d.update(extra="x")},
    "missing_key": {"outer": lambda d: d.pop("child")},
    "h0_meta_v1": {"meta": lambda m: m.update(schema=fh.SCHEMA)},
    "h0_meta_unknown_key": {"meta": lambda m: m.update(extra="x")},
    "h0_array_table": {"outer": lambda d: d["context_arrays"].pop(next(iter(d["context_arrays"])))},
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_cross_binding_and_strict_manifest_refusals_with_recomputed_hashes_happen_before_any_array_or_factory(tmp_path, no_device, tripwires, name):
    ck = forge(tmp_path)
    reseal(ck, **CASES[name])
    with pytest.raises(ccc.ContextCheckpointCudaError):
        need(ck)
    calls = []
    with pytest.raises(ccc.ContextCheckpointCudaError):
        ccc.load_floating_context_checkpoint_cuda(ck, lambda c: calls.append(1), IDENT, expected_device_id=0)
    assert calls == [] and all("composed" not in c[2] for c in no_device)  # not even the stage-1 admission was reached


def test_a_modified_manifest_without_a_recomputed_seal_and_file_level_damage_are_refused(tmp_path, no_device, tripwires):
    ck = forge(tmp_path)
    data = json.loads((ck / cc.MANIFEST_NAME).read_text())
    data["identity"] = {"case": "x"}
    (ck / cc.MANIFEST_NAME).write_text(json.dumps(data))
    with pytest.raises(ccc.ContextCheckpointCudaError, match="manifest hash mismatch"):
        need(ck)
    ck2 = forge(tmp_path / "b")
    (ck2 / cc.ARRAYS_NAME).write_bytes((ck2 / cc.ARRAYS_NAME).read_bytes() + b"x")
    with pytest.raises(ccc.ContextCheckpointCudaError, match="file size mismatch"):
        need(ck2)
    ck3 = forge(tmp_path / "c")
    (ck3 / "extra").write_text("x")
    with pytest.raises(ccc.ContextCheckpointCudaError, match="must contain exactly"):
        need(ck3)
    ck4 = forge(tmp_path / "d")
    victim = ck4 / cc.CHILD_DIR / pc.PAYLOADS[1]
    real = tmp_path / "real.npz"
    shutil.move(victim, real)
    os.symlink(real, victim)
    with pytest.raises(ccc.ContextCheckpointCudaError, match="not a regular file"):
        need(ck4)
    ck5 = forge(tmp_path / "e")
    shutil.rmtree(ck5 / cc.CHILD_DIR)
    with pytest.raises(ccc.ContextCheckpointCudaError, match="must contain exactly"):
        need(ck5)
    link = tmp_path / "link"
    os.symlink(forge(tmp_path / "f"), link)
    with pytest.raises(ccc.ContextCheckpointCudaError, match="plain directory"):
        need(link)


def test_raw_h0_property_arrays_changed_with_recomputed_outer_hashes_are_refused_at_hash_or_identity_never_adopted(tmp_path, no_device):
    """The preflight cannot see raw array VALUES (no array is read there); the values are bound by the file hash and, after `np.load`, by the H0 re-preparation identity. Here: a tampered array WITHOUT recomputed file hashes is
    refused at the hash stage; the recomputed-hash case is covered on the device by the GPU definitions (re-preparation must reproduce the recorded root column/matrix/preferential sha before the factory)."""
    ck = forge(tmp_path)
    with np.load(ck / cc.ARRAYS_NAME, allow_pickle=False) as z:
        data = {k: np.array(z[k]) for k in z.files}
    name = "theta_field_capacity"
    data[name] = data[name] + 1e-3
    os.unlink(ck / cc.ARRAYS_NAME)
    np.savez(ck / cc.ARRAYS_NAME, **data)
    calls = []
    with pytest.raises(ccc.ContextCheckpointCudaError, match="file size mismatch|file hash/size mismatch"):
        ccc.load_floating_context_checkpoint_cuda(ck, lambda c: calls.append(1), IDENT, expected_device_id=0)
    assert calls == []
    reseal(ck)  # the attacker recomputes every hash: the container preflight (sizes, bindings) now PASSES by design; physical identity is checked by the H0 re-preparation
    assert need(ck).device_bytes > 0


# ---- unchanged qualified components -------------------------------------------------------------------------------------------------------------------------------------------------
# CC2: the stale G-F4 pre-edit manifest audit is preserved verbatim, non-default, in `historical_source_audits.py` (with its qualification record). The CURRENT per-interface guards of what this outer container relies on:
BASELINE_PINS = {'floating_context_checkpoint_cuda.py': '0ce855b56f93c7c3e5373263e251976939a186e2d8062e3fd4978e52cdd8bbdd', 'floating_context_checkpoint.py': '6df07c9aa04a156ffaa47d6bd7c11ba7a3c44a34f210c47cba335cb932185916', 'floating_held_context_cuda.py': 'e233dee0243908769980875f909a67088ad8693adfdcdefc93fcd54c41f01c5e', 'floating_held_context.py': '721090647b1cd09c0df211642686e4da0f34bc242a06fb1a42d67f8cc203ea4f', 'floating_resume_cuda.py': 'a4674a1dbe744693782b042415b7c4c4ad716c99f621cbadc99f62ef2b454523', 'pond_checkpoint.py': '94cd2339566646ba44f2015dad13a67194c26a8dd3cc85b56916f62d2b4a704f'}


def test_the_outer_container_and_the_modules_it_relies_on_are_byte_identical_to_the_cc2_baseline():
    for name, sha in BASELINE_PINS.items():
        assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == sha, name  # the context outer, the CPU outer, the H0 pair, the resume module and the historical codec are untouched by CC2


def test_the_cuda_child_interface_the_outer_uses_keeps_the_historical_profiles_and_defaults():
    import inspect

    assert fcc.READ_SCHEMAS == (fcc.GPU_SCHEMA, fcc.GPU_WINDOWED_SCHEMA) and ccc.CHILD_SCHEMAS == {"dense": fcc.GPU_SCHEMA, "windowed": fcc.GPU_WINDOWED_SCHEMA}
    for fn in (fcc.save_floating_pond_checkpoint_cuda, fcc.load_floating_pond_checkpoint_cuda):
        assert inspect.signature(fn).parameters["original_topology"].default is None  # the default route is the historical /7 /8
    assert "original_topology" not in inspect.getsource(ccc)  # the outer never selects the continued profile
    assert fcc._kernel() == fhc.kernel_descriptor()
    assert not set(fcc.CONT_READ_SCHEMAS) & set(fcc.READ_SCHEMAS)


def test_the_cpu_container_and_childs_do_not_know_the_cuda_container():
    for mod in (cc, fc):
        assert "floating_context_checkpoint_cuda" not in inspect.getsource(mod)
