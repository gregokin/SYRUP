"""CC2 HOST tests of the CUDA continued-origin child profile (no GPU/CuPy). They prove: the closed schema/profile/key family and its disjointness from every old profile, that no old definition of the CUDA child changed (AST hashes of the
accepted baseline; the four edited definitions are covered by behaviour tests), that the CUDA/CPU/historical loaders refuse each other, the host metadata reader (header/table/size/kernel/SourcePair preflight BEFORE any decode, with an
`np.load` tripwire) on a HOST-FORGED CONTROL container (a genuine CPU continued file re-labelled as the CUDA profile with the live kernel descriptor and resealed -- a control of the reader only, NEVER a CUDA file), the early typed
refusals of save/load (CPU event; missing original) before any IO, and the explicit modelling of the O(N padded) origin bytes. Real device behaviour is in `test_gpu_definitions.py` (not run here)."""
from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import shutil
import zipfile
from pathlib import Path as Path_

import numpy as np
import pytest
from cont_builders import AVAIL, IDENT, cc, fc, lineage, save

from maple_syrup import floating_checkpoint_cuda as fcc
from maple_syrup import floating_continued_checkpoint_cuda as ccc
from maple_syrup import pond_checkpoint as pc

REFUSE = (fc.FloatingCheckpointError, pc.PondCheckpointError)
BASELINE_GFO = {'FloatingCheckpointCudaError': 'd452c72993ceaf7420f6', '_minting': 'bddd922db56944a9f872', '_fail': '81240ffe086cfb70147d', 'VerifiedFloatingInputsCuda': 'd9e66bff9a2ef28eb1a2', 'held_shape': '9a267afda876bf2d148d', '_layouts_of': '95d07b972be0499f9bc4', '_check_input_layout': 'b67c0c825650bba8342b', 'input_digest': '87f285ad522fdffde1b9', 'require_inputs': '1a34354ae514ebe6562a', 'verify_intact': 'a7cd10ac78079a8b2f78', 'consume_inputs': '1f62f9ac215bf381216f', 'check_event': '8d54baf86602294b5a34', 'storage_of': 'adf0282f494f49b38de3', 'quota_of': 'ed2e45e45bc452ea5f72', 'component_quota': '8621ac1d857c6af5303d', 'gate': '15b5694d171fbfe69cf5', 'admit': '11536fc7ce74ad25c2f0', '_spec_bytes': '456f289b7c1209c5a96b', 'working_bytes': '2c115473b03867c5ffcb', 'canonical_depth_bytes': '84b42ca9b189356a7444', 'prepare_models': '336b2dafab0f44609a36', 'load_models': '6f44bf77a277e853229b', 'save_models': '7779fd8183f4d73fc22a', '_per': '6d3fc2c825d563965810', 'd2h': '1520649a104ef623c7ce', 'h2d': '766e7c0091dc6d911d76', 'stream_equal': '37af8b79ba1baba81eaa', '_gpu_floating_specs': 'c5dbeb57295e361784e7', '_facts': '4542aeca812ecc8cd408', 'check_held_ids': 'c53e532b90863de8e76d', 'check_held_values': 'c9c5202ef9d3cb9601b9', 'check_canonical_depth': '2e15122eac36267b21c0', 'verify_prefix_untrusted': '95615e7098e0a6462ce7', '_fsync_file': 'c5526a05bdc13d76b316', '_fsync_dir': '818d8cb07ba759ed67b4', '_publish_noreplace': '5c48d5c7cf2a626789f6', '_kernel': '8d9b74116e4406747e78', '_read_manifest_checked': 'bba46e04e223ab060bba'}


def _digest(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()[:20]


def test_schema_profile_and_key_family_is_closed_and_disjoint_from_every_old_profile():
    new = {fcc.CONT_GPU_SCHEMA, fcc.CONT_GPU_WINDOWED_SCHEMA}
    old = set(fc.READ_SCHEMAS) | set(fc.CONTINUED_SCHEMAS) | set(fcc.READ_SCHEMAS)
    assert len(new) == 2 and not new & old and fcc.READ_SCHEMAS == (fcc.GPU_SCHEMA, fcc.GPU_WINDOWED_SCHEMA)
    assert fcc.CONT_GPU_PROFILE not in {fc.PROFILE, fc.CONTINUED_PROFILE, fcc.GPU_PROFILE}
    assert fcc.CONT_MANIFEST_KEYS == fcc.MANIFEST_KEYS | {"continued_origin"} and fcc.CONT_WINDOWED_MANIFEST_KEYS == fcc.WINDOWED_MANIFEST_KEYS | {"continued_origin"}
    assert fcc.schema_of("dense") == fcc.GPU_SCHEMA and fcc.schema_of("windowed") == fcc.GPU_WINDOWED_SCHEMA  # the historical defaults are unchanged
    assert fcc.schema_of("dense", True) == fcc.CONT_GPU_SCHEMA and fcc.schema_of("windowed", True) == fcc.CONT_GPU_WINDOWED_SCHEMA


def test_every_untouched_definition_of_the_cuda_child_is_ast_identical_to_the_accepted_baseline():
    tree = ast.parse(inspect.getsource(fcc))
    got = {n.name: _digest(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in BASELINE_GFO}
    assert got == BASELINE_GFO


def test_the_edited_definitions_keep_the_historical_default_arguments_and_the_gate_first_order():
    for fn in (fcc.save_floating_pond_checkpoint_cuda, fcc.load_floating_pond_checkpoint_cuda):
        p = inspect.signature(fn).parameters["original_topology"]
        assert p.default is None and p.kind is inspect.Parameter.KEYWORD_ONLY
    for fn in (fcc.save_floating_pond_checkpoint_cuda, fcc.load_floating_pond_checkpoint_cuda):
        src = inspect.getsource(fn)
        assert src.index("gate(t.column)") < src.index("_check_original") < src.index("Path(path)")  # the CURRENT device gate FIRST, then the metadata-only attestation, then any IO
    assert inspect.getsource(fcc._binding).count("original_topology is not None") == 1


def test_save_and_load_refuse_a_cpu_event_and_a_missing_original_before_any_io(tmp_path):
    ns = lineage(tmp_path / "a", "numpy", False)
    target = tmp_path / "x"
    with pytest.raises(REFUSE, match="CUDA"):
        ccc.save_floating_continued_checkpoint_cuda(target, ns.state, ns.event, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)
    assert not target.exists() and not list(tmp_path.glob(".x.partial-*"))
    with pytest.raises(REFUSE, match="original_topology is required"):
        ccc.save_floating_continued_checkpoint_cuda(target, ns.state, ns.event, IDENT, original_topology=None)
    with pytest.raises(REFUSE, match="original_topology is required"):
        ccc.load_floating_continued_checkpoint_cuda(target, ns.event, IDENT, original_topology=None)
    ck = save(ns, tmp_path / "ck")  # a CPU continued file
    with pytest.raises(REFUSE, match="CUDA"):
        ccc.load_floating_continued_checkpoint_cuda(ck, ns.event, IDENT, original_topology=ns.original)


def test_origin_bytes_are_modelled_in_the_declared_table_of_both_pools(tmp_path):
    ns = lineage(tmp_path / "a", "numpy", False)
    base = {**pc._specs(ns.event, dry=False)}
    extra = fc._origin_specs(ns.event)
    ny, nx = ns.event.shape
    assert set(extra) == {fc.ORIGIN_ARRAY} and fcc._spec_bytes({**base, **extra}) - fcc._spec_bytes(base) == (ny + 2) * (nx + 2) * 8


# ---- host-forged CONTROL container for the metadata reader -------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cpu_ck(tmp_path_factory):
    d = tmp_path_factory.mktemp("cc2")
    ns = lineage(d / "a", "numpy", False)
    return ns, save(ns, d / "ck"), d


def forge(src, tmp, name, *, member=None, manifest=None, kernel=True):
    """RELABEL a genuine CPU continued file as the CUDA profile (schema/profile/key set/live kernel descriptor) and RESEAL it: a control of the HOST reader only. A recomputed seal is not authenticity."""
    d = tmp / name
    shutil.copytree(src, d)
    z = d / "pond_continuation.npz"
    if member is not None:
        with zipfile.ZipFile(z) as f:
            members = {n: f.read(n) for n in f.namelist()}
        member(members)
        with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_STORED) as f:
            for n, b in members.items():
                f.writestr(n, b)
    m = json.loads((d / fc.MANIFEST_NAME).read_text())
    m["schema"] = fcc.CONT_GPU_SCHEMA
    m["profile"] = fcc.CONT_GPU_PROFILE
    m["array_backend"], m["hydraulic_backend"] = "cupy", "cuda"
    m["kernel"] = fcc._kernel() if kernel else {"common_abi_sha256": "0" * 64}
    m["files"][z.name] = pc._hash_file(z, 1 << 40)[0]
    if manifest:
        manifest(m)
    (d / fc.MANIFEST_NAME).write_text(fc._json_dumps(fc._seal(m)))
    return d


class Reached(BaseException):
    pass


def _no_decode(monkeypatch, d, rx):
    reads = []
    monkeypatch.setattr(np, "load", lambda *a, **k: (reads.append(a[0]), (_ for _ in ()).throw(Reached()))[1])
    with pytest.raises(REFUSE, match=rx):
        ccc.read_continued_origin_cuda(d, IDENT, available_bytes_fn=AVAIL)
    assert reads == []
    monkeypatch.undo()


def _npy(a):
    b = io.BytesIO()
    np.save(b, a, allow_pickle=False)
    return b.getvalue()


def test_the_host_reader_reads_the_forged_control_and_every_old_reader_refuses_it(cpu_ck, tmp_path):
    ns, ck, _d = cpu_ck
    d = forge(ck, tmp_path, "ok")
    o = ccc.read_continued_origin_cuda(d, IDENT, available_bytes_fn=AVAIL)
    assert o.routing_generation == ns.state.topology.generation and np.array_equal(o.routing_source_full, ns.event._initial_topology.source_z_full)
    assert ccc.origin_topology(o, ns.original).topology_sha256 == ns.event._initial_topology.topology_sha256
    with pytest.raises(REFUSE, match="not a continued-origin"):
        cc.read_continued_origin(d, IDENT, available_bytes_fn=AVAIL)  # the CPU continued reader refuses the CUDA family
    with pytest.raises(REFUSE, match="not a continued-origin"):
        ccc.read_continued_origin_cuda(ck, IDENT, available_bytes_fn=AVAIL)  # and the CUDA reader the CPU one
    with pytest.raises(REFUSE):
        fc.load_floating_pond_checkpoint(d, ns.event, IDENT, available_bytes_fn=AVAIL)  # the historical CPU loader refuses it by schema
    with pytest.raises(REFUSE):
        cc.load_floating_continued_checkpoint(d, ns.event, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)


def test_kernel_source_header_table_and_size_guards_refuse_before_any_decode(cpu_ck, tmp_path, monkeypatch):
    ns, ck, _d = cpu_ck
    n = ns.event._initial_topology.source_z_full.shape
    _no_decode(monkeypatch, forge(ck, tmp_path, "kernel", kernel=False), "kernel")
    _no_decode(monkeypatch, forge(ck, tmp_path, "src", manifest=lambda m: m["source"].update(maple_syrup="0" * 64)), "source pair")
    _no_decode(monkeypatch, forge(ck, tmp_path, "big", member=lambda m: m.__setitem__("co_routing_source.npy", _npy(np.zeros((100, 100))))), "header|member|archive")
    _no_decode(monkeypatch, forge(ck, tmp_path, "f32", member=lambda m: m.__setitem__("co_routing_source.npy", _npy(np.zeros(n, dtype=np.float32)))), "header|member|archive")
    _no_decode(monkeypatch, forge(ck, tmp_path, "short", member=lambda m: m.__setitem__("co_routing_source.npy", m["co_routing_source.npy"][:-8])), "header|member|archive")
    _no_decode(monkeypatch, forge(ck, tmp_path, "extra", member=lambda m: m.__setitem__("stray.npy", _npy(np.zeros(3)))), "members|table|archive member")
    for name, edit in {"shape": lambda m: m["binding"].update(shape=[3.0, 5]), "hex": lambda m: m["binding"]["initial_topology"].update(source_sha256="XYZ"), "keys": lambda m: m.pop("continued_origin"),
                       "extra_key": lambda m: m.update(unknown=1), "cpu_schema": lambda m: m.update(schema=fc.CONTINUED_SCHEMA), "profile": lambda m: m.update(profile=fc.CONTINUED_PROFILE),
                       "table": lambda m: m["payload"]["arrays"]["datum_full"].update(shape=[10**9, 10**9])}.items():
        _no_decode(monkeypatch, forge(ck, tmp_path, name, manifest=edit), ".")


def test_a_tiny_admission_refuses_the_host_reader_before_any_hash_or_read(cpu_ck, tmp_path, monkeypatch):
    from maple_syrup import native_floating_soil as nfs

    _ns, ck, _d = cpu_ck
    d = forge(ck, tmp_path, "adm")
    hashed = []
    real = pc._hash_file
    monkeypatch.setattr(pc, "_hash_file", lambda *a, **k: (hashed.append(a[0]), real(*a, **k))[1])
    with pytest.raises(nfs.NativeFloatingMemoryError):
        ccc.read_continued_origin_cuda(d, IDENT, available_bytes_fn=lambda: 0)
    assert hashed == []


def test_the_cpu_reader_behaviour_is_unchanged_by_the_shared_core(cpu_ck):
    ns, ck, _d = cpu_ck
    o = cc.read_continued_origin(ck, IDENT, available_bytes_fn=AVAIL)
    assert o.schema == fc.CONTINUED_SCHEMA and o.routing_generation == ns.state.topology.generation


def test_the_gpu_definitions_have_no_assert_with_a_bare_conditional_expression_on_the_right_of_a_comparison():
    """Regression guard of the confirmed precedence bug (`assert x == A if windowed else B` is `assert (x == A) if ... else B`, always truthy for the dense branch): every IfExp inside an assert must be parenthesised as an operand."""
    from pathlib import Path

    tree = ast.parse((Path(__file__).resolve().parent / "test_gpu_definitions.py").read_text())
    bad = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert) and isinstance(n.test, ast.IfExp)]
    assert bad == []


def test_the_gpu_fixture_uses_the_above_threshold_deposit_and_the_cpu_control_before_any_device_claim():
    import inspect

    src = (Path_(__file__).resolve().parent / "test_gpu_definitions.py").read_text()
    assert "cpu_control" in src and "assert_matched" in src and "deposit_rise_m" in src and "lake_kwargs()" in src and inspect is not None
    assert src.index("cpu_control") < src.index("device_event(")  # the genuine matched CPU control is built FIRST
