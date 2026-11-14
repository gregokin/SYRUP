"""S2-CUDA HOST tests (no GPU/CuPy). They prove: the closed profile family and its distinctness; that the CPU foundation outer, the daily producer, the child codecs and every accepted numerical/source module are byte-identical to the
pre-edit manifest while ONLY the declared hooks changed (and that the CPU branches of the changed hooks are textually preserved); the resolution of the two bridge merge conflicts (both the CUDA signatures/quota arguments AND the CPU
`_build_context` / `restore_closed_day` additions survive); the early typed refusals (CURRENT-device gate before any reader/IO/factory); and the pure memory arithmetic of the continued-child header reader on GENUINE CPU continued
children relabelled as the CUDA family (a control of the arithmetic only, never a CUDA file). Real device behaviour is in `test_gpu_definitions.py` (not run here)."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_checkpoint_cuda as fcc
from maple_syrup import floating_context_checkpoint as cc
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_daily_bridge_cuda as bc
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_lifecycle_checkpoint as cpul
from maple_syrup import floating_lifecycle_checkpoint_cuda as lpcc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase25_source_profile"))
import p25_source_profile as p25

HERE = Path(__file__).resolve()
SRC = HERE.parents[2] / "src" / "maple_syrup"
PRE = HERE.parent / "physical_lifecycle_cuda_preedit_source_baseline.json"  # the exact public source-only baseline (the original pre-edit manifest's `files` mapping, values unchanged); no ignored local file is read

# AST digests (first 20 hex of sha256 of ast.dump) of the CPU-only definitions the merge must leave exactly as the accepted CPU foundation has them
CPU_BRIDGE = {"_build_context": "1f591b8fb1f8ef5295a0", "restore_closed_day": "3d2d2680a6bf3bc5a69a"}
CPU_LC = {"saveable_head": "7d5f7df411763f7bca02", "check_closed_day": "bef3b1fe4ce8f8409d07", "_scalar_ledger": "9be2e7a7d7c89f9648d2", "head_record": "bdb5cb0654b02ab70548",
          "_check_provenance": "d4be533f1b8264c38c8c", "_balance_check": "fee38d3180533ee0b29a", "install_restored_head": "1f6b633d29f869bc57fb"}


def _defs(path):
    t = ast.parse(Path(path).read_text())
    return {n.name: hashlib.sha256(ast.dump(n).encode()).hexdigest()[:20] for n in t.body if isinstance(n, ast.FunctionDef)}


def test_profile_family_is_closed_and_distinct():
    assert lpcc.PROFILE == "maple-syrup-floating-lifecycle-checkpoint/2-cuda" and lpcc.PROFILE != cpul.PROFILE
    assert lpcc.MANIFEST_KEYS == cpul.MANIFEST_KEYS and lpcc.MANIFEST_NAME == cpul.MANIFEST_NAME and lpcc.CHILD_NAMES == cpul.CHILD_NAMES
    assert lpcc.CHILD_SCHEMAS == {"dense": fcc.CONT_GPU_SCHEMA, "windowed": fcc.CONT_GPU_WINDOWED_SCHEMA}
    assert not set(lpcc.CHILD_SCHEMAS.values()) & (set(cpul.CHILD_SCHEMAS.values()) | set(fcc.READ_SCHEMAS) | set(fc.READ_SCHEMAS))
    assert cpul.PROFILE in ("maple-syrup-floating-lifecycle-checkpoint/1",) and "2-cuda" in lpcc.PROFILE
    assert lpcc.RestoredLifecycle is cpul.RestoredLifecycle  # the minimal default return: lineage/head/closed/kind/seal ONLY
    assert [f for f in cpul.RestoredLifecycle.__dataclass_fields__] == ["lineage", "head", "closed", "kind", "manifest_sha256"]


def _historical_byte_guard():
    pre = json.loads(PRE.read_text())["files"]
    rel = {k.split("/")[-1]: v for k, v in pre.items() if k.startswith("src/maple_syrup/") and k.endswith(".py")}
    differing = {n for n, h in rel.items() if hashlib.sha256((SRC / n).read_bytes()).hexdigest() != h}
    present = {p.name for p in SRC.glob("*.py")}
    # the bridge differs because the two merge conflicts were resolved; the lifecycle gained ONLY the explicit CUDA restore dispatch; the CUDA bridge helper gained ONLY `restore_closed_day_cuda`
    assert differing == {"floating_event_daily_bridge.py", "floating_event_lifecycle.py", "floating_event_daily_bridge_cuda.py"}
    assert present - set(rel) == {"floating_lifecycle_checkpoint_cuda.py"}
    for name in ("floating_lifecycle_checkpoint.py", "floating_daily.py", "floating_daily_cuda.py", "floating_daily_output.py", "floating_continued_checkpoint.py", "floating_continued_checkpoint_cuda.py", "floating_checkpoint.py",
                 "floating_checkpoint_cuda.py", "floating_context_checkpoint.py", "floating_context_checkpoint_cuda.py", "floating_held_context.py", "floating_held_context_cuda.py", "floating_pond.py", "pond_event.py",
                 "floating_event_lifecycle_cuda.py", "pond_checkpoint.py", "floating_resume.py", "floating_resume_cuda.py"):
        assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == rel[name], name


def _historical_ast_guard():
    b, la = _defs(SRC / "floating_event_daily_bridge.py"), _defs(SRC / "floating_event_lifecycle.py")
    assert {k: b[k] for k in CPU_BRIDGE} == CPU_BRIDGE
    assert {k: la[k] for k in CPU_LC} == CPU_LC

# the modules the byte test names as "accepted, byte-identical to the pre-edit baseline"; Phase 2.5 DELIBERATELY changes some of them: those are then proven by the whole-source lock (their locked bytes), never skipped
ACCEPTED_NAMED = ("floating_lifecycle_checkpoint.py", "floating_daily.py", "floating_daily_cuda.py", "floating_daily_output.py", "floating_continued_checkpoint.py", "floating_continued_checkpoint_cuda.py",
                  "floating_checkpoint.py", "floating_checkpoint_cuda.py", "floating_context_checkpoint.py", "floating_context_checkpoint_cuda.py", "floating_held_context.py", "floating_held_context_cuda.py",
                  "floating_pond.py", "pond_event.py", "floating_event_lifecycle_cuda.py", "pond_checkpoint.py", "floating_resume.py", "floating_resume_cuda.py")
P25_CHANGED_DEFS = {"saveable_head": "floating_event_lifecycle.py"}   # the ONLY CPU-lifecycle definition Phase 2.5 changes (the vegetation tag); proven by the whole-source lock + behavioural regression, not by an AST digest


def test_cpu_foundation_and_accepted_modules_are_byte_identical_to_the_pre_edit_manifest_and_only_the_declared_files_differ():
    """Branch chosen ONLY by the literal-pinned package: the historical base169 package runs the original proof; the Root-locked final package runs the whole-source proof, where the old differing set and added module are
    preserved and every Phase 2.5 locked change/addition is accounted for explicitly (nothing else may differ)."""
    branch = p25.require_branch(SRC, historical=("base169_02388ad",))
    if branch.kind == "historical":
        _historical_byte_guard()
        return
    v = branch.verified
    pre = json.loads(PRE.read_text())["files"]
    rel = {k.split("/")[-1]: h for k, h in pre.items() if k.startswith("src/maple_syrup/") and k.endswith(".py")}
    differing = {n for n, h in rel.items() if v.files[n] != h}
    assert differing == {"floating_event_daily_bridge.py", "floating_event_lifecycle.py", "floating_event_daily_bridge_cuda.py"} | v.changed
    assert set(v.files) - set(rel) == {"floating_lifecycle_checkpoint_cuda.py"} | v.added | {v.wh1_path}
    assert {p.name for p in SRC.glob("*.py")} == set(v.files)                                           # a flat package of top-level production modules
    for name in ACCEPTED_NAMED:
        if name in v.changed:
            assert v.files[name] != rel[name], name           # a locked Phase 2.5 change: its bytes are the locked bytes (proven by the whole-source lock above)
        else:
            assert v.files[name] == rel[name], name


def test_the_cpu_only_definitions_are_ast_identical_to_the_accepted_cpu_foundation():
    """Same branch rule. Final branch: every CPU definition NOT declared changed by Phase 2.5 keeps its accepted AST digest; each declared changed definition lives in a module the lock records as changed."""
    branch = p25.require_branch(SRC, historical=("base169_02388ad",))
    if branch.kind == "historical":
        _historical_ast_guard()
        return
    v = branch.verified
    b, la = _defs(SRC / "floating_event_daily_bridge.py"), _defs(SRC / "floating_event_lifecycle.py")
    assert set(P25_CHANGED_DEFS) <= set(CPU_LC) and not set(P25_CHANGED_DEFS) & set(CPU_BRIDGE)
    assert all(mod in v.changed for mod in P25_CHANGED_DEFS.values())
    assert {k: b[k] for k in CPU_BRIDGE} == CPU_BRIDGE
    untouched = {k: h for k, h in CPU_LC.items() if k not in P25_CHANGED_DEFS}
    assert {k: la[k] for k in untouched} == untouched and len(untouched) == len(CPU_LC) - len(P25_CHANGED_DEFS)



def test_both_merge_conflicts_keep_the_cuda_arguments_and_the_cpu_hooks():
    src = (SRC / "floating_event_daily_bridge.py").read_text()
    assert "<<<<<<<" not in src and ">>>>>>>" not in src and "\n=======\n" not in src
    for fn in (br.open_day, br.hand_back):  # the CUDA quota/gate arguments of the C4 branch survive
        p = inspect.signature(fn).parameters
        assert "host_available_bytes_fn" in p and "host_reserve_bytes" in p
    p = inspect.signature(br.open_day).parameters
    assert "memory_budget_bytes" in p and "host_memory_budget_bytes" in p
    assert callable(br._build_context) and callable(br.restore_closed_day)  # the CPU closed-state hooks survive
    open_src = inspect.getsource(br.open_day)
    assert open_src.index("open_day_cuda") < open_src.index("_build_context(")  # the CUDA dispatch precedes every CPU array
    assert "bc.open_day_cuda" in open_src and "only to a CUDA boundary" in open_src
    assert "restore_closed_day_cuda" in bc.__all__


def test_the_lifecycle_cuda_hooks_preserve_the_cpu_branches_text():
    sr = inspect.getsource(lc.stage_restored_head)
    assert "np.array_equal(np.asarray(payload.depth_m), np.asarray(held.depth0_m))" in sr  # CPU depth comparison unchanged
    assert "float(fsl.floating_storage_m3(held.column, payload.state)) + float(np.asarray(hyd.depth_m).sum()) * area" in sr  # CPU balance unchanged
    assert inspect.signature(lc.stage_restored_head).parameters["cuda_quota"].default is None
    assert inspect.signature(lc.stage_restored_head).parameters["cuda_quota"].kind is inspect.Parameter.KEYWORD_ONLY
    assert 'v.backend not in ("numpy", "numba", "cuda")' in inspect.getsource(lc.verified_restore)
    cuda_part = sr[sr.index("if cuda:"):]
    assert "arrays_equal_cuda" in sr and "stored_m3_cuda" in cuda_part and "meaningless CPU argument" in sr
    assert "np.asarray" not in sr[sr.index("same_depth = flc") : sr.index("else:\n        same_depth")]  # the CUDA depth comparison never mirrors a device array


def test_save_and_load_refuse_before_any_reader_io_or_factory_without_a_device(tmp_path):
    from lcp_builders import IDENT, boundary_ns

    ns = boundary_ns(tmp_path / "a", "numpy", False)
    reads, calls = [], []

    def reader():
        reads.append(1)
        return 1 << 40

    target = tmp_path / "ck"
    with pytest.raises(lpcc.LifecycleCheckpointCudaError, match="device gate"):
        lpcc.save_floating_lifecycle_checkpoint_cuda(target, ns.lin, lambda *a: calls.append(1), IDENT, original_topology=ns.original, available_bytes_fn=reader, host_available_bytes_fn=reader)
    assert not target.exists() and not list(tmp_path.glob(".ck.partial-*")) and reads == [] and calls == []
    with pytest.raises(lpcc.LifecycleCheckpointCudaError, match="original_topology"):
        lpcc.save_floating_lifecycle_checkpoint_cuda(target, ns.lin, ns.factory, IDENT, original_topology=None, available_bytes_fn=reader)
    with pytest.raises(lpcc.LifecycleCheckpointCudaError, match="device gate"):
        lpcc.load_floating_lifecycle_checkpoint_cuda(tmp_path / "missing", lambda *a: calls.append(1), IDENT, "0" * 64, original_topology=ns.original, expected_device_id=0, available_bytes_fn=reader, host_available_bytes_fn=reader)
    for bad in (True, -1, "0", 1.5):
        with pytest.raises(lpcc.LifecycleCheckpointCudaError, match="expected_device_id"):
            lpcc.load_floating_lifecycle_checkpoint_cuda(tmp_path / "missing", ns.factory, IDENT, "0" * 64, original_topology=ns.original, expected_device_id=bad)
    assert reads == [] and calls == []
    assert lc.saveable_head(ns.lin)[0] == "boundary"  # nothing about the head changed


def test_a_cpu_file_is_not_read_by_the_cuda_loader_and_the_cpu_loader_refuses_the_cuda_profile_by_name(tmp_path):
    src = inspect.getsource(cpul._preflight)
    assert 'data["profile"] != PROFILE' in src  # the CPU preflight pins its own profile string: a `/2-cuda` manifest is refused by name
    assert cpul.PROFILE != lpcc.PROFILE
    assert 'data["profile"] != PROFILE' in inspect.getsource(lpcc._preflight)


# ---- the continued-child header reader arithmetic on GENUINE CPU continued children (control of the arithmetic only) ---------------------------------------------------------------------------
@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_child_models_read_the_continued_family_headers_incl_the_origin_array_and_the_windowed_cache(tmp_path, windowed):
    from cont_builders import lineage, save

    ns = lineage(tmp_path / "a", "numpy", windowed)
    ck = save(ns, tmp_path / "ck")
    cman = json.loads((ck / fc.MANIFEST_NAME).read_text())
    child_files = {n: {"size": (ck / n).stat().st_size} for n in cc.CHILD_NAMES}
    cman["schema"] = fcc.CONT_GPU_WINDOWED_SCHEMA if windowed else fcc.CONT_GPU_SCHEMA  # arithmetic control only: the header of a CPU file relabelled
    L = ns.occ.vertical.n_layers
    meta = {"n_classes": ns.event.nc, "n_layers": L}
    dev, host, detail = lpcc._child_models(cman, meta, child_files, 64)
    arrays = cman["payload"]["arrays"]
    want_table = sum(int(np.prod(r["shape"], dtype=np.int64)) * np.dtype(r["dtype"]).itemsize for r in arrays.values())
    assert detail["table_bytes"] == want_table and fc.ORIGIN_ARRAY in arrays and detail["windowed"] is windowed
    ny, nx = ns.event.shape
    without = dict(cman)
    without["payload"] = dict(cman["payload"], arrays={k: v for k, v in arrays.items() if k != fc.ORIGIN_ARRAY})
    d2, h2, det2 = lpcc._child_models(without, meta, child_files, 64)
    assert det2["table_bytes"] == want_table - (ny + 2) * (nx + 2) * 8 and host > h2 and dev >= d2  # the O(N padded) origin array is modelled
    N = ny * nx
    exp_dev, exp_host = fcc.load_models(L, N, cman["binding"]["n_classes"], cman["binding"]["phase_bins"], table_bytes=want_table, dev_table_bytes=want_table, native_bytes=detail["native_bytes"], V=detail["V"], K=detail["K"],
                                        n_active=N, batch_cells=64, windowed=windowed, cache_bytes=detail["cache_bytes"])
    assert (dev, host) == (exp_dev, exp_host)  # the accepted `fcc.load_models`, from the header's own dimensions
    if windowed:
        assert detail["cache_bytes"] == fc.deep_cache_bound_bytes(cman["deep_storage"], ns.event.nc) > 0 and detail["V"] == 0  # the mapping-cache bound, never the archive size
    else:
        assert detail["V"] > 0 and detail["cache_bytes"] == 0
    bad = json.loads(json.dumps(cman))
    bad["binding"]["n_classes"] = "x"
    with pytest.raises(lpcc.LifecycleCheckpointCudaError, match="malformed"):
        lpcc._child_models(bad, meta, child_files, 64)
    assert shutil is not None


def test_the_fresh_workers_explicit_path_list_covers_its_whole_import_closure_without_cupy():
    """A NEW interpreter with ONLY the worker's own `sys.path` edits (no conftest) imports everything the SAVE and LOAD paths import (the chain driver, the CC2 static recipe, the accepted `static_case`, the GPU fixtures by path) --
    and never imports CuPy (a path omission, as the CC2 worker once had, fails here on the host)."""
    import os
    import subprocess
    import sys

    here = HERE.parent
    code = ("import sys\nimport gpu_lifecycle_worker as w\nimport chain, cc2_static_case, static_case, lcp_builders, bridge_builders\nimport test_gpu_definitions\n"
            "assert 'cupy' not in sys.modules, 'cupy was imported'\nprint('closure ok')\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env.update(CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, cwd=str(here), timeout=300, check=False)
    assert r.returncode == 0 and "closure ok" in r.stdout, r.stderr[-3000:]
    src = (here / "gpu_lifecycle_worker.py").read_text()
    body = src[src.index("def main("):]
    assert body.index("tripped = disable_original_constructors()") < body.index("import cc2_static_case")  # the tripwires are installed BEFORE the recipe module is even imported


# ---- correction01: D1 provenance vocabulary, D2 default-reader plumbing, D3 error-class preservation ---------------------------------------------------------------------------------------------------
def _literal_keys(fn, var):
    """The key set of the dict literal assigned to `var` in `fn`'s source (the PRODUCER's own vocabulary, read from the accepted code, not restated)."""
    from maple_syrup import floating_event_lifecycle_cuda as flc  # noqa: F401

    out = set()
    for node in ast.walk(ast.parse(textwrap_dedent(inspect.getsource(fn)))):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == var for t in node.targets) and isinstance(node.value, ast.Dict):
            out |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    return out


def textwrap_dedent(s):
    import textwrap

    return textwrap.dedent(s)


def test_cuda_provenance_validator_accepts_exactly_the_producers_vocabulary_and_roundtrips_the_dict_unchanged():
    from maple_syrup import floating_event_lifecycle_cuda as flc
    from maple_syrup import floating_held_context_cuda as fhc

    keys_open, keys_close = _literal_keys(flc.open_cuda, "prov"), _literal_keys(flc.close_cuda, "prov")
    assert keys_open and keys_close and (keys_open | keys_close) <= lc._PROVENANCE_KEYS_CUDA and {"backend"} <= keys_open and {"backend", "kernel", "area_m2"} <= keys_close  # bound to the real producers
    w = "a" * 64
    opened = {"cold_anchor_witness": w, "property_basis": "initial_full", "execution_epoch": None, "cold_content_hash": None, "backend": "cuda"}
    refreshed = {"cold_anchor_witness": w, "property_basis": lc.fer.PROPERTY_BASIS, "execution_epoch": "b" * 64, "cold_content_hash": None, "area_m2": 0.25, "backend": "cuda", "kernel": fhc.kernel_descriptor()}
    for prov in (opened, refreshed):
        got = lc._check_provenance_cuda(prov, w)
        assert got == prov and got is not prov  # the ORIGINAL canonical dict (a copy), nothing dropped or re-derived
    bad = {
        "unknown key": dict(refreshed, extra=1), "wrong backend": dict(refreshed, backend="numpy"), "no backend": {k: v for k, v in refreshed.items() if k != "backend"},
        "wrong kernel": dict(refreshed, kernel=dict(refreshed["kernel"], fmc_source_sha256="0" * 64)), "kernel missing on a refreshed root": {k: v for k, v in refreshed.items() if k != "kernel"},
        "wrong witness": dict(refreshed, cold_anchor_witness="c" * 64), "bad area": dict(refreshed, area_m2="x"), "epoch": dict(refreshed, execution_epoch="short"), "hash": dict(refreshed, cold_content_hash="d" * 64),
        "not a dict": [], "basis": dict(opened, property_basis="other"),
    }
    for name, prov in bad.items():
        with pytest.raises(lc.LifecycleError):
            lc._check_provenance_cuda(prov, w)
        assert name


def test_the_cpu_provenance_check_stays_strict_and_unchanged_and_refuses_the_cuda_keys():
    w = "a" * 64
    cpu = {"cold_anchor_witness": w, "property_basis": "initial_full", "execution_epoch": None, "cold_content_hash": None}
    assert lc._check_provenance(cpu, w) == cpu
    for extra in ({"backend": "cuda"}, {"kernel": {}}):
        with pytest.raises(lc.LifecycleError):
            lc._check_provenance({**cpu, **extra}, w)  # the CPU vocabulary does not learn the CUDA keys
    assert _defs(SRC / "floating_event_lifecycle.py")["_check_provenance"] == "d4be533f1b8264c38c8c"
    assert 'if vert.backend == "cuda" else _check_provenance(' in inspect.getsource(lc.stage_restored_head)  # explicit backend dispatch; the CPU call is the original one


def test_default_readers_are_resolved_once_after_the_gate_and_callers_readers_are_retained(monkeypatch):
    from maple_syrup import floating_soil_layers_cuda as fcu
    from maple_syrup import spatial_soil_layers as sp

    made = []
    monkeypatch.setattr(fcu, "device_reader", lambda cp: (made.append(cp), lambda: 123)[1])
    q = {"available_bytes_fn": None, "host_available_bytes_fn": None, "reserve_bytes": 7, "host_reserve_bytes": 9, "memory_budget_bytes": 11, "host_memory_budget_bytes": 13}
    out = lpcc._resolve_readers(q, "CP")
    assert made == ["CP"] and out["available_bytes_fn"]() == 123 and out["host_available_bytes_fn"] is sp.default_host_available_bytes
    assert q["available_bytes_fn"] is None  # the input is untouched
    assert {k: out[k] for k in ("reserve_bytes", "host_reserve_bytes", "memory_budget_bytes", "host_memory_budget_bytes")} == {"reserve_bytes": 7, "host_reserve_bytes": 9, "memory_budget_bytes": 11, "host_memory_budget_bytes": 13}  # nothing invented
    mine, host = (lambda: 1), (lambda: 2)
    kept = lpcc._resolve_readers(dict(q, available_bytes_fn=mine, host_available_bytes_fn=host), "CP")
    assert kept["available_bytes_fn"] is mine and kept["host_available_bytes_fn"] is host and made == ["CP"]  # a caller's readers are retained and no default is resolved


def test_every_entry_resolves_the_readers_right_after_the_device_gate_and_before_any_use():
    for fn, gate in ((lpcc.save_floating_lifecycle_checkpoint_cuda, "fcc.gate("), (lpcc.load_floating_lifecycle_checkpoint_cuda, "gate_expected"), (lpcc.lifecycle_load_need_bytes_cuda, "gate_expected")):
        src = inspect.getsource(fn)
        assert src.index(gate) < src.index("_resolve_readers(q, cp)")
        first_use = min(i for i in (src.find('q["available_bytes_fn"]'), src.find("_preflight("), src.find("fcxc._admit("), src.find("from_boundary(")) if i >= 0)
        assert src.index("_resolve_readers(q, cp)") < first_use, fn.__name__


def test_resource_errors_are_not_converted_to_semantic_errors_in_the_closed_day_restore():
    from maple_syrup import native_floating_soil as nfs
    from maple_syrup import subsurface as ss

    assert issubclass(fsl_mem(), ss.SubsurfaceMemoryError) and issubclass(nfs.NativeFloatingMemoryError, ss.SubsurfaceMemoryError)
    assert issubclass(ss.SubsurfaceMemoryError, MemoryError)  # the confirmed root cause: the broad `MemoryError` clause caught it
    assert not issubclass(ss.SubsurfaceMemoryError, (lc.LifecycleError, br.BridgeError, lpcc.LifecycleCheckpointCudaError))
    tree = ast.parse(textwrap_dedent(inspect.getsource(bc.restore_closed_day_cuda)))
    handlers = [h for n in ast.walk(tree) if isinstance(n, ast.Try) for h in n.handlers]
    names = [ast.unparse(h.type) for h in handlers]
    assert names[0] == "MemoryError" and any(isinstance(b, ast.Raise) and b.exc is None for b in handlers[0].body)  # ALL MemoryError subclasses re-raised BEFORE the semantic conversion
    assert "MemoryError" not in names[1] and "br.BridgeError" in ast.unparse(handlers[1])  # the semantic conversion (source/ledger/receipt errors) no longer swallows a resource error


def test_the_production_handler_clauses_reraise_every_memory_error_class_and_convert_only_semantic_errors():
    """BEHAVIOUR of the exact production `except` clauses of `restore_closed_day_cuda` (extracted from its AST and run over a synthetic raising body; nothing else of the function runs)."""
    from maple_syrup import native_floating_soil as nfs
    from maple_syrup import recharge as rc

    tree = ast.parse(textwrap_dedent(inspect.getsource(bc.restore_closed_day_cuda)))
    handlers = [h for n in ast.walk(tree) if isinstance(n, ast.Try) for h in n.handlers]
    src = "def probe(exc):\n    try:\n        raise exc\n" + "".join("    " + line + "\n" for line in "\n".join(ast.unparse(h) for h in handlers).splitlines())
    ns = {**vars(bc), "br": br}  # `br` is the function-local `_bridge()` alias of the real bridge module
    exec(compile(src, "<probe>", "exec"), ns)  # noqa: S102

    class DeviceOOM(MemoryError):  # stands for a CuPy `OutOfMemoryError` (a MemoryError subclass)
        pass

    for exc in (MemoryError("plain"), DeviceOOM("device"), fsl_mem()("typed"), nfs.NativeFloatingMemoryError("native")):
        with pytest.raises(type(exc)) as info:
            ns["probe"](exc)
        assert info.value is exc  # the SAME object: not converted, not wrapped
    from maple_syrup import floating_daily as fd
    from maple_syrup import floating_soil_layers as fsl

    for exc in (rc.RechargeError("x"), KeyError("k"), TypeError("t"), fd.DailyRefusal("state", "r"), fsl.FloatingSoilError("s")):
        with pytest.raises(br.BridgeError, match="cannot be restored"):
            ns["probe"](exc)


def fsl_mem():
    from maple_syrup import floating_soil_layers as fsl

    return fsl.FloatingMemoryError


def test_the_tamper_table_mutations_really_change_the_record_and_the_cursor_control_is_the_daily_cursor(tmp_path):
    """LOGICAL control of the table the device test drives, on a genuine CPU closed-day file (same lifecycle record schema): every `outer` mutation changes the outer manifest and its recomputed seal, the ORIGINAL head cursor is
    None (so `head.cursor = None` would be a no-op) while the daily cursor is not, and the semantic-vs-file-identity seal modes are as declared."""
    import copy

    import tamper_cases
    from lcp_builders import IDENT, closed_day_ns, save

    ns = closed_day_ns(tmp_path / "a", "numpy", False, days=1)
    sv = save(ns, tmp_path / "ck", ns.closed)
    manifest = json.loads((sv.path / cpul.MANIFEST_NAME).read_text())
    D = manifest["lifecycle"]
    assert D["head"]["cursor"] is None and D["daily"]["cursor"] is not None and IDENT
    cases = tamper_cases.lifecycle_cases(D, False)
    assert [c[0] for c in cases][:8] == ["profile_cpu", "lifecycle_kind", "routing_generation", "bridge_stable", "daily_cursor_ordinal", "seq_negative_own_seal", "seq_plus5_writer_seal", "kernel"]
    assert {c[3] for c in cases} == {"own", "writer"} and [c[0] for c in cases if c[3] == "writer"] == ["seq_plus5_writer_seal"]
    for name, kw, rx, which in cases:
        if "outer" not in kw or name in ("profile_cpu", "backend"):  # the last two set the CPU profile / numpy backend: a no-op ON a CPU file, a real mutation on the CUDA file (the device test asserts that itself)
            continue
        forged = copy.deepcopy(manifest)
        kw["outer"](forged)
        assert forged != manifest, f"{name}: a NO-OP mutation"
        assert cc._seal(forged)["manifest_sha256"] != manifest["manifest_sha256"] and rx, name
    # the old no-op: head.cursor = None leaves the manifest EXACTLY equal (the confirmed test defect)
    noop = copy.deepcopy(manifest)
    noop["lifecycle"]["head"].update(cursor=None)
    assert noop == manifest
