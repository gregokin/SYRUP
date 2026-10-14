"""CW-B76: SYRUP accounting steps on a windowed bed with characteristic CSR transport use MAPLE B75's PRIVATE paired
pickup -> transport callback -> deposit transaction; every other combination keeps the original two-call path.

Oracle: the frozen B73 `sediment_event.py` (hash pinned), executed as an isolated module with its OWN state/control
types against the same MAPLE (frozen B75: its public windowed entry is raw-identical to M67, so the B73 module runs the
original two-call path). Differentials compare a complete recursive raw record (arrays: dtype/shape/bytes; NumPy
scalars: dtype/bytes; native floats: raw bytes; nested dataclasses/dicts/slot handles). CPU tests run here; CuPy
variants are for root (a skip claims nothing).
"""
from __future__ import annotations

import ast
import dataclasses
import gc
import hashlib
import importlib
import json
import os
import struct
import subprocess
import sys
import textwrap
import tracemalloc
import types
import weakref
from collections import OrderedDict
from pathlib import Path, PurePath

import numpy as np
import pytest
import test_accounting_step as b20
from b20_cases import build, host

import maple_syrup.sediment_bed as bedmod
import maple_syrup.sediment_event as new
from maple_syrup.sediment_bed import BedState
from maple_syrup.sediment_transport import TransportStepRejected

TREE = Path(__file__).resolve().parents[2]
PKG = TREE / "src" / "maple_syrup"
B73_TREE = Path("/home/okin/SYRUP/outputs/conservative_water/cw_b73_runtime_frozen")
B73 = B73_TREE / "src" / "maple_syrup"
B73_SHA = {"sediment_event.py": "e983733cb90acc295b03a1613497896da53eeb071a9445241cd3424e23eabaa2",
           "sediment_bed.py": "72c4dc06c075b89c303d60cdd3c1597acdd9be3633484e3e82c55f43605f4278"}
B75 = Path("/home/okin/SYRUP/outputs/dependencies/maple_cw_b75_paired_frozen")
B75_DIGEST = "c39e8b3b25dc11e499c2eab184bb3cb54a3adda9528e8f57c775b6c933a64fbf"
M67 = Path("/home/okin/SYRUP/outputs/dependencies/maple_cw_b67_late_boundary_frozen")
ORACLE = "_cw_b76_b73_sediment_event"
BACKENDS = ["numpy", "cupy"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def old():
    path = B73 / "sediment_event.py"
    assert _sha(path) == B73_SHA["sediment_event.py"], "frozen B73 oracle changed"
    module = types.ModuleType(ORACLE)
    module.__file__ = str(path)
    sys.modules[ORACLE] = module
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)  # noqa: S102 -- hash-pinned B73 text
    return module


def as_types(module, obj):
    cls = getattr(module, type(obj).__name__)
    return cls(**{f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if f.init})


# --- complete recursive raw record (as CW-B65 correction 1) ---------------------------------------------------------
_EVENT = {new.__name__, ORACLE}
_CACHE_SLOTS = {"maple.surface.voxels.deep_store.DeepStoreVersion": {"_memmaps", "_memmap_lock"}}


def _tn(t):
    return f"{'maple_syrup.sediment_event' if t.__module__ in _EVENT else t.__module__}.{t.__qualname__}"


def _is_array(v):
    return isinstance(v, np.ndarray) or (type(v).__module__.split(".")[0] == "cupy" and hasattr(v, "__cuda_array_interface__"))


def tree(v, path="r", active=None):
    active = set() if active is None else active
    if isinstance(v, types.ModuleType):
        return ("module", v.__name__)
    if v is None:
        return ("None",)
    if type(v) in (bool, int, str):
        return (type(v).__name__, v)
    if type(v) is float:
        return ("float", struct.pack("<d", v).hex())
    if isinstance(v, np.generic):
        return ("np", v.dtype.str, v.tobytes())
    if _is_array(v):
        a = np.ascontiguousarray(host(v))
        return ("array", type(v).__module__.split(".")[0], a.dtype.str, a.shape, a.tobytes())
    if isinstance(v, PurePath):
        return ("path", str(v))
    assert id(v) not in active, f"cycle at {path}"
    active.add(id(v))
    try:
        t = type(v)
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return ("dc", _tn(t), tuple((f.name, tree(getattr(v, f.name), f"{path}.{f.name}", active))
                                       for f in dataclasses.fields(v)))
        if t in (dict, OrderedDict):
            return (t.__name__, tuple((tree(k, path, active), tree(x, f"{path}[{k!r}]", active)) for k, x in v.items()))
        if t in (tuple, list):
            return (t.__name__, tuple(tree(x, f"{path}[{i}]", active) for i, x in enumerate(v)))
        slots = [s for k in reversed(t.__mro__) for s in ([k.__dict__["__slots__"]] if isinstance(
            k.__dict__.get("__slots__"), str) else k.__dict__.get("__slots__", ()))]
        if slots and not hasattr(v, "__dict__"):
            skip = _CACHE_SLOTS.get(_tn(t), set())
            return ("slots", _tn(t), tuple((s, tree(getattr(v, s), f"{path}.{s}", active))
                                          for s in slots if s not in skip and hasattr(v, s)))
    finally:
        active.discard(id(v))
    raise AssertionError(f"unrecorded {_tn(type(v))} at {path}")


def spy_calls(monkeypatch, module):
    calls = {"paired": 0, "two_call": 0}
    rp, ra = module.apply_bed_pickup_transport_deposit if hasattr(module, "apply_bed_pickup_transport_deposit") \
        else None, module.apply_bed_demand
    if rp is not None:
        def p(*a, **k):
            calls["paired"] += 1
            return rp(*a, **k)
        monkeypatch.setattr(module, "apply_bed_pickup_transport_deposit", p)

    def t(*a, **k):
        calls["two_call"] += 1
        return ra(*a, **k)
    monkeypatch.setattr(module, "apply_bed_demand", t)
    return calls


def case_args(case, xp, tmp_path, scale=1.0):
    s, _ctx, ctl, schedule = build(case, xp, tmp_path)
    args = list(b20.step_args(s, schedule))
    args[3] = args[3] * scale  # rain rate: 0 -> zero pickup; > 1 -> larger pickup (deep misses)
    return s, tuple(args), ctl


def run_both(old, monkeypatch, xp, args, ctl, *, form="accounting", s=None, tmp_path=None):
    before = [v.tobytes() for v in b20.leaves(args[0])]
    pins = b20.store_pins(tmp_path) if tmp_path is not None else None
    fn_new = new._sediment_accounting_step if form == "accounting" else new.sediment_coupled_step
    fn_old = old._sediment_accounting_step if form == "accounting" else old.sediment_coupled_step
    with monkeypatch.context() as m:
        cn = spy_calls(m, new)
        got = fn_new(*args, ctl)
    with monkeypatch.context() as m:
        co = spy_calls(m, old)
        ref = fn_old(as_types(old, args[0]), *args[1:], as_types(old, ctl))
    assert [v.tobytes() for v in b20.leaves(args[0])] == before  # caller inputs unchanged
    if pins is not None:
        assert b20.store_pins(tmp_path) == pins  # no deep-store write before publication
    return got, ref, cn, co


# =====================================================================================================================
# Pins and scope (AST)
# =====================================================================================================================
def test_pins_and_only_the_two_runtime_files_differ_from_frozen_b73():
    import maple

    from maple_syrup.provenance import source_tree_digest

    assert Path(new.__file__).resolve() == PKG / "sediment_event.py"
    mine = {p.relative_to(PKG).as_posix(): _sha(p) for p in PKG.rglob("*.py") if "__pycache__" not in p.parts}
    frozen = {p.relative_to(B73).as_posix(): _sha(p) for p in B73.rglob("*.py") if "__pycache__" not in p.parts}
    assert set(mine) == set(frozen)
    assert sorted(k for k in mine if mine[k] != frozen[k]) == ["sediment_bed.py", "sediment_event.py"]
    assert all(frozen[k] == v for k, v in B73_SHA.items())
    if Path(maple.__file__).resolve().parent == (B75 / "source/src/maple").resolve():
        assert source_tree_digest(B75 / "source/src/maple").digest_sha256 == B75_DIGEST


def _defs(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def _rest(path):
    """Module-level statements other than definitions, imports and the module docstring."""
    body = ast.parse(path.read_text()).body
    return [ast.dump(n) for n in body[1:] if not isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import,
                                                                ast.ImportFrom))]


def test_sediment_bed_only_adds_the_adapter_and_entry_resolution():
    o, n = _defs(B73 / "sediment_bed.py"), _defs(PKG / "sediment_bed.py")
    assert set(n) == set(o) | {"paired_transaction_entry", "apply_bed_pickup_transport_deposit"}
    for name in o:
        assert ast.dump(o[name]) == ast.dump(n[name]), name
    added = [x for x in _rest(PKG / "sediment_bed.py") if x not in _rest(B73 / "sediment_bed.py")]
    assert len(added) == 2 and all("PAIRED_TRANSACTION_" in x for x in added)
    # the adapter's rebuild is EXACTLY apply_bed_demand's replace
    rebuild = [ast.unparse(x) for x in ast.walk(n["apply_bed_demand"]) if isinstance(x, ast.Call)
               and ast.unparse(x.func) == "replace"]
    adapter = [ast.unparse(x) for x in ast.walk(n["apply_bed_pickup_transport_deposit"]) if isinstance(x, ast.Call)
               and ast.unparse(x.func) == "replace"]
    assert rebuild == adapter and len(adapter) == 1


def _strip_doc(fn):
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return body


class _Rename(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "upwind_bed":
            return ast.copy_location(ast.Name(id="bed", ctx=node.ctx), node)
        return node


def test_sediment_event_delta_is_the_factoring_and_the_paired_branch_only():
    o, n = _defs(B73 / "sediment_event.py"), _defs(PKG / "sediment_event.py")
    assert set(n) == set(o) | {"_transport_with_substep_retry"}
    for name in set(o) - {"_sediment_coupled_core"}:
        assert ast.dump(o[name]) == ast.dump(n[name]), name
    added = [x for x in _rest(PKG / "sediment_event.py") if x not in _rest(B73 / "sediment_event.py")]
    assert len(added) == 1 and "_PAIRED_BED_FIELDS" in added[0]
    # rebuild the B73 core: drop the three paired-branch statements, inline the factored function at its call
    core = _strip_doc(n["_sediment_coupled_core"])
    func = _strip_doc(n["_transport_with_substep_retry"])
    assert ast.unparse(func[0]) == "from maple.core.backend import to_float"
    assert isinstance(func[-1], ast.Return) and ast.unparse(func[-1]) == \
        "return (transport, characteristic, n_sub, rejections)"
    inlined = [_Rename().visit(ast.parse(ast.unparse(s)).body[0]) for s in func[1:-1]]
    rebuilt, dropped = [], []
    for stmt in core:
        text = ast.unparse(stmt)
        if text.startswith(("paired_entry = None", "if not full and control.csr and is_windowed_bed",
                            "if paired_entry is not None:")):
            dropped.append(stmt)
            continue
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call) and \
                ast.unparse(stmt.value.func) == "_transport_with_substep_retry":
            rebuilt.extend(inlined)
            continue
        rebuilt.append(stmt)
    assert len(dropped) == 3
    old_core = _strip_doc(o["_sediment_coupled_core"])
    # the core's former first statement moved INTO the factored function with step 4 (its only use)
    assert ast.unparse(old_core[0]) == ast.unparse(func[0])
    rebuilt = [func[0], *rebuilt]
    assert [ast.dump(s) for s in rebuilt] == [ast.dump(ast.parse(ast.unparse(s)).body[0]) for s in old_core] or \
        [ast.unparse(s) for s in rebuilt] == [ast.unparse(s) for s in old_core]
    # the paired branch: the same `with_water` statement, the shared function, the ORIGINAL demand, the same tail
    branch = dropped[2]
    text = ast.unparse(branch)
    assert "bed0 = with_water(state.bed, context, route.depth_m)" in text
    assert "_transport_with_substep_retry(state, control, velocity" in text
    assert "demand = water_demand_from_transport(transport, zeros)" in text
    # namespace walk BEFORE the value check of the deposit water (root qualification)
    assert text.index("resolve_operand_namespace(bed0, water") < text.index("validate_water_state(water")
    old_tail = [ast.unparse(s) for s in old_core[-4:]]  # phase/reconciliation, reconcile, del inputs, return
    assert [ast.unparse(s) for s in branch.body[-4:]] == old_tail


def test_accounting_field_tables_match_the_maple_view_and_bed_fields():
    pytest.importorskip("maple.water.windowed_transaction")
    from maple.water.windowed_transaction import PICKUP_VIEW_FIELDS, PickupView

    assert tuple(f.name for f in dataclasses.fields(PickupView)) == new._PICKUP_ACCOUNTING_FIELDS == \
        tuple(PICKUP_VIEW_FIELDS)
    assert tuple(f.name for f in dataclasses.fields(BedState)) == new._PAIRED_BED_FIELDS


# =====================================================================================================================
# Entry resolution: only the exact missing module falls back
# =====================================================================================================================
def test_entry_resolution_falls_back_only_for_the_exact_missing_module(monkeypatch):
    real = importlib.import_module

    def raising(exc):
        def imp(name, *a, **k):
            if name == bedmod.PAIRED_TRANSACTION_MODULE:
                raise exc
            return real(name, *a, **k)
        return imp

    monkeypatch.setattr(importlib, "import_module", raising(ModuleNotFoundError("x", name=bedmod.PAIRED_TRANSACTION_MODULE)))
    assert bedmod.paired_transaction_entry() is None
    monkeypatch.setattr(importlib, "import_module", raising(ModuleNotFoundError("dep", name="maple.some_dependency")))
    with pytest.raises(ModuleNotFoundError, match="dep"):
        bedmod.paired_transaction_entry()
    monkeypatch.setattr(importlib, "import_module", raising(ImportError("broken helper")))
    with pytest.raises(ImportError, match="broken helper"):
        bedmod.paired_transaction_entry()
    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **k: types.SimpleNamespace())
    with pytest.raises(AttributeError):
        bedmod.paired_transaction_entry()


def test_older_maple_without_the_module_takes_the_two_call_path_in_a_fresh_process(tmp_path):
    """An actual fresh process bound to frozen M67 (no `windowed_transaction`): the entry is None, the step uses the
    two original calls and equals the B73 module under the same M67."""
    if not (M67 / "source/src/maple").is_dir():
        pytest.skip("frozen M67 unavailable")
    script = tmp_path / "older.py"
    script.write_text(textwrap.dedent(f"""
        import sys, types, json, dataclasses, pathlib, numpy as np
        sys.path[:0] = [{str(TREE / 'tests' / 'cw_b76')!r}, {str(TREE / 'tests' / 'phase5')!r},
                        {str(TREE / 'tests' / 'conservative_water')!r}, {str(TREE / 'tests' / 'cw_b20')!r}]
        import test_cw_b76_paired as t
        import maple_syrup.sediment_bed as bedmod, maple_syrup.sediment_event as new
        assert bedmod.paired_transaction_entry() is None
        import maple, importlib
        assert pathlib.Path(maple.__file__).resolve().parent == pathlib.Path({str(M67 / 'source/src/maple')!r}).resolve()
        old = types.ModuleType(t.ORACLE); old.__file__ = str(t.B73 / 'sediment_event.py'); sys.modules[t.ORACLE] = old
        exec(compile((t.B73 / 'sediment_event.py').read_text(), old.__file__, 'exec'), old.__dict__)
        s, args, ctl = t.case_args('window_csr', np, pathlib.Path({str(tmp_path / 'work')!r}))
        calls = {{'two': 0}}
        real = new.apply_bed_demand
        def spy(*a, **k):
            calls['two'] += 1
            return real(*a, **k)
        new.apply_bed_demand = spy
        got = new._sediment_accounting_step(*args, ctl)
        ref = old._sediment_accounting_step(t.as_types(old, args[0]), *args[1:], t.as_types(old, ctl))
        assert calls['two'] == 2
        assert t.tree(got) == t.tree(ref)
        print('OLDER_MAPLE_OK')
    """))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               MAPLE_SYRUP_EXPECTED_MAPLE_ROOT=str(M67 / "source"),
               PYTHONPATH=os.pathsep.join([str(M67 / "editable"), str(M67 / "source/src"), str(TREE / "src")]))
    proc = subprocess.run([sys.executable, "-B", str(script)], env=env, capture_output=True, text=True, timeout=900,
                          check=False)
    assert proc.returncode == 0 and "OLDER_MAPLE_OK" in proc.stdout, proc.stdout[-3000:] + proc.stderr[-3000:]


# =====================================================================================================================
# Differentials against the frozen B73 module (paired vs original two-call)
# =====================================================================================================================
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scale", [1.0, 0.0, 8.0])
def test_paired_accounting_step_raw_identical_to_b73(old, monkeypatch, backend, scale, tmp_path):
    xp = b20.backend_xp(backend)
    _s, args, ctl = case_args("window_csr", xp, tmp_path, scale)
    got, ref, cn, co = run_both(old, monkeypatch, xp, args, ctl, tmp_path=tmp_path)
    assert cn == {"paired": 1, "two_call": 0} and co == {"paired": 0, "two_call": 2}
    assert tree(got) == tree(ref)
    assert type(got).__name__ == "SedimentAccountingStep" and got.phase_reconciliation is not None


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case,form", [("window_dense", "accounting"), ("dense", "accounting"),
                                       ("window_csr", "full"), ("window_dense", "full")])
def test_fallback_paths_never_call_the_paired_entry_and_equal_b73(old, monkeypatch, backend, case, form, tmp_path):
    xp = b20.backend_xp(backend)
    _s, args, ctl = case_args(case, xp, tmp_path)
    got, ref, cn, co = run_both(old, monkeypatch, xp, args, ctl, form=form, tmp_path=tmp_path)
    assert cn == {"paired": 0, "two_call": 2} and co == {"paired": 0, "two_call": 2}
    assert tree(got) == tree(ref)


def test_upwind_accounting_never_calls_the_paired_entry(old, monkeypatch, tmp_path):
    import test_exchange_history as teh

    bed, ctx = teh.make_bed(teh.chain_elevation(3), depth_m=.001, commit_spec=None)
    control = teh.SedimentEventControl(storm=teh.StormControl(max_dt_s=1, implementation="array", max_steps=10000),
                                       transport_scheme="upwind")
    s = teh.setup(bed, ctx, ksat=1e-5, control=control)
    args = b20.step_args(s, teh.constant_rainfall(0, 10, 100))
    got, ref, cn, co = run_both(old, monkeypatch, np, args, control)
    assert cn == co == {"paired": 0, "two_call": 2} and tree(got) == tree(ref)


@dataclasses.dataclass(frozen=True)
class ExtendedBed(BedState):
    extra: int = 0


class SubBed(BedState):
    pass


@pytest.mark.parametrize("kind", ["subclass", "extended_dataclass"])
def test_subclass_and_extended_beds_keep_the_two_call_path(old, monkeypatch, kind, tmp_path):
    _s, args, ctl = case_args("window_csr", np, tmp_path)
    cls = SubBed if kind == "subclass" else ExtendedBed
    bed = cls(**{f.name: getattr(args[0].bed, f.name) for f in dataclasses.fields(BedState)})
    state = dataclasses.replace(args[0], bed=bed)
    args = (state, *args[1:])
    with monkeypatch.context() as m:
        cn = spy_calls(m, new)
        try:
            got = ("ok", tree(new._sediment_accounting_step(*args, ctl)))
        except Exception as exc:  # noqa: BLE001
            got = ("error", type(exc).__name__, str(exc))
    with monkeypatch.context() as m:
        spy_calls(m, old)
        try:
            ref = ("ok", tree(old._sediment_accounting_step(as_types(old, state), *args[1:], as_types(old, ctl))))
        except Exception as exc:  # noqa: BLE001
            ref = ("error", type(exc).__name__, str(exc))
    assert cn["paired"] == 0 and got == ref


# =====================================================================================================================
# Refusals and retry: same outcome as B73, inputs and deep store unchanged
# =====================================================================================================================
def _refusal(old, monkeypatch, tmp_path, inject):
    outcomes = {}
    for label, module in (("new", new), ("old", old)):
        _s, args, ctl = case_args("window_csr", np, tmp_path / label)
        before, pins = [v.tobytes() for v in b20.leaves(args[0])], b20.store_pins(tmp_path / label)
        with monkeypatch.context() as m:
            inject(m, module)
            state, control = (args[0], ctl) if module is new else (as_types(old, args[0]), as_types(old, ctl))
            with pytest.raises(Exception) as info:
                module._sediment_accounting_step(state, *args[1:], control)
        outcomes[label] = (type(info.value).__name__, str(info.value))
        assert [v.tobytes() for v in b20.leaves(args[0])] == before
        assert b20.store_pins(tmp_path / label) == pins
    assert outcomes["new"] == outcomes["old"], outcomes
    return outcomes["new"]


def test_transport_cap_rejection_identical(old, monkeypatch, tmp_path):
    def inject(m, module):
        def reject(*a, **k):
            raise TransportStepRejected("injected substep rejection")
        m.setattr(module, "sparse_characteristic_step", reject)
    out = _refusal(old, monkeypatch, tmp_path, inject)
    assert out[0] == "TransportStepRejected" and "max_transport_substeps" in out[1]


def _with_nan_T(m, module):
    real = module.sparse_characteristic_step

    def kernel(*a, **k):
        step = real(*a, **k)
        T = step.transport.mobile_after_transfer_kg.copy()
        T.reshape(-1)[0] = np.nan
        return dataclasses.replace(step, transport=dataclasses.replace(step.transport, mobile_after_transfer_kg=T))
    m.setattr(module, "sparse_characteristic_step", kernel)


def test_nan_transport_pool_refused_with_the_same_message(old, monkeypatch, tmp_path):
    out = _refusal(old, monkeypatch, tmp_path, _with_nan_T)
    assert "finite" in out[1] or "nan" in out[1].lower()


def test_pickup_and_deposit_refusals_identical(old, monkeypatch, tmp_path):
    def nan_pickup(m, module):
        real = module.sediment_physics_step

        def physics(*a, **k):
            r = real(*a, **k)
            req = r.requested_pickup_kg.copy()
            req.reshape(-1)[0] = np.nan
            return dataclasses.replace(r, requested_pickup_kg=req)
        m.setattr(module, "sediment_physics_step", physics)
    _refusal(old, monkeypatch, tmp_path / "pickup", nan_pickup)

    def nan_deposit(m, module):
        real = module.water_demand_from_transport

        def demand(*a, **k):
            d = real(*a, **k)
            dep = d.requested_deposition_by_cell_class_kg.copy()
            dep.reshape(-1)[0] = -1.0
            return dataclasses.replace(d, requested_deposition_by_cell_class_kg=dep)
        m.setattr(module, "water_demand_from_transport", demand)
    _refusal(old, monkeypatch, tmp_path / "deposit", nan_deposit)


def test_reconciliation_refusal_identical(old, monkeypatch, tmp_path):
    """The pool MAPLE publishes is tampered beyond the FP bound: reconciliation refuses with the B73 message. B73's
    seam is its second `apply_bed_demand` call; B76's is the paired adapter (which no longer calls it)."""
    from maple.core.types.water import WaterState

    def tamper(bed):
        pool = bed.water.mobile_mass_by_cell_class_kg + 1e-3
        return dataclasses.replace(bed, water=WaterState(bed.water.depth_m, pool))

    def inject(m, module):
        if module is new:
            real = new.apply_bed_pickup_transport_deposit

            def paired(*a, **k):
                view, bed, result = real(*a, **k)
                return view, tamper(bed), result
            m.setattr(new, "apply_bed_pickup_transport_deposit", paired)
        else:
            real, n = module.apply_bed_demand, [0]

            def apply(*a, **k):
                n[0] += 1
                bed, result = real(*a, **k)
                return (tamper(bed) if n[0] == 2 else bed), result
            m.setattr(module, "apply_bed_demand", apply)
    out = _refusal(old, monkeypatch, tmp_path, inject)
    assert "beyond the declared FP64 bound" in out[1]


def test_substep_retry_inside_the_callback_identical(old, monkeypatch, tmp_path):
    """The first kernel call rejects (recoverable): the shared loop doubles the substeps and retries, as in B73."""
    results = {}
    _s, args, ctl = case_args("window_csr", np, tmp_path)  # ONE case (same deep store) for both implementations
    for label, module in (("new", new), ("old", old)):
        with monkeypatch.context() as m:
            real, n = module.sparse_characteristic_step, [0]

            def kernel(*a, real=real, n=n, **k):
                n[0] += 1
                if n[0] == 1:
                    raise TransportStepRejected("first attempt rejected")
                return real(*a, **k)
            m.setattr(module, "sparse_characteristic_step", kernel)
            state, control = (args[0], ctl) if module is new else (as_types(old, args[0]), as_types(old, ctl))
            r = module._sediment_accounting_step(state, *args[1:], control)
        assert r.n_transport_rejections == 1
        results[label] = tree(r)
    assert results["new"] == results["old"]


# =====================================================================================================================
# Lifetimes and the MAPLE adoption/donation actually engaged
# =====================================================================================================================
def test_liveness_and_owned_deposit_begin_and_ledger_donation(monkeypatch, tmp_path):
    import maple.coupling.sediment_ledger.accumulate as acc
    import maple.water.windowed_transaction as wt

    _s, args, ctl = case_args("window_csr", np, tmp_path)
    refs, seen = {}, {}
    with monkeypatch.context() as m:
        real_phys = new.sediment_physics_step

        def physics(*a, **k):
            r = real_phys(*a, **k)
            refs["physics"], refs["requested_pickup"] = weakref.ref(r), weakref.ref(r.requested_pickup_kg)
            return r
        m.setattr(new, "sediment_physics_step", physics)
        real_p, first = wt._apply_windowed_water_process_demand, [True]

        def phase(*a, **k):  # MAPLE's own phase-P call inside the paired entry: its post-pickup water pool
            r = real_p(*a, **k)
            if first[0]:
                first[0] = False
                refs["postpickup_pool"] = weakref.ref(r.new_water.mobile_mass_by_cell_class_kg)
            return r
        m.setattr(wt, "_apply_windowed_water_process_demand", phase)
        real_k = new.sparse_characteristic_step

        def kernel(*a, **k):
            gc.collect()
            seen["at_kernel"] = {n: r() is not None for n, r in refs.items()}
            refs["rate"], refs["mask"] = weakref.ref(a[5]), weakref.ref(a[6])
            step = real_k(*a, **k)
            refs["characteristic"], refs["transport"] = weakref.ref(step), weakref.ref(step.transport)
            refs["face_x"] = weakref.ref(step.transport.x_face_gross_kg)
            return step
        m.setattr(new, "sparse_characteristic_step", kernel)
        real_b = wt._begin_owned_windowed_transaction

        def begin(*a, **k):
            gc.collect()
            seen["at_owned_begin"] = {n: r() is not None for n, r in refs.items()}
            return real_b(*a, **k)
        m.setattr(wt, "_begin_owned_windowed_transaction", begin)
        real_o = acc._accumulate_water_transfer_into_owned

        def owned(*a, **k):
            out = real_o(*a, **k)
            seen.setdefault("blockers", []).append(out[1])
            return out
        m.setattr(acc, "_accumulate_water_transfer_into_owned", owned)
        new._sediment_accounting_step(*args, ctl)
    # dead before the transport callback: the requested pickup, the physics result AND the post-pickup MAPLE pool
    assert seen["at_kernel"] == {"physics": False, "requested_pickup": False, "postpickup_pool": False}
    at = seen["at_owned_begin"]
    assert not at["characteristic"] and not at["transport"] and not at["rate"] and not at["mask"]
    assert at["face_x"]  # the faces are held by the deposit demand through MAPLE's deposit validation
    assert seen["blockers"][0] is None  # the first deposit accumulation donated into the private post-pickup ledger


def _numpy_live():
    snap = tracemalloc.take_snapshot().filter_traces([tracemalloc.DomainFilter(True, np.lib.tracemalloc_domain)])
    return sum(t.size for t in snap.traces)


def test_cpu_live_bytes_equal_at_transport_entry_and_return(old, monkeypatch, tmp_path):
    """NumPy-domain live bytes at the transport kernel entry (inside the callback for B76) and after return (results
    kept): B73 - B76 == 0 at both marks (the paired path keeps exactly B71's post-pickup set through transport)."""
    marks = {}
    for label, module in (("new", new), ("old", old)):
        _s, args, ctl = case_args("window_csr", np, tmp_path / label)
        state, control = (args[0], ctl) if module is new else (as_types(old, args[0]), as_types(old, ctl))
        module._sediment_accounting_step(state, *args[1:], control)  # warm-up
        gc.collect()
        mk = {}
        with monkeypatch.context() as m:
            real_k = module.sparse_characteristic_step

            def kernel(*a, real_k=real_k, mk=mk, **k):
                gc.collect()
                mk.setdefault("kernel", _numpy_live() - mk["base"])
                return real_k(*a, **k)
            m.setattr(module, "sparse_characteristic_step", kernel)
            tracemalloc.start()
            try:
                mk["base"] = _numpy_live()
                kept = module._sediment_accounting_step(state, *args[1:], control)
                gc.collect()
                mk["return"] = _numpy_live() - mk["base"]
            finally:
                tracemalloc.stop()
        marks[label] = (mk["kernel"], mk["return"], kept)
    assert marks["old"][0] - marks["new"][0] == 0, marks
    assert marks["old"][1] - marks["new"][1] == 0, marks


# =====================================================================================================================
# Complete event, wet checkpoint and dry reset: byte-identical to the B73 driver (same MAPLE)
# =====================================================================================================================
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", ["window_csr", "window_dense", "dense"])
def test_complete_driver_byte_identical_to_the_b73_tree(backend, case, tmp_path):
    b20.backend_xp(backend)
    a_arrays, a_counts = b20.run_driver(B73_TREE, case, backend, tmp_path, "b73")
    b_arrays, b_counts = b20.run_driver(TREE, case, backend, tmp_path, "b76")
    assert a_counts.pop("maple_syrup") == str(B73) and b_counts.pop("maple_syrup") == str(PKG)
    assert a_counts == b_counts
    assert sorted(a_arrays) == sorted(b_arrays)
    for key in a_arrays:
        assert a_arrays[key].dtype == b_arrays[key].dtype and a_arrays[key].shape == b_arrays[key].shape, key
        assert a_arrays[key].tobytes() == b_arrays[key].tobytes(), key


# =====================================================================================================================
# Device (root)
# =====================================================================================================================
def test_device_mixed_namespace_nan_pool_refusal_priority(old, monkeypatch, tmp_path):
    """A HOST NumPy T containing NaN on a CuPy state: the namespace refusal must come first, exactly as in B73."""
    xp = b20.backend_xp("cupy")

    def host_nan_T(m, module):
        real = module.sparse_characteristic_step

        def kernel(*a, **k):
            step = real(*a, **k)
            T = np.asarray(host(step.transport.mobile_after_transfer_kg)).copy()
            T.reshape(-1)[0] = np.nan
            return dataclasses.replace(step, transport=dataclasses.replace(step.transport, mobile_after_transfer_kg=T))
        m.setattr(module, "sparse_characteristic_step", kernel)
    outcomes = {}
    for label, module in (("new", new), ("old", old)):
        _s, args, ctl = case_args("window_csr", xp, tmp_path / label)
        with monkeypatch.context() as m:
            host_nan_T(m, module)
            state, control = (args[0], ctl) if module is new else (as_types(old, args[0]), as_types(old, ctl))
            with pytest.raises(Exception) as info:
                module._sediment_accounting_step(state, *args[1:], control)
        outcomes[label] = (type(info.value).__name__, str(info.value))
    assert outcomes["new"] == outcomes["old"], outcomes
    assert "namespace" in outcomes["new"][1].lower() or "Mixed" in outcomes["new"][0]


def test_device_memoryhook_transport_entry_and_return_zero(old, monkeypatch, tmp_path):
    """CuPy MemoryHook live bytes, B73 - B76, at the transport kernel entry and after return: both 0 (the deposit-phase
    savings are MAPLE B75's, proven there; here the owned begin / donation spies prove they are engaged).

    CW-B76 correction 1: ONE original case (one deep store) is built once and its SAME immutable state/args/control are
    shared by both implementations (old via the typed wrapper, as in `run_both`), so the complete raw result records
    compare without any metadata normalization; the actual input arrays and deep-store pins are asserted unchanged."""
    cp = pytest.importorskip("cupy")
    xp = b20.backend_xp("cupy")

    class Tracker:
        def __init__(self):
            self.live, self.current = {}, 0

    def hook_for(tr):
        class Hook(cp.cuda.MemoryHook):
            name = "CWB76Live"

            def malloc_postprocess(self, **kw):
                if kw.get("mem_ptr"):
                    key = (int(kw["device_id"]), int(kw["pmem_id"]))
                    tr.current += int(kw["mem_size"]) - tr.live.get(key, 0)
                    tr.live[key] = int(kw["mem_size"])

            def free_postprocess(self, **kw):
                tr.current -= tr.live.pop((int(kw["device_id"]), int(kw["pmem_id"])), 0)
        return Hook()
    marks = {}
    _s, args, ctl = case_args("window_csr", xp, tmp_path)  # ONE case (same deep store) for both implementations
    before, pins = [v.tobytes() for v in b20.leaves(args[0])], b20.store_pins(tmp_path)
    for label, module in (("new", new), ("old", old)):
        state, control = (args[0], ctl) if module is new else (as_types(old, args[0]), as_types(old, ctl))
        module._sediment_accounting_step(state, *args[1:], control)
        gc.collect()
        tr, mk = Tracker(), {}
        with monkeypatch.context() as m:
            real_k = module.sparse_characteristic_step

            def kernel(*a, real_k=real_k, mk=mk, tr=tr, **k):
                cp.cuda.Device().synchronize()
                mk.setdefault("kernel", tr.current)
                return real_k(*a, **k)
            m.setattr(module, "sparse_characteristic_step", kernel)
            with hook_for(tr):
                kept = module._sediment_accounting_step(state, *args[1:], control)
                gc.collect()
                cp.cuda.Device().synchronize()
                mk["return"] = tr.current
        marks[label] = (mk["kernel"], mk["return"], kept)
        assert [v.tobytes() for v in b20.leaves(args[0])] == before, label  # actual input arrays unchanged
        assert b20.store_pins(tmp_path) == pins, label  # deep store unchanged (same pure paths)
    assert marks["old"][0] - marks["new"][0] == 0 and marks["old"][1] - marks["new"][1] == 0, marks
    assert tree(marks["old"][2]) == tree(marks["new"][2])
    json.dumps({"kernel": [marks["old"][0], marks["new"][0]], "return": [marks["old"][1], marks["new"][1]]})
