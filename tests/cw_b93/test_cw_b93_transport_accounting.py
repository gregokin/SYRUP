"""CW-B93: private CUDA CSR accounting retention for the paired MAPLE transaction (measured H4 refusal at the first face
allocation of `cuda_sparse_step`).

Oracle: the frozen B76 runtime (hash-pinned `characteristic_cuda_csr.py`, `sediment_event.py`, `sediment_transport.py`,
`characteristic_sparse.py`), its CUDA module executed under a private name against this tree's otherwise identical
modules. CPU tests: scope (only the four runtime files and this suite differ from B76), unchanged public signatures,
kernels and output-flag packet, the private record's exact field set, private/public type separation, the retention
switch reached only from the paired callback. DEVICE tests (GPU0; skipped without a device, a skip claims nothing):
B76 public == new public raw (complete record) and B76 public == new PRIVATE raw for every field the private record
carries, over dry / signed-zero / wet / stored-bin / terminal / masked cases, 1-3 substeps, chained steps, caller
immutability; identical refusals; injected output corruption with identical first violation; the paired production
path takes the private step on CuPy (and the public step elsewhere); whole-step MemoryHook peaks; and ONE fixed
private-pool cap under which B76 refuses inside the step while the private step returns entirely.
"""
from __future__ import annotations

import ast
import dataclasses
import gc
import hashlib
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maple")

import test_cuda_csr_lifetimes as h  # CW-B52 harness (this tree): networks, cases, raw records, live bytes

from maple_syrup import characteristic_cuda_csr as cuda_csr
from maple_syrup import characteristic_sparse as sparse
from maple_syrup import sediment_event as event
from maple_syrup import sediment_transport as transport_mod
from maple_syrup.characteristic_sparse import (
    SparseCharacteristicStep,
    _SparseAccountingStep,
    _TransportAccountingStep,
)
from maple_syrup.sediment_transport import TransportError, TransportStep

TREE = Path(__file__).resolve().parents[2]
PKG = TREE / "src" / "maple_syrup"
B76 = Path("/home/okin/SYRUP/outputs/conservative_water/cw_b76_runtime_frozen")
B76_PKG = B76 / "src" / "maple_syrup"
B76_SHA = {"characteristic_cuda_csr.py": "d13f4bf5d2642f65f823514b2ea09ea0ddb1c0275aedafb57596adfbb372d5b7",
           "characteristic_sparse.py": "31d733145da8fde9ded73295ba727ee2c1d4ad3b0daec9c55ba525bd81c3de0b",
           "sediment_event.py": "2eabe7ee25f03b7dd214db546104c9426c354c006b589018f3d0a356a4c7dd89",
           "sediment_transport.py": "f123c3cc4cf22a2807ea57366d1117a9eb2429c3a52c1a2301e8f898f3e4fca0"}
CHANGED = sorted(B76_SHA)
PRIVATE_FIELDS = ("dt_s", "n_substeps", "mobile_after_transfer_kg", "deposition_request_kg", "export_request_kg",
                  "x_face_gross_kg", "y_face_gross_kg", "x_face_net_kg", "y_face_net_kg",
                  "deposition_request_by_class_kg", "export_request_by_class_kg", "budget_residual_by_class_kg",
                  "budget_tolerance_by_class_kg", "max_courant", "max_decay_exponent", "max_cell_balance_residual_kg")
needs_gpu = h.needs_gpu
cp = h.cp
POOL_UNIT = 512


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def b76():
    path = B76_PKG / "characteristic_cuda_csr.py"
    if not path.is_file():
        pytest.skip("frozen B76 oracle unavailable; no before/after claim")
    assert _sha(path) == B76_SHA["characteristic_cuda_csr.py"], "B76 oracle differs from its pin"
    module = types.ModuleType("_cw_b93_b76_characteristic_cuda_csr")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)  # noqa: S102 -- the hash-pinned B76 text
    return module


# =====================================================================================================================
# CPU: scope, public API, kernels, packet, private types, switch
# =====================================================================================================================
def _tree(root):
    return {p.relative_to(root).as_posix(): _sha(p) for p in root.rglob("*") if p.is_file()
            and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts and p.suffix not in (".pyc", ".pyo")}


def test_scope_only_four_runtime_files_and_this_suite_differ_from_b76():
    mine, base = _tree(TREE), _tree(B76)
    assert sorted(set(mine) - set(base)) == ["tests/cw_b93/conftest.py",
                                             "tests/cw_b93/test_cw_b93_transport_accounting.py"]
    assert not set(base) - set(mine)
    assert sorted(k for k in set(mine) & set(base) if mine[k] != base[k]) == [f"src/maple_syrup/{f}" for f in CHANGED]
    assert all(base[f"src/maple_syrup/{f}"] == sha for f, sha in B76_SHA.items())


def _defs(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


@pytest.mark.parametrize("module,names", [
    ("characteristic_cuda_csr.py", ["cuda_sparse_step"]),
    ("characteristic_sparse.py", ["sparse_characteristic_step", "SparseCharacteristicStep"]),
    ("sediment_transport.py", ["water_demand_from_transport", "TransportStep"]),
    ("sediment_event.py", ["TransportAccounting", "SedimentAccountingStep", "SedimentCoupledStep",
                           "sediment_coupled_step", "_sediment_accounting_step"])])
def test_public_signatures_and_result_types_unchanged(module, names):
    old, new = _defs(B76_PKG / module), _defs(PKG / module)
    for name in names:
        o, n = old[name], new[name]
        if isinstance(o, ast.FunctionDef):
            assert ast.dump(o.args) == ast.dump(n.args) and ast.dump(o.returns or ast.Constant(None)) == \
                ast.dump(n.returns or ast.Constant(None)), name
            assert ast.get_docstring(o) == ast.get_docstring(n), name
        else:
            assert ast.dump(o) == ast.dump(n), name  # result dataclasses byte-for-byte in the AST


def test_kernels_and_module_constants_unchanged(b76):
    assert cuda_csr.COMPILE_OPTIONS == b76.COMPILE_OPTIONS and cuda_csr.KERNEL_NAMES == b76.KERNEL_NAMES
    for nb in (1, 8, 32, 100, 128):
        assert cuda_csr.kernel_source(nb) == b76.kernel_source(nb)
        assert cuda_csr.estimate_local_scratch_bytes(nb) == b76.estimate_local_scratch_bytes(nb)
    unchanged = set(_defs(B76_PKG / "characteristic_cuda_csr.py")) - {"cuda_sparse_step"}
    old, new = _defs(B76_PKG / "characteristic_cuda_csr.py"), _defs(PKG / "characteristic_cuda_csr.py")
    assert sorted(k for k in unchanged if ast.dump(old[k]) != ast.dump(new[k])) == []
    assert set(new) - set(old) == {"_cuda_sparse_accounting_step", "_cuda_sparse_core"}


def _check_call(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("require", "forbid") \
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "checks":
        return node.func.attr, ast.unparse(node.args[1])
    return None


def _packet(source, func_name):
    """(kind, message) of the OUTPUT flag packet, loops expanded over their literal names (B54 helper, parameterised
    by function name; `if` blocks followed in source order)."""
    func = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == func_name)
    marker = next(i for i, t in enumerate(source.splitlines(), 1) if "# --- output validation" in t)
    appended = sorted((n.lineno, ("forbid", ast.unparse(n.args[0].elts[1]))) for n in ast.walk(func)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "append"
                      and isinstance(n.func.value, ast.Name) and n.func.value.id == "phase_flags")
    out = []

    def visit(statements):
        for st in statements:
            if isinstance(st, ast.FunctionDef):
                continue
            if st.lineno <= marker and not isinstance(st, (ast.If, ast.For)):
                continue
            if isinstance(st, ast.For):
                calls = [c for n in ast.walk(st) if (c := _check_call(n))]
                if isinstance(st.iter, ast.Name) and st.iter.id == "phase_flags":
                    out.extend(m for _, m in appended)
                elif st.lineno > marker and calls:
                    names = [elt.elts[0].value for elt in st.iter.elts]
                    (kind, message), = calls
                    out.extend((kind, message.replace("{name}", nm)) for nm in names)
            elif isinstance(st, ast.If):
                visit(st.body)
                visit(st.orelse)
            elif st.lineno > marker:
                out.extend(c for n in ast.walk(st) if (c := _check_call(n)))
    visit(func.body)
    return out


def test_output_flag_packet_identical_to_b76():
    old = _packet((B76_PKG / "characteristic_cuda_csr.py").read_text(), "cuda_sparse_step")
    new = _packet((PKG / "characteristic_cuda_csr.py").read_text(), "_cuda_sparse_core")
    assert len(old) == 39 and new == old


def test_private_records_have_exactly_the_production_fields_and_are_not_public_types():
    assert tuple(f.name for f in dataclasses.fields(_TransportAccountingStep)) == PRIVATE_FIELDS
    # everything the paired production path reads: the loop's accounting record, MAPLE's demand, reconciliation
    needed = set(event._TRANSPORT_ACCOUNTING_FIELDS) | {
        "mobile_after_transfer_kg", "deposition_request_kg", "export_request_kg", "x_face_gross_kg", "y_face_gross_kg",
        "x_face_net_kg", "y_face_net_kg"}
    assert needed <= set(PRIVATE_FIELDS)
    public = {f.name for f in dataclasses.fields(TransportStep)}
    assert set(PRIVATE_FIELDS) <= public  # same names as the public fields they carry (same objects on device)
    assert {f.name: f.default for f in dataclasses.fields(_TransportAccountingStep)} == \
        {n: dataclasses.MISSING for n in PRIVATE_FIELDS}  # no placeholder defaults
    assert [f.name for f in dataclasses.fields(_SparseAccountingStep)] == [
        "transport", "phase_after", "mobile_remaining_kg", "n_bins", "implementation", "nnz_after", "peak_nnz",
        "device_report"]
    assert all(f.default is dataclasses.MISSING for f in dataclasses.fields(_SparseAccountingStep))
    assert not issubclass(_TransportAccountingStep, TransportStep)
    assert not issubclass(_SparseAccountingStep, SparseCharacteristicStep)
    assert "_sparse_characteristic_accounting_step" not in getattr(sparse, "__all__", ())


def _fake_private():
    a = np.zeros((2, 2, 1))
    return _TransportAccountingStep(*(1.0, 1, a, a, a, np.zeros((2, 3, 1)), np.zeros((3, 2, 1)),
                                      np.zeros((2, 3, 1)), np.zeros((3, 2, 1)), *([np.zeros(1)] * 4), 0.0, 0.0, 0.0))


def test_public_demand_refuses_the_private_record_and_private_demand_refuses_public_types():
    private = _fake_private()
    with pytest.raises(TransportError, match="must be a TransportStep"):
        transport_mod.water_demand_from_transport(private, np.zeros((2, 2, 1)))
    with pytest.raises(TransportError, match="private CUDA CSR accounting transport record"):
        transport_mod._water_demand_from_accounting_transport(object(), np.zeros((2, 2, 1)))
    demand = transport_mod._water_demand_from_accounting_transport(private, np.zeros((2, 2, 1)))
    assert demand.requested_deposition_by_cell_class_kg is private.deposition_request_kg
    assert demand.boundary_export_by_cell_class_kg is private.export_request_kg
    assert demand.face_flux.x_face_crossing_mass_kg is private.x_face_gross_kg
    assert demand.face_flux.y_face_net_crossing_mass_kg is private.y_face_net_kg
    for bad, message in ((np.full((2, 2, 1), np.nan), "finite"), (np.full((2, 2, 1), -1.0), ">= 0"),
                         (np.zeros((2, 2, 2)), "shape")):
        with pytest.raises(TransportError, match=message):
            transport_mod._water_demand_from_accounting_transport(private, bad)


def test_reconcile_inputs_and_accounting_record_accept_the_private_step():
    private = _fake_private()
    step = _SparseAccountingStep(transport=private, phase_after="phase", mobile_remaining_kg=np.ones((2, 2, 1)),
                                 n_bins=8, implementation="cuda", nnz_after=0, peak_nnz=0, device_report={})
    rec = event._PhaseReconcileInputs.of(step)
    assert rec.sparse is True and rec.mobile_after_transfer_kg is private.mobile_after_transfer_kg
    assert rec.mobile_remaining_kg is step.mobile_remaining_kg and rec.phase_after == "phase"
    acc = event.TransportAccounting.of(private)
    assert all(getattr(acc, n) is getattr(private, n) for n in event._TRANSPORT_ACCOUNTING_FIELDS)


def test_retention_switch_only_from_the_paired_callback_and_validated():
    src = (PKG / "sediment_event.py").read_text()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and ast.unparse(n.func) == "_transport_with_substep_retry"]
    assert len(calls) == 2
    kw = [{k.arg: ast.unparse(k.value) for k in c.keywords} for c in calls]
    assert sorted(map(str, kw)) == sorted(map(str, [{"retain": "'accounting'"}, {}]))
    paired = next(c for c, k in zip(calls, kw) if k)
    between = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "between")
    assert between.lineno <= paired.lineno <= between.end_lineno
    assert src.count("_sparse_characteristic_accounting_step(") == 1
    with pytest.raises(event.SedimentEventError, match="retain must be one of"):
        event._transport_with_substep_retry(None, None, None, None, None, None, 1.0, np, None, retain="bogus")
    with pytest.raises(TransportError, match="retain must be one of"):
        cuda_csr._cuda_sparse_core(*([None] * 8), n_substeps=1, courant_max=0.5, max_nnz=None,
                                   max_launched_threads=1, block_threads=1, unbounded_nnz_bytes=1, retain="bogus")


def test_private_dispatch_is_cuda_only(monkeypatch):
    """Numba CSR under the accounting retention keeps the PUBLIC step (spy on both entries)."""
    seen = []
    monkeypatch.setattr(event, "sparse_characteristic_step", lambda *a, **k: seen.append(("public", k)) or
                        types.SimpleNamespace(transport="T"))
    monkeypatch.setattr(event, "_sparse_characteristic_accounting_step", lambda *a, **k: seen.append(("private", k)) or
                        types.SimpleNamespace(transport="P"))
    state = types.SimpleNamespace(network=types.SimpleNamespace(dx_m=1.0), phase="phase",
                                  bed=types.SimpleNamespace(water=types.SimpleNamespace(mobile_mass_by_cell_class_kg="M")))
    for impl, expect in (("numba", "public"), ("cuda", "private")):
        control = types.SimpleNamespace(sediment_courant_max=0.5, max_transport_substeps=8, csr=True,
                                        characteristic=True, phase_max_nnz=7,
                                        resolved_transport_implementation=lambda impl=impl: impl)
        seen.clear()
        out = event._transport_with_substep_retry(state, control, np.zeros(1), "r", "s", "p", 1.0, np, None,
                                                  retain="accounting")
        assert [s[0] for s in seen] == [expect] and out[0] == ("T" if expect == "public" else "P")
        assert seen[0][1]["max_nnz"] == 7 and seen[0][1]["n_substeps"] == 1
        seen.clear()
        event._transport_with_substep_retry(state, control, np.zeros(1), "r", "s", "p", 1.0, np, None)
        assert [s[0] for s in seen] == ["public"]  # the default (two-call / full paths) never takes the private step


# =====================================================================================================================
# DEVICE
# =====================================================================================================================
def _private_vs_public(priv, pub):
    """Every field the private step carries is raw-identical to the public step's field of the same name."""
    assert type(priv) is _SparseAccountingStep and type(priv.transport) is _TransportAccountingStep
    for name in PRIVATE_FIELDS:
        assert h.value_rec(getattr(priv.transport, name)) == h.value_rec(getattr(pub.transport, name)), name
    for name in ("phase_after", "mobile_remaining_kg", "n_bins", "implementation", "nnz_after", "peak_nnz",
                 "device_report"):
        assert h.value_rec(getattr(priv, name)) == h.value_rec(getattr(pub, name)), name


def _outcome(fn, *args, **kw):
    try:
        return ("ok", fn(*args, **kw))
    except (TransportError, ValueError) as exc:
        extra = tuple(getattr(exc, k, None) for k in ("required_nnz", "max_nnz", "substep", "n_substeps"))
        return ("refused", type(exc).__name__, str(exc), extra)


@needs_gpu
@pytest.mark.parametrize("case", h.CASES)
@pytest.mark.parametrize("n_substeps", [1, 2, 3])
@pytest.mark.parametrize("nb,nc", [(8, 1), (32, 6)])
def test_device_public_and_private_raw_against_b76_chained(b76, case, n_substeps, nb, nc):
    net, (mobile, pickup, v, rate, settle), phase = h.build_case(case, nc, nb)
    kw = {"n_substeps": n_substeps, "max_nnz": 1 << 20}
    state = (mobile, phase)
    for k in range(2):  # chained: the second step continues from B76's own published state, with half the pickup
        m, ph = state
        p = pickup if k == 0 else pickup * 0.5
        before = h.caller_rec(m, p, v, rate, settle, phase=ph)
        old = b76.cuda_sparse_step(net, m, ph, p, v, rate, settle, 1.0, **kw)
        new = cuda_csr.cuda_sparse_step(net, m, ph, p, v, rate, settle, 1.0, **kw)
        priv = sparse._sparse_characteristic_accounting_step(net, m, ph, p, v, rate, settle, 1.0, **kw)
        assert h.value_rec(new) == h.value_rec(old)
        _private_vs_public(priv, old)
        assert h.caller_rec(m, p, v, rate, settle, phase=ph) == before  # caller bytes untouched by all three
        state = (old.mobile_remaining_kg, old.phase_after)
    if case == "dry_empty":
        assert old.nnz_after == 0


@needs_gpu
def test_device_signed_zero_outputs_keep_their_sign_bits(b76):
    net, (mobile, pickup, v, rate, settle), phase = h.build_case("signed_zero", 6, 32)
    old = b76.cuda_sparse_step(net, mobile, phase, pickup, v, rate, settle, 1.0)
    priv = cuda_csr._cuda_sparse_accounting_step(net, mobile, phase, pickup, v, rate, settle, 1.0)
    for name in ("mobile_after_transfer_kg", "deposition_request_kg", "export_request_kg", "x_face_net_kg"):
        o, n = cp.asnumpy(getattr(old.transport, name)), cp.asnumpy(getattr(priv.transport, name))
        assert np.array_equal(np.signbit(o), np.signbit(n)) and o.tobytes() == n.tobytes(), name


@needs_gpu
def test_device_identical_refusals_and_ordering(b76):
    nc, nb = 2, 8
    net, (mobile, pickup, v, rate, settle), phase = h.build_case("wet_fresh", nc, nb)
    first = b76.cuda_sparse_step(net, mobile, phase, pickup, v, rate, settle, 1.0)
    base = {"mobile": first.mobile_remaining_kg, "phase": first.phase_after, "pickup": pickup, "v": v, "rate": rate,
            "settle": settle, "dt": 1.0}
    nan = cp.where(cp.arange(nc) == 0, cp.nan, pickup)
    fast = v * 0.0 + 0.3
    cases = {"negative pickup": {"pickup": -pickup - 1.0}, "nonfinite rate": {"rate": rate * cp.inf},
             "courant": {"v": fast}, "nan pickup AND courant": {"pickup": nan, "v": fast},
             "zero pool with stored bins": {"mobile": base["mobile"] * 0.0}, "dt": {"dt": 0.0},
             "substeps": {"n_substeps": 0}, "capacity": {"max_nnz": 1},
             "capacity of 2": {"n_substeps": 2, "max_nnz": 1}}
    for label, change in cases.items():
        a, kw = dict(base), {}
        for key, value in change.items():
            (a if key in a else kw)[key] = value
        args = (net, a["mobile"], a["phase"], a["pickup"], a["v"], a["rate"], a["settle"], a["dt"])
        old = _outcome(b76.cuda_sparse_step, *args, **kw)
        priv = _outcome(sparse._sparse_characteristic_accounting_step, *args, **kw)
        assert old[0] == "refused" and priv == old, (label, old, priv)


def _corrupt_count(module, monkeypatch, part, value):
    """Wrap the count kernel: after its launch, write `value` into one entry of count output `part` (d_decay,
    d_settled, d_export, d_out, d_in, d_arr_decay, d_arr_settled = the last seven launch arguments)."""
    real = module._function

    def wrapped(name, nb):
        kernel = real(name, nb)
        if name != "ms_csr_count":
            return kernel

        def launch(grid, block, args):
            out = kernel(grid, block, args)
            target = args[-7 + part]
            target[0, 0] = value
            return out
        return launch
    monkeypatch.setattr(module, "_function", wrapped)


@needs_gpu
@pytest.mark.parametrize("part,value,expected", [
    (4, float("nan"), "non-finite internal_transfer_in"),
    (3, -1e-3, "negative internal_transfer_out"),
    (0, float("inf"), None),  # decay deposition also enters deposition and T: the first violation is whichever
    (1, -2.0, None)])         # earlier packet entry fires; it must simply be the SAME in all three
def test_device_injected_output_corruption_same_first_violation(b76, monkeypatch, part, value, expected):
    """A corrupted internal-accounting grid must be refused with the SAME first violation on the private path, whose
    flags for these grids are evaluated early (before their release) but enqueued at the original positions."""
    net, (mobile, pickup, v, rate, settle), phase = h.build_case("wet_fresh", 2, 8)
    args = (net, mobile, phase, pickup, v, rate, settle, 1.0)
    out = {}
    for label, module, fn in (("b76", b76, b76.cuda_sparse_step), ("public", cuda_csr, cuda_csr.cuda_sparse_step),
                              ("private", cuda_csr, cuda_csr._cuda_sparse_accounting_step)):
        with monkeypatch.context() as m:
            _corrupt_count(module, m, part, value)
            out[label] = _outcome(fn, *args)
    assert out["b76"][0] == "refused" and (expected is None or expected in out["b76"][2]), out["b76"]
    assert out["public"] == out["b76"] and out["private"] == out["b76"], out


@needs_gpu
def test_device_paired_production_path_takes_the_private_step(monkeypatch, tmp_path):
    """The paired accounting step on a CuPy windowed CSR bed calls the private entry exactly once (and never the public
    step); the two-call full form keeps the public step. Same case, same MAPLE."""
    import test_accounting_step as b20
    import test_cw_b76_paired as b76p

    xp = b20.backend_xp("cupy")
    counts = {"public": 0, "private": 0}
    real_pub, real_priv = event.sparse_characteristic_step, event._sparse_characteristic_accounting_step

    def pub(*a, **k):
        counts["public"] += 1
        return real_pub(*a, **k)

    def priv(*a, **k):
        counts["private"] += 1
        return real_priv(*a, **k)
    monkeypatch.setattr(event, "sparse_characteristic_step", pub)
    monkeypatch.setattr(event, "_sparse_characteristic_accounting_step", priv)
    _s, args, ctl = b76p.case_args("window_csr", xp, tmp_path / "a")
    assert ctl.resolved_transport_implementation() == "cuda"
    got = event._sediment_accounting_step(*args, ctl)
    assert counts == {"public": 0, "private": 1} and got.phase_reconciliation is not None
    counts.update(public=0, private=0)
    _s, args, ctl = b76p.case_args("window_csr", xp, tmp_path / "b")
    event.sediment_coupled_step(*args, ctl)
    assert counts == {"public": 1, "private": 0}


@needs_gpu
def test_device_mixed_namespace_nan_pool_refusal_priority_private(monkeypatch, tmp_path):
    """B76 `test_device_mixed_namespace_nan_pool_refusal_priority` re-targeted at the private entry (the B76 test
    intercepts `sparse_characteristic_step` by name, which the paired CuPy CSR path no longer calls): a HOST NumPy T
    containing NaN must be refused by the namespace check first, identically whether the paired callback received the
    private record or (substituted) the public CUDA step for the same call."""
    import test_accounting_step as b20
    import test_cw_b76_paired as b76p
    from b20_cases import host

    xp = b20.backend_xp("cupy")
    outcomes = {}
    real = event._sparse_characteristic_accounting_step

    def public_cuda(*a, **k):
        return sparse.sparse_characteristic_step(*a, implementation="cuda", **k)
    for label, source in (("private", real), ("public_in_paired", public_cuda)):
        _s, args, ctl = b76p.case_args("window_csr", xp, tmp_path / label)
        with monkeypatch.context() as m:
            def kernel(*a, source=source, **k):
                step = source(*a, **k)
                T = np.asarray(host(step.transport.mobile_after_transfer_kg)).copy()
                T.reshape(-1)[0] = np.nan
                return dataclasses.replace(step, transport=dataclasses.replace(step.transport,
                                                                               mobile_after_transfer_kg=T))
            m.setattr(event, "_sparse_characteristic_accounting_step", kernel)
            with pytest.raises(Exception) as info:
                event._sediment_accounting_step(*args, ctl)
        outcomes[label] = (type(info.value).__name__, str(info.value))
    assert "namespace" in outcomes["private"][1].lower() or "Mixed" in outcomes["private"][0], outcomes
    assert outcomes["private"] == outcomes["public_in_paired"], outcomes


def _peak(fn, args, kw):
    tracker = h._LiveBytes()
    state = {"peak": 0}

    class Peak(cp.cuda.MemoryHook):
        name = "CWB93Peak"

        def malloc_postprocess(self, **k):
            if k.get("mem_ptr"):
                key = (int(k["device_id"]), int(k["pmem_id"]))
                tracker.current += int(k["mem_size"]) - tracker.live.get(key, 0)
                tracker.live[key] = int(k["mem_size"])
                state["peak"] = max(state["peak"], tracker.current)

        def free_postprocess(self, **k):
            tracker.current -= tracker.live.pop((int(k["device_id"]), int(k["pmem_id"])), 0)
    fn(*args, **kw)
    cp.cuda.Device().synchronize()
    gc.collect()
    with Peak():
        out = fn(*args, **kw)
        cp.cuda.Device().synchronize()
        retained = tracker.current
    del out
    return state["peak"], retained


def _sized_case(ny, nx, nc, nb, seed=3):
    rng = np.random.default_rng(seed)
    net = h.device_network(h.random_full(rng, ny, nx))
    shape = (ny, nx, nc)
    pickup = cp.asarray(rng.uniform(0.0, 1.0, shape) * (rng.uniform(size=shape) < 0.6))
    v = cp.asarray(rng.uniform(0.0, 0.2, shape))
    rate = cp.asarray(rng.uniform(0.0, 4.0, shape))
    settle = cp.asarray(rng.uniform(size=shape) < 0.15)
    phase = sparse.empty_sparse_phase_state(net, nc, nb)
    first = cuda_csr.cuda_sparse_step(net, cp.zeros(shape), phase, pickup, v, rate, settle, 1.0, max_nnz=1 << 24)
    return (net, first.mobile_remaining_kg, first.phase_after, pickup * 0.5, v, rate, settle, 1.0)


@needs_gpu
@pytest.mark.parametrize("ny,nx,nc,nb,n_sub", [(120, 60, 6, 32, 1), (120, 60, 6, 32, 2), (96, 64, 3, 8, 1)])
def test_device_whole_step_peak_and_retention(b76, ny, nx, nc, nb, n_sub):
    """Whole-step MemoryHook live peaks (stored-bin case, warmed): the new PUBLIC step equals B76 exactly (peak and
    retained bytes); the PRIVATE step's peak is lower and it retains exactly the 8 class grids it does not return
    (decay / settled deposition, internal in / out, divergence, M + P, arrival, crossing) plus the two (nc,)
    per-class before / after totals (one 512-B pool unit each)."""
    args = _sized_case(ny, nx, nc, nb)
    kw = {"n_substeps": n_sub, "max_nnz": 1 << 24}
    D = h._r512(ny * nx * nc * 8)
    old = [_peak(b76.cuda_sparse_step, args, kw) for _ in range(2)]
    pub = [_peak(cuda_csr.cuda_sparse_step, args, kw) for _ in range(2)]
    priv = [_peak(cuda_csr._cuda_sparse_accounting_step, args, kw) for _ in range(2)]
    print(f"CW-B93 peaks/retained {ny}x{nx}x{nc} B{nb} sub{n_sub} D={D}: B76 {old} public {pub} private {priv}")
    assert old[0] == old[1] and pub[0] == pub[1] and priv[0] == priv[1]
    assert pub[0] == old[0]  # the public path is B76 in memory, not only in values
    assert old[0][1] - priv[0][1] == 8 * D + 2 * POOL_UNIT
    assert priv[0][0] < old[0][0]


@needs_gpu
def test_device_fixed_private_pool_cap_b76_refuses_inside_step_private_returns(b76):
    """ONE deterministic cap in ONE private pool (inputs, network binding and warm caches allocated in it first):
    cap = used-after-warmup + 512-aligned midpoint of (private live peak, B76 live peak). B76 must raise an actual
    OutOfMemoryError INSIDE the step; the private step must then return entirely (synchronized). Skips honestly if no
    separation exists."""
    pool = cp.cuda.MemoryPool()
    previous = cp.cuda.get_allocator()
    cp.cuda.set_allocator(pool.malloc)
    try:
        args = _sized_case(120, 60, 6, 32)
        kw = {"max_nnz": 1 << 24}
        old_peak, _ = _peak(b76.cuda_sparse_step, args, kw)
        new_peak, _ = _peak(cuda_csr._cuda_sparse_accounting_step, args, kw)
        gc.collect()
        cp.cuda.Device().synchronize()
        pool.free_all_blocks()
        used = int(pool.used_bytes())
        if not new_peak < old_peak:
            pytest.skip(f"no separation (private {new_peak}, B76 {old_peak})")
        cap = used + ((new_peak + old_peak) // 2) // POOL_UNIT * POOL_UNIT
        assert used + new_peak <= cap < used + old_peak
        pool.set_limit(size=cap)
        print(f"CW-B93 fixed cap: used {used} + peaks (private {new_peak}, B76 {old_peak}) -> cap {cap}")
        with pytest.raises(cp.cuda.memory.OutOfMemoryError):
            b76.cuda_sparse_step(*args, **kw)
        gc.collect()
        pool.free_all_blocks()
        assert int(pool.used_bytes()) == used  # the refused step left nothing behind
        out = cuda_csr._cuda_sparse_accounting_step(*args, **kw)
        cp.cuda.Device().synchronize()
        assert type(out) is _SparseAccountingStep and out.transport.y_face_net_kg.shape == (121, 60, 6)
        del out
    finally:
        pool.set_limit(size=0)
        cp.cuda.set_allocator(previous)
        gc.collect()
        pool.free_all_blocks()


@needs_gpu
def test_device_integrated_complete_wet_restart_driver_byte_identical_to_b76(tmp_path):
    """The existing B20 complete-driver dump (quiet completion + dry reset, wet pause, checkpoint save / load and
    resumed completion) on the CuPy windowed CSR case, in a fresh process per tree: this tree == frozen B76, every
    array and count."""
    import test_accounting_step as b20

    b20.backend_xp("cupy")
    a_arrays, a_counts = b20.run_driver(B76, "window_csr", "cupy", tmp_path, "b76")
    b_arrays, b_counts = b20.run_driver(TREE, "window_csr", "cupy", tmp_path, "b93")
    assert a_counts.pop("maple_syrup") == str(B76_PKG) and b_counts.pop("maple_syrup") == str(PKG)
    assert a_counts == b_counts and sorted(a_arrays) == sorted(b_arrays)
    for key in a_arrays:
        assert a_arrays[key].dtype == b_arrays[key].dtype and a_arrays[key].shape == b_arrays[key].shape, key
        assert a_arrays[key].tobytes() == b_arrays[key].tobytes(), key


def test_inspect_sources_are_the_tree_files():
    assert Path(inspect.getsourcefile(cuda_csr)).resolve() == PKG / "characteristic_cuda_csr.py"
    assert Path(inspect.getsourcefile(event)).resolve() == PKG / "sediment_event.py"
