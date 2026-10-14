"""CW-B63: the non-outlet export flag is computed before the face arrays (measured B61 refusal site).

Oracle: the stopped B59 candidate `characteristic_cuda_csr.py` (`4c9ea491...`), hard hash-pinned and executed as a
distinct module; every other package module of this tree must equal the B59 parent tree byte for byte.
CPU tests (author): pins, all other 55 modules identical to B59, exact diff (only the flag move), unchanged kernels,
the 39-entry flag packet identical, AST proof the flag is the original expression on never-again-written operands,
used once and released before `resolve`. DEVICE tests (root, GPU0; skipped here, a skip claims nothing): raw-byte
differential against B59, identical refusals (inputs, Courant, capacity, multiple violations, injected output
corruption), the preserved B59 combined-pool first-violation negative control, a targeted injection whose FIRST
violation is the non-outlet flag, and exact MemoryHook accounting at the non-outlet check (old-source negative
control), at the first face allocation (one extra 0-d flag) and at the result (no retained scalar).
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maple")

import test_cuda_csr_lifetimes as h  # corrected CW-B52 harness (this tree)
import test_cuda_pool_flag_lifetime as p59  # CW-B59 suite (this tree; used for its helpers and negative control)
import test_cuda_validation_lifetimes as v  # CW-B54 harness (this tree)

from maple_syrup import characteristic_cuda_csr as cuda_csr

PRIVATE_PKG = Path(__file__).resolve().parents[2] / "src" / "maple_syrup"
PRIVATE_FILE = PRIVATE_PKG / "characteristic_cuda_csr.py"
B59_PKG = Path("/home/okin/SYRUP/outputs/conservative_water/cw_b59_cuda_pool_flag_lifetime/src/maple_syrup")
B59_FILE = B59_PKG / "characteristic_cuda_csr.py"
B59_SHA256 = "4c9ea4910806ff7491861fac9db9da0cf3ec02ae435e1bd72bf5f00ae2767e08"
OUTLET_EXPR = "true_flag(export[~outlet] != 0.0)"
OUTLET_MESSAGE = "export requested at a non-outlet cell"
needs_gpu = h.needs_gpu
_r512 = h._r512


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def b59():
    if not B59_FILE.is_file():
        pytest.skip("pinned B59 oracle unavailable; no before/after claim")
    if _sha(B59_FILE) != B59_SHA256:
        pytest.fail(f"{B59_FILE} no longer has its pinned hash; the oracle comparison would be invalid")
    return v._exec_module(B59_FILE.read_text(), B59_FILE, "_cw_b63_b59_characteristic_cuda_csr")


def _func(source):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "cuda_sparse_step")


def _norm(expr: str) -> str:
    return ast.unparse(ast.parse(expr, mode="eval").body)


# =============================================================================================================
# CPU
# =============================================================================================================
def test_private_module_pinned_oracle_and_b59_parent_modules(b59):
    assert Path(cuda_csr.__file__).resolve() == PRIVATE_FILE
    assert h.cuda_csr is cuda_csr and v.cuda_csr is cuda_csr and p59.cuda_csr is cuda_csr
    assert h.value_rec(np) == ("module", "numpy")  # corrected comparator
    assert b59.cuda_sparse_step is not cuda_csr.cuda_sparse_step
    private = {str(p.relative_to(PRIVATE_PKG)): _sha(p) for p in PRIVATE_PKG.rglob("*.py") if "__pycache__" not in p.parts}
    parent = {str(p.relative_to(B59_PKG)): _sha(p) for p in B59_PKG.rglob("*.py") if "__pycache__" not in p.parts}
    assert set(private) == set(parent) and len(parent) == 55
    assert sorted(k for k in private if private[k] != parent[k]) == ["characteristic_cuda_csr.py"]
    assert parent["characteristic_cuda_csr.py"] == B59_SHA256


def test_diff_against_b59_is_exactly_the_flag_move():
    removed, added = [], []
    for line in difflib.unified_diff(B59_FILE.read_text().splitlines(), PRIVATE_FILE.read_text().splitlines(),
                                     lineterm="", n=0):
        if line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:].strip())
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:].strip())
    assert removed == [f'checks.forbid({OUTLET_EXPR}, "{OUTLET_MESSAGE}")',
                       "del mass_finite_flag, mass_negative_flag, balance_flag, phase_flags, row_flag, pool_flag"]
    assert [t for t in added if not t.startswith("#")] == [
        f"nonoutlet_export_flag = {OUTLET_EXPR}", f'checks.forbid(nonoutlet_export_flag, "{OUTLET_MESSAGE}")',
        "del mass_finite_flag, mass_negative_flag, balance_flag, phase_flags, row_flag, pool_flag, nonoutlet_export_flag"]


def test_kernels_and_packet_identical_to_b59(b59):
    assert cuda_csr.COMPILE_OPTIONS == b59.COMPILE_OPTIONS and cuda_csr.KERNEL_NAMES == b59.KERNEL_NAMES
    for nb in (1, 8, 32, 100, 128):
        assert cuda_csr.kernel_source(nb) == b59.kernel_source(nb)
    expected = v._packet(B59_FILE.read_text())
    assert len(expected) == 39 and v._packet(PRIVATE_FILE.read_text()) == expected
    assert expected[-1] == ("forbid", repr(OUTLET_MESSAGE))  # still the LAST packet entry


def test_flag_is_the_original_expression_on_unwritten_operands_and_released_before_resolve():
    func = _func(PRIVATE_FILE.read_text())

    def single_assign(name):
        nodes = [n for n in ast.walk(func) if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
        assert len(nodes) == 1, name
        return nodes[0]

    flag = single_assign("nonoutlet_export_flag")
    assert ast.unparse(flag.value) == _norm(OUTLET_EXPR)
    at = flag.lineno
    assert single_assign("pool_flag").lineno < at < single_assign("crossing").lineno < single_assign("x_gross").lineno
    assert single_assign("outlet").lineno < at and ast.unparse(single_assign("outlet").value) == "network.outlet_flat"
    assert single_assign("export").lineno < at
    resolve = max(n.lineno for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "resolve")
    enqueue = [n for n in ast.walk(func) if v._check_call(n) and ast.unparse(n.args[0]) == "nonoutlet_export_flag"]
    assert len(enqueue) == 1 and at < enqueue[0].lineno < resolve
    for node in ast.walk(func):  # nothing rebinds or writes `export` / `outlet` after the precompute
        if getattr(node, "lineno", 0) <= at:
            continue
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name) and not isinstance(node, ast.Delete):
                        assert name.id not in ("export", "outlet"), node.lineno
                    if isinstance(name, ast.Subscript):
                        assert ast.unparse(name.value) not in ("export", "outlet"), node.lineno
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                assert not (kw.arg == "out" and ast.unparse(kw.value) in ("export", "outlet")), node.lineno
            if ast.unparse(node.func).endswith("scatter_add"):
                assert ast.unparse(node.args[0]) not in ("export", "outlet")
    loads = [n.lineno for n in ast.walk(func) if isinstance(n, ast.Name) and n.id == "nonoutlet_export_flag"
             and isinstance(n.ctx, ast.Load)]
    assert loads == [enqueue[0].lineno]
    deletes = [n.lineno for n in ast.walk(func) if isinstance(n, ast.Delete)
               and any(isinstance(t, ast.Name) and t.id == "nonoutlet_export_flag" for t in n.targets)]
    assert len(deletes) == 1 and enqueue[0].lineno < deletes[0] < resolve
    assert ast.unparse(func).count(_norm(OUTLET_EXPR)) == 1  # computed once (the precompute), never recomputed


# =============================================================================================================
# Device (root)
# =============================================================================================================
@needs_gpu
@pytest.mark.parametrize("case", h.CASES)
@pytest.mark.parametrize("n_substeps", [1, 2, 3])
@pytest.mark.parametrize("nb", [1, 32])
@pytest.mark.parametrize("nc", [1, 6])
def test_raw_byte_differential_against_b59(b59, case, n_substeps, nb, nc):
    net, (mobile, pickup, vel, rate, settle), phase = h.build_case(case, nc, nb)
    rec, step, _ref = h.assert_same_outcome(b59, net, mobile, phase, pickup, vel, rate, settle, 1.0,
                                            n_substeps=n_substeps, max_nnz=1 << 20)
    assert rec[0] == "ok", rec
    if case not in ("dry_empty", "signed_zero"):
        rec2, _s, _r = h.assert_same_outcome(b59, net, step.mobile_remaining_kg, step.phase_after, pickup * 0.5, vel,
                                             rate, settle, 1.0, n_substeps=n_substeps, max_nnz=1 << 20)
        assert rec2[0] == "ok"


@needs_gpu
def test_identical_input_courant_capacity_and_multiple_violation_refusals(b59):
    p59.test_identical_input_courant_and_capacity_refusals(b59)  # same cases, now against the B59 oracle


@needs_gpu
@pytest.mark.parametrize("mode", v.MODES)
def test_injected_output_corruption_refused_identically_to_b59(b59, monkeypatch, mode):
    p59.test_injected_output_corruption_refused_identically_to_b55(b59, monkeypatch, mode)


@needs_gpu
def test_preserved_combined_pool_first_violation_negative_control(b59, monkeypatch):
    # The B59 targeted combined-pool injection: still the FIRST violation, identically in B59 and B63.
    p59.test_combined_pool_flag_is_the_first_violation_and_identical_to_b55(b59, monkeypatch)


def _redirect_to_export(module, net, cell, monkeypatch):
    """Private copies of the binding's `outlet` (True) and `receiver_slot` (the slot itself) at `cell`'s slot, handed
    to the count and write kernels only; `network.outlet_flat`, the binding and every caller array are untouched.
    `outlet` alone is NOT enough: arrivals are generated from `receiver_slot` (`ms_row_candidates`), so the crossing
    would also arrive at the receiver and the earlier per-class conservation flag would fail first. With both, the
    slot is encoded exactly like a real outlet (`receiver_index = self`), the crossing is exported and every balance
    holds, so the only violation is export at a cell the NETWORK says is not an outlet."""
    cp = h.cp
    binding = module.bind_network(net)
    slot = int(h.to_host(binding.slot_of_cell)[cell])
    outlet_copy = binding.outlet.copy()
    outlet_copy[slot] = True
    receiver_copy = binding.receiver_slot.copy()
    receiver_copy[slot] = slot
    real = module._function

    def function(name, nb):
        kernel = real(name, nb)

        def launch(grid, block, args):
            if name in ("ms_csr_count", "ms_csr_write"):
                args = tuple(outlet_copy if a is binding.outlet else receiver_copy if a is binding.receiver_slot else a
                             for a in args)
            return kernel(grid, block, args)

        return launch

    monkeypatch.setattr(module, "_function", function)
    return cp


@needs_gpu
def test_nonoutlet_export_is_the_first_violation_and_identical_to_b59(b59, monkeypatch):
    """Chain of 4 cells, outlet = cell 0. Pickup at cell 1 (non-outlet), travel 0.25 < dx: step 1 stores a bin; step 2
    carries it across cell 1's out-face. Clean positive control: step 2 succeeds identically in B59 and B63 with an
    internal crossing from cell 1 and no export there. Injected (count/write kernels only): cell 1's crossing is
    exported; B59's FIRST violation is the non-outlet message and B63 is identical; caller bytes unchanged."""
    net = h.device_network(h.chain_full(4))
    ny, nx = net.shape
    outlets = np.flatnonzero(np.asarray(net.outlet_host).reshape(-1))
    cell = next(c for c in range(ny * nx) if c not in outlets and bool(np.asarray(net.active_host).reshape(-1)[c]))
    cp = h.cp
    shape = (ny, nx, 1)
    pickup = np.zeros(shape)
    pickup.reshape(-1, 1)[cell] = 1.0
    mobile, pickup, vel, rate, settle = (cp.asarray(a) for a in (np.zeros(shape), pickup, np.full(shape, 0.25),
                                                                  np.full(shape, 1.0), np.zeros(shape, dtype=bool)))
    states, records = {}, {}
    for label, module in (("b63", cuda_csr), ("b59", b59)):
        phase = module.empty_cuda_sparse_phase_state(net, 1, 8)
        first = module.cuda_sparse_step(net, mobile, phase, pickup, vel, rate, settle, 1.0)
        states[label] = (first.mobile_remaining_kg, first.phase_after, module)
    clean = {}
    for label, (mob, phase, module) in states.items():
        clean[label], step = h.outcome(module, net, mob, phase, pickup * 0.0, vel, rate, settle, 1.0)
        if label == "b59":
            out = float(h.to_host(step.transport.internal_transfer_out_kg).reshape(-1)[cell])
            exp = float(h.to_host(step.transport.export_request_kg).reshape(-1)[cell])
            assert out > 1e-9 and exp == 0.0, (out, exp)  # clean positive control, not assumed
    assert clean["b63"] == clean["b59"] and clean["b59"][0] == "ok"
    for label, (mob, phase, module) in states.items():
        with monkeypatch.context() as m:
            _redirect_to_export(module, net, cell, m)
            outlet_before = h.arr_rec(net.outlet_flat)
            caller = h.caller_rec(mob, pickup, vel, rate, settle, phase=phase)
            records[label], _ = h.outcome(module, net, mob, phase, pickup * 0.0, vel, rate, settle, 1.0)
            assert h.arr_rec(net.outlet_flat) == outlet_before
            assert h.caller_rec(mob, pickup, vel, rate, settle, phase=phase) == caller
    assert records["b59"][:3] == ("refused", "TransportError", OUTLET_MESSAGE), records["b59"]
    assert records["b63"] == records["b59"]


def _outlet_check_record(module, monkeypatch, net, *args, **kwargs):
    """At the non-outlet `true_flag` call of `module`: (live bytes, sizes allocated since the previous true_flag call
    of cuda_sparse_step that are already freed)."""
    from maple.core import backend

    cp = h.cp
    lines, first = inspect.getsourcelines(module.cuda_sparse_step)
    target = first + next(i for i, t in enumerate(lines) if OUTLET_EXPR in t)
    code = module.cuda_sparse_step.__code__
    tracker, out, mark = p59._Hook(), {}, {"i": 0}
    real = backend.true_flag

    def true_flag(arr):
        caller = sys._getframe(1)
        if caller.f_code is code:
            if caller.f_lineno == target:
                window = tracker.log[mark["i"]:]
                out["current"] = tracker.current
                out["freed"] = sorted(size for key, size in window if key not in tracker.live)
            mark["i"] = len(tracker.log)
        return real(arr)

    module.cuda_sparse_step(net, *args, **kwargs)  # warm
    cp.cuda.Device().synchronize()
    monkeypatch.setattr(backend, "true_flag", true_flag)
    with tracker.hook():
        step = module.cuda_sparse_step(net, *args, **kwargs)
        cp.cuda.Device().synchronize()
    monkeypatch.setattr(backend, "true_flag", real)
    assert "current" in out
    return out, step


@needs_gpu
@pytest.mark.parametrize("n_substeps", [1, 2])
@pytest.mark.parametrize("stored", [False, True])
def test_memory_nonoutlet_check_first_face_and_result(b59, monkeypatch, n_substeps, stored):
    """Exact, with each module on a phase bound to its own slot maps:
      at the non-outlet check      B59 - B63 = D (crossing) + 2Fx + 2Fy (faces) + 28 x 512, the output flags B59 has
                                   already enqueued there (13 finite + 11 negative + export>T + deposition>T + residual
                                   finite + residual bound; the pool/balance/phase/row/mass flags are precomputed in
                                   both); in BOTH the window before the flag allocates and frees exactly one gathered
                                   (k, nc) FP64 copy, k = non-outlet cells (old-source negative control: in B59 after
                                   the faces, in B63 before them);
      at the first face allocation B59 - B63 = -512 (B63's retained 0-d flag);
      at the result                B59 - B63 = 0."""
    nc, nb = 6, 8
    net, (mobile, pickup, vel, rate, settle), _phase = h.build_case("masked", nc, nb)

    def own_state(module):
        phase = module.empty_cuda_sparse_phase_state(net, nc, nb)
        mob = mobile
        if stored:
            warm = module.cuda_sparse_step(net, mobile, phase, pickup, vel, rate, settle, 1.0)
            mob, phase = warm.mobile_remaining_kg, warm.phase_after
            assert phase.nnz > 0
        return mob, phase

    args = (pickup, vel, rate, settle, 1.0)
    check, faces, steps = {}, {}, {}
    for label, module in (("b63", cuda_csr), ("b59", b59)):
        mob, phase = own_state(module)
        check[label], steps[label] = _outlet_check_record(module, monkeypatch, net, mob, phase, *args,
                                                          n_substeps=n_substeps)
        mob, phase = own_state(module)
        events, _s = v._live_at_faces_and_result(module, monkeypatch, net, mob, phase, *args, n_substeps=n_substeps)
        faces[label] = dict(events)
    assert h.value_rec(steps["b63"]) == h.value_rec(steps["b59"])
    ny, nx = net.shape
    D = _r512(ny * nx * nc * 8)
    Fx, Fy = _r512(ny * (nx + 1) * nc * 8), _r512((ny + 1) * nx * nc * 8)
    k = int(np.count_nonzero(~np.asarray(net.outlet_host)))
    gather = _r512(k * nc * 8)
    got = {"check": check["b59"]["current"] - check["b63"]["current"],
           "x_gross": faces["b59"]["x_gross"] - faces["b63"]["x_gross"],
           "result": faces["b59"]["result"] - faces["b63"]["result"]}
    want = {"check": D + 2 * Fx + 2 * Fy + 28 * 512, "x_gross": -512, "result": 0}
    print("CW-B63 live-byte differences:", got, "freed windows:", check["b59"]["freed"], check["b63"]["freed"])
    assert got == want, (got, want)
    for label in ("b59", "b63"):
        assert check[label]["freed"].count(gather) == 1, (label, gather, check[label]["freed"])
