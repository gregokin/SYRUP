"""CW-B59: the combined-pool output flag is computed before the face arrays (measured B57 refusal site).

Oracle: the root-adopted B55 `characteristic_cuda_csr.py` (`19f01d9e...`), hash-pinned and executed as a distinct module.
CPU tests (author): pin, exact diff scope, unchanged kernels, the 39-entry flag packet identical to B55, and AST proof
that the flag is the original expression on arrays never written before its enqueue, released before `resolve`.
DEVICE tests (root, GPU0; skipped here, a skip claims nothing): raw-byte differential against B55 over all harness
cases, identical refusals (inputs, Courant, capacity, injected output corruption incl. a corruption whose FIRST
violation is the combined-pool flag), and exact MemoryHook accounting at the combined-pool check (old-source negative
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
import test_cuda_validation_lifetimes as v  # CW-B54 harness (this tree)

from maple_syrup import characteristic_cuda_csr as cuda_csr

PRIVATE_FILE = Path(__file__).resolve().parents[2] / "src" / "maple_syrup" / "characteristic_cuda_csr.py"
B55_FILE = Path("/home/okin/SYRUP/outputs/conservative_water/cw_b55_joint_runtime_frozen/src/maple_syrup/"
                "characteristic_cuda_csr.py")
B55_SHA256 = "19f01d9e0e4ad1c57345f41fadcb1b0fd4a2d4b15257b299d6066d7afb7039fb"
POOL_EXPR = "true_flag((deposition + export) > T * (1.0 + 4.0 * _EPS))"
POOL_MESSAGE = "deposition plus export requests exceed the post-transfer pool"
needs_gpu = h.needs_gpu
_r512 = h._r512


@pytest.fixture(scope="module")
def b55():
    if not B55_FILE.is_file():
        pytest.skip("pinned B55 oracle unavailable; no before/after claim")
    source = B55_FILE.read_bytes()
    if hashlib.sha256(source).hexdigest() != B55_SHA256:
        pytest.fail(f"{B55_FILE} no longer has its pinned hash; the oracle comparison would be invalid")
    return v._exec_module(source.decode(), B55_FILE, "_cw_b59_b55_characteristic_cuda_csr")


def _func(source):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "cuda_sparse_step")


# =============================================================================================================
# CPU
# =============================================================================================================
def test_private_module_and_pinned_oracle(b55):
    assert Path(cuda_csr.__file__).resolve() == PRIVATE_FILE
    assert h.cuda_csr is cuda_csr and v.cuda_csr is cuda_csr
    assert h.value_rec(np) == ("module", "numpy")  # corrected comparator
    assert b55.cuda_sparse_step is not cuda_csr.cuda_sparse_step


def test_diff_against_b55_is_exactly_the_flag_move():
    old = B55_FILE.read_text().splitlines()
    new = PRIVATE_FILE.read_text().splitlines()
    removed, added = [], []
    for line in difflib.unified_diff(old, new, lineterm="", n=0):
        if line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:].strip())
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:].strip())
    code_added = [t for t in added if not t.startswith("#")]
    assert removed == [f"checks.forbid({POOL_EXPR},",
                       "del mass_finite_flag, mass_negative_flag, balance_flag, phase_flags, row_flag"]
    assert code_added == [f"pool_flag = {POOL_EXPR}", "checks.forbid(pool_flag,",
                          "del mass_finite_flag, mass_negative_flag, balance_flag, phase_flags, row_flag, pool_flag"]


def test_kernels_and_packet_identical_to_b55(b55):
    assert cuda_csr.COMPILE_OPTIONS == b55.COMPILE_OPTIONS and cuda_csr.KERNEL_NAMES == b55.KERNEL_NAMES
    for nb in (1, 8, 32, 100, 128):
        assert cuda_csr.kernel_source(nb) == b55.kernel_source(nb)
    expected = v._packet(B55_FILE.read_text())
    assert len(expected) == 39 and v._packet(PRIVATE_FILE.read_text()) == expected
    assert ("forbid", repr(POOL_MESSAGE)) in expected


def test_flag_is_the_original_expression_on_unwritten_arrays_and_released_before_resolve():
    func = _func(PRIVATE_FILE.read_text())
    assign = [n for n in ast.walk(func) if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "pool_flag"]
    assert len(assign) == 1 and ast.unparse(assign[0].value) == ast.unparse(ast.parse(POOL_EXPR, mode="eval").body)
    at = assign[0].lineno
    first = {name: min(n.lineno for n in ast.walk(func) if isinstance(n, ast.Assign) and len(n.targets) == 1
                       and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)
             for name in ("T", "deposition", "crossing", "x_gross")}
    assert first["T"] < at and first["deposition"] < at < first["crossing"] < first["x_gross"]
    resolve = max(n.lineno for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "resolve")
    enqueue = [n for n in ast.walk(func) if v._check_call(n) and ast.unparse(n.args[0]) == "pool_flag"]
    assert len(enqueue) == 1 and at < enqueue[0].lineno < resolve
    # nothing rebinds or writes T / deposition / export between the precompute and the enqueue
    for node in ast.walk(func):
        if not (at < getattr(node, "lineno", 0) < enqueue[0].lineno):
            continue
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in ast.walk(target):
                    assert not (isinstance(name, ast.Name) and name.id in ("T", "deposition", "export")), node.lineno
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                assert not (kw.arg == "out" and ast.unparse(kw.value) in ("T", "deposition", "export")), node.lineno
            if ast.unparse(node.func).endswith("scatter_add"):
                assert ast.unparse(node.args[0]) not in ("T", "deposition", "export")
    loads = [n for n in ast.walk(func) if isinstance(n, ast.Name) and n.id == "pool_flag" and isinstance(n.ctx, ast.Load)]
    assert [n.lineno for n in loads] == [enqueue[0].args[0].lineno]
    deletes = [n.lineno for n in ast.walk(func) if isinstance(n, ast.Delete)
               and any(isinstance(t, ast.Name) and t.id == "pool_flag" for t in n.targets)]
    assert len(deletes) == 1 and enqueue[0].lineno < deletes[0] < resolve


# =============================================================================================================
# Device (root)
# =============================================================================================================
@needs_gpu
@pytest.mark.parametrize("case", h.CASES)
@pytest.mark.parametrize("n_substeps", [1, 2, 3])
@pytest.mark.parametrize("nb", [1, 32])
@pytest.mark.parametrize("nc", [1, 6])
def test_raw_byte_differential_against_b55(b55, case, n_substeps, nb, nc):
    net, (mobile, pickup, vel, rate, settle), phase = h.build_case(case, nc, nb)
    rec, step, _ref = h.assert_same_outcome(b55, net, mobile, phase, pickup, vel, rate, settle, 1.0,
                                            n_substeps=n_substeps, max_nnz=1 << 20)
    assert rec[0] == "ok", rec
    if case not in ("dry_empty", "signed_zero"):
        rec2, _s, _r = h.assert_same_outcome(b55, net, step.mobile_remaining_kg, step.phase_after, pickup * 0.5, vel,
                                             rate, settle, 1.0, n_substeps=n_substeps, max_nnz=1 << 20)
        assert rec2[0] == "ok"


@needs_gpu
def test_identical_input_courant_and_capacity_refusals(b55):
    net, (mobile, pickup, vel, rate, settle), phase = h.build_case("wet_fresh", 2, 8)
    cp = h.cp
    _ok, step, _ = h.assert_same_outcome(b55, net, mobile, phase, pickup, vel, rate, settle, 1.0)
    base_mobile, base_phase = step.mobile_remaining_kg, step.phase_after
    fast = vel * 0.0 + 0.3
    cases = [{"pickup": -pickup - 1.0}, {"rate": rate * cp.inf}, {"v": fast},
             {"pickup": cp.where(cp.arange(2) == 0, cp.nan, pickup), "v": fast, "rate": -rate - 1.0},
             {"mobile": base_mobile * 0.0}, {"max_nnz": 1}]
    for change in cases:
        args = {"mobile": base_mobile, "phase": base_phase, "pickup": pickup, "v": vel, "rate": rate, "settle": settle}
        kw = {k: x for k, x in change.items() if k not in args}
        args.update({k: x for k, x in change.items() if k in args})
        rec, _n, _o = h.assert_same_outcome(b55, net, args["mobile"], args["phase"], args["pickup"], args["v"],
                                            args["rate"], args["settle"], 1.0, **kw)
        assert rec[0] == "refused", (change, rec)


@needs_gpu
@pytest.mark.parametrize("mode", v.MODES)
def test_injected_output_corruption_refused_identically_to_b55(b55, monkeypatch, mode):
    net, (mobile, pickup, vel, rate, settle), _phase = h.build_case("wet_fresh", 3, 8)
    records = {}
    for label, module in (("b59", cuda_csr), ("b55", b55)):
        phase = module.empty_cuda_sparse_phase_state(net, 3, 8)
        with monkeypatch.context() as m:
            v._corrupting(module, mode, m)
            before = h.caller_rec(mobile, pickup, vel, rate, settle, phase=phase)
            records[label], _ = h.outcome(module, net, mobile, phase, pickup, vel, rate, settle, 1.0)
            assert h.caller_rec(mobile, pickup, vel, rate, settle, phase=phase) == before
    assert records["b59"][0] == "refused" and records["b59"] == records["b55"]


def _outlet_chain(nc=1):
    cp = h.cp
    net = h.device_network(h.chain_full(4))
    ny, nx = net.shape
    outlet = int(np.flatnonzero(np.asarray(net.outlet_host).reshape(-1))[0])
    shape = (ny, nx, nc)
    pickup = np.zeros(shape)
    pickup.reshape(-1, nc)[outlet] = 1.0
    fields = [cp.asarray(a) for a in (np.zeros(shape), pickup, np.full(shape, 0.25), np.full(shape, 1.0),
                                      np.zeros(shape, dtype=bool))]
    return net, outlet, fields


@needs_gpu
def test_combined_pool_flag_is_the_first_violation_and_identical_to_b55(b55, monkeypatch):
    """Two steps at an outlet: step 1 stores a bin (travel 0.25 < dx), step 2 carries it across the outlet face, so
    the outlet cell has BOTH deposition and export (asserted on the uncorrupted B55 step). Then the compact remaining
    pool of that row is set to -1e-12 kg after the normalize kernel: T = dep + export - 1e-12 stays >= both requests
    and >= 0, so every earlier flag passes and the combined-pool flag is the FIRST violation, in B55 and in B59."""
    net, outlet, (mobile, pickup, vel, rate, settle) = _outlet_chain()
    states = {}
    for label, module in (("b59", cuda_csr), ("b55", b55)):
        phase = module.empty_cuda_sparse_phase_state(net, 1, 8)
        first = module.cuda_sparse_step(net, mobile, phase, pickup, vel, rate, settle, 1.0)
        states[label] = (first.mobile_remaining_kg, first.phase_after, module)
    clean = b55.cuda_sparse_step(net, states["b55"][0], states["b55"][1], pickup * 0.0, vel, rate, settle, 1.0)
    dep = float(h.to_host(clean.transport.deposition_request_kg).reshape(-1)[outlet])
    exp = float(h.to_host(clean.transport.export_request_kg).reshape(-1)[outlet])
    assert dep > 1e-9 and exp > 1e-9, (dep, exp)  # harness precondition, not assumed
    records = {}
    for label, (mob, phase, module) in states.items():
        row = int(h.to_host(module.bind_network(net).slot_of_cell)[outlet])
        real = module._function

        def function(name, nb, real=real, row=row):
            kernel = real(name, nb)

            def launch(grid, block, args):
                kernel(grid, block, args)
                if name == "ms_csr_rowsum_normalize":
                    args[4][row] = -1e-12

            return launch

        with monkeypatch.context() as m:
            m.setattr(module, "_function", function)
            records[label], _ = h.outcome(module, net, mob, phase, pickup * 0.0, vel, rate, settle, 1.0)
    assert records["b55"][:3] == ("refused", "TransportError", POOL_MESSAGE), records["b55"]
    assert records["b59"] == records["b55"]


class _Hook:
    """Same-thread CuPy pool live bytes (rounded mem_size) and a malloc log."""

    def __init__(self):
        self.live, self.current, self.log = {}, 0, []

    def hook(self):
        tracker = self
        cp = h.cp

        class Hook(cp.cuda.MemoryHook):
            name = "CWB59LiveBytes"

            def malloc_postprocess(self, **kw):
                if kw.get("mem_ptr"):
                    key = (int(kw["device_id"]), int(kw["pmem_id"]))
                    tracker.current += int(kw["mem_size"]) - tracker.live.get(key, 0)
                    tracker.live[key] = int(kw["mem_size"])
                    tracker.log.append((key, int(kw["mem_size"])))

            def free_postprocess(self, **kw):
                key = (int(kw["device_id"]), int(kw["pmem_id"]))
                tracker.current -= tracker.live.pop(key, 0)

        return Hook()


def _pool_check_record(module, monkeypatch, net, *args, **kwargs):
    """At the combined-pool `true_flag` call of `module`: (live bytes, sizes of allocations made since the previous
    true_flag call of cuda_sparse_step that are already freed)."""
    from maple.core import backend

    cp = h.cp
    lines, first = inspect.getsourcelines(module.cuda_sparse_step)
    pool_line = first + next(i for i, t in enumerate(lines) if POOL_EXPR in t)
    code = module.cuda_sparse_step.__code__
    tracker, out, mark = _Hook(), {}, {"i": 0}
    real = backend.true_flag

    def true_flag(arr):
        caller = sys._getframe(1)
        if caller.f_code is code:
            if caller.f_lineno == pool_line:
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
def test_memory_combined_pool_check_first_face_and_result(b55, monkeypatch, n_substeps, stored):
    """Exact, with each module on a phase bound to its own slot maps:
      at the combined-pool check   B55 - B59 = D (crossing) + 2Fx + 2Fy (all four faces) + 26 x 512 (the output
                                   flags B55 has already enqueued there; B59 computes this flag before them), and in
                                   BOTH the original expression's two D temporaries are allocated and freed in the
                                   window just before the flag (old-source negative control: in B55 that window is
                                   after the faces, in B59 before them);
      at the first face allocation B55 - B59 = -512 (B59's retained 0-d flag only);
      at the result                B55 - B59 = 0 (no retained scalar)."""
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
    pool, faces, steps = {}, {}, {}
    for label, module in (("b59", cuda_csr), ("b55", b55)):
        mob, phase = own_state(module)
        pool[label], steps[label] = _pool_check_record(module, monkeypatch, net, mob, phase, *args,
                                                       n_substeps=n_substeps)
        mob, phase = own_state(module)
        events, _s = v._live_at_faces_and_result(module, monkeypatch, net, mob, phase, *args, n_substeps=n_substeps)
        faces[label] = dict(events)
    assert h.value_rec(steps["b59"]) == h.value_rec(steps["b55"])
    ny, nx = net.shape
    D = _r512(ny * nx * nc * 8)
    Fx, Fy = _r512(ny * (nx + 1) * nc * 8), _r512((ny + 1) * nx * nc * 8)
    got = {"pool": pool["b55"]["current"] - pool["b59"]["current"],
           "x_gross": faces["b55"]["x_gross"] - faces["b59"]["x_gross"],
           "result": faces["b55"]["result"] - faces["b59"]["result"]}
    want = {"pool": D + 2 * Fx + 2 * Fy + 26 * 512, "x_gross": -512, "result": 0}
    print("CW-B59 live-byte differences:", got, "freed windows:", pool["b55"]["freed"], pool["b59"]["freed"])
    assert got == want, (got, want)
    for label in ("b55", "b59"):
        assert pool[label]["freed"].count(D) == 2, (label, pool[label]["freed"])  # the two expression temporaries
