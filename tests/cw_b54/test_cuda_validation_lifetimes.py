"""CW-B54: CUDA CSR output-validation scratch released before the face arrays (layered on the CW-B52 candidate).

Oracles (distinct modules, executed from hash-pinned files):
* B46: frozen `cw_b46_runtime_frozen/.../characteristic_cuda_csr.py` (`c32b3ba1...`);
* B52: the CW-B52 candidate `cw_b52_cuda_csr_lifetimes/.../characteristic_cuda_csr.py` (`6ed3a89b...`, NOT accepted;
  used only as the layer this candidate is built on).

CPU tests (author): pins, change confined to the post-loop part of `cuda_sparse_step`, kernels unchanged, the output
flag packet has the SAME (kind, message) order as B52/B46 (with mutation negative controls), and the released names are
never read after their release. DEVICE tests (root, GPU0; skipped here, a skip claims nothing): raw-byte differential
against B46 and B52 over dry/signed-zero/wet/stored/terminal/masked and multi-substep cases; identical refusals for
multiple input errors and for output corruption injected into this step's own kernel outputs or tolerance; mutated
copies (a dropped / a reordered flag) are DETECTED; exact MemoryHook live bytes at the first face allocation and at
the result. The comparator is the corrected CW-B52 harness (module classification before any array probe).
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maple")

import test_cuda_csr_lifetimes as h  # the corrected CW-B52 harness of THIS tree (tests/cw_b52)

from maple_syrup import characteristic_cuda_csr as cuda_csr

PRIVATE_PKG = Path(__file__).resolve().parents[2] / "src" / "maple_syrup"
B46_FILE = Path("/home/okin/SYRUP/outputs/conservative_water/cw_b46_runtime_frozen/src/maple_syrup/"
                "characteristic_cuda_csr.py")
B52_FILE = Path("/home/okin/SYRUP/outputs/conservative_water/cw_b52_cuda_csr_lifetimes/src/maple_syrup/"
                "characteristic_cuda_csr.py")
PINS = {B46_FILE: "c32b3ba19b31e22cf8a225a9af0d27ea4fe0b590dd4cebc9c2afcb7c9404f468",
        B52_FILE: "6ed3a89bbea87a216afa6275593408931eea5e9239b142fc3a8d907aad4cbeb1"}
needs_gpu = h.needs_gpu
_r512 = h._r512


def _exec_module(source: str, path: Path, name: str):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 -- hash-pinned (or derived) oracle text
    return module


def _load_pinned(path: Path, name: str):
    if not path.is_file():
        pytest.skip(f"pinned oracle {path} unavailable; no before/after claim")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != PINS[path]:
        pytest.fail(f"{path} no longer has its pinned hash {PINS[path]}; the oracle comparison would be invalid")
    return _exec_module(source.decode(), path, name)


@pytest.fixture(scope="module")
def b46():
    return _load_pinned(B46_FILE, "_cw_b54_b46_characteristic_cuda_csr")


@pytest.fixture(scope="module")
def b52():
    return _load_pinned(B52_FILE, "_cw_b54_b52_characteristic_cuda_csr")


# =============================================================================================================
# CPU: provenance, scope, packet order (with negative controls), released names
# =============================================================================================================
def test_private_module_and_corrected_harness(b46, b52):
    assert Path(cuda_csr.__file__).resolve() == PRIVATE_PKG / "characteristic_cuda_csr.py"
    assert h.cuda_csr is cuda_csr  # the harness compares THIS tree's module
    assert h.value_rec(np) == ("module", "numpy")  # correction-1 classification: modules before any array probe
    assert b46.cuda_sparse_step is not cuda_csr.cuda_sparse_step is not b52.cuda_sparse_step


def test_change_vs_b52_is_post_loop_only_and_kernels_unchanged(b46, b52):
    old = B52_FILE.read_text().splitlines()
    new = (PRIVATE_PKG / "characteristic_cuda_csr.py").read_text().splitlines()
    lines, first = inspect.getsourcelines(b52.cuda_sparse_step)
    norm = first + next(i for i, t in enumerate(lines) if "norm_fn((grid,), (block,)" in t)
    last = first + len(lines) - 1
    changed = 0
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag != "equal":
            changed += 1
            assert norm < i1 + 1 and i2 <= last, (tag, i1, i2, norm, last)
    assert changed > 0
    for other in (b46, b52):
        assert cuda_csr.COMPILE_OPTIONS == other.COMPILE_OPTIONS and cuda_csr.KERNEL_NAMES == other.KERNEL_NAMES
        for nb in (1, 8, 32, 100, 128):
            assert cuda_csr.kernel_source(nb) == other.kernel_source(nb)


def _check_call(node):
    """(kind, message source) of a `checks.require/forbid(...)` call node, else None."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("require", "forbid") \
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "checks":
        return node.func.attr, ast.unparse(node.args[1])
    return None


def _packet(source: str):
    """Effective (kind, message) sequence of the OUTPUT flag packet of `cuda_sparse_step` in `source`, statement by
    statement after the output-validation marker: each `for name, array in ((name, ...), ...)` loop is expanded over
    its literal names (the f-string `{name}` filled in), each `for flag, message in phase_flags` loop over the
    `phase_flags.append((flag, message))` order, and `if` blocks are followed in source order."""
    func = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "cuda_sparse_step")
    marker = next(i for i, t in enumerate(source.splitlines(), 1) if "# --- output validation" in t)
    appended = sorted((n.lineno, ("forbid", ast.unparse(n.args[0].elts[1]))) for n in ast.walk(func)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "append"
                      and isinstance(n.func.value, ast.Name) and n.func.value.id == "phase_flags")
    out = []

    def visit(statements):
        for st in statements:
            if st.lineno <= marker and not isinstance(st, (ast.If, ast.For)):
                continue
            if isinstance(st, ast.For):
                calls = [c for n in ast.walk(st) if (c := _check_call(n))]
                if isinstance(st.iter, ast.Name) and st.iter.id == "phase_flags":
                    assert calls == [("forbid", "message")], calls
                    out.extend(m for _, m in appended)
                elif st.lineno > marker and calls:
                    names = [elt.elts[0].value for elt in st.iter.elts]
                    (kind, message), = calls
                    out.extend((kind, message.replace("{name}", nm)) for nm in names)
            elif isinstance(st, ast.If):
                visit(st.body)
                visit(st.orelse)
            elif isinstance(st, ast.Expr) and (c := _check_call(st.value)):
                out.append(c)

    visit(func.body)
    return out


def test_output_packet_order_is_identical_to_b52_and_b46_with_negative_controls():
    new_src = (PRIVATE_PKG / "characteristic_cuda_csr.py").read_text()
    expected = _packet(B52_FILE.read_text())
    assert _packet(B46_FILE.read_text()) == expected
    assert len(expected) == 39 and _packet(new_src) == expected
    # negative controls: a dropped flag and a reordered flag are DETECTED by this comparison
    balance = ('    checks.forbid(balance_flag,\n'
               '                  "per-cell mobile balance T - (M + P) = in - out violated beyond FP64 tolerance")\n')
    residual = '    checks.require(finite_flag(residual), "per-class budget residual is non-finite")\n'
    assert new_src.count(balance) == 1 and new_src.count(residual) == 1
    assert _packet(new_src.replace(balance, "")) != expected
    assert _packet(new_src.replace(balance + residual, residual + balance)) != expected
    row = '    checks.forbid(row_flag, "sparse characteristic transport left a row inconsistent with its remaining mass")\n'
    assert new_src.count(row) == 1 and _packet(new_src.replace(row, "")) != expected


def _func(source: str):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "cuda_sparse_step")


def test_precomputed_flags_are_the_original_expressions():
    """Each flag moved before the faces is textually the B52 expression (the balance flag via `abs_balance`, which is
    exactly `cp.abs(cell_balance)`; the maximum is `cp.max` of that same array), in the original order."""
    old = _func(B52_FILE.read_text())
    new = _func((PRIVATE_PKG / "characteristic_cuda_csr.py").read_text())
    old_args = [ast.unparse(n.args[0]) for n in ast.walk(old) if _check_call(n)]
    moved = [ast.unparse(n.args[0].elts[0]) for n in sorted(
        (n for n in ast.walk(new) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
         and n.func.attr == "append" and isinstance(n.func.value, ast.Name) and n.func.value.id == "phase_flags"),
        key=lambda n: n.lineno)]
    phase = ["true_flag(masses <= 0.0)",
             "true_flag((counts_after > 0) & (cp.abs(sums - 1.0) > fraction_sum_tolerance(nb)))",
             "true_flag(positions >= dx)", "true_flag(bin_index(positions, dx, nb, cp) != bins.astype(np.int64))",
             "true_flag(same_row & ~(bins[1:] > bins[:-1]))"]
    assert moved == phase and all(expr in old_args for expr in phase)
    assigned = {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in ast.walk(new)
                if isinstance(n, ast.Assign) and len(n.targets) == 1}
    assert assigned["abs_balance"] == "cp.abs(cell_balance)"
    assert assigned["balance_flag"] == "true_flag(abs_balance > cell_tol)"
    assert "true_flag(cp.abs(cell_balance) > cell_tol)" in old_args
    assert assigned["max_cell_balance"] == "cp.max(abs_balance)"
    assert "max_cell_balance_residual_kg=cp.max(cp.abs(cell_balance))" in ast.unparse(old)
    assert assigned["row_flag"] == "true_flag((counts_after > 0) != (remaining_c.reshape(-1) > 0.0))"
    assert assigned["row_flag"] in old_args
    assert assigned["counts_after"] == "cp.diff(row_ptr)" and assigned["mass_finite_flag"] == "finite_flag(masses)"
    assert assigned["mass_negative_flag"] == "negative_flag(masses)"
    assert assigned["rows_after"] == "_row_of_entry(cp, row_ptr, nnz_after)"
    assert assigned["same_row"] == "rows_after[1:] == rows_after[:-1]"


def test_released_names_are_not_read_after_their_release_and_precede_the_faces():
    func = next(n for n in ast.walk(ast.parse((PRIVATE_PKG / "characteristic_cuda_csr.py").read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == "cuda_sparse_step")
    faces = min(n.lineno for n in ast.walk(func) if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "x_gross")
    released = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    released.setdefault(target.id, []).append(node.lineno)
    for name in ("cell_balance", "cell_tol", "abs_balance", "counts_after", "remaining_c", "masses", "mass_new"):
        line = max(released[name])
        assert line < faces, (name, line, faces)
        later = [n.lineno for n in ast.walk(func) if isinstance(n, ast.Name) and n.id == name
                 and isinstance(n.ctx, ast.Load) and n.lineno > line]
        assert later == [], (name, later)
    resolve = max(n.lineno for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "resolve")
    assert max(released["balance_flag"]) < resolve and max(released["row_flag"]) < resolve
    # correction 1: the phase-flag loop variables must not keep the last flag alive past `resolve` (root's MemoryHook
    # measured exactly one retained 0-d block at the result when nnz > 0 before this release)
    loop = next(n for n in ast.walk(func) if isinstance(n, ast.For) and isinstance(n.iter, ast.Name)
                and n.iter.id == "phase_flags")
    for name in ("flag", "message"):
        assert any(loop.lineno < line < resolve for line in released.get(name, [])), name
    # the two (name, array) validation loops end on the `None` placeholder, so `array` names no device array after them
    literal_loops = [n for n in ast.walk(func) if isinstance(n, ast.For) and isinstance(n.iter, ast.Tuple)
                     and isinstance(n.target, ast.Tuple)]
    assert len(literal_loops) == 2
    assert all(ast.unparse(n.iter.elts[-1].elts[1]) == "None" for n in literal_loops)
    src = inspect.getsource(cuda_csr.cuda_sparse_step)
    assert "max_cell_balance_residual_kg=max_cell_balance" in src and "fastmath" not in src


# =============================================================================================================
# Device: raw-byte differential, refusals, injected corruption, negative controls, live bytes (root runs)
# =============================================================================================================
@needs_gpu
@pytest.mark.parametrize("case", h.CASES)
@pytest.mark.parametrize("n_substeps", [1, 2, 3])
@pytest.mark.parametrize("nb", [1, 32])
@pytest.mark.parametrize("nc", [1, 6])
def test_raw_byte_differential_against_b46_and_b52(b46, b52, case, n_substeps, nb, nc):
    net, (mobile, pickup, v, rate, settle), phase = h.build_case(case, nc, nb)
    for oracle in (b46, b52):
        rec, step, _ref = h.assert_same_outcome(oracle, net, mobile, phase, pickup, v, rate, settle, 1.0,
                                                n_substeps=n_substeps, max_nnz=1 << 20)
        assert rec[0] == "ok", rec
        if case not in ("dry_empty", "signed_zero"):  # continue from the published stored-bin state
            rec2, _s2, _r2 = h.assert_same_outcome(oracle, net, step.mobile_remaining_kg, step.phase_after,
                                                   pickup * 0.5, v, rate, settle, 1.0, n_substeps=n_substeps,
                                                   max_nnz=1 << 20)
            assert rec2[0] == "ok"
    if case == "dry_empty":
        assert step.nnz_after == 0


def _corrupting(module, mode, monkeypatch):
    """Inject corruption into THIS module's own kernel outputs (after the real launch) or its tolerance."""
    real = module._function
    cp = h.cp

    def function(name, nb):
        kernel = real(name, nb)

        def launch(grid, block, args):
            kernel(grid, block, args)
            if name == "ms_csr_write" and mode == "position" and args[-1].size:
                args[-1][...] = float(args[4])  # every stored position = dx
            if name == "ms_csr_write" and mode == "zero_mass" and args[-2].size:
                args[-2][0] = 0.0
            if name == "ms_csr_rowsum_normalize":
                fraction, remaining = args[5], args[4]
                if mode == "fraction_sum" and fraction.size:
                    fraction *= 1.5
                if mode == "nonfinite_fraction" and fraction.size:
                    fraction[0] = cp.nan
                if mode == "row_inconsistent":
                    remaining[...] = 0.0

        return launch

    monkeypatch.setattr(module, "_function", function)
    if mode == "balance":
        real_bound = module.summation_error_bound_kg
        monkeypatch.setattr(module, "summation_error_bound_kg", lambda n, s: -1.0 - real_bound(n, s))


MODES = ("position", "zero_mass", "fraction_sum", "nonfinite_fraction", "row_inconsistent", "balance")


@needs_gpu
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n_substeps", [1, 2])
def test_injected_output_corruption_is_refused_identically_to_b46_and_b52(b46, b52, monkeypatch, mode, n_substeps):
    net, (mobile, pickup, v, rate, settle), _phase = h.build_case("wet_fresh", 3, 8)
    records = {}
    for label, module in (("b54", cuda_csr), ("b46", b46), ("b52", b52)):
        phase = module.empty_cuda_sparse_phase_state(net, 3, 8)
        with monkeypatch.context() as m:
            _corrupting(module, mode, m)
            before = h.caller_rec(mobile, pickup, v, rate, settle, phase=phase)
            records[label], _ = h.outcome(module, net, mobile, phase, pickup, v, rate, settle, 1.0,
                                          n_substeps=n_substeps)
            assert h.caller_rec(mobile, pickup, v, rate, settle, phase=phase) == before
    assert records["b54"][0] == "refused", records["b54"]
    assert records["b54"] == records["b46"] == records["b52"]


@needs_gpu
def test_multiple_input_errors_are_refused_identically(b46, b52):
    net, (mobile, pickup, v, rate, settle), phase = h.build_case("wet_fresh", 2, 8)
    cp = h.cp
    bad = (cp.where(cp.arange(2) == 0, cp.nan, pickup), -rate - 1.0, v * 0.0 + 0.3)
    for oracle in (b46, b52):
        rec, _n, _o = h.assert_same_outcome(oracle, net, mobile, phase, bad[0], bad[2], bad[1], settle, 1.0)
        assert rec[0] == "refused", rec


def _mutated(source_replacements, name):
    text = (PRIVATE_PKG / "characteristic_cuda_csr.py").read_text()
    for old, new in source_replacements:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return _exec_module(text, PRIVATE_PKG / "characteristic_cuda_csr.py", name)


@needs_gpu
def test_negative_controls_dropped_or_reordered_flag_is_detected_on_device(b46, monkeypatch):
    balance = ('    checks.forbid(balance_flag,\n'
               '                  "per-cell mobile balance T - (M + P) = in - out violated beyond FP64 tolerance")\n')
    residual = ('    checks.require(finite_flag(residual), "per-class budget residual is non-finite")\n'
                '    checks.forbid(true_flag(cp.abs(residual) > tolerance),\n'
                '                  "per-class mobile mass not conserved by the transfer beyond the declared FP64 bound")\n')
    # the injected negative tolerance makes BOTH the per-cell balance and the per-class conservation flags fail, so
    # dropping the balance flag or moving it after the conservation flag changes the first reported violation
    dropped = _mutated([(balance, "")], "_cw_b54_mutant_dropped")
    reordered = _mutated([(balance + residual, residual + balance)], "_cw_b54_mutant_reordered")
    net, (mobile, pickup, v, rate, settle), _phase = h.build_case("wet_fresh", 3, 8)
    records = {}
    for label, module in (("b46", b46), ("b54", cuda_csr), ("dropped", dropped), ("reordered", reordered)):
        phase = module.empty_cuda_sparse_phase_state(net, 3, 8)
        with monkeypatch.context() as m:
            _corrupting(module, "balance", m)
            records[label], _ = h.outcome(module, net, mobile, phase, pickup, v, rate, settle, 1.0)
    assert records["b54"] == records["b46"] and "per-cell mobile balance" in records["b46"][2]
    assert records["dropped"] != records["b46"]
    assert records["reordered"] != records["b46"]


def _live_at_faces_and_result(module, monkeypatch, net, *args, **kwargs):
    cp = h.cp
    tracker = h._LiveBytes()
    events = []
    real_zeros, real_result = cp.zeros, module.TransportStep
    x_shape = (net.shape[0], net.shape[1] + 1)
    state = {"on": False}

    def zeros(shape, *a, **k):
        if state["on"] and tuple(np.atleast_1d(shape))[:2] == x_shape and not any(e[0] == "x_gross" for e in events):
            events.append(("x_gross", tracker.current))
        return real_zeros(shape, *a, **k)

    def result(*a, **k):
        events.append(("result", tracker.current))
        return real_result(*a, **k)

    module.cuda_sparse_step(net, *args, **kwargs)  # warm (binding, kernels, self-check)
    cp.cuda.Device().synchronize()
    monkeypatch.setattr(cp, "zeros", zeros)
    monkeypatch.setattr(module, "TransportStep", result)
    with tracker.hook():
        state["on"] = True
        step = module.cuda_sparse_step(net, *args, **kwargs)
        cp.cuda.Device().synchronize()
        state["on"] = False
    monkeypatch.setattr(cp, "zeros", real_zeros)
    monkeypatch.setattr(module, "TransportStep", real_result)
    return events, step


@needs_gpu
@pytest.mark.parametrize("n_substeps", [1, 2])
@pytest.mark.parametrize("stored", [False, True])
def test_live_bytes_at_first_face_allocation_and_result(b46, b52, monkeypatch, n_substeps, stored):
    """Exact live-byte differences (each module runs on a phase bound to ITS OWN binding):
      first face allocation  B46 - B54 = I + 3D (cell scale, tolerance, balance) + R (compact remaining) + Mm
                                         - 512 k   (k = the 0-d flags precomputed by B54 plus its 0-d balance maximum)
                             B52 - B54 = 2D (tolerance, balance) + R + Mm - 512 k
      result construction    B46 - B54 = 3D + 2R + I + faces + [nnz: R + 2 Mm + r512(nnz - 1)]
                             B52 - B54 = D (balance) + Mm
    with R = r512(n_active nc 8), D = r512(n nc 8), I = r512(n), Mm = r512(8 nnz), faces = the two gathered arrays;
    k = 7 + (4 + [nnz > 1] if nnz > 0 else 0) in the CW-B63 tree (5 in B54; + the precomputed combined-pool flag in
    B59; + the precomputed non-outlet export flag in B63)."""
    nc, nb = 6, 8
    net, (mobile, pickup, v, rate, settle), _phase = h.build_case("masked", nc, nb)

    def own_state(module):
        phase = module.empty_cuda_sparse_phase_state(net, nc, nb)
        mob = mobile
        if stored:
            warm = module.cuda_sparse_step(net, mobile, phase, pickup, v, rate, settle, 1.0)
            mob, phase = warm.mobile_remaining_kg, warm.phase_after
            assert phase.nnz > 0
        return mob, phase

    runs = {}
    for label, module in (("b54", cuda_csr), ("b46", b46), ("b52", b52)):
        mob, phase = own_state(module)
        runs[label] = _live_at_faces_and_result(module, monkeypatch, net, mob, phase, pickup, v, rate, settle, 1.0,
                                                n_substeps=n_substeps)
    values = {k: h.value_rec(step) for k, (_e, step) in runs.items()}
    assert values["b54"] == values["b46"] == values["b52"]
    events = {k: dict(e) for k, (e, _s) in runs.items()}
    for k, launched in events.items():
        assert list(launched) == ["x_gross", "result"], (k, launched)
    binding = cuda_csr.bind_network(net)
    n, n_active = binding.n_cells, binding.n_active
    R, D, I = _r512(n_active * nc * 8), _r512(n * nc * 8), _r512(n)
    nnz = int(runs["b54"][1].nnz_after)
    Mm = _r512(8 * nnz)
    # CW-B59: the module under test also retains its precomputed combined-pool flag (one 0-d block) at the first
    # face allocation; the original B54 count was 5 + ... (archived). Result expectations are unchanged.
    # CW-B63: + the precomputed non-outlet export flag (one more 0-d block); B59 used 6 (archived). Result unchanged.
    k = 7 + ((4 + (1 if nnz > 1 else 0)) if nnz else 0)
    faces = _r512(int(net.cells_x.size) * nc * 8) + _r512(int(net.cells_y.size) * nc * 8)
    order = _r512(nnz - 1) if nnz > 1 else 0
    got = {(o, p): events[o][p] - events["b54"][p] for o in ("b46", "b52") for p in ("x_gross", "result")}
    want = {("b46", "x_gross"): I + 3 * D + R + Mm - 512 * k,
            ("b52", "x_gross"): 2 * D + R + Mm - 512 * k,
            ("b46", "result"): 3 * D + 2 * R + I + faces + ((R + 2 * Mm + order) if nnz else 0),
            ("b52", "result"): D + Mm}
    print(f"live-byte differences (R={R}, D={D}, I={I}, nnz={nnz}, k={k}):", got)
    assert got == want, (got, want)
