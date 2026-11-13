"""R2a: refusals, tamper (including tampered PHYSICAL state with recomputed hashes), admission order, late-failure atomicity, CPU-only, historical-codec refusal and the mechanical drift guard."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import shutil
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
from case import AVAIL, IDENT, build, run, snapshot

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_resume as fr
from maple_syrup import pond_checkpoint as pc
from maple_syrup.pond_event import PondEvent

REFUSE = (fc.PondCheckpointError, fr.FloatingResumeError)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    base = tmp_path_factory.mktemp("saved")
    ev, st, _vert = build(base / "a")
    (base / "a").mkdir(exist_ok=True)
    st = run(ev, st, 8)
    ck = base / "ck"
    fc.save_floating_pond_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    return base, ck, ev, st


def fresh(tmp_path, name="t"):
    d = tmp_path / name
    d.mkdir()
    return build(d)


def copy(saved, tmp_path):
    dst = tmp_path / "ck"
    shutil.copytree(saved[1], dst)
    return dst


def retamper(ck, mutate):
    """Mutate the companion arrays, then RECOMPUTE the payload hash and reseal the manifest: only the full physical validation can catch it."""
    path = ck / "pond_continuation.npz"
    with np.load(path) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    mutate(arrays)
    np.savez(path, **arrays)
    man = json.loads((ck / fc.MANIFEST_NAME).read_text())
    man["files"]["pond_continuation.npz"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ck / fc.MANIFEST_NAME).write_text(fc._json_dumps(fc._seal(man)))


def untouched(ev, vert, registry0, stats0):
    assert len(vert._registry) == registry0 and vert.stats == stats0 and vert._provisional is None


def test_wrong_identity_schema_keys_and_hash_are_refused_before_payload(saved, tmp_path):
    ev2, _s, v2 = fresh(tmp_path)
    reg0 = len(v2._registry)  # the original case's own initial state is registered
    with pytest.raises(REFUSE, match="identity"):
        fc.load_floating_pond_checkpoint(saved[1], ev2, {"case": "other"}, available_bytes_fn=AVAIL)
    ck = copy(saved, tmp_path)
    man = json.loads((ck / fc.MANIFEST_NAME).read_text())
    man["extra"] = 1
    (ck / fc.MANIFEST_NAME).write_text(json.dumps(man))
    with pytest.raises(REFUSE, match="keys mismatch"):
        fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL)
    ck2 = tmp_path / "ck2"
    shutil.copytree(saved[1], ck2)
    man = json.loads((ck2 / fc.MANIFEST_NAME).read_text())
    man["schema"] = "maple-syrup-pond-checkpoint/1"
    (ck2 / fc.MANIFEST_NAME).write_text(json.dumps(man))
    with pytest.raises(REFUSE, match="not a floating pond checkpoint schema"):
        fc.load_floating_pond_checkpoint(ck2, ev2, IDENT, available_bytes_fn=AVAIL)
    ck3 = tmp_path / "ck3"
    shutil.copytree(saved[1], ck3)
    data = bytearray((ck3 / "pond_continuation.npz").read_bytes())
    data[len(data) // 2] ^= 0xFF
    (ck3 / "pond_continuation.npz").write_bytes(bytes(data))
    with pytest.raises(REFUSE, match="hash mismatch"):
        fc.load_floating_pond_checkpoint(ck3, ev2, IDENT, available_bytes_fn=AVAIL)
    assert v2._provisional is None and len(v2._registry) == reg0


def test_tampered_held_map_with_recomputed_hashes_is_refused_by_the_template_check(saved, tmp_path):
    ev2, _s, v2 = fresh(tmp_path)
    ck = copy(saved, tmp_path)
    retamper(ck, lambda a: a["fl_held_fc"].__setitem__((0, 0), a["fl_held_fc"][0, 0] * 1.0001))
    with pytest.raises(REFUSE, match="HELD template"):
        fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL)
    assert v2._provisional is None


@pytest.mark.parametrize("which", ["fl_water", "hyd_depth", "fl_depth"])
def test_tampered_physical_state_with_recomputed_hashes_fails_full_validation_and_leaves_the_target_untouched(saved, tmp_path, which):
    ev2, _s, v2 = fresh(tmp_path)
    reg0, stats0 = len(v2._registry), dataclass_copy(v2.stats)

    def mutate(a):
        if which == "fl_water":
            a["fl_water"][0, 0] = a["fl_water"][0, 0] * 5.0 + 1e-3  # more than the CURRENT capacity of that slab
        elif which == "hyd_depth":
            a["hyd_depth"][1, 1] += 1e-4  # surface water out of balance with the native bed / ledgers
        else:
            a["fl_depth"][0] += 1e-3  # a canonical depth that no longer matches the native bed

    ck = copy(saved, tmp_path)
    retamper(ck, mutate)
    with pytest.raises(REFUSE):
        fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL)
    untouched(ev2, v2, reg0, stats0)
    # the same target then loads the GOOD checkpoint and continues exactly (a failed load left no trace)
    got = fc.load_floating_pond_checkpoint(saved[1], ev2, IDENT, available_bytes_fn=AVAIL)
    ref = run(saved[2], saved[3], 3)
    cont = run(ev2, got, 3)
    a, b = snapshot(saved[2], ref, saved[2].vertical), snapshot(ev2, cont, v2)
    assert set(a) == set(b) and all(a[k].tobytes() == b[k].tobytes() for k in a)


def dataclass_copy(stats):
    import dataclasses

    return dataclasses.replace(stats)


def test_backend_and_cuda_and_historical_codec_refusals(saved, tmp_path):
    pytest.importorskip("numba")
    evn, _s, _v = build(tmp_path / "nb", backend="numba") if (tmp_path / "nb").mkdir() is None else None
    with pytest.raises(REFUSE, match="backend|identity|binding"):
        fc.load_floating_pond_checkpoint(saved[1], evn, IDENT, available_bytes_fn=AVAIL)
    # CUDA: refused explicitly before any read or transfer
    stub = object.__new__(PondEvent)
    from maple_syrup import floating_pond as fp

    stub.vertical = object.__new__(fp.FloatingPondVertical)
    stub.vertical.backend = "cuda"
    stub.xp, stub.control = np, SimpleNamespace(backend="cuda")
    with pytest.raises(fc.FloatingCheckpointError, match="CUDA"):
        fc.save_floating_pond_checkpoint(tmp_path / "never", None, stub, IDENT, available_bytes_fn=AVAIL)
    with pytest.raises(fc.FloatingCheckpointError, match="CUDA"):
        fc.load_floating_pond_checkpoint(tmp_path / "never", stub, IDENT, available_bytes_fn=AVAIL)
    assert not (tmp_path / "never").exists()
    # the historical codec refuses a floating event (event.column is None) before any file
    ev, st, _v2 = saved[2], saved[3], None
    with pytest.raises(pc.PondCheckpointError, match="hydraulic state is invalid"):
        pc.save_pond_checkpoint(tmp_path / "old", st, ev, IDENT)
    assert not (tmp_path / "old").exists() and not list(tmp_path.glob(".old.partial-*"))


def test_admission_order_need0_need_minus_1_exact_and_reserve_once(saved, tmp_path, monkeypatch):
    ev2, _s, _v2 = fresh(tmp_path)
    called = []
    real_manifest = fc._read_manifest
    monkeypatch.setattr(fc, "_read_manifest", lambda p: (called.append("manifest"), real_manifest(p))[1])
    with pytest.raises(fp_memory()):
        fc.load_floating_pond_checkpoint(saved[1], ev2, IDENT, available_bytes_fn=lambda: 0)
    assert not called  # need 0: refused before the manifest was read
    reads = []

    def reader_with(limit_second):
        def f():
            reads.append(1)
            return (1 << 40) if len(reads) == 1 else limit_second
        return f

    monkeypatch.setattr(fc.np, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a payload array was loaded before the composed admission")))
    reads.clear()
    with pytest.raises(fp_memory()) as info:
        fc.load_floating_pond_checkpoint(saved[1], ev2, IDENT, available_bytes_fn=reader_with(0))
    need = int(str(info.value).split("needs an estimated ")[1].split(" B")[0])
    monkeypatch.undo()
    for avail, ok in ((need - 1, False), (need, True)):
        ev3, _s3, _v3 = fresh(tmp_path, f"t{avail}")
        reads.clear()
        if ok:
            got = fc.load_floating_pond_checkpoint(saved[1], ev3, IDENT, available_bytes_fn=reader_with(avail))
            assert got is not None
        else:
            with pytest.raises(fp_memory()):
                fc.load_floating_pond_checkpoint(saved[1], ev3, IDENT, available_bytes_fn=reader_with(avail))
    ev4, _s4, _v4 = fresh(tmp_path, "r")
    reads.clear()
    with pytest.raises(fp_memory()):
        fc.load_floating_pond_checkpoint(saved[1], ev4, IDENT, available_bytes_fn=reader_with(need + 6), reserve_bytes=7)  # the reserve is counted ONCE
    ev5, _s5, _v5 = fresh(tmp_path, "r2")
    reads.clear()
    assert fc.load_floating_pond_checkpoint(saved[1], ev5, IDENT, available_bytes_fn=reader_with(need + 7), reserve_bytes=7) is not None


def fp_memory():
    from maple_syrup import native_floating_soil as nfs

    return nfs.NativeFloatingMemoryError


def test_save_admission_and_pending_windowed_refusal_precede_all_work(saved, tmp_path, monkeypatch):
    ev, st = saved[2], saved[3]
    with pytest.raises(fp_memory()):
        fc.save_floating_pond_checkpoint(tmp_path / "s0", st, ev, IDENT, available_bytes_fn=lambda: 0)
    assert not (tmp_path / "s0").exists() and not list(tmp_path.glob(".s0.partial-*"))
    with pytest.raises(fc.FloatingCheckpointError, match="already exists"):
        fc.save_floating_pond_checkpoint(saved[1], st, ev, IDENT, available_bytes_fn=AVAIL)  # no overwrite
    # a pending Windowed journal is refused first (the native predicate; a pending journal cannot be forged here, so the call order is pinned)
    order = []
    monkeypatch.setattr(fc, "is_windowed_bed", lambda b: True)
    monkeypatch.setattr(fc, "_require_published", lambda b: (order.append("published-check"), (_ for _ in ()).throw(fc.PondCheckpointError("pending journal")))[1])
    with pytest.raises(fc.PondCheckpointError, match="pending journal"):
        fc.save_floating_pond_checkpoint(tmp_path / "s1", st, ev, IDENT, available_bytes_fn=AVAIL)
    assert order == ["published-check"] and not (tmp_path / "s1").exists()


def _normalised(fn, drop_hydraulic):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    body = tree.body[1:] if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) else tree.body
    out = []
    for node in body:
        is_block = isinstance(node, ast.Try) and any(isinstance(n, ast.Attribute) and n.attr in ("build_hydraulics",) or isinstance(n, ast.Name) and n.id == "build_pond_hydraulics" for n in ast.walk(node))
        if is_block:
            assert drop_hydraulic
            continue
        out.append(ast.dump(node))
    return tree.args and ast.dump(tree.args), out


def test_validator_copy_is_the_historical_validator_except_the_declared_hydraulic_block():
    old_args, old = _normalised(pc._validate_state, True)
    new_args, new = _normalised(fc._validate_floating_state, True)
    assert old_args == new_args and old == new and len(old) > 20  # mechanical AST guard: nothing else drifted
    # and the declared replacement really is the floating one
    src = inspect.getsource(fc._validate_floating_state)
    assert "event.vertical.build_hydraulics" in src and "build_pond_hydraulics(" not in src


# ---- R2a correction01: verified-input integrity, private one-shot, public-bridge admission ------------------------------------------------------------------------------------------------------------
def capture_inputs(saved, tmp_path, name):
    """A GENUINE unused `VerifiedFloatingInputs` from an actual file verification: the loader's real call to the preparation is intercepted and aborted (nothing is prepared or registered)."""
    ev, _s, v = fresh(tmp_path, name)
    grabbed = {}
    real = fr.prepare_from_verified_inputs

    def spy(event, inputs, **kw):
        grabbed["inputs"] = inputs
        raise RuntimeError("abort before prepare")

    fr.prepare_from_verified_inputs = spy
    try:
        with pytest.raises(fc.FloatingCheckpointError, match="abort before prepare|failed"):
            try:
                fc.load_floating_pond_checkpoint(saved[1], ev, IDENT, available_bytes_fn=AVAIL)
            except RuntimeError as exc:
                raise fc.FloatingCheckpointError(str(exc)) from None
    finally:
        fr.prepare_from_verified_inputs = real
    return ev, v, grabbed["inputs"]


def test_rebinding_a_minted_input_after_verification_is_refused(saved, tmp_path):
    from types import MappingProxyType

    ev, v, inputs = capture_inputs(saved, tmp_path, "rb")
    reg0, stats0 = len(v._registry), dataclass_copy(v.stats)
    w = np.array(inputs.current["water_m"], copy=True)
    w[0, 0] += 2.0 ** -30
    w[1, 0] -= 2.0 ** -30  # the cell total is bit-exact: only the integrity digest can see it
    w.flags.writeable = False
    cur = dict(inputs.current)
    cur["water_m"] = w
    object.__setattr__(inputs, "current", MappingProxyType(cur))
    with pytest.raises(fc.FloatingCheckpointError, match="integrity|rebound|altered"):
        fr.prepare_from_verified_inputs(ev, inputs, available_bytes_fn=AVAIL)
    untouched(ev, v, reg0, stats0)
    ev2, _v2, inputs2 = capture_inputs(saved, tmp_path, "rb2")
    object.__setattr__(inputs2, "soil_t_s", float(inputs2.soil_t_s) + 1.0)
    with pytest.raises(fc.FloatingCheckpointError, match="integrity|rebound|altered"):
        fr.prepare_from_verified_inputs(ev2, inputs2, available_bytes_fn=AVAIL)
    ev3, _v3, inputs3 = capture_inputs(saved, tmp_path, "rb3")
    object.__setattr__(inputs3, "source_witness", "0" * 64)
    with pytest.raises(fc.FloatingCheckpointError, match="integrity|rebound|altered"):
        fr.prepare_from_verified_inputs(ev3, inputs3, available_bytes_fn=AVAIL)
    ev4, _v4, inputs4 = capture_inputs(saved, tmp_path, "rb4")
    object.__setattr__(inputs4, "held", dict(inputs4.held))  # a plain dict instead of the sealed mapping
    with pytest.raises(fc.FloatingCheckpointError, match="containers"):
        fr.prepare_from_verified_inputs(ev4, inputs4, available_bytes_fn=AVAIL)


def test_one_shot_state_is_private_and_cannot_be_resurrected(saved, tmp_path):
    ev, _v, inputs = capture_inputs(saved, tmp_path, "os")
    prepared, state = fr.prepare_from_verified_inputs(ev, inputs, available_bytes_fn=AVAIL)
    assert not hasattr(inputs, "consumed")  # no public caller-resettable flag
    with pytest.raises(fc.FloatingCheckpointError, match="already consumed"):
        fc.require_inputs(inputs)
    object.__setattr__(inputs, "consumed", [False])  # an attempted resurrection has no effect
    with pytest.raises(fc.FloatingCheckpointError, match="already consumed"):
        fr.prepare_from_verified_inputs(ev, inputs, available_bytes_fn=AVAIL)
    assert prepared is not None and state is not None
    # a refusal BEFORE the consumption leaves the inputs reusable, and a failed load can read the file again with a NEW capability
    ev2, _v2, inputs2 = capture_inputs(saved, tmp_path, "os2")
    with pytest.raises(fp_memory()):
        fr.prepare_from_verified_inputs(ev2, inputs2, available_bytes_fn=lambda: 0)
    fc.require_inputs(inputs2)  # still genuine, intact and unconsumed
    got = fc.load_floating_pond_checkpoint(saved[1], ev2, IDENT, available_bytes_fn=AVAIL)
    assert got is not None


def test_public_bridge_admits_from_its_own_reader_before_any_copy_hash_or_view(saved, tmp_path, monkeypatch):
    ev, v, inputs = capture_inputs(saved, tmp_path, "ad")
    calls = []
    for mod, name in ((fr, "_seal"), (fr, "_source_pair"), (fr, "_mint_capture_from_inputs"), (fr.fsl, "floating_geometry_view"), (fr.fsl, "floating_continued_state")):
        real = getattr(mod, name)
        monkeypatch.setattr(mod, name, lambda *a, _real=real, _n=name, **k: (calls.append(_n), _real(*a, **k))[1])
    reads = []

    def reader(value):
        def f():
            reads.append(1)
            return value
        return f

    with pytest.raises(fp_memory()) as info:
        fr.prepare_from_verified_inputs(ev, inputs, available_bytes_fn=reader(0))
    assert reads == [1] and calls == []  # refused by the bridge's OWN reader before any capture, source hash, view or conversion (D == D0 and D changed alike)
    need = int(str(info.value).split("needs an estimated ")[1].split(" B")[0])
    assert need >= fr.prepare_model_bytes(v)
    with pytest.raises(fp_memory()):
        fr.prepare_from_verified_inputs(ev, inputs, available_bytes_fn=reader(need - 1))
    assert calls == []
    with pytest.raises(fp_memory()):
        fr.prepare_from_verified_inputs(ev, inputs, available_bytes_fn=reader(need + 6), reserve_bytes=7)  # reserve counted ONCE
    assert calls == []
    prepared, _state = fr.prepare_from_verified_inputs(ev, inputs, available_bytes_fn=reader(need + 7), reserve_bytes=7)
    assert prepared is not None and "_mint_capture_from_inputs" in calls
