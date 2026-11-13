"""S4-1: the producer-maintained terminal native-call count (`SedimentLedger.n_terminal_native_calls`, updated ONLY by `_settle_terminal`) and its strict floating file profile (schemas /3 dense, /4 windowed). Real retained
terminal deposits on a closed thin lake (dense and published windowed, NumPy and Numba), a genuine second settlement after an actual recession, an empty settlement (0 calls), tampered counts with recomputed manifest seals,
the old /1 files (written by the pre-S4-1 module), the mechanical drift guard of the scalar validator and the byte-identity pin of the historical codec."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import shutil
import sys
import textwrap
from pathlib import Path

import fpond_builders as fb
import numpy as np
import pytest
from case import AVAIL, IDENT, snapshot
from fpond_builders import ph

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_resume as fr
from maple_syrup import pond_checkpoint as pc
from maple_syrup.pond_event import PondCompletionPolicy

REFUSE = (fc.PondCheckpointError, fr.FloatingResumeError)
PC_SHA256 = "94cd2339566646ba44f2015dad13a67194c26a8dd3cc85b56916f62d2b4a704f"  # the historical codec, byte-identical to the accepted file


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def lake(tmp, backend, windowed, *, splash=False, discharge=None):
    z = ph.basin() - 0.08  # thin closed profile
    depth = np.maximum(0.135 - z, 0.0)
    mob = np.zeros((3, 5, 6))
    mob[1, 2, 2] = 1.0e-1
    pol = PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True) if discharge is None else PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-6, max_unit_discharge_m2_s=discharge, deposit_mobile_at_flow_end=True)
    kw = {"surface": z, "depth": depth, "mobile": mob, "closed": True, "bottom": "closed", "saturation": 0.999999, "backend": backend, "splash": splash, "rain_mm_h": 0.0, "end_s": 2.0, "max_dt_s": 1.0,
          "policy": pol}
    if windowed:
        ev, st, vert, *_r = fb.make_pair(tmp, **kw)[1]
        return ev, st, vert
    ev, st, _b, _c, vert, _o = fb.make(**kw)
    return ev, st, vert


def settled(ev, st):
    q = ev.evolve(st, 60.0, stop_when_quiet=True)
    assert q.quiet_ready
    h = ev.settle_retaining_water(q.state)
    assert h.status == "requires_recession"
    return h.state


def same(a, b):
    assert set(a) == set(b)
    for k in a:
        assert a[k].tobytes() == b[k].tobytes(), k


@pytest.mark.parametrize("windowed", [False, True])
def test_retained_terminal_deposit_saves_loads_and_continues_bit_exactly(tmp_path, backend, windowed):
    ev, st, vert = lake(tmp_path / "a", backend, windowed)
    st = settled(ev, st)
    assert st.sediment.n_terminal_native_calls == 1 and st.sediment.n_native_calls == 2 * st.counters.n_accepted_steps + 1  # the producer maintained the count; the historical rule would refuse this state
    assert float(np.asarray(st.sediment.deposited_terminal).sum()) > 0.0
    ck = tmp_path / "ck"
    fc.save_floating_pond_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    man = json.loads((ck / fc.MANIFEST_NAME).read_text())
    assert man["schema"] == (fc.WINDOWED_SCHEMA_V3 if windowed else fc.SCHEMA_V3) and man["terminal_exchange"] == {"n_terminal_native_calls": 1}
    assert man["retained_quiet_reset"] == {"step": st.counters.n_accepted_steps}  # the producer receipt of THIS reset
    ref = ev.step(st, 0.0, 0.2).state  # a real rain-0 recession step on the settled state
    ev2, _s0, vert2 = lake(tmp_path / "b", backend, windowed)
    got = fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL)
    assert got.sediment.n_terminal_native_calls == 1
    same(snapshot(ev2, got, vert2), snapshot(ev, st, vert))
    cont = ev2.step(got, 0.0, 0.2).state
    same(snapshot(ev2, cont, vert2), snapshot(ev, ref, vert))
    with pytest.raises(pc.PondCheckpointError, match="hydraulic state is invalid|native call counter"):
        pc.save_pond_checkpoint(tmp_path / "old", st, ev, IDENT)  # the historical codec still refuses (and wrote nothing)
    assert not (tmp_path / "old").exists()


def test_second_settlement_after_a_real_recession_with_an_empty_pool_is_zero_calls(tmp_path):
    ev, st, vert = lake(tmp_path / "a", "numpy", False)
    st = settled(ev, st)
    q = ev.evolve(st, 120.0, stop_when_quiet=True)  # the actual recession re-certifies a quiet hold
    assert q.quiet_ready
    done = ev.settle_retaining_water(q.state)  # the pool is empty now: the settlement makes NO native call
    assert done.complete and done.state.sediment.n_terminal_native_calls == 1 and done.state.sediment.n_native_calls == st.sediment.n_native_calls + 2 * (q.state.counters.n_accepted_steps - st.counters.n_accepted_steps)
    ck = tmp_path / "ck"
    fc.save_floating_pond_checkpoint(ck, done.state, ev, IDENT, available_bytes_fn=AVAIL)
    ev2, _s0, vert2 = lake(tmp_path / "b", "numpy", False)
    got = fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL)
    same(snapshot(ev2, got, vert2), snapshot(ev, done.state, vert))


def test_tampered_terminal_count_with_recomputed_seal_and_scalars_is_refused(tmp_path):
    ev, st, _vert = lake(tmp_path / "a", "numpy", False)
    st = settled(ev, st)
    ck0 = tmp_path / "ck"
    fc.save_floating_pond_checkpoint(ck0, st, ev, IDENT, available_bytes_fn=AVAIL)
    evt, _s, vt = lake(tmp_path / "t", "numpy", False)
    reg0 = len(vt._registry)

    def tamper(name, edit):
        d = tmp_path / name
        shutil.copytree(ck0, d)
        man = json.loads((d / fc.MANIFEST_NAME).read_text())
        edit(man)
        (d / fc.MANIFEST_NAME).write_text(fc._json_dumps(fc._seal(man)))
        return d

    for name, edit in (("zero", lambda m: m["terminal_exchange"].update(n_terminal_native_calls=0)), ("two", lambda m: m["terminal_exchange"].update(n_terminal_native_calls=2)),
                       ("neg", lambda m: m["terminal_exchange"].update(n_terminal_native_calls=-1)), ("str", lambda m: m["terminal_exchange"].update(n_terminal_native_calls="1")),
                       ("calls", lambda m: m["payload"]["scalars"]["sediment"].update(n_native_calls=m["payload"]["scalars"]["sediment"]["n_native_calls"] + 1)),
                       ("drop", lambda m: m.pop("terminal_exchange"))):
        with pytest.raises(REFUSE):
            fc.load_floating_pond_checkpoint(tamper(name, edit), evt, IDENT, available_bytes_fn=AVAIL)
        assert len(vt._registry) == reg0 and vt._provisional is None  # nothing registered; the target stays retryable
    got = fc.load_floating_pond_checkpoint(ck0, evt, IDENT, available_bytes_fn=AVAIL)  # the same target then loads the good file
    assert got.sediment.n_terminal_native_calls == 1


def test_old_floating_v1_files_are_still_read_with_their_original_semantics(tmp_path):
    """A GENUINE /1 file written by the pre-S4-1 module (loaded under another name) for a state with no terminal exchange loads unchanged; a /1 file cannot carry a terminal count and the new reader assigns 0 to it."""
    data = Path(__file__).resolve().parent / "data" / "floating_checkpoint_r2a_c1_source.txt"  # the accepted R2aC1 module text, kept verbatim as test data
    spec = importlib.util.spec_from_loader("floating_checkpoint_pre_s41", loader=None)
    old = importlib.util.module_from_spec(spec)
    old.__file__ = str(data)
    sys.modules["floating_checkpoint_pre_s41"] = old
    exec(compile(data.read_text(), str(data), "exec"), old.__dict__)  # noqa: S102 - a verbatim copy of an accepted repository module
    from case import build, run

    ev, st, vert = build(tmp_path / "a")
    (tmp_path / "a").mkdir(exist_ok=True)
    st = run(ev, st, 6)
    ck = tmp_path / "ck1"
    old.save_floating_pond_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    man = json.loads((ck / old.MANIFEST_NAME).read_text())
    assert man["schema"] == fc.SCHEMA and "terminal_exchange" not in man  # explicit recorded distinction: an old manifest does not pretend to hold the new field
    ev2, _s0, vert2 = build(tmp_path / "b")
    got = fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL)
    assert got.sediment.n_terminal_native_calls == 0
    same(snapshot(ev2, got, vert2), snapshot(ev, st, vert))
    # the ORIGINAL rule applies to a /1 file: a tampered call counter (with a recomputed seal) is refused
    bad = tmp_path / "ck1b"
    shutil.copytree(ck, bad)
    man = json.loads((bad / old.MANIFEST_NAME).read_text())
    man["payload"]["scalars"]["sediment"]["n_native_calls"] += 1
    (bad / old.MANIFEST_NAME).write_text(fc._json_dumps(fc._seal(man)))
    ev3, _s3, _v3 = build(tmp_path / "c")
    with pytest.raises(REFUSE):
        fc.load_floating_pond_checkpoint(bad, ev3, IDENT, available_bytes_fn=AVAIL)


def _scalar_nodes(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    body = tree.body[1:] if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) else tree.body
    out = []
    for node in body:
        if (isinstance(node, ast.ImportFrom) and node.module == "maple.core.backend" and node.level == 0
                and len(node.names) == 1 and node.names[0].name == "to_host" and node.names[0].asname is None):
            continue  # declared CUDA bridge: counted terminal per-class read; all validation math remains guarded
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        if names & {"base", "calls", "terminal", "fresh", "woken", "reset_step"} or "quiet monitor time differs from the accepted state" in ast.dump(node):
            continue  # the DECLARED differences: the native-call rule (count, ledger cross-check) and the retained-terminal fresh-quiet-monitor evidence rule
        out.append(ast.dump(node))
    return ast.dump(tree.args), out


def test_floating_scalar_validator_is_the_historical_one_except_the_declared_native_call_rule():
    old_args, old = _scalar_nodes(pc._validate_scalars)
    new_args, new = _scalar_nodes(fc._validate_scalars)
    assert old_args == new_args and old == new and len(old) > 8
    src = inspect.getsource(fc._validate_scalars)
    assert "n_terminal_native_calls" in src and "base + terminal" in src and "woken" in src
    assert src.count("from maple.core.backend import to_host") == 1 and "np.asarray(to_host(state.sediment.deposited_terminal))" in src


def test_historical_codec_source_and_ledger_default_are_unchanged():
    import dataclasses

    from maple_syrup.pond_event import SedimentLedger

    assert hashlib.sha256(Path(pc.__file__).read_bytes()).hexdigest() == PC_SHA256
    f = {x.name: x for x in dataclasses.fields(SedimentLedger)}
    assert f["n_terminal_native_calls"].default == 0 and f["retained_quiet_reset_step"].default is None and list(f)[-2:] == ["n_terminal_native_calls", "retained_quiet_reset_step"]  # additive, last, defaulted
    from maple_syrup.pond_event import PondCounters

    assert "n_terminal_native_calls" not in {x.name for x in dataclasses.fields(PondCounters)}  # PondCounters (the historical checkpoint's counter keys) untouched
    assert shutil is not None


def test_second_non_empty_terminal_settlement_after_a_real_recession_counts_two_and_round_trips(tmp_path):
    """GENUINE second terminal deposit: after the first retained settlement, real rain with splash creates a NEW mobile pool, the actual recession re-certifies a quiet hold (an explicit, looser quiet policy of the event (discharge 1e-6 m2/s and
    cumulative depth change 1e-6 m): the default 1e-10 m2/s did not re-certify within 2000 s of this closed lake) and the second settlement makes a second native call."""
    ev, st, vert = lake(tmp_path / "a", "numpy", False, splash=True, discharge=1e-6)
    st = settled(ev, st)
    assert st.sediment.n_terminal_native_calls == 1
    for _ in range(8):
        st = ev.step(st, 100 / 3.6e6, 0.2).state
    assert float(np.asarray(st.bed.water.mobile_mass_by_cell_class_kg).sum()) > 0.0  # a new actual mobile pool
    q = ev.evolve(st, 600.0, stop_when_quiet=True)
    assert q.quiet_ready
    h = ev.settle_retaining_water(q.state)
    two = h.state
    assert two.sediment.n_terminal_native_calls == 2 and two.sediment.n_native_calls == 2 * two.counters.n_accepted_steps + 2 * two.counters.n_splash_exchanges + 2
    ck = tmp_path / "ck"
    fc.save_floating_pond_checkpoint(ck, two, ev, IDENT, available_bytes_fn=AVAIL)
    ref = ev.step(two, 0.0, 0.2).state
    ev2, _s0, vert2 = lake(tmp_path / "b", "numpy", False, splash=True, discharge=1e-6)
    got = fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL)
    assert got.sediment.n_terminal_native_calls == 2
    same(snapshot(ev2, ev2.step(got, 0.0, 0.2).state, vert2), snapshot(ev, ref, vert))
