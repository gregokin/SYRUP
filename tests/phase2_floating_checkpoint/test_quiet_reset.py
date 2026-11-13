"""S4-1 correction01: the fresh quiet monitor of a wet state with accepted steps is justified ONLY by the producer's own retained quiet-reset receipt (`SedimentLedger.retained_quiet_reset_step`, set only by the
`settle_retaining_water` wake branch) at the CURRENT accepted step. Root's reproduction (a fault-injected fresh monitor one ordinary rain-0 step after the genuine wake) is refused; the genuine immediate wake and the
genuine later ordinary state round-trip exactly; stale/future/negative/malformed/wrong receipts with recomputed seals are refused; /1-/4 files keep their original semantics."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from case import AVAIL, IDENT, build, run, snapshot
from test_terminal import (  # noqa: F401  (the lake fixtures and the backend parametrisation)
    backend,
    lake,
    same,
    settled,
)

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_resume as fr
from maple_syrup.pond_event import PondQuiet

REFUSE = (fc.PondCheckpointError, fr.FloatingResumeError)
DATA = Path(__file__).resolve().parent / "data"


def _module(name, text_file):
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(text_file)
    sys.modules[name] = mod
    exec(compile(Path(text_file).read_text(), str(text_file), "exec"), mod.__dict__)  # noqa: S102 - a verbatim copy of an accepted repository module
    return mod


@pytest.mark.parametrize("windowed", [False, True])
def test_root_reproduction_a_fresh_monitor_one_step_after_the_wake_is_refused(tmp_path, backend, windowed):  # noqa: F811
    ev, st, vert = lake(tmp_path / "a", backend, windowed)
    wake = settled(ev, st)
    n0 = wake.counters.n_accepted_steps
    assert wake.sediment.retained_quiet_reset_step == n0 and wake.quiet == PondQuiet()  # the genuine wake: fresh monitor + the producer receipt at THIS step
    later = ev.step(wake, 0.0, 0.2).state  # the next accepted rain-0 step has its own valid (non-fresh) monitor
    assert later.counters.n_accepted_steps == n0 + 1 and later.quiet != PondQuiet() and later.sediment.retained_quiet_reset_step == n0  # an ordinary step PRESERVES the old receipt
    forged = dataclasses.replace(later, quiet=PondQuiet())  # fault injection ONLY of the monitor; physical arrays/ledgers unchanged
    reg0, stats0 = len(vert._registry), dataclasses.replace(vert.stats)
    with pytest.raises(REFUSE, match="quiet monitor time differs"):
        fc.save_floating_pond_checkpoint(tmp_path / "bad", forged, ev, IDENT, available_bytes_fn=AVAIL)
    assert not (tmp_path / "bad").exists() and len(vert._registry) == reg0 and vert.stats == stats0
    # the genuine immediate wake and the genuine later ordinary state both save and load exactly
    for name, good in (("wake", wake), ("later", later)):
        ck = tmp_path / name
        fc.save_floating_pond_checkpoint(ck, good, ev, IDENT, available_bytes_fn=AVAIL)
        ev2, _s0, vert2 = lake(tmp_path / f"t_{name}", backend, windowed)
        got = fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL)
        assert got.sediment.retained_quiet_reset_step == n0
        same(snapshot(ev2, got, vert2), snapshot(ev, good, vert))
        same(snapshot(ev2, ev2.step(got, 0.0, 0.2).state, vert2), snapshot(ev, ev.step(good, 0.0, 0.2).state, vert))


def test_tampered_quiet_reset_receipt_with_recomputed_seal_is_refused(tmp_path):
    ev, st, _vert = lake(tmp_path / "a", "numpy", False)
    wake = settled(ev, st)
    n0 = wake.counters.n_accepted_steps
    ck0 = tmp_path / "ck"
    fc.save_floating_pond_checkpoint(ck0, wake, ev, IDENT, available_bytes_fn=AVAIL)
    later = ev.step(wake, 0.0, 0.2).state
    ck1 = tmp_path / "ck1"
    fc.save_floating_pond_checkpoint(ck1, later, ev, IDENT, available_bytes_fn=AVAIL)
    evt, _s, vt = lake(tmp_path / "t", "numpy", False)
    reg0 = len(vt._registry)

    def tamper(src, name, edit):
        d = tmp_path / name
        shutil.copytree(src, d)
        man = json.loads((d / fc.MANIFEST_NAME).read_text())
        edit(man)
        (d / fc.MANIFEST_NAME).write_text(fc._json_dumps(fc._seal(man)))
        return d

    cases = [("none", ck0, lambda m: m["retained_quiet_reset"].update(step=None)), ("stale", ck0, lambda m: m["retained_quiet_reset"].update(step=n0 - 1)),
             ("future", ck0, lambda m: m["retained_quiet_reset"].update(step=n0 + 5)), ("neg", ck0, lambda m: m["retained_quiet_reset"].update(step=-1)),
             ("str", ck0, lambda m: m["retained_quiet_reset"].update(step=str(n0))), ("bool", ck0, lambda m: m["retained_quiet_reset"].update(step=True)),
             ("extra", ck0, lambda m: m["retained_quiet_reset"].update(extra=1)), ("drop", ck0, lambda m: m.pop("retained_quiet_reset")),
             ("later_future", ck1, lambda m: m["retained_quiet_reset"].update(step=n0 + 9))]
    for name, src, edit in cases:
        with pytest.raises(REFUSE):
            fc.load_floating_pond_checkpoint(tamper(src, name, edit), evt, IDENT, available_bytes_fn=AVAIL)
        assert len(vt._registry) == reg0 and vt._provisional is None
    # a later ordinary state with the receipt removed is still a VALID file (the monitor is not fresh; the receipt is informational there)
    ok = tamper(ck1, "later_none", lambda m: m["retained_quiet_reset"].update(step=None))
    got = fc.load_floating_pond_checkpoint(ok, evt, IDENT, available_bytes_fn=AVAIL)
    assert got.sediment.retained_quiet_reset_step is None


def test_second_real_settlements_record_their_own_receipt(tmp_path):
    ev, st, vert = lake(tmp_path / "a", "numpy", False, splash=True, discharge=1e-6)
    one = settled(ev, st)
    for _ in range(8):
        one = ev.step(one, 100 / 3.6e6, 0.2).state
    q = ev.evolve(one, 600.0, stop_when_quiet=True)
    two = ev.settle_retaining_water(q.state).state  # the SECOND non-empty settlement wakes again: a NEW receipt at the new step
    assert two.sediment.n_terminal_native_calls == 2 and two.sediment.retained_quiet_reset_step == two.counters.n_accepted_steps and two.quiet == PondQuiet()
    ck = tmp_path / "ck"
    fc.save_floating_pond_checkpoint(ck, two, ev, IDENT, available_bytes_fn=AVAIL)
    ev2, _s0, vert2 = lake(tmp_path / "b", "numpy", False, splash=True, discharge=1e-6)
    got = fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL)
    same(snapshot(ev2, got, vert2), snapshot(ev, two, vert))
    # an EMPTY real settlement after a recession does not wake (no reset): the quiet certificate and the previous receipt are untouched
    ev3, st3, _vert3 = lake(tmp_path / "c", "numpy", False)
    w3 = settled(ev3, st3)
    q3 = ev3.evolve(w3, 120.0, stop_when_quiet=True)
    done = ev3.settle_retaining_water(q3.state)
    assert done.complete and done.state.sediment.n_terminal_native_calls == 1 and done.state.sediment.retained_quiet_reset_step == w3.counters.n_accepted_steps
    assert np.asarray(done.state.quiet.ready).item() is True


def test_old_s4_1_v3_v4_files_keep_their_semantics_and_the_overbroad_wake_is_not_rebound(tmp_path):
    """/3 files written by the S4-1 module (kept verbatim as test data) for a state WITHOUT a fresh monitor still load; a /3 file of an immediate wake (which only the overbroad S4-1 rule accepted) is refused: no
    automatic migration and no receipt is invented for it."""
    old = _module("floating_checkpoint_s4_1", DATA / "floating_checkpoint_s4_1_source.txt")
    ev, st, vert = build(tmp_path / "a")
    (tmp_path / "a").mkdir(exist_ok=True)
    st = run(ev, st, 6)
    ck = tmp_path / "ck3"
    old.save_floating_pond_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    man = json.loads((ck / old.MANIFEST_NAME).read_text())
    assert man["schema"] == fc.SCHEMA_V2 and "retained_quiet_reset" not in man
    ev2, _s0, vert2 = build(tmp_path / "b")
    got = fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL)
    assert got.sediment.retained_quiet_reset_step is None
    same(snapshot(ev2, got, vert2), snapshot(ev, st, vert))
    evl, stl, _v = lake(tmp_path / "l", "numpy", False)
    wake = settled(evl, stl)
    ck4 = tmp_path / "ck3w"
    old.save_floating_pond_checkpoint(ck4, wake, evl, IDENT, available_bytes_fn=AVAIL)  # the S4-1 module (overbroad rule) wrote it
    evt, _s, _vt = lake(tmp_path / "t", "numpy", False)
    with pytest.raises(REFUSE, match="quiet monitor time differs"):
        fc.load_floating_pond_checkpoint(ck4, evt, IDENT, available_bytes_fn=AVAIL)
