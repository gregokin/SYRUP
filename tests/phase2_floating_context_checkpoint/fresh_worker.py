"""`python fresh_worker.py save|load WORKDIR BACKEND WINDOWED K M`. `save`: original case (genuine bed/owner) -> K steps -> context container -> M more steps -> dump. `load`: a NEW interpreter with the original-case builder,
the native owner initializer and the Saxton-Rawls provider DISABLED builds the event from the static factory + verified context, loads and continues M steps -> dump."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "phase2_floating_checkpoint"), str(Path(__file__).resolve().parents[1] / "phase2_floating_pond")]
from case import IDENT, build, run, snapshot
from static_case import factory

from maple_syrup import floating_context_checkpoint as cc
from maple_syrup import floating_held_context as fh


def main(mode, workdir, backend, windowed, k, m):
    work = Path(workdir)
    ck = work / "ctx"
    avail = lambda: 1 << 40
    if mode == "save":
        (work / "save").mkdir(parents=True, exist_ok=True)
        ev, st, vert = build(work / "save", backend=backend, windowed=windowed == "1")
        st = run(ev, st, int(k))
        spec = fh.PreferentialSpec(np.full(vert.n_cells, 2.0e-3), 0.5, 1e-5, np.array([0.0, 1.0, 1.0]), (), None)
        cc.save_floating_context_checkpoint(ck, st, ev, IDENT, pref_spec=spec, available_bytes_fn=avail)
        st = run(ev, st, int(m))
    else:
        import case as case_mod

        from maple_syrup import native_floating_soil as nfs

        def boom(*a, **k):
            raise AssertionError("original-case/owner machinery must not run on the restore path")

        from maple_syrup import probe

        case_mod.build = boom
        probe.build_minimal_state = boom  # the static recipe must not allocate a probe bed
        nfs.initialize_native_floating_soil = boom
        nfs.stage_native_refresh = boom
        ev, st = cc.load_floating_context_checkpoint(ck, factory(backend, windowed == "1"), IDENT, available_bytes_fn=avail)
        vert = ev.vertical
        st = run(ev, st, int(m))
    snap = snapshot(ev, st, vert)
    np.savez(work / f"{mode}_final.npz", **snap)


if __name__ == "__main__":
    main(*sys.argv[1:7])
