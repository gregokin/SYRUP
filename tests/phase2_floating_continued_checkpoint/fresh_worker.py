"""`python fresh_worker.py save|load WORKDIR BACKEND WINDOWED M`. `save`: a genuine lifecycle boundary -> continued state -> continued-origin checkpoint + the refreshed-root H0 record files -> M real native steps -> dump. `load`: a NEW
interpreter; the original-case builder is used ONLY for the caller's static recipe (`factory`, `ev0`), then the owner initializer / refresh stage / probe constructor are DISABLED and the restart runs from the H0 record files and the
checkpoint alone (no boundary, lineage, owner or refreshed bed of process A) -> M steps -> dump."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path[:0] = [str(HERE)] + [str(T / x) for x in ("phase2_floating_event_lifecycle", "phase2_floating_event_refresh", "phase2_floating_completion", "phase2_floating_checkpoint", "phase2_floating_pond", "phase2_native_floating_soil",
                                                   "phase2_compact_soil_profile", "pond_event", "pond_windowed")]
from cont_builders import (
    AVAIL,
    IDENT,
    cc,
    lineage,
    record_from_files,
    record_to_files,
    restart_target,
    run,
    save,
    snapshot,
)


def main(mode, workdir, backend, windowed, m):
    work = Path(workdir)
    ck = work / "ck"
    if mode == "save":
        (work / "save").mkdir(parents=True, exist_ok=True)
        ns = lineage(work / "save", backend, windowed == "1")
        save(ns, ck)
        record_to_files(ns.record, work / "h0")
        ev, st, vert = ns.event, ns.state, ns.occ.vertical
    else:
        from lifecycle_builders import build

        (work / "load").mkdir(parents=True, exist_ok=True)
        _owner, _bed, factory, _pref, ev0 = build(work / "load", backend, windowed == "1", loose=True, mobile_kg=1.0)  # the caller's STATIC recipe only

        def boom(*a, **k):
            raise AssertionError("original-case/owner machinery must not run on the restore path")

        from maple_syrup import native_floating_soil as nfs
        from maple_syrup import probe

        nfs.initialize_native_floating_soil = boom
        nfs.stage_native_refresh = boom
        probe.build_minimal_state = boom
        ns = SimpleNamespace(factory=factory, original=ev0._initial_topology, record=record_from_files(work / "h0"))
        ev, vert, _origin = restart_target(ns, ck)
        st = cc.load_floating_continued_checkpoint(ck, ev, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)
        assert vert._owner is None and st.counters.n_refreshes == 0 and st.topology.generation >= 1
    st = run(ev, st, int(m))
    snap = snapshot(ev, st, vert)
    np.savez(work / f"{mode}_final.npz", **snap)


if __name__ == "__main__":
    main(*sys.argv[1:6])
