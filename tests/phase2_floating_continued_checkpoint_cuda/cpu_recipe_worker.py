"""CPU analogue (host-run) of the GPU fresh worker: `python cpu_recipe_worker.py save|load WORKDIR BACKEND WINDOWED M`. SAVE builds the genuine matched CPU generation-1 control, VERIFIES the recipe event against the original bindings,
writes recipe + refreshed-root H0 record + CC1 continued checkpoint and continues. LOAD (a NEW interpreter) disables every original bed/owner/probe/provider/pair constructor BEFORE the recipe is read and restores from recipe + H0 + file alone."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path[:0] = [str(HERE)] + [str(T / x) for x in ("phase2_floating_continued_checkpoint", "phase2_floating_event_lifecycle", "phase2_floating_event_refresh", "phase2_floating_event_refresh_cuda", "phase2_floating_completion",
                                                   "phase2_floating_checkpoint", "phase2_floating_context_checkpoint", "phase2_floating_pond", "phase2_native_floating_soil", "phase2_compact_soil_profile", "pond_event",
                                                   "pond_windowed")]
IDENT = {"case": "cc1", "v": 1}
AVAIL = lambda: 1 << 40


def main(mode, workdir, backend, windowed, m):
    work = Path(workdir)
    ck = work / "ck"
    if mode == "load":
        from gpu_continued_worker import _disable_original_constructors

        tripped = _disable_original_constructors()  # BEFORE the recipe / H0 record / anything else
    import cc2_fixtures as f
    import cc2_static_case as sc2
    from case import snapshot
    from cont_builders import record_from_files, record_to_files, run

    from maple_syrup import floating_context_checkpoint as fcx
    from maple_syrup import floating_continued_checkpoint as cc
    from maple_syrup import floating_held_context as fh

    def ctx_of(rec):
        return fh.reconstruct_held_context(rec, fcx._expected_from(rec.meta()), available_bytes_fn=AVAIL)

    if mode == "save":
        ns = f.cpu_control(work / "save", windowed == "1", backend)
        recipe = sc2.make_recipe(ns.ev0, ns.original)
        sc2.verify_event(recipe, sc2.factory(recipe, backend)(ctx_of(ns.record0)))
        cc.save_floating_continued_checkpoint(ck, ns.state, ns.event, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)
        sc2.write_recipe(work / "recipe.json", recipe)
        record_to_files(ns.record, work / "h0")
        ev, st, vert = ns.event, ns.state, ns.occ.vertical
    else:
        recipe = sc2.read_recipe(work / "recipe.json")
        orig = sc2.original_topology(recipe)
        origin = cc.read_continued_origin(ck, IDENT, available_bytes_fn=AVAIL)
        ev = sc2.factory(recipe, backend)(ctx_of(record_from_files(work / "h0")), cc.origin_topology(origin, orig))
        vert = ev.vertical
        st = cc.load_floating_continued_checkpoint(ck, ev, IDENT, original_topology=orig, available_bytes_fn=AVAIL)
        assert vert._owner is None and st.counters.n_refreshes == 0
        (work / "tripwires.json").write_text(json.dumps({"disabled_before_recipe": list(tripped)}))
    st = run(ev, st, int(m), rate=30.0 / 3.6e6, dt=0.1)
    np.savez(work / f"{mode}_final.npz", **snapshot(ev, st, vert))


if __name__ == "__main__":
    main(*sys.argv[1:6])
