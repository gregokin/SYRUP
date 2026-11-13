"""FRESH-PROCESS worker DEFINITION of the CUDA continued-origin profile (WRITTEN, NOT RUN: no GPU was touched). `python gpu_continued_worker.py save|load WORKDIR WINDOWED M`.
`save`: genuine CUDA lifecycle lake -> boundary -> continued state -> continued CUDA checkpoint + the refreshed-root H0 record parts (meta json + raw arrays) -> M real steps -> dump. `load`: a NEW interpreter; the static recipe
(`device_event` of the E5 lake) is used ONLY to read the caller's static case, then the owner initializer / refresh stage / probe constructor are DISABLED; the H0 record parts + the checkpoint alone restore the state; M steps -> dump.
Requires the composed GPU lifecycle (SAVE only). The LOAD side uses `cc2_static_case` (no original bed/owner/probe/provider: the constructors are disabled BEFORE the recipe is read) and never imports `test_gpu_definitions`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path[:0] = [str(HERE)] + [str(T / x) for x in ("phase2_floating_continued_checkpoint", "phase2_floating_event_lifecycle", "phase2_floating_event_refresh", "phase2_floating_event_refresh_cuda", "phase2_floating_completion", "phase2_floating_checkpoint", "phase2_floating_context_checkpoint",
                                                   "phase2_floating_checkpoint_cuda", "phase2_floating_pond", "phase2_native_floating_soil", "phase2_compact_soil_profile", "pond_event", "pond_windowed")]
BIG = 1 << 40
IDENT = {"case": "cc2", "v": 1}
RAIN = 30.0 / 3.6e6


def _disable_original_constructors():
    """Placed BEFORE any recipe preparation on the LOAD path: every original bed / owner / texture-provider / probe / pair constructor raises."""
    import fpond_builders as fb

    from maple_syrup import native_floating_soil as nfs
    from maple_syrup import probe

    def boom(*a, **k):
        raise AssertionError("an original bed/owner/probe/provider constructor ran on the restore path")

    nfs.initialize_native_floating_soil = boom
    nfs.stage_native_refresh = boom
    probe.build_minimal_state = boom
    fb.make = boom
    fb.make_pair = boom
    fb._windowed_helpers = boom
    fb.ph.make_bed = boom
    return ("initialize_native_floating_soil", "stage_native_refresh", "build_minimal_state", "make", "make_pair", "_windowed_helpers", "make_bed")


def main(mode, workdir, windowed, m):
    work = Path(workdir)
    ck = work / "ck"
    if mode == "load":
        tripped = _disable_original_constructors()  # BEFORE importing the recipe, the H0 record or anything else
    import cc2_static_case as sc2
    from case import snapshot
    from maple.core.backend import cupy_module

    from maple_syrup import floating_continued_checkpoint_cuda as ccc
    from maple_syrup import floating_held_context as fh

    cupy_module()
    if mode == "save":
        import test_gpu_definitions as defs

        L = defs.Lin(cupy_module(), work / "save", windowed == "1")
        recipe = sc2.make_recipe(L.ev0, L.original)
        ctx0 = fh.reconstruct_held_context(L.record0, fh.ExpectedHeldIdentities(*(L.record0.meta()["identities"][k] for k in ("config_sha256", "root_column_sha256", "matrix_op_sha256", "pref_sha256"))),
                                           available_bytes_fn=L.reader, host_available_bytes_fn=lambda: BIG)
        sc2.verify_event(recipe, sc2.factory(recipe, "cuda")(ctx0, None, available_bytes_fn=L.reader, host_available_bytes_fn=lambda: BIG))  # the static recipe must reproduce the ORIGINAL event's bindings, else refuse
        ccc.save_floating_continued_checkpoint_cuda(ck, L.state, L.event, IDENT, original_topology=L.original, available_bytes_fn=L.reader, host_available_bytes_fn=lambda: BIG)
        sc2.write_recipe(work / "recipe.json", recipe)
        meta, arrays = L.record.meta(), {k: np.ascontiguousarray(a.get() if hasattr(a, "get") else a) for k, a in L.record.arrays().items()}
        (work / "h0_meta.json").write_text(json.dumps(meta))
        np.savez(work / "h0_arrays.npz", **arrays)
        ev, st, vert = L.event, L.state, L.occ.vertical
    else:
        recipe = sc2.read_recipe(work / "recipe.json")
        with np.load(work / "h0_arrays.npz", allow_pickle=False) as z:
            record = fh.record_from_parts(json.loads((work / "h0_meta.json").read_text()), {k: np.array(z[k]) for k in z.files}, available_bytes_fn=lambda: BIG)
        ids = record.meta()["identities"]
        ctx = fh.reconstruct_held_context(record, fh.ExpectedHeldIdentities(ids["config_sha256"], ids["root_column_sha256"], ids["matrix_op_sha256"], ids["pref_sha256"]), available_bytes_fn=lambda: BIG,
                                          host_available_bytes_fn=lambda: BIG)
        orig = sc2.original_topology(recipe)
        origin = ccc.read_continued_origin_cuda(ck, IDENT, available_bytes_fn=lambda: BIG)
        ev = sc2.factory(recipe, "cuda")(ctx, ccc.origin_topology(origin, orig), host_available_bytes_fn=lambda: BIG)
        vert = ev.vertical
        st = ccc.load_floating_continued_checkpoint_cuda(ck, ev, IDENT, original_topology=orig, available_bytes_fn=lambda: BIG, host_available_bytes_fn=lambda: BIG)
        assert vert._owner is None and st.counters.n_refreshes == 0
        (work / "tripwires.json").write_text(json.dumps({"disabled_before_recipe": list(tripped)}))
    for _ in range(int(m)):
        st = ev.step(st, RAIN, 0.1).state
    np.savez(work / f"{mode}_final.npz", **snapshot(ev, st, vert))


if __name__ == "__main__":
    main(*sys.argv[1:5])
