"""`python fresh_n3_worker.py save|load WORKDIR BACKEND windowed|dense`: the REAL separate-interpreter seven-day N = 3 integration (correction01 C5).
`save`: genuine vegetated lifecycle (storm -> refresh -> boundary) -> accepted ACTUAL LEAF `PlantSession` generation 1 applied -> N = 3 writer created -> days 1-3 (frame on day 3) -> hand-back -> accepted session generation 2 applied between the frames
-> days 4-5 -> PAUSE at the CLOSED day 5: the output directory is copied (only frame 3) and the physical checkpoint is written -> days 6-7 uninterrupted (frame 6) -> dump (arrays, cursor, digests, attributions, directory tree).
`load` (a NEW interpreter): original bed / owner / probe / provider constructors AND the LEAF plant / projection constructors are DISABLED before anything is loaded; the child needs only the static case recipe, the files, the caller's independent LEAF
pins, the recorded seal and the external output directory (the pause-time copy): physical load -> `VegetationMoistureOutput.resume` -> days 6-7 -> dump. No generation, water or LEAF session is produced on restore (a native FILE decoder class is
constructed normally: the tripwires name ORIGINAL-bed / owner / provider creation only)."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import fresh_worker as fw
import veg_fr_support as S
import veg_lc_builders as vb
from vck_builders import AVAIL, IDENT, PART, RUN, boundary_vns, save

from maple_syrup import floating_daily as fd
from maple_syrup import floating_daily_output_vegetation as fov
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_moisture_frame as g1
from maple_syrup import soil_moisture_output as g0
from maple_syrup import vegetation_lifecycle as vl

LIM = g0.MoistureOutputLimits(max_snapshots=10, max_total_bytes=10**7, max_manifest_bytes=16384, chunk_bytes=4096, host_memory_budget_bytes=10**7, host_reserve_bytes=0, disk_reserve_bytes=0)
DATE0 = "2027-06-01"


def tree(d: Path) -> dict:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


def finish(work, name, state_by_day, out_dir):
    arrays = {}
    meta = {}
    for d, s in state_by_day.items():
        arrays[f"water{d}"], arrays[f"wet{d}"], arrays[f"intake{d}"] = np.asarray(s.water.water_m), np.asarray(s.history.wetting_m), np.asarray(s.history.cumulative_intake_m)
        meta[str(d)] = {"exe": s.execution_digest, "dig": s.digest, "cursor": repr(s.cursor), "t": float(s.t_s), "attr": json.dumps(vl.vegetation_attribution(s), sort_keys=True), "frame": repr(s.frame_identity())}
    np.savez(work / f"{name}_final.npz", **arrays)
    (work / f"{name}_meta.json").write_text(json.dumps({"days": meta, "tree": tree(out_dir)}, sort_keys=True))


def _main(mode, workdir, backend, storage):
    windowed = storage == "windowed"
    work = Path(workdir)
    ck, out_dir, paused = work / "ck", work / "out", work / "paused"
    if mode == "save":
        import leaf.state
        import leaf.traits
        import leaf.water_vegetation

        leaf = {"state": leaf.state, "traits": leaf.traits, "wv": leaf.water_vegetation}
        (work / "save").mkdir(parents=True, exist_ok=True)
        ns = boundary_vns(work / "save", leaf, backend, windowed, apply=False, deep_dense=not windowed)
        sess0, sess1, _cat, surf, mask, ctl, _nfx = S.session_pair(leaf, ns.b)
        nb1 = S.apply_session(ns.lin, ns.b, leaf, sess0, surf, mask, ctl)
        state = br.open_day(ns.lin, nb1, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
        out_dir.mkdir()
        out = fov.VegetationMoistureOutput.create(out_dir, state, expected_parent=fd.expect_parent(state), every_n_days=3, limits=LIM, origin_date=DATE0)
        for d in (1, 2, 3):
            state, note = S.day(state, out=out)
            assert note.day == d and note.due == (d == 3)
        b3 = br.hand_back(ns.lin, state)
        S.apply_session(ns.lin, b3, leaf, sess1, surf, mask, ctl)  # the accepted ACTUAL session generation 2 between the frames
        state = br.open_day(ns.lin, ns.lin.head, partition=PART, available_bytes_fn=AVAIL)
        for d in (4, 5):
            state, note = S.day(state, out=out)
            assert note.day == d and not note.due
        rec = vl.current_record(ns.lin.head)
        assert rec.generation == 2
        shutil.copytree(out_dir, paused)  # the N = 3 directory AT THE PAUSE: only frame 3
        assert [p.name for p in paused.iterdir() if p.name.startswith("frame_")] == [g1.frame_name(3)]
        saved = save(ns, ck, state)
        (work / "seal.json").write_text(json.dumps({"seal": saved.manifest_sha256, "recipe": saved.recipe_id, "generation": saved.generation}))
        days = {}
        s = state
        for d in (6, 7):
            s, note = S.day(s, out=out)
            assert note.day == d and note.due == (d == 6)
            days[d] = s
        finish(work, "save", days, out_dir)
    else:
        def boom(*a, **k):
            raise AssertionError("original-case / owner / probe / provider / LEAF plant-projection machinery must not run on the restore path")

        from maple_syrup import native_floating_soil as nfs
        from maple_syrup import probe

        for owner, names in ((nfs, ("initialize_native_floating_soil", "stage_native_refresh", "_batch_properties")), (probe, ("build_minimal_state",))):
            for n in names:
                setattr(owner, n, boom)
        import leaf.state as ls
        import leaf.water_vegetation as lw

        ls.PlantState.from_records = boom
        lw.project_from_initial_state = boom
        lw.project_from_plant_session = boom
        info = json.loads((work / "seal.json").read_text())
        from maple_syrup import floating_lifecycle_checkpoint_vegetation as lcv

        r = lcv.load_floating_lifecycle_vegetation_checkpoint(ck, fw.static_factory(backend, windowed), IDENT, info["seal"], leaf_package_sha256=vb.leaf_pin(), external_helpers=vb.helper_pin(),
                                                              original_topology=fw.sc.topology(), partition=PART, available_bytes_fn=AVAIL, memory_budget_bytes=10**12)
        assert r.recipe_id == info["recipe"] and r.generation == info["generation"] == 2
        out = fov.VegetationMoistureOutput.resume(paused, r.closed, expected_parent=fd.expect_parent(r.closed), limits=LIM)
        assert out.last_day == 5
        days = {}
        s = r.closed
        for d in (6, 7):
            s, note = S.day(s, out=out)
            assert note.day == d and note.due == (d == 6)
            days[d] = s
        finish(work, "load", days, paused)


def main(mode, workdir, backend, storage):
    before = fw.pins_now()
    _main(mode, workdir, backend, storage)
    assert fw.pins_now() == before


if __name__ == "__main__":
    main(*sys.argv[1:5])
