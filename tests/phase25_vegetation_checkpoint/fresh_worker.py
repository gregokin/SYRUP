"""`python fresh_worker.py save|load WORKDIR BACKEND [windowed|dense]`. `save`: a genuine vegetated lifecycle (storm -> refresh -> boundary -> actual LEAF projection -> apply -> two bridge days) -> the vegetation lifecycle file at the CLOSED
day 2 -> day 3 -> hand-back -> a storm with the BOUND cover -> dump. `load`: a NEW interpreter that imports only the STATIC case recipe (primitive parameters: no original bed / owner / probe / provider / LEAF plant or projection constructor
is available) restores the file with the sealed digest the writer recorded and the INDEPENDENT LEAF pins, and runs the same continuation -> dump. Neither side needs the other's in-memory state."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path[:0] = [str(HERE)] + [str(T / x) for x in ("phase25_vegetation_lifecycle", "phase2_floating_lifecycle_checkpoint", "phase2_floating_event_daily_bridge", "phase2_floating_continued_checkpoint",
                                                   "phase2_floating_event_lifecycle", "phase2_floating_daily", "phase2_floating_checkpoint", "phase2_floating_pond", "phase2_native_floating_soil",
                                                   "phase2_floating_context_checkpoint", "pond_event", "pond_windowed")]
import conftest as _cf
import static_case as sc
from vck_builders import AVAIL, IDENT, PART, closed_day_vns, run_day, save, storm

from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import vegetation_lifecycle as vl
from maple_syrup.pond_event import PondCompletionPolicy

POL = PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-4, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True)
sc.CASE.update({"policy": POL, "max_dt_s": 0.2, "splash": True, "rain": (10.0, 60.0)})


def static_factory(backend, windowed):
    """The lifecycle's `event_factory(vertical, carried_topology, vegetation_cover_fraction[, schedule])` from PRIMITIVE static parameters only."""
    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.pond_event import PondEvent, PondEventControl
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters
    from maple_syrup.splash import splash_parameters

    def make(vert, topology, cover, schedule=None):
        ctl = PondEventControl(backend=backend, time=ExperimentalControl(max_dt_s=0.2, min_dt_s=1.0 / 1024.0, max_retries=12, max_steps=50_000), bed_storage="windowed" if windowed else "dense", deep_publication="step")
        return PondEvent(sc.bed_context(), None, plot1_sediment_parameters(xp=np), constant_rainfall(0.0, 60.0, 10.0) if schedule is None else schedule, topology, control=ctl, policy=POL,
                         splash=splash_parameters(np.linspace(0.4, 3.0, sc.NC), xp=np), vegetation_cover_fraction=cover, floating_vertical=vert)
    return make


def dump(work, name, closed, b, n):
    np.savez(work / f"{name}_final.npz", water=np.asarray(closed.water.water_m), wet=np.asarray(closed.history.wetting_m), intake=np.asarray(closed.history.cumulative_intake_m),
             exe=np.frombuffer(bytes.fromhex(closed.execution_digest), dtype=np.uint8), dig=np.frombuffer(bytes.fromhex(closed.digest), dtype=np.uint8),
             ledger=np.array([b.ledger[k] for k in sorted(b.ledger)], dtype=np.float64), t=np.array([b.t_s]),
             storm_water=np.asarray(n.root.water_state.water_m), storm_wet=np.asarray(n.root.history.wetting_m), storm_t=np.array([n.t_s]),
             storm_ledger=np.array([n.ledger[k] for k in sorted(n.ledger)], dtype=np.float64), recipe=np.frombuffer(bytes.fromhex(vl.current_record(n).recipe_id), dtype=np.uint8))


def pins_now() -> dict:
    """The CHILD asserts the same pins as the parent session: the imported MAPLE under the configured immutable materialization with the full package digest (fbc504bb..., 412 files) and the actual LEAF (38009872..., 28 files)."""
    import leaf
    import maple

    import maple_syrup

    assert Path(maple.__file__).resolve().is_relative_to(_cf.MAPLE_ROOT / "source" / "src"), maple.__file__
    d = {"MAPLE": _cf._digest(maple), "SYRUP": _cf._digest(maple_syrup), "LEAF": _cf._digest(leaf)}
    assert d["MAPLE"]["digest"] == _cf.MAPLE_PACKAGE and d["MAPLE"]["files"] == 412 and d["LEAF"]["digest"] == _cf.LEAF_PACKAGE and d["LEAF"]["files"] == 28, d
    return d


def main(mode, workdir, backend, storage="windowed"):
    before = pins_now()
    _main(mode, workdir, backend, storage)
    assert pins_now() == before  # unchanged after the work


def _main(mode, workdir, backend, storage="windowed"):
    windowed = storage == "windowed"
    work = Path(workdir)
    ck = work / "ck"
    import veg_lc_builders as vb

    if mode == "save":
        (work / "save").mkdir(parents=True, exist_ok=True)
        import leaf.state
        import leaf.traits
        import leaf.water_vegetation

        leaf = {"state": leaf.state, "traits": leaf.traits, "wv": leaf.water_vegetation}
        ns = closed_day_vns(work / "save", leaf, backend, windowed, days=2, deep_dense=not windowed)
        saved = save(ns, ck, ns.closed)
        (work / "seal.json").write_text(json.dumps({"seal": saved.manifest_sha256, "generation": saved.generation, "recipe": saved.recipe_id}))
        b = br.hand_back(ns.lin, ns.closed)
        s3 = br.open_day(ns.lin, b, partition=PART, available_bytes_fn=AVAIL)
        closed3 = run_day(s3, 2).state
        b3 = br.hand_back(ns.lin, closed3)
        n = storm(ns.lin, b3, ns.factory)
        dump(work, "save", closed3, b3, n)
    else:
        def boom(*a, **k):
            raise AssertionError("original-case / owner / probe / provider / LEAF plant-projection machinery must not run on the restore path")

        from maple_syrup import native_floating_soil as nfs
        from maple_syrup import probe

        saved_refresh = (nfs.stage_native_refresh, nfs._batch_properties)  # a LATER genuine closed-event property refresh legitimately reads the restored bed: armed only while LOADING and the restore-time days
        nfs.initialize_native_floating_soil = boom
        nfs.stage_native_refresh = boom
        nfs._batch_properties = boom
        probe.build_minimal_state = boom
        import leaf.state as ls
        import leaf.water_vegetation as lw

        for owner, names in ((ls.PlantState, ("from_records",)), (lw, ("project_from_initial_state", "project_from_plant_session"))):
            for nme in names:
                setattr(owner, nme, boom)
        info = json.loads((work / "seal.json").read_text())
        from maple_syrup import floating_lifecycle_checkpoint_vegetation as lcv

        r = lcv.load_floating_lifecycle_vegetation_checkpoint(ck, static_factory(backend, windowed), IDENT, info["seal"], leaf_package_sha256=vb.leaf_pin(), external_helpers=vb.helper_pin(),
                                                              original_topology=sc.topology(), partition=PART, available_bytes_fn=AVAIL, memory_budget_bytes=10**12)
        assert r.closed.ctx.hold is r.lineage.head and not hasattr(r, "event") and r.recipe_id == info["recipe"] and r.generation == info["generation"]
        assert vl.current_record(r.lineage.head).binding.origin == "cached_derived"
        b = br.hand_back(r.lineage, r.closed)
        s3 = br.open_day(r.lineage, b, partition=PART, available_bytes_fn=AVAIL)
        closed3 = run_day(s3, 2).state
        b3 = br.hand_back(r.lineage, closed3)
        nfs.stage_native_refresh, nfs._batch_properties = saved_refresh
        n = storm(r.lineage, b3, static_factory(backend, windowed))
        dump(work, "load", closed3, b3, n)
    assert lc is not None


if __name__ == "__main__":
    main(*sys.argv[1:5])
