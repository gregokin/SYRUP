"""`python fresh_worker.py save|load WORKDIR BACKEND [windowed|dense]`. `save`: a genuine lifecycle (Windowed bed; storm -> refresh -> boundary -> two bridge days) -> the lifecycle file at the CLOSED day 2 -> day 3 -> dump. `load`: a NEW interpreter that
imports only the STATIC case recipe (primitive parameters, no original bed / owner / probe / provider: those constructors are DISABLED) restores the file with the sealed digest the writer recorded and runs the same day 3 -> dump."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path[:0] = [str(HERE)] + [str(T / x) for x in ("phase2_floating_event_daily_bridge", "phase2_floating_continued_checkpoint", "phase2_floating_event_lifecycle", "phase2_floating_daily", "phase2_floating_checkpoint",
                                                   "phase2_floating_pond", "phase2_native_floating_soil", "phase2_floating_context_checkpoint", "pond_event", "pond_windowed")]
import static_case as sc
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_lifecycle_checkpoint as lcp
from maple_syrup.pond_event import PondCompletionPolicy

from lcp_builders import AVAIL, IDENT, PART, closed_day_ns, run_day, save

POL = PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-4, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True)
sc.CASE.update({"policy": POL, "max_dt_s": 0.2, "splash": False, "rain": (10.0, 60.0)})


def static_factory(backend, windowed):
    """The lifecycle's `event_factory(vertical, carried_topology)` from PRIMITIVE static parameters only."""
    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.pond_event import PondEvent, PondEventControl
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters

    def make(vert, topology):
        ctl = PondEventControl(backend=backend, time=ExperimentalControl(max_dt_s=0.2, min_dt_s=1.0 / 1024.0, max_retries=12, max_steps=50_000), bed_storage="windowed" if windowed else "dense", deep_publication="step")
        return PondEvent(sc.bed_context(), None, plot1_sediment_parameters(xp=np), constant_rainfall(0.0, 60.0, 10.0), topology, control=ctl, policy=POL, splash=None, floating_vertical=vert)
    return make


def dump(work, name, closed, b):
    np.savez(work / f"{name}_final.npz", water=np.asarray(closed.water.water_m), wet=np.asarray(closed.history.wetting_m), intake=np.asarray(closed.history.cumulative_intake_m),
             exe=np.frombuffer(bytes.fromhex(closed.execution_digest), dtype=np.uint8), dig=np.frombuffer(bytes.fromhex(closed.digest), dtype=np.uint8),
             ledger=np.array([b.ledger[k] for k in sorted(b.ledger)], dtype=np.float64), t=np.array([b.t_s]))


def main(mode, workdir, backend, storage="windowed"):
    windowed = storage == "windowed"
    work = Path(workdir)
    ck = work / "ck"
    if mode == "save":
        (work / "save").mkdir(parents=True, exist_ok=True)
        ns = closed_day_ns(work / "save", backend, windowed, days=2, deep_dense=not windowed)
        saved = save(ns, ck, ns.closed)
        (work / "seal.json").write_text(json.dumps({"seal": saved.manifest_sha256}))
        b = br.hand_back(ns.lin, ns.closed)
        s3 = br.open_day(ns.lin, b, partition=PART, available_bytes_fn=AVAIL)
        closed3 = run_day(s3, 2).state
        dump(work, "save", closed3, br.hand_back(ns.lin, closed3))
    else:
        def boom(*a, **k):
            raise AssertionError("original-case / owner / probe / provider machinery must not run on the restore path")

        from maple_syrup import native_floating_soil as nfs
        from maple_syrup import probe

        nfs.initialize_native_floating_soil = boom
        nfs.stage_native_refresh = boom
        nfs._batch_properties = boom
        probe.build_minimal_state = boom
        seal = json.loads((work / "seal.json").read_text())["seal"]
        r = lcp.load_floating_lifecycle_checkpoint(ck, static_factory(backend, windowed), IDENT, seal, original_topology=sc.topology(), partition=PART, available_bytes_fn=AVAIL)
        assert r.closed.ctx.hold is r.lineage.head and not hasattr(r, "event")  # the carrier is not retained; the tripwires above prove no owner/bed/provider ran
        b = br.hand_back(r.lineage, r.closed)
        s3 = br.open_day(r.lineage, b, partition=PART, available_bytes_fn=AVAIL)
        closed3 = run_day(s3, 2).state
        dump(work, "load", closed3, br.hand_back(r.lineage, closed3))


if __name__ == "__main__":
    main(*sys.argv[1:5])
