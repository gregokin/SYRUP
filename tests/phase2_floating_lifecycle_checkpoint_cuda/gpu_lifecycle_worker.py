"""FRESH-PROCESS worker DEFINITION of the CUDA lifecycle file (WRITTEN, NOT RUN: no GPU was touched). `python gpu_lifecycle_worker.py save|load WORKDIR WINDOWED`.
`save`: the genuine matched PIT lifecycle (storm -> boundary -> bridge day -> CLOSED day) -> the recipe of the ORIGINAL event, verified against the original event's bindings (refused on mismatch) -> the lifecycle file + the
caller's recorded seal -> day 1 -> dump. `load` (a NEW interpreter): EVERY original bed/owner/probe/provider/pair constructor is disabled BEFORE anything else is imported or read; the restore needs only the file, `recipe.json`
(static case inputs; the original padded datum terrain is a recorded static input, never inferred from the bed), the recorded seal and `cc2_static_case`; the same day 1 -> dump. It never imports the GPU fixtures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path[:0] = [str(HERE)] + [str(T / x) for x in ("phase2_floating_continued_checkpoint_cuda", "phase2_floating_lifecycle_checkpoint", "phase2_floating_event_daily_bridge_cuda", "phase2_floating_event_daily_bridge",
                                                   "phase2_floating_event_lifecycle_cuda", "phase2_floating_event_lifecycle", "phase2_floating_event_refresh_cuda", "phase2_floating_event_refresh",
                                                   "phase2_floating_daily", "phase2_floating_continued_checkpoint", "phase2_floating_checkpoint_cuda", "phase2_floating_checkpoint", "phase2_floating_context_checkpoint",
                                                   "phase2_floating_pond", "phase2_native_floating_soil", "phase2_compact_soil_profile", "pond_event", "pond_windowed")]
BIG = 1 << 40
IDENT = {"case": "s2cuda", "v": 1}


def disable_original_constructors():
    """Placed BEFORE any recipe preparation on the LOAD path."""
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


def recipe_factory(recipe):
    """`factory(vertical, carried_topology[, schedule]) -> PondEvent` from the closed recipe only (no bed, owner or probe)."""
    from dataclasses import replace

    import cc2_static_case as sc2
    from maple.core.backend import cupy_module, to_device

    from maple_syrup.experimental_storm import ExperimentalControl
    from maple_syrup.pond_event import PondCompletionPolicy, PondEvent, PondEventControl
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters

    r = sc2.check_recipe(recipe)
    cp = cupy_module()

    def make(vert, topology=None, schedule=None):
        ctx = replace(sc2.bed_context(r["ny"], r["nx"]), initial_available_fraction=to_device(sc2.bed_context(r["ny"], r["nx"]).initial_available_fraction, cp))
        c = r["control"]
        ctl = PondEventControl(backend="cuda", time=ExperimentalControl(max_dt_s=c["max_dt_s"], min_dt_s=c["min_dt_s"], max_retries=c["max_retries"], max_steps=c["max_steps"]), bed_storage=r["bed_storage"],
                               deep_publication=c["deep_publication"])
        top = sc2.original_topology(r) if topology is None else topology
        return PondEvent(ctx, None, plot1_sediment_parameters(xp=cp), constant_rainfall(0.0, r["rain_end_s"], r["rain_mm_h"]) if schedule is None else schedule, top, control=ctl, policy=PondCompletionPolicy(**r["policy"]),
                         splash=None, floating_vertical=vert)

    return make


def dump(work, name, boundary, to_host):
    r = boundary.root
    np.savez(work / f"{name}_final.npz", W=to_host(r.water_state.water_m), J=to_host(r.history.wetting_m), I=to_host(r.history.cumulative_intake_m), D=to_host(r.depth_m), H=to_host(boundary.surface_depth_m),
             qx=to_host(boundary.qx_m2_s), qy=to_host(boundary.qy_m2_s), led=np.array([boundary.ledger[k] for k in sorted(boundary.ledger)], dtype=np.float64))


def day_one(lin, closed, fac, rdr, hk, to_host):
    """Hand the closed day back and run the next real native day (wet pit: short fragments, T once) -> the boundary after the second closed day."""
    import chain
    from maple.core.backend import cupy_module

    from maple_syrup import floating_daily as fd
    from maple_syrup import floating_event_daily_bridge as br
    from maple_syrup import recharge as rc
    from maple_syrup.experimental_storm import ExperimentalControl

    cp = cupy_module()
    ctl = ExperimentalControl(max_dt_s=3600.0, min_dt_s=1 / 1024, max_retries=30, max_steps=2_000_000)
    boundary = br.hand_back(lin, closed, available_bytes_fn=rdr, host_available_bytes_fn=hk)
    st = br.open_day(lin, boundary, partition=chain.PART, run_digest=chain.RUN, available_bytes_fn=rdr, host_available_bytes_fn=hk)
    n = st.ctx.n_cells
    res = br.day_action(st, plan=rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=chain.RUN, ordinal=1), pet_terrain_m_s=cp.full(n, 3e-8), cover=cp.full(n, 0.3), available_bytes_fn=rdr, event_factory=fac, day_control=ctl,
                        host_available_bytes_fn=hk)
    t = fd.request_floating_withdrawal(res.state, expected_parent=fd.expect_parent(res.state), request_m=cp.zeros((st.ctx.n_layers, n)), expected_sequence=0, available_bytes_fn=rdr, host_available_bytes_fn=hk)
    closed2 = fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), available_bytes_fn=rdr, host_available_bytes_fn=hk).state
    return br.hand_back(lin, closed2, available_bytes_fn=rdr, host_available_bytes_fn=hk)


def main(mode, workdir, windowed):
    work = Path(workdir)
    ck = work / "ck"
    tripped = disable_original_constructors() if mode == "load" else None  # BEFORE anything else is imported or read
    import cc2_static_case as sc2
    import chain
    from maple.core.backend import cupy_module, to_host

    from maple_syrup import floating_lifecycle_checkpoint_cuda as lpcc
    from maple_syrup import floating_soil_layers_cuda as fcu

    cp = cupy_module()
    rdr = fcu.device_reader(cp)
    hk = lambda: BIG
    if mode == "save":
        import test_gpu_definitions as defs

        from maple_syrup import floating_event_lifecycle as lc
        from maple_syrup import floating_held_context as fh
        from maple_syrup import floating_held_context_cuda as fhc

        g = defs.DevSide(work / "save", defs.bd.pit_kwargs(), True, windowed == "1")
        lin, occ, s0 = lc.open_floating_event(g.owner, g.bed, g.factory, available_bytes_fn=g.rdr, **g.hk)
        rec0 = fhc.export_held_context_cuda(occ.vertical, None, available_bytes_fn=g.rdr)  # the ORIGINAL-root H0 record (before any refresh)
        h = defs.ld.complete(occ, s0)
        b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=g.rdr, **g.hk)
        g.lin = lin
        _st, res = g.day(b, 0)
        _t, closed = g.finish(res)
        original = g.ev._initial_topology
        recipe = sc2.make_recipe(g.ev, original)
        ctx0 = fh.reconstruct_held_context(rec0, fh.ExpectedHeldIdentities(*(rec0.meta()["identities"][k] for k in ("config_sha256", "root_column_sha256", "matrix_op_sha256", "pref_sha256"))),
                                           available_bytes_fn=g.rdr, host_available_bytes_fn=hk)
        sc2.verify_event(recipe, sc2.factory(recipe, "cuda")(ctx0, None, available_bytes_fn=g.rdr, host_available_bytes_fn=hk))  # the recipe must reproduce the ORIGINAL event's bindings, else refuse
        sv = lpcc.save_floating_lifecycle_checkpoint_cuda(ck, lin, g.factory, IDENT, closed=closed, original_topology=original, available_bytes_fn=g.rdr, host_available_bytes_fn=hk)
        sc2.write_recipe(work / "recipe.json", recipe)
        (work / "seal.txt").write_text(sv.manifest_sha256)  # the CALLER's own record of the sealed digest
        dump(work, "save", day_one(lin, closed, g.factory, g.rdr, hk, to_host), to_host)
    else:
        recipe = sc2.read_recipe(work / "recipe.json")
        original = sc2.original_topology(recipe)
        fac = recipe_factory(recipe)
        r = lpcc.load_floating_lifecycle_checkpoint_cuda(ck, fac, IDENT, (work / "seal.txt").read_text().strip(), original_topology=original, expected_device_id=int(cp.cuda.Device().id), partition=chain.PART,
                                                         available_bytes_fn=rdr, host_available_bytes_fn=hk)
        assert r.kind == "closed_day"
        (work / "tripwires.json").write_text(json.dumps({"disabled_before_recipe": list(tripped)}))
        dump(work, "load", day_one(r.lineage, r.closed, fac, rdr, hk, to_host), to_host)


if __name__ == "__main__":
    main(*sys.argv[1:4])
