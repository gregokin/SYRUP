"""FRESH-PROCESS worker DEFINITION of the CUDA vegetation lifecycle file (WRITTEN, NOT RUN: no GPU was touched). `python gpu_fresh_worker.py save|load WORKDIR WINDOWED`.
`save`: the genuine matched DRY lifecycle (storm -> boundary -> actual LEAF projection -> S0 binding -> owning apply -> bridge day -> CLOSED day) -> the static recipe of the ORIGINAL event, verified against the original event's bindings
(refused on mismatch) -> the vegetation lifecycle file + the caller's recorded seal -> day 1 -> dump. `load` (a NEW interpreter): EVERY original bed / owner / probe / provider / pair constructor AND every LEAF plant / projection
constructor is disabled BEFORE anything else is imported or read; the restore needs only the file, `recipe.json` (static case inputs), the recorded seal, the independent LEAF pins and `cc2_static_case`; the same day 1 -> dump."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path[:0] = [str(HERE)] + [str(T / x) for x in ("phase25_vegetation_lifecycle", "phase2_floating_lifecycle_checkpoint_cuda", "phase2_floating_continued_checkpoint_cuda", "phase2_floating_lifecycle_checkpoint",
                                                   "phase2_floating_event_daily_bridge_cuda", "phase2_floating_event_daily_bridge", "phase2_floating_event_lifecycle_cuda", "phase2_floating_event_lifecycle",
                                                   "phase2_floating_event_refresh_cuda", "phase2_floating_event_refresh", "phase2_floating_daily", "phase2_floating_continued_checkpoint",
                                                   "phase2_floating_checkpoint_cuda", "phase2_floating_checkpoint", "phase2_floating_context_checkpoint", "phase2_floating_pond", "phase2_native_floating_soil",
                                                   "phase2_compact_soil_profile", "pond_event", "pond_windowed")]
import conftest as _cf  # noqa: F401  (puts the LEAF candidate root on sys.path)

DEFS_MODULE_NAME = "veg_checkpoint_gpu_definitions_own"


def load_own_definitions():
    """This directory's `test_gpu_definitions.py` loaded by its ACTUAL path under a UNIQUE module name (the plain basename `test_gpu_definitions` collides with sibling test directories that the conftest prepends to sys.path)."""
    import importlib.util

    if DEFS_MODULE_NAME in sys.modules:
        return sys.modules[DEFS_MODULE_NAME]
    path = HERE / "test_gpu_definitions.py"
    spec = importlib.util.spec_from_file_location(DEFS_MODULE_NAME, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[DEFS_MODULE_NAME] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(DEFS_MODULE_NAME, None)
        raise
    return mod

BIG = 1 << 40
IDENT = {"case": "s2vcuda", "v": 1}


def disable_original_constructors():
    """Placed BEFORE any recipe preparation on the LOAD path."""
    import fpond_builders as fb
    import leaf.state as ls
    import leaf.water_vegetation as lw

    from maple_syrup import native_floating_soil as nfs
    from maple_syrup import probe

    def boom(*a, **k):
        raise AssertionError("an original bed / owner / probe / provider / LEAF plant-projection constructor ran on the restore path")

    nfs.initialize_native_floating_soil = boom
    nfs.stage_native_refresh = boom
    probe.build_minimal_state = boom
    fb.make = boom
    fb.make_pair = boom
    fb._windowed_helpers = boom
    fb.ph.make_bed = boom
    ls.PlantState.from_records = boom
    lw.project_from_initial_state = boom
    lw.project_from_plant_session = boom
    return ("initialize_native_floating_soil", "stage_native_refresh", "build_minimal_state", "make", "make_pair", "_windowed_helpers", "make_bed", "from_records", "project_from_initial_state",
            "project_from_plant_session")


def next_day(lin, closed, rdr, hk, bd_):
    """Hand the closed day back and run the next genuine bridge day with the BOUND cover; return the boundary after the second closed day."""
    from maple.core.backend import cupy_module

    from maple_syrup import floating_daily as fd
    from maple_syrup import floating_event_daily_bridge as br
    from maple_syrup import recharge as rc

    cp = cupy_module()
    boundary = br.hand_back(lin, closed, available_bytes_fn=rdr, host_available_bytes_fn=hk)
    st = br.open_day(lin, boundary, partition=bd_.PART, run_digest=bd_.RUN, available_bytes_fn=rdr, host_available_bytes_fn=hk)
    n = st.ctx.n_cells
    res = br.day_action(st, plan=rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=bd_.RUN, ordinal=1), pet_terrain_m_s=cp.full(n, 3e-8), cover=None, available_bytes_fn=rdr, host_available_bytes_fn=hk)
    t = fd.request_floating_withdrawal(res.state, expected_parent=fd.expect_parent(res.state), request_m=cp.full((st.ctx.n_layers, n), 1e-6), expected_sequence=0, available_bytes_fn=rdr, host_available_bytes_fn=hk)
    closed2 = fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), available_bytes_fn=rdr, host_available_bytes_fn=hk).state
    return br.hand_back(lin, closed2, available_bytes_fn=rdr, host_available_bytes_fn=hk)


def dump(work, name, boundary, to_host):
    r = boundary.root
    np.savez(work / f"{name}_final.npz", W=to_host(r.water_state.water_m), J=to_host(r.history.wetting_m), I=to_host(r.history.cumulative_intake_m), D=to_host(r.depth_m), H=to_host(boundary.surface_depth_m),
             qx=to_host(boundary.qx_m2_s), qy=to_host(boundary.qy_m2_s), led=np.array([boundary.ledger[k] for k in sorted(boundary.ledger)], dtype=np.float64))


def main(mode, workdir, windowed):
    work = Path(workdir)
    ck = work / "ck"
    tripped = disable_original_constructors() if mode == "load" else None  # BEFORE anything else is imported or read
    import veg_lc_builders as vb
    import veg_open_recipe as vr
    from maple.core.backend import cupy_module, to_host

    from maple_syrup import floating_lifecycle_checkpoint_vegetation_cuda as cv
    from maple_syrup import floating_soil_layers_cuda as fcu

    cp = cupy_module()
    rdr = fcu.device_reader(cp)
    hk = lambda: BIG
    q = {"available_bytes_fn": rdr, "host_available_bytes_fn": hk, "memory_budget_bytes": BIG, "host_memory_budget_bytes": BIG}
    if mode == "save":
        import leaf.state
        import leaf.traits
        import leaf.water_vegetation
        defs = load_own_definitions()

        from maple_syrup import floating_event_lifecycle as lc
        from maple_syrup import vegetation_lifecycle as vl

        leaf = {"state": leaf.state, "traits": leaf.traits, "wv": leaf.water_vegetation}
        ns = defs.vegetated(defs.SimpleNamespace(cp=cp, m=defs.sup.modules()), leaf, work / "save", windowed == "1")
        st0 = defs.day(ns, ns.b, 0)
        original = ns.original
        captured = []

        def capturing(vert, topology=None, cover=None, schedule=None):  # the ORIGINAL open-plane event, built WITH the bound cover by the save itself: its binding digests are the recipe's reference
            ev = ns.factory(vert, topology, cover, schedule)
            captured.append(ev)
            return ev

        sv = cv.save_floating_lifecycle_vegetation_checkpoint_cuda(ck, ns.lin, capturing, IDENT, closed=st0, original_topology=original, **q)
        recipe = vr.make_open_recipe(captured[0], original)  # NEW open-plane helper (the old closed-lake helper keeps its guard)
        vr.write_open_recipe(work / "recipe.json", recipe)
        (work / "seal.json").write_text(json.dumps({"seal": sv.manifest_sha256, "recipe": sv.recipe_id, "generation": sv.generation}))  # the CALLER's own record of the sealed digest
        assert vl.current_record(ns.lin.head).recipe_id == sv.recipe_id and lc is not None
        dump(work, "save", next_day(ns.lin, st0, rdr, hk, ns.lib), to_host)
    else:
        defs = load_own_definitions()

        bd_ = defs.sup.modules().bd
        recipe = vr.read_open_recipe(work / "recipe.json")
        original = vr.original_topology(recipe)  # rebuilt through the actual API; ALL topology digests compared

        def fac(vert, topology, cover, schedule=None):
            return vr.static_event(recipe, vert, topology, cover, "cuda", schedule)  # ALL original binding digests compared
        info = json.loads((work / "seal.json").read_text())
        r = cv.load_floating_lifecycle_vegetation_checkpoint_cuda(ck, fac, IDENT, info["seal"], leaf_package_sha256=vb.leaf_pin(), external_helpers=vb.helper_pin(), original_topology=original,
                                                                  expected_device_id=int(cp.cuda.Device().id), partition=bd_.PART, **q)
        assert r.kind == "closed_day" and r.recipe_id == info["recipe"] and r.generation == info["generation"]
        (work / "tripwires.json").write_text(json.dumps({"disabled_before_recipe": list(tripped)}))
        dump(work, "load", next_day(r.lineage, r.closed, rdr, hk, bd_), to_host)


if __name__ == "__main__":
    main(*sys.argv[1:4])
