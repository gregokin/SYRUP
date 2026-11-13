"""Fresh-process worker on the GENUINE native file-case factory (NCP source verification, admitted package case, prepared Dense case, layered
lifecycle from a YAML file), following the accepted Root G0 restart-worker pattern but driving the G2a FACADE `close_final_day` (no direct
`close_interevent` + manual G0 notify). `full <backend> <work>`: days 1..4 with N = 2 (frames 2 and 4 sealed), actual V3 checkpoint of the DAY-1
boundary. `resume <backend> <work>`: a NEW process with its own `factory_resume` source-guard paths loads that V3 checkpoint, resumes the output
(frames 2 and 4 are AHEAD), replays days 2..4 and records the same trace. Requires SYRUP_NCP_TARGET (a configured wrong bundle FAILS)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "phase15_recharge"))

import ncpcases as nc
from dcases import dw

from maple_syrup import daily_moisture_output as dmo
from maple_syrup import layered_checkpoint as lc
from maple_syrup import lifecycle_checkpoint_v3 as v3
from maple_syrup import soil_moisture_output as mo
from maple_syrup import water_lifecycle as wl

IDENT = {"case": "g2a_file_case_restart"}
CKLIM = lc.CheckpointLimits(128 << 20, 128 << 20, 0, 128 << 20)


def main(mode: str, backend: str, folder: Path) -> None:
    nc.require_configured_bundle()  # FAILS (never skips) when the configured bundle is not the imported MAPLE
    folder.mkdir(parents=True, exist_ok=True)
    raw = nc.yaml_for(2, bottom="open", theta=0.2)
    raw["compute"]["backend"] = backend
    c = nc.build_file_case(folder / ("factory_" + mode), raw, flat_closed=True)  # separate owned factory/source-guard paths per process
    lim = mo.MoistureOutputLimits(max_snapshots=10, max_total_bytes=16 << 20, max_manifest_bytes=8192, chunk_bytes=64, host_memory_budget_bytes=2 << 20,
                                  host_reserve_bytes=0, disk_reserve_bytes=0)
    lop = c.lop
    if mode == "full":
        state = dw.start_daily_lineage(lop, c.s0, expected_parent=wl.expect_parent(c.s0), run_digest="a5" * 32)
        out_dir = folder / "moisture"
        out_dir.mkdir()
        writer = dmo.create_daily_moisture_output(out_dir, lop, state, expected_parent=wl.expect_parent(state), every_n_days=2, limits=lim,
                                                  origin_date="2026-01-01")
        days = range(1, 5)
    else:
        loaded = v3.load_lifecycle_checkpoint_v3(folder / "ck1", lop, IDENT, expected_role="boundary", limits=CKLIM, host_available_bytes_fn=lambda: 1 << 40,
                                                 reference_bed=c.s0.payload.bed)
        lop, state = loaded.lop, loaded.state
        writer = dmo.resume_daily_moisture_output(folder / "moisture", lop, state, expected_parent=wl.expect_parent(state), limits=lim)
        days = range(2, 5)
    trace = []
    for day in days:
        plan = dw.next_plan(state, 2.0 if day % 2 else 0.0, 10.0, 5.0)
        r = dw.daily_water_action(lop, state, expected_parent=wl.expect_parent(state), plan=plan, pet_terrain_m_s=np.full(c.n, 1e-8))
        req = np.zeros((c.L, c.n))
        req[0] = 1e-5
        got, _ = wl.request_withdrawal(lop, r.state, expected_parent=wl.expect_parent(r.state), depth_m=req, identity=f"g2a_daily_leaf_request_{day}")
        res = dmo.close_final_day(lop, got, expected_parent=wl.expect_parent(got), output=writer)
        state = res.state
        trace.append({"day": day, "physical_digest": v3.physical_digest_v3(lop, state), "state_digest": state.digest,
                      "water_sha256": hashlib.sha256(np.ascontiguousarray(state.payload.soil.water_m).tobytes()).hexdigest(), "action": res.notice.action})
        if mode == "full" and day == 1:
            v3.save_lifecycle_checkpoint_v3(folder / "ck1", lop, state, IDENT, limits=CKLIM, host_available_bytes_fn=lambda: 1 << 40)
    import maple

    record = {"trace": trace, "backend": backend, "status": "PASS", "maple_file": str(Path(maple.__file__).resolve()), "maple_commit": c.profile.maple_commit,
              "syrup_tree_digest": c.syrup_digest, "mode": mode}
    (folder / (mode + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    print(mode, backend, "PASS", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
