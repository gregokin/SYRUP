"""Fresh-process worker (a script, not a test module) for the moisture-output restart test. `A <workdir>` runs days 1..3 (N = 2: day 2 saved), notifying the writer each day, and saves the GENUINE V3 daily physical
checkpoint (lifecycle_checkpoint_v3) of the day-3 closed boundary. `B <workdir>` runs in a NEW process: rebuilds the fixed components, loads that V3 checkpoint, re-binds the output with MoistureOutput.resume, replays
the loaded state (a repeat), runs day 4 (due) and notifies. Both print JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase15_recharge"))

import numpy as np  # noqa: E402
from dcases import act, build, start, wl  # noqa: E402
from storm_cases import layers  # noqa: E402

from maple_syrup import layered_checkpoint as lc  # noqa: E402
from maple_syrup import lifecycle_checkpoint_v3 as v3  # noqa: E402
from maple_syrup import soil_moisture_output as mo  # noqa: E402

GRID = np.full((4, 6), 0.2)
IDENT = {"case": "g0-fresh"}
LIM = lc.CheckpointLimits(1 << 30, 1 << 30, 0, 1 << 30)
LIMITS = mo.MoistureOutputLimits(max_snapshots=20, max_total_bytes=1 << 30, max_manifest_bytes=8192, chunk_bytes=1 << 16, host_memory_budget_bytes=1 << 30, host_reserve_bytes=0, disk_reserve_bytes=0)
CALLER = {"adapter": "g0-test"}


def case():
    return build(GRID, lay=layers(bottom="open"), theta=(0.3, 0.3))


def day(c, state, p):
    _, r = act(c, state, p, pet_rate=2.0 ** -24)
    return wl.close_interevent(c.lop, r.state, expected_parent=wl.expect_parent(r.state))


def main(mode: str, work: Path) -> None:
    out_dir = work / "out"
    c = case()
    if mode == "A":
        out_dir.mkdir()
        s = start(c)
        out = mo.MoistureOutput.create(out_dir, c.lop, s, expected_parent=wl.expect_parent(s), every_n_days=2, limits=LIMITS, caller_identity=CALLER)
        for p in (2.0, 0.0, 1.0):
            s = day(c, s, p)
            out.notify_day_closed(c.lop, s, expected_parent=wl.expect_parent(s))
        v3.save_lifecycle_checkpoint_v3(work / "ck", c.lop, s, IDENT, limits=LIM, host_available_bytes_fn=lambda: 1 << 40)
        print(json.dumps({"day": out.last_day}))
        return
    loaded = v3.load_lifecycle_checkpoint_v3(work / "ck", c.lop, IDENT, expected_role="boundary", limits=LIM, host_available_bytes_fn=lambda: 1 << 40, reference_bed=c.s0.payload.bed)
    lop, s = loaded.lop, loaded.state
    out = mo.MoistureOutput.resume(out_dir, lop, s, expected_parent=wl.expect_parent(s), limits=LIMITS, caller_identity=CALLER)
    again = out.notify_day_closed(lop, s, expected_parent=wl.expect_parent(s))  # day 3 (not due), the very state resume accepted
    c2 = type(c)(**{**c.__dict__, "lop": lop})
    s4 = day(c2, s, 2.0)
    n4 = out.notify_day_closed(lop, s4, expected_parent=wl.expect_parent(s4))
    print(json.dumps({"repeat": again.action, "day4": n4.action, "path": n4.path, "last_day": out.last_day}))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
