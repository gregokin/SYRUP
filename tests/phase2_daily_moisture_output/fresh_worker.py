"""Fresh-process worker of the G2a restart test (a script, not a test module). `A <work>`: days 1..4 through close_final_day with N=2
(frames 2 and 4 sealed), saves the GENUINE V3 checkpoint of the DAY-2 boundary and prints the continuous fingerprints of days 3 and 4.
`B <work>`: a NEW process rebuilds the components, loads that day-2 V3 checkpoint, resumes the (AHEAD) output and replays days 3, 4; the
sealed day-4 frame must be VERIFIED (not rewritten) and the physical fingerprints must equal A's. Both print JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from daily_helpers import case, dmo, fp, interevent, limits, start, wl

from maple_syrup import layered_checkpoint as lc
from maple_syrup import lifecycle_checkpoint_v3 as v3

IDENT = {"case": "g2a-fresh"}
LIM = lc.CheckpointLimits(1 << 30, 1 << 30, 0, 1 << 30)
CALLER = {"adapter": "g2a-test"}
PS = (2.0, 0.0, 1.0, 2.0)


def main(mode: str, work: Path) -> None:
    out_dir = work / "out"
    c = case()
    if mode == "A":
        out_dir.mkdir()
        s = start(c)
        out = dmo.create_daily_moisture_output(out_dir, c.lop, s, expected_parent=wl.expect_parent(s), every_n_days=2, limits=limits(),
                                               caller_identity=CALLER)
        prints = {}
        for day, p in enumerate(PS, 1):
            ie = interevent(c, s, p, withdraw=1e-5 if day == 3 else None)
            s = dmo.close_final_day(c.lop, ie, expected_parent=wl.expect_parent(ie), output=out).state
            if day == 2:
                v3.save_lifecycle_checkpoint_v3(work / "ck", c.lop, s, IDENT, limits=LIM, host_available_bytes_fn=lambda: 1 << 40)
            if day >= 3:
                prints[str(day)] = fp(c, s)
        print(json.dumps({"last_day": out.last_day, "prints": prints}, default=str))
        return
    loaded = v3.load_lifecycle_checkpoint_v3(work / "ck", c.lop, IDENT, expected_role="boundary", limits=LIM, host_available_bytes_fn=lambda: 1 << 40,
                                             reference_bed=c.s0.payload.bed)
    lop, s = loaded.lop, loaded.state
    out = dmo.resume_daily_moisture_output(out_dir, lop, s, expected_parent=wl.expect_parent(s), limits=limits(), caller_identity=CALLER)  # the output is AHEAD (day 4 sealed)
    c2 = type(c)(**{**c.__dict__, "lop": lop})
    prints, actions = {}, []
    for day, p in enumerate(PS, 1):
        if day <= 2:
            continue
        ie = interevent(c2, s, p, withdraw=1e-5 if day == 3 else None)
        r = dmo.close_final_day(lop, ie, expected_parent=wl.expect_parent(ie), output=out)
        s = r.state
        prints[str(day)] = fp(c2, s)
        actions.append(r.notice.action)
    print(json.dumps({"last_day": out.last_day, "prints": prints, "actions": actions}, default=str))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
