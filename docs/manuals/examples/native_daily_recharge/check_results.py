#!/usr/bin/env python
"""Check stored (or freshly produced) example summaries: usage `python check_results.py results/*.json`.

For a run made with --checkpoint --later-storm and a positive --t-request-m it verifies that the FINAL `closure` (native audit of the final accepted state) counts
each input exactly once: rain = first-day rain + later-storm rain; transpiration = the accepted T stage amount (the first-day audit, taken before T, shows 0);
|residual| <= the native floating-point bound; the final state is the completed later storm with its daily occurrence still pending. No model is run."""
import json
import math
import sys

REL = 1e-12


def main(paths):
    assert paths, "give summary JSON files"
    for path in paths:
        d = json.load(open(path))
        first, final, c, later, fs = d["first_day_audit_before_T"], d["final_state"], d["closure"], d["later_storm"], d["final_state"]
        t_stage = [s for s in d["stages"] if s["stage"].startswith("zero-time T")][0]["transpiration_m3"]
        assert first["transpiration_m3"] == 0.0 and t_stage > 0.0, (path, "needs a positive T run")
        assert math.isclose(c["rain_m3"], first["rain_m3"] + later["storm_rain_m3"], rel_tol=REL), (path, "rain not first day + later storm once")
        assert math.isclose(c["transpiration_m3"], t_stage, rel_tol=REL), (path, "T not counted once")
        assert abs(c["residual_m3"]) <= c["bound_m3"], (path, "closure outside the native bound")
        assert fs["role"] == "storm" and fs["pending_occurrence"] and fs["storm_genuinely_completed"] is True, (path, fs)
        assert fs["t_s"] == later["completed_at_s"], (path, "final clock is not the completed later storm")
        print(f"ok {path}: rain {c['rain_m3']} = {first['rain_m3']} + {later['storm_rain_m3']}; T {c['transpiration_m3']}")
    print("PASS")


if __name__ == "__main__":
    main(sys.argv[1:])
