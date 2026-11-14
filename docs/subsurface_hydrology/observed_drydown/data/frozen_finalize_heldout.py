"""Completes a held-out config from the TRAINING result by the frozen selection rule (no refit, nothing re-tuned). Usage: finalize_heldout.py TRAIN_RESULT.json TEMPLATE.json OUT.json
Selection (frozen): over the training scenarios named d{d}_E{E}_open, the best pair minimises the primary overall RMSE; ties -> smaller E, then smaller d; a scenario with undefined RMSE is ineligible. The held-out scenario list is exactly:
best pair, d=0 at the fitted E, E=0 at the fitted d, the unfitted reference (d 0.05, E 1.2), and d0_E0; all open bottom (the closed-bottom sensitivity is only the two training entries)."""
import json
import re
import sys

train, tmpl, out = (json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3])
cands = []
for name, sc in train["scores"].items():
    m = re.fullmatch(r"d([0-9.eE+-]+)_E([0-9.eE+-]+)_open", name)
    if m and sc["overall_rmse"] is not None:
        cands.append((sc["overall_rmse"], float(m[2]), float(m[1])))
if not cands:
    sys.exit("no eligible training scenario (every overall RMSE is undefined)")
rmse, e, d = min(cands)
mk = lambda d, e: {"name": f"d{d:g}_E{e:g}_open", "diffusion_coefficient": d, "evaporation_factor": e, "potential_et": "weather", "bottom": "open"}  # noqa: E731
scen, seen = [], set()
for s in (mk(d, e), mk(0.0, e), mk(d, 0.0), mk(0.05, 1.2), mk(0.0, 0.0)):
    if s["name"] not in seen:
        seen.add(s["name"])
        scen.append(s)
tmpl["scenarios"] = scen
tmpl["provenance"]["heldout_status"] = f"completed from training result {sys.argv[1]}: best pair d={d:g}, E={e:g} (training overall RMSE {rmse})"
json.dump(tmpl, open(out, "w"), indent=1)
print(out, [s["name"] for s in scen])
