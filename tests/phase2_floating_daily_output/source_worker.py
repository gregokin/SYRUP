"""Fresh-process worker of the source-binding control: builds the genuine original state in THIS process (whatever source tree PYTHONPATH selects) and creates / resumes the floating output."""
import json
import sys
from pathlib import Path

mode, out_dir = sys.argv[1], sys.argv[2]
tests = Path(sys.argv[3])
for sub in ("phase2_native_floating_soil", "phase2_compact_soil_profile", "phase2_floating_daily"):
    sys.path.insert(0, str(tests / sub))
from dbuild import lineage

import maple_syrup
from maple_syrup import floating_daily as fd
from maple_syrup import floating_daily_output as fo
from maple_syrup import soil_moisture_output as g0

LIM = g0.MoistureOutputLimits(max_snapshots=10, max_total_bytes=10**7, max_manifest_bytes=8192, chunk_bytes=4096, host_memory_budget_bytes=10**7, host_reserve_bytes=0, disk_reserve_bytes=0)
big = lambda: 10**12
st, *_ = lineage()
res = {"source": str(Path(maple_syrup.__file__).parent), "pair": dict(st.ctx.source_pair)}
try:
    if mode == "create":
        out = fo.FloatingMoistureOutput.create(out_dir, st, expected_parent=fd.expect_parent(st), every_n_days=1, limits=LIM, host_available_bytes_fn=big, disk_available_bytes_fn=big)
        from dbuild import day

        inter = day(st).state
        fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), output=out, host_available_bytes_fn=big, disk_available_bytes_fn=big)
        res["created"] = True
    else:
        fo.FloatingMoistureOutput.resume(out_dir, st, expected_parent=fd.expect_parent(st), limits=LIM, host_available_bytes_fn=big)
        res["accepted"] = True
except Exception as exc:  # noqa: BLE001
    res["accepted"] = False
    res["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(res))
