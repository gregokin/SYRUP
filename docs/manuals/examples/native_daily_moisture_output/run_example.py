"""Minimal runnable example: every-N-days soil-moisture snapshots on the Phase 1.5 daily lifecycle, built from the GENUINE native file-case factory.

Prerequisites (explicit configuration, no default): the materialized MAPLE 9f bundle path in SYRUP_NCP_TARGET; run with that bundle's `editable`,
`source/src` and this SYRUP `src` on PYTHONPATH and MAPLE_SYRUP_EXPECTED_MAPLE_ROOT=<bundle>/source. N and every resource limit are explicit below.

    python run_example.py <workdir> <N>

Scope: the FIXED-full-porous common-column profile only (G0 snapshot schema). The floating porous-depth hook is NOT wired here and remains pending.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTS = HERE.parents[3] / "tests" / "phase15_recharge"  # the existing genuine file-case helpers (ncpcases) of this source tree
sys.path.insert(0, str(TESTS))

import ncpcases as nc

from maple_syrup import daily_moisture_output as dmo
from maple_syrup import daily_water as dw
from maple_syrup import soil_moisture_output as mo
from maple_syrup import water_lifecycle as wl


def main(workdir: Path, every_n_days: int) -> None:
    nc.require_configured_bundle()  # a missing/wrong bundle FAILS; nothing is substituted
    case = nc.build_file_case(workdir / "case", nc.yaml_for(2))
    state = dw.start_daily_lineage(case.lop, case.s0, expected_parent=wl.expect_parent(case.s0), run_digest="ab" * 32)
    out_dir = workdir / "moisture"
    out_dir.mkdir()
    limits = mo.MoistureOutputLimits(max_snapshots=100, max_total_bytes=1 << 30, max_manifest_bytes=8192, chunk_bytes=1 << 16,
                                     host_memory_budget_bytes=1 << 30, host_reserve_bytes=0, disk_reserve_bytes=0)
    output = dmo.create_daily_moisture_output(out_dir, case.lop, state, expected_parent=wl.expect_parent(state), every_n_days=every_n_days,
                                              limits=limits)
    for rain_mm in (0.25, 0.0, 0.25, 0.0):  # four nominal days
        plan = dw.next_plan(state, rain_mm, 3600.0, 1.0)
        daily = dw.daily_water_action(case.lop, state, expected_parent=wl.expect_parent(state), plan=plan, pet_terrain_m_s=np.full(case.n, 1e-8))
        # a caller with plants would call wl.request_withdrawal(...) here (the ACTUAL accepted water); without plants this is a legitimate close
        result = dmo.close_final_day(case.lop, daily.state, expected_parent=wl.expect_parent(daily.state), output=output)
        state = result.state
        print(f"day {result.notice.day}: {result.notice.action} {result.notice.path or ''}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
