"""Standalone example of the G3 S1/S2 API (CPU). It builds a genuine native bed and owner from the repository's own test fixtures, starts a floating daily lineage, runs `days` no-rain days with a LEAF
withdrawal on day 1 and an every-N moisture output, and prints the saved frames. `N` is an ordinary runtime argument of this EXAMPLE script (not a product CLI flag):  python example.py <out_dir> <N> [days]
Run from the repository root with the tests directory on sys.path (the example reuses the test builders so that it needs no private data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["tests/phase2_native_floating_soil", "tests/phase2_compact_soil_profile", "tests/phase2_floating_daily"]


def main(out_dir: str, every_n_days: int, days: int = 4) -> None:
    import builders as b
    from dbuild import PART, RUN

    from maple_syrup import floating_daily as fd
    from maple_syrup import floating_daily_output as fo
    from maple_syrup import floating_moisture_frame as g1
    from maple_syrup import native_floating_soil as nfs
    from maple_syrup import recharge as rc
    from maple_syrup import soil_moisture_output as g0

    avail = lambda: 10**12
    bed = b.stratified()
    owner = nfs.initialize_native_floating_soil(bed, b.config(), initial_relative_saturation=0.5, available_bytes_fn=avail)
    state = fd.start_floating_daily_lineage(owner, bed, partition=PART, run_digest=RUN, available_bytes_fn=avail)
    limits = g0.MoistureOutputLimits(max_snapshots=100, max_total_bytes=10**8, max_manifest_bytes=8192, chunk_bytes=1 << 16, host_memory_budget_bytes=10**8, host_reserve_bytes=0, disk_reserve_bytes=0)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output = fo.FloatingMoistureOutput.create(out_dir, state, expected_parent=fd.expect_parent(state), every_n_days=every_n_days, limits=limits)
    n = state.ctx.n_cells
    for d in range(days):
        plan = rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=state.cursor.ordinal)  # P = 0: a no-rain day
        res = fd.floating_daily_action(state, expected_parent=fd.expect_parent(state), plan=plan, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=avail)
        inter = res.state
        if d == 1:  # LEAF's aggregate request (here a demonstration value) debited once by SYRUP; LEAF receives actual / unmet / scale
            request = np.zeros((inter.ctx.n_layers, n))
            request[0] = 1e-5
            inter = fd.request_floating_withdrawal(inter, expected_parent=fd.expect_parent(inter), request_m=request, expected_sequence=0, available_bytes_fn=avail).state
        closed = fd.close_floating_day(inter, expected_parent=fd.expect_parent(inter), output=output)
        state = closed.state
        print(f"day {closed.notice.day}: {closed.notice.action}")
    for p in sorted(Path(out_dir).glob("frame_*")):
        frame = g1.read_floating_frame(p, max_return_bytes=10**8)
        print(p.name, frame.water_m.shape, f"day={frame.manifest['day']}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 4)
