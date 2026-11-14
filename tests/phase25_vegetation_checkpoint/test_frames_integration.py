"""Seven genuine nominal days with N = 3 (frames on days 3 and 6), a NEW vegetation generation between them (actual LEAF `PlantSession` generations), a pause at CLOSED day 5, the vegetation physical checkpoint, a FRESH-lineage restore and the
continuation of days 6 and 7 compared with the uninterrupted control (arrays, cursor, digests, sealed frame bytes, attributions). The frame modules (`floating_daily_output_vegetation`, `floating_moisture_frame_vegetation`) are a PROVISIONAL
READONLY dependency (diagnostics, NOT restart state; frame C2 is being corrected separately): this test only drives their public API. Their explicit nominal-calendar convention is preserved: the frame writer is resumed from the
restored CLOSED state; no rain / E / T / drainage / time is duplicated."""
from __future__ import annotations

import hashlib
import os
import shutil

import numpy as np
import pytest
import veg_fr_support as S
from vck_builders import AVAIL, PART, RUN, boundary_vns, load, save

from maple_syrup import floating_daily as fd
from maple_syrup import floating_daily_output_vegetation as fov
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_moisture_frame as g1
from maple_syrup import soil_moisture_output as g0

LIM = g0.MoistureOutputLimits(max_snapshots=10, max_total_bytes=10**7, max_manifest_bytes=16384, chunk_bytes=4096, host_memory_budget_bytes=10**7, host_reserve_bytes=0, disk_reserve_bytes=0)
DATE0 = "2027-06-01"


def tree(d) -> dict:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_seven_days_n3_pause_at_closed_day_5_fresh_restore_and_continue_equals_the_uninterrupted_run(tmp_path, leaf, windowed):
    ns = boundary_vns(tmp_path / "a", leaf, "numpy", windowed, apply=False)
    sess0, sess1, _cat, surf, mask, ctl, _nfx = S.session_pair(leaf, ns.b)
    nb1 = S.apply_session(ns.lin, ns.b, leaf, sess0, surf, mask, ctl)
    state = br.open_day(ns.lin, nb1, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = fov.VegetationMoistureOutput.create(out_dir, state, expected_parent=fd.expect_parent(state), every_n_days=3, limits=LIM, origin_date=DATE0)
    for d in (1, 2, 3):
        state, note = S.day(state, out=out)
        assert note.day == d and note.due == (d == 3)
    b3 = br.hand_back(ns.lin, state)
    S.apply_session(ns.lin, b3, leaf, sess1, surf, mask, ctl)  # the NEW generation (the next accepted session sequence) between the frames
    state = br.open_day(ns.lin, ns.lin.head, partition=PART, available_bytes_fn=AVAIL)
    for d in (4, 5):
        state, note = S.day(state, out=out)
        assert note.day == d and not note.due
    closed5 = state
    rec = S.vl.current_record(ns.lin.head)
    assert rec.generation == 2
    paused = tmp_path / "out_paused"
    shutil.copytree(out_dir, paused)  # the immutable pause-time frame directory (day 3 frame only)
    assert sorted(p for p in os.listdir(paused) if p.startswith("frame_")) == [g1.frame_name(3)]
    attr5 = S.vl.vegetation_attribution(closed5)
    saved = save(ns, tmp_path / "ck", closed5)
    assert saved.generation == 2 and saved.recipe_id == rec.recipe_id and S.vl.current_record(ns.lin.head) is rec
    # the uninterrupted control
    ref = {}
    s = closed5
    for d in (6, 7):
        s, note = S.day(s, out=out)
        assert note.day == d and note.due == (d == 6)
        ref[d] = (np.array(s.water.water_m, copy=True), np.array(s.history.wetting_m, copy=True), np.array(s.history.cumulative_intake_m, copy=True), s.execution_digest, s.digest, s.cursor, float(s.t_s), S.vl.vegetation_attribution(s))
    # the restored lineage: a NEW lineage / head / record, the writer resumed from the restored CLOSED state, the same two days
    r = load(ns, saved.path, saved, partition=PART)
    assert S.vl.vegetation_attribution(r.closed) == {**attr5, "backend": attr5["backend"]} and r.closed.frame_identity() == closed5.frame_identity()
    out2 = fov.VegetationMoistureOutput.resume(paused, r.closed, expected_parent=fd.expect_parent(r.closed), limits=LIM)
    assert out2.last_day == 5  # the writer resumed AT the restored closed day (nominal day 5); only day 3 is a published frame so far
    s2 = r.closed
    for d in (6, 7):
        s2, note = S.day(s2, out=out2)
        assert note.day == d and note.due == (d == 6)
        w, j, i, exe, dig, cur, t, attr = ref[d]
        assert np.array_equal(w, s2.water.water_m) and np.array_equal(j, s2.history.wetting_m) and np.array_equal(i, s2.history.cumulative_intake_m)
        assert (exe, dig, cur, t) == (s2.execution_digest, s2.digest, s2.cursor, float(s2.t_s)) and attr == S.vl.vegetation_attribution(s2)
    a, b = tree(out_dir), tree(paused)
    assert a == b and {p.split("/")[0] for p in a if p.startswith("frame_")} == {g1.frame_name(3), g1.frame_name(6)}  # every sealed byte of the frames equals the uninterrupted run's
    fr = g1.frame_name(6)
    assert fr in os.listdir(paused) and fr in os.listdir(out_dir)
    assert hashlib.sha256((paused / fr / g1.MANIFEST).read_bytes()).hexdigest() == hashlib.sha256((out_dir / fr / g1.MANIFEST).read_bytes()).hexdigest()
