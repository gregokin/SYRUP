# ruff: noqa: RUF059
"""Phase 2.5 vegetation-aware moisture frame (v2) and every-N output: REAL LEAF P1 + genuine accepted `PlantSession` generations, real lifecycle boundaries (numpy / numba x Dense / published Windowed), real bridge daily lineage.
Seven genuine daily updates with N = 3 (frames on nominal days 3 and 6) and a NEW vegetation generation applied between day 3 and day 4. Independent expectations: theta reconstructed by hand, the saved water compared with the accepted
state's water, attribution compared with the actual holders. LEAF is never faked (no claim when it cannot be imported)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import veg_fr_support as S
from veg_fr_support import AVAIL, BIG, PART, RUN, br, day, dry_boundary, fd, vl

from maple_syrup import floating_daily_output as fo
from maple_syrup import floating_daily_output_vegetation as fov
from maple_syrup import floating_moisture_frame as g1
from maple_syrup import floating_moisture_frame_vegetation as g2
from maple_syrup import soil_moisture_output as g0

LIM = g0.MoistureOutputLimits(max_snapshots=10, max_total_bytes=10**7, max_manifest_bytes=16384, chunk_bytes=4096, host_memory_budget_bytes=10**7, host_reserve_bytes=0, disk_reserve_bytes=0)
DATE0 = "2027-06-01"


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def open_state(lin, b):
    return br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)


def read(out, d):
    return g2.read_vegetation_frame(os.path.join(out.directory, g1.frame_name(d)), max_return_bytes=BIG)


@pytest.mark.parametrize("windowed", [False, True])
def test_seven_days_n3_with_a_new_vegetation_generation_between_the_frames(tmp_path, leaf, backend, windowed):
    lin, b, _factory = dry_boundary(tmp_path / "a", backend, windowed)
    sess0, sess1, cat, surf, mask, ctl, nfx = S.session_pair(leaf, b)
    assert sess1.sequence > sess0.sequence and sess1.session_id == sess0.session_id
    nb1 = S.apply_session(lin, b, leaf, sess0, surf, mask, ctl)
    rec1 = vl.current_record(nb1)
    state = open_state(lin, nb1)
    (tmp_path / "out").mkdir()
    out = fov.VegetationMoistureOutput.create(tmp_path / "out", state, expected_parent=fd.expect_parent(state), every_n_days=3, limits=LIM, origin_date=DATE0)
    saved = {}
    for d in (1, 2, 3):
        state, note = day(state, out=out)
        assert note.day == d and note.due == (d == 3) and note.action == ("saved" if d == 3 else "not_due")
        saved[d] = (np.array(state.water.water_m, copy=True), np.array(state.ctx.depth_m, copy=True), state.frame_identity(), float(state.t_s))
    # the lifecycle hand-back closes the cumulative ledger; THEN the new generation (the next accepted session sequence) is applied on the complete boundary BEFORE day 4
    b3 = br.hand_back(lin, state)
    assert abs(b3.ledger["residual_m3"]) <= b3.ledger["residual_bound_m3"] and b3.cursor.ordinal == 3
    nb2 = S.apply_session(lin, b3, leaf, sess1, surf, mask, ctl)
    rec2 = vl.current_record(nb2)
    assert rec2.generation == 2 and rec2.recipe_id != rec1.recipe_id and rec2.source["sequence"] == sess1.sequence
    state = open_state(lin, nb2)
    for d in (4, 5, 6, 7):
        state, note = day(state, out=out)
        assert note.day == d and note.due == (d == 6)
        saved[d] = (np.array(state.water.water_m, copy=True), np.array(state.ctx.depth_m, copy=True), state.frame_identity(), float(state.t_s))
    assert sorted(p for p in os.listdir(out.directory) if p.startswith("frame_")) == [g1.frame_name(3), g1.frame_name(6)]  # exactly the due frames (nominal days 3 and 6)
    for d, rec, nb, sess in ((3, rec1, nb1, sess0), (6, rec2, nb2, sess1)):
        fr = read(out, d)
        man = fr.manifest
        w, depth, ident, t_s = saved[d]
        assert man["schema"] == g2.SCHEMA_FRAME_V2 and man["profile"] == g2.PROFILE_V2 and man["day"] == d and man["cadence_n"] == 3 and man["date"] == f"2027-06-{1 + d:02d}"
        assert np.array_equal(fr.water_m, w) and np.array_equal(fr.depth_m, depth)  # the stable water receipt: the accepted state's W / D bit for bit
        assert np.allclose(fr.theta(max_bytes=BIG), S.theta_independent(w, depth, np.asarray(fr.depth_edges_m)), rtol=1e-12, atol=0.0)
        assert float(man["physical_t_s"]) == t_s and man["identity"]["state_digest"] == ident["state_digest"]  # physical seconds are recorded separately from the nominal day
        v = man["vegetation"]
        assert v["generation"] == rec.generation and v["recipe_id"] == rec.recipe_id and v["binding_sha256"] == rec.binding_sha256 and v["source"]["sequence"] == sess.sequence
        assert v["current_column_sha256"] == nb.root.column.sha256 == man["identity"]["property_sha256"]  # OLD property meaning = the CURRENT root column identity
        assert v["current_matrix_op_sha256"] == nb.root.matrix_op.sha256 and v["current_property_generation"] == nb.root.generation == man["identity"]["property_generation"]
        assert v["current_law_id"] != rec.recipe_id and "applied" not in v  # the apply-time ids are never stored
    assert read(out, 3).manifest["vegetation"]["recipe_id"] != read(out, 6).manifest["vegetation"]["recipe_id"]  # the generation changed between the frames (static identity allowed it)
    assert read(out, 3).manifest["identity"]["source_sha256"] == read(out, 6).manifest["identity"]["source_sha256"]  # the STABLE source anchor did not move
    # the manifest stays within its analytic model (literal bytes, not padding)
    raw = g0._read_small(g0._open_dir(os.path.join(out.directory, g1.frame_name(6))), g1.MANIFEST, 1 << 20)
    assert len(raw) <= g2._manifest_model_bytes(int(read(out, 6).water_m.shape[0]), 0)


def test_profiles_refuse_each_other_the_old_writer_still_fails_closed_and_a_lineage_without_a_generation_writes_the_null_block(tmp_path, leaf):
    lin, b, _f = dry_boundary(tmp_path / "a", "numpy", False)
    state = open_state(lin, b)
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    o1 = fo.FloatingMoistureOutput.create(tmp_path / "v1", state, expected_parent=fd.expect_parent(state), every_n_days=1, limits=LIM)
    state1, n1 = day(state, out=o1, bound=False)
    assert n1.due and n1.action == "saved"
    p1 = os.path.join(tmp_path / "v1", g1.frame_name(1))
    assert g1.read_floating_frame(p1, max_return_bytes=BIG).manifest["schema"] == g1.SCHEMA_FRAME
    with pytest.raises(g0.MoistureOutputError):
        g2.read_vegetation_frame(p1, max_return_bytes=BIG)  # the v2 reader refuses a v1 frame
    # the v2 output on a lineage with NO vegetation generation: the explicit null contract (generation 0 is not a fake LEAF value)
    lin2, b2, _f2 = dry_boundary(tmp_path / "b", "numpy", False)
    s2 = open_state(lin2, b2)
    o2 = fov.VegetationMoistureOutput.create(tmp_path / "v2", s2, expected_parent=fd.expect_parent(s2), every_n_days=1, limits=LIM)
    _s, n2 = day(s2, out=o2, bound=False)
    p2 = os.path.join(tmp_path / "v2", g1.frame_name(1))
    assert n2.due and g2.read_vegetation_frame(p2, max_return_bytes=BIG).manifest["vegetation"] is None
    with pytest.raises(g0.MoistureOutputError):
        g1.read_floating_frame(p2, max_return_bytes=BIG)  # the v1 reader refuses a v2 frame
    # directories are never shared: v2 create into a v1 directory, v2 resume of a v1 directory and v1 resume of a v2 directory are refused
    with pytest.raises(g0.MoistureOutputError):
        fov.VegetationMoistureOutput.create(tmp_path / "v1", s2, expected_parent=fd.expect_parent(s2), every_n_days=1, limits=LIM)
    with pytest.raises(g0.MoistureOutputError, match="no vegetation-aware"):
        fov.VegetationMoistureOutput.resume(tmp_path / "v1", state1, expected_parent=fd.expect_parent(state1), limits=LIM)
    with pytest.raises(g0.MoistureOutputError):
        fo.FloatingMoistureOutput.resume(tmp_path / "v2", _s, expected_parent=fd.expect_parent(_s), limits=LIM)
    # the v1 writer still FAILS CLOSED for a vegetation-bound state, and an unknown output object is refused by the close
    sess0, sess1, cat, surf, mask, ctl, nfx = S.session_pair(leaf, b2)
    nb = S.apply_session(lin2, _bridge_boundary(lin2, _s), leaf, sess0, surf, mask, ctl)
    sb = open_state(lin2, nb)
    (tmp_path / "v1b").mkdir()
    with pytest.raises(g0.MoistureOutputError, match="vegetation generation"):
        fo.FloatingMoistureOutput.create(tmp_path / "v1b", sb, expected_parent=fd.expect_parent(sb), every_n_days=1, limits=LIM)
    n = sb.ctx.n_cells
    res = br.day_action(sb, plan=S.rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=int(sb.cursor.ordinal)), pet_terrain_m_s=np.full(n, 3e-8), cover=None, available_bytes_fn=AVAIL)
    t = fd.request_floating_withdrawal(res.state, expected_parent=fd.expect_parent(res.state), request_m=np.zeros((sb.ctx.n_layers, n)), expected_sequence=0, available_bytes_fn=AVAIL)
    with pytest.raises(fd.DailyRefusal, match="no duck-typed"):
        fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), output=object(), available_bytes_fn=AVAIL)


def _bridge_boundary(lin, closed):
    return br.hand_back(lin, closed)


def test_cadence_is_strict_and_a_not_due_day_touches_no_array(tmp_path, monkeypatch):
    lin, b, _f = dry_boundary(tmp_path / "a", "numpy", False)
    state = open_state(lin, b)
    for bad in (0, -1, True, 1.5, "3", None, np.float64(3.0)):
        (tmp_path / "bad").mkdir(exist_ok=True)
        with pytest.raises(g0.MoistureOutputError):
            fov.VegetationMoistureOutput.create(tmp_path / "bad", state, expected_parent=fd.expect_parent(state), every_n_days=bad, limits=LIM)
    (tmp_path / "o").mkdir()
    out = fov.VegetationMoistureOutput.create(tmp_path / "o", state, expected_parent=fd.expect_parent(state), every_n_days=3, limits=LIM)
    writes, hosts = [], []
    real_w, real_h = g2.write_vegetation_frame, g1._to_host
    monkeypatch.setattr(g2, "write_vegetation_frame", lambda *a, **k: (writes.append(1), real_w(*a, **k))[1])
    monkeypatch.setattr(g1, "_to_host", lambda a: (hosts.append(1), real_h(a))[1])
    for _ in (1, 2):
        state, note = day(state, out=out, bound=False)
        assert not note.due and note.action == "not_due" and note.bytes_written == 0
    assert writes == [] and hosts == []  # a not-due day: no write, no array piece read, no transfer
    state, note = day(state, out=out, bound=False)
    assert note.due and writes == [1] and hosts  # the due day streams the bounded chunks
    assert [p for p in os.listdir(tmp_path / "o") if p.startswith("frame_")] == [g1.frame_name(3)]


def test_quotas_retry_replay_resume_stale_and_source_rebind(tmp_path, monkeypatch):
    lin, b, _f = dry_boundary(tmp_path / "a", "numpy", False)
    state = open_state(lin, b)
    (tmp_path / "o").mkdir()
    out = fov.VegetationMoistureOutput.create(tmp_path / "o", state, expected_parent=fd.expect_parent(state), every_n_days=2, limits=LIM)
    state, _ = day(state, out=out, bound=False)
    n, L = state.ctx.n_cells, state.ctx.n_layers
    pl = S.rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=int(state.cursor.ordinal))
    res = br.day_action(state, plan=pl, pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    t = fd.request_floating_withdrawal(res.state, expected_parent=fd.expect_parent(res.state), request_m=np.zeros((L, n)), expected_sequence=0, available_bytes_fn=AVAIL)
    frame = tmp_path / "o" / g1.frame_name(2)
    # host-read and disk quotas refuse BEFORE any write; the accepted interevent parent is untouched and the retry works
    with pytest.raises(g0.MoistureOutputError):
        fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), output=out, available_bytes_fn=AVAIL, host_available_bytes_fn=lambda: 0)
    assert not frame.exists()
    with pytest.raises(g0.MoistureOutputError):
        fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), output=out, available_bytes_fn=AVAIL, disk_available_bytes_fn=lambda: 0)
    assert not frame.exists()
    cr = fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), output=out, available_bytes_fn=AVAIL)
    assert cr.notice.action == "saved" and frame.exists()
    # the same closed state notified again is a no-op repeat; a foreign lineage's closed state is refused
    assert out.notify_day_closed(cr.state, expected_parent=fd.expect_parent(cr.state)).action == "repeat_same_state"
    lin2, b2, _f2 = dry_boundary(tmp_path / "b", "numpy", False)
    other = open_state(lin2, b2)
    other_closed, _n = day(other, bound=False)
    with pytest.raises(g0.MoistureOutputError):
        out.notify_day_closed(other_closed, expected_parent=fd.expect_parent(other_closed))
    # RESUME on the SAME genuine closed state (labelled: not a physical fresh-process restart): the sealed frame is re-proved against the live bytes and day 3 continues
    out2 = fov.VegetationMoistureOutput.resume(tmp_path / "o", cr.state, expected_parent=fd.expect_parent(cr.state), limits=LIM)
    assert out2.notify_day_closed(cr.state, expected_parent=fd.expect_parent(cr.state)).action == "verified_existing"
    s3, n3 = day(cr.state, out=out2, bound=False)
    assert n3.day == 3 and not n3.due
    # a changed writer / frame-codec source is refused (no silent rebind or migration)
    monkeypatch.setattr(fov, "_MODULE_SHA", "0" * 64)
    with pytest.raises(g0.MoistureOutputError, match="different writer"):
        fov.VegetationMoistureOutput.resume(tmp_path / "o", cr.state, expected_parent=fd.expect_parent(cr.state), limits=LIM)
    monkeypatch.undo()
    monkeypatch.setattr(g2, "_MODULE_SHA", "1" * 64)
    with pytest.raises(g0.MoistureOutputError, match="different writer"):
        fov.VegetationMoistureOutput.resume(tmp_path / "o", cr.state, expected_parent=fd.expect_parent(cr.state), limits=LIM)


def test_the_frame_codec_needs_no_leaf_lifecycle_or_model_runtime_and_the_old_codec_and_receipts_are_unchanged(tmp_path):
    code = "import sys, maple_syrup.floating_moisture_frame_vegetation as m; bad=[k for k in sys.modules if k=='leaf' or k.startswith('leaf.') or k in ('maple_syrup.vegetation_lifecycle','maple_syrup.water_vegetation_binding','maple_syrup.water_vegetation_held')]; print(json.dumps(bad))"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    r = subprocess.run([sys.executable, "-B", "-c", "import json;" + code], capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == []  # a plain frame is readable without LEAF / the lifecycle / any model runtime
    assert g1.SCHEMA_FRAME.endswith("/1") and fo.SCHEMA_OUTPUT.endswith("/2") and g1.PROFILE == "surface_following_floating_profile"  # the v1 schemas / profile names are untouched


def _reseal(frame_dir, mutate):
    """Edit ONE sealed frame's manifest as plain JSON and RESEAL its body checksum (a hostile / corrupt past frame the writer never made). Returns the original bytes."""
    path = frame_dir / g1.MANIFEST
    raw = path.read_bytes()
    man = json.loads(raw)
    mutate(man)
    man.pop("body_sha256")
    man["body_sha256"] = g0._body_sha(man)
    path.write_bytes(g0._canon(man))
    return raw


def _lineage(tmp_path, leaf, *, with_generation, days=3):
    lin, b, _f = dry_boundary(tmp_path / "a", "numpy", False)
    if with_generation:
        sess0, _s1, _cat, surf, mask, ctl, _nfx = S.session_pair(leaf, b)
        b = S.apply_session(lin, b, leaf, sess0, surf, mask, ctl)
    state = open_state(lin, b)
    (tmp_path / "o").mkdir()
    out = fov.VegetationMoistureOutput.create(tmp_path / "o", state, expected_parent=fd.expect_parent(state), every_n_days=1, limits=LIM)
    states = []
    for _ in range(days):
        state, _note = day(state, out=out, bound=with_generation)
        states.append(state)
    return out, states


@pytest.mark.parametrize("edit", [
    lambda m: m.__setitem__("profile", "other_profile"), lambda m: m.__setitem__("module_sha256", "2" * 64), lambda m: m["identity"].__setitem__("source_sha256", "3" * 64),
    lambda m: m["identity"].__setitem__("run_digest", "4" * 64), lambda m: m.__setitem__("cadence_n", 7), lambda m: m.__setitem__("origin_ordinal", m["origin_ordinal"] + 5),
    lambda m: m.__setitem__("shape_yx", [m["shape_yx"][1], m["shape_yx"][0]] if m["shape_yx"][0] != m["shape_yx"][1] else [1, m["n_cells"]]), lambda m: m.__setitem__("cell_area_m2", 2.0 * m["cell_area_m2"]),
    lambda m: m.__setitem__("caller_identity", {"x": 1}), lambda m: m.__setitem__("schema", "maple_syrup.floating_moisture_frame/1"),
], ids=["profile", "module", "source", "run", "cadence", "origin", "shape", "area", "caller", "schema"])
def test_resume_proves_every_past_due_frame_manifest_not_only_its_day(tmp_path, leaf, edit):
    out, states = _lineage(tmp_path, leaf, with_generation=False)
    last = states[-1]
    frame1 = tmp_path / "o" / g1.frame_name(1)
    raw = _reseal(frame1, edit)  # a RESEALED hostile past frame: its body checksum is consistent, so only the closed manifest / binding proof can refuse it
    before = sorted(p.name for p in (tmp_path / "o").iterdir())
    with pytest.raises(g0.MoistureOutputError):
        fov.VegetationMoistureOutput.resume(tmp_path / "o", last, expected_parent=fd.expect_parent(last), limits=LIM)
    assert sorted(p.name for p in (tmp_path / "o").iterdir()) == before  # nothing created, nothing overwritten
    (frame1 / g1.MANIFEST).write_bytes(raw)  # the genuine bytes resume again
    out2 = fov.VegetationMoistureOutput.resume(tmp_path / "o", last, expected_parent=fd.expect_parent(last), limits=LIM)
    assert out2.last_day == 3 and out2.notify_day_closed(last, expected_parent=fd.expect_parent(last)).action == "verified_existing"


def test_resume_reproves_the_frame_at_the_resume_day_against_the_state_not_only_the_next_replay(tmp_path, leaf):
    out, states = _lineage(tmp_path, leaf, with_generation=True)
    last = states[-1]
    frame3 = tmp_path / "o" / g1.frame_name(3)
    for edit in (lambda m: m.__setitem__("physical_t_s", m["physical_t_s"] + 1.0), lambda m: m["vegetation"].__setitem__("current_law_id", "5" * 64), lambda m: m["identity"].__setitem__("state_digest", "6" * 64)):
        raw = _reseal(frame3, edit)
        with pytest.raises((g0.MoistureOutputConflict, g0.MoistureOutputError)):
            fov.VegetationMoistureOutput.resume(tmp_path / "o", last, expected_parent=fd.expect_parent(last), limits=LIM)
        (frame3 / g1.MANIFEST).write_bytes(raw)
    assert fov.VegetationMoistureOutput.resume(tmp_path / "o", last, expected_parent=fd.expect_parent(last), limits=LIM).last_day == 3


def test_resume_refuses_a_past_generation_above_the_state_or_a_foreign_recipe_and_the_next_write_cannot_regress(tmp_path, leaf):
    out, states = _lineage(tmp_path, leaf, with_generation=True)
    last = states[-1]
    frame1, frame2 = (tmp_path / "o" / g1.frame_name(d) for d in (1, 2))
    # a sealed PAST frame whose generation is above the resumed state's (a stale / foreign lineage's state): refused
    raw1 = _reseal(frame1, lambda m: m["vegetation"].__setitem__("generation", 9))
    with pytest.raises(g0.MoistureOutputError, match="regress|differs"):
        fov.VegetationMoistureOutput.resume(tmp_path / "o", last, expected_parent=fd.expect_parent(last), limits=LIM)
    (frame1 / g1.MANIFEST).write_bytes(raw1)
    # the same generation with another recipe id between past frames: refused
    raw2 = _reseal(frame2, lambda m: m["vegetation"].__setitem__("recipe_id", "7" * 64))
    with pytest.raises(g0.MoistureOutputError):
        fov.VegetationMoistureOutput.resume(tmp_path / "o", last, expected_parent=fd.expect_parent(last), limits=LIM)
    (frame2 / g1.MANIFEST).write_bytes(raw2)
    # the live output's own order key forbids a later regression
    key = out.order
    assert key is not None and key[0] == 1
    cur = g2.attribution_from_dict(read(out, 3).manifest["vegetation"])
    g2.check_order(key, cur)
    higher = g2.attribution_from_dict({**cur.as_dict(), "generation": 2})
    g2.check_order(key, higher)  # a higher generation (a new accepted session) is legitimate
    with pytest.raises(g0.MoistureOutputError, match="regresses"):
        g2.check_order((2, higher.recipe_id), cur)
    with pytest.raises(g0.MoistureOutputError, match="recipe"):
        g2.check_order(key, g2.attribution_from_dict({**cur.as_dict(), "recipe_id": "8" * 64}))
    with pytest.raises(g0.MoistureOutputError, match="regresses|recipe"):
        g2.check_order(key, None)  # no dropping back to the null block after a generation
    g2.check_order(None, None)


def test_a_lagging_resume_still_accepts_sealed_frames_ahead_with_a_higher_generation(tmp_path, leaf):
    out, states = _lineage(tmp_path, leaf, with_generation=True)
    frame3 = tmp_path / "o" / g1.frame_name(3)
    _reseal(frame3, lambda m: (m["vegetation"].__setitem__("generation", 4), m["vegetation"].__setitem__("recipe_id", "9" * 64)))  # AHEAD of the resumed day 2: not a past frame
    out2 = fov.VegetationMoistureOutput.resume(tmp_path / "o", states[1], expected_parent=fd.expect_parent(states[1]), limits=LIM)
    assert out2.last_day == 2  # frames ahead are verified (and here would conflict) only when the replay reaches them


def test_attribution_field_sizes_are_closed_and_the_worst_admitted_attribution_fits_the_analytic_model(tmp_path, leaf):
    out, _states = _lineage(tmp_path, leaf, with_generation=True, days=1)
    good = g2.attribution_from_dict(read(out, 1).manifest["vegetation"])
    base = good.as_dict()
    big = "a" * (g2.SOURCE_VALUE_MAX_CHARS + 1)
    with pytest.raises(g0.MoistureOutputError):
        g2.attribution_from_dict({**base, "source": {**base["source"], "session_id": big}})
    with pytest.raises(g0.MoistureOutputError):
        g2.attribution_from_dict({**base, "source": {**base["source"], "session_id": 'a"b'}})  # outside the no-escape charset
    def hp(module="leaf.module", sha="sha256:" + "a" * 64, nbytes=10):
        return {"module": module, "sha256": sha, "bytes": nbytes}

    for bad in ([hp()] * (g2.MAX_HELPERS + 1), [hp("m" * (g2.HELPER_MODULE_MAX_CHARS + 1))], [hp(nbytes=-1)], [hp(sha="a" * 64)], [hp(sha="sha256:" + "z" * 64)]):
        with pytest.raises(g0.MoistureOutputError, match="external_helpers"):
            g2.attribution_from_dict({**base, "external_helpers": bad})
    # the WORST admitted attribution (every string at its closed maximum, the maximum helpers) still fits `v1 model + V_MANIFEST_EXTRA_BYTES`
    src = {k: ("a" * g2.SOURCE_VALUE_MAX_CHARS if type(v) is str and k not in ("kind",) else v) for k, v in base["source"].items()}
    worst = g2.attribution_from_dict({**base, "source": src, "external_helpers": [hp("m" * g2.HELPER_MODULE_MAX_CHARS, "sha256:" + "f" * 64, 1 << 40)] * g2.MAX_HELPERS})
    rd = read(out, 1)
    meta, ident = g1.FloatingFrameMeta(day=1, cadence_n=1, origin_ordinal=0, physical_t_s=1.0, origin_date=None, date=None), g1.FloatingFrameIdentity(**rd.manifest["identity"])
    edges = np.array(rd.manifest["depth_edges_m"])
    L, N = rd.manifest["n_layers"], rd.manifest["n_cells"]
    man = g2._build_manifest(meta, ident, worst, None, edges, L, N, tuple(rd.manifest["shape_yx"]), rd.manifest["cell_area_m2"], rd.manifest["n_active"], {k: "0" * 64 for k in (g1.WATER, g1.DEPTH, g1.ACTIVE)})
    assert len(g0._canon(man)) <= g2._manifest_model_bytes(L, 0)


def _good_attribution(tmp_path, leaf):
    out, _s = _lineage(tmp_path, leaf, with_generation=True, days=1)
    return g2.attribution_from_dict(read(out, 1).manifest["vegetation"])


def _owned_peak(fn):
    import tracemalloc

    tracemalloc.start()
    try:
        fn()
    except g0.MoistureOutputError:
        pass
    else:
        raise AssertionError("accepted")
    finally:
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return peak


def test_constructor_and_adapter_refuse_huge_caller_data_with_tiny_owned_peaks(tmp_path, leaf):
    import dataclasses

    good = _good_attribution(tmp_path, leaf)
    base = good.as_dict()
    evil_src = tuple(("unknown" + str(i), "x") for i in range(100000))  # ALL data built before the tracer
    evil_h = [base["external_helpers"][0]] * 100000
    evil_raw_src = {"kind": "plant_session", **{f"k{i}": "x" for i in range(100000)}}
    evil_tuple_h = tuple(tuple(h.values()) for h in [base["external_helpers"][0]] * 100000)
    evil_src_list = list(evil_src)
    cases = {
        "ctor_source": lambda: dataclasses.replace(good, source=evil_src),
        "ctor_helpers": lambda: dataclasses.replace(good, external_helpers=evil_tuple_h),
        "ctor_source_list": lambda: dataclasses.replace(good, source=evil_src_list),
        "adapter_helpers": lambda: g2.attribution_from_dict({**base, "external_helpers": evil_h}),
        "adapter_source": lambda: g2.attribution_from_dict({**base, "source": evil_raw_src}),
    }
    for name, fn in cases.items():
        assert _owned_peak(fn) < 64 * 1024, name  # a structural refusal: no copy / sort / set of the 100000 rows


@pytest.mark.parametrize("field, value", [("generation", True), ("generation", 1.0), ("generation", 1.2), ("generation", "1"), ("current_property_generation", "0"), ("current_property_generation", False),
                                          ("current_property_generation", 0.0), ("generation", None)])
def test_adapter_counts_are_strict_integers_no_coercion(tmp_path, leaf, field, value):
    base = _good_attribution(tmp_path, leaf).as_dict()
    with pytest.raises(g0.MoistureOutputError):
        g2.attribution_from_dict({**base, field: value})


@pytest.mark.parametrize("bad", [True, 1.5, "10", None])
def test_helper_bytes_are_strict_integers(tmp_path, leaf, bad):
    base = _good_attribution(tmp_path, leaf).as_dict()
    h = dict(base["external_helpers"][0])
    h["bytes"] = bad
    with pytest.raises(g0.MoistureOutputError):
        g2.attribution_from_dict({**base, "external_helpers": [h]})


def test_malformed_source_and_helper_shapes_are_typed_refusals(tmp_path, leaf):
    import dataclasses

    good = _good_attribution(tmp_path, leaf)
    base = good.as_dict()
    for src in (None, [], "x", {"kind": "nope"}, {}, {"kind": 5}, {"kind": "plant_session"}, (("kind", "plant_session"),)):
        with pytest.raises(g0.MoistureOutputError):
            g2.attribution_from_dict({**base, "source": src})
    for h in (None, "x", [None], [{"module": "m"}], [[1, 2, 3]], 7):
        with pytest.raises(g0.MoistureOutputError):
            g2.attribution_from_dict({**base, "external_helpers": h})
    with pytest.raises(g0.MoistureOutputError):
        g2.attribution_from_dict({k: v for k, v in base.items() if k != "recipe_id"})
    with pytest.raises(g0.MoistureOutputError):
        g2.attribution_from_dict([])
    for kw in ({"source": [("kind", "initial_state")]}, {"source": (("kind", "initial_state"), 5)}, {"source": ((1, 2),)}, {"external_helpers": [("m", "sha256:" + "a" * 64, 1)]},
               {"external_helpers": (("m", "sha256:" + "a" * 64),)}, {"external_helpers": ((None, None, None),)}):
        with pytest.raises(g0.MoistureOutputError):
            dataclasses.replace(good, **kw)
    # a free-text producer `note` is ignored by the adapter (declared); the genuine record round-trips unchanged
    withnote = {**base, "source": {**base["source"], "note": "free text"}}
    assert g2.attribution_from_dict(withnote) == good and g2.attribution_from_dict(base) == good


def test_resume_refuses_a_past_physical_time_beyond_the_state_or_running_backwards(tmp_path, leaf):
    out, states = _lineage(tmp_path, leaf, with_generation=True)
    last = states[-1]
    frame1, frame2 = (tmp_path / "o" / g1.frame_name(d) for d in (1, 2))
    raw1 = _reseal(frame1, lambda m: m.__setitem__("physical_t_s", float(last.t_s) + 1.0))
    with pytest.raises(g0.MoistureOutputError, match="physical time"):
        fov.VegetationMoistureOutput.resume(tmp_path / "o", last, expected_parent=fd.expect_parent(last), limits=LIM)
    (frame1 / g1.MANIFEST).write_bytes(raw1)
    raw2 = _reseal(frame2, lambda m: m.__setitem__("physical_t_s", 0.0))  # earlier than frame 1: backwards
    with pytest.raises(g0.MoistureOutputError, match="physical time"):
        fov.VegetationMoistureOutput.resume(tmp_path / "o", last, expected_parent=fd.expect_parent(last), limits=LIM)
    (frame2 / g1.MANIFEST).write_bytes(raw2)
    assert fov.VegetationMoistureOutput.resume(tmp_path / "o", last, expected_parent=fd.expect_parent(last), limits=LIM).last_day == 3
