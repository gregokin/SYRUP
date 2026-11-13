"""Actual-GPU test DEFINITIONS of the CUDA lifecycle physical file (WRITTEN, NOT RUN by their author: no GPU/CuPy/NVRTC/nvidia-smi was touched; no device pass is claimed). Opt-in (SYRUP_RELEASE_GPU=1).

FIXTURES are the genuine matched CPU/GPU fixtures of the accepted GPU daily definitions (`Side`: the CPU_C3 closed shallow PIT that genuinely completes and retains standing water, and the DRY plane; `matched_pair` proves the matched
deep landscape BEFORE any comparison): a real native storm -> genuine completion -> selected refresh -> accepted Boundary -> real bridge days. Nothing is fabricated (no capability, generation, owner, held class, quiet state).
They cover: Dense and Windowed boundary and CLOSED-day round trips with bitwise continuation against the uninterrupted device run (and the matched CPU twin within the declared float64 tolerance); T counted once; the CURRENT-device
gate before any reader/IO; two-pool admission (zero / need-1 / exact / reserve once) and the exact published load need; resealed tampers; cross-family refusals; head purity; the same-GPU fresh-process restore with every original
constructor DISABLED before the static recipe; and the seven-day N=3 chain with a real intervening event + property refresh (pause before/after, writer resume, frames verified, never overwritten).
Declared limits (not claimed): pending storm / pending recession / in-flight day files; large N; a whole-wet-day allocator trace; GPU numerics beyond the declared tolerance; a restored vertical is not an owner."""
# ruff: noqa: RUF059
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import chain
import numpy as np
import pytest
import tamper_cases

_T = Path(__file__).resolve().parents[1]
BIG = 1 << 40
IDENT = {"case": "s2cuda", "v": 1}
TOL = {"rtol": 1e-9, "atol": 1e-12}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _T / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bd = _load("bridge_gpu_defs_for_lifecycle_file", "phase2_floating_event_daily_bridge_cuda/test_gpu_definitions.py")
ld = bd.ld
host = bd.host

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_daily as fd
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_lifecycle_checkpoint as cpul
from maple_syrup import floating_lifecycle_checkpoint_cuda as lpcc
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import pond_checkpoint as pc

MEM = fsl.FloatingMemoryError


@pytest.fixture(scope="module")
def cp():
    return _gpu()


def _gpu():
    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA lifecycle file is not exercised (no claim)")
    return backend.cupy_module()


class DevSide(bd.Side):
    def complete(self, occ, s):
        return ld.complete(occ, s)


def sides(tmp, windowed, kw=None, pref=False):
    kw = bd.pit_kwargs() if kw is None else kw
    g, c = DevSide(tmp / "g", kw, True, windowed, pref), bd.Side(tmp / "c", kw, False, windowed, pref)
    bd.assert_matched(g.ev, g.st, g.vert, c.ev, c.st, c.vert, windowed)
    g.boundary()
    c.boundary()
    return g, c


def device_id(cp):
    return int(cp.cuda.Device().id)


def save(g, path, closed=None, **q):
    q = {"available_bytes_fn": g.rdr, "host_available_bytes_fn": lambda: BIG, **q}
    return lpcc.save_floating_lifecycle_checkpoint_cuda(path, g.lin, g.factory, IDENT, closed=closed, original_topology=g.ev._initial_topology, **q)


def load(cp, g, path, sv, partition=None, **q):
    q = {"available_bytes_fn": g.rdr, "host_available_bytes_fn": lambda: BIG, **q}
    return lpcc.load_floating_lifecycle_checkpoint_cuda(path, g.factory, IDENT, sv.manifest_sha256, original_topology=g.ev._initial_topology, expected_device_id=device_id(cp), partition=partition, **q)


def pack(b):
    r = b.root
    return {"W": host(r.water_state.water_m), "J": host(r.history.wetting_m), "I": host(r.history.cumulative_intake_m), "D": host(r.depth_m), "H": host(b.surface_depth_m), "qx": host(b.qx_m2_s), "qy": host(b.qy_m2_s)}


def same(a, b):
    assert set(a) == set(b) and all(a[k].tobytes() == b[k].tobytes() for k in a), [k for k in a if a[k].tobytes() != b[k].tobytes()]


def day_from(g, lin, boundary, ordinal, request=None):
    st = br.open_day(lin, boundary, partition=bd.PART, run_digest=bd.RUN, available_bytes_fn=g.rdr, **g.hk)
    n = st.ctx.n_cells
    pet, cov = g.arrays(n)
    res = br.day_action(st, plan=bd.rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=bd.RUN, ordinal=ordinal), pet_terrain_m_s=pet, cover=cov, available_bytes_fn=g.rdr, event_factory=g.factory, day_control=bd.CTL, **g.hk)
    req = g.xp.zeros((st.ctx.n_layers, n)) if request is None else request
    t = fd.request_floating_withdrawal(res.state, expected_parent=fd.expect_parent(res.state), request_m=req, expected_sequence=0, available_bytes_fn=g.rdr, **g.hk)
    return t, fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), available_bytes_fn=g.rdr, **g.hk).state


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_boundary_round_trip_restores_current_w_j_i_d_h_q_and_continues_bitwise(cp, tmp_path, windowed):
    g, c = sides(tmp_path, windowed)
    before, head0, lin0 = pack(g.b), g.lin.head, g.lin
    sv = save(g, tmp_path / "ck")
    assert g.lin.head is head0 and sv.kind == "boundary"  # a save changes no head
    same(before, pack(g.b))
    r = load(cp, g, tmp_path / "ck", sv)
    assert r.kind == "boundary" and r.closed is None and r.lineage is not lin0 and r.lineage.head is r.head and r.head.seq == g.b.seq and r.head.ledger == g.b.ledger and r.head.cursor == g.b.cursor
    assert r.head.root.host_active is not None and r.head.root.edges_host is not None and r.head.root.host_depth is not None  # sealed host mask/edges/D0 carried
    same(before, pack(r.head))  # exact current W/J/I/D, surface and faces
    assert [f for f in cpul.RestoredLifecycle.__dataclass_fields__] == ["lineage", "head", "closed", "kind", "manifest_sha256"]  # the minimal default return
    (tg, cg), (tr, cr) = day_from(g, g.lin, g.b, 0), day_from(g, r.lineage, r.head, 0)  # real native day: original vs restored (the same device)
    assert cg.execution_digest == cr.execution_digest and cg.digest == cr.digest and host(cg.water.water_m).tobytes() == host(cr.water.water_m).tobytes()
    res_c = c.day(c.b, 0)[1]  # the matched CPU twin day agrees within the declared tolerance
    np.testing.assert_allclose(host(cr.water.water_m), np.asarray(c.finish(res_c)[1].water.water_m), **TOL)
    assert tg.actual_m3 == tr.actual_m3


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_closed_day_round_trip_keeps_identity_cursor_ledger_t_once_and_hands_back_equal(cp, tmp_path, windowed):
    g, c = sides(tmp_path, windowed)
    st, res = g.day(g.b, 0)
    n, L = st.ctx.n_cells, st.ctx.n_layers
    t, closed = g.finish(res, g.xp.full((L, n), 2e-2))  # partial T, once
    assert 0.0 < t.actual_m3 < t.requested_m3
    head0 = g.lin.head
    assert type(head0) is lc.DailyHold
    sv = save(g, tmp_path / "ck", closed=closed)
    assert g.lin.head is head0 and sv.kind == "closed_day"  # nothing handed back, nothing advanced
    r = load(cp, g, tmp_path / "ck", sv, partition=bd.PART)
    assert r.kind == "closed_day" and type(r.head) is lc.DailyHold and r.lineage.head is r.head
    rc_ = r.closed
    assert rc_.execution_digest == closed.execution_digest and rc_.digest == closed.digest and rc_.revision == closed.revision and rc_.ledger == closed.ledger and rc_.cursor == closed.cursor
    assert rc_.frame_identity() == closed.frame_identity()  # the every-N writer's due-day identity is unchanged
    assert host(rc_.water.water_m).tobytes() == host(closed.water.water_m).tobytes() and host(rc_.history.wetting_m).tobytes() == host(closed.history.wetting_m).tobytes()
    bg = g.hand(closed)
    br2 = br.hand_back(r.lineage, rc_, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)
    assert br2.ledger == bg.ledger and abs(br2.ledger["residual_m3"]) <= br2.ledger["residual_bound_m3"]  # the day segment counted ONCE
    same(pack(bg), pack(br2))
    (_t1, c1), (_t2, c2) = day_from(g, g.lin, bg, 1), day_from(g, r.lineage, br2, 1)
    assert c1.execution_digest == c2.execution_digest and host(c1.water.water_m).tobytes() == host(c2.water.water_m).tobytes()


def test_current_device_gate_precedes_every_reader_path_check_and_factory_for_save_and_load(cp, tmp_path, monkeypatch):
    g, _c = sides(tmp_path, False)
    sv = save(g, tmp_path / "ck")
    reads, made = [], []

    def reader():
        reads.append(1)
        return BIG

    real = fcu.current_device
    monkeypatch.setattr(fcu, "current_device", lambda c: real(c) + 1)  # the current device is NOT the vertical's / the expected device: the real gate, an ordering control
    with pytest.raises(lpcc.LifecycleCheckpointCudaError, match="device gate"):
        lpcc.save_floating_lifecycle_checkpoint_cuda(tmp_path / "never", g.lin, g.factory, IDENT, original_topology=g.ev._initial_topology, available_bytes_fn=reader, host_available_bytes_fn=reader)
    with pytest.raises(lpcc.LifecycleCheckpointCudaError, match="device gate"):
        lpcc.load_floating_lifecycle_checkpoint_cuda(tmp_path / "ck", lambda *a, **k: made.append(1), IDENT, sv.manifest_sha256, original_topology=g.ev._initial_topology, expected_device_id=device_id(cp),
                                                     available_bytes_fn=reader, host_available_bytes_fn=reader)
    assert reads == [] and made == [] and not (tmp_path / "never").exists()


def test_two_pool_admission_zero_need_minus_one_exact_reserve_once_and_the_published_load_need(cp, tmp_path, monkeypatch):
    from maple_syrup import floating_context_checkpoint_cuda as fcxc

    g, _c = sides(tmp_path, False)
    calls = []
    real = fcxc._admit

    def spy(cp_, dev, d, h, q, what):
        calls.append((int(d), int(h), what))
        return real(cp_, dev, d, h, q, what)

    monkeypatch.setattr(fcxc, "_admit", spy)
    sv = save(g, tmp_path / "ck")
    d_save, h_save, _w = calls[-1]  # the composed cold-carrier + child save admission
    for kw, label in (({"available_bytes_fn": lambda: 0}, "device zero"), ({"host_available_bytes_fn": lambda: 0}, "host zero"), ({"available_bytes_fn": lambda: d_save - 1}, "device need-1"),
                      ({"host_available_bytes_fn": lambda: h_save + fcu.PY_ALLOWANCE_BYTES - 1}, "host need-1"), ({"available_bytes_fn": lambda: d_save, "reserve_bytes": 1}, "device reserve once"),
                      ({"memory_budget_bytes": 0}, "zero device budget"), ({"host_memory_budget_bytes": 0}, "zero host budget")):
        t = tmp_path / f"r_{label.replace(' ', '_')}"
        with pytest.raises(MEM):
            save(g, t, **kw)
        assert not t.exists() and not list(tmp_path.glob(f".{t.name}*")) and g.lin.head is g.b, label
    need = lpcc.lifecycle_load_need_bytes_cuda(tmp_path / "ck", IDENT, sv.manifest_sha256, expected_device_id=device_id(cp), available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)
    reads, real_load = [], np.load

    class Reached(BaseException):
        pass

    def trip(*a, **k):
        reads.append(1)
        raise Reached

    monkeypatch.setattr(np, "load", trip)
    for kw, label in (({"available_bytes_fn": lambda: need.device_total_bytes - 1}, "device need-1"), ({"host_available_bytes_fn": lambda: need.host_total_bytes - 1}, "host need-1"),
                      ({"available_bytes_fn": lambda: 0}, "device zero"), ({"host_available_bytes_fn": lambda: 0}, "host zero")):
        with pytest.raises(MEM):
            load(cp, g, tmp_path / "ck", sv, **kw)
        assert reads == [], label  # refused before any payload array
    with pytest.raises(Reached):  # the exact published need passes the stage-1 admission and reaches the first array read
        load(cp, g, tmp_path / "ck", sv, available_bytes_fn=lambda: need.device_total_bytes, host_available_bytes_fn=lambda: need.host_total_bytes)
    monkeypatch.setattr(np, "load", real_load)
    assert need.stages["child"]["table_bytes"] > 0


def _reseal(src, tmp, name, outer=None, child=None):
    """Copy `src`, edit the outer manifest and/or the child manifest and RECOMPUTE every hash and the seals (the attacker's best case; a recomputed seal is not authenticity)."""
    from maple_syrup import floating_context_checkpoint as cc

    d = tmp / name
    shutil.copytree(src, d)
    cm = json.loads((d / "child" / fc.MANIFEST_NAME).read_text())
    if child:
        child(cm)
        (d / "child" / fc.MANIFEST_NAME).write_text(json.dumps(cm))
    om = json.loads((d / lpcc.MANIFEST_NAME).read_text())
    rec = om["child"]["files"][fc.MANIFEST_NAME]
    rec["size"] = (d / "child" / fc.MANIFEST_NAME).stat().st_size
    rec["sha256"] = pc._hash_file(d / "child" / fc.MANIFEST_NAME, 1 << 40)[0]
    om["child"]["manifest_sha256"] = rec["sha256"]
    if outer:
        outer(om)
    sealed = cc._seal(om)
    (d / lpcc.MANIFEST_NAME).write_text(pc._json_dumps(sealed))
    return d, sealed["manifest_sha256"]


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_wrong_seal_resealed_tampers_foreign_profiles_and_no_clobber_leave_the_head_and_the_target_retryable(cp, tmp_path, windowed):
    g, _c = sides(tmp_path, windowed)
    st, res = g.day(g.b, 0)
    t, closed = g.finish(res)
    sv = save(g, tmp_path / "ck", closed=closed)
    head0 = g.lin.head
    from types import SimpleNamespace

    def attempt(d, seal, rx):
        with pytest.raises(lpcc.LifecycleCheckpointCudaError if rx else Exception, match=rx or None):
            load(cp, g, d, SimpleNamespace(manifest_sha256=seal), partition=bd.PART)

    attempt(tmp_path / "ck", "0" * 64, "EXPECTED digest")  # the file is never its own external identity
    D = json.loads((tmp_path / "ck" / lpcc.MANIFEST_NAME).read_text())["lifecycle"]
    assert D["head"]["cursor"] is None and D["daily"]["cursor"] is not None  # why a `head.cursor = None` edit would be a NO-OP: the meaningful cursor control is the DAILY cursor
    # (name, edit kwargs, expected refusal regex, seal the loader is given: "own" = the forged file's own valid seal, "writer" = the seal the original writer returned)
    cases = tamper_cases.lifecycle_cases(D, windowed)
    orig_manifest = json.loads((tmp_path / "ck" / lpcc.MANIFEST_NAME).read_text())
    for name, kw, rx, which in cases:
        d, seal = _reseal(tmp_path / "ck", tmp_path, f"bad_{name}", **kw)
        forged = json.loads((d / lpcc.MANIFEST_NAME).read_text())
        assert forged != orig_manifest and seal != sv.manifest_sha256, f"{name}: the mutation is a NO-OP (the forged manifest equals the original)"  # no false no-op controls
        with pytest.raises((lpcc.LifecycleCheckpointCudaError, fc.FloatingCheckpointError, lc.LifecycleError, br.BridgeError), match=rx) as info:
            load(cp, g, d, SimpleNamespace(manifest_sha256=sv.manifest_sha256 if which == "writer" else seal), partition=bd.PART)
        assert g.lin.head is head0, f"{name}: the head moved ({info.value})"
    with pytest.raises(lpcc.LifecycleCheckpointCudaError, match="already exists"):
        save(g, tmp_path / "ck", closed=closed)  # never overwrites
    with pytest.raises(lpcc.LifecycleCheckpointCudaError, match="partition"):
        load(cp, g, tmp_path / "ck", sv)  # a closed-day file needs the caller's partition
    ok = load(cp, g, tmp_path / "ck", sv, partition=bd.PART)  # the untouched file still restores after every refusal
    assert ok.closed.execution_digest == closed.execution_digest and g.lin.head is head0
    # a CPU `/1` loader refuses the CUDA file by profile name (no migration)
    with pytest.raises(cpul.LifecycleCheckpointError):
        cpul.load_floating_lifecycle_checkpoint(tmp_path / "ck", g.factory, IDENT, sv.manifest_sha256, original_topology=g.ev._initial_topology, partition=bd.PART, available_bytes_fn=lambda: BIG)


def test_unsaveable_heads_are_refused_without_a_file_or_a_head_change(cp, tmp_path):
    g, _c = sides(tmp_path, False)
    occ, s = lc.begin_occurrence(g.lin, g.b, g.factory, available_bytes_fn=g.rdr, **g.hk)  # a running native occurrence (a storm / mid-trial) is not a closed head
    head = g.lin.head
    with pytest.raises(lpcc.LifecycleCheckpointCudaError):
        save(g, tmp_path / "occ")
    assert g.lin.head is head and not (tmp_path / "occ").exists() and not list(tmp_path.glob(".occ.partial-*")) and s is not None and occ is head


def _worker(mode, work, windowed):
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve().parent / "gpu_lifecycle_worker.py"), mode, str(work), "1" if windowed else "0"], capture_output=True, text=True, env=env, timeout=3600, check=False)
    assert r.returncode == 0, r.stderr[-3000:]


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_same_gpu_fresh_process_restore_without_original_bed_owner_probe_or_provider(cp, tmp_path, windowed):
    """Process A saves (lifecycle file at a CLOSED day + recipe.json + the caller's recorded seal) and continues one day; process B (NEW interpreter) disables every original bed/owner/probe/provider/pair constructor BEFORE it
    reads the recipe, restores from the file + recipe + the recorded seal alone, runs the same day and dumps; the results are bitwise equal. Claim limited to the worker's tripwires (listed in `tripwires.json`)."""
    _worker("save", tmp_path, windowed)
    _worker("load", tmp_path, windowed)
    a, b = np.load(tmp_path / "save_final.npz"), np.load(tmp_path / "load_final.npz")
    assert set(a.files) == set(b.files) and all(a[k].tobytes() == b[k].tobytes() for k in a.files)
    assert "make_pair" in json.loads((tmp_path / "tripwires.json").read_text())["disabled_before_recipe"]


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_seven_day_n3_chain_with_a_real_event_and_refresh_pauses_before_the_due_day_and_equals_the_uninterrupted_run(cp, tmp_path, monkeypatch, windowed):
    kw = bd.dry_kwargs(splash=True)
    ref_side = DevSide(tmp_path / "ref", kw, True, windowed)
    run_side = DevSide(tmp_path / "run", kw, True, windowed)
    ref = chain.run(ref_side, tmp_path / "out_ref", monkeypatch, host, save=lambda lin, closed: None, load=lambda sv: None)
    original = run_side.ev._initial_topology

    def do_save(lin, closed):
        return lpcc.save_floating_lifecycle_checkpoint_cuda(tmp_path / "ck", lin, run_side.factory, IDENT, closed=closed, original_topology=original, available_bytes_fn=run_side.rdr, host_available_bytes_fn=lambda: BIG)

    def do_load(sv):
        return lpcc.load_floating_lifecycle_checkpoint_cuda(sv.path, run_side.factory, IDENT, sv.manifest_sha256, original_topology=original, expected_device_id=device_id(cp), partition=chain.PART,
                                                            available_bytes_fn=run_side.rdr, host_available_bytes_fn=lambda: BIG)

    got = chain.run(run_side, tmp_path / "out_run", monkeypatch, host, save=do_save, load=do_load, pause_after=1)
    assert got.restored is not None and got.restored.kind == "closed_day"
    chain.check_equal(ref, got)


# ---- correction01 definitions (D1 provenance round trip + re-save, D2 genuinely omitted readers, D3 resource-error class on the closed-day restore) --------------------------------------------------------
def _provenance(path):
    return json.loads((path / lpcc.MANIFEST_NAME).read_text())["lifecycle"]["head"]["provenance"]


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_a_genuine_cuda_boundary_and_first_closed_day_round_trip_their_provenance_and_the_restored_children_save_again(cp, tmp_path, windowed):
    from maple_syrup import floating_held_context_cuda as fhc

    g, _c = sides(tmp_path, windowed)
    prov0 = dict(g.b.provenance)
    assert prov0["backend"] == "cuda" and prov0["kernel"] == fhc.kernel_descriptor()  # what the GPU close really wrote (the keys the CPU validator rejects)
    sv = save(g, tmp_path / "ck")
    assert _provenance(tmp_path / "ck") == prov0
    r = load(cp, g, tmp_path / "ck", sv)
    assert dict(r.head.provenance) == prov0  # the ORIGINAL canonical dict, nothing dropped
    sv2 = lpcc.save_floating_lifecycle_checkpoint_cuda(tmp_path / "ck2", r.lineage, g.factory, IDENT, original_topology=g.ev._initial_topology, available_bytes_fn=g.rdr, host_available_bytes_fn=lambda: BIG)
    assert _provenance(tmp_path / "ck2") == prov0 and sv2.kind == "boundary"  # the restored BOUNDARY saves again
    r2 = load(cp, g, tmp_path / "ck2", sv2)
    same(pack(r.head), pack(r2.head))
    # the FIRST closed DailyHold: its (opened-from-boundary) provenance also round-trips and the restored closed day saves again
    _st, res = g.day(g.b, 0)
    _t, closed = g.finish(res)
    svc = save(g, tmp_path / "ckc", closed=closed)
    rc_ = load(cp, g, tmp_path / "ckc", svc, partition=bd.PART)
    assert dict(rc_.head.provenance) == dict(g.lin.head.provenance)
    svc2 = lpcc.save_floating_lifecycle_checkpoint_cuda(tmp_path / "ckc2", rc_.lineage, g.factory, IDENT, closed=rc_.closed, original_topology=g.ev._initial_topology, available_bytes_fn=g.rdr,
                                                        host_available_bytes_fn=lambda: BIG)
    rc2 = load(cp, g, tmp_path / "ckc2", svc2, partition=bd.PART)
    assert rc2.closed.execution_digest == closed.execution_digest and rc2.closed.digest == closed.digest


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_omitted_readers_use_the_real_defaults_for_save_need_and_load_and_the_bytes_equal_the_explicit_readers(cp, tmp_path, windowed):
    """No `available_bytes_fn` / `host_available_bytes_fn` at all (the `None` real-reader contract): the constructors that require callables must receive the resolved defaults, not `None`."""
    g, _c = sides(tmp_path, windowed)
    head0 = g.lin.head
    sv = lpcc.save_floating_lifecycle_checkpoint_cuda(tmp_path / "ck", g.lin, g.factory, IDENT, original_topology=g.ev._initial_topology)  # omitted readers, omitted budgets
    assert g.lin.head is head0
    need = lpcc.lifecycle_load_need_bytes_cuda(tmp_path / "ck", IDENT, sv.manifest_sha256, expected_device_id=device_id(cp))
    assert need.device_total_bytes > 0 and need.host_total_bytes > 0
    r = lpcc.load_floating_lifecycle_checkpoint_cuda(tmp_path / "ck", g.factory, IDENT, sv.manifest_sha256, original_topology=g.ev._initial_topology, expected_device_id=device_id(cp))
    same(pack(g.b), pack(r.head))
    sv_explicit = save(g, tmp_path / "ck_explicit")
    assert json.loads((tmp_path / "ck" / lpcc.MANIFEST_NAME).read_text())["lifecycle"] == json.loads((tmp_path / "ck_explicit" / lpcc.MANIFEST_NAME).read_text())["lifecycle"] and sv_explicit.kind == "boundary"
    # a caller's reader is retained: a zero host reader still refuses (the default is never substituted for an explicit refusal)
    with pytest.raises(MEM):
        lpcc.save_floating_lifecycle_checkpoint_cuda(tmp_path / "never", g.lin, g.factory, IDENT, original_topology=g.ev._initial_topology, host_available_bytes_fn=lambda: 0)
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize("error", ["typed", "plain_memory_error"])
def test_a_resource_refusal_inside_the_closed_day_restore_keeps_its_class_installs_no_head_and_a_retry_restores(cp, tmp_path, monkeypatch, error):
    from maple_syrup import floating_event_daily_bridge_cuda as bridge_cuda

    g, _c = sides(tmp_path, False)
    _st, res = g.day(g.b, 0)
    _t, closed = g.finish(res)
    sv = save(g, tmp_path / "ck", closed=closed)
    real = bridge_cuda.fdc.admit
    state = {"n": 0}
    exc_type = MEM if error == "typed" else MemoryError  # a CuPy OutOfMemoryError is a MemoryError subclass too

    def flaky(column, d, h, q, what):
        if "closed-day restore" in what and state["n"] == 0:
            state["n"] += 1
            raise exc_type("injected resource refusal inside the closed-day restore (a changing reader)")
        return real(column, d, h, q, what)

    monkeypatch.setattr(bridge_cuda.fdc, "admit", flaky)
    with pytest.raises(exc_type) as info:  # NOT a BridgeError / LifecycleCheckpointCudaError: a resource refusal is retryable with a larger reader
        load(cp, g, tmp_path / "ck", sv, partition=bd.PART)
    assert state["n"] == 1 and type(info.value) is exc_type
    r = load(cp, g, tmp_path / "ck", sv, partition=bd.PART)  # the retry restores a closed day (no head was installed by the refused attempt)
    assert r.closed.execution_digest == closed.execution_digest and r.lineage.head is r.head
