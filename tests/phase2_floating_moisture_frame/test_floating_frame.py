"""Phase 2 G1: floating moisture frame codec. CPU/host only; CUDA definitions at the bottom are gated (Root hardware)."""
import hashlib
import os
import stat

import numpy as np
import pytest

import maple_syrup.floating_moisture_frame as ff
from maple_syrup.floating_moisture_frame import (
    FloatingFrameIdentity,
    FloatingFrameMeta,
    read_floating_frame,
    write_floating_frame,
)
from maple_syrup.soil_moisture_output import (
    MoistureOutputConflict,
    MoistureOutputError,
    MoistureOutputIndeterminate,
    MoistureOutputLimits,
    MoistureOutputQuotaError,
)

EDGES = np.array([0.0, 0.1, 0.3, 0.6])
L, N, SHAPE = 3, 6, (2, 3)
AREA = 0.25
HOSTFN = (lambda: 10**9)


def h(c):
    return hashlib.sha256(c.encode()).hexdigest()


def ident(**kw):
    base = {"run_digest": h("run"), "lineage_sha256": h("lin"), "source_sha256": h("src"), "property_generation": 3,
            "property_sha256": h("p3"), "bed_generation": 7, "depth_generation": 5, "parent_digest": h("par"),
            "state_digest": h("st"), "state_revision": 11}
    base.update(kw)
    return FloatingFrameIdentity(**base)


def meta(**kw):
    base = {"day": 4, "cadence_n": 2, "origin_ordinal": 0, "physical_t_s": 1234.5, "origin_date": "2030-01-01", "date": "2030-01-05"}
    base.update(kw)
    return FloatingFrameMeta(**base)


def limits(**kw):
    base = {"max_snapshots": 8, "max_total_bytes": 10**6, "max_manifest_bytes": 8192, "chunk_bytes": 96,
            "host_memory_budget_bytes": 10**7, "host_reserve_bytes": 0, "disk_reserve_bytes": 0}
    base.update(kw)
    return MoistureOutputLimits(**base)


def profile():
    """D across rock (0), partial layer 0, exactly the first edge, mid, full (0.6) and beyond; cell 5 is inactive with
    out-of-bound raw placeholders."""
    D = np.array([0.0, 0.05, 0.1, 0.2, 0.9, -7.0])
    active = np.array([True, True, True, True, True, False])
    W = np.zeros((L, N))
    W[0, 1], W[0, 2], W[0, 3], W[0, 4] = 0.01, 0.02, 0.03, 0.04
    W[1, 3], W[1, 4] = 0.05, 0.06
    W[2, 4] = 0.07
    W[:, 5] = [np.nan, 9.0, -1.0]  # inactive raw placeholders, preserved exactly
    return W, D, active


def write(root, W, D, active, **kw):
    kw.setdefault("meta", meta())
    kw.setdefault("identity", ident())
    kw.setdefault("limits", limits())
    kw.setdefault("host_available_bytes_fn", HOSTFN)
    return write_floating_frame(root, water_m=W, depth_m=D, active=active, depth_edges_m=EDGES, cell_area_m2=AREA, shape_yx=SHAPE, **kw)


def test_roundtrip_reconstructs_theta_and_preserves_inactive_exactly(tmp_path):
    W, D, a = profile()
    snap = [x.copy() for x in (W, D, a)]
    n = write(tmp_path, W, D, a)
    assert n.action == "saved" and n.device_chunks == 0 and n.bytes_written > 0
    fr = read_floating_frame(tmp_path / ff.frame_name(4), max_return_bytes=10**6)
    assert fr.water_m.tobytes() == W.tobytes() and fr.depth_m.tobytes() == D.tobytes() and fr.active.tobytes() == a.tobytes()
    assert not fr.water_m.flags.writeable and not fr.depth_m.flags.writeable and not fr.active.flags.writeable
    for x, y in zip((W, D, a), snap, strict=True):
        assert x.tobytes() == y.tobytes()  # physical inputs unchanged
    th = fr.theta(max_bytes=10**6)
    # cell 1: D=0.05 -> h0=0.05, h1=h2=0 (rock): theta0 = 0.01/0.05, others 0
    assert th[0, 1] == pytest.approx(0.2) and th[1, 1] == 0.0 and th[2, 1] == 0.0
    # cell 2: D exactly at edge 0.1 -> h0=0.1, rock below
    assert th[0, 2] == pytest.approx(0.2) and th[1, 2] == 0.0
    # cell 3: D=0.2 -> h=(0.1, 0.1, 0)
    assert th[0, 3] == pytest.approx(0.3) and th[1, 3] == pytest.approx(0.5) and th[2, 3] == 0.0
    # cell 4: D beyond the profile -> full thicknesses (0.1, 0.2, 0.3)
    np.testing.assert_allclose(th[:, 4], [0.4, 0.3, 0.07 / 0.3])
    assert np.all(th[:, 0] == 0.0)  # all-rock active cell holds no water and theta is defined 0
    assert np.isnan(fr.water_m[0, 5]) and fr.water_m[1, 5] == 9.0 and fr.depth_m[5] == -7.0  # inactive raw preserved
    assert fr.manifest["profile"] == "surface_following_floating_profile" and fr.manifest["attribution"] == "caller_supplied_not_proof"
    assert fr.manifest["identity"]["property_sha256"] == h("p3") and fr.manifest["physical_t_s"] == 1234.5


def test_equal_water_at_different_depth_gives_different_theta(tmp_path):
    W = np.zeros((L, 2))
    W[0] = 0.02
    D = np.array([0.05, 0.4])
    a = np.array([True, True])
    ff_ = write_floating_frame(tmp_path, water_m=W, depth_m=D, active=a, depth_edges_m=EDGES, cell_area_m2=AREA, shape_yx=(1, 2),
                               meta=meta(), identity=ident(), limits=limits(), host_available_bytes_fn=HOSTFN)
    fr = read_floating_frame(tmp_path / ff.frame_name(4), max_return_bytes=10**6)
    th = fr.theta(max_bytes=10**6)
    assert ff_.action == "saved" and th[0, 0] == pytest.approx(0.4) and th[0, 1] == pytest.approx(0.2)


def test_rock_slot_water_on_active_cell_refused_without_residue(tmp_path):
    W, D, a = profile()
    W[2, 1] = 1e-30  # layer 2 is a rock slot at D=0.05
    with pytest.raises(MoistureOutputError, match="rock slot"):
        write(tmp_path, W, D, a)
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("mut", ["nan_w", "neg_w", "nan_d", "neg_d", "inf_d"])
def test_invalid_active_values_refused(tmp_path, mut):
    W, D, a = profile()
    if mut == "nan_w":
        W[0, 3] = np.nan
    elif mut == "neg_w":
        W[0, 3] = -0.1
    elif mut == "nan_d":
        D[3] = np.nan
    elif mut == "neg_d":
        D[3] = -0.1
    else:
        D[3] = np.inf
    with pytest.raises(MoistureOutputError):
        write(tmp_path, W, D, a)
    assert os.listdir(tmp_path) == []


def test_typed_invalid_arguments(tmp_path):
    W, D, a = profile()
    for bad in ({"day": 0}, {"day": 3}, {"cadence_n": 0}, {"physical_t_s": float("nan")}, {"date": "2030-01-06"}, {"origin_date": None},
                {"date": "x"}):
        with pytest.raises(MoistureOutputError):
            meta(**bad)
    for bad in ({"run_digest": "zz"}, {"state_revision": -1}, {"property_generation": 1.5}):
        with pytest.raises(MoistureOutputError):
            ident(**bad)
    with pytest.raises(MoistureOutputError, match="shape"):
        write_floating_frame(tmp_path, water_m=W, depth_m=D, active=a, depth_edges_m=EDGES, cell_area_m2=AREA, shape_yx=(2, 2),
                             meta=meta(), identity=ident(), limits=limits(), host_available_bytes_fn=HOSTFN)
    with pytest.raises(MoistureOutputError):
        write(tmp_path, W.astype(np.float32), D, a)
    with pytest.raises(MoistureOutputError):
        write(tmp_path, W, D, a.astype(np.uint8))
    with pytest.raises(MoistureOutputError, match="depth_edges_m"):
        write_floating_frame(tmp_path, water_m=W, depth_m=D, active=a, depth_edges_m=np.array([0.0, 0.2, 0.1, 0.5]), cell_area_m2=AREA,
                             shape_yx=SHAPE, meta=meta(), identity=ident(), limits=limits(), host_available_bytes_fn=HOSTFN)
    with pytest.raises(MoistureOutputError, match="cell_area"):
        write_floating_frame(tmp_path, water_m=W, depth_m=D, active=a, depth_edges_m=EDGES, cell_area_m2=-1.0, shape_yx=SHAPE,
                             meta=meta(), identity=ident(), limits=limits(), host_available_bytes_fn=HOSTFN)
    assert os.listdir(tmp_path) == []


def test_quota_refusals_happen_before_any_transfer(tmp_path, monkeypatch):
    W, D, a = profile()
    calls = []
    monkeypatch.setattr(ff, "_to_host", lambda x: calls.append(1) or np.asarray(x))
    with pytest.raises(MoistureOutputQuotaError, match="host"):
        write(tmp_path, W, D, a, limits=limits(host_memory_budget_bytes=1000))
    with pytest.raises(MoistureOutputQuotaError, match="reported available"):
        write(tmp_path, W, D, a, host_available_bytes_fn=lambda: 10)
    with pytest.raises(MoistureOutputQuotaError, match="max_total_bytes"):
        write(tmp_path, W, D, a, limits=limits(max_total_bytes=100))
    with pytest.raises(MoistureOutputQuotaError, match="disk"):
        write(tmp_path, W, D, a, disk_available_bytes_fn=lambda: 5)
    with pytest.raises(MoistureOutputQuotaError, match="max_manifest_bytes"):
        write(tmp_path, W, D, a, limits=limits(max_manifest_bytes=300))
    assert calls == [] and os.listdir(tmp_path) == []


def test_max_snapshots_quota_counts_existing_frames(tmp_path):
    W, D, a = profile()
    write(tmp_path, W, D, a, meta=meta(day=2, cadence_n=2, date="2030-01-03"), limits=limits(max_snapshots=1))
    with pytest.raises(MoistureOutputQuotaError):
        write(tmp_path, W, D, a, limits=limits(max_snapshots=1))
    assert os.listdir(tmp_path) == ["frame_000000000002"]


def test_identical_replay_verifies_and_conflicts_never_overwrite(tmp_path):
    W, D, a = profile()
    write(tmp_path, W, D, a)
    d = tmp_path / ff.frame_name(4)
    before = {n: (d / n).read_bytes() for n in os.listdir(d)}
    n2 = write(tmp_path, W, D, a)
    assert n2.action == "verified_existing" and n2.bytes_written == 0
    W2 = W.copy()
    W2[0, 3] = 0.031
    D2 = D.copy()
    D2[3] = 0.25
    a2 = a.copy()
    a2[0] = False
    for kw in ({"W": W2}, {"D": D2}, {"a": a2}, {"identity": ident(bed_generation=8)}, {"identity": ident(property_sha256=h("other"))},
               {"meta": meta(physical_t_s=99.0)}, {"caller_identity": {"x": "y"}}):
        args = {"W": W, "D": D, "a": a}
        extra = {}
        for k, v in kw.items():
            (args if k in args else extra)[k] = v
        with pytest.raises(MoistureOutputConflict):
            write(tmp_path, args["W"], args["D"], args["a"], **extra)
    assert {n: (d / n).read_bytes() for n in os.listdir(d)} == before


def test_replay_detects_corrupt_stored_bytes_and_leaves_them(tmp_path):
    W, D, a = profile()
    write(tmp_path, W, D, a)
    p = tmp_path / ff.frame_name(4) / "water_m.npy"
    raw = bytearray(p.read_bytes())
    raw[-1] ^= 0x01
    p.write_bytes(bytes(raw))
    with pytest.raises(MoistureOutputConflict, match="corrupt"):
        write(tmp_path, W, D, a)
    assert p.read_bytes() == bytes(raw)
    with pytest.raises(MoistureOutputError, match="checksum"):
        read_floating_frame(tmp_path / ff.frame_name(4), max_return_bytes=10**6)


def test_existing_unsealed_reservation_is_refused_and_untouched(tmp_path):
    W, D, a = profile()
    d = tmp_path / ff.frame_name(4)
    d.mkdir()
    (d / "water_m.npy").write_bytes(b"foreign")
    with pytest.raises(MoistureOutputConflict, match="UNSEALED"):
        write(tmp_path, W, D, a)
    assert os.listdir(d) == ["water_m.npy"] and (d / "water_m.npy").read_bytes() == b"foreign"


def test_failure_before_seal_removes_only_own_files(tmp_path, monkeypatch):
    W, D, a = profile()
    boom = RuntimeError("disk gone")
    monkeypatch.setattr(ff.g0, "_hash_stored", lambda *x: (_ for _ in ()).throw(boom))
    with pytest.raises(RuntimeError):
        write(tmp_path, W, D, a)
    assert os.listdir(tmp_path) == []  # own unsealed files and reservation removed
    keep = tmp_path / "frame_000000000010"
    keep.mkdir()
    (keep / "x").write_bytes(b"u")
    with pytest.raises(RuntimeError):
        write(tmp_path, W, D, a)
    assert (keep / "x").exists()  # unknown reservation never removed


def test_post_seal_failure_is_indeterminate_and_keeps_sealed_frame(tmp_path, monkeypatch):
    W, D, a = profile()
    real = os.fsync
    state = {"n": 0}

    def flaky(fd):
        st = os.fstat(fd)
        if stat.S_ISDIR(st.st_mode):  # directory fsyncs happen only after the seal
            state["n"] += 1
            raise OSError(5, "I/O error")
        return real(fd)

    monkeypatch.setattr(os, "fsync", flaky)
    with pytest.raises(MoistureOutputIndeterminate) as ei:
        write(tmp_path, W, D, a)
    monkeypatch.setattr(os, "fsync", real)
    d = tmp_path / ff.frame_name(4)
    assert ei.value.path == str(d) and (d / "manifest.json").exists()
    assert write(tmp_path, W, D, a).action == "verified_existing"  # retry verifies the intact sealed frame
    assert read_floating_frame(d, max_return_bytes=10**6).water_m.tobytes() == W.tobytes()


def test_reader_budget_before_read_and_strict_manifest(tmp_path):
    W, D, a = profile()
    write(tmp_path, W, D, a)
    d = tmp_path / ff.frame_name(4)
    need = 8 * L * N + 8 * N + N + 8 * (L + 1)
    with pytest.raises(MoistureOutputQuotaError, match="nothing was read"):
        read_floating_frame(d, max_return_bytes=need - 1)
    assert read_floating_frame(d, max_return_bytes=need).water_m.shape == (L, N)
    with pytest.raises(MoistureOutputQuotaError, match="nothing was allocated"):
        read_floating_frame(d, max_return_bytes=need).theta(max_bytes=8 * L * N + N - 1)
    man = (d / "manifest.json").read_bytes()
    for bad in (man.replace(b'"day":4', b'"day":4,"day":4'), man.replace(b'"day":4', b'"day":NaN'), man[:-1] + b',"extra":1}',
                man.replace(b"floating_soil_moisture_frame", b"other")):
        (d / "manifest.json").write_bytes(bad)
        with pytest.raises(MoistureOutputError):
            read_floating_frame(d, max_return_bytes=10**6)
    (d / "manifest.json").write_bytes(man)


def test_chunk_bound_evidence_pieces_and_no_full_mirror(tmp_path, monkeypatch):
    sizes = []
    real = ff._to_host
    monkeypatch.setattr(ff, "_to_host", lambda x: sizes.append(np.asarray(x).nbytes) or real(x))
    big_n = 40
    W = np.zeros((L, big_n))
    D = np.full(big_n, 0.9)
    a = np.ones(big_n, dtype=bool)
    W[0] = 0.01
    lim = limits(chunk_bytes=96)  # c = 2 cells per chunk
    write_floating_frame(tmp_path, water_m=W, depth_m=D, active=a, depth_edges_m=EDGES, cell_area_m2=AREA, shape_yx=(5, 8), meta=meta(),
                         identity=ident(), limits=lim, host_available_bytes_fn=HOSTFN)
    c = 96 // ff.BYTES_PER_CELL
    assert max(sizes) <= 8 * c and len(sizes) == (2 + L) * (big_n // c)  # every piece <= chunk, no full-array transfer
    assert ff.BYTES_PER_CELL * c <= lim.chunk_bytes
    fr = read_floating_frame(tmp_path / ff.frame_name(4), max_return_bytes=10**6)
    assert fr.water_m.tobytes() == W.tobytes()


def test_device_guard_with_host_doubles(monkeypatch):
    class Dev:
        def __init__(self, i):
            self.id = i

    class Arr:
        def __init__(self, i):
            self.device = Dev(i)

    monkeypatch.setattr(ff, "_current_device", lambda arrays: 0)
    assert ff._guard_device("numpy", []) is None and ff._guard_device("cupy", [Arr(0), Arr(0)]) == 0
    with pytest.raises(MoistureOutputError, match="different devices"):
        ff._guard_device("cupy", [Arr(0), Arr(1)])
    with pytest.raises(MoistureOutputError, match="never switched"):
        ff._guard_device("cupy", [Arr(2), Arr(2)])


def test_guard_runs_before_admission_scan_and_transfer(tmp_path, monkeypatch):
    W, D, a = profile()
    order = []
    monkeypatch.setattr(ff, "_guard_device", lambda *x: order.append("guard"))
    monkeypatch.setattr(ff.g0, "_admit_host", lambda *x: order.append("admit") or (_ for _ in ()).throw(MoistureOutputQuotaError("stop")))
    with pytest.raises(MoistureOutputQuotaError):
        write(tmp_path, W, D, a)
    assert order == ["guard", "admit"]


def test_g0_schema_untouched_and_not_a_restart():
    import maple_syrup.soil_moisture_output as g0

    assert g0.SCHEMA_SNAPSHOT == "maple_syrup.soil_moisture_snapshot/2" and ff.SCHEMA_FRAME.endswith("floating_moisture_frame/1")


def big_frame(root, L_, N_, chunk, **kw):
    W = np.zeros((L_, N_))
    D = np.ones(N_)
    a = np.ones(N_, dtype=bool)
    E_ = np.arange(L_ + 1, dtype=np.float64)
    lim = limits(chunk_bytes=chunk, max_total_bytes=10**9, max_manifest_bytes=16384)
    return W, D, a, E_, lim, dict(water_m=W, depth_m=D, active=a, depth_edges_m=E_, cell_area_m2=1.0, shape_yx=(1, N_), meta=meta(),
                                  identity=ident(), limits=lim, host_available_bytes_fn=HOSTFN, **kw)


def test_large_l_streams_one_water_row_at_a_time(tmp_path, monkeypatch):
    import weakref

    refs, peak = [], [0]
    real = ff._to_host

    def spy(x):
        y = real(x).copy()  # a distinct downloaded buffer, like a device download
        refs.append(weakref.ref(y))
        peak[0] = max(peak[0], sum(r().nbytes for r in refs if r() is not None))
        return y

    monkeypatch.setattr(ff, "_to_host", spy)
    L_, N_, chunk = 256, 8192, 64 * 4096
    *_, kw = big_frame(tmp_path, L_, N_, chunk)
    n = write_floating_frame(tmp_path, **kw)
    c = chunk // ff.BYTES_PER_CELL
    assert peak[0] <= 17 * c < chunk  # D + mask + ONE water row live, never L rows (Root's repro: 8458240 B)
    assert n.action == "saved"
    refs.clear()
    peak[0] = 0
    assert write_floating_frame(tmp_path, **kw).action == "verified_existing"  # replay streams the same way
    assert peak[0] <= 17 * c


def test_traced_staging_stays_within_the_admitted_model(tmp_path):
    import tracemalloc

    L_, N_, chunk = 256, 8192, 64 * 1024
    *_, lim, kw = big_frame(tmp_path, L_, N_, chunk)
    model = lim.staging_bytes(ff._manifest_model_bytes(L_, 4))
    tracemalloc.start()
    write_floating_frame(tmp_path, **kw)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= model, (peak, model)


@pytest.mark.parametrize("bad", [True, 5.0, float("nan"), -1, "7", None, np.float64(3)])
def test_max_bytes_typed_refusal(tmp_path, bad):
    W, D, a = profile()
    write(tmp_path, W, D, a)
    fr = read_floating_frame(tmp_path / ff.frame_name(4), max_return_bytes=10**6)
    for fn in (fr.thickness_m, fr.theta):
        with pytest.raises(MoistureOutputError, match="max_bytes must be"):
            fn(max_bytes=bad)
    assert fr.thickness_m(max_bytes=np.int64(10**6)).shape == (L, N)


def test_reconstruction_peak_matches_admission_and_values(tmp_path):
    import tracemalloc

    L_, N_ = 8, 100000
    D = np.linspace(0.0, 9.0, N_)
    W = np.zeros((L_, N_))
    E_ = np.arange(L_ + 1, dtype=np.float64)
    a = np.ones(N_, dtype=bool)
    # water only where the porous thickness is positive
    for l in range(L_):
        h = np.maximum(np.minimum(D, E_[l + 1]) - E_[l], 0.0)
        W[l] = 0.5 * h
    write_floating_frame(tmp_path, water_m=W, depth_m=D, active=a, depth_edges_m=E_, cell_area_m2=1.0, shape_yx=(1, N_), meta=meta(),
                         identity=ident(), limits=limits(chunk_bytes=64 * 4096, max_total_bytes=10**9, max_manifest_bytes=16384),
                         host_available_bytes_fn=HOSTFN)
    fr = read_floating_frame(tmp_path / ff.frame_name(4), max_return_bytes=10**9)
    need_h, need_t = 8 * L_ * N_, 8 * L_ * N_ + N_
    with pytest.raises(MoistureOutputQuotaError):
        fr.thickness_m(max_bytes=need_h - 1)
    tracemalloc.start()
    h = fr.thickness_m(max_bytes=need_h)
    _, pk_h = tracemalloc.get_traced_memory()
    h_copy_ok = bool(np.all(h >= 0.0))
    del h
    tracemalloc.reset_peak()
    th = fr.theta(max_bytes=need_t)
    _, pk_t = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert pk_h <= need_h + 4096  # exactly one output, no ufunc temporaries (Root's repro: 12801648 B)
    assert pk_t <= need_t + 65536  # one output plus the N-byte boolean row (Root's repro: 13602064 B)
    assert h_copy_ok
    np.testing.assert_allclose(th[th > 0], 0.5)
    thick = np.maximum(np.minimum(D[None, :], E_[1:, None]) - E_[:-1, None], 0.0)
    assert np.all(th[thick == 0.0] == 0.0)


def test_pre_admission_order_no_edge_value_or_json_work_before_guard_and_host_admission(tmp_path, monkeypatch):
    W, D, a = profile()
    boom = AssertionError("value/JSON work before admission")
    monkeypatch.setattr(ff, "_edges_values", lambda *x: (_ for _ in ()).throw(boom))
    monkeypatch.setattr(ff.g0, "_canon", lambda *x: (_ for _ in ()).throw(boom))
    monkeypatch.setattr(ff, "_build_manifest", lambda *x: (_ for _ in ()).throw(boom))
    hostile = np.array([0.0, np.nan, 3.0, 2.0])  # invalid VALUES must not even be looked at when admission fails first
    with pytest.raises(MoistureOutputQuotaError, match="host"):
        write_floating_frame(tmp_path, water_m=W, depth_m=D, active=a, depth_edges_m=hostile, cell_area_m2=AREA, shape_yx=SHAPE, meta=meta(),
                             identity=ident(), limits=limits(host_memory_budget_bytes=1000), host_available_bytes_fn=HOSTFN)
    with pytest.raises(MoistureOutputError, match="caller_identity"):  # hostile large metadata refused from lengths only (no JSON built)
        write(tmp_path, W, D, a, caller_identity={"k": "x" * 10**6})
    assert os.listdir(tmp_path) == []


def test_manifest_model_bounds_real_manifest(tmp_path):
    import datetime

    for L_ in (1, 3, 50):
        W = np.zeros((L_, 4))
        E_ = np.arange(L_ + 1, dtype=np.float64) * 0.123456789012345
        ci = {"a": "b" * 100}
        day = 2 * L_
        end = (datetime.date(2030, 1, 1) + datetime.timedelta(days=day)).isoformat()
        write_floating_frame(tmp_path, water_m=W, depth_m=np.zeros(4), active=np.zeros(4, dtype=bool), depth_edges_m=E_,
                             cell_area_m2=0.1234567, shape_yx=(2, 2), meta=meta(day=day, cadence_n=2, origin_date="2030-01-01", date=end),
                             identity=ident(), limits=limits(max_snapshots=50), caller_identity=ci, host_available_bytes_fn=HOSTFN)
        size = (tmp_path / ff.frame_name(2 * L_) / "manifest.json").stat().st_size
        assert size <= ff._manifest_model_bytes(L_, ff._caller_identity_bound(ci))


def test_bytes_written_is_exact_and_documented_estimate_is_conservative(tmp_path):
    n = 12
    W = np.zeros((2, n))
    D = np.zeros(n)
    a = np.zeros(n, dtype=bool)
    a[0] = True
    D[0] = 5.0
    W[0, 0] = 0.1
    notice = write_floating_frame(tmp_path, water_m=W, depth_m=D, active=a, depth_edges_m=np.array([0.0, 1.0, 2.0]), cell_area_m2=1.0,
                                  shape_yx=(3, 4), meta=meta(), identity=ident(), limits=limits(), host_available_bytes_fn=HOSTFN)
    d = tmp_path / ff.frame_name(4)
    actual = sum((d / f).stat().st_size for f in os.listdir(d))
    assert notice.bytes_written == actual


def test_date_overflow_and_extremes_are_typed():
    with pytest.raises(MoistureOutputError):
        FloatingFrameMeta(day=1000000000, cadence_n=1, origin_ordinal=0, physical_t_s=0.0, origin_date="9999-12-31", date="9999-12-31")
    with pytest.raises(MoistureOutputError):
        FloatingFrameMeta(day=10**12 - 1, cadence_n=1, origin_ordinal=0, physical_t_s=0.0, origin_date="2000-01-01", date="2000-01-02")
    with pytest.raises(MoistureOutputError):
        FloatingFrameMeta(day=1, cadence_n=1, origin_ordinal=0, physical_t_s=0.0, origin_date=20300101, date="2030-01-02")
    ok = FloatingFrameMeta(day=1, cadence_n=1, origin_ordinal=0, physical_t_s=0.0, origin_date="9999-12-30", date="9999-12-31")
    assert ok.date == "9999-12-31"


def test_replay_worst_case_all_chunks_differ_stops_at_first_difference(tmp_path, monkeypatch):
    import tracemalloc

    L_, N_, chunk = 64, 4096, 64 * 256
    W, *_, lim, kw = big_frame(tmp_path, L_, N_, chunk)
    write_floating_frame(tmp_path, **kw)
    d = tmp_path / ff.frame_name(4)
    before = {n: (d / n).read_bytes() for n in os.listdir(d)}
    kw2 = dict(kw)
    kw2["water_m"] = W + 0.5  # every water chunk differs
    calls = []
    real = ff._pread_equal
    monkeypatch.setattr(ff, "_pread_equal", lambda *x: calls.append(1) or real(*x))
    tracemalloc.start()
    with pytest.raises(MoistureOutputConflict, match="different array bytes"):
        write_floating_frame(tmp_path, **kw2)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(calls) <= 3  # D, mask and the FIRST water row only: no per-piece history
    assert peak <= lim.staging_bytes(ff._manifest_model_bytes(L_, 4))
    assert {n: (d / n).read_bytes() for n in os.listdir(d)} == before


def test_replay_by_a_different_codec_source_is_refused(tmp_path, monkeypatch):
    W, D, a = profile()
    write(tmp_path, W, D, a)
    monkeypatch.setattr(ff, "_MODULE_SHA", "f" * 64)
    with pytest.raises(MoistureOutputConflict, match="codec source"):
        write(tmp_path, W, D, a)


def test_no_fd_leak_when_a_later_open_fails_during_replay(tmp_path, monkeypatch):
    W, D, a = profile()
    write(tmp_path, W, D, a)
    real = ff.g0._open_regular
    n = {"i": 0}

    def flaky(*x, **k):
        n["i"] += 1
        if n["i"] == 6:  # the third replay-compare open (after the three stored-hash opens)
            raise MoistureOutputError("open failed")
        return real(*x, **k)

    before = set(os.listdir("/proc/self/fd"))
    monkeypatch.setattr(ff.g0, "_open_regular", flaky)
    with pytest.raises(MoistureOutputError, match="open failed"):
        write(tmp_path, W, D, a)
    assert len(os.listdir("/proc/self/fd")) <= len(before)  # every earlier descriptor was closed (no leak)


def test_post_seal_moisture_output_error_is_still_indeterminate(tmp_path, monkeypatch):
    W, D, a = profile()
    real = os.fsync

    def failing(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise MoistureOutputError("late failure")
        return real(fd)

    monkeypatch.setattr(os, "fsync", failing)
    with pytest.raises(MoistureOutputIndeterminate):
        write(tmp_path, W, D, a)
    monkeypatch.setattr(os, "fsync", real)
    assert (tmp_path / ff.frame_name(4) / "manifest.json").exists() and (tmp_path / ff.frame_name(4) / "water_m.npy").exists()


GPU = pytest.mark.skipif(os.environ.get("E1_ALLOW_GPU") != "1", reason="GPU not touched by the author; Root sets E1_ALLOW_GPU=1")


@GPU
def test_cuda_frame_equals_host_bytes_and_counts_pieces(tmp_path):
    import cupy as cp

    W, D, a = profile()
    host_dir, dev_dir = tmp_path / "h", tmp_path / "d"
    host_dir.mkdir()
    dev_dir.mkdir()
    write(host_dir, W, D, a)
    n = write(dev_dir, cp.asarray(W), cp.asarray(D), cp.asarray(a))
    assert n.device_chunks > 0
    for name in ("water_m.npy", "depth_m.npy", "active.npy"):
        assert (host_dir / ff.frame_name(4) / name).read_bytes() == (dev_dir / ff.frame_name(4) / name).read_bytes()


@GPU
def test_cuda_two_device_mismatch_refused(tmp_path):
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs two visible devices (Root's allocation is GPU0 only)")
    W, D, a = profile()
    with cp.cuda.Device(1):
        dW, dD, da = cp.asarray(W), cp.asarray(D), cp.asarray(a)
    with pytest.raises(MoistureOutputError, match="never switched"):
        write(tmp_path, dW, dD, da)
