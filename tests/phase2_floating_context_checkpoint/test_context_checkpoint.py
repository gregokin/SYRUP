# ruff: noqa: RUF059, FURB192
"""H2: context container fresh-process proof (NumPy/Numba x Dense/published Windowed), terminal + second settlement, tamper/refusal and rollback."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from case import AVAIL, IDENT, build, run
from static_case import factory

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_context_checkpoint as cc
from maple_syrup import floating_held_context as fh

WORKER = Path(__file__).resolve().parent / "fresh_worker.py"
REFUSE = (fc.PondCheckpointError,)


def _run(mode, work, backend, windowed):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NUMBA_NUM_THREADS="1")
    r = subprocess.run([sys.executable, str(WORKER), mode, str(work), backend, "1" if windowed else "0", "8", "6"], capture_output=True, text=True, env=env, timeout=900, check=False)
    assert r.returncode == 0, r.stderr[-3000:]


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("windowed", [False, True])
def test_fresh_process_restore_from_the_context_container_is_bit_exact(tmp_path, backend, windowed):
    if backend == "numba":
        pytest.importorskip("numba")
    _run("save", tmp_path, backend, windowed)
    assert {p.name for p in (tmp_path / "ctx").iterdir()} == {cc.MANIFEST_NAME, "context_arrays.npz", "child"}
    _run("load", tmp_path, backend, windowed)
    a, b = np.load(tmp_path / "save_final.npz"), np.load(tmp_path / "load_final.npz")
    assert set(a.files) == set(b.files) and len(a.files) > 40
    for k in a.files:
        assert a[k].tobytes() == b[k].tobytes(), k


def _spec(vert):
    return fh.PreferentialSpec(np.full(vert.n_cells, 2.0e-3), 0.5, 1e-5, np.array([0.0, 1.0, 1.0]), (), None)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    base = tmp_path_factory.mktemp("h2")
    (base / "a").mkdir()
    ev, st, vert = build(base / "a")
    st = run(ev, st, 6)
    ck = base / "ctx"
    cc.save_floating_context_checkpoint(ck, st, ev, IDENT, pref_spec=_spec(vert), available_bytes_fn=AVAIL)
    return base, ck, ev, st


def tamper_manifest(saved, tmp_path, name, edit, reseal=True):
    d = tmp_path / name
    shutil.copytree(saved[1], d)
    man = json.loads((d / cc.MANIFEST_NAME).read_text())
    edit(man)
    from maple_syrup import pond_checkpoint as pc

    (d / cc.MANIFEST_NAME).write_text(pc._json_dumps(cc._seal(man) if reseal else man))
    return d


def load(d, **kw):
    return cc.load_floating_context_checkpoint(d, factory("numpy", False), IDENT, available_bytes_fn=kw.pop("avail", AVAIL), **kw)


def test_refusals_leave_nothing_and_a_good_load_still_works(saved, tmp_path):
    base, ck, _ev, _st = saved
    ev, st = load(ck)
    assert st.counters.n_accepted_steps == 6
    cases = {
        "extra": lambda m: m.update(extra=1), "schema": lambda m: m.update(context_schema="x"), "profile": lambda m: m.update(profile="p/2"),
        "source": lambda m: m["source"].update(maple_syrup_tree_sha256="0" * 64), "backend": lambda m: m.update(hydraulic_backend="cuda"),
        "ctxhash": lambda m: m["files"]["context_arrays.npz"].update(sha256="0" * 64),
        "childhash": lambda m: m["child"]["files"]["pond_continuation.npz"].update(sha256="0" * 64),
        "childschema": lambda m: m["child"].update(schema="x"), "childmanifest": lambda m: m["child"].update(manifest_sha256="0" * 64),
        "childnames": lambda m: m["child"]["files"].update(extra={"size": 0, "sha256": "0" * 64}),
        "table": lambda m: m["context_arrays"][sorted(m["context_arrays"])[0]].update(dtype="<f4"),
        "ids": lambda m: m["context_meta"]["identities"].update(config_sha256="0" * 64),
        "facts": lambda m: m["context_meta"]["facts"].update(generation=7),
        "tbl_shape": lambda m: m["context_arrays"][sorted(m["context_arrays"])[0]].update(shape=[10**6]),
    }
    for name, edit in cases.items():
        with pytest.raises(REFUSE):
            load(tamper_manifest(saved, tmp_path, name, edit))
    with pytest.raises(REFUSE):
        load(tamper_manifest(saved, tmp_path, "unsealed", lambda m: m.update(extra=1), reseal=False))
    with pytest.raises(REFUSE, match="identity"):
        cc.load_floating_context_checkpoint(ck, factory("numpy", False), {"case": "other"}, available_bytes_fn=AVAIL)
    # unknown file / symlink / child entry
    d = tmp_path / "unk"
    shutil.copytree(ck, d)
    (d / "junk").write_text("x")
    with pytest.raises(REFUSE):
        load(d)
    d2 = tmp_path / "sym"
    shutil.copytree(ck, d2)
    (d2 / "context_arrays.npz").rename(d2 / "real.npz")
    (d2 / "context_arrays.npz").symlink_to(d2 / "real.npz")
    with pytest.raises(REFUSE):
        load(d2)
    # tampered payload byte (hash)
    d3 = tmp_path / "bytes"
    shutil.copytree(ck, d3)
    data = bytearray((d3 / "context_arrays.npz").read_bytes())
    data[len(data) // 2] ^= 0xFF
    (d3 / "context_arrays.npz").write_bytes(bytes(data))
    with pytest.raises(REFUSE, match="hash"):
        load(d3)


def test_wrong_static_factory_is_refused_by_the_unchanged_child_binding_and_target_is_retryable(saved, tmp_path):
    import static_case as sc

    _base, ck, _ev, _st = saved
    original = dict(sc.TERRAIN)
    try:
        sc.TERRAIN["drop"] = 0.005  # another terrain -> another topology identity
        with pytest.raises(REFUSE):
            load(ck)
    finally:
        sc.TERRAIN.update(original)
    assert load(ck)[1] is not None


def test_memory_zero_need_minus_one_and_reserve_before_arrays(saved, tmp_path):
    _base, ck, ev, st = saved
    from maple_syrup import native_floating_soil as nfs

    for avail in (lambda: 0,):
        with pytest.raises(nfs.NativeFloatingMemoryError):
            load(ck, avail=avail)
    reads = []

    def probe(limit):
        def f():
            reads.append(limit)
            return limit
        return f

    with pytest.raises(nfs.NativeFloatingMemoryError) as info:
        load(ck, avail=probe(1 << 12))
    assert reads
    # save side: refused before anything is published and no temp dir remains
    with pytest.raises(nfs.NativeFloatingMemoryError):
        cc.save_floating_context_checkpoint(tmp_path / "s0", st, ev, IDENT, pref_spec=_spec(ev.vertical), available_bytes_fn=lambda: 0)
    assert not (tmp_path / "s0").exists() and not list(tmp_path.glob(".s0.partial-*"))
    with pytest.raises(fc.FloatingCheckpointError, match="already exists"):
        cc.save_floating_context_checkpoint(ck, st, ev, IDENT, pref_spec=_spec(ev.vertical), available_bytes_fn=AVAIL)
    assert info is not None


def test_static_recipe_never_materialises_a_bed_and_matches_the_probe_values():
    import static_case as sc

    from maple_syrup import probe

    calls = []
    real = probe.build_minimal_state
    probe.build_minimal_state = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        ctx = sc.bed_context()
    finally:
        probe.build_minimal_state = real
    assert calls == []  # no probe state, no voxel/active arrays
    ref = real(ny=3, nx=4, dx_m=sc.DX, dy_m=sc.DX, n_voxels=sc.NV, voxel_dz_m=sc.DZ)
    assert ctx.geometry == ref.geometry and ctx.grain_classes == ref.grain_classes and ctx.mass_resolution_kg == ref.mass_resolution_kg


def test_large_mismatched_context_is_refused_before_np_load_and_any_array_copy(saved, tmp_path, monkeypatch):
    """Root's defect: a large archive paired with the wrong child and RECOMPUTED outer hashes must be refused from the declared/analytic metadata, before `np.load` and under a small budget."""
    from maple_syrup import pond_checkpoint as pc

    d = tmp_path / "big"
    shutil.copytree(saved[1], d)
    with np.load(d / "context_arrays.npz") as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    name = sorted(arrays)[0]
    arrays[name] = np.zeros(300_000)  # a large member the analytic H0 layout does not allow
    np.savez(d / "context_arrays.npz", **arrays)
    man = json.loads((d / cc.MANIFEST_NAME).read_text())
    man["context_arrays"][name] = {"dtype": "<f8", "shape": [300_000]}
    import hashlib

    man["files"]["context_arrays.npz"] = {"size": (d / "context_arrays.npz").stat().st_size, "sha256": hashlib.sha256((d / "context_arrays.npz").read_bytes()).hexdigest()}
    (d / cc.MANIFEST_NAME).write_text(pc._json_dumps(cc._seal(man)))
    loads = []
    real = np.load
    monkeypatch.setattr(np, "load", lambda *a, **k: (loads.append(1), real(*a, **k))[1])
    with pytest.raises(REFUSE, match="analytic"):
        load(d, avail=lambda: 8 * 1024**2)  # covers stage 0 (manifest + 3 MiB cold allowance) only
    assert loads == []


def test_exact_composed_need_minus_one_and_reserve_once(saved, tmp_path, monkeypatch):
    from maple_syrup import native_floating_soil as nfs

    _base, ck, _ev, _st = saved
    need = cc.context_load_need_bytes(ck, IDENT, available_bytes_fn=AVAIL)
    loads = []
    real = np.load
    monkeypatch.setattr(np, "load", lambda *a, **k: (loads.append(1), real(*a, **k))[1])
    with pytest.raises(nfs.NativeFloatingMemoryError):
        load(ck, avail=lambda: need - 1)
    assert loads == []  # refused by the composed admission before any array was loaded
    with pytest.raises(nfs.NativeFloatingMemoryError):
        cc.load_floating_context_checkpoint(ck, factory("numpy", False), IDENT, available_bytes_fn=lambda: need + 6, reserve_bytes=7)
    assert cc.context_load_need_bytes(ck, IDENT, available_bytes_fn=AVAIL, reserve_bytes=7) == need + 7
    assert load(ck, avail=lambda: 1 << 40)[1] is not None


def test_no_clobber_publication_and_durability(saved, tmp_path, monkeypatch):
    _base, ck, ev, st = saved
    target = tmp_path / "race"
    real = cc._fsync_dir

    def racing(path):
        real(path)
        if Path(path).name.startswith(".race.partial-") and not target.exists():
            target.mkdir()  # an EMPTY destination appears just before publication

    monkeypatch.setattr(cc, "_fsync_dir", racing)
    with pytest.raises(fc.FloatingCheckpointError, match="without replacing"):
        cc.save_floating_context_checkpoint(target, st, ev, IDENT, pref_spec=_spec(ev.vertical), available_bytes_fn=AVAIL)
    assert target.is_dir() and list(target.iterdir()) == [] and not list(tmp_path.glob(".race.partial-*"))  # not replaced, no residue
    monkeypatch.undo()
    # fsync failure AFTER publication: indeterminate, the published directory is preserved
    t2 = tmp_path / "late"

    def failing(path):
        if Path(path) == tmp_path:
            raise OSError("injected")
        real(path)

    monkeypatch.setattr(cc, "_fsync_dir", failing)
    with pytest.raises(fc.FloatingCheckpointError, match="INDETERMINATE"):
        cc.save_floating_context_checkpoint(t2, st, ev, IDENT, pref_spec=_spec(ev.vertical), available_bytes_fn=AVAIL)
    monkeypatch.undo()
    assert (t2 / cc.MANIFEST_NAME).is_file()
    assert load(t2)[1] is not None
