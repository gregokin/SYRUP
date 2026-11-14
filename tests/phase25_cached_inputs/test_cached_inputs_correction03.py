"""Correction03: (1) the recorded LEAF package file_count is strict + equal to the ACTUAL package (live and cached), (2) the sealed-type proofs are O(1) in the species count (no per-species
scan or call on any proof; registry fingerprints O(1)), (3) the explicit-root guard for the two qualified immutable MAPLE materializations. REAL genuine P1 / capture / stage / prepare."""
from __future__ import annotations

import copy
import gc
import json
import sys
import time
import tracemalloc

import pytest
from cached_support import (
    CPU_ROOT,
    CUDA_ROOT,
    GPU_FLAG,
    MAPLE_PACKAGE,
    ROOT_ENV,
    configured_maple_root,
    expected,
    live,
    loadable,
    make_column,
    prepare,
    stage,
    wh,
    wvb,
)
from test_cached_inputs_correction01 import _big_species_case
from veg_binding_support import BIG, full_source, helper_pin, leaf_pin

ERR = wvb.VegetationBindingError


# ---------------------------------------------------------------- 1. strict, actual package file_count
FORGED_COUNTS = [999, "28", True, -5, None, 28.0, 29, 27, 0, "", [28]]


@pytest.mark.parametrize("count", FORGED_COUNTS, ids=[repr(c) for c in FORGED_COUNTS])
def test_cached_package_file_count_forgery_is_refused_before_adoption_and_native_prepare(leaf, count, monkeypatch):
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    meta, arrays = loadable(rec)
    meta["package"]["file_count"] = count
    meta["identity"]["package"]["file_count"] = count                                       # the forger keeps the recorded identity consistent
    pkg_json = wvb._frozen_json(meta["package"]) if isinstance(count, (int, str, float, bool, list, type(None))) else None
    idj = wvb._frozen_json(meta["identity"])
    meta["record_sha256"] = wh._record_sha(rec.content_sha256, rec.controls.sha256, rec.binding_sha256, rec.source_json, pkg_json, rec.helpers_json, idj, rec.mask_sha256, rec.surface_sha256,
                                           rec.receiver_ids, rec.receiver_weights, rec.species_ids, rec.shape, rec.cell_area_m2, rec.schema, rec.compact_version)   # resealed integrity
    from maple_syrup import floating_matrix_infiltration as fm
    calls = []
    real = fm.prepare_floating_matrix
    monkeypatch.setattr(fm, "prepare_floating_matrix", lambda *a, **k: calls.append(1) or real(*a, **k))
    with pytest.raises(ERR, match="file_count"):
        stage(rec, p, b, meta, arrays, m, expected_record_sha256=meta["record_sha256"])      # typed refusal at the stage ...
    assert calls == []                                                                       # ... before any adoption / native preparation


def test_live_bind_refuses_a_forged_identity_package_count(leaf, monkeypatch):
    from cached_support import controls
    p, _b, surf, m = live(leaf)
    real = type(p).identity_record
    for forged in (999, "28", True, -5, 28.0):
        def fake(self, forged=forged):
            d = real(self)
            d["package"]["file_count"] = forged
            return d
        monkeypatch.setattr(type(p), "identity_record", fake)
        with pytest.raises(ERR, match="file_count"):
            wvb.bind_projection(p, controls(), expected_leaf_package_sha256=leaf_pin(), expected_projection_sha256=p.identity_record()["content_sha256"], expected_source=full_source(p),
                                expected_external_helpers=helper_pin(), active_mask=m, current_surface=surf, max_bytes=BIG)
    monkeypatch.setattr(type(p), "identity_record", real)


def test_the_count_is_the_actual_runtime_count_and_ids_are_unchanged(leaf):
    p, b, _surf, m = live(leaf)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    runtime = wvb._runtime_leaf_digest(wvb._load_leaf())
    assert b.leaf_package_file_count == json.loads(rec.package_json)["file_count"] == runtime["file_count"] == 28
    meta, arrays = loadable(rec)
    rec2 = stage(rec, p, b, meta, arrays, m)
    cb = wh.cached_binding(rec2, expected_binding_sha256=b.sha256, max_bytes=BIG)
    assert type(cb.leaf_package_file_count) is int and cb.leaf_package_file_count == 28 and cb.sha256 == b.sha256 and rec2.record_sha256 == rec.record_sha256
    assert prepare(make_column(), cb).law_sha256 == prepare(make_column(), b).law_sha256          # the genuine ids are unchanged by the count validation
    assert expected(p, b, rec)["expected_record_sha256"] == rec.record_sha256
    # cached_binding itself refuses a coerced / invalid recorded count (no int() coercion)
    object.__setattr__(rec2, "package_json", json.dumps({**json.loads(rec2.package_json), "file_count": "28"}))
    with pytest.raises(ERR):
        wh.cached_binding(rec2, expected_binding_sha256=b.sha256, max_bytes=BIG)


# ---------------------------------------------------------------- 2. O(1) proofs
class _Counter:
    """sys.setprofile call counter by Python function name (STRUCTURAL: counts calls, not time)."""
    NAMES = ("stream_estimate", "_count_enabled", "_enabled_index", "_val_fp", "_check_id_tuple", "estimate_stream_bytes")

    def __init__(self):
        self.counts = dict.fromkeys(self.NAMES, 0)

    def __call__(self, frame, event, arg):
        if event == "call" and frame.f_code.co_name in self.counts:
            self.counts[frame.f_code.co_name] += 1

    def run(self, fn):
        self.counts = dict.fromkeys(self.NAMES, 0)
        sys.setprofile(self)
        try:
            fn()
        finally:
            sys.setprofile(None)
        return dict(self.counts)


def _retained(fp) -> int:
    return len(repr(fp))


@pytest.mark.parametrize("S", [8, 2000, 20000])
def test_proofs_do_no_per_species_work_and_retain_o1_registry_state(leaf, S):
    _p, b, _surf, _m = _big_species_case(leaf, S=S)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    wvb.verify_binding(b)
    wh.verify_cached_inputs(rec)
    cnt = _Counter()
    cb_calls = cnt.run(lambda: wvb.verify_binding(b))
    rec_calls = cnt.run(lambda: wh.verify_cached_inputs(rec))
    assert cb_calls["stream_estimate"] == cb_calls["_count_enabled"] == cb_calls["_enabled_index"] == cb_calls["estimate_stream_bytes"] == 0, cb_calls   # no per-species LEAF loop on the proof
    assert rec_calls["stream_estimate"] == rec_calls["_count_enabled"] == rec_calls["estimate_stream_bytes"] == 0, rec_calls
    assert cb_calls["_val_fp"] <= 120 and rec_calls["_val_fp"] <= 160, (cb_calls, rec_calls)                       # a fixed number of field fingerprints, whatever S
    t = {}
    for name, fn in (("verify_binding", lambda: wvb.verify_binding(b)), ("verify_cached", lambda: wh.verify_cached_inputs(rec))):
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        t0 = time.perf_counter()
        for _ in range(20):
            fn()
        t[name] = ((time.perf_counter() - t0) / 20, tracemalloc.get_traced_memory()[1] - base)
        tracemalloc.stop()
    assert t["verify_binding"][1] < 64 * 1024 and t["verify_cached"][1] < 64 * 1024, t                          # owned bytes per proof: a constant (equality across S is asserted below)
    fps = (_retained(wvb._BND_REGISTRY[b]), _retained(wh._REGISTRY[rec]))
    print(f"S={S}: counts={cb_calls}/{rec_calls} seconds/proof={ {k: round(v[0], 6) for k, v in t.items()} } owned={ {k: v[1] for k, v in t.items()} } retained_fp_chars={fps}")
    assert fps[0] < 6000 and fps[1] < 9000, fps                                                                   # the retained fingerprint is O(1) (it was O(S) tuples)


def test_proof_cost_is_independent_of_the_species_count_structurally(leaf):
    res = {}
    for S in (8, 2000, 20000):
        _p, b, _surf, _m = _big_species_case(leaf, S=S)
        rec = wh.capture_cached_inputs(b, max_bytes=BIG)
        wvb.verify_binding(b)
        wh.verify_cached_inputs(rec)
        c = _Counter()
        owned = []
        for fn in (lambda b=b: wvb.verify_binding(b), lambda rec=rec: wh.verify_cached_inputs(rec)):
            gc.collect()
            tracemalloc.start()
            base = tracemalloc.get_traced_memory()[0]
            fn()
            owned.append(tracemalloc.get_traced_memory()[1] - base)
            tracemalloc.stop()
        res[S] = (c.run(lambda b=b: wvb.verify_binding(b)), c.run(lambda rec=rec: wh.verify_cached_inputs(rec)), _retained(wvb._BND_REGISTRY[b]), _retained(wh._REGISTRY[rec]), tuple(owned))
    assert res[8][0] == res[2000][0] == res[20000][0] and res[8][1] == res[2000][1] == res[20000][1]              # identical call profiles for 8 / 2000 / 20000 species
    assert abs(res[20000][2] - res[8][2]) < 64 and abs(res[20000][3] - res[8][3]) < 64                          # identical retained registry state
    for k in (0, 1):
        assert abs(res[20000][4][k] - res[8][4][k]) < 1024 and abs(res[2000][4][k] - res[8][4][k]) < 1024, res    # identical owned bytes per proof (the old proof allocated ~92 KB at S=400)


def test_id_tuples_are_validated_once_and_a_replaced_tuple_is_detected(leaf):
    p, b, _surf, _m = _big_species_case(leaf, S=50)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    wvb.verify_binding(b)
    object.__setattr__(b, "species_ids", tuple(b.species_ids))                       # an EQUAL tuple (rebuilt copy): a different object => detected
    if b.species_ids is not p.species_ids:
        with pytest.raises(ERR):
            wvb.verify_binding(b)
    p, b, _surf, _m = _big_species_case(leaf, S=50)
    object.__setattr__(b, "enabled_species", [*b.enabled_species])                   # not a tuple
    with pytest.raises(ERR):
        wvb.verify_binding(b)
    p, b, _surf, _m = _big_species_case(leaf, S=50)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    object.__setattr__(rec, "receiver_ids", (*rec.receiver_ids, "x"))
    with pytest.raises(ERR):
        wh.verify_cached_inputs(rec)
    for bad in ([1, 2], ("a", 1), None):
        with pytest.raises(ERR):
            wvb._check_id_tuple(bad, "ids")
    # every OTHER field keeps its full fingerprint: the array tuple and scalars still detect replacement
    p, b, _surf, _m = _big_species_case(leaf, S=50)
    rec = wh.capture_cached_inputs(b, max_bytes=BIG)
    object.__setattr__(rec, "arrays", tuple((k, v.copy()) for k, v in rec.arrays))
    with pytest.raises(ERR):
        wh.verify_cached_inputs(rec)
    _ = copy


def test_missing_public_stream_capability_is_typed_unavailable_before_prepare(leaf, monkeypatch):
    p, b, _surf, _m = live(leaf)
    monkeypatch.delattr(type(p), "stream_estimate")
    with pytest.raises(wvb.VegetationConsumerUnavailableError):
        wvb.verify_binding(b)
    with pytest.raises(wvb.VegetationConsumerUnavailableError):
        prepare(make_column(), b)


def test_the_sealed_view_proof_still_detects_a_forged_projection(leaf):
    p, b, _surf, _m = live(leaf)
    wvb.verify_binding(b)
    object.__setattr__(p, "cell_area_m2", 100.0)                                     # Root's scalar-seal probe on the producer: LEAF's own sealed check refuses it
    with pytest.raises(ERR):
        wvb.verify_binding(b)


# ---------------------------------------------------------------- 3. explicit-root guard
def test_root_guard_accepts_exactly_the_two_qualified_immutable_roots(monkeypatch):
    monkeypatch.delenv(ROOT_ENV, raising=False)
    monkeypatch.delenv(GPU_FLAG, raising=False)
    assert configured_maple_root() == CPU_ROOT.resolve()                              # the CPU lane's explicit constant
    for root in (CPU_ROOT, CUDA_ROOT):
        monkeypatch.setenv(ROOT_ENV, str(root))
        assert configured_maple_root() == root.resolve()
        monkeypatch.setenv(GPU_FLAG, "1")
        assert configured_maple_root() == root.resolve()                              # the GPU lane configured it explicitly
        monkeypatch.delenv(GPU_FLAG, raising=False)


def test_root_guard_fails_missing_unconfigured_wrong_and_live_roots(monkeypatch, tmp_path):
    monkeypatch.delenv(ROOT_ENV, raising=False)
    monkeypatch.setenv(GPU_FLAG, "1")
    with pytest.raises(ValueError, match="explicitly"):
        configured_maple_root()                                                       # the GPU lane never falls back to a default
    monkeypatch.delenv(GPU_FLAG)
    for bad in ("", "   ", "/home/okin/MAPLE", "/home/okin/MAPLE/src", str(tmp_path), str(CPU_ROOT) + "/source", str(CUDA_ROOT.parent), str(CPU_ROOT) + "x", "/nonexistent/maple"):
        monkeypatch.setenv(ROOT_ENV, bad)
        with pytest.raises(ValueError):
            configured_maple_root()
    link = tmp_path / "alias"
    link.symlink_to(CPU_ROOT)
    monkeypatch.setenv(ROOT_ENV, str(link))
    assert configured_maple_root() == CPU_ROOT.resolve()                              # a symlink to the qualified root resolves to it (still the exact immutable root)
    fake = tmp_path / "fake"
    (fake / "source" / "src" / "maple").mkdir(parents=True)
    (fake / "source" / "src" / "maple" / "__init__.py").write_text("")
    monkeypatch.setenv(ROOT_ENV, str(fake))
    with pytest.raises(ValueError, match="qualified"):
        configured_maple_root()                                                       # a lookalike tree is not a qualified root


def test_the_session_fixture_asserts_the_configured_root_and_the_full_package(pinned_sources):
    from pathlib import Path

    import maple
    assert Path(maple.__file__).resolve().is_relative_to(configured_maple_root() / "source" / "src")
    assert pinned_sources["MAPLE"]["digest"] == MAPLE_PACKAGE and pinned_sources["MAPLE"]["files"] == 412


def test_cpu_children_stay_cpu_and_gpu_children_use_the_configured_root(monkeypatch):
    from cached_support import native_env
    monkeypatch.setenv(ROOT_ENV, str(CUDA_ROOT))
    env = native_env()
    assert str(CPU_ROOT) in env["PYTHONPATH"] and str(CUDA_ROOT) not in env["PYTHONPATH"] and env["CUDA_VISIBLE_DEVICES"] == ""
    genv = native_env(gpu=True)
    assert str(CUDA_ROOT) in genv["PYTHONPATH"] and str(CPU_ROOT) not in genv["PYTHONPATH"]
    monkeypatch.setenv(ROOT_ENV, "/home/okin/MAPLE")
    with pytest.raises(ValueError):
        native_env(gpu=True)                                                          # no live-mutable fallthrough for a GPU child
    assert str(CPU_ROOT) in native_env()["PYTHONPATH"]
