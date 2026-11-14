"""Dynamic-vegetation held context (/3 CPU) on REAL native lifecycle boundaries: genuine storm -> closed Boundary (NumPy / Numba x Dense / published Windowed) -> the lifecycle's own `apply_vegetation_generation` (ACTUAL LEAF
projection + S0 binding) -> `FloatingPondVertical.from_boundary(new root)` -> capture the held cached binding -> the normal STRICT export -> strict parse -> canonical reconstruct -> genuine `from_held_context` vertical -> STRICT
re-export. No private minting, no generation / head injection, no fake class; independent controls recompute the identities from the actual public objects."""
from __future__ import annotations

import copy
import gc
import inspect
import json
import re

import numpy as np
import pytest
from hc_support import (
    AVAIL,
    BACKENDS,
    BIG,
    World,
    expected_of,
    export,
    fh,
    fhv,
    fp,
    pins_of,
    reconstruct,
    serialise,
    vb,
    wh,
    wvb,
)

ERR = fh.HeldContextError
PARAMS = [(be, w) for be in BACKENDS for w in (False, True)]


@pytest.fixture(scope="module", params=PARAMS, ids=[f"{b}-{'windowed' if w else 'dense'}" for b, w in PARAMS])
def world(request, leaf, tmp_path_factory):
    backend, windowed = request.param
    return World(leaf, tmp_path_factory.mktemp(f"w_{backend}_{windowed}"), backend, windowed)


@pytest.fixture(scope="module")
def dense(leaf, tmp_path_factory):
    return World(leaf, tmp_path_factory.mktemp("w_dense"), "numpy", False)


# ------------------------------------------------------------------------------------------------ the chain
def test_dynamic_boundary_to_strict_export_parse_reconstruct_vertical_and_strict_reexport(world):
    w = world
    t0 = w.vertical.held_template()
    assert t0.matrix_op.layouts["vegetation_factor"] == "cell" and t0.preferential is not None                 # the vertical ACTUALLY holds the dynamic law (lifecycle apply, no injection)
    r1 = export(w.vertical, w.rec, w.pins)
    m = r1.meta()
    assert m["schema"] == fhv.SCHEMA3 and m["backend"] == w.backend and "pref" not in m
    ident = m["identities"]
    assert (ident["config_sha256"], ident["root_column_sha256"], ident["matrix_op_sha256"], ident["pref_sha256"]) == (w.vertical.config.sha256, t0.column.sha256, t0.matrix_op.sha256, t0.preferential.sha256)
    assert m["scalars"]["matrix_vegetation_factor"] == float(w.vertical.config.matrix_vegetation_factor).hex()   # the config keeps the BASE scalar g truthfully
    meta, arrays = serialise(r1)                                                                              # plain parts (what a file codec would hold)
    r1b = fhv.record_from_parts_vegetation(meta, arrays, available_bytes_fn=AVAIL)
    assert r1b.meta() == m
    ctx = reconstruct(r1b, w.pins)
    fh.require_held_context(ctx)
    assert (ctx.config.sha256, ctx.column.sha256, ctx.matrix_op.sha256, ctx.preferential.sha256) == (ident["config_sha256"], ident["root_column_sha256"], ident["matrix_op_sha256"], ident["pref_sha256"])
    assert ctx.generation == t0.generation and ctx.source_witness == t0.source_witness and ctx.layouts == dict(t0.column.layouts) and ctx.area_m2 == t0.column.cell_area_m2
    np.testing.assert_array_equal(ctx.depth0_m, t0.depth0_m)
    np.testing.assert_array_equal(ctx.active, np.asarray(t0.column._maps["active"]))
    assert ctx.vegetation.cached.record_sha256 == w.rec.record_sha256 and ctx.vegetation.law_sha256 == m["vegetation"]["law"]["law_sha256"]
    # the reconstructed operators carry the SAME per-cell g bytes as the lifecycle's operator (independent comparison of the held law arrays)
    from maple_syrup import floating_matrix_infiltration as fm
    np.testing.assert_array_equal(np.asarray(fm.held_cell_vegetation_factor(ctx.matrix_op)), np.asarray(fm.held_cell_vegetation_factor(t0.matrix_op)))
    v1 = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL)
    assert v1.held_template().matrix_op is ctx.matrix_op
    r2 = export(v1, w.rec, w.pins)                                                                            # STRICT re-export from the restored vertical
    assert r2.meta()["identities"] == m["identities"] and r2.meta()["vegetation"] == m["vegetation"]


def test_there_is_no_prospective_export_override_and_a_vertical_without_the_law_is_refused(dense):
    w = dense
    assert "apply_law" not in inspect.signature(fhv.export_held_context_vegetation).parameters
    from maple_syrup import floating_held_context_vegetation_cuda as fvc
    assert "apply_law" not in inspect.signature(fvc.export_held_context_vegetation_cuda).parameters
    with pytest.raises(TypeError):
        export(w.vertical, w.rec, w.pins, apply_law=True)
    with pytest.raises(ERR, match="no preferential operator|CURRENT operators"):
        export(w.old_vertical(), w.rec, w.pins)                                                               # the pre-law vertical: an export never records a law the vertical does not hold


def test_a_cache_that_is_not_the_held_law_is_refused_by_the_always_on_operator_comparison(dense, leaf):
    w = dense
    other = vb.binding(leaf, w.b, w.stack, ctl=vb.controls(w.b, a1=0.02))                                        # the same plants and column, different site controls -> a different canonical law
    rec2 = wh.capture_cached_inputs(other, max_bytes=BIG)
    pins2 = pins_of(w.stack["proj"], other, rec2)
    with pytest.raises(ERR, match="CURRENT operators|identity"):
        export(w.vertical, rec2, pins2)


def test_the_same_held_cache_on_a_new_property_column_gives_new_operator_ids_computed_not_applied(leaf, tmp_path, monkeypatch, dense):
    wb = World(leaf, tmp_path / "chg", "numpy", False, changed=monkeypatch)                                      # a boundary whose provider properties genuinely changed (declared perturbation)
    ra, rb = export(dense.vertical, dense.rec, dense.pins), export(wb.vertical, wb.rec, wb.pins)
    ia, ib = ra.meta(), rb.meta()
    assert ia["identities"]["root_column_sha256"] != ib["identities"]["root_column_sha256"] and ia["identities"]["matrix_op_sha256"] != ib["identities"]["matrix_op_sha256"]
    assert ia["vegetation"]["law"]["g_sha256"] == ib["vegetation"]["law"]["g_sha256"]                           # the same recipe (same R1) ...
    # ... and the OTHER world's cache exports the changed-property vertical strictly: the ids are RECOMPUTED on its column, never copied from the cache
    rc = export(wb.vertical, dense.rec, dense.pins)
    assert rc.meta()["identities"] == ib["identities"]
    assert reconstruct(rc, dense.pins).column.sha256 == ib["identities"]["root_column_sha256"]


# ------------------------------------------------------------------------------------------------ old / new profiles
def test_old_and_new_profiles_refuse_each_other_and_the_old_export_fails_closed_for_a_per_cell_g(dense):
    w = dense
    r = export(w.vertical, w.rec, w.pins)
    meta, arrays = serialise(r)
    with pytest.raises(ERR):
        fh.record_from_parts(meta, arrays)
    with pytest.raises(ERR):
        fh.reconstruct_held_context(r, expected_of(r))
    for verify in (True, False):                                                                              # the guard precedes any record: verify False must refuse too
        with pytest.raises(ERR, match="per-cell vegetation"):
            fh.export_held_context(w.vertical, available_bytes_fn=AVAIL, verify=verify)
    ctx = reconstruct(r, w.pins)
    v1 = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL)
    for verify in (True, False):
        with pytest.raises(ERR, match="per-cell vegetation"):
            fh.export_held_context(v1, available_bytes_fn=AVAIL, verify=verify)
    old = fh.export_held_context(w.old_vertical(), available_bytes_fn=AVAIL, verify=False) if w.old_vertical().held_template().preferential is None else None
    if old is not None:                                                                                       # the old scalar-g record is unchanged and the new parser refuses it
        with pytest.raises(ERR):
            fhv.record_from_parts_vegetation(old.meta(), {k: np.array(x) for k, x in old.arrays().items()})
        with pytest.raises(ERR):
            fh.reconstruct_vegetation_held_context(old, fh.ExpectedHeldIdentities(*[old.meta()["identities"][k] for k in ("config_sha256", "root_column_sha256", "matrix_op_sha256", "pref_sha256")]), w.pins,
                                                   memory_budget_bytes=BIG)
    assert fh.SCHEMA.endswith("/1") and fhv.SCHEMA3.endswith("/3") and fhv.SCHEMA4.endswith("/4")


# ------------------------------------------------------------------------------------------------ tamper / pins / hook
def test_every_identity_scalar_array_and_pin_tamper_is_refused(dense):
    w = dense
    r = export(w.vertical, w.rec, w.pins)
    meta, arrays = serialise(r)
    good = expected_of(r)
    pins = w.pins
    for fld in ("config_sha256", "root_column_sha256", "matrix_op_sha256", "pref_sha256"):
        with pytest.raises(ERR):
            reconstruct(r, pins, fh.ExpectedHeldIdentities(**{**good.__dict__, fld: "0" * 64}))
    with pytest.raises(ERR):
        reconstruct(r, pins, fh.ExpectedHeldIdentities(good.config_sha256, good.root_column_sha256, good.matrix_op_sha256, None))
    for fld, val in (("record_sha256", "0" * 64), ("binding_sha256", "1" * 64), ("projection_sha256", "2" * 64), ("leaf_package_sha256", "3" * 64),
                     ("external_helpers", [{"module": "benchmarks.spatial_mapping", "sha256": "sha256:" + "4" * 64, "bytes": 1}]), ("source", {**pins.source, "state_sha256": "5" * 64})):
        with pytest.raises(ERR):
            reconstruct(r, fhv.ExpectedVegetationPins(**{**pins.__dict__, fld: val}))
    with pytest.raises(ERR):
        reconstruct(r, None)
    with pytest.raises(ERR):
        reconstruct(r, pins, host_reserve_bytes=1)                                                            # host_* only for a CUDA record

    def parse(m, a):
        return fhv.record_from_parts_vegetation(m, a, available_bytes_fn=AVAIL)

    cp = lambda m: m["vegetation"]["cached"]["compact"]
    edits = {
        "schema": lambda m: m.__setitem__("schema", "maple_syrup.floating_held_context/9"),
        "extra key": lambda m: m.__setitem__("pref", None),
        "no vegetation": lambda m: m.pop("vegetation"),
        "law key": lambda m: m["vegetation"]["law"].__setitem__("g_sha256", "zz"),
        "pref identity None": lambda m: m["identities"].__setitem__("pref_sha256", None),
        "layer count": lambda m: m.__setitem__("n_layers", m["n_layers"] + 1),
        "cell count": lambda m: m.__setitem__("n_cells", m["n_cells"] + 2),
        "cached entries": lambda m: cp(m).__setitem__("n_entries", cp(m)["n_entries"] + 1),
        "cached shape product": lambda m: cp(m).__setitem__("shape", [cp(m)["shape"][0] + 1, cp(m)["shape"][1]]),
        "cached version type": lambda m: m["vegetation"]["cached"].__setitem__("version", 2),
        "cached compact version": lambda m: cp(m).__setitem__("version", 3.0),
        "cached extra key": lambda m: m["vegetation"]["cached"].__setitem__("junk", 1),
        "cached layout": lambda m: m["vegetation"]["cached"]["controls"].__setitem__("a3_layout", "bogus"),
        "scalar not canonical": lambda m: m["scalars"].__setitem__("theta_residual", "0.05"),
    }
    for fn in edits.values():
        m2 = copy.deepcopy(meta)
        fn(m2)
        with pytest.raises(ERR):
            parse(m2, arrays)
    for name in list(arrays):
        a2 = dict(arrays)
        del a2[name]
        with pytest.raises(ERR):
            parse(meta, a2)
        a3 = dict(arrays)
        a3[name] = arrays[name].astype(np.float32 if arrays[name].dtype.kind == "f" else np.int16)
        with pytest.raises(ERR):
            parse(meta, a3)
        a4 = dict(arrays)
        a4[name] = np.concatenate([arrays[name], np.zeros((1,) + arrays[name].shape[1:], dtype=arrays[name].dtype)])
        with pytest.raises(ERR):
            parse(meta, a4)
    # value / pin / area / file-count / op-field / mask tampers that PARSE but must be refused by the canonical reconstruction
    def refused(edit_meta=None, edit_arrays=None):
        m2, a2 = copy.deepcopy(meta), {k: np.array(x) for k, x in arrays.items()}
        if edit_meta:
            edit_meta(m2)
        if edit_arrays:
            edit_arrays(a2)
        with pytest.raises(ERR):
            reconstruct(parse(m2, a2), pins, good)

    refused(lambda m: cp(m).__setitem__("cell_area_hex", float(w.vertical.held_template().column.cell_area_m2 * 2).hex()))       # wrong area
    refused(lambda m: m["vegetation"]["cached"]["package"].__setitem__("file_count", 999))                                         # wrong LEAF file count
    refused(lambda m: m["vegetation"]["cached"]["package"].__setitem__("file_count", True))
    refused(lambda m: m["vegetation"]["cached"].__setitem__("mask_sha256", "7" * 64))                                              # wrong mask identity
    refused(lambda m: m["identities"].__setitem__("matrix_op_sha256", "8" * 64))                                                   # operator ids are recomputed, never trusted
    refused(lambda m: m["vegetation"]["law"].__setitem__("law_sha256", "9" * 64))
    refused(lambda m: cp(m).__setitem__("shape", list(reversed(cp(m)["shape"]))) if cp(m)["shape"][0] != cp(m)["shape"][1] else m["vegetation"]["cached"].__setitem__("surface_sha256", "6" * 64))
    refused(None, lambda a: a["active"].__setitem__(0, ~a["active"][0]))                                                           # the held mask
    for name in ("veg.entry_area_live_m2", "veg.matrix_root_index", "veg.weights", "veg.root", "depth0_m", "theta_field_capacity", "grain_texture"):
        def tweak(a, name=name):
            flat = a[name].reshape(-1)
            flat[0] = flat[0] * 1.0000001 + 1e-9
        refused(None, tweak)


def test_the_hook_mints_only_checked_parts(dense):
    w = dense
    r = export(w.vertical, w.rec, w.pins)
    meta = r.meta()
    parts = fhv.prepare_parts(r, w.pins, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    assert set(parts) == {"config", "column", "matrix_op", "preferential", "depth0", "ks", "active", "area", "edges", "host_depth0", "host_active", "host_ks", "cached", "law"}
    fh.require_held_context(fh._mint_vegetation_context(dict(parts), meta, w.pins))
    for key, val in (("cached", object()), ("law", object()), ("config", object()), ("matrix_op", parts["column"]), ("preferential", None), ("column", object())):
        with pytest.raises((ERR, AttributeError, TypeError)):
            fh._mint_vegetation_context({**parts, key: val}, meta, w.pins)
    with pytest.raises(ERR):
        fh._mint_vegetation_context({k: v for k, v in parts.items() if k != "law"}, meta, w.pins)
    with pytest.raises(ERR):
        fh._mint_vegetation_context(dict(parts), meta, fhv.ExpectedVegetationPins(**{**w.pins.__dict__, "record_sha256": "0" * 64}))
    for build in (lambda: fh.HeldVegetation(parts["cached"], "a" * 64, "b" * 64, "c" * 64, "d" * 64), lambda: fh.HeldContext(*([None] * 12)), lambda: fhv.VegetationHeldRecord({}, {}),
                  lambda: copy.copy(r), lambda: copy.deepcopy(r)):
        with pytest.raises(ERR):
            build()
    assert fh.HeldContext.__dataclass_fields__["vegetation"].default is None


# ------------------------------------------------------------------------------------------------ memory
def test_admission_refuses_need_minus_one_before_any_factory_and_a_retry_succeeds(dense):
    w = dense
    r = export(w.vertical, w.rec, w.pins)
    est = fhv.estimate_vegetation_held_bytes(r.meta())
    assert est["stages"]["cache"] > est["live_outside_bytes"] and est["stages"]["law"] > est["live_outside_bytes"] and est["reconstruct_bytes"] >= est["record_bytes"]
    cm = r.meta()["vegetation"]["cached"]
    cache_est = wh.estimate_cached_stage_bytes_from_meta(cm, source_cold=True, with_mask=True)                  # the FINAL public cache model feeds the stage
    assert est["stages"]["cache"] == est["live_outside_bytes"] + cache_est["peak_bytes"]
    from maple_syrup import floating_matrix_infiltration as fmi
    from maple_syrup import floating_soil_layers as fsl
    from maple_syrup import native_floating_soil as nfs

    def boom(*a, **k):
        raise AssertionError("a public factory / the cached stage ran BEFORE the admission refused")

    with pytest.MonkeyPatch.context() as mp:
        for mod, name in ((nfs, "native_floating_config"), (fsl, "prepare_floating_column"), (fmi, "prepare_floating_matrix"), (wh, "stage_cached_inputs"), (wh, "cached_binding"), (wvb, "prepare_vegetation_law")):
            mp.setattr(mod, name, boom)
        with pytest.raises(ERR, match="nothing was allocated|available"):
            reconstruct(r, w.pins, memory_budget_bytes=est["reconstruct_bytes"] - 1)
        with pytest.raises(ERR):
            reconstruct(r, w.pins, available_bytes_fn=lambda: est["reconstruct_bytes"] - 1)
        with pytest.raises(ERR):
            reconstruct(r, w.pins, reserve_bytes=1, memory_budget_bytes=est["reconstruct_bytes"])
        with pytest.raises(ERR):
            export(w.vertical, w.rec, w.pins, memory_budget_bytes=1)
        with pytest.raises(ERR):
            export(w.vertical, w.rec, w.pins, available_bytes_fn=lambda: 1)
    meta, arrays = serialise(r)
    for kw in ({"max_bytes": sum(a.nbytes for a in arrays.values()) - 1}, {"available_bytes_fn": lambda: 10}):
        with pytest.raises(ERR):
            fhv.record_from_parts_vegetation(meta, arrays, **kw)
    assert reconstruct(r, w.pins).vegetation is not None                                                        # the retry succeeds
    reconstruct(r, w.pins, memory_budget_bytes=est["reconstruct_bytes"])                                        # the exact modelled cap passes (the cache stage runs under the SAME numbers)
    reconstruct(r, w.pins, reserve_bytes=7, memory_budget_bytes=est["reconstruct_bytes"] + 7, available_bytes_fn=lambda: est["reconstruct_bytes"] + 7)   # the reserve counted once


def test_a_hostile_cached_metadata_document_is_bounded_before_any_serialisation(dense):
    w = dense
    r = export(w.vertical, w.rec, w.pins)
    meta, arrays = serialise(r)
    m2 = copy.deepcopy(meta)
    m2["vegetation"]["cached"]["identity"]["unexpected_large_list"] = ["x"] * 2_000_000                       # input created BEFORE the measured call
    import tracemalloc
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    with pytest.raises(ERR):
        fhv.record_from_parts_vegetation(m2, arrays, available_bytes_fn=AVAIL)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    assert peak < 1 << 20                                                                                      # the cache's bounded walk refuses it; no 2M-element copy / JSON text exists


def test_the_record_holds_one_compact_recipe_and_no_per_species_n_arrays(dense):
    w = dense
    r = export(w.vertical, w.rec, w.pins)
    names = set(r.arrays())
    assert not any("contribution" in n for n in names) and "veg.entry_cell" in names and "veg.receiver_weights_enabled" in names
    assert all(not x.flags.writeable for x in r.arrays().values())
    ctx = reconstruct(r, w.pins)
    assert ctx.vegetation.cached is not None and type(ctx.vegetation).__name__ == "HeldVegetation"


def test_the_live_leaf_session_and_original_objects_can_be_freed_before_the_restore(leaf, tmp_path):
    import weakref
    w = World(leaf, tmp_path / "free", "numpy", False)
    r = export(w.vertical, w.rec, w.pins)
    meta, arrays = serialise(r)
    pins = w.pins
    wrefs = [weakref.ref(w.stack["proj"]), weakref.ref(w.binding)]
    del w, r
    gc.collect()
    assert all(x() is None for x in wrefs)                                                                      # the projection / binding are gone: the cached record outlives them
    ctx = reconstruct(fhv.record_from_parts_vegetation(meta, arrays, available_bytes_fn=AVAIL), pins)
    assert ctx.vegetation.cached.record_sha256 == pins.record_sha256_hex


def test_ordinary_profiles_stay_leaf_free(tmp_path):
    import os
    import subprocess
    import sys
    code = ("import sys, maple_syrup.floating_held_context as fh, maple_syrup.floating_held_context_cuda; assert 'leaf' not in sys.modules, 'LEAF imported by the old profile'; "
            "assert 'maple_syrup.floating_held_context_vegetation' not in sys.modules; print('ok')")
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=dict(os.environ), check=False)
    assert out.stdout.strip() == "ok", out.stderr


def test_typed_errors_only_no_blanket_allocator_memory_error_catch():
    for mod in (fhv, __import__("maple_syrup.floating_held_context_vegetation_cuda", fromlist=["x"])):
        code = "\n".join(ln.split("#")[0] for ln in inspect.getsource(mod).splitlines())
        assert not re.search(r"(?<![A-Za-z])MemoryError", code), mod.__name__                                    # the typed Floating / Subsurface / Vegetation memory errors are caught through their families only
    code = inspect.getsource(fhv)
    assert "except (wvb.VegetationBindingError, *_ERRORS)" in code


def test_the_native_memory_error_hierarchy_is_what_the_contract_says():
    """`FloatingMemoryError` is a `SubsurfaceMemoryError` (a `MemoryError`) and NEITHER a `FloatingSoilError` nor a `SubsurfaceError`: it needs its own entry in the typed list."""
    from maple_syrup import floating_soil_layers as fsl
    from maple_syrup import subsurface as ss
    assert issubclass(fsl.FloatingMemoryError, ss.SubsurfaceMemoryError) and issubclass(ss.SubsurfaceMemoryError, MemoryError)
    assert not issubclass(fsl.FloatingMemoryError, (fsl.FloatingSoilError, ss.SubsurfaceError))
    assert ss.SubsurfaceMemoryError in fhv._ERRORS and MemoryError not in fhv._ERRORS and BaseException not in fhv._ERRORS and Exception not in fhv._ERRORS
    from maple_syrup import floating_held_context_vegetation_cuda as fvc
    assert "*fhv._ERRORS" in inspect.getsource(fvc.prepare_parts_cuda) and "*fhv._ERRORS" in inspect.getsource(fvc.export_held_context_vegetation_cuda)


def test_typed_native_memory_refusals_are_wrapped_and_a_bare_allocator_memory_error_propagates(dense, monkeypatch):
    from maple_syrup import floating_soil_layers as fsl
    from maple_syrup import subsurface as ss
    w = dense
    r = export(w.vertical, w.rec, w.pins)

    def raising(exc):
        def f(*a, **k):
            raise exc
        return f

    cases = ((fsl, "prepare_floating_column", fsl.FloatingMemoryError("typed floating memory refusal")), (wvb, "prepare_vegetation_law", ss.SubsurfaceMemoryError("typed subsurface memory refusal")),
             (wvb, "prepare_vegetation_law", fsl.FloatingMemoryError("typed floating memory refusal in the law")))
    for mod, name, exc in cases:
        with monkeypatch.context() as mp:                                                                        # reconstruct: the typed refusal becomes a HeldContextError (specific contract), cause suppressed
            mp.setattr(mod, name, raising(exc))
            with pytest.raises(ERR, match="refused"):
                reconstruct(r, w.pins)
        if mod is wvb:
            with monkeypatch.context() as mp:                                                                    # export: the same wrapping on the law preparation
                mp.setattr(mod, name, raising(exc))
                with pytest.raises(ERR, match="canonical law|CURRENT operators"):
                    export(w.vertical, w.rec, w.pins)
    for mod, name in ((fsl, "prepare_floating_column"), (wvb, "prepare_vegetation_law")):
        bare = MemoryError("a REAL allocator failure")
        with monkeypatch.context() as mp:
            mp.setattr(mod, name, raising(bare))
            with pytest.raises(MemoryError) as info:
                reconstruct(r, w.pins)
            assert info.value is bare and not isinstance(info.value, ERR)                                        # not swallowed, not converted
    with monkeypatch.context() as mp:
        mp.setattr(wvb, "prepare_vegetation_law", raising(bare))
        with pytest.raises(MemoryError):
            export(w.vertical, w.rec, w.pins)
    assert reconstruct(r, w.pins).vegetation is not None                                                         # nothing was mutated: the unpatched retry succeeds


def test_the_old_fhc_admit_broad_catch_is_unchanged_scope():
    from maple_syrup import floating_held_context_cuda as fhc
    assert "MemoryError" in inspect.getsource(fhc._admit)                                                        # the published CUDA helper keeps its own (old) scope; the new wrappers do not widen or copy it


def test_fresh_child_restores_from_only_the_written_record_with_pinned_sources(dense, tmp_path):
    """A fresh interpreter loads ONLY the serialised record (json + npz) and the external pins: no projection, binding, plants, session, bed, boundary or original vertical exists in it; the pinned MAPLE / LEAF package
    digests are asserted inside the child before and after the work."""
    import os
    import subprocess
    import sys

    from hc_sources import LEAF_PACKAGE, MAPLE_PACKAGE, MAPLE_ROOT
    w = dense
    r = export(w.vertical, w.rec, w.pins)
    meta, arrays = serialise(r)
    np.savez(tmp_path / "arrays.npz", **{k.replace(".", "__"): a for k, a in arrays.items()})
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    p = w.pins
    (tmp_path / "pins.json").write_text(json.dumps({"record_sha256": p.record_sha256, "binding_sha256": p.binding_sha256, "projection_sha256": p.projection_sha256, "leaf_package_sha256": p.leaf_package_sha256,
                                                    "external_helpers": p.external_helpers, "source": p.source}))
    code = f"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ['LEAF_CANDIDATE_ROOT'])
import numpy as np, maple, leaf
from maple_syrup.provenance import source_tree_digest
assert Path(maple.__file__).resolve().is_relative_to(Path({json.dumps(str(MAPLE_ROOT / 'source' / 'src'))}))
assert source_tree_digest(Path(maple.__file__).resolve().parent).digest_sha256 == {json.dumps(MAPLE_PACKAGE)}
assert source_tree_digest(Path(leaf.__file__).resolve().parent).digest_sha256 == {json.dumps(LEAF_PACKAGE)}
from maple_syrup import floating_held_context as fh, floating_held_context_vegetation as fhv
meta = json.loads(Path({json.dumps(str(tmp_path / 'meta.json'))}).read_text())
z = np.load({json.dumps(str(tmp_path / 'arrays.npz'))})
arrays = {{k.replace('__', '.'): np.array(z[k]) for k in z.files}}
pins = fhv.ExpectedVegetationPins(**json.loads(Path({json.dumps(str(tmp_path / 'pins.json'))}).read_text()))
rec = fhv.record_from_parts_vegetation(meta, arrays, available_bytes_fn=lambda: 1 << 40)
i = meta['identities']
ctx = fh.reconstruct_vegetation_held_context(rec, fh.ExpectedHeldIdentities(i['config_sha256'], i['root_column_sha256'], i['matrix_op_sha256'], i['pref_sha256']), pins, memory_budget_bytes=1 << 40, available_bytes_fn=lambda: 1 << 40)
assert (ctx.matrix_op.sha256, ctx.preferential.sha256) == (i['matrix_op_sha256'], i['pref_sha256'])
assert 'leaf.plant_session' not in sys.modules and 'maple_syrup.vegetation_lifecycle' not in sys.modules and 'maple_syrup.floating_event_lifecycle' not in sys.modules
assert source_tree_digest(Path(maple.__file__).resolve().parent).digest_sha256 == {json.dumps(MAPLE_PACKAGE)}
print('child-ok', ctx.vegetation.law_sha256)
"""
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=dict(os.environ), check=False)
    assert out.returncode == 0 and out.stdout.strip().startswith("child-ok"), out.stderr[-2000:]
    assert out.stdout.strip().split()[-1] == reconstruct(r, w.pins).vegetation.law_sha256
