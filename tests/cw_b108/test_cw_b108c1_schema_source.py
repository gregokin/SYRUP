"""CW-B108C1: strict checkpoint `deep_storage` schema (refused before any payload read, nothing written, caller objects
unchanged) and the unsupported-source refusal order of the windowed runner. Own small fixtures only; no real case or
production model is run."""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import shutil

import pytest

pytest.importorskip("maple")

from maple_syrup import checkpoint
from maple_syrup.sediment_bed import (
    JOURNAL_RECOGNIZER_MAPLE_DIGEST,
    TYPED_JOURNAL_SOURCE_DIGESTS,
    bound_maple_package_digest,
)
from maple_syrup.sediment_event import SedimentEventError

TYPED = bound_maple_package_digest() in TYPED_JOURNAL_SOURCE_DIGESTS  # explicit typed sources: 24c6 and the merged candidate
M94 = bound_maple_package_digest() == JOURNAL_RECOGNIZER_MAPLE_DIGEST


def files(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def tampered(tmp_path, bundle, edit):
    copy = tmp_path / f"tampered_{len(list(tmp_path.glob('tampered_*')))}"
    shutil.copytree(bundle, copy)
    manifest = json.loads((copy / "checkpoint.json").read_text())
    edit(manifest)
    (copy / "checkpoint.json").write_text(json.dumps(manifest))
    return copy


def refuse_without_reads(monkeypatch, copy, ctx, s, identity, match="deep_storage record is malformed"):
    """The refusal must happen BEFORE any payload is hashed or opened (the payload hash reader is never called)."""
    calls = []
    real = checkpoint.sha256_file
    monkeypatch.setattr(checkpoint, "sha256_file", lambda p: calls.append(p) or real(p))
    ctx_before = [(f.name, getattr(ctx, f.name)) for f in dataclasses.fields(ctx)]
    fraction = ctx.initial_available_fraction.tobytes()
    with pytest.raises(SedimentEventError, match=match):
        checkpoint.load_checkpoint(copy, ctx, s["column"], s["sediment"], identity)
    assert calls == [] and all(getattr(ctx, name) is value for name, value in ctx_before)
    assert ctx.initial_available_fraction.tobytes() == fraction
    monkeypatch.setattr(checkpoint, "sha256_file", real)


@pytest.mark.skipif(not TYPED, reason="needs the accepted typed MAPLE source")
@pytest.mark.parametrize("value", [2.0, True, "2", None, 3, 0, [2]])
def test_overlay_record_format_must_be_a_supported_plain_int(tmp_path, monkeypatch, value):
    from test_cw_b108_options import IDENTITY, paused

    over, s, ctx, bundle = paused(tmp_path, "over", "overlay")
    watched = {"store": files(tmp_path / "store_over"), "bundle": files(bundle)}
    copy = tampered(tmp_path, bundle, lambda m: m["deep_storage"].__setitem__("store_format_version", value))
    refuse_without_reads(monkeypatch, copy, ctx, s, IDENTITY)
    assert {"store": files(tmp_path / "store_over"), "bundle": files(bundle)} == watched
    # the untouched bundle still loads (the actual version binding check is retained)
    assert checkpoint.load_checkpoint(bundle, ctx, s["column"], s["sediment"], IDENTITY).result.state.t_s == 12.0


@pytest.mark.skipif(not TYPED, reason="needs the accepted typed MAPLE source")
@pytest.mark.parametrize("key", ["physical_manifest_digest", "logical_manifest_digest"])
@pytest.mark.parametrize("value", [None, 7, "A" * 64, "0" * 63, "g" * 64, ["0" * 64]])
def test_overlay_record_digests_must_be_hex_strings(tmp_path, monkeypatch, key, value):
    from test_cw_b108_options import IDENTITY, paused

    over, s, ctx, bundle = paused(tmp_path, "over", "overlay")
    copy = tampered(tmp_path, bundle, lambda m: m["deep_storage"].__setitem__(key, value))
    refuse_without_reads(monkeypatch, copy, ctx, s, IDENTITY)


@pytest.mark.skipif(not TYPED, reason="needs the accepted typed MAPLE source")
def test_valid_format_wrong_version_is_still_refused_by_the_actual_binding(tmp_path):
    from test_cw_b108_options import IDENTITY, paused

    over, s, ctx, bundle = paused(tmp_path, "over", "overlay")
    copy = tampered(tmp_path, bundle, lambda m: m["deep_storage"].__setitem__("store_format_version", 1))
    with pytest.raises(SedimentEventError, match="disagrees with the loaded deep version"):
        checkpoint.load_checkpoint(copy, ctx, s["column"], s["sediment"], IDENTITY)


@pytest.mark.skipif(not TYPED, reason="needs the accepted typed MAPLE source")
@pytest.mark.parametrize("value", [None, {}, [], "full", {"deep_chunk_encoding": "full", "deep_overlay_max_bytes": None}])
def test_default_bundle_with_a_present_deep_storage_key_is_malformed(tmp_path, monkeypatch, value):
    from test_cw_b108_options import IDENTITY, paused

    full, s, ctx, bundle = paused(tmp_path, "full", "full")
    watched = files(bundle)
    copy = tampered(tmp_path, bundle, lambda m: m.__setitem__("deep_storage", value))
    refuse_without_reads(monkeypatch, copy, ctx, s, IDENTITY)
    assert files(bundle) == watched
    assert checkpoint.load_checkpoint(bundle, ctx, s["column"], s["sediment"], IDENTITY).result.state.t_s == 12.0


def test_setup_checks_the_source_before_importing_the_overlay_constant():
    """Static order in `_setup_windowed`: `check_deep_storage_source(...)` precedes the import of
    `OVERLAY_MAX_VOXELS` (which the M94 source does not define)."""
    from maple_syrup import windowed_experiment

    tree = ast.parse(inspect.getsource(windowed_experiment._setup_windowed))
    source_line = min(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                      and getattr(n.func, "id", None) == "check_deep_storage_source")
    import_line = min(n.lineno for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                      and any(a.name == "OVERLAY_MAX_VOXELS" for a in n.names))
    assert source_line < import_line


@pytest.mark.skipif(not M94, reason="unsupported-source refusal is exercised on M94")
@pytest.mark.parametrize("bound", [None, 4096])
def test_public_runner_refuses_overlay_on_m94_before_any_setup(tmp_path, bound):
    from maple.surface.voxels import deep_store

    from maple_syrup.windowed_experiment import run_complete_windowed

    assert not hasattr(deep_store, "OVERLAY_MAX_VOXELS")  # the constant the old order imported first
    with pytest.raises(SedimentEventError, match="accepted MAPLE source"):
        run_complete_windowed(tmp_path / "no_case", tmp_path / "no_output", case_kind="rfid",
                              store_dir=tmp_path / "no_store", deep_chunk_encoding="overlay",
                              deep_overlay_max_bytes=bound, nodata_value=-9999.0)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


@pytest.mark.skipif(not M94, reason="default-full M94 call")
def test_public_runner_default_full_on_m94_passes_the_option_check(tmp_path):
    """The default call reaches the existing case verification unchanged (it then refuses the missing case for its
    own reason); the deep-storage option never refuses it and nothing is created."""
    from maple_syrup.windowed_experiment import run_complete_windowed

    with pytest.raises(Exception) as info:
        run_complete_windowed(tmp_path / "no_case", tmp_path / "no_output", case_kind="rfid",
                              store_dir=tmp_path / "no_store", nodata_value=-9999.0)
    assert "deep_" not in str(info.value) and "accepted MAPLE source" not in str(info.value)
    assert not (tmp_path / "no_output").exists() and not (tmp_path / "no_store").exists()
