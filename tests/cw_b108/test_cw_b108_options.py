"""CW-B108: deep-chunk storage option validation, pre-mutation refusals, checkpoint metadata and cross-encoding /
bound resume refusals (accepted typed source; CPU NumPy).

Every refusal is checked to leave the store files, any bundle files and the caller's state unchanged.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil

import numpy as np
import pytest

pytest.importorskip("maple")

from b108_fixture import H, assert_same_arrays, build, dense_bed, sediment_parameters, state_arrays

from maple_syrup import sediment_bed
from maple_syrup.checkpoint import _OPTIONAL_DEFAULT_FIELDS, load_checkpoint, save_checkpoint
from maple_syrup.complete_event import CompletionPolicy, EventProgress, complete_event
from maple_syrup.sediment_bed import (
    TYPED_JOURNAL_SOURCE_DIGESTS,
    BedContext,
    BedIntegrationError,
    bound_maple_package_digest,
    check_deep_storage_options,
    deep_storage_publish_options,
    deep_storage_record,
    validate_deep_storage,
)
from maple_syrup.sediment_event import SedimentEventError, evolve_sediment_event

pytestmark = pytest.mark.skipif(bound_maple_package_digest() not in TYPED_JOURNAL_SOURCE_DIGESTS,
                                reason="needs an explicitly pinned typed MAPLE source")

POLICY = CompletionPolicy(hold_s=2.0, max_depth_m=5e-4, surface_volume_m3=1e-2, max_unit_discharge_m2_s=1e-5,
                          outlet_discharge_m3_s=1e-4, deposit_mobile_at_flow_end=True)  # LABELLED (events file)
IDENTITY = {"case": "cw_b108 options", "source": "test"}
BAD_OPTIONS = [("Full", None), ("raw", None), (None, None), (b"full", None), (1, None), ("full", 10),
               ("overlay", 0), ("overlay", -1), ("overlay", True), ("overlay", 1.5), ("overlay", np.int64(64)),
               ("overlay", "64")]


def files(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def evolve(s, ctx, control, schedule, end=4.0, state=None):
    return evolve_sediment_event(state or s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"],
                                 s["sediment"], end, control, report_every_s=2.0)


def test_defaults_are_full_and_forward_nothing():
    fields = {f.name: f.default for f in dataclasses.fields(BedContext)}
    assert fields["deep_chunk_encoding"] == "full" and fields["deep_overlay_max_bytes"] is None
    assert list(fields)[-2:] == ["deep_chunk_encoding", "deep_overlay_max_bytes"]  # appended LAST
    _, ctx = dense_bed()
    assert deep_storage_publish_options(ctx) == {} and deep_storage_record(ctx) is None
    over = dataclasses.replace(ctx, deep_chunk_encoding="overlay", deep_overlay_max_bytes=4096)
    assert deep_storage_publish_options(over) == {"encoding": "overlay", "overlay_max_bytes": 4096}
    assert deep_storage_record(over) == {"deep_chunk_encoding": "overlay", "deep_overlay_max_bytes": 4096}
    assert _OPTIONAL_DEFAULT_FIELDS["BedContext"] == {"window_max_journal_records": None,
                                                      "deep_chunk_encoding": "full", "deep_overlay_max_bytes": None}
    from maple_syrup.event_experiment import _deep_store_counters

    counters = _deep_store_counters()
    assert set(counters) == {"deep_store", "overlay"} and "overlay_slot_materializations" in counters["overlay"]


@pytest.mark.parametrize("encoding,bound", BAD_OPTIONS)
def test_invalid_options_are_refused_before_any_work(tmp_path, encoding, bound):
    with pytest.raises(BedIntegrationError):
        check_deep_storage_options(encoding, bound)
    s, ctx, control, schedule = build(np, tmp_path / "s")
    ctx = dataclasses.replace(ctx, deep_chunk_encoding=encoding, deep_overlay_max_bytes=bound)
    before, store = state_arrays(s["state0"]), files(tmp_path / "s")
    with pytest.raises(SedimentEventError):
        evolve(s, ctx, control, schedule)
    assert files(tmp_path / "s") == store
    assert_same_arrays(state_arrays(s["state0"]), before)
    from maple_syrup.windowed_experiment import run_complete_windowed

    with pytest.raises(SedimentEventError, match="deep_"):  # refused before case verification / setup / output
        run_complete_windowed(tmp_path / "no_case", tmp_path / "no_output", case_kind="rfid",
                              store_dir=tmp_path / "no_store", deep_chunk_encoding=encoding,
                              deep_overlay_max_bytes=bound, nodata_value=-9999.0)
    assert not (tmp_path / "no_output").exists() and not (tmp_path / "no_store").exists()


def test_overlay_on_a_dense_bed_is_refused():
    from test_coupled_backends import column_for_xp, graph_for_xp
    from test_sediment_event import rainfall_field

    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_event import SedimentEventControl, initial_event_state
    from maple_syrup.storm import StormControl

    bed, ctx = dense_bed()
    ctx = dataclasses.replace(ctx, deep_chunk_encoding="overlay")
    with pytest.raises(BedIntegrationError, match="resident-window"):
        validate_deep_storage(ctx, bed)
    control = SedimentEventControl(storm=StormControl(max_dt_s=1.0, implementation="numba", max_steps=100))
    graph, terrain = graph_for_xp(bed, np)
    params, soil0 = column_for_xp(graph, 1e-5, np)
    state0 = initial_event_state(graph, terrain, bed, dataclasses.replace(ctx, deep_chunk_encoding="full"), soil0,
                                 control=control)
    before = state_arrays(state0)
    with pytest.raises(SedimentEventError, match="resident-window"):
        evolve_sediment_event(state0, ctx, params, rainfall_field(*graph.shape, xp=np), constant_rainfall(0, 2, 50),
                              np.zeros(graph.shape), sediment_parameters(np), 2.0, control, report_every_s=2.0)
    assert_same_arrays(state_arrays(state0), before)


def test_full_context_cannot_continue_an_overlay_store(tmp_path):
    s, ctx, control, schedule = build(np, tmp_path / "s", encoding="overlay")
    r = evolve(s, ctx, control, schedule, end=20.0)
    assert r.state.bed.voxel_column.deep_store.format_version == 2
    default = dataclasses.replace(ctx, deep_chunk_encoding="full", deep_overlay_max_bytes=None)
    store = files(tmp_path / "s")
    with pytest.raises(SedimentEventError, match="overlay \\(v2\\)"):
        evolve(s, default, control, schedule, end=24.0, state=r.state)
    assert files(tmp_path / "s") == store
    # An overlay continuation with another explicit bound is allowed by MAPLE's format (bounds govern only new
    # overlays) but a checkpoint resume with a changed bound is refused (below): the bound is part of the run identity.


def paused(tmp_path, label, encoding, bound=None):
    s, ctx, control, schedule = build(np, tmp_path / f"store_{label}", encoding=encoding, overlay_max_bytes=bound)
    box = {}

    def callback(progress):
        if progress.result.state.t_s == 12.0:
            box["p"] = progress
            return True
        return False

    out = complete_event(s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"], s["sediment"],
                         max_end_s=300.0, control=control, policy=POLICY, report_every_s=2.0,
                         checkpoint_callback=callback)
    assert isinstance(out, EventProgress)
    path = tmp_path / f"bundle_{label}"
    save_checkpoint(path, out, ctx, s["column"], s["sediment"], IDENTITY)
    return out, s, ctx, path


def test_cross_encoding_and_bound_resume_is_refused_without_mutation(tmp_path):
    over, s, ctx_over, bundle_over = paused(tmp_path, "over", "overlay")
    full, s_full, ctx_full, bundle_full = paused(tmp_path, "full", "full")
    watched = [tmp_path / "store_over", tmp_path / "store_full", bundle_over, bundle_full]
    before = [files(p) for p in watched]
    cases = [(bundle_over, ctx_full, "cross-encoding"),
             (bundle_over, dataclasses.replace(ctx_over, deep_overlay_max_bytes=4096), "cross-encoding"),
             (bundle_full, dataclasses.replace(ctx_full, deep_chunk_encoding="overlay"), "cross-encoding"),
             (bundle_over, dataclasses.replace(ctx_over, deep_chunk_encoding="raw"), "deep_chunk_encoding")]
    for bundle, ctx, message in cases:
        with pytest.raises(SedimentEventError, match=message):
            load_checkpoint(bundle, ctx, s["column"], s["sediment"], IDENTITY)
    assert [files(p) for p in watched] == before
    # The matching contexts load (same backend, same encoding / bound).
    assert load_checkpoint(bundle_over, ctx_over, s["column"], s["sediment"], IDENTITY).result.state.t_s == 12.0
    assert load_checkpoint(bundle_full, ctx_full, s["column"], s["sediment"], IDENTITY).result.state.t_s == 12.0
    assert [files(p) for p in watched] == before


@pytest.mark.parametrize("tamper", ["physical", "logical", "format", "default_recorded", "extra_key", "dense_tag",
                                    "removed"])
def test_tampered_deep_storage_records_are_refused(tmp_path, tamper):
    over, s, ctx, bundle = paused(tmp_path, "over", "overlay")
    copy = tmp_path / "tampered"
    shutil.copytree(bundle, copy)
    manifest = json.loads((copy / "checkpoint.json").read_text())
    record = manifest["deep_storage"]
    if tamper == "physical":
        record["physical_manifest_digest"] = "0" * 64
    elif tamper == "logical":
        record["logical_manifest_digest"] = "0" * 64
    elif tamper == "format":
        record["store_format_version"] = 1
    elif tamper == "default_recorded":
        record.update(deep_chunk_encoding="full", deep_overlay_max_bytes=None)
    elif tamper == "extra_key":
        record["note"] = "x"
    elif tamper == "dense_tag":
        manifest.pop("bed_storage")
    else:
        manifest.pop("deep_storage")
    (copy / "checkpoint.json").write_text(json.dumps(manifest))
    with pytest.raises(SedimentEventError):
        load_checkpoint(copy, ctx, s["column"], s["sediment"], IDENTITY)


def test_save_refuses_an_option_the_bed_cannot_carry(tmp_path):
    over, s, ctx, bundle = paused(tmp_path, "over", "overlay")
    default = dataclasses.replace(ctx, deep_chunk_encoding="full", deep_overlay_max_bytes=None)
    with pytest.raises(SedimentEventError, match="overlay \\(v2\\)"):
        save_checkpoint(tmp_path / "refused", over, default, s["column"], s["sediment"], IDENTITY)
    assert not (tmp_path / "refused").exists()
    assert not any(p.name.startswith(".refused") for p in tmp_path.iterdir())


def test_default_bundle_and_identity_records_are_unchanged(tmp_path):
    from maple_syrup.event_experiment import control_identity

    full, s, ctx, bundle = paused(tmp_path, "full", "full")
    manifest = json.loads((bundle / "checkpoint.json").read_text())
    assert set(manifest) == {"schema", "identity", "status", "files", "payload", "backend", "bed_storage",
                             "phase_storage"}
    record = control_identity(full.control)
    assert not any(k.startswith("deep_chunk") or k.startswith("deep_overlay") for k in record)
    assert "deep_storage" not in json.dumps(manifest["payload"])


def test_windowed_cli_exposes_the_option_with_full_default(capsys):
    import inspect

    from maple_syrup import windowed_experiment

    for function in (windowed_experiment.run_complete_windowed, windowed_experiment._setup_windowed):
        parameters = inspect.signature(function).parameters
        assert parameters["deep_chunk_encoding"].default == "full"
        assert parameters["deep_overlay_max_bytes"].default is None
    with pytest.raises(SystemExit):
        windowed_experiment.main(["--help"])
    text = capsys.readouterr().out
    assert "--deep-chunk-encoding {full,overlay}" in text and "--deep-overlay-max-bytes" in text


def test_overlay_publication_forwards_the_options_to_maple(tmp_path, monkeypatch):
    from maple.surface.voxels import windowed_pending

    seen = []
    real = windowed_pending.publish_pending_column

    def spy(column, *args, **kwargs):
        seen.append({k: v for k, v in kwargs.items() if k != "max_records"})
        return real(column, *args, **kwargs)

    monkeypatch.setattr(windowed_pending, "publish_pending_column", spy)
    for encoding, bound in (("full", None), ("overlay", None), ("overlay", 2048)):
        seen.clear()
        s, ctx, control, schedule = build(np, tmp_path / f"s_{encoding}_{bound}", encoding=encoding,
                                          overlay_max_bytes=bound)
        evolve(s, ctx, control, schedule)
        expected = {} if encoding == "full" else {"encoding": encoding, "overlay_max_bytes": bound}
        assert seen and all(call == expected for call in seen)
    assert float(H(s["state0"].storm.depth_m).max()) > 0
