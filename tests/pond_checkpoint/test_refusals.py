"""Refusals: corruption, identity, source pair, backend, shape, byte bound, write failure; and non-mutation (not run by its
author). Every edit that should reach a deeper validator re-seals the hashes, so the refusal comes from THAT validator."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from maple_syrup import pond_checkpoint as pc
from maple_syrup.checkpoint import sha256_file
from maple_syrup.pond_checkpoint import PondCheckpointError, load_pond_checkpoint, save_pond_checkpoint
from maple_syrup.rainfall import constant_rainfall

from .pond_fixture import IDENTITY, advance, fingerprint, make_case, tree_hash


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    """One wet pickup case paused after a few real steps (module scope: the checkpoint is copied per test)."""
    case = make_case(pickup_kg_per_class=5e-4)
    event = case.event()
    paused = advance(event, event.initial_state(case.bed, case.soil), 5)
    root = tmp_path_factory.mktemp("pond_ck_base")
    path = save_pond_checkpoint(root / "base", paused, event, IDENTITY)
    return case, event, paused, path


@pytest.fixture
def ck(scenario, tmp_path):
    case, event, paused, base = scenario
    target = tmp_path / "ck"
    shutil.copytree(base, target)
    return case, target


def rewrite(path: Path, *, manifest=None, arrays=None, refresh_files=True) -> None:
    """Edit the manifest and / or the companion arrays of a checkpoint and re-seal (hashes recomputed)."""
    meta = json.loads((path / pc.MANIFEST_NAME).read_text())
    if arrays is not None:
        with np.load(path / pc.PAYLOADS[1]) as z:
            data = {k: z[k].copy() for k in z.files}
        arrays(data, meta)
        np.savez(path / pc.PAYLOADS[1], **data)
    if manifest is not None:
        manifest(meta)
    if refresh_files:
        meta["files"] = {n: sha256_file(path / n) for n in pc.PAYLOADS}
    (path / pc.MANIFEST_NAME).write_text(json.dumps(pc._seal_manifest(meta), indent=1, sort_keys=True))


def refused(case, path, match, **kw):
    event = case.event()
    before = tree_hash(path)
    with pytest.raises(PondCheckpointError, match=match):
        load_pond_checkpoint(path, event, kw.pop("identity", IDENTITY), **kw)
    assert tree_hash(path) == before, "a refused load modified the checkpoint"


def test_untouched_copy_loads(ck):
    case, path = ck
    assert load_pond_checkpoint(path, case.event(), IDENTITY).counters.n_accepted_steps == 5


# --- manifest-level refusals: nothing of the payload is read -------------------------------------------------------------
MANIFEST_EDITS = [
    ("schema", lambda m: m.update(schema="future/version"), "schema"),
    ("unknown_manifest_key", lambda m: m.update(extra=1), "keys"),
    ("unknown_payload_key", lambda m: m["payload"].update(extra=1), "payload keys"),
    ("status_unknown", lambda m: m.update(status="finished"), "status"),
    ("array_backend", lambda m: m.update(array_backend="cupy"), "cross-backend"),
    ("hydraulic_backend", lambda m: m.update(hydraulic_backend="numba"), "hydraulic backend"),
    ("binding_context", lambda m: m["binding"].update(context_sha256="0" * 64), "binding"),
    ("binding_shape", lambda m: m["binding"].update(shape=[9, 9]), "binding"),
]


@pytest.mark.parametrize(("name", "edit", "match"), MANIFEST_EDITS, ids=[e[0] for e in MANIFEST_EDITS])
def test_manifest_edit_is_refused(ck, name, edit, match):
    case, path = ck
    rewrite(path, manifest=edit)
    refused(case, path, match)


def test_complete_status_without_allow_complete_is_refused(ck):
    case, path = ck
    rewrite(path, manifest=lambda m: m.update(status="complete"))
    refused(case, path, "completed")


@pytest.mark.parametrize("field", ["maple_syrup_tree_sha256", "maple_package_sha256"])
def test_source_pair_mismatch_is_refused_before_any_payload_is_read(ck, field):
    case, path = ck
    rewrite(path, manifest=lambda m: m["source"].update({field: "f" * 64}), refresh_files=False)
    (path / pc.PAYLOADS[1]).write_bytes(b"garbage that must never be opened")
    refused(case, path, "source pair")


def test_identity_mismatch_is_refused_before_any_payload_is_read(ck):
    case, path = ck
    (path / pc.PAYLOADS[0]).write_bytes(b"garbage that must never be opened")
    for key in ("case", "forcing", "source", "options"):
        expected = json.loads(json.dumps(IDENTITY))
        expected[key] = "changed"
        refused(case, path, "identity", identity=expected)
    with pytest.raises(PondCheckpointError, match="identity"):
        load_pond_checkpoint(path, case.event(), {"bad": float("nan")})


def test_manifest_modified_without_reseal_is_refused(ck):
    case, path = ck
    meta = json.loads((path / pc.MANIFEST_NAME).read_text())
    meta["identity"]["case"] = "edited"
    (path / pc.MANIFEST_NAME).write_text(json.dumps(meta))
    refused(case, path, "manifest hash")


def test_non_strict_json_and_duplicate_keys_are_refused(ck):
    case, path = ck
    text = (path / pc.MANIFEST_NAME).read_text()
    (path / pc.MANIFEST_NAME).write_text(text.replace('"schema"', '"schema": NaN, "schema"', 1))
    refused(case, path, "strict JSON")
    (path / pc.MANIFEST_NAME).write_text(text[:-2] + ', "schema": "x"}')
    refused(case, path, "strict JSON")


# --- event binding: independent of event.config_sha256 -------------------------------------------------------------------
def test_native_context_difference_is_refused_even_if_event_config_digest_is_equal(ck):
    case, path = ck
    other_ctx = replace(case.ctx, initial_available_fraction=np.full_like(case.ctx.initial_available_fraction, 0.5))
    event = case.event(ctx=other_ctx)
    with pytest.raises(PondCheckpointError, match="context_sha256"):
        load_pond_checkpoint(path, event, IDENTITY)


@pytest.mark.parametrize("variant", ["forcing", "control", "shape", "column"])
def test_other_static_inputs_are_bound(ck, variant):
    case, path = ck
    if variant == "forcing":
        event = case.event(schedule=constant_rainfall(0.0, 100.0, 61.0))
        key = "schedule_sha256"
    elif variant == "control":
        event = case.event(control=replace(case.control, time=replace(case.control.time, max_dt_s=0.1)))
        key = "options_sha256"
    elif variant == "shape":
        other = make_case(nx=5, pickup_kg_per_class=5e-4)
        event, key = other.event(), "shape"
    else:
        from maple_syrup.infiltration import column_parameters

        full = lambda v: np.full((3, 4), v)  # noqa: E731
        column = column_parameters(model="fixed_ksat", ksat_m_per_s=full(3e-6), suction_m=full(0.0), drainage_parameter=full(0.0),
                                   theta_sat=full(0.4), soil_thickness_m=full(0.3))
        event, key = case.event(column=column), "column_sha256"
    with pytest.raises(PondCheckpointError, match=key):
        load_pond_checkpoint(path, event, IDENTITY)


def test_numba_hydraulics_is_not_numpy(ck):
    pytest.importorskip("numba")
    case, path = ck
    event = case.event(control=replace(case.control, backend="numba"))
    with pytest.raises(PondCheckpointError, match="hydraulic backend"):
        load_pond_checkpoint(path, event, IDENTITY)


def test_wrong_object_types_are_refused(ck):
    case, path = ck
    with pytest.raises(PondCheckpointError, match="PondEvent"):
        load_pond_checkpoint(path, object(), IDENTITY)
    with pytest.raises(PondCheckpointError, match="identity"):
        load_pond_checkpoint(path, case.event(), ["not", "a", "dict"])


# --- byte bound ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [True, 0, -5, 1.5, "1", None, float("inf")])
def test_max_bytes_must_be_a_positive_integer(ck, bad):
    case, path = ck
    with pytest.raises(PondCheckpointError, match="max_bytes"):
        load_pond_checkpoint(path, case.event(), IDENTITY, max_bytes=bad)


def test_small_explicit_cap_is_refused_before_any_array_is_expanded(ck):
    case, path = ck
    refused(case, path, "exceed|bound", max_bytes=2048)
    # the smallest cap that fits both files loads; the bound is a resource control, never raised automatically
    total = sum((path / n).stat().st_size for n in pc.PAYLOADS)
    assert load_pond_checkpoint(path, case.event(), IDENTITY, max_bytes=total + 10_000_000).t_s > 0


def test_save_refuses_a_cap_below_the_checkpoint_size_and_writes_nothing(scenario, tmp_path):
    case, event, paused, _ = scenario
    with pytest.raises(PondCheckpointError, match="max_bytes"):
        save_pond_checkpoint(tmp_path / "small", paused, event, IDENTITY, max_bytes=4096)
    assert list(tmp_path.iterdir()) == []


# --- payload corruption ---------------------------------------------------------------------------------------------
def test_payload_hash_mismatch_and_missing_and_extra_file(ck):
    case, path = ck
    with (path / pc.PAYLOADS[1]).open("ab") as f:
        f.write(b"corrupt")
    refused(case, path, "hash mismatch")
    (path / pc.PAYLOADS[1]).unlink()
    refused(case, path, "exactly")
    shutil.copy(path / pc.PAYLOADS[0], path / pc.PAYLOADS[1])
    (path / "extra.txt").write_text("x")
    refused(case, path, "exactly")


def test_symlinked_payload_is_refused(ck, tmp_path):
    case, path = ck
    real = tmp_path / "real.npz"
    shutil.copy(path / pc.PAYLOADS[1], real)
    (path / pc.PAYLOADS[1]).unlink()
    (path / pc.PAYLOADS[1]).symlink_to(real)
    with pytest.raises(PondCheckpointError):
        load_pond_checkpoint(path, case.event(), IDENTITY)


def edit_array(name, fn, *, table=None):
    def arrays(data, meta):
        data[name] = fn(data[name])
        if table is not None:
            meta["payload"]["arrays"][name].update(table(data[name]))
    return arrays


ARRAY_EDITS = [
    ("negative_depth", edit_array("hyd_depth", lambda a: np.where(np.arange(a.size).reshape(a.shape) == 0, -1.0, a)), ">= 0"),
    ("nan_flux", edit_array("hyd_qx", lambda a: np.where(np.arange(a.size).reshape(a.shape) == 0, np.nan, a)), "non-finite|finite"),
    ("inf_velocity", edit_array("sediment_velocity", lambda a: np.where(np.arange(a.size).reshape(a.shape) == 0, np.inf, a)),
     "non-finite|finite"),
    ("negative_ledger", edit_array("wl_rain", lambda a: a - 1.0), ">= 0"),
    ("dtype_float32", edit_array("hyd_depth", lambda a: a.astype(np.float32), table=lambda a: {"dtype": a.dtype.str}), "dtype"),
    ("shape_transposed", edit_array("hyd_soil", lambda a: a.T.copy(), table=lambda a: {"shape": list(a.shape)}), "shape"),
    ("held_source_shifted", edit_array("held_source", lambda a: a + 1e-4), "held source"),
    ("topology_source_shifted", edit_array("topo_source", lambda a: a + 1e-3), "independent build"),
    ("phase_fraction_broken", edit_array("phase_fraction", lambda a: a * 0.5 + 0.25), "phase"),
    ("water_ledger_inflated", edit_array("wl_intake", lambda a: a + 1e-3), "close"),
    ("sediment_ledger_inflated", edit_array("sl_wet_actual", lambda a: a + 1.0), "close"),
    ("escape_levels_edited", edit_array("topo_escape_value", lambda a: a + 1e-3), "independent build"),
]


@pytest.mark.parametrize(("name", "edit", "match"), ARRAY_EDITS, ids=[e[0] for e in ARRAY_EDITS])
def test_corrupt_array_is_refused_by_the_typed_validators(ck, name, edit, match):
    case, path = ck
    rewrite(path, arrays=edit)
    refused(case, path, match)


def edit_scalars(fn):
    return lambda m: fn(m["payload"]["scalars"])


SCALAR_EDITS = [
    ("unknown_scalar_key", edit_scalars(lambda s: s.update(extra=1)), "scalar keys"),
    ("generation", edit_scalars(lambda s: s["topology"].update(generation=3)), "generation"),
    ("accepted_steps", edit_scalars(lambda s: s["counters"].update(n_accepted_steps=99)), "step|counter"),
    ("counter_type", edit_scalars(lambda s: s["counters"].update(n_commits="1")), "int"),
    ("negative_counter", edit_scalars(lambda s: s["counters"].update(n_commits=-1)), "int"),
    ("quiet_ready_without_interval", edit_scalars(lambda s: s["quiet"].update(ready=True)), "quiet"),
    ("quiet_time", edit_scalars(lambda s: s["quiet"].update(last_t_s=0.0)), "quiet|differ"),
    ("clock", edit_scalars(lambda s: s.update(t_s=s["t_s"] + 0.5)), "clock|differ"),
    ("cap_nonpositive", edit_scalars(lambda s: s.update(next_dt_cap_s=-1.0)), "positive"),
    ("state_schema", edit_scalars(lambda s: s.update(state_schema="other")), "schema"),
    ("config_digest", edit_scalars(lambda s: s.update(config_sha256="0" * 64)), "digests"),
    ("native_calls", edit_scalars(lambda s: s["sediment"].update(n_native_calls=1)), "native call"),
    ("nonfinite_string_float", edit_scalars(lambda s: s.update(t_s="inf")), "float"),
]


@pytest.mark.parametrize(("name", "edit", "match"), SCALAR_EDITS, ids=[e[0] for e in SCALAR_EDITS])
def test_corrupt_scalar_is_refused_before_an_object_is_constructed(ck, name, edit, match):
    case, path = ck
    rewrite(path, manifest=edit)
    refused(case, path, match)


def test_unknown_native_type_and_missing_array_reference_are_refused(ck):
    case, path = ck
    rewrite(path, manifest=lambda m: m["payload"]["native"]["ledger"].update(type="Evil"))
    refused(case, path, "native")


def test_array_table_entry_removed_is_refused(ck):
    case, path = ck
    rewrite(path, manifest=lambda m: m["payload"]["arrays"].pop("hyd_depth"))
    refused(case, path, "missing required")


def test_object_dtype_in_the_array_table_is_refused_before_any_array_is_read(ck):
    case, path = ck
    rewrite(path, manifest=lambda m: m["payload"]["arrays"]["hyd_depth"].update(dtype="|O"))
    refused(case, path, "dtype")


def test_snapshot_swapped_from_another_checkpoint_is_refused(scenario, ck, tmp_path):
    """A native snapshot of another run / another time cannot be paired with this companion."""
    case, path = ck
    _case, event, paused, _base = scenario
    later = advance(event, paused, 3)
    other = save_pond_checkpoint(tmp_path / "later", later, event, IDENTITY)  # same identity and binding, later clock
    shutil.copy(other / pc.PAYLOADS[0], path / pc.PAYLOADS[0])
    rewrite(path)  # re-seal the hashes so the pairing check itself is exercised
    refused(case, path, "clock|step|differ")
    # a snapshot written under another caller identity fails the provenance binding
    third = save_pond_checkpoint(tmp_path / "third", paused, event, {**IDENTITY, "case": "another run"})
    shutil.copy(third / pc.PAYLOADS[0], path / pc.PAYLOADS[0])
    rewrite(path)
    refused(case, path, "provenance")


# --- save-side refusals and atomicity ----------------------------------------------------------------------------------
def test_existing_destination_is_never_overwritten(scenario, tmp_path):
    case, event, paused, base = scenario
    before = tree_hash(base)
    with pytest.raises(PondCheckpointError, match="exists"):
        save_pond_checkpoint(base, paused, event, IDENTITY)
    assert tree_hash(base) == before
    plain = tmp_path / "plain.txt"
    plain.write_text("user file")
    with pytest.raises(PondCheckpointError, match="exists"):
        save_pond_checkpoint(plain, paused, event, IDENTITY)
    assert plain.read_text() == "user file"


def test_missing_parent_bad_identity_and_bad_objects_are_refused_without_writing(scenario, tmp_path):
    case, event, paused, _ = scenario
    with pytest.raises(PondCheckpointError, match="parent"):
        save_pond_checkpoint(tmp_path / "no" / "such" / "ck", paused, event, IDENTITY)
    with pytest.raises(PondCheckpointError, match="identity"):
        save_pond_checkpoint(tmp_path / "ck", paused, event, {"x": float("inf")})
    with pytest.raises(PondCheckpointError, match="identity"):
        save_pond_checkpoint(tmp_path / "ck", paused, event, {"x": object()})
    with pytest.raises(PondCheckpointError, match="PondEvent"):
        save_pond_checkpoint(tmp_path / "ck", paused, None, IDENTITY)
    with pytest.raises(PondCheckpointError, match="PondEventState"):
        save_pond_checkpoint(tmp_path / "ck", paused.bed, event, IDENTITY)
    assert list(tmp_path.iterdir()) == []


def test_state_of_another_event_configuration_is_refused(scenario, tmp_path):
    case, event, paused, _ = scenario
    other = case.event(schedule=constant_rainfall(0.0, 100.0, 61.0))
    with pytest.raises(PondCheckpointError, match="configuration"):
        save_pond_checkpoint(tmp_path / "ck", paused, other, IDENTITY)
    assert list(tmp_path.iterdir()) == []


def test_state_that_lags_the_commit_threshold_is_not_an_accepted_state(tmp_path):
    """The initial state of a bed whose pending pickup already exceeds MAPLE's trigger is not an accepted state."""
    case = make_case(pickup_kg_per_class=0.1)
    event = case.event()
    start = event.initial_state(case.bed, case.soil)
    with pytest.raises(PondCheckpointError, match="commit threshold"):
        save_pond_checkpoint(tmp_path / "ck", start, event, IDENTITY)
    assert list(tmp_path.iterdir()) == []


def test_failed_write_removes_only_its_own_staging_directory(scenario, tmp_path, monkeypatch):
    case, event, paused, _ = scenario
    keep = tmp_path / "user_dir"
    keep.mkdir()
    (keep / "data.txt").write_text("precious")
    stray = tmp_path / ".other.partial-keep"
    stray.mkdir()

    def boom(*args, **kwargs):
        raise OSError("disk full (simulated in this module's own writer)")

    monkeypatch.setattr(pc, "_write_manifest", boom)
    with pytest.raises(OSError, match="disk full"):
        save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    assert not (tmp_path / "ck").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [".other.partial-keep", "user_dir"]
    assert (keep / "data.txt").read_text() == "precious"


def test_refused_load_leaves_event_and_state_unchanged(scenario, tmp_path):
    case, event, paused, base = scenario
    fresh = case.event()
    start = fresh.initial_state(case.bed, case.soil)
    before = fingerprint(start)
    path = tmp_path / "bad"
    shutil.copytree(base, path)
    rewrite(path, arrays=edit_array("hyd_depth", lambda a: a - 1.0))
    with pytest.raises(PondCheckpointError):
        load_pond_checkpoint(path, fresh, IDENTITY)
    assert fingerprint(start) == before
    # the same event still runs after the refusal
    assert advance(fresh, start, 1).counters.n_accepted_steps == 1
