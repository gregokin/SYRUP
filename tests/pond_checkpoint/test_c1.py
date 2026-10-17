"""C1 regressions (not run by their author): 0-d payload shapes, bounded `.npy` header preflight before any allocation, tiny
reset-receipt corruption against MAPLE's summation policy, and the exact JSON identity vocabulary."""

from __future__ import annotations

import json
import shutil
import struct
import tracemalloc
import zipfile

import numpy as np
import pytest

from maple_syrup import pond_checkpoint as pc
from maple_syrup.pond_checkpoint import PondCheckpointError, load_pond_checkpoint, save_pond_checkpoint

from .pond_fixture import IDENTITY, advance, dry_case, fingerprint, host, make_case, tree_hash
from .test_refusals import edit_array, rewrite


@pytest.fixture(scope="module")
def wet(tmp_path_factory):
    case = make_case(pickup_kg_per_class=5e-4)
    event = case.event()
    paused = advance(event, event.initial_state(case.bed, case.soil), 4)
    path = save_pond_checkpoint(tmp_path_factory.mktemp("c1_wet") / "ck", paused, event, IDENTITY)
    return case, event, paused, path


@pytest.fixture(scope="module")
def done(tmp_path_factory):
    case = dry_case()
    event = case.event()
    handoff = event.complete(event.initial_state(case.bed, case.soil), 120.0)
    path = save_pond_checkpoint(tmp_path_factory.mktemp("c1_done") / "ck", handoff, event, IDENTITY)
    return case, event, handoff, path


@pytest.fixture
def wet_ck(wet, tmp_path):
    case, _event, _paused, base = wet
    target = tmp_path / "ck"
    shutil.copytree(base, target)
    return case, target


@pytest.fixture
def done_ck(done, tmp_path):
    case, _event, _handoff, base = done
    target = tmp_path / "ck"
    shutil.copytree(base, target)
    return case, target


# --- 0-d vs 1-d ------------------------------------------------------------------------------------------------------
SCALAR_FIELDS = ("wl_export", "sl_numerical_residual", "sl_phase_residual_max", "sl_n_phase_canonicalized", "sl_n_phase_remnants")


def test_zero_d_ledger_scalars_stay_zero_d_in_table_file_and_loaded_state(wet):
    case, _event, paused, path = wet
    meta = json.loads((path / pc.MANIFEST_NAME).read_text())
    table = meta["payload"]["arrays"]
    with np.load(path / pc.PAYLOADS[1]) as z:
        for name in SCALAR_FIELDS:
            assert table[name]["shape"] == [], f"{name} was written with shape {table[name]['shape']}"
            assert z[name].shape == ()
        for name, rec in table.items():  # every array, native ones included, is stored with exactly its table shape
            assert list(z[name].shape) == rec["shape"], name
    loaded = load_pond_checkpoint(path, case.event(), IDENTITY)
    sl = loaded.sediment
    for value in (loaded.water.export, sl.numerical_residual_kg, sl.phase_residual_max_kg, sl.n_phase_canonicalized,
                  sl.n_phase_remnants):
        assert host(value).shape == ()


def test_encoders_and_digest_preserve_the_exact_shape():
    assert pc._digest_of(np.asarray(1.0)) != pc._digest_of(np.asarray([1.0]))
    stored = []

    def add(array):
        stored.append(array)
        return f"n{len(stored)}"

    from maple_syrup.pond_checkpoint import _native_encode

    assert _native_encode(np.asarray(3.0), add) == {"array": "n1"}
    assert _native_encode(np.asarray([3.0]), add) == {"array": "n2"}
    assert stored[0].shape == () and stored[1].shape == (1,)
    assert _native_encode(np.zeros((0, 3)), add) == {"array": "n3"} and stored[2].shape == (0, 3)


def test_a_one_d_scalar_field_is_refused_by_the_schema(wet_ck):
    case, path = wet_ck

    def promote(data, meta):
        data["sl_n_phase_canonicalized"] = data["sl_n_phase_canonicalized"].reshape(1)
        meta["payload"]["arrays"]["sl_n_phase_canonicalized"]["shape"] = [1]

    rewrite(path, arrays=promote)
    with pytest.raises(PondCheckpointError, match="shape"):
        load_pond_checkpoint(path, case.event(), IDENTITY)


# --- bounded .npy header preflight ----------------------------------------------------------------------------------
def npy(descr="<f8", shape=(1,), payload=b"\x00" * 8, *, fortran=False, magic=b"\x93NUMPY\x01\x00", pad_to=None):
    text = f"{{'descr': {descr!r}, 'fortran_order': {fortran!r}, 'shape': {shape!r}, }}".encode("latin1")
    total = 10 + len(text) + 1
    text = text + b" " * ((-total) % 64) + b"\n"
    if pad_to is not None:
        text = b" " * pad_to + b"\n"
    return magic + struct.pack("<H", len(text)) + text + payload


def replace_member(archive_path, member, raw):
    with zipfile.ZipFile(archive_path) as z:
        items = {i.filename: z.read(i.filename) for i in z.infolist()}
    assert member in items, member
    items[member] = raw
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as z:
        for key, value in items.items():
            z.writestr(key, value)


HUGE = 10**6  # 8 TB if allocated
HEADER_CASES = [
    ("huge_shape_companion", pc.PAYLOADS[1], "hyd_soil.npy", lambda: npy(shape=(HUGE, HUGE)), "before any allocation"),
    ("huge_shape_native_snapshot", pc.PAYLOADS[0], "voxel_mass_kg.npy", lambda: npy(shape=(HUGE, HUGE, HUGE)), "before any allocation"),
    ("object_dtype", pc.PAYLOADS[1], "hyd_soil.npy", lambda: npy(descr="|O", shape=(1,), payload=b"\x00" * 8), "not allowed|malformed"),
    # The native archive has no companion shape table: reach the bounded header reader itself.
    ("oversized_header", pc.PAYLOADS[0], "voxel_mass_kg.npy", lambda: npy(pad_to=60000), "malformed"),
    # Keep this member inside the declared-size guard so its invalid magic reaches the header reader.
    ("bad_magic", pc.PAYLOADS[1], "hyd_soil.npy", lambda: b"NOTNUMPY" + b"\x00" * 208, "malformed"),
    ("fortran_order", pc.PAYLOADS[1], "hyd_soil.npy", lambda: npy(shape=(3, 4), fortran=True, payload=b"\x00" * 96), "C-ordered"),
    ("header_differs_from_table", pc.PAYLOADS[1], "hyd_soil.npy", lambda: npy(shape=(4, 3), payload=b"\x00" * 96), "manifest table"),
    ("truncated_data", pc.PAYLOADS[1], "hyd_soil.npy", lambda: npy(shape=(3, 4), payload=b"\x00" * 8), "before any allocation"),
]


@pytest.mark.parametrize(("name", "archive", "member", "make", "match"), HEADER_CASES, ids=[c[0] for c in HEADER_CASES])
def test_malformed_npy_header_is_refused_before_any_allocation(wet_ck, name, archive, member, make, match):
    case, path = wet_ck
    replace_member(path / archive, member, make())
    rewrite(path)  # hashes and manifest re-sealed: the refusal must come from the header audit itself
    before = tree_hash(path)
    tracemalloc.start()
    try:
        with pytest.raises(PondCheckpointError, match=match):
            load_pond_checkpoint(path, case.event(), IDENTITY)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 256 * 1024**2, f"peak traced allocation {peak} B: the header was expanded before it was refused"
    assert tree_hash(path) == before


def test_default_cap_is_not_the_way_around_a_bad_header(wet_ck):
    case, path = wet_ck
    replace_member(path / pc.PAYLOADS[1], "hyd_soil.npy", npy(shape=(HUGE, HUGE)))
    rewrite(path)
    with pytest.raises(PondCheckpointError, match="before any allocation"):
        load_pond_checkpoint(path, case.event(), IDENTITY, max_bytes=pc.DEFAULT_MAX_CHECKPOINT_BYTES)


def test_manifest_and_source_checks_precede_the_header_audit(wet_ck):
    case, path = wet_ck
    replace_member(path / pc.PAYLOADS[1], "hyd_soil.npy", npy(shape=(HUGE, HUGE)))
    rewrite(path, manifest=lambda m: m["source"].update(maple_package_sha256="0" * 64))
    with pytest.raises(PondCheckpointError, match="source pair"):
        load_pond_checkpoint(path, case.event(), IDENTITY)


# --- reset receipts against MAPLE's summation policy -------------------------------------------------------------------
TINY = 1e-15  # surface / terminal scale: ~1e-16 m3 / kg, far below the old 1e-9 floor
TINY_SOIL = 1e-12  # the soil volume of this fixture is O(0.1 m3): its honest FP64 summation bound is ~1e-16 m3


def bump(name, amount=TINY):
    return edit_array(name, lambda a: np.where(np.arange(a.size).reshape(a.shape) == 0, a + amount, a))


def test_untouched_handoff_receipts_pass_the_native_bounds(done_ck):
    case, path = done_ck
    loaded = load_pond_checkpoint(path, case.event(), IDENTITY, allow_complete=True)
    assert loaded.status == "complete"


@pytest.mark.parametrize(("array", "amount", "match"), [("wl_reset_depth", TINY, "surface_removed_m3"),
                                                        ("wl_reset_soil", TINY_SOIL, "soil_removed_m3")])
def test_tiny_reset_array_corruption_is_not_hidden_by_a_tolerance(done_ck, array, amount, match):
    """A 1e-15 m change of one recorded reset surface depth (re-signed manifest) changes the removed volume by 2.5e-16 m3:
    the old 1e-9 floor accepted that, the native summation bound at the real operand scale must not."""
    case, path = done_ck
    rewrite(path, arrays=bump(array, amount))
    with pytest.raises(PondCheckpointError, match=match):
        load_pond_checkpoint(path, case.event(), IDENTITY, allow_complete=True)


@pytest.mark.parametrize(("key", "amount"), [("surface_removed_m3", TINY), ("soil_removed_m3", TINY_SOIL)])
def test_tiny_declared_receipt_corruption_is_refused(done_ck, key, amount):
    case, path = done_ck
    rewrite(path, manifest=lambda m: m["payload"]["accounting"]["dict"].update({key: m["payload"]["accounting"]["dict"][key] + amount}))
    with pytest.raises(PondCheckpointError, match=key):
        load_pond_checkpoint(path, case.event(), IDENTITY, allow_complete=True)


def test_tiny_terminal_receipt_corruption_is_refused(done_ck):
    case, path = done_ck

    def edit(m):
        node = m["payload"]["accounting"]["dict"]["terminal_deposition_by_class_kg"]["list"]
        node[0] = node[0] + TINY

    rewrite(path, manifest=edit)
    with pytest.raises(PondCheckpointError, match="terminal deposition"):
        load_pond_checkpoint(path, case.event(), IDENTITY, allow_complete=True)


# --- exact identity vocabulary ------------------------------------------------------------------------------------------
BAD_IDENTITIES = [{1: "a", "1": "b"}, {"a": {2: 1}}, {None: 1}, {True: 1}, {"a": [{3.5: 1}]}, {"a": object()}, {"a": float("nan")}]


@pytest.mark.parametrize("bad", BAD_IDENTITIES, ids=range(len(BAD_IDENTITIES)))
def test_identity_with_coercible_keys_or_values_is_refused_on_save_and_load(wet, tmp_path, bad):
    case, event, paused, path = wet
    before = fingerprint(paused)
    with pytest.raises(PondCheckpointError, match="identity"):
        save_pond_checkpoint(tmp_path / "ck", paused, event, bad)
    assert list(tmp_path.iterdir()) == [] and fingerprint(paused) == before
    with pytest.raises(PondCheckpointError, match="identity"):
        load_pond_checkpoint(path, case.event(), bad)


def test_numeric_key_cannot_collide_with_its_string_form(wet, tmp_path):
    case, event, paused, _path = wet
    save_pond_checkpoint(tmp_path / "ok", paused, event, {"1": "b"})
    with pytest.raises(PondCheckpointError, match="string keys"):
        load_pond_checkpoint(tmp_path / "ok", case.event(), {1: "b"})
