"""CC1 correction01: `read_continued_origin` audits the companion's table and EVERY `.npy` header (historical bounded checks) BEFORE any array is decoded. Genuine native file, companion RESEALED (a recomputed hash is not authenticity); an `np.load` tripwire proves nothing is decoded; no large real allocation."""
from __future__ import annotations

import io
import json
import shutil
import zipfile

import numpy as np
import pytest
from cont_builders import AVAIL, IDENT, cc, fc, lineage, save

from maple_syrup import pond_checkpoint as pc

REFUSE = (fc.FloatingCheckpointError, pc.PondCheckpointError)


class Reached(BaseException):
    pass


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    d = tmp_path_factory.mktemp("hdr")
    ns = lineage(d / "a", "numpy", False)
    return ns, save(ns, d / "ck"), d


def _npy(a):
    b = io.BytesIO()
    np.save(b, a, allow_pickle=False)
    return b.getvalue()


def _reseal(good_ck, tmp, name, member=None, manifest=None):
    d = tmp / name
    shutil.copytree(good_ck, d)
    z = d / "pond_continuation.npz"
    if member is not None:
        with zipfile.ZipFile(z) as f:
            members = {n: f.read(n) for n in f.namelist()}
        member(members)
        with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_STORED) as f:
            for n, b in members.items():
                f.writestr(n, b)
    m = json.loads((d / fc.MANIFEST_NAME).read_text())
    m["files"][z.name] = pc._hash_file(z, 1 << 40)[0]
    if manifest:
        manifest(m)
    (d / fc.MANIFEST_NAME).write_text(fc._json_dumps(fc._seal(m)))
    return d


def _refused_without_decoding(ck, rx, monkeypatch):
    reads = []

    def trip(*a, **k):
        reads.append(str(a[0]))
        raise Reached

    monkeypatch.setattr(np, "load", trip)
    with pytest.raises(REFUSE, match=rx):
        cc.read_continued_origin(ck, IDENT, available_bytes_fn=AVAIL)
    assert reads == []


def test_the_untouched_file_still_reads(good):
    _ns, ck, _d = good
    assert cc.read_continued_origin(ck, IDENT, available_bytes_fn=AVAIL).routing_generation >= 1


def test_resealed_wrong_shape_dtype_size_and_member_set_are_refused_before_any_decode(good, tmp_path, monkeypatch):
    ns, ck, _d = good
    n = ns.state.topology.source_z_full.shape
    cases = {
        "big": (lambda m: m.__setitem__("co_routing_source.npy", _npy(np.zeros((100, 100)))), "header|member|archive"),
        "dtype": (lambda m: m.__setitem__("co_routing_source.npy", _npy(np.zeros(n, dtype=np.float32))), "header|member|archive"),
        "fortran": (lambda m: m.__setitem__("co_routing_source.npy", _npy(np.asfortranarray(np.arange(n[0] * n[1], dtype=float).reshape(n)))), "header|member|archive|differs|C-ordered"),
        "object": (lambda m: m.__setitem__("co_routing_source.npy", _npy(np.array([None] * 3, dtype=object)) if False else b"\x93NUMPY\x01\x00"), "header|member|archive"),
        "short": (lambda m: m.__setitem__("co_routing_source.npy", m["co_routing_source.npy"][:-8]), "header|member|archive"),
        "extra": (lambda m: m.__setitem__("stray.npy", _npy(np.zeros(3))), "members|table|archive member"),
        "missing_other": (lambda m: m.pop("datum_full.npy"), "members|table|archive member"),
    }
    for name, (edit, rx) in cases.items():
        _refused_without_decoding(_reseal(ck, tmp_path, name, member=edit), rx, monkeypatch)


def test_invalid_manifest_origin_fields_are_typed_refusals_before_any_read(good, tmp_path, monkeypatch):
    _ns, ck, _d = good

    def at(f):
        return lambda m: f(m)

    cases = {
        "shape_type": at(lambda m: m["binding"].update(shape=[3.0, 5])),
        "shape_huge": at(lambda m: m["binding"].update(shape=[10**6, 10**6])),
        "shape_neg": at(lambda m: m["binding"].update(shape=[-1, 5])),
        "rec_dtype": at(lambda m: m["payload"]["arrays"]["co_routing_source"].update(dtype="<f4")),
        "rec_shape": at(lambda m: m["payload"]["arrays"]["co_routing_source"].update(shape=[100, 100])),
        "table_bad": at(lambda m: m["payload"]["arrays"].__setitem__("datum_full", {"dtype": "<f8"})),
        "table_huge": at(lambda m: m["payload"]["arrays"]["datum_full"].update(shape=[10**9, 10**9])),
        "hex": at(lambda m: m["binding"]["initial_topology"].update(source_sha256="XYZ")),
        "gen_type": at(lambda m: m["binding"]["initial_topology"].update(generation="1")),
        "init_keys": at(lambda m: m["binding"]["initial_topology"].update(extra=1)),
        "files_set": at(lambda m: m["files"].pop("maple_state.npz")),
        "payload_none": at(lambda m: m.update(payload=None)),
        "schema_old": at(lambda m: m.update(schema=fc.SCHEMA_V3)),
        "identity": at(lambda m: m.update(identity={"case": "other"})),
    }
    for name, edit in cases.items():
        d = _reseal(ck, tmp_path, name, manifest=edit)
        reads = []
        monkeypatch.setattr(np, "load", lambda *a, _r=reads, **k: (_r.append(a[0]), (_ for _ in ()).throw(Reached()))[1])
        with pytest.raises(REFUSE):
            cc.read_continued_origin(d, IDENT, available_bytes_fn=AVAIL)
        assert reads == [], name
        monkeypatch.undo()


def test_a_wrong_source_pair_is_refused_before_any_payload(good, tmp_path, monkeypatch):
    _ns, ck, _d = good
    d = _reseal(ck, tmp_path, "src", manifest=lambda m: m["source"].update(maple_syrup="0" * 64) if "maple_syrup" in m["source"] else m.update(source={}))
    _refused_without_decoding(d, "source pair", monkeypatch)
