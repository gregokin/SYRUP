"""CC1 builders: a GENUINE continued state (real native thin-lake storm -> genuine completion -> selected refresh -> accepted boundary -> `begin_occurrence`), its H0 root record and a same-process / fresh-process restart TARGET.
No fake capability, producer or owner: every object comes from the public lifecycle."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from case import snapshot
from lifecycle_builders import AVAIL, build, run_to_complete

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_context_checkpoint as fcc
from maple_syrup import floating_continued_checkpoint as cc
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_held_context as fh
from maple_syrup import floating_pond as fp

IDENT = {"case": "cc1", "v": 1}
RAIN = 100.0 / 3.6e6


def lineage(tmp, backend, windowed):
    """`(ns)` after one genuine refresh boundary: `ns.state` is the continued state (carried routing generation >= 1, local counters 0), `ns.record` the H0 record of the REFRESHED root exported from the boundary."""
    owner, bed, factory, pref, ev0 = build(tmp, backend, windowed, loose=True, mobile_kg=1.0)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL, preferential=pref)
    h = run_to_complete(occ.event, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    record = lc.export_boundary_held_context(b, available_bytes_fn=AVAIL)
    occ2, s2 = lc.begin_occurrence(lin, b, factory, available_bytes_fn=AVAIL)
    return SimpleNamespace(lin=lin, occ=occ2, event=occ2.event, state=s2, boundary=b, record=record, factory=factory, ev0=ev0, original=ev0._initial_topology, windowed=windowed, backend=backend)


def restart_target(ns, ck, *, record=None):
    """A restart TARGET from public pieces only: a vertical from the H0 reconstruction of the refreshed root and the event built on the carried routing origin read back from the file."""
    rec = ns.record if record is None else record
    ctx = fh.reconstruct_held_context(rec, fcc._expected_from(rec.meta()), available_bytes_fn=AVAIL)
    vert = fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL)
    origin = cc.read_continued_origin(ck, IDENT, available_bytes_fn=AVAIL)
    return ns.factory(vert, cc.origin_topology(origin, ns.original)), vert, origin


def save(ns, path):
    return cc.save_floating_continued_checkpoint(path, ns.state, ns.event, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)


def run(ev, st, n, rate=RAIN, dt=0.2):
    for _ in range(n):
        st = ev.step(st, rate, dt).state
    return st


def record_to_files(record, directory):
    """The H0 record as plain files (strict JSON meta + `.npz` arrays): the fresh-process test's only carrier of the REFRESHED root."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps(record.meta()))
    np.savez(directory / "arrays.npz", **{k: np.ascontiguousarray(a) for k, a in record.arrays().items()})


def record_from_files(directory):
    directory = Path(directory)
    with np.load(directory / "arrays.npz", allow_pickle=False) as z:
        arrays = {k: np.array(z[k], copy=True) for k in z.files}
    return fh.record_from_parts(json.loads((directory / "meta.json").read_text()), arrays, available_bytes_fn=AVAIL)


__all__ = ["AVAIL", "IDENT", "RAIN", "cc", "fc", "lineage", "record_from_files", "record_to_files", "restart_target", "run", "save", "snapshot"]
