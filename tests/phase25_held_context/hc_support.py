"""Builders of the dynamic-vegetation held-context suite: REAL native lifecycle boundaries (genuine storm -> closed Boundary, Dense and published Windowed, NumPy / Numba) -> the lifecycle's OWN `apply_vegetation_generation`
(ACTUAL LEAF P1 projection + S0 binding) -> `FloatingPondVertical.from_boundary(new root)`: a REAL vertical that actually holds the dynamic law. No private minting, no generation / head injection, no fake class."""
from __future__ import annotations

import cached_support as cs
import numpy as np
import veg_lc_builders as vb
from bridge_builders import AVAIL, dry_boundary, dry_boundary_changed

from maple_syrup import floating_held_context as fh
from maple_syrup import floating_held_context_vegetation as fhv
from maple_syrup import floating_pond as fp
from maple_syrup import vegetation_lifecycle as vl
from maple_syrup import water_vegetation_binding as wvb
from maple_syrup import water_vegetation_held as wh

BIG = vb.BIG
BACKENDS = ["numpy"] + (["numba"] if __import__("importlib").util.find_spec("numba") is not None else [])


def source_of(proj) -> dict:
    ident = proj.identity_record()
    return {k: ident["source"][k] for k in ("kind", "state_sha256", "catalog_hash")}


def pins_of(proj, binding, rec) -> fhv.ExpectedVegetationPins:
    return fhv.ExpectedVegetationPins(rec.record_sha256, binding.sha256, proj.identity_record()["content_sha256"], vb.leaf_pin(), vb.helper_pin(), source_of(proj))


class World:
    """One genuine dynamic vertical and everything needed to export / restore it."""

    def __init__(self, leaf, tmp, backend="numpy", windowed=False, changed=None):
        if changed is None:
            self.lin, self.b, self.factory = dry_boundary(tmp, backend, windowed)
        else:
            self.lin, self.b, self.factory, _orig = dry_boundary_changed(tmp, changed, backend, windowed)
        self.stack = vb.stack(leaf, self.b)
        self.binding = vb.binding(leaf, self.b, self.stack)
        self.nb = vl.apply_vegetation_generation(self.lin, self.b, self.binding, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
        self.vertical = fp.FloatingPondVertical.from_boundary(self.nb.root, available_bytes_fn=AVAIL)
        self.rec = wh.capture_cached_inputs(self.binding, max_bytes=BIG)
        self.pins = pins_of(self.stack["proj"], self.binding, self.rec)
        self.backend, self.windowed = backend, windowed

    def old_vertical(self):
        """A REAL vertical of the SAME boundary BEFORE the law (its operators are the ordinary scalar-g ones)."""
        return fp.FloatingPondVertical.from_boundary(self.b.root, available_bytes_fn=AVAIL)


def export(v, rec, pins, **kw):
    kw.setdefault("memory_budget_bytes", BIG)
    kw.setdefault("available_bytes_fn", AVAIL)
    return fhv.export_held_context_vegetation(v, rec, pins, **kw)


def expected_of(record):
    i = record.meta()["identities"]
    return fh.ExpectedHeldIdentities(i["config_sha256"], i["root_column_sha256"], i["matrix_op_sha256"], i["pref_sha256"])


def reconstruct(record, pins, expected=None, **kw):
    kw.setdefault("memory_budget_bytes", BIG)
    kw.setdefault("available_bytes_fn", AVAIL)
    return fh.reconstruct_vegetation_held_context(record, expected or expected_of(record), pins, **kw)


def serialise(record):
    import copy
    return copy.deepcopy(record.meta()), {k: np.array(v) for k, v in record.arrays().items()}


__all__ = ["AVAIL", "BACKENDS", "BIG", "World", "cs", "expected_of", "export", "fh", "fhv", "fp", "pins_of", "reconstruct", "serialise", "vb", "vl", "wh", "wvb"]
