"""Builders of the H0 suite: real native beds/owners (E3 fixtures), real FloatingPondVertical, real preferential operators. No fake classes."""
from __future__ import annotations

import builders as b
import numpy as np

from maple_syrup import floating_held_context as fh
from maple_syrup import floating_pond as fp
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import native_floating_soil as nfs

BIG = 10**12
AVAIL = lambda: BIG
BACKENDS = ["numpy"] + (["numba"] if __import__("importlib").util.find_spec("numba") is not None else [])
TEXTURE_KS = {"matrix_conductivity_m_s": None, "matrix_conductivity_mode": "texture_ks", "latent_saturated_conductivity_m_s": 3e-6}


def layers_of(L):
    return {1: (0.16,), 3: b.LAYERS, 65: tuple([0.16 / 65] * 65)}[L]


def pref_spec(L, N, kind="cell", species=0, rate="scalar"):
    a3 = np.full(N, 2.0e-3) if kind == "cell" else 2.0e-3
    mr = np.full(N, 1e-5) if rate == "cell" else (float("inf") if rate == "inf" else 1e-5)
    bg = np.ones(L) if L == 1 else np.array([0.0] + [1.0] * (L - 1))
    sw = tuple(np.eye(L)[i % L].copy() for i in range(species))
    sc = tuple(np.full(N, 0.5 + 0.25 * i) for i in range(species)) if species else None
    return fh.PreferentialSpec(a3, 0.5, mr, bg, sw, sc)


def make(*, L=3, backend="numpy", mode="explicit", overrides=None, bottom="open", spec=None, bed=None, saturation=0.5):
    bed = b.stratified() if bed is None else bed
    kw = TEXTURE_KS if mode == "texture_ks" else {}
    cfg = b.config(backend=backend, layers=layers_of(L), bottom=bottom, overrides=overrides, **kw)
    owner = nfs.initialize_native_floating_soil(bed, cfg, initial_relative_saturation=saturation, available_bytes_fn=AVAIL)
    pref = None if spec is None else fpi.prepare_floating_preferential(owner.column, **spec.kwargs(), available_bytes_fn=AVAIL)
    vert = fp.FloatingPondVertical(owner, available_bytes_fn=AVAIL, preferential=pref)
    return vert, owner, bed


def serialise(record):
    """Plain, process-independent parts: strict JSON text + a name -> (dtype, shape, bytes) map (what a file codec would store)."""
    meta = record.meta()
    arrays = {k: (str(v.dtype), tuple(v.shape), v.tobytes()) for k, v in record.arrays().items()}
    return meta, arrays


def deserialise(meta, arrays):
    out = {}
    for k, (dt, shape, raw) in arrays.items():
        out[k] = np.frombuffer(bytearray(raw), dtype=np.dtype(dt)).reshape(shape).copy(order="C")
    return meta, out


def expected_of(vertical):
    t = vertical.held_template()
    return fh.ExpectedHeldIdentities(vertical.config.sha256, t.column.sha256, t.matrix_op.sha256, None if t.preferential is None else t.preferential.sha256)
