"""H0-CUDA host tests (CPU/host: NO GPU, NO CuPy). They pin the strict /2 record, the versioned dispatch, the gate-before-reader order, the model arithmetic and that /1 is unchanged. The device behaviour is defined in
`test_gpu_definitions.py` for Root."""
# ruff: noqa: RUF059
from __future__ import annotations

import copy
import sys

import hbuild as hb
import numpy as np
import pytest
from hbuild import AVAIL, make, pref_spec

from maple_syrup import floating_held_context as fh
from maple_syrup import floating_held_context_cuda as fhc
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import floating_transpiration_cuda  # noqa: F401


def v2_parts(**kw):
    """A /2 record built from a genuine CPU record's parts (host layout identical; the identities are arbitrary hex: parse tests only)."""
    vert, owner, bed = make(**kw)
    rec = fh.export_held_context(vert, kw.get("spec"), available_bytes_fn=AVAIL)
    meta, arrays = hb.deserialise(*hb.serialise(rec))
    m2 = copy.deepcopy(meta)
    m2["schema"], m2["backend"], m2["kernel"] = fh.SCHEMA2, "cuda", fhc.kernel_descriptor()
    return vert, meta, m2, arrays


def test_the_cuda_module_needs_no_cupy_and_the_kernel_descriptor_is_stable():
    d1, d2 = fhc.kernel_descriptor(), fhc.kernel_descriptor()
    assert d1 == d2 and d1["common_abi_sha256"] == "0d454f1ad0fc4c6d23252687c311802788e3ee2a411f714969a2a08e9c9f72e2"
    assert all(len(v) == 64 for v in d1.values()) and "cupy" not in sys.modules


@pytest.mark.parametrize("L", [1, 3])
@pytest.mark.parametrize("mode", ["explicit", "texture_ks"])
def test_a_valid_v2_record_parses_on_the_cpu_and_the_v1_schema_is_unchanged(L, mode):
    vert, m1, m2, arrays = v2_parts(L=L, mode=mode, overrides={}, spec=pref_spec(L, 3, species=2))
    rec2 = fh.record_from_parts(m2, arrays, available_bytes_fn=AVAIL)
    assert rec2.meta()["schema"] == fh.SCHEMA2 and rec2.meta()["backend"] == "cuda"
    rec1 = fh.record_from_parts(m1, arrays, available_bytes_fn=AVAIL)  # /1 still parses exactly as before
    assert rec1.meta()["schema"] == fh.SCHEMA and fh.SCHEMA == "maple_syrup.floating_held_context/1"
    for bad_meta in ({**m1, "kernel": m2["kernel"]}, {k: v for k, v in m2.items() if k != "kernel"}, {**m2, "backend": "numpy"}, {**m1, "backend": "cuda"}):
        with pytest.raises(fh.HeldContextError):
            fh.record_from_parts(bad_meta, arrays, available_bytes_fn=AVAIL)  # no upgrade, no rebind, no cross-version reading


@pytest.mark.parametrize("what", ["kernel_extra", "kernel_missing", "kernel_not_hex", "kernel_huge", "schema", "unknown_top"])
def test_v2_metadata_is_strict_and_bounded(what, monkeypatch):
    vert, m1, m2, arrays = v2_parts()
    m = copy.deepcopy(m2)
    if what == "kernel_extra":
        m["kernel"]["x"] = "0" * 64
    elif what == "kernel_missing":
        del m["kernel"]["fpc_source_sha256"]
    elif what == "kernel_not_hex":
        m["kernel"]["fmc_source_sha256"] = "G" * 64
    elif what == "kernel_huge":
        m["kernel"]["fmc_source_sha256"] = "a" * 2_000_000
    elif what == "schema":
        m["schema"] = "maple_syrup.floating_held_context/3"
    else:
        m["unknown_payload"] = "x" * 2_000_000
    reads = []
    with pytest.raises(fh.HeldContextError):
        fh.record_from_parts(m, arrays, available_bytes_fn=lambda: reads.append(1) or hb.BIG)
    assert reads == []


def test_a_cpu_context_never_reconstructs_a_v2_record_and_v1_refuses_host_pool_arguments():
    vert, m1, m2, arrays = v2_parts()
    r2 = fh.record_from_parts(m2, arrays, available_bytes_fn=AVAIL)
    ident = m2["identities"]
    exp = fh.ExpectedHeldIdentities(ident["config_sha256"], ident["root_column_sha256"], ident["matrix_op_sha256"], ident["pref_sha256"])
    reads = []
    with pytest.raises(fh.HeldContextError, match="CUDA device gate refused"):  # no CuPy/device here: the gate is the FIRST operation, before any reader
        fh.reconstruct_held_context(r2, exp, available_bytes_fn=lambda: reads.append(1) or hb.BIG, host_available_bytes_fn=lambda: reads.append(2) or hb.BIG)
    assert reads == []
    r1 = fh.record_from_parts(m1, arrays, available_bytes_fn=AVAIL)
    e1 = hb.expected_of(vert)
    with pytest.raises(fh.HeldContextError, match="CUDA .*record"):
        fh.reconstruct_held_context(r1, e1, available_bytes_fn=AVAIL, host_reserve_bytes=1)
    assert fh.reconstruct_held_context(r1, e1, available_bytes_fn=AVAIL).backend == "numpy"  # CPU path and context defaults unchanged
    with pytest.raises(fh.HeldContextError, match="identities differ"):
        fh.reconstruct_held_context(r2, fh.ExpectedHeldIdentities("0" * 64, *[ident[k] for k in ("root_column_sha256", "matrix_op_sha256", "pref_sha256")]), available_bytes_fn=AVAIL)


def test_export_refuses_a_cpu_vertical_and_a_misused_pool_before_any_gate_or_read():
    vert, *_ = make()
    with pytest.raises(fh.HeldContextError, match="genuine CUDA"):
        fhc.export_held_context_cuda(vert, None, available_bytes_fn=AVAIL)
    for bad in (-1, True, 1.5, "0", None):
        with pytest.raises(fh.HeldContextError):
            fhc.export_held_context_cuda(vert, None, reserve_bytes=bad) if bad is not None else (_ for _ in ()).throw(fh.HeldContextError("n"))
    with pytest.raises(fh.HeldContextError):
        fhc.export_held_context_cuda(vert, None, host_available_bytes_fn=5)


@pytest.mark.parametrize("L,N,S,kind", [(1, 3, 0, "scalar"), (3, 50003, 2, "cell"), (65, 4096, 0, "scalar")])
def test_the_model_is_exactly_the_existing_component_terms_plus_held_uploads(L, N, S, kind):
    from maple_syrup import floating_matrix_infiltration_cuda as fmc
    from maple_syrup import floating_preferential_infiltration_cuda as fpc

    vert, m1, m2, arrays = v2_parts(L=min(L, 3), mode="texture_ks", spec=pref_spec(min(L, 3), 3, species=S)) if L <= 3 and N == 3 else (None, None, None, None)
    meta = copy.deepcopy(m2) if m2 else None
    if meta is None:  # dimension-scaled metadata (arithmetic only; no arrays exist)
        v, m1b, meta, a = v2_parts(L=3, mode="texture_ks", spec=pref_spec(3, 3, species=S))
        meta["n_layers"], meta["n_cells"] = L, N
        meta["layouts"]["all"] = {"theta_residual": "scalar", "theta_field_capacity": "cell", "theta_saturation": "cell", "interface": "shared" if L > 1 else "scalar"}
        meta["layouts"]["theta_field_capacity"] = meta["layouts"]["theta_saturation"] = "cell"
    dev, host, exp = fhc.plan(meta)
    L, N = meta["n_layers"], meta["n_cells"]
    assert dev >= fcu.retained_terms(L, N, dict(meta["layouts"]["all"]))["total"] + fmc.operation_terms("prepare", L, N)["outputs"]  # retained + matrix terms are included
    if meta["pref"] is not None:
        assert dev >= fpc.operation_terms("prepare", L, N)["outputs"] + 8 * N * (meta["pref"]["n_species"])  # + the preferential terms and the uploaded contributions
    cell_bytes = 8 * L * N if meta["layouts"]["theta_field_capacity"] == "cell" else 0
    assert dev >= 2 * cell_bytes  # the uploaded FC/SAT are held ALIVE while the public preparation copies them (a record+retained-only model would miss this)
    assert host >= fcu.operation_terms("prepare", L, N)["host"] and exp >= fh._analytic(meta)[1]


def test_adopt_parts_is_strict_and_seals_in_place():
    vert, m1, m2, arrays = v2_parts()
    owned = {k: np.array(v, copy=True) for k, v in arrays.items()}
    rec = fh.adopt_parts(m2, owned)
    assert all(not a.flags.writeable for a in owned.values()) and rec.meta()["schema"] == fh.SCHEMA2
    for bad in (lambda d: d.pop("depth0_m"), lambda d: d.__setitem__("depth0_m", d["depth0_m"].astype(np.float32)), lambda d: d.__setitem__("junk", np.zeros(1))):
        o = {k: np.array(v, copy=True) for k, v in arrays.items()}
        bad(o)
        with pytest.raises(fh.HeldContextError):
            fh.adopt_parts(m2, o)


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_namespace_classification_uses_the_actual_backend_not_a_device_attribute():  # NumPy 2.x arrays carry .device == 'cpu': the old hasattr duck type misclassified them
    a = np.zeros(3)
    assert fhc._is_device(a) is False
    for bad in ([1.0], np.matrix(a), 3.0, None, "x"):
        with pytest.raises(fh.HeldContextError, match="exactly a numpy.ndarray"):
            fhc._is_device(bad)
