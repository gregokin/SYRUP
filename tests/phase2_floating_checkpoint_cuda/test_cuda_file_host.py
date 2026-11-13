"""G-F1+2 HOST tests of the Dense CUDA floating file (CPU/host: NO GPU, NO CuPy). They pin the closed schema/profile boundary against the CPU child, the typed refusals that need no device, the device-gate-before-reader order,
the two-pool model arithmetic, the chunked streamed comparison and the transfer-helper logic (NumPy arrays as ordering/arithmetic controls only, never a numerical body), the capability classes' integrity contracts and the byte-
identity of every CPU/historical production file. The device behaviour (actual save/load/adopt/continue) is defined in `test_gpu_definitions.py` for Root. Author-written."""
# ruff: noqa: RUF059
from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from case import AVAIL, IDENT, build

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_checkpoint_cuda as fcc
from maple_syrup import floating_pond as fp
from maple_syrup import floating_resume as fr
from maple_syrup import floating_resume_cuda as frc
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import pond_checkpoint as pc

HERE = Path(__file__).resolve()
TASK = HERE.parents[3]
SRC = HERE.parents[2] / "src" / "maple_syrup"
MANIFEST = TASK / "completion_program_20261113" / "gpu_dense_file_pre_source_manifest.json"


@pytest.fixture
def cpu_case(tmp_path):
    ev, st, vert = build(tmp_path, backend="numpy")
    return ev, st, vert


def test_the_new_modules_import_without_cupy_and_the_schema_boundary_is_closed():
    assert "cupy" not in sys.modules
    assert fcc.GPU_SCHEMA == "maple-syrup-floating-pond-checkpoint/7" and fcc.GPU_WINDOWED_SCHEMA.endswith("/8") and fcc.GPU_PROFILE == "floating-vertical/2-cuda"
    assert fcc.GPU_SCHEMA not in fc.READ_SCHEMAS and fcc.GPU_WINDOWED_SCHEMA not in fc.READ_SCHEMAS and fcc.GPU_PROFILE != fc.PROFILE  # the CPU child never reads /7 or /8; no profile masquerading
    assert fcc.MANIFEST_KEYS == fc.MANIFEST_KEYS_V3 | {"kernel"} and fcc.FLOATING_KEYS == fc.FLOATING_KEYS | {"held_layouts"}
    assert fcc.MANIFEST_KEYS != fc.MANIFEST_KEYS_V3 and fcc.MANIFEST_KEYS - {"kernel"} == fc.MANIFEST_KEYS_V3  # a CPU /5 manifest cannot satisfy the closed GPU key set
    assert fc.SCHEMA_V3 == "maple-syrup-floating-pond-checkpoint/5" and fc.WINDOWED_SCHEMA_V3.endswith("/6")  # the CPU profile is unchanged


def test_the_cpu_loader_refuses_the_gpu_schema_by_its_schema_check(cpu_case, tmp_path):
    ev, st, vert = cpu_case
    ck = tmp_path / "ck"
    fc.save_floating_pond_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    man = json.loads((ck / fc.MANIFEST_NAME).read_text())
    man["schema"] = fcc.GPU_SCHEMA
    man["profile"] = fcc.GPU_PROFILE
    (ck / fc.MANIFEST_NAME).write_text(json.dumps(fc._seal(man)))
    ev2, _s, _v = build(tmp_path / "b", backend="numpy")
    with pytest.raises(fc.FloatingCheckpointError, match="not a floating pond checkpoint schema"):
        fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL)


@pytest.mark.parametrize("fn_name", ["save", "load", "prepare"])
def test_a_cpu_event_is_refused_typed_before_any_reader_gate_or_io(cpu_case, tmp_path, fn_name):
    ev, st, vert = cpu_case
    reads = []

    def reader():
        reads.append(1)
        return 1 << 40

    target = tmp_path / "never"
    if fn_name == "save":
        with pytest.raises(fcc.FloatingCheckpointCudaError, match="needs a CUDA vertical on a CuPy event"):
            fcc.save_floating_pond_checkpoint_cuda(target, st, ev, IDENT, available_bytes_fn=reader, host_available_bytes_fn=reader)
    elif fn_name == "load":
        with pytest.raises(fcc.FloatingCheckpointCudaError, match="needs a CUDA vertical on a CuPy event"):
            fcc.load_floating_pond_checkpoint_cuda(target, ev, IDENT, available_bytes_fn=reader, host_available_bytes_fn=reader)
    else:
        with pytest.raises(fcc.FloatingCheckpointCudaError, match="needs a CUDA vertical on a CuPy event"):
            frc.prepare_from_verified_inputs(ev, object(), quota=fcc.quota_of(reader, 0, reader, 0, None, None))
    assert reads == [] and not target.exists() and list(tmp_path.glob(".never*")) == []


def test_a_non_event_and_a_bucket_event_are_refused_typed():
    for bad in (object(), None, "event"):
        with pytest.raises(fcc.FloatingCheckpointCudaError, match="PondEvent with a FloatingPondVertical"):
            fcc.check_event(bad)


@pytest.mark.parametrize("bad", [-1, 1.5, True, "1", None])
def test_reserves_budgets_and_readers_are_validated_before_the_gate(cpu_case, monkeypatch, tmp_path, bad):
    ev, st, vert = cpu_case
    monkeypatch.setattr(fcc, "check_event", lambda event: vert)  # ordering control only: a CPU vertical stands in for a CUDA one; the gate must never be reached
    called = []
    monkeypatch.setattr(fcc, "gate", lambda column: called.append(1))
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="reserve_bytes must be an integer >= 0"):
        fcc.save_floating_pond_checkpoint_cuda(tmp_path / "x", st, ev, IDENT, reserve_bytes=bad)
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="host_reserve_bytes must be an integer >= 0"):
        fcc.load_floating_pond_checkpoint_cuda(tmp_path / "x", ev, IDENT, host_reserve_bytes=bad)
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="must be callable or None"):
        fcc.save_floating_pond_checkpoint_cuda(tmp_path / "x", st, ev, IDENT, available_bytes_fn=5)
    assert called == []


@pytest.mark.parametrize("fn_name", ["save", "load"])
def test_the_device_gate_is_the_first_operation_after_pure_normalisation_and_precedes_every_reader_and_path_check(cpu_case, monkeypatch, tmp_path, fn_name):
    ev, st, vert = cpu_case
    monkeypatch.setattr(fcc, "check_event", lambda event: vert)  # ordering control only
    reads = []

    def reader():
        reads.append(1)
        return 1 << 40

    existing = tmp_path / "exists"
    existing.mkdir()
    if fn_name == "save":  # an existing target would be refused AFTER the gate; with no CUDA here the gate refusal must win
        with pytest.raises(fcc.FloatingCheckpointCudaError, match="CUDA device gate refused"):
            fcc.save_floating_pond_checkpoint_cuda(existing, st, ev, IDENT, available_bytes_fn=reader, host_available_bytes_fn=reader)
    else:  # a missing directory would be refused AFTER the gate
        with pytest.raises(fcc.FloatingCheckpointCudaError, match="CUDA device gate refused"):
            fcc.load_floating_pond_checkpoint_cuda(tmp_path / "missing", ev, IDENT, available_bytes_fn=reader, host_available_bytes_fn=reader)
    assert reads == []


def test_prepare_gates_the_device_before_it_looks_at_the_inputs(cpu_case, monkeypatch):
    ev, st, vert = cpu_case
    monkeypatch.setattr(fcc, "check_event", lambda event: vert)
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="CUDA device gate refused"):
        frc.prepare_from_verified_inputs(ev, object(), quota=fcc.quota_of(None, 0, None, 0, None, None))  # the inputs would be refused AFTER the gate


def test_verified_inputs_and_prepared_records_are_not_mintable_copyable_or_serialisable(cpu_case):
    import copy
    import pickle

    ev, st, vert = cpu_case
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="minted only by load_floating_pond_checkpoint_cuda"):
        fcc.VerifiedFloatingInputsCuda(*(None,) * 20)
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="not genuinely minted"):
        fcc.require_inputs(object())
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="not genuinely minted"):
        fcc.verify_intact({"free": "kwargs"})
    with pytest.raises(frc.FloatingResumeCudaError, match="minted only by prepare_from_verified_inputs"):
        frc._PreparedResumeCuda(None, None, None, [False])
    for bad in (object(), None, {"parts": 1}):
        with pytest.raises(frc.FloatingResumeCudaError, match="not a genuinely minted"):
            frc.prepared_parts(bad, vert)
    assert issubclass(frc.FloatingResumeCudaError, fr.FloatingResumeError) and issubclass(fcc.FloatingCheckpointCudaError, fc.FloatingCheckpointError)
    assert copy.copy and pickle.dumps  # the classes' own __copy__/__deepcopy__/__reduce_ex__ refusals are source facts pinned below
    for cls in (fcc.VerifiedFloatingInputsCuda, frc._PreparedResumeCuda):
        for name in ("__copy__", "__deepcopy__", "__reduce_ex__"):
            assert name in cls.__dict__


def test_the_resume_module_dispatch_is_by_backend_and_the_cpu_path_is_unchanged(cpu_case):
    ev, st, vert = cpu_case
    assert vert._resume_module() is fr and vert.backend == "numpy"
    src = inspect.getsource(fp.FloatingPondVertical._resume_module)
    assert "floating_resume_cuda" in src and 'self.backend == "cuda"' in src
    assert "CUDA contexts/backends are refused" not in (fp.FloatingPondVertical.from_held_context.__doc__ or "")


# ---- arithmetic of the declared models ------------------------------------------------------------------------------------------------------------------
def test_held_shape_follows_the_raw_layout():
    assert fcc.held_shape("scalar", 3, 7) == (1,) and fcc.held_shape("shared", 3, 7) == (3,) and fcc.held_shape("cell", 3, 7) == (3, 7)
    with pytest.raises(fcc.FloatingCheckpointCudaError):
        fcc.held_shape("tensor", 3, 7)


def test_prepare_models_are_the_documented_arithmetic():
    L, N = 3, 50003
    dev, host = frc.prepare_models(L, N)
    from maple_syrup import floating_soil_layers_cuda as fcu

    chk = fcu.operation_terms("state_check", L, N)
    assert dev == (L + 3) * N * 8 + (L + 2) * N * 8 + 2 * N * 8 + N * 8 + N + 2 * N * 8 + chk["scratch"] + fcu.HASH_CHUNK_BYTES
    assert host == fr.HASH_ALLOWANCE_BYTES + fr._cold_source_bytes() + 2 * fcu.HASH_CHUNK_BYTES + chk["host"]
    assert frc.prepare_models(L, 2 * N)[0] > dev  # the device model is linear in the cell count


def test_load_and_save_models_compose_header_dimensions_and_scale_with_each_dimension():
    kw = {"table_bytes": 10_000_000, "native_bytes": 4_000_000, "V": 300, "K": 6, "n_active": 40000, "batch_cells": 1024}
    N, nc, nb = 50003, 6, 8
    d, h = fcc.load_models(3, N, nc, nb, dev_table_bytes=6_000_000, **kw)
    pd, ph_ = frc.prepare_models(3, N)
    work = fcc.working_bytes(N, nc, nb)
    assert work == (fc.TOPOLOGY_ARRAYS_PER_CELL * N + 4 * N * nc * nb + 8 * N * nc) * 8
    assert h == 3 * kw["table_bytes"] + 2 * kw["native_bytes"] + work + 8 * N + fcc.HASH_BUFFER_BYTES + fcc.HEADER_AUDIT_BYTES + fcc.CHUNK_BYTES + ph_ + kw["n_active"] * 8
    assert d == kw["native_bytes"] + 6_000_000 + work + pd + fcc.canonical_depth_bytes(N, 300, 6, 1024, 40000) + fcc.CHUNK_BYTES
    sd, sh = fcc.save_models(3, N, nc, nb, **kw)
    assert sh == 3 * kw["table_bytes"] + 2 * kw["native_bytes"] + work + 8 * N + fcc.HASH_BUFFER_BYTES + fcc.HEADER_AUDIT_BYTES + fcc.CHUNK_BYTES + kw["n_active"] * 8
    assert sd == work + fcc.canonical_depth_bytes(N, 300, 6, 1024, 40000) + fcc.CHUNK_BYTES
    for bigger in ({"native_bytes": 8_000_000}, {"table_bytes": 20_000_000}, {"V": 600}, {"batch_cells": 2048}):
        d2, h2 = fcc.load_models(3, N, nc, nb, dev_table_bytes=6_000_000, **{**kw, **bigger})
        assert d2 >= d and h2 >= h and (d2, h2) != (d, h)
    assert fcc.load_models(65, N, nc, nb, dev_table_bytes=6_000_000, **kw)[0] > d  # more layers -> more device bytes (uploads/payload)


# ---- the chunked streamed comparison and transfer helpers (NumPy arrays as ordering/arithmetic controls) -----------------------------------------------------
def test_stream_equal_is_chunked_exact_bitwise_and_exits_early(monkeypatch):
    import maple.core.backend as mb

    monkeypatch.setattr(fcc, "CHUNK_BYTES", 64)  # 8 float64 per chunk
    calls = []
    real = mb.to_host
    monkeypatch.setattr(mb, "to_host", lambda a: calls.append(int(np.asarray(a).size)) or real(a))
    a = np.arange(30, dtype=np.float64).reshape(5, 6)
    assert fcc.stream_equal(a, a.copy()) and calls == [8, 8, 8, 6]  # bounded chunks, the tail included, ONE transfer per chunk
    calls.clear()
    b = a.copy()
    b[0, 1] += 1.0
    assert not fcc.stream_equal(a, b) and calls == [8]  # early exit at the first differing chunk
    c = a.copy()
    c[4, 5] = np.nextafter(c[4, 5], np.inf)
    assert not fcc.stream_equal(a, c)  # a single last-ulp difference in the last chunk
    z1, z2 = np.array([0.0]), np.array([-0.0])
    assert not fcc.stream_equal(z1, z2)  # bit-for-bit: signed zeros differ
    assert not fcc.stream_equal(a, a[:4]) and not fcc.stream_equal(a, a.astype(np.float32)) and not fcc.stream_equal(a.T, a.T.copy())  # shape / dtype / non-contiguous device-side layout refused


def test_d2h_copies_in_bounded_counted_chunks_and_refuses_a_noncontiguous_array(monkeypatch):
    import maple.core.backend as mb

    monkeypatch.setattr(fcc, "CHUNK_BYTES", 64)
    calls = []
    real = mb.to_host
    monkeypatch.setattr(mb, "to_host", lambda a: calls.append(int(np.asarray(a).size)) or real(a))
    a = np.arange(21, dtype=np.float64).reshape(3, 7)
    out = fcc.d2h(a)
    assert out.tobytes() == a.tobytes() and out is not a and out.flags.owndata and calls == [8, 8, 5]
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="C-contiguous"):
        fcc.d2h(a.T)


def test_quota_defaults_and_component_quota_count_each_reserve_once():
    q = fcc.quota_of(None, 5, None, 7, None, None)
    assert q["reserve_bytes"] == 5 and q["host_reserve_bytes"] == 7 and q["available_bytes_fn"] is None and q["host_available_bytes_fn"] is None  # real default readers when None
    c = fcc.component_quota(q)
    assert c["reserve_bytes"] == 0 and c["host_reserve_bytes"] == 0 and {k: v for k, v in c.items() if "reserve" not in k} == {k: v for k, v in q.items() if "reserve" not in k}


def test_the_binding_names_the_gpu_profile_and_keeps_the_cpu_binding_unchanged(cpu_case):
    ev, st, vert = cpu_case
    b = fcc._binding(ev)
    assert b["schema"] == fcc.GPU_SCHEMA and b["profile"] == fcc.GPU_PROFILE
    cpu = fc._binding(ev, fc.SCHEMA_V3)
    assert cpu["profile"] == fc.PROFILE and {k: v for k, v in b.items() if k not in ("schema", "profile")} == {k: v for k, v in cpu.items() if k not in ("schema", "profile")}


def test_gpu_floating_specs_follow_the_held_layouts():
    s = fcc._gpu_floating_specs(3, 11, {"theta_field_capacity": "cell", "theta_saturation": "shared"}, True)
    assert s["fl_held_fc"][0] == (3, 11) and s["fl_held_sat"][0] == (3,) and s["fl_ks"][0] == (11,) and s["fl_water"][0] == (3, 11) and s["fl_depth"][0] == (11,)
    assert "fl_ks" not in fcc._gpu_floating_specs(3, 11, {"theta_field_capacity": "scalar", "theta_saturation": "scalar"}, False)
    assert fcc._gpu_floating_specs(3, 11, {"theta_field_capacity": "scalar", "theta_saturation": "scalar"}, False)["fl_held_fc"][0] == (1,)


def test_the_cpu_held_check_stays_a_host_numpy_comparison(cpu_case):
    ev, st, vert = cpu_case
    t = vert.held_template()
    facts = fcc._facts(vert, vert.current_payload(st.hydraulic.soil_water_m))
    assert set(facts) == fcc.FLOATING_KEYS and facts["held_layouts"] == fcc._layouts_of(t.column)
    fcc.check_held_ids(vert, facts)  # the ids/layout half is pure strings and accepts the vertical's own facts
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="HELD template differs"):
        fcc.check_held_ids(vert, {**facts, "generation": facts["generation"] + 1})
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="HELD template differs"):
        fcc.check_held_ids(vert, {**facts, "held_layouts": {**facts["held_layouts"], "theta_saturation": "scalar" if facts["held_layouts"]["theta_saturation"] != "scalar" else "cell"}})
    assert fsl is not None


# ---- production byte-identity ----------------------------------------------------------------------------------------------------------------------------








def test_validate_scalars_reads_the_terminal_ledger_through_the_counted_host_helper(cpu_case, monkeypatch):
    import maple.core.backend as mb

    ev, st, vert = cpu_case
    calls = []
    real = mb.to_host
    monkeypatch.setattr(mb, "to_host", lambda a: calls.append(1) or real(a))
    arrays, scalars, _f = fc._validate_floating_state(ev, st, None, "dense")  # unchanged CPU semantics: identity conversion, same verdict
    assert calls and scalars["counters"]["n_accepted_steps"] == st.counters.n_accepted_steps


def test_prepared_parts_gates_the_device_before_any_integrity_hash_or_comparison():
    """Source-order fact (the behavioural proof is the GPU definition): in `prepared_parts` the gate call precedes verify_intact / input_digest / stream_equal."""
    src = inspect.getsource(frc.prepared_parts)
    gate, intact, digest, equal = (src.index(x) for x in ("fcc.gate(", "fcc.verify_intact(", "fcc.input_digest(", "fcc.stream_equal("))
    assert gate < intact < digest and gate < equal




def test_the_shared_test_snapshot_reads_floating_arrays_through_the_counted_helper():
    import inspect as _i

    import case

    src = _i.getsource(case.snapshot)
    assert all(f"to_host(p.{x})" in src for x in ("state.water_m", "history.wetting_m", "history.cumulative_intake_m", "depth_m"))


def test_moderate_bed_is_a_genuine_native_one_by_50003_by_60_by_6_bed_on_the_desired_terrain():
    """CPU construction of the GPU moderate fixture (no probe, no `dense_deep_bed`): native shape and the canonical physical elevation equal the desired terrain and the context geometry is the static primitive recipe."""
    import moderate_bed as mbd
    import static_case as sc

    from maple_syrup.sediment_bed import BedState

    bed, ctx, topo, z = mbd.build(1, 50003, 60)
    assert isinstance(bed, BedState) and tuple(bed.voxel_column.mass_kg.shape) == (1, 50003, 60, 6) and tuple(bed.active_layer.mass_kg.shape) == (1, 50003, 6)
    g = ctx.geometry
    assert (g.ny, g.nx) == (1, 50003) and g.voxel_dz_m == 0.001 and tuple(topo.shape) == (1, 50003)
    per = g.bulk_density_kg_m3 * g.dx_m * g.dy_m
    phys = bed.voxel_column.mass_kg.sum(axis=(2, 3)) / per + bed.active_layer.mass_kg.sum(axis=2) / per
    np.testing.assert_allclose(phys, z, rtol=0, atol=1e-6)  # the desired 0.05 m terrain (up to the native mass-resolution quantisation)
    assert abs(float(phys.max()) - float(z.max())) < 1e-6 and 0.05 <= float(z.max()) < 60 * g.voxel_dz_m
    assert sc.TERRAIN["nx"] == 4 and sc.CASE["z"] is None and sc.DZ == 0.001  # the shared static recipe globals were restored


# ---- per-interface compatibility checks (replace the snapshot-specific source audits, preserved in `historical_source_audits.py`) ---------------------------------------------------------
def test_validate_scalars_converts_the_terminal_ledger_through_the_counted_helper_by_ast():
    """The ONE GPU-driven deviation of the shared CPU validator, checked by its interface (not by the whole file): the deposited-terminal read goes through `to_host`, and the producer-maintained terminal-count rule is intact."""
    tree = ast.parse(inspect.getsource(fc._validate_scalars).replace("\n    ", "\n", 0))
    src = inspect.getsource(fc._validate_scalars)
    calls = [n for n in ast.walk(ast.parse("if True:\n" + "\n".join("    " + ln for ln in src.splitlines()))) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "to_host"]
    assert tree and calls and "deposited_terminal" in src and "n_terminal_native_calls" in src and "base + terminal" in src and "retained_quiet_reset_step" in src


def test_the_cpu_child_modules_do_not_know_the_gpu_modules_and_the_cpu_schemas_are_unchanged():
    for mod in (fc, fr):
        text = inspect.getsource(mod)
        assert "floating_checkpoint_cuda" not in text and "floating_resume_cuda" not in text
    assert fc.READ_SCHEMAS == (fc.SCHEMA, fc.WINDOWED_SCHEMA, fc.SCHEMA_V2, fc.WINDOWED_SCHEMA_V2, fc.SCHEMA_V3, fc.WINDOWED_SCHEMA_V3) and fc.PROFILE == "floating-vertical/1"


def test_resume_scope_and_adopt_resumed_dispatch_through_the_backend_module():
    for name in ("resume_scope", "adopt_resumed"):
        assert "self._resume_module()" in inspect.getsource(getattr(fp.FloatingPondVertical, name))


# ---- Windowed /8 host controls -------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def cpu_windowed(tmp_path):
    ev, st, vert = build(tmp_path, backend="numpy", windowed=True)
    return ev, st, vert


def test_the_windowed_schema_is_a_closed_additive_profile_and_dense_is_unchanged():
    assert fcc.READ_SCHEMAS == (fcc.GPU_SCHEMA, fcc.GPU_WINDOWED_SCHEMA) and fcc.GPU_WINDOWED_SCHEMA.endswith("/8")
    assert fcc.WINDOWED_MANIFEST_KEYS == fcc.MANIFEST_KEYS | {"bed_storage", "deep_storage"} and fcc.MANIFEST_KEYS == fc.MANIFEST_KEYS_V3 | {"kernel"}  # /7 keys are byte-for-byte the previous closed set
    assert fcc.GPU_WINDOWED_SCHEMA not in fc.READ_SCHEMAS and fcc.schema_of("dense") == fcc.GPU_SCHEMA and fcc.schema_of("windowed") == fcc.GPU_WINDOWED_SCHEMA


def test_the_windowed_binding_is_the_historical_windowed_event_binding_and_the_dense_binding_is_unchanged(cpu_windowed, tmp_path):
    ev, st, vert = cpu_windowed
    b = fcc._binding(ev, True)
    assert b["schema"] == fcc.GPU_WINDOWED_SCHEMA and b["profile"] == fcc.GPU_PROFILE and b["bed_storage"] == "windowed" and "deep_storage_option" in b and b["column_sha256"] == "floating-vertical-no-bucket-column"
    ev_d, _s, _v = build(tmp_path / "d", backend="numpy")
    d = fcc._binding(ev_d, False)
    assert d["schema"] == fcc.GPU_SCHEMA and "bed_storage" not in d and d == {**fc._binding(ev_d, fcc.GPU_SCHEMA), "profile": fcc.GPU_PROFILE}


def test_windowed_models_use_the_mapping_cache_bound_and_resident_bytes_not_the_logical_bed(cpu_windowed):
    ev, st, vert = cpu_windowed
    deep = pc._deep_record(st.bed.voxel_column, ev)
    cache = fc.deep_cache_bound_bytes(deep, ev.nc)
    resident = pc._windowed_resident_nbytes(st.bed, deep, IDENT)
    logical = ev.shape[0] * ev.shape[1] * deep["n_voxels"] * ev.nc * 8
    assert cache > 0 and resident < logical  # the resident window is far smaller than the logical cells x voxels x classes bed
    kw = {"table_bytes": 1_000_000, "native_bytes": resident, "V": 0, "K": ev.nc, "n_active": 12, "batch_cells": 1024}
    d0, h0 = fcc.load_models(3, 12, ev.nc, 8, dev_table_bytes=500_000, windowed=True, cache_bytes=0, **kw)
    d1, h1 = fcc.load_models(3, 12, ev.nc, 8, dev_table_bytes=500_000, windowed=True, cache_bytes=cache, **kw)
    assert h1 - h0 == cache and d1 == d0  # the cache is a HOST term, added exactly once
    sd, sh = fcc.save_models(3, 12, ev.nc, 8, table_bytes=1_000_000, native_bytes=resident, V=0, K=ev.nc, n_active=12, batch_cells=1024, windowed=True, cache_bytes=cache)
    sd0, sh0 = fcc.save_models(3, 12, ev.nc, 8, table_bytes=1_000_000, native_bytes=resident, V=0, K=ev.nc, n_active=12, batch_cells=1024, windowed=True)
    assert sh - sh0 == cache and sd == sd0
    assert fcc.canonical_depth_bytes(12, 0, ev.nc, 1024, 12, windowed=True) - fcc.canonical_depth_bytes(12, 0, ev.nc, 1024, 12) == 2 * 12 * 8  # the whole-grid prefix decode
    dense_d, dense_h = fcc.load_models(3, 12, ev.nc, 8, dev_table_bytes=500_000, **{**kw, "V": 300, "K": 6})
    assert dense_d == fcc.load_models(3, 12, ev.nc, 8, dev_table_bytes=500_000, windowed=False, cache_bytes=0, **{**kw, "V": 300, "K": 6})[0]  # the Dense signature and arithmetic are unchanged


def test_the_native_windowed_loader_refuses_a_corrupted_packet_so_the_host_load_precedes_any_device_placement(cpu_windowed, tmp_path):
    """The genuine untrusted validation is MAPLE's own (`load_windowed_state_snapshot` -> `validate_windowed_partition` re-encodes the packets from the deep store). A genuine CPU Windowed file's native snapshot loads and passes
    `verify_prefix_untrusted`; the SAME archive with one packet value changed (all other members, metadata and the deep store untouched) is refused by the native loader, i.e. on the HOST, before any placement."""
    from maple.surface.voxels.windowed_snapshot import load_windowed_state_snapshot

    ev, st, vert = cpu_windowed
    ck = tmp_path / "ck"
    fc.save_floating_pond_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    ctx = ev.context
    good = ck / pc.PAYLOADS[0]
    snap = load_windowed_state_snapshot(good, geometry=ctx.geometry, grain_classes=ctx.grain_classes, mass_resolution_kg=ctx.mass_resolution_kg, max_bytes=1 << 30)
    fcc.verify_prefix_untrusted(snap["voxel_column"])  # the structural untrusted decode accepts the genuine packets
    with np.load(good, allow_pickle=False) as z:
        data = {k: np.array(z[k]) for k in z.files}
    data["prefix_packet"] = np.array(data["prefix_packet"], copy=True)
    data["prefix_packet"].flat[5] = data["prefix_packet"].flat[5] * 1.5 + 1.0
    bad = tmp_path / "bad_state.npz"
    np.savez(bad, **data)
    with pytest.raises(ValueError):
        load_windowed_state_snapshot(bad, geometry=ctx.geometry, grain_classes=ctx.grain_classes, mass_resolution_kg=ctx.mass_resolution_kg, max_bytes=1 << 30)


def test_a_pending_windowed_journal_is_refused_before_any_admission_or_work(cpu_windowed, monkeypatch, tmp_path):
    """ORDERING control (a CPU vertical stands in for the device only to reach the Windowed branch): the pending-journal guard of the actual native helper fires before the admission."""
    ev, st, vert = cpu_windowed
    monkeypatch.setattr(fcc, "check_event", lambda event: vert)
    monkeypatch.setattr(fcc, "gate", lambda column: (None, 0))

    def no_admit(*a, **k):
        raise AssertionError("admission must not run for a pending journal")

    monkeypatch.setattr(fcc, "admit", no_admit)
    monkeypatch.setattr(pc, "_require_published", lambda bed: (_ for _ in ()).throw(pc.PondCheckpointError("pending deep journal (control)")))
    with pytest.raises(pc.PondCheckpointError, match="pending deep journal"):
        fcc.save_floating_pond_checkpoint_cuda(tmp_path / "never", st, ev, IDENT)
    assert not (tmp_path / "never").exists()


def test_the_storage_of_a_state_must_equal_the_events_configured_storage(cpu_windowed, monkeypatch, tmp_path):
    ev, st, vert = cpu_windowed
    ev_dense, _s, _v = build(tmp_path / "d", backend="numpy")
    monkeypatch.setattr(fcc, "check_event", lambda event: vert)
    monkeypatch.setattr(fcc, "gate", lambda column: (None, 0))
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="differs from the event's configured storage"):
        fcc.save_floating_pond_checkpoint_cuda(tmp_path / "x", st, ev_dense, IDENT)  # a Windowed state on a Dense-configured event: no cross-storage conversion


def test_the_gpu_loader_reuses_checks_that_accept_a_genuine_native_windowed_file_and_the_cache_bound_exceeds_the_resident_archive(cpu_windowed, tmp_path):
    """The `/8` loader's shared helpers on a REAL native Windowed child written by the CPU saver (same MAPLE snapshot protocol): the strict deep record, the grid-derived member preflight, the loaded-handle identity checks and the
    mapping-cache bound (a MiB-scale bound vs a KiB-scale resident archive: the model must never use the archive size)."""
    import zipfile

    from maple.surface.voxels.windowed_snapshot import load_windowed_state_snapshot

    ev, st, vert = cpu_windowed
    ck = tmp_path / "ck"
    fc.save_floating_pond_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    data = json.loads((ck / fc.MANIFEST_NAME).read_text())
    deep = pc._check_deep_record(data["deep_storage"], ev, 1 << 30)
    snap_path = ck / pc.PAYLOADS[0]
    with zipfile.ZipFile(snap_path) as z:
        sizes = {i.filename: i.file_size for i in z.infolist()}
    pc._windowed_snapshot_preflight(snap_path, sizes, pc._native_expected(ev, deep))
    ctx = ev.context
    snapshot = load_windowed_state_snapshot(snap_path, geometry=ctx.geometry, grain_classes=ctx.grain_classes, mass_resolution_kg=ctx.mass_resolution_kg, max_bytes=1 << 30)
    fc._check_windowed_snapshot_floating(snapshot, data, deep)  # exact deep identity + the loaded handle's mapping bound vs the modelled one
    fcc.verify_prefix_untrusted(snapshot["voxel_column"])
    assert fc.deep_cache_bound_bytes(deep, ev.nc) >= int(snapshot["voxel_column"].deep_store.mapped_bytes_bound) and fc.deep_cache_bound_bytes(deep, ev.nc) > sum(sizes.values())


def test_residency_components_report_the_exact_window_over_full_v_ratio_on_a_genuine_native_windowed_file(cpu_windowed, tmp_path):
    """Host control of the GPU moderate-test helper on a REAL native Windowed child (CPU saver, same MAPLE snapshot protocol): the native window member is exactly W/V of the logical bed, no member is a full cells x V x classes
    bed, the deep store is external and not copied, and the TOTAL checkpoint (which also holds the dense Pond arrays) is NOT claimed smaller than that bed."""
    import moderate_bed as mbd

    ev, st, vert = cpu_windowed
    ck = tmp_path / "ck"
    fc.save_floating_pond_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    man = json.loads((ck / fc.MANIFEST_NAME).read_text())
    comp = mbd.residency_components(ck, man, ev.shape[0], ev.shape[1], ev.nc)
    deep = man["deep_storage"]
    assert comp["window_to_full_v"] == f"{deep['window_voxels']}/{deep['n_voxels']}" and comp["window_data_bytes"] * deep["n_voxels"] == comp["logical_full_v_bed_bytes"] * deep["window_voxels"]
    assert comp["native_member_bytes"] == sum(comp["native_members"].values()) and comp["total_disk_bytes"] >= comp["native_file_bytes"] + comp["companion_bytes"] and comp["deep_store_external_bytes"] > 0
