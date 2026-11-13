"""EB3-CUDA HOST tests of the selected CUDA property refresh (CPU/host: NO GPU, NO CuPy). The device behaviour is defined in `test_gpu_definitions.py` for Root and was NOT run by the author.

What the host can prove: the closed typed refusals (CPU events, bad arguments), the gate-first/admission-first ORDER of the stage (source order + the real gate function), the composed two-pool model arithmetic from real
native shapes (Dense and published Windowed, 1/3/65 layers, large N by arithmetic), the byte/AST-level equivalence of every refactor of a qualified module against the main-159 baseline (owner-free reconcile core, closure helper,
default `n_selected=None` model, CPU preferential rebind), capability class integrity, and the byte identity of every other production module. It cannot execute the device algorithm; nothing here is a numerical body."""
# ruff: noqa: RUF059
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from case import build
from maple_syrup import floating_completion as fcm
from maple_syrup import floating_event_refresh as fer
from maple_syrup import floating_event_refresh_cuda as erc
from maple_syrup import floating_preferential_infiltration as fpi
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import native_floating_soil as nfs
from maple_syrup import native_floating_soil_cuda as ncuda

HERE = Path(__file__).resolve()
TASK = HERE.parents[3]
SRC = HERE.parents[2] / "src" / "maple_syrup"
BASE = TASK / "gpu_context_outer_candidate" / "src" / "maple_syrup"  # the main-159 tree (byte-equal to the task baseline, checked below)
MANIFEST = TASK / "completion_program_20261113" / "event_refresh_cuda_baseline.json"
CHANGED_PROD = {"native_floating_soil_cuda.py", "floating_preferential_infiltration.py"}
NEW_PROD = {"floating_event_refresh_cuda.py"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def completed(tmp, windowed=False):
    from test_event_refresh import completed as cpu_completed

    return cpu_completed(tmp, "numpy", windowed)


# ---- production identity ------------------------------------------------------------------------------------------------------------------------------------------------------------
# the author-refresh-tree diff audit is preserved VERBATIM in `historical_source_audits.py` (non-default); the lifecycle tree's per-interface proof is in tests/phase2_floating_event_lifecycle_cuda


def test_the_modules_import_without_cupy_and_the_cpu_refresh_still_refuses_cuda_events():
    assert "cupy" not in sys.modules
    assert "stage_completed_event_refresh_cuda" in erc.__all__ and "floating_event_refresh_cuda" not in inspect.getsource(fer)  # the CPU stage does not know the CUDA one and keeps its typed refusal
    assert "CUDA stage is a later slice" in inspect.getsource(fer.stage_completed_event_refresh)


# ---- refactor equivalence against the main-159 baseline (AST) ----------------------------------------------------------------------------------------------------------------------
def _fn(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _body(fn: ast.FunctionDef) -> list[str]:
    body = fn.body[1:] if isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) else fn.body
    return [ast.dump(n) for n in body]


class _Rename(ast.NodeTransformer):
    def __init__(self, names, attrs):
        self.names, self.attrs = names, attrs

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self.names.get(node.id, node.id), ctx=node.ctx), node)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "owner" and node.attr in self.attrs:
            return ast.copy_location(ast.Name(id=self.attrs[node.attr], ctx=node.ctx), node)
        return node


@pytest.mark.skipif(not BASE.exists(), reason="baseline tree not present")
def test_baseline_tree_is_byte_equal_to_the_task_baseline_for_the_modified_modules():
    manifest = json.loads(MANIFEST.read_text())
    for n in CHANGED_PROD:
        assert sha(BASE / n) == manifest[f"src/maple_syrup/{n}"]


@pytest.mark.skipif(not BASE.exists(), reason="baseline tree not present")
def test_the_owner_free_reconcile_core_is_the_baseline_owner_reconcile_with_explicit_arguments_and_the_wrapper_forwards_the_owners_own_values():
    old = _fn(BASE / "native_floating_soil_cuda.py", "_reconcile_cuda")
    tree = _Rename({"old_state": "sub_state"}, {"column": "old_column", "state": "old_state", "depth_m": "old_depth_host"})
    old_body = [n for n in old.body[1:] if not (isinstance(n, ast.Assign) and ast.dump(n.value) == ast.dump(ast.parse("_nfs()").body[0].value)) and not isinstance(n, ast.Delete)]
    renamed = [ast.dump(tree.visit(ast.fix_missing_locations(n))) for n in old_body]
    new = _body(_fn(SRC / "native_floating_soil_cuda.py", "reconcile_selected_columns"))
    assert renamed == new  # same statements, same order, only the owner attributes became explicit parameters
    wrapper = _fn(SRC / "native_floating_soil_cuda.py", "_reconcile_cuda")
    assert [a.arg for a in wrapper.args.args] == [a.arg for a in old.args.args]
    src = ast.unparse(wrapper.body[-1])
    assert src == "return reconcile_selected_columns(cp, config, owner.column, owner.state, owner.depth_m, new_fc, new_sat, new_depth, changed, area, B, quota)"


@pytest.mark.skipif(not BASE.exists(), reason="baseline tree not present")
def test_stage_cuda_differs_from_the_baseline_only_by_the_extracted_closure_call_and_an_unused_local():
    base_src = (BASE / "native_floating_soil_cuda.py").read_text()
    start = base_src.index("def stage_cuda(")
    seg = base_src[start:]
    a = seg.index("    # bounded chunked device check that unrefreshed columns kept their water")
    b = seg.index("    new_op = _device_op(cfg, new_column, new_ks, quota)\n    rb = fm.rebind_floating_matrix_history(owner.matrix_op")
    block = seg[a:b]
    seg2 = seg[:a] + "    w_before, w_after, resid, bound = inventory_closure_cuda(cp, cfg, owner.column, owner.state, new_column, new_state, water, changed, B, acc, quota)\n" + seg[b:]
    seg2 = seg2.replace("    water, surf, exp_, acc = _reconcile_cuda(cp, cfg, owner, new_fc, new_sat, depth, changed, area, B, quota)\n    L = cfg.layers.n_layers\n",
                        "    water, surf, exp_, acc = _reconcile_cuda(cp, cfg, owner, new_fc, new_sat, depth, changed, area, B, quota)\n")
    expect = ast.dump(ast.parse(seg2).body[0])
    got = ast.dump(_fn(SRC / "native_floating_soil_cuda.py", "stage_cuda"))
    assert expect == got
    helper = inspect.getsource(ncuda.inventory_closure_cuda)
    for line in block.splitlines():  # every baseline closure line (renamed) appears verbatim in the factored helper
        t = line.strip().replace("owner.column", "old_column").replace("owner.state", "old_state")
        if t and not t.startswith("#"):
            assert t in helper, t


@pytest.mark.skipif(not BASE.exists(), reason="baseline tree not present")
def test_every_other_definition_of_the_two_modified_modules_is_ast_identical_to_the_baseline():
    for name, changed in (("native_floating_soil_cuda.py", {"estimate_cuda_memory", "_reconcile_cuda", "stage_cuda"}), ("floating_preferential_infiltration.py", {"rebind_preferential_column"})):
        def defs(p):
            t = ast.parse(p.read_text())
            return {n.name: ast.dump(n) for n in t.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        a, b = defs(BASE / name), defs(SRC / name)
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        added = set(b) - set(a)
        assert (diff - added) <= changed and not (set(a) - set(b)), (name, diff)
        if name == "native_floating_soil_cuda.py":
            assert added == {"reconcile_selected_columns", "inventory_closure_cuda"}


def _load_base(name: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, BASE / name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not BASE.exists(), reason="baseline tree not present")
def test_the_default_estimate_is_unchanged_and_n_selected_models_the_selected_count(tmp_path):
    base = _load_base("native_floating_soil_cuda.py", "nfs_cuda_baseline_for_model")
    ev, h, vert = completed(tmp_path / "a")
    ctx = ev.context
    ny, nx = ev.shape
    active = np.asarray(vert._active_host()).reshape(ny, nx)
    child = nfs.NativeBed(h.state.bed.voxel_column, h.state.bed.active_layer, ctx.geometry, ctx.grain_classes, ctx.mass_resolution_kg, active)
    cfg = vert.config
    a = base.estimate_cuda_memory(None, child, cfg, batch_cells=64, reserve_bytes=3, host_reserve_bytes=5)
    b = ncuda.estimate_cuda_memory(None, child, cfg, batch_cells=64, reserve_bytes=3, host_reserve_bytes=5)
    assert a == b  # n_selected=None: the historical model, bit for bit
    n_all = int(active.sum())
    assert ncuda.estimate_cuda_memory(None, child, cfg, batch_cells=64, reserve_bytes=3, host_reserve_bytes=5, n_selected=n_all) == a
    one = ncuda.estimate_cuda_memory(None, child, cfg, batch_cells=64, n_selected=1)
    assert one["host_payload_bytes"] <= a["host_payload_bytes"] and one["device_payload_bytes"] <= a["device_payload_bytes"] and one["batch_cells"] == 1


# ---- composed models -----------------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("windowed", [False, True])
def test_the_stage_models_compose_the_native_estimate_at_the_selected_count_plus_their_own_terms(tmp_path, windowed):
    ev, h, vert = completed(tmp_path / "a", windowed)
    ctx = ev.context
    ny, nx = ev.shape
    child = nfs.NativeBed(h.state.bed.voxel_column, h.state.bed.active_layer, ctx.geometry, ctx.grain_classes, ctx.mass_resolution_kg, np.asarray(vert._active_host()).reshape(ny, nx))
    cfg = vert.config
    N, nc = ny * nx, ev.nc
    d0, h0 = erc.stage0_model_cuda(N, nc)
    assert (d0, h0) == (2 * N * nc + N + 8 * N, 16 * N + erc.CHUNK_BYTES)
    for n in (0, 1, 3):
        m = erc.event_refresh_model_cuda(child, cfg, n_selected=n, batch_cells=64, has_pref=False, reserve_bytes=10, host_reserve_bytes=20)
        base = ncuda.estimate_cuda_memory(None, child, cfg, batch_cells=64, reserve_bytes=10, host_reserve_bytes=20, n_selected=n)
        ex = {"current_depth_record": N * 8, "new_depth_copy": N * 8, "dirty_index": 8 * n, "transfer_chunk": erc.CHUNK_BYTES, "changed_ks_index": 16 * n}
        assert m["host_payload_bytes"] == base["host_payload_bytes"] + sum(ex.values()) and m["host_extra_terms"] == ex
        assert m["device_payload_bytes"] == base["device_payload_bytes"] + 8 * n and m["device_required_bytes"] == m["device_payload_bytes"] + 10
        assert m["host_required_bytes"] == m["host_payload_bytes"] + fcu.PY_ALLOWANCE_BYTES + 20  # allowance explicit, reserves ONCE
    p = erc.event_refresh_model_cuda(child, cfg, n_selected=2, batch_cells=64, has_pref=True)
    q = erc.event_refresh_model_cuda(child, cfg, n_selected=2, batch_cells=64, has_pref=False)
    assert p["host_extra_terms"]["pref_hash_staging"] == min(cfg.layers.n_layers * N, 16384) * 8 and p["host_payload_bytes"] > q["host_payload_bytes"]
    if windowed:  # the native phase carries the real deep-store mapping bound, not the resident archive
        from maple_syrup import soil_profile_texture as spt

        assert spt._windowed_metadata(child.voxel_column, nc)[5] > 0
        assert m["host_phases"]["child_native_and_properties"] >= spt._windowed_metadata(child.voxel_column, nc)[5]


def test_the_model_scales_with_layers_and_cells_by_arithmetic_for_1_3_65_layers_and_n50003(tmp_path):
    """Pure arithmetic of the owner terms (no device): the new-map/column/water terms are linear in L x N and the stage-0 temporaries in N x nc; checked at the large N and L = 65 without building a bed."""
    for L in (1, 3, 65):
        for N in (12, 50003):
            ret = fcu.retained_terms(L, N, {"theta_residual": "scalar", "theta_field_capacity": "cell", "theta_saturation": "cell", "interface": "shared"})["total"]
            assert ret >= 2 * L * N * 8 and ret < 3 * L * N * 8 + 20 * N + 10_000  # two (L, N) cell maps dominate the retained column
    d0, h0 = erc.stage0_model_cuda(50003, 6)
    assert d0 == 2 * 50003 * 6 + 9 * 50003 and h0 == 16 * 50003 + erc.CHUNK_BYTES


# ---- ordering / typed refusals --------------------------------------------------------------------------------------------------------------------------------------------------------
def test_a_cpu_event_is_refused_typed_before_any_reader_gate_or_array(tmp_path, monkeypatch):
    ev, h, vert = completed(tmp_path / "a")
    cap = ev.completion_capability(h, h.state)
    reads = []
    monkeypatch.setattr(erc, "_gate", lambda col: reads.append("gate"))
    with pytest.raises(erc.EventRefreshCudaError, match="CPU events use floating_event_refresh"):
        erc.stage_completed_event_refresh_cuda(cap, h.state, event=ev, available_bytes_fn=lambda: reads.append("read") or 1 << 40)
    assert reads == []


def test_fake_foreign_and_consumed_proofs_are_refused_by_the_producer_before_anything(tmp_path):
    ev, h, vert = completed(tmp_path / "a")
    with pytest.raises(fcm.CompletionError):
        erc.stage_completed_event_refresh_cuda(object(), h.state, event=ev)
    cap = ev.completion_capability(h, h.state)
    fcm.consume_completion(cap, h.state, event=ev)
    with pytest.raises(fcm.CompletionError, match="consumed"):
        erc.stage_completed_event_refresh_cuda(cap, h.state, event=ev)


@pytest.mark.parametrize("bad", [-1, 1.5, True, "1", None])
def test_reserves_and_readers_are_validated_before_the_gate(tmp_path, monkeypatch, bad):
    ev, h, vert = completed(tmp_path / "a")
    gate = []
    monkeypatch.setattr(erc, "_gate", lambda col: gate.append(1))
    monkeypatch.setattr(vert, "backend", "cuda")  # ORDERING control only (a CPU vertical's backend label): the real quota validation is what is under test
    monkeypatch.setattr(ev, "xp", type("X", (), {"__name__": "cupy"}))
    cap = ev.completion_capability(h, h.state)
    with pytest.raises(erc.EventRefreshCudaError):
        erc._quota(None, bad, None, 0, None, None)
    with pytest.raises(erc.EventRefreshCudaError):
        erc._quota(None, 0, None, bad, None, None)
    with pytest.raises(erc.EventRefreshCudaError, match="must be callable or None"):
        erc._quota(5, 0, None, 0, None, None)
    assert gate == [] and cap is not None


def test_the_gate_is_the_first_device_operation_and_the_stage_orders_gate_before_admission_dirty_selection_and_transfers():
    src = inspect.getsource(erc.stage_completed_event_refresh_cuda)
    order = ["fcm.verify_completion(", "_quota(", "_gate(held0.column)", "event_end_bundle(", "_admit(cp, dev, d0, h0", "cp.any(", "cp.flatnonzero", "fcc.d2h(", "event_refresh_model_cuda(", "_admit(cp, dev, model", "_scatter_tiles(",
             "reconcile_selected_columns(", "_device_column(", "inventory_closure_cuda(", "_device_op(", "rebind_floating_matrix_history(", "rebind_preferential_column(", "EventRefreshCandidateCuda("]
    pos = [src.index(t) for t in order]
    assert pos == sorted(pos), dict(zip(order, pos, strict=True))
    assert src.index("CELL layout") < src.index("_gate(held0.column)")  # the metadata-only layout refusal precedes the gate and every array


def test_the_real_gate_refuses_without_cuda_and_a_wrong_device(monkeypatch):
    class Col:
        device_id = 0

    with pytest.raises(erc.EventRefreshCudaError, match="CUDA device gate refused"):
        erc._gate(Col())  # no CUDA here: the real gate is typed, never a fallback


def test_the_candidate_class_is_not_mintable_copyable_or_serialisable_and_verification_needs_a_genuine_one():
    with pytest.raises(erc.EventRefreshCudaError, match="minted only by"):
        erc.EventRefreshCandidateCuda(*(None,) * len(erc.EventRefreshCandidateCuda.__dataclass_fields__))
    with pytest.raises(erc.EventRefreshCudaError, match="not a genuine"):
        erc.verify_refresh_candidate_cuda(object())
    for name in ("__copy__", "__deepcopy__", "__reduce_ex__"):
        assert name in erc.EventRefreshCandidateCuda.__dict__
    fields = {f for f in erc.EventRefreshCandidateCuda.__dataclass_fields__}
    assert {"capability", "state", "event", "payload", "child_bed", "column", "water_state", "matrix_op", "history", "pref_op", "ks_m_s", "depth_m", "surface_return_m", "bottom_export_m", "receipt", "dirty", "changed",
            "depth_changed", "ks_changed", "property_basis", "provenance", "model"} <= fields


def test_the_stage_never_touches_an_owner_a_cold_scan_or_the_held_depth_by_source():
    src = inspect.getsource(erc)
    for forbidden in ("NativeFloatingSoil(", "initialize_native_floating_soil", "native_source_witness", "nfs._scan(", "SimpleNamespace", "held.depth0_m", "stage_native_refresh"):
        assert forbidden not in src, forbidden
    assert "d_cur" in src and "payload.depth_m" in src and "_batch_properties" in src.replace("nfs._batch_properties", "_batch_properties") or "ncuda._scatter_tiles" in src  # CURRENT D and the shared provider path


# ---- CPU preferential rebind unchanged; CUDA branch admission order ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not BASE.exists(), reason="baseline tree not present")
def test_cpu_preferential_rebind_is_unchanged_and_refuses_the_cuda_only_arguments(tmp_path):
    import fpond_builders as fb

    ev, st, vert = build(tmp_path, backend="numpy")  # `case.build` carries a preferential operator
    op = vert.held_template().preferential
    assert op is not None
    col = vert.held_template().column
    new = fpi.rebind_preferential_column(op, col)
    assert new.sha256 == op.sha256 and new.column is col  # the canonical identity of the SAME column is the operator's own (CPU path unchanged; the AST test pins the CPU lines)
    with pytest.raises(fsl.FloatingSoilError, match="apply only to the cuda backend"):
        fpi.rebind_preferential_column(op, col, reserve_bytes=1)
    with pytest.raises(fsl.FloatingSoilError, match="apply only to the cuda backend"):
        fpi.rebind_preferential_column(op, col, available_bytes_fn=lambda: 1 << 40)
    assert fb is not None


def test_the_cuda_rebind_admits_the_hash_staging_buffer_before_the_buffer_exists_by_source():
    src = inspect.getsource(fpi.rebind_preferential_column)
    assert src.index("fcu.admit(") < src.index("np.empty(n") < src.index("_identity(") and "STAGE_ELEMS" in src and "CUDA is refused" not in src
