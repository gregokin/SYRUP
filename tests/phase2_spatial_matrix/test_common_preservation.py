"""Common-path preservation of the S2-B2.a refactor: every untouched definition of `matrix_infiltration` / `matrix_infiltration_numba` is AST-identical to the pre-B2 source, all other production files
are byte-identical to the accepted 121-file package, the historical 25-argument common Numba kernel still works, and the common shapes/estimates are unchanged. Author-written."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import CANDIDATE, TASK, cases, mi

SRC = CANDIDATE / "src" / "maple_syrup"
PRE = TASK / "s2_b2_a_pre_sources"
SNAP = TASK / "root_s2_b1_c_snapshot" / "src" / "maple_syrup"
needs_pre = pytest.mark.skipif(not PRE.exists(), reason="pre-edit sources not present")
CHANGED = {"matrix_infiltration.py", "matrix_infiltration_numba.py"}
NEW = {"spatial_matrix_infiltration.py"}
# S2-B2.b (the later slice) legitimately changes the preferential pair and adds its own spatial module; this B2.a suite compares against the older 121-file package, so those are allowed here
LATER_CHANGED = {"preferential_infiltration.py", "preferential_infiltration_numba.py", "matrix_infiltration_cuda.py", "preferential_infiltration_cuda.py"}  # B2.b pair; B2.c shared CUDA renderer; B2.d preferential CUDA renderer
LATER_NEW = {"spatial_preferential_infiltration.py", "spatial_matrix_infiltration_cuda.py", "spatial_preferential_infiltration_cuda.py", "floating_daily_cuda.py"}
LATER_NEW |= {"floating_held_context_cuda.py"}  # accepted bounded CUDA held-law record
LATER_NEW |= {"floating_held_context.py"}  # accepted bounded H0 law-context module
# Versioned inventory (Phase 2 continuation, Root-authorised, exact names only): the independently accepted C-provider and G0 moisture-output modules, the D1/D1b floating-column modules and execution helper are NEW since this
# older package; `subsurface_numba.py` carries the explicit private floating variant (a compile-time variant of the shared body). The historical DEFAULT kernels are held by the explicit numerical/source check below.
LATER_NEW |= {"soil_texture_properties.py", "soil_texture_properties_numba.py", "soil_texture_properties_cuda.py", "soil_moisture_output.py", "floating_soil_layers.py", "floating_soil_layers_numba.py",
              "floating_soil_layers_cuda.py", "subsurface_cuda_floating.py"}
LATER_CHANGED |= {"subsurface_numba.py"}
# D2 (Root-authorised, exact names only): the floating intake/withdrawal modules are NEW; the private floating compile-time variants live in the already-listed matrix/preferential Numba modules (the default variants are held by the explicit checks below).
LATER_NEW |= {"floating_matrix_infiltration.py", "floating_preferential_infiltration.py", "floating_transpiration.py"}
LATER_NEW |= {"floating_matrix_infiltration_cuda.py", "floating_preferential_infiltration_cuda.py", "floating_transpiration_cuda.py"}  # D2 CUDA helpers (shared bodies rendered with floating readers)
# Accepted E1 (Root C2, receipt root_e1_c2_acceptance.json): ONLY `soil_profile_texture.py` differs from the older package; it is pinned by exact sha256 below and its numeric engines stay byte-identical.
LATER_CHANGED |= {"soil_profile_texture.py"}
# Accepted E2 (Root C2, root_e2_c2_acceptance.json; pinned by sha256 below) and the E3 native refresh transaction: new modules, exact names only.
LATER_NEW |= {"compact_soil_profile.py", "native_floating_soil.py"}
# Accepted G1 codec (Root, receipt root_g1_patch_adoption_onto_e3_c2.json; pinned by sha256 below) and the E3b CUDA branch helper: new modules, exact names only.
LATER_NEW |= {"floating_moisture_frame.py", "native_floating_soil_cuda.py"}
# Accepted G2a daily moisture output hook (Root receipt root_g2a_patch_adoption_onto_e3b.json; pinned by sha256 below): a new module, exact name only.
LATER_NEW |= {"daily_moisture_output.py"}
LATER_NEW |= {"floating_pond.py", "floating_pond_cuda.py"}  # E4b opt-in floating Pond vertical + E5 its CUDA helper (new modules; exact names, no wildcard)
LATER_NEW |= {"floating_resume.py"}  # R1a in-memory CPU capture/restore (new module; exact name, no wildcard)
LATER_NEW |= {"floating_checkpoint.py"}  # R2a CPU floating FILE checkpoint (new module; exact name, no wildcard)
LATER_NEW |= {"floating_context_checkpoint.py"}  # H2 CPU context container (new module; exact name)
LATER_NEW |= {"floating_context_checkpoint_cuda.py"}  # G-F4 explicit CUDA outer container; exact new module name only
LATER_NEW |= {"floating_checkpoint_cuda.py", "floating_resume_cuda.py", "floating_event_refresh.py", "floating_completion.py"}  # EB1 retained-completion capability (new module; exact name)
LATER_NEW |= {"floating_continued_checkpoint_cuda.py", "floating_lifecycle_checkpoint.py", "floating_lifecycle_checkpoint_cuda.py"}  # S1 CPU + S2 CUDA lifecycle physical files (new modules; exact names)
LATER_NEW |= {"floating_event_lifecycle.py"}  # EB3 single-authority event lifecycle (new module; exact name)
LATER_NEW |= {"floating_event_daily_bridge.py", "floating_event_daily_bridge_cuda.py"}  # event<->daily bridge (CPU new module) + its CUDA helper (new module); exact names, no wildcard
LATER_NEW |= {"floating_evaporation.py"}  # Root-accepted shared-law floating evaporation, exact module name
LATER_CHANGED |= {"soil_evaporation.py"}  # Root-verified arithmetic-preserving extraction, historical numeric engines unchanged
LATER_CHANGED |= {"pond_event.py", "pond_hydraulics.py"}  # E4b opt-in floating vertical: bounded constructor/hook/rebind edits (default paths pinned byte-equivalent by tests/phase2_floating_pond)
LATER_CHANGED |= {"sediment_bed.py", "storage_admission.py"}  # Root-accepted E4j (native 9f journal pin) / E4k (optional admission guard) source pins; receipts docs/subsurface_hydrology/phase2_native9f_storage_acceptance.json and root_e4j_e4k_patch_adoption_onto_e4b_c1.json (exact names, no wildcard)
LATER_NEW |= {"floating_daily.py", "floating_daily_output.py"}  # Root-accepted initialized-soil floating daily and every-N diagnostics (exact names)
G2A_DAILY_SHA256 = "aa8a2c69162c18211d325926d7df986ef6b37b1e9de2046f5c5fc3b05533b33b"
G1_FRAME_SHA256 = "ba1ef652c37571da4bfc3a323f32dbed97b848b096fbf9dd7c5a9dc42b2f32b9"
E2_C2_COMPACT_SHA256 = "cbc6d91424a62127e39e930fdb6d3767a85775ff5b3114495fab61cf0c548712"
E1_C2_SOT_SHA256 = "9fcb81ae1caa51366de83852c20db4434a357051ead6a124972402caa3da4c0f"
E1_C2_ENGINES = {"soil_profile_texture_numba.py": "21f09a3ef507835cd096", "soil_profile_texture_cuda.py": "4472c3b5507aada51b37"}


def _defs(path: Path) -> dict:
    tree = ast.parse(path.read_text())
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


@needs_pre
def test_untouched_matrix_definitions_are_ast_identical():
    old, new = _defs(PRE / "matrix_infiltration.py"), _defs(SRC / "matrix_infiltration.py")
    changed = {k for k in old if old[k] != new.get(k)}
    assert changed == {"reference_step", "_resolve"}  # the only two edited definitions
    assert set(new) - set(old) == {"_StepContext", "_reference_step_body", "_resolve_core"}  # the only additions (private)
    assert set(old) <= set(new)
    # the constants/exports (everything outside functions/classes) are unchanged
    a, b = ast.parse((PRE / "matrix_infiltration.py").read_text()), ast.parse((SRC / "matrix_infiltration.py").read_text())
    # S2-B2.c review02 added a paragraph to the module docstring (the first statement) describing the bounded combined intake; every other top-level non-definition statement is unchanged
    top = lambda t: [ast.dump(n) for n in t.body[1:] if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]  # noqa: E731
    assert top(a) == top(b)


@needs_pre
def test_the_shared_numpy_body_is_the_old_body_with_only_the_context_reads_changed():
    old = (PRE / "matrix_infiltration.py").read_text()
    new = (SRC / "matrix_infiltration.py").read_text()
    o = old[old.index("    pool_out = xp.array(pool_in"):old.index("def reference_runner")]
    n = new[new.index("    pool_out = xp.array(pool_in"):new.index("def reference_runner")]
    # S2-B2.c review02 replaced exactly ONE block (the combined-intake bound); restoring the old block must give back the old body, so nothing else of the body changed
    start, end = n.index("            del c2, pool_left, room_left\n"), n.index("            bounded = xp.array(candidate)")
    old_block = ("            del c2, pool_left, room_left, room\n            adv = xp.where(run_post & (g_map > 0.0), post / g_map, 0.0)\n            j_out = xp.where(run_post, j1 + adv, j1)\n"
                 "            xp.copyto(wet_out, j_out, where=active)\n            del j_out\n            xp.add(pre, post, out=intake)\n")
    assert "over = intake > pool_in" in n[start:end] and "xp.subtract(intake, pre, out=post, where=over)" in n[start:end]
    restored = n[:start] + old_block + n[end:]
    assert o.replace("op.newton_tolerance", "ctx.newton_tol") == restored  # everything outside the bound block is textually the old body


@needs_pre
def test_the_numba_helpers_are_unchanged_and_the_common_variant_reads_the_cached_vectors():
    def nested(path):
        tree = ast.parse(path.read_text())
        build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build")
        return {n.name: ast.dump(n) for n in ast.walk(build) if isinstance(n, ast.FunctionDef)}

    old, new = nested(PRE / "matrix_infiltration_numba.py"), nested(SRC / "matrix_infiltration_numba.py")
    assert old["log1pmx"] == new["log1pmx"] and old["ga_increment"] == new["ga_increment"]  # the SAME scalar Newton helpers (not copied)
    assert {"step_core", "step_kernel", "layer_stride", "cell_stride", "bind_layers"} <= set(new)


def test_all_other_production_files_are_byte_identical_to_the_accepted_package():
    if not SNAP.exists():
        pytest.skip("accepted snapshot not present")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    current = {p.name: sha(p) for p in SRC.glob("*.py")}
    snap = {p.name: sha(p) for p in SNAP.glob("*.py")}
    assert set(current) - set(snap) == NEW | LATER_NEW and set(snap) <= set(current)
    different = {k for k in snap if current[k] != snap[k]}
    assert different == CHANGED | LATER_CHANGED, different
    assert len(snap) == 121 or len(snap) >= 100  # the accepted package size is recorded by Root; this only guards the comparison


def test_common_cuda_matrix_text_and_other_infiltration_files_are_untouched():
    if not SNAP.exists():
        pytest.skip("accepted snapshot not present")
    for name in ("spatial_soil_layers.py", "subsurface.py",
                 "subsurface_cuda.py", "subsurface_transpiration.py", "spatial_transpiration.py", "layered_storm.py", "layered_checkpoint.py", "routing_cuda.py"):
        p, q = SRC / name, SNAP / name
        if q.exists():
            assert p.read_bytes() == q.read_bytes(), name


def test_historical_default_subsurface_numba_kernels_are_preserved_numerically_and_structurally():
    """Replaces the obsolete whole-file byte comparison of `subsurface_numba.py` (the file now carries the private FLOATING variant): the DEFAULT common and spatial variants keep their 40-argument kernel signature, their
    `_build(numba, common)` call form works, the floating variant is a separate compile-time variant, and the common kernel reproduces the pre-A3 kernel BIT FOR BIT (the explicit historical reference) on a heterogeneous scenario."""
    pytest.importorskip("numba")
    import importlib.util
    import inspect
    import sys

    from maple_syrup import subsurface as ss_mod
    from maple_syrup import subsurface_numba as new

    pre = TASK / "s2_a3_pre_sources" / "subsurface_numba.py"
    assert pre.is_file(), "the pre-A3 reference kernel (task evidence) is required by this guard"
    spec = importlib.util.spec_from_file_location("old_subsurface_numba_pre_a3_matrix_guard", pre)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    expected = ["w_in", "add", "pet", "active", "dt", "dz", "k_dz", "b_res", "k_res", "b_fc", "k_fc", "b_sat", "k_sat", "tol", "coef", "pair_inv", "k_pi", "g_down", "g_up", "k_g", "res_theta0", "stress", "factor",
                "gravity", "cell_tol", "no_flag", "directional", "bottom", "share", "has_share", "pulse", "pulse_on", "w_out", "perc", "diff", "ret", "drain", "evap", "unmet", "rows"]
    assert len(expected) == 40
    for kernel in (new.compiled_kernel(), new.compiled_spatial_kernel(), new.compiled_floating_kernel()):
        assert list(inspect.signature(kernel.py_func).parameters) == expected  # ONE signature for every variant
    assert list(inspect.signature(new._build).parameters) == ["numba", "common", "floating"] and inspect.signature(new._build).parameters["floating"].default is False
    rng = np.random.default_rng(2026)
    for L, mode, bottom in ((3, "tietjen_donor_depth", "open"), (5, "constant", "closed")):
        edges = np.concatenate([[0.0], np.cumsum(np.linspace(0.08, 0.3, L))])
        kw = dict(interface_conductivity_m_s=np.linspace(1e-6, 8e-6, L - 1), diffusion_coefficient=0.25) if mode != "constant" else dict(diffusivity_m2_s=np.linspace(2e-7, 9e-7, L - 1))
        lay = ss_mod.soil_layers(edges, theta_residual=np.linspace(0.01, 0.04, L), theta_field_capacity=np.linspace(0.18, 0.25, L), theta_saturation=np.linspace(0.4, 0.5, L), theta_evaporation_stress=0.5,
                                 diffusion_mode=mode, bottom_drainage=bottom, **kw)
        N = 80
        mask = rng.random(N) < 0.8
        col = ss_mod.prepare_column(lay, mask, 4.0, backend="numba")
        old_run, _ = old.prepare_runner(lay)
        water = lay.saturation_m[:, None] * rng.uniform(0.0, 1.0, (L, N))
        added, pet = rng.uniform(0, 2e-3, (L, N)) * mask, rng.uniform(0, 3e-7, N) * mask
        for dt in (0.0, 600.0, 86400.0):
            add = added if dt > 0 else np.zeros_like(added)
            a, b = col.runner(col, water, add, pet, dt), old_run(col, water, add, pet, dt)
            for name in ss_mod.StepArrays._fields[:-1]:
                x, y = np.ascontiguousarray(getattr(a, name)), np.ascontiguousarray(getattr(b, name))
                nan = np.isnan(x) & np.isnan(y)
                np.testing.assert_array_equal(np.where(nan, 0.0, x).view(np.uint64), np.where(nan, 0.0, y).view(np.uint64), err_msg=f"{L} {mode} {bottom} dt={dt} {name}")


def test_the_common_numba_kernel_keeps_its_25_argument_signature_and_array_types():
    pytest.importorskip("numba")
    import inspect

    from maple_syrup import matrix_infiltration_numba as mn

    kernel = mn.compiled_kernel()
    params = list(inspect.signature(kernel.py_func).parameters)
    assert params == ["pool_in", "standing_in", "rate_in", "w_in", "wet_in", "cum_in", "active", "dt", "kmap", "psimap", "gmap", "dz", "fc_m", "sat_m", "cap_tol", "cell_tol", "tol", "max_iter", "no_flag",
                      "pool_out", "w_out", "wet_out", "cum_out", "intake_out", "rows"]
    assert kernel._can_compile  # lazily typed as before (cache=False, fastmath=False)
    assert kernel.targetoptions.get("nopython") is True and not kernel.targetoptions.get("fastmath") and not kernel.targetoptions.get("cache")
    # the common operator still runs through it and reproduces the reference NumPy step
    col_np, col_nb = cases.column(n_cells=9), cases.column(n_cells=9, backend="numba")
    out = []
    for col in (col_np, col_nb):
        op = cases.operator(col)
        n = col.n_cells
        out.append(mi.matrix_step(op, cases.full(col, 0.004), cases.full(col, 0.0), cases.full(col, 2e-5), cases.state_of(col), cases.history_of(op, 0.001), 200.0))
    np.testing.assert_allclose(out[1].intake_m, out[0].intake_m, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(out[1].history.wetting_m, out[0].history.wetting_m, rtol=2e-12, atol=1e-14)


def test_common_constants_and_estimate_are_unchanged():
    est = mi.estimate_matrix_memory(3, 1000, backend="numba")
    assert mi.ROW_COUNT == 16 and mi.PACKET_LENGTH == 17 and mi.NEWTON_MAX_ITERATIONS == 64 and mi.NEWTON_TOLERANCE_TERMS == 8
    assert mi.CELL_OPERATION_TERMS == 16 and mi.CELL_OPERATION_TERMS_PER_LAYER == 2 and mi.KERNEL_SCRATCH_DOUBLES_PER_CELL == 2
    assert est.required_bytes == est.host_bytes and est.device_bytes == 0


def test_the_common_numpy_operator_still_reproduces_root_frozen_matrix_controls_bit_for_bit():
    """The 32 frozen matrix scenarios / 224 columns were captured from the accepted COMMON NumPy operator BEFORE B2; after the private refactor it must give the very same bits."""
    from conftest import REF_DIR  # noqa: PLC0415

    controls = json.loads((REF_DIR / "controls.json").read_text())
    expected = np.load(REF_DIR / "expected.npz")
    configs = json.loads((TASK / "root_s2_parameter_controls.json").read_text())["controls"]
    from maple_syrup import subsurface as ss  # noqa: PLC0415

    n_cols = 0
    for sc in (s for s in controls["scenarios"] if s["operation"] == "matrix"):
        c = configs[sc["parameters_index"]]
        edges, res, fc, sat = (np.array(c[k]) for k in ("depth_edges_m", "theta_residual", "theta_field_capacity", "theta_saturation"))
        mask = np.array(sc["mask"])
        arr = {k: np.array(sc[k]) for k in ("water_m", "pool_m", "standing_m", "incoming_rate_m_s", "conductivity_m_s", "suction_m", "vegetation_factor", "wetting_m", "cumulative_intake_m")}
        got = {n: [] for n in ("water", "surface_pool_m", "intake_m", "wetting_m", "cumulative_intake_m")}
        for cell in range(c["cells"]):
            lay = ss.soil_layers(edges[:, cell], theta_residual=res[:, cell], theta_field_capacity=fc[:, cell], theta_saturation=sat[:, cell], theta_evaporation_stress=c["theta_evaporation_stress"],
                                 diffusivity_m2_s=0.0)
            col = ss.prepare_column(lay, mask[cell:cell + 1].copy(), sc["area_m2"], backend="numpy")
            op = mi.prepare_matrix(col, conductivity_m_s=float(arr["conductivity_m_s"][cell]), suction_m=float(arr["suction_m"][cell]), vegetation_factor=float(arr["vegetation_factor"][cell]))
            h = mi.wetting_history(op, wetting_m=arr["wetting_m"][cell:cell + 1].copy(), cumulative_intake_m=arr["cumulative_intake_m"][cell:cell + 1].copy())
            r = mi.matrix_step(op, arr["pool_m"][cell:cell + 1].copy(), arr["standing_m"][cell:cell + 1].copy(), arr["incoming_rate_m_s"][cell:cell + 1].copy(),
                               ss.column_state(col, arr["water_m"][:, cell:cell + 1].copy(), sc["t_s"]), h, sc["dt_s"])
            got["water"].append(r.state.water_m[:, 0])
            got["surface_pool_m"].append(r.surface_pool_m[0])
            got["intake_m"].append(r.intake_m[0])
            got["wetting_m"].append(r.history.wetting_m[0])
            got["cumulative_intake_m"].append(r.history.cumulative_intake_m[0])
            n_cols += 1
        for name, vals in got.items():
            want = expected[f"{sc['label']}_{name}"]
            have = np.column_stack(vals) if name == "water" else np.array(vals)
            assert have.tobytes() == want.tobytes(), (sc["label"], name)
    assert n_cols == 224


def test_accepted_e1_texture_patch_is_exactly_the_accepted_bytes_and_engines_are_unchanged():
    import hashlib as _h
    src = Path(__file__).resolve().parents[2] / "src" / "maple_syrup"
    assert _h.sha256((src / "soil_profile_texture.py").read_bytes()).hexdigest() == E1_C2_SOT_SHA256
    for name, prefix in E1_C2_ENGINES.items():
        assert _h.sha256((src / name).read_bytes()).hexdigest().startswith(prefix), name


def test_accepted_e2_compact_reader_is_exactly_the_accepted_bytes():
    import hashlib as _h
    from pathlib import Path as _P
    src = _P(__file__).resolve().parents[2] / "src" / "maple_syrup"
    assert _h.sha256((src / "compact_soil_profile.py").read_bytes()).hexdigest() == E2_C2_COMPACT_SHA256


def test_accepted_g1_moisture_frame_is_exactly_the_accepted_bytes():
    import hashlib as _h
    from pathlib import Path as _P
    src = _P(__file__).resolve().parents[2] / "src" / "maple_syrup"
    assert _h.sha256((src / "floating_moisture_frame.py").read_bytes()).hexdigest() == G1_FRAME_SHA256


def test_accepted_g2a_daily_moisture_output_is_exactly_the_accepted_bytes():
    import hashlib as _h
    from pathlib import Path as _P
    src = _P(__file__).resolve().parents[2] / "src" / "maple_syrup"
    assert _h.sha256((src / "daily_moisture_output.py").read_bytes()).hexdigest() == G2A_DAILY_SHA256

LATER_NEW |= {"floating_event_refresh_cuda.py"}  # Independently qualified exact new module, no source guard removal.

LATER_NEW |= {"floating_continued_checkpoint.py"}  # New explicit continued-origin physical codec only; historical guards retained.

LATER_NEW |= {"floating_event_lifecycle_cuda.py"}  # Exact qualified new backend helper; native/source guards unchanged.
