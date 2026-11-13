"""Common-path preservation of the S2-B2.b refactor: every untouched definition of `preferential_infiltration` is AST-identical to the pre-B2.b source, the shared derivation/body are the old text with
only the context reads changed, all other production files are byte-identical to the accepted 122-file package (except the new spatial module), the historical 16-argument common Numba kernel still
works, and the common operators still reproduce Root's frozen controls bit for bit. Author-written."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json

import numpy as np
import pytest
from conftest import CANDIDATE, PARAMS, REF_NUMBA, REF_NUMPY, TASK, pi, ss

SRC = CANDIDATE / "src" / "maple_syrup"
PRE = TASK / "s2_b2_b_pre_sources"
SNAP = TASK / "root_s2_b2_a_snapshot" / "src" / "maple_syrup"
needs_pre = pytest.mark.skipif(not PRE.exists(), reason="pre-edit sources not present")
CHANGED = {"preferential_infiltration.py", "preferential_infiltration_numba.py"}
NEW = {"spatial_preferential_infiltration.py"}
# S2-B2.c (the later CUDA matrix slice) legitimately changes the shared CUDA matrix renderer and the spatial matrix module and adds its CUDA module; this suite compares against the older 122-file package
LATER_CHANGED = {"matrix_infiltration_cuda.py", "spatial_matrix_infiltration.py", "matrix_infiltration.py", "matrix_infiltration_numba.py", "preferential_infiltration.py", "preferential_infiltration_cuda.py"}  # + S2-B2.d: derivation hook and shared CUDA renderer  # + S2-B2.c review02 bounded combined intake in the three shared matrix bodies
LATER_NEW = {"spatial_matrix_infiltration_cuda.py", "spatial_preferential_infiltration_cuda.py", "floating_daily_cuda.py"}
LATER_NEW |= {"floating_held_context_cuda.py"}  # accepted bounded CUDA held-law record
LATER_NEW |= {"floating_held_context.py"}  # accepted bounded H0 law-context module
# Versioned inventory (Phase 2 continuation, Root-authorised): the independently accepted C-provider modules, the G0 moisture-output writer and the D1 floating-column modules are NEW since the older 122-file package, and
# `subsurface_numba.py` carries the explicit private floating-variant hooks of D1 (a third compile-time variant of the shared body; the default variants are held bit-identical by phase2_spatial_numba). Exact names only.
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


def _defs(path):
    tree = ast.parse(path.read_text())
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


@needs_pre
def test_untouched_preferential_definitions_are_ast_identical():
    old, new = _defs(PRE / "preferential_infiltration.py"), _defs(SRC / "preferential_infiltration.py")
    changed = {k for k in old if old[k] != new.get(k)}
    assert changed == {"prepare_preferential", "reference_step", "_resolve"}
    assert set(new) - set(old) == {"_DeriveContext", "_derive_pathway", "_StepContext", "_reference_step_body", "_resolve_core", "_reduce"}  # `_reduce`: S2-B2.d cold-reduction helper
    assert set(old) <= set(new)
    a, b = ast.parse((PRE / "preferential_infiltration.py").read_text()), ast.parse((SRC / "preferential_infiltration.py").read_text())
    # S2-B2.d added the single top-level constant `REDUCE_KINDS`; every other top-level non-definition statement is unchanged
    top = lambda t: [ast.dump(n) for n in t.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "REDUCE_KINDS")]  # noqa: E731
    assert top(a) == top(b)


@needs_pre
def test_the_shared_numpy_body_is_the_old_body_with_only_the_context_reads_changed():
    old = (PRE / "preferential_infiltration.py").read_text()
    new = (SRC / "preferential_infiltration.py").read_text()
    o = old[old.index("    surface_out = xp.array(surface_in"):old.index("def reference_runner")]
    n = new[new.index("    surface_out = xp.array(surface_in"):new.index("def reference_runner")]
    assert o == n  # everything below the preamble is textually the old body


@needs_pre
def test_the_shared_derivation_is_the_old_build_text():
    old = (PRE / "preferential_infiltration.py").read_text()
    new = (SRC / "preferential_infiltration.py").read_text()
    o = old[old.index("    # Build scratch plan"):old.index("    sat_xp = to_device")]
    n = new[new.index("    # Build scratch plan"):new.index("    return rate, receiver, beta, n_enabled")]
    o = o.replace('            ss._check_dynamic(column, f"species_contributions[{s}]", c, (N,))\n', "            ctx.check_contribution(s, c)\n").replace("        xp.copyto(max_rate, 0.0, where=inactive)\n", "")
    # S2-B2.d routed EXACTLY the eight cold reductions through `_reduce`; restoring the previous expressions must give back the old derivation, so nothing else of it changed
    back = {
        '_reduce(ctx, "count_nonfinite", c) != 0.0:  # transient (N,) bool (default path)': 'to_float(xp.sum(~xp.isfinite(c))) != 0.0:  # transient (N,) bool',
        '_reduce(ctx, "min", c) < 0.0:': 'to_float(xp.min(c)) < 0.0:',
        '_reduce(ctx, "count_nonfinite", cover) != 0.0:': 'to_float(xp.sum(~xp.isfinite(cover))) != 0.0:',
        'worst = _reduce(ctx, "max", temp)': 'worst = to_float(xp.max(temp))',
        'lo, hi = _reduce(ctx, "min", beta), _reduce(ctx, "max", beta)': 'lo, hi = to_float(xp.min(beta)), to_float(xp.max(beta))',
        '_reduce(ctx, "count_nonfinite", receiver[layer]) != 0.0:': 'to_float(xp.sum(~xp.isfinite(receiver[layer]))) != 0.0:',
        '_reduce(ctx, "count_nan", rate) != 0.0:': 'to_float(xp.sum(xp.isnan(rate))) != 0.0:',
    }
    for new_text, old_text in back.items():
        assert n.count(new_text) == 1, new_text
        n = n.replace(new_text, old_text)
    assert "_reduce(" not in n
    assert o == n  # the old derivation, moved: only the contribution check and the caller-owned max-rate zeroing differ


@needs_pre
def test_numba_common_kernel_keeps_its_16_argument_signature_and_the_body_arithmetic():
    pytest.importorskip("numba")
    from maple_syrup import preferential_infiltration_numba as pn

    kernel = pn.compiled_kernel()
    assert list(inspect.signature(kernel.py_func).parameters) == ["surface_in", "w_in", "active", "dt", "rate", "max_rate", "recv", "sat_m", "cap_tol", "cell_tol", "no_flag", "surface_out", "w_out",
                                                                   "intake", "total", "rows"]
    assert kernel._can_compile and kernel.targetoptions.get("nopython") is True and not kernel.targetoptions.get("fastmath") and not kernel.targetoptions.get("cache")
    old = (PRE / "preferential_infiltration_numba.py").read_text()
    new = (SRC / "preferential_infiltration_numba.py").read_text()
    # the arithmetic between "isolated attempt" and the closure checks is untouched except the three reads of the bound values / the cap
    o = old[old.index("                # isolated attempt"):old.index("                nonfinite = not math.isfinite(total_v)")]
    n = new[new.index("                # isolated attempt"):new.index("                nonfinite = not math.isfinite(total_v)")]
    o = o.replace("                cap = max_rate[i] * dt\n", "                if common:\n                    cap = max_rate[i] * dt\n                else:\n                    cap = max_rate[i * k_mr] * dt\n").replace("sat_m[l]", "sat_c[l]").replace("cap_tol[l]", "tol_c[l]")
    assert o == n


def test_all_other_production_files_are_byte_identical_to_the_accepted_package():
    if not SNAP.exists():
        pytest.skip("accepted snapshot not present")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    current = {p.name: sha(p) for p in SRC.glob("*.py")}
    snap = {p.name: sha(p) for p in SNAP.glob("*.py")}
    assert set(current) - set(snap) == NEW | LATER_NEW and set(snap) <= set(current)
    assert {k for k in snap if current[k] != snap[k]} == CHANGED | LATER_CHANGED


def test_the_common_cuda_matrix_and_native_files_are_untouched():
    if not SNAP.exists():
        pytest.skip("accepted snapshot not present")
    for name in ("spatial_soil_layers.py",
                 "subsurface.py", "layered_storm.py", "layered_checkpoint.py", "routing_cuda.py"):
        p, q = SRC / name, SNAP / name
        if q.exists():
            assert p.read_bytes() == q.read_bytes(), name


def test_common_constants_and_estimate_are_unchanged():
    est = pi.estimate_preferential_memory(3, 1000, backend="numba", n_species=2)
    assert pi.ROW_COUNT == 8 and pi.PACKET_LENGTH == 9 and pi.CELL_OPERATION_TERMS == 4 and pi.CELL_OPERATION_TERMS_PER_LAYER == 4
    assert est.required_bytes == est.host_bytes and est.device_bytes == 0


@pytest.mark.skipif(not (REF_NUMPY / "controls.json").exists() or not PARAMS.exists(), reason="Root's frozen reference is not present")
@pytest.mark.parametrize("backend_name", ["numpy", "numba"])
def test_the_common_operators_still_reproduce_root_frozen_preferential_controls_bit_for_bit(backend_name):
    """The 48 frozen preferential scenarios / 336 columns were captured from the accepted COMMON operator on each backend BEFORE B2; after the private refactor they must give the same bits."""
    if backend_name == "numba":
        pytest.importorskip("numba")
    ref = REF_NUMPY if backend_name == "numpy" else REF_NUMBA
    controls = json.loads((ref / "controls.json").read_text())
    expected = np.load(ref / "expected.npz")
    configs = json.loads(PARAMS.read_text())["controls"]
    n_cols = 0
    for sc in (s for s in controls["scenarios"] if s["operation"] == "preferential"):
        c = configs[sc["parameters_index"]]
        edges, res, fc, sat = (np.array(c[k]) for k in ("depth_edges_m", "theta_residual", "theta_field_capacity", "theta_saturation"))
        mask, water, surface = np.array(sc["mask"]), np.array(sc["water_m"]), np.array(sc["surface_m"])
        got = {n: [] for n in ("water", "intake_m", "surface_water_m", "intake_total_m")}
        for cell in range(c["cells"]):
            lay = ss.soil_layers(edges[:, cell], theta_residual=res[:, cell], theta_field_capacity=fc[:, cell], theta_saturation=sat[:, cell], theta_evaporation_stress=c["theta_evaporation_stress"],
                                 diffusivity_m2_s=0.0)
            col = ss.prepare_column(lay, mask[cell:cell + 1].copy(), sc["area_m2"], backend=backend_name)
            op = pi.prepare_preferential(col, a3=sc["a3"], max_fraction_per_hour=sc["max_fraction_per_hour"], max_intake_rate_m_s=sc["max_intake_rate_m_s"],
                                         background_receiver_weights=np.array(sc["background_receiver_weights"]))
            r = pi.preferential_step(op, surface[cell:cell + 1].copy(), ss.column_state(col, water[:, cell:cell + 1].copy(), sc["t_s"]), sc["dt_s"])
            got["water"].append(r.state.water_m[:, 0])
            got["intake_m"].append(r.intake_m[:, 0])
            got["surface_water_m"].append(r.surface_water_m[0])
            got["intake_total_m"].append(r.intake_total_m[0])
            n_cols += 1
        for name, vals in got.items():
            want = expected[f"{sc['label']}_{name}"]
            have = np.column_stack(vals) if name in ("water", "intake_m") else np.array(vals)
            assert have.tobytes() == want.tobytes(), (sc["label"], name)
    assert n_cols == 336


def test_accepted_e1_texture_patch_is_exactly_the_accepted_bytes_and_engines_are_unchanged():
    import hashlib as _h
    from pathlib import Path
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
