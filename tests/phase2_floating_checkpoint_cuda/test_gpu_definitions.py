"""Actual-GPU test DEFINITIONS of the Dense CUDA floating file (WRITTEN, NOT RUN by their author: no GPU/CuPy/NVRTC/nvidia-smi was touched). Opt-in (SYRUP_RELEASE_GPU=1); Root runs them on GPU0 with `.venv-cupy`.
Every restart target is the vertical of `FloatingPondVertical.from_held_context` on a genuine H0 `/2` context exported from the ORIGINAL device vertical (or, in one case, a second genuine original owner vertical); no
original bed, owner or probe is rebuilt on the fresh-process path. Fixtures are the qualified E5 device events (`device_event`). The CPU/Numba pond on the same terrain and forcing is the independent numerical cross-check
(floating arrays within the existing 2e-12 / 1e-14 contracts, fields within rtol 1e-9 / atol 1e-12); GPU-restored vs uninterrupted-GPU continuation is bitwise. Windowed beds are refused (typed)."""
# ruff: noqa: RUF059
from __future__ import annotations

import dataclasses
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from case import snapshot

_T = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e5_gpu_defs_for_file", _T / "phase2_floating_pond" / "test_gpu_definitions.py")
e5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e5)

from fpond_builders import make_pair, ph

from maple_syrup import floating_checkpoint as fc
from maple_syrup import floating_checkpoint_cuda as fcc
from maple_syrup import floating_held_context as fh
from maple_syrup import floating_held_context_cuda as fhc
from maple_syrup import floating_pond as fp
from maple_syrup import floating_resume_cuda as frc
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import floating_soil_layers_cuda as fcu
from maple_syrup import pond_checkpoint as pc

BIG = 1 << 40
AVAIL = lambda: BIG
IDENT = {"case": "gf12", "version": 1}
RAIN = 300.0 / 3.6e6
MEM = fsl.FloatingMemoryError


@pytest.fixture(scope="module")
def cp():
    if os.environ.get("SYRUP_RELEASE_GPU") != "1":
        pytest.skip("GPU tests are opt-in (SYRUP_RELEASE_GPU=1); not selected, no GPU claim")
    backend = pytest.importorskip("maple.core.backend")
    if not backend.gpu_execution_available():
        pytest.skip("CuPy or a CUDA device is unavailable; the CUDA floating file is not exercised (no claim)")
    return backend.cupy_module()


def original(cp, tmp, **kw):
    kw = {"windowed": False, "splash": True, "rain_mm_h": 300.0, "end_s": 1000.0, **kw}
    return e5.device_event(tmp, **kw)


def held_vertical(cp, orig_vert):
    """A restart target built ONLY from the genuine H0 /2 context of the original vertical (no owner, no bed, no initial W/J/I)."""
    rec = fhc.export_held_context_cuda(orig_vert, None, available_bytes_fn=AVAIL)
    e = orig_vert.held_template()
    ctx = fh.reconstruct_held_context(rec, fh.ExpectedHeldIdentities(orig_vert.config.sha256, e.column.sha256, e.matrix_op.sha256, None), available_bytes_fn=AVAIL)
    return fp.FloatingPondVertical.from_held_context(ctx, available_bytes_fn=AVAIL), rec


def target_event(orig_ev, vert):
    """The restart event: the original event's own immutable static inputs (same process) on the new vertical."""
    from maple_syrup.pond_event import PondEvent

    return PondEvent(orig_ev.context, None, orig_ev.sediment, orig_ev.schedule, orig_ev._initial_topology, control=orig_ev.control, policy=orig_ev.policy, splash=orig_ev.splash, floating_vertical=vert)


def run(ev, st, n, rate=RAIN, dt=0.2):
    for _ in range(n):
        st = ev.step(st, rate, dt).state
    return st


def same(a, b):
    assert set(a) == set(b)
    for k in a:
        assert a[k].tobytes() == b[k].tobytes(), k


def host(a):
    from maple.core.backend import to_host

    return np.asarray(to_host(a))


def save_load(cp, tmp, orig_ev, st, orig_vert, **qkw):
    ck = tmp / "ck"
    fcc.save_floating_pond_checkpoint_cuda(ck, st, orig_ev, IDENT, **qkw)
    vert, _rec = held_vertical(cp, orig_vert)
    ev2 = target_event(orig_ev, vert)
    return ck, ev2, vert


def retamper(ck, *, arrays=None, manifest=None):
    """Mutate the companion arrays and/or the manifest and RECOMPUTE every file hash and the manifest seal (the attacker's best case)."""
    man = json.loads((ck / fcc.MANIFEST_NAME).read_text())
    if arrays:
        with np.load(ck / pc.PAYLOADS[1], allow_pickle=False) as z:
            data = {k: np.array(z[k]) for k in z.files}
        for name, fn in arrays.items():
            data[name] = fn(data[name])
        (ck / pc.PAYLOADS[1]).unlink()
        np.savez(ck / pc.PAYLOADS[1], **data)
    if manifest:
        manifest(man)
    man["files"] = {n: pc._hash_file(ck / n, 1 << 40)[0] for n in pc.PAYLOADS}
    (ck / fcc.MANIFEST_NAME).write_text(json.dumps(fc._seal(man)))


def untouched(vert):
    return not vert._registry and vert._provisional is None and vert.stats.steps == 0


# ---- the round trip and matched continuation ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("k", [0, 12])
def test_dense_roundtrip_restores_the_current_payload_and_continues_bit_exactly_and_matches_the_cpu_pond(cp, tmp_path, k):
    ev, st, vert, _owner = original(cp, tmp_path / "o")
    st = run(ev, st, k)  # k = 0: the initial state; k = 12: rain + matrix + gravity + wet + splash exchange
    ck, ev2, hvert = save_load(cp, tmp_path, ev, st, vert, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    man = json.loads((ck / fcc.MANIFEST_NAME).read_text())
    assert man["schema"] == fcc.GPU_SCHEMA and man["profile"] == fcc.GPU_PROFILE and man["array_backend"] == "cupy" and man["hydraulic_backend"] == "cuda" and man["kernel"] == fhc.kernel_descriptor()
    assert set(man) == fcc.MANIFEST_KEYS and man["floating"]["held_layouts"] == fcc._layouts_of(vert.held_template().column) and man["native_dense"]["voxel_shape"][0:2] == [3, 4]
    got = fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    assert hvert._owner is None and hvert._initial is None and len(hvert._registry) == 1  # the file's current payload is the ONLY registration; no initial W
    same(snapshot(ev2, got, hvert), snapshot(ev, st, vert))  # W, J, I, D, hydraulic, bed, phase, ledgers, counters: bytes
    live = run(ev, st, 8)
    rest = run(ev2, got, 8)
    same(snapshot(ev2, rest, hvert), snapshot(ev, live, vert))  # GPU restored vs uninterrupted GPU continuation: bitwise
    a = ev2.audit(rest)
    assert a["water"]["closes"] and a["sediment"]["closes"]
    for backend in ("numpy", "numba"):  # independent CPU/Numba cross-check on the same terrain and forcing (E5 tolerances)
        if backend == "numba":
            pytest.importorskip("numba")
        (cev, cst, cvert, *_a), _ = make_pair(tmp_path / f"cpu_{backend}", backend=backend, splash=True, rain_mm_h=300.0, end_s=1000.0)
        cst = run(cev, cst, k + 8)
        np.testing.assert_allclose(hvert.current_water(rest.hydraulic.soil_water_m), cvert.current_water(cst.hydraulic.soil_water_m), rtol=2e-12, atol=1e-14)
        pc_, pg = cvert.current_payload(cst.hydraulic.soil_water_m), hvert.current_payload(rest.hydraulic.soil_water_m)
        np.testing.assert_allclose(host(pg.history.wetting_m), pc_.history.wetting_m, rtol=2e-12, atol=1e-14)
        np.testing.assert_allclose(host(rest.hydraulic.depth_m), np.asarray(cst.hydraulic.depth_m), rtol=1e-9, atol=1e-12)
    gc.collect()


def test_a_second_genuine_original_owner_vertical_is_also_a_valid_restart_target(cp, tmp_path):
    ev, st, vert, _o = original(cp, tmp_path / "o")
    st = run(ev, st, 6)
    ck = tmp_path / "ck"
    fcc.save_floating_pond_checkpoint_cuda(ck, st, ev, IDENT)  # real default readers on both pools
    ev2, _s2, vert2, _o2 = original(cp, tmp_path / "o2")
    got = fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT)
    same(snapshot(ev2, got, vert2), snapshot(ev, st, vert))


# ---- current D != D0 (real native event) and, separately, W above the old capacity (public-factory control) ---------------------------------------------------------------------------
def test_native_event_with_a_changed_current_depth_saves_loads_on_the_current_geometry_and_continues_bit_exactly(cp, tmp_path):
    """A real 200-step storm changes the canonical depth D and reconciles columns; the file restores the payload on the CURRENT view and continues bitwise. This test does NOT demonstrate W above the old D0 capacity: the
    ordinary storm did not add enough room (the historical CPU `test_deposition_geometry.py` documents the same limit), so the current W is still legal on the held D0 and no such claim is made here."""
    z = ph.plane(base=0.12, drop=0.004)
    kw = {"surface": z, "bottom": "closed", "saturation": 0.97, "splash": True, "depth": 0.02, "max_dt_s": 0.001}
    ev, st, vert, _o = original(cp, tmp_path / "o", **kw)
    for _ in range(200):
        st = ev.step(st, RAIN, 0.001).state
    assert vert.stats.depth_changed_columns > 0 and vert.stats.reconciled_columns > 0  # real deposition/erosion changed the canonical depth
    payload = vert.current_payload(st.hydraulic.soil_water_m)
    t = vert.held_template()
    assert not np.array_equal(host(payload.depth_m), host(t.depth0_m)) and payload.column.identity_kind == "geometry_view_token"
    ck, ev2, hvert = save_load(cp, tmp_path, ev, st, vert, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    got = fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    p2 = hvert.current_payload(got.hydraulic.soil_water_m)
    assert p2.column.identity_kind == "geometry_view_token" and p2.column.parent_sha256 == t.column.sha256
    same(snapshot(ev2, got, hvert), snapshot(ev, st, vert))
    same(snapshot(ev2, run(ev2, got, 6), hvert), snapshot(ev, run(ev, st, 6), vert))  # bitwise continuation on the restored current geometry


def test_public_factory_control_water_above_the_old_d0_capacity_is_legal_only_on_the_current_depth_view(cp, tmp_path):
    """NOT a native-event-generated checkpoint: a carefully constructed VALID continued state through the qualified public factories only (`fsl.floating_geometry_view`, `fsl.floating_continued_state`,
    `fmi.floating_matrix_history` + `rebind_matrix_geometry_view`), on the held template of a genuine CUDA vertical. Last-layer water is set to 0.999 of the capacity of the CURRENT (deeper) slab; it exceeds the capacity of the
    held D0 slab, so the held column refuses it and the current-depth view accepts it. This is the exact semantics `floating_resume_cuda.prepare_from_verified_inputs` relies on (it uses these same factories)."""
    from maple.core.backend import to_device

    from maple_syrup import floating_matrix_infiltration as fmi

    ev, st, vert, _o = original(cp, tmp_path / "o")
    t = vert.held_template()
    L, N = vert.n_layers, vert.n_cells
    edges = np.asarray(vert._edges_host, dtype=np.float64)
    d0 = host(t.depth0_m)
    d_new = np.full(N, edges[-1])  # the full nominal depth H
    h_old = np.maximum(np.minimum(d0, edges[-1]) - edges[-2], 0.0)
    h_new = np.maximum(np.minimum(d_new, edges[-1]) - edges[-2], 0.0)
    active = np.asarray(vert._active_host(), dtype=bool).reshape(-1)
    assert np.any(active & (h_new > 1.01 * h_old)), "the fixture needs a deeper current slab than the held D0 slab"
    sat = host(t.column._maps["theta_saturation"])
    sat_last = (sat[-1] if sat.ndim == 2 else (np.full(N, sat[-1]) if sat.shape[0] == L else np.full(N, sat[0])))
    w = np.zeros((L, N))
    w[-1] = np.where(active, 0.999 * sat_last * h_new, 0.0)
    w_dev = to_device(np.ascontiguousarray(w), cp)
    t_s = float(vert.current_payload(st.hydraulic.soil_water_m).state.t_s)
    with pytest.raises(fsl.FloatingSoilError):  # above the held D0 capacity: refused on the held column
        fsl.floating_continued_state(t.column, w_dev, t_s)
    view = fsl.floating_geometry_view(t.column, to_device(np.ascontiguousarray(d_new), cp))
    state = fsl.floating_continued_state(view, w_dev, t_s)  # legal on the CURRENT geometry
    hist = fmi.floating_matrix_history(t.matrix_op, wetting_m=cp.zeros(N), cumulative_intake_m=cp.zeros(N))
    op, hist2 = fmi.rebind_matrix_geometry_view(t.matrix_op, view, hist)
    assert view.identity_kind == "geometry_view_token" and view.parent_sha256 == t.column.sha256 and op.column is view and hist2.operator_sha256 == op.sha256
    assert np.array_equal(host(state.water_m), w)


# ---- terminal exchange counter and quiet-reset receipt (real producers) ----------------------------------------------------------------------------------------------------------
def lake(cp, tmp):
    from maple_syrup.pond_event import PondCompletionPolicy

    z = ph.basin() - 0.08
    depth = np.maximum(0.135 - z, 0.0)
    mob = np.zeros((3, 5, 6))
    mob[1, 2, 2] = 1.0e-1
    kw = {"surface": z, "depth": depth, "mobile": mob, "closed": True, "bottom": "closed", "saturation": 0.999999, "splash": False, "rain_mm_h": 0.0, "end_s": 2.0, "max_dt_s": 1.0,
          "policy": PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)}
    return e5.device_event(tmp, windowed=False, **kw)


def test_terminal_native_call_count_and_quiet_reset_receipt_round_trip_from_the_real_producers(cp, tmp_path):
    ev, st, vert, _o = lake(cp, tmp_path / "o")
    q = ev.evolve(st, 60.0, stop_when_quiet=True)
    assert q.quiet_ready
    h = ev.settle_retaining_water(q.state)
    assert h.status == "requires_recession"
    wake = h.state
    assert wake.sediment.n_terminal_native_calls == 1 and wake.sediment.retained_quiet_reset_step == wake.counters.n_accepted_steps
    later = ev.step(wake, 0.0, 0.2).state
    for name, good in (("wake", wake), ("later", later)):
        ck = tmp_path / name
        fcc.save_floating_pond_checkpoint_cuda(ck, good, ev, IDENT)
        man = json.loads((ck / fcc.MANIFEST_NAME).read_text())
        assert man["terminal_exchange"] == {"n_terminal_native_calls": 1} and man["retained_quiet_reset"] == {"step": wake.counters.n_accepted_steps}
        hvert, _r = held_vertical(cp, vert)
        ev2 = target_event(ev, hvert)
        got = fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT)
        assert got.sediment.n_terminal_native_calls == 1 and got.sediment.retained_quiet_reset_step == wake.counters.n_accepted_steps
        same(snapshot(ev2, got, hvert), snapshot(ev, good, vert))
    # a fault-injected fresh monitor one step after the wake is refused at SAVE; a tampered terminal count with recomputed seals is refused at LOAD
    forged = dataclasses.replace(later, quiet=type(later.quiet)())
    with pytest.raises(fc.PondCheckpointError, match="quiet monitor time differs"):
        fcc.save_floating_pond_checkpoint_cuda(tmp_path / "bad", forged, ev, IDENT)
    assert not (tmp_path / "bad").exists()
    ck = tmp_path / "wake"
    retamper(ck, manifest=lambda m: m["terminal_exchange"].update(n_terminal_native_calls=0))
    hvert, _r = held_vertical(cp, vert)
    ev2 = target_event(ev, hvert)
    with pytest.raises(fc.PondCheckpointError):
        fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT)
    assert untouched(hvert)


# ---- tamper, source/kernel/backend/schema refusals, late failure and retry ------------------------------------------------------------------------------------------------------
def test_tamper_with_recomputed_hashes_is_refused_and_the_target_stays_untouched_and_retryable(cp, tmp_path):
    ev, st, vert, _o = original(cp, tmp_path / "o")
    st = run(ev, st, 6)
    good = tmp_path / "good"
    fcc.save_floating_pond_checkpoint_cuda(good, st, ev, IDENT)
    hvert, _r = held_vertical(cp, vert)
    ev2 = target_event(ev, hvert)
    with np.load(good / pc.PAYLOADS[1], allow_pickle=False) as z:  # independent host precondition: the edited W must REALLY change the aggregated derived soil sum
        w0, soil0 = np.array(z["fl_water"]), np.array(z["hyd_soil"])
    w1 = w0 * (1.0 - 2.0**-20)
    assert np.any(w0 > 0.0) and not np.array_equal(w1.sum(axis=0).reshape(soil0.shape), soil0) and np.all(w1 <= w0)
    cases = {
        "depth": {"arrays": {"fl_depth": lambda a: np.nextafter(a, np.inf)}},  # stale/tampered canonical depth: caught by the device canonical-depth check
        "held_fc": {"arrays": {"fl_held_fc": lambda a: np.nextafter(a, np.inf)}},  # held bytes differ from the REAL held template
        "water_negative": {"arrays": {"fl_water": lambda a: -np.abs(a) - 1.0}},
        "wetting_negative": {"arrays": {"fl_wetting": lambda a: -np.abs(a) - 1.0}},  # DEFINITELY invalid history: the public contract (`fmi.floating_matrix_history`) requires every J >= 0
        "water_soil_sum": {"arrays": {"fl_water": lambda a: a * (1.0 - 2.0**-20)}},  # a still-valid (smaller) W whose layer sum REALLY differs from the stored derived soil sum (precondition asserted below)
        "kernel": {"manifest": lambda m: m["kernel"].update(fmc_source_sha256="0" * 64)},
        "source": {"manifest": lambda m: m["source"].update(maple_package_sha256="0" * 64)},
        "schema_cpu": {"manifest": lambda m: m.update(schema=fc.SCHEMA_V3, profile=fc.PROFILE)},
        "schema_windowed": {"manifest": lambda m: m.update(schema=fcc.GPU_WINDOWED_SCHEMA)},
        "backend": {"manifest": lambda m: m.update(array_backend="numpy")},
        "layouts": {"manifest": lambda m: m["floating"]["held_layouts"].update(theta_saturation="scalar")},
        "generation": {"manifest": lambda m: m["floating"].update(generation=m["floating"]["generation"] + 1)},
    }
    for name, kw in cases.items():
        bad = tmp_path / f"bad_{name}"
        shutil.copytree(good, bad)
        retamper(bad, **kw)
        with pytest.raises((fc.PondCheckpointError, fcc.FloatingCheckpointCudaError, frc.FloatingResumeCudaError, fp.FloatingPondError, fsl.FloatingSoilError)):  # the exact named typed refusals of the file/prepare/validator layers
            fcc.load_floating_pond_checkpoint_cuda(bad, ev2, IDENT)
        assert untouched(hvert), name
    flip = tmp_path / "flip"
    shutil.copytree(good, flip)
    with (flip / pc.PAYLOADS[1]).open("r+b") as h:  # a plain payload bit flip WITHOUT recomputing the hash
        h.seek(200)
        b = h.read(1)
        h.seek(200)
        h.write(bytes([b[0] ^ 1]))
    with pytest.raises(fc.PondCheckpointError, match="hash mismatch"):
        fcc.load_floating_pond_checkpoint_cuda(flip, ev2, IDENT)
    assert untouched(hvert)
    got = fcc.load_floating_pond_checkpoint_cuda(good, ev2, IDENT)  # the SAME vertical/event are retryable after every refusal
    same(snapshot(ev2, got, hvert), snapshot(ev, st, vert))


def test_an_edited_but_valid_wetting_history_restores_as_another_valid_history_no_authenticity_is_claimed(cp, tmp_path):
    """SELF-CONSISTENCY control only (not a native-produced storm, not an accepted altered real run). The public history contract (`fmi.floating_matrix_history`) admits ANY finite J, I >= 0 and cannot verify a genuine
    run boundary; file hashes are INTEGRITY checks of the bytes written (and the static binding/source/held identities bind the file to the case and code), NOT authenticated physical provenance. A file whose author edited J
    and recomputed every hash therefore supplies another VALID history and restores it unchanged: no J <= I rule, reset or scientific constraint is invented here."""
    ev, st, vert, _o = original(cp, tmp_path / "o")
    st = run(ev, st, 6)
    good = tmp_path / "good"
    fcc.save_floating_pond_checkpoint_cuda(good, st, ev, IDENT)
    edited = tmp_path / "edited"
    shutil.copytree(good, edited)
    retamper(edited, arrays={"fl_wetting": lambda a: a + 1.0})
    hvert, _r = held_vertical(cp, vert)
    ev2 = target_event(ev, hvert)
    got = fcc.load_floating_pond_checkpoint_cuda(edited, ev2, IDENT)
    j = host(hvert.current_payload(got.hydraulic.soil_water_m).history.wetting_m)
    assert j.tobytes() == (host(vert.current_payload(st.hydraulic.soil_water_m).history.wetting_m) + 1.0).tobytes()


def test_a_late_audit_failure_inside_the_provisional_scope_leaves_the_vertical_untouched_and_retryable(cp, tmp_path, monkeypatch):
    ev, st, vert, _o = original(cp, tmp_path / "o")
    st = run(ev, st, 6)
    ck, ev2, hvert = save_load(cp, tmp_path, ev, st, vert)
    real = ev2.audit

    def boom(state):
        raise fp.FloatingPondError("injected late audit failure")

    monkeypatch.setattr(ev2, "audit", boom)
    with pytest.raises(fc.PondCheckpointError, match="independent validation"):
        fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT)
    assert untouched(hvert)
    monkeypatch.setattr(ev2, "audit", real)
    got = fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT)  # retry on the same objects succeeds
    same(snapshot(ev2, got, hvert), snapshot(ev, st, vert))


def test_the_prepared_record_and_the_inputs_are_one_shot_and_a_foreign_target_is_refused(cp, tmp_path, monkeypatch):
    ev, st, vert, _o = original(cp, tmp_path / "o")
    st = run(ev, st, 4)
    ck, ev2, hvert = save_load(cp, tmp_path, ev, st, vert)
    captured = {}
    real = frc.prepare_from_verified_inputs

    def spy(event, inputs, *, quota):
        out = real(event, inputs, quota=quota)
        captured["inputs"], captured["prepared"] = inputs, out[0]
        return out

    monkeypatch.setattr(frc, "prepare_from_verified_inputs", spy)
    fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT)
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="already consumed"):
        fcc.require_inputs(captured["inputs"])
    with pytest.raises(frc.FloatingResumeCudaError, match="already adopted"):
        frc.prepared_parts(captured["prepared"], hvert)
    other, _r = held_vertical(cp, vert)
    with pytest.raises(frc.FloatingResumeCudaError):
        frc.prepared_parts(captured["prepared"], other)


def test_a_storage_mismatch_between_file_and_event_is_refused_typed_in_both_directions(cp, tmp_path):
    """The genuine invalid storage binding control (the former Windowed refusal became the positive `/8` path): a Dense `/7` file into a Windowed event and a Windowed `/8` file into a Dense event are refused, no cross-storage conversion."""
    dev_, dst, dvert, _o = original(cp, tmp_path / "d")
    wev, wst, wvert, _o2 = original(cp, tmp_path / "w", windowed=True)
    fcc.save_floating_pond_checkpoint_cuda(tmp_path / "dense_ck", run(dev_, dst, 3), dev_, IDENT)
    fcc.save_floating_pond_checkpoint_cuda(tmp_path / "win_ck", run(wev, wst, 3), wev, IDENT)
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="no cross-storage conversion"):
        fcc.load_floating_pond_checkpoint_cuda(tmp_path / "dense_ck", wev, IDENT)
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="no cross-storage conversion"):
        fcc.load_floating_pond_checkpoint_cuda(tmp_path / "win_ck", dev_, IDENT)
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="differs from the event's configured storage"):
        fcc.save_floating_pond_checkpoint_cuda(tmp_path / "x", dst, wev, IDENT)  # a Dense state on a Windowed-configured event
    assert not (tmp_path / "x").exists()


# ---- native Windowed `/8` ------------------------------------------------------------------------------------------------------------------------------------------------------------
def deep_dir(ck):
    return Path(json.loads((ck / fcc.MANIFEST_NAME).read_text())["deep_storage"]["store_dir"])


@pytest.mark.parametrize("k", [0, 12])
def test_windowed_roundtrip_restores_the_current_payload_continues_bit_exactly_and_matches_the_dense_gpu_control(cp, tmp_path, k):
    wev, wst, wvert, _o = original(cp, tmp_path / "w", windowed=True)
    wst = run(wev, wst, k)
    ck, wev2, hvert = save_load(cp, tmp_path, wev, wst, wvert, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    man = json.loads((ck / fcc.MANIFEST_NAME).read_text())
    assert man["schema"] == fcc.GPU_WINDOWED_SCHEMA and set(man) == fcc.WINDOWED_MANIFEST_KEYS and man["bed_storage"] == "windowed" and man["native_dense"] is None
    assert man["deep_storage"]["store_dir"] == str(deep_dir(ck)) and Path(man["deep_storage"]["store_dir"]).is_dir()  # the deep store is REFERENCED (an external directory), never copied into the checkpoint
    assert {p.name for p in ck.iterdir()} == {fcc.MANIFEST_NAME, *pc.PAYLOADS}
    got = fcc.load_floating_pond_checkpoint_cuda(ck, wev2, IDENT, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    assert hvert._owner is None and hvert._initial is None and len(hvert._registry) == 1
    same(snapshot(wev2, got, hvert), snapshot(wev, wst, wvert))
    live, rest = run(wev, wst, 8), run(wev2, got, 8)
    same(snapshot(wev2, rest, hvert), snapshot(wev, live, wvert))  # restored vs uninterrupted Windowed GPU continuation: bitwise
    a = wev2.audit(rest)
    assert a["water"]["closes"] and a["sediment"]["closes"]
    dev_, dst, dvert, _o2 = original(cp, tmp_path / "d")  # the matched DENSE GPU control on the same terrain/forcing (E5 tolerances)
    dst = run(dev_, dst, k + 8)
    np.testing.assert_allclose(hvert.current_water(rest.hydraulic.soil_water_m), dvert.current_water(dst.hydraulic.soil_water_m), rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(host(rest.hydraulic.depth_m), host(dst.hydraulic.depth_m), rtol=1e-9, atol=1e-12)


def boundary_windowed_event(cp, tmp):
    """The qualified E5 native Windowed device event with `deep_publication="boundary"` and a large high-water mark (so deposition leaves a PENDING deep journal, as the CPU `test_windowed_pending_journal_is_refused`)."""
    import fpond_builders as fb

    from maple_syrup import native_floating_soil as nfs
    from maple_syrup.pond_event import PondEvent
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters
    from maple_syrup.splash import splash_parameters

    wh = fb._windowed_helpers()
    z = ph.plane(base=0.12, drop=0.004)
    dense, ctx = wh.dense_deep_bed(z, depth=0.003)
    topo = ph.make_topology(dense, ph.east_outlets(*z.shape), drop=0.02)
    bed, dctx = wh.windowed_counterpart(dense, ctx, tmp / "store", xp=cp)
    owner = nfs.initialize_native_floating_soil(fb.native_bed(bed, dctx), fb.config(backend="cuda", bottom="open"), initial_relative_saturation=0.5)
    vert = fp.FloatingPondVertical(owner, available_bytes_fn=fcu.device_reader(cp))
    be = type("B", (), {"hydraulics": "cuda", "xp": cp})()
    ctl = wh.control_for(be, windowed=True, max_dt_s=0.2, deep_publication="boundary", high_water=500_000)
    ev = PondEvent(dctx, None, plot1_sediment_parameters(xp=cp), constant_rainfall(0.0, 100.0, 100.0), topo, control=ctl, splash=splash_parameters(np.linspace(0.4, 3.0, fb.NC), xp=cp), floating_vertical=vert)
    return ev, ev.initial_state(bed, vert.initial_soil_water()), vert


def test_a_pending_deep_journal_is_refused_at_save_before_any_file_or_admission(cp, tmp_path):
    from maple_syrup.sediment_bed import pending_journal_records

    wev, wst, wvert = boundary_windowed_event(cp, tmp_path / "w")
    for _ in range(60):
        wst = wev.step(wst, RAIN, 0.2).state
    assert pending_journal_records(wst.bed) > 0, "the fixture must leave a pending deep journal (adjust on hardware if not)"
    reads = []
    with pytest.raises(fc.PondCheckpointError, match="pending"):
        fcc.save_floating_pond_checkpoint_cuda(tmp_path / "ck", wst, wev, IDENT, available_bytes_fn=lambda: reads.append(1) or BIG, host_available_bytes_fn=lambda: reads.append(2) or BIG)
    assert not (tmp_path / "ck").exists() and list(tmp_path.glob(".ck*")) == [] and reads == []


def test_a_corrupted_prefix_with_recomputed_hashes_is_refused_by_the_native_host_load_before_any_device_placement(cp, tmp_path, monkeypatch):
    from maple.core import backend as mb

    wev, wst, wvert, _o = original(cp, tmp_path / "w", windowed=True)
    wst = run(wev, wst, 4)
    ck, wev2, hvert = save_load(cp, tmp_path, wev, wst, wvert)
    retamper_native(ck, "prefix_packet", lambda a: a * 1.5 + 1.0)
    placed = []
    real = mb.to_device_tree
    monkeypatch.setattr(mb, "to_device_tree", lambda *a, **k: placed.append(1) or real(*a, **k))
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="windowed snapshot cannot be loaded"):
        fcc.load_floating_pond_checkpoint_cuda(ck, wev2, IDENT)
    assert placed == [] and untouched(hvert)  # the native HOST load refused it; nothing was placed on the device


def retamper_native(ck, member, fn):
    """Change ONE member of the native windowed snapshot and RECOMPUTE the file hash and the manifest seal."""
    with np.load(ck / pc.PAYLOADS[0], allow_pickle=False) as z:
        data = {k: np.array(z[k]) for k in z.files}
    data[member] = fn(data[member])
    (ck / pc.PAYLOADS[0]).unlink()
    np.savez(ck / pc.PAYLOADS[0], **data)
    retamper(ck)


def test_a_missing_or_wrong_external_deep_store_and_strict_deep_record_edits_are_refused_and_the_target_stays_retryable(cp, tmp_path):
    wev, wst, wvert, _o = original(cp, tmp_path / "w", windowed=True)
    wst = run(wev, wst, 6)
    good = tmp_path / "good"
    fcc.save_floating_pond_checkpoint_cuda(good, wst, wev, IDENT)
    hvert, _r = held_vertical(cp, wvert)
    ev2 = target_event(wev, hvert)
    store = deep_dir(good)
    moved = store.with_name(store.name + "_moved")
    store.rename(moved)  # the EXTERNAL deep store is missing
    try:
        with pytest.raises(fcc.FloatingCheckpointCudaError, match="windowed snapshot cannot be loaded"):
            fcc.load_floating_pond_checkpoint_cuda(good, ev2, IDENT)
        assert untouched(hvert)
    finally:
        moved.rename(store)
    edits = {
        "generation": lambda m: m["deep_storage"].update(generation=m["deep_storage"]["generation"] + 1),
        "physical_digest": lambda m: m["deep_storage"].update(physical_manifest_digest="0" * 64),
        "store_dir": lambda m: m["deep_storage"].update(store_dir=str(tmp_path / "elsewhere")),
        "class_ids": lambda m: m["deep_storage"].update(class_ids=list(reversed(m["deep_storage"]["class_ids"]))),
        "bed_storage": lambda m: m.update(bed_storage="dense"),
        "kernel": lambda m: m["kernel"].update(fpc_source_sha256="0" * 64),
        "source": lambda m: m["source"].update(maple_package_sha256="0" * 64),
        "dense_schema": lambda m: m.update(schema=fcc.GPU_SCHEMA),
        "unknown_key": lambda m: m.update(extra="x"),
    }
    for name, fn in edits.items():
        bad = tmp_path / f"bad_{name}"
        shutil.copytree(good, bad)
        retamper(bad, manifest=fn)
        with pytest.raises((fc.PondCheckpointError, fcc.FloatingCheckpointCudaError, frc.FloatingResumeCudaError)):
            fcc.load_floating_pond_checkpoint_cuda(bad, ev2, IDENT)
        assert untouched(hvert), name
    got = fcc.load_floating_pond_checkpoint_cuda(good, ev2, IDENT)  # the SAME objects restore after every refusal
    same(snapshot(ev2, got, hvert), snapshot(wev, wst, wvert))


def test_windowed_two_pool_admission_zero_need_minus_one_exact_and_reserve_once(cp, tmp_path, monkeypatch):
    wev, wst, wvert, _o = original(cp, tmp_path / "w", windowed=True)
    wst = run(wev, wst, 4)
    calls = _record_admissions(monkeypatch)
    ck = tmp_path / "ck"
    fcc.save_floating_pond_checkpoint_cuda(ck, wst, wev, IDENT, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    d_save, h_save, _w = calls[-1]
    deep = json.loads((ck / fcc.MANIFEST_NAME).read_text())["deep_storage"]
    assert h_save >= fc.deep_cache_bound_bytes(deep, wev.nc)  # the host model carries the deep-store MAPPING-CACHE bound
    for kw, label in (({"available_bytes_fn": lambda: d_save - 1, "host_available_bytes_fn": AVAIL}, "device need-1"), ({"available_bytes_fn": AVAIL, "host_available_bytes_fn": lambda: h_save + fcu.PY_ALLOWANCE_BYTES - 1}, "host need-1"),
                      ({"available_bytes_fn": lambda: d_save, "host_available_bytes_fn": AVAIL, "reserve_bytes": 1}, "device reserve once"), ({"available_bytes_fn": lambda: 0, "host_available_bytes_fn": AVAIL}, "device zero")):
        with pytest.raises(MEM):
            fcc.save_floating_pond_checkpoint_cuda(tmp_path / f"r_{label.replace(' ', '_')}", wst, wev, IDENT, **kw)
        assert not (tmp_path / f"r_{label.replace(' ', '_')}").exists(), label
    fcc.save_floating_pond_checkpoint_cuda(tmp_path / "exact", wst, wev, IDENT, available_bytes_fn=lambda: d_save, host_available_bytes_fn=lambda: h_save + fcu.PY_ALLOWANCE_BYTES)
    hvert, _r = held_vertical(cp, wvert)
    ev2 = target_event(wev, hvert)
    calls.clear()
    fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    stage1 = calls[1]
    assert stage1[1] >= fc.deep_cache_bound_bytes(deep, wev.nc)
    for kw, label in (({"available_bytes_fn": lambda: stage1[0] - 1, "host_available_bytes_fn": AVAIL}, "device"), ({"available_bytes_fn": AVAIL, "host_available_bytes_fn": lambda: stage1[1] + fcu.PY_ALLOWANCE_BYTES - 1}, "host")):
        hv, _ = held_vertical(cp, wvert)
        with pytest.raises(MEM):
            fcc.load_floating_pond_checkpoint_cuda(ck, target_event(wev, hv), IDENT, **kw)
        assert untouched(hv), label


def test_same_gpu_fresh_process_windowed_restore_from_genuine_h0_parts(cp, tmp_path):
    import static_case as sc

    kw = {"surface": sc.terrain(), "depth": 0.004, "bottom": "closed", "saturation": 0.97, "splash": True, "rain_mm_h": 300.0, "end_s": 1000.0, "max_dt_s": 0.2, "windowed": True}
    ev, st, vert, _o = original(cp, tmp_path / "o", **kw)
    st = run(ev, st, 10)
    rec = fhc.export_held_context_cuda(vert, None, available_bytes_fn=AVAIL)
    (tmp_path / "h0_meta.json").write_text(json.dumps(rec.meta()))
    names = {}
    for k, a in rec.arrays().items():
        (tmp_path / f"h0_{k}.bin").write_bytes(a.tobytes())
        names[k] = [str(a.dtype), list(a.shape)]
    (tmp_path / "h0_arrays.json").write_text(json.dumps(names))
    (tmp_path / "case_identity.json").write_text(json.dumps(IDENT))
    fcc.save_floating_pond_checkpoint_cuda(tmp_path / "ck", st, ev, IDENT)
    live = run(ev, st, 8)
    np.savez(tmp_path / "live_final.npz", **snapshot(ev, live, vert))
    worker = Path(__file__).resolve().parent / "gpu_fresh_worker.py"
    proc = subprocess.run([sys.executable, str(worker), str(tmp_path), "8", "windowed"], capture_output=True, text=True, timeout=1800, check=False)
    assert proc.returncode == 0, proc.stderr[-4000:]
    with np.load(tmp_path / "live_final.npz") as a, np.load(tmp_path / "restored_final.npz") as b:
        same({k: a[k] for k in a.files}, {k: b[k] for k in b.files})


def test_moderate_windowed_n50003_native_residency_is_exactly_window_over_full_v_and_the_true_peak_stays_inside_the_model_with_the_cache(cp, tmp_path, monkeypatch, record_property):
    """A genuine native Windowed bed (the moderate Dense fixture `moderate_bed.build` + MAPLE's `windowed_counterpart` with a NEW external deep store). The native window is asserted to be EXACTLY W / full-V of the logical cells x voxels x classes
    bed (3/60 here, with no full-V member and an external, uncopied deep store; the TOTAL checkpoint also holds the dense Pond arrays and is NOT claimed smaller than the logical bed); the observer is the accepted one (pool sampled at every allocator call + measured rounding, no slack); the declared model includes the deep-store mapping-cache bound (a HOST term; the device peak is compared with the
    device model). A model underestimate must be reported, not absorbed. Fixture arithmetic is unrun and may need hardware adjustment."""
    import fpond_builders as fb
    import moderate_bed as mbd

    from maple_syrup import native_floating_soil as nfs
    from maple_syrup.pond_event import PondEvent
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters
    from maple_syrup.splash import splash_parameters

    wh = fb._windowed_helpers()
    dense, ctx, topo, _z = mbd.build(1, 50003, 60)
    wbed, wctx = wh.windowed_counterpart(dense, ctx, tmp_path / "store", xp=cp)
    cfg = fb.config(backend="cuda", bottom="open")
    owner = nfs.initialize_native_floating_soil(fb.native_bed(wbed, wctx), cfg, initial_relative_saturation=0.5)
    vert = fp.FloatingPondVertical(owner, available_bytes_fn=fcu.device_reader(cp))
    be = type("B", (), {"hydraulics": "cuda", "xp": cp})()
    ctl = wh.control_for(be, windowed=True, max_dt_s=0.2)
    ev = PondEvent(wctx, None, plot1_sediment_parameters(xp=cp), constant_rainfall(0.0, 100.0, 100.0), topo, control=ctl, splash=splash_parameters(np.linspace(0.4, 3.0, fb.NC), xp=cp), floating_vertical=vert)
    st = run(ev, ev.initial_state(wbed, vert.initial_soil_water()), 2, rate=100.0 / 3.6e6)
    ck = tmp_path / "ck"
    fcc.save_floating_pond_checkpoint_cuda(ck, st, ev, IDENT)
    man = json.loads((ck / fcc.MANIFEST_NAME).read_text())
    components = mbd.residency_components(ck, man, 1, 50003, ev.nc)
    print("WINDOWED_RESIDENCY_COMPONENTS", json.dumps(components, sort_keys=True))  # visible with `pytest -s` for Root's acceptance log (counted array traffic is not involved: headers only)
    record_property("residency_components", json.dumps(components, sort_keys=True))
    rec = fhc.export_held_context_cuda(vert, None, available_bytes_fn=AVAIL)
    e = vert.held_template()
    held_ctx = fh.reconstruct_held_context(rec, fh.ExpectedHeldIdentities(vert.config.sha256, e.column.sha256, e.matrix_op.sha256, None), available_bytes_fn=AVAIL)
    pool = cp.get_default_memory_pool()
    admitted = []
    real_admit = fcc.admit
    monkeypatch.setattr(fcc, "admit", lambda c, d, dv, h, q, w: admitted.append((int(dv), int(h))) or real_admit(c, d, dv, h, q, w))

    class Peak:
        def __init__(self):
            self.base, self.peak, self.rounding = int(pool.used_bytes()), 0, 0

        def __call__(self, size):
            ptr = pool.malloc(size)
            self.peak = max(self.peak, int(pool.used_bytes()) - self.base)
            self.rounding += int(ptr.mem.size) - int(size)
            return ptr

    plateau = []
    for _ in range(3):
        hvert = fp.FloatingPondVertical.from_held_context(held_ctx, available_bytes_fn=AVAIL)
        ev2 = target_event(ev, hvert)
        gc.collect()
        cp.cuda.Device().synchronize()
        admitted.clear()
        hook = Peak()
        with cp.cuda.using_allocator(hook):
            got = fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT)
        d_model = max(a[0] for a in admitted)
        h_model = max(a[1] for a in admitted)
        assert h_model >= fc.deep_cache_bound_bytes(man["deep_storage"], ev.nc)
        assert hook.peak <= d_model + hook.rounding, (hook.peak, d_model, hook.rounding)  # TRUE device peak vs the declared device model + the measured rounding
        del got, ev2, hvert
        gc.collect()
        cp.cuda.Device().synchronize()
        plateau.append(int(pool.used_bytes()))
    assert max(plateau) - min(plateau) <= hook.rounding, plateau


# ---- two pools, device gate first, zero / need-1 / exact / reserve once -------------------------------------------------------------------------------------------------------------
def _record_admissions(monkeypatch):
    calls = []
    real = fcc.admit

    def spy(cp_, dev, d, h, q, what):
        calls.append((int(d), int(h), what))
        return real(cp_, dev, d, h, q, what)

    monkeypatch.setattr(fcc, "admit", spy)
    return calls


def test_two_pool_admission_zero_need_minus_one_exact_and_reserve_once_for_save_and_load(cp, tmp_path, monkeypatch):
    ev, st, vert, _o = original(cp, tmp_path / "o")
    st = run(ev, st, 4)
    calls = _record_admissions(monkeypatch)
    ck = tmp_path / "ck"
    fcc.save_floating_pond_checkpoint_cuda(ck, st, ev, IDENT, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    d_save, h_save, _w = calls[-1]
    pool = cp.get_default_memory_pool()
    allocs = []

    class Hook:
        def malloc(self, size, stream=None):
            allocs.append(int(size))
            return pool.malloc(size)

    for kw, label in (({"available_bytes_fn": lambda: 0, "host_available_bytes_fn": AVAIL}, "device zero"), ({"available_bytes_fn": AVAIL, "host_available_bytes_fn": lambda: 0}, "host zero"),
                      ({"available_bytes_fn": lambda: d_save - 1, "host_available_bytes_fn": AVAIL, "reserve_bytes": 0}, "device need-1"),
                      ({"available_bytes_fn": AVAIL, "host_available_bytes_fn": lambda: h_save + fcu.PY_ALLOWANCE_BYTES - 1}, "host need-1"),
                      ({"available_bytes_fn": lambda: d_save, "host_available_bytes_fn": AVAIL, "reserve_bytes": 1}, "device reserve counted once"),
                      ({"available_bytes_fn": AVAIL, "host_available_bytes_fn": AVAIL, "memory_budget_bytes": 0}, "zero device budget"), ({"available_bytes_fn": AVAIL, "host_available_bytes_fn": AVAIL, "host_memory_budget_bytes": 0}, "zero host budget")):
        allocs.clear()
        target = tmp_path / f"refused_{label.replace(' ', '_')}"
        with cp.cuda.using_allocator(Hook().malloc), pytest.raises(MEM):
            fcc.save_floating_pond_checkpoint_cuda(target, st, ev, IDENT, **kw)
        assert allocs == [] and not target.exists() and list(tmp_path.glob(f".{target.name}*")) == [], label  # refused BEFORE any owned allocation or IO
    exact = tmp_path / "exact"
    fcc.save_floating_pond_checkpoint_cuda(exact, st, ev, IDENT, available_bytes_fn=lambda: d_save, host_available_bytes_fn=lambda: h_save + fcu.PY_ALLOWANCE_BYTES)  # exactly the need is accepted
    # load: the LAST admission of a successful load is the composed stage-1 model; the loader's stage 0 is the manifest read
    hvert, _r = held_vertical(cp, vert)
    ev2 = target_event(ev, hvert)
    calls.clear()
    fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT, available_bytes_fn=AVAIL, host_available_bytes_fn=AVAIL)
    stage0, stage1, prepare = calls[0], calls[1], calls[2]
    assert stage0[0] == 0 and "manifest" in stage0[2] and "load" in stage1[2] and "manifest" not in stage1[2] and "preparation" in prepare[2]
    for kw, label in (({"available_bytes_fn": lambda: stage1[0] - 1, "host_available_bytes_fn": AVAIL}, "device need-1"), ({"available_bytes_fn": AVAIL, "host_available_bytes_fn": lambda: stage1[1] + fcu.PY_ALLOWANCE_BYTES - 1}, "host need-1"),
                      ({"available_bytes_fn": lambda: 0, "host_available_bytes_fn": AVAIL}, "device zero"), ({"available_bytes_fn": AVAIL, "host_available_bytes_fn": lambda: 0}, "host zero")):
        hv, _ = held_vertical(cp, vert)
        e3 = target_event(ev, hv)
        allocs.clear()
        with cp.cuda.using_allocator(Hook().malloc), pytest.raises(MEM):
            fcc.load_floating_pond_checkpoint_cuda(ck, e3, IDENT, **kw)
        assert untouched(hv) and not [a for a in allocs if a > fcu.HASH_CHUNK_BYTES], label  # nothing large allocated before the typed refusal
    hv, _ = held_vertical(cp, vert)
    fcc.load_floating_pond_checkpoint_cuda(ck, target_event(ev, hv), IDENT, available_bytes_fn=lambda: max(c[0] for c in calls), host_available_bytes_fn=lambda: max(c[1] for c in calls) + fcu.PY_ALLOWANCE_BYTES)


def test_the_current_device_gate_precedes_every_reader_and_path_check(cp, tmp_path, monkeypatch):
    ev, st, vert, _o = original(cp, tmp_path / "o")
    reads = []

    def reader():
        reads.append(1)
        return BIG

    real = fcu.current_device
    monkeypatch.setattr(fcu, "current_device", lambda c: real(c) + 1)  # ORDERING control with the real gate: the current device is not the vertical's device
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="CUDA device gate refused"):
        fcc.save_floating_pond_checkpoint_cuda(tmp_path / "exists_not", st, ev, IDENT, available_bytes_fn=reader, host_available_bytes_fn=reader)
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="CUDA device gate refused"):
        fcc.load_floating_pond_checkpoint_cuda(tmp_path / "missing", ev, IDENT, available_bytes_fn=reader, host_available_bytes_fn=reader)
    assert reads == [] and not (tmp_path / "exists_not").exists()


# ---- counted transfers: bounded cold chunks and attributed traffic -----------------------------------------------------------------------------------------------------------------
def test_floating_arrays_move_in_bounded_counted_chunks_and_the_traffic_is_attributed(cp, tmp_path, monkeypatch):
    """Each helper call is recorded ONCE (the wrapper calls the real helper a single time) and MAPLE's own counters are the independent control. Chunk sizes are attributed to the helper that issued them through a flag."""
    from maple.core import backend as mb

    ev, st, vert, _o = original(cp, tmp_path / "o")
    st = run(ev, st, 6)
    L, N = vert.n_layers, vert.n_cells
    flag = {"in": None}
    helper_calls = {"d2h": [], "h2d": [], "eq": []}
    chunks = {"d2h": [], "h2d": [], "eq": []}
    real = {"d2h": fcc.d2h, "h2d": fcc.h2d, "eq": fcc.stream_equal}

    def wrap(kind, fn, size_of):
        def inner(*a):
            helper_calls[kind].append(size_of(a))
            flag["in"] = kind
            try:
                return fn(*a)
            finally:
                flag["in"] = None
        return inner

    monkeypatch.setattr(fcc, "d2h", wrap("d2h", real["d2h"], lambda a: int(a[0].nbytes)))
    monkeypatch.setattr(fcc, "h2d", wrap("h2d", real["h2d"], lambda a: int(a[1].nbytes)))
    monkeypatch.setattr(fcc, "stream_equal", wrap("eq", real["eq"], lambda a: int(a[1].nbytes)))
    real_to_host, real_to_device = mb.to_host, mb.to_device

    def to_host(x, *a, **k):
        out = real_to_host(x, *a, **k)
        if flag["in"] in ("d2h", "eq"):
            chunks[flag["in"]].append(int(np.asarray(out).nbytes))
        return out

    def to_device(x, xp):
        if flag["in"] == "h2d":
            chunks["h2d"].append(int(getattr(x, "nbytes", 0)))
        return real_to_device(x, xp)

    monkeypatch.setattr(mb, "to_host", to_host)
    monkeypatch.setattr(mb, "to_device", to_device)
    mb.reset_transfer_counters()
    ck = tmp_path / "ck"
    fcc.save_floating_pond_checkpoint_cuda(ck, st, ev, IDENT)
    c_save = mb.read_transfer_counters()
    floating = (L * N + 3 * N) * 8  # W, J, I, D
    t = vert.held_template()
    held = int(t.column._maps["theta_field_capacity"].nbytes) + int(t.column._maps["theta_saturation"].nbytes) + int(t.depth0_m.nbytes) + (0 if t.ks_m_s is None else int(t.ks_m_s.nbytes))
    n_active = int(np.count_nonzero(vert._active_host()))
    assert sum(helper_calls["d2h"]) >= floating + held and sum(chunks["d2h"]) == sum(helper_calls["d2h"]) and max(chunks["d2h"]) <= fcu.HASH_CHUNK_BYTES  # bounded chunks, exact attribution
    assert c_save.device_to_host_bytes >= sum(chunks["d2h"]) and c_save.host_to_device_bytes <= 8 * n_active + 4096  # H2D only the active index list of the canonical-depth check
    hvert, _r = held_vertical(cp, vert)
    ev2 = target_event(ev, hvert)
    for kind, calls_of in helper_calls.items():
        calls_of.clear()
        chunks[kind].clear()
    mb.reset_transfer_counters()
    fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT)
    c_load = mb.read_transfer_counters()
    assert sorted(helper_calls["h2d"]) == sorted([L * N * 8, N * 8, N * 8, N * 8])  # exactly the four companion uploads W, D, J, I, one helper call each
    assert sum(chunks["h2d"]) == sum(helper_calls["h2d"]) and max(chunks["h2d"]) <= fcu.HASH_CHUNK_BYTES
    n_held = 3 + (0 if t.ks_m_s is None else 1)
    assert len(helper_calls["eq"]) == 2 * n_held + 2 * 5  # held compare in the loader and in the preparation + two prepared_parts verifications (scope entry, adoption) x five arrays
    assert max(chunks["eq"]) <= fcu.HASH_CHUNK_BYTES
    assert c_load.host_to_device_bytes >= sum(helper_calls["h2d"]) and c_load.device_to_host_bytes >= sum(chunks["eq"])  # MAPLE's counters are the independent control (they also carry the native bed and Pond arrays)


# ---- same-GPU fresh process ---------------------------------------------------------------------------------------------------------------------------------------------------
def test_same_gpu_fresh_process_restores_from_genuine_h0_parts_without_the_original_bed_owner_or_probe(cp, tmp_path):
    import static_case as sc

    z = sc.terrain()
    kw = {"surface": z, "depth": 0.004, "bottom": "closed", "saturation": 0.97, "splash": True, "rain_mm_h": 300.0, "end_s": 1000.0, "max_dt_s": 0.2}
    ev, st, vert, _o = original(cp, tmp_path / "o", **kw)
    st = run(ev, st, 10)
    rec = fhc.export_held_context_cuda(vert, None, available_bytes_fn=AVAIL)
    (tmp_path / "h0_meta.json").write_text(json.dumps(rec.meta()))
    names = {}
    for k, a in rec.arrays().items():
        (tmp_path / f"h0_{k}.bin").write_bytes(a.tobytes())
        names[k] = [str(a.dtype), list(a.shape)]
    (tmp_path / "h0_arrays.json").write_text(json.dumps(names))
    (tmp_path / "case_identity.json").write_text(json.dumps(IDENT))
    fcc.save_floating_pond_checkpoint_cuda(tmp_path / "ck", st, ev, IDENT)
    live = run(ev, st, 8)
    np.savez(tmp_path / "live_final.npz", **snapshot(ev, live, vert))
    worker = Path(__file__).resolve().parent / "gpu_fresh_worker.py"
    proc = subprocess.run([sys.executable, str(worker), str(tmp_path), "8"], capture_output=True, text=True, timeout=1800, check=False)
    assert proc.returncode == 0, proc.stderr[-4000:]
    with np.load(tmp_path / "live_final.npz") as a, np.load(tmp_path / "restored_final.npz") as b:
        same({k: a[k] for k in a.files}, {k: b[k] for k in b.files})  # the fresh-process continuation is bitwise the uninterrupted one


# ---- moderate N = 50003: true allocator peak vs the declared device model and the released plateau ----------------------------------------------------------------------------
def test_moderate_n50003_true_allocator_peak_stays_inside_the_declared_device_model_and_the_plateau_is_released(cp, tmp_path, monkeypatch):
    """A genuine one-row native Dense bed of N = 50003 columns with a REDUCED voxel depth (NV = 60 of the qualified 0.001 m voxels, about 144 MB of float64 voxel mass), built DIRECTLY from the public native MAPLE constructors by
    `moderate_bed.build` (static primitive geometry/grains, `deposit_surface_mixture_batch` at the desired terrain; no probe, no `dense_deep_bed`). The observer is the accepted one (`test_moderate_n50003_true_allocator_peak_and_released_plateau`): `pool.used_bytes()` is SAMPLED at EVERY allocator call
    relative to the pool before the load, and the ACTUAL per-call rounding (`ptr.mem.size - size`) is measured. Model scope: the declared device model of the LOAD (the admitted stage-1/prepare requirements); the
    already-live caller inputs (original event/owner/state/bed, the H0-built target vertical and its context) exist before the baseline and are EXCLUDED. No arbitrary slack: a peak above model + measured rounding is a
    model underestimate and fails."""
    import fpond_builders as fb
    import moderate_bed as mbd
    from maple.core.backend import to_device, to_device_tree

    from maple_syrup import native_floating_soil as nfs
    from maple_syrup.pond_event import PondEvent
    from maple_syrup.rainfall import constant_rainfall
    from maple_syrup.sediment_physics import plot1_sediment_parameters
    from maple_syrup.splash import splash_parameters

    wh = fb._windowed_helpers()
    ny, nx = 1, 50003
    dense, ctx, topo, _z = mbd.build(ny, nx, 60)  # genuine native bed from the public constructors at the desired 0.05 m terrain (no probe, no dense_deep_bed)
    bed = to_device_tree(dense, cp)
    dctx = dataclasses.replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, cp))
    cfg = fb.config(backend="cuda", bottom="open")
    owner = nfs.initialize_native_floating_soil(fb.native_bed(bed, dctx), cfg, initial_relative_saturation=0.5)
    vert = fp.FloatingPondVertical(owner, available_bytes_fn=fcu.device_reader(cp))
    be = type("B", (), {"hydraulics": "cuda", "xp": cp})()
    ctl = wh.control_for(be, windowed=False, max_dt_s=0.2)
    ev = PondEvent(dctx, None, plot1_sediment_parameters(xp=cp), constant_rainfall(0.0, 100.0, 100.0), topo, control=ctl, splash=splash_parameters(np.linspace(0.4, 3.0, fb.NC), xp=cp), floating_vertical=vert)
    st = run(ev, ev.initial_state(bed, vert.initial_soil_water()), 2, rate=100.0 / 3.6e6)
    ck = tmp_path / "ck"
    fcc.save_floating_pond_checkpoint_cuda(ck, st, ev, IDENT)
    rec = fhc.export_held_context_cuda(vert, None, available_bytes_fn=AVAIL)
    e = vert.held_template()
    held_ctx = fh.reconstruct_held_context(rec, fh.ExpectedHeldIdentities(vert.config.sha256, e.column.sha256, e.matrix_op.sha256, None), available_bytes_fn=AVAIL)
    pool = cp.get_default_memory_pool()
    admitted = []
    real_admit = fcc.admit
    monkeypatch.setattr(fcc, "admit", lambda c, d, dv, h, q, w: admitted.append(int(dv)) or real_admit(c, d, dv, h, q, w))

    class Peak:
        def __init__(self):
            self.base, self.peak, self.rounding = int(pool.used_bytes()), 0, 0

        def __call__(self, size):
            ptr = pool.malloc(size)
            self.peak = max(self.peak, int(pool.used_bytes()) - self.base)  # the ACTUAL pool usage after this allocation
            self.rounding += int(ptr.mem.size) - int(size)  # the ACTUAL rounding of this block
            return ptr

    plateau = []
    for _ in range(3):
        hvert = fp.FloatingPondVertical.from_held_context(held_ctx, available_bytes_fn=AVAIL)
        ev2 = target_event(ev, hvert)
        gc.collect()
        cp.cuda.Device().synchronize()
        admitted.clear()
        hook = Peak()
        with cp.cuda.using_allocator(hook):
            got = fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT)
        model = max(admitted)
        assert hvert.current_payload(got.hydraulic.soil_water_m).state.water_m.shape == (vert.n_layers, vert.n_cells)
        assert hook.peak <= model + hook.rounding, (hook.peak, model, hook.rounding)  # TRUE peak vs the declared model + the ACTUAL rounding (a failure is a model underestimate)
        del got, ev2, hvert  # the restored state, its event and the target vertical (registry and provisional refs die with them)
        gc.collect()
        cp.cuda.Device().synchronize()
        plateau.append(int(pool.used_bytes()))
    assert max(plateau) - min(plateau) <= hook.rounding, plateau  # stable released plateau across repetitions, within the measured rounding


def test_prepared_parts_gates_the_current_device_before_any_hash_reader_or_copy(cp, tmp_path, monkeypatch):
    """A GENUINE prepared record from the REAL loader is retained by injecting a late validator failure (no fake capability). A simulated wrong current device (ORDERING control on one UUID, not a second-device
    qualification) must be refused FIRST: no integrity digest, no stream comparison, no counted transfer."""
    from maple.core import backend as mb

    from maple_syrup import floating_resume_cuda as frc

    ev, st, vert, _o = original(cp, tmp_path / "o")
    st = run(ev, st, 4)
    ck, ev2, hvert = save_load(cp, tmp_path, ev, st, vert)
    kept = {}
    real = frc.prepare_from_verified_inputs

    def spy(event, inputs, *, quota):
        out = real(event, inputs, quota=quota)
        kept["prepared"] = out[0]
        return out

    monkeypatch.setattr(frc, "prepare_from_verified_inputs", spy)
    monkeypatch.setattr(ev2, "audit", lambda state: (_ for _ in ()).throw(fp.FloatingPondError("injected late validator failure")))
    with pytest.raises(fc.PondCheckpointError):
        fcc.load_floating_pond_checkpoint_cuda(ck, ev2, IDENT)
    prepared = kept["prepared"]  # genuine, minted, unconsumed
    seen = []
    for name in ("input_digest", "verify_intact", "stream_equal"):
        real_fn = getattr(fcc, name)
        monkeypatch.setattr(fcc, name, lambda *a, _n=name, _f=real_fn: seen.append(_n) or _f(*a))
    real_dev = fcu.current_device
    monkeypatch.setattr(fcu, "current_device", lambda c: real_dev(c) + 1)
    mb.reset_transfer_counters()
    with pytest.raises(fcc.FloatingCheckpointCudaError, match="CUDA device gate refused"):
        frc.prepared_parts(prepared, hvert)
    c = mb.read_transfer_counters()
    assert seen == [] and c.device_to_host == 0 and c.host_to_device == 0 and c.scalar_reads == 0
    # POSITIVE instrumentation control under the SAME genuine, unconsumed record: with the real current device restored the identical call reaches the hash and the comparisons (so the empty `seen` above was not vacuous)
    monkeypatch.setattr(fcu, "current_device", real_dev)
    frc.prepared_parts(prepared, hvert)
    assert {"verify_intact", "input_digest", "stream_equal"} <= set(seen)
