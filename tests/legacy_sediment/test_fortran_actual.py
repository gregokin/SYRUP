"""Tiny executions of the ORIGINAL MAHLERAN routines through the harness (compiler required; written without being run).

A configured compiler that fails is a failure (the session fixtures raise); only an unconfigured toolchain skips."""
from __future__ import annotations

import importlib.util
import math
import os
import stat
from pathlib import Path

import compare_legacy_sediment as C
import numpy as np
import pytest
import sources as S

from .helpers import FRACTIONS, make_graph, pit_graph, tiny_case_arrays

PLOT1_XML = Path("/home/okin/MAHLERAN/mahleran_input.xml")
needs_xml = pytest.mark.skipif(not PLOT1_XML.is_file(), reason="MAHLERAN Plot 1 XML absent")
needs_numba = pytest.mark.skipif(importlib.util.find_spec("numba") is None, reason="Numba not installed")
COL = {name: i for i, name in enumerate(S.LEDGER_NAMES)}


def xml_block():
    return S.xml_sediment_block(PLOT1_XML, S.sha256_file(PLOT1_XML))


def prepare(tmp_path, graph=None, *, steps=60, capture=(30,), snapshot=(60,)):
    graph = graph or pit_graph()
    arrays, column, kwargs = tiny_case_arrays(graph, n_steps=steps)
    kwargs.update(xml=xml_block(), iroute=5, capture_steps=capture, snapshot_steps=snapshot)
    (tmp_path / "in").mkdir()  # the input directory is protected from the run directory, so keep them apart
    path = tmp_path / "in" / "input.bin"
    S.write_input(path, arrays, **kwargs)
    nr2, nc2 = arrays["aspect"].shape
    expected = {"n_steps": steps, "iroute": 5, "nr2": nr2, "nc2": nc2, "hooked": True, "capture_steps": list(capture),
                "snapshot_steps": list(snapshot), "active_cells": int(arrays["n_active"])}
    return graph, arrays, column, path, expected


@needs_xml
def test_all_six_classes_run_the_original_routines_with_an_independent_clip_and_closed_identity(hooked_build, tmp_path):
    _, _, _, path, expected = prepare(tmp_path)
    rec = S.run_once(hooked_build["executable"], path, tmp_path / "run", expected=expected, build_record=hooked_build)
    assert rec["status"] == "complete", rec.get("reason")
    assert rec["result"]["HOOK_CALLED"].strip() == "T"
    led = rec["ledger"]  # (steps, 13, 6)
    assert led[:, COL["pickup_kg"]].sum(axis=0).min() > 0.0  # every one of the six classes is picked up (no class skip)
    assert led[:, COL["deposition_active_kg"]].sum() > 0.0
    assert led[:, COL["deposition_pit_kg"]].sum() > 0.0  # the pit is credited by the original walk
    assert led[:, COL["cn_export_kg"]].sum() == 0.0  # a closed domain: no outlet
    # the pool identity uses the HOOK clip (the actual unclipped trial), never a residual-derived value
    resid = (led[:, COL["new_mobile_kg"]] - led[:, COL["old_mobile_kg"]]
             - (led[:, COL["pickup_kg"]] - led[:, COL["deposition_active_kg"]])
             + led[:, COL["cn_export_kg"]] - led[:, COL["effective_clip_source_kg"]])
    scale = sum(np.abs(led[:, COL[k]]) for k in ("pickup_kg", "deposition_active_kg", "effective_clip_source_kg",
                                                  "old_mobile_kg", "new_mobile_kg", "cn_export_kg"))
    assert np.max(np.abs(resid) / np.maximum(scale, 1e-300)) < 1e-10
    clipped = led[:, COL["effective_clip_source_kg"]].sum()
    assert math.isclose(float(rec["maps"]["CUMCLIP "].sum()), float(clipped), rel_tol=1e-12, abs_tol=1e-300)
    assert rec["kernel_s"] > 0.0 and rec["diag_s"] >= 0.0 and rec["max_rss_kib"] > 0
    assert S._f(rec["result"], "PROGRESS_SECONDS") >= 0.0  # progress I/O is timed apart and excluded from the loop timers
    assert "legacy_sediment_driver: step 60 of 60" in (tmp_path / "run" / "stderr.log").read_text()  # flushed final-step progress
    assert set(rec["captures"]) == {30} and set(rec["snapshots"]) == {60}


@needs_xml
def test_the_preclip_hook_has_no_effect_on_the_science(hooked_build, nohook_build, tmp_path):
    _, _, _, path, expected = prepare(tmp_path, capture=(), snapshot=())
    on = S.run_once(hooked_build["executable"], path, tmp_path / "on", expected=expected, build_record=hooked_build)
    off = S.run_once(nohook_build["executable"], path, tmp_path / "off", expected={**expected, "hooked": False},
                     build_record=nohook_build)
    assert on["status"] == "complete" and off["status"] == "complete", (on.get("reason"), off.get("reason"))
    keep = [i for i, n in enumerate(S.LEDGER_NAMES) if n != "effective_clip_source_kg"]
    assert np.array_equal(on["ledger"][:, keep], off["ledger"][:, keep])
    for tag in ("CUMDET  ", "CUMDEP  ", "MOBILE  ", "DEPTH_MM", "VELOC_MM", "SOILW_MM", "DISCH_MM"):
        assert np.array_equal(on["maps"][tag], off["maps"][tag]), tag
    assert np.array_equal(on["water_steps"], off["water_steps"])
    assert not off["ledger"][:, COL["effective_clip_source_kg"]].any() and off["result"]["HOOK_CALLED"].strip() == "F"


@needs_xml
@pytest.mark.parametrize("damage", ["truncate", "bad_endian", "trailing", "bad_magic"])
def test_malformed_input_is_a_failed_run_never_a_result(hooked_build, tmp_path, damage):
    _, _, _, path, expected = prepare(tmp_path, steps=5, capture=(), snapshot=())
    data = bytearray(path.read_bytes())
    if damage == "truncate":
        data = data[:-200]
    elif damage == "bad_endian":
        data[12:16] = bytes(reversed(data[12:16]))  # the endian marker, byte-swapped
    elif damage == "trailing":
        data += b"x"
    else:
        data[:8] = b"BADMAGIC"
    (tmp_path / "bad_in").mkdir()
    bad = tmp_path / "bad_in" / "bad.bin"
    bad.write_bytes(bytes(data))
    rec = S.run_once(hooked_build["executable"], bad, tmp_path / "run", expected=expected, build_record=hooked_build)
    assert rec["status"] == "failed" and "ledger" not in rec


def test_a_zero_exit_without_the_completion_marker_is_not_a_result(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "in").mkdir()
    script = tmp_path / "bin" / "fake_driver"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    inp = tmp_path / "in" / "input.bin"
    inp.write_bytes(b"x")
    rec = S.run_once(script, inp, tmp_path / "run", expected={"n_steps": 1, "iroute": 5, "nr2": 3, "nc2": 3, "hooked": True,
                                                                "capture_steps": [], "snapshot_steps": [], "active_cells": 1})
    assert rec["status"] == "failed" and "completion marker" in rec["reason"]  # a clean failure, not a FileNotFoundError
    assert (tmp_path / "run").is_dir()  # the (empty) partial output is preserved as evidence


@needs_xml
def test_executable_integrity_is_enforced(hooked_build, tmp_path):
    _, _, _, path, expected = prepare(tmp_path, steps=2, capture=(), snapshot=())
    stale = dict(hooked_build, executable_sha256="0" * 64)
    with pytest.raises(S.FortranGlueError, match="build record"):
        S.run_once(hooked_build["executable"], path, tmp_path / "run", expected=expected, build_record=stale)
    assert not os.path.exists(tmp_path / "run")  # refused before anything was created


# --- original flow_distrib: native walk semantics ---------------------------------------------------------------------
WALK_GRIDS = {
    "channel_to_ring": ([[5.0, 4.0, 3.0, 2.0, 1.0]], {(1, 6): 0.5}),
    "terminal_pit": ([[3.0, 2.0, 1.0, 2.0, 3.0]], None),
    "pit_in_column_zero": ([[1.0, 2.0, 3.0, 4.0, 5.0]], None),
    "inactive_neighbour": ([[-9999.0, 1.0, 2.0, 3.0]], None),
}
LENGTHS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


@pytest.mark.parametrize("name", sorted(WALK_GRIDS))
@pytest.mark.parametrize("nsteps", [2, 100])
def test_native_walk_matches_the_original_flow_distrib(walk_probe_build, tmp_path, name, nsteps):
    from maple_syrup import legacy_native as N
    from maple_syrup.legacy_native_numba import get_kernels
    from maple_syrup.sediment_physics import REGIME_CODES

    interior, ring_low = WALK_GRIDS[name]
    graph = make_graph(interior, ring_low)
    ny, nx = graph.shape
    net = N.native_network(graph)
    n = net.active.size
    aspect_full = S.full_north_first(np.asarray(graph.aspect, dtype=np.float64), 0.0).astype(int)
    calls = []
    for cell in net.active_idx:  # includes aspect-0 sources: forced detachment, as the original routine would do for one
        r, c = divmod(int(cell), nx)
        for phi in range(1, 7):
            calls.append((ny - r + 1, c + 2, phi, 0.1 * phi, LENGTHS[phi - 1], nsteps))  # north-first row (ny - 1 - r) + 2
    ref = S.run_walk_probe(walk_probe_build["executable"], tmp_path / "w", aspect_full=aspect_full, dx_m=graph.dx_m, dt=1.0,
                           calls=calls)  # (nr2, nc2, 6)
    depos, ring, inactive_dep, erased = np.zeros((n, 6)), np.zeros(6), np.zeros(6), np.zeros(6)
    counts = np.zeros(N.N_COUNTS, dtype=np.int64)
    limits = np.zeros(max(REGIME_CODES.values()) + 1, dtype=np.int64)
    limits[REGIME_CODES["concentrated"]] = nsteps
    kernels = get_kernels(False)
    for cell in net.active_idx:
        for phi in range(1, 7):
            det = np.zeros((n, 6))
            det[int(cell), phi - 1] = 0.1 * phi
            kernels.walk(np.array([int(cell)]), det, np.full((n, 6), 1.0 / LENGTHS[phi - 1]), np.ones((n, 6), dtype=bool),
                         np.full((n, 6), REGIME_CODES["concentrated"], dtype=np.int8), limits, net.slope_zero, net.walk_first,
                         net.walk_next, net.aspect0, net.inactive, np.zeros(n, dtype=bool), False, net.dx_m, 1.0, depos, ring,
                         inactive_dep, erased, counts)
    interior_ref = ref[1:-1, 1:-1][::-1, :, :].reshape(n, 6)  # SYRUP layout: row 0 south
    np.testing.assert_allclose(interior_ref[net.active_idx], depos[net.active_idx], rtol=S.WALK_RTOL, atol=S.WALK_ATOL)
    ring_ref = ref.sum() - ref[1:-1, 1:-1].sum()
    np.testing.assert_allclose(ring_ref, ring.sum(), rtol=S.WALK_RTOL, atol=S.WALK_ATOL)
    np.testing.assert_allclose(interior_ref[net.inactive].sum(), inactive_dep.sum(), rtol=S.WALK_RTOL, atol=S.WALK_ATOL)
    if name == "terminal_pit":
        assert counts[N.C_TERMINAL] > 0 and counts[N.C_ASPECT0] > 0  # the original started the pit source WEST, as A1 does


# --- state injection ---------------------------------------------------------------------------------------------------
@needs_xml
@needs_numba
def test_state_injection_runs_the_a1_step_on_the_state_the_original_saw(hooked_build, tmp_path):
    from maple_syrup.sediment_physics import (
        physics_grid_from_graph,
        plot1_sediment_parameters,
    )

    graph, _, _, path, expected = prepare(tmp_path)
    rec = S.run_once(hooked_build["executable"], path, tmp_path / "run", expected=expected, build_record=hooked_build)
    assert rec["status"] == "complete", rec.get("reason")
    ny, nx = graph.shape
    holdings = np.broadcast_to(FRACTIONS * 2.5, (ny, nx, 6)).copy()
    engine, _ = C.build_engine(graph, plot1_sediment_parameters(), physics_grid_from_graph(graph), np.zeros((ny, nx)), holdings)
    af = S._f(rec["result"], "AF_KG_PER_MM")
    dx_mm = S._f(rec["result"], "DX_MM")
    density = S._f(rec["result"], "DENSITY_G_CM3")  # the widened kind-4 value the original used
    assert density != 2.65 and abs(density - 2.65) < 1e-6  # REAL(2.65) widened: not the nominal double
    cap = rec["captures"][30]
    assert "PRE_DS1 " in cap and "PRE_QS1 " in cap and "PRE_DS1" not in cap  # exact 8-character tags
    post = C.injection_check(cap, engine, nr2=expected["nr2"], nc2=expected["nc2"], af=af, dx_mm=dx_mm, density_g_cm3=density)
    assert post["detachment_positive_mask_mismatches"] == 0, post  # exact branch agreement at the original's own depth
    assert post["fields"]["detachment_rate_kg_s"]["n_bad"] == 0, post  # equation parity, predeclared rtol 2e-6 / atol 1e-14
    assert post["all_close"], post
    previous = C.injection_check(cap, engine, nr2=expected["nr2"], nc2=expected["nc2"], af=af, dx_mm=dx_mm,
                                 density_g_cm3=density, depth="previous")  # quantifies (does not assert) the A1 time-level departure
    assert set(previous["fields"]) == set(post["fields"])
