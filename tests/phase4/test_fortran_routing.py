"""Executed ORIGINAL MAHLERAN `route_water` (iroute 5) versus `route_step`.

These tests compile the unchanged reference sources with the supplied
gfortran (see fortran_reference.py for the environment variables) and run
ONE routing step at a time through a driver that contains no routing
equations. They are an executed-routine benchmark, not a MAHLERAN model
run: no infiltration, rainfall loop, sediment or output code is executed.
Without a compiler every test skips; nothing is then claimed.

Agreed cases use inputs where the legacy bracket [0, 100 (d(1) + excess)]
contains the root and the old inflow is coherent; they must agree within
the legacy bisection tolerance (1e-8 mm = 1e-11 m). Known legacy behaviours
(stale qin(1) gain, bracket truncation, STOP on a negative right-hand side)
are exercised separately and reported, never suppressed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fortran_reference import (
    COMPILED_SOURCES,
    MAHLERAN_ROOT,
    ORIGINAL_FLAGS,
    REFERENCE_SOURCES,
    build_reference,
    interior_si,
    legacy_step,
    locate_toolchain,
    reference_hashes,
    run_route_water,
    watched_listing,
)
from test_routing import (
    chain_full,
    implementations,
    make_graph,
    random_full,
    random_state,
    south_export,
    valley_full,
)

from maple_syrup.routing import (
    RoutingError,
    legacy_stale_inflow_step,
    plot1_routing_graph,
    route_step,
)

REPO = Path(__file__).resolve().parents[2]
RECIPE_PATH = REPO / "cases" / "plot1" / "recipe.yaml"
PLOT1_DIR = MAHLERAN_ROOT / "Input" / "input_p1"
LEGACY_TOL_M = 1.0e-11  # route_water.for 22: tol = 1e-8 mm


@pytest.fixture(scope="module")
def reference_listing():
    pytest.importorskip("maple")
    if not all((MAHLERAN_ROOT / rel).is_file() for rel in REFERENCE_SOURCES):
        pytest.skip(f"MAHLERAN reference sources not available at {MAHLERAN_ROOT}")
    return watched_listing()


@pytest.fixture(scope="module")
def build(tmp_path_factory, reference_listing):
    toolchain = locate_toolchain()
    if toolchain is None:
        pytest.skip("no gfortran found (set MAPLE_SYRUP_GFORTRAN); the original routine was NOT executed "
                    "and no Fortran comparison is claimed")
    result = build_reference(toolchain, tmp_path_factory.mktemp("fortran") / "build")
    assert watched_listing() == reference_listing, "the build wrote into the reference tree"
    print("fortran build:", json.dumps(result.record(), indent=1))
    return result


def _coherent_old(graph, h_old):
    """q(1) = k d(1)^{3/2} and the donor sum of those same values (host)."""
    k = np.asarray(graph.conveyance).reshape(graph.shape)
    q_old = np.where(graph.active, (np.sqrt(h_old) * h_old) * k, 0.0)
    qin_old = np.zeros(graph.shape).reshape(-1)
    receiver = graph.receiver.reshape(-1)
    for cell in np.flatnonzero(graph.active):
        if receiver[cell] >= 0:
            qin_old[receiver[cell]] += q_old.reshape(-1)[cell]
    return q_old, qin_old.reshape(graph.shape)


def _run(build, graph, tmp_path, name, *, h_start, h_old, dt, q_old, qin_old, export_full):
    step = legacy_step(graph, old_flow_depth_m=h_old, depth_start_m=h_start, old_discharge_m2_s=q_old,
                       old_inflow_m2_s=qin_old, dt_s=dt, ring_export_full=export_full)
    return step, run_route_water(build, step, tmp_path / name)


def _legacy_si(result):
    return (interior_si(result.d2_mm, 1e-3), interior_si(result.q2_mm2_s, 1e-6),
            interior_si(result.qin2_mm2_s, 1e-6), interior_si(result.v_mm_s, 1e-3))


def _assert_inside_legacy_bracket(step, depth_m, graph):
    """The agreed cases must not depend on the legacy bracket heuristic."""
    d1, excess = interior_si(step.d1_mm, 1.0), interior_si(step.excess_mm_s, 1.0)
    dhigh = 100.0 * (d1 + excess)
    dhigh = np.where(dhigh == 0.0, 0.5, dhigh)
    wet = graph.active & (depth_m > 0.0)
    assert np.all(depth_m[wet] * 1000.0 < dhigh[wet]), "legacy bracket would truncate this root"


def _assert_matches(graph, ours, result, step):
    assert result.completed, f"original route_water did not complete:\n{result.stdout}\n{result.stderr}"
    assert result.real_bits == (32, 32), "shared_data dt/dx are expected to be default REAL(4)"
    _assert_inside_legacy_bracket(step, np.asarray(ours.depth_m), graph)
    d2, q2, qin2, v2 = _legacy_si(result)
    act = graph.active
    depth = np.asarray(ours.depth_m)
    k = np.asarray(graph.conveyance).reshape(graph.shape)
    np.testing.assert_allclose(depth[act], d2[act], rtol=0.0, atol=LEGACY_TOL_M)
    q_tol = 1.5 * k * np.sqrt(np.maximum(depth, d2) + LEGACY_TOL_M) * LEGACY_TOL_M + 1e-18
    assert np.all(np.abs(np.asarray(ours.discharge_m2_s) - q2)[act] <= q_tol[act])
    assert np.all(np.abs(np.asarray(ours.inflow_m2_s) - qin2)[act] <= 4.0 * q_tol.max())
    wet = act & (depth > 1e-6)
    v_tol = 0.5 * k * LEGACY_TOL_M / np.sqrt(np.where(wet, depth, 1.0)) + 1e-15
    assert np.all(np.abs(np.asarray(ours.velocity_m_s) - v2)[wet] <= v_tol[wet])
    print("max |depth - original| m:", float(np.max(np.abs(depth - d2)[act])))


def test_build_provenance_and_reference_immutability(build):
    assert build.source_sha256 == REFERENCE_SOURCES
    originals = [c for c in build.commands if any(str(MAHLERAN_ROOT / rel) in c for rel in COMPILED_SOURCES)]
    assert len(originals) == len(COMPILED_SOURCES)
    for command in originals:
        assert all(flag in command for flag in ORIGINAL_FLAGS)
    driver = [c for c in build.commands if any("reference_driver.f90" in part for part in c)]
    assert len(driver) == 1 and "-std=legacy" in driver[0] and "-std=f2008" not in driver[0]
    assert reference_hashes() == REFERENCE_SOURCES
    assert build.compiler_version


@pytest.mark.parametrize("impl", implementations())
@pytest.mark.parametrize("dt", [1.0, 0.5])
def test_linear_chain_matches_original(build, tmp_path, dt, impl):
    rng = np.random.default_rng(31)
    z = chain_full(6)
    g = make_graph(z, ff=21.45)
    h_old = rng.uniform(5e-4, 3e-3, g.shape)
    h_start = h_old + 1e-5 * dt
    ours = route_step(g, h_start, h_old, dt, implementation=impl)
    step, result = _run(build, g, tmp_path, "chain", h_start=h_start, h_old=h_old, dt=dt,
                        q_old=np.asarray(ours.old_discharge_m2_s), qin_old=np.asarray(ours.old_inflow_m2_s),
                        export_full=south_export(z))
    _assert_matches(g, ours, result, step)
    assert result.scratch_files == ()


@pytest.mark.parametrize("impl", implementations())
@pytest.mark.parametrize("seed", [4, 5])
def test_branching_networks_match_original(build, tmp_path, seed, impl):
    rng = np.random.default_rng(seed)
    for name, z in (("valley", valley_full(6, 5)), ("random", random_full(rng, 9, 7))):
        g = make_graph(z, ff=rng.uniform(5.0, 30.0, (z.shape[0] - 2, z.shape[1] - 2)))
        h_old = rng.uniform(5e-4, 3e-3, g.shape)
        h_start = h_old + rng.uniform(0.0, 2e-5, g.shape)
        ours = route_step(g, h_start, h_old, 1.0, implementation=impl)
        step, result = _run(build, g, tmp_path, name, h_start=h_start, h_old=h_old, dt=1.0,
                            q_old=np.asarray(ours.old_discharge_m2_s), qin_old=np.asarray(ours.old_inflow_m2_s),
                            export_full=south_export(z))
        _assert_matches(g, ours, result, step)


@pytest.mark.parametrize("impl", implementations())
def test_dry_and_wetting_cells_match_original(build, tmp_path, impl):
    z = valley_full(6, 5)
    g = make_graph(z, ff=21.45)
    h_old = np.zeros(g.shape)
    h_old[4:, :] = 2e-4  # thin water on the top rows; everything below is dry
    h_start = h_old.copy()
    ours = route_step(g, h_start, h_old, 1.0, implementation=impl)
    assert np.all(np.asarray(ours.depth_m)[:4, 2] > 0.0)  # dry centre cells wet in this step
    step, result = _run(build, g, tmp_path, "drywet", h_start=h_start, h_old=h_old, dt=1.0,
                        q_old=np.asarray(ours.old_discharge_m2_s), qin_old=np.asarray(ours.old_inflow_m2_s),
                        export_full=south_export(z))
    _assert_matches(g, ours, result, step)


@pytest.fixture(scope="module")
def plot1_audit(tmp_path_factory):
    if not (MAHLERAN_ROOT / "mahleran_input.xml").is_file() or not PLOT1_DIR.is_dir():
        pytest.skip(f"MAHLERAN reference not available at {MAHLERAN_ROOT}")
    from maple_syrup.case_import import audit_plot1, load_recipe

    return audit_plot1(load_recipe(RECIPE_PATH, mahleran_root=MAHLERAN_ROOT), tmp_path_factory.mktemp("plot1_audit"))


@pytest.mark.parametrize("impl", implementations())
def test_plot1_one_step_matches_original(build, plot1_audit, tmp_path, impl):
    fields = plot1_audit.fields
    g = plot1_routing_graph(fields, plot1_audit.report)
    rng = np.random.default_rng(41)
    h_start, h_old = random_state(rng, g, scale=3.5e-3)
    h_old = np.maximum(h_old, 1e-4)  # keep every cell inside the legacy bracket
    h_start = np.maximum(h_start, h_old)
    ours = route_step(g, h_start, h_old, 1.0, implementation=impl)
    step, result = _run(build, g, tmp_path, "plot1", h_start=h_start, h_old=h_old, dt=1.0,
                        q_old=np.asarray(ours.old_discharge_m2_s), qin_old=np.asarray(ours.old_inflow_m2_s),
                        export_full=np.asarray(fields["legacy_full_rainfall_scaling"]) < 0.0)
    _assert_matches(g, ours, result, step)
    print("plot1: levels", g.n_levels, "outlets", int(g.outlet.sum()),
          "export m3", float(ours.export_m3), "residual m3", float(ours.budget_residual_m3))


@pytest.mark.parametrize("impl", implementations())
def test_stale_old_inflow_gain_of_original_route_water(build, tmp_path, impl):
    """Codex reproducer through the tracked driver: upstream complete
    infiltration zeroed h and q(1), but the receiver's qin(1) keeps
    100 mm2/s. The original routine creates dx^2 c qin = 2.5e-5 m3."""
    z = chain_full(2)
    g = make_graph(z, ff=21.45)
    zero = np.zeros(g.shape)
    stale = np.array([[1e-4], [0.0]])
    _step, result = _run(build, g, tmp_path, "stale", h_start=zero, h_old=zero, dt=1.0, q_old=zero,
                        qin_old=stale, export_full=south_export(z))
    assert result.completed
    d2, q2, _, _ = _legacy_si(result)
    area, c = 0.25, 1.0
    legacy_total = area * float(d2.sum()) + area * c * float(q2[g.outlet].sum())
    print("original route_water storage + export m3:", repr(legacy_total), "(created from nothing)")
    assert legacy_total == pytest.approx(2.5e-5, abs=area * LEGACY_TOL_M)
    literal = legacy_stale_inflow_step(g, zero, zero, 1.0, stale_old_inflow_m2_s=stale, implementation=impl)
    np.testing.assert_allclose(np.asarray(literal.depth_m), d2, rtol=0.0, atol=LEGACY_TOL_M)
    assert float(literal.budget_residual_m3) == pytest.approx(2.5e-5, rel=1e-12)
    coherent = route_step(g, zero, zero, 1.0, implementation=impl)
    assert float(coherent.budget_residual_m3) == 0.0 and float(coherent.export_m3) == 0.0


@pytest.mark.parametrize("impl", implementations())
def test_invalid_legacy_bracket_truncates_the_root_and_loses_water(build, tmp_path, impl):
    """A dry receiver after complete run-on (d(1) = excess = 0) gets the
    fixed bracket [0, 0.5 mm]; with a large inflow the original bisection
    converges to 0.5 mm and the step loses water. Reported, not hidden."""
    z = chain_full(2)
    g = make_graph(z, ff=1.0)
    h_old = np.array([[0.0], [0.02]])  # dry receiver (bottom), 20 mm upstream
    ours = route_step(g, h_old, h_old, 1.0, implementation=impl)
    step, result = _run(build, g, tmp_path, "bracket", h_start=h_old, h_old=h_old, dt=1.0,
                        q_old=np.asarray(ours.old_discharge_m2_s), qin_old=np.asarray(ours.old_inflow_m2_s),
                        export_full=south_export(z))
    assert result.completed
    d2_mm = result.d2_mm[1:-1, 1:-1][::-1]
    q2_mm = result.q2_mm2_s[1:-1, 1:-1][::-1]
    assert d2_mm[0, 0] == pytest.approx(0.5, abs=1e-8)  # converged onto the bracket end
    assert float(ours.depth_m[0, 0]) > 5e-4  # the root the legacy bracket excluded
    c_mm = 1.0 / (2.0 * 500.0)
    qin1 = step.qin1_mm2_s[1:-1, 1:-1][::-1][0, 0]
    qin2 = result.qin2_mm2_s[1:-1, 1:-1][::-1][0, 0]
    rhs_mm = c_mm * (qin1 + qin2)  # d(1) = excess = q(1) = 0 at the receiver
    lost_m3 = (rhs_mm - d2_mm[0, 0] - c_mm * q2_mm[0, 0]) * 1e-3 * 0.25
    print("original route_water bracket truncation: water lost this step m3:", repr(lost_m3))
    assert lost_m3 > 1e-6
    assert abs(float(ours.budget_residual_m3)) <= 1e-16


def test_negative_rhs_stops_original_and_is_a_courant_rejection_here(build, tmp_path):
    z = chain_full(2)
    g = make_graph(z, ff=1.0)
    h = np.full(g.shape, 0.05)
    with pytest.raises(RoutingError, match="Courant"):
        route_step(g, h, h, 64.0)
    q_old, qin_old = _coherent_old(g, h)
    _, result = _run(build, g, tmp_path, "negative", h_start=h, h_old=h, dt=64.0, q_old=q_old,
                     qin_old=qin_old, export_full=south_export(z))
    assert not result.completed
    assert "RHS < 0" in result.stdout
    print("original route_water STOP, exit status", result.returncode)


def test_driver_refuses_dt_not_representable_in_legacy_real(build, tmp_path):
    z = chain_full(2)
    g = make_graph(z)
    h = np.full(g.shape, 1e-3)
    q_old, qin_old = _coherent_old(g, h)
    _, result = _run(build, g, tmp_path, "dt", h_start=h, h_old=h, dt=0.1, q_old=q_old, qin_old=qin_old,
                     export_full=south_export(z))
    assert not result.completed and result.returncode != 0
    assert "representable" in (result.stdout + result.stderr)


def test_reference_tree_unchanged_after_runs(build, reference_listing):
    """Runs last in this module: hashes and directory listings (no .mod, .o or
    fort.* written) are unchanged after every compile and run above."""
    assert reference_hashes() == REFERENCE_SOURCES
    assert watched_listing() == reference_listing
