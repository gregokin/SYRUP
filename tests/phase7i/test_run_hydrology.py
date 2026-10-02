"""The SYRUP replay core (`simulate`), its prepared/reference parity tracker, forcing/log consistency, step-refusal
recording and the real Plot 1 geometry/units gate. Hydrology only: no sediment, no MAPLE bed write.
Nothing here was run by its author (file-only tools); Codex records actual results."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from capture_fixture import render_capture
from test_routing import make_graph, valley_full

pytest.importorskip("maple")

import capture_data as cd
import run_syrup_hydrology as rsh

from maple_syrup import hydrology_numba as hn
from maple_syrup.infiltration import column_parameters, initial_soil_water_m
from maple_syrup.rainfall import rainfall_field
from maple_syrup.routing import RoutingError
from maple_syrup.routing_numba import numba_available

ROOT = Path(__file__).resolve().parents[2]
needs_numba = pytest.mark.skipif(not numba_available(), reason="Numba not installed; no claim made")


def small_case(seed: int = 3):
    rng = np.random.default_rng(seed)
    graph = make_graph(valley_full(4, 3), ff=5.0)  # 2 m of valley: runoff reaches the outlet within the 150 s window
    shape = graph.shape
    ksat = rng.uniform(5.0e-8, 4.0e-6, shape)  # a heterogeneous, strictly positive realization
    params = column_parameters(
        model="pavement_hawkins", ksat_m_per_s=ksat, suction_m=np.full(shape, 0.0466),
        drainage_parameter=np.full(shape, 0.05), theta_sat=np.full(shape, 0.4), soil_thickness_m=np.full(shape, 0.3),
        pavement_cover_fraction=np.full(shape, 1.0), active_mask=graph.active.copy())
    soil0 = initial_soil_water_m(params, np.full(shape, 0.25))
    scale = np.ones(shape)
    scale[1, 1] = 0.5
    return graph, params, rainfall_field(*shape, scale=scale), soil0, scale


def rates_m_s(n=150):
    """360 mm/h-class rain (>> the Hawkins capacity, so every raining cell stays in the no-run-on branch and the strict
    hpre guard cannot be the subject of this test), switching between three exact values, then dry recession."""
    r = np.zeros(n)
    r[2:60] = 1.0e-4 * (1.0 + 0.1 * (np.arange(58) % 3))
    return r


# --- parity tracker -----------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Fake:
    x: np.ndarray
    n: int
    s: str
    scalar: float
    maybe: object = None


def fake(**changes):
    base = {"x": np.linspace(1.0, 2.0, 5), "n": 3, "s": "numba", "scalar": 0.25}
    return Fake(**{**base, **changes})


def test_parity_tracker_passes_identical_and_roundoff_but_flags_everything_else():
    t = rsh.ParityTracker()
    t.update(fake(), fake())
    t.update(fake(), fake(x=np.linspace(1.0, 2.0, 5) * (1.0 + 1.0e-13)))  # inside rtol 2e-12
    assert t.summary()["pass"] and t.summary()["compared_steps"] == 2 and t.summary()["n_public_fields"] == 5
    for change in ({"x": np.linspace(1.0, 2.0, 5) * (1.0 + 1.0e-9)}, {"n": 4}, {"s": "array"}, {"scalar": 0.26},
                   {"x": np.linspace(1.0, 2.0, 6)}, {"x": np.full(5, np.nan)}, {"x": np.linspace(1.0, 2.0, 5).astype(np.float32)},
                   {"maybe": 1}):
        bad = rsh.ParityTracker()
        bad.update(fake(), fake(**change))
        summary = bad.summary()
        assert not summary["pass"] and summary["fields_over_bound"], change
    worse = rsh.ParityTracker()
    worse.update(fake(), fake(n=4))
    summary = worse.summary()  # structural mismatch: JSON-safe null plus an explicit list, never a silent pass
    assert summary["max_normalised_excess"] is None and summary["structural_mismatch_fields"] == ["step.n"]
    assert summary["pass"] is False and json.dumps(summary, allow_nan=False)


def test_logged_forcing_must_be_the_rounding_of_the_captured_rate():
    rval = np.array([0.01, 0.0100045, 0.0])  # mm/s -> 36.0, 36.0162 mm/h
    text = "".join(f" Starting iteration {k + 1:6d} rain intensity: {v * 3600.0:7.2f} time step:  1.00 t:    0\n"
                    for k, v in enumerate(rval))
    record = rsh.check_logged_forcing(rval, text)
    assert record["n_steps"] == 3 and record["max_abs_difference_mm_h"] <= 0.005
    with pytest.raises(cd.CaptureError, match="differs"):
        rsh.check_logged_forcing(rval * 1.01, text)
    with pytest.raises(cd.CaptureError, match="one iteration line"):
        rsh.check_logged_forcing(rval, text.splitlines(True)[0])


# --- simulate ----------------------------------------------------------------------------------------------------
@needs_numba
@pytest.mark.parametrize("substeps", [1, 2])
def test_exact_forcing_budget_and_parity(substeps):
    graph, params, field, soil0, scale = small_case()
    rates = rates_m_s()
    result = rsh.simulate(graph, params, field, soil0, rates, substeps=substeps, implementation="prepared",
                          parity=True, snapshot_seconds=(10, 30))
    h, area = result["history"], graph.dx_m ** 2
    assert result["n_steps"] == rates.size * substeps and result["dt_s"] == 1.0 / substeps
    np.testing.assert_array_equal(h["rate_m_s"], np.repeat(rates, substeps))  # the supplied rate is applied unchanged
    np.testing.assert_allclose(h["t_s"], np.arange(1, rates.size * substeps + 1) / substeps, rtol=0, atol=0)
    expected_rain = float(np.sum(rates) * scale.sum() * area)
    assert result["budget"]["rain"] == pytest.approx(expected_rain, rel=1e-12)
    assert h["rain_m3"].sum() == pytest.approx(expected_rain, rel=1e-12)
    assert result["budget"]["closed"] and abs(result["budget"]["water_residual_m3"]) <= result["budget"]["bound_m3"]
    assert result["parity"]["pass"] and result["parity"]["compared_steps"] == result["n_steps"]
    peak = result["own_peak"]
    assert peak["index"] == int(np.argmax(h["outlet_m3_s"])) and peak["outlet_m3_s"] == h["outlet_m3_s"].max()  # first max
    assert set(result["snapshots"]) == {10, 30} and result["snapshots"][10]["depth_m"].shape == graph.shape
    assert result["final_soil_water_m"].shape == graph.shape and np.all(result["cum_drainage_m"] >= 0.0)
    assert h["outlet_m3_s"].max() > 0.0 and h["export_m3"].sum() > 0.0 and result["preparation"]["n_active"] == graph.n_active


@needs_numba
def test_cumulative_rain_does_not_depend_on_the_substep_count():
    graph, params, field, soil0, _ = small_case()
    one = rsh.simulate(graph, params, field, soil0, rates_m_s(), substeps=1)
    four = rsh.simulate(graph, params, field, soil0, rates_m_s(), substeps=4)
    np.testing.assert_allclose(one["cum_rain_m"], four["cum_rain_m"], rtol=1e-12)
    assert one["budget"]["closed"] and four["budget"]["closed"]


@needs_numba
def test_simulation_does_not_modify_its_inputs():
    graph, params, field, soil0, _ = small_case()
    rates = rates_m_s()
    before = (soil0.copy(), rates.copy(), params.ksat_m_per_s.copy(), params.storage_max_m.copy(),
              field.multiplier.copy(), graph.conveyance.copy())
    rsh.simulate(graph, params, field, soil0, rates, substeps=1, parity=True)
    after = (soil0, rates, params.ksat_m_per_s, params.storage_max_m, field.multiplier, graph.conveyance)
    for b, a in zip(before, after, strict=True):
        np.testing.assert_array_equal(b, a)


@needs_numba
@pytest.mark.parametrize("kwargs, match", [
    ({"rates_m_s": np.array([1.0e-5, -1.0e-5])}, "non-negative"),
    ({"rates_m_s": np.array([1.0e-5, np.nan])}, "finite"),
    ({"rates_m_s": np.zeros((2, 2))}, "1-D"),
    ({"rates_m_s": np.zeros(0)}, "non-empty"),
    ({"substeps": 3}, "substeps"),
    ({"implementation": "array"}, "implementation"),
    ({"implementation": "reference", "parity": True}, "PREPARED"),
])
def test_invalid_requests_are_refused_before_any_step(kwargs, match):
    graph, params, field, soil0, _ = small_case()
    args = {"rates_m_s": rates_m_s(), **kwargs}
    with pytest.raises(ValueError, match=match):
        rsh.simulate(graph, params, field, soil0, args.pop("rates_m_s"), **args)


@needs_numba
def test_a_refused_step_is_recorded_reproducibly_and_never_skipped(monkeypatch):
    graph, params, field, soil0, _ = small_case()
    real = hn.prepared_coupled_step
    calls = []

    def refusing(ctx, rate, state, dt, control):
        calls.append(1)
        if len(calls) == 3:
            raise RoutingError("synthetic strict-guard refusal")
        return real(ctx, rate, state, dt, control)

    monkeypatch.setattr(hn, "prepared_coupled_step", refusing)
    with pytest.raises(rsh.HydrologyStepFailure, match="step 2 refused") as info:
        rsh.simulate(graph, params, field, soil0, rates_m_s(), substeps=1, parity=True)
    record, inputs = info.value.record, info.value.inputs
    assert record["step_index"] == 2 and record["second"] == 2 and record["dt_s"] == 1.0
    assert record["primary_exception"] == {"class": "RoutingError", "message": "synthetic strict-guard refusal"}
    assert record["reference_exception"] is None and record["primary_and_reference_agree"] is False
    assert "NOT relaxed" in record["action"]
    assert {"depth_m", "soil_water_m", "discharge_m2_s", "rain_rate_field_m_s", "ksat_m_per_s", "dt_s", "t_start_s"} <= set(inputs)
    assert inputs["depth_m"].shape == graph.shape and float(inputs["t_start_s"]) == 2.0
    assert inputs["ksat_m_per_s"].tolist() == params.ksat_m_per_s.tolist()
    assert len(calls) == 3  # the run stopped at the refusal: no retry, no skipped step


# --- the real Plot 1 geometry, units and orientation -----------------------------------------------------------
def capture_from_syrup(graph, host, soil0, ksat_mm):
    """A Fortran-layout static capture of the SYRUP case itself, so the gates see the REAL grid, units and dtypes."""
    ny, nx = graph.shape
    nr, nc = ny + 1, nx + 1

    def full(a, ring):
        out = np.full((nr + 1, nc + 1), ring, dtype=a.dtype)
        out[1:nr, 1:nc] = a[::-1]
        return out

    theta_sat = np.asarray(host["theta_sat"])
    arrays = {
        "ksat": full(ksat_mm, 0.0), "psi": full(np.asarray(host["suction_m"]) * 1e3, 0.0), "theta_sat": full(theta_sat, 0.0),
        "theta": full(np.asarray(host["initial_theta"]), 0.0), "cum_inf": full(np.asarray(soil0) * 1e3, 0.0),
        "cum_drain_initial_mm": np.zeros((nr + 1, nc + 1)),
        "stmax": full(theta_sat * np.asarray(host["soil_thickness_m"]) * 1e3, 0.0),
        "drain_par": full(np.asarray(host["drainage_parameter"]), 0.0),
        "pave": full(np.asarray(host["pavement_cover_fraction"]) * 1e-2, 0.0), "slope": full(np.asarray(graph.slope), 0.0),
        "ff": full(np.asarray(graph.friction_factor), 0.0), "rmask": full(np.asarray(host["rainfall_scale"]), -9999.0),
        "aspect": full(np.asarray(graph.aspect).astype(np.int64), 0),
        "order": np.array([[i, j, 1] for i in range(2, nr + 1) for j in range(2, nc + 1)], dtype=np.int64),
        "d_initial_mm": np.zeros((nr + 1, nc + 1)), "q_initial_mm2_s": np.zeros((nr + 1, nc + 1)),
    }
    scalars = {"nr": nr, "nc": nc, "nr1": nr, "nc1": nc, "nr2": nr + 1, "nc2": nc + 1, "ncell1": int(arrays["order"].shape[0]),
               "nit": 5, "ndirn": 4, "iroute": 5, "ff_type": 1, "inf_type": 2, "inf_model": 2, "rain_type": 2,
               "dt_s": 1.0, "dx_mm": graph.dx_m * 1e3, "dy_mm": graph.dx_m * 1e3, "ksat_mod": 1.0, "psi_mod": 1.0,
               "rval_initial_mm_s": 0.01, "stormlength_s": 5.0}
    return cd.parse_capture_text(render_capture("static", scalars, arrays), "static")


@needs_numba
@pytest.mark.skipif(not (ROOT / "outputs/plot1/syrup/plot1_binding.json").is_file(), reason="Plot 1 case not available")
def test_real_plot1_grid_units_orientation_and_injected_realization_run():
    from maple_syrup.case_import import verify_plot1_case
    from maple_syrup.column_experiment import _bed_digest
    from maple_syrup.sediment_experiment import prepare_verified_sediment_case

    verified = verify_plot1_case(ROOT / "outputs/plot1", allow_maple_source_change=True)
    prepared = prepare_verified_sediment_case(verified, end_s=6.0)
    graph, host, field, soil0, case = (prepared[k] for k in ("graph", "host", "field", "soil0", "case"))
    digest = _bed_digest(case)
    rng = np.random.default_rng(11)
    ksat_mm = rng.uniform(5.0e-5, 3.0e-3, graph.shape)  # a heterogeneous realization on the real 60 x 20 grid
    static = capture_from_syrup(graph, host, np.asarray(soil0), ksat_mm)
    nr, nc = int(static.scalars["nr"]), int(static.scalars["nc"])
    report = cd.check_static_consistency(static, cd.reference_arrays(graph, host, np.asarray(soil0)))
    assert report["outlet_cells"]["legacy"] == int(graph.outlet.sum()) > 0  # the output routine's outlet rule agrees
    interior = cd.validate_ksat_mm_s(static.arrays["ksat"], nr, nc)
    np.testing.assert_array_equal(interior, ksat_mm)  # row order, crop and 17-digit text are exact on the real grid
    injected = cd.inject_ksat(host, interior)
    assert injected["ksat_m_per_s"].shape == graph.shape and np.all(host["ksat_m_per_s"] == host["ksat_m_per_s"].flat[0])  # constant values (std has FP roundoff)
    column = column_parameters(model="pavement_hawkins", **{k: injected[k] for k in (
        "ksat_m_per_s", "suction_m", "drainage_parameter", "theta_sat", "soil_thickness_m", "pavement_cover_fraction")})
    np.testing.assert_array_equal(column.ksat_m_per_s, ksat_mm * 1.0e-3)
    # 3600 mm/h for 6 s: far above any Hawkins capacity (<= ~4e-4 m/s), so J < P everywhere and the strict hpre guard
    # (a known, separately reported limitation) is not what this plumbing test exercises.
    rate = 1.0e-3
    result = rsh.simulate(graph, column, field, np.asarray(soil0), np.full(6, rate), substeps=1, parity=True)
    assert result["budget"]["closed"] and result["parity"]["pass"]
    assert result["history"]["rain_m3"].sum() == pytest.approx(6 * rate * float(np.sum(host["rainfall_scale"])) * graph.dx_m ** 2,
                                                               rel=1e-12)
    assert _bed_digest(case) == digest  # hydrology-only: the MAPLE bed is untouched
