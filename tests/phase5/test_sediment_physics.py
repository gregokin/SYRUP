"""Phase 5a wet-physics laws (`maple_syrup.sediment_physics`) against
scalar transcriptions of the MAHLERAN routines in their own legacy units
(mm, mm/s, g cm^-3, dt = 1 s), the root Plot 1 XML, and analytic limits.
Equation-level checks; nothing here executes MAHLERAN."""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maple")

from maple_syrup.sediment_physics import (
    LEGACY_CLASS_RADII_M,
    LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM,
    REGIME_CODES,
    SedimentPhysicsError,
    exponential_mean_from_median,
    median_diameter_m,
    median_from_exponential_mean,
    physics_grid,
    physics_grid_from_graph,
    plot1_sediment_parameters,
    recession_velocity,
    sediment_physics_step,
)

MAHLERAN_ROOT = Path(os.environ.get("MAPLE_SYRUP_MAHLERAN_ROOT", "/home/okin/MAHLERAN"))
EPS = np.finfo(np.float64).eps
AREA = 0.25  # 0.5 m cells
NC = 6


# --- scalar transcription of the legacy routines (legacy units) ----------------------------------
class Legacy:
    """`route_sediment_xml.f90` regime logic with `raindrop_detachment.for`,
    `flow_detachment.for`, `diffuse_flow_transport.for`,
    `conc_flow_transport.for` and `suspended_transport.for`, transcribed
    per cell in legacy units: d mm, v mm/s, r2 mm/s, veg %, slope
    dimensionless, diameter m, density g cm^-3, dt = 1 s. Returns
    detach mm/s, v_soil mm/s (None where no law sets it), travel_dist m
    (None where no law), local_only (all detached mass redeposited at
    once) and the regime label, per class."""

    def __init__(self, params, *, ke_model_type, literal_veg):
        p = params
        self.spa = p.raindrop_a / 1.2e3  # initialize_values 268
        self.spb, self.spc, self.spq = p.raindrop_b, p.raindrop_c, p.raindrop_depth_attenuation_per_cm
        self.hs = p.raindrop_max_depth_per_reference_s_m * 1e3
        self.density = p.particle_density_kg_m3 / 1000.0
        self.hz = p.flow_detachment_depth_scale_m * 1e3
        self.diameter = p.diameter_m
        self.radius = 0.5 * p.diameter_m
        self.viscosity = p.kinematic_viscosity_m2_s
        self.sigma = (1e3 * self.density - 1e3) / 1e3
        self.excess_density = 1000.0 * (self.density - 1.0)
        self.bagnold_density_scale = (self.excess_density / 1650.0) ** (-0.5)
        self.p_par = -2.0 / math.pi
        self.dstar_const = (((self.sigma - 1.0) * 9.81) / self.viscosity ** 2) ** (1.0 / 3.0)
        self.settling = [self.sigma * 9.81 * d ** 2 / (18.0 * self.viscosity) if d < 1e-4
                         else 1.1 * math.sqrt(self.sigma * 9.81 * d) for d in self.diameter]
        self.ke_model_type = ke_model_type
        self.literal_veg = literal_veg
        self.dt = 1.0

    def d50(self, propn):
        dsum, dsumlast, d50 = 0.0, 0.0, -9999.0
        for phi in range(6):
            dsum += propn[phi]
            if dsum >= 0.5 and dsumlast < 0.5:
                if phi == 0:
                    d50 = (self.diameter[0] / dsum) * 0.5
                else:
                    d50 = self.diameter[phi - 1] + ((self.diameter[phi] - self.diameter[phi - 1]) / (dsum - dsumlast)) * (0.5 - dsumlast)
                break
            dsumlast = dsum
        if d50 < 0.0:
            d50 = self.diameter[5]
        return d50

    def raindrop(self, r2, veg, slope, d, propn):
        if self.ke_model_type == 1:
            ke = (11.9 + 8.73 * math.log10(r2 * 3.6e3)) * (1.0 - 8.1e-3 * veg)
        elif self.literal_veg:
            ke = 29.0 - 20.88 * math.exp(-180.0 * r2) * (1.0 - 8.1e-3 * veg)
        else:
            ke = (29.0 - 20.88 * math.exp(-180.0 * r2)) * (1.0 - 8.1e-3 * veg)
        out = []
        for phi in range(6):
            det = self.spa[phi] * (ke * r2 * 1.2e3) ** self.spb[phi] * (slope * 100.0) ** self.spc[phi]
            det = 2.0 * det / self.density / self.dt
            if d > 0.0:
                det *= math.exp(-self.spq[phi] * (d / 10.0))
            det = max(det, 0.0)
            tptable = propn[phi] * self.hs[phi] / self.dt
            if phi == 1 and det > tptable:
                det = tptable
            if propn[phi] == 0.0:
                det = 0.0
            out.append(det)
        return out

    def flow(self, d, slope, propn):
        ustar = math.sqrt(9.81e-3 * d * slope)
        out = []
        for phi in range(6):
            dimless = ustar ** 2 / (self.sigma * 9.81 * self.diameter[phi])
            p_const = math.log(0.049 / (dimless * 0.25))
            sign = p_const / abs(p_const) if p_const != 0.0 else 0.0
            p = 0.5 - 0.5 * sign * math.sqrt(1.0 - math.exp(self.p_par * (p_const / 0.702) ** 2))
            tptable = propn[phi] * self.hs[phi] / self.dt
            det = min(p * self.hz * propn[phi] / self.dt, tptable)
            if propn[phi] == 0.0:
                det = 0.0
            out.append(det)
        return out

    def diffuse(self, r2, d, v, slope, phi):
        ke = (11.9 + 8.73 * math.log10(r2 * 3.6e3)) * r2
        fe = 9.81e-3 * d * v * slope
        p_mass = self.density * 1e6 * (4.0 / 3.0) * math.pi * self.radius[phi] ** 3
        v_soil = 0.525 * ke ** 2.35 * fe ** 0.981 / p_mass * 0.166666666667
        tm = 5.0e-2 * ke ** 1.85 * fe ** 0.481 * p_mass ** (-0.425)
        return min(v_soil, v), tm

    def conc(self, d, v, slope, d50, phi):
        bag = 4.554e-3 * (self.excess_density * d50) ** 1.5 * math.log10(12.0 * d / d50)
        bag = max(bag, 0.0)
        sp = 9.81e-3 * d * slope * v
        xs = sp - bag
        if xs <= 0.0:
            return None
        travel = min(2.85e-3 * xs ** 1.31 * self.diameter[phi] ** (-0.94) * 0.693, 30.0)
        v_soil = min(1.92e-2 * xs ** 1.01 * 2.777777777778e-1, v)
        return v_soil, travel

    def suspended(self, d, v, slope, phi):
        sp = 9.81e-3 * d * slope * v
        spf = min(7.331976e-3 * sp, 100.0)
        travel = 727.51805244 * math.exp(spf) * math.exp(-6.12683698 * self.diameter[phi] * 1000) * 0.693
        return v, travel * self.bagnold_density_scale

    def cell(self, *, d, v, r2, veg, slope, propn):
        detach = [0.0] * 6
        v_soil = [None] * 6
        travel = [None] * 6
        local_only = [False] * 6
        regime = [None] * 6
        if d > 0.0:
            d50 = self.d50(propn)
            ustar = math.sqrt(9.81e-3 * d * slope)
            re = (1e-6 * v * d) / self.viscosity
            if re >= 2500.0:
                detach = self.flow(d, slope, propn)
                for phi in range(6):
                    dstar = self.diameter[phi] * self.dstar_const
                    crit = 4.0 * self.settling[phi] / dstar if dstar <= 10 else 0.4 * self.settling[phi]
                    if ustar >= crit:
                        v_soil[phi], travel[phi] = self.suspended(d, v, slope, phi)
                        regime[phi] = "suspended"
                    else:
                        result = self.conc(d, v, slope, d50, phi)
                        regime[phi] = "concentrated"
                        if result is None:
                            local_only[phi] = True
                        else:
                            v_soil[phi], travel[phi] = result
            elif r2 > 0.0:
                detach = self.raindrop(r2, veg, slope, d, propn)
                if re > 500.0:
                    detach = [a + b for a, b in zip(detach, self.flow(d, slope, propn), strict=True)]
                for phi in range(6):
                    v_soil[phi], travel[phi] = self.diffuse(r2, d, v, slope, phi)
                    regime[phi] = "transitional_rain" if re > 500.0 else "diffuse"
            elif re > 500.0:
                detach = self.flow(d, slope, propn)
                for phi in range(6):
                    result = self.conc(d, v, slope, d50, phi)
                    regime[phi] = "transitional_dry"
                    if result is None:
                        local_only[phi] = True
                    else:
                        v_soil[phi], travel[phi] = result
            else:
                regime = ["wet_no_law"] * 6
        else:
            regime = ["dry"] * 6  # rain splash deferred
        return {"detach": detach, "v_soil": v_soil, "travel": travel, "local_only": local_only, "regime": regime}


# --- fixtures ----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def params():
    return plot1_sediment_parameters(ke_vegetation_form="legacy_literal")


def grid_for(n_cells, slope):
    slope = np.broadcast_to(np.asarray(slope, dtype=np.float64), (1, n_cells)).copy()
    return physics_grid(slope, np.ones((1, n_cells), dtype=bool), AREA)


def holdings(fractions, total_kg=1.0):
    f = np.asarray(fractions, dtype=np.float64)
    return f * total_kg


def run(params, *, depth, velocity, rain_mm_h, veg_fraction, slope, fractions_by_cell, dt=1.0, prev_v=None,
        total_kg=1.0):
    n = len(depth)
    grid = grid_for(n, slope)
    to_cell = lambda seq: np.asarray(seq, dtype=np.float64).reshape(1, n)
    mass = np.stack([holdings(f, total_kg) for f in fractions_by_cell]).reshape(1, n, NC)
    prev = np.zeros((1, n, NC)) if prev_v is None else prev_v
    return sediment_physics_step(
        params, grid, to_cell(depth), to_cell(velocity), to_cell(rain_mm_h) / 3.6e6, to_cell(veg_fraction),
        mass, prev, dt,
    )


def compare_with_legacy(params, step, legacy, cells, *, dt=1.0, cell_index=None):
    """Compare one physics step with the scalar transcription, cell by cell."""
    rho = params.particle_density_kg_m3
    for j, c in enumerate(cells):
        i = j if cell_index is None else cell_index[j]
        ref = legacy.cell(d=c["depth"] * 1e3, v=c["velocity"] * 1e3, r2=c["rain_mm_h"] / 3.6e3,
                          veg=c["veg"] * 100.0, slope=c["slope"], propn=c["fractions"])
        for phi in range(NC):
            expected_kg = ref["detach"][phi] * 1e-3 * AREA * rho * dt
            assert step.requested_pickup_kg[0, i, phi] == pytest.approx(expected_kg, rel=1e-12, abs=1e-300)
            assert REGIME_CODES[ref["regime"][phi]] == int(step.regime[0, i, phi])
            assert bool(step.settle_mask[0, i, phi]) == (ref["local_only"][phi] or c["depth"] <= 0.0)
            if ref["v_soil"][phi] is not None and not ref["local_only"][phi]:
                assert bool(step.law_applies[0, i, phi])
                assert step.sediment_velocity_m_s[0, i, phi] == pytest.approx(ref["v_soil"][phi] * 1e-3, rel=1e-12)
                assert 1.0 / step.deposition_rate_per_m[0, i, phi] == pytest.approx(ref["travel"][phi], rel=1e-12)
            else:
                assert not bool(step.law_applies[0, i, phi])
                assert step.deposition_rate_per_m[0, i, phi] == 0.0


# --- provenance of constants -------------------------------------------------------------------------
def test_plot1_parameters_match_root_xml_shared_data_and_maple_table():
    from maple.core.parameters.water_coupling import MAHLERAN_1_2_1_CLASS_DIAMETERS_M

    xml_path = MAHLERAN_ROOT / "mahleran_input.xml"
    shared = MAHLERAN_ROOT / "src" / "Program_Control" / "shared_data.f90"
    if not xml_path.is_file() or not shared.is_file():
        pytest.skip("MAHLERAN reference tree not available")
    root = ET.parse(xml_path).getroot()

    def per_class(tag):
        node = root.find(tag)
        return tuple(float(node.find(f"phi_{k}").text) for k in range(1, 7))

    p = plot1_sediment_parameters()
    assert tuple(p.raindrop_a) == per_class("Raindrop_detachment_a_parameter_size")
    assert tuple(p.raindrop_b) == per_class("Raindrop_detachment_b_parameter_size")
    assert tuple(p.raindrop_c) == per_class("Raindrop_detachment_c_parameter_size")
    np.testing.assert_allclose(p.raindrop_max_depth_per_reference_s_m,
                               np.array(per_class("Raindrop_detachment_max_parameter_size")) * 1e-3)
    assert p.particle_density_kg_m3 == float(root.find("particle_density").get("value")) * 1000.0
    assert p.flow_detachment_depth_scale_m == float(root.find("active_layer_sensitivity").get("value")) * 1e-3
    assert root.find("KE_model_type").get("value") == "2" and p.ke_model == "verstraeten_exp"
    assert float(root.find("time_step").get("value")) == p.reference_interval_s
    text = shared.read_text(errors="replace")
    radii = re.search(r"data radius\s*/\s*([^/]+)/", text).group(1)
    radii = tuple(float(x.strip().replace("d", "e")) for x in radii.split(","))
    assert radii == LEGACY_CLASS_RADII_M
    np.testing.assert_allclose(p.diameter_m, np.array(MAHLERAN_1_2_1_CLASS_DIAMETERS_M))
    spq = re.search(r"data spq\s*/\s*([^/]+)/", text).group(1)
    spq = tuple(float(x.strip().replace("d", "e")) for x in spq.split(","))
    assert spq == LEGACY_RAINDROP_DEPTH_ATTENUATION_PER_CM
    assert "data viscosity / 1.003d-6 /" in text and p.kinematic_viscosity_m2_s == 1.003e-6
    assert "v_soil (phi, im, jm) = 0.9d0 * v_soil (phi, im, jm)" in (
        MAHLERAN_ROOT / "src" / "Subroutines_Sediment" / "update_sediment_flow.for").read_text(errors="replace")


def test_derived_constants_match_initialize_values():
    p = plot1_sediment_parameters()
    assert p.sigma == pytest.approx(1.65) and p.excess_density_kg_m3 == pytest.approx(1650.0)
    assert p.bagnold_density_scale == pytest.approx(1.0)
    legacy = Legacy(p, ke_model_type=2, literal_veg=True)
    np.testing.assert_allclose(p.settling_velocity_m_s, legacy.settling, rtol=1e-14)
    assert p.dstar_const_per_m == pytest.approx(legacy.dstar_const, rel=1e-14)
    van_rijn = plot1_sediment_parameters(dstar_convention="van_rijn")
    assert van_rijn.dstar_const_per_m == pytest.approx(((1.65 * 9.81) / 1.003e-6 ** 2) ** (1 / 3), rel=1e-14)
    assert p.summary()["recession_timescale_s"] == pytest.approx(-1.0 / math.log(0.9))


# --- d50 ---------------------------------------------------------------------------------------------
def test_d50_matches_legacy_transcription_including_edges(params):
    legacy = Legacy(params, ke_model_type=2, literal_veg=True)
    rng = np.random.default_rng(1)
    cases = [rng.dirichlet(np.ones(NC)) for _ in range(50)]
    cases += [
        np.array([0.6, 0.1, 0.1, 0.1, 0.05, 0.05]),  # crossing in class 1
        np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),  # exactly 0.5 at class 1
        np.array([0.2, 0.3, 0.5, 0.0, 0.0, 0.0]),  # exactly 0.5 at class 2
        np.array([0.1, 0.1, 0.1, 0.1, 0.0, 0.0]),  # sums below 0.5 -> D6
        np.zeros(NC),  # empty cell -> D6
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    fractions = np.stack(cases)
    d50 = median_diameter_m(fractions, params.diameter_m)
    expected = np.array([legacy.d50(f) for f in cases])
    np.testing.assert_allclose(d50, expected, rtol=1e-14)
    assert d50[-3] == params.diameter_m[-1] and d50[-4] == params.diameter_m[-1]
    assert d50[-1] == params.diameter_m[0] * 0.5
    # From current holdings, not a static map: scaling the masses changes nothing, composition does.
    step_a = run(params, depth=[0.004], velocity=[0.1], rain_mm_h=[100.0], veg_fraction=[0.0], slope=0.05,
                 fractions_by_cell=[cases[0]], total_kg=1.0)
    step_b = run(params, depth=[0.004], velocity=[0.1], rain_mm_h=[100.0], veg_fraction=[0.0], slope=0.05,
                 fractions_by_cell=[cases[0]], total_kg=1e-3)
    step_c = run(params, depth=[0.004], velocity=[0.1], rain_mm_h=[100.0], veg_fraction=[0.0], slope=0.05,
                 fractions_by_cell=[cases[1]])
    assert step_a.d50_m[0, 0] == pytest.approx(expected[0], rel=1e-12)
    assert step_b.d50_m[0, 0] == pytest.approx(step_a.d50_m[0, 0], rel=1e-13)  # holdings scale, not composition
    assert abs(step_c.d50_m[0, 0] - step_a.d50_m[0, 0]) > 1e-6 * step_a.d50_m[0, 0]


# --- detachment laws against the transcription -----------------------------------------------------------
PLOT1_FRACTIONS = [0.05, 0.20, 0.25, 0.30, 0.15, 0.05]


def cells_diffuse():
    return [
        {"depth": 0.004, "velocity": 0.1, "rain_mm_h": 138.1068, "veg": 0.0, "slope": 0.05, "fractions": PLOT1_FRACTIONS},
        {"depth": 0.001, "velocity": 0.05, "rain_mm_h": 20.0, "veg": 0.57, "slope": 0.02, "fractions": PLOT1_FRACTIONS},
        {"depth": 0.0005, "velocity": 0.03, "rain_mm_h": 5.0, "veg": 1.0, "slope": 0.12, "fractions": [0.4, 0.4, 0.2, 0, 0, 0]},
        {"depth": 0.0032, "velocity": 0.08, "rain_mm_h": 60.0, "veg": 0.21, "slope": 0.03, "fractions": [0.2, 0.0, 0.3, 0.3, 0.2, 0.0]},
    ]


@pytest.mark.parametrize("ke_model", ["verstraeten_exp", "wainwright_log"])
def test_raindrop_detachment_and_diffuse_transport_match_transcription(ke_model):
    p = plot1_sediment_parameters(ke_model=ke_model, ke_vegetation_form="legacy_literal")
    legacy = Legacy(p, ke_model_type=2 if ke_model == "verstraeten_exp" else 1, literal_veg=True)
    cells = cells_diffuse()
    step = run(p, depth=[c["depth"] for c in cells], velocity=[c["velocity"] for c in cells],
               rain_mm_h=[c["rain_mm_h"] for c in cells], veg_fraction=[c["veg"] for c in cells],
               slope=[c["slope"] for c in cells], fractions_by_cell=[c["fractions"] for c in cells])
    compare_with_legacy(p, step, legacy, cells)
    assert int(step.regime_counts["diffuse"]) == len(cells) * NC
    assert np.all(step.raindrop_pickup_kg[0] > 0.0) == np.all(np.asarray([c["fractions"] for c in cells]) > 0.0)
    assert not np.any(step.flow_pickup_kg)  # Re < 500 in every cell
    # Speed cap: the sediment never outruns the water.
    assert np.all(step.sediment_velocity_m_s[0] <= np.asarray([c["velocity"] for c in cells])[:, None] * (1 + 4 * EPS))


def test_pickup_is_a_rate_over_the_reference_interval_not_per_step(params):
    cells = cells_diffuse()[:2]
    common = {"depth": [c["depth"] for c in cells], "velocity": [c["velocity"] for c in cells],
                  "rain_mm_h": [c["rain_mm_h"] for c in cells], "veg_fraction": [c["veg"] for c in cells],
                  "slope": [c["slope"] for c in cells], "fractions_by_cell": [c["fractions"] for c in cells]}
    one = run(params, dt=1.0, **common)
    quarter = run(params, dt=0.25, **common)
    np.testing.assert_allclose(4.0 * quarter.requested_pickup_kg, one.requested_pickup_kg, rtol=4 * EPS)
    # The legacy per-step form `X / dt` then `* dt` would pick up `X` per step,
    # i.e. four times the physical rate at dt = 0.25 s; not reproduced.
    assert quarter.requested_pickup_kg[0, 0, 1] < one.requested_pickup_kg[0, 0, 1]
    # Velocity and distance do not depend on dt.
    np.testing.assert_array_equal(quarter.sediment_velocity_m_s, one.sediment_velocity_m_s)
    np.testing.assert_array_equal(quarter.deposition_rate_per_m, one.deposition_rate_per_m)
    other = plot1_sediment_parameters(ke_vegetation_form="legacy_literal", reference_interval_s=2.0)
    two = run(other, dt=1.0, **common)
    np.testing.assert_allclose(2.0 * two.requested_pickup_kg, one.requested_pickup_kg, rtol=4 * EPS)


def test_low_rain_energy_is_floored_at_zero_without_nan():
    p = plot1_sediment_parameters(ke_model="wainwright_log")
    # 0.01 mm/h: 11.9 + 8.73 log10(0.01) < 0 -> the legacy raises a negative base to a real power.
    # The floor lies at 10^(-11.9/8.73) = 0.04334 mm/h; 0.05 mm/h is just above it.
    step = run(p, depth=[0.004, 0.004], velocity=[0.1, 0.1], rain_mm_h=[0.01, 0.05], veg_fraction=[0.0, 0.0],
               slope=0.05, fractions_by_cell=[PLOT1_FRACTIONS] * 2)
    assert np.all(np.isfinite(step.requested_pickup_kg)) and not np.any(step.requested_pickup_kg[0, 0])
    assert step.rain_energy_j_m2_mm[0, 0] == 0.0 and step.rain_energy_flux_j_m2_s[0, 0] == 0.0
    # Zero rain energy: no diffuse transport capacity -> the pool settles, nothing travels.
    assert np.all(step.settle_mask[0, 0]) and not np.any(step.law_applies[0, 0])
    assert not np.any(step.sediment_velocity_m_s[0, 0]) and not np.any(step.deposition_rate_per_m[0, 0])
    assert step.rain_energy_j_m2_mm[0, 1] > 0.0 and np.all(step.law_applies[0, 1])
    assert int(step.regime[0, 0, 0]) == REGIME_CODES["diffuse"]


def test_vegetation_forms_intended_reduces_energy_literal_reproduces_fortran():
    intended = plot1_sediment_parameters(ke_vegetation_form="intended")
    literal = plot1_sediment_parameters(ke_vegetation_form="legacy_literal")
    kwargs = {"depth": [0.003, 0.003], "velocity": [0.1, 0.1], "rain_mm_h": [100.0, 100.0], "veg_fraction": [0.0, 0.8],
                  "slope": 0.05, "fractions_by_cell": [PLOT1_FRACTIONS] * 2}
    a, b = run(intended, **kwargs), run(literal, **kwargs)
    r2 = 100.0 / 3.6e3
    bare = 29.0 - 20.88 * math.exp(-180.0 * r2)
    assert a.rain_energy_j_m2_mm[0, 0] == pytest.approx(bare) and b.rain_energy_j_m2_mm[0, 0] == pytest.approx(bare)
    assert a.rain_energy_j_m2_mm[0, 1] == pytest.approx(bare * (1.0 - 8.1e-3 * 80.0))
    assert b.rain_energy_j_m2_mm[0, 1] == pytest.approx(29.0 - 20.88 * math.exp(-180.0 * r2) * (1.0 - 8.1e-3 * 80.0))
    assert a.rain_energy_j_m2_mm[0, 1] < a.rain_energy_j_m2_mm[0, 0]  # cover reduces energy
    assert b.rain_energy_j_m2_mm[0, 1] > b.rain_energy_j_m2_mm[0, 0]  # literal precedence: cover increases it
    assert np.all(a.requested_pickup_kg[0, 1] < a.requested_pickup_kg[0, 0])
    # Model 1 is unambiguous: both forms agree.
    log_a = plot1_sediment_parameters(ke_model="wainwright_log", ke_vegetation_form="intended")
    log_b = plot1_sediment_parameters(ke_model="wainwright_log", ke_vegetation_form="legacy_literal")
    np.testing.assert_array_equal(run(log_a, **kwargs).requested_pickup_kg, run(log_b, **kwargs).requested_pickup_kg)


def test_zero_rain_wet_low_reynolds_has_no_law_and_velocity_decays(params):
    prev = np.full((1, 1, NC), 0.02)
    kwargs = {"depth": [0.004], "velocity": [0.1], "rain_mm_h": [0.0], "veg_fraction": [0.3], "slope": 0.05,
                  "fractions_by_cell": [PLOT1_FRACTIONS]}
    one = run(params, dt=1.0, prev_v=prev, **kwargs)
    assert not np.any(one.requested_pickup_kg) and not np.any(one.law_applies) and not np.any(one.settle_mask)
    assert int(one.regime[0, 0, 0]) == REGIME_CODES["wet_no_law"]
    np.testing.assert_allclose(one.sediment_velocity_m_s, prev * 0.9, rtol=4 * EPS)  # legacy 0.9 per 1 s step
    assert not np.any(one.deposition_rate_per_m)  # the pool advects at the memory velocity, no deposition law
    half = run(params, dt=0.5, prev_v=prev, **kwargs)
    twice = run(params, dt=0.5, prev_v=half.sediment_velocity_m_s, **kwargs)
    np.testing.assert_allclose(twice.sediment_velocity_m_s, prev * 0.9, rtol=8 * EPS)
    np.testing.assert_allclose(recession_velocity(prev, 3.0), prev * 0.9 ** 3, rtol=8 * EPS)
    with pytest.raises(SedimentPhysicsError):
        recession_velocity(prev, 1.0, factor_per_reference_s=1.0)


def test_dry_cell_with_rain_has_no_wet_pickup_and_settles(params):
    prev = np.full((1, 1, NC), 0.01)
    step = run(params, depth=[0.0], velocity=[0.0], rain_mm_h=[120.0], veg_fraction=[0.1], slope=0.05,
               fractions_by_cell=[PLOT1_FRACTIONS], prev_v=prev)
    assert not np.any(step.requested_pickup_kg)  # direct splash deferred, no runoff -> no wet law
    assert np.all(step.settle_mask) and not np.any(step.law_applies)
    assert not np.any(step.sediment_velocity_m_s)  # settled pools do not travel
    assert int(step.regime_counts["dry"]) == NC and int(step.regime[0, 0, 3]) == REGIME_CODES["dry"]
    assert step.shear_velocity_m_s[0, 0] == 0.0 and step.reynolds_number[0, 0] == 0.0


def test_absent_class_gets_no_demand_and_d50_uses_the_others(params):
    fractions = [0.0, 0.5, 0.0, 0.5, 0.0, 0.0]
    step = run(params, depth=[0.004], velocity=[0.1], rain_mm_h=[100.0], veg_fraction=[0.0], slope=0.05,
               fractions_by_cell=[fractions])
    present = np.asarray(fractions) > 0.0
    assert not np.any(step.requested_pickup_kg[0, 0, ~present]) and np.all(step.requested_pickup_kg[0, 0, present] > 0.0)
    assert step.d50_m[0, 0] == pytest.approx(params.diameter_m[1])  # cumulative hits 0.5 exactly at class 2
    # The transport law is still defined for the absent class (nothing to move, but arriving mass would).
    assert np.all(step.law_applies[0, 0])
    empty = run(params, depth=[0.004], velocity=[0.1], rain_mm_h=[100.0], veg_fraction=[0.0], slope=0.05,
                fractions_by_cell=[np.zeros(NC)])
    assert not np.any(empty.requested_pickup_kg) and empty.d50_m[0, 0] == params.diameter_m[-1]


def test_demand_is_uncapped_by_holdings_depletion_ready(params):
    tiny = run(params, depth=[0.004], velocity=[0.1], rain_mm_h=[138.1], veg_fraction=[0.0], slope=0.05,
               fractions_by_cell=[PLOT1_FRACTIONS], total_kg=1e-9)
    big = run(params, depth=[0.004], velocity=[0.1], rain_mm_h=[138.1], veg_fraction=[0.0], slope=0.05,
              fractions_by_cell=[PLOT1_FRACTIONS], total_kg=1e3)
    np.testing.assert_array_equal(tiny.requested_pickup_kg, big.requested_pickup_kg)  # legacy: no fraction scaling
    assert np.all(tiny.requested_pickup_kg[0, 0] > 1e-9 * np.asarray(PLOT1_FRACTIONS))  # exceeds the holdings
    scaled = plot1_sediment_parameters(ke_vegetation_form="legacy_literal", raindrop_composition_scaling="fraction")
    s = run(scaled, depth=[0.004], velocity=[0.1], rain_mm_h=[138.1], veg_fraction=[0.0], slope=0.05,
            fractions_by_cell=[PLOT1_FRACTIONS])
    np.testing.assert_allclose(s.raindrop_pickup_kg[0, 0], big.raindrop_pickup_kg[0, 0] * np.asarray(PLOT1_FRACTIONS),
                               rtol=4 * EPS)


def test_legacy_cap_binds_only_for_phi2_in_raindrop_law(params):
    capped = plot1_sediment_parameters(ke_vegetation_form="legacy_literal", raindrop_max_depth_mm=(1e-9,) * 6)
    step = run(capped, depth=[0.004], velocity=[0.1], rain_mm_h=[138.1], veg_fraction=[0.0], slope=0.05,
               fractions_by_cell=[PLOT1_FRACTIONS])
    free = run(params, depth=[0.004], velocity=[0.1], rain_mm_h=[138.1], veg_fraction=[0.0], slope=0.05,
               fractions_by_cell=[PLOT1_FRACTIONS])
    rho = params.particle_density_kg_m3
    assert bool(step.legacy_cap_applied[0, 0, 1]) and not np.any(step.legacy_cap_applied[0, 0, [0, 2, 3, 4, 5]])
    assert step.requested_pickup_kg[0, 0, 1] == pytest.approx(PLOT1_FRACTIONS[1] * 1e-12 * AREA * rho)
    np.testing.assert_array_equal(step.requested_pickup_kg[0, 0, [0, 2, 3, 4, 5]], free.requested_pickup_kg[0, 0, [0, 2, 3, 4, 5]])


# --- regimes, flow detachment, concentrated and suspended transport -----------------------------------------
def cells_regimes():
    return [
        {"depth": 0.010, "velocity": 0.10, "rain_mm_h": 80.0, "veg": 0.2, "slope": 0.05, "fractions": PLOT1_FRACTIONS},  # Re ~ 1000, rain
        {"depth": 0.010, "velocity": 0.10, "rain_mm_h": 0.0, "veg": 0.2, "slope": 0.05, "fractions": PLOT1_FRACTIONS},  # Re ~ 1000, dry
        {"depth": 0.050, "velocity": 0.10, "rain_mm_h": 80.0, "veg": 0.2, "slope": 0.05, "fractions": PLOT1_FRACTIONS},  # Re ~ 5000
        {"depth": 0.030, "velocity": 0.30, "rain_mm_h": 0.0, "veg": 0.0, "slope": 0.10, "fractions": [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]},
        {"depth": 0.030, "velocity": 0.09, "rain_mm_h": 0.0, "veg": 0.0, "slope": 0.002, "fractions": [0, 0, 0, 0.1, 0.4, 0.5]},  # low power
        {"depth": 0.0051, "velocity": 0.10, "rain_mm_h": 0.0, "veg": 0.0, "slope": 0.05, "fractions": PLOT1_FRACTIONS},  # Re just > 500
        {"depth": 0.0049, "velocity": 0.10, "rain_mm_h": 0.0, "veg": 0.0, "slope": 0.05, "fractions": PLOT1_FRACTIONS},  # Re just < 500
        {"depth": 0.0251, "velocity": 0.10, "rain_mm_h": 50.0, "veg": 0.0, "slope": 0.05, "fractions": PLOT1_FRACTIONS},  # Re just > 2500
    ]


def test_regime_thresholds_flow_detachment_and_transport_match_transcription(params):
    legacy = Legacy(params, ke_model_type=2, literal_veg=True)
    cells = cells_regimes()
    step = run(params, depth=[c["depth"] for c in cells], velocity=[c["velocity"] for c in cells],
               rain_mm_h=[c["rain_mm_h"] for c in cells], veg_fraction=[c["veg"] for c in cells],
               slope=[c["slope"] for c in cells], fractions_by_cell=[c["fractions"] for c in cells])
    compare_with_legacy(params, step, legacy, cells)
    codes = step.regime[0]
    assert np.all(codes[0] == REGIME_CODES["transitional_rain"]) and np.all(codes[1] == REGIME_CODES["transitional_dry"])
    assert set(codes[2].tolist()) <= {REGIME_CODES["concentrated"], REGIME_CODES["suspended"]}
    assert REGIME_CODES["suspended"] in codes[2].tolist() and REGIME_CODES["concentrated"] in codes[2].tolist()
    assert np.all(codes[5] == REGIME_CODES["transitional_dry"]) and np.all(codes[6] == REGIME_CODES["wet_no_law"])
    assert np.all(codes[7] != REGIME_CODES["transitional_rain"])
    # Transitional rain: raindrop AND flow detachment; transitional dry: flow only.
    assert np.all(step.raindrop_pickup_kg[0, 0] > 0.0) and np.all(step.flow_pickup_kg[0, 0] > 0.0)
    assert not np.any(step.raindrop_pickup_kg[0, 1]) and np.all(step.flow_pickup_kg[0, 1] > 0.0)
    # Concentrated with no excess stream power: detached mass settles locally (settle mask), velocity 0.
    assert np.any(step.settle_mask[0, 4]) and not np.any(step.sediment_velocity_m_s[0, 4][step.settle_mask[0, 4]])
    # Suspended classes travel at the water velocity.
    susp = codes[2] == REGIME_CODES["suspended"]
    assert np.allclose(step.sediment_velocity_m_s[0, 2][susp], cells[2]["velocity"])
    assert int(step.regime_counts["suspended"]) == int(np.sum(codes == REGIME_CODES["suspended"]))
    assert np.all(step.pickup_probability <= 1.0) and np.all(step.pickup_probability >= 0.0)


def test_pickup_probability_limits_and_sign_singularity(params):
    p = params
    d = 0.02
    # theta = 0.196 exactly: p_const = 0 (legacy 0/0); continuous limit 0.5.
    theta_star = 0.196
    slope = theta_star * p.sigma * p.diameter_m[2] / d  # for class 3
    step = run(p, depth=[d, d, d], velocity=[0.3, 0.3, 0.3], rain_mm_h=[0.0] * 3, veg_fraction=[0.0] * 3,
               slope=[slope, 1e-9, 1.0], fractions_by_cell=[PLOT1_FRACTIONS] * 3)
    assert step.pickup_probability[0, 0, 2] == pytest.approx(0.5, rel=1e-9)
    assert np.all(np.isfinite(step.pickup_probability))
    assert np.all(step.pickup_probability[0, 1] < 1e-6)  # vanishing shear
    assert step.pickup_probability[0, 2, 0] > 0.99  # very high shear on the finest class
    assert np.all(np.diff(step.pickup_probability[0, 2]) <= 0.0)  # coarser classes are harder to pick up
    # Flow pickup is proportional to the class fraction (flow_detachment 35-36).
    step_half = run(p, depth=[d], velocity=[0.3], rain_mm_h=[0.0], veg_fraction=[0.0], slope=[1.0],
                    fractions_by_cell=[np.asarray(PLOT1_FRACTIONS) * 0.5 + np.array([0.5, 0, 0, 0, 0, 0])])
    ratio = step_half.flow_pickup_kg[0, 0, 1:] / step.flow_pickup_kg[0, 2, 1:]
    np.testing.assert_allclose(ratio, 0.5, rtol=1e-12)


def test_concentrated_conventions_distance_cap_and_bagnold_units():
    literal = plot1_sediment_parameters()
    formula = plot1_sediment_parameters(distance_convention="formula_mean")
    si = plot1_sediment_parameters(bagnold_depth_units="si_m")
    # Cell 1: u* = 0.22 m/s keeps the coarsest class below its suspension threshold (0.27 m/s)
    # while the excess stream power (~147 W/m2) pushes its Hassan distance past the 30 m cap.
    kwargs = {"depth": [0.03, 0.05], "velocity": [0.3, 3.0], "rain_mm_h": [0.0, 0.0], "veg_fraction": [0.0, 0.0],
                  "slope": [0.1, 0.1], "fractions_by_cell": [PLOT1_FRACTIONS] * 2}
    a, b, c = run(literal, **kwargs), run(formula, **kwargs), run(si, **kwargs)
    conc = (a.regime[0, 0] == REGIME_CODES["concentrated"]) & a.law_applies[0, 0]
    assert np.any(conc)
    La, Lb = 1.0 / a.deposition_rate_per_m[0, 0][conc], 1.0 / b.deposition_rate_per_m[0, 0][conc]
    uncapped = La < 30.0 * (1 - 1e-12)
    np.testing.assert_allclose(La[uncapped], 0.693 * Lb[uncapped], rtol=1e-12)
    # Explicit conversion helpers use ln 2, not 0.693.
    assert exponential_mean_from_median(median_from_exponential_mean(2.5)) == pytest.approx(2.5)
    assert median_from_exponential_mean(1.0) == pytest.approx(math.log(2.0))
    # Extreme flow: the coarsest class stays concentrated and the 30 m cap binds.
    assert int(a.regime[0, 1, 5]) == REGIME_CODES["concentrated"] and bool(a.law_applies[0, 1, 5])
    assert 1.0 / a.deposition_rate_per_m[0, 1, 5] == pytest.approx(30.0)
    assert 1.0 / b.deposition_rate_per_m[0, 1, 5] == pytest.approx(30.0)  # the cap applies after the convention
    # Bagnold threshold in SI depth is lower than the literal mm form (log10 of a 1000x smaller argument), so
    # excess stream power is larger and the mean travel distance is longer, on every concentrated class.
    Lc = 1.0 / c.deposition_rate_per_m[0, 0][conc]
    assert np.all(Lc >= La * (1 - 1e-12))
    # Sediment never outruns the water in the concentrated regime either.
    assert np.all(a.sediment_velocity_m_s[0, 0] <= 0.3 * (1 + 4 * EPS))


def test_suspended_transport_exponent_cap_keeps_distances_finite(params):
    step = run(params, depth=[5.0], velocity=[20.0], rain_mm_h=[0.0], veg_fraction=[0.0], slope=[1.0],
               fractions_by_cell=[PLOT1_FRACTIONS])  # stream power ~ 1e6 W/m2: spf would be ~7000
    assert np.all(step.regime[0, 0] == REGIME_CODES["suspended"])
    assert np.all(np.isfinite(step.deposition_rate_per_m)) and np.all(step.deposition_rate_per_m[0, 0] > 0.0)
    L = 1.0 / step.deposition_rate_per_m[0, 0]
    expected = 727.51805244 * math.exp(100.0) * np.exp(-6.12683698 * params.diameter_m * 1000.0) * 0.693
    np.testing.assert_allclose(L, expected, rtol=1e-12)
    assert np.all(step.sediment_velocity_m_s[0, 0] == 20.0)


# --- validation and immutability ----------------------------------------------------------------------------------
def test_inputs_are_unchanged_and_invalid_inputs_refused(params):
    n = 2
    grid = grid_for(n, 0.05)
    depth = np.array([[0.004, 0.0]])
    velocity = np.array([[0.1, 0.0]])
    rain = np.array([[100.0, 100.0]]) / 3.6e6
    veg = np.array([[0.3, 0.0]])
    mass = np.stack([holdings(PLOT1_FRACTIONS)] * n).reshape(1, n, NC)
    prev = np.zeros((1, n, NC))
    copies = [a.copy() for a in (depth, velocity, rain, veg, mass, prev)]
    step = sediment_physics_step(params, grid, depth, velocity, rain, veg, mass, prev, 1.0)
    assert step.requested_pickup_kg.shape == (1, n, NC) and step.regime.dtype == np.int8
    bad = [
        ({"depth_m": np.array([[np.nan, 0.0]])}, "finite"),
        ({"velocity_m_s": np.array([[-0.1, 0.0]])}, ">= 0"),
        ({"vegetation_cover_fraction": np.array([[1.5, 0.0]])}, "<= 1"),
        ({"active_layer_mass_kg": -mass}, ">= 0"),
        ({"rain_rate_m_per_s": rain[:, :1]}, "shape"),
        ({"previous_sediment_velocity_m_s": prev.astype(np.float32)}, "float64"),
        ({"dt_s": 0.0}, "dt_s"),
        ({"dt_s": -1.0}, "dt_s"),
    ]
    for override, match in bad:
        kwargs = {"depth_m": depth, "velocity_m_s": velocity, "rain_rate_m_per_s": rain, "vegetation_cover_fraction": veg,
                      "active_layer_mass_kg": mass, "previous_sediment_velocity_m_s": prev, "dt_s": 1.0}
        kwargs.update(override)
        with pytest.raises(SedimentPhysicsError, match=match):
            sediment_physics_step(params, grid, **kwargs)
    for before, after in zip(copies, (depth, velocity, rain, veg, mass, prev), strict=True):
        assert np.array_equal(before, after)
    with pytest.raises(SedimentPhysicsError):
        physics_grid(np.array([[0.05, -0.01]]), np.ones((1, 2), dtype=bool), AREA)
    # Inactive cells: no demand, zero velocity, regardless of inputs.
    grid2 = physics_grid(np.array([[0.05, 0.05]]), np.array([[True, False]]), AREA)
    step2 = sediment_physics_step(params, grid2, np.array([[0.004, 0.004]]), np.array([[0.1, 0.0]]), rain, veg, mass,
                                  prev, 1.0)
    assert not np.any(step2.requested_pickup_kg[0, 1]) and not np.any(step2.sediment_velocity_m_s[0, 1])
    with pytest.raises(SedimentPhysicsError, match="inactive"):
        sediment_physics_step(params, grid2, np.array([[0.004, 0.004]]), np.array([[0.1, 0.1]]), rain, veg, mass,
                              prev, 1.0)


def test_parameter_validation():
    with pytest.raises(SedimentPhysicsError, match="increase"):
        plot1_sediment_parameters(diameter_m=(1e-3,) * 6)
    with pytest.raises(SedimentPhysicsError, match="shape"):
        plot1_sediment_parameters(raindrop_a=(1.0, 2.0))
    with pytest.raises(SedimentPhysicsError, match="ke_model"):
        plot1_sediment_parameters(ke_model="other")
    with pytest.raises(SedimentPhysicsError, match="exceed 1"):
        plot1_sediment_parameters(particle_density_g_cm3=0.9)
    with pytest.raises(SedimentPhysicsError, match="recession"):
        plot1_sediment_parameters(recession_factor_per_reference_s=1.0)
    with pytest.raises(SedimentPhysicsError, match="distance_convention"):
        plot1_sediment_parameters(distance_convention="median")
    p = plot1_sediment_parameters()
    assert p.summary()["ke_model"] == "verstraeten_exp" and p.n_classes == NC
    assert not p.diameter_m.flags.writeable
    # Defaults preserve the literal source relationships for the matched benchmark.
    assert p.ke_vegetation_form == "legacy_literal" and p.distance_convention == "legacy_literal"
    assert p.dstar_convention == "legacy_sigma_minus_one" and p.bagnold_depth_units == "legacy_mm"
    assert p.raindrop_composition_scaling == "legacy_none"


def test_overflowing_inputs_are_refused_not_masked(params):
    """Codex reproducers: holdings whose per-cell total overflows became empty
    fractions (pickup 0) and an overflowing rain rate returned an infinite
    energy-flux diagnostic on a dry cell. Both are refused."""
    grid = grid_for(1, 0.03)
    huge_bed = np.full((1, 1, NC), 1.0e308)
    with pytest.raises(SedimentPhysicsError, match="overflow"):
        sediment_physics_step(params, grid, np.array([[0.001]]), np.array([[0.01]]), np.array([[1e-5]]),
                              np.array([[0.0]]), huge_bed, np.zeros((1, 1, NC)), 1.0)
    bed = np.ones((1, 1, NC))
    with pytest.raises(SedimentPhysicsError, match="non-finite"):
        sediment_physics_step(params, grid, np.array([[0.0]]), np.array([[0.0]]), np.array([[1e308]]),
                              np.array([[0.0]]), bed, np.zeros((1, 1, NC)), 1.0)
    # A huge but finite bed is accepted and yields the same fractions as a small one.
    big = sediment_physics_step(params, grid, np.array([[0.001]]), np.array([[0.01]]), np.array([[1e-5]]),
                                np.array([[0.0]]), np.full((1, 1, NC), 1.0e300), np.zeros((1, 1, NC)), 1.0)
    small = sediment_physics_step(params, grid, np.array([[0.001]]), np.array([[0.01]]), np.array([[1e-5]]),
                                  np.array([[0.0]]), np.ones((1, 1, NC)), np.zeros((1, 1, NC)), 1.0)
    np.testing.assert_allclose(big.requested_pickup_kg, small.requested_pickup_kg, rtol=1e-12)
    assert np.all(np.isfinite(big.rain_energy_flux_j_m2_s)) and np.all(np.isfinite(big.d50_m))
    assert big.d50_m[0, 0] == pytest.approx(small.d50_m[0, 0], rel=1e-12)


def test_recession_velocity_is_not_capped_at_the_water_velocity(params):
    """Legacy: v_soil decays from its last law value whatever the water does.
    A slowing water leaves a sediment memory faster than itself; the
    transport step must validate its own Courant number (documented)."""
    prev = np.full((1, 1, NC), 0.05)
    step = run(params, depth=[0.002], velocity=[0.01], rain_mm_h=[0.0], veg_fraction=[0.0], slope=0.05,
               fractions_by_cell=[PLOT1_FRACTIONS], prev_v=prev, dt=1.0)
    assert not np.any(step.law_applies)  # Re = 20, no rain: no law
    np.testing.assert_allclose(step.sediment_velocity_m_s, prev * 0.9, rtol=4 * EPS)
    assert np.all(step.sediment_velocity_m_s[0, 0] > 0.01)  # exceeds the water velocity, by design
    # Where a law applies the legacy caps at the water velocity, so the memory is reset below it.
    capped = run(params, depth=[0.002], velocity=[0.01], rain_mm_h=[100.0], veg_fraction=[0.0], slope=0.05,
                 fractions_by_cell=[PLOT1_FRACTIONS], prev_v=prev, dt=1.0)
    assert np.all(capped.law_applies) and np.all(capped.sediment_velocity_m_s[0, 0] <= 0.01 * (1 + 4 * EPS))


def test_physics_grid_from_routing_graph():
    from maple_syrup.routing import build_routing_graph

    z = np.repeat(np.arange(5, dtype=np.float64)[:, None] * 0.02, 3, axis=1)
    z[:, 0] += 1.0
    z[:, 2] += 1.0
    export = np.zeros(z.shape, dtype=bool)
    export[0, :] = True
    graph = build_routing_graph(z, export, np.full((3, 1), 5.0), 0.5)
    grid = physics_grid_from_graph(graph)
    assert grid.shape == (3, 1) and grid.cell_area_m2 == 0.25
    np.testing.assert_array_equal(grid.slope, graph.slope)
    assert np.all(grid.active)
