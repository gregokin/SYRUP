"""S2-B2.c review02 bounded combined-intake suite (author-written, CPU only). Comparison targets are independent arithmetic (exact FP64 identities and Decimal), Root's immutable minimal single-column refusal controls
(`root_s2_b2_c_minimal_refusal_controls.json`, never edited) and room-limited cases found with the PRE-review02 source (`room_limited_controls.json`, new fixture), plus the frozen pre-review02 sources under
`s2_b2_c_review02_pre_sources/` for the conditional-effect accounting. No expected file and no seeded fixture of any earlier suite is edited."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
CANDIDATE = TESTS.parent
TASK = CANDIDATE.parent
PRE = TASK / "s2_b2_c_review02_pre_sources"  # task-only: the immutable pre-review02 sources (old-vs-new comparisons skip explicitly when absent)
#: Root's immutable ORIGINAL (task-only; never edited). The suite does NOT depend on it: the REQUIRED trigger data is the packaged copy below.
ROOT_ORIGINAL = TASK / "root_s2_b2_c_minimal_refusal_controls.json"
MINIMAL_TRIGGERS = HERE / "minimal_trigger_controls.json"  # packaged, self-contained required fixture (exact payload + attribution + digests)
ROOM_CONTROLS = HERE / "room_limited_controls.json"  # packaged required fixture (room-limited cases found with the pre-review02 source)

REQUIRED_LABELS = ("L3_N128_all_shared", "L5_N257_scalar_thetas")
_CONTROL_KEYS = ("label", "cell", "depth_edges_m", "theta_residual", "theta_field_capacity", "theta_saturation", "conductivity_m_s", "suction_m", "vegetation_factor", "water_m", "wetting_m",
                 "cumulative_intake_m", "pool_m", "standing_m", "incoming_rate_m_s", "dt_s", "t_s", "area_m2", "expected_old_error", "old_raw")
_ROOM_KEYS = ("kind", "cell", "depth_edges_m", "theta_residual", "theta_field_capacity", "theta_saturation", "conductivity_m_s", "suction_m", "vegetation_factor", "water_m", "wetting_m",
              "cumulative_intake_m", "pool_m", "standing_m", "incoming_rate_m_s", "dt_s", "old_intake", "old_pool_out", "room", "pre", "post")


class FixtureError(RuntimeError):
    """A REQUIRED packaged regression fixture is missing or malformed: a hard failure, never an empty parameterisation."""


def _read_required(path: Path, what: str):
    if not path.is_file():
        raise FixtureError(f"required regression fixture {what} is missing: {path}")
    try:
        return json.loads(path.read_text())
    except ValueError as exc:
        raise FixtureError(f"required regression fixture {what} is not valid JSON ({path}): {exc}") from exc


def _check_records(records, keys, what):
    if not isinstance(records, list) or not records:
        raise FixtureError(f"{what}: a non-empty list of controls is required")
    for rec in records:
        missing = [k for k in keys if not isinstance(rec, dict) or k not in rec]
        if missing:
            raise FixtureError(f"{what}: a control lacks the required keys {missing}")
    return records


def load_minimal_triggers(path: Path | None = None) -> list[dict]:
    """The two minimal pool-limited trigger controls from the PACKAGED fixture; missing/malformed/incomplete data raises `FixtureError` (the whole suite then fails visibly)."""
    d = _read_required(MINIMAL_TRIGGERS if path is None else path, "minimal_trigger_controls.json")
    if not isinstance(d, dict) or "fixture" not in d or "controls" not in d:
        raise FixtureError("minimal_trigger_controls.json: the `fixture` attribution block and the `controls` payload are required")
    controls = _check_records(d["controls"], _CONTROL_KEYS, "minimal_trigger_controls.json")
    if tuple(c["label"] for c in controls) != REQUIRED_LABELS:
        raise FixtureError(f"minimal_trigger_controls.json: the required controls {REQUIRED_LABELS} are not exactly present (found {[c['label'] for c in controls]})")
    return controls


def load_room_controls(path: Path | None = None) -> list[dict]:
    return _check_records(_read_required(ROOM_CONTROLS if path is None else path, "room_limited_controls.json"), _ROOM_KEYS, "room_limited_controls.json")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


from maple_syrup import matrix_infiltration as mi  # noqa: E402
from maple_syrup import spatial_matrix_infiltration as sm  # noqa: E402
from maple_syrup import spatial_soil_layers as sp  # noqa: E402
from maple_syrup import subsurface as ss  # noqa: E402

BIG = 1 << 33
RTOL, ATOL = 2e-12, 1e-14


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def root_controls():
    return load_minimal_triggers()


def room_controls():
    return load_room_controls()


def control_id(c):
    return c.get("label") or f"{c['kind']}-cell{c['cell']}"


def _lay(c):
    return ss.soil_layers(np.array(c["depth_edges_m"]), theta_residual=np.array(c["theta_residual"]), theta_field_capacity=np.array(c["theta_field_capacity"]), theta_saturation=np.array(c["theta_saturation"]),
                          theta_evaporation_stress=0.5, diffusivity_m2_s=0.0)


def run_common(c, backend, *, pool=None, wetting=None, water=None, dt=None, active=True, area=None):
    """The accepted COMMON operator on the control's single column (same backend). Returns (result, inputs dict)."""
    area = c.get("area_m2", 4.0) if area is None else area
    col = ss.prepare_column(_lay(c), np.array([active]), area, backend=backend)
    op = mi.prepare_matrix(col, conductivity_m_s=c["conductivity_m_s"], suction_m=c["suction_m"], vegetation_factor=c["vegetation_factor"])
    w = np.array(c["water_m"] if water is None else water, dtype=np.float64).reshape(-1, 1)
    hist = mi.wetting_history(op, wetting_m=np.array([c["wetting_m"] if wetting is None else wetting]), cumulative_intake_m=np.array([c["cumulative_intake_m"]]))
    state = ss.column_state(col, w.copy(), c.get("t_s", 3.0))
    p = np.array([c["pool_m"] if pool is None else pool])
    st, rt = np.array([c["standing_m"]]), np.array([c["incoming_rate_m_s"]])
    ins = [np.array(x, copy=True) for x in (p, st, rt, w, hist.wetting_m, hist.cumulative_intake_m)]
    r = mi.matrix_step(op, p, st, rt, state, hist, c["dt_s"] if dt is None else dt)
    for a, b in zip((p, st, rt, w, hist.wetting_m, hist.cumulative_intake_m), ins, strict=True):
        np.testing.assert_array_equal(a, b)  # inputs untouched
    return r


def run_spatial(c, backend, *, pool=None, wetting=None, water=None, dt=None, active=True, area=None):
    area = c.get("area_m2", 4.0) if area is None else area
    spec = sp.spatial_layer_spec(np.array(c["depth_edges_m"]), theta_residual=np.array(c["theta_residual"]), theta_field_capacity=np.array(c["theta_field_capacity"]),
                                 theta_saturation=np.array(c["theta_saturation"]), theta_evaporation_stress=0.5, diffusivity_m2_s=0.0, n_cells=1)
    col = sp.prepare_spatial_column(spec, np.array([active]), area, backend=backend, memory_budget_bytes=BIG)
    op = sm.prepare_spatial_matrix(col, conductivity_m_s=c["conductivity_m_s"], suction_m=c["suction_m"], vegetation_factor=c["vegetation_factor"], memory_budget_bytes=BIG)
    hist = sm.spatial_matrix_history(op, wetting_m=np.array([c["wetting_m"] if wetting is None else wetting]), cumulative_intake_m=np.array([c["cumulative_intake_m"]]), memory_budget_bytes=BIG)
    w = np.ascontiguousarray(np.array(c["water_m"] if water is None else water, dtype=np.float64).reshape(-1, 1))
    state = sp.spatial_column_state(col, w, c.get("t_s", 3.0), memory_budget_bytes=BIG)
    p = np.array([c["pool_m"] if pool is None else pool])
    st, rt = np.array([c["standing_m"]]), np.array([c["incoming_rate_m_s"]])
    ins = [np.array(x, copy=True) for x in (p, st, rt, w, hist.wetting_m, hist.cumulative_intake_m)]
    r = sm.spatial_matrix_step(op, p, st, rt, state, hist, c["dt_s"] if dt is None else dt, memory_budget_bytes=BIG)
    for a, b in zip((p, st, rt, w, hist.wetting_m, hist.cumulative_intake_m), ins, strict=True):
        np.testing.assert_array_equal(a, b)
    return r


def room_of(c, water=None):
    """The top-layer room exactly as the bodies form it: `max(theta_sat0 * dz0 - W0, 0)` with the rounded product first."""
    e = c["depth_edges_m"]
    dz0 = np.float64(e[1]) - np.float64(e[0])
    sat0 = np.float64(c["theta_saturation"][0]) * dz0
    w0 = np.float64((c["water_m"] if water is None else water)[0])
    return max(float(sat0 - w0), 0.0)


def all_controls():
    return [("root", c) for c in root_controls()] + [("room", c) for c in room_controls()]
