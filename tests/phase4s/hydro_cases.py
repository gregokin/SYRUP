"""Phase 4S (task phase4r_gpu_storm, task A) shared builders for the CUDA hydrology tests and benchmark.

Cases mirror `tests/phase7h/test_hydrology_prepared.build_case` (reference-accepted unsaturated forcing, see the notes
there) but never call the Numba-only `prepare_hydrology`: the oracle here is `storm.coupled_step` with the NumPy
`array` implementation (needs neither Numba nor a device), and the prepared CPU step is an optional second oracle.
A `Case` holds the HOST objects; `device_case` builds the CuPy twins from the very same arrays.
Nothing here was run by its author (file-only tools); Codex records results.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from test_routing import chain_full, make_graph, random_full, valley_full

from maple_syrup.infiltration import column_parameters, initial_soil_water_m
from maple_syrup.storm import StormControl, StormState, coupled_step, initial_state

WATER_RTOL = 2.0e-12
WATER_ATOL = 1.0e-14
ACT = (2, 2)  # an active cell of the masked 6 x 5 valley
INACT = (5, 0)  # its top-west corner, masked out


@dataclasses.dataclass
class Case:
    graph: object
    params: object
    state: StormState
    rate_on: np.ndarray
    spec: dict  # host arrays needed to rebuild device twins


@dataclasses.dataclass
class DevCase:
    graph: object
    params: object
    state: StormState
    rate_on: object


def _dims(arg: str, default: str) -> tuple[int, int]:
    ny, nx = (int(v) for v in (arg or default).split("x"))
    return ny, nx


def terrain(kind: str, rng):
    """(full elevation, friction, active mask or None). kinds: random[:NYxNX], valley[:NYxNX], valley_masked,
    plane:NX (3 rows, every row is ONE dependency level of exactly NX cells), chain[:N]."""
    name, _, arg = kind.partition(":")
    if name == "random":
        ny, nx = _dims(arg, "11x8")
        return random_full(rng, ny, nx), rng.uniform(5.0, 30.0, (ny, nx)), None
    if name == "valley":
        ny, nx = _dims(arg, "8x7")
        return valley_full(ny, nx), 5.0, None
    if name == "valley_masked":  # nothing drains into the two top corners, so masking them is a valid graph
        active = np.ones((6, 5), dtype=bool)
        active[-1, 0] = active[-1, -1] = False
        return valley_full(6, 5), 5.0, active
    if name == "plane":
        nx = int(arg)
        return np.repeat((np.arange(5, dtype=np.float64) * 0.015625)[:, None], nx + 2, axis=1), 5.0, None
    if name == "chain":
        return chain_full(int(arg or 3)), 5.0, None
    raise AssertionError(kind)


def build_case(seed: int, *, kind: str = "random", model: str = "pavement_hawkins", saturated: bool = False,
               adversarial: bool = False, rain_mm_h: float = 45.0) -> Case:
    rng = np.random.default_rng(seed)
    z, ff, active = terrain(kind, rng)
    graph = make_graph(z, ff=ff, active=active)
    shape = graph.shape

    def uni(lo, hi):
        return rng.uniform(lo, hi, shape)

    fields = {"suction_m": uni(0.0, 1e-3), "drainage_parameter": uni(0.0, 0.5), "theta_sat": uni(0.3, 0.5),
              "soil_thickness_m": uni(0.1, 0.5)}
    ksat = uni(1e-8, 3e-6)
    if saturated:  # near-full columns, no drainage: capacity is exactly Ksat (expm1 argument >> 40)
        ksat = np.full(shape, 1e-7)
        fields["suction_m"] = np.full(shape, 0.01)
        fields["drainage_parameter"] = np.zeros(shape)
    if model == "pavement_hawkins":
        fields["pavement_cover_fraction"] = uni(0.8, 1.0)
    mask = graph.active.copy()
    params = column_parameters(model=model, ksat_m_per_s=ksat, active_mask=mask, **fields)
    if saturated:
        soil = params.storage_max_m - 1e-6
    else:
        soil = initial_soil_water_m(params, uni(0.05, 0.95) * params.theta_sat)
    if adversarial:
        depth = np.where(graph.active, 1e-3, 0.3)
        rate = np.where(graph.active, 1e-5, 0.0)
    else:
        if saturated:
            wet = uni(0.0, 3e-3) * (rng.random(shape) > 0.3)
        else:
            wet = uni(1e-3, 3e-3)
        depth = np.where(graph.active, wet, 0.3)
        scale = np.where(graph.active, uni(0.5, 1.5) * (rng.random(shape) > 0.2), 0.0)
        rate = scale * (rain_mm_h / 3.6e6)
        if saturated:
            rate = np.where(graph.active, 1e-5, 0.0)
        else:  # J < P on every raining cell: pure no run-on, never the one-ulp hpre/h* guard (see phase 7h)
            rate = np.where(scale > 0.0, uni(1.0, 1.5) * 6e-5, 0.0)
    state = initial_state(graph, depth, soil)
    spec = {"z": z, "ff": ff, "active": active, "ksat": ksat, "fields": fields, "mask": mask, "model": model}
    return Case(graph, params, state, rate, spec)


def device_case(case: Case, cp) -> DevCase:
    """CuPy twins built from the same host arrays (graph through the same builder, parameters through
    `column_parameters`, the state uploaded from the host state so both sides start from identical bits)."""
    spec = case.spec
    graph = make_graph(spec["z"], ff=spec["ff"], active=spec["active"], xp=cp)
    params = column_parameters(model=spec["model"], ksat_m_per_s=cp.asarray(spec["ksat"]),
                               active_mask=cp.asarray(spec["mask"]),
                               **{k: cp.asarray(v) for k, v in spec["fields"].items()})
    st = case.state
    state = StormState(st.t_s, cp.asarray(st.depth_m), cp.asarray(st.soil_water_m), cp.asarray(st.discharge_m2_s))
    return DevCase(graph, params, state, cp.asarray(case.rate_on))


def upload_state(state: StormState, cp) -> StormState:
    return StormState(state.t_s, cp.asarray(state.depth_m), cp.asarray(state.soil_water_m),
                      cp.asarray(state.discharge_m2_s))


def ref_control(**kw) -> StormControl:
    return StormControl(implementation="array", **kw)


def ref_step(case: Case, rate, state, dt, **control):
    """The accepted reference: NumPy array hydrology (no Numba needed)."""
    return coupled_step(case.graph, case.params, rate, state, dt, ref_control(**control))


# --- comparison ---------------------------------------------------------------------------------------------------
def host(x):
    """Host NumPy view of a NumPy/CuPy array or scalar."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "get"):
        return np.asarray(x.get())
    return np.asarray(x)


def compare(ref, new, *, exact_arrays: bool = False, path: str = "step", skip=("implementation",)) -> None:
    """Every public field of a CoupledStep-like dataclass: arrays/scalars within rtol 2e-12 / atol 1e-14 (arrays
    bitwise when `exact_arrays`); integers, bools, strings exact; None stays None. `new` may live on a device."""
    if dataclasses.is_dataclass(ref) and not isinstance(ref, type):
        assert type(new) is type(ref), path
        for f in dataclasses.fields(ref):
            if f.name in skip:
                continue
            compare(getattr(ref, f.name), getattr(new, f.name), exact_arrays=exact_arrays, path=f"{path}.{f.name}",
                    skip=skip)
        return
    if ref is None:
        assert new is None, path
        return
    if isinstance(ref, (str, bool)):
        assert new == ref, path
        return
    a, b = host(ref), host(new)
    assert a.shape == b.shape, f"{path}: shape {a.shape} vs {b.shape}"
    if a.dtype.kind in "iub" or isinstance(ref, (int, np.integer)) or (exact_arrays and a.ndim > 0):
        np.testing.assert_array_equal(b, a, err_msg=path)
    else:
        np.testing.assert_allclose(b, a, rtol=WATER_RTOL, atol=WATER_ATOL, err_msg=path)


def all_arrays(step) -> dict:
    c, r, s = step.column, step.route, step.state
    return {"state.depth": s.depth_m, "state.soil": s.soil_water_m, "state.q": s.discharge_m2_s,
            "col.depth": c.depth_m, "col.soil": c.soil_water_m, "col.rain": c.rain_m, "col.intake": c.intake_m,
            "col.return": c.saturation_return_m, "col.drainage": c.drainage_m,
            "route.depth": r.depth_m, "route.flow": r.flow_depth_m, "route.q": r.discharge_m2_s,
            "route.velocity": r.velocity_m_s, "route.inflow": r.inflow_m2_s, "route.old_q": r.old_discharge_m2_s,
            "route.old_inflow": r.old_inflow_m2_s, "route.face": r.face_volume_m3}


def scalar_fields(step) -> dict:
    r = step.route
    return {k: getattr(r, k) for k in ("export_m3", "outlet_discharge_m3_s", "storage_change_m3",
                                       "budget_residual_m3", "max_courant_old", "max_courant_new",
                                       "max_constitutive_residual_m", "max_cell_balance_residual_m")} | {
        "n_no_runon": step.n_no_runon, "n_partial_runon": step.n_partial_runon,
        "n_complete_runon": step.n_complete_runon}
