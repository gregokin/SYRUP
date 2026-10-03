"""Builders for the hydraulic-candidate tests (not a test module). Nothing here was run by its author (file-only tools)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from hydro_cases import terrain
from test_routing import make_graph

from maple_syrup.experimental_hydrology import (
    CpuHydraulicSolver,
    HydraulicControl,
    build_local_inertial_geometry,
    open_faces_from_graph,
)
from maple_syrup.infiltration import column_parameters
from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule, rainfall_field

DX = 0.5  # `make_graph` default cell size


def param_arrays(shape, *, ksat=0.0, theta=0.4, thickness=0.3, suction=0.01, drainage=0.0, cover=0.9):
    def full(v):
        return np.full(shape, float(v))

    return {"ksat_m_per_s": full(ksat), "suction_m": full(suction), "drainage_parameter": full(drainage),
            "theta_sat": full(theta), "soil_thickness_m": full(thickness), "_cover": full(cover)}


def make_params(arrays, active, *, model="fixed_ksat", xp=np):
    kw = {k: xp.asarray(v) for k, v in arrays.items() if not k.startswith("_")}
    if model == "pavement_hawkins":
        kw["pavement_cover_fraction"] = xp.asarray(arrays["_cover"])
    return column_parameters(model=model, active_mask=xp.asarray(active), **kw)


def build(kind: str, method: str, *, closed: bool = False, control: HydraulicControl | None = None, ksat: float = 0.0,
          soil_fraction: float = 0.0, model: str = "fixed_ksat", depth="wet", seed: int = 0, drainage: float = 0.0,
          theta: float = 0.4, thickness: float = 0.3):
    """Host case: graph, column parameters (no infiltration by default: ksat 0 makes the accepted column the identity on
    depth), geometry (local inertia; `closed=True` opens no face), CPU solver and an initial state."""
    rng = np.random.default_rng(seed)
    z_full, ff, active = terrain(kind, rng)
    graph = make_graph(z_full, ff=ff, active=active)
    shape = graph.shape
    arrays = param_arrays(shape, ksat=ksat, drainage=drainage, theta=theta, thickness=thickness)
    params = make_params(arrays, graph.active, model=model)
    geometry = None
    if method == "local_inertial":
        faces = [] if closed else open_faces_from_graph(graph)
        geometry = build_local_inertial_geometry(z_full, np.asarray(graph.active), np.asarray(graph.friction_factor),
                                                 graph.dx_m, faces)
    solver = CpuHydraulicSolver(method, graph, params, geometry=geometry, control=control)
    if isinstance(depth, str):
        depth_arr = np.where(graph.active, rng.uniform(1e-3, 3e-3, shape), 0.0) if depth == "wet" else np.zeros(shape)
    else:
        depth_arr = np.array(depth, dtype=np.float64)
    soil = soil_fraction * np.asarray(params.storage_max_m)
    state = solver.initial_state(depth_arr, soil)
    return SimpleNamespace(kind=kind, method=method, solver=solver, graph=graph, params=params, geometry=geometry,
                           z_full=z_full, ff=ff, active=active, shape=shape, state=state, arrays=arrays, model=model,
                           control=control, closed=closed)


def device_twin(cs, cp):
    """CuPy twins of a host case built from the same arrays; returns a namespace with the CUDA solver and state."""
    from maple_syrup.experimental_cuda import CudaHydraulicSolver

    graph = make_graph(cs.z_full, ff=cs.ff, active=cs.active, xp=cp)
    params = make_params(cs.arrays, np.asarray(cs.graph.active), model=cs.model, xp=cp)
    solver = CudaHydraulicSolver(cs.method, graph, params, geometry=cs.geometry, control=cs.control)
    st = cs.state
    state = type(st)(st.t_s, cp.asarray(st.depth_m), cp.asarray(st.soil_water_m),
                     None if st.qx_m2_s is None else cp.asarray(st.qx_m2_s),
                     None if st.qy_m2_s is None else cp.asarray(st.qy_m2_s))
    return SimpleNamespace(solver=solver, graph=graph, params=params, state=state)


def schedule(edges, mm_per_h):
    return RainfallSchedule(edges_s=list(edges), intensity_mm_per_h=list(mm_per_h),
                            provenance=RainfallProvenance(kind="constant"))


def field(cs, xp=np, scale=1.0):
    ny, nx = cs.shape
    mult = np.where(cs.graph.active, float(scale), 0.0)
    return rainfall_field(ny, nx, scale=xp.asarray(mult))


def rain_rate(cs, mm_per_h, xp=np):
    ny, nx = cs.shape
    return xp.asarray(np.where(cs.graph.active, mm_per_h / 3.6e6, 0.0).reshape(ny, nx))


def host(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "get"):
        return np.asarray(x.get())
    return np.asarray(x)


def budget(result, state0, area=DX * DX):
    """(residual, MAPLE-derived bound, volumes) of the event water identity
    surface_final + soil_final + drainage + export = surface_initial + soil_initial + rain, in m3."""
    from maple_syrup.conservation import volume_roundoff_bound_m3

    def vol(a):
        return float(np.sum(host(a))) * area

    initial = vol(state0.depth_m) + vol(state0.soil_water_m)
    rain, drain = vol(result.cumulative_rain_m), vol(result.cumulative_drainage_m)
    export = float(host(result.cumulative_export_m3))
    final = vol(result.state.depth_m) + vol(result.state.soil_water_m)
    n_cells = host(result.state.depth_m).size
    bound = volume_roundoff_bound_m3(4 * n_cells * max(result.n_accepted_steps, 1) + 7,
                                     max(initial, rain, drain, export, final, 1e-300))
    return final + drain + export - initial - rain, bound, {"initial": initial, "rain": rain, "drain": drain,
                                                           "export": export, "final": final}
