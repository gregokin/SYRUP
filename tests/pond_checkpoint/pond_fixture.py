"""Bounded ACTUAL-MAPLE fixtures for the pond checkpoint tests (nothing here was run by its author).

A small dense bed from `probe.build_minimal_state`, the actual `BedState` / `BedContext` / committed topography /
availability constructors (as `tests/phase5/test_sediment_bed.py`), a real `PondTopology` over that terrain with explicit
south outlets, the real column / sediment parameters and rain schedule, and a real `PondEvent`. No physics is faked and no
native call is patched. Every case builds a FRESH event from the ORIGINAL topology (as a restarting process would)."""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Any

import numpy as np

from maple_syrup.experimental_storm import ExperimentalControl
from maple_syrup.infiltration import column_parameters, initial_soil_water_m
from maple_syrup.pond_event import PondCompletionPolicy, PondEvent, PondEventControl
from maple_syrup.pond_network import build_pond_topology
from maple_syrup.probe import build_minimal_state
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_bed import BedContext, BedState, apply_bed_demand, with_water
from maple_syrup.sediment_physics import plot1_sediment_parameters

IDENTITY = {"case": "controlled actual MAPLE pond bed", "forcing": "constant 60 mm/h", "source": "test", "options": {"dt": 0.1}}
DT = 0.1


@dataclass
class Case:
    bed: BedState
    ctx: BedContext
    soil: Any
    topology: Any
    column: Any
    sediment: Any
    schedule: Any
    control: PondEventControl
    policy: PondCompletionPolicy
    xp: ModuleType
    mobile_cell: tuple | None

    def event(self, **overrides) -> PondEvent:
        """A fresh event from the ORIGINAL topology (what a restarting process builds)."""
        args = {"ctx": self.ctx, "column": self.column, "sediment": self.sediment, "schedule": self.schedule,
                "topology": self.topology, "control": self.control, "policy": self.policy}
        args.update(overrides)
        return PondEvent(args["ctx"], args["column"], args["sediment"], args["schedule"], args["topology"],
                         control=args["control"], policy=args["policy"])


def make_case(*, xp: ModuleType = np, ny: int = 3, nx: int = 4, pickup_kg_per_class: float | None = None,
              mobile_kg: float | None = None, ksat: float = 2e-6, rain_mm_h: float = 60.0, rain_end_s: float = 100.0,
              backend: str = "numpy", max_dt_s: float = 0.2, policy: PondCompletionPolicy | None = None,
              closed_lake: bool = False, mobile_cell: tuple = (2, 1, 2)) -> Case:
    """`pickup_kg_per_class`: a genuine native MAPLE pickup at one interior cell BEFORE the event starts (it leaves a real
    pending ledger and lowers that cell; 0.1 kg per class crosses MAPLE's 0.75 mm pending-elevation trigger, 0.0005 stays far
    below it). `mobile_kg`: a mobile load put into the native water state at one cell (canonical phase at the upstream face)."""
    from maple.core.backend import to_device, to_device_tree
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.types.sediment_availability import sediment_availability_state_from_active_layer
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.surface.topographic_commit import initial_committed_topography_state_from_physical_state
    from maple.water import WaterProcessDemand

    p = build_minimal_state(ny=ny, nx=nx)
    committed = initial_committed_topography_state_from_physical_state(p.voxel_column, p.active_layer, p.geometry,
                                                                       p.mass_resolution_kg)
    available = sediment_availability_state_from_active_layer(p.active_layer.mass_kg, np.ones_like(p.active_layer.mass_kg))
    bed = BedState(p.voxel_column, p.active_layer, p.water, p.ledger, committed, available)
    ctx = BedContext(p.geometry, p.grain_classes, p.mass_resolution_kg, TopographicCommitSpec(),
                     AvalanchingSpec(enabled=False), np.ones_like(p.active_layer.mass_kg))
    zc = np.array(committed_elevation_m(committed), dtype=np.float64)
    if pickup_kg_per_class is not None:
        demand = np.zeros_like(p.active_layer.mass_kg)
        demand[1, 2] = pickup_kg_per_class
        bed, _ = apply_bed_demand(bed, ctx, WaterProcessDemand(demand, np.zeros_like(demand)))
    depth = np.array(bed.water.depth_m, copy=True)
    if closed_lake:
        # water at rest in the lowest corner cell only (3 mm, below every neighbour's bed): no flow, no outlet
        depth[...] = 0.0
        depth[0, 0] = 0.003
    mobile = None
    cell = None
    if mobile_kg is not None:
        cell = mobile_cell  # row, col, class
        mobile = np.zeros_like(bed.water.mobile_mass_by_cell_class_kg)
        mobile[cell] = mobile_kg
    if mobile is not None or closed_lake:
        bed = with_water(bed, ctx, depth, mobile)
    mobile_cell = cell

    z = np.full((ny + 2, nx + 2), float(zc.max()) + 1.0)
    z[1:-1, 1:-1] = zc
    if closed_lake:
        topology = build_pond_topology(z, np.ones((ny, nx), bool), np.ones((ny, nx)), p.geometry.dx_m, [],
                                       allow_closed_components=True)
    else:
        z[0, 1:-1] = zc[0] - 0.02  # south ring lower than the south row: explicit outlets with a real positive drop
        topology = build_pond_topology(z, np.ones((ny, nx), bool), np.ones((ny, nx)), p.geometry.dx_m,
                                       [(0, c, -1, 0) for c in range(nx)])

    if xp is not np:
        bed = to_device_tree(bed, xp)
        ctx = replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, xp))
    full = lambda v: xp.full((ny, nx), v, dtype=np.float64)  # noqa: E731
    column = column_parameters(model="fixed_ksat", ksat_m_per_s=full(ksat), suction_m=full(0.0), drainage_parameter=full(0.0),
                               theta_sat=full(0.4), soil_thickness_m=full(0.3), active_mask=xp.ones((ny, nx), dtype=bool))
    soil = initial_soil_water_m(column, full(0.1))
    sediment = plot1_sediment_parameters(xp=xp)
    schedule = constant_rainfall(0.0, rain_end_s, rain_mm_h)
    control = PondEventControl(backend=backend, time=ExperimentalControl(max_dt_s=max_dt_s, min_dt_s=1.0 / 1024.0,
                                                                         max_retries=10, max_steps=100_000))
    return Case(bed, ctx, soil, topology, column, sediment, schedule, control, policy or PondCompletionPolicy(), xp, mobile_cell)


def dry_case(**overrides) -> Case:
    """A case that dries and goes quiet fast: rain far below the infiltration capacity, a mobile load to settle, a short
    hold, and `deposit_mobile_at_flow_end` so the terminal settlement is the explicit MAPLE deposition."""
    kw = {"ksat": 1e-3, "rain_end_s": 2.0, "mobile_kg": 1e-3, "max_dt_s": 0.5,
          "policy": PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-2, max_unit_discharge_m2_s=1e-6,
                                         outlet_discharge_m3_s=1e-6, deposit_mobile_at_flow_end=True)}
    kw.update(overrides)
    return make_case(**kw)


def lake_case(**overrides) -> Case:
    """A CLOSED lake at rest: positive surface depth in the lowest cell, no infiltration (ksat 0), no rain, no outlet, and a
    retained mobile load of the finest class in the lake cell. Quiet needs only the hold, and `deposit_mobile_at_flow_end`
    makes the explicit flow-end settlement the terminal transaction, so the terminal MAPLE deposition is exercised (nonzero as
    long as the native wet operator has not already settled the whole pool during the short hold)."""
    kw = {"ksat": 0.0, "rain_mm_h": 0.0, "rain_end_s": 0.5, "mobile_kg": 5e-2, "mobile_cell": (0, 0, 0),
          "closed_lake": True, "max_dt_s": 0.25,
          "policy": PondCompletionPolicy(hold_s=0.5, max_depth_change_m=1e-2, max_unit_discharge_m2_s=1e-6,
                                         outlet_discharge_m3_s=1e-6, deposit_mobile_at_flow_end=True)}
    kw.update(overrides)
    return make_case(**kw)


def advance(event: PondEvent, state, n_steps: int, dt: float = DT):
    """`n_steps` accepted coupled steps of a fixed dt (pure `PondEvent.step`; the schedule rate at the step start)."""
    for _ in range(n_steps):
        state = event.step(state, event.schedule.rate_after_m_per_s(state.t_s), dt).state
    return state


def host(array: Any) -> np.ndarray:
    from maple.core.backend import to_host

    return np.asarray(to_host(array))


#: EXACTLY the native NON-PHYSICAL fields excluded from comparison: MAPLE's `VoxelColumnState.surface_cache` (a derived,
#: `init=False` cache that is None after a native snapshot restore by contract) and the `ArrayBinding` objects inside such
#: caches (compared by object address otherwise). Every physical array, control, time and counter is compared.
NONPHYSICAL_FIELDS = frozenset({"surface_cache"})
NONPHYSICAL_TYPES = frozenset({"ArrayBinding"})


def walk(value: Any, path: str = "state"):
    """Yield `(path, leaf)` over dataclasses / tuples / lists / dicts; arrays and scalars are leaves. Only the native
    non-physical cache fields named above are skipped."""
    from maple.core.backend import is_array

    if type(value).__name__ in NONPHYSICAL_TYPES:
        return
    if isinstance(value, np.ndarray) or is_array(value) or isinstance(value, (np.generic, str, bool, int, float, type(None))):
        yield path, value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        yield path + ".__type__", type(value).__qualname__
        for f in dataclasses.fields(value):
            if f.name in NONPHYSICAL_FIELDS:
                continue
            yield from walk(getattr(value, f.name), f"{path}.{f.name}")
    elif isinstance(value, (tuple, list)):
        yield path + ".__len__", len(value)
        for i, v in enumerate(value):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(value, dict):
        yield path + ".__keys__", tuple(sorted(value))
        for k in sorted(value):
            yield from walk(value[k], f"{path}[{k!r}]")
    elif isinstance(value, ModuleType):
        yield path, value.__name__
    else:
        yield path, repr(value)


def _as_value(leaf: Any) -> Any:
    from maple.core.backend import is_array

    if isinstance(leaf, np.ndarray) or is_array(leaf):
        return host(leaf)
    if isinstance(leaf, np.generic):
        return np.asarray(leaf)
    return leaf


def assert_tree_equal(a: Any, b: Any) -> None:
    """Recursive bitwise equality of two states / handoffs (arrays of any namespace through `to_host`, counters, ledgers,
    topologies, native MAPLE states). A NumPy scalar and a 0-d array of the same value are the same leaf."""
    first, second = list(walk(a)), list(walk(b))
    assert [p for p, _ in first] == [p for p, _ in second], "structure differs"
    for (path, x), (_, y) in zip(first, second, strict=True):
        vx, vy = _as_value(x), _as_value(y)
        if isinstance(vx, np.ndarray) or isinstance(vy, np.ndarray):
            vx, vy = np.asarray(vx), np.asarray(vy)
            if vx.ndim and vy.ndim:
                assert vx.dtype == vy.dtype and vx.shape == vy.shape, f"{path}: dtype / shape differ"
            np.testing.assert_array_equal(vx, vy, err_msg=path)
        else:
            assert vx == vy, f"{path}: {vx!r} != {vy!r}"


def fingerprint(value: Any) -> str:
    """SHA-256 over every leaf (for 'this object was not mutated' checks)."""
    h = hashlib.sha256()
    for path, leaf in walk(value):
        v = _as_value(leaf)
        h.update(path.encode())
        if isinstance(v, np.ndarray):
            h.update(f"{v.dtype.str}{v.shape}".encode())
            h.update(np.array(v, copy=True, order="C").tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()


def tree_hash(path) -> dict:
    """File name -> SHA-256 of a checkpoint directory (for 'the checkpoint was not touched')."""
    from pathlib import Path

    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(path).iterdir()) if p.is_file()}
