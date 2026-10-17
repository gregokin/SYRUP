"""Independent ACTUAL-MAPLE fixtures for the windowed pond checkpoint tests (nothing here was run by its author).

A small dense reference from `probe.build_minimal_state` is given 32 genuinely deeper, class-layered voxels (two alternating
class mixes of the same per-voxel totals), turned into a REAL resident-window bed over a real file-backed deep store
(window 4, base > 0, exact height packets) with the committed baseline explicitly regrouped by MAPLE's own
`windowed_committed_state_from_dense_baseline`. The dense column exists only at this bounded construction boundary. With
`encoding="overlay"` a second, v2 (overlay) deep version of the same store is published by the native journal API and the
bed is rebuilt from it, so full (v1) and overlay (v2) versions are both real. No physics is faked and no native call is patched.
The same terrain / hydraulics as `tests/pond_checkpoint/pond_fixture.py` (this module does not import it, so the tests do not
depend on another workspace). `storage="dense"` builds the dense counterpart of the same case for the schema-separation tests.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from maple_syrup.experimental_storm import ExperimentalControl
from maple_syrup.infiltration import column_parameters, initial_soil_water_m
from maple_syrup.pond_event import PondCompletionPolicy, PondEvent, PondEventControl
from maple_syrup.pond_network import build_pond_topology
from maple_syrup.probe import build_minimal_state
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_bed import BedContext, BedState, apply_bed_demand, publish_bed_storage, with_water
from maple_syrup.sediment_physics import plot1_sediment_parameters

IDENTITY = {"case": "controlled actual MAPLE windowed pond bed", "forcing": "constant 60 mm/h", "source": "test",
            "options": {"dt": 0.1}}
DT = 0.1
N_DEEP = 32
WINDOW = 4


@dataclass
class WCase:
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
    store_dir: Path | None
    storage: str
    v1_manifest_digest: str | None = None  # overlay cases: the genesis (v1) version that the v2 version descends from

    def event(self, **overrides) -> PondEvent:
        """A fresh event from the ORIGINAL topology (what a restarting process builds)."""
        args = {"ctx": self.ctx, "column": self.column, "sediment": self.sediment, "schedule": self.schedule,
                "topology": self.topology, "control": self.control, "policy": self.policy}
        args.update(overrides)
        return PondEvent(args["ctx"], args["column"], args["sediment"], args["schedule"], args["topology"],
                         control=args["control"], policy=args["policy"])


def _layered(p) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(dense_column_mass, w_a, w_b, deep_mass)`: 32 deeper voxels under the probe's own column. The per-voxel totals are the
    probe voxel-0 totals; the lower half has class mix A, the upper half the reversed mix B (heterogeneous strata)."""
    mass = np.asarray(p.voxel_column.mass_kg)
    nc = mass.shape[-1]
    deep = np.repeat(mass[:, :, :1, :], N_DEEP, axis=2).copy()
    total = deep.sum(axis=-1, keepdims=True)
    w_a = np.arange(1.0, nc + 1.0)
    w_a /= w_a.sum()
    w_b = w_a[::-1].copy()
    half = N_DEEP // 2
    deep[:, :, :half] = total[:, :, :half] * w_a
    deep[:, :, half:] = total[:, :, half:] * w_b
    if nc > 1:
        assert not np.array_equal(deep[0, 0, 0], deep[0, 0, -1]), "the deep strata are not class-heterogeneous"
    return np.concatenate((deep, mass), axis=2), w_a, w_b, deep


def make_case(tmp_path: Path, *, xp: ModuleType = np, ny: int = 3, nx: int = 4, pickup_kg_per_class: float | None = None,
              mobile_kg: float | None = None, ksat: float = 2e-6, rain_mm_h: float = 60.0, rain_end_s: float = 100.0,
              backend: str = "numpy", max_dt_s: float = 0.2, policy: PondCompletionPolicy | None = None,
              closed_lake: bool = False, mobile_cell: tuple = (2, 1, 2), storage: str = "windowed",
              encoding: str = "full", store_name: str = "store", chunk_columns: int = 64,
              control_kw: dict | None = None) -> WCase:
    """`storage="windowed"` (default) or `"dense"` (the dense counterpart of the same terrain). `encoding="overlay"` rebuilds the
    bed from a v2 overlay version of the same store (context deep_chunk_encoding = "overlay"). `pickup_kg_per_class` is a
    genuine native pickup at one interior cell BEFORE the event (a real pending ledger; 0.1 kg per class crosses MAPLE's
    pending-elevation trigger, 0.0005 stays far below it); `mobile_kg` a mobile load in the native water state."""
    from maple.core.backend import to_device, to_device_tree
    from maple.core.parameters.avalanching import AvalanchingSpec
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from maple.core.types.sediment_availability import sediment_availability_state_from_active_layer
    from maple.core.types.sediment_ledger import process_index, zeros_sediment_ledger_state
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.core.types.voxel import VoxelColumnState
    from maple.surface.topographic_commit import initial_committed_topography_state_from_physical_state
    from maple.surface.topographic_commit.windowed_commit import windowed_committed_state_from_dense_baseline
    from maple.surface.voxels.deep_store import DeepJournal
    from maple.surface.voxels.windowed_inventory import windowed_column_from_dense, windowed_column_from_version
    from maple.surface.voxels.windowed_prefix import move_prefix_packets
    from maple.water import WaterProcessDemand

    if storage not in ("windowed", "dense"):
        raise ValueError(storage)
    p = build_minimal_state(ny=ny, nx=nx)
    geometry, res = p.geometry, p.mass_resolution_kg
    ids = tuple(g.class_id for g in p.grain_classes.classes)
    nc = len(ids)
    ctx_kw = {}
    store_dir = None
    v1_digest = None
    if storage == "dense":
        committed = initial_committed_topography_state_from_physical_state(p.voxel_column, p.active_layer, geometry, res)
        column = p.voxel_column
        ledger = p.ledger
    else:
        dense_mass, w_a, w_b, deep = _layered(p)
        store_dir = Path(tmp_path) / store_name
        column = windowed_column_from_dense(
            VoxelColumnState(dense_mass), geometry, res, store_dir=store_dir, max_store_bytes=10**8, class_ids=ids,
            window_voxels=WINDOW, chunk_columns=chunk_columns, with_prefix_packets=True)
        if encoding == "overlay":
            v1_digest = column.deep_store.manifest_digest
            records = [(0, 1), (min(5, ny * nx - 1), 2)]  # deep voxels (below every base) of two cells: class mix A -> B
            cells = np.array([c for c, _ in records], dtype=np.int64)
            voxels = np.array([v for _, v in records], dtype=np.int64)
            total = deep.sum(axis=-1)
            changed = dense_mass.copy()
            values = np.empty((len(records), nc), dtype=np.float64)
            for i, (cell, voxel) in enumerate(records):
                r, c = divmod(cell, nx)
                changed[r, c, voxel] = total[r, c, voxel] * w_b
                values[i] = changed[r, c, voxel]
            assert np.all(np.asarray(column.base).reshape(-1)[cells] > voxels), "journal records must lie below every base"
            journal = DeepJournal.from_records(cells, voxels, values, header=column.deep_store.header)
            version = column.deep_store.apply_journal(journal, geometry, res, encoding="overlay")
            assert version.format_version == 2
            column = windowed_column_from_version(version, geometry, res, class_ids=ids, with_prefix_packets=True)
            dense_mass = changed
            ctx_kw = {"deep_chunk_encoding": "overlay"}
        baseline = initial_committed_topography_state_from_physical_state(
            VoxelColumnState(dense_mass), p.active_layer, geometry, res)
        committed, _receipt = windowed_committed_state_from_dense_baseline(baseline, column, p.active_layer, geometry)
        ledger = zeros_sediment_ledger_state(ny, nx, nc, present_processes=(process_index("water_erosion_deposition"),))
        assert np.any(np.asarray(column.base) > 0), "the fixture needs a nonzero deep base"
    available = sediment_availability_state_from_active_layer(p.active_layer.mass_kg, np.ones_like(p.active_layer.mass_kg))
    bed = BedState(column, p.active_layer, p.water, ledger, committed, available)
    ctx = BedContext(geometry, p.grain_classes, res, TopographicCommitSpec(), AvalanchingSpec(enabled=False),
                     np.ones_like(p.active_layer.mass_kg), **ctx_kw)
    zc = np.array(committed_elevation_m(committed), dtype=np.float64)
    if pickup_kg_per_class is not None:
        demand = np.zeros_like(p.active_layer.mass_kg)
        demand[1, 2] = pickup_kg_per_class
        bed, _ = apply_bed_demand(bed, ctx, WaterProcessDemand(demand, np.zeros_like(demand)))
    depth = np.array(bed.water.depth_m, copy=True)
    if closed_lake:
        depth[...] = 0.0
        depth[0, 0] = 0.003
    mobile = None
    cell = None
    if mobile_kg is not None:
        cell = mobile_cell
        mobile = np.zeros_like(bed.water.mobile_mass_by_cell_class_kg)
        mobile[cell] = mobile_kg
    if mobile is not None or closed_lake:
        bed = with_water(bed, ctx, depth, mobile)
    bed = publish_bed_storage(bed, ctx)  # a no-op unless a journal is pending: the starting bed is published
    if storage == "windowed":
        assert bed.voxel_column.pending_deep is None

    z = np.full((ny + 2, nx + 2), float(zc.max()) + 1.0)
    z[1:-1, 1:-1] = zc
    if closed_lake:
        topology = build_pond_topology(z, np.ones((ny, nx), bool), np.ones((ny, nx)), geometry.dx_m, [],
                                       allow_closed_components=True)
    else:
        z[0, 1:-1] = zc[0] - 0.02
        topology = build_pond_topology(z, np.ones((ny, nx), bool), np.ones((ny, nx)), geometry.dx_m,
                                       [(0, c, -1, 0) for c in range(nx)])

    if xp is not np:
        host_column = bed.voxel_column
        bed = to_device_tree(bed, xp)
        if storage == "windowed":
            bed = replace(bed, voxel_column=move_prefix_packets(bed.voxel_column, host_column, xp))
        ctx = replace(ctx, initial_available_fraction=to_device(ctx.initial_available_fraction, xp))
    full = lambda v: xp.full((ny, nx), v, dtype=np.float64)  # noqa: E731
    cparams = column_parameters(model="fixed_ksat", ksat_m_per_s=full(ksat), suction_m=full(0.0), drainage_parameter=full(0.0),
                                theta_sat=full(0.4), soil_thickness_m=full(0.3), active_mask=xp.ones((ny, nx), dtype=bool))
    soil = initial_soil_water_m(cparams, full(0.1))
    sediment = plot1_sediment_parameters(xp=xp)
    schedule = constant_rainfall(0.0, rain_end_s, rain_mm_h)
    control = PondEventControl(backend=backend, time=ExperimentalControl(max_dt_s=max_dt_s, min_dt_s=1.0 / 1024.0,
                                                                         max_retries=10, max_steps=100_000),
                               bed_storage=storage, **(control_kw or {}))
    return WCase(bed, ctx, soil, topology, cparams, sediment, schedule, control, policy or PondCompletionPolicy(), xp, cell,
                 store_dir, storage, v1_digest)


def dry_case(tmp_path: Path, **overrides) -> WCase:
    """Dries and goes quiet fast; `deposit_mobile_at_flow_end` makes the terminal settlement the explicit MAPLE deposition."""
    kw = {"ksat": 1e-3, "rain_end_s": 2.0, "mobile_kg": 1e-3, "max_dt_s": 0.5,
          "policy": PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-2, max_unit_discharge_m2_s=1e-6,
                                         outlet_discharge_m3_s=1e-6, deposit_mobile_at_flow_end=True)}
    kw.update(overrides)
    return make_case(tmp_path, **kw)


def lake_case(tmp_path: Path, **overrides) -> WCase:
    """A closed lake at rest with a retained mobile load: the flow-end settlement is a nonzero terminal MAPLE deposition."""
    kw = {"ksat": 0.0, "rain_mm_h": 0.0, "rain_end_s": 0.5, "mobile_kg": 5e-2, "mobile_cell": (0, 0, 0),
          "closed_lake": True, "max_dt_s": 0.25,
          "policy": PondCompletionPolicy(hold_s=0.5, max_depth_change_m=1e-2, max_unit_discharge_m2_s=1e-6,
                                         outlet_discharge_m3_s=1e-6, deposit_mobile_at_flow_end=True)}
    kw.update(overrides)
    return make_case(tmp_path, **kw)


def advance(event: PondEvent, state, n_steps: int, dt: float = DT):
    """`n_steps` accepted coupled steps of a fixed dt (pure `PondEvent.step`)."""
    for _ in range(n_steps):
        state = event.step(state, event.schedule.rate_after_m_per_s(state.t_s), dt).state
    return state


def published(state, event: PondEvent):
    """The same accepted state with its storage explicitly published through the core's `PondEvent.publish_storage` (no
    physical change; it keeps the publication counter consistent). A checkpoint never does this itself."""
    out = event.publish_storage(state)
    assert getattr(out.bed.voxel_column, "pending_deep", None) is None
    return out


def host(array: Any) -> np.ndarray:
    from maple.core.backend import to_host

    return np.asarray(to_host(array))


NONPHYSICAL_FIELDS = frozenset({"surface_cache", "window_cache", "height_packet"})
NONPHYSICAL_TYPES = frozenset({"ArrayBinding"})


def walk(value: Any, path: str = "state"):
    """Yield `(path, leaf)` over dataclasses / tuples / lists / dicts. The deep-store handle is an opaque version: its format,
    physical / logical identities and location are leaves, and the derived height packet of a windowed column is compared by
    VALUE as its own leaf. Only derived caches (`surface_cache`, `window_cache`) are skipped."""
    from maple.core.backend import is_array
    from maple.core.types.voxel import WindowedVoxelColumnState

    if type(value).__name__ in NONPHYSICAL_TYPES:
        return
    if isinstance(value, np.ndarray) or is_array(value) or isinstance(value, (np.generic, str, bool, int, float, type(None))):
        yield path, value
    elif isinstance(value, WindowedVoxelColumnState):
        yield path + ".__type__", type(value).__qualname__
        for f in dataclasses.fields(value):
            if f.name in NONPHYSICAL_FIELDS:
                continue
            if f.name == "deep_store":
                store = value.deep_store
                yield path + ".deep_store", (int(store.format_version), store.manifest_digest, store.logical_manifest_digest,
                                             str(Path(store.store_dir).resolve()), int(store.version_id))
            else:
                yield from walk(getattr(value, f.name), f"{path}.{f.name}")
        packets = value.height_packet
        yield path + ".height_packet", None if packets is None else packets.packet
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
    """Recursive bitwise equality of two states / handoffs (arrays of any namespace through `to_host`)."""
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
    """File name -> SHA-256 of a checkpoint directory."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(path).iterdir()) if p.is_file()}


def store_pins(store_dir) -> dict:
    """Relative path -> SHA-256 of EVERY file below a deep-store directory (versions, chunks, descriptor): equality before and
    after proves no version was published, replaced or deleted and no file was touched."""
    root = Path(store_dir)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def logical_column_bytes(column: Any) -> bytes:
    """The bounded tiny LOGICAL dense column of a windowed bed (the one place a test reads it densely)."""
    from maple.surface.voxels.windowed_inventory import windowed_to_dense

    return host(windowed_to_dense(column, max_bytes=10**8).mass_kg).tobytes()


def physical(state) -> None:
    """The fixture must have done real work, or equality would be vacuous."""
    assert state.counters.n_accepted_steps > 0 and state.t_s > 0
    assert float(host(state.hydraulic.depth_m).max()) > 0.0
    assert float(np.abs(host(state.hydraulic.qx_m2_s)).max() + np.abs(host(state.hydraulic.qy_m2_s)).max()) > 0.0
    moved = float(host(state.sediment.wet_actual).sum() + host(state.bed.water.mobile_mass_by_cell_class_kg).sum())
    assert moved > 0.0, "no sediment moved: the fixture does not exercise the sediment continuation"
