"""Deterministic MAPLE water-transaction sequence for two-environment differential checks.

Run once under the accepted dependency and once under a candidate; compare the NPZ files bitwise
(benchmarks/phase7e/compare_sequences.py). Covers the actual Plot1 bed and adversarial synthetic
columns: zero demand, dense pickup, dense deposition with boundary export, supply exhaustion of
whole columns, sub-resolution demands, mixed erosion/deposition in one call, columns of 1-3 voxels,
and deep roundoff-room columns.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
from maple.core.boundaries import AxisBoundary, BoundaryKind
from maple.core.parameters.geometry import GeometrySpec
from maple.core.types.active_layer import ActiveLayerState
from maple.core.types.sediment_availability import SedimentAvailabilityState
from maple.core.types.sediment_ledger import zeros_sediment_ledger_state
from maple.core.types.voxel import VoxelColumnState
from maple.core.types.water import WaterState
from maple.surface.active_layer.capacity import compute_active_layer_target_mass_kg
from maple.surface.voxels.capacity import max_voxel_mass_kg
from maple.water import WaterProcessDemand, apply_water_process_demand

from maple_syrup.dependency import resolve_maple_dependency
from maple_syrup.provenance import source_tree_digest


def arrays_of(obj, prefix, out):
    if isinstance(obj, np.ndarray):
        out[prefix] = obj
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            if f.name == 'surface_cache':
                continue  # derived metadata, not physical state
            arrays_of(getattr(obj, f.name), prefix + '.' + f.name, out)
    elif isinstance(obj, (int, float)):
        out[prefix] = np.asarray(obj, dtype=np.float64)


def synthetic_bed(rng, ny, nx, nv, nc, g, *, deep_room=False, empty_fraction=0.0):
    cap = max_voxel_mass_kg(g)
    target = compute_active_layer_target_mass_kg(g)
    mass = np.zeros((ny, nx, nv, nc))
    for y in range(ny):
        for x in range(nx):
            if rng.random() < empty_fraction:
                continue
            full = rng.integers(0, nv + 1)
            for k in range(nv):
                if k < full:
                    frac = rng.dirichlet(np.ones(nc))
                    mass[y, x, k] = frac * cap
                    if deep_room and rng.random() < 0.5:
                        mass[y, x, k] *= 1.0 - 4.0 * np.finfo(float).eps  # roundoff-sized room
            if full < nv and full > 0 and rng.random() < 0.7:
                mass[y, x, full] = rng.dirichlet(np.ones(nc)) * cap * rng.uniform(0.01, 0.99)
    active = rng.dirichlet(np.ones(nc), size=(ny, nx)) * target
    active[rng.random((ny, nx)) < 0.1] *= rng.uniform(0.0, 0.5)  # a few underfilled cells
    return mass, active


def run_case(name, mass, active, g, mres, rng, n_steps, record, classes):
    ny, nx, _nv, nc = mass.shape
    column = VoxelColumnState(mass_kg=mass.copy())
    layer = ActiveLayerState(mass_kg=active.copy())
    avail = SedimentAvailabilityState(available_mass_kg=active.copy(), bound_mass_kg=np.zeros_like(active))
    frac = np.ones((ny, nx, nc))
    water = WaterState(depth_m=np.full((ny, nx), 0.01), mobile_mass_by_cell_class_kg=np.zeros((ny, nx, nc)))
    ledger = zeros_sediment_ledger_state(ny, nx, nc)
    zeros = np.zeros((ny, nx, nc))
    h = hashlib.sha256()
    for step in range(n_steps):
        kind = step % 6
        removal = zeros
        deposition = zeros
        export = None
        mobile = water.mobile_mass_by_cell_class_kg
        if kind == 0:  # zero demand both directions
            pass
        elif kind == 1:  # dense small pickup
            removal = np.minimum(rng.random((ny, nx, nc)) * 3e-4 / nc, layer.mass_kg)
        elif kind == 2:  # deposition of most of the pool plus export at row 0
            deposition = mobile * rng.random((ny, nx, nc)) * 0.9
            export = np.zeros_like(mobile)
            export[0] = (mobile - deposition)[0] * 0.5
        elif kind == 3:  # supply exhaustion: huge demand on a few cells, mixed with deposition
            removal = zeros.copy()
            cells = rng.integers(0, ny * nx, size=max(1, ny * nx // 10))
            removal.reshape(-1, nc)[cells] = 1e3
            deposition = mobile * 0.5
        elif kind == 4:  # sub-resolution and tiny demands everywhere
            removal = np.full((ny, nx, nc), 1e-14)
            deposition = np.minimum(mobile, 1e-15)
        else:  # large deposition (burial across a voxel boundary) from an injected pool
            injected = rng.random((ny, nx, nc)) * max_voxel_mass_kg(g) * 0.3
            water = WaterState(depth_m=water.depth_m, mobile_mass_by_cell_class_kg=mobile + injected)
            mobile = water.mobile_mass_by_cell_class_kg
            deposition = mobile * 0.95
        demand = WaterProcessDemand(removal, deposition, boundary_export_by_cell_class_kg=export)
        inputs = {}
        for label, obj in zip(("column", "layer", "water", "ledger", "availability", "demand"),
                              (column, layer, water, ledger, avail, demand)):
            arrays_of(obj, label, inputs)
        before = {k: v.copy() for k, v in inputs.items()}
        try:
            result = apply_water_process_demand(column, layer, water, ledger, demand, g, classes, mres,
                                                sediment_availability=avail, initial_available_fraction=frac,
                                                adapter_name='phase7e_differential')
        except ValueError as exc:  # a MAPLE refusal (e.g. capacity): recorded, both environments must refuse identically
            assert all(np.array_equal(before[k], v) for k, v in inputs.items()), "failed transaction mutated inputs"
            record[f'{name}.step{step}.error'] = np.frombuffer(repr(exc).encode(), dtype=np.uint8)
            continue
        column, layer, water, ledger, avail = (result.new_voxel_column, result.new_active_layer, result.new_water,
                                               result.new_ledger, result.new_sediment_availability)
        out = {}
        arrays_of(result, f'{name}.step{step}', out)
        for key, value in sorted(out.items()):
            h.update(key.encode()); h.update(np.ascontiguousarray(value).tobytes())
        if step in (0, 1, 2, 3, 4, 5, n_steps - 1):
            record.update(out)
    record[f'{name}.chain_sha256'] = np.frombuffer(h.hexdigest().encode(), dtype=np.uint8)
    record[f'{name}.final_mass'] = column.mass_kg
    record[f'{name}.final_active'] = layer.mass_kg
    return column


def main(output: Path, steps: int):
    rng = np.random.default_rng(20261001)
    record = {}
    from maple_syrup.case_import import verify_plot1_case
    from maple_syrup.sediment_bed import bed_from_case
    verified = verify_plot1_case('outputs/plot1', allow_maple_source_change=True)
    bed, ctx = bed_from_case(verified.case)
    g = ctx.geometry
    # Plot1: real bed, real classes (class metadata object), through the SYRUP adapter path
    mres = ctx.mass_resolution_kg
    classes = ctx.grain_classes
    nc = len(classes.classes)
    ny, nx, _nv, _ = bed.voxel_column.mass_kg.shape

    def run_plot1():
        column, layer, water, ledger, avail = (bed.voxel_column, bed.active_layer, bed.water, bed.ledger,
                                               bed.sediment_availability)
        zeros = np.zeros((ny, nx, nc)); h = hashlib.sha256()
        for step in range(steps):
            kind = step % 4
            removal, deposition, export = zeros, zeros, None
            mobile = water.mobile_mass_by_cell_class_kg
            if kind == 1:
                removal = np.minimum(rng.random((ny, nx, nc)) * 3e-4 / nc, layer.mass_kg)
            elif kind == 2:
                deposition = mobile * rng.random((ny, nx, nc)) * 0.9
                export = np.zeros_like(mobile); export[0] = (mobile - deposition)[0] * 0.5
            elif kind == 3:
                removal = zeros.copy(); removal.reshape(-1, nc)[rng.integers(0, ny * nx, 30)] = 500.0
                deposition = mobile * 0.3
            demand = WaterProcessDemand(removal, deposition, boundary_export_by_cell_class_kg=export)
            result = apply_water_process_demand(column, layer, water, ledger, demand, g, classes, mres,
                                                sediment_availability=avail, initial_available_fraction=ctx.initial_available_fraction,
                                                adapter_name='phase7e_differential')
            column, layer, water, ledger, avail = (result.new_voxel_column, result.new_active_layer,
                                                   result.new_water, result.new_ledger, result.new_sediment_availability)
            out = {}; arrays_of(result, f'plot1.step{step}', out)
            for key, value in sorted(out.items()):
                h.update(key.encode()); h.update(np.ascontiguousarray(value).tobytes())
            if step < 8 or step == steps - 1:
                record.update(out)
        record['plot1.chain_sha256'] = np.frombuffer(h.hexdigest().encode(), dtype=np.uint8)
        record['plot1.final_mass'] = column.mass_kg
        record['plot1.final_active'] = layer.mass_kg
        record['plot1.final_ledger_pending'] = ledger.pending_bed_mass_change_kg
    run_plot1()

    # Synthetic adversarial beds use GeometrySpec directly (no availability object needed: grain
    # classes object is required by the step only for its class count -> reuse Plot1's).
    for label, (ny_, nx_, nv_, kw) in {
        'multilayer': (5, 4, 12, {}),
        'deep_room': (5, 4, 10, {'deep_room': True}),
        'sparse_empty': (6, 3, 8, {'empty_fraction': 0.4}),
        'one_voxel': (4, 3, 1, {}),
        'two_voxels': (4, 3, 2, {}),
        'three_voxels': (4, 3, 3, {}),
    }.items():
        gs = GeometrySpec(nx=nx_, ny=ny_, dx_m=g.dx_m, dy_m=g.dy_m, voxel_dz_m=g.voxel_dz_m,
                          bulk_density_kg_m3=g.bulk_density_kg_m3, active_layer_thickness_m=g.active_layer_thickness_m,
                          boundary_x=AxisBoundary(kind=BoundaryKind.PERIODIC), boundary_y=AxisBoundary(kind=BoundaryKind.PERIODIC))
        mass, active = synthetic_bed(rng, ny_, nx_, nv_, nc, gs, **kw)
        run_case(label, mass, active, gs, mres, rng, steps, record, classes)
    provenance = {'maple_digest': source_tree_digest(resolve_maple_dependency().package_dir).digest_sha256,
                  'steps': steps, 'fields': len(record)}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **record)
    (output.with_suffix('.json')).write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=48)
    a = p.parse_args()
    main(a.output, a.steps)
