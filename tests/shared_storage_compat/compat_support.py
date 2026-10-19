"""Test-only helpers of tests/shared_storage_compat (no production fallback, no fake MAPLE type)."""
from __future__ import annotations

import hashlib
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np


def host(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "get"):
        return np.asarray(x.get())
    return np.asarray(x)


def dataclass_arrays(value):
    """Every NumPy array of a (nested) dataclass, in field order."""
    if isinstance(value, np.ndarray):
        yield value
    elif is_dataclass(value) and not isinstance(value, type):
        for f in fields(value):
            yield from dataclass_arrays(getattr(value, f.name))


def physical_arrays(bed):
    """Named physical arrays of a windowed `BedState` (resident window, deep metadata, packet, layer, water, availability,
    ledger and the pending journal, if any). Store handles and caches are excluded; values are compared EXACTLY."""
    column = bed.voxel_column
    out = {name: host(getattr(column, name)) for name in
           ("window_kg", "base", "deep_by_class_kg", "deep_total_kg", "deep_room_kg")}
    out["height_packet"] = host(column.height_packet.packet)
    out["active_layer"] = host(bed.active_layer.mass_kg)
    out["depth_m"] = host(bed.water.depth_m)
    out["mobile_kg"] = host(bed.water.mobile_mass_by_cell_class_kg)
    out["available_kg"] = host(bed.sediment_availability.available_mass_kg)
    out["bound_kg"] = host(bed.sediment_availability.bound_mass_kg)
    for i, array in enumerate(dataclass_arrays(bed.ledger)):
        out[f"ledger[{i}]"] = array
    pending = column.pending_deep
    if pending is not None:
        journal = pending.journal
        out["journal_cell"], out["journal_voxel"], out["journal_values"] = (
            host(journal.cell_flat), host(journal.voxel), host(journal.values_kg))
    return {k: np.array(v, copy=True) for k, v in out.items()}


def assert_exact_equal(left: dict, right: dict, what: str):
    assert left.keys() == right.keys(), f"{what}: array sets differ: {sorted(set(left) ^ set(right))}"
    for key, value in left.items():
        assert value.shape == right[key].shape and value.dtype == right[key].dtype, f"{what}: {key}"
        np.testing.assert_array_equal(value, right[key], err_msg=f"{what}: {key}")


def tree_fingerprint(root) -> dict:
    """`{relative posix path: sha256}` of every file under `root` (the deep store directory)."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def make_transport(pool0, deposit_fraction):
    """The transport stand-in of these tests: the ACTUAL removal reported by the pickup (a PickupView or a full pickup
    result) joins the mobile pool, and `deposit_fraction` of the post-pickup pool is requested for deposition in place.
    Returns `(transport, seen)`; `seen` records the removal the callback actually received."""
    from maple.water import WaterProcessDemand

    seen = []

    def transport(view):
        removal = host(view.actual_removal_by_cell_class_kg)
        seen.append(removal.copy())
        mobile_after = pool0 + removal
        demand = WaterProcessDemand(np.zeros_like(mobile_after), deposit_fraction * mobile_after)
        return mobile_after, demand

    return transport, seen


def sequential_reference(bed, ctx, depth, request, transport):
    """The two PUBLIC windowed calls the paired transaction must reproduce: pickup, transport, deposit."""
    from maple.water import WaterProcessDemand

    from maple_syrup.sediment_bed import apply_bed_demand, with_water

    zeros = bed.water.mobile_mass_by_cell_class_kg * 0.0
    b1 = with_water(bed, ctx, depth)
    b1, pickup = apply_bed_demand(b1, ctx, WaterProcessDemand(request, zeros))
    mobile_after, demand = transport(pickup)
    b2 = with_water(b1, ctx, depth, mobile_after)
    b3, deposit = apply_bed_demand(b2, ctx, demand)
    return b3, pickup, deposit


def class_inventory_kg(bed):
    """Bed (active layer + resident window + deep) plus mobile pool by class, through the SYRUP adapter's own helpers."""
    from maple_syrup.sediment_bed import bed_inventory

    return host(bed_inventory(bed)) + host(bed.water.mobile_mass_by_cell_class_kg).sum(axis=(0, 1))
