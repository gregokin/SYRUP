"""Phase 1 integration and provenance probe.

Builds a small, real MAPLE state through MAPLE's own public constructors
and initializers -- the same sequence `case_tools/compilers/case_compiler.py`
uses (empty voxel column, `deposit_surface_mixture_batch`,
`initialize_active_layer_from_voxels`, `water_state_from_depth`,
`zeros_sediment_ledger_state`) -- checks its inventories against values
computed independently from the authored inputs, and runs one zero-demand
`apply_water_process_demand` step on the selected backend.

It writes nothing unless `--output` names a new file, installs nothing,
and moves no sediment. The CuPy path only places a 3x4 state on the device
and back; it is not evidence of GPU readiness for any SYRUP solver.

    python -m maple_syrup.probe [--backend numpy|cupy] [--output FILE]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import maple_syrup
from maple_syrup.dependency import MapleDependencyError, resolve_maple_dependency
from maple_syrup.provenance import (
    capture_maple_provenance,
    capture_syrup_provenance,
    environment_record,
    source_tree_digest,
)

__all__ = [
    "PROBE_ADAPTER_NAME",
    "MinimalMapleState",
    "ProbeCheckError",
    "build_minimal_state",
    "check_inventories",
    "exercise_zero_demand_exchange",
    "main",
    "run_probe",
]

PROBE_SCHEMA = "maple_syrup.phase1_probe.v1"
PROBE_ADAPTER_NAME = "maple_syrup_phase1_zero_demand_probe"
DEFAULT_CLASS_FRACTIONS = (0.10, 0.15, 0.25, 0.25, 0.15, 0.10)
# FP64 reconciliation allowance for the independent checks: a few dozen
# roundings at the magnitude of the largest compared quantity.
_FP_ROUNDINGS = 64.0


class ProbeCheckError(RuntimeError):
    """An independent check of the MAPLE state or exchange failed."""


@dataclass(frozen=True)
class MinimalMapleState:
    """Host-resident MAPLE state plus the authored inputs it was built from."""

    geometry: Any
    grain_classes: Any
    mass_resolution_kg: float
    voxel_column: Any
    active_layer: Any
    water: Any
    ledger: Any
    # Authored inputs, kept for independent checks.
    surface_elevation_m: np.ndarray  # (ny, nx)
    class_fractions: np.ndarray  # (n_classes,)
    requested_mass_by_cell_class_kg: np.ndarray  # (ny, nx, n_classes)
    initial_depth_m: np.ndarray  # (ny, nx)
    # Sub-resolution residuals MAPLE itself reported, (ny, nx) kg each.
    deposit_numerical_residual_kg: np.ndarray
    active_layer_init_numerical_residual_kg: np.ndarray


def build_minimal_state(
    *,
    ny: int = 3,
    nx: int = 4,
    dx_m: float = 0.5,
    dy_m: float = 0.5,
    n_voxels: int = 3,
    voxel_dz_m: float = 0.10,
    bulk_density_kg_m3: float = 1250.0,
    active_layer_thickness_m: float = 0.002,
    initial_depth_m: float = 0.001,
    class_fractions: tuple[float, ...] = DEFAULT_CLASS_FRACTIONS,
) -> MinimalMapleState:
    """Build a uniform-mixture bed on a gently tilted surface, with the six
    MAHLERAN 1.2.1 class diameters taken from MAPLE's own table, prescribed
    zero-inflow boundaries, a uniform initial water depth and an empty
    ledger. All arrays are host NumPy (MAPLE's run-boundary factories)."""
    from maple.core.boundaries import AxisBoundary, BoundaryKind
    from maple.core.parameters.geometry import GeometrySpec, validate_geometry
    from maple.core.parameters.grain_classes import (
        GrainClass,
        GrainClassSet,
        validate_grain_classes,
    )
    from maple.core.parameters.numerics import DEFAULT_MASS_RESOLUTION_KG
    from maple.core.parameters.water_coupling import MAHLERAN_1_2_1_CLASS_DIAMETERS_M
    from maple.core.types.sediment_ledger import zeros_sediment_ledger_state
    from maple.core.types.voxel import zeros_voxel_column_state
    from maple.core.types.water import water_state_from_depth
    from maple.surface.active_layer import initialize_active_layer_from_voxels
    from maple.surface.voxels import deposit_surface_mixture_batch
    from maple.water import validate_water_state

    n_classes = len(MAHLERAN_1_2_1_CLASS_DIAMETERS_M)
    fractions = np.asarray(class_fractions, dtype=np.float64)
    if fractions.shape != (n_classes,):
        raise ValueError(f"class_fractions must have {n_classes} entries, got {fractions.shape}")
    if np.any(fractions < 0.0) or abs(float(fractions.sum()) - 1.0) > 1e-12:
        raise ValueError("class_fractions must be non-negative and sum to 1")

    grain_classes = GrainClassSet(
        classes=tuple(
            GrainClass(
                class_id=f"mahleran_{index + 1}",
                diameter_m=float(diameter),
                particle_density_kg_m3=2650.0,
                is_aggregate=False,
            )
            for index, diameter in enumerate(MAHLERAN_1_2_1_CLASS_DIAMETERS_M)
        )
    )
    validate_grain_classes(grain_classes)

    # Prescribed zero inflow on both axes. In MAPLE this is sediment
    # boundary metadata only; it is not a hydraulic wall or outlet.
    def zero_inflow_boundary():
        return AxisBoundary(
            kind=BoundaryKind.PRESCRIBED,
            inflow_flux_kg_m_s={c.class_id: 0.0 for c in grain_classes.classes},
        )

    geometry = GeometrySpec(
        nx=nx,
        ny=ny,
        dx_m=dx_m,
        dy_m=dy_m,
        boundary_x=zero_inflow_boundary(),
        boundary_y=zero_inflow_boundary(),
        voxel_dz_m=voxel_dz_m,
        bulk_density_kg_m3=bulk_density_kg_m3,
        active_layer_thickness_m=active_layer_thickness_m,
    )
    validate_geometry(geometry, grain_classes.ids())
    mass_resolution_kg = float(DEFAULT_MASS_RESOLUTION_KG)

    rows = np.arange(ny, dtype=np.float64)[:, None]
    cols = np.arange(nx, dtype=np.float64)[None, :]
    elevation_m = 0.12 + 0.01 * rows + 0.005 * cols
    if float(elevation_m.max()) >= n_voxels * voxel_dz_m:
        raise ValueError("surface elevation exceeds the allocated voxel column")
    column_mass_kg = elevation_m * bulk_density_kg_m3 * dx_m * dy_m
    request_kg = column_mass_kg[:, :, None] * fractions[None, None, :]

    column = zeros_voxel_column_state(ny, nx, n_voxels, n_classes)
    column, deposit_result = deposit_surface_mixture_batch(
        column, request_kg, geometry, mass_resolution_kg
    )
    column, active_layer, init_result = initialize_active_layer_from_voxels(
        column, geometry, mass_resolution_kg
    )
    if bool(np.any(init_result.underfilled_mask)):
        raise ProbeCheckError("active-layer initialization reported underfilled cells")

    depth_m = np.full((ny, nx), initial_depth_m, dtype=np.float64)
    water = water_state_from_depth(depth_m, n_classes)
    validate_water_state(water, ny, nx, n_classes)
    ledger = zeros_sediment_ledger_state(ny, nx, n_classes)

    return MinimalMapleState(
        geometry=geometry,
        grain_classes=grain_classes,
        mass_resolution_kg=mass_resolution_kg,
        voxel_column=column,
        active_layer=active_layer,
        water=water,
        ledger=ledger,
        surface_elevation_m=elevation_m,
        class_fractions=fractions,
        requested_mass_by_cell_class_kg=request_kg,
        initial_depth_m=depth_m,
        deposit_numerical_residual_kg=np.asarray(
            deposit_result.numerical_residual_total_mass_kg, dtype=np.float64
        ),
        active_layer_init_numerical_residual_kg=np.asarray(
            init_result.numerical_residual_total_mass_kg, dtype=np.float64
        ),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeCheckError(message)


def check_inventories(state: MinimalMapleState) -> dict[str, Any]:
    """Compare MAPLE's state with quantities computed from the authored
    inputs alone (not from MAPLE functions). Raises `ProbeCheckError`."""
    from maple.surface.active_layer import check_active_layer_voxel_partition

    g = state.geometry
    eps = float(np.finfo(np.float64).eps)
    voxel = np.asarray(state.voxel_column.mass_kg)
    active = np.asarray(state.active_layer.mass_kg)
    cell_area_m2 = g.dx_m * g.dy_m

    bed = voxel.sum(axis=2) + active
    request = state.requested_mass_by_cell_class_kg
    bed_error = float(np.max(np.abs(bed - request)))
    bed_tol = _FP_ROUNDINGS * eps * float(request.sum(axis=2).max()) + float(
        state.deposit_numerical_residual_kg.max()
    )
    _require(bed_error <= bed_tol, f"bed inventory differs from request by {bed_error} kg")

    class_total_error = float(np.max(np.abs(bed.sum(axis=(0, 1)) - request.sum(axis=(0, 1)))))
    class_total_tol = bed_tol * request.shape[0] * request.shape[1]
    _require(
        class_total_error <= class_total_tol,
        f"per-class domain inventory differs by {class_total_error} kg",
    )

    target_kg = g.bulk_density_kg_m3 * cell_area_m2 * g.active_layer_thickness_m
    active_error = float(np.max(np.abs(active.sum(axis=2) - target_kg)))
    active_tol = _FP_ROUNDINGS * eps * target_kg + float(
        state.active_layer_init_numerical_residual_kg.max()
    )
    _require(active_error <= active_tol, f"active-layer mass differs from target by {active_error} kg")

    # A uniform mixture extracted proportionally keeps the authored
    # fractions; dimensionless, so an absolute FP64 allowance is used.
    composition = active / active.sum(axis=2, keepdims=True)
    composition_error = float(np.max(np.abs(composition - state.class_fractions[None, None, :])))
    _require(
        composition_error <= 1e-12,
        f"active-layer composition differs from the uniform mixture by {composition_error}",
    )

    elevation = bed.sum(axis=2) / (g.bulk_density_kg_m3 * cell_area_m2)
    elevation_error = float(np.max(np.abs(elevation - state.surface_elevation_m)))
    elevation_tol = _FP_ROUNDINGS * eps * float(state.surface_elevation_m.max()) + float(
        state.deposit_numerical_residual_kg.max()
    ) / (g.bulk_density_kg_m3 * cell_area_m2)
    _require(elevation_error <= elevation_tol, f"elevation differs by {elevation_error} m")

    _require(
        np.array_equal(np.asarray(state.water.depth_m), state.initial_depth_m),
        "water depth differs from the authored depth",
    )
    _require(
        not np.any(np.asarray(state.water.mobile_mass_by_cell_class_kg)),
        "initial water-borne mobile mass is not zero",
    )
    _require(
        not np.any(np.asarray(state.ledger.pending_bed_mass_change_kg))
        and not np.any(np.asarray(state.ledger.process_totals_kg)),
        "initial ledger is not empty",
    )

    # MAPLE's own validator, reported separately from the independent checks.
    check_active_layer_voxel_partition(
        state.active_layer, state.voxel_column, g, state.mass_resolution_kg
    )

    return {
        "bed_vs_request_max_abs_kg": bed_error,
        "bed_vs_request_tolerance_kg": bed_tol,
        "class_total_max_abs_kg": class_total_error,
        "active_layer_vs_target_max_abs_kg": active_error,
        "active_layer_vs_target_tolerance_kg": active_tol,
        "active_layer_target_kg": target_kg,
        "active_composition_max_abs": composition_error,
        "elevation_max_abs_m": elevation_error,
        "total_bed_mass_by_class_kg": [float(v) for v in bed.sum(axis=(0, 1))],
        "maple_partition_validator": "passed",
    }


def exercise_zero_demand_exchange(state: MinimalMapleState, resolved_backend) -> dict[str, Any]:
    """Place the state on the resolved backend, apply MAPLE's exactly-zero
    water demand, bring the result back, and require that nothing moved."""
    from maple.core.backend import read_transfer_counters, to_device_tree, to_host_tree
    from maple.water import apply_water_process_demand, zero_water_demand

    g = state.geometry
    n_classes = len(state.grain_classes.classes)
    xp = resolved_backend.xp
    counters_before = read_transfer_counters()

    voxel, active, water, ledger = to_device_tree(
        (state.voxel_column, state.active_layer, state.water, state.ledger), xp
    )
    demand = zero_water_demand(g.ny, g.nx, n_classes, namespace=xp)
    result = apply_water_process_demand(
        voxel, active, water, ledger, demand, g, state.grain_classes, state.mass_resolution_kg,
        adapter_name=PROBE_ADAPTER_NAME,
    )
    host = to_host_tree(result)
    counters_after = read_transfer_counters()

    _require(
        np.array_equal(host.new_voxel_column.mass_kg, np.asarray(state.voxel_column.mass_kg)),
        "zero-demand step changed the voxel column",
    )
    _require(
        np.array_equal(host.new_active_layer.mass_kg, np.asarray(state.active_layer.mass_kg)),
        "zero-demand step changed the active layer",
    )
    _require(
        np.array_equal(host.new_water.depth_m, np.asarray(state.water.depth_m)),
        "zero-demand step changed water depth",
    )
    _require(
        not np.any(host.new_water.mobile_mass_by_cell_class_kg),
        "zero-demand step created water-borne mobile mass",
    )
    for name in (
        "actual_removal_by_class_kg", "deposited_mass_by_class_kg", "bed_change_by_class_kg",
        "boundary_export_by_class_kg",
    ):
        _require(not np.any(getattr(host, name)), f"zero-demand step reported nonzero {name}")
    _require(
        not np.any(host.new_ledger.pending_bed_mass_change_kg),
        "zero-demand step left pending ledger mass",
    )
    _require(host.adapter_name == PROBE_ADAPTER_NAME, "result does not record the probe adapter")

    return {
        "backend": resolved_backend.backend.value,
        "adapter_name": host.adapter_name,
        "state_unchanged": True,
        "numerical_residual_scalar_kg": float(host.numerical_residual_scalar_kg),
        "transfer_counter_delta": dataclasses.asdict(counters_after.delta(counters_before)),
    }


def run_probe(
    *,
    backend: str = "numpy",
    device_id: int = 0,
    expected_maple_root: str | Path | None = None,
    include_file_manifest: bool = False,
) -> dict[str, Any]:
    """Run every Phase 1 probe step and return a JSON-serializable report.

    Raises `MapleDependencyError` for a missing or mismatched MAPLE,
    MAPLE's `BackendUnavailableError` for an unusable requested backend
    (never a silent CPU fallback), and `ProbeCheckError` for a failed
    check.
    """
    dependency = resolve_maple_dependency(expected_maple_root)
    maple_provenance = capture_maple_provenance(dependency, include_files=include_file_manifest)

    from maple.core.backend import resolve_backend

    resolved = resolve_backend(backend, device_id=device_id)
    state = build_minimal_state()
    inventories = check_inventories(state)
    exchange = exercise_zero_demand_exchange(state, resolved)

    digest_after = source_tree_digest(dependency.package_dir).digest_sha256
    stable = digest_after == maple_provenance["package_source_digest"]["digest_sha256"]
    g = state.geometry
    return {
        "schema": PROBE_SCHEMA,
        "status": "ok" if stable else "maple_source_changed_during_probe",
        "maple_syrup_version": maple_syrup.__version__,
        "backend": {
            "requested": backend,
            "resolved": resolved.backend.value,
            "device_id": resolved.device_id,
            "fingerprint": resolved.fingerprint,
        },
        "gpu_readiness_claimed": False,
        "maple": maple_provenance,
        "maple_source_digest_after_probe": digest_after,
        "source_stable_during_probe": stable,
        "maple_syrup": capture_syrup_provenance(include_files=include_file_manifest),
        "environment": environment_record(),
        "state": {
            "ny": g.ny,
            "nx": g.nx,
            "dx_m": g.dx_m,
            "dy_m": g.dy_m,
            "n_voxels": int(state.voxel_column.mass_kg.shape[2]),
            "voxel_dz_m": g.voxel_dz_m,
            "bulk_density_kg_m3": g.bulk_density_kg_m3,
            "active_layer_thickness_m": g.active_layer_thickness_m,
            "class_ids": [c.class_id for c in state.grain_classes.classes],
            "class_diameters_m": [c.diameter_m for c in state.grain_classes.classes],
            "class_fractions": state.class_fractions.tolist(),
            "boundary_x": g.boundary_x.kind.value,
            "boundary_y": g.boundary_y.kind.value,
            "mass_resolution_kg": state.mass_resolution_kg,
        },
        "inventory_checks": inventories,
        "zero_demand_exchange": exchange,
        "limitations": [
            "No rainfall, infiltration, routing, detachment or deposition physics is exercised.",
            "Sediment availability tracking is not enabled in the probe state.",
            (
                "A backend other than numpy only round-trips a tiny state; it is not evidence "
                "of GPU readiness or performance."
            ),
        ],
    }


def _backend_unavailable_error_type():
    try:
        from maple.core.backend import BackendUnavailableError
    except Exception:  # noqa: BLE001 - maple itself unavailable; handled by run_probe
        return ()
    return BackendUnavailableError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m maple_syrup.probe",
        description="MAPLE-SYRUP Phase 1 dependency, state and provenance probe.",
    )
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--expected-maple-root",
        help="Refuse to run unless the imported MAPLE resolves to this source root "
        "(default: $MAPLE_SYRUP_EXPECTED_MAPLE_ROOT when set).",
    )
    parser.add_argument("--output", help="Also write the JSON report to this NEW file.")
    parser.add_argument(
        "--include-file-manifest", action="store_true",
        help="Include per-file SHA-256 entries in the provenance record.",
    )
    args = parser.parse_args(argv)

    output = Path(args.output) if args.output else None
    if output is not None and output.exists():
        print(f"refusing to overwrite existing file {output}", file=sys.stderr)
        return 2

    try:
        report = run_probe(
            backend=args.backend,
            device_id=args.device_id,
            expected_maple_root=args.expected_maple_root,
            include_file_manifest=args.include_file_manifest,
        )
    except MapleDependencyError as exc:
        print(f"MAPLE dependency check failed: {exc}", file=sys.stderr)
        return 2
    except ProbeCheckError as exc:
        print(f"probe check failed: {exc}", file=sys.stderr)
        return 1
    except _backend_unavailable_error_type() as exc:
        print(f"backend unavailable: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2, sort_keys=True)
    if output is not None:
        source_root = report["maple"]["dependency"]["source_root"]
        if source_root is not None and output.resolve().is_relative_to(Path(source_root)):
            print(f"refusing to write inside the MAPLE source tree {source_root}", file=sys.stderr)
            return 2
        with output.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
