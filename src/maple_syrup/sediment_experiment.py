"""Phase 5b: Plot 1 coupled water + wet-sediment event on the actual MAPLE case.

    python -m maple_syrup.sediment_experiment --case-dir outputs/plot1 \\
        --output-dir outputs/plot1_sediment_dt1 --max-dt-s 1 --implementation numba

The verified Phase 2 case is re-verified (`case_import.verify_plot1_case`),
its hash-checked rainfall parsed, the Phase 3 column parameters and the
Phase 4 routing graph built as in `storm_experiment`, and the wet
sediment laws parameterised from the ORIGINAL root `mahleran_input.xml`
(raindrop a/b/c/hs per class with the legacy `spa / 1200` scaling applied
inside the law, particle density, `active_layer_sensitivity` hz,
`KE_model_type`, `time_step` reference interval) with the six class
diameters and the particle density taken from -- and required to equal --
the actual MAPLE grain classes. Vegetation cover is the Phase 2 sidecar
(prescribed, no growth). The event runs on the MAPLE bed loaded from the
case through `sediment_bed.bed_from_case`: the imported case configures
`water_coupling.enabled = false`, `event_kind = aeolian`,
`depth_update_rule = constant_free_surface`; this run overrides ONLY the
per-run depth ownership to `constant_depth` (recorded, with the resolved
configuration hash) and calls `apply_water_process_demand` directly with
the explicit adapter label `maple_syrup/phase5`. Nothing is relabelled as a
MAPLE stock water solver and no case file is modified.

Outputs (a NEW directory, never inside the case, MAPLE, MAHLERAN or recipe
trees; assembled in a temporary sibling directory and renamed into place
only after every budget, closure and source-stability check passes, so a
failed run leaves nothing): `sediment_summary.json`, `final_state.npz`
(bed, active layer, availability, committed terrain, water depth, mobile
mass, soil water, discharge, velocity, sediment-velocity memory, cumulative
pickup / deposition / export grids), `hydrograph.npz` / `hydrograph.csv`
(Phase 4 water columns plus the per-class sediment columns), and
`maple_final_snapshot.npz` written by MAPLE's own `save_state_snapshot`
(bed, availability and water; validated by MAPLE). The snapshot is a MAPLE
state record, NOT a SYRUP restart point: the hydraulic and transport
memory are only in `final_state.npz` for inspection and no resumability
of the SYRUP event is claimed. The run ends at the configured time; it is
not an event completion (no stop criterion, no dry reset).

Backends (CW-A1): `--backend numpy` (default) runs the array or numba
hydraulics on the host; `--backend cupy` runs ONLY the explicit CUDA
hydrology (`--implementation cuda`) with the whole small Plot 1 bed, the
graph, the column and the transport in CuPy through the actual MAPLE bed
API (`prepare_verified_sediment_case(backend="cupy")`). Every other
pairing (cupy + array/numba, numpy + cuda) is refused before the case is
touched: there is no fallback in either direction. A device memory
preflight (`cuda_memory_preflight`) estimates the whole-bed residency,
MAPLE's per-call whole-mass copies and the characteristic working set
before anything is allocated and refuses beds beyond the small-case cap
or the free device memory. The CUDA hydrology context is held in a
`CudaHydraulics` lifecycle (prepared once, rebuilt per commit) whose counts
and costs are reported in the summary's `backend` block; reporting
converts device results with MAPLE's host helpers and the loop timing is
synchronized before it is read. Not claimed: Chastre-scale beds, GPU
complete events, GPU checkpoints/restart, any speed-up.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup.case_import import (
    CLASS_IDS,
    Plot1ImportError,
    VerifiedPlot1Case,
    _json_safe,
    _refuse_output,
    _sha256_file,
    _write_new_json,
    _xml_record,
    _xml_value,
    parse_mahleran_xml,
    verify_plot1_case,
)
from maple_syrup.characteristic_transport import (
    DEFAULT_COURANT_MAX as CHARACTERISTIC_COURANT_MAX,
)
from maple_syrup.characteristic_transport import DEFAULT_N_BINS, MAX_N_BINS
from maple_syrup.column_experiment import (
    _Clock,
    _source_digests,
    _syrup_provenance,
    plot1_parameters,
)
from maple_syrup.dependency import MapleDependencyError
from maple_syrup.infiltration import (
    LOCAL_BALANCE_RTOL,
    InfiltrationError,
    column_parameters,
    initial_soil_water_m,
)
from maple_syrup.provenance import environment_record
from maple_syrup.rainfall import (
    RainfallError,
    parse_legacy_rainfall_file,
    rainfall_field,
)
from maple_syrup.routing import (
    BALANCE_RTOL,
    RoutingError,
    RoutingGraphError,
    plot1_routing_graph,
)
from maple_syrup.sediment_bed import (
    ADAPTER_NAME,
    BedIntegrationError,
    bed_from_case,
    terrain_reference,
)
from maple_syrup.sediment_event import (
    CUDA_MODES,
    PHASE_STORAGES,
    TRANSPORT_IMPLEMENTATIONS,
    TRANSPORT_SCHEMES,
    CudaHydraulics,
    SedimentEventControl,
    SedimentEventError,
    SedimentEventResult,
    SedimentEventState,
    evolve_sediment_event,
    initial_event_state,
    morphology_summary,
    water_channel_index,
)
from maple_syrup.sediment_physics import (
    KE_MODELS,
    LEGACY_CLASS_RADII_M,
    SedimentPhysicsError,
    SedimentPhysicsParameters,
    plot1_sediment_parameters,
    sediment_physics_parameters,
)
from maple_syrup.sediment_transport import TransportError
from maple_syrup.storm import (
    HYDROGRAPH_COLUMNS,
    STORM_IMPLEMENTATIONS,
    StormControl,
    StormError,
)

__all__ = [
    "BACKEND_PAIRINGS",
    "FINAL_NAME",
    "HYDROGRAPH_CSV",
    "HYDROGRAPH_NPZ",
    "PREFLIGHT_CASE_FIELDS",
    "PREFLIGHT_COEXISTING_STATES",
    "PREFLIGHT_PHASE_STATE_SETS",
    "SMALL_CASE_VOXEL_BYTES_MAX",
    "SNAPSHOT_NAME",
    "SUMMARY_NAME",
    "SedimentRun",
    "cuda_memory_preflight",
    "main",
    "plot1_sediment_parameters_from_xml",
    "run_plot1_sediment_event",
]

# Explicit backend / hydraulic pairings of THIS runner. The event API itself also accepts the generic array
# sweep on CuPy (tests/conservative_water); the runner only executes the explicit CUDA form on a device so a
# `--backend cupy` run never silently means "the reference loop on device arrays".
BACKEND_PAIRINGS = {"numpy": ("array", "numba"), "cupy": ("cuda",)}
# Small-case residency cap for the whole-bed-on-device form of CW-A1 (Plot 1 is ~1.2 MB): a larger voxel array is
# refused before allocation; memory-bounded (windowed) beds are a later MAPLE extension, not this runner.
SMALL_CASE_VOXEL_BYTES_MAX = 256 * 2**20
_PREFLIGHT_SAFETY_FACTOR = 2.0

SUMMARY_SCHEMA = "maple_syrup.plot1_sediment_event.v1"
SUMMARY_NAME = "sediment_summary.json"
FINAL_NAME = "final_state.npz"
HYDROGRAPH_CSV = "hydrograph.csv"
HYDROGRAPH_NPZ = "hydrograph.npz"
SNAPSHOT_NAME = "maple_final_snapshot.npz"
_EPS = float(np.finfo(np.float64).eps)
_STORM_RTOL = LOCAL_BALANCE_RTOL + BALANCE_RTOL
# Resolved first-case scientific conventions (Codex resolution, prompt).
RESOLVED_CONVENTIONS = {
    "ke_vegetation_form": "legacy_literal",
    "distance_convention": "legacy_literal",
    "dstar_convention": "legacy_sigma_minus_one",
    "bagnold_depth_units": "legacy_mm",
    "raindrop_composition_scaling": "legacy_none",
}
_KE_MODEL_BY_XML = {"1": KE_MODELS[0], "2": KE_MODELS[1]}


@dataclass(frozen=True)
class SedimentRun:
    summary: dict[str, Any]
    verified: VerifiedPlot1Case
    state: SedimentEventState  # final MAPLE bed / water / terrain plus SYRUP hydraulic and transport memory
    result: SedimentEventResult
    hydrograph: np.ndarray  # (n_rows, len(HYDROGRAPH_COLUMNS)) host
    sediment_hydrograph: np.ndarray  # (n_rows, len(result.sediment_columns)) host


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value) and value > 0.0):
        raise SedimentEventError(f"{name} must be finite and > 0, got {value!r}")
    return float(value)


def plot1_sediment_parameters_from_xml(xml_path: str | Path, grain_classes: Any, *, xp=None,
                                        **conventions: Any) -> tuple[SedimentPhysicsParameters, dict[str, Any]]:
    """Wet-law parameters from the ORIGINAL root XML (duplicates with
    conflicting content are refused by `_xml_record`) and the actual MAPLE
    grain classes. Diameters must be the legacy `shared_data.f90` radii
    doubled and every class must carry the XML particle density; the
    result must reproduce the module's transcribed Plot 1 constants."""
    xml = parse_mahleran_xml(xml_path)

    def per_class(tag: str) -> tuple[float, ...]:
        children = _xml_record(xml, tag)["children"]
        missing = [c for c in CLASS_IDS if c not in children]
        if missing:
            raise Plot1ImportError(f"mahleran XML: <{tag}> lacks {missing}")
        return tuple(float(children[c]) for c in CLASS_IDS)

    classes = grain_classes.classes
    if tuple(c.class_id for c in classes) != CLASS_IDS:
        raise SedimentEventError(f"MAPLE grain classes {[c.class_id for c in classes]} are not {list(CLASS_IDS)}")
    diameters = tuple(float(c.diameter_m) for c in classes)
    if diameters != tuple(2.0 * r for r in LEGACY_CLASS_RADII_M):
        raise SedimentEventError("MAPLE class diameters differ from the legacy shared_data.f90 radii doubled")
    densities = {float(c.particle_density_kg_m3) for c in classes}
    density_g_cm3 = float(_xml_value(xml, "particle_density"))
    if densities != {density_g_cm3 * 1000.0}:
        raise SedimentEventError(f"MAPLE particle densities {sorted(densities)} differ from the XML "
                                 f"{density_g_cm3} g/cm3")
    ke_raw = _xml_value(xml, "KE_model_type").strip()
    if ke_raw not in _KE_MODEL_BY_XML:
        raise SedimentEventError(f"unsupported KE_model_type {ke_raw!r}")
    kwargs = {
        "diameter_m": diameters,
        "raindrop_a": per_class("Raindrop_detachment_a_parameter_size"),
        "raindrop_b": per_class("Raindrop_detachment_b_parameter_size"),
        "raindrop_c": per_class("Raindrop_detachment_c_parameter_size"),
        "raindrop_max_depth_mm": per_class("Raindrop_detachment_max_parameter_size"),
        "particle_density_g_cm3": density_g_cm3,
        "active_layer_sensitivity_mm": float(_xml_value(xml, "active_layer_sensitivity")),
        "ke_model": _KE_MODEL_BY_XML[ke_raw],
        "reference_interval_s": float(_xml_value(xml, "time_step")),
        **RESOLVED_CONVENTIONS,
        **conventions,
    }
    params = sediment_physics_parameters(xp=xp, **kwargs)
    reference = plot1_sediment_parameters(xp=xp, **{k: kwargs[k] for k in RESOLVED_CONVENTIONS}).summary()
    summary = params.summary()
    if summary != reference:
        differing = sorted(k for k in summary if summary[k] != reference.get(k))
        raise SedimentEventError(f"XML-derived sediment parameters differ from the transcribed Plot 1 "
                                 f"constants in {differing}")
    record = {
        "source": "root mahleran_input.xml (parsed; conflicting duplicate elements refused) + MAPLE grain classes",
        "xml_path": str(Path(xml_path).resolve()),
        "xml_sha256": _sha256_file(Path(xml_path)),
        "ke_model_type_xml": ke_raw,
        "raindrop_scaling": "legacy spa / 1200 applied inside the law (initialize_values_xml 268)",
        "class_diameters_source": "MAPLE grain_classes (== 2 x shared_data.f90 radii, checked)",
        "particle_density_source": "XML particle_density; equals every MAPLE grain class (checked)",
        "matches_transcribed_plot1_constants": True,
        "parameters": summary,
    }
    return params, record


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise SedimentEventError(f"refusing to write into existing path {path}")


def phase_summary(result: SedimentEventResult) -> dict[str, Any]:
    """Host record of the transport scheme's phase bookkeeping for a summary."""
    from maple.core.backend import to_float

    from maple_syrup.characteristic_sparse import SparsePhaseState

    phase = result.state.phase
    csr = {}
    if isinstance(phase, SparsePhaseState):  # keys added only for the CSR storage; dense summaries are unchanged
        csr = {"phase_storage": "csr", "phase_nnz_final": int(phase.nnz), "phase_csr_bytes_final": int(phase.nbytes),
               "phase_storage_rule": ("nonempty-bin CSR of the same B-bin partition; the final arrays are the CSR "
                                      "arrays, no dense (ny, nx, nc, B) array was formed on the event path")}
    return {
        "scheme": "characteristic" if phase is not None else "upwind",
        "phase_bins": None if phase is None else int(phase.n_bins),
        "final_mobile_mass_partitioned_by_phase": phase is not None,
        **csr,
        "max_phase_reconciliation_residual_kg": to_float(result.max_phase_reconciliation_residual_kg),
        "reconciliation_rule": ("kernel remaining pool versus ACTUAL MAPLE pool after the deposit/export call within "
                                "summation_error_bound_kg(16, largest operand) per cell and class; cells whose actual "
                                "pool is exactly 0 are canonicalised; actual mass with a zero predicted pool keeps the "
                                "canonical upstream-face partition and is counted"),
        "n_phase_canonicalized": int(result.n_phase_canonicalized),
        "n_phase_rounding_remnants": int(result.n_phase_rounding_remnants),
        "phase_steered_cells_total": int(result.phase_steered_cells_total),
        "reroute_rule": ("rerouted cells keep their fractions and positions along the new outflow direction "
                         "(documented steering approximation); mass is never reset to the upstream face"),
    }


def _jsonable(value: Any) -> Any:
    """Plain JSON data: `_json_safe` for NumPy/Path values, then enums (by
    value), tuples and any remaining object (by `str`)."""
    return json.loads(json.dumps(_json_safe(value), default=lambda o: getattr(o, "value", str(o))))


# The MAPLE state objects `bed_from_case` places on the device (every array reachable from them, including the
# ledger's eight (n_process, ny, nx, nc) arrays, its pending / touch grids and process totals, the committed
# topography's reference arrays and the voxel surface cache when present).
PREFLIGHT_CASE_FIELDS = ("voxel_column", "active_layer", "water", "sediment_ledger", "committed_topography",
                         "sediment_availability")
# Full MAPLE state sets (every array of `PREFLIGHT_CASE_FIELDS`) that can be alive AT THE SAME TIME on the device
# during one accepted step of the runner's loop. This is a documented, intentionally CONSERVATIVE upper bound on the
# concurrent lifetimes (correction 2, root lifetime review), not an exact peak and not a memory-fit claim:
#   1 the runner's retained initial state `state0` (kept for reporting after the accepted state has moved on);
#   2 the current accepted event state;
#   3-4 the PREVIOUS attempt's pickup and post-deposit MAPLE results (the last `SedimentCoupledStep` stays referenced
#       while the right-hand side of the next `attempt = sediment_coupled_step(...)` is evaluated);
#   5-6 the CURRENT attempt's pickup and post-deposit results;
#   7 the commit candidate (MAPLE's committed bed, which can differ from the post-deposit state);
#   8 one within-call generation (the new arrays an exchange / commit call builds before it returns).
# MAPLE's whole-voxel working copies and the commit snapshots are kept as separate terms on top of these sets.
# CW-B20: the runner's loop now holds the private accounting record instead of the complete `SedimentCoupledStep`
# (items 3-6 hold fewer arrays in practice). The bound is deliberately left unchanged: it stays a conservative
# upper bound, and no estimate is lowered without a measured profile.
PREFLIGHT_COEXISTING_STATES = 8
# Full characteristic phase-state sets (fraction + position, each (ny, nx, nc, B)) that can be alive at once:
#   1 the retained initial state's phase; 2 the current accepted phase; 3 the previous attempt's kernel prediction
#   (`CharacteristicStep.phase_after`, alive with the previous attempt); 4 the current kernel prediction;
#   5 the current reconciled phase (`reconcile_phase` builds new fraction / position arrays).
# The kernel's ~12-array working set is a separate margin term. Conservative upper bound, not an exact peak.
PREFLIGHT_PHASE_STATE_SETS = 5


def _unique_arrays(obj: Any, seen: set[int]) -> list[Any]:
    """Every array reachable through dataclass fields (init and non-init), tuples, lists and dicts of `obj`,
    each object counted once (identity), in a deterministic walk order."""
    from maple.core.backend import is_array

    found: list[Any] = []
    if obj is None or id(obj) in seen:
        return found
    if is_array(obj):
        seen.add(id(obj))
        return [obj]
    seen.add(id(obj))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            found += _unique_arrays(getattr(obj, f.name, None), seen)
    elif isinstance(obj, dict):
        for value in obj.values():
            found += _unique_arrays(value, seen)
    elif isinstance(obj, (tuple, list)):
        for value in obj:
            found += _unique_arrays(value, seen)
    return found


def cuda_memory_preflight(case, *, phase_bins: int, device_id: int, transport_scheme: str = "characteristic",
                          voxel_bytes_max: int = SMALL_CASE_VOXEL_BYTES_MAX,
                          safety_factor: float = _PREFLIGHT_SAFETY_FACTOR, phase_storage: str = "dense",
                          phase_max_nnz: int | None = None) -> dict[str, Any]:
    """Estimate the device residency of a whole-bed CuPy event on the loaded
    MAPLE `case` BEFORE anything is allocated on the device, and refuse it
    (`SedimentEventError`) when the voxel array exceeds the small-case cap or
    the safety-scaled estimate exceeds the free memory of `device_id`.

    Order: argument validation (strict ints for the bins, device and cap, a
    supported transport scheme, a finite safety factor >= 1), the actual
    byte count of every unique array of the MAPLE state fields that
    `bed_from_case` places on the device (`PREFLIGHT_CASE_FIELDS`, recursive,
    identity-deduplicated: the ledger's eight process arrays, pending /
    touch grids and process totals included), the estimate, the small-case
    voxel cap, and ONLY THEN the device query. Terms (bytes): the MAPLE
    state times `PREFLIGHT_COEXISTING_STATES` (= 8, the documented
    conservative upper bound on the full state sets that can be alive at
    once: retained initial, current accepted, previous attempt's pickup and
    deposit results, current attempt's pickup and deposit results, commit
    candidate, one within-call generation); MAPLE's whole-voxel working
    copies inside an exchange call (the copied array and the returned one)
    and the commit's voxel + active-layer snapshots; the characteristic
    phase (fraction + position) times `PREFLIGHT_PHASE_STATE_SETS` (= 5:
    retained initial, current accepted, previous kernel prediction, current
    kernel prediction, current reconciled) and the kernel's ~12-array
    working set (`(n_cells, nc, B + 1)` each); the CUDA hydrology statics,
    per-attempt scratch rows and the three storm-state grids of the
    retained / intermediate / result states; SYRUP's accumulators. The
    lifetime multipliers are intentionally conservative upper bounds (no
    exact-peak and no memory-fit claim until measured); the safety factor
    is an additional margin on top of these terms, not a substitute for
    them. An estimate, not a measurement: the runner also reports the CuPy
    pool after the loop. A later memory-bounded (CW-B) representation gets
    its own preflight counting its production live buffers and owned
    transactions; this one is for the whole-small-bed form only."""
    from maple.core.backend import cupy_module

    if transport_scheme not in TRANSPORT_SCHEMES:
        raise SedimentEventError(f"transport_scheme must be one of {TRANSPORT_SCHEMES}, got {transport_scheme!r}")
    if phase_storage not in PHASE_STORAGES:
        raise SedimentEventError(f"phase_storage must be one of {PHASE_STORAGES}, got {phase_storage!r}")
    if phase_storage == "csr" and transport_scheme != "characteristic":
        raise SedimentEventError("phase_storage 'csr' is a storage of the characteristic scheme's phase")
    if phase_max_nnz is not None and (phase_storage != "csr" or isinstance(phase_max_nnz, bool)
                                      or not isinstance(phase_max_nnz, (int, np.integer)) or int(phase_max_nnz) < 0):
        raise SedimentEventError("phase_max_nnz must be an int >= 0 and needs phase_storage 'csr'")
    for name, value, minimum in (("phase_bins", phase_bins, 1), ("device_id", device_id, 0),
                                 ("voxel_bytes_max", voxel_bytes_max, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
            raise SedimentEventError(f"{name} must be an int >= {minimum}, got {value!r}")
    if int(phase_bins) > MAX_N_BINS:
        raise SedimentEventError(f"phase_bins must be <= {MAX_N_BINS}, got {phase_bins!r}")
    if (isinstance(safety_factor, bool) or not isinstance(safety_factor, (int, float, np.integer, np.floating))
            or not math.isfinite(float(safety_factor)) or float(safety_factor) < 1.0):
        raise SedimentEventError(f"safety_factor must be a finite real >= 1, got {safety_factor!r}")
    bins = int(phase_bins) if transport_scheme == "characteristic" else 0
    factor = float(safety_factor)
    cap = int(voxel_bytes_max)
    device = int(device_id)

    g = case.config.geometry
    voxel = case.voxel_column.mass_kg
    if getattr(voxel, "ndim", 0) != 4:
        raise SedimentEventError("case voxel column must be (ny, nx, nv, nc)")
    ny, nx, nv, nc = (int(v) for v in voxel.shape)
    if (ny, nx) != (g.ny, g.nx):
        raise SedimentEventError("case voxel column does not match the case geometry")
    seen: set[int] = set()
    field_bytes: dict[str, int] = {}
    field_arrays: dict[str, int] = {}
    for name in PREFLIGHT_CASE_FIELDS:
        value = getattr(case, name, None)
        if value is None:
            raise SedimentEventError(f"case lacks the MAPLE state field {name!r} needed for the device estimate")
        arrays = _unique_arrays(value, seen)
        field_bytes[name] = int(sum(int(a.nbytes) for a in arrays))
        field_arrays[name] = len(arrays)
    case_array_bytes = int(sum(field_bytes.values()))
    n_cells = ny * nx
    f64 = 8
    voxel_bytes = int(voxel.nbytes)
    active_bytes = n_cells * nc * f64
    per_class = n_cells * nc * f64
    per_cell = n_cells * f64
    coexisting_states = PREFLIGHT_COEXISTING_STATES * case_array_bytes  # documented concurrent-lifetime upper bound
    maple_working = 2 * voxel_bytes  # whole-mass copy + returned array inside one exchange routine
    commit_snapshots = 2 * (voxel_bytes + active_bytes)  # commit_topography byte-for-byte snapshots
    csr_record: dict[str, Any] = {}
    if phase_storage == "csr" and bins:
        # CW-B5 sparse CSR storage: the phase is `nnz` entries of 17 B plus (n_rows + 1) int64 pointers; `nnz` is
        # bounded by the predeclared capacity (`phase_max_nnz`, enforced by the operator BEFORE its write allocation)
        # or, when no capacity is declared, by full occupancy n_rows x B (the operator itself REQUIRES an explicit
        # capacity above its large-case threshold and never silently sizes every bin). Two CSR generations live during
        # a substep; the kernel working set is the (n_active, nc) bookkeeping, the (n, nc) contract fields, the faces
        # and the launched threads' bounded local scratch. Reported as an ESTIMATE / upper bound, not a fit claim.
        from maple_syrup.characteristic_cuda_csr import (
            DEFAULT_UNBOUNDED_NNZ_BYTES,
            estimate_local_scratch_bytes,
            launch_shape,
        )
        from maple_syrup.characteristic_sparse import ENTRY_BYTES

        n_rows = n_cells * nc  # every cell active in the whole-small-bed form (an upper bound otherwise)
        full_entries = n_rows * bins
        cap_entries = full_entries if phase_max_nnz is None else min(int(phase_max_nnz), full_entries)
        phase_set = cap_entries * ENTRY_BYTES + (n_rows + 1) * 8
        phase = 2 * phase_set + 2 * (n_rows + 1) * 8  # two CSR generations + two pointer arrays
        _grid, _block, threads = launch_shape(n_rows)
        local = estimate_local_scratch_bytes(bins)
        transport_working = ((7 + 6) * n_rows * f64 + 12 * per_class + 2 * (ny * (nx + 1) + (ny + 1) * nx) * nc * f64
                             + threads * local["ms_csr_write"])
        csr_record = {
            "phase_storage": "csr",
            "csr_entry_bytes": ENTRY_BYTES,
            "csr_full_occupancy_entries": full_entries,
            "csr_full_occupancy_bytes": full_entries * ENTRY_BYTES,
            "csr_capacity_entries": cap_entries,
            "csr_capacity_basis": "phase_max_nnz" if phase_max_nnz is not None else "full occupancy (no capacity declared)",
            "csr_capacity_required_by_operator": full_entries * ENTRY_BYTES > DEFAULT_UNBOUNDED_NNZ_BYTES,
            "csr_operator_unbounded_threshold_bytes": DEFAULT_UNBOUNDED_NNZ_BYTES,
            "csr_generations": 2,
            "csr_launched_threads": threads,
            "csr_local_scratch_estimate_bytes": threads * local["ms_csr_write"],
            "csr_rule": ("the operator refuses a substep whose scanned nnz exceeds phase_max_nnz before any write "
                         "allocation and never truncates mass or reduces B / classes; peak nnz is reported by the "
                         "step (peak_nnz), the estimate here is an upper bound, not a resource-fit claim"),
        }
    else:
        phase_set = 2 * n_cells * nc * bins * f64  # one phase state: fraction + position
        phase = PREFLIGHT_PHASE_STATE_SETS * phase_set  # concurrent phase lifetimes (upper bound, see the constant)
        transport_working = 12 * n_cells * nc * (bins + 1) * f64 if bins else 8 * per_class
    hydrology = n_cells * (2 + 8 + 8 * 6) + 20 * per_cell + 3 * 3 * per_cell  # statics, scratch rows, storm states
    accounting = 8 * per_class + 16 * per_cell
    total = coexisting_states + maple_working + commit_snapshots + phase + transport_working + hydrology + accounting
    required = math.ceil(factor * total)
    record = {
        **csr_record,
        "device_id": device, "shape": [ny, nx, nv, nc], "phase_bins": bins, "transport_scheme": transport_scheme,
        "voxel_bytes": voxel_bytes, "case_array_bytes": case_array_bytes, "case_field_bytes": field_bytes,
        "case_field_arrays": field_arrays, "coexisting_states": PREFLIGHT_COEXISTING_STATES,
        "coexisting_state_basis": ("retained initial state0, current accepted state, previous attempt pickup and "
                                   "deposit results (alive during the next attempt's evaluation), current attempt "
                                   "pickup and deposit results, commit candidate bed, one within-call generation; "
                                   "a conservative upper bound on concurrent lifetimes, not an exact peak"),
        "coexisting_state_bytes": coexisting_states, "maple_working_bytes": maple_working,
        "commit_snapshot_bytes": commit_snapshots, "phase_state_bytes": phase_set,
        "phase_state_sets": PREFLIGHT_PHASE_STATE_SETS,
        "phase_state_basis": ("retained initial phase, current accepted phase, previous kernel prediction, current "
                              "kernel prediction, current reconciled phase; conservative upper bound"),
        "phase_bytes": phase,
        "transport_working_bytes": transport_working, "hydrology_bytes": hydrology, "accounting_bytes": accounting,
        "estimate_bytes": total, "safety_factor": factor, "required_bytes": required, "voxel_bytes_max": cap,
        "basis": "actual bytes of every unique array of the MAPLE state fields placed on the device x "
                 f"{PREFLIGHT_COEXISTING_STATES} concurrently alive state sets (documented upper bound) + MAPLE "
                 "whole-voxel working copies and commit snapshots + characteristic phase (fraction + position) x "
                 f"{PREFLIGHT_PHASE_STATE_SETS} concurrently alive phase sets + the 12-array kernel working set + "
                 "CUDA statics/scratch/storm states + accumulators; safety factor as an additional margin; an "
                 "estimate taken BEFORE any device allocation, not a measurement and not a memory-fit claim",
    }
    if voxel_bytes > cap:
        raise SedimentEventError(
            f"refusing the cupy backend: the case voxel array is {voxel_bytes} bytes, above the small-case "
            f"whole-bed cap {cap} bytes of this milestone (memory-bounded beds are not implemented)")
    cp = cupy_module()
    if cp is None:
        raise SedimentEventError("cuda_memory_preflight needs CuPy; the cupy backend is unavailable (no fallback)")
    try:
        free_bytes, total_bytes = (int(v) for v in cp.cuda.Device(device).mem_info)
    except Exception as exc:
        raise SedimentEventError(f"cuda_memory_preflight could not query device {device}: {exc}") from exc
    record["device_free_bytes"] = free_bytes
    record["device_total_bytes"] = total_bytes
    if required > free_bytes:
        raise SedimentEventError(
            f"refusing the cupy backend: the safety-scaled device estimate {required} bytes exceeds the free memory "
            f"{free_bytes} bytes of CUDA device {device}; nothing was allocated")
    record["accepted"] = True
    return record


def prepare_verified_sediment_case(verified, *, backend="numpy", mahleran_root=None, end_s=None, control=None):
    """Shared actual-MAPLE case/physics setup for fixed and complete events.

    Input must come from verify_plot1_case. No simulation, mutation, output,
    or wind configuration is created here. Returned fields are the same
    objects the Phase5 runner previously constructed inline. `control`
    (a `SedimentEventControl`) selects the transport scheme and phase-bin
    count the initial event state is built for; None means the default
    control (characteristic scheme, default bins).
    """
    from maple.core.backend import resolve_backend, to_device, to_host
    from maple.core.types.topographic_commit import committed_elevation_m

    recipe_record = verified.report["recipe"]
    case = verified.case
    settings = verified.report["legacy_options"]["settings"]
    schedule = parse_legacy_rainfall_file(verified.rainfall_path)
    if schedule.provenance.sha256 != verified.report["rainfall"]["sha256"]:
        raise SedimentEventError("parsed rainfall bytes differ from the verified rainfall hash")
    end = float(settings["stormlength_s"]) if end_s is None else float(end_s)
    host, parameter_record = plot1_parameters(verified.report, verified.fields)
    g = case.config.geometry
    ny, nx, n_classes = g.ny, g.nx, len(case.config.grain_classes.classes)
    if host["theta_sat"].shape != (ny, nx) or g.dx_m != g.dy_m:
        raise SedimentEventError("sidecar fields do not match the MAPLE grid, or cells are not square")
    area = g.dx_m * g.dy_m
    veg_host = np.asarray(verified.fields["vegetation_cover_fraction"], dtype=np.float64)
    if veg_host.shape != (ny, nx):
        raise SedimentEventError("vegetation sidecar does not match the MAPLE grid")

    # --- original XML: bound before the loop, re-hashed after -----------------------------------
    xml_path = Path(mahleran_root or recipe_record["mahleran_root"]).resolve() / recipe_record["xml_name"]
    xml_sha_before = _sha256_file(xml_path)
    if xml_sha_before != verified.report["mahleran"]["xml_sha256"]:
        raise SedimentEventError(f"root XML {xml_path} does not hash to the verified import's xml_sha256")

    resolved = resolve_backend(backend)
    xp = resolved.xp
    sediment, sediment_record = plot1_sediment_parameters_from_xml(xml_path, case.config.grain_classes, xp=xp)
    graph = plot1_routing_graph(verified.fields, verified.report, xp=xp)
    if graph.dx_m != g.dx_m:
        raise SedimentEventError("routing graph spacing differs from the MAPLE geometry")
    dev = {name: to_device(array, xp) for name, array in host.items()}
    column = column_parameters(
        model="pavement_hawkins",
        **{k: dev[k] for k in ("ksat_m_per_s", "suction_m", "drainage_parameter", "theta_sat",
                               "soil_thickness_m", "pavement_cover_fraction")},
    )
    field = rainfall_field(ny, nx, scale=dev["rainfall_scale"])
    vegetation = to_device(veg_host, xp)
    soil0 = initial_soil_water_m(column, dev["initial_theta"])

    # --- the actual MAPLE bed and the per-run depth-ownership override --------------------------
    bed0, context = bed_from_case(case, xp=xp)
    water_case = case.config.water_coupling
    if context.depth_update_rule != "constant_depth":
        raise SedimentEventError("bed context must resolve the constant_depth ownership rule")
    full_z = np.asarray(verified.fields["legacy_full_elevation_m"], dtype=np.float64)
    full_rm = np.asarray(verified.fields["legacy_full_rainfall_scaling"], dtype=np.float64)
    terrain = terrain_reference(full_z, full_rm < 0.0, bed0, graph)
    state0 = initial_event_state(graph, terrain, bed0, context, soil0, t_s=0.0, control=control)
    initial_elevation = to_host(committed_elevation_m(bed0.committed_topography)).copy()
    initial_active = to_host(bed0.active_layer.mass_kg).copy()
    initial_available = to_host(bed0.sediment_availability.available_mass_kg).copy()

    return {
        "area": area,
        "case": case,
        "column": column,
        "context": context,
        "end": end,
        "field": field,
        "g": g,
        "graph": graph,
        "host": host,
        "initial_active": initial_active,
        "initial_available": initial_available,
        "initial_elevation": initial_elevation,
        "n_classes": n_classes,
        "nx": nx,
        "ny": ny,
        "parameter_record": parameter_record,
        "resolved": resolved,
        "schedule": schedule,
        "sediment": sediment,
        "sediment_record": sediment_record,
        "settings": settings,
        "soil0": soil0,
        "state0": state0,
        "vegetation": vegetation,
        "water_case": water_case,
        "xml_path": xml_path,
        "xml_sha_before": xml_sha_before,
        "xp": xp,
    }


def run_plot1_sediment_event(
    case_dir: str | Path,
    output_dir: str | Path,
    *,
    max_dt_s: float = 1.0,
    end_s: float | None = None,
    implementation: str = "numba",
    backend: str = "numpy",
    report_every_s: float = 60.0,
    min_dt_s: float = 1.0 / 1024.0,
    max_retries: int = 10,
    max_steps: int = 10_000_000,
    max_report_rows: int = 100_000,
    sediment_courant_max: float = CHARACTERISTIC_COURANT_MAX,
    max_transport_substeps: int = 64,
    commit: bool = True,
    force_final_commit: bool = True,
    transport_scheme: str = "characteristic",
    phase_bins: int = DEFAULT_N_BINS,
    transport_implementation: str = "auto",
    mahleran_root: str | Path | None = None,
    expected_maple_root: str | Path | None = None,
    allow_maple_source_change: bool = False,
    root_solver: str = "bisection",
    cuda_mode: str = "auto",
    phase_storage: str = "dense",
    phase_max_nnz: int | None = None,
) -> SedimentRun:
    from maple.core.backend import (
        BackendUnavailableError,
        read_transfer_counters,
        resolve_backend,
        synchronize,
        to_host,
        to_host_tree,
    )
    from maple.core.parameters import config_to_dict
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.io.outputs.snapshot import save_state_snapshot
    from maple.water import validate_water_state

    from maple_syrup import routing_numba

    max_dt = _positive(max_dt_s, "max_dt_s")
    cadence = _positive(report_every_s, "report_every_s")
    if end_s is not None:
        _positive(end_s, "end_s")
    storm_control = StormControl(max_dt_s=max_dt_s, min_dt_s=min_dt_s, max_retries=max_retries, max_steps=max_steps,
                                 implementation=implementation, root_solver=root_solver).validated()
    control = SedimentEventControl(storm=storm_control, sediment_courant_max=sediment_courant_max,
                                   max_transport_substeps=max_transport_substeps, commit=commit,
                                   force_final_commit=force_final_commit, transport_scheme=transport_scheme,
                                   phase_bins=phase_bins, transport_implementation=transport_implementation,
                                   phase_storage=phase_storage, phase_max_nnz=phase_max_nnz).validated()
    if isinstance(max_report_rows, bool) or not isinstance(max_report_rows, int) or max_report_rows < 1:
        raise SedimentEventError(f"max_report_rows must be a positive int, got {max_report_rows!r}")
    if implementation not in STORM_IMPLEMENTATIONS:
        raise SedimentEventError(f"implementation must be one of {STORM_IMPLEMENTATIONS}, got {implementation!r}")
    if backend not in BACKEND_PAIRINGS:
        raise SedimentEventError(f"backend {backend!r} is refused: must be one of {tuple(BACKEND_PAIRINGS)}")
    if implementation not in BACKEND_PAIRINGS[backend]:
        raise SedimentEventError(
            f"backend {backend!r} with implementation {implementation!r} is refused: this runner pairs numpy with "
            f"{BACKEND_PAIRINGS['numpy']} (host hydraulics) and cupy ONLY with 'cuda' (the explicit prepared CUDA "
            "hydrology on a CuPy state); CUDA is not a CPU fallback and the host sweeps make no device transfer. "
            "The generic array sweep on CuPy arrays is reachable through the event API, not this runner")
    if cuda_mode not in CUDA_MODES:
        raise SedimentEventError(f"cuda_mode must be one of {CUDA_MODES}, got {cuda_mode!r}")
    needs_numba = implementation == "numba" or (control.characteristic
                                                and control.resolved_transport_implementation() == "numba")
    if needs_numba and not routing_numba.numba_available():
        raise SedimentEventError("implementation 'numba' requested but Numba is not importable (optional extra "
                                 "maple-syrup[numba]); there is no fallback -- pass --implementation array explicitly")
    output_dir = Path(output_dir).resolve()
    _ensure_absent(output_dir)
    if output_dir.is_relative_to(Path(case_dir).resolve()):
        raise SedimentEventError("refusing to write inside the bound case directory")

    clock = _Clock()
    syrup_provenance = _syrup_provenance()
    verified = verify_plot1_case(case_dir, mahleran_root=mahleran_root, expected_maple_root=expected_maple_root,
                                 allow_maple_source_change=allow_maple_source_change)
    dependency = verified.maple_dependency
    recipe_record = verified.report["recipe"]
    _refuse_output(output_dir, {
        "MAPLE source": dependency.source_root,
        "MAPLE package": dependency.package_dir,
        "MAHLERAN": Path(mahleran_root or recipe_record["mahleran_root"]).resolve(),
        "recorded MAHLERAN": Path(recipe_record["mahleran_root"]).resolve(),
        "recipe": Path(recipe_record["recipe_path"]).resolve().parent,
    })
    digests_before = {
        "maple_syrup": syrup_provenance["package_source_digest"]["digest_sha256"],
        "maple": verified.maple_provenance["package_source_digest"]["digest_sha256"],
    }
    # Resolve the backend (selects the CUDA device; raises BackendUnavailableError, never falls back) and, on a
    # device, estimate the whole-bed residency BEFORE `prepare_verified_sediment_case` allocates anything there.
    try:
        resolved_backend = resolve_backend(backend)
    except BackendUnavailableError as exc:
        raise SedimentEventError(f"backend {backend!r} is refused: {exc}") from exc
    memory_preflight = None
    hydraulics = None
    if backend == "cupy":
        memory_preflight = cuda_memory_preflight(verified.case, phase_bins=phase_bins,
                                                 device_id=resolved_backend.device_id,
                                                 transport_scheme=transport_scheme, phase_storage=phase_storage,
                                                 phase_max_nnz=phase_max_nnz)
        hydraulics = CudaHydraulics(mode=cuda_mode)
    prepared = prepare_verified_sediment_case(verified, backend=backend, mahleran_root=mahleran_root, end_s=end_s,
                                              control=control)
    area = prepared["area"]
    case = prepared["case"]
    column = prepared["column"]
    context = prepared["context"]
    end = prepared["end"]
    field = prepared["field"]
    g = prepared["g"]
    graph = prepared["graph"]
    host = prepared["host"]
    initial_active = prepared["initial_active"]
    initial_available = prepared["initial_available"]
    initial_elevation = prepared["initial_elevation"]
    n_classes = prepared["n_classes"]
    nx = prepared["nx"]
    ny = prepared["ny"]
    parameter_record = prepared["parameter_record"]
    resolved = prepared["resolved"]
    schedule = prepared["schedule"]
    sediment = prepared["sediment"]
    sediment_record = prepared["sediment_record"]
    settings = prepared["settings"]
    soil0 = prepared["soil0"]
    state0 = prepared["state0"]
    vegetation = prepared["vegetation"]
    water_case = prepared["water_case"]
    xml_path = prepared["xml_path"]
    xml_sha_before = prepared["xml_sha_before"]
    xp = prepared["xp"]

    override = {
        "case_water_coupling": _jsonable(dataclasses.asdict(water_case)),
        "run_depth_update_rule": context.depth_update_rule,
        "rule": "per-run override of water depth ownership only; the case file is unchanged; MAPLE's stock "
                "water adapters are not used and nothing is relabelled as one",
        "adapter_name": ADAPTER_NAME,
        "detachment_integration": "rate_times_dt",
        "commit_path": "sediment_bed.commit_bed: MAPLE evaluate_commit_triggers + commit_topography with the "
                       "case's avalanching spec and the constant_depth water callback; no scheduler wind tables",
        "avalanching_enabled_in_commits": bool(case.config.avalanching.enabled),
    }
    resolved_config = {
        "schema": SUMMARY_SCHEMA,
        "maple_config": _jsonable(config_to_dict(case.config)),
        "maple_case_identity_sha256": verified.binding["maple_case_identity_sha256"],
        "syrup": {
            "override": override,
            "sediment_parameters": sediment_record["parameters"],
            "conventions": RESOLVED_CONVENTIONS,
            "travel_distance_policy": "local hydraulics (deposition rate 1/L from the current cell's law)",
            "sediment_velocity_memory": "explicit recession exp(-dt/tau), tau = -1 s / ln 0.9, uncapped",
            "time_levels": "post-route consistent depth and velocity; rain rate of the step",
            "reference_pickup_interval_s": sediment.reference_interval_s,
            "transaction": "two MAPLE water calls per accepted step: pickup, then transport, then deposit/export",
            "storm_control": dataclasses.asdict(storm_control),
            "transport": control.transport_record(),
            "sediment_courant_max": control.sediment_courant_max,
            "max_transport_substeps": control.max_transport_substeps,
            "commit": control.commit,
            "force_final_commit": control.force_final_commit,
            "end_s": end,
            "report_every_s": cadence,
            "implementation": implementation,
            "backend": backend,
            "column_parameters": parameter_record,
            "vegetation": "Phase 2 sidecar vegetation_cover_fraction (prescribed, no growth)",
            "graph_input_sha256": graph.input_sha256,
            "rainfall_sha256": schedule.provenance.sha256,
        },
    }
    if backend == "cupy":  # keys added only for device runs so the host resolved-config hash is unchanged
        resolved_config["syrup"]["cuda_mode"] = cuda_mode
        resolved_config["syrup"]["backend_pairing"] = control.backend_record()
    resolved_config = _jsonable(resolved_config)
    resolved_config_sha256 = hashlib.sha256(
        json.dumps(resolved_config, sort_keys=True).encode("utf-8")).hexdigest()
    synchronize(xp)  # device uploads of the setup complete before the setup time is read
    setup = clock.lap()

    # --- event loop ----------------------------------------------------------------------------------
    counters_before = read_transfer_counters()
    result = evolve_sediment_event(state0, context, column, field, schedule, vegetation, sediment, end, control,
                                   report_every_s=cadence, max_report_rows=max_report_rows, hydraulics=hydraulics)
    synchronize(xp)  # every launch of the loop has completed before the loop time and the pool are read
    loop = clock.lap()
    transfers = dataclasses.asdict(read_transfer_counters().delta(counters_before))
    device_memory = None
    if backend == "cupy":
        pool = xp.get_default_memory_pool()
        free_bytes, total_bytes = (int(v) for v in xp.cuda.Device(resolved_backend.device_id).mem_info)
        device_memory = {"pool_used_bytes_after_loop": int(pool.used_bytes()),
                         "pool_total_bytes_after_loop": int(pool.total_bytes()),
                         "device_free_bytes_after_loop": free_bytes, "device_total_bytes": total_bytes,
                         "note": "CuPy default pool after the synchronized loop, before host conversion; "
                                 "pool_total is the pool's allocated high-water mark of this process, not a "
                                 "per-kernel peak"}

    # --- reporting boundary --------------------------------------------------------------------------
    final = result.state
    storm_final, bed_final = final.storm, final.bed
    names = ["surface_initial", "soil_initial", "rain", "intake", "saturation_return", "drainage",
             "surface_final", "soil_final"]
    grids = [state0.storm.depth_m, soil0, result.cumulative_rain_m, result.cumulative_intake_m,
             result.cumulative_saturation_return_m, result.cumulative_drainage_m, storm_final.depth_m,
             storm_final.soil_water_m]
    scalars = to_host(xp.stack([xp.asarray(a.sum(), dtype=np.float64) for a in grids] + [
        xp.asarray(v, dtype=np.float64) for v in (
            result.cumulative_export_m3, result.peak_depth_m.max(), result.peak_velocity_m_s.max(),
            storm_final.depth_m.max(), (storm_final.depth_m > 0.0).sum(), storm_final.soil_water_m.min(),
            storm_final.soil_water_m.max(), result.max_courant_old, result.max_courant_new,
            result.cell_steps_no_runon, result.cell_steps_partial_runon, result.cell_steps_complete_runon,
            result.peak_outlet_discharge_m3_s, result.time_of_peak_outlet_s,
            result.max_routing_cell_balance_residual_m, result.max_constitutive_residual_m,
            result.peak_mobile_kg, result.time_of_peak_mobile_s, result.peak_export_rate_kg_s,
            result.time_of_peak_export_s, result.max_sediment_courant, result.max_decay_exponent,
            result.max_transport_cell_residual_kg, result.numerical_residual_scalar_abs_kg,
        )
    ]))
    sums = dict(zip(names, scalars[: len(names)].tolist()))
    (export_m3, peak_depth, peak_velocity, final_max_depth, ponded_cells, soil_min, soil_max, cr_old, cr_new,
     cells_no, cells_partial, cells_complete, true_peak_q, true_peak_t, max_balance, max_constitutive,
     peak_mobile, peak_mobile_t, peak_export_rate, peak_export_t, max_sed_courant, max_decay, max_cell_residual,
     residual_scalar) = scalars[len(names):].tolist()
    by_class = {k: to_host(v).astype(np.float64).copy() for k, v in result.by_class.items()}
    regime = {k: int(to_host(v)) for k, v in result.regime_cell_steps.items()}
    ledger_reset = to_host(result.ledger_process_totals_reset_kg).copy()
    committed_net = to_host(result.committed_net_bed_change_kg).copy()
    hydrograph = to_host(result.hydrograph).copy()
    sediment_hydrograph = to_host(result.sediment_hydrograph).copy()
    closure = result.closure()
    host_bed = to_host_tree(bed_final)
    final_elevation = to_host(committed_elevation_m(bed_final.committed_topography)).copy()
    grids_out = {
        "voxel_mass_kg": np.asarray(host_bed.voxel_column.mass_kg),
        "active_mass_kg": np.asarray(host_bed.active_layer.mass_kg),
        "available_mass_kg": np.asarray(host_bed.sediment_availability.available_mass_kg),
        "bound_mass_kg": np.asarray(host_bed.sediment_availability.bound_mass_kg),
        "committed_elevation_m": final_elevation,
        "initial_committed_elevation_m": initial_elevation,
        "pending_bed_mass_change_kg": np.asarray(host_bed.ledger.pending_bed_mass_change_kg),
        "depth_m": np.asarray(host_bed.water.depth_m),
        "mobile_mass_kg": np.asarray(host_bed.water.mobile_mass_by_cell_class_kg),
        "soil_water_m": to_host(storm_final.soil_water_m).copy(),
        "discharge_m2_s": to_host(storm_final.discharge_m2_s).copy(),
        "velocity_m_s": to_host(result.last_velocity_m_s).copy(),
        "sediment_velocity_m_s": to_host(final.sediment_velocity_m_s).copy(),
        "cumulative_pickup_kg": to_host(result.cumulative_pickup_kg).copy(),
        "cumulative_deposition_kg": to_host(result.cumulative_deposition_kg).copy(),
        "cumulative_export_request_kg": to_host(result.cumulative_export_request_kg).copy(),
        "peak_depth_m": to_host(result.peak_depth_m).copy(),
        "peak_velocity_m_s": to_host(result.peak_velocity_m_s).copy(),
        "cumulative_rain_m": to_host(result.cumulative_rain_m).copy(),
        "cumulative_intake_m": to_host(result.cumulative_intake_m).copy(),
        "cumulative_return_m": to_host(result.cumulative_saturation_return_m).copy(),
        "cumulative_drainage_m": to_host(result.cumulative_drainage_m).copy(),
        "final_aspect": np.asarray(final.graph.aspect),
        "final_slope": np.asarray(final.graph.slope),
        "initial_aspect": np.asarray(graph.aspect),
    }
    # Two distinct grids: the WATER class exchange (deposition - pickup, per class) and the ACTUAL
    # bed inventory change (final - initial voxel.sum(axis=2) + active, per class), which also holds
    # what avalanching inside commits and sub-resolution placement did to the bed.
    grids_out["water_exchange_net_kg"] = grids_out["cumulative_deposition_kg"] - grids_out["cumulative_pickup_kg"]
    grids_out["bed_change_kg"] = to_host(result.bed_change_by_cell_class_kg()).copy()
    if control.csr:
        # CSR storage: the stored CSR arrays themselves (plus the slot map), never a densified (ny, nx, nc, B) field.
        for name in ("row_ptr", "bin", "fraction", "position_m", "cell_of_slot"):
            grids_out[f"phase_csr_{name}"] = to_host(getattr(final.phase, name)).copy()
    elif final.phase is not None:
        grids_out["phase_fraction"] = to_host(final.phase.fraction).copy()
        grids_out["phase_position_m"] = to_host(final.phase.position_m).copy()
    for name, value in grids_out.items():
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise SedimentEventError(f"non-finite final grid {name}; no output written")
    if not np.all(np.isfinite(hydrograph)) or not np.all(np.isfinite(sediment_hydrograph)):
        raise SedimentEventError("non-finite hydrograph; no output written")
    for name in ("depth_m", "mobile_mass_kg", "soil_water_m", "discharge_m2_s", "active_mass_kg", "voxel_mass_kg"):
        if np.any(grids_out[name] < 0.0):
            raise SedimentEventError(f"negative final inventory in {name}; no output written")
    validate_water_state(host_bed.water, ny, nx, n_classes)
    phase_record = phase_summary(result)

    # --- water budget (as the Phase 4 runner) ----------------------------------------------------------
    n_steps, n_cells = result.n_accepted_steps, ny * nx
    export_depth = export_m3 / area
    inventory = sums["surface_initial"] + sums["soil_initial"] + sums["rain"]
    magnitude = inventory + sums["intake"] + sums["drainage"] + sums["saturation_return"] + export_depth
    tolerance = _STORM_RTOL * (n_steps + n_cells) * magnitude
    water_residual = (sums["surface_final"] + sums["soil_final"] + sums["drainage"] + export_depth) - inventory
    surface_residual = sums["surface_final"] - (sums["surface_initial"] + sums["rain"] - sums["intake"]
                                                + sums["saturation_return"] - export_depth)
    soil_residual = sums["soil_final"] - (sums["soil_initial"] + sums["intake"] - sums["drainage"]
                                          - sums["saturation_return"])
    rain_expected = schedule.depth_m(0.0, end) * float(host["rainfall_scale"].sum())
    rain_tolerance = 4.0 * _EPS * (n_steps + n_cells) * max(rain_expected, sums["rain"])
    for label, residual, tol in (("water", water_residual, tolerance), ("surface", surface_residual, tolerance),
                                 ("soil", soil_residual, tolerance),
                                 ("rainfall integral", sums["rain"] - rain_expected, rain_tolerance)):
        if not abs(residual) <= tol:
            raise SedimentEventError(f"{label} budget residual {residual} m exceeds tolerance {tol} m; "
                                     "no output written")
    rows = {name: hydrograph[:, i] for i, name in enumerate(HYDROGRAPH_COLUMNS)}
    row_residual = (rows["surface_storage_m3"] + rows["soil_storage_m3"] + rows["cumulative_drainage_m3"]
                    + rows["cumulative_export_m3"]) - (
        (sums["surface_initial"] + sums["soil_initial"]) * area + rows["cumulative_rain_m3"])
    if not np.all(np.abs(row_residual) <= tolerance * area):
        raise SedimentEventError("a hydrograph row violates the water balance; no output written")

    # --- sediment closure and refusal accounting ----------------------------------------------------------
    if not closure["closed"]:
        raise SedimentEventError(f"per-class sediment closure failed: residual {closure['residual_kg']} kg, "
                                 f"tolerance {closure['tolerance_kg']} kg; no output written")
    if not closure["request_reconciled"]:
        raise SedimentEventError("pickup request reconciliation failed; no output written")
    mres = context.mass_resolution_kg
    from maple.surface.voxels._numerics import summation_error_bound_kg

    # Match MAPLE water.validation's accumulated transfer-identity policy;
    # mass_resolution is physical significance, never a reconciliation floor.
    transfer_scale = max(float(np.max(np.abs(by_class[k]))) for k in
                         ("deposition_requested", "deposition_actual", "export_requested", "export_actual"))
    unmet_tolerance = np.full(n_classes, summation_error_bound_kg(
        4 * n_cells * max(result.n_maple_water_calls, 1), max(transfer_scale, 1.0)))
    if np.any(np.abs(by_class["deposition_unmet"]) > unmet_tolerance) or np.any(
            np.abs(by_class["export_unmet"]) > unmet_tolerance):
        raise SedimentEventError("MAPLE refused deposition or export beyond the sub-resolution allowance; "
                                 "no output written")
    if np.any(np.abs(by_class["transport_budget_residual"]) > by_class["transport_budget_tolerance"]):
        raise SedimentEventError("accumulated transport budget residual exceeds its declared bound; no output written")
    if control.force_final_commit and np.any(grids_out["pending_bed_mass_change_kg"] != 0.0):
        raise SedimentEventError("final terrain is not committed (pending ledger mass remains); no output written")

    # --- source stability ------------------------------------------------------------------------------
    xml_sha_after = _sha256_file(xml_path)
    if xml_sha_after != xml_sha_before:
        raise SedimentEventError("root MAHLERAN XML changed during the run; no output written")
    digests_after = _source_digests(Path(syrup_provenance["package_dir"]), dependency.package_dir)
    changed = [name for name in digests_before if digests_after[name] != digests_before[name]]
    if changed:
        raise SedimentEventError(f"source changed during the run ({', '.join(changed)}); no output written")

    class_ids = list(CLASS_IDS)
    by_class_json = {k: v.tolist() for k, v in by_class.items()}
    active_final = grids_out["active_mass_kg"]
    composition = {
        "initial_active_fraction_by_class": (initial_active.sum(axis=(0, 1)) / initial_active.sum()).tolist(),
        "final_active_fraction_by_class": (active_final.sum(axis=(0, 1)) / active_final.sum()).tolist(),
        "picked_up_fraction_by_class": (by_class["actual_pickup"] / by_class["actual_pickup"].sum()).tolist()
        if by_class["actual_pickup"].sum() > 0.0 else None,
        "exported_fraction_by_class": (by_class["export_actual"] / by_class["export_actual"].sum()).tolist()
        if by_class["export_actual"].sum() > 0.0 else None,
        "final_mobile_by_class_kg": closure["final_mobile_kg"],
        "available_mass_change_by_class_kg": (grids_out["available_mass_kg"].sum(axis=(0, 1))
                                             - initial_available.sum(axis=(0, 1))).tolist(),
    }
    dz = final_elevation - initial_elevation
    morphology = morphology_summary(grids_out["bed_change_kg"], grids_out["water_exchange_net_kg"],
                                    bulk_density_kg_m3=g.bulk_density_kg_m3, cell_area_m2=area)
    graph_summary = final.graph.summary()
    graph_summary.pop("level_widths", None)
    water_channel = water_channel_index()
    sediment_rows = {name: sediment_hydrograph[:, i] for i, name in enumerate(result.sediment_columns)}
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": (f"configured_end_reached at t = {end} s; NOT an event completion: residual surface water "
                   f"{sums['surface_final'] * area} m3, outlet discharge {rows['outlet_discharge_m3_s'][-1]} m3/s and "
                   f"wet mobile sediment {sum(closure['final_mobile_kg'])} kg are retained; no stop criterion, "
                   "no dry reset, no restart"),
        "case": {
            "case_dir": str(verified.case_dir),
            "maple_case_identity_sha256": verified.binding["maple_case_identity_sha256"],
            "maple_artifact_sha256": verified.binding["maple_artifact_sha256"],
            "verification": verified.checks,
        },
        "sources": {
            "rainfall": {"path": str(verified.rainfall_path), "sha256": schedule.provenance.sha256,
                         "n_records": schedule.provenance.n_records, "start_clock": schedule.provenance.start_clock,
                         "convention": schedule.provenance.convention},
            "mahleran_xml": {"path": str(xml_path), "sha256_before": xml_sha_before, "sha256_after": xml_sha_after,
                             "stable": True},
            "mahleran_git": verified.report["mahleran"]["git"],
            "syrup_report_sha256": verified.binding["syrup_report_sha256"],
            "syrup_fields_sha256": verified.binding["syrup_fields_sha256"],
        },
        "provenance": {
            "maple_syrup": syrup_provenance,
            "maple": verified.maple_provenance,
            "maple_matches_import_binding": not verified.checks["maple_source_changed_since_import"],
            "source_stability": {"before": digests_before, "after": digests_after, "stable": True,
                                 "rule": "package source digests and the root XML hash taken before verification "
                                         "and after the loop; any change aborts before outputs are written"},
            "environment": environment_record(),
            "numba": routing_numba.numba_versions(),
            "implementation": implementation,
        },
        "resolved_config": resolved_config,
        "resolved_config_sha256": resolved_config_sha256,
        "override": override,
        "domain": {"ny": ny, "nx": nx, "dx_m": g.dx_m, "cell_area_m2": area, "area_m2": area * n_cells,
                   "active_cells": graph.n_active, "initial_graph": graph.input_sha256, "final_graph": graph_summary,
                   "class_ids": class_ids, "n_voxels": int(grids_out["voxel_mass_kg"].shape[2]),
                   "bulk_density_kg_m3": g.bulk_density_kg_m3,
                   "boundaries": "legacy rmask < 0 ring cells export water and sediment (south edge outlets); "
                                 "fixed boundary ring at every terrain refresh; no other lateral exchange"},
        "time": {
            "start_s": 0.0, "end_s": end, "rainfall_end_s": schedule.end_s,
            "legacy_stormlength_s": float(settings["stormlength_s"]),
            "recession_included": end > schedule.end_s,
            "max_dt_s": max_dt, "min_dt_s": storm_control.min_dt_s, "n_accepted_steps": n_steps,
            "n_rejected_attempts": result.n_rejected_attempts, "rejections_recorded": list(result.rejections),
            "n_transport_substep_rejections": result.n_transport_rejections,
            "max_transport_substeps_used": result.max_transport_substeps_used,
            "dt_min_accepted_s": result.min_accepted_dt_s, "dt_max_accepted_s": result.max_accepted_dt_s,
            "n_boundaries": int(result.boundaries.size), "report_every_s": cadence,
            "n_maple_water_calls": result.n_maple_water_calls,
            "policy": "boundaries = rainfall knots in (0, end) + reporting times + end; dt <= max_dt_s; water "
                      "Courant/negative-RHS and sediment Courant (beyond max_transport_substeps) rejections halve "
                      "dt and recompute the whole unpublished step from the same state, refused below the retry "
                      "floor min_dt_s; no other exception is retried; failed attempts accumulate nothing",
        },
        "parameters": {"column": parameter_record, "sediment": sediment_record},
        "routing": {
            # The default bisection string is unchanged; a non-default root solver is named truthfully.
            "method": "MAHLERAN method 5 (Crank-Nicolson, bisection on [0, R], coherent old inflow)"
                      if storm_control.root_solver == "bisection" else
                      "MAHLERAN method 5 (Crank-Nicolson, safeguarded Newton on the same cell equation with the "
                      "[0, R] bracket, coherent old inflow)",
            "implementation": implementation, "root_solver": storm_control.root_solver,
            "courant_max": storm_control.courant_max,
            "max_courant_old": cr_old, "max_courant_new": cr_new,
            "max_cell_balance_residual_m": max_balance, "max_constitutive_residual_m": max_constitutive,
            "cell_steps": {"no_runon": cells_no, "partial_runon": cells_partial, "complete_runon": cells_complete},
            "terrain_refresh": "graph, transport network and physics grid rebuilt from the MAPLE committed surface "
                               "at every commit; discharge re-initialised from the new conveyance at unchanged depth",
        },
        "budget": {
            "units": "m3",
            "surface_initial_m3": sums["surface_initial"] * area, "soil_initial_m3": sums["soil_initial"] * area,
            "rain_m3": sums["rain"] * area, "rain_expected_m3": rain_expected * area,
            "intake_m3": sums["intake"] * area, "saturation_return_m3": sums["saturation_return"] * area,
            "drainage_m3": sums["drainage"] * area, "export_m3": export_m3,
            "surface_final_m3": sums["surface_final"] * area, "soil_final_m3": sums["soil_final"] * area,
            "water_residual_m3": water_residual * area, "surface_residual_m3": surface_residual * area,
            "soil_residual_m3": soil_residual * area,
            "rainfall_integral_residual_m3": (sums["rain"] - rain_expected) * area,
            "tolerance_m3": tolerance * area, "rainfall_tolerance_m3": rain_tolerance * area,
            "identity": "surface_final + soil_final + drainage + export = surface_initial + soil_initial + rain",
            "commit_water_volume": "constant_depth at every commit: displaced volume 0, depth array-equal (checked)",
        },
        "sediment": {
            "units": "kg",
            "class_ids": class_ids,
            "by_class": by_class_json,
            "totals": {k: float(v.sum()) for k, v in by_class.items()},
            "closure": closure,
            "unmet_tolerance_by_class_kg": unmet_tolerance.tolist(),
            "numerical_residual_scalar_abs_kg": residual_scalar,
            "max_transport_cell_balance_residual_kg": max_cell_residual,
            "max_sediment_courant": max_sed_courant, "max_decay_exponent_v_r_dt": max_decay,
            "regime_cell_class_steps": regime,
            "transport": phase_record,
            "peak_mobile_kg": peak_mobile, "time_of_peak_mobile_s": peak_mobile_t,
            "final_mobile_kg": sum(closure["final_mobile_kg"]),
            "peak_export_rate_kg_s": peak_export_rate, "time_of_peak_export_s": peak_export_t,
            "sampled_peak_export_rate_kg_s": float(sediment_rows["export_rate_kg_s"].max()),
            "export_hydrograph_note": "export_rate_* columns are the last accepted step's actual export / dt at "
                                      "each reporting row; peaks above are true per-step maxima",
            # morphology from ACTUAL inventories (classes summed per cell before the +/- split); the
            # water class exchange is reported separately inside `morphology`
            "net_erosion_kg": morphology["net_erosion_kg"],
            "net_deposition_kg": morphology["net_deposition_kg"],
            "morphology": morphology,
            "water_class_exchange_note": "cumulative_deposition - cumulative_pickup per class "
                                         "(final_state.npz water_exchange_net_kg) is the water exchange only; "
                                         "bed_change_kg is the actual bed inventory change including avalanching",
            "composition": composition,
            "committed_elevation_change_m": {"min": float(dz.min()), "max": float(dz.max()), "mean": float(dz.mean()),
                                             "n_cells_lowered": int(np.sum(dz < 0.0)),
                                             "n_cells_raised": int(np.sum(dz > 0.0))},
            "morphology_note": "pickup demand is solid mass at the particle density (2650 kg/m3); MAPLE converts "
                               "mass to elevation with the case bulk density (1250 kg/m3), i.e. 2.12 x the legacy "
                               "z_change for the same solid mass (documented departure)",
        },
        "commits": {
            "n_commits": result.n_commits, "n_forced_commits": result.n_forced_commits,
            "n_graph_changes": result.n_graph_changes, "rerouted_cells_total": result.rerouted_cells_total,
            "graph_change_meaning": "n_graph_changes counts commits whose refreshed graph changed slope or receiver "
                                    "(physical); graph_rebound in the log marks a differing input_sha256 binding "
                                    "only (the refresh omits the initial graph's nodata metadata)",
            "outlet_policy": "exporting boundary ring and active set fixed; interior outlet cells may be "
                             "reclassified by a valid reroute (outlets_changed per commit); pits and lost "
                             "receivers are refused atomically",
            "final_commit_count": int(bed_final.committed_topography.commit_count),
            "last_commit_time_s": float(bed_final.committed_topography.last_commit_time_s),
            "trigger_spec": dataclasses.asdict(context.topographic_commit_spec),
            "ledger_process_totals_reset_kg": ledger_reset.tolist(),
            "ledger_water_channel_index": water_channel,
            "ledger_water_channel_totals_kg": ledger_reset[water_channel].tolist(),
            "committed_net_bed_change_by_class_kg": committed_net.tolist(),
            "log": list(result.commit_log),
            "log_truncated": len(result.commit_log) < result.n_commits,
            "commit_wall_s": result.commit_wall_s,
        },
        "final_state": {
            "ponded_cells": int(ponded_cells), "max_depth_m": final_max_depth,
            "mean_depth_m": sums["surface_final"] / n_cells,
            "peak_depth_m": peak_depth, "peak_velocity_m_s": peak_velocity,
            "final_outlet_discharge_m3_s": float(rows["outlet_discharge_m3_s"][-1]),
            "peak_outlet_discharge_m3_s": true_peak_q, "time_of_peak_outlet_discharge_s": true_peak_t,
            "sampled_peak_outlet_discharge_m3_s": float(rows["outlet_discharge_m3_s"].max()),
            "soil_water_range_m": [soil_min, soil_max],
            "residual": "surface water, soil water and wet mobile sediment retained; configured end is not event end",
        },
        "maple_state": {
            "authority": "MAPLE voxel column, active layer, availability, ledger, committed topography and "
                         "WaterState (depth + mobile) via sediment_bed; no parallel bed or terrain",
            "snapshot": f"{SNAPSHOT_NAME}: MAPLE save_state_snapshot of the final bed, availability and water; a "
                        "MAPLE state record, not a SYRUP restart (hydraulic and transport memory are in "
                        f"{FINAL_NAME} for inspection only; no resumability claimed)",
        },
        "backend": {
            "backend": resolved.backend.value, "device_id": resolved.device_id,
            "fingerprint": json.loads(json.dumps(resolved.fingerprint, default=str)),
            "pairing": control.backend_record(),
            "loop_transfer_counters": transfers,
            "transfer_boundaries": (
                "commits transfer the committed elevation to the host and rebuild the graph there; numba "
                "hydraulics are host-only; the cupy backend is refused unless the implementation is the explicit "
                "'cuda' hydrology" if backend == "numpy" else
                "whole small bed, graph, column, phase and accounting resident in CuPy; the CUDA hydrology "
                "context is prepared once per graph binding and re-prepared for the refreshed graph of every "
                "commit (counted); each commit downloads the committed elevation once to rebuild the graph on the "
                "host; per attempted step: one 144-byte packet read plus MAPLE's and SYRUP's counted validation "
                "flag reads; numpy + cuda is refused (no CPU fallback)"),
            "gpu": "not exercised; no GPU claim" if backend == "numpy" else {
                "executed": True,
                "cuda_hydraulics": hydraulics.summary(),
                "memory_preflight": memory_preflight,
                "device_memory": device_memory,
                "claims": "small whole-bed CuPy event through the actual MAPLE bed API with the array wet laws and "
                          "characteristic transport on CURRENT holdings; no Chastre-scale bed, no GPU complete "
                          "event, no GPU checkpoint/restart, no speed claim",
            },
        },
        "timings": {
            "setup_and_verification": setup,
            "first_accepted_step": {"wall_s": result.first_step_wall_s, "cpu_s": result.first_step_cpu_s,
                                    "note": "includes lazy import/JIT compilation for the numba implementation; "
                                            "the CUDA context preparation and kernel load precede the first step "
                                            "and are reported under backend.gpu.cuda_hydraulics"},
            "remaining_steps": {"wall_s": result.remaining_wall_s, "cpu_s": result.remaining_cpu_s,
                                "n_steps": max(n_steps - 1, 0)},
            "commits_wall_s": result.commit_wall_s,
            "step_loop": loop,
            "note": "process CPU and wall seconds of one run; not a controlled benchmark" if backend == "numpy" else
                    "process CPU and wall seconds of one run; not a controlled benchmark. Device runs: setup and "
                    "step_loop are read after an explicit stream synchronization; the per-step and commit wall "
                    "times are host timers whose every accepted step ends in synchronizing host reads (the step "
                    "packet, MAPLE validation flags), so they are synchronized at step granularity, not per kernel",
        },
        "hydrograph": {"water_columns": list(HYDROGRAPH_COLUMNS), "sediment_columns": list(result.sediment_columns),
                       "n_rows": int(hydrograph.shape[0]), "files": [HYDROGRAPH_CSV, HYDROGRAPH_NPZ],
                       "sediment_prefix_in_files": "sed_"},
        "assumptions": [
            "wet MAHLERAN laws with the resolved legacy_literal conventions; splash deferred (dry cells: no pickup)",
            ("constant_depth ownership: water rides with the bed at commits; discharge, velocity and outlet "
             "discharge diagnostics re-initialised from the new conveyance at unchanged depth"),
            ("fixed exporting boundary ring and active set; interior outlets may be reclassified by a valid "
             "reroute; a commit that creates a pit or a lost receiver fails atomically (no infill)"),
            "soil-water parameter thickness fixed as legacy despite bed change; no pore water carried by sediment",
            "no evapotranspiration, dry reset, plants, nutrients, splash, wind or restart",
        ],
        "limitations": [
            "not a MAHLERAN Fortran storm benchmark; the equation-level references are in tests/phase5",
            "configured end is not event completion; residual water and mobile sediment are reported, not resolved",
            ("no GPU result; the commit path rebuilds the graph on the host and the cupy backend is refused unless "
             "the explicit 'cuda' hydrology is selected" if backend == "numpy" else
             "small whole-bed device event only: the graph is still rebuilt on the host at every commit, the "
             "MAPLE exchange copies the whole voxel array per call, no memory-bounded bed, no Chastre-scale "
             "support, no GPU complete event or restart, no speed claim"),
            "MAPLE snapshot is not a SYRUP restart; no resumability",
        ],
    }

    # --- outputs: assembled in a temporary sibling, renamed into place ---------------------------------
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.partial-{os.getpid()}"
    _ensure_absent(staging)
    staging.mkdir()
    try:
        snapshot_path = save_state_snapshot(
            host_bed.voxel_column, host_bed.active_layer, end, staging / SNAPSHOT_NAME,
            run_name=case.case_name, seed=int(case.config.seed), step=n_steps,
            reason="maple_syrup phase5b configured end; MAPLE state record, not a SYRUP restart",
            geometry=g, mass_resolution_kg=mres,
            provenance={"schema": SUMMARY_SCHEMA, "resolved_config_sha256": resolved_config_sha256,
                        "maple_case_identity_sha256": verified.binding["maple_case_identity_sha256"],
                        "syrup_restart": False},
            sediment_availability=host_bed.sediment_availability, water=host_bed.water,
        )
        if Path(snapshot_path).name != SNAPSHOT_NAME:
            raise SedimentEventError("MAPLE snapshot path differs from the requested name")
        with (staging / FINAL_NAME).open("xb") as handle:
            np.savez(handle, **grids_out)
        interval = {name: np.diff(rows[name], prepend=0.0) for name in (
            "cumulative_rain_m3", "cumulative_intake_m3", "cumulative_saturation_return_m3",
            "cumulative_drainage_m3", "cumulative_export_m3")}
        sed_out = {f"sed_{k}": v for k, v in sediment_rows.items()}
        with (staging / HYDROGRAPH_NPZ).open("xb") as handle:
            np.savez(handle, **rows, **{f"interval_{k[len('cumulative_'):]}": v for k, v in interval.items()},
                     row_water_residual_m3=row_residual, **sed_out)
        with (staging / HYDROGRAPH_CSV).open("x", newline="", encoding="ascii") as handle:
            writer = csv.writer(handle)
            header = list(HYDROGRAPH_COLUMNS) + [f"interval_{k[len('cumulative_'):]}" for k in interval] + [
                "row_water_residual_m3"] + list(sed_out)
            writer.writerow(header)
            for i in range(hydrograph.shape[0]):
                writer.writerow([repr(float(v)) for v in hydrograph[i]]
                                + [repr(float(interval[k][i])) for k in interval] + [repr(float(row_residual[i]))]
                                + [repr(float(sed_out[k][i])) for k in sed_out])
        summary["outputs"] = {name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                              for name in (SNAPSHOT_NAME, FINAL_NAME, HYDROGRAPH_NPZ, HYDROGRAPH_CSV)}
        summary["timings"]["reporting"] = clock.lap()
        _write_new_json(staging / SUMMARY_NAME, summary)
        _ensure_absent(output_dir)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return SedimentRun(summary=summary, verified=verified, state=final, result=result, hydrograph=hydrograph,
                      sediment_hydrograph=sediment_hydrograph)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m maple_syrup.sediment_experiment",
        description="MAPLE-SYRUP Phase 5b: Plot 1 coupled water + wet-sediment event on the actual MAPLE bed "
                    "(no splash, no stop criterion, no reset, no restart).",
    )
    parser.add_argument("--case-dir", required=True, help="Verified Phase 2 case, e.g. outputs/plot1")
    parser.add_argument("--output-dir", required=True, help="NEW directory for summary, grids, hydrograph, snapshot.")
    parser.add_argument("--max-dt-s", type=float, default=1.0, help="Largest step (s); knots always split.")
    parser.add_argument("--end-s", type=float, default=None,
                        help="Simulated end (s); default the legacy stormlength (5400 s for Plot 1).")
    parser.add_argument("--implementation", default="numba", choices=STORM_IMPLEMENTATIONS,
                        help="Hydraulic sweep implementation (default numba; missing Numba is an error, no fallback; "
                             "'cuda' is the explicit prepared CUDA hydrology and requires --backend cupy).")
    parser.add_argument("--backend", default="numpy", choices=tuple(BACKEND_PAIRINGS),
                        help="numpy (array/numba hydraulics) or cupy (ONLY with --implementation cuda: whole small "
                             "bed on the device, memory preflight first); other pairings are refused, no fallback. "
                             "Select the device with CUDA_VISIBLE_DEVICES before launching.")
    parser.add_argument("--root-solver", default="bisection", choices=("bisection", "newton"),
                        help="Cell-equation root solver of the hydraulic sweep (default bisection).")
    parser.add_argument("--cuda-mode", default="auto", choices=CUDA_MODES,
                        help="CUDA hydrology launch structure (auto/fused/split; cupy backend only).")
    parser.add_argument("--report-every-s", type=float, default=60.0, help="Hydrograph row cadence (s).")
    parser.add_argument("--min-dt-s", type=float, default=1.0 / 1024.0)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10_000_000)
    parser.add_argument("--sediment-courant-max", type=float, default=CHARACTERISTIC_COURANT_MAX,
                        help="Sediment Courant cap per substep (<= 0.5 for characteristic, <= 1 for upwind).")
    parser.add_argument("--max-transport-substeps", type=int, default=64)
    parser.add_argument("--transport-scheme", default="characteristic", choices=TRANSPORT_SCHEMES,
                        help="Lateral transport: characteristic (Phase 7b phase bins, default) or the Phase 5 upwind "
                             "operator kept as an explicit comparison.")
    parser.add_argument("--phase-bins", type=int, default=DEFAULT_N_BINS,
                        help=f"Position bins per cell/class for the characteristic scheme (1..{MAX_N_BINS}; "
                             f"{DEFAULT_N_BINS} is a candidate default, not an accepted convergence).")
    parser.add_argument("--transport-implementation", default="auto", choices=TRANSPORT_IMPLEMENTATIONS,
                        help="Characteristic kernel: auto (follow --implementation), array or numba (dense storage); "
                             "numba or cuda (csr storage); no fallback.")
    parser.add_argument("--phase-storage", default="dense", choices=PHASE_STORAGES,
                        help="Phase storage of the characteristic scheme: dense (default, the accepted (ny,nx,nc,B) "
                             "arrays) or csr (the accepted sparse nonempty-bin state; same physics and bins).")
    parser.add_argument("--phase-max-nnz", type=int, default=None,
                        help="csr only: predeclared capacity (stored bins) refused BEFORE allocation when exceeded; "
                             "never truncates mass. Required by the CUDA operator for large cases.")
    parser.add_argument("--no-commit", action="store_true", help="Do not evaluate MAPLE commit triggers per step.")
    parser.add_argument("--no-final-commit", action="store_true", help="Do not force a final commit at end_s.")
    parser.add_argument("--mahleran-root", help="Override the recipe's MAHLERAN root (read-only).")
    parser.add_argument("--expected-maple-root", help="Refuse any other MAPLE source root.")
    parser.add_argument("--allow-maple-source-change", action="store_true",
                        help="Run even if MAPLE's source digest differs from the import's (recorded).")
    args = parser.parse_args(argv)
    try:
        run = run_plot1_sediment_event(
            args.case_dir, args.output_dir, max_dt_s=args.max_dt_s, end_s=args.end_s,
            implementation=args.implementation, backend=args.backend, report_every_s=args.report_every_s,
            min_dt_s=args.min_dt_s, max_retries=args.max_retries, max_steps=args.max_steps,
            sediment_courant_max=args.sediment_courant_max, max_transport_substeps=args.max_transport_substeps,
            commit=not args.no_commit, force_final_commit=not args.no_final_commit,
            transport_scheme=args.transport_scheme, phase_bins=args.phase_bins,
            transport_implementation=args.transport_implementation,
            mahleran_root=args.mahleran_root, expected_maple_root=args.expected_maple_root,
            allow_maple_source_change=args.allow_maple_source_change,
            root_solver=args.root_solver, cuda_mode=args.cuda_mode,
            phase_storage=args.phase_storage, phase_max_nnz=args.phase_max_nnz,
        )
    except MapleDependencyError as exc:
        print(f"MAPLE dependency check failed: {exc}", file=sys.stderr)
        return 2
    except (Plot1ImportError, SedimentEventError, StormError, BedIntegrationError, SedimentPhysicsError,
            TransportError, RoutingError, RoutingGraphError, InfiltrationError, RainfallError, ValueError) as exc:
        print(f"plot1 sediment event failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    keys = ("status", "time", "budget", "final_state", "commits", "timings")
    printed = {k: run.summary[k] for k in keys}
    printed["sediment"] = {k: run.summary["sediment"][k] for k in (
        "totals", "closure", "peak_mobile_kg", "final_mobile_kg", "peak_export_rate_kg_s", "net_erosion_kg",
        "net_deposition_kg", "regime_cell_class_steps")}
    print(json.dumps(_json_safe(printed), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
