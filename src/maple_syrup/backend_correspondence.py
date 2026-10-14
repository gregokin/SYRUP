"""Operand-qualified CPU/GPU correspondence of saved physical outputs (CW-D1).

What this is
------------
A host comparator for two backends' SAVED outputs of the same event. It
answers one question per cell and grain class: "could these two physical
reservoirs differ by this much purely because two correct FP64 histories of
the same exchanges rounded differently?" The allowance is MAPLE's own
accumulated-roundoff rule, the one `commit_topography` applies when it
reconciles the ledger with the physical bed
(`maple.surface.topographic_commit.commit._reconcile_ledger_with_physical_state`):

    per side:   min(k * max(N_touches, 1) * eps * max(gross_activity, |reference|), mass_resolution)
    paired:     min(bound_cpu + bound_gpu, mass_resolution)

with `k` MAPLE's per-call epsilon multiplier read from that function's own
signature (not copied), `N_touches` the ACTUAL accumulated count of physical
touches at that cell/class, `gross_activity` the accumulated gross physical
activity there, `reference` the inventory at the reconciliation reference and
`mass_resolution` the case's `numerics.mass_resolution_kg`. The paired bound
is capped at ONE mass resolution, stricter than the sum of two per-side caps:
a difference the model itself would call resolved (`> mass_resolution_kg`,
`maple.surface.voxels._numerics.classify_mass_resolution_difference`) can
never pass, however many touches occurred.

What this is not
----------------
* Not conservation. Each backend's water and class-sediment closures at
  MAPLE's summation bound (`maple_syrup.conservation`) remain separate and
  are not evaluated here.
* Not the declared strict field gate of the matched comparisons (rtol/atol).
  That gate's recorded result stands; this module neither reinterprets nor
  replaces it.
* Not a hydrology comparator: water depth, discharge, velocity and volumes
  are not masses and are refused by the reservoir comparison; only exact
  integer/routing controls are checked here, bytewise.
* Not a storm pass by itself. The report distinguishes three things: the
  PHYSICAL reservoir correspondence (the only part that can be called
  "correspondence"), exact controls, and derived diagnostics (consistency
  reports). It names exactly which fields were examined; a field that was
  not supplied is reported as not examined, never as passed, and a report
  with no physical reservoir cannot claim physical correspondence.
* The evidence constructor does not prove that evidence is genuine. The
  per-cell/class touches and gross activity accumulated through EVERY water
  call and EVERY commit (including forced final and avalanche commits) are
  not yet produced by the event runner; MAPLE's commit result does not
  expose the ledger's pre-reset touches and activity, so B3b must add them.
  The caller binds the arrays to their origin through `binding` (source
  pins, receipt and row hashes), which the report echoes verbatim; until
  such evidence exists a caller has none and this module refuses rather
  than infers.

Rules of evidence
-----------------
The comparator never infers touches from step counts or domain size and never
treats missing history as zero. Evidence must be explicit host NumPy arrays
(no implicit device transfer), FP64 finite and nonnegative for masses,
integer and nonnegative for touches, with a positive finite mass resolution
identical on both sides, and `history_complete=True` set by the integration
that accumulated it. Anything else is a refusal (`CorrespondenceEvidenceError`),
distinct from a comparison failure (a result).

Derived net diagnostics (for example the per-cell/class bed change
`(voxel.sum(axis=2) + active) - initial`) are never compared as saved backend
outputs with a bound scaled by their near-zero net. They are recomputed on
the host from each backend's saved operands with one fixed expression and
compared with a bound built from the actual operation count and operand
scale through MAPLE's `summation_error_bound_kg`, plus the measured
differences of the operands themselves (propagation), per cell and class.
They are consistency reports and never gate the physical correspondence.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "PAIRED_RULE",
    "PER_SIDE_RULE",
    "BackendEvidence",
    "CorrespondenceEvidenceError",
    "FieldResult",
    "InventoryChangeOperands",
    "compare_derived_inventory_change",
    "compare_exact_control",
    "compare_reservoir",
    "correspondence_report",
    "host_inventory_change_kg",
    "maple_per_call_eps_multiplier",
    "paired_bound_kg",
    "per_side_roundoff_bound_kg",
    "validate_evidence",
]

EPS = float(np.finfo(np.float64).eps)
PER_SIDE_RULE = ("min(k * max(n_physical_touches, 1) * eps * max(gross_physical_activity_kg, "
                 "|reference_inventory_kg|), mass_resolution_kg); k = MAPLE per_call_eps_multiplier")
PAIRED_RULE = "min(bound_cpu + bound_gpu, mass_resolution_kg)  (ONE resolution cap, stricter than two)"
MAX_RECORDED_FAILURES = 32


class CorrespondenceEvidenceError(ValueError):
    """Malformed, missing or mixed evidence: the comparison is refused, not failed."""


def maple_per_call_eps_multiplier() -> float:
    """MAPLE's per-call epsilon multiplier, read from the reconciliation rule's own default."""
    from maple.surface.topographic_commit.commit import (
        _reconcile_ledger_with_physical_state,
    )

    parameter = inspect.signature(_reconcile_ledger_with_physical_state).parameters["per_call_eps_multiplier"]
    value = float(parameter.default)
    if not np.isfinite(value) or value <= 0.0:
        raise CorrespondenceEvidenceError("MAPLE per_call_eps_multiplier is not a positive finite number")
    return value


# --- evidence ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BackendEvidence:
    """One backend's accumulated per-cell/class exchange evidence, all `(ny, nx, nc)` host arrays.

    `reference_inventory_kg`: inventory at the reconciliation reference (the initial bed of the event).
    `gross_physical_activity_kg`: accumulated gross physical activity (pickup + deposition + burial/refill
    moved through MAPLE), summed over processes, through every water call and commit of the event.
    `n_physical_touches`: accumulated count of physical touches (MAPLE's `n_physical_touches`, captured before
    every commit reset and summed), integer. `history_complete` must be True only when the integration
    covered every water call and every commit, including the forced final commit and avalanche commits.
    `binding`: caller-recorded provenance (for example source pins, receipt and per-array sha256 of the saved
    rows) that ties these arrays to a run; echoed by the report, never checked for truth here. Constructing
    this object proves nothing about the history; the binding is what a reviewer verifies against receipts.
    """

    label: str
    reference_inventory_kg: Any
    gross_physical_activity_kg: Any
    n_physical_touches: Any
    mass_resolution_kg: float
    history_complete: bool
    touches_source: str
    activity_source: str
    binding: Mapping[str, str] | None = None


def _host_array(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise CorrespondenceEvidenceError(f"{name} is missing; it must be supplied explicitly, never assumed")
    if not isinstance(value, np.ndarray):
        raise CorrespondenceEvidenceError(f"{name} must be a host numpy.ndarray (no implicit device transfer), "
                                          f"got {type(value).__name__}")
    return value


def _mass_array(value: Any, name: str, ndim: tuple[int, ...]) -> np.ndarray:
    array = _host_array(value, name)
    if array.dtype != np.float64:
        raise CorrespondenceEvidenceError(f"{name} must be float64, got {array.dtype}")
    if array.ndim not in ndim:
        raise CorrespondenceEvidenceError(f"{name} must have rank {ndim}, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise CorrespondenceEvidenceError(f"{name} contains non-finite values")
    if (array < 0.0).any():
        raise CorrespondenceEvidenceError(f"{name} contains negative mass")
    return array


def _validate_binding(binding: Any, label: str) -> dict[str, str]:
    if binding is None:
        return {}
    if not isinstance(binding, Mapping) or not binding:
        raise CorrespondenceEvidenceError(f"{label}: binding must be a non-empty mapping of names to strings or None")
    out: dict[str, str] = {}
    for key, value in binding.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip():
            raise CorrespondenceEvidenceError(f"{label}: binding entries must be non-empty strings")
        out[key] = value
    return out


def validate_evidence(evidence: BackendEvidence) -> None:
    """Refuse malformed or incomplete evidence before any arithmetic."""
    if not isinstance(evidence, BackendEvidence):
        raise CorrespondenceEvidenceError("evidence must be a BackendEvidence")
    if not isinstance(evidence.label, str) or not evidence.label:
        raise CorrespondenceEvidenceError("evidence label must be a non-empty string")
    for name in ("touches_source", "activity_source"):
        if not isinstance(getattr(evidence, name), str) or not getattr(evidence, name).strip():
            raise CorrespondenceEvidenceError(f"{evidence.label}: {name} must name the actual accumulation source")
    if evidence.history_complete is not True:
        raise CorrespondenceEvidenceError(f"{evidence.label}: history_complete must be True; partial or inferred "
                                          "touch/activity history is refused, never treated as zero")
    reference = _mass_array(evidence.reference_inventory_kg, f"{evidence.label}.reference_inventory_kg", (3,))
    gross = _mass_array(evidence.gross_physical_activity_kg, f"{evidence.label}.gross_physical_activity_kg", (3,))
    touches = _host_array(evidence.n_physical_touches, f"{evidence.label}.n_physical_touches")
    if touches.dtype.kind not in "iu":
        raise CorrespondenceEvidenceError(f"{evidence.label}.n_physical_touches must be an integer array, got "
                                          f"{touches.dtype}")
    if touches.shape != reference.shape or gross.shape != reference.shape:
        raise CorrespondenceEvidenceError(f"{evidence.label}: reference {reference.shape}, gross {gross.shape} and "
                                          f"touches {touches.shape} must share one (ny, nx, nc) shape")
    if (touches < 0).any():
        raise CorrespondenceEvidenceError(f"{evidence.label}.n_physical_touches contains negative counts")
    resolution = evidence.mass_resolution_kg
    if (isinstance(resolution, bool) or not isinstance(resolution, (int, float, np.floating))
            or not np.isfinite(resolution) or float(resolution) <= 0.0):
        raise CorrespondenceEvidenceError(f"{evidence.label}.mass_resolution_kg must be a positive finite number")
    _validate_binding(evidence.binding, evidence.label)


def per_side_roundoff_bound_kg(evidence: BackendEvidence) -> np.ndarray:
    """MAPLE's accumulated-roundoff allowance for ONE backend, per cell and class, `(ny, nx, nc)`."""
    validate_evidence(evidence)
    k = maple_per_call_eps_multiplier()
    reference = np.abs(evidence.reference_inventory_kg)
    local_scale = np.maximum(evidence.gross_physical_activity_kg, reference)
    touches = np.maximum(evidence.n_physical_touches.astype(np.float64), 1.0)
    fp64 = k * touches * EPS * local_scale
    return np.minimum(fp64, float(evidence.mass_resolution_kg))


def paired_bound_kg(cpu: BackendEvidence, gpu: BackendEvidence) -> np.ndarray:
    """Sum of both sides' allowances, capped at ONE mass resolution (documented stricter cap)."""
    bound_cpu = per_side_roundoff_bound_kg(cpu)
    bound_gpu = per_side_roundoff_bound_kg(gpu)
    if bound_cpu.shape != bound_gpu.shape:
        raise CorrespondenceEvidenceError(f"mixed evidence: {cpu.label} {bound_cpu.shape} vs {gpu.label} "
                                          f"{bound_gpu.shape}")
    if float(cpu.mass_resolution_kg) != float(gpu.mass_resolution_kg):
        raise CorrespondenceEvidenceError("mixed evidence: the two sides declare different mass_resolution_kg")
    return np.minimum(bound_cpu + bound_gpu, float(cpu.mass_resolution_kg))


# --- results ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldResult:
    name: str
    kind: str
    passed: bool
    n_compared: int
    n_failed: int
    max_abs_difference: float
    max_bound: float
    min_bound: float
    worst_index: tuple[int, ...] | None
    worst_values: tuple[float, float] | None
    worst_bound: float | None
    failures: tuple[dict[str, Any], ...]
    rule: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "passed": self.passed, "n_compared": self.n_compared,
                "n_failed": self.n_failed, "max_abs_difference": self.max_abs_difference,
                "max_bound": self.max_bound, "min_bound": self.min_bound,
                "worst_index": None if self.worst_index is None else list(self.worst_index),
                "worst_values": None if self.worst_values is None else list(self.worst_values),
                "worst_bound": self.worst_bound, "failures": list(self.failures), "rule": self.rule}


def _bounded_result(name: str, kind: str, a: np.ndarray, b: np.ndarray, bound: np.ndarray, rule: str) -> FieldResult:
    diff = np.abs(a - b)
    failed = diff > bound
    n_failed = int(failed.sum())
    if diff.size:
        excess = np.where(failed, diff - bound, -np.inf)
        worst_flat = int(np.argmax(excess)) if n_failed else int(np.argmax(diff))
        worst = tuple(int(i) for i in np.unravel_index(worst_flat, diff.shape))
        worst_values = (float(a[worst]), float(b[worst]))
        worst_bound = float(np.broadcast_to(bound, diff.shape)[worst])
    else:
        worst, worst_values, worst_bound = None, None, None
    failures = []
    if n_failed:
        bound_full = np.broadcast_to(bound, diff.shape)
        for flat in np.flatnonzero(failed)[:MAX_RECORDED_FAILURES]:
            index = tuple(int(i) for i in np.unravel_index(int(flat), diff.shape))
            failures.append({"index": list(index), "a": float(a[index]), "b": float(b[index]),
                             "difference": float(diff[index]), "bound": float(bound_full[index])})
    return FieldResult(name=name, kind=kind, passed=n_failed == 0, n_compared=int(diff.size), n_failed=n_failed,
                       max_abs_difference=float(diff.max(initial=0.0)),
                       max_bound=float(np.max(bound, initial=0.0)), min_bound=float(np.min(bound, initial=0.0)),
                       worst_index=worst, worst_values=worst_values, worst_bound=worst_bound,
                       failures=tuple(failures), rule=rule)


def compare_reservoir(name: str, a: Any, b: Any, paired_bound: np.ndarray) -> FieldResult:
    """One physical reservoir, `(ny, nx, nc)` or voxel `(ny, nx, nv, nc)`, against the per-cell/class paired
    bound (broadcast over the voxel axis). Masses only: float64, finite, nonnegative on both sides."""
    first = _mass_array(a, f"{name}[a]", (3, 4))
    second = _mass_array(b, f"{name}[b]", (3, 4))
    bound = _host_array(paired_bound, "paired_bound")
    if first.shape != second.shape:
        raise CorrespondenceEvidenceError(f"{name}: shapes differ, {first.shape} vs {second.shape}")
    if bound.ndim != 3 or bound.shape[:2] != first.shape[:2] or bound.shape[-1] != first.shape[-1]:
        raise CorrespondenceEvidenceError(f"{name}: paired bound {bound.shape} does not match the reservoir's "
                                          f"(ny, nx, ..., nc) layout {first.shape}")
    if not np.isfinite(bound).all() or (bound < 0.0).any():
        raise CorrespondenceEvidenceError("paired bound must be finite and nonnegative")
    cast = bound if first.ndim == 3 else bound[:, :, None, :]
    return _bounded_result(name, "reservoir", first, second, np.broadcast_to(cast, first.shape), PAIRED_RULE)


def compare_exact_control(name: str, a: Any, b: Any) -> FieldResult:
    """Integer, boolean or routing controls (aspect, receiver, outlet, commit/step counts): bytewise identity.
    Floating hydrologic quantities are not accepted here; they are not masses and have their own comparator."""
    first, second = np.asarray(a), np.asarray(b)
    if first.dtype.kind not in "iub" or second.dtype.kind not in "iub":
        raise CorrespondenceEvidenceError(f"{name}: exact controls must be integer or boolean, got {first.dtype} "
                                          f"and {second.dtype}")
    same = first.shape == second.shape and first.dtype == second.dtype and first.tobytes() == second.tobytes()
    if same:
        n_failed = 0
    elif first.shape == second.shape:
        n_failed = int((first != second).sum())
    else:
        n_failed = int(max(first.size, second.size))
    worst = None
    failures: list[dict[str, Any]] = []
    if not same and first.shape == second.shape:
        for flat in np.flatnonzero(first != second)[:MAX_RECORDED_FAILURES]:
            index = () if first.ndim == 0 else tuple(int(i) for i in np.unravel_index(int(flat), first.shape))
            failures.append({"index": list(index), "a": int(first[index]), "b": int(second[index])})
        if failures:
            worst = tuple(failures[0]["index"])
    return FieldResult(name=name, kind="exact_control", passed=same, n_compared=int(first.size), n_failed=n_failed,
                       max_abs_difference=0.0 if same else float("nan"), max_bound=0.0, min_bound=0.0,
                       worst_index=worst, worst_values=None, worst_bound=0.0, failures=tuple(failures),
                       rule="bytewise identity (shape, dtype, bytes)")


# --- derived inventory change ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class InventoryChangeOperands:
    """One backend's saved operands of the per-cell/class bed change: voxel column `(ny, nx, nv, nc)`, active
    layer `(ny, nx, nc)`, initial inventory `(ny, nx, nc)`, and optionally the backend's own saved derived
    change `(ny, nx, nc)` for a consistency report."""

    voxel_kg: Any
    active_kg: Any
    initial_kg: Any
    saved_change_kg: Any = None


def host_inventory_change_kg(operands: InventoryChangeOperands) -> np.ndarray:
    """ONE fixed host expression: `(voxel.sum(axis=2) + active) - initial`."""
    voxel = _mass_array(operands.voxel_kg, "voxel_kg", (4,))
    active = _mass_array(operands.active_kg, "active_kg", (3,))
    initial = _mass_array(operands.initial_kg, "initial_kg", (3,))
    if active.shape != initial.shape or voxel.shape[:2] != active.shape[:2] or voxel.shape[-1] != active.shape[-1]:
        raise CorrespondenceEvidenceError("inventory-change operands do not share one (ny, nx, [nv,] nc) layout")
    return (voxel.sum(axis=2) + active) - initial


def _recompute_roundoff_bound_kg(operands: InventoryChangeOperands, change: np.ndarray) -> np.ndarray:
    """MAPLE summation bound for the host recompute: `nv + 2` roundings at the largest operand the arithmetic
    touches at that cell/class (column total, active, initial, result)."""
    from maple.surface.voxels._numerics import summation_error_bound_kg

    voxel = operands.voxel_kg
    n_ops = int(voxel.shape[2]) + 2
    scale = np.maximum.reduce([np.abs(voxel).sum(axis=2), np.abs(operands.active_kg), np.abs(operands.initial_kg),
                               np.abs(change)])
    return summation_error_bound_kg(n_ops, scale)


def compare_derived_inventory_change(name: str, cpu: InventoryChangeOperands,
                                     gpu: InventoryChangeOperands) -> dict[str, Any]:
    """Host-recomputed bed change of both backends with a propagated, operand-scaled bound.

    bound = roundoff(recompute_cpu) + roundoff(recompute_gpu) + sum_v |dvoxel| + |dactive| + |dinitial|,
    all per cell and class. The last three terms are the MEASURED operand differences (each operand is judged
    separately by the reservoir comparison); they are propagated, never scaled by the net result. A saved derived
    array, if supplied, is reported against its own side's host recompute within that side's roundoff bound.
    The returned `consistent` flag is a diagnostic consistency verdict, not a physical correspondence."""
    change_cpu = host_inventory_change_kg(cpu)
    change_gpu = host_inventory_change_kg(gpu)
    if change_cpu.shape != change_gpu.shape or cpu.voxel_kg.shape != gpu.voxel_kg.shape:
        raise CorrespondenceEvidenceError(f"{name}: mixed operand layouts between backends")
    roundoff_cpu = _recompute_roundoff_bound_kg(cpu, change_cpu)
    roundoff_gpu = _recompute_roundoff_bound_kg(gpu, change_gpu)
    propagated = (np.abs(cpu.voxel_kg - gpu.voxel_kg).sum(axis=2) + np.abs(cpu.active_kg - gpu.active_kg)
                  + np.abs(cpu.initial_kg - gpu.initial_kg))
    bound = roundoff_cpu + roundoff_gpu + propagated
    result = _bounded_result(name, "derived_inventory_change", change_cpu, change_gpu, bound,
                             "roundoff(cpu) + roundoff(gpu) + propagated operand differences; "
                             "roundoff = MAPLE summation_error_bound_kg(nv + 2, largest operand)")
    report: dict[str, Any] = {"comparison": result.as_dict(),
                              "max_propagated_operand_difference_kg": float(propagated.max(initial=0.0)),
                              "max_recompute_roundoff_bound_kg": float(max(roundoff_cpu.max(initial=0.0),
                                                                            roundoff_gpu.max(initial=0.0)))}
    saved_ok = True
    for label, operands, change, roundoff in (("cpu", cpu, change_cpu, roundoff_cpu),
                                                ("gpu", gpu, change_gpu, roundoff_gpu)):
        if operands.saved_change_kg is None:
            report[f"saved_{label}"] = None
            continue
        saved = _host_array(operands.saved_change_kg, f"{name}.saved_change_kg[{label}]")
        if saved.shape != change.shape or saved.dtype != np.float64 or not np.isfinite(saved).all():
            raise CorrespondenceEvidenceError(f"{name}: saved derived change of {label} is malformed")
        deviation = np.abs(saved - change)
        within = bool((deviation <= roundoff).all())
        saved_ok = saved_ok and within
        report[f"saved_{label}"] = {"max_deviation_from_host_recompute_kg": float(deviation.max(initial=0.0)),
                                   "within_own_roundoff_bound": within,
                                   "bitwise_equal_to_host_recompute": saved.tobytes() == change.tobytes()}
    report["consistent"] = bool(result.passed and saved_ok)
    return report


# --- report ---------------------------------------------------------------------------------------------------------
def _evidence_summary(evidence: BackendEvidence, bound: np.ndarray) -> dict[str, Any]:
    touches = evidence.n_physical_touches
    resolution = float(evidence.mass_resolution_kg)
    return {"label": evidence.label, "shape": list(bound.shape), "mass_resolution_kg": resolution,
            "touches_source": evidence.touches_source, "activity_source": evidence.activity_source,
            "history_complete": True, "binding": _validate_binding(evidence.binding, evidence.label),
            "touches_min": int(touches.min(initial=0)), "touches_max": int(touches.max(initial=0)),
            "cells_with_zero_touches": int((touches == 0).sum()),
            "gross_activity_max_kg": float(evidence.gross_physical_activity_kg.max(initial=0.0)),
            "reference_inventory_max_kg": float(evidence.reference_inventory_kg.max(initial=0.0)),
            "per_side_bound_min_kg": float(bound.min(initial=0.0)),
            "per_side_bound_max_kg": float(bound.max(initial=0.0)),
            "per_side_bound_capped_at_resolution": int((bound >= resolution).sum())}


def _names(mapping: Mapping[str, Any] | None) -> list[str]:
    return [] if not mapping else [str(name) for name in mapping]


def correspondence_report(
    *,
    cpu: BackendEvidence,
    gpu: BackendEvidence,
    reservoirs: Mapping[str, tuple[Any, Any]] | None = None,
    exact_controls: Mapping[str, tuple[Any, Any]] | None = None,
    derived_inventory_changes: Mapping[str, tuple[InventoryChangeOperands, InventoryChangeOperands]] | None = None,
    expected_reservoirs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Structured correspondence of saved outputs, in three separated sections.

    `physical_reservoirs`: the operand-qualified comparison; `physical_correspondence_passed` is True only when at
    least one physical reservoir was examined, every examined reservoir passed, every supplied exact control
    passed, and (when `expected_reservoirs` is given) no expected reservoir is missing. `exact_controls` and
    `derived_diagnostics` are reported on their own; derived diagnostics never make the physical verdict. Each
    section lists exactly the fields examined; unexamined fields are named as such and never counted as passed.
    Refuses (raises) on malformed or missing evidence, or when nothing at all is supplied to examine."""
    reservoirs = reservoirs or {}
    exact_controls = exact_controls or {}
    derived_inventory_changes = derived_inventory_changes or {}
    if not reservoirs and not exact_controls and not derived_inventory_changes:
        raise CorrespondenceEvidenceError("nothing to examine: supply reservoirs, exact controls or derived operands")
    bound = paired_bound_kg(cpu, gpu)
    bound_cpu = per_side_roundoff_bound_kg(cpu)
    bound_gpu = per_side_roundoff_bound_kg(gpu)
    reservoir_results = {name: compare_reservoir(name, a, b, bound).as_dict() for name, (a, b) in reservoirs.items()}
    control_results = {name: compare_exact_control(name, a, b).as_dict() for name, (a, b) in exact_controls.items()}
    derived_results = {name: compare_derived_inventory_change(name, a, b)
                       for name, (a, b) in derived_inventory_changes.items()}
    expected = [] if expected_reservoirs is None else [str(name) for name in expected_reservoirs]
    missing = [name for name in expected if name not in reservoir_results]
    reservoirs_examined = _names(reservoirs)
    all_reservoirs_pass = bool(reservoir_results) and all(r["passed"] for r in reservoir_results.values())
    all_controls_pass = all(c["passed"] for c in control_results.values())
    all_derived_consistent = all(d["consistent"] for d in derived_results.values())
    physical_passed = all_reservoirs_pass and all_controls_pass and not missing
    reasons: list[str] = []
    if not reservoir_results:
        reasons.append("no physical reservoir examined: no physical correspondence can be claimed")
    if missing:
        reasons.append(f"expected reservoirs not supplied: {missing}")
    if reservoir_results and not all_reservoirs_pass:
        reasons.append("at least one examined physical reservoir exceeds the operand-qualified bound")
    if not all_controls_pass:
        reasons.append("at least one exact control differs")
    if not physical_passed and not reasons:
        reasons.append("physical correspondence not established")
    return {
        "criterion": {"per_side_rule": PER_SIDE_RULE, "paired_rule": PAIRED_RULE, "eps": EPS,
                      "per_call_eps_multiplier": maple_per_call_eps_multiplier(),
                      "source": ("maple.surface.topographic_commit.commit._reconcile_ledger_with_physical_state; "
                                 "maple.surface.voxels._numerics (summation bound, mass-resolution classifier)"),
                      "paired_cap_note": ("the paired difference is capped at ONE mass_resolution_kg, so any "
                                          "resolved difference fails regardless of touches")},
        "evidence": {"cpu": _evidence_summary(cpu, bound_cpu), "gpu": _evidence_summary(gpu, bound_gpu),
                     "completeness": {"history_complete_both": True,
                                      "binding_supplied": {"cpu": bool(cpu.binding), "gpu": bool(gpu.binding)},
                                      "note": ("history_complete and binding are caller declarations echoed here; "
                                               "a reviewer verifies them against receipts, this report does not")}},
        "paired_bound": {"min_kg": float(bound.min(initial=0.0)), "max_kg": float(bound.max(initial=0.0)),
                         "n_capped_at_resolution": int((bound >= float(cpu.mass_resolution_kg)).sum()),
                         "n_zero": int((bound == 0.0).sum())},
        "physical_reservoirs": {"examined": reservoirs_examined, "expected": expected, "missing": missing,
                                "all_examined_passed": all_reservoirs_pass, "results": reservoir_results},
        "exact_controls": {"examined": _names(exact_controls), "all_examined_passed": all_controls_pass,
                           "results": control_results},
        "derived_diagnostics": {"examined": _names(derived_inventory_changes),
                                "all_examined_consistent": all_derived_consistent,
                                "gates_physical_verdict": False, "results": derived_results},
        "physical_correspondence_passed": bool(physical_passed),
        "physical_correspondence_reasons": reasons,
        "qualification": [
            ("operand-qualified correspondence of the examined physical reservoirs only; not conservation, not the "
             "declared strict rtol/atol gate (whose recorded result stands), not a hydrology comparison, not a "
             "storm pass; fields not listed under examined were not compared"),
            ("evidence must be the actual accumulated per-cell/class touches and gross activity through every water "
             "call and commit, bound to its run by the caller; nothing is inferred or verified here"),
        ],
    }
