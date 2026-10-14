"""CW-D1: operand-qualified backend correspondence comparator (host, saved outputs).

Pure NumPy plus MAPLE's shared numerical policy; no event, device or case is needed. Every bound in these tests
is derived in the test from MAPLE's rule with explicit operands, never copied from a failed run.
"""
import dataclasses

import numpy as np
import pytest

from maple_syrup import backend_correspondence as bc
from maple_syrup.backend_correspondence import (
    BackendEvidence,
    CorrespondenceEvidenceError,
    InventoryChangeOperands,
    compare_derived_inventory_change,
    compare_exact_control,
    compare_reservoir,
    correspondence_report,
    maple_per_call_eps_multiplier,
    paired_bound_kg,
    per_side_roundoff_bound_kg,
)

EPS = float(np.finfo(np.float64).eps)
SHAPE = (2, 3, 2)
MRES = 1e-10
DEFAULT = object()  # sentinel: a test that passes None really passes None (missing evidence), never a default


def evidence(label="cpu", *, reference=DEFAULT, gross=DEFAULT, touches=DEFAULT, mres=MRES, complete=True,
             touches_source="test ledger accumulation", activity_source="test gross accumulation", binding=None):
    return BackendEvidence(
        label=label,
        reference_inventory_kg=np.zeros(SHAPE) if reference is DEFAULT else reference,
        gross_physical_activity_kg=np.zeros(SHAPE) if gross is DEFAULT else gross,
        n_physical_touches=np.zeros(SHAPE, dtype=np.int64) if touches is DEFAULT else touches,
        mass_resolution_kg=mres, history_complete=complete,
        touches_source=touches_source, activity_source=activity_source, binding=binding)


def k_maple():
    return maple_per_call_eps_multiplier()


# --- criterion derivation ---------------------------------------------------------------------------------------------
def test_multiplier_is_read_from_the_actual_maple_rule_and_bound_matches_the_formula():
    k = k_maple()
    assert np.isfinite(k) and k > 0.0
    reference = np.full(SHAPE, 0.3)
    gross = np.full(SHAPE, 0.05)
    touches = np.full(SHAPE, 7, dtype=np.int64)
    bound = per_side_roundoff_bound_kg(evidence(reference=reference, gross=gross, touches=touches))
    expected = np.minimum(k * 7 * EPS * np.maximum(gross, reference), MRES)
    assert bound.tobytes() == expected.tobytes()
    # the larger of gross activity and reference inventory is the local scale
    gross[0, 0, 0] = 2.0
    bound = per_side_roundoff_bound_kg(evidence(reference=reference, gross=gross, touches=touches))
    assert bound[0, 0, 0] == min(k * 7 * EPS * 2.0, MRES)


def test_paired_bound_is_sum_of_sides_capped_at_one_resolution():
    k = k_maple()
    big = np.full(SHAPE, 1e3)
    many = np.full(SHAPE, 10**9, dtype=np.int64)
    cpu = evidence("cpu", reference=big, gross=big, touches=many)
    gpu = evidence("gpu", reference=big, gross=big, touches=many)
    assert np.all(per_side_roundoff_bound_kg(cpu) == MRES)  # each side capped by MAPLE's rule
    assert np.all(paired_bound_kg(cpu, gpu) == MRES)  # ONE resolution, not the sum of two caps
    small_touch = evidence("gpu", reference=np.full(SHAPE, 0.5), touches=np.full(SHAPE, 3, dtype=np.int64))
    paired = paired_bound_kg(cpu, small_touch)
    assert np.all(paired == np.minimum(MRES + k * 3 * EPS * 0.5, MRES))


# --- acceptance and rejection ----------------------------------------------------------------------------------------
def test_zero_touches_and_no_activity_accept_only_identity_including_empty_and_tiny_holdings():
    cpu, gpu = evidence("cpu"), evidence("gpu")
    assert np.all(paired_bound_kg(cpu, gpu) == 0.0)
    empty = np.zeros(SHAPE)
    assert compare_reservoir("active", empty, empty.copy(), paired_bound_kg(cpu, gpu)).passed
    nudged = empty.copy()
    nudged[1, 2, 1] = 1e-300
    assert not compare_reservoir("active", empty, nudged, paired_bound_kg(cpu, gpu)).passed
    tiny = np.full(SHAPE, 1e-12)
    cpu_t, gpu_t = evidence("cpu", reference=tiny), evidence("gpu", reference=tiny)
    paired = paired_bound_kg(cpu_t, gpu_t)
    assert np.all(paired == 2 * k_maple() * EPS * 1e-12)  # one touch assumed, tiny scale, far below resolution
    assert compare_reservoir("active", tiny, tiny.copy(), paired).passed
    perturbed = np.nextafter(np.nextafter(tiny, np.inf), np.inf)  # two ulps of 1e-12
    result = compare_reservoir("active", tiny, perturbed, paired)
    assert result.passed == bool(np.all(np.abs(tiny - perturbed) <= paired))


def test_touches_at_an_unrelated_cell_cannot_inflate_another_cells_allowance():
    reference = np.full(SHAPE, 0.3)
    touches = np.zeros(SHAPE, dtype=np.int64)
    touches[0, 0, :] = 10**6  # heavy activity at (0, 0) only
    cpu = evidence("cpu", reference=reference, touches=touches)
    gpu = evidence("gpu", reference=reference, touches=touches)
    paired = paired_bound_kg(cpu, gpu)
    a = reference.copy()
    b = reference.copy()
    b[1, 1, 0] += 1e-14  # roundoff-scale difference at an untouched cell
    result = compare_reservoir("active", a, b, paired)
    assert not result.passed and result.worst_index == (1, 1, 0)
    assert paired[1, 1, 0] == 2 * k_maple() * EPS * 0.3 < 1e-14
    assert paired[0, 0, 0] == MRES


def test_sub_resolution_roundoff_is_accepted_only_when_the_recorded_touches_justify_it():
    k = k_maple()
    reference = np.full(SHAPE, 0.3)
    a = reference.copy()
    b = reference.copy()
    b[0, 1, 1] += 1e-14  # below mass resolution, above a few ulps of 0.3 kg
    needed = int(np.ceil(1e-14 / (2 * k * EPS * 0.3)))  # touches per side that make the paired allowance >= 1e-14
    assert needed > 1
    few = evidence("cpu", reference=reference, touches=np.full(SHAPE, 1, dtype=np.int64))
    few_gpu = evidence("gpu", reference=reference, touches=np.full(SHAPE, 1, dtype=np.int64))
    assert not compare_reservoir("active", a, b, paired_bound_kg(few, few_gpu)).passed
    enough = evidence("cpu", reference=reference, touches=np.full(SHAPE, needed, dtype=np.int64))
    enough_gpu = evidence("gpu", reference=reference, touches=np.full(SHAPE, needed, dtype=np.int64))
    assert compare_reservoir("active", a, b, paired_bound_kg(enough, enough_gpu)).passed


def test_resolved_difference_is_rejected_even_with_enormous_touch_counts():
    big = np.full(SHAPE, 1e3)
    many = np.full(SHAPE, 10**12, dtype=np.int64)
    cpu = evidence("cpu", reference=big, gross=big, touches=many)
    gpu = evidence("gpu", reference=big, gross=big, touches=many)
    paired = paired_bound_kg(cpu, gpu)
    a = big.copy()
    b = big.copy()
    b[0, 0, 0] += 2 * MRES
    assert not compare_reservoir("voxel", a, b, paired).passed
    b[0, 0, 0] = a[0, 0, 0] + 1.5 * MRES  # would pass a two-resolution sum of caps; fails the one-resolution cap
    assert not compare_reservoir("voxel", a, b, paired).passed
    b[0, 0, 0] = a[0, 0, 0] + 0.5 * MRES
    assert compare_reservoir("voxel", a, b, paired).passed


def test_voxel_reservoir_broadcasts_the_cell_class_bound_over_the_voxel_axis():
    reference = np.full(SHAPE, 0.3)
    touches = np.full(SHAPE, 50, dtype=np.int64)
    paired = paired_bound_kg(evidence("cpu", reference=reference, touches=touches),
                             evidence("gpu", reference=reference, touches=touches))
    voxel = np.full((2, 3, 5, 2), 0.25)
    other = voxel.copy()
    other[1, 2, 3, 0] += 0.5 * paired[1, 2, 0]
    assert compare_reservoir("voxel", voxel, other, paired).passed
    other[1, 2, 3, 0] = voxel[1, 2, 3, 0] + 2.0 * paired[1, 2, 0]
    result = compare_reservoir("voxel", voxel, other, paired)
    assert not result.passed and result.worst_index == (1, 2, 3, 0) and result.n_failed == 1
    assert result.failures[0]["index"] == [1, 2, 3, 0]


def test_altered_routing_or_counters_are_rejected_and_floats_are_refused_as_controls():
    aspect = np.array([[1, 2, 3], [4, 1, 2]], dtype=np.int8)
    assert compare_exact_control("aspect", aspect, aspect.copy()).passed
    changed = aspect.copy()
    changed[0, 1] = 4
    result = compare_exact_control("aspect", aspect, changed)
    assert not result.passed and result.n_failed == 1 and result.failures[0]["index"] == [0, 1]
    assert compare_exact_control("commits", 813, 813).passed
    scalar = compare_exact_control("commits", 813, 812)
    assert not scalar.passed and scalar.n_failed == 1 and scalar.failures[0] == {"index": [], "a": 813, "b": 812}
    assert not compare_exact_control("outlet", np.array([True, False]), np.array([True, True])).passed
    with pytest.raises(CorrespondenceEvidenceError, match="integer or boolean"):
        compare_exact_control("depth", np.zeros(3), np.zeros(3))


# --- refusals ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    {"touches": None},  # missing touches: never defaulted to zero
    {"gross": None},  # missing gross activity
    {"reference": None},  # missing reference inventory
    {"touches": np.zeros(SHAPE)},  # float touches
    {"touches": np.full(SHAPE, -1, dtype=np.int64)},
    {"touches": np.zeros((2, 3, 3), dtype=np.int64)},  # shape mismatch
    {"reference": np.full(SHAPE, -1e-3)},
    {"reference": np.full(SHAPE, np.nan)},
    {"gross": np.full(SHAPE, np.inf)},
    {"gross": np.zeros(SHAPE, dtype=np.float32)},
    {"reference": np.zeros((2, 3))},
    {"reference": [[0.0] * 2] * 3},  # not a host ndarray
    {"mres": 0.0},
    {"mres": -1e-10},
    {"mres": np.inf},
    {"mres": True},
    {"complete": False},
    {"touches_source": ""},
    {"activity_source": "  "},
    {"binding": {}},
    {"binding": {"source_pin": ""}},
    {"binding": "sha256:abc"},
])
def test_malformed_or_incomplete_evidence_is_refused(bad):
    with pytest.raises(CorrespondenceEvidenceError):
        per_side_roundoff_bound_kg(evidence(**bad))


def test_missing_history_is_refused_through_the_dataclass_directly():
    """No helper defaults involved: the evidence object itself carries None and is refused."""
    base = evidence()
    for name in ("n_physical_touches", "gross_physical_activity_kg", "reference_inventory_kg"):
        with pytest.raises(CorrespondenceEvidenceError, match="missing"):
            per_side_roundoff_bound_kg(dataclasses.replace(base, **{name: None}))


def test_mixed_evidence_and_malformed_reservoirs_are_refused_not_failed():
    cpu = evidence("cpu")
    with pytest.raises(CorrespondenceEvidenceError, match="mixed evidence"):
        paired_bound_kg(cpu, BackendEvidence("gpu", np.zeros((2, 3, 3)), np.zeros((2, 3, 3)),
                                             np.zeros((2, 3, 3), dtype=np.int64), MRES, True, "s", "s"))
    with pytest.raises(CorrespondenceEvidenceError, match="mass_resolution_kg"):
        paired_bound_kg(cpu, evidence("gpu", mres=2e-10))
    paired = paired_bound_kg(cpu, evidence("gpu"))
    good = np.zeros(SHAPE)
    with pytest.raises(CorrespondenceEvidenceError, match="negative mass"):
        compare_reservoir("active", good, np.full(SHAPE, -1e-20), paired)
    with pytest.raises(CorrespondenceEvidenceError, match="non-finite"):
        compare_reservoir("active", np.full(SHAPE, np.nan), good, paired)
    with pytest.raises(CorrespondenceEvidenceError, match="shapes differ"):
        compare_reservoir("active", good, np.zeros((2, 3, 3)), paired)
    with pytest.raises(CorrespondenceEvidenceError, match="layout"):
        compare_reservoir("voxel", np.zeros((2, 3, 4, 3)), np.zeros((2, 3, 4, 3)), paired)
    with pytest.raises(CorrespondenceEvidenceError, match="float64"):
        compare_reservoir("active", good.astype(np.float32), good.astype(np.float32), paired)
    with pytest.raises(CorrespondenceEvidenceError, match="host numpy.ndarray"):
        compare_reservoir("active", good.tolist(), good, paired)
    with pytest.raises(CorrespondenceEvidenceError, match="nothing to examine"):
        correspondence_report(cpu=cpu, gpu=evidence("gpu"))


# --- derived inventory change -------------------------------------------------------------------------------------------
def _column_case(seed=5, nv=20):
    rng = np.random.default_rng(seed)
    voxel = rng.uniform(14.0, 16.0, size=(2, 3, nv, 2))
    active = rng.uniform(0.3, 0.6, size=SHAPE)
    initial = voxel.sum(axis=2) + active - 1e-5  # a tiny genuine net change against ~300 kg operands
    return voxel, active, initial


def test_derived_net_change_uses_true_operand_scale_not_the_tiny_net():
    from maple.surface.voxels._numerics import summation_error_bound_kg

    voxel, active, initial = _column_case()
    cpu = InventoryChangeOperands(voxel, active, initial)
    # GPU side: last-bit perturbations of some voxel entries, as two reduction histories would leave
    gpu_voxel = voxel.copy()
    gpu_voxel[0, 1, 7, 0] = np.nextafter(gpu_voxel[0, 1, 7, 0], np.inf)
    gpu_voxel[1, 2, 3, 1] = np.nextafter(gpu_voxel[1, 2, 3, 1], -np.inf)
    gpu = InventoryChangeOperands(gpu_voxel, active.copy(), initial.copy())
    report = compare_derived_inventory_change("bed_change", cpu, gpu)
    comparison = report["comparison"]
    assert comparison["passed"] and report["consistent"] and comparison["kind"] == "derived_inventory_change"
    # largest operand the recompute touches at any cell/class: column total, active, initial or the result
    column_total = voxel.sum(axis=2)
    cpu_change = (column_total + active) - initial
    largest = float(np.maximum.reduce([column_total, active, initial, np.abs(cpu_change)]).max())
    expected_roundoff = summation_error_bound_kg(voxel.shape[2] + 2, largest)
    assert report["max_recompute_roundoff_bound_kg"] == pytest.approx(expected_roundoff)
    assert report["max_recompute_roundoff_bound_kg"] > 1e2 * 1e-14  # orders above an atol scaled to the tiny net
    assert float(np.abs(cpu_change).max()) < 1e-3
    assert report["max_propagated_operand_difference_kg"] == pytest.approx(
        float(np.abs(voxel - gpu_voxel).sum(axis=2).max()))
    assert np.max(np.abs(comparison["worst_values"])) < 1e-3


def test_saved_derived_arrays_are_reported_against_their_own_host_recompute():
    voxel, active, initial = _column_case(seed=9)
    host = (voxel.sum(axis=2) + active) - initial
    cpu = InventoryChangeOperands(voxel, active, initial, saved_change_kg=host.copy())
    device_like = host + 1e-13  # a different reduction order's result, within the recompute roundoff bound
    gpu = InventoryChangeOperands(voxel.copy(), active.copy(), initial.copy(), saved_change_kg=device_like)
    report = compare_derived_inventory_change("bed_change", cpu, gpu)
    assert report["saved_cpu"]["bitwise_equal_to_host_recompute"] and report["saved_cpu"]["within_own_roundoff_bound"]
    assert not report["saved_gpu"]["bitwise_equal_to_host_recompute"]
    assert report["saved_gpu"]["within_own_roundoff_bound"] and report["consistent"]
    wrong = InventoryChangeOperands(voxel.copy(), active.copy(), initial.copy(), saved_change_kg=host + 1e-6)
    report = compare_derived_inventory_change("bed_change", cpu, wrong)
    assert report["comparison"]["passed"] and not report["saved_gpu"]["within_own_roundoff_bound"]
    assert not report["consistent"]
    with pytest.raises(CorrespondenceEvidenceError, match="layout"):
        compare_derived_inventory_change("bed_change", cpu, InventoryChangeOperands(voxel, active, np.zeros((2, 3, 3))))


# --- full report ---------------------------------------------------------------------------------------------------------
def _report_inputs():
    reference = np.full(SHAPE, 0.3)
    touches = np.full(SHAPE, 40, dtype=np.int64)
    binding = {"source_pin": "sha256:deadbeef", "rows_sha256": "sha256:cafe"}
    cpu = evidence("cpu", reference=reference, touches=touches, binding=binding)
    gpu = evidence("gpu", reference=reference, touches=touches)
    voxel, active, initial = _column_case(seed=11)
    aspect = np.ones((2, 3), dtype=np.int8)
    return cpu, gpu, voxel, active, initial, aspect


def test_report_separates_physical_controls_and_derived_sections_and_names_examined_fields():
    cpu, gpu, voxel, active, initial, aspect = _report_inputs()
    report = correspondence_report(
        cpu=cpu, gpu=gpu,
        reservoirs={"active": (active, active.copy()), "voxel": (voxel, voxel.copy())},
        exact_controls={"aspect": (aspect, aspect.copy()), "n_commits": (5, 5)},
        derived_inventory_changes={"bed_change": (InventoryChangeOperands(voxel, active, initial),
                                                  InventoryChangeOperands(voxel.copy(), active.copy(), initial.copy()))},
        expected_reservoirs=("active", "voxel"))
    assert report["physical_correspondence_passed"] and report["physical_correspondence_reasons"] == []
    assert report["physical_reservoirs"]["examined"] == ["active", "voxel"]
    assert report["physical_reservoirs"]["missing"] == [] and report["physical_reservoirs"]["all_examined_passed"]
    assert report["exact_controls"]["examined"] == ["aspect", "n_commits"] and report["exact_controls"]["all_examined_passed"]
    assert report["derived_diagnostics"]["examined"] == ["bed_change"]
    assert report["derived_diagnostics"]["all_examined_consistent"]
    assert report["derived_diagnostics"]["gates_physical_verdict"] is False
    assert report["criterion"]["per_call_eps_multiplier"] == k_maple()
    assert report["criterion"]["per_side_rule"] == bc.PER_SIDE_RULE and report["criterion"]["paired_rule"] == bc.PAIRED_RULE
    assert report["evidence"]["cpu"]["binding"] == {"source_pin": "sha256:deadbeef", "rows_sha256": "sha256:cafe"}
    assert report["evidence"]["completeness"]["binding_supplied"] == {"cpu": True, "gpu": False}
    assert report["evidence"]["gpu"]["touches_max"] == 40 and report["paired_bound"]["max_kg"] <= MRES
    assert "not a storm pass" in report["qualification"][0]


def test_report_never_claims_physical_correspondence_without_examined_physical_reservoirs():
    cpu, gpu, voxel, active, initial, aspect = _report_inputs()
    # controls only
    report = correspondence_report(cpu=cpu, gpu=gpu, exact_controls={"aspect": (aspect, aspect.copy())})
    assert not report["physical_correspondence_passed"]
    assert report["physical_reservoirs"]["examined"] == [] and not report["physical_reservoirs"]["all_examined_passed"]
    assert any("no physical reservoir examined" in r for r in report["physical_correspondence_reasons"])
    assert report["exact_controls"]["all_examined_passed"]
    # derived only
    report = correspondence_report(cpu=cpu, gpu=gpu, derived_inventory_changes={
        "bed_change": (InventoryChangeOperands(voxel, active, initial),
                       InventoryChangeOperands(voxel.copy(), active.copy(), initial.copy()))})
    assert not report["physical_correspondence_passed"] and report["derived_diagnostics"]["all_examined_consistent"]
    # empty collections supplied explicitly
    report = correspondence_report(cpu=cpu, gpu=gpu, reservoirs={}, exact_controls={"n": (1, 1)})
    assert not report["physical_correspondence_passed"]
    # an expected reservoir that was not supplied is reported missing and blocks the verdict
    report = correspondence_report(cpu=cpu, gpu=gpu, reservoirs={"active": (active, active.copy())},
                                   expected_reservoirs=("active", "voxel", "mobile"))
    assert not report["physical_correspondence_passed"]
    assert report["physical_reservoirs"]["missing"] == ["voxel", "mobile"]
    assert report["physical_reservoirs"]["all_examined_passed"]
    assert any("expected reservoirs not supplied" in r for r in report["physical_correspondence_reasons"])


def test_report_fail_paths_keep_sections_independent():
    cpu, gpu, voxel, active, initial, aspect = _report_inputs()
    broken = active.copy()
    broken[0, 0, 0] += 3 * MRES
    report = correspondence_report(cpu=cpu, gpu=gpu, reservoirs={"active": (active, broken)},
                                   exact_controls={"aspect": (aspect, aspect.copy())})
    assert not report["physical_correspondence_passed"]
    assert not report["physical_reservoirs"]["results"]["active"]["passed"]
    assert report["physical_reservoirs"]["results"]["active"]["failures"][0]["index"] == [0, 0, 0]
    assert report["exact_controls"]["all_examined_passed"]
    changed = aspect.copy()
    changed[1, 1] = 3
    report = correspondence_report(cpu=cpu, gpu=gpu, reservoirs={"active": (active, active.copy())},
                                   exact_controls={"aspect": (aspect, changed)})
    assert not report["physical_correspondence_passed"] and report["physical_reservoirs"]["all_examined_passed"]
    assert not report["exact_controls"]["all_examined_passed"]
    assert any("exact control differs" in r for r in report["physical_correspondence_reasons"])
    # an inconsistent derived diagnostic is reported but does not change the physical verdict
    wrong = InventoryChangeOperands(voxel.copy(), active.copy(), initial.copy(),
                                    saved_change_kg=((voxel.sum(axis=2) + active) - initial) + 1e-6)
    report = correspondence_report(cpu=cpu, gpu=gpu, reservoirs={"active": (active, active.copy())},
                                   derived_inventory_changes={"bed_change": (InventoryChangeOperands(voxel, active, initial), wrong)})
    assert report["physical_correspondence_passed"] and not report["derived_diagnostics"]["all_examined_consistent"]
