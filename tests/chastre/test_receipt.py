"""The compiled-case initial-fill receipt: a balanced receipt must not be rejected by aggregate-rounding cancellation, and a real
imbalance must still be rejected by the UNCHANGED bound. The pure-arithmetic tests need no MAPLE patch; the compile tests need the
isolated Chastre MAPLE candidate (72310c49 + benchmarks/chastre/maple_receipt.patch) and fail on the unpatched package.
Nothing here was run by its author (file-only tools); Codex records results."""
import copy
import inspect
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "agent_handoffs" / "tasks" / "chastre_timing"


def _bound(n_cells, capacity=125.0):
    from maple.surface.voxels._numerics import summation_error_bound_kg

    return summation_error_bound_kg(n_cells, capacity)


def _ordinary_gap(requested, actual, residual, shortfall):
    """The original arithmetic: four independently rounded aggregate scalars, then one subtraction."""
    totals = [float(np.sum(a)) for a in (requested, actual, residual, shortfall)]
    return abs(totals[0] - (totals[1] + totals[2] + totals[3]))


def _signed_fsum_gap(requested, actual, residual, shortfall):
    return abs(math.fsum(
        value for array, sign in ((requested, 1.0), (actual, -1.0), (residual, -1.0), (shortfall, -1.0))
        for value in (sign * x for x in np.asarray(array, dtype=np.float64).ravel().tolist())))


def test_independent_rational_receipt_balanced_per_cell_but_one_ulp_apart_in_ordinary_scalar_sums():
    """A synthetic receipt that balances exactly per cell yet reproduces the false positive of the ORIGINAL validator's NumPy
    aggregation (`np.sum` of each operand, then `requested - (actual + residual + shortfall)`, here `_ordinary_gap`): one ULP of
    the total. The native captured operands show the same effect at full size
    (`test_native_failed_tile_receipt_signed_sum_is_exactly_zero_but_ordinary_is_one_ulp`). Python's built-in `sum` is NOT used
    as the oracle for the ordinary arithmetic: since Python 3.12 it compensates float sums."""
    ulp = 2.0 ** -25  # the spacing of FP64 at 2**27 kg
    requested = np.array([2.0 ** 27, 0.4 * ulp, 0.4 * ulp])
    actual = np.array([2.0 ** 27, 0.0, 0.0])
    residual = np.array([0.0, 0.4 * ulp, 0.4 * ulp])
    shortfall = np.zeros(3)
    # exact rational arithmetic on the unchanged binary64 operands (Fraction.from_float is exact, and Fraction sums are exact,
    # unlike a Decimal context that rounds to its precision): the receipt balances exactly
    def exact_total(array):
        return sum((Fraction.from_float(float(x)) for x in array), Fraction(0))

    assert exact_total(requested) - exact_total(actual) - exact_total(residual) - exact_total(shortfall) == 0
    # the original NumPy aggregation: np.sum(requested) = 2**27 but np.sum(actual) + np.sum(residual) = 2**27 + ulp
    assert _ordinary_gap(requested, actual, residual, shortfall) == ulp
    assert _signed_fsum_gap(requested, actual, residual, shortfall) == 0.0
    assert _bound(3) < ulp  # the unchanged bound is far below that ordinary-arithmetic gap, so the old evaluation refused it


def test_a_real_signed_imbalance_exceeds_the_unchanged_bound_in_the_compensated_sum_too():
    requested = np.array([2.0 ** 27, 1.0, 1.0])
    actual = np.array([2.0 ** 27, 0.0, 0.0])
    residual = np.array([0.0, 1.0, 1.0])
    shortfall = np.zeros(3)
    assert _signed_fsum_gap(requested, actual, residual, shortfall) == 0.0
    requested[1] += 1.0e-3  # a real imbalance of one cell, far above the bound
    assert _signed_fsum_gap(requested, actual, residual, shortfall) == pytest.approx(1.0e-3, rel=1e-6)
    assert _signed_fsum_gap(requested, actual, residual, shortfall) > _bound(3)


def test_native_failed_tile_receipt_signed_sum_is_exactly_zero_but_ordinary_is_one_ulp():
    arrays_path, audit_path = TASK / "native_receipt_arrays.npz", TASK / "native_receipt_audit.json"
    if not (arrays_path.is_file() and audit_path.is_file()):
        pytest.skip("the archived native receipt evidence is not present")
    audit = json.loads(audit_path.read_text())
    with np.load(arrays_path) as data:
        operands = [data[f"operand{i}"] for i in range(4)]
    n_cells = operands[0].size
    bound = _bound(n_cells, audit["capacity_kg"])
    assert n_cells == audit["n_cells"] and bound == pytest.approx(audit["unchanged_bound_kg"], rel=1e-12)
    ordinary = _ordinary_gap(*operands)
    assert ordinary == audit["ordinary_gap_kg"] == audit["requested_ulp_kg"] and ordinary > bound  # the original false rejection
    assert _signed_fsum_gap(*operands) == 0.0 <= bound  # compensated signed sum (math.fsum): zero, same bound
    # independent exact rational check of the same captured binary64 operands (no floating-point rounding anywhere)
    exact = sum((Fraction.from_float(float(x)) * sign for array, sign in zip(operands, (1, -1, -1, -1), strict=True)
                 for x in array.ravel().tolist()), Fraction(0))
    assert exact == 0
    perturbed = [a.copy() for a in operands]
    perturbed[1].flat[0] -= 1.0e-3  # one actual cell short by far more than the bound: a true imbalance
    assert _signed_fsum_gap(*perturbed) > bound


# --------------------------------------------------------------------------------------------------------------------
# the actual (patched) MAPLE validator through a real small compile
# --------------------------------------------------------------------------------------------------------------------
def _wrap_validator(monkeypatch, mutate=None):
    from maple.case_tools.compilers import case_compiler

    original = case_compiler.validate_compiled_case
    signature = inspect.signature(original)
    calls = []

    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        calls.append(bound.arguments["deposit_result"])
        if mutate is not None and bound.arguments["deposit_result"] is not None:
            bound.arguments["deposit_result"] = mutate(bound.arguments["deposit_result"])
        return original(*bound.args, **bound.kwargs)

    monkeypatch.setattr(case_compiler, "validate_compiled_case", wrapper)
    return calls


def test_candidate_validator_accepts_the_balanced_receipt_and_still_runs_the_independent_validators(
        rfid_source, dtm_path, tmp_path, monkeypatch):
    from maple.case_tools.validators import compiled_case
    from test_tiled_case import _definition

    from maple_syrup.chastre_case import compile_tile

    spy = []
    original = compiled_case.check_active_layer_voxel_partition
    monkeypatch.setattr(compiled_case, "check_active_layer_voxel_partition",
                        lambda *a, **k: (spy.append(1), original(*a, **k))[1])
    calls = _wrap_validator(monkeypatch)
    record, _ = compile_tile(_definition(rfid_source, dtm_path), 0, 0, 4, tmp_path / "tile")
    assert calls and calls[0] is not None and record["files"]
    assert spy, "the independent active-layer/voxel partition validator must still run"


def test_candidate_validator_rejects_a_real_receipt_imbalance_above_the_unchanged_bound(rfid_source, dtm_path, tmp_path,
                                                                                     monkeypatch):
    from test_tiled_case import _definition

    from maple_syrup.chastre_case import compile_tile

    def mutate(receipt):
        changed = copy.copy(receipt)
        requested = np.array(receipt.requested_total_mass_kg, dtype=np.float64, copy=True)
        requested.flat[0] += 1.0e-3  # one actual receipt cell, far above the bound
        object.__setattr__(changed, "requested_total_mass_kg", requested)
        return changed

    _wrap_validator(monkeypatch, mutate)
    with pytest.raises(Exception, match="compensated signed per-cell sum"):
        compile_tile(_definition(rfid_source, dtm_path), 0, 0, 4, tmp_path / "tile")
