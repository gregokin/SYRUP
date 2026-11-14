"""Exact-instance provenance of genuine daily withdrawal outcomes (CPU)."""
from __future__ import annotations

import copy
from dataclasses import replace

import dbuild
import numpy as np
import pytest

from maple_syrup import floating_daily as fd


def _branches():
    origin, *_ = dbuild.lineage(backend="numpy")
    a, b = dbuild.day(origin).state, dbuild.day(origin).state
    assert a is not b and a.digest == b.digest
    request = np.full((a.ctx.n_layers, a.ctx.n_cells), 1e-6)
    outcome = fd.request_floating_withdrawal(a, expected_parent=fd.expect_parent(a), request_m=request,
        expected_sequence=0, available_bytes_fn=dbuild.AVAIL)
    return a, b, outcome


def test_genuine_outcome_accepts_fresh_exact_parent_capability_without_changes():
    a, _, outcome = _branches()
    before = (a.water.water_m.tobytes(), a.digest, a.request_sequence)
    assert fd.verify_floating_withdrawal(a, expected_parent=fd.expect_parent(a), outcome=outcome) is outcome
    assert (a.water.water_m.tobytes(), a.digest, a.request_sequence) == before
    assert outcome.state.request_sequence == 1 and outcome.sequence == 0


def test_equal_digest_alternate_instance_and_wrong_expected_parent_refuse():
    a, b, outcome = _branches()
    with pytest.raises(fd.DailyRefusal, match="exact parent"):
        fd.verify_floating_withdrawal(b, expected_parent=fd.expect_parent(b), outcome=outcome)
    with pytest.raises(fd.DailyRefusal, match="another instance"):
        fd.verify_floating_withdrawal(a, expected_parent=fd.expect_parent(b), outcome=outcome)
    with pytest.raises(fd.DailyRefusal):
        fd.verify_floating_withdrawal(a, expected_parent=None, outcome=outcome)


def test_direct_copied_and_replaced_outcomes_are_not_producer_authority():
    a, _, outcome = _branches()
    forged = fd.WithdrawalOutcome(outcome.state, outcome.actual_m, outcome.unmet_m, outcome.scale,
        outcome.requested_m3, outcome.actual_m3, outcome.unmet_m3, outcome.sequence)
    for bad in (forged, copy.copy(outcome), replace(outcome), object(), None):
        with pytest.raises(fd.DailyRefusal):
            fd.verify_floating_withdrawal(a, expected_parent=fd.expect_parent(a), outcome=bad)


def test_changed_field_on_genuine_outcome_is_detected_and_original_remains_usable():
    a, _, outcome = _branches()
    original = outcome.actual_m
    object.__setattr__(outcome, "actual_m", original.copy())
    try:
        with pytest.raises(fd.DailyRefusal, match="field references"):
            fd.verify_floating_withdrawal(a, expected_parent=fd.expect_parent(a), outcome=outcome)
    finally:
        object.__setattr__(outcome, "actual_m", original)
    assert fd.verify_floating_withdrawal(a, expected_parent=fd.expect_parent(a), outcome=outcome) is outcome
