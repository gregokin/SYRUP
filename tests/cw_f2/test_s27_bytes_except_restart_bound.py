"""CW-F2: the CW-B30 fresh-process byte comparison against the immutable S27 baseline, for the tree under test.

Reuses the CW-B30 harness unchanged (`run_tree` asserts each subprocess imported that tree's own `maple_syrup`). Every
array of every output and checkpoint must be byte-identical; the summaries must be equal after removing ONLY the
validated additive `restart_resource_bound` record (`comparable_candidate_summaries`). In addition this test requires
the record to be PRESENT in every runner summary of the tree under test and ABSENT from the older S27 source. Skips
truthfully when the S27 baseline manifest or tree is unavailable (`s27_tree`)."""
from __future__ import annotations

from pathlib import Path

import pytest
from b30_cases import KINDS
from test_bytes_vs_s27 import (
    B30_ROOT,
    MODES,
    PROVENANCE_TEXT,
    RESTART_BOUND_KEY,
    comparable_candidate_summaries,
    provenance_equal,
    run_tree,
    s27_tree,
)

import maple_syrup


def runner_summaries(summaries):
    """The complete-runner summaries of one mode (the `load` record is a handoff type/accounting, not a summary)."""
    return {label: value for label, value in summaries.items()
            if isinstance(value, dict) and 'status' in value and 'identity' in value}


@pytest.mark.parametrize('kind', KINDS)
def test_arrays_byte_identical_and_summaries_differ_only_by_the_restart_bound(kind, tmp_path):
    # The tree under test is the one this process imports (private candidate or, after adoption, the root project).
    assert Path(maple_syrup.__file__).resolve().parent == B30_ROOT / 'src' / 'maple_syrup'
    s27 = run_tree(s27_tree(), kind, tmp_path / 's27')
    candidate = run_tree(B30_ROOT, kind, tmp_path / 'candidate')
    for mode in MODES:
        a, b = s27[mode]['arrays'], candidate[mode]['arrays']
        assert set(a) == set(b) and len(a) > 0, mode
        bad = [k for k in a if not (a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
                                    and a[k].tobytes() == b[k].tobytes())]
        bad = [k for k in bad if not (k.rsplit('/', 1)[-1] in PROVENANCE_TEXT and provenance_equal(a[k], b[k]))]
        assert not bad, f'{mode}: {len(bad)} arrays differ, first {bad[:5]}'
        old, new = s27[mode]['summaries'], candidate[mode]['summaries']
        assert all(RESTART_BOUND_KEY not in v for v in runner_summaries(old).values()), mode
        assert bool(runner_summaries(new)) == (mode != 'load'), mode  # every runner mode has a summary to check
        assert all(RESTART_BOUND_KEY in v for v in runner_summaries(new).values()), mode
        assert old == comparable_candidate_summaries(old, new), mode
