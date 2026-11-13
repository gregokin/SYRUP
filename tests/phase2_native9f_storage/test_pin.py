"""E4j: the ONE explicit native 9f package pin in `sediment_bed` and its exact recognizer rules (no broad fallback, no message catches)."""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from maple_syrup import sediment_bed as sb

PIN = "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"


def test_the_bound_runtime_is_exactly_the_pinned_native9f_package_and_the_pin_is_recorded():
    import maple

    expected = os.environ.get("MAPLE_SYRUP_EXPECTED_MAPLE_ROOT")
    if expected:  # a configured wrong source FAILS
        assert Path(expected).resolve() in Path(maple.__file__).resolve().parents
    assert sb.bound_maple_package_digest() == PIN == sb.NATIVE9F_MAPLE_DIGEST
    assert PIN in sb.TYPED_JOURNAL_SOURCE_DIGESTS and sb.JOURNAL_RECOGNIZER_SOURCES[PIN] == "typed_journal_capacity_error"
    assert sb.journal_recognizer_source() == "typed_journal_capacity_error"
    # historical pins unchanged
    assert sb.TYPED_JOURNAL_SOURCE_DIGESTS == {sb.TYPED_JOURNAL_RECOGNIZER_MAPLE_DIGEST, sb.MERGED_39BCF79_MAPLE_DIGEST, PIN}
    assert sb.JOURNAL_RECOGNIZER_SOURCES[sb.JOURNAL_RECOGNIZER_MAPLE_DIGEST] == "m94_pinned_message"


def test_class_identity_count_and_context_rules_are_unchanged_on_native9f():
    from maple.surface.voxels import deep_store

    cls = sb.require_typed_journal_capacity_error()
    assert cls is deep_store.JournalCapacityError and issubclass(cls, ValueError)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pond_windowed"))
    import numpy as np
    from pond_helpers import basin
    from windowed_helpers import dense_deep_bed

    _dense, c = dense_deep_bed(basin(), depth=np.zeros(basin().shape))
    cap = 1000
    wc = replace(c, window_max_journal_records=cap)
    def make(cur, new, mx):
        return cls(f"pending journal exceeds max_journal_records={mx}", current_records=cur, new_records=new, max_records=mx)

    good = make(900, 300, cap)
    assert str(good) == f"pending journal exceeds max_journal_records={cap}"
    assert sb.is_carried_append_cap_refusal(good, wc, 400)
    assert not sb.is_carried_append_cap_refusal(good, wc, 0)  # nothing carried
    assert not sb.is_carried_append_cap_refusal(ValueError(str(good)), wc, 400)  # a plain ValueError with the same text is NOT recognised
    assert not sb.is_carried_append_cap_refusal(make(900, 300, cap + 1), wc, 400)  # another cap
    assert not sb.is_carried_append_cap_refusal(make(900, 2000, cap), wc, 400)  # single-step overflow

    class Sub(cls):
        pass

    assert not sb.is_carried_append_cap_refusal(Sub(str(good), current_records=900, new_records=300, max_records=cap), wc, 400)  # exact class only


def test_an_unlisted_source_is_still_refused(monkeypatch):
    monkeypatch.setitem(sb._MAPLE_DIGEST_CACHE, "digest", "0" * 64)
    assert sb.journal_recognizer_source() is None
    with pytest.raises(sb.BedIntegrationError, match="accepted MAPLE source"):
        sb.check_deep_storage_source("overlay", 1 << 20)
    assert sb.check_deep_storage_source("full", None) is None
