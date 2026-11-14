"""Codex fault probes: actual native state fixtures, not wind-physics evidence."""
from dataclasses import replace

import pytest
import wind_builders as wb

from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_wind_handoff as wh
from maple_syrup import native_floating_soil as nfs


def test_strided_integrity_copy_fits_declared_scratch(monkeypatch):
    import hashlib
    import numpy as np
    from maple_syrup import floating_wind_handoff as wh
    values = np.arange(2 * (wh._HASH_CHUNK // 8), dtype=np.float64)[::2]
    copied = []
    real = np.ascontiguousarray

    def observe(array):
        result = real(array)
        copied.append(result.nbytes)
        return result

    monkeypatch.setattr(np, "ascontiguousarray", observe)
    wh._feed_array(hashlib.sha256(), values)
    assert max(copied) == wh._HASH_CHUNK
    assert max(copied) < wh._HASH_SCRATCH <= wh._ADOPT_BYTES


@pytest.mark.parametrize("release_elsewhere", [False, True])
def test_start_cannot_publish_after_another_action_supersedes_the_owner(monkeypatch, release_elsewhere):
    ns = wb.build()
    real_retire = nfs.retire_native_owner_to_lineage
    proposed, replacement = [], []

    def intervene(owner):
        if release_elsewhere:
            real_retire(owner)
        else:
            native_bed = wb.fb.native_bed(ns.bed, ns.ctx)
            candidate = nfs.stage_native_refresh(owner, native_bed, native_bed, available_bytes_fn=wb.AVAIL)
            replacement.append(nfs.adopt_native_refresh(owner, candidate))
        real_retire(owner)  # must raise the genuine native stale-owner refusal

    monkeypatch.setattr(nfs, "retire_native_owner_to_lineage", intervene)
    with pytest.raises(nfs.NativeStaleError):
        wh.start_wind_boundary(ns.owner, ns.bed, ns.factory, bed_context=ns.ctx, limits=wb.LIMITS,
            available_bytes_fn=wb.AVAIL, before_publish=lambda lin, *_: proposed.append(lin))
    assert proposed and proposed[0].head is None
    if replacement:
        assert replacement[0].is_current


def test_replacing_a_repeated_array_reference_is_detected():
    ns = wb.started()
    candidate = wb.stage(ns, wb.child_same(ns.b.bed))
    parts = candidate._cell["parts"]
    candidate._cell["parts"] = replace(parts, qx_m2_s=parts.qy_m2_s, qy_m2_s=parts.qx_m2_s)
    with pytest.raises(wh.WindHandoffError, match="integrity"):
        wh.adopt_wind_handoff(ns.lin, ns.b, candidate)
    assert ns.lin.head is ns.b and not candidate.consumed


def test_initial_callback_receives_the_unpublished_proposal():
    ns = wb.build()
    seen = []

    def before(lineage, boundary, context):
        assert ns.owner.is_current and lineage.head is None
        assert boundary.bed is ns.bed
        seen.append((lineage, boundary, context))

    lin, boundary, context = wh.start_wind_boundary(ns.owner, ns.bed, ns.factory, bed_context=ns.ctx,
        limits=wb.LIMITS, available_bytes_fn=wb.AVAIL, before_publish=before)
    assert seen == [(lin, boundary, context)]
    assert lin.head is boundary and not ns.owner.is_current


def test_retirement_failure_does_not_publish_a_second_authority(monkeypatch):
    ns = wb.build()
    made = []
    real_init = lc.FloatingEventLineage.__init__

    def capture(obj):
        real_init(obj)
        made.append(obj)

    def fail(_owner):
        raise MemoryError("injected before actual retirement")

    monkeypatch.setattr(lc.FloatingEventLineage, "__init__", capture)
    monkeypatch.setattr(nfs, "retire_native_owner_to_lineage", fail)
    with pytest.raises(MemoryError):
        wh.start_wind_boundary(ns.owner, ns.bed, ns.factory, bed_context=ns.ctx,
            limits=wb.LIMITS, available_bytes_fn=wb.AVAIL)
    assert ns.owner.is_current
    assert made and made[0].head is None


def test_interrupt_after_the_head_store_finishes_candidate_cleanup(monkeypatch):
    ns = wb.started()
    parent = ns.b
    candidate = wb.stage(ns, wb.child_same(parent.bed))
    interrupted = []

    def cut(obj, name, value):
        object.__setattr__(obj, name, value)
        if obj is ns.lin and name == "_head" and value is not parent and not interrupted:
            interrupted.append(True)
            raise KeyboardInterrupt("injected immediately after native publication")

    monkeypatch.setattr(lc.FloatingEventLineage, "__setattr__", cut, raising=False)
    with pytest.raises(KeyboardInterrupt):
        wh.adopt_wind_handoff(ns.lin, parent, candidate)
    assert ns.lin.head is not parent and candidate.accepted_boundary_ref() is ns.lin.head
    assert candidate.consumed and candidate._cell["parts"] is None


def test_default_adoption_reader_is_the_actual_host_reader(monkeypatch):
    ns = wb.started()
    candidate = wb.stage(ns, wb.child_same(ns.b.bed))
    monkeypatch.setattr(nfs.fsl.sp, "default_host_available_bytes", lambda: 0)
    with pytest.raises(nfs.NativeFloatingMemoryError):
        wh.adopt_wind_handoff(ns.lin, ns.b, candidate)
    assert ns.lin.head is ns.b and not candidate.consumed
