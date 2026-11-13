"""HOST CONTROL of the seven-day N=3 chain driver (`chain.py`) on the matched CPU twin through the CPU outer: it validates the sequence logic the GPU definition drives (days, due frames only on closed days 3 and 6 after the real
intervening event + property refresh, a save at a closed day before the due day, a restore into a FRESH lineage, writer resume, equality with the uninterrupted run). It is NOT a device test."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import chain
import numpy as np
import pytest

from maple_syrup import floating_lifecycle_checkpoint as cpul

_T = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bridge_gpu_defs_for_chain_host", _T / "phase2_floating_event_daily_bridge_cuda" / "test_gpu_definitions.py")
bd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bd)
IDENT = {"case": "s2host", "v": 1}


class CpuSide(bd.Side):
    def complete(self, occ, s):
        return bd.run_to_complete(occ.event, s)


@pytest.mark.parametrize("windowed", [False, True], ids=["dense", "windowed"])
def test_chain_pause_before_the_first_due_day_restores_into_a_fresh_lineage_and_equals_the_uninterrupted_run(tmp_path, windowed):
    def side(tag):
        return CpuSide(tmp_path / tag, bd.dry_kwargs(splash=True), False, windowed)

    host = np.asarray
    mp = pytest.MonkeyPatch()
    try:
        ref_side = side("ref")
        ref = chain.run(ref_side, tmp_path / "out_ref", mp, host, save=lambda lin, closed: None, load=lambda sv: None)
        s2 = side("run")
        original = s2.ev._initial_topology

        def save(lin, closed):
            return cpul.save_floating_lifecycle_checkpoint(tmp_path / "ck", lin, s2.factory, IDENT, closed=closed, original_topology=original, available_bytes_fn=s2.rdr)

        def load(sv):
            return cpul.load_floating_lifecycle_checkpoint(sv.path, s2.factory, IDENT, sv.manifest_sha256, original_topology=original, partition=chain.PART, available_bytes_fn=s2.rdr)

        got = chain.run(s2, tmp_path / "out_run", mp, host, save=save, load=load, pause_after=1)
    finally:
        mp.undo()
    assert got.restored is not None and got.restored.kind == "closed_day"
    chain.check_equal(ref, got)
