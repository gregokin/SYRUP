"""Phase 2.5 vegetation lifecycle: ordering of biological generations with a GENUINE accepted LEAF `PlantSession` (the LEAF candidate's own native session fixtures). Older / foreign generations are refused, a newer one and a
repeat of the current one are accepted. The consumer skips (no claim) when the real session fixtures cannot be loaded."""
from __future__ import annotations

import numpy as np
import pytest
import veg_lc_builders as vb
from bridge_builders import AVAIL, dry_boundary

from maple_syrup import vegetation_lifecycle as vl

BIG = vb.BIG


def test_session_generations_are_ordered_older_and_foreign_are_refused(tmp_path, leaf):
    nfx = vb.leaf_fixture(leaf)
    lin, b, _factory = dry_boundary(tmp_path / "a", "numpy", False)
    surf = vb.surface_of(b)
    dx = float(b.routing[0].dx_m)
    mask = np.ones(surf.shape, dtype=bool)
    cat = vb.session_catalog(leaf, nfx, b.root.column.n_layers)
    cfg = nfx.make_config(cat)
    sess0 = nfx.start_session(vb.session_state(leaf, cat, surf, dx), cfg, nfx.anchor(), bound=True)
    ctl = vb.controls(b)
    bd0, p0 = vb.session_binding(leaf, b, sess0, surf, mask, ctl)
    assert p0.identity_record()["source"]["kind"] == "plant_session"
    nb0 = vl.apply_vegetation_generation(lin, b, bd0, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    rec0 = vl.current_record(nb0)
    assert rec0.generation == 1 and rec0.source["sequence"] == sess0.sequence and rec0.source["session_id"] == sess0.session_id
    # the NEXT accepted generation of the SAME session (a published dry day): newer sequence, accepted
    sess1 = nfx.run_day(sess0, nfx.anchor(), 0, t=False, g=False)[0]
    assert sess1.sequence > sess0.sequence and sess1.session_id == sess0.session_id
    bd1, _ = vb.session_binding(leaf, nb0, sess1, surf, mask, ctl)
    nb1 = vl.apply_vegetation_generation(lin, nb0, bd1, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    assert vl.current_record(nb1).generation == 2 and vl.current_record(nb1).source["sequence"] == sess1.sequence
    # the OLDER generation (its binding is still genuine and its surface current) is stale: refused, head untouched
    with pytest.raises(vl.VegetationLifecycleError, match="OLDER biological generation"):
        vl.apply_vegetation_generation(lin, nb1, bd0, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    assert lin.head is nb1
    # the current generation again (same binding): idempotent
    assert vl.apply_vegetation_generation(lin, nb1, bd1, memory_budget_bytes=BIG, available_bytes_fn=AVAIL) is nb1
    # a FOREIGN session (another lineage of plants) is refused
    other = nfx.start_session(vb.session_state(leaf, cat, surf, dx), cfg, nfx.anchor(), bound=True)
    assert other.session_id != sess0.session_id
    bdx, _ = vb.session_binding(leaf, nb1, other, surf, mask, ctl)
    with pytest.raises(vl.VegetationLifecycleError, match="foreign session|PlantSession / lineage"):
        vl.apply_vegetation_generation(lin, nb1, bdx, memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    # an initial-state projection cannot follow an accepted session generation
    s = vb.stack(leaf, nb1, cat=cat)
    with pytest.raises(vl.VegetationLifecycleError, match="initial-state projection cannot follow"):
        vl.apply_vegetation_generation(lin, nb1, vb.binding(leaf, nb1, s, ctl=ctl), memory_budget_bytes=BIG, available_bytes_fn=AVAIL)
    assert lin.head is nb1
