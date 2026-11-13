"""R2a: a state whose current canonical depth D EXCEEDS the original D0 (a column thickened by an actual native deposition) is saved and restored through the official geometry view of the held root column; the held maps stay the
original ones and the continuation is bit-exact. LIMIT (reported): a state whose W exceeds the ORIGINAL D0 capacity could not be produced by an ordinary Pond fixture here (the added room did not fill within hundreds of steps), and
a state after `settle_retaining_water` is refused by the historical scalar counter check (`n_native_calls`), so the W > D0-capacity behaviour is covered by the R1a prepare controls (continued state on the CURRENT geometry), not by
a file-level fixture."""
from __future__ import annotations

import fpond_builders as fb
import numpy as np
from case import AVAIL, IDENT, snapshot
from fpond_builders import ph

from maple_syrup import floating_checkpoint as fc


def thick_case():
    z = ph.plane(base=0.12, drop=0.004)
    mob = np.zeros((3, 4, 6))
    mob[:, 0, :] = 0.05  # a mobile pool that deposits downstream in the first steps (actual wet transport + native deposition)
    return fb.make(surface=z, bottom="closed", saturation=0.97, depth=0.004, mobile=mob, splash=False, max_dt_s=0.2)


def test_a_thickened_column_round_trips_through_the_official_view_and_continues_exactly(tmp_path):
    ev, st, _bed, _ctx, vert, _o = thick_case()
    for _ in range(25):
        st = ev.step(st, 300 / 3.6e6, 0.2).state
    p, t = vert.current_payload(st.hydraulic.soil_water_m), vert.held_template()
    assert (p.depth_m > t.depth0_m).any() and p.column.identity_kind == "geometry_view_token"  # real deposition thickened some columns (D > D0)
    ck = tmp_path / "ck"
    fc.save_floating_pond_checkpoint(ck, st, ev, IDENT, available_bytes_fn=AVAIL)
    ref = ev.step(st, 300 / 3.6e6, 0.2).state
    ev2, _s0, _b, _c, vert2, _o2 = thick_case()
    got = fc.load_floating_pond_checkpoint(ck, ev2, IDENT, available_bytes_fn=AVAIL)
    p2 = vert2.current_payload(got.hydraulic.soil_water_m)
    assert p2.column.identity_kind == "geometry_view_token" and np.array_equal(p2.depth_m, p.depth_m)
    assert np.array_equal(np.asarray(vert2.held_template().column._maps["theta_field_capacity"]), np.asarray(t.column._maps["theta_field_capacity"]))  # the held maps stay the ORIGINAL ones
    cont = ev2.step(got, 300 / 3.6e6, 0.2).state
    a, b = snapshot(ev, ref, vert), snapshot(ev2, cont, vert2)
    assert set(a) == set(b) and all(a[k].tobytes() == b[k].tobytes() for k in a)


def test_a_state_after_retained_terminal_settlement_is_refused_by_the_historical_codec_and_saved_by_the_extended_profile(tmp_path):
    """S4-1 supersedes the R2a pin: the HISTORICAL scalar rule (`n_native_calls == 2*steps + 2*splash`) still refuses a retained-settled state (unchanged), while the floating profile /3 saves it through the producer-maintained
    terminal count (round trip and continuation: test_terminal.py)."""
    import pytest

    from maple_syrup import pond_checkpoint as pc
    from maple_syrup.pond_event import PondCompletionPolicy

    z = ph.basin() - 0.08
    mob = np.zeros((3, 5, 6))
    mob[1, 2, 2] = 1.0e-1
    ev, st, _bed, _ctx, _vert, _o = fb.make(surface=z, closed=True, bottom="closed", saturation=0.999999, depth=np.maximum(0.135 - z, 0.0), mobile=mob,
                                            policy=PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True), max_dt_s=1.0, end_s=2.0, rain_mm_h=0.0, splash=False)
    settled = ev.settle_retaining_water(ev.evolve(st, 60.0, stop_when_quiet=True).state).state
    with pytest.raises(pc.PondCheckpointError):
        pc.save_pond_checkpoint(tmp_path / "old", settled, ev, IDENT)
    assert not (tmp_path / "old").exists()
    fc.save_floating_pond_checkpoint(tmp_path / "ck", settled, ev, IDENT, available_bytes_fn=AVAIL)
    assert (tmp_path / "ck" / "floating_pond_checkpoint.json").is_file()
