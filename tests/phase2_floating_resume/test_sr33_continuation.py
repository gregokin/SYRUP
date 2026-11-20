"""NEW (task sr33_default_20261120): physical save/load/advance positive control under the SR33 default. Actual native owner/config, real floating Pond event with preferential flow and erosion,
capture -> restore -> continue bitwise equal, water/sediment audits close. This proves a REAL wet checkpoint capture/restore/advance only; it does NOT prove full-storm completion or genuine quiet/settlement (that is covered separately by the closed-lake fixtures in phase2_floating_checkpoint/completion)."""
from __future__ import annotations

import numpy as np
from fpond_builders import make
from test_floating_resume import roundtrip

from maple_syrup import soil_texture_properties as stp


def test_default_33_native_owner_capture_restore_and_continue_close_and_are_bitwise_equal():
    _ev, _st, _bed, _ctx, vert, owner = make(backend="numpy", splash=True, depth=0.004, preferential=True)
    assert owner.config.field_capacity_method == stp.METHOD_SR33
    fc = np.asarray(owner.column._maps["theta_field_capacity"])
    sat = np.asarray(owner.column._maps["theta_saturation"])
    assert np.all(fc < sat) and np.all(fc < 0.45)
    _ev, _st, vert, _ev2, _r, _vert2, _cap = roundtrip({"splash": True, "depth": 0.004, "preferential": True}, 12, 6, backend="numpy")
    assert vert.stats.depth_changed_columns > 0
