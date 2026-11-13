"""EB1 GPU definitions for Root's hardware host (SYRUP_RELEASE_GPU=1): the existing E5 retained thin-lake case on the device; the capability is host metadata only and the device gate precedes everything (no new reader/hash/transfer)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SYRUP_RELEASE_GPU") != "1", reason="GPU definitions: set SYRUP_RELEASE_GPU=1 on the hardware qualification host (free device)")

_P = Path(__file__).resolve().parents[1] / "phase2_floating_pond" / "test_gpu_definitions.py"


@pytest.mark.parametrize("windowed", [False, True])
def test_device_completion_capability_and_bundle(tmp_path, windowed):
    from fpond_builders import ph

    from maple_syrup import floating_completion as fcm
    from maple_syrup.pond_event import PondCompletionPolicy

    spec = importlib.util.spec_from_file_location("e5_gpu_definitions", _P)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    z = ph.basin() - 0.08
    depth = np.maximum(0.135 - z, 0.0)
    mob = np.zeros((3, 5, 6))
    mob[1, 2, 2] = 1.0e-1
    kw = {"surface": z, "depth": depth, "mobile": mob, "closed": True, "bottom": "closed", "saturation": 0.999999, "splash": False, "rain_mm_h": 0.0, "end_s": 2.0, "max_dt_s": 1.0,
          "policy": PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)}
    ev, st, vert, _o = mod.device_event(tmp_path, windowed=windowed, **kw)
    q = ev.evolve(st, 60.0, stop_when_quiet=True)
    h = ev.settle_retaining_water(q.state)
    assert h.status == "requires_recession"
    with pytest.raises(fcm.CompletionError):
        ev.completion_capability(h, h.state)
    res = ev.evolve(h.state, 120.0, stop_when_quiet=True)
    done = ev.settle_retaining_water(res.state)
    assert done.complete
    cap = ev.completion_capability(done, done.state)
    b = vert.event_end_bundle(cap, done.state, ev)
    assert b.payload is vert.current_payload(done.state.hydraulic.soil_water_m) and b.bed is done.state.bed
    fcm.consume_completion(cap, done.state, event=ev)
    with pytest.raises(fcm.CompletionError, match="consumed"):
        ev.completion_capability(done, done.state)
