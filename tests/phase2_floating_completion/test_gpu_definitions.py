"""EB1 GPU definitions for Root's hardware host (SYRUP_RELEASE_GPU=1): the existing E5 retained thin-lake case on the device; the capability is host metadata only and the device gate precedes everything (no new reader/hash/transfer)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SYRUP_RELEASE_GPU") != "1", reason="GPU definitions: set SYRUP_RELEASE_GPU=1 on the hardware qualification host (free device)")

_P = Path(__file__).resolve().parents[1] / "phase2_floating_pond" / "test_gpu_definitions.py"


# SR33 (task sr33_default_20261120): the quiet closed-lake fixtures need a strictly unsaturated but machine-near-saturated start. With the 33 kPa field capacity the old 0.999999 leaves a
# residual intake (quiet metric 2.2e-9 at 60 s, never ready); nextafter(1, 0) gives genuine quiet by ~3 s (Root probe root_quiet_fixture_probe.json). Initial condition of the fixture only; not a field recommendation.
QUIET_LAKE_SATURATION = float(np.nextafter(1.0, 0.0))


@pytest.mark.parametrize("windowed", [False, True])
def test_device_completion_capability_and_bundle(tmp_path, windowed):
    from fpond_builders import ph

    from maple_syrup import floating_completion as fcm
    from maple_syrup.pond_event import PondCompletionPolicy

    spec = importlib.util.spec_from_file_location("e5_gpu_definitions", _P)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    z = ph.basin() - 0.08
    z[1, 1] = z[1, 3] = 0.14  # Root-proven isolated-pond certificate geometry (one enclosed wet cell [1, 2]); see phase2_floating_checkpoint/test_terminal.isolated_pond_surface
    depth = np.maximum(0.135 - z, 0.0)
    mob = np.zeros((3, 5, 6))
    mob[1, 2, 2] = 1.0e-1
    kw = {"surface": z, "depth": depth, "mobile": mob, "closed": True, "bottom": "closed", "saturation": QUIET_LAKE_SATURATION, "splash": False, "rain_mm_h": 0.0, "end_s": 2.0, "max_dt_s": 1.0,
          "policy": PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)}
    ev, st, vert, _o = mod.device_event(tmp_path, windowed=windowed, **kw)
    q = ev.evolve(st, 60.0, stop_when_quiet=True)
    h = ev.settle_retaining_water(q.state)
    assert h.status == "requires_recession"
    with pytest.raises(fcm.CompletionError):
        ev.completion_capability(h, h.state)
    res = ev.evolve(h.state, 600.0, stop_when_quiet=True)  # bounded test window (isolated pond quiets at ~125 s)
    assert res.quiet_ready
    done = ev.settle_retaining_water(res.state)
    assert done.complete
    cap = ev.completion_capability(done, done.state)
    b = vert.event_end_bundle(cap, done.state, ev)
    assert b.payload is vert.current_payload(done.state.hydraulic.soil_water_m) and b.bed is done.state.bed
    fcm.consume_completion(cap, done.state, event=ev)
    with pytest.raises(fcm.CompletionError, match="consumed"):
        ev.completion_capability(done, done.state)
