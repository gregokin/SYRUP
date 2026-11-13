"""EB2 GPU definition for Root's hardware host (SYRUP_RELEASE_GPU=1): the CUDA stage is a later slice, so a device event's refresh is a typed refusal before any array or transfer."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SYRUP_RELEASE_GPU") != "1", reason="GPU definitions: set SYRUP_RELEASE_GPU=1 on the hardware qualification host (free device)")
_P = Path(__file__).resolve().parents[1] / "phase2_floating_pond" / "test_gpu_definitions.py"


@pytest.mark.parametrize("windowed", [False, True])
def test_device_event_refresh_is_a_typed_refusal(tmp_path, windowed):
    from fpond_builders import ph

    from maple_syrup import floating_event_refresh as fer
    from maple_syrup.pond_event import PondCompletionPolicy

    spec = importlib.util.spec_from_file_location("e5_gpu_definitions", _P)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    z = ph.basin() - 0.08
    mob = np.zeros((3, 5, 6))
    mob[1, 2, 2] = 1.0e-1
    kw = {"surface": z, "depth": np.maximum(0.135 - z, 0.0), "mobile": mob, "closed": True, "bottom": "closed", "saturation": 0.999999, "splash": False, "rain_mm_h": 0.0, "end_s": 2.0, "max_dt_s": 1.0,
          "policy": PondCompletionPolicy(hold_s=1.0, deposit_mobile_at_flow_end=True)}
    ev, st, _vert, _o = mod.device_event(tmp_path, windowed=windowed, **kw)
    h = ev.settle_retaining_water(ev.evolve(st, 60.0, stop_when_quiet=True).state)
    h = ev.settle_retaining_water(ev.evolve(h.state, 120.0, stop_when_quiet=True).state)
    cap = ev.completion_capability(h, h.state)
    with pytest.raises(fer.EventRefreshError, match="CPU"):
        fer.stage_completed_event_refresh(cap, h.state, event=ev, available_bytes_fn=lambda: 1 << 40)
