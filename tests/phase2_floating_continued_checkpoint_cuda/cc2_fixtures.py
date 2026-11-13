"""Device-free fixtures of the CC2 suites: the EXACT deep-pair lake (E5 `device_event` / `fb.make_pair` geometry) with an ABOVE-threshold terminal deposit, and the genuine matched CPU generation-1 control.

Why not `refresh_fixtures.lake_kwargs()`: its 0.1 kg terminal deposit raises the bed by ~0.00032 m (0.1 kg / (0.25 m^2 * 1250 kg/m^3)), BELOW the 0.00075 m net-elevation trigger, so no genuine reroute happens (generation 0). 1 kg
raises it by ~0.0032 m (above the trigger) and the CPU CC1 suite already uses 1 kg. The declared COMPONENT policy is the CPU CC1 one (hold 1 s, max depth change 1e-4 m, max unit discharge 1e-3 m^2/s; the strict default would need a
long recession): it is a per-event option, never a global default, and quiet/completion/generation are all genuine native results."""
from __future__ import annotations

import numpy as np
from fpond_builders import ph

from maple_syrup.pond_event import PondCompletionPolicy

MOBILE_KG = 1.0
THRESHOLD_M = 7.5e-4


def lake_kwargs() -> dict:
    """Public kwargs accepted by BOTH `e5.device_event` and `fpond_builders.make_pair` (the same deep pair geometry/forcing/mobile pool/policy)."""
    z = ph.basin() - 0.08
    mob = np.zeros((3, 5, 6))
    mob[1, 2, 2] = MOBILE_KG
    return {"surface": z, "depth": np.maximum(0.135 - z, 0.0), "closed": True, "bottom": "closed", "saturation": 0.999999, "splash": False, "rain_mm_h": 0.0, "end_s": 2.0, "max_dt_s": 1.0, "mobile": mob,
            "policy": PondCompletionPolicy(hold_s=1.0, max_depth_change_m=1e-4, max_unit_discharge_m2_s=1e-3, deposit_mobile_at_flow_end=True)}


def deposit_rise_m(geometry) -> float:
    """The bed rise of the terminal deposit in its cell: mass / (cell area * bulk density) -- the independent expectation the fixture must exceed the trigger with."""
    return MOBILE_KG / (geometry.dx_m * geometry.dy_m * geometry.bulk_density_kg_m3)


def make_factory(ev0):
    from maple_syrup.pond_event import PondEvent

    def factory(vert, topology=None):
        return PondEvent(ev0.context, None, ev0.sediment, ev0.schedule, ev0._initial_topology if topology is None else topology, control=ev0.control, policy=ev0.policy, splash=ev0.splash, floating_vertical=vert)

    return factory


def settle(ev, st, max_rounds=8):
    """Real quiet + retained settlement loop until a genuinely COMPLETE handoff (the bounded loop of the CPU lifecycle tests; also valid on a device event)."""
    state = st
    for _ in range(max_rounds):
        q = ev.evolve(state, float(state.t_s) + 200.0, stop_when_quiet=True)
        state = q.state
        if q.quiet_ready:
            h = ev.settle_retaining_water(state)
            state = h.state
            if h.complete:
                return h
    raise AssertionError("the storm did not reach a genuine complete handoff in the bounded loop")


def recipe_digest() -> str:
    """Source-binds the static recipe: SHA-256 over the kwargs' arrays/scalars and the declared policy (the CPU control and the GPU fixture must report the same digest)."""
    import hashlib

    kw = lake_kwargs()
    h = hashlib.sha256()
    for k in sorted(kw):
        v = kw[k]
        h.update(k.encode())
        h.update(np.ascontiguousarray(v).tobytes() if isinstance(v, np.ndarray) else repr(v).encode())
    return h.hexdigest()


def cpu_control(tmp, windowed: bool, backend: str = "numpy"):
    """The GENUINE matched CPU generation-1 control on the EXACT deep pair (`fb.make_pair` Dense/Windowed, same kwargs as `e5.device_event`): open -> real native completion -> selected refresh boundary -> `begin_occurrence`.
    Returns a namespace with the continued state, event, factory, original topology, boundary and the pair's initial `(event, state, vertical)` for `refresh_fixtures.assert_matched`."""
    from pathlib import Path
    from types import SimpleNamespace

    import fpond_builders as fb
    from case import AVAIL

    from maple_syrup import floating_event_lifecycle as lc

    ev0, st0, vert0, owner, bed, _c = fb.make_pair(Path(tmp), backend=backend, **lake_kwargs())[1 if windowed else 0]
    factory = make_factory(ev0)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL)
    from maple_syrup import floating_held_context as fh

    record0 = fh.export_held_context(occ.vertical, None, available_bytes_fn=AVAIL)  # the ORIGINAL-root H0 record (before any refresh), for the SAVE-side recipe verification
    h = settle(occ.event, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    record = lc.export_boundary_held_context(b, available_bytes_fn=AVAIL)
    occ2, s2 = lc.begin_occurrence(lin, b, factory, available_bytes_fn=AVAIL)
    return SimpleNamespace(lin=lin, occ=occ2, event=occ2.event, state=s2, boundary=b, record=record, record0=record0, factory=factory, ev0=ev0, st0=st0, vert0=vert0, original=ev0._initial_topology, windowed=windowed,
                           backend=backend)
