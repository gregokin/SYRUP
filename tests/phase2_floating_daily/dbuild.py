"""Builders of the G3 S1/S2 suites (author-written): real native beds/owners (E3 fixtures), real recharge plans, real columns; no fake classes."""
from __future__ import annotations

import builders as b
import numpy as np

from maple_syrup import floating_daily as fd
from maple_syrup import native_floating_soil as nfs
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se

RUN = "ab" * 32
PART = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
BIG = 10**12
AVAIL = lambda: BIG


def make(*, backend="numpy", saturation=0.5, layers=b.LAYERS, bottom="open", bed=None, **cfg):
    bed = b.stratified() if bed is None else bed
    cfg_ = b.config(backend=backend, layers=layers, bottom=bottom, **cfg)
    owner = nfs.initialize_native_floating_soil(bed, cfg_, initial_relative_saturation=saturation, available_bytes_fn=AVAIL)
    return owner, bed


def start(owner, bed, run=RUN, partition=PART):
    return fd.start_floating_daily_lineage(owner, bed, partition=partition, run_digest=run, available_bytes_fn=AVAIL)


def lineage(run=RUN, **kw):
    owner, bed = make(**kw)
    st = start(owner, bed, run=run)
    n = st.ctx.n_cells
    return st, bed, owner, np.full(n, 3e-8), np.full(n, 0.3)


def plan(p, i, t, ordinal, run=RUN):
    return rc.plan_daily_rain(p, i, t, run_digest=run, ordinal=ordinal)


def day(st, p=0.0, i=0.0, t=5.0, pet=None, cover=None, request=None):
    """One completed no-rain/recharge day; returns the interevent result (and the closed state after an optional request)."""
    n = st.ctx.n_cells
    pet = np.full(n, 3e-8) if pet is None else pet
    cover = np.full(n, 0.3) if cover is None else cover
    return fd.floating_daily_action(st, expected_parent=fd.expect_parent(st), plan=plan(p, i, t, st.cursor.ordinal, run=st.cursor.run_digest), pet_terrain_m_s=pet, cover=cover, available_bytes_fn=AVAIL)


def close(st, **kw):
    return fd.close_floating_day(st, expected_parent=fd.expect_parent(st), **kw)


def full_day(st, **kw):
    r = day(st, **kw)
    assert r.outcome == "interevent", r
    return close(r.state).state
