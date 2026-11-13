"""S4 CUDA fresh-process worker (WRITTEN, NOT RUN by the author: no GPU launch/probe; Root guards GPU0 and runs it). The SAME scenario/stage machinery as the CPU worker (`s4_worker`), on the
GENUINE native9f CUDA file factory (`gcases.build_cuda_case`, flat closed owned fixture, default quiet/completion rules) and the CUDA V3 codec: `full` runs every step saving a V3 checkpoint
after each stage; `resume <stage>` rebuilds the fixed components from the genuine CUDA factories in a FRESH process, loads that stage's checkpoint (with the expected plan/StormSpec) and continues.
Digests are `physical_digest_cuda` + the cursor digest. Usage: s4_cuda_worker.py <storm|recharge|fallback> <full|resume> <workdir> <out.json> [stage]. Needs SYRUP_NCP_CUDA_TARGET and the CUDA bundle
on PYTHONPATH (see `gcases`)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import gcases as g
import s4_worker as w
from maple_syrup import layered_checkpoint as lc
from maple_syrup import lifecycle_checkpoint as lk
from maple_syrup import lifecycle_checkpoint_cuda as cs
from maple_syrup import lifecycle_checkpoint_cuda_v3 as v3c
from maple_syrup import water_lifecycle as wl
from maple_syrup.rainfall import rainfall_field

LIMITS = lc.CheckpointLimits(1 << 30, 1 << 30, 1 << 30, 1 << 30)
RES = dict(caller_host_reserve_bytes=1 << 20, caller_device_reserve_bytes=1 << 20)


class CudaRun(w.Run):
    def __init__(self, workdir: Path, scenario: str):
        self.scenario = scenario
        self.backend = "cuda"
        self.work = workdir
        theta, bottom = (0.5, "closed") if scenario == "fallback" else (0.2, "open")
        self.c = g.build_cuda_case(workdir / "case", g.cuda_yaml(2, bottom=bottom, theta=theta), flat_closed=True)
        self.lop, self.state = self.c.lop, self.c.s0
        self.cp = self.c.cp
        self.trace, self.plan, self.origin = {}, None, None
        assert self.lop.backend == "cuda"

    def lop_backend(self):
        return self.lop.backend

    def pet(self, rate=w.PET):
        return self.cp.full(self.c.n, float(rate), dtype=np.float64)  # strictly device float64

    def spec_for(self, plan, origin):
        f = plan.bind_forcing(origin)
        ny, nx = self.lop.shape
        field = rainfall_field(ny, nx, active_mask=self.lop.column.active.reshape(ny, nx), xp=self.lop.xp)  # device-resident mask VIEW, the lifecycle's namespace
        return lk.StormSpec(f.schedule, field, float(origin), w.CONTROL, w.POLICY)

    def digest(self, state):
        return hashlib.sha256((cs.physical_digest_cuda(self.lop, state) + state.daily.digest()).encode()).hexdigest()

    def mark(self, stage, state=None):
        s = self.state if state is None else state
        self.trace[stage] = {"v3": self.digest(s), "state": s.digest, "t_s": s.t_s, "revision": s.revision, "role": s.role, "ordinal": s.daily.ordinal,
                             "pending": s.daily.pending is not None, "seq": s.request_sequence, "rain_m3": s.lifetime.rain_m3, "soil_e": s.lifetime.soil_evaporation_m3}

    def save(self, stage):
        v3c.save_lifecycle_checkpoint_cuda_v3(self.work / f"ck_{stage}", self.lop, self.state, w.IDENT, limits=LIMITS, **RES)

    def load(self, stage):
        role, kind, _storm, _ = w.STAGES[self.scenario][stage]
        spec = plan = None
        if role == "storm":
            origin, p = w.ORIGINS[self.scenario][stage]
            plan = w.rc.plan_daily_rain(*p[0], run_digest=w.RUN, ordinal=p[1])
            spec = self.spec_for(plan, origin)
        out = v3c.load_lifecycle_checkpoint_cuda_v3(self.work / f"ck_{stage}", self.lop, w.IDENT, expected_role=role, expected_source_kind=kind, storm=spec, expected_plan=plan,
                                                    limits=LIMITS, reference_bed=self.c.s0.payload.bed, **RES)
        self.lop, self.state = out.lop, out.state
        self.plan, self.origin = plan, None if spec is None else spec.event_origin_s

    def step_t(self):
        q = self.cp.zeros((self.lop.n_layers, self.lop.n_cells))
        q[0] = w.Q
        self.state, _ = wl.request_withdrawal(self.lop, self.state, expected_parent=wl.expect_parent(self.state), depth_m=q, identity="leaf-s4")

    def soil_sha(self):
        return hashlib.sha256(np.ascontiguousarray(g.host(self.cp, wl._arrays_of(self.state).soil.water_m)).tobytes()).hexdigest()


if __name__ == "__main__":
    w.main(CudaRun)
