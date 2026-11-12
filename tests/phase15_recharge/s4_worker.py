"""S4 fresh-process worker (a script, not a test module): runs ONE daily scenario on the genuine flat closed native file case and either (a) `full`: executes every step,
saving a V3 checkpoint after each checkpoint stage, or (b) `resume <stage>`: in a FRESH process rebuilds the fixed components from the genuine factories, loads that stage's V3
checkpoint (with the expected plan/StormSpec) and continues the remaining steps. Every stage emits the V3 physical digest and key scalars as JSON so the parent can compare the
uninterrupted and the restarted trajectories. Usage: s4_worker.py <scenario> <full|resume> <workdir> <out.json> [stage]."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import ncpcases as nc  # noqa: E402
from dcases import DAY, RUN, STORM_CONTROL, dw, ev, rc, wl  # noqa: E402,F401
from maple_syrup import layered_checkpoint as lc  # noqa: E402
from maple_syrup import lifecycle_checkpoint as lk  # noqa: E402
from maple_syrup import lifecycle_checkpoint_v3 as v3  # noqa: E402
from maple_syrup.rainfall import rainfall_field  # noqa: E402

CONTROL = ev.LayeredEventControl()  # defaults
POLICY = ev.LayeredQuietPolicy()  # defaults
IDENT = {"case": "s4-daily", "scenario": "fixed"}
LIMITS = lc.CheckpointLimits(1 << 30, 1 << 30, 0, 1 << 30)
PET = 1e-8
P_STORM = (1.0, 360.0, 0.5)  # P=1 mm, I=360 mm/h, T=0.5 mm -> D = 10 s, a full storm
Q = 1e-5


class Run:
    def __init__(self, workdir: Path, scenario: str):
        self.scenario = scenario.split("_")[0]  # `storm_numba` runs the `storm` scenario on the COMPILED Numba backend (the others use the NumPy reference backend)
        self.backend = "numba" if scenario.endswith("_numba") else "numpy"
        self.work = workdir
        theta, bottom = (0.5, "closed") if self.scenario == "fallback" else (0.2, "open")
        raw = nc.yaml_for(2, bottom=bottom, theta=theta)
        raw["compute"]["backend"] = self.backend
        self.c = nc.build_file_case(workdir / "case", raw, flat_closed=True)
        assert self.lop_backend() == self.backend
        self.lop, self.state = self.c.lop, self.c.s0
        self.trace: dict = {}
        self.plan = None
        self.origin = None

    def lop_backend(self):
        return self.c.lop.backend

    def soil_sha(self):
        return __import__("hashlib").sha256(np.ascontiguousarray(wl._arrays_of(self.state).soil.water_m).tobytes()).hexdigest()

    # -- helpers ------------------------------------------------------------------------------------------------------------------
    def pet(self, rate=PET):
        return np.full(self.c.n, rate)

    def spec_for(self, plan, origin):
        f = plan.bind_forcing(origin)
        ny, nx = self.lop.shape
        field = rainfall_field(ny, nx, active_mask=np.asarray(self.lop.topology.active).reshape(ny, nx), xp=np)
        return lk.StormSpec(f.schedule, field, float(origin), CONTROL, POLICY)

    def mark(self, stage, state=None):
        s = self.state if state is None else state
        self.trace[stage] = {"v3": v3.physical_digest_v3(self.lop, s), "state": s.digest, "t_s": s.t_s, "revision": s.revision, "role": s.role,
                             "ordinal": s.daily.ordinal, "pending": s.daily.pending is not None, "seq": s.request_sequence,
                             "rain_m3": s.lifetime.rain_m3, "soil_e": s.lifetime.soil_evaporation_m3}

    def save(self, stage):
        v3.save_lifecycle_checkpoint_v3(self.work / f"ck_{stage}", self.lop, self.state, IDENT, limits=LIMITS, host_available_bytes_fn=lambda: 1 << 40)

    def load(self, stage):
        role, kind, storm, plan = STAGES[self.scenario][stage]
        spec = plan_spec = None
        if role == "storm":
            origin, p = ORIGINS[self.scenario][stage]
            plan_spec = rc.plan_daily_rain(*p[0], run_digest=RUN, ordinal=p[1])
            spec = self.spec_for(plan_spec, origin)
        out = v3.load_lifecycle_checkpoint_v3(self.work / f"ck_{stage}", self.lop, IDENT, expected_role=role, limits=LIMITS, host_available_bytes_fn=lambda: 1 << 40,
                                              reference_bed=self.c.s0.payload.bed, storm=spec, expected_source_kind=kind, expected_plan=plan_spec)
        self.lop, self.state = out.lop, out.state
        self.plan, self.origin = plan_spec, None if spec is None else spec.event_origin_s

    # -- steps --------------------------------------------------------------------------------------------------------------------
    def step_start(self):
        self.state = dw.start_daily_lineage(self.lop, self.state, expected_parent=wl.expect_parent(self.state), run_digest=RUN)

    def step_p0day_close(self):
        plan = dw.next_plan(self.state, 0.0, 0.0, 0.5)
        r = dw.daily_water_action(self.lop, self.state, expected_parent=wl.expect_parent(self.state), plan=plan, pet_terrain_m_s=self.pet())
        self.state = wl.close_interevent(self.lop, r.state, expected_parent=wl.expect_parent(r.state))

    def step_recharge_day(self):
        plan = dw.next_plan(self.state, 0.25, 360.0, 0.5)
        r = dw.daily_water_action(self.lop, self.state, expected_parent=wl.expect_parent(self.state), plan=plan, pet_terrain_m_s=self.pet())
        assert r.status == "complete"
        self.state = r.state

    def step_begin_storm(self):
        plan = dw.next_plan(self.state, *P_STORM)
        self.plan, self.origin = plan, self.state.t_s
        self.state = dw.begin_daily_storm(self.lop, self.state, expected_parent=wl.expect_parent(self.state), plan=plan, control=CONTROL, policy=POLICY)

    def step_begin_fallback(self):
        plan = dw.next_plan(self.state, 1.0, 360.0, 2.0)  # P <= T: the recharge attempt on a saturated closed column fails (capacity) and falls back to the FULL storm
        r = dw.daily_water_action(self.lop, self.state, expected_parent=wl.expect_parent(self.state), plan=plan, pet_terrain_m_s=self.pet())
        assert r.status == "capacity_failure"
        self.plan, self.origin = plan, self.state.t_s
        self.state = dw.begin_daily_storm(self.lop, self.state, expected_parent=wl.expect_parent(self.state), plan=plan, fallback_of=r.fallback, control=CONTROL, policy=POLICY)

    def step_partial(self):
        res = wl.evolve_storm(self.lop, self.state, expected_parent=wl.expect_parent(self.state), until_s=self.state.t_s + 40_000.0, max_accepted_steps=8)
        assert not res.complete
        self.state = res.state

    def step_complete(self):
        res = wl.evolve_storm(self.lop, self.state, expected_parent=wl.expect_parent(self.state), until_s=self.state.t_s + 40_000.0)
        for _ in range(12):
            if res.complete:
                break
            res = wl.evolve_storm(self.lop, res.state, expected_parent=wl.expect_parent(res.state), until_s=res.state.t_s + 40_000.0)
        assert res.complete
        self.state = res.state

    def step_finish_day(self):
        # the fallback storm leaves a ~1 mm pond on a saturated closed column: a larger PET empties it before the ONE day router step (otherwise the day CFL-refuses)
        r = dw.finish_daily_storm(self.lop, self.state, expected_parent=wl.expect_parent(self.state), plan=self.plan, pet_terrain_m_s=self.pet(1e-7 if self.scenario == "fallback" else PET))
        self.state = r.state

    def step_t(self):
        q = np.zeros((self.lop.n_layers, self.lop.n_cells))
        q[0] = Q
        self.state, _ = wl.request_withdrawal(self.lop, self.state, expected_parent=wl.expect_parent(self.state), depth_m=q, identity="leaf-s4")

    def step_close(self):
        self.state = wl.close_interevent(self.lop, self.state, expected_parent=wl.expect_parent(self.state))

    def step_later_storm(self):
        self.step_begin_storm()
        self.mark("later_begin")
        self.step_complete()


# stage -> (role, expected source kind, storm?, _); ORIGINS: where the storm of a storm-role stage began and its plan (P, I, T), ordinal
SEQ = {
    "storm": [("B0", "start"), ("B1", "p0day_close"), ("PS", "begin_storm+partial"), ("CS", "complete"), ("D", "finish_day"), ("T", "t"), ("E", "close"), ("L", "later_storm")],
    "recharge": [("B0", "start"), ("R", "recharge_day"), ("T", "t"), ("E", "close"), ("L", "later_storm")],
    "fallback": [("B0", "start"), ("B1", "p0day_close"), ("PS", "begin_fallback+partial"), ("CS", "complete"), ("D", "finish_day")],
}
STAGES = {
    "storm": {"B0": ("boundary", None, None, None), "B1": ("boundary", None, None, None), "PS": ("storm", None, 1, None), "CS": ("storm", None, 1, None),
              "D": ("interevent", "storm_end", None, None), "T": ("interevent", "storm_end", None, None), "E": ("boundary", None, None, None)},
    "recharge": {"B0": ("boundary", None, None, None), "R": ("interevent", "boundary_origin", None, None), "T": ("interevent", "boundary_origin", None, None),
                 "E": ("boundary", None, None, None)},
    "fallback": {"B0": ("boundary", None, None, None), "B1": ("boundary", None, None, None), "PS": ("storm", None, 1, None), "CS": ("storm", None, 1, None)},
}
ORIGINS = {"storm": {"PS": (86400.0, (P_STORM, 1)), "CS": (86400.0, (P_STORM, 1))},
           "fallback": {"PS": (86400.0, ((1.0, 360.0, 2.0), 1)), "CS": (86400.0, ((1.0, 360.0, 2.0), 1))}}
ACTIONS = {"start": "step_start", "p0day_close": "step_p0day_close", "recharge_day": "step_recharge_day", "t": "step_t", "close": "step_close", "complete": "step_complete",
           "finish_day": "step_finish_day", "later_storm": "step_later_storm"}


def run_step(run: Run, action: str) -> None:
    if action == "begin_storm+partial":
        run.step_begin_storm()
        run.step_partial()
    elif action == "begin_fallback+partial":
        run.step_begin_fallback()
        run.step_partial()
    else:
        getattr(run, ACTIONS[action])()


def main(run_cls: type = Run) -> None:
    """`run_cls` lets the CUDA worker (`s4_cuda_worker.py`) reuse this exact scenario/step/stage machinery with device arrays and the CUDA V3 codec."""
    scenario, mode, work, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    run = run_cls(work, scenario)
    scenario = run.scenario
    seq = SEQ[scenario]
    start = 0
    if mode == "resume":
        stage = sys.argv[5]
        run.load(stage)
        start = [s for s, _ in seq].index(stage) + 1
        run.mark(stage)  # the restored state's own digest (compared with the saved one by the parent)
    else:
        start = 0
    for name, action in seq[start:]:
        run_step(run, action)
        run.mark(name)
        if mode == "full" and name in STAGES[scenario]:
            run.save(name)
    run.trace["_final_soil_sha"] = run.soil_sha()
    out.write_text(json.dumps(run.trace, sort_keys=True, indent=1))


if __name__ == "__main__":
    main()
