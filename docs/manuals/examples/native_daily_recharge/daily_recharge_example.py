#!/usr/bin/env python
"""Native Phase 1.5 daily-recharge demonstration (a NATIVE PYTHON API example, not a SYRUP command-line feature).

It runs ONE daily-lineage scenario on a genuine native MAPLE 9f Dense case through the ordinary factories (no test helpers, no copied hydrology):

    verify the qualified source pair -> admit the package case -> prepare the native Dense bed (MAPLE initializes the landscape)
    -> load the layered-water YAML -> prepare the layered lifecycle -> start a DAILY lineage -> one occurrence (recharge, P = 0 day,
    or a full native storm) -> optional zero-time T request, close, V3 physical checkpoint/restore, and a later full storm.

Everything that is a *scientific input* belongs to the native case and the layered-water file. The daily descriptor (P in mm, intensity I in mm/h,
amount threshold T in mm) is a caller argument of the Python API: P/I/T below are ILLUSTRATIVE TEST VALUES, T has no recommended scientific default,
and the example PET rate is an illustrative number, not calibrated weather. Units: depths mm (descriptor) and m (state); intensity mm/h; PET m/s;
clock seconds; volumes m3 over the ACTIVE cells.

Source selection is explicit: the SYRUP package must be the qualified Phase 1.5 package below and MAPLE the historical 9f bundle you point at. The historical
C1 dated-scheduler recipes are NOT touched by this example and still require their C1 code.

Usage (CPU, existing MAPLE environment; run with the bundle's editable and source/src plus this SYRUP source tree on PYTHONPATH):

    python daily_recharge_example.py --bundle <materialized-9f-bundle> --scenario recharge --workdir /tmp/daily_demo_NEW
    python daily_recharge_example.py --bundle <bundle> --scenario fallback --checkpoint --later-storm --workdir /tmp/daily_fb_NEW --backend numba

The work directory (and any --checkpoint-dir) must NOT exist: the script creates them and never deletes, replaces or rewrites existing files; it refuses before its first write.

`--backend cuda` uses the genuine CUDA factories and the CUDA V3 codec against a materialized CUDA 9f bundle on the current device (written for, but not run
by, the example author: a CUDA run needs the CUDA bundle and a free GPU). Prints one compact JSON object on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

QUALIFIED_SYRUP_PACKAGE_SHA256 = "82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107"  # the Phase 1.5 (N15) qualified package; C1 afcad536... is a different, historical source
HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parent / "native_leaf_calendar_cli"  # the public native 9f case + layered_water.yaml (copied, never edited)
RUN_DIGEST = "ab" * 32  # run identity of this demonstration (a caller-chosen 64-hex string; run identity + ordinal, not a date, is the occurrence uniqueness)
SCENARIOS = {
    # name: (bottom, initial theta per layer, flat closed domain, descriptor (P mm, I mm/h, T mm), meaning)
    "recharge": ("open", 0.20, False, (2.0, 10.0, 5.0), "0 < P <= T: top uniform pulse; below field capacity nothing moves, the day's E and diffusion run once"),
    "open_drainage": ("open", 0.30, False, (2.0, 10.0, 5.0), "open bottom, profile above field capacity: gravity moves water down once and the bottom drains it (drainage reported)"),
    "closed_retention": ("closed", 0.25, False, (2.0, 10.0, 5.0), "closed bottom at field capacity: the pulse is retained by the capacity-limited cascade up to saturation (no drainage)"),
    "fallback": ("closed", 0.50, True, (1.0, 360.0, 2.0), "closed saturated column: the pulse cannot fit, the WHOLE trial is discarded and the FULL P is run as a native storm (flat closed owned case)"),
}


# Illustrative explicit PET (m/s) per scenario. The fallback storm leaves about 1 mm of retained water on the saturated closed flat column; the ONE daily
# router step refuses (typed `router_step_rejected`, state unchanged, CFL 0.5 is never loosened and no substeps are invented) unless the illustrative PET
# (not calibrated weather) first empties that pond, so the fallback default is 1e-7 m/s.  Pass --pet-m-s 1e-8 to see the truthful refusal instead.
DEFAULT_PET = {"recharge": 1e-8, "open_drainage": 1e-8, "closed_retention": 1e-8, "fallback": 1e-7}


def build_yaml(scenario, backend):
    bottom, theta, _flat, _desc, _ = SCENARIOS[scenario]
    raw = yaml.safe_load((PUBLIC / "layered_water.yaml").read_text())
    raw["compute"]["backend"] = backend
    raw["soil"]["bottom"] = bottom
    raw["initial"]["soil_moisture"]["by_layer"] = [theta] * len(raw["soil"]["layers"]["thickness_m"])
    return raw


def prepare_case(args, workdir):
    """The genuine factory chain (CPU or CUDA); returns (lifecycle operator, accepted boundary state, reference bed, helpers)."""
    flat = SCENARIOS[args.scenario][2]
    case_dir = workdir / "case"
    if True:  # the work directory is fresh (checked by check_fresh_paths), so these are always new files
        shutil.copytree(PUBLIC / "case", case_dir)  # an OWNED copy of the public native case input
        if flat:  # explicit owned test input: native flat terrain (relief 0) and a closed surface domain (no outlets) so the full storm completes quickly
            case_yaml = case_dir / "case.yaml"
            case = yaml.safe_load(case_yaml.read_text())
            case["topography"]["perturbation"]["relief_m"] = 0.0
            case_yaml.write_text(yaml.safe_dump(case))
    raw = build_yaml(args.scenario, "cuda" if args.backend == "cuda" else args.backend)
    if flat:
        raw["domain"]["outlets"] = []
    layered = workdir / "layered_water.yaml"
    layered.write_text(yaml.safe_dump(raw))
    from maple_syrup import layered_case as lc
    from maple_syrup import native_case_profile as ncp

    mib = 1 << 20
    if args.backend == "cuda":
        from maple_syrup import layered_case_cuda as lcc
        from maple_syrup import native_dense_cuda_profile as ndc

        profile = ncp.verify_source_profile(args.bundle, expected_syrup_digest=QUALIFIED_SYRUP_PACKAGE_SHA256, profile=ncp.CUDA_SOURCE_PROFILE_NAME)
        admission = ncp.admit_package_case(profile, case_dir)
        prep = ndc.prepare_native_dense_cuda_case(admission, device_id=args.device, caller_host_reserve_bytes=mib, caller_device_reserve_bytes=mib)
        setup = lcc.prepare_layered_cuda_case(prep, lc.load_layered_cuda_config(layered), caller_host_reserve_bytes=mib, caller_device_reserve_bytes=mib,
                                              lifecycle_extra_reserve_bytes=mib, native_reserve_bytes=4 * mib)
    else:
        from maple_syrup import native_dense_profile as nd

        profile = ncp.verify_source_profile(args.bundle, expected_syrup_digest=QUALIFIED_SYRUP_PACKAGE_SHA256)
        admission = ncp.admit_package_case(profile, case_dir)
        prep = nd.prepare_native_dense_case(admission, caller_reserve_bytes=mib)
        setup = lc.prepare_layered_case(prep, lc.load_layered_config(layered), caller_reserve_bytes=mib, lifecycle_extra_reserve_bytes=mib)
    return setup, profile


class Ctx:
    """Backend-specific helpers (device arrays on CUDA) so the scenario code below is written once."""

    def __init__(self, args, setup):
        self.args, self.setup, self.lop = args, setup, setup.lifecycle_operator
        self.cuda = args.backend == "cuda"
        self.xp = self.lop.xp

    def pet(self, rate):
        return self.xp.full(self.lop.n_cells, float(rate), dtype=np.float64)  # resident float64, no implicit upload

    def field(self):
        from maple_syrup.rainfall import rainfall_field

        ny, nx = self.lop.shape
        mask = self.lop.column.active.reshape(ny, nx) if self.cuda else np.asarray(self.lop.topology.active).reshape(ny, nx)
        return rainfall_field(ny, nx, active_mask=mask, xp=self.xp)  # uniform multiplier over the ACTIVE cells

    def host(self, a):
        return np.asarray(a.get() if hasattr(a, "get") else a)


def storm_to_completion(wl, lop, state):
    """Drive the existing bounded storm loop until it GENUINELY completes (a step/time/resource limit is never completion)."""
    res = wl.evolve_storm(lop, state, expected_parent=wl.expect_parent(state), until_s=state.t_s + 40_000.0)
    for _ in range(12):
        if res.complete:
            return res.state
        res = wl.evolve_storm(lop, res.state, expected_parent=wl.expect_parent(res.state), until_s=res.state.t_s + 40_000.0)
    raise SystemExit("the storm did not complete within the bounded loops: refusing to treat a limit as completion")


def summary(state, receipt=None):
    out = {"role": state.role, "t_s": state.t_s, "next_ordinal": state.daily.ordinal, "last_branch": state.daily.last_branch, "pending": state.daily.pending is not None}
    if receipt is not None:
        out.update({"branch": receipt.branch, "selection": receipt.selection, "capacity_failing_columns": receipt.capacity_failing_columns,
                    "recharge_m3": receipt.recharge_m3, "modeled_preferential_intake_m3": receipt.preferential_intake_m3, "modeled_matrix_intake_m3": receipt.matrix_intake_m3,
                    "drainage_m3": receipt.drainage_m3, "soil_evaporation_m3": receipt.soil_evaporation_m3, "surface_evaporation_m3": receipt.surface_evaporation_m3,
                    "export_m3": receipt.export_m3, "occurrence_start_s": receipt.occurrence_start_s, "day_start_s": receipt.t_start_s, "day_end_s": receipt.t_end_s})
    return out


def check_fresh_paths(args, workdir):
    """Validate every path BEFORE any write. The example never deletes, replaces or rewrites existing data (no force option): a work directory or a
    checkpoint directory that already exists is refused, so a rerun needs a new --workdir."""
    if not Path(args.bundle).is_dir():
        raise SystemExit(f"--bundle is not a directory: {args.bundle}")
    if os.path.lexists(workdir):
        raise SystemExit(f"refusing: --workdir already exists (nothing was changed): {workdir}; give a NEW directory")
    if not workdir.parent.is_dir():
        raise SystemExit(f"refusing: the parent of --workdir does not exist: {workdir.parent}")
    if args.checkpoint_dir and not args.checkpoint:
        raise SystemExit("--checkpoint-dir needs --checkpoint")
    if args.checkpoint_dir and os.path.lexists(args.checkpoint_dir):
        raise SystemExit(f"refusing: --checkpoint-dir already exists (the native writer never overwrites): {args.checkpoint_dir}")
    if args.checkpoint_dir and not Path(args.checkpoint_dir).resolve().parent.is_dir():
        raise SystemExit("refusing: the parent of --checkpoint-dir does not exist")


def run(args):
    from maple_syrup.subsurface import default_host_available_bytes  # the OS reading of available host memory (a reading, not a reservation)
    from maple_syrup import daily_water as dw
    from maple_syrup import layered_checkpoint as lc
    from maple_syrup import lifecycle_checkpoint as lk
    from maple_syrup import recharge as rc
    from maple_syrup import water_lifecycle as wl

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=False, exist_ok=False)
    setup, profile = prepare_case(args, workdir)
    ctx = Ctx(args, setup)
    lop = ctx.lop
    p_mm, i_mm_h, t_mm = args.p_mm, args.i_mm_h, args.t_mm
    out = {"example": "native_daily_recharge", "scenario": args.scenario, "backend": args.backend, "meaning": SCENARIOS[args.scenario][4],
           "units": {"descriptor": "P mm, I mm/h, T mm", "state": "m", "pet": "m/s", "clock": "s", "volumes": "m3 over active cells"},
           "source": {"syrup_package_sha256": profile.syrup_package_digest, "qualified_syrup_package_sha256": QUALIFIED_SYRUP_PACKAGE_SHA256,
                      "maple_native_commit": profile.maple_commit, "maple_package_sha256": profile.package_digest},
           "descriptor": {"p_mm": p_mm, "i_mm_h": i_mm_h, "t_mm": t_mm, "note": "illustrative test inputs; T has no recommended scientific default"}, "stages": []}
    boundary = setup.state
    state = dw.start_daily_lineage(lop, boundary, expected_parent=wl.expect_parent(boundary), run_digest=RUN_DIGEST)
    out["stages"].append({"stage": "daily lineage started at the accepted boundary", **summary(state)})
    plan = dw.next_plan(state, p_mm, i_mm_h, t_mm)
    out["selection"] = {"selection": plan.selection, "reason": plan.reason, "full_storm_duration_s": plan.duration_s}
    pet = ctx.pet(args.pet_m_s)
    parent = state
    res = dw.daily_water_action(lop, parent, expected_parent=wl.expect_parent(parent), plan=plan, pet_terrain_m_s=pet) if plan.selection != "storm" else None
    storm_plan, fb = plan, None
    if res is not None and res.status == "capacity_failure":
        f = res.fallback
        out["stages"].append({"stage": "recharge trial discarded (capacity failure); the accepted boundary is unchanged", "reason_flag_bit": f.reason_bit,
                              "failing_active_columns": f.failing_columns, "accepted_boundary_clock_s": parent.t_s})
        fb = f
    if res is not None and res.status == "complete":
        day = res.state
        out["stages"].append({"stage": f"{plan.selection} occurrence accepted (one {rc.DAY_S:.0f} s day)", **summary(day, res.receipt)})
    else:  # P > T, or the capacity fallback: the FULL native storm from the ORIGINAL accepted boundary
        control = args.control()
        storm = dw.begin_daily_storm(lop, parent, expected_parent=wl.expect_parent(parent), plan=storm_plan, fallback_of=fb, control=control)
        out["stages"].append({"stage": "full native storm begun (local schedule [0, D] + physical origin)", "origin_s": storm.binding.event.event_origin_s,
                              "absolute_edges_s": [float(x) for x in storm.binding.event.absolute_edges_s], **summary(storm)})
        done = storm_to_completion(wl, lop, storm)
        out["stages"].append({"stage": "storm genuinely completed (rain + recession), no E/T during it", "t_s": done.t_s, "rain_m3": done.payload.water.ledger.rain_m3})
        r2 = dw.finish_daily_storm(lop, done, expected_parent=wl.expect_parent(done), plan=storm_plan, pet_terrain_m_s=pet)
        day = r2.state
        out["stages"].append({"stage": "one daily update after the storm", **summary(day, r2.receipt)})
    out["first_day_audit_before_T"] = audit_record(wl.audit_lifecycle(lop, day), "native audit of the FIRST accepted occurrence, taken BEFORE the T withdrawal, the checkpoint and any later storm (cumulative ledgers at that moment)")
    q = ctx.xp.zeros((lop.n_layers, lop.n_cells))
    q[0] = args.t_request_m  # a zero-time T request (LEAF would compute it); SYRUP accounts what the soil can supply
    day, _feedback = wl.request_withdrawal(lop, day, expected_parent=wl.expect_parent(day), depth_m=q, identity="example-T-1")
    out["stages"].append({"stage": "zero-time T request accepted (no extra day)", "next_request_sequence": day.request_sequence, "transpiration_m3": wl.audit_lifecycle(lop, day)["transpiration_m3"]})
    closed = wl.close_interevent(lop, day, expected_parent=wl.expect_parent(day))
    out["stages"].append({"stage": "interevent closed to an accepted boundary", **summary(closed)})
    if args.checkpoint:
        ck = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else workdir / "checkpoint_v3"  # a NEW directory: the native writer refuses an existing path
        ident = {"example": "native_daily_recharge", "scenario": args.scenario}
        gib = 1 << 30  # CALLER quotas (illustrative values; a quota is never raised or inferred by the model)
        limits = lc.CheckpointLimits(args.ck_disk_bytes, args.ck_host_bytes, args.ck_device_bytes if ctx.cuda else 0, args.ck_scratch_bytes)
        if ctx.cuda:
            from maple_syrup import lifecycle_checkpoint_cuda_v3 as v3

            reserves = dict(caller_host_reserve_bytes=1 << 20, caller_device_reserve_bytes=1 << 20)
            v3.save_lifecycle_checkpoint_cuda_v3(ck, lop, closed, ident, limits=limits, **reserves)
            restored = v3.load_lifecycle_checkpoint_cuda_v3(ck, lop, ident, expected_role="boundary", limits=limits, reference_bed=boundary.payload.bed, **reserves)
        else:
            from maple_syrup import lifecycle_checkpoint_v3 as v3

            v3.save_lifecycle_checkpoint_v3(ck, lop, closed, ident, limits=limits, host_available_bytes_fn=default_host_available_bytes)
            restored = v3.load_lifecycle_checkpoint_v3(ck, lop, ident, expected_role="boundary", limits=limits, host_available_bytes_fn=default_host_available_bytes, reference_bed=boundary.payload.bed)
        lop, closed = restored.lop, restored.state  # the NEW operator on the loaded bed; the caller adopts the restored accepted state
        out["checkpoint_v3"] = {"directory": str(ck), "role": "boundary", "restored_next_ordinal": closed.daily.ordinal, "restored_t_s": closed.t_s,
                                "note": "a V3 directory holds the physical state + the daily cursor; V1/V2 directories are refused and V3 refuses cursor-free states"}
    final_state, final_completed = closed, None  # the CURRENT accepted state: the closed boundary after T (and after the V3 save/load when requested)
    if args.later_storm:  # a later FULL storm at the (nonzero) boundary clock: local forcing [0, D], physical origin = the boundary clock
        plan2 = dw.next_plan(closed, args.later_p_mm, args.later_i_mm_h, 0.0)  # T = 0 mm: every positive P is a full storm in this illustration
        st2 = dw.begin_daily_storm(lop, closed, expected_parent=wl.expect_parent(closed), plan=plan2, control=args.control())
        done2 = storm_to_completion(wl, lop, st2)
        final_state, final_completed = done2, True  # the genuine completed later storm; its daily occurrence is still PENDING (no further day, E/T or clock change is applied)
        out["later_storm"] = {"origin_s": st2.binding.event.event_origin_s, "absolute_edges_s": [float(x) for x in st2.binding.event.absolute_edges_s], "completed_at_s": done2.t_s,
                              "storm_rain_m3": done2.payload.water.ledger.rain_m3, "pending_ordinal": done2.daily.pending.ordinal}
    out["final_state"] = {"which": "completed later storm (daily occurrence still pending)" if args.later_storm else "closed boundary after T" + (" and the V3 save/load" if args.checkpoint else ""),
                          "role": final_state.role, "t_s": final_state.t_s, "next_ordinal": final_state.daily.ordinal, "pending_occurrence": final_state.daily.pending is not None,
                          "storm_genuinely_completed": final_completed}
    # FINAL closure: the native audit of the ACTUAL current accepted state (T, the checkpoint round trip and the later storm all included, each counted once)
    out["closure"] = audit_record(wl.audit_lifecycle(lop, final_state), "native audit of the FINAL state above (cumulative lifetime ledgers: first day + T + later storm)")
    print(json.dumps(out, indent=2, default=float))


def audit_record(audit, scope):
    return {"scope": scope, "residual_m3": audit["residual_m3"], "bound_m3": audit["bound_m3"], "rain_m3": audit["rain_m3"], "drainage_m3": audit["drainage_m3"],
            "soil_evaporation_m3": audit["soil_evaporation_m3"], "transpiration_m3": audit["transpiration_m3"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", required=True, help="the materialized MAPLE 9f bundle directory (CPU or CUDA profile bundle)")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="recharge")
    ap.add_argument("--backend", choices=("numpy", "numba", "cuda"), default="numpy")
    ap.add_argument("--device", type=int, default=0, help="CUDA device id (cuda backend only)")
    ap.add_argument("--workdir", required=True, help="a NEW, not yet existing directory (case copy, layered file, checkpoint); an existing one is refused")
    ap.add_argument("--p-mm", type=float, help="daily precipitation depth, mm (default: the scenario's illustrative value)")
    ap.add_argument("--i-mm-h", type=float, help="rain intensity, mm/h; positive P uses it for the full-storm duration 3600 P / I")
    ap.add_argument("--t-mm", type=float, help="amount threshold, mm: 0 < P <= T attempts recharge (equality attempts it); no scientific default exists")
    ap.add_argument("--pet-m-s", type=float, default=None, help="ILLUSTRATIVE potential evaporation rate applied over the one 86400 s day, m/s (not calibrated weather); default per scenario, see DEFAULT_PET")
    ap.add_argument("--t-request-m", type=float, default=1e-5, help="illustrative zero-time plant-withdrawal request for the top layer, m of water")
    ap.add_argument("--checkpoint", action="store_true", help="save and restore a V3 physical checkpoint of the closed boundary")
    ap.add_argument("--checkpoint-dir", help="a NEW directory for the V3 checkpoint (default WORKDIR/checkpoint_v3); an existing path is refused, never overwritten")
    ap.add_argument("--ck-disk-bytes", type=int, default=1 << 30, help="caller quota: serialized bytes (illustrative 1 GiB)")
    ap.add_argument("--ck-host-bytes", type=int, default=1 << 30, help="caller quota: host peak bytes (illustrative 1 GiB; the real available memory is read from the OS)")
    ap.add_argument("--ck-device-bytes", type=int, default=1 << 30, help="caller quota: device bytes (CUDA only; 0 is used on CPU)")
    ap.add_argument("--ck-scratch-bytes", type=int, default=1 << 30, help="caller quota: host scratch bytes (illustrative 1 GiB)")
    ap.add_argument("--later-storm", action="store_true", help="run a later full storm after close at the new physical clock")
    ap.add_argument("--later-p-mm", type=float, default=1.0)
    ap.add_argument("--later-i-mm-h", type=float, default=360.0)
    args = ap.parse_args(argv)
    d = SCENARIOS[args.scenario][3]
    if args.pet_m_s is None:
        args.pet_m_s = DEFAULT_PET[args.scenario]
    args.p_mm, args.i_mm_h, args.t_mm = (d[0] if args.p_mm is None else args.p_mm, d[1] if args.i_mm_h is None else args.i_mm_h, d[2] if args.t_mm is None else args.t_mm)
    def control():
        from maple_syrup import layered_event as ev

        return ev.LayeredEventControl()  # the DEFAULT storm control and quiet policy: no loosened completion

    args.control = control
    check_fresh_paths(args, Path(args.workdir).resolve())  # every refusal happens HERE, before any import of the model or any write; nothing is overwritten or deleted, ever
    run(args)


if __name__ == "__main__":
    sys.exit(main())
