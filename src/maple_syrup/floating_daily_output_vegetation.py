"""Phase 2.5 vegetation-aware EVERY-N-DAYS soil-moisture output of the event<->daily bridge lineage (profile v2; CPU/host and device arrays; no GPU run by its author).

`VegetationMoistureOutput` is the v2 sibling of `floating_daily_output.FloatingMoistureOutput` (unchanged; it still FAILS CLOSED for a vegetation-bound state). It is notified ONLY by `floating_daily.close_floating_day` for the accepted closed
state (exact recognised classes: the v1 output or this one; no duck typing), with the same cadence semantics: a STRICT positive integer `every_n_days` (no default; no output object = OFF), due days are exactly the completed NOMINAL days
N, 2N, ... from the bound origin ordinal (storm / wind physical time never counts days; no extra E/T/drainage), a due day saves the post-T `W (L, N)`, the canonical `D`, the mask, the edges, the area and the shape through
`floating_moisture_frame_vegetation.write_vegetation_frame`; a not-due day is a host integer comparison only (no array read, copy, hash or transfer).

What differs from v1 and why it is allowed to differ:
* STATIC binding (sealed no-clobber in `floating_output_vegetation.json`): schema, writer-module hashes (this module AND the v2 frame codec), run, the STABLE source identity (`FloatingDailyState.frame_identity()['source_sha256']`: model source pair +
  case-stable physics + cold anchor -- it excludes the evolving property root), the stable physics config, nominal origin ordinal / date, N, geometry, edges, area and limits. It deliberately holds NO vegetation generation and NO property
  generation: both may legitimately change between frames.
* PER-FRAME attribution (not a static anchor): each due frame carries the frame identity of the accepted state (`property_sha256` keeps its v1 meaning = the CURRENT root column identity) and the vegetation block derived from the accepted state by
  `vegetation_lifecycle.vegetation_attribution` -- the stable generation / recipe / binding / source ids and the CURRENT column / matrix / preferential / law ids of the actual holder, never the apply-time ids. A lineage with no
  vegetation generation writes the explicit `null` block. The attribution is DERIVED from the genuine accepted state inside the writer (never a caller argument), so the frame data are caller-pure but the writer is the proof.
* Mutual refusal with v1: a different record file name, schema and codec; the v1 writer refuses a vegetation-bound state; this writer refuses a state of another profile and a directory that holds a v1 output.

Resume re-binds an existing output to a restored closed boundary of the SAME genuine lineage (labelled: this is NOT a physical fresh-process restart; that depends on the later outer files and no state is fabricated from arrays).
Failure/retry/idempotence/no-clobber/unsealed preservation/directory binding/stale-foreign/lagging-resume rules are those of v1 (the same primitives).
"""
from __future__ import annotations

import datetime
import hashlib
import os
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

from maple_syrup import floating_daily as fd
from maple_syrup import floating_daily_output as fo
from maple_syrup import floating_moisture_frame as g1
from maple_syrup import floating_moisture_frame_vegetation as g2
from maple_syrup import soil_moisture_output as g0
from maple_syrup.soil_moisture_output import (
    MoistureOutputConflict,
    MoistureOutputError,
    MoistureOutputIndeterminate,
    MoistureOutputLimits,
    MoistureOutputQuotaError,
)

__all__ = ["SCHEMA_OUTPUT_V2", "VegetationMoistureOutput"]

SCHEMA_OUTPUT_V2 = "maple_syrup.floating_daily_output_vegetation/1"
OUTPUT_NAME = "floating_output_vegetation.json"
_MODULE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_TOKEN = object()
JSON_FIXED_BYTES = 4864
JSON_PER_EDGE_BYTES = 26


def _record(state: Any, n: int, date0: datetime.date | None, ci: dict | None, limits: MoistureOutputLimits) -> dict:
    ident = state.frame_identity()
    c = state.ctx
    return {"schema": SCHEMA_OUTPUT_V2, "kind": "floating_soil_moisture_output_vegetation", "every_n_days": n, "origin_ordinal": int(state.cursor.ordinal), "origin_date": None if date0 is None else date0.isoformat(),
            "run_digest": ident["run_digest"], "source_sha256": ident["source_sha256"], "code_source_pair": {k: c.source_pair[k] for k in sorted(c.source_pair)}, "physics_config_sha256": c.stable_sha256,
            "native_config_sha256": c.config_sha256, "source_witness": c.source_witness, "clock_contract": state.cursor.clock_contract, "profile": c.profile, "frame_profile": g2.PROFILE_V2,
            "n_layers": int(c.n_layers), "n_cells": int(c.n_cells), "shape_yx": [int(s) for s in c.shape], "cell_area_m2": float(c.area), "depth_edges_m": [float(v) for v in c.edges_m],
            "max_manifest_bytes": limits.max_manifest_bytes, "module_sha256": _MODULE_SHA, "frame_module_sha256": g2._MODULE_SHA, "caller_identity": ci, "not_a_restart": True}


def _need_closed(state: Any, expected: Any, *, origin: bool) -> None:
    """The closed bridge-profile state of THIS lineage (a genuine `FloatingDailyState`; with or without a vegetation generation). The historical generation-0 daily profile has no evolving root and is the v1 output's business."""
    fd._check_parent(state, expected, "boundary")
    if not state.ctx.profile:
        raise MoistureOutputError("the vegetation-aware output serves the event<->daily bridge profile (the historical daily profile uses the v1 output)")
    if state.cursor.pending is not None:
        raise MoistureOutputError("a storm occurrence is pending in the cursor: the day is not complete")
    if not origin and state.cursor.last_branch is None:
        raise MoistureOutputError("the state is not the close of a completed daily occurrence")


def _attribution(state: Any) -> g2.VegetationAttribution | None:
    """Derived from the ACCEPTED state by the lifecycle (lazy import: the frame codec itself never imports it). None = the explicit no-generation contract."""
    from maple_syrup import vegetation_lifecycle as vl

    try:
        d = vl.vegetation_attribution(state)
    except vl.VegetationLifecycleError as exc:
        raise MoistureOutputError(f"the state's vegetation record is not genuine: {exc}") from None
    return None if d is None else g2.attribution_from_dict(d)


class VegetationMoistureOutput:
    """Bounded cadence receipt + v2 writer bound to ONE output directory and ONE bridge daily lineage. Construct with `create` or `resume` (no default cadence)."""

    def __init__(self, token: object, **fields: Any) -> None:
        if token is not _TOKEN:
            raise MoistureOutputError("VegetationMoistureOutput is created only by VegetationMoistureOutput.create() or .resume()")
        self.__dict__.update(fields)

    @classmethod
    def create(cls, directory: str | Path, origin_state: Any, *, expected_parent: Any, every_n_days: Any, limits: MoistureOutputLimits, origin_date: str | None = None, caller_identity: dict | None = None,
               host_available_bytes_fn: Callable[[], int] | None = None, disk_available_bytes_fn: Callable[[], int] | None = None) -> VegetationMoistureOutput:
        n = fo._count_n(every_n_days)
        if not isinstance(limits, MoistureOutputLimits):
            raise MoistureOutputError("limits must be a MoistureOutputLimits")
        date0 = None if origin_date is None else g0._parse_date(origin_date)
        ci = g0._check_caller_identity(caller_identity)
        _need_closed(origin_state, expected_parent, origin=True)
        rec = _record(origin_state, n, date0, ci, limits)
        bound = JSON_FIXED_BYTES + (0 if ci is None else len(g0._canon(ci))) + JSON_PER_EDGE_BYTES * (rec["n_layers"] + 1)
        if bound > g0.MAX_SMALL_JSON_BYTES:
            raise MoistureOutputQuotaError("the output record would exceed the record ceiling")
        if bound > limits.max_total_bytes:
            raise MoistureOutputQuotaError(f"the static record needs up to {bound} bytes but max_total_bytes is {limits.max_total_bytes}")
        g0._admit_host(limits, limits.staging_bytes(bound), host_available_bytes_fn)
        rec["body_sha256"] = g0._body_sha(rec)
        blob = g0._canon(rec)
        if len(blob) > bound:
            raise MoistureOutputError("internal: the output record exceeded its analytic size bound")
        rootfd = g0._open_dir(directory)
        try:
            with os.scandir(rootfd) as it:
                if any(True for _ in it):
                    raise MoistureOutputError("the output directory must be empty at creation (use VegetationMoistureOutput.resume for an existing output; a v1 output directory is never reused)")
            g0._check_disk(rootfd, len(blob), limits, disk_available_bytes_fn)
            tmp = OUTPUT_NAME + g0.TMP_SUFFIX
            try:
                g0._write_new(rootfd, tmp, [blob])
                g0._seal_link(rootfd, tmp, OUTPUT_NAME)
            except BaseException as exc:
                if g0._lstat(rootfd, OUTPUT_NAME) is None:
                    try:
                        os.unlink(tmp, dir_fd=rootfd)
                    except OSError:
                        pass
                    raise
                if isinstance(exc, MoistureOutputIndeterminate):
                    raise
                raise MoistureOutputIndeterminate(f"{OUTPUT_NAME} is sealed but a durability step failed ({type(exc).__name__})", str(directory)) from exc
            ident = os.fstat(rootfd)
        finally:
            os.close(rootfd)
        return cls(_TOKEN, directory=str(directory), rec=rec, limits=limits, last_day=0, _state_ref=None, _last_notice=None, ident=(ident.st_dev, ident.st_ino), every_n_days=n,
                   origin_ordinal=rec["origin_ordinal"], run_digest=rec["run_digest"], caller_identity=ci, order=None)

    @classmethod
    def resume(cls, directory: str | Path, state: Any, *, expected_parent: Any, limits: MoistureOutputLimits, caller_identity: dict | None = None,
               host_available_bytes_fn: Callable[[], int] | None = None) -> VegetationMoistureOutput:
        if not isinstance(limits, MoistureOutputLimits):
            raise MoistureOutputError("limits must be a MoistureOutputLimits")
        ci = g0._check_caller_identity(caller_identity)
        _need_closed(state, expected_parent, origin=True)
        rootfd = g0._open_dir(directory)
        try:
            info = g0._lstat(rootfd, OUTPUT_NAME)
            if info is None:
                raise MoistureOutputError("the directory holds no vegetation-aware floating output record (a v1 output is not resumed here)")
            g0._admit_host(limits, limits.staging_bytes(int(info.st_size)), host_available_bytes_fn)
            rec = g0._strict_loads(g0._read_small(rootfd, OUTPUT_NAME, g0.MAX_SMALL_JSON_BYTES))
            if type(rec) is not dict or rec.get("schema") != SCHEMA_OUTPUT_V2 or rec.get("body_sha256") != g0._body_sha(rec):
                raise MoistureOutputError("the vegetation output record is malformed or its checksum differs")
            if rec["module_sha256"] != _MODULE_SHA or rec["frame_module_sha256"] != g2._MODULE_SHA:
                raise MoistureOutputError("the output was written by a different writer / frame-codec module: no silent rebind or migration")
            if rec["caller_identity"] != ci:
                raise MoistureOutputError("the explicit caller identity differs from the one sealed in the output record")
            fresh = _record(state, rec["every_n_days"], None if rec["origin_date"] is None else g0._parse_date(rec["origin_date"]), ci, limits)
            fresh["origin_ordinal"] = rec["origin_ordinal"]
            if {k: rec[k] for k in rec if k != "body_sha256"} != fresh:
                raise MoistureOutputError("the restored state does not match the output's sealed run/source/config/geometry/cadence/origin binding (N, source, run and origin cannot change on resume)")
            d = int(state.cursor.ordinal) - rec["origin_ordinal"]
            if d < 0:
                raise MoistureOutputError("the state is older than the output's origin")
            if d >= 1 and state.cursor.last_branch is None:
                raise MoistureOutputError("the boundary is not the close of a completed daily occurrence")
            ident = os.fstat(rootfd)
            n = rec["every_n_days"]
            prev = None
            order = None
            t_prev = 0.0
            for k in range(1, d // n + 1):  # every due day up to the state must already be SEALED; frames ahead are verified when the replay reaches them
                name = g1.frame_name(k * n)
                dayfd = g0._open_dir(name, dir_fd=rootfd) if g0._lstat(rootfd, name) is not None else None
                if dayfd is None:
                    raise MoistureOutputError(f"the output is missing the due frame of day {k * n}; it cannot be recreated: refused")
                try:
                    man = g0._strict_loads(g0._read_small(dayfd, g1.MANIFEST, rec["max_manifest_bytes"]))
                finally:
                    os.close(dayfd)
                prev = cls._check_past_manifest(man, name, k * n, rec, ci, prev)
                if not t_prev <= man["physical_t_s"] <= float(state.t_s):  # physical time never runs backwards along the sealed past, and never beyond the resumed state
                    raise MoistureOutputError(f"frame {name} carries a physical time outside [previous frame, resumed state]")
                t_prev = man["physical_t_s"]
            cur = _attribution(state)
            try:
                g2.check_order(prev, cur)
            except MoistureOutputError as exc:
                raise MoistureOutputError(f"the resumed state's vegetation lineage disagrees with the sealed frames: {exc}") from None
            # the frame AT the resume day is the state's own: re-prove its sealed identity and attribution now (manifest only), so a later day cannot follow a foreign past
            if d >= 1 and d % n == 0 and (man["identity"] != state.frame_identity() or man["vegetation"] != g2._veg_block(cur) or man["physical_t_s"] != float(state.t_s)):
                raise MoistureOutputConflict(f"the sealed frame of day {d} differs from the resumed state's identity / vegetation attribution / physical time")
            order = prev if cur is None else cur.order_key
        finally:
            os.close(rootfd)
        return cls(_TOKEN, directory=str(directory), rec=rec, limits=limits, last_day=d, _state_ref=weakref.ref(state), _last_notice=None, ident=(ident.st_dev, ident.st_ino), every_n_days=rec["every_n_days"],
                   origin_ordinal=rec["origin_ordinal"], run_digest=rec["run_digest"], caller_identity=ci, order=order)

    @staticmethod
    def _check_past_manifest(man: Any, name: str, day: int, rec: dict, ci: dict | None, prev: tuple | None) -> tuple | None:
        """Manifest-only proof of one sealed past frame (no array read): the full v2 manifest, the sealed static binding, and a non-regressing biological lineage. Returns the new `(generation, recipe_id)` order key."""
        try:
            g2._check_manifest(man)
        except MoistureOutputError as exc:
            raise MoistureOutputError(f"frame {name} is not a valid vegetation frame manifest: {exc}") from None
        ident = man["identity"]
        static = (man["day"] == day and man["module_sha256"] == g2._MODULE_SHA and man["cadence_n"] == rec["every_n_days"] and man["origin_ordinal"] == rec["origin_ordinal"] and man["origin_date"] == rec["origin_date"]
                  and ident["run_digest"] == rec["run_digest"] and ident["source_sha256"] == rec["source_sha256"] and man["n_layers"] == rec["n_layers"] and man["n_cells"] == rec["n_cells"]
                  and man["shape_yx"] == rec["shape_yx"] and man["cell_area_m2"] == rec["cell_area_m2"] and man["depth_edges_m"] == rec["depth_edges_m"] and man["caller_identity"] == ci)
        if not static:
            raise MoistureOutputError(f"frame {name} does not match the output's sealed day / module / cadence / origin / run / source / geometry / caller binding")
        att = None if man["vegetation"] is None else g2.attribution_from_dict(man["vegetation"])
        try:
            g2.check_order(prev, att)
        except MoistureOutputError as exc:
            raise MoistureOutputError(f"frame {name}: {exc}") from None
        return prev if att is None else att.order_key

    def notify_day_closed(self, state: Any, *, expected_parent: Any, host_available_bytes_fn: Callable[[], int] | None = None, disk_available_bytes_fn: Callable[[], int] | None = None) -> fo.FloatingOutputNotice:
        """Call once per accepted FINAL daily close, in order (see `fo.FloatingMoistureOutput.notify_day_closed`: the same sequencing, replay and refusal rules)."""
        _need_closed(state, expected_parent, origin=False)
        ident = state.frame_identity()
        rec = self.rec
        if ident["run_digest"] != self.run_digest or ident["source_sha256"] != rec["source_sha256"] or state.ctx.stable_sha256 != rec["physics_config_sha256"] or state.ctx.profile != rec["profile"] or dict(state.ctx.source_pair) != rec["code_source_pair"]:
            raise MoistureOutputError("the closed state does not belong to this output's run/source/config")
        d = int(state.cursor.ordinal) - self.origin_ordinal
        if d < 0:
            raise MoistureOutputConflict("the state precedes the lineage origin")
        if d == self.last_day and d >= 1:
            ref = self._state_ref
            same = ref is not None and ref() is state
            if same and self._last_notice is not None:
                return fo.FloatingOutputNotice(d, int(state.cursor.ordinal), self._last_notice.due, "repeat_same_state", self._last_notice.path, 0)
            if d % self.every_n_days == 0:
                return self._due(state, d, host_available_bytes_fn, disk_available_bytes_fn)  # re-prove against the SEALED bytes (also the first call after resume)
            if same:
                return fo.FloatingOutputNotice(d, int(state.cursor.ordinal), False, "repeat_same_state", None, 0)
            raise MoistureOutputConflict(f"day {d} was already notified for a different state instance/branch; refused (nothing overwritten)")
        if d > g0.MAX_DAY:
            raise MoistureOutputError(f"day {d} exceeds the representable day-directory range; refused before any I/O")
        if d != self.last_day + 1:
            raise MoistureOutputError(f"days must be notified sequentially: expected day {self.last_day + 1}, got {d} (a skipped, rewound or foreign branch is refused)")
        if rec["origin_date"] is not None:
            try:
                datetime.date.fromisoformat(rec["origin_date"]) + datetime.timedelta(days=d)
            except (OverflowError, ValueError):
                raise MoistureOutputError("the calendar date of this day is not representable; refused before any state change") from None
        if d % self.every_n_days:  # NOT DUE: a host integer comparison only (no array read, copy, hash, attribution or transfer)
            note = fo.FloatingOutputNotice(d, int(state.cursor.ordinal), False, "not_due", None, 0)
            self.__dict__.update(last_day=d, _state_ref=weakref.ref(state), _last_notice=note)
            return note
        return self._due(state, d, host_available_bytes_fn, disk_available_bytes_fn)

    def _due(self, state: Any, d: int, host_fn: Any, disk_fn: Any) -> fo.FloatingOutputNotice:
        rec, c = self.rec, state.ctx
        date = None if rec["origin_date"] is None else (datetime.date.fromisoformat(rec["origin_date"]) + datetime.timedelta(days=d)).isoformat()
        meta = g1.FloatingFrameMeta(day=d, cadence_n=self.every_n_days, origin_ordinal=self.origin_ordinal, physical_t_s=float(state.t_s), origin_date=rec["origin_date"], date=date)
        identity = g1.FloatingFrameIdentity(**state.frame_identity())  # property_sha256 = the CURRENT root column identity (its v1 meaning)
        veg = _attribution(state)  # DERIVED from the accepted state: stable generation ids + the CURRENT law ids of the actual holder
        try:
            g2.check_order(self.order, veg)  # never a regression below / a foreign recipe at the generation of the latest frame written or proved by this output
        except MoistureOutputError as exc:
            raise MoistureOutputConflict(f"day {d}: {exc}") from None
        notice = g2.write_vegetation_frame(self.directory, water_m=state.water.water_m, depth_m=c.depth_m, active=c.active, depth_edges_m=c.edges_m, cell_area_m2=c.area, shape_yx=c.shape, meta=meta,
                                           identity=identity, vegetation=veg, limits=self.limits, caller_identity=self.caller_identity, host_available_bytes_fn=host_fn, disk_available_bytes_fn=disk_fn)
        out = fo.FloatingOutputNotice(d, int(state.cursor.ordinal), True, notice.action, notice.path, int(notice.bytes_written))
        self.__dict__.update(last_day=d, _state_ref=weakref.ref(state), _last_notice=out, order=self.order if veg is None else veg.order_key)
        return out

