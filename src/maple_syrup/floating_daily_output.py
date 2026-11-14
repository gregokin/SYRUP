"""G3 S2: the every-N-days soil-moisture save of the FLOATING daily lineage (`floating_daily`), through the qualified G1 frame writer (CPU/host; no GPU run).

`FloatingMoistureOutput` is a narrow cadence owner: created from a genuine closed boundary of a `FloatingDailyState` lineage with a STRICT positive integer `every_n_days` (bool/float/str refused; there is no default N and with no output
object the option is OFF), bound by an explicit versioned record (schema, writer-module hash, run, source, config, origin ordinal/date, cadence, geometry, nominal edges, area, explicit limits) sealed no-clobber in the (empty) output
directory. The output is notified ONLY by `floating_daily.close_floating_day` for the accepted closed state: due days are exactly the completed nominal days N, 2N, ... from the bound origin ordinal (storm/wind physical time never
counts days). A due day saves the CURRENT post-T water `W (L, N)`, the lineage's canonical depth `D`, the active mask, the nominal edges, area and shape through `floating_moisture_frame.write_floating_frame`; the frame's
identity/generations/parent/state digests are DERIVED from the accepted state (`FloatingDailyState.frame_identity()`), never caller-authored. The lineage digest is an execution-receipt chain (not a water content hash); the G1
writer itself hashes and compares the actual saved arrays, so a replay verifies the sealed bytes. Not-due days and an absent output touch no array and add no copy/hash (a host integer comparison only); a due day streams
bounded chunks (W, D, mask) with admission BEFORE any transfer or file write; no bed copy and no history is kept.

Failure/retry: everything checkable (type, expected parent, binding, sequence, quotas) is validated BEFORE any I/O; if the frame write fails the exception propagates, the cadence is unchanged, `close_floating_day` returned no
closed state and the caller still holds the accepted interevent parent (retry calls `close_floating_day` again; an already SEALED due frame is verified against the live bytes and never overwritten; unknown/unsealed files are
preserved untouched). `resume` re-binds an existing output to a restored closed boundary (a lagging restart; sealed frames beyond it are verified on replay); N, source, run and origin cannot change on resume.

NOT claimed: a physical checkpoint, fresh-process reconstruction of the lineage, a coupled-storm output, GPU behaviour. The frame is a diagnostic.
"""
from __future__ import annotations

import datetime
import hashlib
import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup import floating_daily as fd
from maple_syrup import floating_moisture_frame as g1
from maple_syrup import soil_moisture_output as g0
from maple_syrup.soil_moisture_output import (
    MoistureOutputConflict,
    MoistureOutputError,
    MoistureOutputIndeterminate,
    MoistureOutputLimits,
    MoistureOutputQuotaError,
)

__all__ = ["SCHEMA_OUTPUT", "FloatingMoistureOutput", "FloatingOutputNotice"]

SCHEMA_OUTPUT = "maple_syrup.floating_daily_output/2"  # v2: separate model-source pair and complete physics-config fields (the unadopted v1 is not migrated)
OUTPUT_NAME = "floating_output.json"
_MODULE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_TOKEN = object()
JSON_FIXED_BYTES = 4608
JSON_PER_EDGE_BYTES = 26


@dataclass(frozen=True)
class FloatingOutputNotice:
    day: int  # completed nominal days since the origin
    ordinal: int
    due: bool
    action: str  # not_due | saved | verified_existing | repeat_same_state
    path: str | None
    bytes_written: int


def _record(state: Any, n: int, date0: datetime.date | None, ci: dict | None, limits: MoistureOutputLimits) -> dict:
    ident = state.frame_identity()
    c = state.ctx
    return {"schema": SCHEMA_OUTPUT, "kind": "floating_soil_moisture_output", "every_n_days": n, "origin_ordinal": int(state.cursor.ordinal), "origin_date": None if date0 is None else date0.isoformat(),
            "run_digest": ident["run_digest"], "source_sha256": ident["source_sha256"], "code_source_pair": {k: c.source_pair[k] for k in sorted(c.source_pair)}, "physics_config_sha256": c.stable_sha256 if c.profile else c.physics_config_sha256,
            "native_config_sha256": c.config_sha256, "root_column_sha256": None if c.profile else c.root_column_sha256, "source_witness": c.source_witness, "clock_contract": state.cursor.clock_contract, "profile": c.profile or fd.PROFILE,
            "n_layers": int(c.n_layers), "n_cells": int(c.n_cells), "shape_yx": [int(s) for s in c.shape], "cell_area_m2": float(c.area), "depth_edges_m": [float(v) for v in c.edges_m],
            "max_manifest_bytes": limits.max_manifest_bytes, "module_sha256": _MODULE_SHA, "caller_identity": ci, "not_a_restart": True}


def _need_closed(state: Any, expected: Any, *, origin: bool) -> None:
    fd._check_parent(state, expected, "boundary")
    if getattr(state.ctx.hold, "vegetation", None) is not None:  # the accepted every-N frame profile has no vegetation attribution: it never silently writes an unattributed frame for a vegetation-bound lineage
        raise MoistureOutputError("the lineage has an applied vegetation generation: this frame profile does not carry its attribution (a versioned frame profile comes next; `vegetation_lifecycle.vegetation_attribution` exposes it)")
    if state.cursor.pending is not None:
        raise MoistureOutputError("a storm occurrence is pending in the cursor: the day is not complete")
    if not origin and state.cursor.last_branch is None:
        raise MoistureOutputError("the state is not the close of a completed daily occurrence")


def _count_n(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= int(value) <= 1 << 31:
        raise MoistureOutputError(f"every_n_days must be a strict positive integer (not a bool, float or string), got {value!r}")
    return int(value)


class FloatingMoistureOutput:
    """Bounded cadence receipt + writer bound to ONE output directory and ONE floating daily lineage. Construct with `create` or `resume` (no default cadence)."""

    def __init__(self, token: object, **fields: Any) -> None:
        if token is not _TOKEN:
            raise MoistureOutputError("FloatingMoistureOutput is created only by FloatingMoistureOutput.create() or .resume()")
        self.__dict__.update(fields)

    @classmethod
    def create(cls, directory: str | Path, origin_state: Any, *, expected_parent: Any, every_n_days: Any, limits: MoistureOutputLimits, origin_date: str | None = None, caller_identity: dict | None = None,
               host_available_bytes_fn: Callable[[], int] | None = None, disk_available_bytes_fn: Callable[[], int] | None = None) -> FloatingMoistureOutput:
        n = _count_n(every_n_days)
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
                    raise MoistureOutputError("the output directory must be empty at creation (use FloatingMoistureOutput.resume for an existing output)")
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
                   origin_ordinal=rec["origin_ordinal"], run_digest=rec["run_digest"], caller_identity=ci)

    @classmethod
    def resume(cls, directory: str | Path, state: Any, *, expected_parent: Any, limits: MoistureOutputLimits, caller_identity: dict | None = None,
               host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingMoistureOutput:
        if not isinstance(limits, MoistureOutputLimits):
            raise MoistureOutputError("limits must be a MoistureOutputLimits")
        ci = g0._check_caller_identity(caller_identity)
        _need_closed(state, expected_parent, origin=True)
        rootfd = g0._open_dir(directory)
        try:
            info = g0._lstat(rootfd, OUTPUT_NAME)
            if info is None:
                raise MoistureOutputError("the directory holds no floating output record")
            g0._admit_host(limits, limits.staging_bytes(int(info.st_size)), host_available_bytes_fn)
            rec = g0._strict_loads(g0._read_small(rootfd, OUTPUT_NAME, g0.MAX_SMALL_JSON_BYTES))
            if type(rec) is not dict or rec.get("schema") != SCHEMA_OUTPUT or rec.get("body_sha256") != g0._body_sha(rec):
                raise MoistureOutputError("the floating output record is malformed or its checksum differs")
            if rec["module_sha256"] != _MODULE_SHA:
                raise MoistureOutputError("the output was written by a different writer module: no silent rebind or migration")
            if rec["caller_identity"] != ci:
                raise MoistureOutputError("the explicit caller identity differs from the one sealed in the output record")
            fresh = _record(state, rec["every_n_days"], None if rec["origin_date"] is None else g0._parse_date(rec["origin_date"]), ci, limits)
            fresh["origin_ordinal"] = rec["origin_ordinal"]
            same = {k: rec[k] for k in rec if k != "body_sha256"} == fresh
            if not same:
                raise MoistureOutputError("the restored state does not match the output's sealed run/source/config/geometry/cadence/origin binding (N, source, run and origin cannot change on resume)")
            d = int(state.cursor.ordinal) - rec["origin_ordinal"]
            if d < 0:
                raise MoistureOutputError("the state is older than the output's origin")
            if d >= 1 and state.cursor.last_branch is None:
                raise MoistureOutputError("the boundary is not the close of a completed daily occurrence")
            ident = os.fstat(rootfd)
            n = rec["every_n_days"]
            for k in range(1, d // n + 1):  # every due day up to the state must already be SEALED (manifest-last); frames ahead are verified when the replay reaches them
                name = g1.frame_name(k * n)
                dayfd = g0._open_dir(name, dir_fd=rootfd) if g0._lstat(rootfd, name) is not None else None
                if dayfd is None:
                    raise MoistureOutputError(f"the output is missing the due frame of day {k * n}; it cannot be recreated: refused")
                try:
                    man = g0._strict_loads(g0._read_small(dayfd, g1.MANIFEST, rec["max_manifest_bytes"]))
                finally:
                    os.close(dayfd)
                if man.get("day") != k * n:
                    raise MoistureOutputError(f"frame {name} carries another day")
        finally:
            os.close(rootfd)
        out = cls(_TOKEN, directory=str(directory), rec=rec, limits=limits, last_day=d, _state_ref=weakref.ref(state), _last_notice=None, ident=(ident.st_dev, ident.st_ino), every_n_days=rec["every_n_days"],
                  origin_ordinal=rec["origin_ordinal"], run_digest=rec["run_digest"], caller_identity=ci)
        return out

    def notify_day_closed(self, state: Any, *, expected_parent: Any, host_available_bytes_fn: Callable[[], int] | None = None, disk_available_bytes_fn: Callable[[], int] | None = None) -> FloatingOutputNotice:
        """Call once per accepted FINAL daily close, in order. Validates type/expected parent/binding/sequence before any I/O; a failed save leaves the cadence unchanged."""
        _need_closed(state, expected_parent, origin=False)
        ident = state.frame_identity()
        rec = self.rec
        if ident["run_digest"] != self.run_digest or ident["source_sha256"] != rec["source_sha256"] or (state.ctx.stable_sha256 if state.ctx.profile else state.ctx.physics_config_sha256) != rec["physics_config_sha256"] or (state.ctx.profile or fd.PROFILE) != rec["profile"] or dict(state.ctx.source_pair) != rec["code_source_pair"]:
            raise MoistureOutputError("the closed state does not belong to this output's run/source/config")
        d = int(state.cursor.ordinal) - self.origin_ordinal
        if d < 0:
            raise MoistureOutputConflict("the state precedes the lineage origin")
        if d == self.last_day and d >= 1:
            ref = self._state_ref
            same = ref is not None and ref() is state
            if same and self._last_notice is not None:
                return FloatingOutputNotice(d, int(state.cursor.ordinal), self._last_notice.due, "repeat_same_state", self._last_notice.path, 0)
            if d % self.every_n_days == 0:
                return self._due(state, d, host_available_bytes_fn, disk_available_bytes_fn)  # re-prove against the SEALED bytes (also the first call after resume)
            if same:
                return FloatingOutputNotice(d, int(state.cursor.ordinal), False, "repeat_same_state", None, 0)
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
        if d % self.every_n_days:  # NOT DUE: a host integer comparison only (no array read, copy or hash)
            note = FloatingOutputNotice(d, int(state.cursor.ordinal), False, "not_due", None, 0)
            self.__dict__.update(last_day=d, _state_ref=weakref.ref(state), _last_notice=note)
            return note
        return self._due(state, d, host_available_bytes_fn, disk_available_bytes_fn)

    def _due(self, state: Any, d: int, host_fn: Any, disk_fn: Any) -> FloatingOutputNotice:
        rec, c = self.rec, state.ctx
        date = None if rec["origin_date"] is None else (datetime.date.fromisoformat(rec["origin_date"]) + datetime.timedelta(days=d)).isoformat()
        meta = g1.FloatingFrameMeta(day=d, cadence_n=self.every_n_days, origin_ordinal=self.origin_ordinal, physical_t_s=float(state.t_s), origin_date=rec["origin_date"], date=date)
        identity = g1.FloatingFrameIdentity(**state.frame_identity())
        notice = g1.write_floating_frame(self.directory, water_m=state.water.water_m, depth_m=c.depth_m, active=c.active, depth_edges_m=c.edges_m, cell_area_m2=c.area, shape_yx=c.shape, meta=meta,
                                         identity=identity, limits=self.limits, caller_identity=self.caller_identity, host_available_bytes_fn=host_fn, disk_available_bytes_fn=disk_fn)
        out = FloatingOutputNotice(d, int(state.cursor.ordinal), True, notice.action, notice.path, int(notice.bytes_written))
        self.__dict__.update(last_day=d, _state_ref=weakref.ref(state), _last_notice=out)
        return out
