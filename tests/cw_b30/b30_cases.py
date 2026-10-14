"""A caller-supplied `case_setup` hook for `run_complete_case` over genuine synthetic MAPLE beds (CW-B30 tests).

The hook is an ordinary external case-setup hook: it returns a NEW mapping with a stand-in `verified` record and a
`prepared` case. The physical case is the actual MAPLE bed of the existing test helpers: the wet 3-cell chain with
interval commits (`dense`), or the deep resident-window bed with CSR phase (`window`). The rainfall rates are the
helpers' own; the schedule's provenance digest is bound to a small forcing file the hook writes, so the runner's
source checks run against real files. No science is duplicated or replaced."""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

KINDS = ("dense", "window")


def _files(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("case", "recipe", "mahleran"):
        (root / sub).mkdir(exist_ok=True)
    xml = root / "mahleran" / "model.xml"
    forcing = root / "forcing.txt"
    if not xml.exists():
        xml.write_text("<mahleran_input/>\n")
        forcing.write_text("synthetic constant forcing of the CW-B30 tests\n")
    return xml, forcing


def build_case(kind, control, root: Path):
    """`(s, ctx, schedule)` of the requested genuine MAPLE case, bound to the runner's control."""
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from test_sediment_event import chain_elevation, constant_rainfall, make_bed, setup

    from maple_syrup.sediment_event import initial_event_state

    if kind == "dense":
        bed, ctx = make_bed(chain_elevation(3), depth_m=.001,
                            commit_spec=TopographicCommitSpec(commit_interval_s=2.0))
        s = setup(bed, ctx, ksat=1e-5, control=control)
        return s, s["ctx"], constant_rainfall(0, 10, 100)
    if kind == "window":
        from test_windowed_storm import window_inputs

        s, ctx, _own_control, schedule = window_inputs(np, root / "window", "csr")
        old = s["state0"]
        s = dict(s, state0=initial_event_state(old.graph, old.terrain, old.bed, ctx, old.storm.soil_water_m,
                                               control=control))
        return s, ctx, schedule
    raise ValueError(kind)


def make_hook(kind, root: Path, *, keep=None, on_state0=None, end_s=180.0):
    """A case-setup hook. `keep` (a list) receives the returned mapping when the test wants the hook to retain it;
    `on_state0(state0)` lets a test take WEAK references to the initial state (the hook itself keeps none)."""
    root = Path(root)

    def hook(*, case_dir, backend, control, end_s=None, expected_maple_root=None, mahleran_root=None,
             allow_maple_source_change=False):
        from maple_syrup.dependency import resolve_maple_dependency
        from maple_syrup.provenance import capture_maple_provenance

        xml, forcing = _files(root)
        s, ctx, schedule = build_case(kind, control, root)
        schedule = dataclasses.replace(schedule, provenance=dataclasses.replace(
            schedule.provenance, path=str(forcing), sha256=hashlib.sha256(forcing.read_bytes()).hexdigest()))
        dependency = resolve_maple_dependency()
        verified = SimpleNamespace(
            report={"recipe": {"recipe_path": str(root / "recipe" / "recipe.yaml"),
                               "mahleran_root": str(root / "mahleran")},
                    "mahleran": {"root": str(root / "mahleran")}},
            maple_dependency=dependency, maple_provenance=capture_maple_provenance(dependency),
            binding={"synthetic_case": kind}, checks={"maple_source_changed_since_import": False},
            rainfall_path=forcing)
        prepared = {"xp": np, "end": float(end_s if end_s is not None else default_end), "state0": s["state0"],
                    "context": ctx, "column": s["column"], "field": s["field"], "schedule": schedule,
                    "sediment": s["sediment"], "vegetation": s["vegetation"], "xml_path": xml,
                    "xml_sha_before": hashlib.sha256(xml.read_bytes()).hexdigest(),
                    "sediment_record": {"synthetic": kind}, "parameter_record": {"synthetic": kind},
                    "rainfall_source_path": forcing}
        mapping = {"verified": verified, "prepared": prepared, "memory_preflight": None,
                   "run_configuration": {"synthetic_case": kind}, "case_identity_sha256": "0" * 64,
                   "maple_configuration": {"synthetic_case": kind}}
        if on_state0 is not None:
            on_state0(prepared["state0"])
        if keep is not None:
            keep.append(mapping)
        return mapping

    default_end = end_s
    return hook


def run_options(**extra):
    from maple_syrup.complete_event import CompletionPolicy

    options = {"implementation": "array", "max_dt_s": 1.0, "report_every_s": 2.0, "checkpoint_every_s": 2.0,
               "max_end_s": 180.0, "policy": CompletionPolicy(hold_s=2.0), "record_exchange_history": True}
    options.update(extra)
    return options


def window_options(**extra):
    return run_options(phase_storage="csr", **extra)
