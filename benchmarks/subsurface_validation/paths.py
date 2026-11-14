"""Locations of the pinned inputs. Defaults are those of a NORMAL checkout: `<repository>/agent_handoffs/tasks/<task>/...`. Every location can be overridden by an environment variable, and an isolated worktree or an
archive MUST set them (Root sets all of them for isolated runs). There is no fallback to another dataset or location: a missing file fails its pin check, and the immutable sha256 pins in the config remain the authority.

Environment variables (all optional; absolute paths):
  SYRUP_PHASE3_TASKS_ROOT     directory holding the task folders (default `<repository>/agent_handoffs/tasks`)
  SYRUP_PHASE3_TASK_DIR       this task's folder with the retrieved air/megapit data (default `<TASKS_ROOT>/subsurface_observed_drydown_20261114`)
  SYRUP_PHASE3_RAW_ROOT       the JORN/MOAB RELEASE-2026 raw tree `jorn_three_layer_20261102` (default `<TASKS_ROOT>/jorn_three_layer_20261102`)
  SYRUP_PHASE3_DEPTH_TABLE    the pinned corrected-depth file `swc_depthsV3` (default `<TASKS_ROOT>/nevada_zero_flux_20261101/neon_metadata/swc_depthsV3`)
  SYRUP_PHASE3_AIR_JORN, SYRUP_PHASE3_AIR_MOAB, SYRUP_PHASE3_MEGAPIT_MOAB   the retrieval directories (default `<TASK_DIR>/downloads_air_JORN` etc.)
  SYRUP_PHASE3_INVENTORY      `phase3_observation_inventory.json` (default `<repository>/docs/subsurface_hydrology/phase3_observation_inventory.json`)
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2]                      # the checkout that contains `benchmarks/`
CANDIDATE = REPO                             # historical name
TASKS = Path(os.environ.get("SYRUP_PHASE3_TASKS_ROOT", str(REPO / "agent_handoffs" / "tasks")))
TASK = Path(os.environ.get("SYRUP_PHASE3_TASK_DIR", str(TASKS / "subsurface_observed_drydown_20261114")))
RAW_ROOT = Path(os.environ.get("SYRUP_PHASE3_RAW_ROOT", str(TASKS / "jorn_three_layer_20261102")))
DEPTH_TABLE = Path(os.environ.get("SYRUP_PHASE3_DEPTH_TABLE", str(TASKS / "nevada_zero_flux_20261101" / "neon_metadata" / "swc_depthsV3")))
AIR_JORN = Path(os.environ.get("SYRUP_PHASE3_AIR_JORN", str(TASK / "downloads_air_JORN")))
AIR_MOAB = Path(os.environ.get("SYRUP_PHASE3_AIR_MOAB", str(TASK / "downloads_air_MOAB")))
MEGAPIT_MOAB = Path(os.environ.get("SYRUP_PHASE3_MEGAPIT_MOAB", str(TASK / "downloads_megapit_MOAB")))
INVENTORY = Path(os.environ.get("SYRUP_PHASE3_INVENTORY", str(REPO / "docs" / "subsurface_hydrology" / "phase3_observation_inventory.json")))
ENV_VARS = ("SYRUP_PHASE3_TASKS_ROOT", "SYRUP_PHASE3_TASK_DIR", "SYRUP_PHASE3_RAW_ROOT", "SYRUP_PHASE3_DEPTH_TABLE", "SYRUP_PHASE3_AIR_JORN", "SYRUP_PHASE3_AIR_MOAB", "SYRUP_PHASE3_MEGAPIT_MOAB", "SYRUP_PHASE3_INVENTORY")
