"""Resolve the MAPLE dependency and verify that it is the real MAPLE.

MAPLE's distribution name is `maple`, which is also the name of an
unrelated project on PyPI, and MAPLE is not published anywhere. MAPLE-SYRUP
therefore does not declare it as a PEP 508 requirement (see
pyproject.toml); it resolves whatever `import maple` finds in the running
environment and refuses to continue unless all of the following hold:

1. the imported module and the distribution metadata describe the same
   files (an editable install's `direct_url.json` root contains the
   imported package; a regular install's recorded `maple/__init__.py` is the
   imported one), so a shadowing copy on `sys.path` is detected;
2. the source root's `pyproject.toml` names the project `maple` when a
   source root exists, and a path import with neither distribution metadata
   nor such a `pyproject.toml` is refused as unidentifiable;
3. an optional expected source root (argument or
   `MAPLE_SYRUP_EXPECTED_MAPLE_ROOT`) matches exactly;
4. every symbol in `REQUIRED_MAPLE_API` exists, with the named parameters
   and required constant values.

Item 4 checks only the specific surface MAPLE-SYRUP calls or relies on in
its interface contract. Passing it is not a claim of general API
compatibility with any other MAPLE revision.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

__all__ = [
    "EXPECTED_DISTRIBUTION_NAME",
    "EXPECTED_MAPLE_ROOT_ENV",
    "REQUIRED_MAPLE_API",
    "ApiRequirement",
    "MapleApiIncompatibleError",
    "MapleDependency",
    "MapleDependencyError",
    "check_required_api",
    "resolve_maple_dependency",
]

EXPECTED_DISTRIBUTION_NAME = "maple"
EXPECTED_MAPLE_ROOT_ENV = "MAPLE_SYRUP_EXPECTED_MAPLE_ROOT"


class MapleDependencyError(RuntimeError):
    """The importable `maple` is missing, unidentifiable, or not the
    expected MAPLE source."""


class MapleApiIncompatibleError(MapleDependencyError):
    """The imported MAPLE lacks part of the API surface MAPLE-SYRUP uses."""


@dataclass(frozen=True)
class ApiRequirement:
    """One symbol MAPLE-SYRUP uses. `parameters` must all appear in the
    callable's signature; `contains` values must all be members of the
    attribute (used for registries such as `DEPTH_UPDATE_RULES`)."""

    module: str
    attribute: str
    parameters: tuple[str, ...] = ()
    contains: tuple[Any, ...] = ()


# The surface used by the Phase 1 probe plus the entry points the interface
# contract (docs/phase1/interface_contract.md) builds on. Extend this when a
# later phase starts calling something new; do not add symbols speculatively.
REQUIRED_MAPLE_API: tuple[ApiRequirement, ...] = (
    ApiRequirement("maple.core.boundaries", "AxisBoundary"),
    ApiRequirement("maple.core.boundaries", "BoundaryKind"),
    ApiRequirement("maple.core.parameters.geometry", "GeometrySpec"),
    ApiRequirement("maple.core.parameters.geometry", "validate_geometry"),
    ApiRequirement("maple.core.parameters.grain_classes", "GrainClass"),
    ApiRequirement("maple.core.parameters.grain_classes", "GrainClassSet"),
    ApiRequirement("maple.core.parameters.grain_classes", "validate_grain_classes"),
    ApiRequirement("maple.core.parameters.numerics", "DEFAULT_MASS_RESOLUTION_KG"),
    ApiRequirement(
        "maple.core.parameters.water_coupling", "DEPTH_UPDATE_RULES",
        contains=("constant_depth",),
    ),
    ApiRequirement("maple.core.parameters.water_coupling", "MAHLERAN_1_2_1_CLASS_DIAMETERS_M"),
    ApiRequirement("maple.core.types.voxel", "zeros_voxel_column_state"),
    ApiRequirement(
        "maple.surface.voxels", "deposit_surface_mixture_batch",
        parameters=("column", "request_by_class_kg", "geometry", "mass_resolution_kg"),
    ),
    ApiRequirement(
        "maple.surface.active_layer", "initialize_active_layer_from_voxels",
        parameters=("voxel_column", "geometry", "mass_resolution_kg"),
    ),
    ApiRequirement("maple.surface.active_layer", "check_active_layer_voxel_partition"),
    ApiRequirement("maple.core.types.water", "WaterState"),
    ApiRequirement("maple.core.types.water", "water_state_from_depth"),
    ApiRequirement("maple.core.types.water", "water_state_content_sha256"),
    ApiRequirement(
        "maple.core.types.sediment_ledger", "zeros_sediment_ledger_state",
        parameters=("namespace",),
    ),
    ApiRequirement(
        "maple.water", "apply_water_process_demand",
        parameters=(
            "voxel_column", "active_layer", "water", "ledger", "demand", "geometry",
            "grain_classes", "mass_resolution_kg", "sediment_availability",
            "initial_available_fraction", "adapter_name",
        ),
    ),
    ApiRequirement("maple.water", "zero_water_demand", parameters=("namespace",)),
    ApiRequirement("maple.water", "WaterProcessDemand"),
    ApiRequirement("maple.water", "validate_water_state"),
    ApiRequirement("maple.water", "WaterStateValidationError"),
    ApiRequirement(
        "maple.water", "apply_depth_after_bed_change",
        parameters=("depth_before_m", "delta_z_bed_m", "geometry", "rule"),
    ),
    ApiRequirement(
        "maple.water.commit_callback", "make_water_depth_commit_callback",
        parameters=("depth_update_rule",),
    ),
    ApiRequirement(
        "maple.surface.topographic_commit.commit", "commit_topography",
        parameters=("water_callback", "routing_callback"),
    ),
    ApiRequirement("maple.coupling.water_event", "advance_fluvial_step", parameters=("demand",)),
    ApiRequirement("maple.io.outputs.snapshot", "save_state_snapshot", parameters=("water",)),
    ApiRequirement("maple.core.backend", "resolve_backend", parameters=("backend", "device_id")),
    ApiRequirement("maple.core.backend", "BackendUnavailableError"),
    ApiRequirement("maple.core.backend", "to_device_tree"),
    ApiRequirement("maple.core.backend", "to_host_tree"),
    ApiRequirement("maple.core.backend", "cupy_available"),
)


@dataclass(frozen=True)
class MapleDependency:
    """Where the imported MAPLE came from.

    `source_kind` is one of:

    - `"editable_working_tree"`: an editable install; the imported code is
      the live, mutable content of `source_root` at import time.
    - `"installed_distribution"`: a regular installed distribution.
    - `"path_import"`: found through `sys.path` (for example `PYTHONPATH`)
      with no matching distribution metadata; identified only by the
      source root's `pyproject.toml`.
    """

    package_dir: Path
    source_root: Path | None
    source_kind: str
    distribution_name: str | None
    distribution_version: str | None
    distribution_summary: str | None
    direct_url: dict[str, Any] | None
    project_name: str | None
    project_version: str | None

    def as_record(self) -> dict[str, Any]:
        return {
            "package_dir": str(self.package_dir),
            "source_root": str(self.source_root) if self.source_root is not None else None,
            "source_kind": self.source_kind,
            "distribution_name": self.distribution_name,
            "distribution_version": self.distribution_version,
            "distribution_summary": self.distribution_summary,
            "direct_url": self.direct_url,
            "project_name": self.project_name,
            "project_version": self.project_version,
        }


def check_required_api(requirements: tuple[ApiRequirement, ...] = REQUIRED_MAPLE_API) -> None:
    """Raise `MapleApiIncompatibleError` listing every unmet requirement."""
    violations: list[str] = []
    for req in requirements:
        label = f"{req.module}.{req.attribute}"
        try:
            module = importlib.import_module(req.module)
        except Exception as exc:  # noqa: BLE001 - reported verbatim
            violations.append(f"{label}: module import failed ({type(exc).__name__}: {exc})")
            continue
        if not hasattr(module, req.attribute):
            violations.append(f"{label}: attribute missing")
            continue
        obj = getattr(module, req.attribute)
        if req.parameters:
            try:
                present = set(inspect.signature(obj).parameters)
            except (TypeError, ValueError) as exc:
                violations.append(f"{label}: signature unavailable ({exc})")
                continue
            missing = [name for name in req.parameters if name not in present]
            if missing:
                violations.append(f"{label}: missing parameter(s) {missing}")
        for value in req.contains:
            try:
                ok = value in obj
            except TypeError:
                ok = False
            if not ok:
                violations.append(f"{label}: does not contain required value {value!r}")
    if violations:
        raise MapleApiIncompatibleError(
            "The imported MAPLE does not provide the API surface MAPLE-SYRUP requires:\n  - "
            + "\n  - ".join(violations)
        )


def _read_project_table(source_root: Path) -> dict[str, Any] | None:
    pyproject = source_root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    with pyproject.open("rb") as handle:
        return tomllib.load(handle).get("project", {})


def _file_url_path(url: str) -> Path | None:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path)).resolve()


def resolve_maple_dependency(
    expected_source_root: str | Path | None = None,
    *,
    requirements: tuple[ApiRequirement, ...] = REQUIRED_MAPLE_API,
) -> MapleDependency:
    """Import MAPLE, identify its source, and verify it.

    `expected_source_root` defaults to `$MAPLE_SYRUP_EXPECTED_MAPLE_ROOT`
    when set. Raises `MapleDependencyError` (or its API subclass) on any
    failed check; never falls back to a different MAPLE.
    """
    try:
        maple = importlib.import_module("maple")
    except Exception as exc:
        raise MapleDependencyError(
            f"`import maple` failed ({type(exc).__name__}: {exc}). Run with the MAPLE "
            "environment, e.g. /home/okin/MAPLE/.venv/bin/python (see README.md)."
        ) from exc
    module_file = getattr(maple, "__file__", None)
    if module_file is None:
        raise MapleDependencyError("the imported `maple` has no __file__ (namespace package?)")
    package_dir = Path(module_file).resolve().parent

    try:
        dist = importlib.metadata.distribution(EXPECTED_DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError:
        dist = None

    direct_url: dict[str, Any] | None = None
    source_root: Path | None = None
    if dist is not None:
        raw_direct_url = dist.read_text("direct_url.json")
        direct_url = json.loads(raw_direct_url) if raw_direct_url else None
        editable = bool(direct_url and direct_url.get("dir_info", {}).get("editable"))
        if editable:
            source_root = _file_url_path(direct_url["url"])
            if source_root is None or not package_dir.is_relative_to(source_root):
                raise MapleDependencyError(
                    f"imported maple at {package_dir} is not inside the editable install root "
                    f"{direct_url.get('url')!r} recorded by the `maple` distribution; another "
                    "copy is shadowing it on sys.path"
                )
            source_kind = "editable_working_tree"
        else:
            recorded = Path(dist.locate_file("maple/__init__.py")).resolve().parent
            if recorded != package_dir:
                raise MapleDependencyError(
                    f"imported maple at {package_dir} differs from the installed `maple` "
                    f"distribution's files at {recorded}"
                )
            source_kind = "installed_distribution"
    else:
        source_kind = "path_import"
        if package_dir.parent.name == "src":
            source_root = package_dir.parent.parent

    project = _read_project_table(source_root) if source_root is not None else None
    if project is not None and project.get("name") != EXPECTED_DISTRIBUTION_NAME:
        raise MapleDependencyError(
            f"{source_root / 'pyproject.toml'} names project {project.get('name')!r}, "
            f"not {EXPECTED_DISTRIBUTION_NAME!r}"
        )
    if dist is None and project is None:
        raise MapleDependencyError(
            f"imported maple at {package_dir} has neither distribution metadata nor a source "
            "pyproject.toml naming project 'maple'; it cannot be identified as MAPLE"
        )

    if expected_source_root is None:
        expected_source_root = os.environ.get(EXPECTED_MAPLE_ROOT_ENV) or None
    if expected_source_root is not None:
        expected = Path(expected_source_root).resolve()
        if source_root is None or source_root.resolve() != expected:
            raise MapleDependencyError(
                f"expected MAPLE source root {expected}, but the imported maple resolves to "
                f"source root {source_root} (package {package_dir})"
            )

    check_required_api(requirements)

    return MapleDependency(
        package_dir=package_dir,
        source_root=source_root,
        source_kind=source_kind,
        distribution_name=dist.metadata["Name"] if dist is not None else None,
        distribution_version=dist.version if dist is not None else None,
        distribution_summary=dist.metadata.get("Summary") if dist is not None else None,
        direct_url=direct_url,
        project_name=project.get("name") if project is not None else None,
        project_version=project.get("version") if project is not None else None,
    )
