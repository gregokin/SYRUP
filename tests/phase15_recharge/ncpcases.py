"""Genuine native file-case factory for the S2 daily controls (CPU). NOTHING here is faked: the materialized MAPLE9f bundle is verified by the real
`native_case_profile.verify_source_profile` (cold content/origin/archive/distribution/census), the package case is admitted by
`ncp.admit_package_case`, the Dense case is prepared by `native_dense_profile.prepare_native_dense_case`, and the layered case/lifecycle come
from `layered_case.prepare_layered_case` driven by a YAML file. The bundle path is EXPLICIT configuration (`SYRUP_NCP_TARGET`); without it (or
when this process did not import MAPLE from that bundle) the callers skip strictly. Inputs are copied into an owned directory; the original public
case/YAML files are never edited. A matching SYRUP tree digest proves DEVELOPMENT consistency only (no N15 selection)."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

CANDIDATE = Path(__file__).resolve().parents[2]
EXAMPLE = CANDIDATE / "docs" / "manuals" / "examples" / "native_leaf_calendar_cli"
THICK = 0.0625


class ConfiguredBundleError(RuntimeError):
    """`SYRUP_NCP_TARGET` IS configured but unusable (missing, not a directory, wrong MAPLE import origin): a setup FAILURE of the genuine source
    guard, never a skip."""


def skip_reason() -> str | None:
    """A reason to skip ONLY when `SYRUP_NCP_TARGET` is UNSET (the opt-in was not requested). A configured target is never skipped away."""
    if os.environ.get("SYRUP_NCP_TARGET") is None or os.environ.get("SYRUP_NCP_TARGET") == "":
        return "SYRUP_NCP_TARGET (the materialized MAPLE9f bundle) is not configured: the genuine NCP/file-case factory is not exercised (no claim, no substitute)"
    return None


def require_configured_bundle() -> Path:
    """For a CONFIGURED target: fail clearly when it is missing, not a directory, or when this process did not import MAPLE from it (the genuine
    runtime/source check would refuse). Never imports another MAPLE tree and never substitutes a class."""
    target = os.environ.get("SYRUP_NCP_TARGET")
    if not target:
        raise ConfiguredBundleError("SYRUP_NCP_TARGET is unset")
    path = Path(target)
    if not path.is_dir():
        raise ConfiguredBundleError(f"SYRUP_NCP_TARGET {target!r} is missing or not a directory (setup failure, not a skip)")
    try:
        import maple
    except ImportError as exc:
        raise ConfiguredBundleError(f"maple is not importable although SYRUP_NCP_TARGET is configured: {exc}") from None
    if path.resolve() not in Path(maple.__file__).resolve().parents:
        raise ConfiguredBundleError(f"this process imported MAPLE from {maple.__file__}, not from the configured bundle {target}: wrong origin (setup failure, not a skip)")
    return path.resolve()


def yaml_for(n_layers: int, *, bottom: str = "open", theta: float | list = 0.25, wetting_m: float = 0.001, cumulative_m: float = 0.002) -> dict:
    raw = yaml.safe_load((EXAMPLE / "layered_water.yaml").read_text())
    soil = raw["soil"]
    soil["layers"]["thickness_m"] = [THICK] * n_layers
    for key, value in (("theta_residual", 0.0625), ("theta_field_capacity", 0.25), ("theta_saturation", 0.5)):
        soil[key] = [value] * n_layers
    soil["bottom"] = bottom
    pref = raw["preferential"]
    weights = [0.0] * (n_layers - 1) + [1.0]
    pref["background_receiver_weights"] = list(weights)
    pref["species"][0]["receiver_weights"] = list(weights)
    raw["initial"]["soil_moisture"]["by_layer"] = [theta] * n_layers if not isinstance(theta, list) else list(theta)
    raw["initial"]["wetting_history_m"] = wetting_m
    raw["initial"]["cumulative_intake_m"] = cumulative_m
    return raw


def build_file_case(tmp: Path, raw: dict, *, reserve: int = 1 << 20, flat_closed: bool = False, nz: int | None = None) -> SimpleNamespace:
    """Genuine verified/admitted/prepared native Dense case + layered lifecycle from a YAML FILE written into the owned `tmp`.

    `flat_closed=True` is the explicitly labeled TEST-INPUT selector of correction02: on the OWNED COPIES (the public example files are never edited), BEFORE the
    real NCP admission, `topography.perturbation.relief_m` of the copied native `case.yaml` is set to 0 (native flat base, the same bed thickness `nz`) and the
    owned layered YAML gets `domain.outlets = []` (an enclosed, flat, closed surface domain through the existing builder). No model default, quiet gate or
    class is changed; the same genuine NCP/Case/Dense/layered factories run."""
    target = require_configured_bundle()
    from maple_syrup import layered_case as lc
    from maple_syrup import native_case_profile as ncp
    from maple_syrup import native_dense_profile as nd
    from maple_syrup.provenance import source_tree_digest
    import maple_syrup

    digest = source_tree_digest(Path(maple_syrup.__file__).resolve().parent).digest_sha256
    profile = ncp.verify_source_profile(target, expected_syrup_digest=digest)
    case_dir = tmp / "case"
    if not case_dir.exists():
        shutil.copytree(EXAMPLE / "case", case_dir)
        if flat_closed or nz is not None:
            case_path = case_dir / "case.yaml"
            case = yaml.safe_load(case_path.read_text())
            if flat_closed:
                case["topography"]["perturbation"]["relief_m"] = 0.0
            if nz is not None:  # EXPLICIT owned test geometry: a genuinely deeper native column (never a truncated/relabeled texture check)
                case["topography"]["nz"] = int(nz)
            case_path.write_text(yaml.safe_dump(case))
    if flat_closed:
        raw = {**raw, "domain": {**raw["domain"], "outlets": []}}
    admission = ncp.admit_package_case(profile, case_dir)
    prep = nd.prepare_native_dense_case(admission, caller_reserve_bytes=reserve)
    path = tmp / f"layered_{abs(hash(yaml.safe_dump(raw))) % 10**8}.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = lc.load_layered_config(path)
    box = SimpleNamespace(value=None)

    def free():
        from maple_syrup.subsurface import default_host_available_bytes
        return default_host_available_bytes() if box.value is None else box.value

    setup = lc.prepare_layered_case(prep, cfg, caller_reserve_bytes=reserve, lifecycle_extra_reserve_bytes=reserve, available_bytes_fn=free)
    return SimpleNamespace(admission=admission, prep=prep, cfg=cfg, setup=setup, lop=setup.lifecycle_operator, s0=setup.state, n=setup.lifecycle_operator.n_cells, L=setup.lifecycle_operator.n_layers,
                           profile=profile, syrup_digest=digest, case_dir=case_dir, quota=box, shape=setup.lifecycle_operator.shape)


#: Explicit owned test-case geometry of the 100 mm-layer known-answer fixtures: the public example column (`topography.nz: 4`) represents only ~0.19 m on its
#: thinnest active cells, so a 2 x 0.1 m or 3 x 0.1 m profile is (correctly) REFUSED by the genuine native soil-texture factory. These controls therefore use an OWNED copy of
#: the native case with a genuinely deeper column (`nz = 8`); no texture check is disabled or truncated and the public example is never edited.
NZ_DEEP = 8


def with_inactive_cells(tmp: Path, raw: dict, cells, shape=(4, 6)) -> dict:
    """A configured REAL active mask for the owned layered YAML: a bool `.npy` map written next to the YAML (the parameter file's directory) with the given
    (row, col) cells inactive, referenced as `domain.active_mask: {file: ...}`."""
    import numpy as np

    mask = np.ones(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = False
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "active_mask.npy", mask)
    return {**raw, "domain": {**raw["domain"], "active_mask": {"file": "active_mask.npy"}}}
