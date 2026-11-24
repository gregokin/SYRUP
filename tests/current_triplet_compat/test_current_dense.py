"""Current committed Native source admission through the genuine Dense path.

Root configures/imports the real immutable bundle before this module loads.
No Native classes, source guards or preparation functions are substituted.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import ncpcases as nc
import numpy as np
import pytest
import yaml

import maple_syrup
from maple_syrup import layered_case as layered
from maple_syrup import native_case_profile as ncp
from maple_syrup import native_dense_profile as nd
from maple_syrup import native_maple_1dca as selected
from maple_syrup.provenance import source_tree_digest


def _current(target):
    digest = source_tree_digest(Path(maple_syrup.__file__).resolve().parent).digest_sha256
    return ncp.verify_source_profile(target, profile=ncp.CURRENT_PROFILE_NAME, expected_syrup_digest=digest)


def test_current_source_is_a_separate_exact_cpu_capability(current_bundle):
    profile = _current(current_bundle)
    assert type(profile) is ncp.NativeSourceProfile
    assert profile.profile == selected.PROFILE_NAME == ncp.CURRENT_PROFILE_NAME
    assert profile.maple_commit == selected.PINS["commit"]
    assert profile.package_digest == selected.PINS["package_digest"]
    assert not profile.device_runtime_capability
    before = (profile.profile, profile.maple_commit, profile.bundle_id, profile.package_digest, profile.syrup_package_digest)
    assert profile.recheck() is None
    assert (profile.profile, profile.maple_commit, profile.bundle_id, profile.package_digest, profile.syrup_package_digest) == before
    # The old default continues to require its old exact source.
    assert ncp.NATIVE_COMMIT == "9f466d2cfcd52991b24496cf67f4562d6268afa9"
    with pytest.raises(ncp.NativeCaseProfileError, match="origin"):
        ncp.verify_source_profile(current_bundle)
    with pytest.raises(ncp.NativeCaseProfileError):
        ncp.verify_source_profile(current_bundle, profile=ncp.CURRENT_PROFILE_NAME, expected_syrup_digest="0" * 64)


def test_current_package_admission_prepares_real_dense_and_layered_case(current_bundle, tmp_path):
    profile = _current(current_bundle)
    case = tmp_path / "case"
    shutil.copytree(nc.EXAMPLE / "case", case)
    admission = ncp.admit_package_case(profile, case)
    compiled = admission.compiled
    assert compiled.leaf_vegetation is None
    assert tuple(compiled.__dataclass_fields__) == selected.COMPILED_CASE_FIELDS
    prep = nd.prepare_native_dense_case(admission, caller_reserve_bytes=1 << 20)
    assert prep.profile_name == nd.CURRENT_PROFILE_NAME == selected.DENSE_PROFILE_NAME
    assert prep.package_digest == profile.package_digest and prep.backend == "numpy"
    assert prep.bed.voxel_column.mass_kg is compiled.voxel_column.mass_kg
    assert prep.bed.active_layer.mass_kg is compiled.active_layer.mass_kg
    assert prep.shape == (4, 6) and prep.latitude_deg == 35.0
    assert prep.bed.voxel_column.mass_kg.dtype == np.float64
    water_file = tmp_path / "layered.yaml"
    water_file.write_text(yaml.safe_dump(nc.yaml_for(2)))
    config = layered.load_layered_config(water_file)
    setup = layered.prepare_layered_case(prep, config, caller_reserve_bytes=1 << 20,
        lifecycle_extra_reserve_bytes=1 << 20, available_bytes_fn=lambda: 1 << 40)
    assert setup.lifecycle_operator.shape == prep.shape
    assert setup.lifecycle_operator.n_layers == 2
    assert setup.state.t_s == 0.0
    assert admission.latitude_binding()["value"] == 35.0
