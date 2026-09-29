"""The MAPLE dependency resolves to the real MAPLE and mismatches are refused."""

from __future__ import annotations

from pathlib import Path

import pytest

from maple_syrup.dependency import (
    EXPECTED_MAPLE_ROOT_ENV,
    ApiRequirement,
    MapleApiIncompatibleError,
    MapleDependencyError,
    check_required_api,
    resolve_maple_dependency,
)


def test_resolves_the_imported_maple_and_identifies_it():
    import maple

    dependency = resolve_maple_dependency()

    assert dependency.package_dir == Path(maple.__file__).resolve().parent
    # Identity comes from MAPLE's own project metadata, not merely the
    # importable name that an unrelated PyPI `maple` would also satisfy.
    assert dependency.project_name == "maple"
    assert (dependency.package_dir / "water" / "step.py").is_file()
    if dependency.source_kind == "editable_working_tree":
        assert dependency.distribution_name == "maple"
        assert dependency.direct_url["dir_info"]["editable"] is True
        assert dependency.package_dir.is_relative_to(dependency.source_root)


def test_matching_expected_root_is_accepted():
    dependency = resolve_maple_dependency()
    if dependency.source_root is None:
        pytest.skip("imported MAPLE has no identifiable source root")
    assert resolve_maple_dependency(dependency.source_root).source_root == dependency.source_root


def test_mismatched_expected_root_is_refused(tmp_path, monkeypatch):
    with pytest.raises(MapleDependencyError, match="expected MAPLE source root"):
        resolve_maple_dependency(tmp_path)

    monkeypatch.setenv(EXPECTED_MAPLE_ROOT_ENV, str(tmp_path))
    with pytest.raises(MapleDependencyError, match="expected MAPLE source root"):
        resolve_maple_dependency()


def test_every_missing_api_element_is_reported_together():
    requirements = (
        ApiRequirement("maple.water", "apply_water_process_demand", parameters=("no_such_parameter",)),
        ApiRequirement("maple.water", "no_such_symbol"),
        ApiRequirement("maple.no_such_module", "anything"),
        ApiRequirement(
            "maple.core.parameters.water_coupling", "DEPTH_UPDATE_RULES",
            contains=("no_such_rule",),
        ),
    )
    with pytest.raises(MapleApiIncompatibleError) as excinfo:
        check_required_api(requirements)
    message = str(excinfo.value)
    assert "no_such_parameter" in message
    assert "no_such_symbol" in message
    assert "maple.no_such_module" in message
    assert "no_such_rule" in message
