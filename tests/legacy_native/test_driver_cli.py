"""CLI/driver refusals and the FAILED publication rule (no model run needed)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from maple_syrup import legacy_driver as D


def argv(tmp_path, *extra, kind="plot1", case=None, out="out"):
    return ["--case-kind", kind, "--case", str(case or tmp_path / "no_case"), "--output", str(tmp_path / out),
            "--allow-python-kernels", *extra]


def test_parser_defaults_are_the_legacy_workflow():
    args = D.build_parser().parse_args(["--case-kind", "rfid", "--case", "c", "--output", "o"])
    assert args.depth_time_level == "previous" and args.source_order == "index" and not args.legacy_depos_erase
    assert args.root_solver == "bisection" and args.bisection_iterations is None and args.warmup_s == 0.0


def test_bad_case_publishes_failed_record_and_no_output(tmp_path):
    assert D.main(argv(tmp_path)) == 1
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.partial").exists()
    record = json.loads((tmp_path / "out.FAILED" / "FAILED.json").read_text())
    assert record["error"] and "traceback" in record


def test_existing_output_or_failed_directory_is_refused_without_touching_it(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep").write_text("x")
    assert D.main(argv(tmp_path)) == 1
    assert (tmp_path / "out" / "keep").read_text() == "x" and not (tmp_path / "out.FAILED").exists()
    (tmp_path / "again.FAILED").mkdir()
    assert D.main(argv(tmp_path, out="again")) == 1


def test_output_inside_the_case_directory_is_refused(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    assert D.main(argv(tmp_path, case=case, out="case/out")) == 1
    assert not (case / "out.FAILED").exists()


@pytest.fixture
def no_case_adapter(monkeypatch):
    calls = []

    def fail(*a, **k):
        calls.append(a)
        raise AssertionError("the case adapter must not be called")

    monkeypatch.setattr(D, "legacy_case_for", fail)
    monkeypatch.setattr(D, "_maple_roots", list)
    return calls


@pytest.mark.parametrize("extra", [["--end-s", "nan"], ["--end-s", "-1"], ["--end-s", "0"], ["--end-s", "10.5"],
                                   ["--end-s", "inf"], ["--warmup-s", "-1"], ["--warmup-s", "2.5"], ["--warmup-s", "nan"],
                                   ["--max-memory-gib", "0"], ["--max-memory-gib", "nan"], ["--max-memory-gib", "-4"],
                                   ["--progress-every-s", "-1"], ["--progress-every-s", "nan"],
                                   ["--bisection-iterations", "0"], ["--bisection-iterations", "-3"],
                                   ["--root-solver", "newton", "--newton-max-iterations", "0"],
                                   ["--snapshot-times", "5,nan"], ["--snapshot-times", "0"],
                                   ["--end-s", "10", "--snapshot-times", "11"], ["--legacy-depos-erase"]])
def test_invalid_controls_are_refused_before_adapter_allocation_or_any_write(tmp_path, no_case_adapter, extra):
    assert D.main(argv(tmp_path, *extra)) == 1
    assert not no_case_adapter  # the case adapter was never reached
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.partial").exists() and not (tmp_path / "out.FAILED").exists()


def test_explicit_zero_controls_are_not_replaced_by_defaults(tmp_path):
    args = D.build_parser().parse_args(["--case-kind", "rfid", "--case", "c", "--output", "o", "--bisection-iterations", "0"])
    assert args.bisection_iterations == 0
    with pytest.raises(D.DriverError, match="hydrology control"):
        D.validate_controls(args)


@pytest.mark.parametrize("where", ["src", "src/new_out", "tests/x", "benchmarks", "cases/y"])
def test_output_inside_project_source_trees_is_refused_without_creating_anything(tmp_path, no_case_adapter, where):
    repo = Path(D.__file__).resolve().parents[2]
    target = repo / where
    existed = target.exists()
    assert D.main(["--case-kind", "plot1", "--case", str(tmp_path / "c"), "--output", str(target),
                   "--allow-python-kernels"]) == 1
    assert target.exists() == existed and not no_case_adapter
    for sibling in (target.with_name(target.name + ".partial"), target.with_name(target.name + ".FAILED")):
        assert not sibling.exists()


def test_output_that_contains_a_protected_tree_is_refused(tmp_path, no_case_adapter):
    repo = Path(D.__file__).resolve().parents[2]
    assert D.main(["--case-kind", "plot1", "--case", str(tmp_path / "c"), "--output", str(repo.parent),
                   "--allow-python-kernels"]) == 1  # an ancestor of src/tests/...
    assert not no_case_adapter


def test_reference_and_dependency_trees_are_protected_including_symlinks(tmp_path, monkeypatch, no_case_adapter):
    reference = tmp_path / "mahleran"
    dependency = tmp_path / "maple"
    reference.mkdir()
    dependency.mkdir()
    monkeypatch.setenv("MAPLE_SYRUP_MAHLERAN_ROOT", str(reference))
    monkeypatch.setattr(D, "_maple_roots", lambda: [dependency])
    link = tmp_path / "link"
    link.symlink_to(dependency, target_is_directory=True)
    for target in (reference / "out", dependency / "pkg" / "out", link / "out", reference):
        assert D.main(["--case-kind", "plot1", "--case", str(tmp_path / "c"), "--output", str(target),
                       "--allow-python-kernels"]) == 1
        assert not (target.with_name(target.name + ".FAILED")).exists() and not target.with_name(target.name + ".partial").exists()
    assert not (reference / "out").exists() and not (dependency / "pkg").exists() and not no_case_adapter


def test_failure_does_not_write_a_marker_into_a_tree_protected_only_after_the_case_is_known(tmp_path, monkeypatch):
    """A root discovered while loading the case (simulated: the output itself becomes protected) must stop even the marker."""
    monkeypatch.setattr(D, "_maple_roots", list)

    def fail_after_loading(*a, **k):
        raise RuntimeError("case failure")

    monkeypatch.setattr(D, "legacy_case_for", fail_after_loading)
    original = D.check_output

    def spy(out, roots, **kw):
        if kw.get("require_absent") is False:
            roots["late"] = out  # a bound input tree found while the case was loaded
        return original(out, roots, **kw)

    monkeypatch.setattr(D, "check_output", spy)
    assert D.main(argv(tmp_path)) == 1
    assert not (tmp_path / "out.FAILED").exists() and not (tmp_path / "out.partial").exists()


def test_module_provenance_hashes_real_files_and_does_not_depend_on_sys_modules(monkeypatch):
    import hashlib
    import sys

    monkeypatch.delitem(sys.modules, "maple_syrup.legacy_driver", raising=False)  # as under `python -m` (the module is __main__)
    digests = D.module_digests()
    assert set(digests) == set(D._PROVENANCE_MODULES)
    assert digests["maple_syrup.legacy_driver"] == hashlib.sha256(Path(D.__file__).read_bytes()).hexdigest()
    assert D.module_digests() == digests  # deterministic


def test_post_infiltration_depth_level_is_an_explicit_native_option_and_never_the_default():
    parser = D.build_parser()
    base = ["--case-kind", "rfid", "--case", "c", "--output", "o"]
    assert parser.parse_args(base).depth_time_level == "previous"  # the accepted default is preserved
    assert parser.parse_args([*base, "--depth-time-level", "post_infiltration"]).depth_time_level == "post_infiltration"
    assert D.DEPTH_TIME_LEVELS == ("previous", "current", "post_infiltration")
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--depth-time-level", "somewhere_else"])
