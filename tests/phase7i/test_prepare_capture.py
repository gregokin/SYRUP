"""The capture derivative: hook placement, exact reversibility (so no model line is changed), a static read-only audit of
the appended Fortran, provenance/manifest preservation and unsafe-input refusal. Nothing here compiles or runs
Fortran. Nothing here was run by its author (file-only tools); Codex records actual results."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import prepare_capture as pc
import pytest
from prepare_mahleran import inventory

ROOT = Path(__file__).resolve().parents[2]
REAL_PARENT = ROOT / "outputs/phase7/mahleran_plot1_no_splash_linux"
MINI_STORM = (
    "subroutine MAHLERAN_storm_xml\n"
    "write (6, *) ' About to start Mahleran Storm, iout = ', iout\n"  # no leading blanks, as in the actual parent
    "do iter = 1, nit\n"
    "   write (6, 9999) iter, rval * 3600., dt, t, Julian, istart, q_plot * dx, sed_plot\n"
    "   call infilt\n"
    "   call route_water\n"
    "   call output_hydro_data_xml\n"
    "!   write (6, *) 'Returned from output_hydro_data_xml'\n"
    "   call update_water_flow\n"
    "enddo\n"
    "close (51)\n\n!EVA2016 calculation of annual net erosion\n"
    "end\n"
)


def make_parent(root: Path, *, storm: str = MINI_STORM, distribution: str = "normal") -> Path:
    (root / "src/Program_Control").mkdir(parents=True)
    (root / "src/Program_Control/MAHLERAN_storm_xml.f90").write_bytes(storm.encode("latin-1"))
    (root / "src/Program_Control/other.f90").write_text("end\n")
    (root / "Input/input_p1").mkdir(parents=True)
    (root / "Input/input_p1/a.dat").write_text("1\n")
    (root / "nbproject").mkdir()
    (root / "nbproject/Makefile-Release.mk").write_text("# makefile\n")
    (root / "Makefile").write_text("all:\n")
    (root / "mahleran_input.xml").write_text(f'<m><finalInfiltrationRateDistribution value="{distribution}"/></m>\n')
    manifest = {"status": "prepared", "reference_root": "/reference", "original_sha256": {"mahleran_input.xml": "ab" * 32},
                "prepared_sha256": inventory(root), "case_audit": []}
    (root / "benchmark_manifest.json").write_text(json.dumps(manifest))
    return root


def call_lines(text: str) -> list[tuple[int, str]]:
    body = text.partition(pc.HOOK_SOURCE)[0]
    return [(k, ln) for k, ln in enumerate(body.splitlines()) if pc.HOOK_CALL in ln]


# --- placement and reversibility ---------------------------------------------------------------------------------
def test_hook_calls_are_inserted_at_exactly_the_four_intended_points():
    patched = pc.insert_hooks(MINI_STORM)
    lines = patched.partition(pc.HOOK_SOURCE)[0].splitlines()
    assert [ln.strip() for _, ln in call_lines(patched)] == [f"call syrup_hydro_capture ({n})" for n in range(4)]
    idx = {n: k for k, ln in call_lines(patched) for n in range(4) if ln.strip() == f"call syrup_hydro_capture ({n})"}
    assert "About to start" in lines[idx[0] - 1]  # stage 0: after setup, before step 1
    assert "write (6, 9999) iter" in lines[idx[1] - 1] and lines[idx[1] + 1].strip() == "call infilt"  # before infilt
    assert lines[idx[2] - 1].strip() == "call output_hydro_data_xml" and "Returned from" in lines[idx[2] + 1]
    assert lines[idx[3] - 1] == "enddo" and lines[idx[3] + 1] == "close (51)"  # after the last step
    assert patched.endswith(pc.HOOK_SOURCE) and patched.count(pc.HOOK_MARKER) == 1


def test_insertion_is_exactly_reversible_so_no_original_line_changes():
    patched = pc.insert_hooks(MINI_STORM)
    assert pc.remove_hooks(patched) == MINI_STORM
    kept = [ln for ln in patched.partition(pc.HOOK_SOURCE)[0].splitlines(keepends=True) if pc.HOOK_CALL not in ln]
    assert "".join(kept) == MINI_STORM


@pytest.mark.parametrize("text", [
    MINI_STORM.replace("   call output_hydro_data_xml\n", ""),
    MINI_STORM + "   call output_hydro_data_xml\n",
    MINI_STORM.replace("close (51)\n", ""),
    MINI_STORM.replace("   write (6, 9999)", "   write (6, 9998)"),
])
def test_missing_or_duplicated_anchors_are_refused(text):
    with pytest.raises(ValueError, match="exactly one"):
        pc.insert_hooks(text)


def test_an_already_hooked_parent_is_refused():
    with pytest.raises(ValueError, match="already contains"):
        pc.insert_hooks(pc.insert_hooks(MINI_STORM))


def test_appended_fortran_never_assigns_to_model_state():
    """Static read-only audit: every assignment target and allocation in the hook source is a local `a_` name; the only
    calls are the writer helpers; files are opened with status='new' (never replace) and never an existing model unit."""
    source = pc.HOOK_SOURCE
    code = [ln.split("!")[0] for ln in source.splitlines() if not ln.strip().startswith("!")]
    # join Fortran continuation lines so `status = 'new'` inside an OPEN is not mistaken for an assignment
    joined: list[str] = []
    carry = ""
    for ln in code:
        text = carry + ln.strip().lstrip("&").strip() if carry else ln
        if text.rstrip().endswith("&"):
            carry = text.rstrip()[:-1] + " "
            continue
        carry = ""
        joined.append(text)
    code = joined
    targets = set()
    for ln in code:
        match = re.match(r"^\s*([A-Za-z_]\w*)\s*(\([^)]*\))?\s*=\s*[^=]", ln)
        if match:
            targets.add(match.group(1))
    assert targets and all(t.startswith("a_") for t in targets), sorted(t for t in targets if not t.startswith("a_"))
    for ln in code:
        if re.match(r"^\s*(de)?allocate", ln):
            names = re.findall(r"(\w+)\s*\(", ln.split("allocate", 1)[1])
            assert names and all(n.startswith("a_") for n in names), ln
    called = set(re.findall(r"call\s+(\w+)", "\n".join(code)))
    assert called == {"syrup_wr_si", "syrup_wr_sd", "syrup_wr_d2", "syrup_wr_i2"}
    opens = [ln for ln in "\n".join(code).replace("&\n", " ").splitlines() if re.match(r"^\s*open", ln)]
    full = "\n".join(code)
    assert full.count("open (newunit") == 3 and full.count("status = 'new'") == 3 and "replace" not in full
    assert "rewind" not in full and "close (51)" not in full and not re.search(r"\bread\s*\(", full)
    assert opens  # the audit actually saw the open statements


def test_the_hook_reads_the_same_variables_the_application_uses_for_its_own_peak():
    """The replicated peak logic must use the application's strict-greater test on its single-precision q_plot."""
    source = pc.HOOK_SOURCE
    assert "if (q_plot .gt. a_qmax_s)" in source and "real, save :: a_qmax_s" in source
    assert "a_qmax_s = real (-999.999d0)" in source  # the legacy `data qsum_max / -999.999d0 /` start
    assert "a_rval = dble (rval)" in source  # applied rate read BEFORE infilt (stage 1), widened exactly


@pytest.mark.skipif(not (REAL_PARENT / "src/Program_Control/MAHLERAN_storm_xml.f90").is_file(),
                    reason="parent derivative not available")
def test_real_storm_driver_takes_the_four_hooks_and_is_restored_bytewise():
    text = (REAL_PARENT / pc.STORM).read_bytes().decode("latin-1")
    patched = pc.insert_hooks(text)
    assert len(call_lines(patched)) == 4
    assert pc.remove_hooks(patched) == text  # removing only the inserted calls and the appended source restores it
    assert patched.count("call syrup_hydro_capture") == 4


# --- preparation -------------------------------------------------------------------------------------------------------
def test_prepare_changes_only_the_storm_driver_and_preserves_provenance(tmp_path):
    parent = make_parent(tmp_path / "parent")
    before = inventory(parent)
    manifest_before = (parent / "benchmark_manifest.json").read_bytes()
    out = tmp_path / "derivative"
    record = pc.prepare(parent, out, expected_xml_sha256=None)
    assert record["changed_vs_parent"] == [str(pc.STORM)]
    assert inventory(parent) == before and (parent / "benchmark_manifest.json").read_bytes() == manifest_before
    on_disk = json.loads((out / "benchmark_manifest.json").read_text())
    assert on_disk["original_sha256"] == {"mahleran_input.xml": "ab" * 32} and on_disk["reference_root"] == "/reference"
    assert on_disk["parent_prepared_sha256"] == before and on_disk["prepared_sha256"] == inventory(out)
    assert on_disk["hydrology_capture"] is True and on_disk["copy_root"] == str(out.resolve())
    assert [k for k in before if before[k] != on_disk["prepared_sha256"][k]] == [str(pc.STORM)]
    patched = (out / pc.STORM).read_bytes().decode("latin-1")
    assert pc.remove_hooks(patched) == MINI_STORM
    patch = (out / "syrup_hydro_capture.patch").read_text(encoding="latin-1")
    assert on_disk["patch_sha256"] == hashlib.sha256(patch.encode("latin-1")).hexdigest()
    assert patch.count("+   call syrup_hydro_capture") == 4 and "\n-" not in patch.split("@@", 1)[1].replace("\n---", "")
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]  # no staging directory left behind


def test_unchanged_sources_are_checked_against_the_earlier_build(tmp_path):
    parent = make_parent(tmp_path / "parent")
    build = {"source_sha256": {name: hashlib.sha256((parent / name).read_bytes()).hexdigest() for name in (
        "src/Program_Control/other.f90", str(pc.STORM))}}
    (tmp_path / "build.json").write_text(json.dumps(build))
    record = pc.prepare(parent, tmp_path / "ok", expected_xml_sha256=None, parent_build=tmp_path / "build.json")
    assert record["unchanged_files_match_earlier_build"] is True
    build["source_sha256"]["src/Program_Control/other.f90"] = "00" * 32
    (tmp_path / "bad.json").write_text(json.dumps(build))
    with pytest.raises(ValueError, match="differs from the earlier build"):
        pc.prepare(parent, tmp_path / "bad", expected_xml_sha256=None, parent_build=tmp_path / "bad.json")
    assert not (tmp_path / "bad").exists() and not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_preparation_refuses_the_wrong_case_and_unsafe_paths(tmp_path):
    parent = make_parent(tmp_path / "parent")
    with pytest.raises(ValueError, match="not the heterogeneous"):  # default pin: this XML is not the earlier run's
        pc.prepare(parent, tmp_path / "pinned")
    deterministic = make_parent(tmp_path / "det", distribution="deterministic")
    with pytest.raises(ValueError, match="'normal'"):
        pc.prepare(deterministic, tmp_path / "x", expected_xml_sha256=None)
    (tmp_path / "exists").mkdir()
    with pytest.raises(ValueError, match="new output"):
        pc.prepare(parent, tmp_path / "exists", expected_xml_sha256=None)
    with pytest.raises(ValueError, match="new output"):
        pc.prepare(parent, parent / "inside", expected_xml_sha256=None)
    with pytest.raises(ValueError, match="new output"):
        pc.prepare(parent, tmp_path, expected_xml_sha256=None)
    (parent / "src/Program_Control/other.f90").write_text("changed\n")
    with pytest.raises(ValueError, match="own manifest"):
        pc.prepare(parent, tmp_path / "tampered", expected_xml_sha256=None)
    assert not (tmp_path / "tampered").exists()
    anchorless = make_parent(tmp_path / "anchorless", storm=MINI_STORM.replace("close (51)\n", ""))
    with pytest.raises(ValueError, match="exactly one"):
        pc.prepare(anchorless, tmp_path / "y", expected_xml_sha256=None)
    assert not (tmp_path / "y").exists()


def test_final_hook_distinguishes_the_storm_loop_from_later_output_loops():
    text = MINI_STORM + "do i = 1, nr1\nenddo\nclose (51)\n"
    patched = pc.insert_hooks(text)
    assert patched.count("call syrup_hydro_capture (3)") == 1
    assert "call syrup_hydro_capture (3)\nclose (51)\n\n!EVA2016" in patched
    assert pc.remove_hooks(patched) == text
