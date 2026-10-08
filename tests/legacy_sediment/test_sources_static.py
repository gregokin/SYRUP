"""Static guards of the Fortran sediment harness (no compiler needed; written without being run)."""
from __future__ import annotations

import struct

import numpy as np
import pytest
import sources as S

from .helpers import pit_graph, tiny_case_arrays

ORIGINALS = S.fr.MAHLERAN_ROOT
needs_reference = pytest.mark.skipif(not (ORIGINALS / "src").is_dir(), reason="MAHLERAN reference tree absent")


@needs_reference
def test_pins_match_the_original_sources_and_a_change_is_refused(monkeypatch):
    assert S.check_pins() == S.pinned_hashes()
    monkeypatch.setitem(S.PINS, "src/Subroutines_Sediment/flow_distrib.for", "0" * 64)
    with pytest.raises(S.FortranGlueError, match="differ"):
        S.check_pins()


@needs_reference
def test_derived_constants_are_verbatim_original_lines():
    text = (ORIGINALS / S.INITIALIZE).read_text(encoding="latin-1")
    generated = S.generate_constants_source(text)
    lines = text.splitlines()
    for first, last, _, _ in S.EXTRACT_RANGES:
        for line in lines[first - 1:last]:
            assert line in generated  # every original line, character for character
    assert generated.count("subroutine syrup_derived_constants") == 2  # header and end
    assert "sigma =" in generated and "dstar_const" in generated and "spa (phi) = spa (phi) / 1.2d3" in generated
    shifted = "\n".join(lines[1:])  # a line shift must be refused, not silently extracted
    with pytest.raises(S.FortranGlueError, match="not the expected original text"):
        S.generate_constants_source(shifted)


@needs_reference
def test_route_sediment_patches_apply_exactly_once_and_touch_only_the_declared_places():
    original = (ORIGINALS / S.ROUTE_SEDIMENT).read_text(encoding="latin-1")
    plain = S.patch_route_sediment(original, hooked=False)
    hooked = S.patch_route_sediment(original, hooked=True)
    assert "call splash_transport" in original and "call splash_transport" not in plain and "call splash_transport" not in hooked
    assert plain.count("call raindrop_detachment") == original.count("call raindrop_detachment") - 1  # wet-cell call kept
    assert "syrup_hook" not in plain and "syrup_trial" not in plain
    assert hooked.count("syrup_trial (phi, i, j) = d_soil (phi, 2, i, j)") == 1 and "use syrup_hook" in hooked
    # the hook sits BEFORE the original clip and the clip itself is unchanged
    assert hooked.index("syrup_trial (phi, i, j)") < hooked.index("if (d_soil (phi, 2, i, j).lt.0.d0) then")
    for text in (plain, hooked):
        assert "d_soil (phi, 2, i, j) = 0.0d0" in text
        assert "d_soil (phi, 2, im, jm) = 0." in text  # the Euler branch is untouched
    removed = [ln for ln in original.splitlines() if ln not in hooked.splitlines()]
    assert all("splash_transport" in ln or "raindrop_detachment" in ln for ln in removed)  # only the splash pair is gone
    with pytest.raises(S.FortranGlueError, match="splash"):
        S.patch_route_sediment(original.replace("call splash_transport", "call other_thing"), hooked=False)
    with pytest.raises(S.FortranGlueError, match="hook anchors"):
        S.patch_route_sediment(original.replace(".lt.0.d0) then", ".lt.0.d1) then"), hooked=True)


def _valid_input(tmp_path, **override):
    graph = pit_graph()
    arrays, _, kwargs = tiny_case_arrays(graph, n_steps=3)
    xml = {"sedparam": [0.1] * 30, "ke_model": 2, "particle_density_g_cm3": 2.65, "active_layer_sensitivity_mm": 1.52e-6}
    kwargs = {**kwargs, "xml": xml, "iroute": 5, **override}
    path = tmp_path / "input.bin"
    digest = S.write_input(path, arrays, **kwargs)
    return path, digest, arrays, kwargs


def test_input_layout_is_versioned_little_endian_with_a_trailer(tmp_path):
    path, digest, arrays, _ = _valid_input(tmp_path)
    data = path.read_bytes()
    assert data[:8] == S.MAGIC_IN
    header = struct.unpack_from("<14i", data, 8)
    assert header[0] == 1 and header[1] == S.ENDIAN_MARKER and header[6] == 6 and header[8] == 2  # version, endian, classes, CN
    assert (header[2], header[3]) == arrays["aspect"].shape
    assert data.endswith(S.TRAILER_TAG + struct.pack("<q", 0))
    off = 8 + 56 + 40
    tags = []
    while data[off:off + 8] != S.TRAILER_TAG:
        tag, count = data[off:off + 8].decode(), struct.unpack_from("<q", data, off + 8)[0]
        width = 4 if tag in ("ASPECT  ", "OUTLET  ", "ACTIVE  ", "ORDER   ", "CAPSTEPS", "SNAPSTEP") else 8
        off += 16 + count * width
        tags.append(tag)
    assert tuple(tags) == S.INPUT_BLOCKS and off == len(data) - 16
    assert len(digest) == 64


@pytest.mark.parametrize("override, match", [
    ({"rates_mm_s": [float("nan")] * 3}, "rates"), ({"rates_mm_s": [-1.0] * 3}, "rates"), ({"rates_mm_s": []}, "rates"),
    ({"capture_steps": [99]}, "capture"), ({"snapshot_steps": [0]}, "capture"), ({"iroute": 3}, "iroute"),
])
def test_write_input_refuses_invalid_requests_before_writing(tmp_path, override, match):
    graph = pit_graph()
    arrays, _, kwargs = tiny_case_arrays(graph, n_steps=3)
    kwargs["xml"] = {"sedparam": [0.1] * 30, "ke_model": 2, "particle_density_g_cm3": 2.65, "active_layer_sensitivity_mm": 1.5e-6}
    kwargs.update({"iroute": 5, **override})
    path = tmp_path / "bad.bin"
    with pytest.raises((S.FortranGlueError, ValueError), match=match):
        S.write_input(path, arrays, **kwargs)
    assert not path.exists()


def test_write_input_refuses_nonfinite_state_and_inexact_dx(tmp_path):
    graph = pit_graph()
    arrays, _, kwargs = tiny_case_arrays(graph, n_steps=3)
    kwargs["xml"] = {"sedparam": [0.1] * 30, "ke_model": 2, "particle_density_g_cm3": 2.65, "active_layer_sensitivity_mm": 1.5e-6}
    bad = dict(arrays)
    bad["slope"] = np.array(arrays["slope"], copy=True)
    bad["slope"][1, 1] = np.nan
    with pytest.raises(S.FortranGlueError, match="slope"):
        S.write_input(tmp_path / "a.bin", bad, **kwargs)
    inexact = {**arrays, "dx_mm": 100.1}
    with pytest.raises(S.FortranGlueError, match="REAL"):
        S.write_input(tmp_path / "b.bin", inexact, **kwargs)
    kwargs["sediment_fractions"] = kwargs["sediment_fractions"][:5]
    with pytest.raises(S.FortranGlueError, match="fractions"):
        S.write_input(tmp_path / "c.bin", arrays, **kwargs)


def _output_file(blocks, *, magic=S.MAGIC_OUT, trailer=True, extra=b""):
    body = magic
    for tag, values, dtype in blocks:
        payload = np.asarray(values, dtype=dtype).tobytes()
        body += tag.encode() + struct.pack("<q", np.asarray(values).size) + payload
    if trailer:
        body += S.TRAILER_TAG + struct.pack("<q", 0)
    return body + extra


def test_tagged_reader_is_strict(tmp_path):
    p = tmp_path / "o.bin"
    p.write_bytes(_output_file([("LEDGER  ", [1.0, 2.0], "<f8"), ("COUNTS  ", [3, 4], "<i8")]))
    out = S.read_tagged(p, S.MAGIC_OUT)
    assert out["LEDGER  "].tolist() == [1.0, 2.0] and out["COUNTS  "].tolist() == [3, 4]
    for name, data, match in (
        ("magic", _output_file([("LEDGER  ", [1.0], "<f8")], magic=b"SYRFXXX1"), "magic"),
        ("no_trailer", _output_file([("LEDGER  ", [1.0], "<f8")], trailer=False), "truncated"),
        ("extra", _output_file([("LEDGER  ", [1.0], "<f8")], extra=b"x"), "trailer"),
        ("cut", _output_file([("LEDGER  ", [1.0, 2.0], "<f8")])[:-30], "truncated"),
    ):
        q = tmp_path / f"{name}.bin"
        q.write_bytes(data)
        with pytest.raises(S.FortranGlueError, match=match):
            S.read_tagged(q, S.MAGIC_OUT)


def test_result_requires_the_completion_marker(tmp_path):
    p = tmp_path / "result.txt"
    p.write_text("STEPS 3\n")
    with pytest.raises(S.FortranGlueError, match="completion marker"):
        S.read_result(p)
    p.write_text(f"STEPS 3\n{S.MARKER}\n")
    assert S.read_result(p)["STEPS"] == "3"


def test_plot1_and_model2_cases_are_refused_not_approximated():
    from types import SimpleNamespace

    with pytest.raises(S.FortranGlueUnsupported, match="Plot 1"):
        S.inputs_from_legacy_case(SimpleNamespace(kind="plot1"))
    graph = pit_graph()
    _, column, _ = tiny_case_arrays(graph)
    from maple_syrup.infiltration import column_parameters

    n = tuple(graph.shape)
    hawkins = column_parameters(model="pavement_hawkins", ksat_m_per_s=np.full(n, 1e-6), suction_m=np.full(n, 0.02),
                                drainage_parameter=np.full(n, 0.05), theta_sat=np.full(n, 0.36),
                                soil_thickness_m=np.full(n, 0.21), pavement_cover_fraction=np.full(n, 0.1),
                                active_mask=np.asarray(graph.active))
    with pytest.raises(S.FortranGlueUnsupported, match="fixed_ksat"):
        S.hydrology_arrays(graph, hawkins, np.ones(n), np.zeros(n), 0.004)
    assert column.model == "fixed_ksat"


def _capture_blocks(nr2=3, nc2=3, *, skip=None, duplicate=None, nan_in=None, negative_in=None, short=None, extra=False):
    blocks = []
    for tag, count, dtype, role in S.capture_schema(nr2, nc2):
        if tag == skip:
            continue
        values = np.full(count - (1 if tag == short else 0), 0.25 if role != "finite" else -0.5)
        if tag == nan_in:
            values[0] = np.nan
        if tag == negative_in:
            values[0] = -1.0
        blocks.append((tag, values, dtype))
        if tag == duplicate:
            blocks.append((tag, values, dtype))
    if extra:
        blocks.append(("MYSTERY ", [1.0], "<f8"))
    return blocks


def test_schema_accepts_a_valid_capture_and_only_the_original_trial_may_be_negative(tmp_path):
    p = tmp_path / "cap.bin"
    p.write_bytes(_output_file(_capture_blocks(), magic=S.MAGIC_CAP))
    out = S.read_tagged(p, S.MAGIC_CAP, S.capture_schema(3, 3))
    assert list(out) == [t for t, *_ in S.capture_schema(3, 3)] and out["POST_TRL"].max() < 0.0  # negative trial allowed
    assert "PRE_DS1 " in out and "PRE_DS1" not in out  # exact 8-character tags


@pytest.mark.parametrize("kwargs, match", [
    ({"skip": "PRE_D1  "}, "differ from the schema"), ({"duplicate": "PRE_V   "}, "duplicate"),
    ({"short": "PRE_DS1 "}, "expected"), ({"nan_in": "POST_DET"}, "non-finite"), ({"negative_in": "POST_DEP"}, "negative"),
    ({"nan_in": "POST_TRL"}, "non-finite"), ({"extra": True}, "differ from the schema"),
])
def test_schema_refuses_self_consistent_but_malformed_captures(tmp_path, kwargs, match):
    p = tmp_path / "cap.bin"
    p.write_bytes(_output_file(_capture_blocks(**kwargs), magic=S.MAGIC_CAP))  # every block count is internally consistent
    with pytest.raises(S.FortranGlueError, match=match):
        S.read_tagged(p, S.MAGIC_CAP, S.capture_schema(3, 3))


def test_duplicate_blocks_are_refused_even_without_a_schema(tmp_path):
    p = tmp_path / "d.bin"
    p.write_bytes(_output_file([("LEDGER  ", [1.0], "<f8"), ("LEDGER  ", [2.0], "<f8")]))
    with pytest.raises(S.FortranGlueError, match="duplicate"):
        S.read_tagged(p, S.MAGIC_OUT)


def test_ledger_and_map_schemas_have_the_declared_sizes_and_flag_roles(tmp_path):
    n = 4
    ledger = [(t, np.full(c, 1.0), d) for t, c, d, _ in S.ledger_schema(n)]
    p = tmp_path / "l.bin"
    p.write_bytes(_output_file(ledger))
    assert S.read_tagged(p, S.MAGIC_OUT, S.ledger_schema(n))["COUNTS  "].dtype == np.dtype("<i8")
    maps = [(t, np.full(c, 2 if t == "TERMINAL" else 1.0), d) for t, c, d, _ in S.maps_schema(3, 3)]
    p.write_bytes(_output_file(maps))
    with pytest.raises(S.FortranGlueError, match="flag"):
        S.read_tagged(p, S.MAGIC_OUT, S.maps_schema(3, 3))


@needs_reference
def test_original_pins_are_checked_even_without_a_build_record(tmp_path, monkeypatch):
    import stat

    (tmp_path / "bin").mkdir()
    (tmp_path / "in").mkdir()
    script = tmp_path / "bin" / "fake"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    (tmp_path / "in" / "input.bin").write_bytes(b"x")
    monkeypatch.setitem(S.PINS, "src/Subroutines_Sediment/flow_distrib.for", "0" * 64)
    with pytest.raises(S.FortranGlueError, match="differ"):
        S.run_once(script, tmp_path / "in" / "input.bin", tmp_path / "run", expected={}, build_record=None)
    assert not (tmp_path / "run").exists()


@needs_reference
def test_the_run_directory_may_not_lie_inside_the_input_or_executable_directory(tmp_path):
    import stat

    (tmp_path / "bin").mkdir()
    (tmp_path / "in").mkdir()
    script = tmp_path / "bin" / "fake"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    (tmp_path / "in" / "input.bin").write_bytes(b"x")
    for bad in (tmp_path / "in" / "run", tmp_path / "bin" / "run"):
        with pytest.raises(ValueError, match="refusing"):
            S.run_once(script, tmp_path / "in" / "input.bin", bad, expected={}, build_record=None)
        assert not bad.exists()


def test_output_roots_and_reports_are_validated_new_only(tmp_path):
    ref = S.fr.MAHLERAN_ROOT
    src = S.ROOT / "src"
    for bad in (src, src / "new_out", ref / "out", S.ROOT, S.ROOT.parent):
        with pytest.raises(S.FortranGlueError, match="refusing output root"):
            S.new_output_root(bad)
    reused = tmp_path / "root"
    reused.mkdir()
    assert S.new_output_root(reused) == reused.resolve()  # an existing root may be reused
    guarded = tmp_path / "prepared"
    guarded.mkdir()
    with pytest.raises(S.FortranGlueError, match="refusing output root"):
        S.new_output_root(guarded / "run", {"prepared directory": guarded})
    report = tmp_path / "report.json"
    assert S.exclusive_text(report, "{}\n") == report.resolve()
    with pytest.raises(S.FortranGlueError, match="overwrite"):
        S.exclusive_text(report, "other")
    assert report.read_text() == "{}\n"
    with pytest.raises(S.FortranGlueError, match="refusing output root"):
        S.exclusive_text(src / "report.json", "x")
    assert not (src / "report.json").exists()


def test_input_is_streamed_with_the_digest_of_the_exact_bytes_and_bounded_extra_memory(tmp_path):
    messages = []
    stats: dict = {}
    graph = pit_graph()
    arrays, _, kwargs = tiny_case_arrays(graph, n_steps=3)
    kwargs["xml"] = {"sedparam": [0.1] * 30, "ke_model": 2, "particle_density_g_cm3": 2.65, "active_layer_sensitivity_mm": 1.5e-6}
    path = tmp_path / "s.bin"
    digest = S.write_input(path, arrays, iroute=5, progress=messages.append, stats=stats, **kwargs)
    assert digest == S.sha256_file(path) and stats["bytes_written"] == path.stat().st_size
    assert stats["largest_block_bytes"] < path.stat().st_size and messages


def test_a_missing_toolchain_is_the_only_skip(monkeypatch):
    monkeypatch.setattr(S, "toolchain", lambda: None)
    assert S.require_toolchain() is None
    with pytest.raises(S.FortranGlueError, match="no Fortran toolchain"):
        S.build("unused", hooked=True)
