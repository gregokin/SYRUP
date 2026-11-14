"""LOGIC controls of the strict portable Phase 2.5 source-identity profile, plus the integrity of the SHIPPED immutable proofs. The tiny synthetic trees here are LOGIC controls only: they prove the guard refuses edited /
extra / missing / forged / malformed / unknown inputs and selects branches only by literal pins. They are NOT physical acceptance and NOT the genuine final-180 positive (Root runs that after it supplies the final lock)."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p25_source_pins as real_pins
import p25_source_profile as prof

SHA = lambda b: hashlib.sha256(b).hexdigest()
BASE_FILES = {"a.py": b"a0\n", "b.py": b"b0\n", "c.py": b"c0\n"}
FINAL_FILES = {"a.py": b"a0\n", "b.py": b"b1\n", "c.py": b"c0\n", "w.py": b"w1\n", "n.py": b"n1\n", "res/table.json": b"{}\n"}
WH1_INPUT = b"w0\n"
GIT = "1" * 40
WH1_GIT = "2" * 40


def _dump(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()


def _table(files: dict) -> dict:
    return {p: SHA(b) for p, b in files.items()}


def _write_tree(root: Path, files: dict) -> Path:
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root


def _base_bytes() -> bytes:
    t = _table(BASE_FILES)
    return _dump({"git_base": GIT, "files": [[p, t[p]] for p in sorted(t)], "digest": prof.combine_digest(t), "file_count": len(t)})


def _wh1_bytes() -> bytes:
    return _dump({"schema": prof.WH1_PROOF_SCHEMA, "git_commit": WH1_GIT, "path": "w.py", "sha256": SHA(WH1_INPUT), "origin": "synthetic logic control"})


def _lock_obj(files: dict) -> dict:
    base, t = _table(BASE_FILES), _table(files)
    return {"schema": prof.LOCK_SCHEMA, "status": prof.LOCK_STATUS, "whole_phase_accepted": False, "git_base": GIT,
            "wh1": {"path": "w.py", "input_sha256": SHA(WH1_INPUT), "sha256": t["w.py"]}, "files": [[p, t[p]] for p in sorted(t)],
            "changed": [{"path": p, "base_sha256": base[p], "sha256": t[p]} for p in sorted(base) if t[p] != base[p]],
            "added": [{"path": p, "sha256": t[p]} for p in sorted(t) if p not in base and p != "w.py"],
            "file_count": len(t), "total_bytes": sum(len(b) for b in files.values()), "tree_digest_sha256": prof.combine_digest(t)}


def _pins(lock_bytes: bytes, files: dict, **over) -> SimpleNamespace:
    t = _table(files)
    base_bytes = _base_bytes()
    kw = {"BASE_GIT": GIT, "BASE_FILE_COUNT": len(BASE_FILES), "BASE_MANIFEST_SHA256": SHA(base_bytes), "BASE_TREE_DIGEST_SHA256": prof.combine_digest(_table(BASE_FILES)), "WH1_GIT": WH1_GIT, "WH1_PATH": "w.py",
          "WH1_INPUT_SHA256": SHA(WH1_INPUT), "WH1_PROOF_SHA256": SHA(_wh1_bytes()), "LOCK_SHA256": SHA(lock_bytes), "TREE_DIGEST_SHA256": prof.combine_digest(t), "FILE_COUNT": len(t),
          "TOTAL_BYTES": sum(len(b) for b in files.values()), "CHANGED": frozenset({"b.py"}), "ADDED": frozenset({"n.py", "res/table.json"}),
          "HISTORICAL": {"old": prof.combine_digest(_table(BASE_FILES))}}
    kw.update(over)
    return SimpleNamespace(**kw)


@pytest.fixture
def world(tmp_path):
    fx = tmp_path / "fx"
    fx.mkdir()
    (fx / "base169_manifest.json").write_bytes(_base_bytes())
    (fx / "wh1_953_input_proof.json").write_bytes(_wh1_bytes())
    lock = _dump(_lock_obj(FINAL_FILES))
    (fx / "final_lock.json").write_bytes(lock)
    src = _write_tree(tmp_path / "src", FINAL_FILES)
    return SimpleNamespace(fx=fx, src=src, pins=_pins(lock, FINAL_FILES), lock=lock, tmp=tmp_path)


def verify(w):
    return prof.verify_final(w.src, w.pins, fixtures=w.fx)


def relock(w, mutate) -> None:
    """Write a modified lock AND re-pin it: only the OTHER proof fields can then refuse it."""
    obj = _lock_obj(FINAL_FILES)
    mutate(obj)
    w.lock = _dump(obj)
    (w.fx / "final_lock.json").write_bytes(w.lock)
    w.pins.LOCK_SHA256 = SHA(w.lock)


# ---------------------------------------------------------------- positive logic control
def test_synthetic_exact_source_is_admitted_and_classified(world):
    v = verify(world)
    assert v.kind == "final" and v.changed == {"b.py"} and v.added == {"n.py", "res/table.json"} and v.unchanged == {"a.py", "c.py"}
    assert v.wh1_path == "w.py" and v.wh1_input_sha256 == SHA(WH1_INPUT) and v.wh1_sha256 == SHA(b"w1\n") and set(v.files) == set(FINAL_FILES)
    with pytest.raises(TypeError):
        v.files["x"] = "y"          # the proof table is read-only


def test_combine_digest_is_the_repositorys_own_source_tree_digest(tmp_path):
    from maple_syrup import provenance
    src = _write_tree(tmp_path / "s", FINAL_FILES)
    assert provenance.source_tree_digest(src).digest_sha256 == prof.combine_digest(_table(FINAL_FILES))


# ---------------------------------------------------------------- the actual source is the locked source
def test_one_edited_byte_is_refused_and_named(world):
    (world.src / "a.py").write_bytes(b"a1\n")
    with pytest.raises(prof.SourceProfileError, match=r"edited \['a.py'\]"):
        verify(world)


def test_an_extra_module_is_refused(world):
    (world.src / "extra.py").write_bytes(b"x\n")
    with pytest.raises(prof.SourceProfileError, match=r"extra \['extra.py'\]"):
        verify(world)


def test_an_extra_non_python_resource_is_refused(world):
    (world.src / "res" / "stray.orig").write_bytes(b"x\n")
    with pytest.raises(prof.SourceProfileError, match=r"stray.orig"):
        verify(world)


def test_a_missing_module_is_refused(world):
    (world.src / "n.py").unlink()
    with pytest.raises(prof.SourceProfileError, match=r"missing \['n.py'\]"):
        verify(world)


def test_cache_files_are_the_only_ignored_files(world):
    (world.src / "__pycache__").mkdir()
    (world.src / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"c")
    assert verify(world).kind == "final"


def test_a_symlink_in_the_source_scope_is_refused(world):
    os.symlink(world.src / "a.py", world.src / "link.py")
    with pytest.raises(prof.SourceProfileError, match="not admissible"):
        verify(world)


def test_verification_reads_cold_bytes_not_a_stale_cache(world):
    assert verify(world).kind == "final"
    path = world.src / "a.py"
    st = path.stat()
    path.write_bytes(b"a9\n")                              # same size, restored mtime: an id/size/mtime cache would still admit it
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(prof.SourceProfileError, match="edited"):
        verify(world)


# ---------------------------------------------------------------- literal pins beat a self-consistent forgery
def test_a_forged_self_consistent_lock_of_changed_source_is_refused_by_the_literal_pins(world):
    forged_files = dict(FINAL_FILES, **{"a.py": b"EVIL\n"})
    (world.src / "a.py").write_bytes(b"EVIL\n")
    (world.fx / "final_lock.json").write_bytes(_dump(_lock_obj(forged_files)))     # lock, listing, digest and counts all agree with the forged tree ...
    with pytest.raises(prof.SourceProfileError, match="literal pin"):               # ... but the pinned lock hash is the genuine one
        verify(world)


def test_a_modified_lock_byte_is_refused(world):
    (world.fx / "final_lock.json").write_bytes(world.lock + b" ")
    with pytest.raises(prof.SourceProfileError, match="literal pin"):
        verify(world)


# ---------------------------------------------------------------- missing proofs hard-fail
@pytest.mark.parametrize("name", ["base169_manifest.json", "wh1_953_input_proof.json", "final_lock.json"])
def test_a_missing_proof_is_a_hard_failure(world, name):
    (world.fx / name).unlink()
    with pytest.raises(prof.MissingProof, match=name):
        verify(world)


def test_a_modified_base_manifest_or_wh1_proof_is_refused(world):
    (world.fx / "base169_manifest.json").write_bytes(_base_bytes() + b" ")
    with pytest.raises(prof.SourceProfileError, match="base169 manifest bytes"):
        verify(world)
    (world.fx / "base169_manifest.json").write_bytes(_base_bytes())
    (world.fx / "wh1_953_input_proof.json").write_bytes(_wh1_bytes() + b" ")
    with pytest.raises(prof.SourceProfileError, match="WH1 input-proof bytes"):
        verify(world)


def test_a_self_consistent_forged_wh1_input_is_refused_by_the_literal_pin(world):
    obj = json.loads(_wh1_bytes())
    obj["sha256"] = SHA(b"other\n")
    forged = _dump(obj)
    (world.fx / "wh1_953_input_proof.json").write_bytes(forged)
    with pytest.raises(prof.SourceProfileError, match="literal pin"):
        verify(world)
    world.pins.WH1_PROOF_SHA256 = SHA(forged)               # even if the file pin were forged, the 953 input sha itself is pinned separately
    with pytest.raises(prof.SourceProfileError, match="WH1 input sha256"):
        verify(world)


# ---------------------------------------------------------------- malformed / unknown lock fields
def test_unknown_schema_or_status_or_acceptance_flag_is_refused(world):
    for key, value, match in (("schema", "syrup.phase25.source_lock/2", "schema/status"), ("status", "PROPOSED_NOT_LOCKED", "schema/status"), ("status", "ACCEPTED", "schema/status"),
                              ("whole_phase_accepted", True, "never accepts"), ("whole_phase_accepted", "false", "never accepts"), ("whole_phase_accepted", 0, "never accepts")):
        relock(world, lambda o, k=key, v=value: o.__setitem__(k, v))
        with pytest.raises(prof.SourceProfileError, match=match):
            verify(world)


def test_unknown_extra_and_missing_lock_keys_are_refused(world):
    relock(world, lambda o: o.__setitem__("note", "x"))
    with pytest.raises(prof.SourceProfileError, match="keys must be exactly"):
        verify(world)
    relock(world, lambda o: o.pop("added"))
    with pytest.raises(prof.SourceProfileError, match="keys must be exactly"):
        verify(world)


@pytest.mark.parametrize("bad", [True, 6.0, "6", -1, None])
def test_bool_float_and_malformed_counts_are_refused(world, bad):
    relock(world, lambda o: o.__setitem__("file_count", bad))
    with pytest.raises(prof.SourceProfileError, match="exact int"):
        verify(world)
    relock(world, lambda o: o.__setitem__("total_bytes", bad))
    with pytest.raises(prof.SourceProfileError, match="exact int"):
        verify(world)


def test_a_wrong_but_typed_count_is_refused(world):
    relock(world, lambda o: o.__setitem__("file_count", 7))
    with pytest.raises(prof.SourceProfileError, match="file count"):
        verify(world)


@pytest.mark.parametrize("path", ["../x.py", "/abs.py", "a\\b.py", "a//b.py", "./a.py", "__pycache__/a.py", "a.pyc", "", "a\x00.py", "d/../a.py"])
def test_unsafe_paths_are_refused(world, path):
    def mut(o):
        o["files"].append([path, SHA(b"z")])
    relock(world, mut)
    with pytest.raises(prof.SourceProfileError, match="unsafe|sorted|duplicate"):
        verify(world)


def test_duplicate_and_case_folded_duplicate_paths_are_refused(world):
    relock(world, lambda o: o["files"].insert(1, list(o["files"][0])))
    with pytest.raises(prof.SourceProfileError, match="duplicate"):
        verify(world)
    relock(world, lambda o: o["files"].append(["A.py", SHA(b"q")]))
    with pytest.raises(prof.SourceProfileError, match="duplicate|sorted"):
        verify(world)


def test_a_duplicate_json_key_and_non_finite_constants_are_refused(world):
    with pytest.raises(prof.SourceProfileError, match="duplicate JSON key"):
        prof.strict_json(b'{"a": 1, "a": 2}', "x")
    with pytest.raises(prof.SourceProfileError, match="non-finite"):
        prof.strict_json(b'{"a": NaN}', "x")
    with pytest.raises(prof.SourceProfileError, match="not strict JSON"):
        prof.strict_json(b"{", "x")


def test_a_wrong_classification_is_refused(world):
    relock(world, lambda o: o["changed"].append({"path": "a.py", "base_sha256": SHA(b"a0\n"), "sha256": SHA(b"a0\n")}))
    with pytest.raises(prof.SourceProfileError, match="classification"):
        verify(world)
    relock(world, lambda o: o["added"].pop())
    with pytest.raises(prof.SourceProfileError, match="classification"):
        verify(world)


def test_changed_added_names_must_equal_the_pinned_sets(world):
    world.pins.CHANGED = frozenset({"b.py", "c.py"})
    with pytest.raises(prof.SourceProfileError, match="pinned sets"):
        verify(world)


def test_the_lock_may_not_drop_a_base_module(world):
    def mut(o):
        o["files"] = [r for r in o["files"] if r[0] != "c.py"]
        o["file_count"] -= 1
    relock(world, mut)
    world.pins.FILE_COUNT -= 1
    with pytest.raises(prof.SourceProfileError):
        verify(world)


def test_a_wrong_wh1_record_is_refused(world):
    relock(world, lambda o: o["wh1"].__setitem__("input_sha256", SHA(b"other\n")))
    with pytest.raises(prof.SourceProfileError, match="WH1 record"):
        verify(world)


# ---------------------------------------------------------------- sentinel / branch selection
def test_the_sentinel_pins_hard_fail_not_skip(world):
    for name in ("LOCK_SHA256", "TREE_DIGEST_SHA256", "FILE_COUNT", "TOTAL_BYTES"):
        pins = copy.copy(world.pins)
        setattr(pins, name, prof.SENTINEL)
        with pytest.raises(prof.RequiresRootFinalLock, match=name):
            prof.select_branch(world.src, pins, fixtures=world.fx)


def test_the_shipped_pin_module_is_a_literal_template():
    """The immutable inputs are literal real hashes; the final-lock pins are EITHER the sentinel OR (after Root fills them) real typed values -- never a half-filled or malformed state."""
    for name in ("BASE_MANIFEST_SHA256", "BASE_TREE_DIGEST_SHA256", "WH1_INPUT_SHA256", "WH1_PROOF_SHA256"):
        assert len(getattr(real_pins, name)) == 64 and set(getattr(real_pins, name)) <= set("0123456789abcdef")
    assert real_pins.BASE_GIT == "02388adb70b7f1f22b33b6b3d5f1249e223842e5" and real_pins.WH1_GIT == "953e6fdf85b73e3fdf5a1ac7b0d97dea4df7dab1" and real_pins.WH1_PATH == "floating_wind_handoff.py"
    final = [real_pins.LOCK_SHA256, real_pins.TREE_DIGEST_SHA256, real_pins.FILE_COUNT, real_pins.TOTAL_BYTES]
    assert all(v == real_pins.REQUIRES_ROOT_FINAL_LOCK for v in final) or (
        len(real_pins.LOCK_SHA256) == 64 and len(real_pins.TREE_DIGEST_SHA256) == 64 and type(real_pins.FILE_COUNT) is int and type(real_pins.TOTAL_BYTES) is int)
    assert isinstance(real_pins.CHANGED, frozenset) and isinstance(real_pins.ADDED, frozenset) and real_pins.CHANGED and real_pins.ADDED and not real_pins.CHANGED & real_pins.ADDED


def test_final_branch_selected_only_by_the_exact_pinned_digest(world):
    b = prof.select_branch(world.src, world.pins, historical=("old",), fixtures=world.fx)
    assert b.kind == "final" and b.verified.kind == "final"
    (world.src / "a.py").write_bytes(b"a1\n")
    with pytest.raises(prof.UnknownSource):
        prof.select_branch(world.src, world.pins, historical=("old",), fixtures=world.fx)


def test_historical_branch_only_for_the_exact_literal_historical_digest(world, tmp_path):
    hist = _write_tree(tmp_path / "hist", BASE_FILES)
    b = prof.select_branch(hist, world.pins, historical=("old",), fixtures=world.fx)
    assert (b.kind, b.name, b.verified) == ("historical", "old", None)
    with pytest.raises(prof.UnknownSource):
        prof.select_branch(hist, world.pins, historical=(), fixtures=world.fx)       # a guard that does not name that historical package refuses it
    with pytest.raises(prof.SourceProfileError, match="unknown historical pin"):
        prof.select_branch(hist, world.pins, historical=("nonesuch",), fixtures=world.fx)
    (hist / "b.py").write_bytes(b"edited\n")
    with pytest.raises(prof.UnknownSource):
        prof.select_branch(hist, world.pins, historical=("old",), fixtures=world.fx)


def test_a_historical_package_is_admitted_even_while_the_final_pins_are_the_sentinel(world, tmp_path):
    pins = copy.copy(world.pins)
    pins.LOCK_SHA256 = prof.SENTINEL
    hist = _write_tree(tmp_path / "hist", BASE_FILES)
    assert prof.select_branch(hist, pins, historical=("old",), fixtures=world.fx).kind == "historical"
    with pytest.raises(prof.RequiresRootFinalLock):
        prof.select_branch(world.src, pins, historical=("old",), fixtures=world.fx)


# ---------------------------------------------------------------- Root helper never auto-admits
def test_prepare_lock_proposal_is_refused_by_the_guard_until_sealed_with_the_reviewed_digest(world):
    import prepare_lock as pl
    base_bytes = _base_bytes()
    # the helper reads the SHIPPED pins; bind it to the synthetic world for this logic control only
    pl.pins, pl.prof = world.pins, prof
    proposal = pl.propose(world.src, fixtures=world.fx)
    assert json.loads(proposal)["status"] == "PROPOSED_NOT_LOCKED" and base_bytes
    (world.fx / "final_lock.json").write_bytes(proposal)
    world.pins.LOCK_SHA256 = SHA(proposal)
    with pytest.raises(prof.SourceProfileError, match="schema/status"):
        verify(world)
    with pytest.raises(prof.SourceProfileError, match="reviewed tree digest"):
        pl.seal(proposal, "0" * 64)
    sealed = pl.seal(proposal, json.loads(proposal)["tree_digest_sha256"])
    (world.fx / "final_lock.json").write_bytes(sealed)
    world.pins.LOCK_SHA256 = SHA(sealed)
    assert verify(world).kind == "final"                                       # sealed bytes pinned by the literal: admitted (logic control)
    with pytest.raises(prof.SourceProfileError, match="only an unsealed"):
        pl.seal(sealed, json.loads(proposal)["tree_digest_sha256"])


# ---------------------------------------------------------------- shipped immutable proofs
def test_the_shipped_fixtures_are_the_literal_pinned_bytes_and_internally_consistent():
    base = prof.load_base(prof.read_fixture("base169_manifest.json"), real_pins)
    assert len(base) == 169 and prof.load_wh1_proof(prof.read_fixture("wh1_953_input_proof.json"), real_pins, base) == real_pins.WH1_INPUT_SHA256
    blob = prof.read_fixture("base_blob_floating_held_context.blob")
    assert SHA(blob) == base["floating_held_context.py"]                         # the byte-exact base blob the held-context additive guard diffs against
    assert not (prof.FIXTURES / "final_lock.json").exists() or real_pins.LOCK_SHA256 != real_pins.REQUIRES_ROOT_FINAL_LOCK, "a final_lock.json without Root's literal pin is never admitted"


def test_base_manifest_with_an_unsafe_or_duplicate_entry_is_refused(world):
    t = _table(BASE_FILES)
    for rows in ([["a.py", t["a.py"]], ["a.py", t["a.py"]]], [["../a.py", t["a.py"]]]):
        data = _dump({"git_base": GIT, "files": rows, "digest": "0" * 64, "file_count": len(rows)})
        pins = _pins(world.lock, FINAL_FILES, BASE_MANIFEST_SHA256=SHA(data))
        with pytest.raises(prof.SourceProfileError):
            prof.load_base(data, pins)
    data = _dump({"git_base": GIT, "files": [[p, t[p]] for p in sorted(t)], "digest": prof.combine_digest(t), "file_count": True})
    with pytest.raises(prof.SourceProfileError, match="exact int"):
        prof.load_base(data, _pins(world.lock, FINAL_FILES, BASE_MANIFEST_SHA256=SHA(data)))


# ---------------------------------------------------------------- importability
def test_helper_names_are_unique_across_the_test_tree():
    tests = HERE.parent
    for name in ("p25_source_profile.py", "p25_source_pins.py", "prepare_lock.py", "test_source_profile.py"):
        hits = [p for p in tests.rglob(name) if "__pycache__" not in p.parts]
        assert hits == [HERE / name], hits


def test_the_helper_imports_in_a_fresh_interpreter_without_any_conftest_or_task_path():
    code = f"import sys; sys.path.insert(0, {str(HERE)!r}); import p25_source_profile as p; assert 'conftest' not in sys.modules and p.FIXTURES.name == 'phase25'; print('ok')"
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=False)
    assert out.stdout.strip() == "ok", out.stderr


def test_the_helper_and_pin_sources_have_no_git_environment_or_ambient_path_dependency():
    for name in ("p25_source_profile.py", "p25_source_pins.py"):
        text = (HERE / name).read_text()
        code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith(("#", '"""')))
        for token in ("subprocess", "os.environ", "getenv", "git show", "ls-tree", "except Exception", "pytest.skip", "skipif", "/home/"):
            assert token not in code, (name, token)
