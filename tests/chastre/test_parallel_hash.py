"""The parallel persisted-tile hashing of `benchmarks/chastre/run_chastre_parallel.py` equals the serial original, still detects every
kind of change, and the thin wrapper keeps the original's contract. CPU only (no GPU, no storm timing beyond one tiny run), on
the existing small generated MAPLE tile fixture (a private copy is tampered with, never the shared fixture). Nothing here was run by
its author (file-only tools); Codex records results."""
import shutil
import sys
from pathlib import Path

import pytest
from test_tiled_case import _runner, _tile_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "chastre"))


def _verified(case_dir):
    from maple_syrup.chastre_case import verify_chastre_case

    return verify_chastre_case(case_dir, allow_maple_source_change=True, reload_tiles=False)


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_parallel_current_hashes_equal_the_serial_original_in_tile_order(chastre_case_dir, workers):
    import run_chastre_parallel as parallel

    from maple_syrup.chastre_case import manifest_digest

    verified = _verified(chastre_case_dir)
    case = verified.case
    serial = case.current_hashes()  # the original, unpatched implementation
    got = parallel.parallel_current_hashes(case, workers)
    assert len(got) == len(case.tiles) == 3 and got == serial
    assert manifest_digest(list(case.tiles), got) == verified.binding["tiles_digest"]
    assert [sorted(h) for h in got] == [sorted(t["files"]) for t in case.tiles]  # ordered gather: tile i holds tile i's files


def test_patched_digest_equals_bound_digest_and_the_original_method_is_restored_even_on_failure(chastre_case_dir):
    import run_chastre_parallel as parallel

    from maple_syrup.chastre_case import ChastreCase

    case = _verified(chastre_case_dir).case
    original = ChastreCase.__dict__["current_hashes"]
    with parallel.parallel_hashing(4) as stats:
        assert ChastreCase.__dict__["current_hashes"] is not original
        assert case.persisted_digest() == case.bound_tiles_digest
        assert case.persisted_digest() == case.bound_tiles_digest
    assert stats["calls"] == 2 and stats["seconds"] > 0.0 and ChastreCase.__dict__["current_hashes"] is original
    with pytest.raises(RuntimeError, match="boom"), parallel.parallel_hashing(2):
        raise RuntimeError("boom")
    assert ChastreCase.__dict__["current_hashes"] is original


def test_missing_tile_is_refused_and_extra_or_changed_files_are_found(case_copy):
    import run_chastre_parallel as parallel

    from maple_syrup.case_import import Plot1ImportError

    case = _verified(case_copy).case
    bound = case.bound_tiles_digest
    with parallel.parallel_hashing(4):
        assert case.persisted_digest() == bound
        (case_copy / "tiles" / "tile_000" / "stray.txt").write_text("x")
        assert case.persisted_digest() != bound  # an extra file changes the manifest digest
        (case_copy / "tiles" / "tile_000" / "stray.txt").unlink()
        path = _tile_file(case_copy, 2)
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0x01
        path.write_bytes(bytes(data))
        assert case.persisted_digest() != bound  # a changed byte changes it
        shutil.rmtree(case_copy / "tiles" / "tile_001")
        with pytest.raises(Plot1ImportError, match="missing"):
            case.persisted_digest()
    with pytest.raises(Plot1ImportError, match="missing"):
        parallel.parallel_current_hashes(case, 3)


def test_runguard_with_parallel_hash_detects_a_mutated_persisted_tile_like_the_original(case_copy):
    import run_chastre_parallel as parallel

    verified = _verified(case_copy)
    with parallel.parallel_hashing(4):
        _cc, run, guard, _inputs = _runner("bisection_numba", verified)  # baseline digest taken under the parallel hashing
        raw = run(30.0, [])
        path = _tile_file(case_copy, 0)
        data = bytearray(path.read_bytes())
        data[0] ^= 0x01
        path.write_bytes(bytes(data))
        with pytest.raises(RuntimeError, match="bed changed"):
            guard.validate(raw, 30.0)


@pytest.mark.parametrize("value", ["0", "-1", "33", "abc", "1.5", ""])
def test_hash_workers_must_be_a_positive_integer_at_most_32_with_no_silent_fallback(value, tmp_path):
    import run_chastre_parallel as parallel

    out = tmp_path / "out"
    with pytest.raises(SystemExit) as info:
        parallel.main(["--case-dir", str(tmp_path / "case"), "--output-dir", str(out), f"--hash-workers={value}"])
    assert info.value.code == 2 and not out.exists()


def test_case_and_output_dir_must_be_given_in_full(tmp_path):
    import run_chastre_parallel as parallel

    with pytest.raises(SystemExit) as info:
        parallel.main(["--case", str(tmp_path / "case"), "--output", str(tmp_path / "out")])
    assert info.value.code == 2


def test_wrapper_forwards_arguments_writes_metadata_only_after_success_and_propagates_the_exit_code(tmp_path, monkeypatch):
    import json

    import run_chastre_parallel as parallel
    import run_chastre_timing as original

    from maple_syrup.chastre_case import ChastreCase

    seen = {}
    method = ChastreCase.__dict__["current_hashes"]

    def stub(argv):
        seen["argv"] = list(argv)
        seen["patched"] = ChastreCase.__dict__["current_hashes"] is not method
        out = Path(argv[argv.index("--output-dir") + 1])
        out.mkdir()
        (out / "comparison.json").write_text('{"script_sha256": "original"}\n')
        return seen.get("code", 0)

    monkeypatch.setattr(original, "main", stub)
    out = tmp_path / "ok"
    args = ["--case-dir", str(tmp_path / "case"), "--output-dir", str(out), "--end-s", "60", "--rounds", "1"]
    assert parallel.main(["--hash-workers", "3", *args]) == 0
    assert seen["argv"] == args and seen["patched"] is True  # forwarded verbatim, without the wrapper's own option
    assert ChastreCase.__dict__["current_hashes"] is method
    record = json.loads((out / parallel.METADATA_NAME).read_text())
    assert record["hash_workers"] == 3 and record["original_exit_code"] == 0
    assert record["wrapper_sha256_before"] == record["wrapper_sha256_after"]
    assert record["original_runner_sha256_before"] == record["original_runner_sha256_after"] == parallel._file_sha256(
        parallel.ORIGINAL_RUNNER)
    assert (out / "comparison.json").read_text() == '{"script_sha256": "original"}\n'  # never touched
    # a non-zero original exit code is propagated and writes no metadata
    seen["code"] = 3
    failed = tmp_path / "failed"
    assert parallel.main(["--case-dir", str(tmp_path / "case"), "--output-dir", str(failed)]) == 3
    assert not (failed / parallel.METADATA_NAME).exists()
    # an exception keeps the partial output, writes no metadata and restores the method
    def boom(argv):
        Path(argv[argv.index("--output-dir") + 1]).mkdir()
        raise RuntimeError("boom")

    monkeypatch.setattr(original, "main", boom)
    broken = tmp_path / "broken"
    with pytest.raises(RuntimeError, match="boom"):
        parallel.main(["--case-dir", str(tmp_path / "case"), "--output-dir", str(broken)])
    assert broken.is_dir() and not (broken / parallel.METADATA_NAME).exists()
    assert ChastreCase.__dict__["current_hashes"] is method
