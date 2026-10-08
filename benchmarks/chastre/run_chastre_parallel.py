"""Thin entry over the ORIGINAL `run_chastre_timing.main` that parallelises ONLY the persisted-tile file hashing of the bed guard.

    python benchmarks/chastre/run_chastre_parallel.py [--hash-workers N] <the original run_chastre_timing.py arguments, unchanged>

Everything the original does is unchanged: its parser and defaults, case verification, preparation, warm-up, the timed samples,
the guard FREQUENCY (a full artifact stream-hash at guard construction and after every validated sample, no caching, no stat-only
or deferred check), the budgets, bounds, tolerances, physics and source hashes. The only difference: while the original `main`
runs in THIS process, `ChastreCase.current_hashes` hashes the tile directories concurrently with
`ThreadPoolExecutor.map(hash_tree, tile_dirs)`. `map` gathers in the original tile order, so the manifest digest is exactly the
serial one; a missing tile directory is refused; every changed, missing or extra file is still found because `hash_tree` (bounded
`stream_sha256` chunks, never a whole-file read) walks each tree completely. No MAPLE load/compile happens in a thread, no GPU,
no bed arrays. Memory: one 16 MiB read buffer per worker plus hash state (workers <= 32, default 8). The original method is
restored in a `finally`, also when the run fails.

Options: `--hash-workers` is a positive integer <= 32 (default 8; invalid values are refused, there is no silent fallback). All other
arguments go to the original parser verbatim; give `--case-dir` and `--output-dir` in FULL (abbreviations are not guessed here).
`-h/--help` is forwarded to the original parser (this wrapper's own option is documented here).

Provenance: after a SUCCESSFUL original `main` (return 0) a NEW `validation_parallelism.json` is written into the output directory
(never overwritten). It records the wrapper and original-runner SHA-256 before and after (a change is a terminal exception), the
worker count, the strategy, the memory scope, the number and cumulative wall time of the parallel hash calls and the statement that
no physical code changed. The original `comparison.json` is never touched and its `script_sha256` stays the ORIGINAL runner's;
`run_status.json` already records the actual (wrapper) argv. The metadata is not an acceptance of the physics or of any qualification:
a contender may have failed inside a run that returns 0, and only the independent audit qualifies results. A failing run keeps its
partial output and writes no metadata. Nothing here was run by its author (file-only tools).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _sub in (str(HERE), str(ROOT / "src")):
    if _sub not in sys.path:
        sys.path.insert(0, _sub)

DEFAULT_WORKERS = 8
MAX_WORKERS = 32
METADATA_NAME = "validation_parallelism.json"
ORIGINAL_RUNNER = HERE / "run_chastre_timing.py"
STRATEGY = ("ThreadPoolExecutor.map(hash_tree, tile_dirs): ordered gather in the original tile order; each tree fully walked and "
            "stream-hashed in bounded chunks; the guard still performs a FULL artifact hash at construction and after every "
            "validated sample (no caching, no stat-only, no deferred check)")
MEMORY_SCOPE = ("one 16 MiB read buffer per worker plus hash state; no whole-file reads, no MAPLE load/compile in threads, no GPU, "
                "no bed arrays")


def _workers(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--hash-workers must be an integer, got {text!r}") from exc
    if not 1 <= value <= MAX_WORKERS:
        raise argparse.ArgumentTypeError(f"--hash-workers must be in [1, {MAX_WORKERS}], got {value}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parallel_current_hashes(case, workers: int) -> list[dict[str, str]]:
    """The tile hash maps of `case.current_hashes()`, computed concurrently, in the SAME order as `case.tiles`.

    Raises `Plot1ImportError` for a missing tile directory (before any hashing); any exception of a worker propagates."""
    from maple_syrup.case_import import Plot1ImportError
    from maple_syrup.chastre_case import hash_tree

    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be an int in [1, {MAX_WORKERS}], got {workers!r}")
    tile_dirs = []
    for tile in case.tiles:
        tile_dir = Path(case.case_dir) / tile["dir"]
        if not tile_dir.is_dir():
            raise Plot1ImportError(f"persisted tile {tile['dir']} is missing")
        tile_dirs.append(tile_dir)
    if getattr(case, "_progress", None) is not None:
        case._progress(f"hashing {len(tile_dirs)} tiles with {workers} threads")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(hash_tree, tile_dirs))


@contextlib.contextmanager
def parallel_hashing(workers: int):
    """Temporarily replace `ChastreCase.current_hashes` by the parallel form; restored in `finally`. Yields a stats dict."""
    from maple_syrup.chastre_case import ChastreCase

    stats = {"calls": 0, "seconds": 0.0, "workers": workers}
    original = ChastreCase.__dict__["current_hashes"]

    def current_hashes(self):
        start = time.perf_counter()
        try:
            return parallel_current_hashes(self, workers)
        finally:
            stats["calls"] += 1
            stats["seconds"] += time.perf_counter() - start

    ChastreCase.current_hashes = current_hashes
    try:
        yield stats
    finally:
        ChastreCase.current_hashes = original


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    own = argparse.ArgumentParser(prog="run_chastre_parallel.py", add_help=False, allow_abbrev=False)
    own.add_argument("--hash-workers", type=_workers, default=DEFAULT_WORKERS)
    options, rest = own.parse_known_args(argv)
    reader = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    reader.add_argument("--output-dir")
    reader.add_argument("--case-dir")
    known, _ = reader.parse_known_args(rest)
    asked_for_help = any(a in ("-h", "--help") for a in rest)
    if not asked_for_help and not (known.output_dir and known.case_dir):
        own.error("--case-dir and --output-dir are required and must be given in full (no abbreviations)")

    import run_chastre_timing as original

    wrapper_before, runner_before = _file_sha256(Path(__file__)), _file_sha256(ORIGINAL_RUNNER)
    started = time.time()
    with parallel_hashing(options.hash_workers) as stats:
        code = original.main(rest)
    finished = time.time()
    wrapper_after, runner_after = _file_sha256(Path(__file__)), _file_sha256(ORIGINAL_RUNNER)
    if wrapper_after != wrapper_before or runner_after != runner_before:
        raise RuntimeError("the wrapper or the original runner file changed during the run; no provenance metadata written")
    if code == 0 and known.output_dir:
        record = {
            "schema": "maple_syrup.chastre.validation_parallelism.v1",
            "wrapper": str(Path(__file__).resolve()), "wrapper_sha256_before": wrapper_before, "wrapper_sha256_after": wrapper_after,
            "original_runner": str(ORIGINAL_RUNNER), "original_runner_sha256_before": runner_before,
            "original_runner_sha256_after": runner_after,
            "hash_workers": options.hash_workers, "strategy": STRATEGY, "memory_scope": MEMORY_SCOPE,
            "parallel_hash_calls": stats["calls"], "parallel_hash_seconds_total": stats["seconds"],
            "argv": argv, "pid": os.getpid(), "started_unix_s": started, "finished_unix_s": finished,
            "original_exit_code": code,
            "physical_code_unchanged": ("only the file-hash scheduling of the persisted-tile guard differs; storms, timers, guard "
                                        "frequency, budgets, bounds and tolerances are the original's"),
            "note": ("not an acceptance of the physics or of any qualification: the original returns 0 even when a contender "
                     "failed; only the independent audit qualifies results. comparison.json.script_sha256 is the ORIGINAL runner's."),
        }
        with (Path(known.output_dir) / METADATA_NAME).open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
