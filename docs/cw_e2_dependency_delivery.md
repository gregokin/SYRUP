# CW-E2: reproducible actual-MAPLE dependency delivery

`maple_syrup.dependency_bundle` turns an exact actual-MAPLE project tree into a content-addressed LOCAL artifact. It can then verify that artifact and materialize it as a digest-pinned, read-only project cache. A truthful editable overlay lets the existing resolver (`maple_syrup.dependency`) identify the cache as the `maple` distribution.

The tool does not:
- install anything, modify an environment, run commands or touch git;
- fetch from or publish to the network;
- copy MAPLE wind routines into SYRUP source. The archive is an external dependency artifact.

It is a delivery helper only. Runtime defaults, CLI entry points and model physics are unchanged.

## Commands

The examples use the existing MAPLE `.venv`. `PYTHONPATH` points at SYRUP's `src`.

```bash
PY=/home/okin/MAPLE/.venv/bin/python
export PYTHONPATH=/path/to/syrup/src
$PY -m maple_syrup.dependency_bundle export --project <MAPLE project root> \
    --distribution-metadata <actual maple-<ver>.dist-info> --origin origin.json --output <NEW bundle dir>
$PY -m maple_syrup.dependency_bundle verify <bundle dir>
$PY -m maple_syrup.dependency_bundle materialize <bundle dir> --cache <existing cache dir>
$PY -m maple_syrup.dependency_bundle verify-materialized <cache>/<bundle_id>
$PY -m maple_syrup.dependency_bundle env <cache>/<bundle_id> --syrup-src <syrup src>   # prints, never runs
```

To run against the cache, take `PYTHONPATH` and `MAPLE_SYRUP_EXPECTED_MAPLE_ROOT` from `env` and use ONLY `/home/okin/MAPLE/.venv/bin/python` or `/home/okin/MAPLE/.venv-cupy/bin/python`.

- `PYTHONPATH` is `<target>/editable:<target>/source/src:<syrup src>`.
- `MAPLE_SYRUP_EXPECTED_MAPLE_ROOT` is `<target>/source`.

`check-runtime <target>`, run in that environment, resolves MAPLE in the current process and requires the resolved package to be the cache with the bundle's package digest.

Do not run with a working directory that contains a stray `maple-*.dist-info`. `python -c` puts the working directory first on `sys.path`, and the resolver correctly refuses that shadowing distribution.

## Packaging dependency (CW-E5)

The helper imports the third-party `packaging` distribution: `packaging.version.Version` / `InvalidVersion` (canonical PEP 440 project version) and `packaging.requirements.Requirement` (normalized `Requires-Dist` comparison). The import happens only inside those checks. Every command that validates a project version, a manifest or distribution metadata needs it: `export`, `verify`, `materialize` and `verify-materialized`. When it is missing, these fail with an explicit `BundleError` naming `packaging`; nothing is skipped.

- **Declared** as the optional extra `dependency-delivery = ["packaging>=23"]` in `pyproject.toml`. It is not a base dependency, because the water model runtime does not import it.
- **Availability:** both MAPLE environments (`.venv`, `.venv-cupy`) already provide `packaging` 26.3, which is the only version exercised. The `>=23` floor is a conservative declaration, not a tested minimum. Declaring the extra installs nothing.
- **Not bundled:** the bundle archive carries only the MAPLE project tree and its dist-info. It does not vendor or pin `packaging`, NumPy, CuPy, Numba or any other Python package.

## Inputs

- **Project root.** A real directory holding `pyproject.toml` (project `maple` with a version) and `src/maple/__init__.py`. Use an immutable copied tree, not a live checkout: anything named `.git` is refused.
- **Distribution metadata.** An actual MAPLE `*.dist-info`. For example, the live venv's `maple-0.0.1.dist-info`, or the identical accepted overlay copy beside a frozen tree. It must contain `METADATA`, `WHEEL` and `top_level.txt`; `entry_points.txt` is optional. The METADATA fields `Name`, `Version`, `Summary`, `Requires-Python`, `Requires-Dist` and `Provides-Extra` must match the project's `pyproject.toml`, with requirements normalized by `packaging`. `RECORD`, `direct_url.json`, `INSTALLER` and `REQUESTED` are environment-specific; they are not taken and are regenerated in the overlay.
- **Origin record.** Required JSON with schema `maple_syrup.dependency_origin.v1`, with exactly these keys:
  - `description`;
  - `original_source`, with `kind`, `revision`, `revision_absence_reason` and `dirty_snapshot_note`;
  - `patch_chain`, a list of `{id, description[, evidence]}`;
  - `patch_chain_absence_reason`.

  An unknown revision needs an explicit absence reason, and so does an empty patch chain. Nothing is inferred from git. `.git/HEAD` is read only as `git_supplementary`, typically `not_a_git_repository` for copied trees.

## Bundle

A bundle directory holds exactly two files:

- **`bundle-manifest.json`.** Canonical JSON. `bundle_id` is the SHA-256 of the canonical manifest without `bundle_id`. The manifest records:
  - the schema and the tool module hash and Python version;
  - the limits and the selection rule;
  - the project name and version, the `pyproject.toml` hash and the exported-from path;
  - the full-project digest, and the package digest from the existing `provenance.source_tree_digest`;
  - the supplementary git record and the origin record;
  - the distribution file names;
  - every member's path, size and SHA-256;
  - the totals, and the archive size and SHA-256.
- **`maple-project.tar`.** An uncompressed pax tar of regular files only. Entries are sorted, with mtime 0, uid and gid 0, empty owner names and mode 0644. Members are `source/<path>` and `distribution/<file>`.

Selection rule:
- **Included.** Every project file. `tests/`, `build/` and `*.egg-info` are included because they are not caches.
- **Excluded.** Directories named `__pycache__`, `.pytest_cache`, `.ruff_cache` and `.mypy_cache`, and files ending `.pyc` or `.pyo`.
- **Refused.** Anything named `.git`, symlinks, hard links, devices, FIFOs and sockets.

Executable bits are not preserved.

## Materialized cache

`<cache>/<bundle_id>/` contains:
- `source/`, the project files;
- `distribution/`, the original dist-info files;
- `editable/`, holding `<name>-<ver>.dist-info/` (the verbatim METADATA, WHEEL, top_level and entry_points, plus `INSTALLER=maple_syrup.dependency_bundle`, `direct_url.json` with the editable `file://<target>/source` URL, and a regenerated `RECORD`) and `__editable__.<name>-<ver>.pth`;
- a copy of the manifest and `materialization.json`.

Files are made read-only (0444) and directories 0555. An existing target is re-verified and never rewritten. A corrupt existing target is refused. No console scripts are created.

## Safety properties

- **Bounds before reads.** Count, per-file, total, manifest, origin and archive sizes are checked before any read or allocation. Defaults: 20,000 files, 64 MiB per file, 512 MiB total, a 16 MiB manifest and a 64 KiB origin record.
- **Streaming.** All reading and copying uses bounded 1 MiB chunks. Neither the archive nor all files are ever held in memory.
- **Change detection.** Export hashes each file during its first walk, re-hashes exactly the bytes copied into the tar, re-walks afterwards, and refuses publication if anything changed.
- **Strict manifests and archives.** Duplicate JSON keys are refused, and the schema, keys and types are exact. The archive must be the exact manifest member set. Unlisted, duplicate, missing, oversized, non-regular, link, directory and non-normalized members are refused, as are traversal or absolute paths and unexpected pax headers. Every member hash is verified. Extraction writes only beneath a fresh private staging directory.
- **Atomic publication.** Export and materialize stage in a private sibling directory and publish with `renameat2(RENAME_NOREPLACE)`. A concurrently created path, even an empty directory, is never replaced. Where `renameat2` is unavailable the tool refuses to publish. On failure only the tool's own staging directory is removed, and no accepted or partial output remains.

## Limits

- **Tool-verified identity, not authenticity.** No signature or trust anchor exists; whoever can write the bundle can re-sign its digests.
- **Not interpreter-independent.** Identity depends on the tool module hash and Python version recorded in the manifest.
- **Linux only.** Publication requires `renameat2`.
- **Python dependencies are not pinned or bundled.** The MAPLE venvs provide them.
- **No upstream synchronization, wind integration or deep-store checkpoint archiving.**

## Correction 1 (CW-E2 review): manifest consistency and staged publication

The root reproduced a self-consistent forged manifest: `project.version` was set to `../../../../escaped` and `bundle_id` recomputed. That forgery passed `verify` and made `materialize` write overlay files outside the cache before refusing after publication. Now:

- **Manifest checks (`verify` and `materialize`).** The manifest is validated against this tool's documented schema and against its own member table before any byte is extracted:
  - the project version must be a canonical PEP 440 version and a safe single path component;
  - every generated overlay name must be a safe component and a relative path;
  - the tool, limits, selection rule, git, origin, distribution and archive records must have the exact documented structure;
  - the project and package digests, counts and bytes must equal values recomputed from the members;
  - `pyproject_sha256` must equal its member's hash.
- **Semantic checks against the archive.** The project name and version must equal the ACTUAL archived `pyproject.toml`. The archived dist-info must pass the same cross-checks as export. Only these small records are kept in memory, bounded at 1 MiB each.
- **Staged verification.** `materialize` writes only beneath its private staging directory, through confined joins, and completes short writes. It verifies the COMPLETE staged tree against the intended final target identity BEFORE the atomic no-replace rename: source, distribution, overlay, manifest copy and record. The post-publication check is observation only.

This is still consistency checking, not cryptographic authenticity. A forger who also rewrites the archived source and metadata consistently is not detected.
