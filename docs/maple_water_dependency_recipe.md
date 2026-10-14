# Reproducible actual-MAPLE water runtime: Git revision + net patch (CW-E9 tool, CW-E46 24c6 recipe)

**CW-E46 recipe: root independently ACCEPTED (`cw_e46_root_independent_acceptance.json`) and copied into the root
working tree by CW-E47 (2026-10-14), independently verified by CW-E48; included in this commit.** The pinned
recipe now targets the actual accepted delivered MAPLE **24c6** runtime: package digest `24c6e147…a9ce` (340 package
files) + `pyproject.toml` `e6c7da22…` = **341 runtime files**, byte-equal to `src/maple` + `pyproject.toml` of the
accepted delivered full-project cache `outputs/dependencies/content_cache/b8fa43a3…/source` and of the frozen B107 review
source (`cw_b107_root_review_freeze.json`). Root decision CW-E45: adopt 24c6 directly later, no interim M94.

24c6 = the accepted M94 runtime + CW-B106C1 (exact-overlay deep store, package `4585f8de…`) + CW-B107 (publisher
option forwarding / snapshot materialization). Exactly six existing files change; none is added or removed:

| Stage | File | M94 | B106C1 | 24c6 |
|---|---|---|---|---|
| B106C1 | `src/maple/surface/voxels/deep_store.py` | `e10271b9…` | `c0855474…` | = B106C1 |
| B106C1 + B107 | `src/maple/surface/voxels/windowed_pending.py` | `b86bd89d…` | `f8fdaaa5…` | `b10e4ec9…` |
| B106C1 | `src/maple/surface/voxels/windowed_transfer.py` | `42cb26a0…` | `062d4360…` | = B106C1 |
| B107 | `src/maple/surface/topographic_commit/windowed_commit.py` | `09a4c87e…` | = M94 | `2f481b9a…` |
| B107 | `src/maple/surface/voxels/windowed_snapshot.py` | `c6de6c10…` | = M94 | `f37de83b…` |
| B107 | `src/maple/water/windowed_step.py` | `d4b4963b…` | = M94 | `73baeff1…` |

**Scope of the change:**
- All six are NEW files of the patch relative to the upstream revision (water stages since M29), so no patch section
  is added: the patch still names the same 35 files (19 modified, 16 new), and exactly these six sections differ from
  the CW-E44 M94 patch; the other 29 are byte-identical and in the same order. `pyproject.toml` is unchanged.
- The B106 / B106C1 / B107 MAPLE tests (`tests/conservative_window/test_cw_b106_*`, `test_cw_b106c1_*`,
  `test_cw_b107_*`) are full-project files, not runtime, and are not in the patch.

**Runtime artifact vs full-project artifact (distinct identities).**
- This recipe reproduces ONLY the 341 runtime files and the package digest `24c6e147…`.
- The delivered full-project artifact (bundle `b8fa43a3…`, 1278 project files, project digest `f8aaff8d…`; MAPLE tests,
  docs, `build/`) is a SEPARATE artifact. Its non-runtime ancestry (the B97 1272-file freeze, then the B106C1 / B107
  full manifests) is not reproduced or claimed here.
- The CW-E46 runtime-only bundle (below) has a different bundle id and project digest (342 files incl. the receipt)
  from `b8fa43a3…`; runs bind the package digest, which is the same.

**Ancestry (dirty-upstream preserved).** Git `74dd3e79` + the original two dirty package files (E4) + bed optimization
`72310c49` + water M29 … M94 (E9 … E44 recipes) + B106C1 + B107, recorded in the manifest (`chain`, `stage_files`,
`changed_file_stages`) and in the origin record `cw_e46_24c6_runtime_origin.json`.

**Status (bounded; none is a whole-component acceptance):**
- CW-B106C1 (`cw_b106c1_independent_acceptance.json`) and CW-B107 (`cw_b107_independent_acceptance.json`) are ACCEPTED as
  bounded MAPLE store / publisher candidates (root JUnit: B106C1 47 CPU tests with 3 skipped / 3 GPU; B107 33 CPU / 6
  GPU, no failures), not production adoption.
- The full-project delivery `b8fa43a3…` and both existing-interpreter resolvers are ACCEPTED
  (`cw_b107_root_dependency_delivery_acceptance.json`).
- SYRUP 5f (`5f85a989…`, CW-B108C1) is ACCEPTED bounded optional-overlay integration on 24c6 with separate-process CPU /
  GPU continuation (`cw_b108c1_independent_acceptance.json`).
- **Open:** the current-source large own-wet restart / strict comparison (the root B111 GPU reference is ACTIVE and is
  not pinned), the 89 current CPU–GPU strict FAIL records (qualified only), large storage fit, whole water-component
  acceptance and root adoption. Not B98 / B99 (M95).

**History:** the CW-E9 M44 … CW-E39 M93 and CW-E44 M94 recipes (the last independently accepted; manifest `27bd1900…`,
patch `540dfb7d…`, bundle `a8904a99…`) and their records remain historical and are not relabelled as 24c6.

This recipe produces the **runtime project** of the actual MAPLE 24c6 water dependency, `src/maple` + `pyproject.toml`,
from upstream MAPLE Git. It does not copy wind physics into SYRUP: all unchanged MAPLE code, wind included, comes from
the upstream commit.

## Recipe contents (`dependencies/maple_water/`; included in this commit)

| File | Content |
|---|---|
| `manifest.json` | Pinned revision `74dd3e79d9a7df85244355439e3a4a762abb9273`, pathspecs `src/maple`, `pyproject.toml`; the 325 pre-patch files (git blob id, mode, size, SHA-256); the patch pin and its changed/new/modified file lists; the 341 post-patch SHA-256 values (340 package files + `pyproject.toml`) and the `src/maple` package source digest `24c6e147…a9ce` (same algorithm as `maple_syrup.provenance`); provenance: the explicit target `target_runtime` (`24c6`, with `changed_from_M94`, the per-stage `changed_from_M94_stages` / `changed_from_M94_sha256` (M94 / B106C1 / 24c6), the intermediate B106C1 package digest, the upstream relation, the distinct full-project artifact identity and the bounded status), the historical record `historical_M94_recipe` (CW-E44 manifest / patch / bundle and its unchanged M94 target record with memory and broad-suite failures), the unchanged `historical_M93_recipe` … `historical_M44_recipe`, the stage attribution of every changed file (the historical stages through `water_M94`, then `water_B106C1` / `water_B107`), a `trust_note`, and the SHA-256 of each local evidence record (the 249 E44 records unchanged, plus 31: the archived E44 recipe, acceptance, root test log, origin and resolver record, the B106 prompt / report / diff / closed review, the B106C1 prompt / report / exit / diffs / root freeze / acceptance, the B107 prompt / report / exit / diff / root freeze / acceptance / delivery acceptance / delivery origin / both delivery resolver records, the B108C1 acceptance, the E45 report / root review and the E46 prompt; 280 records; the ACTIVE root B111 GPU reference and its results are not pinned); out-of-scope list |
| `maple_74dd3e79_to_24c6_runtime.patch` | Net text patch: 35 files (19 modified, 16 new, none deleted, no binary or mode changes): the same file set as the CW-E44 M94 patch. It contains only the net original-dirty / bed-optimization / water M29 … M94 / B106C1 / B107 changes; every CW-E44 M94 section except the six files above is byte-identical, in the same order, and no section is added. `pyproject.toml` is unchanged from the revision. The old M94 patch is not part of this delivery (archived in `cw_e46_logs/baseline_archive`) |

The tool `tools/reconstruct_water_maple.py` pins the manifest's SHA-256 (`038dceda…748d`), which is its single trust
anchor.

## Reconstruct

```
PYTHONDONTWRITEBYTECODE=1 python tools/reconstruct_water_maple.py \
    --maple-repo /path/to/MAPLE --dest /new/path/maple_water_runtime
```

- **Repository:** `--maple-repo` must contain the pinned commit. It is used read-only (`rev-parse`, `ls-tree`,
  `archive`). The working tree, HEAD and dirty files are never read, and nothing is checked out.
- **Destination:** `--dest` must not exist, its parent must be an existing real directory, and it may not lie inside
  the repository. The tool writes only into one unique `.<name>.e9-staging-*` directory beside the destination and
  publishes it with Linux `renameat2(RENAME_NOREPLACE)`. The kernel refuses if the destination exists in any form
  (empty directory, file, symlink, dangling symlink), including one created concurrently after the early check.
  On failure only that staging directory is removed. Existing paths are never reused, replaced or deleted.
- **Read-only source guard:** destination decisions use canonical paths, with every symlink resolved including
  ancestor aliases. The repository's actual working-tree root, git dir and common git dir are derived with read-only
  `git rev-parse`, so a subdirectory or symlinked repository argument protects the whole working tree. A destination
  at or inside any of them is refused before any staging. A bare repository, or an argument that is not inside a
  working tree (such as `.git` itself), is refused. Staging and publication use exactly the canonical parent that was
  checked. The parent is assumed stable during a run (a trusted local parent); no guarantee is claimed against
  hostile concurrent ancestor swaps.
- **Platform:** Linux is required. Elsewhere, or without `renameat2`, the tool refuses before any work and again at
  publication. There is no fallback rename.
- **Checks:**
  - manifest pin and structure, with canonical in-scope paths and no duplicates;
  - patch pin and strict structure (no rename, copy, delete, mode, binary or symlink sections);
  - `ls-tree` equality with the pinned blobs;
  - a bounded archive whose members are checked (regular files only, no links, devices, `..` or absolute paths,
    nothing unexpected or duplicated, sizes and SHA-256 values);
  - `git apply --check` and `git apply -p1` with repository discovery fenced at the staging directory;
  - the complete post-patch file set, every SHA-256 and the package source digest, before publication.
- **Requirements:** Python standard library and `git` only. No pip, network or installation.

The result contains `src/maple` (340 files), `pyproject.toml` and the receipt `maple_water_reconstruction.json`.

## Use

**The plain reconstructed `src` is NOT a working runtime path: a source-only `PYTHONPATH` is not installed
metadata.** With
`PYTHONPATH=<dest>/src:<syrup>/src` and `MAPLE_SYRUP_EXPECTED_MAPLE_ROOT=<dest>`, the resolver
(`maple_syrup.dependency.resolve_maple_dependency`, called by every case verifier before a run) refuses with
`MapleDependencyError: imported maple at <dest>/src/maple is not inside the editable install root
'file:///home/okin/MAPLE' recorded by the maple distribution; another copy is shadowing it on sys.path`. The existing
MAPLE venvs carry their own editable `maple` distribution, which points at the live MAPLE checkout. The resolver
needs a distribution overlay that names the reconstructed copy instead. This refusal is intended: it is not bypassed
and not weakened.

Use the accepted CW-E2 helper to wrap the reconstruction in a content-addressed bundle whose materialized cache
carries that overlay. CPU interpreter shown; `/home/okin/MAPLE/.venv-cupy/bin/python` works the same way. Nothing is
installed.

```bash
PY=/home/okin/MAPLE/.venv/bin/python
SY=/path/to/SYRUP                         # this repository
DEST=/new/path/maple_water_runtime        # output of tools/reconstruct_water_maple.py
export PYTHONDONTWRITEBYTECODE=1

# 1. The actual MAPLE distribution metadata of the interpreter you will use (public importlib.metadata API):
DIST=$($PY -c "import importlib.metadata as m; print(next(f for f in m.distribution('maple').files if f.name == 'METADATA').locate().parent)")
#    e.g. /home/okin/MAPLE/.venv/lib/python3.12/site-packages/maple-0.0.1.dist-info

# 2. An origin record (schema maple_syrup.dependency_origin.v1; see below), e.g. $DEST.origin.json

# 3. Export a NEW bundle (output must not exist), materialize into an EXISTING cache directory, print the environment:
PYTHONPATH=$SY/src $PY -m maple_syrup.dependency_bundle export --project "$DEST" \
    --distribution-metadata "$DIST" --origin "$DEST.origin.json" --output /new/path/maple_water_bundle
mkdir -p /new/path/maple_cache
PYTHONPATH=$SY/src $PY -m maple_syrup.dependency_bundle materialize /new/path/maple_water_bundle --cache /new/path/maple_cache
PYTHONPATH=$SY/src $PY -m maple_syrup.dependency_bundle env /new/path/maple_cache/<bundle_id> --syrup-src "$SY/src"
#    -> prints PYTHONPATH=<target>/editable:<target>/source/src:$SY/src and MAPLE_SYRUP_EXPECTED_MAPLE_ROOT=<target>/source
```

`export` prints the `bundle_id`; `materialize` creates `<cache>/<bundle_id>/`. The helper commands and options above
are those of `python -m maple_syrup.dependency_bundle {export,materialize,env} --help`. The helper needs the optional
`dependency-delivery` extra (`packaging`), which is present in both MAPLE venvs. Run the model with the two printed
variables. The resolver then identifies the materialized `source/src/maple`, whose package source digest is
`24c6e147…a9ce`.

**Distribution metadata.**
- `export` cross-checks the dist-info `METADATA` (name, version, requirements, extras) and `top_level.txt` against the
  reconstructed `pyproject.toml`, and refuses a mismatch. A venv whose `maple` was installed from a different MAPLE
  `pyproject.toml` is therefore refused, not silently mixed.
- Where it exists locally, the immutable M94 copy
  `outputs/dependencies/maple_cw_b97_verified_tests_frozen/editable/maple-0.0.1.dist-info` is an alternative; CW-E44
  and CW-E46 used it. Its exported `METADATA`, `WHEEL`, `top_level.txt` and `entry_points.txt` are byte-identical to
  the dist-info of the accepted 24c6 full-project cache `b8fa43a3…` (CW-E46 asserted). `pyproject.toml` is identical
  in M44, M67, M75, M79, M82, M85, M88, M91, M92, M93, M94 and 24c6, SHA-256 `e6c7da22…`.
- No dist-info is tracked in this repository; none is invented here.

**Origin record.** `--origin` must be a JSON object with exactly the keys `schema` (`"maple_syrup.dependency_origin.v1"`),
`description`, `original_source` (`kind`, `revision`, `revision_absence_reason`, `dirty_snapshot_note`),
`patch_chain` (a list of `{id, description[, evidence]}`) and `patch_chain_absence_reason`. The reconstruction receipt
`maple_water_reconstruction.json` is not in this schema. Write a short record from it (revision `74dd3e79…`, the patch
chain recorded in `dependencies/maple_water/manifest.json` under `provenance`, including the `water_M67`,
`water_M75`, `water_M79`, `water_M82`, `water_M85`, `water_M88`, `water_M91`, `water_M92`, `water_M93`, `water_M94`,
`water_B106C1` and `water_B107` stages). The reviewed CW-E6
`cw_e6_M44_origin.json` describes M44, the CW-E12 origin describes M67, any M75 origin describes M75, the CW-E20 origin
describes M79, the CW-E23 origin describes M82, any M85 origin describes M85, the CW-E31 origin describes M88 and the
CW-E34 origin describes M91, the CW-E37 origin describes M92, the CW-E39 origin describes M93 and the CW-E44 origin
describes M94; the B107 delivery origin (`cw_b107_root_delivery_origin.json`) describes the FULL 1278-file project.
None may be reused for a 24c6 runtime-only bundle (CW-E46 wrote its own, `cw_e46_24c6_runtime_origin.json`).

The origin record is **provenance**: what the project was derived from, as stated by its author. The bundle's
content hashes and the manifest pins are **integrity**: these exact bytes. Neither is an authenticity guarantee.

**Identities are distinct.**
- The 24c6 runtime-only bundle built by CW-E46 (root independently accepted with both existing-environment resolvers) is
  `7ee10bd81b0859a891ff8a476cab2c8824bb0964309695ca948d12d5ac4b0c71` (project digest `4d371597…`, 342 files incl. the
  receipt). It differs from the accepted 24c6 FULL-PROJECT delivery `b8fa43a3…` (1278 project files, project digest
  `f8aaff8d…`; same package bytes), from the historical CW-E44 M94 runtime-only bundle `a8904a99…`, the historical
  CW-E39 M93
  runtime-only bundle `20bc496c…`, the historical CW-E37 M92
  runtime-only bundle `3cdf2943…`, the historical CW-E34 M91 runtime-only bundle `901a8abe…`, the historical CW-E31 M88
  runtime-only bundle `a586c823…`, any M85 runtime bundle (none
  was built), the historical CW-E23 M82 runtime-only bundle
  `8a29dc39…`, the historical CW-E20 M79
  runtime-only bundle `89c6e3f0…`, any M75 runtime bundle, the
  historical CW-E12 M67 runtime-only bundle `9bf97f1f…`, the historical M44 runtime-only bundle `43aa5ce6…` and the
  CW-E6 M44 full-project artifact `43a15ee9…` (1259 files). None of these may be treated as the same artifact.
- The 24c6 runtime bundle (and the full-project `b8fa43a3…`) resolve the package source digest `24c6e147…a9ce`; the
  historical M94 bundle resolves `eb14da66…bd6`; the historical M93 bundle resolves
  `3dd4108c…6ab1`; the historical M92 bundle resolves
  `e4398b27…ae88`; the historical M91 bundle resolves
  `a9de77ce…4ff5`; the historical M88 bundle resolves
  `89a22772…25fe`; an M85 bundle would resolve
  `17044c8a…b29c`; the historical M82 bundle resolves
  `1667efe2…ad5b`, the historical M79 bundle
  `7a4b65a4…3107`, an M75 bundle `c39e8b3b…4fbf`, the historical M67 bundle `b08b24d1…9caf` and the historical M44
  bundles `44e4f4c6…8630a`. Runs bind the package digest, not the bundle id.
- Third-party packages (numpy, numba, CuPy, …) are never bundled.

**Availability limits.**
- Reconstruction needs a MAPLE Git repository that contains commit `74dd3e79d9a7df85244355439e3a4a762abb9273`. If
  that commit is not available outside this machine, the recipe is not reproducible elsewhere.
- The manifest's evidence hashes refer to local, git-ignored records (`agent_handoffs/…`). The provenance chain is
  checkable only where those records exist. Reconstruction itself needs only the commit and the tracked recipe.

## Not included

- MAPLE tests, docs, `build/` and other full-project files. This is **not** a full MAPLE checkout, **not** the CW-E6
  artifact and **not** the full-project 24c6 delivery `b8fa43a3…`.
- The artifact id.
- The environment.

This recipe pins the 24c6 package bytes. Its MAPLE changes are accepted only as bounded B106C1 / B107
candidates; SYRUP 5f on 24c6 is accepted only as bounded integration (B108C1).
- **Not pinned here:** the B111 GPU reference and the B114 recovered restart (both completed after this recipe was
  built), their stores and results, and any future proof.
- **Since accepted (recorded outside this manifest):** the current-source rotated Chastre own-240-to-300 s wet restart
  and its strict comparison (`cw_b114_root_actual_restart_acceptance.json`), then CW-E47 root working-tree adoption,
  pending root verification.
- **Pending:** large storage fit, root verification of the CW-E47 adoption, and a separately authorized commit.
- **Not an acceptance:** recipe delivery does not declare the water component accepted.

**Fresh-clone reproducibility (not yet achieved).** In the SYRUP root these recipe files, the tool, its test and this
doc are currently UNTRACKED (and so are nine runtime `maple_syrup` modules; CW-E45). Local evidence records under
`agent_handoffs/` and `outputs/` are ancestry documentation, not fresh-clone trust roots. A fresh clone reproduces the
runtime only once the adopted `dependencies/maple_water/*`, `tools/reconstruct_water_maple.py`,
`tests/test_reconstruct_water_maple.py` and this doc are committed (a separately authorized action) and commit
`74dd3e79` is reachable.

For portability: the recipe reconstructs the exact runtime project only; it does not supply MAPLE's full tests or
docs, or any third-party environment.

## Historical M94 recipe (CW-E44; not 24c6)

The CW-E44 recipe (manifest `27bd1900dda7d59ed93b4b36345a30daeb45b3e62dc464618b17f60d75be7cfc`, patch
`540dfb7d2b06e54b2844046c0f0c9d60197935ad1e0289978f0757fa36fb3544`, package digest `eb14da66…bd6`) was independently
accepted as the bounded M94 runtime delivery, with runtime-only bundle
`a8904a9911200b530c3f8068bbaa537cc05c22f5e2f7fb6f1ee78b7bca535740` and both existing-environment resolvers
(`agent_handoffs/tasks/conservative_water/cw_e44_independent_acceptance.json`; not root adoption).
- **Its memory record** (whole erosion peak M93 6,124,032 B → M94 5,847,552 B; whole paired peak 11,803,136 B unchanged)
  and **broad-suite failures** (CPU 38 → 40, GPU 65 → 67 of 2,101) describe M94, not 24c6.
- **Large-case refusal:** B93 / M94 Chastre H8 (`availability/exchange.py:430`). **Small storm:** C34 B93 / M94 Plot1
  CPU complete + own wet restart is M94 evidence.
- **Recorded as** `historical_M94_recipe` (with the unchanged E44 `target_runtime` as `e44_target_runtime`).

## Historical M93 recipe (CW-E39; not M94)

The CW-E39 recipe (manifest `ef63b0960937d83ec256d0880995d64a5d6668b76915e66f257caf5163e862d7`, patch
`b1cc5a537822d69ec30a813ede33fc12ea792b8a83e13a16040dc11fac43f51b`, package digest `3dd4108c…6ab1`) was independently
accepted as the bounded M93 runtime delivery, with runtime-only bundle
`20bc496c77e7f07b12bad61e4db3434966834e2ecf12888aeab08ff5fc1d5d86` and both existing-environment resolvers
(`agent_handoffs/tasks/conservative_water/cw_e39_independent_acceptance.json`; not root adoption).
- **Its memory record** (proportional-vector helper peak M92 1,141,760 B → M93 568,320 B) describes M93, not M94.
- **Large-case refusal:** B93 / M93 H7 (attempt 5 after 4 accepted seconds, `windowed_exchange.py:101`).
- **Small storms:** the C33 B93 / M93 Plot1 CPU complete + own wet restart (on the E40 M93 INITIAL) is M93 evidence.
- **Recorded as** `historical_M93_recipe`.

## Historical M92 recipe (CW-E36; not M93 or M94)

The CW-E36 recipe (manifest `cec86f4b9baf3b95010f5ee5a5c975a72ea879d4c97ba36bf36733bea629e930`, patch
`1c40312ee9763326d5250bd438a763c3ed9e46e0dd1e13a80d53f648111e0251`, package digest `e4398b27…ae88`) was independently
accepted as the bounded M92 runtime delivery (`agent_handoffs/tasks/conservative_water/cw_e36_independent_acceptance.json`),
and root delivered it as runtime-only bundle `3cdf2943c1c765a6788b4bf8c3f3c0324571f5026184f00b0e7fd36c72163ab2`
(`cw_e37_m92_runtime_resolver_acceptance.json`).
- **Its memory record** (owned-ledger helper peak M91 386,560 B → M92 141,312 B) describes M92, not M93 or M94.
- **Large-case refusals:** B76 / M92 H4 (attempt 14 after 13 accepted seconds, fourth face allocation) and B93 / M92
  H5 (attempt 6 after 5 accepted seconds, `make_proportional_mass_vector`).
- **Small storms:** the C30 / C31 B93 / M92 Plot1 small storms are M92 evidence.
- **Recorded as** `historical_M92_recipe`.

## Historical M91 recipe (CW-E33; not M92, M93 or M94)

The CW-E33 recipe (manifest `9d71822075de73e0cc47e04fe05f84c4d58ec0f22679df30e1253ef22857fe37`, patch
`d3cb04bc24eacd53fe0af95e505bf3b23ccad328a5076792a763b6b6455e348f`, package digest `a9de77ce…4ff5`) was independently
accepted as the bounded M91 runtime delivery (`agent_handoffs/tasks/conservative_water/cw_e33_independent_acceptance.json`),
and root delivered it as runtime-only bundle `901a8abec25aceac3b372a0903c0ab05aa8d10acdc9c16189d473e23f4adb68e`
(`cw_e34_m91_runtime_resolver_acceptance.json`).
- **Its memory record** (true helper peak M88 4 C + 2 mask, B90 4 C + mask, M91 3 C + 2 mask) and its large-case
  refusal describe M91, not M92.
- **The refusal:** the Chastre H3 GPU reference refused at attempt 1 in `_accumulate_water_transfer_into_owned`
  (`cw_h3_root_gpu_refusal.json`).
- **Recorded as** `historical_M91_recipe`.

## Historical M88 recipe (CW-E29; not M91, M92, M93 or M94)

The CW-E29 recipe (manifest `d892c3497f161ff6db481b440eed2a11a108575088cf85247fe27e9f2a251cf1`, patch
`8efcd7e221cda23ee5696a68fb73b19f825c444149064e1be1ad9238ae08c3a0`, package digest `89a22772…25fe`) was independently
accepted as the bounded M88 runtime delivery (`agent_handoffs/tasks/conservative_water/cw_e29_independent_acceptance.json`),
and root delivered it as runtime-only bundle `a586c82382ffe02042a1025af65d47801d8738110a9eabe1001db83e17f8689f`
(`cw_e31_m88_runtime_resolver_acceptance.json`). Its status (the B88 guard; the B85 2-class-array saving; the whole
small-fixture peak unchanged) describes M88, not M91, and is recorded in this manifest as `historical_M88_recipe`.

## Historical M85 recipe (CW-E26; not M88, M91, M92, M93 or M94)

The CW-E26 recipe (manifest `103fa77f2d8631a61f67e7d856300e605df1bf4882f00d60ebc3c9628c6adcf8`, patch
`7d75da4efd4127cbd60396ab8f80eb33ffcf821e7b7624d8838bfde289447d4e`, package digest `17044c8a…b29c`) was independently
accepted as a bounded, exactly reproducible M85 recipe (`cw_e26_independent_acceptance.json`), but NOT for final
adoption: M85 raised `TypeError` for 0-d NumPy and masked 0-d availability / deposition pools that M82 accepts (CW-E26
review, reproduced by root). Production storm arrays were unaffected. CW-B88 resolved this in M88. No M85 bundle was
built. Its status describes M85, not M88, and is recorded in this manifest as `historical_M85_recipe`.

## Historical M82 recipe (CW-E22; not M85, M88, M91, M92, M93 or M94)

The CW-E22 recipe (manifest `6ba0666164375bb6c4346587bbfe9e494db5e229a4d57531271c9b33a37f4895`, patch
`1d70fe3d9e3f6cd140a4e5ec23d01dc0a36073f221c1605e23eeb60312323eb1`, package digest `1667efe2…ad5b`) was independently
accepted as the bounded M82 runtime delivery (`agent_handoffs/tasks/conservative_water/cw_e22_independent_acceptance.json`),
and root delivered it as runtime-only bundle `8a29dc39f0dd482e5710f98b36d2e71a772b28c21005c5b6d6215672d25a94f6`
(`cw_e23_m82_runtime_resolver_acceptance.json`). Its measured B82 status (1 C earlier, 2 C at each deposit-phase
availability closure, whole-call fixture peak unchanged, negative control SKIPPED, no fit claim) describes M82, not
M85 or M88, and is recorded in this manifest as `historical_M82_recipe`.

## Historical M79 recipe (CW-E19; not M82, M85, M88, M91, M92, M93 or M94)

The CW-E19 recipe (manifest `e4e33ba69facc56a432ac3e513995c80ee232ff6b4638316b169dd20ca64cd9a`, patch
`b5286c74e50753ef9f4035f2978782bf4b991caa9276116d6b8c73cb3c1ebf7f`, package digest `7a4b65a4…3107`) was independently
accepted as the bounded M79 runtime delivery (`agent_handoffs/tasks/conservative_water/cw_e19_independent_acceptance.json`),
and root delivered it as runtime-only bundle `89c6e3f0de01ac7ff7622d13004a1be2f0aa0b0d72bd995a7c13b784000bd94c`
(`cw_e20_m79_runtime_resolver_acceptance.json`). It describes M79, not M82, and is recorded in this manifest as
`historical_M79_recipe`.

## Historical M75 recipe (CW-E16; not M79, M82, M85, M88, M91, M92, M93 or M94)

The CW-E16 recipe (manifest `788ae77372108b69452984da32f14750b369fcea7ab04a7982e64b87240d4f51`, patch
`4b0fbbd5fdcebfe7f7fd2a95c086724e61289bcce821c243e90fcd889bd041ab`, package digest `c39e8b3b…4fbf`) was independently
accepted as the bounded M75 runtime delivery (`agent_handoffs/tasks/conservative_water/cw_e16_independent_acceptance.json`).
It describes M75, not M79 or M82, and is recorded in this manifest as `historical_M75_recipe`.

## Historical M67 runtime delivery check (CW-E12; not M75, M79, M82, M85, M88, M91, M92, M93 or M94)

Root delivered the CW-E11 M67 recipe as runtime-only bundle `9bf97f1f0bd332c2f5ab5ceb68747df8c56ea6f6169b599d90e3ca3ff491a70a`
(package digest `b08b24d1…9caf`), resolved in both MAPLE interpreters without installation
(`agent_handoffs/tasks/conservative_water/cw_e12_m67_runtime_resolver_acceptance.json`). It describes M67, not M75, M79 or M82.

## Historical M44 runtime delivery check (CW-E10; not M67, M75, M79, M82, M85, M88, M91, M92, M93 or M94)

Codex exported the reconstruction with the CPU environment's actual distribution metadata, materialized it, and resolved the cache in both existing MAPLE interpreters. No environment installation or physical GPU execution was involved. The runtime-only bundle id is `43aa5ce6aad964aafed7ea7c0bca30d4c7e4eb992e589bc42eec2de43ce7a329`; its package digest remains `44e4f4c62d0262bfed57973640a3378604e9fbeb7923eb53597319838be8630a`. It contains 341 project files (340 runtime files plus the reconstruction receipt), and its archive is 8,376,320 bytes, SHA-256 `f23cb257edcb76ac7d767b35c031fe57a13954718efd4fa7c4945d00bf4d5169`. Evidence: `agent_handoffs/tasks/conservative_water/cw_e10_runtime_delivery_receipt.json`. This is dependency-delivery evidence, not a completed large-storm result.
