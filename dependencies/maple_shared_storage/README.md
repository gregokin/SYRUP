# MAPLE shared-storage runtime pin: merged 39bcf79

Target native MAPLE revision: **39bcf79e1e67b6ed960f78a923ddfedf1d173e68**.
Package digest: **af54626ab432670539ed71480fd884f13581dfdfcfec63d7c85053ca215adfec** (351 files).
Runtime scope: `src/maple` plus `pyproject.toml`, 352 files. No additional SYRUP patch is applied to this merged source.

This is a separate recipe from `dependencies/maple_water` (the historical 24c6 water reference). Keep the old recipe for
reproduction; never silently fall back to it on failure of this pin. MAPLE's merge was initially local;
the upstream developer handoff now records it as pushed. Consumers still need the exact committed objects or the
separately transferred verified bundle, not an assumed remote HEAD. This qualification did not push either repository.

## Reconstruct with the existing guarded tool

The unchanged recipe engine requires a real net patch. This recipe uses the actual committed main predecessor
`b3e8756c26d3661617d7e5ef23dcd29ed08e4687` plus the authentic Git diff to merged `39bcf79`: 16 new and 16 modified runtime
files. The result was independently compared against a direct `git archive 39bcf79` for every runtime byte. The target is
39bcf79, not a new dirty upstream variant. `manifest.json` pins the base Git blobs, net patch and all target hashes.

The trust anchor is the **fixed** manifest SHA256
`877826de3715680cc0b27c40d3c07dd7b2fb0ba0388b63bce74142c8dd583441`.
Do not compute a replacement trust anchor from whatever manifest happens to be on disk.

Use the existing standard-library engine's explicit API; its CLI default intentionally remains pinned to the old recipe:

```python
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

root = Path('/home/okin/SYRUP')
spec = spec_from_file_location('shared_recipe', root / 'tools/reconstruct_water_maple.py')
recipe = module_from_spec(spec)
spec.loader.exec_module(recipe)
recipe.reconstruct(
    '/home/okin/MAPLE', '/tmp/new_maple39_runtime',
    recipe_dir=root / 'dependencies/maple_shared_storage',
    expected_manifest_sha256='877826de3715680cc0b27c40d3c07dd7b2fb0ba0388b63bce74142c8dd583441',
)
```

The destination must be new and its parent must exist. The engine reads Git objects only, refuses unsafe paths, a mismatching
manifest/patch/file, or an existing destination, and publishes atomically without replacement. Nothing in the MAPLE working
tree is changed. The generated `maple_water_reconstruction.json` receipt is outside the pinned 352 runtime files.

## Qualified local dependency delivery

The qualification used the same exact 352 runtime files from a direct committed archive. Existing
`maple_syrup.dependency_bundle` exported and materialized the bundle. The local artifact is:

- `outputs/dependencies/bundles/maple39_shared_storage_20261019/`
- bundle ID `37a1bc69bfa034be7cd512d8aa7c30d33682b7c053a53c3b0c91edc34d59a96f`
- materialization `outputs/dependencies/content_cache/37a1bc69bfa034be7cd512d8aa7c30d33682b7c053a53c3b0c91edc34d59a96f/`

These are local external dependency artifacts, not copied wind code inside SYRUP and not files supplied by a Git checkout.
Transfer the complete verified bundle when needed; archive/manifest identities are in the qualification lock.

The installed live MAPLE metadata was refused because it lacked the new windowed dependency declaration. It was not edited.
Genuine new metadata was built from the exact source using system setuptools 68.1.2 and the pinned wheel 0.45.1 build helper
(helper SHA256 `708e7481cc80179af0e556bbf0cc00b8444c7321e2700b8d8580231d13017248`). All 351 package files in the built wheel
were independently verified against the committed archive. No package was installed and neither MAPLE environment was changed.
The wheel artifact SHA256 and the exact selected dist-info hashes are recorded in the lock and task evidence. A rebuild may
have a different wheel ZIP hash; it must still reproduce every pinned runtime byte and truthful matching distribution metadata.

To regenerate an artifact, use a fresh project containing exactly the 352 manifest runtime paths, leaving reconstruction/build
receipts outside that project. Build genuine matching dist-info from an isolated copy of that source, then use the existing
`dependency_bundle export`, `verify`, `materialize`, `verify-materialized`, `env` and `check-runtime` commands. A different
project selection, build metadata, tool or Python identity yields a different bundle ID; never relabel it as this bundle.
See [delivery rules](../../docs/cw_e2_dependency_delivery.md). No hand-written substitute metadata or origin-guard waiver is allowed.

Select the materialization's existing editable overlay and source path before importing MAPLE; use the environment printed by
`dependency_bundle env`. Set `MAPLE_SYRUP_EXPECTED_MAPLE_ROOT` to its `source` directory and run `check-runtime` in the same
interpreter. Both existing MAPLE CPU and CuPy environments passed that origin check. These are source-delivery checks,
not evidence of GPU physics by themselves.

See [consumer qualification](../../docs/maple39_shared_storage_qualification.md) and `qualification.lock.json` for the exact
SYRUP candidate identity, executed tests, explicit memory-admission scope and remaining limitations. Old checkpoints retain
their original source identities and are not migrated or rebound by this recipe.

## Matching SYRUP runtime

The tested consumer is SYRUP Git base `b4ee14e5bf37814efb0d754de17b7ea604b1cd74` plus
`SYRUP_b4ee14e_runtime.patch` (SHA256 `95a091ca8ee7ccbfdcfd95e7566d7acae9e3327018ba0c2651ffffb9b3360b8e`).
This patch reconstructs the accepted uncommitted **SYRUP** runtime, not a modification of MAPLE. Its resulting
64-file package digest is `c650b786b03601c601b47182f8e3b24b8ad87f9b55b40770d4a38b13cb254392`.
Reconstruction was independently checked in a fresh Git archive of that base; all 65 runtime/config files matched.
The accompanying tests and documentation are part of the working-tree change and are recorded separately in the lock.
After a future authorized commit, record that actual commit without altering the previously tested source identity.
