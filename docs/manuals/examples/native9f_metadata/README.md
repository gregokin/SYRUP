# Native 9f distribution metadata (companion fixture)

Four metadata files, byte for byte as used in the actual committed-runtime qualification of the subsurface release, for the `distribution_metadata` setting of a `masy.runtime.v1` runtime file (`[sources.maple]`).

## Where they come from
They are the `distribution/` folder of the qualified, genuine native MAPLE 9f reference target (bundle id `55d1485a03c385240bab57edf7fba6cad5230628729b30b8a19d96d094ddaa5c`, MAPLE commit `9f466d2cfcd52991b24496cf67f4562d6268afa9`, package SHA-256 `fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e`). The MAPLE `pyproject.toml` of that commit has SHA-256 `8779c55d823bf610da9c9f087c7d5eaaf8d276bfbdc6fb428726e4e7c06350be` and was byte-identical for the earlier 39bcf79 and the 9f commits. They are not generated for, and do not describe, any newer MAPLE.

## The four files (the whole `distribution/` folder, nothing else)
| File | Bytes | SHA-256 |
|---|---|---|
| `METADATA` | 697 | `dc1b5653aab75868578b7d478b3d015b146ba1fe4ea4907cc26d806d68120cca` |
| `WHEEL` | 92 | `b59a1e1a3b56c5645f769944ec4ddde3954f94b3509ef6ca8989f1ee0c00cbc0` |
| `entry_points.txt` | 82 | `9430d41e28e96a53fa455b84c6a4b6122484d21e6ce8151c6c40854db190c870` |
| `top_level.txt` | 6 | `08ab2a56c8503c742c762013475ad2955377d2001a9fc03e4e8bc22e09af0844` |

These are the values pinned as `DISTRIBUTION_SHA256` in SYRUP's native-case profile (`maple_syrup.native_case_profile`, SYRUP commit `e40a1b8fc56a96920db4ffbd182d4068fa5de3e8`). `provenance_manifest.json` repeats them. This README and the manifest sit beside `distribution/`, never inside it: the folder must remain exactly these four files.

## How they are used
The runtime preparation hands this folder, as metadata input, to the actual source-bundle export of the MAPLE 9f clone. All content guards still run: the archive, project, package and `pyproject.toml` digests and the file census of the MAPLE source.

## What this is not
- Not an installation, and it changes no environment.
- Not a validation of an arbitrary MAPLE source or of third-party package versions.
- Not new classes, source content or a fallback; no metadata is invented for the latest MAPLE.
- Not what a new `pip install -e` would write: installer-generated metadata can differ and is not a qualified input.

Use it by path (see the practical guide, Section 14.10). A fresh dependency installation was not re-run for this fixture.
