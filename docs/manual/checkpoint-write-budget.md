# Optional native checkpoint write budget (P1)

`maple_syrup.checkpoint.checkpoint_write_budget(item, context, column,
sediment, identity, *, limits=CheckpointWriteLimits(...))` describes the
native checkpoint writer's prospective wire layout and serialization
increments. This is a source-only prototype pending separate CPU execution
and immutable-runtime qualification. It does not save or validate a checkpoint.

The first profile is an unfinished, exact `EventProgress` on the genuine
default Dense NumPy wet path: at most 64 cells, 6 classes, 16 voxels, 4096
report rows and 10000 storm steps; upwind/array/bisection, ordinary commits,
no forced-final commit, exchange history, phase payload, deep publication or
windowed storage. Existing native save/load functions retain their public
signatures, policies and other supported backends/storage profiles. They do
not select this optional helper.

Pass an exact frozen `CheckpointWriteLimits`. Each limit is a plain positive
integer bounded by the original 1 GiB uncompressed checkpoint cap, 32 MiB
JSON cap, 256 members per archive, 65536 nodes, depth 32, strings 1024 and
integers 256 bits. Smaller caller limits remain authoritative. Unsupported
layouts and exhausted limits produce a short `SedimentEventError`; they
never truncate logs, drop arrays or change a native physical rule.

The returned frozen/slotted records contain only scalar/tuple metadata:
`NpyMemberLayout`, `ArchiveLayout`, `CheckpointWriteBudget`. They confer no
physical, source, file, checkpoint or publication authority. There is no
path argument and no payload data read, `to_host`, graph digest/rebuild,
physical validation, commit, transport or filesystem operation in the helper.
Only bounded metadata scalar arrays are normalized for the native snapshot's
scalar fields. The caller retains the actual physical owners.

## One native encoding traversal

The original entire structural encoder is shared privately between writing
and layout. Writing retains its original conversion, finite-data checks,
native graph digest, bed divergence checks, default omissions, recursion and
error order. Layout alone performs pre-child type/node/depth/string/member
admission, requires the exact primary wet bed and substitutes a clearly named
64-ASCII-hex size placeholder for the graph digest. The same expression
computes each `aNNNN` name and `{'array','dtype','shape'}` reference in both
modes. Every occurrence gets a member even when several occurrences share
one array object or backing allocation.

The helper does not construct a second physical codec or copy a snapshot
writer. A closed source-qualified table describes the one actual Dense MAPLE
subcall: its 16 base fields, 3 availability fields and 3 water fields, in the
native dictionary order. No mobile, vegetation, scheduler, threshold or wind
optional fields are invented. Snapshot schema 2, exact-history schema
`maple-phase4r4-v4`, the native reason string and actual metadata defaults are
part of that source contract.

## Wire sizes

For an admitted occurrence, `data_bytes = product(shape) * itemsize`.
NumPy's actual `header_data_from_array_1_0` and `_write_array_header` routines
produce/count the full NPY header through the data offset. No independently
copied NPY framing, alignment or version-selection routine is used. Header
metadata is admitted first: rank at most 4, bounded plain axes, positive
nonstructured/nonobject legacy dtype without dtype metadata, C contiguity,
header at most 10000 bytes. Metadata-bearing and custom dtypes refuse before
NumPy can warn or select its object/pickle descriptor path.

NumPy tests C contiguity before F contiguity. One-dimensional, zero-dimensional
and other arrays with both flags are accepted with `fortran_order=False`.
F-only and other non-C layouts refuse instead of silently copying payloads.
Snapshot scalars use the actual `np.float64/int64/bool_/bytes_` and
`np.asanyarray` normalization. Empty byte scalar itemsize must be positive on
the actual qualified dependency; neither S0 nor S1 is assumed by this helper.

For each member `N = npy_header_bytes + data_bytes`. The original native raw
cap applies to the sum of **NPY bytes**, including headers, over both archives.
For the qualified NumPy 2.5.3/CPython 3.12 regular seekable-file writer,
`np.savez` uses ZIP_STORED and forces a 20-byte local ZIP64 extra. With safe
ASCII names, at most 256 members and all offsets/sizes below ZIP64_LIMIT:

```
archive_bytes = sum(N) + 22 + sum(96 + 2*len(name + '.npy'))
```

The terms are the native local header 30, forced local extra 20, central
header 46, names twice and end record 22. No central ZIP64 extras are needed
under the enforced limits. At the admitted maximum name length 200, this
overhead is at most 126998 bytes, below the existing native 327680-byte ZIP
metadata slack. This formula is not generalized to compressed/streamed ZIPs.

`checkpoint_json_bytes` counts the actual JSONEncoder with indent 2,
allow_nan false, ensure_ascii true and one final LF. Source raw metadata is
bounded before encoder iteration or materialization. The helper refuses
unless `os.linesep == '\n'` on POSIX. Ensure-ascii makes the body ASCII, so its
byte count does not depend on a locale encoding; newline translation is not
assumed portable. Signed zero, DEL/control characters, escapes and non-BMP
text follow the real encoder.

Snapshot provenance is exactly the four native keys
`syrup_checkpoint_schema`, `requires_companion`, `identity`, `backend`, using
`json.dumps(sort_keys=True)` with default comma-space/colon-space separators.
Identity is counted there and again in `checkpoint.json`. Size-only hash
placeholders describe the same 64 ASCII characters as real hashes; they are
not fabricated digests and are never passed to a loader.

## Serialization memory and caller responsibilities

Each named increment is separate and included in
`additional_serialization_upper_bytes`:

| Returned term | Source-derived role |
| --- | --- |
| `encode_finite_check_bytes` | Maximum companion array element count plus one bool reduction scalar for the unchanged `isfinite(...).all()` |
| `whole_array_serialization_fallback_bytes` | `2*max(data_bytes)` over both archives, conservatively retaining whole-array iterator/bytes fallback |
| `nominal_iterator_buffer_pair_bytes` | `2*max(max(floor(16MiB/itemsize),1)*itemsize)` over both archives; requested parameter, **no chunk-cap claim** |
| `header_scalar_metadata_bytes` | Admitted native headers, forced/local/central metadata, normalized small scalars and provenance scalar copies |
| `encoded_metadata_upper_bytes` | Actual bounded wire tree/reference tables, layout records and native ZipInfo/directory metadata, using CPython object/pointer sizes |
| `json_overlap_upper_bytes` | Counted actual JSON chunks/table, string/join/write overlap and maximum bounded atom escape |
| `graph_digest_scratch_bytes` | Largest actual graph array's metadata byte length, for the separately invoked native `tobytes` scratch |

The two archives are written sequentially, so the iterator terms use maxima
across both, without subtracting overlaps. Array occurrence bytes remain
distinct from retained input backing bytes. `host_payload_copy_bytes` is zero
only for this admitted exact NumPy path: native `to_host` returns the existing
NumPy array. Model arrays remain owned by the caller/native objects.

The helper is not whole-operation or whole-heap admission. Before calling
the unchanged native writer, the caller separately reserves its accepted,
staged/entry and wet owners, qualified native physical validation/graph
rebuild/snapshot work with the returned successful-call multiplicities,
this serialization increment, disk/inodes for its containing attempt and
retained generations, and finite OS AS/RSS/headroom/CPU/wall/output limits.
The native snapshot's finite checks belong to that separate validation
reservation, not the companion finite-check term. Failure frames may retain
owners; catching an error is not evidence that their memory is freed.

This source contract is pinned to immutable current SYRUP/MAPLE and the
qualified NumPy/Python source and compiled environment. Version/API constants
inside the helper are an extra refusal check, not proof that arbitrary patched
binaries or changed source are compatible. A caller must establish the exact
source/environment binding before import or use. Existing fixed native-source
guards must be qualified afresh for this candidate; old runtime identities
must never be rebound to it.

P2 archive audit implementation, P3 genuine finish-witness recovery, P4 store
transaction/publication, active restore and checkpoint process qualification
remain separate. Existing save/load cleanup and durability behavior is not
changed by P1.
