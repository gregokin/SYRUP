# MASY Dense / Windowed compatibility contract

Integrated uncommitted on 2026-10-17 atop SYRUP `e519ea62` plus its existing windowed pond development. See [the executed qualification record](masy02a_integration.md) for exact source identities, tests, preservation evidence and remaining blockers. This is one compatibility fix, not coupled-simulation acceptance.

## Representation classification

`sediment_bed.is_windowed_bed` imports `maple.core.types.voxel` through `importlib.import_module` and reads the actual public module namespace. Import failures propagate. Module `__getattr__` cannot synthesize a missing class; only literal key absence counts as absence.

| Binding / object | Behavior |
| --- | --- |
| Optional `WindowedVoxelColumnState` genuinely absent | Genuine Dense remains usable |
| Optional Windowed present but not a class | `BedIntegrationError`, even for Dense |
| Required `VoxelColumnState` absent | `AttributeError` |
| Required Dense present but not a class | `BedIntegrationError` |
| Exact publicly bound native Windowed object | `True` |
| Exact publicly bound native Dense object | `False` |
| Subclass, foreign object or duck type | `BedIntegrationError` |

Identity is exact `type(column) is binding`, not `isinstance`. Genuine re-exports and aliases are accepted without inspecting definition names. Windowed is checked first; if both names bind the same class, that class is classified as Windowed. There is no speculative extra alias restriction. Arbitrary invalid objects, including a value identical to a former helper-private marker, cannot masquerade as absence.

Classification does not supply missing Windowed functionality or change storage ownership. Native Windowed pond callers still require `storage_kind='windowed'`, their maintained height packet and existing deep-storage/journal/source contracts. Unknown/subclassed objects raise the base `BedIntegrationError` before physical mutation. MASY should catch that base; `PondBedError` is a subclass for recognized pond configuration failures.

## Explicit Dense dependency APIs

The declared required API expands from 33 to 36 requirements:

| Native module | Symbol | Required parameter names |
| --- | --- | --- |
| `maple.core.types.voxel` | `VoxelColumnState` | Presence only; classifier additionally checks class binding |
| `maple.aeolian.flux.step` | `bed_inventory_by_class_kg` | `voxel_column`, `active_layer` |
| `maple.surface.topographic_commit.commit` | `is_already_committed` | `ledger`, `committed_state`, `voxel_column`, `active_layer`, `geometry`, `mass_resolution_kg`, `current_time_s` |

`check_required_api` checks names/signatures; it does not replace metadata/source qualification. There is no optional-import catch-all and no manufactured native class. Both production modules retain every existing function signature. Physics, accounting, storage codecs and provenance checks are unchanged.

## Regression coverage and limits

`tests/masy_compat` covers real Dense callers; strict class identity, aliases and binding failures; early refusal before native mutation; updated pond gates; genuine Windowed compatibility and storage mismatch; Dense checkpoint behavior on capable reference MAPLE; and explicit refusal on current MAPLE's missing compact-ledger capability. Synthetic bindings are confined to classifier/refusal tests, never physical Windowed qualification.

The native-spy fixture resolves modules before installing spies, explicitly restores originals and detects captured aliases. Its isolation tests precede genuine Windowed/pond/checkpoint tests in the verification matrix. Existing windowed and splash tests/code are preserved.

Current MAPLE `5b1a052e` naturally lacks Windowed and `CompactSedimentLedgerState`. This patch permits its genuine Dense operations; both existing checkpoint codecs remain blocked by the missing ledger, and immutable runtime metadata qualification remains unresolved. Reference MAPLE supplies genuine Windowed/compact-ledger classes and passes scoped CPU/GPU regressions. Neither source is claimed to complete the MASY adapters or accept a coupled simulation.
