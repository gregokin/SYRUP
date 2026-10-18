# Dense checkpoints on current MAPLE (MASY-03 prerequisite A)

Status: independently accepted candidate on top of SYRUP `2c60c73b01fe1c0cef576987d52010e84058c568`. Claude authored the change; Codex independently inspected the diff and executed the checks below. Adoption into the active checkout is recorded separately.

## Problem

Both native checkpoint registries imported `CompactSedimentLedgerState` unconditionally:

- `checkpoint._registry`: legacy `maple-syrup-checkpoint/2`.
- `pond_checkpoint._native_registry`: pond `maple-syrup-pond-checkpoint/1` and `/2`.

Current committed MAPLE `5b1a052e` has no such class. On it, every Dense save or load failed with that `ImportError` before writing anything. `tests/masy_compat` asserted that capability gap.

## Change

One helper, `checkpoint._native_ledger_types()`, is now shared by both registries. `pond_checkpoint` imports it from `maple_syrup.checkpoint`. The helper reuses the bed classifier's accepted binding rule, `sediment_bed._maple_voxel_class`, on the public module `maple.core.types.sediment_ledger`.

| Binding in `vars(module)` | Result |
|---|---|
| `SedimentLedgerState` is a class | included; required |
| `SedimentLedgerState` is not a key | `AttributeError` ("lacks required SedimentLedgerState") |
| `CompactSedimentLedgerState` is a class | included |
| `CompactSedimentLedgerState` is not a key (a module `__getattr__` is not consulted) | omitted from both registries |
| either name bound to a non-class (None, number, function, instance, module, ...) | `BedIntegrationError` ("is present but not a class") |
| importing the module fails (`ImportError`, `ModuleNotFoundError`, or any error raised while executing it) | propagates unchanged |

How these errors surface:

- **Nothing catches these errors, and they are never translated.** That includes `SedimentEventError` and `PondCheckpointError`.
- **Saves.** Both registries are built before any checkpoint file or directory is created. A refused save writes nothing. The legacy writer does not even create the parent directory.
- **Loads.** Both registries are built only after the manifest and payload checks. A load never writes.

What is unchanged:

- Classes are taken by identity: a public re-export of the genuine class is accepted. Whitelist keys are still each class's `__name__`.
- Encoding still requires `registry.get(type(value).__name__) is type(value)`. A foreign object that merely carries the compact class's name is refused.
- Decoding still requires a whitelisted type and its exact dataclass field set.
- Where MAPLE lacks the class, a serialized `CompactSedimentLedgerState` node is refused as an unknown type (legacy: "unknown checkpoint type or fields"; pond: "unknown native type or fields"). It is never coerced to the dense class.
- No schema, other whitelist entry, manifest key, codec, solver, storage, conservation tolerance or physics body changed. Only the registry construction changed.

Small behaviour change: a missing Dense class now raises `AttributeError` instead of `ImportError`, matching the classifier.

## Capability by MAPLE source

| | current MAPLE `5b1a052e` | reference MAPLE (has the compact class) |
|---|---|---|
| Dense legacy save/load/continue | enabled by this change | unchanged |
| Dense pond save/load/continue | enabled by this change | unchanged |
| compact ledger codec | not available (class absent; nodes refused) | unchanged |
| windowed pond checkpoints | not available (no windowed class) | unchanged; covered by `tests/pond_windowed_checkpoint` |

## Source binding and old checkpoints

- **Pond checkpoints** record the digest of the imported `maple_syrup` package tree and of `maple`. This change alters the SYRUP tree digest. A pond checkpoint written by `2c60c73`, or by any other tree, is therefore refused by this tree with "source pair mismatch".
- **Legacy checkpoints** are bound to the caller-supplied identity, and their schema is unchanged.
- **No automatic migration or rebinding** exists in either format. Resuming across a source change is not accepted here.
- **Old receipts and acceptance records** are historical and are not rewritten. These include `docs/masy_dense_compatibility.md` and `docs/masy02a_integration.md`, which describe the earlier blocked state.

## Limits

This change does not cover:

- combined wind/water restart;
- the MASY runtime lock (prerequisite B);
- the remaining MASY adapters;
- GPU;
- performance.

Only the two registries changed. Other optional-API imports, for example the windowed paths in `pond_checkpoint`, are untouched. They are reached only on windowed beds.

## Tests

`tests/checkpoint_compat/` is not in `pyproject.toml` `testpaths` (that file is outside this task's scope), so run it explicitly. The directory holds:

- `test_registries.py`: pure registry and refusal paths. Synthetic bindings are installed only as a separate module object in `sys.modules`, and teardown proves exact restoration. Save refusals are checked to write nothing.
- `test_current_dense_restart.py`: genuine Dense wet save, load and continuation through both codecs, with exact state and accounting equality; never skipped. Also unknown or forged compact nodes, malformed ledger fields, and source or identity mismatches.
- `test_reference_compact.py`: genuine compact codec round trip. It skips only where the class is truly absent.

`tests/masy_compat/test_pond_windowed_compat.py` now runs a real Dense pond round trip plus continuation on every MAPLE, in place of the capability-refusal branch.

## Codex verification — October 18, 2026 UTC

- Current committed MAPLE `5b1a052e`: 291 passed, 8 skipped.
- Genuine compact/windowed-capable reference MAPLE (package digest `24c6e1478ed732df22f448eddf8cb16c3b72f2406dc9b37a9414cee9a2d1a9ce`): 404 passed, 6 skipped. This is a labelled regression reference, not a fallback current runtime.
- Selections include the new compatibility tests, existing MASY classifier tests, native phase6 checkpoint tests and pond checkpoints; the reference also includes native windowed checkpoints. Current Dense wet round trips and continuation execute, rather than asserting an unsupported-capability refusal. Device tests and genuinely unavailable reference-only types skip explicitly.
- Both corrected assertions are test-only: synthetic module identity, and the documented nonphysical voxel surface cache reset. All other physical state, inventory, availability, water, native ledger, clock and accounting comparisons remain exact. All other production function/class ASTs are unchanged.

Exact commands, module origins, source hashes, raw streams, JUnit and lifecycle receipts are retained in the MASY task `agent_handoffs/tasks/masy03_prerequisites_20261018/{checkpoint_current_02,checkpoint_reference_02}`. Existing MAPLE CPU interpreter only; no installation, actual terrain/full storm, GPU or coupled run.
