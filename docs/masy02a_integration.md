# MASY-02A compatibility integration — 2026-10-17

Accepted into the SYRUP working tree, **uncommitted**. This is bounded compatibility qualification, not acceptance of a coupled wind–water simulation. Codex applied the reviewed patch and Claude-authored reconciliation after independent review and CPU/GPU verification. No MAPLE or MASY source was modified, and no model job was stopped.

## Baseline and preservation

SYRUP HEAD remains `e519ea62da81e6f14b7af39cb89f14281f27e461`. The incoming patch was checked with `git apply --check --whitespace=error-all` against the actual dirty tree before applying it. SHA256: `943a434f1644a605570f7a6157f60a8c37bb9404455f7b1e28adb8b514f2d278`.

There were 28 pre-existing dirty files, including the newer windowed pond implementation. All 27 files other than `pyproject.toml` remain byte-for-byte identical to the pre-task snapshots. That configuration file only prepends `tests/masy_compat` to its existing test paths; every prior path and all other bytes are preserved. In particular, existing `pond_bed.py`, `pond_event.py`, `pond_checkpoint.py`, `splash.py`, and their windowed tests/docs are untouched by this task. Production changes are confined to `sediment_bed.py` and `dependency.py`; neither was dirty before this task.

Before adoption, process inspection found no running SYRUP/MAPLE model job. GPU0 was idle before isolated validation and released afterwards. GPU2's unrelated activity was preserved. Authors and test workers have finished; there is no background continuation for this bounded task.

## Final changes and reconciliation

The classifier resolves the genuine public MAPLE class bindings. Exact native Dense and Windowed objects are recognized; subclasses, foreign representations and present-invalid class bindings fail. A truly absent optional Windowed binding permits Dense. Genuine native Windowed behavior, storage guards and source/checkpoint checks remain intact. See [the interface contract](masy_dense_compatibility.md).

`dependency.py` is byte-identical to the reviewed incoming patch. Independent review found one additional classifier defect in that patch: comparing a bound value to a private missing-marker object confused a present-invalid binding with absence. A reproducer confirmed that a Dense bed was incorrectly accepted. Claude replaced marker comparison with literal namespace key membership, removed the unused marker, and added three regressions. No other production logic changed.

The incoming pond tests assumed the older interface rejected Windowed in all three calls. Current `native_commit` and `pickup_transport_deposit` intentionally opt into Windowed; those tests now expect their actual maintained-height-packet guard. Physical qualification uses genuine native Windowed objects, never synthetic substitutes.

Independent mixed-suite testing also exposed a spy-fixture leak: lazy native imports captured a spy alias that survived teardown. Claude changed the fixture to resolve all entry modules before installing spies and explicitly undo/check all bindings. New isolation tests pin this behavior. Compatibility tests precede genuine windowed/pond/checkpoint tests in the same process to verify isolation.

All substantive implementation and fixture corrections were Claude-authored in one isolated writer tree. Codex independently inspected the diff, callers, source identities, failure paths, AST scope and test evidence. The exact author session and every launch/correction/report are archived under the task directory.

## Exact tested sources

Digests use the runtime algorithm `sha256(sorted relpath NUL file_sha256 LF)`, excluding bytecode/cache files. A Git HEAD alone does not describe this dirty SYRUP or the reference bundle.

| Source | Exact package SHA256 |
| --- | --- |
| SYRUP pre-task package (HEAD plus existing dirty work) | `b2c8be7378c135f6b54acb2e28f5d7e2821efe6aa03495e09d3f0934b515062a` |
| Final SYRUP, 63 package files | `fc789e240f7b0061e9ac80bc9078b7fe42d302fc8309ee5746073455f7f9f6bb` |
| Current MAPLE, committed `5b1a052e1b9a9edfed0760152aac2a743813e79a` | `97b3df72336e6313e8d89d0b7b20784b9a0f001e30a4fc3035360e6de6068612` |
| Genuine Windowed-capable reference MAPLE | `24c6e1478ed732df22f448eddf8cb16c3b72f2406dc9b37a9414cee9a2d1a9ce` |

Current committed MAPLE sources were frozen read-only in `agent_handoffs/tasks/masy02a_integrate_20261017/current_maple_5b1a052e`; live MAPLE package bytes match. The reference root is `outputs/dependencies/content_cache/7ee10bd81b0859a891ff8a476cab2c8824bb0964309695ca948d12d5ac4b0c71/source`, using its existing editable-distribution metadata. Reference bundle provenance includes upstream dirty changes; the package digest, not a guessed upstream HEAD, identifies it.

## Verification executed by Codex

| Final-source verification | Passed | Skipped | Failed/errors |
| --- | ---: | ---: | ---: |
| Current MAPLE, CPU: compatibility + Dense exchange + missing-API reporting + pond event | 186 | 13 | 0 |
| Genuine Windowed MAPLE, CPU: compatibility, pond, Windowed, checkpoint, splash/overflow and friction callers | 701 | 99 | 0 |
| Same reference selection, CPU/GPU enabled on GPU0 | 799 | 1 | 0 |

Skips on CPU reflect device-dependent tests and, on current MAPLE, genuine Windowed absence. The single GPU-run skip is the default dependency-probe test that deliberately does not exercise a device round trip when CuPy is present; the other selected GPU tests actually run. This is regression coverage, not a storm-performance benchmark.

Tests used existing `/home/okin/MAPLE/.venv/bin/python` and `.venv-cupy/bin/python`, not a SYRUP environment. GPU0 UUID was `GPU-5c8bd3d4-2885-82c6-a0f0-ad2b7f87b871`, with the existing pool cap of 9,092,045,209 bytes. Every final run records exact module origins, unchanged source manifests, all 36 required API checks and the metadata verdict. No wrong module origin was found.

Exact commands/argv, versions, JUnit, stdout/stderr and source manifests are in `agent_handoffs/tasks/masy02a_integrate_20261017/{accepted_current,accepted_reference,accepted_gpu}` and matching `.stdout`/`.stderr` files. The runners are `pytest_child.py` and `pytest_gpu_child.py` in that directory. Tests were run on frozen source-identical candidates, then all 63 adopted package files were checked byte-for-byte. A post-adoption current-MAPLE rerun and default test collection are recorded as `adopted_current*` and `adopted_collection*`.

## Remaining blockers and MASY interface consequences

- **Current MAPLE checkpoints remain unavailable.** MAPLE `5b1a052e` lacks `CompactSedimentLedgerState`, imported unconditionally by both `pond_checkpoint._native_registry` and legacy `checkpoint._registry`. Both imports already exist in the pre-task committed SYRUP. A broader current-MAPLE C1 run had 20 phase6 checkpoint failures from this same ImportError; the raw failures are retained. The final focused test asserts this exact unsupported capability and no output. That passing refusal test does not mean current-MAPLE checkpoint support passed. Genuine reference-MAPLE checkpoint tests do execute and pass. No fake ledger, fallback, codec change or fourth dependency requirement was introduced.
- **Immutable runtime/source metadata still needs qualification.** The actual live current MAPLE expected-root check correctly refuses its unresolved source root; the frozen committed-source copy correctly refuses an editable-install-root mismatch. API checks succeeding are not metadata acceptance. The existing reference metadata passes. No source guard was disabled or fabricated.
- **MASY adapters must pass canonical native objects.** Exact exported class objects are required; wrapper subclasses and foreign lookalikes are refused. Handle `BedIntegrationError` at these gates, including pond/event/checkpoint gates; an exact `PondBedError` handler alone does not cover classifier refusals. Recognized Windowed objects still need the matching storage declaration and maintained-height/deep-storage contracts.
- **Three Dense APIs are now explicitly required** (36 total); their names and parameters are in the contract document. No existing public SYRUP function signature or checkpoint schema changed.
- The SYRUP source digest changes. Existing strict checkpoint/source checks remain in force; this task does not authorize migrating an older checkpoint across source identities.
- Actual wind–water adapters, immutable production dependency selection and coupled-simulation acceptance remain MASY work. These small regressions do not qualify full-Chastre storms, memory fit or performance, and do not change any physical process.

Machine-readable acceptance is [masy02a_acceptance.json](masy02a_acceptance.json). Baseline manifests, raw failures, author records, independent review, exact preservation checks and task-only diff are archived under `agent_handoffs/tasks/masy02a_integrate_20261017/` (local evidence, not assumed tracked).
