# Windowed bed support for the coupled pond event

The pond/splash event now accepts MAPLE's actual resident-window voxel bed. Select it explicitly with
`PondEventControl(bed_storage="windowed", ...)`; dense remains the default. The legacy benchmark and method-5 event defaults
are unchanged. This connects water, wet sediment exchange, dry splash, hydraulic topography refresh and checkpoint/complete
handoff to one native MAPLE sediment authority. No wind code is copied.

For fewer store writes, select `deep_publication="boundary"` and an explicit
`deep_publication_high_water_records` strictly below the native journal cap. Physical exchanges still occur every timestep.
`evolve` publishes on its return by default; `publish_at_return=False` permits a bounded carried journal. `step` may return a
pending journal. Explicitly call `state = event.publish_storage(state)` before checkpointing such a state. Neither publication
nor checkpointing causes an accounting commit or hydraulic refresh. Only the existing maximum per-cell absolute net displacement
from the last actual routing surface, `>= 0.00075 m`, triggers hydraulic refresh.

```python
control = PondEventControl(
    backend="numba",                 # or "cuda" with actual CuPy-resident native state
    bed_storage="windowed",
    deep_publication="boundary",
    deep_publication_high_water_records=5_000,
)
# Construct PondEvent with the existing native BedContext, forcing, column physics and topology.
result = event.evolve(state, end_s)
state = result.state                # published by the default return policy
save_pond_checkpoint(path, state, event, identity)
```

Prepare the native window using MAPLE's constructors and maintained height packets; a deep-store handle alone is insufficient.
Moving a pending journal between host and device is refused because its binding belongs to the original arrays: publish the
accepted state before that explicit transfer and use MAPLE's `move_prefix_packets` with its tree transfer. Checkpoint loading
performs the required packet transfer itself. There is no implicit backend conversion during resume.

Read [core interfaces](core.md), [checkpoint format](checkpoint.md), and [qualification](qualification.md). New tests are
`tests/pond_windowed` and `tests/pond_windowed_checkpoint`. Older module comments describing the original dense-only milestone
are historical; the explicit storage control and these documents describe the current opt-in implementation.

## Important limits

* The characteristic transport/phase arrays are still dense over cells, classes and bins. This connection does not establish
  that the full Chastre coupled model fits in memory, nor optimize its multi-face GPU kernels.
* Checkpoints reference immutable external deep-store versions. Retain the referenced stores and versions. They are not portable
  standalone archives; no automatic garbage collection or tolerance-based dropping is added.
* Snapshot validation streams deep chunks at I/O boundaries and may be expensive. Normal hydraulic height queries use maintained
  packets, the active layer and the resident window; they do not scan the whole buried bed.
* `full` and exact `overlay` storage are supported on the qualified MAPLE source. No cap is raised automatically. A single native
  exchange too large for a clean journal fails. A recognised carried-cap refusal gets one prior-state publication and a retry at
  the same physical dt; an independent genuine physical rejection still follows ordinary bounded adaptation.
* The windowed native pickup view does not expose an independent touched-bed change. Its splash touched-change/residual fields
  are explicitly unavailable (`None`); global native class inventories and mobile/export budgets remain independently checked.
* A failed attempt after a prior-state flush can leave an unreferenced immutable version. The caller's state and all old referenced
  versions remain valid. No speculative candidate is published before its physical and host bookkeeping checks pass.
* Avalanching, periodic/stage-controlled boundaries, sediment inflow, vegetation growth, nutrients and wind scheduling remain
  outside this event's qualification. MASY owns wind/water scheduling and invokes the actual models independently.
