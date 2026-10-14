# CW-B104: net-elevation-only hydraulic rerouting (candidate; not adopted)

**Base.** The accepted, root-frozen C1 tree (`cw_b102c1_frozen`, package `e9a36d6a…`) with this change. MAPLE M94
(`eb14da66…`) is unchanged. The dry-only commit-separation flag is not used.

## Contract

Water rerouting happens ONLY when the maximum, over active cells, of the absolute NET elevation displacement from the
surface used for the LAST hydraulic routing build reaches the actual MAPLE `pending_elevation_trigger_m` of the bed
context (default 0.00075 m). The comparison is `>=`, MAPLE's own pending-elevation convention
(`surface/topographic_commit/triggers.py`). This is a physical geometry threshold, not a conservation tolerance.

- **Cancellation.** Gains and losses within a cell cancel; different cells never cancel.
- **What never reroutes on its own:** elapsed time, gross active-layer turnover, report / checkpoint / publication
  boundaries, and ordinary or final forced commits.
- **What is unchanged:** MAPLE's sediment exchange every accepted step, its commit triggers and cadence, its checks,
  conservation, composition and availability.

The run identity carries `routing_refresh` (contract `cw-b104/net-elevation-only/1`) with the actual threshold.

## State

`SedimentEventState.routing_source_elevation_m` is the HELD hydraulic routing source.

- **What it is:** the `(ny, nx)` float64 committed elevation, in the graph namespace, that `graph` was built from.
- **When it is set:**
  - by `initial_event_state` at initialization, as the bed's committed elevation; the given graph must be the graph
    of that source (one host rebuild, compared on the graph's host evidence);
  - replaced only by an actual reroute.
- **When it is checked:** at every public entry (shape, dtype, namespace, finite). `evolve_sediment_event` also
  rebuilds the graph from the source once on the host and compares it with the state's graph. A missing or
  inconsistent source is refused before any work, never guessed.
- **What it is not:** never mass, never a second terrain authority, never a voxel copy.

## Per accepted step (`evolve_sediment_event`, commits enabled)

1. **Gate.** `routing_gate_m` computes the exact current committed surface, `sediment_bed.current_routing_surface_m`
   (MAPLE's commit arithmetic `reference + (surface - reference)`):
   - windowed: the PUBLIC exact height API `windowed_committed_surface_elevation_m(..., trusted_packet=True)`, which
     uses the height packets plus the resident window and the active layer, with no deep read and no voxel-depth
     scan. It is the same trusted call MAPLE's staged commit makes, and it is valid on states carrying a pending deep
     journal;
   - dense (small cases): MAPLE's `diagnose_combined_surface_state`.

   It then takes the maximum over active cells of |surface - source| through ONE counted scalar read.
2. **Commit.** `commit_bed` evaluates MAPLE's triggers as before. If the gate reached the threshold and MAPLE did not
   trigger, a genuine forced MAPLE commit is made (reason `net_elevation`).
3. **Reroute decision.** After ANY commit, the routing is refreshed iff the POST-commit committed elevation is at
   least the threshold away from the source on some active cell. Without avalanching (windowed always; dense when
   disabled) this equals the gate bitwise.
   - **On a reroute:** the graph is built from the new committed elevation (`routing_graph_from_elevation`, the exact
     `refresh_routing` arithmetic and policy). The network and grid are rebuilt, the discharge is re-initialised from
     the new conveyance and the unchanged depth, the CSR phase is rebound, and the CUDA context is prepared before
     publication and adopted after it. The hydraulic diagnostics are refreshed and the source is replaced.
   - **Otherwise:** the commit publishes the bed only. Graph, network, grid, discharge `q`, phase binding, CUDA
     context and source are untouched; no geometry diagnostic is refreshed.
4. **Accounting is unchanged for every commit:** the water-depth / volume checks, the commit transfers (recorded for
   every commit), the ledger totals and the history fold.

## Commit records and counters

**Commit record keys:**
- `commit_reason` (`maple_trigger` | `net_elevation` | `final_forced`);
- `rerouted`;
- `net_displacement_before_commit_m` (the gate value);
- `net_displacement_m` (post-commit);
- `reroute_threshold_m`.

**Result fields:**

| Field | Meaning |
|---|---|
| `n_reroutes` | commits that refreshed the routing |
| `n_net_commits` | extra commits forced only by the gate |
| `n_routing_gate_evaluations` | gate evaluations |
| `max_unrouted_net_displacement_m` | largest displacement seen without a reroute |
| `routing_source_origin` | `entry` until the first reroute, then `net_reroute` |
| `routing_source_t_s` | time of the last reroute (None until then) |

`n_commits`, `n_forced_commits` (the caller's final forced commits) and `n_graph_changes` keep their meaning.
`commit_wall_s` now includes the per-step gate.

## Completion, pause, checkpoint, restart

- **`finalize_event`.** The terminal deposition and the forced final commit (reason `final_forced`) are an
  accounting commit. It reroutes only if net driven.
- **Wet pause and checkpoints** do not commit and do not reroute.
- **Bundles** carry the source in every encoded state (`routing_source`). Loading:
  - rebuilds the graph from the SAVED source, never from the latest bed;
  - requires the saved graph digest;
  - keeps the saved discharge, phase and hydraulics.

  A state node without the source (written before this contract) is refused.
- **Saving (CW-B104C1/C2).** `save_checkpoint` rebuilds the graph of every saved state from its terrain reference
  and held source once on the host and compares it with the state's graph BEFORE anything is written. The rebuild is
  deduplicated per identical (graph, source, terrain) object triple (C2: terrain added; a distinct terrain object is
  always checked, and one with equal content is accepted). A mismatch
  is refused; no bundle or staging directory is left and the caller's item is untouched. A lagged source (different
  from the committed bed) is valid. Load adds no rebuild: it already rebuilds from the saved source and checks the
  saved digest, so a source corrupted on disk is still refused there.
- **Namespace conversion** (`event_backend.state_to_namespace`) rebuilds from the held source as well. CW-B104C1: the
  rebuilt target graph's host evidence must equal the input graph's (no second build); a mismatch is refused and is
  never repaired under the old binding.

## Costs (measured on small fixtures; see the B104 report)

**Persistent:** one `(ny, nx)` float64 array.

**Per accepted step:**
- the gate's transients: `(ny, nx)` arrays, freed before the commit;
- one counted scalar read on the windowed path.

**Dense CuPy path:** the public dense diagnostic validates the partition, costing 11 counted scalar reads per call.
It is used for small dense cases only.

**Avoided:** a graph build, network, grid, `q` re-initialisation and CUDA re-preparation for every commit that does
not reroute.

## Limits

- No large / Chastre run, fit or speed claim.
- After a skipped reroute, later depth, transport, bed and commit quantities legitimately differ from the C1
  old-cadence oracle. No storm identity with C1 is claimed.
- A displacement within cross-backend round-off of the threshold could split CPU and GPU routing schedules. Not
  observed on the fixtures; the correspondence tests would detect it.
- Comparators / tools that rebuild the graph from the latest committed bed (R5 / R7 / G7 / R8) must be adapted to
  the saved source before use on net-routing outputs. This is a separate task.
