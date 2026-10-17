# Chastre flats, pits and boundaries

Updated October 16, 2026. This is a scientific qualification and follow-up, not an adopted physics change.

The evolving-bed GPU run reached 10,000 simulated seconds and stopped with `IncompleteEventError`. Water and sediment were retained. The last fully accepted interval ends at 9,600 s; its saved routing graph reconstructs exactly with the pinned public helper. No GPU allocation failure was recorded.

## What the case currently means

The rotated Chastre case deliberately has **closed boundaries**. Its 5,336 export flags occupy only the nodata ring. Nodata cells are ineligible receivers, so the graph has zero outlets. The builder and recipe explicitly enforce this setup. Conservation and resource results remain useful, but zero export is not a catchment-export validation.

The current opt-in pit/flat policy stores water without spilling or backwater. At 9,600 s, terminal storage contains 26,411.567 m³ and its maximum depth is 297.76 m. That depth is an unrealistic consequence of this policy. An open external boundary and internal depression treatment are separate requirements.

The saved-state local-threshold audit finds 37,482 terminal cells: 37 flat sinks, 37,430 strict pits and 15 cells with no eligible neighbour. Of these, 37,242 hold depths above their lowest eligible neighbour's bed elevation. At the deepest cell, row 205 / column 1556, the local threshold is only 0.001491 m. These are **single-cell lip diagnostics**, not a multi-cell pond calculation or a prediction of water available for export or deletion. Claude independently reviewed the script, source arithmetic and recorded results; the review did not itself load the arrays.

A follow-up saved-array audit confirms the flat classification exactly matches the graph's own flat-storage mask. There are 37,476 wet terminal cells; the 15 isolated cells hold only 0.21123 m³. Strict-pit local thresholds have a median of 0.007379 m and a 99th percentile of 0.06946 m. Wet terminal depths have a median of 0.24463 m and a 99th percentile of 7.59862 m. These distributions reinforce the limitations of single-cell storage; they do not validate a replacement solver.

Moving cells still hold 19.2805 m³ at 9,600 s. One of the slowest selected cells, row 3 / column 1551, has a 0.06647 m depth and a slope of 1.722e-8 m/m. Its initial elevation equals its eastern neighbour's elevation. A retained net bed increase of approximately 17.22 nm makes it strictly downhill under D4 routing. This is a resolved value in the saved state, not proof of a floating-point bug. The tiny change did not by itself establish a domain rerouting trigger; the actual maximum per-cell net threshold remains 0.00075 m. Selected slow-cell examples do not explain every moving cell or prove completion by 20,000 s.

## Reuse from MAHLERAN

Source inspection of `topog_attrib.for`, `dynamic_topog_attribute.for` and `route_water.for` identifies `iroute=6` as an infilling/overtopping reference. A sink receives a bed-derived threshold equal to its lowest neighbour's elevation minus its own. Water above that threshold can activate flow toward that neighbour. Flat sinks have zero threshold. Cells already draining down a positive bed slope retain that slope, so this option does not resolve the selected near-flat sloping cells.

The reference is not a verified multi-cell pond solver. Its spill edge can lead to a higher bed neighbour, whereas current SYRUP CPU/GPU level ordering requires strictly descending edges. A direct transplant therefore needs a cycle policy. Distinguishing a bed-derived spill connection from its water-dependent flux activation may offer a design route, but the user's net-elevation-only rerouting requirement must remain explicit. The legacy source warns of instability; later edits exist, and no instability has been reproduced here.

The prepared initial Plot1 fixed-terrain benchmark uses routing option 5, which excludes dynamic option 6. Other recorded benchmarks include options 2 and 5; the initial setting should not be generalized to every run. No executed option-6 comparison is claimed.

## Next bounded work

1. The local-threshold, wet-cell, isolated-volume and graph flat-mask audits are complete. Any further basin audit must retain the distinction between local lips and multi-cell pond geometry.
2. Confirm which routing option the executed legacy benchmarks use, and inspect native Chastre boundary inputs before proposing a new open-boundary case.
3. Qualify small pit/flat examples against actual isolated MAHLERAN execution where feasible. Record physical limits separately from numerical agreement.
4. Propose explicit outlet and depression treatment with CPU/GPU costs, conservation, cycles and restart behavior before a new large physical experiment.

Keep all exact sediment changes and existing conservation/completion tolerances. No minimum slope, mass pruning, moving-water reset, sandpile adoption, new outlets or spill implementation is authorized by this document. An unchanged 20,000 s experiment remains unexecuted.

Evidence is retained under `agent_handoffs/tasks/chastre_recession_review_20261016/`: `root_boundary_graph9600.py/.json`, `root_slow_recession_cells9600.py/.json`, `root_slow_recession_terrain9600.json`, `root_mahleran_flat_source_binding.json`, `root_terminal_thresholds9600.py/.json`, and the Claude boundary and threshold reviews. The archived threshold script reproduces the earlier inline terrain JSON's selected values exactly; the original evidence remains unchanged. The original failed horizon receipt and accepted 9,600 s evidence remain under `chastre189_storm_20261016/`.
