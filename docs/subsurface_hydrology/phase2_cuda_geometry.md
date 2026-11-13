# Phase 2 / D3 — CUDA official geometry view, matrix/preferential rebind and SELECTED zero-time reconcile

> **Current status pointer (final documentation draft, 2026-11-13).** The text below is the author's original record and is preserved unchanged, including its "author-written / unrun" wording. Root's actual qualification of this component, with its exact tested scope and source identities, is in `phase2_cuda_geometry_acceptance.json`; where the two differ, that receipt governs. No whole-phase acceptance, publication or field validation is claimed by this page.

Status: author-written; CPU oracle tests and HOST execution of the rendered CUDA source through the plumbing double passed (NOT hardware evidence). Root compiles and runs the real GPU definitions
(`tests/phase2_cuda_geometry/test_gpu_definitions.py`, `SYRUP_RELEASE_GPU=1`).

- `floating_geometry_view(parent, D, ...)` now serves a CUDA prepared column: exact CuPy float64 `(N,)` depth of the owner's device. Order: current-device gate FIRST -> depth metadata -> admission (device owned depth 8N + flags N +
  the 32-byte packet; host 32-byte packet; reserves counted once; real default readers) -> private copy + flags -> the EXISTING resident validation kernel (finite/non-negative depth and every porous slab/pair of the new depth; one
  32-byte packet) -> typed refusal before anything is published. Static maps are shared; identity is a minted token digest (`identity_kind="geometry_view_token"`, `parent_sha256` = the prepared root; a view of a view binds the root).
  No download of D/W/maps/history; payload O(N) independent of L (tested at L = 65, N = 5003; Root's definition at N = 50003). The CPU default-reader admission fix is unchanged.
- `rebind_matrix_geometry_view` / `rebind_preferential_geometry_view` accept CUDA views of the same root, backend, device, shape, area, mask and edges (current device checked first); the sealed law arrays and the history
  arrays are SHARED (namespace/device preserved): no allocation, copy, transfer, hash, reset or rescale.
- NEW `reconcile_floating_selected(old_column, old_state, new_column, selected_mask)` (CPU and CUDA): the unchanged D1 new-capacity cascade/saturation body applied only to the cells of an explicit bool OPERATION mask that must be a
  subset of the actual active cells. CUDA reuses the existing `resident_reconcile` kernel/ABI byte for byte (the mask is passed as its active pointer; `run_reconcile(..., active=None)` keeps the old default); the columns' actual
  active maps are untouched, no fake column. Unselected water is copied bit for bit with zero flux; the output state is validated over the FULL actual domain of the new column (CUDA: the existing 32-byte state check), so an
  unselected cell invalid under the new geometry is a typed refusal. Traffic (v2, correction01): two counted 8-byte selection counts, one 32-byte FULL-OLD-domain state check of the accepted parent (the reconcile kernel's own old-capacity check runs only on the operation-selected cells, so unselected invalid old water would otherwise bypass it), one 56-byte reconcile packet, one 32-byte output state check; an invalid old parent is refused right after the first 32-byte check with nothing computed or published. The old full reconcile still reads one 56-byte packet. The receipt (`SelectedReconcileReceipt`, operation
  `selected_zero_time_reconcile/v1`, `proves_complete_geometry_refresh=False`, a minted selection token, never a mask hash) names only the selected operation; the owner that picked the mask proves completeness. The old
  `reconcile_floating` is unchanged. CPU selected uses the NumPy body for both CPU backends.
- Limits: validation of the new view re-reads the O(L N) static maps on the device (no transfer); no Pond CUDA integration, no device canonical depth, no CudaSurfaceRouter composition, no restart/daily.
