# CUDA checkpoint with a reconstructible soil-law context

The opt-in `maple_syrup.floating_context_checkpoint_cuda` interface saves the
current native Pond state together with the water-free soil configuration and
hydraulic-property maps needed to reconstruct its law context. A fresh process
loads the native bed and current layer water, wetting history, cumulative matrix
intake and actual soil depth. The caller supplies only a static event factory;
no original bed or initial water owner is required.

Use `save_floating_context_checkpoint_cuda`, `context_load_need_bytes_cuda` and
`load_floating_context_checkpoint_cuda`. The new explicit outer profile is
`maple-syrup-floating-context-checkpoint/2-cuda`; it wraps the existing Dense
child `/7` or native Windowed child `/8`. Other profiles retain their separate
loaders. There is no automatic migration or source rebinding.

Both save and load check the current CUDA device and admit separate device and
host working sets. `save_need_bytes_cuda` returns raw pool needs: add the device
reserve to its device value, and the declared 65,536-byte host allowance plus
the host reserve to its host value. The load estimator also exposes inclusive
totals. An explicit budget is enforced before soil-context export or array
copies. A native Windowed bed must be published first; checkpointing does not
publish a pending journal. Its immutable deep-store files remain external
references and must be retained.

Root independently qualified the bounded interface against pinned MAPLE
`9f466d2cfcd52991b24496cf67f4562d6268afa9`. The final correction passed 95 CPU
checks, four targeted CUDA save/round-trip tests, and genuine-device invalid-input
and pending-journal controls. The preceding implementation passed 159 CPU and
22 CUDA tests; its unchanged numeric models additionally passed exact-budget
save/load controls for deep Dense columns, a wider domain and native Windowed
stores with 300 and 5,000 full-depth voxels. See
[the acceptance receipt](phase2_context_checkpoint_cuda_acceptance.json) for
exact tested sources and the distinction between those source sets.

This checkpoint represents a physical event and its soil-law context. It does
not yet store the event/daily lifecycle cursor, prove a coupled simulation or
establish whole-catchment memory fit. Diagnostic every-N-day moisture frames
remain a separate output.
