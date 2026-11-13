# Phase 2 / H1 — `FloatingPondVertical.from_held_context`

CPU public classmethod building a floating Pond vertical from a genuine water-free H0 `HeldContext` (`floating_held_context.require_held_context`: exact class + private registry membership). No `NativeFloatingSoil`, no initial W/J/I,
no bed read, no property provider. The vertical stores the context by reference (held maps and D0 are not copied) and has `_owner = None`, `_initial = None`. `held_template()` returns the context's real root facts; `_active_host`
uses the context's active mask. `initial_state`, `initial_soil_water` and `check_initial` are typed refusals ("no initial payload"): the current payload can exist only through a verified restore (`floating_checkpoint` /5-/6 load with
the caller-supplied context: prepared record, `resume_scope`, full independent validation, `adopt_resumed` last). The CPU memory model is factored into `_set_cpu_models` (shared with the owner constructor; numerics unchanged); the
construction need omits the sealed initial depth copy and is admitted from the real reader before any owned array. CUDA contexts/backends are refused before transfer; the owner constructor (CPU and CUDA) is unchanged.
Limits: the context is neither file provenance nor a bed proof; the codec does not yet store the H0 record (caller supplies the context); `NativeFloatingSoil` retirement is a later lifecycle step (not in `check_initial`).
