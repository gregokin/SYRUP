# Native daily recharge example (Phase 1.5, source N15)

A **Python API** demonstration, not a SYRUP command and not a weather/CSV or scheduler interface. One script, `daily_recharge_example.py`, calls the real factories
(`native_case_profile.verify_source_profile` / `admit_package_case`, `native_dense_profile.prepare_native_dense_case`, `layered_case.load_layered_config` /
`prepare_layered_case`) and the real daily API (`daily_water`, `water_lifecycle`, V3 checkpoint codecs). It reuses the public case and `layered_water.yaml` of
`../native_leaf_calendar_cli/` by **copying** them into your work directory; the originals are never edited.

**Source selection (explicit).** The SYRUP source must be the N15 package (SHA-256 `82e9c0111b9d883f857fd2f28a21f1d8bd8f13e22d473796e736ca4204f30107`, 114 files; the script
passes it as `expected_syrup_digest`) and MAPLE the materialized historical 9f bundle you point at. This example never touches the C1 recipes, locks or receipts
(SYRUP `e40a1b8`, MASY `a704002`, LEAF `52f39fe`), which keep their own identities.

## Run (CPU, existing MAPLE environment)

```
export CUDA_VISIBLE_DEVICES="" MAPLE_SYRUP_EXPECTED_MAPLE_ROOT=BUNDLE/source
export PYTHONPATH=BUNDLE/editable:BUNDLE/source/src:N15_SYRUP/src
MAPLE/.venv/bin/python -B daily_recharge_example.py --bundle BUNDLE --scenario recharge --workdir /tmp/daily_demo_NEW1
MAPLE/.venv/bin/python -B daily_recharge_example.py --bundle BUNDLE --scenario fallback --backend numba --checkpoint --later-storm --workdir /tmp/daily_fb_NEW2
```

| Scenario | Shows |
|---|---|
| `recharge` | 0 < P ≤ T: top uniform pulse; below field capacity nothing moves; the day's E and diffusion run once; P=0 would still run the same one day |
| `open_drainage` | open bottom above field capacity: gravity moves water down and the bottom drains it (reported) |
| `closed_retention` | closed bottom: capacity-limited retention up to saturation, no drainage |
| `fallback` | a real whole-P native full storm after a capacity failure of the recharge trial (owned flat closed copy of the case); local forcing [0, D] with the physical origin; the storm completes before the one day |

**Safety.** `--workdir` (and `--checkpoint-dir`) must be NEW paths: the script validates them before any import or write, refuses an existing path, and never deletes, replaces or rewrites existing data (there is no force option; the native V3 writer also refuses to overwrite). To rerun, use a new directory. `python check_preservation.py` (3 refusals, sentinel bytes identical) proves the refusals on temporary sentinel files (bytes, names and counts unchanged). The checkpoint save/load uses the operating system's real available-memory reading (`subsurface.default_host_available_bytes`) with explicit caller quotas (`--ck-*-bytes`, illustrative 1 GiB); the model never raises a quota.

Options: `--p-mm --i-mm-h --t-mm` (illustrative test values per scenario; **T has no scientific default**), `--pet-m-s` (illustrative explicit PET in m/s, not calibrated
weather), `--t-request-m` (illustrative zero-time plant withdrawal; the stored results use 0.000001 m), `--checkpoint` (V3 save and load of the closed boundary), `--later-storm` (a later full storm at the
new nonzero clock), `--backend numpy|numba|cuda`. Output: one compact JSON record on stdout.

**Which audit is which.** `first_day_audit_before_T` is the native audit of the first accepted occurrence, taken *before* the T withdrawal, the checkpoint and any later storm. `final_state` names the CURRENT accepted state (the closed boundary after T and the V3 save/load, or, with `--later-storm`, the genuinely completed later storm, whose daily occurrence is still *pending*: no further day, E/T or clock change is applied) and `closure` is the native `audit_lifecycle` of that final state (cumulative lifetime ledgers: first-day rain + later-storm rain once, the accepted T once, residual against the floating-point bound). `stages` keeps the step-by-step records. `python check_results.py results/*.json` verifies these identities on stored summaries (no model run).

`fallback` uses PET 1e-7 m/s by default: the saturated closed flat column keeps about 1 mm of pond after the storm, which the single daily router step must first
evaporate, otherwise it is **refused** with the typed `router_step_rejected` (gravity-wave CFL 28720.67 > 0.5, state unchanged). Try `--pet-m-s 1e-8` to see that
refusal; nothing is substepped or loosened. A storm counts as complete only when it genuinely completes (the script refuses to treat a step/time limit as completion).

`results/` holds the stored CPU outputs of the author's corrected runs with the real memory reader (`*.numpy.json`, `*.numba.json`, stderr, the CFL refusal log and the refused-rerun message). They are examples of use, not
qualification evidence; the Root controls are in `docs/subsurface_hydrology/phase15_qualification.md`.

**CUDA.** `--backend cuda --bundle CUDA_BUNDLE --device 0` selects the genuine CUDA factories and the CUDA V3 codec. The author did not run this branch; Root ran it on the assigned device (four scenarios in the earlier snapshot and the recharge scenario on the final snapshot) and recorded the results in `docs/subsurface_hydrology/phase15_acceptance.json`. Run it only under your own device guard; it refuses rather than falling back to the CPU when a precondition is missing.
(it needs the CUDA bundle and a free GPU); run it only under your own device guard, and expect it to refuse rather than fall back when a precondition is missing.

Limits: 4×6 fixed-geometry Dense fixtures; retained-wet/CFL and repeated-storm history limits; recharge is a user-chosen approximation; no splash, remapping, LEAF
feedback, windowed bed, field or performance claim.
