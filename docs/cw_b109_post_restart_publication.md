# CW-B109: separate-process post-restart overlay publication (test-only)

`tests/cw_b109/b109_restart.py` is a reusable bounded helper (reference / pause / resume / compare / controls stages,
each its own process) on the declared B108 fixture (12×7 cells, 40 voxels, 3 classes, overlay, MAPLE default net
threshold and caps, the labelled B108 stopping criteria). LABELLED TEST CADENCE: pause at the original 1 s report
boundary with `report_every_s = 1 s`. In this fixture deep version 1 is published by t = 1 s and version 2 at t = 2 s,
so the second overlay publication occurs only after the disk resume. The held routing source has zero lag at 1 s;
the non-zero-lag continuation evidence remains the separate B108C1 12 s proof.

Checked after the resume: newer generation and version, a new version manifest and a new overlay data file that the
final version references, a reloaded v2 final version whose verified lineage reaches the paused manifest, every paused
store file and the bundle retained unchanged, increased cumulative pickup / deposition, terminal settlement and dry
reset. Reference vs resume (same backend) compares every numerical array and parsed physical metadata exactly with exact
key sets and JSON types; only `roots`, `stage_wall_s`, `store_dir` (both) and `post_restart` (resume only, checked as
above) are named qualifications. The `controls` stage applies negative controls to copies of actual outputs. No runtime
file changed; no cross-backend, real-case or large-storage claim.
