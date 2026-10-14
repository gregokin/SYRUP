# Rotated Chastre retry

Status, 2026-10-12: terrain preparation, genuine MAPLE tile compilation, initial
windowed-bed assembly, and a read-only restore through the actual MAPLE loader
have passed. The first GPU storm is queued for GPU0 to become idle. No large GPU
fit, completed storm, or wet-restart result is claimed yet.

The user requested rotation and resampling to reduce the rectangular allocation
occupied by nodata. This is a separate scientific case; the original Chastre
terrain, cases, initial stores, and failed-run evidence are preserved.

| Interior grid | Native | Rotated |
|---|---:|---:|
| Rows × columns | 1,393 × 1,604 | 886 × 1,780 |
| Allocated positions | 2,234,372 | 1,577,080 |
| Active cells | 1,140,169 | 1,140,189 |
| Cell size | 1 m | 1 m |

The rectangle contains 29.4% fewer positions. The nearest-neighbour mask changes
the sampled active area by +20 m²; rotation is not a one-to-one permutation of
the original pixels.

## Resampling and provenance

The grid axes rotate 43.1556926555° counterclockwise from the original east axis.
The recipe stores an integer axis vector, origin, original recipe and terrain
hashes, and the affine transform back to the original projected coordinates.
The output ESRI header uses a local rotated frame; its axes are not geographic
east and north.

Elevation uses ordinary bilinear interpolation where all four surrounding source
centres are valid. Otherwise it retains the nearest source elevation. Nodata is
never blended or filled. Nested linear interpolation preserves exactly constant
source patches, avoiding the one-ULP gradients found in an initial weighted-sum
implementation. Of the active cells, 1,136,609 use bilinear interpolation and
3,580 use the nearest fallback. All 431 constant-support patches remain exact.

The mask and rainfall use nearest source-cell indices. Rainfall is gathered from
the original Chastre scaling map at those physical source locations; the RFID
pattern is not stretched again to the rotated rectangle. Temporal forcing is
unchanged, including the positive-rain end at 2641 s.

The opt-in importer modification leaves the native path unchanged. Independent
checks passed 71 resampling tests and 27 existing Chastre tests, with one
device-only skip. The independent native-path comparison reproduced all 13
sidecar arrays byte for byte. A SciPy bilinear cross-check differs by at most
5.6843e-14 m, within the predeclared FP64 bound.

## Executed preparation and planned retry

The isolated source is `cw_r1_rotated_chastre_frozen` under
`outputs/conservative_water`, package digest
`008128c4d272c14bc26c54a1696e9a99eb278d566865e2c5da9df8103c755c1d`.
It uses the accepted frozen M94 MAPLE dependency. The later B99 memory
optimization is not substituted into this run.

All 28 MAPLE row tiles compiled and loaded successfully. Initial assembly created
a genuine six-class, 306-voxel bed with a three-voxel resident window. The
generation-zero manifest and all 5,346 unique chunks were hashed. The actual
CPU restore passed with unchanged initial identities and zero initial surface
water and mobile sediment.

The finite run sequence is a GPU reference to 300 s, a fresh-process restart from
its own 240 s checkpoint to 300 s, strict comparison, and then storm completion
from the candidate's own 300 s checkpoint. Each external GPU stage requires an
idle GPU0. The original absolute pool ceiling of 9,092,045,209 bytes and the
85%-of-free rule remain in force. Model stages may append deep-store generations;
the accepted initial operands and generation-zero manifest remain unchanged.

## Scientific limits

Resampling changes D4 slopes and drainage: the rotated case has 37,929 terminal
storage cells and 339 routing levels. No numerical or hydrological identity with
unrotated Chastre or its Fortran benchmark is asserted. These differences also
qualify performance comparisons. Terminal storage still has no spill,
backwater, or overtopping model; this is not a validated flood simulation.

Current evidence and process identities are recorded in
`agent_handoffs/orchestration_state.md` and the `cw_r1_*` task artifacts.
The original controller reached its one-hour GPU-occupation timeout before any
GPU model launched. The independently reviewed model-only continuation passed
re-verification of the existing case and initial store and is now waiting for
GPU0 in a new output namespace; the original evidence is preserved. Full water-component
acceptance and production adoption remain open.
