"""LITERAL pins of the strict portable Phase 2.5 source-identity profile (`p25_source_profile.py`). Uniquely named; no logic, no I/O.

Immutable inputs (filled, literal): the base commit 02388ad listing, the 953 WH1 input proof, the historical component package digests (the actual frozen Root-accepted bounded component sources, each used ONLY by that
component's own historical assertions).

FINAL LOCK (Root-reviewed source identity): `REQUIRES_ROOT_FINAL_LOCK` is a hard-fail sentinel. Root alone fills the four literals below -- the sha256 of `tests/fixtures/phase25/final_lock.json`, the tree digest, file count and total bytes of
the final source -- after ALL production writers close, together with the final changed/added NAME sets (the current 180-module candidate's sets are recorded as the template). The lock states
`SOURCE_LOCKED_FOR_QUALIFICATION` and `whole_phase_accepted: false`: it identifies the exact reviewed source for qualification, it does not accept the model. While any sentinel remains every final-branch guard
FAILS (it never skips and never admits the current mutable tree). `prepare_lock.py` only PREPARES proposed bytes for review."""
from __future__ import annotations

REQUIRES_ROOT_FINAL_LOCK = "REQUIRES_ROOT_FINAL_LOCK"

# ---- immutable base: SYRUP 02388ad (the Phase 2 public package), 169 files
BASE_GIT = "02388adb70b7f1f22b33b6b3d5f1249e223842e5"
BASE_FILE_COUNT = 169
BASE_MANIFEST_SHA256 = "3eef93edb76d584fcd9bf31e4546dd8f11e084dfdc4bdae4a1d60c2ea88bc553"   # tests/fixtures/phase25/base169_manifest.json (byte copy of syrup_preedit_manifest169.json)
BASE_TREE_DIGEST_SHA256 = "3917dba4f59f662650bdf6fb08c10e4fece9d3de601c4b42ad41c9a796382c82"

# ---- the 953 input: SYRUP main 953e6fd adds exactly one module (WH1) to the 169
WH1_GIT = "953e6fdf85b73e3fdf5a1ac7b0d97dea4df7dab1"
WH1_PATH = "floating_wind_handoff.py"
WH1_INPUT_SHA256 = "a289e35b329bc33ec918768133ada9aef1d37089b110d4085fb8c97c4c09156b"
WH1_PROOF_SHA256 = "f54497ed35104520fc0b52bc953489edf87a4ea775638543f639745d00c1a01b"        # tests/fixtures/phase25/wh1_953_input_proof.json

# ---- historical package digests (source_tree_digest over src/maple_syrup of each frozen, Root-accepted bounded component)
HISTORICAL = {
    "wind_compat_c0": "76e3cb20236f03f2812a755131bf4c452077bb9eb574674ca85c7a53ef512bae",  # bounded Root76CPU wind guard composition
    "base169_02388ad": BASE_TREE_DIGEST_SHA256,                                                                # 169 files
    "cache_c3": "692482499cb8d405bb1508fecea5d6570c6295a9dc842366566ef5a03001df53",                           # 171 files (S0 binding + cached inputs)
    "lifecycle_c4": "809cf3e98c553b0f7835ace1130dbed9e489e7df6c0debafda0832ab8c04151b",                       # 173 files
    "frames_c2": "d5f285c8018a1c2a1fb1c945cd7a9bd525af79909deb796b628e366e9f1e9f29",                         # 175 files
    "held_context_c3": "a38d4d929fb9b5b237dcdce010c72f2c2700573fa2c497473add034b12f59c53",                   # 175 files
    "checkpoint_c0": "2ebf0608983124daf93c18f84fdfa325cc4c140571399647d4aac15217ebf13e",                     # 179 files
}

# ---- FINAL LOCK: Root fills after every production writer closes (sentinel = hard failure)
LOCK_SHA256 = '2bd3c422b56724a7bf78689038541bee4b21612227c41fc73b9783a0a48a092c'
TREE_DIGEST_SHA256 = '37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0'
FILE_COUNT = 180
TOTAL_BYTES = 7261186

# the final changed (vs base169) and added names: Root-reviewed source candidate for qualification (11 changed, 10 added + the WH1 module); Root revises them WITH the final lock
CHANGED = frozenset({"floating_daily.py", "floating_daily_output.py", "floating_event_daily_bridge.py", "floating_event_lifecycle.py", "floating_event_lifecycle_cuda.py", "floating_event_refresh.py",
                     "floating_event_refresh_cuda.py", "floating_held_context.py", "floating_matrix_infiltration.py", "native_floating_soil.py", "native_floating_soil_cuda.py"})
ADDED = frozenset({"floating_daily_output_vegetation.py", "floating_held_context_vegetation.py", "floating_held_context_vegetation_cuda.py", "floating_lifecycle_checkpoint_vegetation.py",
                   "floating_lifecycle_checkpoint_vegetation_cuda.py", "floating_moisture_frame_vegetation.py", "vegetation_lifecycle.py", "vegetation_lifecycle_cuda.py", "water_vegetation_binding.py",
                   "water_vegetation_held.py"})
