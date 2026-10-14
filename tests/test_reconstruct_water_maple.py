"""CW-E9 / CW-E11 / CW-E16 / CW-E19 / CW-E22 / CW-E26 / CW-E29 / CW-E33 / CW-E36 / CW-E39 / CW-E44 / CW-E46: tests
of tools/reconstruct_water_maple.py with the pinned 24c6 runtime recipe (CPU,
standard library + git; no install, network or GPU).

The full reconstruction needs a MAPLE Git repository that contains the pinned commit. It is located through
`MAPLE_WATER_RECIPE_REPO` (default /home/okin/MAPLE) and skipped when absent. The optional byte comparison against
an existing 24c6 runtime uses `MAPLE_WATER_RECIPE_REFERENCE` (a directory holding src/maple + pyproject.toml).
Negative controls tamper with private COPIES of the recipe under tmp_path only. The pinned recipe, the repository
and every other path are read-only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[1]
TOOL_PATH = TREE / "tools" / "reconstruct_water_maple.py"
_spec = importlib.util.spec_from_file_location("reconstruct_water_maple", TOOL_PATH)
tool = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tool
_spec.loader.exec_module(tool)

REPO = Path(os.environ.get("MAPLE_WATER_RECIPE_REPO", "/home/okin/MAPLE"))
REFERENCE = os.environ.get("MAPLE_WATER_RECIPE_REFERENCE")
MANIFEST = json.loads((tool.RECIPE_DIR / tool.MANIFEST_NAME).read_bytes())
REV = MANIFEST["upstream"]["revision"]


def _has_revision() -> bool:
    if not (REPO / ".git").exists() and not (REPO / "objects").exists():
        return False
    proc = subprocess.run(["git", "-C", str(REPO), "cat-file", "-t", REV], capture_output=True)
    return proc.returncode == 0 and proc.stdout.strip() == b"commit"


needs_repo = pytest.mark.skipif(not _has_revision(), reason="MAPLE repository with the pinned commit unavailable")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def leftovers(parent: Path):
    return sorted(p.name for p in parent.iterdir() if ".e9-staging-" in p.name)


def repo_state():
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True).stdout
    status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain=v1"], capture_output=True).stdout
    return head, sha(status)


def recipe_copy(tmp_path: Path, *, edit_manifest=None, edit_patch=None, repin_patch=True):
    """A private copy of the recipe, optionally edited; returns (dir, manifest_sha256 to pin)."""
    target = tmp_path / "recipe"
    shutil.copytree(tool.RECIPE_DIR, target)
    manifest = json.loads((target / tool.MANIFEST_NAME).read_bytes())
    if edit_patch is not None:
        patch_path = target / manifest["patch"]["file"]
        data = edit_patch(patch_path.read_bytes())
        patch_path.write_bytes(data)
        if repin_patch:
            manifest["patch"]["sha256"] = sha(data)
            manifest["patch"]["size"] = len(data)
    if edit_manifest is not None:
        edit_manifest(manifest)
    raw = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    (target / tool.MANIFEST_NAME).write_bytes(raw)
    return target, sha(raw)


# --- the pinned recipe itself -----------------------------------------------------------------------------------------
def test_pinned_recipe_is_self_consistent():
    raw = (tool.RECIPE_DIR / tool.MANIFEST_NAME).read_bytes()
    assert sha(raw) == tool.EXPECTED_MANIFEST_SHA256
    m, _ = tool.load_manifest(tool.RECIPE_DIR, tool.EXPECTED_MANIFEST_SHA256)
    patch = (tool.RECIPE_DIR / m["patch"]["file"]).read_bytes()
    assert sha(patch) == m["patch"]["sha256"]
    assert tool.parse_patch(patch, m) and len(m["patch"]["changed_files"]) == 35 and len(m["patch"]["new_files"]) == 16
    assert len(m["patch"]["modified_files"]) == 19 and m["patch"]["deleted_files"] == []
    assert m["pre_patch"]["file_count"] == 325 and m["post_patch"]["file_count"] == 341
    assert m["post_patch"]["package_source_digest"]["file_count"] == 340
    # CW-E46: the explicit expected target is the delivered 24c6 runtime (= M94 + CW-B106C1 + CW-B107); the M94/E44,
    # M93/E39, M92/E36, M91/E33, M88/E29, M85/E26, M82/E22, M79/E19, M75/E16, M67/E11 and M44/E6 records stay
    # historical (never relabelled as 24c6)
    prov = m["provenance"]
    target = prov["target_runtime"]
    assert target["name"] == "24c6" and m["post_patch"]["package_source_digest"]["digest_sha256"] == \
        target["package_source_digest"] == D24C_DIGEST
    assert m["post_patch"]["files"]["pyproject.toml"] == target["pyproject_sha256"] == PYPROJECT_SHA
    assert target["changed_from_M94"] == D24C_CHANGES
    assert target["changed_from_M94_stages"] == {"water_B106C1": sorted(B106C1_MODIFIED),
                                                 "water_B107": sorted(B107_MODIFIED)}
    assert target["intermediate_B106C1_package_source_digest"] == B106C1_DIGEST
    for f in D24C_CHANGES:
        assert m["post_patch"]["files"][f] == target["changed_from_M94_sha256"][f]["24c6"]
    # the runtime recipe is DISTINCT from the full-project delivery (never its project digest or bundle id)
    full = target["full_project_artifact_distinct"]
    assert full["bundle_id"] == FULL_PROJECT_BUNDLE_ID and full["project_source_digest"] == FULL_PROJECT_DIGEST
    assert full["project_file_count"] == 1278 and m["post_patch"]["file_count"] == 341
    assert FULL_PROJECT_DIGEST not in (target["package_source_digest"], m["post_patch"]["package_source_digest"]["digest_sha256"])
    assert "BOUNDED" in target["status"] and "OPEN" in target["status"] and "ACTIVE, not pinned" in target["status"]
    assert "NOT fresh-clone trust roots" in prov["trust_note"]
    hist94 = prov["historical_M94_recipe"]
    assert hist94["package_source_digest"] == M94_DIGEST != target["package_source_digest"]
    assert hist94["e44_manifest_sha256"] == E44_MANIFEST_SHA and hist94["e44_patch_sha256"] == E44_PATCH_SHA
    assert hist94["e44_bundle_id"] == E44_BUNDLE_ID and hist94["e44_target_runtime"]["name"] == "M94"
    assert hist94["e44_target_runtime"]["changed_from_M93"] == M94_CHANGES
    assert prov["historical_M44_recipe"]["e6_package_source_digest"] != target["package_source_digest"]
    assert prov["historical_M67_recipe"]["package_source_digest"] == M67_DIGEST != target["package_source_digest"]
    assert prov["historical_M75_recipe"]["package_source_digest"] == M75_DIGEST != target["package_source_digest"]
    assert prov["historical_M79_recipe"]["package_source_digest"] == M79_DIGEST != target["package_source_digest"]
    assert prov["historical_M82_recipe"]["package_source_digest"] == M82_DIGEST != target["package_source_digest"]
    assert prov["historical_M82_recipe"]["e22_target_runtime"]["name"] == "M82"
    assert prov["historical_M85_recipe"]["package_source_digest"] == M85_DIGEST != target["package_source_digest"]
    assert prov["historical_M85_recipe"]["e26_target_runtime"]["name"] == "M85"
    assert "Resolved by CW-B88 in M88" in prov["historical_M85_recipe"]["known_regression_resolved_in_M88"]
    assert prov["historical_M88_recipe"]["package_source_digest"] == M88_DIGEST != target["package_source_digest"]
    assert prov["historical_M88_recipe"]["e29_target_runtime"]["name"] == "M88"
    assert prov["historical_M91_recipe"]["package_source_digest"] == M91_DIGEST != target["package_source_digest"]
    assert prov["historical_M91_recipe"]["e33_target_runtime"]["name"] == "M91"
    assert "cw_h3_root_gpu_refusal" in prov["historical_M91_recipe"]["known_large_case_refusal"]
    assert prov["historical_M92_recipe"]["package_source_digest"] == M92_DIGEST != target["package_source_digest"]
    assert prov["historical_M92_recipe"]["e36_target_runtime"]["name"] == "M92"
    refusals = " ".join(prov["historical_M92_recipe"]["known_large_case_refusals"])
    assert "cw_h4_root_gpu_refusal" in refusals and "cw_h5_root_gpu_refusal" in refusals
    assert prov["historical_M93_recipe"]["package_source_digest"] == M93_DIGEST != target["package_source_digest"]
    assert prov["historical_M93_recipe"]["e39_target_runtime"]["name"] == "M93"
    assert "cw_h7_root_gpu_refusal" in prov["historical_M93_recipe"]["known_large_case_refusal"]
    assert prov["stage_files"]["water_B106C1"] == sorted(B106C1_MODIFIED)
    assert prov["stage_files"]["water_B107"] == sorted(B107_MODIFIED)
    assert prov["stage_files"]["water_M94"] == hist94["e44_target_runtime"]["changed_from_M93"] == M94_CHANGES
    assert prov["stage_files"]["water_M93"] == M93_CHANGES  # historical stages kept exactly
    assert prov["stage_files"]["water_M92"] == M92_CHANGES
    assert prov["stage_files"]["water_M91"] == M91_CHANGES
    assert prov["stage_files"]["water_M88"] == M88_CHANGES
    assert prov["stage_files"]["water_M85"] == M85_CHANGES
    assert prov["stage_files"]["water_M82"] == M82_CHANGES
    assert sorted(f for f, stages in prov["changed_file_stages"].items() if "water_B106C1" in stages or
                  "water_B107" in stages) == target["changed_from_M94"]
    assert sorted(f for f, stages in prov["changed_file_stages"].items() if "water_M94" in stages) == M94_CHANGES
    assert sorted(f for f, stages in prov["changed_file_stages"].items() if "water_M93" in stages) == M93_CHANGES
    # CW-E33 memory record (M91 closure helper) is kept exactly as history
    hist91 = prov["historical_M91_recipe"]["e33_target_runtime"]["memory"]
    assert hist91["true_whole_helper_peak"] == {"M88": "4 C + 2 mask", "B90": "4 C + mask", "M91": "3 C + 2 mask"}
    assert hist91["statement_boundary_maximum"] == {"M88": "4 C + mask", "B90": "4 C + mask", "M91": "3 C + mask"}
    # CW-E36 memory record (M92 owned-ledger helper) is kept exactly as history
    hist92 = prov["historical_M92_recipe"]["e36_target_runtime"]["memory"]
    assert hist92["actual_device_whole_helper_peak_bytes"] == {"M91": 386560, "M92": 141312}
    assert hist92["actual_device_saving_bytes"].startswith("245248 = 2 C - 512 B")
    # CW-E39 memory record (M93 proportional-vector helper) is kept exactly as history
    hist93 = prov["historical_M93_recipe"]["e39_target_runtime"]["memory"]
    assert hist93["actual_device_whole_helper_peak_bytes"] == {"M92": 1141760, "M93": 568320}
    assert hist93["retained_bytes"] == 245760
    # CW-E44 actual-device whole erosion measurement (M94; bounded) is kept exactly as history
    hist94_mem = hist94["e44_target_runtime"]["memory"]
    assert hist94_mem["actual_device_whole_erosion_peak_bytes"] == {"M93": 6124032, "M94": 5847552}
    assert hist94_mem["retained_bytes"] == 1397760
    assert hist94_mem["whole_paired_peak_bytes_both"] == 11803136
    assert "UNCHANGED" in hist94_mem["not_claimed"]
    # the M94 broad-suite failures stay recorded as history, never a broad PASS
    hist94_status = hist94["e44_target_runtime"]["status"]
    assert "NOT PASS" in hist94_status and "38 / candidate 40" in hist94_status and "65 / 67" in hist94_status
    # every changed file is attributed to at least one recorded provenance stage; unchanged upstream files stay Git's
    assert all(m["provenance"]["changed_file_stages"][f] for f in m["patch"]["changed_files"])
    unchanged = [f for f in m["pre_patch"]["files"] if f not in m["patch"]["changed_files"]]
    assert all(m["pre_patch"]["files"][f]["sha256"] == m["post_patch"]["files"][f] for f in unchanged)


# CW-E33: the reviewed M88 -> M91 change (accepted CW-B90 + CW-B91, frozen a9de77ce) -- exactly ONE modified file,
# none added; the E29 M85 -> M88 stage is kept as history
M67_DIGEST = "b08b24d192330201455dc126560a51c0cd277eb7277ebd23dd853bb9caee9caf"
M75_DIGEST = "c39e8b3b25dc11e499c2eab184bb3cb54a3adda9528e8f57c775b6c933a64fbf"
M79_DIGEST = "7a4b65a4bcffe7401b6d83b758ad1d8be579fd39ce8ad8d07b9cfed9d7ae3107"
M82_DIGEST = "1667efe215d31dfd6baebf3583e3ab4cdc7648f6334d4ac2a59a15530d82ad5b"
M85_DIGEST = "17044c8a4617ccf09a0c2ae9088497c252917ecf1def644924b2829b9be6b29c"
M88_DIGEST = "89a2277243ce6fd9b92db13406b9403fbd7f9781a16fba8998dac742da4425fe"
M91_DIGEST = "a9de77ce4fdeb779e3f0bfecf92367c8e526072b8366093d39706ed23e2e4ff5"
M92_DIGEST = "e4398b27ca2a26e86a714448944fd7d6e3c9ad0eeb8761ae26f839993222ae88"
M93_DIGEST = "3dd4108cabd1994c61bdbeda4570d0019cc3a687e166e392fdbc6f7b9d436ab1"
M94_DIGEST = "eb14da66e0ffee1e11cd37cc98d3379b01b9e8f54e562ad0d12fc22afb11abd6"
# historical E22 M79 -> M82 stage (two files) and E26 M82 -> M85 stage (one file), kept exactly
M82_CHANGES = ["src/maple/surface/availability/exchange.py", "src/maple/water/windowed_step.py"]
M85_CHANGES = ["src/maple/surface/availability/exchange.py"]
M85_HIST_SHA256 = {"src/maple/surface/availability/exchange.py": {
    "M82": "06dd633ee9f48bec16daa3638c873e5f6e21e948db0553f1a1715ab4c15b7481",
    "M85": "df292427b86163dfc50ba4fd005babee6f410fecb670b08d8af363df1de3d804"}}
M88_MODIFIED = {
    "src/maple/surface/availability/exchange.py": (
        "df292427b86163dfc50ba4fd005babee6f410fecb670b08d8af363df1de3d804",
        "7df786797647176dc8d496e9d7a54fe73f576fda9a9dce9f4e6da1e127de21ee"),
}
M88_CHANGES = sorted(M88_MODIFIED)
M88_HIST_SHA256 = {k: {"M85": old, "M88": new} for k, (old, new) in M88_MODIFIED.items()}
M91_MODIFIED = {
    "src/maple/surface/availability/exchange.py": (
        "7df786797647176dc8d496e9d7a54fe73f576fda9a9dce9f4e6da1e127de21ee",
        "fd970c6ae20ba883f7d7d72f3df573793b21444eef4f57040cc42ae8b70b35f2"),
}
M91_CHANGES = sorted(M91_MODIFIED)
M91_HIST_SHA256 = {k: {"M88": old, "M91": new} for k, (old, new) in M91_MODIFIED.items()}
M92_MODIFIED = {
    "src/maple/coupling/sediment_ledger/accumulate.py": (
        "818eef1957c00e08d0b9c9032a100021ea29ef3ffd50038dac9e292e19f9b9ac",
        "610a66f7231b9dfb8fb34ab0f323c57d3bcac69003266bf85d856d69ce984e4b"),
}
M92_CHANGES = sorted(M92_MODIFIED)
M92_HIST_SHA256 = {k: {"M91": old, "M92": new} for k, (old, new) in M92_MODIFIED.items()}
# CW-E39: the reviewed M92 -> M93 change (accepted CW-B96, frozen 3dd4108c) -- exactly TWO modified files, none added
M93_MODIFIED = {
    "src/maple/surface/active_layer/windowed_exchange.py": (
        "c92ba7a061586432d94d5f6663e41e1a70874ad43f658dc4e88158e93cc2301a",
        "9b98413de79cde29f964ee0ce0fc58109b4775b8f572609f37ae4f2283fc7457"),
    "src/maple/surface/voxels/_numerics.py": (
        "42d1ffd6f1d06ea4c388ebb0551f678b435ff427dc56b39e914536b86f53f7a2",
        "7cd74b16eeaf6f7c21ec3c2e8579a496f0bb8ea1b2295ad1ebc7ed9e6b4451e1"),
}
M93_CHANGES = sorted(M93_MODIFIED)
M93_HIST_SHA256 = {k: {"M92": old, "M93": new} for k, (old, new) in M93_MODIFIED.items()}
# the actual relation of the changed files to the upstream revision (derived from the pinned recipe, asserted exactly):
# active_layer/windowed_exchange.py is a NEW file of the water stages (M29 / M79); voxels/_numerics.py is an upstream
# file that was UNMODIFIED through M92 and is modified for the first time in M93 (a new patch section)
M93_UPSTREAM_KIND = {"src/maple/surface/active_layer/windowed_exchange.py": "new",
                     "src/maple/surface/voxels/_numerics.py": "modified"}
M93_PRIOR_STAGES = {"src/maple/surface/active_layer/windowed_exchange.py": ["water_M29", "water_M79"],
                    "src/maple/surface/voxels/_numerics.py": []}
# CW-E44: the reviewed M93 -> M94 change (accepted CW-B97, frozen eb14da66) -- exactly ONE modified file, none added;
# it is a NEW file of the patch (water stages M29 / M79, then M93), so no patch section is added
M94_MODIFIED = {
    "src/maple/surface/active_layer/windowed_exchange.py": (
        "9b98413de79cde29f964ee0ce0fc58109b4775b8f572609f37ae4f2283fc7457",
        "89cd8ce3d155af50bc0ef6ba0280728d3dc8bacac8a537756f04983185a0d5f3"),
}
M94_CHANGES = sorted(M94_MODIFIED)
M94_UPSTREAM_KIND = {"src/maple/surface/active_layer/windowed_exchange.py": "new"}
M94_PRIOR_STAGES = {"src/maple/surface/active_layer/windowed_exchange.py": ["water_M29", "water_M79", "water_M93"]}
# CW-E46: the reviewed M94 -> B106C1 -> B107 (= delivered 24c6) changes (accepted CW-B106C1 frozen 4585f8de and CW-B107
# frozen 24c6e147) -- exactly SIX existing files modified (windowed_pending.py in both stages), none added or removed;
# all six are NEW files of the patch (water stages), so no patch section is added
D24C_DIGEST = "24c6e1478ed732df22f448eddf8cb16c3b72f2406dc9b37a9414cee9a2d1a9ce"
B106C1_DIGEST = "4585f8de7574dc596f10c49c41aab1c43e3dee847797100d3e28a65f0576f076"
PYPROJECT_SHA = "e6c7da22ee9f4bdd7876747fdfcbf694314cafd9032e3700301522b52b3fe3ae"
FULL_PROJECT_BUNDLE_ID = "b8fa43a3b052f4a67cff003257698976f634b12e3d44431b5074fa32ef058636"
FULL_PROJECT_DIGEST = "f8aaff8df8642d8e8b45a9c3e559bb19917deaaececa7ab7c2570f499c92efb7"
E44_MANIFEST_SHA = "27bd1900dda7d59ed93b4b36345a30daeb45b3e62dc464618b17f60d75be7cfc"
E44_PATCH_SHA = "540dfb7d2b06e54b2844046c0f0c9d60197935ad1e0289978f0757fa36fb3544"
E44_BUNDLE_ID = "a8904a9911200b530c3f8068bbaa537cc05c22f5e2f7fb6f1ee78b7bca535740"
B106C1_MODIFIED = {
    "src/maple/surface/voxels/deep_store.py": (
        "e10271b90849f1c5ab9079facfe5f5811e80d000010d88ea011a35573c605cab",
        "c0855474cc79b4a47322b9188196749af2e2c9925f82b3db2d29a916be556677"),
    "src/maple/surface/voxels/windowed_pending.py": (
        "b86bd89d0a3765918c662faa2c70e5f44ea835d09823a0628f338cc78d63c740",
        "f8fdaaa5e3d4d45de1c3b23b489eb9e44996b87ebd044976082f494423a3f1fd"),
    "src/maple/surface/voxels/windowed_transfer.py": (
        "42cb26a020fd48774c371818cd73c481d4514346e3f7ddb03510800030245426",
        "062d4360c20d341529462566d5dc76510d1ee0a2c77a5d0886ed6b09384cb6ea"),
}
B107_MODIFIED = {
    "src/maple/surface/topographic_commit/windowed_commit.py": (
        "09a4c87e39d4e3e61784851203bf77bc3f96f1001b9bbd22ece3f1a1b59e3714",
        "2f481b9a0c8eb3ac77feffc2deabaed994e09460ba8b40cc727f3bb5cf718ff5"),
    "src/maple/surface/voxels/windowed_pending.py": (
        "f8fdaaa5e3d4d45de1c3b23b489eb9e44996b87ebd044976082f494423a3f1fd",
        "b10e4ec955027478184b3405e9080ab7de068a898f8adeeaf9f59845af5e1273"),
    "src/maple/surface/voxels/windowed_snapshot.py": (
        "c6de6c103751556abc6637540f88cde0122953ad21a24a10c12dc7819634edba",
        "f37de83b1c713904007bd8ed8e65d9141b243ac23e784967fd78484f3d952db7"),
    "src/maple/water/windowed_step.py": (
        "d4b4963ba5523adf35455a590800fe21ddb8989856980efeaff74bec3a92783e",
        "73baeff15148404cbe017cf8682424e1afdbeb9ffa74739234c23fbd7b7f6116"),
}
D24C_CHANGES = sorted(set(B106C1_MODIFIED) | set(B107_MODIFIED))
B106C1_B107_TESTS = (  # MAPLE tests of the full-project delivery, never runtime
    "tests/conservative_window/test_cw_b106_overlay_store.py", "tests/conservative_window/test_cw_b106_overlay_windowed.py",
    "tests/conservative_window/test_cw_b106c1_overlay_gpu.py", "tests/conservative_window/test_cw_b106c1_store_corrections.py",
    "tests/conservative_window/test_cw_b107_publish_gpu.py", "tests/conservative_window/test_cw_b107_publish_options.py")
B97_TESTS = ("tests/cw_b97/conftest.py", "tests/cw_b97/test_cw_b97_erosion_scratch.py")  # tests, never runtime
B92_TEST = "tests/conservative_window/test_cw_b92_ledger_increment_scratch.py"  # a test, never runtime
B96_TESTS = ("tests/cw_b96/conftest.py", "tests/cw_b96/test_cw_b96_deposition_vector.py")  # tests, never runtime
B90_B91_TESTS = ("tests/conservative_window/test_cw_b90_closure_scratch.py",
                 "tests/conservative_window/test_cw_b91_closure_output_reuse.py")  # tests, never runtime
B85_TEST = "tests/conservative_window/test_cw_b85_fraction_scratch.py"  # updated by B88; a test, never runtime
B85_EVIDENCE = {
    "e22_M82_recipe_manifest": "cw_e26_logs/baseline_archive/dependencies/maple_water/manifest.json",
    "e22_M82_runtime_patch": "cw_e26_logs/baseline_archive/dependencies/maple_water/maple_74dd3e79_to_M82_runtime.patch",
    "e22_independent_acceptance": "cw_e22_independent_acceptance.json",
    "b85_runtime_patch": "cw_b85_runtime.patch",
    "b85_author_report_corrected": "cw_b85_report.md",
    "b85_correction1_prompt": "cw_b85_correction1_prompt.md",
    "b85_root_scope_review": "cw_b85_root_scope_review.json",
    "b85_root_freeze": "cw_b85_root_freeze.json",
    "b85_root_freeze_script": "freeze_cw_b85_root.py",
    "b85_independent_acceptance": "cw_b85_independent_acceptance.json",
    "b85_root_cpu_log": "cw_b85_root_cpu_tests.log",
    "b85_root_gpu_log": "cw_b85_root_gpu_tests.log",
    "b85_root_frozen_multistep": "cw_b85_root_frozen_multistep.json",
    "b85_root_frozen_multistep_log": "cw_b85_root_frozen_multistep.log",
    "b85_root_frozen_multistep_check": "cw_b85_root_frozen_multistep_check.py",
    "b85_root_multistep_derivation": "cw_b85_root_multistep_derivation.json",
}
# CW-E29: the 22 E26 / B88 records appended to the 81 E26 records (stable completed records only; ACTIVE integrated /
# multi-step / INITIAL / status logs are deliberately not pinned)
B88_EVIDENCE = {
    "e26_M85_recipe_manifest": "cw_e29_logs/baseline_archive/dependencies/maple_water/manifest.json",
    "e26_M85_runtime_patch": "cw_e29_logs/baseline_archive/dependencies/maple_water/maple_74dd3e79_to_M85_runtime.patch",
    "e26_independent_acceptance": "cw_e26_independent_acceptance.json",
    "e26_review_zero_d_probe": "cw_e26_work/b85_review_zero_d_probe.py",
    "e26_review_zero_d_probe_log": "cw_e26_logs/b85_review_zero_d_probe_script.log",
    "b88_prompt": "cw_b88_ndarray_guard_prompt.md",
    "b88_author_report": "cw_b88_report.md",
    "b88_runtime_vs_m85_patch": "cw_b88_runtime_vs_m85.patch",
    "b88_runtime_vs_m82_patch": "cw_b88_runtime_vs_m82.patch",
    "b88_test_vs_m85_diff": "cw_b88_test_vs_m85.diff",
    "b88_author_scalar_kinds_probe": "cw_b88_work/scalar_kinds_probe.py",
    "b88_author_scalar_kinds_before_log": "cw_b88_logs/scalar_kinds_probe_before.log",
    "b88_author_scalar_kinds_after_log": "cw_b88_logs/scalar_kinds_probe_after.log",
    "b88_author_e26_probe_reproduction_log": "cw_b88_logs/e26_probe_reproduction_frozen.log",
    "b88_author_new_tests_on_m85_negative_control_log": "cw_b88_logs/negative_control_new_tests_on_frozen_m85_runtime.log",
    "b88_root_before_probe_log": "cw_b88_root_before_probe.log",
    "b88_root_scope_review": "cw_b88_root_scope_review.json",
    "b88_root_freeze": "cw_b88_root_freeze.json",
    "b88_root_freeze_script": "freeze_cw_b88_root.py",
    "b88_independent_acceptance": "cw_b88_independent_acceptance.json",
    "b88_root_cpu_wind_log": "cw_b88_root_cpu_wind_tests.log",
    "b88_root_gpu_log": "cw_b88_root_gpu_tests.log",
}
# CW-E33: the 39 E29 / B90 / B91 records appended to the 103 E29 records (stable completed records only; ACTIVE
# integrated device-view / paired / INITIAL / queue / status records are deliberately not pinned)
B91_EVIDENCE = {
    "e29_M88_recipe_manifest": "cw_e33_logs/baseline_archive/dependencies/maple_water/manifest.json",
    "e29_M88_runtime_patch": "cw_e33_logs/baseline_archive/dependencies/maple_water/maple_74dd3e79_to_M88_runtime.patch",
    "e29_independent_acceptance": "cw_e29_independent_acceptance.json",
    "b90_prompt": "cw_b90_closure_scratch_prompt.md",
    "b90_author_report": "cw_b90_report.md",
    "b90_runtime_vs_m88_patch": "cw_b90_runtime_vs_m88.patch",
    "b90_author_multistep": "cw_b90_multistep_check.json",
    "b90_author_exit": "cw_b90_author_exit.json",
    "b90_root_cpu_log": "cw_b90_root_cpu_tests.log",
    "b90_root_gpu_log": "cw_b90_root_gpu_tests.log",
    "b90_root_gpu_disposition": "cw_b90_root_gpu_disposition.json",
    "b91_prompt": "cw_b91_closure_output_reuse_prompt.md",
    "b91_author_report": "cw_b91_report.md",
    "b91_runtime_vs_b90_patch": "cw_b91_runtime_vs_b90.patch",
    "b91_runtime_vs_m88_patch": "cw_b91_runtime_vs_m88.patch",
    "b91_author_multistep": "cw_b91_multistep_check.json",
    "b91_author_exit": "cw_b91_author_exit.json",
    "b91_root_freeze_initial": "cw_b91_root_freeze.json",
    "b91_root_cpu_log_initial_tests": "cw_b91_root_cpu_tests.log",
    "b91_root_cpu_exit_initial_tests": "cw_b91_root_cpu_tests_exit.json",
    "b91_root_gpu_log_initial_tests_2_test_defects": "cw_b91_root_gpu_tests.log",
    "b91_root_gpu_exit_initial_tests_2_test_defects": "cw_b91_root_gpu_tests_exit.json",
    "b91_root_gpu_first_observation": "cw_b91_root_gpu_first_observation.json",
    "b91_independent_review_prompt": "cw_b91_independent_review_prompt.md",
    "b91_independent_review_report": "cw_b91_review_report.md",
    "b91_independent_review_exit": "cw_b91_review_exit.json",
    "b91_test_correction_prompt": "cw_b91_gpu_test_correction_prompt.md",
    "b91_test_correction_report": "cw_b91_fix_report.md",
    "b91_test_correction_diff": "cw_b91_fix_test.diff",
    "b91_test_correction_archive_sums": "cw_b91_fix_archive/SHA256SUMS",
    "b91_test_correction_exit": "cw_b91_fix_exit.json",
    "b91_root_corrected_freeze": "cw_b91_corrected_root_freeze.json",
    "b91_root_corrected_cpu_log": "cw_b91_corrected_root_cpu_tests.log",
    "b91_root_corrected_cpu_exit": "cw_b91_corrected_root_cpu_tests_exit.json",
    "b91_root_corrected_gpu_log": "cw_b91_corrected_root_gpu_tests.log",
    "b91_root_corrected_gpu_exit": "cw_b91_corrected_root_gpu_tests_exit.json",
    "b91_root_frozen_multistep": "cw_b91_root_frozen_multistep_check.json",
    "b91_root_frozen_multistep_check": "cw_b91_root_frozen_multistep_check.py",
    "b91_independent_acceptance": "cw_b91_independent_acceptance.json",
}


# CW-E36: the 37 E33 / E34 / H3 / B92 records appended to the 142 E33 records (stable completed records only; RUNNING
# C29 / H4 / CPU 39570 logs, status records and E35 / H4 gates are deliberately not pinned)
B92_EVIDENCE = {
    "e33_M91_recipe_manifest": "cw_e36_logs/baseline_archive/dependencies/maple_water/manifest.json",
    "e33_M91_runtime_patch": "cw_e36_logs/baseline_archive/dependencies/maple_water/maple_74dd3e79_to_M91_runtime.patch",
    "e33_independent_acceptance": "cw_e33_independent_acceptance.json",
    "e34_m91_runtime_resolver_acceptance": "cw_e34_m91_runtime_resolver_acceptance.json",
    "h3_root_gpu_refusal_m91": "cw_h3_root_gpu_refusal.json",
    "b92_prompt": "cw_b92_ledger_increment_scratch_prompt.md",
    "b92_author_report": "cw_b92_report.md",
    "b92_author_exit": "cw_b92_exit.json",
    "b92_runtime_vs_m91_patch": "cw_b92_runtime_vs_m91.patch",
    "b92_author_multistep": "cw_b92_multistep_check.json",
    "b92_author_multistep_check": "cw_b92_multistep_check.py",
    "b92_root_arithmetic_probe": "cw_b92_root_arithmetic_probe.json",
    "b92_root_arithmetic_probe_script": "cw_b92_root_arithmetic_probe.py",
    "b92_root_freeze_first": "cw_b92_root_freeze.json",
    "b92_root_focused_checks_script_first": "run_cw_b92_focused_checks.py",
    "b92_root_focused_cpu_log_first": "cw_b92_root_focused_cpu.log",
    "b92_root_focused_cpu_exit_first": "cw_b92_root_focused_cpu_exit.json",
    "b92_root_focused_gpu_log_first_3_fixture_failures": "cw_b92_root_focused_gpu.log",
    "b92_root_focused_gpu_exit_first_3_fixture_failures": "cw_b92_root_focused_gpu_exit.json",
    "b92_fixture_correction_prompt": "cw_b92_gpu_fixture_correction_prompt.md",
    "b92_fixture_correction_report": "cw_b92_fix_report.md",
    "b92_fixture_correction_exit": "cw_b92_fix_exit.json",
    "b92_fixture_correction_test_diff": "cw_b92_fix_logs/test_diff_vs_original.diff",
    "b92_root_corrected_freeze": "cw_b92_corrected_root_freeze.json",
    "b92_root_corrected_checks_script": "run_cw_b92_corrected_focused_checks.py",
    "b92_root_corrected_focused_cpu_log": "cw_b92_corrected_root_focused_cpu.log",
    "b92_root_corrected_focused_cpu_exit": "cw_b92_corrected_root_focused_cpu_exit.json",
    "b92_root_corrected_focused_gpu_log": "cw_b92_corrected_root_focused_gpu.log",
    "b92_root_corrected_focused_gpu_exit": "cw_b92_corrected_root_focused_gpu_exit.json",
    "b92_independent_acceptance": "cw_b92_independent_acceptance.json",
    "b92_root_integrated_checks_script": "run_cw_b92_integrated_checks.py",
    "b92_root_integrated_cpu_log": "cw_b92_integrated_cpu.log",
    "b92_root_integrated_cpu_exit": "cw_b92_integrated_cpu_exit.json",
    "b92_root_final_integrated_checks_script": "run_cw_b92_final_integrated_checks.py",
    "b92_root_final_integrated_gpu_log": "cw_b92_final_integrated_gpu.log",
    "b92_root_final_integrated_gpu_exit": "cw_b92_final_integrated_gpu_exit.json",
    "b92_integrated_acceptance": "cw_b92_integrated_acceptance.json",
}


# CW-E39: the 33 E36 / E37 / H4 / H5 / B93 / C30 / C31 / B96 records appended to the 179 E36 records (stable completed
# records only; the active / pending H6 pipeline and any future proof are deliberately not pinned)
B96_EVIDENCE = {
    "e36_M92_recipe_manifest": "cw_e39_logs/baseline_archive/dependencies/maple_water/manifest.json",
    "e36_M92_runtime_patch": "cw_e39_logs/baseline_archive/dependencies/maple_water/maple_74dd3e79_to_M92_runtime.patch",
    "e36_independent_acceptance": "cw_e36_independent_acceptance.json",
    "e37_m92_runtime_resolver_acceptance": "cw_e37_m92_runtime_resolver_acceptance.json",
    "h4_root_gpu_refusal_b76_m92": "cw_h4_root_gpu_refusal.json",
    "h5_root_gpu_refusal_b93_m92": "cw_h5_root_gpu_refusal.json",
    "b93_independent_acceptance_syrup": "cw_b93_independent_acceptance.json",
    "b93_integrated_acceptance_syrup_m92": "cw_b93_integrated_acceptance.json",
    "c30_root_cpu_acceptance_b93_m92_plot1": "cw_c30_root_cpu_acceptance.json",
    "c31_root_gpu_acceptance_b93_m92_plot1": "cw_c31_root_gpu_acceptance.json",
    "b96_prompt": "cw_b96_deposition_memory_prompt.md",
    "b96_author_report": "cw_b96_report.md",
    "b96_author_exit": "cw_b96_exit.json",
    "b96_runtime_diff_windowed_exchange": "cw_b96_logs/cw_b96_windowed_exchange.diff",
    "b96_runtime_diff_numerics": "cw_b96_logs/cw_b96_numerics.diff",
    "b96_readonly_review": "cw_b96_readonly_review.md",
    "b96_root_freeze": "cw_b96_root_freeze.json",
    "b96_root_freeze_script": "cw_b96_root_freeze.py",
    "b96_root_run_script": "cw_b96_root_run.py",
    "b96_root_focused_frozen_log": "cw_b96_root_focused_frozen.log",
    "b96_root_focused_frozen_exit": "cw_b96_root_focused_frozen_exit.json",
    "b96_root_focused_candidate_log": "cw_b96_root_focused_candidate.log",
    "b96_root_focused_candidate_exit": "cw_b96_root_focused_candidate_exit.json",
    "b96_root_integrated_candidate_first_invocation_error_stderr": "cw_b96_root_integrated_candidate.stderr",
    "b96_root_integrated_candidate_first_invocation_error_exit4": "cw_b96_root_integrated_candidate_exit.json",
    "b96_root_integrated_candidate_corrected_log": "cw_b96_root_integrated_candidate_corrected.log",
    "b96_root_integrated_candidate_corrected_exit": "cw_b96_root_integrated_candidate_corrected_exit.json",
    "b96_root_integrated_window_frozen_gpu_log": "cw_b96_root_integrated_window_frozen.log",
    "b96_root_integrated_window_frozen_gpu_exit": "cw_b96_root_integrated_window_frozen_exit.json",
    "b96_root_raw_differential_review": "cw_b96_root_raw_differential_review.json",
    "b96_root_results_verify_script": "cw_b96_root_results_verify.py",
    "b96_independent_acceptance": "cw_b96_independent_acceptance.json",
    "b96_integrated_acceptance": "cw_b96_integrated_acceptance.json",
}


# CW-E44: the 37 E39 / H7 / C33 / E40 / B97 / E41 / H8 / E43 / C34 / R1 records appended to the 212 E39 records (stable
# completed records only; the LIVE R4 controller, any of its GPU results and B99 / M95 are deliberately not pinned)
B97_EVIDENCE = {
    "e39_M93_recipe_manifest": "cw_e44_logs/baseline_archive/dependencies/maple_water/manifest.json",
    "e39_M93_runtime_patch": "cw_e44_logs/baseline_archive/dependencies/maple_water/maple_74dd3e79_to_M93_runtime.patch",
    "e39_independent_acceptance": "cw_e39_independent_acceptance.json",
    "e39_root_tests_log": "cw_e39_root_tests.log",
    "e39_m93_runtime_origin": "cw_e39_m93_runtime_origin.json",
    "e39_m93_runtime_resolver_record": "cw_e39_logs/cw_e39_m93_runtime_resolver_record.json",
    "h7_root_gpu_refusal_b93_m93": "cw_h7_root_gpu_refusal.json",
    "c33_root_cpu_acceptance_b93_m93_plot1": "cw_c33_root_cpu_acceptance.json",
    "e40_m93_plot1_initial_acceptance": "cw_e40_m93_plot1_initial_acceptance.json",
    "b97_prompt": "cw_b97_erosion_memory_prompt.md",
    "b97_author_report": "cw_b97_report.md",
    "b97_author_exit": "cw_b97_exit.json",
    "b97_runtime_diff_windowed_exchange": "cw_b97_logs/cw_b97_windowed_exchange.diff",
    "b97_root_freeze": "cw_b97_root_freeze.json",
    "b97_root_freeze_script": "cw_b97_root_freeze.py",
    "b97_root_run_script": "cw_b97_root_run.py",
    "b97_root_focused_cpu_log": "cw_b97_root_focused_cpu.log",
    "b97_root_focused_cpu_exit": "cw_b97_root_focused_cpu_exit.json",
    "b97_root_focused_gpu_log": "cw_b97_root_focused_gpu.log",
    "b97_root_focused_gpu_exit": "cw_b97_root_focused_gpu_exit.json",
    "b97_root_integrated_cpu_log": "cw_b97_root_integrated_cpu.log",
    "b97_root_integrated_cpu_exit": "cw_b97_root_integrated_cpu_exit.json",
    "b97_root_integrated_gpu_log": "cw_b97_root_integrated_gpu.log",
    "b97_root_integrated_gpu_exit": "cw_b97_root_integrated_gpu_exit.json",
    "b97_root_raw_differential_review": "cw_b97_root_raw_differential_review.json",
    "b97_root_junit_review": "cw_b97_root_junit_review.json",
    "b97_root_results_verify_script": "cw_b97_root_results_verify.py",
    "b97_root_accept_script": "cw_b97_root_accept.py",
    "b97_independent_acceptance": "cw_b97_independent_acceptance.json",
    "b97_integrated_acceptance": "cw_b97_integrated_acceptance.json",
    "e41_m94_chastre_gpu_initial_acceptance": "cw_e41_m94_gpu_initial_acceptance.json",
    "h8_root_gpu_refusal_b93_m94": "cw_h8_root_gpu_refusal.json",
    "e43_m94_plot1_initial_acceptance": "cw_e43_m94_plot1_initial_acceptance.json",
    "c34_root_cpu_acceptance_b93_m94_plot1": "cw_c34_root_cpu_acceptance.json",
    "c34_cpu_wet_restart_comparison_b93_m94_plot1": "cw_c34_cpu_wet_restart_comparison.json",
    "r1_root_pipeline_exit_gpu0_busy_no_model": "cw_r1_root_pipeline_exit.json",
    "r1_root_outer_exit": "cw_r1_root_outer_exit.json",
}


# CW-E46: the 31 E44 / B106 / B106C1 / B107 / B108C1 / E45 / E46 records appended to the 249 E44 records (stable completed
# records only; the ACTIVE root B111 GPU reference, its store / results and any future proof are deliberately not pinned)
D24C_EVIDENCE = {
    "e44_M94_recipe_manifest": "cw_e46_logs/baseline_archive/dependencies/maple_water/manifest.json",
    "e44_M94_runtime_patch": "cw_e46_logs/baseline_archive/dependencies/maple_water/maple_74dd3e79_to_M94_runtime.patch",
    "e44_independent_acceptance": "cw_e44_independent_acceptance.json",
    "e44_root_tests_log": "cw_e44_root_tests.log",
    "e44_m94_runtime_origin": "cw_e44_m94_runtime_origin.json",
    "e44_m94_runtime_resolver_record": "cw_e44_logs/cw_e44_m94_runtime_resolver_record.json",
    "b106_prompt": "cw_b106_overlay_implementation_prompt.md",
    "b106_author_report": "cw_b106_report.md",
    "b106_author_source_diff": "cw_b106_source.diff",
    "b106_root_closed_review": "cw_b106_root_closed_review.json",
    "b106c1_correction_prompt": "cw_b106c1_correction_prompt.md",
    "b106c1_author_report": "cw_b106c1_report.md",
    "b106c1_author_exit": "cw_b106c1_exit.json",
    "b106c1_author_source_diff": "cw_b106c1_source.diff",
    "b106c1_source_vs_m94_diff": "cw_b106c1_source_vs_m94.diff",
    "b106c1_root_review_freeze": "cw_b106c1_root_review_freeze.json",
    "b106c1_independent_acceptance": "cw_b106c1_independent_acceptance.json",
    "b107_prompt": "cw_b107_publish_integration_prompt.md",
    "b107_author_report": "cw_b107_report.md",
    "b107_author_exit": "cw_b107_exit.json",
    "b107_author_source_diff": "cw_b107_source.diff",
    "b107_root_review_freeze": "cw_b107_root_review_freeze.json",
    "b107_independent_acceptance": "cw_b107_independent_acceptance.json",
    "b107_root_dependency_delivery_acceptance": "cw_b107_root_dependency_delivery_acceptance.json",
    "b107_root_delivery_origin_full_project": "cw_b107_root_delivery_origin.json",
    "b107_root_delivery_cpu_venv": "cw_b107_root_delivery_cpu_venv.json",
    "b107_root_delivery_cupy_venv_cuda_hidden": "cw_b107_root_delivery_cupy_venv_cuda_hidden.json",
    "b108c1_independent_acceptance_syrup_5f_on_24c6": "cw_b108c1_independent_acceptance.json",
    "e45_audit_report": "cw_e45_report.md",
    "e45_root_review": "cw_e45_root_review.json",
    "e46_prompt": "cw_e46_prompt.md",
}


def test_24c6_modified_files_and_scope_provenance_are_exact():
    """CW-E46: the only M94 -> 24c6 differences are the reviewed SIX modified files (M94 / B106C1 / 24c6 hashes); each
    keeps exactly its earlier (historical) stages and gains ONLY water_B106C1 and/or water_B107; all six stay NEW
    files of the patch (no section added); no file is added or deleted; every other changed file keeps exactly its
    E44 (historical) stages; the E44 M93 -> M94, E39 M92 -> M93 (incl. the first-time _numerics.py section), E36
    M91 -> M92, E33 M88 -> M91 and E29 M85 -> M88 records are kept exactly as history; the B85 / B90 / B91 / B92 / B96 /
    B97 / B106 / B106C1 / B107 tests are NOT runtime files; the patch never touches anything outside src/maple +
    pyproject.toml."""
    m, prov = MANIFEST, MANIFEST["provenance"]
    target = prov["target_runtime"]
    hist94 = prov["historical_M94_recipe"]["e44_target_runtime"]
    hist93 = prov["historical_M93_recipe"]["e39_target_runtime"]
    hist92 = prov["historical_M92_recipe"]["e36_target_runtime"]
    for f in D24C_CHANGES:
        m94_sha = (B106C1_MODIFIED.get(f) or B107_MODIFIED[f])[0]
        b106c1_sha = B106C1_MODIFIED[f][1] if f in B106C1_MODIFIED else m94_sha
        final = B107_MODIFIED[f][1] if f in B107_MODIFIED else b106c1_sha
        if f in B107_MODIFIED:
            assert B107_MODIFIED[f][0] == b106c1_sha, f  # the B107 producer is exactly the B106C1 file
        assert target["changed_from_M94_sha256"][f] == {"M94": m94_sha, "B106C1": b106c1_sha, "24c6": final}, f
        assert f in m["patch"]["new_files"] and m["post_patch"]["files"][f] == final
        stages = prov["changed_file_stages"][f]
        added = (["water_B106C1"] if f in B106C1_MODIFIED else []) + (["water_B107"] if f in B107_MODIFIED else [])
        assert stages[len(stages) - len(added):] == added and stages[:len(stages) - len(added)], (f, stages)
        assert not {"water_B106C1", "water_B107"} & set(stages[:len(stages) - len(added)])
        assert target["upstream_relation"][f] == "new file in the patch since the water stages"
    others_24c = [f for f in m["patch"]["changed_files"] if f not in D24C_CHANGES]
    assert len(others_24c) == len(m["patch"]["changed_files"]) - 6 == 29
    assert all(not {"water_B106C1", "water_B107"} & set(prov["changed_file_stages"][f]) for f in others_24c)
    assert hist94["changed_from_M93_sha256"] == {k: {"M93": old, "M94": new} for k, (old, new) in M94_MODIFIED.items()}
    for f, (old, new) in M94_MODIFIED.items():
        assert f in m["patch"]["changed_files"] and m["post_patch"]["files"][f] == new  # unchanged M94 -> 24c6
        kind = "new" if f in m["patch"]["new_files"] else "modified" if f in m["patch"]["modified_files"] else None
        assert kind == M94_UPSTREAM_KIND[f], (f, kind)
        assert prov["changed_file_stages"][f] == M94_PRIOR_STAGES[f] + ["water_M94"], prov["changed_file_stages"][f]
        assert hist93["changed_from_M92_sha256"][f]["M93"] == old  # the M94 producer is exactly the M93 file
    numerics = "src/maple/surface/voxels/_numerics.py"  # unmodified upstream through M92: its base blob is M92's file
    assert m["pre_patch"]["files"][numerics]["sha256"] == M93_MODIFIED[numerics][0]
    assert m["post_patch"]["files"][numerics] == M93_MODIFIED[numerics][1]  # unchanged M93 -> M94 -> 24c6
    assert prov["changed_file_stages"][numerics] == M93_PRIOR_STAGES[numerics] + ["water_M93"]
    assert M93_UPSTREAM_KIND[numerics] == "modified" and numerics in m["patch"]["modified_files"]
    assert hist93["name"] == "M93" and hist93["changed_from_M92"] == M93_CHANGES
    assert hist93["changed_from_M92_sha256"] == M93_HIST_SHA256
    assert prov["historical_M93_recipe"]["package_source_digest"] == hist93["package_source_digest"] == M93_DIGEST
    assert hist92["name"] == "M92" and hist92["changed_from_M91"] == M92_CHANGES
    assert hist92["changed_from_M91_sha256"] == M92_HIST_SHA256
    assert prov["historical_M92_recipe"]["package_source_digest"] == hist92["package_source_digest"] == M92_DIGEST
    hist = prov["historical_M91_recipe"]["e33_target_runtime"]
    assert hist["name"] == "M91" and hist["changed_from_M88"] == M91_CHANGES
    assert hist["changed_from_M88_sha256"] == M91_HIST_SHA256
    others = [f for f in m["patch"]["changed_files"] if f not in M94_CHANGES]
    assert len(others) == 34 and all("water_M94" not in prov["changed_file_stages"][f] for f in others)
    assert hist["package_source_digest"] == prov["historical_M91_recipe"]["package_source_digest"] == M91_DIGEST
    e29 = prov["historical_M88_recipe"]["e29_target_runtime"]
    assert e29["name"] == "M88" and e29["changed_from_M85"] == M88_CHANGES
    assert e29["changed_from_M85_sha256"] == M88_HIST_SHA256
    assert prov["historical_M88_recipe"]["package_source_digest"] == M88_DIGEST
    e26 = prov["historical_M85_recipe"]["e26_target_runtime"]
    assert e26["name"] == "M85" and e26["changed_from_M82"] == M85_CHANGES
    assert e26["changed_from_M82_sha256"] == M85_HIST_SHA256
    assert prov["historical_M85_recipe"]["package_source_digest"] == M85_DIGEST
    e22 = prov["historical_M82_recipe"]["e22_target_runtime"]
    assert e22["name"] == "M82" and prov["historical_M82_recipe"]["package_source_digest"] == M82_DIGEST
    assert prov["historical_M79_recipe"]["package_source_digest"] == M79_DIGEST
    assert prov["historical_M75_recipe"]["package_source_digest"] == M75_DIGEST
    assert prov["historical_M67_recipe"]["package_source_digest"] == M67_DIGEST
    assert all(f.startswith("src/maple/") or f == "pyproject.toml" for f in m["post_patch"]["files"])
    for test_file in (B85_TEST, *B90_B91_TESTS, B92_TEST, *B96_TESTS, *B97_TESTS, *B106C1_B107_TESTS):
        assert test_file not in m["post_patch"]["files"] and test_file not in m["patch"]["changed_files"]
    assert not any(f.startswith("tests/") for f in m["patch"]["changed_files"])
    assert set(m["post_patch"]["files"]) - set(m["pre_patch"]["files"]) == set(m["patch"]["new_files"])
    assert not set(m["pre_patch"]["files"]) - set(m["post_patch"]["files"])
    assert m["upstream"]["revision"] == "74dd3e79d9a7df85244355439e3a4a762abb9273"
    assert m["patch"]["file"] == "maple_74dd3e79_to_24c6_runtime.patch"
    assert sorted(p.name for p in tool.RECIPE_DIR.iterdir()) == ["manifest.json", m["patch"]["file"]]
    if REFERENCE:  # the reference holds EXACTLY the manifest runtime files (no extra/missing file)
        ref = Path(REFERENCE)
        actual = {"pyproject.toml"} | {p.relative_to(ref).as_posix() for p in (ref / "src/maple").rglob("*")
                                       if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}
        assert actual == set(m["post_patch"]["files"])


def test_recorded_provenance_evidence_matches_the_local_records():
    """The evidence SHA-256 values recorded in the manifest equal the local records they name (SYRUP-relative,
    git-ignored); a missing record is reported, not silently accepted."""
    syrup = TREE.parents[2] if TREE.parent.name == "conservative_water" else TREE
    missing, checked = [], 0
    for key, rec in MANIFEST["provenance"]["evidence_sha256"].items():
        path = syrup / rec["path"]
        if not path.is_file():
            missing.append(key)
            continue
        assert sha(path.read_bytes()) == rec["sha256"], key
        checked += 1
    # CW-E26: the 65 E22 records are carried unchanged and exactly the 16 B85 / E22 records are appended;
    # CW-E29: those 81 E26 records are carried unchanged and exactly the 22 B88 / E26 records are appended;
    # CW-E33: those 103 E29 records are carried unchanged and exactly the 39 E29 / B90 / B91 records are appended;
    # CW-E36: those 142 E33 records are carried unchanged and exactly the 37 E33 / E34 / H3 / B92 records are appended;
    # CW-E39: those 179 E36 records are carried unchanged and exactly the 33 E36 / E37 / H4 / H5 / B93 / C30 / C31 /
    # B96 records are appended;
    # CW-E44: those 212 E39 records are carried unchanged and exactly the 37 E39 / H7 / C33 / E40 / B97 / E41 / H8 /
    # E43 / C34 / R1 records are appended;
    # CW-E46: those 249 E44 records are carried unchanged and exactly the 31 E44 / B106 / B106C1 / B107 / B108C1 / E45 /
    # E46 records are appended
    ev = MANIFEST["provenance"]["evidence_sha256"]
    assert len(ev) == 280 and set(B85_EVIDENCE) <= set(ev) and set(B88_EVIDENCE) <= set(ev) and set(B91_EVIDENCE) <= \
        set(ev) and set(B92_EVIDENCE) <= set(ev) and set(B96_EVIDENCE) <= set(ev) and set(B97_EVIDENCE) <= set(ev) and \
        set(D24C_EVIDENCE) <= set(ev) and len(D24C_EVIDENCE) == 31
    for key, rel in {**B85_EVIDENCE, **B88_EVIDENCE, **B91_EVIDENCE, **B92_EVIDENCE, **B96_EVIDENCE,
                     **B97_EVIDENCE, **D24C_EVIDENCE}.items():
        assert ev[key]["path"] == "agent_handoffs/tasks/conservative_water/" + rel, key
    e22_manifest = syrup / ev["e22_M82_recipe_manifest"]["path"]
    if e22_manifest.is_file():  # the archived accepted E22 manifest (hash-checked above): its 65 records are unchanged
        e22_ev = json.loads(e22_manifest.read_bytes())["provenance"]["evidence_sha256"]
        assert len(e22_ev) == 65 and {k: ev[k] for k in e22_ev} == e22_ev
    e26_manifest = syrup / ev["e26_M85_recipe_manifest"]["path"]
    if e26_manifest.is_file():  # the archived accepted E26 manifest (hash-checked above): its 81 records are unchanged
        e26_ev = json.loads(e26_manifest.read_bytes())["provenance"]["evidence_sha256"]
        assert len(e26_ev) == 81 and {k: ev[k] for k in e26_ev} == e26_ev
        assert set(ev) - set(e26_ev) == set(B88_EVIDENCE) | set(B91_EVIDENCE) | set(B92_EVIDENCE) | \
            set(B96_EVIDENCE) | set(B97_EVIDENCE) | set(D24C_EVIDENCE)
        if e22_manifest.is_file():
            assert set(e26_ev) - set(e22_ev) == set(B85_EVIDENCE)
    e29_manifest = syrup / ev["e29_M88_recipe_manifest"]["path"]
    if e29_manifest.is_file():  # the archived accepted E29 manifest (hash-checked above): its 103 records are unchanged
        e29_ev = json.loads(e29_manifest.read_bytes())["provenance"]["evidence_sha256"]
        assert len(e29_ev) == 103 and {k: ev[k] for k in e29_ev} == e29_ev
        assert set(ev) - set(e29_ev) == set(B91_EVIDENCE) | set(B92_EVIDENCE) | set(B96_EVIDENCE) | \
            set(B97_EVIDENCE) | set(D24C_EVIDENCE)
    e33_manifest = syrup / ev["e33_M91_recipe_manifest"]["path"]
    if e33_manifest.is_file():  # the archived accepted E33 manifest (hash-checked above): its 142 records are unchanged
        e33_ev = json.loads(e33_manifest.read_bytes())["provenance"]["evidence_sha256"]
        assert len(e33_ev) == 142 and {k: ev[k] for k in e33_ev} == e33_ev
        assert set(ev) - set(e33_ev) == set(B92_EVIDENCE) | set(B96_EVIDENCE) | set(B97_EVIDENCE) | set(D24C_EVIDENCE)
    e36_manifest = syrup / ev["e36_M92_recipe_manifest"]["path"]
    if e36_manifest.is_file():  # the archived accepted E36 manifest (hash-checked above): its 179 records are unchanged
        e36_ev = json.loads(e36_manifest.read_bytes())["provenance"]["evidence_sha256"]
        assert len(e36_ev) == 179 and {k: ev[k] for k in e36_ev} == e36_ev
        assert set(ev) - set(e36_ev) == set(B96_EVIDENCE) | set(B97_EVIDENCE) | set(D24C_EVIDENCE)
    e39_manifest = syrup / ev["e39_M93_recipe_manifest"]["path"]
    if e39_manifest.is_file():  # the archived accepted E39 manifest (hash-checked above): its 212 records are unchanged
        e39_ev = json.loads(e39_manifest.read_bytes())["provenance"]["evidence_sha256"]
        assert len(e39_ev) == 212 and {k: ev[k] for k in e39_ev} == e39_ev
        assert set(ev) - set(e39_ev) == set(B97_EVIDENCE) | set(D24C_EVIDENCE)
    e44_manifest = syrup / ev["e44_M94_recipe_manifest"]["path"]
    if e44_manifest.is_file():  # the archived accepted E44 manifest (hash-checked above): its 249 records are unchanged
        assert ev["e44_M94_recipe_manifest"]["sha256"] == E44_MANIFEST_SHA
        assert ev["e44_M94_runtime_patch"]["sha256"] == E44_PATCH_SHA
        e44_ev = json.loads(e44_manifest.read_bytes())["provenance"]["evidence_sha256"]
        assert len(e44_ev) == 249 and {k: ev[k] for k in e44_ev} == e44_ev
        assert set(ev) - set(e44_ev) == set(D24C_EVIDENCE)
    if missing:
        pytest.skip(f"local evidence records unavailable: {missing} ({checked} checked)")


# --- actual reconstruction ----------------------------------------------------------------------------------------------
@needs_repo
def test_actual_reconstruction_is_exact_and_leaves_the_repository_untouched(tmp_path):
    before = repo_state()
    dest = tmp_path / "runtime"
    receipt = tool.reconstruct(REPO, dest)
    assert repo_state() == before
    assert leftovers(tmp_path) == []
    assert receipt["post_patch_file_count"] == 341 and receipt["package_source_digest"] == \
        MANIFEST["post_patch"]["package_source_digest"]["digest_sha256"]
    files = {p.relative_to(dest).as_posix(): p for p in dest.rglob("*") if p.is_file()}
    assert set(files) == set(MANIFEST["post_patch"]["files"]) | {tool.RECEIPT_NAME}
    for rel, digest in MANIFEST["post_patch"]["files"].items():
        assert sha(files[rel].read_bytes()) == digest, rel
    assert not any(p.is_symlink() for p in dest.rglob("*"))
    if REFERENCE:
        ref = Path(REFERENCE)
        for rel in MANIFEST["post_patch"]["files"]:
            assert (ref / rel).read_bytes() == files[rel].read_bytes(), rel


# --- destination safety ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["empty_dir", "file", "symlink", "dangling_symlink"])
def test_existing_destination_is_never_reused_or_replaced(tmp_path, kind):
    dest = tmp_path / "runtime"
    sentinel = tmp_path / "unrelated_sentinel"
    sentinel.write_bytes(b"keep")
    if kind == "empty_dir":
        dest.mkdir()
    elif kind == "file":
        dest.write_bytes(b"existing")
    elif kind == "symlink":
        dest.symlink_to(sentinel)
    else:
        dest.symlink_to(tmp_path / "nowhere")
    with pytest.raises(tool.RecipeError, match="already exists"):
        tool.reconstruct(REPO, dest)
    assert sentinel.read_bytes() == b"keep" and leftovers(tmp_path) == []
    if kind == "empty_dir":
        assert dest.is_dir() and not any(dest.iterdir())
    elif kind == "file":
        assert dest.read_bytes() == b"existing"


def test_destination_inside_the_repository_or_without_parent_is_refused(tmp_path):
    with pytest.raises(tool.RecipeError, match="inside the MAPLE repository"):
        tool.reconstruct(tmp_path, tmp_path / "sub")
    assert not (tmp_path / "sub").exists()
    with pytest.raises(tool.RecipeError, match="parent"):
        tool.reconstruct(REPO, tmp_path / "missing" / "runtime")


# --- recipe integrity -----------------------------------------------------------------------------------------------
def test_tampered_manifest_bytes_are_refused_by_the_pin(tmp_path):
    recipe, _ = recipe_copy(tmp_path, edit_manifest=lambda m: m.update(description="edited"))
    with pytest.raises(tool.RecipeError, match="manifest SHA-256"):
        tool.reconstruct(REPO, tmp_path / "out", recipe_dir=recipe)
    assert not (tmp_path / "out").exists() and leftovers(tmp_path) == []


def test_tampered_patch_bytes_are_refused_by_the_manifest_pin(tmp_path):
    recipe, pin = recipe_copy(tmp_path, edit_patch=lambda b: b + b" ", repin_patch=False)
    with pytest.raises(tool.RecipeError, match="patch SHA-256"):
        tool.reconstruct(REPO, tmp_path / "out", recipe_dir=recipe, expected_manifest_sha256=pin)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("bad", ["../evil.py", "/abs/x.py", "src/maple/../../x.py", "src/other/x.py",
                                 "src/maple/./x.py", "src/maple//x.py", "src\\maple\\x.py", "src/maple"])
def test_unsafe_manifest_paths_are_refused_before_any_git_or_staging(tmp_path, bad):
    def edit(m):
        m["post_patch"]["files"][bad] = "0" * 64
    recipe, pin = recipe_copy(tmp_path, edit_manifest=edit)
    with pytest.raises(tool.RecipeError, match="path|malformed"):
        tool.reconstruct(REPO, tmp_path / "out", recipe_dir=recipe, expected_manifest_sha256=pin,
                         git="/nonexistent/git-must-not-run")
    assert leftovers(tmp_path) == [] and not (tmp_path / "out").exists()


def test_duplicate_json_keys_are_refused(tmp_path):
    recipe, _ = recipe_copy(tmp_path)
    raw = (recipe / tool.MANIFEST_NAME).read_bytes().replace(b'"schema":', b'"schema": "x",\n "schema":', 1)
    (recipe / tool.MANIFEST_NAME).write_bytes(raw)
    with pytest.raises(tool.RecipeError, match="duplicate"):
        tool.reconstruct(REPO, tmp_path / "out", recipe_dir=recipe, expected_manifest_sha256=sha(raw))


# --- wrong Git object -------------------------------------------------------------------------------------------------
@needs_repo
def test_wrong_or_missing_git_object_is_refused(tmp_path):
    parent = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"{REV}^"], capture_output=True,
                            check=True).stdout.decode().strip()
    for rev, message in ((parent, "upstream"), ("0" * 40, "rev-parse|resolves")):
        out = tmp_path / f"out_{rev[:6]}"
        work = tmp_path / f"recipe_{rev[:6]}"
        work.mkdir()
        recipe, pin = recipe_copy(work, edit_manifest=lambda m, r=rev: m["upstream"].update(revision=r))
        with pytest.raises(tool.RecipeError, match=message):
            tool.reconstruct(REPO, out, recipe_dir=recipe, expected_manifest_sha256=pin)
        assert not out.exists() and leftovers(tmp_path) == []


# --- late verification failures publish nothing ------------------------------------------------------------------------
@needs_repo
def test_patch_that_applies_but_produces_wrong_bytes_publishes_nothing(tmp_path):
    def edit(data: bytes) -> bytes:  # change one ADDED line's content; headers and hunk counts are unchanged
        lines = data.split(b"\n")
        for i, line in enumerate(lines):
            if line.startswith(b"+") and not line.startswith(b"+++") and len(line) > 3:
                lines[i] = line + b"#"
                return b"\n".join(lines)
        raise AssertionError("no added line")
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"keep")
    recipe, pin = recipe_copy(tmp_path, edit_patch=edit)
    with pytest.raises(tool.RecipeError, match="post-patch files differ"):
        tool.reconstruct(REPO, tmp_path / "out", recipe_dir=recipe, expected_manifest_sha256=pin)
    assert not (tmp_path / "out").exists() and leftovers(tmp_path) == [] and sentinel.read_bytes() == b"keep"


@needs_repo
def test_corrupt_hunk_fails_git_apply_and_publishes_nothing(tmp_path):
    def edit(data: bytes) -> bytes:  # break one context line of a MODIFIED file
        lines = data.split(b"\n")
        in_mod = False
        for i, line in enumerate(lines):
            if line.startswith(b"--- a/"):
                in_mod = True
            elif line.startswith(b"--- /dev/null"):
                in_mod = False
            elif in_mod and line.startswith(b" ") and len(line) > 3:
                lines[i] = b" CORRUPTED" + line[1:]
                return b"\n".join(lines)
        raise AssertionError("no context line")
    recipe, pin = recipe_copy(tmp_path, edit_patch=edit)
    with pytest.raises(tool.RecipeError, match="git apply"):
        tool.reconstruct(REPO, tmp_path / "out", recipe_dir=recipe, expected_manifest_sha256=pin)
    assert not (tmp_path / "out").exists() and leftovers(tmp_path) == []


@needs_repo
def test_wrong_pinned_post_hash_publishes_nothing(tmp_path):
    def edit(m):
        first = sorted(m["post_patch"]["files"])[0]
        m["post_patch"]["files"][first] = "f" * 64
    recipe, pin = recipe_copy(tmp_path, edit_manifest=edit)
    with pytest.raises(tool.RecipeError, match="differ from their pinned"):
        tool.reconstruct(REPO, tmp_path / "out", recipe_dir=recipe, expected_manifest_sha256=pin)
    assert not (tmp_path / "out").exists() and leftovers(tmp_path) == []


# --- patch structure (unit) --------------------------------------------------------------------------------------------
GOOD = ("diff --git a/src/maple/x.py b/src/maple/x.py\n"
        "index " + "1" * 40 + ".." + "2" * 40 + " 100644\n--- a/src/maple/x.py\n+++ b/src/maple/x.py\n"
        "@@ -1 +1 @@\n-a\n+b\n")


def _m(changed, new=()):
    return {"patch": {"changed_files": list(changed), "new_files": list(new)}}


@pytest.mark.parametrize("label,text", [
    ("rename", GOOD.replace("index ", "similarity index 90%\nrename from src/maple/x.py\nrename to src/maple/y.py\nindex ", 1)),
    ("deleted", GOOD.replace("index ", "deleted file mode 100644\nindex ", 1)),
    ("mode", GOOD.replace("index ", "old mode 100644\nnew mode 100755\nindex ", 1)),
    ("binary", GOOD.replace("@@ -1 +1 @@\n-a\n+b\n", "GIT binary patch\nliteral 1\n")),
    ("symlink_new", GOOD.replace("index ", "new file mode 120000\nindex ", 1)),
    ("mismatched_paths", GOOD.replace("b/src/maple/x.py\nindex", "b/src/maple/y.py\nindex", 1)),
    ("traversal", GOOD.replace("src/maple/x.py", "src/maple/../../x.py")),
    ("preamble", "From: someone\n" + GOOD),
    ("stray_line", GOOD + "garbage line\n"),
    ("no_hunk", GOOD.replace("@@ -1 +1 @@\n-a\n+b\n", "")),
])
def test_hostile_patch_structures_are_refused(label, text):
    with pytest.raises(tool.RecipeError):
        tool.parse_patch(text.encode(), _m(["src/maple/x.py"]))


def test_patch_must_name_exactly_the_manifest_files():
    assert tool.parse_patch(GOOD.encode(), _m(["src/maple/x.py"])) == ["src/maple/x.py"]
    with pytest.raises(tool.RecipeError, match="differ from the manifest"):
        tool.parse_patch(GOOD.encode(), _m(["src/maple/x.py", "src/maple/z.py"]))
    with pytest.raises(tool.RecipeError, match="twice"):
        tool.parse_patch((GOOD + GOOD).encode(), _m(["src/maple/x.py"]))


# --- archive members (unit) ----------------------------------------------------------------------------------------------
def _tar(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, kind, data in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            elif kind == "symlink":
                info.type, info.linkname = tarfile.SYMTYPE, data
                tar.addfile(info)
            elif kind == "hardlink":
                info.type, info.linkname = tarfile.LNKTYPE, data
                tar.addfile(info)
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
    return buf.getvalue()


@pytest.mark.parametrize("label,members", [
    ("symlink", [("src/maple/a.py", "symlink", "/etc/passwd")]),
    ("hardlink", [("src/maple/a.py", "hardlink", "pyproject.toml")]),
    ("dotdot", [("src/maple/../../a.py", "file", b"x")]),
    ("absolute", [("/src/maple/a.py", "file", b"x")]),
    ("unexpected", [("src/maple/a.py", "file", b"x"), ("src/maple/extra.py", "file", b"y")]),
    ("duplicate", [("src/maple/a.py", "file", b"x"), ("src/maple/a.py", "file", b"x")]),
    ("wrong_bytes", [("src/maple/a.py", "file", b"y")]),
    ("unexpected_dir", [("evil", "dir", None), ("src/maple/a.py", "file", b"x")]),
    ("missing", []),
])
def test_hostile_archives_are_refused_and_write_only_inside_the_tree(tmp_path, label, members):
    archive = tmp_path / "a.tar"
    archive.write_bytes(_tar(members))
    tree = tmp_path / "tree"
    tree.mkdir()
    with pytest.raises(tool.RecipeError):
        tool.extract_archive(archive, tree, {"upstream": {"revision": REV},
                                             "pre_patch": {"files": {"src/maple/a.py": {"size": 1, "sha256": sha(b"x")}}}})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.tar", "tree"]


def test_cli_reports_refusals_with_exit_1(tmp_path, capsys):
    dest = tmp_path / "exists"
    dest.mkdir()
    assert tool.main(["--maple-repo", str(REPO), "--dest", str(dest)]) == 1
    assert "already exists" in capsys.readouterr().err


# --- CW-E9 correction 1: no-clobber publication under a concurrent destination ----------------------------------------
def _make_target(dest: Path, kind: str, sentinel: Path) -> None:
    if kind == "empty_dir":
        dest.mkdir()
    elif kind == "file":
        dest.write_bytes(b"concurrent file")
    elif kind == "symlink":
        dest.symlink_to(sentinel)
    else:
        dest.symlink_to(dest.parent / "nowhere")


def _identity(dest: Path):
    info = os.lstat(dest)
    content = os.readlink(dest) if os.path.islink(dest) else (dest.read_bytes() if dest.is_file() else sorted(os.listdir(dest)))
    return info.st_ino, info.st_mode, content


@needs_repo
@pytest.mark.parametrize("kind", ["empty_dir", "file", "symlink", "dangling_symlink"])
def test_destination_created_concurrently_just_before_publication_is_never_replaced(tmp_path, monkeypatch, kind):
    """The original race (root repro: an EMPTY directory created between the check and os.rename was replaced) and
    its other forms: the target appears AFTER every check and immediately before the publishing system call."""
    dest = tmp_path / "runtime"
    sentinel = tmp_path / "unrelated_sentinel"
    sentinel.write_bytes(b"keep")
    original = tool._rename_noreplace
    seen = {}

    def create_then_publish(src, dst):
        assert Path(dst) == dest and not os.path.lexists(dest)
        _make_target(dest, kind, sentinel)
        seen["identity"] = _identity(dest)
        return original(src, dst)

    monkeypatch.setattr(tool, "_rename_noreplace", create_then_publish)
    with pytest.raises(tool.RecipeError, match="without replacing anything"):
        tool.reconstruct(REPO, dest)
    assert _identity(dest) == seen["identity"]  # same inode, mode and content: not replaced
    assert sentinel.read_bytes() == b"keep"
    assert leftovers(tmp_path) == []  # only this run's own staging was removed
    assert sorted(p.name for p in tmp_path.iterdir()) == ["runtime", "unrelated_sentinel"]


def test_publisher_refuses_an_existing_target_of_every_kind_at_the_system_call(tmp_path):
    """The no-replace rename itself (no reconstruction): every existing target form makes the kernel refuse."""
    for kind in ("empty_dir", "file", "symlink", "dangling_symlink"):
        work = tmp_path / kind
        work.mkdir()
        sentinel = work / "sentinel"
        sentinel.write_bytes(b"keep")
        src = work / "src"
        src.mkdir()
        (src / "payload").write_bytes(b"new")
        dst = work / "dst"
        _make_target(dst, kind, sentinel)
        before = _identity(dst)
        with pytest.raises(tool.RecipeError, match="without replacing anything"):
            tool._rename_noreplace(src, dst)
        assert _identity(dst) == before and (src / "payload").read_bytes() == b"new"
        assert sentinel.read_bytes() == b"keep"
    fresh_src, fresh_dst = tmp_path / "fresh_src", tmp_path / "fresh_dst"
    fresh_src.mkdir()
    tool._rename_noreplace(fresh_src, fresh_dst)
    assert fresh_dst.is_dir() and not fresh_src.exists()


@pytest.mark.parametrize("failure", ["platform", "no_renameat2"])
def test_unsupported_platform_fails_closed_before_any_work(tmp_path, monkeypatch, failure):
    if failure == "platform":
        monkeypatch.setattr(tool.sys, "platform", "darwin")
    else:
        class NoRenameat2:
            def __init__(self, *a, **k):
                pass
        monkeypatch.setattr(tool.ctypes, "CDLL", NoRenameat2)
    with pytest.raises(tool.RecipeError, match="renameat2"):
        tool.reconstruct(REPO, tmp_path / "out", git="/nonexistent/git-must-not-run")
    assert sorted(p.name for p in tmp_path.iterdir()) == []


@needs_repo
def test_publication_failing_closed_at_the_end_leaves_nothing(tmp_path, monkeypatch):
    """renameat2 becomes unavailable only at publication time: refusal, no destination, only own staging removed."""
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"keep")
    calls = {"n": 0}
    real = tool._renameat2

    def flaky():
        calls["n"] += 1
        if calls["n"] > 1:
            raise tool.RecipeError("libc has no renameat2; atomic no-replace publication impossible; refusing")
        return real()

    monkeypatch.setattr(tool, "_renameat2", flaky)
    with pytest.raises(tool.RecipeError, match="renameat2"):
        tool.reconstruct(REPO, tmp_path / "out")
    assert calls["n"] == 2 and not os.path.lexists(tmp_path / "out")
    assert leftovers(tmp_path) == [] and sentinel.read_bytes() == b"keep"


# --- CW-E9 correction 2: canonical actual source-root guard -----------------------------------------------------------
import tempfile  # noqa: E402

PREVIOUS_TOOL = Path(os.environ.get(
    "MAPLE_WATER_RECIPE_PREVIOUS_TOOL",
    "/home/okin/SYRUP/agent_handoffs/tasks/conservative_water/cw_e9_correction2_archive/"
    "reconstruct_water_maple_correction1.py"))


class Intercepted(Exception):
    """Raised INSTEAD of creating a staging directory: proves where a tool would have written, writing nothing."""


def _intercept_staging(monkeypatch, record):
    def refuse_mkdtemp(*args, **kwargs):
        record.append(str(kwargs.get("dir")))
        raise Intercepted(kwargs.get("dir"))
    monkeypatch.setattr(tempfile, "mkdtemp", refuse_mkdtemp)


def _source_listing():
    return {name: sorted(os.listdir(REPO / name)) for name in ("", "src", "tests") if (REPO / name).is_dir()}


def _previous_tool():
    if not PREVIOUS_TOOL.is_file():
        pytest.skip("archived correction-1 tool unavailable; negative control not run")
    spec = importlib.util.spec_from_file_location("reconstruct_water_maple_correction1", PREVIOUS_TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bypass_cases(tmp_path):
    alias = tmp_path / "own_alias_to_MAPLE"
    alias.symlink_to(REPO)
    return {
        "ancestor_alias": (REPO, alias / "src" / "cw_e9_guard_new_target", REPO / "src"),
        "nested_repository_argument": (REPO / "src", REPO / "tests" / "cw_e9_guard_new_target", REPO / "tests"),
    }


@needs_repo
@pytest.mark.parametrize("case", ["ancestor_alias", "nested_repository_argument"])
def test_old_tool_reached_staging_inside_actual_source_and_fixed_tool_refuses_before_it(tmp_path, monkeypatch, case):
    before_state, before_listing = repo_state(), _source_listing()
    repo_arg, dest, actual_parent = _bypass_cases(tmp_path)[case]
    old = _previous_tool()
    staged = []
    with monkeypatch.context() as m:
        _intercept_staging(m, staged)
        with pytest.raises(Intercepted):  # negative control: the correction-1 guard let it through to staging
            # CW-E11 / CW-E16: the archived tool's built-in pin is the historical M44 manifest; it is given the
            # CURRENT (24c6) pin explicitly so the control exercises its (old) source guard on this recipe, not its
            # stale pin refusal
            old.reconstruct(repo_arg, dest, recipe_dir=tool.RECIPE_DIR,
                            expected_manifest_sha256=tool.EXPECTED_MANIFEST_SHA256)
    assert [os.path.realpath(p) for p in staged] == [str(actual_parent)]
    staged.clear()
    with monkeypatch.context() as m:
        _intercept_staging(m, staged)
        with pytest.raises(tool.RecipeError, match="inside the actual MAPLE working tree|inside the MAPLE repository"):
            tool.reconstruct(repo_arg, dest)
    assert staged == []  # refused before staging, archive and apply
    assert repo_state() == before_state and _source_listing() == before_listing
    assert not os.path.lexists(dest)


@needs_repo
def test_symlinked_repository_argument_and_git_metadata_or_bare_arguments_are_refused(tmp_path, monkeypatch):
    staged = []
    _intercept_staging(monkeypatch, staged)
    before = repo_state()
    link = tmp_path / "repo_link"
    link.symlink_to(REPO)
    with pytest.raises(tool.RecipeError, match="inside"):  # repository given through a symlink, dest in the real tree
        tool.reconstruct(link, REPO / "cw_e9_guard_new_target")
    with pytest.raises(tool.RecipeError, match="inside"):  # both through aliases
        tool.reconstruct(link, link / "src" / "cw_e9_guard_new_target")
    with pytest.raises(tool.RecipeError, match="not inside a Git working tree|bare"):  # git metadata argument
        tool.reconstruct(REPO / ".git", tmp_path / "out_metadata")
    bare = tmp_path / "bare.git"  # a scratch bare repository (test fixture only, never MAPLE)
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True, capture_output=True)
    with pytest.raises(tool.RecipeError, match="bare"):
        tool.reconstruct(bare, tmp_path / "out_bare")
    assert staged == [] and repo_state() == before
    assert not any(os.path.lexists(tmp_path / n) for n in ("out_metadata", "out_bare"))


@needs_repo
def test_nested_repository_argument_and_aliased_fresh_destination_still_reconstruct_exactly(tmp_path):
    before = repo_state()
    real_out = tmp_path / "real_out"
    real_out.mkdir()
    alias = tmp_path / "out_alias"
    alias.symlink_to(real_out)
    receipt = tool.reconstruct(REPO / "src", alias / "runtime")
    published = real_out / "runtime"  # the canonical location the guard checked
    assert receipt["destination"] == str(published) and published.is_dir() and not published.is_symlink()
    files = {p.relative_to(published).as_posix(): p for p in published.rglob("*") if p.is_file()}
    assert set(files) == set(MANIFEST["post_patch"]["files"]) | {tool.RECEIPT_NAME}
    assert sum(1 for k in files if k != tool.RECEIPT_NAME) == 341
    for rel, digest in MANIFEST["post_patch"]["files"].items():
        assert sha(files[rel].read_bytes()) == digest, rel
    assert repo_state() == before and leftovers(real_out) == [] and leftovers(tmp_path) == []


@needs_repo
def test_protected_roots_are_the_actual_canonical_working_tree_and_git_dirs(tmp_path):
    top, roots = tool.protected_source_roots("git", REPO / "src" / "maple")
    assert top == Path(os.path.realpath(REPO))
    assert roots[0] == os.path.realpath(REPO) and all(os.path.isdir(r) for r in roots)
    assert os.path.realpath(REPO / ".git") in roots
    link = tmp_path / "repo_link"
    link.symlink_to(REPO / "src")
    assert tool.protected_source_roots("git", link)[0] == top
