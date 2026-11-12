"""The immutable historical CUDA V2 wrong-version control (static, CPU, NO device, NO CuPy): a byte-for-byte copy of a GENUINE C1-produced native9f CUDA V2 lifecycle checkpoint
(`fixtures/c1_cuda_v2/`, provenance outside the strict directory in `fixtures/c1_cuda_v2_PROVENANCE.json`). The tests verify its SHA-256/size/layout/sealed-manifest facts against the provenance
receipt, that its recorded sources are the historical C1 ones (not the new candidate sources), and that the V3 codec's closed policy/naming fences cannot accept it. They never alter, reseal or
rename it, never repin V2 and never fabricate an old archive."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from maple_syrup import lifecycle_checkpoint_cuda as cs
from maple_syrup import lifecycle_checkpoint_cuda_load as ld
from maple_syrup import lifecycle_checkpoint_cuda_policy as pol_mod
from maple_syrup import pond_checkpoint as pc

FIX = Path(__file__).resolve().parent / "fixtures"
DIR = FIX / "c1_cuda_v2"
PROV = json.loads((FIX / "c1_cuda_v2_PROVENANCE.json").read_text())


def test_fixture_directory_is_exactly_the_three_historical_files_with_the_recorded_hashes():
    assert sorted(p.name for p in DIR.iterdir()) == sorted(PROV["directory_files_exactly"]) and len(PROV["directory_files_exactly"]) == 3  # nothing extra: strict loaders refuse extras
    for name, meta in PROV["directory_files_exactly"].items():
        data = (DIR / name).read_bytes()
        assert len(data) == meta["bytes"] and hashlib.sha256(data).hexdigest() == meta["sha256"], name
    assert "c1_cuda_v2" in str(DIR) and (FIX / "README.md").exists() and (FIX / "c1_cuda_v2_PROVENANCE.json").exists()  # receipt and README live OUTSIDE the strict directory


def test_manifest_is_a_genuine_sealed_c1_v2_boundary_record_from_the_historical_sources():
    data = json.loads((DIR / pol_mod.V2.manifest_name).read_text())
    facts = PROV["manifest_facts_read_from_the_file"]
    assert (data["schema"], data["profile"], data["role"]) == (facts["schema"], facts["profile"], facts["role"]) == (pol_mod.V2.schema, pol_mod.V2.profile, "boundary")
    assert data["manifest_sha256"] == pc._seal_manifest(data)["manifest_sha256"]  # untouched, sealed by its producer
    assert data["source"] == facts["source"]
    assert data["source"]["maple_package_sha256"] == cs.NATIVE9F_PACKAGE_SHA256  # MAPLE 9f: the same package pin the new codec uses
    assert data["source"]["maple_syrup_tree_sha256"] == "afcad53691b9d295ed560116d313419c08fc5ffe9b227a4bdd9c922f93c52671" and data["source"]["maple_syrup_n_files"] == 109
    assert set(data) == pol_mod.V2.manifest_keys and "daily" not in data["payload"]["scalars"]  # a cursor-free V2 record
    assert set(data["payload"]["scalars"]) == pol_mod.V2.scalar_parts[("boundary", None)]


def test_the_v3_codec_cannot_accept_the_genuine_v2_archive_by_naming_policy_or_helper_identity():
    data = json.loads((DIR / pol_mod.V2.manifest_name).read_text())
    assert data["schema"] != pol_mod.V3.schema and data["profile"] != pol_mod.V3.profile and pol_mod.V2.manifest_name != pol_mod.V3.manifest_name
    dirfd = os.open(DIR, os.O_RDONLY | os.O_DIRECTORY)
    try:
        ld._check_directory_names(dirfd, pol_mod.V2)  # the archive IS a well-formed V2 directory under V2's own closed names ...
        with pytest.raises(cs.LifecycleCheckpointCudaError):
            ld._check_directory_names(dirfd, pol_mod.V3)  # ... and V3's closed name set refuses it before any manifest or payload is read
    finally:
        os.close(dirfd)


def test_v2_helper_pin_still_refuses_the_new_sources_so_a_v2_archive_can_never_be_produced_by_them():
    """The C1 `lifecycle_checkpoint.py` pin is NOT repinned: under the new sources V1/V2 refuse before any archive could be published (the fixture above is therefore the only V2 archive)."""
    for old in (pol_mod.V1, pol_mod.V2):
        with pytest.raises(cs.LifecycleCheckpointCudaError, match="differs from its pinned digest"):
            cs._fresh_source_pair(old)
    assert cs.HELPER_MODULE_PINS["lifecycle_checkpoint.py"] == "ff4d228e9464a29333a544e3ca098577df492d02661b9faaa7c7278c98e6d085"
