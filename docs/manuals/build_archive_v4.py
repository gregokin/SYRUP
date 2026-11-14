"""Final companion archive builder (v4 metadata, Phase 2.5 final edition): the v4-compatible successor of the historical `docs_publication_tools/build_archive.py`. Run from `docs/manuals` AFTER `build_manuals.py --final`:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/syrup-manuals-citation-tools-20261020 <MAPLE>/.venv/bin/python -B build_archive_v4.py

It refuses unless the metadata is final (`load_metadata(require_final=True)`), runs the SAME checks as `verify_archive_v4.py --mode final` for the two Word documents, the fenced blocks, the examples, the historical guards and the Phase 2.5 cross checks,
and then writes, in this order: `documentation_verification.json` (no wall-clock time: the record is deterministic), `archive_manifest.json` (a file table that does not list the ZIP or itself) and the deterministic ZIP
`SYRUP_Manuals_and_Examples.zip` (fixed entry times and attributes, sorted names). No circular hash exists: the ZIP and the manifest are outside their own file table and the verification record holds only the Word documents' hashes.
`verify_archive_v4.py --mode final` re-checks the written archive. Static checks only: no paginated Office rendering."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_archive_v4 as v4  # noqa: E402

ZIP_TIME = (2026, 11, 14, 12, 0, 0)


def main() -> int:
    bm = v4.load_builder()
    meta = bm.load_metadata(require_final=True)
    if "native_phase25" not in meta:
        v4.fail("this builder needs schema v4 metadata")
    docs = [v4.docx_checks(n, meta, "final") for n in ("SYRUP_CLI_User_Guide.docx", "SYRUP_Model_Structure_and_Physics.docx")]
    blocks, ex = v4.fenced_and_examples()
    hist = v4.historical_guards(meta)
    cross = v4.final_cross_checks(meta)
    record = {"release_status": meta["release_status"], "schema": meta["schema"], "prepared_date": meta["prepared_date"], "documents": docs, "fenced_blocks_checked": len(blocks), "fenced_blocks_failed": 0,
              "example_python_scripts_syntax_checked": ex, "historical_guards": hist, "cross_checks": cross,
              "native_phase15_identity": dict(meta["native_phase15"]), "native_phase2_identity": dict(meta["native_phase2"]), "native_phase25_identity": dict(meta["native_phase25"]),
              "limitations": ["Static DOCX ZIP/XML/relationship/token/section/identity checks only; no paginated Office rendering was available and none is claimed",
                              "Example commands in the shell blocks are syntax-checked, not executed; the native Phase 2.5 example script was run on CPU on the exact accepted sources (expected_result_cpu_run.json); the earlier editions' examples keep their own records",
                              "Phase 2.5 claims are those of qualification/phase25_qualification.json (bounded; no field validation, catchment-scale fit, scheduled three-model run or general repeated-storm recovery)"],
              "exact_metadata_fixture": "genuine native9f distribution metadata files byte-identical to the base commit bytes; unchanged by this build"}
    (HERE / "documentation_verification.json").write_text(json.dumps(record, indent=2) + "\n")
    old_manifest = {}
    for key in ("native_phase15", "native_phase2"):
        old = json.loads(v4.base_blob("docs/manuals/archive_manifest.json"))[key]          # the historical block (notes, evidence lists) from the base commit
        if any(old[k] != meta[key][k] for k in old if k in meta[key]):
            v4.fail(f"the historical archive block {key} disagrees with the metadata")
        old_manifest[key] = {**old, **meta[key]}
    files = v4.archive_files()
    table = {f: {"sha256": v4.sha((HERE / f).read_bytes()), "bytes": (HERE / f).stat().st_size} for f in files}
    manifest = {"scope": "Documentation, exact dependency metadata and public configuration examples (C1 scheduled examples; the native Phase 1.5 N15 and Phase 2 records; the native Phase 2.5 qualification receipt, publication record, source manifest, "
                         "exact-code source capsule and runnable CPU example); no model source tree, third-party environment or deep bed beyond the exact-code capsule",
                "files": table, **old_manifest,
                "native_phase25": {**meta["native_phase25"], "note": "accepted at a bounded scope by qualification/phase25_qualification.json; the four component package hashes are historical intermediate compositions, not the current model",
                                   "shipped_portable_evidence": ["qualification/phase25_qualification.json", "qualification/phase25_qualification.md", "qualification/phase25_publication.json", "qualification/phase25_source_manifest.json",
                                                                 "qualification/phase25_source_capsule.zip", "qualification/phase25_wind_compatibility.md", "qualification/phase25_vegetation.md"],
                                   "main_integration": None}}
    mbytes = (json.dumps(manifest, indent=2) + "\n").encode()
    (HERE / "archive_manifest.json").write_bytes(mbytes)
    with zipfile.ZipFile(HERE / "SYRUP_Manuals_and_Examples.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in files + ["archive_manifest.json"]:
            zi = zipfile.ZipInfo(f, date_time=ZIP_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, mbytes if f == "archive_manifest.json" else (HERE / f).read_bytes())
    out = v4.archive_checks(meta)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
