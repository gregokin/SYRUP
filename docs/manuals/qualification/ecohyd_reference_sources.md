# EcoHyD reference sources

Downloaded references are under `/home/okin/SYRUP/reference_models/ecohyd/`.
That whole directory is excluded by `.gitignore`; its source is for inspection,
not a SYRUP runtime dependency. The directory's `README.md`, metadata, original
archives and manifests record actual download status and identities.

The complete Baldauf, Cantón & Tietjen (2023) repository is downloaded at commit
`80b10e6b6b82ad16e9fd41a31901cc3157d880e9`: 277 files, 50,290,949 extracted bytes.
Every file matches its GitHub blob identity and the preserved ZIP; all archive
CRCs pass. The original GPL-3.0 license, parameters, weather, analysis and data
are preserved. No downloaded script, executable or model was run.

- Source: https://github.com/selinaZitrone/Baldauf_et_al_Biocrust_Ecohydrology
- Local code: `reference_models/ecohyd/github_baldauf_2023/source/model/src/`
- Important files: `WaterCell.cpp`, `WaterLandscape.cpp`, `Weather.cpp`.

This is the later biocrust extension of EcoHyD, in the HydroVeg/Tietjen model
lineage. It is not established as the original 2009 HydroVeg implementation,
and has not been built, benchmarked or adopted into SYRUP.

Both additional full Dryad packages and their separate original READMEs were
downloaded on 2026-11-09 using the local credentials file explicitly supplied
by the user. Exact publisher sizes and SHA256 digests pass for all four assets.
Tokens remain in memory, are not written to receipts, and are stripped from
redirects to external download hosts. Earlier unauthenticated 401/403 attempts
and their partial-task receipts remain historical evidence.

| Dataset | Version | Main archive | Bytes | Download status |
|---|---:|---|---:|---|
| https://doi.org/10.5061/dryad.4j0zpc8gn | 212594 | `Irob_et_al_drought_resilience.zip` | 594,941,331 | Downloaded and extracted: 444 files, 2,920,360,733 bytes |
| https://doi.org/10.5061/dryad.ngf1vhj1t | 270102 | `Navigating_Uncertainty_Irob_et_al.zip` | 806,225,003 | Downloaded and extracted: 607 files, 2,862,429,170 bytes |

The extracted C++ source is available at:

- `reference_models/ecohyd/dryad_4j0zpc8gn/extracted/Irob_et_al_drought_resilience/Model/Source/`
- `reference_models/ecohyd/dryad_ngf1vhj1t/extracted/Navigating_Uncertainty_Irob_et_al_23/Source/EcoHyD/`

These are later EcoHyD versions, not verified original 2009 HydroVeg releases.
Original archives, dataset/version/file metadata, publisher license declarations,
READMEs and full per-file extraction manifests are retained. The original
`__MACOSX` resource-fork entries are preserved and counted as archive files;
they must not be mistaken for additional model source. No separate LICENSE
file was found inside either Dryad ZIP; the pinned dataset metadata declares
CC0-1.0, while the GitHub snapshot carries GPL-3.0. The prebuilt `ORYCS`
executable and compile/launch scripts shipped in the 2023 package are preserved
but were extracted without execute permission. No downloaded code was compiled,
imported or run, and no version was adopted into SYRUP.

Initial task evidence: `agent_handoffs/tasks/ecohyd_reference_download_20261109/`.
Authenticated completion and independent integrity evidence:
`agent_handoffs/tasks/ecohyd_dryad_completion_20261109/`.
