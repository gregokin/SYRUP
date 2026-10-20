# SYRUP user and scientific manuals

These materials describe committed SYRUP **d05d8c6c66d9fd2969c007681916254e8d1c2c7c**
with the qualified native MAPLE **39bcf79e1e67b6ed960f78a923ddfedf1d173e68**.
They do not silently adopt a newer MAPLE checkout.

- [CLI User Guide](SYRUP_CLI_User_Guide.docx): installation from GitHub,
  declared dependencies, CPU/GPU choices, imported cases, run and restart
  workflows, directory/output structure, maps/movies and exhaustive CLI tables.
- [Model Structure and Physics](SYRUP_Model_Structure_and_Physics.docx):
  journal-style explanation, governing equations, empirical coefficients,
  boundaries, native bed authority, numerical choices and a reproduction appendix.
- [Companion archive](SYRUP_Manuals_and_Examples.zip): both Word documents,
  their editable sources and the plotting example. Unpack its contents into
  the `$WORK/manuals` directory used by the guide. The materials are new;
  they are not included in the model's pinned d05d8c6 commit.

The practical reference covers all nine installed commands and every other
module in `src/maple_syrup` that exposes an argparse CLI: experimental hydraulic
runs, RFID/Chastre import, extended CPU/GPU legacy replay, dependency delivery
and the maintainer-only initial source-transition utility. Unreviewed initial
transitions cannot be imported, and wet checkpoint source guards remain strict.
The source reference is [cli_reference.json](cli_reference.json); the readable
tables are [cli_reference.md](cli_reference.md).

The guide and theoretical article have separate editable Markdown sources:
[user_guide.md](user_guide.md) and
[model_structure_and_physics.md](model_structure_and_physics.md).
The Word guide includes the CLI-reference appendix as well as its main source.

## Companion output example

`render_outputs.py` requires NumPy and Matplotlib. It reads actual NPZ arrays,
renders cell maps/hydrographs, and makes PNG frames from experimental or legacy
snapshot archives. It does not create missing spatial history, run physics,
decode checkpoints or access a GPU. FFmpeg is used separately to encode videos.
Its format, orientation, time ordering, class selection and failure behavior
were checked on small synthetic files, including a real CPU FFmpeg encode.

## Rebuild the Word files

Document generation additionally needs `python-docx==1.2.0` and Matplotlib:

```bash
python build_manuals.py
```

The builder creates editable Word headings, linked contents, repeating table
headers, equation objects, page fields and an embedded schematic. The generated
packages are reopened and checked structurally; native Word/LibreOffice page
rendering was not available on the authoring machine. Word will paginate the
files and update page fields on opening. No rendered page count is claimed.

## Verification scope

Codex checked the committed sources, all CLI help/parser actions, source
identities and example syntax. Claude authored the practical draft and reviewed
the theoretical description and plotting helper; Codex independently reviewed
and reconciled the findings. Source and review records remain in
`agent_handoffs/tasks/manuals_20261020/` locally. `documentation_verification.json`
records the distributable checks. The physics document also has 25 scientific
references cited beside the relevant relationships and numerical methods, with
clickable DOI links and explicit distinctions between published theory,
MAHLERAN source conventions and new SYRUP choices. Its additional citation
audit and Claude review records are in
`agent_handoffs/tasks/manual_citations_20261020/` locally.

No new storm simulation, GPU computation, model-code change, environment
installation, commit or push is part of this documentation task. New document
generation dependencies were confined to a temporary tool directory. The
existing model's scientific and performance qualification is unchanged.
