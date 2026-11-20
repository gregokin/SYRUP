# JORN field-capacity worked audit

Accepted 20 November 2026. See report/SYRUP_JORN_field_capacity_worked_audit_20261120.docx and report/report.md. This is a scientific input/arithmetic audit, not new hydrology results or a parameter/default change. Actual three-layer texture, raw-density mixing and all calculation steps are supplied.

The archived sandy inputs produce about10–11vol%water at33kPa, but about40vol%when the linear Saxton–Rawls wet branch is evaluated at6kPa, following a Tietjen pressure convention. The arithmetic matches the printed equations; this hybrid is not validated site field capacity. No corrected physical FC is adopted. Preserve old results and conditional calibration provenance.

Root and Claude independently recomputed the inputs and equations at80 and60decimaldigits; evidence and limitations are in root_final_acceptance.json. No new model/GPU/restart tests or Word page rendering are claimed.

Rebuilding report artifacts requires the separately archived local raw sources and task inputs identified by SHA256, plus the existing MAPLE Python environment and task-local report_tools. The builder documents JORN_FC_TASK and JORN_FC_ROOT; no downloaded reference PDF or credentials are included here. Every published artifact is sealed in report/artifact_manifest.json.
