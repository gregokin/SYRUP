"""Exact committed MAPLE1dca Dense NumPy source descriptor.

Pins come from Root_bundle_receipt_02.json and the verified materialized
bundle. Distribution metadata is verified reuse with byte-identical static
pyproject; no new wheel build is claimed. This descriptor grants no device,
windowed, model-capacity or old-checkpoint migration capability.
"""
from types import MappingProxyType

PROFILE_NAME = "native_maple_1dca_case_latitude_cpu_v1"
DENSE_PROFILE_NAME = "native_maple_1dca_dense_cpu_v1"
COMPILED_CASE_FIELDS = (
    "case_path", "processed_path", "diagnostics_path", "provenance_path", "case_name", "code_version", "config",
    "topography", "grain_mixture", "coordinates", "voxel_column", "active_layer", "committed_topography",
    "sediment_ledger", "vegetation", "sediment_availability_spec", "sediment_availability", "water_initial_depth_spec",
    "water", "threshold_modifier_spec", "threshold_modifier_state", "stratigraphy", "initial_mobile_seed_spec",
    "initial_mobile_state", "case_import", "import_sources", "masks", "transform_logs", "masks_path", "topography_result",
    "expected_total_mass_by_class_kg", "expected_mass_by_cell_class_kg", "expected_mass_by_level_class_kg",
    "provenance_record", "compile_seconds", "location", "regional_slope", "regional_slope_result", "roughness_ownership",
    "fixed_base_spec", "fixed_base", "fixed_base_source", "fixed_base_admission", "fixed_base_load_admission",
    "leaf_vegetation")
RESOLVED_PACKAGE_FIELDS = (
    "config", "topography", "grain_mixture", "stratigraphy", "threshold_modifiers", "vegetation", "sediment_availability",
    "initial_mobile_seed", "water_initial_depth", "prescribed_external_shear", "requested_diagnostics", "movie_request",
    "wind_record", "case_import", "import_sources", "case_yaml_path", "model_yaml_path", "wind_yaml_path", "location",
    "regional_slope", "roughness_ownership", "slope_yaml_path", "slope_yaml_sha256", "fixed_base")
PINS = MappingProxyType(dict(
    commit="1dca8539b87bf80e2b4bb72d4c6637d02013a305",
    package_digest="191416153d3eca03e0b5e0817720d1365055de5b0b37391ac95454321aee7d68",
    project_digest="8ce90b733f272879060ce12047e31bcd98b64bb4cfd9d3ce0a90ebd5d5336e67",
    archive_sha256="48ba131addd3c5b7b57390a98cde4005d7bf127a1cd51abd74b591cec44ed9fa",
    pyproject_sha256="8779c55d823bf610da9c9f087c7d5eaaf8d276bfbdc6fb428726e4e7c06350be",
    distribution_sha256=MappingProxyType({
        "METADATA": "dc1b5653aab75868578b7d478b3d015b146ba1fe4ea4907cc26d806d68120cca",
        "WHEEL": "b59a1e1a3b56c5645f769944ec4ddde3954f94b3509ef6ca8989f1ee0c00cbc0",
        "top_level.txt": "08ab2a56c8503c742c762013475ad2955377d2001a9fc03e4e8bc22e09af0844",
        "entry_points.txt": "9430d41e28e96a53fa455b84c6a4b6122484d21e6ce8151c6c40854db190c870"}),
    package_file_count=414, compiled_fields=COMPILED_CASE_FIELDS, resolved_fields=RESOLVED_PACKAGE_FIELDS))
