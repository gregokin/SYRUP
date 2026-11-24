"""Exact committed MAPLEc4b9343 TERRAIN source descriptor (irregular imported elevation / fixed base / domain).

Pins come from the current_triplet_irregular_20261123 bundle receipt (`evidence/pins.json`,
`masy.current_triplet.maple_source_bundle.v1`) and its genuine bundle manifest; the field census comes from
`evidence/maple_field_census.json`, the actual dataclass census of the committed c4b9343 source. Distribution metadata is
verified reuse of the prior genuine materialization with a byte-identical static pyproject; no new wheel build is
claimed. This descriptor grants no device, windowed, model-capacity, water-coupling or old-checkpoint migration
capability; it is selected ONLY by the separately named terrain profiles below. The 9f and 1dca descriptors are
untouched.
"""
from types import MappingProxyType

#: The separately named TERRAIN admission profiles (never a default; the 9f/1dca names and defaults are unchanged).
PROFILE_NAME = "native_maple_terrain_20261123_case_latitude_cpu_v1"
CUDA_PROFILE_NAME = "native_maple_terrain_20261123_case_latitude_cuda_v1"
DENSE_PROFILE_NAME = "native_maple_terrain_20261123_dense_cpu_v1"
DENSE_CUDA_PROFILE_NAME = "native_maple_terrain_20261123_dense_cuda_v1"
#: Actual compiled/resolved dataclass field census of the committed c4b9343 source (51 / 25 fields). A different tuple is a
#: different native API and is refused before any compile or load.
COMPILED_CASE_FIELDS = (
    "case_path", "processed_path", "diagnostics_path", "provenance_path", "case_name", "code_version", "config",
    "topography", "grain_mixture", "coordinates", "voxel_column", "active_layer", "committed_topography",
    "sediment_ledger", "vegetation", "sediment_availability_spec", "sediment_availability", "water_initial_depth_spec",
    "water", "threshold_modifier_spec", "threshold_modifier_state", "stratigraphy", "initial_mobile_seed_spec",
    "initial_mobile_state", "case_import", "import_sources", "masks", "transform_logs", "masks_path", "topography_result",
    "expected_total_mass_by_class_kg", "expected_mass_by_cell_class_kg", "expected_mass_by_level_class_kg",
    "provenance_record", "compile_seconds", "location", "regional_slope", "regional_slope_result", "roughness_ownership",
    "fixed_base_spec", "fixed_base", "fixed_base_source", "fixed_base_admission", "fixed_base_load_admission",
    "leaf_vegetation", "domain_spec", "domain", "domain_admission", "domain_load_admission")
RESOLVED_PACKAGE_FIELDS = (
    "config", "topography", "grain_mixture", "stratigraphy", "threshold_modifiers", "vegetation", "sediment_availability",
    "initial_mobile_seed", "water_initial_depth", "prescribed_external_shear", "requested_diagnostics", "movie_request",
    "wind_record", "case_import", "import_sources", "case_yaml_path", "model_yaml_path", "wind_yaml_path", "location",
    "regional_slope", "roughness_ownership", "slope_yaml_path", "slope_yaml_sha256", "fixed_base", "domain")
#: Informational only (bundle_id also hashes the exporting tool/path/python and is RECORDED, never a gate).
REFERENCE_BUNDLE_ID = "a8131af38a8619505fa1007d67093f103b92621362c887ba71b7b9963afb52a6"
REFERENCE_MANIFEST_SHA256 = "10e423161a64f40c84f6e0bf99f3a05a187b57a09cca2e3c3d2b8a39ea44cbb8"
PINS = MappingProxyType(dict(
    commit="c4b9343706240d3f98e9abb9c7274958a45438cc",
    package_digest="f4e4dc694dff1bbf126ddbb69a8cd0bc089d8006ebb5f4ed2ce1ac30e8e5b8bc",
    project_digest="b1ed830553d683feb4f3bdaa00d8c49227b2959382f1d776fbe77da812675a66",
    archive_sha256="b1053d028f8dfc5fb568e6502da024273e4719a927b9da0076abf2667d874864",
    pyproject_sha256="8779c55d823bf610da9c9f087c7d5eaaf8d276bfbdc6fb428726e4e7c06350be",
    distribution_sha256=MappingProxyType({
        "METADATA": "dc1b5653aab75868578b7d478b3d015b146ba1fe4ea4907cc26d806d68120cca",
        "WHEEL": "b59a1e1a3b56c5645f769944ec4ddde3954f94b3509ef6ca8989f1ee0c00cbc0",
        "top_level.txt": "08ab2a56c8503c742c762013475ad2955377d2001a9fc03e4e8bc22e09af0844",
        "entry_points.txt": "9430d41e28e96a53fa455b84c6a4b6122484d21e6ce8151c6c40854db190c870"}),
    package_file_count=424, compiled_fields=COMPILED_CASE_FIELDS, resolved_fields=RESOLVED_PACKAGE_FIELDS))
