# Native LEAF + water calendar run — example (`masy.native_water_run.v2`)

DEVELOPMENT candidate; unqualified beyond the recorded tests. Nothing here depends on ignored `agent_handoffs` paths: every absolute path is a placeholder filled by `fill_config.py`.

* `case/`, `layered_water.yaml`, `scenario/` — a small MAPLE case (6x4 cells, 0.25 m), a 2-layer SYRUP water file and a scenario (interevent / rain / interevent, T on, E on, growth on).
* `catalog.json` — LEAF 8da species catalog (2 layers); `plants.csv` + `plants.metadata.json` — one plant and its REQUIRED companion (written by LEAF's `plant_files_text`; never hand-edit one without the other).
* `run.template.yaml` — the closed v2 configuration; `fill_config.py` — copies the inputs outside any source tree and fills the paths.

    python fill_config.py /tmp/leafrun --masy-root <sealed masy tree> --syrup-root <sealed SYRUP 109 water root> --leaf-root <clean LEAF 8da capsule> --native-target <native9f bundle>
    cd /tmp/leafrun
    PYTHONPATH=<masy>/src <MAPLE .venv python> -B -m masy.native_water_run manifest <masy> masy_manifest.json
    PYTHONPATH=<masy>/src <MAPLE .venv python> -B -m masy.native_water_run manifest <syrup_root> syrup_manifest.json
    PYTHONPATH=<masy>/src <MAPLE .venv python> -B -m masy.native_water_run run run.yaml
    # pause after one occurrence / resume: add `--pause-after-occurrences 1`, then `run run2.yaml --resume <run_dir>/combined_checkpoint --expected-seal-sha256 <seal>` with a NEW run_dir

The `masy_root` and `syrup_root` must be read-only sealed copies and the manifests must describe them exactly (the same rules as `examples/native_layered_calendar`). The example grid, units and catalog
are dimensionally small test values, not calibrated parameters.
