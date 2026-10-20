# MAPLE-SYRUP Practical User Guide (SYRUP `d05d8c6`, MAPLE `39bcf79`)

This guide is for a new scientific collaborator. It shows how to install the exact pinned sources, import the corrected MAHLERAN Plot 1 case, run the conservative water–sediment event on CPU and GPU, pause and resume, and read the outputs. Each section connects commands with their intended workflow and outputs.

Pinned sources used throughout:

| Component | Repository | Exact commit |
|---|---|---|
| MAPLE (wind/bed model, dependency) | `https://github.com/gregokin/MAPLE.git` | `39bcf79e1e67b6ed960f78a923ddfedf1d173e68` |
| SYRUP (`maple-syrup`, this package) | `https://github.com/gregokin/SYRUP.git` | `d05d8c6c66d9fd2969c007681916254e8d1c2c7c` |
| MAPLE package digest the storage guards accept | — | `af54626ab432670539ed71480fd884f13581dfdfcfec63d7c85053ca215adfec` |

Both projects need Python ≥ 3.11. The qualified interpreter was CPython 3.12.3 with NumPy 2.5.3, Numba 0.67.0 / llvmlite 0.49.0 and, for GPU, CuPy 14.2.0 with CUDA 12.9 toolkit components.

An auto-generated reference appendix covering every flag, default and choice is appended after this guide. The guide itself explains how the important controls fit together and which combinations are refused.

**Companion materials.** This guide, its source Markdown and the companion plotting example `render_outputs.py` are distributed together as `SYRUP_Manuals_and_Examples.zip`. They are **not** part of the pinned SYRUP commit `d05d8c6`, so a fresh clone does not contain them. Unpack the archive to `$WORK/manuals` (Section 1.1); every plotting command below runs the helper from there.

---

## 1. Install MAPLE and SYRUP

### 1.0 System prerequisites

The commands below use Bash on Linux or WSL2. Run them in that environment; native PowerShell syntax and paths differ.

You need:

- Git;
- Python ≥ 3.11 (3.12 qualified) with the `venv` module and pip;
- optionally, FFmpeg, only to turn PNG frames into a video;
- for GPU runs only, an NVIDIA driver supporting CUDA 12.x, installed on the host (on WSL2, the Windows driver). The driver is not a pip package.

On Ubuntu 24.04, whose system Python is 3.12, the system packages are typically:

```bash
sudo apt update
sudo apt install git python3.12 python3.12-venv python3-pip ffmpeg   # ffmpeg optional
```

These commands are an example of what to install, not a required recipe. Everything else (the MAPLE base dependencies, Numba, CuPy, `packaging`, Matplotlib) is installed into the virtual environments with pip below.

### 1.1 Choose absolute paths once

Every later command uses these variables. Pick a directory you own; nothing below writes outside it except pip's own caches.

```bash
export WORK=$HOME/syrup-work                 # any absolute path you own
export MAPLE=$WORK/MAPLE-39bcf79             # NEW, separate MAPLE checkout
export SYRUP=$WORK/SYRUP-d05d8c6             # SYRUP checkout
export MAHLERAN=$WORK/MAHLERAN               # your copy of the MAHLERAN inputs (Section 3)
export RUNS=$WORK/runs                       # cases, outputs, checkpoints, stores
export MANUALS=$WORK/manuals                 # unpacked SYRUP_Manuals_and_Examples.zip
export BIN=$MAPLE/.venv/bin                  # CPU environment
export GBIN=$MAPLE/.venv-cupy/bin            # optional GPU environment
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$WORK" "$RUNS"
```

Keep `$MAPLE` and `$SYRUP` outside `$RUNS`. The runners refuse to write into the MAPLE tree, the MAHLERAN tree, the recipe directory or the SYRUP package.

### 1.2 Clone MAPLE at the exact commit

Use a **new** clone. If you already have a MAPLE checkout that you use for other work, do not repoint it: shared checkouts move forward, and SYRUP binds runs to the exact MAPLE source digest.

```bash
cd "$WORK"
git clone https://github.com/gregokin/MAPLE.git "$MAPLE"
cd "$MAPLE"
git checkout --detach 39bcf79e1e67b6ed960f78a923ddfedf1d173e68
git rev-parse HEAD        # must print 39bcf79e1e67b6ed960f78a923ddfedf1d173e68
git status --porcelain    # should print nothing before you install
```

### 1.3 Create the CPU environment inside the MAPLE clone

```bash
cd "$MAPLE"
python3.12 -m venv .venv
"$BIN/python" -m pip install --upgrade pip
"$BIN/python" -m pip install -e "$MAPLE[windowed]"
```

This installs MAPLE as an **editable** distribution named `maple`, plus its base dependencies from MAPLE's `pyproject.toml`: `jax`, `jaxlib`, `numpy`, `pyyaml`, `xarray>=2024.10`, `zarr>=3.0`, `dask[array]` and `tifffile`. The `windowed` extra adds `numba>=0.67,<0.68`. Without Numba, MAPLE's exact CPU height decode in the resident-window bed refuses with `PrefixPacketError: decode_height on NumPy arrays needs the compiled Numba kernel ...`.

**Never run `pip install maple` from PyPI.** The PyPI project named `maple` is unrelated. MAPLE is not published, so SYRUP deliberately does not declare it as a dependency. Instead, at runtime SYRUP checks which `maple` was imported (`maple_syrup.dependency`) and refuses anything it cannot identify.

### 1.4 Optional: create the GPU environment

```bash
cd "$MAPLE"
python3.12 -m venv .venv-cupy
"$GBIN/python" -m pip install --upgrade pip
"$GBIN/python" -m pip install -e "$MAPLE[windowed,gpu]"
```

The `gpu` extra is `cupy-cuda12x[ctk]>=14.0,<15`. It is **CuPy, not GPU JAX**. The `[ctk]` part pulls in the CUDA 12 toolkit headers that CuPy needs to JIT-compile kernels. Without them, the first kernel launch fails with `Failed to find CUDA headers`. CUDA 12 was chosen because it still supports Pascal (SM 6.1, e.g. GTX 1080 Ti); CUDA 13 does not. The qualified point is CuPy 14.2.0 with CUDA 12.9 components. Other versions in the range are allowed but untested.

The NVIDIA **driver** is not a pip package. Install it separately (on WSL2 that is the Windows host driver). Check with `nvidia-smi` that it supports CUDA 12.x.

### 1.5 Clone SYRUP and install it into the same environment(s)

```bash
cd "$WORK"
git clone https://github.com/gregokin/SYRUP.git "$SYRUP"
cd "$SYRUP"
git checkout --detach d05d8c6c66d9fd2969c007681916254e8d1c2c7c
git rev-parse HEAD        # must print d05d8c6c66d9fd2969c007681916254e8d1c2c7c

"$BIN/python" -m pip install -e "$SYRUP[numba,dependency-delivery]"
"$BIN/python" -m pip install matplotlib          # only for plotting outputs
# GPU environment, if created:
"$GBIN/python" -m pip install -e "$SYRUP[numba,dependency-delivery]"
"$GBIN/python" -m pip install matplotlib
```

`numba` turns on the compiled CPU kernels; nothing falls back silently if they are missing. `dependency-delivery` (`packaging>=23`) is used only by the `maple_syrup.dependency_bundle` helper.

Matplotlib is not a base dependency of either package. MAPLE lists it only in its optional `viz` extra, together with Panel and hvPlot for MAPLE's own interactive viewer. You do **not** need `MAPLE[viz]` to plot SYRUP's NPZ outputs; installing Matplotlib alone is enough for the companion plotting example.

Finally, tell SYRUP which MAPLE it must find:

```bash
export MAPLE_SYRUP_EXPECTED_MAPLE_ROOT=$MAPLE
```

With this set, every resolver call refuses any other MAPLE source root. The plot import, dense event and most other runners also accept `--expected-maple-root`. The windowed runner reads **only** the environment variable.

### 1.6 Using an existing machine's shared environments instead

On a machine whose shared MAPLE environments (for example `/home/okin/MAPLE/.venv` and `.venv-cupy`) already contain the third-party packages, you can reuse them without creating new venvs. However, their editable `maple` points at that machine's live checkout, which has moved past `39bcf79`. **Do not** install your new clone or SYRUP into them. Instead, place the verified materialized MAPLE 39bcf79 bundle ahead of site-packages, and run SYRUP from your pinned source with `python -m`.

A shared environment may **already** contain `maple-syrup-*` console scripts from some other installation. Those scripts launch whatever SYRUP that installation points at, not your pinned clone, so do not use them here. Run every command as `"$PY" -m maple_syrup.<module>` with the `PYTHONPATH` set below, and confirm the selected sources with `check-runtime` and the probe.

```bash
PY=/home/okin/MAPLE/.venv/bin/python             # or .venv-cupy/bin/python for GPU
BIN=$(dirname "$PY")                           # interpreter for the companion plotting commands
GBIN=/home/okin/MAPLE/.venv-cupy/bin             # shared GPU interpreter, when selected
C=/home/okin/SYRUP/outputs/dependencies/content_cache/37a1bc69bfa034be7cd512d8aa7c30d33682b7c053a53c3b0c91edc34d59a96f
PYTHONPATH=$SYRUP/src "$PY" -m maple_syrup.dependency_bundle env "$C" --syrup-src "$SYRUP/src"   # prints; runs nothing
# export exactly what it prints, typically:
export PYTHONPATH=$C/editable:$C/source/src:$SYRUP/src
export MAPLE_SYRUP_EXPECTED_MAPLE_ROOT=$C/source
export MAPLE=$C/source                          # matches explicit --expected-maple-root in later examples
"$PY" -m maple_syrup.dependency_bundle check-runtime "$C"
```

The bundle (ID `37a1bc69…`) is a local artifact, not something a Git clone provides. If it is missing, use the fresh-clone route in Sections 1.2–1.5. To rebuild such artifacts, see `dependencies/maple_shared_storage/README.md` and `docs/cw_e2_dependency_delivery.md`. In that route, substitute `"$PY" -m maple_syrup.<module>` for every console command below. Also do not run from a directory that contains a stray `maple-*.dist-info`: the resolver correctly refuses the shadowing distribution.

---

## 2. Check the installation: the probe

```bash
cd "$WORK"
"$BIN/maple-syrup-probe" --expected-maple-root "$MAPLE" --output "$RUNS/probe_cpu.json"
CUDA_VISIBLE_DEVICES=0 "$GBIN/maple-syrup-probe" --backend cupy --expected-maple-root "$MAPLE" \
  --output "$RUNS/probe_gpu.json"        # optional
```

The probe resolves MAPLE, checks the API surface SYRUP uses, runs a tiny zero-demand MAPLE water exchange and prints a JSON report. `--output` must name a new file outside the MAPLE tree. Check these fields:

- `status` is `"ok"`;
- `maple.dependency.source_root` is your `$MAPLE`;
- `maple.package_source_digest.digest_sha256` is `af54626ab432670539ed71480fd884f13581dfdfcfec63d7c85053ca215adfec`.

A different digest means the imported MAPLE is not the qualified 39bcf79 source. Opt-in overlay storage will then be refused (Section 6.3).

Exit codes: 0 ok; 1 failed probe check; 2 MAPLE dependency refusal, unavailable backend or refused output file. A CuPy probe only round-trips a tiny state. It is not evidence of GPU readiness.

---

## 3. What you must supply: MAHLERAN inputs (not in Git)

The model runs on cases imported from MAHLERAN 1.2.x inputs. **Neither repository contains these inputs.** Obtain them from your collaborators or a MAHLERAN distribution, and treat that tree as read-only.

For corrected Plot 1, `cases/plot1/recipe.yaml` expects:

```
$MAHLERAN/
  mahleran_input.xml          # version 1.2.3 (checked); its input_folder must resolve to Input/input_p1
  Input/input_p1/
    p1dem.asc                 # 62 x 22, 0.5 m cells (checked against the recipe)
    p1vegcover.asc  rm_new.asc  p1pavcoverveg.asc  thetasat39.asc
    <surface-type map named in the XML>
    plot1_phi1.asc ... plot1_phi6.asc
    p1_01_08_06.dat           # rainfall series named by the XML (copied verbatim)
    (p1ksat290806.asc, p1sm290905.asc are hashed only if present)
```

Rules the importer enforces:

- every staged raster must share the DEM header;
- the DEM must have no nodata cells;
- the six grain fractions must close within `1e-6`;
- **no `calib.dat`** may exist in `Input/input_p1`, since legacy calibration multipliers are unsupported.

"Corrected" means the recipe deliberately uses the six distinct `plot1_phi1..6.asc` maps. The supplied `mahleran_input.xml` maps all six classes to `plot1_phi1.asc`; that is a recorded data defect, not a reproduction target. Bed thickness, bulk density (1250 kg m⁻³), voxel size (0.1 m) and active layer (2 mm) are explicit assumptions documented in the recipe, not MAHLERAN values.

---

## 4. Directory layout you will end up with

```
$WORK/
├── MAPLE-39bcf79/                 # pinned MAPLE clone (read-only in use)
│   ├── .venv/  .venv-cupy/        # environments (Section 1)
│   ├── pyproject.toml
│   └── src/maple/
├── SYRUP-d05d8c6/                 # pinned SYRUP clone (read-only in use)
│   ├── cases/plot1/recipe.yaml    # authored recipes (input only; never write here)
│   ├── cases/rfid/recipe.yaml
│   ├── cases/chastre/recipe.yaml
│   ├── src/maple_syrup/
│   └── docs/  dependencies/
├── MAHLERAN/                      # external legacy inputs (Section 3), read-only
└── runs/
    ├── cases/plot1_maple39/       # imported + compiled case package
    │   ├── case.yaml  provenance.yaml
    │   ├── source/mahleran_input_p1/*.asc        # staged copies of the inputs
    │   ├── source/syrup_derived/composition_codes.npy
    │   ├── processed/  diagnostics/              # MAPLE compiler artifacts
    │   └── syrup/
    │       ├── plot1_binding.json                # written last; only then is the import complete
    │       ├── plot1_import_report.json  plot1_fields.npz
    │       └── rainfall/p1_01_08_06.dat
    ├── event_cpu/                 # one run's outputs (must not exist beforehand)
    │   ├── completion_summary.json  final_state.npz  hydrograph.npz
    │   ├── handoff/                # dry-handoff bundle (complete runs only)
    │   └── exchange_history.npz    # only with --record-exchange-history
    ├── event_cpu_checkpoints/
    │   └── step_000001234/         # checkpoint.json, maple_state.npz, continuation.npz
    └── stores/win_cpu/             # resident-window deep store (MAPLE-owned layout)
```

Paths a run creates must be new: the output directory, its sibling `<output>_checkpoints` (unless you pass `--checkpoint-dir` to the dense runner) and, for a fresh windowed run, the `--store-dir`. A run builds its output in a hidden `.<name>.partial-*` sibling and renames it into place only after it succeeds.

The case package records the absolute recipe path and MAHLERAN root. Later runs re-audit it against those paths. **Do not move or edit `$SYRUP/cases/plot1/recipe.yaml` or `$MAHLERAN` after importing.**

---

## 5. Commands: nine installed console scripts, no single `syrup` command

There is no unified `syrup` command. The SYRUP `pyproject.toml` installs nine console scripts. Each one is equivalent to running its module with `python -m`.

| Console script | `python -m` module | Purpose | Backends |
|---|---|---|---|
| `maple-syrup-probe` | `maple_syrup.probe` | dependency/API/provenance check | numpy, cupy |
| `maple-syrup-plot1-import` | `maple_syrup.case_import` | audit + compile corrected Plot 1 | (CPU) |
| `maple-syrup-columns` | `maple_syrup.column_experiment` | no-routing infiltration columns; diagnostic | numpy, cupy |
| `maple-syrup-storm` | `maple_syrup.storm_experiment` | **water-only** storm and recession on frozen terrain | numpy (array/numba), cupy (cuda) |
| `maple-syrup-sediment` | `maple_syrup.sediment_experiment` | water + wet sediment over a **time window** (no stop criterion, reset or restart) | numpy, cupy |
| `maple-syrup-event` | `maple_syrup.event_experiment` | **conservative complete event**, dense MAPLE bed, Plot 1 | numpy (numba/array), cupy (cuda) |
| `maple-syrup-windowed` | `maple_syrup.windowed_experiment` | **conservative complete event**, MAPLE resident-window bed; plot1/rfid/chastre | numpy (numba), cupy (cuda) |
| `maple-syrup-benchmark` | `maple_syrup.benchmark_experiment` | matched fixed-terrain benchmark; default dispatches to the legacy replay | numpy only |
| `maple-syrup-legacy` | `maple_syrup.legacy_experiment` | MAHLERAN legacy sediment replay, Plot 1 | numpy only |

Module-only user commands (no console script):

| Module | Purpose |
|---|---|
| `maple_syrup.rfid_case` | import RFID_2014 (water benchmark case) |
| `maple_syrup.chastre_case` | plan / generate / verify tiled Chastre case |
| `maple_syrup.experimental_experiment` | experimental water-only explicit / local-inertial solvers, with snapshots |
| `maple_syrup.legacy_driver` | legacy replay driver for plot1 / rfid / chastre, CPU/Numba |
| `maple_syrup.legacy_gpu_driver` | resident-GPU legacy replay, plot1 / rfid / chastre |
| `maple_syrup.dependency_bundle` | export / verify / materialize / env / check-runtime of MAPLE bundles |
| `maple_syrup.initial_reimport` | maintainer-only proposal/import of a generation-zero initial assembly under an independently reviewed exact source transition; not wet checkpoint migration |

### 5.1 Which runner should I use?

| You want | Use |
|---|---|
| An actual conservative evolving bed: supply-limited pickup from the real MAPLE active layer and voxels, net-elevation hydraulic refresh, complete event, dry reset, checkpoint/resume | `maple-syrup-event` (dense bed, Plot 1) or `maple-syrup-windowed` (resident window; Plot 1, RFID, Chastre) |
| Comparison with the MAHLERAN legacy program | `maple-syrup-benchmark` / `maple-syrup-legacy` (default `--transport-scheme legacy`) or `legacy_driver` |
| Water only | `maple-syrup-storm`, `experimental_experiment` |
| Infiltration diagnostics | `maple-syrup-columns` |
| Sediment over a fixed window, no completion | `maple-syrup-sediment` |

The two "mainstem" workflows have different defaults, so they are easy to confuse:

| | Benchmark default | Conservative event default |
|---|---|---|
| Transport | `legacy` replay | `characteristic` phase-bin scheme, `--phase-bins 32`, dense phase storage (windowed: CSR, `--phase-max-nnz 1000000`) |
| Sediment supply | fixed composition, **unlimited supply**, explicit clipping source | MAPLE's native exchange: demand capped by currently *available* mass (the same rule as wind), then by active-layer holdings, with refill from the voxel bed; no clipping source |
| Bed | **no evolving MAPLE bed**; frozen terrain and routing | evolving bed; reroute only on net elevation change ≥ 0.00075 m |
| Step | fixed dt = 1 s | adaptive, `--max-dt-s 1` |
| Completion, dry handoff, restart | none | yes (Section 7) |
| GPU | refused (`legacy_gpu_driver` is separate) | `--backend cupy` |

The legacy replay is a benchmark of the old program's behaviour, not MAPLE authority.

**Wind coupling.** A completed conservative event ends with a dry handoff: the dry native MAPLE bed, which a qualified wrapper can pass to MAPLE's wind model. No SYRUP command invokes a wind event. Sequential wind–water coupling is MASY's job (Section 11). The legacy replay produces no handoff bed at all.

### 5.2 CPU vs GPU: explicit pairings, no fallback

| Runner | CPU | GPU |
|---|---|---|
| `maple-syrup-event` | `--backend numpy --implementation numba` (default) or `array` | `--backend cupy --implementation cuda` (`--cuda-mode auto/fused/split`) |
| `maple-syrup-windowed` | `--backend numpy` (Numba hydraulics implied) | `--backend cupy` (CUDA hydraulics implied) |
| `maple-syrup-storm` / `-sediment` | `--implementation numba` (default) or `array` | `--backend cupy --implementation cuda` |
| `experimental_experiment` | `--backend numpy --implementation numpy/numba` | `--backend cupy --implementation cuda` |
| `maple-syrup-legacy`, `maple-syrup-benchmark` | numpy only | refused; use `legacy_gpu_driver` |

For these runners, any pairing not listed in the table (for example `--backend cupy` with `--implementation numba`, or `--implementation cuda` on NumPy) is refused before the case is read. Nothing falls back between CPU and GPU. The table covers the main CLIs only; the exhaustive choices for every command are in the reference appendix. All model arrays are float64. Pascal consumer GPUs have low FP64 throughput, and no speed claim is made for either backend.

Select a GPU with `CUDA_VISIBLE_DEVICES=<index or UUID>`; the runner uses device 0 of the visible set. On shared machines, check that the device is idle first.

- **Dense GPU event:** places the whole small case on the device after a memory preflight.
- **Windowed GPU run:** first sets a hard limit on the CuPy default memory pool, equal to 0.85 × the device's *current* free memory (or a stricter existing limit). It does this before uploading anything and restores the previous limit afterwards.
- **Out-of-memory:** inside that limit, it becomes an explicit `resource refusal`. The failing physical attempt is not adopted or checkpointed; earlier accepted checkpoints remain valid. Initial-store or diagnostic artifacts may already exist. A final successful output directory is not produced for the failed run.
- **Memory plan:** the pre-run plan is known to undercount. Passing it does not mean the run fits.

---

## 6. Plot 1 end-to-end

### 6.1 Import and compile the corrected case

```bash
"$BIN/maple-syrup-plot1-import" \
  --recipe "$SYRUP/cases/plot1/recipe.yaml" \
  --mahleran-root "$MAHLERAN" \
  --expected-maple-root "$MAPLE" \
  --output-dir "$RUNS/cases/plot1_maple39"
```

`--mahleran-root` overrides the recipe's `/home/okin/MAHLERAN` default without editing the recipe. The importer:

1. stages and hashes every input it reads;
2. writes the MAPLE case package;
3. calls MAPLE's real `compile_case`, then reloads and checks the compiled state;
4. writes `syrup/plot1_binding.json` last.

The binding records the case identity, the compiled-artifact hash and the MAPLE package digest. On failure it writes `syrup/FAILED.json` (with traceback) and no binding; delete that directory and use a new path. `--no-compile` writes the audited package without compiling, which is useful only for inspection.

Exit codes: 0 ok; 1 import or MAPLE rejection; 2 MAPLE dependency refusal.

Every later runner re-verifies the case before using it:

- bound file hashes;
- MAPLE's `load_compiled_case` re-hash;
- a fresh audit from the recipe and MAHLERAN inputs;
- the bound MAPLE digest.

If you change MAPLE, re-import (Section 6.5).

### 6.2 Run the conservative event: dense bed, CPU and GPU

```bash
cd "$WORK"
CUDA_VISIBLE_DEVICES='' "$BIN/maple-syrup-event" \
  --case-dir "$RUNS/cases/plot1_maple39" \
  --output-dir "$RUNS/event_cpu" \
  --expected-maple-root "$MAPLE" \
  --max-end-s 5400

CUDA_VISIBLE_DEVICES=0 "$GBIN/maple-syrup-event" \
  --backend cupy --implementation cuda \
  --case-dir "$RUNS/cases/plot1_maple39" \
  --output-dir "$RUNS/event_gpu" \
  --expected-maple-root "$MAPLE" \
  --max-end-s 5400
```

`--max-end-s` is a **horizon, not a target**. After the rain stops, the event keeps running until the completion gates hold: `--hold-s 60`, `--max-depth-m 1e-8`, `--surface-volume-m3 1e-6`, `--max-unit-discharge-m2-s 1e-10`, `--outlet-discharge-m3-s 1e-9`, and the mobile-mass significance threshold. It then makes a forced final terrain commit and the explicit dry reset (Section 7).

If the horizon is reached first, the run **fails** and writes no output directory. It prints `IncompleteEventError: maximum event duration reached; water and sediment were not reset; last accepted checkpoint: <path>` and exits with status 1. That is not completion.

On success, the JSON summary is printed to stdout and saved as `completion_summary.json`. Defaults are listed in the reference appendix; the main ones are `--checkpoint-every-s 600`, `--report-every-s 60`, `--max-dt-s 1`, characteristic transport with 32 bins, and bisection with 40 iterations.

The 5400 s horizon in these examples is a chosen limit, not a newly verified completion time for this source pair. If the strict mobile-load gate prevents drying, a **new** run can explicitly select `--deposit-mobile-at-flow-end`: once all hydraulic gates have held for the quiet period, it deposits the remaining finite mobile mass through MAPLE. This is a model assumption, not a relaxed conservation tolerance; retain the flag in every continuation of that run.

### 6.3 Run the resident-window bed (CPU and GPU)

```bash
CUDA_VISIBLE_DEVICES='' "$BIN/maple-syrup-windowed" \
  --case-kind plot1 --backend numpy \
  --case-dir "$RUNS/cases/plot1_maple39" \
  --store-dir "$RUNS/stores/win_cpu" \
  --output-dir "$RUNS/win_cpu" \
  --max-end-s 5400

CUDA_VISIBLE_DEVICES=0 "$GBIN/maple-syrup-windowed" \
  --case-kind plot1 --backend cupy \
  --case-dir "$RUNS/cases/plot1_maple39" \
  --store-dir "$RUNS/stores/win_gpu" \
  --output-dir "$RUNS/win_gpu" \
  --max-end-s 5400
```

The windowed runner keeps a few top voxels per column resident (`--window-voxels 4`). The deep voxels go to an immutable, versioned MAPLE host store under `--store-dir`. Final outputs and checkpoints contain resident arrays plus **references** to store versions, so they are not standalone archives. **Keep each store** for as long as any output or checkpoint that references it must stay readable or resumable. Old versions are never rewritten. A failed attempt may leave an unreferenced version, but it never damages referenced ones.

Resource controls, all integers in bytes or records:

- `--max-host-bytes` (40 GiB): bounds the host assembly planning estimate and the default checkpoint payload allowance, not the process's operating-system RSS;
- `--max-store-bytes` (256 GiB): bounds the initial deep store's logical size, not the total disk space of every later retained version;
- `--max-journal-records` (1,000,000);
- `--phase-max-nnz` (1,000,000; refused before allocation, mass is never truncated);
- `--max-checkpoint-bytes` (defaults to `--max-host-bytes` and may only tighten it).

The windowed runner disables avalanching for this water-only scope and records the original setting.

`--reuse-initial` restores the persisted initial window assembly from an existing `--store-dir` instead of re-importing it. The source's provenance is recorded. The examples above give each run its own fresh store, which is the simplest safe choice.

**Full vs overlay deep storage.**

- `--deep-chunk-encoding full` is the default.
- `overlay` stores a raw base plus one cumulative, exact final-value overlay per chunk. No tolerance-based dropping and no chains are used.
- Overlay is accepted only when the imported MAPLE package digest is a qualified typed source. For this guide that is `af54626…` (MAPLE 39bcf79); one older MAPLE source is also accepted for historical reproduction. Otherwise it is refused before any case verification or store creation.
- Overlay is also refused for cases with more voxels than MAPLE's `OVERLAY_MAX_VOXELS`, and for dense beds.
- `--deep-overlay-max-bytes` applies only with `overlay`.
- The encoding is part of the bed-storage identity. A run must resume with the same encoding, and a v2 (overlay) store cannot continue under `full`.

### 6.4 Pause and resume (same source, same backend, same controls)

```bash
CASE=$RUNS/cases/plot1_maple39

# Part 1: run to an original boundary and pause (checkpoint written at the pause)
CUDA_VISIBLE_DEVICES='' "$BIN/maple-syrup-event" --case-dir "$CASE" \
  --output-dir "$RUNS/event_pr_part1" --expected-maple-root "$MAPLE" \
  --max-end-s 5400 --pause-at-s 1200

CKPT=$("$BIN/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["last_checkpoint"])' \
       "$RUNS/event_pr_part1/completion_summary.json")
echo "$CKPT"     # .../event_pr_part1_checkpoints/step_NNNNNNNNN

# Part 2: NEW output directory, identical controls and --max-end-s, same interpreter/backend
CUDA_VISIBLE_DEVICES='' "$BIN/maple-syrup-event" --case-dir "$CASE" \
  --output-dir "$RUNS/event_pr_part2" --expected-maple-root "$MAPLE" \
  --max-end-s 5400 --resume "$CKPT"
```

A paused run exits 0 with `"status": "paused"`. `--pause-at-s` must be a future *original* boundary (rain knots or report times, multiples of 60 s by default) before the horizon. Otherwise the run is refused with `pause time must be a future original boundary before the duration limit`.

For windowed runs, use the same pattern and give both parts the **same `--store-dir`**. A resume restores the initial assembly from that store automatically.

Resume is strict. The checkpoint's recorded identity must equal the current one exactly. That identity covers:

- the SYRUP source digest and the resolved MAPLE package digest;
- the case identity and binding, and the MAPLE configuration;
- the MAHLERAN XML and rainfall hashes;
- the sediment and column parameters;
- the solver and transport controls, the completion policy and the graph (pit/nodata) policy;
- `--max-end-s`, the report cadence and the row cap;
- the backend (plus `--cuda-mode` on CUDA);
- for windowed runs, the window, caps, initial deep manifest and deep encoding.

**CPU checkpoints do not resume on CuPy, and GPU checkpoints do not resume on CPU.** `--max-checkpoint-bytes`, `--checkpoint-every-s` and `--pause-at-s` are resource or scheduling controls and are excluded from the identity. During a run, the SYRUP source, MAPLE source, MAHLERAN XML and rainfall file are re-hashed at every checkpoint, and any change aborts the run. Do not `git pull`, edit or reinstall while a run is active.

The resumed run carries the checkpoint's accumulated result and origin, so its hydrograph continues from the event origin. Check `t_s[0]` before concatenating anything yourself.

### 6.5 When MAPLE changes: `--allow-maple-source-change`

Normal runs should not use this flag: re-import the case with the current pin instead. The flag only acknowledges, at **case verification**, that the MAPLE digest recorded at import differs from the MAPLE imported now. The difference is recorded in the summary (`backend.maple_source_changed_since_import`).

It does **not** override any of these:

- checkpoint/restart identity (a resume still refuses a checkpoint with a different recorded MAPLE);
- the overlay/deferred-publication source guards;
- source-bound windowed store and initial-assembly bindings;
- the within-run source re-hashing.

Old cases, outputs, checkpoints and source locks from earlier MAPLE or SYRUP versions keep their original identities. Normal run/restart commands do not migrate or rebind them. The separate maintainer-only `initial_reimport` utility can reuse a genuinely initial Chastre assembly under an exact reviewed producer-to-consumer transition. It requires an independently reviewed artifact and unchanged original operands; its own unreviewed proposal is refused. It does not convert a wet checkpoint or authorize an ordinary user to bypass source guards.

---

## 7. Event semantics you need to interpret results

- **Completion and dry reset.** `"status": "complete"` means every quiet gate held for `--hold-s` after the rain ended, the forced final commit succeeded and the post-commit checks passed. The dry reset then removes the remaining surface water and soil water as an **external, reported removal** (`reset_accounting` in the summary). Mobile sediment is never erased; it must have been deposited first.
- **`--deposit-mobile-at-flow-end`** (off by default) is an explicit end-of-event assumption that the bed dries again. Once all hydraulic gates hold, MAPLE deposits the remaining finite mobile load, which is reported separately. It waives only the mobile-mass criterion and is not a mass tolerance.
- **Exact dry handoff.** Only a `complete` run writes `handoff/`, the dry-state bundle. A completed handoff cannot be resumed as a wet event.
- **Terminal pit storage.**
  - *Dense runner:* opt in with `--allow-pit-storage`. Cells with no strictly lower D4 neighbour then deposit their mobile load through MAPLE and keep their water. `--allow-flat-storage` extends this to flat sinks.
  - *Windowed runner:* pit storage is on automatically for `rfid`/`chastre` (flat storage too for Chastre) and unavailable for `plot1`.
  - *No overflow.* Pits have no spill, overtopping or backwater. `--terminal-storage-water` lets an event complete with water ponded **only** in terminal-storage cells, and the reset removes it. Large Chastre terminal depths are therefore not credible pond predictions.
- **Routing refresh.** Hydraulics reroute only when the largest per-cell |committed surface − held routing surface| reaches 0.00075 m. Time, reports, checkpoints and ordinary accounting commits never reroute by themselves.
- **Excluded from these runners:** running a wind event (the dry handoff bed is the hand-over point; scheduling is MASY's job), direct dry-cell splash, vegetation growth, full ET, nutrients and pit overflow.

**Pond/splash API (no installed CLI).** `maple_syrup.pond_event` is a separate, opt-in Python API. It couples local-inertial pond hydraulics with actual overflow/export at outlet faces, dry rain splash, multi-face characteristic transport and the native MAPLE bed. It supports `PondEventControl(bed_storage="windowed", ...)` and its own checkpoints (`pond_checkpoint`). It is **not** reachable from any console command; see `docs/splash_and_overflow/pond_event.md` and `docs/pond_windowed/README.md`. Local inertia is an alternative hydraulic model, not MAHLERAN method 5/6.

---

## 8. Other workflows

### 8.1 Water-only and diagnostic runners

```bash
C=$RUNS/cases/plot1_maple39
"$BIN/maple-syrup-columns"  --case-dir "$C" --max-dt-s 1 --output-dir "$RUNS/columns"
"$BIN/maple-syrup-storm"    --case-dir "$C" --output-dir "$RUNS/storm_cpu"
CUDA_VISIBLE_DEVICES=0 "$GBIN/maple-syrup-storm" --case-dir "$C" --backend cupy --implementation cuda \
  --output-dir "$RUNS/storm_gpu"
"$BIN/maple-syrup-sediment" --case-dir "$C" --output-dir "$RUNS/sediment_window"
```

- **Columns** are no-routing infiltration diagnostics. They write `column_summary.json` and `final_columns.npz`.
- **Storm** is a water-only storm and recession on frozen terrain with no sediment and no reset. It writes `storm_summary.json`, `final_water.npz`, `hydrograph.npz` and `hydrograph.csv`.
- **Sediment** runs water and wet sediment on the actual bed **up to `--end-s`** (default 5400 s). It has no stop criterion, reset or restart, so it is not a complete storm. It writes `sediment_summary.json`, `final_state.npz` (including `initial_committed_elevation_m`), `hydrograph.npz`, `hydrograph.csv` and `maple_final_snapshot.npz`, which comes from MAPLE's own `save_state_snapshot`.

### 8.2 Benchmark and legacy replay: needs reference artifacts you probably do not have

```bash
"$BIN/maple-syrup-benchmark" --case-dir "$C" --output-dir "$RUNS/legacy_replay" \
  --applied-rainfall "$REF/applied_rainfall.csv" \
  --fortran-ledger "$REF/syrup_sediment_ledger.dat" \
  --reference-run "$REF/mahleran_deterministic_ksat_run"
```

The default `--transport-scheme legacy` dispatches to `maple_syrup.legacy_experiment` (`maple-syrup-legacy` runs the same thing). It **requires** artifacts from an executed, instrumented MAHLERAN reference run:

- the logged applied-rainfall CSV;
- the Fortran sediment ledger;
- a reference-run directory with `execution.json` and `Output/`.

The default values of those flags (`outputs/phase7/...`, `outputs/phase7b/...`) are relative paths to git-ignored local results. A fresh clone does **not** contain them, so this workflow is not turnkey. `$REF` above stands for wherever you hold such artifacts. If they are missing, the run stops with `legacy replay failed: actual Fortran reference ledger and reference run execution.json are required`.

Controls the replay cannot honour are refused with exit status 2 (`legacy replay refused: ...`):

- `--max-dt-s` other than 1;
- `--backend cupy`;
- `--phase-bins`;
- Courant or substep options.

Outputs: `legacy_ledger.npz` and `legacy_summary.json`.

`--transport-scheme characteristic` (or `upwind`) selects the explicit conservative benchmark instead: actual MAPLE exchange on frozen hydraulic geometry with the reference applied rainfall. It still requires `--applied-rainfall`, and has no completion, dry reset, restart, wind or GPU.

`maple_syrup.legacy_driver` (CPU/Numba) and `maple_syrup.legacy_gpu_driver` (resident GPU) support **more cases**: `--case-kind plot1|rfid|chastre`. For Plot 1 they use the parsed case rainfall unless you pass `--applied-rainfall`, and they need no Fortran ledger. They write `legacy_ledger.npz`, `legacy_summary.json` and, with `--snapshot-times`, `legacy_snapshots.npz` (Section 10). A failure leaves `<out>.FAILED/FAILED.json`. Both drivers use frozen terrain, unlimited supply and an explicit clipping source.

### 8.3 Optional case kinds: RFID and Chastre

These cases need extra external inputs and use distinct import commands, not `maple-syrup-plot1-import`.

**RFID_2014** (0.1 m, 106 × 60 including the ring, nodata −9999). You must supply two inputs that are not in Git:

- `$MAHLERAN/Input/RFID_2014/`, with XML version 1.2.0;
- `$NATIVE_CAPTURE`, the `syrup_hydro_steps.txt` written by an executed, instrumented native MAHLERAN run of RFID_2014.

The recipe pins the capture's SHA-256 (`capture_expected_sha256`). A different capture file is refused. The recipe's hydrology values, including its fixed conductivity, come from an audit of that native run. Treat them as values for this one benchmark, not as turnkey defaults for other sites.

```bash
export NATIVE_CAPTURE=/abs/path/to/your/syrup_hydro_steps.txt   # user-supplied artifact
"$BIN/python" -m maple_syrup.rfid_case \
  --recipe "$SYRUP/cases/rfid/recipe.yaml" \
  --mahleran-root "$MAHLERAN" --expected-maple-root "$MAPLE" \
  --applied-forcing-capture "$NATIVE_CAPTURE" \
  --output-dir "$RUNS/cases/rfid_maple39"
```

**Chastre** is a 1 m tiled case (1393 × 1604 interior, about 33 GB of persisted bed). It needs the separately obtained terrain `DTM_Chastre_PD4way.asc`, which the recipe pins by SHA-256, and a completed RFID case. The recipe holds absolute `/home/okin/...` paths, and both `--plan-only` and generation read them as written. Relative paths would resolve against your current directory, not the recipe's directory.

Copy the recipe and set **both** paths to absolute locations you own:

- `terrain.path`: your copy of the DTM;
- `source_rfid.case_dir`: your RFID case.

Keep `terrain.sha256` and every other value unchanged. `--plan-only` always reads `source_rfid.case_dir` from the recipe and ignores `--source-case-dir`. Editing the recipe therefore keeps the plan and the generation consistent, and you can then omit `--source-case-dir`.

```bash
mkdir -p "$WORK/recipes" && cp "$SYRUP/cases/chastre/recipe.yaml" "$WORK/recipes/chastre.yaml"
# edit $WORK/recipes/chastre.yaml:
#   terrain.path:        /abs/path/to/DTM_Chastre_PD4way.asc
#   source_rfid.case_dir: <the value of $RUNS>/cases/rfid_maple39   (write it out in full)
cd "$WORK"
"$BIN/python" -m maple_syrup.chastre_case --recipe "$WORK/recipes/chastre.yaml" \
  --output-dir "$RUNS/cases/chastre_maple39" --plan-only      # writes nothing; exit 3 if disk too small
"$BIN/python" -m maple_syrup.chastre_case --recipe "$WORK/recipes/chastre.yaml" \
  --output-dir "$RUNS/cases/chastre_maple39" --expected-maple-root "$MAPLE"
"$BIN/python" -m maple_syrup.chastre_case --verify-case "$RUNS/cases/chastre_maple39" --hash-only-tiles
```

The case records the copied recipe's absolute path and hash, so keep that copy unchanged after generation.

Run the conservative event on these cases with the windowed runner. You **must** pass the case's declared nodata value; it is never inferred, and a mismatch is refused.

```bash
CUDA_VISIBLE_DEVICES='' "$BIN/maple-syrup-windowed" --case-kind rfid --backend numpy \
  --case-dir "$RUNS/cases/rfid_maple39" --store-dir "$RUNS/stores/rfid_cpu" \
  --output-dir "$RUNS/rfid_cpu" --nodata-value -9999 --terminal-storage-water
```

For Chastre, earlier large runs used `--phase-max-nnz 4000000 --max-journal-records 4000000`. The largest documented run used a separately prepared *rotated* Chastre case. It ran through the Python API `run_complete_windowed(...)` with API-only deferred publication (`deep_publication='boundary'`), not through this CLI, and it stopped **incomplete** at its 10,000 s horizon. A full large storm is not known to fit or complete.

---

## 9. Outputs: files, units, orientation

Read every NPZ with `allow_pickle=False`. All arrays are float64 unless stated.

**Orientation.**

- Grid arrays are `(ny, nx)` with **row 0 = south**. Plot with `origin="lower"`.
- Legacy ASCII grids are north-first; the importer crops the one-cell ring and reverses the rows.
- Sidecar `legacy_full_*` fields in `plot1_fields.npz` keep the ring (62 × 22) and are also south-first.
- Class axes are last, in canonical class order (`c1` = finest phi_1 … `c6` = coarsest).

**Event and windowed outputs (`<output>/`):**

| File / key | Meaning | Units |
|---|---|---|
| `completion_summary.json` | status, identity, budgets, closures, reset accounting, routing counts, timings, `outputs` SHA-256 map | — |
| `final_state.npz: depth_m`, `soil_water_m` | for a `complete` run, the state **after** the dry reset (the removed water is in `reset_accounting`); for a `paused` run, the wet state at the pause | m |
| `pre_reset_depth_m`, `pre_reset_soil_water_m` | physical state at the end of the accepted event, before any reset; use these for final water maps | m |
| `discharge_m2_s` | unit discharge | m² s⁻¹ |
| `committed_elevation_m` | MAPLE committed surface (MAPLE datum: legacy DEM + `datum_offset_m`) | m |
| `active_mass_kg`, `available_mass_kg`, `bound_mass_kg`, `mobile_mass_kg` | per cell and class | kg |
| `voxel_mass_kg` (dense) | full voxel column `(ny, nx, nz, nc)` | kg |
| `window_kg` (windowed) | resident window voxels `(ny, nx, W, nc)` | kg |
| `deep_by_class_kg`, `deep_total_kg`, `deep_room_kg` (windowed) | per-cell totals over the deep voxels below the window: mass by class, total mass, unfilled capacity. The voxel-by-voxel deep contents live in the store | kg |
| `base` (windowed) | `(ny, nx)` int32 logical voxel index of window slot 0, i.e. the first resident level | dimensionless index, **not kg** |
| `terminal_deposition_kg` | final settling deposit (zeros if not complete) | kg |
| `sediment_velocity_m_s` | transport velocity memory per cell and class | m s⁻¹ |
| `phase_fraction` / `phase_csr_fraction` | share of the cell/class mobile mass held in each position bin | dimensionless |
| `phase_position_m` / `phase_csr_position_m` | representative distance from the cell's upstream face | m |
| `phase_csr_row_ptr`, `phase_csr_bin`, `phase_csr_cell_of_slot` | CSR structure | integer indices |
| `hydrograph.npz` | one row per boundary; water columns `t_s`, `cumulative_rain_m3`, `cumulative_intake_m3`, `cumulative_saturation_return_m3`, `cumulative_drainage_m3`, `cumulative_export_m3`, `surface_storage_m3`, `soil_storage_m3`, `outlet_discharge_m3_s`, `max_depth_m`, … ; sediment columns prefixed `sed_` (`sed_mobile_kg`, `sed_cumulative_export_kg`, `sed_export_rate_kg_s`, per-class `sed_cumulative_export_c1_kg` …) | m³, m³ s⁻¹, kg, kg s⁻¹ |
| `exchange_history.npz` | opt-in cumulative per-cell/class MAPLE ledger touch counts and gross activity | counts, kg |

**Interpreting the hydrograph.**

- Columns named `cumulative_*` are running totals since the event origin. Get interval volumes or masses as differences of consecutive rows.
- `outlet_discharge_m3_s` is the **instantaneous** water discharge at the end of the last accepted step before the row.
- `sed_export_rate_kg_s` (and per-class `sed_export_rate_c*_kg_s`) is the actual export of that **last accepted step divided by its Δt**. It is a last-step average, not an average over the report interval.
- For a mean over a report interval, divide the change in a cumulative column by the change in `t_s`, e.g. `np.diff(h["sed_cumulative_export_kg"]) / np.diff(h["t_s"])`. Never multiply either of the rate columns by the report interval to estimate a volume or yield.
- Rows sit at report times (60 s) plus rain knots, so spacing can be irregular.
- The final row includes terminal settling and the final commit, before the external reset.

**Bed change.** `final_state.npz` stores only the final committed surface. For a **Dense Plot 1 event**, compute the change against the initial committed surface of the same compiled case, in the same datum frame:

```python
import numpy as np
from maple.case_tools.compilers.case_compiler import load_compiled_case
from maple.core.types.topographic_commit import committed_elevation_m

case_dir = "/abs/path/runs/cases/plot1_maple39"     # same case the run used
run_dir = "/abs/path/runs/event_cpu"
case = load_compiled_case(case_dir)
z0 = np.asarray(committed_elevation_m(case.committed_topography))
with np.load(f"{run_dir}/final_state.npz", allow_pickle=False) as z:
    z1 = z["committed_elevation_m"]
dz_m = z1 - z0          # + deposition, - erosion; same MAPLE datum, so the offset cancels
```

Run this in the MAPLE environment. `maple-syrup-sediment` already saves `initial_committed_elevation_m` alongside the final surface.

For a **windowed event**, use the actual initial assembly saved in that run's store. Windowed initialization can regroup the Dense baseline for its exact height-packet arithmetic; substituting a separately loaded Dense case can show a small apparent change even before erosion. The saved initial baseline uses the same native committed-elevation definition:

```python
from pathlib import Path
import numpy as np

store = Path("/abs/path/runs/stores/win_cpu")
run = Path("/abs/path/runs/win_cpu")
with np.load(store / "initial_baseline.npz", allow_pickle=False) as initial:
    z0 = initial["reference_elevation_m"] + initial["elevation_offset_m"]
with np.load(run / "final_state.npz", allow_pickle=False) as final:
    dz_m = final["committed_elevation_m"] - z0
np.savez("/abs/path/runs/figures/windowed_change.npz", dz_m=dz_m)
```

Keep the matching store, initial binding and run summary. On a paused run the committed surface can lag the physical bed's pending changes; a completed event forces final accounting. A height-change map of committed surfaces must be labeled accordingly.

`exchange_history.npz` and the per-cell `cumulative_*` maps are **accumulations over the whole event**. They are not time-resolved frames, and you cannot reconstruct a movie from them.

---

## 10. Figures and movies

**What exists.** No committed console runner writes general **evolving-bed spatial movie frames**. The explicit saved map sequences are:

- `experimental_experiment --snapshot-times-s t1,t2,…` writes `snapshots.npz`: **water-only** fixed-terrain maps keyed `t<time>_<field>` (for example `t600_depth_m`, `t600_soil_water_m`, `t600_velocity_m_s`, and face fluxes `t600_qx_m2_s`/`t600_qy_m2_s` for local inertia). The snapshot times are inserted as solver boundaries.
- `legacy_driver` / `legacy_gpu_driver --snapshot-times` (at most 16 whole seconds) writes `legacy_snapshots.npz` with stacked `t_s`, `depth_m` and `mobile_kg`. These are **fixed-bed** legacy maps. Check their shapes with `np.load(..., allow_pickle=False)` before plotting.

A movie of a conservative evolving bed would need explicit capture code written against the event API and checked against the source. There is currently no CLI option for it.

Checkpoints (`<output>_checkpoints/step_*`) hold complete wet states at the checkpoint cadence (600 s by default). A custom script could decode them through the checkpoint API with the run's own context and identity. That is developer work, not a movie command, and the frames would be at checkpoint times only.

### 10.1 The companion plotting example

`$MANUALS/render_outputs.py` is a small companion example; it is not an installed SYRUP command. It needs only NumPy and Matplotlib. It never runs the model, never touches a GPU or checkpoint, and never invents maps that were not saved. Its subcommands are:

- `inspect`: list the keys, shapes and dtypes in an NPZ without loading the payloads;
- `map`: draw one saved 2-D map, or one class (`--class-index`, zero-based) or the class sum (`--sum-classes`) of a 3-D `(row, column, class)` map;
- `hydrograph`: plot one column of `hydrograph.npz` against `t_s`;
- `frames`: render saved snapshots to PNG frames with one fixed colour scale.

Maps are drawn with row 0 at the bottom (south). With `--dx-m` the axes are in metres. The helper refuses to overwrite an existing output file or frames directory.

The helper assumes cell-centred maps. Staggered `qx_m2_s` and `qy_m2_s` need their own face coordinates and should not be passed to its cell-map command. With stacked legacy snapshots it holds one full selected field stack in memory; use small cases or a limited number of snapshots when memory is tight.

```bash
H="$MANUALS/render_outputs.py"
mkdir -p "$RUNS/figures"
"$BIN/python" "$H" inspect "$RUNS/event_cpu/final_state.npz"
"$BIN/python" "$H" map "$RUNS/event_cpu/final_state.npz" --key pre_reset_depth_m \
  --dx-m 0.5 --label "Depth before dry reset (m)" --output "$RUNS/figures/event_cpu_depth.png"
"$BIN/python" "$H" map "$RUNS/event_cpu/final_state.npz" --key mobile_mass_kg --sum-classes \
  --dx-m 0.5 --label "Mobile sediment, all classes (kg)" --output "$RUNS/figures/event_cpu_mobile.png"
"$BIN/python" "$H" hydrograph "$RUNS/event_cpu/hydrograph.npz" --key outlet_discharge_m3_s \
  --label "Outlet discharge, end of last step (m³/s)" --output "$RUNS/figures/event_cpu_q.png"
"$BIN/python" "$H" hydrograph "$RUNS/event_cpu/hydrograph.npz" --key sed_cumulative_export_kg \
  --label "Cumulative sediment export (kg)" --output "$RUNS/figures/event_cpu_sed_export.png"
```

For a complete run, map `pre_reset_depth_m` rather than `depth_m`, which is zero after the dry reset. The default `--label` is `Depth (m)`, so always set a label that matches the key. To map bed change, compute `dz_m` as in Section 9 and save it to a new NPZ (`np.savez(path, dz_m=dz_m)`). Then use `map --key dz_m --cmap RdBu_r` with an explicit symmetric `--vmin`/`--vmax` and label `Committed height change (m)`.

### 10.2 Making a water-depth movie from experimental snapshots

```bash
"$BIN/python" -m maple_syrup.experimental_experiment \
  --case-dir "$RUNS/cases/plot1_maple39" --output-dir "$RUNS/exp_explicit" \
  --solver explicit --backend numpy --implementation numba \
  --max-dt-s 1 --end-s 600 --expected-maple-root "$MAPLE" \
  --snapshot-times-s 60,120,180,240,300,360,420,480,540,600

"$BIN/python" "$H" frames "$RUNS/exp_explicit/snapshots.npz" --field depth_m \
  --dx-m 0.5 --label "Water depth (m)" --frames-dir "$RUNS/frames"

ffmpeg -framerate 4 -i "$RUNS/frames/frame_%05d.png" \
  -c:v libx264 -pix_fmt yuv420p "$RUNS/exp_explicit_depth.mp4"
```

Snapshot times must lie in (start, end]: `0` is refused, and the run starts at 0 s. Each requested time becomes an exact solver boundary. This is a water-only, fixed-terrain experimental solver, not the conservative event and not equivalent to the legacy hydraulics.

### 10.3 Legacy-replay snapshots

```bash
"$BIN/python" -m maple_syrup.legacy_driver --case-kind plot1 \
  --case "$RUNS/cases/plot1_maple39" --output "$RUNS/legacy_plot1" \
  --end-s 600 --snapshot-times 60,120,180,240,300,360,420,480,540,600
"$BIN/python" "$H" frames "$RUNS/legacy_plot1/legacy_snapshots.npz" --field mobile_kg --sum-classes \
  --dx-m 0.5 --label "Mobile sediment, all classes (kg)" --frames-dir "$RUNS/frames_legacy_mobile"
```

`legacy_snapshots.npz` stacks `t_s` (n), `depth_m` (n, ny, nx) and `mobile_kg` (n, ny, nx, nc). Select a class with `--class-index 0`, or sum all classes with `--sum-classes`. These are frozen-terrain legacy maps with unlimited supply. They are not evolving-bed results.

### 10.4 Reading movies correctly

- **Colour scale.** `frames` scans all frames first and uses one fixed scale, which it prints. Pass `--vmin`/`--vmax` to choose a physical range yourself, and use the same range when comparing movies.
- **Time.** Frame rate is playback speed only. The helper writes `frame_times_s.json` and warns when snapshot spacing is unequal; real time maps linearly onto video time only for evenly spaced snapshots. Solver steps are adaptive in the experimental runner, so each frame is labelled with its actual time.
- **Pixel size.** The helper saves fixed-size figures (792 × 624 px at its frame resolution), which suits `-pix_fmt yuv420p`. If a custom Matplotlib configuration changes the saved size (for example `savefig.bbox: tight`), add `-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"` to the FFmpeg command.
- **No invented frames.** There are no spatial flux or bed-change frames to render, because no runner saves them over time.

---

## 11. Scope limits

- **Supply and availability.** MAPLE's native water exchange caps requested pickup by the currently available mass, using exactly the same rule as wind erosion. The bound share is not released automatically. It then caps pickup per class at the active-layer holdings before refilling from the voxel bed. The corrected Plot 1 case starts with every class 100% available. Prescribed pavement and vegetation affect the water laws independently of that native availability split.
- **Wind.** A complete event hands over a dry native bed; MASY owns the sequential wind/water scheduling and calls each model. No SYRUP CLI runs or schedules a wind event.
- **Storage-admission checks.** `maple_syrup.storage_admission` (`admit_public`, `admit_paired`) provides opt-in caller checks for memory admission of windowed exchanges. Existing runners do not call it automatically.
- **Qualification status.** The 39bcf79 pairing is qualified for the storage integration (`docs/maple39_shared_storage_qualification.md`). That does not qualify:
  - alternating wind–water events;
  - large-catchment storms or whole-Chastre memory fit;
  - universal CPU/GPU bitwise equality.
- **Historical status.** README sections and dated status blocks in older docs are historical. Where they disagree with current source and with `docs/maple39_shared_storage_qualification.md` and `docs/pond_windowed/`, follow the source and those docs.

---

## 12. Troubleshooting

| Message (abridged) | Cause | Fix |
|---|---|---|
| `` `import maple` failed `` | wrong interpreter | use `$BIN`/`$GBIN` (or the Section 1.6 `PYTHONPATH`) |
| `expected MAPLE source root X, but the imported maple resolves to source root Y` | a different MAPLE (e.g. the shared live checkout) | check `MAPLE_SYRUP_EXPECTED_MAPLE_ROOT`, the venv, `PYTHONPATH` |
| `imported maple at … is not inside the editable install root` | another copy shadows it on `sys.path` | unset stray `PYTHONPATH`; leave directories containing `maple*` |
| `MAHLERAN XML not found`, `XML version … != expected '1.2.3'`, `input_folder … not the recipe's` | MAHLERAN inputs missing or different | fix `--mahleran-root` and the inputs (Section 3) |
| `calib.dat exists` | legacy calibration not supported | remove it from your input copy, then re-import |
| `refusing to write into existing path`, `output and checkpoint directories must be new` | reused path | choose new `--output-dir` (and its `_checkpoints` sibling) |
| `refusing to write inside the … tree` | output under MAPLE/MAHLERAN/recipe/SYRUP package | write under `$RUNS` |
| `…/syrup/FAILED.json exists: the import failed` | case directory from a failed import | re-import into a new directory |
| `MAPLE source digest … differs from the one bound at import` | case imported under another MAPLE | re-import with the current pin (Section 6.5) |
| `Numba requested but unavailable; no implicit fallback` | missing Numba | install `[windowed]` / `[numba]` extras |
| `PrefixPacketError: decode_height … needs the compiled Numba kernel` | windowed CPU run without Numba | install MAPLE `[windowed]` |
| `backend 'cupy' with implementation 'numba' is refused` | invalid pairing | `--backend cupy --implementation cuda` |
| `backend 'cupy' is refused: …` / `backend unavailable` | no CuPy or no visible device | use `.venv-cupy`; check `CUDA_VISIBLE_DEVICES`, driver |
| `Failed to find CUDA headers` | CuPy without `[ctk]` | reinstall `[windowed,gpu]` |
| `allow_flat_storage extends allow_pit_storage and needs it`; `terminal_storage_water needs an opt-in pit-storage graph` | dense pit options missing | add `--allow-pit-storage` (dense); not available for windowed `plot1` |
| `masked case requires explicit nodata_value` / `nodata_value … differs from the verified case grid` | windowed rfid/chastre | pass `--nodata-value -9999` (the declared value) |
| `deep_chunk_encoding='overlay' is supported only with an accepted MAPLE source` | MAPLE digest not qualified | use the 39bcf79 pin or `full` |
| `pause time must be a future original boundary …` | pause not on a report/rain boundary | pick a multiple of the report cadence before the horizon |
| `IncompleteEventError: maximum event duration reached; … last accepted checkpoint: …` | horizon reached before completion | not a result; inspect or resume only with the same `--max-end-s` (identity), or start a new run with a longer horizon |
| `checkpoint schema, case/source/config identity or payload set mismatch`; `restart stopping or numerical controls changed`; `restart duration or forcing changed` | resume with different source/backend/controls | resume with exactly the original sources, backend and controls |
| `SYRUP source changed during the event` / `MAPLE source changed …` | files edited or reinstalled mid-run | do not touch the trees during runs |
| `resource refusal: CuPy device memory request exceeded the enforced pool hard limit …` | does not fit the device | earlier checkpoints stay valid; free the device or use CPU |
| `resource refusal: the resident-window planning estimate alone exceeds 85% …` | estimate already too large | smaller case or larger device |
| `legacy replay refused: …` (exit 2) / `actual Fortran reference ledger … required` | unsupported legacy control / missing reference artifacts | Section 8.2 |

**Exit codes.**

- 0: success, including `paused` runs.
- 1: run, import or validation failure.
- 2: dependency refusal, usage error (argparse) or legacy refusal.
- 3: `chastre_case --plan-only` disk refusal.

The dense runners print errors to stderr. The windowed runner prints progress and errors to stdout.

---

## 13. Reproducibility checklist

Record these for every study:

1. **Exact commits and cleanliness.** Capture `git -C "$MAPLE" rev-parse HEAD`, `git -C "$SYRUP" rev-parse HEAD` and `git status --porcelain` for both. HEAD alone does not identify uncommitted edits; the digests do.
2. **Probe report.** Save it (`--output`), with the MAPLE package digest `af54626…`, `source_root`, Python and package versions. Also save `pip freeze` from each environment.
3. **Case provenance.**
   - `syrup/plot1_binding.json`: case identity, artifact hash, MAPLE digest, SYRUP provenance, environment.
   - `syrup/plot1_import_report.json`: recipe and XML hashes, staged-source hashes, resolved legacy settings, datum offset.
   - MAPLE's `provenance.yaml`.
4. **Run summary** (`completion_summary.json`):
   - `identity`: `syrup_source_sha256`, `maple_source_sha256`, `case_identity_sha256`, `mahleran_xml_sha256`, `rainfall_sha256`, `sediment_parameters`, `column_parameters`, `control`, `completion_policy`, `graph_policy`, `backend`;
   - `backend.maple_source_changed_since_import`;
   - `outputs`, the SHA-256 of every output file.
5. **Retention.** Keep every checkpoint directory you might resume from, and every windowed `--store-dir` that is referenced. Treat them as immutable.
6. **Dependency pins and bundles (advanced).** `dependencies/maple_shared_storage/README.md` (reconstruction recipe, fixed manifest SHA-256 `877826de…`), `qualification.lock.json`, `docs/cw_e2_dependency_delivery.md` (bundle export/verify/materialize/env/check-runtime) and `docs/maple39_shared_storage_qualification.md`. Bundles certify integrity, not authenticity, and do not carry third-party packages or deep-store data.
