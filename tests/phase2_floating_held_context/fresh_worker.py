"""Fresh-process worker of the held-context control. mode `export`: builds the genuine native bed/owner/vertical (+ explicit preferential spec) and writes the record parts (strict JSON + raw array files) and the expected
identities. mode `restore`: a NEW process with NO bed and NO owner reads only those files, parses, reconstructs with the provider/bed readers DISABLED, and prints the reproduced identities."""
import json
import sys
from pathlib import Path

mode, out = sys.argv[1], Path(sys.argv[2])
tests = Path(sys.argv[3])
backend = sys.argv[4] if len(sys.argv) > 4 else "numpy"
for sub in ("phase2_native_floating_soil", "phase2_compact_soil_profile", "phase2_floating_held_context"):
    sys.path.insert(0, str(tests / sub))
import numpy as np

from maple_syrup import floating_held_context as fh

big = lambda: 10**12
if mode == "export":
    import hbuild as hb

    ov = {"theta_33_override": np.full((3, 3), np.nan)}
    ov["theta_33_override"][1, 1] = 0.3
    spec = hb.pref_spec(3, 3, kind="cell", species=1, rate="inf")
    vert, owner, bed = hb.make(backend=backend, mode="texture_ks", overrides=ov, spec=spec)
    rec = fh.export_held_context(vert, spec, available_bytes_fn=big)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(rec.meta(), sort_keys=True))
    names = {}
    for k, v in rec.arrays().items():
        (out / f"{k}.bin").write_bytes(v.tobytes())
        names[k] = [str(v.dtype), list(v.shape)]
    (out / "arrays.json").write_text(json.dumps(names))
    e = hb.expected_of(vert)
    (out / "expected.json").write_text(json.dumps({"config_sha256": e.config_sha256, "root_column_sha256": e.root_column_sha256, "matrix_op_sha256": e.matrix_op_sha256, "pref_sha256": e.pref_sha256}))
    print(json.dumps({"mode": "export", "identities": vars(e) if hasattr(e, "__dict__") else {}}))
else:
    from maple_syrup import native_floating_soil as nfs
    from maple_syrup import soil_texture_properties as stp

    def boom(*a, **k):
        raise AssertionError("must not run in the restore process")

    stp.estimate_soil_texture_properties = boom
    for name in ("_read_texture", "initialize_native_floating_soil", "native_source_witness", "stage_native_refresh", "read_canonical_depth", "NativeBed"):
        setattr(nfs, name, boom)
    meta = json.loads((out / "meta.json").read_text())
    names = json.loads((out / "arrays.json").read_text())
    arrays = {k: np.frombuffer(bytearray((out / f"{k}.bin").read_bytes()), dtype=np.dtype(dt)).reshape(shape).copy() for k, (dt, shape) in names.items()}
    ex = json.loads((out / "expected.json").read_text())
    rec = fh.record_from_parts(meta, arrays, available_bytes_fn=big)
    ctx = fh.reconstruct_held_context(rec, fh.ExpectedHeldIdentities(**ex), available_bytes_fn=big)
    print(json.dumps({"mode": "restore", "config": ctx.config.sha256, "column": ctx.column.sha256, "matrix": ctx.matrix_op.sha256, "pref": ctx.preferential.sha256, "overrides": sorted(ctx.config.overrides),
                      "override_bytes": ctx.config.overrides["theta_33_override"].tobytes().hex(), "has_water": hasattr(ctx, "water")}))
