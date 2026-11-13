"""Fill run.template.yaml: `python fill_config.py OUT_DIR --masy-root R --syrup-root R --leaf-root R --native-target T [--declared-by NAME]`.
Copies this example's inputs to OUT_DIR (outside the MASY tree) and writes run.yaml; record the two manifests next with the `manifest` command (see the docs). Stdlib only."""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = ("case", "scenario", "layered_water.yaml", "catalog.json", "plants.csv", "plants.metadata.json")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("out")
    for k in ("masy-root", "syrup-root", "leaf-root", "native-target"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--declared-by", default="example operator")
    a = p.parse_args(argv)
    out = Path(a.out).resolve()
    if out.exists():
        sys.exit(f"{out} already exists; nothing written")
    out.mkdir(parents=True)
    (out / "out").mkdir()
    for name in INPUTS:
        s = HERE / name
        shutil.copytree(s, out / name) if s.is_dir() else shutil.copy2(s, out / name)
    text = (HERE / "run.template.yaml").read_text()
    for key, val in (("MASY_ROOT", a.masy_root), ("SYRUP_ROOT", a.syrup_root), ("LEAF_ROOT", a.leaf_root), ("NATIVE_TARGET", a.native_target), ("DECLARED_BY", a.declared_by)):
        text = text.replace(f"@{key}@", str(Path(val).resolve()) if key != "DECLARED_BY" else val)
    (out / "run.yaml").write_text(text)
    print(f"wrote {out / 'run.yaml'}")


if __name__ == "__main__":
    main()
