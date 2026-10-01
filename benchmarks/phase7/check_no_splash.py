"""Compile original/patched routing routines; exercise dry/wet branch selection."""
import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from prepare_mahleran import ROUTE, inventory, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.prepared / 'benchmark_manifest.json').read_text())
    reference, prepared = Path(manifest['reference_root']), args.prepared.resolve()
    if inventory(reference) != manifest['original_sha256'] or inventory(prepared) != manifest['prepared_sha256']:
        raise ValueError('reference or prepared copy differs from manifest')
    args.output.mkdir(parents=True, exist_ok=False)
    compiler = os.environ.get('MAPLE_SYRUP_GFORTRAN', 'gfortran')
    flags = shlex.split(os.environ.get('MAPLE_SYRUP_GFORTRAN_FLAGS', ''))
    flags += ['-std=legacy', '-ffree-line-length-none', '-fcheck=all', '-O0']
    ldflags = shlex.split(os.environ.get('MAPLE_SYRUP_GFORTRAN_LDFLAGS', ''))
    driver = Path(__file__).with_name('no_splash_branch_driver.f90').resolve()
    results = {}
    for label, root in (('original', reference), ('no_splash', prepared)):
        build = (args.output / label).resolve()
        build.mkdir()
        sources = [root / 'src/Program_Control/shared_data.f90', root / 'src/Program_Control/parameters_from_xml.f90',
                   root / ROUTE, driver]
        command = [compiler, *flags, *map(str, sources), *ldflags, '-o', str(build / 'branch_check')]
        (build / 'command.json').write_text(json.dumps(command, indent=2)+'\n')
        compiled = subprocess.run(command, cwd=build, capture_output=True, text=True, timeout=60, check=False)
        (build / 'compile.stdout').write_text(compiled.stdout)
        (build / 'compile.stderr').write_text(compiled.stderr)
        compiled.check_returncode()
        run = subprocess.run([str(build / 'branch_check')], cwd=build, capture_output=True, text=True, timeout=15, check=False)
        (build / 'run.stdout').write_text(run.stdout)
        (build / 'run.stderr').write_text(run.stderr)
        run.check_returncode()
        rows = [[float(v) for v in line.split()] for line in run.stdout.splitlines()]
        if len(rows) != 2:
            raise ValueError('unexpected branch test output')
        results[label] = rows
    assert results['original'][0] == [0,1,1,0,2,2,1]
    assert results['no_splash'][0] == [0,0,0,0,0,0,1]  # stale rates cleared, existing load retained
    assert results['original'][1] == results['no_splash'][1] == [1,1,0,6,2,.5,2.5]
    assert inventory(reference) == manifest['original_sha256']
    assert inventory(prepared) == manifest['prepared_sha256']
    report = {'status': 'passed', 'scope': 'actual route routine with stubbed physics; branch selection, stale rates and retained existing dry-cell load; NOT a full model or physics benchmark',
              'columns': ['wet','rain_calls','splash_calls','diffuse_calls','detach','deposit','mobile_after'],
              'results': results, 'driver_sha256': sha(driver), 'reference_unchanged': True,
              'prepared_unchanged': True, 'compiler': compiler, 'flags': flags}
    (args.output / 'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
