"""Prepare, but do not run, a source-bound Plot1 benchmark without dry splash."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROUTE = Path('src/Subroutines_Sediment/route_sediment_xml.f90')
EXPECTED_ROUTE = 'c4ff42a8a0f5acf86708199082ebec348483b0583bc804b052c2ba025e44a7a4'
EXPECTED_XML = '6f298fe123b7c957acd4583c8baec13c1c788236427933757afd83558e56e1e3'
OLD = '''         elseif (r2 (im, jm).gt.0.0d0) then
            call raindrop_detachment
            call splash_transport   
'''
NEW = '''         elseif (r2 (im, jm).gt.0.0d0) then
! SYRUP BENCHMARK ONLY: no dry-cell splash or its associated pickup.
! Wet-cell raindrop detachment above is retained without modification.
! Clear rate outputs explicitly; never clear d_soil/q_soil mobile inventory.
            detach_soil (:, im, jm) = 0.0d0
            depos_soil (:, im, jm) = 0.0d0
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    paths = [root / 'mahleran_input.xml', root / 'Makefile']
    for directory in ('src', 'Input/input_p1', 'nbproject'):
        paths.extend(p for p in (root / directory).rglob('*') if p.is_file())
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}


def audit_cases(root):
    records = []
    paths = sorted(set(root.glob('mahleran_input*.xml')) | set((root / 'Input').rglob('mahleran_input*.xml')))
    for path in paths:
        raw = path.read_text()
        error = None
        try:
            tree = ET.fromstring(raw)
        except ET.ParseError as exc:
            tree, error = None, str(exc)
        def value(name, tree=tree, raw=raw):
            if tree is None:
                match = re.search(r'<' + re.escape(name) + r'\s+value="([^"]*)"', raw)
                return None if match is None else match.group(1)
            el = tree.find(name)
            return None if el is None else el.get('value', el.text)
        records.append({'path': str(path.relative_to(root)), 'sha256': sha(path), 'format': 'XML',
                        'version': value('version'), 'runtype': value('runtype'), 'xml_parse_error': error,
                        'update_topography': value('update_topography'),
                        'topography_update_interval_steps': value('topography_update_interval'),
                        'flow_routing_solution_method': value('flow-routing_solution_method')})
    for path in sorted((root / 'Input').rglob('mahleran_input.dat')):
        records.append({'path': str(path.relative_to(root)), 'sha256': sha(path), 'format': 'legacy DAT',
                        'header': path.read_text().splitlines()[0], 'update_topography': None,
                        'interpretation': 'No explicit topographic-update setting identified; not an executable XML configuration of the current application.'})
    return records


def prepare(root, output):
    root, output = root.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('new output outside the MAHLERAN reference tree required')
    if sha(root / ROUTE) != EXPECTED_ROUTE or sha(root / 'mahleran_input.xml') != EXPECTED_XML:
        raise ValueError('reference source/XML changed; re-audit before patching')
    before = inventory(root)
    route = (root / ROUTE).read_bytes().decode()
    if route.count(OLD) != 1:
        raise ValueError('expected unique dry-rain branch not found')
    patched = route.replace(OLD, NEW)
    if patched.lower().count('call raindrop_detachment') != 1 or 'call splash_transport' in patched.lower():
        raise ValueError('unexpected wet/dry branch call inventory')
    xml = (root / 'mahleran_input.xml').read_bytes().decode()
    tree = ET.fromstring(xml)
    for name, expected in (('runtype', 'event'), ('update_topography', 'n'), ('flow-routing_solution_method', '5')):
        if tree.find(name).get('value') != expected:
            raise ValueError(f'{name} does not match audited fixed-terrain non-MiC case')
    for old, new in (('.\\Input\\input_p1\\', './Input/input_p1/'), ('.\\Output\\', './Output/')):
        if xml.count(old) != 1:
            raise ValueError(f'expected path binding missing: {old}')
        xml = xml.replace(old, new)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        for directory in ('src', 'Input/input_p1', 'nbproject'):
            shutil.copytree(root / directory, stage / directory)
        shutil.copy2(root / 'Makefile', stage / 'Makefile')
        (stage / ROUTE).write_bytes(patched.encode())
        (stage / 'mahleran_input.xml').write_bytes(xml.encode())
        (stage / 'Output').mkdir()
        patch = ''.join(difflib.unified_diff(route.splitlines(True), patched.splitlines(True),
                                            fromfile='a/' + str(ROUTE), tofile='b/' + str(ROUTE)))
        (stage / 'no_splash.patch').write_text(patch)
        after = inventory(stage)
        changed = [name for name in before if before[name] != after[name]]
        if set(changed) != {str(ROUTE), 'mahleran_input.xml'} or inventory(root) != before:
            raise ValueError('unexpected source differences or changed reference')
        report = {'status': 'prepared, no storm executed; independent review pending',
                  'reference_root': str(root), 'copy_root': str(output), 'original_sha256': before,
                  'prepared_sha256': after, 'changed_files': changed,
                  'source_change': 'disable only the dry-cell rain detachment/splash branch; explicit zero rates, no mobile inventory reset',
                  'configuration_change': 'only Windows-to-POSIX relative input/output paths; existing topography=n, routing=5 retained',
                  'syrup_comparison_contract': {'elevation': 'fixed', 'routing': 'fixed',
                      'sediment_holdings': 'evolve conservatively in actual MAPLE', 'rain_assisted_wet_detachment': 'enabled',
                      'direct_splash': 'disabled', 'status': 'required for future comparison; not implemented by this preparation'},
                  'case_audit': audit_cases(root), 'full_application_build': 'not attempted',
                  'limits': ['event mode only; marker splash paths unchanged and out of scope',
                             'legacy sediment bookkeeping, timestep conventions and water solver unchanged',
                             'no claim of full MAHLERAN/SYRUP sediment equivalence']}
        (stage / 'benchmark_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('/home/okin/MAHLERAN'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.reference, args.output)
    print(json.dumps({'copy_root': report['copy_root'], 'changed_files': report['changed_files'],
                      'case_audit': report['case_audit']}, indent=2))


if __name__ == '__main__':
    main()
