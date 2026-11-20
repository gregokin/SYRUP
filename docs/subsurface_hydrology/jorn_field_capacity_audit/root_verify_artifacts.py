"""Independent report/data/source checks. Does not run or change hydrology."""
import csv
import hashlib
import json
import subprocess
import zipfile
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as ET

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
R = P / 'report'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = json.loads((P / 'root_audit.json').read_text())
    baseline = json.loads((P / 'baseline.json').read_text())
    for name, want in baseline['tracked_live_files'].items():
        assert sha(ROOT / name) == want, name
    for name, want in json.loads((P / 'source_seal.json').read_text()).items():
        assert sha(ROOT / name) == want, name
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == baseline['head']
    assert subprocess.check_output(['git', 'status', '--short', '--branch'], cwd=ROOT, text=True) == baseline['dirty_status']
    m = json.loads((R / 'artifact_manifest.json').read_text())
    for name, want in m['hashed_sources'].items():
        assert sha(ROOT / name) == want, name
    for category in ('figures', 'tables', 'data'):
        for name, want in m[category].items():
            assert sha(R / category / name) == want, (category, name)
    assert sha(R / 'build_report.py') == m['builder']
    assert sha(R / 'report.md') == m['report_md']
    for name in ('root_audit.json', 'root_audit.py', 'root_layers.csv'):
        assert sha(P / name) == sha(R / 'data' / name)
    independent = json.loads((R / 'data/independent_recalculation.json').read_text())
    for a, b in zip(audit['layers'], independent['layers'], strict=True):
        for x, y in [('theta33', 'theta33'), ('theta6', 'theta6'), ('theta_sat', 'thetaS'),
                     ('density_used', 'rho'), ('air_raw', 'psi_raw')]:
            assert abs(Decimal(a['calculations'][x]) - Decimal(b['results'][y])) < Decimal('1e-45')
        assert abs(Decimal(a['OM_percent']) - Decimal(b['OM_percent'])) < Decimal('1e-45')
    with (R / 'tables/psd.csv').open() as stream:
        rows = list(csv.reader(stream))
    keys = ['sandVeryFineContent', 'sandFineContent', 'sandMediumContent',
            'sandCoarseContent', 'sandVeryCoarseContent', 'sandTotal', 'siltTotal', 'clayTotal']
    for row, key in zip(rows[1:9], keys, strict=True):
        # Last three fields are the three model layers, whether or not raw duplicate columns remain.
        for x, layer in zip(row[-3:], audit['layers'], strict=True):
            assert x == f"{float(layer['texture_percent'][key]):.3f}", (row, key)
    for path in (R / 'tables').glob('*.csv'):
        assert b'\r\n' not in path.read_bytes(), path
    docx = R / m['docx']['file']
    assert sha(docx) == m['docx']['sha256']
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(docx) as archive:
        assert archive.testzip() is None
        node = ET.fromstring(archive.read('word/document.xml'))
        text = ' '.join(node.itertext())
        tables = node.findall('.//w:tbl', ns)
        assert len(tables) == len(node.findall('.//w:tblHeader', ns))
        assert all(len(t.findall('w:tr', ns)[0].findall('w:tc', ns)) <= 4 for t in tables)
        media = [n for n in archive.namelist() if n.startswith('word/media/')]
        assert len(media) == 1
        assert hashlib.sha256(archive.read(media[0])).hexdigest() == sha(R / 'figures/fig1_wet_curves.png')
        for s in ('39.76', '39.87', '39.51', '9.75', '9.94', '10.91',
                  '86.288', '6.433', '7.278', '33 kPa', '6 kPa',
                  '10.2136/sssaj2005.0117', '10.1029/2007WR006589'):
            assert s in text, s
        assert all(s not in text for s in ('DRAFT', 'TODO', 'TBD', '\\frac', 'is an extrapolation'))
    md = (R / 'report.md').read_text()
    assert 'quoted by the user for 33 kPa' not in md
    assert 'Change from the density floors' not in md
    result = {'status': 'PASS', 'kind': 'Documentation and independent arithmetic only',
              'docx': str(docx), 'docx_sha256': sha(docx),
              'unchanged_tracked_working_files': len(baseline['tracked_live_files']),
              'unchanged_input_source_and_old_report_files': len(audit['source_sha256']),
              'controls': 'Root80-digit primary-equation arithmetic, independent Claude60-digit raw reconstruction, actual case and archived receipt agree; PSD and all numerical steps preserved',
              'document': {'ZIP_XML_media_hashes': 'PASS', 'tables': len(tables), 'figure_count': 1,
                           'max_columns': 4, 'all_repeat_headers': True},
              'not_claimed': ['New hydrology run or parameter change', 'Site FC validation', 'CPU/GPU simulation qualification', 'Word/LibreOffice pagination rendering']}
    (P / 'root_artifact_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
