"""Independent static Word/table/media/data verification; no Office render claim."""
from pathlib import Path
import json,hashlib,csv,zipfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent; D=P/'analysis_candidate/docs/subsurface_hydrology/jorn_fc_sensitivity';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();J=lambda p:json.loads(Path(p).read_text())
seal=J(P/'root_run_seal.json')['files']
for f,v in seal.items():assert sha(f)==v['sha256'] and Path(f).stat().st_size==v['bytes'],f
for f,h in J(P/'run_input_manifest.json').items():assert sha(f)==h,f
for r in J(P/'config_paths.json')['cases'].values():
 for v in r['variants'].values():assert sha(v['path'])==v['sha256']
manifest=J(D/'artifact_manifest.json')
for name,h in manifest['files'].items():assert sha(D/name)==h,name
path=D/'SYRUP_JORN_160day_uniform_FC_sensitivity_20261120.docx';assert sha(path)==manifest['docx']['sha256']
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};norm=lambda x:' '.join(x.split())
with zipfile.ZipFile(path) as z:
 assert z.testzip() is None
 doc=ET.fromstring(z.read('word/document.xml'));text='\n'.join(''.join(v.text or '' for v in p.findall('.//w:t',ns)) for p in doc.findall('.//w:p',ns));tables=[]
 for t in doc.findall('.//w:tbl',ns):
  rows=[[norm(''.join(v.text or '' for v in c.findall('.//w:t',ns))) for c in r.findall('w:tc',ns)] for r in t.findall('w:tr',ns)]
  assert max(map(len,rows))<=7,('too many columns',rows[0]);assert t.find('w:tr/w:trPr/w:tblHeader',ns) is not None;tables.append(rows)
 csvs={f.name:[[norm(x) for x in row] for row in csv.reader(f.open())] for f in (D/'tables').glob('*.csv')};matches=[]
 for table in tables:
  found=[k for k,rows in csvs.items() if rows==table];assert found,('Word table not exact CSV',table[0]);matches.append(found[0])
 media=[k for k in z.namelist() if k.startswith('word/media/')];assert len(media)>=4
 assert {hashlib.sha256(z.read(k)).hexdigest() for k in media}=={sha(f) for f in (D/'figures').glob('*.png')}
 for literal in ('0.2017','0.30255','25.088','DP1.00096.001','2024-03-15','2024-06-25','282'):
  assert literal in text,('missing important identity/date',literal)
 assert 'GPU' in text and '33' in text and '160' in text
out={'status':'PASS','docx':str(path),'sha256':sha(path),'tables':len(tables),'table_csv_matches':matches,'media_png_exact':len(media),'max_table_columns':max(len(t[0]) for t in tables),'all_report_source_seals_unchanged':len(seal),'all_case_and_execution_input_hashes_unchanged':True,'manifest_verified':True,'scope':__doc__}
(P/'root_artifact_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
