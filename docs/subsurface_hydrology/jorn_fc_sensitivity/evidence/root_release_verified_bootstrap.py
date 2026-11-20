from pathlib import Path
import sys,os,json,importlib
cfg=json.loads(Path(sys.argv[1]).read_text());out=Path(cfg['output']);prefix=cfg['prefix']
for q in cfg['required_files']:assert Path(q).is_file(),q
sys.path[:0]=prefix;os.environ['PYTHONPATH']=os.pathsep.join(prefix)
from maple_syrup.provenance import source_tree_digest
def check():
 rows=[]
 for spec in cfg['sources']:
  mod=importlib.import_module(spec['module']);path=Path(mod.__file__).resolve();assert path.is_relative_to(Path(spec['root']).resolve()),(spec['module'],str(path),spec['root'])
  got=source_tree_digest(path.parent);assert (got.digest_sha256,got.file_count)==(spec['sha256'],spec['files']),(spec['module'],got.digest_sha256,got.file_count)
  rows.append(dict(spec,actual_file=str(path),actual_sha256=got.digest_sha256,actual_files=got.file_count))
 for spec in cfg.get('external_modules', []):
  import hashlib
  from importlib.util import find_spec
  origin=find_spec(spec['module']).origin;path=Path(origin).resolve()
  assert path.is_relative_to(Path(spec['root']).resolve()),(origin,spec)
  b=path.read_bytes();got=hashlib.sha256(b).hexdigest()
  assert (got,len(b))==(spec['sha256'],spec['bytes']),(origin,got,len(b))
  rows.append(dict(spec,actual_file=str(path),actual_sha256=got,actual_bytes=len(b)))
 return rows
(out/'sources_before.json').write_text(json.dumps(check(),indent=2)+chr(10))
try:
 if cfg.get('probe_file'):
  import runpy
  runpy.run_path(cfg['probe_file'],run_name='__main__');rc=0
 else:
  import pytest
  rc=pytest.main(['-o','pythonpath=','-p','no:cacheprovider','-o','faulthandler_timeout=180']+cfg['pytest_args'])
finally:
 (out/'sources_after.json').write_text(json.dumps(check(),indent=2)+chr(10))
raise SystemExit(rc)
