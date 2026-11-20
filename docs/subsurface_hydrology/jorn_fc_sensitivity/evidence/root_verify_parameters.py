"""Independent physical/configuration comparison to the accepted unchanged SR33K1 case."""
from pathlib import Path
import json,hashlib,copy,datetime
P=Path(__file__).resolve().parent; C=P/'analysis_candidate'; sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();J=lambda p:json.loads(Path(p).read_text())
basepath=P.parent/'jorn_sr33_160day_20261120/study/k1/case_numba.json'
assert sha(basepath)=='f024054f5198ec8358c95f7a33683e6228084e6af246f9790cde586d437f3d2a'
base=J(basepath);paths=J(P/'config_paths.json');expected={f'fc{fc:02d}_s{label}':(fc/100.0,mult) for label,mult in [('1',1.0),('1p5',1.5)] for fc in range(9,16)}
assert set(paths['cases'])==set(expected)
records=[]
for key,(fc,mult) in expected.items():
 info=paths['cases'][key]
 for variant,backend,dt in [('numba','numba',1800),('numpy','numpy',1800),('numba_dt900','numba',900)]:
  v=info['variants'][variant];p=Path(v['path']);assert p.is_absolute() and sha(p)==v['sha256'];cfg=J(p)
  assert cfg['layers']['theta_field_capacity']==[fc]*3
  assert cfg['layers']['theta_evaporation_stress']==.2017*mult
  assert cfg['physics']['evaporation_factor']==1.0
  wanted=copy.deepcopy(base);wanted['layers']['theta_field_capacity']=[fc]*3;wanted['layers']['theta_evaporation_stress']=.2017*mult;wanted['controls']={'backend':backend};wanted['storm']={'max_dt_s':dt}
  # Names/provenance change by explicit declaration; no other physical or observational differences.
  for x in (wanted,cfg):x.pop('name');x.pop('provenance')
  assert cfg==wanted,(key,variant)
  rc=J(v['runner_config']);assert rc['cwd']==str(C)
  assert str(C/'src') in rc['prefix'] and str(C) in rc['prefix']
  assert rc['sources'][1]['root']==str(C/'src')
  assert rc['sources'][1]['sha256']=='4e38a472e268a56491b788d5b1271cc3c3da23bb1cc3221a8c59d90f3cc864d2'
  assert rc['sources'][0]['sha256']=='fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e'
  assert sha(Path(v['runner_config']).parent/'root_release_verified_bootstrap.py')==sha(P/'root_release_verified_bootstrap.py')
 records.append({'key':key,'field_capacity_each_layer':fc,'E':1.0,'stress_multiplier':mult,'stress':.2017*mult,'case_sha256':sha(info['variants']['numba']['path'])})
assert all(sha(C/k)==v for k,v in J(P/'root_existing_helper_manifest.json').items())
result={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'physical_deltas':'Only uniform all-three-layer FC and requested exact rounded stress; E1, K/Ks1x and all other physical/weather/observational controls unchanged','old_helper_hashes_exact':True,'cases':records}
(P/'root_parameter_verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS: all42configs exact requested14-combination design and genuine source/import/launcher guards')
