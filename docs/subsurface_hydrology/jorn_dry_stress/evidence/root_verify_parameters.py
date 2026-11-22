"""Independent case/template/default delta audit; no window integration."""
from pathlib import Path
import json,hashlib,copy,subprocess
P=Path(__file__).resolve().parent;C=P/'analysis_candidate';J=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();idx=J(P/'config_paths.json');preset=J(C/'cases/subsurface_validation/jorn_defaults.json');q=preset['parameters'];assert q=={'theta_field_capacity_all_layers':.1,'theta_evaporation_stress':.30255,'evaporation_factor':1.,'diffusion_coefficient':.0010239461376024856,'conductivity_multiplier':1.,'interface_conductivity_method':'harmonic'}
oldagents=subprocess.check_output(['git','show','99465040d947ec6ff377003a88f3b2636d923f15:AGENTS.md'],cwd=C);assert (C/'AGENTS.md').read_bytes().startswith(oldagents)
expected={f'{w}_{label}':(w,s) for w in ['JORN4_A','JORN4_B','MOAB2_C'] for label,s in [('s1',.2017),('s1p5',.30255)]};assert set(idx['cases'])==set(expected);records=[]
moab=[.13932602200462846,.15622473499486667,.15622473499486667];ref=J(P/'root_moab_fc_reference.json')['MOAB_implied_density_adjusted_theta33'];assert all(abs(a-b)<1e-15 for a,b in zip(moab,ref))
for key,(w,stress) in expected.items():
 v=idx['cases'][key];c=J(v['path']);old=J(C/f'docs/subsurface_hydrology/joint_d_stress_calibration/data/actual_cpu/configs/harmonic_{w}.json');wanted=copy.deepcopy(old)
 assert sha(v['path'])==v['sha256'];assert c['layers']['theta_evaporation_stress']==stress
 fc=[.1]*3 if w.startswith('JORN') else moab
 assert c['layers']['theta_field_capacity']==fc
 wanted['layers']['theta_field_capacity']=fc;wanted['layers']['theta_evaporation_stress']=stress;wanted['scenarios'][0]['diffusion_coefficient']=.0010239461376024856;wanted['scenarios'][0]['name']=c['scenarios'][0]['name'];wanted['pins']['expected_sources']['maple_syrup']={'package_sha256':'4e38a472e268a56491b788d5b1271cc3c3da23bb1cc3221a8c59d90f3cc864d2','files':180}
 for x in (c,wanted):x.pop('name');x.pop('provenance')
 assert c==wanted,(key,'unrequested physical/input/control delta')
 rc=J(v['runner_config']);assert rc['cwd']==str(C) and rc['sources'][1]['root']==str(C/'src');assert sha(Path(v['runner_config']).parent/'root_release_verified_bootstrap.py')==sha(P/'root_release_verified_bootstrap.py')
 records.append({'key':key,'stress':stress,'field_capacity':fc,'case_sha256':v['sha256'],'declared_template_deltas_only':True})
assert all(sha(C/k)==h for k,h in J(P/'root_existing_helper_manifest.json').items())
r={'status':'PASS','cases':records,'preset_sha256':sha(C/'cases/subsurface_validation/jorn_defaults.json'),'agents_append_only':True,'current_native_source':'180/4e38','MAPLE':'412/fbc','old_helper_hashes_unchanged':True,'MOAB_native_FC33_matches_independent_retention_inversion':True}
(P/'root_parameter_verification.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS:6cases and scopeddefaults exact; only declared deltas; oldhelpers and AGENTS prefix preserved')
