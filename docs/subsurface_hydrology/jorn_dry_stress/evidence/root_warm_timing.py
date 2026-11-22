"""Warm native column-call timing: includes operator preparation, arrays and daily updates; excludes input preparation/scoring/I/O. Not kernel-only timing."""
from pathlib import Path
import json,time,numpy as np
from benchmarks.subsurface_validation import drydown_driver as dd
P=Path(__file__).resolve().parent;J=lambda p:json.loads(Path(p).read_text());idx=J(P/'config_paths.json');preps={};cfgs={}
for k,v in idx['cases'].items():
 cfg,_=dd.load_config(v['path']);cfgs[k]=cfg;w=v['window']
 if w not in preps:preps[w]=dd.prepare_inputs(cfg)
first=next(iter(idx['cases']));cfg=cfgs[first];prep=preps[idx['cases'][first]['window']]
t=time.perf_counter();dd.run_column(cfg,prep,cfg['scenarios'][0],'numba');warmup=time.perf_counter()-t
rows=[]
for key,v in idx['cases'].items():
 c=cfgs[key];prep=preps[v['window']];t=time.perf_counter();r=dd.run_column(c,prep,c['scenarios'][0],'numba');elapsed=time.perf_counter()-t;ref=next(iter(J(Path(v['result_dir'])/'result.json')['runs'].values()))
 for name,a in r['trajectory'].items():np.testing.assert_array_equal(a,np.asarray(ref['trajectory'][name]))
 assert r['final_storage_m']==ref['final_storage_m']
 rows.append({'key':key,'days':prep['n_days'],'warm_native_column_call_s':elapsed,'exact_trajectory_to_primary':True})
ans={'status':'PASS','scope':__doc__,'backend':'numba','warmup_call_s':warmup,'rows':rows,'extra_native_window_calls':7,'not_a_controlled_machine_speed_benchmark':True}
(P/'root_warm_timing.json').write_text(json.dumps(ans,indent=2)+'\n');print(json.dumps(ans,indent=2))
