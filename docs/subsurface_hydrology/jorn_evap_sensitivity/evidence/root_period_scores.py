"""Read-only period/support diagnostics; no transpiration is introduced."""
from pathlib import Path
from fractions import Fraction as F
import json,math
P=Path(__file__).resolve().parent
KEYS=('baseline','e0p8','e0p6','e0p4','s1p5','s2','s2p5','s3')
records=[]
for key in KEYS:
 r=json.loads((P/'study'/key/'actual_numba/result.json').read_text());out={}
 for period in ('before_March1','March1_onward'):
  by={}
  for scope,inds in [('upper_two',[0,1]),('all_three',[0,1,2]),('501',[0]),('502',[1]),('503',[2])]:
   errors={'raw':[],'aligned':[]}
   for row in r['rows']:
    if (row['date']<'2024-03-01')!=(period=='before_March1'):continue
    for j in inds:
     o=row['obs_theta'][j]
     if o is None:continue
     m=row['layer_theta'][j];m0=r['plan']['initial']['layer_theta'][j];o0=r['plan']['initial']['point_theta'][j]
     errors['raw'].append(F.from_float(m-o));errors['aligned'].append(F.from_float((m-m0)-(o-o0)))
   by[scope]={k:{'n':len(v),'rmse_pp':100*math.sqrt(float(sum(e*e for e in v)/len(v))),'bias_pp':100*float(sum(v)/len(v))} for k,v in errors.items()}
  out[period]=by
 endpoints=[]
 for j in range(3):
  row=next(x for x in reversed(r['rows']) if x['obs_theta'][j] is not None)
  endpoints.append({'level':['501','502','503'][j],'close_utc':row['nominal_close_utc'],'model_theta':row['layer_theta'][j],'observed_theta':row['obs_theta'][j]})
 records.append({'key':key,'periods':out,'endpoints':endpoints})
result={'status':'PASS','split_rule':"Nominal local-day labels row.date <2024-03-01 versus >=2024-03-01; each observation is at the close following that nominal day. No modeled T or calendar rule is changed.",'scope':__doc__,'records':records}
(P/'root_period_scores.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([{'key':x['key'],'early_top2_rmse_raw_pp':x['periods']['before_March1']['upper_two']['raw']['rmse_pp'],'early_top2_rmse_aligned_pp':x['periods']['before_March1']['upper_two']['aligned']['rmse_pp'],'late_top2_bias_raw_pp':x['periods']['March1_onward']['upper_two']['raw']['bias_pp']} for x in records],indent=2))
