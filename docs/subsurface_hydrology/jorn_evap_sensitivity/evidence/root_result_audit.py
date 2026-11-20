"""Read-only independent checks of raw observations, scores and layer budgets."""
from pathlib import Path
from fractions import Fraction as F
import json,math,hashlib
import numpy as np
from maple_syrup.conservation import volume_roundoff_bound_m3
P=Path(__file__).resolve().parent
J=lambda p:json.loads(p.read_text())
audit=J(P/'root_independent_inputs.json')
old=J(P.parent/'jorn_sr33_160day_20261120/study/k1/actual_numba/result.json')
keys=['water_m','layer_theta','physical_t_s','rain_m3','intake_m3','evaporation_m3','wetting_J_m','cumulative_matrix_intake_m','drainage_m3','runoff_pool_m3','diffusion_m3','percolation_m3']
records=[]
SPECS=[('baseline',1.0,1.0),('e0p8',.8,1.0),('e0p6',.6,1.0),('e0p4',.4,1.0),('s1p5',1.0,1.5),('s2',1.0,2.0),('s2p5',1.0,2.5),('s3',1.0,3.0)]
for key,E,stress_multiplier in SPECS:
 factor=key
 q=P/'study'/key;b=J(q/'actual_numba/result.json');n=J(q/'actual_numpy/result.json');fine=J(q/'actual_numba_dt900/result.json');c=J(q/'case_numba.json')
 assert b['complete'] and b['days_done']==160 and b['physical_end_t_s']==audit['physical_end_seconds']
 assert b['plan']['initial']==old['plan']['initial']
 initial_bound=volume_roundoff_bound_m3(64,max(audit['initial_water_m']))  # binary64 conversion + bounded trapezoidal interpolation/integration; same MAPLE conservation method
 assert np.max(np.abs(np.array(b['plan']['initial']['layer_water_m'])-audit['initial_water_m']))<=initial_bound
 parity={k:float(np.max(np.abs(np.array([r[k] for r in b['rows']])-np.array([r[k] for r in n['rows']])))) for k in keys}
 assert all(v<=1e-14 for v in parity.values()),parity
 for row,base in zip(b['rows'],old['rows']):
  for k in ['date','nominal_close_utc','physical_t_s','obs_theta','pet_mm_day','rain_m3']:assert row[k]==base[k],(key,k)
 errors={kind:[] for kind in ['raw','offset_aligned','persistence_raw','persistence_offset_aligned']}
 scores={}
 for i,v in enumerate(['501','502','503']):
  expected=audit['observations'][v]['day_end_values'];assert [r['obs_theta'][i] for r in b['rows']]==expected
  o0=float(audit['observations'][v]['initial_raw']);m0=b['plan']['initial']['layer_theta'][i];by={k:[] for k in errors}
  for row in b['rows']:
   if row['obs_theta'][i] is None:continue
   model,obs=row['layer_theta'][i],row['obs_theta'][i]
   vals={'raw':model-obs,'offset_aligned':(model-m0)-(obs-o0),'persistence_raw':o0-obs,'persistence_offset_aligned':-(obs-o0)}
   for kind,e in vals.items():by[kind].append(F.from_float(e));errors[kind].append(F.from_float(e))
  scores[v]={}
  for kind,es in by.items():
   rmse=math.sqrt(float(sum(e*e for e in es)/len(es)));got=b['scores'][kind][v];assert got['n']==len(es) and math.isclose(got['rmse'],rmse,rel_tol=2e-14)
   scores[v][kind]={'n':len(es),'rmse_pp':100*rmse,'bias_pp':100*float(sum(es)/len(es))}
 pooled={}
 for kind,es in errors.items():
  rmse=math.sqrt(float(sum(e*e for e in es)/len(es)));got=b['scores']['pooled_'+kind];assert got['n']==282 and math.isclose(got['rmse'],rmse,rel_tol=2e-14);pooled[kind]={'n':282,'rmse_pp':100*rmse}
 previous=np.array(b['plan']['initial']['layer_water_m']);maxlayer=0.;gravity=np.zeros(2);diff=np.zeros(2)
 for row in b['rows']:
  w=np.array(row['water_m']);perc=np.array(row['percolation_m3']);dx=np.array(row['diffusion_m3']);bound=row['closure_bound_m3']
  assert row['surface_return_m3']==row['runoff_pool_m3']==row['room_rejected_m3']==0
  predicted=np.array([row['intake_m3']-row['evaporation_m3']-perc[0]-dx[0],perc[0]-perc[1]+dx[0]-dx[1],perc[1]+dx[1]-row['drainage_m3']])
  err=float(np.max(np.abs((w-previous)-predicted)));assert err<=bound,(key,row['date'],err,bound);maxlayer=max(maxlayer,err);previous=w;gravity+=perc;diff+=dx
  assert abs(row['closure_residual_m3'])<=bound and abs(row['rain_closure_residual_m3'])<=row['rain_closure_bound_m3']
 closure=b['cumulative_closure'];assert abs(closure['soil_residual_m3'])<=closure['soil_bound_m3']
 ledger=b['ledger'];assert math.isclose(ledger['rain_m3']*1000,25.088,abs_tol=1e-12) and ledger['intake_m3']==ledger['rain_m3']
 delta=np.array([r['layer_theta'] for r in b['rows']])-np.array([r['layer_theta'] for r in fine['rows']])
 rec={'key':key,'factor':factor,'E':E,'stress_multiplier':stress_multiplier,'stress':c['layers']['theta_evaporation_stress'],'scores':scores,'pooled':pooled,'budgets_mm':{'initial_storage':1000*closure['storage_initial_m3'],'final_storage':1000*closure['storage_final_m3'],'rain':1000*ledger['rain_m3'],'intake':1000*ledger['intake_m3'],'runoff':1000*closure['total_runoff_m3'],'bottom_drainage':1000*ledger['drainage_m3'],'evaporation':1000*ledger['evaporation_m3'],'gravity_transfers':(1000*gravity).tolist(),'net_diffusive_transfers_down_positive':(1000*diff).tolist()},'backend_max_absolute_difference':parity,'dt900_max_theta_difference_pp_by_layer':(100*np.max(abs(delta),axis=0)).tolist(),'max_layer_budget_error_m':maxlayer,'closure':closure,'timing':J(q/'actual_numba/timing.json'),'final_theta':b['rows'][-1]['layer_theta'],'initial_theta':b['plan']['initial']['layer_theta'],'peak_theta':[max(r['layer_theta'][i] for r in b['rows']) for i in range(3)]}
 (q/'root_result_audit.json').write_text(json.dumps({'status':'PASS',**rec},indent=2)+'\n');records.append(rec)
ans={'status':'PASS','scope':__doc__,'reference_SR33_K1_E1':{'pooled_raw_rmse_pp':100*old['scores']['pooled_raw']['rmse'],'pooled_aligned_rmse_pp':100*old['scores']['pooled_offset_aligned']['rmse'],'E_mm':1000*old['ledger']['evaporation_m3'],'bottom_drainage_mm':1000*old['ledger']['drainage_m3']},'records':records}
(P/'root_result_audit.json').write_text(json.dumps(ans,indent=2)+'\n')
print(json.dumps([{'factor':r['factor'],'aligned_rmse_pp':r['pooled']['offset_aligned']['rmse_pp'],'layer_aligned':[r['scores'][v]['offset_aligned']['rmse_pp'] for v in ('501','502','503')],'budgets':r['budgets_mm'],'dt900_pp':r['dt900_max_theta_difference_pp_by_layer']} for r in records],indent=2))
