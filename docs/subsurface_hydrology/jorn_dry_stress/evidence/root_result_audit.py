"""Independent Fraction scores, bounded layer budgets and unchanged observed forcing; no hydrology update."""
from pathlib import Path
from fractions import Fraction as F
import json,math
from maple_syrup.conservation import volume_roundoff_bound_m3
P=Path(__file__).resolve().parent;C=P/'analysis_candidate';oldroot=C/'docs/subsurface_hydrology/joint_d_stress_calibration/data/actual_cpu';J=lambda p:json.loads(Path(p).read_text());records=[];groups={}
for window in ('JORN4_A','JORN4_B','MOAB2_C'):
 ref=J(oldroot/'prepared_inputs'/f'{window}.json')
 for label,stress in [('s1',.2017),('s1p5',.30255)]:
  key=f'{window}_{label}';Q=P/'study'/key;c=J(Q/'case.json');o=J(Q/'actual/result.json');p=o['plan']['prepared_inputs'];r=next(iter(o['runs'].values()));tr=r['trajectory'];h=[b-a for a,b in zip(c['layers']['edges_m'][:-1],c['layers']['edges_m'][1:])]
  assert p==ref,(key,'observed inputs changed');assert c['layers']['theta_evaporation_stress']==stress
  assert c['scenarios'][0]['diffusion_coefficient']==.0010239461376024856 and c['scenarios'][0]['evaporation_factor']==1
  if window.startswith('JORN'):assert c['layers']['theta_field_capacity']==[.1]*3
  assert all(a==0 for a in tr['drainage_m']) and all(a==0 for a in tr['surface_return_m']) and all(a==0 for row in tr['percolation_m'] for a in row)
  assert r['restart']['whole_trajectory_and_cumulative_identical']
  scores={};local={kind:[] for kind in ('raw','aligned','persistence')}
  for j,v in enumerate(('501','502','503')):
   es={kind:[] for kind in local};m0=p['integrated_initial_layer_theta'][j];o0=p['raw_initial_point_theta'][j]
   for i,obs in enumerate(p['observed_end_theta_m3m3'][v]):
    if obs is None:continue
    m=r['layer_theta'][i][j];vals={'raw':m-obs,'aligned':(m-m0)-(obs-o0),'persistence':o0-obs}
    for kind,e in vals.items():
     e=F.from_float(e);es[kind].append(e);local[kind].append(e)
     for scope in ('all_sites_all_layers',f'{c["site"]["site"]}_all_layers',f'{window}_all_layers')+ (('all_sites_upper_two',f'{c["site"]["site"]}_upper_two',f'{window}_upper_two') if j<2 else ()):
      groups.setdefault(stress,{}).setdefault(scope,{}).setdefault(kind,[]).append(e)
   scores[v]={kind:{'n':len(e),'rmse_pp':100*math.sqrt(float(sum(x*x for x in e)/len(e))),'bias_pp':100*float(sum(e)/len(e))} for kind,e in es.items()}
  pooled={kind:{'n':len(e),'rmse_pp':100*math.sqrt(float(sum(x*x for x in e)/len(e))),'bias_pp':100*float(sum(e)/len(e))} for kind,e in local.items()}
  previous=p['integrated_initial_layer_water_m'];maxerr=0.;grad=[0.,0.];perco=[0.,0.]
  for i,w in enumerate(tr['water_m']):
   pc=tr['percolation_m'][i];dx=tr['diffusion_m'][i];E=tr['evaporation_m'][i];bottom=tr['drainage_m'][i];want=[-E-pc[0]-dx[0],pc[0]-pc[1]+dx[0]-dx[1],pc[1]+dx[1]-bottom];err=max(abs((a-b)-d) for a,b,d in zip(w,previous,want));bound=tr['step_bound_m3'][i];assert err<=bound,(key,i,err,bound);assert abs(tr['step_residual_m3'][i])<=bound;maxerr=max(maxerr,err);previous=w;grad=[a+b for a,b in zip(grad,dx)];perco=[a+b for a,b in zip(perco,pc)]
  assert abs(r['cumulative_ledger']['residual_m3'])<=r['cumulative_ledger']['bound_m3']
  n= len(tr['water_m']);assert n=={'JORN4_A':10,'JORN4_B':4,'MOAB2_C':6}[window]
  assert tr['t_s']==[float(86400*(i+1)) for i in range(n)]
  closure=r['cumulative'];assert closure['added_m3']==0 and closure['surface_return_m3']==0 and closure['drainage_m3']==0
  record={'key':key,'window':window,'site':c['site']['site'],'stress':stress,'field_capacity':c['layers']['theta_field_capacity'],'days':n,'scores':scores,'pooled':pooled,'initial_storage_mm':1000*r['initial_storage_m'],'final_storage_mm':1000*r['final_storage_m'],'evaporation_mm':1000*closure['evaporation_m3'],'bottom_drainage_mm':1000*closure['drainage_m3'],'net_gradient_transfer_mm_down_positive':[1000*x for x in grad],'gravity_transfer_mm':[1000*x for x in perco],'max_layer_budget_error_m':maxerr,'ledger':r['cumulative_ledger'],'backend_agreement':r['backend_agreement_max_abs'],'restart':r['restart'],'inputs_exact_historical':True}
  records.append(record)
summary={str(s):{scope:{k:{'n':len(es),'rmse_pp':100*math.sqrt(float(sum(e*e for e in es)/len(es))),'bias_pp':100*float(sum(es)/len(es))} for k,es in kinds.items()} for scope,kinds in scopes.items()} for s,scopes in groups.items()}
assert all(v['all_sites_all_layers']['aligned']['n']==53 and v['JORN_all_layers']['aligned']['n']==39 and v['all_sites_upper_two']['aligned']['n']==36 and v['JORN_upper_two']['aligned']['n']==26 for v in summary.values())
ans={'status':'PASS','scope':__doc__,'records':records,'groups':summary,'all_gravity_and_bottom_drainage_zero':True,'all_prepared_inputs_exact_historical':True,'t0_excluded':True,'third_layer_retained_not_censored':True}
(P/'root_result_audit.json').write_text(json.dumps(ans,indent=2)+'\n');print(json.dumps(summary,indent=2))
