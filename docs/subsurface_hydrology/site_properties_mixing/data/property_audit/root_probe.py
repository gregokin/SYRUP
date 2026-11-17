from pathlib import Path
from decimal import Decimal as D,localcontext
import csv,json,hashlib,math
import numpy as np
from maple_syrup import soil_texture_properties as st
p=Path(__file__).resolve().parent
old=p.parent/'subsurface_observed_drydown_20261114'
rows=[dict(name='JORN_A',S=.867,Si=.062,C=.071,OM=.3448,rho=1419.6),dict(name='JORN_Bw',S=.837,Si=.079,C=.084,OM=.1724,rho=903.5),dict(name='MOAB_A',S=.730,Si=.197,C=.073,OM=1.0344,rho=1519.7),dict(name='MOAB_Bw',S=.712,Si=.170,C=.118,OM=.5172,rho=1481.8)]

def paper(r):
 with localcontext() as ctx:
  ctx.prec=60
  s,c,o,rho=[D(str(r[k])) for k in ['S','C','OM','rho']]
  t=-D('.024')*s+D('.487')*c+D('.006')*o+D('.005')*s*o-D('.013')*c*o+D('.068')*s*c+D('.031');wp=t+D('.14')*t-D('.02')
  t=-D('.251')*s+D('.195')*c+D('.011')*o+D('.006')*s*o-D('.027')*c*o+D('.452')*s*c+D('.299');fc=t+D('1.283')*t*t-D('.374')*t-D('.015')
  t=D('.278')*s+D('.034')*c+D('.022')*o-D('.018')*s*o-D('.027')*c*o-D('.584')*s*c+D('.078');gap=t+D('.636')*t-D('.107');sat=fc+gap-D('.097')*s+D('.043')
  te=-D('21.67')*s-D('27.93')*c-D('81.97')*gap+D('71.12')*s*gap+D('8.29')*c*gap+D('14.05')*s*c+D('27.16');air=te+D('.02')*te*te-D('.113')*te-D('.70')
  rn=(1-sat)*D('2.65');satd=1-rho/D(1000)/D('2.65');fcd=fc-D('.2')*(sat-satd)
  return {k:float(v) for k,v in dict(theta_1500=wp,theta_33_normal=fc,gap_normal=gap,theta_sat_normal=sat,rho_normal_kg_m3=rn*1000,density_factor=rho/1000/rn,air_entry_regression_first=te,air_entry_kpa=air,theta_sat_density_adjusted=satd,theta_33_density_adjusted=fcd).items()}

def mix(a,b,fa,kind,name):
 rho=fa*a['rho']+(1-fa)*b['rho']
 wa=fa*a['rho']/rho if kind=='dry_mass' else fa
 return dict(name=name,**{key:wa*a[key]+(1-wa)*b[key] for key in ['S','Si','C','OM']},rho=rho,weighting=kind,depth_fraction_A=fa,mass_fraction_A=fa*a['rho']/rho)
rows.extend([mix(rows[0],rows[1],.8,k,'JORN_10_20_'+k) for k in ['dry_mass','depth']])
rows.extend([mix(rows[2],rows[3],.6,k,'MOAB_0_10_'+k) for k in ['dry_mass','depth']])
# Normal-density diagnostic is not adoption or a repaired measurement.
for orig in rows[:4]:
 r=dict(orig);r['name']+='_normal_density_control';r['rho']=paper(orig)['rho_normal_kg_m3'];r['density_source']='reference regression, diagnostic only';rows.append(r)
out=[]
for r in rows:
 rec={'input':r,'independent_paper_60_digit':paper(r),'backends':{}}
 for backend in ['numpy','numba']:
  try:
   v=st.estimate_soil_texture_properties(sand_fraction=r['S'],silt_fraction=r['Si'],clay_fraction=r['C'],organic_matter_percent=r['OM'],bulk_density_kg_m3=r['rho'],n_layers=1,n_cells=1,backend=backend,diagnostics=True,memory_budget_bytes=1000000,available_bytes_fn=lambda:100000000)
   rec['backends'][backend]={'status':'accepted','properties':{k:float(v.array(k).item()) for k in v.names()}}
   assert math.isclose(rec['independent_paper_60_digit']['air_entry_kpa'],float(v.array('air_entry_kpa').item()),rel_tol=0,abs_tol=1e-12)
  except st.TexturePropertyError as e:rec['backends'][backend]={'status':'refused','flags':e.flags,'first_layer':e.first_layer,'message':str(e)}
 assert rec['backends']['numpy']['status']==rec['backends']['numba']['status']
 if rec['backends']['numpy']['status']=='refused':assert rec['backends']['numpy']['flags']==rec['backends']['numba']['flags']
 out.append(rec)
# Sensitivity of the raw regression only, not adopted properties or calibrated OM.
sens=[]
for orig in rows[:4]:
 vals=[]
 for om in [0,.1,.1724,.3448,.5172,1.0344,2,4,8]:
  r=dict(orig,OM=om);v=paper(r);vals.append({'OM_percent':om,'air_entry_kpa':v['air_entry_kpa']})
 sens.append({'name':orig['name'],'values':vals})
res={'scope':'Parameter diagnostics ONLY; no water simulation, source/default edits, applied overrides, new fits or changed historical runs.','precision_decimal_digits':60,'cases':out,'OM_regression_sensitivity_not_adopted':sens,'source_files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (old/'analysis_candidate/src/maple_syrup').glob('soil_texture_properties*.py')}}
(p/'root_probe.json').write_text(json.dumps(res,indent=2)+'\n')
for x in out:
 print(x['input']['name'],'air=',round(x['independent_paper_60_digit']['air_entry_kpa'],9),'DF=',round(x['independent_paper_60_digit']['density_factor'],6),[(k,v['status'],v.get('flags')) for k,v in x['backends'].items()])
