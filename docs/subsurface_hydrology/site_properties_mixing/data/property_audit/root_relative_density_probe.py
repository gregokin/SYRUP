from pathlib import Path
import ast,json,math
from decimal import Decimal as D,localcontext
import numpy as np
from maple_syrup import soil_texture_properties as st
p=Path(__file__).resolve().parent
source=(p/'root_probe.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='paper');ns={'D':D,'localcontext':localcontext};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p/'root_probe.py'),'exec'),ns);paper=ns['paper']
old=json.loads((p/'root_probe.json').read_text());raw={x['input']['name']:x['input'] for x in old['cases']}
records=[]
for name in ['JORN_A','JORN_10_20_dry_mass','JORN_Bw','MOAB_0_10_dry_mass','MOAB_Bw']:
 r=dict(raw[name]);pre=paper(r);floor=.901*pre['rho_normal_kg_m3'];r['rho']=max(r['rho'],1000,floor);post=paper(r)
 rho=r['rho']/1000;rhoN=post['rho_normal_kg_m3']/1000;air=max(0,post['air_entry_kpa']);fc33=post['theta_33_density_adjusted'];sat=post['theta_sat_density_adjusted'];wp=post['theta_1500'];B=math.log(1500/33)/math.log(fc33/wp);ks=1930*(sat-fc33)**(3-1/B);fc6=sat if air>=6 else fc33+27*(sat-fc33)/(33-air)
 # Reference calculations only: the native provider remains unchanged. No fake measured override.
 values=st.reference_block(*[np.array([v],dtype=np.float64) for v in [r['S'],r['Si'],r['C'],r['OM'],r['rho'],np.nan,np.nan,np.nan,np.nan,np.nan]],st.ss.summation_error_bound_kg(3,1.) if hasattr(st.ss,'summation_error_bound_kg') else st.FRACTION_CLOSURE_TERMS*4*np.finfo(float).eps)
 native_flags=int(values[0][0]);assert native_flags in (0,256),(name,native_flags)
 rec={'name':name,'raw_density_g_cm3':raw[name]['rho']/1000,'predicted_normal_g_cm3':rhoN,'effective_density_g_cm3':rho,'density_factor':post['density_factor'],'raw_air_entry_kpa':post['air_entry_kpa'],'effective_air_entry_kpa':air,'theta_saturation':sat,'FC6_with_zero_air_floor':fc6,'Ks_mm_h':ks,'theta1500':wp,'theta33':fc33,'native_flags_without_air_floor':native_flags,'all_other_native_guards_pass':native_flags in (0,256),'hypothetical_air_floor_removes_only_256':True}
 records.append(rec)
(p/'root_relative_density_probe.json').write_text(json.dumps({'scope':'Parameter what-if only; unchanged native provider checked with density rule; air floor evaluated independently without mislabelling it measured. No soil-water run or default change.','effective_density_rule':'max(raw_density,1000kg/m3,0.901*normal_density), applied after raw-density soil-texture/OM layer assembly.','records':records},indent=2)+'\n')
for r in records:print(json.dumps(r))
