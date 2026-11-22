"""Independent90-digit closed water-budget control; no production-step call."""
from pathlib import Path
from decimal import Decimal as D,localcontext
import json,sys
Q=Path(sys.argv[1]).resolve();c=json.loads((Q/'case.json').read_text());out=json.loads((Q/'actual/result.json').read_text());run=next(iter(out['runs'].values()));tr=run['trajectory'];plan=out['plan'];lay=c['layers'];scen=c['scenarios'][0]
hf=[b-a for a,b in zip(lay['edges_m'][:-1],lay['edges_m'][1:])];h=[D.from_float(x) for x in hf];fc=[D.from_float(t*x) for t,x in zip(lay['theta_field_capacity'],hf)];sat=[D.from_float(t*x) for t,x in zip(lay['theta_saturation'],hf)];res=[D.from_float(t*x) for t,x in zip(lay['theta_residual'],hf)];stress=D.from_float(lay['theta_evaporation_stress']);down=[];up=[];inv=[D.from_float(1/hf[i]+1/hf[i+1]) for i in range(2)]
for i,k in enumerate(lay['interface_conductivity_m_s']):
 sep=(hf[i]+hf[i+1])*0.5;down.append(D.from_float(((scen['diffusion_coefficient']*k)*hf[i])/sep));up.append(D.from_float(((scen['diffusion_coefficient']*k)*hf[i+1])/sep))
with localcontext() as context:
 context.prec=90;water=[D.from_float(x) for x in plan['initial_layer_water_m']];maxerr=D(0);totE=totDrain=D(0);totalperc=[D(0),D(0)]
 for day,pet in enumerate(plan['pet_mm_day']):
  perc=[]
  for i in range(2):
   a=max(D(0),water[i]-fc[i]);water[i]-=a;water[i+1]+=a;perc.append(a)
  drain=max(D(0),water[2]-fc[2]);water[2]-=drain
  avail=max(D(0),water[0]-res[0]);mult=min(D(1),(water[0]/h[0]/stress)**2) if avail>0 else D(0)
  rate=(pet/1000.)/86400.;e=min(avail,D.from_float(scen['evaporation_factor'])*D.from_float(rate)*D(86400)*mult);water[0]-=e
  for i in range(2):
   delta=water[i]/h[i]-water[i+1]/h[i+1];donor,receiver=(i,i+1) if delta>0 else (i+1,i);cond=down[i] if donor==i else up[i];wanted=delta/inv[i]*(D(1)-(-cond*inv[i]*D(86400)).exp());a=min(abs(wanted),max(D(0),water[donor]-res[donor]),max(D(0),sat[receiver]-water[receiver]));water[donor]-=a;water[receiver]+=a
  err=max(abs(D.from_float(a)-b) for a,b in zip(tr['water_m'][day],water));maxerr=max(maxerr,err);bound=tr['step_bound_m3'][day];assert float(err)<=bound,(day,float(err),bound)
  for val,got in [(e,tr['evaporation_m'][day]),(drain,tr['drainage_m'][day]),*zip(perc,tr['percolation_m'][day])]:assert abs(float(val)-got)<=bound,(day,float(val),got,bound)
  assert tr['surface_return_m'][day]==0;totE+=e;totDrain+=drain;totalperc=[a+b for a,b in zip(totalperc,perc)]
ans={'status':'PASS','scope':__doc__,'days':len(plan['pet_mm_day']),'decimal_digits':90,'max_water_error_m':float(maxerr),'evaporation_mm':float(1000*totE),'bottom_drainage_mm':float(1000*totDrain),'gravity_transfer_mm':[float(1000*x) for x in totalperc]}
(Q/'root_decimal_verification.json').write_text(json.dumps(ans,indent=2)+'\n');print(json.dumps(ans,indent=2))
