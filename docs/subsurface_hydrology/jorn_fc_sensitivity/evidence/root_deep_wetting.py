"""Independent, same-timestamp January response and deep-layer support audit."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent;records=[]
for label in ('1','1p5'):
 for fc in range(9,16):
  key=f'fc{fc:02d}_s{label}';r=json.loads((P/'study'/key/'actual_numba/result.json').read_text());rows=r['rows'];before=next(x for x in rows if x['nominal_close_utc']=='2024-01-23T07:00:00Z');after=next(x for x in rows if x['nominal_close_utc']=='2024-01-24T07:00:00Z');obsday=next(x for x in rows if x['nominal_close_utc']=='2024-01-25T07:00:00Z')
  assert after['date']=='2024-01-23' and abs(after['rain_m3']*1000-10.036)<1e-12
  assert obsday['obs_theta'][2]==.0634
  peak=max(rows,key=lambda x:x['layer_theta'][2]);initial=r['plan']['initial']['layer_theta']
  records.append({'key':key,'fc_percent_all_layers':fc,'stress':.2017*(1 if label=='1' else 1.5),'initial_layer_theta_percent':[100*x for x in initial],'prestorm_layer_theta_percent':[100*x for x in before['layer_theta']],'poststorm_close':after['nominal_close_utc'],'poststorm_layer_theta_percent':[100*x for x in after['layer_theta']],'poststorm_deep_change_pp':100*(after['layer_theta'][2]-before['layer_theta'][2]),'storm_gravity_transfers_mm':[1000*x for x in after['percolation_m3']],'storm_bottom_drainage_mm':1000*after['drainage_m3'],'Jan25_model_deep_percent':100*obsday['layer_theta'][2],'Jan25_observed_deep_percent':100*obsday['obs_theta'][2],'whole_record_model_deep_peak_percent':100*peak['layer_theta'][2],'whole_record_peak_close':peak['nominal_close_utc'],'whole_record_observed_deep_peak_percent':100*max(x['obs_theta'][2] for x in rows if x['obs_theta'][2] is not None)})
ans={'status':'PASS','scope':__doc__,'date_convention':'Jan23 storm is nominal dayJan23, closing Jan24 07UTC. First following validdeepcomparison is Jan25 07UTC. No point-to-bucket equivalence or sensor-fault conclusion assumed.','records':records}
(P/'root_deep_wetting.json').write_text(json.dumps(ans,indent=2)+'\n');print(json.dumps(records,indent=2))
