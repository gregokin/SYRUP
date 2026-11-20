from pathlib import Path
import csv
import math
import json
from fractions import Fraction as Q
from datetime import datetime, timedelta
import hashlib

P=Path(__file__).resolve().parent
CASE=Path(__import__('sys').argv[1])
c=json.loads(CASE.read_text())
T0=datetime.fromisoformat(c['forcing']['start_utc'])
def good(r):
    try:return math.isfinite(float(r['VSWCMean'])) and float(r['VSWCMean'])>=0 and float(r['VSWCFinalQF'])==0 and (r['VSWCFinalQFSciRvw']=='' or float(r['VSWCFinalQFSciRvw'])==0)
    except (ValueError,TypeError):return False
obs={}
for v,entries in c['swc']['levels'].items():
    series={}
    for rec in entries:
        p=Path(rec['path']);b=p.read_bytes();assert len(b)==rec['bytes'] and hashlib.sha256(b).hexdigest()==rec['sha256']
        for row in csv.DictReader(p.open()):
            key=datetime.fromisoformat(row['endDateTime']);assert key not in series
            series[key]=row
    assert good(series[T0])
    obs[v]=dict(initial_raw=series[T0]['VSWCMean'],initial_interval_start=series[T0]['startDateTime'],
                day_end_values=[float(series[t]['VSWCMean']) if t in series and good(series[t]) else None for t in [T0+timedelta(days=k) for k in range(1,161)]])

z=list(map(Q,['.058','.158','.258']));theta=[Q(obs[v]['initial_raw']) for v in ['501','502','503']]
edges=list(map(Q,['0','.1','.2','.3']))
def value(x):
    if x<=z[0]:return theta[0]
    if x>=z[-1]:return theta[-1]
    i=next(i for i in range(2) if z[i]<=x<=z[i+1])
    return theta[i]+(theta[i+1]-theta[i])*(x-z[i])/(z[i+1]-z[i])
water=[]
for lo,hi in zip(edges[:-1],edges[1:]):
    knots=sorted({lo,hi,*[d for d in z if lo<d<hi]})
    water.append(sum((value(a)+value(b))*(b-a)/2 for a,b in zip(knots[:-1],knots[1:])))

with Path(c['forcing']['files']['rainfall_30min']['path']).open() as h:rain=list(csv.DictReader(h))
days={}
for r in rain:
    t=datetime.fromisoformat(r['startDateTime']);d=(t-timedelta(hours=7)).date().isoformat();days.setdefault(d,[]).append(r)
wet=[]
for day,rows in days.items():
    positive=[i for i,r in enumerate(rows) if Q(r['precipBulk_mm'])>0]
    if positive:wet.append(dict(day=day,first=positive[0],last=positive[-1],span_seconds=(positive[-1]-positive[0]+1)*1800,
                                rain_mm_exact=str(sum(Q(rows[i]['precipBulk_mm']) for i in positive))))
total=sum(Q(r['precipBulk_mm']) for r in rain)
result=dict(status='PASS_INDEPENDENT_INPUT_AUDIT',observations=obs,valid_targets={v:sum(x is not None for x in a['day_end_values']) for v,a in obs.items()},
            exact_initial_water_fractions=[str(w) for w in water],initial_water_m=list(map(float,water)),
            initial_layer_theta=[float(w/(edges[i+1]-edges[i])) for i,w in enumerate(water)],initial_total_water_mm=float(sum(water)*1000),
            rain_total_mm_exact=str(total),wet_days=wet,added_storm_seconds=sum(d['span_seconds'] for d in wet),
            nominal_seconds=160*86400,physical_end_seconds=160*86400+sum(d['span_seconds'] for d in wet),
            maximum_intensity_mm_h=float(max(Q(r['precipBulk_mm'])*2 for r in rain)))
(P/'root_independent_inputs.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['observations','wet_days']},indent=2))
