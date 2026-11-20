"""Independent 90-digit rain + open gravity cascade + E + pair relaxation control.

Uses declared binary64 parameters and raw pinned rain; never calls a production
hydrology update. This is an equation control, not an alternative field solver.
"""
from pathlib import Path
from decimal import Decimal as D, localcontext
import json
import csv
import sys
import math

Q = Path(sys.argv[1]).resolve()
answers = []
for case in ('numba', 'numpy', 'numba_dt900'):
    cfg = json.loads((Q / f'case_{case}.json').read_text())
    result = json.loads((Q / f'actual_{case}' / 'result.json').read_text())
    lay, phy = cfg['layers'], cfg['physics']
    edges = lay['edges_m']
    h_float = [b-a for a,b in zip(edges[:-1],edges[1:])]
    h = [D.from_float(x) for x in h_float]
    fc = [D.from_float(t*x) for t,x in zip(lay['theta_field_capacity'],h_float)]
    sat = [D.from_float(t*x) for t,x in zip(lay['theta_saturation'],h_float)]
    res = [D.from_float(t*x) for t,x in zip(lay['theta_residual'],h_float)]
    inv = [D.from_float(1/h_float[i]+1/h_float[i+1]) for i in range(2)]
    down, up = [], []
    for i, k in enumerate(lay['interface_conductivity_m_s']):
        sep = (h_float[i]+h_float[i+1])*0.5
        down.append(D.from_float(((phy['diffusion_coefficient']*k)*h_float[i])/sep))
        up.append(D.from_float(((phy['diffusion_coefficient']*k)*h_float[i+1])/sep))
    stress = D.from_float(lay['theta_evaporation_stress'])
    with Path(cfg['forcing']['files']['rainfall_30min']['path']).open() as f:
        forcing = list(csv.DictReader(f))
    with localcontext() as context:
        context.prec = 90
        water = [D.from_float(x) for x in result['plan']['initial']['layer_water_m']]
        max_error = D(0)
        largest_bound_ratio = 0.0
        total_e = total_drain = D(0)
        total_perc = [D(0),D(0)]
        def advance(dt, pet_rate):
            perc = [D(0),D(0)]
            for i in range(2):
                amount = max(D(0),water[i]-fc[i])
                water[i] -= amount
                water[i+1] += amount
                perc[i] = amount
            drain = max(D(0),water[2]-fc[2])
            water[2] -= drain
            available = max(D(0),water[0]-res[0])
            mult = min(D(1),(water[0]/h[0]/stress)**2) if available>0 else D(0)
            e = min(available,D.from_float(phy['evaporation_factor'])*D.from_float(pet_rate)*D(dt)*mult)
            water[0] -= e
            for i in range(2):
                delta = water[i]/h[i]-water[i+1]/h[i+1]
                donor, receiver = (i,i+1) if delta>0 else (i+1,i)
                conductance = down[i] if donor==i else up[i]
                wanted = delta/inv[i]*(D(1)-(-conductance*inv[i]*D(dt)).exp())
                amount = min(abs(wanted),max(D(0),water[donor]-res[donor]),max(D(0),sat[receiver]-water[receiver]))
                water[donor] -= amount
                water[receiver] += amount
            return e,drain,perc
        for day,row in enumerate(result['rows']):
            daily_e = daily_drain = D(0)
            daily_perc = [D(0),D(0)]
            rain = [float(x['precipBulk_mm']) for x in forcing[48*day:48*(day+1)]]
            wet = [i for i,x in enumerate(rain) if x>0]
            dt = cfg['storm']['max_dt_s']
            if wet:
                for j in range(wet[0],wet[-1]+1):
                    rate = (rain[j]*2.0)*(1.0/3.6e6)
                    for unused in range(1800//dt):
                        water[0] += D.from_float(rate*dt)
                        # This control is valid only while all rain is supply-limited.
                        assert water[0]<sat[0] and rate<cfg['matrix']['conductivity_m_s']
                        e,drain,perc = advance(dt,0.0)
                        daily_drain += drain
                        daily_perc = [a+b for a,b in zip(daily_perc,perc)]
            e,drain,perc = advance(86400,row['pet_mm_day']/1000.0/86400.0)
            daily_e += e
            daily_drain += drain
            daily_perc = [a+b for a,b in zip(daily_perc,perc)]
            actual = [D.from_float(x) for x in row['water_m']]
            error = max(abs(a-b) for a,b in zip(actual,water))
            max_error = max(max_error,error)
            bound = row['closure_bound_m3']
            assert float(error)<=bound,(case,day,'water',float(error),bound)
            for label,expected,got in [('E',daily_e,row['evaporation_m3']),('bottom drainage',daily_drain,row['drainage_m3'])]:
                assert abs(float(expected)-got)<=bound,(case,day,label,float(expected),got,bound)
            assert all(abs(float(a)-b)<=bound for a,b in zip(daily_perc,row['percolation_m3']))
            assert all(row[k]==0 for k in ('runoff_pool_m3','surface_return_m3','room_rejected_m3','n_ponded_steps','n_onsets'))
            largest_bound_ratio = max(largest_bound_ratio,float(error)/bound)
            total_e += daily_e
            total_drain += daily_drain
            total_perc = [a+b for a,b in zip(total_perc,daily_perc)]
        answers.append({'case':case,'days':160,'decimal_digits':90,'max_water_error_m':float(max_error),'max_error_over_declared_daily_bound':largest_bound_ratio,'evaporation_mm':float(total_e*1000),'bottom_drainage_mm':float(total_drain*1000),'gravity_percolation_mm':[float(v*1000) for v in total_perc]})
(Q/'root_decimal_verification.json').write_text(json.dumps({'status':'PASS','scope':__doc__,'cases':answers},indent=2)+'\n')
print(json.dumps(answers,indent=2))
