"""Read-only input audit and independent 80-digit arithmetic; no model imports."""
import csv
import hashlib
import json
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
RAW = ROOT / 'agent_handoffs/tasks/jorn_three_layer_20261102/downloads/DP1.00096.001/2013-03'
OLD = ROOT / 'agent_handoffs/tasks/site_hydraulics_interface_comparison_20261117/analysis_candidate'
SIZES = ['sandVeryFineContent', 'sandFineContent', 'sandMediumContent',
         'sandCoarseContent', 'sandVeryCoarseContent', 'siltTotal', 'clayTotal']


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def read(kind):
    f = next(RAW.glob('*mgp_' + kind + '.*.csv'))
    with f.open() as stream:
        rows = list(csv.DictReader(stream))
    return f, {r['horizonName']: r for r in rows if r['horizonName'] in ('A', 'Bw')}


def calculate(s, c, om, rawrho):
    """Equations transcribed independently from S&R Table 1; densities g/cm3."""
    t = -D('.251')*s+D('.195')*c+D('.011')*om+D('.006')*s*om-D('.027')*c*om+D('.452')*s*c+D('.299')
    fcn = t+D('1.283')*t*t-D('.374')*t-D('.015')
    u = D('.278')*s+D('.034')*c+D('.022')*om-D('.018')*s*om-D('.027')*c*om-D('.584')*s*c+D('.078')
    g = D('1.636')*u-D('.107')
    sn = fcn+g-D('.097')*s+D('.043')
    rn = (1-sn)*D('2.65')
    rho = max(rawrho, D(1), D('.901')*rn)
    sat = 1-rho/D('2.65')
    fc33 = fcn-D('.2')*(sn-sat)
    et = -D('21.67')*s-D('27.93')*c-D('81.97')*g+D('71.12')*s*g+D('8.29')*c*g+D('14.05')*s*c+D('27.16')
    air = et+D('.02')*et*et-D('.113')*et-D('.70')
    ae = max(D(0), air)
    fc6 = fc33+D(27)*(sat-fc33)/(D(33)-ae)
    raw_air_diagnostic = fc33+D(27)*(sat-fc33)/(D(33)-air)
    normal_fc6_diagnostic = fcn+D(27)*(sn-fcn)/(D(33)-ae)
    return {k: str(v) for k, v in {'t33': t, 'theta33_normal': fcn,
        'tgap': u, 'gap_normal': g, 'saturation_normal': sn, 'density_normal': rn,
        'density_raw': rawrho, 'density_used': rho, 'theta33': fc33, 'theta_sat': sat,
        'air_first': et, 'air_raw': air, 'air_used': ae, 'theta6': fc6,
        'theta6_raw_air_diagnostic_not_adopted': raw_air_diagnostic,
        'theta6_normal_density_zero_air_diagnostic_not_adopted': normal_fc6_diagnostic,
        'added_water_above_theta33_mm_per_10cm': (fc6-fc33)*100}.items()}


def main():
    sources = {}
    bf, b = read('perbiogeosample')
    hf, h = read('perhorizon')
    df, dr = read('perbulksample')
    vf = next(RAW.glob('*.variables.*.csv'))
    for f in (bf, hf, df, vf):
        sources[str(f.relative_to(ROOT))] = sha(f)
    receipt = OLD / 'docs/subsurface_hydrology/site_properties_mixing/data/root_relative_density_probe.json'
    ref = json.loads(receipt.read_text())['records'][:3]
    sources[str(receipt.relative_to(ROOT))] = sha(receipt)
    physics = OLD / 'src/maple_syrup/soil_texture_properties.py'
    sources[str(physics.relative_to(ROOT))] = sha(physics)
    actual_config = ROOT / 'agent_handoffs/tasks/jorn_160_day_run_20261119/case_numba.json'
    actual = json.loads(actual_config.read_text())
    sources[str(actual_config.relative_to(ROOT))] = sha(actual_config)
    for f in (ROOT / 'outputs/subsurface_site_comparison_20261117').glob('*.docx'):
        sources[str(f.relative_to(ROOT))] = sha(f)
    pdf = ROOT / 'agent_handoffs/tasks/neon_texture_failure_audit_20261117/saxton_rawls_2006.pdf'
    sources[str(pdf.relative_to(ROOT))] = sha(pdf)
    with vf.open() as s:
        variables = [r for r in csv.DictReader(s) if r['fieldName'] in SIZES + ['sandTotal', 'estimatedOC', 'bulkDensExclCoarseFrag']]
    with localcontext() as ctx:
        ctx.prec = 80
        a, bw = b['A'], b['Bw']
        ra, rb = D(dr['A']['bulkDensExclCoarseFrag']), D(dr['Bw']['bulkDensExclCoarseFrag'])
        wa = D('.8')*ra/(D('.8')*ra+D('.2')*rb)
        layers = []
        for i, (w, rho) in enumerate(((D(1), ra), (wa, D('.8')*ra+D('.2')*rb), (D(0), rb))):
            vals = {k: w*D(a[k])+(1-w)*D(bw[k]) for k in SIZES + ['sandTotal', 'estimatedOC']}
            assert sum(vals[k] for k in SIZES) == 100
            assert sum(vals[k] for k in SIZES[:5]) == vals['sandTotal']
            om = vals['estimatedOC']*D('1.724')/10
            calc = calculate(vals['sandTotal']/100, vals['clayTotal']/100, om, rho)
            r = ref[i]
            for key, rk in [('theta33', 'theta33'), ('theta6', 'FC6_with_zero_air_floor'),
                            ('theta_sat', 'theta_saturation'), ('density_used', 'effective_density_g_cm3'),
                            ('air_raw', 'raw_air_entry_kpa')]:
                assert abs(float(calc[key])-r[rk]) < 2e-13, (i, key, calc[key], r[rk])
            assert abs(float(calc['theta6'])-actual['layers']['theta_field_capacity'][i]) < 2e-13
            assert abs(float(calc['theta_sat'])-actual['layers']['theta_saturation'][i]) < 2e-13
            layers.append({'layer': i+1, 'depth_cm': [i*10, (i+1)*10], 'A_mass_weight': str(w),
                'texture_percent': {k: str(v) for k, v in vals.items()}, 'OM_percent': str(om), 'calculations': calc})
        # Independent published Table 3 control: sand S=.88, C=.05, OM=2.5, normal density.
        control = calculate(D('.88'), D('.05'), D('2.5'), D('1.5'))
        assert round(float(control['theta33_normal'])*100) == 10
    raw_selected = {}
    for key in ('A', 'Bw'):
        raw_selected[key] = {
            'biogeosample': {k: b[key][k] for k in ['uid', 'horizonID', 'horizonName', 'sandTotal', 'estimatedOC'] + SIZES},
            'horizon': {k: h[key][k] for k in ['uid', 'horizonID', 'horizonTopDepth', 'horizonBottomDepth']},
            'density': {k: dr[key][k] for k in ['uid', 'bulkDensID', 'bulkDensCenterDepth', 'bulkDensTopDepth', 'bulkDensBottomDepth', 'bulkDensExclCoarseFrag']}}
    result = {'status': 'PASS', 'scope': 'Input and arithmetic audit only; no hydrology run or parameter/default adoption',
        'precision_decimal_digits': 80, 'raw_selected': raw_selected, 'variables': variables,
        'middle_A_mass_weight': str(wa), 'layers': layers,
        'published_Table3_sand_theta33_normal': control['theta33_normal'],
        'exact_prior_receipt_agreement_absolute_tolerance': 2e-13,
        'equation_provenance': {'Saxton_Rawls': '2006, Table1 Eq2–12; Table2 FC=33kPa; Table3 sandy examples',
            'Tietjen': '2009, Section3.1 paragraph25, FC from similar-site field observations at -6kPa',
            'SYRUP': 'Hybrid pressure choice with study-specific density/air-entry floors; not a measured site FC'},
        'source_sha256': sources}
    (P/'root_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    with (P/'root_layers.csv').open('w', newline='') as stream:
        names = ['layer','depth_cm','sand_percent','silt_percent','clay_percent','OM_percent','density_raw','density_used','theta33','theta6','theta_sat']
        writer=csv.DictWriter(stream,fieldnames=names,lineterminator='\n');writer.writeheader()
        for r in layers:
            writer.writerow(dict(layer=r['layer'],depth_cm='–'.join(map(str,r['depth_cm'])),
                sand_percent=r['texture_percent']['sandTotal'],silt_percent=r['texture_percent']['siltTotal'],
                clay_percent=r['texture_percent']['clayTotal'],OM_percent=r['OM_percent'],
                **{k:r['calculations'][k] for k in names[6:]}))
    print('PASS: three raw PSD sums, independently recomputed layer mixture, all prior hydraulic values, Table3 control; no model execution')
    for r in layers:
        print(r['layer'], {k:float(r['calculations'][k]) for k in ['theta33','theta6','theta_sat','density_used']})


if __name__ == '__main__':
    main()
