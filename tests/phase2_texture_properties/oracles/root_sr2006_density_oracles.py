"""Independent Decimal80 density-branch controls; pure mathematical references, no provider/native/moving-bed adoption."""
from pathlib import Path
from decimal import Decimal as D,localcontext
import json
p=Path(__file__).resolve().parent;out=p/'root_sr2006_density_oracles.json';assert not out.exists();normal=json.loads((p/'root_sr2006_normal_density_oracles.json').read_text());rows=[]
with localcontext() as ctx:
 ctx.prec=80
 for src in normal['controls']:
  wp,fc0,sat0,rho0,air=(D(src[k]) for k in ('theta_1500','theta_33','theta_sat','normal_density_g_cm3','air_entry_kpa'))
  for factor in map(D,('.9','.95','1','1.1','1.2','1.3')):
   rho=rho0*factor;sat=1-rho/D('2.65');fc=fc0-D('.2')*(sat0-sat);gap=sat-fc;flags=[]
   if not D(1)<=rho<=D('1.8'):flags.append('density_outside_1_to_1p8_g_cm3')
   if not D('.9')<=factor<=D('1.3'):flags.append('DF_outside_0p9_to_1p3')
   if gap<D('.005'):flags.append('density_gap_below_0p005')
   if not 0<wp<fc<sat<1:flags.append('invalid_retention_anchor_order')
   if not 0<=air<33:flags.append('invalid_air_entry')
   result={}
   if 0<wp<fc<sat<1:
    B=(D(1500).ln()-D(33).ln())/(fc.ln()-wp.ln());lam=1/B;ks=D(1930)*((3-lam)*gap.ln()).exp();theta6=sat if D(6)<=air else fc+D(27)*gap/(D(33)-air)
    result=dict(theta_1500=str(wp),theta_33=str(fc),theta_sat=str(sat),air_entry_kpa=str(air),B=str(B),lambda_value=str(lam),Ks_mm_h=str(ks),Ks_m_s=str(ks/D(3600000)),theta_6_algebraic=str(theta6))
   rows.append(dict(label=src['label']+'_DF'+str(factor),sand_fraction=src['sand_fraction'],silt_fraction=src['silt_fraction'],clay_fraction=src['clay_fraction'],organic_matter_percent=src['organic_matter_percent'],bulk_density_g_cm3=str(rho),density_factor=str(factor),density_gap=str(gap),expected_unsupported_flags=flags,mathematically_admissible_fullcurve=not flags,values=result))
record=dict(decimal_precision=80,source='SR2006 DOI10.2136/sssaj2005.0117 Table1 Eqs7-10,11-16,18; parsed primary-PDF text, no visual-render or fullSPAW identity claim.',recipe='Air entry held at normal-density Eq4 prediction; density modifies SAT/33/B/lambda/Ks. Refusal flags are independent mathematical/input-domain checks; no clipping/floor/density adjustment imposed.',scope='Fixed explicit density controls only; no field validation, native-bed porosity/texture mapping, state adoption, elevation change or remapping policy.',controls=rows)
out.write_text(json.dumps(record,indent=2)+'\n');print('GENERATED',len(rows),'density controls;',sum(x['mathematically_admissible_fullcurve'] for x in rows),'admissible full curves; invalid predictions retained and labelled')
