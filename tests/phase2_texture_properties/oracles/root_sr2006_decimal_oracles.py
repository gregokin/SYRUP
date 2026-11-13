"""Independent high-precision checks of published SR2006 Table1 (normal-density slice only), NOT a production provider or field fit."""
from pathlib import Path
from decimal import Decimal as D,localcontext
import json
p=Path(__file__).resolve().parent;out=p/'root_sr2006_normal_density_oracles.json';assert not out.exists()
# Table3 rounded published examples: S,C percentages; theta_wp/theta33/thetaSAT percentages; Ks mm/hour.
cases=[('sand','88','5','5','10','46','108.1'),('sandy_loam','65','10','8','18','45','50.3'),('loam','40','20','14','28','46','15.5'),('silty_clay_loam','10','35','22','38','51','5.7'),('clay','25','50','30','42','50','1.1')]
rows=[]
with localcontext() as ctx:
 ctx.prec=80
 for name,S,C,wp_ref,fc_ref,sat_ref,ks_ref in cases:
  S,C,OM=D(S)/100,D(C)/100,D('2.5')
  tw=-D('.024')*S+D('.487')*C+D('.006')*OM+D('.005')*S*OM-D('.013')*C*OM+D('.068')*S*C+D('.031');wp=tw+D('.14')*tw-D('.02')
  tf=-D('.251')*S+D('.195')*C+D('.011')*OM+D('.006')*S*OM-D('.027')*C*OM+D('.452')*S*C+D('.299');fc=tf+D('1.283')*tf*tf-D('.374')*tf-D('.015')
  tg=D('.278')*S+D('.034')*C+D('.022')*OM-D('.018')*S*OM-D('.027')*C*OM-D('.584')*S*C+D('.078');gap=tg+D('.636')*tg-D('.107');sat=fc+gap-D('.097')*S+D('.043')
  et=-D('21.67')*S-D('27.93')*C-D('81.97')*gap+D('71.12')*S*gap+D('8.29')*C*gap+D('14.05')*S*C+D('27.16');air=et+D('.02')*et*et-D('.113')*et-D('.70')
  B=(D(1500).ln()-D(33).ln())/(fc.ln()-wp.ln());lam=1/B;ks=D(1930)*((3-lam)*(sat-fc).ln()).exp();rho=(1-sat)*D('2.65')
  theta6=sat if D(6)<=air else fc+(D(33)-D(6))*(sat-fc)/(D(33)-air)
  assert wp>0 and wp<fc<sat<1 and air<33 and fc<=theta6<=sat, (name, str(wp), str(fc), str(sat), str(air), str(theta6))
  # Table3 is rounded at 1vol% /0.1mm/hour; tolerate half-unit at printed resolution.
  assert abs(wp*100-D(wp_ref))<=D('.5') and abs(fc*100-D(fc_ref))<=D('.5') and abs(sat*100-D(sat_ref))<=D('.5')
  assert abs(ks-D(ks_ref))<=D('.051'),(name,ks,ks_ref)
  rows.append(dict(label=name,sand_fraction=str(S),silt_fraction=str(1-S-C),clay_fraction=str(C),organic_matter_percent=str(OM),normal_density_g_cm3=str(rho),theta_1500=str(wp),theta_33=str(fc),theta_sat=str(sat),air_entry_kpa=str(air),B=str(B),lambda_value=str(lam),Ks_mm_h=str(ks),Ks_m_s=str(ks/D(3600000)),theta_6_algebraic=str(theta6),air_entry_nonnegative=bool(air>=0),published_rounded=dict(wp_percent=wp_ref,theta33_percent=fc_ref,sat_percent=sat_ref,Ks_mm_h=ks_ref)))
record=dict(source_pdf='reference_sources/saxton_rawls_2006.pdf',source_sha256='24038366b95d508de140786b5b71fb634164dc9f89d1de158944c445c77186e5',equations='Table1 Eqs1-6,11-16,18; Table3 rounded corroboration',scope='Normal-density mathematical controls, not density-policy acceptance or new measured/field validation. Density is analytically generated explicitly for each control; no default field value inferred. No conductivity-to-interface mixing or wetting-front suction inferred.',decimal_precision=80,controls=rows)
out.write_text(json.dumps(record,indent=2)+'\n');print('PASS',len(rows),'Decimal80 normal-density controls match independently printed Table3 rounding')
