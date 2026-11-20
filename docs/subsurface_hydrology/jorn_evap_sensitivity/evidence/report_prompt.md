# Stage2: independently review completed E/stress sweeps and write report

Stage1 author closed0. Root inspected generator and all24configs, reran55tests with actual4e38SYRUP/fbcMAPLE beforeafter guards, Ruffclean. Root corrected TWO test-only mass assertions from arbitrary1e-15 to res.budget.bound_m3, all55repassed; review those two lines (no physical source changes).

Root completed all24native160daycases:8Numba1800,8NumPy1800,8Numba900; allsource/input/helper seals unchanged. Root90digit reference (independent openFCgravity/E/pairrelax, no productionstepcall) checked all24; maxerror1.07144e-17m. NumPyNumba trackedoutputsdifference0. Max900s refinement0.000233902percentagepoints. Fresh baseline rows/ledger exactly reproduce prioracceptedSR33K1 case. Root s3 day80/newprocessresume control finishing, root_restart_verification.json forthcoming; leavependinguntilrecordexists, don'tpredict.

Read root_run_seal.json (218files, verifybefore/afterconsuming), root_result_audit.json (records have key/E/stress_multiplier/stress, sourcefacts, perlayer/fullpooledscores, waterbudgets, timesteps, native timing), root_period_scores.json (nominaldaylabels beforeMar1 vs>=Mar1; BOTHtop2/all3 andbydepth, bias/raw/aligned/counts), root_independent_inputs.json, root_parameter_verification.json, report_pending_notes.md, study/*/root_decimal_verification.json.
All8physicalresults unchanged. Source180/4e38a472e268a56491b788d5b1271cc3c3da23bb1cc3221a8c59d90f3cc864d2, genuineMAPLE412/fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e. NoGPU/newphysics/defaultadoption.

## Author write scope
Solewriter NEW docs/subsurface_hydrology/jorn_evap_sensitivity/ reportbuilder, Markdown, DOCX, figures, CSV/data, artifactmanifest, verification. NO generator/setup-test edits now (Root sourcefreeze), no models/config/result/prod/oldhelper/data/environment/Git/GPU/network. Task report_author_report.md allowed.
Use existing /home/okin/MAPLE/.venv with report_tools path from priorreport for docx, matplotlibAgg. No installs. Reuse prior report code/styles where useful; avoid its corrected issues (timestamppairing, crowdedtables, datasetrefs, overlappinggravitycounts).

## Requested report
File SYRUP_JORN_160day_SR33_1x_evaporation_sensitivity_20261120.docx in docs; Rootwillcopybesideoldoutputreports afterreview.
Two families: E1/.8/.6/.4 atstress.201705811; E1 andstress1/1.5/2/2.5/3×base.8unique sharedbaseline, no9-runmass/countclaim. K/KsALLfixed1x, d.0010239461 fixed, FC33~9.75/9.94/10.91%, SAT/residual/soil/initial22.9424mm/weather/rain25.088mm unchanged; openbottom, dailyhydrology, noT/growth/plants, matrixonly, zeroresidencesurfaceexport. NoE/Tduringrain, no newdry-day substep! Nominal160days+27.5stormhours sameclock. Allcaseconfigsdiffonlyrequested Eorσ andexplicitprovenance.

PrimaryactualE_mm inrequestedorder baseline/e.8/e.6/e.4/s1.5/s2/s2.5/s3:
29.4733/27.9929/25.8960/22.2917/23.2138/18.4469/15.1554/12.6903.
Bottomdrainmm1.2822/2.0543/3.2710/5.4938/4.9011/7.7136/9.2421/10.1466.
FullalignedRMSEpp2.2731/2.3923/2.5284/2.7394/2.7004/2.9230/3.0966/3.2280.
Don'thardcodeasresults; derivefromsealedrawrowsandcompareRoot.
LowEdoesNOTscale160dayactualElinearlybecausemoisturefeedback/drainage. LessEkeepssoilwetterbutmorewaterescapesbottominstead ofallbeingretained. Rainintakeall25.088mm/runoffzero anddeepJanuarywettingdiscrepancyremain; verifytheseactual facts.
Earlytop2rawRMSEbestE.8(.74632pp), alignedbestE.6(.98829pp), notsamechoice/fullpooledbestbaseline. No declaredcalibration/bestphysicalparameterfromthisgrid.

## Human explanation/provenance essential
Answerlatestuserquestioninreport: stress ismoistureatwhichfullpotentialEallowed; belowitquadratic theta/thresholdratio; separate residualstopswithdrawal. Physicalproxyforincreasingevapresistanceasdry, NOT directlymeasuredsoilproperty, suction/FC/SAT orindependentwaterstore. Current20.1706% cameJOINTdryperiodfit d/σ atE1,53JORN/MOAB2025points, not SRtheta1500. Tietjen2009Sect2.3 wsc thresholdforcombinedET/stomatalclosure(Table2 uses-1500kPa); laterEcoHyD codeWPquadratic; Tabernas.18CASEinput notuniversal. Distinguish thisheritagefromSYRUPpureEwithLEAFseparateT. Citeprimarypaper and actuallocalcalibrationreport.
Rain-freecalibration isNOTprooftranspiration-free: JORNApril/MOABMay2025 fitT0 assumption, ifplantsactivecouldabsorbmissingT; DO NOTclaimplantactivity/Tobserved, onlyconditional inference. UserMarch1 onset iscontextfor interpreting2024latebias, not implementednewT.
Because allE-stage topwater<=FC9.75%<everyσ, samequadraticbranchalways. EffectivecoefficientE/r^2:σ1.5=.444444,2=.25,2.5=.16,3=.111111. NativecomponenttestconservedwaterandconfirmedmatchingactualE/stateforcorrespondingE; testsarelargelysameknob, cannotseparatelyidentifyEandσhere. Largeσ50.43/60.51% exceedSATbutarehypotheticalnormalizationscales acceptedWITHOUTchangingcapacity; notphysicalmeasuredWPclaims.

## Comparison presentation
Clear raw and aligned familyplots withobservedpoints/gaps, FCthresholdsandcorrectunits; separateEandstressfamilies ratherthan9overplottedcurves. Includeperdepth/raw/aligned RMSE/counts andearly/lateMar1 diagnosticsespeciallytop2. Distinguish equaldata/support/scoring; doNOTdeclare lowestpooled160dayerrorbestEwhennonTlateTmissing. HighernoTsoilwaterlatecanreflectmissingT; stilldeepsoilpredictedoverwetinJanuarybeforeassumedT. Modelwhole20-30cmbucketwettingimplies25.8probe shouldrespond; causeunresolvedtransportapprox/soilproxy/sensor, neverassumesensorfaultorremoveit.
Finalupper2observationsNOTJune26: lastshallowMar15T07, middleJun25T07, deepJun26T07. Every modelvsobscomparison SAMEtimestamp; endpointtablederiveindices, no usinglastfiniteobswithmodelfinal. Finalmodelinventoriesvalidbutnoimpliedobs. Rootperiodsplitusesnominalday label row.date beforeMar1 vs>=; dailyclose followsday, explicitlystate, no newcalendar/Tphysics.
Waterbudget rain/intake,E,bottomdrain,finalstorage,gravityinterfacesSEPARATE internal counts (maycountsamewater); don'tstackinternalswithexternalmasslosses. CommonEeffplotoptionalillustratesdegeneracy. Timingintegration/preparationJIT/freshcommand andchecksummary; allone1m2CPUcolumn, noGPU/domain-scale/fieldvalidation/nativecheckpoint/coupledclaim.
Tables<=7columns ideally5-6 withgrid+cellwidths,repeatingheaders,readable9pt. Accurate citations: SR2006,Tietjen2009,Hargreaves1985, NEONsoilwater/rain/air PLUSseparate2013megapitDP1.00096.001 spatialproxyfor2024plot004. ExplicitsoilweatherT, shadingalloff.
Keepfigures4-6readable; reportapproximately8-12pagesfine, avoidtoo muchtechnicalclutter; sourceidentity appendix/data. AllnumericCSVusedinWordsameformattedvalues; imagesactualPNGs embedded. NoOfficerenderclaim.
VerifyRoot218filesbeforeafter, independentrecompute metrics/budgets/periods/controls, archiveonlynewreportinputsneededno rawNEON. Includeallsource/filehashes inartifactmanifest butEXCLUDEignoredcachefiles; no manifestsrecursingruffcachetemp. No mutationofsealedoutputs. Saveauthorreport, leaveuncommittedandstopforRootindependentreview.

