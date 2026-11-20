# Root report scope / interpretation notes

- User's last instruction confirms FC ALL layers, and adds stress 1.5x0.2017. This is a crossed7FCx2stress14-case sensitivity. Each stressfamily should show all7 curves plus observedpoints (separatefigures toavoid14curveoverload).
- Exactstress0.2017 and0.30255; old accepteddepth-specificSR33reference has0.20170581117963332. Mark it historical reference, not a bit-identical newbaseline or identicalstress. AllK/Ks remainoriginalSR33 1x evenwhenFCoverridden: do not imply entireSRretentioncurvewasrecomputed.
- Sameobservedinitialwater in ALL14cases, noinitialrescaling withFC; storageinitial22.9424mm, noT, evaporationdaily, rain25.088mm, addedstormduration99000s unchanged.
- FC thresholds are9/10/11/12/13/14/15vol%, NOTwaterdepthmm. Each10cmlayer has9-15mmstorageatFC; totalprofile27-45mm. Saturation/remnant/porecapacityunchanged. This distinction may help explain firststorm/thirdlayer behavior.
- GravityaboveFC is immediatewithinsoilstep, not setbyK; TietjenK/dstillcontrolgradientexchange. ChangingFCaffectsretentionandwhichrainpulsescascade throughinterfaces. Evaporationmaychangebecauseavailabletopmoisturechanges; do not attributeallchangesonlytodrainage.
- InspectJanuary23deepwettingandJanuary25firstfollowingdeepobservationforALL14cases; whetherdeepFCreachedmustbecomputed notassumed. Needexplicitdeeppeakandtime, middle/gravitytransfer andbottom totals, same-timeobscomparison anduncertainty.
- Keepseasonal upper2diagnosticsbeforeMarch1vsfromMarch1; noTmodel vsassumeduseronsetTmeans pooledlate/fullRMSEnotsolephysicaloptimum. NeitherfieldTnorprobefaultmeasured/established. CalibrationE1d/stressoldFC/T0conditional.
- Rootindependent90digitcontrolsvalidonlyifallrainsupplylimited/noSATreturns; theseconditionswillbechecked, neverassumedtosuppressreturns. Finedtcontrolsrefinestormsonly, dailyEunchanged. NoGPU/scaling/coupled/nativecheckpointacceptance.
