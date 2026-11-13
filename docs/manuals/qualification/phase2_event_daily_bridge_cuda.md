# CPU and CUDA event/daily bridge

The native Python bridge is independently accepted at the bounded scope recorded in [its source-specific receipt](phase2_event_daily_bridge_cuda_acceptance.json): 149 host/CPU checks and 21 actual GPU checks on the composed 167-module source, with earlier component evidence kept separate. It is part of the still-unpublished Phase 2 source, not a new command-line mode or acceptance of the entire phase.

A genuinely completed native water event can hand its current layered water, wetting history, surface water and momentum to daily hydrology. Dry days use one soil update. Wet days use the user-approved short hydraulic and evaporation steps totaling exactly 86400 seconds; transpiration is requested once at day end, then closure is audited. Evaporation and transpiration remain off during rainfall and recession. A time or resource cap does not establish event completion.

Use floating_event_daily_bridge.open_day, day_action, begin_storm_from_day, day_after_storm and hand_back with actual lifecycle objects. The rainfall plan retains Phase 1.5 selection and whole-occurrence fallback. The actual accepted storm forcing history must conform; another history with the same total is insufficient. Current moisture and Green–Ampt history persist. Properties refresh at completed transport boundaries, independently of net-only hydraulic rerouting.

Optional every-N-day moisture output follows completed daily withdrawals and audited closure. Seven-day tests with actual intervening native events and changed properties saved days 3 and 6 for N=3. These frames are diagnostics and do not themselves restore the complete model.

CUDA keeps water and history on the device. Device and host resource pools, reserves and gates are separate. Wet-day setup avoids sediment-phase allocation and whole-voxel inventory scans; the zero rain row is reused. Opening of a 4096-cell, three-layer fixture peaked at 230400 device bytes against its 491872-byte model, with stable released pools over three trials. This is an opening measurement, not a whole wet-day or large-catchment memory guarantee; cold static input hashing can transfer non-water arrays.

The source also retains the separately qualified Dense and native Windowed continued-event checkpoint. Closed-day/accepted-boundary physical lifecycle files remain a separate task. This bridge does not implement the temporal coordinator, wind adapter, plant model or field validation. Historical pins and checkpoint profiles remain strict and are never rebound.
