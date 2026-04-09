# Spatial Generalization Evaluation Using Best Tuned MTSLSTM

- best run: `2augfge8_hd64-64_bs128_H72_D365_do0.4_lossnse_loss_reg1.0_sch1-5e-4-10-1e-4-25-5e-5`
- model path: `/home/kongw0a/MTS_LSTM/experiment_withcursor/MTSLSTM/runs/20260314/MTSLSTM/2augfge8_hd64-64_bs128_H72_D365_do0.4_lossnse_loss_reg1.0_sch1-5e-4-10-1e-4-25-5e-5/best_model.pth`
- validation period: `2003-10-01` to `2008-09-30`
- test period: `2008-10-01` to `2015-09-30`

## Regional medians

| scheme_code | scheme_label | n_stations | median_val_kge | median_val_nse | median_test_kge | median_test_nse |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | S1 Csb-CA | 8 | -297.889422 | -15778.469648 | -456.588224 | -54231.379631 |
| S2 | S2 Cfa-SE | 17 | -1.074481 | -2.243925 | -0.873409 | -1.003757 |
| U1 | U1 Dfa-MW | 10 | -2.759527 | -9.579906 | -0.566493 | -1.164195 |
| U2 | U2 Dfb(E)-GL | 10 | -0.517029 | -0.518290 | -0.700912 | -0.666368 |
| U3 | U3 Dfb(W)-RK | 6 | -7.268991 | -59.375783 | -10.244819 | -48.785526 |
| U4 | U4 BSk-SW | 3 | -10711.379834 | -11582532.642951 | -648.971577 | -335.703000 |
| U5 | U5 BSk-NP | 3 | -128.198936 | -2528.655451 | -48.518268 | -855.428086 |
