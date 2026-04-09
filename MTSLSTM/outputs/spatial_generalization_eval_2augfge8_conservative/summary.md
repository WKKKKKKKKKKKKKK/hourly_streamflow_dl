# Spatial Generalization Evaluation Using Best Tuned MTSLSTM

- best run: `2augfge8_hd64-64_bs128_H72_D365_do0.4_lossnse_loss_reg1.0_sch1-5e-4-10-1e-4-25-5e-5`
- model path: `/home/kongw0a/MTS_LSTM/experiment_withcursor/MTSLSTM/runs/20260314/MTSLSTM/2augfge8_hd64-64_bs128_H72_D365_do0.4_lossnse_loss_reg1.0_sch1-5e-4-10-1e-4-25-5e-5/best_model.pth`
- samples csv: `/home/kongw0a/MTS_LSTM/experiment_withcursor/station_maps/outputs/spatial_generalization_station_samples_conservative.csv`
- validation period: `2003-10-01` to `2008-09-30`
- test period: `2008-10-01` to `2015-09-30`
- evaluation path: reuse `MultiscaleLSTMDataset` + `Train.evaluate_per_station`

## Regional medians

| scheme_code | scheme_label | n_total_stations | n_valid_stations | n_excluded_stations | median_val_kge | median_val_nse | median_test_kge | median_test_nse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | S1 Csb-CA | 8 | 8 | 0 | -58.449408 | -2285.021786 | -102.257268 | -4758.877067 |
| S2 | S2 Cfa-SE | 17 | 17 | 0 | -0.572881 | -0.697804 | -0.599533 | -0.623219 |
| U1 | U1 Dfa-MW | 10 | 10 | 0 | -1.805751 | -5.981371 | -1.214555 | -1.499396 |
| U2 | U2 Dfb(E)-GL | 10 | 10 | 0 | -1.516913 | -1.483555 | -0.936484 | -0.612546 |
| U3 | U3 Dfb(W)-RK | 6 | 6 | 0 | -13.902955 | -235.962930 | -13.542980 | -136.408198 |
| U4 | U4 BSk-SW | 3 | 3 | 0 | -1645.949527 | -76348.713522 | -77.925934 | -335.703000 |
| U5 | U5 BSk-NP | 3 | 3 | 0 | -44.190386 | -1902.579641 | -7.372296 | -10.617989 |
