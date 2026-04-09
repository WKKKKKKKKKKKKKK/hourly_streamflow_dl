# Spatial Generalization Evaluation on Training Period Using Best 100-Station Tuned MTSLSTM

- selected by validation median KGE from `mts100_tune`.
- best run: `idx2_bs128_do0.4_hs64_H168_D365`
- tuned val median KGE: `0.782548238948517`
- tuned test median KGE: `0.721158881476193`
- model path: `/ibex/project/c2266/wkkong/data/CAEMLSH/data_workstation/CAMELSH/training_runs/20260407_mtslstm_100stations_tuning_topo18_v100/idx2_bs128_do0.4_hs64_H168_D365/best_model.pth`
- scaler path: `/ibex/project/c2266/wkkong/data/CAEMLSH/data_workstation/CAMELSH/training_runs/20260407_mtslstm_100stations_tuning_topo18_v100/idx2_bs128_do0.4_hs64_H168_D365/scalers.pkl`
- samples csv: `/home/kongw0a/MTS_LSTM/experiment_withcursor/station_maps/outputs/spatial_generalization_station_samples_conservative.csv`
- static csv: `/home/kongw0a/MTS_LSTM/experiment_withcursor/data/static_h_topo_priority27.csv`
- training period: `1990-10-01` to `2003-09-30`
- evaluation path: reuse `MultiscaleLSTMDataset` + `Train.evaluate_per_station`

## Regional medians

| scheme_code | scheme_label | n_total_stations | n_valid_stations | n_excluded_stations | median_train_kge | median_train_nse |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | S1 Csb-CA | 8 | 6 | 2 | -124.199621 | -4179.203332 |
| S2 | S2 Cfa-SE | 17 | 12 | 5 | -0.236703 | -0.104209 |
| U1 | U1 Dfa-MW | 10 | 9 | 1 | -0.908246 | -3.604577 |
| U2 | U2 Dfb(E)-GL | 10 | 7 | 3 | 0.310208 | -0.200658 |
| U3 | U3 Dfb(W)-RK | 6 | 6 | 0 | -22.204016 | -776.969784 |
| U4 | U4 BSk-SW | 3 | 3 | 0 | -583.495922 | -3325.278397 |
| U5 | U5 BSk-NP | 3 | 3 | 0 | -34.223278 | -626.119758 |
