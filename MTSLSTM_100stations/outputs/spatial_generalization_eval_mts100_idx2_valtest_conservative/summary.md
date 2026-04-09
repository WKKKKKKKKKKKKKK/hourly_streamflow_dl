Spatial Generalization Evaluation on Validation and Test Periods Using Best 100 Station Tuned MTSLSTM

Selected by validation median KGE from mts100_tune.
Best run: idx2_bs128_do0.4_hs64_H168_D365
Tuned validation median KGE: 0.782548238948517
Tuned validation median NSE: 0.717950727401788
Tuned test median KGE: 0.721158881476193
Tuned test median NSE: 0.693819337627870
Model path: /ibex/project/c2266/wkkong/data/CAEMLSH/data_workstation/CAMELSH/training_runs/20260407_mtslstm_100stations_tuning_topo18_v100/idx2_bs128_do0.4_hs64_H168_D365/best_model.pth
Scaler path: /ibex/project/c2266/wkkong/data/CAEMLSH/data_workstation/CAMELSH/training_runs/20260407_mtslstm_100stations_tuning_topo18_v100/idx2_bs128_do0.4_hs64_H168_D365/scalers.pkl
Samples csv: /home/kongw0a/MTS_LSTM/experiment_withcursor/station_maps/outputs/spatial_generalization_station_samples_conservative.csv
Static csv: /home/kongw0a/MTS_LSTM/experiment_withcursor/data/static_h_topo_priority27.csv
Validation period: 2003-10-01 to 2008-09-30
Test period: 2008-10-01 to 2015-09-30
Evaluation path: MultiscaleLSTMDataset plus Train.evaluate_per_station

Regional medians

scheme_code,scheme_label,n_total_stations,n_valid_stations,n_excluded_stations,median_val_kge,median_val_nse,median_test_kge,median_test_nse
S1,S1 Csb-CA,8,8,0,-107.612869,-950.682961,-207.706429,-2189.951437
S2,S2 Cfa-SE,17,17,0,-0.167581,-1.156828,0.019637,-1.006107
U1,U1 Dfa-MW,10,10,0,-3.457255,-25.513912,-1.834728,-5.761150
U2,U2 Dfb(E)-GL,10,10,0,0.166099,-0.521995,0.143104,-0.623878
U3,U3 Dfb(W)-RK,6,6,0,-25.032609,-952.541222,-16.773674,-107.939258
U4,U4 BSk-SW,3,3,0,-11032.106127,-111776.527949,-421.137508,-2927.499082
U5,U5 BSk-NP,3,3,0,-52.622275,-484.582331,-45.732652,-321.857649
