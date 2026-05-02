S2 random-30 evaluation with best MTSLSTM 100-station model

Model run: idx2_bs128_do0.4_hs64_H168_D365
Model path: /home/kongw0a/hourly_streamflow_dl/MTSLSTM_100stations/training_runs/20260407_mtslstm_100stations_tuning_topo18_v100/idx2_bs128_do0.4_hs64_H168_D365/best_model.pth
Scaler path: /home/kongw0a/hourly_streamflow_dl/MTSLSTM_100stations/training_runs/20260407_mtslstm_100stations_tuning_topo18_v100/idx2_bs128_do0.4_hs64_H168_D365/scalers.pkl
Selection seed: 42
Selected stations: 30
S2 box inference: lon=[-95.0, -75.0], lat=[28.0, 36.8]

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse
train,30,22,8,0.003849,-0.038209,-1.875250,-16.653476
val,30,28,2,0.112148,-0.011605,-3.088982,-26.123397
test,30,28,2,0.092582,-0.113022,-2.904119,-51.005967
