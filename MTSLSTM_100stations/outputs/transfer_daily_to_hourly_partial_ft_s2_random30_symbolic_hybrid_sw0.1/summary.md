Daily-branch transfer learning affecting hourly outputs with hybrid symbolic prior

Source pretrained model: /home/kongw0a/hourly_streamflow_dl/MTSLSTM_100stations/training_runs/20260407_mtslstm_100stations_tuning_topo18_v100/idx2_bs128_do0.4_hs64_H168_D365/best_model.pth
Best epoch by hourly val KGE: 6
Learning rate: 0.0005
Weight decay: 1e-05
Aggregate-hourly daily loss weight: 0.5
Symbolic prior loss weight: 0.1

Trainable modules: lstm_daily, transfer_h, transfer_c, head_daily, head_hourly
Objective: target daily-branch loss + target aggregated-hourly daily loss + symbolic hybrid prior loss
Final metrics below are HOURLY KGE/NSE.

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse
train,30,22,8,0.741238,0.675806
val,30,28,2,0.527833,0.256749
test,30,28,2,0.498042,0.239969
