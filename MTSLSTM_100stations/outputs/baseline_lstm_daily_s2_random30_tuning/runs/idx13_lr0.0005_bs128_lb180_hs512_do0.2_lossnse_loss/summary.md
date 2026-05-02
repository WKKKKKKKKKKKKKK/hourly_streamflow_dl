BaselineLSTM daily aggregation tuning run `idx13_lr0.0005_bs128_lb180_hs512_do0.2_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 180
Batch size: 128
Hidden size: 512
Dropout: 0.2
Learning rate: 0.0005
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.830972,0.833712,0.777886,0.776629,0,0
val,30,28,2,0.609839,0.663951,-3.210332,-259.459407,6,6
test,30,28,2,0.661637,0.613594,-3.748901,-321.878282,7,7
