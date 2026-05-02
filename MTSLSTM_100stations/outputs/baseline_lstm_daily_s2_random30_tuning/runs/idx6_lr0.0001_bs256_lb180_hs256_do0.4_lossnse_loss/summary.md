BaselineLSTM daily aggregation tuning run `idx6_lr0.0001_bs256_lb180_hs256_do0.4_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 180
Batch size: 256
Hidden size: 256
Dropout: 0.4
Learning rate: 0.0001
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.721542,0.714199,0.673577,0.678179,1,0
val,30,28,2,0.537218,0.630831,-1.023179,-33.569757,7,6
test,30,28,2,0.579741,0.535152,-2.074904,-135.060943,6,8
