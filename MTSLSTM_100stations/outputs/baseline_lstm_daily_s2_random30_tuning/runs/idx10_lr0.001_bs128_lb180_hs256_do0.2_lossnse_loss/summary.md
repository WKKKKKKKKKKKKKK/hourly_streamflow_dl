BaselineLSTM daily aggregation tuning run `idx10_lr0.001_bs128_lb180_hs256_do0.2_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 180
Batch size: 128
Hidden size: 256
Dropout: 0.2
Learning rate: 0.001
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.866033,0.892525,0.825816,0.833405,0,0
val,30,28,2,0.651911,0.664135,-1.819143,-71.307832,5,6
test,30,28,2,0.696133,0.641921,-2.156950,-87.344809,7,7
