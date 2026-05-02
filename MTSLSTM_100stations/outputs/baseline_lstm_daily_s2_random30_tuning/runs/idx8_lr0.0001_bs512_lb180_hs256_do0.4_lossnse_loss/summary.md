BaselineLSTM daily aggregation tuning run `idx8_lr0.0001_bs512_lb180_hs256_do0.4_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 180
Batch size: 512
Hidden size: 256
Dropout: 0.4
Learning rate: 0.0001
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.738039,0.722795,0.715898,0.690766,1,0
val,30,28,2,0.544860,0.633245,-2.837353,-161.666487,6,7
test,30,28,2,0.540158,0.474971,-3.207969,-186.934390,7,8
