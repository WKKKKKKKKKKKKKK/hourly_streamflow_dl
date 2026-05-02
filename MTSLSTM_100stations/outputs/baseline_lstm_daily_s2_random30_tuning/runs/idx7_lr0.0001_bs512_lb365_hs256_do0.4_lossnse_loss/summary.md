BaselineLSTM daily aggregation tuning run `idx7_lr0.0001_bs512_lb365_hs256_do0.4_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 365
Batch size: 512
Hidden size: 256
Dropout: 0.4
Learning rate: 0.0001
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.581821,0.653701,0.577191,0.620979,1,0
val,30,28,2,0.547931,0.596396,-2.929730,-147.378002,6,6
test,30,28,2,0.532670,0.520857,-2.757224,-96.868653,6,7
