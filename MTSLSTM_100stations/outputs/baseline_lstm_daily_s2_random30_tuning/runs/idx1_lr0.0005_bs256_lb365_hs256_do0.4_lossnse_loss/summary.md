BaselineLSTM daily aggregation tuning run `idx1_lr0.0005_bs256_lb365_hs256_do0.4_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 365
Batch size: 256
Hidden size: 256
Dropout: 0.4
Learning rate: 0.0005
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.833040,0.874757,0.780491,0.811364,0,0
val,30,28,2,0.590467,0.647569,-3.521798,-339.826183,5,5
test,30,28,2,0.649839,0.666872,-3.366853,-294.990198,7,7
