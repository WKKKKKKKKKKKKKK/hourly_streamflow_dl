BaselineLSTM daily aggregation tuning run `idx5_lr0.0001_bs256_lb365_hs256_do0.4_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 365
Batch size: 256
Hidden size: 256
Dropout: 0.4
Learning rate: 0.0001
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.730971,0.723702,0.683638,0.701528,0,0
val,30,28,2,0.622520,0.622223,-2.377896,-128.938553,6,6
test,30,28,2,0.541674,0.554760,-2.575810,-173.986012,7,8
