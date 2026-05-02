BaselineLSTM daily aggregation tuning run `idx14_lr0.0005_bs128_lb365_hs512_do0.6_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 365
Batch size: 128
Hidden size: 512
Dropout: 0.6
Learning rate: 0.0005
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.701764,0.827869,0.657373,0.746241,2,0
val,30,28,2,0.579614,0.588423,-2.140412,-85.017176,6,6
test,30,28,2,0.543847,0.564297,-3.225963,-212.396592,8,7
