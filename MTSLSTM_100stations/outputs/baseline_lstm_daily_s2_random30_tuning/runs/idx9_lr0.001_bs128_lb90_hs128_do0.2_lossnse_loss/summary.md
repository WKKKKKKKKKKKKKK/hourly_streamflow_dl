BaselineLSTM daily aggregation tuning run `idx9_lr0.001_bs128_lb90_hs128_do0.2_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 90
Batch size: 128
Hidden size: 128
Dropout: 0.2
Learning rate: 0.001
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.815785,0.842795,0.758556,0.789031,0,0
val,30,28,2,0.662898,0.663122,-4.085852,-457.614961,6,7
test,30,28,2,0.640016,0.577719,-3.693570,-338.176683,7,8
