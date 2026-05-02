BaselineLSTM daily aggregation tuning run `idx12_lr0.0005_bs128_lb90_hs128_do0.2_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 90
Batch size: 128
Hidden size: 128
Dropout: 0.2
Learning rate: 0.0005
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.724292,0.822633,0.700942,0.773215,0,0
val,30,28,2,0.560344,0.641458,-4.180583,-374.197560,7,7
test,30,28,2,0.628835,0.574387,-4.573006,-416.069031,7,7
