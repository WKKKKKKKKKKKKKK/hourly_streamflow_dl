BaselineLSTM daily aggregation tuning run `idx2_lr0.0005_bs256_lb180_hs256_do0.4_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 180
Batch size: 256
Hidden size: 256
Dropout: 0.4
Learning rate: 0.0005
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.865715,0.882291,0.806707,0.812591,0,0
val,30,28,2,0.659222,0.648469,-2.999796,-268.943516,5,5
test,30,28,2,0.672703,0.630996,-4.366754,-530.712806,6,8
