BaselineLSTM daily aggregation tuning run `idx3_lr0.0005_bs512_lb365_hs256_do0.4_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 365
Batch size: 512
Hidden size: 256
Dropout: 0.4
Learning rate: 0.0005
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.834143,0.853863,0.796591,0.799754,0,0
val,30,28,2,0.637511,0.655626,-1.705670,-89.149411,5,5
test,30,28,2,0.672530,0.581436,-4.429322,-536.636604,7,8
