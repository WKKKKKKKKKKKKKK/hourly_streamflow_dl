BaselineLSTM daily aggregation tuning run `idx16_lr0.0001_bs256_lb180_hs512_do0.6_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 180
Batch size: 256
Hidden size: 512
Dropout: 0.6
Learning rate: 0.0001
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.728374,0.812570,0.698683,0.744730,1,0
val,30,28,2,0.590262,0.618483,-1.697505,-87.778164,5,7
test,30,28,2,0.540373,0.539812,-3.190907,-314.471651,7,8
