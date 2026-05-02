BaselineLSTM daily aggregation tuning run `idx15_lr0.0001_bs256_lb90_hs512_do0.2_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 90
Batch size: 256
Hidden size: 512
Dropout: 0.2
Learning rate: 0.0001
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.774377,0.832072,0.747406,0.782132,0,0
val,30,28,2,0.582138,0.671829,-5.511859,-877.820053,7,8
test,30,28,2,0.540982,0.498782,-5.049025,-750.296562,7,7
