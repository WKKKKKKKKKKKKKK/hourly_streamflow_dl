BaselineLSTM daily aggregation tuning run `idx4_lr0.0005_bs512_lb180_hs256_do0.4_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 180
Batch size: 512
Hidden size: 256
Dropout: 0.4
Learning rate: 0.0005
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.828077,0.860085,0.777638,0.822755,0,0
val,30,28,2,0.632653,0.704676,-2.167356,-132.932312,5,6
test,30,28,2,0.562697,0.530864,-3.489691,-314.459465,7,8
