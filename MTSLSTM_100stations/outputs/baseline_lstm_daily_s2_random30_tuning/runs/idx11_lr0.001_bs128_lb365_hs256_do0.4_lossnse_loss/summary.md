BaselineLSTM daily aggregation tuning run `idx11_lr0.001_bs128_lb365_hs256_do0.4_lossnse_loss` on S2 random 30 stations

Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.

Lookback days: 365
Batch size: 128
Hidden size: 256
Dropout: 0.4
Learning rate: 0.001
Epoch cap: 55
Early stopping patience: 10

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations
train,30,22,8,0.715081,0.821496,0.692301,0.756858,1,0
val,30,28,2,0.598948,0.618037,-1.438946,-61.080697,6,5
test,30,28,2,0.610734,0.610316,-1.790413,-78.748714,7,7
