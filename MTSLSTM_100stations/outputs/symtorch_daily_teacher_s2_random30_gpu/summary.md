Daily LSTM teacher for SymTorch prior extraction on S2 random 30 stations

Metric note: KGE/NSE below are computed on inverse-standardized raw daily streamflow.

Seed: 42
Lookback days: 60
Hidden size: 64
Dropout: 0.2
Learning rate: 0.0005
Weight decay: 1e-05
Best epoch by saved val_median_kge history: 11

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse
train,22,22,0,0.393343,0.399426,0.218154,0.259642
val,28,28,0,0.114167,0.207679,-1.861620,-59.383146
test,28,28,0,0.240340,0.237860,-1.556107,-24.587976
