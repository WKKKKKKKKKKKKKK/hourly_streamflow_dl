Daily LSTM teacher for SymTorch prior extraction on S2 random 30 stations

Device: cuda
Seed: 42
Lookback days: 60
Hidden size: 64
Dropout: 0.2
Learning rate: 0.0005
Weight decay: 1e-05
Best epoch by val median KGE: 23

Dynamic Z features:
- rain_1
- rain_3
- rain_7
- rain_30
- pet_1
- pet_7
- pet_30
- tair_1
- tair_7
- tair_30
- wetness_7
- wetness_30
- rain_pet_ratio_30
- doy_sin
- doy_cos

Static features:
- aridity_index
- BFI_AVE
- DRAIN_SQKM
- SLOPE_PCT
- CLAYAVE
- SANDAVE
- for_pc_use
- urb_pc_use

Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse
train,22,22,0,0.600498,0.605588,0.553904,0.545602
val,28,28,0,0.465512,0.478135,-0.514201,-15.205774
test,28,28,0,0.394860,0.374285,-0.929849,-33.937948
