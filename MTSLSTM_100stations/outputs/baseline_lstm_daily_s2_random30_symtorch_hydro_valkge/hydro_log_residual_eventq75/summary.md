Hydrology-informed SymTorch experiment `hydro_log_residual_eventq75`

Target mode: log
Event quantile gate: 0.75
Train distill sample size: 9964
Total candidate features: 63
PySR select_k_features: 28

Best symbolic equation:
-0.070106834 / cos(square(PERMAVE))

Equation complexity: 8
Equation loss: 0.17045460

Selected features:
- rain_sum_1
- rain_sum_7
- rain_sum_30
- rain_sum_60
- pet_sum_60
- pet_sum_90
- tair_mean_30
- tair_mean_60
- tair_mean_90
- wetness_30
- wetness_90
- api_7
- rain_pet_logratio_30
- rain_pet_logratio_90
- storminess_logratio_3_30
- storminess_logratio_7_90
- tair_diff_30_90
- lstm_pred_raw
- lstm_pred_log1p
- pet_mean
- aridity_index
- p_seasonality
- frac_snow
- low_prec_dur
- SLOPE_PCT
- CLAYAVE
- PERMAVE
- BDAVE

Target fidelity:
- train: RMSE=0.410044, MAE=0.287152, R2=-0.017496, corr=0.077007
- val: RMSE=0.603874, MAE=0.403580, R2=-0.069992, corr=0.072301
- test: RMSE=0.781867, MAE=0.516768, R2=0.003140, corr=0.089027

Hydrology metrics against observed daily streamflow:
- train: baseline dailyLSTM KGE=0.815785, NSE=0.842795; corrected KGE=0.776582, NSE=0.840731
- val: baseline dailyLSTM KGE=0.662899, NSE=0.663122; corrected KGE=0.643246, NSE=0.663564
- test: baseline dailyLSTM KGE=0.640016, NSE=0.577719; corrected KGE=0.659897, NSE=0.592222
