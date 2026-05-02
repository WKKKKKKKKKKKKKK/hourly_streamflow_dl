Hydrology-informed SymTorch experiment `hydro_log_residual`

Target mode: log
Event quantile gate: None
Train distill sample size: 10937
Total candidate features: 63
PySR select_k_features: 28

Best symbolic equation:
(SLOPE_PCT + (rain_pet_logratio_90 - p_mean)) * ((0.5582391 - for_pc_use) * 0.06458572)

Equation complexity: 11
Equation loss: 0.16795258

Selected features:
- rain_sum_14
- rain_sum_30
- rain_sum_60
- rain_sum_90
- pet_sum_30
- pet_sum_60
- pet_sum_90
- tair_mean_90
- wetness_90
- api_30
- rain_pet_logratio_90
- lstm_pred_raw
- lstm_pred_log1p
- p_mean
- pet_mean
- aridity_index
- p_seasonality
- high_prec_freq
- high_prec_dur
- low_prec_freq
- low_prec_dur
- ELEV_MEAN_M_BASIN
- SLOPE_PCT
- BAS_COMPACTNESS
- TOPWET
- PERMAVE
- for_pc_use
- crp_pc_use

Target fidelity:
- train: RMSE=0.391421, MAE=0.271771, R2=0.072827, corr=0.269924
- val: RMSE=0.586787, MAE=0.383953, R2=-0.010298, corr=0.141882
- test: RMSE=0.773193, MAE=0.503161, R2=0.025135, corr=0.160605

Hydrology metrics against observed daily streamflow:
- train: baseline dailyLSTM KGE=0.815785, NSE=0.842795; corrected KGE=0.797274, NSE=0.841342
- val: baseline dailyLSTM KGE=0.662899, NSE=0.663122; corrected KGE=0.675974, NSE=0.704051
- test: baseline dailyLSTM KGE=0.640016, NSE=0.577719; corrected KGE=0.628849, NSE=0.575530
