Hydrology-informed SymTorch experiment `hydro_rawstd_residual`

Target mode: rawstd
Event quantile gate: None
Train distill sample size: 10937
Total candidate features: 63
PySR select_k_features: 28

Best symbolic equation:
(ELEV_MEAN_M_BASIN * 0.07590193) * lstm_pred_raw

Equation complexity: 5
Equation loss: 0.04993170

Selected features:
- rain_sum_1
- rain_sum_3
- rain_sum_7
- rain_sum_14
- rain_sum_30
- pet_sum_3
- pet_sum_7
- wetness_7
- wetness_30
- api_7
- rain_pet_logratio_90
- storminess_logratio_7_90
- doy_sin
- lstm_pred_raw
- lstm_pred_log1p
- aridity_index
- p_seasonality
- frac_snow
- high_prec_freq
- ELEV_MEAN_M_BASIN
- SLOPE_PCT
- STRAHLER_MAX
- TOPWET
- CLAYAVE
- PERMAVE
- for_pc_use
- crp_pc_use
- urb_pc_use

Target fidelity:
- train: RMSE=0.231113, MAE=0.099433, R2=0.023502, corr=0.187084
- val: RMSE=0.378767, MAE=0.144376, R2=0.050879, corr=0.299339
- test: RMSE=0.373889, MAE=0.152085, R2=-0.002458, corr=0.069653

Hydrology metrics against observed daily streamflow:
- train: baseline dailyLSTM KGE=0.815785, NSE=0.842795; corrected KGE=0.808330, NSE=0.840066
- val: baseline dailyLSTM KGE=0.662899, NSE=0.663122; corrected KGE=0.556958, NSE=0.617340
- test: baseline dailyLSTM KGE=0.640016, NSE=0.577719; corrected KGE=0.559303, NSE=0.469310
