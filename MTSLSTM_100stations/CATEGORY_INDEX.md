# Category Index

This file is a lightweight classification index for `MTSLSTM_100stations`.
It does not change any existing file paths.
Use it as a quick map for locating code and outputs by experiment family.

## 1. Daily LSTM on the selected 30 stations

Code:
- `code/tune_baseline_lstm_daily_s2_random30.py`

Supporting inputs and station subset:
- `outputs/s2_random30_idx2_eval/selected_stations.csv`
- `outputs/s2_random30_idx2_eval/summary.md`

Main outputs:
- `outputs/baseline_lstm_daily_s2_random30_tuning/`

Best-run artifacts:
- `outputs/baseline_lstm_daily_s2_random30_tuning/best_run_by_val_kge.json`
- `outputs/baseline_lstm_daily_s2_random30_tuning/runs/idx9_lr0.001_bs128_lb90_hs128_do0.2_lossnse_loss/`

## 2. SymTorch distillation and symbolic residual experiments

Code:
- `code/export_best_daily_lstm_for_symtorch.py`
- `code/distill_best_daily_lstm_with_symtorch.py`
- `code/distill_daily_lstm_obs_residual_with_symtorch.py`
- `code/distill_daily_lstm_hydro_residual_with_symtorch.py`
- `code/fuse_hydro_log_symbolic_residuals.py`

Notebook reference/example [On the application of LSTM in meteorology (e.g., MJO prediction) using SymTorch, but unrelated to the main experiment.]:
- `Sym_lstm/MJO_prediction.ipynb`

Direct export and direct distillation outputs:
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/symtorch_direct_distill/`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/symtorch_obs_minus_dailylstm_residual/`

Hydrology-informed symbolic outputs:
- `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/hydro_rawstd_residual/`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/hydro_log_residual/`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/hydro_log_residual_eventq75/`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/hybrid_log_residual_smoothgate/`

## 3. Transfer learning from daily to hourly

Code:
- `code/transfer_daily_to_hourly_partial_ft_s2_random30.py`

Outputs:
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30/`

Comparison plots and summaries:
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30/baseline_vs_transfer_plots/`

## 4. Transfer learning with symbolic prior

Code:
- `code/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid.py`

Outputs:
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.05/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.1/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.2/`

## 5. Daily-head-only transfer ablation

Code:
- `code/transfer_daily_head_tune_s2_random30.py`
- `code/transfer_daily_head_tune_symbolic_s2_random30.py`

Outputs:
- `outputs/head_only_daily_head_fc_transfer_s2_random30/`
- `outputs/head_only_daily_head_fc_transfer_s2_random30/ordinary_transfer/`
- `outputs/head_only_daily_head_fc_transfer_s2_random30/symbolic_prior_sw0.05/`

Included model files:
- `outputs/head_only_daily_head_fc_transfer_s2_random30/ordinary_transfer/best_transfer_model.pth`
- `outputs/head_only_daily_head_fc_transfer_s2_random30/symbolic_prior_sw0.05/best_transfer_model.pth`
- `outputs/head_only_daily_head_fc_transfer_s2_random30/ordinary_transfer/trials/*.pth`
- `outputs/head_only_daily_head_fc_transfer_s2_random30/symbolic_prior_sw0.05/trials/*.pth`

Comparison summary:
- `outputs/head_only_daily_head_fc_transfer_s2_random30/README.md`
- `outputs/head_only_daily_head_fc_transfer_s2_random30/comparison_summary.csv`
- reference original daily-branch transfer result: `outputs/transfer_daily_to_hourly_partial_ft_s2_random30/summary.md`
- reference symbolic-prior daily-branch transfer result: `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.05/summary.md`

## 6. Evaluation and presentation plots

Code:
- `code/evaluate_s2_random30_idx2.py`
- `code/evaluate_s2_random30_alt_threeway.py`
- `code/evaluate_s1_csb_ca_threeway.py`
- `code/evaluate_u2_dfb_e_gl_threeway.py`
- `code/plot_s2_random30_baseline_vs_transfer.py`
- `code/plot_s2_random30_threeway_ppt.py`
- `code/plot_three_method_peak_lag_cdfs.py`

Outputs:
- `outputs/s2_random30_idx2_eval/`
- `outputs/s2_random30_threeway_ppt_plots/`
- `outputs/s2_random30_alt_threeway_ppt_plots/`
- `outputs/s1_csb_ca_threeway_ppt_plots/`
- `outputs/u2_dfb_e_gl_threeway_ppt_plots/`

## 7. Significance and source-domain degradation analyses

Code:
- `code/run_s2_threeway_significance_tests.py`
- `code/evaluate_transfer_on_source_domain.py`
- `code/analyze_source_domain_transfer_degradation.py`
- `code/plot_source_domain_kge_vs_mean_flow.py`

Station-level paired significance outputs:
- `outputs/s2_random30_threeway_ppt_plots/significance_tests_vs_baseline.csv`
- `outputs/s2_random30_threeway_ppt_plots/symbolic_vs_transfer_significance_tests.csv`
- `outputs/s2_random30_threeway_ppt_plots/s2_threeway_all_paired_significance_tests.csv`

Source-domain degradation outputs:
- `outputs/source_domain_transfer_retention_eval/`
- `outputs/source_domain_transfer_retention_eval/per_station_source_domain_metrics.csv`
- `outputs/source_domain_transfer_retention_eval/per_station_source_domain_metrics_with_flow.csv`
- `outputs/source_domain_transfer_retention_eval/lowflow_failure_by_quartile.csv`
- `outputs/source_domain_transfer_retention_eval/retention_failure_counts.csv`
- `outputs/source_domain_transfer_retention_eval/kge_vs_flow_test_binned_summary.csv`
- `outputs/source_domain_transfer_retention_eval/nse_vs_flow_test_binned_summary.csv`
- `outputs/source_domain_transfer_retention_eval/figures/source_domain_test_kge_vs_mean_flow_three_models.png`
- `outputs/source_domain_transfer_retention_eval/figures/source_domain_test_nse_vs_mean_flow_three_models.png`

## 8. Original 100-station archive

Core code and metadata:
- `code/Modelzoo.py`
- `code/Train.py`
- `code/config.py`
- `code/inference.py`
- `code/loder.py`
- `code/losses.py`
- `code/trainer.py`
- `metadata/`
- `tuning/`
- `training_runs/`
- `logs/`
- `summaries/`

Spatial generalization outputs:
- `outputs/spatial_generalization_eval_mts100_idx2_trainperiod_conservative/`
- `outputs/spatial_generalization_eval_mts100_idx2_valtest_conservative/`

## 9. Fast lookup

If you want the shortest path:

- Daily LSTM:
  `code/tune_baseline_lstm_daily_s2_random30.py`
  `outputs/baseline_lstm_daily_s2_random30_tuning/`

- SymTorch:
  `code/export_best_daily_lstm_for_symtorch.py`
  `code/distill_best_daily_lstm_with_symtorch.py`
  `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/`
  `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/`

- Transfer learning:
  `code/transfer_daily_to_hourly_partial_ft_s2_random30.py`
  `outputs/transfer_daily_to_hourly_partial_ft_s2_random30/`

- Transfer learning + symbolic prior:
  `code/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid.py`
  `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.05/`
  `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.1/`
  `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.2/`

- Daily-head-only transfer ablation:
  `code/transfer_daily_head_tune_s2_random30.py`
  `code/transfer_daily_head_tune_symbolic_s2_random30.py`
  `outputs/head_only_daily_head_fc_transfer_s2_random30/`

- Significance tests:
  `code/run_s2_threeway_significance_tests.py`
  `outputs/s2_random30_threeway_ppt_plots/s2_threeway_all_paired_significance_tests.csv`

- Source-domain degradation:
  `code/evaluate_transfer_on_source_domain.py`
  `code/analyze_source_domain_transfer_degradation.py`
  `code/plot_source_domain_kge_vs_mean_flow.py`
  `outputs/source_domain_transfer_retention_eval/`
